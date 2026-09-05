# [VPTO backend] PyPTO DeepSeek-V4 Pro 算子 lowering 失败汇总（6 类根因 + 最小复现）

## 背景

在 PyPTO 的 `kernel_backend=vpto`（`--vpto-emit-merged-device-only` merged-device 路线）下编译 pypto-lib 的 DeepSeek-V4 Pro 全部 27 个单卡算子测例（`models/deepseek_v4_pro/*.py`，platform=a5），**只有 hc_post 端到端跑通**；其余 26 个全部在 PTOAS 编译期失败。EmitC 后端（同输入）大部分通过，因此失败点是 **VPTO  lowering 链路特有**的。

环境：PTOAS `main-llvm19-build`（c57bda69d）+ merged-device 移植；CANN 9.1.0；arch=a5。
统一复现命令（PTO 级复现）：

```bash
ptoas <case>.pto --pto-arch a5 --pto-backend vpto --enable-insert-sync \
  --pto-level=level3 --vpto-emit-merged-device-only -o /tmp/out.o
```

PyPTO 级复现（生成 PTO + 走完整 vpto 编译）：`PYPTO_KERNEL_BACKEND=vpto python repro_<case>.py`（`compile_only=True`，无需设备）。

按失败首因归类（一个 case 可命中多类）：

| # | 根因 | 受影响测例数 |
|---|---|---|
| 1 | `pto.trsqrt` 3-operand 形式无模板 | 11 |
| 2 | `pto.tgather` compare-form 无模板 | 20 |
| 3 | `pto.tmax` / `pto.tcolexpand` 模板 custom constraints 拒绝 | 2 |
| 4 | `pto.tsetval` 无 VPTO lowering → `unrealized_conversion_cast` | 11 |
| 5 | `pto.initialize_l2l_pipe` peer-init-pair 校验失败 | 4 |
| 6 | merged-device 一 ELF 一 entry 限制 vs AIC+AIV 混合 group | 13 |

---

## 1. `pto.trsqrt`：high-precision 3-operand 形式无模板（11 case）

**报错：**
```
NoMatchingTemplate: no legal template for op='pto.trsqrt' target='a5';
  template_trsqrt: operand binding failed: template 'template_trsqrt' expects 2 operands, got 3;
  template_trsqrt_1d: operand binding failed: template 'template_trsqrt_1d' expects 2 operands, got 3;
  vmi_trsqrt: operand binding failed: template 'vmi_trsqrt' expects 2 operands, got 3;
  vmi_trsqrt_with_tmp: custom constraints are not satisfied
```

**根因：** PyPTO 对 `pl.rsqrt(..., high_precision=True)` 生成 `pto.trsqrt ins(src, tmp) outs(dst)`（3 操作数，tmp 为迭代 refine 的中间 buffer）。`lib/TileOps/a5/trsqrt.py` 的 `template_trsqrt`/`template_trsqrt_1d`/`vmi_trsqrt` 都是 `register_unary`（2 操作数）；唯一 3-operand 的 `vmi_trsqrt_with_tmp` 的 context constraints 与 PyPTO 生成的调用上下文不匹配。

**最小复现（PyPTO 级）`repro_trsqrt.py`：**
```python
@pl.jit
def rsqrt_hp(x: pl.Tensor[[8, 64], pl.FP32], y: pl.Tensor[[8, 64], pl.FP32]):
    for blk in pl.spmd(1, name_hint="rsqrt_hp"):
        sq = pl.full([1, 8], dtype=pl.FP32, value=0.0)
        for kb in pl.pipeline(2, stage=2):
            chunk = x[0:8, kb*32:(kb+1)*32]
            sq = pl.add(sq, pl.reshape(pl.row_sum(pl.mul(chunk, chunk)), [1, 8]))
        inv = pl.rsqrt(pl.add(pl.mul(sq, 1.0/64), 1e-6), high_precision=True)  # ← 触发
        y[0:8, 0:64] = pl.mul(x[0:8, 0:64], pl.reshape(inv, [8, 1]))[0:8, 0:64]
```
（模式对齐 `qkv_proj_rope.py:304`。）

**修复方向：** a5 tilelib 增加 (src, tmp, dst) 的 trsqrt 模板，或放宽 `vmi_trsqrt_with_tmp` 的 constraints。

## 2. `pto.tgather`：compare-form 无模板（20 case）

**报错：**
```
NoMatchingTemplate: no legal template for op='pto.tgather' target='a5';
  template_tgather: operand binding failed: template 'template_tgather' expects 3 operands, got 4;
  template_tgather_mask_row: operand binding failed: template 'template_tgather_mask_row' expects 2 operands, got 4;
  template_tgather_mask_col: operand binding failed: template 'template_tgather_mask_col' expects 2 operands, got 4
```

**根因：** PyPTO `tile.gather_compare`（`pto_ops_datamove.cpp:655`）生成 compare-form：`pto.tgather ins(src, kvalue, tmp {cmpMode, offset}) outs(dst, cdst)`。`lib/TileOps/a5/tgather.py` 只有普通 gather（3 op）和 mask-pattern gather（2 op），无 compare-form 模板。

**最小复现（PyPTO 级）`repro_tgather.py`：**
```python
@pl.jit
def gather_cmp(x: pl.Tensor[[8, 64], pl.FP32], idx: pl.Tensor[[8, 8], pl.INT32],
               y: pl.Tensor[[8, 8], pl.FP32]):
    for blk in pl.spmd(1, name_hint="gather_cmp"):
        y[0:8, 0:8] = pl.gather(x[0:8, 0:64], dim=-1, index=idx[0:8, 0:8])[0:8, 0:8]  # ← 触发
```

**修复方向：** a5 tilelib 注册 compare-form tgather 模板（ins 3 / outs 2，映射 vgather2 compare 语义）。

## 3. `pto.tmax` / `pto.tcolexpand`：custom constraints 不满足（2 case）

**报错（tmax，expert_shared.py:208）：**
```
NoMatchingTemplate: no legal template for op='pto.tmax' target='a5';
  template_tmax: custom constraints are not satisfied;
  template_tmax_1d: custom constraints are not satisfied;
  vmi_tmax: custom constraints are not satisfied
```
tcolexpand（hc_head.py:154）同型报错。

**根因：** 操作数可绑定但 constraints 谓词拒绝。真实 kernel 的 tile 形状（tmax: 累积 max 的 `1x8` f32 tile；tcolexpand: `1x16` → `8x16` broadcast）不在现有模板约束覆盖内。

**最小复现（PTO 级，真实 kernel）：**
- tmax：`sh_gate_up_act_q.pto`（来自 expert_shared 的 `sh_gate_up_act_q` kernel，`pto.tmax ins(1x8xf32, 1x8xf32) outs(1x8xf32)`）
- tcolexpand：`hc_head_pre_fused.pto`（来自 hc_head 的 `hc_head_pre_fused` kernel，`pto.tcolexpand ins(1x16xf32) outs(8x16xf32)`）

（注：简化的 pypto 级 8x64 row_max 和 col_expand 复现都能通过——需要真实 kernel 的具体 tile 形状/pipeline 上下文才触发，所以此类直接附 PTO。）

**修复方向：** 对照 dump 的 IR 逐一放宽对应模板的 constraints，或补充覆盖该形状的模板变体。

## 4. `pto.tsetval` 无 VPTO lowering → `unrealized_conversion_cast`（11 case）

**报错：**
```
error: LLVM Translation failed for operation: builtin.unrealized_conversion_cast
VPTO LLVM emission failed: LLVM IR export failed for vector module
Error: Failed to lower VPTO to LLVM modules.
```

**根因：** PyPTO per-element 写 tile 场景生成 `pto.tsetval ins(%idx, %val : index, i32) outs(%tile_buf)`。`TSetValOp` 的 lowering **只存在于 EmitC 路径**（`lib/PTO/Transforms/PTOToEmitC.cpp`）；`VPTOLLVMEmitter.cpp` 无任何 TSetValOp 转换模式，`translateModuleToLLVMIR` 插入 unrealized cast 兜底后被 LLVM 翻译层拒绝。对照：通过的 hc_post 的 IR 无 tsetval（已 diff 验证）。

**最小复现（PTO 级）：** `prefill_c4_write_map.pto`（来自 prefill_fwd 的 `prefill_c4_write_map` kernel；IR 含 2 个 `pto.tsetval`，均在 `scf.if` 内用 `index` 类型写 `!pto.tile_buf<vec, 1x32xi32>`）。

**修复方向：** VPTOLLVMEmitter（或 CANN900 pipeline）补 TSetValOp lowering（tile_buf 元素写入的向量/标量存储序列）。

## 5. `pto.initialize_l2l_pipe` peer-init-pair 校验失败（4 case）

**报错：**
```
error: 'pto.initialize_l2l_pipe' op requires a complete compatible peer init pair
when local_addr comes from pto.reserve_buffer or pto.import_reserved_buffer
```

**根因：** PTO 输入文件里并无 l2l_pipe op——它是 VPTO pipeline 某个 pass 为 GM pipe buffer 引入的，其 verifier 要求 reserve_buffer 来源的 local_addr 有完整配对的 peer init；PyPTO 生成的单侧 init 无法满足。EmitC 路不做该拆分所以不受影响。

**最小复现（PTO 级）：** `mtp_projection_linear_aiv.pto`（来自 mtp_projection 的 `mtp_projection_linear_aiv` kernel，触发点 `mtp_projection.py:169` 的 `pl.at(CORE_GROUP)` + `pl.matmul` 流水线）。

**修复方向：** 检查引入 initialize_l2l_pipe 的 pass 的配对逻辑，或放宽 reserve_buffer 场景的 verifier（配对可推导时自动补 peer）。

## 6. merged-device 一 ELF 一 entry 限制 vs AIC+AIV 混合 group（13 case）

**报错：**
```
Error: merged device wrapper emission currently requires exactly one PTO entry
function per device ELF; got 2.
Error: Failed to emit VPTO device wrapper source.
```

**根因：** PyPTO 的 kernel 分组把一个 AIC kernel + 一个 AIV kernel 写进同一 PTO 文件（如 `qk_pv_aiv.pto` 含 `@qk_pv_aic` + `@qk_pv_aiv` 两个 entry）。merged-device 模式（`--vpto-emit-merged-device-only`）设计上每 ELF 只带一个 `kernel_entry` wrapper（simpler 按 func_id 选 ELF 跳唯一 entry），遇到多 entry 直接报错。

**最小复现（PTO 级）：** `qk_pv_aiv.pto`（来自 prefill_sparse_attn 的 `qk_pv` 融合组，AIC matmul + AIV softmax）。

**修复方向：** PyPTO 侧在 `kernel_backend=vpto` 时按 kernel kind 拆分 group（一 group 一 ELF），或 PTOAS 侧支持多 entry ELF（wrapper 按 dispatch 参数里的 kernel id 分发到对应 body）。

---

## 附件

- 复现文件（4 个 pypto 级脚本 + 5 个 PTO）：`/home/liuzidi/v4pro_results/repro/`
- 完整 27 case 的失败日志：`/home/liuzidi/v4pro_results/vpto/`
- 详细分析报告：`/home/liuzidi/v4pro_results/vpto_lowering_failure_report.md`
- 对照：EmitC 路线同 27 case 21 个上板 PASS（其余失败均与环境/既有 API 废弃相关，见报告）

## 环境备注

- PTOAS 为 main-llvm19-build + `fix/vpto-merged-device-pypto-abi` 分支（merged-device flag 移植 + PyPTO ABI 适配，见该分支 PR）
- PyPTO 为 `feat/vpto-merged-device-backend`（PR #1）
- pypto-lib 为 `feat/vpto-merged-device-runtime`（PR #1）
