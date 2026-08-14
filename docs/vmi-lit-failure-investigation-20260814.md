# VMI lit 大量失败调查记录

## 背景

本记录针对 `feature-vmi-vf` 在 rebase 主线后进行 VMI/VPTO lit 验证时出现的大量失败，供后续接手人员复现和继续定位。本文只记录当前证据和调查结论，不修改测试期望，也不把失败标记为已修复。

调查时的代码状态：

```text
branch: feature-vmi-vf
HEAD:   dcbaf03edf800152d0d85acbd2f8fa4c38cfa242
```

工作区另有未提交的 VMI 编译器、PTODSL runtime 和 lit 适配修改；这些修改必须与本文记录的构建产物区分开。

## 复现条件

使用 `build-llvm21-current` 构建目录和其对应的 PTOAS/MLIR Python 环境，运行此前的 VMI/VPTO lit subset。具体测试集合以当时保存的 lit 过滤条件为准，覆盖 `test/lit/vpto`、`test/lit/vmi_new` 以及少量 `test/lit/pto`。

典型结果为：

```text
114 pass
37 fail
1686 excluded
```

失败可用以下最小 IR 现象复现（把 `pto.yield` 放在 fusion region 内）：

```mlir
module {
  func.func @f(%x: i32) {
    %r = pto.fusion_region {
      pto.yield(%x) : (i32) -> ()
    } : i32
    return
  }
}
```

在出问题的新 Python/CLI 构建中，解析或验证该结构时会报：

```text
block with no terminator, has "pto.yield"()
```

受影响的测试包括 PTODSL VMI candidate、fusion region loop fusion、load/store elision、softmax compute 和 tile handle control-flow 等多组用例，例如：

```text
ptodsl_vmi_local_elementwise_candidates
ptodsl_vmi_composite_provider
op_fusion_region_pipeline_level2/3
op_fusion_low_level_loop_*
vmi_fusion_region_loop_elide
softmax_compute_ops
materialize_tile_handles_control_flow_result
```

## 已确认事实

1. `include/PTO/IR/PTOOps.td` 中 `YieldOp` 的定义包含 `Terminator` trait：

   ```tablegen
   def YieldOp : PTO_Op<"yield", [Terminator, ParentOneOf<["FusionRegionOp"]>]>;
   ```

2. 新生成的 PTO op 头文件中也包含 `::mlir::OpTrait::IsTerminator`。

3. 使用 `pto-test-opt` 解析同类最小 fusion region 时可以通过，说明 ODS 定义和文本形式本身不是明显错误。

4. 旧的 `build-llvm21-assert` Python 构建也能正确识别 `pto.yield` 为 terminator。

5. 新的 `build-llvm21-current` Python/CLI 构建会把同一个 `pto.yield` 当成非 terminator。

### 复核（20260814 续查）

复核推翻了上文第 5、6 条的归因，并定位到更直接的事实：

6. `build-llvm21-current` 是一个**未完成的构建**，而非一套完整但 ABI 不同的运行时：

   - `build-llvm21-current/tools/pto-test-opt/` 下只有 CMake 残留（`CMakeFiles/`、`cmake_install.cmake`），**没有 `pto-test-opt` 二进制**。
   - `build-llvm21-current/python/` 下**没有任何 `.so` 扩展**（既无 `ptoas/_core...so`、`pto/ptoas.so`，也无 `mlir/_mlir_libs/_mlir...so` 和 `libPTOASCompiler.so`）。
   - `build-llvm21-current/tools/ptoas/ptoas` 是一个 3KB Python wrapper，启动时即报 `unable to locate the configured ptoas Python package root: expected .../ptoas/_cli.py` 并退出，因为 `build-llvm21-current/python/ptoas/` 下确实没有 `_cli.py`。

7. 因此 lit 在 `build-llvm21-current` 下找不到本构建目录的 `pto-test-opt`，按 `lit.cfg.py` 的 `tool_dirs` 顺序回退到 `llvm_tools_dir`（`llvm-build-assert/bin`），但该目录下**也没有 `pto-test-opt`**（只有 `llvm-lit`、`mlir-opt` 等通用工具）。也就是说，文档中"114 pass 37 fail"的结果是在一个工具链不完整、`pto-test-opt` 实际缺失或来自其它意外路径的环境下跑出来的，不能当作可靠结果。

8. 用完整且一致的 `build-llvm21-assert` 直接复现最小 fusion region，`pto.yield` 被正确识别为 terminator，无 `block with no terminator` 错误：

   ```text
   $ pto-test-opt min_fusion_region.pto    # build-llvm21-assert
   module {
     func.func @f(%arg0: i32) {
       %0 = pto.fusion_region {
         pto.yield(%arg0) : (i32) -> ()
       } : i32
       return
     }
   }
   ```

9. 用 `build-llvm21-assert` 重跑文档列为失败的 `vmi_to_vpto_iota_group1_tail.pto`（RUN 行 `pto-test-opt %s -vmi-lower-unified-to-legacy -vmi-to-vpto | FileCheck %s`），FileCheck 退出码为 0，**测试通过**。`pto-test-opt` 与 FileCheck 均来自 `llvm-build-assert`，MLIR 运行时为同一套。

## 当前判断

修正后的根因：37 个失败来自一个**不完整的 `build-llvm21-current` 构建环境**，而非"PTOAS Python binding 与 MLIR runtime 来自不同 LLVM/MLIR 构建导致 op trait 注册不一致"。证据：

- `build-llvm21-current` 既缺 `pto-test-opt` 二进制，又缺 Python 扩展 `.so`，`ptoas` wrapper 无法启动——它根本不是一套可用的运行时；
- 文档第 6 条所称"两套 `libMLIRIR.so` 内容不同、assertions on/off 不同"在 PTOAS 本机并不能复现成"terminator 被当非 terminator"，因为 `pto-test-opt` 在同一个 `llvm-build-assert` MLIR 上工作正常；
- 同一套 `build-llvm21-assert`（含完整 `pto-test-opt`、完整 Python 扩展、`llvm-build-assert` 的 MLIR）下，最小 fusion region 和被列为失败的用例均通过。

因此暂不应通过删除或放宽 FileCheck 来处理这些失败。那样可能掩盖统一的构建完整性问题。

## 构建现状和限制

复核确认 `build-llvm21-current` 是一个**中断的构建**：`pto-test-opt` 二进制和 Python 扩展 `.so` 均未生成，`ptoas` wrapper 无法启动。此前尝试强制重编译时，`lib/PTO/IR/PTO.cpp` 编译耗时过长导致构建被中止，这与现在观察到的产物缺失一致。因此该构建目录下的任何 lit 结果都不可靠，37 个失败不能据此判定为真实测试缺陷。

作为对照，`build-llvm21-assert` 是**完整且一致的构建**：`pto-test-opt` 二进制、Python 扩展 `.so`、`libPTOASCompiler.so` 齐全，且共用 `llvm-build-assert` 的 MLIR 运行时。最小 fusion region 与抽样复跑的被列失败用例均在此构建下通过。

工作区中的 `build-llvm21-current/` 是未跟踪构建产物，不属于源代码修改。合规检查器扫描它时会报路径越界，这是检查范围问题，不是 PTOAS 源码合规错误。

## 全量 lit 回归（build-llvm21-assert）

在完整且一致的 `build-llvm21-assert` 下执行全量 lit（1837 个测试）：

```text
Total Discovered Tests: 1837
  Unsupported:    1 (0.05%)
  Passed     : 1808 (98.42%)
  Failed     :   28 (1.52%)
```

文档原先记录的 37 个失败在该完整构建下降为 28 个，且**不再出现 `block with no terminator` 类错误**，印证了上文"37 个失败由 `build-llvm21-current` 工具链缺失导致"的判断。

### 28 个失败的定性

经逐类排查，这 28 个失败**不是构建目录完整性问题**，也**不是二进制过时**——`build-llvm21-assert` 的 `pto-test-opt` 二进制（08-13 20:57）已编译进工作区所有相关源码改动（`.o` 时间戳均晚于对应 `.cpp`，例如 `ExpandTileOp.cpp` 源 20:02 → `.o` 20:06，`FoldTileBufIntrinsics.cpp` 源 20:26 → `.o` 20:27）。其性质为**工作区源码改动产生的 IR 输出形态变化与 `.pto` 测试的 FileCheck 期望不一致**：

- 绝大多数失败用 RUN 行调用 `ptoas`（Python binding），底层 `pto-test-opt` 直接运行时 exit 0、IR 正常输出，说明不是崩溃或工具链问题；
- 典型例如 `fold_tile_buf_intrinsics`：FileCheck `ADDR` 前缀仍期望旧形态 `pto.pointer_cast(`，而 `FoldTileBufIntrinsics.cpp` 改动（+98 行）后的实际输出已变为 `pto.tile_buf_addr` + `pto.alloc_tile addr =` 形态；
- 受影响用例集中在 `ExpandTileOp`、`FoldTileBufIntrinsics`、`PTOVmiLoadStoreElision`、`VMILowerUnifiedToLegacy`、`Utils` 等 5 个有实质改动（+198/-94 行）的 Transform 源文件所覆盖的测试，以及对应的 `ptodsl_vmi_*`、`vmi_fusion_region_loop_elide`、`materialize_tile_handles_control_flow_result`、`expand_tile_op_*`、`fold_tile_buf_intrinsics`、`op_fusion_region_pipeline_*` 等用例。

28 个失败清单：

```text
vpto/ptodsl_vmi_local_reduce_candidates
vpto/ptodsl_vmi_local_broadcast_candidates
vpto/ptodsl_vmi_tileop_provider
vpto/ptodsl_vmi_local_elementwise_fallback
vpto/ptodsl_vmi_local_convert_fallback
vpto/ptodsl_vmi_local_convert_candidates
vpto/ptodsl_vmi_high_precision_div_ops
vpto/ptodsl_vmi_local_reduce_fallback
vpto/ptodsl_vmi_composite_provider
vpto/vmi_fusion_region_loop_elide
vpto/ptodsl_vmi_sinkhorn_grouped_candidates
vpto/ptodsl_vmi_local_elementwise_candidates
vpto/vmi_plan_f3_boundary
vpto/expand_tile_op_tilelang_trecip
vpto/ptodsl_vmi_flash_attention_softmax
vpto/auto_vecscope_infer_revert_cse_mask
tile_fusion/op_fusion_backend_lifecycle_level3
vpto/fold_tile_buf_intrinsics
vpto/expand_tile_op_ptodsl_tadd
vpto/ptodsl_vmi_narrow_row_broadcast_candidates
pto/tpush_tpop_dynamic_validshape_a5
vpto/expand_tile_op_tilelang_tdivs
vpto/ptodsl_vmi_rope_128b_candidates
vpto/ptodsl_vmi_trowsum_wide_workspace
vpto/auto_vecscope_infer_shared_pintlv
vpto/ptodsl_vmi_sqrt_ops
pto/materialize_tile_handles_control_flow_result
tile_fusion/op_fusion_region_pipeline_level2
```

## 构建目录处理

已删除 `build-llvm21-current/`（123M，git 未跟踪、无任何跟踪文件引用），原因：该目录是中断的构建产物，无 `pto-test-opt` 二进制、无 Python 扩展 `.so`、`ptoas` wrapper 无法启动，其 lit 结果不可信。

保留 `build-llvm21-assert/`（1.3G，完整且一致）作为基准构建。

## 建议的后续步骤

1. 28 个失败属"源码改动 vs 测试期望不一致"，应逐个核对实际 IR 输出与 FileCheck 期望，确认源码改动是有意为之后再同步更新 `.pto` 测试期望（仍不应盲目放宽 FileCheck）。
2. 若某些源码改动并非有意，应回退对应 `.cpp` 而非改测试。
3. 28 个失败全部转为 pass 后，再执行全量 lit 确认无回归，并重新运行 `fa-softmax-dn-init-rowplusone` 性能验证。

## 结论

修正后的结论分两层：

1. 原先的 37 个失败中，由 `build-llvm21-current` 构建不完整导致的部分已在完整构建 `build-llvm21-assert` 下消除，`block with no terminator` 类错误不再出现。`build-llvm21-current` 已删除。
2. 残留的 28 个失败是**工作区源码改动（已编译进二进制）产生的 IR 形态变化与 FileCheck 期望不一致**，属真实待处理项，需逐个核对源码意图后同步测试期望或回退源码，而非批量放宽 FileCheck。
