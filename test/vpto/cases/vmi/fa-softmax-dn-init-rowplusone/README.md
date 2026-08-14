# fa-softmax-dn-init-rowplusone 复现速查

> 在本 case 上跑 VMI 融合 / 非 VMI 两路，以及带 `--enable-vecscope-mem-bar` 的组合：性能（summary + 指令时间序列 + hazard 计数）、device 汇编（shim 捕获 IR → bisheng 出 `.s`）、MLIR（各 pass 中间 IR）。
>
> 本文档只记录"怎么跑 + 各产物的期待形态"。所有路径以仓库根为基准的相对路径表示（环境变量里的绝对路径按本机实际情况替换）。

## 0. 提示词（复制粘贴驱动跑产物）

以下提示词可粘给 AI 助手或人执行。每个独立成块，按需选用。执行前先 source §2 的环境变量。

### 0.1 跑性能（sim）

```
在 fa-softmax-dn-init-rowplusone case 上跑 VMI 融合和非 VMI 两路 sim 性能。
参照 README.md §2 准备环境、§3 跑 validation 脚本（切 flags + 跑完恢复）。
两路都跑完后，按 §3 的取数命令给出：rvec_busy、kernal ticks、mte2/mte3 busy cycle、
hazard 总数及类型分解（overlaps with）、compare 是否 PASS。
把 core0_summary_log、core0.veccore0.instr_log.dump、validation.log 三类文件存档到
log/ 下。若要看 membar pass 效果，额外加一路 --enable-vecscope-mem-bar=false 的组合
（注意：--enable-vecscope-mem-bar 默认为 on，关闭它才是需要显式指定的"额外一路"）。
```

期待产物：`core0_summary_log`（SU busy cycle + ticks）、`core0.veccore0.instr_log.dump`（逐指令时间序列）、`validation.log`（含 hazard warning 行）。hazard 数与 compare PASS/FAIL 是正确性关键指标。

### 0.2 出汇编（device IR + `.s`）

```
参照 README.md §5 构造 shim capture 工具，捕获 VMI / 非 VMI 两路真正喂给 bisheng 的
device IR（device_input.ll）。然后用 §5.3 的 bisheng 命令编出 aicore .s（可读汇编）
和 .o（校验用）。验证 .o 是 ELF arch 0x1029。
如需 vf-fusion 关闭的汇编对照，按 §5.4 用 -mllvm 后端选项（7 个，false 分支）出 .s，
并对照 vf-fusion on 版本。两版 .s 行数和 SMEM_BAR 数会不同。
```

期待产物：`vmi_device_input.ll` / `novmi_device_input.ll`（shim 捕获的 IR）、`rpo_vmi.s` / `rpo_novmi.s`（可读汇编）、`*.aicore.o`（ELF 0x1029，`.text` size 校验）。vf-fusion off 版 `.s` 用 `output_vfoff.s` 命名，对照 `output_vfon.s`。

### 0.3 出中间 MLIR

```
参照 README.md §7，用 --mlir-print-ir-after-all 先抓全 pipeline pass 列表，再用
--mlir-print-ir-after=<pass-arg> dump 单个 pass 后的 IR。重点：dump
pto-insert-vecscope-mem-bar 的 before（PTOInferVPTOVecScope 后）和 after，看
pto.mem_bar 插在哪、什么 kind。也 dump vmi-loop-fusion / vmi-load-store-elision 看
融合/消除效果。
```

期待产物：各 pass 的 `.mlir` 文件。membar pass 前后 diff 应只有 `pto.mem_bar` 行的插入（无指令重排）。

## 1. 路由

- **VMI 路**：`--enable-vmi --enable-op-fusion=true`。ptoas 做 VMI 循环融合（`PTOVmiLoopFusion`）+ UB load/store elision（`PTOVmiLoadStoreElision`）。
- **非 VMI 路**：`--enable-op-fusion=false`。走 legacy 低层融合流水线（`PTOLowLevelLoopFusion` + `PTOFusionLoadStoreElision`）。

两者互斥。`--enable-vmi` 单独置 true 无效，必须配 `--enable-op-fusion=true` 才进 VMI 路径。

> **bisheng VF 融合**：VPTO 路径（无论 VMI 还是非 VMI）默认全关。PTOAS 自己负责所有 fusion 和 UB load/store elision，bisheng 只做 codegen，不重复跑 VF 融合/ldst 消除。因此不再需要手动传 `--disable-bisheng-vf-fusion` 或 `-mllvm -cce-vf-enable-*=false`。

## 2. 环境（每次新 shell 都要 source）

```bash
cd <PTOAS_REPO_ROOT>
export ASCEND_HOME_PATH=<your CANN root, e.g. /usr/local/Ascend/cann-9.1.0-beta.3>
source "$ASCEND_HOME_PATH/bin/setenv.bash"
export PTODSL_PYTHON_ROOT=<PTOAS_REPO_ROOT>/ptodsl
# ptodsl 源码 + mlir_core 必须在 install 副本之前
export PYTHONPATH=<PTOAS_REPO_ROOT>/ptodsl:<your mlir_core path>:${PYTHONPATH:-}
export PTO_ISA_ROOT=<your pto-isa root>
export PTOAS_BIN=<PTOAS_REPO_ROOT>/build311/tools/ptoas/ptoas
CASE=test/vpto/cases/vmi/fa-softmax-dn-init-rowplusone
```

> build 用 `build311/tools/ptoas/ptoas`（python wrapper 加载 `build311/python/ptoas/mlir/_mlir_libs/libPTOASCompiler.so`）。重编：`ninja -C build311 libPTOASCompiler.so`。
>
> 也可以写一个 wrapper 脚本封装 venv + env：
> ```bash
> # /tmp/rpo_ptoas_wrapper.sh
> #!/usr/bin/env bash
> export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.1.0-beta.3
> export PYTHONPATH=<mlir_core path>:<PTOAS_REPO_ROOT>/build311/python
> export PTO_ISA_ROOT=<pto-isa root>
> export LD_LIBRARY_PATH=<llvm build-shared>/lib:${LD_LIBRARY_PATH:-}
> exec <PTOAS_REPO_ROOT>/.venv311/bin/python <PTOAS_REPO_ROOT>/build311/tools/ptoas/ptoas "$@"
> ```
> 这样 validation 脚本的 `PTOAS_BIN` 直接指向 wrapper 即可。

## 3. 两路性能（VPTO 路径，sim）

validation 脚本 `test/vpto/scripts/run_host_vpto_validation.sh` 读 case 的 `ptoas.flags`（**优先于** `PTOAS_FLAGS` 环境变量），所以切 VMI 路要临时改 flags、跑完恢复。非 VMI 路不用动（case 默认就是 `--enable-op-fusion=false`）。

```bash
cd <PTOAS_REPO_ROOT>
# 接 §1 的环境变量
CASE_DIR=test/vpto/cases/vmi/fa-softmax-dn-init-rowplusone
FLAGS_FILE="$CASE_DIR/ptoas.flags"
cp "$FLAGS_FILE" /tmp/rpo_flags.bak
trap 'cp /tmp/rpo_flags.bak "$FLAGS_FILE"' EXIT   # 退出必恢复

# --- 测1：VMI 融合 ---
echo "--pto-arch a5 --pto-backend=vpto --enable-vmi --enable-op-fusion=true" > "$FLAGS_FILE"
WS=/tmp/fa_rpo_t1; rm -rf "$WS"; mkdir -p "$WS"
WORK_SPACE="$WS" CASES_ROOT=$PWD/test/vpto/cases/vmi \
  ASCEND_HOME_PATH="$ASCEND_HOME_PATH" PTOAS_BIN="$PTOAS_BIN" \
  CASE_NAME=fa-softmax-dn-init-rowplusone DEVICE=SIM \
  PTODSL_SIM_SOC_VERSION=Ascend950PR_9599 \
  bash test/vpto/scripts/run_host_vpto_validation.sh > /tmp/rpo_t1.run.log 2>&1

# --- 测2：非 VMI（case 默认 flags，直接跑即可）---
cp /tmp/rpo_flags.bak "$FLAGS_FILE"   # 恢复成默认非 VMI flags
WS=/tmp/fa_rpo_t2; rm -rf "$WS"; mkdir -p "$WS"
WORK_SPACE="$WS" CASES_ROOT=$PWD/test/vpto/cases/vmi \
  ASCEND_HOME_PATH="$ASCEND_HOME_PATH" PTOAS_BIN="$PTOAS_BIN" \
  CASE_NAME=fa-softmax-dn-init-rowplusone DEVICE=SIM \
  PTODSL_SIM_SOC_VERSION=Ascend950PR_9599 \
  bash test/vpto/scripts/run_host_vpto_validation.sh > /tmp/rpo_t2.run.log 2>&1

cp /tmp/rpo_flags.bak "$FLAGS_FILE"   # 手动恢复（trap 兜底）
```

**取数 + 查 hazard**：

```bash
for n in 1 2; do
  WS=/tmp/fa_rpo_t$n
  DIR=$(find "$WS" -type d -name "fa-softmax-dn-init-rowplusone" | head -1)
  SUMMARY="$DIR/core0_summary_log"
  INSTLOG="$DIR/core0.veccore0.instr_log.dump"
  echo "=== 测$n ==="
  grep -E "rvec_veccore0_busy_cycle|kernal total ticks|mte2_veccore0_su_busy_cycle|mte3_veccore0_su_busy_cycle" "$SUMMARY"
  echo "hazard count: $(grep -c 'overlaps with' /tmp/rpo_t$n.run.log)"
  grep "overlaps with" /tmp/rpo_t$n.run.log \
    | sed -E 's/.*cur_instr (RV_[A-Z]+).*pre_instr (RV_[A-Z]+).*/\1 vs \2/' | sort | uniq -c
  grep -E "compare passed|nz compare" /tmp/rpo_t$n.run.log | tail -2
done
```

### 性能数据产物

两个文件都要存：

| 文件 | 内容 | 看什么 |
|---|---|---|
| `core0_summary_log` | 各 SU busy cycle 汇总 + kernal/system ticks | mte2/mte3/rvec busy cycle，宏观性能 |
| `core0.veccore0.instr_log.dump` | 每条指令的执行时间序列 | 逐指令分析：哪条 RV_VMAX/VLD/VST 占用、时刻、PC、二进制编码 |

`instr_log.dump` 格式（每行一条指令）：
```
[info] [00002676] (PC: 0x10d0d214) RVECEX   : (Binary: 0x82180782) (ID: 000103) RV_VMAX Dtype: F32
[info] [00002677] (PC: 0x10d0d210) RVECLD   : (Binary: 0x00280008) (ID: 000112) RV_VLD
```
方括号内是 tick 时刻，RVECEX/RVECLD/SCALAR 是执行单元，ID 是指令序号。

存档两路性能数据：
```bash
DEST=log/fa_vmi_rowplusone_full
mkdir -p "$DEST"
for n in 1 2; do  # 1=VMI, 2=非VMI
  DIR=$(find /tmp/fa_rpo_t$n -type d -name "fa-softmax-dn-init-rowplusone" | head -1)
  tag=$([ "$n" = 1 ] && echo vmi || echo novmi)
  cp "$DIR/core0_summary_log" "$DEST/perf_${tag}_core0_summary.txt"
  cp "$DIR/core0.veccore0.instr_log.dump" "$DEST/perf_${tag}_instr_log.dump"
done
```

## 4. 直接 ptoas 出 IR / 汇编（不走 validation 脚本）

只看 IR 或汇编、不需要跑 sim 时，直接调 ptoas（绕过 `ptoas.flags` 临时改文件）：

```bash
# VMI 路：全量 MLIR（每个 pass 后的 IR）
$PTOAS_BIN --pto-arch=a5 --pto-backend=vpto --pto-level=level2 \
  --enable-vmi --enable-op-fusion=true \
  --mlir-print-ir-after-all \
  "$CASE/kernel.pto" -o /tmp/rpo_vmi.fatobj.o > /tmp/rpo_vmi_mlir.log 2>&1

# VMI 路：最终汇编（VPTO→LLVM IR）
$PTOAS_BIN --pto-arch=a5 --pto-backend=vpto --pto-level=level2 \
  --enable-vmi --enable-op-fusion=true \
  --emit-vpto-llvm-ir \
  "$CASE/kernel.pto" -o /tmp/rpo_vmi.ll

# 非 VMI 路：把 --enable-vmi --enable-op-fusion=true 换成 --enable-op-fusion=false
```

`--pto-level=level2` 必带（fusion 流水线在 level2/3 才跑；level1 不跑且有 warning）。

> `--emit-vpto-llvm-ir` 出的是 ptoas 内部 dump 的 IR，**不等于**真正喂给 bisheng 编 aicore 的 IR。要拿能编 aicore `.o` 的 device-only IR，用 §5 的 shim 捕获。

## 5. 汇编捕获（shim capture）：device IR + aicore `.o` / `.s`

ptoas 编 VPTO fatobj 时把 device LLVM IR 经 stdin 喂给 bisheng（带 `-cce-bitcode-is-aicore`）。用一个 shim 假 bisheng 把这路 stdin IR `tee` 落盘，就能拿到**真正喂给 bisheng 的那版 device IR**，再用真 bisheng 编出 aicore `.o` / `.s`。

### 5.1 构造 capture shim（一次性）

shim = 真 CANN 的全量软链，只把 `bin/bisheng` 换成捕获脚本：

```bash
CANN="$ASCEND_HOME_PATH"
SHIM=/tmp/perf/asm/shim_capture
rm -rf "$SHIM"; mkdir -p "$SHIM/bin"

# 顶层所有非 bin 目录软链到真 CANN
for d in "$CANN"/*/; do
  b=$(basename "$d"); [ "$b" = "bin" ] && continue
  ln -sfn "$d" "$SHIM/$b"
done
# bin/ 下除 bisheng 外全部软链
for e in "$CANN"/bin/*; do
  b=$(basename "$e"); [ "$b" = "bisheng" ] && continue
  ln -sfn "$e" "$SHIM/bin/$b"
done

# bin/bisheng = 捕获脚本（只在 -cce-bitcode-is-aicore + stdin 时落盘 IR，其余原样转发）
cat > "$SHIM/bin/bisheng" <<'EOF'
#!/usr/bin/env bash
set -o pipefail
real_bisheng="<your CANN>/bin/bisheng"
capture_file="${PTOAS_CAPTURE_LL:-/tmp/perf/asm/captured.ll}"
capture=0; stdin_ir=0
for arg in "$@"; do
  [[ "$arg" == "-cce-bitcode-is-aicore" ]] && capture=1
  [[ "$arg" == "-" ]] && stdin_ir=1
done
if [[ "$capture" == "1" && "$stdin_ir" == "1" ]]; then
  tee "$capture_file" | "$real_bisheng" "$@"
  exit "${PIPESTATUS[1]}"
fi
exec "$real_bisheng" "$@"
EOF
chmod +x "$SHIM/bin/bisheng"
```

> 必须软链整个 CANN 目录树（顶层所有 dir + bin 下除 bisheng 外所有条目）。漏 `tools/`（含 `bisheng_compiler/bin/bisheng` cc1 前端）报 `unable to locate bisheng cc1 frontend`。

### 5.2 捕获 device IR

```bash
cd <PTOAS_REPO_ROOT>
# 接 §1 的环境变量
PTOAS=build311/tools/ptoas/ptoas
K=test/vpto/cases/vmi/fa-softmax-dn-init-rowplusone/kernel.pto
SHIM=/tmp/perf/asm/shim_capture

# VMI 路:
ART=/tmp/perf/asm/vmi; rm -rf "$ART"; mkdir -p "$ART"
PTOAS_CAPTURE_LL="$ART/device_input.ll" \
  ASCEND_HOME_PATH="$SHIM" \
  "$PTOAS" --pto-arch=a5 --pto-backend=vpto --pto-level=level2 \
  --enable-vmi --enable-op-fusion=true \
  "$K" -o "$ART/fa.fatobj.o"

# 非 VMI 路:
ART=/tmp/perf/asm/novmi; rm -rf "$ART"; mkdir -p "$ART"
PTOAS_CAPTURE_LL="$ART/device_input.ll" \
  ASCEND_HOME_PATH="$SHIM" \
  "$PTOAS" --pto-arch=a5 --pto-backend=vpto --pto-level=level2 \
  --enable-op-fusion=false \
  "$K" -o "$ART/fa.fatobj.o"

ls -la /tmp/perf/asm/vmi/device_input.ll /tmp/perf/asm/novmi/device_input.ll
```

> 关键 env：`PTOAS_CAPTURE_LL` 命名落盘文件，`ASCEND_HOME_PATH="$SHIM"` 让 ptoas 调 shim 而非真 bisheng。`--pto-level=level2` 要带（fusion 流水线才跑）。
>
> 注意：如果用 wrapper 脚本（§2），它内部 `export ASCEND_HOME_PATH` 会覆盖命令行的 `ASCEND_HOME_PATH="$SHIM"`。捕获 device IR 时需直接调 python，不经 wrapper，或在 wrapper 中允许外部 `ASCEND_HOME_PATH` 优先。

### 5.3 用 bisheng 编 aicore `.o` / `.s`（汇编）

捕获的 device IR 可编 aicore object，也可直接出汇编文本。**优先出 `.s`**（人可读的汇编指令），`.o` 留作体积校验：

```bash
# object (校验用):
bisheng --target=hiipu64-hisilicon-cce -march=dav-c310-vec \
  --cce-aicore-arch=dav-c310-vec --cce-aicore-only -O2 \
  -c -x ir <device_input.ll> -o rpo_vmi.aicore.o

# assembly (看指令用), 只改 -c→-S:
bisheng --target=hiipu64-hisilicon-cce -march=dav-c310-vec \
  --cce-aicore-arch=dav-c310-vec --cce-aicore-only -O2 \
  -S -x ir <device_input.ll> -o rpo_vmi.s
# 非 VMI 路: 同上, 把 vmi 换成 novmi (IR 文件 + -o 输出名)
```

验证 `.o`：`file rpo_vmi.aicore.o` 应为 `ELF 64-bit LSB relocatable, *unknown arch 0x1029*`。

### 5.4 vf-fusion off 的可读汇编

> **VPTO 路径默认行为已改变**：PTOAS 现在对所有 VPTO 路径（VMI 和非 VMI）都自动禁用 bisheng 的 VF 融合/ldst 消除（见 §1）。因此 sim 和 `.o` 编译已经都是 VFOFF 行为，不再需要手动注入。

要单独编出可读 `.s`，直接用 bisheng 的 `-S` 路径。由于 ptoas 在调 bisheng 时已经传了 VFOFF 选项（见 `ObjectEmission.cpp`），捕获的 device IR 编出的 `.s` 就是 VFOFF 效果：

```bash
# VMI / 非 VMI 路都可读汇编（bisheng -S，VF 已由 ptoas 关闭）
bisheng --target=hiipu64-hisilicon-cce -march=dav-c310-vec \
  --cce-aicore-arch=dav-c310-vec --cce-aicore-only -O2 \
  -S -x ir <device_input.ll> -o rpo_vmi.s
```

如果需要对照 VFON（VF 融合开启）的汇编——即让 bisheng 后端自己做循环融合和 ldst 消除——不要传 `-mllvm -cce-vf-enable-*=false`：

```bash
# VFON 对照（bisheng -S，不传 VF 选项 = bisheng 默认开启）
bisheng --target=hiipu64-hisilicon-cce -march=dav-c310-vec \
  --cce-aicore-arch=dav-c310-vec --cce-aicore-only -O2 \
  -S -x ir <device_input.ll> -o output_vfon.s
```

> 注意：sim 性能跑出的结果就是 VFOFF 行为（ptoas 已在 bisheng argv 中传了关闭选项）。VFON 汇编仅供对照，不反映 sim 实际性能。
>
> 如果 bisheng `-S` 对某 IR 报 `unsupported option '-S' on device side`（full ptoas argv 带 `-cce-bitcode-is-aicore`），是因为该 argv 走的是 fatobj 路径。用 shim 捕获的 device IR 是裸 IR，直接 `-S` 可用。

## 6. hazard 计数 + 运行日志保存

sim 的 `overlaps with` warning = 运行时访存流水 hazard。每行带完整字段：

```
[warning] [00002857] cur_instr RV_VLDI (id:548 pc:0x10d0d230 vloop_id:1 vloop_pc:0x10d0d228) overlaps with pre_instr RV_VSTI (id:545 pc:0x10d0d224 vloop_id:0 vloop_pc:0x0)
```

字段：方括号内 tick，`cur_instr`/`pre_instr` 是冲突对，`id` 指令序号，`pc` 真实地址，`vloop_id`/`vloop_pc` 虚拟循环位置。两类典型冲突：
- **VLDI vs VSTI**：上个 vloop 写 UB 与下个 vloop 读同一 UB 无同步。
- **VSTI vs VSTI**：相邻两次 UB 写冲突。

### 6.1 运行日志保存 + hazard 计数

每次跑 sim 的运行日志（validation 脚本的 `validation.log`，含 hazard warning 行）必须存档，不能只存 summary。hazard 个数是性能/正确性关键指标。

```bash
DEST=log/fa_vmi_rowplusone_full
for n in 1 2; do  # 1=VMI, 2=非VMI
  DIR=$(find /tmp/fa_rpo_t$n -type d -name "fa-softmax-dn-init-rowplusone" | head -1)
  tag=$([ "$n" = 1 ] && echo vmi || echo novmi)
  cp "$DIR/validation.log" "$DEST/perf_${tag}_run.log"
done

# hazard 计数 + 类型分解
for tag in vmi novmi; do
  LOG="$DEST/perf_${tag}_run.log"
  echo "=== $tag ==="
  echo "hazard 总数: $(grep -c 'overlaps with' "$LOG")"
  grep "overlaps with" "$LOG" \
    | sed -E 's/.*cur_instr (RV_[A-Z]+).*pre_instr (RV_[A-Z]+).*/\1 vs \2/' | sort | uniq -c
done
```

## 7. Dump 各 pass 中间 MLIR

```bash
# 抓全 pipeline pass 列表（先做这步知道有哪些 pass 可 dump）
$PTOAS_BIN --pto-arch=a5 --pto-backend=vpto --pto-level=level2 \
  --enable-vmi --enable-op-fusion=true \
  --emit-vpto -o /dev/null --mlir-print-ir-after-all 2>&1 \
  | grep -oE "IR Dump After [A-Za-z0-9_]+ \([a-z0-9-]+\)" | awk '!seen[$0]++'
```

dump 单个 pass：

```bash
$PTOAS_BIN --pto-arch=a5 --pto-backend=vpto --pto-level=level2 \
  --enable-vmi --enable-op-fusion=true \
  --emit-vpto "$CASE/kernel.pto" -o /dev/null \
  --mlir-print-ir-after="<pass-arg>" 2>&1 \
  | grep -v '^TileLib daemon\|^Info: ptodsl' \
  | awk '/^\/\/ -----.*IR Dump/ {p=1} p {print}' > stage.mlir
```

> `--enable-vecscope-mem-bar` 默认为 on（pass 名 `pto-insert-vecscope-mem-bar`，跑在 `PTOInferVPTOVecScope` 之后、`VPTOExpandWrapperOps` 之前）。要 dump 它的 before/after 看 membar 插入：

```bash
$PTOAS_BIN --pto-arch=a5 --pto-backend=vpto --pto-level=level2 \
  --enable-vmi --enable-op-fusion=true \
  --emit-vpto "$CASE/kernel.pto" -o /dev/null \
  --mlir-print-ir-after="pto-insert-vecscope-mem-bar" 2>&1 \
  | grep -v '^TileLib daemon\|^Info: ptodsl' > membar_after.mlir
```

## 8. 产物清单

跑完上述各步，典型产物：

```
log/fa_vmi_rowplusone_full/
  perf_vmi_core0_summary.txt       # VMI 路 summary
  perf_novmi_core0_summary.txt     # 非 VMI 路 summary
  perf_vmi_instr_log.dump          # VMI 路指令时间序列
  perf_novmi_instr_log.dump        # 非 VMI 路指令时间序列
  perf_vmi_run.log                 # VMI 路 validation 日志（含 hazard warning）
  perf_novmi_run.log               # 非 VMI 路
  vmi_device_input.ll              # VMI 路 device IR（shim 捕获）
  novmi_device_input.ll            # 非 VMI 路 device IR（shim 捕获）
  rpo_vmi.s / rpo_novmi.s          # 两路汇编（VFOFF：bisheng VF 已关）
  rpo_vmi.aicore.o / rpo_novmi.aicore.o   # aicore obj（.text 校验，ELF arch 0x1029）
  vmi_mlir_all.log                 # VMI 路全 pass 中间 MLIR
  novmi_mlir_all.log               # 非 VMI 路全 pass 中间 MLIR
```

> 注意：两路 `.s` / `.o` / sim 性能现在都是 VFOFF 行为（bisheng VF 融合已由 ptoas 默认关闭）。如需 VFON 对照汇编，参考 §5.4。
