# PTOAS VMI 接入 VfSim Cost Model 开发设计

## 1. 目标

在 VMI 已经 lower 为 VPTO low-level IR、真实 MI op 类型基本确定后接入 VfSim，完成
ABCABC 和 AABBCC 两种 loop unroll 形式的寻优。

本阶段只开发 unroll，不开发指令重排。基于 VfSim cost model 的 MI 重排属于后续独立
开发任务，不在本文的 pass、接口、开关和验收范围内。

设计边界：

- PTOAS 负责生成 IR、保证变换合法性并应用优化方案。
- VfSim 负责将候选方案转换为 `VfInfo`、预测 cycle 并选择最优方案。
- PTOAS 与 VfSim 通过同一进程内的 MLIR IR 交互，不使用 JSON 或临时文件。

## 2. 接入位置

### 2.1 编译链路

VfSim 接在 VMI lowering 完成后的公共 VPTO emission pipeline 中。建议链路为：

```text
VMIToVPTO
  -> VPTO emission preparation passes
  -> VPTOSoftPostUpdate（可选）
  -> LoopInvariantCodeMotion
  -> PTONarrowVPTOLoopCounters
  -> Canonicalizer
  -> CSE
  -> VPTOCombineReductions
  -> CSE
  -> VfSimUnrollPlanner          （可选，VfSim 评估两类 unroll）
  -> ApplyLoopUnroll             （可选，PTOAS 按计划修改 IR）
  -> Canonicalizer
  -> CSE
  -> PTOValidateVPTOEmissionIR
  -> VPTO LLVM lowering
```

具体接入点是 `VPTOCombineReductions` 后的第二个 `CSE` 与
`PTOValidateVPTOEmissionIR` 之间。

现有 `VPTOScheduler` 不属于本设计的开发范围，不移动、不删除。开发和验证本阶段
VfSim unroll 链路时保持该 pass 关闭。

### 2.2 该位置的 IR 特征

此时：

- TileOp 和 VMI 已完成展开，`vexp`、`vcvt`、`vpack`、`vsstb` 等 VPTO op 已确定。
- VMI loop fusion、layout assignment、load/store elision 等既有优化已经完成。
- `VPTOCombineReductions` 已完成 reduction 指令组合，VfSim 看到的是稳定的 MI 形态。
- LICM、Canonicalizer 和 CSE 已清理循环不变量及重复辅助 op。
- `scf.for`、SSA use-def、dtype、vector width 和地址关系仍然保留，适合构建 `VfInfo`。
- 尚未进入 LLVM lowering，仍可通过 MLIR pass 安全地写回和应用优化方案。

## 3. 对接形式

### 3.1 源码管理与构建

VfSimulator 以 git submodule 形式放入 PTOAS，并以 C++ 源码参与构建：

```text
PTOAS/
  3rdparty/
    VfSimulator/        # submodule，PTOAS commit 固定具体 VfSim commit
```

```text
VfSimulator repository
  -> native C++ library targets
  -> PTOAS links targets at build time
  -> PTOAS pass calls VfSim C++ API in process
```

约定：

- PTOAS 不复制和维护 VfSim 核心源码。
- PTOAS 只记录 submodule commit，VfSim 版本升级通过更新该 commit 完成。
- `PTO_ENABLE_VFSIM_COSTMODEL=OFF` 时不编译、不链接 VfSim。
- 首期只支持 build tree；install/export 作为后续独立任务。

### 3.2 C++ API 边界

接口采用 MLIR generic operation，VfSim 不 include PTOAS dialect 的强类型 C++ op 头文件：

```cpp
namespace vfsim {

struct PlannerOptions {
  unsigned maxUnrollFactor = 8;
  bool dumpCandidates = false;
};

mlir::LogicalResult planVmiUnrollIR(
    mlir::Operation *scope,
    const PlannerOptions &options);

} // namespace vfsim
```

| 项目 | 约定 |
|---|---|
| 输入对象 | 包含目标 `pto.vec_scope` / `scf.for` 的 `mlir::Operation *` |
| IR 所有权 | 始终归 PTOAS，VfSim 不跨 pass 保存 operation 指针 |
| op 识别 | 使用 operation name、operand/result type、attribute 和 SSA use-def |
| VfSim 行为 | 只读取 IR 并写计划 attr，不增删、替换或移动 operation |
| 返回值 | `success()` 表示完成或合法跳过；接口或 IR 契约损坏返回 `failure()` |

调用链：

```text
PTOAS VfSimUnrollPlanner pass
  -> 将目标 vec_scope / scf.for 交给 VfSim generic MLIR adapter
  -> adapter 构造 VfInfo
  -> VfSim core 预测候选 cycle
  -> adapter 把最优 factor 写回目标 scf.for
  -> PTOAS 后端可按属性来源区分并消费 factor
```

### 3.3 Legacy 与 VMI 接口路由

PTOAS 同时保留两条 VfSim 接入链路：

| 编译链路 | VfSim 接入层次 | C++ API | 输出 |
|---|---|---|---|
| Legacy TileOp -> MI | FusionPlan 后的 TileOp IR | `planTileFusionIR()` | legacy row/col unroll attrs |
| VMI -> VPTO | `VPTOCombineReductions + CSE` 后的 low-level VPTO IR | `planVmiUnrollIR()` | loop 上的 `pto.vfsim.unroll_factor : i32` |

PTOAS 在构建 pass pipeline 时统一选择 planner mode：

```cpp
enum class VfSimPlannerMode {
  Disabled,
  LegacyTileOp,
  VmiLowLevel,
};
```

选择矩阵：

| Cost model | 实际 VMI fusion pipeline | Planner mode | 接入位置 |
|---:|---:|---|---|
| 关 | 关 | `Disabled` | 不调用 VfSim，保持默认 legacy 链路 |
| 关 | 开 | `Disabled` | 不调用 VfSim，保持默认 VMI 链路 |
| 开 | 关 | `LegacyTileOp` | FusionPlan 后 |
| 开 | 开 | `VmiLowLevel` | `VPTOCombineReductions + CSE` 后 |

模式选择必须使用 PTOAS 最终计算出的 `useVMIFusionPipeline`，不能只检查原始
`--enable-vmi`。VMI 是否真正生效还受到 target arch、op fusion 和 backend mode 等条件
约束。

路由规则：

- 同一次编译只能调用一个 VfSim planner。
- `LegacyTileOp` 模式只在 FusionPlan 后调用 `planTileFusionIR()`。
- `VmiLowLevel` 模式跳过 FusionPlan 中的 legacy VfSim 调用，只在 low-level 接入点调用
  `planVmiUnrollIR()`。
- VfSim 不通过观察输入 IR 内容猜测当前模式；PTOAS 显式选择并调用对应 API。
- 用户显式请求 VMI cost model，但实际无法形成 `useVMIFusionPipeline` 时给出明确诊断，
  不静默回退到 legacy planner。

## 4. 输入协议

VfSim 接收经过 `VPTOCombineReductions + CSE` 的 low-level VPTO IR。

```mlir
pto.vec_scope {
  %sum_next = scf.for %i = %c0 to %c128 step %c1
      iter_args(%sum = %sum_init) -> !pto.vreg<64xf32> {
    %off = arith.muli %i, %c64 : index
    %x = pto.vlds %src[%off]
        : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
    %d = pto.vsub %x, %max, %pred
        : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32>
          -> !pto.vreg<64xf32>
    %e = pto.vexp %d, %pred
        : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
    %next = pto.vadd %sum, %e, %pred
        : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32>
          -> !pto.vreg<64xf32>
    scf.yield %next : !pto.vreg<64xf32>
  }
}
```

Adapter 从 IR 提取：

| 信息 | 来源 |
|---|---|
| 候选循环及 trip count | `scf.for` 的 lower bound、upper bound、step |
| 指令类型 | operation name，例如 `pto.vexp` |
| 数据依赖 | SSA operand/result use-def |
| dtype 与 vector width | `!pto.vreg<...>`、pointer 和 mask 类型 |
| loop-carried 依赖 | `iter_args` 与 `scf.yield` |
| 内存访问 | `pto.vlds`、`pto.vsts`、`pto.vsstb` 及地址表达式 |
| vec scope 边界 | `pto.vec_scope` / `pto.strict_vec_scope` |

Adapter 将真实 MI op 转为 `VfInfo`。`vbitcast` 等 zero-cost alias 不进入 VfSim 指令流，
但其 producer-consumer 依赖必须重连；常量、地址计算和 mask 构造按 adapter 分类处理。

### 4.1 MI 判定依据

不能根据 op 名字是否以 `pto.v` 开头，也不能根据是否继承 `PTO_VectorMicroOp` 判断其
是否对应真实 MI。`pto.vadd` 和 `pto.vbitcast` 都属于 vector micro-op IR，但二者的
emission 行为不同：

```text
pto.vadd
  -> LowerBinaryMaskedOpPattern
  -> llvm.hivm.vadd.* intrinsic
  -> 真实 RV_VADD 指令

pto.vbitcast
  -> LowerVbitcastOpPattern
  -> LLVM::BitcastOp
  -> 通常不产生独立硬件指令
```

PTOAS 当前没有一张完整、统一、机器可读的“VPTO op 到真实 MI”映射表。Adapter 的判定
依据按以下优先级确定：

1. `VPTOLLVMEmitter.cpp` 中该 VPTO op 的 lowering pattern。
2. lowering 是否生成 `llvm.hivm.*` intrinsic call，以及对应的 intrinsic family。
3. `docs/isa/micro-isa/` 中记录的 PTO op、RV opcode 和 pipeline 信息。
4. CCE 最终反汇编；这是确认真实指令类型和数量的最终依据。

`VPTOOps.td` 只用于确认 op 的 operand、result、type 和 attribute 结构，不能单独作为
“是否产生真实 MI”的依据。

### 4.2 Adapter 显式分类

Adapter 使用显式 registry，不对未知 `pto.*` op 做名称推断：

```cpp
enum class OpModelKind {
  PhysicalMI,
  ZeroCostAlias,
  Structural,
  ScalarAddress,
  PredicateSetup,
  Composite,
  Unsupported,
};
```

| 分类 | 处理 | 示例 |
|---|---|---|
| `PhysicalMI` | 生成一个真实 `VfInst` | `vadd`、`vexp`、`vlds`、`vsts` |
| `ZeroCostAlias` | 不生成指令，重连 value 和依赖 | `vbitcast` |
| `Structural` | 构造 program/loop 结构，不计为 MI | `scf.for`、`scf.yield`、`vec_scope` |
| `ScalarAddress` | 解析地址；循环内动态指令需要建模或拒绝 | `arith.muli`、`pto.addptr` |
| `PredicateSetup` | 解析 mask；是否计时取决于真实 lowering 和所在位置 | `pge`、`pset` |
| `Composite` | 按 emitter 语义展开成多个 `VfInst` | 一个 VPTO op 对应多个 intrinsic 的情况 |
| `Unsupported` | warning 并跳过当前 loop | registry 中未登记的 op/form |

Registry 的 key 不能只有 op name。对于 `vcvt` 等指令，至少还要包含输入/输出 dtype、
vector width 和决定真实 intrinsic form 的 attributes。

### 4.3 Zero-cost alias 与依赖重连

`vbitcast` 不进入 `VfInfo.instructions`，但不能直接删除其 SSA 依赖。Adapter 维护
canonical value 映射：

```mlir
%a = pto.vcvt ...
%b = pto.vbitcast %a
%c = pto.vpack %b, "LOWER"
```

转换结果：

```text
VfInst(vcvt) -> VfInst(vpack)
canonicalValue[%b] = canonicalValue[%a]
```

`%b` 可以保留自己的逻辑 dtype 和 shape 信息，但复用 `%a` 的物理 storage/value ID 和
producer；它不产生 cycle，也不分配新的物理 vector register。连续多个 alias 需要递归
解析到最终 canonical producer。

### 4.4 标量与 mask op

标量和 mask op 不能统一按零开销过滤：

- 循环外、所有 unroll 候选共享的固定开销可以不进入候选差异时间。
- 循环内且数量随 factor 改变的地址计算可能生成真实 scalar instruction，需要建模；首期
  无法建模时跳过该 loop。
- `pge/pset` 可能 lower 成真实 predicate instruction。首期可以忽略已经被 CSE/LICM
  提到循环外的 mask setup；循环内动态 mask 必须建模或拒绝。
- 被忽略的辅助 op 仍要保留其对真实 MI operand、地址和有效 lane 信息的语义影响。

首期 registry 采用严格 allowlist。遇到未分类 op、未支持 dtype/form 或一个 op 的 lowering
结果不明确时，输出明确的 skip reason，不把它默认当成 vector MI 或 zero-cost op。

### 4.5 候选 loop 识别

VfSim 只对 VMI loop fusion 生成的循环做 unroll 寻优。VMI loop fusion pass 在成功创建
融合循环后写入：

```mlir
scf.for ... {
  ...
} {
  pto.vmi.loop_fused,
  pto.vmi.loop_fusion.id = 0 : i64
}
```

| 属性 | 含义 |
|---|---|
| `pto.vmi.loop_fused` | UnitAttr，表示该 `scf.for` 是 VMI loop fusion 的结果 |
| `pto.vmi.loop_fusion.id` | 当前 function 内稳定且唯一的融合 loop ID，用于关联、dump 和诊断 |

VfSim planner 的候选入口条件为：

```text
isa<scf::ForOp>(op)
&& op.hasAttr("pto.vmi.loop_fused")
&& op.hasAttr("pto.vmi.loop_fusion.id")
```

不带 `pto.vmi.loop_fused` 的 loop 一律不进入 unroll 搜索，包括 VMI 模板内部已经固定的
循环，例如 `tcolmax` 模板中的 split loop。VfSim 不根据 `pto.tilelib.candidate`、
`pto.vmi.fusion.tileop`、loop body 中的 op 数量或属性缺失情况反向推断 loop 来源。

`VMIToVPTO`、Canonicalizer、CSE 以及中间的 loop rewrite 必须保留这两个来源属性，直到
`VfSimUnrollPlanner` 完成候选识别。planner 成功返回后，PTOAS wrapper 删除这两个内部
选择属性，只保留 costmodel 结果和其他仍由后端使用的语义属性。

识别为 fused loop 后，还需要满足以下条件才进入 cycle 搜索：

- trip count 为静态常量且大于 1；
- 至少存在一个位于 `2..min(8, trip_count)` 的候选 factor；
- loop body 中所有影响预测的 op/form 均可由 adapter 处理；
- loop-carried 和 live-out 语义可以正确构造 ABCABC/AABBCC 候选。

## 5. Unroll 语义

### 5.1 两种展开形式

设原循环体按依赖顺序记为 `ABC`，factor 为 2：

| 模式 | 展开后的主要指令顺序 | 作用 |
|---|---|---|
| ABCABC | `A1 B1 C1 A2 B2 C2` | 增加单次循环体指令数，摊薄 loop 控制开销 |
| AABBCC | `A1 A2 B1 B2 C1 C2` | 同时改变跨迭代指令顺序，为乱序发射和双发射提供更多机会 |

AABBCC 可能增加同时存活的中间 value 和寄存器压力。VfSim 必须在候选 program 中执行
虚拟寄存器 live-range normalization，并把潜在 register spill 的影响纳入候选有效性或
时间评估。

### 5.2 完整生命周期位于循环内

完整生命周期是指一份数据完成以下过程：

```text
UB load -> 一系列 vector 运算 -> UB store
```

如果 load、compute 和 store 都位于同一个 loop body：

- ABCABC 的主要收益来自增加循环体指令数，降低 loop 结构本身的性能损失。
- AABBCC 除了降低 loop 开销，还改变指令顺序，可能提高乱序发射和双发射效率。
- AABBCC 同时可能延长 live range、提高寄存器压力并触发 register spill。

### 5.3 存在 loop-carried 或 loop live-out value

如果 store 位于循环外，循环中的中间结果通过 `scf.iter_args`、`scf.yield` 或循环外 use
继续存活，则 unroll 不能只做文本复制。每个 unroll lane 必须使用独立 value，循环结束后
再按原语义归并。

原始形式：

```text
for (...) {
  vlds(v1, mem1)
  vadds(v2, v1, 0.1)
  vadd(v2, v2, v1)
}
vsts(v2, mem2)
```

AABBCC factor=2：

```text
for (...) {
  vlds(v1_1, mem1)
  vlds(v1_2, mem1 + offset)
  vadds(v2_1, v1_1, 0.1)
  vadds(v2_2, v1_2, 0.1)
  vadd(v2_1, v2_1, v1_1)
  vadd(v2_2, v2_2, v1_2)
}
vadd(v2, v2_1, v2_2)
vsts(v2, mem2)
```

ABCABC factor=2：

```text
for (...) {
  vlds(v1_1, mem1)
  vadds(v2_1, v1_1, 0.1)
  vadd(v2_1, v2_1, v1_1)
  vlds(v1_2, mem1 + offset)
  vadds(v2_2, v1_2, 0.1)
  vadd(v2_2, v2_2, v1_2)
}
vadd(v2, v2_1, v2_2)
vsts(v2, mem2)
```

这种情况下，ABCABC 也会改变实际依赖图：原来的单条 loop-carried 依赖链被拆成多条
lane-local 依赖链，并在循环外归并，因此其收益不再只来自 loop 控制开销。

Adapter 必须根据 SSA 判断生命周期边界：

| IR 信息 | 处理 |
|---|---|
| `scf.iter_args` | 识别 loop-carried value，并为每个 unroll lane 建立独立初值和更新链 |
| `scf.yield` | 识别下一次迭代状态和循环结果 |
| loop result 的外部 user | 识别 loop live-out value，并生成语义等价的循环后归并 |
| loop 内 `vsts/vsstb` | 识别生命周期已在每个迭代内部闭合的结果 |

归并操作必须依据原算子语义生成。只有 reduction/association 语义明确且合法时，才能使用
示例中的 `vadd` 归并；普通覆盖式 live-out 需要保留原迭代顺序所决定的最终 value。

## 6. 输出协议

VfSim 将选中的计划写到目标 `scf.for`，不直接展开循环。对外只返回最优候选的
factor，值为 `1` 表示不展开：

```mlir
scf.for %i = %c0 to %c128 step %c1
    attributes {
      pto.vfsim.unroll_factor = 4 : i32
    } {
  ...
}
```

| 属性 | 含义 |
|---|---|
| `pto.vfsim.unroll_factor` | VfSim 全局最优候选的展开因子，signless i32；`1` 表示不展开 |
| `pto.unroll_factor` | 手写或前端生成的 unroll hint；不由 VfSim planner 写入或覆盖 |

`pto.vmi.loop_fused` 和 `pto.vmi.loop_fusion.id` 只用于 planner 候选识别，预测阶段
结束后由 PTOAS 删除，不进入最终 VPTO IR。

首期将两种形式作为互斥候选统一比较，但 attr 不表达展开形式。ABCABC/AABBCC 模式
只保留在 VfSim 的候选预测和调试输出中，下游统一按自己的展开与调度策略消费 factor。

首期候选规则：

```text
1 <= factor <= min(8, trip_count)
候选集合 = {no-unroll} U {ABCABC(f)} U {AABBCC(f)}
```

`factor=1` 只对应 no-unroll baseline。对于 `factor>1`，即使
`trip_count % factor != 0` 也必须构造候选：主体使用完整 unroll group，剩余
`trip_count % factor` 次迭代构造成 residual/tail loop，并与主体一起预测。

两类候选必须分别按真实语义构造。loop-carried value 必须按 unroll lane 拆分，避免错误
复用同一虚拟寄存器而人为拉长依赖链；需要归并时，归并指令也必须进入 VfSimProgram。

当前 `PTOUnrollLoops` 只读取手写 hint `pto.unroll_factor`，不消费
`pto.vfsim.unroll_factor`。VfSim 不在 planner 调用过程中直接执行结构性 rewrite；
后续 costmodel-aware 的应用阶段可单独读取专用属性，避免与手写策略混淆。

## 7. Pass 职责

| Pass | 所属 | 职责 |
|---|---|---|
| `VPTOCombineReductions` | PTOAS | 固定 reduction 的最终 VPTO op 组合 |
| `VfSimUnrollPlanner` | PTOAS + VfSim | 枚举 ABCABC/AABBCC 候选、构造 `VfInfo`、写入 `pto.vfsim.unroll_factor` 并清理内部选择属性 |
| 后端 loop-unroll 阶段 | PTOAS | 区分手写 hint 与 costmodel 结果，并按后端策略执行 loop rewrite |
| `PTOValidateVPTOEmissionIR` | PTOAS | 检查进入 LLVM emitter 前的 IR 合法性 |

规划与应用分离后，cost model 可独立迭代，PTOAS 仍掌握所有结构性 IR 变换和合法性。

## 8. 开关与组合

VfSim 的编译可用性、运行时调用和链路选择分别由以下选项控制：

| 选项 | 类型 | 作用 |
|---|---|---|
| `PTO_ENABLE_VFSIM_COSTMODEL` | CMake | 编译并链接 VfSim C++ planner |
| `--enable-vfsim-costmodel-optimization` | CLI | 当前编译任务是否调用 VfSim planner |
| `--enable-vmi` | CLI | 请求 VMI 链路；最终以 `useVMIFusionPipeline` 为路由依据 |
| `--dump-vfsim-costmodel` | CLI | 打印候选、预测 cycle、选择结果和 skip reason |

运行行为：

| Cost model | VMI pipeline | 行为 |
|---:|---:|---|
| 关 | 任意 | 不调用 VfSim |
| 开 | 关 | 调用 legacy TileOp planner |
| 开 | 开 | 调用 VMI low-level unroll planner 和 apply pass |

本阶段开发和验证默认关闭现有 `VPTOScheduler`。该 pass 的实现和控制逻辑保持不变。

## 9. 诊断与失败策略

标准 dump 至少包含：

```text
function / vec_scope / loop location
candidate unroll factor
candidate unroll mode: ABCABC / AABBCC
candidate predicted cycle
selected mode and factor
skip reason
```

处理规则：

- 未显式启用 VfSim：完全保持原编译行为。
- 显式启用后遇到不支持的 op、动态 trip count 或缺少参数：warning 并跳过当前 loop。
- 参数数据库加载失败、输入 IR 契约损坏或 apply 校验失败：pass failure，不能静默继续。
- planner 未返回有效方案：不写 attr，保留原 IR。

## 10. 开发阶段

1. 增加 `VfSimPlannerMode` 解析，基于 cost model 开关和 `useVMIFusionPipeline` 选择唯一接口。
2. 增加四种路由组合测试，确认 legacy/VMI planner 不会在同一次编译中重复调用。
3. VMI loop fusion pass 给融合结果写入 `pto.vmi.loop_fused` 和唯一的 `pto.vmi.loop_fusion.id`。
4. 验证来源属性保留到 planner 接入点，并在 planner 完成后清理。
5. 在 `VPTOCombineReductions + CSE` 后插入 `VfSimUnrollPlanner`，只扫描带 `loop_fused` 的循环。
6. 依据 VPTO emitter 建立 `PhysicalMI/ZeroCostAlias/Structural/ScalarAddress/PredicateSetup/Composite` registry。
7. 完成 low-level VPTO IR 到 `VfInfo` 的 adapter。
8. 支持 softmax 主循环所需真实 MI op，以及 `vbitcast` canonical value 和依赖重连。
9. 为已支持 op/form、zero-cost alias 和 unknown-op skip 行为增加 adapter 单元测试。
10. 识别完整生命周期、loop-carried value 和 loop live-out value。
11. 分别构造 ABCABC/AABBCC 候选，包括 lane-local value 和必要的循环后归并。
12. 扫描 `1..min(8, trip_count)`；非整除 factor 同时建模主体和 tail，选择全局最优模式和 factor，只写回专用 costmodel attr。
13. 后续由 costmodel-aware 的后端阶段消费专用 attr，端到端对齐 VfSim 与 camodel 趋势。

本阶段验收以 planner 链路为准：能够识别目标 loop、输出候选预测时间并写回专用
unroll attr；实际消费与结构性 loop rewrite 由后续后端阶段完成。
