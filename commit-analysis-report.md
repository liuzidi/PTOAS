# VMI Fusion 系列提交分析报告

> 分析对象：当前分支 `rebuild/vmi-fusion-files` 最近 5 个 commit
> 生成日期：2026-08-15

---

## 一、总览

这 5 个 commit 构成 PTOAS 编译器引入 **VMI（Vector Machine IR）Loop Fusion 流水线** 的完整工作链，从核心算法、TileLib 候选模板、内存安全屏障、问题修复到基础设施接线，逐层落地。按时间顺序（旧→新）：

| 序号 | Commit | 标题 | 规模 | 性质 |
|:---:|--------|------|------|------|
| 1 | `fe74fcc54` | feat(vmi): add VMI loop fusion pass and load/store elision | 42 文件 +10143/-1799 | 核心算法 |
| 2 | `6be71e73b` | feat(vmi): add VMI tilelib candidates and ptodsl lowering support | 259 文件 +11949/-229 | 模板生态 |
| 3 | `17f745135` | feat(vpto): add vecscope memory barrier pass | 31 文件 +4023 | 内存安全 |
| 4 | `6d4ea1903` | fix(vpto): fix VPTO lowering and emit issues exposed by VMI fusion | 86 文件 +3160/-262 | 问题修复 |
| 5 | `b807f92c8` | infra: add VMI pass registration, CLI options, and shared utilities | 11 文件 +1212/-64 | 基础设施 |

**整体叙事线**：先用 PTODSL VMI TileLib 把 `PIPE_V` TileOp 展开为唯一 canonical VMI 实现（commit 2）→ 在 VMI 层做同 header `scf.for` 保守融合 + 融后 mem2reg 消除 UB 往返（commit 1）→ 为融合后更复杂的 SSA 数据流补充 VecScope 域内 UB 别名屏障（commit 3）→ 修复融合暴露的 lowering/emit 问题（commit 4）→ 注册 pass、扩展 CLI、提取共享 Utils（commit 5）。

**编译流水线**（commit 1 设计文档 §9）：
```
ExpandTileOp → PTOInlineLibCall → FoldTileBufIntrinsics(shape-only)
  → VMI region-local 合法性分析
  → VMI Loop Fusion → VMI Mem2Reg(load/store elision)
  → VMI layout assignment → VMIToVPTO → PTOInferVPTOVecScope
```

---

## 二、Commit 逐个分析

### Commit 1 — `fe74fcc54`：VMI Loop Fusion + Load/Store Elision

**核心目标**：在 VMI 层建立"复用 Tile 层 `fusion_region` 边界 + 同 header `scf.for` 保守逐对合并 + 跨迭代 UB 守卫 + 融后 mem2reg 消除 UB 往返"的**最小正确性闭环**。

#### 2.1 VMI Loop Fusion Pass（`PTOVmiLoopFusion.cpp`，1076 行）

输入约束：只处理 `pto.fusion_region` 体内直接嵌套的单层 `scf.for`（VMI tile-library compute 恒为单层 inner VL 循环）。

算法核心：
1. **同 header 判定**：两个 `scf.for` 可融合前提是 lower/upper/step **结构等价**（递归比较 defining op 的 name/operands/attributes/result types），且剥离 provenance 属性后语义属性相同。
2. **between-ops 依赖分量分析**：两循环间的操作按 SSA def-use 或同 UB store→load 连成依赖连通分量，每个分量必须整体移动——能 **hoist 到融合 for 之上**（loop-invariant）或 **sink 到融合 for 之下**（结果不被任何成员读）；都不能移动的分量阻塞融合（如 tmuls 链：读 ColMax 的 final UB 不能 hoist，其 store 被 ColExpand-sub 循环读不能 sink）。
3. **融合构造**：拼接各成员 init args（reduce carry）与 yield 操作数，body-builder 原地建 yield。
4. **跨迭代 UB 守卫（首版合法性关键）**：候选循环加入 run 的条件是它与 run 交换的每个 UB 都是**同迭代传递**——producer 写 offset `f(i)`、consumer 读 `g(i)`，融合后 consumer 在迭代 i 读到 producer 在迭代 i 写的值当且仅当两个 offset 都依赖 IV、都是单射仿射、且结构等价。因此 stencil（`A[i+1]`）、固定 offset（`UB[0]`）、`i%2` 全部被 block。
5. **保守原则**：不重新划分或扩大 `fusion_region` 边界，只在 region 内做 fusion 与 mem2reg；任何分析失败保持原独立实现。

#### 2.2 Load/Store Elision / VMI Mem2Reg（`PTOVmiLoadStoreElision.cpp`，1094 行）

运行在 `PTOVmiLoopFusion + CSE` 之后、`VMILowerUnifiedToLegacy` 之前。融合后跨 for 的 UB 往返已变成同 block 内直线 store-load 对，本 pass 消除它们。

**两遍扫描**：
- **Pass 1（build，正向）**：建 content-version 表，标记 forward 目标——store→load elision（读 lane 集被前导 store 写集覆盖）、vload→vload dedup、dead-store-erase。merge store 只失效其写 lane 区间内前导条目。
- **Pass 2（eliminate，反向）**：替换 load 的 uses 为记录的 source value，逆序擦除 dead load/store（保证后转发 op 消费的 value 先被替换再擦除）。

**保守边界**：只建模连续、单结果 vload / 单值 vstore，无 stride/block_stride/repeat_stride/group/dist_mode；base 必须解析为编译期或仿射 UB 身份（`castptr→memref→pointer_cast` 链到常量地址，或 `addi(muli(%iv,c),b)` 仿射地址）；运行时指针不能可靠匹配；透明性用封闭策略（带 region op / func.call / vload/vstore / 显式 sync 永不透明）；read-lane 集仅从封闭白名单 consumer 推断。

#### 2.3 两个编译期分析支撑

| 分析 | 头文件 | 作用 |
|------|--------|------|
| **VmiMemoryLocation** | `VmiMemoryLocation.h`（57 行）+ 实现 289 行 | 为 fusion 和 elision 提供 VMI UB 访存的编译期 location/alias 分析：`VmiStorageRoot`（编译期 storage root）、`mayAliasVmiStorageRoot`（静态已知 view 是否可能重叠）、`VmiAccessLocation`（{root, elementOffset, accessBytes}，同 element-index 域比较） |
| **TileShapeStateAnalysis** | `TileShapeStateAnalysis.h`（49 行）+ 实现 242 行 | 为 candidate 选择/expansion/fusion 提供 tile 形状事实：`TileShapeState`（Full/Partial/Unknown）、`resolveStaticTileValidShape`（用 Dominance 解析支配 useOp 的最新静态已知 valid_shape 更新）、`hasStaticFullTileValidShape`（判断是否满足 1VL inner contract） |

#### 2.4 VMIToVPTO 大规模重构（5186 行，体量最大）

- 风格统一：大量单行 `if` 补齐为大括号块。
- 新增 `isVMIPackedFloatCarrierType`：判断 packed float 载体类型（`PTOHiFloat8x2`/`PTOFloat4Packed`/`PTObf16x2`），配合 fusion 后更宽 logical vreg 的物化与打包类型处理。
- 据 ADR-0002，重构动机是为 fusion 产生的更复杂 vreg 生命周期、packed 类型物化、以及 ADR-0002 要求的分层资源控制做铺垫。

#### 2.5 测试覆盖

**lit 测试（11 个）**：正向（双/三 elementwise 合并、mem2reg 消除、1VL RowMax→Broadcast→Exp→RowSum 同 row loop、动态 upper bound 同 SSA 可融合）、负向（stencil/固定 offset/`i%2` 跨迭代 block、中间 sync/call 不融合、may-alias/WAW/WAR 不证明不融合、group/stride/post_update 不 elide）、F3 邻接分组边界、错误 strategy 拒绝。

**端到端 case（2 套 FA-Softmax）**：`fa-softmax-dn-init` 与 `fa-softmax-dn-init-rowplusone`（含 kernel.pto + launch.cpp + main.cpp + golden.py + compare.py + ptoas.flags），后者测试 BR<VL 的 `[1,BR]` compact state reshape 边界。

---

### Commit 2 — `6be71e73b`：VMI TileLib Candidates + PTODSL Lowering

**核心目标**：引入 **PTODSL VMI TileLib backend**，给 TileOp 的 Expand 阶段增加 `--tile-lib-backend=ptodsl-vmi` 选项，让 `PIPE_V` 走 PTODSL 自己的 VMI 候选模板。这是从"旧 VPTO/MI loop-fusion pipeline"切换到"新 VMI Fusion pipeline"的前置基础设施。

#### 2.6 tilelib candidates 机制

- 模板注册时可打 `tags`（特别是 `vmi` tag）。Python 侧 `registry.legal_candidates` 默认 `include_hidden=False`，把带 `vmi` tag 的候选从普通查询隐藏，只有 VMI provider 显式请求时可见——VMI 候选和普通候选即使同名也不混选。
- **`SelectTemplateCandidate.cpp`（新增 377 行 C++ pass）**：C++ 侧候选裁决，从 TileOp 读 `candidates` 属性选出一个写入 `pto.tilelib.selected_candidate`；估算 VMI 向量资源（A5 物理向量 256 字节为粒度）；判定 VMI fusion boundary（硬边界算子 `tload/tstore/tmatmul*/tmrgsort/...` 和非 `PIPE_V` 一律 fallback）；检测 materialization-sensitive subview 时拒绝 VMI candidate。
- RFC 要求每个 `(target, PIPE_V TileOp)` 恰好一个 canonical VMI 实现，不做 priority 选择/candidate locking/cost model。

#### 2.7 新增算子模板

`ptodsl/ptodsl/tilelib/templates/a5/` 下新增/扩展覆盖静态 Softmax compute harness 所需算子：
- **elementwise**：`tadd/tadds/tsub/tmul/tmuls/tmax/tmaxs/tmins/tdivs/tmov/texp`
- **row reduce**：`trowmax/trowsum/trowexpandsub`
- **col reduce**：`tcolmax/tcolmin/tcolsum/tcolexpand*`
- **convert**：`tcvt`
- 其他：`tdiv/trecip/tneg/tabs/tsqrt/trsqrt/texpand/tgather/tload/tstore/tmov_nd2nz`

约束：dense f32 Tile，physical inner 固定 64 lanes，每行一个 logical block；RowReduce 单 row loop + 单输入 block，`[rows,1]` compact result。

#### 2.8 `_vmi_common.py`（2583 行）的角色

所有 per-op VMI 候选共享的**公共发射/合法性基础设施**：dtype 适配、vreg/mask 包装器、复用 `_tile_template_tracing` 的 CanonicalBlockMap/Coordinate/VectorValue/MaskValue 体系、通用约束函数（如 `row_reduce_vmi_constraint`）、`canonical_vmi_template` 装饰器和 `VMI_TILELIB_REGISTRY`。使每个 `t*.py` 只需几十行声明式候选。

#### 2.9 tilelib 框架与 IR 修改

- `registry.py`：`include_hidden` 机制 + 显式指定 hidden 候选时也能选中。
- `metadata.py`/`decorator.py`/`constraints.py`：支持 `tags` 字段。
- `_tile_template_tracing.py`（+876）：扩展 TileSpec（`b_layout/valid_shape/compact_mode`）、支持 `ir_level="vpto"|"vmi"` 模板选择、fixed-shape VMI logical-block helpers。
- `_vmi_namespace.py`（+123）：`pto.vmi.vstore` 支持 `updated_base` 可选结果。
- `VMIOps.td`：`VMIFPToSIOp`/`VMISIToFPOp` 增加 `rounding` 可选属性；`VMIvStoreOp`/`VMIStrideStoreOp` 增加可选 `updated_base` 结果（post-update address state，为后续 Mem2Reg 准备）。

#### 2.10 测试 sample 大量新增的含义

`test/samples/` 为每个算子成对新增 `*_compare.py`/`*_golden.py`，覆盖几十个算子。这说明项目在为 VMI lowering 建立**端到端数值正确性回归基线**——"算子级 golden + 全流程 compare"的双层验证网，配合 lit IR 快照和 Python 单测，构成三层验证。

---

### Commit 3 — `17f745135`：VecScope Memory Barrier Pass

**核心目标**：在 PTO 的 `__VEC_SCOPE__`（向量执行域）内，硬件**不跟踪 UB 地址在 reg↔UB 访问之间的别名**。当 UB 地址在向量 load/store 间重叠或别名时，必须显式插入 `pto.mem_bar`，否则 load 看到陈旧数据、store 被错误重排。本 pass 专门负责域内 UB 别名 hazard 的 barrier 插入。

#### 2.11 五层架构（职责严格分离）

| 层 | 文件 | 行数 | 职责 |
|---|------|------|------|
| **IR** | `VecScopeMemBarIR.h/.cpp` | — | 底层基础设施：调度树节点（Sequence/Loop/Access/ExistingBarrier）、loop info、UB-backed 谓词、`getStoredValues`/`getLoadedValues`、`valueDependsOn` |
| **MemoryFootprint** | `VecScopeMemoryFootprint.h/.cpp` | 744 | 内存足迹建模：root kind（Absolute/ProvenAllocation/Symbolic/Unknown）、仿射字节偏移 `AffineByteExpr`、同迭代别名判定（NoAlias/MayAlias/MustOrPartialAlias） |
| **Analysis** | `VecScopeMemBarAnalysis.h/.cpp` | 1191 | 核心：构建调度树、用 **Presburger IntegerPolyhedron** 枚举 RAW/WAW/WAR hazard，按 scope 分类（SameIteration/InnerLoopCarried/OuterLoopCarried）带 IterationDistance；只读不改 IR |
| **Placement** | `VecScopeMemBarPlacement.h/.cpp` | 551 | 求解：为每个 hazard 选锚点（BeforeOperation/BeforeLoop/AfterLoop/BeforeLoopTerminator），同锚点同 kind 合并、多 kind 折叠为 VV_ALL，已有 barrier 覆盖检查，**传递式 WAW 冗余消除**（SSA use-def 反向可达）；输出确定性 Plan，不改 IR |
| **Codegen** | `VecScopeMemBarCodegen.h/.cpp` | 74 | 落地：校验锚点、按词法序插入 `pto.mem_bar`、多 directed kind 规范化为 VV_ALL；唯一改 IR 的层 |

数据流：IR → Footprint → Analysis → Placement → Codegen。前四层只读确定性，只有 Codegen 用 IRRewriter 改 IR。

#### 2.12 依赖类型

- **RAW**（Store→Load）= `VST_VLD`
- **WAW**（Store→Store）= `VST_VST`
- **WAR**（Load→Store）= `VLD_VST` — kind 已定义，但**本版本不对同迭代 WAR 生成 barrier**（由 `war_not_generated` 测试固化）；loop-carried WAR 有专门测试
- **RAR** 忽略

#### 2.13 两个 pass 的区别

| | `--enable-vecscope-mem-bar`（MemBar） | `--enable-vecscope-mem-bar-all`（MemBarAll） |
|---|---|---|
| 默认 | 开 | 关 |
| 流程 | 完整 Analysis→Placement→Codegen + 函数级跨 scope 分析 | **完全跳过分析**，每个 UB-backed 向量访存前各插一条 `VV_ALL` |
| 用途 | 正式编译 | 同步正确性**调试**，保守诊断模式 |

#### 2.14 测试覆盖（18 个场景）

同迭代 RAW/WAW、循环携带（RAW+WAW 共享 latch 折叠、WAR、外层携带跨层）、跨 scope、NoAlias 精确跳过、subview 行不重叠、多 kind、传递式 WAW 冗余消除、已有 barrier 幂等、WAR 不生成（版本边界）、动态上界保守、vsstb 保守、uvld 展开、scf.if 控制流、RoPE KV cache 端到端、保守调试模式对比。

---

### Commit 4 — `6d4ea1903`：修复 VMI Fusion 暴露的 Lowering/Emit 问题

**核心目标**：修复 VMI fusion 落地后暴露出的 VPTO lowering 和 emit 问题，并新增 30+ 测试。86 文件 +3160/-262。

#### 2.15 各 pass 适配性修改

| Pass | 关键修改 |
|------|----------|
| **ExpandTileOp.cpp** | 引入 VMI fusion 相关属性常量；改为消费 `SelectTemplateCandidate` 已选实现（不再自己做候选选择）；`OperandTypeInfo` 新增 `compactMode`（Normal/RowPlusOne）修复 RowPlusOne ND2NZ compact_mode 丢失；新增 tcvt 饱和模式传递 |
| **InsertTemplateAttributes.cpp** | 候选元数据扩展（`tags`/`resource_scope`/`resource_vector_values`/`resource_chunk_streaming`）；`getStaticIntFromValue` 支持递归折叠 `arith.addi/subi/muli`（修复 rank-3 ND 静态 stride 折叠）；动态 `valid_shape` 调用 `resolveStaticTileValidShape` |
| **PTOInferVPTOVecScope.cpp** | 大幅扩展"可克隆共享生产者"识别（mask 逻辑 `pand/por/pnot/pintlv`、标量广播 `vbr`/`vdup`、索引向量 `vci/vand` 仅供 `vgather2/vscatter` 消费）；新增 `pruneExternallyConsumedCloneableProducers` 剪枝（避免被 scope 外消费的克隆生产者误移入 inferred vecscope） |
| **PTOMaterializeTileHandles.cpp** | 新增 `canRetypeValueUsesToTile`/`canMaterializeYieldOperandUse`：支持跨 region 边界的 tile handle 物化；`computeExplicitAddress` 新增 `alloc_tile` 的显式 `$addr` 暴露 |
| **PTOViewToMemref.cpp** | 新增 `foldAddPtrIntoVectorMemoryOp`（把 `pto.addptr` 链折叠进 vlds/vsts）；`lowerSubViewOps` 按嵌套深度降序处理 + 增量同步 fusion region 结果类型 |
| **VPTOLLVMEmitter / VPTOCANN900LLVMEmitter** | 修复 `UnrealizedConversionCastOp` lowering（用 `ExtractAlignedPointerAsIndexOp` + `IndexCastUIOp` + `IntToPtrOp` 显式取地址）；新增 `ConvertPtoTGetValOp`/`ConvertPtoTSetValOp`；`buildCopyMatrixCcToUbCallee` 增加 BF16/i32；调用 Utils 的 normalize/legalize/cleanup/lowerA5UnifiedL2LPipeOps |
| **VPTOSplitCVModule.cpp** | 新增 `getFunctionKernelKind`/`cloneFuncIntoModule`：支持把顶层 kernel-kind 函数克隆进对应 kind child module |
| **PTO.cpp** | `sync.set` 校验放宽到允许 `PIPE_V`/`PIPE_MTE1`；`MemBarOp` 实现 `MemoryEffectsOpInterface`（Read+Write，否则被 effect-free 分析误判）；`isLocallyBoundTileSource` 穿透 `SubViewOp`/`FusionRegionOp` |
| **FoldTileBufIntrinsics.cpp** | 先 unwrap bridging casts 再查 fusion region 结果（修复 RowPlusOne bridging cast 击败 region-yield→alloc 恢复） |
| **PTOInstantiateAndInlineOpLib.cpp** | `copyTileLibSelectionAttrs`：inline 时复制 VMI/tilelib 选择属性到 clone |
| **PTOOps.td** | `TCvtOp.sat_mode` 从 `DefaultValuedAttr(OFF)` 改为 `OptionalAttr`（区分"未指定"与"显式 OFF"） |
| **VPTOOps.td** | `MemBarOp` 声明 `MemoryEffectsOpInterface` |

#### 2.16 TileFusion 相关修改

- **FusionOpSemantics.cpp**：新增 `ColBroadcastBinary`（tcolexpandsub/add/mul/div）和 `Convert`（tcvt）compute family；扩展 Elementwise 白名单；新增 `isFusionTransparentScaffold`（alloc_tile/sub_view/make_tensor_view + 纯 index/arith plumbing 视为透明脚手架）；`tileOutputs` 为空时改为 `HardBoundary`；识别 `pto.vmi.fusion.boundary` 属性。
- **FusionAnalysis.cpp**：`Convert` 按 Elementwise 处理形状约束；新增 `ColBroadcastBinary` 形状约束（col_values 行固定为 1）。
- **PTOFusionPlan.cpp**：新增 `VMIUBDisjointStrategyEngine`——F3 邻接策略，按"两个可规划 compute node 之间是否有真实不可规划 op"分组。
- **PTOOpScheduling.cpp**：用 `isFusionTransparentScaffold` 替代单独判 `AllocTileOp`；local boundary 仅在带 `pto.vmi.fusion.boundary` 时才阻挡调度。

#### 2.17 测试基础设施

- `fake_bisheng.sh`/`fake_ld_lld.sh`：假编译器/链接器，记录 argv 到日志并生成空输出文件，使 lit 测试无需真实 CANN toolchain 即可断言传递给 Bisheng 的参数。
- `lit.cfg.py`：新增 `%python_executable`/`%mlir_python_root` 替换；设置 `MLIR_PYTHON_ROOT` 环境变量供 PTODSL Python daemon 使用。

#### 2.18 新增测试覆盖

VecScope 推断共享生产者（mask/pintlv/vdup/vbr/gather 索引/sink/控制流/负向拒绝，~10 个）、候选选择（prefer-vmi vs ordinary-only/hard boundary/resource guard）、L2L FIFO、Bisheng VF 参数传递、ExpandTileOp（rank3 ND stride/tmov nd2nz/tpop addr fold）、materialize tile handles fusion region subview、vlds/vsts addptr fallback、split top-level kernel-kind funcs 等。

---

### Commit 5 — `b807f92c8`：Pass 注册 + CLI + 共享 Utils

**核心目标**：纯基础设施，注册新 pass、扩展 CLI、提取共享 Utils。11 文件 +1212/-64。

#### 2.19 Pass 注册（Passes.h + Passes.td）

新增 5 个 pass 工厂声明与定义：

| Pass | 关键选项 | 职责 |
|------|----------|------|
| **SelectTemplateCandidate** | `prefer-vmi`/`ordinary-only`、`max-candidate-vector-bytes`(默认 6144)、`emit-resource-remarks` | 候选选择策略，记录所选实现和 VMI fusion boundary 供 ExpandTileOp 消费 |
| **PTOVmiLoopFusion** | — | 在 `fusion_region` 内合并同 header scf.for，替代 legacy `PTOLowLevelLoopFusion` |
| **PTOVmiLoadStoreElision** | — | 两遍扫描 + lane interval 跟踪，前向替换 vload、消除死 vstore；须在 LoopFusion + CSE 之后、VMILowerUnifiedToLegacy 之前 |
| **PTOInsertVecScopeMemBar** | — | 基于依赖分析在 vecscope 内存冒险处插入 `mem_bar` |
| **PTOInsertVecScopeMemBarAll** | — | debug 模式，每个 UB 向量访存前插 VV_ALL |

`FusionPlan` pass 新增 `fusion-strategy` 选项（`conservative-dag-greedy` 默认 / `vmi-ub-disjoint`），非法值直接失败而非静默回退。`VMILowerUnifiedToLegacy` 描述更新：全掩码算术标量 op（vadds/vmuls/vmaxs/vmins）改为保留供直接 VMIToVPTO lowering，仅部分掩码才 lower。

#### 2.20 Utils.cpp / Utils.h（812 行共享工具）

| 函数 | 作用 |
|------|------|
| `normalizePTOAddressSpacesForLLVM` | 把 PTO `AddressSpaceAttr` 转为官方 memref-to-LLVM 期望的 i64 memory space |
| `legalizeIndexUnrealizedCasts` | 将 i64↔index 的 `UnrealizedConversionCastOp` 替换为 `arith::IndexCastOp`（A5 index 为 64 位） |
| `cleanupPTOArtifactsAfterLLVMLowering` | 清理 use_empty 的 `AllocTileOp` 和冗余 `UnrealizedConversionCastOp` |
| `lowerA5UnifiedL2LPipeOpsForLLVM`（约 500 行） | A5 统一 L2L pipe 的 LLVM lowering：为每个 `initialize_l2l_pipe` 分配 producer/consumer 索引（LLVM alloca i64），发射构造/析构 sync（C2V 用 PIPE_V/PIPE_FIX，V2C 用 PIPE_MTE1/PIPE_MTE3，支持 dir_mask 1/2/3）；`lowerPush`（环形地址 + 满槽周期性 `sync.wait` + ACC→VEC 用 `CopyMatrixCcToUbOp`、VEC→MAT 用 `CopyUbufToCbufOp`）；`lowerPop`（`sync.wait` + 环形地址 + `replaceDeclaredTilePointerUsers`）；`lowerFree`（周期性 `sync.set`）；校验 a5 + flag_base + nosplit + dir_mask + kernel_kind |

#### 2.21 ptoas CLI 新增选项

| 选项 | 默认 | 作用 |
|------|------|------|
| `--ptodsl-python-exe` | `PTOAS_DEFAULT_PTODSL_PYTHON_EXE` | 指定匹配 PTODSL PTO 绑定的 Python 可执行文件 |
| `--enable-vmi` | on | VMI 流水线总开关 |
| `--enable-vmi-loop-fusion` | on | VMI loop fusion 开关 |
| `--enable-vmi-load-store-elision` | on | VMI load/store elision 开关 |
| `--disable-bisheng-vf-fusion` | off | 禁用 Bisheng VF/loop-fusion/load-store-elimination |
| `--enable-vecscope-mem-bar` | on | VecScope membar 开关 |
| `--enable-vecscope-mem-bar-all` | off | VecScope membar 调试模式开关 |

管道编排：`useVMIFusionPipeline = enableVMI && enableOpFusion && enableA5VPTOFusionPath`；VMI 启用时主 PM 在 `InsertTemplateAttributes` 后插入 `SelectTemplateCandidate(prefer-vmi)`、`FusionPlan` 用 `vmi-ub-disjoint` 策略；`appendVMISemanticPipeline` 新增可选的 loop fusion + load/store elision 段；VMI 路径关闭 legacy fusion lifecycle；tile-lib-backend 判断从 `== PTODSL` 改为 `!= TileLang`（PTODSL 成为默认）。

#### 2.22 ObjectEmission 修改

新增 `ObjectEmissionOptions`（仅 `disableBishengVFFusion` bool）。当为真：强制 `-cce-vf-auto-sync=off` + 一组 `-cce-vf-enable-*=false` 禁用 Bisheng 的 VF fusion/loop-extender/loop-fusion/ldst-elimination/ub-dead-st-elimination/ifelse-extender（避免 Bisheng 二次优化 PTOAS VMI 已处理的内容）。`emitVPTOBackendResult` 在 `useVMIFusionPipeline` 或显式 `--disable-bisheng-vf-fusion` 时置 `disableBishengVFFusion=true`。

---

## 三、关键设计决策与风险

### 3.1 保守正确性优先

贯穿所有 commit 的核心原则：**任何分析失败均保持原独立实现，不改变程序语义**。
- Fusion：不重新划分 `fusion_region` 边界，只在 region 内做融合；跨迭代 UB 不能证明同迭代传递则 block。
- Elision：只处理连续单结果 vload/单值 vstore；运行时指针/非仿射 base 不 forward；may-alias 不消除。
- MemBar：不可建模的依赖记为 `UnknownDependence`；动态上界保守插 barrier。

### 3.2 分层职责分离

VecScope MemBar pass 是典型：前四层（IR/Footprint/Analysis/Placement）只读确定性，只有 Codegen 改 IR。利于正确性验证和回归测试。

### 3.3 ADR-0002 暴露的风险

ADR-0002 记录：A5 验证表明仅限制 loop fusion 不能解决 vector-function stack overflow——14 个历史用例 `emit-vpto` 全部成功且 VPTO 不残留 VMI，但 **A5 device object 成功 0/14**，后端 stack object 8480B–37152B、Vector Slots 33–145。诊断用例关掉 VMI loop fusion 和 load/store elision 后仍超限，而同输入用 ordinary PTODSL/VPTO candidate 能生成 A5 object。结论是**单个 VMI candidate、wide logical vreg 物化、vecscope 调度及后续后端 lowering 已产生过高压力**，需要 ADR-0002 的分层资源控制（candidate resource guard → fusion/forwarding resource planning → backend feedback）。VMIToVPTO 的大规模重构和 `isVMIPackedFloatCarrierType` 新增正是为此铺路。

### 3.4 版本边界

- WAR 依赖：kind 与分支已定义，但本版本不对同迭代 WAR 生成 barrier（由测试 `war_not_generated` 固化）。
- VMI tilelib 尚未覆盖：动态 row/column、tail mask、row-wise 归一化、全部 Convert/高精度除法变体。
- `--enable-op-fusion` 在 VMI Fusion pipeline 接入前会被 provider 拒绝。

---

## 四、文件影响统计

| Commit | 文件数 | 增 | 删 | 净 |
|:---:|:---:|---:|---:|---:|
| fe74fcc54 | 42 | +10143 | -1799 | +8344 |
| 6be71e73b | 259 | +11949 | -229 | +11720 |
| 17f745135 | 31 | +4023 | 0 | +4023 |
| 6d4ea1903 | 86 | +3160 | -262 | +2898 |
| b807f92c8 | 11 | +1212 | -64 | +1148 |
| **合计** | **429** | **+30487** | **-2354** | **+28133** |

> 注：文件数含跨 commit 重复修改的同一文件，去重后实际改动文件数更少。

---

## 五、一句话总结

这 5 个 commit 在 PTOAS 编译器中建立了完整的 VMI Loop Fusion 流水线：用 PTODSL VMI TileLib 把 `PIPE_V` TileOp 展开为唯一 canonical VMI 实现（C2）→ 在 VMI 层保守融合同 header `scf.for` 并用 mem2reg 消除 UB 往返（C1）→ 补充 VecScope 域内 UB 别名屏障保证内存安全（C3）→ 修复融合暴露的 lowering/emit 问题（C4）→ 注册 pass、扩展 CLI、提取共享 Utils 并禁用 Bisheng 二次优化（C5）。全程以保守正确性为首要原则，任何分析失败均回退到原独立实现。
