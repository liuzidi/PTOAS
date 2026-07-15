# ADR-0001: VMI VF Fusion 采用唯一 canonical 实现与融后 mem2reg

- 日期: 2026-07-15
- 状态: Proposed

## Context

PTOAS 已能通过 `--tile-lib-backend=ptodsl-vmi` 将 `PIPE_V` TileOp 展开为
独立可执行的 VMI 模板。每个模板包含一个主 `scf.for` 和完整的 VMI
load/compute/store，但不显式生成 `pto.vecscope`，因此多个连续 TileOp 展开后仍会
产生多个循环和中间 UB 往返。

该 backend 是组合式路由：`PIPE_V` 使用唯一 canonical VMI provider，其他 Pipe
继续使用现有 PTODSL TileLib daemon。它不会将非向量 TileOp 回退到 TileLang。

旧的 `FusionPlan` / `FusionRegionGen` 分析 Tile-native PTO IR，
`PTOLowLevelLoopFusion` 分析已经物理化的 VPTO/MI IR。二者都不能直接承担新的
VMI loop fusion。此前讨论过为同一 TileOp 提供多个 VMI schedule candidate，再由
region-aware cost model 选择并锁定实现；该方案会把实现版本选择提前引入本期，扩大
设计和验证范围。

## Decision

1. RFC 首期对每个 `(target, TileOp)` 只允许一个 canonical VMI 实现。
   该实现必须脱离融合独立正确执行；不存在 candidate 竞争、锁定或回退选择。
2. VMI Fusion 在 `ExpandTileOp`、`PTOInlineLibCall` 和 shape intrinsic folding 之后
   运行，直接分析真实的 VMI `scf.for`，不消费 `pto.fusion_region`。
3. 仅处理带 TileLib provenance 的 canonical VMI fusion unit。任意用户手写 VMI、
   非 canonical 模板或无法证明来源的循环默认不参与融合。
4. 融合采用保守、部分融合策略：只合并边界、循环域、依赖、alias、访问模式和 mask
   均可证明兼容的相邻循环；其余循环保持原样。
5. VMI mem2reg 必须在 loop fusion 之后运行。它只提升可证明同 location、同形状的
   VMI store-load，使融合后暴露的中间 UB 往返变为 SSA 直传。
6. 融合和 mem2reg 位于 VMI layout assignment 之前。物理 vreg layout、interleave、
   pack、post-update 和指令选择仍由现有 VMI semantic/layout pipeline 负责。

## Alternatives

### A. 在 Tile 层先选择并锁定多个 VMI candidate

暂不采用。它需要 schedule family、region-aware selection、代价模型和稳定的 fallback
协议，属于后续性能迭代，不是验证 VMI 融合基本闭环的前置条件。

### B. 直接复用 `FusionPlan` / `FusionRegionGen`

不采用为 VMI 实现本体。可以复用其 block-local DFG、活跃性和迭代域分析思路或抽取
通用 utility，但现有 pass 的输入、语义节点和 region op 都是 Tile-native PTO IR。

### C. 复用 `PTOLowLevelLoopFusion`

不采用。该 pass 面向 VPTO/MI 物理循环，运行位置过晚，会重新引入物理 layout、
predicate 和地址模式对融合分析的干扰。

## Consequences

### Pros

- 首期输入唯一、结果确定，便于建立 IR contract 和正确性测试。
- 单个 TileOp 在融合失败时仍能独立 lower，天然具备保守 fallback。
- loop fusion 与 UB store-load elimination 顺序正确。
- VMI 层保留逻辑 lane、mask 和 SSA 数据流，避免在 MI 层恢复高层语义。

### Cons / Risks

- canonical 实现不一定是每个固定 Shape 的最优实现。
- trip count 不同的 Reduce、Broadcast、Elementwise 循环不会因为存在其他可选 schedule
  而被强行融合；FA/Softmax 的完整深融合不是本 ADR 首期验收目标。
- 当前 VMI load/store 使用线性 offset，alias 分析必须保守规范化 storage root 和
  index expression；无法证明时必须拒绝融合或提升。
- `PTOInlineLibCall` 需要保留 TileLib provenance，否则无法可靠区分模板代码与用户
  手写 VMI。

## Follow-ups

- 实现 VMI fusion-unit provenance、识别、规划、loop fusion 和 mem2reg passes；复用
  现有 late `PTOInferVPTOVecScope` 统一生成物理 VPTO vecscope。
- 完成 elementwise 链的正向和负向 lit tests。
- 在基本闭环稳定后，再独立评审多 candidate、Reduce schedule、cost model、unroll
  和算法专项深融合。
