// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/Transforms/TileFusion/FusionAnalysis.h"
#include "PTO/Transforms/TileFusion/FusionOpSemantics.h"

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSwitch.h"

#include <algorithm>

namespace mlir {
namespace pto {
// Passes.h (included above) pulls in the global GEN_PASS_DECL block, which
// defines GEN_PASS_DECL_FUSIONPLAN and leaves it set.  Undef it before
// re-including the .inc for GEN_PASS_DEF so the options struct is not defined
// twice.
#undef GEN_PASS_DECL_FUSIONPLAN
#define GEN_PASS_DEF_FUSIONPLAN
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;

namespace {

static constexpr llvm::StringLiteral kFusionGroupIdAttr =
    "pto.fusion.group_id";
static constexpr llvm::StringLiteral kFusionOrderAttr = "pto.fusion.order";
static constexpr llvm::StringLiteral kVmiFusionBoundaryAttr =
    "pto.vmi.fusion.boundary";

struct PlannedFusionGroup {
  SmallVector<const pto::FusionComputeNode *, 8> members;
};

struct PlanningContext {
  const pto::FusionBlockAnalysis &blockAnalysis;
};

struct PlanningCost {
  int64_t dependencyBenefit = 0;
  int64_t loopMergeBenefit = 0;
  int64_t liveTilePenalty = 0;
  int64_t vfParameterPenalty = 0;
  bool rejectedForDynamicShape = false;

  int64_t total() const {
    return dependencyBenefit + loopMergeBenefit - liveTilePenalty -
           vfParameterPenalty;
  }
};

struct PlanningDecision {
  bool accept = false;
  PlanningCost cost;
};

static bool isCurrentlyPlannableOp(StringRef opName) {
  return llvm::StringSwitch<bool>(opName)
      .Cases("tmul", "tdiv", "tadd", "tsub", "tmax", "tmin", true)
      .Cases("tmuls", "tdivs", "tadds", "tsubs", "tmaxs", "tmins", true)
      .Cases("texp", "tabs", "tneg", "trecip", "tsqrt", "trsqrt", true)
      .Case("tmov", true)
      .Case("texpands", true)
      .Cases("trowexpandsub", "trowexpandmul", "trowexpanddiv", true)
      .Cases("tcolexpandsub", "tcolexpandadd", "tcolexpandmul", "tcolexpanddiv",
             true)
      .Cases("trowsum", "trowmax", "trowmin", true)
      .Cases("tcolsum", "tcolmax", "tcolmin", true)
      .Case("tcvt", true)
      .Default(false);
}

static bool isProvenIterationDomain(
    const pto::FusionBlockAnalysis &blockAnalysis,
    const pto::FusionComputeNode &node) {
  if (node.iterationDomainClass >= blockAnalysis.iterationDomainClasses.size())
    return false;
  return blockAnalysis.iterationDomainClasses[node.iterationDomainClass]
             .info.proof == pto::IterationDomainProof::Proven;
}

static bool hasHardBoundaryBetween(const pto::FusionComputeNode &a,
                                   const pto::FusionComputeNode &b) {
  const pto::FusionComputeNode &earlier =
      a.blockOrder < b.blockOrder ? a : b;
  const pto::FusionComputeNode &later =
      a.blockOrder < b.blockOrder ? b : a;

  Operation *cursor = earlier.op->getNextNode();
  while (cursor && cursor != later.op) {
    if (cursor->hasTrait<OpTrait::IsTerminator>() ||
        !cursor->getRegions().empty() || isa<CallOpInterface>(cursor))
      return true;
    cursor = cursor->getNextNode();
  }
  return false;
}

static bool hasHardBoundaryToGroup(
    ArrayRef<const pto::FusionComputeNode *> group,
    const pto::FusionComputeNode &candidate) {
  for (const pto::FusionComputeNode *member : group)
    if (hasHardBoundaryBetween(*member, candidate))
      return true;
  return false;
}

static bool dependsOnPreviousNode(
    const pto::FusionBlockAnalysis &blockAnalysis,
    const pto::FusionComputeNode &previous,
    const pto::FusionComputeNode &current) {
  for (unsigned edgeId : current.incomingEdges) {
    if (edgeId >= blockAnalysis.edges.size())
      continue;
    if (blockAnalysis.edges[edgeId].producerNode == previous.id)
      return true;
  }

  for (Value output : previous.semantics.tileOutputs)
    if (llvm::is_contained(current.semantics.tileInputs, output))
      return true;

  return false;
}

static SmallVector<const pto::FusionComputeNode *, 8>
buildStableInGroupOrder(ArrayRef<const pto::FusionComputeNode *> members) {
  SmallVector<const pto::FusionComputeNode *, 8> ordered(members.begin(),
                                                         members.end());
  llvm::stable_sort(ordered, [](const pto::FusionComputeNode *lhs,
                                const pto::FusionComputeNode *rhs) {
    if (lhs->blockOrder != rhs->blockOrder)
      return lhs->blockOrder < rhs->blockOrder;
    return lhs->id < rhs->id;
  });
  return ordered;
}

static void assignStableGroupMetadata(ArrayRef<PlannedFusionGroup> groups,
                                      MLIRContext *ctx,
                                      int64_t &nextGroupId) {
  SmallVector<const PlannedFusionGroup *, 8> orderedGroups;
  orderedGroups.reserve(groups.size());
  for (const PlannedFusionGroup &group : groups)
    orderedGroups.push_back(&group);

  llvm::stable_sort(orderedGroups, [](const PlannedFusionGroup *lhs,
                                      const PlannedFusionGroup *rhs) {
    const pto::FusionComputeNode *lhsFirst = lhs->members.front();
    const pto::FusionComputeNode *rhsFirst = rhs->members.front();
    if (lhsFirst->blockOrder != rhsFirst->blockOrder)
      return lhsFirst->blockOrder < rhsFirst->blockOrder;
    return lhsFirst->id < rhsFirst->id;
  });

  for (const PlannedFusionGroup *group : orderedGroups) {
    const int64_t groupId = nextGroupId++;
    SmallVector<const pto::FusionComputeNode *, 8> stableOrder =
        buildStableInGroupOrder(group->members);
    DenseSet<Operation *> memberOps;
    for (const pto::FusionComputeNode *node : stableOrder)
      memberOps.insert(node->op);

    // Local fallbacks and structural scaffold are part of the same loose
    // FusionRegion, but are not VMI compute nodes. Give them ordering metadata
    // when they sit between selected compute members so RegionGen preserves
    // their SSA definitions in the contiguous span. PTOVmiLoopFusion still
    // stops at a non-VMI fallback loop.
    Operation *first = stableOrder.front()->op;
    Operation *last = stableOrder.back()->op;
    int64_t order = 0;
    for (Operation *op = first; op; op = op->getNextNode()) {
      bool include = memberOps.contains(op);
      if (!include)
        include = pto::isFusionTransparentScaffold(op);
      if (!include) {
        FailureOr<pto::FusionOpSemantics> semanticsOr =
            pto::getFusionOpSemantics(op);
        include = succeeded(semanticsOr) &&
                  semanticsOr->kind == pto::FusionOpKind::LocalBoundary &&
                  op->hasAttr(kVmiFusionBoundaryAttr);
      }
      if (!include) {
        if (op == last)
          break;
        continue;
      }
      op->setAttr(kFusionGroupIdAttr,
                  IntegerAttr::get(IntegerType::get(ctx, 64), groupId));
      op->setAttr(kFusionOrderAttr,
                  IntegerAttr::get(IntegerType::get(ctx, 64), order++));
      if (op == last)
        break;
    }
  }
}

static bool isSupportedPlanningNode(const pto::FusionComputeNode &node) {
  return node.semantics.kind == pto::FusionOpKind::Compute &&
         isCurrentlyPlannableOp(node.semantics.opName);
}

static unsigned
countEdgesFromGroup(const pto::FusionBlockAnalysis &blockAnalysis,
                    ArrayRef<const pto::FusionComputeNode *> group,
                    const pto::FusionComputeNode &candidate) {
  DenseSet<unsigned> producerIds;
  for (const pto::FusionComputeNode *member : group)
    producerIds.insert(member->id);

  unsigned count = 0;
  for (unsigned edgeId : candidate.incomingEdges) {
    if (edgeId >= blockAnalysis.edges.size())
      continue;
    if (producerIds.contains(blockAnalysis.edges[edgeId].producerNode))
      ++count;
  }
  return count;
}

struct GroupFootprint {
  unsigned liveTileCount = 0;
  unsigned vfParameterCount = 0;
};

static bool nodesHaveDirectDataFlowConnection(
    const pto::FusionBlockAnalysis &blockAnalysis,
    const pto::FusionComputeNode &lhs, const pto::FusionComputeNode &rhs) {
  for (unsigned edgeId : lhs.outgoingEdges) {
    if (edgeId >= blockAnalysis.edges.size())
      continue;
    if (blockAnalysis.edges[edgeId].consumerNode == rhs.id)
      return true;
  }

  for (unsigned edgeId : lhs.incomingEdges) {
    if (edgeId >= blockAnalysis.edges.size())
      continue;
    if (blockAnalysis.edges[edgeId].producerNode == rhs.id)
      return true;
  }

  for (Value output : lhs.semantics.tileOutputs)
    if (llvm::is_contained(rhs.semantics.tileInputs, output))
      return true;

  for (Value output : rhs.semantics.tileOutputs)
    if (llvm::is_contained(lhs.semantics.tileInputs, output))
      return true;

  return false;
}

static unsigned
countConnectionsToGroup(const pto::FusionBlockAnalysis &blockAnalysis,
                        ArrayRef<const pto::FusionComputeNode *> group,
                        const pto::FusionComputeNode &candidate) {
  unsigned connections = 0;
  for (const pto::FusionComputeNode *member : group)
    if (nodesHaveDirectDataFlowConnection(blockAnalysis, *member, candidate))
      ++connections;
  return connections;
}

static GroupFootprint
computeGroupFootprint(ArrayRef<const pto::FusionComputeNode *> members) {
  DenseSet<Value> producedTiles;
  DenseSet<Value> touchedTiles;
  DenseSet<Value> externalInputs;

  for (const pto::FusionComputeNode *member : members) {
    for (Value output : member->semantics.tileOutputs) {
      producedTiles.insert(output);
      touchedTiles.insert(output);
    }
  }

  for (const pto::FusionComputeNode *member : members) {
    for (Value input : member->semantics.tileInputs) {
      touchedTiles.insert(input);
      if (!producedTiles.contains(input))
        externalInputs.insert(input);
    }
  }

  GroupFootprint footprint;
  footprint.liveTileCount = touchedTiles.size();
  footprint.vfParameterCount = externalInputs.size() + producedTiles.size();
  return footprint;
}

class CostModel {
public:
  virtual ~CostModel() = default;

  virtual PlanningDecision evaluateSeed(const PlanningContext &ctx,
                                        const pto::FusionComputeNode &candidate)
      const = 0;

  virtual PlanningDecision
  evaluateAppend(const PlanningContext &ctx,
                 ArrayRef<const pto::FusionComputeNode *> currentGroup,
                 const pto::FusionComputeNode &candidate) const = 0;
};

class ConservativeGreedyCostModel final : public CostModel {
public:
  PlanningDecision
  evaluateSeed(const PlanningContext &ctx,
               const pto::FusionComputeNode &candidate) const override {
    PlanningDecision decision;
    if (!isSupportedPlanningNode(candidate))
      return decision;

    if (!isProvenIterationDomain(ctx.blockAnalysis, candidate)) {
      decision.cost.rejectedForDynamicShape = true;
      return decision;
    }

    decision.accept = true;
    return decision;
  }

  PlanningDecision
  evaluateAppend(const PlanningContext &ctx,
                 ArrayRef<const pto::FusionComputeNode *> currentGroup,
                 const pto::FusionComputeNode &candidate) const override {
    PlanningDecision seedDecision = evaluateSeed(ctx, candidate);
    if (!seedDecision.accept)
      return seedDecision;

    PlanningDecision decision;
    if (currentGroup.empty()) {
      decision.accept = true;
      return decision;
    }

    const pto::FusionComputeNode &previous = *currentGroup.back();
    const bool sameDomainClass =
        previous.iterationDomainClass == candidate.iterationDomainClass;
    const bool contiguousInBlock =
        candidate.blockOrder == previous.blockOrder + 1;
    const bool directlyDependent =
        dependsOnPreviousNode(ctx.blockAnalysis, previous, candidate);
    if (!sameDomainClass || !contiguousInBlock || !directlyDependent)
      return decision;

    SmallVector<const pto::FusionComputeNode *, 8> proposedGroup(
        currentGroup.begin(), currentGroup.end());
    proposedGroup.push_back(&candidate);
    GroupFootprint footprint = computeGroupFootprint(proposedGroup);

    decision.cost.dependencyBenefit =
        4 * static_cast<int64_t>(
                countEdgesFromGroup(ctx.blockAnalysis, currentGroup, candidate));
    decision.cost.loopMergeBenefit = 2;
    decision.cost.liveTilePenalty =
        std::max<int64_t>(0, static_cast<int64_t>(footprint.liveTileCount) - 4);
    decision.cost.vfParameterPenalty = std::max<int64_t>(
        0, static_cast<int64_t>(footprint.vfParameterCount) - 6);
    decision.accept = decision.cost.total() > 0;
    return decision;
  }
};

class ConservativeDAGGreedyCostModel final : public CostModel {
public:
  PlanningDecision
  evaluateSeed(const PlanningContext &ctx,
               const pto::FusionComputeNode &candidate) const override {
    PlanningDecision decision;
    if (!isSupportedPlanningNode(candidate))
      return decision;

    if (!isProvenIterationDomain(ctx.blockAnalysis, candidate)) {
      decision.cost.rejectedForDynamicShape = true;
      return decision;
    }

    decision.accept = true;
    return decision;
  }

  PlanningDecision
  evaluateAppend(const PlanningContext &ctx,
                 ArrayRef<const pto::FusionComputeNode *> currentGroup,
                 const pto::FusionComputeNode &candidate) const override {
    PlanningDecision seedDecision = evaluateSeed(ctx, candidate);
    if (!seedDecision.accept)
      return seedDecision;

    PlanningDecision decision;
    if (currentGroup.empty()) {
      decision.accept = true;
      return decision;
    }

    if (currentGroup.front()->iterationDomainClass !=
        candidate.iterationDomainClass)
      return decision;

    if (hasHardBoundaryToGroup(currentGroup, candidate))
      return decision;

    const unsigned connectionCount =
        countConnectionsToGroup(ctx.blockAnalysis, currentGroup, candidate);
    if (connectionCount == 0)
      return decision;

    SmallVector<const pto::FusionComputeNode *, 8> proposedGroup(
        currentGroup.begin(), currentGroup.end());
    proposedGroup.push_back(&candidate);
    GroupFootprint footprint = computeGroupFootprint(proposedGroup);

    decision.cost.dependencyBenefit = 4 * static_cast<int64_t>(connectionCount);
    decision.cost.loopMergeBenefit = 4;
    decision.cost.liveTilePenalty = std::max<int64_t>(
        0, static_cast<int64_t>(footprint.liveTileCount) - 10);
    decision.cost.vfParameterPenalty = std::max<int64_t>(
        0, static_cast<int64_t>(footprint.vfParameterCount) - 12);
    decision.accept = decision.cost.total() > 0;
    return decision;
  }
};

class StrategyEngine {
public:
  virtual ~StrategyEngine() = default;

  virtual SmallVector<PlannedFusionGroup, 8>
  planBlock(const PlanningContext &ctx, const CostModel &costModel) const = 0;
};

class ConservativeGreedyStrategyEngine final : public StrategyEngine {
public:
  SmallVector<PlannedFusionGroup, 8>
  planBlock(const PlanningContext &ctx,
            const CostModel &costModel) const override {
    SmallVector<PlannedFusionGroup, 8> groups;
    SmallVector<const pto::FusionComputeNode *, 8> chain;

    auto flushChain = [&]() {
      if (chain.size() < 2) {
        chain.clear();
        return;
      }

      PlannedFusionGroup group;
      group.members = chain;
      groups.push_back(std::move(group));
      chain.clear();
    };

    for (const pto::FusionComputeNode &node : ctx.blockAnalysis.computeNodes) {
      PlanningDecision seedDecision = costModel.evaluateSeed(ctx, node);
      if (!seedDecision.accept) {
        flushChain();
        continue;
      }

      if (chain.empty()) {
        chain.push_back(&node);
        continue;
      }

      PlanningDecision appendDecision =
          costModel.evaluateAppend(ctx, chain, node);
      if (!appendDecision.accept) {
        flushChain();
        chain.push_back(&node);
        continue;
      }

      chain.push_back(&node);
    }

    flushChain();
    return groups;
  }
};

class ConservativeDAGGreedyStrategyEngine final : public StrategyEngine {
public:
  SmallVector<PlannedFusionGroup, 8>
  planBlock(const PlanningContext &ctx,
            const CostModel &costModel) const override {
    SmallVector<PlannedFusionGroup, 8> groups;
    DenseSet<unsigned> assignedNodes;

    for (const pto::FusionComputeNode &seed : ctx.blockAnalysis.computeNodes) {
      if (assignedNodes.contains(seed.id))
        continue;

      PlanningDecision seedDecision = costModel.evaluateSeed(ctx, seed);
      if (!seedDecision.accept)
        continue;

      SmallVector<const pto::FusionComputeNode *, 8> groupMembers;
      DenseSet<unsigned> groupNodeIds;
      groupMembers.push_back(&seed);
      groupNodeIds.insert(seed.id);

      bool changed = true;
      while (changed) {
        changed = false;
        for (const pto::FusionComputeNode &candidate :
             ctx.blockAnalysis.computeNodes) {
          if (assignedNodes.contains(candidate.id) ||
              groupNodeIds.contains(candidate.id))
            continue;

          PlanningDecision appendDecision =
              costModel.evaluateAppend(ctx, groupMembers, candidate);
          if (!appendDecision.accept)
            continue;

          groupMembers.push_back(&candidate);
          groupNodeIds.insert(candidate.id);
          changed = true;
        }
      }

      if (groupMembers.size() < 2)
        continue;

      PlannedFusionGroup group;
      group.members = buildStableInGroupOrder(groupMembers);
      groups.push_back(group);
      for (const pto::FusionComputeNode *member : group.members)
        assignedNodes.insert(member->id);
    }

    return groups;
  }
};

// VMI F3-adjacency strategy: every plannable compute node (already filtered
// into computeNodes by PreFusionAnalysis via getFusionOpSemantics' whitelist)
// is wrapped into a loose fusion group. Pure alloc_tile resource declarations
// and tile subviews are transparent: PTOOpScheduling can move compute members
// across them before RegionGen builds a contiguous span. Hard non-plannable ops
// (tload/tstore DMA, wait_flag/mem_bar sync, any other unknown op) close the
// group. A local
// fallback remains in the same loose group, but is not a compute node and
// therefore still stops downstream VMI loop fusion.
//
// This layer does NOT do UB-overlap (MustAlias/NoAlias/partial-alias)
// checking, unlike the deleted PTOPlanVmiFusionRegion pass. At this
// tile-native PTO IR layer a tile_buf is a single SSA value with one fixed
// type, so partial-alias (same address, different type) cannot arise — it
// only becomes possible after PlanMemory binds addresses, which is
// downstream of this pass. The reduce-final "stuck" case (a ColReduce whose
// result is read by a later elementwise loop — a producer/consumer
// dependency that loop-fusion refuses to fold into one iteration even though
// the two nodes share a region) is deliberately handled downstream by
// PTOVmiLoopFusion, which can see the scf.for-level SSA/UB def-use that this
// compute-node layer cannot. A fusion_region assigned here is a container,
// not a fusion mandate: same group does not force the inner scf.for ops to
// fuse.
//
// Unlike the Conservative* engines this one ignores iteration-domain class,
// direct dependency, and cost. Single-node groups are kept (not dropped) so
// every plannable compute TileOp gets a fusion_region.
class VMIUBDisjointStrategyEngine final : public StrategyEngine {
public:
  SmallVector<PlannedFusionGroup, 8>
  planBlock(const PlanningContext &ctx,
            const CostModel &costModel) const override {
    const pto::FusionBlockAnalysis &block = ctx.blockAnalysis;
    if (block.computeNodes.empty())
      return {};

    // Map node id -> whether a real hard boundary precedes it in block order.
    // Structural allocation/view scaffold and pure scalar/index plumbing are
    // transparent and are later annotated into the loose region. A local
    // fallback also stays in the loose group, but stops downstream loop fusion.
    DenseMap<unsigned, bool> precededByNonPlannable;
    DenseMap<Operation *, unsigned> nodeIdByOp;
    for (const pto::FusionComputeNode &n : block.computeNodes)
      nodeIdByOp[n.op] = n.id;
    bool sawCompute = false;
    bool hardBoundarySinceCompute = false;
    for (Operation &op : *block.block) {
      auto it = nodeIdByOp.find(&op);
      if (it != nodeIdByOp.end()) {
        precededByNonPlannable[it->second] =
            sawCompute && hardBoundarySinceCompute;
        sawCompute = true;
        hardBoundarySinceCompute = false;
        continue;
      }
      if (pto::isFusionTransparentScaffold(&op))
        continue;
      FailureOr<pto::FusionOpSemantics> semanticsOr =
          pto::getFusionOpSemantics(&op);
      if (failed(semanticsOr) ||
          semanticsOr->kind == pto::FusionOpKind::HardBoundary ||
          (semanticsOr->kind == pto::FusionOpKind::LocalBoundary &&
           !op.hasAttr(kVmiFusionBoundaryAttr)))
        hardBoundarySinceCompute = true;
    }

    SmallVector<PlannedFusionGroup, 8> groups;
    SmallVector<const pto::FusionComputeNode *, 8> curMembers;
    auto flushCurrent = [&]() {
      if (curMembers.empty())
        return;
      PlannedFusionGroup group;
      group.members = buildStableInGroupOrder(curMembers);
      groups.push_back(std::move(group));
      curMembers.clear();
    };

    for (const pto::FusionComputeNode &node : block.computeNodes) {
      // F3 boundary: a real non-plannable op sits between the previous
      // plannable node and this one. Transparent scaffold was filtered above,
      // so this closes the group for sync/DMA/call/unknown PTO operations.
      auto precIt = precededByNonPlannable.find(node.id);
      if (precIt != precededByNonPlannable.end() && precIt->second)
        flushCurrent();

      curMembers.push_back(&node);
    }
    flushCurrent();
    return groups;
  }
};

static void clearPlanningAttrs(func::FuncOp func) {
  func.walk([](Operation *op) {
    op->removeAttr(kFusionGroupIdAttr);
    op->removeAttr(kFusionOrderAttr);
  });
}

struct FusionPlanPass : public pto::impl::FusionPlanBase<FusionPlanPass> {
  using pto::impl::FusionPlanBase<FusionPlanPass>::FusionPlanBase;

  FusionPlanPass() = default;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (func.isExternal())
      return;

    clearPlanningAttrs(func);

    // Reuse the shared (analysis-manager-cached) pre-fusion dataflow graph
    // rather than rebuilding it from scratch.  The DFG — compute nodes, edges,
    // liveness, write instances — is identical regardless of whether shape
    // inference is enabled, so it is built once by PreFusionAnalysis and shared
    // with FusionRegionGen via the analysis manager.  Only iteration-domain
    // inference depends on the --enable-shape-inference option, so we run that
    // separable step ourselves on a local copy of the cached graph.
    const pto::PreFusionAnalysis &sharedAnalysis =
        getAnalysis<pto::PreFusionAnalysis>();
    if (!sharedAnalysis.isValid()) {
      signalPassFailure();
      return;
    }
    pto::PreFusionAnalysisResult analysis = sharedAnalysis.getResult();
    if (failed(pto::inferIterationDomainClasses(analysis, enableShapeInference))) {
      signalPassFailure();
      return;
    }

    MLIRContext *ctx = &getContext();
    int64_t nextGroupId = 0;
    ConservativeDAGGreedyCostModel costModel;
    // Strategy selection. Only these two enumerated values are accepted; any
    // other string (including typos) fails the pass instead of silently
    // falling back, so a misconfigured --fusion-strategy cannot quietly change
    // compilation behavior. The legacy a5 path uses ConservativeDAGGreedy; the
    // VMI path uses VMIUBDisjoint, which groups plannable compute nodes by
    // F3 adjacency (a non-plannable op between two plannable nodes closes the
    // group) and keeps single-node groups so every compute TileOp gets a
    // fusion_region. It does NOT judge UB disjointness, DFG dependency,
    // iteration-domain class, or cost here — the resulting fusion_region is a
    // *container* for downstream VMI analysis, not a proof that its inner
    // scf.for loops can be fused (that is PTOVmiLoopFusion's job).
    std::unique_ptr<StrategyEngine> strategyEngine;
    const std::string strategyVal = strategy.getValue();
    if (strategyVal == "conservative-dag-greedy")
      strategyEngine =
          std::make_unique<ConservativeDAGGreedyStrategyEngine>();
    else if (strategyVal == "vmi-ub-disjoint")
      strategyEngine = std::make_unique<VMIUBDisjointStrategyEngine>();
    else {
      emitError(getOperation()->getLoc())
          << "unknown pto-fusion-plan --fusion-strategy='" << strategyVal
          << "'; expected 'conservative-dag-greedy' or 'vmi-ub-disjoint'";
      signalPassFailure();
      return;
    }

    for (const pto::FusionBlockAnalysis &blockAnalysis : analysis.blocks) {
      PlanningContext planningCtx{blockAnalysis};
      SmallVector<PlannedFusionGroup, 8> groups =
          strategyEngine->planBlock(planningCtx, costModel);
      assignStableGroupMetadata(groups, ctx, nextGroupId);
    }

    // The fusion metadata we annotate (group_id/order) is a planning *output*;
    // it does not alter tile semantics, operand types, aliasing or liveness,
    // so it cannot invalidate the shared PreFusionAnalysis DFG.  Preserve it so
    // the downstream FusionRegionGen pass reuses the cached graph instead of
    // rebuilding it.
    markAnalysesPreserved<pto::PreFusionAnalysis>();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createFusionPlanPass() {
  return std::make_unique<FusionPlanPass>();
}

std::unique_ptr<Pass>
mlir::pto::createFusionPlanPass(const pto::FusionPlanOptions &options) {
  return std::make_unique<FusionPlanPass>(options);
}
