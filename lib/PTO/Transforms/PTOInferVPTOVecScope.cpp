// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOInferVPTOVecScope.cpp ------------------------------------------===//
//
// VPTO automatic vecscope inference.
//
//===----------------------------------------------------------------------===//

#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/SmallPtrSet.h"

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_PTOINFERVPTOVECSCOPE
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;

namespace {

enum class VPTOInferenceOpClass {
  Vector,
  SafeScalar,
  Boundary,
};

struct NestedRegionSummary {
  bool hasVectorOperation = false;
  bool hasBoundaryOperation = false;
};

struct EscapingMovedValue {
  Value value;
  Operation *producer = nullptr;
  Operation *user = nullptr;
  bool requiresDiagnostic = false;
};

struct ResultlessScopePlan {
  SmallVector<Operation *, 16> hoistOps;
  SmallVector<Operation *, 16> moveOps;
};

static VPTOInferenceOpClass classifyOperationForInference(Operation *op);
static LogicalResult inferVecScopesInRegion(Region &region,
                                            MLIRContext *context);

static bool isVecScopeType(Type type) {
  return isa<pto::VRegType, pto::MaskType, pto::AlignType>(type);
}

static bool isPTOOperation(Operation *op) {
  return op && op->getName().getStringRef().starts_with("pto.");
}

static bool isExplicitVectorScopeCarrier(Operation *op) {
  return isa<pto::VecScopeOp, pto::StrictVecScopeOp>(op);
}

static Operation *getEnclosingExplicitVectorScope(Operation *op) {
  for (Operation *parent = op ? op->getParentOp() : nullptr; parent;
       parent = parent->getParentOp()) {
    if (isExplicitVectorScopeCarrier(parent))
      return parent;
  }
  return nullptr;
}

static bool isForbiddenInsideInferredVectorScope(Operation *op) {
  return isa<pto::VbitsortOp, pto::Vmrgsort4Op>(op);
}

static bool isCloneableMaskProducer(Operation *op) {
  return isa<pto::PsetB8Op, pto::PsetB16Op, pto::PsetB32Op, pto::PgeB8Op,
             pto::PgeB16Op, pto::PgeB32Op, pto::PltB8Op, pto::PltB16Op,
             pto::PltB32Op, pto::PandOp, pto::PorOp, pto::PnotOp,
             pto::PintlvB8Op, pto::PintlvB16Op, pto::PintlvB32Op>(op);
}

static bool isCloneableScalarBroadcastProducer(Operation *op) {
  if (isa<pto::VbrOp>(op))
    return true;
  auto vdup = dyn_cast<pto::VdupOp>(op);
  return vdup && !isa<pto::VRegType>(vdup.getInput().getType());
}

static bool isIndexedVectorMemoryConsumer(Operation *op) {
  return isa<pto::Vgather2Op, pto::Vgather2BcOp, pto::VscatterOp>(op);
}

static bool hasOnlyIndexedVectorMemoryUsers(Operation *op) {
  for (Value result : op->getResults()) {
    for (Operation *user : result.getUsers()) {
      if (isIndexedVectorMemoryConsumer(user))
        continue;
      auto vand = dyn_cast<pto::VandOp>(user);
      if (!vand || !llvm::all_of(vand->getUsers(),
                                 isIndexedVectorMemoryConsumer))
        return false;
    }
  }
  return true;
}

static bool isCloneableIndexedMemoryIndexProducer(Operation *op) {
  return isa<pto::VciOp, pto::VandOp>(op) &&
         hasOnlyIndexedVectorMemoryUsers(op);
}

static bool isCloneableScalarVRegProducer(Operation *op) {
  return isa<pto::VbrOp>(op);
}

static bool isCloneableSharedProducer(Operation *op) {
  return isCloneableMaskProducer(op) ||
         isCloneableScalarBroadcastProducer(op) ||
         isCloneableScalarVRegProducer(op) ||
         isCloneableIndexedMemoryIndexProducer(op);
}

static bool isVectorScopeBoundaryOperation(Operation *op) {
  return isa<pto::BarrierOp, pto::BarrierSyncOp>(op);
}

static bool isPureAddressOperation(Operation *op) {
  return isa<pto::PointerCastOp, pto::CastPtrOp, pto::AddPtrOp,
             pto::TileBufAddrOp, pto::BindTileOp>(op);
}

static bool hasVecScopeTypedOperandOrResult(Operation *op) {
  for (Type type : op->getOperandTypes()) {
    if (isVecScopeType(type))
      return true;
  }
  for (Type type : op->getResultTypes()) {
    if (isVecScopeType(type))
      return true;
  }
  return false;
}

static bool requiresVectorScope(Operation *op) {
  if (!isPTOOperation(op))
    return false;

  return hasVecScopeTypedOperandOrResult(op) ||
         isa<pto::MemBarOp, pto::SprclrOp>(op);
}

static bool isAtomicControlFlowCandidate(Operation *op) {
  return isa<scf::IfOp, scf::ForOp>(op);
}

static bool isSafeScalarOperation(Operation *op) {
  if (op->getNumRegions() != 0)
    return false;
  if (op->hasTrait<OpTrait::IsTerminator>())
    return false;
  if (isa<func::CallOp>(op))
    return false;
  if (isPureAddressOperation(op))
    return true;
  if (isPTOOperation(op) && !isMemoryEffectFree(op))
    return false;
  return isMemoryEffectFree(op);
}

static void summarizeNestedRegionForAtomicCluster(
    Region &region, NestedRegionSummary &summary) {
  for (Block &block : region) {
    for (Operation &op : block) {
      if (op.hasTrait<OpTrait::IsTerminator>())
        continue;

      switch (classifyOperationForInference(&op)) {
      case VPTOInferenceOpClass::Vector:
        summary.hasVectorOperation = true;
        break;
      case VPTOInferenceOpClass::SafeScalar:
        break;
      case VPTOInferenceOpClass::Boundary:
        summary.hasBoundaryOperation = true;
        return;
      }
    }
  }
}

static bool canTreatAsAtomicControlFlow(Operation *op) {
  if (!isAtomicControlFlowCandidate(op))
    return false;

  NestedRegionSummary summary;
  for (Region &region : op->getRegions()) {
    summarizeNestedRegionForAtomicCluster(region, summary);
    if (summary.hasBoundaryOperation)
      return false;
  }
  return summary.hasVectorOperation;
}

static VPTOInferenceOpClass classifyOperationForInference(Operation *op) {
  if (!op)
    return VPTOInferenceOpClass::Boundary;

  if (isExplicitVectorScopeCarrier(op))
    return VPTOInferenceOpClass::Boundary;
  if (op->hasTrait<OpTrait::IsTerminator>())
    return VPTOInferenceOpClass::Boundary;
  if (isa<func::CallOp>(op))
    return VPTOInferenceOpClass::Boundary;
  if (isVectorScopeBoundaryOperation(op))
    return VPTOInferenceOpClass::Boundary;
  if (isForbiddenInsideInferredVectorScope(op))
    return VPTOInferenceOpClass::Boundary;

  if (requiresVectorScope(op))
    return VPTOInferenceOpClass::Vector;

  if (canTreatAsAtomicControlFlow(op))
    return VPTOInferenceOpClass::Vector;

  if (isSafeScalarOperation(op))
    return VPTOInferenceOpClass::SafeScalar;

  return VPTOInferenceOpClass::Boundary;
}

static bool hasVectorOperation(ArrayRef<Operation *> ops) {
  return llvm::any_of(ops, [](Operation *op) {
    return classifyOperationForInference(op) == VPTOInferenceOpClass::Vector;
  });
}

static bool isUserInsideCluster(Operation *user,
                                const llvm::SmallPtrSetImpl<Operation *> &ops) {
  for (Operation *cur = user; cur; cur = cur->getParentOp()) {
    if (ops.contains(cur))
      return true;
  }
  return false;
}

static bool anyUserIsMoved(Value result,
                           const llvm::SmallPtrSetImpl<Operation *> &movedOps) {
  for (Operation *user : result.getUsers()) {
    if (isUserInsideCluster(user, movedOps))
      return true;
  }
  return false;
}

static llvm::SmallPtrSet<Operation *, 16>
computeMovedOpsForResultlessScope(ArrayRef<Operation *> ops) {
  llvm::SmallPtrSet<Operation *, 16> movedOps;
  for (Operation *op : ops) {
    if (classifyOperationForInference(op) == VPTOInferenceOpClass::Vector)
      movedOps.insert(op);
  }

  bool changed = true;
  while (changed) {
    changed = false;
    for (Operation *op : llvm::reverse(ops)) {
      if (movedOps.contains(op) ||
          classifyOperationForInference(op) !=
              VPTOInferenceOpClass::SafeScalar)
        continue;

      bool hasMovedUser = false;
      bool allUsersMoved = true;
      for (Value result : op->getResults()) {
        for (Operation *user : result.getUsers()) {
          if (isUserInsideCluster(user, movedOps)) {
            hasMovedUser = true;
            continue;
          }
          if (!isUserInsideCluster(user, movedOps)) {
            allUsersMoved = false;
            break;
          }
        }
        if (!allUsersMoved)
          break;
      }

      if (hasMovedUser && allUsersMoved) {
        movedOps.insert(op);
        changed = true;
      }
    }
  }
  return movedOps;
}

static void pruneExternallyConsumedCloneableProducers(
    llvm::SmallPtrSetImpl<Operation *> &movedOps) {
  bool changed = true;
  while (changed) {
    changed = false;
    SmallVector<Operation *, 8> producersToPrune;
    for (Operation *op : movedOps) {
      if (!isCloneableSharedProducer(op))
        continue;

      bool hasExternalUser = false;
      for (Value result : op->getResults()) {
        for (Operation *user : result.getUsers()) {
          if (!isUserInsideCluster(user, movedOps))
            hasExternalUser = true;
        }
      }

      // Keep cloneable producers out of the resultless scope whenever one of
      // their users is external. They can be rematerialized in the eventual
      // consumer scope; retaining them in the moved set would make the mask
      // escape check fail before that rematerialization can happen.
      if (hasExternalUser)
        producersToPrune.push_back(op);
    }

    for (Operation *op : producersToPrune)
      changed |= movedOps.erase(op);
  }
}

static Operation *getAncestorInBlock(Operation *op, Block &block) {
  for (Operation *cur = op; cur; cur = cur->getParentOp()) {
    if (cur->getBlock() == &block)
      return cur;
  }
  return nullptr;
}

static bool isUseBeforeInBlock(OpOperand *lhs, OpOperand *rhs, Block &block) {
  Operation *lhsAncestor = getAncestorInBlock(lhs->getOwner(), block);
  Operation *rhsAncestor = getAncestorInBlock(rhs->getOwner(), block);
  if (!lhsAncestor || !rhsAncestor || lhsAncestor == rhsAncestor)
    return false;
  return lhsAncestor->isBeforeInBlock(rhsAncestor);
}

static void keepEarliestUseFirst(SmallVectorImpl<OpOperand *> &uses,
                                 Block &block) {
  if (uses.size() < 2)
    return;

  unsigned earliest = 0;
  for (unsigned i = 1, e = uses.size(); i < e; ++i) {
    if (isUseBeforeInBlock(uses[i], uses[earliest], block))
      earliest = i;
  }
  if (earliest != 0)
    std::swap(uses.front(), uses[earliest]);
}

static bool isSeparatedByInferenceBoundary(Operation *producer,
                                           Operation *user, Block &block) {
  Operation *userAncestor = getAncestorInBlock(user, block);
  if (!producer || producer->getBlock() != &block || !userAncestor ||
      producer == userAncestor || !producer->isBeforeInBlock(userAncestor))
    return false;

  for (Operation *current = producer->getNextNode();
       current && current != userAncestor; current = current->getNextNode()) {
    if (classifyOperationForInference(current) ==
        VPTOInferenceOpClass::Boundary)
      return true;
  }
  return false;
}

static bool shouldCloneSharedResult(OpResult result) {
  Type type = result.getType();
  return isa<pto::MaskType, pto::VRegType>(type);
}

static void cloneSharedProducers(Block &block, MLIRContext *context) {
  IRRewriter rewriter(context);
  bool changed = true;
  while (changed) {
    changed = false;

    SmallVector<Operation *, 32> ops;
    for (Operation &op : block)
      ops.push_back(&op);

    for (Operation *op : ops) {
      if (!isCloneableSharedProducer(op))
        continue;

      SmallVector<OpOperand *, 8> uses;
      for (OpResult result : op->getOpResults()) {
        if (!shouldCloneSharedResult(result) || result.use_empty())
          continue;
        for (OpOperand &use : result.getUses())
          uses.push_back(&use);
      }
      if (uses.size() < 2) {
        if (uses.empty() ||
            !isSeparatedByInferenceBoundary(op, uses.front()->getOwner(),
                                            block))
          continue;
      }

      keepEarliestUseFirst(uses, block);
      bool cloneFirst = isSeparatedByInferenceBoundary(
          op, uses.front()->getOwner(), block);
      ArrayRef<OpOperand *> usesToClone = uses;
      if (!cloneFirst)
        usesToClone = usesToClone.drop_front();
      for (OpOperand *use : usesToClone) {
        auto result = cast<OpResult>(use->get());
        Operation *user = use->getOwner();
        rewriter.setInsertionPoint(user);
        Operation *clone = rewriter.clone(*op);
        use->set(clone->getResult(result.getResultNumber()));
        changed = true;
      }
      if (cloneFirst && op->use_empty())
        rewriter.eraseOp(op);
    }
  }
}

// A boundary between a pure mask producer and an atomic vector loop can cause
// the loop to be wrapped without its producer. If every use ended up in the
// same dedicated vecscope, rematerialize the producer there instead of leaving
// vector-scope-only data at the parent level.
static void sinkCloneableProducersIntoDedicatedScopes(Block &block) {
  SmallVector<Operation *, 16> producers;
  for (Operation &op : block) {
    if (isCloneableSharedProducer(&op))
      producers.push_back(&op);
  }

  for (Operation *producer : llvm::reverse(producers)) {
    Operation *commonScope = nullptr;
    bool hasUse = false;
    bool allUsersShareScope = true;
    for (Value result : producer->getResults()) {
      for (Operation *user : result.getUsers()) {
        hasUse = true;
        Operation *scope = getEnclosingExplicitVectorScope(user);
        if (!scope || (commonScope && commonScope != scope)) {
          allUsersShareScope = false;
          break;
        }
        commonScope = scope;
      }
      if (!allUsersShareScope)
        break;
    }

    if (!hasUse || !allUsersShareScope || !commonScope)
      continue;

    // Keep rematerialization local to the block currently being inferred.
    // In particular, do not move an outer producer into a vecscope nested in
    // an scf.for/scf.if region merely because all of its users happen to be
    // there. Also require every operand to remain available at the scope
    // entry before moving the producer to the start of the scope body.
    if (commonScope->getBlock() != &block)
      continue;
    DominanceInfo dominance;
    if (!llvm::all_of(producer->getOperands(), [&](Value operand) {
          return dominance.dominates(operand, commonScope);
        }))
      continue;

    Block &scopeBody = commonScope->getRegion(0).front();
    producer->moveBefore(&scopeBody, scopeBody.begin());
  }
}

static bool findEscapingMovedResult(
    const llvm::SmallPtrSetImpl<Operation *> &movedOps,
    EscapingMovedValue &escapingValue) {
  for (Operation *op : movedOps) {
    for (Value result : op->getResults()) {
      for (Operation *user : result.getUsers()) {
        if (isUserInsideCluster(user, movedOps))
          continue;

        escapingValue.value = result;
        escapingValue.producer = op;
        escapingValue.user = user;
        escapingValue.requiresDiagnostic = isVecScopeType(result.getType());
        return true;
      }
    }
  }
  return false;
}

static LogicalResult
emitEscapingVectorScopeValueError(const EscapingMovedValue &escapingValue) {
  Operation *producer = escapingValue.producer;
  if (!producer)
    return failure();

  InFlightDiagnostic diag = producer->emitOpError()
                            << "cannot infer resultless pto.vecscope because "
                               "VPTO vector-scope data cannot have external "
                               "users";
  if (escapingValue.value)
    diag << "; escaping value type is " << escapingValue.value.getType();
  if (escapingValue.user) {
    diag << "; external user is '"
         << escapingValue.user->getName().getStringRef() << "'";
    diag.attachNote(escapingValue.user->getLoc())
        << "external user is here";
  }
  return failure();
}

// classify which operations need to be moved into a vecscope, which can be hoisted out of the
// vecscope, and check for any vector-scope-typed values that would escape the vecscope if we were to
// move the candidate operations into a resultless vecscope. Returns failure if the candidate cluster
// is not suitable for vecscope inference.
static LogicalResult
buildResultlessScopePlan(ArrayRef<Operation *> ops, ResultlessScopePlan &plan,
                         EscapingMovedValue &escapingValue) {
  if (ops.empty() || !hasVectorOperation(ops))
    return failure();

  llvm::SmallPtrSet<Operation *, 16> movedOps =
      computeMovedOpsForResultlessScope(ops);
  pruneExternallyConsumedCloneableProducers(movedOps);
  if (movedOps.empty())
    return failure();

  if (findEscapingMovedResult(movedOps, escapingValue))
    return failure();

  llvm::SmallPtrSet<Operation *, 16> hoistedOps;
  for (Operation *op : ops) {
    if (movedOps.contains(op) ||
        classifyOperationForInference(op) != VPTOInferenceOpClass::SafeScalar)
      continue;

    for (Value result : op->getResults()) {
      if (anyUserIsMoved(result, movedOps)) {
        hoistedOps.insert(op);
        break;
      }
    }
  }

  bool changed = true;
  while (changed) {
    changed = false;
    for (Operation *op : llvm::reverse(ops)) {
      if (movedOps.contains(op) || hoistedOps.contains(op) ||
          classifyOperationForInference(op) !=
              VPTOInferenceOpClass::SafeScalar)
        continue;

      bool feedsHoistedOp = false;
      for (Value result : op->getResults()) {
        for (Operation *user : result.getUsers()) {
          if (isUserInsideCluster(user, hoistedOps)) {
            feedsHoistedOp = true;
            break;
          }
        }
        if (feedsHoistedOp)
          break;
      }

      if (feedsHoistedOp) {
        hoistedOps.insert(op);
        changed = true;
      }
    }
  }

  plan.hoistOps.clear();
  plan.moveOps.clear();
  for (Operation *op : ops) {
    if (hoistedOps.contains(op))
      plan.hoistOps.push_back(op);
    if (movedOps.contains(op))
      plan.moveOps.push_back(op);
  }
  return success();
}

static void wrapCluster(const ResultlessScopePlan &plan, MLIRContext *context) {
  if (plan.moveOps.empty())
    return;

  Operation *first = plan.moveOps.front();
  Block *parentBlock = first->getBlock();

  IRRewriter rewriter(context);
  rewriter.setInsertionPoint(first);
  auto scope = rewriter.create<pto::VecScopeOp>(first->getLoc());
  scope.getBody().push_back(new Block());

  for (Operation *op : plan.hoistOps) {
    if (op->getBlock() == parentBlock && scope->isBeforeInBlock(op))
      op->moveBefore(scope);
  }

  Block &scopeBody = scope.getBody().front();
  for (Operation *op : plan.moveOps) {
    scopeBody.getOperations().splice(scopeBody.end(),
                                     parentBlock->getOperations(),
                                     Block::iterator(op));
  }
}

static LogicalResult wrapGreedySubclusters(ArrayRef<Operation *> ops,
                                           MLIRContext *context) {
  for (size_t begin = 0; begin < ops.size();) {
    size_t bestEnd = begin;
    ResultlessScopePlan bestPlan;
    EscapingMovedValue escapingValue;
    bool sawEscapingMovedResult = false;

    for (size_t end = ops.size(); end > begin; --end) {
      ArrayRef<Operation *> candidate = ops.slice(begin, end - begin);
      if (!hasVectorOperation(candidate))
        continue;

      // Prefer the largest suffix-preserving candidate that actually needs a
      // vecscope and can be moved into today's resultless pto.vecscope form.
      ResultlessScopePlan plan;
      EscapingMovedValue candidateEscapingValue;
      if (succeeded(buildResultlessScopePlan(candidate, plan,
                                             candidateEscapingValue))) {
        bestEnd = end;
        bestPlan = std::move(plan);
        break;
      }

      if (!sawEscapingMovedResult && candidateEscapingValue.producer) {
        escapingValue = candidateEscapingValue;
        sawEscapingMovedResult = true;
      }
    }

    if (bestEnd == begin) {
      if (classifyOperationForInference(ops[begin]) ==
              VPTOInferenceOpClass::Vector &&
          sawEscapingMovedResult && escapingValue.requiresDiagnostic)
        return emitEscapingVectorScopeValueError(escapingValue);
      ++begin;
      continue;
    }

    wrapCluster(bestPlan, context);
    begin = bestEnd;
  }
  return success();
}

static LogicalResult inferVecScopesInBlock(Block &block, MLIRContext *context) {
  cloneSharedProducers(block, context);

  SmallVector<Operation *, 16> pending;

  auto flush = [&]() -> LogicalResult {
    if (failed(wrapGreedySubclusters(pending, context)))
      return failure();
    pending.clear();
    return success();
  };

  SmallVector<Operation *, 32> ops;
  for (Operation &op : block)
    ops.push_back(&op);

  for (Operation *op : ops) {
    switch (classifyOperationForInference(op)) {
    case VPTOInferenceOpClass::Vector:
    case VPTOInferenceOpClass::SafeScalar:
      pending.push_back(op);
      continue;
    case VPTOInferenceOpClass::Boundary:
      if (failed(flush()))
        return failure();
      continue;
    }
  }
  if (failed(flush()))
    return failure();

  sinkCloneableProducersIntoDedicatedScopes(block);

  SmallVector<Operation *, 32> remainingOps;
  for (Operation &op : block)
    remainingOps.push_back(&op);

  for (Operation *op : remainingOps) {
    if (isExplicitVectorScopeCarrier(op))
      continue;
    for (Region &nested : op->getRegions()) {
      if (failed(inferVecScopesInRegion(nested, context)))
        return failure();
    }
  }
  return success();
}

static LogicalResult inferVecScopesInRegion(Region &region,
                                            MLIRContext *context) {
  for (Block &block : region) {
    if (failed(inferVecScopesInBlock(block, context)))
      return failure();
  }
  return success();
}

struct PTOInferVPTOVecScopePass
    : public pto::impl::PTOInferVPTOVecScopeBase<
          PTOInferVPTOVecScopePass> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (failed(inferVecScopesInRegion(func.getBody(), &getContext())))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createPTOInferVPTOVecScopePass() {
  return std::make_unique<PTOInferVPTOVecScopePass>();
}
