// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/StringSwitch.h"

using namespace mlir;

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_SELECTTEMPLATECANDIDATE
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

namespace {

constexpr llvm::StringLiteral kCandidatesAttr = "candidates";
constexpr llvm::StringLiteral kSelectedCandidateAttr =
    "pto.tilelib.selected_candidate";
constexpr llvm::StringLiteral kTileLibImplAttr = "pto.tilelib.impl";
constexpr llvm::StringLiteral kVmiFusionBoundaryAttr =
    "pto.vmi.fusion.boundary";
constexpr llvm::StringLiteral kVmiFusionBoundaryReasonAttr =
    "pto.vmi.fusion.boundary_reason";

static bool candidateHasTag(DictionaryAttr candidate, StringRef tag) {
  auto tags = candidate.getAs<ArrayAttr>("tags");
  if (!tags)
    return false;
  return llvm::any_of(tags, [tag](Attribute attr) {
    auto value = dyn_cast<StringAttr>(attr);
    return value && value.getValue() == tag;
  });
}

static bool getStaticIntFromValue(Value value, int64_t &out) {
  if (auto cOp = value.getDefiningOp<arith::ConstantIndexOp>()) {
    out = cOp.value();
    return true;
  }
  if (auto cInt = value.getDefiningOp<arith::ConstantIntOp>()) {
    out = cInt.value();
    return true;
  }
  return false;
}

static bool resolveStaticTileValidShape(Value value,
                                        SmallVectorImpl<int64_t> &validShape) {
  Value validRow;
  Value validCol;
  Operation *def = value.getDefiningOp();
  if (!def)
    return false;

  if (auto alloc = dyn_cast<pto::AllocTileOp>(def)) {
    validRow = alloc.getValidRow();
    validCol = alloc.getValidCol();
  } else if (auto materialize = dyn_cast<pto::MaterializeTileOp>(def)) {
    validRow = materialize.getValidRow();
    validCol = materialize.getValidCol();
  } else if (auto subview = dyn_cast<pto::SubViewOp>(def)) {
    validRow = subview.getValidRow();
    validCol = subview.getValidCol();
  } else if (auto popAic = dyn_cast<pto::TPopFromAicOp>(def)) {
    validRow = popAic.getValidRow();
    validCol = popAic.getValidCol();
  } else if (auto popAiv = dyn_cast<pto::TPopFromAivOp>(def)) {
    validRow = popAiv.getValidRow();
    validCol = popAiv.getValidCol();
  } else if (auto fusionRegion = dyn_cast<pto::FusionRegionOp>(def)) {
    auto result = dyn_cast<OpResult>(value);
    if (!result)
      return false;
    auto yieldOp =
        dyn_cast<pto::YieldOp>(fusionRegion.getBody().front().getTerminator());
    if (!yieldOp || result.getResultNumber() >= yieldOp.getNumOperands())
      return false;
    return resolveStaticTileValidShape(yieldOp.getOperand(result.getResultNumber()),
                                       validShape);
  }

  if (!validRow || !validCol)
    return false;

  int64_t row = ShapedType::kDynamic;
  int64_t col = ShapedType::kDynamic;
  if (!getStaticIntFromValue(validRow, row) ||
      !getStaticIntFromValue(validCol, col))
    return false;

  validShape.assign({row, col});
  return true;
}

static bool hasStaticFullTileValidShape(Operation *op) {
  for (Value operand : op->getOperands()) {
    auto tile = dyn_cast<pto::TileBufType>(operand.getType());
    if (!tile)
      continue;
    ArrayRef<int64_t> shape = tile.getShape();
    ArrayRef<int64_t> validShape = tile.getValidShape();
    SmallVector<int64_t, 2> resolvedValidShape;
    if ((validShape.empty() ||
         llvm::any_of(validShape, ShapedType::isDynamic)) &&
        resolveStaticTileValidShape(operand, resolvedValidShape)) {
      validShape = resolvedValidShape;
    }
    if (shape.size() != validShape.size())
      return false;
    for (auto [shapeDim, validDim] : llvm::zip_equal(shape, validShape))
      if (ShapedType::isDynamic(shapeDim) ||
          ShapedType::isDynamic(validDim) || shapeDim != validDim)
        return false;
  }
  return true;
}

static StringRef getTileOpName(Operation *op) {
  return op->getName().getStringRef().split('.').second;
}

static bool isHardBoundaryFallbackOp(Operation *op) {
  auto pipeOp = dyn_cast<pto::OpPipeInterface>(op);
  if (!pipeOp || pipeOp.getPipe() != pto::PIPE::PIPE_V)
    return true;
  return llvm::StringSwitch<bool>(getTileOpName(op))
      .Cases("tload", "tstore", "tmatmul", "tmatmul_acc", "tmatmul_bias",
             "tmatmul_mx", true)
      .Cases("tmrgsort", "tsort32", "tpush", "tpop", "tfree", true)
      .Cases("tgather", "tgatherb", "tscatter", "tscatterb", true)
      .Cases("textract", "textract_fp", "tfillpad", "tfillpad_expand",
             "tfillpad_inplace", true)
      .Cases("tconcat", "tinsert", "tci", true)
      .Default(false);
}

static bool isHardBoundaryFallback(Operation *op, DictionaryAttr candidate) {
  return isHardBoundaryFallbackOp(op) ||
         candidateHasTag(candidate, "hard_boundary");
}

static void recordSelection(Operation *op, DictionaryAttr candidate,
                            bool isVMI) {
  op->setAttr(kSelectedCandidateAttr, candidate);
  op->setAttr(kTileLibImplAttr,
              StringAttr::get(op->getContext(), isVMI ? "vmi" : "ptodsl"));
  op->removeAttr(kVmiFusionBoundaryAttr);
  op->removeAttr(kVmiFusionBoundaryReasonAttr);
  if (isVMI)
    return;

  bool hard = isHardBoundaryFallback(op, candidate);
  op->setAttr(kVmiFusionBoundaryAttr,
              StringAttr::get(op->getContext(), hard ? "hard" : "local"));
  op->setAttr(kVmiFusionBoundaryReasonAttr,
              StringAttr::get(op->getContext(),
                              hard ? "non_vmi_hard_boundary_fallback"
                                   : "non_vmi_local_boundary_fallback"));
}

struct SelectTemplateCandidatePass
    : pto::impl::SelectTemplateCandidateBase<SelectTemplateCandidatePass> {
  using SelectTemplateCandidateBase::SelectTemplateCandidateBase;

  void runOnOperation() override {
    if (selectionPolicy != "prefer-vmi" && selectionPolicy != "ordinary-only") {
      getOperation().emitError("unsupported template selection policy '")
          << selectionPolicy << "'";
      return signalPassFailure();
    }

    WalkResult result = getOperation().walk([&](Operation *op) {
      auto candidates = op->getAttrOfType<ArrayAttr>(kCandidatesAttr);
      if (!candidates)
        return WalkResult::advance();

      SmallVector<DictionaryAttr> parsed;
      for (Attribute attr : candidates) {
        auto candidate = dyn_cast<DictionaryAttr>(attr);
        if (!candidate || !candidate.getAs<StringAttr>("name")) {
          op->emitError("template candidate must be a dictionary with a string name");
          return WalkResult::interrupt();
        }
        parsed.push_back(candidate);
      }

      const bool hardBoundary =
          isHardBoundaryFallbackOp(op) ||
          llvm::any_of(parsed, [](DictionaryAttr candidate) {
            return candidateHasTag(candidate, "hard_boundary");
          });
      if (selectionPolicy == "prefer-vmi" && !hardBoundary) {
        for (DictionaryAttr candidate : parsed) {
          if (candidateHasTag(candidate, "vmi") &&
              candidateHasTag(candidate, "fusion_eligible") &&
              (hasStaticFullTileValidShape(op) ||
               candidateHasTag(candidate, "supports_partial_valid_shape"))) {
            recordSelection(op, candidate, true);
            return WalkResult::advance();
          }
        }
      }
      for (DictionaryAttr candidate : parsed) {
        if (!candidateHasTag(candidate, "vmi")) {
          recordSelection(op, candidate, false);
          return WalkResult::advance();
        }
      }
      op->emitError("no legal PTODSL TileLib candidate selected under policy '")
          << selectionPolicy << "'";
      return WalkResult::interrupt();
    });
    if (result.wasInterrupted())
      signalPassFailure();
  }
};

} // namespace

namespace mlir {
namespace pto {
std::unique_ptr<Pass> createSelectTemplateCandidatePass() {
  return std::make_unique<SelectTemplateCandidatePass>();
}
std::unique_ptr<Pass> createSelectTemplateCandidatePass(
    const SelectTemplateCandidateOptions &options) {
  return std::make_unique<SelectTemplateCandidatePass>(options);
}
} // namespace pto
} // namespace mlir
