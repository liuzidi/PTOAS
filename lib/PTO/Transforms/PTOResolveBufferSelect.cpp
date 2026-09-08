// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOResolveBufferSelect.cpp -----------------------------------------===//
//
// Lowering for tile-native buffer views and multi-buffer slot selection.
//
// Consumes tile-native `pto.subview` and `pto.multi_tile_get` operations after
// memory planning. Subviews rooted at `pto.alloc_tile` reuse the planned
// address, while subviews rooted at `pto.declare_tile` materialize the runtime
// address assigned by a preceding pipe operation. Constant multi-buffer slots
// become addressed `pto.alloc_tile` handles directly; dynamic slots select an
// address through an N-way `arith.select` chain. The user SSA remains the slot
// selector; this pass does not synthesize `iv mod N`.
//
//===----------------------------------------------------------------------===//

#include "PTO/Support/CodeConstants.h"
#include "PTO/IR/PTO.h"
#include "PTO/IR/PTOMultiBuffer.h"
#include "PTO/IR/PTOTypeUtils.h"
#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_PTORESOLVEBUFFERSELECT
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

#define DEBUG_TYPE "pto-resolve-buffer-select"

using namespace mlir;

namespace {

constexpr int64_t kSFractal1024 = 1024;
constexpr int64_t kSFractal512 = 512;
constexpr int64_t kSFractal32 = 32;
constexpr int64_t kFractalInnerDimension = 16;
constexpr int64_t kSFractal32InnerColumnCount = 2;
constexpr uint64_t kCubeTileAddressAlignmentBytes = 512;
constexpr uint64_t kVectorTileAddressAlignmentBytes = 32;
constexpr unsigned kI64BitWidth = 64;

static uint64_t alignUp(uint64_t value, uint64_t align) {
  if (align == 0) {
    return value;
  }
  return ((value + align - 1) / align) * align;
}

static Value ensureI64(Value value, IRRewriter &rewriter, Location loc) {
  if (!value) {
    return {};
  }
  if (value.getType().isInteger(kI64BitWidth)) {
    return value;
  }
  if (value.getType().isIndex()) {
    return rewriter.create<arith::IndexCastOp>(loc, rewriter.getI64Type(), value);
  }
  if (isa<IntegerType>(value.getType())) {
    return rewriter.create<arith::ExtSIOp>(loc, rewriter.getI64Type(), value);
  }
  return {};
}

static bool getTilePointerStrides(pto::TileBufType type, int64_t &rowStride,
                                  int64_t &colStride) {
  auto shape = type.getShape();
  if (shape.size() != mlir::pto::kValue2 || llvm::is_contained(shape, ShapedType::kDynamic)) {
    return false;
  }

  auto config = type.getConfigAttr();
  int32_t bl = static_cast<int32_t>(config.getBLayout().getValue());
  int32_t sl = static_cast<int32_t>(config.getSLayout().getValue());
  if (sl == 0) {
    bool rowPlusOne =
        type.getCompactModeI32() ==
        static_cast<int32_t>(pto::CompactMode::RowPlusOne);
    rowStride = bl == 1 ? 1 : shape[1] + (rowPlusOne ? 1 : 0);
    colStride = bl == 1 ? shape[0] + (rowPlusOne ? 1 : 0) : 1;
    return true;
  }

  unsigned elemBytes = pto::getPTOStorageElemByteSize(type.getElementType());
  if (elemBytes == 0) {
    return false;
  }
  int64_t innerRows = 1;
  int64_t innerCols = 1;
  int32_t fractal = config.getSFractalSize().getInt();
  if (fractal == kSFractal1024) {
    innerRows = kFractalInnerDimension;
    innerCols = kFractalInnerDimension;
  } else if (fractal == kSFractal32) {
    innerRows = kFractalInnerDimension;
    innerCols = kSFractal32InnerColumnCount;
  } else if (fractal == kSFractal512 && sl == 1) {
    innerRows = kFractalInnerDimension;
    innerCols = kSFractal32 / elemBytes;
  } else if (fractal == kSFractal512 && sl == 2) {
    innerRows = kSFractal32 / elemBytes;
    innerCols = kFractalInnerDimension;
  } else {
    return false;
  }

  if (bl == 1) {
    if (sl != 1) {
      return false;
    }
    rowStride = innerCols;
    colStride =
        shape[0] +
        (type.getCompactModeI32() ==
                 static_cast<int32_t>(pto::CompactMode::RowPlusOne)
             ? 1
             : 0);
  } else {
    rowStride =
        shape[1] +
        (type.getCompactModeI32() ==
                 static_cast<int32_t>(pto::CompactMode::RowPlusOne)
             ? 1
             : 0);
    colStride = innerRows;
  }
  return true;
}

static uint64_t getTileAddressAlignmentBytes(pto::TileBufType type) {
  auto addrSpace =
      dyn_cast_or_null<pto::AddressSpaceAttr>(type.getMemorySpace());
  if (!addrSpace) {
    return 1;
  }

  switch (addrSpace.getAddressSpace()) {
  case pto::AddressSpace::LEFT:
  case pto::AddressSpace::RIGHT:
  case pto::AddressSpace::ACC:
    return kCubeTileAddressAlignmentBytes;
  case pto::AddressSpace::VEC:
  case pto::AddressSpace::MAT:
  case pto::AddressSpace::BIAS:
  case pto::AddressSpace::SCALING:
    return kVectorTileAddressAlignmentBytes;
  case pto::AddressSpace::GM:
  case pto::AddressSpace::Zero:
    return 1;
  }
  return 1;
}

// Clone a pure integer-arithmetic address chain into the current insertion
// point. Values defined inside a fusion_region do not dominate uses after
// the region, so the outer subview materialization must recompute the
// address arithmetic outside instead of referencing region-local values.
// Constants and function-entry values are used directly; region-local
// arith ops are cloned recursively.
static Value cloneAddressMathOutsideRegion(Value value,
                                           IRRewriter &rewriter) {
  if (auto blockArg = dyn_cast<BlockArgument>(value)) {
    return blockArg;
  }
  Operation *defOp = value.getDefiningOp();
  if (!defOp) {
    return value;
  }
  if (isa<arith::ConstantOp, arith::ConstantIntOp, arith::ConstantIndexOp>(
          defOp)) {
    Operation *clone = rewriter.clone(*defOp);
    return clone->getResult(0);
  }
  if (isa<arith::AddIOp, arith::MulIOp, arith::IndexCastOp,
          arith::IndexCastUIOp, arith::ExtSIOp, arith::ExtUIOp>(defOp)) {
    SmallVector<Value, mlir::pto::kValue2> operands;
    for (Value operand : defOp->getOperands()) {
      operands.push_back(cloneAddressMathOutsideRegion(operand, rewriter));
    }
    OperationState state(defOp->getLoc(), defOp->getName());
    state.addTypes(defOp->getResultTypes());
    state.addOperands(operands);
    state.addAttributes(defOp->getAttrs());
    Operation *clone = rewriter.create(state);
    return clone->getResult(0);
  }
  // Not pure integer math (e.g. a tile_buf_addr on a declare_tile handle).
  // Such values are only produced outside regions; a region-internal
  // producer cannot be recovered here.
  return {};
}

static Value computeTileAddress(Value value, IRRewriter &rewriter,
                                Location loc) {
  if (auto alloc = value.getDefiningOp<pto::AllocTileOp>()) {
    return ensureI64(alloc.getAddr(), rewriter, loc);
  }
  // A fusion_region result yields a locally materialized tile handle. The
  // VMI fusion pipeline (PTOVmiLoopFusion) re-binds tile handles through
  // region results, so subviews anchored on them must recover the yielded
  // local alloc/declare handle to compute the runtime address. The handle's
  // address arithmetic is region-local, so it is cloned at the current
  // insertion point to preserve dominance for outer uses.
  if (auto regionResult = dyn_cast<OpResult>(value)) {
    if (auto fusionRegion =
            dyn_cast<pto::FusionRegionOp>(regionResult.getOwner())) {
      auto yieldOp = dyn_cast<pto::YieldOp>(
          fusionRegion.getBody().front().getTerminator());
      unsigned resultIndex = regionResult.getResultNumber();
      if (!yieldOp || resultIndex >= yieldOp.getOperands().size()) {
        return {};
      }
      Value yielded = yieldOp.getOperands()[resultIndex];
      if (auto alloc = yielded.getDefiningOp<pto::AllocTileOp>()) {
        if (!alloc.getAddr()) {
          return {};
        }
        return cloneAddressMathOutsideRegion(alloc.getAddr(), rewriter);
      }
      if (auto subview = yielded.getDefiningOp<pto::SubViewOp>()) {
        // The yielded handle is itself a subview that was already
        // materialized inside the region. Recompute its address chain from
        // the region-local anchor into the outer scope.
        Value base = computeTileAddress(subview.getSource(), rewriter, loc);
        auto sourceType = subview.getSource().getType();
        int64_t rowStride = 0;
        int64_t colStride = 0;
        if (!base || !getTilePointerStrides(sourceType, rowStride,
                                            colStride) ||
            subview.getOffsets().size() != mlir::pto::kValue2) {
          return {};
        }
        Value row = cloneAddressMathOutsideRegion(subview.getOffsets()[0],
                                                  rewriter);
        Value col = cloneAddressMathOutsideRegion(subview.getOffsets()[1],
                                                  rewriter);
        if (!row || !col) {
          return {};
        }
        row = rewriter.create<arith::MulIOp>(loc, row,
            rewriter.create<arith::ConstantIntOp>(loc, rowStride, 64));
        col = rewriter.create<arith::MulIOp>(loc, col,
            rewriter.create<arith::ConstantIntOp>(loc, colStride, 64));
        Value elements = rewriter.create<arith::AddIOp>(loc, row, col);
        int64_t elemBytes = static_cast<int64_t>(
            pto::getPTOStorageElemByteSize(sourceType.getElementType()));
        if (elemBytes == 0) {
          return {};
        }
        Value bytes = rewriter.create<arith::MulIOp>(
            loc, elements,
            rewriter.create<arith::ConstantIntOp>(loc, elemBytes, 64));
        return rewriter.create<arith::AddIOp>(loc, base, bytes);
      }
      return {};
    }
  }
  if (value.getDefiningOp<pto::DeclareTileOp>()) {
    auto tileType = dyn_cast<pto::TileBufType>(value.getType());
    if (!tileType) {
      return {};
    }
    auto memorySpace =
        dyn_cast_or_null<pto::AddressSpaceAttr>(tileType.getMemorySpace());
    if (!memorySpace) {
      return {};
    }
    auto ptrType = pto::PtrType::get(rewriter.getContext(),
                                     tileType.getElementType(), memorySpace);
    Value ptr =
        rewriter.create<pto::TileBufAddrOp>(loc, ptrType, value).getDst();
    return rewriter.create<pto::PtrToIntOp>(loc, ptr).getResult();
  }
  if (auto subview = value.getDefiningOp<pto::SubViewOp>()) {
    Value base = computeTileAddress(subview.getSource(), rewriter, loc);
    auto sourceType = subview.getSource().getType();
    int64_t rowStride = 0;
    int64_t colStride = 0;
    if (!base || !getTilePointerStrides(sourceType, rowStride, colStride) ||
        subview.getOffsets().size() != mlir::pto::kValue2) {
      return {};
    }
    Value row = ensureI64(subview.getOffsets()[0], rewriter, loc);
    Value col = ensureI64(subview.getOffsets()[1], rewriter, loc);
    if (!row || !col) {
      return {};
    }
    Value rowScale = rewriter.create<arith::ConstantIntOp>(loc, rowStride, 64);
    Value colScale = rewriter.create<arith::ConstantIntOp>(loc, colStride, 64);
    row = rewriter.create<arith::MulIOp>(loc, row, rowScale);
    col = rewriter.create<arith::MulIOp>(loc, col, colScale);
    Value elements = rewriter.create<arith::AddIOp>(loc, row, col);
    int64_t elemBytes = static_cast<int64_t>(
        pto::getPTOStorageElemByteSize(sourceType.getElementType()));
    if (elemBytes == 0) {
      return {};
    }
    Value byteScale = rewriter.create<arith::ConstantIntOp>(loc, elemBytes, 64);
    Value bytes = rewriter.create<arith::MulIOp>(loc, elements, byteScale);
    return rewriter.create<arith::AddIOp>(loc, base, bytes);
  }
  return {};
}

static pto::TileBufType getSubviewPhysicalType(pto::SubViewOp op) {
  pto::TileBufType sourceType = op.getSource().getType();
  pto::TileBufType resultType = op.getResult().getType();
  ArrayRef<int64_t> physicalShape = sourceType.getShape();

  int64_t inheritedRowStride = 0;
  int64_t inheritedColStride = 0;
  int64_t childRowStride = 0;
  int64_t childColStride = 0;
  auto compactChildType = pto::TileBufType::get(
      op.getContext(), resultType.getShape(), resultType.getElementType(),
      resultType.getMemorySpace(), resultType.getValidShape(),
      resultType.getConfigAttr());
  // If the child tile's compact layout has the same pointer stride as the
  // inherited subview, the addressed handle can use the child physical shape.
  // Otherwise keep the parent shape and express the logical slice via valid.
  if (getTilePointerStrides(sourceType, inheritedRowStride,
                            inheritedColStride) &&
      getTilePointerStrides(compactChildType, childRowStride, childColStride) &&
      inheritedRowStride == childRowStride &&
      inheritedColStride == childColStride) {
    physicalShape = resultType.getShape();
  }

  return pto::TileBufType::get(
      op.getContext(), physicalShape, resultType.getElementType(),
      resultType.getMemorySpace(), resultType.getValidShape(),
      resultType.getConfigAttr());
}

static Value getSubviewValidOperand(pto::SubViewOp op,
                                    pto::TileBufType physicalType,
                                    unsigned dim, IRRewriter &rewriter) {
  Value operand = dim == 0 ? op.getValidRow() : op.getValidCol();
  ArrayRef<int64_t> validShape = physicalType.getValidShape();
  if (validShape.size() <= dim || validShape[dim] >= 0) {
    return {};
  }
  if (operand) {
    return operand;
  }
  ArrayRef<int64_t> shape = physicalType.getShape();
  if (shape.size() > dim && shape[dim] != ShapedType::kDynamic) {
    return rewriter.create<arith::ConstantIndexOp>(op.getLoc(), shape[dim]);
  }
  return {};
}

static LogicalResult resolveTileNativeSubviews(ModuleOp module,
                                               MLIRContext *ctx) {
  SmallVector<pto::SubViewOp, mlir::pto::kValue16> subviews;
  module.walk([&](pto::SubViewOp op) { subviews.push_back(op); });
  for (pto::SubViewOp op : subviews) {
    IRRewriter rewriter(ctx);
    rewriter.setInsertionPoint(op);
    Value addr = computeTileAddress(op.getResult(), rewriter, op.getLoc());
    // A tile function argument is a symbolic runtime-bound handle. Keep its
    // subview tile-native; only planned local roots and declare_tile handles
    // assigned by preceding pipe operations can be normalized here.
    if (!addr) {
      continue;
    }
    pto::TileBufType physicalType = getSubviewPhysicalType(op);
    auto alloc = rewriter.create<pto::AllocTileOp>(
        op.getLoc(), physicalType, addr,
        getSubviewValidOperand(op, physicalType, 0, rewriter),
        getSubviewValidOperand(op, physicalType, 1, rewriter));
    alloc->setAttr("pto.view_semantics", rewriter.getStringAttr("subview"));
    rewriter.replaceOp(op, alloc.getResult());
  }
  return success();
}

static FailureOr<uint64_t> getStaticSlotBytes(pto::TileBufType slotType) {
  uint64_t elemBytes = pto::getPTOStorageElemByteSize(slotType.getElementType());
  if (elemBytes == 0) {
    return failure();
  }
  uint64_t bytes = elemBytes;
  for (int64_t dim : slotType.getShape()) {
    if (dim == ShapedType::kDynamic) {
      return failure();
    }
    bytes *= static_cast<uint64_t>(dim);
  }
  return bytes;
}

static LogicalResult getMultiTileAddresses(pto::AllocMultiTileOp alloc,
                                           IRRewriter &rewriter,
                                           SmallVectorImpl<Value> &addrs) {
  uint32_t count = alloc.getResult().getType().getCount();
  if (auto planned = alloc->getAttrOfType<DenseI64ArrayAttr>(
          pto::kPtoMultiBufferAddrsAttrName)) {
    if (planned.size() != count) {
      return alloc.emitError("planned address count does not match slot count");
    }
    for (int64_t address : planned.asArrayRef()) {
      addrs.push_back(rewriter.create<arith::ConstantIntOp>(
          alloc.getLoc(), address, kI64BitWidth));
    }
    return success();
  }

  Value base = alloc.getAddr();
  if (!base) {
    return alloc.emitError(
        "has neither a level3 base address nor planner-assigned slot addresses");
  }
  auto slotBytes = getStaticSlotBytes(alloc.getResult().getType().getSlotType());
  if (failed(slotBytes)) {
    return alloc.emitError(
        "requires a static slot shape and known element byte size");
  }

  uint64_t slotStride =
      alignUp(*slotBytes,
              getTileAddressAlignmentBytes(
                  alloc.getResult().getType().getSlotType()));

  addrs.push_back(base);
  for (uint32_t slot = 1; slot < count; ++slot) {
    Value offset = rewriter.create<arith::ConstantIntOp>(
        alloc.getLoc(), static_cast<int64_t>(slot * slotStride), 64);
    addrs.push_back(
        rewriter.create<arith::AddIOp>(alloc.getLoc(), base, offset));
  }
  return success();
}

static LogicalResult resolveTileNativeMultiGets(ModuleOp module,
                                                MLIRContext *ctx) {
  SmallVector<pto::MultiTileGetOp, mlir::pto::kValue8> gets;
  module.walk([&](pto::MultiTileGetOp op) { gets.push_back(op); });

  for (pto::MultiTileGetOp op : gets) {
    auto alloc = op.getSource().getDefiningOp<pto::AllocMultiTileOp>();
    if (!alloc) {
      return op.emitError(
          "currently requires a direct pto.alloc_multi_tile source");
    }

    IRRewriter rewriter(ctx);
    rewriter.setInsertionPoint(op);
    SmallVector<Value, mlir::pto::kValue8> addrs;
    if (failed(getMultiTileAddresses(alloc, rewriter, addrs))) {
      return failure();
    }

    Value selectedAddr;
    IntegerAttr constSlotAttr;
    if (matchPattern(op.getSlot(), m_Constant(&constSlotAttr))) {
      int64_t slot = constSlotAttr.getValue().getSExtValue();
      if (slot < 0 || slot >= static_cast<int64_t>(addrs.size())) {
        return op.emitError("constant slot is outside planned address range");
      }
      selectedAddr = addrs[static_cast<size_t>(slot)];
    } else {
      selectedAddr = addrs.front();
      for (uint32_t slot = 1; slot < addrs.size(); ++slot) {
        Value slotValue = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), slot);
        Value matches = rewriter.create<arith::CmpIOp>(
            op.getLoc(), arith::CmpIPredicate::eq, op.getSlot(), slotValue);
        selectedAddr = rewriter.create<arith::SelectOp>(
            op.getLoc(), matches, addrs[slot], selectedAddr);
      }
    }

    auto slotHandle = rewriter.create<pto::AllocTileOp>(
        op.getLoc(), op.getResult().getType(), selectedAddr,
        alloc.getValidRow() ? alloc.getValidRow() : Value(),
        alloc.getValidCol() ? alloc.getValidCol() : Value());
    rewriter.replaceOp(op, slotHandle.getResult());
  }

  SmallVector<pto::AllocMultiTileOp, mlir::pto::kValue8> allocs;
  module.walk([&](pto::AllocMultiTileOp op) { allocs.push_back(op); });
  for (pto::AllocMultiTileOp alloc : allocs) {
    if (!alloc.getResult().use_empty()) {
      return alloc.emitError(
          "has unsupported uses after resolving pto.multi_tile_get");
    }
    alloc.erase();
  }
  return success();
}

struct PTOResolveBufferSelectPass
    : public mlir::pto::impl::PTOResolveBufferSelectBase<
          PTOResolveBufferSelectPass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PTOResolveBufferSelectPass)

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    MLIRContext *ctx = &getContext();

    if (failed(resolveTileNativeMultiGets(mod, ctx))) {
      signalPassFailure();
      return;
    }
    if (failed(resolveTileNativeSubviews(mod, ctx))) {
      signalPassFailure();
      return;
    }
  }
};
} // namespace

std::unique_ptr<Pass> mlir::pto::createPTOResolveBufferSelectPass() {
  return std::make_unique<PTOResolveBufferSelectPass>();
}
