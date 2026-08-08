// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "Utils.h"
#include "PTO/IR/PTO.h"
#include "PTO/IR/PTOMultiBuffer.h"
#include "PTO/IR/PTOTypeUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include <limits>

#define DEBUG_TYPE "pto-utils"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")
#define DBGSNL() (llvm::dbgs() << "\n")

namespace mlir {
namespace pto {

static constexpr llvm::StringLiteral kFrontendPipeIdAttrName =
    "__pto.frontend_id";

FailureOr<bool> hasTFillPadExpandedPhysicalShape(TFillPadOp op) {
  auto srcType = dyn_cast<TileBufType>(op.getSrc().getType());
  auto dstType = dyn_cast<TileBufType>(op.getDst().getType());
  if (!srcType || !dstType || srcType.getRank() != dstType.getRank())
    return failure();

  bool expanded = false;
  for (auto [srcDim, dstDim] :
       llvm::zip_equal(srcType.getShape(), dstType.getShape())) {
    if (srcDim == dstDim)
      continue;
    if (ShapedType::isDynamic(srcDim) || ShapedType::isDynamic(dstDim) ||
        dstDim < srcDim)
      return failure();
    expanded = true;
  }
  return expanded;
}

static Value peelTFillPadStorageAlias(Value value) {
  constexpr unsigned kMaxDepth = 32;
  for (unsigned depth = 0; value && depth < kMaxDepth; ++depth) {
    Operation *def = value.getDefiningOp();
    if (!def)
      break;
    if (auto cast = dyn_cast<UnrealizedConversionCastOp>(def)) {
      if (cast.getNumOperands() != 1 || cast.getNumResults() != 1)
        break;
      value = cast.getOperand(0);
      continue;
    }
    if (auto bitcast = dyn_cast<BitcastOp>(def)) {
      value = bitcast.getSrc();
      continue;
    }
    if (auto reshape = dyn_cast<TReshapeOp>(def)) {
      value = reshape.getSrc();
      continue;
    }
    break;
  }
  return value;
}

static bool haveSameKnownTFillPadStartAddress(Value src, Value dst) {
  src = peelTFillPadStorageAlias(src);
  dst = peelTFillPadStorageAlias(dst);
  if (src == dst)
    return true;

  auto srcAlloc = src.getDefiningOp<AllocTileOp>();
  auto dstAlloc = dst.getDefiningOp<AllocTileOp>();
  if (!srcAlloc || !dstAlloc || !srcAlloc.getAddr() || !dstAlloc.getAddr())
    return false;

  Value srcAddr = srcAlloc.getAddr();
  Value dstAddr = dstAlloc.getAddr();
  if (srcAddr == dstAddr)
    return true;

  IntegerAttr srcConst;
  IntegerAttr dstConst;
  return matchPattern(srcAddr, m_Constant(&srcConst)) &&
         matchPattern(dstAddr, m_Constant(&dstConst)) &&
         srcConst.getValue() == dstConst.getValue();
}

FailureOr<TFillPadLoweringKind>
inferTFillPadLoweringKindAfterMemoryPlanning(TFillPadOp op) {
  FailureOr<bool> expanded = hasTFillPadExpandedPhysicalShape(op);
  if (failed(expanded))
    return failure();

  auto srcSpace = GetBufferSpaceAttr(op.getSrc());
  auto dstSpace = GetBufferSpaceAttr(op.getDst());
  bool isVec = srcSpace && dstSpace &&
               srcSpace->getAddressSpace() == AddressSpace::VEC &&
               dstSpace->getAddressSpace() == AddressSpace::VEC;

  if (*expanded) {
    if (!isVec)
      return failure();
    return TFillPadLoweringKind::Expand;
  }
  if (isVec &&
      haveSameKnownTFillPadStartAddress(op.getSrc(), op.getDst()))
    return TFillPadLoweringKind::InPlace;
  return TFillPadLoweringKind::Normal;
}

std::optional<PhysicalSectionKind>
inferPhysicalSectionKindFromPipe(Operation *op) {
  auto pipeOp = dyn_cast_or_null<OpPipeInterface>(op);
  if (!pipeOp) {
    return std::nullopt;
  }

  switch (pipeOp.getPipe()) {
  case PIPE::PIPE_M:
  case PIPE::PIPE_MTE1:
    return PhysicalSectionKind::Cube;
  case PIPE::PIPE_V:
  case PIPE::PIPE_V2:
  case PIPE::PIPE_S:
    return PhysicalSectionKind::Vector;
  default:
    return std::nullopt;
  }
}

func::ReturnOp getAssumedUniqueReturnOp(func::FuncOp funcOp) {
  func::ReturnOp returnOp;
  for (Block &b : funcOp.getBody()) {
    if (auto candidateOp = dyn_cast<func::ReturnOp>(b.getTerminator())) {
      if (returnOp) {
        return nullptr;
      }
      returnOp = candidateOp;
    }
  }
  return returnOp;
}

Value peelUnrealized(Value value) {
  if (auto castOp = value.getDefiningOp<UnrealizedConversionCastOp>())
    return castOp.getOperand(0);
  return value;
}

bool isScalarFixpipeQuant(FixpipeQuant quant) {
  switch (quant) {
  case FixpipeQuant::DEQF16Scalar:
  case FixpipeQuant::REQ8Scalar:
  case FixpipeQuant::QF322B8PreScalar:
  case FixpipeQuant::QF322F16PreScalar:
  case FixpipeQuant::QF322BF16PreScalar:
  case FixpipeQuant::QS322BF16PreScalar:
  case FixpipeQuant::QF322HIF8PreScalar:
  case FixpipeQuant::QF322FP8PreScalar:
    return true;
  default:
    return false;
  }
}

bool isVectorFixpipeQuant(FixpipeQuant quant) {
  switch (quant) {
  case FixpipeQuant::DEQF16Vec:
  case FixpipeQuant::REQ8Vec:
  case FixpipeQuant::QF322B8PreVec:
  case FixpipeQuant::QS322BF16PreVec:
    return true;
  default:
    return false;
  }
}

Operation *getPipeInitDef(Value pipeHandle) {
  pipeHandle = peelUnrealized(pipeHandle);
  return pipeHandle ? pipeHandle.getDefiningOp() : nullptr;
}

AccPushEpilogueAttr getPipeInitAccPushEpilogue(Operation *initOp) {
  if (auto init = dyn_cast_or_null<InitializeL2LPipeOp>(initOp))
    return init.getAccPushEpilogueAttr();
  if (auto init = dyn_cast_or_null<InitializeL2G2LPipeOp>(initOp))
    return init.getAccPushEpilogueAttr();
  return {};
}

std::optional<int32_t> getFrontendPipeIdFromInit(Operation *initOp) {
  if (!initOp)
    return std::nullopt;
  if (auto attr = initOp->getAttrOfType<IntegerAttr>(kFrontendPipeIdAttrName))
    return static_cast<int32_t>(attr.getInt());
  return std::nullopt;
}

std::optional<int32_t> getFrontendPipeIdFromHandle(Value pipeHandle) {
  return getFrontendPipeIdFromInit(getPipeInitDef(pipeHandle));
}

Type normalizePTOAddressSpaceForLLVM(Type type, Builder &builder) {
  if (auto memrefType = dyn_cast<MemRefType>(type)) {
    if (auto addressSpace = dyn_cast_or_null<AddressSpaceAttr>(
            memrefType.getMemorySpace()))
      return MemRefType::get(
          memrefType.getShape(), memrefType.getElementType(),
          memrefType.getLayout(),
          builder.getI64IntegerAttr(
              static_cast<int64_t>(addressSpace.getAddressSpace())));
  }
  if (auto memrefType = dyn_cast<UnrankedMemRefType>(type)) {
    if (auto addressSpace = dyn_cast_or_null<AddressSpaceAttr>(
            memrefType.getMemorySpace()))
      return UnrankedMemRefType::get(
          memrefType.getElementType(),
          builder.getI64IntegerAttr(
              static_cast<int64_t>(addressSpace.getAddressSpace())));
  }
  return type;
}

void normalizePTOAddressSpacesForLLVM(ModuleOp module) {
  Builder builder(module.getContext());
  module.walk([&](Operation *op) {
    for (OpResult result : op->getResults()) {
      Type normalized = normalizePTOAddressSpaceForLLVM(result.getType(), builder);
      if (normalized != result.getType())
        result.setType(normalized);
    }
    for (Region &region : op->getRegions())
      for (Block &block : region)
        for (BlockArgument argument : block.getArguments()) {
          Type normalized =
              normalizePTOAddressSpaceForLLVM(argument.getType(), builder);
          if (normalized != argument.getType())
            argument.setType(normalized);
        }
    if (auto funcOp = dyn_cast<func::FuncOp>(op)) {
      FunctionType oldType = funcOp.getFunctionType();
      SmallVector<Type> inputs, results;
      bool changed = false;
      for (Type input : oldType.getInputs()) {
        Type normalized = normalizePTOAddressSpaceForLLVM(input, builder);
        changed |= normalized != input;
        inputs.push_back(normalized);
      }
      for (Type result : oldType.getResults()) {
        Type normalized = normalizePTOAddressSpaceForLLVM(result, builder);
        changed |= normalized != result;
        results.push_back(normalized);
      }
      if (changed)
        funcOp.setFunctionType(builder.getFunctionType(inputs, results));
    }
  });
}

namespace {

struct NormalizePTOAddressSpacesForLLVMPass final
    : public PassWrapper<NormalizePTOAddressSpacesForLLVMPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      NormalizePTOAddressSpacesForLLVMPass)

  void runOnOperation() override {
    normalizePTOAddressSpacesForLLVM(getOperation());
  }
};

} // namespace

std::unique_ptr<Pass> createNormalizePTOAddressSpacesForLLVMPass() {
  return std::make_unique<NormalizePTOAddressSpacesForLLVMPass>();
}

void legalizeIndexUnrealizedCasts(ModuleOp module) {
  // The A5 LLVM pipeline is configured with a 64-bit index type.  Only
  // i64<->index boundaries are therefore rewritten here; converting an
  // arbitrary-width integer to index would silently change semantics on a
  // target whose index width differs from that integer width.
  SmallVector<UnrealizedConversionCastOp> casts;
  module.walk([&](UnrealizedConversionCastOp op) {
    if (op->getNumOperands() != 1 || op->getNumResults() != 1)
      return;
    Type sourceType = op.getOperand(0).getType();
    Type resultType = op.getResult(0).getType();
    auto isI64 = [](Type type) {
      auto integerType = dyn_cast<IntegerType>(type);
      return integerType && integerType.getWidth() == 64;
    };
    if ((sourceType.isIndex() && isI64(resultType)) ||
        (isI64(sourceType) && resultType.isIndex()))
      casts.push_back(op);
  });

  for (UnrealizedConversionCastOp op : casts) {
    OpBuilder builder(op);
    Value converted = builder.create<arith::IndexCastOp>(
        op.getLoc(), op.getResult(0).getType(), op.getOperand(0));
    op.getResult(0).replaceAllUsesWith(converted);
    op.erase();
  }
}

void cleanupPTOArtifactsAfterLLVMLowering(ModuleOp module) {
  while (true) {
    SmallVector<AllocTileOp> deadAllocs;
    module.walk([&](AllocTileOp op) {
      if (op.use_empty())
        deadAllocs.push_back(op);
    });
    if (deadAllocs.empty())
      break;
    for (AllocTileOp op : llvm::reverse(deadAllocs))
      op.erase();
  }

  SmallVector<UnrealizedConversionCastOp> removableCasts;
  module.walk([&](UnrealizedConversionCastOp op) {
    if (op.use_empty()) {
      removableCasts.push_back(op);
      return;
    }
    if (op->getNumOperands() == 1 && op->getNumResults() == 1 &&
        op.getOperand(0).getType() == op.getResult(0).getType())
      removableCasts.push_back(op);
  });
  for (UnrealizedConversionCastOp op : llvm::reverse(removableCasts)) {
    if (!op.use_empty())
      op.getResult(0).replaceAllUsesWith(op.getOperand(0));
    op.erase();
  }
}

namespace {

struct A5L2LPipeLoweringState {
  InitializeL2LPipeOp init;
  func::FuncOp func;
  FunctionKernelKind kernelKind;
  Value producerIndex;
  Value consumerIndex;
};

static Value getI64Constant(OpBuilder &builder, Location loc, int64_t value) {
  return builder.create<arith::ConstantIntOp>(loc, value, 64);
}

static FailureOr<Value> castUnsignedToI64(OpBuilder &builder, Location loc,
                                          Value value) {
  if (!value)
    return failure();
  if (value.getType().isInteger(64))
    return value;
  if (value.getType().isIndex())
    return builder.create<arith::IndexCastUIOp>(loc, builder.getI64Type(),
                                                 value)
        .getResult();
  auto integerType = dyn_cast<IntegerType>(value.getType());
  if (!integerType || integerType.getWidth() > 64)
    return failure();
  return builder.create<arith::ExtUIOp>(loc, builder.getI64Type(), value)
      .getResult();
}

static FailureOr<unsigned> getElementByteWidth(Type type) {
  if (auto integerType = dyn_cast<IntegerType>(type)) {
    if (integerType.getWidth() % 8 != 0)
      return failure();
    return integerType.getWidth() / 8;
  }
  if (auto floatType = dyn_cast<FloatType>(type)) {
    if (floatType.getWidth() % 8 != 0)
      return failure();
    return floatType.getWidth() / 8;
  }
  return failure();
}

static AddressSpaceAttr getAddressSpaceAttr(MLIRContext *context,
                                            AddressSpace addressSpace) {
  return AddressSpaceAttr::get(context, addressSpace);
}

static Value createPtr(OpBuilder &builder, Location loc, Value address,
                       Type elementType, AddressSpace addressSpace) {
  auto ptrType = PtrType::get(builder.getContext(), elementType,
                              getAddressSpaceAttr(builder.getContext(),
                                                  addressSpace));
  return builder.create<CastPtrOp>(loc, ptrType, address).getResult();
}

static PipeAttr getPipeAttr(OpBuilder &builder, PIPE pipe) {
  return PipeAttr::get(builder.getContext(), pipe);
}

static void emitSyncSet(OpBuilder &builder, Location loc, PIPE pipe,
                        int64_t eventId) {
  builder.create<SyncSetOp>(loc, getPipeAttr(builder, pipe),
                            builder.getI32IntegerAttr(eventId), IntegerAttr{},
                            Value{});
}

static void emitSyncWait(OpBuilder &builder, Location loc, PIPE pipe,
                         int64_t eventId) {
  builder.create<SyncWaitOp>(loc, getPipeAttr(builder, pipe),
                             builder.getI32IntegerAttr(eventId), Value{});
}

template <typename EmitFn>
static void emitIf(OpBuilder &builder, Location loc, Value condition,
                   EmitFn &&emit) {
  OpBuilder::InsertionGuard guard(builder);
  auto ifOp = builder.create<scf::IfOp>(loc, TypeRange{}, condition,
                                        /*withElseRegion=*/false);
  builder.setInsertionPointToStart(ifOp.thenBlock());
  emit();
}

static Value loadPipeIndex(OpBuilder &builder, Location loc, Value state) {
  return builder.create<LLVM::LoadOp>(loc, builder.getI64Type(), state);
}

static void storePipeIndex(OpBuilder &builder, Location loc, Value state,
                           Value value) {
  builder.create<LLVM::StoreOp>(loc, value, state);
}

static Value computeRingAddress(OpBuilder &builder, Location loc, Value base,
                                Value index, uint32_t slotNum,
                                uint32_t slotSize) {
  Value slot = builder.create<arith::RemUIOp>(
      loc, index, getI64Constant(builder, loc, slotNum));
  Value byteOffset = builder.create<arith::MulIOp>(
      loc, slot, getI64Constant(builder, loc, slotSize));
  return builder.create<arith::AddIOp>(loc, base, byteOffset);
}

static LogicalResult validateCommonPipeForm(InitializeL2LPipeOp init) {
  ModuleOp module = init->getParentOfType<ModuleOp>();
  auto arch = module ? module->getAttrOfType<StringAttr>("pto.target_arch")
                     : StringAttr{};
  if (!arch || !arch.getValue().equals_insensitive("a5"))
    return init.emitOpError(
        "LLVM unified L2L lowering currently requires target_arch = a5");
  if (!init.getFlagBaseAttr())
    return init.emitOpError(
        "LLVM unified L2L lowering requires an explicit flag_base");
  if (!init.getNosplitAttr() || !init.getNosplitAttr().getValue())
    return init.emitOpError(
        "LLVM unified L2L lowering currently requires nosplit = true");
  if (init.getDirMask() != 1 && init.getDirMask() != 2 &&
      init.getDirMask() != 3)
    return init.emitOpError(
        "LLVM unified L2L lowering supports only dir_mask 1, 2, or 3");
  return success();
}

static FailureOr<FunctionKernelKind>
getLoweringKernelKind(InitializeL2LPipeOp init) {
  func::FuncOp func = init->getParentOfType<func::FuncOp>();
  if (!func)
    return failure();
  auto kindAttr = func->getAttrOfType<FunctionKernelKindAttr>(
      FunctionKernelKindAttr::name);
  if (!kindAttr)
    return failure();
  return kindAttr.getKernelKind();
}

static LogicalResult initializeIndexState(A5L2LPipeLoweringState &state) {
  if (state.func.empty())
    return state.init.emitOpError(
        "LLVM unified L2L lowering requires a defined function");
  OpBuilder builder(state.func.getContext());
  Block &entry = state.func.front();
  builder.setInsertionPointToStart(&entry);
  Location loc = state.init.getLoc();
  auto ptrType = LLVM::LLVMPointerType::get(builder.getContext());
  state.producerIndex = builder.create<LLVM::AllocaOp>(
      loc, ptrType, builder.getI64Type(), getI64Constant(builder, loc, 1),
      /*alignment=*/8);
  state.consumerIndex = builder.create<LLVM::AllocaOp>(
      loc, ptrType, builder.getI64Type(), getI64Constant(builder, loc, 1),
      /*alignment=*/8);
  Value zero = getI64Constant(builder, loc, 0);
  storePipeIndex(builder, loc, state.producerIndex, zero);
  storePipeIndex(builder, loc, state.consumerIndex, zero);
  return success();
}

static bool isCube(FunctionKernelKind kind) {
  return kind == FunctionKernelKind::Cube;
}

static bool isVector(FunctionKernelKind kind) {
  return kind == FunctionKernelKind::Vector;
}

static bool isC2VProducer(A5L2LPipeLoweringState &state) {
  return isCube(state.kernelKind) &&
         (state.init.getDirMask() == 1 || state.init.getDirMask() == 3);
}

static bool isC2VConsumer(A5L2LPipeLoweringState &state) {
  return isVector(state.kernelKind) &&
         (state.init.getDirMask() == 1 || state.init.getDirMask() == 3);
}

static bool isV2CProducer(A5L2LPipeLoweringState &state) {
  return isVector(state.kernelKind) &&
         (state.init.getDirMask() == 2 || state.init.getDirMask() == 3);
}

static bool isV2CConsumer(A5L2LPipeLoweringState &state) {
  return isCube(state.kernelKind) &&
         (state.init.getDirMask() == 2 || state.init.getDirMask() == 3);
}

static int64_t getFlagBase(InitializeL2LPipeOp init) {
  return init.getFlagBaseAttr().getInt();
}

static uint32_t getSyncPeriod(InitializeL2LPipeOp init) {
  return init.getSlotNum() <= 2 ? init.getSlotNum() : init.getSlotNum() / 2;
}

static void emitPipeConstructionSync(A5L2LPipeLoweringState &state) {
  OpBuilder builder(state.init);
  Location loc = state.init.getLoc();
  int64_t flag = getFlagBase(state.init);
  uint32_t syncPeriod = getSyncPeriod(state.init);
  for (uint32_t i = 0; i < syncPeriod; ++i) {
    if (isC2VConsumer(state))
      emitSyncSet(builder, loc, PIPE::PIPE_V, flag + 1);
    if (isV2CConsumer(state))
      emitSyncSet(builder, loc, PIPE::PIPE_MTE1,
                  flag + (state.init.getDirMask() == 3 ? 3 : 1));
  }
}

static void emitPipeDestructionSync(A5L2LPipeLoweringState &state) {
  SmallVector<func::ReturnOp> returns;
  state.func.walk([&](func::ReturnOp op) { returns.push_back(op); });
  int64_t flag = getFlagBase(state.init);
  uint32_t syncPeriod = getSyncPeriod(state.init);
  for (func::ReturnOp returnOp : returns) {
    OpBuilder builder(returnOp);
    for (uint32_t i = 0; i < syncPeriod; ++i) {
      if (isC2VProducer(state))
        emitSyncWait(builder, returnOp.getLoc(), PIPE::PIPE_FIX, flag + 1);
      if (isV2CProducer(state))
        emitSyncWait(builder, returnOp.getLoc(), PIPE::PIPE_MTE3,
                     flag + (state.init.getDirMask() == 3 ? 3 : 1));
    }
  }
}

static FailureOr<Value> getPipeLocalBase(A5L2LPipeLoweringState &state,
                                         AddressSpace addressSpace,
                                         OpBuilder &builder, Location loc) {
  Value address;
  if (addressSpace == AddressSpace::VEC) {
    if (state.init.getDirMask() != 1 && state.init.getDirMask() != 3)
      return failure();
    address = state.init.getLocalAddr();
  } else if (addressSpace == AddressSpace::MAT) {
    if (state.init.getDirMask() == 2)
      address = state.init.getLocalAddr();
    else if (state.init.getDirMask() == 3)
      address = state.init.getPeerLocalAddr();
    else
      return failure();
  } else {
    return failure();
  }
  return castUnsignedToI64(builder, loc, address);
}

static FailureOr<std::pair<Value, Value>>
getValidShapeI64(OpBuilder &builder, Location loc, AllocTileOp alloc,
                 TileBufType tileType) {
  ArrayRef<int64_t> shape = tileType.getShape();
  if (shape.size() != 2)
    return failure();
  Value row;
  Value col;
  if (alloc.getValidRow()) {
    FailureOr<Value> converted =
        castUnsignedToI64(builder, loc, alloc.getValidRow());
    if (failed(converted))
      return failure();
    row = *converted;
  } else if (!ShapedType::isDynamic(shape[0])) {
    row = getI64Constant(builder, loc, shape[0]);
  }
  if (alloc.getValidCol()) {
    FailureOr<Value> converted =
        castUnsignedToI64(builder, loc, alloc.getValidCol());
    if (failed(converted))
      return failure();
    col = *converted;
  } else if (!ShapedType::isDynamic(shape[1])) {
    col = getI64Constant(builder, loc, shape[1]);
  }
  if (!row || !col)
    return failure();
  return std::make_pair(row, col);
}

static Value ceilDivUI(OpBuilder &builder, Location loc, Value value,
                       uint64_t divisor) {
  Value divisorValue = getI64Constant(builder, loc, divisor);
  Value adjusted = builder.create<arith::AddIOp>(
      loc, value, getI64Constant(builder, loc, divisor - 1));
  return builder.create<arith::DivUIOp>(loc, adjusted, divisorValue);
}

static LogicalResult emitAccToVecCopy(TPushOp push,
                                      A5L2LPipeLoweringState &state,
                                      AllocTileOp alloc, Value ringAddress,
                                      OpBuilder &builder) {
  auto tileType = dyn_cast<TileBufType>(push.getTile().getType());
  if (!tileType || tileType.getShape().size() != 2)
    return push.emitOpError(
        "ACC-to-VEC LLVM FIFO lowering requires a rank-2 ACC producer tile");
  Type elementType = tileType.getElementType();
  bool supported = elementType.isF32();
  if (auto intType = dyn_cast<IntegerType>(elementType))
    supported |= intType.getWidth() == 32;
  if (!supported)
    return push.emitOpError(
        "ACC-to-VEC LLVM FIFO lowering supports only f32 or i32");
  FailureOr<std::pair<Value, Value>> valid =
      getValidShapeI64(builder, push.getLoc(), alloc, tileType);
  if (failed(valid))
    return push.emitOpError("failed to resolve ACC producer valid shape");
  Value validRow = valid->first;
  Value validCol = valid->second;
  int64_t dstStride = tileType.getShape()[1];
  if (ShapedType::isDynamic(dstStride))
    return push.emitOpError("requires a static ACC-to-VEC destination stride");
  Value srcStride = builder.create<arith::MulIOp>(
      push.getLoc(), ceilDivUI(builder, push.getLoc(), validRow, 16),
      getI64Constant(builder, push.getLoc(), 16));
  Value config0 = builder.create<arith::ShLIOp>(
      push.getLoc(), validCol, getI64Constant(builder, push.getLoc(), 4));
  config0 = builder.create<arith::OrIOp>(
      push.getLoc(), config0,
      builder.create<arith::ShLIOp>(
          push.getLoc(), validRow,
          getI64Constant(builder, push.getLoc(), 16)));
  config0 = builder.create<arith::OrIOp>(
      push.getLoc(), config0,
      builder.create<arith::ShLIOp>(
          push.getLoc(), getI64Constant(builder, push.getLoc(), dstStride),
          getI64Constant(builder, push.getLoc(), 32)));
  Value config1 = builder.create<arith::OrIOp>(
      push.getLoc(), srcStride,
      getI64Constant(builder, push.getLoc(), int64_t{1} << 43));
  Value source = createPtr(builder, push.getLoc(), alloc.getAddr(), elementType,
                           AddressSpace::ACC);
  Value destination = createPtr(builder, push.getLoc(), ringAddress,
                                elementType, AddressSpace::VEC);
  builder.create<CopyMatrixCcToUbOp>(push.getLoc(), source, destination,
                                     config0, config1);
  return success();
}

static LogicalResult emitVecToMatCopy(TPushOp push,
                                      A5L2LPipeLoweringState &state,
                                      AllocTileOp alloc, Value ringAddress,
                                      OpBuilder &builder) {
  (void)state;
  auto tileType = dyn_cast<TileBufType>(push.getTile().getType());
  if (!tileType || tileType.getShape().size() != 2 ||
      tileType.getBLayoutValueI32() != static_cast<int32_t>(BLayout::ColMajor) ||
      tileType.getSLayoutValueI32() != static_cast<int32_t>(SLayout::RowMajor))
    return push.emitOpError(
        "VEC-to-MAT LLVM FIFO lowering requires a rank-2 NZ tile");
  FailureOr<unsigned> elementBytes =
      getElementByteWidth(tileType.getElementType());
  if (failed(elementBytes) || *elementBytes == 0 || 32 % *elementBytes != 0)
    return push.emitOpError("unsupported VEC-to-MAT FIFO element type");
  FailureOr<std::pair<Value, Value>> valid =
      getValidShapeI64(builder, push.getLoc(), alloc, tileType);
  if (failed(valid))
    return push.emitOpError("failed to resolve VEC producer valid shape");
  Value validRow = valid->first;
  Value validCol = valid->second;
  int64_t sourceRows = tileType.getShape()[0];
  if (ShapedType::isDynamic(sourceRows))
    return push.emitOpError("requires a static VEC-to-MAT row stride");
  uint64_t c0 = 32 / *elementBytes;
  Value nBurst = ceilDivUI(builder, push.getLoc(), validCol, c0);
  Value lenBurst = builder.create<arith::MulIOp>(
      push.getLoc(), validRow, getI64Constant(builder, push.getLoc(), c0));
  lenBurst = builder.create<arith::MulIOp>(
      push.getLoc(), lenBurst,
      getI64Constant(builder, push.getLoc(), *elementBytes));
  lenBurst = builder.create<arith::DivUIOp>(
      push.getLoc(), lenBurst, getI64Constant(builder, push.getLoc(), 32));
  Value rowExtent = getI64Constant(builder, push.getLoc(), sourceRows);
  Value srcGap =
      builder.create<arith::SubIOp>(push.getLoc(), rowExtent, validRow);
  Value dstGap =
      builder.create<arith::SubIOp>(push.getLoc(), rowExtent, validRow);
  Value source = createPtr(builder, push.getLoc(), alloc.getAddr(),
                           tileType.getElementType(), AddressSpace::VEC);
  Value destination = createPtr(builder, push.getLoc(), ringAddress,
                                tileType.getElementType(), AddressSpace::MAT);
  builder.create<CopyUbufToCbufOp>(
      push.getLoc(), source, destination,
      getI64Constant(builder, push.getLoc(), 0), nBurst, lenBurst, srcGap,
      dstGap);
  return success();
}

static LogicalResult lowerPush(TPushOp push,
                               A5L2LPipeLoweringState &state) {
  if (push.getSplit() != 0 || push.getAivSubblockid())
    return push.emitOpError(
        "LLVM unified L2L lowering currently supports only split = 0");
  auto alloc = push.getTile().getDefiningOp<AllocTileOp>();
  auto tileType = dyn_cast<TileBufType>(push.getTile().getType());
  if (!alloc || !alloc.getAddr() || !tileType)
    return push.emitOpError(
        "LLVM unified L2L tpush requires an explicitly-addressed alloc_tile");
  if (!alloc.getResult().hasOneUse())
    return push.emitOpError(
        "LLVM unified L2L tpush requires a uniquely-used alloc_tile producer");
  auto addressSpace = dyn_cast_or_null<AddressSpaceAttr>(
      tileType.getMemorySpace());
  if (!addressSpace)
    return push.emitOpError("tpush tile requires a PTO address space");
  bool accToVec = addressSpace.getAddressSpace() == AddressSpace::ACC;
  bool vecToMat = addressSpace.getAddressSpace() == AddressSpace::VEC;
  if ((accToVec && !isC2VProducer(state)) ||
      (vecToMat && !isV2CProducer(state)) || (!accToVec && !vecToMat))
    return push.emitOpError(
        "tpush tile address space does not match this kernel's FIFO role");

  OpBuilder builder(push);
  Location loc = push.getLoc();
  Value index = loadPipeIndex(builder, loc, state.producerIndex);
  uint32_t syncPeriod = getSyncPeriod(state.init);
  int64_t flag = getFlagBase(state.init);
  PIPE waitPipe = accToVec ? PIPE::PIPE_FIX : PIPE::PIPE_MTE3;
  int64_t waitFlag =
      flag + (accToVec ? 1 : (state.init.getDirMask() == 3 ? 3 : 1));
  if (state.init.getSlotNum() == 1) {
    emitSyncWait(builder, loc, waitPipe, waitFlag);
  } else {
    Value full = builder.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::uge, index,
        getI64Constant(builder, loc, state.init.getSlotNum()));
    Value periodic = builder.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq,
        builder.create<arith::RemUIOp>(
            loc, index, getI64Constant(builder, loc, syncPeriod)),
        getI64Constant(builder, loc, 0));
    Value shouldWait = builder.create<arith::AndIOp>(loc, full, periodic);
    emitIf(builder, loc, shouldWait,
           [&] { emitSyncWait(builder, loc, waitPipe, waitFlag); });
    builder.setInsertionPoint(push);
  }

  FailureOr<Value> base = getPipeLocalBase(
      state, accToVec ? AddressSpace::VEC : AddressSpace::MAT, builder, loc);
  if (failed(base))
    return push.emitOpError("failed to resolve FIFO local base address");
  Value ringAddress = computeRingAddress(builder, loc, *base, index,
                                         state.init.getSlotNum(),
                                         state.init.getSlotSize());
  if (accToVec) {
    if (failed(emitAccToVecCopy(push, state, alloc, ringAddress, builder)))
      return failure();
  } else if (failed(
                 emitVecToMatCopy(push, state, alloc, ringAddress, builder))) {
    return failure();
  }
  Value next = builder.create<arith::AddIOp>(
      loc, index, getI64Constant(builder, loc, 1));
  storePipeIndex(builder, loc, state.producerIndex, next);
  emitSyncSet(builder, loc, accToVec ? PIPE::PIPE_FIX : PIPE::PIPE_MTE3,
              flag + (accToVec ? 0
                               : (state.init.getDirMask() == 3 ? 2 : 0)));
  push.erase();
  if (!alloc.use_empty())
    return alloc.emitOpError("still has users after lowering its tpush");
  alloc.erase();
  return success();
}

static LogicalResult replaceDeclaredTilePointerUsers(
    TPopOp pop, MaterializeTileOp materialize, Value ringAddress) {
  auto declaration =
      materialize.getSource().getDefiningOp<DeclareTileMemRefOp>();
  if (!declaration)
    return pop.emitOpError(
        "LLVM unified L2L tpop requires declare_tile_memref backing");
  auto tileType = dyn_cast<TileBufType>(materialize.getResult().getType());
  auto tileSpace = tileType ? dyn_cast_or_null<AddressSpaceAttr>(
                                  tileType.getMemorySpace())
                            : AddressSpaceAttr{};
  auto memrefSpace = dyn_cast_or_null<AddressSpaceAttr>(
      declaration.getResult().getType().getMemorySpace());
  if (!tileType || !tileSpace || tileSpace != memrefSpace)
    return pop.emitOpError(
        "declared consumer memref and tile must use the same PTO address space");

  SmallVector<Operation *> pointerBridges;
  for (Operation *user : declaration.getResult().getUsers()) {
    if (user == materialize.getOperation())
      continue;
    bool isPointerBridge = isa<CastPtrOp>(user);
    if (auto bridge = dyn_cast<UnrealizedConversionCastOp>(user))
      isPointerBridge = bridge->getNumOperands() == 1 &&
                        bridge->getNumResults() == 1;
    if (!isPointerBridge || user->getNumResults() != 1 ||
        !isa<PtrType>(user->getResult(0).getType()))
      return pop.emitOpError(
          "declare_tile_memref has an unsupported non-pointer user");
    pointerBridges.push_back(user);
  }

  DominanceInfo dominance(pop->getParentOfType<func::FuncOp>());
  for (Operation *bridge : pointerBridges) {
    auto ptrType = cast<PtrType>(bridge->getResult(0).getType());
    if (ptrType.getElementType() != tileType.getElementType() ||
        ptrType.getMemorySpace() != tileSpace)
      return pop.emitOpError(
          "declared tile pointer bridge type does not match the consumer tile");
    for (Operation *user : bridge->getResult(0).getUsers())
      if (!dominance.dominates(pop.getOperation(), user))
        return pop.emitOpError(
            "declared tile pointer is used before its tpop assigns a FIFO slot");
  }

  OpBuilder builder(pop);
  Value pointer = builder.create<CastPtrOp>(
      pop.getLoc(),
      PtrType::get(builder.getContext(), tileType.getElementType(), tileSpace),
      ringAddress);
  for (Operation *bridge : pointerBridges) {
    bridge->getResult(0).replaceAllUsesWith(pointer);
    bridge->erase();
  }
  return success();
}

static LogicalResult lowerPop(TPopOp pop,
                              A5L2LPipeLoweringState &state) {
  if (pop.getSplit() != 0 || pop.getAivSubblockid())
    return pop.emitOpError(
        "LLVM unified L2L lowering currently supports only split = 0");
  auto materialize = pop.getTile().getDefiningOp<MaterializeTileOp>();
  auto tileType = dyn_cast<TileBufType>(pop.getTile().getType());
  if (!materialize || !tileType || !materialize.getResult().hasOneUse())
    return pop.emitOpError(
        "tpop requires a uniquely-used materialize_tile consumer handle");
  auto addressSpace = dyn_cast_or_null<AddressSpaceAttr>(
      tileType.getMemorySpace());
  if (!addressSpace)
    return pop.emitOpError("tpop tile requires a PTO address space");
  bool vecConsumer = addressSpace.getAddressSpace() == AddressSpace::VEC;
  bool matConsumer = addressSpace.getAddressSpace() == AddressSpace::MAT;
  if ((vecConsumer && !isC2VConsumer(state)) ||
      (matConsumer && !isV2CConsumer(state)) ||
      (!vecConsumer && !matConsumer))
    return pop.emitOpError(
        "tpop tile address space does not match this kernel's FIFO role");

  OpBuilder builder(pop);
  Location loc = pop.getLoc();
  int64_t flag = getFlagBase(state.init);
  emitSyncWait(builder, loc,
               vecConsumer ? PIPE::PIPE_V : PIPE::PIPE_MTE1,
               flag + (vecConsumer ? 0
                                   : (state.init.getDirMask() == 3 ? 2 : 0)));
  Value index = loadPipeIndex(builder, loc, state.consumerIndex);
  FailureOr<Value> base = getPipeLocalBase(
      state, vecConsumer ? AddressSpace::VEC : AddressSpace::MAT, builder, loc);
  if (failed(base))
    return pop.emitOpError("failed to resolve FIFO consumer base address");
  Value ringAddress = computeRingAddress(builder, loc, *base, index,
                                         state.init.getSlotNum(),
                                         state.init.getSlotSize());
  if (failed(replaceDeclaredTilePointerUsers(pop, materialize, ringAddress)))
    return failure();
  Value next = builder.create<arith::AddIOp>(
      loc, index, getI64Constant(builder, loc, 1));
  storePipeIndex(builder, loc, state.consumerIndex, next);
  pop.erase();
  if (!materialize.use_empty())
    return materialize.emitOpError(
        "materialize_tile still has users after lowering its tpop");
  auto declaration =
      materialize.getSource().getDefiningOp<DeclareTileMemRefOp>();
  materialize.erase();
  if (declaration) {
    if (!declaration.use_empty())
      return declaration.emitOpError(
          "declare_tile_memref still has users after FIFO pointer replacement");
    declaration.erase();
  }
  return success();
}

static LogicalResult lowerFree(TFreeOp free,
                               A5L2LPipeLoweringState &state) {
  if (free.getSplit() != 0)
    return free.emitOpError(
        "LLVM unified L2L lowering currently supports only split = 0");
  if (!isC2VConsumer(state) && !isV2CConsumer(state))
    return free.emitOpError("tfree appears in a non-consumer kernel");
  OpBuilder builder(free);
  Location loc = free.getLoc();
  Value index = loadPipeIndex(builder, loc, state.consumerIndex);
  uint32_t syncPeriod = getSyncPeriod(state.init);
  int64_t flag = getFlagBase(state.init);
  PIPE pipe = isC2VConsumer(state) ? PIPE::PIPE_V : PIPE::PIPE_MTE1;
  int64_t eventId =
      flag + (isC2VConsumer(state)
                  ? 1
                  : (state.init.getDirMask() == 3 ? 3 : 1));
  if (state.init.getSlotNum() == 1) {
    emitSyncSet(builder, loc, pipe, eventId);
  } else {
    Value shouldFree = builder.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq,
        builder.create<arith::RemUIOp>(
            loc, index, getI64Constant(builder, loc, syncPeriod)),
        getI64Constant(builder, loc, 0));
    emitIf(builder, loc, shouldFree,
           [&] { emitSyncSet(builder, loc, pipe, eventId); });
  }
  free.erase();
  return success();
}

static LogicalResult lowerOnePipe(InitializeL2LPipeOp init) {
  if (failed(validateCommonPipeForm(init)))
    return failure();
  FailureOr<FunctionKernelKind> kernelKind = getLoweringKernelKind(init);
  if (failed(kernelKind))
    return init.emitOpError(
        "LLVM unified L2L lowering requires pto.kernel_kind on the function");
  A5L2LPipeLoweringState state{init, init->getParentOfType<func::FuncOp>(),
                               *kernelKind, Value{}, Value{}};

  SmallVector<TPushOp> pushes;
  SmallVector<TPopOp> pops;
  SmallVector<TFreeOp> frees;
  for (Operation *user : init.getPipe().getUsers()) {
    if (auto push = dyn_cast<TPushOp>(user))
      pushes.push_back(push);
    else if (auto pop = dyn_cast<TPopOp>(user))
      pops.push_back(pop);
    else if (auto free = dyn_cast<TFreeOp>(user))
      frees.push_back(free);
    else
      return init.emitOpError()
             << "has unsupported LLVM FIFO user " << user->getName();
  }
  if (failed(initializeIndexState(state)))
    return failure();
  emitPipeConstructionSync(state);
  emitPipeDestructionSync(state);
  for (TPushOp push : pushes)
    if (failed(lowerPush(push, state)))
      return failure();
  for (TPopOp pop : pops)
    if (failed(lowerPop(pop, state)))
      return failure();
  for (TFreeOp free : frees)
    if (failed(lowerFree(free, state)))
      return failure();
  if (!init.getPipe().use_empty())
    return init.emitOpError("pipe handle still has users after LLVM lowering");
  init.erase();
  return success();
}

} // namespace

LogicalResult lowerA5UnifiedL2LPipeOpsForLLVM(ModuleOp module) {
  SmallVector<InitializeL2LPipeOp> pipes;
  module.walk([&](InitializeL2LPipeOp op) { pipes.push_back(op); });
  for (InitializeL2LPipeOp pipe : pipes)
    if (failed(lowerOnePipe(pipe)))
      return failure();

  bool hasOrphanedPlaceholder = false;
  module.walk([&](Operation *op) {
    if (!isa<TPushOp, TPopOp, TFreeOp, MaterializeTileOp,
             DeclareTileMemRefOp>(op))
      return WalkResult::advance();
    op->emitOpError(
        "authoring-stage FIFO placeholder survived A5 LLVM lowering");
    hasOrphanedPlaceholder = true;
    return WalkResult::interrupt();
  });
  return failure(hasOrphanedPlaceholder);
}

// New helper function to get the updated BaseMemRefType
BaseMemRefType getBaseMemRefTypeWithNewScope(BaseMemRefType type,
                                             AddressSpaceAttr targetMemScope) {
  if (auto memRefType = dyn_cast<MemRefType>(type)) {
    return MemRefType::Builder(memRefType).setMemorySpace(targetMemScope);
  } else if (auto unrankedMemRefType = dyn_cast<UnrankedMemRefType>(type)) {
    return UnrankedMemRefType::get(unrankedMemRefType.getElementType(),
                                   targetMemScope);
  }
  llvm_unreachable("Unexpected BaseMemRefType");
  return type;
}

void setBaseMemRefTypeScope(Value val, AddressSpaceAttr targetMemScope) {
  Type type = val.getType();
  if (!isa<BaseMemRefType>(type)) {
    return;
  }

  if (auto curMemScope = dyn_cast_if_present<AddressSpaceAttr>(
          dyn_cast<BaseMemRefType>(type).getMemorySpace())) {
    if (curMemScope != targetMemScope) {
      llvm::report_fatal_error("memref scope mismatch while propagating PTO address space");
    }
    return;
  }

  auto memRefType = cast<BaseMemRefType>(type);
  auto newMemRefType =
      getBaseMemRefTypeWithNewScope(memRefType, targetMemScope);
  val.setType(newMemRefType);
}


std::optional<AddressSpaceAttr> GetBufferSpaceAttr(Value operand) {
  if (auto tileTy = dyn_cast<pto::TileBufType>(operand.getType())) {
    if (auto memorySpaceAttr =
            dyn_cast_or_null<AddressSpaceAttr>(tileTy.getMemorySpace()))
      return memorySpaceAttr;
    return std::nullopt;
  }
  if (auto multiTy = dyn_cast<pto::MultiTileBufType>(operand.getType())) {
    if (auto memorySpaceAttr = dyn_cast_or_null<AddressSpaceAttr>(
            multiTy.getSlotType().getMemorySpace()))
      return memorySpaceAttr;
    return std::nullopt;
  }

  if (!llvm::isa<MemRefType>(operand.getType())) {
    return std::nullopt;
  }
  auto memRefType = cast<MemRefType>(operand.getType());
  auto memorySpace = memRefType.getMemorySpace();
  if (!memorySpace)
    return std::nullopt;
  auto memorySpaceAttr = dyn_cast<AddressSpaceAttr>(memorySpace);
  if (!memorySpaceAttr) {
    return std::nullopt;
  }
  return memorySpaceAttr;
}

std::optional<std::pair<Value, Value>> getOperationAliasInfo(Operation *op) {
  if (auto subViewOp = dyn_cast<memref::SubViewOp>(op)) {
    return std::make_pair(subViewOp.getResult(), subViewOp.getViewSource());
  } else if (auto makeViewOp = dyn_cast<pto::MakeTensorViewOp>(op)) {
    return std::make_pair(makeViewOp.getResult(), makeViewOp.getPtr());
  } else if (auto partViewOp = dyn_cast<pto::PartitionViewOp>(op)) {
    return std::make_pair(partViewOp.getResult(), partViewOp.getSource());
  } else if (auto addPtrOp = dyn_cast<pto::AddPtrOp>(op)) {
    return std::make_pair(addPtrOp.getResult(), addPtrOp.getPtr());
  } else if (auto ptrToIntOp = dyn_cast<pto::PtrToIntOp>(op)) {
    return std::make_pair(ptrToIntOp.getResult(), ptrToIntOp.getPtr());
  } else if (auto intToPtrOp = dyn_cast<pto::IntToPtrOp>(op)) {
    return std::make_pair(intToPtrOp.getResult(), intToPtrOp.getAddr());
  } else if (auto castPtrOp = dyn_cast<pto::CastPtrOp>(op)) {
    return std::make_pair(castPtrOp.getResult(), castPtrOp.getInput());
  } else if (auto subViewOp = dyn_cast<pto::SubViewOp>(op)) {
    return std::make_pair(subViewOp.getResult(), subViewOp.getSource());
  } else if (auto bitcastOp = dyn_cast<pto::BitcastOp>(op)) {
    return std::make_pair(bitcastOp.getResult(), bitcastOp.getSrc());
  } else if (auto reshapeOp = dyn_cast<pto::TReshapeOp>(op)) {
    return std::make_pair(reshapeOp.getResult(), reshapeOp.getSrc());
  } else if (auto multiGetOp = dyn_cast<pto::MultiTileGetOp>(op)) {
    return std::make_pair(multiGetOp.getResult(), multiGetOp.getSource());
  } else if (auto extSliceOp = dyn_cast<tensor::ExtractSliceOp>(op)) {
    return std::make_pair(extSliceOp.getResult(), extSliceOp.getSource());
  } else if (auto collapseShapeOp = dyn_cast<memref::CollapseShapeOp>(op)) {
    return std::make_pair(collapseShapeOp.getResult(),
                          collapseShapeOp.getViewSource());
  } else if (auto expandShapeOp = dyn_cast<memref::ExpandShapeOp>(op)) {
    return std::make_pair(expandShapeOp.getResult(),
                          expandShapeOp.getViewSource());
  } else if (auto viewOp = dyn_cast<memref::ViewOp>(op)) {
    return std::make_pair(viewOp.getResult(), viewOp.getViewSource());
  } else if (auto reinterpretCastOp = dyn_cast<memref::ReinterpretCastOp>(op)) {
    return std::make_pair(reinterpretCastOp.getResult(),
                          reinterpretCastOp.getViewSource());
  } else if (auto reshapeOp = dyn_cast<memref::ReshapeOp>(op)) {
    return std::make_pair(reshapeOp.getResult(), reshapeOp.getViewSource());
  } else if (auto castOp = dyn_cast<memref::CastOp>(op)) {
    return std::make_pair(castOp.getResult(), castOp.getViewSource());
  } else if (auto castOp = dyn_cast<UnrealizedConversionCastOp>(op)) {
    if (castOp.getNumOperands() == 1 && castOp.getNumResults() == 1)
      return std::make_pair(castOp.getResult(0), castOp.getOperand(0));
  } else if (auto extractStridedMetadataOp =
                 dyn_cast<memref::ExtractStridedMetadataOp>(op)) {
    return std::make_pair(extractStridedMetadataOp.getBaseBuffer(),
                          extractStridedMetadataOp.getViewSource());
  } else if (auto toBufferOp = dyn_cast<bufferization::ToBufferOp>(op)) {
    return std::make_pair(toBufferOp.getBuffer(), toBufferOp.getTensor());
  } else if (auto toTensorOp = dyn_cast<bufferization::ToTensorOp>(op)) {
    return std::make_pair(toTensorOp.getResult(), toTensorOp.getOperand());
  }
  return std::nullopt;
}

SmallVector<std::pair<Value, Value>, 15>
getSemanticNoAliasPairs(Operation *op) {
  SmallVector<std::pair<Value, Value>, 15> pairs;
  if (auto tmov = dyn_cast<TMovOp>(op)) {
    if (classifyTMovForm(tmov.getFp()) == TMovForm::XToZz) {
      pairs.emplace_back(tmov.getSrc(), tmov.getDst());
      pairs.emplace_back(tmov.getSrc(), tmov.getFp());
      pairs.emplace_back(tmov.getFp(), tmov.getDst());
    }
    return pairs;
  }

  if (auto tquant = dyn_cast<TQuantMxOp>(op)) {
    SmallVector<Value, 6> tiles{tquant.getSrc(), tquant.getDst(),
                                tquant.getExp(), tquant.getMax(),
                                tquant.getScaling()};
    if (Value expZz = tquant.getExpZz())
      tiles.push_back(expZz);
    for (unsigned lhs = 0; lhs < tiles.size(); ++lhs)
      for (unsigned rhs = lhs + 1; rhs < tiles.size(); ++rhs)
        pairs.emplace_back(tiles[lhs], tiles[rhs]);
  }
  return pairs;
}

namespace {

struct SemanticRange {
  Value root;
  uint64_t relativeBegin = 0;
  uint64_t bytes = 0;
  std::optional<uint64_t> absoluteBegin;
  std::optional<AddressSpace> addressSpace;
  std::optional<uint64_t> rowStrideBytes;
  std::optional<uint64_t> colStrideBytes;
  uint64_t elemBytes = 0;
};

struct StaticTileStrides {
  uint64_t rowBytes;
  uint64_t colBytes;
  uint64_t elemBytes;
};

static std::optional<uint64_t> getStaticTileBytes(TileBufType type) {
  unsigned elemBytes = getPTOStorageElemByteSize(type.getElementType());
  if (elemBytes == 0)
    return std::nullopt;
  ArrayRef<int64_t> shape = type.getShape();
  uint64_t elements = 1;
  if (type.getCompactModeI32() ==
      static_cast<int32_t>(CompactMode::RowPlusOne)) {
    if (shape.size() != 2 || llvm::is_contained(shape, ShapedType::kDynamic))
      return std::nullopt;
    bool rowMajor = type.getBLayoutValueI32() ==
                    static_cast<int32_t>(BLayout::RowMajor);
    uint64_t major = static_cast<uint64_t>(rowMajor ? shape[0] : shape[1]);
    uint64_t minor = static_cast<uint64_t>(rowMajor ? shape[1] : shape[0]);
    if (major == 0 || minor == 0)
      return uint64_t{0};
    if (minor == std::numeric_limits<uint64_t>::max() ||
        major - 1 > std::numeric_limits<uint64_t>::max() / (minor + 1))
      return std::nullopt;
    elements = (major - 1) * (minor + 1);
    if (minor > std::numeric_limits<uint64_t>::max() - elements)
      return std::nullopt;
    elements += minor;
  } else {
    for (int64_t dim : shape) {
      if (dim < 0 || elements > std::numeric_limits<uint64_t>::max() /
                                  static_cast<uint64_t>(dim))
        return std::nullopt;
      elements *= static_cast<uint64_t>(dim);
    }
  }
  if (elements > std::numeric_limits<uint64_t>::max() / elemBytes)
    return std::nullopt;
  return elements * elemBytes;
}

static std::optional<uint64_t> getConstantAddress(Value value) {
  IntegerAttr attr;
  if (!value || !matchPattern(value, m_Constant(&attr)) || attr.getInt() < 0)
    return std::nullopt;
  return static_cast<uint64_t>(attr.getInt());
}

static std::optional<StaticTileStrides>
getStaticTileStrides(TileBufType type) {
  ArrayRef<int64_t> shape = type.getShape();
  unsigned elemBytes = getPTOStorageElemByteSize(type.getElementType());
  if (shape.size() != 2 || elemBytes == 0 ||
      llvm::is_contained(shape, ShapedType::kDynamic) || shape[0] < 0 ||
      shape[1] < 0)
    return std::nullopt;

  // Boxed layouts are not affine rank-2 row/column views. Callers preserve the
  // complete parent range for them instead of guessing an offset envelope.
  if (type.getSLayoutValueI32() != static_cast<int32_t>(SLayout::NoneBox))
    return std::nullopt;

  bool rowMajor = type.getBLayoutValueI32() ==
                  static_cast<int32_t>(BLayout::RowMajor);
  uint64_t rows = static_cast<uint64_t>(shape[0]);
  uint64_t cols = static_cast<uint64_t>(shape[1]);
  uint64_t rowElems = rowMajor ? cols : 1;
  uint64_t colElems = rowMajor ? 1 : rows;
  if (type.getCompactModeI32() ==
      static_cast<int32_t>(CompactMode::RowPlusOne)) {
    if (rowMajor) {
      if (cols == std::numeric_limits<uint64_t>::max())
        return std::nullopt;
      rowElems = cols + 1;
    } else {
      if (rows == std::numeric_limits<uint64_t>::max())
        return std::nullopt;
      colElems = rows + 1;
    }
  }
  if (rowElems > std::numeric_limits<uint64_t>::max() / elemBytes ||
      colElems > std::numeric_limits<uint64_t>::max() / elemBytes)
    return std::nullopt;
  return StaticTileStrides{rowElems * elemBytes, colElems * elemBytes,
                           elemBytes};
}

static std::optional<AddressSpace> getTileAddressSpace(TileBufType type) {
  auto attr = dyn_cast_or_null<AddressSpaceAttr>(type.getMemorySpace());
  if (!attr)
    return std::nullopt;
  return attr.getAddressSpace();
}

static std::optional<uint64_t>
getSubviewByteOffset(SubViewOp op, const SemanticRange &source) {
  if (op.getOffsets().size() != 2)
    return std::nullopt;
  IntegerAttr rowAttr;
  IntegerAttr colAttr;
  if (!matchPattern(op.getOffsets()[0], m_Constant(&rowAttr)) ||
      !matchPattern(op.getOffsets()[1], m_Constant(&colAttr)) ||
      rowAttr.getInt() < 0 || colAttr.getInt() < 0)
    return std::nullopt;
  if (!source.rowStrideBytes || !source.colStrideBytes)
    return std::nullopt;
  uint64_t row = static_cast<uint64_t>(rowAttr.getInt());
  uint64_t col = static_cast<uint64_t>(colAttr.getInt());
  if (row > std::numeric_limits<uint64_t>::max() /
                *source.rowStrideBytes)
    return std::nullopt;
  uint64_t bytes = row * *source.rowStrideBytes;
  if (col > std::numeric_limits<uint64_t>::max() /
                *source.colStrideBytes)
    return std::nullopt;
  uint64_t colBytes = col * *source.colStrideBytes;
  if (colBytes > std::numeric_limits<uint64_t>::max() - bytes)
    return std::nullopt;
  return bytes + colBytes;
}

static std::optional<uint64_t>
getSubviewByteSpan(SubViewOp op, const SemanticRange &source) {
  if (!source.rowStrideBytes || !source.colStrideBytes ||
      source.elemBytes == 0)
    return std::nullopt;
  ArrayAttr sizes = op.getSizes();
  if (!sizes || sizes.size() != 2)
    return std::nullopt;
  int64_t rowsValue = cast<IntegerAttr>(sizes[0]).getInt();
  int64_t colsValue = cast<IntegerAttr>(sizes[1]).getInt();
  if (rowsValue < 0 || colsValue < 0)
    return std::nullopt;
  uint64_t rows = static_cast<uint64_t>(rowsValue);
  uint64_t cols = static_cast<uint64_t>(colsValue);
  if (rows == 0 || cols == 0)
    return uint64_t{0};
  if (rows - 1 > std::numeric_limits<uint64_t>::max() /
                     *source.rowStrideBytes ||
      cols - 1 > std::numeric_limits<uint64_t>::max() /
                     *source.colStrideBytes)
    return std::nullopt;
  uint64_t span = (rows - 1) * *source.rowStrideBytes;
  uint64_t colSpan = (cols - 1) * *source.colStrideBytes;
  if (colSpan > std::numeric_limits<uint64_t>::max() - span)
    return std::nullopt;
  span += colSpan;
  if (source.elemBytes > std::numeric_limits<uint64_t>::max() - span)
    return std::nullopt;
  return span + source.elemBytes;
}

static std::optional<SemanticRange> resolveSemanticRange(Value value) {
  if (!value)
    return std::nullopt;
  if (auto alloc = value.getDefiningOp<AllocTileOp>()) {
    auto tileType = dyn_cast<TileBufType>(alloc.getResult().getType());
    auto bytes = tileType ? getStaticTileBytes(tileType) : std::nullopt;
    if (!tileType || !bytes)
      return std::nullopt;
    auto strides = getStaticTileStrides(tileType);
    return SemanticRange{
        alloc.getResult(), 0, *bytes, getConstantAddress(alloc.getAddr()),
        getTileAddressSpace(tileType),
        strides ? std::optional<uint64_t>(strides->rowBytes) : std::nullopt,
        strides ? std::optional<uint64_t>(strides->colBytes) : std::nullopt,
        strides ? strides->elemBytes : uint64_t{0}};
  }
  if (auto multiGet = value.getDefiningOp<MultiTileGetOp>()) {
    auto alloc = multiGet.getSource().getDefiningOp<AllocMultiTileOp>();
    auto slotType = dyn_cast<TileBufType>(multiGet.getResult().getType());
    IntegerAttr slotAttr;
    if (!alloc || !slotType ||
        !matchPattern(multiGet.getSlot(), m_Constant(&slotAttr)) ||
        slotAttr.getInt() < 0)
      return std::nullopt;
    auto slotBytes = getStaticTileBytes(slotType);
    if (!slotBytes)
      return std::nullopt;
    std::optional<uint64_t> base = getConstantAddress(alloc.getAddr());
    if (!base) {
      if (auto addresses = alloc->getAttrOfType<DenseI64ArrayAttr>(
              kPtoMultiBufferAddrsAttrName)) {
        if (slotAttr.getInt() >= static_cast<int64_t>(addresses.size()) ||
            addresses[slotAttr.getInt()] < 0)
          return std::nullopt;
        base = static_cast<uint64_t>(addresses[slotAttr.getInt()]);
      }
    } else {
      uint64_t slot = static_cast<uint64_t>(slotAttr.getInt());
      if (slot > std::numeric_limits<uint64_t>::max() / *slotBytes ||
          *base > std::numeric_limits<uint64_t>::max() - slot * *slotBytes)
        return std::nullopt;
      *base += slot * *slotBytes;
    }
    auto strides = getStaticTileStrides(slotType);
    return SemanticRange{
        alloc.getResult(), 0, *slotBytes, base, getTileAddressSpace(slotType),
        strides ? std::optional<uint64_t>(strides->rowBytes) : std::nullopt,
        strides ? std::optional<uint64_t>(strides->colBytes) : std::nullopt,
        strides ? strides->elemBytes : uint64_t{0}};
  }
  if (auto subview = value.getDefiningOp<SubViewOp>()) {
    auto source = resolveSemanticRange(subview.getSource());
    if (!source)
      return std::nullopt;
    auto offset = getSubviewByteOffset(subview, *source);
    auto bytes = getSubviewByteSpan(subview, *source);
    // A boxed view has no simple affine row/column stride. Preserve the full
    // parent range so semantic no-alias checking remains conservative.
    if (!offset || !bytes)
      return source;
    if (*offset > source->bytes || *bytes > source->bytes - *offset)
      return std::nullopt;
    if (*offset > std::numeric_limits<uint64_t>::max() -
                      source->relativeBegin)
      return std::nullopt;
    source->relativeBegin += *offset;
    source->bytes = *bytes;
    if (source->absoluteBegin) {
      if (*offset > std::numeric_limits<uint64_t>::max() -
                        *source->absoluteBegin)
        return std::nullopt;
      *source->absoluteBegin += *offset;
    }
    return source;
  }
  if (auto bitcast = value.getDefiningOp<BitcastOp>()) {
    auto source = resolveSemanticRange(bitcast.getSrc());
    auto viewType = dyn_cast<TileBufType>(bitcast.getResult().getType());
    if (!source || !viewType)
      return std::nullopt;
    if (auto strides = getStaticTileStrides(viewType)) {
      source->rowStrideBytes = strides->rowBytes;
      source->colStrideBytes = strides->colBytes;
      source->elemBytes = strides->elemBytes;
    } else {
      source->rowStrideBytes.reset();
      source->colStrideBytes.reset();
      source->elemBytes = 0;
    }
    return source;
  }
  if (auto reshape = value.getDefiningOp<TReshapeOp>()) {
    auto source = resolveSemanticRange(reshape.getSrc());
    auto viewType = dyn_cast<TileBufType>(reshape.getResult().getType());
    if (!source || !viewType)
      return std::nullopt;
    if (auto strides = getStaticTileStrides(viewType)) {
      source->rowStrideBytes = strides->rowBytes;
      source->colStrideBytes = strides->colBytes;
      source->elemBytes = strides->elemBytes;
    } else {
      source->rowStrideBytes.reset();
      source->colStrideBytes.reset();
      source->elemBytes = 0;
    }
    return source;
  }
  if (auto cast = value.getDefiningOp<UnrealizedConversionCastOp>()) {
    if (cast.getNumOperands() == 1)
      return resolveSemanticRange(cast.getOperand(0));
  }
  return std::nullopt;
}

static bool rangesOverlap(const SemanticRange &lhs, const SemanticRange &rhs) {
  auto halfOpenRangesOverlap = [](uint64_t lhsBegin, uint64_t lhsBytes,
                                  uint64_t rhsBegin, uint64_t rhsBytes) {
    if (lhsBytes == 0 || rhsBytes == 0)
      return false;
    if (lhsBegin <= rhsBegin)
      return rhsBegin - lhsBegin < lhsBytes;
    return lhsBegin - rhsBegin < rhsBytes;
  };
  if (lhs.root == rhs.root)
    return halfOpenRangesOverlap(lhs.relativeBegin, lhs.bytes,
                                 rhs.relativeBegin, rhs.bytes);
  if (!lhs.absoluteBegin || !rhs.absoluteBegin ||
      lhs.addressSpace != rhs.addressSpace)
    return false;
  return halfOpenRangesOverlap(*lhs.absoluteBegin, lhs.bytes,
                               *rhs.absoluteBegin, rhs.bytes);
}

} // namespace

LogicalResult verifySemanticNoAliasRanges(func::FuncOp func) {
  LogicalResult result = success();
  func.walk([&](Operation *op) {
    if (failed(result))
      return;
    for (auto [lhs, rhs] : getSemanticNoAliasPairs(op)) {
      auto lhsRange = resolveSemanticRange(lhs);
      auto rhsRange = resolveSemanticRange(rhs);
      if (!lhsRange || !rhsRange || !rangesOverlap(*lhsRange, *rhsRange))
        continue;
      op->emitError("PlanMemory semantic no-alias violation: operand byte ranges overlap");
      result = failure();
      return;
    }
  });
  return result;
}

static Value tracebackImpl(Value memrefVal) {
  // case 1: v is the iter_arg of a scf.for
  if (auto arg = dyn_cast<BlockArgument>(memrefVal)) {
    if (auto forOp =
            dyn_cast<scf::ForOp>(arg.getParentRegion()->getParentOp())) {
      if (arg.getArgNumber() > 0 &&
          forOp.getInitArgs().size() > arg.getArgNumber() - 1) {
        return forOp.getInitArgs()[arg.getArgNumber() - 1];
      }
    }
  }

  Value result;
  Operation *def = memrefVal.getDefiningOp();
  if (!def) {
    // failed to trace back
    return result;
  }

  // case 2: v is the result of cast-like ops
  //  - memref.cast
  //  - memref.collapse_shape
  //  - memref.expand_shape
  //  - memref.memory_space_cast
  //  - memref.reinterpret_cast
  //  - memref.reshape
  //  - memref.transpose
  if (auto op = dyn_cast<memref::CastOp>(def)) {
    result = op.getSource();
  } else if (auto op = dyn_cast<memref::CollapseShapeOp>(def)) {
    result = op.getSrc();
  } else if (auto op = dyn_cast<memref::ExpandShapeOp>(def)) {
    result = op.getSrc();
  } else if (auto op = dyn_cast<memref::MemorySpaceCastOp>(def)) {
    result = op.getSource();
  } else if (auto op = dyn_cast<memref::ReinterpretCastOp>(def)) {
    result = op.getSource();
  } else if (auto op = dyn_cast<memref::ReshapeOp>(def)) {
    result = op.getSource();
  } else if (auto op = dyn_cast<memref::TransposeOp>(def)) {
    result = op.getIn();
  } else if (auto op = dyn_cast<UnrealizedConversionCastOp>(def)) {
    result = op.getOperand(cast<OpResult>(memrefVal).getResultNumber());
  } else if (auto op = dyn_cast<scf::ForOp>(def)) {
    // trace back memref.alloc support scf.for
    result = op.getInitArgs()[cast<OpResult>(memrefVal).getResultNumber()];
  }

  if (result) {
    return result;
  }

  // case 3: v is the result of the view-like ops
  //  - memref::view
  //  - memref::subview
  if (auto op = dyn_cast<memref::ViewOp>(def)) {
    result = op.getViewSource();
  } else if (auto op = dyn_cast<memref::SubViewOp>(def)) {
    result = op.getViewSource();
  }

  return result;
}

static bool isAllocLikeOp(Operation *op) {
  if (!op) {
    return false;
  }
  return isa<memref::AllocOp>(op) || isa<memref::AllocaOp>(op);
}

static bool isAllocLikeOp(Value val) {
  return isAllocLikeOp(val.getDefiningOp());
}

std::optional<int64_t> getStaticTotalSize(const ArrayRef<int64_t> &shapes) {
  int64_t totalSize = 1;
  for (const auto &shape : shapes) {
    if (ShapedType::isDynamic(shape)) {
      return std::nullopt;
    }
    totalSize = totalSize * shape;
  }
  return totalSize;
}

uint64_t AlignUp(uint64_t lhs, uint64_t rhs) {
  if (rhs == 0) {
    return lhs;
  }
  if (lhs % rhs != 0) {
    lhs += rhs - (lhs % rhs);
  }
  return lhs;
}

Value tracebackMemRef(Value memrefVal) {
  int loopBound = 256;
  while (memrefVal && !isAllocLikeOp(memrefVal)) {
    auto upward = tracebackImpl(memrefVal);
    if (!upward) {
      break;
    }

    memrefVal = upward;

    // avoid infinite loop
    if (loopBound-- < 0) {
      LLVM_DEBUG(llvm::dbgs()
                 << "tracebackMemRef exceeds loopBound(" << loopBound << ")!");
      break;
    }
  }

  return memrefVal;
}

std::optional<memref::AllocOp> tracebackMemRefToAlloc(Value memrefVal) {
  auto tracedValue = tracebackMemRef(memrefVal);
  return isAllocLikeOp(tracedValue)
             ? tracedValue.getDefiningOp<memref::AllocOp>()
             : std::optional<memref::AllocOp>();
}

/// trace value and judge if it is function argument
bool isFromFunctionArg(mlir::Value v) {
  return tracebackMemRef(v).getDefiningOp() == nullptr;
}

bool isLocalBuffer(std::optional<AddressSpaceAttr> memorySpaceAttr) {
  if (!memorySpaceAttr.has_value()) {
    return false;
  }

  if (memorySpaceAttr.value().getAddressSpace() == pto::AddressSpace::GM) {
    return false;
  }
  if (LocalBufferSpace.count(memorySpaceAttr.value().getAddressSpace())) {
    return true;
  }
  llvm_unreachable("Currently only support (UB | L1 | L0C) allocation");
}

static SmallVector<Value> getOpTouchBuffer(Operation *op) {
  SmallVector<Value> touchBuffer;
  touchBuffer.insert(touchBuffer.end(), op->getResults().begin(),
                     op->getResults().end());
  for (OpOperand &operand : op->getOpOperands()) {
    touchBuffer.push_back(operand.get());
  }
  return touchBuffer;
}

bool isOpTouchLocalBuffer(Operation *op) {
  auto touchBuffer = getOpTouchBuffer(op);
  for (Value buffer : touchBuffer) {
    auto bufferSpace = GetBufferSpaceAttr(buffer);
    if (isLocalBuffer(bufferSpace)) {
      return true;
    }
  }
  return false;
}

ModuleOp getTopLevelModuleOp(Operation *op) {
  ModuleOp moduleOp = op->getParentOfType<ModuleOp>();
  while (moduleOp && moduleOp->getParentOp()) {
    moduleOp = moduleOp->getParentOfType<ModuleOp>();
  }
  return moduleOp;
}

/// Index of yielded value where is alias of targetVal.
static std::optional<int> getYieldValueIdx(Value targetVal, ValueRange yieldedValues) {
  auto it = std::find(yieldedValues.begin(), yieldedValues.end(), targetVal);
  if (it != yieldedValues.end()) {
    return it - yieldedValues.begin();
  }

  return std::nullopt;
}

LoopLikeOpInterface getParentLoop(Value val) {
  if (!val.getDefiningOp()) {
    return nullptr;
  }

  // Firstly, get parent loop
  LoopLikeOpInterface parentLoop =
      val.getDefiningOp()->getParentOfType<LoopLikeOpInterface>();
  if (!parentLoop) {
    return nullptr;
  }

  // Need to determine whether val is yielded by the loop.
  auto yieldedValues = parentLoop.getYieldedValues();
  if (yieldedValues.empty()) {
    return parentLoop;
  }

  auto idxLoopRes = getYieldValueIdx(val, yieldedValues);
  if (idxLoopRes.has_value()) {
    // The val is yielded by loop, so need to find parent of parent loop.
    auto res = parentLoop.getLoopResults().value()[*idxLoopRes];
    return getParentLoop(res);
  }

  // Need to determine whether val is yielded by if/else.
  auto parentIf = val.getDefiningOp()->getParentOfType<scf::IfOp>();
  if (!parentIf || parentIf.getResults().empty()) {
    return parentLoop;
  }

  auto thenYieldOp = parentIf.thenYield();
  auto thenYieldOpers = thenYieldOp.getOperands();

  auto idxThenYielded = getYieldValueIdx(val, thenYieldOpers);
  if (idxThenYielded.has_value()) {
    // The val is yielded by ifOp, need to find parent loop of ifOp's result
    auto res = parentIf.getResults()[*idxThenYielded];
    return getParentLoop(res);
  }

  auto elseYieldOp = parentIf.elseYield();
  auto elseYieldOpers = elseYieldOp.getOperands();
  auto idxElseYielded = getYieldValueIdx(val, elseYieldOpers);
  if (idxElseYielded.has_value()) {
    auto res = parentIf.getResults()[*idxElseYielded];
    return getParentLoop(res);
  }

  return parentLoop;
}

}
}
