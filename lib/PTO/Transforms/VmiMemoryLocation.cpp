// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/Transforms/VmiMemoryLocation.h"

#include "PTO/IR/PTO.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"

using namespace mlir;

namespace {

static std::optional<int64_t> getConstantInteger(Value value) {
  if (!value)
    return std::nullopt;
  if (auto c = value.getDefiningOp<arith::ConstantOp>()) {
    if (auto attr = dyn_cast<IntegerAttr>(c.getValue()))
      return attr.getInt();
  }
  if (auto c = value.getDefiningOp<arith::ConstantIndexOp>())
    return c.value();
  if (auto c = value.getDefiningOp<arith::ConstantIntOp>())
    return c.value();
  if (auto add = value.getDefiningOp<arith::AddIOp>()) {
    auto lhs = getConstantInteger(add.getLhs());
    auto rhs = getConstantInteger(add.getRhs());
    int64_t result = 0;
    if (lhs && rhs && !__builtin_add_overflow(*lhs, *rhs, &result))
      return result;
  }
  if (auto sub = value.getDefiningOp<arith::SubIOp>()) {
    auto lhs = getConstantInteger(sub.getLhs());
    auto rhs = getConstantInteger(sub.getRhs());
    int64_t result = 0;
    if (lhs && rhs && !__builtin_sub_overflow(*lhs, *rhs, &result))
      return result;
  }
  if (auto mul = value.getDefiningOp<arith::MulIOp>()) {
    auto lhs = getConstantInteger(mul.getLhs());
    auto rhs = getConstantInteger(mul.getRhs());
    int64_t result = 0;
    if (lhs && rhs && !__builtin_mul_overflow(*lhs, *rhs, &result))
      return result;
  }
  return std::nullopt;
}

static std::optional<int64_t> getConstantI64(Value value) {
  auto c = value ? value.getDefiningOp<arith::ConstantOp>() : nullptr;
  if (!c)
    return std::nullopt;
  auto attr = dyn_cast<IntegerAttr>(c.getValue());
  if (!attr || !attr.getType().isInteger(64))
    return std::nullopt;
  return attr.getInt();
}

static std::optional<int64_t> getElementBytes(Type type) {
  Type element;
  if (auto memref = dyn_cast<MemRefType>(type))
    element = memref.getElementType();
  else if (auto unranked = dyn_cast<UnrankedMemRefType>(type))
    element = unranked.getElementType();
  if (!element)
    return std::nullopt;
  if (element.isF32() || element.isInteger(32))
    return 4;
  if (element.isF16() || element.isBF16() || element.isInteger(16))
    return 2;
  if (element.isInteger(8) || element.isInteger(1))
    return 1;
  if (element.isInteger(64))
    return 8;
  return std::nullopt;
}

static std::optional<int64_t> getStaticMemrefBytes(Type type) {
  auto memref = dyn_cast<MemRefType>(type);
  if (!memref)
    return std::nullopt;
  auto elementBytes = getElementBytes(type);
  if (!elementBytes)
    return std::nullopt;
  SmallVector<int64_t, 4> strides;
  int64_t offset = 0;
  if (failed(memref.getStridesAndOffset(strides, offset)) ||
      ShapedType::isDynamic(offset) ||
      llvm::any_of(strides, ShapedType::isDynamic))
    return std::nullopt;
  int64_t elements = 1;
  for (auto [dim, stride] : llvm::zip(memref.getShape(), strides)) {
    if (ShapedType::isDynamic(dim) || dim < 0 || stride < 0)
      return std::nullopt;
    if (dim == 0) {
      elements = 0;
      break;
    }
    int64_t term = 0;
    if (dim > 0 && __builtin_mul_overflow(dim - 1, stride, &term))
      return std::nullopt;
    if (__builtin_add_overflow(elements, term, &elements))
      return std::nullopt;
  }
  if (elements > INT64_MAX / *elementBytes)
    return std::nullopt;
  return elements * *elementBytes;
}

static std::optional<int64_t> getStaticOffset(Value value) {
  return getConstantInteger(value);
}

static std::optional<int64_t>
getStaticSubviewByteOffset(memref::SubViewOp subview, MemRefType sourceType) {
  SmallVector<int64_t, 4> strides;
  int64_t sourceOffset = 0;
  if (failed(sourceType.getStridesAndOffset(strides, sourceOffset)) ||
      ShapedType::isDynamic(sourceOffset) ||
      llvm::any_of(strides, ShapedType::isDynamic))
    return std::nullopt;
  auto elementBytes = getElementBytes(sourceType);
  if (!elementBytes)
    return std::nullopt;

  // resolveVmiStorageRoot(source) already accounts for sourceType's static
  // layout offset. Add only the subview's mixed offsets here.
  int64_t offset = 0;
  int64_t byteOffset = 0;
  auto mixedOffsets = subview.getMixedOffsets();
  if (mixedOffsets.size() != strides.size())
    return std::nullopt;
  for (auto [mixed, stride] : llvm::zip(mixedOffsets, strides)) {
    auto value = dyn_cast<Value>(mixed);
    std::optional<int64_t> constant;
    if (value) {
      constant = getStaticOffset(value);
    } else if (auto attr = dyn_cast<Attribute>(mixed)) {
      if (auto integer = dyn_cast<IntegerAttr>(attr))
        constant = integer.getInt();
    }
    if (!constant || stride < 0)
      return std::nullopt;
    int64_t term = 0;
    if (__builtin_mul_overflow(*constant, stride, &term) ||
        __builtin_add_overflow(offset, term, &offset))
      return std::nullopt;
  }
  if (__builtin_mul_overflow(offset, *elementBytes, &byteOffset))
    return std::nullopt;
  return byteOffset;
}

static std::optional<int64_t>
getStaticSubviewSpanBytes(memref::SubViewOp subview) {
  auto resultType = dyn_cast<MemRefType>(subview.getResult().getType());
  if (!resultType)
    return std::nullopt;
  SmallVector<int64_t, 4> strides;
  int64_t offset = 0;
  auto elementBytes = getElementBytes(resultType);
  if (failed(resultType.getStridesAndOffset(strides, offset)) ||
      llvm::any_of(strides, ShapedType::isDynamic) || !elementBytes)
    return std::nullopt;
  int64_t spanElements = 1;
  for (auto [dim, stride] : llvm::zip(resultType.getShape(), strides)) {
    if (ShapedType::isDynamic(dim) || stride < 0)
      return std::nullopt;
    if (dim == 0) {
      spanElements = 0;
      break;
    }
    int64_t term = 0;
    if (dim > 0 && __builtin_mul_overflow(dim - 1, stride, &term))
      return std::nullopt;
    if (__builtin_add_overflow(spanElements, term, &spanElements))
      return std::nullopt;
  }
  int64_t spanBytes = 0;
  if (__builtin_mul_overflow(spanElements, *elementBytes, &spanBytes))
    return std::nullopt;
  return spanBytes;
}

} // namespace

std::optional<mlir::pto::VmiStorageRoot>
mlir::pto::resolveVmiStorageRoot(Value base) {
  while (base) {
    if (auto cast = base.getDefiningOp<pto::CastPtrOp>()) {
      base = cast->getOperand(0);
      continue;
    }
    if (auto pointerCast = base.getDefiningOp<pto::PointerCastOp>()) {
      if (pointerCast.getAddrs().empty())
        return std::nullopt;
      auto address = getConstantI64(pointerCast.getAddrs().front());
      if (!address)
        return std::nullopt;
      Type viewType = pointerCast.getResult().getType();
      if (auto memref = dyn_cast<MemRefType>(viewType)) {
        SmallVector<int64_t, 4> strides;
        int64_t offset = 0;
        if (failed(memref.getStridesAndOffset(strides, offset)) ||
            ShapedType::isDynamic(offset))
          return std::nullopt;
        auto elementBytes = getElementBytes(viewType);
        int64_t offsetBytes = 0;
        int64_t logicalAddress = 0;
        if (!elementBytes ||
            __builtin_mul_overflow(offset, *elementBytes, &offsetBytes) ||
            __builtin_add_overflow(*address, offsetBytes, &logicalAddress))
          return std::nullopt;
        *address = logicalAddress;
      }
      return VmiStorageRoot{*address, getStaticMemrefBytes(viewType), viewType};
    }
    if (auto bind = base.getDefiningOp<pto::BindTileOp>()) {
      base = bind.getSource();
      continue;
    }
    if (auto cast = base.getDefiningOp<memref::CastOp>()) {
      base = cast.getSource();
      continue;
    }
    if (auto subview = base.getDefiningOp<memref::SubViewOp>()) {
      auto sourceType = dyn_cast<MemRefType>(subview.getSource().getType());
      if (!sourceType)
        return std::nullopt;
      auto sourceRoot = resolveVmiStorageRoot(subview.getSource());
      auto byteOffset = getStaticSubviewByteOffset(subview, sourceType);
      auto spanBytes = getStaticSubviewSpanBytes(subview);
      if (!sourceRoot || !byteOffset || !spanBytes)
        return std::nullopt;
      int64_t address = 0;
      if (__builtin_add_overflow(sourceRoot->address, *byteOffset, &address))
        return std::nullopt;
      return VmiStorageRoot{address, *spanBytes, subview.getResult().getType()};
    }
    return std::nullopt;
  }
  return std::nullopt;
}

bool mlir::pto::mayAliasVmiStorageRoot(const VmiStorageRoot &lhs,
                                       const VmiStorageRoot &rhs) {
  if (lhs.address == rhs.address)
    return true;
  if (!lhs.storageBytes || !rhs.storageBytes)
    return true;
  if (*lhs.storageBytes < 0 || *rhs.storageBytes < 0)
    return true;
  const int64_t lhsEnd = lhs.address > INT64_MAX - *lhs.storageBytes
                             ? INT64_MAX
                             : lhs.address + *lhs.storageBytes;
  const int64_t rhsEnd = rhs.address > INT64_MAX - *rhs.storageBytes
                             ? INT64_MAX
                             : rhs.address + *rhs.storageBytes;
  return lhs.address < rhsEnd && rhs.address < lhsEnd;
}

bool mlir::pto::mayAliasVmiAccess(const VmiAccessLocation &lhs,
                                  const VmiAccessLocation &rhs) {
  if (!mayAliasVmiStorageRoot(lhs.root, rhs.root))
    return false;
  if (lhs.root.viewType != rhs.root.viewType || lhs.accessBytes <= 0 ||
      rhs.accessBytes <= 0)
    return true;
  auto lhsOffset = getConstantInteger(lhs.elementOffset);
  auto rhsOffset = getConstantInteger(rhs.elementOffset);
  auto elementBytes = getElementBytes(lhs.root.viewType);
  if (!lhsOffset || !rhsOffset || !elementBytes)
    return true;
  int64_t lhsDelta = 0;
  int64_t rhsDelta = 0;
  if (__builtin_mul_overflow(*lhsOffset, *elementBytes, &lhsDelta) ||
      __builtin_mul_overflow(*rhsOffset, *elementBytes, &rhsDelta))
    return true;
  int64_t lhsBegin = 0;
  int64_t rhsBegin = 0;
  if (__builtin_add_overflow(lhs.root.address, lhsDelta, &lhsBegin) ||
      __builtin_add_overflow(rhs.root.address, rhsDelta, &rhsBegin))
    return true;
  int64_t lhsEnd = 0;
  int64_t rhsEnd = 0;
  if (__builtin_add_overflow(lhsBegin, lhs.accessBytes, &lhsEnd) ||
      __builtin_add_overflow(rhsBegin, rhs.accessBytes, &rhsEnd))
    return true;
  return lhsBegin < rhsEnd && rhsBegin < lhsEnd;
}
