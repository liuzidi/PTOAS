// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This file is distributed under the CANN Open Software License Agreement
// Version 2.0. Please refer to the LICENSE file in the repository root.

#ifndef PTO_TRANSFORMS_TILESHAPESTATEANALYSIS_H
#define PTO_TRANSFORMS_TILESHAPESTATEANALYSIS_H

#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace pto {

/// The shape facts used by candidate selection, expansion and fusion.  A
/// missing or non-dominating valid_shape update is Unknown; it is never
/// silently treated as a full tile.
struct TileShapeState {
  enum class Kind { Full, Partial, Unknown } kind = Kind::Unknown;
  llvm::SmallVector<int64_t, 2> shape;
  llvm::SmallVector<int64_t, 2> validShape;

  bool isFull() const { return kind == Kind::Full; }
  bool isPartial() const { return kind == Kind::Partial; }
  bool isUnknown() const { return kind == Kind::Unknown; }
};

/// Resolve the latest statically-known valid_shape update that dominates
/// `useOp`.  If the control-flow path does not prove one unique value, return
/// false.  With no use operation, only an unambiguous declaration/update is
/// accepted.
bool resolveStaticTileValidShape(Value value,
                                 llvm::SmallVectorImpl<int64_t> &validShape,
                                 Operation *useOp = nullptr);

TileShapeState analyzeTileShape(Value value, Operation *useOp = nullptr);

/// Returns true only when every tile operand has a statically-proven valid
/// shape equal to its physical shape.  Operations without tile operands are
/// not rejected by this helper.
bool hasStaticFullTileValidShape(Operation *op);

} // namespace pto
} // namespace mlir

#endif
