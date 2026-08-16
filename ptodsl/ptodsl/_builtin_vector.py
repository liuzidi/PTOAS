# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Builtin MLIR vector helpers for PTODSL scalar-contiguous access."""

from ._bootstrap import make_context  # ensure MLIR is on sys.path  # noqa: F401
from ._scalar_coercion import coerce_scalar_to_type
from ._surface_values import VecValue, unwrap_surface_value
from ._types import _resolve, _validate_vec_size, vec_type

from mlir.dialects import arith
from mlir.dialects import llvm
from mlir.ir import IntegerType, VectorType


def Vec(dtype, size: int, *, init=None):
    """Create a builtin vector type descriptor or broadcast vector value."""
    size = _validate_vec_size(size, context="pto.Vec(...)")
    descriptor = vec_type(dtype, size)
    if init is None:
        return descriptor
    return _broadcast_vec_value(descriptor, init)


def _broadcast_vec_value(descriptor, init):
    vector_type = _resolve(descriptor)
    element_type = VectorType(vector_type).element_type
    raw_init = unwrap_surface_value(init)

    if hasattr(raw_init, "type") and VectorType.isinstance(raw_init.type):
        vec_value = VecValue(raw_init)
        if vec_value.type != vector_type:
            raise TypeError(f"pto.Vec(..., init=vector) expected {vector_type}, got {vec_value.type}")
        return vec_value

    scalar_value = coerce_scalar_to_type(init, element_type, context="pto.Vec(..., init=...)")
    current = llvm.UndefOp(vector_type).res
    i32 = IntegerType.get_signless(32)
    for index in range(descriptor.size):
        element_index = arith.ConstantOp(i32, index).result
        current = llvm.InsertElementOp(current, scalar_value, element_index).res
    return VecValue(current)


__all__ = ["Vec"]
