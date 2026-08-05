# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Shared helpers for A5 VMI TileLib candidates.

Per-op VMI candidates live next to the ordinary A5 TileLib template for the
same TileOp (for example ``tadd.py`` owns both the normal and VMI ``tadd``
candidates).  This module only contains common emitters, legality helpers, and
algorithm fragments reused by those per-op candidates.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from ptodsl import pto
from ptodsl._surface_values import unwrap_surface_value
from ptodsl._surface_types import Tile
from ptodsl._runtime_scalar_ops import emit_runtime_binary_op
from ptodsl._tile_template_tracing import (
    CanonicalBlockMap,
    CanonicalBlockCoordinate,
    _MaskValue,
    _TileProxy,
    _Value,
    _VectorValue,
    ScalarType,
    f16,
    bf16,
    f32,
    i16,
    i32,
    for_,
    index_add,
    index_mul,
    tile_template as _trace_tile_template,
)
from ptodsl._vmi_namespace import vmi as _vmi_builder
from ptodsl.tilelib import registry as _tilelib_registry
from ptodsl.tilelib.registry import TileTemplateRegistry
from mlir.dialects import pto as _pto_dialect


ElementwiseCompute = Callable[[Sequence[_VectorValue], _MaskValue], _VectorValue]
FLOAT_DTYPES = (f32, f16)
ui32 = ScalarType("ui32", lanes=64, mask_bits=32, bytewidth=4)
ui16 = ScalarType("ui16", lanes=128, mask_bits=16, bytewidth=2)

# The helpers below only adapt traced TileLib values and dtype metadata.  Use
# the raw VMI builder alias instead of pto.vmi so tile-template tracing helpers
# cannot shadow the builder namespace while a VMI template is being traced.


def _pto_dtype(dtype: ScalarType):
    descriptors = {
        "f32": pto.f32,
        "f16": pto.f16,
        "bf16": pto.bf16,
        "i8": pto.i8,
        "i16": pto.i16,
        "i32": pto.i32,
        "ui16": pto.ui16,
        "ui32": pto.ui32,
    }
    try:
        return descriptors[dtype.name]
    except KeyError as exc:
        raise ValueError(f"unsupported VMI TileLib dtype {dtype}") from exc


def _wrap_vreg(value, dtype: ScalarType) -> _VectorValue:
    return _VectorValue(unwrap_surface_value(value), dtype)


def _wrap_mask(value, dtype: ScalarType) -> _MaskValue:
    return _MaskValue(unwrap_surface_value(value), dtype)


def row_reduce_vmi_constraint(
    src_shape=(),
    src_valid_shape=(),
    workspace_shape=(),
    dst_shape=(),
    dst_valid_shape=(),
    src_dtype=None,
    workspace_dtype=None,
    dst_dtype=None,
    src_config=None,
    dst_config=None,
    **_,
):
    if (
        src_dtype != "f32"
        or workspace_dtype != "f32"
        or dst_dtype != "f32"
        or len(src_shape) != 2
        or len(src_valid_shape) != 2
        or len(workspace_shape) != 2
        or len(dst_shape) != 2
        or len(dst_valid_shape) != 2
    ):
        return False
    rows, cols = src_shape
    valid_rows, valid_cols = src_valid_shape
    workspace_rows, workspace_cols = workspace_shape
    return (
        rows == valid_rows
        and workspace_rows == rows
        and workspace_cols >= 1
        and dst_shape == (rows, 1)
        and dst_valid_shape == (rows, 1)
        and src_config is not None
        and src_config.b_layout == "row_major"
        and dst_config is not None
        and dst_config.b_layout == "col_major"
        and isinstance(cols, int)
        and isinstance(valid_cols, int)
        and cols > 0
        and valid_cols > 0
        and valid_cols <= cols
        and valid_cols % f32.lanes == 0
    )


def _vreg_lanes(value: _VectorValue) -> int:
    return _pto_dialect.VMIVRegType(value.value.type).element_count


def _validate_same_dtype(operation: str, *values: _VectorValue) -> ScalarType:
    if not values:
        raise ValueError(f"{operation} expects at least one vector")
    dtype = values[0].dtype
    if any(value.dtype != dtype for value in values):
        raise TypeError(f"{operation} operands must use the same dtype")
    return dtype


def _validate_mask(operation: str, mask: _MaskValue, dtype: ScalarType) -> None:
    if not isinstance(mask, _MaskValue):
        raise TypeError(f"{operation} expects a VMI mask")
    if mask.dtype.mask_bits != dtype.mask_bits:
        raise TypeError(
            f"{operation} mask granularity b{mask.dtype.mask_bits} is incompatible "
            f"with {dtype} lanes using b{dtype.mask_bits}"
        )


def _validate_block_access(
    tile: _TileProxy,
    coordinate: CanonicalBlockCoordinate,
    *,
    operation: str,
) -> None:
    if not isinstance(tile, _TileProxy):
        raise TypeError(f"{operation} expects a traced Tile argument")
    if not isinstance(coordinate, CanonicalBlockCoordinate):
        raise TypeError(f"{operation} expects a CanonicalBlockCoordinate")
    if tile._spec.shape != coordinate.block_map.shape:
        raise ValueError(
            f"{operation} tile shape {tile._spec.shape} does not match "
            f"CanonicalBlockMap shape {coordinate.block_map.shape}"
        )


def _create_mask_lanes(
    active_lanes: int,
    vector_lanes: int,
    dtype: ScalarType,
    *,
    trace,
) -> _MaskValue:
    if not isinstance(dtype, ScalarType):
        raise TypeError("_create_mask_lanes expects a tile-template ScalarType")
    if not 0 < active_lanes <= vector_lanes:
        raise ValueError("active_lanes must be in the range [1, vector_lanes]")
    active = trace.index_const(active_lanes)
    return _wrap_mask(_vmi_builder.create_mask(active.value, size=vector_lanes), dtype)


def _create_mask(
    block_map: CanonicalBlockMap,
    dtype: ScalarType,
    *,
    trace,
) -> _MaskValue:
    if not isinstance(block_map, CanonicalBlockMap):
        raise TypeError("_create_mask expects a CanonicalBlockMap")
    return _create_mask_lanes(
        block_map.logical_lanes,
        block_map.logical_lanes,
        dtype,
        trace=trace,
    )


def _prepare_tile_access(*tiles: _TileProxy) -> None:
    if not tiles:
        raise ValueError("_prepare_tile_access requires at least one Tile")
    for tile in tiles:
        if not isinstance(tile, _TileProxy):
            raise TypeError("_prepare_tile_access expects traced Tile arguments")
        tile._trace.ensure_tile_ptr(tile)


def _vload(tile: _TileProxy, coordinate: CanonicalBlockCoordinate) -> _VectorValue:
    _validate_block_access(tile, coordinate, operation="_vload")
    ptr_value = tile._trace.ensure_tile_ptr(tile)
    offset = tile._trace._coerce_index(coordinate.linear_offset)
    return _wrap_vreg(
        _vmi_builder.vload(
            ptr_value.value,
            offset.value,
            size=coordinate.block_map.logical_lanes,
        ),
        tile.element_type,
    )


def _vload_linear(tile: _TileProxy, offset, *, lanes: int) -> _VectorValue:
    if not isinstance(tile, _TileProxy):
        raise TypeError("_vload_linear expects a traced Tile argument")
    if not isinstance(lanes, int) or lanes <= 0:
        raise ValueError("_vload_linear lanes must be a positive integer")
    ptr_value = tile._trace.ensure_tile_ptr(tile)
    offset_value = tile._trace._coerce_index(offset)
    return _wrap_vreg(
        _vmi_builder.vload(ptr_value.value, offset_value.value, size=lanes),
        tile.element_type,
    )


def _vstore(
    vec: _VectorValue,
    tile: _TileProxy,
    coordinate: CanonicalBlockCoordinate,
    mask: _MaskValue,
) -> None:
    _validate_block_access(tile, coordinate, operation="_vstore")
    if vec.dtype != tile.element_type:
        raise TypeError("_vstore value and destination must use the same dtype")
    _validate_mask("_vstore", mask, vec.dtype)
    ptr_value = tile._trace.ensure_tile_ptr(tile)
    offset = tile._trace._coerce_index(coordinate.linear_offset)
    _vmi_builder.vstore(vec.value, ptr_value.value, offset.value, mask.value)


def _vstore_linear(
    vec: _VectorValue,
    tile: _TileProxy,
    offset,
    mask: _MaskValue,
) -> None:
    if not isinstance(tile, _TileProxy):
        raise TypeError("_vstore_linear expects a traced Tile destination")
    if vec.dtype != tile.element_type:
        raise TypeError("_vstore_linear value and destination must use the same dtype")
    _validate_mask("_vstore_linear", mask, vec.dtype)
    ptr_value = tile._trace.ensure_tile_ptr(tile)
    offset_value = tile._trace._coerce_index(offset)
    _vmi_builder.vstore(vec.value, ptr_value.value, offset_value.value, mask.value)


def _vbinary(name: str, lhs: _VectorValue, rhs: _VectorValue, mask: _MaskValue) -> _VectorValue:
    dtype = _validate_same_dtype(f"pto.vmi.{name}", lhs, rhs)
    _validate_mask(f"pto.vmi.{name}", mask, dtype)
    builder = getattr(_vmi_builder, name)
    return _wrap_vreg(builder(lhs.value, rhs.value, mask.value), dtype)


def _vadd(lhs: _VectorValue, rhs: _VectorValue, mask: _MaskValue) -> _VectorValue:
    return _vbinary("vadd", lhs, rhs, mask)


def _vsub(lhs: _VectorValue, rhs: _VectorValue, mask: _MaskValue) -> _VectorValue:
    return _vbinary("vsub", lhs, rhs, mask)


def _vmul(lhs: _VectorValue, rhs: _VectorValue, mask: _MaskValue) -> _VectorValue:
    return _vbinary("vmul", lhs, rhs, mask)


def _vdiv(lhs: _VectorValue, rhs: _VectorValue, mask: _MaskValue) -> _VectorValue:
    return _vbinary("vdiv", lhs, rhs, mask)


def _vmax(lhs: _VectorValue, rhs: _VectorValue, mask: _MaskValue) -> _VectorValue:
    return _vbinary("vmax", lhs, rhs, mask)


def _vmin(lhs: _VectorValue, rhs: _VectorValue, mask: _MaskValue) -> _VectorValue:
    return _vbinary("vmin", lhs, rhs, mask)


def _vand(lhs: _VectorValue, rhs: _VectorValue, mask: _MaskValue) -> _VectorValue:
    return _vbinary("vand", lhs, rhs, mask)


def _vor(lhs: _VectorValue, rhs: _VectorValue, mask: _MaskValue) -> _VectorValue:
    return _vbinary("vor", lhs, rhs, mask)


def _vshl(lhs: _VectorValue, rhs: _VectorValue, mask: _MaskValue) -> _VectorValue:
    return _vbinary("vshl", lhs, rhs, mask)


def _vshr(lhs: _VectorValue, rhs: _VectorValue, mask: _MaskValue) -> _VectorValue:
    return _vbinary("vshr", lhs, rhs, mask)


def _vunary(name: str, source: _VectorValue, mask: _MaskValue) -> _VectorValue:
    _validate_mask(f"pto.vmi.{name}", mask, source.dtype)
    builder = getattr(_vmi_builder, name)
    return _wrap_vreg(builder(source.value, mask.value), source.dtype)


def _vexp(source: _VectorValue, mask: _MaskValue) -> _VectorValue:
    return _vunary("vexp", source, mask)


def _vabs(source: _VectorValue, mask: _MaskValue) -> _VectorValue:
    return _vunary("vabs", source, mask)


def _vneg(source: _VectorValue, mask: _MaskValue) -> _VectorValue:
    return _vunary("vneg", source, mask)


def _vsqrt(source: _VectorValue, mask: _MaskValue) -> _VectorValue:
    return _vunary("vsqrt", source, mask)


def _vvec_scalar(
    name: str,
    source: _VectorValue,
    scalar: _Value,
    mask: _MaskValue,
) -> _VectorValue:
    _validate_mask(f"pto.vmi.{name}", mask, source.dtype)
    builder = getattr(_vmi_builder, name)
    return _wrap_vreg(builder(source.value, scalar.value, mask.value), source.dtype)


def _vadds(source: _VectorValue, scalar: _Value, mask: _MaskValue) -> _VectorValue:
    return _vvec_scalar("vadds", source, scalar, mask)


def _negate_scalar(scalar: _Value, dtype: ScalarType) -> _Value:
    zero = _scalar_constant(0.0, dtype)
    return _Value(emit_runtime_binary_op("sub", zero.value, scalar.value))


def _vmuls(source: _VectorValue, scalar: _Value, mask: _MaskValue) -> _VectorValue:
    return _vvec_scalar("vmuls", source, scalar, mask)


def _vmaxs(source: _VectorValue, scalar: _Value, mask: _MaskValue) -> _VectorValue:
    return _vvec_scalar("vmaxs", source, scalar, mask)


def _vmins(source: _VectorValue, scalar: _Value, mask: _MaskValue) -> _VectorValue:
    return _vvec_scalar("vmins", source, scalar, mask)


def _vcmp(
    lhs: _VectorValue,
    rhs: _VectorValue,
    seed: _MaskValue,
    cmp: str,
) -> _MaskValue:
    dtype = _validate_same_dtype("pto.vmi.vcmp", lhs, rhs)
    _validate_mask("pto.vmi.vcmp", seed, dtype)
    return _wrap_mask(_vmi_builder.vcmp(lhs.value, rhs.value, seed.value, cmp), dtype)


def _vcmps(
    source: _VectorValue,
    scalar: _Value,
    seed: _MaskValue,
    cmp: str,
) -> _MaskValue:
    _validate_mask("pto.vmi.vcmps", seed, source.dtype)
    return _wrap_mask(
        _vmi_builder.vcmps(source.value, scalar.value, seed.value, cmp),
        source.dtype,
    )


def _vsel(
    true_value: _VectorValue,
    false_value: _VectorValue,
    mask: _MaskValue,
) -> _VectorValue:
    dtype = _validate_same_dtype("pto.vmi.vsel", true_value, false_value)
    _validate_mask("pto.vmi.vsel", mask, dtype)
    return _wrap_vreg(
        _vmi_builder.vsel(mask.value, true_value.value, false_value.value),
        dtype,
    )


def _vmula(
    acc: _VectorValue,
    lhs: _VectorValue,
    rhs: _VectorValue,
    mask: _MaskValue,
) -> _VectorValue:
    dtype = _validate_same_dtype("pto.vmi.vmula", acc, lhs, rhs)
    _validate_mask("pto.vmi.vmula", mask, dtype)
    return _wrap_vreg(
        _vmi_builder.vmula(acc.value, lhs.value, rhs.value, mask.value),
        dtype,
    )


def _pand(lhs: _MaskValue, rhs: _MaskValue) -> _MaskValue:
    if lhs.dtype.mask_bits != rhs.dtype.mask_bits:
        raise TypeError("pto.vmi.vand mask operands must use the same granularity")
    return _wrap_mask(_vmi_builder.vand(lhs.value, rhs.value), lhs.dtype)


def _por(lhs: _MaskValue, rhs: _MaskValue) -> _MaskValue:
    if lhs.dtype.mask_bits != rhs.dtype.mask_bits:
        raise TypeError("pto.vmi.vor mask operands must use the same granularity")
    return _wrap_mask(_vmi_builder.vor(lhs.value, rhs.value), lhs.dtype)


def _pnot(mask: _MaskValue) -> _MaskValue:
    return _wrap_mask(_vmi_builder.vnot(mask.value), mask.dtype)


def _scalar_constant(value: float | int, dtype: ScalarType) -> _Value:
    return _Value(unwrap_surface_value(pto.const(value, dtype=_pto_dtype(dtype))))


def _vbrc(source: _VectorValue, *, lanes: int) -> _VectorValue:
    if not isinstance(lanes, int) or lanes <= 0:
        raise ValueError("_vbrc lanes must be a positive integer")
    return _wrap_vreg(_vmi_builder.vbrc(source.value, size=lanes), source.dtype)


def _vbrc_scalar(
    scalar: _Value,
    *,
    like: _VectorValue | None = None,
    dtype: ScalarType | None = None,
) -> _VectorValue:
    if like is None and dtype is None:
        raise TypeError("_vbrc_scalar requires like= or dtype=")
    ref_dtype = like.dtype if like is not None else dtype
    size = _vreg_lanes(like) if like is not None else dtype.lanes
    return _wrap_vreg(_vmi_builder.vbrc(scalar.value, size=size), ref_dtype)


def _vconstant(
    value: float | int,
    dtype: ScalarType,
    *,
    like: _VectorValue | None = None,
    lanes: int | None = None,
) -> _VectorValue:
    if like is None and lanes is None:
        raise TypeError("_vconstant requires like= or lanes=")
    if like is not None and lanes is not None:
        raise TypeError("_vconstant lanes cannot be combined with like=")
    scalar = _scalar_constant(value, dtype)
    if like is not None:
        return _vbrc_scalar(scalar, like=like)
    return _wrap_vreg(_vmi_builder.vbrc(scalar.value, size=lanes), dtype)


def _vreduce_max(source: _VectorValue, mask: _MaskValue) -> _VectorValue:
    _validate_mask("pto.vmi.vcmax", mask, source.dtype)
    return _wrap_vreg(_vmi_builder.vcmax(source.value, mask.value), source.dtype)


def _vreduce_add(source: _VectorValue, mask: _MaskValue) -> _VectorValue:
    _validate_mask("pto.vmi.vcadd", mask, source.dtype)
    return _wrap_vreg(
        _vmi_builder.vcadd(source.value, mask.value, reassoc=True),
        source.dtype,
    )


def _vcvt(source: _VectorValue, dst_dtype: ScalarType) -> _VectorValue:
    if not isinstance(dst_dtype, ScalarType):
        raise TypeError("_vcvt expects a tile-template destination ScalarType")
    return _wrap_vreg(
        _vmi_builder.vcvt(source.value, to_dtype=_pto_dtype(dst_dtype)),
        dst_dtype,
    )


def _vinterpret_cast(source: _VectorValue, dst_dtype: ScalarType) -> _VectorValue:
    if not isinstance(dst_dtype, ScalarType):
        raise TypeError("_vinterpret_cast expects a destination ScalarType")
    source_type = _pto_dialect.VMIVRegType(source.value.type)
    source_bits = source_type.element_count * source.dtype.bytewidth * 8
    dst_bits = dst_dtype.bytewidth * 8
    if source_bits % dst_bits != 0:
        raise ValueError("_vinterpret_cast requires matching total bit width")
    return _wrap_vreg(
        _vmi_builder.vinterpret_cast(source.value, to_dtype=_pto_dtype(dst_dtype)),
        dst_dtype,
    )


def _qualify_op_name(op: str) -> str:
    return op if op.startswith("pto.") else f"pto.{op}"


def _normalize_op_name(op: str) -> str:
    return op[4:] if op.startswith("pto.") else op


class _VMITileTemplateRegistry(TileTemplateRegistry):
    def lookup(self, op: str, target: str) -> list:
        candidates = super().lookup(op, target)
        if candidates:
            return candidates
        qualified = _qualify_op_name(op)
        if qualified != op:
            candidates = super().lookup(qualified, target)
            if candidates:
                return candidates
        normalized = _normalize_op_name(op)
        if normalized != op:
            return super().lookup(normalized, target)
        return []


VMI_TILELIB_REGISTRY = _VMITileTemplateRegistry()


# Reduce kind -> (merge op, identity element). The identity mirrors pto-isa
# `TColReduceOps.hpp` `InstrOp::InitVal` / a5 `Padding<T>::Min/Max`:
#   max -> vmax, init -inf (Padding<T>::Min)
#   min -> vmin, init +inf (Padding<T>::Max)
#   add -> vadd, init 0
#   prod-> vmul, init 1
# `emit_col_reduce_vmi` only exercises max/add today; min/prod map to vmi
# merge ops that the VMI tilelib does not yet expose as elementwise-vector
# forms (only the -s scalar variants), so they raise if used.
_REDUCE_MERGE_OP = {
    "max": _vmax,
    "add": _vadd,
}


def canonical_vmi_template(
    *,
    target: str = "a5",
    op: str,
    name: str | None = None,
    dtypes: tuple | list = (),
    context_constraints: dict[str, tuple[object, ...]] | None = None,
    constraints: tuple[object, ...] | list[object] = (),
    tags: tuple[str, ...] | list[str] = (),
):
    """Register one canonical VMI implementation in this provider module."""

    def decorator(fn):
        qualified_op = _qualify_op_name(op)
        descriptor = _trace_tile_template(
            target=target,
            op=qualified_op,
            name=name,
            ir_level="vmi",
            dtypes=dtypes,
            context_constraints=context_constraints,
            constraints=tuple(constraints),
            tags=tuple(tags),
        )(fn)
        _tilelib_registry.register(descriptor)
        VMI_TILELIB_REGISTRY.register(descriptor)
        return descriptor

    return decorator


def emit_elementwise_vmi(
    dst: _TileProxy,
    sources: Sequence[_TileProxy],
    compute: ElementwiseCompute,
    *,
    logical_lanes: int | None = None,
    allowed_dtypes: Sequence[ScalarType] = (f32,),
) -> None:
    """Emit one flat logical-row loop for a standalone elementwise candidate."""

    if not sources:
        raise ValueError("emit_elementwise_vmi requires at least one source tile")
    if logical_lanes is None:
        logical_lanes = dst._spec.shape[1]
    _validate_elementwise_tiles(
        dst,
        sources,
        logical_lanes=logical_lanes,
        allowed_dtypes=allowed_dtypes,
    )
    block_map = CanonicalBlockMap.from_tile(dst, logical_lanes=logical_lanes)

    _prepare_tile_access(*sources, dst)
    mask = _create_mask(block_map, dst.element_type, trace=dst._trace)
    with for_(0, block_map.logical_block_count, step=1) as logical_block:
        coordinate = block_map.coordinate(logical_block)
        values = tuple(_vload(source, coordinate) for source in sources)
        result = compute(values, mask)
        _vstore(result, dst, coordinate, mask)


def emit_scalar_fill_vmi(
    scalar: _Value,
    dst: _TileProxy,
    *,
    allowed_dtypes: Sequence[ScalarType] = (f32,),
) -> None:
    """Broadcast a runtime scalar into each wide logical row of ``dst``."""

    if not isinstance(dst, _TileProxy):
        raise TypeError("scalar-fill VMI candidate destination must be a traced Tile")
    if dst.element_type not in allowed_dtypes:
        raise ValueError(
            "VMI scalar-fill candidate dtype is not supported; "
            f"got {dst.element_type}, expected one of {tuple(allowed_dtypes)}"
        )
    if dst._spec.b_layout != "row_major":
        raise ValueError("VMI scalar-fill candidates require row-major tiles")

    block_map = CanonicalBlockMap.from_tile(dst)
    _prepare_tile_access(dst)
    mask = _create_mask(block_map, dst.element_type, trace=dst._trace)
    fill = _wrap_vreg(
        _vmi_builder.vbrc(scalar.value, size=block_map.logical_lanes),
        dst.element_type,
    )
    with for_(0, block_map.logical_block_count, step=1) as logical_block:
        _vstore(fill, dst, block_map.coordinate(logical_block), mask)


def _validate_elementwise_tiles(
    dst: _TileProxy,
    sources: Sequence[_TileProxy],
    *,
    logical_lanes: int,
    allowed_dtypes: Sequence[ScalarType],
) -> None:
    if not isinstance(dst, _TileProxy):
        raise TypeError("elementwise VMI candidate destination must be a traced Tile")
    if dst.element_type not in allowed_dtypes:
        raise ValueError(
            "VMI elementwise candidate dtype is not supported; "
            f"got {dst.element_type}, expected one of {tuple(allowed_dtypes)}"
        )
    if dst._spec.b_layout != "row_major":
        raise ValueError("VMI elementwise candidates require row-major tiles")
    for source in sources:
        if not isinstance(source, _TileProxy):
            raise TypeError("elementwise VMI candidate sources must be traced Tiles")
        if source._spec.shape != dst._spec.shape:
            raise ValueError(
                "elementwise VMI candidate source and destination shapes must match; "
                f"got {source._spec.shape} and {dst._spec.shape}"
            )
        if source.element_type != dst.element_type:
            raise ValueError(
                "elementwise VMI candidate source and destination dtypes must match; "
                f"got {source.element_type} and {dst.element_type}"
            )
        if source._spec.b_layout != dst._spec.b_layout:
            raise ValueError("elementwise VMI candidate layouts must match")


def _add(values: Sequence[_VectorValue], mask: _MaskValue) -> _VectorValue:
    if len(values) != 2:
        raise ValueError("tadd VMI candidate expects two source vectors")
    return _vadd(values[0], values[1], mask)


def _exp(values: Sequence[_VectorValue], mask: _MaskValue) -> _VectorValue:
    if len(values) != 1:
        raise ValueError("texp VMI candidate expects one source vector")
    return _vexp(values[0], mask)


def _abs(values: Sequence[_VectorValue], mask: _MaskValue) -> _VectorValue:
    if len(values) != 1:
        raise ValueError("tabs VMI candidate expects one source vector")
    return _vabs(values[0], mask)


def _neg(values: Sequence[_VectorValue], mask: _MaskValue) -> _VectorValue:
    if len(values) != 1:
        raise ValueError("tneg VMI candidate expects one source vector")
    return _vneg(values[0], mask)


def _sub(values: Sequence[_VectorValue], mask: _MaskValue) -> _VectorValue:
    if len(values) != 2:
        raise ValueError("tsub VMI candidate expects two source vectors")
    return _vsub(values[0], values[1], mask)


def _mul(values: Sequence[_VectorValue], mask: _MaskValue) -> _VectorValue:
    if len(values) != 2:
        raise ValueError("tmul VMI candidate expects two source vectors")
    return _vmul(values[0], values[1], mask)


def _max(values: Sequence[_VectorValue], mask: _MaskValue) -> _VectorValue:
    if len(values) != 2:
        raise ValueError("tmax VMI candidate expects two source vectors")
    return _vmax(values[0], values[1], mask)


def _move(values: Sequence[_VectorValue], mask: _MaskValue) -> _VectorValue:
    if len(values) != 1:
        raise ValueError("tmov VMI candidate expects one source vector")
    return values[0]


def _divide_by_scalar(
    value: _VectorValue, scalar: _Value, mask: _MaskValue
) -> _VectorValue:
    scalar_vector = _vbrc_scalar(scalar, like=value)
    return _vdiv(value, scalar_vector, mask)


def _divide_scalar_by_vector(
    scalar: _Value, value: _VectorValue, mask: _MaskValue
) -> _VectorValue:
    scalar_vector = _vbrc_scalar(scalar, like=value)
    return _vdiv(scalar_vector, value, mask)


def _mask_as(mask: _MaskValue, dtype: ScalarType) -> _MaskValue:
    return _MaskValue(mask.value, dtype)


def _vbrc_constant(
    value: float | int, dtype: ScalarType, like: _VectorValue
) -> _VectorValue:
    if dtype.name.startswith("ui"):
        return _vconstant(value, dtype, like=like)
    return _vbrc_scalar(_scalar_constant(value, dtype), like=like)


def _div_three_candidate_search_f32(
    lhs: _VectorValue, rhs: _VectorValue, mask: _MaskValue
) -> _VectorValue:
    lhs_u32 = _vinterpret_cast(lhs, ui32)
    inf_bound = _vbrc_constant(0x7F800000, ui32, like=lhs_u32)
    sign_bit = _vbrc_constant(0x80000000, ui32, like=lhs_u32)
    zero = _vbrc_constant(0.0, f32, like=lhs)
    one = _vbrc_constant(1.0, f32, like=lhs)
    neg_one = _vbrc_constant(-1.0, f32, like=lhs)

    z = _vdiv(lhs, rhs, mask)
    z_u32 = _vinterpret_cast(z, ui32)
    z_or_sign = _vor(z_u32, sign_bit, _mask_as(mask, ui32))
    is_inf_nan = _vcmp(z_or_sign, inf_bound, _mask_as(mask, ui32), "ge")
    is_zero = _vcmp(z, zero, mask, "eq")
    special_mask = _por(is_inf_nan, is_zero)

    y = _vmul(rhs, neg_one, mask)
    residual = _vmula(lhs, z, y, mask)
    z_pre = _vadd(z, neg_one, mask)
    z_next = _vadd(z, one, mask)
    residual_pre = _vmula(lhs, z_pre, y, mask)
    residual_next = _vmula(lhs, z_next, y, mask)

    residual_abs = _vabs(residual, mask)
    residual_pre_abs = _vabs(residual_pre, mask)
    residual_next_abs = _vabs(residual_next, mask)
    better_pre = _vcmp(residual_pre_abs, residual_abs, mask, "lt")
    z_best = _vsel(z_pre, z, better_pre)
    residual_best_abs = _vsel(residual_pre_abs, residual_abs, better_pre)
    better_next = _vcmp(residual_next_abs, residual_best_abs, mask, "lt")
    z_best = _vsel(z_next, z_best, better_next)
    return _vsel(z, z_best, special_mask)


def _div_ieee754_f32_vmi(
    src0: _VectorValue, src1: _VectorValue, mask: _MaskValue
) -> _VectorValue:
    int_mask = _mask_as(mask, ui32)
    src0_u32 = _vinterpret_cast(src0, ui32)
    f32_inf = _vbrc_constant(0x7F800000, ui32, like=src0_u32)
    sign_extractor = _vbrc_constant(0x80000000, ui32, like=src0_u32)
    exponent_extractor = _vbrc_constant(0x807FFFFF, ui32, like=src0_u32)
    exponent_normalizer = _vbrc_constant(0x3F800000, ui32, like=src0_u32)
    subnormal_threshold = _vbrc_constant(0x007FFFFF, ui32, like=src0_u32)
    nan_value = _vbrc_constant(0x7FC00000, ui32, like=src0_u32)
    min_denormal = _vbrc_constant(0x1, ui32, like=src0_u32)
    zero_u32 = _vbrc_constant(0, ui32, like=src0_u32)
    normalize_scale_enlarge = _vbrc_constant(8388608.0, f32, like=src0)
    normalize_scale_reduce = _vbrc_constant(1.1920928955078125e-07, f32, like=src0)

    src0_abs = _vabs(src0, mask)
    src1_abs = _vabs(src1, mask)
    src0_abs_u32 = _vinterpret_cast(src0_abs, ui32)
    src1_abs_u32 = _vinterpret_cast(src1_abs, ui32)

    mask_inf_src0 = _vcmp(src0_abs_u32, f32_inf, int_mask, "eq")
    mask_inf_src1 = _vcmp(src1_abs_u32, f32_inf, int_mask, "eq")
    mask_invalid = _por(mask_inf_src0, mask_inf_src1)
    mask_zero_src0 = _vcmp(src0_abs_u32, zero_u32, int_mask, "eq")
    mask_invalid = _por(mask_invalid, mask_zero_src0)
    mask_zero_src1 = _vcmp(src1_abs_u32, zero_u32, int_mask, "eq")
    mask_invalid = _por(mask_invalid, mask_zero_src1)
    mask_valid = _pnot(mask_invalid)

    mask_src0_subnormal = _vcmp(src0_abs_u32, subnormal_threshold, int_mask, "eq")
    mask_src0_normal = _pnot(mask_src0_subnormal)
    src0_subnormal = _vmul(
        src0, normalize_scale_enlarge, _mask_as(mask_src0_subnormal, f32)
    )
    mask_src1_subnormal = _vcmp(src1_abs_u32, subnormal_threshold, int_mask, "lt")
    mask_src1_normal = _pnot(mask_src1_subnormal)
    src1_subnormal = _vmul(
        src1, normalize_scale_enlarge, _mask_as(mask_src1_subnormal, f32)
    )

    src0_all = _vsel(src0, src0_subnormal, _mask_as(mask_src0_normal, f32))
    src1_all = _vsel(src1, src1_subnormal, _mask_as(mask_src1_normal, f32))
    src0_all_u32 = _vinterpret_cast(src0_all, ui32)
    src1_all_u32 = _vinterpret_cast(src1_all, ui32)

    src0_norm_u32 = _vand(src0_all_u32, exponent_extractor, mask_valid)
    src1_norm_u32 = _vand(src1_all_u32, exponent_extractor, mask_valid)
    src0_norm_u32 = _vadd(src0_norm_u32, exponent_normalizer, mask_valid)
    src1_norm_u32 = _vadd(src1_norm_u32, exponent_normalizer, mask_valid)
    src0_norm = _vsel(
        _vinterpret_cast(src0_norm_u32, f32), src0_all, _mask_as(mask_valid, f32)
    )
    src1_norm = _vsel(
        _vinterpret_cast(src1_norm_u32, f32), src1_all, _mask_as(mask_valid, f32)
    )

    divided = _div_three_candidate_search_f32(
        src0_norm, src1_norm, _mask_as(mask_valid, f32)
    )
    mask0 = _pand(mask_src0_subnormal, mask_src1_normal)
    divided = _vsel(
        _vmul(divided, normalize_scale_reduce, _mask_as(mask0, f32)),
        divided,
        _mask_as(mask0, f32),
    )
    mask0 = _pand(mask_src0_normal, mask_src1_subnormal)
    divided = _vsel(
        _vmul(divided, normalize_scale_enlarge, _mask_as(mask0, f32)),
        divided,
        _mask_as(mask0, f32),
    )

    divided_u32 = _vinterpret_cast(divided, ui32)
    divided_sign = _vand(divided_u32, sign_extractor, int_mask)
    src0_exponent = _vand(src0_all_u32, f32_inf, int_mask)
    src1_exponent = _vand(src1_all_u32, f32_inf, int_mask)
    shift23 = _vbrc_constant(23, ui32, like=src0_exponent)
    src0_exp_shifted = _vshr(src0_exponent, shift23, int_mask)
    src1_exp_shifted = _vshr(src1_exponent, shift23, int_mask)

    scale = _vinterpret_cast(
        _vsub(src0_exp_shifted, src1_exp_shifted, int_mask), i32
    )
    scale_mask = _mask_as(mask, i32)
    scale = _vadds(scale, _scalar_constant(127, i32), scale_mask)

    neg23 = _vbrc_constant(-23, i32, like=scale)
    mask_underflow1 = _vcmp(scale, neg23, scale_mask, "eq")
    mask_underflow1 = _pand(mask_underflow1, mask_valid)
    z1_u32 = _vadd(divided_sign, min_denormal, mask_underflow1)
    z2_u32 = _vadd(divided_sign, zero_u32, mask_underflow1)

    src0_norm_abs = _vabs(src0_norm, _mask_as(mask_valid, f32))
    src1_norm_abs = _vabs(src1_norm, _mask_as(mask_valid, f32))
    mask_norm = _vcmp(src0_norm_abs, src1_norm_abs, _mask_as(mask_valid, f32), "le")
    divided_u32_temp = _vsel(
        _vsel(z2_u32, z1_u32, mask_norm), divided_u32, mask_underflow1
    )

    mask_valid_temp = _pand(_pnot(mask_underflow1), mask_valid)
    mask_underflow2 = _vcmp(scale, neg23, scale_mask, "lt")
    mask_underflow2 = _pand(mask_underflow2, mask_valid_temp)
    divided_u32_temp = _vsel(
        _vadd(divided_sign, zero_u32, mask_underflow2),
        divided_u32_temp,
        mask_underflow2,
    )

    mask_valid_temp = _pand(_pnot(mask_underflow2), mask_valid_temp)
    max_exp = _vbrc_constant(255, i32, like=scale)
    mask_overflow1 = _vcmp(scale, max_exp, scale_mask, "eq")
    mask_overflow1 = _pand(mask_overflow1, mask_valid_temp)
    scale = _vsel(
        _vadds(scale, _scalar_constant(-1, i32), mask_overflow1),
        scale,
        mask_overflow1,
    )

    divided_f32_temp = _vinterpret_cast(divided_u32_temp, f32)
    divided_f32_temp = _vsel(
        _vmul(
            divided_f32_temp,
            _vbrc_constant(2.0, f32, like=src0),
            _mask_as(mask_overflow1, f32),
        ),
        divided_f32_temp,
        _mask_as(mask_overflow1, f32),
    )

    mask_overflow2 = _vcmp(scale, max_exp, scale_mask, "gt")
    mask_overflow2 = _pand(mask_overflow2, mask_valid_temp)
    divided_u32_temp = _vsel(
        _vadd(divided_sign, f32_inf, mask_overflow2),
        _vinterpret_cast(divided_f32_temp, ui32),
        mask_overflow2,
    )

    mask_valid_final = _pand(_pnot(mask_overflow2), mask_valid_temp)
    zero_exp = _vbrc_constant(0, i32, like=scale)
    mask_pos_exp = _vcmp(scale, zero_exp, _mask_as(mask_valid_final, i32), "gt")
    scale_u32 = _vinterpret_cast(scale, ui32)
    exp_shifted = _vshl(scale_u32, shift23, _mask_as(mask_pos_exp, ui32))
    exp_factor_f32 = _vinterpret_cast(exp_shifted, f32)
    divided_f32_temp = _vinterpret_cast(divided_u32_temp, f32)
    divided_f32_temp = _vsel(
        _vmul(divided_f32_temp, exp_factor_f32, _mask_as(mask_pos_exp, f32)),
        divided_f32_temp,
        _mask_as(mask_pos_exp, f32),
    )

    mask_pos_exp_not = _pnot(mask_pos_exp)
    scale_abs = _vabs(scale, mask_pos_exp_not)
    shr_factor_u32 = _vshr(
        _vbrc_constant(4194304, ui32, like=scale_u32),
        _vinterpret_cast(scale_abs, ui32),
        _mask_as(mask_pos_exp_not, ui32),
    )
    divided_f32_temp = _vsel(
        _vmul(
            divided_f32_temp,
            _vinterpret_cast(shr_factor_u32, f32),
            _mask_as(mask_pos_exp_not, f32),
        ),
        divided_f32_temp,
        _mask_as(mask_pos_exp_not, f32),
    )

    mask_nan = _por(
        _vcmp(src0_abs, src0_abs, mask, "ne"),
        _vcmp(src1_abs, src1_abs, mask, "ne"),
    )
    return _vsel(
        _vinterpret_cast(nan_value, f32), divided_f32_temp, mask_nan
    )


def _div_ieee754_f16_vmi(
    src0: _VectorValue, src1: _VectorValue, mask: _MaskValue
) -> _VectorValue:
    int_mask = _mask_as(mask, ui16)
    src0_u16 = _vinterpret_cast(src0, ui16)
    f16_inf = _vbrc_constant(0x7C00, ui16, like=src0_u16)
    exponent_extractor = _vbrc_constant(0x83FF, ui16, like=src0_u16)
    exponent_normalizer = _vbrc_constant(0x3C00, ui16, like=src0_u16)
    sign_extractor = _vbrc_constant(0x8000, ui16, like=src0_u16)
    subnormal_threshold = _vbrc_constant(0x03FF, ui16, like=src0_u16)
    nan_value = _vbrc_constant(0x7E00, ui16, like=src0_u16)
    min_denormal = _vbrc_constant(0x1, ui16, like=src0_u16)
    zero_u16 = _vbrc_constant(0, ui16, like=src0_u16)
    normalize_scale_enlarge = _vbrc_constant(1024.0, f16, like=src0)
    normalize_scale_reduce = _vbrc_constant(0.0009765625, f16, like=src0)

    src0_abs = _vabs(src0, mask)
    src1_abs = _vabs(src1, mask)
    src0_abs_u16 = _vinterpret_cast(src0_abs, ui16)
    src1_abs_u16 = _vinterpret_cast(src1_abs, ui16)

    mask_inf_src0 = _vcmp(src0_abs_u16, f16_inf, int_mask, "eq")
    mask_inf_src1 = _vcmp(src1_abs_u16, f16_inf, int_mask, "eq")
    mask_invalid = _por(mask_inf_src0, mask_inf_src1)
    mask_zero_src0 = _vcmp(src0_abs_u16, zero_u16, int_mask, "eq")
    mask_invalid = _por(mask_invalid, mask_zero_src0)
    mask_zero_src1 = _vcmp(src1_abs_u16, zero_u16, int_mask, "eq")
    mask_invalid = _por(mask_invalid, mask_zero_src1)
    mask_valid = _pnot(mask_invalid)

    mask_src0_subnormal = _vcmp(src0_abs_u16, subnormal_threshold, int_mask, "lt")
    mask_src0_normal = _pnot(mask_src0_subnormal)
    src0_subnormal = _vmul(
        src0, normalize_scale_enlarge, _mask_as(mask_src0_subnormal, f16)
    )
    mask_src1_subnormal = _vcmp(src1_abs_u16, subnormal_threshold, int_mask, "lt")
    mask_src1_normal = _pnot(mask_src1_subnormal)
    src1_subnormal = _vmul(
        src1, normalize_scale_enlarge, _mask_as(mask_src1_subnormal, f16)
    )

    src0_all = _vsel(src0, src0_subnormal, _mask_as(mask_src0_normal, f16))
    src1_all = _vsel(src1, src1_subnormal, _mask_as(mask_src1_normal, f16))
    src0_all_u16 = _vinterpret_cast(src0_all, ui16)
    src1_all_u16 = _vinterpret_cast(src1_all, ui16)

    src0_norm_u16 = _vand(src0_all_u16, exponent_extractor, mask_valid)
    src1_norm_u16 = _vand(src1_all_u16, exponent_extractor, mask_valid)
    src0_norm_u16 = _vadd(src0_norm_u16, exponent_normalizer, mask_valid)
    src1_norm_u16 = _vadd(src1_norm_u16, exponent_normalizer, mask_valid)
    src0_norm = _vsel(
        _vinterpret_cast(src0_norm_u16, f16), src0_all, _mask_as(mask_valid, f16)
    )
    src1_norm = _vsel(
        _vinterpret_cast(src1_norm_u16, f16), src1_all, _mask_as(mask_valid, f16)
    )

    src0_norm_abs = _vabs(src0_norm, _mask_as(mask_valid, f16))
    src1_norm_abs = _vabs(src1_norm, _mask_as(mask_valid, f16))
    mask_norm = _vcmp(src0_norm_abs, src1_norm_abs, _mask_as(mask_valid, f16), "le")
    divided = _vdiv(src0_norm, src1_norm, _mask_as(mask_valid, f16))

    mask0 = _pand(mask_src0_subnormal, mask_src1_normal)
    divided = _vsel(
        _vmul(divided, normalize_scale_reduce, _mask_as(mask0, f16)),
        divided,
        _mask_as(mask0, f16),
    )
    mask0 = _pand(mask_src0_normal, mask_src1_subnormal)
    divided = _vsel(
        _vmul(divided, normalize_scale_enlarge, _mask_as(mask0, f16)),
        divided,
        _mask_as(mask0, f16),
    )

    divided_u16 = _vinterpret_cast(divided, ui16)
    divided_sign = _vand(divided_u16, sign_extractor, int_mask)
    src0_exponent = _vand(src0_all_u16, f16_inf, int_mask)
    src1_exponent = _vand(src1_all_u16, f16_inf, int_mask)
    shift10 = _vbrc_constant(10, ui16, like=src0_exponent)
    src0_exp_shifted = _vshr(src0_exponent, shift10, int_mask)
    src1_exp_shifted = _vshr(src1_exponent, shift10, int_mask)

    scale = _vinterpret_cast(
        _vsub(src0_exp_shifted, src1_exp_shifted, int_mask), i16
    )
    scale_mask = _mask_as(mask, i16)
    scale = _vadds(scale, _scalar_constant(15, i16), scale_mask)

    neg9 = _vbrc_constant(-9, i16, like=scale)
    mask_underflow1 = _vcmp(scale, neg9, scale_mask, "eq")
    mask_underflow1 = _pand(mask_underflow1, mask_valid)
    z1_u16 = _vadd(divided_sign, min_denormal, mask_underflow1)
    z2_u16 = _vadd(divided_sign, zero_u16, mask_underflow1)
    divided_u16_temp = _vsel(
        _vsel(z2_u16, z1_u16, mask_norm), divided_u16, mask_underflow1
    )

    mask_valid_temp = _pand(_pnot(mask_underflow1), mask_valid)
    mask_underflow2 = _vcmp(scale, neg9, scale_mask, "lt")
    mask_underflow2 = _pand(mask_underflow2, mask_valid_temp)
    divided_u16_temp = _vsel(
        _vadd(divided_sign, zero_u16, mask_underflow2),
        divided_u16_temp,
        mask_underflow2,
    )

    mask_valid_temp = _pand(_pnot(mask_underflow2), mask_valid_temp)
    max_exp = _vbrc_constant(31, i16, like=scale)
    mask_overflow1 = _vcmp(scale, max_exp, scale_mask, "eq")
    mask_overflow1 = _pand(mask_overflow1, mask_valid_temp)
    scale = _vsel(
        _vadds(scale, _scalar_constant(-1, i16), mask_overflow1),
        scale,
        mask_overflow1,
    )

    divided_f16_temp = _vinterpret_cast(divided_u16_temp, f16)
    divided_f16_temp = _vsel(
        _vmul(
            divided_f16_temp,
            _vbrc_constant(2.0, f16, like=src0),
            _mask_as(mask_overflow1, f16),
        ),
        divided_f16_temp,
        _mask_as(mask_overflow1, f16),
    )

    mask_overflow2 = _vcmp(scale, max_exp, scale_mask, "gt")
    mask_overflow2 = _pand(mask_overflow2, mask_valid_temp)
    divided_u16_temp = _vsel(
        _vadd(divided_sign, f16_inf, mask_overflow2),
        _vinterpret_cast(divided_f16_temp, ui16),
        mask_overflow2,
    )

    mask_valid_final = _pand(_pnot(mask_overflow2), mask_valid_temp)
    zero_exp = _vbrc_constant(0, i16, like=scale)
    mask_pos_exp = _vcmp(scale, zero_exp, _mask_as(mask_valid_final, i16), "gt")
    scale_u16 = _vinterpret_cast(scale, ui16)
    exp_factor_f16 = _vinterpret_cast(
        _vshl(scale_u16, shift10, _mask_as(mask_pos_exp, ui16)), f16
    )
    divided_f16_temp = _vinterpret_cast(divided_u16_temp, f16)
    divided_f16_temp = _vsel(
        _vmul(divided_f16_temp, exp_factor_f16, _mask_as(mask_pos_exp, f16)),
        divided_f16_temp,
        _mask_as(mask_pos_exp, f16),
    )

    mask_pos_exp_not = _pnot(mask_pos_exp)
    scale_abs = _vabs(scale, mask_pos_exp_not)
    shr_factor_u16 = _vshr(
        _vbrc_constant(512, ui16, like=scale_u16),
        _vinterpret_cast(scale_abs, ui16),
        _mask_as(mask_pos_exp_not, ui16),
    )
    divided_f16_temp = _vsel(
        _vmul(
            divided_f16_temp,
            _vinterpret_cast(shr_factor_u16, f16),
            _mask_as(mask_pos_exp_not, f16),
        ),
        divided_f16_temp,
        _mask_as(mask_pos_exp_not, f16),
    )

    mask_nan = _por(
        _vcmp(src0_abs, src0_abs, mask, "ne"),
        _vcmp(src1_abs, src1_abs, mask, "ne"),
    )
    return _vsel(
        _vinterpret_cast(nan_value, f16), divided_f16_temp, mask_nan
    )


def _div_high_precision(
    lhs: _VectorValue, rhs: _VectorValue, mask: _MaskValue
) -> _VectorValue:
    if lhs.dtype != rhs.dtype:
        raise ValueError("high-precision VMI division requires matching dtypes")
    if lhs.dtype == f32:
        return _div_ieee754_f32_vmi(lhs, rhs, mask)
    if lhs.dtype == f16:
        return _div_ieee754_f16_vmi(lhs, rhs, mask)
    raise ValueError("high-precision VMI division requires f16 or f32")


def _divide_by_scalar_high_precision(
    value: _VectorValue, scalar: _Value, mask: _MaskValue
) -> _VectorValue:
    return _div_high_precision(value, _vbrc_scalar(scalar, like=value), mask)


def _divide_scalar_by_vector_high_precision(
    scalar: _Value, value: _VectorValue, mask: _MaskValue
) -> _VectorValue:
    return _div_high_precision(_vbrc_scalar(scalar, like=value), value, mask)


def _sqrt_high_precision_f16(source: _VectorValue, mask: _MaskValue) -> _VectorValue:
    subnormal_mask = _vcmps(
        source,
        _scalar_constant(6.097555160522461e-05, f16),
        mask,
        "lt",
    )
    scaled_source = _vmuls(source, _scalar_constant(4096.0, f16), subnormal_mask)
    source_adjusted = _vsel(scaled_source, source, subnormal_mask)
    root = _vsqrt(source_adjusted, mask)
    scaled_root = _vmuls(root, _scalar_constant(0.015625, f16), subnormal_mask)
    return _vsel(scaled_root, root, subnormal_mask)


def _sqrt_high_precision_f32(source: _VectorValue, mask: _MaskValue) -> _VectorValue:
    subnormal_mask = _vcmps(source, _scalar_constant(1.0, f32), mask, "lt")
    scaled_source = _vmuls(
        source, _scalar_constant(16777216.0, f32), subnormal_mask
    )
    source_adjusted = _vsel(scaled_source, source, subnormal_mask)

    one = _vbrc_scalar(_scalar_constant(1.0, f32), like=source)
    root = _vsqrt(source_adjusted, mask)
    reciprocal = _vdiv(one, root, mask)
    neg_reciprocal = _vmuls(reciprocal, _scalar_constant(-1.0, f32), mask)
    err = _vmul(reciprocal, source_adjusted, mask)
    one_adjusted = _vmula(one, err, neg_reciprocal, mask)
    half_reciprocal = _vmuls(reciprocal, _scalar_constant(0.5, f32), mask)
    refined = _vmula(reciprocal, one_adjusted, half_reciprocal, mask)

    result = _vmul(refined, source_adjusted, mask)
    neg_result = _vmuls(result, _scalar_constant(-1.0, f32), mask)
    err = _vmula(source_adjusted, result, neg_result, mask)
    half_refined = _vmuls(refined, _scalar_constant(0.5, f32), mask)
    correction = _vmul(err, half_refined, mask)
    corrected = _vadd(correction, result, mask)

    scaled_corrected = _vmuls(
        corrected, _scalar_constant(0.000244140625, f32), mask
    )
    result = _vsel(scaled_corrected, corrected, subnormal_mask)

    source_bits = _vinterpret_cast(source_adjusted, ui32)
    is_inf = _vcmp(
        source_bits,
        _vbrc_constant(0x7F800000, ui32, like=source_bits),
        _mask_as(mask, ui32),
        "eq",
    )
    sign_bit = _vbrc_constant(0x80000000, ui32, like=source_bits)
    source_with_sign = _vor(source_bits, sign_bit, _mask_as(mask, ui32))
    is_zero = _vcmp(
        source_with_sign,
        _vbrc_constant(0x80000000, ui32, like=source_bits),
        _mask_as(mask, ui32),
        "eq",
    )
    return _vsel(source_adjusted, result, _por(is_zero, is_inf))


def _sqrt_high_precision(
    values: Sequence[_VectorValue], mask: _MaskValue
) -> _VectorValue:
    if len(values) != 1:
        raise ValueError("tsqrt high-precision VMI candidate expects one source vector")
    source = values[0]
    if source.dtype == f16:
        return _sqrt_high_precision_f16(source, mask)
    if source.dtype == f32:
        return _sqrt_high_precision_f32(source, mask)
    raise ValueError("tsqrt high-precision VMI candidate requires f16 or f32")


def _context_attr(tile: _TileProxy, name: str, default=None):
    return getattr(tile._trace, "context_attrs", {}).get(name, default)


def _operand_kinds_are(expected: tuple[str, ...]):
    def predicate(operand_kinds=(), **_):
        return tuple(operand_kinds) == expected

    return predicate


def emit_sqrt_vmi(src: _TileProxy, dst: _TileProxy) -> None:
    emit_elementwise_vmi(
        dst,
        (src,),
        lambda values, mask: _vsqrt(values[0], mask),
        allowed_dtypes=FLOAT_DTYPES,
    )


def emit_sqrt_high_precision_vmi(src: _TileProxy, dst: _TileProxy) -> None:
    emit_elementwise_vmi(
        dst,
        (src,),
        _sqrt_high_precision,
        allowed_dtypes=FLOAT_DTYPES,
    )


def emit_recip_vmi(src: _TileProxy, dst: _TileProxy, *, high_precision: bool) -> None:
    def reciprocal(values, mask):
        one = _vbrc_scalar(
            _scalar_constant(1.0, values[0].dtype), like=values[0]
        )
        if high_precision:
            return _div_high_precision(one, values[0], mask)
        return _vdiv(one, values[0], mask)

    emit_elementwise_vmi(dst, (src,), reciprocal, allowed_dtypes=FLOAT_DTYPES)


def emit_rsqrt_vmi(
    src: _TileProxy,
    dst: _TileProxy,
    *,
    high_precision: bool,
) -> None:
    def reciprocal_sqrt(values, mask):
        root = (
            _sqrt_high_precision(values, mask)
            if high_precision
            else _vsqrt(values[0], mask)
        )
        one = _vbrc_scalar(
            _scalar_constant(1.0, values[0].dtype), like=values[0]
        )
        if high_precision:
            return _div_high_precision(one, root, mask)
        return _vdiv(one, root, mask)

    emit_elementwise_vmi(dst, (src,), reciprocal_sqrt, allowed_dtypes=FLOAT_DTYPES)


def _validate_row_reduce_tiles(
    src: _TileProxy, workspace: _TileProxy, dst: _TileProxy
) -> tuple[int, int, int]:
    if (
        src.element_type != f32
        or workspace.element_type != f32
        or dst.element_type != f32
    ):
        raise ValueError("row-reduce VMI candidates currently support only f32")
    if src._spec.b_layout != "row_major":
        raise ValueError("row-reduce source must be row-major")
    rows, physical_cols = src._spec.shape
    valid_rows, valid_cols = src._spec.effective_valid_shape
    if valid_rows != rows:
        raise ValueError("row-reduce VMI candidates do not support tail rows")
    if valid_cols <= 0 or valid_cols > physical_cols:
        raise ValueError("row-reduce valid columns must be within the source tile")
    workspace_rows, workspace_cols = workspace._spec.shape
    if workspace_rows != rows or workspace_cols < 1:
        raise ValueError("row-reduce workspace must have matching rows")
    if dst._spec.shape != (rows, 1) or dst._spec.b_layout != "col_major":
        raise ValueError("row-reduce destination must be a col-major [rows, 1] tile")
    if dst._spec.effective_valid_shape != (rows, 1):
        raise ValueError("row-reduce destination valid shape must be [rows, 1]")
    return rows, physical_cols, valid_cols


def emit_row_reduce_vmi(
    src: _TileProxy,
    workspace: _TileProxy,
    dst: _TileProxy,
    *,
    kind: str,
) -> None:
    rows, physical_cols, valid_cols = _validate_row_reduce_tiles(src, workspace, dst)
    reduce_op = _vreduce_max if kind == "max" else _vreduce_add

    _prepare_tile_access(src, dst)
    full_mask = _create_mask_lanes(valid_cols, valid_cols, f32, trace=src._trace)
    scalar_mask = _create_mask_lanes(1, 1, f32, trace=src._trace)
    with for_(0, rows, step=1) as row:
        row_offset = index_mul(row, physical_cols)
        accumulator = reduce_op(
            _vload_linear(src, row_offset, lanes=valid_cols), full_mask
        )
        _vstore_linear(accumulator, dst, row, scalar_mask)


def emit_row_expand_sub_vmi(
    src: _TileProxy, row_values: _TileProxy, dst: _TileProxy
) -> None:
    if (
        src.element_type != f32
        or row_values.element_type != f32
        or dst.element_type != f32
    ):
        raise ValueError("trowexpandsub VMI candidate currently supports only f32")
    if src._spec.shape != dst._spec.shape:
        raise ValueError("trowexpandsub source and destination shapes must match")
    if src._spec.b_layout != "row_major" or dst._spec.b_layout != "row_major":
        raise ValueError("trowexpandsub source and destination must be row-major")
    rows, cols = src._spec.shape
    if (
        row_values._spec.shape != (rows, 1)
        or row_values._spec.b_layout != "col_major"
    ):
        raise ValueError("trowexpandsub row values must be a col-major [rows, 1] tile")
    block_map = CanonicalBlockMap.from_tile(src, logical_lanes=cols)

    _prepare_tile_access(src, row_values, dst)
    full_mask = _create_mask(block_map, f32, trace=src._trace)
    with for_(0, rows, step=1) as row:
        row_scalar = _vload_linear(row_values, row, lanes=1)
        broadcast = _vbrc(row_scalar, lanes=cols)
        row_block_base = index_mul(row, block_map.blocks_per_row)
        for block_in_row in range(block_map.blocks_per_row):
            coordinate = block_map.coordinate(
                index_add(row_block_base, block_in_row)
            )
            value = _vload(src, coordinate)
            result = _vsub(value, broadcast, full_mask)
            _vstore(result, dst, coordinate, full_mask)


def _validate_col_reduce_tiles(
    src: _TileProxy, dst: _TileProxy
) -> CanonicalBlockMap:
    """Validate tiles for a ColReduce (tcolmax / tcolsum) VMI candidate.

    Mirror of `_validate_row_reduce_tiles` but the surviving axis is the column
    dimension: src is [rows, cols] row-major, dst is [1, cols] row-major, and the
    reduction runs across all rows as a single logical row-width vector.
    """
    if src.element_type != f32 or dst.element_type != f32:
        raise ValueError("col-reduce VMI candidates currently support only f32")
    if src._spec.b_layout != "row_major" or dst._spec.b_layout != "row_major":
        raise ValueError("col-reduce source and destination must be row-major")
    rows, cols = src._spec.shape
    if dst._spec.shape != (1, cols):
        raise ValueError("col-reduce destination must be a row-major [1, cols] tile")
    return CanonicalBlockMap.from_tile(src, logical_lanes=cols)


def emit_col_reduce_vmi(
    src: _TileProxy,
    dst: _TileProxy,
    *,
    kind: str,
) -> None:
    """Emit a ColReduce (tcolmax / tcolsum / tcolmin / ...) VMI candidate.

    Mirrors pto-isa `TColReduceInstr_NoPostUpdate` over one logical row:
      acc = vbr(InitVal)                        # row-wide, reduce-neutral init
      for row in 0..rows: acc = op(acc, load(row))   # runtime scf.for
      store(acc, dst)

    The accumulator stays row-wide for the whole reduction (the column axis is
    the surviving axis). This intentionally avoids `_vreduce_max`/`vmi_vcmax`,
    which collapse to a 1-lane scalar — wrong for a column-preserving ColMax.

    The init is the reduce's identity element (max->-inf, min->+inf, add->0,
    prod->1), broadcast to the logical row via `vbr` — exactly pto-isa's
    `vbr(dstVReg, InstrOp::InitVal)` (see a5/common.hpp `Padding<T>::Min/Max`).
    The reduce runs from row 0 (not 1): iteration 0 does op(InitVal, load(0))
    which absorbs row 0 through the op (e.g. max(-inf, x) = x, 0 + x = x), so a
    c0..rows header matches the element-wise VMI candidates' c0..N header and
    the downstream loop-fusion pass can merge this reduce with its same-index
    neighbors into one scf.for.

    The cross-row reduction is a runtime ``scf.for`` carrying the row-wide
    accumulator as loop state (one ``vmi.vmax``/``vmi.vadd`` per iteration),
    matching the pto-isa repeat loop. It must NOT be a Python ``range`` here:
    a trace-time ``range`` would statically unroll one merge per row (e.g. 127
    for ``rows=128``), producing a flat vmax chain with no surrounding loop.
    """
    # Reduce identity element per kind (pto-isa InstrOp::InitVal /
    # a5 Padding<T>::Min/Max): max -> -inf (0xff800000), min -> +inf
    # (0x7f800000), add -> 0.0, prod -> 1.0. Broadcast row-wide with vbr.
    reduce_identity = {
        "max": float("-inf"),
        "min": float("inf"),
        "add": 0.0,
        "prod": 1.0,
    }[kind]
    block_map = _validate_col_reduce_tiles(src, dst)
    merge_op = _REDUCE_MERGE_OP[kind]

    _prepare_tile_access(src, dst)
    full_mask = _create_mask(block_map, f32, trace=src._trace)
    # Seed the row-wide accumulator with the reduce-neutral identity (vbr InitVal,
    # matching pto-isa `TColReduceInstr_NoPostUpdate`), so the loop runs c0..rows
    # and absorbs row 0 via op(InitVal, load(0)) instead of preloading row 0.
    # The broadcast takes the element type/lanes from `f32` directly — no dummy
    # load needed (a vload would carry a Read memory effect and survive DCE,
    # duplicating the row-0 read the loop itself does).
    accumulator = _vconstant(reduce_identity, f32, lanes=block_map.cols)
    # The whole reduction is a runtime scf.for from row 0 carrying the
    # row-wide accumulator; each iteration does one element-wise merge over the
    # full logical row. Row r maps to logical block r*blocks_per_row.
    with for_(0, block_map.rows, step=1, state={"acc": accumulator}) as loop:
        row_block_base = index_mul(loop.iv, block_map.blocks_per_row)
        loaded = _vload(src, block_map.coordinate(row_block_base))
        merged = merge_op(loop.state.acc, loaded, full_mask)
        loop.yield_state(acc=merged)
    accumulator = loop.results[0]
    # dst [1, cols] is one logical row; store via linear offset to avoid the
    # src/dst shape mismatch in CanonicalBlockCoordinate validation (src is
    # [rows, cols], dst is [1, cols]).
    _vstore_linear(accumulator, dst, 0, full_mask)


def _validate_col_expand_binary_tiles(
    src: _TileProxy, col_values: _TileProxy, dst: _TileProxy
) -> CanonicalBlockMap:
    """Validate tiles for a ColExpandBinary (tcolexpandsub/...) VMI candidate.

    src is [rows, cols] row-major, col_values is [1, cols] row-major (one
    logical row of surviving reduce result), dst is [rows, cols] row-major.
    """
    if (
        src.element_type != f32
        or col_values.element_type != f32
        or dst.element_type != f32
    ):
        raise ValueError("col-expand-binary VMI candidates currently support only f32")
    if src._spec.shape != dst._spec.shape:
        raise ValueError("col-expand-binary source and destination shapes must match")
    if src._spec.b_layout != "row_major" or dst._spec.b_layout != "row_major":
        raise ValueError("col-expand-binary source and destination must be row-major")
    rows, cols = src._spec.shape
    if (
        col_values._spec.shape != (1, cols)
        or col_values._spec.b_layout != "row_major"
    ):
        raise ValueError(
            "col-expand-binary col_values must be a row-major [1, cols] tile"
        )
    return CanonicalBlockMap.from_tile(src, logical_lanes=cols)


def emit_col_expand_binary_vmi(
    src: _TileProxy,
    col_values: _TileProxy,
    dst: _TileProxy,
    *,
    binop: str,
) -> None:
    """Emit a ColExpandBinary (tcolexpandsub/add/mul/div) VMI candidate.

    Mirrors pto-isa `TColExpandBinOp`: the single logical row of col_values is
    broadcast to every row, then a binary op is applied per row block.
    """
    binop_dispatch = {
        "sub": _vsub,
        "add": _vadd,
        "mul": _vmul,
        "div": _vdiv,
    }
    if binop not in binop_dispatch:
        raise ValueError(
            f"col-expand-binary VMI candidate does not support op {binop!r}; "
            f"expected one of {sorted(binop_dispatch)}"
        )
    op_fn = binop_dispatch[binop]
    block_map = _validate_col_expand_binary_tiles(src, col_values, dst)

    _prepare_tile_access(src, col_values, dst)
    full_mask = _create_mask(block_map, f32, trace=src._trace)
    # pto-isa TColExpandBinOp broadcasts by reloading the same col_values row
    # block per row (vlds with fixed offset), NOT a 1-lane vbrc. col_values is
    # [1, cols] (one logical row), so the broadcast load is loop-invariant: hoist
    # it out of the row loop so a later mem2reg (Stage C) can forward the
    # ColMax result directly to the consumer without a per-row reload.
    broadcast = _vload_linear(col_values, 0, lanes=block_map.cols)
    with for_(0, block_map.rows, step=1) as row:
        coordinate = block_map.coordinate(index_mul(row, block_map.blocks_per_row))
        value = _vload(src, coordinate)
        result = op_fn(value, broadcast, full_mask)
        _vstore(result, dst, coordinate, full_mask)


def emit_convert_vmi(src: _TileProxy, dst: _TileProxy) -> None:
    if src.element_type != f32:
        raise ValueError("tcvt VMI candidate currently supports f32 source")
    if dst.element_type not in (f16, bf16):
        raise ValueError("tcvt VMI candidate currently supports f32 -> f16/bf16")
    if src._spec.shape != dst._spec.shape:
        raise ValueError("tcvt source and destination shapes must match")
    if src._spec.b_layout != "row_major" or dst._spec.b_layout != "row_major":
        raise ValueError("tcvt VMI candidate requires row-major tiles")
    _, cols = src._spec.shape
    block_map = CanonicalBlockMap.from_tile(src, logical_lanes=cols)

    _prepare_tile_access(src, dst)
    dst_mask = _create_mask_lanes(
        cols, cols, dst.element_type, trace=src._trace
    )
    with for_(0, block_map.logical_block_count, step=1) as logical_block:
        coordinate = block_map.coordinate(logical_block)
        converted = _vcvt(_vload(src, coordinate), dst.element_type)
        _vstore(converted, dst, coordinate, dst_mask)


__all__ = [
    "FLOAT_DTYPES",
    "Tile",
    "VMI_TILELIB_REGISTRY",
    "_abs",
    "_add",
    "_context_attr",
    "_divide_by_scalar",
    "_divide_by_scalar_high_precision",
    "_divide_scalar_by_vector",
    "_divide_scalar_by_vector_high_precision",
    "_div_high_precision",
    "_exp",
    "_max",
    "_move",
    "_mul",
    "_neg",
    "_negate_scalar",
    "_operand_kinds_are",
    "_sub",
    "_vadds",
    "_vdiv",
    "_vmaxs",
    "_vmins",
    "_vmuls",
    "canonical_vmi_template",
    "emit_elementwise_vmi",
    "emit_scalar_fill_vmi",
    "emit_col_expand_binary_vmi",
    "emit_col_reduce_vmi",
    "emit_convert_vmi",
    "emit_recip_vmi",
    "emit_row_expand_sub_vmi",
    "emit_row_reduce_vmi",
    "emit_rsqrt_vmi",
    "emit_sqrt_high_precision_vmi",
    "emit_sqrt_vmi",
    "f32",
    "row_reduce_vmi_constraint",
]
