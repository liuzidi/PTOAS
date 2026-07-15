# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Initial canonical VMI TileLib candidates for static Softmax-related coverage."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from ._surface_types import Tile
from ._tile_template_tracing import (
    CanonicalBlockMap,
    _MaskValue,
    _TileProxy,
    _Value,
    _VectorValue,
    f16,
    f32,
    for_,
    index_add,
    index_mul,
    tile_template as _trace_tile_template,
    vmi_create_mask,
    vmi_create_mask_lanes,
    vmi_prepare_tile_access,
    vmi_vadd,
    vmi_vadds,
    vmi_vbroadcast,
    vmi_vbroadcast_scalar,
    vmi_vcvt,
    vmi_vdiv,
    vmi_vexp,
    vmi_vload,
    vmi_vload_linear,
    vmi_vmax,
    vmi_vmaxs,
    vmi_vmins,
    vmi_vmuls,
    vmi_vmul,
    vmi_vreduce_add,
    vmi_vreduce_max,
    vmi_vsub,
    vmi_vstore,
    vmi_vstore_linear,
)
from .tilelib.registry import TileTemplateRegistry


ElementwiseCompute = Callable[[Sequence[_VectorValue], _MaskValue], _VectorValue]


VMI_TILELIB_REGISTRY = TileTemplateRegistry()


def canonical_vmi_template(
    *,
    target: str = "a5",
    op: str,
    name: str | None = None,
):
    """Register one canonical VMI implementation in this provider module."""

    def decorator(fn):
        normalized_op = op[4:] if op.startswith("pto.") else op
        descriptor = _trace_tile_template(
            target=target,
            op=normalized_op,
            name=name,
            ir_level="vmi",
        )(fn)
        VMI_TILELIB_REGISTRY.register(descriptor)
        return descriptor

    return decorator


def emit_elementwise_vmi(
    dst: _TileProxy,
    sources: Sequence[_TileProxy],
    compute: ElementwiseCompute,
    *,
    logical_lanes: int | None = None,
) -> None:
    """Emit one flat logical-block loop for a standalone elementwise candidate."""

    if not sources:
        raise ValueError("emit_elementwise_vmi requires at least one source tile")
    if logical_lanes is None:
        logical_lanes = min(dst.element_type.lanes, dst._spec.shape[1])
    _validate_elementwise_tiles(dst, sources, logical_lanes=logical_lanes)
    block_map = CanonicalBlockMap.from_tile(dst, logical_lanes=logical_lanes)

    vmi_prepare_tile_access(*sources, dst)
    mask = vmi_create_mask(block_map, dst.element_type)
    with for_(0, block_map.logical_block_count, step=1) as logical_block:
        coordinate = block_map.coordinate(logical_block)
        values = tuple(vmi_vload(source, coordinate) for source in sources)
        result = compute(values, mask)
        vmi_vstore(result, dst, coordinate, mask)


def _validate_elementwise_tiles(
    dst: _TileProxy,
    sources: Sequence[_TileProxy],
    *,
    logical_lanes: int,
) -> None:
    if not isinstance(dst, _TileProxy):
        raise TypeError("elementwise VMI candidate destination must be a traced Tile")
    if dst.element_type != f32 or not 0 < logical_lanes <= f32.lanes:
        raise ValueError(
            "VMI elementwise candidates require an f32 logical block no wider than 64 lanes"
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
    return vmi_vadd(values[0], values[1], mask)


def _exp(values: Sequence[_VectorValue], mask: _MaskValue) -> _VectorValue:
    if len(values) != 1:
        raise ValueError("texp VMI candidate expects one source vector")
    return vmi_vexp(values[0], mask)


def _sub(values: Sequence[_VectorValue], mask: _MaskValue) -> _VectorValue:
    if len(values) != 2:
        raise ValueError("tsub VMI candidate expects two source vectors")
    return vmi_vsub(values[0], values[1], mask)


def _mul(values: Sequence[_VectorValue], mask: _MaskValue) -> _VectorValue:
    if len(values) != 2:
        raise ValueError("tmul VMI candidate expects two source vectors")
    return vmi_vmul(values[0], values[1], mask)


def _max(values: Sequence[_VectorValue], mask: _MaskValue) -> _VectorValue:
    if len(values) != 2:
        raise ValueError("tmax VMI candidate expects two source vectors")
    return vmi_vmax(values[0], values[1], mask)


def _move(values: Sequence[_VectorValue], mask: _MaskValue) -> _VectorValue:
    if len(values) != 1:
        raise ValueError("tmov VMI candidate expects one source vector")
    return values[0]


def _divide_by_scalar(
    value: _VectorValue, scalar: _Value, mask: _MaskValue
) -> _VectorValue:
    scalar_vector = vmi_vbroadcast_scalar(scalar, like=value)
    return vmi_vdiv(value, scalar_vector, mask)


def _validate_row_reduce_tiles(
    src: _TileProxy, workspace: _TileProxy, dst: _TileProxy
) -> CanonicalBlockMap:
    if (
        src.element_type != f32
        or workspace.element_type != f32
        or dst.element_type != f32
    ):
        raise ValueError("row-reduce VMI candidates currently support only f32")
    if (
        src._spec.b_layout != "row_major"
        or workspace._spec.b_layout != "row_major"
    ):
        raise ValueError("row-reduce source and workspace must be row-major")
    if workspace._spec.shape != src._spec.shape:
        raise ValueError("row-reduce workspace shape must match the source")
    rows, cols = src._spec.shape
    if dst._spec.shape != (rows, 1) or dst._spec.b_layout != "col_major":
        raise ValueError("row-reduce destination must be a col-major [rows, 1] tile")
    if cols % f32.lanes != 0:
        raise ValueError("row-reduce source columns must contain full f32 VL blocks")
    return CanonicalBlockMap.from_tile(src, logical_lanes=f32.lanes)


def emit_row_reduce_vmi(
    src: _TileProxy,
    workspace: _TileProxy,
    dst: _TileProxy,
    *,
    kind: str,
) -> None:
    block_map = _validate_row_reduce_tiles(src, workspace, dst)
    reduce_op = vmi_vreduce_max if kind == "max" else vmi_vreduce_add
    merge_op = vmi_vmax if kind == "max" else vmi_vadd

    vmi_prepare_tile_access(src, dst)
    full_mask = vmi_create_mask(block_map, f32)
    scalar_mask = vmi_create_mask_lanes(1, 1, f32)
    with for_(0, block_map.rows, step=1) as row:
        row_block_base = index_mul(row, block_map.blocks_per_row)
        first_coordinate = block_map.coordinate(row_block_base)
        accumulator = reduce_op(vmi_vload(src, first_coordinate), full_mask)
        for block_in_row in range(1, block_map.blocks_per_row):
            coordinate = block_map.coordinate(
                index_add(row_block_base, block_in_row)
            )
            reduced = reduce_op(vmi_vload(src, coordinate), full_mask)
            accumulator = merge_op(accumulator, reduced, scalar_mask)
        vmi_vstore_linear(accumulator, dst, row, scalar_mask)


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
    if cols % f32.lanes != 0:
        raise ValueError("trowexpandsub columns must contain full f32 VL blocks")
    block_map = CanonicalBlockMap.from_tile(src, logical_lanes=f32.lanes)

    vmi_prepare_tile_access(src, row_values, dst)
    full_mask = vmi_create_mask(block_map, f32)
    with for_(0, rows, step=1) as row:
        row_scalar = vmi_vload_linear(row_values, row, lanes=1)
        broadcast = vmi_vbroadcast(row_scalar, lanes=f32.lanes)
        row_block_base = index_mul(row, block_map.blocks_per_row)
        for block_in_row in range(block_map.blocks_per_row):
            coordinate = block_map.coordinate(
                index_add(row_block_base, block_in_row)
            )
            value = vmi_vload(src, coordinate)
            result = vmi_vsub(value, broadcast, full_mask)
            vmi_vstore(result, dst, coordinate, full_mask)


def emit_convert_vmi(src: _TileProxy, dst: _TileProxy) -> None:
    if src.element_type != f32 or dst.element_type != f16:
        raise ValueError("tcvt VMI candidate currently supports f32 to f16")
    if src._spec.shape != dst._spec.shape:
        raise ValueError("tcvt source and destination shapes must match")
    if src._spec.b_layout != "row_major" or dst._spec.b_layout != "row_major":
        raise ValueError("tcvt VMI candidate requires row-major tiles")
    block_map = CanonicalBlockMap.from_tile(src, logical_lanes=f32.lanes)

    vmi_prepare_tile_access(src, dst)
    dst_mask = vmi_create_mask_lanes(f32.lanes, f32.lanes, f16)
    with for_(0, block_map.logical_block_count, step=1) as logical_block:
        coordinate = block_map.coordinate(logical_block)
        converted = vmi_vcvt(vmi_vload(src, coordinate), f16)
        vmi_vstore(converted, dst, coordinate, dst_mask)


@canonical_vmi_template(
    target="a5",
    op="tadd",
    name="vmi_tadd_block64",
)
def vmi_tadd_block64(src0: Tile, src1: Tile, dst: Tile):
    emit_elementwise_vmi(dst, (src0, src1), _add)


@canonical_vmi_template(
    target="a5",
    op="texp",
    name="vmi_texp_block64",
)
def vmi_texp_block64(src: Tile, dst: Tile):
    emit_elementwise_vmi(dst, (src,), _exp)


@canonical_vmi_template(target="a5", op="tsub", name="vmi_tsub")
def vmi_tsub(src0: Tile, src1: Tile, dst: Tile):
    emit_elementwise_vmi(dst, (src0, src1), _sub)


@canonical_vmi_template(target="a5", op="tmul", name="vmi_tmul")
def vmi_tmul(src0: Tile, src1: Tile, dst: Tile):
    emit_elementwise_vmi(dst, (src0, src1), _mul)


@canonical_vmi_template(target="a5", op="tmax", name="vmi_tmax")
def vmi_tmax(src0: Tile, src1: Tile, dst: Tile):
    emit_elementwise_vmi(dst, (src0, src1), _max)


@canonical_vmi_template(target="a5", op="tmov", name="vmi_tmov")
def vmi_tmov(src: Tile, dst: Tile):
    emit_elementwise_vmi(dst, (src,), _move)


@canonical_vmi_template(target="a5", op="tmuls", name="vmi_tmuls")
def vmi_tmuls(src: Tile, scale: f32, dst: Tile):
    emit_elementwise_vmi(
        dst,
        (src,),
        lambda values, mask: vmi_vmuls(values[0], scale, mask),
    )


@canonical_vmi_template(target="a5", op="tadds", name="vmi_tadds")
def vmi_tadds(src: Tile, scalar: f32, dst: Tile):
    emit_elementwise_vmi(
        dst,
        (src,),
        lambda values, mask: vmi_vadds(values[0], scalar, mask),
    )


@canonical_vmi_template(target="a5", op="tmaxs", name="vmi_tmaxs")
def vmi_tmaxs(src: Tile, scalar: f32, dst: Tile):
    emit_elementwise_vmi(
        dst,
        (src,),
        lambda values, mask: vmi_vmaxs(values[0], scalar, mask),
    )


@canonical_vmi_template(target="a5", op="tmins", name="vmi_tmins")
def vmi_tmins(src: Tile, scalar: f32, dst: Tile):
    emit_elementwise_vmi(
        dst,
        (src,),
        lambda values, mask: vmi_vmins(values[0], scalar, mask),
    )


@canonical_vmi_template(target="a5", op="tdivs", name="vmi_tdivs")
def vmi_tdivs(src: Tile, scalar: f32, dst: Tile):
    emit_elementwise_vmi(
        dst,
        (src,),
        lambda values, mask: _divide_by_scalar(values[0], scalar, mask),
    )


@canonical_vmi_template(target="a5", op="trowmax", name="vmi_trowmax")
def vmi_trowmax(src: Tile, workspace: Tile, dst: Tile):
    emit_row_reduce_vmi(src, workspace, dst, kind="max")


@canonical_vmi_template(target="a5", op="trowsum", name="vmi_trowsum")
def vmi_trowsum(src: Tile, workspace: Tile, dst: Tile):
    emit_row_reduce_vmi(src, workspace, dst, kind="sum")


@canonical_vmi_template(
    target="a5",
    op="trowexpandsub",
    name="vmi_trowexpandsub",
)
def vmi_trowexpandsub(src: Tile, row_values: Tile, dst: Tile):
    emit_row_expand_sub_vmi(src, row_values, dst)


@canonical_vmi_template(target="a5", op="tcvt", name="vmi_tcvt_f32_f16")
def vmi_tcvt_f32_f16(src: Tile, dst: Tile):
    emit_convert_vmi(src, dst)


__all__ = [
    "VMI_TILELIB_REGISTRY",
    "canonical_vmi_template",
    "emit_elementwise_vmi",
    "vmi_tadd_block64",
    "vmi_texp_block64",
    "vmi_tsub",
    "vmi_tmul",
    "vmi_tmax",
    "vmi_tmov",
    "vmi_tmuls",
    "vmi_tadds",
    "vmi_tmaxs",
    "vmi_tmins",
    "vmi_tdivs",
    "vmi_trowmax",
    "vmi_trowsum",
    "vmi_trowexpandsub",
    "vmi_tcvt_f32_f16",
]
