# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Canonical VMI TileLib candidates for static Softmax-related coverage.

The candidates in this module are registered into the ordinary PTODSL TileLib
registry as ``ir_level="vmi"`` candidates.  ``VMI_TILELIB_REGISTRY`` is kept as
a compatibility view for direct unit tests and the legacy helper path.
"""

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
    bf16,
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
    vmi_scalar_constant,
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
from .tilelib import registry as _tilelib_registry
from .tilelib.registry import TileTemplateRegistry


ElementwiseCompute = Callable[[Sequence[_VectorValue], _MaskValue], _VectorValue]


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
    "max": vmi_vmax,
    "add": vmi_vadd,
}


def canonical_vmi_template(
    *,
    target: str = "a5",
    op: str,
    name: str | None = None,
    context_constraints: dict[str, tuple[object, ...]] | None = None,
):
    """Register one canonical VMI implementation in this provider module."""

    def decorator(fn):
        qualified_op = _qualify_op_name(op)
        descriptor = _trace_tile_template(
            target=target,
            op=qualified_op,
            name=name,
            ir_level="vmi",
            context_constraints=context_constraints,
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
) -> None:
    """Emit one flat logical-block loop for a standalone elementwise candidate."""

    if not sources:
        raise ValueError("emit_elementwise_vmi requires at least one source tile")
    if logical_lanes is None:
        logical_lanes = dst.element_type.lanes
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
    if dst.element_type != f32 or logical_lanes != f32.lanes:
        raise ValueError(
            "VMI elementwise candidates require exactly one 64-lane f32 VL per row"
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
    if cols != f32.lanes:
        raise ValueError("row-reduce source rows must contain exactly one f32 VL block")
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
    if cols != f32.lanes:
        raise ValueError("trowexpandsub rows must contain exactly one f32 VL block")
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


def _validate_col_reduce_tiles(
    src: _TileProxy, dst: _TileProxy
) -> CanonicalBlockMap:
    """Validate tiles for a ColReduce (tcolmax / tcolsum) VMI candidate.

    Mirror of `_validate_row_reduce_tiles` but the surviving axis is the column
    dimension: src is [rows, cols] row-major, dst is [1, cols] row-major, and the
    reduction runs across all rows. First slice only supports a single VL block
    wide tile (cols == VL), matching the pto-isa `TColReduceInstr_NoPostUpdate`
    one-repeat layout.
    """
    if src.element_type != f32 or dst.element_type != f32:
        raise ValueError("col-reduce VMI candidates currently support only f32")
    if src._spec.b_layout != "row_major" or dst._spec.b_layout != "row_major":
        raise ValueError("col-reduce source and destination must be row-major")
    rows, cols = src._spec.shape
    if dst._spec.shape != (1, cols):
        raise ValueError("col-reduce destination must be a row-major [1, cols] tile")
    if cols != f32.lanes:
        raise ValueError(
            "col-reduce VMI candidates currently support only cols == VL(f32) "
            f"(got cols={cols}, VL={f32.lanes})"
        )
    return CanonicalBlockMap.from_tile(src, logical_lanes=f32.lanes)


def emit_col_reduce_vmi(
    src: _TileProxy,
    dst: _TileProxy,
    *,
    kind: str,
) -> None:
    """Emit a ColReduce (tcolmax / tcolsum / tcolmin / ...) VMI candidate.

    Mirrors pto-isa `TColReduceInstr_NoPostUpdate` with a single VL block:
      acc = vbr(InitVal)                        # VL-wide, reduce-neutral init
      for row in 0..rows: acc = op(acc, load(row))   # runtime scf.for
      store(acc, dst)

    The accumulator stays VL-wide for the whole reduction (the column axis is
    the surviving axis). This intentionally avoids `vmi_vreduce_max`/`vmi_vcmax`,
    which collapse to a 1-lane scalar — wrong for a column-preserving ColMax.

    The init is the reduce's identity element (max->-inf, min->+inf, add->0,
    prod->1), broadcast VL-wide via `vbr` — exactly pto-isa's
    `vbr(dstVReg, InstrOp::InitVal)` (see a5/common.hpp `Padding<T>::Min/Max`).
    The reduce runs from row 0 (not 1): iteration 0 does op(InitVal, load(0))
    which absorbs row 0 through the op (e.g. max(-inf, x) = x, 0 + x = x), so a
    c0..rows header matches the element-wise VMI candidates' c0..N header and
    the downstream loop-fusion pass can merge this reduce with its same-index
    neighbors into one scf.for.

    The cross-row reduction is a runtime ``scf.for`` carrying the VL-wide
    accumulator as loop state (one ``vmi.vmax``/``vmi.vadd`` per iteration),
    matching the pto-isa repeat loop. It must NOT be a Python ``range`` here:
    a trace-time ``range`` would statically unroll one merge per row (e.g. 127
    for ``rows=128``), producing a flat vmax chain with no surrounding loop.
    """
    # Reduce identity element per kind (pto-isa InstrOp::InitVal /
    # a5 Padding<T>::Min/Max): max -> -inf (0xff800000), min -> +inf
    # (0x7f800000), add -> 0.0, prod -> 1.0. Broadcast VL-wide with vbr.
    reduce_identity = {
        "max": float("-inf"),
        "min": float("inf"),
        "add": 0.0,
        "prod": 1.0,
    }[kind]
    block_map = _validate_col_reduce_tiles(src, dst)
    merge_op = _REDUCE_MERGE_OP[kind]

    vmi_prepare_tile_access(src, dst)
    full_mask = vmi_create_mask(block_map, f32)
    # Seed the VL-wide accumulator with the reduce-neutral identity (vbr InitVal,
    # matching pto-isa `TColReduceInstr_NoPostUpdate`), so the loop runs c0..rows
    # and absorbs row 0 via op(InitVal, load(0)) instead of preloading row 0.
    # The broadcast takes the element type/lanes from `f32` directly — no dummy
    # load needed (a vload would carry a Read memory effect and survive DCE,
    # duplicating the row-0 read the loop itself does).
    accumulator = vmi_vbroadcast_scalar(
        vmi_scalar_constant(reduce_identity, f32), dtype=f32
    )
    # The whole reduction is a runtime scf.for from row 0 carrying the
    # VL-wide accumulator; each iteration does one element-wise merge (VL
    # stays full). Row r maps to logical block r*blocks_per_row.
    with for_(0, block_map.rows, step=1, state={"acc": accumulator}) as loop:
        row_block_base = index_mul(loop.iv, block_map.blocks_per_row)
        loaded = vmi_vload(src, block_map.coordinate(row_block_base))
        merged = merge_op(loop.state.acc, loaded, full_mask)
        loop.yield_state(acc=merged)
    accumulator = loop.results[0]
    # dst [1, cols] is a single VL block; store via linear offset to avoid the
    # src/dst shape mismatch in CanonicalBlockCoordinate validation (src is
    # [rows, cols], dst is [1, cols]).
    vmi_vstore_linear(accumulator, dst, 0, full_mask)


def _validate_col_expand_binary_tiles(
    src: _TileProxy, col_values: _TileProxy, dst: _TileProxy
) -> CanonicalBlockMap:
    """Validate tiles for a ColExpandBinary (tcolexpandsub/...) VMI candidate.

    src is [rows, cols] row-major, col_values is [1, cols] row-major (one VL
    block of surviving reduce result), dst is [rows, cols] row-major. cols must
    equal VL(f32) so the single broadcast loads exactly one VL block.
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
    if cols != f32.lanes:
        raise ValueError(
            "col-expand-binary VMI candidates currently support only cols == VL(f32)"
        )
    return CanonicalBlockMap.from_tile(src, logical_lanes=f32.lanes)


def emit_col_expand_binary_vmi(
    src: _TileProxy,
    col_values: _TileProxy,
    dst: _TileProxy,
    *,
    binop: str,
) -> None:
    """Emit a ColExpandBinary (tcolexpandsub/add/mul/div) VMI candidate.

    Mirrors pto-isa `TColExpandBinOp`: the single VL block of col_values is
    broadcast to every row, then a binary op is applied per row block.
    """
    binop_dispatch = {
        "sub": vmi_vsub,
        "add": vmi_vadd,
        "mul": vmi_vmul,
        "div": vmi_vdiv,
    }
    if binop not in binop_dispatch:
        raise ValueError(
            f"col-expand-binary VMI candidate does not support op {binop!r}; "
            f"expected one of {sorted(binop_dispatch)}"
        )
    op_fn = binop_dispatch[binop]
    block_map = _validate_col_expand_binary_tiles(src, col_values, dst)

    vmi_prepare_tile_access(src, col_values, dst)
    full_mask = vmi_create_mask(block_map, f32)
    # pto-isa TColExpandBinOp broadcasts by reloading the same col_values VL
    # block per row (vlds with fixed offset), NOT a 1-lane vbrc. col_values is
    # [1, cols] (one VL block), so the broadcast load is loop-invariant: hoist
    # it out of the row loop so a later mem2reg (Stage C) can forward the
    # ColMax result directly to the consumer without a per-row reload.
    broadcast = vmi_vload_linear(col_values, 0, lanes=f32.lanes)
    with for_(0, block_map.rows, step=1) as row:
        coordinate = block_map.coordinate(index_mul(row, block_map.blocks_per_row))
        value = vmi_vload(src, coordinate)
        result = op_fn(value, broadcast, full_mask)
        vmi_vstore(result, dst, coordinate, full_mask)


def emit_convert_vmi(src: _TileProxy, dst: _TileProxy) -> None:
    if src.element_type != f32:
        raise ValueError("tcvt VMI candidate currently supports f32 source")
    if dst.element_type not in (f16, bf16):
        raise ValueError("tcvt VMI candidate currently supports f32 -> f16/bf16")
    if src._spec.shape != dst._spec.shape:
        raise ValueError("tcvt source and destination shapes must match")
    if src._spec.b_layout != "row_major" or dst._spec.b_layout != "row_major":
        raise ValueError("tcvt VMI candidate requires row-major tiles")
    block_map = CanonicalBlockMap.from_tile(src, logical_lanes=f32.lanes)

    vmi_prepare_tile_access(src, dst)
    dst_mask = vmi_create_mask_lanes(f32.lanes, f32.lanes, dst.element_type)
    with for_(0, block_map.logical_block_count, step=1) as logical_block:
        coordinate = block_map.coordinate(logical_block)
        converted = vmi_vcvt(vmi_vload(src, coordinate), dst.element_type)
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
    context_constraints={"precisionType": ("default",)},
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


@canonical_vmi_template(
    target="a5",
    op="tdivs",
    name="vmi_tdivs",
    context_constraints={"precisionType": ("default",)},
)
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


@canonical_vmi_template(target="a5", op="tcolmax", name="vmi_tcolmax")
def vmi_tcolmax(src: Tile, dst: Tile):
    emit_col_reduce_vmi(src, dst, kind="max")


@canonical_vmi_template(target="a5", op="tcolsum", name="vmi_tcolsum")
def vmi_tcolsum(src: Tile, dst: Tile):
    emit_col_reduce_vmi(src, dst, kind="add")


@canonical_vmi_template(target="a5", op="tcolexpandsub", name="vmi_tcolexpandsub")
def vmi_tcolexpandsub(src: Tile, col_values: Tile, dst: Tile):
    emit_col_expand_binary_vmi(src, col_values, dst, binop="sub")


@canonical_vmi_template(target="a5", op="tcolexpandadd", name="vmi_tcolexpandadd")
def vmi_tcolexpandadd(src: Tile, col_values: Tile, dst: Tile):
    emit_col_expand_binary_vmi(src, col_values, dst, binop="add")


@canonical_vmi_template(target="a5", op="tcolexpandmul", name="vmi_tcolexpandmul")
def vmi_tcolexpandmul(src: Tile, col_values: Tile, dst: Tile):
    emit_col_expand_binary_vmi(src, col_values, dst, binop="mul")


@canonical_vmi_template(
    target="a5",
    op="tcolexpanddiv",
    name="vmi_tcolexpanddiv",
    # ExpandTileOp::appendOpContextAttrs unconditionally adds a `precisionType`
    # context attr to TColExpandDivOp (even when default), and validate_context_attrs
    # rejects attrs the candidate didn't declare — so the candidate must declare it.
    context_constraints={"precisionType": ("default",)},
)
def vmi_tcolexpanddiv(src: Tile, col_values: Tile, dst: Tile):
    emit_col_expand_binary_vmi(src, col_values, dst, binop="div")


@canonical_vmi_template(
    target="a5",
    op="tcvt",
    name="vmi_tcvt",
    context_constraints={"round_mode": ("RINT",)},
)
def vmi_tcvt(src: Tile, dst: Tile):
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
    "vmi_tcvt",
    "vmi_tcolmax",
    "vmi_tcolsum",
    "vmi_tcolexpandsub",
    "vmi_tcolexpandadd",
    "vmi_tcolexpandmul",
    "vmi_tcolexpanddiv",
]
