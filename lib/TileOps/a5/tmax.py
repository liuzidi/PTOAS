# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""PTODSL TileLib template for pto.tmax."""

from ptodsl import pto
import ptodsl.tilelib as tilelib

from ._common import same_dtype_signatures
from ._elementwise import register_binary


def _vmax(lhs, rhs, mask):
    return pto.vmax(lhs, rhs, mask)


_DTYPES = same_dtype_signatures(3)


template_tmax = register_binary(
    op="pto.tmax",
    name="template_tmax",
    vector_op=_vmax,
    dtypes=_DTYPES,
)


template_tmax_1d = register_binary(
    op="pto.tmax",
    name="template_tmax_1d",
    vector_op=_vmax,
    dtypes=_DTYPES,
    traversal="1d",
)


def _ub_or_vec_row_major(operand_memory_spaces, operand_b_layouts, operand_s_layouts, **_):
    return (
        all(space in {"ub", "vec"} for space in operand_memory_spaces)
        and all(layout == "row_major" for layout in operand_b_layouts)
        and all(layout == "none_box" for layout in operand_s_layouts)
    )


def _dst_valid_prefix_operand(src0_shape=(), src1_shape=(), dst_shape=(),
                              src0_valid_shape=(), src1_valid_shape=(),
                              dst_valid_shape=(), **_):
    """Accept the PyPTO row-reduce accumulate form for tmax.

    The running maximum tile (src0/dst) is a padded [1, C] row-reduce
    accumulator while the per-chunk source (src1) only fills a strict
    prefix of the same physical row (e.g. valid 1x2 inside a 1x8 tile).
    The A5 TMAX reference implements the elementwise max over the
    destination's valid region and reads whatever the source physically
    holds beyond its valid prefix — exactly what a dst-anchored traversal
    emits — so the ordinary elementwise body is reused with a relaxed
    valid-shape contract. The prefix must be strict: equal valid shapes
    are already owned by template_tmax, and overlapping legality would
    make selection ambiguous.
    """

    def rank2(shape):
        return len(shape) == 2

    if not (rank2(src0_shape) and rank2(src1_shape) and rank2(dst_shape)):
        return False
    if not (rank2(src0_valid_shape) and rank2(src1_valid_shape)
            and rank2(dst_valid_shape)):
        return False
    return (
        src0_shape == dst_shape
        and src1_shape == dst_shape
        and src0_valid_shape == dst_valid_shape
        and src1_valid_shape[0] == dst_valid_shape[0]
        and 0 < src1_valid_shape[1] < dst_valid_shape[1]
    )


@tilelib.tile_template(
    op="pto.tmax",
    target="a5",
    name="template_tmax_dst_valid_prefix",
    dtypes=_DTYPES,
    iteration_axis="none",
    op_engine="vector",
    op_class="elementwise",
    constraints=[
        _ub_or_vec_row_major,
        _dst_valid_prefix_operand,
    ],
    id=3,
    loop_depth=2,
    is_post_update=False,
    tags=("elementwise", "binary"),
)
def template_tmax_dst_valid_prefix(src0: pto.Tile, src1: pto.Tile, dst: pto.Tile):
    from ._elementwise import emit_binary_2d

    emit_binary_2d(src0, src1, dst, _vmax)


from ._vmi_common import (  # noqa: E402
    NUMERIC_DTYPES,
    _max as _vmi_max,
    canonical_vmi_template,
    emit_elementwise_vmi,
)


@canonical_vmi_template(
    target="a5",
    op="tmax",
    name="vmi_tmax",
    dtypes=(
        ("f32", "f32", "f32"),
        ("f16", "f16", "f16"),
        ("i8", "i8", "i8"),
        ("i16", "i16", "i16"),
        ("i32", "i32", "i32"),
        ("ui8", "ui8", "ui8"),
        ("ui16", "ui16", "ui16"),
        ("ui32", "ui32", "ui32"),
    ),
)
def vmi_tmax(src0: pto.Tile, src1: pto.Tile, dst: pto.Tile):
    # A5 tmax ODS rejects bf16 (only i8/i16/i32/ui8/ui16/ui32/f16/f32); bf16 tmax
    # conservatively falls back to the ordinary PTODSL path.
    emit_elementwise_vmi(dst, (src0, src1), _vmi_max, allowed_dtypes=NUMERIC_DTYPES)
