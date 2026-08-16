# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""PTODSL TileLib template for pto.tlrelu."""

from ptodsl import pto, scalar
import ptodsl.tilelib as tilelib
from ptodsl._types import _resolve


def _ub_or_vec_row_major(operand_memory_spaces, operand_b_layouts, operand_s_layouts, **_):
    return (
        all(space in {"ub", "vec"} for space in operand_memory_spaces)
        and all(layout == "row_major" for layout in operand_b_layouts)
        and all(layout == "none_box" for layout in operand_s_layouts)
    )


@tilelib.tile_template(
    op="pto.tlrelu",
    target="a5",
    name="template_tlrelu",
    dtypes=[
        ("f16", "f16", "f16"),
        ("f16", "f32", "f16"),
        ("f32", "f32", "f32"),
    ],
    iteration_axis="none",
    op_engine="vector",
    op_class="elementwise",
    constraints=[
        _ub_or_vec_row_major,
        tilelib.require_same_valid_shape("src", "dst"),
    ],
    id=0,
    loop_depth=2,
    is_post_update=False,
    tags=("elementwise", "scalar"),
)
def template_tlrelu(src: pto.Tile, slope, dst: pto.Tile):
    dtype = dst.dtype
    valid_rows, valid_cols = dst.valid_shape
    lanes = pto.elements_per_vreg(dtype)
    slope_scalar = slope
    if str(dtype) == "f16":
        slope_scalar = scalar.coerce_scalar_to_type(
            slope,
            _resolve(pto.f16),
            context="template_tlrelu(slope)",
        )

    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, lanes):
            mask, remained = pto.make_mask(dtype, remained)
            value = pto.vlds(src[row, col:])
            result = pto.vlrelu(value, slope_scalar, mask)
            pto.vsts(result, dst[row, col:], mask)
