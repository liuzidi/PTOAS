# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""PTODSL TileLib template for pto.tmin (ported from lib/TileOps/tmin_template.py)."""

from ptodsl import pto
import ptodsl.tilelib as tilelib

from ._common import NUMERIC_DTYPES


def _ub_or_vec_row_major(operand_memory_spaces, operand_b_layouts, operand_s_layouts, **_):
    return (
        all(space in {"ub", "vec"} for space in operand_memory_spaces)
        and all(layout == "row_major" for layout in operand_b_layouts)
        and all(layout == "none_box" for layout in operand_s_layouts)
    )


@tilelib.tile_template(
    op="pto.tmin",
    target="a5",
    name="template_tmin",
    dtypes=[(dtype, dtype, dtype) for dtype in NUMERIC_DTYPES],
    iteration_axis="none",
    op_engine="vector",
    op_class="elementwise",
    constraints=[
        _ub_or_vec_row_major,
        tilelib.require_same_valid_shape("src0", "src1", "dst"),
    ],
    priority=0,
    id=0,
    loop_depth=2,
    is_post_update=False,
)
def template_tmin(src0: pto.Tile, src1: pto.Tile, dst: pto.Tile):
    dtype = dst.dtype
    valid_rows, valid_cols = dst.valid_shape
    lanes = pto.elements_per_vreg(dtype)

    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, lanes):
            mask, remained = pto.make_mask(dtype, remained)
            lhs = pto.vlds(src0[row, col:])
            rhs = pto.vlds(src1[row, col:])
            min_val = pto.vmin(lhs, rhs, mask)
            pto.vsts(min_val, dst[row, col:], mask)
