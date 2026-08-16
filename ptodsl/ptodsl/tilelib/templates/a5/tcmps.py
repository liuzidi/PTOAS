# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""PTODSL TileLib template for pto.tcmps."""

from ptodsl import pto
import ptodsl.tilelib as tilelib


def _ub_or_vec_row_major(operand_memory_spaces, operand_b_layouts, operand_s_layouts, **_):
    return (
        all(space in {"ub", "vec"} for space in operand_memory_spaces)
        and all(layout == "row_major" for layout in operand_b_layouts)
        and all(layout == "none_box" for layout in operand_s_layouts)
    )


@tilelib.tile_template(
    op="pto.tcmps",
    target="a5",
    name="template_tcmps",
    dtypes=[
        ("f32", "f32", "ui8"),
        ("i32", "i32", "ui8"),
        ("f16", "f16", "ui8"),
        ("i16", "i16", "ui8"),
        ("i8", "i8", "ui8"),
        ("ui8", "ui8", "ui8"),
    ],
    iteration_axis="none",
    op_engine="vector",
    op_class="other",
    constraints=[_ub_or_vec_row_major],
    id=0,
    loop_depth=2,
    is_post_update=False,
    tags=("compare", "scalar", "predicate-store"),
)
def template_tcmps(src: pto.Tile, scalar, dst: pto.Tile):
    dtype = src.dtype
    valid_rows, valid_cols = src.valid_shape
    lanes = pto.elements_per_vreg(dtype)
    cmp_mode = pto.get_op_attr("cmp_mode", "eq")
    dst_ptr = dst.as_ptr()
    src_ptr = src.as_ptr()

    if str(dtype) in {"f32", "i32"}:
        total_elm = valid_rows * valid_cols
        repeat_times = (total_elm + lanes - 1) // lanes + 1
        iterations = repeat_times // 2

        for iteration in range(0, iterations, 1):
            elem_offset0 = iteration * 2 * lanes
            elem_offset1 = (iteration * 2 + 1) * lanes

            remaining0 = total_elm - elem_offset0
            remaining1 = total_elm - elem_offset1

            mask0, _ = pto.make_mask(dtype, remaining0)
            mask1, _ = pto.make_mask(dtype, remaining1)

            vec0 = pto.vlds(src_ptr, elem_offset0)
            vec1 = pto.vlds(src_ptr, elem_offset1)

            cmp0 = pto.vcmps(vec0, scalar, mask0, cmp_mode)
            cmp1 = pto.vcmps(vec1, scalar, mask1, cmp_mode)

            cmp0_b8 = pto.pbitcast(cmp0, pto.mask_b8)
            cmp1_b8 = pto.pbitcast(cmp1, pto.mask_b8)
            packed_low, _ = pto.pdintlv_b8(cmp0_b8, cmp1_b8)

            store_offset = iteration * 16
            pto.psts(packed_low, dst_ptr, store_offset, dist=pto.PredicateDist.PK)
    elif str(dtype) in {"f16", "i16"}:
        bytes_per_iter = 16
        iters_per_row = (valid_cols + lanes - 1) // lanes

        for row in range(0, valid_rows, 1):
            remained = valid_cols
            for col in range(0, valid_cols, lanes):
                mask, remained = pto.make_mask(dtype, remained)
                vec = pto.vlds(src[row, col:])
                cmp = pto.vcmps(vec, scalar, mask, cmp_mode)
                store_offset = (row * iters_per_row + col // lanes) * bytes_per_iter
                pto.psts(cmp, dst_ptr, store_offset, dist=pto.PredicateDist.PK)
    else:
        bytes_per_iter = 32
        iters_per_row = (valid_cols + lanes - 1) // lanes

        for row in range(0, valid_rows, 1):
            remained = valid_cols
            for col in range(0, valid_cols, lanes):
                mask, remained = pto.make_mask(dtype, remained)
                vec = pto.vlds(src[row, col:])
                cmp = pto.vcmps(vec, scalar, mask, cmp_mode)
                store_offset = (row * iters_per_row + col // lanes) * bytes_per_iter
                pto.psts(cmp, dst_ptr, store_offset, dist=pto.PredicateDist.NORM)
