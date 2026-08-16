# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""PTODSL TileLib template for pto.tsels."""

from ptodsl import pto
import ptodsl.tilelib as tilelib


def _tsels_shapes(mask_valid_shape=(), src_valid_shape=(), tmp_valid_shape=(), dst_valid_shape=(), **_):
    _ = mask_valid_shape, tmp_valid_shape
    return len(src_valid_shape) == 2 and tuple(src_valid_shape) == tuple(dst_valid_shape)


@tilelib.tile_template(
    op="pto.tsels",
    target="a5",
    name="template_tsels",
    dtypes=[
        ("i8", "i8", "i8", "i8", "i8"),
        ("i16", "i8", "i8", "i8", "i8"),
        ("i32", "i8", "i8", "i8", "i8"),
        ("i8", "i16", "i16", "i16", "i16"),
        ("i16", "i16", "i16", "i16", "i16"),
        ("i32", "i16", "i16", "i16", "i16"),
        ("i8", "i32", "i32", "i32", "i32"),
        ("i16", "i32", "i32", "i32", "i32"),
        ("i32", "i32", "i32", "i32", "i32"),
        ("i8", "f32", "f32", "f32", "f32"),
        ("i16", "f32", "f32", "f32", "f32"),
        ("i32", "f32", "f32", "f32", "f32"),
        ("i8", "f16", "f16", "f16", "f16"),
        ("i16", "f16", "f16", "f16", "f16"),
        ("i32", "f16", "f16", "f16", "f16"),
    ],
    iteration_axis="none",
    op_engine="vector",
    op_class="other",
    constraints=[
        tilelib.check_memory_space("ub"),
        tilelib.check_layout("row_major"),
        tilelib.check_s_layout("none_box"),
        _tsels_shapes,
    ],
    id=0,
    loop_depth=2,
    is_post_update=False,
    tags=("select", "scalar", "predicate-load"),
)
def template_tsels(mask: pto.Tile, src: pto.Tile, tmp: pto.Tile, scalar, dst: pto.Tile):
    _ = tmp
    dtype = dst.dtype
    valid_rows, valid_cols = dst.valid_shape
    lanes = pto.elements_per_vreg(dtype)
    mask_ptr = pto.castptr(mask.as_ptr(), pto.ptr(pto.ui8, "ub"))
    mask_stride = mask.shape[1] * pto.bytewidth(mask.dtype)
    full_mask, _ = pto.make_mask(dtype, lanes)
    scalar_vec = pto.vdup(scalar, full_mask)

    if lanes == 64:
        full_mask_b16 = pto.pset_b16(pto.PAT.ALL)
        pair_width = lanes * 2
        paired_cols = (valid_cols // pair_width) * pair_width
        for row in range(0, valid_rows, 1):
            for col in range(0, paired_cols, pair_width):
                mask_offset = row * mask_stride + col // 8
                select_mask_raw = pto.plds(mask_ptr, mask_offset, dist=pto.PredicateDist.US)
                select_mask = pto.pbitcast(select_mask_raw, pto.mask_b16)
                pred0, _ = pto.make_mask(dtype, pair_width)
                pred1, _ = pto.make_mask(dtype, lanes)
                select_mask0, select_mask1 = pto.pintlv_b16(select_mask, full_mask_b16)
                select_mask0 = pto.pbitcast(select_mask0, pto.mask_b32)
                select_mask1 = pto.pbitcast(select_mask1, pto.mask_b32)
                src0 = pto.vlds(src[row, col:])
                src1 = pto.vlds(src[row, col + lanes:])
                selected0 = pto.vsel(src0, scalar_vec, select_mask0)
                selected1 = pto.vsel(src1, scalar_vec, select_mask1)
                pto.vsts(selected0, dst[row, col:], pred0)
                pto.vsts(selected1, dst[row, col + lanes:], pred1)
            tail_cols = valid_cols - paired_cols
            if tail_cols > 0:
                col = paired_cols
                mask_offset = row * mask_stride + col // 8
                select_mask_raw = pto.plds(mask_ptr, mask_offset, dist=pto.PredicateDist.US)
                select_mask = pto.pbitcast(select_mask_raw, pto.mask_b16)
                select_mask0 = pto.punpack(select_mask, pto.PredicatePart.LOWER)
                select_mask0 = pto.pbitcast(select_mask0, pto.mask_b32)
                pred0, _ = pto.make_mask(dtype, tail_cols)
                src0 = pto.vlds(src[row, col:])
                selected0 = pto.vsel(src0, scalar_vec, select_mask0)
                pto.vsts(selected0, dst[row, col:], pred0)
    elif lanes == 128:
        for row in range(0, valid_rows, 1):
            remained = valid_cols
            for col in range(0, valid_cols, lanes):
                pred, remained = pto.make_mask(dtype, remained)
                mask_offset = row * mask_stride + col // 8
                select_mask = pto.plds(mask_ptr, mask_offset, dist=pto.PredicateDist.US)
                select_mask = pto.pbitcast(select_mask, pto.mask_b16)
                lhs = pto.vlds(src[row, col:])
                result = pto.vsel(lhs, scalar_vec, select_mask)
                pto.vsts(result, dst[row, col:], pred)
    else:
        for row in range(0, valid_rows, 1):
            remained = valid_cols
            for col in range(0, valid_cols, lanes):
                pred, remained = pto.make_mask(dtype, remained)
                mask_offset = row * mask_stride + col // 8
                select_mask = pto.plds(mask_ptr, mask_offset, dist=pto.PredicateDist.NORM)
                lhs = pto.vlds(src[row, col:])
                result = pto.vsel(lhs, scalar_vec, select_mask)
                pto.vsts(result, dst[row, col:], pred)
