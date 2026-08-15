# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""PTODSL TileLib fallback template for ``pto.tgather``."""

from ptodsl import pto
import ptodsl.tilelib as tilelib


def _index_form(src_dtype, dst_dtype, indices_dtype, tmp_dtype, **_):
    return (
        src_dtype == dst_dtype
        and indices_dtype in {"i16", "i32"}
        and tmp_dtype == indices_dtype
    )


def _mask_form(src_dtype, dst_dtype, mask_pattern=None, **_):
    if mask_pattern not in {"P0101", "P1010", "P1111"}:
        return False
    if mask_pattern == "P1010":
        return src_dtype == "f32" and dst_dtype in {"i32", "ui32"}
    return src_dtype == dst_dtype


def _mask_form_valid_shape(src_valid_shape, dst_valid_shape, mask_pattern=None, **_):
    if len(src_valid_shape) != 2 or len(dst_valid_shape) != 2:
        return False
    if not _known_eq(src_valid_shape[0], dst_valid_shape[0]):
        return False
    if mask_pattern == "P1111":
        return _known_eq(src_valid_shape[1], dst_valid_shape[1])
    return _known_eq(src_valid_shape[1], dst_valid_shape[1] * 2)


def _known_eq(lhs, rhs) -> bool:
    return lhs is None or rhs is None or lhs == rhs


def _known_le(lhs, rhs) -> bool:
    return lhs is None or rhs is None or lhs <= rhs


def _index_form_valid_shape(
    src_shape,
    src_valid_shape,
    dst_valid_shape,
    indices_valid_shape,
    tmp_valid_shape,
    **_,
):
    if None in (src_shape, src_valid_shape, dst_valid_shape, indices_valid_shape, tmp_valid_shape):
        return False
    if dst_valid_shape != indices_valid_shape or indices_valid_shape != tmp_valid_shape:
        return False
    if not _known_eq(src_valid_shape[0], dst_valid_shape[0]):
        return False
    if not _known_le(src_valid_shape[1], src_shape[1]):
        return False
    return _known_le(src_valid_shape[1], dst_valid_shape[1])


@tilelib.tile_template(
    op="pto.tgather",
    target="a5",
    name="template_tgather_mask",
    dtypes=[
        ("f32", "f32"),
        ("f32", "i32"),
        ("f32", "ui32"),
        ("f16", "f16"),
        ("bf16", "bf16"),
    ],
    iteration_axis="none",
    op_engine="vector",
    op_class="other",
    constraints=[
        tilelib.check_memory_space("ub"),
        tilelib.check_layout("row_major"),
        tilelib.check_s_layout("none_box"),
        _mask_form_valid_shape,
        _mask_form,
    ],
    id=1,
    loop_depth=2,
    is_post_update=False,
    tags=("gather", "mask", "hard_boundary"),
)
def template_tgather_mask(src: pto.Tile, dst: pto.Tile):
    dtype = src.dtype
    valid_rows, dst_cols = dst.valid_shape
    lanes = pto.elements_per_vreg(dtype)
    src_ptr = src.as_ptr()
    pattern = pto.get_op_attr("mask_pattern", "P1111")
    base_indices = pto.vci(pto.i32(0), "ASC")

    for row in range(0, valid_rows, 1):
        remained = dst_cols
        for col in range(0, dst_cols, lanes):
            mask, remained = pto.make_mask(dtype, remained)
            base = pto.vadds(base_indices, col, mask)
            if pattern == "P1111":
                offsets = base
            else:
                offsets = pto.vadd(base, base, mask)
                if pattern == "P1010":
                    offsets = pto.vadds(offsets, pto.i32(1), mask)
            data = pto.vgather2(src_ptr, offsets, mask)
            if str(src.dtype) != str(dst.dtype):
                data = pto.vbitcast(data, dst.dtype)
            pto.vsts(data, dst[row, col:], mask)


@tilelib.tile_template(
    op="pto.tgather",
    target="a5",
    name="template_tgather_index",
    dtypes=[
        ("f32", "f32", "i32", "i32"),
        ("f16", "f16", "i16", "i16"),
        ("bf16", "bf16", "i16", "i16"),
    ],
    iteration_axis="none",
    op_engine="vector",
    op_class="other",
    constraints=[
        tilelib.check_memory_space("ub"),
        tilelib.check_layout("row_major"),
        tilelib.check_s_layout("none_box"),
        _index_form_valid_shape,
        _index_form,
    ],
    id=0,
    loop_depth=2,
    is_post_update=False,
    tags=("gather", "index", "hard_boundary"),
)
def template_tgather_index(
    src: pto.Tile,
    dst: pto.Tile,
    indices: pto.Tile,
    tmp: pto.Tile,
):
    dtype = dst.dtype
    valid_rows, valid_cols = dst.valid_shape
    lanes = pto.elements_per_vreg(dtype)
    src_ptr = src.as_ptr()

    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, lanes):
            mask, remained = pto.make_mask(dtype, remained)
            offsets = pto.vlds(indices[row, col:])
            _ = pto.vlds(tmp[row, col:])
            data = pto.vgather2(src_ptr, offsets, mask)
            pto.vsts(data, dst[row, col:], mask)
