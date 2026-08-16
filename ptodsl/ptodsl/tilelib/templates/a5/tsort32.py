# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""PTODSL TileLib template for the aligned ``pto.tsort32`` path."""

from ptodsl import pto
import ptodsl.tilelib as tilelib


BLOCK_SIZE = 32
FLOAT_DST_STRIDE_COEF = 2
HALF_DST_STRIDE_COEF = 4
MAX_UB_TMP = 32 * 255
REPEAT_MAX = 255


def _static_valid_dim(tile, index):
    static_valid_shape = getattr(tile, "_template_static_valid_shape", None)
    if static_valid_shape is None:
        static_valid_shape = getattr(tile, "static_valid_shape", None)
    if static_valid_shape is not None and static_valid_shape[index] is not None:
        return static_valid_shape[index]
    return tile.valid_shape[index]


def _aligned(src_valid_shape, **_):
    return len(src_valid_shape) == 2 and src_valid_shape[1] % BLOCK_SIZE == 0


def _bytewidth(dtype):
    if dtype in {"f32", "i32", "ui32"}:
        return 4
    if dtype in {"f16", "bf16", "i16", "ui16"}:
        return 2
    return 1


def _unaligned(src_valid_shape, src_dtype, **_):
    if len(src_valid_shape) != 2:
        return False
    valid_cols = src_valid_shape[1]
    return valid_cols % BLOCK_SIZE != 0


def _pad_min(dtype):
    name = str(dtype)
    if name == "f16":
        return pto.f16(0xFC00)
    if name == "bf16":
        return pto.bf16(0xFF80)
    return pto.f32(0xFF800000)


@tilelib.tile_template(
    op="pto.tsort32",
    target="a5",
    name="template_tsort32",
    dtypes=[("f16", "ui32", "f16"), ("bf16", "ui32", "bf16"), ("f32", "ui32", "f32")],
    iteration_axis="none",
    op_engine="vector",
    op_class="other",
    constraints=[_aligned],
    id=0,
    loop_depth=2,
    is_post_update=False,
    tags=("sort", "aligned"),
)
def template_tsort32(src: pto.Tile, idx: pto.Tile, dst: pto.Tile):
    dtype = dst.dtype
    valid_rows = _static_valid_dim(dst, 0)
    valid_cols = _static_valid_dim(src, 1)

    dst_ptr = dst.as_ptr()
    src_ptr = src.as_ptr()
    idx_ptr = idx.as_ptr()

    elem_bytes = pto.bytewidth(dtype)
    dst_stride = ((dst.shape[1] * elem_bytes + BLOCK_SIZE - 1) // BLOCK_SIZE * BLOCK_SIZE) // elem_bytes
    src_stride = ((src.shape[1] * elem_bytes + BLOCK_SIZE - 1) // BLOCK_SIZE * BLOCK_SIZE) // elem_bytes
    idx_stride = ((idx.shape[1] * 4 + BLOCK_SIZE - 1) // BLOCK_SIZE * BLOCK_SIZE) // 4
    if _static_valid_dim(idx, 0) == 1:
        idx_stride = 0

    type_coef = HALF_DST_STRIDE_COEF
    if str(dtype) == "f32":
        type_coef = FLOAT_DST_STRIDE_COEF

    repeat_num_per_row = (valid_cols + BLOCK_SIZE - 1) // BLOCK_SIZE

    if repeat_num_per_row <= REPEAT_MAX:
        for row in range(0, valid_rows, 1):
            pto.vbitsort(
                pto.addptr(dst_ptr, row * dst_stride),
                pto.addptr(src_ptr, row * src_stride),
                pto.addptr(idx_ptr, row * idx_stride),
                repeat_num_per_row,
            )
    else:
        loop_num = (repeat_num_per_row + REPEAT_MAX - 1) // REPEAT_MAX
        tail_repeat_num = repeat_num_per_row % REPEAT_MAX
        for row in range(0, valid_rows, 1):
            for chunk in range(0, loop_num, 1):
                dst_addr = pto.addptr(
                    dst_ptr, row * dst_stride + chunk * REPEAT_MAX * BLOCK_SIZE * type_coef
                )
                src_addr = pto.addptr(
                    src_ptr, row * src_stride + chunk * REPEAT_MAX * BLOCK_SIZE
                )
                idx_addr = pto.addptr(
                    idx_ptr, row * idx_stride + chunk * REPEAT_MAX * BLOCK_SIZE
                )
                if chunk == loop_num - 1:
                    pto.vbitsort(dst_addr, src_addr, idx_addr, tail_repeat_num)
                else:
                    pto.vbitsort(dst_addr, src_addr, idx_addr, REPEAT_MAX)


@tilelib.tile_template(
    op="pto.tsort32",
    target="a5",
    name="template_tsort32_with_tmp",
    dtypes=[("f16", "ui32", "f16", "f16"), ("bf16", "ui32", "bf16", "bf16"), ("f32", "ui32", "f32", "f32")],
    iteration_axis="none",
    op_engine="vector",
    op_class="other",
    constraints=[_unaligned],
    id=1,
    loop_depth=1,
    is_post_update=False,
    tags=("sort", "unaligned", "tmp"),
)
def template_tsort32_with_tmp(src: pto.Tile, idx: pto.Tile, tmp: pto.Tile, dst: pto.Tile):
    dtype = dst.dtype
    valid_rows = _static_valid_dim(dst, 0)
    valid_cols = _static_valid_dim(src, 1)

    dst_ptr = dst.as_ptr()
    src_ptr = src.as_ptr()
    idx_ptr = idx.as_ptr()
    tmp_ptr = tmp.as_ptr()

    elem_bytes = pto.bytewidth(dtype)
    dst_stride = ((dst.shape[1] * elem_bytes + BLOCK_SIZE - 1) // BLOCK_SIZE * BLOCK_SIZE) // elem_bytes
    src_stride = ((src.shape[1] * elem_bytes + BLOCK_SIZE - 1) // BLOCK_SIZE * BLOCK_SIZE) // elem_bytes
    idx_stride = ((idx.shape[1] * 4 + BLOCK_SIZE - 1) // BLOCK_SIZE * BLOCK_SIZE) // 4
    if _static_valid_dim(idx, 0) == 1:
        idx_stride = 0

    type_coef = HALF_DST_STRIDE_COEF
    if str(dtype) == "f32":
        type_coef = FLOAT_DST_STRIDE_COEF

    repeat_num_per_row = (valid_cols + BLOCK_SIZE - 1) // BLOCK_SIZE
    src_tail_per_row = valid_cols % BLOCK_SIZE
    pad_value = _pad_min(dtype)
    src_shape_bytes_per_row = valid_cols * elem_bytes

    if src_shape_bytes_per_row <= MAX_UB_TMP:
        len_burst = (src_shape_bytes_per_row + BLOCK_SIZE - 1) // BLOCK_SIZE
        tmp_last_offset = repeat_num_per_row * BLOCK_SIZE - BLOCK_SIZE

        for row in range(0, valid_rows, 1):
            pto.mte_ub_ub(
                pto.addptr(src_ptr, row * src_stride),
                tmp_ptr,
                len_burst,
                nburst=(1, 0, 0),
            )
            pad_mask, _ = pto.make_mask(dtype, BLOCK_SIZE - src_tail_per_row)
            pad_vec = pto.vdup(pad_value, pad_mask)
            pto.vsts(pad_vec, tmp_ptr, tmp_last_offset, pad_mask)
            pto.vbitsort(
                pto.addptr(dst_ptr, row * dst_stride),
                tmp_ptr,
                pto.addptr(idx_ptr, row * idx_stride),
                repeat_num_per_row,
            )
        return

    loop_num = (repeat_num_per_row + REPEAT_MAX - 1) // REPEAT_MAX
    src_tail_repeat_num = repeat_num_per_row % REPEAT_MAX

    for row in range(0, valid_rows, 1):
        for chunk in range(0, loop_num, 1):
            if chunk < loop_num - 1:
                pto.vbitsort(
                    pto.addptr(
                        dst_ptr, row * dst_stride + chunk * REPEAT_MAX * BLOCK_SIZE * type_coef
                    ),
                    pto.addptr(src_ptr, row * src_stride + chunk * REPEAT_MAX * BLOCK_SIZE),
                    pto.addptr(idx_ptr, row * idx_stride + chunk * REPEAT_MAX * BLOCK_SIZE),
                    REPEAT_MAX,
                )
            else:
                if src_tail_repeat_num > 0:
                    if src_tail_repeat_num > 1:
                        pto.vbitsort(
                            pto.addptr(
                                dst_ptr,
                                row * dst_stride + chunk * REPEAT_MAX * BLOCK_SIZE * type_coef,
                            ),
                            pto.addptr(
                                src_ptr, row * src_stride + chunk * REPEAT_MAX * BLOCK_SIZE
                            ),
                            pto.addptr(
                                idx_ptr, row * idx_stride + chunk * REPEAT_MAX * BLOCK_SIZE
                            ),
                            src_tail_repeat_num - 1,
                        )

                    tail_src_offset = (
                        chunk * REPEAT_MAX + (src_tail_repeat_num - 1)
                    ) * BLOCK_SIZE
                    tail_dst_offset = (
                        (chunk * REPEAT_MAX + (src_tail_repeat_num - 1))
                        * BLOCK_SIZE
                        * type_coef
                    )
                    len_burst = (src_tail_per_row * elem_bytes + BLOCK_SIZE - 1) // BLOCK_SIZE

                    pto.mte_ub_ub(
                        pto.addptr(src_ptr, row * src_stride + tail_src_offset),
                        tmp_ptr,
                        len_burst,
                        nburst=(1, 0, 0),
                    )

                    tmp_last_offset = (
                        ((src_tail_per_row + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
                    ) - BLOCK_SIZE
                    pad_mask, _ = pto.make_mask(dtype, BLOCK_SIZE - src_tail_per_row)
                    pad_vec = pto.vdup(pad_value, pad_mask)
                    pto.vsts(pad_vec, tmp_ptr, tmp_last_offset, pad_mask)

                    pto.vbitsort(
                        pto.addptr(dst_ptr, row * dst_stride + tail_dst_offset),
                        tmp_ptr,
                        pto.addptr(idx_ptr, row * idx_stride + tail_src_offset),
                        1,
                    )
