# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""PTODSL TileLib templates for pto.tgather."""

from ptodsl import pto
import ptodsl.tilelib as tilelib
from ._common import NUMERIC_DTYPES
from ._common import element_store_dist
from ._row_arg import _scalar_literal


def _axis_is_row(axis="row", **_):
    return axis == "row"


def _axis_is_col(axis="col", **_):
    return axis == "col"


def _no_mask_pattern(mask_pattern=True):
    return mask_pattern is True


def _dst_shape_le_indices_shape(dst_valid_shape=(), indices_valid_shape=(), **_):
    return (
        len(dst_valid_shape) == 2
        and len(indices_valid_shape) == 2
        and dst_valid_shape[0] <= indices_valid_shape[0]
        and dst_valid_shape[1] <= indices_valid_shape[1]
    )


def gather_dtype_signatures(dtypes=NUMERIC_DTYPES):
    res = []
    for dtype in dtypes:
        for dtype_indices in ('i16', 'ui16', "i32", "ui32"):
            res.append((dtype, dtype, dtype_indices))
    return res


def gather_tmp_dtype_signatures(dtypes=NUMERIC_DTYPES):
    # PyPTO's A5 flat-index gather emits a 4-operand index form whose tmp
    # workspace is not read by the A5 vgather2 sequence. The op verifier
    # couples the workspace to the indices dtype (same element type), and
    # the index width follows the data width exactly like the 3-operand
    # template (4-byte data -> i32, 2-byte data -> i16).
    dtype_index = {
        "f32": "i32", "i32": "i32", "ui32": "i32",
        "f16": "i16", "bf16": "i16", "i16": "i16", "ui16": "i16",
        "i8": "i16", "ui8": "i16",
    }
    res = []
    for dtype in dtypes:
        dtype_indices = dtype_index[dtype]
        res.append((dtype, dtype, dtype_indices, dtype_indices))
    return res


@tilelib.tile_template(
    op="pto.tgather",
    target="a5",
    name="template_tgather",
    dtypes=gather_dtype_signatures(),
    iteration_axis="none",
    op_engine="vector",
    op_class="other",
    layouts=("row_major",),
    loop_depth=2,
    is_post_update=False,
    constraints=(_no_mask_pattern, _dst_shape_le_indices_shape),
    id=0,
)
def template_tgather(
    src: pto.Tile,
    dst: pto.Tile,
    indices: pto.Tile):
    # The vgather2 sequence below is deliberately duplicated in
    # template_tgather_tmp: the template tracer evaluates loop bounds
    # through the traced function's scope, so a shared helper breaks
    # range() tracing. Keep both bodies in sync on any dtype/lane change.
    dtype = dst.element_type
    dtype_indices = indices.element_type
    elem_bytes = pto.bytewidth(dtype)
    elem_bytes_s1 = pto.bytewidth(dtype_indices)
    lanes = pto.elements_per_vreg(dtype_indices)
    valid_rows, valid_cols = dst.valid_shape
    src_ptr = src.as_ptr()
    if pto.const_expr(elem_bytes == 2 and elem_bytes_s1 == 4):
        indices_type = pto.ui32
        mask_elem = indices_type
    elif pto.const_expr(elem_bytes == 1):
        indices_type = pto.ui16
        lanes = pto.elements_per_vreg(dtype) >> 1
        result_elem = pto.ui16 if pto.const_expr(str(dtype) in ("ui8",)) else pto.i16
        result_ty = pto.vreg_type(lanes, result_elem)
        mask_elem = result_elem
    elif pto.const_expr(elem_bytes == 4):
        indices_type = pto.ui32
        mask_elem = indices_type
    else:
        indices_type = pto.ui16
        mask_elem = indices_type
    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, lanes):
            indices_reg = pto.vlds(indices[row, col:])
            mask, remained = pto.make_mask(mask_elem, remained)
            if pto.const_expr(elem_bytes == 2 and elem_bytes_s1 == 4):
                dst_reg = pto.vgather2_bc(src_ptr, pto.vbitcast(indices_reg, indices_type), mask)
            elif pto.const_expr(elem_bytes == 1):
                dst_reg = pto.vgather2(src_ptr, pto.vbitcast(indices_reg, indices_type), mask, result_vreg_type=result_ty)
            else:
                dst_reg = pto.vgather2(src_ptr, pto.vbitcast(indices_reg, indices_type), mask)
            if pto.const_expr(elem_bytes == 1):
                pto.vsts(dst_reg, dst[row, col:], mask, dist=pto.VStoreDist.PK_B16)
            else:
                pto.vsts(dst_reg, dst[row, col:], mask)


@tilelib.tile_template(
    op="pto.tgather",
    target="a5",
    name="template_tgather_tmp",
    dtypes=gather_tmp_dtype_signatures(),
    iteration_axis="none",
    op_engine="vector",
    op_class="other",
    layouts=("row_major",),
    loop_depth=2,
    is_post_update=False,
    constraints=(_no_mask_pattern, _dst_shape_le_indices_shape),
    id=4,
)
def template_tgather_tmp(
    src: pto.Tile,
    dst: pto.Tile,
    indices: pto.Tile,
    tmp: pto.Tile):
    # PyPTO's A5 flat-index gather path emits ins(src, indices, tmp) where
    # the tmp workspace only feeds the pto-isa C++ kernel; the vgather2
    # sequence below never reads it.
    _ = tmp
    dtype = dst.element_type
    dtype_indices = indices.element_type
    elem_bytes = pto.bytewidth(dtype)
    elem_bytes_s1 = pto.bytewidth(dtype_indices)
    lanes = pto.elements_per_vreg(dtype_indices)
    valid_rows, valid_cols = dst.valid_shape
    src_ptr = src.as_ptr()
    if pto.const_expr(elem_bytes == 2 and elem_bytes_s1 == 4):
        indices_type = pto.ui32
        mask_elem = indices_type
    elif pto.const_expr(elem_bytes == 1):
        indices_type = pto.ui16
        lanes = pto.elements_per_vreg(dtype) >> 1
        result_elem = pto.ui16 if pto.const_expr(str(dtype) in ("ui8",)) else pto.i16
        result_ty = pto.vreg_type(lanes, result_elem)
        mask_elem = result_elem
    elif pto.const_expr(elem_bytes == 4):
        indices_type = pto.ui32
        mask_elem = indices_type
    else:
        indices_type = pto.ui16
        mask_elem = indices_type
    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, lanes):
            indices_reg = pto.vlds(indices[row, col:])
            mask, remained = pto.make_mask(mask_elem, remained)
            if pto.const_expr(elem_bytes == 2 and elem_bytes_s1 == 4):
                dst_reg = pto.vgather2_bc(src_ptr, pto.vbitcast(indices_reg, indices_type), mask)
            elif pto.const_expr(elem_bytes == 1):
                dst_reg = pto.vgather2(src_ptr, pto.vbitcast(indices_reg, indices_type), mask, result_vreg_type=result_ty)
            else:
                dst_reg = pto.vgather2(src_ptr, pto.vbitcast(indices_reg, indices_type), mask)
            if pto.const_expr(elem_bytes == 1):
                pto.vsts(dst_reg, dst[row, col:], mask, dist=pto.VStoreDist.PK_B16)
            else:
                pto.vsts(dst_reg, dst[row, col:], mask)


_GATHER_MASK_DTYPES = [(dtype, dtype) for dtype in NUMERIC_DTYPES]

# Cross-dtype mask-pattern signatures: PyPTO decodes packed sort keys by
# gathering with output_dtype != src.dtype (e.g. FP32 sort32 key bits read
# back as INT32 indices). The hardware performs a bit reinterpretation, so
# only equal-storage-width combinations are legal.
_GATHER_MASK_CROSS_DTYPES = [
    (src_dtype, dst_dtype)
    for src_dtype in ("f16", "i16", "ui16")
    for dst_dtype in ("i16", "ui16", "f16")
] + [
    (src_dtype, dst_dtype)
    for src_dtype in ("f32", "i32", "ui32")
    for dst_dtype in ("i32", "ui32", "f32")
]


_MASK_PATTERN_TO_INTERLEAVE = {
    "P1111": (),
    "P0101": (True,),
    "P1010": (False,),
    "P0001": (True, True),
    "P0010": (True, False),
    "P0100": (False, True),
    "P1000": (False, False),
}

_MASK_PATTERN_TO_STRIDE = {
    "P1111": (0, 1),
    "P0101": (0, 2),
    "P1010": (1, 2),
    "P0001": (0, 4),
    "P0010": (1, 4),
    "P0100": (2, 4),
    "P1000": (3, 4),
}


@tilelib.tile_template(
    op="pto.tgather",
    target="a5",
    name="template_tgather_mask_row",
    dtypes=_GATHER_MASK_DTYPES + _GATHER_MASK_CROSS_DTYPES,
    iteration_axis="none",
    op_engine="vector",
    op_class="other",
    layouts=("row_major",),
    loop_depth=2,
    is_post_update=False,
    tags=("gather", "mask"),
    constraints=(_axis_is_row,),
    id=1,
)
def template_tgather_mask_row(src: pto.Tile, dst: pto.Tile):
    mask_pattern = pto.get_op_attr("mask_pattern", "P1111")
    interleave_args = _MASK_PATTERN_TO_INTERLEAVE[mask_pattern]
    dtype = dst.dtype
    src_dtype = src.dtype
    valid_rows, valid_cols = dst.valid_shape
    lanes = pto.elements_per_vreg(dtype)
    # Cross-dtype mask gathers reinterpret the gathered lanes as the
    # destination dtype (e.g. FP32 sort keys read back as INT32 indices);
    # the combinations are storage-width-equal so the vreg bit pattern is
    # preserved verbatim.
    reinterpret = not pto.const_expr(str(src_dtype) == str(dtype))
    times = 1 << len(interleave_args)

    def _emit(dst_reg):
        if reinterpret:
            dst_reg = pto.vbitcast(dst_reg, dtype)
        mask, remained[0] = pto.make_mask(dtype, remained[0])
        pto.vsts(dst_reg, dst[row, col:], mask)

    for row in range(0, valid_rows, 1):
        remained = [valid_cols]
        for col in range(0, valid_cols, lanes):
            if not interleave_args:
                src_reg = pto.vlds(src[row, col:])
                _emit(src_reg)
            elif len(interleave_args) == 1:
                reg0 = pto.vlds(src[row, col * times:])
                reg1 = pto.vlds(src[row, col * times + lanes:])
                res0, res1 = pto.vdintlv(reg0, reg1)
                if interleave_args[0]:
                    _emit(res0)
                else:
                    _emit(res1)
            else:
                reg0 = pto.vlds(src[row, col * times:])
                reg1 = pto.vlds(src[row, col * times + lanes:])
                reg2 = pto.vlds(src[row, col * times + lanes * 2:])
                reg3 = pto.vlds(src[row, col * times + lanes * 3:])
                r0_a, r0_b = pto.vdintlv(reg0, reg1)
                r1_a, r1_b = pto.vdintlv(reg2, reg3)
                if interleave_args[1]:
                    tmp0 = r0_a
                    tmp1 = r1_a
                else:
                    tmp0 = r0_b
                    tmp1 = r1_b
                final_a, final_b = pto.vdintlv(tmp0, tmp1)
                if interleave_args[0]:
                    _emit(final_a)
                else:
                    _emit(final_b)


@tilelib.tile_template(
    op="pto.tgather",
    target="a5",
    name="template_tgather_mask_col",
    dtypes=_GATHER_MASK_DTYPES,
    iteration_axis="none",
    op_engine="vector",
    op_class="other",
    layouts=("row_major",),
    loop_depth=2,
    is_post_update=False,
    tags=("gather", "mask"),
    constraints=(_axis_is_col,),
    id=2,
)
def template_tgather_mask_col(src: pto.Tile, dst: pto.Tile):
    mask_pattern = pto.get_op_attr("mask_pattern", "P1111")
    start, stride = _MASK_PATTERN_TO_STRIDE[mask_pattern]
    dtype = dst.dtype
    valid_rows, valid_cols = dst.valid_shape
    lanes = pto.elements_per_vreg(dtype)
    for row in range(0, valid_rows, 1):
        remained = valid_cols
        row_src = row * stride + start
        for col in range(0, valid_cols, lanes):
            src_reg = pto.vlds(src[row_src, col:])
            mask, remained = pto.make_mask(dtype, remained)
            pto.vsts(src_reg, dst[row, col:], mask)

