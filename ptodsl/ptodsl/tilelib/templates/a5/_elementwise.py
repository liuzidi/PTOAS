# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Shared PTODSL implementations for straightforward A5 elementwise TileOps."""

from ptodsl import pto
import ptodsl.tilelib as tilelib


def _ub_or_vec_row_major(operand_memory_spaces, operand_b_layouts, operand_s_layouts, **_):
    return (
        all(space in {"ub", "vec"} for space in operand_memory_spaces)
        and all(layout == "row_major" for layout in operand_b_layouts)
        and all(layout == "none_box" for layout in operand_s_layouts)
    )


def _known_eq(lhs, rhs):
    return lhs is None or rhs is None or lhs == rhs


def _same_physical_shape(*operand_names):
    def predicate(**context):
        shapes = [context.get(f"{name}_shape") for name in operand_names]
        return (
            bool(shapes)
            and None not in shapes
            and all(shape == shapes[0] for shape in shapes[1:])
        )

    return predicate


def _same_or_dynamic_valid_shape(*operand_names):
    def predicate(**context):
        shapes = [context.get(f"{name}_valid_shape") for name in operand_names]
        if not shapes or None in shapes:
            return False
        rank = len(shapes[0])
        if any(len(shape) != rank for shape in shapes):
            return False
        return all(
            _known_eq(lhs, rhs)
            for shape in shapes[1:]
            for lhs, rhs in zip(shapes[0], shape)
        )

    return predicate


def _common_constraints(*operand_names):
    return [
        _ub_or_vec_row_major,
        _same_physical_shape(*operand_names),
        _same_or_dynamic_valid_shape(*operand_names),
    ]


def register_unary(*, op, name, vector_op, dtypes, constraints=()):
    """Register a unary tile traversal using a public PTODSL vector operation."""

    candidate_constraints = _common_constraints("src", "dst") + list(constraints)

    @tilelib.tile_template(
        op=op,
        target="a5",
        name=name,
        dtypes=dtypes,
        iteration_axis="none",
        op_engine="vector",
        op_class="elementwise",
        constraints=candidate_constraints,
        id=0,
        loop_depth=2,
        is_post_update=False,
        tags=("elementwise", "unary"),
    )
    def template(src: pto.Tile, dst: pto.Tile):
        dtype = dst.dtype
        valid_rows, valid_cols = dst.valid_shape
        lanes = pto.elements_per_vreg(dtype)

        for row in range(0, valid_rows, 1):
            remained = valid_cols
            for col in range(0, valid_cols, lanes):
                mask, remained = pto.make_mask(dtype, remained)
                value = pto.vlds(src[row, col:])
                result = vector_op(value, mask)
                pto.vsts(result, dst[row, col:], mask)

    return template


def register_binary(*, op, name, vector_op, dtypes, has_tmp=False):
    """Register a binary tile traversal, retaining an optional TileOp tmp operand."""

    if has_tmp:

        @tilelib.tile_template(
            op=op,
            target="a5",
            name=name,
            dtypes=dtypes,
            iteration_axis="none",
            op_engine="vector",
            op_class="elementwise",
            constraints=_common_constraints("src0", "src1", "tmp", "dst"),
            id=0,
            loop_depth=2,
            is_post_update=False,
            tags=("elementwise", "binary"),
        )
        def template(src0: pto.Tile, src1: pto.Tile, tmp: pto.Tile, dst: pto.Tile):
            _ = tmp
            dtype = dst.dtype
            valid_rows, valid_cols = dst.valid_shape
            lanes = pto.elements_per_vreg(dtype)

            for row in range(0, valid_rows, 1):
                remained = valid_cols
                for col in range(0, valid_cols, lanes):
                    mask, remained = pto.make_mask(dtype, remained)
                    lhs = pto.vlds(src0[row, col:])
                    rhs = pto.vlds(src1[row, col:])
                    result = vector_op(lhs, rhs, mask)
                    pto.vsts(result, dst[row, col:], mask)

        return template

    @tilelib.tile_template(
        op=op,
        target="a5",
        name=name,
        dtypes=dtypes,
        iteration_axis="none",
        op_engine="vector",
        op_class="elementwise",
        constraints=_common_constraints("src0", "src1", "dst"),
        id=0,
        loop_depth=2,
        is_post_update=False,
        tags=("elementwise", "binary"),
    )
    def template(src0: pto.Tile, src1: pto.Tile, dst: pto.Tile):
        dtype = dst.dtype
        valid_rows, valid_cols = dst.valid_shape
        lanes = pto.elements_per_vreg(dtype)

        for row in range(0, valid_rows, 1):
            remained = valid_cols
            for col in range(0, valid_cols, lanes):
                mask, remained = pto.make_mask(dtype, remained)
                lhs = pto.vlds(src0[row, col:])
                rhs = pto.vlds(src1[row, col:])
                result = vector_op(lhs, rhs, mask)
                pto.vsts(result, dst[row, col:], mask)

    return template


def register_scalar_binary(*, op, name, vector_op, dtypes, broadcast_scalar=False,
                           has_tmp=False, tmp_matches_src_dst=True,
                           reverse_name=None):
    """Register a tile/scalar traversal using either a vector-scalar or broadcast op."""

    constraints = _common_constraints("src", "dst")
    if has_tmp:
        if tmp_matches_src_dst:
            constraints = _common_constraints("src", "tmp", "dst")
        else:
            constraints = _common_constraints("src", "dst") + [
                _ub_or_vec_row_major,
            ]

        @tilelib.tile_template(
            op=op,
            target="a5",
            name=name,
            dtypes=dtypes,
            iteration_axis="none",
            op_engine="vector",
            op_class="elementwise",
            constraints=constraints,
            id=0,
            loop_depth=2,
            is_post_update=False,
            tags=("elementwise", "scalar"),
        )
        def template(src: pto.Tile, scalar, tmp: pto.Tile, dst: pto.Tile):
            _ = tmp
            _emit_scalar_binary_body(src, scalar, dst, vector_op, broadcast_scalar)

        if reverse_name:

            @tilelib.tile_template(
                op=op,
                target="a5",
                name=reverse_name,
                dtypes=dtypes,
                iteration_axis="none",
                op_engine="vector",
                op_class="elementwise",
                constraints=constraints,
                id=1,
                loop_depth=2,
                is_post_update=False,
                tags=("elementwise", "scalar"),
            )
            def reverse_template(scalar, src: pto.Tile, tmp: pto.Tile, dst: pto.Tile):
                _ = tmp
                _emit_scalar_binary_body(
                    src,
                    scalar,
                    dst,
                    vector_op,
                    broadcast_scalar=True,
                    scalar_lhs=True,
                )

            return template, reverse_template

        return template

    @tilelib.tile_template(
        op=op,
        target="a5",
        name=name,
        dtypes=dtypes,
        iteration_axis="none",
        op_engine="vector",
        op_class="elementwise",
        constraints=constraints,
        id=0,
        loop_depth=2,
        is_post_update=False,
        tags=("elementwise", "scalar"),
    )
    def template(src: pto.Tile, scalar, dst: pto.Tile):
        _emit_scalar_binary_body(src, scalar, dst, vector_op, broadcast_scalar)

    if reverse_name:

        @tilelib.tile_template(
            op=op,
            target="a5",
            name=reverse_name,
            dtypes=dtypes,
            iteration_axis="none",
            op_engine="vector",
            op_class="elementwise",
            constraints=constraints,
            id=1,
            loop_depth=2,
            is_post_update=False,
            tags=("elementwise", "scalar"),
        )
        def reverse_template(scalar, src: pto.Tile, dst: pto.Tile):
            _emit_scalar_binary_body(
                src,
                scalar,
                dst,
                vector_op,
                broadcast_scalar=True,
                scalar_lhs=True,
            )

        return template, reverse_template

    return template


def register_scalar_fill(*, op, name, dtypes):
    """Register a scalar-to-tile fill traversal."""

    @tilelib.tile_template(
        op=op,
        target="a5",
        name=name,
        dtypes=dtypes,
        iteration_axis="none",
        op_engine="vector",
        op_class="elementwise",
        constraints=[
            tilelib.check_memory_space("ub"),
            tilelib.check_layout("row_major"),
            tilelib.check_s_layout("none_box"),
        ],
        id=0,
        loop_depth=2,
        is_post_update=False,
        tags=("elementwise", "scalar", "fill"),
    )
    def template(scalar, dst: pto.Tile):
        dtype = dst.dtype
        valid_rows, valid_cols = dst.valid_shape
        cols = dst.shape[1]
        lanes = pto.elements_per_vreg(dtype)
        dst_ptr = dst.as_ptr()

        for row in range(0, valid_rows, 1):
            remained = valid_cols
            for col in range(0, valid_cols, lanes):
                mask, remained = pto.make_mask(dtype, remained)
                value = pto.vdup(scalar, mask)
                addr = pto.addptr(dst_ptr, row * cols + col)
                pto.vsts(value, addr, 0, mask)

    return template


def _emit_scalar_binary_body(src, scalar, dst, vector_op, broadcast_scalar,
                             scalar_lhs=False):
    dtype = dst.dtype
    valid_rows, valid_cols = dst.valid_shape
    lanes = pto.elements_per_vreg(dtype)

    with pto.for_(0, valid_rows, step=1) as row:
        col_loop = pto.for_(0, valid_cols, step=lanes).carry(remained=valid_cols)
        with col_loop:
            col = col_loop.iv
            mask, remained = pto.make_mask(dtype, col_loop.remained)
            value = pto.vlds(src[row, col:])
            if broadcast_scalar or scalar_lhs:
                scalar_value = pto.vbr(scalar)
                lhs, rhs = (
                    (scalar_value, value) if scalar_lhs else (value, scalar_value)
                )
                result = vector_op(lhs, rhs, mask)
            else:
                result = vector_op(value, scalar, mask)
            pto.vsts(result, dst[row, col:], mask)
            col_loop.update(remained=remained)


__all__ = [
    "register_binary",
    "register_scalar_binary",
    "register_scalar_fill",
    "register_unary",
]
