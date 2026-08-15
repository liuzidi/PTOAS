# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""PTODSL TileLib templates for row-wise ``pto.tcvt`` paths."""

from ptodsl import pto
from ptodsl._surface_values import unwrap_surface_value, wrap_surface_value
import ptodsl.tilelib as tilelib
from mlir.dialects import pto as _pto


def _rowwise(src_shape, src_valid_shape, dst_shape, dst_valid_shape, src_config, dst_config, **_):
    return (
        tuple(src_shape) == tuple(dst_shape)
        and tuple(src_valid_shape) == tuple(dst_valid_shape)
        and src_config.b_layout == "row_major"
        and dst_config.b_layout == "row_major"
        and src_config.s_layout == "none_box"
        and dst_config.s_layout == "none_box"
    )


def _rowwise_bf16_to_fp4(src_shape, src_valid_shape, dst_shape, dst_valid_shape, src_config, dst_config, **_):
    return (
        len(src_shape) == 2
        and len(dst_shape) == 2
        and src_shape[0] == dst_shape[0]
        and src_shape[1] == dst_shape[1] * 2
        and src_valid_shape[0] == dst_valid_shape[0]
        and src_valid_shape[1] == dst_valid_shape[1] * 2
        and src_config.b_layout == "row_major"
        and dst_config.b_layout == "row_major"
        and src_config.s_layout == "none_box"
        and dst_config.s_layout == "none_box"
    )


def _round_mode():
    round_mode = pto.get_op_attr("round_mode", "RINT")
    if round_mode == "ROUND":
        return pto.VcvtRoundMode.A
    if round_mode == "FLOOR":
        return pto.VcvtRoundMode.F
    if round_mode == "CEIL":
        return pto.VcvtRoundMode.C
    if round_mode == "TRUNC":
        return pto.VcvtRoundMode.Z
    if round_mode == "ODD":
        return pto.VcvtRoundMode.O
    return pto.VcvtRoundMode.R


def _sat_mode(token):
    if token == "nosat":
        return pto.VcvtSatMode.NOSAT
    if token == "sat":
        return pto.VcvtSatMode.SAT
    return None


def _part_mode(token):
    if token == "even":
        return pto.VcvtPartMode.EVEN
    if token == "p0":
        return pto.VcvtPartMode.P0
    return None


def _vselr_low_precision(src, idx):
    raw_src = unwrap_surface_value(src)
    return wrap_surface_value(_pto.VselrOp(raw_src.type, raw_src, unwrap_surface_value(idx)).result)


def _render_tcvt(
    src,
    dst,
    *,
    rnd=False,
    sat=None,
    part=None,
    load_dist=None,
    store_dist=None,
    mask_dtype="dst",
    convert_mask="store",
):
    valid_rows, valid_cols = dst.valid_shape
    dtype = dst.dtype
    loop_dtype = src.dtype if mask_dtype == "src" else dtype
    lanes = pto.elements_per_vreg(loop_dtype)
    with pto.for_(0, valid_rows, step=1) as row:
        remained = valid_cols
        col_loop = pto.for_(0, valid_cols, step=lanes).carry(remained=remained)
        with col_loop:
            col = col_loop.iv
            mask, remained = pto.make_mask(loop_dtype, remained)
            convert_mask_value = mask
            if convert_mask == "src_full":
                convert_mask_value = pto.make_mask(src.dtype, pto.PAT.ALL)
            vec = pto.vlds(src[row, col:], dist=load_dist) if load_dist else pto.vlds(src[row, col:])
            kwargs = {}
            if rnd:
                kwargs["rnd"] = _round_mode()
            sat_mode = _sat_mode(sat)
            if sat_mode is not None:
                kwargs["sat"] = sat_mode
            part_mode = _part_mode(part)
            if part_mode is not None:
                kwargs["part"] = part_mode
            converted = pto.vcvt(vec, dtype, convert_mask_value, **kwargs)
            if store_dist:
                pto.vsts(converted, dst[row, col:], mask, dist=store_dist)
            else:
                pto.vsts(converted, dst[row, col:], mask)
            col_loop.update(remained=remained)


def _register_tcvt(
    *,
    name,
    dtypes,
    idx,
    rnd,
    sat=None,
    part=None,
    load_dist=None,
    store_dist=None,
    mask_dtype="dst",
    convert_mask="store",
):
    @tilelib.tile_template(
        op="pto.tcvt",
        target="a5",
        name=name,
        dtypes=[dtypes],
        iteration_axis="none",
        op_engine="vector",
        op_class="other",
        constraints=[_rowwise],
        id=idx,
        loop_depth=2,
        is_post_update=False,
        tags=("convert", "rowwise"),
    )
    def template(src: pto.Tile, dst: pto.Tile):
        _render_tcvt(
            src,
            dst,
            rnd=rnd,
            sat=sat,
            part=part,
            load_dist=load_dist,
            store_dist=store_dist,
            mask_dtype=mask_dtype,
            convert_mask=convert_mask,
        )

    return template


template_tcvt_f32_to_i32 = _register_tcvt(
    name="template_tcvt_f32_to_i32",
    dtypes=("f32", "i32"),
    idx=0,
    rnd=True,
    sat="sat",
)

template_tcvt_i32_to_f32 = _register_tcvt(
    name="template_tcvt_i32_to_f32",
    dtypes=("i32", "f32"),
    idx=1,
    rnd=True,
)

template_tcvt_i16_to_f16 = _register_tcvt(
    name="template_tcvt_i16_to_f16",
    dtypes=("i16", "f16"),
    idx=2,
    rnd=True,
)

@tilelib.tile_template(
    op="pto.tcvt",
    target="a5",
    name="template_tcvt_f16_to_i16",
    dtypes=[("f16", "i16")],
    iteration_axis="none",
    op_engine="vector",
    op_class="other",
    constraints=[_rowwise],
    id=3,
    loop_depth=2,
    is_post_update=False,
    tags=("convert", "rowwise"),
)
def template_tcvt_f16_to_i16(src: pto.Tile, dst: pto.Tile):
    valid_rows, valid_cols = dst.valid_shape
    lanes_f32 = pto.elements_per_vreg(pto.f32)
    full_mask_b16 = pto.make_mask(src.dtype, pto.PAT.ALL)
    full_mask_b32 = pto.make_mask(pto.i32, pto.PAT.ALL)
    with pto.for_(0, valid_rows, step=1) as row:
        remained = valid_cols
        col_loop = pto.for_(0, valid_cols, step=lanes_f32).carry(remained=remained)
        with col_loop:
            col = col_loop.iv
            store_mask, remained = pto.make_mask(pto.i32, remained)
            vec_f16 = pto.vlds(src[row, col:], dist="UNPK_B16")
            vec_i32 = pto.vcvt(
                vec_f16,
                pto.i32,
                full_mask_b16,
                rnd=_round_mode(),
                part=pto.VcvtPartMode.EVEN,
            )
            vec_i16 = pto.vcvt(
                vec_i32,
                pto.i16,
                full_mask_b32,
                sat=pto.VcvtSatMode.NOSAT,
                part=pto.VcvtPartMode.EVEN,
            )
            pto.vsts(vec_i16, dst[row, col:], store_mask, dist=pto.VStoreDist.PK_B32)
            col_loop.update(remained=remained)

template_tcvt_bf16_to_f16 = _register_tcvt(
    name="template_tcvt_bf16_to_f16",
    dtypes=("bf16", "f16"),
    idx=4,
    rnd=True,
    sat="sat",
)

template_tcvt_f32_to_f16 = _register_tcvt(
    name="template_tcvt_f32_to_f16",
    dtypes=("f32", "f16"),
    idx=5,
    rnd=True,
    sat="sat",
    part="even",
    store_dist=pto.VStoreDist.PK_B32,
    mask_dtype="src",
)

template_tcvt_f32_to_bf16 = _register_tcvt(
    name="template_tcvt_f32_to_bf16",
    dtypes=("f32", "bf16"),
    idx=6,
    rnd=True,
    sat="sat",
    part="even",
    store_dist=pto.VStoreDist.PK_B32,
    mask_dtype="src",
)


@tilelib.tile_template(
    op="pto.tcvt",
    target="a5",
    name="template_tcvt_f32_to_f32",
    dtypes=[("f32", "f32")],
    iteration_axis="none",
    op_engine="vector",
    op_class="other",
    constraints=[_rowwise],
    id=17,
    loop_depth=2,
    is_post_update=False,
    tags=("convert", "rowwise"),
)
def template_tcvt_f32_to_f32(src: pto.Tile, dst: pto.Tile):
    valid_rows, valid_cols = dst.valid_shape
    lanes_f32 = pto.elements_per_vreg(src.dtype)
    with pto.for_(0, valid_rows, step=1) as row:
        remained = valid_cols
        col_loop = pto.for_(0, valid_cols, step=lanes_f32).carry(remained=remained)
        with col_loop:
            col = col_loop.iv
            mask, remained = pto.make_mask(src.dtype, remained)
            vec = pto.vlds(src[row, col:])
            converted = pto.vtrc(vec, mask, rnd=_round_mode())
            pto.vsts(converted, dst[row, col:], mask)
            col_loop.update(remained=remained)


@tilelib.tile_template(
    op="pto.tcvt",
    target="a5",
    name="template_tcvt_f32_to_i16",
    dtypes=[("f32", "i16")],
    iteration_axis="none",
    op_engine="vector",
    op_class="other",
    constraints=[_rowwise],
    id=15,
    loop_depth=2,
    is_post_update=False,
    tags=("convert", "rowwise"),
)
def template_tcvt_f32_to_i16(src: pto.Tile, dst: pto.Tile):
    valid_rows, valid_cols = dst.valid_shape
    lanes_f32 = pto.elements_per_vreg(src.dtype)
    full_mask = pto.make_mask(src.dtype, pto.PAT.ALL)
    with pto.for_(0, valid_rows, step=1) as row:
        remained = valid_cols
        col_loop = pto.for_(0, valid_cols, step=lanes_f32).carry(remained=remained)
        with col_loop:
            col = col_loop.iv
            store_mask, remained = pto.make_mask(src.dtype, remained)
            vec_f32 = pto.vlds(src[row, col:])
            vec_i32 = pto.vcvt(
                vec_f32,
                pto.i32,
                full_mask,
                rnd=_round_mode(),
                sat=pto.VcvtSatMode.NOSAT,
            )
            vec_i16 = pto.vcvt(
                vec_i32,
                pto.i16,
                full_mask,
                sat=pto.VcvtSatMode.NOSAT,
                part=pto.VcvtPartMode.EVEN,
            )
            pto.vsts(vec_i16, dst[row, col:], store_mask, dist=pto.VStoreDist.PK_B32)
            col_loop.update(remained=remained)


@tilelib.tile_template(
    op="pto.tcvt",
    target="a5",
    name="template_tcvt_f32_to_i64",
    dtypes=[("f32", "i64")],
    iteration_axis="none",
    op_engine="vector",
    op_class="other",
    constraints=[_rowwise],
    id=16,
    loop_depth=2,
    is_post_update=False,
    tags=("convert", "rowwise"),
)
def template_tcvt_f32_to_i64(src: pto.Tile, dst: pto.Tile):
    valid_rows, valid_cols = dst.valid_shape
    lanes_i64 = pto.elements_per_vreg(dst.dtype)
    full_mask = pto.make_mask(src.dtype, pto.PAT.ALL)
    with pto.for_(0, valid_rows, step=1) as row:
        remained = valid_cols * 2
        col_loop = pto.for_(0, valid_cols, step=lanes_i64).carry(remained=remained)
        with col_loop:
            col = col_loop.iv
            store_mask, remained = pto.make_mask(pto.i32, remained)
            vec = pto.vlds(src[row, col:], dist="UNPK_B32")
            converted = pto.vcvt(
                vec,
                pto.i64,
                full_mask,
                rnd=_round_mode(),
                sat=pto.VcvtSatMode.SAT,
                part=pto.VcvtPartMode.EVEN,
            )
            pto.vsts(converted, dst[row, col:], store_mask, dist=pto.VStoreDist.NORM_B32)
            col_loop.update(remained=remained)

template_tcvt_f16_to_i32 = _register_tcvt(
    name="template_tcvt_f16_to_i32",
    dtypes=("f16", "i32"),
    idx=7,
    rnd=True,
    part="even",
    load_dist="UNPK_B16",
    convert_mask="src_full",
)

template_tcvt_f16_to_f32 = _register_tcvt(
    name="template_tcvt_f16_to_f32",
    dtypes=("f16", "f32"),
    idx=8,
    rnd=False,
    part="even",
    load_dist="UNPK_B16",
    convert_mask="src_full",
)


@tilelib.tile_template(
    op="pto.tcvt",
    target="a5",
    name="template_tcvt_f16_to_ui8",
    dtypes=[("f16", "ui8")],
    iteration_axis="none",
    op_engine="vector",
    op_class="other",
    constraints=[_rowwise],
    id=18,
    loop_depth=2,
    is_post_update=False,
    tags=("convert", "rowwise"),
)
def template_tcvt_f16_to_ui8(src: pto.Tile, dst: pto.Tile):
    valid_rows, valid_cols = dst.valid_shape
    lanes_f16 = pto.elements_per_vreg(src.dtype)
    full_mask = pto.make_mask(src.dtype, pto.PAT.ALL)
    with pto.for_(0, valid_rows, step=1) as row:
        remained = valid_cols
        col_loop = pto.for_(0, valid_cols, step=lanes_f16).carry(remained=remained)
        with col_loop:
            col = col_loop.iv
            store_mask, remained = pto.make_mask(src.dtype, remained)
            vec = pto.vlds(src[row, col:])
            converted = pto.vcvt(
                vec,
                pto.ui8,
                full_mask,
                rnd=_round_mode(),
                sat=pto.VcvtSatMode.NOSAT,
                part=pto.VcvtPartMode.EVEN,
            )
            pto.vsts(converted, dst[row, col:], store_mask, dist=pto.VStoreDist.PK_B16)
            col_loop.update(remained=remained)


@tilelib.tile_template(
    op="pto.tcvt",
    target="a5",
    name="template_tcvt_f16_to_si8",
    dtypes=[("f16", "si8"), ("f16", "i8")],
    iteration_axis="none",
    op_engine="vector",
    op_class="other",
    constraints=[_rowwise],
    id=19,
    loop_depth=2,
    is_post_update=False,
    tags=("convert", "rowwise"),
)
def template_tcvt_f16_to_si8(src: pto.Tile, dst: pto.Tile):
    valid_rows, valid_cols = dst.valid_shape
    lanes_f16 = pto.elements_per_vreg(src.dtype)
    pg = pto.make_mask(src.dtype, pto.PAT.ALL)
    with pto.for_(0, valid_rows, step=1) as row:
        remained = valid_cols
        col_loop = pto.for_(0, valid_cols, step=lanes_f16).carry(remained=remained)
        with col_loop:
            col = col_loop.iv
            full_mask, _ = pto.make_mask(src.dtype, lanes_f16)
            store_mask, remained = pto.make_mask(src.dtype, remained)
            vec_f16 = pto.vlds(src[row, col:])
            vec_i16 = pto.vcvt(
                vec_f16,
                pto.i16,
                full_mask,
                rnd=_round_mode(),
                sat=pto.VcvtSatMode.NOSAT,
            )
            v_mask = pto.vdup(pto.i16(255), pg)
            vec_i16_and = pto.vand(vec_i16, v_mask, store_mask)
            vec_f16_temp = pto.vcvt(
                vec_i16_and,
                pto.f16,
                full_mask,
                rnd=_round_mode(),
            )
            vec_si8 = pto.vcvt(
                vec_f16_temp,
                pto.si8,
                full_mask,
                rnd=_round_mode(),
                sat=pto.VcvtSatMode.NOSAT,
                part=pto.VcvtPartMode.EVEN,
            )
            if dst.dtype == "i8":
                vec_si8 = pto.vbitcast(vec_si8, pto.i8)
            pto.vsts(vec_si8, dst[row, col:], store_mask, dist=pto.VStoreDist.PK_B16)
            col_loop.update(remained=remained)


template_tcvt_bf16_to_f32 = _register_tcvt(
    name="template_tcvt_bf16_to_f32",
    dtypes=("bf16", "f32"),
    idx=20,
    rnd=False,
    part="even",
    load_dist="UNPK_B16",
    convert_mask="src_full",
)

template_tcvt_i16_to_f32 = _register_tcvt(
    name="template_tcvt_i16_to_f32",
    dtypes=("i16", "f32"),
    idx=21,
    rnd=False,
    part="even",
    load_dist="UNPK_B16",
    convert_mask="src_full",
)

template_tcvt_i16_to_i32 = _register_tcvt(
    name="template_tcvt_i16_to_i32",
    dtypes=("i16", "i32"),
    idx=22,
    rnd=False,
    part="even",
    load_dist="UNPK_B16",
    convert_mask="src_full",
)

template_tcvt_i16_to_ui32 = _register_tcvt(
    name="template_tcvt_i16_to_ui32",
    dtypes=("i16", "ui32"),
    idx=23,
    rnd=False,
    part="even",
    load_dist="UNPK_B16",
    convert_mask="src_full",
)

template_tcvt_bf16_to_i32 = _register_tcvt(
    name="template_tcvt_bf16_to_i32",
    dtypes=("bf16", "i32"),
    idx=9,
    rnd=True,
    sat="sat",
    part="even",
    load_dist="UNPK_B16",
    convert_mask="src_full",
)

template_tcvt_ui8_to_ui16 = _register_tcvt(
    name="template_tcvt_ui8_to_ui16",
    dtypes=("ui8", "ui16"),
    idx=10,
    rnd=False,
    part="even",
    load_dist="UNPK_B8",
    convert_mask="src_full",
)

template_tcvt_ui8_to_f16 = _register_tcvt(
    name="template_tcvt_ui8_to_f16",
    dtypes=("ui8", "f16"),
    idx=24,
    rnd=False,
    part="even",
    load_dist="UNPK_B8",
    convert_mask="src_full",
)

template_tcvt_si8_to_f16 = _register_tcvt(
    name="template_tcvt_si8_to_f16",
    dtypes=("si8", "f16"),
    idx=25,
    rnd=False,
    part="even",
    load_dist="UNPK_B8",
    convert_mask="src_full",
)

template_tcvt_si8_to_si16 = _register_tcvt(
    name="template_tcvt_si8_to_si16",
    dtypes=("si8", "si16"),
    idx=26,
    rnd=False,
    part="even",
    load_dist="UNPK_B8",
    store_dist=pto.VStoreDist.NORM_B16,
    convert_mask="src_full",
)

template_tcvt_i32_to_i16 = _register_tcvt(
    name="template_tcvt_i32_to_i16",
    dtypes=("i32", "i16"),
    idx=28,
    rnd=False,
    sat="nosat",
    part="even",
    store_dist=pto.VStoreDist.PK_B32,
    mask_dtype="src",
    convert_mask="src_full",
)

template_tcvt_i32_to_ui16 = _register_tcvt(
    name="template_tcvt_i32_to_ui16",
    dtypes=("i32", "ui16"),
    idx=29,
    rnd=False,
    sat="sat",
    part="even",
    store_dist=pto.VStoreDist.PK_B32,
    mask_dtype="src",
    convert_mask="src_full",
)


@tilelib.tile_template(
    op="pto.tcvt",
    target="a5",
    name="template_tcvt_i32_to_f16",
    dtypes=[("i32", "f16")],
    iteration_axis="none",
    op_engine="vector",
    op_class="other",
    constraints=[_rowwise],
    id=38,
    loop_depth=2,
    is_post_update=False,
    tags=("convert", "rowwise"),
)
def template_tcvt_i32_to_f16(src: pto.Tile, dst: pto.Tile):
    valid_rows, valid_cols = dst.valid_shape
    lanes_i32 = pto.elements_per_vreg(src.dtype)
    full_mask_i32 = pto.make_mask(src.dtype, pto.PAT.ALL)
    full_mask_f32 = pto.make_mask(pto.f32, pto.PAT.ALL)
    with pto.for_(0, valid_rows, step=1) as row:
        remained = valid_cols
        col_loop = pto.for_(0, valid_cols, step=lanes_i32).carry(remained=remained)
        with col_loop:
            col = col_loop.iv
            store_mask, remained = pto.make_mask(src.dtype, remained)
            vec_i32 = pto.vlds(src[row, col:])
            vec_f32 = pto.vcvt(
                vec_i32,
                pto.f32,
                full_mask_i32,
                rnd=_round_mode(),
            )
            vec_f16 = pto.vcvt(
                vec_f32,
                pto.f16,
                full_mask_f32,
                rnd=_round_mode(),
                sat=pto.VcvtSatMode.SAT,
                part=pto.VcvtPartMode.EVEN,
            )
            pto.vsts(vec_f16, dst[row, col:], store_mask, dist=pto.VStoreDist.PK_B32)
            col_loop.update(remained=remained)


template_tcvt_ui32_to_i16 = _register_tcvt(
    name="template_tcvt_ui32_to_i16",
    dtypes=("ui32", "i16"),
    idx=30,
    rnd=False,
    sat="sat",
    part="even",
    store_dist=pto.VStoreDist.PK_B32,
    mask_dtype="src",
    convert_mask="src_full",
)

template_tcvt_ui32_to_ui16 = _register_tcvt(
    name="template_tcvt_ui32_to_ui16",
    dtypes=("ui32", "ui16"),
    idx=31,
    rnd=False,
    sat="sat",
    part="even",
    store_dist=pto.VStoreDist.PK_B32,
    mask_dtype="src",
    convert_mask="src_full",
)


@tilelib.tile_template(
    op="pto.tcvt",
    target="a5",
    name="template_tcvt_si8_to_i32",
    dtypes=[("si8", "i32")],
    iteration_axis="none",
    op_engine="vector",
    op_class="other",
    constraints=[_rowwise],
    id=32,
    loop_depth=2,
    is_post_update=False,
    tags=("convert", "rowwise"),
)
def template_tcvt_si8_to_i32(src: pto.Tile, dst: pto.Tile):
    valid_rows, valid_cols = dst.valid_shape
    b8_mask = pto.make_mask(pto.ui8, pto.PAT.ALL)
    v_zero = pto.vbitcast(pto.vdup(pto.i8(0), b8_mask), pto.ui8)
    lanes_i16 = pto.elements_per_vreg(pto.i16)
    lanes_i32 = pto.elements_per_vreg(pto.i32)
    with pto.for_(0, valid_rows, step=1) as row:
        remained = valid_cols
        next_remained = valid_cols - lanes_i32
        col_loop = pto.for_(0, valid_cols, step=lanes_i16).carry(
            remained=remained,
            next_remained=next_remained,
        )
        with col_loop:
            col = col_loop.iv
            mask_b16_cur, remained = pto.make_mask(pto.i16, remained)
            mask_b16_next, next_remained = pto.make_mask(pto.i16, next_remained)
            mask_b32_cur = pto.punpack(mask_b16_cur, pto.PredicatePart.LOWER, to_type=pto.mask_b32)
            mask_b32_next = pto.punpack(mask_b16_next, pto.PredicatePart.LOWER, to_type=pto.mask_b32)
            vec_si8_0 = pto.vlds(src[row, col:], dist="UNPK_B8")
            vec_ui8_0 = pto.vbitcast(vec_si8_0, pto.ui8)
            vec_ui8_1, vec_ui8_2 = pto.vintlv(vec_ui8_0, v_zero)
            vec_si8_1 = pto.vbitcast(vec_ui8_1, pto.si8)
            vec_si8_2 = pto.vbitcast(vec_ui8_2, pto.si8)
            output_0 = pto.vcvt(vec_si8_1, pto.i32, b8_mask, part=pto.VcvtPartMode.P0)
            output_1 = pto.vcvt(vec_si8_2, pto.i32, b8_mask, part=pto.VcvtPartMode.P0)
            pto.vsts(output_0, dst[row, col:], mask_b32_cur, dist=pto.VStoreDist.NORM_B32)
            pto.vsts(output_1, dst[row, col + lanes_i32:], mask_b32_next, dist=pto.VStoreDist.NORM_B32)
            col_loop.update(remained=remained, next_remained=next_remained)


def _render_32_to_ui8(src: pto.Tile, dst: pto.Tile):
    valid_rows, valid_cols = dst.valid_shape
    full_mask = pto.make_mask(src.dtype, pto.PAT.ALL)
    idx_mask_b8 = pto.pset_b8(pto.PAT.ALL)
    idx_mask_b16 = pto.pbitcast(idx_mask_b8, pto.mask_b16)
    lanes = pto.elements_per_vreg(src.dtype)
    v_idx = pto.vci(pto.i8(0), "ASC")
    v_idx_i16 = pto.vbitcast(v_idx, pto.i16)
    v_idx_i16 = pto.vmuls(v_idx_i16, pto.i16(4), idx_mask_b16)
    v_idx_ui8 = pto.vbitcast(v_idx_i16, pto.ui8)
    with pto.for_(0, valid_rows, step=1) as row:
        remained = valid_cols
        col_loop = pto.for_(0, valid_cols, step=lanes).carry(remained=remained)
        with col_loop:
            col = col_loop.iv
            store_mask, remained = pto.make_mask(pto.ui8, remained)
            vec = pto.vlds(src[row, col:])
            converted = pto.vcvt(
                vec,
                pto.ui8,
                full_mask,
                sat=pto.VcvtSatMode.SAT,
                part=pto.VcvtPartMode.P0,
            )
            result = pto.vselr(converted, v_idx_ui8)
            pto.mem_bar(pto.BarrierType.VST_VST)
            pto.vsts(result, dst[row, col:], store_mask, dist=pto.VStoreDist.NORM_B8)
            col_loop.update(remained=remained)


@tilelib.tile_template(
    op="pto.tcvt",
    target="a5",
    name="template_tcvt_i32_to_ui8",
    dtypes=[("i32", "ui8")],
    iteration_axis="none",
    op_engine="vector",
    op_class="other",
    constraints=[_rowwise],
    id=33,
    loop_depth=2,
    is_post_update=False,
    tags=("convert", "rowwise"),
)
def template_tcvt_i32_to_ui8(src: pto.Tile, dst: pto.Tile):
    _render_32_to_ui8(src, dst)


@tilelib.tile_template(
    op="pto.tcvt",
    target="a5",
    name="template_tcvt_ui32_to_ui8",
    dtypes=[("ui32", "ui8")],
    iteration_axis="none",
    op_engine="vector",
    op_class="other",
    constraints=[_rowwise],
    id=34,
    loop_depth=2,
    is_post_update=False,
    tags=("convert", "rowwise"),
)
def template_tcvt_ui32_to_ui8(src: pto.Tile, dst: pto.Tile):
    _render_32_to_ui8(src, dst)


@tilelib.tile_template(
    op="pto.tcvt",
    target="a5",
    name="template_tcvt_i16_to_ui8",
    dtypes=[("i16", "ui8")],
    iteration_axis="none",
    op_engine="vector",
    op_class="other",
    constraints=[_rowwise],
    id=35,
    loop_depth=2,
    is_post_update=False,
    tags=("convert", "rowwise"),
)
def template_tcvt_i16_to_ui8(src: pto.Tile, dst: pto.Tile):
    valid_rows, valid_cols = dst.valid_shape
    lanes_i16 = pto.elements_per_vreg(src.dtype)
    full_mask = pto.make_mask(src.dtype, pto.PAT.ALL)
    with pto.for_(0, valid_rows, step=1) as row:
        remained = valid_cols
        col_loop = pto.for_(0, valid_cols, step=lanes_i16).carry(remained=remained)
        with col_loop:
            col = col_loop.iv
            store_mask, remained = pto.make_mask(src.dtype, remained)
            vec = pto.vlds(src[row, col:])
            converted = pto.vcvt(
                vec,
                pto.ui8,
                full_mask,
                sat=pto.VcvtSatMode.SAT,
                part=pto.VcvtPartMode.EVEN,
            )
            pto.vsts(converted, dst[row, col:], store_mask, dist=pto.VStoreDist.PK_B16)
            col_loop.update(remained=remained)


@tilelib.tile_template(
    op="pto.tcvt",
    target="a5",
    name="template_tcvt_i32_to_i64",
    dtypes=[("i32", "i64")],
    iteration_axis="none",
    op_engine="vector",
    op_class="other",
    constraints=[_rowwise],
    id=27,
    loop_depth=2,
    is_post_update=False,
    tags=("convert", "rowwise"),
)
def template_tcvt_i32_to_i64(src: pto.Tile, dst: pto.Tile):
    valid_rows, valid_cols = dst.valid_shape
    lanes_i64 = pto.elements_per_vreg(dst.dtype)
    full_mask = pto.make_mask(src.dtype, pto.PAT.ALL)
    with pto.for_(0, valid_rows, step=1) as row:
        remained = valid_cols * 2
        col_loop = pto.for_(0, valid_cols, step=lanes_i64).carry(remained=remained)
        with col_loop:
            col = col_loop.iv
            store_mask, remained = pto.make_mask(pto.i32, remained)
            vec = pto.vlds(src[row, col:], dist="UNPK_B32")
            converted = pto.vcvt(
                vec,
                pto.i64,
                full_mask,
                part=pto.VcvtPartMode.EVEN,
            )
            pto.vsts(converted, dst[row, col:], store_mask, dist=pto.VStoreDist.NORM_B32)
            col_loop.update(remained=remained)


def _render_i64_to_32(src: pto.Tile, dst: pto.Tile, *, use_rounding: bool):
    valid_rows, valid_cols = dst.valid_shape
    lanes_i64 = pto.elements_per_vreg(src.dtype)
    with pto.for_(0, valid_rows, step=1) as row:
        remained = valid_cols * 2
        full_mask, _ = pto.make_mask(pto.i32, remained)
        col_loop = pto.for_(0, valid_cols, step=lanes_i64).carry(remained=remained)
        with col_loop:
            col = col_loop.iv
            store_mask, remained = pto.make_mask(dst.dtype, remained)
            vec = pto.vlds(src[row, col:])
            kwargs = {"part": pto.VcvtPartMode.EVEN}
            if use_rounding:
                kwargs["rnd"] = _round_mode()
            else:
                kwargs["sat"] = pto.VcvtSatMode.NOSAT
            converted = pto.vcvt(vec, dst.dtype, full_mask, **kwargs)
            pto.vsts(converted, dst[row, col:], store_mask, dist=pto.VStoreDist.PK_B64)
            col_loop.update(remained=remained)


@tilelib.tile_template(
    op="pto.tcvt",
    target="a5",
    name="template_tcvt_i64_to_f32",
    dtypes=[("i64", "f32")],
    iteration_axis="none",
    op_engine="vector",
    op_class="other",
    constraints=[_rowwise],
    id=36,
    loop_depth=2,
    is_post_update=False,
    tags=("convert", "rowwise"),
)
def template_tcvt_i64_to_f32(src: pto.Tile, dst: pto.Tile):
    _render_i64_to_32(src, dst, use_rounding=True)


@tilelib.tile_template(
    op="pto.tcvt",
    target="a5",
    name="template_tcvt_i64_to_i32",
    dtypes=[("i64", "i32")],
    iteration_axis="none",
    op_engine="vector",
    op_class="other",
    constraints=[_rowwise],
    id=37,
    loop_depth=2,
    is_post_update=False,
    tags=("convert", "rowwise"),
)
def template_tcvt_i64_to_i32(src: pto.Tile, dst: pto.Tile):
    _render_i64_to_32(src, dst, use_rounding=False)


@tilelib.tile_template(
    op="pto.tcvt",
    target="a5",
    name="template_tcvt_f32_to_fp8",
    dtypes=[("f32", "f8e4m3"), ("f32", "f8e5m2")],
    iteration_axis="none",
    op_engine="vector",
    op_class="other",
    constraints=[_rowwise],
    id=11,
    loop_depth=2,
    is_post_update=False,
    tags=("convert", "rowwise", "low_precision"),
)
def template_tcvt_f32_to_fp8(src: pto.Tile, dst: pto.Tile):
    valid_rows, valid_cols = dst.valid_shape
    dst_dtype = dst.dtype
    lanes_f32 = pto.elements_per_vreg(src.dtype)
    src_mask = pto.make_mask(src.dtype, pto.PAT.ALL)
    idx_mask_b8 = pto.pset_b8(pto.PAT.ALL)
    idx_mask_b16 = pto.pbitcast(idx_mask_b8, pto.mask_b16)
    v_idx = pto.vci(pto.i8(0), "ASC")
    v_idx_i16 = pto.vbitcast(v_idx, pto.i16)
    v_idx_i16 = pto.vmuls(v_idx_i16, pto.i16(4), idx_mask_b16)
    v_idx_ui8 = pto.vbitcast(v_idx_i16, pto.ui8)
    with pto.for_(0, valid_rows, step=1) as row:
        remained = valid_cols
        col_loop = pto.for_(0, valid_cols, step=lanes_f32).carry(remained=remained)
        with col_loop:
            col = col_loop.iv
            dst_mask, remained = pto.make_mask(dst_dtype, remained)
            vec = pto.vlds(src[row, col:])
            converted = pto.vcvt(
                vec,
                dst_dtype,
                src_mask,
                rnd=_round_mode(),
                sat=pto.VcvtSatMode.SAT,
                part=pto.VcvtPartMode.P0,
            )
            result = _vselr_low_precision(converted, v_idx_ui8)
            pto.mem_bar(pto.BarrierType.VST_VST)
            pto.vsts(result, dst[row, col:], dst_mask, dist=pto.VStoreDist.NORM_B8)
            col_loop.update(remained=remained)


@tilelib.tile_template(
    op="pto.tcvt",
    target="a5",
    name="template_tcvt_f32_to_hif8",
    dtypes=[("f32", "hif8")],
    iteration_axis="none",
    op_engine="vector",
    op_class="other",
    constraints=[_rowwise],
    id=12,
    loop_depth=2,
    is_post_update=False,
    tags=("convert", "rowwise", "low_precision"),
)
def template_tcvt_f32_to_hif8(src: pto.Tile, dst: pto.Tile):
    valid_rows, valid_cols = dst.valid_shape
    dst_dtype = dst.dtype
    lanes_f32 = pto.elements_per_vreg(src.dtype)
    src_mask = pto.make_mask(src.dtype, pto.PAT.ALL)
    idx_mask_b8 = pto.pset_b8(pto.PAT.ALL)
    idx_mask_b16 = pto.pbitcast(idx_mask_b8, pto.mask_b16)
    v_idx = pto.vci(pto.i8(0), "ASC")
    v_idx_i16 = pto.vbitcast(v_idx, pto.i16)
    v_idx_i16 = pto.vmuls(v_idx_i16, pto.i16(4), idx_mask_b16)
    v_idx_ui8 = pto.vbitcast(v_idx_i16, pto.ui8)
    with pto.for_(0, valid_rows, step=1) as row:
        remained = valid_cols
        col_loop = pto.for_(0, valid_cols, step=lanes_f32).carry(remained=remained)
        with col_loop:
            col = col_loop.iv
            dst_mask, remained = pto.make_mask(dst_dtype, remained)
            vec = pto.vlds(src[row, col:])
            converted = pto.vcvt(
                vec,
                dst_dtype,
                src_mask,
                rnd=pto.VcvtRoundMode.A,
                sat=pto.VcvtSatMode.NOSAT,
                part=pto.VcvtPartMode.P0,
            )
            result = _vselr_low_precision(converted, v_idx_ui8)
            pto.mem_bar(pto.BarrierType.VST_VST)
            pto.vsts(result, dst[row, col:], dst_mask, dist=pto.VStoreDist.NORM_B8)
            col_loop.update(remained=remained)


@tilelib.tile_template(
    op="pto.tcvt",
    target="a5",
    name="template_tcvt_f16_to_hif8",
    dtypes=[("f16", "hif8")],
    iteration_axis="none",
    op_engine="vector",
    op_class="other",
    constraints=[_rowwise],
    id=13,
    loop_depth=2,
    is_post_update=False,
    tags=("convert", "rowwise", "low_precision"),
)
def template_tcvt_f16_to_hif8(src: pto.Tile, dst: pto.Tile):
    valid_rows, valid_cols = dst.valid_shape
    dst_dtype = dst.dtype
    lanes_f16 = pto.elements_per_vreg(src.dtype)
    src_mask = pto.make_mask(src.dtype, pto.PAT.ALL)
    with pto.for_(0, valid_rows, step=1) as row:
        remained = valid_cols
        col_loop = pto.for_(0, valid_cols, step=lanes_f16).carry(remained=remained)
        with col_loop:
            col = col_loop.iv
            dst_mask, remained = pto.make_mask(src.dtype, remained)
            vec = pto.vlds(src[row, col:])
            converted = pto.vcvt(
                vec,
                dst_dtype,
                src_mask,
                rnd=pto.VcvtRoundMode.A,
                sat=pto.VcvtSatMode.NOSAT,
                part=pto.VcvtPartMode.EVEN,
            )
            pto.vsts(converted, dst[row, col:], dst_mask, dist=pto.VStoreDist.PK_B16)
            col_loop.update(remained=remained)


@tilelib.tile_template(
    op="pto.tcvt",
    target="a5",
    name="template_tcvt_bf16_to_fp4",
    dtypes=[("bf16", "f4e1m2x2"), ("bf16", "f4e2m1x2")],
    iteration_axis="none",
    op_engine="vector",
    op_class="other",
    constraints=[_rowwise_bf16_to_fp4],
    id=14,
    loop_depth=2,
    is_post_update=False,
    tags=("convert", "rowwise", "low_precision"),
)
def template_tcvt_bf16_to_fp4(src: pto.Tile, dst: pto.Tile):
    valid_rows, valid_cols = dst.valid_shape
    dst_dtype = dst.dtype
    lanes_bf16 = pto.elements_per_vreg(src.dtype)
    dst_chunk_cols = lanes_bf16 // 2
    idx_mask_b8 = pto.pset_b8(pto.PAT.ALL)
    idx_mask_b16 = pto.pbitcast(idx_mask_b8, pto.mask_b16)
    v_idx = pto.vci(pto.i8(0), "ASC")
    v_idx_i16 = pto.vbitcast(v_idx, pto.i16)
    v_idx_i16 = pto.vmuls(v_idx_i16, pto.i16(4), idx_mask_b16)
    v_idx_ui8 = pto.vbitcast(v_idx_i16, pto.ui8)
    with pto.for_(0, valid_rows, step=1) as row:
        remained = valid_cols
        src_remained = valid_cols * 2
        col_loop = pto.for_(0, valid_cols, step=dst_chunk_cols).carry(
            remained=remained,
            src_remained=src_remained,
        )
        with col_loop:
            col = col_loop.iv
            dst_mask, remained = pto.make_mask(dst_dtype, remained)
            src_mask, src_remained = pto.make_mask(src.dtype, src_remained)
            vec = pto.vlds(src[row, col * 2:])
            converted = pto.vcvt(
                vec,
                dst_dtype,
                src_mask,
                rnd=_round_mode(),
                part=pto.VcvtPartMode.P0,
            )
            result = _vselr_low_precision(converted, v_idx_ui8)
            pto.mem_bar(pto.BarrierType.VST_VST)
            pto.vsts(result, dst[row, col:], dst_mask, dist=pto.VStoreDist.NORM_B8)
            col_loop.update(remained=remained, src_remained=src_remained)


from ._vmi_common import (  # noqa: E402
    canonical_vmi_template,
    convert_vmi_constraint,
    emit_convert_vmi,
)


@canonical_vmi_template(
    target="a5",
    op="tcvt",
    name="vmi_tcvt",
    dtypes=(
        ("bf16", "f32"),
        ("f16", "f32"),
        ("i32", "f32"),
        ("f32", "bf16"),
        ("f32", "f16"),
        ("f32", "i32"),
        ("i32", "f16"),
    ),
    context_constraints={
        "round_mode": ("RINT", "ROUND", "TRUNC"),
        "sat_mode": ("DEFAULT", "ON", "OFF"),
    },
    constraints=(convert_vmi_constraint,),
    min_row_bytes=128,
)
def vmi_tcvt(src: pto.Tile, dst: pto.Tile):
    emit_convert_vmi(src, dst)
