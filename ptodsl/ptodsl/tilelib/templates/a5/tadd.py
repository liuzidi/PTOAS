# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""PTODSL TileLib template for pto.tadd."""

from ptodsl import pto

from ._common import same_dtype_signatures
from ._elementwise import register_binary


def _vadd(lhs, rhs, mask):
    return pto.vadd(lhs, rhs, mask)


template_tadd = register_binary(
    op="pto.tadd",
    name="template_tadd",
    vector_op=_vadd,
    dtypes=same_dtype_signatures(3),
)


from ._vmi_common import (  # noqa: E402
    _add as _vmi_add,
    canonical_vmi_template,
    emit_elementwise_vmi,
    sinkhorn_compact_elementwise_vmi_constraint,
)


@canonical_vmi_template(
    target="a5",
    op="tadd",
    name="vmi_tadd_block64",
    dtypes=(("f32", "f32", "f32"),),
    min_row_bytes=128,
)
def vmi_tadd_block64(src0: pto.Tile, src1: pto.Tile, dst: pto.Tile):
    emit_elementwise_vmi(dst, (src0, src1), _vmi_add)


@canonical_vmi_template(
    target="a5",
    op="tadd",
    name="vmi_tadd_sinkhorn_compact",
    dtypes=(("f32", "f32", "f32"),),
    constraints=(sinkhorn_compact_elementwise_vmi_constraint,),
    requires_full_physical_row=False,
    tags=("supports_partial_valid_shape",),
)
def vmi_tadd_sinkhorn_compact(src0: pto.Tile, src1: pto.Tile, dst: pto.Tile):
    emit_elementwise_vmi(dst, (src0, src1), _vmi_add)
