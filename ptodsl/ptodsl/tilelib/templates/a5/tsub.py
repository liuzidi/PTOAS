# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""PTODSL TileLib template for pto.tsub."""

from ptodsl import pto

from ._common import same_dtype_signatures
from ._elementwise import register_binary


def _vsub(lhs, rhs, mask):
    return pto.vsub(lhs, rhs, mask)


template_tsub = register_binary(
    op="pto.tsub",
    name="template_tsub",
    vector_op=_vsub,
    dtypes=same_dtype_signatures(3),
)


from ._vmi_common import (  # noqa: E402
    _sub as _vmi_sub,
    canonical_vmi_template,
    emit_elementwise_vmi,
)


@canonical_vmi_template(
    target="a5",
    op="tsub",
    name="vmi_tsub",
    dtypes=(("f32", "f32", "f32"),),
    min_row_bytes=128,
)
def vmi_tsub(src0: pto.Tile, src1: pto.Tile, dst: pto.Tile):
    emit_elementwise_vmi(dst, (src0, src1), _vmi_sub)
