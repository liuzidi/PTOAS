# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Instantiate a PTODSL VMI TileLib candidate for ``ExpandTileOp``."""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import sys

from ._tile_template_tracing import (
    TileSpec,
    bf16,
    f16,
    f32,
    i8,
    i16,
    i32,
)
from .tilelib.registry import TileTemplateRegistry


_DTYPE_MAP = {
    "f32": f32,
    "f16": f16,
    "bf16": bf16,
    "i32": i32,
    "i16": i16,
    "i8": i8,
}


def _normalize_op_name(op_name: str) -> str:
    return op_name[4:] if op_name.startswith("pto.") else op_name


def _parse_operand_specs(spec_text: str) -> list[dict]:
    try:
        raw_specs = json.loads(spec_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid operand-specs JSON: {exc}") from exc
    if not isinstance(raw_specs, list) or not raw_specs:
        raise ValueError("operand-specs must be a non-empty JSON array")
    return raw_specs


def _parse_context_attrs(spec_text: str | None) -> dict[str, object]:
    if not spec_text:
        return {}
    try:
        attrs = json.loads(spec_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid context-attrs JSON: {exc}") from exc
    if not isinstance(attrs, dict):
        raise ValueError("context-attrs must be a JSON object")
    return attrs


def _validate_candidate_context(op_name: str, attrs: dict[str, object]) -> None:
    if not attrs:
        return
    if op_name in {"texp", "tdivs"} and attrs == {"precisionType": "default"}:
        return
    if op_name == "tcvt" and attrs == {"round_mode": "RINT"}:
        return
    raise ValueError(
        f"initial PTODSL VMI candidate for {op_name!r} does not support context attrs {attrs!r}"
    )


def _parse_dtype(raw: dict, index: int):
    dtype_name = raw.get("dtype")
    dtype = _DTYPE_MAP.get(dtype_name)
    if dtype is None:
        raise ValueError(f"operand-specs[{index}] has unsupported dtype {dtype_name!r}")
    return dtype


def _parse_parameter_spec(raw: dict, index: int):
    if not isinstance(raw, dict):
        raise ValueError(f"operand-specs[{index}] must be an object")
    kind = raw.get("kind")
    if kind == "scalar":
        return _parse_dtype(raw, index)
    if kind != "tile":
        raise ValueError(
            f"operand-specs[{index}] must be a tile or scalar for the PTODSL VMI provider"
        )

    dtype = _parse_dtype(raw, index)
    shape = raw.get("shape")
    if not isinstance(shape, list) or len(shape) != 2:
        raise ValueError(f"operand-specs[{index}] requires a static rank-2 shape")
    try:
        parsed_shape = tuple(int(dim) for dim in shape)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"operand-specs[{index}] shape must contain integers") from exc

    valid_shape = raw.get("valid_shape")
    if valid_shape is not None:
        if not isinstance(valid_shape, list) or len(valid_shape) != 2:
            raise ValueError(f"operand-specs[{index}] valid_shape must be rank-2")
        if any(dim is None for dim in valid_shape):
            raise ValueError(
                "initial PTODSL VMI provider does not support dynamic valid_shape"
            )
        if tuple(int(dim) for dim in valid_shape) != parsed_shape:
            raise ValueError(
                "initial PTODSL VMI provider requires valid_shape to equal physical shape"
            )

    memory_space = raw.get("memory_space", "ub")
    if memory_space != "ub":
        raise ValueError(
            f"initial PTODSL VMI provider supports only UB tiles, got {memory_space!r}"
        )
    b_layout = _parse_tile_config(raw.get("config"), index)
    return TileSpec(parsed_shape, dtype, memory_space="ub", b_layout=b_layout)


def _parse_tile_config(config: object, index: int) -> str:
    if config is None:
        return "row_major"
    if not isinstance(config, dict):
        raise ValueError(f"operand-specs[{index}] config must be an object")
    expected = {
        "s_layout": "none_box",
        "s_fractal_size": 512,
        "pad_value": "0x0",
    }
    for key, expected_value in expected.items():
        value = config.get(key, expected_value)
        if key == "pad_value" and isinstance(value, str):
            value = value.lower()
        if value != expected_value:
            raise ValueError(
                "initial PTODSL VMI provider supports only the default secondary layout; "
                f"operand-specs[{index}] has {key}={config.get(key)!r}"
            )
    b_layout = config.get("b_layout", "row_major")
    if b_layout not in {"row_major", "col_major"}:
        raise ValueError(
            "initial PTODSL VMI provider supports row-major or col-major tiles; "
            f"operand-specs[{index}] has b_layout={b_layout!r}"
        )
    return b_layout


def _find_candidates(module, *, target: str, op_name: str) -> list:
    registry = getattr(module, "VMI_TILELIB_REGISTRY", None)
    if not isinstance(registry, TileTemplateRegistry):
        raise TypeError(
            f"PTODSL VMI provider module {module.__name__!r} must expose "
            "VMI_TILELIB_REGISTRY as a TileTemplateRegistry"
        )
    return registry.lookup(op_name, target)


def instantiate_candidate(
    *,
    target: str,
    op_name: str,
    operand_specs: list[dict],
    provider_module: str,
    context_attrs: dict[str, object] | None = None,
):
    module = importlib.import_module(provider_module)
    normalized_op = _normalize_op_name(op_name)
    _validate_candidate_context(normalized_op, dict(context_attrs or {}))
    candidates = _find_candidates(module, target=target, op_name=normalized_op)
    if not candidates:
        raise LookupError(
            f"no PTODSL VMI candidate for target={target!r}, op={normalized_op!r} "
            f"in module {provider_module!r}"
        )
    if len(candidates) != 1:
        names = ", ".join(candidate.name for candidate in candidates)
        raise LookupError(
            "RFC-mode PTODSL VMI provider requires exactly one canonical "
            f"candidate per (target, op); target={target!r}, op={normalized_op!r}, "
            f"found {len(candidates)} in module {provider_module!r}: {names}"
        )

    candidate = candidates[0]
    parameters = tuple(inspect.signature(candidate.py_fn).parameters)
    if len(parameters) != len(operand_specs):
        raise ValueError(
            f"candidate {candidate.name!r} expects {len(parameters)} operands, "
            f"got {len(operand_specs)}"
        )
    parameter_specs = {
        name: _parse_parameter_spec(raw_spec, index)
        for index, (name, raw_spec) in enumerate(zip(parameters, operand_specs))
    }
    return candidate.specialize(**parameter_specs)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PTODSL VMI TileLib expand helper")
    parser.add_argument("--target", default="a5")
    parser.add_argument("--op", required=True)
    parser.add_argument("--operand-specs", required=True)
    parser.add_argument("--context-attrs")
    parser.add_argument("--provider-module", default="ptodsl.vmi_tilelib")
    args = parser.parse_args(argv)

    try:
        operand_specs = _parse_operand_specs(args.operand_specs)
        context_attrs = _parse_context_attrs(args.context_attrs)
        artifact = instantiate_candidate(
            target=args.target,
            op_name=args.op,
            operand_specs=operand_specs,
            provider_module=args.provider_module,
            context_attrs=context_attrs,
        )
        mlir_text = artifact.mlir_text()
    except Exception as exc:
        print(f"vmi_tilelib_helper: error: {exc}", file=sys.stderr)
        return 1

    sys.stdout.write(mlir_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
