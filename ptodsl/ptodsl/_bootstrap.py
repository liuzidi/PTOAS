# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
MLIR path bootstrap and context factory.

Discovers local LLVM MLIR Python bindings plus PTO Python dialect artifacts so
that ``ptodsl`` can import ``mlir`` / ``mlir.dialects.pto`` directly from a
developer workspace without requiring the caller to pre-seed ``PYTHONPATH``.
When the current interpreter already has a complete installed MLIR/PTO stack,
leave ``sys.path`` untouched so wheel installs do not accidentally mix in a
developer checkout from environment variables or nearby directories.
"""

import importlib.util
import os
import sys
from pathlib import Path


def _path_has(path: Path, relative: str) -> bool:
    try:
        return (path / relative).exists()
    except OSError:
        return False


def _pythonpath_already_provides_mlir_and_pto() -> bool:
    """Return true when the caller has already configured compatible MLIR/PTO roots.

    Lit preserves ``PYTHONPATH`` but not the convenience environment variables
    used by ``ptoas_env.sh``.  In that case we should not prepend repository
    default fallbacks like ``build/python`` / ``install`` because they may belong
    to an older build tree and can mix Python/native MLIR bindings.
    """
    has_mlir_core = False
    has_pto_dialect = False
    for entry in sys.path:
        if not entry:
            continue
        path = Path(entry)
        has_mlir_core = has_mlir_core or _path_has(path, "mlir/ir.py")
        has_pto_dialect = has_pto_dialect or _path_has(
            path, "mlir/dialects/pto.py"
        )
        if has_mlir_core and has_pto_dialect:
            return True
    return False


def _candidate_python_roots() -> list[Path]:
    here = Path(__file__).resolve()
    repo_root = here.parents[2]
    owner_root = repo_root.parent
    github_root = owner_root.parent
    env_roots = []
    has_explicit_pto_root = False
    for env_name in (
        "MLIR_PYTHON_ROOT",
        "PTO_PYTHON_BUILD_ROOT",
        "PTO_PYTHON_ROOT",
        "PTO_INSTALL_DIR",
    ):
        raw = os.environ.get(env_name)
        if raw:
            env_roots.append(Path(raw))
            if env_name != "MLIR_PYTHON_ROOT":
                has_explicit_pto_root = True

    if not env_roots and _pythonpath_already_provides_mlir_and_pto():
        return []

    fallback_roots = [
        github_root / "llvm" / "llvm-project" / "build" / "tools" / "mlir" / "python_packages" / "mlir_core",
        github_root / "llvm" / "llvm-project" / "install" / "python_packages" / "mlir_core",
        github_root / "llvm" / "llvm-project" / "build-shared" / "tools" / "mlir" / "python_packages" / "mlir_core",
    ]
    if not has_explicit_pto_root:
        fallback_roots = [
            repo_root / "build" / "python",
            repo_root / "install",
            *fallback_roots,
        ]

    return [
        *env_roots,
        *fallback_roots,
    ]


def _bootstrap_python_paths() -> None:
    ordered_roots: list[str] = []
    seen = set()
    for root in _candidate_python_roots():
        if not root or not root.is_dir():
            continue
        if not (root / "mlir").exists():
            continue
        root_text = str(root)
        if root_text in seen:
            continue
        ordered_roots.append(root_text)
        seen.add(root_text)
    for root_text in reversed(ordered_roots):
        if root_text in sys.path:
            sys.path.remove(root_text)
        sys.path.insert(0, root_text)


def _can_import_active_python_mlir() -> bool:
    required_modules = ("mlir.ir", "mlir.dialects.pto")
    for module_name in required_modules:
        try:
            if importlib.util.find_spec(module_name) is None:
                return False
        except (ImportError, ValueError, ModuleNotFoundError):
            return False
    return True


if not _can_import_active_python_mlir():
    _bootstrap_python_paths()

from mlir.dialects import pto as _pto_dialect  # noqa: E402
try:
    from mlir.dialects import llvm as _llvm_dialect  # noqa: E402
except Exception:  # pragma: no cover - depends on the installed MLIR package.
    _llvm_dialect = None
from mlir.ir import Context, Location           # noqa: E402


def make_context() -> Context:
    """Create a fresh MLIR Context with the PTO dialect loaded."""
    ctx = Context()
    _pto_dialect.register_dialect(ctx, load=True)
    if _llvm_dialect is not None and hasattr(_llvm_dialect, "register_dialect"):
        _llvm_dialect.register_dialect(ctx, load=True)
    return ctx


__all__ = ["make_context"]
