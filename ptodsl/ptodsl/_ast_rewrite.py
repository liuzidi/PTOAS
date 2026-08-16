# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Source-to-source AST rewrite for ``@pto.jit(ast_rewrite=True)``."""

from __future__ import annotations

import ast
import copy
import inspect
import textwrap
from dataclasses import dataclass


class PTODSLAstRewriteError(SyntaxError):
    """Raised when AST rewrite sees unsupported Python control flow."""


def rewrite_jit_function(fn):
    """Return a function whose Python if/for control flow lowers to PTODSL APIs."""
    try:
        source = inspect.getsource(fn)
    except (OSError, TypeError) as exc:
        # Dynamically-created functions from exec/REPL/notebook contexts may
        # not have retrievable source. Keep existing tracing behavior for those
        # functions instead of making default-on AST rewrite a compatibility
        # break; native Python runtime control flow will still fail during
        # tracing with the usual PTODSL diagnostics.
        _ = exc
        return fn

    tree = ast.parse(textwrap.dedent(source))
    function_def = _find_function_def(tree, fn.__name__)
    if function_def is None:
        return fn

    function_def.decorator_list = []
    closure_vars = inspect.getclosurevars(fn)
    static_env = dict(fn.__globals__)
    static_env.update(closure_vars.nonlocals)
    _inject_closure_defaults(function_def, closure_vars.nonlocals)
    _sanitize_signature_for_exec(function_def)
    function_def = _ConditionalExpressionNormalizer().visit(function_def)
    rewriter = _ControlFlowRewriter(static_env)
    function_def.body = rewriter.rewrite_block(function_def.body, live_after=set())
    tree = ast.Module(body=[function_def], type_ignores=[])
    ast.fix_missing_locations(tree)

    locals_ns = {}
    try:
        source_file = inspect.getsourcefile(fn)
    except (OSError, TypeError):
        source_file = None
    code = compile(tree, source_file or "<ptodsl-ast-rewrite>", "exec")
    globals_ns = fn.__globals__
    restored_globals = _temporarily_bind_globals(globals_ns, closure_vars.nonlocals)
    try:
        exec(code, globals_ns, locals_ns)
    finally:
        _restore_globals(globals_ns, restored_globals)
    rewritten = locals_ns[function_def.name]
    rewritten.__defaults__ = fn.__defaults__
    rewritten_kwdefaults = dict(rewritten.__kwdefaults__ or {})
    rewritten_kwdefaults.update(closure_vars.nonlocals)
    if fn.__kwdefaults__:
        rewritten_kwdefaults.update(fn.__kwdefaults__)
    rewritten.__kwdefaults__ = rewritten_kwdefaults
    rewritten.__annotations__ = dict(getattr(fn, "__annotations__", {}))
    rewritten.__doc__ = fn.__doc__
    rewritten.__module__ = fn.__module__
    rewritten.__qualname__ = fn.__qualname__
    return rewritten


def _find_function_def(tree, name: str):
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    if len(matches) != 1:
        return None
    return matches[0]


_MISSING_GLOBAL = object()


def _temporarily_bind_globals(globals_ns, bindings):
    restored = {}
    for name, value in bindings.items():
        restored[name] = globals_ns.get(name, _MISSING_GLOBAL)
        globals_ns[name] = value
    return restored


def _restore_globals(globals_ns, restored):
    for name, value in restored.items():
        if value is _MISSING_GLOBAL:
            globals_ns.pop(name, None)
        else:
            globals_ns[name] = value


def _inject_closure_defaults(function_def, closure_bindings):
    if not closure_bindings:
        return
    existing = {
        arg.arg
        for arg in (
            list(function_def.args.posonlyargs)
            + list(function_def.args.args)
            + list(function_def.args.kwonlyargs)
        )
    }
    if function_def.args.vararg is not None:
        existing.add(function_def.args.vararg.arg)
    if function_def.args.kwarg is not None:
        existing.add(function_def.args.kwarg.arg)

    for name in closure_bindings:
        if name in existing:
            continue
        function_def.args.kwonlyargs.append(ast.arg(arg=name))
        function_def.args.kw_defaults.append(ast.Constant(None))


def _sanitize_signature_for_exec(function_def):
    args = function_def.args
    args.defaults = [ast.Constant(None) for _ in args.defaults]
    args.kw_defaults = [
        ast.Constant(None) if default is not None else None
        for default in args.kw_defaults
    ]
    for arg in (
        list(args.posonlyargs)
        + list(args.args)
        + list(args.kwonlyargs)
    ):
        arg.annotation = None
    if args.vararg is not None:
        args.vararg.annotation = None
    if args.kwarg is not None:
        args.kwarg.annotation = None
    function_def.returns = None


def _is_normalizable_ifexp_assign_target(node) -> bool:
    return isinstance(node, ast.Name)


class _ConditionalExpressionNormalizer(ast.NodeTransformer):
    """Normalize assign-form ``IfExp`` into statement ``if`` before rewrite."""

    def visit_Assign(self, node):
        node = self.generic_visit(node)
        if not isinstance(node.value, ast.IfExp):
            return node
        if not node.targets or not all(_is_normalizable_ifexp_assign_target(target) for target in node.targets):
            return node
        return self._normalize_ifexp_assignment(node, node.value)

    def visit_AnnAssign(self, node):
        node = self.generic_visit(node)
        if node.value is None or not isinstance(node.value, ast.IfExp):
            return node
        if not _is_normalizable_ifexp_assign_target(node.target):
            return node
        return self._normalize_ifexp_assignment(node, node.value)

    def _normalize_ifexp_assignment(self, stmt, value):
        then_stmt = copy.deepcopy(stmt)
        then_stmt.value = value.body
        else_stmt = copy.deepcopy(stmt)
        else_stmt.value = value.orelse
        if_stmt = ast.If(
            test=value.test,
            body=[then_stmt],
            orelse=[else_stmt],
        )
        return ast.copy_location(self.generic_visit(if_stmt), stmt)


@dataclass(frozen=True)
class _NameInfo:
    loads: set[str]
    stores: set[str]


@dataclass(frozen=True, order=True)
class _SubscriptSlot:
    base: str
    index: int

    @property
    def display(self) -> str:
        return f"{self.base}[{self.index}]"


@dataclass(frozen=True)
class _SlotInfo:
    loads: set[_SubscriptSlot]
    stores: set[_SubscriptSlot]
    invalid_stores: tuple[str, ...] = ()


class _NameInfoVisitor(ast.NodeVisitor):
    def __init__(self):
        self.loads = set()
        self.stores = set()

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            self.loads.add(node.id)
        elif isinstance(node.ctx, (ast.Store, ast.Del)):
            self.stores.add(node.id)

    def visit_AugAssign(self, node):
        self._visit_augassign_target_load(node.target)
        self.visit(node.value)
        self.visit(node.target)

    def _visit_augassign_target_load(self, node):
        if isinstance(node, ast.Name):
            self.loads.add(node.id)
            return
        if isinstance(node, (ast.Attribute, ast.Subscript)):
            self.visit(node.value)
            return
        self.visit(node)

    def visit_FunctionDef(self, node):
        self.stores.add(node.name)
        for decorator in node.decorator_list:
            self.visit(decorator)
        self._visit_arguments_defaults(node.args)
        self.loads.update(_function_free_vars(node))

    def visit_AsyncFunctionDef(self, node):
        self.visit_FunctionDef(node)

    def visit_Lambda(self, node):
        self._visit_arguments_defaults(node.args)
        self.loads.update(_lambda_free_vars(node))

    def visit_ClassDef(self, node):
        self.stores.add(node.name)
        for decorator in node.decorator_list:
            self.visit(decorator)
        for base in node.bases:
            self.visit(base)
        for keyword in node.keywords:
            self.visit(keyword.value)
        self.loads.update(_class_body_free_vars(node))

    def visit_ListComp(self, node):
        self._visit_comprehension(node.generators, (node.elt,))

    def visit_SetComp(self, node):
        self._visit_comprehension(node.generators, (node.elt,))

    def visit_GeneratorExp(self, node):
        self._visit_comprehension(node.generators, (node.elt,))

    def visit_DictComp(self, node):
        self._visit_comprehension(node.generators, (node.key, node.value))

    def _visit_arguments_defaults(self, args):
        for default in args.defaults:
            self.visit(default)
        for default in args.kw_defaults:
            if default is not None:
                self.visit(default)

    def _visit_comprehension(self, generators, result_nodes):
        bound = set()
        for generator in generators:
            self._visit_comprehension_expr(generator.iter, bound)
            bound |= _target_stores(generator.target)
            for if_node in generator.ifs:
                self._visit_comprehension_expr(if_node, bound)
        for result_node in result_nodes:
            self._visit_comprehension_expr(result_node, bound)

    def _visit_comprehension_expr(self, node, bound):
        info = _name_info(node)
        self.loads.update(info.loads - set(bound))
        self.stores.update(info.stores - set(bound))


def _name_info(node) -> _NameInfo:
    visitor = _NameInfoVisitor()
    if isinstance(node, list):
        for item in node:
            visitor.visit(item)
    else:
        visitor.visit(node)
    return _NameInfo(visitor.loads, visitor.stores)


class _SlotInfoVisitor(ast.NodeVisitor):
    def __init__(self, static_env, static_iters=None):
        self._static_env = static_env
        self._static_iters = dict(static_iters or {})
        self.loads = set()
        self.stores = set()
        self.invalid_stores = []

    def visit_Subscript(self, node):
        if isinstance(node.ctx, ast.Load):
            self.loads.update(_resolve_subscript_slots(node, self._static_iters, require_static=False))
            return
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            slots = _resolve_subscript_slots(node, self._static_iters, require_static=True)
            if slots:
                self.stores.update(slots)
            else:
                self.invalid_stores.append(_unsupported_subscript_store_message(node))
            return
        self.generic_visit(node)

    def visit_AugAssign(self, node):
        if isinstance(node.target, ast.Subscript):
            slots = _resolve_subscript_slots(node.target, self._static_iters, require_static=True)
            if slots:
                self.loads.update(slots)
                self.stores.update(slots)
            else:
                self.invalid_stores.append(_unsupported_subscript_store_message(node.target))
        else:
            self.visit(node.target)
        self.visit(node.value)

    def visit_For(self, node):
        if _is_pto_attr_call(node.iter, "static_range") and isinstance(node.target, ast.Name):
            values = _try_eval_static_range(node.iter, self._static_env)
            if values is None:
                for stmt in node.body:
                    self.visit(stmt)
                for stmt in node.orelse:
                    self.visit(stmt)
                return
            old = self._static_iters.get(node.target.id)
            self._static_iters[node.target.id] = values
            try:
                for stmt in node.body:
                    self.visit(stmt)
            finally:
                if old is None:
                    self._static_iters.pop(node.target.id, None)
                else:
                    self._static_iters[node.target.id] = old
            for stmt in node.orelse:
                self.visit(stmt)
            return
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        return

    def visit_AsyncFunctionDef(self, node):
        return

    def visit_Lambda(self, node):
        return

    def visit_ClassDef(self, node):
        return


def _slot_info(node, static_env, static_iters=None) -> _SlotInfo:
    visitor = _SlotInfoVisitor(static_env, static_iters)
    if isinstance(node, list):
        for item in node:
            visitor.visit(item)
    else:
        visitor.visit(node)
    return _SlotInfo(visitor.loads, visitor.stores, tuple(visitor.invalid_stores))


def _slot_live_before_block(stmts, live_after, static_env, static_iters=None) -> set[_SubscriptSlot]:
    live = set(live_after)
    for stmt in reversed(stmts):
        live = _slot_live_before_stmt(stmt, live, static_env, static_iters or {})
    return live


def _slot_live_before_stmt(stmt, live_after, static_env, static_iters) -> set[_SubscriptSlot]:
    if isinstance(stmt, ast.If):
        test_info = _slot_info(stmt.test, static_env, static_iters)
        return (
            set(test_info.loads)
            | _slot_live_before_block(stmt.body, live_after, static_env, static_iters)
            | _slot_live_before_block(stmt.orelse, live_after, static_env, static_iters)
        )
    if isinstance(stmt, ast.For):
        if _is_pto_attr_call(stmt.iter, "static_range") and isinstance(stmt.target, ast.Name):
            values = _try_eval_static_range(stmt.iter, static_env)
            if values is not None:
                next_static_iters = dict(static_iters)
                next_static_iters[stmt.target.id] = values
                return (
                    _slot_live_before_block(stmt.body, live_after, static_env, next_static_iters)
                    | _slot_live_before_block(stmt.orelse, live_after, static_env, static_iters)
                )
        iter_info = _slot_info(stmt.iter, static_env, static_iters)
        body_info = _slot_info(stmt.body, static_env, static_iters)
        orelse_info = _slot_info(stmt.orelse, static_env, static_iters)
        assigned = body_info.stores | orelse_info.stores
        return (
            (set(live_after) - assigned)
            | set(iter_info.loads)
            | _slot_live_before_block(stmt.body, set(), static_env, static_iters)
            | _slot_live_before_block(stmt.orelse, set(), static_env, static_iters)
        )
    info = _slot_info(stmt, static_env, static_iters)
    live = _kill_slots_for_assigned_bases(live_after, stmt)
    return (set(live) - info.stores) | info.loads


def _read_before_assignment_slots(stmts, static_env, static_iters=None) -> set[_SubscriptSlot]:
    return _slot_live_before_block(stmts, set(), static_env, static_iters)


def _kill_slots_for_assigned_bases(slots, stmt) -> set[_SubscriptSlot]:
    assigned_bases = _assigned_name_targets(stmt)
    if not assigned_bases:
        return set(slots)
    return {
        slot
        for slot in slots
        if slot.base not in assigned_bases
    }


def _assigned_name_targets(stmt) -> set[str]:
    if isinstance(stmt, ast.Assign):
        names = set()
        for target in stmt.targets:
            names.update(_simple_name_targets(target))
        return names
    if isinstance(stmt, ast.AnnAssign):
        return _simple_name_targets(stmt.target)
    if isinstance(stmt, (ast.For, ast.AsyncFor)):
        return _simple_name_targets(stmt.target)
    if isinstance(stmt, (ast.With, ast.AsyncWith)):
        names = set()
        for item in stmt.items:
            if item.optional_vars is not None:
                names.update(_simple_name_targets(item.optional_vars))
        return names
    return set()


def _simple_name_targets(target) -> set[str]:
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        names = set()
        for elt in target.elts:
            names.update(_simple_name_targets(elt))
        return names
    return set()


def _resolve_subscript_slots(node, static_iters, *, require_static) -> set[_SubscriptSlot]:
    if not isinstance(node.value, ast.Name):
        return set()
    index_values = _static_index_values(node.slice, static_iters)
    if index_values is None:
        return set()
    return {
        _SubscriptSlot(node.value.id, index)
        for index in index_values
    }


def _static_index_values(node, static_iters):
    if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
        return (node.value,)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        values = _static_index_values(node.operand, static_iters)
        if values is not None and len(values) == 1:
            return (-values[0],)
    if isinstance(node, ast.Name) and node.id in static_iters:
        return tuple(static_iters[node.id])
    return None


def _unsupported_subscript_store_message(node) -> str:
    try:
        text = ast.unparse(node)
    except Exception:
        text = "<subscript>"
    return (
        "ast_rewrite=True only supports static subscript carry stores of the form "
        f"simple_name[static_int_or_static_range_iv]; got {text!r}"
    )


def _try_eval_static_range(call, static_env):
    if not _is_pto_attr_call(call, "static_range") or call.keywords:
        return None
    try:
        values = [_eval_static_int(arg, static_env) for arg in call.args]
    except PTODSLAstRewriteError:
        return None
    if len(values) == 1:
        return tuple(range(values[0]))
    if len(values) == 2:
        return tuple(range(values[0], values[1]))
    if len(values) == 3:
        return tuple(range(values[0], values[1], values[2]))
    return None


def _eval_static_int(node, static_env) -> int:
    if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.Name):
        value = static_env.get(node.id, _MISSING_GLOBAL)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        raise PTODSLAstRewriteError("static value is not an integer")
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd):
        return +_eval_static_int(node.operand, static_env)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_eval_static_int(node.operand, static_env)
    if isinstance(node, ast.BinOp):
        lhs = _eval_static_int(node.left, static_env)
        rhs = _eval_static_int(node.right, static_env)
        if isinstance(node.op, ast.Add):
            return lhs + rhs
        if isinstance(node.op, ast.Sub):
            return lhs - rhs
        if isinstance(node.op, ast.Mult):
            return lhs * rhs
        if isinstance(node.op, ast.FloorDiv):
            return lhs // rhs
        if isinstance(node.op, ast.Mod):
            return lhs % rhs
    raise PTODSLAstRewriteError("unsupported static integer expression")


class _ScopeBindingVisitor(ast.NodeVisitor):
    def __init__(self):
        self.stores = set()
        self.globals = set()
        self.nonlocals = set()

    def visit_Name(self, node):
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self.stores.add(node.id)

    def visit_FunctionDef(self, node):
        self.stores.add(node.name)

    def visit_AsyncFunctionDef(self, node):
        self.visit_FunctionDef(node)

    def visit_Lambda(self, node):
        return

    def visit_ClassDef(self, node):
        self.stores.add(node.name)

    def visit_ListComp(self, node):
        self._visit_comprehension(node.generators, (node.elt,))

    def visit_SetComp(self, node):
        self._visit_comprehension(node.generators, (node.elt,))

    def visit_GeneratorExp(self, node):
        self._visit_comprehension(node.generators, (node.elt,))

    def visit_DictComp(self, node):
        self._visit_comprehension(node.generators, (node.key, node.value))

    def visit_Global(self, node):
        self.globals.update(node.names)

    def visit_Nonlocal(self, node):
        self.nonlocals.update(node.names)

    def visit_Import(self, node):
        for alias in node.names:
            self.stores.add(alias.asname or alias.name.split(".", 1)[0])

    def visit_ImportFrom(self, node):
        for alias in node.names:
            if alias.name == "*":
                continue
            self.stores.add(alias.asname or alias.name)

    def _visit_comprehension(self, generators, result_nodes):
        for generator in generators:
            self.visit(generator.iter)
            for if_node in generator.ifs:
                self.visit(if_node)
        for result_node in result_nodes:
            self.visit(result_node)


def _argument_names(args) -> set[str]:
    names = {arg.arg for arg in list(args.posonlyargs) + list(args.args) + list(args.kwonlyargs)}
    if args.vararg is not None:
        names.add(args.vararg.arg)
    if args.kwarg is not None:
        names.add(args.kwarg.arg)
    return names


def _local_bindings(stmts) -> tuple[set[str], set[str]]:
    visitor = _ScopeBindingVisitor()
    for stmt in stmts:
        visitor.visit(stmt)
    local_stores = visitor.stores - visitor.globals - visitor.nonlocals
    return local_stores, visitor.globals


def _function_free_vars(node) -> set[str]:
    local_stores, globals_declared = _local_bindings(node.body)
    bound = _argument_names(node.args) | local_stores
    body_info = _name_info(node.body)
    return body_info.loads - bound - globals_declared


def _lambda_free_vars(node) -> set[str]:
    bound = _argument_names(node.args)
    body_info = _name_info(node.body)
    return body_info.loads - bound


def _class_body_free_vars(node) -> set[str]:
    local_stores, globals_declared = _local_bindings(node.body)
    body_info = _name_info(node.body)
    return body_info.loads - local_stores - globals_declared


def _target_stores(node) -> set[str]:
    return _name_info(node).stores


def _live_before_block(stmts, live_after) -> set[str]:
    live = set(live_after)
    for stmt in reversed(stmts):
        live = _live_before_stmt(stmt, live)
    return live


def _live_before_stmt(stmt, live_after) -> set[str]:
    if isinstance(stmt, ast.If):
        test_info = _name_info(stmt.test)
        return (
            set(test_info.loads)
            | _live_before_block(stmt.body, live_after)
            | _live_before_block(stmt.orelse, live_after)
        )
    if isinstance(stmt, ast.For):
        iter_info = _name_info(stmt.iter)
        target_stores = _target_stores(stmt.target)
        body_info = _name_info(stmt.body)
        orelse_info = _name_info(stmt.orelse)
        assigned = target_stores | body_info.stores | orelse_info.stores
        return (
            (set(live_after) - assigned)
            | set(iter_info.loads)
            | (_live_before_block(stmt.body, set()) - target_stores)
            | _live_before_block(stmt.orelse, set())
        )
    info = _name_info(stmt)
    return (set(live_after) - info.stores) | info.loads


def _read_before_assignment_names(stmts):
    return _live_before_block(stmts, set())


def _is_pto_attr_call(node, name: str) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == name
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "pto"
    )


def _is_range_call(node) -> bool:
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "range"


def _range_triplet(call):
    if not _is_range_call(call):
        raise PTODSLAstRewriteError("ast_rewrite=True only rewrites for-loops over range(...)")
    if call.keywords:
        raise PTODSLAstRewriteError("ast_rewrite=True range(...) loops do not support keyword arguments")
    args = call.args
    if len(args) == 1:
        return ast.Constant(0), args[0], ast.Constant(1)
    if len(args) == 2:
        return args[0], args[1], ast.Constant(1)
    if len(args) == 3:
        return args[0], args[1], args[2]
    raise PTODSLAstRewriteError("ast_rewrite=True range(...) loops require 1 to 3 arguments")


def _pto_attr(name: str, ctx=ast.Load()):
    return ast.Attribute(value=ast.Name(id="pto", ctx=ast.Load()), attr=name, ctx=ctx)


def _name(name: str, ctx=ast.Load()):
    return ast.Name(id=name, ctx=ctx)


def _slot_subscript(slot: _SubscriptSlot, ctx=ast.Load()):
    return ast.Subscript(
        value=_name(slot.base),
        slice=ast.Constant(slot.index),
        ctx=ctx,
    )


def _map_subscript(map_name: str, index: int, ctx=ast.Load()):
    return ast.Subscript(
        value=_name(map_name),
        slice=ast.Constant(index),
        ctx=ctx,
    )


class _SlotCarryRewriter(ast.NodeTransformer):
    def __init__(self, slot_maps, static_env, static_iters=None):
        self._slot_maps = slot_maps
        self._static_env = static_env
        self._static_iters = dict(static_iters or {})

    def visit_For(self, node):
        if _is_pto_attr_call(node.iter, "static_range") and isinstance(node.target, ast.Name):
            values = _try_eval_static_range(node.iter, self._static_env)
            old = self._static_iters.get(node.target.id)
            if values is not None:
                self._static_iters[node.target.id] = values
            try:
                node.body = [self.visit(stmt) for stmt in node.body]
            finally:
                if values is not None:
                    if old is None:
                        self._static_iters.pop(node.target.id, None)
                    else:
                        self._static_iters[node.target.id] = old
            node.orelse = [self.visit(stmt) for stmt in node.orelse]
            return node
        return self.generic_visit(node)

    def visit_Subscript(self, node):
        slots = _resolve_subscript_slots(node, self._static_iters, require_static=False)
        if slots and len({slot.base for slot in slots}) == 1:
            base = next(iter(slots)).base
            if base in self._slot_maps and slots <= set(self._slot_maps[base]["slots"]):
                return ast.copy_location(
                    ast.Subscript(
                        value=_name(self._slot_maps[base]["map_name"]),
                        slice=copy.deepcopy(node.slice),
                        ctx=node.ctx,
                    ),
                    node,
                )
        return self.generic_visit(node)


class _ControlFlowRewriter:
    def __init__(self, static_env=None):
        self._static_env = dict(static_env or {})
        self._counter = 0

    def _fresh(self, prefix: str) -> str:
        value = f"__pto_ast_{prefix}_{self._counter}"
        self._counter += 1
        return value

    def rewrite_block(self, stmts, *, live_after, live_after_slots=None, allow_loop_control=False, static_iters=None):
        rewritten_reversed = []
        live = set(live_after)
        live_slots = set(live_after_slots or ())
        static_iters = dict(static_iters or {})
        for stmt in reversed(stmts):
            # Compute liveness from the authored AST before rewrite_stmt mutates
            # sibling statements in-place, otherwise later rewrites can pollute
            # earlier live-after analysis.
            live_before = _live_before_stmt(stmt, live)
            live_before_slots = _slot_live_before_stmt(stmt, live_slots, self._static_env, static_iters)
            rewritten = self.rewrite_stmt(
                stmt,
                live_after=live,
                live_after_slots=live_slots,
                allow_loop_control=allow_loop_control,
                static_iters=static_iters,
            )
            rewritten_reversed[:0] = rewritten
            live = live_before
            live_slots = live_before_slots
        return rewritten_reversed

    def rewrite_stmt(self, stmt, *, live_after, live_after_slots=None, allow_loop_control=False, static_iters=None):
        live_after_slots = set(live_after_slots or ())
        static_iters = dict(static_iters or {})
        if isinstance(stmt, ast.If):
            return self._rewrite_if(
                stmt,
                live_after=live_after,
                live_after_slots=live_after_slots,
                allow_loop_control=allow_loop_control,
                static_iters=static_iters,
            )
        if isinstance(stmt, ast.For):
            return self._rewrite_for(
                stmt,
                live_after=live_after,
                live_after_slots=live_after_slots,
                allow_loop_control=allow_loop_control,
                static_iters=static_iters,
            )
        if isinstance(stmt, (ast.Break, ast.Continue)):
            if allow_loop_control:
                return [stmt]
            raise PTODSLAstRewriteError("ast_rewrite=True does not support break/continue in rewritten control flow")
        return [
            self._rewrite_nested(
                stmt,
                live_after=live_after,
                live_after_slots=live_after_slots,
                allow_loop_control=allow_loop_control,
                static_iters=static_iters,
            )
        ]

    def _rewrite_nested(self, stmt, *, live_after, live_after_slots=None, allow_loop_control=False, static_iters=None):
        live_after_slots = set(live_after_slots or ())
        static_iters = dict(static_iters or {})
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            stmt.body = self.rewrite_block(
                stmt.body,
                live_after=set(),
                live_after_slots=set(),
                allow_loop_control=False,
                static_iters={},
            )
            return stmt
        if isinstance(stmt, (ast.Lambda, ast.ClassDef)):
            return stmt
        for field, value in ast.iter_fields(stmt):
            if field in {"body", "orelse", "finalbody"} and isinstance(value, list):
                setattr(
                    stmt,
                    field,
                    self.rewrite_block(
                        value,
                        live_after=live_after,
                        live_after_slots=live_after_slots,
                        allow_loop_control=allow_loop_control,
                        static_iters=static_iters,
                    ),
                )
            elif isinstance(value, ast.AST):
                self._rewrite_nested(
                    value,
                    live_after=live_after,
                    live_after_slots=live_after_slots,
                    allow_loop_control=allow_loop_control,
                    static_iters=static_iters,
                )
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, ast.AST):
                        self._rewrite_nested(
                            item,
                            live_after=live_after,
                            live_after_slots=live_after_slots,
                            allow_loop_control=allow_loop_control,
                            static_iters=static_iters,
                        )
        return stmt

    def _rewrite_if(self, stmt, *, live_after, live_after_slots=None, allow_loop_control=False, static_iters=None):
        live_after_slots = set(live_after_slots or ())
        static_iters = dict(static_iters or {})
        if _is_pto_attr_call(stmt.test, "const_expr"):
            stmt.body = self.rewrite_block(
                stmt.body,
                live_after=live_after,
                live_after_slots=live_after_slots,
                allow_loop_control=allow_loop_control,
                static_iters=static_iters,
            )
            stmt.orelse = self.rewrite_block(
                stmt.orelse,
                live_after=live_after,
                live_after_slots=live_after_slots,
                allow_loop_control=allow_loop_control,
                static_iters=static_iters,
            )
            return [stmt]

        cond_name = self._fresh("cond")
        then_info = _name_info(stmt.body)
        else_info = _name_info(stmt.orelse)
        assigned_slots = (
            _slot_info(stmt.body, self._static_env, static_iters).stores
            | _slot_info(stmt.orelse, self._static_env, static_iters).stores
        )
        if live_after_slots & assigned_slots:
            slots = ", ".join(slot.display for slot in sorted(live_after_slots & assigned_slots))
            raise PTODSLAstRewriteError(
                "ast_rewrite=True does not support automatic branch merges for static subscript slots yet; "
                f"rewrite {slots} with explicit scalar temporaries"
            )
        assigned_any = then_info.stores | else_info.stores
        merge_names = tuple(sorted(live_after & assigned_any))
        old_value_names = {
            name: self._fresh(f"old_{name}")
            for name in merge_names
            if name not in then_info.stores or name not in else_info.stores
        }

        branch_live_after = set(live_after) | set(merge_names)
        then_body = self.rewrite_block(
            stmt.body,
            live_after=branch_live_after,
            live_after_slots=live_after_slots,
            allow_loop_control=False,
            static_iters=static_iters,
        )
        else_body = self.rewrite_block(
            stmt.orelse,
            live_after=branch_live_after,
            live_after_slots=live_after_slots,
            allow_loop_control=False,
            static_iters=static_iters,
        )
        trace_time_if = ast.If(
            test=_name(cond_name),
            body=copy.deepcopy(then_body) or [ast.Pass()],
            orelse=copy.deepcopy(else_body) or [ast.Pass()],
        )
        branch_name = self._fresh("br")

        dynamic_then_body = copy.deepcopy(then_body)
        dynamic_else_body = copy.deepcopy(else_body)
        if merge_names:
            dynamic_then_body.append(
                self._branch_assign(
                    branch_name,
                    merge_names,
                    old_value_names=old_value_names,
                    assigned_names=then_info.stores,
                )
            )
            dynamic_else_body.append(
                self._branch_assign(
                    branch_name,
                    merge_names,
                    old_value_names=old_value_names,
                    assigned_names=else_info.stores,
                )
            )

        with_stmt = ast.With(
            items=[
                ast.withitem(
                    context_expr=ast.Call(func=_pto_attr("if_"), args=[_name(cond_name)], keywords=[]),
                    optional_vars=_name(branch_name, ast.Store()),
                )
            ],
            body=[
                ast.With(
                    items=[
                        ast.withitem(
                            context_expr=ast.Attribute(
                                value=_name(branch_name),
                                attr="then_",
                                ctx=ast.Load(),
                            ),
                            optional_vars=None,
                        )
                    ],
                    body=dynamic_then_body or [ast.Pass()],
                    type_comment=None,
                )
            ],
            type_comment=None,
        )
        if stmt.orelse or dynamic_else_body:
            with_stmt.body.append(
                ast.With(
                    items=[
                        ast.withitem(
                            context_expr=ast.Attribute(
                                value=_name(branch_name),
                                attr="else_",
                                ctx=ast.Load(),
                            ),
                            optional_vars=None,
                        )
                    ],
                    body=dynamic_else_body or [ast.Pass()],
                    type_comment=None,
                )
            )
        dynamic_body = [with_stmt]
        dynamic_body.extend(
            ast.Assign(
                targets=[_name(name, ast.Store())],
                value=ast.Attribute(value=_name(branch_name), attr=name, ctx=ast.Load()),
            )
            for name in merge_names
        )

        result = [
            ast.Assign(
                targets=[_name(cond_name, ast.Store())],
                value=stmt.test,
            )
        ]
        result.extend(
            ast.Assign(
                targets=[_name(old_name, ast.Store())],
                value=_name(name),
            )
            for name, old_name in old_value_names.items()
        )
        result.append(
            ast.copy_location(
                ast.If(
                    test=ast.Call(
                        func=_name("isinstance"),
                        args=[_name(cond_name), _name("bool")],
                        keywords=[],
                    ),
                    body=[trace_time_if],
                    orelse=dynamic_body,
                ),
                stmt,
            )
        )
        return result

    def _branch_assign(self, branch_name, names, *, old_value_names, assigned_names):
        return ast.Expr(
            value=ast.Call(
                func=ast.Attribute(value=_name(branch_name), attr="assign", ctx=ast.Load()),
                args=[],
                keywords=[
                    ast.keyword(
                        arg=name,
                        value=_name(name if name in assigned_names else old_value_names[name]),
                    )
                    for name in names
                ],
            )
        )

    def _rewrite_for(self, stmt, *, live_after, live_after_slots=None, allow_loop_control=False, static_iters=None):
        live_after_slots = set(live_after_slots or ())
        static_iters = dict(static_iters or {})
        if _is_pto_attr_call(stmt.iter, "static_range"):
            next_static_iters = dict(static_iters)
            if isinstance(stmt.target, ast.Name):
                values = _try_eval_static_range(stmt.iter, self._static_env)
                if values is not None:
                    next_static_iters[stmt.target.id] = values
            stmt.body = self.rewrite_block(
                stmt.body,
                live_after=live_after,
                live_after_slots=live_after_slots,
                allow_loop_control=True,
                static_iters=next_static_iters,
            )
            stmt.orelse = self.rewrite_block(
                stmt.orelse,
                live_after=live_after,
                live_after_slots=live_after_slots,
                allow_loop_control=allow_loop_control,
                static_iters=static_iters,
            )
            return [stmt]

        if stmt.orelse:
            raise PTODSLAstRewriteError("ast_rewrite=True does not support for-else on runtime loops")
        if not isinstance(stmt.target, ast.Name):
            raise PTODSLAstRewriteError("ast_rewrite=True runtime for-loops require a simple name target")
        if stmt.target.id in live_after:
            raise PTODSLAstRewriteError(
                "ast_rewrite=True runtime for-loops cannot expose the loop induction variable outside the loop yet; "
                f"use explicit pto.for_(...) for {stmt.target.id!r}"
            )

        start, stop, step = _range_triplet(stmt.iter)
        body_info = _name_info(stmt.body)
        body_slot_info = _slot_info(stmt.body, self._static_env, static_iters)
        if body_slot_info.invalid_stores:
            raise PTODSLAstRewriteError(body_slot_info.invalid_stores[0])
        reads_before = _read_before_assignment_names(stmt.body)
        slot_reads_before = _read_before_assignment_slots(stmt.body, self._static_env, static_iters)
        assigned_live_after = body_info.stores & set(live_after)
        assigned_slots_live_after = body_slot_info.stores & set(live_after_slots)
        loop_carried = tuple(sorted(body_info.stores & reads_before))
        loop_carried_slots = tuple(sorted(body_slot_info.stores & slot_reads_before))
        unsupported_last_values = sorted(assigned_live_after - set(loop_carried))
        if unsupported_last_values:
            raise PTODSLAstRewriteError(
                "ast_rewrite=True runtime for-loops cannot expose last-iteration-only values yet; "
                f"use explicit pto.for_(...).carry(...) for {unsupported_last_values}"
            )
        unsupported_last_slots = sorted(assigned_slots_live_after - set(loop_carried_slots))
        if unsupported_last_slots:
            slots = [slot.display for slot in unsupported_last_slots]
            raise PTODSLAstRewriteError(
                "ast_rewrite=True runtime for-loops cannot expose last-iteration-only static subscript values yet; "
                f"use explicit scalar temporaries for {slots}"
            )

        loop_name = self._fresh("loop")
        loop_live_after = set(live_after) | set(loop_carried)
        loop_live_after_slots = set(live_after_slots) | set(loop_carried_slots)
        body = self.rewrite_block(
            stmt.body,
            live_after=loop_live_after,
            live_after_slots=loop_live_after_slots,
            static_iters=static_iters,
        )

        slot_carry_names = {
            slot: self._fresh(f"slot_{slot.base}_{slot.index}")
            for slot in loop_carried_slots
        }
        slot_maps = {}
        for slot in loop_carried_slots:
            slot_maps.setdefault(slot.base, {"map_name": self._fresh(f"slot_{slot.base}_map"), "slots": []})
            slot_maps[slot.base]["slots"].append(slot)
        for data in slot_maps.values():
            data["slots"] = tuple(sorted(data["slots"]))

        if loop_carried or loop_carried_slots:
            slot_initializers = [
                ast.Assign(
                    targets=[_name(slot_carry_names[slot], ast.Store())],
                    value=_slot_subscript(slot),
                )
                for slot in loop_carried_slots
            ]
            setup = ast.Assign(
                targets=[_name(loop_name, ast.Store())],
                value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Call(
                            func=_pto_attr("for_"),
                            args=[start, stop],
                            keywords=[ast.keyword(arg="step", value=step)],
                        ),
                        attr="carry",
                        ctx=ast.Load(),
                    ),
                    args=[],
                    keywords=[
                        ast.keyword(arg=name, value=_name(name))
                        for name in loop_carried
                    ] + [
                        ast.keyword(arg=slot_carry_names[slot], value=_name(slot_carry_names[slot]))
                        for slot in loop_carried_slots
                    ],
                ),
            )
            prologue = [
                ast.Assign(
                    targets=[_name(stmt.target.id, ast.Store())],
                    value=ast.Attribute(value=_name(loop_name), attr="iv", ctx=ast.Load()),
                )
            ]
            prologue.extend(
                ast.Assign(
                    targets=[_name(name, ast.Store())],
                    value=ast.Attribute(value=_name(loop_name), attr=name, ctx=ast.Load()),
                )
                for name in loop_carried
            )
            prologue.extend(
                ast.Assign(
                    targets=[_name(slot_carry_names[slot], ast.Store())],
                    value=ast.Attribute(value=_name(loop_name), attr=slot_carry_names[slot], ctx=ast.Load()),
                )
                for slot in loop_carried_slots
            )
            prologue.extend(
                ast.Assign(
                    targets=[_name(data["map_name"], ast.Store())],
                    value=ast.Dict(
                        keys=[ast.Constant(slot.index) for slot in data["slots"]],
                        values=[_name(slot_carry_names[slot]) for slot in data["slots"]],
                    ),
                )
                for data in slot_maps.values()
            )
            if loop_carried_slots:
                body = [
                    _SlotCarryRewriter(slot_maps, self._static_env, static_iters).visit(stmt)
                    for stmt in body
                ]
            slot_epilogue = [
                ast.Assign(
                    targets=[_name(slot_carry_names[slot], ast.Store())],
                    value=_map_subscript(slot_maps[slot.base]["map_name"], slot.index),
                )
                for slot in loop_carried_slots
            ]
            body = prologue + body + [
                *slot_epilogue,
                ast.Expr(
                    value=ast.Call(
                        func=ast.Attribute(value=_name(loop_name), attr="update", ctx=ast.Load()),
                        args=[],
                        keywords=[
                            ast.keyword(arg=name, value=_name(name))
                            for name in loop_carried
                        ] + [
                            ast.keyword(arg=slot_carry_names[slot], value=_name(slot_carry_names[slot]))
                            for slot in loop_carried_slots
                        ],
                    )
                )
            ]
            with_stmt = ast.With(
                items=[ast.withitem(context_expr=_name(loop_name), optional_vars=None)],
                body=body or [ast.Pass()],
                type_comment=None,
            )
            result = slot_initializers + [ast.copy_location(setup, stmt), ast.copy_location(with_stmt, stmt)]
            for name in loop_carried:
                result.append(
                    ast.Assign(
                        targets=[_name(name, ast.Store())],
                        value=ast.Call(
                            func=ast.Attribute(value=_name(loop_name), attr="final", ctx=ast.Load()),
                            args=[ast.Constant(name)],
                            keywords=[],
                        ),
                    )
                )
            for slot in loop_carried_slots:
                result.append(
                    ast.Assign(
                        targets=[_slot_subscript(slot, ast.Store())],
                        value=ast.Call(
                            func=ast.Attribute(value=_name(loop_name), attr="final", ctx=ast.Load()),
                            args=[ast.Constant(slot_carry_names[slot])],
                            keywords=[],
                        ),
                    )
                )
            return result

        with_stmt = ast.With(
            items=[
                ast.withitem(
                    context_expr=ast.Call(
                        func=_pto_attr("for_"),
                        args=[start, stop],
                        keywords=[ast.keyword(arg="step", value=step)],
                    ),
                    optional_vars=_name(stmt.target.id, ast.Store()),
                )
            ],
            body=body or [ast.Pass()],
            type_comment=None,
        )
        return [ast.copy_location(with_stmt, stmt)]


__all__ = [
    "PTODSLAstRewriteError",
    "rewrite_jit_function",
]
