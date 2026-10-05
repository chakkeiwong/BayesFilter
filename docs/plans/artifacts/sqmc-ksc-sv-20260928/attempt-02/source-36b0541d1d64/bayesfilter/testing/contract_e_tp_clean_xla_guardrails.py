"""Static-source and concrete-graph guardrails for Contract E--TP clean XLA.

The source audit does not infer loop semantics from identifier names.  Callers
declare the role of every reachable Python loop; undeclared loops fail closed.
"""

from __future__ import annotations

import ast
import builtins
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


_BUILTINS = frozenset(dir(builtins))
_FUNCTIONAL_LOOP_OPS = frozenset({"While", "StatelessWhile", "Scan"})


@dataclass(frozen=True)
class LoopRole:
    role: str
    dynamic: bool


@dataclass(frozen=True)
class SourceRouteSpec:
    roots: tuple[str, ...]
    loop_roles: Mapping[str, LoopRole]
    required_reachable: tuple[str, ...] = ()


class _FunctionIndex(ast.NodeVisitor):
    def __init__(self) -> None:
        self.functions: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
        self._scope: list[str] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._scope.append(node.name)
        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                self.visit(child)
        self._scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        symbol = ".".join((*self._scope, node.name))
        self.functions[symbol] = node
        self._scope.append(node.name)
        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.visit(child)
        self._scope.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        symbol = ".".join((*self._scope, node.name))
        self.functions[symbol] = node
        self._scope.append(node.name)
        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.visit(child)
        self._scope.pop()


class _BodyVisitor(ast.NodeVisitor):
    """Inspect one function body without descending into nested definitions."""

    def __init__(self) -> None:
        self.calls: list[ast.Call] = []
        self.loops: list[ast.For | ast.AsyncFor | ast.While] = []
        self.nested_names: set[str] = set()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.nested_names.add(node.name)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.nested_names.add(node.name)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        return

    def visit_Call(self, node: ast.Call) -> None:
        self.calls.append(node)
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        self.loops.append(node)
        self.generic_visit(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self.loops.append(node)
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> None:
        self.loops.append(node)
        self.generic_visit(node)


def _module_bindings(tree: ast.Module) -> tuple[set[str], set[str]]:
    imported: set[str] = set()
    assigned: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported.update(alias.asname or alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.update(alias.asname or alias.name for alias in node.names)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    assigned.add(target.id)
    return imported, assigned


def _direct_callee(
    call: ast.Call,
    *,
    current_symbol: str,
    functions: Mapping[str, ast.FunctionDef | ast.AsyncFunctionDef],
) -> str | None:
    if isinstance(call.func, ast.Name):
        nested = f"{current_symbol}.{call.func.id}"
        if nested in functions:
            return nested
    if isinstance(call.func, ast.Name) and call.func.id in functions:
        return call.func.id
    if (
        isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id in {"self", "cls"}
        and "." in current_symbol
    ):
        candidate = f"{current_symbol.rsplit('.', 1)[0]}.{call.func.attr}"
        return candidate if candidate in functions else None
    return None


def audit_source_text(
    source: str,
    spec: SourceRouteSpec,
    *,
    source_id: str = "<memory>",
) -> dict[str, Any]:
    tree = ast.parse(source, filename=source_id)
    index = _FunctionIndex()
    index.visit(tree)
    imported, assigned = _module_bindings(tree)
    missing_roots = sorted(set(spec.roots) - set(index.functions))
    if missing_roots:
        return {
            "source_id": source_id,
            "status": "REJECT_MISSING_ROOT",
            "approved": False,
            "missing_roots": missing_roots,
            "reachable_symbols": [],
            "loop_findings": [],
            "unresolved_local_calls": [],
        }

    reachable: set[str] = set()
    queue = list(spec.roots)
    body_by_symbol: dict[str, _BodyVisitor] = {}
    unresolved: list[dict[str, Any]] = []
    while queue:
        symbol = queue.pop()
        if symbol in reachable:
            continue
        reachable.add(symbol)
        node = index.functions[symbol]
        visitor = _BodyVisitor()
        for statement in node.body:
            visitor.visit(statement)
        body_by_symbol[symbol] = visitor
        nested_prefix = f"{symbol}."
        for candidate in index.functions:
            if candidate.startswith(nested_prefix) and "." not in candidate[len(nested_prefix) :]:
                queue.append(candidate)
        parameters = {
            argument.arg
            for argument in (
                list(node.args.posonlyargs)
                + list(node.args.args)
                + list(node.args.kwonlyargs)
            )
        }
        for call in visitor.calls:
            callee = _direct_callee(call, current_symbol=symbol, functions=index.functions)
            if callee is not None:
                queue.append(callee)
                continue
            if isinstance(call.func, ast.Name):
                name = call.func.id
                allowed = (
                    name in _BUILTINS
                    or name in imported
                    or name in assigned
                    or name in parameters
                    or name in visitor.nested_names
                )
                if not allowed:
                    unresolved.append({"caller": symbol, "callee": name, "line": call.lineno})

    missing_required = sorted(set(spec.required_reachable) - reachable)
    loop_findings: list[dict[str, Any]] = []
    for symbol in sorted(reachable):
        for node in body_by_symbol[symbol].loops:
            declared = spec.loop_roles.get(symbol)
            if isinstance(node, (ast.For, ast.AsyncFor)):
                expression = ast.unparse(node.iter)
            else:
                expression = ast.unparse(node.test)
            loop_findings.append(
                {
                    "symbol": symbol,
                    "line": int(node.lineno),
                    "kind": type(node).__name__,
                    "expression": expression,
                    "role": declared.role if declared is not None else "undeclared",
                    "dynamic": declared.dynamic if declared is not None else None,
                    "disposition": (
                        "reject_dynamic"
                        if declared is not None and declared.dynamic
                        else "permit_declared_fixed"
                        if declared is not None
                        else "reject_undeclared"
                    ),
                }
            )
    forbidden = [
        finding
        for finding in loop_findings
        if finding["disposition"] != "permit_declared_fixed"
    ]
    approved = not forbidden and not unresolved and not missing_required
    return {
        "source_id": source_id,
        "status": "PASS_SOURCE_GUARD" if approved else "REJECT_SOURCE_GUARD",
        "approved": approved,
        "missing_roots": [],
        "missing_required_reachable": missing_required,
        "reachable_symbols": sorted(reachable),
        "loop_findings": loop_findings,
        "forbidden_loop_findings": forbidden,
        "unresolved_local_calls": unresolved,
    }


def audit_source_path(path: Path, spec: SourceRouteSpec) -> dict[str, Any]:
    return audit_source_text(path.read_text(encoding="utf-8"), spec, source_id=str(path))


def inventory_graph_def(graph_def: Any) -> dict[str, Any]:
    top_nodes = list(graph_def.node)
    functions = list(graph_def.library.function)
    function_nodes = [node for function in functions for node in function.node_def]
    functional = [
        {"scope": "top", "name": node.name, "op": node.op}
        for node in top_nodes
        if node.op in _FUNCTIONAL_LOOP_OPS
    ] + [
        {"scope": "function", "name": node.name, "op": node.op}
        for node in function_nodes
        if node.op in _FUNCTIONAL_LOOP_OPS
    ]
    return {
        "top_level_nodes": len(top_nodes),
        "function_nodes": len(function_nodes),
        "function_count": len(functions),
        "graphdef_bytes": int(graph_def.ByteSize()),
        "functional_loop_count": len(functional),
        "functional_loops": functional,
    }


__all__ = [
    "LoopRole",
    "SourceRouteSpec",
    "audit_source_path",
    "audit_source_text",
    "inventory_graph_def",
]
