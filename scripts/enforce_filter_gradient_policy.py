"""Exact-source enforcement for the reviewed filtering execution closure.

This static guard complements executed consumer and enclosing-HLO checks. It
cannot prove dynamic callback purity or grant scientific admission. Exceptions
bind one AST node, function and path; changed/new numerical loops fail closed.
"""

from __future__ import annotations

import ast
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "scripts/filter_gradient_runtime_policy.json"
ITERATION = (ast.For, ast.AsyncFor, ast.While, ast.ListComp, ast.SetComp,
             ast.DictComp, ast.GeneratorExp)


def digest(node):
    return hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()


def inspect_source(path, source, scopes=None):
    """Return candidate violations; every exemption is checked separately."""
    tree = ast.parse(source)
    aliases = {}
    stack = []
    findings = []

    def active(function, kind):
        selected = scopes.get(kind) if isinstance(scopes, dict) else scopes
        return selected is None or any(function == scope or function.startswith(scope + ".") for scope in selected)

    def add(node, kind):
        function = ".".join(stack)
        if active(function, kind):
            findings.append(dict(path=path, function=function, line=node.lineno,
                kind=kind, ast_sha256=digest(node)))

    def resolve(name):
        head, separator, tail = name.partition(".")
        return aliases.get(head, head) + (separator + tail if separator else "")

    class Visitor(ast.NodeVisitor):
        def visit_ClassDef(self, node):
            stack.append(node.name)
            self.generic_visit(node)
            stack.pop()

        def visit_FunctionDef(self, node):
            stack.append(node.name)
            for decorator in node.decorator_list:
                if not isinstance(decorator, ast.Call) and resolve(ast.unparse(decorator)) == "tensorflow.function":
                    add(decorator, "non_xla_function")
            positional = [*node.args.posonlyargs, *node.args.args]
            defaults = list(zip(positional[len(positional)-len(node.args.defaults):], node.args.defaults))
            defaults += list(zip(node.args.kwonlyargs, node.args.kw_defaults))
            for argument, default in defaults:
                if argument.arg == "jit_compile" and isinstance(default, ast.Constant) and default.value is False:
                    add(node, "default_jit_disabled")
            self.generic_visit(node)
            stack.pop()

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_Import(self, node):
            for alias in node.names:
                aliases[alias.asname or alias.name.split(".")[0]] = alias.name if alias.asname else alias.name.split(".")[0]
                if alias.name == "numpy" or alias.name.startswith("numpy."):
                    add(node, "numpy_import")

        def visit_ImportFrom(self, node):
            for alias in node.names:
                aliases[alias.asname or alias.name] = (node.module or "") + "." + alias.name
            if node.module == "numpy" or (node.module or "").startswith("numpy."):
                add(node, "numpy_import")

        def generic_visit(self, node):
            if isinstance(node, ITERATION):
                add(node, "python_iteration")
            if isinstance(node, ast.Call):
                call = resolve(ast.unparse(node.func))
                terminal = call.rsplit(".", 1)[-1]
                keywords = {keyword.arg: keyword.value for keyword in node.keywords}
                if call.startswith(("numpy.", "tensorflow.experimental.numpy.")):
                    add(node, "numpy_numerics")
                if call in ("tensorflow.py_function", "tensorflow.numpy_function"):
                    add(node, "python_numerical_callback")
                if call == "tensorflow.function":
                    option = keywords.get("jit_compile")
                    if option is None or isinstance(option, ast.Constant) and option.value is not True:
                        add(node, "non_xla_function")
                if terminal in ("pfor", "vectorized_map"):
                    add(node, "pfor")
                if terminal in ("jacobian", "batch_jacobian"):
                    option = keywords.get("experimental_use_pfor")
                    if not isinstance(option, ast.Constant) or option.value is not False:
                        add(node, "implicit_pfor")
            super().generic_visit(node)

    Visitor().visit(tree)
    return findings


def verify(root=ROOT, policy_path=POLICY):
    policy = json.loads(policy_path.read_text())
    exceptions = policy["exceptions"]
    keys = [(row["path"], row["function"], row["kind"], row["ast_sha256"]) for row in exceptions]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate policy exception")
    if any(row["role"] not in ("fixed_schema", "host_validation", "host_reporting", "independent_reference", "enclosed_graph")
           or not row.get("reason", "").strip() for row in exceptions):
        raise ValueError("invalid policy exception role or missing reason")
    if any(row["role"] == "enclosed_graph" and (row["kind"] != "non_xla_function"
           or not row.get("enclosing_endpoint") or not row.get("compilation_test")) for row in exceptions):
        raise ValueError("enclosed graph exception requires its XLA endpoint and compilation test")
    by_key = dict(zip(keys, exceptions, strict=True))
    seen, violations = Counter(), []
    for path, scopes in policy["sources"].items():
        for finding in inspect_source(path, (root / path).read_text(), scopes):
            key = tuple(finding[name] for name in ("path", "function", "kind", "ast_sha256"))
            if key in by_key:
                seen[key] += 1
            else:
                violations.append(finding)
    stale = [by_key[key] for key in keys if seen[key] != by_key[key].get("occurrences", 1)]
    return dict(violations=violations, stale_exceptions=stale,
        guarded_source_count=len(policy["sources"]), exact_exceptions_used=len(seen),
        scope_note=policy["scope_note"],
        passed=not violations and not stale)


if __name__ == "__main__":
    report = verify()
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 1)
