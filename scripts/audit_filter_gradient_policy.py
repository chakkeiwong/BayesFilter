"""Repository-wide, syntax-only filter/gradient execution policy inventory.

Counts are search leads, NOT violation verdicts. The companion reviewed report
classifies actual numerical loops and follows consumers. This tool uses only
the standard library and never imports numerical modules being audited.
"""

from __future__ import annotations

import argparse
import ast
import collections
import gzip
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOKENS = re.compile(r"kalman|ukf|srukf|sigma|sgqf|particle|pfpf|sinkhorn|sqmc|tensor_train|filter|score|gradient|derivative|jvp|vjp|quadrature|cubature|ledh", re.IGNORECASE)
LOOPS = (ast.For, ast.AsyncFor, ast.While, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
BOUNDARIES = {"function", "while_loop", "scan", "map_fn", "vectorized_map", "py_function", "numpy_function", "GradientTape", "ForwardAccumulator", "jacobian", "batch_jacobian", "numpy", "pfor", "custom_gradient"}


def module_name(path):
    parts = Path(path).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def ownership(path):
    if path.startswith((".localsource/", ".localresources/")) or "/vendor/" in path:
        return "external_reference_source"
    if path.startswith("tests/"):
        return "test"
    if "/artifacts/" in path and path.startswith("docs/"):
        return "archived_document_artifact_python"
    if path.startswith("docs/"):
        return "owned_document_harness"
    if path.startswith("bayesfilter/testing/"):
        return "package_test_support"
    return "owned_source"


def inspect(path):
    source = (ROOT / path).read_text()
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return {"path": path, "ownership": ownership(path), "parse_error": str(exc)}
    module = module_name(path)
    package = module if path.endswith("/__init__.py") else module.rpartition(".")[0]
    aliases = {}
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                aliases[alias.asname or alias.name.split(".")[0]] = alias.name if alias.asname else alias.name.split(".")[0]
                imports.append({"line": node.lineno, "module": alias.name, "name": None})
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                components = package.split(".")
                base = ".".join(components[:len(components) - node.level + 1] + ([base] if base else []))
            for alias in node.names:
                aliases[alias.asname or alias.name] = base + "." + alias.name
                imports.append({"line": node.lineno, "module": base, "name": alias.name})
    row = {"path": path, "module": module, "ownership": ownership(path), "sha256": hashlib.sha256(source.encode()).hexdigest(), "doc": (ast.get_docstring(tree) or "")[:1000], "imports": imports, "functions": [], "loops": [], "boundaries": [], "calls": [], "numpy_calls": [], "lazy_exports": {}}
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id == "_EXPORT_MODULES" for target in targets):
                try:
                    value = ast.literal_eval(node.value)
                    if isinstance(value, dict) and all(isinstance(item, str) for item in value.values()):
                        row["lazy_exports"] = value
                except (ValueError, TypeError):
                    pass
    stack = []

    def expand(name):
        head, sep, tail = name.partition(".")
        return aliases.get(head, head) + (sep + tail if sep else "")

    class Visitor(ast.NodeVisitor):
        def visit_ClassDef(self, node):
            stack.append(node.name)
            self.generic_visit(node)
            stack.pop()

        def visit_FunctionDef(self, node):
            stack.append(node.name)
            positional = [*node.args.posonlyargs, *node.args.args]
            defaults = {arg.arg: ast.unparse(value) for arg, value in zip(positional[len(positional) - len(node.args.defaults):], node.args.defaults)}
            defaults.update({arg.arg: ast.unparse(value) for arg, value in zip(node.args.kwonlyargs, node.args.kw_defaults) if value is not None})
            row["functions"].append({"name": ".".join(stack), "line": node.lineno, "end_line": node.end_lineno, "defaults": defaults, "decorators": [ast.unparse(item) for item in node.decorator_list], "doc": (ast.get_docstring(node) or "").split("\n")[0][:240]})
            for decorator in node.decorator_list:
                if not isinstance(decorator, ast.Call) and expand(ast.unparse(decorator)) == "tensorflow.function":
                    row["boundaries"].append({"function": ".".join(stack), "line": decorator.lineno, "call": "tensorflow.function", "keywords": {}, "bare_decorator": True})
            self.generic_visit(node)
            stack.pop()

        visit_AsyncFunctionDef = visit_FunctionDef

        def generic_visit(self, node):
            if isinstance(node, LOOPS):
                if isinstance(node, (ast.For, ast.AsyncFor)):
                    header = ast.unparse(node.target) + " in " + ast.unparse(node.iter)
                elif isinstance(node, ast.While):
                    header = ast.unparse(node.test)
                else:
                    header = "; ".join(ast.unparse(item.target) + " in " + ast.unparse(item.iter) for item in node.generators)
                row["loops"].append({"function": ".".join(stack), "line": node.lineno, "kind": type(node).__name__, "header": header[:300], "ast_sha256": hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()})
            if isinstance(node, ast.Call):
                name = ast.unparse(node.func)
                resolved = expand(name)
                row["calls"].append({"function": ".".join(stack), "line": node.lineno, "name": name, "import_expanded": resolved, "kind": "call"})
                for argument in [*node.args, *(item.value for item in node.keywords)]:
                    if isinstance(argument, (ast.Name, ast.Attribute)):
                        reference = ast.unparse(argument)
                        row["calls"].append({"function": ".".join(stack), "line": argument.lineno, "name": reference, "import_expanded": expand(reference), "kind": "callback_reference"})
                if resolved.startswith("numpy."):
                    row["numpy_calls"].append({"function": ".".join(stack), "line": node.lineno, "call": resolved})
                if resolved.rsplit(".", 1)[-1] in BOUNDARIES:
                    row["boundaries"].append({"function": ".".join(stack), "line": node.lineno, "call": resolved, "keywords": {item.arg: ast.unparse(item.value)[:250] for item in node.keywords if item.arg}})
            if isinstance(node, ast.Return) and isinstance(node.value, ast.Name):
                reference = node.value.id
                row["calls"].append({"function": ".".join(stack), "line": node.lineno, "name": reference, "import_expanded": expand(reference), "kind": "returned_function_reference"})
            super().generic_visit(node)

    Visitor().visit(tree)
    # Exclude the package prefix: otherwise 'BayesFilter' selects everything.
    relative = path.removeprefix("bayesfilter/")
    row["algorithm_name_match"] = bool(TOKENS.search(relative) or any(TOKENS.search(fn["name"]) for fn in row["functions"]))
    return row


def audit():
    tracked = subprocess.check_output(["git", "ls-files", "*.py"], cwd=ROOT, text=True).splitlines()
    discovered = sorted(set(subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "*.py"],
        cwd=ROOT, text=True).splitlines()))
    rows = [inspect(path) for path in discovered if (ROOT / path).exists()]
    modules = {row["module"]: row for row in rows if "module" in row}
    definitions = {row["module"] + "." + fn["name"] for row in modules.values() for fn in row["functions"]}
    exports = {}
    for row in modules.values():
        for name, target in row["lazy_exports"].items():
            exports[row["module"] + "." + name] = target + "." + name
        if row["path"].endswith("/__init__.py"):
            for item in row["imports"]:
                if item["name"] and item["name"] != "*":
                    exports[row["module"] + "." + item["name"]] = item["module"] + "." + item["name"]
    edges = []
    for row in modules.values():
        row["numpy_imports"] = [item for item in row["imports"] if item["module"] == "numpy" or item["module"].startswith("numpy.")]
        for call in row.pop("calls"):
            target = call["import_expanded"]
            seen = set()
            while target in exports and target not in seen:
                seen.add(target)
                target = exports[target]
            if target not in definitions and "." not in call["name"]:
                scope = call["function"].split(".")
                while scope:
                    candidate = row["module"] + "." + ".".join(scope + [call["name"]])
                    if candidate in definitions:
                        target = candidate
                        break
                    scope.pop()
                else:
                    target = row["module"] + "." + call["name"]
            if target in definitions:
                edges.append({"caller": row["module"] + ("." + call["function"] if call["function"] else ""), "line": call["line"], "target": target, "path": row["path"], "kind": call["kind"]})
    owned = [row for row in modules.values() if row["ownership"].startswith("owned_")]
    summary = {"tracked_python_files": len(tracked), "parsed_files": len(modules), "parse_errors": len(rows) - len(modules), "ownership_counts": dict(collections.Counter(row["ownership"] for row in rows)), "owned_source_modules": len(owned), "owned_name_matched_modules": sum(row["algorithm_name_match"] for row in owned), "owned_loop_syntax_sites": sum(len(row["loops"]) for row in owned), "owned_numpy_import_sites": sum(len(row["numpy_imports"]) for row in owned), "resolved_static_call_edges": len(edges)}
    summary["working_tree_python_files"] = len(discovered)
    summary["untracked_python_files"] = len(set(discovered) - set(tracked))
    return {"schema": "bayesfilter.filter_gradient_policy_audit.v2", "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "summary": summary, "modules": rows, "resolved_static_call_edges": edges, "limits": ["Syntax counts are not violation counts or compliance certificates.", "Import/name resolution is static and best-effort; rebinding, method dispatch, closure factories and callbacks need runtime checks.", "Ownership records provenance, not an automatic policy exemption.", "Tracked and nonignored untracked Python files are discovered; ignored files and non-Python implementations are not claimed covered.", "A decorator does not prove execution or enclosing compilation; memory workers inspect reachable GraphDefs and export HLO."]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()
    report = audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with (gzip.open(args.output, "xt") if args.output.suffix == ".gz" else args.output.open("x")) as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    if args.markdown:
        lines = ["# Repository filter/gradient static inventory", "", "Syntax counts are search leads, not violation verdicts. See the reviewed result for classification.", "", "```json", json.dumps(report["summary"], indent=2), "```", "", "## Owned source modules", "", "| Module | Python loop sites | NumPy imports | TF function declarations |", "|---|---:|---:|---:|"]
        for row in report["modules"]:
            if row["ownership"].startswith("owned_") and "module" in row:
                declarations = sum(item["call"] == "tensorflow.function" for item in row["boundaries"])
                lines.append(f"| `{row['path']}` | {len(row['loops'])} | {len(row['numpy_imports'])} | {declarations} |")
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        with args.markdown.open("x") as handle:
            handle.write("\n".join(lines) + "\n")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
