"""Bounded, read-only inventory for ordinary HMC tuning migration debt.

The audit deliberately uses only the Python standard library.  It classifies
source references and policy text; it does not import a numerical backend,
construct a chain, or infer a consumer's scientific role from a filename.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCHEMA = "bayesfilter.ordinary_hmc_migration_surface_audit.v1"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "docs"
    / "plans"
    / "artifacts"
    / "ordinary-hmc-migration-debt-2026-09-03"
)
DEFAULT_DOWNSTREAM_ROOTS = (
    Path("/home/ubuntu/python/MacroFinance"),
    Path("/home/ubuntu/python/dsge_hmc"),
)

BAYESFILTER_SOURCE_RELATIVE_PATHS = (
    "bayesfilter/hmc_route_contract.py",
    "bayesfilter/inference/hmc_tuning_dispatch.py",
    "bayesfilter/inference/hmc_kernel_tuning.py",
    "bayesfilter/inference/hmc_kernel_selection.py",
    "bayesfilter/inference/hmc_tuning.py",
    "bayesfilter/inference/hmc_budget_ladder.py",
    "bayesfilter/inference/generic_hmc_tuning.py",
    "bayesfilter/inference/hmc_fixed_metric_grid_search.py",
    "bayesfilter/inference/hmc_operational_broad_grid.py",
    "bayesfilter/inference/hmc_robust_broad_grid.py",
    "bayesfilter/inference/fixed_trajectory_hmc_tuning_v2.py",
    "bayesfilter/inference/hmc_verification.py",
    "bayesfilter/inference/hmc_tuning_artifacts.py",
    "bayesfilter/inference/tuning_contract.py",
)

RELEVANT_SYMBOLS = frozenset(
    {
        "tune_hmc_kernel",
        "tune_fixed_transport_hmc_kernel",
        "TensorFlowHMCKernelTuningConfig",
        "HMCKernelTuningConfig",
        "FullChainHMCConfig",
        "run_full_chain_tfp_hmc",
        "run_full_chain_neural_force_hmc",
        "run_fixed_mass_hmc_tuning_budget_ladder",
        "run_fixed_metric_grid_search",
        "run_operational_broad_grid",
        "run_hmc_kernel_robust_broad_grid",
        "orchestrate_generic_hmc_tuning",
        "run_generic_hmc_tuning_orchestration",
        "run_hmc_tune_verify_repair_loop",
        "joint_l_epsilon_grid_fixed_mass_hmc",
    }
)
RELEVANT_NAME_FRAGMENTS = (
    "hmc_tuning",
    "hmc_kernel",
    "full_chain_tfp_hmc",
    "full_chain_neural_force_hmc",
    "joint_l_epsilon",
    "TensorFlowHMCKernelTuningConfig",
    "HMCKernelTuningConfig",
)

EXCLUDED_PARTS = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "env",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".tox",
        "build",
        "dist",
        "site-packages",
        ".localresources",
        "source_snapshot",
        "artifacts",
        "generated",
        "results",
        "private_wrapper",
        "functional_source_snapshot",
        "snapshots",
        "outputs",
        "output",
        "logs",
        "tmp",
    }
)
ORDINARY_NUMPY_MODULE_NAMES = frozenset(
    {
        "hmc_kernel_tuning.py",
        "hmc_kernel_selection.py",
        "hmc_tuning.py",
        "hmc_budget_ladder.py",
        "generic_hmc_tuning.py",
        "hmc_fixed_metric_grid_search.py",
        "hmc_operational_broad_grid.py",
        "hmc_robust_broad_grid.py",
        "fixed_trajectory_hmc_tuning_v2.py",
        "hmc_verification.py",
        "hmc_tuning_artifacts.py",
    }
)
CLAIM_MARKER_RE = re.compile(
    r"\b(?:claim|promotion|admission|authority|posterior|production|"
    r"default|leaderboard|serious|canonical)\b",
    re.IGNORECASE,
)
HISTORICAL_MARKER_RE = re.compile(
    r"\b(?:historical|legacy|diagnostic|smoke|reference|non[-_ ]promot|"
    r"migration debt|policy mismatch|reviewed exception)\b",
    re.IGNORECASE,
)
UNQUALIFIED_NON_XLA_RE = re.compile(
    r"(?:default|defaults|defaulting)[^\n]{0,100}(?:use[_\\]?xla\s*[=:]\s*false|"
    r"non[- ]xla)",
    re.IGNORECASE,
)


def _dotted_name(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted_name(node.value)
        return None if prefix is None else f"{prefix}.{node.attr}"
    return None


def _constant_string(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


class _ReferenceVisitor(ast.NodeVisitor):
    """Collect imports, relevant calls, NumPy calls, and dynamic indirection."""

    def __init__(self, source: str) -> None:
        self.source = source
        self.hmc_context = any(
            fragment.lower() in source.lower()
            for fragment in (*RELEVANT_NAME_FRAGMENTS, "bayesfilter")
        )
        self.aliases: dict[str, str] = {}
        self.imports: list[dict[str, Any]] = []
        self.references: list[dict[str, Any]] = []
        self.numpy_imports: list[dict[str, Any]] = []
        self.numpy_calls: list[dict[str, Any]] = []
        self.dynamic_imports: list[dict[str, Any]] = []
        self._function_stack: list[str] = []

    def _resolve(self, name: str | None) -> str | None:
        if name is None:
            return None
        first, dot, rest = name.partition(".")
        replacement = self.aliases.get(first, first)
        return replacement + (dot + rest if dot else "")

    def _record_reference(self, name: str, node: ast.AST, kind: str) -> None:
        resolved = self._resolve(name)
        if resolved is None:
            return
        if not (
            resolved in RELEVANT_SYMBOLS
            or any(fragment in resolved for fragment in RELEVANT_NAME_FRAGMENTS)
        ):
            return
        self.references.append(
            {
                "kind": kind,
                "name": resolved,
                "line": int(getattr(node, "lineno", 0)),
                "function": self._function_stack[-1]
                if self._function_stack
                else None,
            }
        )

    def visit_Import(self, node: ast.Import) -> Any:
        for alias in node.names:
            local = alias.asname or alias.name.split(".", 1)[0]
            self.aliases[local] = alias.name
            if alias.name == "numpy" or alias.name.startswith("numpy."):
                self.numpy_imports.append(
                    {"line": int(node.lineno), "name": alias.name, "alias": local}
                )
            if alias.name.startswith("bayesfilter"):
                self.imports.append(
                    {"line": int(node.lineno), "module": alias.name, "alias": local}
                )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> Any:
        module = node.module or ""
        for alias in node.names:
            if alias.name == "*":
                continue
            local = alias.asname or alias.name
            qualified = f"{module}.{alias.name}" if module else alias.name
            self.aliases[local] = qualified
            if module == "numpy" or module.startswith("numpy."):
                self.numpy_imports.append(
                    {"line": int(node.lineno), "name": qualified, "alias": local}
                )
            if module.startswith("bayesfilter"):
                self.imports.append(
                    {"line": int(node.lineno), "module": qualified, "alias": local}
                )
                self._record_reference(qualified, node, "import")
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        self._function_stack.append(node.name)
        self.generic_visit(node)
        self._function_stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
        self._function_stack.append(node.name)
        self.generic_visit(node)
        self._function_stack.pop()

    def visit_Call(self, node: ast.Call) -> Any:
        dotted = _dotted_name(node.func)
        resolved = self._resolve(dotted)
        if dotted is not None:
            self._record_reference(dotted, node, "call")
        if resolved and (
            resolved == "numpy"
            or resolved.startswith("numpy.")
            or resolved.startswith("numpy")
        ):
            self.numpy_calls.append(
                {
                    "line": int(node.lineno),
                    "name": resolved,
                    "function": self._function_stack[-1]
                    if self._function_stack
                    else None,
                }
            )

        dynamic_reason: str | None = None
        if resolved in {"importlib.import_module", "import_module", "__import__"}:
            if not self.hmc_context:
                dynamic_reason = None
            elif not node.args or _constant_string(node.args[0]) is None:
                dynamic_reason = "computed_import_module"
            else:
                dynamic_reason = "constant_import_module"
        elif resolved in {
            "importlib.util.spec_from_file_location",
            "spec_from_file_location",
        }:
            if not self.hmc_context:
                dynamic_reason = None
            elif not node.args or _constant_string(node.args[0]) is None:
                dynamic_reason = "computed_spec_from_file_location"
            else:
                dynamic_reason = "constant_spec_from_file_location"
        elif resolved in {"importlib.metadata.entry_points", "entry_points"}:
            if self.hmc_context:
                dynamic_reason = "entry_point_resolution"
        elif resolved in {"getattr", "builtins.getattr"}:
            attribute = _constant_string(node.args[1]) if len(node.args) >= 2 else None
            base = _dotted_name(node.args[0]) if node.args else None
            hmc_attribute = bool(
                attribute
                and any(
                    fragment.lower() in attribute.lower()
                    for fragment in RELEVANT_NAME_FRAGMENTS
                )
            )
            module_like_base = bool(
                base
                and any(
                    marker in base.lower()
                    for marker in ("module", "import", "bayesfilter", "runtime")
                )
            )
            if self.hmc_context and (
                attribute is None or hmc_attribute or module_like_base
            ):
                dynamic_reason = (
                    "computed_getattr_import_like"
                    if module_like_base or hmc_attribute
                    else "computed_getattr"
                )

        if dynamic_reason is not None:
            self.dynamic_imports.append(
                {
                    "line": int(node.lineno),
                    "expression": resolved or dotted or "<call>",
                    "classification": (
                        "unknown_dynamic_import"
                        if dynamic_reason in {
                            "computed_import_module",
                            "computed_spec_from_file_location",
                            "computed_getattr_import_like",
                            "entry_point_resolution",
                        }
                        else "unresolved_dynamic_attribute"
                        if dynamic_reason == "computed_getattr"
                        else "constant_dynamic_import"
                    ),
                    "reason": dynamic_reason,
                    "function": self._function_stack[-1]
                    if self._function_stack
                    else None,
                }
            )
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> Any:
        resolved = self._resolve(node.id)
        if resolved and (
            resolved in RELEVANT_SYMBOLS
            or any(fragment in resolved for fragment in RELEVANT_NAME_FRAGMENTS)
        ):
            self._record_reference(node.id, node, "name")
        self.generic_visit(node)


def _excluded(path: Path) -> bool:
    return any(part in EXCLUDED_PARTS for part in path.parts)


def _iter_python_files(root: Path, *, max_files: int) -> tuple[list[Path], list[str]]:
    if not root.is_dir():
        return [], [f"missing_root:{root}"]
    paths: list[Path] = []
    notes: list[str] = []
    for path in sorted(root.rglob("*.py")):
        if _excluded(path) or path.is_symlink():
            continue
        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        # The downstream roots contain many unrelated model and test files.
        # Keep the bounded inventory focused on source that names the HMC or
        # BayesFilter surfaces under review; the exact paths retained are
        # recorded in the report.
        lowered = source.lower()
        if not any(fragment.lower() in lowered for fragment in RELEVANT_NAME_FRAGMENTS):
            continue
        paths.append(path)
        if len(paths) >= max_files:
            notes.append(f"file_cap_reached:{max_files}")
            break
    return paths, notes


def _claim_markers(source: str) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(source.splitlines(), start=1):
        matches = tuple(sorted({m.group(0).lower() for m in CLAIM_MARKER_RE.finditer(line)}))
        if matches:
            rows.append({"line": line_number, "markers": matches})
    return tuple(rows)


def _scan_python_file(path: Path, *, root_label: str) -> dict[str, Any]:
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return {
            "path": str(path),
            "root": root_label,
            "status": "unreadable",
            "error": str(exc),
        }
    row: dict[str, Any] = {
        "path": str(path),
        "root": root_label,
        "sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "line_count": len(source.splitlines()),
        "claim_markers": _claim_markers(source),
    }
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        row.update(
            {
                "status": "syntax_error",
                "error": f"{exc.msg} at line {exc.lineno}",
                "references": (),
                "dynamic_imports": (),
                "numpy_imports": (),
                "numpy_calls": (),
            }
        )
        return row

    visitor = _ReferenceVisitor(source)
    visitor.visit(tree)
    references = tuple(
        sorted(
            visitor.references,
            key=lambda value: (value["line"], value["kind"], value["name"]),
        )
    )
    dynamic = tuple(
        sorted(visitor.dynamic_imports, key=lambda value: value["line"])
    )
    claim_adjacent = bool(visitor.references and row["claim_markers"])
    kinds = {reference["name"] for reference in references}
    public = any(
        name.endswith(".tune_hmc_kernel")
        or name.endswith(".tune_fixed_transport_hmc_kernel")
        for name in kinds
    )
    diagnostic = any(
        marker in name
        for name in kinds
        for marker in (
            "budget_ladder",
            "generic_hmc_tuning",
            "fixed_metric",
            "operational_broad_grid",
            "robust_broad_grid",
            "fixed_trajectory_hmc_tuning_v2",
        )
    )
    raw_runner = any(
        name.endswith(".run_full_chain_tfp_hmc")
        or name.endswith(".run_full_chain_neural_force_hmc")
        or name.endswith(".FullChainHMCConfig")
        for name in kinds
    )
    typed_mechanics = any(
        name.endswith(".TensorFlowHMCKernelTuningConfig") for name in kinds
    )
    if dynamic and any(
        value["classification"]
        in {"unknown_dynamic_import", "unresolved_dynamic_attribute"}
        for value in dynamic
    ):
        role = "unknown_dynamic_import"
    elif public and (diagnostic or raw_runner):
        role = "mixed_public_and_lower_level"
    elif public:
        role = "public_tuner_reference"
    elif typed_mechanics:
        role = "typed_mechanics_reference"
    elif diagnostic:
        role = "diagnostic_or_historical_reference"
    elif raw_runner:
        role = "raw_runner_reference"
    elif references:
        role = "other_hmc_reference"
    else:
        role = "no_relevant_reference"
    row.update(
        {
            "status": "scanned",
            "references": references,
            "dynamic_imports": dynamic,
            "numpy_imports": tuple(visitor.numpy_imports),
            "numpy_calls": tuple(visitor.numpy_calls),
            "consumer_role": role,
            "claim_adjacent": claim_adjacent,
            "requires_manual_role_review": bool(
                claim_adjacent and role != "no_relevant_reference"
            ),
        }
    )
    return row


def _git_output(repo_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _branch_reachability() -> tuple[dict[str, Any], ...]:
    # Import the pure route module only after disabling the optional package
    # preload.  This keeps the audit independent of TensorFlow initialization.
    os.environ["BAYESFILTER_PRELOAD_CUSTOM_OP"] = "0"
    from bayesfilter.hmc_route_contract import (  # pylint: disable=import-outside-toplevel
        HMC_FIXED_TRAJECTORY_STAGE,
        HMC_TOP_LEVEL_SELECTION_STAGE,
        HMC_WINDOWED_MASS_STAGE,
        LEGACY_JOINT_L_EPSILON_ALGORITHM_ID,
        LEGACY_SEGMENTED_WINDOWED_MASS_ALGORITHM_ID,
        ORDINARY_BROAD_FIXED_METRIC_ALGORITHM_ID,
        OPERATIONAL_WINDOWED_WARMUP_ALGORITHM_ID,
        resolve_hmc_algorithm_route,
        require_hmc_artifact_authority_route,
    )

    cases = (
        (
            "default_ordinary_config",
            "ordinary_hmc",
            ORDINARY_BROAD_FIXED_METRIC_ALGORITHM_ID,
            HMC_TOP_LEVEL_SELECTION_STAGE,
            "public ordinary construction",
        ),
        (
            "explicit_legacy_config",
            "ordinary_hmc_legacy_diagnostic",
            LEGACY_JOINT_L_EPSILON_ALGORITHM_ID,
            HMC_TOP_LEVEL_SELECTION_STAGE,
            "public ordinary construction",
        ),
        (
            "operational_windowed_stage",
            "ordinary_internal_stage",
            OPERATIONAL_WINDOWED_WARMUP_ALGORITHM_ID,
            HMC_WINDOWED_MASS_STAGE,
            "internal warm-up stage",
        ),
        (
            "legacy_windowed_stage",
            "ordinary_internal_legacy_stage",
            LEGACY_SEGMENTED_WINDOWED_MASS_ALGORITHM_ID,
            HMC_WINDOWED_MASS_STAGE,
            "direct diagnostic stage",
        ),
        (
            "legacy_fixed_stage",
            "ordinary_internal_legacy_stage",
            LEGACY_JOINT_L_EPSILON_ALGORITHM_ID,
            HMC_FIXED_TRAJECTORY_STAGE,
            "direct diagnostic stage",
        ),
    )
    rows: list[dict[str, Any]] = []
    for case, variant, algorithm_id, stage, entry in cases:
        decision = resolve_hmc_algorithm_route(
            algorithm_id=algorithm_id,
            stage=stage,
        )
        guard = "not_attempted"
        if stage == HMC_TOP_LEVEL_SELECTION_STAGE:
            try:
                require_hmc_artifact_authority_route(
                    algorithm_id=algorithm_id,
                    stage=stage,
                )
            except Exception as exc:  # the type is part of the recorded result
                guard = f"rejected:{type(exc).__name__}"
            else:
                guard = "accepted_replayable_route_only"
        rows.append(
            {
                "case": case,
                "config_variant": variant,
                "entry": entry,
                "algorithm_id": algorithm_id,
                "stage": stage,
                "supported": decision.supported,
                "operational_authority": decision.operational_authority,
                "artifact_authority": decision.artifact_authority,
                "scientific_promotion_authority": decision.scientific_promotion_authority,
                "promotion_role": decision.promotion_role,
                "artifact_guard": guard,
                "source_anchors": (
                    "bayesfilter/hmc_route_contract.py",
                    "bayesfilter/inference/hmc_tuning_dispatch.py",
                    "bayesfilter/inference/hmc_kernel_tuning.py",
                ),
            }
        )
    rows.append(
        {
            "case": "typed_tensorflow_mechanics_config",
            "config_variant": "tensorflow_mechanics",
            "entry": "public dispatcher typed branch",
            "algorithm_id": None,
            "stage": "typed_dispatch",
            "supported": True,
            "operational_authority": False,
            "artifact_authority": False,
            "scientific_promotion_authority": False,
            "promotion_role": "mechanics_only",
            "artifact_guard": "not_an_ordinary_authority_route",
            "source_anchors": (
                "bayesfilter/inference/hmc_tuning_dispatch.py",
                "bayesfilter/inference/hmc_tensorflow_tuning.py",
            ),
        }
    )
    return tuple(rows)


def _stale_policy_scan(repo_root: Path) -> tuple[dict[str, Any], ...]:
    paths: list[Path] = [
        repo_root / "docs/reference/hmc-tuning-interface.md",
        repo_root / "docs/chapters/ch21b_hmc_tuning_interfaces.tex",
    ]
    paths.extend(sorted((repo_root / "docs/examples").glob("hmc_tuning*.py")))
    rows: list[dict[str, Any]] = []
    for path in paths:
        if not path.is_file():
            rows.append({"path": str(path), "status": "missing"})
            continue
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not any(
                token in line.lower()
                for token in ("use_xla", "non-xla", "joint", "promoted_default")
            ):
                continue
            historical_or_qualified = bool(HISTORICAL_MARKER_RE.search(line))
            unqualified_non_xla = bool(UNQUALIFIED_NON_XLA_RE.search(line))
            rows.append(
                {
                    "path": str(path),
                    "line": line_number,
                    "text": line.strip(),
                    "historical_or_qualified": historical_or_qualified,
                    "unqualified_non_xla_default": unqualified_non_xla
                    and not historical_or_qualified,
                    "classification": (
                        "historical_or_qualified"
                        if historical_or_qualified
                        else "active_guidance_candidate"
                    ),
                }
            )
    return tuple(rows)


def _numpy_ledger(rows: Iterable[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    ledger: list[dict[str, Any]] = []
    for row in rows:
        imports = tuple(row.get("numpy_imports", ()))
        calls = tuple(row.get("numpy_calls", ()))
        if not imports and not calls:
            continue
        path = Path(str(row["path"]))
        classification = (
            "runtime_candidate"
            if path.name in ORDINARY_NUMPY_MODULE_NAMES
            else "diagnostic_or_reference_or_unclassified"
        )
        ledger.append(
            {
                "path": str(path),
                "root": row.get("root"),
                "classification": classification,
                "imports": imports,
                "calls": calls,
                "claim_adjacent": bool(row.get("claim_adjacent", False)),
            }
        )
    return tuple(ledger)


def _markdown_report(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Ordinary HMC Migration Surface Audit",
        "",
        f"Schema: `{payload['schema']}`.",
        f"Source revision: `{payload['source_revision']}`.",
        "",
        "This is a bounded static classification report. It contains no HMC, "
        "tuning, GPU, or numerical evidence.",
        "",
        "## Summary",
        "",
        f"- Python files scanned: {len(payload['scanned_files'])}",
        f"- Consumer rows with relevant references: {payload['summary']['relevant_consumers']}",
        f"- Unknown dynamic-import rows: {payload['summary']['unknown_dynamic_imports']}",
        f"- Unresolved dynamic-attribute rows: {payload['summary']['unresolved_dynamic_attributes']}",
        f"- NumPy runtime-candidate modules: {payload['summary']['numpy_runtime_candidates']}",
        f"- Unqualified non-XLA findings: {payload['summary']['unqualified_non_xla_defaults']}",
        "",
        "## Branch Reachability",
        "",
        "| Case | Variant | Algorithm | Stage | Artifact guard | Operational | Artifact | Scientific |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in payload["branch_reachability"]:
        lines.append(
            "| {case} | {config_variant} | `{algorithm_id}` | `{stage}` | "
            "{artifact_guard} | {operational_authority} | {artifact_authority} | "
            "{scientific_promotion_authority} |".format(**row)
        )
    lines.extend(["", "## Consumer Role Ledger", ""])
    for row in payload["consumer_rows"]:
        if row.get("role") == "no_relevant_reference":
            continue
        lines.append(
            f"- `{row['path']}`: `{row['role']}`; claim-adjacent="
            f"`{row['claim_adjacent']}`; manual-review=`{row['requires_manual_role_review']}`."
        )
        for dynamic in row.get("dynamic_imports", ()):
            lines.append(
                f"  - line {dynamic['line']}: `{dynamic['classification']}` "
                f"({dynamic['reason']})."
            )
    lines.extend(["", "## NumPy Call-Chain Ledger", ""])
    for row in payload["numpy_ledger"]:
        lines.append(
            f"- `{row['path']}`: `{row['classification']}`; "
            f"imports={len(row['imports'])}; calls={len(row['calls'])}."
        )
    lines.extend(["", "## Stale-Policy Scan", ""])
    for row in payload["stale_policy_findings"]:
        if row.get("status") == "missing":
            lines.append(f"- missing: `{row['path']}`")
        else:
            lines.append(
                f"- `{row['path']}:{row['line']}`: `{row['classification']}` "
                f"{row['text']}"
            )
    lines.extend(
        [
            "",
            "## Scope And Provenance",
            "",
            f"- Commands: `{json.dumps(payload['commands'])}`",
            f"- Downstream roots: `{json.dumps(payload['downstream_roots'])}`",
            f"- Exclusions: `{json.dumps(payload['exclusions'])}`",
            f"- Git status SHA-256: `{payload['git_status_sha256']}`",
            "",
            "Unknown dynamic imports and claim-adjacent rows require manual role "
            "classification before any authority or scientific admission. The "
            "NumPy rows are migration findings, not a promotion result.",
            "",
        ]
    )
    return "\n".join(lines)


def build_payload(
    repo_root: Path,
    downstream_roots: Sequence[Path],
    *,
    max_files: int,
    command: Sequence[str],
) -> dict[str, Any]:
    source_rows: list[dict[str, Any]] = []
    scanned_files: list[str] = []
    scan_notes: list[str] = []
    for relative in BAYESFILTER_SOURCE_RELATIVE_PATHS:
        path = repo_root / relative
        if not path.is_file():
            source_rows.append(
                {"path": str(path), "root": "bayesfilter", "status": "missing"}
            )
            continue
        source_rows.append(_scan_python_file(path, root_label="bayesfilter"))
        scanned_files.append(str(path))

    consumer_rows: list[dict[str, Any]] = []
    for root in downstream_roots:
        paths, notes = _iter_python_files(root, max_files=max_files)
        scan_notes.extend(notes)
        for path in paths:
            row = _scan_python_file(path, root_label=str(root))
            if row.get("references") or row.get("dynamic_imports"):
                consumer_rows.append(
                    {
                        "path": row["path"],
                        "root": row["root"],
                        "role": row.get("consumer_role"),
                        "claim_adjacent": row.get("claim_adjacent", False),
                        "requires_manual_role_review": row.get(
                            "requires_manual_role_review", False
                        ),
                        "references": row.get("references", ()),
                        "dynamic_imports": row.get("dynamic_imports", ()),
                        "sha256": row.get("sha256"),
                    }
                )
            scanned_files.append(str(path))

    status = _git_output(repo_root, "status", "--short", "--untracked-files=all")
    status_bytes = status.encode("utf-8")
    numpy_rows = _numpy_ledger(source_rows)
    stale_rows = _stale_policy_scan(repo_root)
    unknown_dynamic = sum(
        1
        for row in consumer_rows
        for dynamic in row.get("dynamic_imports", ())
        if dynamic.get("classification") == "unknown_dynamic_import"
    )
    unresolved_dynamic_attributes = sum(
        1
        for row in consumer_rows
        for dynamic in row.get("dynamic_imports", ())
        if dynamic.get("classification") == "unresolved_dynamic_attribute"
    )
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "source_revision": _git_output(repo_root, "rev-parse", "HEAD"),
        "observed_worktree_dirty": bool(status),
        "git_status_sha256": hashlib.sha256(status_bytes).hexdigest(),
        "python": sys.version,
        "package_versions": {
            "tensorflow": _package_version("tensorflow"),
            "tensorflow_probability": _package_version("tensorflow-probability"),
        },
        "commands": [list(command)],
        "downstream_roots": [str(root) for root in downstream_roots],
        "scanned_files": tuple(sorted(set(scanned_files))),
        "scan_notes": tuple(sorted(set(scan_notes))),
        "source_rows": tuple(source_rows),
        "consumer_rows": tuple(consumer_rows),
        "branch_reachability": _branch_reachability(),
        "numpy_ledger": numpy_rows,
        "stale_policy_findings": stale_rows,
        "exclusions": tuple(sorted(EXCLUDED_PARTS)),
        "summary": {
            "relevant_consumers": len(consumer_rows),
            "unknown_dynamic_imports": unknown_dynamic,
            "unresolved_dynamic_attributes": unresolved_dynamic_attributes,
            "numpy_runtime_candidates": sum(
                1 for row in numpy_rows if row["classification"] == "runtime_candidate"
            ),
            "unqualified_non_xla_defaults": sum(
                1
                for row in stale_rows
                if row.get("unqualified_non_xla_default") is True
            ),
        },
    }
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--downstream-root",
        action="append",
        type=Path,
        dest="downstream_roots",
        help="bounded Python source root to scan; may be repeated",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-files", type=int, default=1200)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    repo_root = args.repo_root.resolve()
    downstream_roots = tuple(
        root.resolve()
        for root in (
            args.downstream_roots
            if args.downstream_roots is not None
            else DEFAULT_DOWNSTREAM_ROOTS
        )
    )
    command = [sys.executable, str(Path(__file__).resolve()), *(argv or sys.argv[1:])]
    payload = build_payload(
        repo_root,
        downstream_roots,
        max_files=max(1, int(args.max_files)),
        command=command,
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "ordinary_hmc_surface_inventory.json"
    markdown_path = output_dir / "ordinary_hmc_surface_inventory.md"
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=list) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(_markdown_report(payload) + "\n", encoding="utf-8")
    derived_payloads = {
        "branch_reachability.json": payload["branch_reachability"],
        "consumer_role_ledger.json": payload["consumer_rows"],
        "numpy_call_chain_ledger.json": payload["numpy_ledger"],
        "provenance_capture.json": {
            "source_revision": payload["source_revision"],
            "observed_worktree_dirty": payload["observed_worktree_dirty"],
            "git_status_sha256": payload["git_status_sha256"],
            "python": payload["python"],
            "package_versions": payload["package_versions"],
            "commands": payload["commands"],
            "downstream_roots": payload["downstream_roots"],
            "scanned_files": payload["scanned_files"],
            "scan_notes": payload["scan_notes"],
            "exclusions": payload["exclusions"],
        },
    }
    for filename, value in derived_payloads.items():
        (output_dir / filename).write_text(
            json.dumps(value, indent=2, sort_keys=True, default=list) + "\n",
            encoding="utf-8",
        )
    trace_path = output_dir / "trace_note.md"
    trace_path.write_text(
        _markdown_report(payload)
        + "\n\nThe combined JSON inventory is the source for the derived ledgers above.\n",
        encoding="utf-8",
    )
    print(json_path)
    print(markdown_path)
    for filename in sorted(derived_payloads):
        print(output_dir / filename)
    print(trace_path)
    if args.check:
        if (
            payload["summary"]["unknown_dynamic_imports"]
            or payload["summary"]["unresolved_dynamic_attributes"]
        ):
            return 1
        if payload["summary"]["unqualified_non_xla_defaults"]:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
