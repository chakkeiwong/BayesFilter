"""Machine-checkable route policy for claim-bearing NeuTra HMC."""

from __future__ import annotations

import ast
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


NEUTRA_SEQUENTIAL_HMC_POLICY_ID = "bayesfilter_neutra_sequential_hmc_v1"
NEUTRA_HMC_ROUTE_LEDGER_SCHEMA = "bayesfilter.neutra_hmc_route_ledger.v1"
NEUTRA_HMC_ROUTE_CLASSES = frozenset(
    {
        "active_claim_bearing",
        "historical_or_superseded",
        "smoke_mechanics_or_reference",
        "training_or_non_hmc",
    }
)


class NeuTraHMCRoutePolicyError(ValueError):
    """Raised when route discovery, classification, or default binding drifts."""


def load_neutra_hmc_route_ledger(path: str | Path) -> Mapping[str, Any]:
    ledger_path = Path(path)
    try:
        payload = json.loads(ledger_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise NeuTraHMCRoutePolicyError(f"cannot read route ledger: {ledger_path}") from exc
    if not isinstance(payload, Mapping):
        raise NeuTraHMCRoutePolicyError("route ledger must be a JSON object")
    if payload.get("schema") != NEUTRA_HMC_ROUTE_LEDGER_SCHEMA:
        raise NeuTraHMCRoutePolicyError("route ledger schema mismatch")
    if payload.get("canonical_policy_id") != NEUTRA_SEQUENTIAL_HMC_POLICY_ID:
        raise NeuTraHMCRoutePolicyError("canonical NeuTra HMC policy id mismatch")
    return payload


def discover_neutra_hmc_routes(
    repository_root: str | Path,
    ledger: Mapping[str, Any],
) -> tuple[str, ...]:
    """Discover repository-owned Python routes from ledger-versioned markers."""

    root = Path(repository_root).resolve()
    discovery = _mapping(ledger.get("discovery"), "discovery")
    roots = _string_sequence(discovery.get("roots"), "discovery.roots")
    suffix = str(discovery.get("suffix", ""))
    required_marker = str(discovery.get("required_case_insensitive_marker", ""))
    behavior_markers = _string_sequence(
        discovery.get("behavior_markers"), "discovery.behavior_markers"
    )
    if suffix != ".py" or not required_marker or not behavior_markers:
        raise NeuTraHMCRoutePolicyError("route discovery rules are incomplete")
    discovered: set[str] = set()
    for relative_root in roots:
        scan_root = (root / relative_root).resolve()
        if not scan_root.is_dir():
            raise NeuTraHMCRoutePolicyError(
                f"route discovery root does not exist: {relative_root}"
            )
        for path in scan_root.rglob(f"*{suffix}"):
            source = path.read_text(encoding="utf-8")
            lowered = source.lower()
            if required_marker.lower() not in lowered:
                continue
            if not any(marker.lower() in lowered for marker in behavior_markers):
                continue
            discovered.add(str(path.resolve().relative_to(root)))
    exclusions = _ledger_path_records(discovery.get("exclusions"), "exclusions")
    for path, record in exclusions.items():
        absolute = root / path
        if not absolute.is_file():
            raise NeuTraHMCRoutePolicyError(f"stale discovery exclusion: {path}")
        if not str(record.get("reason", "")).strip():
            raise NeuTraHMCRoutePolicyError(f"discovery exclusion lacks reason: {path}")
        discovered.discard(path)
    return tuple(sorted(discovered))


def audit_neutra_hmc_route_policy(
    repository_root: str | Path,
    ledger: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Validate discovery completeness and canonical active-route binding."""

    root = Path(repository_root).resolve()
    routes = _ledger_path_records(ledger.get("routes"), "routes")
    discovered = set(discover_neutra_hmc_routes(root, ledger))
    classified = set(routes)
    errors: list[str] = []
    for path in sorted(discovered - classified):
        errors.append(f"unledgered_qualifying_route:{path}")
    for path in sorted(classified - discovered):
        errors.append(f"stale_or_undiscovered_ledger_route:{path}")
    for path, record in routes.items():
        absolute = root / path
        if not absolute.is_file():
            errors.append(f"stale_ledger_path:{path}")
            continue
        classification = str(record.get("classification", ""))
        if classification not in NEUTRA_HMC_ROUTE_CLASSES:
            errors.append(f"invalid_classification:{path}:{classification}")
            continue
        if not str(record.get("reason", "")).strip():
            errors.append(f"missing_classification_reason:{path}")
        if classification == "active_claim_bearing":
            _audit_active_route(root, path, record, routes, errors)
    return {
        "schema": "bayesfilter.neutra_hmc_route_policy_audit.v1",
        "passed": not errors,
        "canonical_policy_id": NEUTRA_SEQUENTIAL_HMC_POLICY_ID,
        "discovered_routes": tuple(sorted(discovered)),
        "classified_routes": tuple(sorted(classified)),
        "errors": tuple(errors),
    }


def require_neutra_hmc_route_policy(
    repository_root: str | Path,
    ledger: Mapping[str, Any],
) -> Mapping[str, Any]:
    audit = audit_neutra_hmc_route_policy(repository_root, ledger)
    if audit["passed"] is not True:
        raise NeuTraHMCRoutePolicyError(
            "NeuTra HMC route policy failed: " + ", ".join(audit["errors"])
        )
    return audit


def _audit_active_route(
    root: Path,
    path: str,
    record: Mapping[str, Any],
    routes: Mapping[str, Mapping[str, Any]],
    errors: list[str],
) -> None:
    source = (root / path).read_text(encoding="utf-8")
    if record.get("policy_id") != NEUTRA_SEQUENTIAL_HMC_POLICY_ID:
        errors.append(f"active_route_policy_id_mismatch:{path}")
    binding = str(record.get("core_binding", ""))
    if binding == "direct":
        entry_points = _string_sequence(
            record.get("active_entry_points"), f"active_entry_points:{path}"
        )
        required = _string_sequence(
            record.get("required_symbols"), f"required_symbols:{path}"
        )
        for symbol in required:
            if symbol not in source:
                errors.append(f"active_route_missing_core_symbol:{path}:{symbol}")
        fixed_burnin, fixed_results, missing_entries = _reachable_fixed_budget_flags(
            source, entry_points
        )
        reachable_calls = _reachable_call_names(source, entry_points)
        local_sampler_calls = _reachable_sampler_calls(source, entry_points)
        for entry_point in missing_entries:
            errors.append(f"active_route_missing_entry_point:{path}:{entry_point}")
        if fixed_burnin and fixed_results:
            fixed_role = record.get("fixed_budget_role")
            nomination_role = (
                "kernel_nomination_only_before_shared_sequential_admission"
            )
            if fixed_role != nomination_role:
                errors.append(f"active_route_fixed_terminal_budget:{path}")
            elif (
                "run_batched_hmc" not in reachable_calls
                or "run_sequential_neutra_hmc" not in reachable_calls
                or '"acceptance_role": "nomination_only"' not in source
            ):
                errors.append(f"active_route_invalid_fixed_nomination_exception:{path}")
        for marker in local_sampler_calls:
            errors.append(f"active_route_local_sampler_bypass:{path}:{marker}")
    elif binding == "delegated":
        delegate = str(record.get("delegate_path", ""))
        delegate_record = routes.get(delegate)
        if not delegate or delegate_record is None:
            errors.append(f"active_route_missing_delegate:{path}")
        elif delegate_record.get("classification") != "active_claim_bearing":
            errors.append(f"active_route_delegate_not_active:{path}:{delegate}")
    else:
        errors.append(f"active_route_invalid_core_binding:{path}:{binding}")


def _reachable_fixed_budget_flags(
    source: str,
    entry_points: Sequence[str],
) -> tuple[bool, bool, tuple[str, ...]]:
    """Inspect only functions reachable from declared active entry points."""

    tree = ast.parse(source)
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    missing = tuple(name for name in entry_points if name not in functions)
    pending = [name for name in entry_points if name in functions]
    reachable: set[str] = set()
    while pending:
        name = pending.pop()
        if name in reachable:
            continue
        reachable.add(name)
        for node in ast.walk(functions[name]):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in functions and node.func.id not in reachable:
                    pending.append(node.func.id)
    burnin = False
    results = False
    for name in reachable:
        for node in ast.walk(functions[name]):
            field = None
            value = None
            if isinstance(node, ast.keyword):
                field = node.arg
                value = node.value
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                if len(targets) == 1 and isinstance(targets[0], ast.Name):
                    field = targets[0].id
                    value = node.value
            if field == "num_burnin_steps" and _positive_integer_literal(value):
                burnin = True
            if field == "num_results" and _positive_integer_literal(value):
                results = True
    return burnin, results, missing


def _positive_integer_literal(value: ast.AST | None) -> bool:
    return (
        isinstance(value, ast.Constant)
        and isinstance(value.value, int)
        and not isinstance(value.value, bool)
        and value.value > 0
    )


def _reachable_sampler_calls(source: str, entry_points: Sequence[str]) -> tuple[str, ...]:
    tree = ast.parse(source)
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    pending = [name for name in entry_points if name in functions]
    reachable: set[str] = set()
    markers: set[str] = set()
    while pending:
        name = pending.pop()
        if name in reachable:
            continue
        reachable.add(name)
        for node in ast.walk(functions[name]):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name):
                if node.func.id in functions and node.func.id not in reachable:
                    pending.append(node.func.id)
                if node.func.id == "HamiltonianMonteCarlo":
                    markers.add("HamiltonianMonteCarlo")
            elif isinstance(node.func, ast.Attribute):
                if node.func.attr in {"HamiltonianMonteCarlo", "sample_chain"}:
                    markers.add(node.func.attr)
    return tuple(sorted(markers))


def _reachable_call_names(source: str, entry_points: Sequence[str]) -> tuple[str, ...]:
    tree = ast.parse(source)
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    pending = [name for name in entry_points if name in functions]
    reachable: set[str] = set()
    calls: set[str] = set()
    while pending:
        name = pending.pop()
        if name in reachable:
            continue
        reachable.add(name)
        for node in ast.walk(functions[name]):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name):
                calls.add(node.func.id)
                if node.func.id in functions and node.func.id not in reachable:
                    pending.append(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                calls.add(node.func.attr)
    return tuple(sorted(calls))


def _ledger_path_records(value: Any, label: str) -> dict[str, Mapping[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise NeuTraHMCRoutePolicyError(f"{label} must be a sequence")
    records: dict[str, Mapping[str, Any]] = {}
    for item in value:
        record = _mapping(item, label)
        path = str(record.get("path", ""))
        if not path:
            raise NeuTraHMCRoutePolicyError(f"{label} entry lacks path")
        if path in records:
            raise NeuTraHMCRoutePolicyError(f"duplicate {label} path: {path}")
        records[path] = record
    return records


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise NeuTraHMCRoutePolicyError(f"{label} must be a mapping")
    return value


def _string_sequence(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise NeuTraHMCRoutePolicyError(f"{label} must be a sequence")
    result = tuple(str(item) for item in value)
    if not result or any(not item for item in result):
        raise NeuTraHMCRoutePolicyError(f"{label} must contain nonempty strings")
    return result
