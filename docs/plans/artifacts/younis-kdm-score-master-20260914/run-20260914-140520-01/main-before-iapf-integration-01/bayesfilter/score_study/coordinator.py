"""Versioned execution and repair/resume for trusted local research.

No numerical backend is imported until an eligible endpoint runs. Ordinary
SHA-256 provenance detects accidental reuse of stale results. It is not an
authorization-token or adversarial security protocol.
"""
from __future__ import annotations

import ast
from dataclasses import asdict
import importlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Callable

from .contracts import Registry, digest, seed_pair, validate_result, validate_study


REPO = Path(__file__).resolve().parents[2]


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def source_closure(registry: Registry, repo: Path = REPO) -> dict[str, str]:
    """Hash declared endpoints plus their statically imported local dependencies.

    Providers using dynamic imports must declare those files in source_paths.
    The catalogue and coordinator are included even when every row is blocked.
    """
    pending = list((repo / "bayesfilter/score_study").glob("*.py"))
    pending += [repo / "scripts/run_younis_score_master.py"]
    pending += [repo / p for e in registry.estimators.values() for p in e.source_paths]
    pending += [repo / (e.endpoint.split(":")[0].replace(".", "/") + ".py")
                for e in registry.estimators.values() if e.endpoint]
    hashes = {}
    while pending:
        path = pending.pop().resolve()
        key = str(path.relative_to(repo))
        if key in hashes:
            continue
        if not path.is_file():
            raise ValueError(f"missing declared source dependency: {key}")
        source = path.read_text()
        hashes[key] = digest(source)
        # Importing a submodule executes every parent package initializer too.
        for parent in path.parents:
            if parent == repo:
                break
            initializer = parent / "__init__.py"
            if initializer.is_file():
                pending.append(initializer)
        if path.suffix != ".py":
            continue
        module = path.relative_to(repo).with_suffix("").parts
        for node in ast.walk(ast.parse(source)):
            names = []
            if isinstance(node, ast.Import):
                names = [x.name for x in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    base = module[:-node.level]
                    name = ".".join((*base, *(node.module or "").split("."))).rstrip(".")
                else:
                    name = node.module or ""
                names = [name] + [name + "." + x.name for x in node.names]
            for name in names:
                if name == "bayesfilter" or name.startswith("bayesfilter."):
                    candidate = repo / (name.replace(".", "/") + ".py")
                    if candidate.is_file():
                        pending.append(candidate)
                    initializer = repo / name.replace(".", "/") / "__init__.py"
                    if initializer.is_file():
                        pending.append(initializer)
    return dict(sorted(hashes.items()))


def fingerprint(study: dict, registry: Registry) -> dict:
    return {"study": digest(study), "sources": source_closure(registry),
            "registry": digest(asdict(registry))}


def load_endpoint(name: str) -> Callable:
    module, function = name.split(":")
    return getattr(importlib.import_module(module), function)


def manifest(study: dict, registry: Registry) -> dict:
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, text=True,
                              capture_output=True, check=True).stdout.strip()
    return {"schema": "younis_score_run_v1", "git_commit": revision,
            "command": sys.argv, "python": sys.executable, "python_version": sys.version,
            "plan": study["plan"], "study": study, "fingerprint": fingerprint(study, registry),
            "started_unix": time.time(), "wall_seconds": 0.0,
            "execution_status": "pending", "attempts": [], "rows": {}, "events": []}


def _valid_previous(root: Path, row_state: dict, row: dict, registry: Registry) -> tuple[bool, str]:
    try:
        path = root / row_state["result_path"]
        data = json.loads(path.read_text())
        if digest(data) != row_state["result_digest"]:
            return False, "result digest mismatch"
        validate_result(data, row, registry)
        return True, ""
    except (KeyError, OSError, ValueError, TypeError) as error:
        return False, f"incomplete or invalid result: {error}"


def execute(study: dict, registry: Registry, output: Path, *, resume: bool = False,
            endpoint_loader: Callable = load_endpoint) -> dict:
    decisions = {x["id"]: x for x in validate_study(study, registry)}
    state_path = output / "state.json"
    if resume:
        state = json.loads(state_path.read_text())
        for attempt in state["attempts"]:
            if attempt["status"] == "running":
                # A killed process cannot attest its last wall time. Charge the
                # observed interval conservatively and preserve the old attempt.
                charged = max(0.0, time.time() - attempt["started_unix"])
                attempt.update(status="interrupted_unknown_duration", wall_seconds=charged)
                state["wall_seconds"] += charged
                state["events"].append({"event": "unreported_interruption", "row": attempt["row"],
                                        "conservative_wall_charge": charged})
        current = fingerprint(study, registry)
        if state["fingerprint"] != current:
            state["events"].append({"event": "repair_refresh", "at": time.time(),
                                    "old_fingerprint": state["fingerprint"],
                                    "new_fingerprint": current,
                                    "reason": "source, registry or study changed; invalidate previous rows"})
            for old in state["rows"].values():
                old["execution_status"] = "invalidated"
            state["fingerprint"] = current
            state["study"] = study
    else:
        output.mkdir(parents=True, exist_ok=False)
        state = manifest(study, registry)
        write_json(output / "initial-manifest.json", state)
    start = time.monotonic()
    completed = set()
    pending = {r["id"]: r for r in study["rows"]}
    for row_id, row in list(pending.items()):
        previous = state["rows"].get(row_id, {})
        if previous.get("execution_status") == "complete":
            valid, reason = _valid_previous(output, previous, row, registry)
            if valid:
                completed.add(row_id)
                pending.pop(row_id)
            else:
                previous["execution_status"] = "invalidated"
                state["events"].append({"event": "invalidated_evidence", "row": row_id, "reason": reason})
    # Invalidating a provider also invalidates completed consumers.
    changed = True
    while changed:
        changed = False
        for row in study["rows"]:
            if row["id"] in completed and set(row.get("depends_on", [])) - completed:
                completed.remove(row["id"])
                pending[row["id"]] = row
                state["rows"][row["id"]]["execution_status"] = "invalidated"
                changed = True
    baseline_wall = state["wall_seconds"]
    try:
        while pending:
            ready = [r for r in pending.values() if not set(r.get("depends_on", [])) & pending.keys()]
            if not ready:
                raise ValueError("cyclic dependencies after validation")
            for row in sorted(ready, key=lambda r: r["id"]):
                row_id = row["id"]
                pending.pop(row_id)
                decision = decisions[row_id]
                reasons = list(decision["reasons"])
                blocked = set(row.get("depends_on", [])) - completed
                if blocked:
                    reasons.append(f"prerequisite rows not complete: {sorted(blocked)}")
                if decision["status"] != "runnable" or reasons:
                    state["rows"][row_id] = {"execution_status": "deferred" if decision["status"] == "deferred" else "blocked",
                                             "engineering_status": "not_checked", "numerical_validity": "not_checked",
                                             "scientific_decision": "not_evaluated", "reasons": reasons}
                    write_json(state_path, state)
                    continue
                attempts = [a for a in state["attempts"] if a["row"] == row_id]
                used_wall = baseline_wall + time.monotonic() - start
                if (len(state["attempts"]) >= study["budget"]["max_attempts"] or
                        len(attempts) >= study["budget"]["max_attempts_per_row"] or
                        used_wall >= study["budget"]["wall_seconds"]):
                    state["rows"][row_id] = {"execution_status": "budget_exhausted", "reasons": ["initial or repair allocation exhausted"],
                                             "engineering_status": "not_checked", "numerical_validity": "not_checked",
                                             "scientific_decision": "not_evaluated"}
                    continue
                attempt = {"row": row_id, "number": len(attempts) + 1,
                           "status": "running", "started_unix": time.time(),
                           "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                               text=True, capture_output=True, check=True).stdout.strip(),
                           "fingerprint": digest(state["fingerprint"])}
                directory = output / "attempts" / row_id / f"attempt-{attempt['number']:03d}"
                directory.mkdir(parents=True, exist_ok=False)
                attempt["path"] = str(directory.relative_to(output))
                state["attempts"].append(attempt)
                state["rows"][row_id] = {"execution_status": "running", "attempt": attempt["number"]}
                write_json(directory / "input.json", {"row": row, "study": study, "fingerprint": state["fingerprint"]})
                write_json(state_path, state)
                row_start = time.monotonic()
                try:
                    estimator = registry.estimators[row["estimator"]]
                    endpoint = endpoint_loader(estimator.endpoint)
                    context = {"study": study, "output": str(directory), "registry": registry,
                               "seed": seed_pair(master_seed=study["seed"], model=row["model"],
                                   dataset=row["dataset"], replicate=row["replicate"], stream="particles",
                                   coupling_group=row.get("coupling_group", "baseline"))}
                    result = endpoint(row, context)
                    validate_result(result, row, registry)
                    write_json(directory / "result.json", result)
                    attempt["status"] = "complete"
                    state["rows"][row_id] = {
                        "execution_status": "complete", "engineering_status": "pass",
                        "numerical_validity": result["numerical_validity"],
                        "scientific_decision": result["inference_status"],
                        "attempt": attempt["number"], "result_digest": digest(result),
                        "result_path": str((directory / "result.json").relative_to(output))}
                    completed.add(row_id)
                except BaseException as error:
                    attempt["status"] = "interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed"
                    (directory / "error.log").write_text(traceback.format_exc())
                    state["rows"][row_id] = {"execution_status": attempt["status"],
                                             "engineering_status": "failed", "numerical_validity": "not_admitted",
                                             "scientific_decision": "not_evaluated", "reasons": [str(error)],
                                             "attempt": attempt["number"]}
                    if attempt["status"] == "interrupted":
                        raise
                finally:
                    attempt["wall_seconds"] = time.monotonic() - row_start
                    state["wall_seconds"] = baseline_wall + time.monotonic() - start
                    write_json(directory / "attempt.json", attempt)
                    write_json(state_path, state)
    finally:
        state["wall_seconds"] = baseline_wall + time.monotonic() - start
        state["execution_status"] = "complete" if len(completed) == len(study["rows"]) else "incomplete"
        write_json(state_path, state)
        report(output, registry)
    return state


def report(output: Path, registry: Registry) -> dict:
    state = json.loads((output / "state.json").read_text())
    source_current = state["fingerprint"] == fingerprint(state["study"], registry)
    rows = []
    for row in state["study"]["rows"]:
        saved = state["rows"].get(row["id"], {"execution_status": "pending"})
        if saved.get("execution_status") == "complete":
            valid, reason = _valid_previous(output, saved, row, registry)
            if not source_current:
                valid, reason = False, "source/registry changed since execution"
            if not valid:
                saved = {**saved, "execution_status": "invalid_artifact", "reasons": [reason]}
        rows.append({"id": row["id"], "proposal": row["proposal"], **saved})
    summary = {"execution_status": "complete" if all(r["execution_status"] == "complete" for r in rows) else "incomplete", "rows": rows,
               "attempts": len(state["attempts"]), "wall_seconds": state["wall_seconds"],
               "statistically_supported_ranking": False, "default_ready": False}
    write_json(output / "report.json", summary)
    lines = ["# Score master execution", "", "| Row | Proposal | Execution | Scientific inference |",
             "|---|---|---|---|"]
    for row in rows:
        lines.append(f"| {row['id']} | {row['proposal']} | {row['execution_status']} | {row.get('scientific_decision', 'not_evaluated')} |")
    lines += ["", "Execution completeness is separate from numerical validity and scientific ranking.",
              "Failed and blocked rows remain visible. This orchestration report establishes no superiority or default readiness.", ""]
    (output / "report.md").write_text("\n".join(lines))
    return summary
