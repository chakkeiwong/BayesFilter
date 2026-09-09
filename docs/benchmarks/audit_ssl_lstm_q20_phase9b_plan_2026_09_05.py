#!/usr/bin/env python3
"""Run the bounded, non-HMC Phase 9B source and authority preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "docs/plans/bayesfilter-ssl-lstm-q20-phase9b-sequential-validation-plan-2026-09-05.md"
MASTER = ROOT / "docs/plans/bayesfilter-ssl-lstm-q20-tempered-rkl-transport-ensemble-master-program-2026-09-02.md"
RESET_MEMO = ROOT / "docs/plans/bayesfilter-ssl-lstm-q20-tempered-rkl-transport-ensemble-master-program-reset-memo-2026-09-06.md"
SUPPLIED_RESULT = ROOT / "docs/plans/bayesfilter-ssl-lstm-q20-factor-route-fresh-tuning-admission-result-2026-09-04.md"
SOURCE_ROOT = ROOT / "docs/plans/artifacts/ssl-lstm-q20-factor-route-fresh-tuning-2026-09-04/source-sync-20260905T203000Z"
SOURCE_MANIFEST = SOURCE_ROOT / "run_manifest.json"
SOURCE_AUDIT = SOURCE_ROOT / "source-sync-audit.json"

TARGET_SIGNATURE = "9a86e60081f1b9cd288dbdb1dcbe1e9a5b5e23d9b5ef97afdb72ee95c23d7278"
FACTOR_BACKEND = "tensorflow_eigh_strict_factor_cached"
EXPECTED_MANIFEST_SHA256 = "e0225382192ceb9da1e075cb9c7a91ed424e2c5c67adcfec7a0f34c05003a4e1"
EXPECTED_AUDIT_SHA256 = "9848779a1fe1611f3b3acfb666bec8b08bef6a42296fe30e84d3762d892c5933"
SCOPES = tuple(
    (chart_index, beta)
    for chart_index in range(2)
    for beta in (0.0, 0.5, 1.0)
)


def _strict_json(path: Path) -> Mapping[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value} in {path}")

    payload = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=reject_constant,
    )
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON root is not an object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _git(command: tuple[str, ...]) -> str:
    try:
        return subprocess.check_output(
            command,
            cwd=ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"unavailable:{type(exc).__name__}"


def _relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _scope_paths(chart_index: int, beta: float) -> tuple[Path, Path]:
    scope_root = SOURCE_ROOT / f"chart-{chart_index}" / f"beta-{beta:g}"
    return scope_root / "chart_checkpoint.json", scope_root / "fixed_transport_hmc_tuning_result.json"


def _check_file(path: Path, failures: list[str]) -> None:
    if not path.is_file():
        failures.append(f"missing_file:{_relative(path)}")


def _run(output_dir: Path) -> int:
    if output_dir.exists():
        raise RuntimeError(f"refusing to overwrite output directory: {output_dir}")
    output_dir.mkdir(parents=True)

    failures: list[str] = []
    checks: dict[str, Any] = {
        "session_date_boundary": {
            "applicable_session_date": "2026-09-05",
            "future_dated_documents": [
                _relative(RESET_MEMO),
                _relative(SUPPLIED_RESULT),
            ],
            "future_documents_consumed_as_authority": False,
        },
        "required_paths": {},
        "scope_checks": [],
        "source_tree_checks": {},
    }

    required_paths = (PLAN, MASTER, RESET_MEMO, SUPPLIED_RESULT, SOURCE_MANIFEST, SOURCE_AUDIT)
    for path in required_paths:
        _check_file(path, failures)
        checks["required_paths"][_relative(path)] = path.is_file()

    if SOURCE_MANIFEST.is_file():
        try:
            manifest = _strict_json(SOURCE_MANIFEST)
            checks["source_manifest"] = {
                "status": manifest.get("status"),
                "schema": manifest.get("schema"),
                "target_signature": manifest.get("target_signature"),
                "principal_sqrt_backend": manifest.get("principal_sqrt_backend"),
                "profile_id": manifest.get("profile_id"),
                "manifest_hash": manifest.get("manifest_hash"),
            }
            if manifest.get("status") != "PASS_PHASE9A_SCOPE_PREFLIGHT":
                failures.append("source_manifest_status_mismatch")
            if manifest.get("target_signature") != TARGET_SIGNATURE:
                failures.append("source_manifest_target_signature_mismatch")
            if manifest.get("principal_sqrt_backend") != FACTOR_BACKEND:
                failures.append("source_manifest_factor_backend_mismatch")
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
            failures.append(f"source_manifest_unreadable:{type(exc).__name__}")

    if SOURCE_AUDIT.is_file():
        try:
            audit = _strict_json(SOURCE_AUDIT)
            checks["source_audit"] = {
                "status": audit.get("status"),
                "schema": audit.get("schema"),
                "target_signature": audit.get("target_signature"),
                "principal_sqrt_backend": audit.get("principal_sqrt_backend"),
                "scope_count": len(audit.get("scope_summaries", ())),
            }
            if audit.get("status") != "PASS_FACTOR_SOURCE-SYNC_AUDIT":
                failures.append("source_audit_status_mismatch")
            if audit.get("principal_sqrt_backend") != FACTOR_BACKEND:
                failures.append("source_audit_factor_backend_mismatch")
            if len(audit.get("scope_summaries", ())) != len(SCOPES):
                failures.append("source_audit_scope_count_mismatch")
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
            failures.append(f"source_audit_unreadable:{type(exc).__name__}")

    if SOURCE_MANIFEST.is_file():
        observed = _sha256(SOURCE_MANIFEST)
        checks["source_manifest_sha256"] = {
            "expected": EXPECTED_MANIFEST_SHA256,
            "observed": observed,
            "passed": observed == EXPECTED_MANIFEST_SHA256,
        }
        if observed != EXPECTED_MANIFEST_SHA256:
            failures.append("source_manifest_sha256_mismatch")
    if SOURCE_AUDIT.is_file():
        observed = _sha256(SOURCE_AUDIT)
        checks["source_audit_sha256"] = {
            "expected": EXPECTED_AUDIT_SHA256,
            "observed": observed,
            "passed": observed == EXPECTED_AUDIT_SHA256,
        }
        if observed != EXPECTED_AUDIT_SHA256:
            failures.append("source_audit_sha256_mismatch")

    for chart_index, beta in SCOPES:
        checkpoint_path, tuning_path = _scope_paths(chart_index, beta)
        row: dict[str, Any] = {
            "chart_index": chart_index,
            "beta": beta,
            "checkpoint": _relative(checkpoint_path),
            "tuning_artifact": _relative(tuning_path),
        }
        _check_file(checkpoint_path, failures)
        _check_file(tuning_path, failures)
        try:
            checkpoint_wrapper = _strict_json(checkpoint_path)
            checkpoint = checkpoint_wrapper.get("checkpoint")
            if not isinstance(checkpoint, Mapping):
                raise ValueError("checkpoint field is not an object")
            scope = checkpoint.get("checkpoint_scope")
            row["checkpoint_status"] = checkpoint_wrapper.get("status")
            row["checkpoint_scope"] = scope
            if checkpoint_wrapper.get("status") != "PASS_FRESH_CHART_CHECKPOINT":
                failures.append(f"checkpoint_status_mismatch:chart{chart_index}:beta{beta:g}")
            if checkpoint.get("target_signature") != TARGET_SIGNATURE:
                failures.append(f"checkpoint_target_signature_mismatch:chart{chart_index}:beta{beta:g}")
            if not isinstance(scope, Mapping) or scope.get("principal_sqrt_backend") != FACTOR_BACKEND:
                failures.append(f"checkpoint_backend_mismatch:chart{chart_index}:beta{beta:g}")
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
            failures.append(f"checkpoint_unreadable:chart{chart_index}:beta{beta:g}:{type(exc).__name__}")
        try:
            tuning = _strict_json(tuning_path)
            config = tuning.get("config")
            kernel = tuning.get("final_kernel_payload")
            row["tuning_status"] = tuning.get("passed")
            row["tuning_policy"] = tuning.get("tuning_policy")
            row["target_scope"] = config.get("target_scope") if isinstance(config, Mapping) else None
            row["selected_step_size"] = kernel.get("step_size") if isinstance(kernel, Mapping) else None
            row["selected_num_leapfrog_steps"] = kernel.get("num_leapfrog_steps") if isinstance(kernel, Mapping) else None
            if tuning.get("passed") is not True:
                failures.append(f"tuning_not_passed:chart{chart_index}:beta{beta:g}")
            if tuning.get("tuning_policy") != "measured_joint_grid_v1":
                failures.append(f"tuning_policy_mismatch:chart{chart_index}:beta{beta:g}")
            if not isinstance(config, Mapping) or not str(config.get("target_scope", "")):
                failures.append(f"tuning_scope_missing:chart{chart_index}:beta{beta:g}")
            if not isinstance(kernel, Mapping) or int(kernel.get("num_leapfrog_steps", 0)) < 2:
                failures.append(f"tuning_kernel_invalid:chart{chart_index}:beta{beta:g}")
        except (OSError, UnicodeError, ValueError, TypeError, json.JSONDecodeError) as exc:
            failures.append(f"tuning_unreadable:chart{chart_index}:beta{beta:g}:{type(exc).__name__}")
        checks["scope_checks"].append(row)

    source_paths = (
        ROOT / "bayesfilter/inference/neutra_hmc.py",
        ROOT / "bayesfilter/inference/fixed_transport_hmc_tuning_tf.py",
        ROOT / "bayesfilter/inference/tempered_transitions_tf.py",
        ROOT / "bayesfilter/inference/tempered_target_tf.py",
    )
    for path in source_paths:
        _check_file(path, failures)
    checks["source_tree_checks"] = {
        "paths": [_relative(path) for path in source_paths],
        "all_present": all(path.is_file() for path in source_paths),
        "sequential_policy_marker": "bayesfilter_neutra_sequential_hmc_v1"
        in (ROOT / "bayesfilter/inference/neutra_hmc_policy.py").read_text(encoding="utf-8"),
        "shared_controller_marker": "run_sequential_neutra_hmc"
        in (ROOT / "bayesfilter/inference/neutra_hmc.py").read_text(encoding="utf-8"),
        "supported_fixed_tuner_marker": "tune_fixed_transport_hmc_kernel"
        in (ROOT / "bayesfilter/inference/fixed_transport_hmc_tuning_tf.py").read_text(encoding="utf-8"),
    }
    if not checks["source_tree_checks"]["sequential_policy_marker"]:
        failures.append("sequential_policy_marker_missing")
    if not checks["source_tree_checks"]["shared_controller_marker"]:
        failures.append("shared_sequential_controller_missing")
    if not checks["source_tree_checks"]["supported_fixed_tuner_marker"]:
        failures.append("supported_fixed_transport_tuner_missing")

    status_output = _git(("git", "status", "--porcelain"))
    git_payload = {
        "commit": _git(("git", "rev-parse", "HEAD")),
        "status_sha256": hashlib.sha256(status_output.encode("utf-8")).hexdigest(),
        "worktree_dirty": bool(status_output),
    }
    checks["git"] = git_payload
    checks["claim_boundary"] = "phase9b_p0_source_authority_preflight_only"
    checks["nonclaims"] = [
        "no HMC execution",
        "no posterior samples",
        "no convergence claim",
        "no backend admission from future-dated documents",
        "no chart-quality or whitening claim",
    ]

    manifest = {
        "schema": "bayesfilter.ssl_lstm_q20.phase9b_p0_run_manifest.v1",
        "status": "PASS_PHASE9B_P0_SOURCE_PREFLIGHT" if not failures else "FAIL_PHASE9B_P0_SOURCE_PREFLIGHT",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": list(sys.argv),
        "plan": {"path": _relative(PLAN), "sha256": _sha256(PLAN)},
        "master": {"path": _relative(MASTER), "sha256": _sha256(MASTER)},
        "output_dir": _relative(output_dir),
        "target_signature": TARGET_SIGNATURE,
        "proposed_backend": FACTOR_BACKEND,
        "checks": checks,
        "failures": sorted(set(failures)),
    }
    manifest["manifest_hash"] = _stable_hash(manifest)
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "p0-preflight.json").write_text(
        json.dumps(
            {
                "schema": "bayesfilter.ssl_lstm_q20.phase9b_p0_preflight.v1",
                "status": manifest["status"],
                "manifest_path": _relative(output_dir / "run_manifest.json"),
                "manifest_sha256": _sha256(output_dir / "run_manifest.json"),
                "failures": manifest["failures"],
                "claim_boundary": checks["claim_boundary"],
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": manifest["status"], "output_dir": _relative(output_dir), "failures": manifest["failures"]}, sort_keys=True))
    return 0 if not failures else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    try:
        return _run(args.output_dir.expanduser().resolve())
    except Exception as exc:  # noqa: BLE001 - preserve a bounded diagnostic failure.
        print(json.dumps({"status": "FAIL_PHASE9B_P0_SOURCE_PREFLIGHT", "error_type": type(exc).__name__, "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
