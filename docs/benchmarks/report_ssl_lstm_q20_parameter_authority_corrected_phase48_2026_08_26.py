"""Report the paired identity versus independent-proposal MH diagnostic.

This CPU-only report validates the v3.0 fixture and boundary receipts, checks
the independent-MH acceptance bookkeeping, and emits descriptive replicate
spreads. It does not turn a finite spread relation into convergence,
whitening, posterior, or method-superiority evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
    raise RuntimeError("Phase 48 report requires CUDA_VISIBLE_DEVICES=-1")
if os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH") != "true":
    raise RuntimeError("Phase 48 report requires TF_FORCE_GPU_ALLOW_GROWTH=true")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf

RUNNER = Path(__file__).resolve()
PLAN = ROOT / "docs/plans/bayesfilter-ssl-lstm-q20-parameter-authority-corrected-continuation-2026-08-25.md"
EXPECTED_VERSION = "v3.0-independent-proposal-mutation"
EXPECTED_TARGET = "9a86e60081f1b9cd288dbdb1dcbe1e9a5b5e23d9b5ef97afdb72ee95c23d7278"
EXPECTED_SCHEMA = "bayesfilter.ssl_lstm.q20.corrected_theta_independent_mh_boundary.v1"
EXPECTED_FIXTURE_SCHEMA = "bayesfilter.ssl_lstm.q20.corrected_theta_independent_mh_fixture.v1"
EXPECTED_FIXTURE_STATUS = "PASS_V3_0_INDEPENDENT_MH_FIXTURE"
EXPECTED_BOUNDARY_STATUS = "PASS_V3_0_INDEPENDENT_MH_BOUNDARY"
EXPECTED_ARM_STATUS = "PASS_V3_0_MUTATION_ARM"


class Phase48ReportError(RuntimeError):
    """Raised when a v3.0 report cannot be validated."""


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, tf.TensorShape):
        return [_safe(item) for item in value.as_list()]
    if isinstance(value, tf.dtypes.DType):
        return value.name
    if hasattr(value, "numpy"):
        return _safe(value.numpy())
    if hasattr(value, "tolist"):
        return _safe(value.tolist())
    if hasattr(value, "item"):
        return _safe(value.item())
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise Phase48ReportError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(_safe(payload), sort_keys=True, indent=2, allow_nan=False) + "\n").encode("ascii")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(encoded)
    temporary.replace(path)


def _load(root: Path, name: str = "result.json") -> tuple[Path, Mapping[str, Any]]:
    if root.is_absolute() or ".." in root.parts:
        raise Phase48ReportError(f"path must be repository-relative: {root}")
    path = ROOT / root / name
    if not path.is_file():
        raise Phase48ReportError(f"missing receipt: {path}")
    return path, json.loads(path.read_text(encoding="utf-8"))


def _finite(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite(item) for item in value)
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return bool(tf.math.is_finite(tf.constant(float(value), tf.float64)).numpy())
    return True


def _arm_summary(arm: Mapping[str, Any]) -> Mapping[str, Any]:
    summary = arm.get("final_summary")
    if not isinstance(summary, Mapping):
        raise Phase48ReportError("mutation arm has no final summary")
    required = (
        "theta_mean_0",
        "covariance_offdiag_max_abs",
        "negative_mode_fraction",
        "root_count",
        "weighted_ess_fraction",
    )
    for key in required:
        if key not in summary or not _finite(summary[key]):
            raise Phase48ReportError(f"non-finite or missing summary field: {key}")
    return {key: summary[key] for key in required}


def _spread(rows: list[Mapping[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows]
    return max(values) - min(values)


def _mutation_receipt(arm: Mapping[str, Any], *, require_independent: bool) -> Mapping[str, Any]:
    stages = arm.get("stages")
    if not isinstance(stages, list) or len(stages) != 5:
        raise Phase48ReportError("expected five annealing stage rows")
    active = []
    for stage in stages:
        if not _finite(stage):
            raise Phase48ReportError("non-finite stage diagnostic")
        mutation = stage.get("mutation")
        if not isinstance(mutation, Mapping):
            raise Phase48ReportError("stage has no mutation receipt")
        if int(mutation.get("accepted_invalid_count", 0)) != 0:
            raise Phase48ReportError("an invalid candidate was accepted")
        if int(mutation.get("steps", 0)) > 0:
            if require_independent and mutation.get("kernel") != "independent_mh":
                raise Phase48ReportError("independent arm contains a non-independent mutation kernel")
            active.append(mutation)
    if require_independent and len(active) != 4:
        raise Phase48ReportError("independent arm does not have four active mutation stages")
    denominator = max(1, len(active))
    return {
        "active_stage_count": len(active),
        "acceptance_rate_mean": float(sum(float(row["acceptance_rate"]) for row in active) / denominator),
        "move_fraction_mean": float(sum(float(row["move_fraction"]) for row in active) / denominator),
        "candidate_safe_fraction_mean": float(sum(float(row.get("candidate_safe_fraction", 0.0)) for row in active) / denominator),
        "mean_displacement_mean": float(sum(float(row["mean_displacement"]) for row in active) / denominator),
        "invalid_candidate_count": int(sum(int(row["invalid_candidate_count"]) for row in active)),
        "accepted_invalid_count": int(sum(int(row.get("accepted_invalid_count", 0)) for row in active)),
        "stage_rows": active,
    }


def _markdown(result: Mapping[str, Any]) -> str:
    lines = [
        "# v3.0 Paired Identity/Independent-MH Report",
        "",
        f"Status: `{result['status']}`",
        f"Branch: `{result['branch']}` (descriptive only)",
        "",
        "| Replicate | Identity mean0 | Independent-MH mean0 | Identity ESS | Independent-MH ESS | Identity neg | Independent-MH neg |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["replicate_rows"]:
        lines.append(
            f"| {row['replicate']} | {float(row['identity']['theta_mean_0']):.6f} | "
            f"{float(row['independent_mh']['theta_mean_0']):.6f} | "
            f"{float(row['identity']['weighted_ess_fraction']):.6f} | "
            f"{float(row['independent_mh']['weighted_ess_fraction']):.6f} | "
            f"{float(row['identity']['negative_mode_fraction']):.6f} | "
            f"{float(row['independent_mh']['negative_mode_fraction']):.6f} |"
        )
    lines.extend(
        [
            "",
            "Raw paired mutation diagnostics only; no convergence, whitening, posterior, HMC, or LEDH claim.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-root", required=True, type=Path)
    parser.add_argument("--boundary-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    paths = (args.fixture_root, args.boundary_root, args.output_root)
    if any(path.is_absolute() or ".." in path.parts for path in paths):
        raise Phase48ReportError("all paths must be repository-relative")
    output = ROOT / args.output_root
    if output.exists():
        raise Phase48ReportError(f"refusing to overwrite {output}")
    started = time.perf_counter()
    fixture_path, fixture = _load(args.fixture_root)
    boundary_path, boundary = _load(args.boundary_root)
    if fixture.get("schema") != EXPECTED_FIXTURE_SCHEMA or fixture.get("status") != EXPECTED_FIXTURE_STATUS:
        raise Phase48ReportError("independent-MH fixture is not passing")
    if (
        boundary.get("schema") != EXPECTED_SCHEMA
        or boundary.get("status") != EXPECTED_BOUNDARY_STATUS
        or boundary.get("plan_version") != EXPECTED_VERSION
        or boundary.get("target_signature") != EXPECTED_TARGET
    ):
        raise Phase48ReportError("q=20 independent-MH boundary is not passing")
    replicates = boundary.get("replicates")
    if not isinstance(replicates, list) or len(replicates) != 3:
        raise Phase48ReportError("expected exactly three paired replicates")
    rows: list[Mapping[str, Any]] = []
    for index, replicate in enumerate(replicates, start=1):
        if replicate.get("replicate") != index:
            raise Phase48ReportError("replicate numbering is not deterministic")
        paired = replicate.get("paired", {})
        required_pairing = (
            "same_initial_cloud",
            "same_resampling_seeds",
            "phase47_initial_cloud_reproduced",
            "phase47_identity_final_tensors_reproduced",
        )
        if any(paired.get(key) is not True for key in required_pairing):
            raise Phase48ReportError(f"pairing/replay gate failed for replicate {index}")
        identity = replicate.get("identity", {})
        independent = replicate.get("independent_mh", {})
        if identity.get("status") != EXPECTED_ARM_STATUS or independent.get("status") != EXPECTED_ARM_STATUS:
            raise Phase48ReportError(f"mutation arm failed for replicate {index}")
        if not _finite(identity) or not _finite(independent):
            raise Phase48ReportError(f"non-finite arm receipt for replicate {index}")
        identity_mutation = _mutation_receipt(identity, require_independent=False)
        independent_mutation = _mutation_receipt(independent, require_independent=True)
        rows.append(
            {
                "replicate": index,
                "pilot_root": replicate.get("pilot_root"),
                "pilot_sha256": replicate.get("pilot_sha256"),
                "identity": _arm_summary(identity),
                "independent_mh": _arm_summary(independent),
                "identity_mutation": identity_mutation,
                "independent_mh_mutation": independent_mutation,
            }
        )
    metrics = (
        "theta_mean_0",
        "covariance_offdiag_max_abs",
        "negative_mode_fraction",
        "root_count",
        "weighted_ess_fraction",
    )
    identity_spreads = {key: _spread([row["identity"] for row in rows], key) for key in metrics}
    independent_spreads = {key: _spread([row["independent_mh"] for row in rows], key) for key in metrics}
    reductions = {key: independent_spreads[key] <= identity_spreads[key] for key in metrics}
    primary_metrics = ("theta_mean_0", "negative_mode_fraction", "covariance_offdiag_max_abs")
    branch = (
        "independent_mh_reduces_between_bank_variability_descriptive"
        if all(reductions[key] for key in primary_metrics)
        else "independent_mh_does_not_reduce_variability"
    )
    result = {
        "schema": "bayesfilter.ssl_lstm.q20.corrected_theta_independent_mh_report.v1",
        "status": "PASS_V3_0_INDEPENDENT_MH_REPORT",
        "plan_version": EXPECTED_VERSION,
        "role": "read_only_paired_identity_vs_independent_mh_report",
        "target_signature": EXPECTED_TARGET,
        "fixture_status": fixture["status"],
        "boundary_status": boundary["status"],
        "branch": branch,
        "branch_is_statistical_ranking": False,
        "replicate_rows": rows,
        "identity_spreads": identity_spreads,
        "independent_mh_spreads": independent_spreads,
        "spread_reductions_or_equal": reductions,
        "decision_table": [
            {
                "decision": "retain_theta_target",
                "status": "pass",
                "primary_criterion": "fixture/boundary/pairing/replay gates",
                "veto": "none",
                "next_action": "retain theta authority",
                "not_concluded": "posterior correctness",
            },
            {
                "decision": "promote_IID_whitening",
                "status": "veto",
                "primary_criterion": "finite mutation clouds",
                "veto": "finite replicate comparison is not a Gaussian law",
                "next_action": "keep whitening closed",
                "not_concluded": "IID Gaussian law",
            },
            {
                "decision": "promote_independent_MH_as_default",
                "status": "defer",
                "primary_criterion": "descriptive paired spread",
                "veto": "three replicates and no uncertainty model",
                "next_action": "retain independent MH as a role-limited candidate only",
                "not_concluded": "superiority or default readiness",
            },
        ],
        "inference_status": {
            "hard_veto_screen": "passed",
            "statistically_supported_ranking": "none",
            "descriptive_differences": "paired identity/independent-MH support and mutation diagnostics",
            "default_readiness": "not_ready",
            "next_evidence": "longer independent validation with uncertainty and downstream target checks",
        },
        "red_team": {
            "strongest_alternative": "the fixed defensive proposal itself has support bias, so mutation can move between proposal components without repairing the target representation",
            "overturning_evidence": "fresh proposal/support construction with stable downstream behavior under uncertainty-aware validation",
            "weakest_evidence": "three replicates and two independent proposals per nonterminal stage without Monte Carlo intervals",
        },
        "nonclaims": [
            "No finite-run convergence, posterior correctness, IID whitening, exhaustive mode discovery, HMC, canonical LEDH, superiority, or default claim.",
            "Acceptance and spread relations are descriptive only.",
            "No mutation cloud was used to train NeuTra or select an objective.",
        ],
        "sources": {
            "fixture_root": args.fixture_root,
            "boundary_root": args.boundary_root,
            "fixture_sha256": _sha(fixture_path),
            "boundary_sha256": _sha(boundary_path),
        },
        "run_manifest": {
            "program": PLAN.as_posix(),
            "runner": RUNNER.as_posix(),
            "command": " ".join(sys.argv),
            "git_commit": subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=ROOT, text=True).strip(),
            "git_dirty": bool(subprocess.check_output(("git", "status", "--porcelain"), cwd=ROOT, text=True).strip()),
            "python": sys.executable,
            "python_version": platform.python_version(),
            "tensorflow": tf.__version__,
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "tf_force_gpu_allow_growth": os.environ["TF_FORCE_GPU_ALLOW_GROWTH"],
            "gpu_hidden_intentionally": True,
            "jit_compile": False,
            "wall_seconds": time.perf_counter() - started,
            "source_sha256": {"plan": _sha(PLAN), "runner": _sha(RUNNER)},
        },
    }
    _write_json(output / "result.json", result)
    (output / "result.md").write_text(_markdown(result), encoding="ascii")
    print(json.dumps({"status": result["status"], "branch": branch, "output_root": args.output_root.as_posix()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
