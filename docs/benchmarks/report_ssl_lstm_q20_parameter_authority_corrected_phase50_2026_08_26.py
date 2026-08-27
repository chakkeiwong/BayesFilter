"""Report the v3.2 paired broadened-proposal support diagnostic.

This CPU-only report validates the exact q-base/r-proposal receipts and
compares the support arm with the frozen Phase-49 depth-eight arm.  It does
not turn finite spread differences into convergence, whitening, posterior, or
method-superiority evidence.
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
    raise RuntimeError("Phase 50 report requires CUDA_VISIBLE_DEVICES=-1")
if os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH") != "true":
    raise RuntimeError("Phase 50 report requires TF_FORCE_GPU_ALLOW_GROWTH=true")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf

RUNNER = Path(__file__).resolve()
PLAN = ROOT / "docs/plans/bayesfilter-ssl-lstm-q20-parameter-authority-corrected-continuation-2026-08-25.md"
PHASE49_REPORT = ROOT / "docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase49-independent-proposal-depth/report/result.json"
EXPECTED_TARGET = "9a86e60081f1b9cd288dbdb1dcbe1e9a5b5e23d9b5ef97afdb72ee95c23d7278"
EXPECTED_VERSION = "v3.2-defensive-proposal-support"
EXPECTED_SCHEMA = "bayesfilter.ssl_lstm.q20.corrected_theta_defensive_support_boundary.v1"
EXPECTED_BOUNDARY_STATUS = "PASS_V3_2_DEFENSIVE_SUPPORT_BOUNDARY"
EXPECTED_FIXTURE_SCHEMA = "bayesfilter.ssl_lstm.q20.corrected_theta_defensive_support_fixture.v1"
EXPECTED_FIXTURE_STATUS = "PASS_V3_2_DEFENSIVE_SUPPORT_FIXTURE"
EXPECTED_ARM_STATUS = "PASS_V3_2_MUTATION_ARM"
EXPECTED_STEPS = 8
EXPECTED_RHO = 0.50
EXPECTED_STD = 4.0
PHASE49_SCHEMA = "bayesfilter.ssl_lstm.q20.corrected_theta_independent_mh_depth_report.v1"
PHASE49_STATUS = "PASS_V3_1_INDEPENDENT_MH_DEPTH_REPORT"
PHASE49_VERSION = "v3.1-independent-proposal-depth"
METRICS = ("theta_mean_0", "covariance_offdiag_max_abs", "negative_mode_fraction", "root_count", "weighted_ess_fraction")
PRIMARY_METRICS = ("theta_mean_0", "negative_mode_fraction", "covariance_offdiag_max_abs")


class Phase50ReportError(RuntimeError):
    """Raised when a v3.2 report cannot be validated."""


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
        raise Phase50ReportError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(_safe(payload), sort_keys=True, indent=2, allow_nan=False) + "\n").encode("ascii")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(encoded)
    temporary.replace(path)


def _load(root: Path, name: str = "result.json") -> tuple[Path, Mapping[str, Any]]:
    if root.is_absolute() or ".." in root.parts:
        raise Phase50ReportError(f"path must be repository-relative: {root}")
    path = ROOT / root / name
    if not path.is_file():
        raise Phase50ReportError(f"missing receipt: {path}")
    return path, json.loads(path.read_text(encoding="utf-8"))


def _load_frozen_phase49() -> tuple[Path, Mapping[str, Any]]:
    if not PHASE49_REPORT.is_file():
        raise Phase50ReportError(f"missing frozen Phase 49 report: {PHASE49_REPORT}")
    payload = json.loads(PHASE49_REPORT.read_text(encoding="utf-8"))
    if (
        payload.get("schema") != PHASE49_SCHEMA
        or payload.get("status") != PHASE49_STATUS
        or payload.get("plan_version") != PHASE49_VERSION
        or payload.get("target_signature") != EXPECTED_TARGET
        or payload.get("branch") != "depth8_does_not_reduce_variability"
    ):
        raise Phase50ReportError("frozen Phase 49 comparator is stale or has the wrong repair branch")
    rows = payload.get("replicate_rows")
    if not isinstance(rows, list) or len(rows) != 3:
        raise Phase50ReportError("frozen Phase 49 comparator must contain three rows")
    for index, row in enumerate(rows, start=1):
        if row.get("replicate") != index or not isinstance(row.get("independent_mh_depth8"), Mapping):
            raise Phase50ReportError(f"frozen Phase 49 comparator row {index} is incomplete")
        _summary_fields(row["independent_mh_depth8"])
    return PHASE49_REPORT, payload


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


def _summary_fields(summary: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(summary, Mapping):
        raise Phase50ReportError("summary is not a mapping")
    for key in METRICS:
        if key not in summary or not _finite(summary[key]):
            raise Phase50ReportError(f"non-finite or missing summary field: {key}")
    return {key: summary[key] for key in METRICS}


def _arm_summary(arm: Mapping[str, Any]) -> Mapping[str, Any]:
    summary = arm.get("final_summary")
    if not isinstance(summary, Mapping):
        raise Phase50ReportError("mutation arm has no final summary")
    return _summary_fields(summary)


def _spread(rows: list[Mapping[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows]
    return max(values) - min(values)


def _mutation_receipt(arm: Mapping[str, Any], *, active: bool) -> Mapping[str, Any]:
    stages = arm.get("stages")
    if not isinstance(stages, list) or len(stages) != 5:
        raise Phase50ReportError("expected five annealing stage rows")
    active_rows = []
    for stage in stages:
        if not _finite(stage):
            raise Phase50ReportError("non-finite stage diagnostic")
        mutation = stage.get("mutation")
        if not isinstance(mutation, Mapping):
            raise Phase50ReportError("stage has no mutation receipt")
        if int(mutation.get("accepted_invalid_count", 0)) != 0:
            raise Phase50ReportError("an invalid candidate was accepted")
        steps = int(mutation.get("steps", 0))
        if active and steps > 0:
            if mutation.get("kernel") != "independent_mh" or steps != EXPECTED_STEPS:
                raise Phase50ReportError("broadened arm has the wrong mutation kernel or depth")
            if not _finite(mutation.get("candidate_broad_fraction")):
                raise Phase50ReportError("broadened arm lacks finite component diagnostics")
            active_rows.append(mutation)
        elif not active and steps != 0:
            raise Phase50ReportError("identity arm unexpectedly contains active mutation stages")
    if active and len(active_rows) != 4:
        raise Phase50ReportError("broadened arm does not have four active mutation stages")
    denominator = max(1, len(active_rows))
    return {
        "active_stage_count": len(active_rows),
        "acceptance_rate_mean": float(sum(float(row["acceptance_rate"]) for row in active_rows) / denominator),
        "move_fraction_mean": float(sum(float(row["move_fraction"]) for row in active_rows) / denominator),
        "candidate_broad_fraction_mean": float(sum(float(row.get("candidate_broad_fraction", 0.0)) for row in active_rows) / denominator),
        "candidate_safe_fraction_mean": float(sum(float(row.get("candidate_safe_fraction", 0.0)) for row in active_rows) / denominator),
        "mean_displacement_mean": float(sum(float(row["mean_displacement"]) for row in active_rows) / denominator),
        "invalid_candidate_count": int(sum(int(row["invalid_candidate_count"]) for row in active_rows)),
        "accepted_invalid_count": int(sum(int(row.get("accepted_invalid_count", 0)) for row in active_rows)),
        "stage_rows": active_rows,
    }


def _markdown(result: Mapping[str, Any]) -> str:
    lines = [
        "# v3.2 Paired Identity/Broadened-Support Report",
        "",
        f"Status: `{result['status']}`",
        f"Branch: `{result['branch']}` (descriptive only)",
        "",
        "| Replicate | Identity mean0 | Broadened mean0 | Phase49 depth8 mean0 | Broadened ESS | Phase49 depth8 ESS |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["replicate_rows"]:
        lines.append(
            f"| {row['replicate']} | {float(row['identity']['theta_mean_0']):.6f} | "
            f"{float(row['broadened_support_mh']['theta_mean_0']):.6f} | "
            f"{float(row['phase49_independent_mh_depth8']['theta_mean_0']):.6f} | "
            f"{float(row['broadened_support_mh']['weighted_ess_fraction']):.6f} | "
            f"{float(row['phase49_independent_mh_depth8']['weighted_ess_fraction']):.6f} |"
        )
    lines.extend(["", "Finite paired support diagnostics only; no convergence, whitening, posterior, HMC, or LEDH claim.", ""])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-root", required=True, type=Path)
    parser.add_argument("--boundary-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    paths = (args.fixture_root, args.boundary_root, args.output_root)
    if any(path.is_absolute() or ".." in path.parts for path in paths):
        raise Phase50ReportError("all paths must be repository-relative")
    output = ROOT / args.output_root
    if output.exists():
        raise Phase50ReportError(f"refusing to overwrite {output}")
    started = time.perf_counter()
    fixture_path, fixture = _load(args.fixture_root)
    boundary_path, boundary = _load(args.boundary_root)
    phase49_path, phase49 = _load_frozen_phase49()
    if fixture.get("schema") != EXPECTED_FIXTURE_SCHEMA or fixture.get("status") != EXPECTED_FIXTURE_STATUS or not all(fixture.get("gates", {}).values()):
        raise Phase50ReportError("defensive-support fixture is not passing")
    if fixture.get("depth_steps") != EXPECTED_STEPS or float(fixture.get("support_rho")) != EXPECTED_RHO or float(fixture.get("support_std")) != EXPECTED_STD:
        raise Phase50ReportError("fixture support parameters or depth do not match the plan")
    if (
        boundary.get("schema") != EXPECTED_SCHEMA
        or boundary.get("status") != EXPECTED_BOUNDARY_STATUS
        or boundary.get("plan_version") != EXPECTED_VERSION
        or boundary.get("target_signature") != EXPECTED_TARGET
        or boundary.get("mh_steps") != EXPECTED_STEPS
        or float(boundary.get("support_rho")) != EXPECTED_RHO
        or float(boundary.get("support_std")) != EXPECTED_STD
    ):
        raise Phase50ReportError("q=20 broadened-support boundary is not passing")
    replicates = boundary.get("replicates")
    if not isinstance(replicates, list) or len(replicates) != 3:
        raise Phase50ReportError("expected exactly three paired replicates")
    rows: list[Mapping[str, Any]] = []
    for index, replicate in enumerate(replicates, start=1):
        if replicate.get("replicate") != index:
            raise Phase50ReportError("replicate numbering is not deterministic")
        paired = replicate.get("paired", {})
        required_pairing = (
            "same_initial_cloud",
            "same_resampling_seeds",
            "phase47_initial_cloud_reproduced",
            "phase47_identity_final_tensors_reproduced",
        )
        if any(paired.get(key) is not True for key in required_pairing):
            raise Phase50ReportError(f"pairing/replay gate failed for replicate {index}")
        identity = replicate.get("identity", {})
        broadened = replicate.get("broadened_support_mh", {})
        if identity.get("status") != EXPECTED_ARM_STATUS or broadened.get("status") != EXPECTED_ARM_STATUS:
            raise Phase50ReportError(f"mutation arm failed for replicate {index}")
        if not _finite(identity) or not _finite(broadened):
            raise Phase50ReportError(f"non-finite arm receipt for replicate {index}")
        identity_mutation = _mutation_receipt(identity, active=False)
        broadened_mutation = _mutation_receipt(broadened, active=True)
        comparator_summary = _summary_fields(phase49["replicate_rows"][index - 1]["independent_mh_depth8"])
        rows.append(
            {
                "replicate": index,
                "pilot_root": replicate.get("pilot_root"),
                "pilot_sha256": replicate.get("pilot_sha256"),
                "identity": _arm_summary(identity),
                "broadened_support_mh": _arm_summary(broadened),
                "phase49_independent_mh_depth8": comparator_summary,
                "identity_mutation": identity_mutation,
                "broadened_support_mh_mutation": broadened_mutation,
            }
        )
    identity_spreads = {key: _spread([row["identity"] for row in rows], key) for key in METRICS}
    support_spreads = {key: _spread([row["broadened_support_mh"] for row in rows], key) for key in METRICS}
    phase49_spreads = {key: _spread([row["phase49_independent_mh_depth8"] for row in rows], key) for key in METRICS}
    reductions = {key: support_spreads[key] <= phase49_spreads[key] for key in METRICS}
    branch = "support_broadened_reduces_between_bank_variability_descriptive" if all(reductions[key] for key in PRIMARY_METRICS) else "support_broadened_does_not_reduce_variability"
    result = {
        "schema": "bayesfilter.ssl_lstm.q20.corrected_theta_defensive_support_report.v1",
        "status": "PASS_V3_2_DEFENSIVE_SUPPORT_REPORT",
        "plan_version": EXPECTED_VERSION,
        "role": "read_only_paired_identity_vs_broadened_support_report",
        "target_signature": EXPECTED_TARGET,
        "fixture_status": fixture["status"],
        "boundary_status": boundary["status"],
        "phase49_comparator_status": phase49["status"],
        "phase49_comparator_branch": phase49["branch"],
        "support_rho": EXPECTED_RHO,
        "support_std": EXPECTED_STD,
        "branch": branch,
        "branch_is_statistical_ranking": False,
        "replicate_rows": rows,
        "identity_spreads": identity_spreads,
        "support_spreads": support_spreads,
        "phase49_depth8_spreads": phase49_spreads,
        "spread_reductions_or_equal_vs_phase49_depth8": reductions,
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
                "decision": "promote_broadened_support_as_default",
                "status": "defer",
                "primary_criterion": "paired descriptive spread versus frozen Phase49 depth-eight comparator",
                "veto": "three replicates and no uncertainty model",
                "next_action": "retain support broadening only as a role-limited candidate",
                "not_concluded": "superiority or default readiness",
            },
        ],
        "inference_status": {
            "hard_veto_screen": "passed",
            "statistically_supported_ranking": "none",
            "descriptive_differences": "paired identity/broadened-support versus frozen Phase49 depth-eight diagnostics",
            "default_readiness": "not_ready",
            "next_evidence": "proposal-geometry repair or uncertainty-aware downstream validation, selected by branch",
        },
        "red_team": {
            "strongest_alternative": "the broad component changes finite proposal coverage without fixing target representation or mode mass",
            "overturning_evidence": "fresh support construction with uncertainty-aware downstream target agreement",
            "weakest_evidence": "three paired replicates and a descriptive spread comparison without a Monte Carlo uncertainty model",
        },
        "nonclaims": [
            "No finite-run convergence, posterior correctness, IID whitening, exhaustive mode discovery, HMC, canonical LEDH, superiority, or default claim.",
            "Acceptance, ESS, root, mode, and spread relations are descriptive only.",
            "No mutation cloud was used to train NeuTra or select an objective.",
        ],
        "sources": {
            "fixture_root": args.fixture_root,
            "boundary_root": args.boundary_root,
            "phase49_report": phase49_path,
            "fixture_sha256": _sha(fixture_path),
            "boundary_sha256": _sha(boundary_path),
            "phase49_report_sha256": _sha(phase49_path),
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
            "source_sha256": {
                "plan": _sha(PLAN),
                "runner": _sha(RUNNER),
                "phase49_report": _sha(phase49_path),
            },
        },
    }
    _write_json(output / "result.json", result)
    (output / "result.md").write_text(_markdown(result), encoding="ascii")
    print(json.dumps({"status": result["status"], "branch": branch, "output_root": args.output_root.as_posix()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
