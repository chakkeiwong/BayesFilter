"""Report the v3.1 paired identity versus independent-MH depth diagnostic.

This CPU-only report validates the v3.1 fixture and boundary receipts, then
compares depth eight with the frozen v3.0 depth-two arm. It does not turn a
finite spread relation into convergence, whitening, posterior, or method-
superiority evidence.
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
    raise RuntimeError("Phase 49 report requires CUDA_VISIBLE_DEVICES=-1")
if os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH") != "true":
    raise RuntimeError("Phase 49 report requires TF_FORCE_GPU_ALLOW_GROWTH=true")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf

RUNNER = Path(__file__).resolve()
PLAN = ROOT / "docs/plans/bayesfilter-ssl-lstm-q20-parameter-authority-corrected-continuation-2026-08-25.md"
EXPECTED_TARGET = "9a86e60081f1b9cd288dbdb1dcbe1e9a5b5e23d9b5ef97afdb72ee95c23d7278"
EXPECTED_VERSION = "v3.1-independent-proposal-depth"
EXPECTED_SCHEMA = "bayesfilter.ssl_lstm.q20.corrected_theta_independent_mh_depth_boundary.v1"
EXPECTED_FIXTURE_SCHEMA = "bayesfilter.ssl_lstm.q20.corrected_theta_independent_mh_depth_fixture.v1"
EXPECTED_FIXTURE_STATUS = "PASS_V3_1_INDEPENDENT_MH_DEPTH_FIXTURE"
EXPECTED_BOUNDARY_STATUS = "PASS_V3_1_INDEPENDENT_MH_DEPTH_BOUNDARY"
EXPECTED_ARM_STATUS = "PASS_V3_1_MUTATION_ARM"
EXPECTED_STEPS = 8
PHASE48_REPORT = ROOT / "docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase48-independent-proposal-mutation/report/result.json"
PHASE48_SCHEMA = "bayesfilter.ssl_lstm.q20.corrected_theta_independent_mh_report.v1"
PHASE48_STATUS = "PASS_V3_0_INDEPENDENT_MH_REPORT"
PHASE48_VERSION = "v3.0-independent-proposal-mutation"
METRICS = ("theta_mean_0", "covariance_offdiag_max_abs", "negative_mode_fraction", "root_count", "weighted_ess_fraction")
PRIMARY_METRICS = ("theta_mean_0", "negative_mode_fraction", "covariance_offdiag_max_abs")


class Phase49ReportError(RuntimeError):
    """Raised when a v3.1 report cannot be validated."""


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
        raise Phase49ReportError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(_safe(payload), sort_keys=True, indent=2, allow_nan=False) + "\n").encode("ascii")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(encoded)
    temporary.replace(path)


def _load(root: Path, name: str = "result.json") -> tuple[Path, Mapping[str, Any]]:
    if root.is_absolute() or ".." in root.parts:
        raise Phase49ReportError(f"path must be repository-relative: {root}")
    path = ROOT / root / name
    if not path.is_file():
        raise Phase49ReportError(f"missing receipt: {path}")
    return path, json.loads(path.read_text(encoding="utf-8"))


def _load_frozen_phase48() -> tuple[Path, Mapping[str, Any]]:
    """Load the v3.0 report as an immutable, branch-checked comparator."""
    if not PHASE48_REPORT.is_file():
        raise Phase49ReportError(f"missing frozen Phase 48 report: {PHASE48_REPORT}")
    payload = json.loads(PHASE48_REPORT.read_text(encoding="utf-8"))
    if (
        payload.get("schema") != PHASE48_SCHEMA
        or payload.get("status") != PHASE48_STATUS
        or payload.get("plan_version") != PHASE48_VERSION
        or payload.get("target_signature") != EXPECTED_TARGET
        or payload.get("branch") != "independent_mh_does_not_reduce_variability"
    ):
        raise Phase49ReportError("frozen Phase 48 comparator is stale or has the wrong repair branch")
    rows = payload.get("replicate_rows")
    if not isinstance(rows, list) or len(rows) != 3:
        raise Phase49ReportError("frozen Phase 48 comparator must contain three rows")
    for index, row in enumerate(rows, start=1):
        if row.get("replicate") != index or not isinstance(row.get("independent_mh"), Mapping):
            raise Phase49ReportError(f"frozen Phase 48 comparator row {index} is incomplete")
        _summary_fields(row["independent_mh"])
    return PHASE48_REPORT, payload


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
        raise Phase49ReportError("summary is not a mapping")
    for key in METRICS:
        if key not in summary or not _finite(summary[key]):
            raise Phase49ReportError(f"non-finite or missing summary field: {key}")
    return {key: summary[key] for key in METRICS}


def _arm_summary(arm: Mapping[str, Any]) -> Mapping[str, Any]:
    summary = arm.get("final_summary")
    if not isinstance(summary, Mapping):
        raise Phase49ReportError("mutation arm has no final summary")
    return _summary_fields(summary)


def _spread(rows: list[Mapping[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows]
    return max(values) - min(values)


def _mutation_receipt(
    arm: Mapping[str, Any], *, require_independent: bool, expected_steps: int
) -> Mapping[str, Any]:
    stages = arm.get("stages")
    if not isinstance(stages, list) or len(stages) != 5:
        raise Phase49ReportError("expected five annealing stage rows")
    active = []
    for stage in stages:
        if not _finite(stage):
            raise Phase49ReportError("non-finite stage diagnostic")
        mutation = stage.get("mutation")
        if not isinstance(mutation, Mapping):
            raise Phase49ReportError("stage has no mutation receipt")
        if int(mutation.get("accepted_invalid_count", 0)) != 0:
            raise Phase49ReportError("an invalid candidate was accepted")
        if int(mutation.get("steps", 0)) > 0:
            if not require_independent or mutation.get("kernel") != "independent_mh":
                raise Phase49ReportError("active arm contains an unexpected mutation kernel")
            if int(mutation.get("steps", 0)) != expected_steps:
                raise Phase49ReportError(
                    f"expected {expected_steps} mutation steps, got {mutation.get('steps')}"
                )
            active.append(mutation)
        elif expected_steps == 0 and int(mutation.get("steps", 0)) != 0:
            raise Phase49ReportError("identity arm has an unexpected mutation step count")
    if require_independent and len(active) != 4:
        raise Phase49ReportError("independent arm does not have four active mutation stages")
    if not require_independent and active:
        raise Phase49ReportError("identity arm unexpectedly contains active mutation stages")
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
        "# v3.1 Paired Identity/Independent-MH Depth Report",
        "",
        f"Status: `{result['status']}`",
        f"Branch: `{result['branch']}` (descriptive only)",
        "",
        "| Replicate | Identity mean0 | Depth-8 mean0 | Frozen depth-2 mean0 | Depth-8 ESS | Frozen depth-2 ESS |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["replicate_rows"]:
        lines.append(
            f"| {row['replicate']} | {float(row['identity']['theta_mean_0']):.6f} | "
            f"{float(row['independent_mh_depth8']['theta_mean_0']):.6f} | "
            f"{float(row['phase48_independent_mh_depth2']['theta_mean_0']):.6f} | "
            f"{float(row['independent_mh_depth8']['weighted_ess_fraction']):.6f} | "
            f"{float(row['phase48_independent_mh_depth2']['weighted_ess_fraction']):.6f} |"
        )
    lines.extend(
        [
            "",
            "Finite paired mutation diagnostics only; no convergence, whitening, posterior, HMC, or LEDH claim.",
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
        raise Phase49ReportError("all paths must be repository-relative")
    output = ROOT / args.output_root
    if output.exists():
        raise Phase49ReportError(f"refusing to overwrite {output}")
    started = time.perf_counter()
    fixture_path, fixture = _load(args.fixture_root)
    boundary_path, boundary = _load(args.boundary_root)
    phase48_path, phase48 = _load_frozen_phase48()
    if fixture.get("schema") != EXPECTED_FIXTURE_SCHEMA or fixture.get("status") != EXPECTED_FIXTURE_STATUS:
        raise Phase49ReportError("independent-MH depth fixture is not passing")
    if fixture.get("depth_steps") != EXPECTED_STEPS or not all(fixture.get("gates", {}).values()):
        raise Phase49ReportError("depth fixture does not certify every repeated-step gate")
    if (
        boundary.get("schema") != EXPECTED_SCHEMA
        or boundary.get("status") != EXPECTED_BOUNDARY_STATUS
        or boundary.get("plan_version") != EXPECTED_VERSION
        or boundary.get("target_signature") != EXPECTED_TARGET
        or boundary.get("mh_steps") != EXPECTED_STEPS
    ):
        raise Phase49ReportError("q=20 independent-MH depth boundary is not passing")
    replicates = boundary.get("replicates")
    if not isinstance(replicates, list) or len(replicates) != 3:
        raise Phase49ReportError("expected exactly three paired replicates")
    rows: list[Mapping[str, Any]] = []
    for index, replicate in enumerate(replicates, start=1):
        if replicate.get("replicate") != index:
            raise Phase49ReportError("replicate numbering is not deterministic")
        paired = replicate.get("paired", {})
        required_pairing = (
            "same_initial_cloud",
            "same_resampling_seeds",
            "phase47_initial_cloud_reproduced",
            "phase47_identity_final_tensors_reproduced",
        )
        if any(paired.get(key) is not True for key in required_pairing):
            raise Phase49ReportError(f"pairing/replay gate failed for replicate {index}")
        identity = replicate.get("identity", {})
        depth8 = replicate.get("independent_mh", {})
        if identity.get("status") != EXPECTED_ARM_STATUS or depth8.get("status") != EXPECTED_ARM_STATUS:
            raise Phase49ReportError(f"mutation arm failed for replicate {index}")
        if not _finite(identity) or not _finite(depth8):
            raise Phase49ReportError(f"non-finite arm receipt for replicate {index}")
        identity_mutation = _mutation_receipt(identity, require_independent=False, expected_steps=0)
        depth8_mutation = _mutation_receipt(depth8, require_independent=True, expected_steps=EXPECTED_STEPS)
        comparator = phase48["replicate_rows"][index - 1]
        comparator_summary = _summary_fields(comparator["independent_mh"])
        rows.append(
            {
                "replicate": index,
                "pilot_root": replicate.get("pilot_root"),
                "pilot_sha256": replicate.get("pilot_sha256"),
                "identity": _arm_summary(identity),
                "independent_mh_depth8": _arm_summary(depth8),
                "phase48_independent_mh_depth2": comparator_summary,
                "identity_mutation": identity_mutation,
                "independent_mh_depth8_mutation": depth8_mutation,
            }
        )
    identity_spreads = {key: _spread([row["identity"] for row in rows], key) for key in METRICS}
    depth8_spreads = {key: _spread([row["independent_mh_depth8"] for row in rows], key) for key in METRICS}
    depth2_spreads = {key: _spread([row["phase48_independent_mh_depth2"] for row in rows], key) for key in METRICS}
    reductions = {key: depth8_spreads[key] <= depth2_spreads[key] for key in METRICS}
    branch = (
        "depth8_reduces_between_bank_variability_descriptive"
        if all(reductions[key] for key in PRIMARY_METRICS)
        else "depth8_does_not_reduce_variability"
    )
    result = {
        "schema": "bayesfilter.ssl_lstm.q20.corrected_theta_independent_mh_depth_report.v1",
        "status": "PASS_V3_1_INDEPENDENT_MH_DEPTH_REPORT",
        "plan_version": EXPECTED_VERSION,
        "role": "read_only_paired_identity_vs_independent_mh_depth_report",
        "target_signature": EXPECTED_TARGET,
        "fixture_status": fixture["status"],
        "boundary_status": boundary["status"],
        "phase48_comparator_status": phase48["status"],
        "phase48_comparator_branch": phase48["branch"],
        "branch": branch,
        "branch_is_statistical_ranking": False,
        "replicate_rows": rows,
        "identity_spreads": identity_spreads,
        "depth8_spreads": depth8_spreads,
        "phase48_depth2_spreads": depth2_spreads,
        "spread_reductions_or_equal_vs_depth2": reductions,
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
                "decision": "promote_depth8_MH_as_default",
                "status": "defer",
                "primary_criterion": "paired descriptive spread versus frozen depth-2 comparator",
                "veto": "three replicates and no uncertainty model",
                "next_action": "retain depth eight only as a role-limited candidate",
                "not_concluded": "superiority or default readiness",
            },
        ],
        "inference_status": {
            "hard_veto_screen": "passed",
            "statistically_supported_ranking": "none",
            "descriptive_differences": "paired identity/depth-8 versus frozen depth-2 mutation diagnostics",
            "default_readiness": "not_ready",
            "next_evidence": "proposal-support repair or uncertainty-aware downstream validation, selected by branch",
        },
        "red_team": {
            "strongest_alternative": "the fixed defensive proposal has support bias, so more proposals may change finite spread without correcting the target representation",
            "overturning_evidence": "fresh proposal/support construction with uncertainty-aware downstream target agreement",
            "weakest_evidence": "three paired replicates and a descriptive depth comparison without a Monte Carlo uncertainty model",
        },
        "nonclaims": [
            "No finite-run convergence, posterior correctness, IID whitening, exhaustive mode discovery, HMC, canonical LEDH, superiority, or default claim.",
            "Acceptance, ESS, root, mode, and spread relations are descriptive only.",
            "No mutation cloud was used to train NeuTra or select an objective.",
        ],
        "sources": {
            "fixture_root": args.fixture_root,
            "boundary_root": args.boundary_root,
            "phase48_report": phase48_path,
            "fixture_sha256": _sha(fixture_path),
            "boundary_sha256": _sha(boundary_path),
            "phase48_report_sha256": _sha(phase48_path),
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
                "phase48_report": _sha(phase48_path),
            },
        },
    }
    _write_json(output / "result.json", result)
    (output / "result.md").write_text(_markdown(result), encoding="ascii")
    print(json.dumps({"status": result["status"], "branch": branch, "output_root": args.output_root.as_posix()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
