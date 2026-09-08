"""Report paired identity versus MH finite-support mutation diagnostics.

The report is CPU-only and read-only. It validates the analytic fixture and
the q=20 boundary receipt, then reports raw replicate spreads.  No descriptive
spread relation is promoted to convergence, whitening, or posterior evidence.
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
    raise RuntimeError("Phase 47 report requires CUDA_VISIBLE_DEVICES=-1")
if os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH") != "true":
    raise RuntimeError("Phase 47 report requires TF_FORCE_GPU_ALLOW_GROWTH=true")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf

RUNNER = Path(__file__).resolve()
PLAN = ROOT / "docs/plans/bayesfilter-ssl-lstm-q20-parameter-authority-corrected-continuation-2026-08-25.md"
EXPECTED_VERSION = "v2.9-invariant-mutation-diagnostic"
EXPECTED_TARGET = "9a86e60081f1b9cd288dbdb1dcbe1e9a5b5e23d9b5ef97afdb72ee95c23d7278"
EXPECTED_SCHEMA = "bayesfilter.ssl_lstm.q20.corrected_theta_mutation_boundary.v1"
EXPECTED_FIXTURE_SCHEMA = "bayesfilter.ssl_lstm.q20.corrected_theta_mh_fixture.v1"
EXPECTED_FIXTURE_STATUS = "PASS_V2_9_MH_FIXTURE"
EXPECTED_BOUNDARY_STATUS = "PASS_V2_9_MUTATION_BOUNDARY"


class Phase47ReportError(RuntimeError):
    """Raised when a mutation report cannot be validated."""


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
        raise Phase47ReportError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(_safe(payload), sort_keys=True, indent=2, allow_nan=False) + "\n").encode("ascii")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(encoded)
    temporary.replace(path)


def _load(path: Path) -> Mapping[str, Any]:
    if path.is_absolute() or ".." in path.parts:
        raise Phase47ReportError(f"path must be repository-relative: {path}")
    full = ROOT / path
    if not full.is_file():
        raise Phase47ReportError(f"missing receipt: {full}")
    return json.loads(full.read_text(encoding="utf-8"))


def _finite(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite(item) for item in value)
    if isinstance(value, (int, float)):
        return bool(tf.math.is_finite(tf.constant(float(value), tf.float64)).numpy())
    return True


def _arm_summary(arm: Mapping[str, Any]) -> Mapping[str, Any]:
    summary = arm.get("final_summary")
    if not isinstance(summary, Mapping):
        raise Phase47ReportError("mutation arm has no final summary")
    required = ("theta_mean_0", "covariance_offdiag_max_abs", "negative_mode_fraction", "root_count", "weighted_ess_fraction")
    for key in required:
        if key not in summary or not _finite(summary[key]):
            raise Phase47ReportError(f"non-finite or missing summary field: {key}")
    return {key: summary[key] for key in required}


def _spread(rows: list[Mapping[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows]
    return max(values) - min(values)


def _markdown(result: Mapping[str, Any]) -> str:
    lines = [
        "# v2.9 Paired Identity/MH Mutation Report", "", f"Status: `{result['status']}`", f"Branch: `{result['branch']}` (descriptive only)", "",
        "| Replicate | Identity mean0 | MH mean0 | Identity ESS | MH ESS | Identity neg | MH neg |", "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["replicate_rows"]:
        lines.append(f"| {row['replicate']} | {float(row['identity']['theta_mean_0']):.6f} | {float(row['mh']['theta_mean_0']):.6f} | {float(row['identity']['weighted_ess_fraction']):.6f} | {float(row['mh']['weighted_ess_fraction']):.6f} | {float(row['identity']['negative_mode_fraction']):.6f} | {float(row['mh']['negative_mode_fraction']):.6f} |")
    lines.extend(["", "Raw paired mutation diagnostics only; no convergence, whitening, posterior, HMC, or LEDH claim.", ""])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-root", required=True, type=Path)
    parser.add_argument("--boundary-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    paths = (args.fixture_root, args.boundary_root, args.output_root)
    if any(path.is_absolute() or ".." in path.parts for path in paths):
        raise Phase47ReportError("all paths must be repository-relative")
    output = ROOT / args.output_root
    if output.exists():
        raise Phase47ReportError(f"refusing to overwrite {output}")
    started = time.perf_counter()
    fixture = _load(args.fixture_root / "result.json")
    boundary = _load(args.boundary_root / "result.json")
    if fixture.get("schema") != EXPECTED_FIXTURE_SCHEMA or fixture.get("status") != EXPECTED_FIXTURE_STATUS:
        raise Phase47ReportError("analytic MH fixture is not passing")
    if boundary.get("schema") != EXPECTED_SCHEMA or boundary.get("status") != EXPECTED_BOUNDARY_STATUS or boundary.get("plan_version") != EXPECTED_VERSION or boundary.get("target_signature") != EXPECTED_TARGET:
        raise Phase47ReportError("q=20 mutation boundary is not passing")
    replicates = boundary.get("replicates")
    if not isinstance(replicates, list) or len(replicates) != 3:
        raise Phase47ReportError("expected exactly three paired replicates")
    rows: list[Mapping[str, Any]] = []
    for index, replicate in enumerate(replicates, start=1):
        if replicate.get("replicate") != index:
            raise Phase47ReportError("replicate numbering is not deterministic")
        paired = replicate.get("paired", {})
        if paired.get("same_initial_cloud") is not True or paired.get("same_resampling_seeds") is not True:
            raise Phase47ReportError(f"pairing gate failed for replicate {index}")
        identity = replicate.get("identity", {})
        mh = replicate.get("mh", {})
        if identity.get("status") != "PASS_V2_9_MUTATION_ARM" or mh.get("status") != "PASS_V2_9_MUTATION_ARM":
            raise Phase47ReportError(f"mutation arm failed for replicate {index}")
        if not _finite(identity.get("stages")) or not _finite(mh.get("stages")):
            raise Phase47ReportError(f"stage diagnostics non-finite for replicate {index}")
        rows.append({"replicate": index, "pilot_root": replicate.get("pilot_root"), "pilot_sha256": replicate.get("pilot_sha256"), "identity": _arm_summary(identity), "mh": _arm_summary(mh), "identity_acceptance": 0.0, "mh_acceptance": float(sum(float(stage["mutation"]["acceptance_rate"]) for stage in mh["stages"]) / max(1, sum(1 for stage in mh["stages"] if int(stage["mutation"]["steps"]) > 0))), "mh_move_fraction": float(sum(float(stage["mutation"]["move_fraction"]) for stage in mh["stages"]) / max(1, sum(1 for stage in mh["stages"] if int(stage["mutation"]["steps"]) > 0)))})
    identity_spreads = {key: _spread([row["identity"] for row in rows], key) for key in ("theta_mean_0", "covariance_offdiag_max_abs", "negative_mode_fraction", "root_count", "weighted_ess_fraction")}
    mh_spreads = {key: _spread([row["mh"] for row in rows], key) for key in identity_spreads}
    reductions = {key: mh_spreads[key] <= identity_spreads[key] for key in identity_spreads}
    # This is deliberately a descriptive branch: no tolerance or superiority
    # claim is attached to the component-wise spread relation.
    branch = "mh_rejuvenation_reduces_between_bank_variability_descriptive" if all(reductions[key] for key in ("theta_mean_0", "negative_mode_fraction", "covariance_offdiag_max_abs")) else "mh_rejuvenation_does_not_reduce_variability"
    result = {
        "schema": "bayesfilter.ssl_lstm.q20.corrected_theta_mutation_report.v1",
        "status": "PASS_V2_9_MUTATION_REPORT",
        "plan_version": EXPECTED_VERSION,
        "role": "read_only_paired_identity_vs_mh_mutation_report",
        "target_signature": EXPECTED_TARGET,
        "fixture_status": fixture["status"],
        "boundary_status": boundary["status"],
        "branch": branch,
        "branch_is_statistical_ranking": False,
        "replicate_rows": rows,
        "identity_spreads": identity_spreads,
        "mh_spreads": mh_spreads,
        "spread_reductions_or_equal": reductions,
        "decision_table": [
            {"decision": "retain_theta_target", "status": "pass", "primary_criterion": "fixture/boundary/pairing gates", "veto": "none", "next_action": "retain theta authority", "not_concluded": "posterior correctness"},
            {"decision": "promote_IID_whitening", "status": "veto", "primary_criterion": "finite mutation clouds", "veto": "finite replicate comparison is not a Gaussian law", "next_action": "keep whitening closed", "not_concluded": "IID Gaussian law"},
            {"decision": "promote_MH_as_default", "status": "defer", "primary_criterion": "descriptive paired spread", "veto": "three replicates and no uncertainty model", "next_action": "retain MH as a role-limited candidate only", "not_concluded": "superiority or default readiness"},
        ],
        "inference_status": {"hard_veto_screen": "passed", "statistically_supported_ranking": "none", "descriptive_differences": "paired identity/MH support and mutation diagnostics", "default_readiness": "not_ready", "next_evidence": "longer independent validation with uncertainty and downstream target checks"},
        "red_team": {"strongest_alternative": "the selected MH scale changes the finite result rather than repairing support generally", "overturning_evidence": "fresh scales and independent validation with stable downstream behavior", "weakest_evidence": "three replicates, two mutation steps per stage, no MC intervals"},
        "nonclaims": ["No finite-run convergence, posterior correctness, IID whitening, exhaustive mode discovery, HMC, canonical LEDH, superiority, or default claim.", "Acceptance and spread relations are descriptive only.", "No mutation cloud was used to train NeuTra or select an objective."],
        "sources": {"fixture_root": args.fixture_root, "boundary_root": args.boundary_root, "fixture_sha256": _sha(ROOT / args.fixture_root / "result.json"), "boundary_sha256": _sha(ROOT / args.boundary_root / "result.json")},
        "run_manifest": {"program": PLAN.as_posix(), "runner": RUNNER.as_posix(), "command": " ".join(sys.argv), "git_commit": subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=ROOT, text=True).strip(), "git_dirty": bool(subprocess.check_output(("git", "status", "--porcelain"), cwd=ROOT, text=True).strip()), "python": sys.executable, "python_version": platform.python_version(), "tensorflow": tf.__version__, "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"], "tf_force_gpu_allow_growth": os.environ["TF_FORCE_GPU_ALLOW_GROWTH"], "gpu_hidden_intentionally": True, "jit_compile": False, "wall_seconds": time.perf_counter() - started, "source_sha256": {"plan": _sha(PLAN), "runner": _sha(RUNNER)}}
    }
    _write_json(output / "result.json", result)
    (output / "result.md").write_text(_markdown(result), encoding="ascii")
    print(json.dumps({"status": result["status"], "branch": branch, "output_root": args.output_root.as_posix()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
