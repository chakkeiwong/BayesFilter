"""Report the v2.3 independent-bank support and transport comparison.

This is a CPU-hidden, read-only diagnostic. It verifies the frozen v2.2
authority, the fresh M0 audit bank, and the terminal v2.3 transport receipt,
then emits a normalized comparison without selecting a candidate.
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
    raise RuntimeError("Phase 41 report is CPU diagnostic-only; hide CUDA before import")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf


RUNNER = Path(__file__).resolve()
PLAN = ROOT / "docs/plans/bayesfilter-ssl-lstm-q20-parameter-authority-corrected-continuation-2026-08-25.md"
PHASE40_REPORT = ROOT / "docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase40-root-group-stratified-boundary/measure-separation/result.json"
EXPECTED_VERSION = "v2.3-independent-audit-bank"
EXPECTED_TARGET_SIGNATURE = "9a86e60081f1b9cd288dbdb1dcbe1e9a5b5e23d9b5ef97afdb72ee95c23d7278"
EXPECTED_M0_PROTOCOL = "a1f0f0493bb8bd594923b61ee9a92f3c8dcb72a612b64ad675b9ab7ff4723631"
EXPECTED_C0_PROTOCOL = "270fc99b81d08e23670c62fcd02e69e7452f26b5e5641187c3083faecbac7067"
EXPECTED_AUDIT_SCHEMA = "bayesfilter.ssl_lstm.q20.corrected_theta_neutra_boundary.v4_independent_audit_bank"
EXPECTED_MEASURE_SCHEMA = "bayesfilter.ssl_lstm.q20.corrected_theta_measure_separation.v2_root_group_stratified"


class Phase41ReportError(RuntimeError):
    """Raised when a read-only comparison receipt is not auditable."""


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
        raise Phase41ReportError(f"refusing to overwrite output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(_safe(payload), sort_keys=True, indent=2, allow_nan=False) + "\n").encode("ascii")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(encoded)
    temporary.replace(path)


def _load(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise Phase41ReportError(f"missing JSON receipt: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_tensor(receipt: Mapping[str, Any]) -> tf.Tensor:
    path = Path(str(receipt["path"]))
    if not path.is_absolute():
        path = ROOT / path
    encoded = path.read_bytes()
    if hashlib.sha256(encoded).hexdigest() != str(receipt["sha256"]):
        raise Phase41ReportError(f"tensor hash mismatch: {path}")
    tensor = tf.io.parse_tensor(encoded, out_type=getattr(tf, str(receipt["dtype"])))
    tensor = tf.ensure_shape(tensor, receipt["shape"])
    if tensor.dtype.is_floating or tensor.dtype.is_complex:
        tf.debugging.assert_all_finite(tensor, f"non-finite tensor {path}")
    return tensor


def _arm(pilot: Mapping[str, Any], name: str, protocol: str) -> Mapping[str, Any]:
    if pilot.get("status") != "PASS_THETA_MEASURE_PILOT":
        raise Phase41ReportError("pilot is not a passing theta-measure receipt")
    arm = pilot.get("arms", {}).get(name)
    if not isinstance(arm, Mapping) or arm.get("status") != "PASS_THETA_MEASURE_PILOT":
        raise Phase41ReportError(f"pilot arm {name} is not passing")
    if arm.get("target_signature") != EXPECTED_TARGET_SIGNATURE:
        raise Phase41ReportError(f"{name} target signature mismatch")
    if arm.get("protocol", {}).get("measure") != "theta_R4":
        raise Phase41ReportError(f"{name} measure mismatch")
    if arm.get("configuration", {}).get("protocol_hash") != protocol:
        raise Phase41ReportError(f"{name} protocol hash mismatch")
    return arm


def _fresh_summary(arm: Mapping[str, Any]) -> Mapping[str, Any]:
    receipts = arm["receipts"]
    theta = _load_tensor(receipts["final_theta"])
    target = _load_tensor(receipts["final_target_log_theta"])
    proposal = _load_tensor(receipts["final_proposal_log_theta"])
    weights = _load_tensor(receipts["final_normalized_weights"])
    roots = _load_tensor(receipts["final_roots"])
    n = int(theta.shape[0])
    if theta.shape != (n, 4) or any(value.shape != (n,) for value in (target, proposal, weights, roots)):
        raise Phase41ReportError("fresh cloud shape mismatch")
    weights = tf.maximum(tf.cast(weights, tf.float64), tf.constant(1.0e-300, tf.float64))
    weights = weights / tf.reduce_sum(weights)
    centered = theta - tf.reduce_sum(weights[:, tf.newaxis] * theta, axis=0)[tf.newaxis, :]
    mean = tf.reduce_sum(weights[:, tf.newaxis] * theta, axis=0)
    covariance = tf.einsum("n,ni,nj->ij", weights, centered, centered)
    ratio = target - proposal
    sign = theta[:, 2] < 0.0
    unique_roots = tf.size(tf.unique(roots).y)
    return {
        "rows": n,
        "root_count": unique_roots,
        "effective_sample_size_fraction": tf.math.reciprocal(tf.cast(n, tf.float64) * tf.reduce_sum(tf.square(weights))),
        "maximum_normalized_weight": tf.reduce_max(weights),
        "weighted_negative_mode_fraction": tf.reduce_sum(tf.where(sign, weights, tf.zeros_like(weights))),
        "negative_count": tf.reduce_sum(tf.cast(sign, tf.int32)),
        "positive_count": tf.reduce_sum(tf.cast(tf.logical_not(sign), tf.int32)),
        "theta_mean": mean,
        "theta_covariance": covariance,
        "target_log_theta_min": tf.reduce_min(target),
        "target_log_theta_max": tf.reduce_max(target),
        "proposal_log_theta_min": tf.reduce_min(proposal),
        "proposal_log_theta_max": tf.reduce_max(proposal),
        "log_ratio_min": tf.reduce_min(ratio),
        "log_ratio_max": tf.reduce_max(ratio),
        "weighted_log_ratio_mean": tf.reduce_sum(weights * ratio),
        "finite": tf.reduce_all(tf.math.is_finite(tf.concat((tf.reshape(theta, [-1]), tf.reshape(target, [-1]), tf.reshape(proposal, [-1]), tf.reshape(weights, [-1])), axis=0))),
    }


def _scalar(value: Any) -> float:
    return float(tf.convert_to_tensor(value).numpy())


def _max_abs(values: Any) -> float:
    tensor = tf.cast(tf.convert_to_tensor(values), tf.float64)
    return float(tf.reduce_max(tf.abs(tensor)).numpy())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority-root", required=True, type=Path)
    parser.add_argument("--fresh-root", required=True, type=Path)
    parser.add_argument("--audit-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    for path in (args.authority_root, args.fresh_root, args.audit_root, args.output_root):
        if path.is_absolute() or ".." in path.parts:
            raise Phase41ReportError("all paths must be repository-relative")
    output = ROOT / args.output_root
    if output.exists():
        raise Phase41ReportError(f"refusing to overwrite output root: {output}")
    started = time.perf_counter()
    authority_pilot_path = ROOT / args.authority_root / "pilot.json"
    fresh_pilot_path = ROOT / args.fresh_root / "pilot.json"
    audit_path = ROOT / args.audit_root / "result.json"
    authority_pilot = _load(authority_pilot_path)
    fresh_pilot = _load(fresh_pilot_path)
    audit = _load(audit_path)
    old_measure = _load(PHASE40_REPORT)
    if audit.get("schema") != EXPECTED_AUDIT_SCHEMA or audit.get("status") != "PASS_V2_3_INDEPENDENT_AUDIT_BOUNDARY":
        raise Phase41ReportError("audit receipt is not active v2.3")
    if audit.get("plan_version") != EXPECTED_VERSION or audit.get("target_signature") != EXPECTED_TARGET_SIGNATURE:
        raise Phase41ReportError("audit version or target signature mismatch")
    if old_measure.get("schema") != EXPECTED_MEASURE_SCHEMA or old_measure.get("status") != "PASS_V2_2_THETA_MEASURE_SEPARATION_DIAGNOSTIC":
        raise Phase41ReportError("old v2.2 measure report is not a passing frozen source")
    old_m0 = _arm(authority_pilot, "M0", EXPECTED_M0_PROTOCOL)
    fresh_m0 = _arm(fresh_pilot, "M0", EXPECTED_M0_PROTOCOL)
    _arm(authority_pilot, "C0", EXPECTED_C0_PROTOCOL)
    _arm(fresh_pilot, "C0", EXPECTED_C0_PROTOCOL)
    if _sha(authority_pilot_path) != audit["authority"]["pilot_sha256"]:
        raise Phase41ReportError("audit authority pilot hash mismatch")
    if _sha(authority_pilot_path) == _sha(fresh_pilot_path):
        raise Phase41ReportError("fresh pilot hash equals authority pilot hash")
    if audit.get("fresh_rows_used_for_training") or audit.get("fresh_rows_used_for_selection"):
        raise Phase41ReportError("fresh bank was used by a forbidden path")
    fresh = _fresh_summary(fresh_m0)
    if not bool(fresh["finite"].numpy()):
        raise Phase41ReportError("fresh summary is non-finite")
    rows: list[Mapping[str, Any]] = []
    old_partitions = old_measure["partitions"]
    for name in ("train", "validation", "audit"):
        physical = old_partitions[name]["physical"]
        affine = old_partitions[name]["train_measure_affine"]
        rows.append({
            "source": f"old_v2_2_{name}",
            "rows": physical["count"],
            "roots": physical["root_count"],
            "ess_fraction": physical["effective_sample_size_fraction"],
            "negative_mode_fraction": physical["weighted_negative_mode_fraction"],
            "theta_mean_0": physical["theta_mean"][0],
            "target_log_range": [physical["target_log_theta_min"], physical["target_log_theta_max"]],
            "proposal_log_range": [physical["proposal_log_theta_min"], physical["proposal_log_theta_max"]],
            "log_ratio_range": [physical["log_ratio_min"], physical["log_ratio_max"]],
            "affine_latent_mean_max_abs": _max_abs(affine["latent_mean"]),
            "affine_covariance_offdiag_max_abs": _max_abs(tf.convert_to_tensor(affine["latent_covariance"]) - tf.linalg.diag(tf.linalg.diag_part(tf.convert_to_tensor(affine["latent_covariance"])) )),
        })
    rows.append({
        "source": "fresh_v2_3_M0",
        "rows": fresh["rows"],
        "roots": fresh["root_count"],
        "ess_fraction": fresh["effective_sample_size_fraction"],
        "negative_mode_fraction": fresh["weighted_negative_mode_fraction"],
        "theta_mean_0": fresh["theta_mean"][0],
        "target_log_range": [fresh["target_log_theta_min"], fresh["target_log_theta_max"]],
        "proposal_log_range": [fresh["proposal_log_theta_min"], fresh["proposal_log_theta_max"]],
        "log_ratio_range": [fresh["log_ratio_min"], fresh["log_ratio_max"]],
        "affine_latent_mean_max_abs": None,
        "affine_covariance_offdiag_max_abs": None,
    })
    transport_rows = []
    for name, arm in audit["arms"].items():
        validation = arm["validation"]
        fresh_audit = arm["fresh_audit"]
        transport_rows.append({
            "arm": name,
            "precondition": arm["precondition"],
            "status": arm["status"],
            "old_validation_loss": validation["loss"],
            "old_validation_mean_max_abs": _max_abs(validation["latent_weighted_mean"]),
            "old_validation_covariance_offdiag_max_abs": _max_abs(tf.convert_to_tensor(validation["latent_weighted_covariance"]) - tf.linalg.diag(tf.linalg.diag_part(tf.convert_to_tensor(validation["latent_weighted_covariance"])) )),
            "fresh_audit_loss": fresh_audit["loss"],
            "fresh_audit_mean_max_abs": fresh_audit["latent_mean_max_abs"],
            "fresh_audit_covariance_offdiag_max_abs": fresh_audit["latent_covariance_max_abs_offdiag"],
            "fresh_rows_used_for_training": arm["fresh_rows_used_for_training"],
            "fresh_rows_used_for_selection": arm["fresh_rows_used_for_selection"],
        })
    # These are predeclared explanatory branches, not a ranking or promotion.
    identity_rows = [row for row in transport_rows if row["precondition"] == "identity"]
    fresh_improvement = all(row["fresh_audit_mean_max_abs"] < row["old_validation_mean_max_abs"] and row["fresh_audit_covariance_offdiag_max_abs"] < row["old_validation_covariance_offdiag_max_abs"] for row in identity_rows)
    result = {
        "schema": "bayesfilter.ssl_lstm.q20.corrected_theta_independent_audit_report.v1",
        "status": "PASS_V2_3_INDEPENDENT_AUDIT_REPORT",
        "plan_version": EXPECTED_VERSION,
        "role": "read_only_support_vs_objective_explanation_diagnostic",
        "measure": "theta_R4",
        "target_signature": EXPECTED_TARGET_SIGNATURE,
        "authority": {"root": args.authority_root, "pilot_sha256": _sha(authority_pilot_path), "m0_protocol_hash": EXPECTED_M0_PROTOCOL, "split_policy": audit["authority"]["split_policy"]},
        "fresh_audit": {"root": args.fresh_root, "pilot_sha256": _sha(fresh_pilot_path), "m0_protocol_hash": EXPECTED_M0_PROTOCOL, "independent": True, "untouched": True},
        "source_reports": {"phase40_measure_report": PHASE40_REPORT, "phase40_measure_report_sha256": _sha(PHASE40_REPORT), "phase41_audit": audit_path, "phase41_audit_sha256": _sha(audit_path)},
        "support_rows": rows,
        "transport_rows": transport_rows,
        "explanatory_branch": "finite_holdout_mismatch_is_plausible" if fresh_improvement else "objective_or_capacity_mismatch_remains_plausible",
        "branch_is_statistical_ranking": False,
        "decision_table": [{"decision": "retain_theta_target", "primary_criterion": "all target/protocol/status gates pass", "status": "pass", "next_action": "continue bounded parameter-space repair", "not_concluded": "posterior correctness"}, {"decision": "whitening_promotion", "primary_criterion": "independent fresh audit residuals", "status": "veto", "next_action": "design next repair from explanatory branch", "not_concluded": "IID Gaussian whitening"}],
        "inference_status": {"hard_veto_screen": "passed", "statistically_supported_ranking": "none", "descriptive_differences": "support ranges, moments, losses, ESS, and residuals", "default_readiness": "not_ready", "next_evidence": "fresh-bank replication or objective/support repair with predeclared gates"},
        "red_team": {"strongest_alternative": "one fresh bank can still share the proposal's mode bias and one-seed residuals are noisy", "overturning_evidence": "replicated fresh banks with comparable support and persistent residuals would weaken the split explanation", "weakest_evidence": "only one independent bank and one training seed"},
        "nonclaims": ["No IID Gaussian whitening, posterior correctness, exhaustive mode discovery, normalizer, HMC, canonical LEDH, or superiority claim.", "The branch label is a repair hypothesis, not a statistical ranking.", "Fresh rows were not used for optimization or selection."],
        "run_manifest": {"program": PLAN, "runner": RUNNER, "command": " ".join(sys.argv), "git_commit": subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=ROOT, text=True).strip(), "git_dirty": bool(subprocess.check_output(("git", "status", "--porcelain"), cwd=ROOT, text=True).strip()), "python": sys.executable, "python_version": platform.python_version(), "tensorflow": tf.__version__, "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"], "wall_seconds": time.perf_counter() - started, "source_sha256": {"plan": _sha(PLAN), "runner": _sha(RUNNER), "phase40_report": _sha(PHASE40_REPORT), "authority_pilot": _sha(authority_pilot_path), "fresh_pilot": _sha(fresh_pilot_path), "audit": _sha(audit_path)}},
    }
    _write_json(output / "result.json", result)
    lines = ["# v2.3 Independent Theta Audit Report", "", f"Status: `{result['status']}`", "", f"Explanatory branch: `{result['explanatory_branch']}` (not a statistical ranking)", "", "| Source | rows | roots | ESS | neg-mode | theta mean[0] | affine mean max | affine covariance offdiag max |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['source']} | {row['rows']} | {row['roots']} | {float(row['ess_fraction']):.6f} | {float(row['negative_mode_fraction']):.6f} | {float(row['theta_mean_0']):.6f} | {row['affine_latent_mean_max_abs'] if row['affine_latent_mean_max_abs'] is not None else 'n/a'} | {row['affine_covariance_offdiag_max_abs'] if row['affine_covariance_offdiag_max_abs'] is not None else 'n/a'} |")
    lines.extend(["", "| Arm | old validation mean/cov | fresh audit mean/cov |", "|---|---:|---:|"])
    for row in transport_rows:
        lines.append(f"| {row['arm']} | {row['old_validation_mean_max_abs']:.6f} / {row['old_validation_covariance_offdiag_max_abs']:.6f} | {row['fresh_audit_mean_max_abs']:.6f} / {row['fresh_audit_covariance_offdiag_max_abs']:.6f} |")
    lines.extend(["", "The hard boundary passed, but residuals remain descriptive and no whitening or posterior claim is made.", ""])
    (output / "result.md").write_text("\n".join(lines), encoding="ascii")
    print(json.dumps({"status": result["status"], "output_root": args.output_root.as_posix(), "branch": result["explanatory_branch"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
