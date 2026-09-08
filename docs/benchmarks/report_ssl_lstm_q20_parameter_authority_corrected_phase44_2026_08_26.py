"""Report the v2.6 N=256/N=512 frozen-state support audit.

CPU-hidden and read-only. The reporter verifies the old authority, three N=256
fresh pilots, one N=512 fresh pilot, the v2.4 state-hash reference, and the
shared-state GPU receipt, then compares bank-specific support and transport
diagnostics without pooling or selecting a bank.
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
    raise RuntimeError("Phase 44 report is CPU diagnostic-only; hide CUDA before import")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf


RUNNER = Path(__file__).resolve()
PLAN = ROOT / "docs/plans/bayesfilter-ssl-lstm-q20-parameter-authority-corrected-continuation-2026-08-25.md"
PHASE40_REPORT = ROOT / "docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase40-root-group-stratified-boundary/measure-separation/result.json"
EXPECTED_VERSION = "v2.6-larger-n-support-diagnostic"
EXPECTED_TARGET = "9a86e60081f1b9cd288dbdb1dcbe1e9a5b5e23d9b5ef97afdb72ee95c23d7278"
EXPECTED_M0 = "a1f0f0493bb8bd594923b61ee9a92f3c8dcb72a612b64ad675b9ab7ff4723631"
EXPECTED_C0 = "270fc99b81d08e23670c62fcd02e69e7452f26b5e5641187c3083faecbac7067"
EXPECTED_AUDIT_SCHEMA = "bayesfilter.ssl_lstm.q20.corrected_theta_neutra_boundary.v7_four_bank_mixed_n"
REFERENCE_AUDIT_SCHEMA = "bayesfilter.ssl_lstm.q20.corrected_theta_neutra_boundary.v5_two_independent_banks"


class Phase44ReportError(RuntimeError):
    """Raised when the larger-N report cannot be audited."""


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
        raise Phase44ReportError(f"refusing to overwrite output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(_safe(payload), sort_keys=True, indent=2, allow_nan=False) + "\n").encode("ascii")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(encoded)
    temporary.replace(path)


def _load(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise Phase44ReportError(f"missing receipt: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_tensor(receipt: Mapping[str, Any]) -> tf.Tensor:
    path = Path(str(receipt["path"]))
    if not path.is_absolute():
        path = ROOT / path
    encoded = path.read_bytes()
    if hashlib.sha256(encoded).hexdigest() != str(receipt["sha256"]):
        raise Phase44ReportError(f"tensor hash mismatch: {path}")
    value = tf.io.parse_tensor(encoded, out_type=getattr(tf, str(receipt["dtype"])))
    value = tf.ensure_shape(value, receipt["shape"])
    if value.dtype.is_floating or value.dtype.is_complex:
        tf.debugging.assert_all_finite(value, f"non-finite tensor {path}")
    return value


def _pilot_arm(
    pilot: Mapping[str, Any],
    name: str,
    protocol: str,
    *,
    expected_particles: int,
    expected_calibration_particles: int,
) -> Mapping[str, Any]:
    if pilot.get("status") != "PASS_THETA_MEASURE_PILOT":
        raise Phase44ReportError("pilot is not passing")
    arm = pilot.get("arms", {}).get(name)
    if not isinstance(arm, Mapping) or arm.get("status") != "PASS_THETA_MEASURE_PILOT":
        raise Phase44ReportError(f"pilot arm {name} is not passing")
    if arm.get("target_signature") != EXPECTED_TARGET:
        raise Phase44ReportError(f"{name} target signature mismatch")
    if arm.get("protocol", {}).get("measure") != "theta_R4":
        raise Phase44ReportError(f"{name} measure mismatch")
    if arm.get("configuration", {}).get("protocol_hash") != protocol:
        raise Phase44ReportError(f"{name} protocol mismatch")
    if int(arm.get("configuration", {}).get("particles", -1)) != int(expected_particles):
        raise Phase44ReportError(f"{name} particle count mismatch")
    if int(pilot.get("calibration", {}).get("particle_count", -1)) != int(expected_calibration_particles):
        raise Phase44ReportError(f"{name} calibration particle count mismatch")
    return arm


def _summary(arm: Mapping[str, Any]) -> Mapping[str, Any]:
    receipts = arm["receipts"]
    theta = _load_tensor(receipts["final_theta"])
    target = _load_tensor(receipts["final_target_log_theta"])
    proposal = _load_tensor(receipts["final_proposal_log_theta"])
    weights = _load_tensor(receipts["final_normalized_weights"])
    roots = _load_tensor(receipts["final_roots"])
    n = int(theta.shape[0])
    if theta.shape != (n, 4) or any(value.shape != (n,) for value in (target, proposal, weights, roots)):
        raise Phase44ReportError("bank tensor shape mismatch")
    weights = tf.cast(weights, tf.float64)
    weights = tf.maximum(weights, tf.constant(1.0e-300, tf.float64))
    weights = weights / tf.reduce_sum(weights)
    mean = tf.reduce_sum(weights[:, tf.newaxis] * theta, axis=0)
    centered = theta - mean[tf.newaxis, :]
    covariance = tf.einsum("n,ni,nj->ij", weights, centered, centered)
    ratio = target - proposal
    sign = theta[:, 2] < 0.0
    return {
        "rows": n,
        "root_count": tf.size(tf.unique(roots).y),
        "ess_fraction": tf.math.reciprocal(tf.cast(n, tf.float64) * tf.reduce_sum(tf.square(weights))),
        "maximum_normalized_weight": tf.reduce_max(weights),
        "negative_mode_fraction": tf.reduce_sum(tf.where(sign, weights, tf.zeros_like(weights))),
        "negative_count": tf.reduce_sum(tf.cast(sign, tf.int32)),
        "positive_count": tf.reduce_sum(tf.cast(tf.logical_not(sign), tf.int32)),
        "theta_mean": mean,
        "theta_covariance": covariance,
        "target_log_range": (tf.reduce_min(target), tf.reduce_max(target)),
        "proposal_log_range": (tf.reduce_min(proposal), tf.reduce_max(proposal)),
        "log_ratio_range": (tf.reduce_min(ratio), tf.reduce_max(ratio)),
        "finite": tf.reduce_all(tf.math.is_finite(tf.concat((tf.reshape(theta, [-1]), tf.reshape(target, [-1]), tf.reshape(proposal, [-1]), tf.reshape(weights, [-1])), axis=0))),
    }


def _float(value: Any) -> float:
    return float(tf.convert_to_tensor(value).numpy())


def _max_abs(value: Any) -> float:
    return float(tf.reduce_max(tf.abs(tf.cast(tf.convert_to_tensor(value), tf.float64))).numpy())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority-root", required=True, type=Path)
    parser.add_argument("--fresh-root-a", required=True, type=Path)
    parser.add_argument("--fresh-root-b", required=True, type=Path)
    parser.add_argument("--fresh-root-c", required=True, type=Path)
    parser.add_argument("--fresh-root-n512", required=True, type=Path)
    parser.add_argument("--reference-audit-root", required=True, type=Path)
    parser.add_argument("--audit-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    for path in (args.authority_root, args.fresh_root_a, args.fresh_root_b, args.fresh_root_c, args.fresh_root_n512, args.reference_audit_root, args.audit_root, args.output_root):
        if path.is_absolute() or ".." in path.parts:
            raise Phase44ReportError("all paths must be repository-relative")
    output = ROOT / args.output_root
    if output.exists():
        raise Phase44ReportError(f"refusing to overwrite output root: {output}")
    started = time.perf_counter()
    authority_path = ROOT / args.authority_root / "pilot.json"
    bank_a_path = ROOT / args.fresh_root_a / "pilot.json"
    bank_b_path = ROOT / args.fresh_root_b / "pilot.json"
    bank_c_path = ROOT / args.fresh_root_c / "pilot.json"
    bank_n512_path = ROOT / args.fresh_root_n512 / "pilot.json"
    audit_path = ROOT / args.audit_root / "result.json"
    reference_audit_path = ROOT / args.reference_audit_root / "result.json"
    authority = _load(authority_path)
    bank_a = _load(bank_a_path)
    bank_b = _load(bank_b_path)
    bank_c = _load(bank_c_path)
    bank_n512 = _load(bank_n512_path)
    audit = _load(audit_path)
    reference_audit = _load(reference_audit_path)
    old_measure = _load(PHASE40_REPORT)
    if audit.get("schema") != EXPECTED_AUDIT_SCHEMA or audit.get("status") != "PASS_V2_6_LARGER_N_BOUNDARY":
        raise Phase44ReportError("audit is not a passing v2.6 receipt")
    if audit.get("plan_version") != EXPECTED_VERSION or audit.get("target_signature") != EXPECTED_TARGET:
        raise Phase44ReportError("audit version/target mismatch")
    if audit.get("one_trainer_state_per_arm_for_four_banks") is not True:
        raise Phase44ReportError("four-bank shared trainer-state receipt missing")
    if audit.get("fresh_rows_used_for_training") or audit.get("fresh_rows_used_for_selection"):
        raise Phase44ReportError("fresh rows used on forbidden path")
    if reference_audit.get("schema") != REFERENCE_AUDIT_SCHEMA or reference_audit.get("status") != "PASS_V2_4_TWO_BANK_BOUNDARY":
        raise Phase44ReportError("v2.4 reference audit is not passing")
    if reference_audit.get("target_signature") != EXPECTED_TARGET:
        raise Phase44ReportError("v2.4 reference target mismatch")
    if old_measure.get("status") != "PASS_V2_2_THETA_MEASURE_SEPARATION_DIAGNOSTIC":
        raise Phase44ReportError("old measure report is not a passing frozen source")
    old_m0 = _pilot_arm(authority, "M0", EXPECTED_M0, expected_particles=256, expected_calibration_particles=64)
    a_m0 = _pilot_arm(bank_a, "M0", EXPECTED_M0, expected_particles=256, expected_calibration_particles=64)
    b_m0 = _pilot_arm(bank_b, "M0", EXPECTED_M0, expected_particles=256, expected_calibration_particles=64)
    c_m0 = _pilot_arm(bank_c, "M0", EXPECTED_M0, expected_particles=256, expected_calibration_particles=64)
    n512_m0 = _pilot_arm(bank_n512, "M0", EXPECTED_M0, expected_particles=512, expected_calibration_particles=128)
    _pilot_arm(authority, "C0", EXPECTED_C0, expected_particles=256, expected_calibration_particles=64)
    _pilot_arm(bank_a, "C0", EXPECTED_C0, expected_particles=256, expected_calibration_particles=64)
    _pilot_arm(bank_b, "C0", EXPECTED_C0, expected_particles=256, expected_calibration_particles=64)
    _pilot_arm(bank_c, "C0", EXPECTED_C0, expected_particles=256, expected_calibration_particles=64)
    _pilot_arm(bank_n512, "C0", EXPECTED_C0, expected_particles=512, expected_calibration_particles=128)
    pilot_paths = (authority_path, bank_a_path, bank_b_path, bank_c_path, bank_n512_path)
    if len({_sha(path) for path in pilot_paths}) != 5:
        raise Phase44ReportError("pilot receipts are not mutually independent")
    if _sha(authority_path) != audit["authority"]["pilot_sha256"]:
        raise Phase44ReportError("authority pilot hash mismatch")
    expected_fresh_hashes = {"bank_a": _sha(bank_a_path), "bank_b": _sha(bank_b_path), "bank_c": _sha(bank_c_path), "bank_n512": _sha(bank_n512_path)}
    for label, path in expected_fresh_hashes.items():
        if audit["fresh_banks"][label]["pilot_sha256"] != path:
            raise Phase44ReportError(f"{label} pilot hash mismatch")
    summaries = {"authority": _summary(old_m0), "bank_a": _summary(a_m0), "bank_b": _summary(b_m0), "bank_c": _summary(c_m0), "bank_n512": _summary(n512_m0)}
    if not all(bool(value["finite"].numpy()) for value in summaries.values()):
        raise Phase44ReportError("non-finite support summary")
    support_rows = []
    for label, summary in summaries.items():
        support_rows.append({
            "source": label,
            "rows": summary["rows"],
            "roots": summary["root_count"],
            "ess_fraction": summary["ess_fraction"],
            "negative_mode_fraction": summary["negative_mode_fraction"],
            "negative_count": summary["negative_count"],
            "positive_count": summary["positive_count"],
            "theta_mean_0": summary["theta_mean"][0],
            "target_log_range": summary["target_log_range"],
            "proposal_log_range": summary["proposal_log_range"],
            "log_ratio_range": summary["log_ratio_range"],
        })
    old_partitions = old_measure["partitions"]
    for label in ("train", "validation", "audit"):
        physical = old_partitions[label]["physical"]
        affine = old_partitions[label]["train_measure_affine"]
        support_rows.append({
            "source": f"old_v2_2_{label}",
            "rows": physical["count"],
            "roots": physical["root_count"],
            "ess_fraction": physical["effective_sample_size_fraction"],
            "negative_mode_fraction": physical["weighted_negative_mode_fraction"],
            "negative_count": physical["negative_count"],
            "positive_count": physical["positive_count"],
            "theta_mean_0": physical["theta_mean"][0],
            "target_log_range": (physical["target_log_theta_min"], physical["target_log_theta_max"]),
            "proposal_log_range": (physical["proposal_log_theta_min"], physical["proposal_log_theta_max"]),
            "log_ratio_range": (physical["log_ratio_min"], physical["log_ratio_max"]),
            "affine_latent_mean_max_abs": _max_abs(affine["latent_mean"]),
            "affine_covariance_offdiag_max_abs": _max_abs(tf.cast(tf.convert_to_tensor(affine["latent_covariance"]), tf.float64) - tf.linalg.diag(tf.linalg.diag_part(tf.cast(tf.convert_to_tensor(affine["latent_covariance"]), tf.float64)))),
        })
    reference_states = reference_audit["arms"]
    transport_rows = []
    for arm_name, arm in audit["arms"].items():
        reference_arm = reference_states.get(arm_name)
        if not isinstance(reference_arm, Mapping):
            raise Phase44ReportError(f"missing v2.4 reference arm {arm_name}")
        if arm.get("reference_state_hash") != reference_arm.get("state_hash"):
            raise Phase44ReportError(f"state hash mismatch for {arm_name}")
        if arm.get("gates", {}).get("reference_state_hash_match") is not True:
            raise Phase44ReportError(f"runner state-hash gate failed for {arm_name}")
        validation = arm["validation"]
        old_mean = _max_abs(validation["latent_weighted_mean"])
        old_cov = _max_abs(tf.cast(tf.convert_to_tensor(validation["latent_weighted_covariance"]), tf.float64) - tf.linalg.diag(tf.linalg.diag_part(tf.cast(tf.convert_to_tensor(validation["latent_weighted_covariance"]), tf.float64))))
        bank_values = {}
        for label in ("bank_a", "bank_b", "bank_c", "bank_n512"):
            item = arm["fresh_audits"][label]
            bank_values[label] = {"loss": item["loss"], "mean_max_abs": item["latent_mean_max_abs"], "covariance_offdiag_max_abs": item["latent_covariance_max_abs_offdiag"], "mean_better_than_old_validation": item["latent_mean_max_abs"] < old_mean, "covariance_better_than_old_validation": item["latent_covariance_max_abs_offdiag"] < old_cov}
        transport_rows.append({"arm": arm_name, "precondition": arm["precondition"], "status": arm["status"], "old_validation_mean_max_abs": old_mean, "old_validation_covariance_offdiag_max_abs": old_cov, "banks": bank_values, "fresh_rows_used_for_training": arm["fresh_rows_used_for_training"], "fresh_rows_used_for_selection": arm["fresh_rows_used_for_selection"], "one_trainer_state_for_four_banks": arm["one_trainer_state_for_four_banks"], "reference_state_hash_match": True})
    n512_better_than_a = all(
        row["banks"]["bank_n512"]["mean_max_abs"] < row["banks"]["bank_a"]["mean_max_abs"]
        and row["banks"]["bank_n512"]["covariance_offdiag_max_abs"] < row["banks"]["bank_a"]["covariance_offdiag_max_abs"]
        for row in transport_rows
    )
    n512_at_or_above_old_comparator = any(
        not row["banks"]["bank_n512"]["mean_better_than_old_validation"]
        or not row["banks"]["bank_n512"]["covariance_better_than_old_validation"]
        for row in transport_rows
    )
    if n512_better_than_a:
        branch = "larger_n_descriptively_better_than_bank_a"
    elif n512_at_or_above_old_comparator:
        branch = "larger_n_does_not_resolve_support_variability"
    else:
        branch = "larger_n_residuals_descriptive"
    identity_rows = [row for row in transport_rows if row["precondition"] == "identity"]
    all_fresh_identity_better = bool(identity_rows) and all(
        value["mean_better_than_old_validation"] and value["covariance_better_than_old_validation"]
        for row in identity_rows for value in row["banks"].values()
    )
    result = {
        "schema": "bayesfilter.ssl_lstm.q20.corrected_theta_larger_n_support_report.v1",
        "status": "PASS_V2_6_LARGER_N_REPORT",
        "plan_version": EXPECTED_VERSION,
        "role": "read_only_n256_n512_support_diagnostic",
        "measure": "theta_R4",
        "target_signature": EXPECTED_TARGET,
        "explanatory_branch": branch,
        "n512_descriptively_better_than_bank_a": n512_better_than_a,
        "branch_is_statistical_ranking": False,
        "support_rows": support_rows,
        "transport_rows": transport_rows,
        "decision_table": [
            {"decision": "retain_theta_target", "primary_criterion": "v2.6 target/status/protocol and state-hash gates", "status": "pass", "veto": "none", "next_action": "retain theta authority and adjudicate support", "not_concluded": "posterior correctness"},
            {"decision": "whitening_promotion", "primary_criterion": "N=256/N=512 residuals", "status": "veto", "veto": "material finite-bank residuals", "next_action": "keep whitening closed; scope support repair if needed", "not_concluded": "IID Gaussian whitening"},
            {"decision": "objective_change", "primary_criterion": "independent-bank support branch", "status": "defer", "veto": "no uncertainty-supported objective comparison", "next_action": "separate objective-repair plan only after support evidence", "not_concluded": "objective superiority"},
        ],
        "inference_status": {"hard_veto_screen": "passed", "statistically_supported_ranking": "none", "descriptive_differences": "N=256 bank variation and N=512 support/residual differences", "default_readiness": "not_ready", "next_evidence": "independent N=512 replication or explicitly scoped support/proposal repair"},
        "red_team": {"strongest_alternative": "all banks share a proposal-support bias or the old validation comparator is unrepresentative", "overturning_evidence": "an independent N=512 replication with stable support and persistent residuals would weaken a finite-count explanation", "weakest_evidence": "one N=512 bank, one frozen state, and no uncertainty interval"},
        "sources": {"authority_root": args.authority_root, "authority_pilot_sha256": _sha(authority_path), "fresh_a_root": args.fresh_root_a, "fresh_a_pilot_sha256": _sha(bank_a_path), "fresh_b_root": args.fresh_root_b, "fresh_b_pilot_sha256": _sha(bank_b_path), "fresh_c_root": args.fresh_root_c, "fresh_c_pilot_sha256": _sha(bank_c_path), "fresh_n512_root": args.fresh_root_n512, "fresh_n512_pilot_sha256": _sha(bank_n512_path), "audit_root": args.audit_root, "audit_sha256": _sha(audit_path), "reference_audit_root": args.reference_audit_root, "reference_audit_sha256": _sha(reference_audit_path), "phase40_report": PHASE40_REPORT, "phase40_report_sha256": _sha(PHASE40_REPORT)},
        "nonclaims": ["No IID Gaussian whitening, posterior correctness, exhaustive mode discovery, normalizer, HMC, canonical LEDH, superiority, or default promotion claim.", "The branch is a repair hypothesis, not a statistical ranking.", "No bank was pooled, dropped, or used for training or selection."],
        "run_manifest": {"program": PLAN, "runner": RUNNER, "command": " ".join(sys.argv), "git_commit": subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=ROOT, text=True).strip(), "git_dirty": bool(subprocess.check_output(("git", "status", "--porcelain"), cwd=ROOT, text=True).strip()), "python": sys.executable, "python_version": platform.python_version(), "tensorflow": tf.__version__, "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"], "wall_seconds": time.perf_counter() - started, "source_sha256": {"plan": _sha(PLAN), "runner": _sha(RUNNER), "audit": _sha(audit_path), "authority_pilot": _sha(authority_path), "fresh_a_pilot": _sha(bank_a_path), "fresh_b_pilot": _sha(bank_b_path), "fresh_c_pilot": _sha(bank_c_path), "fresh_n512_pilot": _sha(bank_n512_path), "reference_audit": _sha(reference_audit_path), "phase40_report": _sha(PHASE40_REPORT)}},
    }
    _write_json(output / "result.json", result)
    lines = ["# v2.6 Larger-N Support Report", "", f"Status: `{result['status']}`", "", f"Explanatory branch: `{branch}` (not a statistical ranking)", "", "| Source | rows | roots | ESS | neg-mode | theta mean[0] |", "|---|---:|---:|---:|---:|---:|"]
    for row in support_rows:
        lines.append(f"| {row['source']} | {row['rows']} | {row['roots']} | {float(row['ess_fraction']):.6f} | {float(row['negative_mode_fraction']):.6f} | {float(row['theta_mean_0']):.6f} |")
    lines.extend(["", "| Arm | old validation mean/cov | bank A mean/cov | bank B mean/cov | bank C mean/cov | N=512 mean/cov |", "|---|---:|---:|---:|---:|---:|"])
    for row in transport_rows:
        a = row["banks"]["bank_a"]; b = row["banks"]["bank_b"]; c = row["banks"]["bank_c"]; n512 = row["banks"]["bank_n512"]
        lines.append(f"| {row['arm']} | {row['old_validation_mean_max_abs']:.6f} / {row['old_validation_covariance_offdiag_max_abs']:.6f} | {a['mean_max_abs']:.6f} / {a['covariance_offdiag_max_abs']:.6f} | {b['mean_max_abs']:.6f} / {b['covariance_offdiag_max_abs']:.6f} | {c['mean_max_abs']:.6f} / {c['covariance_offdiag_max_abs']:.6f} | {n512['mean_max_abs']:.6f} / {n512['covariance_offdiag_max_abs']:.6f} |")
    lines.extend(["", "Hard boundary and state-hash gates passed. N=256/N=512 differences remain descriptive; no whitening or posterior claim is made.", ""])
    (output / "result.md").write_text("\n".join(lines), encoding="ascii")
    print(json.dumps({"status": result["status"], "output_root": args.output_root.as_posix(), "branch": branch}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
