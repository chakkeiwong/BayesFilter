"""Report the v2.7 independent N=512 support replication (CPU-only)."""

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
    raise RuntimeError("Phase 45 report is CPU diagnostic-only; hide CUDA before import")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import tensorflow as tf

RUNNER = Path(__file__).resolve()
PLAN = ROOT / "docs/plans/bayesfilter-ssl-lstm-q20-parameter-authority-corrected-continuation-2026-08-25.md"
PHASE40_REPORT = ROOT / "docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase40-root-group-stratified-boundary/measure-separation/result.json"
EXPECTED_VERSION = "v2.7-independent-n512-replication"
EXPECTED_TARGET = "9a86e60081f1b9cd288dbdb1dcbe1e9a5b5e23d9b5ef97afdb72ee95c23d7278"
EXPECTED_M0 = "a1f0f0493bb8bd594923b61ee9a92f3c8dcb72a612b64ad675b9ab7ff4723631"
EXPECTED_C0 = "270fc99b81d08e23670c62fcd02e69e7452f26b5e5641187c3083faecbac7067"
EXPECTED_AUDIT_SCHEMA = "bayesfilter.ssl_lstm.q20.corrected_theta_neutra_boundary.v8_five_bank_mixed_n"
REFERENCE_SCHEMA = "bayesfilter.ssl_lstm.q20.corrected_theta_neutra_boundary.v7_four_bank_mixed_n"
BANK_LABELS = ("bank_a", "bank_b", "bank_c", "bank_n512_a", "bank_n512_b")


class Phase45ReportError(RuntimeError):
    """Raised when the v2.7 report cannot be audited."""


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
        raise Phase45ReportError(f"refusing to overwrite output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(_safe(payload), sort_keys=True, indent=2, allow_nan=False) + "\n").encode("ascii")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(encoded)
    temporary.replace(path)


def _load(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise Phase45ReportError(f"missing artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_tensor(receipt: Mapping[str, Any]) -> tf.Tensor:
    path = Path(str(receipt["path"]))
    if not path.is_absolute():
        path = ROOT / path
    encoded = path.read_bytes()
    if hashlib.sha256(encoded).hexdigest() != str(receipt["sha256"]):
        raise Phase45ReportError(f"tensor hash mismatch: {path}")
    value = tf.io.parse_tensor(encoded, out_type=getattr(tf, str(receipt["dtype"])))
    value = tf.ensure_shape(value, receipt["shape"])
    if value.dtype.is_floating or value.dtype.is_complex:
        tf.debugging.assert_all_finite(value, f"non-finite tensor {path}")
    return value


def _pilot_arm(pilot: Mapping[str, Any], name: str, protocol: str, particles: int, calibration: int) -> Mapping[str, Any]:
    if pilot.get("status") != "PASS_THETA_MEASURE_PILOT":
        raise Phase45ReportError("pilot status is not passing")
    arm = pilot.get("arms", {}).get(name)
    if not isinstance(arm, Mapping) or arm.get("status") != "PASS_THETA_MEASURE_PILOT":
        raise Phase45ReportError(f"pilot arm {name} is not passing")
    if arm.get("target_signature") != EXPECTED_TARGET or arm.get("protocol", {}).get("measure") != "theta_R4":
        raise Phase45ReportError(f"pilot arm {name} target/measure mismatch")
    if arm.get("configuration", {}).get("protocol_hash") != protocol:
        raise Phase45ReportError(f"pilot arm {name} protocol mismatch")
    if int(arm.get("configuration", {}).get("particles", -1)) != particles:
        raise Phase45ReportError(f"pilot arm {name} particle count mismatch")
    if int(pilot.get("calibration", {}).get("particle_count", -1)) != calibration:
        raise Phase45ReportError(f"pilot arm {name} calibration count mismatch")
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
        raise Phase45ReportError("bank tensor shape mismatch")
    weights = tf.maximum(tf.cast(weights, tf.float64), tf.constant(1.0e-300, tf.float64))
    weights = weights / tf.reduce_sum(weights)
    mean = tf.reduce_sum(weights[:, None] * theta, axis=0)
    sign = theta[:, 2] < 0.0
    finite = tf.reduce_all(tf.math.is_finite(tf.concat((tf.reshape(theta, [-1]), target, proposal, weights), axis=0)))
    return {
        "rows": n,
        "root_count": tf.size(tf.unique(roots).y),
        "ess_fraction": tf.math.reciprocal(tf.cast(n, tf.float64) * tf.reduce_sum(tf.square(weights))),
        "maximum_normalized_weight": tf.reduce_max(weights),
        "negative_mode_fraction": tf.reduce_sum(tf.where(sign, weights, tf.zeros_like(weights))),
        "negative_count": tf.reduce_sum(tf.cast(sign, tf.int32)),
        "positive_count": tf.reduce_sum(tf.cast(tf.logical_not(sign), tf.int32)),
        "theta_mean": mean,
        "target_log_range": (tf.reduce_min(target), tf.reduce_max(target)),
        "proposal_log_range": (tf.reduce_min(proposal), tf.reduce_max(proposal)),
        "log_ratio_range": (tf.reduce_min(target - proposal), tf.reduce_max(target - proposal)),
        "finite": finite,
    }


def _max_abs(value: Any) -> float:
    return float(tf.reduce_max(tf.abs(tf.cast(tf.convert_to_tensor(value), tf.float64))).numpy())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("authority-root", "fresh-root-a", "fresh-root-b", "fresh-root-c", "fresh-root-n512-a", "fresh-root-n512-b", "reference-audit-root", "audit-root", "output-root"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    paths = (args.authority_root, args.fresh_root_a, args.fresh_root_b, args.fresh_root_c, args.fresh_root_n512_a, args.fresh_root_n512_b, args.reference_audit_root, args.audit_root, args.output_root)
    if any(path.is_absolute() or ".." in path.parts for path in paths):
        raise Phase45ReportError("all paths must be repository-relative")
    output = ROOT / args.output_root
    if output.exists():
        raise Phase45ReportError(f"refusing to overwrite output root: {output}")
    started = time.perf_counter()
    root_map = {"authority": args.authority_root, "bank_a": args.fresh_root_a, "bank_b": args.fresh_root_b, "bank_c": args.fresh_root_c, "bank_n512_a": args.fresh_root_n512_a, "bank_n512_b": args.fresh_root_n512_b}
    pilot_paths = {label: ROOT / root / "pilot.json" for label, root in root_map.items()}
    pilots = {label: _load(path) for label, path in pilot_paths.items()}
    audit_path = ROOT / args.audit_root / "result.json"
    reference_path = ROOT / args.reference_audit_root / "result.json"
    audit = _load(audit_path)
    reference = _load(reference_path)
    old_measure = _load(PHASE40_REPORT)
    if audit.get("schema") != EXPECTED_AUDIT_SCHEMA or audit.get("status") != "PASS_V2_7_INDEPENDENT_N512_BOUNDARY" or audit.get("plan_version") != EXPECTED_VERSION or audit.get("target_signature") != EXPECTED_TARGET:
        raise Phase45ReportError("audit is not a passing v2.7 receipt")
    if audit.get("one_trainer_state_per_arm_for_five_audit_banks") is not True or audit.get("fresh_rows_used_for_training") or audit.get("fresh_rows_used_for_selection"):
        raise Phase45ReportError("five-bank or fresh-use boundary missing")
    if reference.get("schema") != REFERENCE_SCHEMA or reference.get("status") != "PASS_V2_6_LARGER_N_BOUNDARY" or reference.get("target_signature") != EXPECTED_TARGET:
        raise Phase45ReportError("Phase 44 reference is not passing")
    if old_measure.get("status") != "PASS_V2_2_THETA_MEASURE_SEPARATION_DIAGNOSTIC":
        raise Phase45ReportError("frozen measure source is not passing")
    counts = {label: ((512, 128) if label.startswith("bank_n512") else (256, 64)) for label in root_map}
    m0 = {}
    c0 = {}
    for label in root_map:
        m0[label] = _pilot_arm(pilots[label], "M0", EXPECTED_M0, *counts[label])
        c0[label] = _pilot_arm(pilots[label], "C0", EXPECTED_C0, *counts[label])
    if len({_sha(path) for path in pilot_paths.values()}) != 6:
        raise Phase45ReportError("pilot receipts are not distinct")
    if audit.get("authority", {}).get("pilot_sha256") != _sha(pilot_paths["authority"]):
        raise Phase45ReportError("authority pilot hash mismatch")
    for label in BANK_LABELS:
        if audit.get("fresh_banks", {}).get(label, {}).get("pilot_sha256") != _sha(pilot_paths[label]):
            raise Phase45ReportError(f"{label} pilot hash mismatch")
    for key in ("final_theta", "final_normalized_weights", "final_roots"):
        if len({str(m0[label]["receipts"][key]["sha256"]) for label in root_map}) != 6 or len({str(c0[label]["receipts"][key]["sha256"]) for label in root_map}) != 6:
            raise Phase45ReportError(f"tensor hash collision for {key}")
    summaries = {label: _summary(m0[label]) for label in root_map}
    if not all(bool(item["finite"].numpy()) for item in summaries.values()):
        raise Phase45ReportError("non-finite support summary")
    support_rows = [{"source": label, "rows": item["rows"], "roots": item["root_count"], "ess_fraction": item["ess_fraction"], "negative_mode_fraction": item["negative_mode_fraction"], "negative_count": item["negative_count"], "positive_count": item["positive_count"], "theta_mean_0": item["theta_mean"][0], "target_log_range": item["target_log_range"], "proposal_log_range": item["proposal_log_range"], "log_ratio_range": item["log_ratio_range"]} for label, item in summaries.items()]
    for label in ("train", "validation", "audit"):
        physical = old_measure["partitions"][label]["physical"]
        support_rows.append({"source": f"old_v2_2_{label}", "rows": physical["count"], "roots": physical["root_count"], "ess_fraction": physical["effective_sample_size_fraction"], "negative_mode_fraction": physical["weighted_negative_mode_fraction"], "negative_count": physical["negative_count"], "positive_count": physical["positive_count"], "theta_mean_0": physical["theta_mean"][0], "target_log_range": (physical["target_log_theta_min"], physical["target_log_theta_max"]), "proposal_log_range": (physical["proposal_log_theta_min"], physical["proposal_log_theta_max"]), "log_ratio_range": (physical["log_ratio_min"], physical["log_ratio_max"])})
    transport_rows = []
    for arm_name, arm in audit["arms"].items():
        reference_arm = reference.get("arms", {}).get(arm_name)
        if not isinstance(reference_arm, Mapping) or arm.get("reference_state_hash") != reference_arm.get("state_hash") or arm.get("gates", {}).get("reference_state_hash_match") is not True:
            raise Phase45ReportError(f"state hash mismatch for {arm_name}")
        old_mean = _max_abs(arm["validation"]["latent_weighted_mean"])
        old_cov = _max_abs(tf.cast(tf.convert_to_tensor(arm["validation"]["latent_weighted_covariance"]), tf.float64) - tf.linalg.diag(tf.linalg.diag_part(tf.cast(tf.convert_to_tensor(arm["validation"]["latent_weighted_covariance"]), tf.float64))))
        banks = {}
        for label in BANK_LABELS:
            item = arm["fresh_audits"][label]
            a_item = arm["fresh_audits"]["bank_a"]
            banks[label] = {"loss": item["loss"], "mean_max_abs": item["latent_mean_max_abs"], "covariance_offdiag_max_abs": item["latent_covariance_max_abs_offdiag"], "mean_better_than_old": item["latent_mean_max_abs"] < old_mean, "covariance_better_than_old": item["latent_covariance_max_abs_offdiag"] < old_cov, "mean_better_than_a": item["latent_mean_max_abs"] < a_item["latent_mean_max_abs"], "covariance_better_than_a": item["latent_covariance_max_abs_offdiag"] < a_item["latent_covariance_max_abs_offdiag"]}
        transport_rows.append({"arm": arm_name, "precondition": arm["precondition"], "status": arm["status"], "old_validation_mean_max_abs": old_mean, "old_validation_covariance_offdiag_max_abs": old_cov, "banks": banks, "reference_state_hash_match": True, "fresh_rows_used_for_training": arm["fresh_rows_used_for_training"], "fresh_rows_used_for_selection": arm["fresh_rows_used_for_selection"]})
    both_better_a = all(row["banks"][label]["mean_better_than_a"] and row["banks"][label]["covariance_better_than_a"] for row in transport_rows for label in ("bank_n512_a", "bank_n512_b"))
    both_below_old = all(row["banks"][label]["mean_better_than_old"] and row["banks"][label]["covariance_better_than_old"] for row in transport_rows for label in ("bank_n512_a", "bank_n512_b"))
    if both_better_a and both_below_old:
        branch = "n512_replication_support_order_reproduced_and_below_old"
    elif both_better_a:
        branch = "n512_replication_order_reproduced_but_support_mixed"
    else:
        branch = "n512_support_variability_persists"
    result = {"schema": "bayesfilter.ssl_lstm.q20.corrected_theta_n512_replication_report.v1", "status": "PASS_V2_7_INDEPENDENT_N512_REPORT", "plan_version": EXPECTED_VERSION, "role": "read_only_five_bank_n512_replication", "measure": "theta_R4", "target_signature": EXPECTED_TARGET, "explanatory_branch": branch, "n512_order_better_than_bank_a": both_better_a, "n512_both_below_old_comparator": both_below_old, "branch_is_statistical_ranking": False, "support_rows": support_rows, "transport_rows": transport_rows, "decision_table": [{"decision": "retain_theta_target", "primary_criterion": "v2.7 target/status/protocol/hash gates", "status": "pass", "veto": "none", "next_action": "retain theta authority", "not_concluded": "posterior correctness"}, {"decision": "promote_IID_whitening", "primary_criterion": "replicated finite-bank residuals", "status": "veto", "veto": "finite residuals and no population uncertainty", "next_action": "keep whitening closed", "not_concluded": "IID Gaussian law"}, {"decision": "change_objective", "primary_criterion": "independent support replication", "status": "defer", "veto": "no uncertainty-supported objective comparison", "next_action": "scope support/proposal repair before objective", "not_concluded": "objective superiority"}], "inference_status": {"hard_veto_screen": "passed", "statistically_supported_ranking": "none", "descriptive_differences": "two N=512 banks versus N=256 context", "default_readiness": "not_ready", "next_evidence": "scoped support/proposal repair or objective hypothesis with independent validation"}, "red_team": {"strongest_alternative": "both N=512 banks share proposal-support bias", "overturning_evidence": "a separately generated support/proposal route with stable downstream residuals", "weakest_evidence": "two N=512 banks and one frozen trainer state without uncertainty intervals"}, "sources": {"audit_root": args.audit_root, "audit_sha256": _sha(audit_path), "reference_audit_root": args.reference_audit_root, "reference_audit_sha256": _sha(reference_path), "phase40_report": PHASE40_REPORT, "phase40_report_sha256": _sha(PHASE40_REPORT), **{f"{label}_root": root_map[label] for label in root_map}, **{f"{label}_pilot_sha256": _sha(pilot_paths[label]) for label in root_map}}, "nonclaims": ["No IID Gaussian whitening, posterior correctness, exhaustive mode discovery, normalizer, HMC, canonical LEDH, superiority, or default promotion claim.", "The branch is descriptive and not a statistical ranking.", "No bank was pooled, dropped, trained on, or selected."], "run_manifest": {"program": PLAN, "runner": RUNNER, "command": " ".join(sys.argv), "git_commit": subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=ROOT, text=True).strip(), "git_dirty": bool(subprocess.check_output(("git", "status", "--porcelain"), cwd=ROOT, text=True).strip()), "python": sys.executable, "python_version": platform.python_version(), "tensorflow": tf.__version__, "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"], "wall_seconds": time.perf_counter() - started, "source_sha256": {"plan": _sha(PLAN), "runner": _sha(RUNNER), "audit": _sha(audit_path), "reference_audit": _sha(reference_path), "phase40_report": _sha(PHASE40_REPORT), **{f"{label}_pilot": _sha(pilot_paths[label]) for label in root_map}}}}
    _write_json(output / "result.json", result)
    lines = ["# v2.7 Independent N=512 Replication Report", "", f"Status: `{result['status']}`", "", f"Explanatory branch: `{branch}` (not a statistical ranking)", "", "| Source | rows | roots | ESS | neg-mode | theta mean[0] |", "|---|---:|---:|---:|---:|---:|"]
    for row in support_rows:
        lines.append(f"| {row['source']} | {row['rows']} | {row['roots']} | {float(row['ess_fraction']):.6f} | {float(row['negative_mode_fraction']):.6f} | {float(row['theta_mean_0']):.6f} |")
    lines.extend(["", "| Arm | A | B | C | N512-a | N512-b |", "|---|---:|---:|---:|---:|---:|"])
    for row in transport_rows:
        values = [row["banks"][label] for label in BANK_LABELS]
        cells = [f"{float(item['mean_max_abs']):.6f} / {float(item['covariance_offdiag_max_abs']):.6f}" for item in values]
        lines.append(f"| {row['arm']} | " + " | ".join(cells) + " |")
    lines.extend(["", "Hard gates passed. Bank differences remain descriptive; no whitening or posterior claim is made.", ""])
    (output / "result.md").write_text("\n".join(lines), encoding="ascii")
    print(json.dumps({"status": result["status"], "output_root": args.output_root.as_posix(), "branch": branch}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
