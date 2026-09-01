"""Audit three independent theta banks under one reconstructed trainer state.

Each arm creates one trainer, trains only on the frozen v2.2 training split,
and evaluates banks A, B, and C only after the final optimizer update.  The
terminal state hash must match the v2.4 reference audit. The phase is
diagnostic and does not launch HMC.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping


if os.environ.get("CUDA_VISIBLE_DEVICES") == "-1":
    raise RuntimeError("Phase 43 requires a visible trusted GPU")
if os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH") != "true":
    raise RuntimeError("Phase 43 requires TF_FORCE_GPU_ALLOW_GROWTH=true")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HELPER_PATH = ROOT / "docs/benchmarks/run_ssl_lstm_q20_parameter_authority_corrected_phase31_2026_08_25.py"
SPEC = importlib.util.spec_from_file_location("phase31_helpers_phase43", HELPER_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load audited Phase 40 helper")
HELPER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HELPER)

import tensorflow as tf

from bayesfilter.nonlinear.ssl_lstm_complexity_batched_target_tf import (
    batch_native_complexity_posterior_target,
)


RUNNER = Path(__file__).resolve()
PLAN = ROOT / "docs/plans/bayesfilter-ssl-lstm-q20-parameter-authority-corrected-continuation-2026-08-25.md"
TARGET_MODULE = ROOT / "bayesfilter/nonlinear/ssl_lstm_complexity_batched_target_tf.py"
TRAINING_MODULE = ROOT / "bayesfilter/inference/neutra_weighted_training.py"
EXPECTED_VERSION = "v2.5-third-bank-support-diagnostic"
EXPECTED_MEASURE = "theta_R4"
EXPECTED_TARGET_SIGNATURE = "9a86e60081f1b9cd288dbdb1dcbe1e9a5b5e23d9b5ef97afdb72ee95c23d7278"
EXPECTED_M0_PROTOCOL = "a1f0f0493bb8bd594923b61ee9a92f3c8dcb72a612b64ad675b9ab7ff4723631"
EXPECTED_C0_PROTOCOL = "270fc99b81d08e23670c62fcd02e69e7452f26b5e5641187c3083faecbac7067"
ROOT_GROUP_POLICY = "root_group_stratified_v1"
EXPECTED_SCHEMA = "bayesfilter.ssl_lstm.q20.corrected_theta_neutra_boundary.v6_three_independent_banks"
REFERENCE_AUDIT_SCHEMA = "bayesfilter.ssl_lstm.q20.corrected_theta_neutra_boundary.v5_two_independent_banks"


class Phase43Error(RuntimeError):
    """Raised when the three-bank boundary cannot be audited."""


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
        raise Phase43Error(f"refusing to overwrite artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(_safe(payload), sort_keys=True, indent=2, allow_nan=False) + "\n").encode("ascii")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(encoded)
    temporary.replace(path)


def _receipt(root: Path, name: str) -> tuple[Path, Mapping[str, Any]]:
    if root.is_absolute() or ".." in root.parts:
        raise Phase43Error(f"path must be repository-relative: {root}")
    path = ROOT / root / name
    if not path.is_file():
        raise Phase43Error(f"missing receipt: {path}")
    return path, json.loads(path.read_text(encoding="utf-8"))


def _load_tensor(receipt: Mapping[str, Any]) -> tf.Tensor:
    path = Path(str(receipt["path"]))
    if not path.is_absolute():
        path = ROOT / path
    encoded = path.read_bytes()
    if hashlib.sha256(encoded).hexdigest() != str(receipt["sha256"]):
        raise Phase43Error(f"tensor hash mismatch: {path}")
    value = tf.io.parse_tensor(encoded, out_type=getattr(tf, str(receipt["dtype"])))
    value = tf.ensure_shape(value, receipt["shape"])
    if value.dtype.is_floating or value.dtype.is_complex:
        tf.debugging.assert_all_finite(value, f"non-finite tensor {path}")
    return value


def _arm(
    pilot: Mapping[str, Any], name: str, expected_protocol: str
) -> Mapping[str, Any]:
    if pilot.get("status") != "PASS_THETA_MEASURE_PILOT":
        raise Phase43Error("pilot is not a passing theta-measure receipt")
    arm = pilot.get("arms", {}).get(name)
    if not isinstance(arm, Mapping) or arm.get("status") != "PASS_THETA_MEASURE_PILOT":
        raise Phase43Error(f"{name} arm is not passing")
    if arm.get("target_signature") != EXPECTED_TARGET_SIGNATURE:
        raise Phase43Error(f"{name} target signature mismatch")
    if arm.get("protocol", {}).get("measure") != EXPECTED_MEASURE:
        raise Phase43Error(f"{name} measure mismatch")
    if arm.get("configuration", {}).get("protocol_hash") != expected_protocol:
        raise Phase43Error(f"{name} protocol hash mismatch")
    return arm


def _cloud(arm: Mapping[str, Any]) -> Mapping[str, tf.Tensor]:
    receipts = arm["receipts"]
    values = {
        "theta": _load_tensor(receipts["final_theta"]),
        "weights": _load_tensor(receipts["final_normalized_weights"]),
        "roots": _load_tensor(receipts["final_roots"]),
    }
    n = int(values["theta"].shape[0])
    if values["theta"].shape.rank != 2 or values["theta"].shape[1] != 4:
        raise Phase43Error(f"theta shape is not [N,4]: {values['theta'].shape}")
    if values["weights"].shape != (n,) or values["roots"].shape != (n,):
        raise Phase43Error("cloud vector shape mismatch")
    weights = tf.maximum(tf.cast(values["weights"], tf.float64), tf.constant(1.0e-300, tf.float64))
    values["weights"] = tf.ensure_shape(weights / tf.reduce_sum(weights), (n,))
    return values


def _affine(theta: tf.Tensor, weights: tf.Tensor, indices: list[int]) -> Mapping[str, tf.Tensor]:
    idx = tf.constant(indices, tf.int32)
    rows = tf.gather(theta, idx)
    raw = tf.gather(weights, idx)
    normalized = raw / tf.reduce_sum(raw)
    mean = tf.reduce_sum(normalized[:, tf.newaxis] * rows, axis=0)
    centered = rows - mean[tf.newaxis, :]
    covariance = tf.einsum("n,ni,nj->ij", normalized, centered, centered)
    covariance = 0.5 * (covariance + tf.transpose(covariance))
    eigenvalues = tf.linalg.eigvalsh(covariance)
    tf.debugging.assert_positive(eigenvalues, "training covariance eigenvalues")
    chol = tf.linalg.cholesky(covariance)
    return {
        "mean": mean,
        "covariance": covariance,
        "eigenvalues": eigenvalues,
        "chol": chol,
        "logdet": tf.reduce_sum(tf.math.log(tf.linalg.diag_part(chol))),
        "weights": normalized,
        "moment_measure": "frozen_v2_2_root_group_train_weights",
    }


def _moment_payload(validation: Any) -> Mapping[str, Any]:
    covariance = validation.latent_weighted_covariance
    offdiag = covariance - tf.linalg.diag(tf.linalg.diag_part(covariance))
    return {
        "loss": validation.loss,
        "effective_sample_size_fraction": validation.effective_sample_size_fraction,
        "maximum_normalized_weight": validation.maximum_normalized_weight,
        "latent_weighted_mean": validation.latent_weighted_mean,
        "latent_weighted_covariance": covariance,
        "latent_mean_max_abs": tf.reduce_max(tf.abs(validation.latent_weighted_mean)),
        "latent_covariance_max_abs_offdiag": tf.reduce_max(tf.abs(offdiag)),
    }


def _target_gate(target: Any, rows: tf.Tensor, affine: Mapping[str, tf.Tensor] | None) -> Mapping[str, Any]:
    physical = HELPER._affine_inverse(rows, affine)
    value, score, status = target.neutra_batch_log_prob_and_grad_status(physical)
    valid = HELPER._status_valid(status)
    return {
        "target_log_theta": value,
        "score": score,
        "status_code": status["status_code"],
        "valid": valid,
        "finite_value": tf.reduce_all(tf.math.is_finite(value)),
        "finite_score": tf.reduce_all(tf.math.is_finite(score)),
        "status_valid": tf.reduce_all(valid),
    }


def _run_arm(
    *,
    name: str,
    precondition: str,
    seed: tuple[int, int],
    old: Mapping[str, tf.Tensor],
    banks: Mapping[str, Mapping[str, tf.Tensor]],
    train_indices: list[int],
    validation_indices: list[int],
    target: Any,
    steps: int,
    expected_state_hash: str,
) -> Mapping[str, Any]:
    train_idx = tf.constant(train_indices, tf.int32)
    validation_idx = tf.constant(validation_indices, tf.int32)
    train_weights = tf.gather(old["weights"], train_idx)
    validation_weights = tf.gather(old["weights"], validation_idx)
    train_weights = tf.maximum(train_weights, tf.constant(1.0e-300, tf.float64))
    validation_weights = tf.maximum(validation_weights, tf.constant(1.0e-300, tf.float64))
    train_weights = tf.ensure_shape(train_weights / tf.reduce_sum(train_weights), (len(train_indices),))
    validation_weights = tf.ensure_shape(validation_weights / tf.reduce_sum(validation_weights), (len(validation_indices),))
    affine = _affine(old["theta"], old["weights"], train_indices) if precondition == "affine" else None
    train_rows = HELPER._affine_forward(tf.gather(old["theta"], train_idx), affine)
    validation_rows = HELPER._affine_forward(tf.gather(old["theta"], validation_idx), affine)
    train_rows = tf.ensure_shape(train_rows, (len(train_indices), 4))
    validation_rows = tf.ensure_shape(validation_rows, (len(validation_indices), 4))
    affine_round_trip_residual = tf.constant(0.0, tf.float64)
    oracle = None
    if affine is not None:
        affine_round_trip_residual = tf.reduce_max(
            tf.abs(HELPER._affine_inverse(HELPER._affine_forward(old["theta"], affine), affine) - old["theta"])
        )
        train_mean = tf.reduce_sum(affine["weights"][:, tf.newaxis] * train_rows, axis=0)
        train_centered = train_rows - train_mean[tf.newaxis, :]
        train_covariance = tf.einsum("n,ni,nj->ij", affine["weights"], train_centered, train_centered)
        oracle = {
            "max_abs_mean": tf.reduce_max(tf.abs(train_mean)),
            "max_abs_covariance_residual": tf.reduce_max(tf.abs(train_covariance - tf.eye(4, dtype=tf.float64))),
        }
    audit_rows: dict[str, tf.Tensor] = {}
    audit_weights: dict[str, tf.Tensor] = {}
    for label, bank in banks.items():
        audit_rows[label] = tf.ensure_shape(
            HELPER._affine_forward(bank["theta"], affine),
            (int(bank["theta"].shape[0]), 4),
        )
        weights = tf.maximum(bank["weights"], tf.constant(1.0e-300, tf.float64))
        audit_weights[label] = tf.ensure_shape(
            weights / tf.reduce_sum(weights), (int(bank["weights"].shape[0]),)
        )
    config = HELPER._config(name, seed)
    trainer = HELPER.WeightedForwardKLNeuTraTrainer(config)
    trace: list[Mapping[str, Any]] = []
    # The only optimizer input is the old training partition.
    for step in range(1, int(steps) + 1):
        update = trainer.train_step(train_rows, tf.math.log(train_weights))
        validation = trainer.validation_batch(validation_rows, tf.math.log(validation_weights))
        trace.append({
            "step": step,
            "training": HELPER._step_payload(update),
            "validation": _moment_payload(validation),
        })
    # All independent banks are observed only after the last update, using the
    # same trainer state and no checkpoint-selection branch.
    audits: dict[str, Any] = {}
    target_receipts: dict[str, Any] = {}
    for label in ("bank_a", "bank_b", "bank_c"):
        validation_audit = trainer.validation_batch(
            audit_rows[label], tf.math.log(audit_weights[label])
        )
        target_receipt = _target_gate(target, audit_rows[label], affine)
        audits[label] = _moment_payload(validation_audit)
        target_receipts[label] = target_receipt
    probe = tf.random.stateless_normal((12, 4), seed=(20260826, 7421), dtype=tf.float64)
    transformed, forward_logdet = trainer.transport.forward_and_logdet(probe)
    recovered, inverse_logdet = trainer.transport.inverse_and_forward_logdet(transformed)
    parity = {
        "roundtrip_max_abs": tf.reduce_max(tf.abs(recovered - probe)),
        "logdet_roundtrip_max_abs": tf.reduce_max(tf.abs(inverse_logdet - forward_logdet)),
        "finite": tf.reduce_all(tf.math.is_finite(tf.concat((tf.reshape(transformed, [-1]), tf.reshape(recovered, [-1]), tf.reshape(forward_logdet, [-1]), tf.reshape(inverse_logdet, [-1])), axis=0))),
    }
    state_hash = trainer.state_payload()["state_hash"]
    gates: dict[str, Any] = {
        "batch_size_gt_one": len(train_indices) > 1,
        "batch_shape_N_by_4": train_rows.shape == (len(train_indices), 4),
        "xla_configured": bool(config.jit_compile),
        "training_trace_nonempty": bool(trace),
        "finite_training_trace": all(
            bool(tf.reduce_all(tf.math.is_finite(item["training"][key])).numpy())
            for item in trace for key in ("loss", "gradient_norm", "clipped_gradient_norm")
        ),
        "finite_validation": bool(tf.reduce_all(tf.math.is_finite(trace[-1]["validation"]["loss"])).numpy()),
        "transport_roundtrip": bool(parity["finite"].numpy()) and float(parity["roundtrip_max_abs"].numpy()) <= 1.0e-8,
        "transport_logdet_roundtrip": bool(parity["finite"].numpy()) and float(parity["logdet_roundtrip_max_abs"].numpy()) <= 1.0e-8,
        "affine_training_measure_oracle": affine is None or (
            oracle is not None and float(oracle["max_abs_mean"].numpy()) <= 1.0e-10
            and float(oracle["max_abs_covariance_residual"].numpy()) <= 1.0e-10
        ),
        "reference_state_hash_match": state_hash == expected_state_hash,
    }
    for label in ("bank_a", "bank_b", "bank_c"):
        gates[f"{label}_target_finite"] = bool(target_receipts[label]["finite_value"].numpy())
        gates[f"{label}_score_finite"] = bool(target_receipts[label]["finite_score"].numpy())
        gates[f"{label}_target_status_valid"] = bool(target_receipts[label]["status_valid"].numpy())
    return {
        "status": "PASS_NEUTRA_BOUNDARY_CANDIDATE" if all(gates.values()) else "PHASE43_CANDIDATE_FAIL_REPAIR",
        "config": config.manifest_payload(),
        "precondition": precondition,
        "seed": list(seed),
        "steps": int(steps),
        "gates": gates,
        "training_rows": len(train_indices),
        "validation": trace[-1]["validation"],
        "training_trace": trace,
        "fresh_audits": audits,
        "fresh_target_receipts": target_receipts,
        "parity": parity,
        "affine_training_oracle": oracle,
        "affine_round_trip_residual": affine_round_trip_residual,
        "state_hash": state_hash,
        "reference_state_hash": expected_state_hash,
        "fresh_rows_used_for_training": False,
        "fresh_rows_used_for_selection": False,
        "one_trainer_state_for_three_banks": True,
        "nonclaims": [
            "The three fresh banks are finite audits, not IID or posterior proofs.",
            "Bank-specific moments, losses, ESS, and residuals are descriptive only.",
            "No HMC, convergence, exhaustive mode discovery, canonical LEDH, or default claim.",
        ],
    }


def _markdown(result: Mapping[str, Any]) -> str:
    lines = [
        "# v2.5 Three-Bank Frozen-Training Theta Audit",
        "",
        f"Status: `{result['status']}`",
        "",
        "Each arm uses one trainer state trained on the old v2.2 train split, then evaluates banks A, B, and C after the final update.",
        "",
        "| Arm | Precondition | State hash | A mean/cov | B mean/cov | C mean/cov |",
        "|---|---|---|---:|---:|---:|",
    ]
    for key, arm in result["arms"].items():
        a = arm["fresh_audits"]["bank_a"]
        b = arm["fresh_audits"]["bank_b"]
        c = arm["fresh_audits"]["bank_c"]
        lines.append(
            f"| {key} | {arm['precondition']} | `{arm['gates']['reference_state_hash_match']}` | {float(a['latent_mean_max_abs']):.6f} / {float(a['latent_covariance_max_abs_offdiag']):.6f} | {float(b['latent_mean_max_abs']):.6f} / {float(b['latent_covariance_max_abs_offdiag']):.6f} | {float(c['latent_mean_max_abs']):.6f} / {float(c['latent_covariance_max_abs_offdiag']):.6f} |"
        )
    lines.extend([
        "",
        "This is role-limited support evidence. Exact v2.4 state-hash reconstruction is required; no IID Gaussian whitening, posterior correctness, or statistical superiority is established.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority-root", required=True, type=Path)
    parser.add_argument("--fresh-root-a", required=True, type=Path)
    parser.add_argument("--fresh-root-b", required=True, type=Path)
    parser.add_argument("--fresh-root-c", required=True, type=Path)
    parser.add_argument("--reference-audit-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", nargs=2, type=int, default=(20260826, 4211))
    args = parser.parse_args()
    if int(args.steps) <= 0:
        raise Phase43Error("steps must be positive")
    for path in (args.authority_root, args.fresh_root_a, args.fresh_root_b, args.fresh_root_c, args.reference_audit_root, args.output_root):
        if path.is_absolute() or ".." in path.parts:
            raise Phase43Error("all paths must be repository-relative")
    output = ROOT / args.output_root
    if output.exists():
        raise Phase43Error(f"refusing to overwrite output root: {output}")
    started = time.perf_counter()
    authority_path, authority_pilot = _receipt(args.authority_root, "pilot.json")
    fresh_a_path, fresh_a_pilot = _receipt(args.fresh_root_a, "pilot.json")
    fresh_b_path, fresh_b_pilot = _receipt(args.fresh_root_b, "pilot.json")
    fresh_c_path, fresh_c_pilot = _receipt(args.fresh_root_c, "pilot.json")
    reference_audit_path, reference_audit = _receipt(args.reference_audit_root, "result.json")
    if reference_audit.get("schema") != REFERENCE_AUDIT_SCHEMA or reference_audit.get("status") != "PASS_V2_4_TWO_BANK_BOUNDARY":
        raise Phase43Error("reference audit is not a passing v2.4 boundary receipt")
    if reference_audit.get("target_signature") != EXPECTED_TARGET_SIGNATURE:
        raise Phase43Error("reference audit target signature mismatch")
    reference_arms = reference_audit.get("arms", {})
    expected_state_hashes: dict[str, str] = {}
    for arm_key in ("identity:compact", "identity:wide_low_lr", "affine:compact", "affine:wide_low_lr"):
        item = reference_arms.get(arm_key)
        if not isinstance(item, Mapping) or not isinstance(item.get("state_hash"), str):
            raise Phase43Error(f"missing v2.4 reference state hash for {arm_key}")
        expected_state_hashes[arm_key] = str(item["state_hash"])
    pilot_paths = (authority_path, fresh_a_path, fresh_b_path, fresh_c_path)
    if len({_sha(path) for path in pilot_paths}) != 4:
        raise Phase43Error("authority and fresh pilot receipts are not all distinct")
    old_m0 = _arm(authority_pilot, "M0", EXPECTED_M0_PROTOCOL)
    fresh_a_m0 = _arm(fresh_a_pilot, "M0", EXPECTED_M0_PROTOCOL)
    fresh_b_m0 = _arm(fresh_b_pilot, "M0", EXPECTED_M0_PROTOCOL)
    fresh_c_m0 = _arm(fresh_c_pilot, "M0", EXPECTED_M0_PROTOCOL)
    old_c0 = _arm(authority_pilot, "C0", EXPECTED_C0_PROTOCOL)
    fresh_a_c0 = _arm(fresh_a_pilot, "C0", EXPECTED_C0_PROTOCOL)
    fresh_b_c0 = _arm(fresh_b_pilot, "C0", EXPECTED_C0_PROTOCOL)
    fresh_c_c0 = _arm(fresh_c_pilot, "C0", EXPECTED_C0_PROTOCOL)
    old = _cloud(old_m0)
    banks = {"bank_a": _cloud(fresh_a_m0), "bank_b": _cloud(fresh_b_m0), "bank_c": _cloud(fresh_c_m0)}
    tensor_hashes: dict[str, dict[str, str]] = {}
    for label, arm in (("authority", old_m0), ("bank_a", fresh_a_m0), ("bank_b", fresh_b_m0), ("bank_c", fresh_c_m0)):
        tensor_hashes[label] = {
            key: str(arm["receipts"][key]["sha256"])
            for key in ("final_theta", "final_normalized_weights", "final_roots")
        }
    c0_tensor_hashes: dict[str, dict[str, str]] = {}
    for label, arm in (("authority", old_c0), ("bank_a", fresh_a_c0), ("bank_b", fresh_b_c0), ("bank_c", fresh_c_c0)):
        c0_tensor_hashes[label] = {
            key: str(arm["receipts"][key]["sha256"])
            for key in ("final_theta", "final_normalized_weights", "final_roots")
        }
    for key in tensor_hashes["authority"]:
        if len({tensor_hashes[label][key] for label in tensor_hashes}) != 4:
            raise Phase43Error(f"M0 tensor hash collision for {key}")
        if len({c0_tensor_hashes[label][key] for label in c0_tensor_hashes}) != 4:
            raise Phase43Error(f"C0 tensor hash collision for {key}")
    train_indices, validation_indices, audit_indices, split = HELPER._split_indices(
        old["theta"], policy=ROOT_GROUP_POLICY, roots=old["roots"]
    )
    if not all(split.get(key) is True for key in ("root_disjoint", "row_partition_complete", "row_partition_disjoint")):
        raise Phase43Error("frozen root-group split invariant failed")
    target = batch_native_complexity_posterior_target(20, jit_compile=True, principal_sqrt_backend="tensorflow_eigh")
    arms: dict[str, Any] = {}
    for pre_index, precondition in enumerate(("identity", "affine")):
        for arm_index, name in enumerate(("compact", "wide_low_lr")):
            key = f"{precondition}:{name}"
            arms[key] = _run_arm(
                name=name,
                precondition=precondition,
                seed=(int(args.seed[0]), int(args.seed[1]) + pre_index * 100 + arm_index),
                old=old,
                banks=banks,
                train_indices=train_indices,
                validation_indices=validation_indices,
                target=target,
                steps=int(args.steps),
                expected_state_hash=expected_state_hashes[key],
            )
    candidate_pass = all(arm["status"] == "PASS_NEUTRA_BOUNDARY_CANDIDATE" for arm in arms.values())
    result = {
        "schema": EXPECTED_SCHEMA,
        "status": "PASS_V2_5_THREE_BANK_BOUNDARY" if candidate_pass else "PHASE43_CANDIDATE_FAIL_REPAIR",
        "plan_version": EXPECTED_VERSION,
        "role": "frozen_v2_2_training_measure_three_independent_fresh_theta_audits",
        "measure": EXPECTED_MEASURE,
        "target_signature": EXPECTED_TARGET_SIGNATURE,
        "authority": {
            "root": args.authority_root,
            "pilot_sha256": _sha(authority_path),
            "m0_protocol_hash": EXPECTED_M0_PROTOCOL,
            "split_policy": ROOT_GROUP_POLICY,
            "train_rows": len(train_indices),
            "validation_rows": len(validation_indices),
            "historical_audit_rows_not_used": len(audit_indices),
        },
        "fresh_banks": {
            "bank_a": {"root": args.fresh_root_a, "pilot_sha256": _sha(fresh_a_path), "m0_protocol_hash": EXPECTED_M0_PROTOCOL, "particles": int(banks["bank_a"]["theta"].shape[0]), "untouched": True},
            "bank_b": {"root": args.fresh_root_b, "pilot_sha256": _sha(fresh_b_path), "m0_protocol_hash": EXPECTED_M0_PROTOCOL, "particles": int(banks["bank_b"]["theta"].shape[0]), "untouched": True},
            "bank_c": {"root": args.fresh_root_c, "pilot_sha256": _sha(fresh_c_path), "m0_protocol_hash": EXPECTED_M0_PROTOCOL, "particles": int(banks["bank_c"]["theta"].shape[0]), "untouched": True},
        },
        "reference_audit": {"root": args.reference_audit_root, "result_sha256": _sha(reference_audit_path), "state_hashes": expected_state_hashes},
        "tensor_hashes": tensor_hashes,
        "c0_tensor_hashes": c0_tensor_hashes,
        "split": {**split, "fresh_banks_not_split_or_selected": True},
        "arms": arms,
        "one_trainer_state_per_arm_for_three_banks": True,
        "fresh_rows_used_for_training": False,
        "fresh_rows_used_for_selection": False,
        "hmc_launched": False,
        "device": {
            "gpu_memory_policy": HELPER.GPU_POLICY,
            "physical_devices": [device.name for device in HELPER.PHYSICAL_GPUS],
            "logical_devices": [device.name for device in HELPER.LOGICAL_GPUS],
            "tf32_enabled": bool(tf.config.experimental.tensor_float_32_execution_enabled()),
            "jit_compile_per_function": True,
        },
        "run_manifest": {
            "program": PLAN,
            "runner": RUNNER,
            "helper_runner": HELPER_PATH,
            "command": " ".join(sys.argv),
            "git_commit": subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=ROOT, text=True).strip(),
            "git_dirty": bool(subprocess.check_output(("git", "status", "--porcelain"), cwd=ROOT, text=True).strip()),
            "python": sys.executable,
            "python_version": platform.python_version(),
            "tensorflow": tf.__version__,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "tf_force_gpu_allow_growth": os.environ["TF_FORCE_GPU_ALLOW_GROWTH"],
            "gpu_memory_growth_verified": True,
            "jit_compile": True,
            "seed": list(args.seed),
            "wall_seconds": time.perf_counter() - started,
            "source_sha256": {
                "plan": _sha(PLAN),
                "runner": _sha(RUNNER),
                "helper_runner": _sha(HELPER_PATH),
                "target_module": _sha(TARGET_MODULE),
                "training_module": _sha(TRAINING_MODULE),
                "authority_pilot": _sha(authority_path),
                "fresh_a_pilot": _sha(fresh_a_path),
                "fresh_b_pilot": _sha(fresh_b_path),
                "fresh_c_pilot": _sha(fresh_c_path),
                "reference_audit": _sha(reference_audit_path),
            },
        },
        "nonclaims": [
            "Three fresh banks are finite audits, not an IID or posterior proof.",
            "Bank-to-bank residual differences are descriptive with one frozen state and no uncertainty model.",
            "No HMC, convergence, exhaustive mode discovery, canonical LEDH, superiority, or default-readiness claim.",
        ],
    }
    _write_json(output / "result.json", result)
    (output / "result.md").write_text(_markdown(result), encoding="ascii")
    print(json.dumps({"status": result["status"], "output_root": args.output_root.as_posix()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
