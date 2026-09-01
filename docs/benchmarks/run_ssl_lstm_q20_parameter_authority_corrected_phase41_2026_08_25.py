"""Audit a frozen v2.2 NeuTra training measure on an independent theta bank.

Only the old root-group training rows reach ``train_step``.  The fresh bank is
passed to the already-audited runner solely through its final-step audit hook.
This is a role-limited diagnostic and does not launch HMC.
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
    raise RuntimeError("Phase 41 requires a visible trusted GPU")
if os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH") != "true":
    raise RuntimeError("Phase 41 requires TF_FORCE_GPU_ALLOW_GROWTH=true")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HELPER_PATH = ROOT / "docs/benchmarks/run_ssl_lstm_q20_parameter_authority_corrected_phase31_2026_08_25.py"
SPEC = importlib.util.spec_from_file_location("phase31_helpers", HELPER_PATH)
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
EXPECTED_VERSION = "v2.3-independent-audit-bank"
EXPECTED_MEASURE = "theta_R4"
EXPECTED_TARGET_SIGNATURE = "9a86e60081f1b9cd288dbdb1dcbe1e9a5b5e23d9b5ef97afdb72ee95c23d7278"
EXPECTED_PROTOCOL_HASH = "a1f0f0493bb8bd594923b61ee9a92f3c8dcb72a612b64ad675b9ab7ff4723631"
EXPECTED_C0_PROTOCOL_HASH = "270fc99b81d08e23670c62fcd02e69e7452f26b5e5641187c3083faecbac7067"
ROOT_GROUP_POLICY = "root_group_stratified_v1"
EXPECTED_SCHEMA = "bayesfilter.ssl_lstm.q20.corrected_theta_neutra_boundary.v4_independent_audit_bank"


class Phase41Error(RuntimeError):
    """Raised when the frozen-training/audit boundary is invalid."""


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
        raise Phase41Error(f"refusing to overwrite artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(_safe(payload), sort_keys=True, indent=2, allow_nan=False) + "\n").encode("ascii")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(encoded)
    temporary.replace(path)


def _receipt(root: Path, name: str) -> tuple[Path, Mapping[str, Any]]:
    if root.is_absolute() or ".." in root.parts:
        raise Phase41Error(f"path must be repository-relative: {root}")
    path = ROOT / root / name
    if not path.is_file():
        raise Phase41Error(f"missing receipt: {path}")
    return path, json.loads(path.read_text(encoding="utf-8"))


def _load_tensor(receipt: Mapping[str, Any]) -> tf.Tensor:
    path = Path(str(receipt["path"]))
    if not path.is_absolute():
        path = ROOT / path
    encoded = path.read_bytes()
    if hashlib.sha256(encoded).hexdigest() != str(receipt["sha256"]):
        raise Phase41Error(f"tensor hash mismatch: {path}")
    value = tf.io.parse_tensor(encoded, out_type=getattr(tf, str(receipt["dtype"])))
    value = tf.ensure_shape(value, receipt["shape"])
    if value.dtype.is_floating or value.dtype.is_complex:
        tf.debugging.assert_all_finite(value, f"non-finite tensor {path}")
    return value


def _arm(
    pilot: Mapping[str, Any], name: str, expected_protocol_hash: str | None = None
) -> Mapping[str, Any]:
    arm = pilot.get("arms", {}).get(name)
    if not isinstance(arm, Mapping) or arm.get("status") != "PASS_THETA_MEASURE_PILOT":
        raise Phase41Error(f"{name} is not a passing theta pilot arm")
    if arm.get("target_signature") != EXPECTED_TARGET_SIGNATURE:
        raise Phase41Error(f"{name} target signature mismatch")
    if arm.get("protocol", {}).get("measure") != EXPECTED_MEASURE:
        raise Phase41Error(f"{name} measure mismatch")
    if expected_protocol_hash is not None and arm.get("configuration", {}).get("protocol_hash") != expected_protocol_hash:
        raise Phase41Error(f"{name} protocol hash mismatch")
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
        raise Phase41Error(f"theta shape is not [N,4]: {values['theta'].shape}")
    if values["weights"].shape != (n,) or values["roots"].shape != (n,):
        raise Phase41Error("cloud vector shape mismatch")
    values["weights"] = tf.maximum(tf.cast(values["weights"], tf.float64), tf.constant(1.0e-300, tf.float64))
    values["weights"] = values["weights"] / tf.reduce_sum(values["weights"])
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


def _extract_audit(arm: Mapping[str, Any], steps: int) -> Mapping[str, Any]:
    checkpoints = [
        item for item in arm["training_trace"]
        if int(item["training"]["step"]) == int(steps)
    ]
    if len(checkpoints) != 1 or "audit_checkpoint" not in checkpoints[0]:
        raise Phase41Error("fresh audit checkpoint missing at terminal step")
    return checkpoints[0]["audit_checkpoint"]


def _run_one(
    *, name: str, precondition: str, seed: tuple[int, int], old: Mapping[str, tf.Tensor],
    fresh: Mapping[str, tf.Tensor], train_indices: list[int], validation_indices: list[int],
    target: Any, steps: int,
) -> Mapping[str, Any]:
    train_idx = tf.constant(train_indices, tf.int32)
    validation_idx = tf.constant(validation_indices, tf.int32)
    train_weights = tf.maximum(tf.gather(old["weights"], train_idx), tf.constant(1.0e-300, tf.float64))
    validation_weights = tf.maximum(tf.gather(old["weights"], validation_idx), tf.constant(1.0e-300, tf.float64))
    audit_weights = tf.maximum(fresh["weights"], tf.constant(1.0e-300, tf.float64))
    affine = _affine(old["theta"], old["weights"], train_indices) if precondition == "affine" else None
    train_rows = HELPER._affine_forward(tf.gather(old["theta"], train_idx), affine)
    validation_rows = HELPER._affine_forward(tf.gather(old["theta"], validation_idx), affine)
    audit_rows = HELPER._affine_forward(fresh["theta"], affine)
    audit_rows = tf.ensure_shape(audit_rows, (int(fresh["theta"].shape[0]), 4))
    affine_round_trip_residual = tf.constant(0.0, tf.float64)
    if affine is not None:
        affine_round_trip_residual = tf.reduce_max(
            tf.abs(HELPER._affine_inverse(HELPER._affine_forward(old["theta"], affine), affine) - old["theta"])
        )
    train_weights = train_weights / tf.reduce_sum(train_weights)
    validation_weights = validation_weights / tf.reduce_sum(validation_weights)
    audit_weights = audit_weights / tf.reduce_sum(audit_weights)
    audit_weights = tf.ensure_shape(audit_weights, (int(fresh["weights"].shape[0]),))
    train_measure_weights = train_weights / tf.reduce_sum(train_weights)
    oracle = None
    if affine is not None:
        mean = tf.reduce_sum(train_measure_weights[:, tf.newaxis] * train_rows, axis=0)
        centered = train_rows - mean[tf.newaxis, :]
        covariance = tf.einsum("n,ni,nj->ij", train_measure_weights, centered, centered)
        oracle = {
            "max_abs_mean": tf.reduce_max(tf.abs(mean)),
            "max_abs_covariance_residual": tf.reduce_max(tf.abs(covariance - tf.eye(4, dtype=tf.float64))),
        }
    # The helper's audit checkpoint runs after each update.  Passing only the
    # final step makes the fresh bank observable but never an optimizer input.
    result = HELPER._run_arm(
        name=name, seed=seed, train_rows=train_rows, train_log_weights=tf.math.log(train_weights),
        validation_rows=validation_rows, validation_log_weights=tf.math.log(validation_weights),
        audit_rows=audit_rows, target=target, steps=steps, affine=affine,
        affine_round_trip_residual=affine_round_trip_residual,
        affine_training_oracle_gate=(affine is None or (
            oracle is not None and float(oracle["max_abs_mean"].numpy()) <= 1.0e-10
            and float(oracle["max_abs_covariance_residual"].numpy()) <= 1.0e-10)),
        audit_log_weights=tf.math.log(audit_weights), checkpoint_steps=(steps,),
    )
    audit = _extract_audit(result, steps)
    result = dict(result)
    result["fresh_audit"] = audit
    result["affine_training_oracle"] = oracle
    result["fresh_rows_used_for_training"] = False
    result["fresh_rows_used_for_selection"] = False
    result["precondition"] = precondition
    return result


def _markdown(result: Mapping[str, Any]) -> str:
    lines = [
        "# v2.3 Frozen-Training Independent Theta Audit", "", f"Status: `{result['status']}`", "",
        "The old v2.2 training rows are the only optimizer input. The fresh M0 bank is evaluated at the terminal step only.",
        "", "| Arm | Precondition | Status | Fresh mean max | Fresh covariance offdiag max |", "|---|---|---|---:|---:|",
    ]
    for key, arm in result["arms"].items():
        audit = arm["fresh_audit"]
        lines.append(f"| {key} | {arm['precondition']} | `{arm['status']}` | {float(audit['latent_mean_max_abs']):.6f} | {float(audit['latent_covariance_max_abs_offdiag']):.6f} |")
    lines.extend(["", "This is role-limited independent-bank evidence; no IID, posterior, HMC, LEDH, or default claim is made.", ""])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority-root", required=True, type=Path)
    parser.add_argument("--fresh-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--precondition", choices=("identity", "affine", "both"), default="both")
    parser.add_argument("--split-policy", choices=(ROOT_GROUP_POLICY,), default=ROOT_GROUP_POLICY)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", nargs=2, type=int, default=(20260825, 4011))
    args = parser.parse_args()
    if int(args.steps) <= 0:
        raise Phase41Error("steps must be positive")
    for path in (args.authority_root, args.fresh_root, args.output_root):
        if path.is_absolute() or ".." in path.parts:
            raise Phase41Error("all paths must be repository-relative")
    output = ROOT / args.output_root
    if output.exists():
        raise Phase41Error(f"refusing to overwrite output root: {output}")
    started = time.perf_counter()
    old_pilot_path, old_pilot = _receipt(args.authority_root, "pilot.json")
    fresh_pilot_path, fresh_pilot = _receipt(args.fresh_root, "pilot.json")
    if old_pilot_path == fresh_pilot_path or _sha(old_pilot_path) == _sha(fresh_pilot_path):
        raise Phase41Error("fresh pilot is not independent of authority pilot")
    if old_pilot.get("status") != "PASS_THETA_MEASURE_PILOT" or fresh_pilot.get("status") != "PASS_THETA_MEASURE_PILOT":
        raise Phase41Error("old or fresh pilot is not passing")
    old_arm = _arm(old_pilot, "M0", EXPECTED_PROTOCOL_HASH)
    fresh_arm = _arm(fresh_pilot, "M0", EXPECTED_PROTOCOL_HASH)
    old_c0 = _arm(old_pilot, "C0", EXPECTED_C0_PROTOCOL_HASH)
    fresh_c0 = _arm(fresh_pilot, "C0", EXPECTED_C0_PROTOCOL_HASH)
    if old_c0["configuration"]["protocol_hash"] != fresh_c0["configuration"]["protocol_hash"]:
        raise Phase41Error("old and fresh C0 protocol hashes differ")
    old = _cloud(old_arm)
    fresh = _cloud(fresh_arm)
    old_receipts = old_arm["receipts"]
    fresh_receipts = fresh_arm["receipts"]
    for key in ("final_theta", "final_normalized_weights", "final_roots"):
        if old_receipts[key]["sha256"] == fresh_receipts[key]["sha256"]:
            raise Phase41Error(f"fresh tensor copied for {key}")
    train_indices, validation_indices, audit_indices, split = HELPER._split_indices(
        old["theta"], policy=args.split_policy, roots=old["roots"]
    )
    if not all(split.get(key) is True for key in ("root_disjoint", "row_partition_complete", "row_partition_disjoint")):
        raise Phase41Error("frozen v2.2 split invariant failed")
    target = batch_native_complexity_posterior_target(20, jit_compile=True, principal_sqrt_backend="tensorflow_eigh")
    preconditions = ("identity", "affine") if args.precondition == "both" else (args.precondition,)
    arms: dict[str, Any] = {}
    for pre_index, precondition in enumerate(preconditions):
        for arm_index, name in enumerate(("compact", "wide_low_lr")):
            key = f"{precondition}:{name}"
            arms[key] = _run_one(
                name=name, precondition=precondition,
                seed=(int(args.seed[0]), int(args.seed[1]) + pre_index * 100 + arm_index),
                old=old, fresh=fresh, train_indices=train_indices,
                validation_indices=validation_indices, target=target, steps=int(args.steps),
            )
    candidate_pass = all(arm["status"] == "PASS_NEUTRA_BOUNDARY_CANDIDATE" for arm in arms.values())
    result = {
        "schema": EXPECTED_SCHEMA,
        "status": "PASS_V2_3_INDEPENDENT_AUDIT_BOUNDARY" if candidate_pass else "PHASE41_CANDIDATE_FAIL_REPAIR",
        "plan_version": EXPECTED_VERSION,
        "role": "frozen_v2_2_training_measure_independent_fresh_theta_audit",
        "measure": EXPECTED_MEASURE,
        "target_signature": EXPECTED_TARGET_SIGNATURE,
        "authority": {"root": args.authority_root, "pilot_sha256": _sha(old_pilot_path), "m0_protocol_hash": old_arm["configuration"]["protocol_hash"], "split_policy": args.split_policy, "train_rows": len(train_indices), "validation_rows": len(validation_indices), "historical_audit_rows_not_used": len(audit_indices)},
        "fresh_audit": {"root": args.fresh_root, "pilot_sha256": _sha(fresh_pilot_path), "m0_protocol_hash": fresh_arm["configuration"]["protocol_hash"], "particles": int(fresh["theta"].shape[0]), "untouched": True},
        "split": {**split, "fresh_bank_not_split_or_selected": True},
        "arms": arms,
        "fresh_rows_used_for_training": False,
        "fresh_rows_used_for_selection": False,
        "hmc_launched": False,
        "device": {"gpu_memory_policy": HELPER.GPU_POLICY, "physical_devices": [device.name for device in HELPER.PHYSICAL_GPUS], "logical_devices": [device.name for device in HELPER.LOGICAL_GPUS], "tf32_enabled": bool(tf.config.experimental.tensor_float_32_execution_enabled()), "jit_compile_per_function": True},
        "run_manifest": {"program": PLAN, "runner": RUNNER, "helper_runner": HELPER_PATH, "command": " ".join(sys.argv), "git_commit": subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=ROOT, text=True).strip(), "git_dirty": bool(subprocess.check_output(("git", "status", "--porcelain"), cwd=ROOT, text=True).strip()), "python": sys.executable, "python_version": platform.python_version(), "tensorflow": tf.__version__, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "tf_force_gpu_allow_growth": os.environ["TF_FORCE_GPU_ALLOW_GROWTH"], "gpu_memory_growth_verified": True, "jit_compile": True, "seed": list(args.seed), "wall_seconds": time.perf_counter() - started, "source_sha256": {"plan": _sha(PLAN), "runner": _sha(RUNNER), "helper_runner": _sha(HELPER_PATH), "target_module": _sha(TARGET_MODULE), "training_module": _sha(TRAINING_MODULE), "authority_pilot": _sha(old_pilot_path), "fresh_pilot": _sha(fresh_pilot_path)}},
        "nonclaims": ["The fresh bank is a finite independent audit, not an IID or posterior proof.", "Moment, loss, ESS, and mode differences are descriptive only.", "No HMC, convergence, exhaustive mode discovery, canonical LEDH, or default-readiness claim."],
    }
    _write_json(output / "result.json", result)
    (output / "result.md").write_text(_markdown(result), encoding="ascii")
    print(json.dumps({"status": result["status"], "output_root": args.output_root.as_posix()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
