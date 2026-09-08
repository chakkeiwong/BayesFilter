"""Run the independent N=512 theta-bank replication (v2.7).

Each arm trains on the frozen v2.2 root-group training measure and evaluates
five untouched audit banks only after the final update. The Phase 44 terminal
state hash is an exact determinism gate; this diagnostic does not launch HMC.
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
    raise RuntimeError("Phase 45 requires a visible trusted GPU")
if os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH") != "true":
    raise RuntimeError("Phase 45 requires TF_FORCE_GPU_ALLOW_GROWTH=true")

ROOT = Path(__file__).resolve().parents[2]
BASE_PATH = ROOT / "docs/benchmarks/run_ssl_lstm_q20_parameter_authority_corrected_phase44_2026_08_26.py"
SPEC = importlib.util.spec_from_file_location("phase44_base_for_phase45", BASE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load the audited Phase 44 base helpers")
BASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BASE)
tf = BASE.tf
HELPER = BASE.HELPER

from bayesfilter.nonlinear.ssl_lstm_complexity_batched_target_tf import (
    batch_native_complexity_posterior_target,
)

RUNNER = Path(__file__).resolve()
PLAN = ROOT / "docs/plans/bayesfilter-ssl-lstm-q20-parameter-authority-corrected-continuation-2026-08-25.md"
TARGET_MODULE = ROOT / "bayesfilter/nonlinear/ssl_lstm_complexity_batched_target_tf.py"
TRAINING_MODULE = ROOT / "bayesfilter/inference/neutra_weighted_training.py"
EXPECTED_VERSION = "v2.7-independent-n512-replication"
EXPECTED_MEASURE = "theta_R4"
EXPECTED_TARGET = "9a86e60081f1b9cd288dbdb1dcbe1e9a5b5e23d9b5ef97afdb72ee95c23d7278"
EXPECTED_M0 = "a1f0f0493bb8bd594923b61ee9a92f3c8dcb72a612b64ad675b9ab7ff4723631"
EXPECTED_C0 = "270fc99b81d08e23670c62fcd02e69e7452f26b5e5641187c3083faecbac7067"
ROOT_GROUP_POLICY = "root_group_stratified_v1"
EXPECTED_SCHEMA = "bayesfilter.ssl_lstm.q20.corrected_theta_neutra_boundary.v8_five_bank_mixed_n"
REFERENCE_SCHEMA = "bayesfilter.ssl_lstm.q20.corrected_theta_neutra_boundary.v7_four_bank_mixed_n"
BANK_LABELS = ("bank_a", "bank_b", "bank_c", "bank_n512_a", "bank_n512_b")
EXPECTED_TRAINER_SEED = (20260826, 4211)


class Phase45Error(RuntimeError):
    """Raised when the independent replication cannot be audited."""


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
        raise Phase45Error(f"refusing to overwrite artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(_safe(payload), sort_keys=True, indent=2, allow_nan=False) + "\n").encode("ascii")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(encoded)
    temporary.replace(path)


def _receipt(root: Path, name: str) -> tuple[Path, Mapping[str, Any]]:
    if root.is_absolute() or ".." in root.parts:
        raise Phase45Error(f"path must be repository-relative: {root}")
    path = ROOT / root / name
    if not path.is_file():
        raise Phase45Error(f"missing receipt: {path}")
    return path, json.loads(path.read_text(encoding="utf-8"))


def _load_tensor(receipt: Mapping[str, Any]) -> tf.Tensor:
    path = Path(str(receipt["path"]))
    if not path.is_absolute():
        path = ROOT / path
    encoded = path.read_bytes()
    if hashlib.sha256(encoded).hexdigest() != str(receipt["sha256"]):
        raise Phase45Error(f"tensor hash mismatch: {path}")
    value = tf.io.parse_tensor(encoded, out_type=getattr(tf, str(receipt["dtype"])))
    value = tf.ensure_shape(value, receipt["shape"])
    if value.dtype.is_floating or value.dtype.is_complex:
        tf.debugging.assert_all_finite(value, f"non-finite tensor {path}")
    return value


def _pilot_arm(pilot: Mapping[str, Any], name: str, protocol: str, particles: int, calibration_particles: int) -> Mapping[str, Any]:
    if pilot.get("status") != "PASS_THETA_MEASURE_PILOT":
        raise Phase45Error("pilot is not passing")
    arm = pilot.get("arms", {}).get(name)
    if not isinstance(arm, Mapping) or arm.get("status") != "PASS_THETA_MEASURE_PILOT":
        raise Phase45Error(f"pilot arm {name} is not passing")
    if arm.get("target_signature") != EXPECTED_TARGET:
        raise Phase45Error(f"{name} target signature mismatch")
    if arm.get("protocol", {}).get("measure") != EXPECTED_MEASURE:
        raise Phase45Error(f"{name} measure mismatch")
    if arm.get("configuration", {}).get("protocol_hash") != protocol:
        raise Phase45Error(f"{name} protocol mismatch")
    if int(arm.get("configuration", {}).get("particles", -1)) != particles:
        raise Phase45Error(f"{name} particle count mismatch")
    if int(pilot.get("calibration", {}).get("particle_count", -1)) != calibration_particles:
        raise Phase45Error(f"{name} calibration particle count mismatch")
    return arm


def _cloud(arm: Mapping[str, Any]) -> Mapping[str, tf.Tensor]:
    receipts = arm["receipts"]
    values = {key: _load_tensor(receipts[key]) for key in ("final_theta", "final_normalized_weights", "final_roots")}
    n = int(values["final_theta"].shape[0])
    if values["final_theta"].shape != (n, 4) or values["final_normalized_weights"].shape != (n,) or values["final_roots"].shape != (n,):
        raise Phase45Error("cloud shape mismatch")
    weights = tf.maximum(tf.cast(values["final_normalized_weights"], tf.float64), tf.constant(1.0e-300, tf.float64))
    return {"theta": values["final_theta"], "weights": tf.ensure_shape(weights / tf.reduce_sum(weights), (n,)), "roots": values["final_roots"]}


def _run_arm(name: str, precondition: str, seed: tuple[int, int], old: Mapping[str, tf.Tensor], banks: Mapping[str, Mapping[str, tf.Tensor]], train_indices: list[int], validation_indices: list[int], target: Any, steps: int, expected_state_hash: str) -> Mapping[str, Any]:
    train_idx = tf.constant(train_indices, tf.int32)
    validation_idx = tf.constant(validation_indices, tf.int32)
    train_weights = tf.gather(old["weights"], train_idx)
    validation_weights = tf.gather(old["weights"], validation_idx)
    train_weights = tf.maximum(train_weights, tf.constant(1.0e-300, tf.float64)); validation_weights = tf.maximum(validation_weights, tf.constant(1.0e-300, tf.float64))
    train_weights = tf.ensure_shape(train_weights / tf.reduce_sum(train_weights), (len(train_indices),)); validation_weights = tf.ensure_shape(validation_weights / tf.reduce_sum(validation_weights), (len(validation_indices),))
    affine = BASE._affine(old["theta"], old["weights"], train_indices) if precondition == "affine" else None
    train_rows = tf.ensure_shape(HELPER._affine_forward(tf.gather(old["theta"], train_idx), affine), (len(train_indices), 4))
    validation_rows = tf.ensure_shape(HELPER._affine_forward(tf.gather(old["theta"], validation_idx), affine), (len(validation_indices), 4))
    audit_rows = {label: tf.ensure_shape(HELPER._affine_forward(bank["theta"], affine), (int(bank["theta"].shape[0]), 4)) for label, bank in banks.items()}
    audit_weights = {label: tf.ensure_shape(bank["weights"] / tf.reduce_sum(bank["weights"]), (int(bank["weights"].shape[0]),)) for label, bank in banks.items()}
    config = HELPER._config(name, seed)
    trainer = HELPER.WeightedForwardKLNeuTraTrainer(config)
    trace = []
    for step in range(1, int(steps) + 1):
        update = trainer.train_step(train_rows, tf.math.log(train_weights))
        validation = trainer.validation_batch(validation_rows, tf.math.log(validation_weights))
        trace.append({"step": step, "training": HELPER._step_payload(update), "validation": BASE._moment_payload(validation)})
    audits: dict[str, Any] = {}; targets: dict[str, Any] = {}
    for label in BANK_LABELS:
        audits[label] = BASE._moment_payload(trainer.validation_batch(audit_rows[label], tf.math.log(audit_weights[label])))
        targets[label] = BASE._target_gate(target, audit_rows[label], affine)
    probe = tf.random.stateless_normal((12, 4), seed=(20260826, 7521), dtype=tf.float64)
    transformed, f_logdet = trainer.transport.forward_and_logdet(probe); recovered, i_logdet = trainer.transport.inverse_and_forward_logdet(transformed)
    finite_parity = tf.reduce_all(tf.math.is_finite(tf.concat((tf.reshape(transformed, [-1]), tf.reshape(recovered, [-1]), tf.reshape(f_logdet, [-1]), tf.reshape(i_logdet, [-1])), axis=0)))
    state_hash = trainer.state_payload()["state_hash"]
    affine_oracle = None
    if affine is not None:
        train_mean = tf.reduce_sum(affine["weights"][:, None] * train_rows, axis=0)
        centered = train_rows - train_mean[None, :]
        train_covariance = tf.einsum("n,ni,nj->ij", affine["weights"], centered, centered)
        affine_oracle = {"max_abs_mean": tf.reduce_max(tf.abs(train_mean)), "max_abs_covariance_residual": tf.reduce_max(tf.abs(train_covariance - tf.eye(4, dtype=tf.float64)))}
    gates: dict[str, Any] = {"batch_size_gt_one": len(train_indices) > 1, "batch_shape_N_by_4": train_rows.shape == (len(train_indices), 4), "xla_configured": bool(config.jit_compile), "training_trace_nonempty": bool(trace), "finite_training_trace": all(bool(tf.reduce_all(tf.math.is_finite(item["training"][key])).numpy()) for item in trace for key in ("loss", "gradient_norm", "clipped_gradient_norm")), "finite_validation": bool(tf.reduce_all(tf.math.is_finite(trace[-1]["validation"]["loss"])).numpy()), "transport_roundtrip": bool(finite_parity.numpy()) and float(tf.reduce_max(tf.abs(recovered - probe)).numpy()) <= 1.0e-8, "transport_logdet_roundtrip": bool(finite_parity.numpy()) and float(tf.reduce_max(tf.abs(i_logdet - f_logdet)).numpy()) <= 1.0e-8, "affine_training_measure_oracle": affine is None or (affine_oracle is not None and float(affine_oracle["max_abs_mean"].numpy()) <= 1.0e-10 and float(affine_oracle["max_abs_covariance_residual"].numpy()) <= 1.0e-10), "reference_state_hash_match": state_hash == expected_state_hash}
    for label in BANK_LABELS:
        gates[f"{label}_target_finite"] = bool(targets[label]["finite_value"].numpy()); gates[f"{label}_score_finite"] = bool(targets[label]["finite_score"].numpy()); gates[f"{label}_target_status_valid"] = bool(targets[label]["status_valid"].numpy())
    return {"status": "PASS_NEUTRA_BOUNDARY_CANDIDATE" if all(gates.values()) else "PHASE45_CANDIDATE_FAIL_REPAIR", "config": config.manifest_payload(), "precondition": precondition, "seed": list(seed), "steps": int(steps), "gates": gates, "validation": trace[-1]["validation"], "training_trace": trace, "fresh_audits": audits, "fresh_target_receipts": targets, "parity": {"roundtrip_max_abs": tf.reduce_max(tf.abs(recovered - probe)), "logdet_roundtrip_max_abs": tf.reduce_max(tf.abs(i_logdet - f_logdet)), "finite": finite_parity}, "affine_training_oracle": affine_oracle, "state_hash": state_hash, "reference_state_hash": expected_state_hash, "fresh_rows_used_for_training": False, "fresh_rows_used_for_selection": False, "one_trainer_state_per_five_audit_banks": True, "nonclaims": ["Finite theta banks are not IID or posterior proofs.", "Bank-specific moments and residuals are descriptive only.", "No HMC, convergence, exhaustive mode discovery, canonical LEDH, superiority, or default claim."]}


def _markdown(result: Mapping[str, Any]) -> str:
    lines = ["# v2.7 Five-Bank Frozen-Training Theta Audit", "", f"Status: `{result['status']}`", "", "| Arm | A | B | C | N512-a | N512-b |", "|---|---:|---:|---:|---:|---:|"]
    for key, arm in result["arms"].items():
        values = [arm["fresh_audits"][label] for label in BANK_LABELS]
        cells = [f"{float(item['latent_mean_max_abs']):.6f} / {float(item['latent_covariance_max_abs_offdiag']):.6f}" for item in values]
        lines.append(f"| {key} | " + " | ".join(cells) + " |")
    lines.extend(["", "Role-limited support evidence; no IID Gaussian whitening, posterior correctness, or statistical ranking is established.", ""])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("authority-root", "fresh-root-a", "fresh-root-b", "fresh-root-c", "fresh-root-n512-a", "fresh-root-n512-b", "reference-audit-root", "output-root"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=200); parser.add_argument("--seed", nargs=2, type=int, default=(20260826, 4211))
    args = parser.parse_args()
    if tuple(int(value) for value in args.seed) != EXPECTED_TRAINER_SEED:
        raise Phase45Error(f"trainer seed must remain frozen at {EXPECTED_TRAINER_SEED}")
    root_args = {"authority": args.authority_root, "bank_a": args.fresh_root_a, "bank_b": args.fresh_root_b, "bank_c": args.fresh_root_c, "bank_n512_a": args.fresh_root_n512_a, "bank_n512_b": args.fresh_root_n512_b}
    all_paths = tuple(root_args.values()) + (args.reference_audit_root, args.output_root)
    if int(args.steps) <= 0 or any(path.is_absolute() or ".." in path.parts for path in all_paths):
        raise Phase45Error("invalid steps or repository-relative path")
    output = ROOT / args.output_root
    if output.exists(): raise Phase45Error(f"refusing to overwrite output root: {output}")
    started = time.perf_counter(); pilots: dict[str, Mapping[str, Any]] = {}; pilot_paths: dict[str, Path] = {}
    for label, root in root_args.items(): pilot_paths[label], pilots[label] = _receipt(root, "pilot.json")
    reference_path, reference = _receipt(args.reference_audit_root, "result.json")
    if reference.get("schema") != REFERENCE_SCHEMA or reference.get("status") != "PASS_V2_6_LARGER_N_BOUNDARY" or reference.get("target_signature") != EXPECTED_TARGET:
        raise Phase45Error("Phase 44 reference audit is not a passing v2.6 receipt")
    expected_states = {key: str(value["state_hash"]) for key, value in reference["arms"].items()}
    if set(expected_states) != {"identity:compact", "identity:wide_low_lr", "affine:compact", "affine:wide_low_lr"}: raise Phase45Error("reference arm set mismatch")
    if len({_sha(path) for path in pilot_paths.values()}) != 6: raise Phase45Error("pilot receipts are not mutually independent")
    counts = {label: ((512, 128) if label.startswith("bank_n512") else (256, 64)) for label in root_args}
    m0 = {}; c0 = {}
    for label in root_args:
        m0[label] = _pilot_arm(pilots[label], "M0", EXPECTED_M0, *counts[label]); c0[label] = _pilot_arm(pilots[label], "C0", EXPECTED_C0, *counts[label])
    tensor_hashes = {label: {key: str(m0[label]["receipts"][key]["sha256"]) for key in ("final_theta", "final_normalized_weights", "final_roots")} for label in root_args}; c0_hashes = {label: {key: str(c0[label]["receipts"][key]["sha256"]) for key in ("final_theta", "final_normalized_weights", "final_roots")} for label in root_args}
    for key in tensor_hashes["authority"]:
        if len({tensor_hashes[label][key] for label in root_args}) != 6 or len({c0_hashes[label][key] for label in root_args}) != 6: raise Phase45Error(f"tensor hash collision for {key}")
    clouds = {label: _cloud(m0[label]) for label in root_args}; old = clouds["authority"]; banks = {label: clouds[label] for label in BANK_LABELS}
    train_indices, validation_indices, audit_indices, split = HELPER._split_indices(old["theta"], policy=ROOT_GROUP_POLICY, roots=old["roots"])
    if not all(split.get(key) is True for key in ("root_disjoint", "row_partition_complete", "row_partition_disjoint")): raise Phase45Error("frozen root-group split invariant failed")
    target = batch_native_complexity_posterior_target(20, jit_compile=True, principal_sqrt_backend="tensorflow_eigh"); arms = {}
    for pre_index, precondition in enumerate(("identity", "affine")):
        for arm_index, name in enumerate(("compact", "wide_low_lr")):
            key = f"{precondition}:{name}"; arms[key] = _run_arm(name, precondition, (int(args.seed[0]), int(args.seed[1]) + pre_index * 100 + arm_index), old, banks, train_indices, validation_indices, target, int(args.steps), expected_states[key])
    candidate_pass = all(arm["status"] == "PASS_NEUTRA_BOUNDARY_CANDIDATE" for arm in arms.values())
    result = {"schema": EXPECTED_SCHEMA, "status": "PASS_V2_7_INDEPENDENT_N512_BOUNDARY" if candidate_pass else "PHASE45_CANDIDATE_FAIL_REPAIR", "plan_version": EXPECTED_VERSION, "role": "frozen_v2_2_training_measure_three_n256_plus_two_n512_fresh_theta_audits", "measure": EXPECTED_MEASURE, "target_signature": EXPECTED_TARGET, "authority": {"root": args.authority_root, "pilot_sha256": _sha(pilot_paths["authority"]), "m0_protocol_hash": EXPECTED_M0, "split_policy": ROOT_GROUP_POLICY, "train_rows": len(train_indices), "validation_rows": len(validation_indices), "historical_audit_rows_not_used": len(audit_indices)}, "fresh_banks": {label: {"root": root_args[label], "pilot_sha256": _sha(pilot_paths[label]), "m0_protocol_hash": EXPECTED_M0, "particles": counts[label][0], "calibration_particles": counts[label][1], "untouched": True} for label in BANK_LABELS}, "reference_audit": {"root": args.reference_audit_root, "result_sha256": _sha(reference_path), "state_hashes": expected_states}, "tensor_hashes": tensor_hashes, "c0_tensor_hashes": c0_hashes, "split": {**split, "fresh_banks_not_split_or_selected": True}, "arms": arms, "one_trainer_state_per_arm_for_five_audit_banks": True, "fresh_rows_used_for_training": False, "fresh_rows_used_for_selection": False, "hmc_launched": False, "device": {"gpu_memory_policy": HELPER.GPU_POLICY, "physical_devices": [device.name for device in HELPER.PHYSICAL_GPUS], "logical_devices": [device.name for device in HELPER.LOGICAL_GPUS], "tf32_enabled": bool(tf.config.experimental.tensor_float_32_execution_enabled()), "jit_compile_per_function": True}, "run_manifest": {"program": PLAN, "runner": RUNNER, "base_runner": BASE_PATH, "command": " ".join(sys.argv), "git_commit": subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=ROOT, text=True).strip(), "git_dirty": bool(subprocess.check_output(("git", "status", "--porcelain"), cwd=ROOT, text=True).strip()), "python": sys.executable, "python_version": platform.python_version(), "tensorflow": tf.__version__, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "tf_force_gpu_allow_growth": os.environ["TF_FORCE_GPU_ALLOW_GROWTH"], "gpu_memory_growth_verified": True, "jit_compile": True, "seed": list(args.seed), "wall_seconds": time.perf_counter() - started, "source_sha256": {"plan": _sha(PLAN), "runner": _sha(RUNNER), "base_runner": _sha(BASE_PATH), "target_module": _sha(TARGET_MODULE), "training_module": _sha(TRAINING_MODULE), "reference_audit": _sha(reference_path), **{f"{label}_pilot": _sha(path) for label, path in pilot_paths.items()}}}, "nonclaims": ["Finite theta banks are not IID or posterior proofs.", "Bank-to-bank moments and residual differences are descriptive with no uncertainty model.", "No HMC, convergence, exhaustive mode discovery, canonical LEDH, superiority, or default-readiness claim."]}
    _write_json(output / "result.json", result); (output / "result.md").write_text(_markdown(result), encoding="ascii")
    print(json.dumps({"status": result["status"], "output_root": args.output_root.as_posix()}, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
