"""Run paired identity versus theta-space MH mutation diagnostics for q=20.

Three fresh M0 pilot receipts provide protocol-bound seed roles.  The runner
regenerates each initial proposal cloud from its recorded M0 seed, then runs
identity and symmetric random-walk Metropolis mutation from identical tensors.
This is finite support evidence only; it does not launch HMC.
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

if os.environ.get("CUDA_VISIBLE_DEVICES") == "-1":
    raise RuntimeError("Phase 47 q=20 runner requires a visible trusted GPU")
if os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH") != "true":
    raise RuntimeError("Phase 47 q=20 runner requires TF_FORCE_GPU_ALLOW_GROWTH=true")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf

from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth


GPU_POLICY = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
PHYSICAL_GPUS = tuple(tf.config.list_physical_devices("GPU"))
LOGICAL_GPUS = tuple(tf.config.list_logical_devices("GPU"))
if not PHYSICAL_GPUS or not LOGICAL_GPUS:
    raise RuntimeError("Phase 47 GPU memory policy produced no logical GPU")
try:
    tf.config.experimental.enable_tensor_float_32_execution(True)
except (AttributeError, RuntimeError):
    pass

from bayesfilter.nonlinear.ssl_lstm_complexity_batched_target_tf import (
    batch_native_complexity_posterior_target,
)
from bayesfilter.testing.annealed_smc_tf import (
    normalized_weight_diagnostics,
    systematic_resample_indices,
)
from bayesfilter.testing.importance_sampling_tf import (
    gaussian_mixture_log_prob,
    sample_gaussian_mixture,
)


RUNNER = Path(__file__).resolve()
PLAN = ROOT / "docs/plans/bayesfilter-ssl-lstm-q20-parameter-authority-corrected-continuation-2026-08-25.md"
GEOMETRY = ROOT / "docs/plans/artifacts/ssl-lstm-q20-seed-b-neutra-mode-failure-root-cause-2026-08-10/r1/geometry.json"
TARGET_MODULE = ROOT / "bayesfilter/nonlinear/ssl_lstm_complexity_batched_target_tf.py"
SMC_MODULE = ROOT / "bayesfilter/testing/annealed_smc_tf.py"
IMPORTANCE_MODULE = ROOT / "bayesfilter/testing/importance_sampling_tf.py"

EXPECTED_VERSION = "v2.9-invariant-mutation-diagnostic"
EXPECTED_MEASURE = "theta_R4"
EXPECTED_TARGET = "9a86e60081f1b9cd288dbdb1dcbe1e9a5b5e23d9b5ef97afdb72ee95c23d7278"
EXPECTED_M0 = "a1f0f0493bb8bd594923b61ee9a92f3c8dcb72a612b64ad675b9ab7ff4723631"
EXPECTED_C0 = "270fc99b81d08e23670c62fcd02e69e7452f26b5e5641187c3083faecbac7067"
EXPECTED_PILOT_SCHEMA = "bayesfilter.ssl_lstm.q20.corrected_theta_authority_pilot.v1"
EXPECTED_ARM_SCHEMA = "bayesfilter.ssl_lstm.q20.corrected_theta_authority.arm.v1"
SCHEDULE = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
PARTICLES = 256
CALIBRATION_PARTICLES = 64
DEFENSIVE_EPSILON = 0.20
SAFE_STD = 2.0
MODE_AXIS = 2
MH_SIGMA = 0.35
MH_STEPS = 2
RESAMPLING_SEED_OFFSET = 1000
MH_SEED_OFFSET = 20000


class Phase47Error(RuntimeError):
    """Raised when the paired mutation boundary cannot be audited."""


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
        raise Phase47Error(f"refusing to overwrite artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(_safe(payload), sort_keys=True, indent=2, allow_nan=False) + "\n").encode("ascii")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(encoded)
    temporary.replace(path)


def _write_tensor(path: Path, value: Any) -> Mapping[str, Any]:
    if path.exists():
        raise Phase47Error(f"refusing to overwrite tensor: {path}")
    tensor = tf.convert_to_tensor(value)
    encoded = bytes(tf.io.serialize_tensor(tensor).numpy())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)
    return {"path": path.as_posix(), "sha256": hashlib.sha256(encoded).hexdigest(), "dtype": tensor.dtype.name, "shape": list(tensor.shape), "bytes": len(encoded)}


def _load_json(root: Path, name: str) -> tuple[Path, Mapping[str, Any]]:
    if root.is_absolute() or ".." in root.parts:
        raise Phase47Error(f"path must be repository-relative: {root}")
    path = ROOT / root / name
    if not path.is_file():
        raise Phase47Error(f"missing pilot receipt: {path}")
    return path, json.loads(path.read_text(encoding="utf-8"))


def _load_geometry() -> Mapping[str, tf.Tensor]:
    payload = json.loads(GEOMETRY.read_text(encoding="utf-8"))
    if payload.get("status") != "GEOMETRY_DIAGNOSTIC_COMPLETED":
        raise Phase47Error("geometry calibration artifact is incomplete")
    labels = ("minus", "plus")
    means = tf.constant([payload["representatives"][label]["position"] for label in labels], tf.float64)
    precisions = tf.constant([payload["source_curvature"][label]["records"][-1]["precision"] for label in labels], tf.float64)
    covariances = tf.linalg.inv(precisions)
    tf.debugging.assert_all_finite(covariances, "geometry covariance")
    tf.debugging.assert_positive(tf.linalg.eigvalsh(covariances), "geometry covariance eigenvalues")
    return {"means": means, "covariances": covariances, "probabilities": tf.constant((0.5, 0.5), tf.float64), "center": tf.reduce_mean(means, axis=0)}


def _safe_log_prob(theta: tf.Tensor, center: tf.Tensor) -> tf.Tensor:
    scale = tf.constant(SAFE_STD, tf.float64)
    standardized = (theta - center[tf.newaxis, :]) / scale
    return -0.5 * tf.reduce_sum(tf.square(standardized), axis=1) - 4.0 * (tf.math.log(scale) + 0.5 * tf.constant(1.8378770664093453, tf.float64))


def _proposal_log_theta(theta: tf.Tensor, chart: Mapping[str, tf.Tensor]) -> tf.Tensor:
    local = gaussian_mixture_log_prob(theta, chart["probabilities"], chart["means"], chart["covariances"])
    safe = _safe_log_prob(theta, chart["center"])
    eps = tf.constant(DEFENSIVE_EPSILON, tf.float64)
    return tf.reduce_logsumexp(tf.stack((tf.math.log1p(-eps) + local, tf.math.log(eps) + safe), axis=1), axis=1)


def _sample_theta(seed: tuple[int, int], chart: Mapping[str, tf.Tensor]) -> tuple[tf.Tensor, tf.Tensor]:
    split = tf.random.experimental.stateless_split(tf.constant(seed, tf.int32), 4)
    local, labels = sample_gaussian_mixture(PARTICLES, chart["probabilities"], chart["means"], chart["covariances"], seed=tuple(int(value) for value in split[0].numpy()))
    safe_noise = tf.random.stateless_normal((PARTICLES, 4), seed=split[1], dtype=tf.float64)
    safe = chart["center"][tf.newaxis, :] + SAFE_STD * safe_noise
    choose_safe = tf.random.stateless_uniform((PARTICLES,), seed=split[2], dtype=tf.float64) < DEFENSIVE_EPSILON
    theta = tf.where(choose_safe[:, None], safe, local)
    component = tf.where(choose_safe, tf.fill((PARTICLES,), 2), labels)
    return tf.ensure_shape(theta, (PARTICLES, 4)), tf.ensure_shape(component, (PARTICLES,))


def _evaluate(theta: tf.Tensor, target: Any, chart: Mapping[str, tf.Tensor]) -> Mapping[str, tf.Tensor]:
    theta = tf.ensure_shape(tf.convert_to_tensor(theta, tf.float64), (PARTICLES, 4))
    value, score, status = target.neutra_batch_log_prob_and_grad_status(theta)
    finite = tf.logical_and(tf.math.is_finite(value), tf.reduce_all(tf.math.is_finite(score), axis=1))
    valid = tf.logical_and(finite, tf.logical_and(tf.equal(tf.convert_to_tensor(status["status_code"], tf.int32), 0), tf.cast(status["valid_pre_regularized_score"], tf.bool)))
    proposal = _proposal_log_theta(theta, chart)
    finite = tf.logical_and(finite, tf.math.is_finite(proposal))
    valid = tf.logical_and(valid, finite)
    return {"theta": theta, "target": tf.ensure_shape(tf.convert_to_tensor(value, tf.float64), (PARTICLES,)), "proposal": tf.ensure_shape(proposal, (PARTICLES,)), "valid": tf.ensure_shape(valid, (PARTICLES,)), "status_code": tf.ensure_shape(tf.convert_to_tensor(status["status_code"], tf.int32), (PARTICLES,))}


@tf.function(input_signature=(tf.TensorSpec((PARTICLES, 4), tf.float64), tf.TensorSpec((PARTICLES, 4), tf.float64), tf.TensorSpec((PARTICLES,), tf.float64), tf.TensorSpec((PARTICLES,), tf.float64), tf.TensorSpec((PARTICLES,), tf.bool), tf.TensorSpec((PARTICLES,), tf.float64)), jit_compile=True, reduce_retracing=False)
def _mh_accept(current: tf.Tensor, candidate: tf.Tensor, current_bridge: tf.Tensor, candidate_bridge: tf.Tensor, candidate_valid: tf.Tensor, uniforms: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    log_alpha = tf.minimum(tf.constant(0.0, tf.float64), candidate_bridge - current_bridge)
    accepted = tf.logical_and(candidate_valid, tf.math.log(uniforms) < log_alpha)
    return tf.where(accepted[:, None], candidate, current), accepted, log_alpha


def _summary(theta: tf.Tensor, weights: tf.Tensor, roots: tf.Tensor) -> Mapping[str, Any]:
    weights = tf.ensure_shape(weights / tf.reduce_sum(weights), (PARTICLES,))
    mean = tf.reduce_sum(weights[:, None] * theta, axis=0)
    centered = theta - mean[None, :]
    covariance = tf.einsum("n,ni,nj->ij", weights, centered, centered)
    offdiag = covariance - tf.linalg.diag(tf.linalg.diag_part(covariance))
    return {"theta_mean": mean, "theta_mean_0": mean[0], "covariance": covariance, "covariance_offdiag_max_abs": tf.reduce_max(tf.abs(offdiag)), "negative_mode_fraction": tf.reduce_sum(weights * tf.cast(theta[:, MODE_AXIS] < 0.0, tf.float64)), "root_count": tf.size(tf.unique(roots).y), "coordinate_min": tf.reduce_min(theta, axis=0), "coordinate_max": tf.reduce_max(theta, axis=0), "weighted_ess_fraction": tf.math.reciprocal(tf.reduce_sum(tf.square(weights))) / tf.cast(PARTICLES, tf.float64)}


def _mutate(theta: tf.Tensor, values: Mapping[str, tf.Tensor], beta: float, target: Any, chart: Mapping[str, tf.Tensor], seed: tuple[int, int], enabled: bool) -> tuple[Mapping[str, tf.Tensor], Mapping[str, Any]]:
    if not enabled:
        return values, {"steps": 0, "accepted_count": 0, "invalid_candidate_count": 0, "move_fraction": 0.0, "acceptance_rate": 0.0, "mean_displacement": 0.0, "log_alpha_min": 0.0, "log_alpha_max": 0.0}
    current_theta = theta
    current = values
    accepted_total = tf.constant(0, tf.int32)
    invalid_total = tf.constant(0, tf.int32)
    displacement_total = tf.constant(0.0, tf.float64)
    log_alpha_values: list[tf.Tensor] = []
    for step in range(MH_STEPS):
        split = tf.random.experimental.stateless_split(tf.constant((seed[0], seed[1] + step), tf.int32), 2)
        noise = tf.random.stateless_normal((PARTICLES, 4), seed=split[0], dtype=tf.float64)
        uniforms = tf.random.stateless_uniform((PARTICLES,), seed=split[1], minval=1.0e-12, maxval=1.0, dtype=tf.float64)
        candidate_theta = current_theta + tf.constant(MH_SIGMA, tf.float64) * noise
        candidate = _evaluate(candidate_theta, target, chart)
        current_bridge = (1.0 - beta) * current["proposal"] + beta * current["target"]
        candidate_bridge = (1.0 - beta) * candidate["proposal"] + beta * candidate["target"]
        candidate_bridge = tf.where(candidate["valid"], candidate_bridge, tf.fill((PARTICLES,), tf.constant(float("-inf"), tf.float64)))
        next_theta, accepted, log_alpha = _mh_accept(current_theta, candidate_theta, current_bridge, candidate_bridge, candidate["valid"], uniforms)
        displacement = tf.where(
            accepted[:, None], candidate_theta - current_theta, tf.zeros_like(candidate_theta)
        )
        current = {key: tf.where(accepted, candidate[key], current[key]) if key != "theta" else next_theta for key in ("theta", "target", "proposal", "valid", "status_code")}
        current_theta = next_theta
        accepted_total += tf.reduce_sum(tf.cast(accepted, tf.int32))
        invalid_total += tf.reduce_sum(tf.cast(tf.logical_not(candidate["valid"]), tf.int32))
        displacement_total += tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.square(displacement), axis=1)))
        log_alpha_values.append(tf.boolean_mask(log_alpha, candidate["valid"]))
    accepted_float = tf.cast(accepted_total, tf.float64)
    log_values = tf.concat(log_alpha_values, axis=0)
    log_values = tf.cond(
        tf.size(log_values) > 0,
        lambda: log_values,
        lambda: tf.zeros((1,), tf.float64),
    )
    return current, {"steps": MH_STEPS, "accepted_count": accepted_total, "invalid_candidate_count": invalid_total, "move_fraction": accepted_float / tf.cast(PARTICLES * MH_STEPS, tf.float64), "acceptance_rate": accepted_float / tf.cast(PARTICLES * MH_STEPS, tf.float64), "mean_displacement": displacement_total / tf.cast(PARTICLES * MH_STEPS, tf.float64), "log_alpha_min": tf.reduce_min(log_values), "log_alpha_max": tf.reduce_max(log_values)}


def _run_arm(initial: Mapping[str, tf.Tensor], arm: str, target: Any, chart: Mapping[str, tf.Tensor], replicate_seed: tuple[int, int], output: Path) -> Mapping[str, Any]:
    theta = tf.identity(initial["theta"])
    values = {key: tf.identity(value) for key, value in initial["values"].items()}
    roots = tf.identity(initial["roots"])
    log_weights = tf.zeros((PARTICLES,), tf.float64)
    stage_rows: list[Mapping[str, Any]] = []
    all_valid = True
    for stage_index, (left, right) in enumerate(zip(SCHEDULE[:-1], SCHEDULE[1:])):
        delta = tf.constant(right - left, tf.float64)
        ratio = values["target"] - values["proposal"]
        log_weights = log_weights + delta * ratio
        diagnostics = normalized_weight_diagnostics(log_weights)
        terminal = stage_index == len(SCHEDULE) - 2
        if not terminal:
            parents = systematic_resample_indices(diagnostics["normalized_log_weights"], seed=(replicate_seed[0], replicate_seed[1] + RESAMPLING_SEED_OFFSET + stage_index))
            theta = tf.gather(theta, parents)
            values = {key: tf.gather(value, parents) for key, value in values.items()}
            roots = tf.gather(roots, parents)
            log_weights = tf.zeros((PARTICLES,), tf.float64)
            values, mutation = _mutate(theta, values, right, target, chart, (replicate_seed[0], replicate_seed[1] + MH_SEED_OFFSET + stage_index * 100), arm == "mh")
        else:
            mutation = {"steps": 0, "accepted_count": 0, "invalid_candidate_count": 0, "move_fraction": 0.0, "acceptance_rate": 0.0, "mean_displacement": 0.0, "log_alpha_min": 0.0, "log_alpha_max": 0.0}
        theta = values["theta"]
        valid_now = bool(tf.reduce_all(values["valid"]).numpy())
        all_valid = all_valid and valid_now
        stage_rows.append({"stage_index": stage_index, "previous_beta": left, "beta": right, "pre_resampling_ess_fraction": diagnostics["effective_sample_size_fraction"], "pre_resampling_maximum_weight": diagnostics["maximum_normalized_weight"], "resampled": not terminal, "unique_root_count_after_resampling": tf.size(tf.unique(roots).y), "mutation": mutation, "all_current_status_valid": valid_now})
    weights = normalized_weight_diagnostics(log_weights)["normalized_weights"]
    summary = _summary(theta, weights, roots)
    tensors = {"final_theta": _write_tensor(output / f"{arm}-final-theta.tftensor", theta), "final_roots": _write_tensor(output / f"{arm}-final-roots.tftensor", roots), "final_weights": _write_tensor(output / f"{arm}-final-weights.tftensor", weights)}
    # Terminal non-resampling is a protocol fact, not a Boolean pass gate.  The
    # previous harness inserted ``False`` into this gate map and then required
    # every value to be true, rejecting otherwise-valid arms by construction.
    gates = {"final_shape_N_by_4": theta.shape == (PARTICLES, 4), "all_status_valid": all_valid, "finite_theta": bool(tf.reduce_all(tf.math.is_finite(theta)).numpy()), "finite_weights": bool(tf.reduce_all(tf.math.is_finite(weights)).numpy()), "finite_summary": bool(tf.reduce_all(tf.math.is_finite(summary["theta_mean"])).numpy())}
    return {"status": "PASS_V2_9_MUTATION_ARM" if all(gates.values()) else "PHASE47_MUTATION_ARM_FAIL", "arm": arm, "replicate_seed": list(replicate_seed), "sigma": MH_SIGMA, "mh_steps": MH_STEPS, "gates": gates, "stages": stage_rows, "final_summary": summary, "final_tensors": tensors, "nonclaims": ["Finite mutation clouds are not posterior or IID draws.", "Acceptance and support differences are descriptive only.", "No HMC, whitening, exhaustive mode discovery, canonical LEDH, superiority, or default claim."]}


def _pilot(path: Path) -> tuple[Path, Mapping[str, Any], Mapping[str, Any]]:
    pilot_path, payload = _load_json(path, "pilot.json")
    if payload.get("schema") != EXPECTED_PILOT_SCHEMA or payload.get("status") != "PASS_THETA_MEASURE_PILOT" or payload.get("measure") != EXPECTED_MEASURE:
        raise Phase47Error(f"pilot contract failed: {pilot_path}")
    m0 = payload.get("arms", {}).get("M0")
    c0 = payload.get("arms", {}).get("C0")
    if not isinstance(m0, Mapping) or not isinstance(c0, Mapping) or m0.get("schema") != EXPECTED_ARM_SCHEMA or c0.get("schema") != EXPECTED_ARM_SCHEMA:
        raise Phase47Error(f"pilot arm schema failed: {pilot_path}")
    if m0.get("status") != "PASS_THETA_MEASURE_PILOT" or c0.get("status") != "PASS_THETA_MEASURE_PILOT":
        raise Phase47Error(f"pilot arms are not passing: {pilot_path}")
    if m0.get("target_signature") != EXPECTED_TARGET or c0.get("target_signature") != EXPECTED_TARGET:
        raise Phase47Error(f"pilot target mismatch: {pilot_path}")
    if m0.get("configuration", {}).get("protocol_hash") != EXPECTED_M0 or c0.get("configuration", {}).get("protocol_hash") != EXPECTED_C0:
        raise Phase47Error(f"pilot protocol mismatch: {pilot_path}")
    if int(m0.get("configuration", {}).get("particles", -1)) != PARTICLES or int(payload.get("calibration", {}).get("particle_count", -1)) != CALIBRATION_PARTICLES:
        raise Phase47Error(f"pilot count mismatch: {pilot_path}")
    seed = tuple(int(value) for value in m0["configuration"]["seed"])
    return pilot_path, payload, {"m0": m0, "c0": c0, "seed": seed}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-root-1", required=True, type=Path)
    parser.add_argument("--pilot-root-2", required=True, type=Path)
    parser.add_argument("--pilot-root-3", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    roots = (args.pilot_root_1, args.pilot_root_2, args.pilot_root_3)
    all_paths = roots + (args.output_root,)
    if any(path.is_absolute() or ".." in path.parts for path in all_paths):
        raise Phase47Error("all paths must be repository-relative")
    output = ROOT / args.output_root
    if output.exists():
        raise Phase47Error(f"refusing to overwrite output root: {output}")
    started = time.perf_counter()
    pilot_records = [_pilot(path) for path in roots]
    pilot_paths = [record[0] for record in pilot_records]
    if len({_sha(path) for path in pilot_paths}) != 3:
        raise Phase47Error("pilot receipts are not distinct")
    seeds = [record[2]["seed"] for record in pilot_records]
    if len(set(seeds)) != 3:
        raise Phase47Error("pilot M0 seeds are not distinct")
    chart = _load_geometry()
    target = batch_native_complexity_posterior_target(20, jit_compile=True, principal_sqrt_backend="tensorflow_eigh")
    replicates: list[Mapping[str, Any]] = []
    for index, ((pilot_path, pilot, arm_info), root_seed) in enumerate(zip(pilot_records, seeds)):
        replicate_root = output / f"replicate-{index + 1:02d}"
        replicate_root.mkdir(parents=True)
        initial_theta, component = _sample_theta(root_seed, chart)
        initial_values = _evaluate(initial_theta, target, chart)
        if not bool(tf.reduce_all(initial_values["valid"]).numpy()):
            raise Phase47Error(f"initial q=20 proposal contains invalid rows in replicate {index + 1}")
        initial = {"theta": initial_theta, "values": initial_values, "roots": tf.range(PARTICLES, dtype=tf.int32)}
        initial_tensors = {"theta": _write_tensor(replicate_root / "initial-theta.tftensor", initial_theta), "proposal": _write_tensor(replicate_root / "initial-proposal-log-theta.tftensor", initial_values["proposal"]), "target": _write_tensor(replicate_root / "initial-target-log-theta.tftensor", initial_values["target"]), "roots": _write_tensor(replicate_root / "initial-roots.tftensor", initial["roots"]), "components": _write_tensor(replicate_root / "initial-proposal-components.tftensor", component)}
        initial_hash = hashlib.sha256((replicate_root / "initial-theta.tftensor").read_bytes()).hexdigest()
        identity = _run_arm(initial, "identity", target, chart, (root_seed[0], root_seed[1]), replicate_root / "identity")
        mh = _run_arm(initial, "mh", target, chart, (root_seed[0], root_seed[1]), replicate_root / "mh")
        paired = {"initial_tensor_hash": initial_hash, "identity_initial_tensor_hash": initial_hash, "mh_initial_tensor_hash": initial_hash, "resampling_seed_offset": RESAMPLING_SEED_OFFSET, "same_resampling_seeds": True, "same_initial_cloud": True}
        replicates.append({"replicate": index + 1, "pilot_root": roots[index], "pilot_sha256": _sha(pilot_path), "pilot_m0_seed": list(root_seed), "initial_tensors": initial_tensors, "paired": paired, "identity": identity, "mh": mh})
    hard_pass = all(rep["identity"]["status"] == "PASS_V2_9_MUTATION_ARM" and rep["mh"]["status"] == "PASS_V2_9_MUTATION_ARM" and rep["paired"]["same_initial_cloud"] and rep["paired"]["same_resampling_seeds"] for rep in replicates)
    result = {"schema": "bayesfilter.ssl_lstm.q20.corrected_theta_mutation_boundary.v1", "status": "PASS_V2_9_MUTATION_BOUNDARY" if hard_pass else "PHASE47_MUTATION_BOUNDARY_FAIL", "plan_version": EXPECTED_VERSION, "role": "paired_identity_vs_symmetric_theta_mh_finite_support_diagnostic", "measure": EXPECTED_MEASURE, "target_signature": target.target_signature(), "target_adapter_signature": target.adapter_signature(), "schedule": list(SCHEDULE), "particles": PARTICLES, "calibration_particles": CALIBRATION_PARTICLES, "defensive_epsilon": DEFENSIVE_EPSILON, "safe_std": SAFE_STD, "mode_axis": MODE_AXIS, "mh_sigma": MH_SIGMA, "mh_steps": MH_STEPS, "terminal_resampling": False, "fixture_required": True, "pilot_receipts_distinct": True, "replicates": replicates, "fresh_rows_used_for_training": False, "fresh_rows_used_for_selection": False, "hmc_launched": False, "device": {"gpu_memory_policy": GPU_POLICY, "physical_devices": [device.name for device in PHYSICAL_GPUS], "logical_devices": [device.name for device in LOGICAL_GPUS], "tf32_enabled": bool(tf.config.experimental.tensor_float_32_execution_enabled()), "jit_compile_target": True, "jit_compile_mutation": True}, "run_manifest": {"program": PLAN.as_posix(), "runner": RUNNER.as_posix(), "command": " ".join(sys.argv), "git_commit": subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=ROOT, text=True).strip(), "git_dirty": bool(subprocess.check_output(("git", "status", "--porcelain"), cwd=ROOT, text=True).strip()), "python": sys.executable, "python_version": platform.python_version(), "tensorflow": tf.__version__, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "tf_force_gpu_allow_growth": os.environ["TF_FORCE_GPU_ALLOW_GROWTH"], "gpu_memory_growth_verified": True, "jit_compile": True, "wall_seconds": time.perf_counter() - started, "source_sha256": {"plan": _sha(PLAN), "runner": _sha(RUNNER), "geometry": _sha(GEOMETRY), "target_module": _sha(TARGET_MODULE), "smc_module": _sha(SMC_MODULE), "importance_module": _sha(IMPORTANCE_MODULE), **{f"pilot_{index + 1}": _sha(path) for index, path in enumerate(pilot_paths)}}}, "nonclaims": ["Identity and MH clouds are finite diagnostics, not IID or posterior proofs.", "Acceptance, ESS, mode mass, root count, and support differences are descriptive only.", "No HMC, convergence, exhaustive mode discovery, canonical LEDH, whitening, superiority, or default claim."]}
    _write_json(output / "result.json", result)
    (output / "result.md").write_text("# v2.9 Paired Identity/MH Mutation Boundary\n\nStatus: `" + result["status"] + "`\n\nFinite support diagnostic; no whitening or posterior claim.\n", encoding="ascii")
    print(json.dumps({"status": result["status"], "output_root": args.output_root.as_posix()}, sort_keys=True))
    return 0 if hard_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
