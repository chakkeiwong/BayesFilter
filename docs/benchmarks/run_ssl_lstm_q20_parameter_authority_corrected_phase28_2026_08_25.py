"""Run a fresh paired C0/M0 pilot in the corrected theta measure.

This diagnostic deliberately keeps proposal and target log densities in
theta in R^4.  It uses the old geometry artifact only as a hash-bound proposal
calibration warm start; no old particle is loaded or reused.
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
    raise RuntimeError("Phase 28 requires CUDA_VISIBLE_DEVICES=-1")
if os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH") != "true":
    raise RuntimeError("Phase 28 requires TF_FORCE_GPU_ALLOW_GROWTH=true")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf

tf.config.set_visible_devices([], "GPU")
if tf.config.list_physical_devices("GPU"):
    raise RuntimeError("Phase 28 found a visible GPU in the reference lane")

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
from bayesfilter.testing.particle_authority_contracts_tf import canonical_protocol_hash


RUNNER = ROOT / "docs/benchmarks/run_ssl_lstm_q20_parameter_authority_corrected_phase28_2026_08_25.py"
PLAN = ROOT / "docs/plans/bayesfilter-ssl-lstm-q20-parameter-authority-corrected-continuation-2026-08-25.md"
GEOMETRY = ROOT / "docs/plans/artifacts/ssl-lstm-q20-seed-b-neutra-mode-failure-root-cause-2026-08-10/r1/geometry.json"
TARGET_MODULE = ROOT / "bayesfilter/nonlinear/ssl_lstm_complexity_batched_target_tf.py"
SMC_MODULE = ROOT / "bayesfilter/testing/annealed_smc_tf.py"
IMPORTANCE_MODULE = ROOT / "bayesfilter/testing/importance_sampling_tf.py"

SCHEDULE = (0.0, 0.20, 0.40, 0.60, 0.80, 1.0)
DEFAULT_PARTICLES = 64
DEFAULT_CALIBRATION = 16
DEFAULT_SEED = (20260825, 2801)
DEFENSIVE_EPSILON = 0.20
SAFE_STD = 2.0
MODE_AXIS = 2


class Phase28Error(RuntimeError):
    """Raised when the pilot cannot preserve a versioned measure receipt."""


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
        raise Phase28Error(f"refusing to overwrite artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(_safe(payload), sort_keys=True, indent=2, allow_nan=False) + "\n").encode("ascii")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(encoded)
    temporary.replace(path)


def _write_tensor(path: Path, value: Any) -> Mapping[str, Any]:
    if path.exists():
        raise Phase28Error(f"refusing to overwrite tensor artifact: {path}")
    tensor = tf.convert_to_tensor(value)
    encoded = bytes(tf.io.serialize_tensor(tensor).numpy())
    path.write_bytes(encoded)
    return {
        "path": path.as_posix(),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "bytes": len(encoded),
        "dtype": tensor.dtype.name,
        "shape": list(tensor.shape),
    }


def _load_geometry() -> Mapping[str, tf.Tensor]:
    payload = json.loads(GEOMETRY.read_text(encoding="utf-8"))
    if payload.get("status") != "GEOMETRY_DIAGNOSTIC_COMPLETED":
        raise Phase28Error("geometry calibration artifact is incomplete")
    labels = ("minus", "plus")
    means = tf.constant(
        [payload["representatives"][label]["position"] for label in labels],
        tf.float64,
    )
    precisions = tf.constant(
        [payload["source_curvature"][label]["records"][-1]["precision"] for label in labels],
        tf.float64,
    )
    covariances = tf.linalg.inv(precisions)
    tf.debugging.assert_all_finite(covariances, "geometry covariance")
    tf.debugging.assert_positive(tf.linalg.eigvalsh(covariances), "geometry covariance eigenvalues")
    center = tf.reduce_mean(means, axis=0)
    return {
        "means": means,
        "precisions": precisions,
        "covariances": covariances,
        "probabilities": tf.constant((0.5, 0.5), tf.float64),
        "center": center,
    }


def _safe_log_prob(theta: tf.Tensor, center: tf.Tensor) -> tf.Tensor:
    scale = tf.constant(SAFE_STD, tf.float64)
    centered = (theta - center[tf.newaxis, :]) / scale
    return -0.5 * tf.reduce_sum(tf.square(centered), axis=1) - 4.0 * (
        tf.math.log(scale) + 0.5 * tf.constant(1.8378770664093453, tf.float64)
    )


def _proposal_log_theta(theta: tf.Tensor, chart: Mapping[str, tf.Tensor], epsilon: float) -> tf.Tensor:
    local = gaussian_mixture_log_prob(
        theta, chart["probabilities"], chart["means"], chart["covariances"]
    )
    if float(epsilon) <= 0.0:
        return local
    safe = _safe_log_prob(theta, chart["center"])
    eps = tf.constant(float(epsilon), tf.float64)
    return tf.reduce_logsumexp(
        tf.stack(
            (tf.math.log1p(-eps) + local, tf.math.log(eps) + safe), axis=1
        ),
        axis=1,
    )


def _sample_theta(
    count: int, chart: Mapping[str, tf.Tensor], epsilon: float, seed: tuple[int, int]
) -> tuple[tf.Tensor, tf.Tensor]:
    split = tf.random.experimental.stateless_split(tf.constant(seed, tf.int32), 4)
    local, labels = sample_gaussian_mixture(
        count,
        chart["probabilities"],
        chart["means"],
        chart["covariances"],
        seed=tuple(int(value) for value in split[0].numpy()),
    )
    if float(epsilon) <= 0.0:
        return local, labels
    safe_noise = tf.random.stateless_normal((count, 4), seed=split[1], dtype=tf.float64)
    safe = chart["center"][tf.newaxis, :] + SAFE_STD * safe_noise
    choose_safe = tf.random.stateless_uniform((count,), seed=split[2], dtype=tf.float64) < float(epsilon)
    theta = tf.where(choose_safe[:, tf.newaxis], safe, local)
    component = tf.where(choose_safe, tf.fill((count,), 2), labels)
    return theta, component


def _evaluate(theta: tf.Tensor, target: Any, chart: Mapping[str, tf.Tensor], epsilon: float) -> Mapping[str, tf.Tensor]:
    if theta.shape.rank != 2 or theta.shape[1] != 4:
        raise Phase28Error(f"particle rows must be [N,4], got {theta.shape}")
    value, score, status = target.neutra_batch_log_prob_and_grad_status(theta)
    valid = tf.logical_and(
        tf.equal(tf.convert_to_tensor(status["status_code"], tf.int32), 0),
        tf.cast(status["valid_pre_regularized_score"], tf.bool),
    )
    proposal = _proposal_log_theta(theta, chart, epsilon)
    tf.debugging.assert_all_finite(value, "theta target log density")
    tf.debugging.assert_all_finite(proposal, "theta proposal log density")
    tf.debugging.assert_all_finite(score, "theta target score")
    return {
        "theta": theta,
        "target_log_theta": tf.convert_to_tensor(value, tf.float64),
        "proposal_log_theta": proposal,
        "score": score,
        "status_code": tf.convert_to_tensor(status["status_code"], tf.int32),
        "valid": valid,
        "sign": theta[:, MODE_AXIS] < 0.0,
    }


def _run_arm(
    arm: str,
    count: int,
    seed: tuple[int, int],
    chart: Mapping[str, tf.Tensor],
    target: Any,
    output: Path,
) -> Mapping[str, Any]:
    epsilon = DEFENSIVE_EPSILON if arm == "M0" else 0.0
    theta, component = _sample_theta(count, chart, epsilon, seed)
    values = _evaluate(theta, target, chart, epsilon)
    if not bool(tf.reduce_all(values["valid"]).numpy()):
        raise Phase28Error(f"{arm} initial theta proposal contains invalid rows")
    roots = tf.range(count, dtype=tf.int32)
    log_weights = tf.zeros((count,), tf.float64)
    log_mass = tf.constant(0.0, tf.float64)
    stages: list[Mapping[str, Any]] = []
    resampling_count = 0
    for stage_index, (left, right) in enumerate(zip(SCHEDULE[:-1], SCHEDULE[1:])):
        delta = tf.constant(right - left, tf.float64)
        ratio = values["target_log_theta"] - values["proposal_log_theta"]
        log_weights = log_weights + delta * ratio
        diagnostics = normalized_weight_diagnostics(log_weights)
        log_mass = log_mass + tf.reduce_logsumexp(log_weights) - tf.math.log(tf.cast(count, tf.float64))
        terminal = stage_index == len(SCHEDULE) - 2
        stage: dict[str, Any] = {
            "stage_index": stage_index,
            "previous_beta": left,
            "beta": right,
            "delta_beta": right - left,
            "pre_resampling_ess_fraction": diagnostics["effective_sample_size_fraction"],
            "pre_resampling_maximum_weight": diagnostics["maximum_normalized_weight"],
            "log_unnormalized_mass_estimate": log_mass,
            "unique_root_count": tf.size(tf.unique(roots).y),
            "resampled": not terminal,
        }
        if not terminal:
            parents = systematic_resample_indices(
                diagnostics["normalized_log_weights"],
                seed=(seed[0], seed[1] + 1000 + stage_index),
            )
            values = {key: tf.gather(value, parents) for key, value in values.items()}
            component = tf.gather(component, parents)
            roots = tf.gather(roots, parents)
            log_weights = tf.zeros((count,), tf.float64)
            stage["unique_parent_count"] = tf.size(tf.unique(parents).y)
            resampling_count += 1
        _write_json(output / f"{arm.lower()}-stage-{stage_index:02d}.json", stage)
        stages.append(stage)

    final = normalized_weight_diagnostics(log_weights)
    weights = final["normalized_weights"]
    finite_terms = tf.reduce_all(
        tf.math.is_finite(values["target_log_theta"]) & tf.math.is_finite(values["proposal_log_theta"])
    )
    gates = {
        "theta_shape_N_by_4": values["theta"].shape == (count, 4),
        "target_status_valid": bool(tf.reduce_all(values["valid"]).numpy()),
        "density_terms_finite": bool(finite_terms.numpy()),
        "beta_one_reached": SCHEDULE[-1] == 1.0,
        "protocol_hash_present": True,
        "finite_log_mass": bool(tf.math.is_finite(log_mass).numpy()),
        "epsilon_in_declared_range": 0.0 <= epsilon <= 1.0,
    }
    protocol = {
        "schema": "bayesfilter.q20.corrected_theta_authority.protocol.v2",
        "measure": "theta_R4",
        "arm": arm,
        "beta_schedule": list(SCHEDULE),
        "resampling": "systematic_at_each_nonterminal_fixed_stage",
        "mutation": "identity_invariant_reference",
        "defensive_epsilon": epsilon,
        "safe_std": SAFE_STD,
        "mode_axis": MODE_AXIS,
        "target_signature": target.target_signature(),
        "adapter_signature": target.adapter_signature(),
        "geometry_role": "calibration_warm_start_only",
        "geometry_sha256": _sha(GEOMETRY),
        "density_jacobian": "none_in_theta_log_terms; chart_jacobian_not_used",
    }
    protocol_hash = canonical_protocol_hash(protocol)
    payload = {
        "schema": "bayesfilter.ssl_lstm.q20.corrected_theta_authority.arm.v1",
        "status": "PASS_THETA_MEASURE_PILOT" if all(gates.values()) else "PHASE28_ARM_FAIL_REPAIR",
        "arm": arm,
        "role": "fresh_theta_c0_descriptive_comparator" if arm == "C0" else "fresh_theta_m0_candidate_not_smu_u_admitted",
        "configuration": {
            "particles": count,
            "seed": list(seed),
            "schedule": list(SCHEDULE),
            "defensive_epsilon": epsilon,
            "protocol_hash": protocol_hash,
        },
        "gates": gates,
        "diagnostics": {
            "terminal_ess_fraction": final["effective_sample_size_fraction"],
            "terminal_maximum_weight": final["maximum_normalized_weight"],
            "log_unnormalized_mass_estimate": log_mass,
            "weighted_negative_mode_fraction": tf.reduce_sum(weights * tf.cast(values["sign"], tf.float64)),
            "initial_negative_mode_fraction": tf.reduce_mean(tf.cast(theta[:, MODE_AXIS] < 0.0, tf.float64)),
            "terminal_negative_root_count": tf.size(tf.unique(tf.boolean_mask(roots, values["sign"])).y),
            "terminal_positive_root_count": tf.size(tf.unique(tf.boolean_mask(roots, tf.logical_not(values["sign"]))).y),
            "resampling_count": resampling_count,
            "stages": stages,
        },
        "protocol": protocol,
        "receipts": {
            "final_theta": _write_tensor(output / f"{arm.lower()}-final-theta.tftensor", values["theta"]),
            "final_target_log_theta": _write_tensor(output / f"{arm.lower()}-final-target-log-theta.tftensor", values["target_log_theta"]),
            "final_proposal_log_theta": _write_tensor(output / f"{arm.lower()}-final-proposal-log-theta.tftensor", values["proposal_log_theta"]),
            "final_normalized_weights": _write_tensor(output / f"{arm.lower()}-final-normalized-weights.tftensor", weights),
            "final_roots": _write_tensor(output / f"{arm.lower()}-final-roots.tftensor", roots),
            "proposal_components": _write_tensor(output / f"{arm.lower()}-proposal-components.tftensor", component),
        },
        "target_signature": target.target_signature(),
        "adapter_signature": target.adapter_signature(),
        "nonclaims": [
            "M0 is a candidate role; this finite pilot is not an SMC-U proof.",
            "Identity mutation is invariant but has no finite-run mode-mixing guarantee.",
            "No posterior correctness, IID whitening, exhaustive mode discovery, LEDH, HMC, or default claim.",
        ],
    }
    _write_json(output / f"{arm.lower()}.json", payload)
    return payload


def _markdown(receipt: Mapping[str, Any]) -> str:
    lines = [
        "# Corrected q=20 Fresh Theta Pilot",
        "",
        f"Status: `{receipt['status']}`",
        "",
        "Both arms keep target and proposal log densities in theta in R^4. The geometry artifact is a calibration warm start only.",
        "",
        "| Arm | Status | Role |",
        "|---|---|---|",
    ]
    for arm, value in receipt["arms"].items():
        lines.append(f"| {arm} | `{value['status']}` | {value['role']} |")
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "Finite measure and status gates can nominate Phase 29. They do not admit an SMC-U authority or a posterior claim.",
            "",
            "## Nonclaims",
            "",
            "- ESS, mass, mode occupancy, and root counts are descriptive diagnostics.",
            "- No IID, whitening, mode-discovery, LEDH, NeuTra, HMC, or default promotion claim.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--particles", type=int, default=DEFAULT_PARTICLES)
    parser.add_argument("--calibration-particles", type=int, default=DEFAULT_CALIBRATION)
    parser.add_argument("--seed", nargs=2, type=int, default=DEFAULT_SEED)
    parser.add_argument("--arms", choices=("m0", "c0", "both"), default="both")
    args = parser.parse_args()
    if args.output_root.is_absolute() or ".." in args.output_root.parts:
        raise Phase28Error("output root must be repository-relative")
    if args.output_root.exists():
        raise Phase28Error(f"refusing to overwrite output root: {args.output_root}")
    if int(args.particles) < 8 or int(args.calibration_particles) < 8:
        raise Phase28Error("particle counts must be at least eight")
    output = ROOT / args.output_root
    output.mkdir(parents=True)
    started = time.perf_counter()
    chart = _load_geometry()
    target = batch_native_complexity_posterior_target(
        20, jit_compile=True, principal_sqrt_backend="tensorflow_eigh"
    )
    # Calibration is deliberately separate from the claim/candidate bank.
    calibration_theta, _ = _sample_theta(
        int(args.calibration_particles), chart, DEFENSIVE_EPSILON,
        (int(args.seed[0]), int(args.seed[1]) + 10),
    )
    calibration = _evaluate(calibration_theta, target, chart, DEFENSIVE_EPSILON)
    if not bool(tf.reduce_all(calibration["valid"]).numpy()):
        raise Phase28Error("calibration theta rows contain invalid target status")
    calibration_ratio = calibration["target_log_theta"] - calibration["proposal_log_theta"]
    calibration_payload = {
        "schema": "bayesfilter.ssl_lstm.q20.corrected_theta_pilot.calibration.v1",
        "status": "CALIBRATION_COMPLETED",
        "measure": "theta_R4",
        "particle_count": int(args.calibration_particles),
        "seed": [int(args.seed[0]), int(args.seed[1]) + 10],
        "ratio_min": tf.reduce_min(calibration_ratio),
        "ratio_max": tf.reduce_max(calibration_ratio),
        "ratio_finite": tf.reduce_all(tf.math.is_finite(calibration_ratio)),
        "target_signature": target.target_signature(),
        "geometry_sha256": _sha(GEOMETRY),
        "schedule": list(SCHEDULE),
        "schedule_selection": "fixed_hypothesis_not_adaptive",
    }
    _write_json(output / "calibration.json", calibration_payload)
    requested = ("M0", "C0") if args.arms == "both" else (("M0",) if args.arms == "m0" else ("C0",))
    arms: dict[str, Any] = {}
    for index, arm in enumerate(requested):
        arms[arm] = _run_arm(
            arm,
            int(args.particles),
            (int(args.seed[0]), int(args.seed[1]) + 100 + index),
            chart,
            target,
            output,
        )
    hard_pass = all(value["status"] == "PASS_THETA_MEASURE_PILOT" for value in arms.values())
    receipt = {
        "schema": "bayesfilter.ssl_lstm.q20.corrected_theta_authority_pilot.v1",
        "status": "PASS_THETA_MEASURE_PILOT" if hard_pass else "PHASE28_FAIL_REPAIR",
        "measure": "theta_R4",
        "arms": arms,
        "calibration": calibration_payload,
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
            "physical_gpus": [device.name for device in tf.config.list_physical_devices("GPU")],
            "logical_gpus": [device.name for device in tf.config.list_logical_devices("GPU")],
            "jit_compile": True,
            "seeds": {"root": list(args.seed), "calibration": [int(args.seed[0]), int(args.seed[1]) + 10]},
            "wall_seconds": time.perf_counter() - started,
            "source_sha256": {
                "plan": _sha(PLAN),
                "runner": _sha(RUNNER),
                "geometry": _sha(GEOMETRY),
                "target_module": _sha(TARGET_MODULE),
                "smc_module": _sha(SMC_MODULE),
                "importance_module": _sha(IMPORTANCE_MODULE),
            },
        },
        "nonclaims": [
            "Fresh rows are not a finite-run mode-discovery guarantee.",
            "M0 is not admitted SMC-U authority; its proof and uncertainty gates remain open.",
            "No ETPF/GenUT/LEDH arm, NeuTra training, HMC, posterior correctness, IID whitening, or default change.",
        ],
    }
    _write_json(output / "pilot.json", receipt)
    (output / "result.md").write_text(_markdown(receipt), encoding="ascii")
    print(json.dumps({"status": receipt["status"], "output_root": args.output_root.as_posix()}, sort_keys=True))
    return 0 if hard_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
