"""Run the first fresh q=20 C0/M0 authority pilot.

The pilot is a CPU-hidden, batch-native TensorFlow reference lane.  It uses a
fixed, hash-bound tempering protocol and systematic resampling.  The default
mutation is the identity kernel: it is invariant for every target but cannot
mix modes.  This deliberately exposes the proposal/mutation boundary before a
more expensive mutation repair is attempted.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping


if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
    raise RuntimeError("q20 authority pilot requires CUDA_VISIBLE_DEVICES=-1")
if os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH") != "true":
    raise RuntimeError("q20 authority pilot requires TF_FORCE_GPU_ALLOW_GROWTH=true")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf

tf.config.set_visible_devices([], "GPU")
if tf.config.list_physical_devices("GPU"):
    raise RuntimeError("q20 authority pilot found a visible GPU in CPU lane")

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


GEOMETRY = ROOT / "docs/plans/artifacts/ssl-lstm-q20-seed-b-neutra-mode-failure-root-cause-2026-08-10/r1/geometry.json"
AIS_RUNNER = ROOT / "docs/benchmarks/run_ssl_lstm_q20_physical_ais_repair_2026_08_10.py"
PLAN = ROOT / "docs/plans/bayesfilter-ssl-lstm-q20-particle-authority-master-program-2026-08-25.md"
RUNNER = ROOT / "docs/benchmarks/run_ssl_lstm_q20_particle_authority_pilot_2026_08_25.py"

TARGET_ESS_FRACTION = 0.50
DEFENSIVE_EPSILON = 0.20
SAFE_STD = 2.0
CALIBRATION_PARTICLES = 16
DEFAULT_PARTICLES = 100
DEFAULT_SEED = (20260825, 401)
FIXED_MUTATION = "identity_invariant_reference"
RANDOM_WALK_MUTATION = "random_walk_metropolis_symmetric_reference"
DEFAULT_MUTATION_SCALE = 0.50
DEFAULT_MUTATION_STEPS = 0
MODE_AXIS = 2
SCHEDULE_CANDIDATES = (
    (0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.90, 1.0),
    (0.0, 0.10, 0.25, 0.40, 0.60, 0.75, 0.90, 1.0),
    (0.0, 0.20, 0.40, 0.60, 0.80, 1.0),
)


class PilotError(RuntimeError):
    """Raised when the pilot cannot preserve an auditable artifact."""


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, bytes):
        return value.hex()
    if hasattr(value, "numpy"):
        return _safe(value.numpy())
    if hasattr(value, "tolist"):
        return _safe(value.tolist())
    if hasattr(value, "item"):
        return _safe(value.item())
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise PilotError(f"refusing to overwrite artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(_safe(payload), sort_keys=True, indent=2, allow_nan=False) + "\n").encode("ascii")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(encoded)
    temporary.replace(path)


def _write_tensor(path: Path, value: Any) -> Mapping[str, Any]:
    if path.exists():
        raise PilotError(f"refusing to overwrite tensor artifact: {path}")
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


def _load_geometry_proposal() -> Mapping[str, Any]:
    payload = json.loads(GEOMETRY.read_text(encoding="utf-8"))
    if payload.get("status") != "GEOMETRY_DIAGNOSTIC_COMPLETED":
        raise PilotError("geometry source is incomplete")
    labels = ("plus", "minus")
    return {
        "means": [payload["representatives"][label]["position"] for label in labels],
        "precisions": [payload["source_curvature"][label]["records"][-1]["precision"] for label in labels],
    }


def _make_chart(proposal: Mapping[str, Any]) -> Mapping[str, tf.Tensor]:
    means = tf.constant(proposal["means"], tf.float64)
    precisions = tf.constant(proposal["precisions"], tf.float64)
    covariances = tf.linalg.inv(precisions)
    probabilities = tf.constant((0.5, 0.5), tf.float64)
    center = tf.reduce_mean(means, axis=0)
    displacement = means - center
    pooled = tf.reduce_mean(covariances, axis=0) + tf.einsum(
        "ni,nj->ij", displacement, displacement
    ) / 2.0
    eigenvalues, eigenvectors = tf.linalg.eigh(pooled)
    factor = tf.matmul(
        eigenvectors * tf.sqrt(eigenvalues)[tf.newaxis, :],
        eigenvectors,
        transpose_b=True,
    )
    return {
        "means": means,
        "precisions": precisions,
        "covariances": covariances,
        "probabilities": probabilities,
        "center": center,
        "factor": factor,
        "eigenvalues": eigenvalues,
        "log_jacobian": tf.reduce_sum(tf.math.log(eigenvalues)) / 2.0,
    }


def _safe_log_prob(theta: tf.Tensor, chart: Mapping[str, tf.Tensor]) -> tf.Tensor:
    centered = (theta - chart["center"]) / tf.constant(SAFE_STD, tf.float64)
    return -0.5 * tf.reduce_sum(tf.square(centered), axis=1) - 4.0 * (
        tf.math.log(tf.constant(SAFE_STD, tf.float64)) + 0.5 * tf.constant(1.8378770664093453, tf.float64)
    )


def _proposal_log_prob(theta: tf.Tensor, chart: Mapping[str, tf.Tensor], epsilon: float) -> tf.Tensor:
    local = gaussian_mixture_log_prob(
        theta,
        chart["probabilities"],
        chart["means"],
        chart["covariances"],
    )
    safe = _safe_log_prob(theta, chart)
    eps = tf.constant(max(float(epsilon), 1.0e-30), tf.float64)
    mixture = tf.reduce_logsumexp(
        tf.stack(
            [tf.math.log(1.0 - eps) + local, tf.math.log(eps) + safe],
            axis=1,
        ),
        axis=1,
    )
    return mixture + chart["log_jacobian"]


def _sample_proposal(
    count: int,
    chart: Mapping[str, tf.Tensor],
    epsilon: float,
    seed: tuple[int, int],
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    split = tf.random.experimental.stateless_split(tf.constant(seed, tf.int32), 4)
    local, local_components = sample_gaussian_mixture(
        count,
        chart["probabilities"],
        chart["means"],
        chart["covariances"],
        seed=tuple(int(v) for v in split[0].numpy()),
    )
    safe_noise = tf.random.stateless_normal((count, 4), seed=split[1], dtype=tf.float64)
    safe = chart["center"][tf.newaxis, :] + SAFE_STD * safe_noise
    choose_safe = tf.random.stateless_uniform((count,), seed=split[2], dtype=tf.float64) < float(epsilon)
    theta = tf.where(choose_safe[:, tf.newaxis], safe, local)
    component = tf.where(choose_safe, tf.ones((count,), tf.int32), tf.zeros((count,), tf.int32))
    z = tf.transpose(tf.linalg.solve(chart["factor"], tf.transpose(theta - chart["center"])))
    del local_components
    return z, theta, component


def _evaluate(
    z: tf.Tensor,
    chart: Mapping[str, tf.Tensor],
    target: Any,
    epsilon: float,
) -> Mapping[str, tf.Tensor]:
    theta = chart["center"] + tf.matmul(z, chart["factor"], transpose_b=True)
    target_value, _score, status = target.neutra_batch_log_prob_and_grad_status(theta)
    valid = tf.logical_and(
        tf.convert_to_tensor(status["status_code"], tf.int32) == 0,
        tf.convert_to_tensor(status["valid_pre_regularized_score"], tf.bool),
    )
    target_log = tf.convert_to_tensor(target_value, tf.float64) + chart["log_jacobian"]
    proposal_log = _proposal_log_prob(theta, chart, epsilon)
    tf.debugging.assert_all_finite(target_log, "q20 target log density")
    tf.debugging.assert_all_finite(proposal_log, "q20 proposal log density")
    return {
        "z": z,
        "theta": theta,
        "target_log_prob": target_log,
        "proposal_log_prob": proposal_log,
        "valid": valid,
        "status_code": tf.convert_to_tensor(status["status_code"], tf.int32),
        "sign": theta[:, MODE_AXIS] < 0.0,
    }


def _ess_fraction(log_weights: tf.Tensor) -> float:
    return float(normalized_weight_diagnostics(log_weights)["effective_sample_size_fraction"].numpy())


def _select_fixed_schedule(calibration_ratio: tf.Tensor) -> tuple[tuple[float, ...], Mapping[str, Any]]:
    scores = []
    for candidate in SCHEDULE_CANDIDATES:
        minimum = 1.0
        for left, right in zip(candidate[:-1], candidate[1:]):
            minimum = min(minimum, _ess_fraction((right - left) * calibration_ratio))
        scores.append({"schedule": candidate, "minimum_calibration_ess_fraction": minimum})
    eligible = [item for item in scores if item["minimum_calibration_ess_fraction"] >= TARGET_ESS_FRACTION]
    selected = eligible[0] if eligible else scores[0]
    return tuple(float(x) for x in selected["schedule"]), {
        "candidates": scores,
        "selected": selected,
        "selection_rule": "first candidate with every calibration increment cESS >= target, otherwise densest candidate",
        "calibration_only": True,
    }


def _random_walk_mutate(
    values: Mapping[str, tf.Tensor],
    *,
    beta: float,
    chart: Mapping[str, tf.Tensor],
    target: Any,
    epsilon: float,
    scale: float,
    steps: int,
    seed: tuple[int, int],
) -> tuple[Mapping[str, tf.Tensor], Mapping[str, Any]]:
    """Apply a batched symmetric MH kernel to the tempered target in z-space."""
    if int(steps) <= 0:
        return values, {
            "steps": 0,
            "acceptance_rate": 0.0,
            "invalid_proposal_count": 0,
            "transition_log_density_residual": 0.0,
        }
    if not math.isfinite(float(scale)) or float(scale) <= 0.0:
        raise PilotError("mutation scale must be finite and positive")
    current = {str(key): tf.convert_to_tensor(value) for key, value in values.items()}
    accepted_total = tf.constant(0, tf.int32)
    invalid_total = tf.constant(0, tf.int32)
    root_seed = tf.constant(seed, tf.int32)
    for _step_index in range(int(steps)):
        split = tf.random.experimental.stateless_split(root_seed, 2)
        root_seed = split[0]
        noise = tf.random.stateless_normal(
            tf.shape(current["z"]), seed=split[1], dtype=tf.float64
        )
        candidate_z = current["z"] + tf.constant(float(scale), tf.float64) * noise
        candidate = _evaluate(candidate_z, chart, target, epsilon)
        current_log = (1.0 - float(beta)) * current["proposal_log_prob"] + float(beta) * current["target_log_prob"]
        candidate_log = (1.0 - float(beta)) * candidate["proposal_log_prob"] + float(beta) * candidate["target_log_prob"]
        log_alpha = tf.minimum(tf.zeros_like(current_log), candidate_log - current_log)
        uniforms = tf.random.stateless_uniform(
            tf.shape(current_log), seed=split[0], dtype=tf.float64
        )
        accepted = tf.logical_and(
            tf.cast(candidate["valid"], tf.bool),
            tf.math.log(tf.maximum(uniforms, tf.constant(1.0e-300, tf.float64))) < log_alpha,
        )
        accepted_total += tf.reduce_sum(tf.cast(accepted, tf.int32))
        invalid_total += tf.reduce_sum(
            tf.cast(tf.logical_not(tf.cast(candidate["valid"], tf.bool)), tf.int32)
        )
        current = {
            key: tf.where(
                accepted[:, tf.newaxis] if candidate[key].shape.rank == 2 else accepted,
                candidate[key],
                current[key],
            )
            for key in current
        }
    # Each MH proposal is one particle-level transition.  tf.size(z) is N*d,
    # so using it here would under-report acceptance by the state dimension.
    particle_count = tf.cast(tf.shape(current["z"])[0], tf.float64)
    proposal_count = particle_count * tf.cast(int(steps), tf.float64)
    return current, {
        "steps": int(steps),
        "accepted_count": accepted_total,
        "proposal_count": proposal_count,
        "acceptance_rate": tf.cast(accepted_total, tf.float64) / proposal_count,
        "invalid_proposal_count": invalid_total,
        "transition_log_density_residual": 0.0,
        "kernel": RANDOM_WALK_MUTATION,
        "scale": float(scale),
    }


def _run_arm(
    arm: str,
    count: int,
    seed: tuple[int, int],
    schedule: tuple[float, ...],
    chart: Mapping[str, tf.Tensor],
    target: Any,
    output_root: Path,
    mutation_kernel: str = FIXED_MUTATION,
    mutation_steps: int = DEFAULT_MUTATION_STEPS,
    mutation_scale: float = DEFAULT_MUTATION_SCALE,
) -> Mapping[str, Any]:
    epsilon = DEFENSIVE_EPSILON if arm == "M0" else 0.0
    z, theta, component = _sample_proposal(count, chart, epsilon, seed)
    values = _evaluate(z, chart, target, epsilon)
    if not bool(tf.reduce_all(values["valid"]).numpy()):
        raise PilotError(f"{arm} initial proposal contains invalid target rows")
    roots = tf.range(count, dtype=tf.int32)
    root_signs = tf.identity(values["sign"])
    initial_negative_fraction = tf.reduce_mean(tf.cast(root_signs, tf.float64))
    log_weights = tf.zeros((count,), tf.float64)
    log_normalizer = tf.constant(0.0, tf.float64)
    stages = []
    resampling_count = 0
    mutation_receipts = []
    for stage_index, (left, right) in enumerate(zip(schedule[:-1], schedule[1:])):
        delta = tf.constant(right - left, tf.float64)
        log_weights = log_weights + delta * (values["target_log_prob"] - values["proposal_log_prob"])
        diagnostics = normalized_weight_diagnostics(log_weights)
        log_normalizer = log_normalizer + tf.reduce_logsumexp(log_weights) - tf.math.log(tf.cast(count, tf.float64))
        terminal = stage_index == len(schedule) - 2
        stage = {
            "stage_index": stage_index,
            "previous_beta": left,
            "beta": right,
            "delta_beta": right - left,
            "terminal_pre_resampling": terminal,
            "pre_resampling_ess_fraction": diagnostics["effective_sample_size_fraction"],
            "pre_resampling_maximum_weight": diagnostics["maximum_normalized_weight"],
            "log_normalizer": log_normalizer,
            "unique_root_count": tf.size(tf.unique(roots).y),
        }
        if not terminal:
            parents = systematic_resample_indices(
                diagnostics["normalized_log_weights"],
                seed=(seed[0], seed[1] + 1000 + stage_index),
            )
            z = tf.gather(values["z"], parents)
            theta = tf.gather(values["theta"], parents)
            values = {key: tf.gather(value, parents) for key, value in values.items()}
            component = tf.gather(component, parents)
            roots = tf.gather(roots, parents)
            root_signs = tf.gather(root_signs, parents)
            log_weights = tf.zeros((count,), tf.float64)
            resampling_count += 1
            stage["resampled"] = True
            stage["unique_parent_count"] = tf.size(tf.unique(parents).y)
            if mutation_kernel == RANDOM_WALK_MUTATION:
                values, mutation = _random_walk_mutate(
                    values,
                    beta=right,
                    chart=chart,
                    target=target,
                    epsilon=epsilon,
                    scale=mutation_scale,
                    steps=mutation_steps,
                    seed=(seed[0], seed[1] + 5000 + stage_index),
                )
                mutation_receipts.append(mutation)
                stage["mutation"] = mutation
            elif mutation_kernel != FIXED_MUTATION:
                raise PilotError(f"unsupported mutation kernel: {mutation_kernel}")
        else:
            stage["resampled"] = False
        _write_json(output_root / f"{arm.lower()}-stage-{stage_index:02d}.json", stage)
        stages.append(stage)

    final_diagnostics = normalized_weight_diagnostics(log_weights)
    final_weights = final_diagnostics["normalized_weights"]
    weighted_negative = tf.reduce_sum(final_weights * tf.cast(values["sign"], tf.float64))
    all_valid = bool(tf.reduce_all(values["valid"]).numpy())
    finite_mass = bool(tf.math.is_finite(log_normalizer).numpy())
    protocol = {
        "schema": "bayesfilter.q20.authority.protocol.v1",
        "arm": arm,
        "beta_schedule": list(schedule),
        "resampling_trigger": "every_nonterminal_fixed_stage",
        "mutation_kernel": mutation_kernel,
        "mutation_steps": int(mutation_steps),
        "mutation_scale": float(mutation_scale),
        "mode_axis": MODE_AXIS,
        "mutation_transition": (
            "symmetric_gaussian_random_walk"
            if mutation_kernel == RANDOM_WALK_MUTATION
            else "identity"
        ),
        "defensive_epsilon": epsilon,
        "safe_std": SAFE_STD,
        "target_ess_fraction_calibration": TARGET_ESS_FRACTION,
        "target_signature": target.target_signature(),
        "adapter_signature": target.adapter_signature(),
    }
    protocol_hash = canonical_protocol_hash(protocol)
    receipts = {
        "final_z": _write_tensor(output_root / f"{arm.lower()}-final-z.tftensor", values["z"]),
        "final_theta": _write_tensor(output_root / f"{arm.lower()}-final-theta.tftensor", values["theta"]),
        "final_target_log_prob": _write_tensor(output_root / f"{arm.lower()}-final-target-log-prob.tftensor", values["target_log_prob"]),
        "final_proposal_log_prob": _write_tensor(output_root / f"{arm.lower()}-final-proposal-log-prob.tftensor", values["proposal_log_prob"]),
        "final_normalized_weights": _write_tensor(output_root / f"{arm.lower()}-final-normalized-weights.tftensor", final_weights),
        "final_roots": _write_tensor(output_root / f"{arm.lower()}-final-roots.tftensor", roots),
        "proposal_components": _write_tensor(output_root / f"{arm.lower()}-proposal-components.tftensor", component),
    }
    gates = {
        "all_rows_target_status_valid": all_valid,
        "all_density_terms_finite": bool(tf.reduce_all(tf.math.is_finite(values["target_log_prob"]) & tf.math.is_finite(values["proposal_log_prob"])).numpy()),
        "reached_beta_one": abs(schedule[-1] - 1.0) <= 1.0e-12,
        "protocol_hash_present": bool(protocol_hash),
        "finite_unnormalized_mass_estimate": finite_mass,
        "defensive_support_parameter_valid": 0.0 < epsilon <= 1.0 if arm == "M0" else True,
    }
    payload = {
        "schema": "bayesfilter.ssl_lstm.q20.authority_pilot.arm.v1",
        "status": "PASS_GATE" if all(gates.values()) else "CANDIDATE_FAIL_REPAIR",
        "arm": arm,
        "role": "fresh_q20_m0_candidate" if arm == "M0" else "fresh_q20_c0_descriptive_comparator",
        "configuration": {
            "particles": count,
            "seed": list(seed),
            "mutation_kernel": mutation_kernel,
            "mutation_steps": int(mutation_steps),
            "mutation_scale": float(mutation_scale),
            "defensive_epsilon": epsilon,
            "safe_std": SAFE_STD,
            "beta_schedule": list(schedule),
            "protocol_hash": protocol_hash,
        },
        "gates": gates,
        "diagnostics": {
            "log_unnormalized_mass_estimate": log_normalizer,
            "terminal_effective_sample_size_fraction": final_diagnostics["effective_sample_size_fraction"],
            "terminal_maximum_normalized_weight": final_diagnostics["maximum_normalized_weight"],
            "terminal_weighted_negative_mode_fraction": weighted_negative,
            "initial_negative_mode_fraction": initial_negative_fraction,
            "terminal_negative_root_count": tf.size(tf.unique(tf.boolean_mask(roots, root_signs)).y),
            "terminal_positive_root_count": tf.size(tf.unique(tf.boolean_mask(roots, tf.logical_not(root_signs))).y),
            "mode_axis": MODE_AXIS,
            "resampling_count": resampling_count,
            "mutation_receipts": mutation_receipts,
        },
        "protocol": protocol,
        "receipts": receipts,
        "target_signature": target.target_signature(),
        "adapter_signature": target.adapter_signature(),
        "nonclaims": [
            (
                "Identity mutation is invariant but has no mode-mixing guarantee."
                if mutation_kernel == FIXED_MUTATION
                else "The symmetric random-walk mutation has an invariant-kernel derivation but no finite-run mixing guarantee."
            ),
            "A finite pilot does not establish exhaustive mode discovery or posterior correctness.",
            "The M0 label remains a candidate until an SMC-U fixture and independent q20 evidence pass.",
        ],
    }
    _write_json(output_root / f"{arm.lower()}.json", payload)
    return payload


def _result_markdown(receipt: Mapping[str, Any]) -> str:
    lines = [
        "# q=20 Authority Pilot Result",
        "",
        f"Status: `{receipt['status']}`",
        "",
        "This run uses fresh particles and a fixed/hash-bound protocol. It is a pilot, not an authority admission or posterior result.",
        "",
        "| Arm | Status | Role |",
        "|---|---|---|",
    ]
    for arm, payload in receipt["arms"].items():
        lines.append(f"| {arm} | `{payload['status']}` | {payload['role']} |")
    lines.extend(
        [
            "",
            "## Gate interpretation",
            "",
            "Finite/status, density, support, protocol, and mass checks are hard or promotion gates. Mode occupancy, ESS, and root counts are explanatory/repair diagnostics; they do not prove mode discovery.",
            "",
            "## Nonclaims",
            "",
            "- No historical six-bank particle was reused.",
            "- Identity mutation is an invariant reference kernel but cannot establish mixing.",
            "- No posterior correctness, IID whitening, NeuTra readiness, HMC convergence, or default promotion is claimed.",
        ]
    )
    return "\n".join(lines) + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--particles", type=int, default=DEFAULT_PARTICLES)
    parser.add_argument("--calibration-particles", type=int, default=CALIBRATION_PARTICLES)
    parser.add_argument("--seed", nargs=2, type=int, default=DEFAULT_SEED)
    parser.add_argument("--arms", choices=("m0", "c0", "both"), default="both")
    parser.add_argument(
        "--mutation",
        choices=("identity", "random-walk"),
        default="identity",
        help="use the invariant identity reference or a symmetric MH repair",
    )
    parser.add_argument("--mutation-steps", type=int, default=DEFAULT_MUTATION_STEPS)
    parser.add_argument("--mutation-scale", type=float, default=DEFAULT_MUTATION_SCALE)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.output_root.is_absolute() or ".." in args.output_root.parts:
        raise PilotError("output root must be repository-relative")
    if args.output_root.exists():
        raise PilotError(f"refusing to overwrite existing output root: {args.output_root}")
    if int(args.particles) < 8 or int(args.calibration_particles) < 8:
        raise PilotError("particle counts must be at least eight")
    if int(args.mutation_steps) < 0:
        raise PilotError("mutation steps must be nonnegative")
    mutation_kernel = (
        RANDOM_WALK_MUTATION if args.mutation == "random-walk" else FIXED_MUTATION
    )
    args.output_root.mkdir(parents=True)
    started = time.perf_counter()
    proposal = _load_geometry_proposal()
    chart = _make_chart(proposal)
    target = batch_native_complexity_posterior_target(
        20, jit_compile=True, principal_sqrt_backend="tensorflow_eigh"
    )
    calibration_z, calibration_theta, _calibration_components = _sample_proposal(
        int(args.calibration_particles), chart, DEFENSIVE_EPSILON, (int(args.seed[0]), int(args.seed[1]) + 10)
    )
    calibration = _evaluate(calibration_z, chart, target, DEFENSIVE_EPSILON)
    if not bool(tf.reduce_all(calibration["valid"]).numpy()):
        raise PilotError("calibration proposal contains invalid target rows")
    calibration_ratio = calibration["target_log_prob"] - calibration["proposal_log_prob"]
    schedule, schedule_receipt = _select_fixed_schedule(calibration_ratio)
    calibration_payload = {
        "schema": "bayesfilter.ssl_lstm.q20.authority_pilot.calibration.v1",
        "status": "CALIBRATION_COMPLETED",
        "particle_count": int(args.calibration_particles),
        "schedule_selection": schedule_receipt,
        "selected_schedule": list(schedule),
        "protocol_seed": list(args.seed),
        "calibration_target_signature": target.target_signature(),
        "calibration_adapter_signature": target.adapter_signature(),
        "receipt": _write_tensor(args.output_root / "calibration-log-ratio.tftensor", calibration_ratio),
    }
    _write_json(args.output_root / "calibration.json", calibration_payload)
    arms: dict[str, Any] = {}
    requested = ("M0", "C0") if args.arms == "both" else (("M0",) if args.arms == "m0" else ("C0",))
    for index, arm in enumerate(requested):
        arms[arm] = _run_arm(
            arm,
            int(args.particles),
            (int(args.seed[0]), int(args.seed[1]) + 100 + index),
            schedule,
            chart,
            target,
            args.output_root,
            mutation_kernel,
            int(args.mutation_steps),
            float(args.mutation_scale),
        )
    hard_pass = all(payload["status"] == "PASS_GATE" for payload in arms.values())
    receipt = {
        "schema": "bayesfilter.ssl_lstm.q20.authority_pilot.v1",
        "status": "PASS_GATE" if hard_pass else "CANDIDATE_FAIL_REPAIR",
        "fixture_precondition": "phase1_contracts_passed",
        "arms": arms,
        "selected_schedule": list(schedule),
        "schedule_selection": schedule_receipt,
        "mutation": {
            "kernel": mutation_kernel,
            "steps": int(args.mutation_steps),
            "scale": float(args.mutation_scale),
            "transition_log_density": "symmetric_forward_equals_reverse"
            if mutation_kernel == RANDOM_WALK_MUTATION
            else "identity",
        },
        "run_manifest": {
            "program": PLAN.as_posix(),
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()),
            "command": " ".join(sys.argv),
            "python": sys.executable,
            "python_version": platform.python_version(),
            "tensorflow": tf.__version__,
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "tf_force_gpu_allow_growth": os.environ["TF_FORCE_GPU_ALLOW_GROWTH"],
            "physical_gpus": [device.name for device in tf.config.list_physical_devices("GPU")],
            "logical_gpus": [device.name for device in tf.config.list_logical_devices("GPU")],
            "jit_compile": True,
            "wall_seconds": time.perf_counter() - started,
            "source_sha256": {
                "plan": _sha(PLAN),
                "runner": _sha(RUNNER),
                "geometry": _sha(GEOMETRY),
                "contracts_helper": _sha(ROOT / "bayesfilter/testing/particle_authority_contracts_tf.py"),
            },
        },
        "nonclaims": [
            "Fresh q20 particles are not a finite-run mode-discovery guarantee.",
            "M0 is a candidate authority only; its conditional SMC-U identity remains an obligation.",
            "No modular ETPF/GenUT/LEDH arm, NeuTra training, HMC, posterior correctness, or default change was run.",
        ],
    }
    _write_json(args.output_root / "pilot.json", receipt)
    (args.output_root / "result.md").write_text(_result_markdown(receipt), encoding="ascii")
    print(json.dumps({"status": receipt["status"], "output_root": args.output_root.as_posix()}, sort_keys=True))
    return 0 if hard_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
