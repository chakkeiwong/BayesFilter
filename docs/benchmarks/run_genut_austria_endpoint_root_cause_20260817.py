#!/usr/bin/env python3
"""Bounded Austria batch-GenUT endpoint root-cause diagnostics.

This is a diagnostic lane. It imports the current repository kernels without
changing production source and keeps interior capture in eager mode only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping


def _early_option(name: str, default: str) -> str:
    try:
        index = os.sys.argv.index(name)
    except ValueError:
        return default
    if index + 1 >= len(os.sys.argv):
        raise RuntimeError(f"{name} requires a value")
    return os.sys.argv[index + 1]


os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/bayesfilter-matplotlib")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
    os.environ["CUDA_VISIBLE_DEVICES"] = _early_option("--gpu-index", "0")
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf

from bayesfilter.runtime.gpu_memory_policy import (
    configure_tensorflow_gpu_memory_growth,
)


_EARLY_DEVICE = _early_option("--device", "gpu")
if _EARLY_DEVICE == "gpu":
    _MEMORY_POLICY = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
elif _EARLY_DEVICE == "cpu":
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise RuntimeError("CPU diagnostic requires CUDA_VISIBLE_DEVICES=-1 before import")
    _MEMORY_POLICY = {
        "schema": "bayesfilter.tensorflow.cpu_only_diagnostic.v1",
        "mode": "cpu_only",
        "cuda_visible_devices": "-1",
    }
else:
    raise RuntimeError(f"unknown early device option: {_EARLY_DEVICE}")
tf.config.experimental.enable_op_determinism()

from bayesfilter.highdim import cubature_genut_batch_tf as batch
from bayesfilter.highdim.cubature_genut_batch_tf import BatchCandidateModelAdapter
from bayesfilter.highdim.cubature_genut_neutra_targets import (
    GenUTNeuTraTargetAdapter,
    make_genut_neutra_target,
)
from bayesfilter.highdim.genut_shape_lm_tf import scaled_lm_coefficients_value
from docs.benchmarks.genut_fd_regression import (
    FD_REGRESSION_STEPS,
    evaluate_regression_derivative,
    fit_quadratic_step_regression,
)


PLAN = ROOT / "docs/plans/bayesfilter-austria-genut-neutra-root-cause-hypotheses-fable-handoff-2026-08-17.md"
DEFAULT_OUTPUT = ROOT / "docs/benchmarks/artifacts/genut_austria_endpoint_root_cause_20260817/attempt01/result.json"
EXPECTED_TARGET_SIGNATURE = "4845e7322685e19650024e5886e47d89c8b9c4b70c5d36a639c9b1218d39b5c3"
EXPECTED_ADAPTER_SIGNATURE = "6a56c7a9cb9f488f2f2a44cf86316d4ad80be45ab86b74d33b019f720fd0fee6"
EXPECTED_TARGET_HASHES = {
    "observations": "40c793fb374e84fcd347c66b189352b5997740cc753ea0be03441ecf32828009",
    "initial_noise": "21b49995edf6c72188de0870e1282348178b8ae1be1a63812933be3d30827e82",
    "process_noise": "98e6cf19066e5e3a480d41b5073d3224751a9031bfe793eac4acabf2ef9b526e",
    "design": "d8ad7e0b986cc7c90b6f55b3aaf1f582f7040b77b8cfa5ec7f5f48875f950edd",
}


def _json_value(value: Any) -> Any:
    if isinstance(value, tf.Tensor):
        host = value.numpy()
        if host.shape == ():
            scalar = host.item()
            if isinstance(scalar, bytes):
                return scalar.decode("utf-8")
            if isinstance(scalar, float) and not math.isfinite(scalar):
                return str(scalar)
            return scalar
        return _json_value(host.tolist())
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_value(dict(payload)), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_hash(value: tf.Tensor) -> str:
    encoded = tf.io.serialize_tensor(tf.convert_to_tensor(value)).numpy()
    return hashlib.sha256(encoded).hexdigest()


def _summary(value: tf.Tensor) -> dict[str, Any]:
    tensor = tf.convert_to_tensor(value)
    finite = tf.math.is_finite(tensor) if tensor.dtype.is_floating else tf.ones_like(tensor, tf.bool)
    safe = tf.where(finite, tensor, tf.zeros_like(tensor)) if tensor.dtype.is_floating else tensor
    result: dict[str, Any] = {
        "shape": tensor.shape.as_list(),
        "dtype": tensor.dtype.name,
        "sha256": _tensor_hash(tensor),
        "all_finite": bool(tf.reduce_all(finite).numpy()),
    }
    if tensor.dtype.is_floating and int(tf.size(tensor).numpy()) > 0:
        result.update(
            {
                "minimum": float(tf.reduce_min(safe).numpy()),
                "maximum": float(tf.reduce_max(safe).numpy()),
                "maximum_abs": float(tf.reduce_max(tf.abs(safe)).numpy()),
            }
        )
    return result


def _comparison(left: tf.Tensor, right: tf.Tensor) -> dict[str, Any]:
    first = tf.convert_to_tensor(left)
    second = tf.convert_to_tensor(right, dtype=first.dtype)
    finite_pair = tf.math.is_finite(first) & tf.math.is_finite(second)
    difference = tf.where(finite_pair, tf.abs(first - second), tf.zeros_like(first))
    scale = tf.maximum(tf.maximum(tf.abs(first), tf.abs(second)), tf.ones_like(first))
    spacing_first = tf.abs(tf.math.nextafter(first, tf.fill(tf.shape(first), tf.cast(float("inf"), first.dtype))) - first)
    spacing_second = tf.abs(tf.math.nextafter(second, tf.fill(tf.shape(second), tf.cast(float("inf"), second.dtype))) - second)
    spacing = tf.maximum(
        tf.maximum(spacing_first, spacing_second),
        tf.cast(1.1754943508222875e-38, first.dtype),
    )
    return {
        "exact_equal": bool(tf.reduce_all(tf.equal(first, second)).numpy()),
        "finite_pattern_equal": bool(
            tf.reduce_all(tf.equal(tf.math.is_finite(first), tf.math.is_finite(second))).numpy()
        ),
        "maximum_absolute_error": float(tf.reduce_max(difference).numpy()),
        "maximum_scale_relative_error": float(tf.reduce_max(difference / scale).numpy()),
        "maximum_approximate_ulp_error": float(tf.reduce_max(difference / spacing).numpy()),
        "left_sha256": _tensor_hash(first),
        "right_sha256": _tensor_hash(second),
    }


def _controls(target: GenUTNeuTraTargetAdapter, *, correction_steps: int) -> dict[str, Any]:
    controls = target.controls
    return {
        "epsilon": controls.epsilon,
        "sinkhorn_steps": controls.sinkhorn_steps,
        "balance_steps": controls.balance_steps,
        "ridge": controls.ridge,
        "transition_before_first_observation": target.transition_before_first_observation,
        "higher_moment_correction_steps": correction_steps,
        "higher_moment_strength": controls.higher_moment_strength,
        "higher_moment_floor": controls.higher_moment_floor,
        "higher_moment_lm_damping": controls.higher_moment_lm_damping,
        "higher_moment_lm_scale_floor": controls.higher_moment_lm_scale_floor,
        "higher_moment_trust_radius": controls.higher_moment_trust_radius,
    }


def _endpoint(
    target: GenUTNeuTraTargetAdapter,
    *,
    horizon: int,
    correction_steps: int,
    mode: str,
) -> dict[str, Any]:
    theta = tf.zeros([1, target.parameter_dim], tf.float32)
    observations = target.observations[:horizon]
    process_noise = target.process_noise[:horizon]
    kwargs = _controls(target, correction_steps=correction_steps)

    def value_call(values: tf.Tensor) -> tuple[tf.Tensor, Mapping[str, tf.Tensor]]:
        return batch.batch_finite_value(
            target.filter_adapter,
            values,
            observations,
            target.initial_noise,
            process_noise,
            target.design,
            **kwargs,
        )

    def score_call(values: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, Mapping[str, tf.Tensor]]:
        return batch.batch_finite_value_score(
            target.filter_adapter,
            values,
            observations,
            target.initial_noise,
            process_noise,
            target.design,
            **kwargs,
        )

    if mode == "eager":
        value, value_status = value_call(theta)
        score_value, score, score_status = score_call(theta)
    else:
        jit_compile = mode == "xla"
        value_graph = tf.function(value_call, jit_compile=jit_compile, autograph=False)
        score_graph = tf.function(score_call, jit_compile=jit_compile, autograph=False)
        value, value_status = value_graph(theta)
        score_value, score, score_status = score_graph(theta)
    return {
        "mode": mode,
        "horizon": horizon,
        "correction_steps": correction_steps,
        "value_only": _summary(value),
        "value_score_value": _summary(score_value),
        "score": _summary(score),
        "value_comparison": _comparison(value, score_value),
        "value_only_program_valid": _json_value(value_status["program_valid"]),
        "value_score_program_valid": _json_value(score_status["program_valid"]),
    }


def _moments(values: tf.Tensor, weights: tf.Tensor | None = None) -> tuple[tf.Tensor, tf.Tensor]:
    if weights is None:
        mean = tf.reduce_mean(values, axis=1)
        centered = values - mean[:, None, :]
        covariance = tf.einsum("bni,bnj->bij", centered, centered) / tf.cast(
            tf.shape(values)[1], values.dtype
        )
    else:
        mean = tf.reduce_sum(weights[:, :, None] * values, axis=1)
        centered = values - mean[:, None, :]
        covariance = tf.einsum("bn,bni,bnj->bij", weights, centered, centered)
    return mean, batch._sym(covariance)  # noqa: SLF001


def _standardize(values: tf.Tensor) -> tf.Tensor:
    mean, covariance = _moments(values)
    return batch._right_solve(  # noqa: SLF001
        tf.linalg.cholesky(covariance), values - mean[:, None, :]
    )


def _shape_primal_trace(
    standardized: tf.Tensor,
    target_skew: tf.Tensor,
    target_kurtosis: tf.Tensor,
    *,
    strength: float,
    floor: float,
    lm_damping: float,
    lm_scale_floor: float,
    trust_radius: float,
) -> tuple[tf.Tensor, dict[str, tf.Tensor]]:
    m3 = tf.reduce_mean(tf.pow(standardized, 3.0), axis=1)
    m4 = tf.reduce_mean(tf.pow(standardized, 4.0), axis=1)
    residual3 = target_skew - m3
    residual4 = target_kurtosis - m4
    direction3 = tf.square(standardized) - 1.0 - m3[:, None, :] * standardized
    direction4 = (
        tf.pow(standardized, 3.0)
        - m3[:, None, :]
        - m4[:, None, :] * standardized
    )
    j33 = tf.reduce_mean(3.0 * tf.square(standardized) * direction3, axis=1)
    j34 = tf.reduce_mean(3.0 * tf.square(standardized) * direction4, axis=1)
    j43 = tf.reduce_mean(4.0 * tf.pow(standardized, 3.0) * direction3, axis=1)
    j44 = tf.reduce_mean(4.0 * tf.pow(standardized, 3.0) * direction4, axis=1)
    jacobian = tf.stack(
        [tf.stack([j33, j34], axis=-1), tf.stack([j43, j44], axis=-1)],
        axis=-2,
    )
    residual = tf.stack([residual3, residual4], axis=-1)
    normal = tf.linalg.matmul(jacobian, jacobian, transpose_a=True)
    normal += tf.cast(floor, standardized.dtype) * tf.eye(
        2,
        batch_shape=[tf.shape(standardized)[0], tf.shape(standardized)[2]],
        dtype=standardized.dtype,
    )
    rhs = tf.linalg.matvec(jacobian, residual, transpose_a=True)
    if lm_damping > 0.0:
        coefficient = scaled_lm_coefficients_value(
            jacobian,
            residual,
            strength=strength,
            damping=lm_damping,
            scale_floor=lm_scale_floor,
        )["coefficient"]
    else:
        coefficient = tf.cast(strength, standardized.dtype) * tf.linalg.solve(
            normal, rhs[:, :, :, None]
        )[:, :, :, 0]
    displacement = (
        direction3 * coefficient[:, None, :, 0]
        + direction4 * coefficient[:, None, :, 1]
    )
    if trust_radius > 0.0:
        from bayesfilter.highdim.genut_shape_lm_tf import smooth_rms_cap_value

        displacement = smooth_rms_cap_value(
            displacement, radius=trust_radius
        )["displacement"]
    corrected = standardized + displacement
    corrected_standardized = _standardize(corrected)
    trace = {
        "iteration_standardized": standardized,
        "m3": m3,
        "m4": m4,
        "residual": residual,
        "direction3": direction3,
        "direction4": direction4,
        "jacobian": jacobian,
        "normal": normal,
        "rhs": rhs,
        "coefficient": coefficient,
        "displacement": displacement,
        "corrected": corrected,
        "post_correction_standardized": corrected_standardized,
    }
    return corrected_standardized, trace


def _projected_higher_primal(
    source: tf.Tensor,
    weights: tf.Tensor,
    points: tf.Tensor,
    *,
    correction_steps: int,
    redundant_iteration_standardization: bool,
    controls: Mapping[str, Any],
) -> tuple[tf.Tensor, list[dict[str, tf.Tensor]]]:
    mean, covariance = _moments(source, weights)
    target_chol = tf.linalg.cholesky(covariance)
    source_standardized = batch._right_solve(  # noqa: SLF001
        target_chol, source - mean[:, None, :]
    )
    target_skew = tf.reduce_sum(
        weights[:, :, None] * tf.pow(source_standardized, 3.0), axis=1
    )
    target_kurtosis = tf.reduce_sum(
        weights[:, :, None] * tf.pow(source_standardized, 4.0), axis=1
    )
    standardized = _standardize(points)
    traces: list[dict[str, tf.Tensor]] = []
    for _ in range(correction_steps):
        iteration_input = standardized
        if redundant_iteration_standardization:
            standardized = _standardize(standardized)
        standardized, trace = _shape_primal_trace(
            standardized,
            target_skew,
            target_kurtosis,
            strength=float(controls["higher_moment_strength"]),
            floor=float(controls["higher_moment_floor"]),
            lm_damping=float(controls["higher_moment_lm_damping"]),
            lm_scale_floor=float(controls["higher_moment_lm_scale_floor"]),
            trust_radius=float(controls["higher_moment_trust_radius"]),
        )
        trace["pre_iteration_input"] = iteration_input
        trace["post_optional_input_standardization"] = trace[
            "iteration_standardized"
        ]
        traces.append(trace)
    output = mean[:, None, :] + tf.linalg.matmul(
        standardized, target_chol, transpose_b=True
    )
    return output, traces


def _first_step_inputs(target: GenUTNeuTraTargetAdapter) -> dict[str, tf.Tensor]:
    theta = tf.zeros([1, target.parameter_dim], tf.float32)
    adapter = target.filter_adapter
    time_index = tf.constant(0, tf.int32)
    initial = adapter.initial_value(theta, target.initial_noise)
    initial_tangent = adapter.initial_tangent(theta, target.initial_noise)
    transitioned = adapter.transition_value(
        theta, initial, target.process_noise[0], time_index
    )
    transitioned_tangent = adapter.transition_tangent(
        theta, initial, target.process_noise[0], initial_tangent, time_index
    )
    likelihood = adapter.observation_value(
        theta, transitioned, target.observations[0], time_index
    )
    likelihood_tangent = adapter.observation_tangent(
        theta,
        transitioned,
        transitioned_tangent,
        target.observations[0],
        time_index,
    )
    count = tf.shape(transitioned)[1]
    weights = tf.fill([1, count], tf.cast(1.0, tf.float32) / tf.cast(count, tf.float32))
    weight_tangent = tf.zeros([1, count, target.parameter_dim], tf.float32)
    log_weights = tf.math.log(weights) + likelihood
    increment = tf.reduce_logsumexp(log_weights, axis=1)
    normalized_weights = tf.exp(log_weights - increment[:, None])
    log_weight_tangent = likelihood_tangent
    increment_tangent = tf.reduce_sum(
        normalized_weights[:, :, None] * log_weight_tangent, axis=1
    )
    normalized_weight_tangent = normalized_weights[:, :, None] * (
        log_weight_tangent - increment_tangent[:, None, :]
    )
    return {
        "theta": theta,
        "initial": initial,
        "initial_tangent": initial_tangent,
        "transitioned": transitioned,
        "transitioned_tangent": transitioned_tangent,
        "likelihood": likelihood,
        "likelihood_tangent": likelihood_tangent,
        "increment": increment,
        "increment_tangent": increment_tangent,
        "weights": normalized_weights,
        "weight_tangent": normalized_weight_tangent,
    }


def _localization_and_h1(target: GenUTNeuTraTargetAdapter) -> dict[str, Any]:
    inputs = _first_step_inputs(target)
    controls = _controls(target, correction_steps=4)
    zeros_particles = tf.zeros_like(inputs["transitioned_tangent"])
    zeros_weights = tf.zeros_like(inputs["weight_tangent"])
    value_transport = batch._sinkhorn_barycentric_batch_value(  # noqa: SLF001
        inputs["transitioned"],
        inputs["weights"],
        epsilon=controls["epsilon"],
        sinkhorn_steps=controls["sinkhorn_steps"],
        balance_steps=controls["balance_steps"],
    )
    jvp_transport = batch._sinkhorn_barycentric_batch_jvp(  # noqa: SLF001
        inputs["transitioned"],
        inputs["weights"],
        zeros_particles,
        zeros_weights,
        epsilon=controls["epsilon"],
        sinkhorn_steps=controls["sinkhorn_steps"],
        balance_steps=controls["balance_steps"],
    )
    value_reset = batch._restore_cloud_batch_value(  # noqa: SLF001
        inputs["transitioned"],
        inputs["weights"],
        target.design,
        epsilon=controls["epsilon"],
        sinkhorn_steps=controls["sinkhorn_steps"],
        balance_steps=controls["balance_steps"],
        ridge=controls["ridge"],
    )
    jvp_reset = batch._restore_cloud_batch_jvp(  # noqa: SLF001
        inputs["transitioned"],
        inputs["weights"],
        zeros_particles,
        zeros_weights,
        target.design,
        epsilon=controls["epsilon"],
        sinkhorn_steps=controls["sinkhorn_steps"],
        balance_steps=controls["balance_steps"],
        ridge=controls["ridge"],
    )
    value_higher = batch._higher_moment_batch_value(  # noqa: SLF001
        inputs["transitioned"],
        inputs["weights"],
        value_reset["particles"],
        correction_steps=4,
        strength=controls["higher_moment_strength"],
        floor=controls["higher_moment_floor"],
        lm_damping=controls["higher_moment_lm_damping"],
        lm_scale_floor=controls["higher_moment_lm_scale_floor"],
        trust_radius=controls["higher_moment_trust_radius"],
    )
    jvp_higher = batch._higher_moment_batch_jvp(  # noqa: SLF001
        inputs["transitioned"],
        inputs["weights"],
        zeros_particles,
        zeros_weights,
        jvp_reset["particles"],
        jvp_reset["particles_tangent"],
        correction_steps=4,
        strength=controls["higher_moment_strength"],
        floor=controls["higher_moment_floor"],
        lm_damping=controls["higher_moment_lm_damping"],
        lm_scale_floor=controls["higher_moment_lm_scale_floor"],
        trust_radius=controls["higher_moment_trust_radius"],
    )
    value_projection, value_traces = _projected_higher_primal(
        inputs["transitioned"],
        inputs["weights"],
        value_reset["particles"],
        correction_steps=4,
        redundant_iteration_standardization=False,
        controls=controls,
    )
    jvp_projection, jvp_traces = _projected_higher_primal(
        inputs["transitioned"],
        inputs["weights"],
        jvp_reset["particles"],
        correction_steps=4,
        redundant_iteration_standardization=True,
        controls=controls,
    )
    first_trace = []
    trace_names = tuple(value_traces[0])
    for name in trace_names:
        first_trace.append(
            {
                "tensor": name,
                "comparison": _comparison(value_traces[0][name], jvp_traces[0][name]),
            }
        )
    first_unequal = next(
        (
            row["tensor"]
            for row in first_trace
            if not row["comparison"]["exact_equal"]
        ),
        None,
    )
    return {
        "clarifications": {
            "C1": "The stop rule applies to particle-path tensors; an H3A gap-scalar difference alone does not stop localization.",
            "C2": "Extracted graph/XLA block tests are mode-sensitivity evidence only; endpoint claims require endpoint-only arms.",
            "C3": "The eager endpoint mismatch is reproduced before interpreting this localization.",
        },
        "upstream": {
            "initial": _summary(inputs["initial"]),
            "transitioned": _summary(inputs["transitioned"]),
            "likelihood": _summary(inputs["likelihood"]),
            "increment": _summary(inputs["increment"]),
            "normalized_weights": _summary(inputs["weights"]),
            "sinkhorn_particles": _comparison(
                value_transport["particles"], jvp_transport["particles"]
            ),
            "contract_e_particles": _comparison(
                value_reset["particles"], jvp_reset["particles"]
            ),
            "contract_e_valid_equal": bool(
                tf.reduce_all(tf.equal(value_reset["valid"], jvp_reset["valid"])).numpy()
            ),
            "contract_e_minimum_gap": _comparison(
                value_reset["minimum_gap_eigenvalue"],
                jvp_reset["minimum_gap_eigenvalue"],
            ),
        },
        "current_higher_particles": _comparison(
            value_higher["particles"], jvp_higher["particles"]
        ),
        "projection_validation": {
            "value_projection_matches_current_value": _comparison(
                value_projection, value_higher["particles"]
            ),
            "jvp_order_projection_matches_current_jvp_primal": _comparison(
                jvp_projection, jvp_higher["particles"]
            ),
        },
        "h1_causal_arms": {
            "forward_skip_redundant_standardization_vs_value": _comparison(
                value_projection, value_higher["particles"]
            ),
            "reverse_value_adds_redundant_standardization_vs_current_jvp": _comparison(
                jvp_projection, jvp_higher["particles"]
            ),
            "forward_vs_current_jvp": _comparison(
                value_projection, jvp_higher["particles"]
            ),
            "reverse_vs_current_value": _comparison(
                jvp_projection, value_higher["particles"]
            ),
        },
        "first_iteration_trace": first_trace,
        "first_unequal_particle_path_tensor": first_unequal,
    }


def _solver_diagnostic(target: GenUTNeuTraTargetAdapter) -> dict[str, Any]:
    inputs = _first_step_inputs(target)
    controls = _controls(target, correction_steps=1)
    reset = batch._restore_cloud_batch_value(  # noqa: SLF001
        inputs["transitioned"],
        inputs["weights"],
        target.design,
        epsilon=controls["epsilon"],
        sinkhorn_steps=controls["sinkhorn_steps"],
        balance_steps=controls["balance_steps"],
        ridge=controls["ridge"],
    )
    source_mean, source_covariance = _moments(
        inputs["transitioned"], inputs["weights"]
    )
    source_chol = tf.linalg.cholesky(source_covariance)
    source_standardized = batch._right_solve(  # noqa: SLF001
        source_chol, inputs["transitioned"] - source_mean[:, None, :]
    )
    target_skew = tf.reduce_sum(
        inputs["weights"][:, :, None] * tf.pow(source_standardized, 3.0), axis=1
    )
    target_kurtosis = tf.reduce_sum(
        inputs["weights"][:, :, None] * tf.pow(source_standardized, 4.0), axis=1
    )
    standardized = _standardize(reset["particles"])
    _, trace = _shape_primal_trace(
        standardized,
        target_skew,
        target_kurtosis,
        strength=controls["higher_moment_strength"],
        floor=controls["higher_moment_floor"],
        lm_damping=0.0,
        lm_scale_floor=controls["higher_moment_lm_scale_floor"],
        trust_radius=0.0,
    )
    jacobian = trace["jacobian"]
    residual = trace["residual"]
    normal_coefficient = trace["coefficient"]
    lm = scaled_lm_coefficients_value(
        jacobian,
        residual,
        strength=controls["higher_moment_strength"],
        damping=1.0e-2,
        scale_floor=1.0e-4,
    )
    flat_jacobian = tf.reshape(jacobian, [-1, 2, 2])
    flat_residual = tf.reshape(residual, [-1, 2, 1])
    qr_coefficient = tf.reshape(
        tf.cast(controls["higher_moment_strength"], tf.float32)
        * tf.linalg.lstsq(flat_jacobian, flat_residual, fast=False)[:, :, 0],
        tf.shape(normal_coefficient),
    )
    jacobian64 = tf.cast(flat_jacobian, tf.float64)
    residual64 = tf.cast(flat_residual, tf.float64)
    svd_coefficient64 = tf.reshape(
        tf.cast(controls["higher_moment_strength"], tf.float64)
        * tf.linalg.matmul(tf.linalg.pinv(jacobian64), residual64)[:, :, 0],
        tf.shape(normal_coefficient),
    )
    singular_values = tf.linalg.svd(flat_jacobian, compute_uv=False)
    normal_eigenvalues = tf.linalg.eigvalsh(
        tf.linalg.matmul(flat_jacobian, flat_jacobian, transpose_a=True)
    )

    def solve_row(name: str, coefficient: tf.Tensor) -> dict[str, Any]:
        coefficient32 = tf.cast(coefficient, tf.float32)
        residual_after = tf.linalg.matvec(jacobian, coefficient32) - tf.cast(
            controls["higher_moment_strength"], tf.float32
        ) * residual
        return {
            "name": name,
            "coefficient": _summary(coefficient),
            "maximum_equation_residual": float(
                tf.reduce_max(tf.abs(residual_after)).numpy()
            ),
        }

    return {
        "fixed_input_hashes": {
            "jacobian": _tensor_hash(jacobian),
            "residual": _tensor_hash(residual),
        },
        "unregularized_jacobian_condition": {
            "maximum": float(
                tf.reduce_max(singular_values[:, 0] / singular_values[:, 1]).numpy()
            ),
            "minimum_singular_value": float(tf.reduce_min(singular_values[:, 1]).numpy()),
        },
        "unregularized_normal_system_condition": {
            "maximum": float(
                tf.reduce_max(normal_eigenvalues[:, 1] / normal_eigenvalues[:, 0]).numpy()
            ),
            "minimum_eigenvalue": float(tf.reduce_min(normal_eigenvalues[:, 0]).numpy()),
        },
        "solvers": [
            solve_row("current_normal_equations", normal_coefficient),
            solve_row("column_scaled_lm_damping_1e-2", lm["coefficient"]),
            solve_row("direct_qr_lstsq", qr_coefficient),
            solve_row("float64_svd_pseudoinverse_reference", svd_coefficient64),
        ],
    }


def _invalid_tangent_adapter(adapter: BatchCandidateModelAdapter) -> BatchCandidateModelAdapter:
    def invalid_initial_tangent(theta: tf.Tensor, noise: tf.Tensor) -> tf.Tensor:
        shape = [tf.shape(theta)[0], tf.shape(noise)[0], adapter.state_dimension, adapter.parameter_count]
        return tf.fill(shape, tf.constant(float("nan"), tf.float32))

    return BatchCandidateModelAdapter(
        adapter.state_dimension,
        adapter.parameter_count,
        adapter.initial_value,
        invalid_initial_tangent,
        adapter.transition_value,
        adapter.transition_tangent,
        adapter.observation_value,
        adapter.observation_tangent,
    )


def _invariants(target: GenUTNeuTraTargetAdapter) -> dict[str, Any]:
    kwargs = _controls(target, correction_steps=1)
    theta = tf.zeros([1, target.parameter_dim], tf.float32)
    observations = target.observations[:1]
    process_noise = target.process_noise[:1]
    invalid_value, invalid_score, invalid_status = batch.batch_finite_value_score(
        _invalid_tangent_adapter(target.filter_adapter),
        theta,
        observations,
        target.initial_noise,
        process_noise,
        target.design,
        **kwargs,
    )
    single_value, single_score, single_status = batch.batch_finite_value_score(
        target.filter_adapter,
        theta,
        observations,
        target.initial_noise,
        process_noise,
        target.design,
        **kwargs,
    )
    pair_theta = tf.concat([theta, theta + tf.constant([[0.01, -0.01, 0.005]], tf.float32)], axis=0)
    pair_value, pair_score, pair_status = batch.batch_finite_value_score(
        target.filter_adapter,
        pair_theta,
        observations,
        target.initial_noise,
        process_noise,
        target.design,
        **kwargs,
    )
    return {
        "tangent_only_invalidity": {
            "value": _json_value(invalid_value),
            "score": _json_value(invalid_score),
            "program_valid": _json_value(invalid_status["program_valid"]),
            "pass_fail_closed": bool(
                tf.reduce_all(~tf.math.is_finite(invalid_value)).numpy()
                and tf.reduce_all(~tf.math.is_finite(invalid_score)).numpy()
                and tf.reduce_all(~invalid_status["program_valid"]).numpy()
            ),
        },
        "batch_composition": {
            "row0_value": _comparison(single_value[0], pair_value[0]),
            "row0_score": _comparison(single_score[0], pair_score[0]),
            "row0_valid_equal": bool(
                tf.equal(single_status["program_valid"][0], pair_status["program_valid"][0]).numpy()
            ),
        },
    }


def _local_autodiff_checks(target: GenUTNeuTraTargetAdapter) -> dict[str, Any]:
    adapter = target.filter_adapter
    theta = tf.zeros([1, target.parameter_dim], tf.float32)
    count = 36
    initial_noise = target.initial_noise[:count]
    process_noise = target.process_noise[0, :count]
    time_index = tf.constant(0, tf.int32)
    initial = adapter.initial_value(theta, initial_noise)
    initial_tangent = adapter.initial_tangent(theta, initial_noise)
    manual_transition = adapter.transition_value(
        theta, initial, process_noise, time_index
    )
    manual_transition_tangent = adapter.transition_tangent(
        theta, initial, process_noise, initial_tangent, time_index
    )
    with tf.GradientTape() as tape:
        tape.watch(theta)
        autodiff_initial = adapter.initial_value(theta, initial_noise)
        autodiff_transition = adapter.transition_value(
            theta, autodiff_initial, process_noise, time_index
        )
    autodiff_transition_tangent = tape.batch_jacobian(
        autodiff_transition, theta
    )
    manual_observation = adapter.observation_value(
        theta,
        manual_transition,
        target.observations[0],
        time_index,
    )
    manual_observation_tangent = adapter.observation_tangent(
        theta,
        manual_transition,
        manual_transition_tangent,
        target.observations[0],
        time_index,
    )
    with tf.GradientTape() as tape:
        tape.watch(theta)
        autodiff_initial = adapter.initial_value(theta, initial_noise)
        autodiff_transition = adapter.transition_value(
            theta, autodiff_initial, process_noise, time_index
        )
        autodiff_observation = adapter.observation_value(
            theta,
            autodiff_transition,
            target.observations[0],
            time_index,
        )
    autodiff_observation_tangent = tape.batch_jacobian(
        autodiff_observation, theta
    )
    return {
        "particle_count": count,
        "transition_value": _comparison(
            manual_transition, autodiff_transition
        ),
        "transition_tangent": _comparison(
            manual_transition_tangent, autodiff_transition_tangent
        ),
        "observation_value": _comparison(
            manual_observation, autodiff_observation
        ),
        "observation_total_tangent": _comparison(
            manual_observation_tangent, autodiff_observation_tangent
        ),
    }


def _first_step_filter_block_autodiff(
    target: GenUTNeuTraTargetAdapter,
) -> dict[str, Any]:
    inputs = _first_step_inputs(target)
    controls = _controls(target, correction_steps=4)
    reset_value = batch._restore_cloud_batch_value(  # noqa: SLF001
        inputs["transitioned"],
        inputs["weights"],
        target.design,
        epsilon=controls["epsilon"],
        sinkhorn_steps=controls["sinkhorn_steps"],
        balance_steps=controls["balance_steps"],
        ridge=controls["ridge"],
    )
    reset_autodiff_directions = []
    for parameter in range(target.parameter_dim):
        with tf.autodiff.ForwardAccumulator(
            (inputs["transitioned"], inputs["weights"]),
            (
                inputs["transitioned_tangent"][..., parameter],
                inputs["weight_tangent"][..., parameter],
            ),
        ) as accumulator:
            reset_direction = batch._restore_cloud_batch_value(  # noqa: SLF001
                inputs["transitioned"],
                inputs["weights"],
                target.design,
                epsilon=controls["epsilon"],
                sinkhorn_steps=controls["sinkhorn_steps"],
                balance_steps=controls["balance_steps"],
                ridge=controls["ridge"],
            )["particles"]
        reset_autodiff_directions.append(accumulator.jvp(reset_direction))
    reset_autodiff_tangent = tf.stack(reset_autodiff_directions, axis=-1)
    reset_manual = batch._restore_cloud_batch_jvp(  # noqa: SLF001
        inputs["transitioned"],
        inputs["weights"],
        inputs["transitioned_tangent"],
        inputs["weight_tangent"],
        target.design,
        epsilon=controls["epsilon"],
        sinkhorn_steps=controls["sinkhorn_steps"],
        balance_steps=controls["balance_steps"],
        ridge=controls["ridge"],
    )
    def higher_value(correction_steps: int) -> tf.Tensor:
        return batch._higher_moment_batch_value(  # noqa: SLF001
            inputs["transitioned"],
            inputs["weights"],
            reset_value["particles"],
            correction_steps=correction_steps,
            strength=controls["higher_moment_strength"],
            floor=controls["higher_moment_floor"],
            lm_damping=controls["higher_moment_lm_damping"],
            lm_scale_floor=controls["higher_moment_lm_scale_floor"],
            trust_radius=controls["higher_moment_trust_radius"],
        )["particles"]

    higher_value_particles = higher_value(4)
    higher_autodiff_directions = []
    for parameter in range(target.parameter_dim):
        with tf.autodiff.ForwardAccumulator(
            (
                inputs["transitioned"],
                inputs["weights"],
                reset_value["particles"],
            ),
            (
                inputs["transitioned_tangent"][..., parameter],
                inputs["weight_tangent"][..., parameter],
                reset_autodiff_tangent[..., parameter],
            ),
        ) as accumulator:
            higher_direction = higher_value(4)
        higher_autodiff_directions.append(accumulator.jvp(higher_direction))
    higher_autodiff_tangent = tf.stack(higher_autodiff_directions, axis=-1)
    higher_manual = batch._higher_moment_batch_jvp(  # noqa: SLF001
        inputs["transitioned"],
        inputs["weights"],
        inputs["transitioned_tangent"],
        inputs["weight_tangent"],
        reset_manual["particles"],
        reset_manual["particles_tangent"],
        correction_steps=controls["higher_moment_correction_steps"],
        strength=controls["higher_moment_strength"],
        floor=controls["higher_moment_floor"],
        lm_damping=controls["higher_moment_lm_damping"],
        lm_scale_floor=controls["higher_moment_lm_scale_floor"],
        trust_radius=controls["higher_moment_trust_radius"],
    )
    higher_manual_with_autodiff_reset = batch._higher_moment_batch_jvp(  # noqa: SLF001
        inputs["transitioned"],
        inputs["weights"],
        inputs["transitioned_tangent"],
        inputs["weight_tangent"],
        reset_value["particles"],
        reset_autodiff_tangent,
        correction_steps=controls["higher_moment_correction_steps"],
        strength=controls["higher_moment_strength"],
        floor=controls["higher_moment_floor"],
        lm_damping=controls["higher_moment_lm_damping"],
        lm_scale_floor=controls["higher_moment_lm_scale_floor"],
        trust_radius=controls["higher_moment_trust_radius"],
    )
    tangent_ablations = {}
    zero_source = tf.zeros_like(inputs["transitioned_tangent"])
    zero_weights = tf.zeros_like(inputs["weight_tangent"])
    zero_points = tf.zeros_like(reset_manual["particles_tangent"])
    ablation_cases = {
        "zero_all": (zero_source, zero_weights, zero_points),
        "source_only": (inputs["transitioned_tangent"], zero_weights, zero_points),
        "point_only": (zero_source, zero_weights, reset_manual["particles_tangent"]),
        "full": (
            inputs["transitioned_tangent"],
            inputs["weight_tangent"],
            reset_manual["particles_tangent"],
        ),
    }
    for name, (source_tangent, weights_tangent, points_tangent) in ablation_cases.items():
        with tf.autodiff.ForwardAccumulator(
            (
                inputs["transitioned"],
                inputs["weights"],
                reset_value["particles"],
            ),
            (
                source_tangent[..., 0],
                weights_tangent[..., 0],
                points_tangent[..., 0],
            ),
        ) as accumulator:
            value_direction = higher_value(4)
        autodiff_direction = accumulator.jvp(value_direction)
        manual_direction = batch._higher_moment_batch_jvp(  # noqa: SLF001
            inputs["transitioned"],
            inputs["weights"],
            source_tangent,
            weights_tangent,
            reset_value["particles"],
            points_tangent,
            correction_steps=4,
            strength=controls["higher_moment_strength"],
            floor=controls["higher_moment_floor"],
            lm_damping=controls["higher_moment_lm_damping"],
            lm_scale_floor=controls["higher_moment_lm_scale_floor"],
            trust_radius=controls["higher_moment_trust_radius"],
        )
        tangent_ablations[name] = _comparison(
            autodiff_direction, manual_direction["particles_tangent"][..., 0]
        )
    depth_ladder = []
    point_direction = reset_manual["particles_tangent"][..., 0]
    for depth in range(0, 5):
        with tf.autodiff.ForwardAccumulator(
            reset_value["particles"], point_direction
        ) as accumulator:
            depth_value = batch._higher_moment_batch_value(  # noqa: SLF001
                inputs["transitioned"],
                inputs["weights"],
                reset_value["particles"],
                correction_steps=depth,
                strength=controls["higher_moment_strength"],
                floor=controls["higher_moment_floor"],
                lm_damping=controls["higher_moment_lm_damping"],
                lm_scale_floor=controls["higher_moment_lm_scale_floor"],
                trust_radius=controls["higher_moment_trust_radius"],
            )["particles"]
        depth_autodiff = accumulator.jvp(depth_value)
        depth_manual = batch._higher_moment_batch_jvp(  # noqa: SLF001
            inputs["transitioned"],
            inputs["weights"],
            zero_source,
            zero_weights,
            reset_value["particles"],
            tf.concat(
                [point_direction[..., None], tf.zeros_like(point_direction[..., None]), tf.zeros_like(point_direction[..., None])],
                axis=-1,
            ),
            correction_steps=depth,
            strength=controls["higher_moment_strength"],
            floor=controls["higher_moment_floor"],
            lm_damping=controls["higher_moment_lm_damping"],
            lm_scale_floor=controls["higher_moment_lm_scale_floor"],
            trust_radius=controls["higher_moment_trust_radius"],
        )
        depth_ladder.append(
            {
                "correction_steps": depth,
                "comparison": _comparison(
                    depth_autodiff,
                    depth_manual["particles_tangent"][..., 0],
                ),
            }
        )
    return {
        "reset_primal": _comparison(
            reset_value["particles"], reset_manual["particles"]
        ),
        "reset_tangent": _comparison(
            reset_autodiff_tangent, reset_manual["particles_tangent"]
        ),
        "higher_moment_primal": _comparison(
            higher_value_particles, higher_manual["particles"]
        ),
        "higher_moment_tangent": _comparison(
            higher_autodiff_tangent, higher_manual["particles_tangent"]
        ),
        "higher_moment_tangent_with_autodiff_reset_input": _comparison(
            higher_autodiff_tangent,
            higher_manual_with_autodiff_reset["particles_tangent"],
        ),
        "tangent_ablations_parameter0": tangent_ablations,
        "point_tangent_correction_depth_ladder": depth_ladder,
    }


def _standardization_primitive_checks(
    target: GenUTNeuTraTargetAdapter,
) -> dict[str, Any]:
    inputs = _first_step_inputs(target)
    controls = _controls(target, correction_steps=4)
    reset_value = batch._restore_cloud_batch_value(  # noqa: SLF001
        inputs["transitioned"],
        inputs["weights"],
        target.design,
        epsilon=controls["epsilon"],
        sinkhorn_steps=controls["sinkhorn_steps"],
        balance_steps=controls["balance_steps"],
        ridge=controls["ridge"],
    )
    reset_manual = batch._restore_cloud_batch_jvp(  # noqa: SLF001
        inputs["transitioned"],
        inputs["weights"],
        inputs["transitioned_tangent"],
        inputs["weight_tangent"],
        target.design,
        epsilon=controls["epsilon"],
        sinkhorn_steps=controls["sinkhorn_steps"],
        balance_steps=controls["balance_steps"],
        ridge=controls["ridge"],
    )
    points = reset_value["particles"]
    points_tangent = reset_manual["particles_tangent"]
    mean, covariance, mean_tangent, covariance_tangent = batch._uniform_moments_jvp(  # noqa: SLF001
        points, points_tangent
    )
    with tf.autodiff.ForwardAccumulator(
        points, points_tangent[..., 0]
    ) as accumulator:
        auto_mean = tf.reduce_mean(points, axis=1)
        centered = points - auto_mean[:, None, :]
        auto_covariance = batch._sym(  # noqa: SLF001
            tf.einsum("bni,bnj->bij", centered, centered)
            / tf.cast(tf.shape(points)[1], points.dtype)
        )
        auto_chol = tf.linalg.cholesky(auto_covariance)
        auto_standardized = batch._right_solve(  # noqa: SLF001
            auto_chol, centered
        )
    auto_mean_tangent = accumulator.jvp(auto_mean)
    auto_covariance_tangent = accumulator.jvp(auto_covariance)
    auto_chol_tangent = accumulator.jvp(auto_chol)
    auto_standardized_tangent = accumulator.jvp(auto_standardized)
    manual_chol = tf.linalg.cholesky(covariance)
    manual_chol_tangent = batch._cholesky_jvp(  # noqa: SLF001
        manual_chol, covariance_tangent
    )
    centered = points - mean[:, None, :]
    manual_standardized_tangent = batch._right_solve_jvp(  # noqa: SLF001
        manual_chol,
        manual_chol_tangent,
        centered,
        points_tangent - mean_tangent[:, None, :, :],
    )
    return {
        "uniform_mean_tangent": _comparison(
            mean_tangent[..., 0], auto_mean_tangent
        ),
        "uniform_covariance_tangent": _comparison(
            covariance_tangent[..., 0], auto_covariance_tangent
        ),
        "cholesky_tangent": _comparison(
            manual_chol_tangent[..., 0], auto_chol_tangent
        ),
        "right_solve_tangent": _comparison(
            manual_standardized_tangent[..., 0], auto_standardized_tangent
        ),
    }


def _full_program_autodiff(
    target: GenUTNeuTraTargetAdapter,
) -> list[dict[str, Any]]:
    rows = []
    controls = _controls(target, correction_steps=4)
    for horizon in (1, 2, 20):
        theta = tf.zeros([1, target.parameter_dim], tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(theta)
            value, value_status = batch.batch_finite_value(
                target.filter_adapter,
                theta,
                target.observations[:horizon],
                target.initial_noise,
                target.process_noise[:horizon],
                target.design,
                **controls,
            )
        reverse_gradient = tape.gradient(value[0], theta)
        forward_directions = []
        for parameter in range(target.parameter_dim):
            direction = tf.one_hot(
                parameter, target.parameter_dim, dtype=tf.float32
            )[None, :]
            with tf.autodiff.ForwardAccumulator(
                theta, direction
            ) as accumulator:
                forward_value, _forward_status = batch.batch_finite_value(
                    target.filter_adapter,
                    theta,
                    target.observations[:horizon],
                    target.initial_noise,
                    target.process_noise[:horizon],
                    target.design,
                    **controls,
                )
            forward_directions.append(accumulator.jvp(forward_value)[0])
        forward_score = tf.stack(forward_directions)
        score_value, manual_score, score_status = batch.batch_finite_value_score(
            target.filter_adapter,
            theta,
            target.observations[:horizon],
            target.initial_noise,
            target.process_noise[:horizon],
            target.design,
            **controls,
        )
        rows.append(
            {
                "horizon": horizon,
                "value_identity": _comparison(value, score_value),
                "forward_score_comparison": _comparison(
                    forward_score, manual_score[0]
                ),
                "reverse_score_available": reverse_gradient is not None,
                "reverse_score_comparison": (
                    _comparison(reverse_gradient[0], manual_score[0])
                    if reverse_gradient is not None
                    else None
                ),
                "value_program_valid": bool(
                    value_status["program_valid"][0].numpy()
                ),
                "score_program_valid": bool(
                    score_status["program_valid"][0].numpy()
                ),
            }
        )
    return rows


def _tangent_norm_trace(target: GenUTNeuTraTargetAdapter) -> list[dict[str, Any]]:
    adapter = target.filter_adapter
    theta = tf.zeros([1, target.parameter_dim], tf.float32)
    particles = adapter.initial_value(theta, target.initial_noise)
    particles_tangent = adapter.initial_tangent(theta, target.initial_noise)
    particle_count = tf.shape(particles)[1]
    weights = tf.fill(
        [1, particle_count],
        tf.cast(1.0, tf.float32) / tf.cast(particle_count, tf.float32),
    )
    weights_tangent = tf.zeros(
        [1, particle_count, target.parameter_dim], tf.float32
    )
    controls = _controls(target, correction_steps=4)
    rows = []
    for time_value in range(int(target.observations.shape[0])):
        time_index = tf.constant(time_value, tf.int32)
        noise = target.process_noise[time_value]
        next_particles = adapter.transition_value(
            theta, particles, noise, time_index
        )
        next_tangent = adapter.transition_tangent(
            theta, particles, noise, particles_tangent, time_index
        )
        likelihood = adapter.observation_value(
            theta, next_particles, target.observations[time_value], time_index
        )
        likelihood_tangent = adapter.observation_tangent(
            theta,
            next_particles,
            next_tangent,
            target.observations[time_value],
            time_index,
        )
        log_weights = tf.math.log(weights) + likelihood
        log_weight_tangent = (
            weights_tangent / weights[:, :, None] + likelihood_tangent
        )
        increment = tf.reduce_logsumexp(log_weights, axis=1)
        normalized_weights = tf.exp(log_weights - increment[:, None])
        increment_tangent = tf.reduce_sum(
            normalized_weights[:, :, None] * log_weight_tangent, axis=1
        )
        normalized_weight_tangent = normalized_weights[:, :, None] * (
            log_weight_tangent - increment_tangent[:, None, :]
        )
        restored = batch._restore_cloud_batch_jvp(  # noqa: SLF001
            next_particles,
            normalized_weights,
            next_tangent,
            normalized_weight_tangent,
            target.design,
            epsilon=controls["epsilon"],
            sinkhorn_steps=controls["sinkhorn_steps"],
            balance_steps=controls["balance_steps"],
            ridge=controls["ridge"],
        )
        higher = batch._higher_moment_batch_jvp(  # noqa: SLF001
            next_particles,
            normalized_weights,
            next_tangent,
            normalized_weight_tangent,
            restored["particles"],
            restored["particles_tangent"],
            correction_steps=controls["higher_moment_correction_steps"],
            strength=controls["higher_moment_strength"],
            floor=controls["higher_moment_floor"],
            lm_damping=controls["higher_moment_lm_damping"],
            lm_scale_floor=controls["higher_moment_lm_scale_floor"],
            trust_radius=controls["higher_moment_trust_radius"],
        )
        tangent_rms = tf.sqrt(
            tf.reduce_mean(tf.square(higher["particles_tangent"]), axis=[1, 2])
        )[0]
        tangent_max = tf.reduce_max(
            tf.abs(higher["particles_tangent"]), axis=[1, 2]
        )[0]
        rows.append(
            {
                "time": time_value,
                "increment": float(increment[0].numpy()),
                "increment_tangent": _json_value(increment_tangent[0]),
                "particle_tangent_rms_by_parameter": _json_value(tangent_rms),
                "particle_tangent_max_abs_by_parameter": _json_value(tangent_max),
                "program_valid": bool(
                    (
                        restored["valid"][0]
                        & higher["valid"][0]
                        & tf.math.is_finite(increment[0])
                    ).numpy()
                ),
            }
        )
        particles = higher["particles"]
        particles_tangent = higher["particles_tangent"]
        weights = tf.fill(
            [1, particle_count],
            tf.cast(1.0, tf.float32) / tf.cast(particle_count, tf.float32),
        )
        weights_tangent = tf.zeros_like(weights_tangent)
    return rows


def _finite_difference_regression(
    target: GenUTNeuTraTargetAdapter,
) -> dict[str, Any]:
    theta = tf.zeros([1, target.parameter_dim], tf.float32)
    controls = _controls(target, correction_steps=4)
    score_value, score, score_status = batch.batch_finite_value_score(
        target.filter_adapter,
        theta,
        target.observations,
        target.initial_noise,
        target.process_noise,
        target.design,
        **controls,
    )
    rows = []
    row_keys = []
    basis = tf.eye(target.parameter_dim, dtype=tf.float32)
    for step in FD_REGRESSION_STEPS:
        step_tensor = tf.cast(step, tf.float32)
        for parameter in range(target.parameter_dim):
            direction = basis[parameter]
            rows.extend([theta[0] + step_tensor * direction, theta[0] - step_tensor * direction])
            row_keys.append((parameter, float(step)))
    perturbations = tf.stack(rows, axis=0)
    values, statuses = batch.batch_finite_value(
        target.filter_adapter,
        perturbations,
        target.observations,
        target.initial_noise,
        target.process_noise,
        target.design,
        **controls,
    )
    finite_differences: dict[int, list[tuple[float, float]]] = {
        parameter: [] for parameter in range(target.parameter_dim)
    }
    for index, (parameter, step) in enumerate(row_keys):
        plus = values[2 * index]
        minus = values[2 * index + 1]
        derivative = (plus - minus) / tf.cast(2.0 * step, tf.float32)
        finite_differences[parameter].append((step, float(derivative.numpy())))
    parameter_rows = []
    for parameter in range(target.parameter_dim):
        ladder = finite_differences[parameter]
        regression = fit_quadratic_step_regression(
            [row[0] for row in ladder], [row[1] for row in ladder]
        )
        parameter_rows.append(
            {
                "parameter": target.parameter_names[parameter],
                **evaluate_regression_derivative(
                    float(score[0, parameter].numpy()), regression
                ),
            }
        )
    return {
        "score_value": float(score_value[0].numpy()),
        "score": _json_value(score[0]),
        "score_program_valid": bool(score_status["program_valid"][0].numpy()),
        "all_perturbations_valid": bool(
            tf.reduce_all(statuses["program_valid"]).numpy()
        ),
        "parameters": parameter_rows,
        "all_diagnostic_pass": all(
            bool(row["diagnostic_pass"]) for row in parameter_rows
        ),
    }


def _derivative_diagnostic(
    target: GenUTNeuTraTargetAdapter,
    *,
    include_tangent_trace: bool,
    scope: str,
) -> dict[str, Any]:
    result = {
        "local_autodiff": _local_autodiff_checks(target),
        "first_step_filter_block_autodiff": _first_step_filter_block_autodiff(
            target
        ),
        "standardization_primitive_checks": _standardization_primitive_checks(
            target
        ),
    }
    if scope == "full":
        result["full_program_autodiff"] = _full_program_autodiff(target)
        result["finite_difference_regression"] = _finite_difference_regression(target)
    if include_tangent_trace:
        result["tangent_norm_trace"] = _tangent_norm_trace(target)
    return result


def _manifest(
    *, output: Path, device: str, memory_policy: Mapping[str, Any], started: float
) -> dict[str, Any]:
    command = " ".join(subprocess.list2cmdline([item]) for item in os.sys.argv)
    return {
        "schema": "bayesfilter.genut_austria_endpoint_root_cause_result.v1",
        "plan": str(PLAN.relative_to(ROOT)),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "command": command,
        "python": os.sys.executable,
        "tensorflow": tf.__version__,
        "host": platform.node(),
        "device_request": device,
        "physical_gpus": [item.name for item in tf.config.list_physical_devices("GPU")],
        "logical_gpus": [item.name for item in tf.config.list_logical_devices("GPU")],
        "memory_policy": memory_policy,
        "tf32_enabled": tf.config.experimental.tensor_float_32_execution_enabled(),
        "deterministic_ops_env": os.environ.get("TF_DETERMINISTIC_OPS"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
        "dtype": "float32",
        "output": _display_path(output),
        "started_unix": started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--gpu-index", choices=("0", "1"), default="0")
    parser.add_argument(
        "--phase",
        choices=("all", "endpoint", "localize", "solver", "invariants", "derivative"),
        default="all",
    )
    parser.add_argument("--skip-tangent-trace", action="store_true")
    parser.add_argument(
        "--derivative-scope", choices=("blocks", "full"), default="full"
    )
    parser.add_argument(
        "--endpoint-modes",
        nargs="+",
        choices=("eager", "graph", "xla"),
        default=("eager", "graph", "xla"),
    )
    args = parser.parse_args()
    output = args.output.resolve()
    started = time.time()
    if args.device != _EARLY_DEVICE:
        raise RuntimeError("parsed device does not match pre-import device selection")
    memory_policy = _MEMORY_POLICY
    # Stateless RNG streams are device-specific in this TensorFlow build. Build
    # the frozen target on CPU so its hashes do not change with execution lane.
    with tf.device("/CPU:0"):
        target = make_genut_neutra_target("austria_sir", particle_count=1008)
    target_hashes = {
        "observations": _tensor_hash(target.observations),
        "initial_noise": _tensor_hash(target.initial_noise),
        "process_noise": _tensor_hash(target.process_noise),
        "design": _tensor_hash(target.design),
    }
    if target.target_signature != EXPECTED_TARGET_SIGNATURE:
        raise RuntimeError("frozen target signature mismatch")
    if target.adapter_signature() != EXPECTED_ADAPTER_SIGNATURE:
        raise RuntimeError("frozen adapter signature mismatch")
    if target_hashes != EXPECTED_TARGET_HASHES:
        raise RuntimeError("frozen target tensor hash mismatch")
    source_paths = (
        ROOT / "bayesfilter/highdim/cubature_genut_batch_tf.py",
        ROOT / "bayesfilter/highdim/cubature_genut_neutra_targets.py",
        ROOT / "bayesfilter/highdim/cubature_genut_batch_adapters.py",
        ROOT / "bayesfilter/highdim/higher_moment_contract_e.py",
        ROOT / "bayesfilter/highdim/cubature_genut_filter.py",
    )
    payload: dict[str, Any] = _manifest(
        output=output, device=args.device, memory_policy=memory_policy, started=started
    )
    payload.update(
        {
            "source_sha256": {
                str(path.relative_to(ROOT)): _sha256_file(path) for path in source_paths
            },
            "target_signature": target.target_signature,
            "adapter_signature": target.adapter_signature(),
            "target_construction_device": "/CPU:0",
            "execution_device": "/GPU:0" if args.device == "gpu" else "/CPU:0",
            "target_hashes": target_hashes,
            "frozen_identity_guard": "PASS",
            "route_classification": "batch_diagonal_candidate",
            "historical_tuning_eligibility": "stale_ineligible_after_any_repair",
            "results": {},
        }
    )
    _write_json(output, payload | {"status": "RUNNING"})
    try:
        execution_device = "/GPU:0" if args.device == "gpu" else "/CPU:0"
        with tf.device(execution_device):
            if args.phase in ("all", "endpoint"):
                endpoint_results = []
                for mode in args.endpoint_modes:
                    endpoint_results.append(
                        _endpoint(target, horizon=20, correction_steps=4, mode=mode)
                    )
                    endpoint_results.append(
                        _endpoint(target, horizon=1, correction_steps=0, mode=mode)
                    )
                payload["results"]["endpoint"] = endpoint_results
                _write_json(output, payload | {"status": "RUNNING"})
            if args.phase in ("all", "localize"):
                payload["results"]["localization_and_h1"] = _localization_and_h1(target)
                _write_json(output, payload | {"status": "RUNNING"})
            if args.phase in ("all", "solver"):
                payload["results"]["solver"] = _solver_diagnostic(target)
                _write_json(output, payload | {"status": "RUNNING"})
            if args.phase in ("all", "invariants"):
                payload["results"]["invariants"] = _invariants(target)
                _write_json(output, payload | {"status": "RUNNING"})
            if args.phase in ("all", "derivative"):
                payload["results"]["derivative"] = _derivative_diagnostic(
                    target,
                    include_tangent_trace=not args.skip_tangent_trace,
                    scope=args.derivative_scope,
                )
                _write_json(output, payload | {"status": "RUNNING"})
        payload["status"] = "COMPLETE"
    except Exception as exc:
        payload["status"] = "FAILED"
        payload["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        payload["wall_seconds"] = time.time() - started
        _write_json(output, payload)


if __name__ == "__main__":
    main()
