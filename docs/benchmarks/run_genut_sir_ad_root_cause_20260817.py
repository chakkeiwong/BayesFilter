#!/usr/bin/env python3
"""Same-program AD localization for the Austria-SIR GenUT j0 discrepancy."""

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
from typing import Any

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf

from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth

REQUIRE_GPU = os.environ.get("BAYESFILTER_GENUT_AD_REQUIRE_GPU", "1") != "0"
GPU_MEMORY_POLICY = configure_tensorflow_gpu_memory_growth(
    tf, require_gpu=REQUIRE_GPU
)

from bayesfilter.highdim import cubature_genut_filter as genut_filter
from bayesfilter.highdim import ledh_contract_e_reset_tf as reset
from bayesfilter.highdim.cubature_genut_adapters import (
    parameterized_austria_sir_candidate_adapter,
)
from bayesfilter.highdim.cubature_genut_candidate import cubature_design
from bayesfilter.highdim.higher_moment_contract_e import higher_moment_shape_jvp
from bayesfilter.independent_score import sir_observation_simulator_tf as sir


PLAN = ROOT / "docs/plans/bayesfilter-genut-sir-ad-root-cause-localization-plan-2026-08-17.md"
PRIOR = ROOT / "docs/benchmarks/artifacts/genut-sir-root-cause-hypotheses-20260817/attempt05/result.json"
EXPECTED_OBSERVATION_HASH = "cd794ad6e90a74f7cf6dc06b33550bff4bef6fbf66bb0917846d0691b5910f07"
PARAMETER_COUNT = 3
STATE_DIMENSION = 18
SEED = 98201


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tensor_sha(value: tf.Tensor) -> str:
    return hashlib.sha256(tf.io.serialize_tensor(value).numpy()).hexdigest()


def _safe(value: Any) -> Any:
    if hasattr(value, "numpy"):
        value = value.numpy()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _metrics(manual: tf.Tensor, reference: tf.Tensor) -> dict[str, Any]:
    manual = tf.convert_to_tensor(manual)
    reference = tf.cast(tf.convert_to_tensor(reference), manual.dtype)
    difference = manual - reference
    flat_manual = tf.reshape(manual, [-1])
    flat_reference = tf.reshape(reference, [-1])
    flat_difference = tf.reshape(difference, [-1])
    manual_norm = tf.linalg.norm(flat_manual)
    reference_norm = tf.linalg.norm(flat_reference)
    difference_norm = tf.linalg.norm(flat_difference)
    count = tf.cast(tf.size(flat_difference), manual.dtype)
    tiny = tf.cast(1.0e-30 if manual.dtype == tf.float64 else 1.0e-20, manual.dtype)
    denominator = manual_norm + reference_norm + tiny
    cosine_denominator = manual_norm * reference_norm + tiny
    return _safe(
        {
            "dtype": manual.dtype.name,
            "element_count": int(tf.size(flat_difference).numpy()),
            "finite": bool(
                tf.reduce_all(tf.math.is_finite(manual)).numpy()
                and tf.reduce_all(tf.math.is_finite(reference)).numpy()
            ),
            "manual_l2": manual_norm,
            "reference_l2": reference_norm,
            "difference_l2": difference_norm,
            "maximum_absolute_error": tf.reduce_max(tf.abs(difference)),
            "rms_error": tf.sqrt(tf.reduce_sum(tf.square(flat_difference)) / count),
            "reference_rms": tf.sqrt(tf.reduce_sum(tf.square(flat_reference)) / count),
            "symmetric_relative_l2": difference_norm / denominator,
            "cosine_similarity": tf.reduce_sum(flat_manual * flat_reference)
            / cosine_denominator,
            "unit_roundoff": tf.constant(
                2.0**-53 if manual.dtype == tf.float64 else 2.0**-24,
                manual.dtype,
            ),
        }
    )


def _controls(kind: str) -> dict[str, float | int]:
    result: dict[str, float | int] = {
        "epsilon": 8.0,
        "sinkhorn_steps": 16,
        "balance_steps": 16,
        "ridge": 1.0e-5,
        "higher_moment_correction_steps": 0,
        "higher_moment_strength": 0.0,
        "higher_moment_floor": 1.0e-5,
        "pairwise_moment_correction_steps": 0,
        "pairwise_moment_strength": 0.0,
        "pairwise_moment_floor": 1.0e-5,
        "pairwise_particle_rms_cap": 0.0,
        "coordinatewise_standardized_cap": 0.0,
        "coordinatewise_standardized_cap_power": 8,
    }
    if kind in ("diagonal", "pairwise", "dual_cap"):
        result.update(
            {
                "higher_moment_correction_steps": 4,
                "higher_moment_strength": 0.2,
            }
        )
    if kind in ("pairwise", "dual_cap"):
        result.update(
            {
                "pairwise_moment_correction_steps": 4,
                "pairwise_moment_strength": 0.02,
            }
        )
    if kind == "dual_cap":
        result.update(
            {
                "pairwise_particle_rms_cap": 2.0,
                "coordinatewise_standardized_cap": 0.98,
            }
        )
    return result


def _shape_controls(kind: str) -> dict[str, float | int]:
    controls = _controls(kind)
    return {
        "correction_steps": controls["higher_moment_correction_steps"],
        "strength": controls["higher_moment_strength"],
        "floor": controls["higher_moment_floor"],
        "pairwise_correction_steps": controls["pairwise_moment_correction_steps"],
        "pairwise_strength": controls["pairwise_moment_strength"],
        "pairwise_floor": controls["pairwise_moment_floor"],
        "pairwise_particle_rms_cap": controls["pairwise_particle_rms_cap"],
        "coordinatewise_standardized_cap": controls[
            "coordinatewise_standardized_cap"
        ],
        "coordinatewise_standardized_cap_power": controls[
            "coordinatewise_standardized_cap_power"
        ],
    }


def _noise(seed: int, particle_count: int, horizon: int) -> tuple[tf.Tensor, tf.Tensor]:
    return (
        tf.random.stateless_normal(
            [particle_count, STATE_DIMENSION], [seed, 101], dtype=tf.float32
        ),
        tf.random.stateless_normal(
            [horizon, particle_count, STATE_DIMENSION],
            [seed, 102],
            dtype=tf.float32,
        ),
    )


def _normalized_weight_tangent(weights: tf.Tensor, raw: tf.Tensor) -> tf.Tensor:
    return weights[:, None] * (
        raw - tf.reduce_sum(weights[:, None] * raw, axis=0, keepdims=True)
    )


def _law_and_callback_audit(adapter: Any) -> dict[str, Any]:
    theta = tf.zeros([PARAMETER_COUNT], tf.float32)
    states = tf.cast(
        sir.INITIAL_MEAN[None, :]
        + tf.random.stateless_normal(
            [32, STATE_DIMENSION], [731, 1], dtype=tf.float64
        ),
        tf.float32,
    )
    process_noise = tf.random.stateless_normal(
        [32, STATE_DIMENSION], [731, 2], dtype=tf.float32
    )
    state_tangent = 0.1 * tf.random.stateless_normal(
        [32, STATE_DIMENSION, PARAMETER_COUNT], [731, 3], dtype=tf.float32
    )
    observation = tf.cast(sir.fixed_observed_path(81120, 1)[0], tf.float32)

    @tf.function(jit_compile=False)
    def callback_kernel(theta_value, state_value, noise_value, tangent_value, obs):
        transition_manual = adapter.transition_tangent(
            theta_value,
            state_value,
            noise_value,
            tangent_value,
            tf.constant(0),
        )
        observation_manual = adapter.observation_tangent(
            theta_value, state_value, tangent_value, obs, tf.constant(0)
        )
        transition_ad = []
        observation_ad = []
        for parameter_index in range(PARAMETER_COUNT):
            direction = tf.one_hot(parameter_index, PARAMETER_COUNT, dtype=tf.float32)
            with tf.autodiff.ForwardAccumulator(
                (theta_value, state_value),
                (direction, tangent_value[:, :, parameter_index]),
            ) as accumulator:
                transition_value = adapter.transition_value(
                    theta_value, state_value, noise_value, tf.constant(0)
                )
            transition_ad.append(accumulator.jvp(transition_value))
            with tf.autodiff.ForwardAccumulator(
                (theta_value, state_value),
                (direction, tangent_value[:, :, parameter_index]),
            ) as accumulator:
                observation_value = adapter.observation_value(
                    theta_value, state_value, obs, tf.constant(0)
                )
            observation_ad.append(accumulator.jvp(observation_value))
        return (
            transition_manual,
            tf.stack(transition_ad, axis=-1),
            observation_manual,
            tf.stack(observation_ad, axis=-1),
        )

    transition_manual, transition_ad, observation_manual, observation_ad = (
        callback_kernel(theta, states, process_noise, state_tangent, observation)
    )
    adapter_mean = adapter.transition_value(
        theta, states, tf.zeros_like(states), tf.constant(0)
    )
    simulator_mean = sir._transition_mean(  # noqa: SLF001 - independent reference audit
        tf.cast(states, tf.float64), sir.BASE_KAPPA, sir.BASE_NU
    )

    clip_initial = tf.random.stateless_normal(
        [8192, STATE_DIMENSION], [81120, 301], dtype=tf.float64
    )
    clip_process = tf.random.stateless_normal(
        [8192, STATE_DIMENSION], [81120, 302], dtype=tf.float64
    )
    clip_states = sir.INITIAL_MEAN[None, :] + clip_initial
    clip_latent = sir._transition_mean(  # noqa: SLF001 - law audit
        clip_states, sir.BASE_KAPPA, sir.BASE_NU
    ) + clip_process
    invalid_susceptible = clip_latent[:, 0::2] < 0.0
    return {
        "transition_total_tangent": _metrics(transition_manual, transition_ad),
        "observation_total_tangent": _metrics(observation_manual, observation_ad),
        "adapter_fp32_simulator_fp64_transition": {
            "maximum_absolute_difference": float(
                tf.reduce_max(
                    tf.abs(tf.cast(adapter_mean, tf.float64) - simulator_mean)
                ).numpy()
            ),
            "simulator_reference_rms": float(
                tf.sqrt(tf.reduce_mean(tf.square(simulator_mean))).numpy()
            ),
        },
        "one_step_susceptible_clipping_probe": {
            "path_count": 8192,
            "coordinate_count": int(tf.size(invalid_susceptible).numpy()),
            "invalid_coordinate_count": int(
                tf.reduce_sum(tf.cast(invalid_susceptible, tf.int32)).numpy()
            ),
            "invalid_path_count": int(
                tf.reduce_sum(
                    tf.cast(tf.reduce_any(invalid_susceptible, axis=1), tf.int32)
                ).numpy()
            ),
            "invalid_path_rate": float(
                tf.reduce_mean(
                    tf.cast(tf.reduce_any(invalid_susceptible, axis=1), tf.float64)
                ).numpy()
            ),
        },
        "law_note": (
            "The adapter and simulator share the half-step fourth RK stage. "
            "The default adapter leaves post-noise susceptible values unprojected; "
            "the simulator clips them before the next transition."
        ),
    }


def _transport_audit() -> dict[str, Any]:
    particle_count = 36
    particles = tf.random.stateless_normal(
        [particle_count, STATE_DIMENSION], [801, 1], dtype=tf.float32
    )
    logits = tf.random.stateless_normal([particle_count], [801, 2], dtype=tf.float32)
    weights = tf.nn.softmax(logits)
    particle_tangent = 0.1 * tf.random.stateless_normal(
        [particle_count, STATE_DIMENSION, PARAMETER_COUNT],
        [801, 3],
        dtype=tf.float32,
    )
    raw_weight_tangent = tf.random.stateless_normal(
        [particle_count, PARAMETER_COUNT], [801, 4], dtype=tf.float32
    )
    weight_tangent = _normalized_weight_tangent(weights, raw_weight_tangent)

    def pure_value(particles_value, weights_value):
        deltas = particles_value[:, None, :] - particles_value[None, :, :]
        cost = tf.reduce_sum(tf.square(deltas), axis=-1)
        cost_scale = tf.maximum(tf.reduce_mean(cost), tf.constant(1.0e-3, tf.float32))
        kernel_value = tf.exp(-cost / (cost_scale * tf.constant(8.0, tf.float32)))
        uniform = tf.fill([particle_count], tf.constant(1.0 / particle_count, tf.float32))
        left = tf.ones_like(uniform)
        right = tf.ones_like(uniform)
        tiny = tf.constant(1.0e-7, tf.float32)
        # Fixed-count unrolling avoids differentiating TensorFlow loop machinery.
        for _ in range(32):
            left = uniform / (tf.linalg.matvec(kernel_value, right) + tiny)
            right = weights_value / (
                tf.linalg.matvec(tf.transpose(kernel_value), left) + tiny
            )
        coupling_value = left[:, None] * kernel_value * right[None, :]
        row_mass = tf.reduce_sum(coupling_value, axis=1)
        particles_output = (coupling_value @ particles_value) / row_mass[:, None]
        return particles_output, coupling_value

    @tf.function(jit_compile=False)
    def kernel(particles_value, weights_value, particles_dot, weights_dot):
        manual = genut_filter._sinkhorn_barycentric_jvp_core(  # noqa: SLF001
            particles_value,
            weights_value,
            particles_dot,
            weights_dot,
            epsilon=8.0,
            sinkhorn_steps=16,
            balance_steps=16,
        )
        particles_ad = []
        coupling_ad = []
        for parameter_index in range(PARAMETER_COUNT):
            with tf.autodiff.ForwardAccumulator(
                (particles_value, weights_value),
                (
                    particles_dot[:, :, parameter_index],
                    weights_dot[:, parameter_index],
                ),
            ) as accumulator:
                forward_particles, forward_coupling = pure_value(
                    particles_value, weights_value
                )
            particles_ad.append(accumulator.jvp(forward_particles))
            coupling_ad.append(accumulator.jvp(forward_coupling))
        return (
            manual["particles_tangent"],
            tf.stack(particles_ad, axis=-1),
            manual["coupling_tangent"],
            tf.stack(coupling_ad, axis=-1),
            manual["minimum_row_mass"],
            manual["post_quotient_column_tv_error"],
            manual["marginal_valid"],
        )

    manual_particles, ad_particles, manual_coupling, ad_coupling, row_mass, tv, valid = (
        kernel(particles, weights, particle_tangent, weight_tangent)
    )
    return {
        "particles": _metrics(manual_particles, ad_particles),
        "coupling": _metrics(manual_coupling, ad_coupling),
        "minimum_row_mass": float(row_mass.numpy()),
        "post_quotient_column_tv_error": float(tv.numpy()),
        "marginal_valid": bool(valid.numpy()),
    }


def _reset_case(
    *, dtype: tf.dtypes.DType, condition_exponent: float, transported_scale: float
) -> dict[str, Any]:
    particle_count = 36
    dimension = 18
    base = tf.random.stateless_normal(
        [1, particle_count, dimension], [901, 1], dtype=dtype
    )
    scales = tf.pow(
        tf.cast(10.0, dtype),
        tf.linspace(tf.cast(0.0, dtype), tf.cast(condition_exponent, dtype), dimension),
    )
    source = base * scales[None, None, :]
    weights = tf.fill([1, particle_count], tf.cast(1.0 / particle_count, dtype))
    transported = tf.cast(transported_scale, dtype) * source
    design = tf.cast(
        cubature_design(dim=dimension, num_particles=particle_count), dtype
    )[None, :, :]
    ridge = tf.fill([1], tf.cast(1.0e-5, dtype))
    source_tangent = 0.01 * tf.random.stateless_normal(
        [1, particle_count, dimension, PARAMETER_COUNT], [901, 2], dtype=dtype
    )
    transported_tangent = 0.01 * tf.random.stateless_normal(
        [1, particle_count, dimension, PARAMETER_COUNT], [901, 3], dtype=dtype
    )
    weight_tangent = tf.zeros([1, particle_count, PARAMETER_COUNT], dtype)
    design_tangent = tf.zeros([1, particle_count, dimension, PARAMETER_COUNT], dtype)
    ridge_tangent = tf.zeros([1, PARAMETER_COUNT], dtype)

    @tf.function(jit_compile=False)
    def kernel(
        source_value,
        weights_value,
        transported_value,
        design_value,
        ridge_value,
        source_dot,
        weights_dot,
        transported_dot,
        design_dot,
        ridge_dot,
    ):
        forward = reset._contract_e_chol_cloud_forward_core(  # noqa: SLF001
            source_value,
            weights_value,
            transported_value,
            design_value,
            ridge_value,
        )
        names = (
            "particles",
            "target_cov",
            "gap_chol",
            "injected_cov",
            "target_chol",
            "injected_chol",
            "affine",
        )
        manual_by_name = {name: [] for name in names}
        ad_by_name = {name: [] for name in names}
        for parameter_index in range(PARAMETER_COUNT):
            manual = reset._contract_e_chol_cloud_jvp_from_forward_core(  # noqa: SLF001
                forward,
                source_value,
                weights_value,
                transported_value,
                design_value,
                ridge_value,
                source_dot[:, :, :, parameter_index],
                weights_dot[:, :, parameter_index],
                transported_dot[:, :, :, parameter_index],
                design_dot[:, :, :, parameter_index],
                ridge_dot[:, parameter_index],
            )
            with tf.autodiff.ForwardAccumulator(
                (
                    source_value,
                    weights_value,
                    transported_value,
                    design_value,
                    ridge_value,
                ),
                (
                    source_dot[:, :, :, parameter_index],
                    weights_dot[:, :, parameter_index],
                    transported_dot[:, :, :, parameter_index],
                    design_dot[:, :, :, parameter_index],
                    ridge_dot[:, parameter_index],
                ),
            ) as accumulator:
                ad_forward = reset._contract_e_chol_cloud_forward_core(  # noqa: SLF001
                    source_value,
                    weights_value,
                    transported_value,
                    design_value,
                    ridge_value,
                )
            for name in names:
                manual_by_name[name].append(manual[name])
                ad_by_name[name].append(accumulator.jvp(ad_forward[name]))
        return (
            {name: tf.stack(manual_by_name[name], axis=-1) for name in names},
            {name: tf.stack(ad_by_name[name], axis=-1) for name in names},
            forward["gap_condition_proxy"],
            forward["target_condition_proxy"],
            forward["injected_condition_proxy"],
            tf.reduce_min(forward["gap_eigenvalues"]),
            forward["finite"],
            forward["factor_diagonal_positive"],
        )

    manual, ad, gap_condition, target_condition, injected_condition, gap_min, finite, positive = kernel(
        source,
        weights,
        transported,
        design,
        ridge,
        source_tangent,
        weight_tangent,
        transported_tangent,
        design_tangent,
        ridge_tangent,
    )
    return {
        "dtype": dtype.name,
        "condition_exponent": condition_exponent,
        "transported_scale": transported_scale,
        "gap_condition_proxy": float(gap_condition[0].numpy()),
        "target_condition_proxy": float(target_condition[0].numpy()),
        "injected_condition_proxy": float(injected_condition[0].numpy()),
        "minimum_gap_eigenvalue": float(gap_min.numpy()),
        "finite": bool(finite[0].numpy()),
        "factor_diagonal_positive": bool(positive[0].numpy()),
        "comparisons": {name: _metrics(manual[name], ad[name]) for name in manual},
    }


def _reset_audit() -> list[dict[str, Any]]:
    cases = []
    for dtype in (tf.float32, tf.float64):
        for condition_exponent, transported_scale in (
            (0.0, 0.5),
            (1.0, 0.5),
            (2.0, 0.5),
            (1.0, 0.95),
        ):
            cases.append(
                _reset_case(
                    dtype=dtype,
                    condition_exponent=condition_exponent,
                    transported_scale=transported_scale,
                )
            )
    return cases


def _shape_audit(kind: str) -> dict[str, Any]:
    particle_count = 36
    source = tf.random.stateless_normal(
        [particle_count, STATE_DIMENSION], [1001, 1], dtype=tf.float32
    )
    logits = tf.random.stateless_normal([particle_count], [1001, 2], dtype=tf.float32)
    weights = tf.nn.softmax(logits)
    points = 0.7 * source + 0.3 * tf.random.stateless_normal(
        [particle_count, STATE_DIMENSION], [1001, 3], dtype=tf.float32
    )
    source_tangent = 0.05 * tf.random.stateless_normal(
        [particle_count, STATE_DIMENSION, PARAMETER_COUNT],
        [1001, 4],
        dtype=tf.float32,
    )
    points_tangent = 0.05 * tf.random.stateless_normal(
        [particle_count, STATE_DIMENSION, PARAMETER_COUNT],
        [1001, 5],
        dtype=tf.float32,
    )
    raw_weight_tangent = tf.random.stateless_normal(
        [particle_count, PARAMETER_COUNT], [1001, 6], dtype=tf.float32
    )
    weight_tangent = _normalized_weight_tangent(weights, raw_weight_tangent)
    controls = _shape_controls(kind)

    @tf.function(jit_compile=False)
    def kernel(source_value, weights_value, source_dot, weights_dot, points_value, points_dot):
        manual = higher_moment_shape_jvp(
            source_value,
            weights_value,
            source_dot,
            weights_dot,
            points_value,
            points_dot,
            **controls,
        )
        ad = []
        for parameter_index in range(PARAMETER_COUNT):
            with tf.autodiff.ForwardAccumulator(
                (source_value, weights_value, points_value),
                (
                    source_dot[:, :, parameter_index],
                    weights_dot[:, parameter_index],
                    points_dot[:, :, parameter_index],
                ),
            ) as accumulator:
                forward = higher_moment_shape_jvp(
                    source_value,
                    weights_value,
                    source_dot,
                    weights_dot,
                    points_value,
                    points_dot,
                    **controls,
                )
            ad.append(accumulator.jvp(forward["particles"]))
        return (
            manual["particles_tangent"],
            tf.stack(ad, axis=-1),
            manual["valid"],
            manual["maximum_diagonal_scaled_system_condition"],
            manual["minimum_coordinatewise_cap_derivative"],
        )

    manual, ad, valid, condition, cap_derivative = kernel(
        source, weights, source_tangent, weight_tangent, points, points_tangent
    )
    return {
        "arm": kind,
        "controls": controls,
        "particles": _metrics(manual, ad),
        "valid": bool(valid.numpy()),
        "maximum_diagonal_scaled_system_condition": float(condition.numpy()),
        "minimum_coordinatewise_cap_derivative": float(cap_derivative.numpy()),
    }


def _full_program_case(
    adapter: Any,
    observations: tf.Tensor,
    *,
    particle_count: int,
    horizon: int,
    arm: str,
) -> dict[str, Any]:
    initial, process = _noise(SEED, particle_count, horizon)
    design = cubature_design(dim=STATE_DIMENSION, num_particles=particle_count)
    theta = tf.zeros([PARAMETER_COUNT], tf.float32)
    controls = _controls(arm)

    @tf.function(jit_compile=True, reduce_retracing=True)
    def xla_kernel(theta_value, obs, initial_value, process_value, design_value):
        value, score, diagnostics = genut_filter.finite_value_score(
            adapter,
            theta_value,
            obs,
            initial_value,
            process_value,
            design_value,
            **controls,
        )
        return (
            value,
            score,
            diagnostics["program_valid"],
            diagnostics["minimum_covariance_gap_eigenvalue"],
            diagnostics["maximum_shape_displacement"],
            diagnostics["maximum_normalized_shape_displacement"],
            diagnostics["score_increments"],
        )

    @tf.function(jit_compile=False, reduce_retracing=True)
    def scalar_graph_ad_kernel(
        theta_value, obs, initial_value, process_value, design_value
    ):
        with tf.GradientTape() as tape:
            tape.watch(theta_value)
            value, score, diagnostics = genut_filter.finite_value_score(
                adapter,
                theta_value,
                obs,
                initial_value,
                process_value,
                design_value,
                **controls,
            )
        return value, score, tape.gradient(value, theta_value), diagnostics["program_valid"]

    started = time.perf_counter()
    value, score, valid, gap, displacement, normalized_displacement, increments = xla_kernel(
        theta,
        observations[:horizon],
        initial,
        process,
        design,
    )
    ad_value, graph_score, ad_score, ad_valid = scalar_graph_ad_kernel(
        theta, observations[:horizon], initial, process, design
    )
    ad_route = "scalar_graph_reverse_ad"
    elapsed = time.perf_counter() - started
    return {
        "particle_count": particle_count,
        "horizon": horizon,
        "arm": arm,
        "value": float(value.numpy()),
        "manual_score": _safe(score),
        "manual_j0": float(score[0].numpy()),
        "ad_j0": float(ad_score[0].numpy()),
        "j0_comparison": _metrics(score[0], ad_score[0]),
        "xla_ad_primal_value_comparison": _metrics(value, ad_value),
        "xla_graph_manual_score_comparison": _metrics(score, graph_score),
        "ad_route": ad_route,
        "program_valid": bool(valid.numpy()) and bool(ad_valid.numpy()),
        "minimum_covariance_gap_eigenvalue": float(gap.numpy()),
        "maximum_shape_displacement": float(displacement.numpy()),
        "maximum_normalized_shape_displacement": float(
            normalized_displacement.numpy()
        ),
        "maximum_absolute_j0_increment": float(
            tf.reduce_max(tf.abs(increments[:, 0])).numpy()
        ),
        "wall_time_seconds": elapsed,
    }


def _full_program_ladder(
    adapter: Any, observations: tf.Tensor, progress_path: Path
) -> list[dict[str, Any]]:
    rows = []
    rungs = (
        (36, 2, ("none", "diagonal", "pairwise", "dual_cap")),
        (36, 5, ("none", "diagonal", "pairwise", "dual_cap")),
        (36, 20, ("none", "diagonal", "dual_cap")),
        (252, 2, ("diagonal", "dual_cap")),
        (252, 5, ("diagonal",)),
        (252, 20, ("diagonal",)),
        (1008, 2, ("none", "diagonal")),
        (1008, 5, ("diagonal",)),
        (1008, 20, ("none", "diagonal")),
    )
    for particle_count, horizon, arms in rungs:
        for arm in arms:
            row = _full_program_case(
                adapter,
                observations,
                particle_count=particle_count,
                horizon=horizon,
                arm=arm,
            )
            rows.append(row)
            progress_path.write_text(
                json.dumps(_safe(rows), indent=2, sort_keys=True) + "\n",
                encoding="ascii",
            )
            print(
                f"full N={particle_count} T={horizon} arm={arm} "
                f"manual={row['manual_j0']:.9g} ad={row['ad_j0']:.9g} "
                f"rel={row['j0_comparison']['symmetric_relative_l2']:.3g}",
                flush=True,
            )
    return rows


def _fd_secondary(
    adapter: Any, observations: tf.Tensor, *, particle_count: int, horizon: int, arm: str
) -> dict[str, Any]:
    initial, process = _noise(SEED, particle_count, horizon)
    design = cubature_design(dim=STATE_DIMENSION, num_particles=particle_count)
    theta = tf.zeros([PARAMETER_COUNT], tf.float32)
    direction = tf.one_hot(0, PARAMETER_COUNT, dtype=tf.float32)
    controls = _controls(arm)

    @tf.function(jit_compile=True, reduce_retracing=True)
    def value_kernel(theta_value):
        return genut_filter.finite_value_score(
            adapter,
            theta_value,
            observations[:horizon],
            initial,
            process,
            design,
            **controls,
        )[:2]

    value, score = value_kernel(theta)
    rows = []
    for step in (0.03, 0.01, 0.003):
        plus, _ = value_kernel(theta + tf.cast(step, tf.float32) * direction)
        minus, _ = value_kernel(theta - tf.cast(step, tf.float32) * direction)
        rows.append(
            {
                "step": step,
                "central_difference": float(((plus - minus) / (2.0 * step)).numpy()),
                "plus": float(plus.numpy()),
                "minus": float(minus.numpy()),
            }
        )
    return {
        "particle_count": particle_count,
        "horizon": horizon,
        "arm": arm,
        "value": float(value.numpy()),
        "manual_j0": float(score[0].numpy()),
        "rows": rows,
        "role": "secondary_precision_diagnostic_only",
    }


def _source_hashes() -> dict[str, str]:
    paths = (
        Path(__file__).resolve(),
        PLAN.resolve(),
        ROOT / "bayesfilter/highdim/cubature_genut_filter.py",
        ROOT / "bayesfilter/highdim/cubature_genut_adapters.py",
        ROOT / "bayesfilter/highdim/higher_moment_contract_e.py",
        ROOT / "bayesfilter/highdim/ledh_contract_e_reset_tf.py",
        ROOT / "bayesfilter/independent_score/sir_observation_simulator_tf.py",
    )
    return {str(path.relative_to(ROOT)): _sha(path) for path in paths}


def _write_markdown(payload: dict[str, Any], output: Path) -> None:
    lines = [
        "# GenUT Austria-SIR AD Root-Cause Localization",
        "",
        f"- Status: `{payload['status']}`",
        f"- Root-cause classification: `{payload['decision']['root_cause_classification']}`",
        f"- Exact SIR score established: `{payload['decision']['exact_sir_score_established']}`",
        "",
        "## Full-Program j0",
        "",
        "| N | T | Arm | Manual | AD | Symmetric relative L2 | Valid |",
        "|---:|---:|---|---:|---:|---:|---|",
    ]
    for row in payload["full_program"]:
        lines.append(
            f"| {row['particle_count']} | {row['horizon']} | {row['arm']} | "
            f"{row['manual_j0']:.8g} | {row['ad_j0']:.8g} | "
            f"{row['j0_comparison']['symmetric_relative_l2']:.3g} | "
            f"{row['program_valid']} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "| Question | Result |",
            "|---|---|",
        ]
    )
    for key, value in payload["decision"].items():
        lines.append(f"| `{key}` | `{value}` |")
    lines.extend(
        [
            "",
            "All rankings remain unsupported. Automatic differentiation checks "
            "the derivative of the finite program, not equality to the exact "
            "observed-data SIR likelihood.",
        ]
    )
    (output / "result.md").write_text("\n".join(lines) + "\n", encoding="ascii")


def run(output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    log_lines = []

    def stage(message: str) -> None:
        elapsed = time.perf_counter() - started
        line = f"[{elapsed:8.3f}s] {message}"
        print(line, flush=True)
        log_lines.append(line)

    checkpoint_payload: dict[str, Any] = {
        "schema": "bayesfilter.genut_sir_ad_root_cause_checkpoint.v1",
        "status": "RUNNING",
        "completed_stages": [],
    }

    def checkpoint(stage_name: str, key: str, value: Any) -> None:
        checkpoint_payload["completed_stages"].append(stage_name)
        checkpoint_payload[key] = _safe(value)
        checkpoint_payload["elapsed_seconds"] = time.perf_counter() - started
        (output / "checkpoint.json").write_text(
            json.dumps(_safe(checkpoint_payload), indent=2, sort_keys=True) + "\n",
            encoding="ascii",
        )

    tf.config.set_soft_device_placement(False)
    physical = tf.config.list_physical_devices("GPU")
    logical = tf.config.list_logical_devices("GPU")
    if not logical:
        raise RuntimeError("GPU required")
    adapter = parameterized_austria_sir_candidate_adapter(latent_preclip=False)
    with tf.device("/CPU:0"):
        observations = sir.fixed_observed_path(81120, 20)
    observation_hash = _tensor_sha(observations)
    if observation_hash != EXPECTED_OBSERVATION_HASH:
        raise RuntimeError("fixed observation hash mismatch")
    observations = tf.cast(observations, tf.float32)

    stage("law and callback AD audit")
    law = _law_and_callback_audit(adapter)
    checkpoint("law_and_callbacks", "law_and_callbacks", law)
    stage("Sinkhorn row-quotient AD audit")
    transport = _transport_audit()
    checkpoint("transport", "transport", transport)
    stage("Contract-E FP32/FP64 condition ladder")
    reset_rows = _reset_audit()
    checkpoint("reset_condition_ladder", "reset_condition_ladder", reset_rows)
    stage("higher-moment and cap AD audits")
    shape_rows = [_shape_audit(arm) for arm in ("none", "diagonal", "pairwise", "dual_cap")]
    checkpoint("shape_maps", "shape_maps", shape_rows)
    stage("full finite-program AD ladder")
    full_rows = _full_program_ladder(
        adapter, observations, output / "full_program_checkpoint.json"
    )
    checkpoint("full_program", "full_program", full_rows)
    stage("secondary FP32 finite-difference checks")
    fd_rows = [
        _fd_secondary(adapter, observations, particle_count=36, horizon=2, arm="diagonal"),
        _fd_secondary(adapter, observations, particle_count=1008, horizon=20, arm="diagonal"),
    ]
    checkpoint("finite_difference_secondary", "finite_difference_secondary", fd_rows)

    comparisons = [
        law["transition_total_tangent"],
        law["observation_total_tangent"],
        transport["particles"],
        transport["coupling"],
    ]
    comparisons.extend(
        row["comparisons"][name]
        for row in reset_rows
        for name in row["comparisons"]
    )
    comparisons.extend(row["particles"] for row in shape_rows)
    local_max_relative = max(
        float(row["symmetric_relative_l2"]) for row in comparisons
    )
    full_max_relative = max(
        float(row["j0_comparison"]["symmetric_relative_l2"]) for row in full_rows
    )
    primal_max_relative = max(
        float(row["xla_ad_primal_value_comparison"]["symmetric_relative_l2"])
        for row in full_rows
    )
    graph_manual_max_relative = max(
        (
            float(row["xla_graph_manual_score_comparison"]["symmetric_relative_l2"])
            for row in full_rows
            if row["xla_graph_manual_score_comparison"] is not None
        ),
        default=0.0,
    )
    invalid_full_rows = sum(not row["program_valid"] for row in full_rows)
    nonfinite_comparisons = sum(not row["finite"] for row in comparisons)

    # These are descriptive classifications. Raw scale-normalized errors and
    # the precision ladder, not this label, carry the numerical evidence.
    if nonfinite_comparisons or invalid_full_rows:
        root_classification = "diagnostic_or_finite_program_invalid"
    elif primal_max_relative > 1.0e-4 or graph_manual_max_relative > 1.0e-4:
        root_classification = "ad_fallback_primal_or_manual_route_mismatch"
    elif full_max_relative > 1.0e-2:
        root_classification = "manual_full_program_jvp_mismatch"
    elif local_max_relative > 1.0e-2:
        root_classification = "manual_local_map_jvp_mismatch"
    else:
        root_classification = "manual_jvp_consistent_variance_or_finite_program_bias_remains"

    prior_summary = None
    if PRIOR.exists():
        prior = json.loads(PRIOR.read_text(encoding="utf-8"))
        prior_summary = {
            arm: value.get("summary_j0")
            for arm, value in prior.get("arms", {}).items()
        }

    wall_time = time.perf_counter() - started
    payload = {
        "schema": "bayesfilter.genut_sir_ad_root_cause_localization.v1",
        "status": "COMPLETE",
        "plan": str(PLAN.relative_to(ROOT)),
        "target": {
            "model": "Austria-SIR",
            "parameter": "j0_log_kappa_scale",
            "theta": [0.0, 0.0, 0.0],
            "observation_hash": observation_hash,
            "finite_program_target_only": True,
        },
        "law_and_callbacks": law,
        "transport": transport,
        "reset_condition_ladder": reset_rows,
        "shape_maps": shape_rows,
        "full_program": full_rows,
        "finite_difference_secondary": fd_rows,
        "prior_three_seed_descriptive_summary": prior_summary,
        "decision": {
            "root_cause_classification": root_classification,
            "local_max_symmetric_relative_l2": local_max_relative,
            "full_max_symmetric_relative_l2": full_max_relative,
            "fallback_primal_max_symmetric_relative_l2": primal_max_relative,
            "graph_xla_manual_score_max_symmetric_relative_l2": graph_manual_max_relative,
            "invalid_full_program_rows": invalid_full_rows,
            "nonfinite_local_comparisons": nonfinite_comparisons,
            "exact_sir_score_established": False,
            "algorithm_ranking_supported": False,
            "default_or_hmc_readiness": False,
        },
        "decision_table": [
            {
                "decision": "locate first same-program derivative mismatch",
                "primary_criterion_status": root_classification,
                "veto_diagnostic_status": {
                    "invalid_full_rows": invalid_full_rows,
                    "nonfinite_local_comparisons": nonfinite_comparisons,
                },
                "main_uncertainty": (
                    "AD consistency cannot establish equality between the finite "
                    "GenUT likelihood approximation and the exact observed-data likelihood"
                ),
                "next_justified_action": (
                    "repair the first mismatching local map if present; otherwise "
                    "treat j0 spread as variance/finite-program-bias evidence and "
                    "seek an admitted independent oracle"
                ),
                "not_concluded": "exact SIR score or method superiority",
            }
        ],
        "inference_status": {
            "hard_veto_screen": (
                "passed" if not invalid_full_rows and not nonfinite_comparisons else "failed"
            ),
            "statistically_supported_ranking": "none",
            "descriptive_only_differences": "three-seed score SD and arm means",
            "default_readiness": "not supported",
            "next_evidence_needed": (
                "a validated independent observed-data score reference and replicated "
                "uncertainty analysis after any derivative repair"
            ),
        },
        "post_run_red_team": {
            "strongest_alternative_explanation": (
                "The manual derivative can be internally exact while the finite GenUT "
                "value remains a biased or high-variance approximation to the true likelihood"
            ),
            "overturning_result": (
                "A reproducible, scale-material manual/AD mismatch that persists in "
                "the relevant precision and condition regime"
            ),
            "weakest_evidence": (
                "Only one seed is used for AD localization and three prior seeds for variance"
            ),
        },
        "manifest": {
            "command": sys.argv,
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "environment": "/home/chakwong/anaconda3/envs/tftwogpu",
            "python": sys.executable,
            "tensorflow": tf.__version__,
            "host": platform.node(),
            "physical_gpus": [
                {
                    "device": device.name,
                    "details": tf.config.experimental.get_device_details(device),
                }
                for device in physical
            ],
            "logical_gpus": [device.name for device in logical],
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "memory_policy": dict(GPU_MEMORY_POLICY),
            "dtype": "FP32 full program; FP32/FP64 local reset",
            "tf32": bool(tf.config.experimental.tensor_float_32_execution_enabled()),
            "jit_compile": True,
            "seed": SEED,
            "wall_time_seconds": wall_time,
            "output": str(output.relative_to(ROOT)),
            "source_hashes": _source_hashes(),
            "trust_basis": "owner_designated_managed_session_visible_gpu_trusted",
        },
        "upstream_vetoes": {
            "classifier_gaussian_exact_oracle": "FAILED_8_OF_9_CELLS",
            "genut_lgssm_kalman_oracle": "FAILED",
        },
        "nonclaims": [
            "no exact observed-data SIR score",
            "no classifier oracle admission",
            "no SQMC or GenUT ranking",
            "no default readiness",
            "no HMC readiness",
        ],
    }
    (output / "result.json").write_text(
        json.dumps(_safe(payload), indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    _write_markdown(payload, output)
    stage("artifact complete")
    (output / "run.log").write_text("\n".join(log_lines) + "\n", encoding="ascii")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args.output.resolve())


if __name__ == "__main__":
    main()
