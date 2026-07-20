"""Phase 2 LGSSM value diagnostic for the Contract E reset.

This script reuses the LGSSM harness in
``tests/test_ledh_pfpf_ot_lgssm_kalman_statistical.py`` and evaluates the same
transition-first log marginal likelihood scalar for three reset arms:

* ``ledh_no_ot`` keeps the weighted LEDH cloud;
* ``old_barycentric_ot`` applies the finite dense Sinkhorn barycentric reset;
* ``contract_e`` uses the same positive finite Sinkhorn first stage, adds a
  centered reparameterized residual, and applies affine moment restoration.

It is a diagnostic for the reviewed Phase 2 evidence contract only.  It does
not certify gradients, nonlinear models, production readiness, HMC readiness,
posterior correctness, or broad Contract E validity.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import importlib.util
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
import contract_e_reset_tf

HARNESS_PATH = ROOT / "tests" / "test_ledh_pfpf_ot_lgssm_kalman_statistical.py"
PHASE2_SUBPLAN = (
    "docs/plans/"
    "bayesfilter-ledh-pfpf-ot-contract-e-phase2-lgssm-value-subplan-2026-06-28.md"
)
SCHEMA_VERSION = "filter_bench.ledh_pfpf_ot_contract_e_lgssm_value.v1"
RESET_ARMS = ("ledh_no_ot", "old_barycentric_ot", "contract_e")


def _parse_setting(value: str) -> dict[str, Any]:
    try:
        epsilon_text, steps_text = value.split(":", maxsplit=1)
        epsilon = float(epsilon_text)
        steps = int(steps_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("settings must be <epsilon>:<steps>") from exc
    if epsilon <= 0.0 or steps <= 0:
        raise argparse.ArgumentTypeError("epsilon and steps must be positive")
    return {"epsilon": epsilon, "steps": steps, "label": f"eps{epsilon:g}_steps{steps}"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--device-scope", choices=("cpu", "visible"), default="cpu")
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--num-particles", type=int, default=64)
    parser.add_argument("--seed-count", type=int, default=10)
    parser.add_argument("--time-steps", type=int, default=10)
    parser.add_argument("--state-dims", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--settings", type=_parse_setting, nargs="+", default=[_parse_setting("0.5:20")])
    parser.add_argument("--rho", type=float, default=1.0)
    parser.add_argument("--tau", type=float, default=1.0e-6)
    parser.add_argument("--spectral-floor", type=float, default=1.0e-6)
    parser.add_argument(
        "--contract-e-reset-factorization",
        choices=("eigh-support", "cholesky-ridge"),
        default="eigh-support",
    )
    parser.add_argument("--chol-ridge-rel", type=float, default=1.0e-5)
    parser.add_argument("--chol-ridge-abs", type=float, default=1.0e-8)
    parser.add_argument("--chol-ridge-escalation", type=float, default=10.0)
    parser.add_argument("--chol-ridge-max-attempts", type=int, default=1)
    parser.add_argument("--covariance-residual-limit", type=float, default=5.0e-4)
    parser.add_argument("--condition-limit", type=float, default=1.0e8)
    parser.add_argument("--tf32-mode", choices=("enabled", "disabled"), default="enabled")
    parser.add_argument("--xla", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gate-mode", choices=("smoke", "material"), default="smoke")
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown-output", default="")
    args = parser.parse_args()
    if args.num_particles <= 2:
        raise ValueError("--num-particles must exceed 2")
    if args.seed_count != 10:
        raise ValueError("this diagnostic reuses harness SEED_COUNT=10")
    if args.time_steps != 10:
        raise ValueError("Phase 2 contract freezes T=10")
    if any(dim not in (1, 2) for dim in args.state_dims):
        raise ValueError("--state-dims supports only 1 and 2")
    if args.rho <= 0.0 or args.rho > 1.0:
        raise ValueError("--rho must be in (0, 1]")
    if args.tau < 0.0:
        raise ValueError("--tau must be nonnegative")
    if args.spectral_floor <= 0.0:
        raise ValueError("--spectral-floor must be positive")
    if args.chol_ridge_rel < 0.0:
        raise ValueError("--chol-ridge-rel must be nonnegative")
    if args.chol_ridge_abs <= 0.0:
        raise ValueError("--chol-ridge-abs must be strictly positive")
    if args.chol_ridge_escalation <= 1.0:
        raise ValueError("--chol-ridge-escalation must exceed one")
    if args.chol_ridge_max_attempts <= 0:
        raise ValueError("--chol-ridge-max-attempts must be positive")
    if args.covariance_residual_limit <= 0.0:
        raise ValueError("--covariance-residual-limit must be positive")
    if args.condition_limit <= 1.0:
        raise ValueError("--condition-limit must exceed one")
    return args


def _configure_import_environment(args: argparse.Namespace) -> None:
    if args.device_scope == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
    os.environ["BAYESFILTER_RUN_LEDHPFPFOT_LGSSM_N1000"] = "1"
    os.environ["BAYESFILTER_LEDHPFPFOT_LGSSM_NUM_PARTICLES"] = str(args.num_particles)
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")


def _load_harness(args: argparse.Namespace) -> Any:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location("ledh_contract_e_lgssm_harness", HARNESS_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load harness from {HARNESS_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.TIME_STEPS = int(args.time_steps)
    module.NUM_PARTICLES = int(args.num_particles)
    module.ROW_CHUNK_SIZE = int(args.num_particles)
    module.COL_CHUNK_SIZE = int(args.num_particles)
    module.PARTICLE_CHUNK_SIZE = int(args.num_particles)
    module.core_ledh.DTYPE = module.DTYPE
    module.annealed_transport_tf.DTYPE = module.DTYPE
    module.tf.config.experimental.enable_tensor_float_32_execution(args.tf32_mode == "enabled")
    return module


def _device_manifest(harness: Any, args: argparse.Namespace) -> dict[str, Any]:
    tf = harness.tf
    physical_gpus = tf.config.list_physical_devices("GPU")
    for gpu in physical_gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    return {
        "device_scope": args.device_scope,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "physical_gpus": [str(device) for device in physical_gpus],
        "logical_gpus": [str(device) for device in tf.config.list_logical_devices("GPU")],
        "tf32_execution_enabled": bool(tf.config.experimental.tensor_float_32_execution_enabled()),
        "xla": bool(args.xla),
    }


def _stateless_residual_normals_batch(harness: Any, state_dim: int) -> Any:
    tf = harness.tf
    seed_indices = tf.range(harness.SEED_COUNT, dtype=tf.int32) + tf.constant(9100, dtype=tf.int32)
    time_offsets = tf.range(harness.TIME_STEPS, dtype=tf.int32) + tf.constant(43, dtype=tf.int32)
    seed_first = tf.repeat(seed_indices, repeats=harness.TIME_STEPS)
    seed_second = tf.tile(time_offsets, [harness.SEED_COUNT])
    seeds = tf.stack([seed_first, seed_second], axis=1)
    draws = tf.map_fn(
        lambda seed: tf.random.stateless_normal(
            [harness.NUM_PARTICLES, state_dim],
            seed=seed,
            dtype=harness.DTYPE,
        ),
        seeds,
        fn_output_signature=tf.TensorSpec([harness.NUM_PARTICLES, state_dim], harness.DTYPE),
    )
    return tf.reshape(
        draws,
        [harness.SEED_COUNT, harness.TIME_STEPS, harness.NUM_PARTICLES, state_dim],
    )


def _mean_sd_mcse(samples: Any) -> tuple[float, float, float]:
    count = samples.shape[0]
    mean = float(samples.mean())
    sd = float(samples.std(ddof=1)) if count > 1 else 0.0
    mcse = sd / math.sqrt(count) if count > 0 else float("nan")
    return mean, sd, mcse


def _finite_or_none(value: float) -> float | None:
    return value if math.isfinite(value) else None


def _nan_reduce_or_none(np: Any, values: Any, *, reduction: str) -> float | None:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return None
    if reduction == "max":
        return float(np.max(finite))
    if reduction == "min":
        return float(np.min(finite))
    raise ValueError(f"unknown reduction: {reduction}")


def _make_compiled_arm(
    harness: Any,
    state_dim: int,
    arm: str,
    setting: dict[str, Any],
    args: argparse.Namespace,
) -> Any:
    tf = harness.tf
    core = harness.core_ledh
    annealed = harness.annealed_transport_tf
    dtype = harness.DTYPE
    batch_size = harness.SEED_COUNT
    num_particles = harness.NUM_PARTICLES
    time_steps = harness.TIME_STEPS
    rho = tf.constant(float(args.rho), dtype=dtype)
    tau = tf.constant(float(args.tau), dtype=dtype)
    spectral_floor = tf.constant(float(args.spectral_floor), dtype=dtype)
    chol_ridge_rel = tf.constant(float(args.chol_ridge_rel), dtype=dtype)
    chol_ridge_abs = tf.constant(float(args.chol_ridge_abs), dtype=dtype)
    chol_ridge_escalation = tf.constant(float(args.chol_ridge_escalation), dtype=dtype)
    chol_ridge_max_attempts = tf.constant(int(args.chol_ridge_max_attempts), dtype=tf.int32)
    nan = tf.constant(float("nan"), dtype=dtype)

    def sym(matrix: Any) -> Any:
        return 0.5 * (matrix + tf.linalg.matrix_transpose(matrix))

    def small_batch_mm(left: Any, right: Any) -> Any:
        return tf.reduce_sum(left[:, :, :, None] * right[:, None, :, :], axis=2)

    def apply_batch_linear_rows(points: Any, matrix: Any) -> Any:
        return tf.reduce_sum(points[:, :, None, :] * matrix[:, None, :, :], axis=-1)

    def weighted_moments(points: Any, weights: Any) -> tuple[Any, Any]:
        mean = tf.reduce_sum(weights[:, :, None] * points, axis=1)
        centered = points - mean[:, None, :]
        # Keep small moment diagnostics off TF32 matmul paths; the Phase 2
        # covariance gate is about the reset algebra, not GEMM rounding.
        covariance = tf.reduce_sum(
            weights[:, :, None, None] * centered[:, :, :, None] * centered[:, :, None, :],
            axis=1,
        )
        return mean, sym(covariance)

    def uniform_moments(points: Any) -> tuple[Any, Any]:
        count = tf.cast(tf.shape(points)[1], dtype)
        weights = tf.fill([tf.shape(points)[0], tf.shape(points)[1]], 1.0 / count)
        return weighted_moments(points, weights)

    def sqrt_psd(matrix: Any, floor: Any) -> Any:
        values, vectors = tf.linalg.eigh(sym(matrix))
        clipped = tf.maximum(values, floor)
        return small_batch_mm(
            vectors * tf.sqrt(clipped)[:, None, :],
            tf.linalg.matrix_transpose(vectors),
        )

    def pinv_sqrt_psd(matrix: Any, floor: Any) -> Any:
        values, vectors = tf.linalg.eigh(sym(matrix))
        inv = tf.where(values > floor, tf.math.rsqrt(tf.maximum(values, floor)), tf.zeros_like(values))
        return small_batch_mm(
            vectors * inv[:, None, :],
            tf.linalg.matrix_transpose(vectors),
        )

    def projector_psd(matrix: Any, floor: Any) -> tuple[Any, Any, Any]:
        values, vectors = tf.linalg.eigh(sym(matrix))
        mask = tf.cast(values > floor, dtype)
        projector = small_batch_mm(
            vectors * mask[:, None, :],
            tf.linalg.matrix_transpose(vectors),
        )
        rank = tf.reduce_sum(tf.cast(values > floor, tf.int32), axis=1)
        return projector, rank, values

    def condition_number(matrix: Any, floor: Any) -> tuple[Any, Any, Any, Any]:
        values, _vectors = tf.linalg.eigh(sym(matrix))
        retained = values > floor
        large = tf.where(retained, values, tf.fill(tf.shape(values), tf.constant(float("inf"), dtype=dtype)))
        small = tf.where(retained, values, tf.zeros_like(values))
        min_positive = tf.reduce_min(large, axis=1)
        max_positive = tf.reduce_max(small, axis=1)
        cond = max_positive / min_positive
        rank = tf.reduce_sum(tf.cast(retained, tf.int32), axis=1)
        return cond, rank, values, min_positive

    def dense_transport_matrix(post_flow: Any, normalized_log_weights: Any) -> Any:
        center = tf.stop_gradient(tf.reduce_mean(post_flow, axis=1, keepdims=True))
        scale = tf.stop_gradient(annealed._filterflow_scale(post_flow))
        scaled_x = (post_flow - center) / scale[:, None, None]
        epsilon = tf.constant(float(setting["epsilon"]), dtype=dtype)
        epsilon0 = tf.stop_gradient(annealed._filterflow_epsilon_start(scaled_x))
        return annealed._filterflow_manual_dense_finite_transport_matrix_value_stopped_scale_keys(
            scaled_x,
            normalized_log_weights,
            epsilon,
            epsilon0,
            tf.constant(harness.ANNEALED_SCALING, dtype=dtype),
            steps=int(setting["steps"]),
        )

    def matrix_residuals(matrix: Any, weights: Any) -> tuple[Any, Any]:
        float_n = tf.cast(tf.shape(matrix)[1], dtype)
        row_mass = tf.reduce_sum(matrix, axis=2)
        column_mass = tf.reduce_sum(matrix, axis=1)
        row_residual = tf.reduce_max(tf.abs(row_mass - 1.0))
        column_residual = tf.reduce_max(tf.abs(column_mass - float_n * weights))
        return row_residual, column_residual

    def contract_e_reset(
        post_flow: Any,
        weights: Any,
        matrix: Any,
        residual_noise: Any,
    ) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any]:
        target_mean, target_cov = weighted_moments(post_flow, weights)
        y_plus = tf.linalg.matmul(matrix, post_flow)
        _plus_mean, plus_cov = uniform_moments(y_plus)
        gap = sym(target_cov - plus_cov)
        projector, target_rank, target_eigs = projector_psd(target_cov, spectral_floor)
        residual_cov = sym(gap + tau * projector)
        b_matrix = tf.sqrt(rho) * sqrt_psd(residual_cov, tf.constant(0.0, dtype=dtype))
        centered_noise = residual_noise - tf.reduce_mean(residual_noise, axis=1, keepdims=True)
        xi = tf.sqrt(tf.cast(num_particles, dtype) / tf.cast(num_particles - 1, dtype)) * centered_noise
        y_tilde = y_plus + apply_batch_linear_rows(xi, b_matrix)
        tilde_mean, tilde_cov = uniform_moments(y_tilde)
        target_sqrt = sqrt_psd(target_cov, tf.constant(0.0, dtype=dtype))
        tilde_pinv_sqrt = pinv_sqrt_psd(tilde_cov, spectral_floor)
        affine = small_batch_mm(target_sqrt, tilde_pinv_sqrt)
        y_star = target_mean[:, None, :] + apply_batch_linear_rows(
            y_tilde - tilde_mean[:, None, :],
            affine,
        )
        star_mean, star_cov = uniform_moments(y_star)
        cov_norm = tf.norm(target_cov, ord="fro", axis=[-2, -1])
        cov_residual = tf.norm(star_cov - target_cov, ord="fro", axis=[-2, -1]) / tf.maximum(
            cov_norm,
            tf.constant(1.0e-30, dtype=dtype),
        )
        mean_residual = tf.reduce_max(tf.abs(star_mean - target_mean), axis=1)
        tilde_cond, tilde_rank, tilde_eigs, min_positive = condition_number(
            tilde_cov,
            spectral_floor,
        )
        gap_eigs, _vectors = tf.linalg.eigh(gap)
        return (
            y_star,
            tf.reduce_max(cov_residual),
            tf.reduce_max(mean_residual),
            tf.reduce_min(gap_eigs),
            tf.reduce_max(tilde_cond),
            tf.reduce_min(min_positive),
            tf.reduce_min(tf.cast(tilde_rank - target_rank, dtype)),
            tf.reduce_min(target_eigs),
        )

    def contract_e_cholesky_ridge_reset(
        post_flow: Any,
        weights: Any,
        matrix: Any,
        residual_noise: Any,
    ) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any]:
        reset = contract_e_reset_tf.contract_e_cholesky_ridge_reset(
            tf,
            post_flow=post_flow,
            weights=weights,
            matrix=matrix,
            residual_noise=residual_noise,
            rho=rho,
            ridge_rel=chol_ridge_rel,
            ridge_abs=chol_ridge_abs,
            ridge_escalation=chol_ridge_escalation,
            ridge_max_attempts=chol_ridge_max_attempts,
        )
        return (
            reset["particles"],
            reset["max_covariance_relative_residual"],
            reset["max_mean_linf_residual"],
            reset["min_gap_diagnostic"],
            reset["max_condition_proxy"],
            reset["min_tilde_positive_diagnostic"],
            reset["min_rank_margin_diagnostic"],
            reset["min_target_positive_diagnostic"],
        )

    @tf.function(
        input_signature=[
            tf.TensorSpec([batch_size, 3], dtype),
            tf.TensorSpec([batch_size, num_particles, state_dim], dtype),
            tf.TensorSpec([batch_size, time_steps, num_particles, state_dim], dtype),
            tf.TensorSpec([batch_size, time_steps, num_particles, state_dim], dtype),
            tf.TensorSpec([time_steps, state_dim], dtype),
        ],
        jit_compile=bool(args.xla),
        reduce_retracing=True,
    )
    def compiled(
        values: Any,
        initial_particles: Any,
        transition_noise: Any,
        residual_noise: Any,
        observations: Any,
    ) -> dict[str, Any]:
        transition_matrix, transition_covariance, observation_covariance = (
            harness._theta_to_lgssm(values, state_dim)
        )
        transition_std = tf.sqrt(tf.linalg.diag_part(transition_covariance))
        h_jac = tf.tile(
            tf.eye(state_dim, dtype=dtype)[None, None, :, :],
            [batch_size, num_particles, 1, 1],
        )
        particles = initial_particles
        log_weights = core.uniform_log_weights(batch_size, num_particles)
        increments = tf.TensorArray(
            dtype=dtype,
            size=time_steps,
            element_shape=tf.TensorShape([batch_size]),
        )
        cov_ratios = tf.TensorArray(dtype=dtype, size=time_steps, element_shape=tf.TensorShape([]))
        mean_shifts = tf.TensorArray(dtype=dtype, size=time_steps, element_shape=tf.TensorShape([]))
        row_residuals = tf.TensorArray(dtype=dtype, size=time_steps, element_shape=tf.TensorShape([]))
        column_residuals = tf.TensorArray(dtype=dtype, size=time_steps, element_shape=tf.TensorShape([]))
        contract_cov_residuals = tf.TensorArray(dtype=dtype, size=time_steps, element_shape=tf.TensorShape([]))
        contract_mean_residuals = tf.TensorArray(dtype=dtype, size=time_steps, element_shape=tf.TensorShape([]))
        contract_min_gap_eigs = tf.TensorArray(dtype=dtype, size=time_steps, element_shape=tf.TensorShape([]))
        contract_conditions = tf.TensorArray(dtype=dtype, size=time_steps, element_shape=tf.TensorShape([]))
        contract_min_tilde_eigs = tf.TensorArray(dtype=dtype, size=time_steps, element_shape=tf.TensorShape([]))
        contract_rank_margins = tf.TensorArray(dtype=dtype, size=time_steps, element_shape=tf.TensorShape([]))
        contract_ridges = tf.TensorArray(dtype=dtype, size=time_steps, element_shape=tf.TensorShape([]))
        contract_ridge_attempts = tf.TensorArray(
            dtype=tf.int32,
            size=time_steps,
            element_shape=tf.TensorShape([]),
        )
        contract_ridge_failures = tf.TensorArray(
            dtype=tf.bool,
            size=time_steps,
            element_shape=tf.TensorShape([]),
        )

        def cond(time_index: Any, *_loop_vars: Any) -> Any:
            return time_index < tf.constant(time_steps, dtype=tf.int32)

        def body(
            time_index: Any,
            current_particles: Any,
            current_log_weights: Any,
            increment_acc: Any,
            cov_ratio_acc: Any,
            mean_shift_acc: Any,
            row_residual_acc: Any,
            column_residual_acc: Any,
            contract_cov_residual_acc: Any,
            contract_mean_residual_acc: Any,
            contract_min_gap_eig_acc: Any,
            contract_condition_acc: Any,
            contract_min_tilde_eig_acc: Any,
            contract_rank_margin_acc: Any,
            contract_ridge_acc: Any,
            contract_ridge_attempt_acc: Any,
            contract_ridge_failure_acc: Any,
        ) -> tuple[Any, ...]:
            observation = observations[time_index]
            prior_mean = tf.einsum("bnj,bdj->bnd", current_particles, transition_matrix)
            noise = transition_noise[:, time_index, :, :]
            pre_flow = prior_mean + noise * transition_std[:, None, :]
            residual = observation[None, None, :] - pre_flow
            flow, _flow_aux = core._batched_ledh_linearized_flow_with_aux_tf(
                pre_flow_particles=pre_flow,
                prior_means=prior_mean,
                observation_jacobian=h_jac,
                observation_residual=residual,
                transition_covariance=transition_covariance,
                observation_covariance=observation_covariance,
            )
            post_flow = flow.post_flow_particles
            transition_log_density = harness._diag_gaussian_logpdf(
                post_flow - prior_mean,
                transition_covariance,
            )
            observation_log_density = harness._diag_gaussian_logpdf(
                post_flow - observation[None, None, :],
                observation_covariance,
            )
            corrected_log_weights = (
                current_log_weights
                + transition_log_density
                + observation_log_density
                - flow.pre_flow_log_density
                + flow.forward_log_det
            )
            weights, increment = core._normalize_log_weights(corrected_log_weights)
            normalized_log_weights = tf.math.log(tf.maximum(weights, core._log_weight_floor()))
            pre_mean, pre_cov = weighted_moments(post_flow, weights)

            if arm == "ledh_no_ot":
                next_particles = post_flow
                next_log_weights = normalized_log_weights
                post_mean = pre_mean
                post_cov = pre_cov
                row_residual = nan
                column_residual = nan
                contract_cov_residual = nan
                contract_mean_residual = nan
                contract_min_gap_eig = nan
                contract_condition = nan
                contract_min_tilde_eig = nan
                contract_rank_margin = nan
                contract_ridge = nan
                contract_ridge_attempt = tf.constant(0, dtype=tf.int32)
                contract_ridge_failure = tf.constant(False, dtype=tf.bool)
            else:
                matrix = dense_transport_matrix(post_flow, normalized_log_weights)
                row_residual, column_residual = matrix_residuals(matrix, weights)
                if arm == "old_barycentric_ot":
                    next_particles = tf.linalg.matmul(matrix, post_flow)
                    contract_cov_residual = nan
                    contract_mean_residual = nan
                    contract_min_gap_eig = nan
                    contract_condition = nan
                    contract_min_tilde_eig = nan
                    contract_rank_margin = nan
                    contract_ridge = nan
                    contract_ridge_attempt = tf.constant(0, dtype=tf.int32)
                    contract_ridge_failure = tf.constant(False, dtype=tf.bool)
                elif arm == "contract_e":
                    if args.contract_e_reset_factorization == "cholesky-ridge":
                        reset = contract_e_reset_tf.contract_e_cholesky_ridge_reset(
                            tf,
                            post_flow=post_flow,
                            weights=weights,
                            matrix=matrix,
                            residual_noise=residual_noise[:, time_index, :, :],
                            rho=rho,
                            ridge_rel=chol_ridge_rel,
                            ridge_abs=chol_ridge_abs,
                            ridge_escalation=chol_ridge_escalation,
                            ridge_max_attempts=chol_ridge_max_attempts,
                        )
                        next_particles = reset["particles"]
                        contract_cov_residual = reset["max_covariance_relative_residual"]
                        contract_mean_residual = reset["max_mean_linf_residual"]
                        contract_min_gap_eig = reset["min_gap_diagnostic"]
                        contract_condition = reset["max_condition_proxy"]
                        contract_min_tilde_eig = reset["min_tilde_positive_diagnostic"]
                        contract_rank_margin = reset["min_rank_margin_diagnostic"]
                        contract_ridge = reset["max_realized_ridge"]
                        contract_ridge_attempt = reset["ridge_attempts_used"]
                        contract_ridge_failure = reset["ridge_failure"]
                    else:
                        (
                            next_particles,
                            contract_cov_residual,
                            contract_mean_residual,
                            contract_min_gap_eig,
                            contract_condition,
                            contract_min_tilde_eig,
                            contract_rank_margin,
                            _target_min_eig,
                        ) = contract_e_reset(
                            post_flow,
                            weights,
                            matrix,
                            residual_noise[:, time_index, :, :],
                        )
                        contract_ridge = nan
                        contract_ridge_attempt = tf.constant(0, dtype=tf.int32)
                        contract_ridge_failure = tf.constant(False, dtype=tf.bool)
                else:
                    raise ValueError(f"unknown arm: {arm}")
                next_log_weights = core.uniform_log_weights(batch_size, num_particles)
                post_mean, post_cov = uniform_moments(next_particles)

            pre_trace = tf.linalg.trace(pre_cov)
            post_trace = tf.linalg.trace(post_cov)
            cov_ratio = tf.reduce_mean(post_trace / tf.maximum(pre_trace, tf.constant(1.0e-30, dtype)))
            mean_shift = tf.reduce_mean(tf.norm(post_mean - pre_mean, axis=1))
            return (
                time_index + 1,
                next_particles,
                next_log_weights,
                increment_acc.write(time_index, increment),
                cov_ratio_acc.write(time_index, cov_ratio),
                mean_shift_acc.write(time_index, mean_shift),
                row_residual_acc.write(time_index, row_residual),
                column_residual_acc.write(time_index, column_residual),
                contract_cov_residual_acc.write(time_index, contract_cov_residual),
                contract_mean_residual_acc.write(time_index, contract_mean_residual),
                contract_min_gap_eig_acc.write(time_index, contract_min_gap_eig),
                contract_condition_acc.write(time_index, contract_condition),
                contract_min_tilde_eig_acc.write(time_index, contract_min_tilde_eig),
                contract_rank_margin_acc.write(time_index, contract_rank_margin),
                contract_ridge_acc.write(time_index, contract_ridge),
                contract_ridge_attempt_acc.write(time_index, contract_ridge_attempt),
                contract_ridge_failure_acc.write(time_index, contract_ridge_failure),
            )

        (
            _time_index,
            _final_particles,
            _final_log_weights,
            increments,
            cov_ratios,
            mean_shifts,
            row_residuals,
            column_residuals,
            contract_cov_residuals,
            contract_mean_residuals,
            contract_min_gap_eigs,
            contract_conditions,
            contract_min_tilde_eigs,
            contract_rank_margins,
            contract_ridges,
            contract_ridge_attempts,
            contract_ridge_failures,
        ) = tf.while_loop(
            cond,
            body,
            loop_vars=(
                tf.constant(0, dtype=tf.int32),
                particles,
                log_weights,
                increments,
                cov_ratios,
                mean_shifts,
                row_residuals,
                column_residuals,
                contract_cov_residuals,
                contract_mean_residuals,
                contract_min_gap_eigs,
                contract_conditions,
                contract_min_tilde_eigs,
                contract_rank_margins,
                contract_ridges,
                contract_ridge_attempts,
                contract_ridge_failures,
            ),
            parallel_iterations=1,
            maximum_iterations=time_steps,
        )
        return {
            "increments": tf.transpose(increments.stack(), [1, 0]),
            "cov_ratios": cov_ratios.stack(),
            "mean_shifts": mean_shifts.stack(),
            "row_residuals": row_residuals.stack(),
            "column_residuals": column_residuals.stack(),
            "contract_cov_residuals": contract_cov_residuals.stack(),
            "contract_mean_residuals": contract_mean_residuals.stack(),
            "contract_min_gap_eigs": contract_min_gap_eigs.stack(),
            "contract_conditions": contract_conditions.stack(),
            "contract_min_tilde_eigs": contract_min_tilde_eigs.stack(),
            "contract_rank_margins": contract_rank_margins.stack(),
            "contract_ridges": contract_ridges.stack(),
            "contract_ridge_attempts": contract_ridge_attempts.stack(),
            "contract_ridge_failures": contract_ridge_failures.stack(),
        }

    return compiled


def _run_arm(
    harness: Any,
    state_dim: int,
    setting: dict[str, Any],
    arm: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    tf = harness.tf
    np = harness.np
    observations = harness._observations(state_dim)
    initial_noise, transition_noise = harness._stateless_seeded_normals_batch(state_dim)
    residual_noise = _stateless_residual_normals_batch(harness, state_dim)
    initial_particles = tf.sqrt(tf.constant(0.7, harness.DTYPE)) * initial_noise
    theta_batch = tf.tile(harness.THETA[None, :], [harness.SEED_COUNT, 1])
    compiled = _make_compiled_arm(harness, state_dim, arm, setting, args)
    raw = compiled(theta_batch, initial_particles, transition_noise, residual_noise, observations)
    increments = raw["increments"].numpy().astype(np.float64)
    totals = np.cumsum(increments, axis=1)[:, -1]
    mean, sd, mcse = _mean_sd_mcse(totals)
    kalman_value, _kalman_score = harness._kalman_value_and_score(state_dim)
    kalman_total = float(kalman_value.numpy()[0])
    delta = mean - kalman_total
    return {
        "state_dim": int(state_dim),
        "setting": dict(setting),
        "arm": arm,
        "value": {
            "mean": mean,
            "sd": sd,
            "mcse": mcse,
            "kalman": kalman_total,
            "delta": delta,
            "abs_delta": abs(delta),
            "abs_z_mcse": None if mcse <= 0.0 else abs(delta) / mcse,
            "seed_values": [float(x) for x in totals],
        },
        "moments": {
            "mean_post_pre_cov_trace_ratio_t0": float(raw["cov_ratios"].numpy()[0]),
            "mean_post_pre_cov_trace_ratio_mean_time": float(raw["cov_ratios"].numpy().mean()),
            "max_post_pre_mean_shift": float(raw["mean_shifts"].numpy().max()),
        },
        "transport": {
            "old_barycentric_mapping": (
                "annealed_transport_tf."
                "_filterflow_manual_dense_finite_transport_matrix_value_stopped_scale_keys "
                "followed by tf.linalg.matmul(matrix, post_flow)"
            )
            if arm in ("old_barycentric_ot", "contract_e")
            else None,
            "max_row_residual": _nan_reduce_or_none(np, raw["row_residuals"].numpy(), reduction="max"),
            "max_column_residual": _nan_reduce_or_none(
                np,
                raw["column_residuals"].numpy(),
                reduction="max",
            ),
        },
        "contract_e": {
            "rho": float(args.rho),
            "tau": float(args.tau),
            "spectral_floor": float(args.spectral_floor),
            "residual_noise_seed_schedule": (
                "for seed in 9100..9109 and time t=0..9 use stateless seed [seed, 43+t]"
            ),
            "reset_factorization": args.contract_e_reset_factorization,
            "gap_diagnostic_label": (
                "min residual Cholesky diagonal of sym(target_cov - plus_cov)+lambda I"
                if args.contract_e_reset_factorization == "cholesky-ridge"
                else "minimum eigenvalue of sym(target_cov - plus_cov)"
            ),
            "tilde_positive_diagnostic_label": (
                "min Cholesky diagonal of tilde_cov+lambda I"
                if args.contract_e_reset_factorization == "cholesky-ridge"
                else "minimum retained positive eigenvalue of tilde_cov"
            ),
            "rank_margin_diagnostic_label": (
                "legacy field: repeats min residual Cholesky diagonal under cholesky-ridge"
                if args.contract_e_reset_factorization == "cholesky-ridge"
                else "tilde rank minus target rank on the retained eigensupport"
            ),
            "chol_ridge_rel": float(args.chol_ridge_rel),
            "chol_ridge_abs": float(args.chol_ridge_abs),
            "chol_ridge_escalation": float(args.chol_ridge_escalation),
            "chol_ridge_max_attempts": int(args.chol_ridge_max_attempts),
            "max_covariance_relative_residual": _nan_reduce_or_none(
                np,
                raw["contract_cov_residuals"].numpy(),
                reduction="max",
            ),
            "max_mean_linf_residual": _nan_reduce_or_none(
                np,
                raw["contract_mean_residuals"].numpy(),
                reduction="max",
            ),
            "min_gap_eig": _nan_reduce_or_none(
                np,
                raw["contract_min_gap_eigs"].numpy(),
                reduction="min",
            ),
            "max_tilde_condition": _nan_reduce_or_none(
                np,
                raw["contract_conditions"].numpy(),
                reduction="max",
            ),
            "min_tilde_positive_eig": _nan_reduce_or_none(
                np,
                raw["contract_min_tilde_eigs"].numpy(),
                reduction="min",
            ),
            "min_rank_margin": _nan_reduce_or_none(
                np,
                raw["contract_rank_margins"].numpy(),
                reduction="min",
            ),
            "max_realized_ridge": _nan_reduce_or_none(
                np,
                raw["contract_ridges"].numpy(),
                reduction="max",
            ),
            "max_ridge_attempts_used": int(np.max(raw["contract_ridge_attempts"].numpy())),
            "any_ridge_failure": bool(np.any(raw["contract_ridge_failures"].numpy())),
        },
    }


def _gate_records(records: list[dict[str, Any]], args: argparse.Namespace, device: dict[str, Any]) -> dict[str, Any]:
    grouped: dict[tuple[int, str], dict[str, dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault((record["state_dim"], record["setting"]["label"]), {})[record["arm"]] = record

    fixture_gates = []
    for (state_dim, label), arms in sorted(grouped.items()):
        contract = arms.get("contract_e")
        old = arms.get("old_barycentric_ot")
        no_ot = arms.get("ledh_no_ot")
        if contract is None or old is None or no_ot is None:
            fixture_gates.append(
                {
                    "state_dim": state_dim,
                    "setting": label,
                    "status": "fail",
                    "reason": "missing required reset arm",
                }
            )
            continue
        contract_mcse = contract["value"]["mcse"]
        within_kalman_2mcse = (
            math.isfinite(contract_mcse)
            and contract_mcse > 0.0
            and contract["value"]["abs_delta"] <= 2.0 * contract_mcse
        )
        improves_old = contract["value"]["abs_delta"] < old["value"]["abs_delta"]
        cov_residual = contract["contract_e"]["max_covariance_relative_residual"]
        condition = contract["contract_e"]["max_tilde_condition"]
        finite_values = all(
            math.isfinite(arm["value"]["mean"])
            and math.isfinite(arm["value"]["sd"])
            and math.isfinite(arm["value"]["mcse"])
            for arm in (contract, old, no_ot)
        )
        arms_distinguishable_metadata = (
            old["arm"] == "old_barycentric_ot"
            and contract["arm"] == "contract_e"
            and no_ot["arm"] == "ledh_no_ot"
            and old["transport"]["old_barycentric_mapping"] is not None
            and contract["transport"]["old_barycentric_mapping"] is not None
            and contract["contract_e"]["max_covariance_relative_residual"] is not None
            and old["contract_e"]["max_covariance_relative_residual"] is None
        )
        covariance_ok = cov_residual is not None and cov_residual <= args.covariance_residual_limit
        condition_ok = condition is not None and condition <= args.condition_limit
        status = (
            "pass"
            if (
                within_kalman_2mcse
                and improves_old
                and finite_values
                and covariance_ok
                and condition_ok
            )
            else "fail"
        )
        fixture_gates.append(
            {
                "state_dim": state_dim,
                "setting": label,
                "status": status,
                "within_kalman_2mcse": within_kalman_2mcse,
                "improves_old_barycentric_abs_delta": improves_old,
                "finite_values": finite_values,
                "arms_distinguishable_metadata": arms_distinguishable_metadata,
                "covariance_restoration_ok": covariance_ok,
                "conditioning_ok": condition_ok,
                "contract_abs_delta": contract["value"]["abs_delta"],
                "old_abs_delta": old["value"]["abs_delta"],
                "contract_mcse": contract_mcse,
                "contract_covariance_residual": cov_residual,
                "contract_condition": condition,
            }
        )
    gpu_claim_ok = True
    if args.device_scope == "visible" and args.gate_mode == "material":
        gpu_claim_ok = bool(device["logical_gpus"])
    status = "passed" if fixture_gates and all(g["status"] == "pass" for g in fixture_gates) and gpu_claim_ok else "failed"
    if args.gate_mode == "smoke":
        status = "smoke_passed" if all(g.get("finite_values", True) for g in fixture_gates) else "smoke_failed"
    return {
        "status": status,
        "fixture_gates": fixture_gates,
        "gpu_claim_ok": gpu_claim_ok,
        "primary_criterion": (
            "material Contract E mean within 2 MCSE of exact Kalman and smaller "
            "absolute Kalman-value error than old_barycentric_ot on every fixture"
        ),
    }


def _render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Contract E LGSSM Value Diagnostic",
        "",
        f"Date: {payload['timestamp_utc']}",
        "",
        f"Status: `{payload['gate']['status']}`",
        "",
        "## Manifest",
        "",
        f"- gate_mode: `{payload['manifest']['gate_mode']}`",
        f"- num_particles: `{payload['manifest']['num_particles']}`",
        f"- seed_count: `{payload['manifest']['seed_count']}`",
        f"- time_steps: `{payload['manifest']['time_steps']}`",
        f"- state_dims: `{payload['manifest']['state_dims']}`",
        f"- settings: `{payload['manifest']['settings']}`",
        f"- device_scope: `{payload['device']['device_scope']}`",
        f"- logical_gpus: `{payload['device']['logical_gpus']}`",
        f"- xla: `{payload['device']['xla']}`",
        f"- tf32_execution_enabled: `{payload['device']['tf32_execution_enabled']}`",
        "",
        "## Value Table",
        "",
        "| dim | setting | arm | mean | Kalman | delta | sd | mcse | abs z MCSE | cov residual | condition |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for record in payload["records"]:
        value = record["value"]
        contract = record["contract_e"]
        abs_z = "NA" if value["abs_z_mcse"] is None else f"{value['abs_z_mcse']:.3f}"
        cov_residual = (
            "NA"
            if contract["max_covariance_relative_residual"] is None
            else f"{contract['max_covariance_relative_residual']:.3e}"
        )
        condition = (
            "NA"
            if contract["max_tilde_condition"] is None
            else f"{contract['max_tilde_condition']:.3e}"
        )
        lines.append(
            "| "
            f"{record['state_dim']} | "
            f"{record['setting']['label']} | "
            f"`{record['arm']}` | "
            f"{value['mean']:.6f} | "
            f"{value['kalman']:.6f} | "
            f"{value['delta']:.6f} | "
            f"{value['sd']:.6f} | "
            f"{value['mcse']:.6f} | "
            f"{abs_z} | "
            f"{cov_residual} | "
            f"{condition} |"
        )
    lines.extend(
        [
            "",
            "## Gate",
            "",
            "```json",
            json.dumps(payload["gate"], indent=2, sort_keys=True),
            "```",
            "",
            "## Nonclaims",
            "",
            "- This diagnostic does not certify gradients.",
            "- This diagnostic does not certify SIR/SV/nonlinear correctness.",
            "- This diagnostic does not certify production readiness, HMC readiness, or posterior correctness.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    raise RuntimeError("ARCHIVAL_WRONG_TRANSPORT_CHUNK_POLICY: this route is preserved only as provenance and cannot emit new evidence")
    args = _parse_args()
    _configure_import_environment(args)
    start = time.perf_counter()
    harness = _load_harness(args)
    device = _device_manifest(harness, args)
    records: list[dict[str, Any]] = []
    for state_dim in args.state_dims:
        for setting in args.settings:
            for arm in RESET_ARMS:
                records.append(_run_arm(harness, int(state_dim), setting, arm, args))
    gate = _gate_records(records, args, device)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "timestamp_utc": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
        "host": platform.node(),
        "python_version": platform.python_version(),
        "tensorflow_version": harness.tf.__version__,
        "phase2_subplan": PHASE2_SUBPLAN,
        "manifest": {
            "gate_mode": args.gate_mode,
            "num_particles": int(args.num_particles),
            "seed_count": int(args.seed_count),
            "time_steps": int(args.time_steps),
            "state_dims": [int(x) for x in args.state_dims],
            "settings": args.settings,
            "theta": [float(x) for x in harness.THETA.numpy()],
            "initial_transition_seed_schedule": (
                "seed indices 9100..9109; initial seeds [seed,17]; transition seeds [seed,29]"
            ),
            "contract_e_residual_seed_schedule": (
                "seed indices 9100..9109; residual seeds [seed,43+t] for t=0..9"
            ),
            "rho": float(args.rho),
            "tau": float(args.tau),
            "spectral_floor": float(args.spectral_floor),
            "contract_e_reset_factorization": args.contract_e_reset_factorization,
            "contract_e_diagnostic_labels": {
                "gap": (
                    "min residual Cholesky diagonal of sym(target_cov - plus_cov)+lambda I"
                    if args.contract_e_reset_factorization == "cholesky-ridge"
                    else "minimum eigenvalue of sym(target_cov - plus_cov)"
                ),
                "tilde_positive": (
                    "min Cholesky diagonal of tilde_cov+lambda I"
                    if args.contract_e_reset_factorization == "cholesky-ridge"
                    else "minimum retained positive eigenvalue of tilde_cov"
                ),
                "rank_margin": (
                    "legacy field: repeats min residual Cholesky diagonal under cholesky-ridge"
                    if args.contract_e_reset_factorization == "cholesky-ridge"
                    else "tilde rank minus target rank on the retained eigensupport"
                ),
            },
            "chol_ridge_rel": float(args.chol_ridge_rel),
            "chol_ridge_abs": float(args.chol_ridge_abs),
            "chol_ridge_escalation": float(args.chol_ridge_escalation),
            "chol_ridge_max_attempts": int(args.chol_ridge_max_attempts),
            "covariance_residual_limit": float(args.covariance_residual_limit),
            "condition_limit": float(args.condition_limit),
        },
        "device": device,
        "records": records,
        "gate": gate,
        "elapsed_seconds": time.perf_counter() - start,
        "nonclaims": [
            "not gradient correctness",
            "not SIR/SV/nonlinear correctness",
            "not production readiness",
            "not HMC readiness",
            "not posterior correctness",
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown_output:
        markdown_output = Path(args.markdown_output)
        markdown_output.parent.mkdir(parents=True, exist_ok=True)
        markdown_output.write_text(_render_markdown(payload), encoding="utf-8")
    print(json.dumps({"status": gate["status"], "elapsed_seconds": payload["elapsed_seconds"]}, sort_keys=True))
    if args.gate_mode == "material" and gate["status"] != "passed":
        raise SystemExit(1)
    if args.gate_mode == "smoke" and gate["status"] == "smoke_failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
