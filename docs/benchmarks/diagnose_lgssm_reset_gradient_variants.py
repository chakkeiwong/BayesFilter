"""Causally localize the LGSSM q-scale gradient error across reset variants.

This CPU/float64 diagnostic reuses the fixed-noise LGSSM scalar recursion and
changes only the state reset after weight normalization. It is not a production
transport implementation or leaderboard-admission route.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf

from bayesfilter.linear.kalman_tf import tf_kalman_log_likelihood
from docs.benchmarks import (
    benchmark_ledh_same_target_lgssm_m3_t50_compact_score_adapter as adapter,
)
from docs.benchmarks import benchmark_ledh_same_target_lgssm_m3_t50_value as lgssm
from experiments.dpf_implementation.tf_tfp.resampling import annealed_transport_tf


VARIANTS = ("no_reset", "current", "row_normalized", "moment_restored")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--num-particles", type=int, default=32)
    parser.add_argument("--time-steps", type=int, default=50)
    parser.add_argument("--seeds", type=int, nargs="+", default=(81120,))
    parser.add_argument("--variants", choices=VARIANTS, nargs="+", default=VARIANTS)
    parser.add_argument("--sinkhorn-iterations", type=int, default=10)
    parser.add_argument("--sinkhorn-epsilon", type=float, default=0.5)
    parser.add_argument("--fd-step", type=float, default=1.0e-5)
    parser.add_argument("--moment-jitter", type=float, default=1.0e-9)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.num_particles <= 2 or args.time_steps <= 0:
        raise ValueError("num_particles must exceed two and time_steps must be positive")
    if args.fd_step <= 0.0 or args.moment_jitter <= 0.0:
        raise ValueError("fd_step and moment_jitter must be positive")
    return args


def _module_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        batch_seeds=list(args.seeds),
        num_particles=args.num_particles,
        time_steps=args.time_steps,
        transport_policy="active-all",
        sinkhorn_iterations=args.sinkhorn_iterations,
        sinkhorn_epsilon=args.sinkhorn_epsilon,
        annealed_scaling=0.9,
        annealed_convergence_threshold=1.0e-3,
        transport_gradient_mode=lgssm.core_tf.MANUAL_STREAMING_FINITE_TRANSPORT_GRADIENT_MODE,
        transport_ad_mode="full",
        row_chunk_size=args.num_particles,
        col_chunk_size=args.num_particles,
        particle_chunk_size=args.num_particles,
        score_fd_step=args.fd_step,
        score_fd_atol=5.0e-3,
        score_fd_rtol=5.0e-3,
        score_fd_tf32_mode="match",
        score_mode="compact-sensitivity",
        score_diagnostic_stage="score-and-fd",
        score_reference_json=None,
        history_mode="value-only",
        warmups=0,
        repeats=1,
        dtype="float64",
        tf32_mode="disabled",
        device="/CPU:0",
        device_scope="cpu",
        cuda_visible_devices=None,
        expect_device_kind="cpu",
        output=args.output,
        markdown_output=None,
        aggregate_score_shards=None,
    )


def _weighted_moments(points: tf.Tensor, weights: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    mean = tf.reduce_sum(weights[:, :, None] * points, axis=1)
    centered = points - mean[:, None, :]
    covariance = tf.einsum("bn,bni,bnj->bij", weights, centered, centered)
    return mean, covariance


def _uniform_moments(points: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    count = tf.cast(tf.shape(points)[1], points.dtype)
    weights = tf.fill(tf.shape(points)[:2], tf.cast(1.0, points.dtype) / count)
    return _weighted_moments(points, weights)


def _restore_moments(
    points: tf.Tensor,
    target_mean: tf.Tensor,
    target_covariance: tf.Tensor,
    jitter: float,
) -> tf.Tensor:
    source_mean, source_covariance = _uniform_moments(points)
    state_dim = tf.shape(points)[2]
    eye = tf.eye(state_dim, dtype=points.dtype)[None, :, :]
    ridge = tf.cast(jitter, points.dtype) * eye
    source_chol = tf.linalg.cholesky(source_covariance + ridge)
    target_chol = tf.linalg.cholesky(target_covariance + ridge)
    transform_t = tf.linalg.triangular_solve(
        tf.linalg.matrix_transpose(source_chol),
        tf.linalg.matrix_transpose(target_chol),
        lower=False,
    )
    return tf.matmul(points - source_mean[:, None, :], transform_t) + target_mean[:, None, :]


def _transport_factory(
    variant: str,
    diagnostic_args: argparse.Namespace,
) -> Callable[..., tuple[tf.Tensor, tf.Tensor]]:
    def transport(
        *,
        post_flow: tf.Tensor,
        normalized_log_weights: tf.Tensor,
        mask: tf.Tensor,
        args: argparse.Namespace,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        del args
        batch_size, num_particles, _state_dim = lgssm.core_tf._static_shape(  # noqa: SLF001
            post_flow, "post_flow"
        )
        if variant == "no_reset":
            return post_flow, normalized_log_weights

        center = tf.reduce_mean(post_flow, axis=1, keepdims=True)
        scale = annealed_transport_tf._filterflow_scale(post_flow)  # noqa: SLF001
        scaled_x = (post_flow - center) / scale[:, None, None]
        epsilon = tf.cast(diagnostic_args.sinkhorn_epsilon, post_flow.dtype)
        epsilon0 = annealed_transport_tf._filterflow_epsilon_start(scaled_x)  # noqa: SLF001
        scaling = tf.cast(0.9, post_flow.dtype)
        steps = lgssm.core_tf._manual_dense_finite_steps(  # noqa: SLF001
            diagnostic_args.sinkhorn_iterations
        )
        transported, _row_residual = (
            annealed_transport_tf._filterflow_manual_streaming_finite_transport_total_vjp(  # noqa: SLF001
                scaled_x,
                post_flow,
                normalized_log_weights,
                epsilon,
                epsilon0,
                scaling,
                steps=steps,
                row_chunk_size=diagnostic_args.num_particles,
                col_chunk_size=diagnostic_args.num_particles,
            )
        )

        if variant == "row_normalized":
            ones = tf.ones([batch_size, num_particles, 1], dtype=post_flow.dtype)
            row_mass, _ = (
                annealed_transport_tf._filterflow_manual_streaming_finite_transport_total_vjp(  # noqa: SLF001
                    scaled_x,
                    ones,
                    normalized_log_weights,
                    epsilon,
                    epsilon0,
                    scaling,
                    steps=steps,
                    row_chunk_size=diagnostic_args.num_particles,
                    col_chunk_size=diagnostic_args.num_particles,
                )
            )
            transported = transported / tf.maximum(row_mass, tf.cast(1.0e-300, post_flow.dtype))
        elif variant == "moment_restored":
            target_mean, target_covariance = _weighted_moments(
                post_flow, tf.exp(normalized_log_weights)
            )
            transported = _restore_moments(
                transported,
                target_mean,
                target_covariance,
                diagnostic_args.moment_jitter,
            )

        uniform = lgssm.core_tf.uniform_log_weights(batch_size, num_particles)
        return (
            tf.where(mask[:, None, None], transported, post_flow),
            tf.where(mask[:, None], uniform, normalized_log_weights),
        )

    return transport


def _kalman_oracle(time_steps: int) -> dict[str, Any]:
    observations = tf.cast(
        lgssm._lgssm_dataset(lgssm.DATASET_SEED)["observations"][:time_steps],  # noqa: SLF001
        tf.float64,
    )
    theta = tf.constant(lgssm.TRUTH_THETA, tf.float64)

    def value(candidate: tf.Tensor) -> tf.Tensor:
        phi = candidate[:3]
        q_scale = candidate[3]
        r_scale = candidate[4]
        return tf_kalman_log_likelihood(
            observations=observations,
            transition_offset=tf.zeros([3], tf.float64),
            transition_matrix=tf.linalg.diag(phi),
            transition_covariance=tf.square(q_scale) * tf.eye(3, dtype=tf.float64),
            observation_offset=tf.zeros([3], tf.float64),
            observation_matrix=tf.constant(
                [[1.0, 0.25, -0.15], [0.2, 1.1, 0.3], [-0.1, 0.35, 0.9]], tf.float64
            ),
            observation_covariance=tf.square(r_scale) * tf.eye(3, dtype=tf.float64),
            initial_state_mean=tf.zeros([3], tf.float64),
            initial_state_covariance=tf.linalg.diag(
                tf.square(q_scale) / (1.0 - tf.square(phi))
            ),
        )

    with tf.GradientTape() as tape:
        tape.watch(theta)
        objective = value(theta)
    score = tape.gradient(objective, theta)
    return {"objective": float(objective.numpy()), "score": score.numpy().tolist()}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = _parse_args()
    module_args = _module_args(args)
    lgssm._configure_precision(module_args)  # noqa: SLF001
    theta = tf.constant(lgssm.TRUTH_THETA, dtype=lgssm.DTYPE)
    prepared_tensors = lgssm._build_lgssm_manual_tensors(module_args, theta)  # noqa: SLF001
    original_transport = lgssm._manual_forward_transport_tf  # noqa: SLF001
    records = []
    try:
        for variant in args.variants:
            lgssm._manual_forward_transport_tf = _transport_factory(variant, args)  # noqa: SLF001

            def value(candidate: tf.Tensor) -> tf.Tensor:
                tensors = adapter._candidate_tensors(prepared_tensors, candidate)  # noqa: SLF001
                return lgssm._same_target_value_from_components(  # noqa: SLF001
                    tensors, module_args, candidate
                )["objective"]

            started = time.perf_counter()
            with tf.GradientTape() as tape:
                tape.watch(theta)
                objective = value(theta)
            score = tape.gradient(objective, theta)
            direction = tf.one_hot(3, len(lgssm.PARAMETER_NAMES), dtype=lgssm.DTYPE)
            step = tf.cast(args.fd_step, lgssm.DTYPE)
            minus = value(theta - step * direction)
            plus = value(theta + step * direction)
            fd = (plus - minus) / (2.0 * step)
            records.append(
                {
                    "variant": variant,
                    "objective": float(objective.numpy()),
                    "score": score.numpy().tolist(),
                    "q_scale_score": float(score[3].numpy()),
                    "q_scale_fd": float(fd.numpy()),
                    "q_scale_jvp_fd_gap": float((score[3] - fd).numpy()),
                    "minus_objective": float(minus.numpy()),
                    "plus_objective": float(plus.numpy()),
                    "wall_seconds": time.perf_counter() - started,
                }
            )
    finally:
        lgssm._manual_forward_transport_tf = original_transport  # noqa: SLF001

    kalman = _kalman_oracle(args.time_steps)
    for record in records:
        record["q_scale_gap_to_kalman"] = record["q_scale_score"] - kalman["score"][3]
        record["objective_gap_to_kalman"] = record["objective"] - kalman["objective"]
    payload = {
        "schema_version": "bayesfilter.lgssm_reset_gradient_variants.v1",
        "artifact_status": "completed",
        "terminal_artifact": True,
        "timestamp_utc": dt.datetime.now(tz=dt.UTC).isoformat(),
        "diagnostic_only": True,
        "execution": {
            "device": "CPU",
            "gpu_hidden_deliberately": True,
            "dtype": "float64",
            "jit_compile": False,
            "reason": "small reference/root-cause diagnostic",
        },
        "shape": {
            "num_particles": args.num_particles,
            "time_steps": args.time_steps,
            "seeds": list(args.seeds),
        },
        "transport": {
            "sinkhorn_iterations": args.sinkhorn_iterations,
            "sinkhorn_epsilon": args.sinkhorn_epsilon,
            "moment_jitter": args.moment_jitter,
        },
        "fd_step": args.fd_step,
        "kalman_oracle": kalman,
        "variants": records,
        "nonclaims": [
            "not a production reset implementation",
            "not leaderboard admission",
            "not multi-seed uncertainty unless multiple seeds are supplied",
            "not HMC or posterior correctness evidence",
        ],
    }
    _write_json(Path(args.output), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
