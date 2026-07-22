#!/usr/bin/env python3
"""Trace the first non-finite STR-UKF GenUT stage with scalar diagnostics."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from docs.benchmarks import run_genut_str_ukf_leaderboard as campaign

tf = campaign.tf

from bayesfilter.highdim import cubature_genut_filter as generic
from bayesfilter.highdim import ledh_contract_e_reset_tf as reset


PLAN = Path(
    "docs/plans/bayesfilter-genut-str-ukf-nonfinite-root-cause-plan-2026-07-22.md"
)
SCHEMA = "bayesfilter.genut_str_ukf_stage_trace.v1"
SEEDS = (2026072296, 2026072291)
CONTROLS = {
    "epsilon": 4.0,
    "sinkhorn_steps": 4,
    "balance_steps": 8,
    "ridge": 1.0e-6,
}

BOOL_METRICS = (
    "input_particles_finite",
    "input_tangent_finite",
    "transition_particles_finite",
    "transition_tangent_finite",
    "log_likelihood_finite",
    "log_likelihood_tangent_finite",
    "increment_finite",
    "normalized_weights_finite",
    "normalized_weight_tangent_finite",
    "barycentric_finite",
    "barycentric_tangent_finite",
    "reset_forward_finite",
    "reset_forward_batch_finite",
    "reset_tangent_finite",
    "restored_particles_finite",
    "restored_tangent_finite",
)

FLOAT_METRICS = (
    "input_particles_max_abs",
    "input_tangent_max_abs",
    "transition_particles_max_abs",
    "transition_tangent_max_abs",
    "transition_residual",
    "log_likelihood_min",
    "log_likelihood_max",
    "log_likelihood_tangent_max_abs",
    "increment",
    "normalized_weight_min",
    "normalized_weight_max",
    "normalized_weight_ess",
    "normalized_weight_tangent_max_abs",
    "barycentric_max_abs",
    "barycentric_tangent_max_abs",
    "sinkhorn_row_residual",
    "sinkhorn_col_residual",
    "gap_min_eigenvalue",
    "gap_plus_ridge_min_eigenvalue",
    "target_cov_min_eigenvalue",
    "injected_cov_min_eigenvalue",
    "gap_chol_min_diagonal",
    "target_chol_min_diagonal",
    "injected_chol_min_diagonal",
    "gap_condition_proxy",
    "target_condition_proxy",
    "injected_condition_proxy",
    "reset_mean_residual",
    "restored_particles_max_abs",
    "restored_tangent_max_abs",
    "score_increment_max_abs",
)


def _all_finite(value: tf.Tensor) -> tf.Tensor:
    return tf.reduce_all(tf.math.is_finite(value))


def _max_abs(value: tf.Tensor) -> tf.Tensor:
    return tf.reduce_max(tf.abs(value))


def _build_stage_trace():
    adapter = campaign.structural_ukf_candidate_adapter()
    n = campaign.N
    horizon = campaign.HORIZON
    parameter_count = 5
    ridge = tf.constant([CONTROLS["ridge"]], tf.float32)

    @tf.function(jit_compile=True, reduce_retracing=True)
    def trace(
        theta: tf.Tensor,
        observations: tf.Tensor,
        initial_noise: tf.Tensor,
        process_noise: tf.Tensor,
        design: tf.Tensor,
    ) -> dict[str, tf.Tensor]:
        theta = tf.ensure_shape(theta, [parameter_count])
        observations = tf.ensure_shape(observations, [horizon, 1])
        initial_noise = tf.ensure_shape(initial_noise, [n, 2])
        process_noise = tf.ensure_shape(process_noise, [horizon, n, 1])
        design = tf.ensure_shape(design, [n, 2])
        particles = adapter.initial_value(theta, initial_noise)
        particle_tangent = adapter.initial_tangent(theta, initial_noise)
        weights = tf.fill([n], tf.constant(1.0 / n, tf.float32))
        weight_tangent = tf.zeros([n, parameter_count], tf.float32)
        bool_arrays = {
            name: tf.TensorArray(tf.bool, size=horizon, element_shape=())
            for name in BOOL_METRICS
        }
        float_arrays = {
            name: tf.TensorArray(tf.float32, size=horizon, element_shape=())
            for name in FLOAT_METRICS
        }

        def body(
            index,
            particles_value,
            tangent_value,
            weights_value,
            weight_tangent_value,
            bool_values,
            float_values,
        ):
            transition = tf.not_equal(index, tf.constant(0, tf.int32))
            particles_next = tf.cond(
                transition,
                lambda: adapter.transition_value(
                    theta, particles_value, process_noise[index], index
                ),
                lambda: particles_value,
            )
            tangent_next = tf.cond(
                transition,
                lambda: adapter.transition_tangent(
                    theta,
                    particles_value,
                    process_noise[index],
                    tangent_value,
                    index,
                ),
                lambda: tangent_value,
            )
            transition_residual = tf.cond(
                transition,
                lambda: tf.reduce_max(
                    tf.abs(
                        adapter.transition_residual(
                            theta, particles_value, particles_next, index
                        )
                    )
                ),
                lambda: tf.constant(0.0, tf.float32),
            )
            log_likelihood = adapter.observation_value(
                theta, particles_next, observations[index], index
            )
            log_likelihood_tangent = adapter.observation_tangent(
                theta,
                particles_next,
                tangent_next,
                observations[index],
                index,
            )
            log_weights = tf.math.log(weights_value) + log_likelihood
            log_weight_tangent = (
                weight_tangent_value / weights_value[:, None]
                + log_likelihood_tangent
            )
            increment = tf.reduce_logsumexp(log_weights)
            normalized_weights = tf.exp(log_weights - increment)
            increment_tangent = tf.reduce_sum(
                normalized_weights[:, None] * log_weight_tangent, axis=0
            )
            normalized_weight_tangent = normalized_weights[:, None] * (
                log_weight_tangent - increment_tangent[None, :]
            )

            barycentric, barycentric_tangent, row_residual, col_residual = (
                generic._sinkhorn_barycentric_jvp(  # noqa: SLF001
                    particles_next,
                    normalized_weights,
                    tangent_next,
                    normalized_weight_tangent,
                    epsilon=CONTROLS["epsilon"],
                    sinkhorn_steps=CONTROLS["sinkhorn_steps"],
                    balance_steps=CONTROLS["balance_steps"],
                )
            )
            source = particles_next[None, :, :]
            source_weights = normalized_weights[None, :]
            transported = barycentric[None, :, :]
            residual_design = design[None, :, :]
            source_batch = tf.broadcast_to(source, [parameter_count, n, 2])
            source_weights_batch = tf.broadcast_to(
                source_weights, [parameter_count, n]
            )
            transported_batch = tf.broadcast_to(
                transported, [parameter_count, n, 2]
            )
            design_batch = tf.broadcast_to(
                residual_design, [parameter_count, n, 2]
            )
            ridge_batch = tf.broadcast_to(ridge, [parameter_count])
            forward_batch = reset._contract_e_chol_cloud_forward_core(  # noqa: SLF001
                source_batch,
                source_weights_batch,
                transported_batch,
                design_batch,
                ridge_batch,
            )
            tangent_batch = reset._contract_e_chol_cloud_jvp_from_forward_core(  # noqa: SLF001
                forward_batch,
                source_batch,
                source_weights_batch,
                transported_batch,
                design_batch,
                ridge_batch,
                tf.transpose(tangent_next, [2, 0, 1]),
                tf.transpose(normalized_weight_tangent, [1, 0]),
                tf.transpose(barycentric_tangent, [2, 0, 1]),
                tf.zeros_like(design_batch),
                tf.zeros_like(ridge_batch),
            )["particles"]
            restored_tangent = tf.transpose(tangent_batch, [1, 2, 0])
            forward = reset._contract_e_chol_cloud_forward_core(  # noqa: SLF001
                source,
                source_weights,
                transported,
                residual_design,
                ridge,
            )
            restored = forward["particles"][0]

            bool_step = {
                "input_particles_finite": _all_finite(particles_value),
                "input_tangent_finite": _all_finite(tangent_value),
                "transition_particles_finite": _all_finite(particles_next),
                "transition_tangent_finite": _all_finite(tangent_next),
                "log_likelihood_finite": _all_finite(log_likelihood),
                "log_likelihood_tangent_finite": _all_finite(
                    log_likelihood_tangent
                ),
                "increment_finite": tf.math.is_finite(increment),
                "normalized_weights_finite": _all_finite(normalized_weights),
                "normalized_weight_tangent_finite": _all_finite(
                    normalized_weight_tangent
                ),
                "barycentric_finite": _all_finite(barycentric),
                "barycentric_tangent_finite": _all_finite(
                    barycentric_tangent
                ),
                "reset_forward_finite": forward["finite"][0],
                "reset_forward_batch_finite": tf.reduce_all(
                    forward_batch["finite"]
                ),
                "reset_tangent_finite": _all_finite(tangent_batch),
                "restored_particles_finite": _all_finite(restored),
                "restored_tangent_finite": _all_finite(restored_tangent),
            }
            float_step = {
                "input_particles_max_abs": _max_abs(particles_value),
                "input_tangent_max_abs": _max_abs(tangent_value),
                "transition_particles_max_abs": _max_abs(particles_next),
                "transition_tangent_max_abs": _max_abs(tangent_next),
                "transition_residual": transition_residual,
                "log_likelihood_min": tf.reduce_min(log_likelihood),
                "log_likelihood_max": tf.reduce_max(log_likelihood),
                "log_likelihood_tangent_max_abs": _max_abs(
                    log_likelihood_tangent
                ),
                "increment": increment,
                "normalized_weight_min": tf.reduce_min(normalized_weights),
                "normalized_weight_max": tf.reduce_max(normalized_weights),
                "normalized_weight_ess": tf.math.reciprocal(
                    tf.reduce_sum(tf.square(normalized_weights))
                ),
                "normalized_weight_tangent_max_abs": _max_abs(
                    normalized_weight_tangent
                ),
                "barycentric_max_abs": _max_abs(barycentric),
                "barycentric_tangent_max_abs": _max_abs(barycentric_tangent),
                "sinkhorn_row_residual": row_residual,
                "sinkhorn_col_residual": col_residual,
                "gap_min_eigenvalue": tf.reduce_min(forward["gap_eigenvalues"]),
                "gap_plus_ridge_min_eigenvalue": (
                    tf.reduce_min(forward["gap_eigenvalues"]) + ridge[0]
                ),
                "target_cov_min_eigenvalue": tf.reduce_min(
                    tf.linalg.eigvalsh(forward["target_cov"])
                ),
                "injected_cov_min_eigenvalue": tf.reduce_min(
                    tf.linalg.eigvalsh(forward["injected_cov"])
                ),
                "gap_chol_min_diagonal": tf.reduce_min(
                    forward["gap_chol_diagonal"]
                ),
                "target_chol_min_diagonal": tf.reduce_min(
                    forward["target_chol_diagonal"]
                ),
                "injected_chol_min_diagonal": tf.reduce_min(
                    forward["injected_chol_diagonal"]
                ),
                "gap_condition_proxy": tf.reduce_max(
                    forward["gap_condition_proxy"]
                ),
                "target_condition_proxy": tf.reduce_max(
                    forward["target_condition_proxy"]
                ),
                "injected_condition_proxy": tf.reduce_max(
                    forward["injected_condition_proxy"]
                ),
                "reset_mean_residual": tf.reduce_max(
                    tf.abs(forward["mean_residual"])
                ),
                "restored_particles_max_abs": _max_abs(restored),
                "restored_tangent_max_abs": _max_abs(restored_tangent),
                "score_increment_max_abs": _max_abs(increment_tangent),
            }
            bool_next = {
                name: bool_values[name].write(index, bool_step[name])
                for name in BOOL_METRICS
            }
            float_next = {
                name: float_values[name].write(index, float_step[name])
                for name in FLOAT_METRICS
            }
            return (
                index + 1,
                restored,
                restored_tangent,
                tf.fill([n], tf.constant(1.0 / n, tf.float32)),
                tf.zeros_like(weight_tangent_value),
                bool_next,
                float_next,
            )

        result = tf.while_loop(
            lambda index, *_: index < horizon,
            body,
            (
                tf.constant(0, tf.int32),
                particles,
                particle_tangent,
                weights,
                weight_tangent,
                bool_arrays,
                float_arrays,
            ),
            parallel_iterations=1,
        )
        return {
            **{name: result[5][name].stack() for name in BOOL_METRICS},
            **{name: result[6][name].stack() for name in FLOAT_METRICS},
        }

    return trace


def _json_float(value: float) -> float | str:
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "+Inf" if value > 0 else "-Inf"
    return value


def _first_false(values: list[bool]) -> int | None:
    return next((index for index, value in enumerate(values) if not value), None)


def run(output_root: Path) -> dict[str, object]:
    campaign._require_serious_gpu_policy()  # noqa: SLF001
    output_root.mkdir(parents=True, exist_ok=False)
    tf.config.set_soft_device_placement(False)
    tf.config.experimental.enable_tensor_float_32_execution(True)
    logical = tf.config.list_logical_devices("GPU")
    if not logical:
        raise RuntimeError("stage trace requires a logical GPU")
    _states, observations64 = campaign.generate_frozen_structural_dataset_tf()
    observations = tf.cast(observations64, tf.float32)
    theta = tf.cast(campaign.structural_truth_source(), tf.float32)
    design = campaign._genut_design()  # noqa: SLF001
    trace = _build_stage_trace()
    rows = []
    for seed in SEEDS:
        initial, process = campaign._particle_noise(seed)  # noqa: SLF001
        started = time.perf_counter()
        result = trace(theta, observations, initial, process, design)
        row: dict[str, object] = {
            "seed": seed,
            "wall_time_seconds_including_first_compile": time.perf_counter()
            - started,
        }
        for name in BOOL_METRICS:
            values = [bool(item) for item in result[name].numpy()]
            row[name] = values
            row[f"first_false_{name}"] = _first_false(values)
        for name in FLOAT_METRICS:
            row[name] = [_json_float(float(item)) for item in result[name].numpy()]
        rows.append(row)
        (output_root / f"stage_seed_{seed}.json").write_text(
            json.dumps(row, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    payload = {
        "schema_version": SCHEMA,
        "plan": PLAN.as_posix(),
        "target_scope": campaign.STRUCTURAL_UKF_SCOPE,
        "particle_count": campaign.N,
        "horizon": campaign.HORIZON,
        "controls": CONTROLS,
        "seeds": SEEDS,
        "rows": rows,
        "device": {
            "logical_devices": [item.name for item in logical],
            "dtype": "float32",
            "tf32_enabled": bool(
                tf.config.experimental.tensor_float_32_execution_enabled()
            ),
            "jit_compile": True,
        },
        "memory_policy": campaign.MEMORY_POLICY,
        "allocator": {
            key: int(value)
            for key, value in tf.config.experimental.get_memory_info("GPU:0").items()
        },
        "classification": "diagnostic_only_consumed_claim_seed_stage_trace",
    }
    (output_root / "stage_trace.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.output_root)
    print(json.dumps({"seeds": result["seeds"], "output": str(args.output_root)}))


if __name__ == "__main__":
    main()
