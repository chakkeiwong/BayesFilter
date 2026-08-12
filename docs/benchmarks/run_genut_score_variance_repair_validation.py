#!/usr/bin/env python3
"""Diagnostic validation for the GenUT score-variance repair note.

This runner deliberately reports finite-time directional tangent growth rather
than an asymptotic Lyapunov exponent. It uses the same finite value program as
the July 30 comparison, with additional probe columns propagated through the
state Jacobian and zero explicit parameter source.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
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


N = 1008
K = 8
AUSTRIA_SEEDS = (98201, 98202, 98203)
LGSSM_SEEDS = (98201, 98202, 98203)
SCHEMA = "bayesfilter.genut_score_variance_repair_validation.v1"
PLAN = "docs/plans/bayesfilter-genut-score-variance-repair-validation-plan-2026-07-31.md"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _safe(value: Any) -> Any:
    if hasattr(value, "numpy"):
        value = value.numpy()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _whiten_diagnostic() -> dict[str, Any]:
    rows = []
    for seed in (71001, 71002, 71003):
        z = tf.random.stateless_normal([N, 18], [seed, 19], dtype=tf.float64)
        centered = z - tf.reduce_mean(z, axis=0, keepdims=True)
        covariance = tf.transpose(centered) @ centered / tf.cast(N, tf.float64)
        chol = tf.linalg.cholesky(covariance)
        xi = tf.transpose(
            tf.linalg.triangular_solve(chol, tf.transpose(centered), lower=True)
        )
        mean_error = tf.reduce_max(tf.abs(tf.reduce_mean(xi, axis=0)))
        covariance_error = tf.reduce_max(
            tf.abs(tf.transpose(xi) @ xi / tf.cast(N, tf.float64) - tf.eye(18, dtype=tf.float64))
        )
        standardized = xi
        kurtosis = tf.reduce_mean(tf.pow(standardized, 4), axis=0)
        m22 = tf.einsum("ni,nj->ij", tf.square(standardized), tf.square(standardized)) / tf.cast(N, tf.float64)
        offdiag = 1.0 - tf.eye(18, dtype=tf.float64)
        rows.append(
            {
                "seed": seed,
                "mean_error": float(mean_error.numpy()),
                "covariance_error": float(covariance_error.numpy()),
                "kurtosis_mean": float(tf.reduce_mean(kurtosis).numpy()),
                "kurtosis_max_abs_error": float(tf.reduce_max(tf.abs(kurtosis - 3.0)).numpy()),
                "studentized_co_kurtosis_mean": float(
                    tf.reduce_sum(offdiag * m22) / tf.reduce_sum(offdiag)
                ),
                "studentized_co_kurtosis_max_abs_error": float(
                    tf.reduce_max(tf.abs(offdiag * (m22 - 1.0))).numpy()
                ),
            }
        )
    return {
        "exact_identity_pass": all(
            row["mean_error"] < 2.0e-12 and row["covariance_error"] < 2.0e-12
            for row in rows
        ),
        "rows": rows,
        "interpretation": "raw m22 has sqrt(8/N) first-order SD; studentized co-kurtosis has sqrt(4/N) first-order SD; full whitening is not replaced by this heuristic.",
    }


def _load_context():
    from docs.benchmarks import run_moment_retuned_genut_whole_leaderboard as base
    from bayesfilter.highdim.cubature_genut_adapters import (
        diagonal_lgssm_candidate_adapter,
        parameterized_austria_sir_candidate_adapter,
    )
    from scripts.filtering_value_gradient_benchmark_generate_p8_datasets import _lgssm_dataset
    from bayesfilter.highdim.models import zhao_cui_sir_austria_model
    from bayesfilter.testing.sir_filter_neutra_target_design_tf import generate_frozen_sir_dataset_tf

    lgssm = _lgssm_dataset(81100)
    lg_observations = tf.cast(lgssm["observations"][:50], tf.float32)
    lg_adapter = diagonal_lgssm_candidate_adapter(
        observation_matrix=tf.constant([[1.0, 0.25, -0.15], [0.2, 1.1, 0.3], [-0.1, 0.35, 0.9]], tf.float32)
    )
    _, sir_observations, _ = generate_frozen_sir_dataset_tf()
    return base, {
        "lgssm_diagonal": {
            "adapter": lg_adapter,
            "theta": tf.constant([0.72, 0.55, 0.35, 0.35, 0.45], tf.float32),
            "observations": lg_observations,
            "transition_before": False,
            "seeds": LGSSM_SEEDS,
            "controls": {"epsilon": 2.0, "sinkhorn_steps": 8, "balance_steps": 8, "ridge": 1.0e-5, "higher_moment_correction_steps": 4, "higher_moment_strength": 0.2, "higher_moment_floor": 1.0e-5},
            "state_dim": 3,
            "explicit_gaussian_targets": False,
        },
        "lgssm_gaussian_targets": {
            "adapter": lg_adapter,
            "theta": tf.constant([0.72, 0.55, 0.35, 0.35, 0.45], tf.float32),
            "observations": lg_observations,
            "transition_before": False,
            "seeds": LGSSM_SEEDS,
            "controls": {"epsilon": 2.0, "sinkhorn_steps": 8, "balance_steps": 8, "ridge": 1.0e-5, "higher_moment_correction_steps": 4, "higher_moment_strength": 0.2, "higher_moment_floor": 1.0e-5},
            "state_dim": 3,
            "explicit_gaussian_targets": True,
        },
        "austria_diagonal": {
            "adapter": parameterized_austria_sir_candidate_adapter(),
            "theta": tf.zeros([3], tf.float32),
            "observations": tf.cast(sir_observations, tf.float32),
            "transition_before": True,
            "seeds": AUSTRIA_SEEDS,
            "controls": {"epsilon": 8.0, "sinkhorn_steps": 16, "balance_steps": 16, "ridge": 1.0e-5, "higher_moment_correction_steps": 4, "higher_moment_strength": 0.2, "higher_moment_floor": 1.0e-5},
            "state_dim": 18,
            "explicit_gaussian_targets": False,
        },
        "austria_pairwise": {
            "adapter": parameterized_austria_sir_candidate_adapter(),
            "theta": tf.zeros([3], tf.float32),
            "observations": tf.cast(sir_observations, tf.float32),
            "transition_before": True,
            "seeds": AUSTRIA_SEEDS,
            "controls": {"epsilon": 8.0, "sinkhorn_steps": 16, "balance_steps": 16, "ridge": 1.0e-5, "higher_moment_correction_steps": 4, "higher_moment_strength": 0.2, "higher_moment_floor": 1.0e-5, "pairwise_moment_correction_steps": 4, "pairwise_moment_strength": 0.02, "pairwise_moment_floor": 1.0e-5},
            "state_dim": 18,
            "explicit_gaussian_targets": False,
        },
    }


def _explicit_target_kwargs(
    context: dict[str, Any], *, tangent_count: int | None = None
) -> dict[str, tf.Tensor]:
    if not context.get("explicit_gaussian_targets", False):
        return {}
    dimension = int(context["state_dim"])
    parameter_count = int(
        context["adapter"].parameter_count if tangent_count is None else tangent_count
    )
    off_diagonal = 1.0 - tf.eye(dimension, dtype=tf.float32)
    return {
        "explicit_target_skew": tf.zeros([dimension], tf.float32),
        "explicit_target_kurtosis": tf.fill([dimension], tf.constant(3.0, tf.float32)),
        "explicit_target_skew_tangent": tf.zeros([dimension, parameter_count], tf.float32),
        "explicit_target_kurtosis_tangent": tf.zeros([dimension, parameter_count], tf.float32),
        "explicit_target_pairwise_co_skew": tf.zeros([dimension, dimension], tf.float32),
        "explicit_target_pairwise_co_kurtosis": off_diagonal,
        "explicit_target_pairwise_co_skew_tangent": tf.zeros([dimension, dimension, parameter_count], tf.float32),
        "explicit_target_pairwise_co_kurtosis_tangent": tf.zeros([dimension, dimension, parameter_count], tf.float32),
        "pairwise_co_skew_target_mask": off_diagonal,
        "pairwise_co_kurtosis_target_mask": off_diagonal,
    }


def _make_evaluator(context: dict[str, Any]):
    from bayesfilter.highdim.cubature_genut_filter import finite_value_score

    horizon = int(context["observations"].shape[0])
    observation_dim = int(context["observations"].shape[1])
    state_dim = int(context["state_dim"])
    parameter_dim = int(context["adapter"].parameter_count)
    controls = context["controls"]
    explicit = _explicit_target_kwargs(context)

    @tf.function(jit_compile=True, reduce_retracing=True)
    def evaluate(theta, observations, initial_noise, process_noise, design):
        theta = tf.ensure_shape(theta, [parameter_dim])
        observations = tf.ensure_shape(observations, [horizon, observation_dim])
        initial_noise = tf.ensure_shape(initial_noise, [N, state_dim])
        process_noise = tf.ensure_shape(process_noise, [horizon, N, state_dim])
        design = tf.ensure_shape(design, [N, state_dim])
        with tf.device("/GPU:0"):
            return finite_value_score(
                context["adapter"], theta, observations, initial_noise, process_noise, design,
                epsilon=float(controls["epsilon"]), sinkhorn_steps=int(controls["sinkhorn_steps"]),
                balance_steps=int(controls["balance_steps"]), ridge=float(controls["ridge"]),
                transition_before_first_observation=context["transition_before"],
                higher_moment_correction_steps=int(controls["higher_moment_correction_steps"]),
                higher_moment_strength=float(controls["higher_moment_strength"]),
                higher_moment_floor=float(controls["higher_moment_floor"]),
                pairwise_moment_correction_steps=int(controls.get("pairwise_moment_correction_steps", 0)),
                pairwise_moment_strength=float(controls.get("pairwise_moment_strength", 0.0)),
                pairwise_moment_floor=float(controls.get("pairwise_moment_floor", 1.0e-5)),
                **explicit,
            )
    return evaluate


def _run_arm(
    base,
    context: dict[str, Any],
    arm_id: str,
    *,
    growth_only: bool = False,
    probe_batch_size: int = K,
    skip_growth: bool = False,
) -> dict[str, Any]:
    from bayesfilter.highdim.cubature_genut_candidate import cubature_design
    from bayesfilter.highdim.cubature_genut_filter import finite_value_score

    state_dim = context["state_dim"]
    horizon = int(context["observations"].shape[0])
    design = cubature_design(dim=state_dim, num_particles=N)
    evaluator = None if growth_only else _make_evaluator(context)
    rows = []
    for seed in context["seeds"]:
        initial = tf.random.stateless_normal([N, state_dim], [seed, 101], dtype=tf.float32)
        process = tf.random.stateless_normal(
            [horizon, N, state_dim], [seed, 102], dtype=tf.float32
        )
        if evaluator is not None:
            value, score, diagnostics = evaluator(
                context["theta"], context["observations"], initial, process, design
            )
            increments = diagnostics["score_increments"]
            score_residual = tf.reduce_max(
                tf.abs(tf.reduce_sum(increments, axis=0) - score)
            )
        else:
            value = score = diagnostics = score_residual = None
        probe = (
            {
                "finite": True,
                "gammahat": [],
                "per_step_log_growth": [],
                "skipped": True,
            }
            if skip_growth
            else _finite_time_probe(
                context=context,
                initial_noise=initial,
                process_noise=process,
                design=design,
                seed=seed,
                probe_batch_size=probe_batch_size,
            )
        )
        rows.append({
            "seed": seed,
            "value": None if value is None else float(value.numpy()),
            "score": None if score is None else [float(x) for x in score.numpy()],
            "score_increment_sum_residual": (
                None if score_residual is None else float(score_residual.numpy())
            ),
            "finite": (
                True
                if value is None
                else bool(tf.math.is_finite(value).numpy())
                and bool(tf.reduce_all(tf.math.is_finite(score)).numpy())
                and bool(diagnostics["program_valid"].numpy())
            ),
            "max_mean_residual": (
                None
                if diagnostics is None
                else float(diagnostics["max_mean_residual"].numpy())
            ),
            "max_row_residual": (
                None
                if diagnostics is None
                else float(diagnostics["max_row_residual"].numpy())
            ),
            "max_col_residual": (
                None
                if diagnostics is None
                else float(diagnostics["max_col_residual"].numpy())
            ),
            "finite_time_directional_growth": probe,
        })
    score_sds = None
    if not growth_only:
        score_matrix = [
            [row["score"][j] for row in rows]
            for j in range(context["adapter"].parameter_count)
        ]
        score_sds = [
            math.sqrt(
                sum((value - statistics.mean(values)) ** 2 for value in values)
                / (len(values) - 1)
            )
            for values in score_matrix
        ]
    gamma_values = [
        value
        for row in rows
        for value in row["finite_time_directional_growth"]["gammahat"]
    ]
    return {
        "arm_id": arm_id,
        "controls": context["controls"],
        "rows": rows,
        "score_sd": score_sds,
        "finite_time_gammahat_mean": (
            None if not gamma_values else statistics.mean(gamma_values)
        ),
        "finite_time_gammahat_min": None if not gamma_values else min(gamma_values),
        "finite_time_gammahat_max": None if not gamma_values else max(gamma_values),
        "hard_valid": all(
            row["finite"]
            and (
                row["score_increment_sum_residual"] is None
                or row["score_increment_sum_residual"] < 1.0e-4
            )
            and row["finite_time_directional_growth"]["finite"]
            for row in rows
        ),
    }


def _finite_time_probe(
    *,
    context: dict[str, Any],
    initial_noise: tf.Tensor,
    process_noise: tf.Tensor,
    design: tf.Tensor,
    seed: int,
    probe_batch_size: int = K,
) -> dict[str, Any]:
    if probe_batch_size < 1 or probe_batch_size > K:
        raise ValueError(f"probe_batch_size must be in [1,{K}]")
    batches = []
    for start in range(0, K, probe_batch_size):
        stop = min(start + probe_batch_size, K)
        batches.append(
            _finite_time_probe_batch(
                context=context,
                initial_noise=initial_noise,
                process_noise=process_noise,
                design=design,
                seed=seed,
                probe_start=start,
                probe_stop=stop,
            )
        )
    if not all(batch["finite"] for batch in batches):
        return {
            "finite": False,
            "gammahat": [],
            "per_step_log_growth": [],
            "probe_batch_size": probe_batch_size,
        }
    horizon = len(batches[0]["per_step_log_growth"])
    return {
        "finite": True,
        "gammahat": [value for batch in batches for value in batch["gammahat"]],
        "per_step_log_growth": [
            [
                value
                for batch in batches
                for value in batch["per_step_log_growth"][time_index]
            ]
            for time_index in range(horizon)
        ],
        "probe_batch_size": probe_batch_size,
    }


def _finite_time_probe_batch(
    *,
    context: dict[str, Any],
    initial_noise: tf.Tensor,
    process_noise: tf.Tensor,
    design: tf.Tensor,
    seed: int,
    probe_start: int,
    probe_stop: int,
) -> dict[str, Any]:
    from bayesfilter.highdim.cubature_genut_filter import _restore_cloud_jvp_core
    from bayesfilter.highdim.higher_moment_contract_e import higher_moment_shape_jvp

    adapter = context["adapter"]
    theta = context["theta"]
    controls = context["controls"]
    observations = context["observations"]
    probe_count = probe_stop - probe_start
    explicit = _explicit_target_kwargs(context, tangent_count=probe_count)
    particles = adapter.initial_value(theta, initial_noise)
    n = int(particles.shape[0])
    weights = tf.fill([n], tf.cast(1.0 / n, tf.float32))
    all_probes = tf.random.stateless_normal(
        [n, context["state_dim"], K], [seed, 7001], dtype=tf.float32
    )
    probes = all_probes[:, :, probe_start:probe_stop]
    probes /= tf.sqrt(tf.reduce_sum(tf.square(probes), axis=[0, 1], keepdims=True))
    log_growth = tf.zeros([probe_count], tf.float32)
    per_step = []

    for time_index in range(int(observations.shape[0])):
        noise = process_noise[time_index]
        transition = context["transition_before"] or time_index != 0
        if transition:
            probe_columns = []
            particles_next = None
            for probe_index in range(probe_count):
                with tf.autodiff.ForwardAccumulator(
                    particles, probes[:, :, probe_index]
                ) as acc:
                    current_particles_next = adapter.transition_value(
                        theta, particles, noise, tf.constant(time_index, tf.int32)
                    )
                particles_next = current_particles_next
                probe_columns.append(acc.jvp(current_particles_next))
            assert particles_next is not None
            probes_next = tf.stack(probe_columns, axis=-1)
        else:
            particles_next = particles
            probes_next = probes

        likelihood_probe_columns = []
        log_likelihood = None
        for probe_index in range(probe_count):
            with tf.autodiff.ForwardAccumulator(
                particles_next, probes_next[:, :, probe_index]
            ) as acc:
                current_log_likelihood = adapter.observation_value(
                    theta,
                    particles_next,
                    observations[time_index],
                    tf.constant(time_index, tf.int32),
                )
            log_likelihood = current_log_likelihood
            likelihood_probe_columns.append(acc.jvp(current_log_likelihood))
        assert log_likelihood is not None
        log_likelihood_probe = tf.stack(likelihood_probe_columns, axis=-1)
        log_weights = tf.math.log(weights) + log_likelihood
        normalized_weights = tf.nn.softmax(log_weights)
        centered_probe = log_likelihood_probe - tf.reduce_sum(
            normalized_weights[:, None] * log_likelihood_probe, axis=0
        )[None, :]
        weight_probe = normalized_weights[:, None] * centered_probe
        current_design = design if design.shape.rank == 2 else design[time_index]
        restored = _restore_cloud_jvp_core(
            particles_next,
            normalized_weights,
            probes_next,
            weight_probe,
            current_design,
            epsilon=float(controls["epsilon"]),
            sinkhorn_steps=int(controls["sinkhorn_steps"]),
            balance_steps=int(controls["balance_steps"]),
            ridge=float(controls["ridge"]),
            parameter_count=probe_count,
        )
        higher = higher_moment_shape_jvp(
            particles_next,
            normalized_weights,
            probes_next,
            weight_probe,
            restored["particles"],
            restored["particles_tangent"],
            correction_steps=int(controls["higher_moment_correction_steps"]),
            strength=float(controls["higher_moment_strength"]),
            floor=float(controls["higher_moment_floor"]),
            pairwise_correction_steps=int(controls.get("pairwise_moment_correction_steps", 0)),
            pairwise_strength=float(controls.get("pairwise_moment_strength", 0.0)),
            pairwise_floor=float(controls.get("pairwise_moment_floor", 1.0e-5)),
            **explicit,
        )
        step_valid = bool(restored["reset_valid"].numpy()) and bool(higher["valid"].numpy())
        next_probes = higher["particles_tangent"]
        norms = tf.sqrt(tf.reduce_sum(tf.square(next_probes), axis=[0, 1]))
        safe_norms = tf.maximum(norms, tf.constant(1.0e-30, tf.float32))
        step_log = tf.math.log(safe_norms)
        log_growth += step_log
        per_step.append([float(value) for value in step_log.numpy()])
        probes = next_probes / safe_norms[None, None, :]
        particles = higher["particles"]
        weights = tf.fill([n], tf.cast(1.0 / n, tf.float32))
        if not step_valid:
            return {"finite": False, "gammahat": [], "per_step_log_growth": per_step}

    gammahat = log_growth / tf.cast(tf.shape(observations)[0], tf.float32)
    return {
        "finite": bool(tf.reduce_all(tf.math.is_finite(gammahat)).numpy()),
        "gammahat": [float(value) for value in gammahat.numpy()],
        "per_step_log_growth": per_step,
    }


def run(
    output_root: Path,
    *,
    arm_ids: tuple[str, ...] | None = None,
    growth_only: bool = False,
    probe_batch_size: int = K,
    skip_growth: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    memory_policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.experimental.enable_tensor_float_32_execution(True)
    logical = tf.config.list_logical_devices("GPU")
    if not logical:
        raise RuntimeError("validation requires a visible GPU")
    base, contexts = _load_context()
    whitening = _whiten_diagnostic()
    selected_names = tuple(contexts) if arm_ids is None else arm_ids
    unknown = sorted(set(selected_names) - set(contexts))
    if unknown:
        raise ValueError(f"unknown arm ids: {unknown}")
    arms = [
        _run_arm(
            base,
            contexts[name],
            name,
            growth_only=growth_only,
            probe_batch_size=probe_batch_size,
            skip_growth=skip_growth,
        )
        for name in selected_names
    ]
    payload = {
        "schema": SCHEMA,
        "campaign_id": "genut-score-variance-repair-validation-20260731",
        "plan": PLAN,
        "git_commit": _git_commit(),
        "host": platform.node(),
        "tensorflow": tf.__version__,
        "device": [device.name for device in logical],
        "memory_policy": _safe(memory_policy),
        "configuration": {
            "particles": N,
            "probe_columns": K,
            "probe_batch_size": probe_batch_size,
            "probe_status": "diagnostic_forward_accumulator_plus_existing_manual_reset_jvp",
            "growth_only": growth_only,
            "skip_growth": skip_growth,
            "tf32": True,
            "jit_compile": True,
        },
        "whitening": whitening,
        "arms": arms,
        "hard_valid": whitening["exact_identity_pass"] and all(arm["hard_valid"] for arm in arms),
        "wall_time_seconds": time.perf_counter() - started,
        "nonclaims": ["no asymptotic Lyapunov exponent", "no Var(score)=O(T) theorem", "no score superiority", "no exact nonlinear score oracle"],
    }
    output_root.mkdir(parents=True, exist_ok=False)
    (output_root / "result.json").write_text(json.dumps(_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# GenUT Score-Variance Repair Validation", "", f"- hard_valid: `{payload['hard_valid']}`", f"- whitening_identity_pass: `{whitening['exact_identity_pass']}`", "", "| Arm | Hard valid | Mean finite-time growth | Range | Score SD |", "|---|---:|---:|---|---|"]
    for arm in arms:
        mean_growth = "N/A" if arm["finite_time_gammahat_mean"] is None else f"{arm['finite_time_gammahat_mean']:.6g}"
        min_growth = "N/A" if arm["finite_time_gammahat_min"] is None else f"{arm['finite_time_gammahat_min']:.6g}"
        max_growth = "N/A" if arm["finite_time_gammahat_max"] is None else f"{arm['finite_time_gammahat_max']:.6g}"
        lines.append(f"| {arm['arm_id']} | {arm['hard_valid']} | {mean_growth} | [{min_growth}, {max_growth}] | `{arm['score_sd']}` |")
    (output_root / "result.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output_root / "run_manifest.json").write_text(json.dumps({"schema": "bayesfilter.genut_score_variance_repair_validation_manifest.v1", "result_sha256": _sha256(output_root / "result.json"), "command": "docs/benchmarks/run_genut_score_variance_repair_validation.py", "plan": PLAN, "output_root": str(output_root)}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output_root), "hard_valid": payload["hard_valid"]}, indent=2))
    return payload


def main() -> None:
    global N
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--particles", type=int, default=N)
    parser.add_argument("--arms", nargs="+", default=None)
    parser.add_argument("--growth-only", action="store_true")
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument("--probe-batch-size", type=int, default=K)
    args = parser.parse_args()
    if args.particles < 1:
        raise ValueError("--particles must be positive")
    N = args.particles
    run(
        args.output_root.resolve(),
        arm_ids=None if args.arms is None else tuple(args.arms),
        growth_only=args.growth_only,
        probe_batch_size=args.probe_batch_size,
        skip_growth=args.score_only,
    )


if __name__ == "__main__":
    main()
