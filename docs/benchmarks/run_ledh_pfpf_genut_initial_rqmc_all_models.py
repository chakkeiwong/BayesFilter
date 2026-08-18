#!/usr/bin/env python3
"""CPU/XLA initial-only RQMC mechanics campaign over active GenUT models."""

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
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import tensorflow as tf
import tensorflow_probability as tfp

from bayesfilter.highdim.cubature_genut_candidate import (
    cubature_design,
    gaussian_genut_design,
    replicate_positive_genut,
)
from bayesfilter.highdim.gaussian_cloud_designs_tf import (
    cloud_diagnostics,
    standard_normal_cloud,
)
from bayesfilter.highdim.ledh_pfpf_genut_initial_rqmc_tf import (
    finite_value_standard_score_initial_rqmc,
)
from bayesfilter.highdim.ledh_pfpf_genut_model_callbacks_tf import (
    LEDHGenUTModelCallbacks,
    austria_sir_callbacks,
    diagonal_lgssm_callbacks,
    exact_sv_callbacks,
    generalized_sv_callbacks,
    ksc_sv_callbacks,
    predator_prey_callbacks,
)
from bayesfilter.highdim.models import p30_predator_prey_fixture_model
from bayesfilter.highdim.sir_latent_preclip_tf import (
    latent_preclip_zhao_cui_sir_austria_model,
)
from bayesfilter.highdim.sqmc_tf import (
    ENDPOINT_POLICY_ID,
    POINT_SET_ID,
    randomized_halton_gaussian,
)
from bayesfilter.highdim.sv_mixture_cut4 import (
    exact_transformed_sv_independent_panel_fixed_sgqf_filter,
    exact_transformed_sv_independent_panel_fixed_sgqf_score,
    independent_panel_sv_mixture_fixed_sgqf_filter,
    independent_panel_sv_mixture_fixed_sgqf_score,
)


SCHEMA = "bayesfilter.ledh_pfpf_genut.initial_rqmc_all_models.v1"
PLAN = Path(
    "docs/plans/bayesfilter-genut-sqmc-particle-count-trust-region-plan-2026-08-17.md"
)
ARTIFACT_ROOT = Path(
    "docs/benchmarks/artifacts/ledh_pfpf_genut_initial_rqmc_all_models_20260806"
)
HORIZON = 6
PARTICLE_COUNT = 72
ARMS = ("iid_initial", "rqmc_initial")
SMOKE_SEEDS = (95100,)
PILOT_SEEDS = tuple(range(95101, 95117))
EPSILON = 2.0
SINKHORN_STEPS = 8
BALANCE_STEPS = 8
RIDGE = 1.0e-5
SOURCE_PATHS = (
    PLAN,
    Path("bayesfilter/highdim/ledh_pfpf_genut_model_callbacks_tf.py"),
    Path("bayesfilter/highdim/ledh_pfpf_genut_initial_rqmc_tf.py"),
    Path("bayesfilter/highdim/sqmc_tf.py"),
    Path("docs/benchmarks/run_ledh_pfpf_genut_initial_rqmc_all_models.py"),
)


@dataclass(frozen=True)
class CampaignModel:
    row_id: str
    callbacks: LEDHGenUTModelCallbacks
    theta: tf.Tensor
    observations: tf.Tensor
    raw_observations: tf.Tensor | None
    reference: dict[str, Any] | None
    event_order: str


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def _tensor_sha256(value: tf.Tensor) -> str:
    return hashlib.sha256(tf.io.serialize_tensor(value).numpy()).hexdigest()


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _configure_cpu_xla() -> dict[str, Any]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise RuntimeError("CPU campaign requires CUDA_VISIBLE_DEVICES=-1")
    if tf.config.list_physical_devices("GPU"):
        raise RuntimeError("CPU campaign found a visible GPU")
    tf.config.experimental.enable_tensor_float_32_execution(False)
    logical = tf.config.list_logical_devices("CPU")
    if not logical:
        raise RuntimeError("CPU campaign found no logical CPU")
    return {
        "cuda_visible_devices": "-1",
        "gpu_intentionally_hidden": True,
        "physical_devices": [item.name for item in tf.config.list_physical_devices()],
        "logical_cpu_devices": [item.name for item in logical],
        "device": "/device:CPU:0",
        "dtype": "float32",
        "jit_compile": True,
        "xla_platform": "Host",
        "tf32_execution_enabled": False,
        "trust_basis": "cpu_reference_gpu_intentionally_hidden",
        "production_target_status": "explicit_cpu_xla_reference_exception",
    }


def _output_directory(stage: str) -> Path:
    root = ROOT / ARTIFACT_ROOT
    root.mkdir(parents=True, exist_ok=True)
    for index in range(1, 100):
        path = root / f"{stage}_attempt{index:02d}"
        try:
            path.mkdir()
        except FileExistsError:
            continue
        return path
    raise RuntimeError("no unused attempt directory remains")


def _design(dimension: int, particle_count: int = PARTICLE_COUNT) -> tf.Tensor:
    if particle_count < 1:
        raise ValueError("particle_count must be positive")
    if dimension >= 18:
        return cubature_design(dim=dimension, num_particles=particle_count)
    return replicate_positive_genut(
        gaussian_genut_design(dim=dimension), num_particles=particle_count
    )


def _reference_payload(
    *, reference_id: str, role: str, value: tf.Tensor, score: tf.Tensor
) -> dict[str, Any]:
    finite = tf.math.is_finite(value) & tf.reduce_all(tf.math.is_finite(score))
    return {
        "reference_id": reference_id,
        "role": role,
        "value": float(value.numpy()),
        "score": [float(item) for item in score.numpy()],
        "finite": bool(finite.numpy()),
    }


def _lgssm_reference(theta: tf.Tensor, observations: tf.Tensor) -> dict[str, Any]:
    from docs.benchmarks.run_lgssm_cubature_genut_fp32 import _kalman_value_score

    value, score = _kalman_value_score(theta, observations)
    return _reference_payload(
        reference_id="exact_affine_kalman_analytical_physical_parameter_score",
        role="exact_same_target_accuracy_reference",
        value=value,
        score=score,
    )


def _sv_reference(
    *, exact: bool, theta: tf.Tensor, raw_observations: tf.Tensor
) -> dict[str, Any]:
    theta64 = tf.cast(theta, tf.float64)
    gamma = 0.5 * (
        tf.constant(1.0, tf.float64)
        + tf.math.erf(theta64[0] / tf.sqrt(tf.constant(2.0, tf.float64)))
    )
    beta = tf.exp(theta64[1])
    if exact:
        value_result = exact_transformed_sv_independent_panel_fixed_sgqf_filter(
            raw_observations, gamma=gamma, beta=beta, sigma=1.0, sparse_level=2
        )
        score_result = exact_transformed_sv_independent_panel_fixed_sgqf_score(
            raw_observations, gamma=gamma, beta=beta, sigma=1.0, sparse_level=2
        )
        reference_id = "fixed_sgqf_level2_exact_transformed_sv_manual_score"
    else:
        value_result = independent_panel_sv_mixture_fixed_sgqf_filter(
            raw_observations, gamma=gamma, beta=beta, sigma=1.0, sparse_level=2
        )
        score_result = independent_panel_sv_mixture_fixed_sgqf_score(
            raw_observations, gamma=gamma, beta=beta, sigma=1.0, sparse_level=2
        )
        reference_id = "fixed_sgqf_level2_ksc_mixture_manual_score"
    if score_result.score is None or score_result.log_likelihood is None:
        raise RuntimeError("SV reference failed to emit value and score")
    mismatch = abs(
        float(value_result.log_likelihood.numpy())
        - float(score_result.log_likelihood.numpy())
    )
    if mismatch > 1.0e-8:
        raise RuntimeError("SV reference value and score programs disagree")
    return _reference_payload(
        reference_id=reference_id,
        role="approximate_same_target_explanatory_reference",
        value=value_result.log_likelihood,
        score=score_result.score,
    )


def _generalized_reference(theta: tf.Tensor, observations: tf.Tensor) -> dict[str, Any]:
    from bayesfilter.highdim.generalized_sv_sgqf_tf import (
        generalized_sv_sgqf_value_score_status,
    )

    value, score, status = generalized_sv_sgqf_value_score_status(
        tf.cast(theta, tf.float64), tf.cast(observations, tf.float64), sparse_level=3
    )
    if int(status["status_code"].numpy()) != 0:
        raise RuntimeError("generalized-SV SGQF reference status veto")
    return _reference_payload(
        reference_id="fixed_sgqf_generalized_sv_raw_y_level3_manual_score",
        role="approximate_same_target_explanatory_reference",
        value=value,
        score=score,
    )


def _predator_reference(theta: tf.Tensor, observations: tf.Tensor) -> dict[str, Any]:
    del theta, observations
    return {
        "reference_id": "fixed_sgqf_level2_predator_prey_physical_manual_score",
        "role": "approximate_same_target_reference_unavailable_in_worktree",
        "finite": False,
        "reason": "optional symmetric-Sylvester custom op is absent in this worktree",
    }


def _sir_reference(theta: tf.Tensor, observations: tf.Tensor) -> dict[str, Any]:
    del theta, observations
    return {
        "reference_id": "fixed_sgqf_level2_axis_austria_sir_manual_score",
        "role": "approximate_same_target_reference_unavailable_in_worktree",
        "finite": False,
        "reason": "optional symmetric-Sylvester custom op is absent in this worktree",
    }


def build_campaign_models(*, include_references: bool) -> tuple[CampaignModel, ...]:
    from scripts.filtering_value_gradient_benchmark_generate_p8_datasets import (
        _generalized_sv_prior_mean_dataset,
        _lgssm_dataset,
        _sv_dataset,
    )
    from bayesfilter.highdim.sv_mixture_cut4 import (
        exact_transformed_sv_observations,
        transformed_sv_observations,
    )

    lg_payload = _lgssm_dataset(81100)
    lg_theta = tf.constant([0.72, 0.55, 0.35, 0.35, 0.45], tf.float32)
    lg_observations = tf.cast(lg_payload["observations"][:HORIZON], tf.float32)
    sv_payload = _sv_dataset(81101)
    sv_theta = tf.cast(sv_payload["truth_theta"], tf.float32)
    sv_raw = tf.cast(sv_payload["observations"][:HORIZON], tf.float64)
    exact_observations = tf.cast(
        exact_transformed_sv_observations(sv_raw), tf.float32
    )
    ksc_observations = tf.cast(
        transformed_sv_observations(sv_raw, offset=1.0e-8), tf.float32
    )
    generalized_payload = _generalized_sv_prior_mean_dataset(81105)
    generalized_theta = tf.cast(generalized_payload["truth_theta"], tf.float32)
    generalized_observations = tf.cast(
        generalized_payload["observations"][:HORIZON], tf.float32
    )
    pp_model = p30_predator_prey_fixture_model()
    _pp_states, pp_observations_full = pp_model.simulate(
        pp_model.true_parameters(), final_time=20, seed=81104
    )
    pp_observations_full = pp_observations_full[1:]
    pp_observations = tf.cast(pp_observations_full[:HORIZON], tf.float32)
    pp_theta = tf.cast(pp_model.true_parameters(), tf.float32)
    sir_model = latent_preclip_zhao_cui_sir_austria_model()
    _sir_states, sir_all_observations = sir_model.physical_model.base_model.simulate(
        final_time=20, seed=81120
    )
    sir_observations_full = sir_all_observations[1:]
    sir_observations = tf.cast(sir_observations_full[:HORIZON], tf.float32)
    sir_theta = tf.zeros([3], tf.float32)

    return (
        CampaignModel(
            "lgssm_T50",
            diagonal_lgssm_callbacks(),
            lg_theta,
            lg_observations,
            None,
            _lgssm_reference(lg_theta, lg_observations)
            if include_references
            else None,
            "stationary_initial_draw_then_observe_y0_then_transitions",
        ),
        CampaignModel(
            "ksc_sv_T10",
            ksc_sv_callbacks(),
            sv_theta,
            ksc_observations,
            sv_raw,
            _sv_reference(exact=False, theta=sv_theta, raw_observations=sv_raw)
            if include_references
            else None,
            "stationary_initial_draw_then_observe_y0_then_transitions",
        ),
        CampaignModel(
            "exact_sv_T10",
            exact_sv_callbacks(),
            sv_theta,
            exact_observations,
            sv_raw,
            _sv_reference(exact=True, theta=sv_theta, raw_observations=sv_raw)
            if include_references
            else None,
            "stationary_initial_draw_then_observe_y0_then_transitions",
        ),
        CampaignModel(
            "generalized_sv_T10",
            generalized_sv_callbacks(),
            generalized_theta,
            generalized_observations,
            generalized_observations,
            _generalized_reference(generalized_theta, generalized_observations)
            if include_references
            else None,
            "stationary_initial_draw_then_transition_before_every_observation",
        ),
        CampaignModel(
            "predator_prey_T20",
            predator_prey_callbacks(pp_model),
            pp_theta,
            pp_observations,
            None,
            _predator_reference(pp_theta, pp_observations)
            if include_references
            else None,
            "x0_then_transition_1_to_6_then_observe_y1_to_y6",
        ),
        CampaignModel(
            "austria_sir_T20",
            austria_sir_callbacks(sir_model),
            sir_theta,
            sir_observations,
            None,
            _sir_reference(sir_theta, sir_observations)
            if include_references
            else None,
            "x0_then_transition_1_to_6_then_observe_y1_to_y6",
        ),
    )


def make_evaluator(model: CampaignModel):
    d = model.callbacks.state_dimension
    o = model.callbacks.observation_dimension
    p = model.callbacks.parameter_count
    process_steps = (
        HORIZON
        if model.callbacks.transition_before_first_observation
        else HORIZON - 1
    )

    @tf.function(
        input_signature=(
            tf.TensorSpec([p], tf.float32),
            tf.TensorSpec([HORIZON, o], tf.float32),
            tf.TensorSpec([PARTICLE_COUNT, d], tf.float32),
            tf.TensorSpec([process_steps, PARTICLE_COUNT, d], tf.float32),
            tf.TensorSpec([PARTICLE_COUNT, d], tf.float32),
        ),
        jit_compile=True,
        reduce_retracing=True,
    )
    def evaluate(theta, observations, initial_noise, process_noise, design):
        with tf.device("/CPU:0"):
            return finite_value_standard_score_initial_rqmc(
                model.callbacks,
                theta,
                observations,
                initial_noise,
                process_noise,
                design,
                epsilon=EPSILON,
                sinkhorn_steps=SINKHORN_STEPS,
                balance_steps=BALANCE_STEPS,
                ridge=RIDGE,
            )

    return evaluate


def paired_inputs(
    model: CampaignModel, seed: int
) -> tuple[dict[str, tf.Tensor], tf.Tensor, dict[str, Any]]:
    d = model.callbacks.state_dimension
    process_steps = (
        HORIZON
        if model.callbacks.transition_before_first_observation
        else HORIZON - 1
    )
    initial = {
        "iid_initial": standard_normal_cloud(
            "iid_gaussian",
            num_particles=PARTICLE_COUNT,
            dimension=d,
            seed=seed,
            salt=101,
        ),
        "rqmc_initial": randomized_halton_gaussian(
            num_particles=PARTICLE_COUNT,
            dimension=d,
            seed=seed,
            salt=301,
        ),
    }
    process = tf.stack(
        [
            standard_normal_cloud(
                "iid_gaussian",
                num_particles=PARTICLE_COUNT,
                dimension=d,
                seed=seed,
                salt=1001 + time_index,
            )
            for time_index in range(process_steps)
        ]
    )
    return initial, process, {
        "process_noise_sha256": _tensor_sha256(process),
        "initial_noise_sha256": {
            arm: _tensor_sha256(value) for arm, value in initial.items()
        },
    }


def _run_row(
    model: CampaignModel,
    evaluator,
    *,
    arm: str,
    seed: int,
    initial: tf.Tensor,
    process: tf.Tensor,
    input_hashes: dict[str, Any],
) -> dict[str, Any]:
    design = _design(model.callbacks.state_dimension)
    started = time.perf_counter()
    value, score, diagnostics = evaluator(
        model.theta, model.observations, initial, process, design
    )
    elapsed = time.perf_counter() - started
    finite = bool(tf.math.is_finite(value).numpy()) and bool(
        tf.reduce_all(tf.math.is_finite(score)).numpy()
    )
    row: dict[str, Any] = {
        "model_id": model.row_id,
        "arm": arm,
        "seed": seed,
        "value": float(value.numpy()),
        "score": [float(item) for item in score.numpy()],
        "score_l2": float(tf.linalg.norm(score).numpy()),
        "finite": finite,
        "program_valid": bool(diagnostics["program_valid"].numpy()),
        "minimum_ess": float(tf.reduce_min(diagnostics["ess"]).numpy()),
        "maximum_normalized_weight": float(
            tf.reduce_max(diagnostics["maximum_normalized_weight"]).numpy()
        ),
        "maximum_reset_mean_residual": float(
            tf.reduce_max(diagnostics["reset_mean_residual"]).numpy()
        ),
        "elapsed_seconds": elapsed,
        "value_device": value.device,
        "process_noise_sha256": input_hashes["process_noise_sha256"],
        "initial_noise_sha256": input_hashes["initial_noise_sha256"][arm],
        "initial_cloud_diagnostics": {
            key: float(item.numpy())
            for key, item in cloud_diagnostics(initial).items()
        },
    }
    if model.reference is not None and model.reference.get("finite") is True:
        row["absolute_value_error"] = float(
            tf.abs(value - tf.cast(model.reference["value"], value.dtype)).numpy()
        )
        reference_score = tf.constant(model.reference["score"], score.dtype)
        row["score_l2_error"] = float(tf.linalg.norm(score - reference_score).numpy())
    if not finite or not row["program_valid"]:
        raise RuntimeError(f"nonfinite or invalid row: {model.row_id}/{arm}/{seed}")
    return row


def _mean_se(values: list[float]) -> dict[str, float]:
    mean = statistics.mean(values)
    variance = statistics.variance(values) if len(values) > 1 else 0.0
    return {
        "mean": mean,
        "standard_error": math.sqrt(variance / len(values)),
        "replicate_variance": variance,
    }


def _aggregate(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summaries: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    for model_id in sorted({row["model_id"] for row in rows}):
        model_rows = [row for row in rows if row["model_id"] == model_id]
        by_arm = {arm: [row for row in model_rows if row["arm"] == arm] for arm in ARMS}
        for arm, arm_rows in by_arm.items():
            summary = {
                "model_id": model_id,
                "arm": arm,
                "replications": len(arm_rows),
                "value": _mean_se([row["value"] for row in arm_rows]),
                "score_l2": _mean_se([row["score_l2"] for row in arm_rows]),
                "minimum_ess_mean": statistics.mean(
                    row["minimum_ess"] for row in arm_rows
                ),
                "maximum_weight_mean": statistics.mean(
                    row["maximum_normalized_weight"] for row in arm_rows
                ),
            }
            if "absolute_value_error" in arm_rows[0]:
                summary["absolute_value_error"] = _mean_se(
                    [row["absolute_value_error"] for row in arm_rows]
                )
                summary["score_l2_error"] = _mean_se(
                    [row["score_l2_error"] for row in arm_rows]
                )
            summaries.append(summary)
        iid = {row["seed"]: row for row in by_arm["iid_initial"]}
        rqmc = {row["seed"]: row for row in by_arm["rqmc_initial"]}
        if set(iid) != set(rqmc):
            raise RuntimeError(f"paired seeds differ for {model_id}")
        if any(
            iid[seed]["process_noise_sha256"] != rqmc[seed]["process_noise_sha256"]
            for seed in iid
        ):
            raise RuntimeError(f"paired transition noise differs for {model_id}")
        comparison: dict[str, Any] = {
            "model_id": model_id,
            "rqmc_minus_iid_value": _mean_se(
                [rqmc[seed]["value"] - iid[seed]["value"] for seed in sorted(iid)]
            ),
            "rqmc_minus_iid_score_l2": _mean_se(
                [
                    rqmc[seed]["score_l2"] - iid[seed]["score_l2"]
                    for seed in sorted(iid)
                ]
            ),
            "value_variance_ratio_rqmc_over_iid": (
                statistics.variance(rqmc[seed]["value"] for seed in iid)
                / statistics.variance(iid[seed]["value"] for seed in iid)
                if len(iid) > 1
                and statistics.variance(iid[seed]["value"] for seed in iid) > 0.0
                else None
            ),
            "score_l2_variance_ratio_rqmc_over_iid": (
                statistics.variance(rqmc[seed]["score_l2"] for seed in iid)
                / statistics.variance(iid[seed]["score_l2"] for seed in iid)
                if len(iid) > 1
                and statistics.variance(iid[seed]["score_l2"] for seed in iid) > 0.0
                else None
            ),
        }
        if "absolute_value_error" in next(iter(iid.values())):
            comparison["rqmc_minus_iid_absolute_value_error"] = _mean_se(
                [
                    rqmc[seed]["absolute_value_error"]
                    - iid[seed]["absolute_value_error"]
                    for seed in sorted(iid)
                ]
            )
            comparison["rqmc_minus_iid_score_l2_error"] = _mean_se(
                [
                    rqmc[seed]["score_l2_error"]
                    - iid[seed]["score_l2_error"]
                    for seed in sorted(iid)
                ]
            )
        comparisons.append(comparison)
    return summaries, comparisons


def _markdown(result: dict[str, Any]) -> str:
    def format_optional(value: float | None) -> str:
        return "N/A" if value is None else f"{value:.6g}"

    lines = [
        "# Initial-Only RQMC All-Model Mechanics Result",
        "",
        f"Status: `{result['status']}`",
        "",
        "All uncertainty values are standard errors over paired randomizations. "
        "Differences are descriptive, not confirmatory evidence.",
        "",
        "| Model | IID abs value error +/- SE | RQMC abs value error +/- SE | Paired error difference +/- SE | IID score error +/- SE | RQMC score error +/- SE | Paired score-error difference +/- SE |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    summaries = {
        (row["model_id"], row["arm"]): row for row in result["cell_summaries"]
    }
    comparisons = {row["model_id"]: row for row in result["paired_comparisons"]}
    for model_id in result["model_order"]:
        iid = summaries[(model_id, "iid_initial")]
        rqmc = summaries[(model_id, "rqmc_initial")]
        comparison = comparisons[model_id]
        if "absolute_value_error" not in iid:
            lines.append(f"| {model_id} | N/A | N/A | N/A | N/A | N/A | N/A |")
            continue
        value_difference = comparison["rqmc_minus_iid_absolute_value_error"]
        score_difference = comparison["rqmc_minus_iid_score_l2_error"]
        lines.append(
            f"| {model_id} | {iid['absolute_value_error']['mean']:.6g} +/- {iid['absolute_value_error']['standard_error']:.3g} | "
            f"{rqmc['absolute_value_error']['mean']:.6g} +/- {rqmc['absolute_value_error']['standard_error']:.3g} | "
            f"{value_difference['mean']:.6g} +/- {value_difference['standard_error']:.3g} | "
            f"{iid['score_l2_error']['mean']:.6g} +/- {iid['score_l2_error']['standard_error']:.3g} | "
            f"{rqmc['score_l2_error']['mean']:.6g} +/- {rqmc['score_l2_error']['standard_error']:.3g} | "
            f"{score_difference['mean']:.6g} +/- {score_difference['standard_error']:.3g} |"
        )
    lines.extend(
        [
            "",
            "| Model | Value variance ratio RQMC/IID | Score-norm variance ratio RQMC/IID |",
            "|---|---:|---:|",
        ]
    )
    for model_id in result["model_order"]:
        comparison = comparisons[model_id]
        lines.append(
            f"| {model_id} | {format_optional(comparison['value_variance_ratio_rqmc_over_iid'])} | "
            f"{format_optional(comparison['score_l2_variance_ratio_rqmc_over_iid'])} |"
        )
    lines.extend(
        [
            "",
            "The LGSSM reference is exact. Every other reference is an approximate "
            "same-target SGQF diagnostic and cannot establish exact error or superiority.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("smoke", "pilot"), required=True)
    args = parser.parse_args()
    seeds = SMOKE_SEEDS if args.stage == "smoke" else PILOT_SEEDS
    device = _configure_cpu_xla()
    output = _output_directory(args.stage)
    started = time.perf_counter()
    models = build_campaign_models(include_references=True)
    model_order = [model.row_id for model in models]
    manifest = {
        "schema": SCHEMA,
        "stage": args.stage,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty_paths": _git("status", "--short").splitlines(),
        "command": [sys.executable, *sys.argv],
        "working_directory": str(ROOT),
        "python": sys.version,
        "platform": platform.platform(),
        "tensorflow_version": tf.__version__,
        "tensorflow_probability_version": tfp.__version__,
        "environment": os.environ.get("CONDA_DEFAULT_ENV", "unknown"),
        "device_policy": device,
        "particle_count": PARTICLE_COUNT,
        "horizon": HORIZON,
        "seeds": list(seeds),
        "arms": list(ARMS),
        "models": [
            {
                "model_id": model.row_id,
                "state_dimension": model.callbacks.state_dimension,
                "parameter_count": model.callbacks.parameter_count,
                "observation_dimension": model.callbacks.observation_dimension,
                "event_order": model.event_order,
                "observation_sha256": _tensor_sha256(model.observations),
                "reference": model.reference,
            }
            for model in models
        ],
        "filter_controls": {
            "epsilon": EPSILON,
            "sinkhorn_steps": SINKHORN_STEPS,
            "balance_steps": BALANCE_STEPS,
            "ridge": RIDGE,
            "reset_status": "experimental_contract_e_chol_primal_not_canonical_identity",
            "tuning_status": "inherited_warm_start_not_scope_tuned",
        },
        "initialization": {
            "candidate_point_set_id": POINT_SET_ID,
            "endpoint_policy_id": ENDPOINT_POLICY_ID,
            "scope": "initial_only",
            "transition_noise": "paired_byte_identical_iid_stateless_normal",
        },
        "score_backend": {
            "definition": "repository_standard_pairwise_backward_filtering_score",
            "local_scores": "repository_model_analytical_parameter_scores",
            "autodiff": False,
            "handwritten_in_experiment_runner": False,
        },
        "plan": str(PLAN),
        "source_sha256": {str(path): _sha256(path) for path in SOURCE_PATHS},
        "output_directory": str(output.relative_to(ROOT)),
    }
    _write_json(output / "run_manifest.json", manifest)

    evaluators = {model.row_id: make_evaluator(model) for model in models}
    compile_records = []
    rows = []
    for model in models:
        evaluator = evaluators[model.row_id]
        warm_initial, warm_process, _ = paired_inputs(model, 95099)
        design = _design(model.callbacks.state_dimension)
        compile_started = time.perf_counter()
        value, score, diagnostics = evaluator(
            model.theta,
            model.observations,
            warm_initial["rqmc_initial"],
            warm_process,
            design,
        )
        concrete = evaluator.get_concrete_function()
        must_compile = concrete.function_def.attr.get("_XlaMustCompile")
        compile_records.append(
            {
                "model_id": model.row_id,
                "compile_and_first_call_seconds": time.perf_counter()
                - compile_started,
                "program_valid": bool(diagnostics["program_valid"].numpy()),
                "value_finite": bool(tf.math.is_finite(value).numpy()),
                "score_finite": bool(
                    tf.reduce_all(tf.math.is_finite(score)).numpy()
                ),
                "value_device": value.device,
                "xla_must_compile_attribute": bool(must_compile.b)
                if must_compile
                else None,
                "tracing_count": evaluator.experimental_get_tracing_count(),
            }
        )
        for seed in seeds:
            initial, process, hashes = paired_inputs(model, seed)
            for arm in ARMS:
                rows.append(
                    _run_row(
                        model,
                        evaluator,
                        arm=arm,
                        seed=seed,
                        initial=initial[arm],
                        process=process,
                        input_hashes=hashes,
                    )
                )

    expected_rows = len(models) * len(ARMS) * len(seeds)
    if len(rows) != expected_rows:
        raise RuntimeError("campaign row count mismatch")
    if any("CPU:0" not in row["value_device"] for row in rows):
        raise RuntimeError("a result was not placed on CPU")
    if any(record["xla_must_compile_attribute"] is not True for record in compile_records):
        raise RuntimeError("an evaluator lacks the XLA must-compile attribute")
    if any(record["tracing_count"] != 1 for record in compile_records):
        raise RuntimeError("an evaluator retraced")
    summaries, comparisons = _aggregate(rows)
    elapsed = time.perf_counter() - started
    status = "smoke_pass" if args.stage == "smoke" else "mechanics_pilot_pass_no_promotion"
    raw = {
        "schema": SCHEMA,
        "stage": args.stage,
        "expected_row_count": expected_rows,
        "rows": rows,
        "compile_records": compile_records,
        "references": {model.row_id: model.reference for model in models},
    }
    result = {
        "schema": SCHEMA,
        "status": status,
        "row_count": len(rows),
        "expected_row_count": expected_rows,
        "elapsed_seconds": elapsed,
        "model_order": model_order,
        "cell_summaries": summaries,
        "paired_comparisons": comparisons,
        "hard_veto_screen": "passed",
        "statistically_supported_ranking": "none_mechanics_pilot",
        "default_readiness": "not_evaluated",
        "canonical_admission": "ineligible_experimental_reset_route_identity",
        "not_concluded": [
            "unbiased likelihood",
            "formal RQMC rate through LEDH and Contract E",
            "statistical superiority",
            "universal initialization improvement",
            "canonical score admission",
            "default readiness",
            "GPU performance",
            "HMC readiness",
        ],
        "artifact_paths": {
            "raw": str((output / "raw.json").relative_to(ROOT)),
            "result": str((output / "result.json").relative_to(ROOT)),
            "markdown": str((output / "result.md").relative_to(ROOT)),
            "manifest": str((output / "run_manifest.json").relative_to(ROOT)),
        },
    }
    _write_json(output / "raw.json", raw)
    _write_json(output / "result.json", result)
    (output / "result.md").write_text(_markdown(result), encoding="utf-8")
    manifest.update(
        {
            "wall_time_seconds": elapsed,
            "row_count": len(rows),
            "status": status,
            "artifact_paths": result["artifact_paths"],
        }
    )
    _write_json(output / "run_manifest.json", manifest)
    print(json.dumps({"status": status, "output": str(output), "rows": len(rows)}))


if __name__ == "__main__":
    main()
