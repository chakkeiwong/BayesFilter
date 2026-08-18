#!/usr/bin/env python3
"""CPU/XLA Full-SQMC mechanics campaign over the active GenUT models."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import tensorflow as tf
import tensorflow_probability as tfp

from bayesfilter.highdim.gaussian_cloud_designs_tf import (
    cloud_diagnostics,
    standard_normal_cloud,
)
from bayesfilter.highdim.ledh_pfpf_genut_initial_rqmc_tf import (
    finite_value_standard_score_initial_rqmc,
)
from bayesfilter.highdim.sqmc_tf import (
    ANCESTOR_CDF_ID,
    ENDPOINT_POLICY_ID,
    HILBERT_IMPLEMENTATION_ID,
    POINT_SET_ID,
    STATE_MAP_ID,
    randomized_halton_gaussian,
    randomized_halton_joint,
)
from docs.benchmarks.run_ledh_pfpf_genut_initial_rqmc_all_models import (
    CampaignModel,
    HORIZON,
    PARTICLE_COUNT,
    _configure_cpu_xla,
    _design,
    _git,
    _sha256,
    _tensor_sha256,
    _write_json,
    build_campaign_models,
)


SCHEMA = "bayesfilter.ledh_pfpf_genut.full_sqmc_all_models.v1"
PLAN = Path(
    "docs/plans/bayesfilter-genut-sqmc-particle-count-trust-region-plan-2026-08-17.md"
)
ARTIFACT_ROOT = Path(
    "docs/benchmarks/artifacts/ledh_pfpf_genut_full_sqmc_all_models_20260806"
)
ARMS = ("iid_existing", "initial_rqmc", "full_sqmc_halton")
SMOKE_SEEDS = (95100,)
PILOT_SEEDS = tuple(range(95101, 95117))
HILBERT_BITS = 12
EPSILON = 2.0
SINKHORN_STEPS = 8
BALANCE_STEPS = 8
RIDGE = 1.0e-5
SOURCE_PATHS = (
    PLAN,
    Path("bayesfilter/highdim/sqmc_tf.py"),
    Path("bayesfilter/highdim/ledh_pfpf_genut_initial_rqmc_tf.py"),
    Path("bayesfilter/highdim/ledh_pfpf_genut_model_callbacks_tf.py"),
    Path("docs/benchmarks/run_ledh_pfpf_genut_initial_rqmc_all_models.py"),
    Path("docs/benchmarks/run_ledh_pfpf_genut_full_sqmc_all_models.py"),
)


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


def state_map(model: CampaignModel) -> tuple[tf.Tensor, tf.Tensor]:
    dimension = model.callbacks.state_dimension
    zero_noise = tf.zeros([1, dimension], tf.float32)
    location = model.callbacks.push_adapter.initial_value(
        model.theta, zero_noise
    )[0]
    scale = tf.sqrt(
        tf.linalg.diag_part(model.callbacks.initial_covariance(model.theta))
    )
    tf.debugging.assert_all_finite(location, "state-map location")
    tf.debugging.assert_positive(scale)
    return location, scale


def make_evaluator(model: CampaignModel, *, full_sqmc: bool) -> Callable[..., Any]:
    dimension = model.callbacks.state_dimension
    observation_dimension = model.callbacks.observation_dimension
    parameter_count = model.callbacks.parameter_count
    process_steps = (
        HORIZON
        if model.callbacks.transition_before_first_observation
        else HORIZON - 1
    )
    policy = "hilbert_inverse_cdf" if full_sqmc else "existing_one_to_one"

    @tf.function(
        input_signature=(
            tf.TensorSpec([parameter_count], tf.float32),
            tf.TensorSpec([HORIZON, observation_dimension], tf.float32),
            tf.TensorSpec([PARTICLE_COUNT, dimension], tf.float32),
            tf.TensorSpec(
                [process_steps, PARTICLE_COUNT, dimension], tf.float32
            ),
            tf.TensorSpec([process_steps, PARTICLE_COUNT], tf.float32),
            tf.TensorSpec([dimension], tf.float32),
            tf.TensorSpec([dimension], tf.float32),
            tf.TensorSpec([PARTICLE_COUNT, dimension], tf.float32),
        ),
        jit_compile=True,
        autograph=False,
    )
    def evaluate(
        theta,
        observations,
        initial_noise,
        process_noise,
        ancestor_uniforms,
        location,
        scale,
        design,
    ):
        with tf.device("/CPU:0"):
            return finite_value_standard_score_initial_rqmc(
                model.callbacks,
                theta,
                observations,
                initial_noise,
                process_noise,
                design,
                ancestry_policy=policy,
                process_ancestor_uniforms=ancestor_uniforms,
                state_map_location=location,
                state_map_scale=scale,
                hilbert_bits=HILBERT_BITS,
                epsilon=EPSILON,
                sinkhorn_steps=SINKHORN_STEPS,
                balance_steps=BALANCE_STEPS,
                ridge=RIDGE,
            )

    return evaluate


def campaign_inputs(model: CampaignModel, seed: int) -> dict[str, dict[str, Any]]:
    dimension = model.callbacks.state_dimension
    process_steps = (
        HORIZON
        if model.callbacks.transition_before_first_observation
        else HORIZON - 1
    )
    iid_initial = standard_normal_cloud(
        "iid_gaussian",
        num_particles=PARTICLE_COUNT,
        dimension=dimension,
        seed=seed,
        salt=101,
    )
    rqmc_initial = randomized_halton_gaussian(
        num_particles=PARTICLE_COUNT,
        dimension=dimension,
        seed=seed,
        salt=301,
    )
    iid_process = tf.stack(
        [
            standard_normal_cloud(
                "iid_gaussian",
                num_particles=PARTICLE_COUNT,
                dimension=dimension,
                seed=seed,
                salt=1001 + time_index,
            )
            for time_index in range(process_steps)
        ]
    )
    full_process_rows: list[tf.Tensor] = []
    full_ancestor_rows: list[tf.Tensor] = []
    raw_joint_hashes: list[str] = []
    for time_index in range(process_steps):
        raw, ancestors, innovations = randomized_halton_joint(
            num_particles=PARTICLE_COUNT,
            state_dimension=dimension,
            seed=seed,
            salt=3001 + time_index,
        )
        full_process_rows.append(tf.math.ndtri(innovations))
        full_ancestor_rows.append(ancestors)
        raw_joint_hashes.append(_tensor_sha256(raw))
    full_process = tf.stack(full_process_rows)
    full_ancestors = tf.stack(full_ancestor_rows)
    ignored_ancestors = tf.zeros(
        [process_steps, PARTICLE_COUNT], tf.float32
    )
    iid_process_hash = _tensor_sha256(iid_process)
    return {
        "iid_existing": {
            "initial": iid_initial,
            "process": iid_process,
            "ancestors": ignored_ancestors,
            "initial_sha256": _tensor_sha256(iid_initial),
            "process_sha256": iid_process_hash,
            "ancestor_sha256": None,
            "raw_joint_sha256": [],
        },
        "initial_rqmc": {
            "initial": rqmc_initial,
            "process": iid_process,
            "ancestors": ignored_ancestors,
            "initial_sha256": _tensor_sha256(rqmc_initial),
            "process_sha256": iid_process_hash,
            "ancestor_sha256": None,
            "raw_joint_sha256": [],
        },
        "full_sqmc_halton": {
            "initial": rqmc_initial,
            "process": full_process,
            "ancestors": full_ancestors,
            "initial_sha256": _tensor_sha256(rqmc_initial),
            "process_sha256": _tensor_sha256(full_process),
            "ancestor_sha256": _tensor_sha256(full_ancestors),
            "raw_joint_sha256": raw_joint_hashes,
        },
    }


def _run_row(
    model: CampaignModel,
    evaluator: Callable[..., Any],
    *,
    arm: str,
    seed: int,
    inputs: dict[str, Any],
    location: tf.Tensor,
    scale: tf.Tensor,
) -> dict[str, Any]:
    started = time.perf_counter()
    value, score, diagnostics = evaluator(
        model.theta,
        model.observations,
        inputs["initial"],
        inputs["process"],
        inputs["ancestors"],
        location,
        scale,
        _design(model.callbacks.state_dimension),
    )
    elapsed = time.perf_counter() - started
    finite = bool(tf.math.is_finite(value).numpy()) and bool(
        tf.reduce_all(tf.math.is_finite(score)).numpy()
    )
    valid = bool(diagnostics["program_valid"].numpy())
    maximum_saturation = float(
        tf.reduce_max(diagnostics["state_map_saturation_rate"]).numpy()
    )
    row: dict[str, Any] = {
        "model_id": model.row_id,
        "arm": arm,
        "seed": seed,
        "value": float(value.numpy()),
        "score": [float(item) for item in score.numpy()],
        "score_l2": float(tf.linalg.norm(score).numpy()),
        "finite": finite,
        "program_valid": valid,
        "minimum_ess": float(tf.reduce_min(diagnostics["ess"]).numpy()),
        "maximum_normalized_weight": float(
            tf.reduce_max(diagnostics["maximum_normalized_weight"]).numpy()
        ),
        "maximum_reset_mean_residual": float(
            tf.reduce_max(diagnostics["reset_mean_residual"]).numpy()
        ),
        "minimum_unique_ancestor_count": int(
            tf.reduce_min(diagnostics["ancestry_unique_count"]).numpy()
        ),
        "maximum_hilbert_tie_count": int(
            tf.reduce_max(diagnostics["hilbert_tie_count"]).numpy()
        ),
        "maximum_state_map_saturation_rate": maximum_saturation,
        "elapsed_seconds": elapsed,
        "value_device": value.device,
        "initial_sha256": inputs["initial_sha256"],
        "process_sha256": inputs["process_sha256"],
        "ancestor_sha256": inputs["ancestor_sha256"],
        "raw_joint_sha256": inputs["raw_joint_sha256"],
        "initial_cloud_diagnostics": {
            key: float(item.numpy())
            for key, item in cloud_diagnostics(inputs["initial"]).items()
        },
    }
    if model.reference is not None and model.reference.get("finite") is True:
        reference_value = tf.cast(model.reference["value"], value.dtype)
        reference_score = tf.constant(model.reference["score"], score.dtype)
        row["absolute_value_error"] = float(
            tf.abs(value - reference_value).numpy()
        )
        row["score_l2_error"] = float(
            tf.linalg.norm(score - reference_score).numpy()
        )
    if not finite or not valid:
        raise RuntimeError(f"nonfinite or invalid row: {model.row_id}/{arm}/{seed}")
    if maximum_saturation != 0.0:
        raise RuntimeError(f"state-map saturation: {model.row_id}/{arm}/{seed}")
    return row


def _mean_se(values: list[float]) -> dict[str, float]:
    variance = statistics.variance(values) if len(values) > 1 else 0.0
    return {
        "mean": statistics.fmean(values),
        "standard_error": math.sqrt(variance / len(values)),
        "replicate_variance": variance,
    }


def _score_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    parameter_count = len(rows[0]["score"])
    components = [
        _mean_se([row["score"][index] for row in rows])
        for index in range(parameter_count)
    ]
    return {
        "components": components,
        "total_componentwise_replicate_variance": sum(
            item["replicate_variance"] for item in components
        ),
        "l2_norm": _mean_se([row["score_l2"] for row in rows]),
    }


def _paired_field(
    candidate: dict[int, dict[str, Any]],
    comparator: dict[int, dict[str, Any]],
    field: str,
) -> dict[str, float]:
    return _mean_se(
        [
            candidate[seed][field] - comparator[seed][field]
            for seed in sorted(candidate)
        ]
    )


def _paired_score_components(
    candidate: dict[int, dict[str, Any]],
    comparator: dict[int, dict[str, Any]],
) -> list[dict[str, float]]:
    parameter_count = len(next(iter(candidate.values()))["score"])
    return [
        _mean_se(
            [
                candidate[seed]["score"][index]
                - comparator[seed]["score"][index]
                for seed in sorted(candidate)
            ]
        )
        for index in range(parameter_count)
    ]


def _ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator > 0.0 else None


def aggregate(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summaries: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    for model_id in sorted({row["model_id"] for row in rows}):
        model_rows = [row for row in rows if row["model_id"] == model_id]
        by_arm_rows = {
            arm: [row for row in model_rows if row["arm"] == arm]
            for arm in ARMS
        }
        arm_summaries: dict[str, dict[str, Any]] = {}
        for arm, arm_rows in by_arm_rows.items():
            summary: dict[str, Any] = {
                "model_id": model_id,
                "arm": arm,
                "replications": len(arm_rows),
                "value": _mean_se([row["value"] for row in arm_rows]),
                "score": _score_summary(arm_rows),
                "minimum_ess": _mean_se(
                    [row["minimum_ess"] for row in arm_rows]
                ),
                "minimum_unique_ancestor_count": _mean_se(
                    [float(row["minimum_unique_ancestor_count"]) for row in arm_rows]
                ),
                "elapsed_seconds": _mean_se(
                    [row["elapsed_seconds"] for row in arm_rows]
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
            arm_summaries[arm] = summary

        indexed = {
            arm: {row["seed"]: row for row in arm_rows}
            for arm, arm_rows in by_arm_rows.items()
        }
        if any(set(rows_by_seed) != set(indexed[ARMS[0]]) for rows_by_seed in indexed.values()):
            raise RuntimeError(f"paired seeds differ for {model_id}")
        if any(
            indexed["iid_existing"][seed]["process_sha256"]
            != indexed["initial_rqmc"][seed]["process_sha256"]
            for seed in indexed["iid_existing"]
        ):
            raise RuntimeError(f"IID transition inputs differ for {model_id}")
        if any(
            indexed["initial_rqmc"][seed]["initial_sha256"]
            != indexed["full_sqmc_halton"][seed]["initial_sha256"]
            for seed in indexed["iid_existing"]
        ):
            raise RuntimeError(f"RQMC initial inputs differ for {model_id}")

        iid_summary = arm_summaries["iid_existing"]
        comparison: dict[str, Any] = {
            "model_id": model_id,
            "variance_ratios_over_iid": {},
            "paired_full_minus_iid": {},
            "paired_full_minus_initial_rqmc": {},
        }
        for arm in ("initial_rqmc", "full_sqmc_halton"):
            comparison["variance_ratios_over_iid"][arm] = {
                "value": _ratio(
                    arm_summaries[arm]["value"]["replicate_variance"],
                    iid_summary["value"]["replicate_variance"],
                ),
                "total_score": _ratio(
                    arm_summaries[arm]["score"][
                        "total_componentwise_replicate_variance"
                    ],
                    iid_summary["score"][
                        "total_componentwise_replicate_variance"
                    ],
                ),
            }
        full = indexed["full_sqmc_halton"]
        for comparator_arm, output_key in (
            ("iid_existing", "paired_full_minus_iid"),
            ("initial_rqmc", "paired_full_minus_initial_rqmc"),
        ):
            comparator = indexed[comparator_arm]
            paired = {
                "value": _paired_field(full, comparator, "value"),
                "score_l2": _paired_field(full, comparator, "score_l2"),
                "score_components": _paired_score_components(full, comparator),
            }
            if "absolute_value_error" in next(iter(full.values())):
                paired["absolute_value_error"] = _paired_field(
                    full, comparator, "absolute_value_error"
                )
                paired["score_l2_error"] = _paired_field(
                    full, comparator, "score_l2_error"
                )
            comparison[output_key] = paired
        comparisons.append(comparison)
    return summaries, comparisons


def _formatted(summary: dict[str, float]) -> str:
    return f"{summary['mean']:.6g} +/- {summary['standard_error']:.3g}"


def _optional(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.4g}"


def markdown(result: dict[str, Any]) -> str:
    summaries = {
        (item["model_id"], item["arm"]): item
        for item in result["cell_summaries"]
    }
    comparisons = {
        item["model_id"]: item for item in result["paired_comparisons"]
    }
    lines = [
        "# Full-SQMC All-Model Mechanics Result",
        "",
        f"Status: `{result['status']}`",
        "",
        "Ratios below are replicate variance ratios. Total score variance is the "
        "sum of the componentwise score variances, not variance of the score norm.",
        "",
        "| Model | Initial RQMC value/IID | Full SQMC value/IID | Initial RQMC total score/IID | Full SQMC total score/IID |",
        "|---|---:|---:|---:|---:|",
    ]
    for model_id in result["model_order"]:
        ratios = comparisons[model_id]["variance_ratios_over_iid"]
        initial = ratios["initial_rqmc"]
        full = ratios["full_sqmc_halton"]
        lines.append(
            f"| {model_id} | {_optional(initial['value'])} | "
            f"{_optional(full['value'])} | {_optional(initial['total_score'])} | "
            f"{_optional(full['total_score'])} |"
        )
    lines.extend(
        [
            "",
            "| Model | Full-IID abs-value-error difference | Full-initial-RQMC abs-value-error difference | Full-IID score-error difference | Full-initial-RQMC score-error difference |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for model_id in result["model_order"]:
        comparison = comparisons[model_id]
        versus_iid = comparison["paired_full_minus_iid"]
        versus_initial = comparison["paired_full_minus_initial_rqmc"]
        if "absolute_value_error" not in versus_iid:
            lines.append(f"| {model_id} | N/A | N/A | N/A | N/A |")
        else:
            lines.append(
                f"| {model_id} | {_formatted(versus_iid['absolute_value_error'])} | "
                f"{_formatted(versus_initial['absolute_value_error'])} | "
                f"{_formatted(versus_iid['score_l2_error'])} | "
                f"{_formatted(versus_initial['score_l2_error'])} |"
            )
    lines.extend(
        [
            "",
            "Negative error differences are descriptively favorable to Full SQMC. "
            "With 16 seeds these are pilot statistics; no ranking is statistically supported.",
            "",
            "The LGSSM accuracy reference is exact. SV references are approximate "
            "same-target SGQF diagnostics. Predator-prey and Austria-SIR references "
            "are unavailable in this worktree.",
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
    manifest: dict[str, Any] = {
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
        "sqmc": {
            "point_set_id": POINT_SET_ID,
            "endpoint_policy_id": ENDPOINT_POLICY_ID,
            "hilbert_implementation_id": HILBERT_IMPLEMENTATION_ID,
            "state_map_id": STATE_MAP_ID,
            "ancestor_cdf_id": ANCESTOR_CDF_ID,
            "hilbert_bits_per_coordinate": HILBERT_BITS,
            "ordering_coordinates": "complete_state_no_projection",
        },
        "filter_controls": {
            "epsilon": EPSILON,
            "sinkhorn_steps": SINKHORN_STEPS,
            "balance_steps": BALANCE_STEPS,
            "ridge": RIDGE,
            "reset_status": "experimental_contract_e_chol_primal_not_canonical_identity",
            "tuning_status": "inherited_warm_start_not_scope_tuned",
        },
        "score_backend": {
            "definition": "repository_standard_pairwise_backward_filtering_score",
            "local_scores": "repository_model_analytical_parameter_scores",
            "autodiff": False,
            "finite_difference": False,
            "handwritten_in_experiment_runner": False,
            "selected_ancestor_replaces_all_parent_recursion": False,
        },
        "plan": str(PLAN),
        "source_sha256": {str(path): _sha256(path) for path in SOURCE_PATHS},
        "output_directory": str(output.relative_to(ROOT)),
    }
    _write_json(output / "run_manifest.json", manifest)

    evaluators: dict[tuple[str, str], Callable[..., Any]] = {}
    for model in models:
        evaluators[(model.row_id, "existing")] = make_evaluator(
            model, full_sqmc=False
        )
        evaluators[(model.row_id, "full_sqmc")] = make_evaluator(
            model, full_sqmc=True
        )

    compile_records: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for model in models:
        location, scale = state_map(model)
        warm_inputs = campaign_inputs(model, 95099)
        for evaluator_kind, warm_arm in (
            ("existing", "initial_rqmc"),
            ("full_sqmc", "full_sqmc_halton"),
        ):
            evaluator = evaluators[(model.row_id, evaluator_kind)]
            warm = warm_inputs[warm_arm]
            compile_started = time.perf_counter()
            value, score, diagnostics = evaluator(
                model.theta,
                model.observations,
                warm["initial"],
                warm["process"],
                warm["ancestors"],
                location,
                scale,
                _design(model.callbacks.state_dimension),
            )
            concrete = evaluator.get_concrete_function()
            must_compile = concrete.function_def.attr.get("_XlaMustCompile")
            compile_records.append(
                {
                    "model_id": model.row_id,
                    "evaluator_kind": evaluator_kind,
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
            inputs_by_arm = campaign_inputs(model, seed)
            for arm in ARMS:
                evaluator_kind = (
                    "full_sqmc" if arm == "full_sqmc_halton" else "existing"
                )
                rows.append(
                    _run_row(
                        model,
                        evaluators[(model.row_id, evaluator_kind)],
                        arm=arm,
                        seed=seed,
                        inputs=inputs_by_arm[arm],
                        location=location,
                        scale=scale,
                    )
                )

    expected_rows = len(models) * len(ARMS) * len(seeds)
    if len(rows) != expected_rows:
        raise RuntimeError("campaign row count mismatch")
    if any("CPU:0" not in row["value_device"] for row in rows):
        raise RuntimeError("a result was not placed on CPU")
    if any(
        record["xla_must_compile_attribute"] is not True
        for record in compile_records
    ):
        raise RuntimeError("an evaluator lacks the XLA must-compile attribute")
    if any(record["tracing_count"] != 1 for record in compile_records):
        raise RuntimeError("an evaluator retraced")
    cell_summaries, paired_comparisons = aggregate(rows)
    elapsed = time.perf_counter() - started
    status = (
        "smoke_pass"
        if args.stage == "smoke"
        else "mechanics_pilot_pass_no_promotion"
    )
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
        "cell_summaries": cell_summaries,
        "paired_comparisons": paired_comparisons,
        "hard_veto_screen": "passed",
        "statistically_supported_ranking": "none_mechanics_pilot",
        "descriptive_differences_only": True,
        "default_readiness": "not_evaluated",
        "canonical_admission": "ineligible_experimental_reset_route_identity",
        "not_concluded": [
            "unbiased likelihood",
            "formal SQMC rate through LEDH and Contract E",
            "causal attribution to inverse-CDF ancestry",
            "statistical superiority",
            "universal Full-SQMC improvement",
            "full-horizon behavior",
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
    (output / "result.md").write_text(markdown(result), encoding="utf-8")
    manifest.update(
        {
            "wall_time_seconds": elapsed,
            "row_count": len(rows),
            "compile_records": compile_records,
            "status": status,
            "artifact_paths": result["artifact_paths"],
        }
    )
    _write_json(output / "run_manifest.json", manifest)
    print(
        json.dumps(
            {"status": status, "output": str(output), "rows": len(rows)}
        )
    )


if __name__ == "__main__":
    main()
