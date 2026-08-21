#!/usr/bin/env python3
"""CPU/XLA Full-SQMC campaign at each active model's complete horizon."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import tensorflow as tf
import tensorflow_probability as tfp

from bayesfilter.highdim.gaussian_cloud_designs_tf import standard_normal_cloud
from bayesfilter.highdim.ledh_pfpf_genut_initial_rqmc_tf import (
    finite_value_standard_score_initial_rqmc,
)
from bayesfilter.highdim.ledh_pfpf_genut_model_callbacks_tf import (
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
    ANCESTOR_CDF_ID,
    ENDPOINT_POLICY_ID,
    HILBERT_IMPLEMENTATION_ID,
    POINT_SET_ID,
    STATE_MAP_ID,
    randomized_halton_gaussian,
    randomized_halton_joint,
)
from bayesfilter.highdim.sv_mixture_cut4 import (
    exact_transformed_sv_observations,
    transformed_sv_observations,
)
from docs.benchmarks.run_ledh_pfpf_genut_full_sqmc_all_models import (
    ARMS,
    BALANCE_STEPS,
    EPSILON,
    HILBERT_BITS,
    PILOT_SEEDS,
    RIDGE,
    SINKHORN_STEPS,
    SMOKE_SEEDS,
    _run_row,
    aggregate,
    markdown,
    state_map,
)
from docs.benchmarks.run_ledh_pfpf_genut_initial_rqmc_all_models import (
    CampaignModel,
    PARTICLE_COUNT,
    _configure_cpu_xla,
    _design,
    _generalized_reference,
    _git,
    _lgssm_reference,
    _predator_reference,
    _sha256,
    _sir_reference,
    _sv_reference,
    _tensor_sha256,
    _write_json,
)


SCHEMA = "bayesfilter.ledh_pfpf_genut.full_sqmc_full_horizons.v1"
PLAN = Path(
    "docs/plans/bayesfilter-genut-sqmc-particle-count-trust-region-plan-2026-08-17.md"
)
ARTIFACT_ROOT = Path(
    "docs/benchmarks/artifacts/ledh_pfpf_genut_full_sqmc_full_horizons_20260806"
)
MODEL_HORIZONS = {
    "lgssm_T50": 50,
    "ksc_sv_T10": 10,
    "exact_sv_T10": 10,
    "generalized_sv_T10": 10,
    "predator_prey_T20": 20,
    "austria_sir_T20": 20,
}
SOURCE_PATHS = (
    PLAN,
    Path("bayesfilter/highdim/sqmc_tf.py"),
    Path("bayesfilter/highdim/ledh_pfpf_genut_initial_rqmc_tf.py"),
    Path("bayesfilter/highdim/ledh_pfpf_genut_model_callbacks_tf.py"),
    Path("docs/benchmarks/run_ledh_pfpf_genut_full_sqmc_all_models.py"),
    Path("docs/benchmarks/run_ledh_pfpf_genut_full_sqmc_full_horizons.py"),
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


def build_full_horizon_models(
    *, include_references: bool
) -> tuple[CampaignModel, ...]:
    from scripts.filtering_value_gradient_benchmark_generate_p8_datasets import (
        _generalized_sv_prior_mean_dataset,
        _lgssm_dataset,
        _sv_dataset,
    )

    lg_payload = _lgssm_dataset(81100)
    lg_theta = tf.constant([0.72, 0.55, 0.35, 0.35, 0.45], tf.float32)
    lg_observations = tf.cast(
        lg_payload["observations"][: MODEL_HORIZONS["lgssm_T50"]], tf.float32
    )

    sv_payload = _sv_dataset(81101)
    sv_theta = tf.cast(sv_payload["truth_theta"], tf.float32)
    sv_raw = tf.cast(
        sv_payload["observations"][: MODEL_HORIZONS["ksc_sv_T10"]],
        tf.float64,
    )
    exact_observations = tf.cast(
        exact_transformed_sv_observations(sv_raw), tf.float32
    )
    ksc_observations = tf.cast(
        transformed_sv_observations(sv_raw, offset=1.0e-8), tf.float32
    )

    generalized_payload = _generalized_sv_prior_mean_dataset(81105)
    generalized_theta = tf.cast(generalized_payload["truth_theta"], tf.float32)
    generalized_observations = tf.cast(
        generalized_payload["observations"][: MODEL_HORIZONS["generalized_sv_T10"]],
        tf.float32,
    )

    pp_model = p30_predator_prey_fixture_model()
    _pp_states, pp_all_observations = pp_model.simulate(
        pp_model.true_parameters(), final_time=20, seed=81104
    )
    pp_observations = tf.cast(
        pp_all_observations[1 : 1 + MODEL_HORIZONS["predator_prey_T20"]],
        tf.float32,
    )
    pp_theta = tf.cast(pp_model.true_parameters(), tf.float32)

    sir_model = latent_preclip_zhao_cui_sir_austria_model()
    _sir_states, sir_all_observations = sir_model.physical_model.base_model.simulate(
        final_time=20, seed=81120
    )
    sir_observations = tf.cast(
        sir_all_observations[1 : 1 + MODEL_HORIZONS["austria_sir_T20"]],
        tf.float32,
    )
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
            "stationary_x0_observe_y0_then_transitions_1_to_49",
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
            "stationary_x0_observe_y0_then_transitions_1_to_9",
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
            "stationary_x0_observe_y0_then_transitions_1_to_9",
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
            "stationary_x0_then_transitions_0_to_9_before_observations",
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
            "x0_then_transitions_1_to_20_then_observe_y1_to_y20",
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
            "x0_then_transitions_1_to_20_then_observe_y1_to_y20",
        ),
    )


def make_evaluator(model: CampaignModel, *, full_sqmc: bool) -> Callable[..., Any]:
    horizon = MODEL_HORIZONS[model.row_id]
    dimension = model.callbacks.state_dimension
    observation_dimension = model.callbacks.observation_dimension
    parameter_count = model.callbacks.parameter_count
    process_steps = (
        horizon
        if model.callbacks.transition_before_first_observation
        else horizon - 1
    )
    policy = "hilbert_inverse_cdf" if full_sqmc else "existing_one_to_one"

    @tf.function(
        input_signature=(
            tf.TensorSpec([parameter_count], tf.float32),
            tf.TensorSpec([horizon, observation_dimension], tf.float32),
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
                functional_time_loop=True,
                epsilon=EPSILON,
                sinkhorn_steps=SINKHORN_STEPS,
                balance_steps=BALANCE_STEPS,
                ridge=RIDGE,
            )

    return evaluate


def campaign_inputs(
    model: CampaignModel, seed: int, particle_count: int = PARTICLE_COUNT
) -> dict[str, dict[str, Any]]:
    if particle_count < 1:
        raise ValueError("particle_count must be positive")
    horizon = MODEL_HORIZONS[model.row_id]
    dimension = model.callbacks.state_dimension
    process_steps = (
        horizon
        if model.callbacks.transition_before_first_observation
        else horizon - 1
    )
    iid_initial = standard_normal_cloud(
        "iid_gaussian",
        num_particles=particle_count,
        dimension=dimension,
        seed=seed,
        salt=101,
    )
    rqmc_initial = randomized_halton_gaussian(
        num_particles=particle_count,
        dimension=dimension,
        seed=seed,
        salt=301,
    )
    iid_process = tf.stack(
        [
            standard_normal_cloud(
                "iid_gaussian",
                num_particles=particle_count,
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
            num_particles=particle_count,
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
        [process_steps, particle_count], tf.float32
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


def _chapter18b_ledger() -> dict[str, Any]:
    return {
        "model_id": "chapter18b_quadratic_structural",
        "target_id": "STR-UKF-five-probit-T100-structural-innovation-v1",
        "full_horizon": 100,
        "historical_particle_count": 1002,
        "historical_test_status": "previously_tested_T100_failed_candidate",
        "authoritative_result": (
            "docs/plans/bayesfilter-genut-str-ukf-nonfinite-root-cause-result-2026-07-22.md"
        ),
        "failure": (
            "one nonfinite claim seed after invalid transport/reset at t=64; "
            "tuning score instability and inadequate selection"
        ),
        "full_sqmc_standard_score_status": "not_implemented_not_executed",
        "reason": (
            "rank-one structural transition requires support-preserving "
            "innovation-space proposal; current Full-SQMC core assumes a "
            "nonsingular state-dimensional transition, and the existing "
            "structural GenUT score is a finite-program forward derivative "
            "rather than this campaign's standard all-parent filtering score"
        ),
        "artificial_state_noise_added": False,
        "score_substitution_used": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("smoke", "pilot"), required=True)
    args = parser.parse_args()
    seeds = SMOKE_SEEDS if args.stage == "smoke" else PILOT_SEEDS
    device = _configure_cpu_xla()
    output = _output_directory(args.stage)
    started = time.perf_counter()
    models = build_full_horizon_models(include_references=True)
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
        "seeds": list(seeds),
        "arms": list(ARMS),
        "model_horizons": MODEL_HORIZONS,
        "models": [
            {
                "model_id": model.row_id,
                "horizon": MODEL_HORIZONS[model.row_id],
                "state_dimension": model.callbacks.state_dimension,
                "parameter_count": model.callbacks.parameter_count,
                "observation_dimension": model.callbacks.observation_dimension,
                "event_order": model.event_order,
                "observation_sha256": _tensor_sha256(model.observations),
                "reference": model.reference,
            }
            for model in models
        ],
        "chapter18b": _chapter18b_ledger(),
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
            "tuning_status": "inherited_warm_start_not_full_horizon_scope_tuned",
        },
        "score_backend": {
            "definition": "repository_standard_pairwise_backward_filtering_score",
            "autodiff": False,
            "finite_difference": False,
            "handwritten_in_experiment_runner": False,
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
                    "horizon": MODEL_HORIZONS[model.row_id],
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
                row = _run_row(
                    model,
                    evaluators[(model.row_id, evaluator_kind)],
                    arm=arm,
                    seed=seed,
                    inputs=inputs_by_arm[arm],
                    location=location,
                    scale=scale,
                )
                row["horizon"] = MODEL_HORIZONS[model.row_id]
                rows.append(row)

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
        "full_horizon_smoke_pass"
        if args.stage == "smoke"
        else "full_horizon_mechanics_pilot_pass_no_promotion"
    )
    raw = {
        "schema": SCHEMA,
        "stage": args.stage,
        "expected_row_count": expected_rows,
        "rows": rows,
        "compile_records": compile_records,
        "references": {model.row_id: model.reference for model in models},
        "chapter18b": _chapter18b_ledger(),
    }
    result = {
        "schema": SCHEMA,
        "status": status,
        "row_count": len(rows),
        "expected_row_count": expected_rows,
        "elapsed_seconds": elapsed,
        "model_order": model_order,
        "model_horizons": MODEL_HORIZONS,
        "cell_summaries": cell_summaries,
        "paired_comparisons": paired_comparisons,
        "chapter18b": _chapter18b_ledger(),
        "hard_veto_screen": "passed",
        "statistically_supported_ranking": "none_descriptive_pilot",
        "descriptive_differences_only": True,
        "default_readiness": "not_evaluated",
        "canonical_admission": "ineligible_experimental_untuned_reset_route",
        "artifact_paths": {
            "raw": str((output / "raw.json").relative_to(ROOT)),
            "result": str((output / "result.json").relative_to(ROOT)),
            "markdown": str((output / "result.md").relative_to(ROOT)),
            "manifest": str((output / "run_manifest.json").relative_to(ROOT)),
        },
    }
    _write_json(output / "raw.json", raw)
    _write_json(output / "result.json", result)
    report = markdown(result).replace(
        "# Full-SQMC All-Model Mechanics Result",
        "# Full-SQMC Full-Horizon Mechanics Result",
    )
    report += (
        "\n## Chapter 18b\n\n"
        "Chapter 18b was previously tested at `T=100`, `N=1002`, but the "
        "candidate failed: one claim seed became nonfinite after an invalid "
        "transport/reset at `t=64`, and tuning showed severe score instability. "
        "It was not executed as Full SQMC here because the current core is not "
        "support-correct for its rank-one structural transition and lacks the "
        "required standard-score route.\n"
    )
    (output / "result.md").write_text(report, encoding="utf-8")
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
