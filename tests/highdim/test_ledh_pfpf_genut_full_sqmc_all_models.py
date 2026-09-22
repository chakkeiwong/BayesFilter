from __future__ import annotations

import inspect
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import tensorflow as tf

from bayesfilter.highdim import ledh_pfpf_genut_initial_rqmc_tf as core
from docs.benchmarks.run_ledh_pfpf_genut_full_sqmc_all_models import (
    ARMS,
    HORIZON,
    PARTICLE_COUNT,
    _design,
    aggregate,
    campaign_inputs,
    make_evaluator,
    state_map,
)
from docs.benchmarks.run_ledh_pfpf_genut_initial_rqmc_all_models import (
    build_campaign_models,
)


def test_full_sqmc_input_pairing_and_shapes_for_all_models() -> None:
    for model in build_campaign_models(include_references=False):
        inputs = campaign_inputs(model, 95123)
        assert tuple(inputs) == ARMS
        dimension = model.callbacks.state_dimension
        process_steps = (
            HORIZON
            if model.callbacks.transition_before_first_observation
            else HORIZON - 1
        )
        for arm in ARMS:
            assert inputs[arm]["initial"].shape == (PARTICLE_COUNT, dimension)
            assert inputs[arm]["process"].shape == (
                process_steps,
                PARTICLE_COUNT,
                dimension,
            )
            assert inputs[arm]["ancestors"].shape == (
                process_steps,
                PARTICLE_COUNT,
            )
        assert (
            inputs["iid_existing"]["process_sha256"]
            == inputs["initial_rqmc"]["process_sha256"]
        )
        assert (
            inputs["initial_rqmc"]["initial_sha256"]
            == inputs["full_sqmc_halton"]["initial_sha256"]
        )
        assert len(inputs["full_sqmc_halton"]["raw_joint_sha256"]) == process_steps


def test_full_sqmc_ancestor_selection_is_record_permutation_equivariant() -> None:
    particles = tf.constant(
        ((-1.2, 0.4), (0.5, 1.1), (1.6, -0.8), (-0.3, -1.5)),
        tf.float32,
    )
    weights = tf.constant((0.1, 0.2, 0.3, 0.4), tf.float32)
    uniforms = tf.constant((0.05, 0.25, 0.55, 0.95), tf.float32)
    arguments = {
        "ancestry_policy": "hilbert_inverse_cdf",
        "state_map_location": tf.zeros([2]),
        "state_map_scale": tf.ones([2]),
        "hilbert_bits": 12,
    }
    first = core._transition_ancestors(  # noqa: SLF001
        particles, weights, uniforms, **arguments
    )
    permutation = tf.constant((2, 0, 3, 1), tf.int32)
    second = core._transition_ancestors(  # noqa: SLF001
        tf.gather(particles, permutation),
        tf.gather(weights, permutation),
        uniforms,
        **arguments,
    )
    tf.debugging.assert_equal(first["flow_ancestors"], second["flow_ancestors"])


def test_state_maps_use_all_model_state_coordinates() -> None:
    for model in build_campaign_models(include_references=False):
        location, scale = state_map(model)
        assert location.shape == (model.callbacks.state_dimension,)
        assert scale.shape == (model.callbacks.state_dimension,)
        tf.debugging.assert_all_finite(location, "location")
        tf.debugging.assert_positive(scale)


def test_core_uses_standard_score_and_has_no_runner_score_substitute() -> None:
    source = inspect.getsource(core)
    assert "_target_model_progressive_score_marks" in source
    assert "standard_pairwise_backward_marks" in source
    assert "flow_ancestors" in source
    for forbidden in (
        "GradientTape",
        "ForwardAccumulator",
        "finite_difference",
        "importance_ratio_w_over_r",
        "ancestor_proposal_probability",
    ):
        assert forbidden not in source


def test_lgssm_full_sqmc_core_compiles_on_cpu_xla() -> None:
    if tf.config.list_physical_devices("GPU"):
        raise RuntimeError("test requires CUDA_VISIBLE_DEVICES=-1 before import")
    model = build_campaign_models(include_references=False)[0]
    evaluator = make_evaluator(model, full_sqmc=True)
    inputs = campaign_inputs(model, 95100)["full_sqmc_halton"]
    location, scale = state_map(model)
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
    concrete = evaluator.get_concrete_function()
    must_compile = concrete.function_def.attr.get("_XlaMustCompile")
    assert must_compile is not None and must_compile.b
    assert evaluator.experimental_get_tracing_count() == 1
    assert "CPU:0" in value.device
    assert bool(diagnostics["program_valid"].numpy())
    tf.debugging.assert_all_finite(value, "value")
    tf.debugging.assert_all_finite(score, "score")
    assert int(tf.reduce_min(diagnostics["ancestry_unique_count"]).numpy()) > 0


def test_aggregate_uses_componentwise_total_score_variance() -> None:
    rows = []
    scores = {
        "iid_existing": ((0.0, 0.0), (2.0, 4.0)),
        "initial_rqmc": ((0.0, 0.0), (1.0, 2.0)),
        "full_sqmc_halton": ((0.0, 0.0), (0.5, 1.0)),
    }
    for arm, arm_scores in scores.items():
        for seed, score in enumerate(arm_scores):
            rows.append(
                {
                    "model_id": "fixture",
                    "arm": arm,
                    "seed": seed,
                    "value": float(score[0]),
                    "score": list(score),
                    "score_l2": float(tf.linalg.norm(score).numpy()),
                    "minimum_ess": 1.0,
                    "minimum_unique_ancestor_count": 1,
                    "elapsed_seconds": 1.0,
                    "process_sha256": "iid" if arm != "full_sqmc_halton" else "full",
                    "initial_sha256": "iid" if arm == "iid_existing" else "rqmc",
                }
            )
    _, comparisons = aggregate(rows)
    ratios = comparisons[0]["variance_ratios_over_iid"]
    assert ratios["initial_rqmc"]["total_score"] == 0.25
    assert ratios["full_sqmc_halton"]["total_score"] == 0.0625

