from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import tensorflow as tf

from bayesfilter.highdim import ledh_contract_e_tp_lgssm_tf as lgssm_score

from bayesfilter.highdim.ledh_pfpf_genut_initial_rqmc_tf import (
    standard_pairwise_backward_marks,
)
from bayesfilter.highdim.ledh_pfpf_genut_model_callbacks_tf import (
    diagonal_lgssm_callbacks,
)
from docs.benchmarks.run_ledh_pfpf_genut_initial_rqmc_all_models import (
    HORIZON,
    PARTICLE_COUNT,
    _design,
    build_campaign_models,
    make_evaluator,
    paired_inputs,
    _markdown,
)


def test_active_inventory_dimensions_and_event_orders() -> None:
    models = build_campaign_models(include_references=False)
    assert [model.row_id for model in models] == [
        "lgssm_T50",
        "ksc_sv_T10",
        "exact_sv_T10",
        "generalized_sv_T10",
        "predator_prey_T20",
        "austria_sir_T20",
    ]
    expected = {
        "lgssm_T50": (3, 5, 3, False, 0),
        "ksc_sv_T10": (1, 2, 1, False, 0),
        "exact_sv_T10": (1, 2, 1, False, 0),
        "generalized_sv_T10": (1, 3, 1, True, 0),
        "predator_prey_T20": (2, 6, 2, True, 1),
        "austria_sir_T20": (18, 3, 9, True, 1),
    }
    for model in models:
        callbacks = model.callbacks
        assert (
            callbacks.state_dimension,
            callbacks.parameter_count,
            callbacks.observation_dimension,
            callbacks.transition_before_first_observation,
            callbacks.target_time_offset,
        ) == expected[model.row_id]
        assert model.observations.shape == (
            HORIZON,
            callbacks.observation_dimension,
        )


def test_paired_inputs_change_only_initial_cloud() -> None:
    for model in build_campaign_models(include_references=False):
        initial, process, hashes = paired_inputs(model, 95123)
        assert initial["iid_initial"].shape == initial["rqmc_initial"].shape
        assert not bool(
            tf.reduce_all(
                tf.equal(initial["iid_initial"], initial["rqmc_initial"])
            ).numpy()
        )
        expected_steps = (
            HORIZON
            if model.callbacks.transition_before_first_observation
            else HORIZON - 1
        )
        assert process.shape == (
            expected_steps,
            PARTICLE_COUNT,
            model.callbacks.state_dimension,
        )
        assert hashes["process_noise_sha256"]


def test_d18_uses_exact_covariance_cubature_design() -> None:
    design = _design(18)
    tf.debugging.assert_near(tf.reduce_mean(design, axis=0), tf.zeros([18]))
    centered = design - tf.reduce_mean(design, axis=0, keepdims=True)
    covariance = tf.einsum("ni,nj->ij", centered, centered) / PARTICLE_COUNT
    tf.debugging.assert_near(covariance, tf.eye(18), atol=1.0e-6, rtol=1.0e-6)


def test_standard_backward_marks_match_direct_lgssm_formula() -> None:
    callbacks = diagonal_lgssm_callbacks()
    theta = tf.constant([0.72, 0.55, 0.35, 0.35, 0.45], tf.float32)
    parents = tf.constant(
        [[-0.4, 0.2, 0.1], [0.5, -0.3, 0.7], [0.1, 0.4, -0.2]],
        tf.float32,
    )
    children = tf.constant(
        [[-0.1, 0.3, 0.2], [0.3, -0.1, 0.5], [0.2, 0.1, -0.4]],
        tf.float32,
    )
    parent_marks = tf.constant(
        [[0.2, 0.1, -0.3, 0.4, 0.5], [-0.1, 0.3, 0.2, -0.2, 0.1], [0.4, -0.2, 0.1, 0.3, -0.1]],
        tf.float32,
    )
    parent_weights = tf.constant([0.2, 0.5, 0.3], tf.float32)
    observation = tf.constant([0.1, -0.2, 0.3], tf.float32)
    actual = standard_pairwise_backward_marks(
        callbacks,
        theta,
        parents,
        tf.math.log(parent_weights),
        parent_marks,
        children,
        observation,
        observation_index=1,
    )
    expected = lgssm_score._target_model_progressive_score_marks(  # noqa: SLF001
        theta,
        parents,
        tf.math.log(parent_weights),
        parent_marks,
        children,
        observation,
    )
    tf.debugging.assert_near(actual, expected, atol=1.0e-6, rtol=1.0e-6)


def test_lgssm_cpu_xla_initial_rqmc_smoke() -> None:
    if tf.config.list_physical_devices("GPU"):
        raise RuntimeError("test requires CUDA_VISIBLE_DEVICES=-1 before TensorFlow import")
    model = build_campaign_models(include_references=False)[0]
    evaluator = make_evaluator(model)
    initial, process, _ = paired_inputs(model, 95100)
    value, score, diagnostics = evaluator(
        model.theta,
        model.observations,
        initial["rqmc_initial"],
        process,
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


def test_markdown_accepts_one_seed_undefined_variance_ratios() -> None:
    payload = {
        "status": "smoke_pass",
        "model_order": ["fixture"],
        "cell_summaries": [
            {"model_id": "fixture", "arm": arm}
            for arm in ("iid_initial", "rqmc_initial")
        ],
        "paired_comparisons": [
            {
                "model_id": "fixture",
                "value_variance_ratio_rqmc_over_iid": None,
                "score_l2_variance_ratio_rqmc_over_iid": None,
            }
        ],
    }
    assert "| fixture | N/A | N/A |" in _markdown(payload)
