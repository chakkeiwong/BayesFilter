from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import pytest
import tensorflow as tf
from numpy.polynomial.legendre import leggauss

from bayesfilter.highdim import ledh_contract_e_tp_scalar_sv_tf as model
from bayesfilter.highdim.ledh_forward_contract import ACTUAL_SV_ROW_ID, KSC_SV_ROW_ID
from scripts.filtering_value_gradient_benchmark_generate_p8_datasets import _sv_dataset


ROOT = Path(__file__).resolve().parents[2]
DTYPE = tf.float64
THETA = tf.constant([0.2533471031357997, -0.916290731874155], DTYPE)
PREPARATIONS = {
    (ACTUAL_SV_ROW_ID, 1): ROOT / "docs/benchmarks/artifacts/contract_e_tp_all_models_2026_07_15/phase5_actual_sv_t1_order25_lookahead1_preparation_20260715.json",
    (ACTUAL_SV_ROW_ID, 2): ROOT / "docs/benchmarks/artifacts/contract_e_tp_all_models_2026_07_15/phase7_actual_sv_t2_order41_lookahead1_preparation_20260715.json",
    (ACTUAL_SV_ROW_ID, 10): ROOT / "docs/benchmarks/artifacts/contract_e_tp_all_models_2026_07_15/phase5_actual_sv_t10_order41_lookahead8_preparation_20260715.json",
    (KSC_SV_ROW_ID, 1): ROOT / "docs/benchmarks/artifacts/contract_e_tp_all_models_2026_07_15/phase5_ksc_sv_t1_order25_lookahead1_preparation_20260715.json",
    (KSC_SV_ROW_ID, 2): ROOT / "docs/benchmarks/artifacts/contract_e_tp_all_models_2026_07_15/phase7_ksc_sv_t2_order41_lookahead1_preparation_20260715.json",
    (KSC_SV_ROW_ID, 10): ROOT / "docs/benchmarks/artifacts/contract_e_tp_all_models_2026_07_15/phase5_ksc_sv_t10_order41_lookahead8_preparation_20260715.json",
}


def _prepared(row_id: str, time_steps: int):
    payload = json.loads(PREPARATIONS[(row_id, time_steps)].read_text(encoding="utf-8"))
    target = payload["target"]
    spec = model.make_scalar_sv_spec(row_id)
    raw = tf.convert_to_tensor(_sv_dataset(81101)["observations"][:time_steps], DTYPE)
    target_observations, flow_observations = model.target_and_flow_observations(
        spec, raw
    )
    active = tf.reshape(
        tf.constant(payload["active_indices"], tf.int32),
        [time_steps - 1, model.FEATURE_COUNT],
    )
    scales = tf.reshape(
        tf.constant(payload["row_scales"], DTYPE),
        [time_steps - 1, model.FEATURE_COUNT],
    )
    return (
        spec,
        tf.constant(target["theta"], DTYPE),
        raw,
        target_observations,
        flow_observations,
        tf.constant(payload["teacher_quadrature"]["nodes"], DTYPE),
        tf.constant(payload["teacher_quadrature"]["weights"], DTYPE),
        active,
        scales,
        tf.constant(payload["continuation_quadrature"]["points"], DTYPE),
        tf.constant(payload["continuation_quadrature"]["weights"], DTYPE),
        int(payload["feature_contract"]["lookahead_steps"]),
    )


@pytest.mark.parametrize("row_id", (ACTUAL_SV_ROW_ID, KSC_SV_ROW_ID))
def test_scalar_sv_model_wrapper_traces_without_graph_time_construction(
    row_id: str,
) -> None:
    spec = model.make_scalar_sv_spec(row_id)

    @tf.function(input_signature=[tf.TensorSpec([2], DTYPE)])
    def evaluate(theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        initial = spec.model.initial_log_density(
            theta, tf.constant([[-0.25], [0.5]], DTYPE)
        )
        transition = spec.model.transition_log_density(
            theta,
            tf.constant([[-0.25], [0.5]], DTYPE),
            tf.constant([[0.1], [-0.2]], DTYPE),
            t=1,
        )
        return initial, transition

    initial, transition = evaluate(THETA)
    assert bool(tf.reduce_all(tf.math.is_finite(initial)).numpy())
    assert bool(tf.reduce_all(tf.math.is_finite(transition)).numpy())


@pytest.mark.parametrize("row_id", (ACTUAL_SV_ROW_ID, KSC_SV_ROW_ID))
def test_scalar_sv_model_wrapper_preserves_eager_nonfinite_rejection(
    row_id: str,
) -> None:
    spec = model.make_scalar_sv_spec(row_id)
    with pytest.raises(ValueError, match="must be finite"):
        spec.model.observation_log_density(
            THETA,
            tf.constant([[float("nan")]], DTYPE),
            tf.constant([0.25], DTYPE),
            t=0,
        )


@pytest.mark.parametrize("row_id", (ACTUAL_SV_ROW_ID, KSC_SV_ROW_ID))
@pytest.mark.parametrize("future_count", (1, 2, 8))
def test_loop_continuation_matches_unrolled_value_and_jacobian(
    row_id: str, future_count: int
) -> None:
    spec = model.make_scalar_sv_spec(row_id)
    nodes, weights = leggauss(33)
    grid = tf.constant(10.0 * nodes, DTYPE)
    grid_weights = tf.constant(10.0 * weights, DTYPE)
    raw = tf.constant(np.linspace(0.35, 0.85, future_count)[:, None], DTYPE)
    target, _flow = model.target_and_flow_observations(spec, raw)
    points = tf.constant([[-0.8], [0.1], [0.9]], DTYPE)
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(THETA)
        old = model.target_continuation_log_likelihood(
            spec,
            THETA,
            points,
            target,
            grid,
            grid_weights,
            first_future_time_index=1,
        )
        new = model.target_continuation_log_likelihood_loop(
            spec,
            THETA,
            points,
            target,
            tf.constant(future_count, tf.int32),
            grid,
            grid_weights,
            first_future_time_index=tf.constant(1, tf.int32),
        )
    np.testing.assert_allclose(new, old, rtol=3e-14, atol=3e-14)
    np.testing.assert_allclose(
        tape.jacobian(new, THETA), tape.jacobian(old, THETA), rtol=2e-13, atol=3e-13
    )


@pytest.mark.parametrize("row_id", (ACTUAL_SV_ROW_ID, KSC_SV_ROW_ID))
@pytest.mark.parametrize("time_steps", (1, 2, 10))
def test_scalar_sv_loop_matches_frozen_unrolled_program(
    row_id: str, time_steps: int
) -> None:
    (
        spec,
        theta,
        _raw,
        target,
        flow,
        nodes,
        weights,
        active,
        scales,
        grid,
        grid_weights,
        lookahead,
    ) = _prepared(row_id, time_steps)
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(theta)
        old = model.contract_e_tp_scalar_sv_recursive_core(
            spec,
            theta,
            target,
            flow,
            nodes,
            weights,
            active,
            scales,
            grid,
            grid_weights,
            lookahead_steps=lookahead,
        )
        new = model.contract_e_tp_scalar_sv_loop_core(
            spec,
            theta,
            target,
            flow,
            nodes,
            weights,
            active,
            scales,
            grid,
            grid_weights,
            lookahead_steps=lookahead,
        )
    for name in (
        "objective",
        "increment_history",
        "minimum_weight_history",
        "condition_number_history",
        "feature_residual_history",
        "valid_history",
        "final_particles",
        "final_log_unnormalized_weights",
    ):
        np.testing.assert_allclose(new[name], old[name], rtol=3e-13, atol=5e-13)
    np.testing.assert_allclose(
        tape.gradient(new["objective"], theta),
        tape.gradient(old["objective"], theta),
        rtol=3e-12,
        atol=8e-12,
    )


@pytest.mark.parametrize("row_id", (ACTUAL_SV_ROW_ID, KSC_SV_ROW_ID))
def test_scalar_sv_compiled_invalid_theta_poisons_every_claim_output(
    row_id: str,
) -> None:
    (
        spec,
        theta,
        _raw,
        target,
        flow,
        nodes,
        weights,
        active,
        scales,
        grid,
        grid_weights,
        lookahead,
    ) = _prepared(row_id, 2)
    factory = model.make_contract_e_tp_scalar_sv_loop_tf(
        spec,
        target,
        flow,
        nodes,
        weights,
        active,
        scales,
        grid,
        grid_weights,
        lookahead_steps=lookahead,
        jit_compile=True,
    )
    invalid_theta = tf.stack([tf.constant(4.0, DTYPE), theta[1]])
    result = factory(invalid_theta)

    assert bool(result["valid"].numpy()) is False
    for name in (
        "objective",
        "score",
        "increment_history",
        "final_particles",
        "final_log_unnormalized_weights",
    ):
        assert not bool(tf.reduce_any(tf.math.is_finite(result[name])).numpy()), name
