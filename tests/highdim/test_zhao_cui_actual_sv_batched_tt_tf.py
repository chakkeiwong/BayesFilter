from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import tensorflow as tf
import tensorflow_probability as tfp

import docs.benchmarks.run_contract_e_tp_phase6_zhao_cui_comparator as comparator
from bayesfilter.highdim.zhao_cui_actual_sv_batched_tt_tf import (
    batched_fixed_tt_likelihood_value_trace,
)
from bayesfilter.highdim.zhao_cui_fixed_adjacent_tt_tf import (
    scalar_adjacent_state_fixed_tt_score,
    scalar_adjacent_state_fixed_tt_value,
)
from bayesfilter.inference.neutra_batching import bind_batch_native_neutra_target
from bayesfilter.ssm import stable_ssm_target_signature
from bayesfilter.testing.zhao_cui_actual_sv_neutra_target_tf import (
    ActualSVZCLikelihoodRecomposer,
    make_actual_sv_zc_neutra_adapter,
    posterior_value_score_status,
)


def _truth_source() -> tf.Tensor:
    normal = tfp.distributions.Normal(
        tf.constant(0.0, tf.float64), tf.constant(1.0, tf.float64)
    )
    return tf.stack(
        (
            normal.quantile(tf.constant(0.625, tf.float64)),
            normal.quantile(tf.constant(0.375, tf.float64)),
        )
    )


def _scalar_center_result():
    model, center, observations = comparator._row_inputs("actual_sv", 10)
    raw = tf.convert_to_tensor(
        comparator._sv_dataset(81101)["observations"], tf.float64
    )[:10]
    initial, adjacent, initializer = comparator._ukf_initial_cores(
        model=model,
        theta=center,
        raw_observations=raw,
        degree=10,
        order=25,
        rank=2,
        coordinate_half_width=8.0,
    )
    config = comparator._comparator_config(
        degree=10,
        order=25,
        rank=2,
        seed="svx-zc-neutra-target-center-frozen-v1",
        transition_before_first_observation=False,
        coordinate_half_width=8.0,
        density_tau=0.0,
        initial_cores=initial,
        adjacent_initial_cores=adjacent,
        initialization_rule=str(initializer["initializer_rule"]),
    )
    value = scalar_adjacent_state_fixed_tt_value(
        model,
        center,
        observations,
        config,
        branch_seed_prefix="svx-zc-neutra-target-center-frozen-v1",
    )
    score = scalar_adjacent_state_fixed_tt_score(
        model,
        center,
        observations,
        config,
        branch_seed_prefix="svx-zc-neutra-target-center-frozen-v1",
    )
    return value, score


def test_batch_native_likelihood_matches_admitted_scalar_center() -> None:
    adapter = make_actual_sv_zc_neutra_adapter()
    source = tf.stack((_truth_source(), _truth_source()), axis=0)
    likelihood = ActualSVZCLikelihoodRecomposer(adapter)
    value, score = likelihood(source)
    scalar_value, _scalar_score = _scalar_center_result()
    tf.debugging.assert_near(
        value,
        tf.fill([2], scalar_value.log_likelihood),
        atol=2.0e-10,
        rtol=2.0e-10,
    )
    tf.debugging.assert_all_finite(score, "batch likelihood score must be finite")


def test_batch_native_source_score_matches_centered_finite_difference() -> None:
    adapter = make_actual_sv_zc_neutra_adapter()
    theta = tf.stack(
        (
            _truth_source(),
            _truth_source() + tf.constant([0.03, -0.02], tf.float64),
            _truth_source() + tf.constant([-0.04, 0.05], tf.float64),
        )
    )
    likelihood = ActualSVZCLikelihoodRecomposer(adapter)
    value, score = likelihood(theta)
    step = tf.constant(1.0e-5, tf.float64)
    columns = []
    for index in range(2):
        direction = tf.one_hot(index, 2, dtype=tf.float64)[None, :]
        plus, _ = likelihood(theta + step * direction)
        minus, _ = likelihood(theta - step * direction)
        columns.append((plus - minus) / (2.0 * step))
    finite_difference = tf.stack(columns, axis=1)
    tf.debugging.assert_all_finite(value, "batch likelihood value must be finite")
    tf.debugging.assert_near(score, finite_difference, atol=2.0e-6, rtol=2.0e-6)


def test_batch_permutation_and_status_are_equivariant() -> None:
    adapter = make_actual_sv_zc_neutra_adapter()
    truth = _truth_source()
    theta = tf.stack(
        (
            truth,
            truth + tf.constant([0.05, 0.0], tf.float64),
            truth + tf.constant([0.0, -0.05], tf.float64),
            truth + tf.constant([-0.03, 0.04], tf.float64),
        )
    )
    value, score, status = adapter.neutra_batch_log_prob_and_grad_status(theta)
    permutation = tf.constant([3, 1, 0, 2], tf.int32)
    permuted = adapter.neutra_batch_log_prob_and_grad_status(
        tf.gather(theta, permutation)
    )
    tf.debugging.assert_near(permuted[0], tf.gather(value, permutation))
    tf.debugging.assert_near(permuted[1], tf.gather(score, permutation))
    for name in status:
        tf.debugging.assert_near(
            tf.cast(permuted[2][name], tf.float64),
            tf.cast(tf.gather(status[name], permutation), tf.float64),
        )
    assert bool(tf.reduce_all(status["valid_pre_regularized_score"]).numpy())
    assert bool(tf.reduce_all(tf.equal(status["status_code"], 0)).numpy())


def test_posterior_recomposition_is_exact() -> None:
    adapter = make_actual_sv_zc_neutra_adapter()
    theta = tf.stack(
        (_truth_source(), _truth_source() + tf.constant([0.02, 0.03], tf.float64))
    )
    direct = adapter.neutra_batch_log_prob_and_grad_status(theta)
    recomposed = posterior_value_score_status(theta, **adapter.program_tensors)
    tf.debugging.assert_near(direct[0], recomposed[0])
    tf.debugging.assert_near(direct[1], recomposed[1])


def test_binding_is_rejected_while_xla_ready_is_false() -> None:
    adapter = make_actual_sv_zc_neutra_adapter()
    target_signature = stable_ssm_target_signature(adapter.contract)
    try:
        bind_batch_native_neutra_target(adapter, target_signature=target_signature)
    except Exception as error:
        assert "XLA ready" in str(error)
    else:
        raise AssertionError("binding should fail while xla_hmc_ready is false")




def test_batched_trace_exposes_time_local_targets_and_sweeps() -> None:
    adapter = make_actual_sv_zc_neutra_adapter()
    theta = tf.stack((_truth_source(), _truth_source()), axis=0)
    trace = batched_fixed_tt_likelihood_value_trace(theta, **adapter.program_tensors)

    assert len(trace.steps) == 10
    first_step = trace.steps[0]
    second_step = trace.steps[1]

    assert first_step.target_kind == "initial_state_observation"
    assert first_step.one_axis_fit is not None
    assert first_step.two_axis_fit is None
    assert first_step.previous_density is None
    assert tuple(first_step.log_target.shape) == (2, 25)
    assert tuple(first_step.sqrt_target.shape) == (2, 25)
    assert tuple(first_step.one_axis_fit.condition_by_sweep.shape) == (2, 2)

    assert second_step.target_kind == "adjacent_state_update"
    assert second_step.one_axis_fit is None
    assert second_step.two_axis_fit is not None
    assert second_step.previous_density is not None
    assert tuple(second_step.log_target.shape) == (2, 625)
    assert tuple(second_step.sqrt_target.shape) == (2, 625)
    assert tuple(second_step.previous_density.shape) == (2, 625)
    assert tuple(second_step.two_axis_fit.condition_by_update.shape) == (2, 8)
    assert len(second_step.two_axis_fit.sweeps) == 8
    assert tuple(sweep.axis for sweep in second_step.two_axis_fit.sweeps[:4]) == (0, 1, 1, 0)


def test_cpu_xla_batch_target_compiles() -> None:
    adapter = make_actual_sv_zc_neutra_adapter()

    @tf.function(
        input_signature=[tf.TensorSpec([None, 2], tf.float64)],
        jit_compile=True,
    )
    def compiled(theta):
        return adapter.neutra_batch_log_prob_and_grad_status(theta)

    theta = tf.stack((_truth_source(), _truth_source()), axis=0)
    value, score, status = compiled(theta)
    tf.debugging.assert_all_finite(value, "XLA value must be finite")
    tf.debugging.assert_all_finite(score, "XLA score must be finite")
    assert bool(tf.reduce_all(status["valid_pre_regularized_score"]).numpy())
