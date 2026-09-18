"""Pinned parity and enclosing XLA checks for source-route preparation.

The independent baseline is an execution-refactor oracle, not a historical
LEDH result or a source-faithfulness claim for repository initializers.
"""

from dataclasses import replace

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.highdim import source_route as candidate
from bayesfilter.highdim import (
    source_route_preparation_runtime_tf as preparation_runtime,
)
from bayesfilter.highdim import source_route_preparation_tf as native
from bayesfilter.highdim.models import zhao_cui_sir_austria_model
from bayesfilter.highdim.tt import TTCore
from tests.test_filter_repair_remaining_routes import _graph, _original

D = tf.float64


@pytest.mark.parametrize("count,dimension", [(1, 2), (5, 3), (9, 6)])
@pytest.mark.parametrize("unit", [False, True])
def test_reference_design_preserves_baseline_and_has_hlo(count, dimension, unit):
    name = "_p59_author_sir_unit_reference_points" if unit else "_p59_author_sir_reference_points"
    expected = getattr(_original("source_route"), name)(count, dimension)
    actual = getattr(candidate, name)(count, dimension)
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)
    call = native.reference_points_program(count, dimension, unit)
    assert "HloModule" in call.experimental_get_compiler_ir()(stage="hlo")
    _graph(call)


@pytest.mark.parametrize("ranks,width", [((1, 1), 1), ((1, 3, 1), 1),
                                       ((1, 3, 2, 4, 1), 4)])
@pytest.mark.parametrize("seeded", [False, True])
def test_initial_cores_preserve_all_channels_and_compilation(ranks, width, seeded):
    name = "_source_route_seeded_channel_initial_cores" if seeded else "_source_route_constant_path_initial_cores"
    arguments = {"ranks": ranks, "basis_dim": width, "constant_value": tf.constant(1.73, D)}
    expected = getattr(_original("source_route"), name)(**arguments)
    actual = getattr(candidate, name)(**arguments)
    for left, right in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(left.values, right.values)
    call = native.initial_cores_program(ranks, width, seeded)
    assert "HloModule" in call.experimental_get_compiler_ir(tf.constant(1.73, D),
                       tf.constant(1e-3, D))(stage="hlo")
    _graph(call)


@pytest.mark.parametrize("ranks,widths,fit_rank", [((1, 1), (2,), 1),
    ((1, 3, 1), (2, 4), 4), ((1, 3, 2, 4, 1), (2, 3, 1, 4), 4)])
@pytest.mark.parametrize("zero", [False, True])
def test_channel_activity_preserves_decisions_and_heterogeneous_shapes(ranks, widths, fit_rank, zero):
    cores = tuple(TTCore(tf.reshape(tf.sin(tf.cast(tf.range(left * width * right), D)) *
        (0. if zero else .03), [left, width, right]))
        for left, width, right in zip(ranks[:-1], widths, ranks[1:], strict=True))
    arguments = {"cores": cores, "target_dim": len(cores), "fit_rank": fit_rank}
    expected = _original("source_route")._p70_channel_activity_diagnostics(**arguments)
    actual = candidate._p70_channel_activity_diagnostics(**arguments)
    for name in ("status", "minimum_active_bonds", "extra_channel_active_bond_counts", "inactive_extra_channels"):
        assert actual[name] == expected[name]
    for actual_row, expected_row in zip(actual["score_by_bond"], expected["score_by_bond"], strict=True):
        np.testing.assert_allclose(actual_row, expected_row, atol=1e-10, rtol=1e-10)
    for name in ("reference_channel_score", "activity_threshold"):
        np.testing.assert_allclose(actual[name], expected[name], atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("use_quantile,minimum_ess", [(False, 1000.), (True, 0.), (True, 1000.)])
@pytest.mark.parametrize("masked", [False, True])
def test_recenter_preserves_pinned_frame_and_jit_with_discarded_samples(use_quantile, minimum_ess, masked):
    values = tf.cast(tf.range(32), D)
    samples = tf.stack([tf.sin(.37 * values), tf.cos(.53 * values), .1 * values], axis=0)
    weights = tf.math.log(.5 + tf.square(tf.cos(.23 * values)))
    if masked:
        samples = tf.tensor_scatter_nd_update(samples, [[0, 2], [2, 7]],
                                              tf.constant([float("nan"), float("inf")], D))
        weights = tf.tensor_scatter_nd_update(weights, [[5]], tf.constant([float("nan")], D))
    arguments = {"samples": samples, "log_weights": weights, "expansion_factor": 1.3,
        "covariance_jitter": 1e-5, "use_quantile_scale": use_quantile,
        "min_ess_for_quantile_scale": minimum_ess, "quantile_fraction": .1}
    expected = _original("source_route").source_route_recenter(**arguments)
    actual = candidate.source_route_recenter(**arguments)
    np.testing.assert_allclose(actual.mu, expected.mu, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(actual.matrix, expected.matrix, atol=1e-10, rtol=1e-10)
    inputs = (samples, weights, tf.constant(1.3, D), tf.constant(1e-5, D),
              tf.constant(.1, D), tf.constant(minimum_ess, D))
    call = tf.function(lambda *args: native.recenter.python_function(*args,
        use_quantile_scale=use_quantile), input_signature=tuple(tf.TensorSpec(x.shape, x.dtype)
        for x in inputs), jit_compile=True, autograph=False)
    assert bool(call(*inputs)[2])
    assert "HloModule" in call.experimental_get_compiler_ir(*inputs)(stage="hlo")
    _graph(call)


def test_recenter_rejects_empty_finite_cloud():
    with pytest.raises(ValueError, match="NONFINITE_VALUE"):
        candidate.source_route_recenter(samples=tf.constant([[float("nan")]], D),
            log_weights=tf.zeros([1], D), expansion_factor=1.)


def test_coordinate_time_recurrence_matches_pinned_and_reuses_signature():
    model = zhao_cui_sir_austria_model()
    call = native.coordinate_frame_program(model)
    for time in (1, 4, 9):
        expected = _original("source_route")._p59_author_sir_36d_coordinate_frame_for_time(model, time_index=time)
        actual = candidate._p59_author_sir_36d_coordinate_frame_for_time(model, time_index=time)
        np.testing.assert_allclose(actual.mu, expected.mu, atol=1e-10, rtol=1e-10)
        np.testing.assert_array_equal(actual.matrix, expected.matrix)
    assert call.experimental_get_tracing_count() == 1
    assert "HloModule" in call.experimental_get_compiler_ir(tf.constant(4), model.initial_mean,
        model.process_covariance, model.initial_covariance)(stage="hlo")
    _graph(call)
    changed = replace(model, kappa=model.kappa * .9)
    alternate = candidate._p59_author_sir_36d_coordinate_frame_for_time(changed, time_index=4)
    assert np.max(np.abs(alternate.mu - actual.mu)) > 1e-10
    assert native.coordinate_frame_program(changed) is not call


def test_channel_decisions_execute_in_xla_at_the_inclusive_threshold():
    cores = tf.ones([2, 1, 1, 1], D)
    inputs = (cores, tf.constant([1]), tf.constant(1), tf.constant(1., D), tf.constant(0., D))
    call = tf.function(native.channel_activity.python_function,
        input_signature=tuple(tf.TensorSpec(x.shape, x.dtype) for x in inputs),
        jit_compile=True, autograph=False)
    scores, _, threshold, counts, _, valid = call(*inputs)
    np.testing.assert_array_equal(scores, [[1.]])
    np.testing.assert_array_equal(counts, [1])
    assert bool(valid) and float(threshold) == 1.
    assert "HloModule" in call.experimental_get_compiler_ir(*inputs)(stage="hlo")
    _graph(call)


def test_design_and_channel_graphs_do_not_grow_with_dimension():
    for create in (lambda dimension: native.reference_points_program(8, dimension, True),
                   lambda dimension: native.initial_cores_program((1, *([3] * (dimension - 1)), 1), 4, True)):
        assert len(_graph(create(3))) == len(_graph(create(8)))


def test_reachable_author_sampling_push_and_resampling_preserve_stream_and_xla():
    model = zhao_cui_sir_austria_model()
    original = _original("source_route")
    sample_count = 4
    prior = candidate._p59_author_sir_prior_sample_batch(
        model=model, sample_count=sample_count, seed=6301)
    frozen_prior = original._p59_author_sir_prior_sample_batch(
        model=model, sample_count=sample_count, seed=6301)
    tf.debugging.assert_near(prior.samples, frozen_prior.samples, atol=1e-12, rtol=1e-12)
    observation = model.simulate(final_time=2, seed=5901)[1][1]
    pushed = candidate._p59_author_sir_source_push_result(
        model=model, previous_batch=prior, observation=observation,
        time_index=1, process_noise_seed=6401)
    frozen_pushed = original._p59_author_sir_source_push_result(
        model=model, previous_batch=frozen_prior, observation=observation,
        time_index=1, process_noise_seed=6401)
    tf.debugging.assert_near(pushed.augmented_batch.samples,
                             frozen_pushed.augmented_batch.samples, atol=1e-10, rtol=1e-10)
    tf.debugging.assert_near(pushed.augmented_batch.log_weights,
                             frozen_pushed.augmented_batch.log_weights, atol=1e-12, rtol=1e-12)
    samples, indices = candidate._p59_author_sir_deterministic_weighted_resample(
        samples=pushed.augmented_batch.samples,
        log_weights=pushed.augmented_batch.log_weights)
    frozen_samples, frozen_indices = original._p59_author_sir_deterministic_weighted_resample(
        samples=frozen_pushed.augmented_batch.samples,
        log_weights=frozen_pushed.augmented_batch.log_weights)
    tf.debugging.assert_near(samples, frozen_samples, atol=1e-10, rtol=1e-10)
    tf.debugging.assert_equal(indices, frozen_indices)
    prior_program = preparation_runtime.prior_sample_program(
        model.parameter_dim(), model.state_dim(), sample_count)
    push_program = preparation_runtime.source_push_program(
        model, model.parameter_dim(), model.state_dim(), sample_count, 1)
    resample_program = preparation_runtime.deterministic_weighted_resample_program(
        int(pushed.augmented_batch.samples.shape[0]), sample_count)
    assert "HloModule" in prior_program.experimental_get_compiler_ir(
        model.initial_mean, model.initial_covariance, preparation_runtime.seed_state(6301))(stage="hlo")
    noise = preparation_runtime.normal_matrix_program(sample_count, model.state_dim())(
        preparation_runtime.seed_state(6401))
    assert "HloModule" in push_program.experimental_get_compiler_ir(
        prior.samples, prior.log_weights, noise, observation)(stage="hlo")
    assert "HloModule" in resample_program.experimental_get_compiler_ir(
        pushed.augmented_batch.samples, pushed.augmented_batch.log_weights)(stage="hlo")


@pytest.mark.parametrize("invalid", ["route", "dimension", "time", "type"])
def test_reachable_push_preserves_input_vetoes(invalid):
    model = zhao_cui_sir_austria_model()
    values = tf.zeros([model.parameter_dim() + model.state_dim(), 2], D)
    prior = candidate.SourceRouteSampleBatch(values, tf.zeros([2], D), 0,
        candidate.SOURCE_FAITHFUL_ROUTE_LABEL, "prior")
    if invalid == "route":
        prior = replace(prior, route_label="unsupported")
    elif invalid == "dimension":
        prior = replace(prior, samples=values[:-1])
    elif invalid == "type":
        prior = object()
    error, message = {
        "route": (ValueError, "source_faithful_filtering input"),
        "dimension": (ValueError, "dimension must equal"),
        "time": (ValueError, "time_index must advance"),
        "type": (TypeError, "previous_batch must be"),
    }[invalid]
    with pytest.raises(error, match=message):
        candidate._p59_author_sir_source_push_result(model=model, previous_batch=prior,
            observation=tf.zeros([1], D), time_index=0 if invalid == "time" else 1,
            process_noise_seed=6401)


@pytest.mark.parametrize("jit_compile", [False, True])
def test_source_coordinate_and_target_preparation_preserves_complete_pullbacks(jit_compile):
    matrix = tf.constant([[1.2, .1], [.2, .9]], D)
    center = tf.constant([.1, -.2], D)
    samples = tf.constant([[-2., -.1, .4, 2.], [-.3, .4, .1, .8]], D)
    sources = (matrix, center, samples)
    program = preparation_runtime.coordinate_transform_program(2, 4, jit_compile=jit_compile)

    def reference(a, b, x):
        local = tf.linalg.solve(a, x - b[:, None])
        mask = tf.abs(local) > tf.constant(1., D)
        clipped = tf.clip_by_value(local, -1., 1.)
        return local, mask, tf.reduce_mean(tf.cast(mask, D)), clipped, a @ clipped + b[:, None]

    outcomes = []
    for call in (reference, program):
        with tf.GradientTape() as tape:
            tape.watch(sources)
            outputs = call(*sources)
            loss = tf.reduce_sum(outputs[0]) + tf.reduce_sum(outputs[3]) + tf.reduce_sum(outputs[4])
        outcomes.append((outputs, tape.gradient(loss, sources)))
    tf.debugging.assert_equal(outcomes[0][0][1], outcomes[1][0][1])
    for actual, expected in zip(tf.nest.flatten(outcomes[1]), tf.nest.flatten(outcomes[0]), strict=True):
        if actual.dtype != tf.bool:
            tf.debugging.assert_near(actual, expected, atol=1e-10, rtol=1e-10)

    negative_log = tf.constant([2., 3., 4.], D)
    log_det = tf.constant(.3, D)
    fixed_shift = tf.constant(.8, D)
    for fixed in (False, True):
        sources = (negative_log, log_det, fixed_shift) if fixed else (negative_log, log_det)
        compiled = (preparation_runtime.target_values_with_shift_program if fixed
                    else preparation_runtime.target_values_program)(3, jit_compile=jit_compile)
        outcomes = []
        for native_call in (False, True):
            with tf.GradientTape() as tape:
                tape.watch(sources)
                if native_call:
                    result = compiled(*sources)
                    assert bool(result[-1])
                    values = result[:-1]
                else:
                    local = negative_log - log_det
                    shift = fixed_shift if fixed else tf.reduce_min(local)
                    weights = tf.exp(-.5 * (local - shift))
                    values = (local, weights) if fixed else (local, shift, weights)
                loss = tf.add_n([tf.reduce_sum(value) for value in values])
            outcomes.append((values, tape.gradient(loss, sources)))
        for actual, expected in zip(tf.nest.flatten(outcomes[1]), tf.nest.flatten(outcomes[0]), strict=True):
            assert actual is not None and expected is not None
            tf.debugging.assert_near(actual, expected, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("invalid", ["positive_infinity", "negative_infinity", "nan", "log_det", "shift"])
def test_target_preparation_does_not_hide_nonfinite_inputs(fixed, invalid):
    value = {"positive_infinity": float("inf"), "negative_infinity": -float("inf"),
             "nan": float("nan")}.get(invalid, 4.)
    negative_log = tf.constant([2., 3., value], D)
    log_det = tf.constant(-float("inf") if invalid == "log_det" else .3, D)
    shift = tf.constant(-float("inf") if invalid == "shift" else .8, D)
    if not fixed and invalid == "shift":
        # The computed shift cannot be invalid without an invalid local target.
        negative_log = tf.fill([3], tf.constant(float("inf"), D))
    program = (preparation_runtime.target_values_with_shift_program if fixed
               else preparation_runtime.target_values_program)(3)
    inputs = (negative_log, log_det, shift) if fixed else (negative_log, log_det)
    assert not bool(program(*inputs)[-1])


def test_resample_preserves_nonfinite_weight_rejection():
    with pytest.raises(ValueError, match="NONFINITE_VALUE"):
        candidate._p59_author_sir_deterministic_weighted_resample(
            samples=tf.zeros([2, 3], D), log_weights=tf.constant([0., float("inf"), 1.], D))
