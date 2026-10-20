"""Diagnostic fixed-program parity for the TT moment-teacher execution repair."""

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.highdim import zhao_cui_moment_teacher_lgssm_tf as teacher
from bayesfilter.highdim import zhao_cui_moment_teacher_nonlinear_tf as nonlinear
from bayesfilter.highdim.cubature_genut_adapters import predator_prey_candidate_adapter
from bayesfilter.highdim.moment_teacher_native_tf import (
    freeze_scale_shift_core,
    operator_power_program,
    teacher_normal,
    teacher_uniform_float32,
)
from tests.test_filter_repair_remaining_routes import _graph, _original

D = tf.float64


def _prepared(module, horizon):
    theta = tf.constant([0.55, 0.45, 0.35, 0.8, 0.6], D)
    prepared = module.prepare_lgssm_teacher_inputs(
        observations=tf.reshape(tf.linspace(tf.constant(-.1, D), .2, 3 * horizon), [horizon, 3]),
        time_steps=horizon, fit_rows=24, basis_size=2, rank=1, sweeps=1,
        chart_scale=2.5, defensive_weight=.05, root_seed=91751, dtype=D, center_theta=theta,
    )
    return theta, prepared


def _compare(actual, expected, tolerance=1e-10):
    for result, reference in zip(tf.nest.flatten(actual), tf.nest.flatten(expected), strict=True):
        if result.dtype == tf.bool or result.dtype.is_integer:
            np.testing.assert_array_equal(result, reference)
        else:
            assert bool(tf.reduce_all(tf.math.is_finite(result)))
            np.testing.assert_allclose(result, reference, rtol=tolerance, atol=tolerance)


def _controls():
    return teacher.MomentTeacherControls(
        sinkhorn_steps=2, balance_steps=100, correction_steps=0,
        correction_strength=0., correction_floor=1e-6,
        pairwise_correction_steps=0, pairwise_strength=0., pairwise_floor=1e-6,
        tt_ridge=1e-5, column_scale_floor=1e-6, condition_number_veto=1e10, fit_residual_veto=2.)


@pytest.mark.parametrize("horizon", [2, 4])
def test_lgssm_teacher_preparation_and_base_directions(horizon):
    before = _original("zhao_cui_moment_teacher_lgssm_tf")
    theta, prepared = _prepared(teacher, horizon)
    _, original = _prepared(before, horizon)
    _compare(prepared, original)
    expected = before._teacher_base_log_targets(theta, prepared)
    call = tf.function(lambda value: teacher._teacher_base_log_targets(value, prepared),
                       input_signature=[tf.TensorSpec([5], D)], jit_compile=True, autograph=False)
    actual = call(theta)
    _compare(actual, expected)
    direction = tf.constant([.1, -.05, .03, .02, -.04], D)
    delta = 1e-5
    fd = (call(theta + delta * direction)[0] - call(theta - delta * direction)[0]) / (2 * delta)
    np.testing.assert_allclose(tf.einsum("tnp,p->tn", actual[1], direction), fd, rtol=1e-7, atol=1e-8)
    assert "HloModule" in call.experimental_get_compiler_ir(theta)(stage="hlo")
    _graph(call)


@pytest.mark.parametrize("horizon", [2, 4])
def test_complete_teacher_direction_recursion(horizon):
    before = _original("zhao_cui_moment_teacher_lgssm_tf")
    theta, prepared = _prepared(teacher, horizon)
    controls = _controls()
    signature = [tf.TensorSpec([5], D)]
    reference = tf.function(lambda value: before._teacher_targets(value, prepared, controls, setup_static=True),
                            input_signature=signature, jit_compile=True, autograph=False)
    candidate = tf.function(lambda value: teacher._teacher_targets(value, prepared, controls, setup_static=True),
                            input_signature=signature, jit_compile=True, autograph=False)
    expected, actual = reference(theta), candidate(theta)
    assert bool(expected["valid"]) and bool(actual["valid"])
    _compare(actual, expected)
    assert "HloModule" in candidate.experimental_get_compiler_ir(theta)(stage="hlo")
    _graph(candidate)


@pytest.mark.parametrize("dtype", [tf.float32, tf.float64])
def test_polynomial_preparation_preserves_dtype_and_batch_axes(dtype):
    before = _original("zhao_cui_moment_teacher_lgssm_tf")
    points = tf.reshape(tf.linspace(tf.constant(-.9, dtype), .9, 30), [3, 10])
    actual = teacher._legendre_basis_values(points, 7)
    expected = before._legendre_basis_values(points, 7)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("basis_size", [1, 3, 7])
def test_batched_operator_powers_match_original_rules(basis_size):
    before = _original("zhao_cui_moment_teacher_lgssm_tf")
    basis = before.LegendreBasis1D(before.BoundedInterval(-1., 1.), basis_size - 1)
    expected = tf.stack([before.legendre_monomial_operator_matrix(
        basis, power, before.MassMeasure.REFERENCE_MEASURE) for power in range(5)])
    call = operator_power_program(basis_size)
    _compare(call(), expected)
    assert "HloModule" in call.experimental_get_compiler_ir()(stage="hlo")


@pytest.mark.parametrize("dtype", [tf.float32, tf.float64])
@pytest.mark.parametrize("count", [7, 32])
def test_teacher_random_stream_preserves_legacy_draws(dtype, count):
    seed = tf.constant([91751, 3119])
    call = tf.function(lambda seed: teacher_normal([count, 3], seed, dtype),
        input_signature=[tf.TensorSpec([2], tf.int32)], jit_compile=True, autograph=False)
    expected = tf.random.stateless_normal([count, 3], seed, dtype=dtype)
    _compare(call(seed), expected, 1e-6 if dtype == tf.float32 else 1e-10)
    uniform = tf.function(lambda seed: teacher_uniform_float32([count, 3], seed),
        input_signature=[tf.TensorSpec([2], tf.int32)], jit_compile=True, autograph=False)
    np.testing.assert_array_equal(uniform(seed), tf.random.stateless_uniform([count, 3], seed))


@pytest.mark.parametrize("horizon", [2, 4])
def test_scale_shift_freeze_matches_baseline_and_compiles(horizon):
    before = _original("zhao_cui_moment_teacher_lgssm_tf")
    _, prepared = _prepared(teacher, horizon)
    controls = _controls()
    expected = before.freeze_teacher_scale_shift_indices(prepared, controls)
    actual = teacher.freeze_teacher_scale_shift_indices(prepared, controls)
    _compare(actual, expected)
    specification = tuple((name, tf.TensorSpec(value.shape, value.dtype)) for name, value in prepared.items())
    program = teacher._lgssm_freeze_program(specification, controls, 8, True)
    assert "HloModule" in program.experimental_get_compiler_ir(prepared)(stage="hlo")
    _graph(program)
    invalid = dict(prepared, center_theta=tf.constant([.55, .45, .35, -.8, .6], D))
    with pytest.raises(ValueError, match="invalid TT fit"):
        teacher.freeze_teacher_scale_shift_indices(invalid, controls)


def test_scale_shift_exhaustion_and_invalid_fit_stop_without_extra_iterations():
    def evaluate(valid):
        def targets(indices):
            # Toggle the second date's maximizing row on every iteration.
            first = tf.where(indices[1] == 0, tf.constant([1., 2.]), tf.constant([2., 1.]))
            return {"marginal_values": tf.stack([first, first]), "valid": valid}
        return freeze_scale_shift_core(tf.zeros([2, 2]), tf.zeros([2], tf.int32), targets, 4)
    call = tf.function(evaluate, input_signature=[tf.TensorSpec([], tf.bool)],
                       jit_compile=True, autograph=False)
    indices, valid, stable = call(True)
    np.testing.assert_array_equal(indices, [0, 0])
    assert bool(valid) and not bool(stable)
    indices, valid, stable = call(False)
    np.testing.assert_array_equal(indices, [0, 1])
    assert not bool(valid) and not bool(stable)


def test_preparation_and_freeze_graphs_do_not_unroll_horizon():
    sizes = []
    for horizon in (2, 4):
        theta, prepared = _prepared(teacher, horizon)
        prepare = teacher._lgssm_preparation_program(horizon, 24, 2, 1, 1, D, True)
        inputs = (prepared["observations"], theta, prepared["chart_scale"],
                  tf.constant(.05, D), tf.constant(91751))
        assert "HloModule" in prepare.experimental_get_compiler_ir(*inputs)(stage="hlo")
        specification = tuple((name, tf.TensorSpec(value.shape, value.dtype))
                              for name, value in prepared.items())
        freeze = teacher._lgssm_freeze_program(specification, _controls(), 8, True)
        sizes.append((len(_graph(prepare)), len(_graph(freeze))))
    assert sizes[0] == sizes[1]


def _nonlinear_prepared(module, horizon):
    adapter = predator_prey_candidate_adapter()
    theta = tf.constant([.6, 114., 25., .3, .5, .5])
    prepared = module.prepare_nonlinear_teacher_inputs(
        adapter=adapter, observations=tf.tile(tf.constant([[52., 4.5], [55., 5.]]), [horizon // 2, 1]),
        state_offset=tf.constant([60., 5.]), state_scale=tf.constant([30., 12.]), center_theta=theta,
        initial_standard_deviation=1., process_standard_deviation=2., fit_rows=48,
        basis_size=2, rank=1, sweeps=1, defensive_weight=0.,
        pair_indices=tf.constant([[0, 1], [1, 0]]), root_seed=9121,
    )
    return adapter, theta, prepared


@pytest.mark.parametrize("horizon", [2, 4])
def test_nonlinear_preparation_directions_and_freezing(horizon):
    before = _original("zhao_cui_moment_teacher_nonlinear_tf")
    adapter, theta, prepared = _nonlinear_prepared(nonlinear, horizon)
    _, _, expected = _nonlinear_prepared(before, horizon)
    _compare(prepared, expected, 1e-6)
    controls = _controls()
    frozen = nonlinear.freeze_nonlinear_teacher_scale_shift_indices(
        prepared, controls, adapter, initial_variance=1., process_variance=4.)
    expected_frozen = before.freeze_nonlinear_teacher_scale_shift_indices(
        prepared, controls, adapter, initial_variance=1., process_variance=4.)
    np.testing.assert_array_equal(frozen["scale_shift_indices"], expected_frozen["scale_shift_indices"])
    signature = [tf.TensorSpec([6], tf.float32)]

    def program(module):
        return tf.function(lambda value: module._teacher_targets(value, frozen, controls, adapter,
            initial_variance=1., process_variance=4., setup_static=True),
            input_signature=signature, jit_compile=True, autograph=False)

    candidate, reference = program(nonlinear), program(before)
    actual, expected = candidate(theta), reference(theta)
    assert bool(actual["valid"]) and bool(expected["valid"])
    _compare(actual, expected, 1e-5)
    assert "HloModule" in candidate.experimental_get_compiler_ir(theta)(stage="hlo")
    _graph(candidate)
