"""Independent scheduling, transient-fault and tangent diagnostics for LEDH flow."""

import pytest
import tensorflow as tf

from bayesfilter.highdim import ledh_canonical_score_tf as shared


def flow_case(dtype, state_dimension=3, observation_dimension=2, offset=0., states=None):
    if states is None:
        states = tf.reshape(tf.cast(tf.range(3 * state_dimension), dtype),
                            [3, state_dimension]) * .03 + .1
    count = tf.shape(states)[0]
    observation_matrix = tf.reshape(
        tf.cast(tf.range(observation_dimension * state_dimension), dtype),
        [observation_dimension, state_dimension]) * .04 + .1
    nonlinear_matrix = tf.ones_like(observation_matrix) * .02
    observation_covariance = .3 * tf.eye(observation_dimension, dtype=dtype)
    covariance = tf.broadcast_to(.2 * tf.eye(state_dimension, dtype=dtype),
                                 [count, state_dimension, state_dimension])
    state_tangent = .01 + .02 * states
    mean_tangent = .03 + .01 * states
    covariance_tangent = .04 * covariance
    observation_covariance_tangent = .05 * observation_covariance
    offset = tf.cast(offset, dtype)
    observation_covariance += offset * observation_covariance_tangent
    covariance += offset * covariance_tangent
    inverse = tf.linalg.inv(observation_covariance)
    inverse_tangent = -inverse @ observation_covariance_tangent @ inverse
    model = shared.NonlinearScoreModel(
        transition_mean_fn=lambda values, theta: values,
        transition_mean_tangent_fn=lambda values, tangent, theta: tangent,
        observation_fn=lambda values: tf.matmul(values, observation_matrix, transpose_b=True)
        + tf.matmul(values * values, nonlinear_matrix, transpose_b=True),
        observation_jacobian_fn=lambda values: observation_matrix[None]
        + 2 * nonlinear_matrix[None] * values[:, None, :],
        observation_tangent_fn=lambda values, tangent: tf.matmul(
            tangent, observation_matrix, transpose_b=True)
        + 2 * tf.matmul(values * tangent, nonlinear_matrix, transpose_b=True),
        observation_jacobian_tangent_fn=lambda values, tangent:
        2 * nonlinear_matrix[None] * tangent[:, None, :],
        process_covariance=tf.eye(state_dimension, dtype=dtype),
        observation_covariance=observation_covariance)
    return (model, states + offset * state_tangent, state_tangent,
            .9 * states + offset * mean_tangent, mean_tangent,
            covariance, covariance_tangent,
            tf.ones([observation_dimension], dtype) * .15,
            observation_covariance, observation_covariance_tangent, inverse, inverse_tangent)


def inline_reference(arguments, substeps, return_affine):
    """Restore the original per-step guard schedule using the shared arithmetic."""
    model, actual, d_actual, auxiliary, d_auxiliary, *remaining = arguments
    dtype = actual.dtype
    count = tf.shape(actual)[0]
    eye = tf.eye(int(actual.shape[-1]), dtype=dtype)
    log_det = tf.zeros([count], dtype)
    d_log_det = tf.zeros_like(log_det)
    value_valid = tf.ones([count], tf.bool)
    force_valid = tf.ones_like(value_valid)
    first_value = tf.fill([count], -1)
    first_force = tf.fill([count], -1)
    affine = (tf.broadcast_to(eye, [count, int(actual.shape[-1]), int(actual.shape[-1])]),)
    affine += (tf.zeros_like(affine[0]),)
    epsilon = tf.constant(1. / substeps, dtype)
    for step in range(substeps):
        outputs = shared._flow_substep_body_impl(
            tf.constant(step), actual, d_actual, auxiliary, d_auxiliary,
            model, arguments[3], arguments[4], *remaining,
            tf.constant(substeps), eye, epsilon, value_valid, force_valid,
            affine_state=affine if return_affine else None)
        _, actual, d_actual, auxiliary, d_auxiliary, matrix, tangent, local_value, local_force = outputs[:9]
        step_log, step_tangent, factor_value, factor_force = shared.checked_log_abs_det_tangent(
            matrix, tangent)
        log_det += step_log
        d_log_det += epsilon * step_tangent
        value_valid = local_value & factor_value & tf.math.is_finite(log_det)
        force_valid = local_force & value_valid & factor_force & tf.math.is_finite(d_log_det)
        first_value = tf.where((first_value < 0) & ~value_valid, step, first_value)
        first_force = tf.where((first_force < 0) & ~force_valid, step, first_force)
        actual = tf.where(value_valid[:, None], actual, tf.zeros_like(actual))
        d_actual = tf.where(force_valid[:, None], d_actual, tf.zeros_like(d_actual))
        auxiliary = tf.where(value_valid[:, None], auxiliary, tf.zeros_like(auxiliary))
        d_auxiliary = tf.where(force_valid[:, None], d_auxiliary, tf.zeros_like(d_auxiliary))
        log_det = tf.where(value_valid, log_det, tf.zeros_like(log_det))
        d_log_det = tf.where(force_valid, d_log_det, tf.zeros_like(d_log_det))
        if return_affine:
            affine = outputs[9:]
    result = (actual, d_actual, log_det, d_log_det, value_valid, force_valid)
    if return_affine:
        result += (auxiliary, d_auxiliary, *affine)
    return result + ({"first_value_failure_substep": first_value,
                      "first_force_failure_substep": first_force},)


def execute(arguments, substeps=5, return_affine=False):
    return shared._flow_substeps_with_tangent(
        *arguments, substeps=substeps,
        eye=tf.eye(int(arguments[1].shape[-1]), dtype=arguments[1].dtype),
        return_affine=return_affine, return_diagnostics=True)


@pytest.mark.parametrize("dtype", [tf.float32, tf.float64])
@pytest.mark.parametrize("dimensions", [(3, 1), (2, 2), (2, 3)])
@pytest.mark.parametrize("return_affine", [False, True])
def test_xla_matches_per_step_reference(dtype, dimensions, return_affine):
    arguments = flow_case(dtype, *dimensions)
    signature = [tf.TensorSpec([], dtype)]
    program = tf.function(lambda offset: execute(flow_case(dtype, *dimensions, offset),
                                                return_affine=return_affine),
                          input_signature=signature, jit_compile=True, autograph=False)
    reference = tf.function(lambda offset: inline_reference(
        flow_case(dtype, *dimensions, offset), 5, return_affine),
        input_signature=signature, jit_compile=True, autograph=False)
    candidate, expected = program(tf.constant(0., dtype)), reference(tf.constant(0., dtype))
    tolerance = 3e-6 if dtype == tf.float32 else 2e-12
    for index, (actual, authority) in enumerate(zip(candidate[:-1], expected[:-1])):
        if index in (4, 5):
            tf.debugging.assert_equal(actual, authority)
        else:
            tf.debugging.assert_near(actual, authority, atol=tolerance, rtol=tolerance)
    tf.debugging.assert_equal(candidate[-1]["determinant_group_size"], 5)
    for key, value in expected[-1].items():
        tf.debugging.assert_equal(candidate[-1][key], value)
    program(tf.constant(.01, dtype))
    assert program.experimental_get_tracing_count() == 1
    assert arguments[1].shape[0] == 3


@pytest.mark.parametrize("return_affine", [False, True])
def test_groups_cross_boundaries_and_partial_last_group(monkeypatch, return_affine):
    monkeypatch.setattr(shared, "_FLOW_DETERMINANT_HISTORY_BYTES", 2 * 2 * 3 * 2 * 2 * 8)
    arguments = flow_case(tf.float64)
    candidate = execute(arguments, return_affine=return_affine)
    expected = inline_reference(arguments, 5, return_affine)
    tf.debugging.assert_equal(candidate[-1]["determinant_group_size"], 2)
    for index, (actual, authority) in enumerate(zip(candidate[:-1], expected[:-1])):
        if index in (4, 5):
            tf.debugging.assert_equal(actual, authority)
        else:
            tf.debugging.assert_near(actual, authority, atol=2e-12, rtol=2e-12)


@pytest.mark.parametrize("fault", ["singular", "nonfinite_matrix", "nonfinite_tangent",
                                    "local_before_factor", "factor_before_local"])
@pytest.mark.parametrize("group_size", [2, 5])
def test_recovering_fault_stays_rejected_with_first_failure(monkeypatch, fault, group_size):
    original = shared._flow_substep_body_impl
    monkeypatch.setattr(shared, "_FLOW_DETERMINANT_HISTORY_BYTES", group_size * 2 * 3 * 2 * 2 * 8)
    factor_step = 3 if fault == "local_before_factor" else 1
    local_step = 1 if fault == "local_before_factor" else 3

    def injected(step, *arguments, **keywords):
        outputs = list(original(step, *arguments, **keywords))
        matrix, tangent = outputs[5:7]
        if fault == "nonfinite_tangent":
            bad_tangent = tf.tensor_scatter_nd_update(tangent, [[0, 0, 0]],
                                                       tf.constant([float("nan")], tangent.dtype))
            outputs[6] = tf.where(step == factor_step, bad_tangent, tangent)
        else:
            bad_row = tf.zeros_like(matrix[0])
            if fault == "nonfinite_matrix":
                bad_row += tf.constant(float("nan"), matrix.dtype)
            bad_matrix = tf.concat([bad_row[None], matrix[1:]], axis=0)
            outputs[5] = tf.where(step == factor_step, bad_matrix, matrix)
        if fault in ("local_before_factor", "factor_before_local"):
            local_value = tf.concat([tf.constant([False]), outputs[7][1:]], axis=0)
            outputs[7] = tf.where(step == local_step, local_value, outputs[7])
            outputs[8] &= outputs[7]
        return tuple(outputs)

    monkeypatch.setattr(shared, "_flow_substep_body_impl", injected)
    arguments = flow_case(tf.float64)
    program = tf.function(lambda: execute(arguments), input_signature=[],
                          jit_compile=True, autograph=False)
    candidate, expected = program(), inline_reference(arguments, 5, False)
    for index in (4, 5):
        tf.debugging.assert_equal(candidate[index], expected[index])
    for key, value in expected[-1].items():
        tf.debugging.assert_equal(candidate[-1][key], value)
    tf.debugging.assert_equal(candidate[4], [fault == "nonfinite_tangent", True, True])
    tf.debugging.assert_equal(candidate[5], [False, True, True])
    for index in range(4):
        tf.debugging.assert_all_finite(candidate[index], "fault placeholders must remain finite")
        tf.debugging.assert_near(candidate[index][1:], expected[index][1:], atol=2e-12, rtol=2e-12)
    if fault == "nonfinite_tangent":
        tf.debugging.assert_near(candidate[0], expected[0], atol=2e-12, rtol=2e-12)


def test_transient_prefix_overflow_is_not_hidden_by_recovery(monkeypatch):
    def injected(matrix, tangent):
        count = tf.shape(matrix)[1]
        huge = tf.constant(2.5e38, matrix.dtype)
        logs = tf.stack([huge, huge, -huge, -huge, huge])
        return (tf.broadcast_to(logs[:, None], [5, count]),
                tf.zeros([5, count], matrix.dtype),
                tf.ones([5, count], tf.bool), tf.ones([5, count], tf.bool))

    monkeypatch.setattr(shared, "checked_log_abs_det_tangent", injected)
    candidate = execute(flow_case(tf.float32))
    tf.debugging.assert_equal(candidate[4] | candidate[5], [False] * 3)
    tf.debugging.assert_equal(candidate[-1]["first_value_failure_substep"], [1] * 3)
    tf.debugging.assert_all_finite(candidate[2], "overflow must leave finite placeholders")


def test_total_flow_tangent_matches_finite_difference():
    dtype = tf.float64
    program = tf.function(lambda offset: execute(flow_case(dtype, offset=offset), return_affine=True),
                          input_signature=[tf.TensorSpec([], dtype)],
                          jit_compile=True, autograph=False)
    step = tf.constant(1e-5, dtype)
    base, plus, minus = program(tf.constant(0., dtype)), program(step), program(-step)
    for value_index, tangent_index in ((0, 1), (2, 3), (6, 7), (8, 9)):
        difference = (plus[value_index] - minus[value_index]) / (2 * step)
        tf.debugging.assert_near(base[tangent_index], difference, atol=2e-10, rtol=2e-8)


@pytest.mark.parametrize("observation_dimension", [1, 3])
def test_reduced_coordinates_share_checked_flow_and_total_tangent(monkeypatch, observation_dimension):
    from bayesfilter.highdim.ledh_structural_innovation_tf import affine_innovation_geometry

    dtype = tf.float64
    matrix = tf.constant([[1., .1], [.2, 1.], [.3, -.1]], dtype)
    tangent = tf.constant([[.01, -.02], [.03, .01], [-.01, .02]], dtype)

    def numerical(offset):
        arguments = flow_case(dtype, 3, observation_dimension, offset)
        geometry = affine_innovation_geometry(matrix + offset * tangent, tangent,
                                              state_dim=3, innovation_dim=2)
        return shared._flow_substeps_with_tangent(
            *arguments, substeps=5, eye=tf.eye(3, dtype=dtype),
            innovation_geometry=geometry, return_diagnostics=True)

    signature = [tf.TensorSpec([], dtype)]
    program = tf.function(numerical, input_signature=signature, jit_compile=True,
                          autograph=False)
    base = program(tf.constant(0., dtype))
    monkeypatch.setattr(shared, "_FLOW_DETERMINANT_HISTORY_BYTES", 1)
    reference = tf.function(numerical, input_signature=signature, jit_compile=True,
                            autograph=False)
    expected = reference(tf.constant(0., dtype))
    tf.debugging.assert_equal(base[-1]["determinant_group_size"], 5)
    tf.debugging.assert_equal(expected[-1]["determinant_group_size"], 1)
    for index in range(4):
        tf.debugging.assert_near(base[index], expected[index], atol=2e-12, rtol=2e-12)
    tf.debugging.assert_equal(base[4] & base[5], [True] * 3)
    step = tf.constant(1e-5, dtype)
    plus, minus = program(step), program(-step)
    for value_index, tangent_index in ((0, 1), (2, 3)):
        difference = (plus[value_index] - minus[value_index]) / (2 * step)
        tf.debugging.assert_near(base[tangent_index], difference, atol=2e-10, rtol=2e-8)


def test_history_cap_and_unknown_cloud_fallback():
    selector = shared._flow_determinant_group_size
    assert selector(24, 64, 9, tf.float64) == 24
    assert selector(24, None, 9, tf.float64) == 1
    assert selector(24, 10000, 18, tf.float64) == 1
    for count in (1, 64, 1024, 10000):
        group = selector(24, count, 9, tf.float64)
        pair_bytes = 2 * count * 9 * 9 * 8
        assert group * pair_bytes <= max(pair_bytes, shared._FLOW_DETERMINANT_HISTORY_BYTES)
    with pytest.raises(ValueError):
        selector(0, 3, 2, tf.float64)


def test_unknown_particle_count_has_one_stable_graph():
    dtype = tf.float64
    program = tf.function(lambda states: execute(flow_case(dtype, 2, 2, states=states)),
                          input_signature=[tf.TensorSpec([None, 2], dtype)],
                          jit_compile=True, autograph=False)
    for count in (2, 4):
        states = tf.ones([count, 2], dtype) * .1
        candidate = program(states)
        expected = inline_reference(flow_case(dtype, 2, 2, states=states), 5, False)
        tf.debugging.assert_equal(candidate[-1]["determinant_group_size"], 1)
        tf.debugging.assert_near(candidate[0], expected[0], atol=2e-12, rtol=2e-12)
        tf.debugging.assert_equal(candidate[4] & candidate[5], tf.ones([count], tf.bool))
    assert program.experimental_get_tracing_count() == 1
