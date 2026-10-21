"""CPU reference checks for optional frozen backward-guided LEDH proposals."""

from dataclasses import replace
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import pytest
import tensorflow as tf
import tensorflow_probability as tfp

from bayesfilter.highdim import ledh_canonical_score_tf as shared_score
from bayesfilter.highdim import ledh_canonical_reset_score_tf as shared_reset
from bayesfilter.highdim import ledh_unified_correction_tf as shared_correction
from bayesfilter.highdim.ledh_backward_guide_tf import (
    FrozenMixtureBackwardGuide,
    FrozenQuadraticBackwardGuide,
    build_backward_guided_ledh_kernel,
    guided_gaussian_draw,
)
from tests.highdim.test_ledh_canonical_total_gaussian_tf import (
    DTYPE,
    _fixture,
    _parameterized_model,
)


def _mixture():
    return FrozenMixtureBackwardGuide(
        tf.constant([[0.2, -0.4], [0.2, -0.4]], DTYPE),
        tf.constant([[[-0.7, 0.4], [0.8, -0.5]]] * 2, DTYPE),
        tf.constant([[[[1.1, 0.2], [0.2, 0.7]], [[0.6, -0.1], [-0.1, 1.3]]]] * 2, DTYPE),
        tf.constant([True, False]),
    )


def _primitive_inputs():
    return (
        tf.constant([[-0.3, 0.6], [0.7, -0.2]], DTYPE),
        tf.constant([[0.2, -0.1], [-0.3, 0.4]], DTYPE),
        tf.constant([[0.8, 0.2], [0.2, 1.2]], DTYPE),
        tf.constant([[0.1, -0.07], [-0.07, -0.2]], DTYPE),
        tf.constant([[0.3, -0.8], [-0.2, 0.9]], DTYPE),
    )


def test_quadratic_square_completion_matches_normalized_density():
    anchors, d_anchors, covariance, d_covariance, noise = _primitive_inputs()
    guide = FrozenQuadraticBackwardGuide.from_controlled_smc(
        tf.constant([[[0.3, 0.05], [0.05, 0.2]]], DTYPE),
        tf.constant([[-0.4, 0.2]], DTYPE),
    )
    points, _, log_density, _, valid = guided_gaussian_draw(
        guide, 0, anchors, d_anchors, covariance, d_covariance, noise
    )
    precision = tf.linalg.inv(covariance)
    guided_precision = precision + guide.precision[0]
    proposal_covariance = tf.linalg.inv(guided_precision)
    proposal_mean = tf.einsum(
        "ij,nj->ni", proposal_covariance,
        tf.einsum("ij,nj->ni", precision, anchors) + guide.linear[0],
    )
    log_normalizer = 0.5 * (
        tf.linalg.logdet(proposal_covariance) - tf.linalg.logdet(covariance)
        - tf.einsum("ni,ij,nj->n", anchors, precision, anchors)
        + tf.einsum("ni,ij,nj->n", proposal_mean, guided_precision, proposal_mean)
    )
    physical = tfp.distributions.MultivariateNormalTriL(
        anchors, tf.linalg.cholesky(covariance)
    ).log_prob(points)
    log_guide = -0.5 * tf.einsum(
        "ni,ij,nj->n", points, guide.precision[0], points
    ) + tf.einsum("ni,i->n", points, guide.linear[0])
    assert bool(valid)
    np.testing.assert_allclose(log_density, physical + log_guide - log_normalizer, atol=2e-14)


def test_mixture_coefficient_tangents_and_spatial_gradient():
    anchors, d_anchors, _, _, _ = _primitive_inputs()
    guide = _mixture()
    matrices, vectors, d_matrices, d_vectors, valid = guide.evaluate(0, anchors, d_anchors)
    delta = tf.constant(2e-6, DTYPE)
    plus = guide.evaluate(0, anchors + delta * d_anchors, d_anchors)
    minus = guide.evaluate(0, anchors - delta * d_anchors, d_anchors)
    for analytical, upper, lower in zip((d_matrices, d_vectors), plus[:2], minus[:2]):
        np.testing.assert_allclose(analytical, (upper - lower) / (2 * delta), rtol=2e-8, atol=1e-10)
    with tf.GradientTape() as tape:
        tape.watch(anchors)
        components = tfp.distributions.MultivariateNormalTriL(
            guide.centers[0], tf.linalg.cholesky(guide.covariances[0])
        ).log_prob(anchors[:, None])
        total = tf.reduce_sum(tf.reduce_logsumexp(components + guide.logits[0], axis=-1))
    np.testing.assert_allclose(
        vectors - tf.einsum("nij,nj->ni", matrices, anchors), tape.gradient(total, anchors), atol=2e-14
    )
    assert bool(valid)
    for inactive in guide.evaluate(1, anchors, d_anchors)[:4]:
        np.testing.assert_array_equal(inactive, tf.zeros_like(inactive))


@pytest.mark.parametrize("conditioning", ["positive_information", "mixture_product_moments"])
def test_draw_and_moving_density_tangents_with_full_covariance(conditioning):
    anchors, d_anchors, covariance, d_covariance, noise = _primitive_inputs()
    guide = replace(_mixture(), conditioning=conditioning)
    result = guided_gaussian_draw(guide, 0, anchors, d_anchors, covariance, d_covariance, noise)
    delta = tf.constant(2e-6, DTYPE)
    plus = guided_gaussian_draw(
        guide, 0, anchors + delta * d_anchors, d_anchors, covariance + delta * d_covariance, d_covariance, noise
    )
    minus = guided_gaussian_draw(
        guide, 0, anchors - delta * d_anchors, d_anchors, covariance - delta * d_covariance, d_covariance, noise
    )
    assert bool(result[4])
    for analytical, upper, lower in ((result[1], plus[0], minus[0]), (result[3], plus[2], minus[2])):
        np.testing.assert_allclose(analytical, (upper - lower) / (2 * delta), rtol=2e-8, atol=2e-10)


@pytest.fixture(scope="module")
def integrated():
    fixture = _fixture()
    fixture["options"].update(
        reset_epsilon=1.0, reset_sinkhorn_steps=5, reset_balance_steps=32,
    )
    kernel = build_backward_guided_ledh_kernel(
        _parameterized_model, _mixture(), horizon=2, particle_count=8,
        state_dimension=2, observation_dimension=2, parameter_dimension=4,
        dtype=DTYPE, jit_compile=False, **fixture["options"],
    )
    return fixture, kernel


def test_dynamic_product_mixture_calls_full_shared_executor():
    fixture = _fixture()
    fixture["options"].update(reset_epsilon=1., reset_sinkhorn_steps=5, reset_balance_steps=32)
    guide = replace(_mixture(), conditioning="mixture_product_moments")
    options = dict(horizon=2, particle_count=8, state_dimension=2,
                   observation_dimension=2, parameter_dimension=4, dtype=DTYPE,
                   jit_compile=False, **fixture["options"])
    kernel = build_backward_guided_ledh_kernel(_parameterized_model, None,
        mixture_components=2, mixture_conditioning="mixture_product_moments", **options)
    theta = tf.constant([.82, 1.1, -.35, -.2], DTYPE)
    direction = tf.constant([.4, -.3, .2, .5], DTYPE)
    arguments = _arguments(theta, direction, fixture)
    value, score = kernel(*arguments, guide.logits, guide.centers, guide.covariances, guide.active)
    expected_value, expected_score = shared_score._value_and_analytical_score_impl(
        _parameterized_model(theta, direction), theta, *arguments[2:6],
        with_score=True, initial_state_tangent=arguments[6],
        initial_covariance_tangent=arguments[7], backward_guide=guide, **fixture["options"])
    np.testing.assert_allclose(value, expected_value, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(score, expected_score, rtol=1e-11, atol=1e-11)


def _arguments(theta, direction, fixture, d_states=None, d_covariances=None):
    states = fixture["initial_states"]
    covariances = fixture["initial_covariances"]
    return (
        theta, direction, states, covariances, fixture["noises"], fixture["observations"],
        tf.zeros_like(states) if d_states is None else d_states,
        tf.zeros_like(covariances) if d_covariances is None else d_covariances,
    )


def _curved_observation_model(theta, direction):
    model = _parameterized_model(theta, direction)
    curvature = tf.constant(0.35, DTYPE)
    return replace(
        model,
        observation_fn=lambda points: theta[1] * points + curvature * tf.sin(points),
        observation_jacobian_fn=lambda points: tf.linalg.diag(
            theta[1] + curvature * tf.cos(points)
        ),
        observation_tangent_fn=lambda points, d_points: (
            direction[1] * points
            + (theta[1] + curvature * tf.cos(points)) * d_points
        ),
        observation_jacobian_tangent_fn=lambda points, d_points: tf.linalg.diag(
            direction[1] - curvature * tf.sin(points) * d_points
        ),
    )


def _curved_guided_program(theta, direction, fixture):
    return shared_score._value_and_analytical_score_impl(
        _curved_observation_model(theta, direction), theta,
        fixture["initial_states"], fixture["initial_covariances"],
        fixture["noises"], fixture["observations"], with_score=True,
        return_trace=True, backward_guide=_mixture(), **fixture["options"],
    )


def test_guided_consumer_actual_nonlinear_flow_jacobian(integrated, monkeypatch):
    fixture, _ = integrated
    original = shared_score._flow_substeps_with_tangent
    captured = []

    def recording(*args, **kwargs):
        result = original(*args, **kwargs)
        argument_count = len(args) - 1

        def record_executed(*values):
            executed_args = (args[0],) + values[:argument_count]
            executed_kwargs = dict(kwargs, eye=values[argument_count])
            captured.append((
                executed_args, executed_kwargs, values[argument_count + 1:],
            ))
            return tf.constant(0, tf.int32)

        token = tf.py_function(
            record_executed, list(args[1:]) + [kwargs["eye"]] + list(result),
            Tout=tf.int32,
        )
        with tf.control_dependencies([token]):
            return tuple(tf.identity(value) for value in result)

    monkeypatch.setattr(shared_score, "_flow_substeps_with_tangent", recording)
    theta = tf.constant([0.82, 1.10, -0.35, -0.20], DTYPE)
    direction = tf.constant([0.4, -0.3, 0.2, 0.5], DTYPE)
    _, _, trace = _curved_guided_program(theta, direction, fixture)
    assert len(captured) == len(trace) == 2
    delta = tf.constant(2e-6, DTYPE)
    for args, kwargs, result in captured:
        assert kwargs["return_affine"]
        model, actual, _, anchors = args[:4]
        assert float(tf.reduce_max(tf.abs(result[6] - anchors))) > 1e-3
        assert float(tf.reduce_max(tf.abs(
            model.observation_jacobian_fn(anchors + 0.17)
            - model.observation_jacobian_fn(anchors)
        ))) > 1e-3
        coordinate_count = int(tf.size(actual))
        columns = []
        for coordinate in range(coordinate_count):
            perturbation = delta * tf.reshape(
                tf.one_hot(coordinate, coordinate_count, dtype=DTYPE), actual.shape
            )
            upper = original(*(args[:1] + (actual + perturbation,) + args[2:]), **kwargs)[0]
            lower = original(*(args[:1] + (actual - perturbation,) + args[2:]), **kwargs)[0]
            columns.append(tf.reshape((upper - lower) / (2 * delta), [-1]))
        numerical = tf.stack(columns, axis=1)
        affine = result[8]
        expected = tf.reshape(
            tf.einsum("nm,nij->nimj", tf.eye(actual.shape[0], dtype=DTYPE), affine),
            [coordinate_count, coordinate_count],
        )
        assert bool(tf.reduce_all(tf.math.is_finite(numerical)))
        np.testing.assert_allclose(numerical, expected, rtol=2e-6, atol=2e-8)
        for particle in range(actual.shape[0]):
            start = particle * actual.shape[1]
            stop = start + actual.shape[1]
            sign, log_determinant = tf.linalg.slogdet(numerical[start:stop, start:stop])
            assert float(sign) == 1.0
            np.testing.assert_allclose(
                result[2][particle], log_determinant, rtol=2e-6, atol=2e-8,
            )


def test_quadratic_guide_is_applied_in_preflow_coordinates(integrated):
    fixture, _ = integrated
    theta = tf.constant([0.82, 1.10, -0.35, -0.20], DTYPE)
    direction = tf.zeros_like(theta)
    model = _curved_observation_model(theta, direction)
    guide = FrozenQuadraticBackwardGuide(
        tf.constant([[[0.8, 0.1], [0.1, 0.4]]] * 2, DTYPE),
        tf.constant([[0.5, -0.3]] * 2, DTYPE),
    )
    _, _, trace = shared_score._value_and_analytical_score_impl(
        model, theta, fixture["initial_states"], fixture["initial_covariances"],
        fixture["noises"], fixture["observations"], with_score=True,
        return_trace=True, backward_guide=guide, **fixture["options"],
    )
    record = trace[0]
    anchors = model.transition_mean_fn(theta, fixture["initial_states"])
    points, tangent, _, _, valid = guided_gaussian_draw(
        guide, 0, anchors, tf.zeros_like(anchors), model.process_covariance,
        tf.zeros_like(model.process_covariance), fixture["noises"][0],
    )
    assert bool(valid)
    np.testing.assert_allclose(record["pre_flow"], points, atol=1e-13)
    flow = shared_score._flow_substeps_with_tangent(
        model, points, tangent, anchors, tf.zeros_like(anchors),
        record["predicted_covariances"], record["d_predicted_covariances"],
        fixture["observations"][0], model.observation_covariance,
        tf.zeros_like(model.observation_covariance),
        tf.linalg.inv(model.observation_covariance),
        tf.zeros_like(model.observation_covariance),
        substeps=fixture["options"]["flow_substeps"], eye=tf.eye(2, dtype=DTYPE),
        return_affine=True,
    )
    affine = flow[8]
    shift = flow[6] - tf.einsum("nij,nj->ni", affine, anchors)
    children = tf.einsum("nij,nj->ni", affine, points) + shift
    np.testing.assert_allclose(record["children"], children, atol=1e-13)
    recovered = tf.linalg.solve(affine, (children - shift)[..., None])[..., 0]
    np.testing.assert_allclose(recovered, points, atol=1e-13)

    def log_guide(values):
        return (-0.5 * tf.einsum("ni,ij,nj->n", values, guide.precision[0], values)
                + tf.einsum("ni,i->n", values, guide.linear[0]))

    assert float(tf.reduce_max(tf.abs(log_guide(recovered) - log_guide(children)))) > 1e-2
    pulled_precision = tf.einsum("nji,jk,nkl->nil", affine, guide.precision[0], affine)
    assert float(tf.reduce_max(tf.abs(pulled_precision - guide.precision[0]))) > 1e-2
    pulled_linear = tf.einsum(
        "nji,nj->ni", affine,
        guide.linear[0] - tf.einsum("ij,nj->ni", guide.precision[0], shift),
    )
    pulled_value = (-0.5 * tf.einsum("ni,nij,nj->n", points, pulled_precision, points)
                    + tf.einsum("ni,ni->n", points, pulled_linear) + log_guide(shift))
    np.testing.assert_allclose(pulled_value, log_guide(children), atol=1e-13)


def test_curved_guided_flow_logdet_and_score_total_derivative(integrated):
    fixture, _ = integrated
    theta = tf.constant([0.82, 1.10, -0.35, -0.20], DTYPE)
    direction = tf.constant([0.4, -0.3, 0.2, 0.5], DTYPE)
    _, derivative, trace = _curved_guided_program(theta, direction, fixture)
    assert float(tf.reduce_max(tf.abs(trace[0]["d_states_after_reset"]))) > 1e-3
    for delta_value in (2e-6, 1e-6):
        delta = tf.constant(delta_value, DTYPE)
        upper = _curved_guided_program(theta + delta * direction, tf.zeros_like(theta), fixture)
        lower = _curved_guided_program(theta - delta * direction, tf.zeros_like(theta), fixture)
        np.testing.assert_allclose(
            derivative[0], (upper[0] - lower[0]) / (2 * delta), rtol=2e-5, atol=2e-7,
        )
        for record, upper_record, lower_record in zip(trace, upper[2], lower[2]):
            numerical = (
                upper_record["flow_log_determinant"] - lower_record["flow_log_determinant"]
            ) / (2 * delta)
            assert bool(tf.reduce_all(tf.math.is_finite(numerical)))
            np.testing.assert_allclose(
                record["d_flow_log_determinant"], numerical, rtol=2e-5, atol=2e-7,
            )


def test_shared_full_program_total_derivative(integrated):
    fixture, kernel = integrated
    theta = tf.constant([0.82, 1.10, -0.35, -0.20], DTYPE)
    delta = tf.constant(2e-6, DTYPE)
    for direction in tf.eye(4, dtype=DTYPE):
        value, derivative = kernel(*_arguments(theta, direction, fixture))
        upper = kernel(*_arguments(theta + delta * direction, tf.zeros_like(theta), fixture))[0]
        lower = kernel(*_arguments(theta - delta * direction, tf.zeros_like(theta), fixture))[0]
        assert bool(tf.math.is_finite(value))
        np.testing.assert_allclose(derivative[0], (upper - lower) / (2 * delta), rtol=3e-4, atol=3e-5)
    assert kernel.experimental_get_tracing_count() == 1


def test_initial_state_and_covariance_total_derivative(integrated):
    fixture, kernel = integrated
    theta = tf.constant([0.82, 1.10, -0.35, -0.20], DTYPE)
    d_states = 0.13 * tf.ones_like(fixture["initial_states"])
    d_covariances = 0.09 * fixture["initial_covariances"]
    _, derivative = kernel(*_arguments(theta, tf.zeros_like(theta), fixture, d_states, d_covariances))
    delta = tf.constant(2e-6, DTYPE)
    shifted = []
    for sign in (-1, 1):
        candidate = dict(fixture)
        candidate["initial_states"] = fixture["initial_states"] + sign * delta * d_states
        candidate["initial_covariances"] = fixture["initial_covariances"] + sign * delta * d_covariances
        shifted.append(kernel(*_arguments(theta, tf.zeros_like(theta), candidate))[0])
    np.testing.assert_allclose(derivative[0], (shifted[1] - shifted[0]) / (2 * delta), rtol=3e-4, atol=3e-5)


def test_zero_guide_reproduces_existing_complete_recursion(integrated):
    fixture, _ = integrated
    theta = tf.constant([0.82, 1.10, -0.35, -0.20], DTYPE)
    direction = tf.constant([0.4, -0.3, 0.2, 0.5], DTYPE)
    model = _parameterized_model(theta, direction)
    guide = FrozenQuadraticBackwardGuide(tf.zeros([2, 2, 2], DTYPE), tf.zeros([2, 2], DTYPE))
    arguments = (model, theta, fixture["initial_states"], fixture["initial_covariances"], fixture["noises"], fixture["observations"])
    baseline = shared_score._value_and_analytical_score_impl(*arguments, with_score=True, **fixture["options"])
    guided = shared_score._value_and_analytical_score_impl(*arguments, with_score=True, backward_guide=guide, **fixture["options"])
    for actual, expected in zip(guided, baseline):
        assert bool(tf.reduce_all(tf.math.is_finite(actual)))
        assert bool(tf.reduce_all(tf.math.is_finite(expected)))
        np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=2e-11)


def test_shared_call_chain_and_physical_importance_denominator(integrated, monkeypatch):
    fixture, _ = integrated
    called = set()
    for module, name in (
        (shared_score, "ukf_predict_with_parameter_tangent"),
        (shared_score, "ukf_update_with_parameter_tangent"),
        (shared_score, "_flow_substeps_with_tangent"),
        (shared_reset, "sinkhorn_contract_e_reset_triple_with_tangent"),
        (shared_correction, "batched_higher_moment_shape_jvp"),
    ):
        original = getattr(module, name)

        def recording(*args, _name=name, _original=original, **kwargs):
            called.add(_name)
            return _original(*args, **kwargs)

        monkeypatch.setattr(module, name, recording)
    theta = tf.constant([0.82, 1.10, -0.35, -0.20], DTYPE)
    direction = tf.constant([0.4, -0.3, 0.2, 0.5], DTYPE)
    model = _parameterized_model(theta, direction)
    _, _, trace = shared_score._value_and_analytical_score_impl(
        model, theta, fixture["initial_states"], fixture["initial_covariances"],
        fixture["noises"], fixture["observations"], with_score=True,
        return_trace=True, backward_guide=_mixture(), **fixture["options"],
    )
    assert len(called) == 5
    states = fixture["initial_states"]
    for record in trace:
        anchors = model.transition_mean_fn(theta, states)
        physical = tfp.distributions.MultivariateNormalTriL(
            anchors, tf.linalg.cholesky(model.process_covariance)
        ).log_prob(record["children"])
        np.testing.assert_allclose(record["transition_log_density"], physical, atol=2e-12)
        np.testing.assert_allclose(
            record["prior_observation_logits"], record["incoming_log_weights"] + physical
            + record["flow_log_determinant"] - record["proposal_log_density"], atol=2e-12,
        )
        assert bool(record["backward_guide_valid"])
        assert record["reset_transport"].shape == (8, 8)
        states = record["states_after_reset"]


@pytest.mark.parametrize("precision", [[[-0.2, 0.0], [0.0, 0.2]], [[0.2, 0.1], [0.0, 0.2]]])
def test_invalid_quadratic_is_flagged(precision):
    anchors, d_anchors, covariance, d_covariance, noise = _primitive_inputs()
    guide = FrozenQuadraticBackwardGuide(tf.constant([precision], DTYPE), tf.zeros([1, 2], DTYPE))
    result = guided_gaussian_draw(guide, 0, anchors, d_anchors, covariance, d_covariance, noise)
    assert not bool(result[-1])
    assert bool(tf.reduce_all(tf.math.is_nan(result[0])))


def test_indefinite_information_requires_positive_total_precision():
    anchors, d_anchors, covariance, d_covariance, noise = _primitive_inputs()
    for strength, expected in ((-.2, True), (-2., False)):
        guide = FrozenQuadraticBackwardGuide(tf.constant([[[strength, 0.], [0., .2]]], DTYPE),
            tf.zeros([1, 2], DTYPE), allow_indefinite_information=True)
        result = guided_gaussian_draw(guide, 0, anchors, d_anchors, covariance, d_covariance, noise)
        assert bool(result[-1]) == expected
        if expected:
            delta = tf.constant(2e-6, DTYPE)
            plus = guided_gaussian_draw(guide, 0, anchors + delta*d_anchors, d_anchors,
                covariance + delta*d_covariance, d_covariance, noise)
            minus = guided_gaussian_draw(guide, 0, anchors - delta*d_anchors, d_anchors,
                covariance - delta*d_covariance, d_covariance, noise)
            np.testing.assert_allclose(result[1], (plus[0]-minus[0])/(2*delta), rtol=2e-8, atol=2e-10)
            np.testing.assert_allclose(result[3], (plus[2]-minus[2])/(2*delta), rtol=2e-8, atol=2e-10)


def test_non_gaussian_transition_and_reduced_reset_are_rejected(integrated):
    fixture, _ = integrated
    theta = tf.constant([0.82, 1.10, -0.35, -0.20], DTYPE)
    model = replace(_parameterized_model(theta, tf.zeros_like(theta)), transition_log_density_fn=lambda *args: None)
    with pytest.raises(ValueError, match="additive Gaussian"):
        shared_score._value_and_analytical_score_impl(
            model, theta, fixture["initial_states"], fixture["initial_covariances"], fixture["noises"],
            fixture["observations"], with_score=True, backward_guide=_mixture(), **fixture["options"],
        )
    options = dict(fixture["options"], pairwise_steps=0)
    with pytest.raises(ValueError, match="full higher-moment"):
        build_backward_guided_ledh_kernel(
            _parameterized_model, _mixture(), horizon=2, particle_count=8, state_dimension=2,
            observation_dimension=2, parameter_dimension=4, dtype=DTYPE, **options,
        )


@pytest.mark.parametrize("extra, message", [
    ({"importance_weight_policy": "marginal_mixture"}, "ancestor importance weights"),
    ({"annealed_stages": 2}, "one annealing stage"),
    ({"ancestry_policy": "hilbert_inverse_cdf"}, "existing_one_to_one ancestry"),
])
def test_unsupported_guide_combinations_fail_at_both_endpoints(integrated, extra, message):
    fixture, _ = integrated
    theta = tf.constant([0.82, 1.10, -0.35, -0.20], DTYPE)
    options = dict(fixture["options"], **extra)
    with pytest.raises(ValueError, match=message):
        build_backward_guided_ledh_kernel(
            _parameterized_model, _mixture(), horizon=2, particle_count=8, state_dimension=2,
            observation_dimension=2, parameter_dimension=4, dtype=DTYPE, **options,
        )
    with pytest.raises(ValueError, match=message):
        shared_score._value_and_analytical_score_impl(
            _parameterized_model(theta, tf.zeros_like(theta)), theta,
            fixture["initial_states"], fixture["initial_covariances"], fixture["noises"],
            fixture["observations"], with_score=True, backward_guide=_mixture(), **options,
        )


@pytest.mark.parametrize("failure", ["guide_validity", "guide_tangent"])
def test_guided_proposal_preserves_split_failure_status(integrated, monkeypatch, failure):
    """A merged guided lane must retain value/force failure separation."""
    from bayesfilter.highdim import ledh_backward_guide_tf as guide_module

    fixture, _ = integrated
    original = guide_module.guided_gaussian_draw

    def failing_draw(*args, **kwargs):
        values = list(original(*args, **kwargs))
        if failure == "guide_validity":
            # Finite arithmetic must not conceal the guide's invalidity flag.
            values[-1] = tf.constant(False)
        else:
            # A failed derivative does not invalidate the finite scalar value.
            values[3] = tf.fill(tf.shape(values[3]), tf.constant(float("nan"), DTYPE))
        return tuple(values)

    monkeypatch.setattr(guide_module, "guided_gaussian_draw", failing_draw)
    theta = tf.constant([0.82, 1.10, -0.35, -0.20], DTYPE)
    direction = tf.constant([0.4, -0.3, 0.2, 0.5], DTYPE)
    value, score, status = shared_score._value_and_analytical_score_impl(
        _parameterized_model(theta, direction), theta,
        fixture["initial_states"], fixture["initial_covariances"], fixture["noises"],
        fixture["observations"], with_score=True, return_status=True,
        backward_guide=_mixture(), **fixture["options"],
    )
    assert not bool(status["force_valid"])
    assert bool(tf.reduce_all(tf.math.is_finite(score)))
    if failure == "guide_validity":
        assert not bool(status["value_valid"])
        assert float(value) == float("-inf")
    else:
        assert bool(status["value_valid"])
        assert bool(tf.math.is_finite(value))
