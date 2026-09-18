"""Pinned and independent checks for native centered-TT contractions."""

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.highdim import centered_tt_native_tf as native
from bayesfilter.highdim.bases import BoundedInterval, LegendreBasis1D, ProductBasis
from bayesfilter.highdim.zhao_cui_austria_sir_centered_density_tf import (
    centered_lane_b_product_basis,
)
from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_tf import lane_b_measure_convention
from tests.test_filter_repair_remaining_routes import _graph, _original

D = tf.float64


def _inputs(dimension, family):
    basis = ProductBasis(tuple(LegendreBasis1D(BoundedInterval(-1., 1.), 1 + axis % 2)
                               for axis in range(dimension)), lane_b_measure_convention())
    if family == "centered_lagrange":
        full = centered_lane_b_product_basis(order=2, num_elems=1)
        basis = ProductBasis(full.bases[:dimension], full.convention)
    components = []
    for index, rank in enumerate((1, 2, 3)):
        ranks = (1, *([rank] * (dimension - 1)), 1)
        cores = []
        for axis, local in enumerate(basis.bases):
            shape = (ranks[axis], local.basis_dim, ranks[axis + 1])
            values = tf.reshape(tf.cast(tf.range(np.prod(shape)), D), shape)
            cores.append(.25 + .1 * tf.sin(values + axis + index))
        components.append(tuple(cores))
    points = tf.reshape(tf.linspace(tf.constant(-.6, D), .7, 4 * dimension), [4, dimension])
    return basis, tuple(components), points


@pytest.mark.parametrize("dimension", [2, 5])
@pytest.mark.parametrize("jit", [False, True])
@pytest.mark.parametrize("family", ["legendre", "centered_lagrange"])
def test_complete_component_pair_contractions_and_derivatives(dimension, jit, family):
    basis, components, points = _inputs(dimension, family)
    before = _original("zhao_cui_austria_sir_centered_density_tf")
    prefix = min(2, dimension - 1)
    core_specs = tf.nest.map_structure(lambda value: tf.TensorSpec(value.shape, D), components)

    def old(cores, query):
        values = tf.stack([before._evaluate_component(core, basis, query) for core in cores], 1)
        mass = tf.stack([tf.stack([before._cross_mass(left, right, basis) for right in cores])
                         for left in cores])
        marginal = tf.stack([tf.stack([before._cross_prefix_values(left, right, basis, query[:, :prefix])
                                      for right in cores], 1) for left in cores], 1)
        return values, mass, marginal

    def current(cores, query):
        return (native.evaluate_components(cores, basis, query),
                native.cross_components(cores, cores, basis),
                native.cross_components(cores, cores, basis, query[:, :prefix]))

    def differentiable(function):
        def evaluate(cores, query):
            with tf.GradientTape() as tape:
                tape.watch((cores, query))
                result = function(cores, query)
                objective = sum(tf.reduce_sum(value) for value in result)
            return result, tape.gradient(objective, (cores, query))
        return tf.function(evaluate, input_signature=[core_specs, tf.TensorSpec(points.shape, D)],
                           jit_compile=jit, autograph=False)

    call, reference = differentiable(current), differentiable(old)
    actual, expected = call(components, points), reference(components, points)
    for result, authority in zip(tf.nest.flatten(actual), tf.nest.flatten(expected), strict=True):
        np.testing.assert_allclose(result, authority, atol=1e-10, rtol=1e-10)
    # Exercise the public compiled primitive under an external eager tape too.
    with tf.GradientTape() as tape:
        tape.watch((components, points))
        result = current(components, points)
        objective = sum(tf.reduce_sum(value) for value in result)
    for result, authority in zip(tf.nest.flatten(tape.gradient(objective, (components, points))),
                                 tf.nest.flatten(expected[1]), strict=True):
        np.testing.assert_allclose(result, authority, atol=1e-10, rtol=1e-10)
    direction = .1 * tf.cos(points)
    delta = 1e-5
    plus = sum(tf.reduce_sum(x) for x in call(components, points + delta * direction)[0])
    minus = sum(tf.reduce_sum(x) for x in call(components, points - delta * direction)[0])
    np.testing.assert_allclose(tf.reduce_sum(actual[1][1] * direction),
                               (plus - minus) / (2 * delta), atol=1e-8, rtol=1e-7)
    _graph(call)
    if jit:
        assert "HloModule" in call.experimental_get_compiler_ir(components, points)(stage="hlo")


@pytest.mark.parametrize("degree", [0, 1, 3])
def test_legendre_endpoint_gradients_compile_at_zero_substeps(degree):
    basis = LegendreBasis1D(BoundedInterval(-1., 1.), degree)

    @tf.function(input_signature=[tf.TensorSpec([4], D)], jit_compile=True, autograph=False)
    def gradient(points):
        with tf.GradientTape() as tape:
            tape.watch(points)
            values = tf.reduce_sum(basis.evaluate(points))
        return tape.gradient(values, points, unconnected_gradients=tf.UnconnectedGradients.ZERO)

    points = tf.constant([-.7, -.2, .1, .6], D)
    np.testing.assert_allclose(gradient(points), tf.reduce_sum(basis.derivative(points), axis=-1),
                               atol=1e-12, rtol=1e-12)


def test_batched_polynomial_feature_values_and_jacobian():
    before = _original("zhao_cui_austria_sir_centered_density_tf")
    identifiers = ("linear_0", "linear_2", "quadratic_1", "interaction_0_2")
    theta = tf.constant([[.02, -.04, .06], [.07, -.08, .09]], D)
    families = tf.constant([0, 0, 1, 2])
    left, right = tf.constant([0, 2, 1, 0]), tf.constant([0, 2, 1, 2])
    expected = before.CenteredThetaFeatures(identifiers).batch_values_and_jacobian(theta)
    actual = native.feature_values_and_jacobian(theta, families, left, right)
    for result, authority in zip(actual, expected, strict=True):
        np.testing.assert_allclose(result, authority, atol=1e-16, rtol=1e-15)
    with tf.GradientTape() as tape:
        tape.watch(theta)
        values = native.feature_values_and_jacobian(theta, families, left, right)[0]
        loss = tf.reduce_sum(values)
    np.testing.assert_allclose(tape.gradient(loss, theta), tf.reduce_sum(actual[1], axis=1),
                               atol=1e-16, rtol=1e-15)


def _fresh_parent():
    """Fresh exact constant-density fixture; no historical trained result."""
    from bayesfilter.highdim.source_route import SourceRouteCoordinateFrame
    from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_target_tf import tensor_sha256
    from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_tf import (
        LaneBLogNormalizerEstimate,
        LaneBT1Artifact,
        LaneBT1Settings,
        issue_lane_b_t1_identity,
        source_closure,
    )

    settings = LaneBT1Settings(arm_id="filter_repair_exact_reference", rank=1, basis_order=2,
        basis_num_elems=1, learning_rate=.001, l1_weight=1e-6, l2_weight=0., batch_size=4,
        train_steps=1, expansion_factor=1., covariance_jitter=1e-5, quantile_fraction=.1,
        use_quantile_scale=False, tau=.05, gradient_clip_norm=10., cdf_grid_size=16,
        cdf_bisection_steps=8, cdf_max_working_bytes=2**24)
    cores = tuple(tf.ones([1, 3, 1], D) for _ in range(36))
    exact = tf.math.log(tf.constant(1.05, D))

    def estimate(role):
        return LaneBLogNormalizerEstimate(role=role, seed=0, sample_count=4,
            shift_constant=tf.zeros([], D), log_evidence=exact, log_shifted_normalizer=exact,
            log_standard_error=tf.zeros([], D), log_likelihood_sha256=tensor_sha256(tf.fill([4], exact)))

    values = {"settings": settings, "frame": SourceRouteCoordinateFrame(mu=tf.zeros([36], D),
        matrix=tf.eye(36, dtype=D), expansion_factor=1.), "cores": cores, "shift_constant": tf.zeros([], D),
        "calibration_estimate": estimate("exact_test_calibration"), "validation_estimate": estimate("exact_test_validation"),
        "frozen_reference_points": tf.fill([36, 4], tf.constant(.5, D)),
        "training_cloud_manifest": {"role": "exact_constant_reference_no_training"},
        "validation_cloud_manifest": {"role": "exact_constant_reference_no_training"}, "source_hashes": source_closure()}
    return LaneBT1Artifact(**values, identity=issue_lane_b_t1_identity(**values))


def test_complete_centered_child_scores_preserve_pinned_consumer():
    from bayesfilter.highdim import (
        zhao_cui_austria_sir_centered_density_tf as candidate,
    )

    parent = _fresh_parent()
    components = tuple(tuple(core * tf.reshape(1. + .04 * tf.sin(tf.cast(tf.range(3), D) + index + axis), [1, 3, 1])
                             for axis, core in enumerate(parent.cores)) for index in range(3))
    before = _original("zhao_cui_austria_sir_centered_density_tf")
    actual = candidate.LaneBCenteredResidualChild(parent, components)
    expected = before.LaneBCenteredResidualChild(parent, components)
    points = tf.reshape(tf.linspace(tf.constant(-.3, D), .4, 4 * 36), [4, 36])
    theta = tf.constant([.02, -.01, .03], D)

    def endpoint(child, parameters, query):
        return (child.increment_and_score(parameters), child.point_log_density_and_score(parameters, query),
                child.prefix_log_marginal_and_score(parameters, query[:, :18]))

    call = tf.function(lambda parameters, query: endpoint(actual, parameters, query),
        input_signature=[tf.TensorSpec(theta.shape, D), tf.TensorSpec(points.shape, D)],
        jit_compile=True, autograph=False)
    result = call(theta, points)
    for value, authority in zip(tf.nest.flatten(result), tf.nest.flatten(endpoint(expected, theta, points)), strict=True):
        np.testing.assert_allclose(value, authority, atol=1e-10, rtol=1e-10)
    for method, query in ((actual.point_log_density_and_score, points),
                          (actual.prefix_log_marginal_and_score, points[:, :18])):
        with tf.GradientTape() as tape:
            tape.watch(theta)
            value, manual = method(theta, query)
            objective = tf.reduce_sum(value)
        np.testing.assert_allclose(tape.gradient(objective, theta), tf.reduce_sum(manual, axis=0),
                                   atol=1e-10, rtol=1e-10)
    # Differentiate both returned values and scores w.r.t. parameters and
    # query coordinates across the standalone compiled endpoint boundary.
    derivatives = []
    prefix = points[:, :18]
    for child in (actual, expected):
        with tf.GradientTape() as tape:
            tape.watch((theta, prefix))
            value, score = child.prefix_log_marginal_and_score(theta, prefix)
            objective = tf.reduce_sum(value) + .1 * tf.reduce_sum(score)
        derivatives.append(tape.gradient(objective, (theta, prefix)))
    for result, authority in zip(derivatives[0], derivatives[1], strict=True):
        np.testing.assert_allclose(result, authority, atol=1e-10, rtol=1e-10)
    public_program = next(row[1].compiled for key, row in native._METHOD_PROGRAMS.items()
        if key[0] == id(actual) and key[1].__name__ == "prefix_log_marginal_and_score")
    assert "HloModule" in public_program.experimental_get_compiler_ir(theta, prefix)(stage="hlo")
    _graph(public_program)
    assert "HloModule" in call.experimental_get_compiler_ir(theta, points)(stage="hlo")


def test_centered_training_loss_and_all_core_gradients(monkeypatch):
    """Compare the full existing loss, including point/global/prefix scores."""
    from bayesfilter.highdim import (
        zhao_cui_austria_sir_parameter_density_training_tf as candidate,
    )

    parent = _fresh_parent()
    before = _original("zhao_cui_austria_sir_parameter_density_training_tf")
    contractions = _original("zhao_cui_austria_sir_centered_density_tf")
    for name in ("_evaluate_component", "_cross_mass", "_cross_prefix_values", "CenteredThetaFeatures"):
        monkeypatch.setattr(before, name, getattr(contractions, name))
    components = tuple(tuple(core * tf.reshape(1. + .02 * tf.sin(tf.cast(tf.range(3), D) + index + axis), [1, 3, 1])
                             for axis, core in enumerate(parent.cores)) for index in range(3))
    actual = candidate.CenteredResidualTrainer(parent, initial_residual_components=components)
    expected = before.CenteredResidualTrainer(parent, initial_residual_components=components)
    theta = tf.constant([[0., 0., 0.], [.02, -.01, .03]], D)
    points = tf.reshape(tf.linspace(tf.constant(-.3, D), .4, 2 * 4 * 36), [2, 4, 36])
    log_weights = tf.constant([[.2, -.1, .3, -.2], [.1, -.2, .2, -.1]], D)

    def endpoint(trainer, parameters, query, weights):
        target = tf.reshape(tf.linspace(tf.constant(-.2, D), .3, 12), [4, 3])
        with tf.GradientTape() as tape:
            loss = trainer.absolute_density_loss_arrays(parameters, query, weights,
                l1_weight=1e-5, l2_weight=1e-4, derivative_points=query[0],
                derivative_target_score=target, derivative_importance_log_weight=weights[0],
                derivative_weight=.02)
            global_metrics = trainer.origin_global_score_metrics_arrays(target[0], tf.ones([3], D))
            prefix_metrics = trainer.origin_prefix_score_metrics_arrays(query[0, :, :18], target, tf.ones([4, 3], D))
            objective = loss.total_loss + .03 * global_metrics["loss"] + .04 * prefix_metrics["loss"]
        gradients = tape.gradient(objective, trainer.trainable_variables)
        assert all(value is not None for value in gradients)
        return ((loss.total_loss, loss.absolute_density_loss, loss.derivative_matching_loss,
                 loss.exact_child_mass, loss.target_log_density_term, loss.target_mass_estimate,
                 loss.target_mass_standard_error, loss.minimum_rho), global_metrics, prefix_metrics, gradients)

    call = tf.function(lambda parameters, query, weights: endpoint(actual, parameters, query, weights),
        input_signature=[tf.TensorSpec(theta.shape, D), tf.TensorSpec(points.shape, D), tf.TensorSpec(log_weights.shape, D)],
        jit_compile=True, autograph=False)
    result = call(theta, points, log_weights)
    authority = endpoint(expected, theta, points, log_weights)
    for value, reference in zip(tf.nest.flatten(result), tf.nest.flatten(authority), strict=True):
        np.testing.assert_allclose(value, reference, atol=1e-10, rtol=1e-10)
    assert "HloModule" in call.experimental_get_compiler_ir(theta, points, log_weights)(stage="hlo")
    _graph(call)


def test_centered_training_callback_has_stable_signature_and_preserves_update(monkeypatch):
    from bayesfilter.highdim import (
        zhao_cui_austria_sir_parameter_density_training_tf as candidate,
    )

    parent = _fresh_parent()
    before = _original("zhao_cui_austria_sir_parameter_density_training_tf")
    reference = _original("zhao_cui_austria_sir_centered_density_tf")
    for name in ("_evaluate_component", "_cross_mass", "_cross_prefix_values", "CenteredThetaFeatures"):
        monkeypatch.setattr(before, name, getattr(reference, name))
    trainers = (candidate.CenteredResidualTrainer(parent), before.CenteredResidualTrainer(parent))
    calls = tuple(module.make_compiled_origin_score_prefit_step(trainer,
        tf.keras.optimizers.SGD(learning_rate=.001), gradient_clip_norm=10.)
        for module, trainer in zip((candidate, before), trainers, strict=True))
    points = tf.reshape(tf.linspace(tf.constant(-.3, D), .4, 4 * 36), [4, 36])
    target = tf.reshape(tf.linspace(tf.constant(-.2, D), .3, 12), [4, 3])
    weights = tf.constant([.2, -.1, .3, -.2], D)
    for _ in range(2):
        actual, expected = (call(points, target, weights) for call in calls)
        for value, authority in zip(actual, expected, strict=True):
            np.testing.assert_allclose(value, authority, atol=1e-10, rtol=1e-10)
        np.testing.assert_allclose(trainers[0].position(), trainers[1].position(), atol=1e-10, rtol=1e-10)
    assert calls[0].specialization_cache_info().currsize == 1
    assert calls[0].experimental_get_tracing_count() == 1
    assert "HloModule" in calls[0].experimental_get_compiler_ir(points, target, weights)(stage="hlo")
