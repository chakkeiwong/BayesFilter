"""Complete update parity and XLA rejection checks for centered TT training."""

from dataclasses import fields

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.highdim import (
    zhao_cui_austria_sir_parameter_density_training_tf as candidate,
)
from tests.test_filter_repair_centered_tt import _fresh_parent
from tests.test_filter_repair_remaining_routes import _graph, _original

D = tf.float64


def test_metric_sqrt_preserves_active_derivatives_and_zero_unused_pullbacks():
    from bayesfilter.highdim.centered_training_native_tf import metric_sqrt

    @tf.function(input_signature=[tf.TensorSpec([3], D), tf.TensorSpec([3], D)],
                 jit_compile=True, autograph=False)
    def gradient(values, cotangents):
        with tf.GradientTape() as tape:
            tape.watch(values)
            result = metric_sqrt(values)
        return result, tape.gradient(result, values, output_gradients=cotangents)

    values = tf.constant([0., 4., 9.], D)
    result, derivative = gradient(values, tf.constant([0., 2., -3.], D))
    np.testing.assert_array_equal(result, tf.sqrt(values))
    np.testing.assert_array_equal(derivative, [0., .5, -.5])
    assert np.isinf(gradient(values, tf.ones([3], D))[1][0])


def _inputs():
    points = tf.reshape(tf.linspace(tf.constant(-.3, D), .4, 4 * 36), [4, 36])
    target = tf.reshape(tf.linspace(tf.constant(-.2, D), .3, 12), [4, 3])
    weights = tf.constant([.2, -.1, .3, -.2], D)
    return points, target, weights


def _callback(module, trainer, optimizer, kind):
    points, target, weights = _inputs()
    if kind == "absolute":
        call = module.make_compiled_absolute_train_step(trainer, optimizer,
            l1_weight=1e-5, l2_weight=1e-4, derivative_weight=.02, gradient_clip_norm=10.)
        inputs = (tf.constant([[0., 0., 0.], [.02, -.01, .03]], D),
            tf.stack([points, points + .03]), tf.stack([weights, weights - .01]),
            points, target, weights)
        invalid_index = 4
    elif kind == "prefit":
        call = module.make_compiled_origin_score_prefit_step(trainer, optimizer, gradient_clip_norm=10.)
        inputs, invalid_index = (points, target, weights), 1
    else:
        call = module.make_compiled_origin_total_score_train_step(trainer, optimizer,
            point_weight=.02, global_weight=.03, prefix_weight=.04, l2_weight=1e-4,
            gradient_clip_norm=10.)
        inputs = (points, target, weights, target[0], tf.ones([3], D),
            points[:, :18], target, tf.ones([4, 3], D))
        invalid_index = 6
    return call, inputs, invalid_index


@pytest.mark.parametrize("trainer_name", ["CenteredResidualTrainer", "CoreAffineTangentTrainer"])
def test_direct_objectives_compile_and_preserve_all_variable_and_input_gradients(trainer_name, monkeypatch):
    parent = _fresh_parent()
    before = _original("zhao_cui_austria_sir_parameter_density_training_tf")
    contractions = _original("zhao_cui_austria_sir_centered_density_tf")
    for name in ("_evaluate_component", "_cross_mass", "_cross_prefix_values", "CenteredThetaFeatures"):
        monkeypatch.setattr(before, name, getattr(contractions, name))
    trainers = tuple(getattr(module, trainer_name)(parent) for module in (candidate, before))
    points, targets, weights = _inputs()

    def endpoint(trainer):
        with tf.GradientTape() as tape:
            tape.watch((points, targets, weights))
            point = trainer.origin_point_score_metrics_arrays(points, targets, weights)
            global_metrics = trainer.origin_global_score_metrics_arrays(targets[0], tf.ones([3], D))
            prefix = trainer.origin_prefix_score_metrics_arrays(points[:, :18], targets, tf.ones([4, 3], D))
            loss = point["loss"] + .03 * global_metrics["loss"] + .04 * prefix["loss"]
        gradients = tape.gradient(loss, (*trainer.trainable_variables, points, targets, weights))
        assert all(value is not None for value in gradients)
        return (point, global_metrics, prefix), gradients

    actual, expected = (endpoint(trainer) for trainer in trainers)
    for value, authority in zip(tf.nest.flatten(actual), tf.nest.flatten(expected), strict=True):
        np.testing.assert_allclose(value, authority, atol=1e-10, rtol=1e-10)
    programs = tuple(row[0].compiled for row in trainers[0]._execution_programs.values())
    assert len(programs) == 3
    assert all(program._jit_compile and program.input_signature for program in programs)
    assert all(program.experimental_get_tracing_count() == 1 for program in programs)
    with pytest.raises(tf.errors.InvalidArgumentError, match="invalid compiled density"):
        trainers[0].origin_global_score_metrics_arrays(targets[0], -tf.ones([3], D))
    # Changed variable values must not be captured by the shape-only cache.
    for trainer in trainers:
        trainer.trainable_variables[0].assign_add(tf.ones_like(trainer.trainable_variables[0]) * .002)
    for value, authority in zip(tf.nest.flatten(endpoint(trainers[0])),
                                tf.nest.flatten(endpoint(trainers[1])), strict=True):
        np.testing.assert_allclose(value, authority, atol=1e-10, rtol=1e-10)
    assert len(trainers[0]._execution_programs) == 3


@pytest.mark.parametrize("trainer_name", ["CenteredResidualTrainer", "CoreAffineTangentTrainer"])
def test_public_batch_metrics_and_absolute_loss_preserve_external_pullbacks(trainer_name, monkeypatch):
    parent = _fresh_parent()
    before = _original("zhao_cui_austria_sir_parameter_density_training_tf")
    contractions = _original("zhao_cui_austria_sir_centered_density_tf")
    for name in ("_evaluate_component", "_cross_mass", "_cross_prefix_values", "CenteredThetaFeatures"):
        monkeypatch.setattr(before, name, getattr(contractions, name))
    points, scores, weights = _inputs()
    points = tf.stack([points, points + .03])
    weights = tf.stack([weights, weights - .01])
    batch = candidate.T1ParameterDensityBatch(theta=tf.constant([[0., 0., 0.], [.02, -.01, .03]], D),
        physical_points=points, local_points=points, reference_points=tf.tanh(points),
        target_log_density_reference=weights + .1, proposal_log_density=tf.zeros([2, 4], D),
        coordinate_log_jacobian=tf.zeros([2, 4], D), observation_log_density=weights,
        complete_data_score=tf.stack([scores, scores + .03]), role="fresh_reference")
    inputs = (batch.theta, batch.local_points, batch.target_log_density_reference, batch.observation_log_density)
    trainers = tuple(getattr(module, trainer_name)(parent) for module in (candidate, before))

    def endpoint(trainer):
        with tf.GradientTape() as tape:
            tape.watch(inputs)
            metrics = trainer.heldout_metrics(batch)
            loss = tf.reduce_sum(metrics["normalized_log_density_rms"] + .1 * metrics["child_log_mass"])
            absolute = ()
            if trainer_name == "CenteredResidualTrainer":
                terms = trainer.absolute_density_loss(batch, l1_weight=1e-5, l2_weight=1e-4, derivative_weight=.02)
                loss += terms.total_loss
                absolute = tuple(getattr(terms, field.name) for field in fields(terms))
        gradients = tape.gradient(loss, (*trainer.trainable_variables, *inputs))
        assert all(value is not None for value in gradients)
        return (metrics, absolute), gradients

    for actual, expected in zip(tf.nest.flatten(endpoint(trainers[0])),
                                tf.nest.flatten(endpoint(trainers[1])), strict=True):
        np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)
    if trainer_name == "CenteredResidualTrainer":
        actual, expected = (trainer.absolute_density_loss_arrays(batch.theta, points, weights)
                            for trainer in trainers)
        for field in fields(actual):
            np.testing.assert_allclose(getattr(actual, field.name), getattr(expected, field.name),
                                       atol=1e-10, rtol=1e-10)
    programs = tuple(row[0].compiled for row in trainers[0]._execution_programs.values())
    assert len(programs) == (3 if trainer_name == "CenteredResidualTrainer" else 1)
    assert all(program._jit_compile and program.input_signature for program in programs)
    assert all(program.experimental_get_tracing_count() == 1 for program in programs)


@pytest.mark.parametrize("kind", ["absolute", "prefit", "total"])
def test_centered_callbacks_preserve_adam_updates_and_reject_invalid_xla(kind, monkeypatch):
    parent = _fresh_parent()
    before = _original("zhao_cui_austria_sir_parameter_density_training_tf")
    contractions = _original("zhao_cui_austria_sir_centered_density_tf")
    for name in ("_evaluate_component", "_cross_mass", "_cross_prefix_values", "CenteredThetaFeatures"):
        monkeypatch.setattr(before, name, getattr(contractions, name))
    trainers = (candidate.CenteredResidualTrainer(parent), before.CenteredResidualTrainer(parent))
    optimizers = tuple(tf.keras.optimizers.Adam(learning_rate=.001) for _ in trainers)
    for actual, expected in zip(trainers[0].trainable_variables, trainers[1].trainable_variables, strict=True):
        np.testing.assert_array_equal(actual, expected)
    call, inputs, invalid_index = _callback(candidate, trainers[0], optimizers[0], kind)
    reference, _, _ = _callback(before, trainers[1], optimizers[1], kind)
    for _ in range(2):
        actual, expected = call(*inputs), reference.python_function(*inputs)
        for value, authority in zip(actual, expected, strict=True):
            np.testing.assert_allclose(value, authority, atol=1e-10, rtol=1e-10)
        for value, authority in zip((*trainers[0].trainable_variables, *optimizers[0].variables),
                                     (*trainers[1].trainable_variables, *optimizers[1].variables), strict=True):
            np.testing.assert_allclose(value, authority, atol=1e-10, rtol=1e-10)
    assert call.specialization_cache_info().currsize == 1
    assert call.experimental_get_tracing_count() == 1
    assert "HloModule" in call.experimental_get_compiler_ir(*inputs)(stage="hlo")
    assert not any(node.op in ("PyFunc", "EagerPyFunc")
                   for node in call.get_concrete_function(*inputs).graph.as_graph_def().node)

    state = (*trainers[0].trainable_variables, *optimizers[0].variables)
    saved = tuple(value.numpy().copy() for value in state)
    invalid = list(inputs)
    invalid[invalid_index] = tf.fill(inputs[invalid_index].shape, tf.constant(float("nan"), D))
    # Check both the public dispatcher and its body inside an enclosing XLA call.
    enclosing = tf.function(call.python_function,
        input_signature=[tf.TensorSpec(value.shape, D) for value in inputs],
        jit_compile=True, autograph=False)
    for evaluate in (call, enclosing):
        rejected = evaluate(*invalid)
        assert not np.isfinite(rejected[-2].numpy())
        for actual, expected in zip(state, saved, strict=True):
            np.testing.assert_array_equal(actual, expected)
    if kind == "total":
        for index in (4, 7):
            invalid = list(inputs)
            invalid[index] = -tf.ones_like(inputs[index])
            rejected = call(*invalid)
            assert not np.isfinite(rejected[-2].numpy())
            for actual, expected in zip(state, saved, strict=True):
                np.testing.assert_array_equal(actual, expected)
    _graph(enclosing)
