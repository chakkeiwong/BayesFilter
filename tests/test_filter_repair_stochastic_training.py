"""Pinned value/gradient/update checks for the generic stochastic TT repair."""

from dataclasses import fields

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.highdim import stochastic_density_training as candidate
from bayesfilter.highdim import stochastic_training_native_tf as native
from bayesfilter.highdim.bases import BoundedInterval, LegendreBasis1D, ProductBasis
from tests.highdim.test_p75_stochastic_density_training import (
    _batch,
    _config,
    _convention,
    _initial_cores,
)
from tests.test_filter_repair_remaining_routes import _original

D = tf.float64


def _authority():
    original = _original("stochastic_density_training")
    config = _config()
    values = {field.name: getattr(config, field.name) for field in fields(config)}
    values.update(l1_weight=tf.constant(.013, D), logz_anchor_weight=tf.constant(.03, D),
                  logz_reference=tf.constant(-1., D))
    return (candidate.TrainableFunctionalTT(candidate.P75TrainableTTConfig(**values), _initial_cores()),
            original.TrainableFunctionalTT(original.P75TrainableTTConfig(**values), _initial_cores()))


@pytest.mark.parametrize("prefit", [False, True])
def test_complete_objective_input_and_core_gradients_and_adam_updates(prefit):
    actual, expected = _authority()
    batch = _batch()
    options = {"reference_cores": _initial_cores(), "reference_l2_weight": .025} if prefit else {}
    name = "square_root_prefit_objective" if prefit else "objective"
    records, derivatives = [], []
    for trainer in (actual, expected):
        with tf.GradientTape() as tape:
            tape.watch((batch.points, batch.target_values, batch.weights))
            terms = getattr(trainer, name)(batch, **options)
        records.append(terms)
        derivatives.append(tape.gradient(terms.total_loss,
            (*trainer.variables, batch.points, batch.target_values, batch.weights)))
    for field in fields(records[0]):
        if tf.is_tensor(getattr(records[0], field.name)):
            np.testing.assert_allclose(getattr(records[0], field.name), getattr(records[1], field.name), atol=1e-10, rtol=1e-10)
    for value, reference in zip(*derivatives, strict=True):
        assert value is not None and reference is not None
        np.testing.assert_allclose(value, reference, atol=1e-10, rtol=1e-10)
    direction, step = .1 * tf.cos(actual.cores[0]), 1e-5
    original_core = tf.identity(actual.cores[0])
    actual.cores[0].assign(original_core + step * direction)
    plus = getattr(actual, name)(batch, **options).total_loss
    actual.cores[0].assign(original_core - step * direction)
    minus = getattr(actual, name)(batch, **options).total_loss
    actual.cores[0].assign(original_core)
    np.testing.assert_allclose(tf.reduce_sum(derivatives[0][0] * direction), (plus-minus)/(2*step), atol=1e-8, rtol=1e-7)
    optimizers = [candidate.make_adam_optimizer(trainer.config) for trainer in (actual, expected)]
    step_name = "square_root_prefit_step" if prefit else "train_step"
    for _ in range(2):
        outputs = [getattr(trainer, step_name)(batch, optimizer, **options)
                   for trainer, optimizer in zip((actual, expected), optimizers, strict=True)]
        np.testing.assert_allclose(outputs[0].gradient_norm, outputs[1].gradient_norm, atol=1e-10, rtol=1e-10)
        for value, reference in zip(actual.variables, expected.variables, strict=True):
            np.testing.assert_allclose(value, reference, atol=1e-10, rtol=1e-10)
    programs = [row[0].compiled for row in actual._execution_programs.values()]
    programs += [row[1] for row in actual._optimizer_programs.values()]
    assert programs and all(program.experimental_get_tracing_count() == 1 for program in programs)
    assert all(program.input_signature is not None and program._jit_compile for program in programs)
    key, (_optimizer, program) = next(iter(actual._optimizer_programs.items()))
    del key
    tensors, *_ = native._arguments(actual, name, {"batch": batch, **options})
    assert "HloModule" in program.experimental_get_compiler_ir(*tensors)(stage="hlo")


@pytest.mark.parametrize("dimension,seed", [(2, 7501), (5, -17)])
def test_heterogeneous_core_values_mass_and_original_random_stream(dimension, seed):
    original = _original("stochastic_density_training")
    basis = ProductBasis(tuple(LegendreBasis1D(BoundedInterval(-1., 1.), axis % 3 + 1)
                               for axis in range(dimension)), _convention())
    ranks = (1, *(axis % 3 + 1 for axis in range(1, dimension)), 1)
    options = {"product_basis": basis, "ranks": ranks, "seed": seed}
    expected = original.TrainableFunctionalTT(original.P75TrainableTTConfig(**options))
    actual = candidate.TrainableFunctionalTT(candidate.P75TrainableTTConfig(**options))
    for value, reference in zip(actual.variables, expected.variables, strict=True):
        np.testing.assert_allclose(value, reference, atol=1e-14, rtol=1e-14)
    points = tf.reshape(tf.linspace(tf.constant(-.8, D), .7, dimension * 4), [4, dimension])
    np.testing.assert_allclose(actual.evaluate(points), expected.evaluate(points), atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(actual.normalizer(), expected.normalizer(), atol=1e-12, rtol=1e-12)
    program = native.random_core_program(tuple(tuple(core.shape) for core in actual.cores))
    assert "HloModule" in program.experimental_get_compiler_ir(tf.constant(seed, tf.int32))(stage="hlo")


def test_compiled_optimizer_rejects_invalid_prefit_weight_before_update():
    trainer, _expected = _authority()
    optimizer = candidate.make_adam_optimizer(trainer.config)
    before = tuple(tf.identity(core) for core in trainer.cores)
    with pytest.raises(tf.errors.InvalidArgumentError, match="invalid stochastic"):
        trainer.square_root_prefit_step(_batch(), optimizer, reference_l2_weight=-.01)
    for value, reference in zip(trainer.cores, before, strict=True):
        np.testing.assert_array_equal(value, reference)
    assert int(optimizer.iterations) == 0


@pytest.mark.parametrize("invalid", [-.01, float("nan")])
def test_enclosing_xla_rejects_invalid_prefit_without_mutating_parameters_or_slots(invalid):
    trainer, _expected = _authority()
    optimizer = candidate.make_adam_optimizer(trainer.config)
    optimizer.build(trainer.variables)
    batch = _batch()
    state = (*trainer.variables, *optimizer.variables)
    before = tuple(tf.identity(value) for value in state)

    @tf.function(input_signature=[tf.TensorSpec([], D)], jit_compile=True, autograph=False)
    def step(weight):
        result = trainer.square_root_prefit_step(batch, optimizer, reference_l2_weight=weight)
        return result.total_loss, result.gradient_norm

    _loss, norm = step(tf.constant(invalid, D))
    assert not bool(tf.math.is_finite(norm))
    for value, reference in zip(state, before, strict=True):
        np.testing.assert_array_equal(value, reference)


def test_explicit_graph_encloses_numerics_without_nested_xla():
    trainer, _expected = _authority()
    batch = _batch()
    program = tf.function(lambda points: (trainer.evaluate(points), trainer.normalizer()),
        input_signature=[tf.TensorSpec(batch.points.shape, D)], jit_compile=False, autograph=False)
    program(batch.points)
    graph = program.get_concrete_function().graph.as_graph_def()
    assert not any(function.attr.get("_XlaMustCompile") and function.attr["_XlaMustCompile"].b
                   for function in graph.library.function)
    assert not any(node.op in ("PyFunc", "EagerPyFunc", "PyFuncStateless")
                   for node in [*graph.node, *(n for f in graph.library.function for n in f.node_def)])
