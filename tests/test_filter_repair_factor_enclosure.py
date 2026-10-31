"""Diagnostic isolation of factor-fit arithmetic in an enclosing XLA program."""

import inspect
import json
from collections import namedtuple
from pathlib import Path

import numpy as np
import tensorflow as tf

from bayesfilter.inference import factor_correlation_geometry as factor
from tests.test_filter_repair_fixed_fitting import _inputs

D = tf.float64


def _enclose(program, dimension):
    @tf.function(input_signature=program.input_signature[:5],
        jit_compile=True, autograph=False)
    def enclosed(center, training, scores, heldout, heldout_scores):
        weights = tf.fill([3 * dimension], tf.constant(1. / (3 * dimension), D))
        return program(center, training, scores, heldout, heldout_scores, weights)
    return enclosed


def test_enclosed_initial_weights_value_and_gradient_localization(monkeypatch, request):
    """Observe unchanged inputs before attributing optimizer drift to a stage."""
    observed = namedtuple("InitialArithmetic", (
        "position", "objective_value", "objective_gradient", "converged", "failed",
        "num_iterations", "num_objective_evaluations", "weights", "response"))

    def observe(objective, *, initial_position, **_):
        closure = inspect.getclosurevars(objective.python_function).nonlocals
        loss = inspect.getclosurevars(closure["loss"]).nonlocals
        value, gradient = objective(initial_position)
        return observed(initial_position, value, gradient, tf.constant(False),
            tf.constant(False), tf.constant(0), tf.constant(1), loss["weights"],
            loss["train_response"])

    report = []
    with monkeypatch.context() as context:
        context.setattr(factor.tfp.optimizer, "lbfgs_minimize", observe)
        factor._make_factor_program.cache_clear()
        try:
            for dimension, factors in ((3, 1), (5, 1), (5, 2)):
                inputs = _inputs(dimension)
                config = factor.FactorCorrelationGeometryConfig(factor_count=factors)
                program = factor._make_factor_program(dimension, 3 * dimension, 2 * dimension,
                    config, True, factor._prediction_jacobian_diagnostics)

                enclosed = _enclose(program, dimension)

                for replicate in (0, 1):
                    arguments = tuple(tf.constant(value, D) for value in (
                        inputs[1], inputs[2][replicate], inputs[3][replicate],
                        inputs[4][replicate], inputs[5][replicate]))
                    weights = tf.fill([3 * dimension], tf.constant(1. / (3 * dimension), D))
                    before = program(*arguments, weights)["optimizer"]
                    after = enclosed(*arguments)["optimizer"]
                    fields = {}
                    for name in ("weights", "response", "position", "objective_value", "objective_gradient"):
                        left, right = getattr(before, name), getattr(after, name)
                        fields[name] = {"before": left.numpy().tolist(), "after": right.numpy().tolist(),
                            "max_error": float(tf.reduce_max(tf.abs(left - right)))}
                        assert bool(tf.reduce_all(tf.math.is_finite(left)))
                        assert bool(tf.reduce_all(tf.math.is_finite(right)))
                    np.testing.assert_array_equal(before.response, after.response)
                    report.append({"dimension": dimension, "factors": factors,
                        "replicate": replicate, "fields": fields})
        finally:
            factor._make_factor_program.cache_clear()
    directory = Path(request.config.getoption("xmlpath")).parent
    with (directory / "factor-enclosure-initial-arithmetic.json").open("x") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print("FACTOR_ENCLOSURE_INITIAL_ARITHMETIC " + json.dumps(report, sort_keys=True))
