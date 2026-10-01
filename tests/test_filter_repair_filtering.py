"""Frozen-source parity and compilation checks for retained-filter wrappers."""

from dataclasses import replace

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.highdim import filtering as current
from bayesfilter.highdim.filtering_native_tf import (
    gaussian_fit_program,
    gaussian_history_program,
    scalar_dense_program,
)
from tests.highdim.test_filtering_kalman_exact import (
    _filter_config,
    _scalar_model,
    _tt_filter_config,
)
from tests.highdim.test_p30_sv_short_sequential_tt_value_path import _model, _theta
from tests.test_filter_repair_scalar_retained import baseline  # noqa: F401

D = tf.float64


def test_heterogeneous_default_cores_match_original(baseline):  # noqa: F811
    config = _tt_filter_config()
    basis = current.ProductBasis(tuple(
        current.LegendreBasis1D(current.BoundedInterval(-1.0, 1.0), degree)
        for degree in (0, 2, 1)), config.measure_convention)
    fit = replace(config.fit_config, ranks=(1, 3, 2, 1), sweep_order=(0, 1, 2))
    actual = current._default_initial_cores(basis, fit)
    expected = baseline._default_initial_cores(basis, fit)
    for core, reference in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(core.values, reference.values)


@pytest.mark.parametrize("jit", [False, True])
def test_gaussian_tt_preserves_normalizer_veto(baseline, monkeypatch, jit):  # noqa: F811
    config = replace(_tt_filter_config(), normalizer_floor=1e8)
    frozen = baseline.FixedBranchFilterConfig(**config.__dict__)
    inputs = (_scalar_model(), tf.zeros([0], D), tf.constant([[0.2]], D))
    with monkeypatch.context() as patch:
        patch.setattr(current.FixedTTFitter, "fit", current.FixedTTFitter.fit_reference)
        with pytest.raises(ValueError, match="NORMALIZER_FLOOR_EXCEEDED"):
            baseline.FixedBranchSquaredTTFilter(frozen).log_likelihood(*inputs)
    with pytest.raises(ValueError, match="NORMALIZER_FLOOR_EXCEEDED"):
        current.FixedBranchSquaredTTFilter(config).log_likelihood(*inputs, jit_compile=jit)


@pytest.mark.parametrize("jit", [False, True])
@pytest.mark.parametrize("dates", [1, 3])
@pytest.mark.parametrize("method", ["dense", "kalman", "kalman_tt"])
def test_filter_histories_match_original(baseline, monkeypatch, method, dates, jit):  # noqa: F811
    config = _tt_filter_config() if method == "kalman_tt" else _filter_config()
    if method == "dense":
        config = replace(config, fit_quadrature_order=31,
            coordinate_maps=(current.AffineCoordinateMap(tf.constant([0.0], D), tf.constant([[3.0]], D)),))
    model = _model() if method == "dense" else _scalar_model()
    theta = _theta() if method == "dense" else tf.zeros([0], D)
    observations = tf.constant([[0.2], [-0.1], [0.4]], D)[:dates]
    frozen_config = baseline.FixedBranchFilterConfig(**config.__dict__)
    with monkeypatch.context() as patch:
        patch.setattr(current.FixedTTFitter, "fit", current.FixedTTFitter.fit_reference)
        expected = baseline.FixedBranchSquaredTTFilter(frozen_config).log_likelihood(
            model, theta, observations)
    actual = current.FixedBranchSquaredTTFilter(config).log_likelihood(
        model, theta, observations, jit_compile=jit)
    np.testing.assert_allclose(actual.log_likelihood, expected.log_likelihood, rtol=1e-10, atol=1e-10)
    for row, reference in zip(actual.steps, expected.steps, strict=True):
        for value, oracle in ((row.log_normalizer, reference.log_normalizer),
                (row.diagnostics["retained_mean"], reference.diagnostics["retained_mean"]),
                (row.diagnostics["retained_covariance"], reference.diagnostics["retained_covariance"])):
            np.testing.assert_allclose(value, oracle, rtol=1e-10, atol=1e-10)
        if method == "kalman_tt":
            np.testing.assert_allclose(row.fit_result.fitted_tt.cores[0].values,
                reference.fit_result.fitted_tt.cores[0].values, rtol=1e-10, atol=1e-10)
            np.testing.assert_allclose(row.diagnostics["tt_density_normalizer"],
                reference.diagnostics["tt_density_normalizer"], rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("method", ["dense", "kalman", "kalman_tt"])
def test_bounded_graph_and_runtime_inputs(method):
    counts = []
    model = _model() if method == "dense" else _scalar_model()
    config = _tt_filter_config()
    for dates in (2, 4):
        observations = tf.reshape(tf.linspace(tf.constant(0.1, D), tf.constant(0.4, D), dates), [-1, 1])
        if method == "dense":
            program = scalar_dense_program(model, observations.shape, 17)
            nodes, weights = current.legendre_gauss_nodes_weights(17)
            arguments = (_theta(), observations, nodes[:, None], weights, tf.zeros([17], D))
            changed = (arguments[0] + 0.1, *arguments[1:])
            field = lambda value: value["log_likelihood"]
        elif method == "kalman":
            program = gaussian_history_program(dates, 1, 1)
            arguments = (observations, model.initial_mean, model.initial_covariance,
                model.transition_matrix, model.transition_covariance, model.observation_matrix,
                model.observation_covariance, model.transition_offset, model.observation_offset)
            changed = (observations + 0.1, *arguments[1:])
            field = lambda value: value[2]
        else:
            prepared = gaussian_fit_program(config, dates, 1)
            program = prepared.call
            means = observations
            covariance = tf.ones([dates, 1, 1], D) * 0.2
            arguments = (means, covariance, prepared.points, prepared.weights,
                prepared.physical, prepared.logdet, prepared.core_values)
            changed = (means + 0.1, *arguments[1:])
            field = lambda value: value["target"]
        original = program(*arguments)
        updated = program(*changed)
        assert not np.allclose(field(original), field(updated))
        assert program.experimental_get_tracing_count() == 1
        graph = program.get_concrete_function().graph.as_graph_def()
        nodes = list(graph.node) + [node for function in graph.library.function for node in function.node_def]
        assert not {node.op for node in nodes} & {"PyFunc", "EagerPyFunc", "PyFuncStateless"}
        assert "HloModule" in program.experimental_get_compiler_ir(*arguments)(stage="hlo")
        counts.append(len(nodes))
    assert counts[0] == counts[1]
