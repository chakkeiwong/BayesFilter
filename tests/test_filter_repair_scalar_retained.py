"""Independent legacy-loop and finite-difference authorities for native dates."""

import importlib.util
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.highdim import filtering as current
from bayesfilter.highdim.derivatives import FixedBranchDerivativeConfig
from bayesfilter.highdim.scalar_retained_native_tf import make_scalar_retained_program
from tests.highdim.test_p30_sv_short_sequential_tt_value_path import (
    _model,
    _observations,
    _theta,
    _tt_config,
)


@pytest.fixture(scope="module")
def baseline(tmp_path_factory):
    path = tmp_path_factory.mktemp("scalar_retained_reference") / "filtering.py"
    original = subprocess.check_output([
        "git", "show", "3582b4ac5fea67fb5da7fa60a1cbfaf19df35adf:bayesfilter/highdim/filtering.py",
    ], cwd=Path(__file__).resolve().parents[1], text=True)
    path.write_text(original)
    spec = importlib.util.spec_from_file_location("scalar_retained_reference_diagnostic", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _small_config():
    from bayesfilter.highdim import LegendreBasis1D, ProductBasis, TTCore
    config = _tt_config()
    part = config.product_basis.bases[0]
    basis = ProductBasis([LegendreBasis1D(part.domain, 12)], config.measure_convention)
    return replace(config, product_basis=basis, fit_quadrature_order=31,
                   initial_cores=(TTCore(tf.ones([1, basis.bases[0].basis_dim, 1], tf.float64)),))


@pytest.mark.parametrize("jit", [False, True])
def test_complete_value_analytical_score_and_histories_match_legacy(baseline, jit):
    model, config = _model(), _small_config()
    theta, observations = _theta(), _observations()
    derivative = FixedBranchDerivativeConfig(parameter_indices=(0, 1), finite_difference_h=())
    arguments = {"retained_moment_order": 33, "retained_propagation_order": 41}
    expected_config = baseline.FixedBranchFilterConfig(**config.__dict__)
    expected_value = baseline.scalar_nonlinear_fixed_design_tt_value_path(
        model, theta, observations, expected_config, **arguments)
    expected = baseline.scalar_nonlinear_fixed_design_tt_score_path(
        model, theta, observations, expected_config, derivative, **arguments)
    result = current.scalar_nonlinear_fixed_design_tt_score_path(
        model, theta, observations, config, derivative, jit_compile=jit, **arguments)
    value = current.scalar_nonlinear_fixed_design_tt_value_path(
        model, theta, observations, config, jit_compile=jit, **arguments)
    np.testing.assert_allclose(result.log_likelihood, expected.log_likelihood, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(result.score, expected.score, rtol=1e-10, atol=1e-10)
    for row, reference in zip(value.steps, expected_value.steps):
        for name in ("retained_mean", "retained_variance", "retained_moment_mass"):
            np.testing.assert_allclose(row.diagnostics[name], reference.diagnostics[name],
                                       rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(row.fit_result.fitted_tt.cores[0].values,
                                   reference.fit_result.fitted_tt.cores[0].values,
                                   rtol=1e-10, atol=1e-10)


def test_all_parameter_columns_match_finite_difference_and_bounded_xla():
    model, config, theta = _model(), _small_config(), _theta()
    derivative = FixedBranchDerivativeConfig(parameter_indices=(0, 1), finite_difference_h=())
    counts = []
    for horizon in (2, 4):
        observations = tf.tile(_observations(), [2, 1])[:horizon]
        kwargs = {"moment_order": 33, "propagation_order": 41}
        program = make_scalar_retained_program(model, config, observations.shape,
                                               derivative_config=derivative, **kwargs)
        assert program is make_scalar_retained_program(model, config, observations.shape,
                                                       derivative_config=derivative, **kwargs)
        output = program.call(theta, observations)
        primal = make_scalar_retained_program(model, config, observations.shape, **kwargs)
        for column in range(2):
            delta = tf.one_hot(column, 2, dtype=tf.float64) * 1e-5
            plus = primal.call(theta + delta, observations)["log_likelihood"]
            minus = primal.call(theta - delta, observations)["log_likelihood"]
            np.testing.assert_allclose(output["score"][column], (plus - minus) / 2e-5,
                                       rtol=2e-7, atol=2e-7)
        graph = program.call.get_concrete_function().graph.as_graph_def()
        nodes = list(graph.node) + [node for fn in graph.library.function for node in fn.node_def]
        assert not {node.op for node in nodes} & {"PyFunc", "EagerPyFunc", "PyFuncStateless"}
        assert program.call.experimental_get_tracing_count() == 1
        assert "HloModule" in program.call.experimental_get_compiler_ir(theta, observations)(stage="hlo")
        counts.append(len(nodes))
    assert counts[0] == counts[1]


def test_derivative_and_primal_vetoes_remain_closed():
    model, config = _model(), _small_config()
    derivative = FixedBranchDerivativeConfig(parameter_indices=(0,), finite_difference_h=(),
                                             solve_condition_number_veto=0.5)
    with pytest.raises(ValueError, match="DERIVATIVE_SOLVE_FAILURE"):
        current.scalar_nonlinear_fixed_design_tt_score_path(
            model, _theta(), _observations(), config, derivative,
            retained_moment_order=33, retained_propagation_order=41)


@pytest.mark.parametrize("horizon,lebesgue", [(1, False), (2, True)])
def test_defensive_mass_reordered_columns_and_changed_inputs(baseline, horizon, lebesgue):
    from bayesfilter.highdim import DensityMeasure, MassMeasure, ProductBasis

    model, config = _model(), _small_config()
    if lebesgue:
        measure = replace(config.measure_convention,
                          density_measure=DensityMeasure.REFERENCE_LEBESGUE,
                          mass_measure=MassMeasure.REFERENCE_LEBESGUE)
        config = replace(config, measure_convention=measure,
                         product_basis=ProductBasis(config.product_basis.bases, measure))
    config = replace(config, density_tau=0.002)
    derivative = FixedBranchDerivativeConfig(parameter_indices=(1, 0), finite_difference_h=())
    observations = _observations()[:horizon]
    arguments = {"retained_moment_order": 33, "retained_propagation_order": 41}
    program = make_scalar_retained_program(model, config, observations.shape,
        derivative_config=derivative, moment_order=33, propagation_order=41)
    for theta, data in ((_theta(), observations), (_theta() + 0.02, observations * 1.3)):
        expected = baseline.scalar_nonlinear_fixed_design_tt_score_path(
            model, theta, data, baseline.FixedBranchFilterConfig(**config.__dict__),
            derivative, **arguments)
        result = program.call(theta, data)
        program.validate(result["history"])
        np.testing.assert_allclose(result["log_likelihood"], expected.log_likelihood,
                                   rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(result["score"], expected.score, rtol=1e-10, atol=1e-10)
    assert program.call.experimental_get_tracing_count() == 1


def test_fit_resource_and_nonfinite_statuses_remain_distinct():
    model, config = _model(), _small_config()
    blocked = replace(config, fit_config=replace(config.fit_config, row_budget=1))
    program = make_scalar_retained_program(model, blocked, _observations().shape,
                                           moment_order=33, propagation_order=41)
    with pytest.raises(ValueError, match="COMPLEXITY_GATE"):
        program.validate(program.call(_theta(), _observations())["history"])
    program = make_scalar_retained_program(model, config, _observations().shape,
                                           moment_order=33, propagation_order=41)
    history = program.call(_theta(), _observations())["history"]
    invalid_fit = dict(history["fit"], valid=tf.constant([False, False]),
                       codes=tf.constant([[5, -1], [5, -1]]))
    with pytest.raises(ValueError, match="NONFINITE_VALUE"):
        program.validate(dict(history, fit=invalid_fit))
    with pytest.raises(ValueError, match="CONDITION_NUMBER_VETO"):
        current.scalar_nonlinear_fixed_design_tt_value_path(
            model, _theta(), _observations(), replace(config, fit_config=replace(
                config.fit_config, condition_number_veto=0.5)),
            retained_moment_order=33, retained_propagation_order=41)
