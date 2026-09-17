"""Pinned scalar and finite-difference authorities for independent SV panels."""

from dataclasses import replace

import numpy as np
import pytest
import tensorflow as tf
import tensorflow_probability as tfp

from bayesfilter.highdim import sv_mixture_cut4 as panel
from bayesfilter.highdim.derivatives import FixedBranchDerivativeConfig
from bayesfilter.highdim.sv_panel_retained_native_tf import make_sv_panel_retained_program
from tests.test_filter_repair_scalar_retained import _small_config, baseline  # noqa: F401

D = tf.float64


def _inputs(horizon=2, width=2):
    observations = tf.constant([[0.6, -1.1, 0.8], [-0.4, 0.3, -0.7]], D)[:horizon, :width]
    gamma = tf.constant([0.45, 0.72, 0.6], D)[:width]
    beta = tf.constant([0.8, 1.2, 0.9], D)[:width]
    sigma = tf.constant([0.65, 0.9, 0.75], D)[:width]
    return observations, gamma, beta, sigma


def _routes(mixture):
    if mixture is None:
        return (panel.exact_transformed_sv_independent_panel_zhaocui_tt_filter,
                panel.exact_transformed_sv_independent_panel_zhaocui_tt_score)
    return (panel.independent_panel_sv_mixture_zhaocui_tt_filter,
            panel.independent_panel_sv_mixture_zhaocui_tt_score)


@pytest.mark.parametrize("jit", [False, True])
@pytest.mark.parametrize("ksc,horizon", [(False, 1), (False, 2), (True, 2)])
def test_public_panel_matches_pinned_scalar_values_scores_and_artifacts(baseline, jit, ksc, horizon):
    observations, gamma, beta, sigma = _inputs(horizon)
    config = _small_config()
    derivative = FixedBranchDerivativeConfig(parameter_indices=(1, 0), finite_difference_h=())
    mixture = panel.ksc_1998_log_chi_square_mixture() if ksc else None
    z = (panel.transformed_sv_panel_observations(observations, offset=1e-8) if ksc
         else panel.exact_transformed_sv_observations(observations))
    value_route, score_route = _routes(mixture)
    arguments = dict(gamma=gamma, beta=beta, sigma=sigma, config=config, jit_compile=jit)
    if ksc:
        arguments["mixture"] = mixture
    value = value_route(observations, **arguments)
    result = score_route(observations, derivative_config=derivative, **arguments)
    expected_values, expected_scores, expected_increments = [], [], []
    for axis in range(2):
        model = (panel.KSCMixtureTransformedSVSSM(sigma=sigma[axis], mixture=mixture) if ksc
                 else panel.ExactTransformedSVSSM(sigma=sigma[axis]))
        theta = model.unconstrained_from_physical(gamma[axis], beta[axis])
        expected_config = baseline.FixedBranchFilterConfig(**config.__dict__)
        expected = baseline.scalar_nonlinear_fixed_design_tt_value_path(
            model, theta, z[:, axis:axis+1], expected_config)
        score = baseline.scalar_nonlinear_fixed_design_tt_score_path(
            model, theta, z[:, axis:axis+1], expected_config, derivative)
        expected_values.append(expected.log_likelihood)
        expected_scores.append(score.score)
        expected_increments.append(tf.stack([step.log_normalizer for step in expected.steps]))
        for date, reference in enumerate(expected.steps):
            if ksc:
                np.testing.assert_allclose(value.mean_path[date, axis],
                    reference.diagnostics["retained_mean"][0], rtol=1e-10, atol=1e-10)
                np.testing.assert_allclose(value.covariance_path[date, axis, axis],
                    reference.diagnostics["retained_variance"], rtol=1e-10, atol=1e-10)
            else:
                row = value.coordinate_results[axis].steps[date]
                for name in ("retained_mean", "retained_variance", "retained_moment_mass"):
                    np.testing.assert_allclose(row.diagnostics[name], reference.diagnostics[name],
                                               rtol=1e-10, atol=1e-10)
                np.testing.assert_allclose(row.fit_result.fitted_tt.cores[0].values,
                    reference.fit_result.fitted_tt.cores[0].values, rtol=1e-10, atol=1e-10)
                assert row.branch_identity.manifest.payload["model"]["sigma"] == sigma[axis]
                assert row.diagnostics["retained_propagation_order"] == 321
    np.testing.assert_allclose(value.log_likelihood, tf.reduce_sum(expected_values), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(result.log_likelihood, value.log_likelihood, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(result.score, tf.concat(expected_scores, axis=0), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(value.log_normalizers, tf.reduce_sum(expected_increments, axis=0),
                               rtol=1e-10, atol=1e-10)
    assert value.diagnostics["jit_compile"] is jit
    assert result.diagnostics["jit_compile"] is jit


@pytest.mark.parametrize("ksc", [False, True])
def test_native_panel_fd_cache_and_bounded_xla(ksc):
    config = _small_config()
    derivative = FixedBranchDerivativeConfig(parameter_indices=(1, 0), finite_difference_h=())
    mixture = panel.ksc_1998_log_chi_square_mixture() if ksc else None
    normal = tfp.distributions.Normal(tf.constant(0.0, D), tf.constant(1.0, D))
    counts = []
    for width in (1, 3):
        observations, gamma, beta, sigma = _inputs(width=width)
        z = tf.math.log(tf.square(observations) + (1e-8 if ksc else 0.0))
        program, _ = make_sv_panel_retained_program(config, z.shape, mixture=mixture,
                                                   derivative_config=derivative)
        assert program is make_sv_panel_retained_program(config, z.shape, mixture=mixture,
                                                         derivative_config=derivative)[0]
        primal, _ = make_sv_panel_retained_program(config, z.shape, mixture=mixture)
        output = program(z, gamma, beta, sigma)
        theta = tf.stack([normal.quantile(gamma), tf.math.log(beta)], axis=1)
        for axis in range(width):
            for column, parameter in enumerate((1, 0)):
                delta = tf.reshape(tf.one_hot(2 * axis + parameter, 2 * width, dtype=D), [width, 2]) * 1e-5
                plus, minus = theta + delta, theta - delta
                upper = primal(z, normal.cdf(plus[:, 0]), tf.exp(plus[:, 1]), sigma)["log_likelihood"]
                lower = primal(z, normal.cdf(minus[:, 0]), tf.exp(minus[:, 1]), sigma)["log_likelihood"]
                np.testing.assert_allclose(output["score"][2 * axis + column], (upper-lower)/2e-5,
                                           rtol=2e-7, atol=2e-7)
        changed = program(z + 0.04, gamma - 0.03, beta + 0.05, sigma + 0.1)
        assert not np.isclose(changed["log_likelihood"], output["log_likelihood"])
        assert program.experimental_get_tracing_count() == 1
        graph = program.get_concrete_function().graph.as_graph_def()
        nodes = list(graph.node) + [node for fn in graph.library.function for node in fn.node_def]
        assert not {node.op for node in nodes} & {"PyFunc", "EagerPyFunc", "PyFuncStateless"}
        assert "HloModule" in program.experimental_get_compiler_ir(z, gamma, beta, sigma)(stage="hlo")
        counts.append(len(nodes))
    assert counts[0] == counts[1]


def test_panel_preserves_column_selection_and_validation():
    data, gamma, beta, sigma = _inputs(horizon=1)
    config = _small_config()
    derivative = FixedBranchDerivativeConfig(parameter_indices=(0,), finite_difference_h=())
    arguments = dict(gamma=gamma, beta=beta, sigma=sigma, config=config)
    result = panel.exact_transformed_sv_independent_panel_zhaocui_tt_score(
        data, derivative_config=derivative, **arguments)
    full = panel.exact_transformed_sv_independent_panel_zhaocui_tt_score(
        data, derivative_config=replace(derivative, parameter_indices=(0, 1)), **arguments)
    np.testing.assert_array_equal(result.score, full.score)
    with pytest.raises(ValueError, match="DERIVATIVE_SOLVE_FAILURE"):
        panel.exact_transformed_sv_independent_panel_zhaocui_tt_score(
            data, derivative_config=replace(derivative, solve_condition_number_veto=0.5), **arguments)
    with pytest.raises(ValueError):
        panel.exact_transformed_sv_independent_panel_zhaocui_tt_filter(
            data, **dict(arguments, sigma=-sigma))
