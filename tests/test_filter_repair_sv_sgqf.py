"""Pinned-source and finite-difference checks for compiled SV quadrature."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf
import tensorflow_probability as tfp

from bayesfilter.highdim import sv_mixture_cut4 as panel
from bayesfilter.highdim.sv_direct_sgqf_native_tf import direct_sgqf_program

D = tf.float64


@pytest.fixture(scope="module")
def baseline(tmp_path_factory):
    path = tmp_path_factory.mktemp("sv_sgqf_reference") / "sv_mixture_cut4.py"
    source = subprocess.check_output(
        [
            "git",
            "show",
            "3582b4ac5fea67fb5da7fa60a1cbfaf19df35adf:bayesfilter/highdim/sv_mixture_cut4.py",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
    )
    path.write_text(source)
    spec = importlib.util.spec_from_file_location("sv_sgqf_reference_diagnostic", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def inputs(dates=3, width=2):
    y = tf.constant([[0.6, -1.1], [-0.4, 0.3], [0.8, -0.7]], D)[:dates, :width]
    return (
        y,
        tf.constant([0.45, 0.72], D)[:width],
        tf.constant([0.8, 1.2], D)[:width],
        tf.constant([0.65, 0.9], D)[:width],
    )


@pytest.mark.parametrize("jit", [False, True])
@pytest.mark.parametrize("dates,width", [(1, 1), (3, 2)])
def test_direct_value_score_histories_match_pinned_source(baseline, jit, dates, width):
    y, gamma, beta, sigma = inputs(dates, width)
    cloud = panel.tf_fixed_sgqf_cloud(dim=1, sparse_level=3)
    arguments = dict(gamma=gamma, beta=beta, sigma=sigma, cloud=cloud)
    for name in (
        "exact_transformed_sv_independent_panel_fixed_sgqf_filter",
        "exact_transformed_sv_independent_panel_fixed_sgqf_score",
    ):
        expected = getattr(baseline, name)(y, **arguments)
        result = getattr(panel, name)(y, jit_compile=jit, **arguments)
        for field in ("log_likelihood", "mean_path", "covariance_path"):
            np.testing.assert_allclose(
                getattr(result, field), getattr(expected, field), rtol=1e-10, atol=1e-10
            )
        field = "score" if name.endswith("score") else "log_normalizers"
        np.testing.assert_allclose(
            getattr(result, field), getattr(expected, field), rtol=1e-10, atol=1e-10
        )
        assert result.diagnostics["jit_compile"] is jit


def test_direct_all_score_columns_fd_and_bounded_xla():
    cloud = panel.tf_fixed_sgqf_cloud(dim=1, sparse_level=3)
    nodes, weights = tf.reshape(cloud.points, [-1]), cloud.weights
    normal = tfp.distributions.Normal(tf.constant(0.0, D), tf.constant(1.0, D))
    counts = []
    for dates, width in ((1, 1), (3, 2)):
        y, gamma, beta, sigma = inputs(dates, width)
        z = tf.math.log(y**2)
        program = direct_sgqf_program(dates, width, cloud.point_count, with_score=True)
        assert program is direct_sgqf_program(
            dates, width, cloud.point_count, with_score=True
        )
        primal = direct_sgqf_program(dates, width, cloud.point_count)
        arguments = (z, gamma, beta, sigma, nodes, weights)
        output = program(*arguments)
        theta = tf.stack([normal.quantile(gamma), tf.math.log(beta)], axis=1)
        for column in range(2 * width):
            delta = (
                tf.reshape(tf.one_hot(column, 2 * width, dtype=D), [width, 2]) * 1e-5
            )
            plus, minus = theta + delta, theta - delta
            upper = tf.reduce_sum(
                primal(
                    z, normal.cdf(plus[:, 0]), tf.exp(plus[:, 1]), sigma, nodes, weights
                )[0]
            )
            lower = tf.reduce_sum(
                primal(
                    z,
                    normal.cdf(minus[:, 0]),
                    tf.exp(minus[:, 1]),
                    sigma,
                    nodes,
                    weights,
                )[0]
            )
            np.testing.assert_allclose(
                output[1][column], (upper - lower) / 2e-5, rtol=2e-7, atol=2e-7
            )
        changed = program(
            z + 0.04, gamma - 0.03, beta + 0.05, sigma + 0.1, nodes, weights
        )
        assert not np.isclose(tf.reduce_sum(changed[0]), tf.reduce_sum(output[0]))
        assert program.experimental_get_tracing_count() == 1
        graph = program.get_concrete_function().graph.as_graph_def()
        operations = list(graph.node) + [
            node for fn in graph.library.function for node in fn.node_def
        ]
        assert not {node.op for node in operations} & {
            "PyFunc",
            "EagerPyFunc",
            "PyFuncStateless",
        }
        assert "HloModule" in program.experimental_get_compiler_ir(*arguments)(
            stage="hlo"
        )
        counts.append(len(operations))
    assert counts[0] == counts[1]


def test_direct_validation_preserved():
    y, gamma, beta, sigma = inputs()
    with pytest.raises(ValueError, match="gamma"):
        panel.exact_transformed_sv_independent_panel_fixed_sgqf_score(
            y, gamma=-gamma, beta=beta, sigma=sigma
        )
    with pytest.raises(ValueError, match="one-dimensional"):
        panel.exact_transformed_sv_independent_panel_fixed_sgqf_filter(
            y,
            gamma=gamma,
            beta=beta,
            sigma=sigma,
            cloud=panel.tf_fixed_sgqf_cloud(dim=2, sparse_level=2),
        )


@pytest.mark.parametrize("jit", [False, True])
@pytest.mark.parametrize("with_score", [False, True])
def test_mixture_ukf_nested_execution_switch_preserves_value_and_score(baseline, jit, with_score):
    y, gamma, beta, sigma = inputs(2, 1)
    mixture = panel.SVLogChiSquareGaussianMixture(
        tf.constant([.4, .6], D), tf.constant([-1.1, .4], D),
        tf.constant([.7, 1.4], D), source="repair_frozen_fixture",
    )
    indices = tf.constant(panel._component_tuples(2, 1), tf.int32)
    suffix = "score" if with_score else "filter"
    expected = getattr(baseline, "independent_panel_sv_mixture_ukf_" + suffix)(
        y, gamma=gamma, beta=beta, sigma=sigma, mixture=mixture,
    )

    @tf.function(input_signature=[tf.TensorSpec(y.shape, D)], jit_compile=jit, autograph=False)
    def evaluate(observations):
        return panel._compiled_panel_mixture_ukf(
            tf.math.log(observations**2 + 1e-8), gamma, beta, sigma, mixture, indices,
            with_score=with_score, jit_compile=jit,
        )

    terms, scores, means, covariances, d_means, d_covariances, weights = evaluate(y)
    outputs = [(tf.reduce_sum(terms), expected.log_likelihood),
               (means, expected.mean_path), (covariances, expected.covariance_path)]
    if with_score:
        outputs += [(tf.reduce_sum(scores, axis=0), expected.score),
                    (d_means, expected.d_mean_path), (d_covariances, expected.d_covariance_path)]
    else:
        outputs += [(terms, expected.log_normalizers), (weights, expected.component_weights)]
    for actual, reference in outputs:
        np.testing.assert_allclose(actual, reference, rtol=1e-10, atol=1e-10)
    graph = evaluate.get_concrete_function().graph.as_graph_def()
    nodes = list(graph.node) + [node for fn in graph.library.function for node in fn.node_def]
    assert not {node.op for node in nodes} & {"PyFunc", "PyFuncStateless", "EagerPyFunc"}
    if jit:
        assert "HloModule" in evaluate.experimental_get_compiler_ir(y)(stage="hlo")
    else:
        assert not any(fn.attr.get("_XlaMustCompile") and fn.attr["_XlaMustCompile"].b
                       for fn in graph.library.function)
        assert not any("Xla" in node.op for node in nodes)


@pytest.mark.parametrize("jit", [False, True])
def test_augmented_value_histories_and_diagnostic_score_match_baseline(baseline, jit):
    y, gamma, beta, sigma = inputs()
    cloud = panel.tf_fixed_sgqf_cloud(dim=2, sparse_level=3)
    arguments = dict(gamma=gamma, beta=beta, sigma=sigma, cloud=cloud)
    name = "actual_transformed_sv_independent_panel_augmented_noise_fixed_sgqf_filter"
    expected = getattr(baseline, name)(y, **arguments, jit_compile=False)
    result = getattr(panel, name)(y, **arguments, jit_compile=jit)
    for field in ("log_likelihood", "log_normalizers", "mean_path", "covariance_path"):
        np.testing.assert_allclose(
            getattr(result, field), getattr(expected, field), rtol=1e-10, atol=1e-10
        )
    assert result.diagnostics["jit_compile"] is jit
    if not jit:
        name = (
            "actual_transformed_sv_independent_panel_augmented_noise_fixed_sgqf_score"
        )
        expected = getattr(baseline, name)(y, **arguments)
        result = getattr(panel, name)(y, **arguments)
        np.testing.assert_allclose(result.score, expected.score, rtol=1e-10, atol=1e-10)
        assert result.diagnostics["wrapper_score_contract"].startswith("gradient_tape_")


def test_augmented_signature_hlo_bounded_graph_and_failure(baseline):
    from bayesfilter.highdim.sv_panel_quadrature_native_tf import augmented_sgqf_program

    config = panel.TFFixedSGQFBranchConfig(
        predictive_epsilon=1e-10, innovation_epsilon=1e-10
    )
    cloud = panel.tf_fixed_sgqf_cloud(dim=2, sparse_level=3)
    counts = []
    for dates, width in ((1, 1), (3, 2)):
        arguments = (
            *inputs(dates, width),
            cloud.points,
            cloud.weights,
            tf.constant(1e-10, D),
        )
        program = augmented_sgqf_program(dates, width, cloud.point_count, config)
        result = program(*arguments)
        assert bool(tf.reduce_all(result["status_code"] == 0))
        changed = program(arguments[0] * 1.2, *arguments[1:])
        assert not np.allclose(result["log_normalizers"], changed["log_normalizers"])
        assert program.experimental_get_tracing_count() == 1
        graph = program.get_concrete_function().graph.as_graph_def()
        operations = list(graph.node) + [
            node for fn in graph.library.function for node in fn.node_def
        ]
        assert not {node.op for node in operations} & {
            "PyFunc",
            "EagerPyFunc",
            "PyFuncStateless",
        }
        assert "HloModule" in program.experimental_get_compiler_ir(*arguments)(
            stage="hlo"
        )
        counts.append(len(operations))
    assert counts[0] == counts[1]
    name = "actual_transformed_sv_independent_panel_augmented_noise_fixed_sgqf_filter"
    y, gamma, beta, sigma = inputs(1, 1)
    for module in (baseline, panel):
        with pytest.raises(
            ValueError,
            match="stage=numerical_output, time_index=0, reason=nonfinite_output_veto",
        ):
            getattr(module, name)(
                y * float("nan"),
                gamma=gamma,
                beta=beta,
                sigma=sigma,
                cloud=cloud,
                jit_compile=False,
            )


def mixture_fixture(components=2):
    return panel.SVLogChiSquareGaussianMixture(
        weights=tf.constant([0.4, 0.6], D) if components == 2 else tf.ones([1], D),
        means=tf.constant([-1.1, 0.4], D)[:components],
        variances=tf.constant([0.7, 1.4], D)[:components],
        source="repair_fixture",
    )


@pytest.mark.parametrize("jit", [False, True])
@pytest.mark.parametrize("dates,width", [(1, 1), (3, 2)])
@pytest.mark.parametrize(
    "method", ["kalman_filter", "cut4_filter", "fixed_sgqf_filter", "fixed_sgqf_score"]
)
def test_mixture_endpoints_match_original(baseline, jit, dates, width, method):
    y, gamma, beta, sigma = inputs(dates, width)
    arguments = dict(gamma=gamma, beta=beta, sigma=sigma, mixture=mixture_fixture())
    if "sgqf" in method:
        arguments["cloud"] = panel.tf_fixed_sgqf_cloud(dim=width, sparse_level=3)
    name = "independent_panel_sv_mixture_" + method
    expected = getattr(baseline, name)(y, **arguments)
    result = getattr(panel, name)(y, **arguments, jit_compile=jit)
    fields = ("log_likelihood", "mean_path", "covariance_path")
    fields += (
        ("score", "d_mean_path", "d_covariance_path")
        if method.endswith("score")
        else ("log_normalizers", "component_weights")
    )
    for field in fields:
        np.testing.assert_allclose(
            getattr(result, field), getattr(expected, field), rtol=1e-10, atol=1e-10
        )
    for key, value in expected.diagnostics.items():
        assert result.diagnostics[key] == value
    assert result.diagnostics["jit_compile"] is jit


@pytest.mark.parametrize("jit", [False, True])
def test_scalar_cut4_retains_own_collapse_and_padding(baseline, jit):
    model = panel.StochasticVolatilitySSM(sigma=0.6)
    theta = tf.constant([0.3, -0.2], D)
    y = inputs(3, 1)[0]
    mixture = mixture_fixture()
    expected = baseline.scalar_sv_mixture_cut4_filter(model, theta, y, mixture=mixture)
    result = panel.scalar_sv_mixture_cut4_filter(
        model, theta, y, mixture=mixture, jit_compile=jit
    )
    for field in (
        "log_likelihood",
        "log_normalizers",
        "mean_path",
        "variance_path",
        "component_weights",
    ):
        np.testing.assert_allclose(
            getattr(result, field), getattr(expected, field), rtol=1e-10, atol=1e-10
        )
    for key, value in expected.diagnostics.items():
        assert result.diagnostics[key] == value


def test_mixture_analytical_score_fd_and_signature_hlo():
    from bayesfilter.highdim.sv_panel_quadrature_native_tf import mixture_program

    counts = []
    normal = tfp.distributions.Normal(tf.constant(0.0, D), tf.constant(1.0, D))
    config = panel.TFFixedSGQFBranchConfig()
    for dates, width, components in ((1, 1, 1), (3, 1, 2), (1, 2, 1), (3, 2, 2)):
        y, gamma, beta, sigma = inputs(dates, width)
        z = tf.math.log(y**2 + tf.constant(1e-8, D))
        cloud = panel.tf_fixed_sgqf_cloud(dim=width, sparse_level=3)
        mixture = mixture_fixture(components)
        indices = tf.constant(panel._component_tuples(components, width), tf.int32)
        tail = (
            sigma,
            mixture.weights,
            mixture.means,
            mixture.variances,
            indices,
            cloud.points,
            cloud.weights,
            tf.constant(1e-12, D),
        )
        program = mixture_program(
            dates, width, components, cloud.point_count, config, with_score=True
        )
        primal = mixture_program(dates, width, components, cloud.point_count, config)
        output = program(z, gamma, beta, *tail)
        theta = tf.stack([normal.quantile(gamma), tf.math.log(beta)], axis=1)
        for column in range(2 * width):
            delta = (
                tf.reshape(tf.one_hot(column, 2 * width, dtype=D), [width, 2]) * 1e-5
            )
            plus, minus = theta + delta, theta - delta
            upper = tf.reduce_sum(
                primal(z, normal.cdf(plus[:, 0]), tf.exp(plus[:, 1]), *tail)[
                    "log_normalizers"
                ]
            )
            lower = tf.reduce_sum(
                primal(z, normal.cdf(minus[:, 0]), tf.exp(minus[:, 1]), *tail)[
                    "log_normalizers"
                ]
            )
            np.testing.assert_allclose(
                output["score"][column], (upper - lower) / 2e-5, rtol=2e-7, atol=2e-7
            )
        changed = program(z + 0.1, gamma - 0.05, beta + 0.04, *tail)
        assert not np.allclose(output["log_normalizers"], changed["log_normalizers"])
        assert program.experimental_get_tracing_count() == 1
        graph = program.get_concrete_function().graph.as_graph_def()
        operations = list(graph.node) + [
            node for fn in graph.library.function for node in fn.node_def
        ]
        assert not {node.op for node in operations} & {
            "PyFunc",
            "EagerPyFunc",
            "PyFuncStateless",
        }
        assert "HloModule" in program.experimental_get_compiler_ir(
            z, gamma, beta, *tail
        )(stage="hlo")
        counts.append(len(operations))
    # Scalar symmetrization specializes the graph; date/component counts do not.
    assert counts[0] == counts[1]
    assert counts[2] == counts[3]


def test_derivative_helpers_accept_tensor_dates_and_preserve_columns(baseline):
    cloud = tf.constant([[-0.4, 0.2], [0.7, -0.1], [0.5, 0.3]], D)
    for t in (0, 1):
        arguments = dict(
            d_current_mean=tf.constant([[0.1, 0.0], [-0.2, 0.0]], D),
            d_current_covariance=tf.constant(
                [[[0.3, 0.0], [0.0, 0.0]], [[-0.1, 0.0], [0.0, 0.0]]], D
            ),
            gamma=tf.constant(0.6, D),
            beta=tf.constant(0.8, D),
            sigma=tf.constant(0.7, D),
            observation_variance_floor=1e-10,
        )
        expected = (
            baseline._actual_transformed_sv_augmented_noise_fixed_sgqf_derivatives(
                time_index=t, **arguments
            )
        )

        @tf.function(jit_compile=True, autograph=False)
        def evaluate(time):
            derivative = (
                panel._actual_transformed_sv_augmented_noise_fixed_sgqf_derivatives(
                    time_index=time, **arguments
                )
            )
            return (
                derivative.transition_state_jacobian_fn(cloud),
                derivative.d_transition_fn(cloud),
                derivative.observation_state_jacobian_fn(cloud),
                derivative.d_observation_fn(cloud),
            )

        actual = evaluate(tf.constant(t))
        for result, name in zip(
            actual,
            (
                "transition_state_jacobian_fn",
                "d_transition_fn",
                "observation_state_jacobian_fn",
                "d_observation_fn",
            ),
            strict=True,
        ):
            np.testing.assert_allclose(
                result, getattr(expected, name)(cloud), rtol=1e-10, atol=1e-10
            )
