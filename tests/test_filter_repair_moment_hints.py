"""Reference companion-filter parity and frozen callback graph compatibility."""

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.highdim.gaussian_moment_hints_tf import (
    _normal_hermite_rule,
    prepare_lgssm_moment_hints,
    prepare_sv_gaussian_moment_hints,
)
from docs.benchmarks.sv_fixture_c2_20260826 import (
    sv_gh_hint_factory,
    sv_model,
    sv_simulate,
)


@pytest.mark.parametrize("jit", [False, True])
@pytest.mark.parametrize("order", [2, 9, 17])
def test_normal_hermite_nodes_and_weights(jit, order):
    kernel = tf.function(
        lambda: _normal_hermite_rule(order, jit),
        input_signature=[], jit_compile=jit, autograph=False,
    )
    actual_nodes, actual_weights = kernel()
    nodes, weights = np.polynomial.hermite_e.hermegauss(order)
    weights /= np.sqrt(2.0 * np.pi)
    np.testing.assert_allclose(actual_nodes, nodes, atol=1e-13, rtol=1e-13)
    np.testing.assert_allclose(actual_weights, weights, atol=1e-13, rtol=1e-13)


@pytest.mark.parametrize("n,horizon", [(1, 1), (1, 4), (2, 4)])
@pytest.mark.parametrize("jit", [False, True])
def test_full_hint_recurrence_matches_original_numpy_companion(n, horizon, jit):
    model = sv_model(n, 194)
    observations = sv_simulate(model, horizon, 193)
    initial_reference, joint_reference = sv_gh_hint_factory(model)
    initial = initial_reference(observations[0])
    joints = [joint_reference(t, observations[t]) for t in range(1, horizon)]
    prepared = prepare_sv_gaussian_moment_hints(model, observations, jit_compile=jit)
    for left, right in zip((prepared.initial_mean, prepared.initial_covariance), initial, strict=True):
        np.testing.assert_allclose(left, right, rtol=1e-10, atol=1e-10)
    for t, (mean, covariance) in enumerate(joints):
        np.testing.assert_allclose(prepared.joint_means[t], mean, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(prepared.joint_covariances[t], covariance, rtol=1e-10, atol=1e-10)
    first, following = prepared.callbacks()

    @tf.function(input_signature=[tf.TensorSpec([], tf.int32)], jit_compile=True)
    def consumer(date):
        return following(date, None) if horizon > 1 else first(None)

    actual = consumer(tf.constant(horizon-1 if horizon > 1 else 0))
    expected = joints[-1] if joints else initial
    for left, right in zip(actual, expected, strict=True):
        np.testing.assert_allclose(left, right, rtol=1e-10, atol=1e-10)
    assert prepared.filtered_covariances().shape == (horizon, n, n)
    graph = consumer.get_concrete_function().graph.as_graph_def()
    assert not {n.op for n in graph.node} & {"PyFunc", "EagerPyFunc"}


@pytest.mark.parametrize("n,horizon", [(1, 1), (1, 4), (2, 4)])
@pytest.mark.parametrize("jit", [False, True])
def test_kalman_hint_history_preserves_original_frozen_fixture(n, horizon, jit):
    from docs.benchmarks.run_adapted_engine_validation_20260820 import (
        kalman_hint_factory,
    )
    from docs.benchmarks.run_n4_step_localization_20260819 import case_with_steps

    _, all_observations, _, model = case_with_steps(n, 42+n, include_model=True)
    observations = all_observations[:horizon]
    reference_joint, reference_start = kalman_hint_factory(n, 42+n)
    reference_start(observations[0])
    joints = [reference_joint(t, observations[t]) for t in range(1, horizon)]
    prepared = prepare_lgssm_moment_hints(observations, *model, jit_compile=jit)
    np.testing.assert_allclose(prepared.initial_mean, observations[0] * (2.0/3.0), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(prepared.initial_covariance, np.eye(n)/3.0, rtol=1e-12, atol=1e-12)
    for t, (mean, covariance) in enumerate(joints):
        np.testing.assert_allclose(prepared.joint_means[t], mean, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(prepared.joint_covariances[t], covariance, rtol=1e-12, atol=1e-12)


def test_adapted_engine_uses_tensor_indexed_kalman_hints():
    from bayesfilter.highdim.squared_tt_engine_adapted_tf import (
        run_value_filter_branch_axis_adapted_reference as run_value_filter_branch_axis_adapted,
    )
    from bayesfilter.highdim.squared_tt_engine_adapted_xla_tf import (
        run_value_filter_branch_axis_adapted_xla,
    )
    from bayesfilter.highdim.squared_tt_engine_v0_tf import EngineConfig
    from docs.benchmarks.run_adapted_engine_validation_20260820 import (
        kalman_hint_factory,
    )
    from docs.benchmarks.run_n4_step_localization_20260819 import case_with_steps

    adapter, all_observations, _, model = case_with_steps(1, 43, include_model=True)
    observations = all_observations[:3]
    reference_joint, reference_start = kalman_hint_factory(1, 43)
    reference_start(observations[0])
    config = EngineConfig(basis_degree=2, rank=2, row_count=64, sweeps=2,
        ridge=1e-10, tau=1e-6, coordinate_half_width=3.0, seed=7781, row_design="sobol")
    expected, _ = run_value_filter_branch_axis_adapted(
        adapter, observations, config, predictive_moment_hint=reference_joint,
    )
    _, joint_hint = prepare_lgssm_moment_hints(observations, *model).callbacks()
    actual, _ = run_value_filter_branch_axis_adapted_xla(
        adapter, observations, config, predictive_moment_hint=joint_hint,
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
