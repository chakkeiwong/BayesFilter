"""Reference gates for full-mixture fixed-anchor IWSG resampling."""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.highdim.ledh_canonical_score_tf import (
    NonlinearScoreModel,
    _value_and_analytical_score_impl,
    canonical_value_and_analytical_score,
)
from bayesfilter.highdim.ledh_canonical_score_stages_tf import (
    ukf_predict_with_parameter_tangent,
)
from bayesfilter.highdim.ledh_younis_kdm_resampling_tf import (
    BANDWIDTH_POLICY,
    COVARIANCE_MARK_POLICY,
    DERIVATIVE_SEMANTICS,
    INCOMING_WEIGHT_POLICY,
    NORMALIZATION_TOLERANCE_POLICY,
    RESPONSIBILITY_COVARIANCE_MARK_POLICY,
    RESAMPLING_ROUTE_ID,
    SCORE_OUTPUT_SEMANTICS,
    SELECTED_LABEL_COVARIANCE_MARK_POLICY,
    make_resampling_anchor_kernel,
    make_resampling_replay_kernel,
)
from bayesfilter.highdim.ledh_younis_kdm_tf import (
    FULL_MIXTURE_RESAMPLING_ROUTE_ID,
    RESKDM_IWSG_FINITE_TARGET,
    make_full_mixture_iwsg_resampling_kernel,
    make_gaussian_kdm_kernel,
)


DTYPE = tf.float64


def _fixture():
    samples = np.array([[-0.45, 0.15], [0.35, -0.25], [1.15, 0.55]], dtype=np.float64)
    weights = np.array([0.2, 0.35, 0.45], dtype=np.float64)
    means = np.array([[-0.5, 0.1], [0.2, -0.35], [1.0, 0.65]], dtype=np.float64)
    bandwidths = np.array(
        [
            [[0.35, 0.04], [0.04, 0.25]],
            [[0.42, -0.03], [-0.03, 0.31]],
            [[0.28, 0.02], [0.02, 0.38]],
        ],
        dtype=np.float64,
    )
    marks = np.array(
        [
            [[0.7, 0.05], [0.05, 0.5]],
            [[0.45, -0.02], [-0.02, 0.8]],
            [[0.6, 0.03], [0.03, 0.65]],
        ],
        dtype=np.float64,
    )
    d_weights = np.array([[0.08, -0.03, -0.05]], dtype=np.float64)
    d_means = np.array([[[0.11, -0.07], [-0.13, 0.09], [0.04, 0.12]]], dtype=np.float64)
    d_bandwidths = np.array(
        [
            [
                [[0.025, -0.006], [-0.006, 0.018]],
                [[-0.014, 0.004], [0.004, 0.021]],
                [[0.017, 0.003], [0.003, -0.012]],
            ]
        ],
        dtype=np.float64,
    )
    d_marks = np.array(
        [
            [
                [[0.03, -0.004], [-0.004, -0.01]],
                [[-0.02, 0.006], [0.006, 0.025]],
                [[0.015, -0.003], [-0.003, 0.02]],
            ]
        ],
        dtype=np.float64,
    )
    return (
        samples,
        weights,
        means,
        bandwidths,
        marks,
        d_weights,
        d_means,
        d_bandwidths,
        d_marks,
    )


def _proposal_log_density(samples, weights, means, bandwidths):
    kernel = make_gaussian_kdm_kernel(
        evaluation_count=3,
        component_count=3,
        dimension=2,
        dtype=DTYPE,
        jit_compile=False,
    )
    result = kernel(
        tf.constant(samples, DTYPE),
        tf.constant(weights, DTYPE),
        tf.constant(means, DTYPE),
        tf.constant(bandwidths, DTYPE),
        tf.zeros([1, 3, 2], DTYPE),
        tf.zeros([1, 3], DTYPE),
        tf.zeros([1, 3, 2], DTYPE),
        tf.zeros([1, 3, 2, 2], DTYPE),
    )
    return result["log_density"].numpy()


def _kernel_call(fixture, proposal, *, zero_tangents=False):
    (
        samples,
        weights,
        means,
        bandwidths,
        marks,
        d_weights,
        d_means,
        d_bandwidths,
        d_marks,
    ) = fixture
    if zero_tangents:
        d_weights = np.zeros_like(d_weights)
        d_means = np.zeros_like(d_means)
        d_bandwidths = np.zeros_like(d_bandwidths)
        d_marks = np.zeros_like(d_marks)
    kernel = make_full_mixture_iwsg_resampling_kernel(
        particle_count=3,
        dimension=2,
        dtype=DTYPE,
        jit_compile=False,
    )
    result = kernel(
        tf.constant(samples, DTYPE),
        tf.constant(weights, DTYPE),
        tf.constant(means, DTYPE),
        tf.constant(bandwidths, DTYPE),
        tf.constant(marks, DTYPE),
        tf.constant(proposal, DTYPE),
        tf.constant(d_weights, DTYPE),
        tf.constant(d_means, DTYPE),
        tf.constant(d_bandwidths, DTYPE),
        tf.constant(d_marks, DTYPE),
    )
    return kernel, result


def test_full_mixture_resampling_joint_tangent_matches_finite_difference():
    fixture = _fixture()
    proposal = _proposal_log_density(*fixture[:4])
    kernel, result = _kernel_call(fixture, proposal)
    assert kernel.route_id == FULL_MIXTURE_RESAMPLING_ROUTE_ID
    assert kernel.target_label == RESKDM_IWSG_FINITE_TARGET
    assert bool(result["valid"].numpy())
    np.testing.assert_allclose(result["log_ratio"].numpy(), 0.0, atol=2e-14)
    np.testing.assert_allclose(
        result["normalized_weights"].numpy(), np.full(3, 1.0 / 3.0), atol=2e-14
    )
    np.testing.assert_allclose(
        result["importance_weights"].numpy(), np.full(3, 1.0 / 3.0), atol=2e-14
    )
    np.testing.assert_allclose(
        result["d_importance_log_weights"].numpy(),
        result["d_log_density"].numpy(),
        atol=0.0,
    )
    assert abs(float(result["importance_weight_tangent_sum"].numpy()[0])) > 1.0e-8
    assert int(result["complexity_pair_count"].numpy()) == 9

    epsilon = 2.0e-6
    plus = list(fixture)
    minus = list(fixture)
    for value_index, tangent_index in ((1, 5), (2, 6), (3, 7), (4, 8)):
        plus[value_index] = fixture[value_index] + epsilon * fixture[tangent_index][0]
        minus[value_index] = fixture[value_index] - epsilon * fixture[tangent_index][0]
    _, plus_result = _kernel_call(tuple(plus), proposal, zero_tangents=True)
    _, minus_result = _kernel_call(tuple(minus), proposal, zero_tangents=True)

    comparisons = (
        ("log_density", "d_log_density"),
        ("importance_log_weights", "d_importance_log_weights"),
        ("importance_weights", "d_importance_weights"),
        ("normalized_log_weights", "d_normalized_log_weights"),
        ("responsibilities", "d_responsibilities"),
        ("transported_covariance_marks", "d_transported_covariance_marks"),
    )
    for value_name, tangent_name in comparisons:
        finite_difference = (
            plus_result[value_name].numpy() - minus_result[value_name].numpy()
        ) / (2.0 * epsilon)
        np.testing.assert_allclose(
            result[tangent_name].numpy()[0],
            finite_difference,
            rtol=2.0e-6,
            atol=2.0e-7,
            err_msg=tangent_name,
        )


def test_full_mixture_resampling_each_declared_tangent_matches_finite_difference():
    fixture = _fixture()
    proposal = _proposal_log_density(*fixture[:4])
    epsilon = 2.0e-6
    dependency_pairs = (
        (1, 5, "component weights"),
        (2, 6, "component means"),
        (3, 7, "bandwidth covariances"),
        (4, 8, "covariance marks"),
    )

    for value_index, tangent_index, label in dependency_pairs:
        isolated = list(fixture)
        for index in range(5, 9):
            isolated[index] = np.zeros_like(fixture[index])
        isolated[tangent_index] = fixture[tangent_index]
        _, analytical = _kernel_call(tuple(isolated), proposal)

        plus = list(fixture)
        minus = list(fixture)
        plus[value_index] = fixture[value_index] + epsilon * fixture[tangent_index][0]
        minus[value_index] = fixture[value_index] - epsilon * fixture[tangent_index][0]
        _, plus_result = _kernel_call(tuple(plus), proposal, zero_tangents=True)
        _, minus_result = _kernel_call(tuple(minus), proposal, zero_tangents=True)

        comparisons = (
            ("transported_covariance_marks", "d_transported_covariance_marks"),
            ("importance_log_weights", "d_importance_log_weights"),
            ("importance_weights", "d_importance_weights"),
            (
                "normalized_log_weights",
                "d_normalized_log_weights",
            ),
            ("responsibilities", "d_responsibilities"),
            ("log_density", "d_log_density"),
        )
        for value_name, tangent_name in comparisons:
            finite_difference = (
                plus_result[value_name].numpy() - minus_result[value_name].numpy()
            ) / (2.0 * epsilon)
            np.testing.assert_allclose(
                analytical[tangent_name].numpy()[0],
                finite_difference,
                rtol=2.0e-6,
                atol=2.0e-7,
                err_msg=f"{label}: {tangent_name}",
            )


def test_resampling_normalization_permutation_and_all_component_influence():
    fixture = _fixture()
    proposal = _proposal_log_density(*fixture[:4])
    _, result = _kernel_call(fixture, proposal)
    np.testing.assert_allclose(
        result["responsibilities"].numpy().sum(axis=1), 1.0, atol=2e-14
    )
    np.testing.assert_allclose(
        result["d_responsibilities"].numpy().sum(axis=2), 0.0, atol=2e-14
    )
    np.testing.assert_allclose(
        result["d_normalized_weights"].numpy().sum(axis=1), 0.0, atol=2e-14
    )

    order = np.array([2, 0, 1])
    permuted = list(fixture)
    for index in (1, 2, 3, 4):
        permuted[index] = fixture[index][order]
    for index in (5, 6, 7, 8):
        permuted[index] = fixture[index][:, order]
    _, permuted_result = _kernel_call(tuple(permuted), proposal)
    for name in (
        "log_density",
        "d_log_density",
        "normalized_weights",
        "d_normalized_weights",
        "transported_covariance_marks",
        "d_transported_covariance_marks",
    ):
        np.testing.assert_allclose(
            result[name].numpy(), permuted_result[name].numpy(), atol=3e-14
        )
    np.testing.assert_allclose(
        result["responsibilities"].numpy()[:, order],
        permuted_result["responsibilities"].numpy(),
        atol=3e-14,
    )

    third_component_only = list(fixture)
    third_component_only[5] = np.zeros_like(fixture[5])
    third_component_only[6] = np.zeros_like(fixture[6])
    third_component_only[7] = np.zeros_like(fixture[7])
    third_component_only[8] = np.zeros_like(fixture[8])
    third_component_only[6][0, 2, 0] = 0.2
    _, third_result = _kernel_call(tuple(third_component_only), proposal)
    assert abs(float(third_result["d_log_density"].numpy()[0, 0])) > 1.0e-8


def test_resampling_fails_closed_for_invalid_declared_inputs():
    def assert_invalid(candidate, candidate_proposal):
        _, result = _kernel_call(tuple(candidate), candidate_proposal)
        assert not bool(result["valid"].numpy())
        assert np.all(np.isnan(result["normalized_weights"].numpy()))

    fixture = list(_fixture())
    proposal = _proposal_log_density(*fixture[:4])
    fixture[3] = fixture[3].copy()
    fixture[3][1, 0, 0] = -1.0
    assert_invalid(fixture, proposal)

    fixture = list(_fixture())
    fixture[3] = fixture[3].copy()
    fixture[3][0, 0, 1] += 0.2
    assert_invalid(fixture, proposal)

    fixture = list(_fixture())
    fixture[7] = fixture[7].copy()
    fixture[7][0, 0, 0, 1] += 0.2
    assert_invalid(fixture, proposal)

    fixture = list(_fixture())
    fixture[4] = fixture[4].copy()
    fixture[4][0, 0, 1] += 0.2
    assert_invalid(fixture, proposal)

    fixture = list(_fixture())
    fixture[8] = fixture[8].copy()
    fixture[8][0, 0, 0, 1] += 0.2
    assert_invalid(fixture, proposal)

    fixture = list(_fixture())
    fixture[0] = fixture[0].copy()
    fixture[0][0, 0] = np.nan
    assert_invalid(fixture, proposal)

    fixture = list(_fixture())
    fixture[1] = 1.1 * fixture[1]
    assert_invalid(fixture, proposal)

    invalid_proposal = proposal.copy()
    invalid_proposal[0] = np.nan
    assert_invalid(list(_fixture()), invalid_proposal)


def _sequential_model(dtype=DTYPE):
    def transition_mean_fn(theta, points):
        return points + theta[0] * tf.sin(points)

    def transition_mean_tangent_fn(theta, points, d_points):
        return tf.sin(points) + d_points + theta[0] * tf.cos(points) * d_points

    return NonlinearScoreModel(
        transition_mean_fn=transition_mean_fn,
        transition_mean_tangent_fn=transition_mean_tangent_fn,
        observation_fn=lambda points: points,
        observation_jacobian_fn=lambda points: tf.broadcast_to(
            tf.eye(2, dtype=points.dtype), [tf.shape(points)[0], 2, 2]
        ),
        observation_tangent_fn=lambda points, d_points: d_points,
        process_covariance=0.35 * tf.eye(2, dtype=dtype),
        observation_covariance=0.55 * tf.eye(2, dtype=dtype),
        observation_jacobian_tangent_fn=lambda points, d_points: tf.zeros(
            [tf.shape(points)[0], 2, 2], dtype=points.dtype
        ),
    )


def _sequential_fixture(dtype=DTYPE):
    rng = np.random.default_rng(2026090817)
    particle_count, dimension, horizon = 8, 2, 2
    initial_states = tf.constant(
        rng.normal(scale=0.7, size=(particle_count, dimension)), dtype
    )
    initial_covariances = tf.constant(
        np.stack(
            [
                np.array([[0.45 + 0.02 * i, 0.01], [0.01, 0.6 + 0.015 * i]])
                for i in range(particle_count)
            ]
        ),
        dtype,
    )
    noises = tf.constant(rng.normal(size=(horizon, particle_count, dimension)), dtype)
    observations = tf.constant(rng.normal(scale=0.8, size=(horizon, dimension)), dtype)
    bandwidths = tf.broadcast_to(
        tf.constant([[0.09, 0.012], [0.012, 0.07]], dtype),
        [horizon, dimension, dimension],
    )
    d_bandwidths = tf.zeros_like(bandwidths)
    stratified_uniforms = tf.constant(
        rng.uniform(0.05, 0.95, size=(horizon, particle_count)), dtype
    )
    kdm_noises = tf.constant(
        rng.normal(scale=0.6, size=(horizon, particle_count, dimension)), dtype
    )
    base = np.concatenate([np.eye(dimension), -np.eye(dimension)], axis=0)
    options = {
        "flow_substeps": 4,
        "reset_policy": "contract_e",
        "reset_design": tf.constant(
            np.tile(base, (particle_count // (2 * dimension), 1)), dtype
        ),
        "reset_sinkhorn_steps": 4,
        "reset_balance_steps": 2,
        "correction_steps": 1,
        "pairwise_steps": 1,
        "coordinate_cap": 0.95,
    }
    return {
        "theta": tf.constant([0.42], dtype),
        "initial_states": initial_states,
        "initial_covariances": initial_covariances,
        "noises": noises,
        "observations": observations,
        "bandwidths": bandwidths,
        "d_bandwidths": d_bandwidths,
        "stratified_uniforms": stratified_uniforms,
        "kdm_noises": kdm_noises,
        "options": options,
    }


def _anchor_and_replay_kernels(
    fixture, covariance_mark_policy=RESPONSIBILITY_COVARIANCE_MARK_POLICY
):
    common = {
        "theta_dimension": 1,
        "particle_count": 8,
        "state_dimension": 2,
        "observation_dimension": 2,
        "horizon": 2,
        "canonical_options": fixture["options"],
        "dtype": fixture["theta"].dtype,
        "jit_compile": False,
        "covariance_mark_policy": covariance_mark_policy,
    }
    model = _sequential_model(fixture["theta"].dtype)
    return (
        make_resampling_anchor_kernel(model, **common),
        make_resampling_replay_kernel(model, **common),
    )


def _shared_sequential_arguments(fixture, theta=None):
    return (
        fixture["theta"] if theta is None else theta,
        fixture["initial_states"],
        fixture["initial_covariances"],
        fixture["noises"],
        fixture["observations"],
        fixture["bandwidths"],
        fixture["d_bandwidths"],
    )


def test_sequential_anchor_replay_and_total_tangent_match():
    fixture = _sequential_fixture()
    anchor_kernel, replay_kernel = _anchor_and_replay_kernels(fixture)
    anchor = anchor_kernel(
        *_shared_sequential_arguments(fixture),
        fixture["stratified_uniforms"],
        fixture["kdm_noises"],
    )
    assert anchor_kernel.route_id == RESAMPLING_ROUTE_ID
    assert anchor_kernel.target_label == RESKDM_IWSG_FINITE_TARGET
    assert anchor_kernel.bandwidth_policy == BANDWIDTH_POLICY
    assert anchor_kernel.incoming_weight_policy == INCOMING_WEIGHT_POLICY
    assert anchor_kernel.covariance_mark_policy == COVARIANCE_MARK_POLICY
    assert anchor_kernel.derivative_semantics == DERIVATIVE_SEMANTICS
    assert anchor_kernel.score_output_semantics == SCORE_OUTPUT_SEMANTICS
    assert anchor_kernel.normalization_tolerance_policy == (
        NORMALIZATION_TOLERANCE_POLICY
    )
    np.testing.assert_equal(
        anchor_kernel.effective_normalization_tolerance,
        max(
            1.0e-8,
            8.0
            * 8.0
            * np.finfo(fixture["theta"].dtype.as_numpy_dtype).eps,
        ),
    )
    assert tuple(anchor_kernel.input_signature[5].shape) == (2, 2, 2)
    assert bool(anchor["valid"].numpy())
    assert int(anchor["pair_count"].numpy()) == 2 * 8 * 8
    # N strata and N uniform mixture masses select each component exactly once.
    # This is the sampling law used in the note's small-bandwidth argument.
    np.testing.assert_array_equal(
        anchor["component_indices"].numpy(), np.tile(np.arange(8), (2, 1))
    )
    assert np.all(anchor["higher_moment_valid"].numpy())
    assert np.all(anchor["higher_moment_pairwise_configured"].numpy())
    np.testing.assert_array_equal(
        anchor["higher_moment_pairwise_target_mask"].numpy(),
        np.tile(np.array([[[0.0, 1.0], [1.0, 0.0]]]), (2, 1, 1)),
    )
    assert (
        np.max(anchor["higher_moment_maximum_pairwise_pre_cap_particle_rms"].numpy())
        > 0.0
    )
    assert np.all(
        anchor["higher_moment_minimum_pairwise_particle_cap_scale"].numpy() > 0.0
    )
    assert np.all(
        anchor["higher_moment_minimum_pairwise_particle_cap_scale"].numpy() <= 1.0
    )
    assert np.all(
        anchor["higher_moment_minimum_coordinatewise_cap_derivative"].numpy() > 0.0
    )
    assert np.all(
        anchor["higher_moment_minimum_coordinatewise_cap_derivative"].numpy() <= 1.0
    )
    np.testing.assert_array_equal(
        anchor["component_indices"].numpy(), np.tile(np.arange(8), (2, 1))
    )
    np.testing.assert_allclose(
        anchor["maximum_anchor_log_ratio_error"].numpy(), 0.0, atol=2e-14
    )
    np.testing.assert_allclose(
        anchor["normalized_weights"].numpy(), 1.0 / 8.0, atol=2e-14
    )
    np.testing.assert_allclose(
        anchor["importance_weights"].numpy(), 1.0 / 8.0, atol=2e-14
    )
    assert np.max(np.abs(anchor["d_importance_weight_sums"].numpy())) > 1.0e-8
    np.testing.assert_allclose(
        anchor["d_outgoing_log_weights"].numpy(),
        anchor["d_importance_log_weights"].numpy(),
        atol=0.0,
    )
    assert (
        np.max(
            np.abs(
                anchor["d_outgoing_log_weights"].numpy()
                - anchor["d_normalized_weights"].numpy()
                / anchor["normalized_weights"].numpy()
            )
        )
        > 1.0e-8
    )
    np.testing.assert_allclose(
        anchor["d_states_after_resampling"].numpy(), 0.0, atol=0.0
    )

    replay = replay_kernel(
        *_shared_sequential_arguments(fixture),
        anchor["fixed_samples"],
        anchor["fixed_proposal_log_densities"],
        anchor["component_indices"],
    )
    assert replay_kernel.mode == "replay"
    assert bool(replay["valid"].numpy())
    np.testing.assert_allclose(
        replay["value"].numpy(), anchor["value"].numpy(), atol=0.0
    )
    np.testing.assert_allclose(
        replay["score"].numpy(), anchor["score"].numpy(), atol=0.0
    )
    np.testing.assert_allclose(
        replay["fixed_samples"].numpy(), anchor["fixed_samples"].numpy(), atol=0.0
    )
    np.testing.assert_allclose(
        replay["incoming_log_weights"].numpy()[1],
        replay["outgoing_log_weights"].numpy()[0],
        atol=0.0,
    )
    np.testing.assert_allclose(
        replay["d_incoming_log_weights"].numpy()[1],
        replay["d_outgoing_log_weights"].numpy()[0],
        atol=0.0,
    )
    assert np.max(np.abs(replay["d_incoming_log_weights"].numpy()[1])) > 1.0e-8

    model = _sequential_model()

    def mean_fn(points):
        return model.transition_mean_fn(fixture["theta"], points)

    def mean_tangent_fn(points, d_points):
        return model.transition_mean_tangent_fn(fixture["theta"], points, d_points)

    expected_prediction = ukf_predict_with_parameter_tangent(
        replay["states_after_resampling"][0],
        replay["covariance_marks"][0],
        replay["d_states_after_resampling"][0],
        replay["d_covariance_marks"][0],
        mean_fn,
        mean_tangent_fn,
        model.process_covariance,
    )
    np.testing.assert_allclose(
        replay["predicted_covariances"].numpy()[1],
        expected_prediction[1].numpy(),
        rtol=2e-12,
        atol=2e-12,
    )
    np.testing.assert_allclose(
        replay["d_predicted_covariances"].numpy()[1],
        expected_prediction[3].numpy(),
        rtol=2e-12,
        atol=2e-12,
    )

    epsilon = 1.0e-5
    plus = replay_kernel(
        *_shared_sequential_arguments(
            fixture, fixture["theta"] + tf.constant([epsilon], DTYPE)
        ),
        anchor["fixed_samples"],
        anchor["fixed_proposal_log_densities"],
        anchor["component_indices"],
    )
    minus = replay_kernel(
        *_shared_sequential_arguments(
            fixture, fixture["theta"] - tf.constant([epsilon], DTYPE)
        ),
        anchor["fixed_samples"],
        anchor["fixed_proposal_log_densities"],
        anchor["component_indices"],
    )
    finite_difference = (plus["value"] - minus["value"]) / (2.0 * epsilon)
    np.testing.assert_allclose(
        replay["score"].numpy()[0],
        finite_difference.numpy(),
        rtol=3.0e-4,
        atol=3.0e-5,
    )

    per_component_bandwidths = tf.broadcast_to(
        fixture["bandwidths"][:, tf.newaxis, :, :], [2, 8, 2, 2]
    )
    with pytest.raises((TypeError, ValueError)):
        anchor_kernel(
            fixture["theta"],
            fixture["initial_states"],
            fixture["initial_covariances"],
            fixture["noises"],
            fixture["observations"],
            per_component_bandwidths,
            tf.zeros_like(per_component_bandwidths),
            fixture["stratified_uniforms"],
            fixture["kdm_noises"],
        )


def test_selected_label_covariance_mark_ablation_is_fixed_and_total():
    fixture = _sequential_fixture()
    responsibility_anchor, _ = _anchor_and_replay_kernels(fixture)
    selected_anchor_kernel, selected_replay_kernel = _anchor_and_replay_kernels(
        fixture, SELECTED_LABEL_COVARIANCE_MARK_POLICY
    )
    shared = _shared_sequential_arguments(fixture)
    responsibility = responsibility_anchor(
        *shared,
        fixture["stratified_uniforms"],
        fixture["kdm_noises"],
    )
    selected = selected_anchor_kernel(
        *shared,
        fixture["stratified_uniforms"],
        fixture["kdm_noises"],
    )
    assert selected_anchor_kernel.covariance_mark_policy == (
        SELECTED_LABEL_COVARIANCE_MARK_POLICY
    )
    for time_index in range(2):
        expected = selected["source_covariance_marks"].numpy()[time_index][
            selected["component_indices"].numpy()[time_index]
        ]
        np.testing.assert_allclose(
            selected["covariance_marks"].numpy()[time_index],
            expected,
            atol=0.0,
        )
    assert abs(float(selected["value"].numpy() - responsibility["value"].numpy())) > (
        1.0e-8
    )

    replay_args = (
        selected["fixed_samples"],
        selected["fixed_proposal_log_densities"],
        selected["component_indices"],
    )
    replay = selected_replay_kernel(*shared, *replay_args)
    epsilon = tf.constant(1.0e-5, DTYPE)
    plus = selected_replay_kernel(
        *_shared_sequential_arguments(fixture, fixture["theta"] + epsilon),
        *replay_args,
    )
    minus = selected_replay_kernel(
        *_shared_sequential_arguments(fixture, fixture["theta"] - epsilon),
        *replay_args,
    )
    finite_difference = (plus["value"] - minus["value"]) / (2.0 * epsilon)
    np.testing.assert_allclose(
        replay["score"].numpy()[0],
        finite_difference.numpy(),
        rtol=3.0e-4,
        atol=3.0e-5,
    )

    invalid_indices = tf.tensor_scatter_nd_update(
        selected["component_indices"], [[0, 0]], [tf.constant(8, tf.int32)]
    )
    invalid = selected_replay_kernel(
        *shared,
        selected["fixed_samples"],
        selected["fixed_proposal_log_densities"],
        invalid_indices,
    )
    assert not bool(invalid["valid"].numpy())


def test_unknown_covariance_mark_policy_is_rejected():
    fixture = _sequential_fixture()
    with pytest.raises(ValueError, match="covariance_mark_policy"):
        make_resampling_anchor_kernel(
            _sequential_model(),
            theta_dimension=1,
            particle_count=8,
            state_dimension=2,
            observation_dimension=2,
            horizon=2,
            canonical_options=fixture["options"],
            covariance_mark_policy="undeclared_mark_policy",
            dtype=DTYPE,
            jit_compile=False,
        )


def test_post_reset_extension_preserves_canonical_no_hook_behavior():
    fixture = _sequential_fixture()
    model = _sequential_model()
    uniform_log = -tf.math.log(tf.constant(8.0, DTYPE)) * tf.ones([8], DTYPE)

    def identity_hook(time_index, states, d_states, covariances, d_covariances):
        del time_index
        return (
            states,
            d_states,
            covariances,
            d_covariances,
            uniform_log,
            tf.zeros_like(uniform_log),
            {},
        )

    baseline_value, baseline_score, baseline_trace = (
        canonical_value_and_analytical_score(
            model,
            *_shared_sequential_arguments(fixture)[:5],
            with_score=True,
            return_trace=True,
            **fixture["options"],
        )
    )
    hooked_value, hooked_score, hooked_trace = _value_and_analytical_score_impl(
        model,
        *_shared_sequential_arguments(fixture)[:5],
        with_score=True,
        return_trace=True,
        observation_factor_override=None,
        post_reset_transform=identity_hook,
        **fixture["options"],
    )
    np.testing.assert_array_equal(hooked_value.numpy(), baseline_value.numpy())
    np.testing.assert_array_equal(hooked_score.numpy(), baseline_score.numpy())
    for step_index in range(2):
        for name in (
            "states_after_reset",
            "d_states_after_reset",
            "covariances_after_reset",
            "d_covariances_after_reset",
            "outgoing_log_weights",
            "d_outgoing_log_weights",
            "higher_moment_valid",
            "higher_moment_pairwise_configured",
            "higher_moment_pairwise_target_mask",
            "higher_moment_maximum_pairwise_pre_cap_particle_rms",
            "higher_moment_minimum_pairwise_particle_cap_scale",
            "higher_moment_minimum_coordinatewise_cap_derivative",
        ):
            np.testing.assert_array_equal(
                hooked_trace[step_index][name].numpy(),
                baseline_trace[step_index][name].numpy(),
            )


def test_post_reset_state_weight_and_covariance_values_reach_next_pfpf_step():
    fixture = _sequential_fixture()
    model = _sequential_model()
    options = fixture["options"]
    uniform_log = -tf.math.log(tf.constant(8.0, DTYPE)) * tf.ones([8], DTYPE)
    state_shift = tf.constant(
        [[0.02, -0.01], [-0.01, 0.015], [0.012, 0.008], [-0.016, -0.006]],
        DTYPE,
    )
    state_shift = tf.tile(state_shift, [2, 1])
    weight_shift = tf.constant(
        [0.10, -0.08, 0.06, -0.04, 0.025, -0.015, -0.03, -0.02],
        DTYPE,
    )

    _, _, baseline_trace = canonical_value_and_analytical_score(
        model,
        *_shared_sequential_arguments(fixture)[:5],
        with_score=True,
        return_trace=True,
        **options,
    )

    def run_hook(kind):
        def hook(time_index, states, d_states, covariances, d_covariances):
            if time_index != 0:
                return (
                    states,
                    d_states,
                    covariances,
                    d_covariances,
                    uniform_log,
                    tf.zeros_like(uniform_log),
                    {},
                )
            next_states = states + state_shift if kind == "state" else states
            next_covariances = (
                covariances + 0.04 * tf.eye(2, batch_shape=[8], dtype=DTYPE)
                if kind == "covariance"
                else covariances
            )
            next_log_weights = (
                tf.nn.log_softmax(uniform_log + weight_shift)
                if kind == "weight"
                else uniform_log
            )
            return (
                next_states,
                d_states,
                next_covariances,
                d_covariances,
                next_log_weights,
                tf.zeros_like(uniform_log),
                {},
            )

        _, _, trace = _value_and_analytical_score_impl(
            model,
            *_shared_sequential_arguments(fixture)[:5],
            with_score=True,
            return_trace=True,
            observation_factor_override=None,
            post_reset_transform=hook,
            **options,
        )
        return trace

    baseline_logits = baseline_trace[1]["prior_observation_logits"].numpy()
    for kind in ("state", "weight", "covariance"):
        changed_logits = run_hook(kind)[1]["prior_observation_logits"].numpy()
        assert np.max(np.abs(changed_logits - baseline_logits)) > 1.0e-8, kind


def test_post_reset_log_weight_tangent_is_consumed_by_next_pfpf_logits():
    fixture = _sequential_fixture()
    model = _sequential_model()
    options = fixture["options"]
    uniform_log = -tf.math.log(tf.constant(8.0, DTYPE)) * tf.ones([8], DTYPE)
    injected = tf.constant([0.12, -0.09, 0.07, -0.04, 0.03, -0.02, -0.05, -0.02], DTYPE)
    np.testing.assert_allclose(tf.reduce_sum(injected).numpy(), 0.0, atol=2e-16)

    def identity_hook(time_index, states, d_states, covariances, d_covariances):
        tangent = injected if time_index == 0 else tf.zeros_like(injected)
        return (
            states,
            d_states,
            covariances,
            d_covariances,
            uniform_log,
            tangent,
            {},
        )

    _, _, baseline_trace = canonical_value_and_analytical_score(
        model,
        *_shared_sequential_arguments(fixture)[:5],
        with_score=True,
        return_trace=True,
        **options,
    )
    _, _, injected_trace = _value_and_analytical_score_impl(
        model,
        *_shared_sequential_arguments(fixture)[:5],
        with_score=True,
        return_trace=True,
        observation_factor_override=None,
        post_reset_transform=identity_hook,
        **options,
    )
    np.testing.assert_allclose(
        injected_trace[1]["d_prior_observation_logits"].numpy()
        - baseline_trace[1]["d_prior_observation_logits"].numpy(),
        injected.numpy(),
        rtol=2e-12,
        atol=2e-12,
    )
    np.testing.assert_allclose(
        injected_trace[1]["incoming_log_weights"].numpy(),
        uniform_log.numpy(),
        atol=0.0,
    )
    assert COVARIANCE_MARK_POLICY == "responsibility_conditional_mean_no_scatter_v1"


@pytest.mark.parametrize("dtype", (tf.float32, tf.float64))
def test_sequential_normalization_tolerance_keeps_valid_route(dtype):
    fixture = _sequential_fixture(dtype)
    anchor_kernel, _ = _anchor_and_replay_kernels(fixture)
    expected = max(
        1.0e-8,
        8.0 * 8.0 * np.finfo(dtype.as_numpy_dtype).eps,
    )
    np.testing.assert_equal(anchor_kernel.effective_normalization_tolerance, expected)
    anchor = anchor_kernel(
        *_shared_sequential_arguments(fixture),
        fixture["stratified_uniforms"],
        fixture["kdm_noises"],
    )
    assert bool(anchor["valid"].numpy())
