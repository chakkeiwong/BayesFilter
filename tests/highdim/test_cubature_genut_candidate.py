from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.highdim.cubature_genut_candidate import (
    CandidateRouteScope,
    cubature_design,
    gaussian_genut_design,
    genut_design,
    issue_candidate_route_identity,
    issue_repository_candidate_route_identity,
    map_standardized_design,
    replicate_positive_genut,
    validate_candidate_route_identity,
    validate_repository_candidate_route_identity,
)


def _moments(points: tf.Tensor, weights: tf.Tensor | None = None):
    if weights is None:
        weights = tf.fill([tf.shape(points)[0]], 1.0 / tf.cast(tf.shape(points)[0], tf.float32))
    mean = tf.reduce_sum(points * weights[:, None], axis=0)
    centered = points - mean[None, :]
    covariance = tf.einsum("n,ni,nj->ij", weights, centered, centered)
    return mean.numpy(), covariance.numpy()


def test_cubature_design_exact_population_moments():
    points = cubature_design(dim=3, num_particles=12)
    mean, covariance = _moments(points)
    np.testing.assert_allclose(mean, np.zeros(3), atol=1e-6)
    np.testing.assert_allclose(covariance, np.eye(3), atol=1e-6)


def test_gaussian_genut_is_cubature_after_positive_replication():
    design = gaussian_genut_design(dim=3)
    assert design.positive
    replicated = replicate_positive_genut(design, num_particles=12)
    cubature = cubature_design(dim=3, num_particles=12)
    np.testing.assert_array_equal(
        np.sort(replicated.numpy(), axis=0), np.sort(cubature.numpy(), axis=0)
    )
    mean, covariance = _moments(design.points, design.weights)
    np.testing.assert_allclose(mean, np.zeros(3), atol=1e-6)
    np.testing.assert_allclose(covariance, np.eye(3), atol=1e-6)


def test_scalar_gaussian_genut_is_weighted_three_point_not_cubature():
    design = gaussian_genut_design(dim=1)
    np.testing.assert_allclose(
        design.points.numpy().reshape(-1),
        [0.0, -np.sqrt(3.0), np.sqrt(3.0)],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        design.weights.numpy(), [2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0], atol=1e-6
    )
    assert design.central_weight == pytest.approx(2.0 / 3.0)
    assert not np.array_equal(
        replicate_positive_genut(design, num_particles=12).numpy(),
        cubature_design(dim=1, num_particles=12).numpy(),
    )


def test_genut_rejects_infeasible_axis_and_nonrepresentable_weights():
    with pytest.raises(ValueError, match="feasible"):
        genut_design(standardized_skewness=(1.0,), standardized_kurtosis=(0.5,))
    design = genut_design(standardized_skewness=(0.5,), standardized_kurtosis=(3.0,))
    with pytest.raises(ValueError, match="representable"):
        replicate_positive_genut(design, num_particles=7)


def test_genut_negative_central_weight_is_not_an_ot_design():
    design = genut_design(
        standardized_skewness=(0.0, 0.0), standardized_kurtosis=(0.5, 0.5)
    )
    assert not design.positive
    with pytest.raises(ValueError, match="signed"):
        replicate_positive_genut(design, num_particles=10)


def test_mapping_preserves_weighted_moments():
    design = gaussian_genut_design(dim=2)
    points, weights = map_standardized_design(
        design,
        mean=tf.constant([1.0, -2.0]),
        square_root=tf.constant([[2.0, 0.0], [0.5, 1.5]]),
    )
    mean, covariance = _moments(points, weights)
    np.testing.assert_allclose(mean, [1.0, -2.0], atol=1e-6)
    root = np.array([[2.0, 0.0], [0.5, 1.5]])
    np.testing.assert_allclose(covariance, root @ root.T, atol=1e-6)


def test_candidate_identity_is_deterministic_and_tamper_evident():
    scope = CandidateRouteScope(
        model_id="toy_nonlinear",
        target_id="toy_target_v1",
        horizon=10,
        particle_count=12,
        state_dimension=2,
        parameter_count=3,
        dtype="float32",
        tf32_enabled=True,
        jit_compile=True,
        design_family="cubature",
        control_family_id="candidate_controls_v1",
    )
    first = issue_candidate_route_identity(
        scope,
        prepared_data_id="prepared_v1",
        residual_design_id="cubature_replicated_v1",
        controls={"epsilon": "2", "sinkhorn_steps": "8"},
        callable_dependency_ids=("bayesfilter.highdim.cubature_genut_filter:finite_value_score",),
        source_dependency_closure_id="cubature_genut_source_closure_v1",
    )
    second = issue_candidate_route_identity(
        scope,
        prepared_data_id="prepared_v1",
        residual_design_id="cubature_replicated_v1",
        controls={"sinkhorn_steps": "8", "epsilon": "2"},
        callable_dependency_ids=("bayesfilter.highdim.cubature_genut_filter:finite_value_score",),
        source_dependency_closure_id="cubature_genut_source_closure_v1",
    )
    assert first.identity_sha256 == second.identity_sha256
    validate_candidate_route_identity(first)
    forged = dataclasses.replace(first, identity_sha256="0" * 64)
    with pytest.raises(ValueError, match="digest"):
        validate_candidate_route_identity(forged)


def test_candidate_identity_rejects_caller_constructed_seal():
    scope = CandidateRouteScope(
        model_id="toy_nonlinear",
        target_id="toy_target_v1",
        horizon=2,
        particle_count=12,
        state_dimension=1,
        parameter_count=1,
        dtype="float32",
        tf32_enabled=True,
        jit_compile=True,
        design_family="cubature",
        control_family_id="candidate_controls_v1",
    )
    identity = issue_candidate_route_identity(
        scope,
        prepared_data_id="prepared_v1",
        residual_design_id="cubature_replicated_v1",
        controls={"epsilon": "2"},
        callable_dependency_ids=("bayesfilter.highdim.cubature_genut_filter:finite_value_score",),
        source_dependency_closure_id="cubature_genut_source_closure_v1",
    )
    forged = dataclasses.replace(identity, _factory_seal=object())
    with pytest.raises(ValueError, match="issuance seal"):
        validate_candidate_route_identity(forged)


def test_repository_candidate_identity_binds_registered_source_closure():
    scope = CandidateRouteScope(
        model_id="exact_sv",
        target_id="exact_sv_target_v1",
        horizon=2,
        particle_count=12,
        state_dimension=1,
        parameter_count=2,
        dtype="float32",
        tf32_enabled=True,
        jit_compile=True,
        design_family="cubature",
        control_family_id="candidate_controls_v1",
    )
    identity = issue_repository_candidate_route_identity(
        scope,
        prepared_data_id="prepared_v1",
        residual_design_id="cubature_replicated_v1",
        controls={"epsilon": "2"},
        adapter_id="exact_transformed_sv_v1",
    )
    validate_repository_candidate_route_identity(identity)
    forged = dataclasses.replace(
        identity,
        source_dependency_closure_id="0" * 64,
    )
    with pytest.raises(ValueError, match="digest"):
        validate_repository_candidate_route_identity(forged)


def test_repository_candidate_identity_supports_predator_prey_adapter():
    scope = CandidateRouteScope(
        model_id="predator_prey_additive_gaussian",
        target_id="zhao_cui_predator_prey_T20",
        horizon=20,
        particle_count=1002,
        state_dimension=2,
        parameter_count=6,
        dtype="float32",
        tf32_enabled=True,
        jit_compile=True,
        design_family="genut",
        control_family_id="predator_prey_genut_controls_v1",
    )
    identity = issue_repository_candidate_route_identity(
        scope,
        prepared_data_id="sha256:canonical-seed-81104",
        residual_design_id="gaussian_genut_dim2_equal_mass_n1002_v1",
        controls={"epsilon": "2", "sinkhorn_steps": "8", "ridge": "1e-5"},
        adapter_id="predator_prey_additive_gaussian_v1",
    )
    validate_repository_candidate_route_identity(identity)
    assert (
        dict(identity.controls)["adapter_id"]
        == "predator_prey_additive_gaussian_v1"
    )


def test_repository_candidate_identity_binds_chapter18b_structural_primitives():
    scope = CandidateRouteScope(
        model_id="chapter18b_quadratic_structural",
        target_id="STR-UKF-five-probit-T100-structural-innovation-v1",
        horizon=100,
        particle_count=1002,
        state_dimension=2,
        parameter_count=5,
        dtype="float32",
        tf32_enabled=True,
        jit_compile=True,
        design_family="genut",
        control_family_id="str_ukf_genut_controls_v1",
    )
    identity = issue_repository_candidate_route_identity(
        scope,
        prepared_data_id="sha256:frozen-structural-observations",
        residual_design_id="gaussian_genut_dim2_equal_mass_n1002_v1",
        controls={"epsilon": "4", "sinkhorn_steps": "4", "ridge": "1e-6"},
        adapter_id="chapter18b_structural_shared_primitives_v1",
    )
    validate_repository_candidate_route_identity(identity)
    assert dict(identity.controls)["adapter_id"] == (
        "chapter18b_structural_shared_primitives_v1"
    )
    dependencies = set(identity.callable_dependency_ids)
    assert any("structural_transition_residual_dtype" in item for item in dependencies)
    assert any("structural_observation_log_density_tangent_dtype" in item for item in dependencies)
