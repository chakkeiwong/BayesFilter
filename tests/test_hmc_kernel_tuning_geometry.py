from __future__ import annotations

import inspect

import numpy as np
import pytest

from bayesfilter.inference.hmc_kernel_tuning import (
    GEOMETRY_INITIALIZATION_NONCLAIMS,
    HMCGeometryInitializationConfig,
    HMCGeometryInitializationResult,
    initialize_hmc_kernel_geometry,
)


class Adapter:
    parameter_dim = 2

    def parameter_names(self):
        return ("a", "b")


def test_identity_geometry_formula_returns_finite_kernel() -> None:
    result = initialize_hmc_kernel_geometry(
        adapter=Adapter(),
        initial_position=np.array([0.0, 0.0]),
        config=HMCGeometryInitializationConfig(
            geometry_scaling_c=0.5,
            stability_guard=0.8,
            covariance_jitter=0.0,
            seed=(11, 22),
        ),
    )

    assert isinstance(result, HMCGeometryInitializationResult)
    assert result.hint_report["selected_hint"] == "identity"
    assert result.target_dimension == 2
    assert result.initial_step_size == pytest.approx(0.5 * 2.0 ** (-0.25))
    assert result.target_trajectory_length == pytest.approx(np.pi / 2.0)
    assert result.initial_num_leapfrog_steps == int(
        np.ceil(result.target_trajectory_length / result.initial_step_size)
    )
    assert result.mass_artifact.position_role == "initial_position"
    assert "no posterior convergence claim" in result.nonclaims
    assert result.payload()["reports_hmc_runtime_readiness"] is False
    assert result.payload()["reports_tuning_success"] is False


def test_fixed_identity_geometry_ignores_supplied_hints_and_records_policy() -> None:
    result = initialize_hmc_kernel_geometry(
        adapter=Adapter(),
        initial_position=np.zeros(2),
        config=HMCGeometryInitializationConfig(
            covariance_jitter=0.0,
            mass_policy="fixed_identity",
        ),
        initial_covariance=np.diag([3.0, 7.0]),
        parameter_scales=np.array([4.0, 5.0]),
    )
    np.testing.assert_allclose(result.mass_artifact.covariance, np.eye(2))
    assert result.config.mass_policy == "fixed_identity"
    assert result.payload()["config"]["mass_policy"] == "fixed_identity"


def test_negative_hessian_takes_precedence_over_covariance_and_scales() -> None:
    result = initialize_hmc_kernel_geometry(
        adapter=Adapter(),
        initial_position=np.zeros(2),
        config=HMCGeometryInitializationConfig(covariance_jitter=0.0),
        negative_hessian=np.diag([4.0, 9.0]),
        initial_covariance=np.diag([100.0, 100.0]),
        parameter_scales=np.array([10.0, 10.0]),
    )

    assert result.hint_report["selected_hint"] == "negative_hessian"
    assert result.hint_report["supplied_hints"] == {
        "negative_hessian": True,
        "initial_covariance": True,
        "parameter_scales": True,
    }
    np.testing.assert_allclose(result.mass_artifact.covariance, np.diag([0.25, 1.0 / 9.0]))


def test_covariance_takes_precedence_over_scales_when_hessian_absent() -> None:
    result = initialize_hmc_kernel_geometry(
        adapter=Adapter(),
        initial_position=np.zeros(2),
        config=HMCGeometryInitializationConfig(covariance_jitter=0.0),
        initial_covariance=np.diag([2.0, 3.0]),
        parameter_scales=np.array([10.0, 10.0]),
    )

    assert result.hint_report["selected_hint"] == "initial_covariance"
    np.testing.assert_allclose(result.mass_artifact.covariance, np.diag([2.0, 3.0]))


def test_scales_build_diagonal_covariance() -> None:
    result = initialize_hmc_kernel_geometry(
        adapter=Adapter(),
        initial_position=np.zeros(2),
        config=HMCGeometryInitializationConfig(covariance_jitter=0.0),
        parameter_scales=np.array([2.0, 3.0]),
    )

    assert result.hint_report["selected_hint"] == "parameter_scales"
    np.testing.assert_allclose(result.mass_artifact.covariance, np.diag([4.0, 9.0]))


def test_nonfinite_higher_precedence_hint_fails_closed_by_default() -> None:
    with pytest.raises(ValueError, match="negative_hessian must be finite"):
        initialize_hmc_kernel_geometry(
            adapter=Adapter(),
            initial_position=np.zeros(2),
            negative_hessian=np.array([[1.0, np.nan], [np.nan, 1.0]]),
            initial_covariance=np.eye(2),
        )


def test_allowed_fallback_records_failed_hint() -> None:
    result = initialize_hmc_kernel_geometry(
        adapter=Adapter(),
        initial_position=np.zeros(2),
        config=HMCGeometryInitializationConfig(
            covariance_jitter=0.0,
            allow_geometry_fallback=True,
        ),
        negative_hessian=np.array([[1.0, np.nan], [np.nan, 1.0]]),
        initial_covariance=np.diag([2.0, 3.0]),
    )

    assert result.hint_report["selected_hint"] == "initial_covariance"
    assert result.hint_report["fallback_used"] is True
    assert result.hint_report["fallback_failures"][0]["kind"] == "negative_hessian"


def test_regularized_indefinite_hessian_records_metadata() -> None:
    result = initialize_hmc_kernel_geometry(
        adapter=Adapter(),
        initial_position=np.zeros(2),
        config=HMCGeometryInitializationConfig(
            covariance_jitter=0.0,
            eigenvalue_floor=0.25,
        ),
        negative_hessian=np.diag([2.0, -0.5]),
    )

    report = result.hint_report["regularization_report"]
    assert report["raw_nonpositive_eigenvalue_count"] == 1
    assert report["clipped_eigenvalue_count"] == 1
    assert report["effective_eigenvalue_floor"] == pytest.approx(0.25)
    np.testing.assert_allclose(result.mass_artifact.covariance, np.diag([0.5, 4.0]))


def test_shape_mismatch_hard_vetoes() -> None:
    with pytest.raises(ValueError, match="initial_covariance shape"):
        initialize_hmc_kernel_geometry(
            adapter=Adapter(),
            initial_position=np.zeros(2),
            initial_covariance=np.eye(3),
        )


def test_geometry_config_does_not_expose_hmc_tuning_mechanics() -> None:
    parameters = set(inspect.signature(HMCGeometryInitializationConfig).parameters)
    forbidden = {
        "step_size",
        "num_leapfrog_steps",
        "min_leapfrog",
        "step_size_candidates",
        "num_leapfrog_step_candidates",
        "trajectory_grid",
        "mass_window_schedule",
        "budget_schedule",
    }

    assert parameters.isdisjoint(forbidden)
    assert "max_leapfrog_steps" in parameters


def test_seed_and_payload_are_deterministic() -> None:
    kwargs = {
        "adapter": Adapter(),
        "initial_position": np.zeros(2),
        "config": HMCGeometryInitializationConfig(seed=(123, 456)),
        "parameter_scales": np.array([1.5, 2.0]),
    }
    result_a = initialize_hmc_kernel_geometry(**kwargs)
    result_b = initialize_hmc_kernel_geometry(**kwargs)

    assert result_a.seed_report == result_b.seed_report
    assert result_a.artifact_hash == result_b.artifact_hash
    assert result_a.seed_report["root_seed"] == (123, 456)
    assert result_a.seed_report["geometry_seed"] != result_a.seed_report[
        "fresh_verification_seed_reserved"
    ]


def test_nonclaims_constant_contains_no_tuning_claim() -> None:
    assert GEOMETRY_INITIALIZATION_NONCLAIMS == (
        "geometry initialization only",
        "no HMC runtime claim",
        "no tuning success claim",
        "no posterior convergence claim",
        "no sampler superiority claim",
        "no default-readiness claim",
    )
