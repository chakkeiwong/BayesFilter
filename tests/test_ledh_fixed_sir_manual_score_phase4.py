from __future__ import annotations

import argparse
import ast
import inspect
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.highdim.ledh_forward_contract import (
    FIXED_SIR_AUSTRIA_ROW_ID,
    LEDH_TARGET_SCALAR_OBSERVED_DATA_LOG_LIKELIHOOD,
    MAIN_OBSERVED_DATA_ROW_SCOPE,
    validate_ledh_forward_contract_manifest,
)
from docs.benchmarks import benchmark_ledh_same_target_fixed_sir_score as fixed_sir
from docs.benchmarks import benchmark_p8p_parameterized_sir_gradient as p8p
from scripts import audit_ledh_no_autodiff as audit


def _args(*, dtype: str = "float64", transport_ad_mode: str = "full") -> argparse.Namespace:
    return argparse.Namespace(
        batch_seeds=[81120],
        time_steps=1,
        num_particles=2,
        theta_values=[0.02, -0.01, 0.01],
        transport_policy="active-all",
        sinkhorn_iterations=1,
        sinkhorn_epsilon=1.0,
        annealed_scaling=0.9,
        annealed_convergence_threshold=1.0e-3,
        transport_plan_mode="streaming",
        transport_gradient_mode=p8p.core_tf.MANUAL_STREAMING_FINITE_TRANSPORT_GRADIENT_MODE,
        transport_ad_mode=transport_ad_mode,
        row_chunk_size=2,
        col_chunk_size=2,
        particle_chunk_size=2,
        dtype=dtype,
        tf32_mode="disabled" if dtype == "float64" else "enabled",
        fd_step=1.0e-4 if dtype == "float64" else 1.0e-3,
        score_fd_atol=2.0e-4 if dtype == "float64" else 1.0e-1,
        score_fd_rtol=2.0e-3 if dtype == "float64" else 5.0e-2,
        historical_raw_diagnostic=True,
    )


def test_phase4_fixed_sir_manual_score_matches_same_scalar_directional_fd() -> None:
    args = _args()
    fixed_sir._configure_precision(args)

    fd = fixed_sir._fixed_sir_same_scalar_fd_diagnostic(args, args.theta_values)
    base = fd["base"]
    contract = base["forward_contract"]

    assert fd["status"] == "pass"
    assert fd["row_id"] == FIXED_SIR_AUSTRIA_ROW_ID
    assert fd["target_scope"] == MAIN_OBSERVED_DATA_ROW_SCOPE
    assert base["score_route"] == fixed_sir.FIXED_SIR_MEMORY_STYLE_SCORE_ROUTE_ID
    assert base["historical_manual_score_route"] == fixed_sir.FIXED_SIR_MANUAL_SCORE_ROUTE_ID
    assert base["value_score_route_status"] == "same_route_value_score"
    assert base["value_score_same_transport_algorithm"] is True
    assert base["old_parameterized_route_status"] == "historical_diagnostic_only"
    assert contract["row_id"] == FIXED_SIR_AUSTRIA_ROW_ID
    assert contract["row_scope"] == MAIN_OBSERVED_DATA_ROW_SCOPE
    assert contract["target_scalar"] == LEDH_TARGET_SCALAR_OBSERVED_DATA_LOG_LIKELIHOOD
    assert contract["theta_contract"]["theta_coordinate_system"] == "sir_log_scale_theta"
    assert validate_ledh_forward_contract_manifest(contract) == contract

    component_names = base["manual_score_component_names"]
    assert component_names == list(fixed_sir.MANUAL_SCORE_COMPONENT_NAMES)
    for required in (
        "observation_density_covariance",
        "ledh_flow_observation_covariance",
        "transition_mean_from_transition_density",
        "transition_mean_from_ledh_flow_prior",
        "transition_mean_from_pre_flow_clamp",
    ):
        assert required in component_names
    assert float(base["manual_score_component_reconstruction_max_abs_delta"].numpy()) <= 1.0e-7


def test_phase4_fixed_sir_manual_score_runs_under_runtime_autodiff_sentinel() -> None:
    args = _args(dtype="float32")
    fixed_sir._configure_precision(args)

    with audit.AutodiffRuntimeSentinel(p8p.tf, route_id="phase4_fixed_sir_score"):
        diagnostic = fixed_sir._fixed_sir_manual_score_diagnostic(args, args.theta_values)

    assert diagnostic["no_autodiff_score_route"] is True
    assert bool(tf.reduce_all(tf.math.is_finite(diagnostic["log_likelihood"])).numpy())
    assert bool(tf.reduce_all(tf.math.is_finite(diagnostic["gradient_tensor"])).numpy())


def test_phase4_fixed_sir_admitted_score_rejects_stopped_scale_transport() -> None:
    args = _args(transport_ad_mode="stabilized")
    fixed_sir._configure_precision(args)

    with pytest.raises(ValueError, match="transport_ad_mode='full'"):
        fixed_sir._fixed_sir_manual_score_diagnostic(args, args.theta_values)


def test_phase4_fixed_sir_fixed_adapter_matches_existing_manual_vjp_numeric_route() -> None:
    args = _args()
    fixed_sir._configure_precision(args)
    tensors, _semantics = p8p._build_base_tensors(args)

    fixed = fixed_sir._fixed_sir_manual_score_diagnostic(args, args.theta_values)
    existing = p8p._manual_value_and_score_from_components(
        tensors,
        args,
        p8p._theta_components(args.theta_values),
        return_score_decomposition=True,
    )

    np.testing.assert_allclose(
        fixed["log_likelihood"].numpy(),
        existing["log_likelihood"].numpy(),
        atol=1.0e-10,
        rtol=1.0e-10,
    )
    np.testing.assert_allclose(
        fixed["gradient_tensor"].numpy(),
        existing["gradient_tensor"].numpy(),
        atol=1.0e-9,
        rtol=1.0e-9,
    )


def test_phase4_fixed_sir_adapter_source_has_no_tape_or_forward_accumulator() -> None:
    helper_names = {
        "_fixed_sir_manual_score_diagnostic",
        "_fixed_sir_same_scalar_fd_diagnostic",
        "_require_fixed_sir_score_args",
    }
    forbidden_attrs = {"GradientTape", "ForwardAccumulator"}
    forbidden_calls = {"gradient", "jacobian", "batch_jacobian", "watch"}
    for name in helper_names:
        source = inspect.getsource(getattr(fixed_sir, name))
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                assert node.attr not in forbidden_attrs | forbidden_calls
            if isinstance(node, ast.Name):
                assert node.id not in forbidden_attrs
