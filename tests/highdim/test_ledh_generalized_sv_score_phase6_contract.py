from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest
import tensorflow as tf

from bayesfilter.highdim.ledh_forward_contract import (
    GENERALIZED_SV_ROW_ID,
    validate_ledh_forward_scalar_artifact,
)
from bayesfilter.highdim.ledh_score_contract import validate_ledh_score_artifact
from docs.benchmarks import benchmark_ledh_same_target_generalized_sv_score as score_module
from experiments.dpf_implementation.tf_tfp.filters import (
    experimental_batched_ledh_pfpf_ot_tf as core_tf,
)
from scripts import audit_ledh_no_autodiff as audit


ROOT = Path(__file__).resolve().parents[2]
GENERALIZED_SV_VALUE_PATH = (
    ROOT / "docs/plans/ledh-phase6-generalized-sv-forward-scalar-artifact-2026-07-07.json"
)
GENERALIZED_SV_VALUE_REL = (
    "docs/plans/ledh-phase6-generalized-sv-forward-scalar-artifact-2026-07-07.json"
)
PARAMETER_NAMES = ("gamma_unconstrained", "log_tau", "mu")


def _load_value() -> dict:
    return json.loads(GENERALIZED_SV_VALUE_PATH.read_text(encoding="utf-8"))


def _tiny_score_args():
    class Args:
        batch_seeds = [81120]
        time_steps = 2
        num_particles = 8
        transport_policy = "active-all"
        sinkhorn_iterations = 1
        sinkhorn_epsilon = 1.0
        annealed_scaling = 0.9
        annealed_convergence_threshold = 1.0e-3
        flow_observation_variance = 2.0
        transport_plan_mode = "streaming"
        transport_gradient_mode = core_tf.MANUAL_STREAMING_FINITE_TRANSPORT_GRADIENT_MODE
        transport_ad_mode = "full"
        row_chunk_size = 4
        col_chunk_size = 4
        particle_chunk_size = 4
        dtype = "float64"
        tf32_mode = "disabled"
        fd_step = 1.0e-5
        score_fd_atol = 1.0e-3
        score_fd_rtol = 2.0e-3

    return Args()


def test_phase6_generalized_sv_value_artifact_is_admitted_but_not_score_artifact() -> None:
    value = _load_value()
    normalized = validate_ledh_forward_scalar_artifact(
        value,
        expected_row_id=GENERALIZED_SV_ROW_ID,
        require_admitted=True,
    )

    assert normalized["row_id"] == GENERALIZED_SV_ROW_ID
    assert normalized["target_scalar"] == "observed_data_log_likelihood_estimator"
    assert normalized["target_output_tensor_field"] == "log_likelihood"
    assert normalized["target_observation_policy"] == "source_route_prior_mean_generalized_sv"
    assert normalized["theta_coordinate_system"] == "source_route_active_transformed_prior_mean"
    assert tuple(normalized["forward_contract"]["theta_contract"]["parameter_order"]) == PARAMETER_NAMES
    with pytest.raises(ValueError, match="score artifact schema_version"):
        validate_ledh_score_artifact(
            value,
            source_value_artifact=value,
            expected_row_id=GENERALIZED_SV_ROW_ID,
            require_admitted=True,
        )


def test_phase6_generalized_sv_tiny_total_score_runs_under_no_autodiff_sentinel() -> None:
    args = _tiny_score_args()
    with audit.AutodiffRuntimeSentinel(
        score_module.tf,
        route_id="phase6_generalized_sv_tiny_total_score",
    ):
        result = score_module._manual_value_and_score_across_seeds(  # noqa: SLF001
            args,
            list(score_module.TRUTH_THETA),
        )

    assert result["score_route"] == score_module.GENERALIZED_SV_COMPACT_SCORE_ROUTE_ID
    assert result["no_autodiff_score_route"] is True
    assert result["gradient_tensor"].shape == (len(PARAMETER_NAMES),)
    assert bool(tf.reduce_all(tf.math.is_finite(result["gradient_tensor"])).numpy())
    assert bool(tf.reduce_all(tf.math.is_finite(result["log_likelihood"])).numpy())


def test_phase6_generalized_sv_compact_default_source_has_no_reverse_records_or_autodiff() -> None:
    source = "\n".join(
        [
            inspect.getsource(score_module._compact_value_and_score_from_components),  # noqa: SLF001
            inspect.getsource(score_module._manual_value_and_score_across_seeds),  # noqa: SLF001
            inspect.getsource(score_module._coordinate_fd_score_diagnostic),  # noqa: SLF001
            inspect.getsource(score_module._score_artifact_from_diagnostic),  # noqa: SLF001
        ]
    )

    assert "records.append" not in source
    assert "reversed(records)" not in source
    assert "_transport_vjp_tf" not in source
    assert "GradientTape" not in source
    assert "ForwardAccumulator" not in source
    assert "stop_gradient" not in source


def test_phase6_generalized_sv_compact_score_objective_matches_value_route() -> None:
    args = _tiny_score_args()
    compact = score_module._compact_value_and_score_from_components(  # noqa: SLF001
        args,
        list(score_module.TRUTH_THETA),
    )
    replay_value = score_module._manual_value_only_from_components(  # noqa: SLF001
        args,
        list(score_module.TRUTH_THETA),
    )
    tensors, _semantics = score_module.value_mod._build_generalized_sv_tensors(args)  # noqa: SLF001
    core_value = score_module.value_mod._generalized_sv_value_core(  # noqa: SLF001
        observations=tensors["observations"],
        raw_observations=tensors["raw_observations"],
        initial_particles=tensors["initial_particles"],
        proposal_seed_bases=tensors["proposal_seed_bases"],
        fixed_resampling_mask=tensors["fixed_resampling_mask"],
        gamma=tensors["gamma"],
        tau=tensors["tau"],
        mu=tensors["mu"],
        flow_observation_covariance=tensors["flow_observation_covariance"],
        sinkhorn_epsilon=args.sinkhorn_epsilon,
        annealed_scaling=args.annealed_scaling,
        annealed_convergence_threshold=args.annealed_convergence_threshold,
        sinkhorn_iterations=args.sinkhorn_iterations,
        row_chunk_size=args.row_chunk_size,
        col_chunk_size=args.col_chunk_size,
        particle_chunk_size=args.particle_chunk_size,
        return_history=False,
    )

    replay_diff = tf.abs(compact["objective"] - replay_value["objective"])
    core_diff = tf.abs(compact["objective"] - tf.reduce_mean(core_value.log_likelihood))
    assert float(replay_diff.numpy()) <= 1.0e-12
    assert float(core_diff.numpy()) <= 1.0e-12


def test_phase6_generalized_sv_tiny_total_score_matches_coordinate_fd() -> None:
    args = _tiny_score_args()
    diagnostic = score_module._coordinate_fd_score_diagnostic(  # noqa: SLF001
        args,
        list(score_module.TRUTH_THETA),
        fd_step=1.0e-5,
        atol=1.0e-3,
        rtol=2.0e-3,
    )

    assert diagnostic["status"] == "pass"
    assert float(diagnostic["max_abs_error"].numpy()) <= 1.0e-3
    assert float(diagnostic["max_rel_error"].numpy()) <= 2.0e-3


def test_phase6_generalized_sv_tiny_total_score_artifact_is_not_admitted() -> None:
    args = _tiny_score_args()
    value = _load_value()
    diagnostic = score_module._coordinate_fd_score_diagnostic(  # noqa: SLF001
        args,
        list(score_module.TRUTH_THETA),
        fd_step=1.0e-5,
        atol=1.0e-3,
        rtol=2.0e-3,
    )
    artifact = score_module._score_artifact_from_diagnostic(  # noqa: SLF001
        diagnostic,
        source_value_artifact=value,
        source_value_artifact_path=GENERALIZED_SV_VALUE_REL,
        require_all_parameter_correctness=False,
        memory_diagnostics={
            "n10000_memory_pass": False,
            "peak_mib": None,
            "budget_mib": 14000.0,
        },
    )

    assert artifact["score_admission_status"] == "tiny_score_diagnostic_not_admitted"
    assert artifact["score_derivative_provenance"] == score_module.GENERALIZED_SV_COMPACT_SCORE_ROUTE_ID
    assert artifact["score_parameter_names"] == list(PARAMETER_NAMES)
    assert artifact["target_observation_policy"] == "source_route_prior_mean_generalized_sv"
    with pytest.raises(ValueError, match="not admitted"):
        validate_ledh_score_artifact(
            artifact,
            source_value_artifact=value,
            expected_row_id=GENERALIZED_SV_ROW_ID,
            require_admitted=True,
        )


def test_phase6_generalized_sv_failed_correctness_blocks_artifact() -> None:
    args = _tiny_score_args()
    diagnostic = score_module._coordinate_fd_score_diagnostic(  # noqa: SLF001
        args,
        list(score_module.TRUTH_THETA),
        fd_step=1.0e-5,
        atol=1.0e-3,
        rtol=2.0e-3,
    )
    diagnostic["status"] = "fail"
    artifact = score_module._score_artifact_from_diagnostic(  # noqa: SLF001
        diagnostic,
        source_value_artifact=_load_value(),
        source_value_artifact_path=GENERALIZED_SV_VALUE_REL,
        require_all_parameter_correctness=False,
    )

    assert artifact["score_admission_status"] == "blocked_score_not_run"
    with pytest.raises(ValueError, match="correctness status"):
        validate_ledh_score_artifact(
            artifact,
            source_value_artifact=_load_value(),
            expected_row_id=GENERALIZED_SV_ROW_ID,
            require_admitted=False,
        )


def test_phase6_generalized_sv_full_admission_requires_memory_and_full_shape() -> None:
    args = _tiny_score_args()
    diagnostic = score_module._coordinate_fd_score_diagnostic(  # noqa: SLF001
        args,
        list(score_module.TRUTH_THETA),
        fd_step=1.0e-5,
        atol=1.0e-3,
        rtol=2.0e-3,
    )

    with pytest.raises(ValueError, match="memory pass"):
        score_module._score_artifact_from_diagnostic(  # noqa: SLF001
            diagnostic,
            source_value_artifact=_load_value(),
            source_value_artifact_path=GENERALIZED_SV_VALUE_REL,
            require_all_parameter_correctness=True,
            memory_diagnostics={"n10000_memory_pass": False, "peak_mib": None},
        )


def test_phase6_generalized_sv_rejects_target_substitution_over_artifact_contract() -> None:
    args = _tiny_score_args()
    diagnostic = score_module._coordinate_fd_score_diagnostic(  # noqa: SLF001
        args,
        list(score_module.TRUTH_THETA),
        fd_step=1.0e-5,
        atol=1.0e-3,
        rtol=2.0e-3,
    )
    artifact = score_module._score_artifact_from_diagnostic(  # noqa: SLF001
        diagnostic,
        source_value_artifact=_load_value(),
        source_value_artifact_path=GENERALIZED_SV_VALUE_REL,
        require_all_parameter_correctness=False,
    )

    artifact["target_observation_policy"] = "ksc_log_chi_square_gaussian_mixture_surrogate"
    with pytest.raises(ValueError, match="target_observation_policy"):
        validate_ledh_score_artifact(
            artifact,
            source_value_artifact=_load_value(),
            expected_row_id=GENERALIZED_SV_ROW_ID,
            require_admitted=False,
        )

    artifact["target_observation_policy"] = "source_route_prior_mean_generalized_sv"
    artifact["score_parameter_names"] = ["log_tau", "gamma_unconstrained", "mu"]
    with pytest.raises(ValueError, match="parameter"):
        validate_ledh_score_artifact(
            artifact,
            source_value_artifact=_load_value(),
            expected_row_id=GENERALIZED_SV_ROW_ID,
            require_admitted=False,
        )
