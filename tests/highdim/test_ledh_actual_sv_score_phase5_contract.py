from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest
import tensorflow as tf

from bayesfilter.highdim.ledh_forward_contract import (
    ACTUAL_SV_ROW_ID,
    validate_ledh_forward_scalar_artifact,
)
from bayesfilter.highdim.ledh_score_contract import (
    LEDH_SCORE_ADMISSION_STATUS_FULL,
    LEDH_SCORE_ADMISSION_STATUS_BLOCKED_NOT_RUN,
    LEDH_SCORE_ADMISSION_STATUS_TINY,
    LEDH_SCORE_ARTIFACT_SCHEMA_VERSION,
    LEDH_SCORE_TARGET_KIND_REALIZED_FINITE_N_ESTIMATOR,
    LEDH_SCORE_VALUE_ROUTE_STATUS_SAME,
    validate_ledh_score_artifact,
)
from docs.benchmarks import benchmark_ledh_same_target_actual_sv_score as score_module
from experiments.dpf_implementation.tf_tfp.filters import (
    experimental_batched_ledh_pfpf_ot_tf as core_tf,
)
from scripts import audit_ledh_no_autodiff as audit


ROOT = Path(__file__).resolve().parents[2]
ACTUAL_SV_VALUE_PATH = (
    ROOT / "docs/plans/ledh-phase5-actual-sv-forward-scalar-artifact-2026-07-07.json"
)
ACTUAL_SV_VALUE_REL = "docs/plans/ledh-phase5-actual-sv-forward-scalar-artifact-2026-07-07.json"
PARAMETER_NAMES = ("gamma_unconstrained", "log_beta")


def _load_value() -> dict:
    return json.loads(ACTUAL_SV_VALUE_PATH.read_text(encoding="utf-8"))


def _tiny_score_args():
    class Args:
        batch_seeds = [81120]
        time_steps = 2
        num_particles = 9
        transport_policy = "active-all"
        sinkhorn_iterations = 1
        sinkhorn_epsilon = 1.0
        annealed_scaling = 0.9
        annealed_convergence_threshold = 1.0e-3
        flow_observation_variance = 3.141592653589793**2 / 2.0
        transport_plan_mode = "streaming"
        transport_gradient_mode = core_tf.MANUAL_STREAMING_FINITE_TRANSPORT_GRADIENT_MODE
        transport_ad_mode = "full"
        row_chunk_size = 9
        col_chunk_size = 9
        particle_chunk_size = 4
        dtype = "float64"
        tf32_mode = "disabled"

    return Args()


def _production_full_diagnostic_from_tiny() -> dict:
    args = _tiny_score_args()
    diagnostic = score_module._coordinate_fd_score_diagnostic(  # noqa: SLF001
        args,
        list(score_module.TRUTH_THETA),
        fd_step=1.0e-5,
    )
    diagnostic["score_precision"] = {
        "dtype": "float32",
        "active_dtype": "float32",
        "tf_dtype": "float32",
        "tf32_mode": "enabled",
        "tf32_execution_enabled": True,
    }
    diagnostic["base"]["num_particles"] = 10000
    diagnostic["base"]["time_steps"] = 1000
    diagnostic["base"]["batch_seeds"] = [81120, 81121, 81122, 81123, 81124]
    return diagnostic


def _value_route_log_likelihood(args, theta_values) -> tf.Tensor:
    score_module._configure_precision(args)  # noqa: SLF001
    theta = score_module._as_theta(theta_values)  # noqa: SLF001
    gamma, beta, _dgamma_dtheta = score_module._gamma_beta(theta)  # noqa: SLF001
    sigma = tf.constant(1.0, dtype=score_module.DTYPE)
    stationary_variance = score_module._stationary_variance(gamma, sigma)  # noqa: SLF001
    tensors, _semantics = score_module.value_mod._build_actual_sv_tensors(args)  # noqa: SLF001
    value = score_module.value_mod._actual_sv_value_core(  # noqa: SLF001
        observations=tensors["observations"],
        initial_particles=tensors["initial_particles"],
        proposal_seed_bases=tensors["proposal_seed_bases"],
        fixed_resampling_mask=tensors["fixed_resampling_mask"],
        gamma=gamma,
        beta=beta,
        sigma=sigma,
        stationary_variance=stationary_variance,
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
    return value.log_likelihood


def test_phase5_actual_sv_score_forward_matches_value_route_with_padding() -> None:
    args = _tiny_score_args()
    score_value = score_module._manual_value_only_from_components(  # noqa: SLF001
        args,
        list(score_module.TRUTH_THETA),
    )["log_likelihood"]
    value_route = _value_route_log_likelihood(args, list(score_module.TRUTH_THETA))

    diff = tf.reduce_max(tf.abs(score_value - value_route))
    assert float(diff.numpy()) <= 1.0e-12


def test_phase5_actual_sv_compact_score_objective_matches_value_route() -> None:
    args = _tiny_score_args()
    compact = score_module._compact_value_and_score_from_components(  # noqa: SLF001
        args,
        list(score_module.TRUTH_THETA),
    )
    value_route = _value_route_log_likelihood(args, list(score_module.TRUTH_THETA))

    diff = tf.reduce_max(tf.abs(compact["log_likelihood"] - value_route))
    assert float(diff.numpy()) <= 1.0e-12


def test_phase5_actual_sv_score_forward_is_invariant_to_particle_chunk_size() -> None:
    args_chunked = _tiny_score_args()
    args_full = _tiny_score_args()
    args_full.particle_chunk_size = int(args_full.num_particles)
    chunked = score_module._manual_value_only_from_components(  # noqa: SLF001
        args_chunked,
        list(score_module.TRUTH_THETA),
    )["log_likelihood"]
    full_chunk = score_module._manual_value_only_from_components(  # noqa: SLF001
        args_full,
        list(score_module.TRUTH_THETA),
    )["log_likelihood"]

    diff = tf.reduce_max(tf.abs(chunked - full_chunk))
    assert float(diff.numpy()) <= 1.0e-12


def _directional_only_score_artifact() -> dict:
    return {
        "schema_version": LEDH_SCORE_ARTIFACT_SCHEMA_VERSION,
        "row_id": ACTUAL_SV_ROW_ID,
        "source_value_artifact": ACTUAL_SV_VALUE_REL,
        "score_target_kind": LEDH_SCORE_TARGET_KIND_REALIZED_FINITE_N_ESTIMATOR,
        "target_scalar": "observed_data_log_likelihood_estimator",
        "target_output_tensor_field": "log_likelihood",
        "target_observation_policy": "transformed_actual_sv_log_y_square",
        "theta_coordinate_system": "synthetic_unconstrained",
        "score_parameter_names": list(PARAMETER_NAMES),
        "score": [0.0] * len(PARAMETER_NAMES),
        "score_derivative_provenance": (
            "manual_total_vjp_no_autodiff_same_scalar_actual_sv_ledh_pfpf_ot"
        ),
        "value_score_route_status": LEDH_SCORE_VALUE_ROUTE_STATUS_SAME,
        "value_score_same_transport_algorithm": True,
        "no_autodiff_score_route": True,
        "uses_gradient_tape": False,
        "uses_forward_accumulator": False,
        "uses_stopped_partial_derivative": False,
        "claims_exact_native_actual_sv_likelihood": False,
        "score_correctness": {
            "kind": "same_scalar_directional_finite_difference",
            "status": "pass",
            "abs_error": 0.0,
            "rel_error": 0.0,
        },
        "score_admission_status": LEDH_SCORE_ADMISSION_STATUS_TINY,
        "memory_diagnostics": {
            "n10000_memory_pass": False,
            "peak_mib": None,
            "budget_mib": 14000.0,
        },
    }


def test_phase5_actual_sv_value_artifact_is_admitted_but_not_score_artifact() -> None:
    value = _load_value()
    normalized = validate_ledh_forward_scalar_artifact(
        value,
        expected_row_id=ACTUAL_SV_ROW_ID,
        require_admitted=True,
    )

    assert normalized["row_id"] == ACTUAL_SV_ROW_ID
    assert normalized["target_scalar"] == "observed_data_log_likelihood_estimator"
    assert normalized["target_output_tensor_field"] == "log_likelihood"
    assert normalized["target_observation_policy"] == "transformed_actual_sv_log_y_square"
    assert normalized["theta_coordinate_system"] == "synthetic_unconstrained"
    assert tuple(normalized["forward_contract"]["theta_contract"]["parameter_order"]) == PARAMETER_NAMES
    with pytest.raises(ValueError, match="score artifact schema_version"):
        validate_ledh_score_artifact(
            value,
            source_value_artifact=value,
            expected_row_id=ACTUAL_SV_ROW_ID,
            require_admitted=True,
        )


def test_phase5_actual_sv_directional_only_score_is_not_admitted() -> None:
    value = _load_value()
    artifact = _directional_only_score_artifact()

    with pytest.raises(ValueError, match="same-scalar FD or exact reference"):
        validate_ledh_score_artifact(
            artifact,
            source_value_artifact=value,
            expected_row_id=ACTUAL_SV_ROW_ID,
            require_admitted=True,
        )


def test_phase5_actual_sv_score_rejects_ksc_or_raw_target_substitution() -> None:
    value = _load_value()
    base = score_module._compact_value_and_score_from_components(  # noqa: SLF001
        _tiny_score_args(),
        list(score_module.TRUTH_THETA),
    )
    diagnostic = {
        "status": "pass",
        "base": base,
        "max_abs_error": tf.constant(0.0, dtype=score_module.DTYPE),
        "max_rel_error": tf.constant(0.0, dtype=score_module.DTYPE),
        "parameter_names": list(PARAMETER_NAMES),
    }
    artifact = score_module._score_artifact_from_diagnostic(  # noqa: SLF001
        diagnostic,
        source_value_artifact=value,
        source_value_artifact_path=ACTUAL_SV_VALUE_REL,
        require_all_parameter_correctness=False,
    )
    assert artifact["target_observation_policy"] == "transformed_actual_sv_log_y_square"
    artifact["target_observation_policy"] = "ksc_gaussian_mixture_surrogate"
    with pytest.raises(ValueError, match="target_observation_policy"):
        validate_ledh_score_artifact(
            artifact,
            source_value_artifact=value,
            expected_row_id=ACTUAL_SV_ROW_ID,
            require_admitted=False,
        )


def test_phase5_actual_sv_compact_score_runs_under_no_autodiff_sentinel() -> None:
    args = _tiny_score_args()
    with audit.AutodiffRuntimeSentinel(
        score_module.tf,
        route_id="phase5_actual_sv_compact_score",
    ):
        result = score_module._compact_value_and_score_from_components(  # noqa: SLF001
            args,
            list(score_module.TRUTH_THETA),
        )

    assert result["score_route"] == score_module.ACTUAL_SV_COMPACT_SCORE_ROUTE_ID
    assert result["no_autodiff_score_route"] is True
    assert result["gradient_tensor"].shape == (len(PARAMETER_NAMES),)
    assert bool(tf.reduce_all(tf.math.is_finite(result["gradient_tensor"])).numpy())
    assert bool(tf.reduce_all(tf.math.is_finite(result["log_likelihood"])).numpy())


def test_phase5_actual_sv_default_score_source_uses_compact_not_reverse_records() -> None:
    source = "\n".join(
        [
            inspect.getsource(score_module._compact_value_and_score_from_components),  # noqa: SLF001
            inspect.getsource(score_module._coordinate_fd_score_diagnostic),  # noqa: SLF001
            inspect.getsource(score_module._score_artifact_from_diagnostic),  # noqa: SLF001
        ]
    )

    assert "_compact_value_and_score_from_components" in source
    assert "_manual_value_and_score_across_seeds" not in source
    assert "_manual_value_and_score_from_components" not in source
    assert "records.append" not in source
    assert "reversed(records)" not in source
    assert "_transport_vjp_tf" not in source
    assert "GradientTape" not in source
    assert "ForwardAccumulator" not in source


def test_phase5_actual_sv_cli_defaults_to_production_score_precision() -> None:
    source = inspect.getsource(score_module._parse_args)  # noqa: SLF001

    assert 'parser.add_argument("--dtype", choices=("float64", "float32"), default="float32")' in source
    assert (
        'parser.add_argument("--tf32-mode", choices=("default", "enabled", "disabled"), default="enabled")'
        in source
    )


def test_phase5_actual_sv_tiny_total_score_matches_coordinate_fd() -> None:
    args = _tiny_score_args()
    diagnostic = score_module._coordinate_fd_score_diagnostic(  # noqa: SLF001
        args,
        list(score_module.TRUTH_THETA),
        fd_step=1.0e-5,
        atol=5.0e-3,
        rtol=5.0e-3,
    )

    assert float(diagnostic["max_abs_error"].numpy()) <= 5.0e-3
    assert float(diagnostic["max_rel_error"].numpy()) <= 5.0e-3


def test_phase5_actual_sv_tiny_total_score_artifact_is_not_admitted() -> None:
    args = _tiny_score_args()
    value = _load_value()
    diagnostic = score_module._coordinate_fd_score_diagnostic(  # noqa: SLF001
        args,
        list(score_module.TRUTH_THETA),
        fd_step=1.0e-5,
    )
    artifact = score_module._score_artifact_from_diagnostic(  # noqa: SLF001
        diagnostic,
        source_value_artifact=value,
        source_value_artifact_path=ACTUAL_SV_VALUE_REL,
        require_all_parameter_correctness=False,
    )

    assert artifact["score_admission_status"] == "tiny_score_diagnostic_not_admitted"
    assert artifact["score_derivative_provenance"] == score_module.ACTUAL_SV_COMPACT_SCORE_ROUTE_ID
    assert artifact["score_precision"]["dtype"] == "float64"
    assert artifact["score_parameter_names"] == list(PARAMETER_NAMES)
    assert artifact["theta_coordinate_system"] == "synthetic_unconstrained"
    with pytest.raises(ValueError, match="not admitted"):
        validate_ledh_score_artifact(
            artifact,
            source_value_artifact=value,
            expected_row_id=ACTUAL_SV_ROW_ID,
            require_admitted=True,
        )


def test_phase5_actual_sv_full_admission_requires_memory_gate() -> None:
    args = _tiny_score_args()
    diagnostic = score_module._coordinate_fd_score_diagnostic(  # noqa: SLF001
        args,
        list(score_module.TRUTH_THETA),
        fd_step=1.0e-5,
    )
    diagnostic["parameter_names"] = list(PARAMETER_NAMES)

    with pytest.raises(ValueError, match="memory pass"):
        score_module._score_artifact_from_diagnostic(  # noqa: SLF001
            diagnostic,
            source_value_artifact=_load_value(),
            source_value_artifact_path=ACTUAL_SV_VALUE_REL,
            require_all_parameter_correctness=True,
            memory_diagnostics={
                "n10000_memory_pass": False,
                "peak_mib": None,
                "budget_mib": 14000.0,
            },
        )


def test_phase5_actual_sv_full_admission_requires_production_precision() -> None:
    diagnostic = _production_full_diagnostic_from_tiny()
    diagnostic["score_precision"]["tf32_mode"] = "disabled"
    diagnostic["score_precision"]["tf32_execution_enabled"] = False

    with pytest.raises(ValueError, match="score_precision.tf32_mode"):
        score_module._score_artifact_from_diagnostic(  # noqa: SLF001
            diagnostic,
            source_value_artifact=_load_value(),
            source_value_artifact_path=ACTUAL_SV_VALUE_REL,
            require_all_parameter_correctness=True,
            memory_diagnostics={
                "n10000_memory_pass": True,
                "source": "trusted_gpu_score_memory_artifact",
                "peak_mib": 512.0,
                "budget_mib": 14000.0,
            },
        )


def test_phase5_actual_sv_full_admission_accepts_compact_production_fixture() -> None:
    artifact = score_module._score_artifact_from_diagnostic(  # noqa: SLF001
        _production_full_diagnostic_from_tiny(),
        source_value_artifact=_load_value(),
        source_value_artifact_path=ACTUAL_SV_VALUE_REL,
        require_all_parameter_correctness=True,
        memory_diagnostics={
            "n10000_memory_pass": True,
            "source": "trusted_gpu_score_memory_artifact",
            "peak_mib": 512.0,
            "budget_mib": 14000.0,
        },
    )

    normalized = validate_ledh_score_artifact(
        artifact,
        source_value_artifact=_load_value(),
        expected_row_id=ACTUAL_SV_ROW_ID,
        require_admitted=True,
    )
    assert normalized["score_admission_status"] == LEDH_SCORE_ADMISSION_STATUS_FULL
    assert normalized["score_derivative_provenance"] == score_module.ACTUAL_SV_COMPACT_SCORE_ROUTE_ID
    assert normalized["score_precision"]["tf32_mode"] == "enabled"
    assert artifact["claims_exact_native_actual_sv_likelihood"] is False


@pytest.mark.parametrize(
    ("base_key", "base_value", "match"),
    [
        ("score_route", score_module.ACTUAL_SV_MANUAL_SCORE_ROUTE_ID, "compact score route"),
        ("score_route", score_module.ACTUAL_SV_MEMORY_STYLE_SCORE_ROUTE_ID, "compact score route"),
        ("no_autodiff_score_route", False, "no_autodiff_score_route"),
        ("value_score_route_status", "different_route", "same_route_value_score"),
    ],
)
def test_phase5_actual_sv_compact_artifact_rejects_nested_historical_relabeling(
    base_key: str,
    base_value,
    match: str,
) -> None:
    diagnostic = _production_full_diagnostic_from_tiny()
    diagnostic["base"][base_key] = base_value

    with pytest.raises(ValueError, match=match):
        score_module._score_artifact_from_diagnostic(  # noqa: SLF001
            diagnostic,
            source_value_artifact=_load_value(),
            source_value_artifact_path=ACTUAL_SV_VALUE_REL,
            require_all_parameter_correctness=True,
            memory_diagnostics={
                "n10000_memory_pass": True,
                "source": "trusted_gpu_score_memory_artifact",
                "peak_mib": 512.0,
                "budget_mib": 14000.0,
            },
        )


def test_phase5_actual_sv_compact_full_admission_requires_full_shape() -> None:
    diagnostic = _production_full_diagnostic_from_tiny()
    diagnostic["base"]["time_steps"] = 2

    with pytest.raises(ValueError, match="full time_steps"):
        score_module._score_artifact_from_diagnostic(  # noqa: SLF001
            diagnostic,
            source_value_artifact=_load_value(),
            source_value_artifact_path=ACTUAL_SV_VALUE_REL,
            require_all_parameter_correctness=True,
            memory_diagnostics={
                "n10000_memory_pass": True,
                "source": "trusted_gpu_score_memory_artifact",
                "peak_mib": 512.0,
                "budget_mib": 14000.0,
            },
        )


def test_phase5_actual_sv_old_manual_total_vjp_alias_cannot_full_admit_even_with_memory_pass() -> None:
    args = _tiny_score_args()
    diagnostic = score_module._coordinate_fd_score_diagnostic(  # noqa: SLF001
        args,
        list(score_module.TRUTH_THETA),
        fd_step=1.0e-5,
    )
    diagnostic["parameter_names"] = list(PARAMETER_NAMES)
    diagnostic["base"]["score_route"] = score_module.ACTUAL_SV_MANUAL_SCORE_ROUTE_ID

    with pytest.raises(ValueError, match="compact score route"):
        score_module._score_artifact_from_diagnostic(  # noqa: SLF001
            diagnostic,
            source_value_artifact=_load_value(),
            source_value_artifact_path=ACTUAL_SV_VALUE_REL,
            require_all_parameter_correctness=True,
            memory_diagnostics={
                "n10000_memory_pass": True,
                "source": "trusted_gpu_score_memory_artifact",
                "peak_mib": 1.0,
                "budget_mib": 14000.0,
            },
        )


def test_phase5_actual_sv_value_score_only_diagnostic_is_not_admitted() -> None:
    args = _tiny_score_args()
    base = score_module._manual_value_and_score_across_seeds(  # noqa: SLF001
        args,
        list(score_module.TRUTH_THETA),
    )
    artifact = score_module._score_artifact_from_value_score(  # noqa: SLF001
        base,
        source_value_artifact=_load_value(),
        source_value_artifact_path=ACTUAL_SV_VALUE_REL,
        memory_diagnostics={
            "n10000_memory_pass": False,
            "peak_mib": 1.0,
            "budget_mib": 14000.0,
        },
    )

    assert artifact["score_admission_status"] == LEDH_SCORE_ADMISSION_STATUS_BLOCKED_NOT_RUN
    assert artifact["score_correctness"]["status"] == "not_run"
    with pytest.raises(ValueError, match="score correctness status must pass"):
        validate_ledh_score_artifact(
            artifact,
            source_value_artifact=_load_value(),
            expected_row_id=ACTUAL_SV_ROW_ID,
            require_admitted=True,
        )
