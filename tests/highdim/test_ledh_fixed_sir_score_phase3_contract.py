from __future__ import annotations

import copy
import inspect
import json
from pathlib import Path

import pytest
import tensorflow as tf

from bayesfilter.highdim.ledh_forward_contract import (
    FIXED_SIR_AUSTRIA_ROW_ID,
    PARAMETERIZED_SIR_DIAGNOSTIC_ROW_ID,
)
from bayesfilter.highdim.ledh_score_contract import (
    LEDH_SCORE_ADMISSION_STATUS_FULL,
    LEDH_SCORE_ADMISSION_STATUS_BLOCKED_NOT_RUN,
    validate_ledh_score_artifact,
)
from docs.benchmarks import benchmark_ledh_same_target_fixed_sir_score as fixed_sir
from docs.benchmarks import benchmark_p8p_parameterized_sir_gradient as p8p
from scripts import audit_ledh_no_autodiff as audit


ROOT = Path(__file__).resolve().parents[2]
FIXED_SIR_VALUE_PATH = ROOT / "docs/plans/ledh-phase3-fixed-sir-forward-scalar-artifact-2026-07-07.json"
FIXED_SIR_VALUE_REL = "docs/plans/ledh-phase3-fixed-sir-forward-scalar-artifact-2026-07-07.json"
FIXED_SIR_SCORE_MEMORY_PATH = ROOT / "docs/plans/ledh-phase5-fixed-sir-score-memory-n10000-2026-07-06.json"


def _load_value() -> dict:
    return json.loads(FIXED_SIR_VALUE_PATH.read_text(encoding="utf-8"))


def _load_score_memory() -> dict:
    return json.loads(FIXED_SIR_SCORE_MEMORY_PATH.read_text(encoding="utf-8"))


def _tiny_compact_args():
    class Args:
        batch_seeds = [81120]
        time_steps = 2
        num_particles = 8
        theta_values = [0.0, 0.0, 0.0]
        fd_step = 1.0e-3
        score_fd_atol = 1.0e-2
        score_fd_rtol = 5.0e-2
        transport_policy = "active-all"
        sinkhorn_iterations = 2
        sinkhorn_epsilon = 1.0
        annealed_scaling = 0.9
        annealed_convergence_threshold = 1.0e-3
        transport_plan_mode = "streaming"
        transport_gradient_mode = p8p.core_tf.MANUAL_STREAMING_FINITE_TRANSPORT_GRADIENT_MODE
        transport_ad_mode = "full"
        row_chunk_size = 4
        col_chunk_size = 4
        particle_chunk_size = 4
        dtype = "float64"
        tf32_mode = "disabled"
        device = "/CPU:0"
        device_scope = "cpu"
        cuda_visible_devices = None
        expect_device_kind = "cpu"
        output = "/tmp/fixed_sir_compact_tiny.json"

    return Args()


def _production_full_diagnostic_from_tiny() -> dict:
    args = _tiny_compact_args()
    diagnostic = fixed_sir._fixed_sir_compact_coordinate_fd_diagnostic(  # noqa: SLF001
        args,
        [0.0, 0.0, 0.0],
        fd_step=1.0e-3,
    )
    diagnostic = copy.deepcopy(diagnostic)
    diagnostic["score_precision"] = {
        "dtype": "float32",
        "active_dtype": "float32",
        "tf_dtype": "float32",
        "tf32_mode": "enabled",
        "tf32_execution_enabled": True,
    }
    diagnostic["base"]["num_particles"] = 10000
    diagnostic["base"]["time_steps"] = 20
    diagnostic["base"]["batch_seeds"] = [81120, 81121, 81122, 81123, 81124]
    return diagnostic


def test_phase4_fixed_sir_compact_score_runs_under_no_autodiff_sentinel() -> None:
    args = _tiny_compact_args()
    with audit.AutodiffRuntimeSentinel(
        fixed_sir.tf,
        route_id="phase4_fixed_sir_compact_score",
    ):
        result = fixed_sir._compact_value_and_score_from_components(  # noqa: SLF001
            args,
            [0.0, 0.0, 0.0],
        )

    assert result["score_route"] == fixed_sir.FIXED_SIR_COMPACT_SCORE_ROUTE_ID
    assert result["no_autodiff_score_route"] is True
    assert result["gradient_tensor"].shape == (len(fixed_sir.PARAMETER_NAMES),)
    assert bool(tf.reduce_all(tf.math.is_finite(result["gradient_tensor"])).numpy())
    assert bool(tf.reduce_all(tf.math.is_finite(result["log_likelihood"])).numpy())


def test_phase4_fixed_sir_compact_score_matches_same_scalar_fd() -> None:
    diagnostic = fixed_sir._fixed_sir_compact_coordinate_fd_diagnostic(  # noqa: SLF001
        _tiny_compact_args(),
        [0.0, 0.0, 0.0],
        fd_step=1.0e-3,
    )

    assert diagnostic["status"] == "pass"
    assert float(diagnostic["max_abs_error"].numpy()) <= 1.0e-2
    assert float(diagnostic["max_rel_error"].numpy()) <= 5.0e-2


def test_phase4_fixed_sir_compact_score_objective_matches_historical_same_target_value() -> None:
    args = _tiny_compact_args()
    compact = fixed_sir._compact_value_and_score_from_components(  # noqa: SLF001
        args,
        [0.0, 0.0, 0.0],
    )
    historical = fixed_sir._fixed_sir_manual_score_diagnostic(  # noqa: SLF001
        args,
        [0.0, 0.0, 0.0],
        return_score_decomposition=False,
    )

    diff = tf.abs(
        compact["objective"] - tf.convert_to_tensor(historical["objective"], dtype=p8p.DTYPE)
    )
    assert float(diff.numpy()) <= 1.0e-4


def test_phase4_fixed_sir_compact_default_source_has_no_reverse_records_or_autodiff() -> None:
    source = "\n".join(
        [
            inspect.getsource(fixed_sir._compact_value_and_score_from_components),  # noqa: SLF001
            inspect.getsource(fixed_sir._fixed_sir_compact_coordinate_fd_diagnostic),  # noqa: SLF001
            inspect.getsource(fixed_sir._fixed_sir_compact_score_artifact_from_diagnostic),  # noqa: SLF001
        ]
    )

    assert "records.append" not in source
    assert "reversed(records)" not in source
    assert "_manual_transport_vjp_tf" not in source
    assert "_transport_vjp_tf" not in source
    assert "GradientTape" not in source
    assert "ForwardAccumulator" not in source


def test_phase4_fixed_sir_compact_tiny_score_artifact_is_not_admitted() -> None:
    diagnostic = fixed_sir._fixed_sir_compact_coordinate_fd_diagnostic(  # noqa: SLF001
        _tiny_compact_args(),
        [0.0, 0.0, 0.0],
        fd_step=1.0e-3,
    )
    artifact = fixed_sir._fixed_sir_compact_score_artifact_from_diagnostic(  # noqa: SLF001
        diagnostic,
        source_value_artifact=_load_value(),
        source_value_artifact_path=FIXED_SIR_VALUE_REL,
        require_all_parameter_correctness=False,
        memory_diagnostics={
            "n10000_memory_pass": False,
            "peak_mib": None,
            "budget_mib": 14000.0,
        },
    )

    assert artifact["score_admission_status"] == "tiny_score_diagnostic_not_admitted"
    assert artifact["score_derivative_provenance"] == fixed_sir.FIXED_SIR_COMPACT_SCORE_ROUTE_ID
    assert artifact["score_precision"]["dtype"] == "float64"
    assert artifact["score_parameter_names"] == list(fixed_sir.PARAMETER_NAMES)
    with pytest.raises(ValueError, match="not admitted"):
        validate_ledh_score_artifact(
            artifact,
            source_value_artifact=_load_value(),
            expected_row_id=FIXED_SIR_AUSTRIA_ROW_ID,
            require_admitted=True,
        )


def test_phase4_fixed_sir_compact_full_admission_requires_n10000_memory_and_shape() -> None:
    diagnostic = fixed_sir._fixed_sir_compact_coordinate_fd_diagnostic(  # noqa: SLF001
        _tiny_compact_args(),
        [0.0, 0.0, 0.0],
        fd_step=1.0e-3,
    )

    with pytest.raises(ValueError, match="memory pass"):
        fixed_sir._fixed_sir_compact_score_artifact_from_diagnostic(  # noqa: SLF001
            diagnostic,
            source_value_artifact=_load_value(),
            source_value_artifact_path=FIXED_SIR_VALUE_REL,
            require_all_parameter_correctness=True,
            memory_diagnostics={"n10000_memory_pass": False, "peak_mib": None},
        )

    with pytest.raises(ValueError, match="N=10000 diagnostic shape"):
        fixed_sir._fixed_sir_compact_score_artifact_from_diagnostic(  # noqa: SLF001
            diagnostic,
            source_value_artifact=_load_value(),
            source_value_artifact_path=FIXED_SIR_VALUE_REL,
            require_all_parameter_correctness=True,
            memory_diagnostics={"n10000_memory_pass": True, "peak_mib": 1.0},
        )


def test_phase4_fixed_sir_compact_full_admission_requires_production_precision() -> None:
    diagnostic = _production_full_diagnostic_from_tiny()
    diagnostic["score_precision"]["tf32_mode"] = "disabled"
    diagnostic["score_precision"]["tf32_execution_enabled"] = False

    with pytest.raises(ValueError, match="score_precision.tf32_mode"):
        fixed_sir._fixed_sir_compact_score_artifact_from_diagnostic(  # noqa: SLF001
            diagnostic,
            source_value_artifact=_load_value(),
            source_value_artifact_path=FIXED_SIR_VALUE_REL,
            require_all_parameter_correctness=True,
            memory_diagnostics={
                "n10000_memory_pass": True,
                "source": "trusted_gpu_score_memory_artifact",
                "peak_mib": 512.0,
                "budget_mib": 14000.0,
            },
        )


def test_phase4_fixed_sir_compact_full_admission_accepts_compact_production_fixture() -> None:
    artifact = fixed_sir._fixed_sir_compact_score_artifact_from_diagnostic(  # noqa: SLF001
        _production_full_diagnostic_from_tiny(),
        source_value_artifact=_load_value(),
        source_value_artifact_path=FIXED_SIR_VALUE_REL,
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
        expected_row_id=FIXED_SIR_AUSTRIA_ROW_ID,
        require_admitted=True,
    )
    assert normalized["score_admission_status"] == LEDH_SCORE_ADMISSION_STATUS_FULL
    assert normalized["score_derivative_provenance"] == fixed_sir.FIXED_SIR_COMPACT_SCORE_ROUTE_ID
    assert normalized["score_precision"]["tf32_mode"] == "enabled"


@pytest.mark.parametrize(
    ("base_key", "base_value", "match"),
    [
        ("score_route", fixed_sir.FIXED_SIR_MANUAL_SCORE_ROUTE_ID, "compact score route"),
        ("score_route", fixed_sir.FIXED_SIR_MEMORY_STYLE_SCORE_ROUTE_ID, "compact score route"),
        ("no_autodiff_score_route", False, "no_autodiff_score_route"),
        ("value_score_route_status", "different_route", "same_route_value_score"),
    ],
)
def test_phase4_fixed_sir_compact_artifact_rejects_nested_historical_relabeling(
    base_key: str,
    base_value,
    match: str,
) -> None:
    diagnostic = _production_full_diagnostic_from_tiny()
    diagnostic["base"][base_key] = base_value

    with pytest.raises(ValueError, match=match):
        fixed_sir._fixed_sir_compact_score_artifact_from_diagnostic(  # noqa: SLF001
            diagnostic,
            source_value_artifact=_load_value(),
            source_value_artifact_path=FIXED_SIR_VALUE_REL,
            require_all_parameter_correctness=True,
            memory_diagnostics={
                "n10000_memory_pass": True,
                "source": "trusted_gpu_score_memory_artifact",
                "peak_mib": 512.0,
                "budget_mib": 14000.0,
            },
        )


def test_phase4_fixed_sir_value_score_only_diagnostic_is_not_admitted() -> None:
    base = fixed_sir._compact_value_and_score_from_components(  # noqa: SLF001
        _tiny_compact_args(),
        [0.0, 0.0, 0.0],
    )
    diagnostic = {
        "status": "not_run",
        "base": base,
        "parameter_names": list(fixed_sir.PARAMETER_NAMES),
        "max_abs_error": tf.constant(0.0, dtype=p8p.DTYPE),
        "max_rel_error": tf.constant(0.0, dtype=p8p.DTYPE),
    }
    artifact = fixed_sir._fixed_sir_compact_score_artifact_from_diagnostic(  # noqa: SLF001
        diagnostic,
        source_value_artifact=_load_value(),
        source_value_artifact_path=FIXED_SIR_VALUE_REL,
        memory_diagnostics={},
    )

    assert artifact["score_admission_status"] == LEDH_SCORE_ADMISSION_STATUS_BLOCKED_NOT_RUN
    with pytest.raises(ValueError, match="score correctness status must pass"):
        validate_ledh_score_artifact(
            artifact,
            source_value_artifact=_load_value(),
            expected_row_id=FIXED_SIR_AUSTRIA_ROW_ID,
            require_admitted=False,
        )


def test_phase3_fixed_sir_memory_artifact_normalizes_as_tiny_directional_only() -> None:
    artifact = fixed_sir._fixed_sir_score_artifact_from_memory_result(
        _load_score_memory(),
        source_value_artifact=_load_value(),
        source_value_artifact_path=FIXED_SIR_VALUE_REL,
    )

    assert artifact["row_id"] == FIXED_SIR_AUSTRIA_ROW_ID
    assert artifact["score_admission_status"] == "tiny_score_diagnostic_not_admitted"
    assert artifact["score_correctness"]["kind"] == "same_scalar_directional_finite_difference"
    assert artifact["score_derivative_provenance"] == fixed_sir.FIXED_SIR_MEMORY_STYLE_SCORE_ROUTE_ID
    assert artifact["historical_memory_style_score_route"] == fixed_sir.FIXED_SIR_MEMORY_STYLE_SCORE_ROUTE_ID
    with pytest.raises(ValueError, match="same-scalar FD or exact reference"):
        validate_ledh_score_artifact(
            artifact,
            source_value_artifact=_load_value(),
            expected_row_id=FIXED_SIR_AUSTRIA_ROW_ID,
            require_admitted=True,
        )


def test_phase3_fixed_sir_rejects_flag_only_all_parameter_promotion() -> None:
    with pytest.raises(ValueError, match="diagnostic only"):
        fixed_sir._fixed_sir_score_artifact_from_memory_result(
            _load_score_memory(),
            source_value_artifact=_load_value(),
            source_value_artifact_path=FIXED_SIR_VALUE_REL,
            require_all_parameter_correctness=True,
        )


def test_phase3_fixed_sir_old_manual_alias_is_historical_not_full_admission() -> None:
    payload = _load_score_memory()
    payload["all_parameter_score_correctness"] = {
        "kind": "same_scalar_finite_difference",
        "status": "pass",
        "parameter_names": list(fixed_sir.PARAMETER_NAMES),
    }
    with pytest.raises(ValueError, match="diagnostic only"):
        fixed_sir._fixed_sir_score_artifact_from_memory_result(
            payload,
            source_value_artifact=_load_value(),
            source_value_artifact_path=FIXED_SIR_VALUE_REL,
            require_all_parameter_correctness=True,
        )


def test_phase3_fixed_sir_rejects_parameterized_diagnostic_row() -> None:
    payload = _load_score_memory()
    payload["row_id"] = PARAMETERIZED_SIR_DIAGNOSTIC_ROW_ID

    with pytest.raises(ValueError, match="main fixed-SIR"):
        fixed_sir._fixed_sir_score_artifact_from_memory_result(
            payload,
            source_value_artifact=_load_value(),
            source_value_artifact_path=FIXED_SIR_VALUE_REL,
        )


def test_phase3_fixed_sir_rejects_wrong_score_route_or_parameter_count() -> None:
    wrong_route = copy.deepcopy(_load_score_memory())
    wrong_route["score_route"] = "manual_reverse_scan_no_autodiff"
    with pytest.raises(ValueError, match="manual total VJP"):
        fixed_sir._fixed_sir_score_artifact_from_memory_result(
            wrong_route,
            source_value_artifact=_load_value(),
            source_value_artifact_path=FIXED_SIR_VALUE_REL,
        )

    wrong_score = copy.deepcopy(_load_score_memory())
    wrong_score["score"] = wrong_score["score"][:2]
    with pytest.raises(ValueError, match="score length"):
        fixed_sir._fixed_sir_score_artifact_from_memory_result(
            wrong_score,
            source_value_artifact=_load_value(),
            source_value_artifact_path=FIXED_SIR_VALUE_REL,
        )
