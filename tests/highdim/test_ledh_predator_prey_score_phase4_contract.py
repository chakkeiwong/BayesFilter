from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest
import tensorflow as tf

from bayesfilter.highdim.ledh_forward_contract import (
    PREDATOR_PREY_ROW_ID,
    validate_ledh_forward_scalar_artifact,
)
from bayesfilter.highdim.ledh_score_contract import (
    LEDH_SCORE_ADMISSION_STATUS_FULL,
    LEDH_SCORE_ADMISSION_STATUS_TINY,
    LEDH_SCORE_ARTIFACT_SCHEMA_VERSION,
    LEDH_SCORE_TARGET_KIND_REALIZED_FINITE_N_ESTIMATOR,
    LEDH_SCORE_VALUE_ROUTE_STATUS_SAME,
    validate_ledh_score_artifact,
)
from docs.benchmarks import benchmark_ledh_same_target_predator_prey_score as score_module
from experiments.dpf_implementation.tf_tfp.filters import (
    experimental_batched_ledh_pfpf_ot_tf as core_tf,
)
from scripts import audit_ledh_no_autodiff as audit


ROOT = Path(__file__).resolve().parents[2]
PREDATOR_PREY_VALUE_PATH = (
    ROOT / "docs/plans/ledh-phase4-predator-prey-forward-scalar-artifact-2026-07-07.json"
)
PREDATOR_PREY_VALUE_REL = (
    "docs/plans/ledh-phase4-predator-prey-forward-scalar-artifact-2026-07-07.json"
)
PARAMETER_NAMES = ("r", "K", "a", "s", "u", "v")


def _load_value() -> dict:
    return json.loads(PREDATOR_PREY_VALUE_PATH.read_text(encoding="utf-8"))


def _tiny_score_args():
    class Args:
        batch_seeds = [81120]
        time_steps = 1
        num_particles = 2
        transport_policy = "active-all"
        sinkhorn_iterations = 1
        sinkhorn_epsilon = 1.0
        annealed_scaling = 0.9
        annealed_convergence_threshold = 1.0e-3
        transport_plan_mode = "streaming"
        transport_gradient_mode = core_tf.MANUAL_STREAMING_FINITE_TRANSPORT_GRADIENT_MODE
        transport_ad_mode = "full"
        row_chunk_size = 2
        col_chunk_size = 2
        particle_chunk_size = 2
        dtype = "float64"
        tf32_mode = "disabled"

    return Args()


def _production_full_diagnostic_from_tiny() -> dict:
    args = _tiny_score_args()
    diagnostic = score_module._coordinate_fd_score_diagnostic(  # noqa: SLF001
        args,
        list(score_module.TRUTH_THETA),
        fd_step=1.0e-4,
    )
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


def _directional_only_score_artifact() -> dict:
    return {
        "schema_version": LEDH_SCORE_ARTIFACT_SCHEMA_VERSION,
        "row_id": PREDATOR_PREY_ROW_ID,
        "source_value_artifact": PREDATOR_PREY_VALUE_REL,
        "score_target_kind": LEDH_SCORE_TARGET_KIND_REALIZED_FINITE_N_ESTIMATOR,
        "target_scalar": "observed_data_log_likelihood_estimator",
        "target_output_tensor_field": "log_likelihood",
        "target_observation_policy": "additive_gaussian_predator_prey",
        "theta_coordinate_system": "physical",
        "score_parameter_names": list(PARAMETER_NAMES),
        "score": [0.0] * len(PARAMETER_NAMES),
        "score_derivative_provenance": (
            "manual_total_vjp_no_autodiff_same_scalar_predator_prey_ledh_pfpf_ot"
        ),
        "value_score_route_status": LEDH_SCORE_VALUE_ROUTE_STATUS_SAME,
        "value_score_same_transport_algorithm": True,
        "no_autodiff_score_route": True,
        "uses_gradient_tape": False,
        "uses_forward_accumulator": False,
        "uses_stopped_partial_derivative": False,
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


def test_phase4_predator_prey_value_artifact_is_admitted_but_not_score_artifact() -> None:
    value = _load_value()
    normalized = validate_ledh_forward_scalar_artifact(
        value,
        expected_row_id=PREDATOR_PREY_ROW_ID,
        require_admitted=True,
    )

    assert normalized["row_id"] == PREDATOR_PREY_ROW_ID
    assert normalized["target_scalar"] == "observed_data_log_likelihood_estimator"
    assert normalized["target_output_tensor_field"] == "log_likelihood"
    with pytest.raises(ValueError, match="score artifact schema_version"):
        validate_ledh_score_artifact(
            value,
            source_value_artifact=value,
            expected_row_id=PREDATOR_PREY_ROW_ID,
            require_admitted=True,
        )


def test_phase4_predator_prey_directional_only_score_is_not_admitted() -> None:
    value = _load_value()
    artifact = _directional_only_score_artifact()

    with pytest.raises(ValueError, match="same-scalar FD or exact reference"):
        validate_ledh_score_artifact(
            artifact,
            source_value_artifact=value,
            expected_row_id=PREDATOR_PREY_ROW_ID,
            require_admitted=True,
        )


def _central_difference_theta(fn, theta: tf.Tensor, index: int, step: float) -> tf.Tensor:
    basis = tf.one_hot(index, tf.shape(theta)[0], dtype=score_module.DTYPE)
    plus = fn(theta + step * basis)
    minus = fn(theta - step * basis)
    return (plus - minus) / (2.0 * step)


def _central_difference_state(fn, state: tf.Tensor, row: int, col: int, step: float) -> tf.Tensor:
    basis = tf.scatter_nd(
        indices=[[row, col]],
        updates=[tf.constant(1.0, dtype=score_module.DTYPE)],
        shape=tf.shape(state),
    )
    plus = fn(state + step * basis)
    minus = fn(state - step * basis)
    return (plus - minus) / (2.0 * step)


def test_phase4_predator_prey_rhs_vjp_matches_all_parameter_and_state_fd() -> None:
    theta = tf.constant(score_module.TRUTH_THETA, dtype=score_module.DTYPE)
    state = tf.constant([[50.0, 5.0], [80.0, 3.0]], dtype=score_module.DTYPE)
    upstream = tf.constant([[0.7, -0.2], [-0.3, 0.5]], dtype=score_module.DTYPE)
    bar_state, bar_theta = score_module._predator_prey_rhs_vjp_tf(
        theta,
        state,
        upstream,
    )
    step = 1.0e-2

    def scalar_theta(candidate: tf.Tensor) -> tf.Tensor:
        return tf.reduce_sum(score_module._predator_prey_rhs_tf(candidate, state) * upstream)

    for index, name in enumerate(score_module.PARAMETER_NAMES):
        fd_value = _central_difference_theta(scalar_theta, theta, index, step)
        tf.debugging.assert_near(
            bar_theta[index],
            fd_value,
            atol=2.0e-4,
            rtol=2.0e-4,
            message=name,
        )

    def scalar_state(candidate: tf.Tensor) -> tf.Tensor:
        return tf.reduce_sum(score_module._predator_prey_rhs_tf(theta, candidate) * upstream)

    for row in range(int(state.shape[0])):
        for col in range(int(state.shape[1])):
            fd_value = _central_difference_state(scalar_state, state, row, col, step)
            tf.debugging.assert_near(
                bar_state[row, col],
                fd_value,
                atol=2.0e-4,
                rtol=2.0e-4,
                message=f"state[{row},{col}]",
            )


def test_phase4_predator_prey_transition_mean_vjp_matches_all_parameter_and_state_fd() -> None:
    theta = tf.constant(score_module.TRUTH_THETA, dtype=score_module.DTYPE)
    state = tf.constant([[50.0, 5.0], [80.0, 3.0]], dtype=score_module.DTYPE)
    upstream = tf.constant([[0.07, -0.02], [-0.03, 0.05]], dtype=score_module.DTYPE)
    mean, aux = score_module._predator_prey_transition_mean_with_aux_tf(theta, state)
    assert mean.shape == state.shape
    bar_state, bar_theta = score_module._predator_prey_transition_mean_vjp_tf(
        theta,
        aux,
        upstream,
    )
    step = 1.0e-2

    def scalar_theta(candidate: tf.Tensor) -> tf.Tensor:
        candidate_mean, _aux = score_module._predator_prey_transition_mean_with_aux_tf(
            candidate,
            state,
        )
        return tf.reduce_sum(candidate_mean * upstream)

    for index, name in enumerate(score_module.PARAMETER_NAMES):
        fd_value = _central_difference_theta(scalar_theta, theta, index, step)
        tf.debugging.assert_near(
            bar_theta[index],
            fd_value,
            atol=1.0e-4,
            rtol=1.0e-4,
            message=name,
        )

    def scalar_state(candidate: tf.Tensor) -> tf.Tensor:
        candidate_mean, _aux = score_module._predator_prey_transition_mean_with_aux_tf(
            theta,
            candidate,
        )
        return tf.reduce_sum(candidate_mean * upstream)

    for row in range(int(state.shape[0])):
        for col in range(int(state.shape[1])):
            fd_value = _central_difference_state(scalar_state, state, row, col, step)
            tf.debugging.assert_near(
                bar_state[row, col],
                fd_value,
                atol=1.0e-4,
                rtol=1.0e-4,
                message=f"state[{row},{col}]",
            )


def test_phase4_predator_prey_compact_score_runs_under_no_autodiff_sentinel() -> None:
    args = _tiny_score_args()
    with audit.AutodiffRuntimeSentinel(
        score_module.tf,
        route_id="phase4_predator_prey_compact_score",
    ):
        result = score_module._compact_value_and_score_from_components(  # noqa: SLF001
            args,
            list(score_module.TRUTH_THETA),
        )

    assert result["score_route"] == score_module.PREDATOR_PREY_COMPACT_SCORE_ROUTE_ID
    assert result["no_autodiff_score_route"] is True
    assert result["gradient_tensor"].shape == (len(score_module.PARAMETER_NAMES),)
    assert bool(tf.reduce_all(tf.math.is_finite(result["gradient_tensor"])).numpy())
    assert bool(tf.reduce_all(tf.math.is_finite(result["log_likelihood"])).numpy())


def test_phase4_predator_prey_default_score_source_uses_compact_not_reverse_records() -> None:
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


def test_phase4_predator_prey_cli_defaults_to_production_score_precision() -> None:
    source = inspect.getsource(score_module._parse_args)  # noqa: SLF001

    assert 'parser.add_argument("--dtype", choices=("float64", "float32"), default="float32")' in source
    assert (
        'parser.add_argument("--tf32-mode", choices=("default", "enabled", "disabled"), default="enabled")'
        in source
    )


def test_phase5_predator_prey_compact_score_objective_matches_value_route() -> None:
    args = _tiny_score_args()
    compact = score_module._compact_value_and_score_from_components(  # noqa: SLF001
        args,
        list(score_module.TRUTH_THETA),
    )
    value = score_module._manual_value_only_from_components(  # noqa: SLF001
        args,
        list(score_module.TRUTH_THETA),
    )

    diff = tf.abs(compact["objective"] - value["objective"])
    assert float(diff.numpy()) <= 1.0e-12


def test_phase5_predator_prey_historical_manual_route_remains_diagnostic_only() -> None:
    args = _tiny_score_args()
    result = score_module._manual_value_and_score_from_components(  # noqa: SLF001
        args,
        list(score_module.TRUTH_THETA),
    )

    assert result["score_route"] == score_module.PREDATOR_PREY_MEMORY_STYLE_SCORE_ROUTE_ID
    assert result["historical_manual_score_route"] == score_module.PREDATOR_PREY_MANUAL_SCORE_ROUTE_ID
    assert score_module.PREDATOR_PREY_MANUAL_SCORE_ROUTE_ID != score_module.PREDATOR_PREY_COMPACT_SCORE_ROUTE_ID


def test_phase4_predator_prey_tiny_total_score_matches_coordinate_fd() -> None:
    args = _tiny_score_args()
    diagnostic = score_module._coordinate_fd_score_diagnostic(  # noqa: SLF001
        args,
        list(score_module.TRUTH_THETA),
        fd_step=1.0e-4,
    )

    assert float(diagnostic["max_abs_error"].numpy()) <= 5.0e-3
    assert float(diagnostic["max_rel_error"].numpy()) <= 5.0e-3


def test_phase4_predator_prey_tiny_total_score_artifact_is_not_admitted() -> None:
    args = _tiny_score_args()
    value = _load_value()
    diagnostic = score_module._coordinate_fd_score_diagnostic(  # noqa: SLF001
        args,
        list(score_module.TRUTH_THETA),
        fd_step=1.0e-4,
    )
    artifact = score_module._score_artifact_from_diagnostic(  # noqa: SLF001
        diagnostic,
        source_value_artifact=value,
        source_value_artifact_path=PREDATOR_PREY_VALUE_REL,
        require_all_parameter_correctness=False,
    )

    assert artifact["score_admission_status"] == "tiny_score_diagnostic_not_admitted"
    assert artifact["score_derivative_provenance"] == score_module.PREDATOR_PREY_COMPACT_SCORE_ROUTE_ID
    assert artifact["score_precision"]["dtype"] == "float64"
    assert artifact["score_parameter_names"] == list(score_module.PARAMETER_NAMES)
    with pytest.raises(ValueError, match="not admitted"):
        validate_ledh_score_artifact(
            artifact,
            source_value_artifact=value,
            expected_row_id=PREDATOR_PREY_ROW_ID,
            require_admitted=True,
        )


def test_phase4_predator_prey_failed_correctness_blocks_artifact() -> None:
    args = _tiny_score_args()
    diagnostic = score_module._coordinate_fd_score_diagnostic(  # noqa: SLF001
        args,
        list(score_module.TRUTH_THETA),
        fd_step=1.0e-4,
    )
    diagnostic["status"] = "fail"
    artifact = score_module._score_artifact_from_diagnostic(  # noqa: SLF001
        diagnostic,
        source_value_artifact=_load_value(),
        source_value_artifact_path=PREDATOR_PREY_VALUE_REL,
        require_all_parameter_correctness=False,
    )

    assert artifact["score_admission_status"] == "blocked_score_not_run"
    with pytest.raises(ValueError, match="correctness status"):
        validate_ledh_score_artifact(
            artifact,
            source_value_artifact=_load_value(),
            expected_row_id=PREDATOR_PREY_ROW_ID,
            require_admitted=False,
        )


def test_phase4_predator_prey_full_admission_requires_memory_gate() -> None:
    args = _tiny_score_args()
    diagnostic = score_module._coordinate_fd_score_diagnostic(  # noqa: SLF001
        args,
        list(score_module.TRUTH_THETA),
        fd_step=1.0e-4,
    )
    diagnostic["parameter_names"] = list(score_module.PARAMETER_NAMES)

    with pytest.raises(ValueError, match="memory pass"):
        score_module._score_artifact_from_diagnostic(  # noqa: SLF001
            diagnostic,
            source_value_artifact=_load_value(),
            source_value_artifact_path=PREDATOR_PREY_VALUE_REL,
            require_all_parameter_correctness=True,
            memory_diagnostics={"n10000_memory_pass": False},
        )


def test_phase4_predator_prey_full_admission_requires_production_precision() -> None:
    diagnostic = _production_full_diagnostic_from_tiny()
    diagnostic["score_precision"]["tf32_mode"] = "disabled"
    diagnostic["score_precision"]["tf32_execution_enabled"] = False

    with pytest.raises(ValueError, match="score_precision.tf32_mode"):
        score_module._score_artifact_from_diagnostic(  # noqa: SLF001
            diagnostic,
            source_value_artifact=_load_value(),
            source_value_artifact_path=PREDATOR_PREY_VALUE_REL,
            require_all_parameter_correctness=True,
            memory_diagnostics={
                "n10000_memory_pass": True,
                "source": "trusted_gpu_score_memory_artifact",
                "peak_mib": 512.0,
                "budget_mib": 14000.0,
            },
        )


def test_phase4_predator_prey_full_admission_accepts_compact_production_fixture() -> None:
    artifact = score_module._score_artifact_from_diagnostic(  # noqa: SLF001
        _production_full_diagnostic_from_tiny(),
        source_value_artifact=_load_value(),
        source_value_artifact_path=PREDATOR_PREY_VALUE_REL,
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
        expected_row_id=PREDATOR_PREY_ROW_ID,
        require_admitted=True,
    )
    assert normalized["score_admission_status"] == LEDH_SCORE_ADMISSION_STATUS_FULL
    assert normalized["score_derivative_provenance"] == score_module.PREDATOR_PREY_COMPACT_SCORE_ROUTE_ID
    assert normalized["score_precision"]["tf32_mode"] == "enabled"


@pytest.mark.parametrize(
    ("base_key", "base_value", "match"),
    [
        ("score_route", score_module.PREDATOR_PREY_MANUAL_SCORE_ROUTE_ID, "compact score route"),
        ("score_route", score_module.PREDATOR_PREY_MEMORY_STYLE_SCORE_ROUTE_ID, "compact score route"),
        ("no_autodiff_score_route", False, "no_autodiff_score_route"),
        ("value_score_route_status", "different_route", "same_route_value_score"),
    ],
)
def test_phase4_predator_prey_compact_artifact_rejects_nested_historical_relabeling(
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
            source_value_artifact_path=PREDATOR_PREY_VALUE_REL,
            require_all_parameter_correctness=True,
            memory_diagnostics={
                "n10000_memory_pass": True,
                "source": "trusted_gpu_score_memory_artifact",
                "peak_mib": 512.0,
                "budget_mib": 14000.0,
            },
        )


def test_phase4_predator_prey_compact_full_admission_requires_full_shape() -> None:
    diagnostic = _production_full_diagnostic_from_tiny()
    diagnostic["base"]["num_particles"] = 2

    with pytest.raises(ValueError, match="N=10000 diagnostic shape"):
        score_module._score_artifact_from_diagnostic(  # noqa: SLF001
            diagnostic,
            source_value_artifact=_load_value(),
            source_value_artifact_path=PREDATOR_PREY_VALUE_REL,
            require_all_parameter_correctness=True,
            memory_diagnostics={
                "n10000_memory_pass": True,
                "source": "trusted_gpu_score_memory_artifact",
                "peak_mib": 512.0,
                "budget_mib": 14000.0,
            },
        )


def test_phase4_predator_prey_old_manual_total_vjp_alias_cannot_full_admit_even_with_memory_pass() -> None:
    args = _tiny_score_args()
    diagnostic = score_module._coordinate_fd_score_diagnostic(  # noqa: SLF001
        args,
        list(score_module.TRUTH_THETA),
        fd_step=1.0e-4,
    )
    diagnostic["parameter_names"] = list(score_module.PARAMETER_NAMES)
    diagnostic["base"]["score_route"] = score_module.PREDATOR_PREY_MANUAL_SCORE_ROUTE_ID

    with pytest.raises(ValueError, match="compact score route"):
        score_module._score_artifact_from_diagnostic(  # noqa: SLF001
            diagnostic,
            source_value_artifact=_load_value(),
            source_value_artifact_path=PREDATOR_PREY_VALUE_REL,
            require_all_parameter_correctness=True,
            memory_diagnostics={
                "n10000_memory_pass": True,
                "source": "trusted_gpu_score_memory_artifact",
                "peak_mib": 1.0,
                "budget_mib": 14000.0,
            },
        )
