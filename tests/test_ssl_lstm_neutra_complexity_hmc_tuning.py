from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import tensorflow as tf
ROOT = Path(__file__).resolve().parents[1]
RUNNER = (
    ROOT
    / "docs/benchmarks/run_ssl_lstm_neutra_complexity_hmc_tuning_2026_07_19.py"
)


def load_runner():
    name = "ssl_lstm_neutra_complexity_hmc_tuning_runner"
    spec = importlib.util.spec_from_file_location(name, RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runner = load_runner()


def _trace(acceptance: tuple[float, float, float, float], draws: int = 4):
    accepted = tf.stack(
        [
            tf.concat(
                (
                    tf.ones((round(draws * rate),), tf.bool),
                    tf.zeros((draws - round(draws * rate),), tf.bool),
                ),
                axis=0,
            )
            for rate in acceptance
        ],
        axis=1,
    )
    return {
        "is_accepted": accepted,
        "log_accept_ratio": tf.zeros((draws, 4), tf.float64),
        "target_log_prob": -tf.ones((draws, 4), tf.float64),
    }


def _samples(draws: int = 4):
    time = tf.reshape(tf.cast(tf.range(draws), tf.float64), (draws, 1, 1))
    chains = tf.reshape(tf.cast(tf.range(4), tf.float64), (1, 4, 1))
    return 0.2 * time + chains + tf.zeros((draws, 4, 4), tf.float64)


def _row(step: float, leapfrog: int, viable: bool, acceptance: float):
    return {
        "step_size": step,
        "num_leapfrog_steps": leapfrog,
        "diagnostics": {
            "viable": viable,
            "acceptance_rate_by_chain": [acceptance] * 4,
            "hard_vetoes": [],
            "acceptance_vetoes": [],
        },
    }


def test_contract_is_q_general_and_uses_selected_batched_xla_topology() -> None:
    args = runner.parse_args(["--mode", "contract-smoke", "--q", "20"])
    runner.validate_material_args(args)
    payload = runner.contract_payload(args)
    assert payload["status"] == "PASSED"
    assert payload["target_acceptance"] == pytest.approx(0.70)
    assert payload["required_phase3_result_count"] == 2
    assert payload["confirmation_acceptance_band"] == [0.60, 0.80]
    assert payload["selected_hmc_topology"] == (
        "single_tfp_sample_chain_batched_four_chain_xla"
    )
    assert payload["material_execution_authorized"] is False


def test_material_modes_fail_closed_without_two_phase3_receipts_and_authority() -> None:
    args = runner.parse_args(["--mode", "tune", "--q", "1"])
    with pytest.raises(runner.ComplexityHMCError, match="authorize-material-run"):
        runner.validate_material_args(args)
    args.authorize_material_run = True
    args.cap_seconds = 60.0
    args.output_root = Path("docs/plans/artifacts/test-hmc")
    with pytest.raises(runner.ComplexityHMCError, match="two Phase 3 result"):
        runner.validate_material_args(args)


def test_phase3_receipt_must_be_admitted_and_q_bound(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    receipt = tmp_path / "result.json"
    receipt.write_text(
        '{"schema":"bayesfilter.ssl_lstm.neutra_complexity_training.v1",'
        '"status":"VETOED","q":20}\n',
        encoding="utf-8",
    )
    with pytest.raises(runner.ComplexityHMCError, match="not ADMITTED"):
        runner.load_binding(20, "chart-a", Path("result.json"))


def test_two_phase4_bindings_must_be_independent() -> None:
    row = {
        "phase3_result_path": "same-result.json",
        "phase3_stream_label": "seed-a",
        "payload_path": "same-payload.json",
        "payload_sha256": "payload",
        "artifact_signature": "artifact",
        "binding_signature": "binding",
    }
    with pytest.raises(runner.ComplexityHMCError, match="independent"):
        runner.validate_distinct_bindings({"chart-a": row, "chart-b": dict(row)})


def test_diagnostics_reject_per_chain_masking_and_preserve_divergence_unknown() -> None:
    samples = _samples()
    samples = tf.concat((samples[:, :3], tf.zeros((4, 1, 4), tf.float64)), axis=1)
    diagnostics = runner.diagnose_run(
        samples=samples,
        initial_state=tf.zeros((4, 4), tf.float64),
        trace=_trace((0.75, 0.75, 0.75, 0.0)),
        acceptance_band=(0.50, 0.90),
        min_movement=0.25,
        min_rms_jump=0.05,
    )
    assert diagnostics["viable"] is False
    assert "unmoved_chain" in diagnostics["hard_vetoes"]
    assert diagnostics["native_divergence_status"] == "unavailable_not_zero"


def test_descriptive_jump_references_do_not_become_hidden_hard_vetoes() -> None:
    samples = 0.01 * _samples()
    diagnostics = runner.diagnose_run(
        samples=samples,
        initial_state=tf.zeros((4, 4), tf.float64),
        trace=_trace((0.75, 0.75, 0.75, 0.75)),
        acceptance_band=(0.50, 0.90),
        min_movement=1.1,
        min_rms_jump=10.0,
    )
    assert diagnostics["viable"] is True
    assert diagnostics["hard_vetoes"] == []
    assert diagnostics["explanatory_flags"] == [
        "per_chain_movement_below_descriptive_reference",
        "per_chain_rms_jump_below_descriptive_reference",
    ]


def test_selection_rules_are_deterministic_and_not_runtime_rankings() -> None:
    scales = [_row(0.1, 4, True, 0.60), _row(0.4, 4, True, 0.80)]
    assert runner.select_scale(scales)["step_size"] == 0.4
    trajectories = [
        _row(0.4, leapfrog, leapfrog in (4, 8, 16), 0.70)
        for leapfrog in runner.TRAJECTORY_GRID
    ]
    assert runner.select_trajectory(trajectories)["num_leapfrog_steps"] == 8


def test_scale_bracket_repair_is_single_geometric_midpoint() -> None:
    high = _row(0.1, 4, False, 0.95)
    high["diagnostics"]["acceptance_vetoes"] = [
        "per_chain_acceptance_above_band"
    ]
    low = _row(0.2, 4, False, 0.40)
    low["diagnostics"]["acceptance_vetoes"] = [
        "per_chain_acceptance_below_band"
    ]
    assert runner.scale_bracket_repair([low, high]) == pytest.approx(
        (0.1 * 0.2) ** 0.5
    )
    mixed = _row(0.1, 4, False, 0.95)
    mixed["diagnostics"]["acceptance_rate_by_chain"][-1] = 0.70
    assert runner.scale_bracket_repair([mixed, low]) is None


def test_adjacent_repair_only_handles_acceptance_only_failure() -> None:
    rows = [_row(0.4, value, True, 0.70) for value in (2, 4, 8, 16)]
    low = _row(0.4, 8, False, 0.50)
    low["diagnostics"]["acceptance_vetoes"] = ["per_chain_acceptance_below_band"]
    assert runner.adjacent_repair_candidate(low, rows)["num_leapfrog_steps"] == 4
    low["diagnostics"]["hard_vetoes"] = ["nonfinite_hmc_telemetry"]
    assert runner.adjacent_repair_candidate(low, rows) is None


def test_budget_reserve_scales_with_observed_transition_leapfrog_cost() -> None:
    budget = runner.Budget(10_000.0)
    assert budget.arm_reserve(transitions=24, leapfrog_steps=4, cold_runner=True) == (
        runner.FIRST_COMPILED_ARM_RESERVE_SECONDS
    )
    budget.observe(2.0)
    assert budget.arm_reserve(
        transitions=96, leapfrog_steps=8, cold_runner=False
    ) == pytest.approx(2.0 * 96 * 8 * runner.ARM_COST_MARGIN)


def test_runner_persists_arm_receipts_and_normalizes_by_l() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "seconds_per_transition_leapfrog" in source
    assert "run_or_reuse_arm" in source
    assert "dynamic_num_leapfrog_steps=True" in source
    assert "samples_retained_as_posterior_evidence" in source
    assert "native divergence unavailability is not zero divergences" in source
