from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HARNESS_PATH = (
    ROOT
    / "docs/benchmarks/"
    "benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2v_longer_selected_map_local_screen_2026_07_09.py"
)


def _load_harness():
    spec = importlib.util.spec_from_file_location(
        "scalar_ssl_lstm_filtering_hmc_validation_phase2v_longer_selected_map_local_screen",
        HARNESS_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _phase2u_payload(*, selected_index: int = 0, initial=None):
    if initial is None:
        initial = [0.0, 0.0, 0.0, 0.0]
    return {
        "schema_version": "scalar_ssl_lstm.filtering_hmc_validation_phase2u_retuned_map_local_hmc_screen.v1",
        "decision": {
            "phase2u_retuned_map_local_hmc_screen_passed": True,
            "vetoes": [],
            "selected_candidate": {
                "candidate_index": selected_index,
                "num_leapfrog_steps": 2,
                "step_size": 0.785,
                "trajectory_length_L_times_epsilon": 1.57,
            },
            "viable_for_phase3_gpu_xla_subplan": False,
        },
        "candidate_rows": [
            {
                "candidate_index": 0,
                "initial": {"u_new": initial},
            }
        ],
    }


def test_phase2v_settings_lock_selected_kernel_and_initial_state() -> None:
    harness = _load_harness()

    settings = harness.Phase2VScreenSettings()
    payload = settings.payload()

    assert payload["num_leapfrog_steps"] == 2
    assert payload["step_size"] == 0.785
    assert payload["trajectory_length_L_times_epsilon"] == 1.57
    assert payload["num_results"] == 128
    assert payload["num_burnin_steps"] == 8
    assert payload["seed"] == (20260709, 6401)
    assert payload["initial_state_u_new"] == (0.0, 0.0, 0.0, 0.0)


def test_phase2u_handoff_requires_selected_candidate_and_zero_initial_state() -> None:
    harness = _load_harness()

    precondition = harness.validate_phase2u_handoff(_phase2u_payload())

    assert precondition["passed"] is True


def test_phase2u_handoff_vetoes_selected_candidate_mismatch() -> None:
    harness = _load_harness()

    precondition = harness.validate_phase2u_handoff(
        _phase2u_payload(selected_index=1)
    )

    assert precondition["passed"] is False
    assert "phase2u_selected_candidate_index_mismatch" in precondition["vetoes"]


def test_phase2u_handoff_vetoes_nonzero_initial_state() -> None:
    harness = _load_harness()

    precondition = harness.validate_phase2u_handoff(
        _phase2u_payload(initial=[1.0, 0.0, 0.0, 0.0])
    )

    assert precondition["passed"] is False
    assert "phase2u_selected_initial_state_not_zero" in precondition["vetoes"]


def test_phase2v_gate_requires_acceptance_envelope_and_zero_initial_state() -> None:
    harness = _load_harness()
    settings = harness.Phase2VScreenSettings()
    row = {
        "status": "passed_hard_vetoes",
        "hard_vetoes": (),
        "initial": {"u_new": [0.0, 0.0, 0.0, 0.0]},
        "acceptance_rate": 0.5,
    }

    gate = harness.evaluate_phase2v_gate(row, settings)

    assert gate["passed"] is True


def test_phase2v_gate_vetoes_acceptance_outside_envelope() -> None:
    harness = _load_harness()
    settings = harness.Phase2VScreenSettings()
    row = {
        "status": "passed_hard_vetoes",
        "hard_vetoes": (),
        "initial": {"u_new": [0.0, 0.0, 0.0, 0.0]},
        "acceptance_rate": 1.0,
    }

    gate = harness.evaluate_phase2v_gate(row, settings)

    assert gate["passed"] is False
    assert "selected_kernel_acceptance_outside_envelope" in gate["vetoes"]
