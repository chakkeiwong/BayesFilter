from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import ModuleType

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "docs/benchmarks/validate_ssl_lstm_neutra_phase8_terminal_orientation_repair_2026_07_17.py"
)


@pytest.fixture(scope="module")
def harness() -> ModuleType:
    spec = importlib.util.spec_from_file_location("phase8_terminal_validation", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_validation_binds_exact_decomposition_diagnostic(harness: ModuleType) -> None:
    receipt = harness.validate_decomposition_receipt()
    assert receipt["prior_diagnostic_binding"]["failure_indices"] == [
        33,
        68,
        144,
        189,
        200,
        201,
    ]
    assert receipt["decision"] == "PHASE8_NO_PRINCIPAL_DECOMPOSITION_REPAIR_IDENTIFIED"
    assert receipt["candidate_summary"]["svd_left_passed_unchanged_gates"] is False
    assert receipt["candidate_summary"]["svd_right_passed_unchanged_gates"] is False


def test_source_is_terminal_only_and_keeps_gates(harness: ModuleType) -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "forecast_ssl_lstm_paths" not in source
    assert '"forecast_executed": False' in source
    assert '"target_pilot_retried": False' in source
    assert '"projection_multiplier": 8.0' in source
    assert '"factor_reconstruction_multiplier": 16.0' in source
    assert "for chart in (\"fresh-g\", \"fresh-h\")" in source
    assert "ssl_lstm_terminal_covariance_audit_compiled_program" in source
    assert "extract_ssl_lstm_terminal_states" in source


def test_write_json_refuses_overwrite(harness: ModuleType, tmp_path: Path) -> None:
    output = tmp_path / "receipt.json"
    harness._write_json(output, {"status": "first"})
    with pytest.raises(harness.TerminalRepairValidationError, match="overwrite"):
        harness._write_json(output, {"status": "second"})
