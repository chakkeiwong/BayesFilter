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
    / "docs/benchmarks/diagnose_ssl_lstm_neutra_phase8_terminal_projection_2026_07_17.py"
)


@pytest.fixture(scope="module")
def harness() -> ModuleType:
    spec = importlib.util.spec_from_file_location("phase8_terminal_diagnostic", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_diagnostic_binds_projection_status(harness: ModuleType) -> None:
    assert harness.EXPECTED_FAILURE_STATUS == harness.predictive.STATUS_PROJECTION == 8
    assert harness.predictive.STATUS_MATERIALLY_INDEFINITE == 4
    assert harness.predictive.STATUS_FACTOR_RECONSTRUCTION == 16


def test_summary_preserves_failure_indices_and_ratios(harness: ModuleType) -> None:
    rows = [
        {
            "index": 0,
            "status": 0,
            "symmetry_ratio": 0.1,
            "projection_ratio": 0.2,
            "factor_reconstruction_ratio": 0.3,
            "negative_eigenvalue_ratio": 0.0,
        },
        {
            "index": 1,
            "status": 8,
            "symmetry_ratio": 0.2,
            "projection_ratio": 8.5,
            "factor_reconstruction_ratio": 0.4,
            "negative_eigenvalue_ratio": 0.0,
        },
    ]
    summary = harness._summary(rows)
    assert summary["failure_count"] == 1
    assert summary["failure_indices"] == [1]
    assert summary["failure_statuses"] == [8]
    assert summary["maximum_projection_ratio"] == pytest.approx(8.5)


def test_source_forbids_forecast_and_automatic_retry() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "forecast_ssl_lstm_paths" not in source
    assert '"forecast_executed": False' in source
    assert '"automatic_pilot_retry": False' in source
    assert '"confirmation_suffix_selected": False' in source
