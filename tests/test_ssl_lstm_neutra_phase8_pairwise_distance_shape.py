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
    / "docs/benchmarks/validate_ssl_lstm_neutra_phase8_pairwise_distance_shape_2026_07_17.py"
)


@pytest.fixture(scope="module")
def harness() -> ModuleType:
    spec = importlib.util.spec_from_file_location("phase8_distance_shape", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_distance_canary_uses_exact_failed_shape(harness: ModuleType) -> None:
    assert harness.SHAPE == (8, 64, 2, 10)
    assert harness.math.prod(harness.SHAPE[:-1]) == 1024


def test_distance_canary_forbids_target_and_forecast_inputs(harness: ModuleType) -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "retained-acquisition.json" not in source
    assert "forecast_ssl_lstm_paths" not in source
    assert '"retained_samples_read": False' in source
    assert '"forecast_artifacts_read": False' in source
    assert '"g_h_difference_computed": False' in source


def test_distance_canary_binds_repair_02_failure(harness: ModuleType) -> None:
    record = harness.json.loads(harness._absolute(harness.FAILURE_RECORD_PATH).read_text())
    assert record["status"] == "FAILED_IMPLEMENTATION_KERNEL_SHAPE"
    assert record["receipt_written"] is False
