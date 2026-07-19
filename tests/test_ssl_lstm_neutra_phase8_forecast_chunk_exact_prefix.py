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
    / "docs/benchmarks/validate_ssl_lstm_neutra_phase8_forecast_chunk_exact_prefix_2026_07_17.py"
)


@pytest.fixture(scope="module")
def harness() -> ModuleType:
    spec = importlib.util.spec_from_file_location("phase8_chunk_exact_prefix", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_exact_prefix_validator_binds_passed_canary_and_failed_attempt(
    harness: ModuleType,
) -> None:
    bindings = harness._validate_bindings()
    assert bindings["canary"]["contract"]["draw_chunk_size"] == 16
    assert bindings["timeout"]["exit_code"] == 124
    assert bindings["timeout"]["receipt_written"] is False


def test_exact_prefix_validator_forbids_summary_and_confirmation(
    harness: ModuleType,
) -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "summarize_forecast_paths" not in source
    assert "mean_log_variance_influence" not in source
    assert "pooled_pairwise_distance_scale" not in source
    assert '"confirmation_suffix_selected": False' in source
    assert '"confirmation_forecast_opened": False' in source
    assert '"predictive_summary_computed": False' in source
    assert '"g_h_predictive_difference_computed": False' in source
    assert '"target_pilot_retried": False' in source


def test_exact_prefix_validator_reuses_frozen_pilot_domains(
    harness: ModuleType,
) -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "pilot.PILOT_SEED" in source
    assert "pilot.ARM_IDS[chart]" in source
    assert "pilot.FORECAST_DRAW_CHUNK_SIZE" in source
    assert "pilot.read_frozen_pilot_prefix" in source
