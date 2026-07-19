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
    / "docs/benchmarks/validate_ssl_lstm_neutra_phase8_forecast_chunk_repair_2026_07_17.py"
)


@pytest.fixture(scope="module")
def harness() -> ModuleType:
    spec = importlib.util.spec_from_file_location("phase8_chunk_repair", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_chunk_canary_has_small_fixed_shape_and_fresh_seed(harness: ModuleType) -> None:
    assert harness.DRAW_COUNT == 32
    assert harness.DRAW_CHUNK_SIZE == 16
    assert harness.DRAW_COUNT % harness.DRAW_CHUNK_SIZE == 0
    assert harness.SEED == (13001, 13002)


def test_chunk_canary_does_not_name_retained_archive(harness: ModuleType) -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "retained-acquisition.json" not in source
    assert "segment-000_retained_samples" not in source
    assert '"retained_samples_read": False' in source
    assert '"confirmation_forecast_opened": False' in source
    assert "draw_chunk_size=DRAW_CHUNK_SIZE" in source


def test_chunk_canary_binds_validated_terminal_receipt(harness: ModuleType) -> None:
    assert harness._sha256(harness.TERMINAL_VALIDATION_PATH) == (
        harness.TERMINAL_VALIDATION_SHA256
    )
