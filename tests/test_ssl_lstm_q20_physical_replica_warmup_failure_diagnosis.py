"""Contracts for the target-free physical replica warm-up diagnosis."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / (
    "docs/benchmarks/"
    "diagnose_ssl_lstm_q20_physical_replica_warmup_failure_2026_08_11.py"
)


def _load_runner():
    name = "test_ssl_lstm_q20_physical_replica_warmup_failure_runner"
    spec = importlib.util.spec_from_file_location(name, RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_diagnosis_is_target_free_bound_and_explanatory_only() -> None:
    runner = _load_runner()
    source = RUNNER.read_text(encoding="utf-8")
    assert runner.WINDOW_ENDPOINTS == (600, 700, 800, 900, 1000)
    assert runner.WINDOW_DRAWS == 300
    assert runner.RHAT_THRESHOLD == 1.05
    assert "target_evaluations\": 0" in source
    assert "source-center subtraction is an explanatory decomposition" in source
    assert "import numpy" not in source
    assert runner._safe(float("nan")) is None
    assert runner._safe(float("inf")) is None
    assert math.isfinite(runner._safe(1.5))


def test_classification_separates_occupancy_and_residual_disagreement() -> None:
    runner = _load_runner()

    def report(passed: bool):
        return {"passed": passed, "nonfinite_rhat_count": 0}

    assert runner._classify(report(False), report(True)).startswith("OCCUPANCY_")
    assert runner._classify(report(True), report(False)).startswith("WITHIN_REGION_")
    assert runner._classify(report(False), report(False)).startswith("MIXED_")
    assert runner._classify(report(True), report(True)).startswith("NEITHER_")
