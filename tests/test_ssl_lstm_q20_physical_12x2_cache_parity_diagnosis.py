"""Contracts for the bounded 12x2 cache-parity diagnosis."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / (
    "docs/benchmarks/"
    "diagnose_ssl_lstm_q20_physical_12x2_cache_parity_2026_08_10.py"
)


def _load_runner():
    name = "test_ssl_lstm_q20_physical_12x2_cache_parity_diagnosis_runner"
    spec = importlib.util.spec_from_file_location(name, RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_diagnosis_is_bound_and_cannot_claim_science() -> None:
    runner = _load_runner()
    source = RUNNER.read_text(encoding="utf-8")
    assert runner.FAILED_CANARY_SHA256 == (
        "08e9d29fee2af56aeadc3622f01a6f97487384c4446e01f16fc00dedb2ecb3ac"
    )
    assert "tf.roll" in source
    assert "tf.argsort" in source
    assert "original_vs_cached" in source
    assert "changed_pair_grouping" in source
    assert "no sampler, topology, travel, convergence, or posterior claim" in source
    assert "import numpy" not in source
