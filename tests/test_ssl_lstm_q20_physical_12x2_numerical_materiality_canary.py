"""Contracts for the bounded 12x2 numerical-materiality canary."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / (
    "docs/benchmarks/"
    "run_ssl_lstm_q20_physical_12x2_numerical_materiality_canary_2026_08_11.py"
)


def _load_runner():
    name = "test_ssl_lstm_q20_physical_12x2_numerical_materiality_runner"
    spec = importlib.util.spec_from_file_location(name, RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_materiality_canary_freezes_fair_topologies_and_budget() -> None:
    runner = _load_runner()
    assert runner.ROWS == 24
    assert runner.ONE_ROW_WORKERS == 24
    assert runner.TWO_ROW_WORKERS == 12
    assert runner.MASTER_SEED == (20260810, 7301)
    assert runner.MATERIAL_TRANSITIONS == 1300
    assert runner.MATERIAL_MARGIN == 1.5
    assert runner.MATERIAL_CAP_SECONDS == 20000.0
    one = runner._pool_config(
        workers=24, rows_per_worker=1, cpu_ids=runner.ONE_ROW_CPU_IDS
    )
    two = runner._pool_config(
        workers=12, rows_per_worker=2, cpu_ids=runner.TWO_ROW_CPU_IDS
    )
    assert one.batch_sizes == (1,) and one.batch_per_worker == 1
    assert two.batch_sizes == (2,) and two.batch_per_worker == 2
    assert one.factory_config == two.factory_config


def test_materiality_canary_tests_decisions_not_old_absolute_cache_thresholds() -> None:
    runner = _load_runner()
    source = RUNNER.read_text(encoding="utf-8")
    assert runner.FAILED_CANARY_SHA256 == (
        "08e9d29fee2af56aeadc3622f01a6f97487384c4446e01f16fc00dedb2ecb3ac"
    )
    assert runner.PAIR_DIAGNOSIS_SHA256 == (
        "1a29bd118fb75481aa86dde0dd6a3353d4f7b729b6e9c6cf0bf55ac2e5774363"
    )
    assert runner.R6_FAILURE_SHA256 == (
        "408645656995f123f334a4c92e1c8eb779cd9dd2633540a89e3029b0cd93caa9"
    )
    assert "hmc_accept_decisions_identical" in source
    assert "swap_accept_decisions_identical" in source
    assert "tf.boolean_mask" in source
    assert "100.0 * math.sqrt(sys.float_info.epsilon)" in source
    assert "1.0e-9" not in source
    assert "1.0e-8" not in source
    assert "import numpy" not in source
    assert math.isfinite(runner.CAP_SECONDS)
