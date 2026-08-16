"""Focused contracts for the frozen 12x2 topology repair canary."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "docs/benchmarks/run_ssl_lstm_q20_physical_distributed_replica_12x2_canary_2026_08_10.py"


def _load_runner():
    name = "test_ssl_lstm_q20_physical_distributed_replica_12x2_runner"
    spec = importlib.util.spec_from_file_location(name, RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_topology_and_cost_screen_are_frozen() -> None:
    runner = _load_runner()
    assert runner.WORKERS == 12
    assert runner.ROWS_PER_WORKER == 2
    assert runner.ROWS == 24
    assert runner.WORKER_CPU_IDS == tuple(range(32, 44))
    assert runner.CAP_SECONDS == 300.0
    assert runner.MATERIAL_TRANSITIONS == 1300
    assert runner.MARGIN == 1.5
    assert runner.MATERIAL_CAP_SECONDS == 20000.0
    config = runner._pool_config()
    assert config.worker_count == 12
    assert config.batch_sizes == (2,)
    assert config.batch_per_worker == 2
    assert config.factory_config["jit_compile"] is True


def test_canary_binds_failed_checkpoint_and_cannot_claim_science() -> None:
    runner = _load_runner()
    source = RUNNER.read_text(encoding="utf-8")
    assert runner.CHECKPOINT_SHA256 == (
        "8276947db5785786567c5194b469c0938907820faf8d1bafd0265b1d4f87adab"
    )
    assert "material_projection_within_20000_seconds" in source
    assert "transition_seconds + cache_seconds / 5.0" in source
    assert "terminal-cache-check" in source
    assert 'with_suffix(absolute.suffix + ".tmp")' in source
    assert "TOPOLOGY_12X2_CANARY_HARNESS_FAILED" in source
    assert "no travel, convergence, posterior" in source
    assert "import numpy" not in source
