from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

ROOT = Path(__file__).resolve().parents[1]
RUNNER = (
    ROOT
    / "docs/benchmarks/benchmark_ssl_lstm_complexity_forecast_pool_2026_07_19.py"
)


def load_runner():
    name = "ssl_lstm_complexity_forecast_pool_timing_runner"
    spec = importlib.util.spec_from_file_location(name, RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runner = load_runner()


def test_contract_matches_phase6_production_block_and_worker_topology() -> None:
    payload = runner.contract_payload(20)
    assert payload["block_draws"] == 256
    assert payload["warm_repeats"] == 2
    assert payload["worker_count"] == 16
    assert payload["forecast_replication_count"] == 2
    assert payload["forecast_horizon"] == 10
    assert payload["material_total_blocks"] == 388
    assert payload["material_fresh_pool_starts"] == 3
    assert payload["material_warm_blocks"] == 385
    assert payload["material_execution_authorized"] is False


def test_projection_and_seed_contract_are_exact() -> None:
    assert runner.projection_seconds(10.0, 2.0) == pytest.approx(
        1.5 * (3 * 10.0 + 385 * 2.0)
    )
    for q in runner.Q_VALUES:
        seeds = runner.canary_seeds(q)
        assert seeds.shape == (256, 2)
        assert {tuple(row) for row in seeds.tolist()}.isdisjoint(
            runner.material_seed_set(q)
        )
