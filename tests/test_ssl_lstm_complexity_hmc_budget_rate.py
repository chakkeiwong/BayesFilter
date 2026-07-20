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
    / "docs/benchmarks/benchmark_ssl_lstm_complexity_hmc_budget_rate_2026_07_19.py"
)


def load_runner():
    name = "ssl_lstm_complexity_hmc_budget_rate_runner"
    spec = importlib.util.spec_from_file_location(name, RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runner = load_runner()


def test_hmc_budget_contract_matches_implemented_worst_case() -> None:
    payload = runner.contract_payload(20)
    assert payload["num_results"] == 2
    assert payload["num_burnin_steps"] == 1
    assert payload["num_leapfrog_steps"] == 1
    assert payload["warm_repeats"] == 2
    assert payload["hmc_transition_leapfrogs_per_rung"] == 408800
    assert payload["hmc_cold_reserve_seconds_per_rung"] == 9000.0
    assert payload["hmc_margin"] == 1.5
    assert payload["material_execution_authorized"] is False


def test_hmc_budget_formula_adds_margin_and_cold_reserve() -> None:
    assert runner.hmc_budget_seconds(2.0) == pytest.approx(
        1.5 * 2.0 * 408800 + 9000.0
    )
