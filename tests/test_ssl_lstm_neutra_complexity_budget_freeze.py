from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = (
    ROOT
    / "docs/benchmarks/freeze_ssl_lstm_neutra_complexity_budget_2026_07_19.py"
)


def load_runner():
    name = "ssl_lstm_neutra_complexity_budget_freeze_runner"
    spec = importlib.util.spec_from_file_location(name, RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runner = load_runner()


def test_phase3_operation_and_cold_start_contract_is_exact() -> None:
    assert runner.PHASE3_TOTAL_PROSPECTIVE_STEPS == 19_800
    assert runner.PHASE3_FRESH_POOL_LAUNCHES == 3
    assert runner.PHASE3_TRAINER_CONSTRUCTIONS == 15
    assert runner.PHASE3_TRAINERS_COVERED_BY_CANARY_LAUNCHES == 6
    assert runner.PHASE3_ADDITIONAL_TRAINER_COLD_STARTS == 9


def test_phase3_startup_terms_use_second_trainer_compile_excess() -> None:
    receipt = {
        "run_manifest": {"wall_seconds": 100.0},
        "streams": [
            {"first_step_seconds": 80.0, "warm_step_max_seconds": 2.0},
            {"first_step_seconds": 7.0, "warm_step_max_seconds": 2.0},
        ],
    }
    assert runner.phase3_startup_terms(receipt) == pytest.approx((100.0, 5.0))


def test_real_receipts_freeze_complete_non_authorizing_budget() -> None:
    payload = runner.build_budget()
    assert payload["status"] == "BUDGET_FROZEN_MATERIAL_LAUNCH_UNAUTHORIZED"
    assert payload["phase3_contract"]["subtotal_hours"] == pytest.approx(
        121.28743920202483
    )
    assert payload["hmc_contract"]["subtotal_hours"] == pytest.approx(
        421.5831899725211
    )
    assert payload["forecast_contract"]["subtotal_hours"] == pytest.approx(
        1.152858414776662
    )
    assert payload["totals"]["gpu_active_hours"] == pytest.approx(
        542.8706291745459
    )
    assert payload["totals"]["sequential_wall_cap_hours"] == pytest.approx(
        544.0234875893226
    )
    assert payload["sequential_stopping"]["material_launch_authorized"] is False
    assert len(payload["phase3_contract"]["rows"]) == 5
    assert len(payload["hmc_contract"]["rows"]) == 5
    assert len(payload["forecast_contract"]["rows"]) == 5
