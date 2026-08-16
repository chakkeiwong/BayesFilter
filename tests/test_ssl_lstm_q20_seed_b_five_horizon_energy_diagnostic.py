from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / (
    "docs/benchmarks/"
    "run_ssl_lstm_q20_seed_b_five_horizon_energy_diagnostic_2026_08_09.py"
)


def _module():
    spec = importlib.util.spec_from_file_location("five_horizon_energy", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_user_fixed_geometry_and_no_joint_decision_are_source_bound() -> None:
    module = _module()
    source = RUNNER.read_text(encoding="utf-8")
    assert module.HORIZONS == (10, 20, 30, 50, 100)
    assert len(module.HORIZONS) == 5
    assert module.SAMPLE_SIZE == 1000
    assert module.PERMUTATION_COUNT == 9999
    assert module.ALPHA == 0.01
    assert '"joint_test_computed": False' in source
    assert '"combined_p_value_computed": False' in source
    assert '"multiplicity_adjustment_applied": False' in source
    assert "arm_specific_standardization_used" in source
    assert "standardize_forecast_paths" not in source


def test_p_value_rule_is_strictly_below_one_percent() -> None:
    module = _module()
    assert module.classify_p_value(0.0099) == "DISTINGUISHED_AT_1_PERCENT"
    assert module.classify_p_value(0.01) == "NOT_DISTINGUISHED_AT_1_PERCENT"
    assert module.classify_p_value(0.0101) == "NOT_DISTINGUISHED_AT_1_PERCENT"
    with pytest.raises(module.EnergyCampaignError):
        module.classify_p_value(0.0)


def test_horizon_seeds_are_disjoint_and_fwer_arithmetic_is_correct() -> None:
    module = _module()
    seeds = [value for horizon in module.HORIZONS for value in module._seeds(horizon).values()]
    assert len(seeds) == len(set(seeds)) == 15
    assert 1.0 - (1.0 - module.ALPHA) ** 5 == pytest.approx(0.0490099501)
