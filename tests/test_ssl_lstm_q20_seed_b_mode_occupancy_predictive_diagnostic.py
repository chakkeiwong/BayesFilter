from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / (
    "docs/benchmarks/"
    "run_ssl_lstm_q20_seed_b_mode_occupancy_predictive_diagnostic_2026_08_09.py"
)


def _module():
    spec = importlib.util.spec_from_file_location("mode_diagnostic", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_user_fixed_predictive_geometry_and_separate_decisions() -> None:
    module = _module()
    source = RUNNER.read_text(encoding="utf-8")
    assert module.HORIZONS == (10, 20, 30, 50, 100)
    assert module.SAMPLE_SIZE == 1000
    assert module.PERMUTATION_COUNT == 9999
    assert module.ALPHA == 0.01
    assert module.CHAIN_COUNT == 4
    assert module.RETAINED_PER_CHAIN == 1000
    assert '"joint_test_computed": False' in source
    assert '"combined_p_value_computed": False' in source
    assert '"multiplicity_adjustment_applied": False' in source
    assert '"mode_mass_estimated": False' in source
    assert '"posterior_predictive_mixture_computed": False' in source


def test_p_value_rule_is_strictly_below_one_percent() -> None:
    module = _module()
    assert module.classify_p_value(0.0099) == "DISTINGUISHED_AT_1_PERCENT"
    assert module.classify_p_value(0.01) == "NOT_DISTINGUISHED_AT_1_PERCENT"
    with pytest.raises(module.ModeDiagnosticError):
        module.classify_p_value(0.0)


def test_representative_horizon_seed_banks_are_disjoint() -> None:
    module = _module()
    seeds = [
        seed
        for representative in ("plus", "minus")
        for horizon in module.HORIZONS
        for seed in module._seeds(representative, horizon).values()
    ]
    assert len(seeds) == len(set(seeds)) == 30
    assert not set(seeds).intersection(module._canary_seeds().values())


def test_region_coverage_is_half_space_only_and_not_basin_assignment() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert '"classification": "DESCRIPTIVE_HALF_SPACE_COVERAGE_ONLY"' in source
    assert '"coverage_coordinate": "observation_weight.0.0"' in source
    assert '"coverage_boundary": 0.0' in source
    assert "nearest_map_geometry" not in source
    assert "assignment_disagreement_count" not in source
    assert "not an estimate of integrated mode mass" in source
    assert "not formal basin-membership classification" in source
