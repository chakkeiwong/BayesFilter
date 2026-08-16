from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import tensorflow as tf


ROOT = Path(__file__).resolve().parents[2]


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runner_preserves_v6_seed_and_exact_noise_prefixes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BAYESFILTER_CPU_ONLY_SMOKE", "true")
    runner = _load(
        ROOT / "docs/benchmarks/run_classifier_score_path_count_bundle_20260815.py",
        "path_count_runner",
    )
    old = _load(
        ROOT / "docs/benchmarks/run_classifier_score_variance_bundle_20260815.py",
        "variance_runner_for_prefix",
    )
    assert runner.ROOT_SEED == old.ROOT_SEED == 95170
    key = (3, 10, 1, 2, 0)
    for kind in ("gaussian", "sir"):
        small = old.make_noise(kind, 8192, key)
        large = runner.make_noise(kind, 32768, key)
        for left, right in zip(small, large):
            tf.debugging.assert_equal(left, right[:8192])
        assert old.noise_hash(small, 8192) == runner.noise_hash(large, 8192)


def test_runner_uses_fixed_8192_simulation_blocks() -> None:
    runner_path = (
        ROOT / "docs/benchmarks/run_classifier_score_path_count_bundle_20260815.py"
    )
    source = runner_path.read_text(encoding="utf-8")
    assert "SIMULATION_BLOCK = 8192" in source
    assert "simulate_in_fixed_blocks(" in source
    assert '"not fixed-update sample-size scaling"' in source


def test_aggregator_binds_frozen_v6_source_hashes() -> None:
    path = ROOT / "docs/benchmarks/aggregate_classifier_score_path_count_20260815.py"
    source = path.read_text(encoding="utf-8")
    assert "FROZEN_BASELINE_SOURCE_HASHES" in source
    assert "current source no longer matches baseline" in source
    assert "exact_nested_prefixes" in source


def test_aggregator_refuses_missing_baseline(tmp_path: Path) -> None:
    module = _load(
        ROOT / "docs/benchmarks/aggregate_classifier_score_path_count_20260815.py",
        "path_count_aggregator",
    )
    with pytest.raises(ValueError, match="missing result"):
        module.load_baseline(tmp_path)


def test_continuation_rule_requires_every_gate() -> None:
    module = _load(
        ROOT / "docs/benchmarks/aggregate_classifier_score_path_count_20260815.py",
        "path_count_continuation",
    )
    gaussian = {
        "all_hard_valid": True,
        "summary": {
            "fixed_adjacent_scaling": [{"ratio_lower_95": 0.2}],
            "exact_mse_adjacent_scaling": [{"ratio_lower_95": 0.4}],
            "exact_fixed_mse_adjacent_scaling": [{"ratio_lower_95": 0.5}],
        },
    }
    sir = {
        "all_hard_valid": True,
        "summary": {
            "audit_adjacent_scaling": [{"ratio_upper_95": 0.8}],
            "fixed_adjacent_scaling": [{"ratio_lower_95": 0.3}],
        },
    }
    assert module._continuation_decision(gaussian, sir)["continue_to_32768"]
    sir["summary"]["audit_adjacent_scaling"][0]["ratio_upper_95"] = 1.1
    assert not module._continuation_decision(gaussian, sir)["continue_to_32768"]
