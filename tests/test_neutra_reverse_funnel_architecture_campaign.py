"""Static checks for the reverse-funnel architecture campaign."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "docs/benchmarks/run_neutra_reverse_funnel_architecture_campaign_2026_08_15.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("reverse_funnel_architecture_campaign", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_campaign_has_predeclared_architecture_and_tuning_ladders() -> None:
    runner = _load_runner()
    assert set(runner.ARCHITECTURES) == {
        "one_stage_exact",
        "three_root_preserving",
        "three_full_reverse",
        "three_root_wide",
    }
    assert runner.ARCHITECTURES["one_stage_exact"]["stages"] == 1
    assert runner.ARCHITECTURES["three_root_wide"]["hidden_width"] == 200
    assert runner.ARCHITECTURES["three_root_preserving"]["permutation_policy"] == "root_preserving_reverse"
    assert runner.PEAK_RATES == (2.0e-4, 5.0e-4, 1.0e-3, 2.0e-3)
    assert runner.SCHEDULES == ("constant", "piecewise_60_85")


def test_campaign_calibration_and_confirmation_are_separate() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert 'run_mode="calibration"' in source
    assert 'run_mode="confirmation"' in source
    assert 'seed_index=0' in source
    assert 'for seed_index in (1, 2)' in source
    assert '"--first-stage-unbounded-scale-linear"' in source
    assert '"--resume"' in source
    assert "import numpy" not in source
