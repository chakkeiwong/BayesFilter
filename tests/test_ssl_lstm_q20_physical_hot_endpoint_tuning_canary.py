"""Contracts for the bounded hot-endpoint tuning canaries."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / (
    "docs/benchmarks/"
    "run_ssl_lstm_q20_physical_hot_endpoint_tuning_canary_2026_08_11.py"
)


def _load_runner():
    name = "test_ssl_lstm_q20_physical_hot_endpoint_tuning_canary_runner"
    spec = importlib.util.spec_from_file_location(name, RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_candidate_grid_and_selection_are_frozen() -> None:
    runner = _load_runner()
    assert runner.CANDIDATES["ratio-0p40"]["ratio"] == 0.40
    assert runner.CANDIDATES["ratio-0p35"]["ratio"] == 0.35
    assert runner.CANDIDATES["hot-step-1p5"]["hot_step_multiplier"] == 1.5
    assert runner.CANDIDATES["hot-step-2p0"]["hot_step_multiplier"] == 2.0
    assert runner.CANDIDATES["hot-step-1p5"]["ratio"] == 0.50
    assert runner.CANDIDATES["hot-step-2p0"]["ratio"] == 0.50
    assert runner.CANDIDATES["ratio-0p40"]["cpu_start"] == 32
    assert runner.CANDIDATES["ratio-0p35"]["cpu_start"] == 64
    assert runner.CANDIDATES["dense-mass-step-0p35"]["base_step"] == 0.35
    assert runner.CANDIDATES["dense-mass-step-0p70"]["base_step"] == 0.70
    assert runner.CANDIDATES["dense-mass-step-0p35"]["dense_mass"] is True
    assert runner.CANDIDATES["dense-mass-step-0p35-l4"]["leapfrog"] == 4
    assert runner.DENSE_MASS_0P35_SHA256 == (
        "3b785f78ca2e18e44162756cde7a69088c8bb3723f2549dee1106f4567dc63f0"
    )
    assert runner.TRANSITIONS == 100
    assert runner.LEAPFROG == 8
    assert runner.WORKERS == 24
    assert runner.CAP_SECONDS == 2400.0


def test_canary_selects_hot_forgetting_communication_and_acceptance_only() -> None:
    runner = _load_runner()
    source = RUNNER.read_text(encoding="utf-8")
    assert runner.FAILED_MATERIAL_SHA256 == (
        "9e6771652842b6f96e304509a042949dc2513923ef8279021a7783b4fd82b9d9"
    )
    assert runner.RATIO_0P40_SHA256 == (
        "b58381e92dc609ff2b33dade8901c62558df5cbc23d82c6fb25a5eb6a261e570"
    )
    assert runner.RATIO_0P35_SHA256 == (
        "22349ca8141f2b89d921adbc22eafb5774901eadf50f07d0a05d6fc4618394b2"
    )
    assert "hot_forgetting_all_chains" in source
    assert "every_adjacent_pair_communicated" in source
    assert "acceptance_means_in_band" in source
    assert "invalid_paths_self_rejected" in source
    assert "log_acceptance_finite_or_invalid_negative_infinity" in source
    assert '"distributed_helper": _sha(DISTRIBUTED_HELPER)' in source
    assert "R-hat, occupancy, and runtime are not selection criteria" in source
    assert "import numpy" not in source
    assert "mean_two_checked_mapped_local_precisions" in source
    assert "mass_matrix=mass_matrix" in source
