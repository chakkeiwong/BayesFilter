"""Contracts for exact transition-500 material continuation."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / (
    "docs/benchmarks/"
    "resume_ssl_lstm_q20_physical_distributed_replica_material_2026_08_11.py"
)


def _load_runner():
    name = "test_ssl_lstm_q20_physical_distributed_replica_resume_runner"
    spec = importlib.util.spec_from_file_location(name, RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_resume_preserves_exact_kernel_seed_and_absolute_budget() -> None:
    runner = _load_runner()
    assert runner.RESUME_TRANSITION == 500
    assert runner.MASTER_SEED == (20260811, 8101)
    assert runner.WARMUP_MILESTONES == (600, 700, 800, 900, 1000)
    assert runner.WARMUP_WINDOW == 300
    assert runner.RETAINED_MILESTONES == (1000, 1250, 1500)
    assert (runner.CAMPAIGN_END - runner.CAMPAIGN_START).total_seconds() == 28800
    assert runner.FINALIZATION_RESERVE_SECONDS == 300.0


def test_resume_binds_all_prior_attempts_and_excludes_tuning_draws() -> None:
    runner = _load_runner()
    source = RUNNER.read_text(encoding="utf-8")
    assert runner.R8_SHA256 == "9e6771652842b6f96e304509a042949dc2513923ef8279021a7783b4fd82b9d9"
    assert runner.R10_2P0_SHA256 == "f039cfc4b285a10385a1d7c73dd89cbe8f9f4a740a8502ce7de7a3de9cac4235"
    assert "transition_index=transition_index" in source
    assert "master_seed=MASTER_SEED" in source
    assert "resume-cache-validation" in source
    assert "for name, receipt in receipts.items()" in source
    assert "r8 history does not end at resume transition" in source
    assert "finite_log_accept_or_invalid" in source
    assert "terminal_cache_target_status_invalid" in source
    assert "explanatory_only_status_invalidity_is_a_hard_veto" in source
    assert "tuning canary draws excluded from this continuation" in source
    assert "state_rows[warmup_cutoff:]" not in source
    assert "import numpy" not in source
