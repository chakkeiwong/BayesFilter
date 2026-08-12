from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_repair_plan_has_cpu_first_and_no_training_boundary() -> None:
    text = (
        ROOT
        / "docs/plans/bayesfilter-neutra-four-blocked-target-repair-and-admission-plan-2026-07-31.md"
    ).read_text(encoding="utf-8")
    assert "NeuTra training, HMC, leaderboard, or default-readiness claim" in text
    assert "CPU-first" in text
    assert "source_faithful" in text
    assert "fixed_hmc_adaptation" in text


def test_ksc_repair_diagnostic_is_explicitly_not_admission() -> None:
    text = (
        ROOT / "docs/benchmarks/run_neutra_ksc_ukf_repair_diagnostic_20260731.py"
    ).read_text(encoding="utf-8")
    assert "TERMINAL_KSC_UKF_REPAIR_DIAGNOSTIC" in text
    assert '"training_launched": True' not in text
    assert "component-enumerated" in text
