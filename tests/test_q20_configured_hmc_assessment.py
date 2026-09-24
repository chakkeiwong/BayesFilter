"""Orchestration checks for the bounded frozen-map assessment."""
import importlib.util
from pathlib import Path


def module():
    path = Path(__file__).resolve().parents[1]/"docs/benchmarks/run_q20_configured_hmc_2026_09_24.py"
    spec = importlib.util.spec_from_file_location("q20_configured_hmc", path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def test_protocol_keeps_identity_mass_and_separate_hmc_budget():
    config = module().protocol()
    assert config["role"] == "development"
    assert config["jit_compile"] is True
    assert config["tuning"]["l_grid"] == [3, 5, 9]
    assert config["tuning"]["repair_reserve_units"] == 20
    assert config["posterior"]["warmup_min"] >= 2000
    assert config["posterior"]["retained_rhat"] <= 1.01


def test_nomination_prefers_replicated_family_without_ranking_metrics():
    loaded = module()
    comparisons = []
    for family in ("iaf16", "naf16"):
        for seed in range(3):
            comparisons.append({"family": family, "root_seed": seed,
                "fixed_checkpoint_training_improvement": False,
                "checkpoint": f"{family}-{seed}-checkpoint",
                "finalized": f"{family}-{seed}-finalized"})
    report = {"families": {"iaf16": {"replicated_local_improvement": False},
        "naf16": {"replicated_local_improvement": True}}, "comparisons": comparisons}
    ordered = loaded.nominate(report)
    assert [row["family"] for row in ordered[:3]] == ["naf16"]*3
    assert {row["root_seed"] for row in ordered[:3]} == {0, 1, 2}
