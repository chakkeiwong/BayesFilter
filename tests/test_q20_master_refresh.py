"""Small CPU fixtures for checkpoint import, remaining costs and master refresh."""
import copy
import json
from pathlib import Path

import pytest

from bayesfilter.inference.q20_campaign_runtime import source_snapshot, atomic_json
from bayesfilter.inference.q20_campaign_costs import training_reservation
from bayesfilter.inference.q20_production_config import digest
from bayesfilter.inference.q20_training_resume import (
    check_refresh_sources, import_training_checkpoint, read_training_checkpoint,
)
from tests.test_q20_campaign_costs import measured_row
from tests.test_q20_production_repair import tiny_protocol, four_dimensional_bridge


REPO = Path(__file__).resolve().parents[1]


def small_repo(root):
    for name, text in {
        "bayesfilter/inference/q20_master_program.py": "VERSION = 1\n",
        "bayesfilter/inference/neutra_training_protocol.py": "def assess_training_rung():\n    return 1\n\ndef numerical_step():\n    return 3\n",
        "bayesfilter/inference/target.py": "TARGET = 7\n",
        "docs/benchmarks/run_ssl_lstm_q20_production_2026_09_15.py": "# cli\n",
        "docs/benchmarks/diagnose_q20_hmc_status_reuse_2026_09_16.py": "# diagnostic\n",
    }.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return root


def test_source_import_allows_only_reviewed_coordinator_or_assessment_changes(tmp_path):
    before, after = small_repo(tmp_path / "old"), small_repo(tmp_path / "new")
    old = source_snapshot(before)
    path = after / "bayesfilter/inference/neutra_training_protocol.py"
    path.write_text(path.read_text().replace("return 1", "return 2"))
    current, changes = check_refresh_sources(old, before, after)
    assert changes == ["bayesfilter/inference/neutra_training_protocol.py"]
    path.write_text(path.read_text().replace("return 3", "return 4"))
    with pytest.raises(ValueError, match="numerical source change"):
        check_refresh_sources(old, before, after)


def test_import_preserves_optimizer_rng_cache_and_ordinary_restart(tmp_path):
    from bayesfilter.inference.q20_production_training import run_training_cohort
    from bayesfilter.inference.q20_training_resume import cached_loss_rows
    config, bridge = tiny_protocol(), four_dimensional_bridge()
    kwargs = dict(memory_policy={"mode": "tiny_cpu_reference"}, max_seconds=120.)
    partial = run_training_cohort(config, bridge, tmp_path / "partial", calibration_only=True, **kwargs)
    original = read_training_checkpoint(partial["checkpoint"], config)
    imported = import_training_checkpoint(partial["checkpoint"], config, tmp_path / "import",
        previous_root=REPO, current_root=REPO)
    state = read_training_checkpoint(imported["checkpoint"], config, sources=source_snapshot(REPO))
    for name, item in state["cohort"].items():
        assert item["session"] == original["cohort"][name]["session"]
        assert not item["exports"]
        assert not item["assessments"]
    assert cached_loss_rows(imported["checkpoint"], config) == cached_loss_rows(partial["checkpoint"], config)
    prices = [{**measured_row(4, b), "batch_size": 8} for b in (.5, 1.)]
    config["training"]["pricing_batches"] = [8]
    # Pricing metadata is not silently changed in a saved numerical protocol.
    with pytest.raises(ValueError, match="configuration"):
        training_reservation(config, prices, checkpoint=imported["checkpoint"])
    config = tiny_protocol()
    before = training_reservation(config, prices)
    after = training_reservation(config, prices, checkpoint=imported["checkpoint"])
    assert after["scenarios"]["calibration"]["optimizer_updates"] == 0
    assert after["scenarios"]["calibration"]["target_validation_rows"] == 0
    assert after["scenarios"]["floor_first_bank"]["credited_updates"] == 2
    assert after["scenarios"]["floor_first_bank"]["optimizer_updates"] == before["scenarios"]["floor_first_bank"]["optimizer_updates"] - 2
    imported_run = run_training_cohort(tiny_protocol(), bridge, tmp_path / "resumed", resume=imported["checkpoint"], **kwargs)
    direct_run = run_training_cohort(tiny_protocol(), bridge, tmp_path / "direct", resume=partial["checkpoint"], **kwargs)
    a = read_training_checkpoint(imported_run["checkpoint"], tiny_protocol())
    b = read_training_checkpoint(direct_run["checkpoint"], tiny_protocol())
    assert a["source_import"] == state["source_import"]
    for name in a["cohort"]:
        for field in ("optimizer", "rng_index", "iteration", "level_updates"):
            assert a["cohort"][name]["session"][field] == b["cohort"][name]["session"][field]
        assert a["cohort"][name]["session"]["map"]["variables"] == b["cohort"][name]["session"]["map"]["variables"]
        assert a["cohort"][name]["exports"]
        assert any(row["decision"]["validation_policy"] == "bounded_learning_screen_for_hmc_trial_v1"
                   for row in a["cohort"][name]["assessments"])


def test_refresh_master_runs_real_workers_and_reuses_finished_stages(tmp_path):
    from bayesfilter.inference.q20_production_training import run_training_cohort
    from bayesfilter.inference.q20_master_program import execute_master
    from tests.test_q20_master_integration import protocol
    config = protocol()
    config["training"].update(pricing_batches=[8])
    config["reference"].update(banks=4, rungs=[64,128], batch_size=32, ess_min=1., minimum_tail_rows=1)
    import tensorflow as tf
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    memory = configure_tensorflow_gpu_memory_growth(tf, require_gpu=False)
    trained = run_training_cohort(config, four_dimensional_bridge(), tmp_path / "training",
        memory_policy=memory, max_seconds=120., calibration_only=True)
    path = Path(trained["checkpoint"])
    kwargs = dict(repo=REPO, fixture=True, stop_after="refresh", training_checkpoint=str(path), previous_source_root=REPO,
        allowance={"campaign_remaining_seconds": 7200., "diagnostic_remaining_seconds": 3600.})
    result = execute_master(config, tmp_path / "campaign", **kwargs)
    assert result["status"] == "MASTER_REFRESH_COMPLETE", result
    ledger = json.loads((tmp_path / "campaign/campaign.json").read_text())
    assert set(ledger["stages"]) == {"refresh-price-training", "refresh-price-downstream"}
    priced = json.loads(Path(ledger["stages"]["refresh-price-training"]["result_path"]).read_text())["result"]
    assert all(r["exact_state_restore"] for r in priced["restored_training_states"])
    assert priced["method"] == result["details"]["method"] == "neutra"
    assert result["details"]["missing_cost_categories"] == []
    assert result["details"]["training_quote"]["scenarios"]["calibration"]["optimizer_updates"] == 0
    repeated = execute_master(config, tmp_path / "campaign", **kwargs)
    assert repeated == result
    assert json.loads((tmp_path / "campaign/campaign.json").read_text())["spent_seconds"] == ledger["spent_seconds"]


@pytest.mark.parametrize("failed", [False, True])
def test_preparation_pricing_preserves_progress_and_structured_failure(tmp_path, monkeypatch, failed):
    import tensorflow as tf
    from bayesfilter.inference import hmc_kernel_tuning, q20_production_hmc
    from bayesfilter.inference.hmc_preparation import HMCPreparationFailure
    from bayesfilter.inference.q20_master_stages import price_preparation
    monkeypatch.setattr(q20_production_hmc, "draw_start_bank",
        lambda *args: (tf.zeros([4, 4], tf.float64), {"role": "test_start_fixture"}))

    def prepare(**kwargs):
        kwargs["progress_callback"]("geometry_completed", {"checked": True})
        if failed:
            raise HMCPreparationFailure("fixture rejection", details={
                "stage": "bootstrap", "hard_vetoes": ["nonfinite_proposal"],
                "diagnostics": {"energy": float("nan")}})
        kwargs["progress_callback"]("handoff_completed", {"checked": True})

    monkeypatch.setattr(hmc_kernel_tuning, "prepare_operational_windowed_mass_handoff", prepare)
    output = tmp_path / "preparation"
    if failed:
        with pytest.raises(HMCPreparationFailure, match="fixture rejection"):
            price_preparation(tiny_protocol(), four_dimensional_bridge(), output, beta=1., max_seconds=30.)
    else:
        result = price_preparation(tiny_protocol(), four_dimensional_bridge(), output, beta=1., max_seconds=30.)
        assert result["preparation_progress_file"] == str(output / "preparation_progress.json")
    progress = json.loads((output / "preparation_progress.json").read_text())
    assert progress["artifact_authority"] is False
    assert any(e["phase"] == "geometry_completed" for e in progress["events"])
    assert progress["status"] == ("failed" if failed else "prepared")
    if failed:
        assert progress["failure"]["details"]["hard_vetoes"] == ["nonfinite_proposal"]
        assert progress["failure"]["details"]["diagnostics"]["energy"] == {"nonfinite": "nan"}
