"""Complete process-supervised known-target execution, explicitly smoke only."""
import json
from pathlib import Path
import pytest

from tests.test_q20_master_integration import protocol
from bayesfilter.inference.q20_master_program import execute_master
from bayesfilter.inference.q20_production_config import validate_protocol


def test_calibration_master_prices_only_training_and_preserves_partial_cohort(tmp_path):
    config = protocol()
    config["training"].update(pricing_batches=[8])
    root = tmp_path / "calibration"
    kwargs = dict(repo=Path(__file__).resolve().parents[1], fixture=True, stop_after="calibrate",
        allowance={"campaign_remaining_seconds": 1200., "diagnostic_remaining_seconds": 300.})
    result = execute_master(config, root, **kwargs)
    assert result["status"] == "TRAINING_SANITY_PILOT_COMPLETE", result
    assert result["production_qualified"] is False
    state = json.loads((root / "campaign.json").read_text())
    assert set(state["stages"]) == {"price-training", "calibration"}
    checkpoint = json.loads(Path(result["details"]["training"]["checkpoint"]).read_text())
    assert checkpoint["status"] == "sanity_pilot_complete"
    assert len(checkpoint["cohort"]) == 1
    assert all(name.startswith("direct-") for name in checkpoint["cohort"])
    before = state["spent_seconds"]
    repeated = execute_master(config, root, **kwargs)
    assert repeated == result
    assert json.loads((root / "campaign.json").read_text())["spent_seconds"] == before


def test_real_pricing_worker_stops_early_and_resume_does_not_repeat(tmp_path):
    from tests.test_q20_production_repair import tiny_protocol
    config = tiny_protocol()
    # Inflate the planned work, not the two actual pricing updates. This must
    # stop after the first measured scope, before any expensive work starts.
    config["training"].update(pricing_batches=[8], rungs=[100000, 200000],
                              cohort_min_updates=100000)
    validate_protocol(config)
    kwargs = dict(repo=Path(__file__).resolve().parents[1], fixture=True,
        allowance={"campaign_remaining_seconds":120., "diagnostic_remaining_seconds":100.}, stop_after="price")
    root = tmp_path / "early-cost-stop"
    result = execute_master(config, root, **kwargs)
    assert result["status"] == "UNDER_BUDGETED", result
    assert result["details"]["forecast"]["full_campaign_priced"] is False
    before = json.loads((root / "campaign.json").read_text())
    assert [a["stage"] for a in before["attempts"]] == ["price-neutra"]
    receipt = json.loads(Path(before["stages"]["price-neutra"]["result_path"]).read_text())
    assert len(receipt["result"]["training"]["rows"]) == 1
    # One width at beta one is the entire plain-NeuTra price inventory.
    assert not receipt["result"]["training_quote"]["missing_training_scopes"]
    assert receipt["result"]["training"]["rows"][0]["beta"] == 1.
    assert execute_master(config, root, **kwargs) == result
    after = json.loads((root / "campaign.json").read_text())
    assert before["attempts"] == after["attempts"]
    assert before["spent_seconds"] == after["spent_seconds"]


@pytest.mark.xfail(strict=True, reason=(
    "Existing posterior assessment applies continuous tail ESS to binary "
    "positive_theta_2; its constant upper-tail indicator prevents admission. "
    "Keep the success contract pending the separate diagnostic repair."))
def test_master_stops_after_valid_plain_estimate_and_reuses_completed_stages(tmp_path):
    config=protocol()
    config["training"].update(betas=[0.,1.],pricing_batches=[8])
    config["tuning"].update(initial_epsilon=.3,max_repairs_per_family=4,
        practical_region=[.00001,.99999],repair_region=[.000001,.999999],
        total_budget_units=50,repair_reserve_units=20)
    # Loose screens exercise dispatch and receipts on a known distribution.
    # They are smoke hypotheses and cannot qualify research accuracy.
    config["posterior"].update(warmup_rhat=10.,retained_rhat=10.,bulk_ess=1.,tail_ess=1.)
    config["assessment"].update(mean_margin_sd=10.,quantile_margin_sd=10.,event_margin=1.)
    config["reference"].update(banks=4,rungs=[512,2048],batch_size=32,ess_min=2.,minimum_tail_rows=1)
    validate_protocol(config)
    kwargs=dict(repo=Path(__file__).resolve().parents[1],fixture=True,
        allowance={"campaign_remaining_seconds":20000.,"diagnostic_remaining_seconds":1200.})
    result=execute_master(config,tmp_path/"campaign",**kwargs)
    assert result["status"]=="SMOKE_ESTIMATION_CHECKS_PASSED",result
    assert result["details"]["method"] == "neutra"
    assert result["production_qualified"] is False
    before=json.loads((tmp_path/"campaign/campaign.json").read_text())
    assert set(before["stages"]) == {"price-neutra", "train-neutra", "tune-neutra-beta1",
                                    "sample-neutra", "reference", "assess-neutra"}
    repeated=execute_master(config,tmp_path/"campaign",**kwargs)
    after=json.loads((tmp_path/"campaign/campaign.json").read_text())
    assert repeated==result
    assert before["attempts"]==after["attempts"]
    assert before["spent_seconds"]==after["spent_seconds"]
