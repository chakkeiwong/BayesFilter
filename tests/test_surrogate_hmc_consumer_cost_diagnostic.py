"""CPU-only tiny call-chain check for consumer-cost reporting; no timing claim."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from dataclasses import replace
import json
import pytest
import tensorflow as tf
from bayesfilter.testing import surrogate_hmc_five_model as runtime
from scripts import run_surrogate_hmc_performance_repair as master


@pytest.mark.parametrize("policy", ["uniform", "fp64_recursion_fp32_pairs"])
@pytest.mark.parametrize("interrupt_native", [False, True])
def test_consumer_diagnostic_executes_actual_lanes_and_serializes_tensors(tmp_path, monkeypatch, interrupt_native, policy):
    case = runtime.load_case("lgssm")
    monkeypatch.setattr(runtime, "N", 64)
    case = replace(case, observations=case.observations[:2],
        fixed=(case.fixed[0][:64],case.fixed[1][:2,:64],case.fixed[2][:2,:64]),
        controls=replace(case.controls,flow_steps=2,reset_steps=4,correction_steps=0))
    case = runtime.configure_precision(case, policy)
    case.selection_hash = "cpu_only_tiny_fixture_not_tuning_authority"
    monkeypatch.setattr(tf.config.experimental, "get_memory_info",
                        lambda _: {"cpu_only_fixture": True})
    if interrupt_native:
        from bayesfilter.highdim import covariance_proposal_score_batch_tf as batch
        original = batch.make_batched_score_filter
        def factory(*args, **kwargs):
            if kwargs.get("batch_size") == 4:
                raise RuntimeError("simulated native B4 failure")
            return original(*args, **kwargs)
        monkeypatch.setattr(batch, "make_batched_score_filter", factory)
        with pytest.raises(RuntimeError, match="simulated native B4"):
            master.consumer_cost_diagnostic(case, tmp_path)
        saved = json.loads((tmp_path/"consumer-cost.json").read_text())
        assert len(saved["measurements"]) == 5
        assert all(row["output"] for row in saved["measurements"])
        assert all(saved["lane_health"].values())
        assert set(saved["scaled_gaps"]) == {"B2_projection","B2_force","B2_B4_grouped_force"}
        assert all(gap <= .01 for gap in saved["scaled_gaps"].values())
        assert saved["status"] == "RUNNING"
        assert saved["posterior_sampling_complete"] is False
        return
    result = master.consumer_cost_diagnostic(case,tmp_path)
    assert result["status"] == "CONSUMER_PARITY_PASS_COST_MEASURED"
    assert result["scope"]["precision_policy"] == policy
    saved = json.loads((tmp_path/"consumer-cost.json").read_text())
    assert len(saved["measurements"]) == 6
    assert all(len(row["warmed_seconds"]) == 2 for row in saved["measurements"])
    assert saved["posterior_sampling_complete"] is False
    assert all(gap <= .01 for gap in saved["scaled_gaps"].values())
