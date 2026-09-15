"""Tiny CPU mechanics/reference tests for the real master consumers."""
import json
from pathlib import Path

import pytest
import tensorflow as tf

from tests.test_q20_production_repair import tiny_protocol, four_dimensional_bridge
from bayesfilter.inference.q20_production_hmc import tune_scope, sample_member, reverify_member, draw_start_bank
from bayesfilter.inference.q20_production_training import run_training_cohort


def protocol():
    config = tiny_protocol()
    config["tuning"].update(target_acceptance=.5, practical_region=[.01, .99],
                            repair_region=[.001, .999], initial_epsilon=.3)
    config["comparison"]["confirmation_replicates"] = 1
    return config


@pytest.mark.parametrize("method", ["identity", "classical", "neutra"])
def test_actual_tuning_member_and_posterior(tmp_path, method):
    config, bridge = protocol(), four_dimensional_bridge()
    if method in {"identity", "classical"}:
        config["tuning"]["initial_epsilon"] = 1.1
    export = None
    if method == "neutra":
        trained = run_training_cohort(config, bridge, tmp_path / "training",
            memory_policy={"mode": "tiny_cpu_reference"}, max_seconds=120.)
        cohort = json.loads(Path(trained["checkpoint"]).read_text())["cohort"]
        export = next(iter(cohort.values()))["exports"]["1.0"]
    tuned = tune_scope(config, bridge, tmp_path / "tune", method=method, training_export=export)
    assert tuned["verified_members"], tuned
    member = tuned["verified_members"][sorted(tuned["verified_members"])[0]]
    result = sample_member(config, bridge, tmp_path / "posterior", member_path=member, label=method)
    assert result["private_retained_raw"] is not None
    saved = json.loads((tmp_path / "posterior/result.json").read_text())
    assert saved["summary"]["reference_agreement"] == "incomplete"
    assert saved["production_qualified"] is False
    fresh, _ = draw_start_bank(config, bridge, 1., "fresh-confirmation")
    confirmation = reverify_member(config, bridge, tmp_path / "reverify",
        parent_member_path=member, beta=1., label="confirm-"+method, initial_position=fresh)
    assert confirmation["verified_members"], confirmation


def test_fixed_beta_scalar_and_retained_shapes_are_exact_batch_views():
    adapter = four_dimensional_bridge().fixed_beta_adapter(.5)
    points = tf.reshape(tf.range(24,dtype=tf.float64)/10., [2,3,4])
    flat = tf.reshape(points, [6,4])
    a,b,s = adapter.log_prob_and_grad_status(points)
    c,d,t = adapter.log_prob_and_grad_status(flat)
    tf.debugging.assert_equal(tf.reshape(a,[-1]),c)
    tf.debugging.assert_equal(tf.reshape(b,[6,4]),d)
    for key in s:
        tf.debugging.assert_equal(tf.reshape(s[key],[-1]),tf.reshape(t[key],[-1]))
    x,y,_ = adapter.log_prob_and_grad_status(flat[0])
    tf.debugging.assert_equal(x,c[0])
    tf.debugging.assert_equal(y,d[0])


def test_actual_chart_mixture_and_physical_replica_dispatch(tmp_path):
    from bayesfilter.inference.q20_master_stages import dispatch
    config,bridge=protocol(),four_dimensional_bridge()
    memory={"mode":"tiny_cpu_reference"}
    trained=dispatch(config,bridge,tmp_path/"train",{"stage":"train","cooperative_seconds":120.},memory)
    cohort=json.loads(Path(trained["checkpoint"]).read_text())["cohort"]
    maps=[item["exports"] for name,item in cohort.items() if name.startswith("continuation-")]
    chart_members,physical_members={},{}
    for beta in config["training"]["betas"][1:]:
        chart_members[str(beta)]=[]
        for index,exports in enumerate(maps):
            tuned=dispatch(config,bridge,tmp_path/f"chart-{beta}-{index}",
                {"stage":"tune","method":"neutra","beta":beta,
                 "training_export":exports[str(beta)],"start_label":"matched"},memory)
            assert tuned["verified_members"],tuned
            chart_members[str(beta)].append(next(iter(tuned["verified_members"].values())))
        classical=dispatch(config,bridge,tmp_path/f"classical-{beta}",
            {"stage":"tune","method":"classical","beta":beta,"start_label":"matched"},memory)
        assert classical["verified_members"],classical
        physical_members[str(beta)]=[next(iter(classical["verified_members"].values()))]
    for stage,members in (("ensemble",chart_members),("replica_exchange",physical_members)):
        result=dispatch(config,bridge,tmp_path/stage,{"stage":stage,"members_by_beta":members,
            "label":stage,"start_label":"posterior-matched"},memory)
        assert result["summary"]["reference_agreement"]=="incomplete"
        assert result["production_qualified"] is False
        # Copying checkpoints preserves exact transitions and summaries.
        resumed=dispatch(config,bridge,tmp_path/(stage+"-resumed"),{"stage":stage,"members_by_beta":members,
            "label":stage,"start_label":"posterior-matched","resume_chunks":str(tmp_path/stage/"chunks")},memory)
        a,b=result["summary"],resumed["summary"]
        assert a["retained_archive"]["sha256"]==b["retained_archive"]["sha256"]
        assert a["quantities"]==b["quantities"]


def test_actual_pricing_dispatch_and_finite_complete_forecast(tmp_path):
    from bayesfilter.inference.q20_master_stages import dispatch
    from bayesfilter.inference.q20_master_program import forecast_campaign
    config,bridge=protocol(),four_dimensional_bridge()
    config["training"]["pricing_batches"]=[8]
    config["reference"].update(banks=4,rungs=[64,128],batch_size=32,ess_min=1.,minimum_tail_rows=1)
    priced=dispatch(config,bridge,tmp_path/"price",{"stage":"price"},{"mode":"tiny_cpu_reference"})
    priced["worker_initialization_seconds"]=1. # Harness startup fixture; numerical prices are measured.
    quote=forecast_campaign(config,priced)
    assert quote["minimum_complete_seconds"]>0
    assert all(x>0 for x in quote["reserves"].values())
