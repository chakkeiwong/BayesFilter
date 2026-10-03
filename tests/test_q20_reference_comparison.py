"""Known-distribution integration, uncertainty veto and resumable evaluation."""
import copy
import json
from pathlib import Path

import pytest
import tensorflow as tf

from tests.test_q20_master_integration import protocol
from tests.test_q20_production_repair import four_dimensional_bridge
from bayesfilter.inference.q20_production_reference import importance_summary, run_reference
from bayesfilter.inference.q20_production_comparison import compare_quantities, posterior_summary
from bayesfilter.inference.q20_production_config import frozen_scope_hash


def test_independent_importance_estimates_match_known_gaussian():
    config=protocol()
    values=tf.random.stateless_normal([8,4096,4],[981,12],dtype=tf.float64)
    # Exponential tilt of N(0,I) by exp(a*x-a^2/2) is N(a,I).
    shift=tf.constant([.1,-.2,.3,.05],tf.float64)
    logs=tf.reduce_sum(values*shift,-1)-tf.reduce_sum(shift**2)/2
    summary=importance_summary(config,values,logs)
    from bayesfilter.inference.q20_production_config import PARAMETERS
    for i,name in enumerate(PARAMETERS):
        row=summary["quantities"][name]
        assert row["valid"]
        assert abs(row["estimate"]-float(shift[i]))<6*row["mcse"]
    assert summary["concentration_passed"] and summary["tails_observed"]
    # A numeric but invalid uncertainty result cannot pass equality with itself.
    bad=copy.deepcopy(summary["quantities"])
    bad[PARAMETERS[0]]["valid"]=False
    assert not compare_quantities(config,bad,bad)["passed"]


def test_unvisited_sign_and_collapsed_bank_do_not_get_zero_error():
    config=protocol()
    values=tf.abs(tf.random.stateless_normal([4,128,4],[23,1],dtype=tf.float64))
    logs=tf.concat([tf.zeros([1,128],tf.float64),tf.fill([3,128],tf.constant(-1e300,tf.float64))],0)
    report=importance_summary(config,values,logs)
    assert not report["concentration_passed"]
    assert not report["quantities"]["positive_theta_2"]["valid"]
    assert report["quantities"]["positive_theta_2"]["mcse"] is None
    json.dumps(report,allow_nan=False)


def test_reference_checkpoint_exact_replay_and_empty_posterior(tmp_path):
    config=protocol()
    config["reference"].update(banks=4,rungs=[64,128],batch_size=32,ess_min=1.,minimum_tail_rows=1)
    bridge=four_dimensional_bridge()
    first=run_reference(config,bridge,tmp_path/"first")
    resumed=run_reference(config,bridge,tmp_path/"again",resume_chunks=tmp_path/"first/chunks")
    assert first==resumed
    summary=posterior_summary(config,tf.zeros([0,4,4],tf.float64),target_signature="fixture",label="empty",sequential_passed=False)
    assert summary["quantities"]=={}
    assert summary["production_qualified"] is False
    changed=copy.deepcopy(config)
    changed["seed"][0]+=1
    from bayesfilter.runtime.durable_tensor_checkpoint import CheckpointError
    with pytest.raises(CheckpointError):
        run_reference(changed,bridge,tmp_path/"wrong",resume_chunks=tmp_path/"first/chunks")
    changed=copy.deepcopy(config)
    changed["budget"]["campaign_remaining_seconds"]-=1
    assert frozen_scope_hash(config)==frozen_scope_hash(changed)
