"""Diagnostic gate tests, intentionally CPU-only."""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES","-1")
import tensorflow as tf
from bayesfilter.testing.surrogate_hmc_covariance_cost_diagnostic import compare_initial_region

def fixture(score,valid=(True,True)):
    return dict(value=tf.constant([1.,2.]),score=tf.constant(score),valid=tf.constant(valid),
        steps_completed=tf.constant([2,2]),particles=tf.ones([2,4,1]),particle_tangent=tf.ones([2,3,4,1]))

def test_offcentre_value_score_error_veto():
    base=fixture([[1.,2.],[3.,4.]])
    candidate=fixture([[1.,2.],[3.1,4.]])
    result=compare_initial_region(candidate,base,2)
    assert not result["passed"] and not result["arithmetic_passed"]

def test_preexisting_invalid_row_does_not_stop_repair():
    base=fixture([[1.,2.],[3.,4.]],(True,False))
    candidate=fixture([[1.,2.],[8.,9.]],(True,False))
    result=compare_initial_region(candidate,base,2)
    assert result["passed"] and not result["hmc_health_passed"]

def test_newly_invalid_row_vetoes_repair():
    base=fixture([[1.,2.],[3.,4.]])
    candidate=fixture([[1.,2.],[3.,4.]],(False,True))
    assert not compare_initial_region(candidate,base,2)["passed"]


def test_consumer_factory_is_frozen_and_restored(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from bayesfilter.testing import surrogate_hmc_five_model as runtime
    from bayesfilter.testing.surrogate_hmc_covariance_cost_diagnostic import make_consumer_pair
    baseline=tmp_path/"baseline.py"
    baseline.write_text("def make_batched_score_filter(): return 3.0\n")
    factory=lambda:7.0
    monkeypatch.setattr(runtime,"make_batched_score_filter",factory)
    def bindings(case):
        @tf.function(input_signature=[tf.TensorSpec([None,1],tf.float64)],autograph=False)
        def force(q):
            return q*tf.cast(runtime.make_batched_score_filter(),tf.float64)
        return {"force":SimpleNamespace(function=force)}
    monkeypatch.setattr(runtime,"bindings",bindings)
    before,after=make_consumer_pair(None,baseline)
    assert runtime.make_batched_score_filter is factory
    tf.debugging.assert_equal(before(tf.ones([4,1],tf.float64)),tf.fill([4,1],tf.constant(3.,tf.float64)))
    tf.debugging.assert_equal(after(tf.ones([4,1],tf.float64)),tf.fill([4,1],tf.constant(7.,tf.float64)))


def test_consumer_factory_restored_after_trace_failure(tmp_path,monkeypatch):
    import pytest
    from types import SimpleNamespace
    from bayesfilter.testing import surrogate_hmc_five_model as runtime
    from bayesfilter.testing.surrogate_hmc_covariance_cost_diagnostic import make_consumer_pair
    baseline=tmp_path/"baseline.py"
    baseline.write_text("def make_batched_score_filter(): return 3.0\n")
    factory=runtime.make_batched_score_filter
    class Broken:
        def get_concrete_function(self): raise RuntimeError("deliberate trace failure")
    monkeypatch.setattr(runtime,"bindings",lambda case:{"force":SimpleNamespace(function=Broken())})
    with pytest.raises(RuntimeError,match="deliberate trace"):
        make_consumer_pair(None,baseline)
    assert runtime.make_batched_score_filter is factory


def test_actual_consumer_pair_tiny_cpu(tmp_path,monkeypatch):
    from dataclasses import replace
    from pathlib import Path
    import json
    from bayesfilter.testing import surrogate_hmc_five_model as runtime
    from bayesfilter.highdim import covariance_proposal_score_batch_tf as batch
    from bayesfilter.testing.surrogate_hmc_covariance_cost_diagnostic import run_consumer_pair
    case=runtime.load_case("lgssm")
    monkeypatch.setattr(runtime,"N",64)
    case=replace(case,observations=case.observations[:2],
        fixed=(case.fixed[0][:64],case.fixed[1][:2,:64],case.fixed[2][:2,:64]),
        controls=replace(case.controls,flow_steps=2,reset_steps=4,correction_steps=0))
    case.selection_hash="tiny_cpu_mechanics_fixture"
    answer=run_consumer_pair(case,tmp_path,Path(batch.__file__),repetitions=1)
    saved=json.loads((tmp_path/"consumer-paired.json").read_text())
    assert answer["status"]=="CONSUMER_PAIRED_PARITY_PASS_COST_MEASURED"
    assert answer["maximum_scaled_force_gap"]==0
    assert all(len(v)==1 for v in saved["warmed_seconds"].values())
    assert saved["cpu_only"] and not saved["canonical_admission"]
    assert not saved["baseline_tuning_authority"]
