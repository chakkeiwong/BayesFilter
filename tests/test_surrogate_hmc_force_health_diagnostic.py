"""CPU-only diagnostic authority and interrupted call-chain recovery tests."""
from dataclasses import asdict
import json
from types import SimpleNamespace
import pytest
import tensorflow as tf
from bayesfilter.testing import surrogate_hmc_force_health_diagnostic as diagnostic
from bayesfilter.testing import surrogate_hmc_five_model as runtime


def fixture_record(tmp_path, **scope_change):
    scope=dict(model="fixture",horizon=2,particles=4,source_sha256={"source.py":"old"})
    record=dict(scope=scope,status="FILTER_SCOPE_SELECTED_FOR_EXPLORATORY_HMC",
        selected=dict(valid=True,controls=asdict(runtime.ProposalControls()),
                      beta=[1/3]*3))
    record["selection_hash"]=runtime.hashed(record)
    path=tmp_path/"selection.json"
    runtime.save(path,record)
    current=dict(scope,source_sha256={"source.py":"new"},**scope_change)
    case=SimpleNamespace(scope=lambda:current,force_dtype=tf.float64)
    return case,path


def test_replay_preserves_fp64_inputs_without_issuing_tuning_authority(tmp_path):
    case,path=fixture_record(tmp_path)
    result=diagnostic.replay_inputs(case,path)
    tf.debugging.assert_equal(case.beta,tf.constant([1/3]*3,tf.float64))
    assert result["source_changes"]=={"source.py":{"previous":"old","current":"new"}}
    assert not result["tuning_authority"]
    assert not hasattr(case,"selection_hash")


def test_replay_rejects_changed_scientific_scope(tmp_path):
    case,path=fixture_record(tmp_path,horizon=3)
    with pytest.raises(ValueError,match="scope mismatch: horizon"):
        diagnostic.replay_inputs(case,path)


def test_replay_rejects_corrupt_record(tmp_path):
    case,path=fixture_record(tmp_path)
    record=json.loads(path.read_text())
    record["selected"]["beta"]=[.1,.1,.8]
    runtime.save(path,record)
    with pytest.raises(ValueError,match="corrupt"):
        diagnostic.replay_inputs(case,path)


def test_failed_consumer_is_not_promoted_by_healthy_standalone():
    force=tf.constant([[1.,2.],[3.,4.]],tf.float64)
    result=diagnostic.compare_paths(force,tf.constant([True,False]),
                                   force,tf.constant([True,True]))
    assert not result["passed"] and not result["same_health"]
    assert result["consumer_healthy"].numpy().tolist()==[True,False]


def test_actual_consumer_evidence_survives_trace_compile_failure(tmp_path,monkeypatch):
    case,path=fixture_record(tmp_path)
    case.p=2; case.horizon=2; case.spec=object(); case.precision_policy="uniform"
    case.observations=tf.zeros([2,1],tf.float64)
    case.fixed=(tf.zeros([4,1],tf.float64),tf.zeros([2,4,1],tf.float64),
                tf.zeros([2,4,1],tf.float64))
    calls=[]
    def actual(q):
        calls.append(q)
        return SimpleNamespace(force=tf.ones_like(q),force_valid=tf.constant([True,False,True,True]))
    monkeypatch.setattr(runtime,"bindings",lambda c:dict(force=SimpleNamespace(function=actual)))
    def fails(*a,**k):
        raise RuntimeError("trace unavailable")
    monkeypatch.setattr(diagnostic,"make_batched_score_filter",fails)
    with pytest.raises(RuntimeError,match="trace unavailable"):
        diagnostic.run_diagnostic(case,tmp_path,path)
    saved=json.loads((tmp_path/"force-health.json").read_text())
    assert saved["consumer"]["valid"]==[True,False,True,True]
    assert not saved["hmc_executed"] and not saved["posterior_sampling_complete"]
    tf.debugging.assert_equal(calls[0],
        tf.constant([[s,s] for s in (-.15,-.05,.05,.15)],tf.float64))
