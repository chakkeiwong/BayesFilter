"""CPU-only recovery checks for interrupted mechanics diagnostics."""
import json
from types import SimpleNamespace
import pytest
import tensorflow as tf
from bayesfilter.testing import surrogate_hmc_five_model as runtime


def setup_case(monkeypatch, valid):
    case = SimpleNamespace(p=1,scope=lambda: {"fixture":"progress-test"})
    target = SimpleNamespace(function=lambda q:SimpleNamespace(potential=tf.zeros([4],tf.float64)))
    force = SimpleNamespace(function=lambda q:SimpleNamespace(
        force=tf.zeros_like(q), force_valid=tf.constant(valid)))
    monkeypatch.setattr(runtime,"bindings",lambda c:dict(target=target,force=force))
    return case


def test_invalid_initial_force_is_saved_without_sampling(tmp_path,monkeypatch):
    case=setup_case(monkeypatch,[True,False,True,True])
    def forbidden(*a,**k):
        raise AssertionError("invalid start must not launch sampling")
    monkeypatch.setattr(runtime,"sample_neural_force_hmc",forbidden)
    result=runtime.pilot(case,tmp_path)
    saved=json.loads((tmp_path/"pilot.json").read_text())
    assert result["status"] == saved["status"] == "PILOT_INITIAL_HEALTH_VETO"
    assert not saved["hmc_executed"] and not saved["posterior_sampling_complete"]
    assert saved["initial_force_valid"] == [True,False,True,True]


def test_completed_cold_chain_survives_failure_of_warmed_run(tmp_path,monkeypatch):
    case=setup_case(monkeypatch,[True]*4)
    monkeypatch.setattr(runtime.tf,"function",lambda **kw:lambda f:f)
    chain=SimpleNamespace(positions=tf.zeros([2,4,1],tf.float64),
        potentials=tf.zeros([2,4],tf.float64),accepted=tf.ones([2,4],tf.bool),
        delta_h=tf.zeros([2,4],tf.float64),force_fallback=tf.zeros([2,4],tf.bool),
        divergence=tf.zeros([2,4],tf.bool),finite_status=tf.ones([2,4],tf.bool))
    calls=[]
    def sample(*a,**kw):
        calls.append(1)
        if len(calls)==2:
            raise TimeoutError("simulated interruption after completed cold run")
        return chain
    monkeypatch.setattr(runtime,"sample_neural_force_hmc",sample)
    with pytest.raises(TimeoutError):
        runtime.pilot(case,tmp_path)
    progress=json.loads((tmp_path/"pilot-progress.json").read_text())
    assert progress["status"] == "COLD_RUN_COMPLETE_WARM_RUN_PENDING"
    assert not progress["cold_run_health"]["veto"]
    assert len(progress["cold_archive"]) == 6
    restored=tf.io.parse_tensor(tf.io.read_file(str(tmp_path/"cold_positions.tensor")),
                                out_type=tf.float64)
    tf.debugging.assert_equal(restored,chain.positions)
    assert not (tmp_path/"pilot.json").exists()
