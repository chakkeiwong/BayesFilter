"""Stage-status injection in the real canonical engine, never admission.

Inject at the checked producer/consumer boundary. Numerical factor producers
are tested separately; these cases verify sticky propagation through every
real stage and public batch wrapper without fabricating a natural failure rate.
"""
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES','-1')
from pathlib import Path
import runpy
import pytest
import tensorflow as tf
import bayesfilter.highdim.ledh_canonical_score_tf as core
from bayesfilter.highdim.ledh_canonical_batch_fused_tf import (
    canonical_batch_fused_value_score,canonical_batch_fused_value_score_whileloop,
    _single_cloud_model)
from bayesfilter.highdim.ledh_failure_status_tf import FailureStage

fixture=runpy.run_path(str(Path(__file__).with_name('test_ledh_failure_contract.py')))['fixture']

@pytest.mark.parametrize('entry_name',['direct','fused','whileloop'])
def test_first_middle_last_checked_stage_failures(monkeypatch,entry_name):
    model,inputs,controls=fixture()
    original=core.record_failure
    injection={}
    def injected(status,value_valid,force_valid,*,time,stage,**kwargs):
        fire=tf.equal(tf.cast(time,tf.int32),injection['time']) & tf.equal(int(stage),injection['stage'])
        value_valid=tf.reduce_all(value_valid) & ~(fire & ~injection['force_only'])
        force_valid=tf.reduce_all(force_valid) & ~fire
        return original(status,value_valid,force_valid,time=time,stage=stage,**kwargs)
    monkeypatch.setattr(core,'record_failure',injected)
    @tf.function(input_signature=[tf.TensorSpec([],tf.int32),tf.TensorSpec([],tf.int32),tf.TensorSpec([],tf.bool)],jit_compile=True)
    def run(stage,time,force_only):
        injection.update(stage=stage,time=time,force_only=force_only)
        theta=tf.constant([.6],tf.float64)
        if entry_name=='direct':
            kw=dict(controls);kw['flow_substeps']=kw.pop('substeps')
            return core.canonical_value_and_analytical_score(
                _single_cloud_model(model,theta,tf.ones_like(theta)),theta,*inputs,
                with_score=True,return_status=True,**kw)
        entry=canonical_batch_fused_value_score if entry_name=='fused' else canonical_batch_fused_value_score_whileloop
        return entry(model,theta[None],tf.ones([1,1,1],tf.float64),*inputs,**controls)
    baseline,score,status=run(tf.constant(-99),tf.constant(-99),tf.constant(False))
    assert bool(tf.reduce_all(status['value_valid'])) and bool(tf.reduce_all(status['force_valid']))
    for stage in (FailureStage.INITIAL,FailureStage.PREDICT,FailureStage.FLOW,
                  FailureStage.WEIGHTS,FailureStage.UPDATE,FailureStage.RESET,
                  FailureStage.CORRECTION,FailureStage.CALLBACK):
        for time in (0,1,2):
            for force_only in (False,True):
                value,force,status=run(tf.constant(int(stage)),tf.constant(time),tf.constant(force_only))
                assert not bool(tf.reduce_any(status['force_valid'])),(entry_name,stage,time,force_only)
                tf.debugging.assert_equal(force,tf.zeros_like(force))
                tf.debugging.assert_equal(status['first_failure_time'],tf.fill(tf.shape(status['first_failure_time']),time))
                tf.debugging.assert_equal(status['first_failure_stage'],tf.fill(tf.shape(status['first_failure_stage']),int(stage)))
                if force_only:
                    assert bool(tf.reduce_all(status['value_valid']))
                    tf.debugging.assert_equal(value,baseline)
                else:
                    assert not bool(tf.reduce_any(status['value_valid']))
                    tf.debugging.assert_equal(value,tf.fill(tf.shape(value),tf.constant(float('-inf'),value.dtype)))
    assert run.experimental_get_tracing_count()==1
