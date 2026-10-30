"""Focused geometry/reference and actual shortlist-consumer regressions."""
import json

import pytest
import tensorflow as tf

from bayesfilter.testing.neutra_warm_start_campaign import make_transport,write_json
from bayesfilter.testing.neutra_warm_start_targets_tf import WarmStartTarget,F64
from bayesfilter.testing.neutra_warm_start_diagnostics import (
    features,feature_names,SHAPE_DIAGNOSTIC_PROFILE,RARE_EVENT_DIAGNOSTIC_PROFILE)
from bayesfilter.testing.neutra_geometry_diagnostics import GeometryProgram
from bayesfilter.testing import neutra_warm_start_qualification as consumer


def test_shape_profile_distinguishes_equal_moments_and_halfspace_but_wrong_valley():
    target=WarmStartTarget('mixture')
    a=tf.sqrt(tf.constant(26.,F64));b=tf.sqrt(tf.constant(51.,F64))
    left=tf.stack(([-a,-a,a,a],tf.constant([1.,-1.,1.,-1.],F64)),axis=1)
    right=tf.stack(([-b,tf.constant(-1.,F64),tf.constant(1.,F64),b],left[:,1]),axis=1)
    tf.debugging.assert_near(tf.reduce_mean(features(target,left),0),tf.reduce_mean(features(target,right),0))
    names=feature_names(target,profile=SHAPE_DIAGNOSTIC_PROFILE)
    index=names.index('valley_abs_x0_lt2')
    assert float(tf.reduce_mean(features(target,left,profile=SHAPE_DIAGNOSTIC_PROFILE),0)[index])==0.
    assert float(tf.reduce_mean(features(target,right,profile=SHAPE_DIAGNOSTIC_PROFILE),0)[index])==.5


def test_warped_events_use_the_unwarped_coordinate():
    target=WarmStartTarget('warped_mixture')
    unwarped=tf.constant([[-5.,-3.],[5.,1.],[8.,3.]],F64)
    actual=features(target,target.warp(unwarped),profile=SHAPE_DIAGNOSTIC_PROFILE)
    tf.debugging.assert_equal(actual[:,-3:],tf.cast(unwarped[:,1:2]<tf.constant([[-2.,0.,2.]],F64),F64))


def test_geometry_matches_scalar_finite_difference_and_detects_corrupt_pullback():
    target=WarmStartTarget('mixture');flow=make_transport(target,4,(11,91),variance_scale=.2)
    z=tf.constant([[.2,-.8],[1.2,.5],[-1.,2.]],F64)
    program=GeometryProgram(flow,target);rows=program.evaluate(z)
    tf.debugging.assert_less(tf.reduce_max(rows['manual_ad_scaled_error']),tf.constant(1e-10,F64))
    fd=program.differences(z)
    tf.debugging.assert_near(fd[2],rows['autodiff_score'],atol=1e-7,rtol=1e-7)
    original=flow.pullback_score_batch
    flow.pullback_score_batch=lambda z,g:original(z,g)+tf.constant(.125,F64)
    corrupted=GeometryProgram(flow,target).evaluate(z)
    assert float(tf.reduce_max(corrupted['manual_ad_scaled_error']))>1e-4


@pytest.mark.parametrize('confirm_pass',[True,False])
def test_shortlist_tries_another_map_before_one_holdout_and_stops_after_holdout(tmp_path,monkeypatch,confirm_pass):
    training=tmp_path/'training';training.mkdir();rows=[]
    for stage in ('rkl-2048','rkl-1024','warm'):
        write_json(training/f'{stage}.json',{'transport_hash':stage})
        write_json(training/f'{stage}-probe.json',{'complete':True,'finite':True,'rows':1000,'valid_rows':1000,'transport_hash':stage})
        rows.append({'stage':stage,'filename':f'{stage}.json','transport_hash':stage,
            'eligible':True,'probe_file':f'{stage}-probe.json'})
    write_json(training/'checkpoint-candidates.json',{'schema':'neutra.checkpoint_candidates.v1',
        'selection_order':'declared','candidates':rows})
    calls=[];opened=[]
    def qualify(target_name,seed,prepared,training,output,config,**kwargs):
        assert kwargs['confirm'] is False and not opened
        calls.append(kwargs['frozen_filename'])
        return {'selection_screen_passed':len(calls)==2,'reason':'fixture'}
    def confirm(target_name,prepared,output):
        opened.append(output)
        assert calls==['rkl-2048.json','rkl-1024.json']
        return {'qualified':confirm_pass,'heldout_consumed':True}
    monkeypatch.setattr(consumer,'qualify',qualify)
    monkeypatch.setattr(consumer,'confirm_qualification',confirm)
    result=consumer.qualify_shortlist('mixture',11,tmp_path/'reference',training,tmp_path/'result',
        {'hmc':{'job_wall_seconds':60}})
    assert len(opened)==1 and result['qualified']==confirm_pass
    assert result['selected_stage']=='rkl-1024'


def test_eligible_shortlist_without_valid_probe_never_reaches_hmc(tmp_path,monkeypatch):
    write_json(tmp_path/'frozen.json',{'transport_hash':'bad'})
    write_json(tmp_path/'probe.json',{'complete':True,'finite':False,'rows':1000,'valid_rows':999})
    write_json(tmp_path/'checkpoint-candidates.json',{'schema':'neutra.checkpoint_candidates.v1',
        'candidates':[{'stage':'bad','filename':'frozen.json','transport_hash':'bad',
            'eligible':True,'probe_file':'probe.json'}]})
    monkeypatch.setattr(consumer,'qualify',lambda *a,**k:pytest.fail('invalid map reached HMC'))
    with pytest.raises(ValueError,match='1000-point probe'):
        consumer.qualify_shortlist('mixture',11,tmp_path,tmp_path,tmp_path/'result',{'hmc':{'job_wall_seconds':60}})


def test_resuming_controller_cannot_skip_a_preserved_numerical_veto():
    import importlib.util
    from pathlib import Path
    path=Path(__file__).resolve().parents[1]/'scripts/run_neutra_geometry_repair.py'
    spec=importlib.util.spec_from_file_location('geometry_resume_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    controller=module.Controller.__new__(module.Controller)
    controller.record={'jobs':{'bad':{'output':'preserved-failed-diagnostic','result':{'continuation_veto':True}}}}
    with pytest.raises(RuntimeError,match='preserved numerical veto'):
        controller.phase('bad','geometry','mixture',11,prepared='unused')


def test_rare_event_reference_rule_rejects_the_observed_low_count_false_pass():
    estimate=tf.constant([2/24000],F64);expected=tf.constant([.001220703125],F64)
    mcse=tf.constant([.00008333159250226152],F64)
    variance=expected*(1-expected)
    error,old=consumer.reference_agreement(estimate,expected,mcse,variance,32768,
        continuous_count=0,strict_events=False)
    _,new=consumer.reference_agreement(estimate,expected,mcse,variance,32768,
        continuous_count=0,strict_events=True)
    assert bool((error<=old)[0]) and not bool((error<=new)[0])


def test_rare_event_is_required_in_each_retained_chain():
    target=WarmStartTarget('mixture')
    # All chains see the mode sides, but only chain3 sees the valley.
    x=tf.constant([[[-5.,-3.],[5.,3.],[-5.,-3.],[5.,3.]],
                   [[5.,3.],[-5.,-3.],[5.,3.],[0.,0.]]],F64)
    result=consumer.retained_event_check(target,x,profile=RARE_EVENT_DIAGNOSTIC_PROFILE)
    i=result['quantity_names'].index('valley_abs_x0_lt2')
    assert not result['passed']
    tf.debugging.assert_equal(result['event_counts_per_chain'][:,i],tf.constant([0.,0.,0.,1.],F64))
