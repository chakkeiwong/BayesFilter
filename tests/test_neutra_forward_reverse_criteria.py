"""Reference fixtures for warm-start versus final endpoint decisions."""
import copy
import json
import pytest

from bayesfilter.testing.neutra_forward_reverse_criteria import (
    PAIR_CRITERION, forward_warm_start, assess_pair_criteria,
)


def endpoint():
    return dict(passed=False,checkpoint_reloaded=True,heldout=dict(
        finite=True,cross_entropy_delta=-2.,maximum_responsibility_discrepancy=.01,
        q_summary=[.11,.89,.2],reference_summary=[.10,.90,0.],summary_z_max=8.))


def probe():
    return dict(complete=True,finite=True,rows=1000,valid_rows=1000)


def test_shifted_forward_map_is_eligible_without_erasing_accuracy_rejection():
    e=endpoint();result=forward_warm_start(e,probe(),2)
    assert result['passed'] and not result['full_accuracy_screen_passed']
    assert result['feature_z_max']==8. and result['criterion']==PAIR_CRITERION
    assert e['passed'] is False


@pytest.mark.parametrize('mass',(0.,.049))
def test_absolute_mass_tolerance_cannot_hide_loss_of_small_mode(mass):
    e=endpoint();e['heldout'].update(q_summary=[mass,1-mass],maximum_responsibility_discrepancy=.1-mass)
    result=forward_warm_start(e,probe(),2)
    assert not result['passed'] and 'underrepresented_mode' in result['reasons']


@pytest.mark.parametrize('field,value',[
    ('finite',False),('cross_entropy_delta',.251),('cross_entropy_delta',None),
    ('maximum_responsibility_discrepancy',.151),('q_summary',[float('nan'),.9]),
    ('reference_summary',[0.,1.]),('q_summary',[]),
])
def test_invalid_or_grossly_mismatched_warm_start_is_rejected(field,value):
    e=endpoint();e['heldout'][field]=value
    assert not forward_warm_start(e,probe(),2)['passed']


@pytest.mark.parametrize('field,value',[('finite',False),('complete',False),('valid_rows',999),('rows',999)])
def test_missing_or_nonfinite_probe_still_vetoes_warm_start(field,value):
    p=probe();p[field]=value
    assert not forward_warm_start(endpoint(),p,2)['passed']


def test_checkpoint_reload_remains_required():
    e=endpoint();e['checkpoint_reloaded']=False
    assert not forward_warm_start(e,probe(),2)['passed']


def test_pair_reassessment_preserves_original_and_final_vetoes(tmp_path):
    path=tmp_path/'probe.json';path.write_text(json.dumps(probe()))
    e=endpoint();e['probe_path']=str(path)
    training=[dict(phase='forward',updates=10,finite=True,clipped_updates=0),
              dict(phase='reverse',rate=.0001,rung=256,updates=256,finite=True,clipped_updates=0),
              dict(phase='reverse',rate=.0001,rung=1024,updates=768,finite=True,clipped_updates=0)]
    r=dict(status='fit_complete',forward=e,training=training,branches=[
        dict(rate=.0001,updates=256,passed=False,endpoint={'passed':True}),
        dict(rate=.0001,updates=1024,passed=False,endpoint={'passed':False})])
    original=copy.deepcopy(r)
    view=assess_pair_criteria(r,2)
    assert view['branches'][0]['passed'] and not view['branches'][1]['passed']
    assert not view['branches'][0]['legacy_both_endpoint_passed']
    assert r==original
    assert assess_pair_criteria(view,2)==view
    for key,value in [('clipped_updates',129),('finite',False)]:
        bad=copy.deepcopy(r);bad['training'][1][key]=value
        assert not assess_pair_criteria(bad,2)['branches'][0]['passed']
    bad=copy.deepcopy(r);bad['training']=bad['training'][:1]
    assert not assess_pair_criteria(bad,2)['branches'][0]['passed']
