"""Independent statistical/reporting checks; no TensorFlow or GPU import."""
import importlib.util
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
def load(name):
    path=ROOT/'docs/benchmarks'/f'{name}.py'
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def row(error):
    return dict(score_error=error,score=error,value=0.,absolute_value_error=0.,
      absolute_score_error=list(map(abs,error)),score_l2_error=math.sqrt(sum(x*x for x in error)))

def test_mean_vector_uncertainty_distinguishes_error_norms():
    report=load('summarize_sqmc_ksc_reset_repair')
    r=report.moments([row([1.,0.]),row([-1.,0.]),row([1.,0.]),row([-1.,0.])])
    assert r['norm_mean_score_error']==0.
    assert r['mean_replicate_score_l2_error']==1.
    assert abs(r['mean_score_error_component_se'][0]-1/math.sqrt(3))<1e-14
    assert r['mean_score_error_component_se'][1]==0.
    assert r['norm_mean_score_error_bootstrap_sd']>0.

def test_identical_replicates_have_zero_monte_carlo_uncertainty():
    report=load('summarize_sqmc_ksc_reset_repair')
    r=report.moments([row([3.,4.]) for _ in range(4)])
    assert r['norm_mean_score_error']==5.
    assert r['norm_mean_score_error_bootstrap_95ci']==[5.,5.]
    assert r['mean_score_error_component_se']==[0.,0.]

def test_selection_rejects_invalid_incomplete_and_value_harming_arms():
    runner=load('run_sqmc_ksc_reset_repair')
    def records(arm,n,value_error,score_error,valid=True):
        return [dict(arm=arm,valid=valid,absolute_value_error=value_error,score_l2_error=score_error) for _ in range(n)]
    rows=records('baseline',4,1.,2.)+records('design',4,1.05,1.)
    rows+=records('protection',4,2.,.1)+records('combined',4,.1,.1,False)
    rows+=records('steps4',3,.1,.1)
    assert runner.select_arm(rows)=='design'

def test_mean_vector_covariance_retains_correlated_coordinates():
    report=load('summarize_sqmc_ksc_reset_repair')
    r=report.moments([row([1.,2.]),row([-1.,-2.]),row([1.,2.]),row([-1.,-2.])])
    c=r['mean_score_error_covariance']
    assert abs(c[0][0]-1/3)<1e-14
    assert abs(c[0][1]-2/3)<1e-14
    assert abs(c[1][1]-4/3)<1e-14
    assert abs(r['mean_score_error_vector_rms_se']-math.sqrt(5/3))<1e-14
