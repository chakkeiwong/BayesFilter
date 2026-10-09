"""Diagnostic arithmetic tests: cancellation, pooling, and incomplete evidence."""
import importlib.util
import math
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('independent_report',Path(__file__).resolve().parents[1]/'docs/benchmarks/summarize_ledh_independent_validation.py')
report=importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def test_mean_cancellation_does_not_hide_per_run_error():
    rows=[dict(seed=i,value=0.,score=[(-1.)**i*10.],valid=True) for i in range(8)]
    result=report.arm(rows,dict(value=0.,score=[0.]))
    assert result['score_error_of_mean']==0.
    assert result['score_l2_error']['mean']==10.
    rows[0]['valid']=False
    failed=report.arm(rows,dict(value=0.,score=[0.]))
    assert not failed['complete'] and failed['invalid']==[0]
    assert 'score_l2_error' not in failed


def test_pooled_score_is_derivative_of_pooled_likelihood():
    rows=[dict(value=math.log(1.),score=[4.]),dict(value=math.log(3.),score=[-2.])]
    pooled=report.pooled_reference(rows)
    assert pooled['value']==pytest.approx(math.log(2.))
    assert pooled['score']==pytest.approx([-.5])
    h=1e-5
    def likelihood(x): return math.log((math.exp(4*x)+3*math.exp(-2*x))/2)
    assert pooled['score'][0]==pytest.approx((likelihood(h)-likelihood(-h))/(2*h),abs=1e-8)
    assert pooled['score_se'][0]>0


def test_pairing_uses_seed_and_reports_error_direction():
    rows=[]
    for i in range(8):
        rows.extend([dict(seed=i,method='old',value=float(i+2),score=[float(i+2)],valid=True),
                     dict(seed=i,method='new',value=float(i+1),score=[float(i+1)],valid=True)])
    reference=dict(value=0.,score=[0.])
    paired=report.paired(list(reversed(rows)),'old',reference)
    assert paired['absolute_value_error']['mean']==-1.
    assert paired['score_l2_error']['interval95']==[-1.,-1.]
    assert not report.paired(rows[:-1],'old',reference)['complete']


def test_unfinished_reference_is_not_interpreted_from_partial_files(tmp_path):
    # A worker can have written some results but still fail or be writing others.
    (tmp_path/'summary.json').write_text('not yet valid JSON')
    (tmp_path/'result.json').write_text('not yet valid JSON')
    jobs=[dict(kind=kind,status='running',output=str(tmp_path),model='sir_d18',data_seed=31)
          for kind in ('bootstrap','zhao_score')]
    sources={}
    assert report.reference_rows(jobs,'sir_d18',31,'unused',sources)=={}
    assert sources=={}
