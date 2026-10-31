"""CPU-only scheduler controls; no frameworks, fitting or sampling."""
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def module():
    path=Path(__file__).resolve().parents[1]/'scripts/neutra_attribution_campaign.py'
    spec=importlib.util.spec_from_file_location('attribution_queue_test',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m


def fixture_controller(module,tmp_path,monkeypatch,*,benefit=1.):
    outputs={}
    class Controller:
        def __init__(self):
            self.state={'attempts':[{'output':str(tmp_path/'prior')}]}
            self.calls=[]
        def remaining(self):return dict(gpu_process_seconds=1000000.,cpu_core_seconds=1000000.)
        def sync(self,*args):pass
        def execute(self,job,**kwargs):
            self.calls.append((job,kwargs))
            p=tmp_path/(job+'-r1');p.mkdir()
            row=dict(job=job,output=str(p),source=str(tmp_path/'source'),wall_seconds=10.,
                cpu_core_seconds=12.,status='complete')
            endpoint=dict(passed=True,training=[dict(wall_seconds=1.,updates=512)],
                heldout=dict(heldout_cross_entropy=2.),forward_kl_reference=dict(mean=.1,
                normal_approximation_99_lower=0.,normal_approximation_99_upper=.2))
            outputs[str(p)]=dict(endpoints={'affine':endpoint,'nonlinear':endpoint,'canonical':endpoint},
                paired_kl_benefit=dict(mean=benefit,normal_approximation_99_lower=benefit-.1),
                affine_kl_lower_bound=.458,nonlinear_below_proven_bound=True,
                total_updates=32768,achieved_time_ratio=1.1,stopped='time_target_reached')
            return row
    monkeypatch.setattr(module,'checked_result',lambda row:outputs[row['output']])
    return Controller(),outputs


def test_complete_queue_includes_controls_and_terminal_resume_launches_none(module,tmp_path,monkeypatch):
    c,_=fixture_controller(module,tmp_path,monkeypatch)
    assert module.run_campaign(c)==0
    names=[j for j,_ in c.calls]
    assert sum(j.startswith('attribution-confirm-') for j in names)==32
    assert sum(j.startswith('attribution-capacity-') for j in names)==24
    assert sum(j.startswith('attribution-time-') for j in names)==4
    assert sum(j.startswith('attribution-transfer-') for j in names)==8
    assert sum(j.startswith('attribution-rate-') for j in names)==8
    calls=len(c.calls)
    assert module.run_campaign(c)==0 and len(c.calls)==calls
    result=json.loads((tmp_path/'attribution-result.json').read_text())
    assert not result['scientific_promotion']
    assert (tmp_path/'attribution-results.md').exists()


def test_negative_primary_still_runs_capacity_repairs_and_reports_no_transfer(module,tmp_path,monkeypatch):
    c,_=fixture_controller(module,tmp_path,monkeypatch,benefit=-1.)
    assert module.run_campaign(c)==0
    names=[j for j,_ in c.calls]
    assert any(j.startswith('attribution-time-') for j in names)
    assert any(j.startswith('attribution-capacity-') for j in names)
    assert not any(j.startswith('attribution-transfer-') for j in names)
    result=json.loads((tmp_path/'attribution-result.json').read_text())
    assert all(not r['passed'] for r in result['comparisons'].values())
