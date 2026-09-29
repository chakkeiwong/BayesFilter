"""Negative evidence checks for the historical diagnostic reader."""
import importlib.util
import json
from pathlib import Path
import pytest

path=Path(__file__).parents[2]/'docs/benchmarks/analyze_sqmc_4route_comparison.py'
loader=importlib.util.spec_from_file_location('historical_sqmc_report',path)
report=importlib.util.module_from_spec(loader)
loader.loader.exec_module(report)

@pytest.mark.parametrize('defect',['missing_status','nonfinite','duplicate'])
def test_historical_loader_rejects_incomplete_evidence(tmp_path,defect):
    rows=[dict(seed=i,value=-1.,finite=True,program_valid=True,particle_count=1008) for i in range(16)]
    if defect=='missing_status':
        del rows[0]['program_valid']
    elif defect=='nonfinite':
        rows[0]['value']=float('nan')
    else:
        rows[0]['seed']=rows[1]['seed']
    p=tmp_path/'claim_attempt01'; p.mkdir()
    (p/'result.json').write_text(json.dumps(dict(rows=rows)))
    with pytest.raises(ValueError):
        report.load_route_results(tmp_path,'example',1)

def test_zero_containing_interval_does_not_claim_equivalence():
    assert report.statistical_verdict(-100.,100.)=='ZERO_INCLUDED_NO_EQUIVALENCE'
    assert report.statistical_verdict(1.,2.)=='POSITIVE_VALUE_DIFFERENCE'
