"""CPU-only execution/accounting regression; no scientific accuracy evidence."""
import importlib.util
import json
from pathlib import Path
from datetime import datetime,timezone,timedelta
import pytest


def runner():
    path=Path(__file__).parents[2]/'docs/benchmarks/run_sqmc_expanded_comparison.py'
    spec=importlib.util.spec_from_file_location('expanded_control_test_runner',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_aggregate_budget_cannot_be_reset_by_new_attempt(tmp_path):
    module=runner()
    deadline=(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat()
    with pytest.raises(ValueError,match='aggregate'):
        module.supervisor(tmp_path/'new',43200,deadline,1918)
    assert not (tmp_path/'new').exists()


def test_expired_elapsed_window_cannot_launch(tmp_path):
    module=runner()
    with pytest.raises(ValueError,match='expired'):
        module.supervisor(tmp_path/'new',100,'2000-01-01T00:00:00+00:00',1918)
    assert not (tmp_path/'new').exists()


def test_reuse_verifies_saved_data_and_keeps_failed_launch_count(tmp_path):
    module=runner()
    unit='p44_d3_T10__iid_dual_cap'
    directory=tmp_path/unit
    directory.mkdir()
    case=dict(scope='p44_d3_T10',route='iid_dual_cap',status='complete',parameter_count=4,
              rows=[dict(data_seed=d,filter_seed=f,score=[1.,2.,3.,4.]) for d,f in module.final_pairs()])
    for name,value in [('result.json',case),('tuning.json',{}),('data.json',{}),('random_designs.json',{})]:
        module.dump(directory/name,value)
    sources={'bayesfilter/example.py':'abc','docs/plan.md':'old'}
    record=dict(status='complete',comparison_contract=module.comparison_contract(),source_sha256=sources,
        result_sha256=module.digest(directory/'result.json'),tuning_sha256=module.digest(directory/'tuning.json'),
        random_designs_sha256=module.digest(directory/'random_designs.json'),
        data_version=dict(sha256=module.digest(directory/'data.json')))
    module.dump(directory/'manifest.json',record)
    module.dump(tmp_path/'manifest.json',dict(units=[dict(unit=unit,status='complete')]))
    reused,launches=module.reusable_units([tmp_path],dict(sources,**{'docs/plan.md':'administrative repair'}))
    assert reused[unit]['rows']==case['rows']
    assert launches[unit]==2  # historical failed launch plus completed current launch
    (directory/'data.json').write_text('{"changed":true}')
    with pytest.raises(ValueError,match='checksum'):
        module.reusable_units([tmp_path],sources)


def test_invalid_coordinate_evidence_is_written_without_admitting_it(tmp_path):
    module=runner()
    case=dict(scope='p44_d3_T10',route='iid_dual_cap',dimension=3,parameter_count=4,
        parameter_names=['a','q','r','m'],program='fp64_reference',tuning_status='frozen',
        rows=[dict(valid=False,data_seed=197001,filter_seed=198001,score=None,
             raw_score=[1.,'NaN',2.,3.],oracle_score=[1.,2.,3.,4.])])
    module.assemble(tmp_path,[case])
    import csv
    with (tmp_path/'scores.csv').open() as stream:
        rows=list(csv.DictReader(stream))
    assert len(rows)==4
    assert all(r['valid']=='False' for r in rows)
    assert rows[1]['estimated_score']=='NaN' and rows[1]['absolute_error']==''
    assert all(v is None for v in case['component_rmse'].values())
