"""Recovery mechanics; CPU checks are not MacroFinance sampler evidence."""
from dataclasses import replace
import json
from pathlib import Path
import shutil

import pytest

from scripts.resume_modern_hmc_macrofinance_public import (
    copy_parent, digest, retry_allowance, resume_checkpoint,
)
from scripts.run_modern_hmc_filter_health_gpu import checkpoint_progress


def charge(ordinal=0, cost=3072):
    return dict(event='numerical_chunk_charged',work_item_id='work',trial_ordinal=ordinal,
        trial_chunk_index=0,gradient_work=cost,seed=[101,ordinal],transitions=256)


def test_retry_allowance_covers_only_missing_results():
    rows=[charge(0),charge(1),charge(1)]
    assert retry_allowance(rows,set())==0
    assert retry_allowance(rows,{('work',1,0)})==3072
    with pytest.raises(ValueError,match='unrecorded'):
        retry_allowance(rows,{('work',2,0)})
    with pytest.raises(ValueError,match='inconsistent'):
        retry_allowance([charge(),charge(cost=4000)],{('work',0,0)})


def test_progress_comes_from_checkpoint_despite_stale_stage(tmp_path):
    directory=tmp_path/'evidence/tuning';directory.mkdir(parents=True)
    payload=dict(result=dict(observations=[dict(observation=dict(acceptance_evidence=dict(completed_trials=32)))],
        candidate_states={'a':'verified'},completion_status='partial_budget'),partial_chunks={'w':['a','b']})
    (directory/'tuning_checkpoint.json').write_text(json.dumps(payload))
    result=checkpoint_progress(tmp_path)
    assert result['complete_work_trials']==32 and result['partial_chunk_count']==2
    assert result['candidate_states']=={'a':'verified'}
    assert checkpoint_progress(tmp_path/'missing') is None


def parent_fixture(tmp_path, monkeypatch):
    source = tmp_path/'prior/evidence'
    (source/'tuning').mkdir(parents=True)
    inputs = tmp_path/'inputs.json'; inputs.write_text('{}')
    numerical = tmp_path/'numerical.py'; numerical.write_text('# frozen source')
    manifest = dict(inputs_sha256=digest(inputs),
        source_sha256={str(numerical): digest(numerical)},
        physical_gpu=dict(uuid='GPU-1'))
    (source/'manifest.json').write_text(json.dumps(manifest))
    (source.parent/'manifest.json').write_text(json.dumps(
        dict(active_run=False, status='wall_budget_exhausted')))
    spec = dict(execution=dict(runtime_policy=dict(device_type='GPU',
        cuda_visible_devices='GPU-1'), config=dict(use_xla=True)))
    (source/'tuning/execution_spec.json').write_text(json.dumps(spec))
    (source/'tuning/tuning_checkpoint.json').write_text('{}')
    (source/'design.json').write_text('{}')
    output = tmp_path/'recovery'; output.mkdir()
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'GPU-1')
    return source, output, inputs, numerical


def test_copy_preserves_parent_and_frozen_execution(tmp_path, monkeypatch):
    source, output, inputs, _ = parent_fixture(tmp_path, monkeypatch)
    preserved = {p: p.read_bytes() for p in source.rglob('*') if p.is_file()}
    copy_parent(source, output, inputs)
    assert all(p.read_bytes() == value for p, value in preserved.items())
    assert (output/'tuning/execution_spec.json').read_bytes() == (source/'tuning/execution_spec.json').read_bytes()
    assert json.loads((output/'parent-evidence.json').read_text()) == {str(p): digest(p) for p in preserved}
    with pytest.raises(ValueError, match='outside'):
        copy_parent(source, source/'child', inputs)


@pytest.mark.parametrize('fault,match', [
    ('inputs', 'inputs differ'), ('source', 'source changed'),
    ('uuid', 'GPU/XLA'), ('xla', 'GPU/XLA'), ('cpu', 'GPU/XLA'),
    ('active', 'terminal'), ('status', 'terminal'),
])
def test_parent_mismatch_refused_before_copy(tmp_path, monkeypatch, fault, match):
    source, output, inputs, numerical = parent_fixture(tmp_path, monkeypatch)
    if fault == 'inputs':
        inputs.write_text('{"changed": true}')
    elif fault == 'source':
        numerical.write_text('# different source')
    elif fault == 'uuid':
        monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'GPU-0')
    elif fault in ('xla', 'cpu'):
        path = source/'tuning/execution_spec.json'
        spec = json.loads(path.read_text())
        if fault == 'xla': spec['execution']['config']['use_xla'] = False
        else: spec['execution']['runtime_policy']['device_type'] = 'CPU'
        path.write_text(json.dumps(spec))
    else:
        path = source.parent/'manifest.json'
        receipt = json.loads(path.read_text())
        if fault == 'active': receipt['active_run'] = True
        else: receipt['status'] = 'completed'
        path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match=match):
        copy_parent(source, output, inputs)
    assert not (output/'tuning').exists()


def test_interrupted_trial_resumes_once_with_derived_allowance(tmp_path,monkeypatch):
    from tests.test_modern_hmc_pipeline import setup
    from bayesfilter.inference.hmc_candidate_set_adapters import run_typed_hmc_candidate_set
    from bayesfilter.inference.hmc_acceptance_protocol import replicated_evidence_preflight
    from bayesfilter.inference.hmc_candidate_set_tuning import HMCInfrastructureFailure
    from bayesfilter.runtime.execution_budget import execution_budget

    binding,search=setup()
    quote=replicated_evidence_preflight(search.replicated_acceptance_policy,
        evidence_rungs=search.evidence_rungs,candidate_cap=2,leapfrog_steps=(3,5),
        transition_profile=binding.config.transition_profile)
    search=replace(search,max_gradient_work=quote['initial_cohort_mandatory_gradient_work'])
    original=binding._run;calls=[]
    def interrupted(candidate,state,count,seed):
        calls.append(seed)
        if len(calls)==2:
            raise HMCInfrastructureFailure('test interruption before a durable result')
        return original(candidate,state,count,seed)
    monkeypatch.setattr(binding,'_run',interrupted)
    parent=tmp_path/'parent'
    stopped=run_typed_hmc_candidate_set(binding.typed_adapter,search,output_dir=parent)
    assert stopped.result.completion_status=='paused_infrastructure'
    preserved={p:p.read_bytes() for p in parent.rglob('*') if p.is_file()}
    output=tmp_path/'recovery';output.mkdir()
    shutil.copytree(parent,output/'tuning')
    with execution_budget(check=lambda:True,replace_wall_time_limit=True):
        result=resume_checkpoint(output/'tuning/tuning_checkpoint.json',binding._base_adapter,output)
    assert result['completion_status']=='complete'
    assert len(result['verified'])==2
    assert result['additional_gradient_work']==2048
    assert result['native_audit']['interrupted_attempt_count']==1
    assert result['native_audit']['returned_chunk_count']==64
    assert result['native_audit']['attempted_chunk_count']==65
    assert all(p.read_bytes()==content for p,content in preserved.items())
