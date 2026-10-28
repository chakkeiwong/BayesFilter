"""Host recovery branch checks using the real budget API; no GPU claims."""
from types import SimpleNamespace
import pytest

from scripts import run_modern_hmc_macrofinance_frozen_diagnostic as frozen
from scripts import resume_modern_hmc_snapshot_diagnostic as recovery


@pytest.mark.parametrize('implementation',[frozen,recovery],ids=['existing-worker','external-host-worker'])
@pytest.mark.parametrize('enclosing_open',[True,False])
def test_actual_worker_resume_branch_uses_live_budget_context(tmp_path,monkeypatch,implementation,enclosing_open):
    # Do not mock execution_budget: the original defect occurred at its entry.
    import tensorflow as tf
    from bayesfilter.runtime.execution_budget import execution_budget,execution_budget_available
    from bayesfilter.runtime import gpu_provenance
    from scripts import resume_modern_hmc_macrofinance_public as resume
    from scripts import diagnose_modern_hmc_macrofinance_public as target
    from scripts import continue_modern_hmc_macrofinance_broad as broad

    campaign=tmp_path/'campaign';campaign.mkdir()
    frozen.write(campaign/'frozen-cohort.json',dict(qualification_seed=[101,1],reload_seed=[101,2],prior_price_vetoes=[]))
    parent=tmp_path/'parent';parent.mkdir()
    preserved=parent/'preserved.json';preserved.write_text('{}')
    inputs=tmp_path/'inputs.json';inputs.write_text('{}')
    source=tmp_path/'source.py';source.write_text('# immutable numerical source')
    output=tmp_path/'output';output.mkdir()
    monkeypatch.setenv('TF_FORCE_GPU_ALLOW_GROWTH','true')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','GPU-mock-no-device-work')
    monkeypatch.setattr(frozen,'INPUTS',inputs)
    monkeypatch.setattr(frozen,'source_paths',lambda _:([source],None))
    monkeypatch.setattr(frozen.subprocess,'check_output',lambda *a,**k:'test-only-commit\n')
    monkeypatch.setattr(gpu_provenance,'query_nvidia_smi_gpus',lambda:[])
    monkeypatch.setattr(gpu_provenance,'selected_nvidia_gpu',lambda _:dict(uuid='GPU-mock-no-device-work'))
    monkeypatch.setattr(target,'build',lambda *a,**k:(object(),{}))
    monkeypatch.setattr(tf.config.experimental,'get_memory_info',lambda _:dict(current=0,peak=0))
    runtime=SimpleNamespace(source=dict(git_commit='mock',source_sha256={str(source):recovery.digest(source)}),
        frozen=frozen,snapshot=SimpleNamespace(audit_imports=lambda _:{}))
    monkeypatch.setattr(recovery,'load_runtime',lambda _:runtime)
    calls=[]
    def copy(source_path,out,inputs_path):
        assert source_path==parent and inputs_path==inputs
        (out/'tuning').mkdir()
        frozen.write(out/'tuning/tuning_checkpoint.json',dict(result=dict(candidates=[])))
        frozen.write(out/'parent-evidence.json',{str(preserved):recovery.digest(preserved)})
        return dict(data_version='mock engineering fixture')
    def resume_call(checkpoint,adapter,out):
        # Cumulative old clock is replaced; enclosing live checks still apply.
        available=execution_budget_available(elapsed_seconds=100000,wall_time_limit_seconds=1)
        assert available is enclosing_open
        calls.append('resumed')
        return dict(completion_status='complete' if enclosing_open else 'partial_budget',
            final_status='complete' if enclosing_open else 'partial_budget',
            verified=[],candidate_states={},all_verified_members_exported=True)
    def reload(*args,**kwargs):
        calls.append('reload')
        assert kwargs['seed_root']==[101,2]
        return dict(verified=[],rows=[],all_verified_members_reloaded=True)
    monkeypatch.setattr(resume,'copy_parent',copy)
    monkeypatch.setattr(resume,'resume_checkpoint',resume_call)
    monkeypatch.setattr(broad,'reload',reload)
    args=SimpleNamespace(output=output,campaign=campaign,source=parent,wall_seconds=1200.,plan='test-plan')
    with execution_budget(check=lambda:enclosing_open):
        code=implementation.worker(args)
    report=frozen.read(output/'evidence/result.json')
    assert report.get('error') is None,report
    assert report['status']==('completed' if enclosing_open else 'resource_stopped')
    assert code==(0 if enclosing_open else 1)
    assert calls==(['resumed','reload'] if enclosing_open else ['resumed'])
    assert preserved.read_text()=='{}'


def test_external_worker_preserves_error_before_checkpoint(tmp_path,monkeypatch):
    monkeypatch.setenv('TF_FORCE_GPU_ALLOW_GROWTH','true')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','GPU-mock')
    def invalid(_):raise ValueError('frozen source changed before copy')
    monkeypatch.setattr(recovery,'load_runtime',invalid)
    output=tmp_path/'output';output.mkdir()
    args=SimpleNamespace(output=output,campaign=tmp_path,source=tmp_path/'parent',wall_seconds=1200.,plan='test-plan')
    assert recovery.worker(args)==1
    result=recovery.read(output/'evidence/result.json')
    assert result['status']=='error' and 'source changed before copy' in result['error']
    assert not (output/'evidence/tuning/tuning_checkpoint.json').exists()
