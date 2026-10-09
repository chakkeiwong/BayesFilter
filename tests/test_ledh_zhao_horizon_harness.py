"""Diagnostic tests of campaign wiring and exact-prefix reference identity."""
import json
from pathlib import Path
from types import SimpleNamespace
import hashlib
import numpy as np
import pytest
import tensorflow as tf
from scipy.io import savemat
from docs.benchmarks import run_ledh_nonlinear_master as master
from docs.benchmarks.run_zhao_cui_quadratic_score_reference import load_proposal
from bayesfilter.highdim.sqmc_campaign_tf import numerical_settings


def test_master_forwards_both_policies_to_shared_settings(capsys):
    args=master.parser().parse_args(['plan','--models','predator_prey','--horizons','10',
        '--importance-weight-policies','ancestor','marginal_mixture','--routes','iid_dual_cap',
        '--data-seeds','26100611','--design-seeds','261006101'])
    assert master.controller(args)==0
    jobs=json.loads(capsys.readouterr().out)['jobs']
    assert {numerical_settings(job['controls'])['importance_weight_policy'] for job in jobs} == {'ancestor','marginal_mixture'}
    with pytest.raises(ValueError,match='importance weight'):
        numerical_settings(dict(jobs[0]['controls'],importance_weight_policy='typo'))


def proposal_fixture(tmp_path):
    spec=SimpleNamespace(name='predator_prey',target_id='fixture',dimension=2)
    dataset=dict(observations=[[1.,2.],[3.,4.],[5.,6.]],dtype='float64',target_id='fixture')
    tensor=tf.constant(dataset['observations'],tf.float64)
    dataset['observation_sha256']=hashlib.sha256(bytes(tf.io.serialize_tensor(tensor).numpy())).hexdigest()
    for name,value in [('manifest.json',dict(settings=dict(profile='current_target',route='linear',model='pp',horizon=3,fit_seed=2,smooth_seed=3,rank=20))),
                       ('result.json',dict(status='complete',source_tree_unchanged=True)),('pp-input-dataset.json',dataset)]:
        (tmp_path/name).write_text(json.dumps(value))
    (tmp_path/'target-parity-errors.csv').write_text('0,0,0\n')
    savemat(tmp_path/'smoothing-t02.mat',dict(thetas=np.empty((0,4)),sams=np.zeros((2,4,3)),
           raw_log_weight=np.zeros((4,1)),proposal_history=np.zeros((4,2))))
    return spec,dataset


def test_prefix_uses_matching_checkpoint_and_hash(tmp_path):
    spec,dataset=proposal_fixture(tmp_path)
    paths,raw,history,observations,record=load_proposal(tmp_path,spec,2)
    np.testing.assert_equal(observations,dataset['observations'][:2])
    assert paths.shape==(4,3,2)
    assert record['source_observation_sha256']==dataset['observation_sha256']
    assert record['observation_sha256']!=record['source_observation_sha256']
    assert record['horizon']==2 and record['source_horizon']==3
    with pytest.raises((ValueError, LookupError)):
        load_proposal(tmp_path,spec,4)


def test_prefix_rejects_corrupt_full_source_data(tmp_path):
    spec,dataset=proposal_fixture(tmp_path)
    dataset['observations'][-1][0]=999.
    (tmp_path/'pp-input-dataset.json').write_text(json.dumps(dataset))
    with pytest.raises(ValueError,match='observation hash'):
        load_proposal(tmp_path,spec,2)


def test_completed_prefix_survives_later_timeout(tmp_path):
    spec,_=proposal_fixture(tmp_path)
    (tmp_path/'result.json').write_text(json.dumps(dict(status='timeout',source_tree_unchanged=True)))
    with pytest.raises(LookupError,match='completion row'):
        load_proposal(tmp_path,spec,2)
    (tmp_path/'smoothing-summary.csv').write_text('time,ess,finite_fraction\n2,3.9,1\n')
    *_,record=load_proposal(tmp_path,spec,2)
    assert record['parent_status']=='timeout' and record['horizon']==2
    (tmp_path/'result.json').write_text(json.dumps(dict(status='failed',source_tree_unchanged=False)))
    with pytest.raises(ValueError,match='source changed'):
        load_proposal(tmp_path,spec,2)


@pytest.mark.parametrize('ess,fraction',[('nan','1'),('3.9','nan'),('inf','1'),('3.9','0')])
def test_recovered_prefix_rejects_invalid_completion_diagnostics(tmp_path,ess,fraction):
    spec,_=proposal_fixture(tmp_path)
    (tmp_path/'result.json').write_text(json.dumps(dict(status='timeout',source_tree_unchanged=True)))
    (tmp_path/'smoothing-summary.csv').write_text(f'time,ess,finite_fraction\n2,{ess},{fraction}\n')
    with pytest.raises(LookupError,match='completion row'):
        load_proposal(tmp_path,spec,2)


def test_budget_charges_failed_jobs_auxiliary_checks_and_active_elapsed(tmp_path,monkeypatch):
    from docs.benchmarks import run_ledh_zhao_horizons as campaign
    import datetime as dt
    monkeypatch.setattr(campaign,'OUT',tmp_path)
    check=tmp_path/'score-check-001';check.mkdir()
    (check/'supervisor.json').write_text(json.dumps(dict(wall_seconds=7.5)))
    start=(dt.datetime.now(dt.timezone.utc)-dt.timedelta(seconds=10)).isoformat()
    rows=[dict(status='failed',wall_seconds=20.,limit_seconds=1200),
          dict(status='complete',wall_seconds=5.,limit_seconds=1200),
          dict(status='running',started_utc=start,limit_seconds=100)]
    budget=campaign.budget_accounting(rows)
    assert budget['completed_job_seconds']==32.5
    assert 42.5<=budget['used_including_running_job_seconds']<44.5
    assert budget['reserved_including_running_caps']==132.5


def test_recovered_radii_are_one_fit_and_require_prespecified_small_radius():
    from docs.benchmarks.summarize_ledh_zhao_horizons import selected_author_scores
    common=dict(model='sir_d18',horizon=40,method='zhao_cui_rank20',proposal=dict(directory='same-fit'),source='fixture')
    large=dict(radius=.000625,score=[1.,2.,3.])
    small=dict(radius=.00015625,score=[1.1,2.1,3.1])
    assert selected_author_scores([dict(common,estimates=[large])])==[]
    rows=[dict(common,estimates=[large]),dict(common,estimates=[small])]
    selected=selected_author_scores(rows+rows)
    assert len(selected)==1 and selected[0]['estimate']==small
    with pytest.raises(ValueError,match='conflicting repeated score'):
        selected_author_scores(rows+[dict(common,estimates=[dict(small,score=[9.,2.,3.])])])


def test_launch_accounting_includes_separate_score_repair_ledger(tmp_path,monkeypatch):
    from docs.benchmarks import run_ledh_zhao_horizons as campaign
    import datetime as dt
    monkeypatch.setattr(campaign,'OUT',tmp_path)
    (tmp_path/'attempts.json').write_text(json.dumps([
        dict(status='complete',wall_seconds=20.,limit_seconds=100)]))
    (tmp_path/'score-repair-attempts.json').write_text(json.dumps([
        dict(status='failed',wall_seconds=7.,limit_seconds=10),
        dict(status='running',started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),limit_seconds=100)]))
    assert campaign.budget_accounting()['reserved_including_running_caps']==127.


@pytest.mark.parametrize('linear_output',[False,True])
def test_generated_checkpoint_writer_accepts_both_author_return_formats(tmp_path,linear_output):
    """Execute the generated writer; pre_sol has no legacy-lml diagnostic."""
    import os
    import shutil
    import subprocess
    from scipy.io import loadmat
    from docs.benchmarks.run_zhao_cui_publication_replication import prepare, SOURCE
    octave=shutil.which('octave-cli')
    if octave is None: pytest.skip('Octave is required for the generated writer check')
    derived=prepare(tmp_path,'author_driver')
    (tmp_path/'smooth.m').write_text("""function [thetas,sams,w,history,lml,stats]=smooth(sol,N,T)
    thetas=[]; sams=reshape(1:(2*N*(T+1)),2,N,T+1);
    w=ones(1,N)/N;history=zeros(N,T);
    stats=struct('raw_log_weight',zeros(1,N),'finite_fraction',1);
    lml=NaN;
    if sol.linear_output, lml=1.25;stats.legacy_mean_log_weight=-2.5; end
    end
    """)
    source=(derived/'models').as_posix().replace("'","''")
    compat=(SOURCE/'octave_compat').as_posix().replace("'","''")
    script=f"addpath('{compat}'); addpath('{source}'); sol=struct('model',struct('m',2),'linear_output',{int(linear_output)}); reference_save_smoothing(sol,1);"
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='-1',BAYESFILTER_SMOOTH_REPETITIONS='1',
        BAYESFILTER_REPEATED_TIMES='1',BAYESFILTER_SMOOTH_SEED='17',BAYESFILTER_SMOOTH_N='4')
    completed=subprocess.run([octave,'--quiet','--no-gui','--eval',script],cwd=tmp_path,
        env=env,capture_output=True,text=True,timeout=45)
    assert completed.returncode==0,completed.stderr
    saved=loadmat(tmp_path/'smoothing-t01.mat')
    assert bool(saved['legacy_mean_log_weight_available'].item())==linear_output
    if linear_output:
        assert saved['legacy_mean_log_weight'].item()==-2.5 and saved['lml'].item()==1.25
    else:
        assert np.isnan(saved['legacy_mean_log_weight'].item()) and np.isnan(saved['lml'].item())
    np.testing.assert_array_equal(saved['w'],np.full((1,4),.25))
    assert (tmp_path/'smoothing-summary.csv').read_text().startswith('1,4,1,0.25,1,')
