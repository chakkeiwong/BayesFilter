"""The release harness must actually execute its declared full search."""
from copy import deepcopy

import pytest

from bayesfilter.testing.acceptance_decision_models import validate_configuration
from bayesfilter.testing.acceptance_release_validation import full_search_configuration, PRIMARY_L


@pytest.mark.parametrize("case",["gaussian","lgssm_qr","nonlinear","funnel_residual"])
def test_full_search_configuration_preserves_declared_stages(case):
    config = full_search_configuration(case,seed=(20261002,2101),wall_seconds=300)
    policy,search = validate_configuration(config)
    assert search.primary_l_grid == PRIMARY_L
    assert search.max_candidates == policy.max_candidates == 100
    assert search.refinement_rounds == 1 and search.refinement_l_grid == (4,7)
    assert config['max_repairs_per_family'] == 4
    assert search.evidence_rungs == (1,2,4,8)


@pytest.mark.parametrize("field,value",[("epsilon_by_l",[[1,[1.35]]]),("evidence_rungs",[1,2]),
                                      ("wall_seconds",299)])
def test_runner_cannot_silently_override_frozen_search(field,value):
    config = full_search_configuration("gaussian",seed=(20261002,2101),wall_seconds=300)
    bad = deepcopy(config)
    bad[field] = value
    with pytest.raises(ValueError,match="disagrees"):
        validate_configuration(bad)


def test_optional_trial_batching_preserves_full_scientific_design():
    original=full_search_configuration('lgssm_qr',seed=(20261002,2101),wall_seconds=300)
    batched=full_search_configuration('lgssm_qr',seed=(20261002,2101),wall_seconds=300,
                                      replicated_trial_batch_size=32)
    validate_configuration(batched)
    for field in ('search','policy','seed','data','active_starts','geometry','epsilon_by_l'):
        assert batched[field]==original[field]
    assert batched['replicated_trial_batch_size']==32
    assert 'replicated_trial_batch_size' not in original
    batched['replicated_trial_batch_size']=True
    with pytest.raises(ValueError,match='batch size'):
        validate_configuration(batched)


def test_v7_windowed_preparation_reaches_the_same_public_controller(tmp_path):
    """Real preparation plus one trial tests wiring; it cannot claim delivery."""
    from bayesfilter.inference import HMCKernelTuningConfig, HMCCandidateExecutionConfig, tune_hmc_kernel
    from bayesfilter.inference.hmc_candidate_set_tuning import HMCControllerConfig
    from tests.test_hmc_candidate_set_execution import GaussianTarget
    from tests.test_hmc_acceptance_protocol import policy
    p = policy(base_repetitions=1,max_repetitions=1,max_candidates=100)
    search = HMCControllerConfig(primary_l_grid=PRIMARY_L,initial_epsilon=.8,
        pilot_enabled=True,refinement_rounds=1,refinement_l_grid=(4,7),
        max_candidates=100,total_budget_units=300,repair_reserve_units=20,
        evidence_rungs=(1,),replicated_acceptance_policy=p)
    execution = HMCCandidateExecutionConfig(measurement_num_results=65,
        verification_num_results=65,num_warmup_steps=3,acceptance_policy=p,seed=(20261002,2121),
        target_status_trace_policy="per_chain_step",use_xla=True,
        chain_mode="batched",reuse_leapfrog_graphs=True,chunk_max_results=68)
    run = tune_hmc_kernel(adapter=GaussianTarget(),initial_position=[.2,-.3],
        parameter_scales=[1.,1.],
        config=HMCKernelTuningConfig.standard(use_xla=True,
            target_scope="candidate-bridge-test",target_status_trace_policy="per_chain_step"),
        search_config=search,execution_config=execution,
        target_lineage={"model":"standard Gaussian","prior":"standard normal","data":"none"},
        source_paths=[__file__],output_dir=tmp_path,max_work_items=1)
    binding = run.adapter._execution_binding
    assert len(binding._transforms) == 2
    assert run.result.config.replicated_acceptance_policy == p
    assert run.result.config.primary_l_grid == PRIMARY_L
    assert run.result.work_items[0].stage == "pilot"
    assert len(run.result.observations) == 1
    assert not run.result.verified_candidate_ids
    evidence = next(iter(binding._evidence.values()))
    assert len(evidence['trials']) == 1
    assert evidence['trials'][0]['scores'] is not None


def test_release_status_cannot_relabel_existing_v7_numerical_identity():
    import hashlib
    import json
    from bayesfilter.inference.hmc_acceptance_protocol import HMCReplicatedAcceptancePolicy
    from tests.test_hmc_acceptance_protocol import policy
    p = policy()
    raw = json.loads(json.dumps(p.payload()))
    # The October 2 codec keeps this field in identity even if an external
    # capability record later describes a supported explicit release.
    assert raw['default_promotion_status'] == 'experimental_pending_validation'
    expected = hashlib.sha256(json.dumps(raw,sort_keys=True,separators=(',',':'),
                                        allow_nan=False).encode()).hexdigest()
    assert HMCReplicatedAcceptancePolicy.from_payload(raw).identity == expected == p.identity
    raw['default_promotion_status'] = 'released'
    with pytest.raises(ValueError,match='metadata'):
        HMCReplicatedAcceptancePolicy.from_payload(raw)


def repair_fixture():
    from bayesfilter.inference.hmc_candidate_set_tuning import HMCControllerConfig
    from tests.test_hmc_acceptance_protocol import policy
    from tests.test_hmc_candidate_set_execution import execution_config,make_binding
    p = policy(base_repetitions=32,max_repetitions=128,max_candidates=8)
    execution = execution_config(measurement_num_results=65,verification_num_results=65,
        num_warmup_steps=3,acceptance_policy=p,seed=(20261002,2141),use_xla=True,
        non_xla_reason=None,chain_mode='batched',reuse_leapfrog_graphs=True,chunk_max_results=68)
    binding = make_binding(config=execution,repair_factor=1.3,max_repairs_per_family=3)
    search = HMCControllerConfig(primary_l_grid=(1,),epsilon_by_l=((1,(1.35,1.8)),),
        max_candidates=8,total_budget_units=100,repair_reserve_units=20,
        evidence_rungs=(1,2,4),replicated_acceptance_policy=p)
    return binding,search


@pytest.mark.parametrize('boundary',['repair_measurement','fresh_verification'])
def test_v7_repair_and_verification_survive_fresh_process(boundary,tmp_path,monkeypatch):
    import json
    import os
    from pathlib import Path
    import subprocess
    import sys
    import bayesfilter.inference.hmc_acceptance_trials as trials
    from bayesfilter.inference import tune_hmc_kernel
    from bayesfilter.inference.hmc_candidate_set_tuning import HMCInfrastructureFailure
    from tests.test_hmc_acceptance_ssm_recovery import numerical_summary
    baseline,search = repair_fixture()
    complete = tune_hmc_kernel(adapter=baseline._base_adapter,
        initial_position=baseline.initial_active_state,config=search,
        candidate_set_adapter=baseline.typed_adapter)
    expected = numerical_summary(baseline,complete.result)
    assert complete.result.completion_status == 'complete'
    assert len(complete.result.verified_candidate_ids) >= 2
    assert any(a.qualified_repair_status == 'executed_and_verified'
               for a in complete.result.repair_actions)
    binding,_ = repair_fixture()
    original = trials.before_numerical_chunk
    interrupted = []
    def fail_after_durable_charge(runtime,work,candidate,**kwargs):
        original(runtime,work,candidate,**kwargs)
        match = (candidate.parent_candidate_id is not None and work.stage == 'measurement'
                 if boundary == 'repair_measurement' else work.stage == 'verification')
        if match and not interrupted:
            interrupted.append(kwargs['count']*kwargs['chains']*(candidate.leapfrog_steps+1))
            raise HMCInfrastructureFailure('injected after durable attempted charge')
    with monkeypatch.context() as patch:
        patch.setattr(trials,'before_numerical_chunk',fail_after_durable_charge)
        paused = tune_hmc_kernel(adapter=binding._base_adapter,
            initial_position=binding.initial_active_state,config=search,
            candidate_set_adapter=binding.typed_adapter,output_dir=tmp_path/'tuning')
    assert interrupted and paused.result.completion_status == 'paused_infrastructure'
    assert not getattr(binding,'_incremental_checkpoint',False)
    code = '''import json,sys
from pathlib import Path
from tests.test_hmc_acceptance_release_validation import repair_fixture
from tests.test_hmc_acceptance_ssm_recovery import numerical_summary
from bayesfilter.inference import resume_hmc_candidate_set_tuning
binding,_ = repair_fixture()
root = Path(sys.argv[1])
run = resume_hmc_candidate_set_tuning(root/'tuning/tuning_checkpoint.json',adapter=binding._base_adapter)
(root/'resumed.json').write_text(json.dumps(numerical_summary(run.adapter._execution_binding,run.result),allow_nan=False))
'''
    env = {k:os.environ[k] for k in ('PATH','LANG','LD_LIBRARY_PATH') if k in os.environ}
    env.update(CUDA_VISIBLE_DEVICES='-1',TF_FORCE_GPU_ALLOW_GROWTH='true',TF_NUM_INTRAOP_THREADS='2',
        TF_NUM_INTEROP_THREADS='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',
        TF_CPP_MIN_LOG_LEVEL='2',BAYESFILTER_PRELOAD_CUSTOM_OP='0')
    with (tmp_path/'child.log').open('w') as log:
        child = subprocess.run([sys.executable,'-c',code,str(tmp_path)],env=env,
            stdout=log,stderr=subprocess.STDOUT,timeout=180)
    assert child.returncode == 0,(tmp_path/'child.log').read_text()[-4000:]
    actual = json.loads((tmp_path/'resumed.json').read_text())
    expected = json.loads(json.dumps(expected))
    for key in ('states','trials','decisions'):
        assert actual[key] == expected[key]
    assert actual['accounting']['gradient_work'] == expected['accounting']['gradient_work']+interrupted[0]
    assert actual['accounting']['independent_trial_streams'] == expected['accounting']['independent_trial_streams']
