"""Repeated exact-reference handoff checks, not SBC or coverage calibration.

The smaller explicit posterior budgets are CPU reference settings. They do not
replace the production sequential warmup defaults or prove universal burn-in.
"""
import json

import pytest
import tensorflow as tf

from bayesfilter.inference import load_hmc_candidate_retained_runner
from bayesfilter.inference.hmc_posterior_assessment import HMCPosteriorAssessmentPolicy
from bayesfilter.inference.hmc_precision import mean_precision
from bayesfilter.inference.neutra_hmc import SequentialNeuTraHMCConfig
from bayesfilter.testing.acceptance_decision_models import run_model
from bayesfilter.testing.acceptance_validation_inventory import model_configuration
from bayesfilter.testing.inference_validation.targets import ValidationTarget
from bayesfilter.testing.inference_validation.references.ssm import location_posterior
from tests.test_hmc_acceptance_protocol import policy


@pytest.mark.parametrize('replication',[0,1,2])
def test_checked_v7_member_runs_sequential_warmup_and_exact_location_reference(tmp_path,replication):
    # Original panels/seeds are preserved. The release profile makes the
    # chance endpoint return reporting-only, keeping actual movement vetoes.
    p = policy(base_repetitions=32,max_repetitions=128,max_candidates=2,
               min_normalized_return_displacement=0.)
    cfg = model_configuration('k0',policy_payload=p.payload(),evidence_rungs=(1,2,4),
        seed=(20261002,2200+replication),wall_seconds=240,
        classification='regression',expected_outcome='positive_delivery')
    cfg['data'] = [v+.3*replication for v in cfg['data']]
    cfg['provenance'] += '; three predetermined shifted observation panels and independent tuning/posterior streams; exact-reference diagnostic, no coverage claim'
    root = tmp_path/'model'
    result = run_model(cfg,root)
    assert result['expectation_met'] is True
    tuning_before = (root/'result.json').read_bytes()
    target = ValidationTarget(cfg['target'],cfg['parameters'],cfg['data'])
    member = load_hmc_candidate_retained_runner(next(root.glob('*-member.json')),adapter=target)
    config = SequentialNeuTraHMCConfig(step_size=member.step_size,
        num_leapfrog_steps=member.num_leapfrog_steps,
        warmup_seed=(20261002,2300+replication),retained_seed=(20261002,2400+replication),
        jit_compile=True,warmup_chunk_results=512,warmup_min_results=512,
        warmup_check_window_results=512,warmup_max_results=2048,
        retained_chunk_results=1024,retained_min_results=1024,retained_max_results=4096,
        assessment_policy=HMCPosteriorAssessmentPolicy(retained_bulk_ess_min=128.,
                                                       retained_tail_ess_min=128.))
    archives = []
    def archive(**row):
        path = tmp_path/(row['stage']+'-'+str(row['chunk_index'])+'-'+str(row['cumulative'])+'.tensor')
        tf.io.write_file(str(path),tf.io.serialize_tensor(row['model_samples']))
        receipt = {'stage':row['stage'],'cumulative':row['cumulative'],
                   'draws':int(row['model_samples'].shape[0]),'path':str(path)}
        archives.append(receipt)
        return receipt
    posterior = member.run_sequential(config=config,parameter_names=('location',),archive_callback=archive)
    assert posterior['passed'], {k:v for k,v in posterior.items() if not k.startswith('private_')}
    assert posterior['warmup_excluded_from_posterior'] is True
    samples = posterior['private_retained_raw']
    assert int(samples.shape[0]) == posterior['retained_results_per_chain']
    assert sum(r['draws'] for r in archives if r['stage']=='warmup' and not r['cumulative']) == posterior['warmup_results_per_chain']
    assert sum(r['draws'] for r in archives if r['stage']=='retained' and not r['cumulative']) == int(samples.shape[0])
    mean,variance = location_posterior(cfg['data'])
    quantities = tf.concat([samples,tf.square(samples-mean)],axis=-1)
    precision = mean_precision(quantities)
    errors = tf.abs(precision['estimate']-tf.constant([mean,variance],tf.float64))
    tf.debugging.assert_all_finite(precision['mcse'],'missing reference precision')
    tf.debugging.assert_positive(precision['mcse'])
    assert bool(tf.reduce_all(errors <= 4.*precision['mcse']))
    (tmp_path/'reference.json').write_text(json.dumps({
        'reference':[mean,variance],'estimate':precision['estimate'].numpy().tolist(),
        'mcse':precision['mcse'].numpy().tolist(),'absolute_error':errors.numpy().tolist(),
        'warmup_draws':posterior['warmup_results_per_chain'],'retained_draws':int(samples.shape[0]),
        'criterion':'four estimated MCSE, diagnostic only','coverage_or_sbc':'not_established'},indent=2)+'\n')
    assert (root/'result.json').read_bytes() == tuning_before
    assert result['verified_candidate_ids']
