"""Explicit posterior budgets preserve defaults, checks, streams and replay."""
from dataclasses import replace

import pytest
import tensorflow as tf

from bayesfilter.inference.neutra_hmc import (
    SequentialExactTransitionConfig, SequentialNeuTraHMCConfig,
    run_sequential_exact_transition,
)
from bayesfilter.testing.inference_validation.engines.controller_stopping import GaussianAR1Transition


def config(kind, **kwargs):
    common = dict(warmup_seed=(317, 451), retained_seed=(727, 823), **kwargs)
    if kind == 'exact':
        return SequentialExactTransitionConfig('diagnostic_gaussian_ar1:0.0:4', **common)
    return SequentialNeuTraHMCConfig(step_size=1.3, num_leapfrog_steps=3,
                                    jit_compile=False, **common)


@pytest.mark.parametrize('kind', ['exact', 'shared'])
def test_default_limit_and_historical_payload_are_preserved(kind):
    cfg = config(kind)
    assert cfg.max_results_per_chain == cfg.warmup_max_results == cfg.retained_max_results == 10000
    assert 'max_results_per_chain' not in cfg.payload()
    assert 'count_budget_reason' not in cfg.payload()
    assert 'count_budget_policy' not in cfg.payload()
    for field in ('warmup_max_results', 'retained_max_results'):
        with pytest.raises(ValueError, match='must not exceed 10000'):
            replace(cfg, **{field: 10001})
    assert replace(cfg, max_results_per_chain=10000).payload() == cfg.payload()


@pytest.mark.parametrize('kind', ['exact', 'shared'])
@pytest.mark.parametrize('options', [
    {'max_results_per_chain': True}, {'max_results_per_chain': 0},
    {'max_results_per_chain': 12000.5}, {'max_results_per_chain': 12000},
    {'max_results_per_chain': 12000, 'count_budget_reason': ''},
    {'max_results_per_chain': 12000, 'count_budget_reason': '  '},
    {'max_results_per_chain': 12000, 'count_budget_reason': 42},
    {'max_results_per_chain': 12000, 'count_budget_reason': 'test allocation', 'retained_max_results': 12001},
])
def test_invalid_or_unexplained_count_budgets_fail(kind, options):
    with pytest.raises(ValueError):
        config(kind, **options)


@pytest.mark.parametrize('kind', ['exact', 'shared'])
def test_nondefault_count_budget_changes_serialized_identity(kind):
    from bayesfilter.runtime.durable_tensor_checkpoint import payload_hash
    cfg = config(kind)
    longer = replace(cfg, max_results_per_chain=12000, count_budget_reason='derived precision allocation',
                     retained_max_results=12000)
    assert longer.payload()['max_results_per_chain'] == 12000
    assert longer.payload()['count_budget_reason'] == 'derived precision allocation'
    assert longer.payload()['count_budget_policy'] == 'explicit_nondefault_posterior_allocation'
    assert payload_hash(cfg.payload()) != payload_hash(longer.payload())
    assert cfg.warmup_rhat_max == longer.warmup_rhat_max
    assert cfg.retained_rhat_max == longer.retained_rhat_max
    assert cfg.warmup_seed == longer.warmup_seed and cfg.retained_seed == longer.retained_seed


def test_actual_exact_controller_crosses_old_cap_with_unchanged_draws():
    transition = GaussianAR1Transition(0., jit_compile=False)
    cfg = config('exact', warmup_chunk_results=6000, warmup_min_results=12000,
        warmup_check_window_results=6000, warmup_max_results=12000,
        retained_chunk_results=6000, retained_min_results=12000, retained_max_results=12000,
        max_results_per_chain=12000, count_budget_reason='CPU independent-normal mechanics check')
    calls = []
    def archive(**values):
        calls.append(values)
        return {'stage': values['stage']}
    result = run_sequential_exact_transition(transition_program=transition,
        initial_transition_state=tf.zeros((4,1), tf.float64), posterior_state_fn=lambda x:x,
        parameter_names=('x',), config=cfg, archive_callback=archive)
    assert result['passed'] and not result['hard_vetoes']
    assert result['warmup_results_per_chain'] == result['retained_results_per_chain'] == 12000
    for stage in ('warmup', 'retained'):
        observed = tf.concat([v['posterior_samples'] for v in calls
                              if v['stage'] == stage and not v['cumulative']], axis=0)
        tf.debugging.assert_equal(observed, result['private_'+stage+'_beta_one'])
    assert result['warmup_excluded_from_posterior']
    seeds = [tuple(v['seed']) for v in calls if not v['cumulative']]
    assert len(seeds) == len(set(seeds)) == 4


def test_actual_verified_member_exceeds_old_cap_and_replays(tmp_path):
    from tests.test_hmc_candidate_set_execution import make_binding
    from tests.test_hmc_whole_procedure_repair import _config
    from bayesfilter.inference import (run_typed_hmc_candidate_set,
        build_retained_bound_hmc_archive_runner_from_candidate_set_result, run_hmc_posterior)
    from bayesfilter.runtime.durable_tensor_checkpoint import DurableTensorCheckpoint
    binding = make_binding()
    run = run_typed_hmc_candidate_set(binding.typed_adapter,
        _config(grid=(3,), epsilons=((3,(1.3,)),)))
    member = build_retained_bound_hmc_archive_runner_from_candidate_set_result(
        candidate_set_result=run.result, candidate_id=run.result.verified_candidate_ids[0],
        retained_binding=binding)
    cfg = config('shared', warmup_chunk_results=6000, warmup_min_results=12000,
        warmup_check_window_results=6000, warmup_max_results=12000,
        retained_chunk_results=6000, retained_min_results=12000, retained_max_results=12000,
        max_results_per_chain=12000, count_budget_reason='public member checkpoint integration')
    assert cfg.step_size == member.step_size and cfg.num_leapfrog_steps == member.num_leapfrog_steps
    identity = {'member': member.member_hash, 'policy': cfg.payload()}
    with DurableTensorCheckpoint(tmp_path/'chunks', identity) as store:
        first = run_hmc_posterior(member=member, config=cfg, parameter_names=('x','y'), checkpoint_store=store)
    with DurableTensorCheckpoint(tmp_path/'chunks', identity) as store:
        second = run_hmc_posterior(member=member, config=cfg, parameter_names=('x','y'), checkpoint_store=store)
        assert store.records and all(r['replayed'] for r in store.records)
    assert first['warmup_results_per_chain'] == first['retained_results_per_chain'] == 12000
    tf.debugging.assert_equal(first['private_retained_raw'], second['private_retained_raw'])
    assert first['warmup_excluded_from_posterior'] and not first['hard_vetoes']
    assert member.candidate.candidate_id in run.result.verified_candidate_ids
