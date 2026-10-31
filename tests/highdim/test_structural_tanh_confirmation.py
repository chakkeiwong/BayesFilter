"""Focused restart, freshness and consumer-wiring checks for Model B."""
import importlib.util
from pathlib import Path
import sys

import pytest


def module():
    scripts = Path(__file__).resolve().parents[2]/'scripts'
    sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location('tanh_confirmation', scripts/'run_structural_tanh_confirmation.py')
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_budget_resume_and_controller_source_are_bound(tmp_path):
    m = module()
    campaign = m.Confirmation(tmp_path/'run', 12)
    campaign.state['attempts'].append(dict(key='failed', gpu=True, charged_seconds=321))
    campaign.persist()
    resumed = m.Confirmation(tmp_path/'run')
    assert resumed.remaining(True) == 12*3600-321
    assert resumed.state['plan'] == m.PLAN
    assert 'scripts/run_structural_tanh_confirmation.py' in resumed.state['source_identity']
    assert m.SEED_OFFSET != 0
    with pytest.raises(ValueError):
        m.Confirmation(tmp_path/'excess', 13)


def test_controller_dispatches_actual_shared_workers_with_fresh_target(tmp_path, monkeypatch):
    m = module()
    campaign = m.Confirmation(tmp_path/'run', 12)
    calls = []
    statuses = dict(prepare='PREPARED', references='REFERENCES_COMPUTED',
                    probe='PROBE_VALID', training_setup='TRAINING_SETUP_VALID', integration='INTEGRATION_VALID')
    def task(key, kind, allowance, gpu=True, **kwargs):
        calls.append(dict(key=key, kind=kind, gpu=gpu, **kwargs))
        assert kind in m.base.WORKERS
        return dict(status=statuses[kind]), tmp_path
    monkeypatch.setattr(campaign, 'task', task)
    campaign.run('pilot')
    assert calls[0]['seed_offset'] == m.SEED_OFFSET
    assert calls[0]['scopes'] == [('model_b', 2), ('model_b', 50)]
    assert not calls[0]['gpu'] and not calls[1]['gpu']
    for call in calls[2:]:
        assert call['model'] == 'model_b'
        assert call['reset_precision_policy'] == 'conditional_fp64'
    assert campaign.state['decision'] == 'PILOT_COMPLETE_CAMPAIGN_PENDING'


def test_near_zero_reference_score_is_not_percentage_division():
    m = module()
    reference = dict(estimates=[dict(q=[0, 0, 0], mean=[-2., 0., 0., 0.],
        mcse=[.01]*4, refinement=[.001]*4, kind='replicated')],
        heuristics=[dict(region='truth')])
    result = m.compare_probe(dict(value=[-2.], score=[[.005, 0., 0.]]), reference)[0]
    assert result['signed_error'] == [0., .005, 0., 0.]
    assert not any(result['outside_uncertainty_plus_one_percent'])
    assert result['heuristic_dominance_verdict'].startswith('UNRESOLVED')
    assert not result['statistically_supported_ranking']


def test_budget_exhaustion_is_never_sampling_completion(tmp_path, monkeypatch):
    m = module()
    campaign = m.Confirmation(tmp_path/'run', 12)
    monkeypatch.setattr(campaign, 'task', lambda *a, **kw:(dict(status='BUDGET_EXHAUSTED'), None))
    campaign.run()
    assert campaign.state['decision'] == 'PREPARATION_REPAIR_REQUIRED'
    assert 'COMPLETE' not in campaign.state['decision']


def test_generated_model_matches_owner_equations_and_initial_variances():
    import numpy as np
    import tensorflow as tf
    from bayesfilter.testing.structural_ledh_models_tf import StructuralModelSpec, make_structural_data
    spec = StructuralModelSpec('model_b')
    data = make_structural_data(spec, 10, 26302450, tf.float64)
    initial = tf.random.stateless_normal([2], [26302450, 1], dtype=tf.float64).numpy()
    initial *= np.sqrt([.25, .20])
    noise = tf.random.stateless_normal([10, 2], [26302450, 2], dtype=tf.float64).numpy()
    state = initial.copy()
    states, obs = [], []
    for innovation, measurement in noise:
        m = .70*state[0]+.25*innovation
        k = .55*state[1]+.80*np.tanh(m)
        state = np.array([m, k])
        states.append(state.copy())
        obs.append([m+k+.30*measurement])
    np.testing.assert_allclose(data['initial_state'], initial, rtol=0, atol=1e-14)
    np.testing.assert_allclose(data['states'], states, rtol=0, atol=1e-14)
    np.testing.assert_allclose(data['observations'], obs, rtol=0, atol=1e-14)


def test_production_probe_reports_gpu_arrays_without_eager_gpu_arithmetic(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import numpy as np
    m = module()
    class DeviceArray:
        def __init__(self, value): self.value = np.asarray(value)
        def numpy(self): return self.value
    result = {k:DeviceArray(v) for k,v in dict(value=[-1.]*4, score=[[0.]*3]*4,
        valid=[True]*4, steps_completed=[2]*4, reset_precision_fallback=[[False, True]]*4).items()}
    target = SimpleNamespace(horizon=2, target_signature='checked', observations=None,
        design=dict(initial=None, innovations=None), program=lambda *a:result)
    monkeypatch.setattr(m.base, 'target_for', lambda *a:target)
    data, _ = m.base.prepared_paths(tmp_path, 'model_b', 2, 'calibration')
    m.base.save(data.with_name(data.stem+'-probes.json'), [[0.]*3]*4)
    tf = SimpleNamespace(constant=lambda x,d:x, float32='float32',
        config=SimpleNamespace(experimental=SimpleNamespace(get_memory_info=lambda d:dict(current=0, peak=0))))
    row = m.base.probe(tf, dict(root=str(tmp_path), model='model_b', horizon=2, controls={}), tmp_path)
    assert row['status'] == 'PROBE_VALID'
    assert row['fixed_target_repeatable']
    assert row['rows'][0]['reset_precision_fallback_count'] == [1]*4


@pytest.mark.parametrize('sampling_complete', [True, False])
def test_full_orchestration_carries_selected_target_through_training_and_hmc(tmp_path, monkeypatch, sampling_complete):
    m = module()
    campaign = m.Confirmation(tmp_path/'run', 12)
    calls = []
    def task(key, kind, allowance, gpu=True, **kwargs):
        calls.append(dict(key=key, kind=kind, **kwargs))
        out = campaign.root/'fake-workers'/key
        out.mkdir(parents=True, exist_ok=True)
        if kind == 'prepare':
            for model, horizon in m.SCOPES:
                for partition in m.base.PARTITIONS:
                    data, design = m.base.prepared_paths(campaign.root, model, horizon, partition)
                    m.base.save(data, dict(model=model, horizon=horizon, partition=partition))
                    m.base.save(design, dict(fresh=True))
            result = dict(status='PREPARED')
        elif kind == 'references':
            for model, horizon in m.SCOPES:
                for partition in m.base.PARTITIONS:
                    reference = dict(estimates=[dict(q=[0]*3, mean=[-1., 0., 0., 0.],
                        mcse=[0.]*4, refinement=[0.]*4, kind='test')]*4,
                        heuristics=[dict(region=r) for r in ('truth','positive_local','negative_local','prior_center')])
                    m.base.save(campaign.root/'references'/f'{model}-T{horizon}-{partition}.json', reference)
            result = dict(status='REFERENCES_COMPUTED')
        elif kind == 'probe':
            result = dict(status='PROBE_VALID', rows=[dict(value=[-1.]*4,score=[[0.]*3]*4)])
        elif kind in ('training_setup', 'integration'):
            result = dict(status={'training_setup':'TRAINING_SETUP_VALID','integration':'INTEGRATION_VALID'}[kind])
        else:
            selection = m.base.read(kwargs['selection'])
            assert selection['source_identity'] == m.base.source_identity()
            assert selection['controls'] == kwargs['controls']
            assert selection['reset_precision_policy'] == kwargs['reset_precision_policy'] == m.RESET
            data, design = m.base.prepared_paths(campaign.root, 'model_b', 50, 'final')
            assert selection['data_sha256'] == m.base.hash_file(data)
            assert selection['design_sha256'] == m.base.hash_file(design)
            assert kwargs['model'] == 'model_b' and kwargs['horizon'] == 50
            if kind == 'training':
                result = dict(status='TRAINING_CANDIDATE_VALID', heldout_losses=[1., 2.],
                    checkpoint=str(out/'checkpoint.json'), frozen_map=str(out/'frozen-map.json'))
            else:
                assert kind == 'hmc' and kwargs['frozen_map'].endswith('B-train-extend/frozen-map.json')
                result = dict(status='VERIFIED' if sampling_complete else 'WARMUP_LIMIT',
                              posterior_sampling_complete=sampling_complete)
        m.base.save(out/'result.json', result)
        return result, out
    monkeypatch.setattr(campaign, 'task', task)
    campaign.run()
    assert [x['updates'] for x in calls if x['kind']=='training'] == [256,256,256,256,1024]
    assert len([x for x in calls if x['kind']=='hmc']) == 1
    expected = 'SAMPLING_COMPLETE_ACCURACY_UNRESOLVED' if sampling_complete else 'SAMPLING_INCOMPLETE'
    assert campaign.state['decision'] == expected
    assert not m.base.read(campaign.root/'analysis.json')['exact_posterior_claim']
