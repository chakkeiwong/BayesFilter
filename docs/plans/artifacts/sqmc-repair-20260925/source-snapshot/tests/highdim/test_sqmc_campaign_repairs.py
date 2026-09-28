"""Independent CPU reference checks for the repaired SQMC consumers."""
import importlib.util
from pathlib import Path

import pytest
import tensorflow as tf

from bayesfilter.highdim.sqmc_lgssm_tf import LGSSMSpec
from bayesfilter.highdim import sqmc_campaign_tf as campaign

DTYPE = tf.float64
CONTROLS = dict(reset_epsilon=32., reset_sinkhorn_steps=16, reset_balance_steps=16,
                correction_steps=0, correction_strength=.15, pairwise_steps=0,
                pairwise_strength=.03, flow_substeps=2)


@pytest.mark.parametrize('family,dim', [('p44', 3), ('p44', 10), ('diagonal_ar', 3), ('diagonal_ar', 10), ('frozen_3d', 3)])
def test_parameter_callbacks_match_independent_finite_difference(family, dim):
    spec = LGSSMSpec(family, dim)
    theta = spec.default_theta()
    points = tf.reshape(tf.range(2 * dim, dtype=DTYPE) / 10., [2, dim])
    dx = tf.ones_like(points) * .13
    means, dm = points * .8, dx * .7
    obs = tf.ones([dim], DTYPE) * .4
    h = tf.constant(1e-5, DTYPE)
    for i in range(spec.parameter_count):
        v = tf.one_hot(i, spec.parameter_count, dtype=DTYPE)
        model, _ = spec.model(theta, v)
        plus, _ = spec.model(theta + h*v)
        minus, _ = spec.model(theta - h*v)
        pairs = [
            (model.transition_mean_tangent_fn(theta, points, dx),
             (plus.transition_mean_fn(theta+h*v, points+h*dx)-minus.transition_mean_fn(theta-h*v, points-h*dx))/(2*h)),
            (model.transition_log_density_tangent_fn(theta, points, means, dx, dm),
             (plus.transition_log_density_fn(theta+h*v, points+h*dx, means+h*dm)-minus.transition_log_density_fn(theta-h*v, points-h*dx, means-h*dm))/(2*h)),
            (model.observation_log_density_tangent_fn(theta, points, obs, dx),
             (plus.observation_log_density_fn(theta+h*v, points+h*dx, obs)-minus.observation_log_density_fn(theta-h*v, points-h*dx, obs))/(2*h)),
            (model.process_covariance_tangent_fn(theta), (plus.process_covariance-minus.process_covariance)/(2*h)),
            (model.observation_covariance_tangent_fn(theta), (plus.observation_covariance-minus.observation_covariance)/(2*h)),
        ]
        for actual, expected in pairs:
            tf.debugging.assert_near(actual, expected, atol=2e-7, rtol=2e-6)


def test_p44_matches_original_physical_model_and_observation_first_oracle():
    path = Path(__file__).with_name('test_p44_lgssm_exact_baseline.py')
    loader = importlib.util.spec_from_file_location('p44_reference', path)
    p44 = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(p44)
    spec = LGSSMSpec('p44', 3)
    theta = spec.default_theta()
    source = p44._physical_parts(theta, 3)
    params = spec.kalman_parameters(theta)
    for ours, theirs in [('transition_matrix','transition_matrix'), ('process_covariance','transition_covariance'),
                         ('observation_covariance','observation_covariance'), ('initial_mean','raw_initial_mean'),
                         ('initial_covariance','raw_initial_covariance')]:
        tf.debugging.assert_near(params[ours], source[theirs], atol=1e-14)
    from bayesfilter.linear.kalman_tf import tf_linear_gaussian_log_likelihood
    from bayesfilter.structural_tf import affine_structural_to_linear_gaussian_tf
    obs = spec.simulate(theta, 3, 70001)
    def source_value(t):
        linear = affine_structural_to_linear_gaussian_tf(p44._structural_model(t, 3))
        return tf_linear_gaussian_log_likelihood(obs, linear, backend='tf_cholesky', jitter=tf.constant(0., DTYPE), return_filtered=False).log_likelihood
    exact = source_value(theta)
    scores = []
    for i in range(4):
        delta = tf.one_hot(i, 4, dtype=DTYPE)*1e-5
        scores.append((source_value(theta+delta)-source_value(theta-delta))/2e-5)
    score = tf.stack(scores)
    value, actual = spec.reference_value_and_score(theta, obs)
    tf.debugging.assert_near(value, exact, atol=1e-10)
    tf.debugging.assert_near(actual, score, atol=2e-7, rtol=2e-6)


@pytest.mark.parametrize('family,dim', [('p44', 3), ('diagonal_ar', 3), ('diagonal_ar', 10)])
def test_full_finite_program_score_all_directions(family, dim):
    spec = LGSSMSpec(family, dim)
    theta = spec.default_theta()
    obs = spec.simulate(theta, 1, 70002)
    n = 12 if dim == 3 else 20
    inputs = campaign.random_inputs('iid_dual_cap', 70003, n, dim, 1)
    def run(t):
        return campaign.value_and_score(spec, 'iid_dual_cap', CONTROLS, t, obs, 70003, n, jit_compile=False, inputs=inputs)
    value, score, valid = run(theta)
    assert bool(valid)
    expected = []
    for i in range(spec.parameter_count):
        delta = tf.one_hot(i, spec.parameter_count, dtype=DTYPE) * 1e-5
        a, _, av = run(theta+delta)
        b, _, bv = run(theta-delta)
        assert bool(av & bv)
        expected.append((a-b)/2e-5)
    tf.debugging.assert_near(score, tf.stack(expected), atol=3e-5, rtol=3e-4)
    assert bool(tf.reduce_all(tf.abs(score) > 1e-6))


def test_route_ablation_and_zero_reference_metric():
    a = campaign.route_settings('repaired_permutation')
    b = campaign.route_settings('repaired_permutation_ablation')
    assert a['coordinate_cap'] == .98 and b['coordinate_cap'] == .97
    with pytest.raises(ValueError):
        campaign.route_settings('typo')
    metrics = campaign.score_metrics(tf.constant([1.], DTYPE), tf.constant([0.], DTYPE))
    assert metrics['score_l2_error'] == 1.
    assert metrics['relative_score_error'] is None
    assert 'induced_hmc_error' not in metrics and 'fisher_scaled_errors' not in metrics


def test_nonfinite_sentinel_rejected(monkeypatch):
    monkeypatch.setattr(campaign, 'value_and_score', lambda *args, **kwargs: (
        tf.constant(float('-inf'), DTYPE), tf.zeros([4], DTYPE), tf.constant(True)))
    spec = LGSSMSpec('p44', 3)
    result = campaign.evaluate_diagnostic(spec, 'iid_dual_cap', CONTROLS, tf.zeros([1,3], DTYPE), spec.default_theta(), 1, 12, jit_compile=False)
    assert not result['valid'] and not result['claim_eligible']


def load_runner(name):
    path = Path(__file__).parents[2]/'docs/benchmarks'/name
    loader = importlib.util.spec_from_file_location(name.replace('.py',''),path)
    module = importlib.util.module_from_spec(loader)
    import sys
    sys.path.insert(0,str(path.parent))
    sys.modules[loader.name] = module
    loader.loader.exec_module(module)
    return module


@pytest.mark.parametrize('name', ['run_sqmc_generic_lgssm.py','run_sqmc_tuning.py',
                                  'run_sqmc_dimension_transfer_t20.py','run_sqmc_horizon_transfer.py',
                                  'run_sqmc_10d_t120_tuned.py','run_sqmc_3d_t120_tuned.py'])
def test_every_campaign_endpoint_calls_shared_evaluator(name, monkeypatch):
    module = load_runner(name)
    received = []
    def evaluator(*args, **kwargs):
        received.append((args,kwargs))
        return {'shared_evaluator': True}
    monkeypatch.setattr(module,'evaluate_diagnostic',evaluator)
    if name == 'run_sqmc_generic_lgssm.py':
        spec=LGSSMSpec('p44',3)
    elif name in ('run_sqmc_tuning.py','run_sqmc_3d_t120_tuned.py'):
        spec=LGSSMSpec('frozen_3d',3)
    else:
        spec=LGSSMSpec('diagonal_ar',10 if '10d' in name else 3)
    theta=spec.default_theta()
    obs=spec.simulate(theta,1,74001)
    oracle=spec.reference_value_and_score(theta,obs)[1]
    route='repaired_permutation_ablation'
    if name == 'run_sqmc_generic_lgssm.py':
        result=module._evaluate_generic_sqmc(route,CONTROLS,obs,theta,oracle,1,1,12,3,jit_compile=False)
    elif name == 'run_sqmc_tuning.py':
        result=module._evaluate_controls('repaired_permutation',CONTROLS,obs,theta,oracle,1,1,12,True,jit_compile=False)
    elif name == 'run_sqmc_dimension_transfer_t20.py':
        monkeypatch.setattr(module,'HORIZON',1)
        result=module._evaluate_sqmc(route,CONTROLS,obs,theta,1,3,12,jit_compile=False)
    elif name == 'run_sqmc_horizon_transfer.py':
        result=module._evaluate_sqmc(route,CONTROLS,obs,theta,1,3,12,1,jit_compile=False)
    else:
        monkeypatch.setattr(module,'PARTICLE_COUNT',20)
        result=module._evaluate_route_on_seed(route,CONTROLS,obs,theta,1,jit_compile=False)
    assert result == {'shared_evaluator':True}
    assert received[0][0][0] == spec
    assert received[0][0][1] == route
    assert received[0][1]['jit_compile'] is False


def test_tuning_requires_all_seeds_and_exact_scope(monkeypatch):
    from dataclasses import replace
    from bayesfilter.highdim import sqmc_campaign_tuning as tuning
    spec=LGSSMSpec('p44',3)
    theta=spec.default_theta()
    def result(spec,route,controls,observations,theta,seed,n,**kwargs):
        return dict(valid=True,seed=seed,value=1.,oracle_value=1.,score_l2_error=controls['reset_epsilon'],
                    score=[1.]*4,oracle_score=[2.]*4)
    monkeypatch.setattr(tuning,'evaluate_diagnostic',result)
    with pytest.raises(ValueError,match='disjoint'):
        tuning.tune_campaign(spec,'iid_dual_cap',[CONTROLS],theta,1,12,[1,2],[2,3],[4],jit_compile=False)
    artifact,report=tuning.tune_campaign(spec,'iid_dual_cap',[CONTROLS],theta,1,12,[1,2],[3,4],[5,6],jit_compile=False)
    assert artifact is not None
    assert tuning.evaluate_untouched(spec,'iid_dual_cap',artifact,theta,1,12,5,jit_compile=False)['valid']
    with pytest.raises(ValueError,match='scope'):
        tuning.evaluate_untouched(spec,'iid_dual_cap',artifact,theta,2,12,5,jit_compile=False)
    with pytest.raises(ValueError,match='partition'):
        tuning.evaluate_untouched(spec,'iid_dual_cap',artifact,theta,1,12,1,jit_compile=False)
    with pytest.raises(TypeError,match='repository-issued'):
        tuning.evaluate_untouched(spec,'iid_dual_cap',replace(artifact),theta,1,12,5,jit_compile=False)
    good=[result(spec,'iid_dual_cap',CONTROLS,None,theta,i,12) for i in range(16)]
    assert tuning.complete_valid_results(good,list(range(16)),4)
    good[15]['valid']=False
    assert not tuning.complete_valid_results(good,list(range(16)),4)
    def failed_validation(*args,**kwargs):
        row=result(*args,**kwargs)
        row['valid']=row['seed'] != 4
        return row
    monkeypatch.setattr(tuning,'evaluate_diagnostic',failed_validation)
    bad,report=tuning.tune_campaign(spec,'iid_dual_cap',[CONTROLS],theta,1,12,[1,2],[3,4],[5,6],jit_compile=False)
    assert bad is None and report['decision'].startswith('validation_failed')


@pytest.mark.parametrize('route', list(campaign.ROUTES))
def test_two_observation_corrected_score_matches_finite_program(route):
    spec=LGSSMSpec('p44',3)
    theta=spec.default_theta()
    observations=spec.simulate(theta,2,81101)
    controls=dict(CONTROLS,correction_steps=1,pairwise_steps=1,reset_epsilon=.4,
                  reset_sinkhorn_steps=24,reset_balance_steps=12)
    inputs=campaign.random_inputs(route,81102,12,3,2)
    def run(value):
        return campaign.value_and_score(spec,route,controls,value,observations,81102,12,
                                         jit_compile=False,inputs=inputs)
    value,score,valid=run(theta)
    assert bool(valid)
    expected=[]
    for i in range(spec.parameter_count):
        h=tf.one_hot(i,spec.parameter_count,dtype=DTYPE)*1e-5
        plus,_,a=run(theta+h); minus,_,b=run(theta-h)
        assert bool(a & b)
        expected.append((plus-minus)/2e-5)
    tf.debugging.assert_near(score,tf.stack(expected),atol=3e-5,rtol=3e-4)
    if route=='repaired_permutation':
        other=campaign.value_and_score(spec,'repaired_permutation_ablation',controls,theta,observations,
                                       81102,12,jit_compile=False,inputs=inputs)
        assert bool(other[2])
        assert float(tf.abs(value-other[0])) > 1e-10


@pytest.mark.parametrize('route',['hilbert_inverse_cdf','hilbert_permutation_one_to_one'])
def test_ancestor_count_matches_independent_unique_reference(route):
    from bayesfilter.highdim.ledh_pfpf_genut_initial_rqmc_tf import _transition_ancestors
    points=tf.reshape(tf.range(24,dtype=DTYPE),[12,2])*.1
    uniforms=tf.constant([.02,.03,.05,.3,.31,.32,.6,.61,.8,.85,.9,.99],DTYPE)
    result=_transition_ancestors(points,tf.fill([12],tf.constant(1/12,DTYPE)),uniforms,
        ancestry_policy=route,state_map_location=tf.zeros([2],DTYPE),state_map_scale=tf.ones([2],DTYPE),hilbert_bits=8)
    unique=tf.unique(result['selected_row_identities']).y
    assert int(result['ancestry_unique_count']) == int(tf.size(unique))


def test_controls_reject_fractional_loop_count():
    with pytest.raises(ValueError,match='integer'):
        campaign.numerical_settings(dict(CONTROLS,flow_substeps=1.5))


def test_generic_smoke_rejects_missing_dimension_cells(tmp_path):
    from types import SimpleNamespace
    runner=load_runner('run_sqmc_generic_lgssm.py')
    with pytest.raises(ValueError,match='every requested dimension'):
        runner.smoke_test(SimpleNamespace(dimensions=[3,10],seeds=[1],horizon=1,
                                         particles=12,output_dir=str(tmp_path)))


def test_legacy_metric_clis_use_repaired_tuner():
    tuner=load_runner('run_sqmc_tuning.py')
    for name in ('run_sqmc_test1b_insample.py','run_sqmc_test1b_insample_v2.py',
                 'run_sqmc_test1b_insample_temp.py','run_sqmc_tuned_vs_untuned.py',
                 'run_sqmc_route_comparison.py','calibrate_sqmc_tuning_vetoes.py',
                 'smoke_sqmc_tuning_single_cell.py'):
        assert load_runner(name).main is tuner.main
