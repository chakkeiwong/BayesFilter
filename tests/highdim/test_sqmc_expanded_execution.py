"""CPU-only mechanics and independent finite-difference checks, not accuracy evidence."""
import importlib.util
from pathlib import Path

import pytest
import tensorflow as tf

from bayesfilter.highdim import sqmc_campaign_tf as campaign
from bayesfilter.highdim.sqmc_full_lgssm_tf import FullLGSSMSpec
from bayesfilter.highdim.ledh_canonical_score_tf import canonical_value_and_analytical_score


def runner():
    path = Path(__file__).parents[2]/'docs/benchmarks/run_sqmc_expanded_comparison.py'
    spec = importlib.util.spec_from_file_location('expanded_runner', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_final_seed_design_is_cartesian():
    module = runner()
    assert set(module.final_pairs()) == {(197001,198001),(197001,198002),(197002,198001),(197002,198002)}


def test_first_observation_baseline_is_measured_against_full_oracle():
    module = runner()
    row = dict(valid=True, score=[100.,-100.], score_l2_error=100.)
    module.attach_heuristics(row, [3.,4.], [0.,4.])
    assert row['heuristic_zero_score_l2_error'] == 5.
    assert row['heuristic_first_observation_only_l2_error'] == 3.
    assert row['heuristic_dominance']['first_observation_only'] == 'observed_loss'


@pytest.mark.parametrize('dimension', [3, 10])
@pytest.mark.parametrize('route', list(campaign.ROUTES))
def test_full_matrix_directional_score_matches_finite_program(dimension, route):
    spec = FullLGSSMSpec('full_matrix', dimension)
    dtype = tf.float64
    theta = spec.default_theta(dtype)
    direction = tf.sin(tf.cast(tf.range(spec.parameter_count)+1, dtype))
    direction /= tf.linalg.norm(direction)
    n = 4*dimension
    observations = spec.simulate(theta, 2, 179001, jit_compile=False)
    initial, noise, uniforms = campaign.random_inputs(route, 179002, n, dimension, 2, dtype)
    controls = dict(flow_substeps=2, reset_epsilon=.4, reset_sinkhorn_steps=24,
                    reset_balance_steps=12, correction_steps=1, correction_strength=.12,
                    pairwise_steps=1, pairwise_strength=.03)
    settings = dict(campaign.numerical_settings(controls), **campaign.route_settings(route))
    design = campaign.reset_design(n, dimension, dtype)

    @tf.function(input_signature=[tf.TensorSpec([spec.parameter_count],dtype)]*2,
                 jit_compile=False, autograph=False)
    def evaluate(parameters, tangent):
        model, _ = spec.model(parameters, tangent)
        states, covariances, ds, dc = spec.initial_cloud(parameters, initial, tangent)
        return canonical_value_and_analytical_score(
            model, parameters, states, covariances, noise, observations, with_score=True,
            initial_state_tangent=ds, initial_covariance_tangent=dc,
            reset_design=design, process_ancestor_uniforms=uniforms, **settings)

    value, score = evaluate(theta, direction)
    zero = tf.zeros_like(theta)
    primal, _ = evaluate(theta, zero)
    tf.debugging.assert_near(value, primal, atol=1e-10, rtol=1e-10)
    h = tf.constant(2e-6, dtype)
    plus, _ = evaluate(theta+h*direction, zero)
    minus, _ = evaluate(theta-h*direction, zero)
    expected = (plus-minus)/(2*h)
    tf.debugging.assert_all_finite(tf.stack([value,score[0],expected]), 'invalid mechanics fixture')
    tf.debugging.assert_near(score[0], expected, atol=3e-5, rtol=3e-4)
