"""Independent numerical and lifecycle oracles with deliberate wrong programs.

These are CPU reference checks. Mutations test sensitivity of the oracles;
passing the positive controls establishes no posterior or performance claim.
"""
import math

import numpy as np  # Independent closed-form reference only.
import pytest
import tensorflow as tf

from bayesfilter.inference import hmc_modern_kernels as kernels
from tests.test_modern_hmc_kernels import runner
from tests.test_hmc_candidate_set_execution import GaussianTarget


def gaussian_mams_oracle(q, u, epsilon, count):
    """Hyperbolic exact sphere flow; independent of the scaled runtime formula."""
    position, direction = q.copy(), u.copy()
    log_jacobian = np.zeros(len(q))

    def force(direction, score):
        norm = np.linalg.norm(score, axis=-1, keepdims=True)
        axis = score / norm
        delta = .5 * epsilon * norm / (q.shape[-1]-1)
        projection = np.sum(axis*direction, axis=-1, keepdims=True)
        denominator = np.cosh(delta) + projection*np.sinh(delta)
        updated = (direction + axis*(np.sinh(delta) + projection*(np.cosh(delta)-1))) / denominator
        return updated, -(q.shape[-1]-1)*np.log(denominator[:, 0])

    for _ in range(count):
        direction, log_j = force(direction, -position)
        log_jacobian += log_j
        position = position + epsilon*direction
        direction, log_j = force(direction, -position)
        log_jacobian += log_j
    ratio = -.5*np.sum(position*position-q*q, axis=-1) + log_jacobian
    return position, ratio


@pytest.mark.parametrize("mutation", ["none", "wrong_jacobian_sign", "stale_force"])
def test_full_transition_oracle_detects_wrong_correction_and_stale_force(monkeypatch, mutation):
    q = tf.constant([[.2,-.8], [-1.,1.], [.1,.9], [2.,-.3]], tf.float64)
    if mutation == "wrong_jacobian_sign":
        correct_force = kernels.sphere_force_substep
        def wrong_force(*args):
            direction, work = correct_force(*args)
            return direction, -work
        monkeypatch.setattr(kernels, "sphere_force_substep", wrong_force)
    target = None
    if mutation == "stale_force":
        class StaleForce(GaussianTarget):
            def log_prob_and_grad(self, position):
                value, _ = super().log_prob_and_grad(position)
                return value, -q
        target = StaleForce()
    native = runner(family="mams", trajectory="fixed", count=1, target=target)
    _, trace = native.one_step(q, tf.constant([381,52]), tf.constant(.8,tf.float64), tf.constant(3))
    expected_state, expected_ratio = gaussian_mams_oracle(q.numpy(), trace["initial_momentum"].numpy(), .8, 3)
    def oracle():
        np.testing.assert_allclose(trace["proposed_state"], expected_state, atol=2e-13, rtol=2e-13)
        np.testing.assert_allclose(trace["log_accept_ratio"], expected_ratio, atol=2e-13, rtol=2e-13)
    if mutation == "none":
        oracle()
    else:
        with pytest.raises(AssertionError):
            oracle()


@pytest.mark.parametrize("disabled", [False, True])
def test_fixed_travel_proposer_detects_disabled_epsilon_adaptation(tmp_path, monkeypatch, disabled):
    from bayesfilter.inference import hmc_mams_trajectory_proposals as proposals
    from tests.test_hmc_mams_trajectory_proposals import config_for
    from tests.test_modern_hmc_pipeline import setup
    binding, _ = setup(family="mams")
    config = config_for(binding, duration_factors=(1.,), max_candidates=1)
    if disabled:
        monkeypatch.setattr(proposals, "_bracketed_epsilon",
            lambda eps, acceptance, target, lower, upper, gain, domain: (eps, lower, upper, eps, False))
    result = proposals.propose_mams_trajectory_candidates(binding, config, output_dir=tmp_path/"pilot")
    first = result["duration_rows"][0]
    probe = first["adaptation"][0]
    # Numerical fixture has acceptance above target. A no-op update would
    # reproduce the historical absent-adaptation mechanism despite finite draws.
    assert probe["acceptance"] > config.target_accept_probability
    def adaptation_oracle():
        assert probe["bounded_next_epsilon"] > probe["epsilon"]
    if disabled:
        with pytest.raises(AssertionError):
            adaptation_oracle()
    else:
        adaptation_oracle()
    compiled = first["compiled_duration"]
    assert abs(compiled["realized_mean_duration"] - config.initial_mean_duration) <= first["epsilon"]/4 + 1e-14


def test_uniform_duration_oracle_rejects_half_maximum_mutation():
    from bayesfilter.inference.hmc_trajectory_duration import compile_mean_duration
    from bayesfilter.inference.hmc_transition_spec import HMCTransitionProfile
    compiled = compile_mean_duration(profile=HMCTransitionProfile("mams", "uniform_duration"),
        epsilon=.4, requested_mean_duration=1.2, max_steps_cap=32)
    count = compiled.frozen.max_steps
    independent_mean = math.fsum(.4*n for n in range(1,count+1))/count
    assert compiled.realized_mean_duration == pytest.approx(independent_mean)
    with pytest.raises(AssertionError):
        assert .4*count/2 == pytest.approx(independent_mean)


def test_qualification_rejects_reused_measurement_stream_as_verification(tmp_path, monkeypatch):
    from dataclasses import replace
    from bayesfilter.inference import tune_hmc_kernel
    from bayesfilter.inference import hmc_acceptance_trials as trials
    from tests.test_modern_hmc_pipeline import setup
    binding, template = setup(family="mams")
    config = replace(template, primary_l_grid=(3,), epsilon_by_l=((3, (.2,)),))
    original = trials.work_seed
    measurement_roots = {}
    def leaky_seed(runtime, work):
        if work.stage == "measurement":
            root = original(runtime, work)
            measurement_roots[work.candidate_id] = root
            return root
        return measurement_roots[work.candidate_id]
    monkeypatch.setattr(trials, "work_seed", leaky_seed)
    run = tune_hmc_kernel(adapter=binding._base_adapter, initial_position=binding.initial_active_state,
                         config=config, candidate_set_adapter=binding.typed_adapter, output_dir=tmp_path/"tuning")
    assert run.result.completion_status == "shared_invalidity"
    assert not run.result.verified_candidate_ids
    failures = [e for e in run.result.accounting_events if e.get("event") == "execution_failure"]
    assert any(e["reason"] == "duplicate independent-trial random stream" for e in failures)
