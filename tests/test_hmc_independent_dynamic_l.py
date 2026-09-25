"""CPU mechanics for scalar-chain graph reuse; no sampler convergence claim."""
from dataclasses import replace

import pytest
import tensorflow as tf

from bayesfilter.inference.hmc import FullChainHMCConfig, build_independent_chain_tfp_hmc_runner
from tests.test_hmc_status_reuse import StatusGaussian, assert_same


def config(**overrides):
    return replace(FullChainHMCConfig(num_results=8, num_burnin_steps=0,
        step_size=.2, num_leapfrog_steps=3, seed=(73, 11), use_xla=False,
        target_status_trace_policy="per_chain_step", capture_candidate_health=True,
        target_scope="candidate-bridge-test"), **overrides)


def starts():
    return tf.constant([[-1., -.5], [-.3, .2], [.4, -.2], [1., .5]], tf.float64)


@pytest.mark.parametrize("mode", ["serial", "threaded"])
def test_dynamic_l_preserves_all_scalar_streams_health_and_momenta(mode):
    target = StatusGaussian(invalid_radius=3.)
    shared = build_independent_chain_tfp_hmc_runner(target, starts(), config(), dynamic_num_leapfrog_steps=True)
    # Changed state, epsilon, L and seed followed by returning to the first
    # configuration exposes stale captures rather than only first-call parity.
    for length, step, shift, seed in [(3, .2, 0., (1, 9)), (7, 1.8, .1, (2, 8)), (3, .2, 0., (1, 9))]:
        state = starts() + shift
        static = build_independent_chain_tfp_hmc_runner(target, state, config(num_leapfrog_steps=length))
        expected = static.run(current_state=state, root_seed=seed, step_size=step, mode=mode)
        actual = shared.run(current_state=state, root_seed=seed, step_size=step, mode=mode, num_leapfrog_steps=length)
        assert_same((actual.samples, actual.trace), (expected.samples, expected.trace))
        assert actual.metadata["chain_seeds"] == expected.metadata["chain_seeds"]
        assert actual.metadata["num_leapfrog_steps"] == length
    assert [runner._runner.experimental_get_tracing_count() for runner in shared._runners] == [1] * 4


@pytest.mark.parametrize("length", [0, -1, [3]])
def test_invalid_dynamic_l_fails_before_tracing(length):
    runner = build_independent_chain_tfp_hmc_runner(StatusGaussian(), starts(), config(), dynamic_num_leapfrog_steps=True)
    with pytest.raises(ValueError, match="num_leapfrog_steps"):
        runner.run(num_leapfrog_steps=length)
    assert all(r._runner.experimental_get_tracing_count() == 0 for r in runner._runners)


def test_static_runner_does_not_silently_ignore_a_changed_l():
    runner = build_independent_chain_tfp_hmc_runner(StatusGaussian(), starts(), config())
    with pytest.raises(ValueError, match="dynamic_num_leapfrog_steps"):
        runner.run(num_leapfrog_steps=7)
