"""Bounded Phase 1 checks for the C2 per-ancestor UKF/APF adapter."""

from __future__ import annotations

import json
import math

import tensorflow as tf

from bayesfilter.highdim.c2_mixture_ukf_apf_c2_adapter import (
    ROUTE_ID,
    compile_c2_per_ancestor_ukf_apf_k1,
)
from bayesfilter.highdim.c2_sv_frozen_proposal_apf_tf import (
    C2StochasticVolatilityFrozenAPFModel,
)
from bayesfilter.highdim.zhao_cui_frozen_proposal_apf_tf import (
    prepare_frozen_proposal_apf_program,
)


DTYPE = tf.float64
FIXTURE = "docs/benchmarks/fixtures/c2_sv_n4_seed52_obs42_t20_frozen_v1.json"


def _fixture(horizon: int = 5):
    payload = json.loads(open(FIXTURE, encoding="utf-8").read())
    theta = tf.constant([payload["gamma"], math.log(payload["beta"])], DTYPE)
    transition = tf.constant(payload["transition_matrix"], DTYPE)
    coupling = transition - theta[0] * tf.eye(4, dtype=DTYPE)
    model = C2StochasticVolatilityFrozenAPFModel(
        coupling_matrix=coupling, sigma=float(payload["sigma"])
    )
    observations = tf.constant(payload["observations"][:horizon], DTYPE)
    return model, theta, observations


def _finite_difference(program, theta: tf.Tensor, index: int, step: float = 1e-5):
    direction = tf.one_hot(index, int(theta.shape[0]), dtype=DTYPE)
    return (
        program.evaluate(theta + step * direction)["log_likelihood"]
        - program.evaluate(theta - step * direction)["log_likelihood"]
    ) / (2.0 * step)


def test_phase1_adapter_is_generic_kernel_call_chain_and_finite() -> None:
    model, theta, observations = _fixture()
    compilation = compile_c2_per_ancestor_ukf_apf_k1(
        model=model,
        observations=observations,
        theta_reference=theta,
        particle_count=16,
        seed=9101,
        jit_compile=True,
    )
    assert compilation.manifest["route_id"] == ROUTE_ID
    assert compilation.manifest["proposal_family"] == (
        "per_ancestor_gaussian_ukf_posterior_k1"
    )
    assert compilation.branch.states.shape == (5, 16, 4)
    assert compilation.branch.ancestors.shape == (4, 16)
    assert len(compilation.proposal_diagnostics) == 4
    assert all(
        bool(row["proposal_finite"].numpy())
        and bool(row["exact_prefix_finite"].numpy())
        for row in compilation.proposal_diagnostics
    )
    assert all(
        float(row["proposal_density_recomposition_max_abs"].numpy()) <= 2e-12
        for row in compilation.proposal_diagnostics
    )


def test_phase1_frozen_exact_score_matches_central_difference() -> None:
    model, theta, observations = _fixture(horizon=4)
    compilation = compile_c2_per_ancestor_ukf_apf_k1(
        model=model,
        observations=observations,
        theta_reference=theta,
        particle_count=24,
        seed=9102,
        jit_compile=True,
    )
    program = prepare_frozen_proposal_apf_program(model, compilation.branch)
    result = program.evaluate(theta)
    finite_difference = tf.stack(
        [_finite_difference(program, theta, index) for index in range(2)]
    )
    tf.debugging.assert_near(result["score"], finite_difference, atol=2e-7, rtol=2e-7)
    assert bool(result["finite"].numpy())
    compiled = program.compiled(jit_compile=False)(theta)
    tf.debugging.assert_near(
        result["log_likelihood"], compiled["log_likelihood"], atol=2e-10
    )
    tf.debugging.assert_near(result["score"], compiled["score"], atol=2e-10)


def test_phase1_apf_law_is_observation_sensitive_per_ancestor() -> None:
    model, theta, observations = _fixture(horizon=3)
    first = compile_c2_per_ancestor_ukf_apf_k1(
        model=model,
        observations=observations,
        theta_reference=theta,
        particle_count=20,
        seed=9103,
        jit_compile=False,
    )
    shifted = tf.tensor_scatter_nd_add(
        observations, tf.constant([[1, 0]], tf.int32), tf.constant([0.7], DTYPE)
    )
    second = compile_c2_per_ancestor_ukf_apf_k1(
        model=model,
        observations=shifted,
        theta_reference=theta,
        particle_count=20,
        seed=9103,
        jit_compile=False,
    )
    first_means = first.proposal_diagnostics[0]["posterior_mean"]
    second_means = second.proposal_diagnostics[0]["posterior_mean"]
    assert float(tf.reduce_max(tf.abs(first_means - second_means)).numpy()) > 1e-4
    first_law = first.proposal_diagnostics[0]["ancestor_log_probabilities"]
    second_law = second.proposal_diagnostics[0]["ancestor_log_probabilities"]
    assert float(tf.reduce_max(tf.abs(first_law - second_law)).numpy()) > 1e-4


def test_phase1_xla_and_non_xla_frozen_branch_are_reproducible() -> None:
    model, theta, observations = _fixture(horizon=3)
    xla = compile_c2_per_ancestor_ukf_apf_k1(
        model=model,
        observations=observations,
        theta_reference=theta,
        particle_count=16,
        seed=9104,
        jit_compile=True,
    )
    eager = compile_c2_per_ancestor_ukf_apf_k1(
        model=model,
        observations=observations,
        theta_reference=theta,
        particle_count=16,
        seed=9104,
        jit_compile=False,
    )
    tf.debugging.assert_near(xla.branch.states, eager.branch.states, atol=2e-10)
    tf.debugging.assert_equal(xla.branch.ancestors, eager.branch.ancestors)
    tf.debugging.assert_near(
        xla.branch.transition_log_proposal_density,
        eager.branch.transition_log_proposal_density,
        atol=2e-10,
    )
