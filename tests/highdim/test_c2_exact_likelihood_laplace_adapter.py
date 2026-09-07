"""C2 Phase 8A derivative, limit, call-chain, and frozen-score tests."""

from __future__ import annotations

import json
import math
from pathlib import Path

import tensorflow as tf

import bayesfilter.highdim.c2_exact_likelihood_laplace_adapter as adapter_module
from bayesfilter.highdim.c2_exact_likelihood_laplace_adapter import (
    ROUTE_ID,
    compile_c2_exact_likelihood_laplace_apf_k1,
)
from bayesfilter.highdim.c2_sv_frozen_proposal_apf_tf import (
    C2StochasticVolatilityFrozenAPFModel,
)
from bayesfilter.highdim.exact_likelihood_laplace_apf_tf import (
    DTYPE,
    FixedLaplaceConfig,
    make_fixed_laplace_bank_kernel,
    single_start_offsets,
)
from bayesfilter.highdim.zhao_cui_frozen_proposal_apf_tf import (
    prepare_frozen_proposal_apf_program,
)


FIXTURE = Path("docs/benchmarks/fixtures/c2_sv_n4_seed52_obs42_t20_frozen_v1.json")


def _fixture():
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    theta = tf.constant([payload["gamma"], math.log(payload["beta"])], DTYPE)
    transition = tf.constant(payload["transition_matrix"], DTYPE)
    coupling = transition - theta[0] * tf.eye(4, dtype=DTYPE)
    model = C2StochasticVolatilityFrozenAPFModel(
        coupling_matrix=coupling, sigma=float(payload["sigma"])
    )
    observations = tf.constant(payload["observations"], DTYPE)
    return model, theta, observations


def _config() -> FixedLaplaceConfig:
    return FixedLaplaceConfig(
        tempering_schedule=(0.25, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0, 1.0),
        step_fractions=(1.0,) * 8,
        start_scale=0.0,
        stationarity_relative_tolerance=2.0e-10,
    )


def _finite_difference(program, theta: tf.Tensor, index: int, step: float = 1e-5):
    direction = tf.one_hot(index, int(theta.shape[0]), dtype=DTYPE)
    return (
        program.evaluate(theta + step * direction)["log_likelihood"]
        - program.evaluate(theta - step * direction)["log_likelihood"]
    ) / (2.0 * step)


def test_c2_state_score_and_negative_hessian_match_finite_differences() -> None:
    model, theta, observations = _fixture()
    states = tf.constant(
        [[-0.7, 0.2, 1.1, -1.3], [0.4, -0.6, 0.1, 0.8]], DTYPE
    )
    observation = observations[14]
    score = model.observation_log_density_state_score(theta, states, observation)
    information = model.observation_log_density_state_negative_hessian(
        theta, states, observation
    )
    step = tf.constant(1.0e-5, DTYPE)
    score_columns = []
    curvature_columns = []
    for index in range(4):
        direction = tf.one_hot(index, 4, dtype=DTYPE)[None, :]
        plus = model.observation_log_density(theta, states + step * direction, observation, 0)
        minus = model.observation_log_density(theta, states - step * direction, observation, 0)
        score_columns.append((plus - minus) / (2.0 * step))
        score_plus = model.observation_log_density_state_score(
            theta, states + step * direction, observation
        )[:, index]
        score_minus = model.observation_log_density_state_score(
            theta, states - step * direction, observation
        )[:, index]
        curvature_columns.append(-(score_plus - score_minus) / (2.0 * step))
    finite_score = tf.stack(score_columns, axis=1)
    finite_diagonal = tf.stack(curvature_columns, axis=1)
    tf.debugging.assert_near(score, finite_score, atol=2e-10, rtol=2e-10)
    tf.debugging.assert_near(
        tf.linalg.diag_part(information), finite_diagonal, atol=2e-10, rtol=2e-10
    )
    off_diagonal = information - tf.linalg.diag(tf.linalg.diag_part(information))
    tf.debugging.assert_equal(off_diagonal, tf.zeros_like(off_diagonal))


def test_c2_near_zero_observation_has_finite_exact_laplace_limit() -> None:
    model, theta, _ = _fixture()
    prior_mean = tf.constant([[0.7]], DTYPE)
    prior_variance = tf.constant([[[1.3]]], DTYPE)
    tiny_observation = tf.constant([1.0e-14], DTYPE)

    def log_likelihood(states: tf.Tensor, observation: tf.Tensor) -> tf.Tensor:
        xi = theta[1]
        scaled = tf.square(observation)[None, :] * tf.exp(-states - 2.0 * xi)
        return tf.reduce_sum(-xi - 0.5 * states - 0.5 * scaled, axis=1)

    def score(states: tf.Tensor, observation: tf.Tensor) -> tf.Tensor:
        xi = theta[1]
        scaled = tf.square(observation)[None, :] * tf.exp(-states - 2.0 * xi)
        return -0.5 * tf.ones_like(states) + 0.5 * scaled

    def information(states: tf.Tensor, observation: tf.Tensor) -> tf.Tensor:
        xi = theta[1]
        scaled = tf.square(observation)[None, :] * tf.exp(-states - 2.0 * xi)
        return tf.linalg.diag(0.5 * scaled)

    kernel = make_fixed_laplace_bank_kernel(
        batch_size=1,
        state_dim=1,
        observation_dim=1,
        component_offsets=single_start_offsets(1),
        log_likelihood_fn=log_likelihood,
        state_score_fn=score,
        state_negative_hessian_fn=information,
        config=FixedLaplaceConfig((1.0,), (1.0,), 0.0, 1e-12),
        jit_compile=False,
    )
    result = kernel(prior_mean, prior_variance, tiny_observation)
    assert bool(result["valid"].numpy())
    tf.debugging.assert_near(
        result["component_means"][0, 0, 0],
        prior_mean[0, 0] - 0.5 * prior_variance[0, 0, 0],
        atol=2e-13,
    )
    tf.debugging.assert_near(
        result["component_covariances"][0, 0, 0, 0],
        prior_variance[0, 0, 0],
        atol=2e-13,
    )


def test_c2_compilation_calls_generic_kernel_and_bounds_tracing(monkeypatch) -> None:
    model, theta, observations = _fixture()
    calls = []
    original = adapter_module.make_fixed_laplace_bank_kernel

    def recording_factory(**kwargs):
        calls.append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(
        adapter_module, "make_fixed_laplace_bank_kernel", recording_factory
    )
    selected_observations = tf.stack(
        [observations[0], observations[14], observations[15]], axis=0
    )
    compilation = compile_c2_exact_likelihood_laplace_apf_k1(
        model=model,
        observations=selected_observations,
        theta_reference=theta,
        particle_count=16,
        seed=9801,
        laplace_config=_config(),
        jit_compile=False,
    )
    assert len(calls) == 1
    assert calls[0]["batch_size"] == 16
    assert calls[0]["state_dim"] == 4
    assert compilation.manifest["route_id"] == ROUTE_ID
    assert compilation.manifest["generic_kernel_route_id"] == (
        "exact_likelihood_fixed_laplace_bank_candidate_v1"
    )
    assert compilation.manifest["laplace_kernel_trace_count"] == 1
    assert compilation.manifest["sampler_trace_count"] == 1
    assert compilation.branch.states.shape == (3, 16, 4)
    for row in compilation.proposal_diagnostics:
        assert bool(row["laplace_valid"].numpy())
        assert bool(row["proposal_finite"].numpy())
        assert float(row["proposal_density_recomposition_max_abs"].numpy()) < 2e-12
        assert row["iteration_objective_before_step"].shape == (8, 16)
        assert row["iteration_objective_after_step"].shape == (8, 16)
        tf.debugging.assert_all_finite(row["iteration_objective_before_step"], "before")
        tf.debugging.assert_all_finite(row["iteration_objective_after_step"], "after")


def test_c2_laplace_random_stream_matches_k1_comparator_contract() -> None:
    model, theta, observations = _fixture()
    selected_observations = tf.stack(
        [observations[0], observations[14], observations[15]], axis=0
    )
    seed = 9811
    compilation = compile_c2_exact_likelihood_laplace_apf_k1(
        model=model,
        observations=selected_observations,
        theta_reference=theta,
        particle_count=12,
        seed=seed,
        laplace_config=_config(),
        jit_compile=False,
    )
    covariance, _ = model.stationary_covariance_and_derivative(theta)
    expected_initial = tf.einsum(
        "ij,nj->ni",
        tf.linalg.cholesky(covariance),
        tf.random.stateless_normal([12, 4], [seed, 1001], dtype=DTYPE),
    )
    tf.debugging.assert_near(compilation.branch.states[0], expected_initial, atol=0.0)
    assert compilation.manifest["random_key_offsets"] == {
        "initial_normal": 1001,
        "categorical_uniforms": {"base": 5100, "stride": 41},
        "standard_normal": {"base": 5200, "stride": 43},
        "paired_with": "c2_per_ancestor_ukf_apf_k1",
    }


def test_c2_laplace_frozen_exact_score_matches_central_difference() -> None:
    model, theta, observations = _fixture()
    selected_observations = tf.stack(
        [observations[0], observations[14], observations[15]], axis=0
    )
    compilation = compile_c2_exact_likelihood_laplace_apf_k1(
        model=model,
        observations=selected_observations,
        theta_reference=theta,
        particle_count=24,
        seed=9802,
        laplace_config=_config(),
        jit_compile=False,
    )
    program = prepare_frozen_proposal_apf_program(model, compilation.branch)
    result = program.evaluate(theta)
    finite_difference = tf.stack(
        [_finite_difference(program, theta, index) for index in range(2)]
    )
    assert bool(result["finite"].numpy())
    tf.debugging.assert_near(
        result["score"], finite_difference, atol=3e-7, rtol=3e-7
    )
