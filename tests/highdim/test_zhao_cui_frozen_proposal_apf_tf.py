from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import pytest
import tensorflow as tf

from bayesfilter.highdim.zhao_cui_frozen_proposal_apf_tf import (
    MEASURE_ID,
    ROUTE_CLASSIFICATION,
    SCORE_BACKEND_ID,
    TARGET_CLASS,
    prepare_frozen_proposal_apf_program,
    prepare_frozen_proposal_branch,
)


DTYPE = tf.float64
LOG_TWO_PI = tf.constant(math.log(2.0 * math.pi), DTYPE)


@dataclass(frozen=True)
class _LocationLGSSM:
    dimension: int
    prior_variance: float = 1.25
    transition_variance: float = 0.7
    observation_variance: float = 0.8
    transition_scale: float = 0.65
    measure_id: str = MEASURE_ID

    def parameter_dim(self) -> int:
        return 2

    def state_dim(self) -> int:
        return int(self.dimension)

    def observation_dim(self) -> int:
        return int(self.dimension)

    def frozen_apf_measure_id(self) -> str:
        return self.measure_id

    def frozen_apf_score_backend_id(self) -> str:
        return SCORE_BACKEND_ID

    def initial_log_density(self, theta: tf.Tensor, x0: tf.Tensor) -> tf.Tensor:
        mean = tf.fill([self.dimension], theta[0])
        return _isotropic_log_density(x0, mean, self.prior_variance)

    def transition_log_density(
        self,
        theta: tf.Tensor,
        x_previous: tf.Tensor,
        x_current: tf.Tensor,
        time_index: int,
    ) -> tf.Tensor:
        del time_index
        mean = self.transition_scale * x_previous + theta[0]
        return _isotropic_log_density(x_current, mean, self.transition_variance)

    def observation_log_density(
        self,
        theta: tf.Tensor,
        state: tf.Tensor,
        observation: tf.Tensor,
        time_index: int,
    ) -> tf.Tensor:
        del time_index
        mean = state + theta[1]
        observed = tf.broadcast_to(observation[None, :], tf.shape(state))
        return _isotropic_log_density(observed, mean, self.observation_variance)

    def initial_log_density_parameter_score(
        self, theta: tf.Tensor, x0: tf.Tensor
    ) -> tf.Tensor:
        residual = x0 - theta[0]
        score0 = tf.reduce_sum(residual, axis=1) / self.prior_variance
        return tf.stack([score0, tf.zeros_like(score0)], axis=1)

    def transition_log_density_parameter_score(
        self,
        theta: tf.Tensor,
        x_previous: tf.Tensor,
        x_current: tf.Tensor,
        time_index: int,
    ) -> tf.Tensor:
        del time_index
        residual = x_current - (self.transition_scale * x_previous + theta[0])
        score0 = tf.reduce_sum(residual, axis=1) / self.transition_variance
        return tf.stack([score0, tf.zeros_like(score0)], axis=1)

    def observation_log_density_parameter_score(
        self,
        theta: tf.Tensor,
        state: tf.Tensor,
        observation: tf.Tensor,
        time_index: int,
    ) -> tf.Tensor:
        del time_index
        residual = observation[None, :] - (state + theta[1])
        score1 = tf.reduce_sum(residual, axis=1) / self.observation_variance
        return tf.stack([tf.zeros_like(score1), score1], axis=1)

    def manifest_payload(self) -> Mapping[str, object]:
        return {
            "family": "test_location_lgssm",
            "dimension": self.dimension,
            "prior_variance": self.prior_variance,
            "transition_variance": self.transition_variance,
            "observation_variance": self.observation_variance,
            "transition_scale": self.transition_scale,
            "measure_id": self.measure_id,
        }


def _isotropic_log_density(
    value: tf.Tensor, mean: tf.Tensor, variance: float
) -> tf.Tensor:
    residual = value - mean
    dimension = tf.cast(tf.shape(residual)[-1], DTYPE)
    variance_tensor = tf.constant(variance, DTYPE)
    return -0.5 * (
        dimension * (LOG_TWO_PI + tf.math.log(variance_tensor))
        + tf.reduce_sum(tf.square(residual), axis=-1) / variance_tensor
    )


def _empty_transition_branch(particle_count: int) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    return (
        tf.zeros([0, particle_count], tf.int32),
        tf.zeros([0, particle_count], DTYPE),
        tf.zeros([0, particle_count], DTYPE),
    )


def _direct_fixed_branch_scalar(
    model: _LocationLGSSM,
    theta: tf.Tensor,
    *,
    observations: tf.Tensor,
    states: tf.Tensor,
    initial_log_q: tf.Tensor,
    ancestors: tf.Tensor,
    auxiliary_log_probabilities: tf.Tensor,
    transition_log_q: tf.Tensor,
) -> tf.Tensor:
    """Independent value-only statement of the reviewed APF normalization."""

    particle_count = int(states.shape[1])
    log_particle_count = tf.math.log(tf.cast(particle_count, DTYPE))
    initial_log_weight = (
        model.initial_log_density(theta, states[0])
        + model.observation_log_density(theta, states[0], observations[0], 0)
        - initial_log_q
    )
    log_sum = tf.reduce_logsumexp(initial_log_weight)
    value = log_sum - log_particle_count
    previous_log_weight = initial_log_weight - log_sum
    for time_index in range(1, int(states.shape[0])):
        ancestor = ancestors[time_index - 1]
        parent = tf.gather(states[time_index - 1], ancestor)
        selected_previous = tf.gather(previous_log_weight, ancestor)
        selected_auxiliary = tf.gather(
            auxiliary_log_probabilities[time_index - 1], ancestor
        )
        current_log_weight = (
            selected_previous
            + model.transition_log_density(
                theta, parent, states[time_index], time_index
            )
            + model.observation_log_density(
                theta,
                states[time_index],
                observations[time_index],
                time_index,
            )
            - selected_auxiliary
            - transition_log_q[time_index - 1]
        )
        log_sum = tf.reduce_logsumexp(current_log_weight)
        value = value + log_sum - log_particle_count
        previous_log_weight = current_log_weight - log_sum
    return value


def test_24d_posterior_proposal_matches_exact_gaussian_value_and_score() -> None:
    dimension = 24
    model = _LocationLGSSM(dimension)
    theta = tf.constant([0.18, -0.11], DTYPE)
    observation = tf.linspace(
        tf.constant(-0.4, DTYPE), tf.constant(0.6, DTYPE), dimension
    )
    total_variance = model.prior_variance + model.observation_variance
    innovation = observation - theta[0] - theta[1]
    posterior_variance = (
        model.prior_variance * model.observation_variance / total_variance
    )
    posterior_mean = theta[0] + (
        model.prior_variance / total_variance
    ) * innovation

    radius = tf.sqrt(tf.cast(dimension, DTYPE) * posterior_variance)
    directions = radius * tf.eye(dimension, dtype=DTYPE)
    states = tf.concat(
        [posterior_mean[None, :] + directions, posterior_mean[None, :] - directions],
        axis=0,
    )
    particle_count = 2 * dimension
    log_q = _isotropic_log_density(states, posterior_mean, posterior_variance)
    ancestors, log_auxiliary, transition_log_q = _empty_transition_branch(
        particle_count
    )
    branch = prepare_frozen_proposal_branch(
        observations=observation[None, :],
        states=states[None, :, :],
        initial_log_proposal_density=log_q,
        ancestors=ancestors,
        auxiliary_log_probabilities=log_auxiliary,
        transition_log_proposal_density=transition_log_q,
    )
    program = prepare_frozen_proposal_apf_program(model, branch)

    result = program.evaluate(theta)
    exact_value = _isotropic_log_density(
        observation[None, :],
        tf.fill([dimension], theta[0] + theta[1]),
        total_variance,
    )[0]
    exact_component = tf.reduce_sum(innovation) / total_variance
    exact_score = tf.stack([exact_component, exact_component])

    tf.debugging.assert_near(result["log_likelihood"], exact_value, atol=2e-11)
    tf.debugging.assert_near(result["score"], exact_score, atol=2e-11)
    tf.debugging.assert_near(
        result["minimum_ess"], tf.cast(particle_count, DTYPE), atol=2e-10
    )
    assert bool(result["finite"].numpy())
    assert len(branch.branch_id) == 64
    assert len(program.program_id) == 64
    manifest = program.manifest_payload()
    assert manifest["route_classification"] == ROUTE_CLASSIFICATION
    assert manifest["target_class"] == TARGET_CLASS
    assert manifest["pseudo_marginal_exact_target_claimed"] is False

    compiled = program.compiled()
    compiled_result = compiled(theta)
    tf.debugging.assert_near(
        compiled_result["log_likelihood"], result["log_likelihood"], atol=2e-11
    )
    tf.debugging.assert_near(compiled_result["score"], result["score"], atol=2e-11)


def test_multistep_nonuniform_auxiliary_branch_score_matches_same_scalar_fd() -> None:
    model = _LocationLGSSM(3)
    theta = tf.constant([0.14, -0.08], DTYPE)
    states = tf.constant(
        [
            [[-0.8, 0.1, 0.5], [0.2, -0.4, 0.9], [0.7, 0.3, -0.2], [1.0, -0.6, 0.4]],
            [[-0.3, 0.5, 0.7], [0.6, -0.1, 0.2], [0.9, 0.8, -0.5], [-0.4, -0.7, 0.1]],
            [[0.1, 0.9, 0.3], [0.8, 0.2, -0.4], [-0.2, -0.5, 0.6], [1.1, -0.3, 0.0]],
        ],
        DTYPE,
    )
    observations = tf.constant(
        [[0.2, -0.1, 0.4], [0.5, 0.3, -0.2], [0.7, -0.4, 0.1]], DTYPE
    )
    ancestors = tf.constant([[3, 0, 2, 2], [1, 3, 0, 1]], tf.int32)
    auxiliary = tf.constant(
        [[0.1, 0.2, 0.3, 0.4], [0.35, 0.15, 0.25, 0.25]], DTYPE
    )
    initial_log_q = _isotropic_log_density(
        states[0], tf.zeros([3], DTYPE), 1.6
    )
    transition_log_q = tf.stack(
        [
            _isotropic_log_density(states[1], tf.zeros([3], DTYPE), 1.4),
            _isotropic_log_density(states[2], tf.zeros([3], DTYPE), 1.9),
        ]
    )
    branch = prepare_frozen_proposal_branch(
        observations=observations,
        states=states,
        initial_log_proposal_density=initial_log_q,
        ancestors=ancestors,
        auxiliary_log_probabilities=tf.math.log(auxiliary),
        transition_log_proposal_density=transition_log_q,
    )
    program = prepare_frozen_proposal_apf_program(model, branch)
    result = program.evaluate(theta)
    direct_value = _direct_fixed_branch_scalar(
        model,
        theta,
        observations=observations,
        states=states,
        initial_log_q=initial_log_q,
        ancestors=ancestors,
        auxiliary_log_probabilities=tf.math.log(auxiliary),
        transition_log_q=transition_log_q,
    )

    step = tf.constant(1e-5, DTYPE)
    fd_entries = []
    for parameter_index in range(model.parameter_dim()):
        direction = tf.one_hot(parameter_index, model.parameter_dim(), dtype=DTYPE)
        plus = program.evaluate(theta + step * direction)["log_likelihood"]
        minus = program.evaluate(theta - step * direction)["log_likelihood"]
        fd_entries.append((plus - minus) / (2.0 * step))
    finite_difference = tf.stack(fd_entries)

    tf.debugging.assert_near(result["log_likelihood"], direct_value, atol=2e-12)
    tf.debugging.assert_near(result["score"], finite_difference, atol=2e-8, rtol=2e-8)
    tf.debugging.assert_near(
        result["score"], tf.reduce_sum(result["increment_scores"], axis=0), atol=2e-12
    )
    tf.debugging.assert_near(
        tf.reduce_logsumexp(result["final_log_weights"]), 0.0, atol=2e-12
    )
    assert bool(result["finite"].numpy())

    compiled_result = program.compiled()(theta)
    tf.debugging.assert_near(
        compiled_result["log_likelihood"], result["log_likelihood"], atol=2e-11
    )
    tf.debugging.assert_near(compiled_result["score"], result["score"], atol=2e-11)


def test_preparation_fails_closed_for_invalid_law_and_singular_model() -> None:
    model = _LocationLGSSM(2)
    states = tf.zeros([2, 3, 2], DTYPE)
    observations = tf.zeros([2, 2], DTYPE)
    initial_log_q = tf.zeros([3], DTYPE)
    ancestors = tf.constant([[0, 1, 2]], tf.int32)
    transition_log_q = tf.zeros([1, 3], DTYPE)

    with pytest.raises(ValueError, match="categorical law"):
        prepare_frozen_proposal_branch(
            observations=observations,
            states=states,
            initial_log_proposal_density=initial_log_q,
            ancestors=ancestors,
            auxiliary_log_probabilities=tf.math.log(
                tf.constant([[0.2, 0.2, 0.2]], DTYPE)
            ),
            transition_log_proposal_density=transition_log_q,
        )

    branch = prepare_frozen_proposal_branch(
        observations=observations,
        states=states,
        initial_log_proposal_density=initial_log_q,
        ancestors=ancestors,
        auxiliary_log_probabilities=tf.math.log(
            tf.constant([[0.2, 0.3, 0.5]], DTYPE)
        ),
        transition_log_proposal_density=transition_log_q,
    )
    singular_model = _LocationLGSSM(2, measure_id="singular_full_state_delta_v1")
    with pytest.raises(ValueError, match="innovation-coordinate"):
        prepare_frozen_proposal_apf_program(singular_model, branch)
