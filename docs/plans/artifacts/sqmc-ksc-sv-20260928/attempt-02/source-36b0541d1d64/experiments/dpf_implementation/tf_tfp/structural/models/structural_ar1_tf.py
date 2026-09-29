"""Structural AR(1) model implementing the experimental structural contract."""

from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf

from experiments.dpf_implementation.tf_tfp.fixtures.structural_ar1_quadratic_tf import (
    DTYPE,
    StructuralAR1QuadraticTFFixture,
    complete_k_tf,
    structural_observation_mean_tf,
)
from experiments.dpf_implementation.tf_tfp.structural.contracts_tf import StructuralFlowProposalTF


@dataclass(frozen=True)
class StructuralAR1ModelTF:
    fixture: StructuralAR1QuadraticTFFixture
    b: tf.Tensor
    model_id: str = "structural_ar1_completion_contract_tf"

    @property
    def z_dim(self) -> int:
        return 1

    @property
    def s_dim(self) -> int:
        return 1

    def initial_z_sample(self, num_particles: int, seed: int) -> tf.Tensor:
        draws = tf.random.stateless_normal(
            [num_particles, 1],
            seed=_seed_pair(seed, 1),
            dtype=DTYPE,
        )
        return self.fixture.m0_mean + tf.sqrt(self.fixture.m0_variance) * draws

    def initial_z_log_prob(self, z: tf.Tensor) -> tf.Tensor:
        return _normal_logpdf(tf.reshape(tf.cast(z, DTYPE), [-1]) - self.fixture.m0_mean, tf.sqrt(self.fixture.m0_variance))

    def initial_s_from_z(self, z: tf.Tensor) -> tf.Tensor:
        count = tf.shape(tf.cast(z, DTYPE))[0]
        return tf.fill([count, 1], tf.cast(self.fixture.k0, DTYPE))

    def transition_z_sample(self, previous_z: tf.Tensor, seed: int, time_index: int) -> tf.Tensor:
        noise = tf.random.stateless_normal(
            tf.shape(tf.cast(previous_z, DTYPE)),
            seed=_seed_pair(seed, 1000 + time_index),
            dtype=DTYPE,
        )
        return self.fixture.rho * tf.cast(previous_z, DTYPE) + self.fixture.sigma * noise

    def transition_z_log_prob(self, current_z: tf.Tensor, previous_z: tf.Tensor, time_index: int) -> tf.Tensor:
        del time_index
        mean = self.fixture.rho * tf.cast(previous_z, DTYPE)
        return _normal_logpdf(tf.reshape(tf.cast(current_z, DTYPE) - mean, [-1]), self.fixture.sigma)

    def complete_s(self, previous_s: tf.Tensor, previous_z: tf.Tensor, current_z: tf.Tensor) -> tf.Tensor:
        completed = complete_k_tf(
            previous_k=tf.cast(previous_s, DTYPE),
            previous_m=tf.cast(previous_z, DTYPE),
            current_m=tf.cast(current_z, DTYPE),
            a=self.fixture.a,
            b=tf.cast(self.b, DTYPE),
            c=self.fixture.c,
            d=self.fixture.d,
        )
        return tf.reshape(completed, [-1, 1])

    def observation_log_prob(
        self,
        current_z: tf.Tensor,
        current_s: tf.Tensor,
        observation: tf.Tensor,
        time_index: int,
    ) -> tf.Tensor:
        del time_index
        state = tf.concat([tf.cast(current_z, DTYPE), tf.cast(current_s, DTYPE)], axis=1)
        mean = structural_observation_mean_tf(state, self.fixture.lam)
        return _normal_logpdf(tf.cast(observation, DTYPE) - mean, self.fixture.observation_scale)

    def completion_residual(
        self,
        previous_s: tf.Tensor,
        previous_z: tf.Tensor,
        current_z: tf.Tensor,
        current_s: tf.Tensor,
    ) -> tf.Tensor:
        return tf.cast(current_s, DTYPE) - self.complete_s(previous_s, previous_z, current_z)

    def local_flow_proposal(
        self,
        pre_z: tf.Tensor,
        previous_z: tf.Tensor,
        previous_s: tf.Tensor,
        observation: tf.Tensor,
        time_index: int,
    ) -> StructuralFlowProposalTF:
        del time_index
        prior_var = self.fixture.sigma * self.fixture.sigma
        obs_var = self.fixture.observation_scale * self.fixture.observation_scale
        prior_mean = self.fixture.rho * tf.cast(previous_z, DTYPE)
        pre_s = self.complete_s(previous_s, previous_z, pre_z)
        state = tf.concat([tf.cast(pre_z, DTYPE), pre_s], axis=1)
        predicted = tf.reshape(structural_observation_mean_tf(state, self.fixture.lam), [-1, 1])
        jacobian = (
            tf.cast(self.b, DTYPE)
            + 2.0 * self.fixture.c * tf.cast(pre_z, DTYPE)
            + self.fixture.d * tf.cast(previous_z, DTYPE)
            + self.fixture.lam
        )
        intercept = predicted - jacobian * tf.cast(pre_z, DTYPE)
        post_var = 1.0 / (1.0 / prior_var + jacobian * jacobian / obs_var)
        post_mean = post_var * (prior_mean / prior_var + jacobian * (tf.cast(observation, DTYPE) - intercept) / obs_var)
        scale = tf.sqrt(tf.maximum(post_var, tf.constant(1e-12, DTYPE)) / prior_var)
        post_z = post_mean + scale * (tf.cast(pre_z, DTYPE) - prior_mean)
        forward_log_det = tf.reshape(tf.math.log(scale), [-1])
        pre_flow_log_density = _normal_logpdf(tf.reshape(tf.cast(pre_z, DTYPE) - prior_mean, [-1]), tf.sqrt(prior_var))
        return StructuralFlowProposalTF(
            post_z=tf.reshape(post_z, [-1, 1]),
            pre_flow_log_density=pre_flow_log_density,
            forward_log_det=forward_log_det,
            diagnostics={
                "proposal_id": "local_gaussian_ledh_stochastic_z_only",
                "min_abs_observation_jacobian_z": _float(tf.reduce_min(tf.abs(jacobian))),
                "min_flow_scale_z": _float(tf.reduce_min(scale)),
                "max_flow_scale_z": _float(tf.reduce_max(scale)),
            },
        )


def _normal_logpdf(residual: tf.Tensor, scale: tf.Tensor) -> tf.Tensor:
    residual = tf.cast(residual, DTYPE)
    scale = tf.cast(scale, DTYPE)
    variance = scale * scale
    return -0.5 * (
        tf.math.log(tf.constant(2.0 * 3.141592653589793, DTYPE) * variance)
        + residual * residual / variance
    )


def _seed_pair(seed: int, salt: int) -> tf.Tensor:
    return tf.constant([int(seed) % 2147483647, int(salt) % 2147483647], dtype=tf.int32)


def _float(value: tf.Tensor) -> float:
    return float(tf.cast(value, DTYPE).numpy())
