"""TensorFlow LGSSM fixture for experimental OT-DPF diagnostics."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import tensorflow as tf
import tensorflow_probability as tfp


tfd = tfp.distributions
DTYPE = tf.float64


@dataclass(frozen=True)
class LGSSMTFFixture:
    name: str
    A: tf.Tensor
    C: tf.Tensor
    Q: tf.Tensor
    R: tf.Tensor
    m0: tf.Tensor
    P0: tf.Tensor
    states: tf.Tensor
    observations: tf.Tensor
    fixture_generation_seed: int
    model_checksum: str
    observation_checksum: str

    @property
    def state_dim(self) -> int:
        return int(self.A.shape[0])

    @property
    def obs_dim(self) -> int:
        return int(self.C.shape[0])

    @property
    def horizon(self) -> int:
        return int(self.observations.shape[0])

    def model_definition(self) -> dict[str, Any]:
        return {
            "model_id": self.name,
            "state_equation": "x_t = A x_{t-1} + q_t, q_t ~ Normal(0, Q)",
            "observation_equation": "y_t = C x_t + r_t, r_t ~ Normal(0, R)",
            "initial_distribution": "x_0 ~ Normal(m0, P0)",
            "A": _tensor_to_nested_float(self.A),
            "C": _tensor_to_nested_float(self.C),
            "Q": _tensor_to_nested_float(self.Q),
            "R": _tensor_to_nested_float(self.R),
            "m0": _tensor_to_nested_float(self.m0),
            "P0": _tensor_to_nested_float(self.P0),
            "horizon": self.horizon,
            "fixture_generation_seed": self.fixture_generation_seed,
            "model_checksum": self.model_checksum,
            "observation_checksum": self.observation_checksum,
            "backend": "tensorflow_tensorflow_probability",
        }


def build_lgssm_fixture_tf(
    *,
    horizon: int = 25,
    fixture_generation_seed: int = 2026052801,
) -> LGSSMTFFixture:
    """Build a fixed, small LGSSM fixture with TF stateless sampling."""

    A = tf.constant([[0.92, 0.18], [-0.04, 0.86]], dtype=DTYPE)
    C = tf.constant([[1.0, 0.35]], dtype=DTYPE)
    Q = tf.constant([[0.08, 0.015], [0.015, 0.05]], dtype=DTYPE)
    R = tf.constant([[0.18]], dtype=DTYPE)
    m0 = tf.constant([0.25, -0.15], dtype=DTYPE)
    P0 = tf.constant([[0.45, 0.04], [0.04, 0.32]], dtype=DTYPE)
    states, observations = _simulate_lgssm(
        A=A,
        C=C,
        Q=Q,
        R=R,
        m0=m0,
        P0=P0,
        horizon=horizon,
        seed=fixture_generation_seed,
    )
    model_payload = {
        "name": "lgssm_ot_tf_tfp_smoke",
        "A": _tensor_to_nested_float(A),
        "C": _tensor_to_nested_float(C),
        "Q": _tensor_to_nested_float(Q),
        "R": _tensor_to_nested_float(R),
        "m0": _tensor_to_nested_float(m0),
        "P0": _tensor_to_nested_float(P0),
        "horizon": int(horizon),
        "fixture_generation_seed": int(fixture_generation_seed),
        "backend": "tensorflow_tensorflow_probability",
    }
    model_checksum = stable_digest(model_payload)
    observation_checksum = stable_digest(
        {
            "states": _tensor_to_nested_float(states),
            "observations": _tensor_to_nested_float(observations),
        }
    )
    return LGSSMTFFixture(
        name="lgssm_ot_tf_tfp_smoke",
        A=A,
        C=C,
        Q=Q,
        R=R,
        m0=m0,
        P0=P0,
        states=states,
        observations=observations,
        fixture_generation_seed=int(fixture_generation_seed),
        model_checksum=model_checksum,
        observation_checksum=observation_checksum,
    )


def sample_initial_particles_tf(
    fixture: LGSSMTFFixture,
    *,
    num_particles: int,
    seed: int,
) -> tf.Tensor:
    dist = tfd.MultivariateNormalTriL(loc=fixture.m0, scale_tril=_chol(fixture.P0))
    return dist.sample(num_particles, seed=_seed_pair(seed, 11))


def sample_transition_particles_tf(
    fixture: LGSSMTFFixture,
    previous_particles: tf.Tensor,
    *,
    seed: int,
    time_index: int,
) -> tf.Tensor:
    mean = tf.linalg.matmul(previous_particles, fixture.A, transpose_b=True)
    noise_dist = tfd.MultivariateNormalTriL(
        loc=tf.zeros([fixture.state_dim], dtype=DTYPE),
        scale_tril=_chol(fixture.Q),
    )
    noise = noise_dist.sample(
        tf.shape(previous_particles)[0],
        seed=_seed_pair(seed, 1000 + int(time_index)),
    )
    return mean + noise


def observation_log_density_tf(
    fixture: LGSSMTFFixture,
    particles: tf.Tensor,
    observation: tf.Tensor,
) -> tf.Tensor:
    predicted = tf.linalg.matmul(particles, fixture.C, transpose_b=True)
    residual = tf.reshape(tf.cast(observation, DTYPE), [1, fixture.obs_dim]) - predicted
    return gaussian_logpdf_zero_mean_tf(residual, fixture.R)


def gaussian_logpdf_zero_mean_tf(residuals: tf.Tensor, covariance: tf.Tensor) -> tf.Tensor:
    residuals = tf.cast(residuals, DTYPE)
    covariance = tf.cast(covariance, DTYPE)
    chol = _chol(covariance)
    solved = tf.linalg.cholesky_solve(chol, tf.transpose(residuals))
    quad = tf.reduce_sum(tf.transpose(solved) * residuals, axis=1)
    dim = tf.cast(tf.shape(covariance)[0], DTYPE)
    logdet = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(chol)))
    return -0.5 * (dim * tf.math.log(tf.constant(2.0 * 3.141592653589793, DTYPE)) + logdet + quad)


def stable_digest(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _simulate_lgssm(
    *,
    A: tf.Tensor,
    C: tf.Tensor,
    Q: tf.Tensor,
    R: tf.Tensor,
    m0: tf.Tensor,
    P0: tf.Tensor,
    horizon: int,
    seed: int,
) -> tuple[tf.Tensor, tf.Tensor]:
    state_dist = tfd.MultivariateNormalTriL(loc=m0, scale_tril=_chol(P0))
    x0 = state_dist.sample(seed=_seed_pair(seed, 1))
    states = [x0]
    observations = []
    q_dist = tfd.MultivariateNormalTriL(
        loc=tf.zeros([tf.shape(A)[0]], dtype=DTYPE),
        scale_tril=_chol(Q),
    )
    r_dist = tfd.MultivariateNormalTriL(
        loc=tf.zeros([tf.shape(C)[0]], dtype=DTYPE),
        scale_tril=_chol(R),
    )
    for t in range(horizon):
        mean_state = tf.linalg.matvec(A, states[-1])
        state = mean_state + q_dist.sample(seed=_seed_pair(seed, 10 + t))
        obs_mean = tf.linalg.matvec(C, state)
        obs = obs_mean + r_dist.sample(seed=_seed_pair(seed, 100 + t))
        states.append(state)
        observations.append(obs)
    return tf.stack(states, axis=0), tf.stack(observations, axis=0)


def _chol(covariance: tf.Tensor) -> tf.Tensor:
    return tf.linalg.cholesky(tf.cast(covariance, DTYPE))


def _seed_pair(seed: int, salt: int) -> tf.Tensor:
    return tf.constant([int(seed) % 2147483647, int(salt) % 2147483647], dtype=tf.int32)


def _tensor_to_nested_float(value: tf.Tensor) -> Any:
    return tf.cast(value, DTYPE).numpy().tolist()
