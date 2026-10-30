"""TensorFlow range-bearing fixture for experimental OT-DPF diagnostics."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import tensorflow as tf
import tensorflow_probability as tfp


tfd = tfp.distributions
DTYPE = tf.float64
PI = tf.constant(3.141592653589793, dtype=DTYPE)


@dataclass(frozen=True)
class RangeBearingTFFixture:
    name: str
    A: tf.Tensor
    Q: tf.Tensor
    R: tf.Tensor
    m0: tf.Tensor
    P0: tf.Tensor
    states: tf.Tensor
    observations: tf.Tensor
    dt: float
    fixture_generation_seed: int
    model_checksum: str
    observation_checksum: str

    @property
    def state_dim(self) -> int:
        return int(self.A.shape[0])

    @property
    def obs_dim(self) -> int:
        return int(self.R.shape[0])

    @property
    def horizon(self) -> int:
        return int(self.observations.shape[0])

    def model_definition(self) -> dict[str, Any]:
        return {
            "model_id": self.name,
            "state_equation": "x_t = A x_{t-1} + q_t, q_t ~ Normal(0, Q)",
            "observation_equation": "y_t = [sqrt(px^2+py^2), atan2(py,px)] + r_t",
            "initial_distribution": "x_0 ~ Normal(m0, P0)",
            "A": _tensor_to_nested_float(self.A),
            "Q": _tensor_to_nested_float(self.Q),
            "R": _tensor_to_nested_float(self.R),
            "m0": _tensor_to_nested_float(self.m0),
            "P0": _tensor_to_nested_float(self.P0),
            "horizon": self.horizon,
            "fixture_generation_seed": self.fixture_generation_seed,
            "model_checksum": self.model_checksum,
            "observation_checksum": self.observation_checksum,
            "reference_status": "UKF is approximate for this nonlinear fixture.",
            "backend": "tensorflow_tensorflow_probability",
        }


def build_range_bearing_fixture_tf(
    name: str = "range_bearing_gaussian_tf_tfp_moderate",
) -> RangeBearingTFFixture:
    if name != "range_bearing_gaussian_tf_tfp_moderate":
        raise ValueError("only range_bearing_gaussian_tf_tfp_moderate is authorized")
    dt = 0.1
    A = tf.constant(
        [
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=DTYPE,
    )
    Q = tf.linalg.diag(tf.constant([0.0015, 0.0015, 0.0008, 0.0008], dtype=DTYPE))
    sigma_range = tf.constant(0.12, dtype=DTYPE)
    sigma_bearing = tf.constant(0.04, dtype=DTYPE)
    R = tf.linalg.diag(tf.stack([sigma_range * sigma_range, sigma_bearing * sigma_bearing]))
    m0 = tf.constant([1.2, 0.7, 0.18, -0.06], dtype=DTYPE)
    P0 = tf.linalg.diag(tf.constant([0.04, 0.04, 0.01, 0.01], dtype=DTYPE))
    seed = 701
    states, observations = _simulate_range_bearing(
        A=A,
        Q=Q,
        R=R,
        m0=m0,
        P0=P0,
        horizon=20,
        seed=seed,
    )
    model_payload = {
        "name": name,
        "A": _tensor_to_nested_float(A),
        "Q": _tensor_to_nested_float(Q),
        "R": _tensor_to_nested_float(R),
        "m0": _tensor_to_nested_float(m0),
        "P0": _tensor_to_nested_float(P0),
        "horizon": 20,
        "seed": seed,
        "backend": "tensorflow_tensorflow_probability",
    }
    model_checksum = stable_digest(model_payload)
    observation_checksum = stable_digest(
        {
            "states": _tensor_to_nested_float(states),
            "observations": _tensor_to_nested_float(observations),
        }
    )
    return RangeBearingTFFixture(
        name=name,
        A=A,
        Q=Q,
        R=R,
        m0=m0,
        P0=P0,
        states=states,
        observations=observations,
        dt=dt,
        fixture_generation_seed=seed,
        model_checksum=model_checksum,
        observation_checksum=observation_checksum,
    )


def sample_initial_particles_tf(
    fixture: RangeBearingTFFixture,
    *,
    num_particles: int,
    seed: int,
) -> tf.Tensor:
    dist = tfd.MultivariateNormalTriL(loc=fixture.m0, scale_tril=_chol(fixture.P0))
    return dist.sample(num_particles, seed=_seed_pair(seed, 17))


def sample_transition_particles_tf(
    fixture: RangeBearingTFFixture,
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
        seed=_seed_pair(seed, 2000 + int(time_index)),
    )
    return mean + noise


def observation_log_density_tf(
    fixture: RangeBearingTFFixture,
    particles: tf.Tensor,
    observation: tf.Tensor,
) -> tf.Tensor:
    predicted = range_bearing_observation_tf(particles)
    residual = observation_residual_tf(predicted, observation)
    return gaussian_logpdf_zero_mean_tf(residual, fixture.R)


def range_bearing_observation_tf(x: tf.Tensor, *, eps: float = 1e-12) -> tf.Tensor:
    x = tf.cast(x, DTYPE)
    px = x[..., 0]
    py = x[..., 1]
    rng = tf.sqrt(px * px + py * py + tf.constant(eps, dtype=DTYPE))
    bearing = tf.atan2(py, px)
    return tf.stack([rng, bearing], axis=-1)


def observation_residual_tf(predicted: tf.Tensor, observed: tf.Tensor) -> tf.Tensor:
    residual = tf.cast(observed, DTYPE) - tf.cast(predicted, DTYPE)
    first = residual[..., :1]
    second = wrap_angle_tf(residual[..., 1:2])
    return tf.concat([first, second], axis=-1)


def wrap_angle_tf(value: tf.Tensor) -> tf.Tensor:
    return tf.math.floormod(tf.cast(value, DTYPE) + PI, 2.0 * PI) - PI


def gaussian_logpdf_zero_mean_tf(residuals: tf.Tensor, covariance: tf.Tensor) -> tf.Tensor:
    residuals = tf.cast(residuals, DTYPE)
    covariance = tf.cast(covariance, DTYPE)
    chol = _chol(covariance)
    solved = tf.linalg.cholesky_solve(chol, tf.transpose(residuals))
    quad = tf.reduce_sum(tf.transpose(solved) * residuals, axis=1)
    dim = tf.cast(tf.shape(covariance)[0], DTYPE)
    logdet = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(chol)))
    return -0.5 * (dim * tf.math.log(2.0 * PI) + logdet + quad)


def stable_digest(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _simulate_range_bearing(
    *,
    A: tf.Tensor,
    Q: tf.Tensor,
    R: tf.Tensor,
    m0: tf.Tensor,
    P0: tf.Tensor,
    horizon: int,
    seed: int,
) -> tuple[tf.Tensor, tf.Tensor]:
    state_dist = tfd.MultivariateNormalTriL(loc=m0, scale_tril=_chol(P0))
    state = state_dist.sample(seed=_seed_pair(seed, 1))
    states = [state]
    observations = []
    q_dist = tfd.MultivariateNormalTriL(
        loc=tf.zeros([tf.shape(A)[0]], dtype=DTYPE),
        scale_tril=_chol(Q),
    )
    r_dist = tfd.MultivariateNormalTriL(
        loc=tf.zeros([tf.shape(R)[0]], dtype=DTYPE),
        scale_tril=_chol(R),
    )
    for t in range(horizon):
        state = tf.linalg.matvec(A, state) + q_dist.sample(seed=_seed_pair(seed, 30 + t))
        obs_mean = range_bearing_observation_tf(state)
        obs = obs_mean + r_dist.sample(seed=_seed_pair(seed, 300 + t))
        obs = tf.stack([obs[0], wrap_angle_tf(obs[1])], axis=0)
        states.append(state)
        observations.append(obs)
    return tf.stack(states, axis=0), tf.stack(observations, axis=0)


def _chol(covariance: tf.Tensor) -> tf.Tensor:
    return tf.linalg.cholesky(tf.cast(covariance, DTYPE))


def _seed_pair(seed: int, salt: int) -> tf.Tensor:
    return tf.constant([int(seed) % 2147483647, int(salt) % 2147483647], dtype=tf.int32)


def _tensor_to_nested_float(value: tf.Tensor) -> Any:
    return tf.cast(value, DTYPE).numpy().tolist()
