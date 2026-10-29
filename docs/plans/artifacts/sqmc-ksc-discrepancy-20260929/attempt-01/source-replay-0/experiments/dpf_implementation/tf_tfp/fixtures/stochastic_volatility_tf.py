"""TensorFlow stochastic-volatility fixture for DPF nonlinear-SSM evidence."""

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
class StochasticVolatilityTFFixture:
    name: str
    mu: tf.Tensor
    phi: tf.Tensor
    sigma: tf.Tensor
    h0_mean: tf.Tensor
    h0_variance: tf.Tensor
    states: tf.Tensor
    observations: tf.Tensor
    fixture_generation_seed: int
    model_checksum: str
    observation_checksum: str

    @property
    def horizon(self) -> int:
        return int(self.observations.shape[0])

    def model_definition(self) -> dict[str, Any]:
        return {
            "model_id": self.name,
            "state_equation": "h_t = mu + phi (h_{t-1} - mu) + sigma eta_t",
            "observation_equation": "y_t | h_t ~ Normal(0, exp(h_t))",
            "mu": _float(self.mu),
            "phi": _float(self.phi),
            "sigma": _float(self.sigma),
            "h0_mean": _float(self.h0_mean),
            "h0_variance": _float(self.h0_variance),
            "horizon": self.horizon,
            "fixture_generation_seed": self.fixture_generation_seed,
            "model_checksum": self.model_checksum,
            "observation_checksum": self.observation_checksum,
            "backend": "tensorflow_tensorflow_probability",
        }


def build_stochastic_volatility_fixture_tf(
    *,
    horizon: int = 18,
    fixture_generation_seed: int = 2026052901,
) -> StochasticVolatilityTFFixture:
    mu = tf.constant(-0.7, dtype=DTYPE)
    phi = tf.constant(0.92, dtype=DTYPE)
    sigma = tf.constant(0.28, dtype=DTYPE)
    h0_mean = mu
    h0_variance = sigma * sigma / (1.0 - phi * phi)
    states, observations = _simulate_sv(
        mu=mu,
        phi=phi,
        sigma=sigma,
        h0_mean=h0_mean,
        h0_variance=h0_variance,
        horizon=horizon,
        seed=fixture_generation_seed,
    )
    model_payload = {
        "name": "sv_cut4_ledh_smoke",
        "mu": _float(mu),
        "phi": _float(phi),
        "sigma": _float(sigma),
        "h0_mean": _float(h0_mean),
        "h0_variance": _float(h0_variance),
        "horizon": int(horizon),
        "fixture_generation_seed": int(fixture_generation_seed),
    }
    model_checksum = stable_digest(model_payload)
    observation_checksum = stable_digest(
        {
            "states": _tensor_to_nested_float(states),
            "observations": _tensor_to_nested_float(observations),
        }
    )
    return StochasticVolatilityTFFixture(
        name="sv_cut4_ledh_smoke",
        mu=mu,
        phi=phi,
        sigma=sigma,
        h0_mean=h0_mean,
        h0_variance=h0_variance,
        states=states,
        observations=observations,
        fixture_generation_seed=fixture_generation_seed,
        model_checksum=model_checksum,
        observation_checksum=observation_checksum,
    )


def sv_transition_mean_tf(
    previous_h: tf.Tensor,
    *,
    mu: tf.Tensor,
    phi: tf.Tensor,
) -> tf.Tensor:
    return tf.cast(mu, DTYPE) + tf.cast(phi, DTYPE) * (tf.cast(previous_h, DTYPE) - tf.cast(mu, DTYPE))


def sv_observation_log_density_tf(h: tf.Tensor, observation: tf.Tensor) -> tf.Tensor:
    h = tf.cast(h, DTYPE)
    y = tf.cast(observation, DTYPE)
    variance = tf.exp(h)
    return -0.5 * (
        tf.math.log(tf.constant(2.0 * 3.141592653589793, dtype=DTYPE))
        + h
        + y * y / variance
    )


def stable_digest(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _simulate_sv(
    *,
    mu: tf.Tensor,
    phi: tf.Tensor,
    sigma: tf.Tensor,
    h0_mean: tf.Tensor,
    h0_variance: tf.Tensor,
    horizon: int,
    seed: int,
) -> tuple[tf.Tensor, tf.Tensor]:
    h = tfd.Normal(loc=h0_mean, scale=tf.sqrt(h0_variance)).sample(seed=_seed_pair(seed, 1))
    states = []
    observations = []
    for t in range(horizon):
        eta = tfd.Normal(loc=tf.constant(0.0, DTYPE), scale=tf.constant(1.0, DTYPE)).sample(
            seed=_seed_pair(seed, 10 + t)
        )
        h = sv_transition_mean_tf(h, mu=mu, phi=phi) + sigma * eta
        eps = tfd.Normal(loc=tf.constant(0.0, DTYPE), scale=tf.constant(1.0, DTYPE)).sample(
            seed=_seed_pair(seed, 200 + t)
        )
        y = tf.exp(0.5 * h) * eps
        states.append(h)
        observations.append(y)
    return tf.stack(states, axis=0), tf.stack(observations, axis=0)


def _seed_pair(seed: int, salt: int) -> tf.Tensor:
    return tf.constant([int(seed) % 2147483647, int(salt) % 2147483647], dtype=tf.int32)


def _float(value: tf.Tensor) -> float:
    return float(tf.cast(value, DTYPE).numpy())


def _tensor_to_nested_float(value: tf.Tensor) -> Any:
    return tf.cast(value, DTYPE).numpy().tolist()
