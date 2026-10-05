"""TensorFlow structural AR(1) quadratic-completion fixture."""

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
class StructuralAR1QuadraticTFFixture:
    name: str
    rho: tf.Tensor
    sigma: tf.Tensor
    a: tf.Tensor
    b: tf.Tensor
    c: tf.Tensor
    d: tf.Tensor
    lam: tf.Tensor
    observation_scale: tf.Tensor
    m0_mean: tf.Tensor
    m0_variance: tf.Tensor
    k0: tf.Tensor
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
            "state_equation": "m_t=rho m_{t-1}+sigma eps_t; k_t=a k_{t-1}+b m_t+c m_t^2+d m_{t-1} m_t",
            "observation_equation": "y_t = k_t + lambda m_t + eta_t",
            "rho": _float(self.rho),
            "sigma": _float(self.sigma),
            "a": _float(self.a),
            "b": _float(self.b),
            "c": _float(self.c),
            "d": _float(self.d),
            "lambda": _float(self.lam),
            "observation_scale": _float(self.observation_scale),
            "horizon": self.horizon,
            "fixture_generation_seed": self.fixture_generation_seed,
            "model_checksum": self.model_checksum,
            "observation_checksum": self.observation_checksum,
            "structural_status": "toy non-DSGE structural split fixture",
            "backend": "tensorflow_tensorflow_probability",
        }


def build_structural_ar1_quadratic_fixture_tf(
    *,
    horizon: int = 16,
    fixture_generation_seed: int = 2026052902,
    name: str = "structural_ar1_quadratic_completion_smoke",
    c_value: float = 0.16,
    d_value: float = -0.08,
) -> StructuralAR1QuadraticTFFixture:
    rho = tf.constant(0.82, dtype=DTYPE)
    sigma = tf.constant(0.24, dtype=DTYPE)
    a = tf.constant(0.55, dtype=DTYPE)
    b = tf.constant(0.65, dtype=DTYPE)
    c = tf.constant(c_value, dtype=DTYPE)
    d = tf.constant(d_value, dtype=DTYPE)
    lam = tf.constant(0.35, dtype=DTYPE)
    observation_scale = tf.constant(0.18, dtype=DTYPE)
    m0_mean = tf.constant(0.1, dtype=DTYPE)
    m0_variance = tf.constant(0.12, dtype=DTYPE)
    k0 = tf.constant(-0.05, dtype=DTYPE)
    states, observations = _simulate_structural(
        rho=rho,
        sigma=sigma,
        a=a,
        b=b,
        c=c,
        d=d,
        lam=lam,
        observation_scale=observation_scale,
        m0_mean=m0_mean,
        m0_variance=m0_variance,
        k0=k0,
        horizon=horizon,
        seed=fixture_generation_seed,
    )
    model_payload = {
        "name": name,
        "rho": _float(rho),
        "sigma": _float(sigma),
        "a": _float(a),
        "b": _float(b),
        "c": _float(c),
        "d": _float(d),
        "lambda": _float(lam),
        "observation_scale": _float(observation_scale),
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
    return StructuralAR1QuadraticTFFixture(
        name=name,
        rho=rho,
        sigma=sigma,
        a=a,
        b=b,
        c=c,
        d=d,
        lam=lam,
        observation_scale=observation_scale,
        m0_mean=m0_mean,
        m0_variance=m0_variance,
        k0=k0,
        states=states,
        observations=observations,
        fixture_generation_seed=fixture_generation_seed,
        model_checksum=model_checksum,
        observation_checksum=observation_checksum,
    )


def complete_k_tf(
    *,
    previous_k: tf.Tensor,
    previous_m: tf.Tensor,
    current_m: tf.Tensor,
    a: tf.Tensor,
    b: tf.Tensor,
    c: tf.Tensor,
    d: tf.Tensor,
) -> tf.Tensor:
    current_m = tf.cast(current_m, DTYPE)
    return (
        tf.cast(a, DTYPE) * tf.cast(previous_k, DTYPE)
        + tf.cast(b, DTYPE) * current_m
        + tf.cast(c, DTYPE) * current_m * current_m
        + tf.cast(d, DTYPE) * tf.cast(previous_m, DTYPE) * current_m
    )


def structural_observation_mean_tf(state: tf.Tensor, lam: tf.Tensor) -> tf.Tensor:
    state = tf.cast(state, DTYPE)
    return state[..., 1] + tf.cast(lam, DTYPE) * state[..., 0]


def stable_digest(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _simulate_structural(
    *,
    rho: tf.Tensor,
    sigma: tf.Tensor,
    a: tf.Tensor,
    b: tf.Tensor,
    c: tf.Tensor,
    d: tf.Tensor,
    lam: tf.Tensor,
    observation_scale: tf.Tensor,
    m0_mean: tf.Tensor,
    m0_variance: tf.Tensor,
    k0: tf.Tensor,
    horizon: int,
    seed: int,
) -> tuple[tf.Tensor, tf.Tensor]:
    m = tfd.Normal(loc=m0_mean, scale=tf.sqrt(m0_variance)).sample(seed=_seed_pair(seed, 1))
    k = tf.cast(k0, DTYPE)
    states = []
    observations = []
    for t in range(horizon):
        prev_m = m
        prev_k = k
        eps = tfd.Normal(loc=tf.constant(0.0, DTYPE), scale=tf.constant(1.0, DTYPE)).sample(
            seed=_seed_pair(seed, 10 + t)
        )
        m = rho * prev_m + sigma * eps
        k = complete_k_tf(
            previous_k=prev_k,
            previous_m=prev_m,
            current_m=m,
            a=a,
            b=b,
            c=c,
            d=d,
        )
        obs_noise = tfd.Normal(loc=tf.constant(0.0, DTYPE), scale=observation_scale).sample(
            seed=_seed_pair(seed, 200 + t)
        )
        y = structural_observation_mean_tf(tf.stack([m, k], axis=0), lam) + obs_noise
        states.append(tf.stack([m, k], axis=0))
        observations.append(y)
    return tf.stack(states, axis=0), tf.stack(observations, axis=0)


def _seed_pair(seed: int, salt: int) -> tf.Tensor:
    return tf.constant([int(seed) % 2147483647, int(salt) % 2147483647], dtype=tf.int32)


def _float(value: tf.Tensor) -> float:
    return float(tf.cast(value, DTYPE).numpy())


def _tensor_to_nested_float(value: tf.Tensor) -> Any:
    return tf.cast(value, DTYPE).numpy().tolist()
