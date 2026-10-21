"""Clean-room stochastic-volatility fixture for DPF smoke diagnostics."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class StochasticVolatilityParameters:
    mu: float = -0.7
    phi: float = 0.95
    sigma: float = 0.25
    beta: float = 0.65

    @property
    def stationary_variance(self) -> float:
        return (self.sigma * self.sigma) / (1.0 - self.phi * self.phi)


@dataclass(frozen=True)
class StochasticVolatilityFixture:
    name: str
    parameters: StochasticVolatilityParameters
    horizon: int
    fixture_generation_seed: int
    latent_states: np.ndarray
    observations: np.ndarray
    model_checksum: str
    observation_checksum: str

    def model_definition(self) -> dict[str, Any]:
        params = self.parameters
        return {
            "model_id": self.name,
            "state_equation": "x_t = mu + phi * (x_{t-1} - mu) + sigma * eta_t",
            "observation_equation": "y_t | x_t ~ Normal(0, beta^2 * exp(x_t))",
            "initial_distribution": "x_0 ~ Normal(mu, sigma^2 / (1 - phi^2))",
            "parameters": {
                "mu": params.mu,
                "phi": params.phi,
                "sigma": params.sigma,
                "beta": params.beta,
                "stationary_variance": params.stationary_variance,
            },
            "horizon": self.horizon,
            "fixture_generation_seed": self.fixture_generation_seed,
            "model_checksum": self.model_checksum,
            "observation_checksum": self.observation_checksum,
        }

    def summary(self) -> dict[str, Any]:
        return {
            "latent_mean": float(np.mean(self.latent_states)),
            "latent_std": float(np.std(self.latent_states)),
            "observation_mean": float(np.mean(self.observations)),
            "observation_std": float(np.std(self.observations)),
            "max_abs_observation": float(np.max(np.abs(self.observations))),
        }


def build_stochastic_volatility_fixture(
    *,
    horizon: int = 30,
    fixture_generation_seed: int = 20260528,
    parameters: StochasticVolatilityParameters | None = None,
) -> StochasticVolatilityFixture:
    params = parameters or StochasticVolatilityParameters()
    rng = np.random.default_rng(fixture_generation_seed)
    latent_states = np.zeros(horizon, dtype=np.float64)
    observations = np.zeros(horizon, dtype=np.float64)
    latent_states[0] = rng.normal(params.mu, math.sqrt(params.stationary_variance))
    observations[0] = _sample_observation(rng, latent_states[0], params.beta)

    for t in range(1, horizon):
        latent_states[t] = params.mu + params.phi * (
            latent_states[t - 1] - params.mu
        ) + params.sigma * rng.normal()
        observations[t] = _sample_observation(rng, latent_states[t], params.beta)

    model_checksum = _stable_digest(
        {
            "name": "sv_smoke_mu_phi_sigma_beta",
            "parameters": {
                "mu": params.mu,
                "phi": params.phi,
                "sigma": params.sigma,
                "beta": params.beta,
            },
            "horizon": horizon,
            "fixture_generation_seed": fixture_generation_seed,
        }
    )
    observation_checksum = _stable_digest(
        {
            "observations": observations.tolist(),
            "latent_states": latent_states.tolist(),
        }
    )
    return StochasticVolatilityFixture(
        name="sv_smoke_mu_phi_sigma_beta",
        parameters=params,
        horizon=horizon,
        fixture_generation_seed=fixture_generation_seed,
        latent_states=latent_states,
        observations=observations,
        model_checksum=model_checksum,
        observation_checksum=observation_checksum,
    )


def transition_sample(
    rng: np.random.Generator,
    previous_states: np.ndarray,
    parameters: StochasticVolatilityParameters,
) -> np.ndarray:
    noise = rng.normal(size=previous_states.shape[0])
    return (
        parameters.mu
        + parameters.phi * (previous_states - parameters.mu)
        + parameters.sigma * noise
    )


def initial_sample(
    rng: np.random.Generator,
    num_particles: int,
    parameters: StochasticVolatilityParameters,
) -> np.ndarray:
    return rng.normal(
        loc=parameters.mu,
        scale=math.sqrt(parameters.stationary_variance),
        size=num_particles,
    )


def observation_log_density(
    observations: float | np.ndarray,
    states: np.ndarray,
    parameters: StochasticVolatilityParameters,
) -> np.ndarray:
    obs = np.asarray(observations, dtype=np.float64)
    log_variance = 2.0 * math.log(parameters.beta) + states
    variance = np.exp(log_variance)
    return -0.5 * (math.log(2.0 * math.pi) + log_variance + (obs * obs) / variance)


def _sample_observation(
    rng: np.random.Generator,
    state: float,
    beta: float,
) -> float:
    return float(rng.normal(0.0, beta * math.exp(0.5 * state)))


def _stable_digest(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()

