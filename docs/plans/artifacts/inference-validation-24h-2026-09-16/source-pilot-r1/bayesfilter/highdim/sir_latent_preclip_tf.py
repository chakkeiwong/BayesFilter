"""Continuous latent representation of the clipped Austria SIR simulator.

The source simulator draws an unclipped Gaussian initial state and clips only
susceptible coordinates after subsequent process-noise draws.  This module
filters the pre-clipping variables so their transition law has an ordinary
Lebesgue density while preserving the simulator's physical-state law.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import tensorflow as tf

from bayesfilter.highdim.models import (
    ParameterizedZhaoCuiSIRSSM,
    SpatialSIRSSM,
    parameterized_zhao_cui_sir_austria_model,
)


DTYPE = tf.float64
LATENT_PRECLIP_TARGET_ID = "zhao_cui_sir_austria_latent_preclip_simulator_law_v1"
LATENT_PRECLIP_REPRESENTATION_CLASS = "extension_or_invention"
LATENT_PRECLIP_CANONICAL_STATUS = "not_yet_contract_e_chol_canonical"


def _as_rows(values: tf.Tensor, width: int, name: str) -> tf.Tensor:
    tensor = tf.convert_to_tensor(values, DTYPE)
    if tensor.shape.rank == 1:
        tensor = tensor[tf.newaxis, :]
    if tensor.shape.rank != 2 or tensor.shape[1] != width:
        raise ValueError(f"{name} must have shape [batch,{width}]")
    return tensor


@dataclass(frozen=True)
class LatentPreclipSIRSSM:
    """Parameterized SIR model whose filtering state is pre-clipping ``z_t``."""

    physical_model: ParameterizedZhaoCuiSIRSSM

    def __post_init__(self) -> None:
        if not isinstance(self.physical_model, ParameterizedZhaoCuiSIRSSM):
            raise TypeError("physical_model must be ParameterizedZhaoCuiSIRSSM")
        if self.physical_model.base_model.process_noise_policy != "clip_susceptible_after_noise":
            raise ValueError("latent pre-clipping target requires the clipped source simulator")

    def parameter_dim(self) -> int:
        return self.physical_model.parameter_dim()

    def state_dim(self) -> int:
        return self.physical_model.state_dim()

    def observation_dim(self) -> int:
        return self.physical_model.observation_dim()

    def physical_state(self, latent_state: tf.Tensor, *, time_index: int) -> tf.Tensor:
        """Map ``z_t`` to source-simulator ``x_t`` with exact source time order."""

        latent = _as_rows(latent_state, self.state_dim(), "latent_state")
        if int(time_index) == 0:
            return latent
        susceptible = tf.maximum(latent[:, 0::2], tf.constant(0.0, DTYPE))
        infectious = latent[:, 1::2]
        return tf.reshape(
            tf.stack([susceptible, infectious], axis=2),
            [tf.shape(latent)[0], self.state_dim()],
        )

    def initial_log_density(self, theta: tf.Tensor, z0: tf.Tensor) -> tf.Tensor:
        return self.physical_model.initial_log_density(theta, z0)

    def initial_log_density_parameter_score(
        self, theta: tf.Tensor, z0: tf.Tensor
    ) -> tf.Tensor:
        return self.physical_model.initial_log_density_parameter_score(theta, z0)

    def transition_mean(
        self, theta: tf.Tensor, z_previous: tf.Tensor, *, time_index: int
    ) -> tf.Tensor:
        if int(time_index) < 1:
            raise ValueError("transition time_index must be at least one")
        previous_physical = self.physical_state(
            z_previous, time_index=int(time_index) - 1
        )
        return self.physical_model.transition_mean(theta, previous_physical)

    def transition_log_density(
        self,
        theta: tf.Tensor,
        z_previous: tf.Tensor,
        z_next: tf.Tensor,
        t: int,
    ) -> tf.Tensor:
        previous_physical = self.physical_state(z_previous, time_index=int(t) - 1)
        return self.physical_model.transition_log_density(
            theta, previous_physical, z_next, t=int(t)
        )

    def transition_log_density_parameter_score(
        self,
        theta: tf.Tensor,
        z_previous: tf.Tensor,
        z_next: tf.Tensor,
        t: int,
    ) -> tf.Tensor:
        previous_physical = self.physical_state(z_previous, time_index=int(t) - 1)
        return self.physical_model.transition_log_density_parameter_score(
            theta, previous_physical, z_next, t=int(t)
        )

    def transition_push_from_standard_normal(
        self,
        theta: tf.Tensor,
        z_previous: tf.Tensor,
        standard_normal_noise: tf.Tensor,
        t: int,
    ) -> tf.Tensor:
        """Return pre-clipping ``z_t``; deliberately do not project the output."""

        noise = _as_rows(standard_normal_noise, self.state_dim(), "standard_normal_noise")
        mean = self.transition_mean(theta, z_previous, time_index=int(t))
        process_chol = tf.linalg.cholesky(
            self.physical_model.scaled_model(theta).process_covariance
        )
        return mean + tf.linalg.matmul(noise, process_chol, transpose_b=True)

    def observation_log_density(
        self,
        theta: tf.Tensor,
        z_t: tf.Tensor,
        y_t: tf.Tensor,
        t: int,
    ) -> tf.Tensor:
        physical = self.physical_state(z_t, time_index=int(t))
        return self.physical_model.observation_log_density(theta, physical, y_t, t=int(t))

    def observation_log_density_parameter_score(
        self,
        theta: tf.Tensor,
        z_t: tf.Tensor,
        y_t: tf.Tensor,
        t: int,
    ) -> tf.Tensor:
        physical = self.physical_state(z_t, time_index=int(t))
        return self.physical_model.observation_log_density_parameter_score(
            theta, physical, y_t, t=int(t)
        )

    def simulate_from_standard_normals(
        self,
        theta: tf.Tensor,
        initial_noise: tf.Tensor,
        transition_noise: tf.Tensor,
        observation_noise: tf.Tensor,
    ) -> Mapping[str, tf.Tensor]:
        """Return paired latent, physical, and observed paths for fixed noise."""

        initial_noise = tf.reshape(
            tf.convert_to_tensor(initial_noise, DTYPE), [self.state_dim()]
        )
        transition_noise = tf.convert_to_tensor(transition_noise, DTYPE)
        observation_noise = tf.convert_to_tensor(observation_noise, DTYPE)
        if transition_noise.shape.rank != 2 or transition_noise.shape[1] != self.state_dim():
            raise ValueError("transition_noise must have shape [T,state_dim]")
        time_steps = transition_noise.shape[0]
        if time_steps is None:
            raise ValueError("transition horizon must be static")
        if observation_noise.shape != (time_steps + 1, self.observation_dim()):
            raise ValueError("observation_noise must have shape [T+1,observation_dim]")

        scaled = self.physical_model.scaled_model(theta)
        initial_chol = tf.linalg.cholesky(scaled.initial_covariance)
        observation_chol = tf.linalg.cholesky(scaled.observation_covariance)
        latent = scaled.initial_mean + tf.linalg.matvec(initial_chol, initial_noise)
        latent_path = [latent]
        physical_path = [latent]
        observations = [
            self.physical_model.infectious_components(latent)[0]
            + tf.linalg.matvec(observation_chol, observation_noise[0])
        ]
        for time_index in range(1, int(time_steps) + 1):
            latent = self.transition_push_from_standard_normal(
                theta,
                latent[tf.newaxis, :],
                transition_noise[time_index - 1][tf.newaxis, :],
                time_index,
            )[0]
            physical = self.physical_state(latent, time_index=time_index)[0]
            latent_path.append(latent)
            physical_path.append(physical)
            observations.append(
                self.physical_model.infectious_components(physical)[0]
                + tf.linalg.matvec(observation_chol, observation_noise[time_index])
            )
        return {
            "latent_path": tf.stack(latent_path),
            "physical_path": tf.stack(physical_path),
            "observations": tf.stack(observations),
        }

    def manifest_payload(self) -> Mapping[str, object]:
        return {
            "target_id": LATENT_PRECLIP_TARGET_ID,
            "family": "LatentPreclipSIRSSM",
            "representation_classification": LATENT_PRECLIP_REPRESENTATION_CLASS,
            "canonical_contract_e_status": LATENT_PRECLIP_CANONICAL_STATUS,
            "filtering_state": "pre_clipping_gaussian_z_t",
            "physical_state_time_order": "x_0=z_0; x_t=clip_susceptible(z_t) for t>=1",
            "clip_jacobian_policy": "none_deterministic_generative_map_not_coordinate_change",
            "parameter_order": (
                "log_kappa_scale",
                "log_nu_scale",
                "log_obs_noise_scale",
            ),
            "state_dimension": self.state_dim(),
            "observation_dimension": self.observation_dim(),
            "source_density_diagnostic_status": "separate_target_not_silently_upgraded",
            "what_is_not_claimed": (
                "source_faithful_author_filtering_implementation",
                "contract_e_chol_canonical_admission",
                "fixed_ttsirt_total_score",
                "full_horizon_correctness",
                "hmc_or_leaderboard_readiness",
            ),
        }


def latent_preclip_zhao_cui_sir_austria_model() -> LatentPreclipSIRSSM:
    return LatentPreclipSIRSSM(parameterized_zhao_cui_sir_austria_model())


def latent_preclip_two_node_spatial_sir_model() -> LatentPreclipSIRSSM:
    """Return the explicit coupled two-node Phase 5 target hypothesis."""

    base = SpatialSIRSSM(
        kappa=tf.constant([0.1, 0.1], DTYPE),
        nu=tf.constant([18.0, 18.0], DTYPE),
        initial_mean=tf.constant([487.0, 13.0, 488.0, 12.0], DTYPE),
        neighbor_sets=((1,), (0,)),
        delta=0.02,
        rk4_internal_step=0.005,
        process_covariance=tf.eye(4, dtype=DTYPE),
        observation_covariance=100.0 * tf.eye(2, dtype=DTYPE),
        initial_covariance=tf.eye(4, dtype=DTYPE),
        rk4_variant="zhao_cui_sir_step",
        process_noise_policy="clip_susceptible_after_noise",
    )
    return LatentPreclipSIRSSM(ParameterizedZhaoCuiSIRSSM(base))


__all__ = [
    "LATENT_PRECLIP_CANONICAL_STATUS",
    "LATENT_PRECLIP_REPRESENTATION_CLASS",
    "LATENT_PRECLIP_TARGET_ID",
    "LatentPreclipSIRSSM",
    "latent_preclip_zhao_cui_sir_austria_model",
    "latent_preclip_two_node_spatial_sir_model",
]
