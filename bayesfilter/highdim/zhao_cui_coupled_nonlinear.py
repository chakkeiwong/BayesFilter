"""Coupled nonlinear diagnostic model for the Zhao-Cui APF rung-2 lane.

This is a small BayesFilter target used to test within-block nonlinear TT
proposal fitting.  It is not an author-model reproduction and is deliberately
kept separate from the repository's source-route model registry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import tensorflow as tf

from bayesfilter.highdim.zhao_cui_frozen_proposal_apf_tf import (
    MEASURE_ID,
    SCORE_BACKEND_ID,
)


DTYPE = tf.float64
ONLINE_DTYPE = tf.float32
LOG_TWO_PI = 1.8378770664093453


@dataclass(frozen=True)
class ShiftedAlgebraicCoordinateMap:
    """Full-support algebraic map with a fixed physical-space location."""

    locations: tf.Tensor
    scales: tf.Tensor

    def __post_init__(self) -> None:
        locations = tf.reshape(tf.convert_to_tensor(self.locations, DTYPE), [-1])
        scales = tf.reshape(tf.convert_to_tensor(self.scales, DTYPE), [-1])
        if locations.shape[0] is None or locations.shape != scales.shape:
            raise ValueError("locations and scales must have the same nonempty shape")
        if not bool(
            tf.reduce_all(tf.math.is_finite(locations)).numpy()
            and tf.reduce_all(tf.math.is_finite(scales)).numpy()
            and tf.reduce_all(scales > 0.0).numpy()
        ):
            raise ValueError("locations and scales must be finite with positive scales")
        object.__setattr__(self, "locations", locations)
        object.__setattr__(self, "scales", scales)

    @property
    def dimension(self) -> int:
        return int(self.locations.shape[0])

    def _reference_points(self, reference_points: tf.Tensor) -> tf.Tensor:
        reference = tf.convert_to_tensor(reference_points, DTYPE)
        if reference.shape.rank != 2 or reference.shape[1] != self.dimension:
            raise ValueError("reference_points must have shape [sample,dimension]")
        return tf.clip_by_value(reference, -1.0 + 1e-12, 1.0 - 1e-12)

    def _physical_points(self, physical_points: tf.Tensor) -> tf.Tensor:
        physical = tf.convert_to_tensor(physical_points, DTYPE)
        if physical.shape.rank != 2 or physical.shape[1] != self.dimension:
            raise ValueError("physical_points must have shape [sample,dimension]")
        return physical

    def forward_log_det_components(self, reference_points: tf.Tensor) -> tf.Tensor:
        """Return per-axis ``log |dx_j / dz_j|`` terms."""

        clipped = self._reference_points(reference_points)
        return (
            tf.math.log(self.scales)[None, :]
            - 1.5 * tf.math.log(1.0 - tf.square(clipped))
        )

    def forward(self, reference_points: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        clipped = self._reference_points(reference_points)
        one_minus_square = 1.0 - tf.square(clipped)
        physical = (
            self.locations[None, :]
            + self.scales[None, :] * clipped * tf.math.rsqrt(one_minus_square)
        )
        log_det = tf.reduce_sum(
            self.forward_log_det_components(clipped), axis=1
        )
        return physical, log_det

    def inverse_log_det_components(self, physical_points: tf.Tensor) -> tf.Tensor:
        """Return per-axis ``log |dz_j / dx_j|`` terms."""

        physical = self._physical_points(physical_points)
        scaled = (physical - self.locations[None, :]) / self.scales[None, :]
        return (
            -tf.math.log(self.scales)[None, :]
            - 1.5 * tf.math.log1p(tf.square(scaled))
        )

    def inverse(self, physical_points: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        physical = self._physical_points(physical_points)
        scaled = (physical - self.locations[None, :]) / self.scales[None, :]
        reference = scaled * tf.math.rsqrt(1.0 + tf.square(scaled))
        log_det = tf.reduce_sum(
            self.inverse_log_det_components(physical), axis=1
        )
        return reference, log_det

    def manifest_payload(self) -> Mapping[str, object]:
        return {
            "family": "ShiftedAlgebraicCoordinateMap",
            "route_classification": "extension_or_invention",
            "locations": self.locations,
            "scales": self.scales,
            "formula": "x=location+scale*z/sqrt(1-z^2)",
            "measure_role": "reference_to_physical_jacobian_explicit",
        }


@dataclass(frozen=True)
class GaussianQuantileCoordinateMap:
    """Full-support Gaussian-quantile map from ``[-1, 1]`` to physical space.

    A uniform reference coordinate is transformed to a Gaussian defensive
    proposal.  The map keeps tail states bounded at the fixed quadrature and
    particle resolutions used by this diagnostic while retaining full support.
    """

    locations: tf.Tensor
    scales: tf.Tensor

    def __post_init__(self) -> None:
        locations = tf.reshape(tf.convert_to_tensor(self.locations, DTYPE), [-1])
        scales = tf.reshape(tf.convert_to_tensor(self.scales, DTYPE), [-1])
        if locations.shape[0] is None or locations.shape != scales.shape:
            raise ValueError("locations and scales must have the same nonempty shape")
        if not bool(
            tf.reduce_all(tf.math.is_finite(locations)).numpy()
            and tf.reduce_all(tf.math.is_finite(scales)).numpy()
            and tf.reduce_all(scales > 0.0).numpy()
        ):
            raise ValueError("locations and scales must be finite with positive scales")
        object.__setattr__(self, "locations", locations)
        object.__setattr__(self, "scales", scales)

    @property
    def dimension(self) -> int:
        return int(self.locations.shape[0])

    def _reference_points(self, reference_points: tf.Tensor) -> tf.Tensor:
        reference = tf.convert_to_tensor(reference_points, DTYPE)
        if reference.shape.rank != 2 or reference.shape[1] != self.dimension:
            raise ValueError("reference_points must have shape [sample,dimension]")
        return tf.clip_by_value(reference, -1.0 + 1e-12, 1.0 - 1e-12)

    def _physical_points(self, physical_points: tf.Tensor) -> tf.Tensor:
        physical = tf.convert_to_tensor(physical_points, DTYPE)
        if physical.shape.rank != 2 or physical.shape[1] != self.dimension:
            raise ValueError("physical_points must have shape [sample,dimension]")
        return physical

    def _standard_normal_quantiles(self, reference_points: tf.Tensor) -> tf.Tensor:
        clipped = self._reference_points(reference_points)
        return tf.sqrt(tf.constant(2.0, DTYPE)) * tf.math.erfinv(clipped)

    def forward_log_det_components(self, reference_points: tf.Tensor) -> tf.Tensor:
        quantiles = self._standard_normal_quantiles(reference_points)
        return (
            tf.math.log(self.scales)[None, :]
            + 0.5 * tf.math.log(tf.constant(3.141592653589793 / 2.0, DTYPE))
            + 0.5 * tf.square(quantiles)
        )

    def forward(self, reference_points: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        quantiles = self._standard_normal_quantiles(reference_points)
        physical = self.locations[None, :] + self.scales[None, :] * quantiles
        return physical, tf.reduce_sum(
            self.forward_log_det_components(reference_points), axis=1
        )

    def inverse_log_det_components(self, physical_points: tf.Tensor) -> tf.Tensor:
        physical = self._physical_points(physical_points)
        quantiles = (physical - self.locations[None, :]) / self.scales[None, :]
        return (
            0.5 * tf.math.log(tf.constant(2.0 / 3.141592653589793, DTYPE))
            - tf.math.log(self.scales)[None, :]
            - 0.5 * tf.square(quantiles)
        )

    def inverse(self, physical_points: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        physical = self._physical_points(physical_points)
        quantiles = (physical - self.locations[None, :]) / self.scales[None, :]
        reference = tf.math.erf(quantiles / tf.sqrt(tf.constant(2.0, DTYPE)))
        return reference, tf.reduce_sum(
            self.inverse_log_det_components(physical), axis=1
        )

    def manifest_payload(self) -> Mapping[str, object]:
        return {
            "family": "GaussianQuantileCoordinateMap",
            "route_classification": "extension_or_invention",
            "locations": self.locations,
            "scales": self.scales,
            "formula": "x=location+scale*sqrt(2)*erfinv(z)",
            "measure_role": "reference_to_physical_jacobian_explicit",
            "defensive_physical_family": "diagonal_gaussian_from_uniform_quantile_map",
        }


@dataclass(frozen=True)
class CoupledNonlinearGaussianModel:
    """Independent replicated two-state nonlinear Gaussian transition blocks.

    The transition mean couples the two coordinates through ``s * i`` while
    the noise remains diagonal and strictly positive.  This keeps the exact
    conditional proposal available for a matched comparator without turning
    the rung into a singular-measure or clipping experiment.
    """

    block_count: int
    delta: float = 0.08
    base_kappa: float = 0.35
    base_nu: float = 0.25
    process_variance_s: float = 0.04
    process_variance_i: float = 0.04
    observation_variance: float = 0.09
    initial_mean_s: float = 0.8
    initial_mean_i: float = 0.2
    initial_variance_s: float = 0.04
    initial_variance_i: float = 0.04
    dtype: tf.dtypes.DType = ONLINE_DTYPE

    def __post_init__(self) -> None:
        if isinstance(self.block_count, bool) or int(self.block_count) < 1:
            raise ValueError("block_count must be a positive integer")
        object.__setattr__(self, "block_count", int(self.block_count))
        for name in (
            "delta",
            "base_kappa",
            "base_nu",
            "process_variance_s",
            "process_variance_i",
            "observation_variance",
            "initial_mean_s",
            "initial_mean_i",
            "initial_variance_s",
            "initial_variance_i",
        ):
            value = float(getattr(self, name))
            if not tf.math.is_finite(tf.constant(value, DTYPE)).numpy() or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)
        if self.dtype not in (DTYPE, ONLINE_DTYPE):
            raise ValueError("dtype must be float64 or float32")

    def parameter_dim(self) -> int:
        return 3

    def state_dim(self) -> int:
        return 2 * self.block_count

    def observation_dim(self) -> int:
        return self.block_count

    def frozen_apf_measure_id(self) -> str:
        return MEASURE_ID

    def frozen_apf_score_backend_id(self) -> str:
        return SCORE_BACKEND_ID

    def _theta(self, theta: tf.Tensor, dtype: tf.dtypes.DType | None = None) -> tf.Tensor:
        target_dtype = self.dtype if dtype is None else dtype
        values = tf.reshape(tf.convert_to_tensor(theta, dtype=target_dtype), [3])
        if values.shape != (3,):
            raise ValueError("theta must have shape [3]")
        return values

    def _row_state(self, state: tf.Tensor, dtype: tf.dtypes.DType | None = None) -> tf.Tensor:
        target_dtype = self.dtype if dtype is None else dtype
        values = tf.convert_to_tensor(state, dtype=target_dtype)
        if values.shape.rank != 2 or values.shape[1] != self.state_dim():
            raise ValueError(f"state must have shape [batch,{self.state_dim()}]")
        return values

    def _observation(self, observation: tf.Tensor, dtype: tf.dtypes.DType | None = None) -> tf.Tensor:
        target_dtype = self.dtype if dtype is None else dtype
        values = tf.reshape(tf.convert_to_tensor(observation, dtype=target_dtype), [self.observation_dim()])
        if values.shape != (self.observation_dim(),):
            raise ValueError(f"observation must have shape [{self.observation_dim()}]")
        return values

    def physical_parameters(self, theta: tf.Tensor) -> Mapping[str, tf.Tensor]:
        values = self._theta(theta)
        return {
            "kappa": tf.constant(self.base_kappa, self.dtype) * tf.exp(values[0]),
            "nu": tf.constant(self.base_nu, self.dtype) * tf.exp(values[1]),
            "observation_offset": values[2],
        }

    def transition_mean(self, theta: tf.Tensor, previous: tf.Tensor) -> tf.Tensor:
        values = self._row_state(previous)
        parameters = self.physical_parameters(theta)
        susceptible = values[:, 0::2]
        infectious = values[:, 1::2]
        infection = parameters["kappa"] * susceptible * infectious
        next_susceptible = susceptible - tf.constant(self.delta, self.dtype) * infection
        next_infectious = infectious + tf.constant(self.delta, self.dtype) * (
            infection - parameters["nu"] * infectious
        )
        return tf.reshape(tf.stack([next_susceptible, next_infectious], axis=2), [tf.shape(values)[0], self.state_dim()])

    def predictive_observation_mean(self, theta: tf.Tensor, previous: tf.Tensor) -> tf.Tensor:
        mean = self.transition_mean(theta, previous)
        parameters = self.physical_parameters(theta)
        return mean[:, 1::2] + parameters["observation_offset"]

    def predictive_observation_variance(self) -> tf.Tensor:
        return tf.fill(
            [self.observation_dim()],
            tf.constant(self.process_variance_i + self.observation_variance, self.dtype),
        )

    def conditional_parameters(
        self,
        theta: tf.Tensor,
        previous: tf.Tensor,
        observation: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Return exact Gaussian ``q(x_t | x_{t-1}, y_t)`` mean and diagonal variance."""

        transition = self.transition_mean(theta, previous)
        observed = self._observation(observation)
        parameters = self.physical_parameters(theta)
        predictive_variance = tf.constant(
            self.process_variance_i + self.observation_variance, self.dtype
        )
        gain = tf.constant(self.process_variance_i, self.dtype) / predictive_variance
        innovation = observed[None, :] - (transition[:, 1::2] + parameters["observation_offset"])
        conditional = tf.tensor_scatter_nd_add(
            transition,
            tf.stack(
                [
                    tf.repeat(tf.range(tf.shape(transition)[0]), self.block_count),
                    tf.tile(tf.range(1, self.state_dim(), delta=2), [tf.shape(transition)[0]]),
                ],
                axis=1,
            ),
            tf.reshape(gain * innovation, [-1]),
        )
        variance = tf.reshape(
            tf.tile(
                tf.stack(
                    [
                        tf.constant(self.process_variance_s, self.dtype),
                        tf.constant(self.process_variance_i * self.observation_variance, self.dtype)
                        / predictive_variance,
                    ]
                ),
                [self.block_count],
            ),
            [self.state_dim()],
        )
        return conditional, variance

    def conditional_log_density(
        self,
        theta: tf.Tensor,
        previous: tf.Tensor,
        current: tf.Tensor,
        observation: tf.Tensor,
    ) -> tf.Tensor:
        mean, variance = self.conditional_parameters(theta, previous, observation)
        values = self._row_state(current)
        return _diagonal_normal_log_density(values, mean, variance)

    def initial_log_density(self, theta: tf.Tensor, state: tf.Tensor) -> tf.Tensor:
        del theta
        values = self._row_state(state)
        mean = tf.reshape(
            tf.tile(
                tf.constant([self.initial_mean_s, self.initial_mean_i], self.dtype)[None, :],
                [1, self.block_count],
            ),
            [self.state_dim()],
        )
        variance = tf.reshape(
            tf.tile(
                tf.constant([self.initial_variance_s, self.initial_variance_i], self.dtype)[None, :],
                [1, self.block_count],
            ),
            [self.state_dim()],
        )
        return _diagonal_normal_log_density(values, mean[None, :], variance[None, :])

    def transition_log_density(
        self,
        theta: tf.Tensor,
        previous: tf.Tensor,
        current: tf.Tensor,
        time_index: int,
    ) -> tf.Tensor:
        del time_index
        mean = self.transition_mean(theta, previous)
        variance = tf.reshape(
            tf.tile(
                tf.constant([self.process_variance_s, self.process_variance_i], self.dtype)[None, :],
                [1, self.block_count],
            ),
            [self.state_dim()],
        )
        return _diagonal_normal_log_density(self._row_state(current), mean, variance[None, :])

    def observation_log_density(
        self,
        theta: tf.Tensor,
        state: tf.Tensor,
        observation: tf.Tensor,
        time_index: int,
    ) -> tf.Tensor:
        del time_index
        values = self._row_state(state)
        observed = self._observation(observation)
        parameters = self.physical_parameters(theta)
        mean = values[:, 1::2] + parameters["observation_offset"]
        variance = tf.fill([self.block_count], tf.constant(self.observation_variance, self.dtype))
        return _diagonal_normal_log_density(
            tf.broadcast_to(observed[None, :], tf.shape(mean)), mean, variance[None, :]
        )

    def initial_log_density_parameter_score(self, theta: tf.Tensor, state: tf.Tensor) -> tf.Tensor:
        values = self._row_state(state)
        return tf.zeros([tf.shape(values)[0], self.parameter_dim()], dtype=self.dtype)

    def transition_log_density_parameter_score(
        self,
        theta: tf.Tensor,
        previous: tf.Tensor,
        current: tf.Tensor,
        time_index: int,
    ) -> tf.Tensor:
        del time_index
        previous_values = self._row_state(previous)
        current_values = self._row_state(current)
        parameters = self.physical_parameters(theta)
        susceptible = previous_values[:, 0::2]
        infectious = previous_values[:, 1::2]
        derivative_kappa = tf.constant(self.delta, self.dtype) * parameters["kappa"] * susceptible * infectious
        derivative_nu = -tf.constant(self.delta, self.dtype) * parameters["nu"] * infectious
        d_mean_kappa = tf.reshape(tf.stack([-derivative_kappa, derivative_kappa], axis=2), [tf.shape(previous_values)[0], self.state_dim()])
        d_mean_nu = tf.reshape(tf.stack([tf.zeros_like(derivative_nu), derivative_nu], axis=2), [tf.shape(previous_values)[0], self.state_dim()])
        residual = current_values - self.transition_mean(theta, previous_values)
        variance = tf.reshape(
            tf.tile(tf.constant([self.process_variance_s, self.process_variance_i], self.dtype)[None, :], [1, self.block_count]),
            [self.state_dim()],
        )
        return tf.stack(
            [
                tf.reduce_sum(residual * d_mean_kappa / variance[None, :], axis=1),
                tf.reduce_sum(residual * d_mean_nu / variance[None, :], axis=1),
                tf.zeros([tf.shape(residual)[0]], dtype=self.dtype),
            ],
            axis=1,
        )

    def observation_log_density_parameter_score(
        self,
        theta: tf.Tensor,
        state: tf.Tensor,
        observation: tf.Tensor,
        time_index: int,
    ) -> tf.Tensor:
        del time_index
        values = self._row_state(state)
        observed = self._observation(observation)
        parameters = self.physical_parameters(theta)
        residual = tf.broadcast_to(observed[None, :], [tf.shape(values)[0], self.block_count]) - (
            values[:, 1::2] + parameters["observation_offset"]
        )
        offset_score = tf.reduce_sum(
            residual / tf.constant(self.observation_variance, self.dtype), axis=1
        )
        return tf.stack(
            [
                tf.zeros_like(offset_score),
                tf.zeros_like(offset_score),
                offset_score,
            ],
            axis=1,
        )

    def manifest_payload(self) -> Mapping[str, object]:
        return {
            "family": "coupled_nonlinear_gaussian_sir_inspired_blocks",
            "route_classification": "extension_or_invention",
            "block_count": self.block_count,
            "state_dimension": self.state_dim(),
            "observation_dimension": self.observation_dim(),
            "parameter_order": ("log_kappa_scale", "log_nu_scale", "observation_offset"),
            "transition_equation": "s_next=s-delta*kappa*s*i; i_next=i+delta*(kappa*s*i-nu*i)",
            "observation_equation": "y=i+observation_offset+epsilon",
            "process_variance": (self.process_variance_s, self.process_variance_i),
            "observation_variance": self.observation_variance,
            "full_support_measure": "diagonal_gaussian_process_and_observation_lebesgue",
            "exact_conditional_available": True,
            "what_is_not_claimed": (
                "source_faithful_Zhao_Cui_model",
                "Austria_SIR_or_NAWM",
                "long_horizon_nonlinear_scalability",
                "posterior_or_HMC_readiness",
            ),
        }


def _diagonal_normal_log_density(
    values: tf.Tensor,
    means: tf.Tensor,
    variances: tf.Tensor,
) -> tf.Tensor:
    residual = values - means
    return -0.5 * tf.reduce_sum(
        tf.constant(LOG_TWO_PI, values.dtype) + tf.math.log(variances) + tf.square(residual) / variances,
        axis=-1,
    )


__all__ = [
    "CoupledNonlinearGaussianModel",
    "GaussianQuantileCoordinateMap",
    "ShiftedAlgebraicCoordinateMap",
]
