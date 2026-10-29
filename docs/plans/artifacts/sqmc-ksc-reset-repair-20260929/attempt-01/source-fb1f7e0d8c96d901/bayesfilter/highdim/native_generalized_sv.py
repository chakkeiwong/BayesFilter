"""Native generalized stochastic-volatility dense reference fixtures."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from bayesfilter.highdim.diagnostics import HighDimStatus


@dataclass(frozen=True)
class NativeGeneralizedSVSSM:
    """Tiny native generalized SV model for same-target reference gates."""

    parameterization: str = "rho_s_rho_h_log_sigma_s_log_sigma_h_log_beta"

    def __post_init__(self) -> None:
        if self.parameterization != "rho_s_rho_h_log_sigma_s_log_sigma_h_log_beta":
            raise ValueError("unsupported native generalized SV parameterization")

    def parameter_dim(self) -> int:
        return 5

    def state_dim(self) -> int:
        return 2

    def observation_dim(self) -> int:
        return 1

    def physical_parameters(self, theta: tf.Tensor) -> Mapping[str, tf.Tensor]:
        theta_vector = _as_parameter_vector(theta, self.parameter_dim(), "theta")
        return {
            "rho_s": tf.tanh(theta_vector[0]),
            "rho_h": tf.tanh(theta_vector[1]),
            "sigma_s": tf.exp(theta_vector[2]),
            "sigma_h": tf.exp(theta_vector[3]),
            "beta": tf.exp(theta_vector[4]),
        }

    def unconstrained_from_physical(
        self,
        *,
        rho_s: float | tf.Tensor,
        rho_h: float | tf.Tensor,
        sigma_s: float | tf.Tensor,
        sigma_h: float | tf.Tensor,
        beta: float | tf.Tensor,
    ) -> tf.Tensor:
        rho_s_tensor = tf.convert_to_tensor(rho_s, dtype=tf.float64)
        rho_h_tensor = tf.convert_to_tensor(rho_h, dtype=tf.float64)
        sigma_s_tensor = tf.convert_to_tensor(sigma_s, dtype=tf.float64)
        sigma_h_tensor = tf.convert_to_tensor(sigma_h, dtype=tf.float64)
        beta_tensor = tf.convert_to_tensor(beta, dtype=tf.float64)
        for name, value in (
            ("rho_s", rho_s_tensor),
            ("rho_h", rho_h_tensor),
            ("sigma_s", sigma_s_tensor),
            ("sigma_h", sigma_h_tensor),
            ("beta", beta_tensor),
        ):
            if value.shape.rank != 0:
                raise ValueError(f"{name}: {HighDimStatus.INVALID_SHAPE.value}")
        checks = tf.stack(
            [
                tf.math.is_finite(rho_s_tensor),
                tf.math.is_finite(rho_h_tensor),
                tf.math.is_finite(sigma_s_tensor),
                tf.math.is_finite(sigma_h_tensor),
                tf.math.is_finite(beta_tensor),
                tf.abs(rho_s_tensor) < 1.0,
                tf.abs(rho_h_tensor) < 1.0,
                sigma_s_tensor > 0.0,
                sigma_h_tensor > 0.0,
                beta_tensor > 0.0,
            ]
        )
        if not bool(tf.reduce_all(checks).numpy()):
            raise ValueError(f"physical parameter: {HighDimStatus.NONFINITE_VALUE.value}")
        return tf.stack(
            [
                tf.atanh(rho_s_tensor),
                tf.atanh(rho_h_tensor),
                tf.math.log(sigma_s_tensor),
                tf.math.log(sigma_h_tensor),
                tf.math.log(beta_tensor),
            ]
        )

    def initial_log_density(self, theta: tf.Tensor, x0: tf.Tensor) -> tf.Tensor:
        values = _as_row_matrix(x0, self.state_dim(), "x0")
        parameters = self.physical_parameters(theta)
        scales = tf.stack(
            [
                parameters["sigma_s"] / tf.sqrt(1.0 - tf.square(parameters["rho_s"])),
                parameters["sigma_h"] / tf.sqrt(1.0 - tf.square(parameters["rho_h"])),
            ]
        )
        distribution = tfp.distributions.Normal(
            loc=tf.zeros([2], dtype=tf.float64),
            scale=scales,
        )
        return tf.reduce_sum(distribution.log_prob(values), axis=1)

    def transition_log_density(
        self,
        theta: tf.Tensor,
        x_prev: tf.Tensor,
        x_next: tf.Tensor,
        t: int,
    ) -> tf.Tensor:
        del t
        previous = _as_row_matrix(x_prev, self.state_dim(), "x_prev")
        next_values = _as_row_matrix(x_next, self.state_dim(), "x_next")
        parameters = self.physical_parameters(theta)
        loc = tf.stack(
            [
                parameters["rho_s"] * previous[:, 0],
                parameters["rho_h"] * previous[:, 1],
            ],
            axis=1,
        )
        scale = tf.stack([parameters["sigma_s"], parameters["sigma_h"]])
        distribution = tfp.distributions.Normal(loc=loc, scale=scale[tf.newaxis, :])
        return tf.reduce_sum(distribution.log_prob(next_values), axis=1)

    def observation_log_density(
        self,
        theta: tf.Tensor,
        x_t: tf.Tensor,
        y_t: tf.Tensor,
        t: int,
    ) -> tf.Tensor:
        del t
        values = _as_row_matrix(x_t, self.state_dim(), "x_t")
        observation = tf.reshape(tf.convert_to_tensor(y_t, dtype=tf.float64), [1])
        parameters = self.physical_parameters(theta)
        distribution = tfp.distributions.Normal(
            loc=parameters["beta"] * values[:, 0],
            scale=tf.exp(0.5 * values[:, 1]),
        )
        return distribution.log_prob(tf.broadcast_to(observation[0], [tf.shape(values)[0]]))

    def manifest_payload(self) -> Mapping[str, object]:
        return {
            "family": "NativeGeneralizedSVSSM",
            "state": "x_t=(s_t,h_t)",
            "state_law": "independent stationary AR(1) states for s_t and h_t",
            "observation_law": "y_t | s_t,h_t,beta has mean beta s_t and variance exp(h_t)",
            "parameterization": self.parameterization,
            "parameter_vector": (
                "atanh(rho_s)",
                "atanh(rho_h)",
                "log(sigma_s)",
                "log(sigma_h)",
                "log(beta)",
            ),
            "what_is_not_claimed": (
                "transformed-residual exactness",
                "KSC Gaussian-mixture approximation",
                "CUT4 same-target equality",
                "Zhao-Cui same-target equality",
                "HMC readiness",
                "production generalized SV readiness",
            ),
        }


@dataclass(frozen=True)
class NativeGeneralizedSVDenseReferenceResult:
    """Sequential dense-grid value-path result for native generalized SV."""

    log_likelihood: tf.Tensor
    log_normalizers: tf.Tensor
    mean_path: tf.Tensor
    covariance_path: tf.Tensor
    diagnostics: Mapping[str, object]

    def __post_init__(self) -> None:
        value = tf.convert_to_tensor(self.log_likelihood, dtype=tf.float64)
        log_normalizers = tf.convert_to_tensor(self.log_normalizers, dtype=tf.float64)
        mean_path = tf.convert_to_tensor(self.mean_path, dtype=tf.float64)
        covariance_path = tf.convert_to_tensor(self.covariance_path, dtype=tf.float64)
        if value.shape.rank != 0 or log_normalizers.shape.rank != 1:
            raise ValueError(f"native generalized SV result: {HighDimStatus.INVALID_SHAPE.value}")
        if mean_path.shape.rank != 2 or covariance_path.shape.rank != 3:
            raise ValueError(f"native generalized SV moments: {HighDimStatus.INVALID_SHAPE.value}")
        if not bool(
            tf.math.is_finite(value).numpy()
            and tf.reduce_all(tf.math.is_finite(log_normalizers)).numpy()
            and tf.reduce_all(tf.math.is_finite(mean_path)).numpy()
            and tf.reduce_all(tf.math.is_finite(covariance_path)).numpy()
        ):
            raise ValueError(f"native generalized SV result: {HighDimStatus.NONFINITE_VALUE.value}")
        object.__setattr__(self, "log_likelihood", value)
        object.__setattr__(self, "log_normalizers", log_normalizers)
        object.__setattr__(self, "mean_path", mean_path)
        object.__setattr__(self, "covariance_path", covariance_path)
        object.__setattr__(
            self,
            "diagnostics",
            MappingProxyType({str(k): v for k, v in self.diagnostics.items()}),
        )


def native_generalized_sv_dense_reference(
    model: NativeGeneralizedSVSSM,
    theta: tf.Tensor,
    observations: tf.Tensor,
    *,
    order: int = 25,
    radius_s: float = 3.5,
    radius_h: float = 3.5,
) -> NativeGeneralizedSVDenseReferenceResult:
    """Return a tiny dense same-target reference for the native raw-y model."""

    if not isinstance(model, NativeGeneralizedSVSSM):
        raise TypeError("model must be NativeGeneralizedSVSSM")
    if int(order) < 3:
        raise ValueError("order must be at least 3")
    theta_vector = _as_parameter_vector(theta, model.parameter_dim(), "theta")
    observation_matrix = _as_observation_matrix(observations, model.observation_dim())
    points, weights = _tensor_product_legendre_grid(
        order=int(order),
        radius_s=float(radius_s),
        radius_h=float(radius_h),
    )
    log_terms = []
    mean_terms = []
    covariance_terms = []
    log_posterior: tf.Tensor | None = None
    for time_index in range(int(observation_matrix.shape[0])):
        if time_index == 0:
            log_unnormalized = model.initial_log_density(theta_vector, points)
        else:
            if log_posterior is None:
                raise RuntimeError("missing retained native generalized SV posterior")
            transition_log = _pairwise_transition_log_density(
                model=model,
                theta=theta_vector,
                current_points=points,
                previous_points=points,
                t=time_index,
            )
            log_predictive = tf.reduce_logsumexp(
                tf.math.log(weights)[tf.newaxis, :] + log_posterior[tf.newaxis, :] + transition_log,
                axis=1,
            )
            log_unnormalized = log_predictive
        log_unnormalized = log_unnormalized + model.observation_log_density(
            theta_vector,
            points,
            observation_matrix[time_index],
            t=time_index,
        )
        log_normalizer = _logsumexp_weighted(log_unnormalized, weights)
        log_posterior = log_unnormalized - log_normalizer
        mean, covariance = _grid_moments(points, weights, log_posterior)
        log_terms.append(log_normalizer)
        mean_terms.append(mean)
        covariance_terms.append(covariance)
    log_normalizers = tf.stack(log_terms)
    return NativeGeneralizedSVDenseReferenceResult(
        log_likelihood=tf.reduce_sum(log_normalizers),
        log_normalizers=log_normalizers,
        mean_path=tf.stack(mean_terms),
        covariance_path=tf.stack(covariance_terms),
        diagnostics={
            "backend": "dense_native_generalized_sv_raw_observation",
            "grid_order": int(order),
            "state_dim": model.state_dim(),
            "observation_dim": model.observation_dim(),
            "target": "native raw-y generalized SV",
            "radius_s": float(radius_s),
            "radius_h": float(radius_h),
            "non_claims": (
                "not transformed-residual diagnostic",
                "not KSC Gaussian mixture approximation",
                "no CUT4 same-target equality",
                "no Zhao-Cui same-target equality",
                "no HMC readiness",
                "no production generalized SV readiness",
            ),
        },
    )


def _as_parameter_vector(values: tf.Tensor, width: int, name: str) -> tf.Tensor:
    tensor = tf.convert_to_tensor(values, dtype=tf.float64)
    if tensor.shape.rank == 2 and tensor.shape[0] == 1:
        tensor = tensor[0]
    if tensor.shape.rank != 1 or tensor.shape[0] != int(width):
        raise ValueError(f"{name}: {HighDimStatus.INVALID_SHAPE.value}")
    if not bool(tf.reduce_all(tf.math.is_finite(tensor)).numpy()):
        raise ValueError(f"{name}: {HighDimStatus.NONFINITE_VALUE.value}")
    return tensor


def _as_row_matrix(values: tf.Tensor, width: int, name: str) -> tf.Tensor:
    tensor = tf.convert_to_tensor(values, dtype=tf.float64)
    if tensor.shape.rank == 1:
        tensor = tensor[tf.newaxis, :]
    if tensor.shape.rank != 2 or tensor.shape[1] != int(width):
        raise ValueError(f"{name}: {HighDimStatus.INVALID_SHAPE.value}")
    if not bool(tf.reduce_all(tf.math.is_finite(tensor)).numpy()):
        raise ValueError(f"{name}: {HighDimStatus.NONFINITE_VALUE.value}")
    return tensor


def _as_observation_matrix(values: tf.Tensor, width: int) -> tf.Tensor:
    tensor = tf.convert_to_tensor(values, dtype=tf.float64)
    if tensor.shape.rank == 1:
        tensor = tensor[:, tf.newaxis]
    if tensor.shape.rank != 2 or tensor.shape[1] != int(width):
        raise ValueError(f"observations: {HighDimStatus.INVALID_SHAPE.value}")
    if int(tensor.shape[0]) <= 0:
        raise ValueError("observations must have at least one row")
    if not bool(tf.reduce_all(tf.math.is_finite(tensor)).numpy()):
        raise ValueError(f"observations: {HighDimStatus.NONFINITE_VALUE.value}")
    return tensor


def _tensor_product_legendre_grid(
    *,
    order: int,
    radius_s: float,
    radius_h: float,
) -> tuple[tf.Tensor, tf.Tensor]:
    s_nodes, s_weights = _legendre_interval_nodes_weights(
        order=order,
        left=-float(radius_s),
        right=float(radius_s),
    )
    h_nodes, h_weights = _legendre_interval_nodes_weights(
        order=order,
        left=-float(radius_h),
        right=float(radius_h),
    )
    s_mesh, h_mesh = tf.meshgrid(s_nodes, h_nodes, indexing="ij")
    s_weight_mesh, h_weight_mesh = tf.meshgrid(s_weights, h_weights, indexing="ij")
    points = tf.stack([tf.reshape(s_mesh, [-1]), tf.reshape(h_mesh, [-1])], axis=1)
    weights = tf.reshape(s_weight_mesh * h_weight_mesh, [-1])
    return points, weights


def _legendre_interval_nodes_weights(
    *,
    order: int,
    left: float,
    right: float,
) -> tuple[tf.Tensor, tf.Tensor]:
    nodes, weights = np.polynomial.legendre.leggauss(int(order))
    nodes_tensor = tf.constant(nodes, dtype=tf.float64)
    weights_tensor = tf.constant(weights, dtype=tf.float64)
    left_tensor = tf.constant(float(left), dtype=tf.float64)
    right_tensor = tf.constant(float(right), dtype=tf.float64)
    center = 0.5 * (right_tensor + left_tensor)
    scale = 0.5 * (right_tensor - left_tensor)
    return center + scale * nodes_tensor, scale * weights_tensor


def _pairwise_transition_log_density(
    *,
    model: NativeGeneralizedSVSSM,
    theta: tf.Tensor,
    current_points: tf.Tensor,
    previous_points: tf.Tensor,
    t: int,
) -> tf.Tensor:
    current = _as_row_matrix(current_points, model.state_dim(), "current_points")
    previous = _as_row_matrix(previous_points, model.state_dim(), "previous_points")
    current_tiled = tf.reshape(
        tf.tile(current[:, tf.newaxis, :], [1, tf.shape(previous)[0], 1]),
        [-1, model.state_dim()],
    )
    previous_tiled = tf.reshape(
        tf.tile(previous[tf.newaxis, :, :], [tf.shape(current)[0], 1, 1]),
        [-1, model.state_dim()],
    )
    values = model.transition_log_density(theta, previous_tiled, current_tiled, t=t)
    return tf.reshape(values, [tf.shape(current)[0], tf.shape(previous)[0]])


def _logsumexp_weighted(log_values: tf.Tensor, weights: tf.Tensor) -> tf.Tensor:
    values = tf.convert_to_tensor(log_values, dtype=tf.float64)
    max_log = tf.reduce_max(values)
    return tf.math.log(tf.reduce_sum(weights * tf.exp(values - max_log))) + max_log


def _grid_moments(
    points: tf.Tensor,
    weights: tf.Tensor,
    log_density: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    normalized_weights = tf.convert_to_tensor(weights, dtype=tf.float64) * tf.exp(
        tf.convert_to_tensor(log_density, dtype=tf.float64)
    )
    mean = tf.reduce_sum(normalized_weights[:, tf.newaxis] * points, axis=0)
    centered = points - mean[tf.newaxis, :]
    covariance = tf.einsum("n,ni,nj->ij", normalized_weights, centered, centered)
    return mean, 0.5 * (covariance + tf.transpose(covariance))
