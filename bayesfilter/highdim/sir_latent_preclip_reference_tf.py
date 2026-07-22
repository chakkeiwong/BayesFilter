"""Reduced deterministic reference for the latent pre-clipping SIR target.

This module is an independent CPU/float64 reference, not a production filtering
backend.  It uses fixed tensor-product Gauss--Legendre grids and explicitly
propagates the normalized previous-marginal score.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import tensorflow as tf

from bayesfilter.highdim.models import ParameterizedZhaoCuiSIRSSM, SpatialSIRSSM
from bayesfilter.highdim.sir_latent_preclip_tf import LatentPreclipSIRSSM


DTYPE = tf.float64
PARAMETER_COUNT = 3
REFERENCE_ID = "latent_preclip_sir_j1_dense_legendre_filtering_score_v1"


@dataclass(frozen=True)
class DenseGrid:
    points: tf.Tensor
    weights: tf.Tensor
    boundary_mask: tf.Tensor
    center: tf.Tensor
    scale: tf.Tensor
    order: int
    radius: float
    integration_rule: str


def reduced_latent_preclip_sir_model() -> LatentPreclipSIRSSM:
    """Return a J=1 fixture whose clipping boundary has material probability."""

    base = SpatialSIRSSM(
        kappa=tf.constant([0.1], DTYPE),
        nu=tf.constant([1.0], DTYPE),
        initial_mean=tf.constant([0.3, 0.2], DTYPE),
        neighbor_sets=((),),
        delta=0.02,
        rk4_internal_step=0.005,
        process_covariance=tf.constant([[0.25, 0.0], [0.0, 0.16]], DTYPE),
        observation_covariance=tf.constant([[0.16]], DTYPE),
        initial_covariance=tf.constant([[0.25, 0.0], [0.0, 0.16]], DTYPE),
        process_noise_policy="clip_susceptible_after_noise",
    )
    return LatentPreclipSIRSSM(ParameterizedZhaoCuiSIRSSM(base))


def _legendre_product_grid(
    center: tf.Tensor, scale: tf.Tensor, *, order: int, radius: float
) -> DenseGrid:
    if int(order) < 3:
        raise ValueError("order must be at least three")
    if float(radius) <= 0.0:
        raise ValueError("radius must be positive")
    center = tf.reshape(tf.convert_to_tensor(center, DTYPE), [2])
    scale = tf.reshape(tf.convert_to_tensor(scale, DTYPE), [2])
    if not bool(tf.reduce_all(scale > 0.0).numpy()):
        raise ValueError("grid scale must be positive")
    nodes_np, weights_np = np.polynomial.legendre.leggauss(int(order))
    nodes = tf.constant(nodes_np, DTYPE)
    weights = tf.constant(weights_np, DTYPE)
    axis_points = []
    axis_weights = []
    for axis in range(2):
        half_width = tf.constant(float(radius), DTYPE) * scale[axis]
        axis_points.append(center[axis] + half_width * nodes)
        axis_weights.append(half_width * weights)
    mesh = tf.meshgrid(axis_points[0], axis_points[1], indexing="ij")
    weight_mesh = tf.meshgrid(axis_weights[0], axis_weights[1], indexing="ij")
    points = tf.stack([tf.reshape(value, [-1]) for value in mesh], axis=1)
    product_weights = tf.reshape(weight_mesh[0] * weight_mesh[1], [-1])
    indices = tf.range(int(order), dtype=tf.int32)
    boundary_axis = (indices == 0) | (indices == int(order) - 1)
    boundary_mesh = tf.meshgrid(boundary_axis, boundary_axis, indexing="ij")
    boundary_mask = tf.reshape(boundary_mesh[0] | boundary_mesh[1], [-1])
    return DenseGrid(
        points=points,
        weights=product_weights,
        boundary_mask=boundary_mask,
        center=center,
        scale=scale,
        order=int(order),
        radius=float(radius),
        integration_rule="gauss_legendre_bounded",
    )


def _hermite_product_grid(
    center: tf.Tensor, scale: tf.Tensor, *, order: int, scale_inflation: float
) -> DenseGrid:
    """Return a Lebesgue-integral grid based on Gaussian-weighted nodes."""

    if int(order) < 3:
        raise ValueError("order must be at least three")
    if float(scale_inflation) <= 0.0:
        raise ValueError("scale_inflation must be positive")
    center = tf.reshape(tf.convert_to_tensor(center, DTYPE), [2])
    scale = tf.reshape(tf.convert_to_tensor(scale, DTYPE), [2])
    if not bool(tf.reduce_all(scale > 0.0).numpy()):
        raise ValueError("grid scale must be positive")
    nodes_np, weights_np = np.polynomial.hermite.hermgauss(int(order))
    nodes = tf.constant(nodes_np, DTYPE)
    hermite_weights = tf.constant(weights_np, DTYPE)
    axis_points = []
    axis_weights = []
    root_two = tf.sqrt(tf.constant(2.0, DTYPE))
    for axis in range(2):
        physical_scale = tf.constant(float(scale_inflation), DTYPE) * scale[axis]
        axis_points.append(center[axis] + root_two * physical_scale * nodes)
        # hermgauss integrates exp(-x^2) f(x); undo that weight to
        # approximate the Lebesgue integral in the transformed coordinate.
        axis_weights.append(
            root_two
            * physical_scale
            * hermite_weights
            * tf.exp(tf.square(nodes))
        )
    mesh = tf.meshgrid(axis_points[0], axis_points[1], indexing="ij")
    weight_mesh = tf.meshgrid(axis_weights[0], axis_weights[1], indexing="ij")
    points = tf.stack([tf.reshape(value, [-1]) for value in mesh], axis=1)
    product_weights = tf.reshape(weight_mesh[0] * weight_mesh[1], [-1])
    indices = tf.range(int(order), dtype=tf.int32)
    boundary_axis = (indices == 0) | (indices == int(order) - 1)
    boundary_mesh = tf.meshgrid(boundary_axis, boundary_axis, indexing="ij")
    boundary_mask = tf.reshape(boundary_mesh[0] | boundary_mesh[1], [-1])
    return DenseGrid(
        points=points,
        weights=product_weights,
        boundary_mask=boundary_mask,
        center=center,
        scale=scale,
        order=int(order),
        radius=float(scale_inflation),
        integration_rule="gauss_hermite_lebesgue_reweighted",
    )


def _split_legendre_product_grid(
    center: tf.Tensor, scale: tf.Tensor, *, order: int, radius: float
) -> DenseGrid:
    """Split the susceptible integration interval at the clipping boundary."""

    if int(order) < 3:
        raise ValueError("order must be at least three")
    if float(radius) <= 0.0:
        raise ValueError("radius must be positive")
    center = tf.reshape(tf.convert_to_tensor(center, DTYPE), [2])
    scale = tf.reshape(tf.convert_to_tensor(scale, DTYPE), [2])
    nodes_np, weights_np = np.polynomial.legendre.leggauss(int(order))
    nodes = tf.constant(nodes_np, DTYPE)
    base_weights = tf.constant(weights_np, DTYPE)

    lower = center - tf.constant(float(radius), DTYPE) * scale
    upper = center + tf.constant(float(radius), DTYPE) * scale
    susceptible_panels = []
    lower_s = float(lower[0].numpy())
    upper_s = float(upper[0].numpy())
    if lower_s < 0.0 < upper_s:
        susceptible_panels.extend(((lower_s, 0.0), (0.0, upper_s)))
    else:
        susceptible_panels.append((lower_s, upper_s))

    susceptible_points = []
    susceptible_weights = []
    susceptible_outer_boundary = []
    for panel_index, (panel_lower, panel_upper) in enumerate(susceptible_panels):
        midpoint = tf.constant(0.5 * (panel_lower + panel_upper), DTYPE)
        half_width = tf.constant(0.5 * (panel_upper - panel_lower), DTYPE)
        susceptible_points.append(midpoint + half_width * nodes)
        susceptible_weights.append(half_width * base_weights)
        mask = tf.zeros([int(order)], tf.bool)
        if panel_index == 0:
            mask = tf.tensor_scatter_nd_update(mask, [[0]], [True])
        if panel_index == len(susceptible_panels) - 1:
            mask = tf.tensor_scatter_nd_update(mask, [[int(order) - 1]], [True])
        susceptible_outer_boundary.append(mask)
    susceptible_axis = tf.concat(susceptible_points, axis=0)
    susceptible_axis_weights = tf.concat(susceptible_weights, axis=0)
    susceptible_boundary = tf.concat(susceptible_outer_boundary, axis=0)

    infectious_midpoint = 0.5 * (lower[1] + upper[1])
    infectious_half_width = 0.5 * (upper[1] - lower[1])
    infectious_axis = infectious_midpoint + infectious_half_width * nodes
    infectious_axis_weights = infectious_half_width * base_weights
    infectious_boundary = (tf.range(int(order)) == 0) | (
        tf.range(int(order)) == int(order) - 1
    )

    mesh = tf.meshgrid(susceptible_axis, infectious_axis, indexing="ij")
    weight_mesh = tf.meshgrid(
        susceptible_axis_weights, infectious_axis_weights, indexing="ij"
    )
    boundary_mesh = tf.meshgrid(
        susceptible_boundary, infectious_boundary, indexing="ij"
    )
    return DenseGrid(
        points=tf.stack([tf.reshape(value, [-1]) for value in mesh], axis=1),
        weights=tf.reshape(weight_mesh[0] * weight_mesh[1], [-1]),
        boundary_mask=tf.reshape(boundary_mesh[0] | boundary_mesh[1], [-1]),
        center=center,
        scale=scale,
        order=int(order),
        radius=float(radius),
        integration_rule="gauss_legendre_susceptible_split_at_zero",
    )


def prepare_reduced_dense_grids(
    model: LatentPreclipSIRSSM,
    theta: tf.Tensor,
    *,
    time_steps: int,
    order: int,
    radius: float,
    integration_rule: str = "split_gauss_legendre",
) -> tuple[DenseGrid, ...]:
    """Prepare fixed grids from an EKF-style center/scale scout.

    The scout selects integration regions only.  It is not used in the
    filtering value or score and cannot serve as correctness evidence.
    """

    if model.state_dim() != 2:
        raise ValueError("reduced dense reference supports J=1 only")
    if int(time_steps) < 0:
        raise ValueError("time_steps must be nonnegative")
    scaled = model.physical_model.scaled_model(theta)
    center = tf.identity(scaled.initial_mean)
    covariance = tf.identity(scaled.initial_covariance)
    grids = []
    for time_index in range(int(time_steps) + 1):
        scale = tf.sqrt(tf.maximum(tf.linalg.diag_part(covariance), 1.0e-12))
        if integration_rule == "split_gauss_legendre":
            grid = _split_legendre_product_grid(
                center, scale, order=int(order), radius=float(radius)
            )
        elif integration_rule == "gauss_hermite":
            grid = _hermite_product_grid(
                center,
                scale,
                order=int(order),
                scale_inflation=float(radius),
            )
        elif integration_rule == "gauss_legendre":
            grid = _legendre_product_grid(
                center, scale, order=int(order), radius=float(radius)
            )
        else:
            raise ValueError(
                "integration_rule must be split_gauss_legendre, gauss_hermite, "
                "or gauss_legendre"
            )
        grids.append(grid)
        if time_index == int(time_steps):
            continue
        physical_center = model.physical_state(
            center, time_index=time_index
        )[0]
        center_variable = tf.Variable(physical_center)
        with tf.GradientTape() as tape:
            next_center = model.physical_model.transition_mean(
                theta, center_variable[tf.newaxis, :]
            )[0]
        jacobian = tape.jacobian(next_center, center_variable)
        if jacobian is None:
            raise ValueError("grid scout transition Jacobian is disconnected")
        covariance = (
            jacobian @ covariance @ tf.transpose(jacobian)
            + scaled.process_covariance
        )
        center = next_center
    return tuple(grids)


def _pairwise_transition_terms(
    model: LatentPreclipSIRSSM,
    theta: tf.Tensor,
    previous: tf.Tensor,
    current: tf.Tensor,
    *,
    time_index: int,
) -> tuple[tf.Tensor, tf.Tensor]:
    previous_count = int(previous.shape[0])
    current_count = int(current.shape[0])
    repeated_previous = tf.repeat(previous, repeats=current_count, axis=0)
    tiled_current = tf.tile(current, [previous_count, 1])
    log_density = model.transition_log_density(
        theta, repeated_previous, tiled_current, t=int(time_index)
    )
    score = model.transition_log_density_parameter_score(
        theta, repeated_previous, tiled_current, t=int(time_index)
    )
    return (
        tf.reshape(log_density, [previous_count, current_count]),
        tf.reshape(score, [previous_count, current_count, PARAMETER_COUNT]),
    )


def dense_latent_sir_value_and_manual_score(
    model: LatentPreclipSIRSSM,
    theta: tf.Tensor,
    observations: tf.Tensor,
    grids: Sequence[DenseGrid],
    *,
    stop_previous_marginal_score: bool = False,
) -> dict[str, tf.Tensor | str]:
    """Return value and explicit normalized-filtering score on fixed grids."""

    theta = tf.reshape(tf.convert_to_tensor(theta, DTYPE), [PARAMETER_COUNT])
    observations = tf.convert_to_tensor(observations, DTYPE)
    if observations.shape.rank != 2 or observations.shape[1] != model.observation_dim():
        raise ValueError("observations must have shape [T+1,observation_dim]")
    if len(grids) != int(observations.shape[0]):
        raise ValueError("one fixed grid is required per observation")

    increments = []
    increment_scores = []
    boundary_masses = []
    posterior_score = None
    log_posterior = None
    for time_index, grid in enumerate(grids):
        log_weights = tf.math.log(grid.weights)
        if time_index == 0:
            log_predictive = model.initial_log_density(theta, grid.points)
            predictive_score = model.initial_log_density_parameter_score(
                theta, grid.points
            )
        else:
            previous_grid = grids[time_index - 1]
            transition_log, transition_score = _pairwise_transition_terms(
                model,
                theta,
                previous_grid.points,
                grid.points,
                time_index=time_index,
            )
            parent_terms = (
                tf.math.log(previous_grid.weights)[:, None]
                + log_posterior[:, None]
                + transition_log
            )
            log_predictive = tf.reduce_logsumexp(parent_terms, axis=0)
            normalized_parent = tf.exp(parent_terms - log_predictive[None, :])
            carry = (
                tf.zeros_like(posterior_score)
                if stop_previous_marginal_score
                else posterior_score
            )
            predictive_score = tf.reduce_sum(
                normalized_parent[:, :, None]
                * (carry[:, None, :] + transition_score),
                axis=0,
            )

        observation_log = model.observation_log_density(
            theta, grid.points, observations[time_index], t=time_index
        )
        observation_score = model.observation_log_density_parameter_score(
            theta, grid.points, observations[time_index], t=time_index
        )
        log_unnormalized = log_predictive + observation_log
        local_score = predictive_score + observation_score
        increment = tf.reduce_logsumexp(log_weights + log_unnormalized)
        posterior_mass = tf.exp(log_weights + log_unnormalized - increment)
        increment_score = tf.reduce_sum(posterior_mass[:, None] * local_score, axis=0)
        log_posterior = log_unnormalized - increment
        posterior_score = local_score - increment_score[None, :]
        increments.append(increment)
        increment_scores.append(increment_score)
        boundary_masses.append(
            tf.reduce_sum(tf.boolean_mask(posterior_mass, grid.boundary_mask))
        )

    increment_history = tf.stack(increments)
    increment_score_history = tf.stack(increment_scores)
    return {
        "reference_id": REFERENCE_ID,
        "objective": tf.reduce_sum(increment_history),
        "score": tf.reduce_sum(increment_score_history, axis=0),
        "increment_history": increment_history,
        "increment_score_history": increment_score_history,
        "boundary_mass_history": tf.stack(boundary_masses),
        "final_log_posterior": log_posterior,
        "final_log_posterior_score": posterior_score,
        "previous_marginal_score_status": (
            "stopped_negative_control"
            if stop_previous_marginal_score
            else "included_total_filtering_score"
        ),
    }


def dense_latent_sir_value(
    model: LatentPreclipSIRSSM,
    theta: tf.Tensor,
    observations: tf.Tensor,
    grids: Sequence[DenseGrid],
) -> tf.Tensor:
    """Return only the same fixed-grid scalar for autodiff/FD diagnostics."""

    return tf.convert_to_tensor(
        dense_latent_sir_value_and_manual_score(
            model, theta, observations, grids
        )["objective"],
        DTYPE,
    )


__all__ = [
    "DTYPE",
    "DenseGrid",
    "REFERENCE_ID",
    "dense_latent_sir_value",
    "dense_latent_sir_value_and_manual_score",
    "prepare_reduced_dense_grids",
    "reduced_latent_preclip_sir_model",
]
