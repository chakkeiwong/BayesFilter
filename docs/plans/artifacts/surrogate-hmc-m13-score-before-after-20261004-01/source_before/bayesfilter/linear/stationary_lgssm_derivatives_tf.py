"""TensorFlow first derivatives for stationary linear Gaussian models.

This module builds first-order derivatives for the stationary continuous-time
to discrete-time LGSSM map used by large MacroFinance-style targets. It is a
generic BayesFilter utility: client projects provide economic primitive tensors
and their direct first derivatives, while BayesFilter owns the differentiable
stationary transforms needed by SVD score backends.

No generic autodiff Jacobian construction is used here. Tests may use finite
differences as an oracle, but the runtime formulas below are analytic linear
solves and Frechet derivatives.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import tensorflow as tf

from bayesfilter.linear.types_tf import (
    TFLinearGaussianStateSpace,
    TFLinearGaussianStateSpaceFirstDerivatives,
)


def _require_rank(tensor: tf.Tensor, name: str, rank: int) -> None:
    if tensor.shape.rank != rank:
        raise ValueError(f"{name} must have rank {rank}, got {tensor.shape.rank}")


def _static_dim(tensor: tf.Tensor, axis: int, name: str) -> int:
    dim = tensor.shape[axis]
    if dim is None:
        raise ValueError(f"{name} requires static dimension at axis {axis}")
    return int(dim)


def _kron(left: tf.Tensor, right: tf.Tensor) -> tf.Tensor:
    left = tf.convert_to_tensor(left, dtype=tf.float64)
    right = tf.convert_to_tensor(right, dtype=tf.float64)
    product = tf.einsum("ab,cd->acbd", left, right)
    return tf.reshape(
        product,
        (
            int(left.shape[0]) * int(right.shape[0]),
            int(left.shape[1]) * int(right.shape[1]),
        ),
    )


def continuous_lyapunov_solution_tf(
    drift: Any,
    diffusion: Any,
) -> tf.Tensor:
    """Solve ``K Omega + Omega K' = D`` for a stable drift matrix ``K``."""

    drift = tf.convert_to_tensor(drift, dtype=tf.float64)
    diffusion = tf.convert_to_tensor(diffusion, dtype=tf.float64)
    _require_rank(drift, "drift", 2)
    _require_rank(diffusion, "diffusion", 2)
    state_dim = _static_dim(drift, 0, "drift")
    if drift.shape[1] != state_dim or tuple(diffusion.shape.as_list()) != (state_dim, state_dim):
        raise ValueError("drift and diffusion must be square with matching static state dimension")
    identity = tf.eye(state_dim, dtype=tf.float64)
    system = _kron(drift, identity) + _kron(identity, drift)
    rhs = tf.reshape(diffusion, (state_dim * state_dim, 1))
    solution = tf.reshape(tf.linalg.solve(system, rhs), (state_dim, state_dim))
    return 0.5 * (solution + tf.transpose(solution))


def continuous_lyapunov_first_derivatives_tf(
    *,
    drift: Any,
    diffusion: Any,
    d_drift: Any,
    d_diffusion: Any,
    stationary_covariance: Any | None = None,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Return ``Omega`` and first derivatives of a continuous Lyapunov solve.

    Differentiating ``K Omega + Omega K' = D`` gives

    ``K dOmega_i + dOmega_i K' = dD_i - dK_i Omega - Omega dK_i'``.
    """

    drift = tf.convert_to_tensor(drift, dtype=tf.float64)
    diffusion = tf.convert_to_tensor(diffusion, dtype=tf.float64)
    d_drift = tf.convert_to_tensor(d_drift, dtype=tf.float64)
    d_diffusion = tf.convert_to_tensor(d_diffusion, dtype=tf.float64)
    _require_rank(d_drift, "d_drift", 3)
    _require_rank(d_diffusion, "d_diffusion", 3)
    parameter_dim = _static_dim(d_drift, 0, "d_drift")
    state_dim = _static_dim(drift, 0, "drift")
    expected = (parameter_dim, state_dim, state_dim)
    if tuple(d_drift.shape.as_list()) != expected:
        raise ValueError(f"d_drift must have shape {expected}")
    if tuple(d_diffusion.shape.as_list()) != expected:
        raise ValueError(f"d_diffusion must have shape {expected}")
    omega = (
        continuous_lyapunov_solution_tf(drift, diffusion)
        if stationary_covariance is None
        else tf.convert_to_tensor(stationary_covariance, dtype=tf.float64)
    )
    identity = tf.eye(state_dim, dtype=tf.float64)
    system = _kron(drift, identity) + _kron(identity, drift)
    rhs = d_diffusion - d_drift @ omega - omega[None, :, :] @ tf.linalg.matrix_transpose(d_drift)
    batched_system = tf.broadcast_to(
        system,
        (parameter_dim, state_dim * state_dim, state_dim * state_dim),
    )
    d_omega = tf.reshape(
        tf.linalg.solve(
            batched_system,
            tf.reshape(rhs, (parameter_dim, state_dim * state_dim, 1)),
        ),
        (parameter_dim, state_dim, state_dim),
    )
    d_omega = 0.5 * (d_omega + tf.linalg.matrix_transpose(d_omega))
    return omega, d_omega


def matrix_exponential_frechet_tf(
    matrix: Any,
    directions: Any,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Return ``expm(matrix)`` and Frechet derivatives in ``directions``.

    For each direction ``E_i`` this uses the standard block-matrix identity:

    ``expm([[A, E_i], [0, A]])[:n, n:] = L_expm(A, E_i)``.
    """

    matrix = tf.convert_to_tensor(matrix, dtype=tf.float64)
    directions = tf.convert_to_tensor(directions, dtype=tf.float64)
    _require_rank(matrix, "matrix", 2)
    _require_rank(directions, "directions", 3)
    parameter_dim = _static_dim(directions, 0, "directions")
    state_dim = _static_dim(matrix, 0, "matrix")
    if matrix.shape[1] != state_dim or tuple(directions.shape.as_list()) != (
        parameter_dim,
        state_dim,
        state_dim,
    ):
        raise ValueError("matrix and directions must have matching static square dimensions")
    exp_matrix = tf.linalg.expm(matrix)
    zeros = tf.zeros_like(matrix)
    blocks = tf.concat(
        [
            tf.concat([tf.broadcast_to(matrix, (parameter_dim, state_dim, state_dim)), directions], axis=2),
            tf.concat(
                [
                    tf.broadcast_to(zeros, (parameter_dim, state_dim, state_dim)),
                    tf.broadcast_to(matrix, (parameter_dim, state_dim, state_dim)),
                ],
                axis=2,
            ),
        ],
        axis=1,
    )
    exp_blocks = tf.linalg.expm(blocks)
    d_exp = exp_blocks[:, :state_dim, state_dim:]
    return exp_matrix, d_exp


def diffusion_from_cholesky_first_derivatives_tf(
    factor: Any,
    d_factor: Any,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Return ``D = Sigma Sigma'`` and first derivatives from a diffusion factor."""

    factor = tf.convert_to_tensor(factor, dtype=tf.float64)
    d_factor = tf.convert_to_tensor(d_factor, dtype=tf.float64)
    _require_rank(factor, "factor", 2)
    _require_rank(d_factor, "d_factor", 3)
    parameter_dim = _static_dim(d_factor, 0, "d_factor")
    state_dim = _static_dim(factor, 0, "factor")
    if factor.shape[1] != state_dim or tuple(d_factor.shape.as_list()) != (
        parameter_dim,
        state_dim,
        state_dim,
    ):
        raise ValueError("factor and d_factor must have matching static square dimensions")
    diffusion = factor @ tf.transpose(factor)
    d_diffusion = d_factor @ tf.transpose(factor) + factor[None, :, :] @ tf.linalg.matrix_transpose(d_factor)
    d_diffusion = 0.5 * (d_diffusion + tf.linalg.matrix_transpose(d_diffusion))
    return diffusion, d_diffusion


def stationary_lgssm_from_continuous_first_derivatives_tf(
    *,
    drift: Any,
    long_run_mean: Any,
    diffusion_factor: Any,
    d_drift: Any,
    d_long_run_mean: Any,
    d_diffusion_factor: Any,
    observation_offset: Any,
    observation_matrix: Any,
    observation_covariance: Any,
    d_observation_offset: Any,
    d_observation_matrix: Any,
    d_observation_covariance: Any,
) -> tuple[TFLinearGaussianStateSpace, TFLinearGaussianStateSpaceFirstDerivatives]:
    """Build a stationary discrete LGSSM and first derivatives.

    The continuous-time primitives are ``dX = K(theta - X) dt + Sigma dW``.
    The returned discrete law uses unit time step:

    ``Phi = expm(-K)``, ``c = (I - Phi) theta``,
    ``Q = Omega - Phi Omega Phi'``, and
    ``K Omega + Omega K' = Sigma Sigma'``.
    """

    drift = tf.convert_to_tensor(drift, dtype=tf.float64)
    long_run_mean = tf.convert_to_tensor(long_run_mean, dtype=tf.float64)
    diffusion_factor = tf.convert_to_tensor(diffusion_factor, dtype=tf.float64)
    d_drift = tf.convert_to_tensor(d_drift, dtype=tf.float64)
    d_long_run_mean = tf.convert_to_tensor(d_long_run_mean, dtype=tf.float64)
    d_diffusion_factor = tf.convert_to_tensor(d_diffusion_factor, dtype=tf.float64)
    observation_offset = tf.convert_to_tensor(observation_offset, dtype=tf.float64)
    observation_matrix = tf.convert_to_tensor(observation_matrix, dtype=tf.float64)
    observation_covariance = tf.convert_to_tensor(observation_covariance, dtype=tf.float64)
    d_observation_offset = tf.convert_to_tensor(d_observation_offset, dtype=tf.float64)
    d_observation_matrix = tf.convert_to_tensor(d_observation_matrix, dtype=tf.float64)
    d_observation_covariance = tf.convert_to_tensor(d_observation_covariance, dtype=tf.float64)

    _require_rank(long_run_mean, "long_run_mean", 1)
    _require_rank(d_long_run_mean, "d_long_run_mean", 2)
    parameter_dim = _static_dim(d_drift, 0, "d_drift")
    state_dim = _static_dim(drift, 0, "drift")
    observation_dim = _static_dim(observation_offset, 0, "observation_offset")
    if tuple(d_long_run_mean.shape.as_list()) != (parameter_dim, state_dim):
        raise ValueError("d_long_run_mean has incompatible shape")
    if tuple(d_observation_offset.shape.as_list()) != (parameter_dim, observation_dim):
        raise ValueError("d_observation_offset has incompatible shape")
    if tuple(d_observation_matrix.shape.as_list()) != (parameter_dim, observation_dim, state_dim):
        raise ValueError("d_observation_matrix has incompatible shape")
    if tuple(d_observation_covariance.shape.as_list()) != (
        parameter_dim,
        observation_dim,
        observation_dim,
    ):
        raise ValueError("d_observation_covariance has incompatible shape")

    phi, d_phi = matrix_exponential_frechet_tf(-drift, -d_drift)
    identity = tf.eye(state_dim, dtype=tf.float64)
    diffusion, d_diffusion = diffusion_from_cholesky_first_derivatives_tf(
        diffusion_factor,
        d_diffusion_factor,
    )
    omega, d_omega = continuous_lyapunov_first_derivatives_tf(
        drift=drift,
        diffusion=diffusion,
        d_drift=d_drift,
        d_diffusion=d_diffusion,
    )
    transition_offset = tf.linalg.matvec(identity - phi, long_run_mean)
    d_transition_offset = (
        -tf.linalg.matvec(d_phi, long_run_mean)
        + tf.linalg.matvec(identity - phi, d_long_run_mean)
    )
    transition_covariance = omega - phi @ omega @ tf.transpose(phi)
    transition_covariance = 0.5 * (transition_covariance + tf.transpose(transition_covariance))
    d_transition_covariance = (
        d_omega
        - d_phi @ omega @ tf.transpose(phi)
        - phi[None, :, :] @ d_omega @ tf.transpose(phi)[None, :, :]
        - phi[None, :, :] @ omega[None, :, :] @ tf.linalg.matrix_transpose(d_phi)
    )
    d_transition_covariance = 0.5 * (
        d_transition_covariance + tf.linalg.matrix_transpose(d_transition_covariance)
    )
    solve_rhs = tf.reshape(transition_offset, (state_dim, 1))
    solve_matrix = identity - phi
    initial_mean = tf.linalg.solve(solve_matrix, solve_rhs)[:, 0]
    batched_solve_matrix = tf.broadcast_to(
        solve_matrix,
        (parameter_dim, state_dim, state_dim),
    )
    d_initial_mean = tf.linalg.solve(
        batched_solve_matrix,
        tf.reshape(d_transition_offset + tf.linalg.matvec(d_phi, initial_mean), (parameter_dim, state_dim, 1)),
    )[:, :, 0]
    model = TFLinearGaussianStateSpace(
        initial_mean=initial_mean,
        initial_covariance=omega,
        transition_offset=transition_offset,
        transition_matrix=phi,
        transition_covariance=transition_covariance,
        observation_offset=observation_offset,
        observation_matrix=observation_matrix,
        observation_covariance=observation_covariance,
    )
    derivatives = TFLinearGaussianStateSpaceFirstDerivatives(
        d_initial_mean=d_initial_mean,
        d_initial_covariance=d_omega,
        d_transition_offset=d_transition_offset,
        d_transition_matrix=d_phi,
        d_transition_covariance=d_transition_covariance,
        d_observation_offset=d_observation_offset,
        d_observation_matrix=d_observation_matrix,
        d_observation_covariance=d_observation_covariance,
    )
    return model, derivatives


@dataclass(frozen=True)
class StationaryLGSSMFirstDerivativeCoverage:
    """Eager metadata for first-derivative parameter coverage."""

    parameter_names: tuple[str, ...]
    block_names: tuple[str, ...]
    covered_parameter_names: tuple[str, ...]
    missing_parameter_names: tuple[str, ...]
    coverage_complete: bool
    source: str = "stationary_lgssm_first_derivative_coverage"


def stationary_lgssm_first_derivative_coverage(
    parameter_names: tuple[str, ...] | list[str],
    derivatives: TFLinearGaussianStateSpaceFirstDerivatives,
    *,
    atol: float = 0.0,
) -> StationaryLGSSMFirstDerivativeCoverage:
    """Return eager coverage metadata for a first-derivative payload."""

    names = tuple(str(name) for name in parameter_names)
    if derivatives.parameter_dim != len(names):
        raise ValueError("parameter_names length does not match derivatives.parameter_dim")
    blocks = tuple(derivatives.__dataclass_fields__)
    covered = tf.zeros((len(names),), dtype=tf.bool)
    threshold = tf.constant(float(atol), dtype=tf.float64)
    for block in blocks:
        tensor = tf.convert_to_tensor(getattr(derivatives, block), dtype=tf.float64)
        axes = tuple(range(1, tensor.shape.rank))
        covered = tf.logical_or(covered, tf.reduce_any(tf.abs(tensor) > threshold, axis=axes))
    covered_np = tuple(bool(item) for item in covered.numpy().tolist())
    covered_names = tuple(name for name, is_covered in zip(names, covered_np, strict=True) if is_covered)
    missing_names = tuple(name for name, is_covered in zip(names, covered_np, strict=True) if not is_covered)
    return StationaryLGSSMFirstDerivativeCoverage(
        parameter_names=names,
        block_names=blocks,
        covered_parameter_names=covered_names,
        missing_parameter_names=missing_names,
        coverage_complete=not missing_names,
    )
