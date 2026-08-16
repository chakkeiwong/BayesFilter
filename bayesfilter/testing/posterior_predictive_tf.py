"""Diagnostic-only TensorFlow posterior-predictive path construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import tensorflow as tf

from bayesfilter.testing.two_sample_energy_tf import (
    EnergyPermutationResult,
    whole_path_energy_permutation_test,
)


class PosteriorPredictiveDiagnosticError(ValueError):
    """Raised when posterior-predictive diagnostic inputs are invalid."""


@dataclass(frozen=True)
class EmpiricalPosteriorPredictivePaths:
    """One independently selected empirical-posterior row per simulated path."""

    paths: tf.Tensor
    posterior_indices: tf.Tensor
    selected_parameters: tf.Tensor
    posterior_draw_count: int
    path_count: int
    parameter_dim: int
    horizon: int
    posterior_seed: tuple[int, int]
    simulator_seed: tuple[int, int]
    sampling_with_replacement: bool


@dataclass(frozen=True)
class PosteriorPredictiveEnergyResult:
    """Posterior-predictive and true-parameter path banks plus energy result."""

    posterior_predictive: EmpiricalPosteriorPredictivePaths
    true_paths: tf.Tensor
    true_parameter: tf.Tensor
    energy: EnergyPermutationResult
    truth_simulator_seed: tuple[int, int]
    permutation_seed: tuple[int, int]


BatchConditionalSimulator = Callable[[tf.Tensor, tf.Tensor], Any]


def _seed_tuple(seed: Any, name: str) -> tuple[tf.Tensor, tuple[int, int]]:
    tensor = tf.convert_to_tensor(seed, tf.int32)
    if tensor.shape != (2,):
        raise PosteriorPredictiveDiagnosticError(f"{name} must have shape (2,)")
    values = tuple(int(value) for value in tensor.numpy().tolist())
    return tensor, (values[0], values[1])


def _positive_python_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PosteriorPredictiveDiagnosticError(
            f"{name} must be a positive Python integer"
        )
    return int(value)


def _posterior_draw_matrix(posterior_draws: Any) -> tuple[tf.Tensor, int, int]:
    draws = tf.convert_to_tensor(posterior_draws, tf.float64)
    if (
        draws.shape.rank != 2
        or not draws.shape.is_fully_defined()
        or int(draws.shape[0]) < 1
        or int(draws.shape[1]) < 1
    ):
        raise PosteriorPredictiveDiagnosticError(
            "posterior_draws must have finite static shape [draw,parameter]"
        )
    try:
        tf.debugging.assert_all_finite(draws, "posterior draws must be finite")
    except tf.errors.InvalidArgumentError as exc:
        raise PosteriorPredictiveDiagnosticError(
            "posterior draws must be finite"
        ) from exc
    return draws, int(draws.shape[0]), int(draws.shape[1])


def _path_matrix(paths: Any, *, path_count: int, label: str) -> tf.Tensor:
    values = tf.convert_to_tensor(paths, tf.float64)
    if (
        values.shape.rank != 2
        or not values.shape.is_fully_defined()
        or int(values.shape[0]) != path_count
        or int(values.shape[1]) < 1
    ):
        raise PosteriorPredictiveDiagnosticError(
            f"{label} must return exactly one complete path per parameter row"
        )
    try:
        tf.debugging.assert_all_finite(values, f"{label} paths must be finite")
    except tf.errors.InvalidArgumentError as exc:
        raise PosteriorPredictiveDiagnosticError(
            f"{label} paths must be finite"
        ) from exc
    return values


def _require_disjoint_seeds(**seeds: tuple[int, int]) -> None:
    values = tuple(seeds.values())
    if len(set(values)) != len(values):
        raise PosteriorPredictiveDiagnosticError(
            "posterior, simulator, truth, and permutation seed domains must be disjoint"
        )


def sample_empirical_posterior_predictive_paths(
    posterior_draws: Any,
    *,
    path_count: int,
    posterior_seed: Any,
    simulator_seed: Any,
    conditional_simulator: BatchConditionalSimulator,
) -> EmpiricalPosteriorPredictivePaths:
    """Sample iid paths from a finite equal-weight empirical posterior mixture.

    One archive index is sampled independently with replacement for every path.
    ``conditional_simulator`` receives the selected parameter matrix and one
    stateless seed and must return a rank-two matrix with one path per row.
    """

    count = _positive_python_int(path_count, "path_count")
    draws, draw_count, parameter_dim = _posterior_draw_matrix(posterior_draws)
    posterior_seed_tensor, posterior_seed_values = _seed_tuple(
        posterior_seed, "posterior_seed"
    )
    simulator_seed_tensor, simulator_seed_values = _seed_tuple(
        simulator_seed, "simulator_seed"
    )
    _require_disjoint_seeds(
        posterior=posterior_seed_values, simulator=simulator_seed_values
    )
    indices = tf.random.stateless_uniform(
        [count],
        posterior_seed_tensor,
        minval=0,
        maxval=draw_count,
        dtype=tf.int32,
        alg="philox",
    )
    selected = tf.gather(draws, indices, axis=0)
    paths = _path_matrix(
        conditional_simulator(selected, simulator_seed_tensor),
        path_count=count,
        label="conditional_simulator",
    )
    return EmpiricalPosteriorPredictivePaths(
        paths=paths,
        posterior_indices=indices,
        selected_parameters=selected,
        posterior_draw_count=draw_count,
        path_count=count,
        parameter_dim=parameter_dim,
        horizon=int(paths.shape[1]),
        posterior_seed=posterior_seed_values,
        simulator_seed=simulator_seed_values,
        sampling_with_replacement=True,
    )


def posterior_predictive_energy_test(
    posterior_draws: Any,
    true_parameter: Any,
    *,
    path_count: int,
    posterior_seed: Any,
    posterior_simulator_seed: Any,
    truth_simulator_seed: Any,
    permutation_seed: Any,
    conditional_simulator: BatchConditionalSimulator,
    permutation_count: int,
    permutation_batch_size: int = 250,
    jit_compile: bool = True,
) -> PosteriorPredictiveEnergyResult:
    """Compare an empirical posterior-predictive law with a true-parameter law."""

    count = _positive_python_int(path_count, "path_count")
    draws, _, parameter_dim = _posterior_draw_matrix(posterior_draws)
    truth = tf.convert_to_tensor(true_parameter, tf.float64)
    if truth.shape != (parameter_dim,):
        raise PosteriorPredictiveDiagnosticError(
            "true_parameter shape must match one posterior draw"
        )
    try:
        tf.debugging.assert_all_finite(truth, "true parameter must be finite")
    except tf.errors.InvalidArgumentError as exc:
        raise PosteriorPredictiveDiagnosticError(
            "true parameter must be finite"
        ) from exc
    _, posterior_seed_values = _seed_tuple(posterior_seed, "posterior_seed")
    _, posterior_simulator_seed_values = _seed_tuple(
        posterior_simulator_seed, "posterior_simulator_seed"
    )
    truth_seed_tensor, truth_seed_values = _seed_tuple(
        truth_simulator_seed, "truth_simulator_seed"
    )
    _, permutation_seed_values = _seed_tuple(
        permutation_seed, "permutation_seed"
    )
    _require_disjoint_seeds(
        posterior=posterior_seed_values,
        posterior_simulator=posterior_simulator_seed_values,
        truth_simulator=truth_seed_values,
        permutation=permutation_seed_values,
    )
    posterior_predictive = sample_empirical_posterior_predictive_paths(
        draws,
        path_count=count,
        posterior_seed=posterior_seed_values,
        simulator_seed=posterior_simulator_seed_values,
        conditional_simulator=conditional_simulator,
    )
    true_rows = tf.broadcast_to(truth[tf.newaxis, :], [count, parameter_dim])
    true_paths = _path_matrix(
        conditional_simulator(true_rows, truth_seed_tensor),
        path_count=count,
        label="conditional_simulator truth arm",
    )
    if true_paths.shape != posterior_predictive.paths.shape:
        raise PosteriorPredictiveDiagnosticError(
            "candidate and truth simulators returned different path shapes"
        )
    energy = whole_path_energy_permutation_test(
        posterior_predictive.paths,
        true_paths,
        permutation_count=permutation_count,
        seed=permutation_seed_values,
        permutation_batch_size=permutation_batch_size,
        jit_compile=jit_compile,
    )
    return PosteriorPredictiveEnergyResult(
        posterior_predictive=posterior_predictive,
        true_paths=true_paths,
        true_parameter=truth,
        energy=energy,
        truth_simulator_seed=truth_seed_values,
        permutation_seed=permutation_seed_values,
    )


__all__ = [
    "BatchConditionalSimulator",
    "EmpiricalPosteriorPredictivePaths",
    "PosteriorPredictiveDiagnosticError",
    "PosteriorPredictiveEnergyResult",
    "posterior_predictive_energy_test",
    "sample_empirical_posterior_predictive_paths",
]
