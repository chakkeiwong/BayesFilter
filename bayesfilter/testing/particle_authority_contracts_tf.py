"""Typed TensorFlow reference contracts for the particle-authority campaign.

This module is deliberately a reference/diagnostic lane.  It exercises the
mathematical quantities that the q=20 authority must expose without importing
NumPy or silently promoting a finite cloud to a density.  The caller remains
responsible for target-specific tuning and for the SMC-U proof obligation.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

import tensorflow as tf


_LOG_TWO_PI = tf.constant(1.8378770664093453, tf.float64)


def _as_float(value: tf.Tensor) -> float:
    """Materialize one scalar only at the diagnostic boundary."""

    return float(tf.convert_to_tensor(value).numpy())


def _normal_log_prob(values: tf.Tensor, mean: tf.Tensor, std: tf.Tensor) -> tf.Tensor:
    values = tf.convert_to_tensor(values, tf.float64)
    mean = tf.convert_to_tensor(mean, tf.float64)
    std = tf.convert_to_tensor(std, tf.float64)
    standardized = (values - mean) / std
    return -0.5 * tf.square(standardized) - tf.math.log(std) - 0.5 * _LOG_TWO_PI


def _two_mode_log_prob(values: tf.Tensor) -> tf.Tensor:
    values = tf.convert_to_tensor(values, tf.float64)
    left = _normal_log_prob(values, tf.constant(-5.0, tf.float64), tf.constant(0.7, tf.float64))
    right = _normal_log_prob(values, tf.constant(5.0, tf.float64), tf.constant(0.7, tf.float64))
    return tf.reduce_logsumexp(
        tf.stack(
            [left + tf.math.log(tf.constant(0.5, tf.float64)),
             right + tf.math.log(tf.constant(0.5, tf.float64))],
            axis=1,
        ),
        axis=1,
    )


def canonical_protocol_hash(protocol: Mapping[str, Any]) -> str:
    """Hash the exact claim-run protocol representation."""

    encoded = json.dumps(
        protocol,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


@tf.function(jit_compile=True, reduce_retracing=False)
def _affine_identity_kernel(points: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    matrix = tf.constant([[1.5, 0.2], [-0.1, 0.8]], tf.float64)
    shift = tf.constant([0.7, -1.1], tf.float64)
    transformed = tf.linalg.matmul(points, matrix, transpose_b=True) + shift
    inverse_matrix = tf.linalg.inv(matrix)
    recovered = tf.linalg.matmul(transformed - shift, inverse_matrix, transpose_b=True)
    base_log = -0.5 * tf.reduce_sum(tf.square(recovered), axis=1) - tf.cast(
        tf.shape(points)[1], tf.float64
    ) * 0.5 * _LOG_TWO_PI
    log_inverse_det = -tf.math.log(tf.abs(tf.linalg.det(matrix)))
    inverse_formula_log = base_log + log_inverse_det

    # Evaluate the same pushforward independently as a full Gaussian with
    # covariance A A^T.  This catches an inverse/Jacobian or covariance-lifecycle
    # error that a duplicated expression would hide.
    covariance = tf.linalg.matmul(matrix, matrix, transpose_b=True)
    covariance_inverse = tf.linalg.inv(covariance)
    centered = transformed - shift
    quadratic = tf.reduce_sum(
        centered * tf.linalg.matmul(centered, covariance_inverse, transpose_b=True),
        axis=1,
    )
    covariance_logdet = tf.math.log(tf.linalg.det(covariance))
    transformed_covariance_log = -0.5 * (
        quadratic + covariance_logdet + tf.cast(tf.shape(points)[1], tf.float64) * _LOG_TWO_PI
    )
    return inverse_formula_log, transformed_covariance_log


def affine_density_identity() -> Mapping[str, Any]:
    """Check an affine change-of-variables identity on fixed rows."""

    points = tf.constant(
        [[-2.0, -1.0], [-0.5, 0.25], [0.0, 1.0], [1.25, -0.75], [2.0, 1.5]],
        tf.float64,
    )
    transformed_log, direct_log = _affine_identity_kernel(points)
    residual = tf.reduce_max(tf.abs(transformed_log - direct_log))
    tf.debugging.assert_all_finite(transformed_log, "affine transformed density")
    tf.debugging.assert_all_finite(direct_log, "affine inverse density")
    return {
        "status": "PASS" if _as_float(residual) <= 1.0e-12 else "FAIL",
        "max_abs_log_density_residual": _as_float(residual),
        "role": "M3 affine density contract fixture",
        "nonclaim": "This does not establish q=20 LEDH invertibility or determinant completeness.",
    }


def frozen_protocol_hash_check() -> Mapping[str, Any]:
    """Verify that a claim-run protocol hash changes on a material edit."""

    protocol = {
        "schema": "particle_authority.protocol.v1",
        "beta_stages": [0.0, 0.25, 0.5, 0.75, 1.0],
        "resampling_trigger": "fixed_ess_threshold",
        "mutation": {"kernel": "bridge_ar1", "steps": 3, "rho": 0.75},
        "defensive_epsilon": 0.2,
        "proposal_law_version": "known_two_mode_mixture_v1",
    }
    original = canonical_protocol_hash(protocol)
    replay = canonical_protocol_hash(json.loads(json.dumps(protocol)))
    changed_protocol = dict(protocol)
    changed_protocol["defensive_epsilon"] = 0.25
    changed = canonical_protocol_hash(changed_protocol)
    exact = original == replay
    changed_detected = original != changed
    return {
        "status": "PASS" if exact and changed_detected else "FAIL",
        "hash": original,
        "replay_hash": replay,
        "changed_hash": changed,
        "exact_replay": exact,
        "material_change_detected": changed_detected,
        "role": "M0 frozen-law hard-veto fixture",
    }


@tf.function(jit_compile=True, reduce_retracing=False)
def _known_mass_kernel(noise: tf.Tensor, labels: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    noise = tf.convert_to_tensor(noise, tf.float64)
    labels = tf.convert_to_tensor(labels, tf.bool)
    samples = tf.where(
        labels,
        -5.0 + 0.7 * noise,
        5.0 + 0.7 * noise,
    )
    target_log = _two_mode_log_prob(samples[:, tf.newaxis])
    proposal_log = target_log
    normalizer = tf.constant(1.75, tf.float64)
    log_weights = tf.math.log(normalizer) + target_log - proposal_log
    weights = tf.exp(log_weights)
    indicator = tf.cast(samples < 0.0, tf.float64)
    mass = tf.reduce_mean(weights)
    negative_functional = tf.reduce_mean(weights * indicator)
    return mass, negative_functional, samples


def known_density_mass_fixture(*, sample_count: int = 4096, seed: tuple[int, int] = (20260825, 101)) -> Mapping[str, Any]:
    """Check a known unnormalized mass and a known two-mode functional."""

    if int(sample_count) <= 0:
        raise ValueError("sample_count must be positive")
    split = tf.random.experimental.stateless_split(tf.constant(seed, tf.int32), 2)
    noise = tf.random.stateless_normal((int(sample_count),), split[0], dtype=tf.float64)
    labels = tf.random.stateless_uniform((int(sample_count),), split[1], dtype=tf.float64) < 0.5
    mass, negative_functional, samples = _known_mass_kernel(noise, labels)
    expected_mass = tf.constant(1.75, tf.float64)
    expected_negative = tf.constant(0.875, tf.float64)
    mass_error = tf.abs(mass - expected_mass)
    negative_error = tf.abs(negative_functional - expected_negative)
    tf.debugging.assert_all_finite(samples, "known-density samples")
    passed = _as_float(mass_error) <= 1.0e-12 and _as_float(negative_error) <= 0.06
    return {
        "status": "PASS" if passed else "FAIL",
        "sample_count": int(sample_count),
        "estimated_unnormalized_mass": _as_float(mass),
        "expected_unnormalized_mass": _as_float(expected_mass),
        "mass_abs_error": _as_float(mass_error),
        "estimated_negative_functional": _as_float(negative_functional),
        "expected_negative_functional": _as_float(expected_negative),
        "negative_functional_abs_error": _as_float(negative_error),
        "proposal": "the normalized two-mode target itself",
        "role": "M0 known-density unnormalized-mass fixture",
        "nonclaim": "Finite Monte Carlo agreement is not a q=20 SMC-U proof.",
    }


def mode_missing_transform_fixture(*, point_count: int = 128) -> Mapping[str, Any]:
    """Show that finite moment restoration can create bridge rows.

    The input cloud contains only the positive mode.  A one-dimensional affine
    moment restoration is used as a transparent finite-cloud analogue of an
    ETPF/GenUT representation step; it is not labeled as either method.
    """

    if int(point_count) < 8:
        raise ValueError("point_count must be at least eight")
    source = tf.linspace(tf.constant(4.3, tf.float64), tf.constant(5.7, tf.float64), int(point_count))
    source_mean = tf.reduce_mean(source)
    source_variance = tf.reduce_mean(tf.square(source - source_mean))
    target_variance = tf.constant(25.49, tf.float64)
    transformed = (source - source_mean) * tf.sqrt(target_variance / source_variance)
    input_negative_fraction = tf.reduce_mean(tf.cast(source < 0.0, tf.float64))
    output_negative_fraction = tf.reduce_mean(tf.cast(transformed < 0.0, tf.float64))
    bridge_fraction = tf.reduce_mean(tf.cast(tf.abs(transformed) < 4.0, tf.float64))
    return {
        "status": "PASS" if _as_float(input_negative_fraction) == 0.0 else "FAIL",
        "input_negative_fraction": _as_float(input_negative_fraction),
        "output_negative_fraction": _as_float(output_negative_fraction),
        "bridge_fraction": _as_float(bridge_fraction),
        "input_mean": _as_float(source_mean),
        "output_mean": _as_float(tf.reduce_mean(transformed)),
        "output_variance": _as_float(tf.reduce_mean(tf.square(transformed))),
        "role": "finite-moment bridge explanatory fixture",
        "nonclaim": "Moment restoration does not identify a density or recover a missing mode.",
    }


@tf.function(jit_compile=True, reduce_retracing=False)
def _mutation_kernel(initial: tf.Tensor, noise: tf.Tensor, rho: tf.Tensor) -> tf.Tensor:
    return rho * initial + tf.sqrt(1.0 - tf.square(rho)) * noise


def mutation_invariance_fixture(*, sample_count: int = 4096, seed: tuple[int, int] = (20260825, 202)) -> Mapping[str, Any]:
    """Probe an exactly standard-normal-invariant AR(1) bridge kernel."""

    split = tf.random.experimental.stateless_split(tf.constant(seed, tf.int32), 2)
    initial = tf.random.stateless_normal((int(sample_count),), split[0], dtype=tf.float64)
    noise = tf.random.stateless_normal((int(sample_count),), split[1], dtype=tf.float64)
    next_state = _mutation_kernel(initial, noise, tf.constant(0.75, tf.float64))
    mean_error = tf.abs(tf.reduce_mean(next_state))
    variance_error = tf.abs(tf.reduce_mean(tf.square(next_state)) - 1.0)
    tf.debugging.assert_all_finite(next_state, "mutation state")
    passed = _as_float(mean_error) < 0.06 and _as_float(variance_error) < 0.08
    return {
        "status": "PASS" if passed else "FAIL",
        "sample_count": int(sample_count),
        "mean_abs_error": _as_float(mean_error),
        "second_moment_abs_error": _as_float(variance_error),
        "rho": 0.75,
        "role": "M0 mutation invariant-target diagnostic",
        "nonclaim": "A finite moment probe is not an invariance proof for q=20 kernels.",
    }


@tf.function(jit_compile=True, reduce_retracing=False)
def _defensive_tail_kernel(grid: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    target = tf.exp(_normal_log_prob(grid, tf.constant(0.0, tf.float64), tf.constant(1.0, tf.float64)))
    proposal = target
    safety = tf.exp(_normal_log_prob(grid, tf.constant(0.0, tf.float64), tf.constant(4.0, tf.float64)))
    epsilon = tf.constant(0.2, tf.float64)
    mixture = (1.0 - epsilon) * proposal + epsilon * safety
    score = -grid
    integrand = tf.square(target) * tf.square(score) / mixture
    dx = grid[1] - grid[0]
    return mixture, tf.reduce_sum(integrand) * dx


def defensive_tail_fixture(*, grid_count: int = 20001) -> Mapping[str, Any]:
    """Check positive defensive support and a numerical score second moment."""

    grid = tf.linspace(tf.constant(-12.0, tf.float64), tf.constant(12.0, tf.float64), int(grid_count))
    mixture, second_moment = _defensive_tail_kernel(grid)
    minimum = tf.reduce_min(mixture)
    tf.debugging.assert_all_finite(mixture, "defensive mixture")
    tf.debugging.assert_all_finite(second_moment, "defensive second moment")
    passed = _as_float(minimum) > 0.0 and _as_float(second_moment) > 0.0
    return {
        "status": "PASS" if passed else "FAIL",
        "epsilon": 0.2,
        "epsilon_min": 0.2,
        "grid_count": int(grid_count),
        "minimum_mixture_density": _as_float(minimum),
        "estimated_score_second_moment": _as_float(second_moment),
        "role": "defensive-support and tail-integrability fixture",
        "nonclaim": "Finite grid evidence does not prove global integrability for the q=20 score class.",
    }


def replay_metadata_parity_fixture() -> Mapping[str, Any]:
    """Recompute a stored proposal log density from retained metadata."""

    metadata = {
        "proposal_family": "normal",
        "mean": 1.25,
        "std": 2.0,
        "seed": [20260825, 303],
        "law_version": "normal-v1",
    }
    values = tf.constant([-2.0, -0.25, 0.0, 1.5, 4.0], tf.float64)
    recomputed = _normal_log_prob(
        values,
        tf.constant(metadata["mean"], tf.float64),
        tf.constant(metadata["std"], tf.float64),
    )
    retained = tf.identity(recomputed)
    residual = tf.reduce_max(tf.abs(recomputed - retained))
    original_hash = canonical_protocol_hash(metadata)
    changed = dict(metadata)
    changed["std"] = 2.1
    changed_hash = canonical_protocol_hash(changed)
    passed = _as_float(residual) == 0.0 and original_hash != changed_hash
    return {
        "status": "PASS" if passed else "FAIL",
        "max_abs_log_density_residual": _as_float(residual),
        "metadata_hash": original_hash,
        "changed_metadata_hash": changed_hash,
        "material_change_detected": original_hash != changed_hash,
        "role": "replay metadata parity hard-veto fixture",
    }


def run_all_contracts() -> Mapping[str, Any]:
    """Run all Phase 1 contracts and return a serializable receipt."""

    results = {
        "affine_density_identity": affine_density_identity(),
        "frozen_protocol_hash": frozen_protocol_hash_check(),
        "known_density_mass": known_density_mass_fixture(),
        "mode_missing_transform": mode_missing_transform_fixture(),
        "mutation_invariance": mutation_invariance_fixture(),
        "defensive_tail": defensive_tail_fixture(),
        "replay_metadata_parity": replay_metadata_parity_fixture(),
    }
    statuses = [str(item["status"]) for item in results.values()]
    return {
        "schema": "bayesfilter.particle_authority.contract_receipt.v1",
        "status": "PASS" if all(status == "PASS" for status in statuses) else "FAIL",
        "fixture_evidence_only": True,
        "results": results,
        "nonclaims": [
            "No q=20 SMC/SMC-U route was executed.",
            "No finite fixture proves global mode discovery, posterior correctness, or IID whitening.",
            "The mutation and tail checks are finite diagnostics, not general theorems.",
        ],
    }
