"""Randomized standard-Gaussian cloud designs for fixed particle filters."""

from __future__ import annotations

import tensorflow as tf
import tensorflow_probability as tfp


Tensor = tf.Tensor
SUPPORTED_DESIGNS = (
    "iid_gaussian",
    "antithetic_gaussian",
    "scrambled_halton_gaussian",
    "antithetic_scrambled_halton_gaussian",
    "latin_hypercube_gaussian",
    "orthogonal_gaussian_blocks",
)
def _seed(seed: int, salt: int) -> Tensor:
    modulus = 2_147_483_647
    return tf.constant([int(seed) % modulus, int(salt) % modulus], tf.int32)


def _normal_quantile(uniforms: Tensor, dtype: tf.dtypes.DType) -> Tensor:
    uniforms = tf.cast(uniforms, dtype)
    zero = tf.zeros([], dtype)
    one = tf.ones([], dtype)
    lower = tf.math.nextafter(zero, one)
    upper = tf.math.nextafter(one, zero)
    guarded = tf.clip_by_value(uniforms, lower, upper)
    return tf.math.ndtri(guarded)


def standard_normal_cloud(
    design: str,
    *,
    num_particles: int,
    dimension: int,
    seed: int,
    salt: int = 0,
    dtype: tf.dtypes.DType = tf.float32,
) -> Tensor:
    """Return one randomized cloud whose rows are marginally standard Gaussian."""

    if design not in SUPPORTED_DESIGNS:
        raise ValueError(f"unsupported Gaussian cloud design: {design}")
    if num_particles < 2 or dimension < 1:
        raise ValueError("particle count and dimension must be positive")
    if design in {
        "antithetic_gaussian",
        "antithetic_scrambled_halton_gaussian",
    } and num_particles % 2:
        raise ValueError("antithetic designs require an even particle count")
    if design == "orthogonal_gaussian_blocks" and num_particles % dimension:
        raise ValueError("orthogonal design requires N divisible by dimension")

    if design == "iid_gaussian":
        cloud = tf.random.stateless_normal(
            [num_particles, dimension], _seed(seed, 101 + salt), dtype=dtype
        )
    elif design == "antithetic_gaussian":
        half = tf.random.stateless_normal(
            [num_particles // 2, dimension],
            _seed(seed, 211 + salt),
            dtype=dtype,
        )
        cloud = tf.concat([half, -half], axis=0)
    elif design == "scrambled_halton_gaussian":
        uniforms = tfp.mcmc.sample_halton_sequence(
            dimension,
            num_results=num_particles,
            dtype=dtype,
            randomized=True,
            seed=_seed(seed, 307 + salt),
        )
        cloud = _normal_quantile(uniforms, dtype)
    elif design == "antithetic_scrambled_halton_gaussian":
        uniforms = tfp.mcmc.sample_halton_sequence(
            dimension,
            num_results=num_particles // 2,
            dtype=dtype,
            randomized=True,
            seed=_seed(seed, 401 + salt),
        )
        half = _normal_quantile(uniforms, dtype)
        cloud = tf.concat([half, -half], axis=0)
    elif design == "latin_hypercube_gaussian":
        rows = tf.range(num_particles, dtype=tf.int32)
        columns = []
        for axis in range(dimension):
            permutation = tf.random.experimental.stateless_shuffle(
                rows, seed=_seed(seed, 503 + salt + 37 * axis)
            )
            jitter = tf.random.stateless_uniform(
                [num_particles],
                seed=_seed(seed, 509 + salt + 37 * axis),
                dtype=dtype,
            )
            columns.append(
                (tf.cast(permutation, dtype) + jitter)
                / tf.cast(num_particles, dtype)
            )
        cloud = _normal_quantile(tf.stack(columns, axis=1), dtype)
    else:
        block_count = num_particles // dimension
        matrices = tf.random.stateless_normal(
            [block_count, dimension, dimension],
            seed=_seed(seed, 601 + salt),
            dtype=dtype,
        )
        orthogonal, triangular = tf.linalg.qr(matrices, full_matrices=False)
        signs = tf.where(
            tf.linalg.diag_part(triangular) < 0.0,
            -tf.ones([block_count, dimension], dtype),
            tf.ones([block_count, dimension], dtype),
        )
        directions = orthogonal * signs[:, None, :]
        gamma = tf.random.stateless_gamma(
            [block_count, dimension],
            seed=_seed(seed, 607 + salt),
            alpha=tf.cast(dimension / 2.0, dtype),
            beta=tf.cast(0.5, dtype),
            dtype=dtype,
        )
        radii = tf.sqrt(gamma)
        cloud = tf.reshape(
            tf.transpose(directions, [0, 2, 1]) * radii[:, :, None],
            [num_particles, dimension],
        )

    cloud = tf.ensure_shape(cloud, [num_particles, dimension])
    tf.debugging.assert_all_finite(cloud, "Gaussian cloud must be finite")
    return cloud


def cloud_diagnostics(cloud: Tensor) -> dict[str, Tensor]:
    """Return low-order diagnostics without changing the cloud."""

    cloud = tf.convert_to_tensor(cloud)
    count = tf.cast(tf.shape(cloud)[0], cloud.dtype)
    mean = tf.reduce_mean(cloud, axis=0)
    centered = cloud - mean[None, :]
    covariance = tf.einsum("ni,nj->ij", centered, centered) / count
    identity = tf.eye(tf.shape(cloud)[1], dtype=cloud.dtype)
    return {
        "maximum_absolute_mean": tf.reduce_max(tf.abs(mean)),
        "covariance_frobenius_error": tf.linalg.norm(covariance - identity),
        "maximum_absolute_value": tf.reduce_max(tf.abs(cloud)),
    }


__all__ = ["SUPPORTED_DESIGNS", "cloud_diagnostics", "standard_normal_cloud"]
