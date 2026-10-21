"""Prospective diagnostic candidate; no admission or default-policy authority.

The randomization null is exchangeable vectors of batch means. This is exact
for independent observations and an empirical hypothesis for MCMC batches.
"""
from functools import lru_cache

import tensorflow as tf


def permutation_statistics(means, permutations):
    """Joint-function lag test with shared batch permutations across chains."""
    if means.shape.rank != 3 or any(size is None for size in means.shape):
        raise ValueError("static [chain, batch, function] means required")
    chains, batches, functions = means.shape
    if chains < 2 or batches < 4 or functions < 1:
        raise ValueError("at least two chains, four batches and one function required")
    if permutations.shape.rank != 2 or permutations.shape[1] != batches:
        raise ValueError("permutations must have shape [replication, batch]")
    if permutations.shape[0] is None or permutations.shape[0] < 1:
        raise ValueError("positive static permutation count required")
    centered = means - tf.reduce_mean(means, axis=1, keepdims=True)
    denominator = tf.reduce_sum(tf.square(centered), axis=[0, 1])
    lag = tf.math.divide_no_nan(tf.reduce_sum(centered[:, 1:] * centered[:, :-1], axis=[0, 1]),
                                denominator)
    bias = tf.constant(1. / batches, tf.float64)
    statistic = tf.reduce_max(tf.abs(lag + bias))
    shuffled = tf.gather(centered, permutations, axis=1)  # [chain, permutation, batch, function]
    permuted_lag = tf.math.divide_no_nan(
        tf.reduce_sum(shuffled[:, :, 1:] * shuffled[:, :, :-1], axis=[0, 2]), denominator)
    permuted_statistic = tf.reduce_max(tf.abs(permuted_lag + bias), axis=1)
    tolerance = tf.constant(1e-12, tf.float64) * (1. + tf.abs(statistic))
    exceedances = tf.reduce_sum(tf.cast(permuted_statistic >= statistic-tolerance, tf.int32))
    p_value = (1. + tf.cast(exceedances, tf.float64)) / (1. + permutations.shape[0])
    valid = (tf.reduce_all(tf.math.is_finite(means)) & tf.reduce_all(denominator > 0.)
             & tf.reduce_all(tf.math.is_finite(denominator))
             & tf.reduce_all(tf.sort(permutations, axis=1) == tf.range(batches)))
    return dict(valid=valid, lag=lag, statistic=statistic, p_value=p_value,
                permuted_lag=permuted_lag, permuted_statistic=permuted_statistic,
                exceedances=exceedances)


@lru_cache(maxsize=4)
def batch_randomization(draws, function_count=15, permutation_count=1023):
    if draws not in (1000, 2000) or function_count < 1 or permutation_count < 1:
        raise ValueError("predeclared look and positive function/permutation counts required")

    @tf.function(input_signature=[tf.TensorSpec([4, draws, function_count], tf.float64),
                                 tf.TensorSpec([2], tf.int32)], jit_compile=True, autograph=False)
    def calculate(values, seed):
        means = tf.reduce_mean(tf.reshape(values, [4, 40, draws//40, function_count]), axis=2)
        keys = tf.random.stateless_uniform([permutation_count, 40], seed, dtype=tf.float64)
        permutations = tf.argsort(keys, axis=1, stable=True)
        result = permutation_statistics(means, permutations)
        sorted_keys = tf.sort(keys, axis=1)
        # Conditional on no ties, iid random keys induce uniform permutations.
        # Reject a discrete RNG tie instead of relying on its index tie-break.
        result['valid'] &= tf.reduce_all(sorted_keys[:, 1:] != sorted_keys[:, :-1])
        return {key: result[key] for key in ("valid", "lag", "statistic", "p_value", "exceedances")}
    return calculate
