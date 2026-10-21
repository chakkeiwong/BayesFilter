"""Prospective diagnostic candidate; no admission or default-policy authority.

The randomization null is exchangeable vectors of batch means. This is exact
for independent observations and an empirical hypothesis for MCMC batches.
"""
from functools import lru_cache

import tensorflow as tf


def binary_function_rhat(values, components):
    """Diagnostic-only policy for the uninformative fold of balanced indicators.

    Keep the raw components for reporting. This does not assign a value to 0/0
    or repair constant/stuck/nonbinary series. Call inside a fixed-shape graph.
    """
    binary = tf.reduce_all((values == 0.) | (values == 1.), axis=[0, 1])
    balanced = tf.reduce_sum(values, axis=[0, 1]) == tf.cast(
        tf.shape(values)[0] * tf.shape(values)[1], values.dtype) / 2.
    uninformative = binary & balanced & tf.math.is_finite(components['bulk'])
    maximum = tf.where(uninformative, tf.maximum(tf.ones_like(components['bulk']), components['bulk']),
                       components['maximum'])
    # A constant rank transform can acquire tiny roundoff variance and appear
    # finite. Raw finite/nonconstant data are required independently of that.
    informative = (tf.reduce_all(tf.math.is_finite(values), axis=[0, 1])
                   & (tf.reduce_max(values, axis=[0, 1]) > tf.reduce_min(values, axis=[0, 1])))
    maximum = tf.where(informative, maximum, tf.fill(tf.shape(maximum), tf.constant(float('inf'), tf.float64)))
    return maximum, uninformative


@lru_cache(maxsize=4)
def diagnostic_batch_statistics(draws, function_count=15):
    """Longer diagnostic scopes only; parity-checked against the runtime formula.

    This extension cannot authorize an additional inferential look in HMC.
    The production helper deliberately accepts only its frozen two looks.
    """
    if draws not in (1000, 2000, 4000, 8000):
        raise ValueError('unplanned diagnostic sample size')

    @tf.function(input_signature=[tf.TensorSpec([4, draws, function_count], tf.float64)],
                 jit_compile=True, autograph=False)
    def calculate(values):
        estimates, correlations = [], []
        for batches in (40, 20, 10):
            means = tf.reduce_mean(tf.reshape(values, [4, batches, draws//batches, function_count]), axis=2)
            centered = means - tf.reduce_mean(means, axis=1, keepdims=True)
            variance = tf.reduce_sum(tf.square(centered), axis=1)/(batches-1)
            estimates.append(tf.sqrt(tf.reduce_mean(variance, axis=0)/(4*batches)))
            correlations.append(tf.reduce_sum(centered[:, 1:]*centered[:, :-1], axis=[0, 1])
                                / tf.reduce_sum(tf.square(centered), axis=[0, 1]))
        return tf.stack(estimates), tf.stack(correlations)
    return calculate


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
    if draws not in (1000, 2000, 4000, 8000) or function_count < 1 or permutation_count < 1:
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
