"""Post-run TF diagnostics for predeclared surrogate-HMC equivalence checks.

Batch means estimate serial-dependence-aware uncertainty. A result is conditional
on the stationary-chain CLT; sensitivity and mixing screens can veto inference.
"""
from functools import lru_cache
import math

import tensorflow as tf
import tensorflow_probability as tfp

from .hmc_posterior_diagnostics import (
    rank_normalized_split_rhat, rank_normalized_bulk_tail_ess)


def finite_report(value):
    if isinstance(value, dict):
        return {k:finite_report(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_report(v) for v in value]
    return None if isinstance(value, float) and not math.isfinite(value) else value


@lru_cache(maxsize=4)
def _diagnostics(draws, parameter_dimension=1, function_count=6):
    if draws not in (1000, 2000):
        raise ValueError("unplanned inferential look")
    # Tied-rank segments are data-dependent and use a non-XLA diagnostic graph.
    @tf.function(input_signature=[tf.TensorSpec([4, draws, parameter_dimension], tf.float64),
                                 tf.TensorSpec([4, draws, function_count], tf.float64)],
                 jit_compile=False, autograph=False)
    def calculate(parameters, functions):
        return (rank_normalized_split_rhat(parameters), rank_normalized_bulk_tail_ess(parameters),
                rank_normalized_split_rhat(functions))
    return calculate


@lru_cache(maxsize=4)
def _batch_statistics(draws, function_count=6):
    if draws not in (1000, 2000):
        raise ValueError("unplanned inferential look")
    @tf.function(input_signature=[tf.TensorSpec([4, draws, function_count], tf.float64)],
                 jit_compile=True, autograph=False)
    def calculate(values):
        mcse, lag_one = [], []
        for size in (draws//40, draws//20, draws//10):
            batches = draws//size
            means = tf.reduce_mean(tf.reshape(values, [4, batches, size, function_count]), axis=2)
            centered = means - tf.reduce_mean(means, axis=1, keepdims=True)
            per_chain_variance = tf.reduce_sum(tf.square(centered), axis=1)/(batches-1)
            mcse.append(tf.sqrt(tf.reduce_mean(per_chain_variance, axis=0)/(4*batches)))
            numerator = tf.reduce_sum(centered[:, 1:]*centered[:, :-1], axis=[0, 1])
            denominator = tf.reduce_sum(tf.square(centered), axis=[0, 1])
            lag_one.append(numerator/denominator)
        return tf.stack(mcse), tf.stack(lag_one)
    return calculate


def assess_equivalence(parameters, functions, reference, allowance, margins, *, comparison_families=1,
                       parameter_dimension=1, function_count=6, analytic_reference=False):
    """Chain-major arrays; dimensions, comparison count and two looks are predeclared."""
    draws = parameters.shape[1]
    if comparison_families not in (1, 2, 4):
        raise ValueError("unplanned comparison family count")
    if any(type(value) is not int or value <= 0 for value in (parameter_dimension, function_count)):
        raise ValueError("positive static parameter and function dimensions required")
    if functions.shape != (4, draws, function_count) or parameters.shape != (4, draws, parameter_dimension):
        raise ValueError("expected four chains and the declared parameter/function dimensions")
    if any(tensor.shape != (function_count,) for tensor in (reference, allowance, margins)):
        raise ValueError("reference, allowance and margins must match the declared functions")
    for tensor in (parameters, functions, reference, allowance, margins):
        tf.debugging.assert_all_finite(tensor, "nonfinite equivalence input")
    tf.debugging.assert_non_negative(allowance)
    tf.debugging.assert_positive(margins)
    rhat, ess, function_rhat = _diagnostics(draws, parameter_dimension, function_count)(parameters, functions)
    mcse, lag = _batch_statistics(draws, function_count)(functions)
    conservative_mcse = tf.reduce_max(mcse, axis=0)
    sensitivity = conservative_mcse/tf.reduce_min(mcse, axis=0)
    # At the largest batch length there are 10 batches per chain. Use the
    # smallest pooled within-chain degrees of freedom across the sensitivity grid.
    degrees = 4*(10-1)
    critical = tfp.distributions.StudentT(tf.constant(degrees, tf.float64),
        tf.constant(0., tf.float64), tf.constant(1., tf.float64)).quantile(
            tf.constant(1-.05/(2*function_count*2*comparison_families), tf.float64))
    difference = tf.reduce_mean(functions, axis=[0, 1]) - reference
    radius = critical*conservative_mcse + allowance
    lower, upper = difference-radius, difference+radius
    mixing = (tf.reduce_all(tf.math.is_finite(rhat["maximum"]) & (rhat["maximum"] <= 1.01))
              & tf.reduce_all(tf.math.is_finite(ess["bulk"]) & (ess["bulk"] >= 400))
              & tf.reduce_all(tf.math.is_finite(ess["tail"]) & (ess["tail"] >= 400)))
    qualified = (tf.math.is_finite(sensitivity) & (sensitivity <= 2.)
                 & tf.math.is_finite(lag[-1]) & (tf.abs(lag[-1]) <= .3)
                 & tf.math.is_finite(function_rhat["maximum"])
                 & (function_rhat["maximum"] <= 1.01) & (conservative_mcse > 0))
    equivalent = (lower > -margins) & (upper < margins)
    discrepant = (lower > margins) | (upper < -margins)
    valid = bool(mixing & tf.reduce_all(qualified))
    status = ("EQUIVALENT" if valid and bool(tf.reduce_all(equivalent)) else
              "DISCREPANCY" if valid and bool(tf.reduce_any(discrepant)) else "INCONCLUSIVE")
    return finite_report(dict(status=status, diagnostic_valid=valid, draws_per_chain=draws,
        rhat={k:v.numpy().tolist() for k,v in rhat.items()},
        ess={k:v.numpy().tolist() for k,v in ess.items()},
        function_rhat=function_rhat["maximum"].numpy().tolist(),
        function_mcse_qualified=qualified.numpy().tolist(),
        batch_lengths=[draws//40,draws//20,draws//10], batch_mcse=mcse.numpy().tolist(),
        batch_lag1=lag.numpy().tolist(), mcse_sensitivity=sensitivity.numpy().tolist(),
        difference=difference.numpy().tolist(), lower=lower.numpy().tolist(), upper=upper.numpy().tolist(),
        equivalent=equivalent.numpy().tolist(), discrepant=discrepant.numpy().tolist(),
        reference_allowance=allowance.numpy().tolist(), reference_bound_is_rigorous=False,
        reference_type="analytic" if analytic_reference else "empirical_refinement",
        family_alpha=.05, family_functions=function_count, comparison_families=comparison_families,
        planned_looks=2, degrees_of_freedom=degrees,
        uncertainty_assumption="stationary-chain CLT; batch means approximately independent"))
