"""Prospective AR2 diagnostics with a guarded over-lugsail MCSE candidate.

Vats--Flegal (2021), eqs. (6)--(7), permits r=2,c=2/3 and r=3,c=1/2.
This candidate takes the maximum of ordinary, r=3, and r=2 lugsail
variances. The maximum is a local conservative extension, not a coverage
theorem. Qualification requires fresh AR2 calibration before retained-chain
use.
"""
from functools import lru_cache

import tensorflow as tf

from bayesfilter.inference.surrogate_hmc_equivalence import finite_report
from bayesfilter.testing.surrogate_hmc_mcse_randomization import diagnostic_batch_statistics
from bayesfilter.testing.surrogate_hmc_scoped_assessment import (
    critical_value, diagnostic_calculator as baseline_calculator,
)

METHOD = "ar2_guarded_over_lugsail_r2_c_two_thirds_plus_r3_v2"


@lru_cache(maxsize=4)
def lugsail_statistics(count, function_count=13):
    ordinary = diagnostic_batch_statistics(count, function_count)

    @tf.function(input_signature=[tf.TensorSpec([4, count, function_count], tf.float64)],
                 jit_compile=True, autograph=False)
    def calculate(values):
        estimates, _ = ordinary(values)
        short_variances_r3, short_variances_r2 = [], []
        center = tf.reduce_mean(values, axis=1, keepdims=True)
        for batches in (40, 20, 10):
            length = (count//batches)//3
            short_count = count//length
            means = tf.reduce_mean(tf.reshape(values[:, :short_count*length],
                                   [4, short_count, length, function_count]), axis=2)
            # Author mbmC centers even truncated batches on the full chain mean.
            asymptotic_variance = length*tf.reduce_sum(tf.square(means-center), axis=1)/(short_count-1)
            short_variances_r3.append(tf.reduce_mean(asymptotic_variance, axis=0)/(4*count))
            short2 = (count//batches)//2
            count2 = count//short2
            means2 = tf.reduce_mean(tf.reshape(values[:, :count2*short2],
                                    [4, count2, short2, function_count]), axis=2)
            variance2 = short2*tf.reduce_sum(tf.square(means2-center), axis=1)/(count2-1)
            short_variances_r2.append(tf.reduce_mean(variance2, axis=0)/(4*count))
        ordinary_variance = tf.square(estimates)
        raw_variance = 2*ordinary_variance-tf.stack(short_variances_r3)
        raw_variance_r2 = 3*ordinary_variance-2*tf.stack(short_variances_r2)
        guarded_variance = tf.maximum(ordinary_variance, tf.maximum(raw_variance, raw_variance_r2))
        guarded = tf.sqrt(tf.maximum(guarded_variance, 0.))
        mcse = tf.reduce_max(guarded, axis=0)
        return dict(mcse=mcse, ordinary_mcse=tf.reduce_max(estimates, axis=0),
                    lugsail_raw_variance=raw_variance, over_lugsail_r2_raw_variance=raw_variance_r2,
                    guarded_mcse_estimates=guarded,
                    mcse_sensitivity=mcse/tf.reduce_min(guarded, axis=0),
                    lugsail_guard_active=(raw_variance < ordinary_variance) | (raw_variance_r2 < ordinary_variance))
    return calculate


@lru_cache(maxsize=4)
def diagnostic_calculator(count):
    baseline, repaired = baseline_calculator(count), lugsail_statistics(count)

    @tf.function(input_signature=[tf.TensorSpec([4, count, 2], tf.float64),
        tf.TensorSpec([4, count, 13], tf.float64), tf.TensorSpec([2], tf.int32)],
        jit_compile=False, autograph=False)
    def calculate(points, values, seed):
        # The unchanged tie-aware rank diagnostics remain a non-XLA exception.
        result, corrected = baseline(points, values, seed), repaired(values)
        result["ordinary_mcse_sensitivity"] = result["mcse_sensitivity"]
        valid = (tf.reduce_all(tf.math.is_finite(corrected["mcse"]))
                 & tf.reduce_all(corrected["mcse"] > 0.)
                 & tf.reduce_all(corrected["mcse_sensitivity"] <= 2.))
        result["qualified"] &= valid
        result["variance_valid"] &= valid
        result.update(corrected)
        return result
    return calculate


def assess(points, values, reference, allowance, margins, *, seed):
    """Apply the frozen two-look decision using the calibration's exact callable."""
    for value in (points, values, reference, allowance, margins):
        tf.debugging.assert_all_finite(value, "nonfinite posterior assessment input")
    tf.debugging.assert_non_negative(allowance)
    tf.debugging.assert_positive(margins)
    result = diagnostic_calculator(int(points.shape[1]))(points, values, tf.constant(seed, tf.int32))
    difference = result["means"]-reference
    radius = critical_value()*result["mcse"]+allowance
    equivalent = tf.abs(difference)+radius <= margins
    discrepant = tf.abs(difference)-radius > margins
    qualified = bool(result["qualified"])
    status = ("EQUIVALENT" if qualified and bool(tf.reduce_all(equivalent)) else
              "DISCREPANCY" if qualified and bool(tf.reduce_any(discrepant)) else "INCONCLUSIVE")
    row = {k: v.numpy().tolist() for k, v in result.items()}
    row.update(status=status, method=METHOD, difference=difference.numpy().tolist(),
        lower=(difference-radius).numpy().tolist(), upper=(difference+radius).numpy().tolist(),
        equivalent=equivalent.numpy().tolist(), discrepant=discrepant.numpy().tolist(),
        reference_allowance=allowance.numpy().tolist(), reference_bound_is_rigorous=False,
        family_alpha=.05, family_functions=13, comparison_families=1, planned_looks=2,
        degrees_of_freedom=36, ranking="NOT_ESTABLISHED",
        practical_judgment=("OBSERVED_DIFFERENCES_WITHIN_PILOT_MARGINS"
            if bool(tf.reduce_all(tf.abs(difference) <= margins)) else "OBSERVED_DIFFERENCE_EXCEEDS_PILOT_MARGIN"),
        practical_judgment_is_formal_confirmation=False, owner_acceptance="NOT_ASSIGNED",
        uncertainty_assumption="stationary-chain CLT; guarded lugsail variance; empirical AR2 scope calibration")
    return finite_report(row)
