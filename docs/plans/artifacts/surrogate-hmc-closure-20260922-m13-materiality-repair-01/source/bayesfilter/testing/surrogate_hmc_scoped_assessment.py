"""Prospective surrogate-HMC assessment shared by calibration and retained tests.

This diagnostic does not change HMC or reclassify archived chains. Its use for
a new scope requires calibration on that scope's reference-law functions.
"""
from functools import lru_cache

import tensorflow as tf
import tensorflow_probability as tfp

from bayesfilter.inference.hmc_posterior_diagnostics import rank_normalized_split_rhat, rank_normalized_bulk_tail_ess
from bayesfilter.inference.surrogate_hmc_equivalence import finite_report
from bayesfilter.testing.surrogate_hmc_mcse_randomization import (
    batch_randomization, binary_function_rhat, diagnostic_batch_statistics,
)


def critical_value():
    # One primary B/R family, thirteen functions, two predeclared looks.
    return tfp.distributions.StudentT(tf.constant(36., tf.float64), tf.constant(0., tf.float64),
        tf.constant(1., tf.float64)).quantile(tf.constant(1-.05/(2*13*2), tf.float64))


@lru_cache(maxsize=8)
def diagnostic_calculator(count):
    batches, randomized = diagnostic_batch_statistics(count, 13), batch_randomization(count, 13)

    @tf.function(input_signature=[tf.TensorSpec([4, count, 2], tf.float64),
        tf.TensorSpec([4, count, 13], tf.float64), tf.TensorSpec([2], tf.int32)],
        jit_compile=False, autograph=False)
    def calculate(points, values, seed):
        # Tie-aware ranking remains the repository's explicit non-XLA
        # diagnostic exception. Batch means and permutations execute in XLA.
        pr = rank_normalized_split_rhat(points)
        ess = rank_normalized_bulk_tail_ess(points)
        fr = rank_normalized_split_rhat(values)
        effective, binary = binary_function_rhat(values, fr)
        estimates, lag = batches(values)
        mcse = tf.reduce_max(estimates, 0)
        sensitivity = mcse/tf.reduce_min(estimates, 0)
        randomization = randomized(values, seed)
        finite = tf.reduce_all(tf.math.is_finite(points)) & tf.reduce_all(tf.math.is_finite(values))
        mixing = finite & tf.reduce_all(pr["maximum"] <= 1.01) & tf.reduce_all(ess["bulk"] >= 400.) & tf.reduce_all(ess["tail"] >= 400.)
        variance_valid = tf.reduce_all(tf.math.is_finite(mcse) & (mcse > 0.) & (sensitivity <= 2.))
        qualified = mixing & variance_valid & tf.reduce_all(effective <= 1.01) & randomization["valid"] & (randomization["p_value"] > .01)
        return dict(qualified=qualified, finite=finite, parameter_mixing=mixing, variance_valid=variance_valid,
            means=tf.reduce_mean(values, [0, 1]), mcse=mcse, mcse_sensitivity=sensitivity,
            parameter_rhat=pr["maximum"], bulk_ess=ess["bulk"], tail_ess=ess["tail"],
            function_rhat=effective, raw_folded_function_rhat=fr["folded"], binary_fold_uninformative=binary,
            batch_lag=lag, randomization_p=randomization["p_value"])
    return calculate


def assess(points, values, reference, allowance, margins, *, seed):
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
    descriptive_within = bool(tf.reduce_all(tf.abs(difference) <= margins))
    row = {k: v.numpy().tolist() for k, v in result.items()}
    row.update(status=status, difference=difference.numpy().tolist(), lower=(difference-radius).numpy().tolist(),
        upper=(difference+radius).numpy().tolist(), equivalent=equivalent.numpy().tolist(),
        discrepant=discrepant.numpy().tolist(), reference_allowance=allowance.numpy().tolist(),
        reference_bound_is_rigorous=False, family_alpha=.05, family_functions=13, comparison_families=1,
        planned_looks=2, degrees_of_freedom=36, ranking="NOT_ESTABLISHED",
        practical_judgment="OBSERVED_DIFFERENCES_WITHIN_PILOT_MARGINS" if descriptive_within else "OBSERVED_DIFFERENCE_EXCEEDS_PILOT_MARGIN",
        practical_judgment_is_formal_confirmation=False, owner_acceptance="NOT_ASSIGNED",
        uncertainty_assumption="stationary-chain CLT and approximately exchangeable batch vectors; scope-calibrated diagnostic")
    return finite_report(row)
