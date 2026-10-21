"""Optional six-look comparison; extends the archived MCSE equations to six looks.

The old reference-building module remains byte-identical. This separate policy
changes the supported counts and critical value. The retained estimator body is
identical to the archived version; a parity regression checks the common counts.
"""
from functools import lru_cache

import tensorflow as tf
import tensorflow_probability as tfp
from bayesfilter.testing import surrogate_hmc_m13_posterior_assessment as baseline
from bayesfilter.testing.surrogate_hmc_m13_posterior_assessment import (
    F64, METHOD, LOOKS, MARGINS, draw_bilinear,
    finite_report, rank_normalized_split_rhat, rank_normalized_bulk_tail_ess,
)

EXTENDED_LOOKS = (1000, 2000, 4000, 6000, 8000, 10000)
EXTENDED_METHOD = "m13_ten_functions_guarded_lugsail_six_looks_v2"


def assessment_scope(looks):
    looks = tuple(looks)
    if any(type(n) is not int for n in looks):
        raise ValueError("integer prospective assessment looks required")
    if looks == LOOKS:
        return METHOD
    if looks == EXTENDED_LOOKS:
        return EXTENDED_METHOD
    raise ValueError("unsupported prospective M13 assessment schedule")

def critical_value(df=36, *, looks=LOOKS):
    assessment_scope(looks)
    # Split family alpha .05 equally between chain and reference uncertainty.
    # Two-sided, ten functions, explicitly bound prospective looks. Reference is frozen.
    return tfp.distributions.StudentT(tf.constant(float(df), F64), tf.constant(0., F64),
        tf.constant(1., F64)).quantile(tf.constant(1-.025/(2*10*len(looks)), F64))


@lru_cache(maxsize=6)
def estimates_graph(count):
    if count not in EXTENDED_LOOKS:
        raise ValueError("M13 assessment requires its predeclared looks")

    @tf.function(input_signature=[tf.TensorSpec([None, 4, count, 2], F64),
        tf.TensorSpec([2], F64), tf.TensorSpec([2], F64), tf.TensorSpec([2, 3], F64)],
        jit_compile=True, autograph=False)
    def calculate(points, reference_mean, reference_sd, cuts):
        z = (points-reference_mean)/reference_sd
        mean = tf.reduce_mean(z, [1, 2])
        centered = z-mean[:, None, None, :]
        sd = tf.sqrt(tf.reduce_mean(centered*centered, [1, 2]))
        indicators = tf.reshape(tf.cast(points[..., None] <= cuts, F64),
            [tf.shape(points)[0], 4, count, 6])
        estimates = tf.concat([mean, sd, tf.reduce_mean(indicators, [1, 2])], -1)
        # SD influence is ((z-Ez)^2-variance)/(2 SD). Removing its constant
        # does not change any batch or between-chain variance estimate.
        influence = tf.concat([z, centered*centered/(2*sd[:, None, None, :]), indicators], -1)
        variances = []
        for batches in (10, 20, 40):
            batch_means = tf.reduce_mean(tf.reshape(influence,
                [tf.shape(points)[0], 4, batches, count//batches, 10]), 3)
            residual = batch_means-tf.reduce_mean(batch_means, 2, keepdims=True)
            # Four independent chain means, each estimated from b batches.
            variances.append(tf.reduce_sum(residual*residual, [1, 2])/(16*batches*(batches-1)))
        chain_means = tf.reduce_mean(influence, 2)
        between = tf.reduce_sum((chain_means-tf.reduce_mean(chain_means, 1, keepdims=True))**2, 1)/12
        protected = tf.stack([*variances, 2*variances[0]-variances[1],
                              2*variances[1]-variances[2], between], 0)
        mcse = tf.sqrt(tf.reduce_max(protected, 0))
        return dict(estimates=estimates, mcse=mcse, batch_variances=tf.stack(variances, 0),
            between_chain_variance=between,
            finite=tf.reduce_all(tf.math.is_finite(points), [1, 2, 3]) &
                   tf.reduce_all(tf.math.is_finite(estimates) & tf.math.is_finite(mcse) & (mcse > 0), -1))
    return calculate


def assess(points, reference, *, calibration, looks=LOOKS):
    """points are [4,draws,2]; only a scoped coverage check can admit intervals."""
    method = assessment_scope(looks)
    if tuple(looks) == LOOKS:
        return baseline.assess(points, reference, calibration=calibration)
    if (calibration.get("method") != method or calibration.get("qualified") is not True
            or calibration.get("looks") != list(looks)
            or calibration.get("reference_sha256") != reference.get("reference_sha256")):
        raise ValueError("missing or mismatched M13 uncertainty qualification")
    count = int(points.shape[1])
    if count not in looks:
        raise ValueError("M13 assessment requires a prospective scheduled look")
    if points.shape[0] != 4 or points.shape[2] != 2:
        raise ValueError("four chains and two parameters required")
    tf.debugging.assert_all_finite(points, "nonfinite retained point")
    mean, sd, cuts = [tf.constant(reference[k], F64) for k in ("mean", "sd", "cuts")]
    calculated = estimates_graph(count)(points[None], mean, sd, cuts)
    expectation = tf.constant([0., 0., 1., 1.]+[.25, .5, .75]*2, F64)
    difference = calculated["estimates"][0]-expectation
    mcse = calculated["mcse"][0]
    radius = critical_value(looks=looks)*mcse+tf.constant(reference["empirical_allowance"], F64)
    # These tie-aware diagnostics use the existing explicit non-XLA exception.
    rhat = rank_normalized_split_rhat(points)["maximum"]
    ess = rank_normalized_bulk_tail_ess(points)
    mixing = bool(tf.reduce_all(rhat <= 1.01) & tf.reduce_all(ess["bulk"] >= 400) &
                  tf.reduce_all(ess["tail"] >= 400) & calculated["finite"][0])
    margins = tf.constant(MARGINS, F64)
    equivalent = tf.abs(difference)+radius <= margins
    discrepancy = tf.abs(difference)-radius > margins
    status = ("EQUIVALENT" if mixing and bool(tf.reduce_all(equivalent)) else
              "DISCREPANCY" if mixing and bool(tf.reduce_any(discrepancy)) else "INCONCLUSIVE")
    return finite_report(dict(status=status, method=method, draws_per_chain=count,
        parameter_mixing=mixing, rhat=rhat.numpy().tolist(), bulk_ess=ess["bulk"].numpy().tolist(),
        tail_ess=ess["tail"].numpy().tolist(), difference=difference.numpy().tolist(),
        mcse=mcse.numpy().tolist(), lower=(difference-radius).numpy().tolist(),
        upper=(difference+radius).numpy().tolist(), margins=list(MARGINS),
        practical_agreement=bool(tf.reduce_all(tf.abs(difference) <= margins)),
        family_alpha=.05, comparison_families=1, functions=10, planned_looks=list(looks),
        reference_allowance_is_rigorous_bound=False, ranking_supported=False,
        assumption="stationary-chain CLT and approximately independent long batches; scope coverage tested"))
