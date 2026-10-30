"""Diagnostic M13 posterior comparison; never changes a runtime HMC target.

Means and SDs use posterior-SD units; six CDFs use probability units. The
uncertainty method needs the separate M13 reference-law coverage check before
it can qualify a retained pilot. All numerics are TensorFlow/TFP.
"""
from functools import lru_cache

import tensorflow as tf
import tensorflow_probability as tfp

from bayesfilter.inference.hmc_posterior_diagnostics import (
    rank_normalized_split_rhat, rank_normalized_bulk_tail_ess,
)
from bayesfilter.inference.surrogate_hmc_equivalence import finite_report

F64 = tf.float64
METHOD = "m13_ten_functions_guarded_lugsail_two_looks_v1"
LOOKS = (1000, 2000)
MARGINS = (.1, .1, .1, .1, .05, .05, .05, .05, .05, .05)


def critical_value(df=36):
    # Split family alpha .05 equally between chain and reference uncertainty.
    # Two-sided, ten functions, two prospective looks. Reference is frozen.
    return tfp.distributions.StudentT(tf.constant(float(df), F64), tf.constant(0., F64),
        tf.constant(1., F64)).quantile(tf.constant(1-.025/(2*10*2), F64))


def piecewise_cdf(axis, density, cut):
    """Integrate a normalized piecewise linear 1D density exactly.

    density: [..., nodes]; cut: [..., cuts]. On each segment with width h,
    integral to displacement s is f_left*s+(f_right-f_left)*s*s/(2*h).
    """
    h = axis[1:]-axis[:-1]
    s = tf.clip_by_value(cut[..., :, None]-axis[:-1], tf.constant(0., F64), h)
    return tf.reduce_sum(density[..., None, :-1]*s +
        (density[..., None, 1:]-density[..., None, :-1])*s*s/(2*h), -1)


def linear_density_moments(axis, density):
    """Exact first and second moments of a normalized linear interpolant."""
    a, b = axis[:-1], axis[1:]
    h = b-a
    left, right = density[..., :-1], density[..., 1:]
    mean = tf.reduce_sum(h*((2*a+b)*left+(a+2*b)*right)/6, -1)
    second = tf.reduce_sum(h*((3*a*a+2*a*b+b*b)*left+
                              (a*a+2*a*b+3*b*b)*right)/12, -1)
    return mean, tf.sqrt(second-mean*mean)


def grid_density(axis, log_density):
    """Normalize a bilinear two-dimensional interpolant, retaining replicates.

    Inputs have [nodes], [nodes,nodes,replicates]. Returns pooled and replicate
    marginal densities and likelihood-normalizer influence weights. Finite-box
    interpolation is an independent diagnostic, not the full-support endpoint.
    """
    n = tf.shape(axis)[0]
    h = axis[1]-axis[0]
    w = tf.concat([tf.ones([1], F64)*.5, tf.ones([n-2], F64), tf.ones([1], F64)*.5], 0)*h
    offset = tf.reduce_max(log_density)
    density = tf.exp(log_density-offset)
    normals = tf.einsum("ijr,i,j->r", density, w, w)
    average_normal = tf.reduce_mean(normals)
    ratios = normals/average_normal
    pooled = tf.reduce_mean(density, -1)/average_normal
    # [replicate, parameter, node]
    marginal = tf.stack([tf.einsum("ijr,j->ri", density, w),
                         tf.einsum("ijr,i->rj", density, w)], 1)/normals[:, None, None]
    pooled_marginal = tf.reduce_mean(marginal*ratios[:, None, None], 0)
    return pooled, pooled_marginal, marginal, ratios


def reference_quantiles(axis, marginals):
    target = tf.constant([[.25, .5, .75], [.25, .5, .75]], F64)
    low, high = tf.ones([2, 3], F64)*axis[0], tf.ones([2, 3], F64)*axis[-1]
    def body(i, low, high):
        mid = (low+high)/2
        below = piecewise_cdf(axis, marginals, mid) < target
        return i+1, tf.where(below, mid, low), tf.where(below, high, mid)
    _, low, high = tf.while_loop(lambda i, *_: i < 48, body, (0, low, high))
    return (low+high)/2


def inverse_cdf_indices(cumulative, values):
    """Upper-bound search with O(draws + categories) live storage under XLA.

    Some XLA LowerBound/UpperBound implementations broadcast both arrays.
    A fixed-depth vector binary search makes the storage bound explicit.
    """
    size = cumulative.shape[0]
    if size is None or size < 1:
        raise ValueError("reference CDF size must be statically known")
    low = tf.zeros(tf.shape(values), tf.int32)
    high = tf.fill(tf.shape(values), size)
    def body(i, low, high):
        middle = (low+high)//2
        below = tf.gather(cumulative, tf.minimum(middle, size-1)) <= values
        active = low < high
        return (i+1, tf.where(active & below, middle+1, low),
                tf.where(active & ~below, middle, high))
    _, low, _ = tf.while_loop(lambda i, *_: i < size.bit_length(), body,
                              (tf.constant(0), low, high))
    return low


def draw_bilinear(axis, density, shape, seed):
    """Exact IID draws from the bilinear interpolant by triangular mixtures.

    Each cell is a four-component mixture of separable linear basis functions.
    Choose a corner with mass proportional to its density, then draw Beta(2,1)
    toward its upper coordinate or Beta(1,2) toward its lower coordinate. Equal
    cell areas cancel in the categorical weights on this uniform grid.
    """
    n = tf.shape(axis)[0]-1
    corners = tf.stack([density[:-1, :-1], density[:-1, 1:],
                        density[1:, :-1], density[1:, 1:]], -1)
    count = tf.reduce_prod(shape)
    seeds = tf.random.experimental.stateless_split(seed, 2)
    # Inverse-CDF sampling avoids an XLA categorical lowering that can
    # allocate [draws,categories] random variates on a large reference grid.
    cumulative = tf.cumsum(tf.reshape(corners, [-1]))
    cumulative = cumulative/cumulative[-1]
    choice = tf.random.stateless_uniform([count], seeds[0], dtype=F64)
    index = inverse_cdf_indices(cumulative, choice)
    corner = index % 4
    row, column = (index//4)//n, (index//4) % n
    upper = tf.stack([corner//2, corner % 2], -1) != 0
    uniform = tf.random.stateless_uniform([count, 2], seeds[1], dtype=F64)
    fraction = tf.where(upper, tf.sqrt(uniform), 1-tf.sqrt(uniform))
    lower = tf.stack([tf.gather(axis, row), tf.gather(axis, column)], -1)
    return tf.reshape(lower+(axis[1]-axis[0])*fraction, tf.concat([shape, [2]], 0))


@lru_cache(maxsize=2)
def estimates_graph(count):
    if count not in LOOKS:
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


def assess(points, reference, *, calibration):
    """points are [4,draws,2]; only a scoped coverage check can admit intervals."""
    if (calibration.get("method") != METHOD or calibration.get("qualified") is not True
            or calibration.get("looks") != list(LOOKS)
            or calibration.get("reference_sha256") != reference.get("reference_sha256")):
        raise ValueError("missing or mismatched M13 uncertainty qualification")
    count = int(points.shape[1])
    if points.shape[0] != 4 or points.shape[2] != 2:
        raise ValueError("four chains and two parameters required")
    tf.debugging.assert_all_finite(points, "nonfinite retained point")
    mean, sd, cuts = [tf.constant(reference[k], F64) for k in ("mean", "sd", "cuts")]
    calculated = estimates_graph(count)(points[None], mean, sd, cuts)
    expectation = tf.constant([0., 0., 1., 1.]+[.25, .5, .75]*2, F64)
    difference = calculated["estimates"][0]-expectation
    mcse = calculated["mcse"][0]
    radius = critical_value()*mcse+tf.constant(reference["empirical_allowance"], F64)
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
    return finite_report(dict(status=status, method=METHOD, draws_per_chain=count,
        parameter_mixing=mixing, rhat=rhat.numpy().tolist(), bulk_ess=ess["bulk"].numpy().tolist(),
        tail_ess=ess["tail"].numpy().tolist(), difference=difference.numpy().tolist(),
        mcse=mcse.numpy().tolist(), lower=(difference-radius).numpy().tolist(),
        upper=(difference+radius).numpy().tolist(), margins=list(MARGINS),
        practical_agreement=bool(tf.reduce_all(tf.abs(difference) <= margins)),
        family_alpha=.05, comparison_families=1, functions=10, planned_looks=list(LOOKS),
        reference_allowance_is_rigorous_bound=False, ranking_supported=False,
        assumption="stationary-chain CLT and approximately independent long batches; scope coverage tested"))
