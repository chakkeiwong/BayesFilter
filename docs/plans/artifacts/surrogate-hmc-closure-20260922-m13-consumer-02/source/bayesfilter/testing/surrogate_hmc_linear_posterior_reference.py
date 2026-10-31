"""Independent FP64 posterior quadrature for two-parameter linear references.

This is an explicit CPU reference exception, never a DCTR runtime or tuning
implementation. Refinement allowances are empirical and are not error bounds.
"""
import tensorflow as tf

from bayesfilter.testing.surrogate_hmc_linear_reference import make_linear_reference


FUNCTION_NAMES = (
    "standardized_a", "standardized_b", "centered_a_second_moment",
    "centered_b_second_moment", "standardized_cross_moment",
    "a_cdf_q10", "a_cdf_q50", "a_cdf_q90", "b_cdf_q10", "b_cdf_q50", "b_cdf_q90",
    "predictive_y21_1_positive", "predictive_y21_2_positive",
)
MARGINS = (.1,) * 5 + (.02,) * 6 + (.03,) * 2


def trapezoid_weights(knots):
    widths = knots[1:] - knots[:-1]
    return .5 * tf.concat([widths[:1], widths[1:] + widths[:-1], widths[-1:]], 0)


def marginal_cdf(knots, density, cuts):
    """Integrate a piecewise-linear marginal, including the fractional cell."""
    widths = knots[1:] - knots[:-1]
    cumulative = tf.concat([tf.zeros([1], tf.float64), tf.cumsum(
        .5 * (density[1:] + density[:-1]) * widths)], 0)
    index = tf.clip_by_value(tf.searchsorted(knots, cuts, side="right") - 1, 0, tf.size(knots) - 2)
    offset = tf.minimum(tf.maximum(cuts - tf.gather(knots, index), 0.), tf.gather(widths, index))
    left = tf.gather(density, index)
    slope = (tf.gather(density, index + 1) - left) / tf.gather(widths, index)
    return tf.gather(cumulative, index) + left * offset + .5 * slope * offset**2


def grid_summary(first, second, density):
    weights = trapezoid_weights(first)[:, None] * trapezoid_weights(second)[None, :]
    mass = density * weights
    mass /= tf.reduce_sum(mass)
    first_mean = tf.reduce_sum(mass * first[:, None])
    second_mean = tf.reduce_sum(mass * second[None, :])
    means = tf.stack([first_mean, second_mean])
    variances = tf.stack([tf.reduce_sum(mass * (first[:, None] - means[0])**2),
                          tf.reduce_sum(mass * (second[None, :] - means[1])**2)])
    return means, tf.sqrt(variances)


def marginal_quantiles(knots, density):
    # Bisection in the actual piecewise-linear CDF; no grid-node indicator bias.
    probability = tf.constant([.1, .5, .9], tf.float64)
    low = tf.fill([3], knots[0])
    high = tf.fill([3], knots[-1])
    def step(index, low, high):
        mid = .5 * (low + high)
        below = marginal_cdf(knots, density, mid) < probability
        return index + 1, tf.where(below, mid, low), tf.where(below, high, mid)
    _, low, high = tf.while_loop(lambda index, *_: index < 50, step,
                                  (tf.constant(0), low, high), maximum_iterations=50)
    return .5 * (low + high)


def grid_expectations(first, second, density, predictive, means, scales, cuts):
    first_weights, second_weights = trapezoid_weights(first), trapezoid_weights(second)
    mass = density * first_weights[:, None] * second_weights[None, :]
    mass /= tf.reduce_sum(mass)
    a = (first[:, None] - means[0]) / scales[0]
    b = (second[None, :] - means[1]) / scales[1]
    moments = tf.stack([tf.reduce_sum(mass * a), tf.reduce_sum(mass * b),
                        tf.reduce_sum(mass * a**2), tf.reduce_sum(mass * b**2),
                        tf.reduce_sum(mass * a * b)])
    first_density = tf.linalg.matvec(density, second_weights)
    second_density = tf.linalg.matvec(density, first_weights, transpose_a=True)
    cdfs = tf.concat([marginal_cdf(first, first_density, cuts[0]),
                      marginal_cdf(second, second_density, cuts[1])], 0)
    probabilities = tf.reduce_sum(mass[:, :, None] * predictive, axis=[0, 1])
    return tf.concat([moments, cdfs, probabilities], 0)


GRID_SIGNATURE = [tf.TensorSpec([None], tf.float64), tf.TensorSpec([None], tf.float64),
                  tf.TensorSpec([None, None], tf.float64)]
summarize_grid = tf.function(grid_summary, input_signature=GRID_SIGNATURE,
                             jit_compile=True, autograph=False)
quantile_graph = tf.function(marginal_quantiles, input_signature=[
    tf.TensorSpec([None], tf.float64), tf.TensorSpec([None], tf.float64)],
    jit_compile=True, autograph=False)
expectation_graph = tf.function(grid_expectations, input_signature=GRID_SIGNATURE + [
    tf.TensorSpec([None, None, 2], tf.float64), tf.TensorSpec([2], tf.float64),
    tf.TensorSpec([2], tf.float64), tf.TensorSpec([2, 3], tf.float64)],
    jit_compile=True, autograph=False)


def prepare_reference(observations):
    reference = make_linear_reference("LG2")
    obs = tf.cast(tf.convert_to_tensor(observations, tf.float32), tf.float64)
    tf.debugging.assert_equal(tf.shape(obs), [20, 2])
    tf.debugging.assert_all_finite(obs, "observations")
    grids = []
    for intervals in (32, 64, 128, 256):
        first = tf.linspace(tf.constant(.5, tf.float64), tf.constant(1.2, tf.float64), intervals + 1)
        second = tf.linspace(tf.constant(-1., tf.float64), tf.constant(1., tf.float64), intervals + 1)
        mesh = tf.meshgrid(first, second, indexing="ij")
        points = tf.reshape(tf.stack(mesh, -1), [-1, 2])
        values, probabilities = [], []
        # Fixed 1024-row numerical batches bound XLA shapes and memory. Each
        # call is batch native; this loop shards arrays, never individual rows.
        for offset in range(0, int(points.shape[0]), 1024):
            count = min(1024, int(points.shape[0]) - offset)
            batch = points[offset:offset + count]
            if count < 1024:
                batch = tf.concat([batch, tf.repeat(batch[:1], 1024-count, axis=0)], 0)
            value, predictive = reference.physical_statistics(batch, obs)
            tf.debugging.assert_all_finite(value, "grid likelihood")
            tf.debugging.assert_all_finite(predictive, "grid prediction")
            values.append(value[:count])
            probabilities.append(predictive[:count])
        likelihood = tf.reshape(tf.concat(values, 0), [intervals+1, intervals+1])
        predictive = tf.reshape(tf.concat(probabilities, 0), [intervals+1, intervals+1, 2])
        unnormalized = tf.exp(likelihood - tf.reduce_max(likelihood))
        integral = tf.reduce_sum(unnormalized * trapezoid_weights(first)[:, None]
                                 * trapezoid_weights(second)[None, :])
        tf.debugging.assert_positive(integral, "normalization")
        tf.debugging.assert_all_finite(integral, "normalization")
        grids.append((first, second, unnormalized / integral, predictive))
    first, second, density, _ = grids[-1]
    means, scales = summarize_grid(first, second, density)
    tf.debugging.assert_positive(scales, "posterior scales")
    cuts = tf.stack([quantile_graph(first, tf.linalg.matvec(density, trapezoid_weights(second))),
                     quantile_graph(second, tf.linalg.matvec(density, trapezoid_weights(first), transpose_a=True))])
    expectations = tf.stack([expectation_graph(*grid, means, scales, cuts) for grid in grids])
    changes = tf.abs(expectations[1:] - expectations[:-1])
    allowance = 2. * tf.reduce_max(changes[-2:], axis=0)
    precision = allowance <= tf.constant(MARGINS, tf.float64) / 10.
    refinement = changes[-1] <= .5 * changes[-2] + 1e-5
    qualified = bool(tf.reduce_all(precision & refinement))
    report = dict(model_id="LG2", qualified=qualified, function_names=FUNCTION_NAMES,
                  margins=MARGINS, reference_expectations=expectations[-1].numpy().tolist(),
                  posterior_mean=means.numpy().tolist(), posterior_sd=scales.numpy().tolist(),
                  marginal_quantiles=cuts.numpy().tolist(),
                  empirical_refinement_allowance=allowance.numpy().tolist(),
                  allowance_is_rigorous_bound=False, intervals_per_axis=[32,64,128,256],
                  grid_expectations=expectations.numpy().tolist(),
                  precision_passed=precision.numpy().tolist(), refinement_passed=refinement.numpy().tolist(),
                  observation_precision="FP32 rounded once then promoted exactly to FP64",
                  posterior_comparisons="NOT_RUN", heuristic_dominance="NOT_RUN")
    tensors = dict(first_axis=first, second_axis=second, density=density,
                   predictive=grids[-1][3], means=means, scales=scales, cuts=cuts)
    return report, tensors
