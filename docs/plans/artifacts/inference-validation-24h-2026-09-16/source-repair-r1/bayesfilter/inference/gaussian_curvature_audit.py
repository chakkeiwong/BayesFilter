"""Audit local curvature across a frozen Gaussian position guide using TF.

For theta = center + factor @ z and U = -log p, the diagnostic estimates
G(z) = factor.T @ Hessian(U)(theta) @ factor. A matched Gaussian has G = I.
The interval [1/4, 4] permits local quadratic widths between 1/2 and 2 in
guide units. This is an interpretable engineering tolerance, not a universal
normality threshold or a guarantee of NeuTra/HMC performance. See the CIP
monograph, Chapter 20, eq:mass_hessian, and Neal (2011), MCMC Using Hamiltonian
Dynamics, for the curvature/preconditioning and harmonic-oscillator basis.

This is an opt-in, bounded eager diagnostic with TensorFlow float64 numerical
kernels and fixed-shape batched target calls. It performs no optimization or
sampling transitions. The caller owns compilation of the actual target and
GPU memory-growth setup. No outer-XLA compatibility claim is made here.
"""

import math
from collections.abc import Callable
from dataclasses import asdict, dataclass

import tensorflow as tf


@dataclass(frozen=True)
class GaussianCurvatureAuditConfig:
    """Freeze the guide-region design and numerical resolution before probing.

    Two central-difference step sizes expose truncation/cancellation problems.
    Their agreement is a numerical diagnostic, not a certified derivative
    error bound. The resolution tolerance never widens the curvature band.
    """

    probe_count: int = 64
    batch_size: int = 8
    seed: tuple[int, int] = (20260912, 4101)
    probe_radius: float = 1.0
    steps: tuple[float, float] = (0.001, 0.0005)
    minimum_curvature: float = 0.25
    maximum_curvature: float = 4.0
    resolution_atol: float = 1.0e-5
    resolution_rtol: float = 1.0e-3
    confidence_alpha: float = 0.05
    max_nonpass_probability: float = 0.05
    max_physical_rows: int = 40000

    def __post_init__(self):
        if min(self.probe_count, self.batch_size, self.max_physical_rows) < 1:
            raise ValueError("counts must be positive")
        if not 0 < self.steps[1] < self.steps[0] or not all(map(math.isfinite, self.steps)):
            raise ValueError("steps must be positive finite and decreasing")
        if not math.isfinite(self.probe_radius) or self.probe_radius <= 0:
            raise ValueError("probe radius must be positive and finite")
        if not 0 < self.minimum_curvature < self.maximum_curvature < math.inf:
            raise ValueError("curvature band must be positive finite and ordered")
        if any(not math.isfinite(value) or value < 0 for value in
               (self.resolution_atol, self.resolution_rtol)):
            raise ValueError("resolution tolerances must be finite and nonnegative")
        if not 0 < self.confidence_alpha < 1 or not 0 < self.max_nonpass_probability < 1:
            raise ValueError("probabilities must lie strictly between zero and one")


def binomial_nonpass_upper(nonpasses: int, total: int, alpha: float = 0.05) -> float:
    """One-sided Clopper--Pearson upper limit, conditional on a fixed design.

    Invert I_upper(k+1,n-k) = 1-alpha using bounded TF bisection. Counting
    numerically inconclusive probes as nonpasses prevents their exclusion from
    manufacturing a favorable rate. Independence applies to probe locations,
    not to the many finite-difference rows surrounding each location.
    """
    if total < 1 or not 0 <= nonpasses <= total or not 0 < alpha < 1:
        raise ValueError("invalid binomial count or probability")
    if nonpasses == total:
        return 1.0
    shape_first = tf.constant(nonpasses + 1, tf.float64)
    shape_second = tf.constant(total - nonpasses, tf.float64)

    def body(iteration, lower, upper):
        middle = (lower + upper) * 0.5
        below = tf.math.betainc(shape_first, shape_second, middle) < 1.0 - alpha
        return iteration + 1, tf.where(below, middle, lower), tf.where(below, upper, middle)

    _, _, upper = tf.while_loop(
        lambda iteration, lower, upper: iteration < 60, body,
        (tf.constant(0), tf.constant(0.0, tf.float64), tf.constant(1.0, tf.float64)),
    )
    return float(upper.numpy())


def compare_curvature_steps(coarse, fine, *, roundoff_margin=0.0, config=None):
    """Inspect the full spectrum with an empirical numerical-resolution margin.

    Every coordinate direction is represented. An average residual or a few
    random Rayleigh quotients could hide one stiff direction in a large model.
    We retain antisymmetry before symmetrization, since a nonconservative score
    must not silently become a plausible precision matrix.
    """
    config = GaussianCurvatureAuditConfig() if config is None else config
    coarse = tf.convert_to_tensor(coarse, dtype=tf.float64)
    fine = tf.convert_to_tensor(fine, dtype=tf.float64)
    tf.debugging.assert_shapes([(coarse, ("dimension", "dimension")),
                                (fine, ("dimension", "dimension"))])
    tf.debugging.assert_all_finite(coarse, "nonfinite coarse curvature")
    tf.debugging.assert_all_finite(fine, "nonfinite fine curvature")
    symmetric_coarse = 0.5 * (coarse + tf.transpose(coarse))
    symmetric_fine = 0.5 * (fine + tf.transpose(fine))
    eigenvalues = tf.linalg.eigvalsh(symmetric_fine)
    difference = tf.reduce_max(tf.abs(tf.linalg.eigvalsh(symmetric_fine - symmetric_coarse)))
    antisymmetry = tf.maximum(tf.linalg.norm(coarse - symmetric_coarse),
                             tf.linalg.norm(fine - symmetric_fine))
    spectral_scale = tf.maximum(tf.constant(1.0, tf.float64), tf.reduce_max(tf.abs(eigenvalues)))
    tolerance = config.resolution_atol + config.resolution_rtol * spectral_scale
    # Weyl's inequality motivates buffering eigenvalues by a matrix-error norm.
    # This empirical difference/symmetry/roundoff margin is not a proof that all
    # truncation error is bounded. Analytic-Hessian fixtures check that gap.
    margin = difference + antisymmetry + tf.cast(roundoff_margin, tf.float64)
    resolved = (difference <= tolerance) & (antisymmetry <= tolerance) & (margin <= tolerance)
    minimum = eigenvalues[0]
    maximum = eigenvalues[-1]
    inside = (minimum - margin >= config.minimum_curvature) & (maximum + margin <= config.maximum_curvature)
    outside = (minimum + margin < config.minimum_curvature) | (maximum - margin > config.maximum_curvature)
    return {
        "eigenvalues": eigenvalues,
        "minimum": minimum, "maximum": maximum,
        "step_difference": difference, "antisymmetry": antisymmetry,
        "resolution_margin": margin, "resolution_tolerance": tolerance,
        "resolved": resolved, "passed": resolved & inside,
        "outside_band": resolved & outside,
        "inconclusive": ~resolved | (~inside & ~outside),
    }


def audit_gaussian_curvature(
    batched_target: Callable, center, position_factor, *, config=None,
    record_batch: Callable | None = None,
):
    """Probe a frozen guide using a [B,D] -> (values,scores,eligible) callback.

    Scores must be raw-coordinate gradients of the returned log density, and
    eligibility must reject finite support sentinels. Every attempted row,
    including padding, must be eligible and finite. ``record_batch`` receives
    role, raw positions, values, scores and eligibility before validity checks,
    allowing a caller to retain the first failure. Programming errors propagate.

    In row notation score_z = score_theta @ factor. Central differences of
    minus this score give rows indexed by perturbation direction; transpose
    them to obtain Hessian columns. The affine log determinant is constant.
    No fitted center, guide scale or random point is changed after inspection.
    """
    if not tf.executing_eagerly():
        raise RuntimeError("call bounded audit orchestration outside tf.function")
    config = GaussianCurvatureAuditConfig() if config is None else config
    center = tf.convert_to_tensor(center, dtype=tf.float64)
    factor = tf.convert_to_tensor(position_factor, dtype=tf.float64)
    tf.debugging.assert_shapes([(center, ("dimension",)), (factor, ("dimension", "dimension"))])
    tf.debugging.assert_all_finite(center, "nonfinite center")
    tf.debugging.assert_all_finite(factor, "nonfinite position factor")
    tf.debugging.assert_equal(factor, tf.linalg.band_part(factor, -1, 0))
    tf.debugging.assert_positive(tf.linalg.diag_part(factor))
    dimension = int(center.shape[0])
    if dimension < 1:
        raise ValueError("dimension must be positive")

    def padded(count):
        return ((count + config.batch_size - 1) // config.batch_size) * config.batch_size

    planned = padded(config.probe_count) + config.probe_count * padded(4 * dimension)
    if planned > config.max_physical_rows:
        raise ValueError("curvature audit exceeds row budget before target calls")
    result = {
        "passed": False, "status": "incomplete", "config": asdict(config),
        "planned_physical_rows": planned, "physical_rows": 0, "logical_rows": 0,
        "target_calls": 0, "point_diagnostics": [], "nonpasses": None,
        "nonpass_probability_upper": None,
        "probability_measure": "uniform volume in frozen standardized ball, conditional on numerical classifier",
    }

    def evaluate(points, role):
        values_collected, scores_collected = [], []
        row_count = int(points.shape[0])
        for first in range(0, row_count, config.batch_size):
            real_rows = min(config.batch_size, row_count - first)
            batch = points[first:first + real_rows]
            if real_rows < config.batch_size:
                batch = tf.concat([batch, tf.repeat(batch[-1:], config.batch_size - real_rows, axis=0)], axis=0)
            if not bool(tf.reduce_all(tf.math.is_finite(batch))):
                result["status"] = "nonfinite_position"
                return None
            result["physical_rows"] += config.batch_size
            result["logical_rows"] += real_rows
            result["target_calls"] += 1
            values, scores, eligible = batched_target(batch)
            if values.dtype != tf.float64 or scores.dtype != tf.float64 or eligible.dtype != tf.bool:
                raise TypeError("callback must return float64 values/scores and bool eligibility")
            if values.shape != (config.batch_size,) or scores.shape != batch.shape or eligible.shape != values.shape:
                raise ValueError("callback returned incorrect fixed-batch shapes")
            if record_batch is not None:
                record_batch(role, batch, values, scores, eligible)
            if not bool(tf.reduce_all(tf.math.is_finite(values))) or not bool(tf.reduce_all(tf.math.is_finite(scores))):
                result["status"] = "nonfinite_target"
                return None
            if not bool(tf.reduce_all(eligible)):
                result["status"] = "ineligible_target"
                return None
            values_collected.append(values[:real_rows])
            scores_collected.append(scores[:real_rows])
        return tf.concat(values_collected, axis=0), tf.concat(scores_collected, axis=0)

    # Test local scale usability, not Gaussianity over the typical d-dimensional
    # Gaussian radius. This is the volume-uniform law in ||z|| <= radius.
    random = tf.random.stateless_normal([config.probe_count, dimension], config.seed, dtype=tf.float64)
    directions = random / tf.linalg.norm(random, axis=1, keepdims=True)
    uniforms = tf.random.stateless_uniform(
        [config.probe_count, 1], [config.seed[0], config.seed[1] + 1], dtype=tf.float64)
    radii = config.probe_radius * tf.pow(uniforms, 1.0 / dimension)
    latent = directions * radii
    locations = center + tf.matmul(latent, factor, transpose_b=True)
    result["latent_points"] = latent
    base = evaluate(locations, "guide")
    if base is None:
        return result
    result["guide_values"], result["guide_scores"] = base
    directions = tf.eye(dimension, dtype=tf.float64)
    perturbations = tf.concat([config.steps[0] * directions, -config.steps[0] * directions,
                              config.steps[1] * directions, -config.steps[1] * directions], axis=0)
    offsets = tf.matmul(perturbations, factor, transpose_b=True)
    for point_index in range(config.probe_count):
        evaluated = evaluate(locations[point_index] + offsets, f"curvature-{point_index:04d}")
        if evaluated is None:
            return result
        values, raw_scores = evaluated
        scores_z = tf.matmul(raw_scores, factor)
        if not bool(tf.reduce_all(tf.math.is_finite(scores_z))):
            result["status"] = "nonfinite_transformed_score"
            return result
        coarse = -tf.transpose(scores_z[:dimension] - scores_z[dimension:2 * dimension]) / (2 * config.steps[0])
        fine = -tf.transpose(scores_z[2 * dimension:3 * dimension] - scores_z[3 * dimension:]) / (2 * config.steps[1])
        # A large constant score cancels algebraically but loses precision in
        # floating point. Estimate that risk without dividing by the response.
        roundoff = (100 * 2.220446049250313e-16 / config.steps[1]) * tf.maximum(
            tf.constant(1.0, tf.float64), tf.linalg.norm(scores_z))
        diagnostics = compare_curvature_steps(coarse, fine, roundoff_margin=roundoff, config=config)
        # The same log-density values independently check that the manual
        # analytical score differentiates the actual scalar target. A smooth
        # but incorrect conservative score could otherwise pass the spectrum.
        value_slope_coarse = (values[:dimension] - values[dimension:2 * dimension]) / (2 * config.steps[0])
        value_slope_fine = (values[2 * dimension:3 * dimension] - values[3 * dimension:]) / (2 * config.steps[1])
        expected_score = tf.linalg.matvec(factor, result["guide_scores"][point_index], transpose_a=True)
        score_scale = tf.maximum(tf.constant(1.0, tf.float64), tf.reduce_max(tf.abs(expected_score)))
        value_gradient_error = tf.reduce_max(tf.abs(value_slope_fine - expected_score))
        value_gradient_step_error = tf.reduce_max(tf.abs(value_slope_fine - value_slope_coarse))
        value_gradient_consistent = tf.maximum(value_gradient_error, value_gradient_step_error) <= (
            config.resolution_atol + config.resolution_rtol * score_scale)
        diagnostics.update(value_gradient_error=value_gradient_error,
                           value_gradient_step_error=value_gradient_step_error,
                           value_gradient_consistent=value_gradient_consistent)
        diagnostics["passed"] &= value_gradient_consistent
        diagnostics["resolved"] &= value_gradient_consistent
        diagnostics["outside_band"] &= value_gradient_consistent
        diagnostics["inconclusive"] |= ~value_gradient_consistent
        result["point_diagnostics"].append({
            **diagnostics, "index": point_index, "radius": tf.linalg.norm(latent[point_index]),
            "coarse_curvature": coarse, "fine_curvature": fine,
        })
    nonpasses = sum(not bool(point["passed"]) for point in result["point_diagnostics"])
    upper = binomial_nonpass_upper(nonpasses, config.probe_count, config.confidence_alpha)
    result.update(nonpasses=nonpasses, nonpass_probability_upper=upper,
                  passed=upper <= config.max_nonpass_probability, status="completed")
    return result
