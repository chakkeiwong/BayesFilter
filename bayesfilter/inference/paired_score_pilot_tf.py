"""Two-step local score curvature, not posterior-region qualification.

For theta=c+F z, callers supply row scores g_z=g_theta F. Axial designs
are ordered [+h I; -h I]. Their unrestricted quadratic least-squares fit
is the transpose of (g_z(-h I)-g_z(+h I))/(2h). Both raw symmetric fits
must pass before the smaller step is usable; no SPD projection is applied.
An independent rotated design may veto but cannot change the fitted matrix.

This TF kernel never evaluates a target, moves a center or mutates a guide.
The caller owns eligibility, fixed-batch padding/replay and the position
factor with covariance F K^-1 F'. Probe width is not covariance scaling.
The fixed-signature factory defaults to XLA. This numerical dependency does
not qualify an enclosing target, initializer, transport or HMC path.
"""

from __future__ import annotations

import math

import tensorflow as tf

from bayesfilter.inference.mass_matrix_tf import eigenpair_program
from bayesfilter.ops.compiled_tensor_program_tf import in_xla_context


def validate_paired_steps(steps):
    """Require two finite, positive, strictly decreasing declared probe widths."""
    if len(steps) != 2 or not all(math.isfinite(step) and step > 0 for step in steps):
        raise ValueError("paired steps must contain two positive finite widths")
    if not steps[0] > steps[1]:
        raise ValueError("paired steps must be strictly decreasing")


def paired_score_probe_designs_tf(dimension, *, seed, round_index, steps=(0.001, 0.0001)):
    """Return two axial designs and a disjoint stateless rotated check design.

    Each design has 2D rows, positive directions followed by negative ones.
    The check frame uses [seed,7000+round_index], never the fitting responses.
    Padding belongs to the caller and is excluded from all fitted matrices.
    """
    validate_paired_steps(steps)
    axes = tf.eye(dimension, dtype=tf.float64)
    normal = tf.random.stateless_normal([dimension, dimension], [seed, 7000 + round_index], dtype=tf.float64)
    frame, _ = tf.linalg.qr(normal)
    return (tf.concat((steps[0] * axes, -steps[0] * axes), axis=0),
            tf.concat((steps[1] * axes, -steps[1] * axes), axis=0),
            tf.concat((steps[0] * frame, -steps[0] * frame), axis=0))


def _norm(tensor):
    scale = tf.reduce_max(tf.abs(tensor))
    return scale * tf.linalg.norm(tf.math.divide_no_nan(tensor, scale))


def _ratio(numerator, denominator):
    return tf.where((denominator > 0) & tf.math.is_finite(denominator),
                    numerator / denominator, tf.constant(math.inf, tf.float64))


def covariance_unit_score_error_tf(precision, response, offsets):
    """Compute ||(Y-ZK') C^-T||_F / ||ZC||_F, C=chol(K).

    Inputs are finite float64 arrays and K is SPD; the fitting entry point
    guards these prerequisites. The normalization is invariant under a
    consistent invertible coordinate change at the same physical points.
    A zero/nonfinite denominator rejects, including a zero probe design.
    """
    cholesky = tf.linalg.cholesky(precision)
    residual = response - tf.matmul(offsets, precision, transpose_b=True)
    whitened = tf.linalg.triangular_solve(cholesky, tf.transpose(residual))
    return _ratio(_norm(whitened), _norm(tf.matmul(offsets, cholesky)))


def fit_paired_score_precision_tf(center_score, first_scores, second_scores,
                                  check_offsets, check_scores, *, steps=(0.001, 0.0001),
                                  precision_condition_cap=1e10, model_relative_rmse_cap=0.20):
    """Fit local K and return raw diagnostics plus an explicit acceptance flag.

    Scores and check offsets have shape [2D,D]; center_score has shape [D].
    Both axial fits must be finite/SPD, condition<=cap, relative antisymmetry
    <=1e-4, and generalized eigenvalues (large relative to small step) within
    [1/1.05,1.05]. The independent check must pass response RMSE and covariance
    error<=0.05. These are numerical heuristics, not Hessian error bounds.

    Raw matrices survive rejection. Internal identity operands only protect
    undefined factorizations; accepted=False must never supply usable geometry.
    The smaller-step raw_precision is returned even on rejection for inspection,
    never replaced with the better-looking step or a positive-definite matrix.
    """
    validate_paired_steps(steps)
    tensors = tuple(tf.convert_to_tensor(value) for value in
                    (center_score, first_scores, second_scores, check_offsets, check_scores))
    if any(tensor.dtype != tf.float64 for tensor in tensors):
        raise TypeError("paired score inputs must have dtype float64")
    center, first, second, offsets, checks = tensors
    if center.shape.rank != 1 or not center.shape.is_fully_defined() or center.shape[0] < 1:
        raise ValueError("center_score must have static shape [D], D>0")
    dimension = int(center.shape[0])
    if any(tensor.shape != (2 * dimension, dimension) for tensor in tensors[1:]):
        raise ValueError("paired score matrices must have shape [2D,D]")
    finite_input = tf.reduce_all(tf.math.is_finite(center))
    finite_input &= tf.reduce_all(tf.math.is_finite(tf.stack((first, second, offsets, checks))))
    raw_first = tf.transpose((first[dimension:] - first[:dimension]) / (2 * steps[0]))
    raw_second = tf.transpose((second[dimension:] - second[:dimension]) / (2 * steps[1]))
    raw = tf.stack((raw_first, raw_second))
    symmetric = 0.5 * raw + 0.5 * tf.linalg.matrix_transpose(raw)
    finite_fit = tf.reduce_all(tf.math.is_finite(symmetric), axis=[1, 2])
    scale = tf.reduce_max(tf.abs(symmetric), axis=[1, 2])
    normalized = symmetric / tf.where(finite_fit & (scale > 0), scale, 1.)[:, None, None]
    identity = tf.eye(dimension, batch_shape=[2], dtype=tf.float64)
    safe_normalized = tf.where(finite_fit[:, None, None], normalized, identity)
    if in_xla_context():
        # These are the two fixed probe widths, not a sample-wise map.
        # Both spectra need the same residual refinement as the consistency
        # congruence, especially for nearly repeated eigenvalues.
        eigenpairs = eigenpair_program(dimension)
        eigenvalues = tf.stack((eigenpairs(safe_normalized[0])[0],
                                eigenpairs(safe_normalized[1])[0]))
    else:
        eigenvalues = tf.linalg.eigvalsh(safe_normalized)
    condition = tf.where(eigenvalues[:, 0] > 0, eigenvalues[:, -1] / eigenvalues[:, 0],
                         tf.constant(math.inf, tf.float64))
    spd = finite_fit & tf.reduce_all(eigenvalues > 0, axis=1)
    antisymmetry = tf.stack((_ratio(_norm(raw_first - tf.transpose(raw_first)), _norm(raw_first)),
                            _ratio(_norm(raw_second - tf.transpose(raw_second)), _norm(raw_second))))
    preliminary = finite_input & tf.reduce_all(spd & (condition <= precision_condition_cap))
    preliminary &= tf.reduce_all(tf.math.is_finite(antisymmetry) & (antisymmetry <= 1e-4))
    safe_precision = tf.where(preliminary, symmetric, identity)
    cholesky = tf.linalg.cholesky(safe_precision[1])
    left = tf.linalg.triangular_solve(cholesky, safe_precision[0])
    congruence = tf.transpose(tf.linalg.triangular_solve(cholesky, tf.transpose(left)))
    generalized_matrix = 0.5 * congruence + 0.5 * tf.transpose(congruence)
    generalized = (eigenpair_program(dimension)(generalized_matrix)[0]
                   if in_xla_context() else tf.linalg.eigvalsh(generalized_matrix))
    consistent = tf.reduce_all(tf.math.is_finite(generalized) & (generalized >= 1 / 1.05) & (generalized <= 1.05))
    response = center[None, :] - checks
    prediction = tf.matmul(offsets, symmetric[1], transpose_b=True)
    rmse = _ratio(_norm(prediction - response), _norm(response))
    covariance_error = covariance_unit_score_error_tf(safe_precision[1], response, offsets)
    accepted = preliminary & consistent & tf.math.is_finite(rmse) & (rmse <= model_relative_rmse_cap)
    accepted &= tf.math.is_finite(covariance_error) & (covariance_error <= 0.05)
    diagnostic_nan = tf.constant(math.nan, tf.float64)
    return {"accepted": accepted, "raw_precision": symmetric[1],
            "raw_unrestricted_precisions": raw, "raw_symmetric_precisions": symmetric,
            "raw_eigenvalues": tf.where(finite_fit[:, None], eigenvalues * scale[:, None], diagnostic_nan),
            "raw_spd": tf.reduce_all(spd), "step_raw_spd": spd,
            "precision_condition": tf.reduce_max(condition), "step_precision_conditions": condition,
            "relative_antisymmetry": antisymmetry,
            "generalized_eigenvalues": tf.where(preliminary, generalized, diagnostic_nan),
            "selection_relative_rmse": rmse,
            "covariance_unit_error": tf.where(preliminary, covariance_error, diagnostic_nan),
            "design_rank": tf.constant(dimension, tf.int32),
            "design_condition": tf.constant(1., tf.float64)}


def make_paired_score_precision_program(dimension, *, steps=(0.001, 0.0001),
                                       precision_condition_cap=1e10,
                                       model_relative_rmse_cap=0.20, jit_compile=True):
    """Bind immutable fit controls; all five score/design arrays stay operands.

    Explicit graph mode is a reference/debug exception. The caller checks the
    returned acceptance flag before using geometry, including nonfinite inputs.
    """
    validate_paired_steps(steps)
    if dimension < 1:
        raise ValueError("dimension must be positive")
    signature = [tf.TensorSpec([dimension], tf.float64)] + [
        tf.TensorSpec([2 * dimension, dimension], tf.float64)] * 4

    @tf.function(input_signature=signature, jit_compile=jit_compile, autograph=False)
    def fit(center, first, second, check_offsets, check_scores):
        return fit_paired_score_precision_tf(
            center, first, second, check_offsets, check_scores, steps=steps,
            precision_condition_cap=precision_condition_cap,
            model_relative_rmse_cap=model_relative_rmse_cap)

    return fit
