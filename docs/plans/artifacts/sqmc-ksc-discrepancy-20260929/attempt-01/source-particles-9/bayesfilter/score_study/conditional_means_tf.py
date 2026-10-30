"""Shared conditional means for Gaussian-noise score-study fixtures.

Nonlinear terms are scalar fixture coefficients held fixed in the six-parameter
score. Zero curvature preserves the preceding linear algebra exactly.
"""
import math
import tensorflow as tf


def validate_curves(d, o, transition_curve, observation_curve):
    if not all(isinstance(v, (int, float)) and math.isfinite(v)
               for v in (transition_curve, observation_curve)):
        raise ValueError("finite conditional-mean curvature required")
    if (transition_curve or observation_curve) and (d != 1 or o != 1):
        raise ValueError("nonlinear conditional means require scalar state and observation")


def model_curves(row, settings):
    if row["model"] == "nonlinear_scalar":
        curves = {k: settings[k] for k in ("transition_curve", "observation_curve")}
    else:
        curves = {"transition_curve": 0., "observation_curve": 0.}
        if any(settings.get(k, 0.) != 0. for k in curves):
            raise ValueError("nonzero curvature requires the nonlinear model consumer")
    validate_curves(settings["dimension"], settings["observation_dimension"], **curves)
    return curves


def conditional_mean(x, matrix, curve=0., *, quadratic=False):
    value = tf.einsum("ij,nj->ni", matrix, x)
    if curve:
        value += tf.cast(curve, x.dtype) * (x*x if quadratic else tf.sin(x))
    return value


def conditional_mean_and_tangent(x, dx, matrix, dmatrix, curve=0., *, quadratic=False):
    tangent = tf.einsum("pij,nj->pni", dmatrix, x) + tf.einsum("ij,pnj->pni", matrix, dx)
    if curve:
        slope = 2*x if quadratic else tf.cos(x)
        tangent += tf.cast(curve, x.dtype) * slope[None, :, :] * dx
    return conditional_mean(x, matrix, curve, quadratic=quadratic), tangent
