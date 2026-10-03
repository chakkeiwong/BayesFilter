"""Independent analytic residual-geometry diagnostic for supplied funnel maps.

For v=scale*z0, x_i=exp(s(v))*z_i, r(v)=2*s(v)-v, the Jacobian gives
U(z)=z0**2/2 + exp(r)*sum(z_i**2)/2 - k*r/2, up to a constant.
The original normalized funnel and the transformed law differ by exactly the
map Jacobian. This module only reports geometry; it cannot tune or admit a pair.
"""
from __future__ import annotations

from functools import lru_cache
import math

import tensorflow as tf


@lru_cache(maxsize=12)
def residual_geometry_program(kind, *, scale=3., jit_compile=True):
    """Return analytic potential/score/Hessian with one batch-polymorphic graph.

    Exact whitening has r=0. Partial maps use the M20 fixture
    s(v)=2*scale*tanh(a*v/(4*scale)); their child curvature exp(r) is unbounded
    as v tends to negative infinity. Positive local eigenvalues describe a
    harmonic approximation, not a stability theorem for nonlinear trajectories.
    """
    if kind not in {"exact", "partial", "partial_half"}:
        raise ValueError("unknown supplied funnel map")
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and positive")
    strength = {"exact": None, "partial": 1., "partial_half": .5}[kind]

    @tf.function(input_signature=[tf.TensorSpec((None, 3), tf.float64)],
                 autograph=False, jit_compile=jit_compile)
    def compute(latent):
        v = scale * latent[:, 0]
        if strength is None:
            residual = tf.zeros_like(v)
            first = tf.zeros_like(v)
            second = tf.zeros_like(v)
            log_scale = v / 2.
        else:
            saturation = 2. * scale
            tangent = tf.tanh(strength * v / (2. * saturation))
            sech2 = 1. - tf.square(tangent)
            log_scale = saturation * tangent
            residual = 2. * log_scale - v
            first = strength * sech2 - 1.
            second = -(strength**2 / saturation) * sech2 * tangent
        child_curvature = tf.exp(residual)
        children = latent[:, 1:]
        weighted_square = child_curvature * tf.reduce_sum(tf.square(children), axis=1)
        potential = .5 * tf.square(latent[:, 0]) + .5 * weighted_square - residual
        gradient = tf.concat((
            (latent[:, 0] + scale * first * (.5 * weighted_square - 1.))[:, None],
            child_curvature[:, None] * children), axis=1)
        vv = 1. + scale**2 * (
            .5 * weighted_square * (tf.square(first) + second) - second)
        cross = scale * first[:, None] * child_curvature[:, None] * children
        zeros = tf.zeros_like(v)
        hessian = tf.stack((
            tf.concat((vv[:, None], cross), axis=1),
            tf.stack((cross[:, 0], child_curvature, zeros), axis=1),
            tf.stack((cross[:, 1], zeros, child_curvature), axis=1)), axis=1)
        return {"potential": potential, "score": -gradient, "hessian": hessian,
                "v": v, "log_scale": log_scale,
                "log_child_curvature": residual, "child_curvature": child_curvature,
                "log_jacobian": math.log(scale) + 2. * log_scale}

    return compute


def model_to_latent(values, kind, *, scale=3.):
    """Invert the analytic *model-coordinate* map for saved-state diagnosis."""
    values = tf.convert_to_tensor(values, tf.float64)
    if kind not in {"exact", "partial", "partial_half"}:
        raise ValueError("unknown supplied funnel map")
    v = values[..., 0]
    if kind == "exact":
        log_scale = v / 2.
    else:
        strength = 1. if kind == "partial" else .5
        log_scale = 2. * scale * tf.tanh(strength * v / (4. * scale))
    return tf.concat(((v / scale)[..., None],
                      values[..., 1:] * tf.exp(-log_scale[..., None])), axis=-1)
