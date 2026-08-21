"""Dynamic Nelson--Siegel curve maps with a hard or softplus lower bound.

Implements survey eqs. (76)--(78): forward-curve loadings, Gauss--Legendre
maturity averaging, and the two bound maps. All runtime tensors are
float64 TensorFlow. The Gauss--Legendre nodes/weights are host-side module
constants (declared boundary in the master program, Sec. 5).
"""

from __future__ import annotations

import numpy as _np  # host-side constants only; see module docstring
import tensorflow as tf

DTYPE = tf.float64
DEFAULT_QUADRATURE_ORDER = 40


def gauss_legendre_unit(order: int) -> tuple[tf.Tensor, tf.Tensor]:
    """Nodes and weights of order-``order`` Gauss--Legendre on [0, 1]."""
    x, w = _np.polynomial.legendre.leggauss(int(order))
    nodes = 0.5 * (x + 1.0)
    weights = 0.5 * w
    return (
        tf.constant(nodes, dtype=DTYPE),
        tf.constant(weights, dtype=DTYPE),
    )


def dns_loadings(s: tf.Tensor, decay: float) -> tf.Tensor:
    """DNS forward loadings a_c(s) = (1, e^{-lam s}, lam s e^{-lam s}).

    ``s`` has shape [...]; the result has shape [..., 3].
    """
    s = tf.convert_to_tensor(s, DTYPE)
    lam = tf.constant(decay, DTYPE)
    e = tf.exp(-lam * s)
    return tf.stack([tf.ones_like(s), e, lam * s * e], axis=-1)


def bound_hard(u: tf.Tensor, lower_bound: float) -> tf.Tensor:
    """m_ell(u) = max(ell, u): the C1 censoring map."""
    return tf.maximum(tf.constant(lower_bound, DTYPE), u)


def bound_softplus(u: tf.Tensor, lower_bound: float, alpha: float) -> tf.Tensor:
    """s_{ell,alpha}(u) = ell + alpha * softplus((u - ell)/alpha): S1 map."""
    ell = tf.constant(lower_bound, DTYPE)
    a = tf.constant(alpha, DTYPE)
    return ell + a * tf.nn.softplus((u - ell) / a)


def yield_curve(
    factors: tf.Tensor,
    maturities: tf.Tensor,
    decay: float,
    lower_bound: float,
    alpha: float,
    bound_map: str,
    order: int = DEFAULT_QUADRATURE_ORDER,
) -> tf.Tensor:
    """Bounded-forward yields y(tau) for one country (survey eq. (78)).

    ``factors``: [..., 3]; ``maturities``: [M]. Returns [..., M].
    ``bound_map``: "hard" (C1) or "softplus" (S1).
    """
    factors = tf.convert_to_tensor(factors, DTYPE)
    maturities = tf.convert_to_tensor(maturities, DTYPE)
    nodes, weights = gauss_legendre_unit(order)
    # horizons s_{ik} = tau_i * v_k, shape [M, K]
    horizons = maturities[:, None] * nodes[None, :]
    loadings = dns_loadings(horizons, decay)  # [M, K, 3]
    # forwards f(s_{ik}; x) = a(s_{ik}) . x, batched: [..., M, K]
    forwards = tf.einsum("mki,...i->...mk", loadings, factors)
    if bound_map == "hard":
        bounded = bound_hard(forwards, lower_bound)
    elif bound_map == "softplus":
        bounded = bound_softplus(forwards, lower_bound, alpha)
    else:
        raise ValueError(f"unknown bound_map: {bound_map!r}")
    return tf.einsum("k,...mk->...m", weights, bounded)
