"""Exact moments of a RetainedQuadraticForm (adapted-maps design note M1).

Design note: docs/plans/bayesfilter-adapted-coordinate-maps-design-note-2026-08-20.md
(Section 3, source `retained_exact`).

Lemma (derivation): with p_ret_ref(z) = (H_L(z) E H_L(z)' + tau) / Zc on
B = [-1,1]^n under the normalized reference measure mu, every moment of
the quadratic-form part is a prefix Gram chain in which selected axes
carry a MOMENT-WEIGHTED mass matrix

    M^(p)_kl = int z^p phi_k(z) phi_l(z) mu(dz),   p in {0,1,2},

(M^(0) = I for the orthonormal reference basis). Concretely,

    int z_j q(z) mu(dz)      = < chain(M^(1) at axis j) , E >
    int z_j^2 q(z) mu(dz)    = < chain(M^(2) at axis j) , E >
    int z_j z_k q(z) mu(dz)  = < chain(M^(1) at axes j,k) , E >,  j != k,

with chain(...) the standard prefix Gram recursion of
`prefix_gram_matrix` and <.,.> the Frobenius pairing with the suffix
Gram E. The defensive part contributes tau * E_mu[z_j] = 0 to first
moments and tau * delta_jk / 3 to second moments (uniform reference).
All contractions are TensorFlow; the Gauss-Legendre nodes used to build
the per-axis M^(p) constants are frozen setup constants (same status as
`_gauss_rows`). Integrands are polynomials of degree <= 2*deg + 2, so
order deg + 2 nodes are exact.

Validated by tests/highdim/test_retained_moments.py against dense
tensor-quadrature reference moments (U-MAP-MOM-1).
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from bayesfilter.highdim.bases import ProductBasis
from bayesfilter.highdim.retained_quadratic_form_tf import RetainedQuadraticForm

DTYPE = tf.float64


def _moment_mass_matrix(product_basis: ProductBasis, axis: int, power: int) -> tf.Tensor:
    """M^(p)_kl = int z^p phi_k phi_l dmu on axis `axis` (exact GL setup constant)."""

    basis = product_basis.bases[axis]
    order = int(basis.basis_dim) + 2
    nodes, weights = np.polynomial.legendre.leggauss(order)
    # normalized reference measure on [-1,1]: mu-weights = GL weights / 2
    values = basis.evaluate(tf.constant(nodes, DTYPE))  # [order, K]
    w = tf.constant(weights / 2.0, DTYPE) * tf.constant(nodes, DTYPE) ** power
    return tf.einsum("q,qk,ql->kl", w, values, values)


def _weighted_prefix_chain(
    prefix_cores, product_basis: ProductBasis, special: dict[int, tf.Tensor]
) -> tf.Tensor:
    """Prefix Gram chain with axis-indexed mass overrides (else reference I)."""

    state = tf.ones([1, 1], DTYPE)
    for axis, core in enumerate(prefix_cores):
        if axis in special:
            mass = special[axis]
            state = tf.einsum("akb,AlB,kl,aA->bB", core.values, core.values, mass, state)
        else:
            # reference-measure mass matrix is the identity (orthonormal basis)
            state = tf.einsum("akb,AkB,aA->bB", core.values, core.values, state)
    return state


def retained_reference_moments(
    retained: RetainedQuadraticForm,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Exact (mean [n], covariance [n,n]) of the retained law in z-coordinates."""

    cores = retained.prefix_cores
    basis = retained.prefix_basis
    gram = retained.suffix_gram
    tau = retained.tau
    zc = retained.z_complete_ref
    n = len(cores)
    m1 = {j: _moment_mass_matrix(basis, j, 1) for j in range(n)}
    m2 = {j: _moment_mass_matrix(basis, j, 2) for j in range(n)}

    def paired(special: dict[int, tf.Tensor]) -> tf.Tensor:
        return tf.einsum(
            "ab,ab->", _weighted_prefix_chain(cores, basis, special), gram
        )

    mean = tf.stack([paired({j: m1[j]}) / zc for j in range(n)])
    second = [[None] * n for _ in range(n)]
    third = tf.constant(1.0 / 3.0, DTYPE)
    for j in range(n):
        for k in range(j, n):
            if j == k:
                value = (paired({j: m2[j]}) + tau * third) / zc
            else:
                value = paired({j: m1[j], k: m1[k]}) / zc
            second[j][k] = value
            second[k][j] = value
    second_matrix = tf.stack([tf.stack(row) for row in second])
    covariance = second_matrix - mean[:, None] * mean[None, :]
    return mean, covariance


__all__ = ["retained_reference_moments"]
