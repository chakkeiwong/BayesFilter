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

from collections import OrderedDict
from types import SimpleNamespace

import tensorflow as tf

from bayesfilter.highdim.bases import LegendreBasis1D, ProductBasis, _legendre_values
from bayesfilter.highdim.retained_quadratic_form_tf import RetainedQuadraticForm
from bayesfilter.highdim.tt import TTCore
from bayesfilter.ops.quadrature_tf import gauss_legendre

DTYPE = tf.float64
_MOMENT_CACHE = OrderedDict()


def _moment_mass_matrix(product_basis: ProductBasis, axis: int, power: int, *, jit_compile=True) -> tf.Tensor:
    """M^(p)_kl = int z^p phi_k phi_l dmu on axis `axis` (exact GL setup constant)."""

    basis = product_basis.bases[axis]
    order = int(basis.basis_dim) + 2
    nodes, weights = gauss_legendre(order, jit_compile=jit_compile)
    # normalized reference measure on [-1,1]: mu-weights = GL weights / 2
    values = basis.evaluate(nodes)  # [order, K]
    w = (weights / 2.0) * nodes ** power
    return tf.einsum("q,qk,ql->kl", w, values, values)


def _legendre_moment_masses(basis, width, *, jit_compile):
    """Batch the same per-axis rules and Legendre recurrence, including mixed degrees."""
    rules = tuple(gauss_legendre(int(part.basis_dim) + 2, jit_compile=jit_compile)
                  for part in basis.bases)
    nodes = tf.stack(tuple(tf.pad(rule[0], [[0, width + 2 - rule[0].shape[0]]])
                           for rule in rules))
    weights = tf.stack(tuple(tf.pad(rule[1], [[0, width + 2 - rule[1].shape[0]]])
                             for rule in rules))
    bounds = tf.stack(tuple((part.domain.left, part.domain.right) for part in basis.bases))
    counts = tf.constant(tuple(int(part.basis_dim) for part in basis.bases))
    xi = 2.0 * (nodes - bounds[:, :1]) / (bounds[:, 1:] - bounds[:, :1]) - 1.0
    mask = tf.sequence_mask(counts, width, dtype=DTYPE)
    scales = tf.sqrt(tf.cast(2 * tf.range(width) + 1, DTYPE))
    values = _legendre_values(xi, width - 1) * scales * mask[:, None, :]
    weighted_powers = (weights[:, None, :] / 2.0
                       * tf.stack([nodes ** 1, nodes ** 2], axis=1))
    moments = tf.einsum("apq,aqk,aql->apkl", weighted_powers, values, values)
    return tf.concat([tf.linalg.diag(mask)[:, None], moments], axis=1)


def retained_reference_moments(
    retained: RetainedQuadraticForm,
    *,
    jit_compile: bool = True,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Exact (mean [n], covariance [n,n]) of the retained law in z-coordinates."""

    program, inputs = retained_moment_program(retained, jit_compile=jit_compile)
    return program(*inputs)


def retained_moment_program(retained, *, jit_compile=True):
    """Compile complete moments for a fixed basis and live numerical inputs."""
    basis = retained.prefix_basis
    shapes = tuple(tuple(core.values.shape) for core in retained.prefix_cores)
    gram_shape = tuple(retained.suffix_gram.shape)
    key = (id(basis), shapes, gram_shape, bool(jit_compile))
    if key in _MOMENT_CACHE:
        _MOMENT_CACHE.move_to_end(key)
        program = _MOMENT_CACHE[key][1]
    else:
        @tf.function(input_signature=[
            tuple(tf.TensorSpec(shape, DTYPE) for shape in shapes),
            tf.TensorSpec(gram_shape, DTYPE), tf.TensorSpec([], DTYPE),
            tf.TensorSpec([], DTYPE),
        ], jit_compile=jit_compile, autograph=False)
        def program(values, gram, tau, zc):
            current = SimpleNamespace(
                prefix_cores=tuple(TTCore(value) for value in values),
                prefix_basis=basis, suffix_gram=gram, tau=tau, z_complete_ref=zc)
            return _retained_reference_moments(current, jit_compile=jit_compile)

        _MOMENT_CACHE[key] = (basis, program)
        if len(_MOMENT_CACHE) > 16:
            _MOMENT_CACHE.popitem(last=False)
    return program, (tuple(core.values for core in retained.prefix_cores),
                     retained.suffix_gram, retained.tau, retained.z_complete_ref)


def _retained_reference_moments(retained, *, jit_compile):
    cores = retained.prefix_cores
    basis = retained.prefix_basis
    gram = retained.suffix_gram
    tau = retained.tau
    zc = retained.z_complete_ref
    n = len(cores)
    # Heterogeneous object-to-tensor packing is fixed schema plumbing. All
    # numerical axis and moment recurrences below execute in one TF loop.
    rank = max(max(int(core.values.shape[0]), int(core.values.shape[2])) for core in cores)
    width = max(int(core.values.shape[1]) for core in cores)
    packed = tf.stack(tuple(tf.pad(core.values, [
        [0, rank - int(core.values.shape[0])],
        [0, width - int(core.values.shape[1])],
        [0, rank - int(core.values.shape[2])],
    ]) for core in cores))

    def mass_branch(axis):
        def evaluate():
            count = int(basis.bases[axis].basis_dim)
            first = _moment_mass_matrix(basis, axis, 1, jit_compile=jit_compile)
            second = _moment_mass_matrix(basis, axis, 2, jit_compile=jit_compile)
            return tf.pad(tf.stack([tf.eye(count, dtype=DTYPE), first, second]),
                          [[0, 0], [0, width - count], [0, width - count]])
        return evaluate
    if all(isinstance(part, LegendreBasis1D) for part in basis.bases):
        # Basis domains and quadrature are immutable setup, independent of the
        # live cores, Gram matrix, defensive weight and normalizer.
        with tf.init_scope():
            prepare_masses = tf.function(
                lambda: _legendre_moment_masses(basis, width, jit_compile=jit_compile),
                input_signature=[], jit_compile=jit_compile, autograph=False,
            )
            masses = prepare_masses()
    else:
        masses = None
        branches = tuple(mass_branch(axis) for axis in range(n))
    axis_index = tf.range(n)
    first_power = tf.eye(n, dtype=tf.int32)
    j, k = tf.meshgrid(axis_index, axis_index, indexing="ij")
    second_power = (tf.one_hot(tf.reshape(j, [-1]), n, dtype=tf.int32)
                    + tf.one_hot(tf.reshape(k, [-1]), n, dtype=tf.int32))
    powers = tf.concat([first_power, second_power], axis=0)
    initial = tf.pad(tf.ones([n + n * n, 1, 1], DTYPE), [[0, 0], [0, rank - 1], [0, rank - 1]])

    def contract(axis, state):
        axis_mass = masses[axis] if masses is not None else tf.switch_case(axis, branch_fns=branches)
        mass = tf.gather(axis_mass, powers[:, axis])
        core = packed[axis]
        return axis + 1, tf.einsum("akb,cld,qkl,qac->qbd", core, core, mass, state)
    _, state = tf.while_loop(
        lambda axis, _: axis < n, contract,
        (tf.constant(0), initial), maximum_iterations=n,
    )
    padded_gram = tf.pad(gram, [[0, rank - int(gram.shape[0])], [0, rank - int(gram.shape[1])]])
    contractions = tf.einsum("qab,ab->q", state, padded_gram)
    mean = contractions[:n] / zc
    second_matrix = (tf.reshape(contractions[n:], [n, n]) + tau * tf.eye(n, dtype=DTYPE) / 3.0) / zc
    covariance = second_matrix - mean[:, None] * mean[None, :]
    return mean, covariance


__all__ = ["retained_reference_moments"]
