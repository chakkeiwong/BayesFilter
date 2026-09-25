"""Paired current/previous TT utilities.

The cores in this module have shape ``[r_left, n_u, n_v, r_right]`` and
represent ``h(u,v)`` in the interleaved order
``(u_1,v_1,u_2,v_2,...)``.  This is an extension of the Zhao--Cui route.  The
TensorFlow kernels are batch native; NumPy is intentionally absent.
"""
from __future__ import annotations

import math
from typing import Sequence

import tensorflow as tf

from bayesfilter.highdim.c2_gaussian_hermite_proposal_tf import (
    _normalized_hermite_values,
    _paired_right_environments,
    _log_standard_normal,
    inverse_hermite_polynomial_kr,
)

D = tf.float64


def pair_to_scalar_cores(cores: Sequence[tf.Tensor]) -> tuple[tf.Tensor, ...]:
    """Expand each two-variable core into two scalar cores exactly.

    The intermediate rank of block ``i`` is ``n*r_right``.  The second core is
    a fixed selection tensor, so this expansion preserves the represented
    polynomial exactly and is useful as an independent KR/reference route.
    """
    result = []
    for core in cores:
        core = tf.convert_to_tensor(core, D)
        rl, nu, nv, rr = (int(x) for x in core.shape)
        first = tf.reshape(core, [rl, nu, nv * rr])
        second = tf.reshape(tf.eye(nv * rr, dtype=D), [nv * rr, nv, rr])
        result.extend((first, second))
    return tuple(result)


def evaluate_pair_cores(cores: Sequence[tf.Tensor], points: tf.Tensor) -> tf.Tensor:
    """Evaluate pair cores at ``points[N,d,2]``."""
    points = tf.convert_to_tensor(points, D)
    state = tf.ones([tf.shape(points)[0], 1], D)
    for axis, core in enumerate(cores):
        degree = int(core.shape[1]) - 1
        fu = _normalized_hermite_values(points[:, axis, 0], degree)
        fv = _normalized_hermite_values(points[:, axis, 1], degree)
        state = tf.einsum("na,aklc,nk,nl->nc", state, core, fu, fv)
    return state[:, 0]


def pair_total_mass(cores: Sequence[tf.Tensor]) -> tf.Tensor:
    """Return ``integral h(u,v)^2`` under the product Gaussian reference."""
    scalar = pair_to_scalar_cores(cores)
    return _paired_right_environments(scalar, tf.ones([1, 1], D))[0][0, 0]


def pair_retained_quadratic(cores: Sequence[tf.Tensor], u: tf.Tensor) -> tf.Tensor:
    """Evaluate ``Q(u)=integral h(u,v)^2 rho(v) dv`` exactly by matrix maps."""
    u = tf.convert_to_tensor(u, D)
    env = tf.ones([tf.shape(u)[0], 1, 1], D)
    for axis in range(len(cores) - 1, -1, -1):
        core = cores[axis]
        degree = int(core.shape[1]) - 1
        basis = _normalized_hermite_values(u[:, axis], degree)
        # B[n,a,beta,c] = sum_alpha C[a,alpha,beta,c] phi_alpha(u).
        block = tf.einsum("aklc,nk->nalc", core, basis)
        env = tf.einsum("ncd,nakc,nbkd->nab", env, block, block)
    return env[:, 0, 0]


def pair_conditional_cores(cores: Sequence[tf.Tensor], v: tf.Tensor) -> tuple[tf.Tensor, ...]:
    """Contract each past coordinate, yielding particle-specific scalar cores."""
    v = tf.convert_to_tensor(v, D)
    result = []
    for axis, core in enumerate(cores):
        degree = int(core.shape[1]) - 1
        basis = _normalized_hermite_values(v[:, axis], degree)
        result.append(tf.einsum("aklc,nl->nakc", core, basis))
    return tuple(result)


def conditional_normalizer_batched(cores: Sequence[tf.Tensor]) -> tf.Tensor:
    """Normalizers for batched scalar TT amplitudes under Gaussian reference."""
    n = tf.shape(cores[0])[0]
    env = tf.ones([n, 1, 1], D)
    for core in cores:
        degree = int(core.shape[2]) - 1
        gram = tf.eye(degree + 1, dtype=D)
        env = tf.einsum("nab,nakc,nbld,kl->ncd", env, core, core, gram)
    return env[:, 0, 0]


def batched_inverse_kr(cores: Sequence[tf.Tensor], uniforms: tf.Tensor,
                       bisection_iterations: int = 64):
    """The shared Hermite KR authority, with particle-specific core batches."""
    return inverse_hermite_polynomial_kr(
        cores, tf.ones([1, 1], D), uniforms,
        bisection_iterations=bisection_iterations)


def sample_pair_conditional(cores: Sequence[tf.Tensor], v: tf.Tensor, tau: tf.Tensor,
                            mixture_uniform: tf.Tensor, uniforms: tf.Tensor,
                            normal_noise: tf.Tensor):
    """Sample ``q(u|v)`` and return its coordinate-space log density."""
    conditional = pair_conditional_cores(cores, v)
    z = conditional_normalizer_batched(conditional)
    # A zero polynomial has zero mixture probability. Give the unused KR
    # branch a normalized constant amplitude so 0/0 cannot poison the batch.
    safe_conditional = []
    for core in conditional:
        unit = tf.reshape(tf.one_hot(0, int(core.shape[1])*int(core.shape[2])*int(core.shape[3]), dtype=D),
                          [1, core.shape[1], core.shape[2], core.shape[3]])
        safe_conditional.append(tf.where((z > 0)[:, None, None, None], core, unit))
    kr = batched_inverse_kr(tuple(safe_conditional), uniforms)
    selected = mixture_uniform < z / (z + tau)
    u = tf.where(selected[:, None], kr["reference_points"], normal_noise)
    state = tf.ones([tf.shape(u)[0], 1], D)
    for axis, core in enumerate(conditional):
        basis = _normalized_hermite_values(u[:, axis], int(core.shape[2]) - 1)
        state = tf.einsum("na,nakb,nk->nb", state, core, basis)
    quadratic = tf.square(state[:, 0])
    logq = tf.math.log(quadratic + tau) + _log_standard_normal(u) - tf.math.log(z + tau)
    gaussian_fraction = tau / (z + tau)
    return u, logq, {"cdf_residual": tf.reduce_max(kr["maximum_inverse_cdf_residual"]),
                     "minimum_conditional_gaussian_fraction": tf.reduce_min(gaussian_fraction),
                     "maximum_conditional_gaussian_fraction": tf.reduce_max(gaussian_fraction),
                     "gaussian_component_draws": tf.reduce_sum(tf.cast(~selected, tf.int32)),
                     "minimum_polynomial_conditional_mass": tf.reduce_min(z),
                     "cdf_bracket_valid": kr["cdf_bracket_valid"],
                     "minimum_endpoint_margin": kr["minimum_endpoint_margin"],
                     "finite": kr["finite"] & kr["cdf_bracket_valid"] & tf.reduce_all(z >= 0) & (tau > 0) & tf.reduce_all(tf.math.is_finite(logq))}


_PAIR_FITTERS = {}


def _pair_features(points: tf.Tensor, degree: int) -> tf.Tensor:
    """Return ``[N,d,n,n]`` Hermite features for paired coordinates."""
    return tf.stack([
        tf.einsum("na,nb->nab", _normalized_hermite_values(points[:, i, 0], degree),
                  _normalized_hermite_values(points[:, i, 1], degree))
        for i in range(int(points.shape[1]))
    ], axis=1)


def compiled_pair_fitter(dimension: int, degree: int = 3, rank: int = 3,
                         sweeps: int = 4, proximal_steps: int = 128,
                         jit_compile: bool = True, regularization_center: str = "zero"):
    """ALS/proximal fit for pair cores with weighted L2 rows."""
    if regularization_center not in ("zero", "initial"):
        raise ValueError("regularization_center must be zero or initial")
    key = (dimension, degree, rank, sweeps, proximal_steps, jit_compile, regularization_center)
    if key in _PAIR_FITTERS:
        return _PAIR_FITTERS[key]
    n = degree + 1
    ranks = (1,) + (rank,) * (dimension - 1) + (1,)
    shapes = tuple((ranks[i], n, n, ranks[i + 1]) for i in range(dimension))
    signature = [tf.TensorSpec([None, dimension, n, n], D), tf.TensorSpec([None], D),
                 tf.TensorSpec([None], D), tf.TensorSpec([], D),
                 tuple(tf.TensorSpec(s, D) for s in shapes)]

    @tf.function(input_signature=signature, jit_compile=jit_compile, autograph=False)
    def fit(feat, target, row_weights, penalty, initial):
        cores = list(initial)
        decreases, conditions = [], []
        sqrtw = tf.sqrt(row_weights)
        for _ in range(sweeps):
            for axis in range(dimension):
                left = tf.ones([tf.shape(feat)[0], 1], D)
                for j in range(axis):
                    left = tf.einsum("na,aklb,nkl->nb", left, cores[j], feat[:, j])
                right = tf.ones([tf.shape(feat)[0], 1], D)
                for j in range(dimension - 1, axis, -1):
                    right = tf.einsum("aklb,nkl,nb->na", cores[j], feat[:, j], right)
                design = tf.reshape(tf.einsum("na,nkl,nb->naklb", left, feat[:, axis], right),
                                    [tf.shape(feat)[0], -1])
                design = design * sqrtw[:, None]
                weighted_target = target * sqrtw
                norm = tf.cast(tf.shape(target)[0], D)
                gram = tf.linalg.matmul(design, design, transpose_a=True) / norm
                rhs = tf.linalg.matvec(design, weighted_target, transpose_a=True) / norm
                eigenvalues = tf.linalg.eigvalsh(gram)
                largest = tf.reduce_max(eigenvalues)
                lipschitz = tf.where(largest > 0., largest, tf.constant(1., D))
                conditions.append(largest / tf.maximum(tf.reduce_min(eigenvalues),
                    tf.constant(2.220446049250313e-16, D)*lipschitz))
                old = tf.reshape(cores[axis], [-1])
                anchor = (tf.reshape(initial[axis], [-1]) if regularization_center == "initial"
                          else tf.zeros_like(old))

                def objective(v):
                    return (0.5 * tf.tensordot(v, tf.linalg.matvec(gram, v), 1)
                            - tf.tensordot(v, rhs, 1) + penalty * tf.reduce_sum(tf.abs(v-anchor)))

                def body(k, value):
                    proposal = value - (tf.linalg.matvec(gram, value) - rhs) / lipschitz
                    shifted = proposal-anchor
                    value = anchor + tf.sign(shifted) * tf.maximum(tf.abs(shifted) - penalty / lipschitz, 0.)
                    return k + 1, value

                old_obj = objective(old)
                _, value = tf.while_loop(lambda k, _: k < proximal_steps, body,
                                         (tf.constant(0), old), parallel_iterations=1)
                decreases.append(old_obj - objective(value))
                cores[axis] = tf.reshape(value, shapes[axis])
        final_grad = []
        for axis in range(dimension):
            left = tf.ones([tf.shape(feat)[0], 1], D)
            for j in range(axis):
                left = tf.einsum("na,aklb,nkl->nb", left, cores[j], feat[:, j])
            right = tf.ones([tf.shape(feat)[0], 1], D)
            for j in range(dimension - 1, axis, -1):
                right = tf.einsum("aklb,nkl,nb->na", cores[j], feat[:, j], right)
            design = tf.reshape(tf.einsum("na,nkl,nb->naklb", left, feat[:, axis], right), [tf.shape(feat)[0], -1])
            grad = tf.linalg.matvec(tf.linalg.matmul(design * sqrtw[:, None], design * sqrtw[:, None], transpose_a=True) /
                                    tf.cast(tf.shape(target)[0], D), tf.reshape(cores[axis], [-1])) - \
                   tf.linalg.matvec(design * sqrtw[:, None], target * sqrtw, transpose_a=True) / tf.cast(tf.shape(target)[0], D)
            coeff = tf.reshape(cores[axis], [-1])
            if regularization_center == "initial":
                coeff = coeff-tf.reshape(initial[axis], [-1])
            kkt = tf.where(coeff != 0., tf.abs(grad + penalty * tf.sign(coeff)),
                           tf.maximum(tf.abs(grad) - penalty, 0.))
            final_grad.append(tf.reduce_max(kkt))
        return tuple(cores), tf.reduce_min(tf.stack(decreases)), tf.reduce_max(tf.stack(final_grad)), tf.reduce_max(tf.stack(conditions))

    _PAIR_FITTERS[key] = fit
    return fit


def initial_pair_cores(dimension: int, degree: int = 3, rank: int = 3):
    """Existing deterministic generic start, also exposed for controlled diagnostics."""
    n = degree + 1
    ranks = (1,) + (rank,) * (dimension - 1) + (1,)
    initial = []
    for axis in range(dimension):
        shape = (ranks[axis], n, n, ranks[axis + 1])
        # Excite every channel: zero padding would trap one-site ALS at rank one.
        core = tf.random.stateless_normal(shape, [7919, axis], dtype=D) / tf.sqrt(tf.cast(n*n*ranks[axis+1], D))
        initial.append(tf.tensor_scatter_nd_add(core, [[0, 0, 0, 0]], [tf.constant(1., D)]))
    return tuple(initial)


def fit_pair_features(features: tf.Tensor, target: tf.Tensor, row_weights: tf.Tensor,
                      degree: int = 3, rank: int = 3, sweeps: int = 4,
                      proximal_steps: int = 128, penalty: float = 0., initial=None,
                      jit_compile: bool = True, regularization_center: str = "zero"):
    """Fit one weighted split; a small wrapper used by tests and the driver."""
    dimension = int(features.shape[1])
    n = degree + 1
    ranks = (1,) + (rank,) * (dimension - 1) + (1,)
    supplied_initial = initial is not None
    if initial is None:
        initial = initial_pair_cores(dimension, degree, rank)
    fitter = compiled_pair_fitter(dimension, degree, rank, sweeps, proximal_steps,
                                 jit_compile, regularization_center)
    cores, minimum_decrease, kkt, condition = fitter(features, target, row_weights,
                                          tf.constant(penalty, D), tuple(initial))
    tf.debugging.assert_all_finite(kkt, "pair KKT residual")
    tf.debugging.assert_greater_equal(minimum_decrease, tf.constant(-1e-8, D),
                                      "pair proximal objective increased")
    return cores, {"minimum_objective_decrease": minimum_decrease,
                   "maximum_core_gram_condition_capped": condition,
                   "gauge_policy": "raw_coefficients_no_infit_gauge_changes",
                   "regularization_center": regularization_center,
                   "initialization": ("explicit_initial_cores" if supplied_initial else
                       "stateless_7919_unit_expected_local_row_energy_plus_constant"),
                   "kkt_residual": kkt,
                   "rank_profile": list(ranks),
                   "weighted_rows": tf.reduce_sum(row_weights),
                   "effective_row_sample_size": tf.square(tf.reduce_sum(row_weights)) /
                   tf.reduce_sum(tf.square(row_weights))}


_PAIR_SAMPLERS = {}

def compiled_pair_sampler(shapes, jit_compile=True):
    """Compile the full pair conditional consumer with stable batch signatures."""
    key = (tuple(tuple(s) for s in shapes), jit_compile)
    if key not in _PAIR_SAMPLERS:
        d = len(shapes)
        signature = [tuple(tf.TensorSpec(s, D) for s in shapes),
                     tf.TensorSpec([None, d], D), tf.TensorSpec([], D),
                     tf.TensorSpec([None], D), tf.TensorSpec([None, d], D),
                     tf.TensorSpec([None, d], D)]
        _PAIR_SAMPLERS[key] = tf.function(sample_pair_conditional,
            input_signature=signature, jit_compile=jit_compile, autograph=False)
    return _PAIR_SAMPLERS[key]


def pair_bond_spectra(cores):
    """Post-fit diagnostic Schmidt spectra; never modifies fitted cores."""
    work = [tf.reshape(c, [c.shape[0], c.shape[1]*c.shape[2], c.shape[3]]) for c in cores]
    for i in range(len(work)-1, 0, -1):
        c = work[i]
        q, r = tf.linalg.qr(tf.transpose(tf.reshape(c, [c.shape[0], -1])), full_matrices=False)
        work[i] = tf.reshape(tf.transpose(q), [q.shape[1], c.shape[1], c.shape[2]])
        work[i-1] = tf.einsum('anb,cb->anc', work[i-1], r)
    result = []
    for i in range(len(work)-1):
        c = work[i]
        matrix = tf.reshape(c, [-1, c.shape[2]])
        singular, left, right = tf.linalg.svd(matrix, full_matrices=False)
        result.append({'singular_values': singular,
            'effective_rank': tf.reduce_sum(tf.cast(singular > singular[0]*tf.cast(max(matrix.shape), D)*2.220446049250313e-16, tf.int32))})
        work[i] = tf.reshape(left, [c.shape[0], c.shape[1], left.shape[1]])
        work[i+1] = tf.einsum('ab,bnc->anc', singular[:, None]*tf.transpose(right), work[i+1])
    return result
