"""Gaussian-Hermite retained-TT proposals for the C2 diagnostic route.

The fitted squared TT is normalized and used only as a proposal.  This module
implements the Gaussian-reference incomplete Hermite Gram and a batched KR
inverse for the retained polynomial component.  It does not evaluate model
evidence and it does not differentiate through proposal construction.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache

import tensorflow as tf

from bayesfilter.highdim.bases import ProductBasis, _hermite_normalized_values
from bayesfilter.highdim.retained_quadratic_form_tf import (
    TTCore,
    suffix_gram_matrix,
)
from bayesfilter.highdim.squared_tt_engine_gaussian_tf import (
    _hermite_product_basis,
)
from bayesfilter.highdim.squared_tt_engine_gaussian_xla_tf import (
    GaussianXLAFrozenTransitionSnapshot,
    GaussianXLARetainedProposalSnapshot,
    gaussian_xla_frozen_snapshot_fingerprint,
    gaussian_xla_retained_proposal_snapshot_fingerprint,
)
from bayesfilter.highdim.squared_tt_engine_v0_tf import DiscreteIndicatorBasis1D
from bayesfilter.ops.stateless_gamma_tf import philox_gamma_float64
from bayesfilter.ops.stateless_random_tf import (
    philox_normal_float64,
    philox_uniform_float64,
)

DTYPE = tf.float64
ROUTE_ID = "c2_gaussian_hermite_retained_tt_proposal_v1"
ROUTE_CLASSIFICATION = "extension_or_invention"
INCOMPLETE_GRAM_ID = "normalized_probabilists_hermite_closed_form_v1"
KR_INVERSE_ID = "paired_environment_bisection_float64_v1"
DEFAULT_BISECTION_ITERATIONS = 64
DEFAULT_INNER_BRACKET = 12.0
DEFAULT_OUTER_BRACKET = 24.0
_COMPILED_SAMPLER_CACHE = OrderedDict()
_COMPILED_DENSITY_CACHE = OrderedDict()


@lru_cache(maxsize=16)
def _gram_program(shape, degree):
    return tf.function(
        lambda points: normalized_hermite_incomplete_gram(points, degree),
        input_signature=[tf.TensorSpec(shape, DTYPE)], jit_compile=True, autograph=False,
    )


def normalized_hermite_incomplete_gram(
    points: tf.Tensor,
    max_degree: int,
) -> tf.Tensor:
    """Return M_ab(z)=int_-inf^z psi_a psi_b phi for every input z.

    The basis is psi_k=He_k/sqrt(k!), matching ``HermiteBasis1D``.  The
    Polynomial order and contraction sums use tensor recurrences; matrix
    entries share the recurrence instead of expanding one graph per degree.
    """

    if int(max_degree) < 0:
        raise ValueError("max_degree must be nonnegative")
    z = tf.convert_to_tensor(points, DTYPE)
    if tf.executing_eagerly():
        return _gram_program(tuple(z.shape), int(max_degree))(z)

    @tf.custom_gradient
    def evaluate(values):
        gram = _normalized_hermite_incomplete_gram_value(values, int(max_degree))

        def pullback(cotangent):
            basis = _normalized_hermite_values(values, int(max_degree))
            density = tf.exp(-0.5 * tf.square(values)) / tf.sqrt(
                tf.constant(2.0 * math.pi, DTYPE))
            derivative = basis[..., :, None] * basis[..., None, :] * density[..., None, None]
            return tf.reduce_sum(cotangent * derivative, axis=[-2, -1])

        return gram, pullback

    return evaluate(z)


def _normalized_hermite_incomplete_gram_value(z, max_degree):
    flat = tf.reshape(z, [-1])
    maximum_order = 2 * int(max_degree)

    hermites = tf.TensorArray(DTYPE, maximum_order + 1, element_shape=flat.shape,
                              clear_after_read=False).write(0, tf.ones_like(flat))
    if maximum_order >= 1:
        hermites = hermites.write(1, flat)

    def polynomial_step(order, values):
        value = flat * values.read(order) - tf.cast(order, DTYPE) * values.read(order - 1)
        return order + 1, values.write(order + 1, value)

    _, hermites = tf.while_loop(lambda order, _: order < maximum_order, polynomial_step,
        (tf.constant(1), hermites), maximum_iterations=max(0, maximum_order - 1))

    log_two_pi = tf.constant(math.log(2.0 * math.pi), DTYPE)
    phi = tf.exp(-0.5 * tf.square(flat) - 0.5 * log_two_pi)
    cdf = 0.5 * tf.math.erfc(-flat / tf.sqrt(tf.constant(2.0, DTYPE)))
    antiderivatives = tf.concat([cdf[None, :], -phi[None, :] * hermites.stack()[:-1]], 0)
    degrees = tf.range(max_degree + 1)
    left, right = degrees[:, None], degrees[None, :]
    factorials = tf.math.cumprod(tf.cast(tf.maximum(degrees, 1), DTYPE))
    normalization_squared = factorials[:, None] * factorials[None, :]

    def contraction_step(contraction, values):
        valid = contraction <= tf.minimum(left, right)
        coefficient = normalization_squared / (
            factorials[contraction]
            * tf.gather(factorials, tf.maximum(left - contraction, 0))
            * tf.gather(factorials, tf.maximum(right - contraction, 0)))
        order = tf.maximum(left + right - 2 * contraction, 0)
        term = coefficient[:, :, None] * tf.gather(antiderivatives, order)
        return contraction + 1, values + tf.where(valid[:, :, None], term, 0.0)

    _, values = tf.while_loop(lambda contraction, _: contraction <= max_degree,
        contraction_step,
        (tf.constant(0), tf.zeros([max_degree + 1, max_degree + 1, tf.size(flat)], DTYPE)),
        maximum_iterations=max_degree + 1, parallel_iterations=1)
    matrix = tf.transpose(values / tf.sqrt(normalization_squared)[:, :, None], [2, 0, 1])
    output_shape = tf.concat(
        [tf.shape(z), [max_degree + 1, max_degree + 1]], axis=0
    )
    return tf.reshape(matrix, output_shape)


def _normalized_hermite_values(points: tf.Tensor, max_degree: int) -> tf.Tensor:
    return _hermite_normalized_values(points, max_degree)


def _pack_prefix_cores(core_values):
    rank = max(max(core.shape[-3], core.shape[-1]) for core in core_values)
    batch_shape = tf.TensorShape([])
    batch_extent = tf.constant([], tf.int32)
    for core in core_values:
        batch_shape = tf.broadcast_static_shape(batch_shape, core.shape[:-3])
        batch_extent = tf.broadcast_dynamic_shape(batch_extent, tf.shape(core)[:-3])
    # Core topology is fixed when tracing; values remain runtime operands.
    return tf.stack(tuple(tf.ensure_shape(tf.broadcast_to(
        tf.pad(core, [[0, 0]] * (core.shape.rank - 3)
            + [[0, rank - core.shape[-3]], [0, 0], [0, rank - core.shape[-1]]]),
        tf.concat([batch_extent, [rank, core.shape[-2], rank]], 0)),
        batch_shape.concatenate([rank, core.shape[-2], rank]))
        for core in core_values))


def _prefix_row_vectors(
    core_values: Sequence[tf.Tensor], points: tf.Tensor
) -> tf.Tensor:
    points = tf.convert_to_tensor(points, DTYPE)
    if not core_values:
        raise ValueError("at least one prefix core is required")
    degree = int(core_values[0].shape[1]) - 1
    packed = _pack_prefix_cores(core_values)
    state = tf.one_hot(tf.zeros([tf.shape(points)[0]], tf.int32), packed.shape[1], dtype=DTYPE)
    basis = _normalized_hermite_values(points, degree)

    def step(axis, state):
        return axis + 1, tf.einsum("na,akb,nk->nb", state, packed[axis], basis[:, axis])

    _, state = tf.while_loop(lambda axis, _: axis < len(core_values), step,
        (tf.constant(0), state), maximum_iterations=len(core_values), parallel_iterations=1)
    return state[:, :core_values[-1].shape[2]]


def _packed_right_environments(packed, suffix_gram):
    count, rank = packed.shape[0], packed.shape[-1]
    state = tf.pad(suffix_gram, [[0, 0]] * (suffix_gram.shape.rank - 2)
        + [[0, rank - suffix_gram.shape[-2]], [0, rank - suffix_gram.shape[-1]]])
    state = tf.broadcast_to(state, tf.concat([
        tf.broadcast_dynamic_shape(tf.shape(packed)[1:-3], tf.shape(state)[:-2]),
        [rank, rank]], 0))
    state = tf.ensure_shape(state, tf.broadcast_static_shape(
        packed.shape[1:-3], suffix_gram.shape[:-2]).concatenate([rank, rank]))
    values = tf.TensorArray(DTYPE, count + 1, element_shape=state.shape).write(count, state)

    def step(index, state, values):
        core = packed[count - index - 1]
        state = tf.einsum("...akb,...ckd,...bd->...ac", core, core, state)
        return index + 1, state, values.write(count - index - 1, state)

    _, _, values = tf.while_loop(lambda index, *_: index < count, step,
        (tf.constant(0), state, values), maximum_iterations=count, parallel_iterations=1)
    return values.stack()


@lru_cache(maxsize=16)
def _environments_program(specifications):
    return tf.function(lambda *values: _paired_right_environments(values[:-1], values[-1]),
                       input_signature=specifications, jit_compile=True, autograph=False)


def _paired_right_environments(
    core_values: Sequence[tf.Tensor], suffix_gram: tf.Tensor
) -> tuple[tf.Tensor, ...]:
    if not core_values:
        return (tf.convert_to_tensor(suffix_gram, DTYPE),)
    if tf.executing_eagerly():
        values = (*core_values, suffix_gram)
        signature = tuple(tf.TensorSpec(value.shape, DTYPE) for value in values)
        return _environments_program(signature)(*values)
    packed = _pack_prefix_cores(core_values)
    values = _packed_right_environments(packed, tf.convert_to_tensor(suffix_gram, DTYPE))
    return tuple(values[axis, ..., :core.shape[-3], :core.shape[-3]]
                 for axis, core in enumerate(core_values)) + (
                     values[-1, ..., :suffix_gram.shape[-2], :suffix_gram.shape[-1]],)


def _conditional_mass(
    left_environment: tf.Tensor,
    core: tf.Tensor,
    right_environment: tf.Tensor,
) -> tf.Tensor:
    return tf.einsum(
        "...ac,...akb,...ckd,...bd->...",
        left_environment,
        core,
        core,
        right_environment,
    )


def _conditional_cdf(
    left_environment: tf.Tensor,
    core: tf.Tensor,
    right_environment: tf.Tensor,
    points: tf.Tensor,
    degree: int,
    denominator: tf.Tensor,
) -> tf.Tensor:
    incomplete = normalized_hermite_incomplete_gram(points, degree)
    numerator = tf.einsum(
        "...ac,...akb,...cld,...kl,...bd->...",
        left_environment,
        core,
        core,
        incomplete,
        right_environment,
    )
    return numerator / denominator


def _update_left_environment(
    left_environment: tf.Tensor,
    core: tf.Tensor,
    points: tf.Tensor,
    degree: int,
) -> tf.Tensor:
    basis = _normalized_hermite_values(points, degree)
    return tf.einsum(
        "...ac,...akb,...cld,...k,...l->...bd",
        left_environment,
        core,
        core,
        basis,
        basis,
    )


def _log_standard_normal(points: tf.Tensor) -> tf.Tensor:
    points = tf.convert_to_tensor(points, DTYPE)
    dimension = tf.cast(tf.shape(points)[1], DTYPE)
    return -0.5 * (
        dimension * tf.constant(math.log(2.0 * math.pi), DTYPE)
        + tf.reduce_sum(tf.square(points), axis=1)
    )


def _log_product_student_t(points: tf.Tensor, nu: float) -> tf.Tensor:
    points = tf.convert_to_tensor(points, DTYPE)
    nu_tensor = tf.constant(float(nu), DTYPE)
    log_constant = tf.constant(
        math.lgamma((float(nu) + 1.0) / 2.0)
        - math.lgamma(float(nu) / 2.0)
        - 0.5 * math.log(float(nu) * math.pi),
        DTYPE,
    )
    return tf.reduce_sum(
        log_constant
        - 0.5 * (nu_tensor + 1.0)
        * tf.math.log1p(tf.square(points) / nu_tensor),
        axis=1,
    )


@lru_cache(maxsize=16)
def _inverse_program(signature, bisection_iterations, reverse):
    return tf.function(lambda cores, suffix, uniforms: inverse_hermite_polynomial_kr(
        cores, suffix, uniforms, bisection_iterations=bisection_iterations, reverse=reverse),
        input_signature=signature, jit_compile=True, autograph=False)


def inverse_hermite_polynomial_kr(
    core_values: Sequence[tf.Tensor], suffix_gram: tf.Tensor, uniforms: tf.Tensor,
    *, bisection_iterations: int = DEFAULT_BISECTION_ITERATIONS, reverse: bool = False,
) -> Mapping[str, tf.Tensor]:
    """Exact incomplete-Gram KR, including upper sampling with a fixed suffix.

    ``reverse=True`` generates the rightmost current axis first. Its initial
    paired environment is the evaluated conditioning-suffix outer product,
    which may differ for each particle. Both directions use the same CDF and
    inverse kernel. Invalid masses/brackets are reported, never silently clipped.
    """
    if not core_values:
        raise ValueError("at least one prefix core is required")
    core_values = tuple(tf.convert_to_tensor(core, DTYPE) for core in core_values)
    suffix_gram = tf.convert_to_tensor(suffix_gram, DTYPE)
    uniforms = tf.convert_to_tensor(uniforms, DTYPE)
    if tf.executing_eagerly():
        signature = (tuple(tf.TensorSpec(core.shape, DTYPE) for core in core_values),
            tf.TensorSpec(suffix_gram.shape, DTYPE), tf.TensorSpec(uniforms.shape, DTYPE))
        return _inverse_program(signature, int(bisection_iterations), bool(reverse))(
            core_values, suffix_gram, uniforms)
    dimension = len(core_values)
    degree = int(core_values[0].shape[-2]) - 1
    suffix_gram = tf.convert_to_tensor(suffix_gram, DTYPE)
    work_cores = (tuple(tf.einsum("...akb->...bka", c) for c in reversed(core_values))
                  if reverse else tuple(core_values))
    integration_boundary = tf.ones([1, 1], DTYPE) if reverse else suffix_gram
    if reverse:
        uniforms = tf.reverse(uniforms, axis=[1])
    uniforms = tf.convert_to_tensor(uniforms, DTYPE)
    particle_count = tf.shape(uniforms)[0]
    packed = _pack_prefix_cores(work_cores)
    rank = packed.shape[-1]
    right_environments = _packed_right_environments(packed, integration_boundary)
    left = (tf.broadcast_to(suffix_gram, [particle_count, tf.shape(suffix_gram)[-2], tf.shape(suffix_gram)[-1]])
            if reverse else tf.ones([particle_count, 1, 1], DTYPE))
    left = tf.pad(left, [[0, 0], [0, rank - left.shape[-2]], [0, rank - left.shape[-1]]])
    generated = tf.TensorArray(DTYPE, dimension, element_shape=uniforms[:, 0].shape)
    residuals = tf.TensorArray(DTYPE, dimension, element_shape=[])
    conditional_masses = tf.TensorArray(DTYPE, dimension, element_shape=[])
    endpoint_margins = tf.TensorArray(DTYPE, dimension, element_shape=[])
    bracket_flags = tf.TensorArray(tf.bool, dimension, element_shape=[])

    def axis_step(axis, left, generated, residuals, conditional_masses, endpoint_margins, bracket_flags):
        core = packed[axis]
        right = right_environments[axis + 1]
        denominator = _conditional_mass(left, core, right)
        conditional_masses = conditional_masses.write(axis, tf.reduce_min(denominator))
        target = uniforms[:, axis]
        inner_lower = tf.fill([particle_count], tf.constant(-DEFAULT_INNER_BRACKET, DTYPE))
        inner_upper = tf.fill([particle_count], tf.constant(DEFAULT_INNER_BRACKET, DTYPE))
        lower_cdf_inner = _conditional_cdf(
            left, core, right, inner_lower, degree, denominator
        )
        upper_cdf_inner = _conditional_cdf(
            left, core, right, inner_upper, degree, denominator
        )
        lower = tf.where(
            target >= lower_cdf_inner,
            inner_lower,
            tf.fill([particle_count], tf.constant(-DEFAULT_OUTER_BRACKET, DTYPE)),
        )
        upper = tf.where(
            target <= upper_cdf_inner,
            inner_upper,
            tf.fill([particle_count], tf.constant(DEFAULT_OUTER_BRACKET, DTYPE)),
        )
        lower_cdf = _conditional_cdf(
            left, core, right, lower, degree, denominator
        )
        upper_cdf = _conditional_cdf(
            left, core, right, upper, degree, denominator
        )
        bracket_valid = (
            (denominator > 0.0)
            & tf.math.is_finite(denominator)
            & tf.math.is_finite(lower_cdf)
            & tf.math.is_finite(upper_cdf)
            & (lower_cdf <= target)
            & (target <= upper_cdf)
        )
        bracket_flags = bracket_flags.write(axis, tf.reduce_all(bracket_valid))
        endpoint_margins = endpoint_margins.write(axis,
            tf.reduce_min(tf.minimum(target - lower_cdf, upper_cdf - target))
        )

        def condition(iteration, _lower, _upper):
            return iteration < int(bisection_iterations)

        def body(iteration, current_lower, current_upper):
            midpoint = 0.5 * (current_lower + current_upper)
            midpoint_cdf = _conditional_cdf(
                left, core, right, midpoint, degree, denominator
            )
            move_lower = midpoint_cdf < target
            return (
                iteration + 1,
                tf.where(move_lower, midpoint, current_lower),
                tf.where(move_lower, current_upper, midpoint),
            )

        _, lower_final, upper_final = tf.while_loop(
            condition,
            body,
            (tf.constant(0, tf.int32), lower, upper),
            parallel_iterations=1,
        )
        root = 0.5 * (lower_final + upper_final)
        root_cdf = _conditional_cdf(
            left, core, right, root, degree, denominator
        )
        residuals = residuals.write(axis, tf.reduce_max(tf.abs(root_cdf - target)))
        generated = generated.write(axis, root)
        left = _update_left_environment(left, core, root, degree)

        return axis + 1, left, generated, residuals, conditional_masses, endpoint_margins, bracket_flags

    _, _, generated, residuals, conditional_masses, endpoint_margins, bracket_flags = tf.while_loop(
        lambda axis, *_: axis < dimension, axis_step,
        (tf.constant(0), left, generated, residuals, conditional_masses, endpoint_margins, bracket_flags),
        maximum_iterations=dimension, parallel_iterations=1)
    reference_points = tf.transpose(generated.stack())
    if reverse:
        reference_points = tf.reverse(reference_points, axis=[1])
    residuals = residuals.stack()
    return {
        "reference_points": reference_points,
        "maximum_inverse_cdf_residual": tf.reduce_max(residuals),
        "minimum_conditional_mass": tf.reduce_min(conditional_masses.stack()),
        "minimum_endpoint_margin": tf.reduce_min(endpoint_margins.stack()),
        "cdf_bracket_valid": tf.reduce_all(bracket_flags.stack()),
        "finite": tf.reduce_all(tf.math.is_finite(reference_points))
        & tf.reduce_all(tf.math.is_finite(residuals)),
    }


@dataclass(frozen=True)
class GaussianHermiteRetainedProposal:
    """A normalized retained squared-TT and defensive reference mixture."""

    prefix_core_values: tuple[tf.Tensor, ...]
    suffix_gram: tf.Tensor
    z_h: tf.Tensor
    tau_abs: tf.Tensor
    coordinate_offset: tf.Tensor
    coordinate_matrix: tf.Tensor
    defensive_nu: float | None
    time_index: int
    source_snapshot_fingerprint: str
    proposal_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not tf.executing_eagerly():
            raise RuntimeError("construct retained proposals before tracing")
        cores = tuple(tf.convert_to_tensor(value, DTYPE) for value in self.prefix_core_values)
        if not cores:
            raise ValueError("prefix_core_values must not be empty")
        basis_dimension = int(cores[0].shape[1])
        if basis_dimension < 1:
            raise ValueError("proposal basis dimension must be positive")
        previous_right = 1
        for axis, core in enumerate(cores):
            if core.shape.rank != 3 or not core.shape.is_fully_defined():
                raise ValueError("proposal cores require static rank-three shapes")
            left_rank, local_basis_dimension, right_rank = core.shape.as_list()
            if left_rank != previous_right:
                raise ValueError(f"proposal core rank mismatch at axis {axis}")
            if local_basis_dimension != basis_dimension:
                raise ValueError("proposal cores must share one Hermite degree")
            previous_right = right_rank
            _require_all_finite(f"prefix_core_values[{axis}]", core)

        gram = tf.convert_to_tensor(self.suffix_gram, DTYPE)
        if gram.shape != (previous_right, previous_right):
            raise ValueError("suffix_gram shape must close the prefix boundary")
        _require_all_finite("suffix_gram", gram)
        asymmetry = tf.reduce_max(tf.abs(gram - tf.transpose(gram)))
        gram_scale = tf.maximum(tf.reduce_max(tf.abs(gram)), tf.constant(1.0, DTYPE))
        if float(asymmetry.numpy()) > 1e-11 * float(gram_scale.numpy()):
            raise ValueError("suffix_gram must be symmetric")
        eigenvalues = tf.linalg.eigvalsh(gram)
        if float(eigenvalues[0].numpy()) < -1e-11 * float(gram_scale.numpy()):
            raise ValueError("suffix_gram must be positive semidefinite")

        z_h = tf.reshape(tf.convert_to_tensor(self.z_h, DTYPE), [])
        tau_abs = tf.reshape(tf.convert_to_tensor(self.tau_abs, DTYPE), [])
        if not bool(tf.math.is_finite(z_h).numpy()) or float(z_h.numpy()) <= 0.0:
            raise ValueError("z_h must be finite and positive")
        if not bool(tf.math.is_finite(tau_abs).numpy()) or float(tau_abs.numpy()) <= 0.0:
            raise ValueError("tau_abs must be finite and strictly positive")

        right_environments = _paired_right_environments(cores, gram)
        recomputed_z_h = tf.reshape(right_environments[0], [])
        tolerance = 2e-10 * max(1.0, abs(float(z_h.numpy())))
        if abs(float((recomputed_z_h - z_h).numpy())) > tolerance:
            raise ValueError("proposal prefix/suffix contraction does not match z_h")

        dimension = len(cores)
        offset = tf.reshape(tf.convert_to_tensor(self.coordinate_offset, DTYPE), [dimension])
        matrix = tf.convert_to_tensor(self.coordinate_matrix, DTYPE)
        if matrix.shape != (dimension, dimension):
            raise ValueError("coordinate_matrix must have shape [dimension, dimension]")
        _require_all_finite("coordinate_offset", offset)
        _require_all_finite("coordinate_matrix", matrix)
        upper = tf.linalg.band_part(matrix, 0, -1) - tf.linalg.band_part(matrix, 0, 0)
        if float(tf.reduce_max(tf.abs(upper)).numpy()) > 1e-12:
            raise ValueError("coordinate_matrix must be lower triangular")
        if bool(tf.reduce_any(tf.linalg.diag_part(matrix) <= 0.0).numpy()):
            raise ValueError("coordinate_matrix diagonal must be positive")
        if self.defensive_nu is not None and float(self.defensive_nu) <= 0.0:
            raise ValueError("defensive_nu must be positive when supplied")
        # The observation-guided consumer also retains the initial (t=0) fit.
        if int(self.time_index) < 0:
            raise ValueError("retained proposals require time_index >= 0")
        if len(str(self.source_snapshot_fingerprint)) != 64:
            raise ValueError("source_snapshot_fingerprint must be a SHA-256 digest")

        object.__setattr__(self, "prefix_core_values", cores)
        object.__setattr__(self, "suffix_gram", gram)
        object.__setattr__(self, "z_h", z_h)
        object.__setattr__(self, "tau_abs", tau_abs)
        object.__setattr__(self, "coordinate_offset", offset)
        object.__setattr__(self, "coordinate_matrix", matrix)
        object.__setattr__(self, "proposal_id", _proposal_fingerprint(self))

    @property
    def dimension(self) -> int:
        return len(self.prefix_core_values)

    @property
    def degree(self) -> int:
        return int(self.prefix_core_values[0].shape[1]) - 1

    @property
    def z_complete(self) -> tf.Tensor:
        return self.z_h + self.tau_abs

    def reference_quadratic_form(self, reference_points: tf.Tensor) -> tf.Tensor:
        reference_points = tf.convert_to_tensor(reference_points, DTYPE)
        if tf.executing_eagerly():
            return self._compiled_density(reference_points.shape, "reference_quadratic_form")(reference_points)
        vectors = _prefix_row_vectors(self.prefix_core_values, reference_points)
        return tf.einsum("na,ab,nb->n", vectors, self.suffix_gram, vectors)

    def reference_log_density(self, reference_points: tf.Tensor) -> tf.Tensor:
        reference = tf.ensure_shape(
            tf.convert_to_tensor(reference_points, DTYPE), [None, self.dimension]
        )
        if tf.executing_eagerly():
            return self._compiled_density(reference.shape, "reference_log_density")(reference)
        quadratic = self.reference_quadratic_form(reference)
        valid_quadratic = tf.where(
            quadratic >= 0.0,
            quadratic,
            tf.fill(tf.shape(quadratic), tf.constant(float("nan"), DTYPE)),
        )
        polynomial_log = tf.math.log(valid_quadratic) + _log_standard_normal(reference)
        if self.defensive_nu is None:
            defensive_log = _log_standard_normal(reference)
        else:
            defensive_log = _log_product_student_t(reference, self.defensive_nu)
        mixture_log = tf.reduce_logsumexp(
            tf.stack(
                [
                    polynomial_log,
                    tf.math.log(self.tau_abs) + defensive_log,
                ],
                axis=1,
            ),
            axis=1,
        )
        return mixture_log - tf.math.log(self.z_complete)

    def physical_log_density(self, physical_points: tf.Tensor) -> tf.Tensor:
        physical = tf.ensure_shape(
            tf.convert_to_tensor(physical_points, DTYPE), [None, self.dimension]
        )
        if tf.executing_eagerly():
            return self._compiled_density(physical.shape, "physical_log_density")(physical)
        centered = physical - self.coordinate_offset[None, :]
        reference = tf.transpose(
            tf.linalg.triangular_solve(
                self.coordinate_matrix, tf.transpose(centered), lower=True
            )
        )
        log_det = tf.reduce_sum(tf.math.log(tf.linalg.diag_part(self.coordinate_matrix)))
        return self.reference_log_density(reference) - log_det

    def _compiled_density(self, shape, endpoint):
        key = (self.proposal_id, tuple(shape), endpoint)
        if key in _COMPILED_DENSITY_CACHE:
            _COMPILED_DENSITY_CACHE.move_to_end(key)
            return _COMPILED_DENSITY_CACHE[key]
        program = tf.function(getattr(self, endpoint), input_signature=[tf.TensorSpec(shape, DTYPE)],
                              jit_compile=True, autograph=False)
        _COMPILED_DENSITY_CACHE[key] = program
        if len(_COMPILED_DENSITY_CACHE) > 16:
            _COMPILED_DENSITY_CACHE.popitem(last=False)
        return program

    def sample_reference(
        self,
        mixture_uniforms: tf.Tensor,
        hermite_uniforms: tf.Tensor,
        defensive_samples: tf.Tensor,
        *,
        bisection_iterations: int = DEFAULT_BISECTION_ITERATIONS,
    ) -> Mapping[str, tf.Tensor]:
        mixture_uniforms = tf.reshape(tf.convert_to_tensor(mixture_uniforms, DTYPE), [-1])
        particle_count = mixture_uniforms.shape[0]
        if particle_count is None:
            raise ValueError("sample_reference requires a setup-static particle count")
        hermite_uniforms = tf.ensure_shape(
            tf.convert_to_tensor(hermite_uniforms, DTYPE),
            [particle_count, self.dimension],
        )
        defensive_samples = tf.ensure_shape(
            tf.convert_to_tensor(defensive_samples, DTYPE),
            [particle_count, self.dimension],
        )
        if tf.executing_eagerly():
            return self._compiled_sampler(particle_count, bisection_iterations,
                jit_compile=True, physical=False)(mixture_uniforms, hermite_uniforms, defensive_samples)
        hermite_result = self._inverse_polynomial_component(
            hermite_uniforms, bisection_iterations=int(bisection_iterations)
        )
        polynomial_probability = self.z_h / self.z_complete
        selected_polynomial = mixture_uniforms < polynomial_probability
        reference = tf.where(
            selected_polynomial[:, None],
            hermite_result["reference_points"],
            defensive_samples,
        )
        return {
            "reference_points": reference,
            "reference_log_density": self.reference_log_density(reference),
            "selected_polynomial": selected_polynomial,
            "polynomial_probability": polynomial_probability,
            "maximum_inverse_cdf_residual": hermite_result[
                "maximum_inverse_cdf_residual"
            ],
            "minimum_conditional_mass": hermite_result[
                "minimum_conditional_mass"
            ],
            "minimum_endpoint_margin": hermite_result["minimum_endpoint_margin"],
            "cdf_bracket_valid": hermite_result["cdf_bracket_valid"],
            "finite": hermite_result["finite"]
            & tf.reduce_all(tf.math.is_finite(reference)),
        }

    def sample_physical(
        self,
        mixture_uniforms: tf.Tensor,
        hermite_uniforms: tf.Tensor,
        defensive_samples: tf.Tensor,
        *,
        bisection_iterations: int = DEFAULT_BISECTION_ITERATIONS,
    ) -> Mapping[str, tf.Tensor]:
        if tf.executing_eagerly():
            mixture_uniforms = tf.reshape(tf.convert_to_tensor(mixture_uniforms, DTYPE), [-1])
            return self.compiled_sampler(mixture_uniforms.shape[0],
                bisection_iterations=bisection_iterations)(
                    mixture_uniforms, hermite_uniforms, defensive_samples)
        result = dict(
            self.sample_reference(
                mixture_uniforms,
                hermite_uniforms,
                defensive_samples,
                bisection_iterations=bisection_iterations,
            )
        )
        reference = result["reference_points"]
        physical = self.coordinate_offset[None, :] + tf.einsum(
            "ij,nj->ni", self.coordinate_matrix, reference
        )
        log_det = tf.reduce_sum(tf.math.log(tf.linalg.diag_part(self.coordinate_matrix)))
        result["physical_points"] = physical
        result["physical_log_density"] = result["reference_log_density"] - log_det
        result["finite"] = result["finite"] & tf.reduce_all(
            tf.math.is_finite(result["physical_log_density"])
        )
        return result

    def compiled_sampler(
        self,
        particle_count: int,
        *,
        bisection_iterations: int = DEFAULT_BISECTION_ITERATIONS,
        jit_compile: bool = True,
    ):
        return self._compiled_sampler(particle_count, bisection_iterations,
                                      jit_compile=jit_compile, physical=True)

    def _compiled_sampler(self, particle_count, bisection_iterations, *, jit_compile, physical):
        count = int(particle_count)
        dimension = self.dimension
        cache_key = (
            self.proposal_id,
            count,
            int(bisection_iterations),
            bool(jit_compile),
            bool(physical),
        )
        cached = _COMPILED_SAMPLER_CACHE.get(cache_key)
        if cached is not None:
            _COMPILED_SAMPLER_CACHE.move_to_end(cache_key)
            return cached

        @tf.function(
            input_signature=[
                tf.TensorSpec([count], DTYPE),
                tf.TensorSpec([count, dimension], DTYPE),
                tf.TensorSpec([count, dimension], DTYPE),
            ],
            jit_compile=bool(jit_compile),
            autograph=False,
        )
        def sample(mixture_uniforms, hermite_uniforms, defensive_samples):
            endpoint = self.sample_physical if physical else self.sample_reference
            return endpoint(
                mixture_uniforms,
                hermite_uniforms,
                defensive_samples,
                bisection_iterations=int(bisection_iterations),
            )

        _COMPILED_SAMPLER_CACHE[cache_key] = sample
        if len(_COMPILED_SAMPLER_CACHE) > 16:
            _COMPILED_SAMPLER_CACHE.popitem(last=False)
        return sample

    def _inverse_polynomial_component(
        self, uniforms: tf.Tensor, *, bisection_iterations: int
    ) -> Mapping[str, tf.Tensor]:
        return inverse_hermite_polynomial_kr(
            self.prefix_core_values, self.suffix_gram, uniforms,
            bisection_iterations=bisection_iterations,
        )

    def manifest_payload(self) -> Mapping[str, object]:
        return {
            "route_id": ROUTE_ID,
            "route_classification": ROUTE_CLASSIFICATION,
            "incomplete_gram_id": INCOMPLETE_GRAM_ID,
            "kr_inverse_id": KR_INVERSE_ID,
            "proposal_id": self.proposal_id,
            "time_index": int(self.time_index),
            "dimension": self.dimension,
            "degree": self.degree,
            "z_h": self.z_h,
            "tau_abs": self.tau_abs,
            "z_complete": self.z_complete,
            "defensive_nu": self.defensive_nu,
            "source_snapshot_fingerprint": self.source_snapshot_fingerprint,
            "proposal_parameter_dependence": "none_after_compilation",
            "complete_mixture_density": True,
            "exact_pseudo_marginal_claimed": False,
        }


@lru_cache(maxsize=16)
def _snapshot_preparation_program(dimension, degree, branch_count, specifications):
    current_basis = _hermite_product_basis(dimension, degree)
    mixed_basis = ProductBasis(
        list(current_basis.bases) + [DiscreteIndicatorBasis1D(branch_count)]
        + list(current_basis.bases), current_basis.convention)

    @tf.function(input_signature=specifications, jit_compile=True, autograph=False)
    def prepare(raw_increment, corrected_increment, z_h, mean, chol, *core_values):
        suffix = tuple(TTCore(value) for value in core_values[dimension:])
        gram = suffix_gram_matrix(suffix, mixed_basis, axis_offset=dimension)
        tau_relative = tf.math.expm1(raw_increment - corrected_increment)
        return (gram, tau_relative * z_h, mean[:dimension], chol[:dimension, :dimension],
                z_h * (1.0 + tau_relative))
    return prepare


def retained_proposal_from_transition_snapshot(
    snapshot: GaussianXLAFrozenTransitionSnapshot
    | GaussianXLARetainedProposalSnapshot,
) -> GaussianHermiteRetainedProposal:
    """Construct the post-update retained proposal from production-captured cores."""

    if isinstance(snapshot, GaussianXLARetainedProposalSnapshot):
        if snapshot.basis_identity != "hermite_retained_quadratic_form_v1":
            raise ValueError("snapshot does not use the retained Hermite basis")
        prefix_core_values = snapshot.prefix_core_values
        suffix_gram = snapshot.suffix_gram
        tau_abs = snapshot.tau_abs
        coordinate_offset = snapshot.coordinate_offset
        coordinate_matrix = snapshot.coordinate_matrix
        expected_complete = snapshot.z_complete
        source_fingerprint = gaussian_xla_retained_proposal_snapshot_fingerprint(
            snapshot
        )
    else:
        if snapshot.basis_identity != "hermite_reference_counting_branch_v1":
            raise ValueError("snapshot does not use the C2 Hermite/counting basis")
        n = int(snapshot.state_dim)
        values = (snapshot.raw_increment, snapshot.corrected_increment, snapshot.z_h,
                  snapshot.joint_mean, snapshot.joint_chol, *snapshot.fitted_core_values)
        specifications = tuple(tf.TensorSpec(value.shape, DTYPE) for value in values)
        suffix_gram, tau_abs, coordinate_offset, coordinate_matrix, expected_complete = (
            _snapshot_preparation_program(n, snapshot.basis_degree, snapshot.branch_count,
                                          specifications)(*values))
        prefix_core_values = snapshot.fitted_core_values[:n]
        source_fingerprint = gaussian_xla_frozen_snapshot_fingerprint(snapshot)
    proposal = GaussianHermiteRetainedProposal(
        prefix_core_values=prefix_core_values,
        suffix_gram=suffix_gram,
        z_h=snapshot.z_h,
        tau_abs=tau_abs,
        coordinate_offset=coordinate_offset,
        coordinate_matrix=coordinate_matrix,
        defensive_nu=snapshot.defensive_nu,
        time_index=snapshot.time_index,
        source_snapshot_fingerprint=source_fingerprint,
    )
    if abs(float((proposal.z_complete - expected_complete).numpy())) > (
        2e-10 * max(1.0, abs(float(expected_complete.numpy())))
    ):
        raise ValueError("snapshot tau reconstruction does not close z_complete")
    return proposal


@lru_cache(maxsize=16)
def _random_inputs_program(count, dimension, defensive_nu):
    @tf.function(input_signature=[tf.TensorSpec([2], tf.int64)],
                 jit_compile=True, autograph=False)
    def generate(seed):
        mixture = philox_uniform_float64([count], seed + [0, 1])
        hermite = philox_uniform_float64([count, dimension], seed + [0, 2])
        normal = philox_normal_float64([count, dimension], seed + [0, 3])
        if defensive_nu is None:
            defensive = normal
        else:
            chi_square = philox_gamma_float64([count, dimension], seed + [0, 4],
                tf.constant(defensive_nu / 2., DTYPE), tf.constant(.5, DTYPE))
            defensive = normal / tf.sqrt(chi_square / defensive_nu)
        return mixture, hermite, defensive

    return generate


def stateless_proposal_random_inputs(
    proposal: GaussianHermiteRetainedProposal,
    particle_count: int,
    seed: tuple[int, int],
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Generate one fixed set of mixture, KR, and defensive random inputs."""

    count = int(particle_count)
    first_seed, second_seed = (int(seed[0]), int(seed[1]))
    if count < 0:
        raise ValueError("particle_count must be nonnegative")
    if not -2**63 <= first_seed < 2**63 or not -2**63 <= second_seed < 2**63 - 4:
        raise ValueError("proposal seed and offsets must fit int64")
    return _random_inputs_program(count, proposal.dimension, proposal.defensive_nu)(
        tf.constant([first_seed, second_seed], tf.int64))


def _proposal_fingerprint(proposal: GaussianHermiteRetainedProposal) -> str:
    digest = hashlib.sha256()
    metadata = {
        "route_id": ROUTE_ID,
        "time_index": int(proposal.time_index),
        "defensive_nu": proposal.defensive_nu,
        "source_snapshot_fingerprint": proposal.source_snapshot_fingerprint,
    }
    digest.update(json.dumps(metadata, sort_keys=True).encode("utf-8"))
    for name, value in (
        *((f"core_{index}", core) for index, core in enumerate(proposal.prefix_core_values)),
        ("suffix_gram", proposal.suffix_gram),
        ("z_h", proposal.z_h),
        ("tau_abs", proposal.tau_abs),
        ("coordinate_offset", proposal.coordinate_offset),
        ("coordinate_matrix", proposal.coordinate_matrix),
    ):
        digest.update(name.encode("utf-8"))
        digest.update(bytes(tf.io.serialize_tensor(value).numpy()))
    return digest.hexdigest()


def _require_all_finite(name: str, value: tf.Tensor) -> None:
    if not bool(tf.reduce_all(tf.math.is_finite(value)).numpy()):
        raise ValueError(f"{name} must contain only finite values")


__all__ = [
    "DEFAULT_BISECTION_ITERATIONS",
    "INCOMPLETE_GRAM_ID",
    "KR_INVERSE_ID",
    "ROUTE_CLASSIFICATION",
    "ROUTE_ID",
    "GaussianHermiteRetainedProposal",
    "normalized_hermite_incomplete_gram",
    "retained_proposal_from_transition_snapshot",
    "stateless_proposal_random_inputs",
]
