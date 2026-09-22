"""Gaussian-Hermite retained-TT proposals for the C2 diagnostic route.

The fitted squared TT is normalized and used only as a proposal.  This module
implements the Gaussian-reference incomplete Hermite Gram and a batched KR
inverse for the retained polynomial component.  It does not evaluate model
evidence and it does not differentiate through proposal construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Mapping, Sequence

import tensorflow as tf

from bayesfilter.highdim.bases import ProductBasis
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


DTYPE = tf.float64
ROUTE_ID = "c2_gaussian_hermite_retained_tt_proposal_v1"
ROUTE_CLASSIFICATION = "extension_or_invention"
INCOMPLETE_GRAM_ID = "normalized_probabilists_hermite_closed_form_v1"
KR_INVERSE_ID = "paired_environment_bisection_float64_v1"
DEFAULT_BISECTION_ITERATIONS = 64
DEFAULT_INNER_BRACKET = 12.0
DEFAULT_OUTER_BRACKET = 24.0
_COMPILED_SAMPLER_CACHE: dict[tuple[str, int, int, bool], object] = {}


def normalized_hermite_incomplete_gram(
    points: tf.Tensor,
    max_degree: int,
) -> tf.Tensor:
    """Return M_ab(z)=int_-inf^z psi_a psi_b phi for every input z.

    The basis is psi_k=He_k/sqrt(k!), matching ``HermiteBasis1D``.  The
    implementation is a fixed finite sum, so its graph size depends only on
    the setup-static degree.
    """

    if int(max_degree) < 0:
        raise ValueError("max_degree must be nonnegative")
    z = tf.convert_to_tensor(points, DTYPE)
    flat = tf.reshape(z, [-1])
    maximum_order = 2 * int(max_degree)

    hermites = [tf.ones_like(flat)]
    if maximum_order >= 1:
        hermites.append(flat)
    for order in range(1, maximum_order):
        hermites.append(
            flat * hermites[order]
            - tf.cast(order, DTYPE) * hermites[order - 1]
        )

    log_two_pi = tf.constant(math.log(2.0 * math.pi), DTYPE)
    phi = tf.exp(-0.5 * tf.square(flat) - 0.5 * log_two_pi)
    cdf = 0.5 * tf.math.erfc(-flat / tf.sqrt(tf.constant(2.0, DTYPE)))
    antiderivatives = [cdf]
    for order in range(1, maximum_order + 1):
        antiderivatives.append(-phi * hermites[order - 1])

    rows = []
    for left_degree in range(max_degree + 1):
        columns = []
        for right_degree in range(max_degree + 1):
            value = tf.zeros_like(flat)
            for contraction in range(min(left_degree, right_degree) + 1):
                coefficient = (
                    math.factorial(contraction)
                    * math.comb(left_degree, contraction)
                    * math.comb(right_degree, contraction)
                )
                order = left_degree + right_degree - 2 * contraction
                value = value + tf.cast(coefficient, DTYPE) * antiderivatives[order]
            value = value / tf.constant(
                math.sqrt(
                    math.factorial(left_degree) * math.factorial(right_degree)
                ),
                DTYPE,
            )
            columns.append(value)
        rows.append(tf.stack(columns, axis=-1))
    matrix = tf.stack(rows, axis=-2)
    output_shape = tf.concat(
        [tf.shape(z), [max_degree + 1, max_degree + 1]], axis=0
    )
    return tf.reshape(matrix, output_shape)


def _normalized_hermite_values(points: tf.Tensor, max_degree: int) -> tf.Tensor:
    points = tf.convert_to_tensor(points, DTYPE)
    flat = tf.reshape(points, [-1])
    values = [tf.ones_like(flat)]
    if max_degree >= 1:
        values.append(flat)
    for degree in range(1, max_degree):
        degree_value = tf.cast(degree, DTYPE)
        values.append(
            (
                flat * values[degree]
                - tf.sqrt(degree_value) * values[degree - 1]
            )
            / tf.sqrt(degree_value + 1.0)
        )
    stacked = tf.stack(values, axis=-1)
    return tf.reshape(
        stacked, tf.concat([tf.shape(points), [max_degree + 1]], axis=0)
    )


def _prefix_row_vectors(
    core_values: Sequence[tf.Tensor], points: tf.Tensor
) -> tf.Tensor:
    points = tf.convert_to_tensor(points, DTYPE)
    if not core_values:
        raise ValueError("at least one prefix core is required")
    degree = int(core_values[0].shape[1]) - 1
    state = tf.ones([tf.shape(points)[0], 1], DTYPE)
    for axis, core in enumerate(core_values):
        basis = _normalized_hermite_values(points[:, axis], degree)
        state = tf.einsum("na,akb,nk->nb", state, core, basis)
    return state


def _paired_right_environments(
    core_values: Sequence[tf.Tensor], suffix_gram: tf.Tensor
) -> tuple[tf.Tensor, ...]:
    environments = [tf.convert_to_tensor(suffix_gram, DTYPE)]
    for core in reversed(tuple(core_values)):
        environments.append(
            tf.einsum("akb,ckd,bd->ac", core, core, environments[-1])
        )
    return tuple(reversed(environments))


def _conditional_mass(
    left_environment: tf.Tensor,
    core: tf.Tensor,
    right_environment: tf.Tensor,
) -> tf.Tensor:
    return tf.einsum(
        "nac,akb,ckd,bd->n",
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
        "nac,akb,cld,nkl,bd->n",
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
        "nac,akb,cld,nk,nl->nbd",
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
        if int(self.time_index) < 1:
            raise ValueError("retained transition proposals require time_index >= 1")
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
        vectors = _prefix_row_vectors(self.prefix_core_values, reference_points)
        return tf.einsum("na,ab,nb->n", vectors, self.suffix_gram, vectors)

    def reference_log_density(self, reference_points: tf.Tensor) -> tf.Tensor:
        reference = tf.ensure_shape(
            tf.convert_to_tensor(reference_points, DTYPE), [None, self.dimension]
        )
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
        centered = physical - self.coordinate_offset[None, :]
        reference = tf.transpose(
            tf.linalg.triangular_solve(
                self.coordinate_matrix, tf.transpose(centered), lower=True
            )
        )
        log_det = tf.reduce_sum(tf.math.log(tf.linalg.diag_part(self.coordinate_matrix)))
        return self.reference_log_density(reference) - log_det

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
        count = int(particle_count)
        dimension = self.dimension
        cache_key = (
            self.proposal_id,
            count,
            int(bisection_iterations),
            bool(jit_compile),
        )
        cached = _COMPILED_SAMPLER_CACHE.get(cache_key)
        if cached is not None:
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
            return self.sample_physical(
                mixture_uniforms,
                hermite_uniforms,
                defensive_samples,
                bisection_iterations=int(bisection_iterations),
            )

        _COMPILED_SAMPLER_CACHE[cache_key] = sample
        return sample

    def _inverse_polynomial_component(
        self, uniforms: tf.Tensor, *, bisection_iterations: int
    ) -> Mapping[str, tf.Tensor]:
        uniforms = tf.convert_to_tensor(uniforms, DTYPE)
        particle_count = tf.shape(uniforms)[0]
        right_environments = _paired_right_environments(
            self.prefix_core_values, self.suffix_gram
        )
        left = tf.ones([particle_count, 1, 1], DTYPE)
        generated = []
        residuals = []
        conditional_masses = []
        endpoint_margins = []
        bracket_flags = []

        for axis, core in enumerate(self.prefix_core_values):
            right = right_environments[axis + 1]
            denominator = _conditional_mass(left, core, right)
            conditional_masses.append(tf.reduce_min(denominator))
            target = uniforms[:, axis]
            inner_lower = tf.fill([particle_count], tf.constant(-DEFAULT_INNER_BRACKET, DTYPE))
            inner_upper = tf.fill([particle_count], tf.constant(DEFAULT_INNER_BRACKET, DTYPE))
            lower_cdf_inner = _conditional_cdf(
                left, core, right, inner_lower, self.degree, denominator
            )
            upper_cdf_inner = _conditional_cdf(
                left, core, right, inner_upper, self.degree, denominator
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
                left, core, right, lower, self.degree, denominator
            )
            upper_cdf = _conditional_cdf(
                left, core, right, upper, self.degree, denominator
            )
            bracket_valid = (
                (denominator > 0.0)
                & tf.math.is_finite(denominator)
                & tf.math.is_finite(lower_cdf)
                & tf.math.is_finite(upper_cdf)
                & (lower_cdf <= target)
                & (target <= upper_cdf)
            )
            bracket_flags.append(tf.reduce_all(bracket_valid))
            endpoint_margins.append(
                tf.reduce_min(tf.minimum(target - lower_cdf, upper_cdf - target))
            )

            def condition(iteration, _lower, _upper):
                return iteration < int(bisection_iterations)

            def body(iteration, current_lower, current_upper):
                midpoint = 0.5 * (current_lower + current_upper)
                midpoint_cdf = _conditional_cdf(
                    left, core, right, midpoint, self.degree, denominator
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
                left, core, right, root, self.degree, denominator
            )
            residuals.append(tf.reduce_max(tf.abs(root_cdf - target)))
            generated.append(root)
            left = _update_left_environment(left, core, root, self.degree)

        reference_points = tf.stack(generated, axis=1)
        return {
            "reference_points": reference_points,
            "maximum_inverse_cdf_residual": tf.reduce_max(tf.stack(residuals)),
            "minimum_conditional_mass": tf.reduce_min(tf.stack(conditional_masses)),
            "minimum_endpoint_margin": tf.reduce_min(tf.stack(endpoint_margins)),
            "cdf_bracket_valid": tf.reduce_all(tf.stack(bracket_flags)),
            "finite": tf.reduce_all(tf.math.is_finite(reference_points))
            & tf.reduce_all(tf.math.is_finite(tf.stack(residuals))),
        }

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
        full_cores = tuple(
            TTCore(tf.convert_to_tensor(value, DTYPE))
            for value in snapshot.fitted_core_values
        )
        current_basis = _hermite_product_basis(n, snapshot.basis_degree)
        mixed_basis = ProductBasis(
            list(current_basis.bases)
            + [DiscreteIndicatorBasis1D(snapshot.branch_count)]
            + list(_hermite_product_basis(n, snapshot.basis_degree).bases),
            current_basis.convention,
        )
        suffix_gram = suffix_gram_matrix(
            full_cores[n:], mixed_basis, axis_offset=n
        )
        tau_relative = tf.math.expm1(
            snapshot.raw_increment - snapshot.corrected_increment
        )
        tau_abs = tau_relative * snapshot.z_h
        prefix_core_values = tuple(core.values for core in full_cores[:n])
        coordinate_offset = snapshot.joint_mean[:n]
        coordinate_matrix = snapshot.joint_chol[:n, :n]
        expected_complete = snapshot.z_h * (1.0 + tau_relative)
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


def stateless_proposal_random_inputs(
    proposal: GaussianHermiteRetainedProposal,
    particle_count: int,
    seed: tuple[int, int],
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Generate one fixed set of mixture, KR, and defensive random inputs."""

    count = int(particle_count)
    dimension = proposal.dimension
    first_seed, second_seed = (int(seed[0]), int(seed[1]))
    mixture = tf.random.stateless_uniform(
        [count], [first_seed, second_seed + 1], dtype=DTYPE
    )
    hermite = tf.random.stateless_uniform(
        [count, dimension], [first_seed, second_seed + 2], dtype=DTYPE
    )
    normal = tf.random.stateless_normal(
        [count, dimension], [first_seed, second_seed + 3], dtype=DTYPE
    )
    if proposal.defensive_nu is None:
        defensive = normal
    else:
        chi_square = tf.random.stateless_gamma(
            [count, dimension],
            [first_seed, second_seed + 4],
            alpha=tf.constant(proposal.defensive_nu / 2.0, DTYPE),
            beta=tf.constant(0.5, DTYPE),
            dtype=DTYPE,
        )
        defensive = normal / tf.sqrt(chi_square / proposal.defensive_nu)
    return mixture, hermite, defensive


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
    "GaussianHermiteRetainedProposal",
    "INCOMPLETE_GRAM_ID",
    "KR_INVERSE_ID",
    "ROUTE_CLASSIFICATION",
    "ROUTE_ID",
    "normalized_hermite_incomplete_gram",
    "retained_proposal_from_transition_snapshot",
    "stateless_proposal_random_inputs",
]
