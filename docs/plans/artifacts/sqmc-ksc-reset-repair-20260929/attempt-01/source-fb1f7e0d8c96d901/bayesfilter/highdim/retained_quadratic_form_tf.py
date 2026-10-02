"""RetainedQuadraticForm: exact end-block marginal of a squared TT (P1A).

Program: docs/plans/bayesfilter-zhao-cui-generic-highdim-analytic-score-program-plan-2026-08-15.md
Derivation: docs/plans/bayesfilter-zhao-cui-generic-program-ub1-score-derivation-note-2026-08-15.md
(Sections 1(V5) and 2; content gate passed 2026-08-16).

The retained filtered law of the generic squared-TT engine is the exact
marginal of a squared functional TT over its suffix (previous-state) axes.
Per Zhao-Cui Proposition 2 / Eq. (14) (author structure
`@TTSIRT/marginalise.m:25-85`), this marginal is a quadratic form

    p_ret_ref(z) = ( H_L(z) E H_L(z)' + tau * q0_ret_ref(z) ) / Zc_ref

with prefix row vector H_L(z) in R^{1 x r_c}, suffix Gram matrix E (PSD,
generally rank > 1), and the COMPLETE normalizer Zc_ref = Z_h + tau * Z0_ref.
It is generally NOT one scalar squared TT, and this module never converts
it to one (program veto V10).

Measure contract (UB-1 Section 1(V1)/(V5)): the object is defined ONCE as a
density with respect to the normalized reference measure
``mu(dz) = prod_i dz_i / length_i`` on the basis box (the measure under
which the repository Legendre bases are orthonormal: mass matrix = identity
for ``MassMeasure.REFERENCE_MEASURE``). The physical evaluator is the fixed
conversion

    p_ret_phys(x) = p_ret_ref(R^{-1}(x)) * w_ref(R^{-1}(x)) / J_R(R^{-1}(x))

with ``w_ref(z) = prod_i 1/length_i`` the reference-measure Lebesgue
density and ``J_R`` the coordinate-map Jacobian. There is no separate
physical normalizer: mass is preserved by the conversion (U-MEASURE-1
asserts this numerically).

The defensive marginal in this v1 module is the reference-uniform density
``q0_ret_ref(z) = 1`` (i.e. the reference measure itself), which is
product-form with ``Z0_ref = 1``. tau is a per-scope TUNED parameter
(owner decision D1); this module treats it as a declared input.

Tangent state (UB-1 Section 2): per parameter direction, ``dot_prefix``
core tangents and ``dot_E`` propagate through

    dot q = 2 v E dot_v' + v dot_E v',   dot Zc = dot Z_h,
    dot log p_ret_ref = (dot q) / (q + tau q0) - dot Zc / Zc,

with dot Z_h computed exactly from the prefix/suffix Gram chains
(bilinear; no quadrature). All tensors are TensorFlow float64.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import tensorflow as tf

from bayesfilter.highdim.bases import ProductBasis
from bayesfilter.highdim.diagnostics import MassMeasure
from bayesfilter.highdim.filtering import HighDimCoordinateMap
from bayesfilter.highdim.tt import TTCore

DTYPE = tf.float64


def _check_cores(cores: Sequence[TTCore], dimension: int, name: str) -> tuple[TTCore, ...]:
    checked = tuple(cores)
    if len(checked) != dimension:
        raise ValueError(f"{name}: expected {dimension} cores, got {len(checked)}")
    for core in checked:
        if not isinstance(core, TTCore):
            raise TypeError(f"{name}: cores must be TTCore objects")
    return checked


def prefix_row_vectors(
    cores: Sequence[TTCore],
    product_basis: ProductBasis,
    points: tf.Tensor,
) -> tf.Tensor:
    """Evaluate the prefix TT block as row vectors H_L(z) of shape [N, r_c]."""

    points = tf.convert_to_tensor(points, DTYPE)
    if points.shape.rank != 2 or int(points.shape[1]) != len(cores):
        raise ValueError("points must be [N, n_prefix_axes]")
    state = tf.ones([tf.shape(points)[0], 1], DTYPE)
    for axis, core in enumerate(cores):
        basis_values = product_basis.evaluate_axis(axis, points[:, axis])
        state = tf.einsum("nl,lkr,nk->nr", state, core.values, basis_values)
    return state


def prefix_row_vectors_tangent(
    cores: Sequence[TTCore],
    dot_cores: Sequence[TTCore],
    product_basis: ProductBasis,
    points: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Return (H_L(z), dot H_L(z)) by product-rule forward propagation."""

    points = tf.convert_to_tensor(points, DTYPE)
    state = tf.ones([tf.shape(points)[0], 1], DTYPE)
    dot_state = tf.zeros_like(state)
    for axis, (core, dot_core) in enumerate(zip(cores, dot_cores)):
        basis_values = product_basis.evaluate_axis(axis, points[:, axis])
        new_state = tf.einsum("nl,lkr,nk->nr", state, core.values, basis_values)
        dot_state = tf.einsum("nl,lkr,nk->nr", dot_state, core.values, basis_values) + tf.einsum(
            "nl,lkr,nk->nr", state, dot_core.values, basis_values
        )
        state = new_state
    return state, dot_state


def _axis_mass(product_basis: ProductBasis, axis: int, measure: MassMeasure) -> tf.Tensor:
    return product_basis.bases[axis].mass_matrix(measure)


def suffix_gram_matrix(
    suffix_cores: Sequence[TTCore],
    product_basis: ProductBasis,
    *,
    axis_offset: int,
    measure: MassMeasure = MassMeasure.REFERENCE_MEASURE,
) -> tf.Tensor:
    """Exact suffix Gram E = int H_R(u) H_R(u)' mu(du), shape [r_c, r_c].

    ``axis_offset`` is the index of the first suffix axis within the full
    product basis (bases are indexed over the full adjacent block).
    """

    state = tf.ones([1, 1], DTYPE)
    for step, core in enumerate(reversed(tuple(suffix_cores))):
        axis = axis_offset + len(suffix_cores) - 1 - step
        mass = _axis_mass(product_basis, axis, measure)
        state = tf.einsum("akb,AlB,kl,bB->aA", core.values, core.values, mass, state)
    return state


def suffix_gram_matrix_tangent(
    suffix_cores: Sequence[TTCore],
    dot_suffix_cores: Sequence[TTCore],
    product_basis: ProductBasis,
    *,
    axis_offset: int,
    measure: MassMeasure = MassMeasure.REFERENCE_MEASURE,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Return (E, dot E) by product rule through the suffix Gram chain."""

    state = tf.ones([1, 1], DTYPE)
    dot_state = tf.zeros_like(state)
    ordered = tuple(zip(suffix_cores, dot_suffix_cores))
    for step, (core, dot_core) in enumerate(reversed(ordered)):
        axis = axis_offset + len(ordered) - 1 - step
        mass = _axis_mass(product_basis, axis, measure)
        new_state = tf.einsum("akb,AlB,kl,bB->aA", core.values, core.values, mass, state)
        dot_state = (
            tf.einsum("akb,AlB,kl,bB->aA", dot_core.values, core.values, mass, state)
            + tf.einsum("akb,AlB,kl,bB->aA", core.values, dot_core.values, mass, state)
            + tf.einsum("akb,AlB,kl,bB->aA", core.values, core.values, mass, dot_state)
        )
        state = new_state
    return state, dot_state


def prefix_gram_matrix(
    prefix_cores: Sequence[TTCore],
    product_basis: ProductBasis,
    *,
    measure: MassMeasure = MassMeasure.REFERENCE_MEASURE,
) -> tf.Tensor:
    """Exact prefix Gram P = int H_L(z)' H_L(z) mu(dz), shape [r_c, r_c]."""

    state = tf.ones([1, 1], DTYPE)
    for axis, core in enumerate(prefix_cores):
        mass = _axis_mass(product_basis, axis, measure)
        state = tf.einsum("akb,AlB,kl,aA->bB", core.values, core.values, mass, state)
    return state


def prefix_gram_matrix_tangent(
    prefix_cores: Sequence[TTCore],
    dot_prefix_cores: Sequence[TTCore],
    product_basis: ProductBasis,
    *,
    measure: MassMeasure = MassMeasure.REFERENCE_MEASURE,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Return (P, dot P) by product rule through the prefix Gram chain."""

    state = tf.ones([1, 1], DTYPE)
    dot_state = tf.zeros_like(state)
    for axis, (core, dot_core) in enumerate(zip(prefix_cores, dot_prefix_cores)):
        mass = _axis_mass(product_basis, axis, measure)
        new_state = tf.einsum("akb,AlB,kl,aA->bB", core.values, core.values, mass, state)
        dot_state = (
            tf.einsum("akb,AlB,kl,aA->bB", dot_core.values, core.values, mass, state)
            + tf.einsum("akb,AlB,kl,aA->bB", core.values, dot_core.values, mass, state)
            + tf.einsum("akb,AlB,kl,aA->bB", core.values, core.values, mass, dot_state)
        )
        state = new_state
    return state, dot_state


@dataclass(frozen=True)
class RetainedQuadraticForm:
    """Exact suffix-marginal of a squared TT, typed by the reference measure.

    Fields follow UB-1 Section 1(V5): prefix cores of H_L over the retained
    axes, suffix Gram E, tuned tau with the reference-uniform defensive
    density (q0_ret_ref = 1, Z0_ref = 1), the single stored complete
    normalizer ``z_complete_ref = Z_h + tau``, the retained-axes product
    basis, and the coordinate map for the physical evaluator.
    """

    prefix_cores: tuple[TTCore, ...]
    suffix_gram: tf.Tensor
    tau: tf.Tensor
    z_complete_ref: tf.Tensor
    prefix_basis: ProductBasis
    coordinate_map: HighDimCoordinateMap
    mass_measure: MassMeasure = MassMeasure.REFERENCE_MEASURE

    def __post_init__(self) -> None:
        gram = tf.convert_to_tensor(self.suffix_gram, DTYPE)
        if gram.shape.rank != 2 or int(gram.shape[0]) != int(gram.shape[1]):
            raise ValueError("suffix_gram must be square")
        boundary_rank = int(self.prefix_cores[-1].right_rank)
        if int(gram.shape[0]) != boundary_rank:
            raise ValueError(
                "suffix_gram size must equal the prefix boundary rank "
                f"({int(gram.shape[0])} vs {boundary_rank})"
            )
        asymmetry = tf.reduce_max(tf.abs(gram - tf.transpose(gram)))
        scale = tf.maximum(tf.reduce_max(tf.abs(gram)), tf.constant(1.0, DTYPE))
        if bool((asymmetry > tf.constant(1e-12, DTYPE) * scale).numpy()):
            raise ValueError("suffix_gram must be symmetric (assert, not symmetrize)")
        tau = tf.convert_to_tensor(self.tau, DTYPE)
        if bool((tau < 0.0).numpy()):
            raise ValueError("tau must be nonnegative")
        z_complete = tf.convert_to_tensor(self.z_complete_ref, DTYPE)
        if not bool(tf.math.is_finite(z_complete).numpy()) or bool((z_complete <= 0.0).numpy()):
            raise ValueError("z_complete_ref must be finite and positive")
        object.__setattr__(self, "suffix_gram", gram)
        object.__setattr__(self, "tau", tau)
        object.__setattr__(self, "z_complete_ref", z_complete)

    @property
    def boundary_rank(self) -> int:
        return int(self.suffix_gram.shape[0])

    def suffix_gram_condition_estimate(self) -> tf.Tensor:
        eigenvalues = tf.linalg.eigvalsh(self.suffix_gram)
        floor = tf.constant(1e-300, DTYPE)
        return eigenvalues[-1] / tf.maximum(eigenvalues[0], floor)

    def _reference_log_weight_density(self) -> tf.Tensor:
        """log w_ref = -sum_i log(length_i): reference-measure Lebesgue density."""

        total = tf.constant(0.0, DTYPE)
        for basis in self.prefix_basis.bases:
            total = total - tf.math.log(basis.domain.length)
        return total

    def quadratic_form_values(self, reference_points: tf.Tensor) -> tf.Tensor:
        v = prefix_row_vectors(self.prefix_cores, self.prefix_basis, reference_points)
        return tf.einsum("na,ab,nb->n", v, self.suffix_gram, v)

    def evaluate_reference_density(self, reference_points: tf.Tensor) -> tf.Tensor:
        """p_ret_ref(z): density with respect to the normalized reference measure."""

        q = self.quadratic_form_values(reference_points)
        return (q + self.tau) / self.z_complete_ref

    def evaluate_physical_density(self, physical_points: tf.Tensor) -> tf.Tensor:
        """p_ret_phys(x) = p_ret_ref(R^{-1}(x)) * w_ref / J_R  (Lebesgue density)."""

        reference_points, log_abs_det_inverse = self.coordinate_map.inverse(
            tf.convert_to_tensor(physical_points, DTYPE)
        )
        # inverse() returns log|det D R^{-1}| = -log|det DR|.
        log_conversion = self._reference_log_weight_density() + log_abs_det_inverse
        return self.evaluate_reference_density(reference_points) * tf.exp(log_conversion)

    def reference_log_density_tangent(
        self,
        reference_points: tf.Tensor,
        dot_prefix_cores: Sequence[TTCore],
        dot_suffix_gram: tf.Tensor,
        dot_z_complete: tf.Tensor,
    ) -> tf.Tensor:
        """dot log p_ret_ref(z) for one parameter direction (UB-1 Sec. 2)."""

        v, dot_v = prefix_row_vectors_tangent(
            self.prefix_cores, dot_prefix_cores, self.prefix_basis, reference_points
        )
        q = tf.einsum("na,ab,nb->n", v, self.suffix_gram, v)
        dot_q = 2.0 * tf.einsum("na,ab,nb->n", v, self.suffix_gram, dot_v) + tf.einsum(
            "na,ab,nb->n", v, tf.convert_to_tensor(dot_suffix_gram, DTYPE), v
        )
        return dot_q / (q + self.tau) - tf.convert_to_tensor(dot_z_complete, DTYPE) / self.z_complete_ref


def retained_quadratic_form_from_squared_tt(
    cores: Sequence[TTCore],
    product_basis: ProductBasis,
    *,
    split_index: int,
    tau: float | tf.Tensor,
    prefix_basis: ProductBasis,
    coordinate_map: HighDimCoordinateMap,
    mass_measure: MassMeasure = MassMeasure.REFERENCE_MEASURE,
) -> RetainedQuadraticForm:
    """Build the exact retained marginal of ``h^2`` over the suffix axes.

    ``cores`` span the full adjacent block under ``product_basis``;
    axes ``[0, split_index)`` are retained (current state), axes
    ``[split_index, d)`` are integrated out (previous state).
    ``Zc_ref = Z_h + tau`` with the reference-uniform defensive density
    (Z0_ref = 1).
    """

    full = _check_cores(cores, len(product_basis.bases), "cores")
    if split_index < 1 or split_index >= len(full):
        raise ValueError("split_index must leave at least one axis on each side")
    prefix = full[:split_index]
    suffix = full[split_index:]
    gram = suffix_gram_matrix(
        suffix, product_basis, axis_offset=split_index, measure=mass_measure
    )
    prefix_gram = prefix_gram_matrix(prefix, product_basis, measure=mass_measure)
    z_h = tf.einsum("ab,ab->", prefix_gram, gram)
    tau_tensor = tf.convert_to_tensor(tau, DTYPE)
    return RetainedQuadraticForm(
        prefix_cores=prefix,
        suffix_gram=gram,
        tau=tau_tensor,
        z_complete_ref=z_h + tau_tensor,
        prefix_basis=prefix_basis,
        coordinate_map=coordinate_map,
        mass_measure=mass_measure,
    )


def retained_quadratic_form_tangent_from_squared_tt(
    cores: Sequence[TTCore],
    dot_cores: Sequence[TTCore],
    product_basis: ProductBasis,
    *,
    split_index: int,
    mass_measure: MassMeasure = MassMeasure.REFERENCE_MEASURE,
) -> tuple[tuple[TTCore, ...], tf.Tensor, tf.Tensor]:
    """Return (dot_prefix_cores, dot_E, dot_Zh) for one tangent direction.

    ``dot_Zc = dot_Zh`` because tau and Z0_ref are fixed (UB-1 Sec. 2).
    """

    full = _check_cores(cores, len(product_basis.bases), "cores")
    dots = _check_cores(dot_cores, len(product_basis.bases), "dot_cores")
    prefix, suffix = full[:split_index], full[split_index:]
    dot_prefix, dot_suffix = dots[:split_index], dots[split_index:]
    gram, dot_gram = suffix_gram_matrix_tangent(
        suffix, dot_suffix, product_basis, axis_offset=split_index, measure=mass_measure
    )
    prefix_gram, dot_prefix_gram = prefix_gram_matrix_tangent(
        prefix, dot_prefix, product_basis, measure=mass_measure
    )
    dot_z_h = tf.einsum("ab,ab->", dot_prefix_gram, gram) + tf.einsum(
        "ab,ab->", prefix_gram, dot_gram
    )
    return dot_prefix, dot_gram, dot_z_h


__all__ = [
    "RetainedQuadraticForm",
    "prefix_row_vectors",
    "prefix_row_vectors_tangent",
    "prefix_gram_matrix",
    "prefix_gram_matrix_tangent",
    "suffix_gram_matrix",
    "suffix_gram_matrix_tangent",
    "retained_quadratic_form_from_squared_tt",
    "retained_quadratic_form_tangent_from_squared_tt",
]
