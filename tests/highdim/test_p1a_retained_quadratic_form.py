"""P1A gate tests: RetainedQuadraticForm exact marginal, tangents, measures.

Binds UB-1 Sections 1(V5) and 2:
- U-MARG-TYPE-1: retained evaluator == brute-force suffix integration of
  h^2; rank>1 suffix Gram is represented as a quadratic form, never a
  scalar square.
- U-MARG-DERIV-1: (dot_prefix, dot_E, dot_Zh) tangents vs centered FD.
- U-MEASURE-1: reference/physical evaluator conversion identity and mass
  preservation under both, including the defensive component.
- U-TAU-1: complete-normalizer identity at tau > 0 and dot_Zc == dot_Zh.
"""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import tensorflow as tf

import bayesfilter.highdim as highdim
from bayesfilter.highdim.diagnostics import MassMeasure
from bayesfilter.highdim.retained_quadratic_form_tf import (
    RetainedQuadraticForm,
    prefix_row_vectors,
    retained_quadratic_form_from_squared_tt,
    retained_quadratic_form_tangent_from_squared_tt,
    suffix_gram_matrix,
)
from bayesfilter.highdim.tt import TTCore

DTYPE = tf.float64
SEED = 20260816


def _convention() -> highdim.MeasureConvention:
    return highdim.MeasureConvention(
        density_measure=highdim.DensityMeasure.REFERENCE_MEASURE,
        mass_measure=highdim.MassMeasure.REFERENCE_MEASURE,
        reference_weight_name="omega",
    )


def _product_basis(dimension: int, degree: int = 5) -> highdim.ProductBasis:
    return highdim.ProductBasis(
        [
            highdim.LegendreBasis1D(highdim.BoundedInterval(-1.0, 1.0), degree)
            for _ in range(dimension)
        ],
        _convention(),
    )


def _random_cores(
    dimension: int, degree: int, ranks: tuple[int, ...], seed: int
) -> tuple[TTCore, ...]:
    basis_dim = int(_product_basis(1, degree).bases[0].basis_dim)
    rng = np.random.default_rng(seed)
    cores = []
    for axis in range(dimension):
        shape = (ranks[axis], basis_dim, ranks[axis + 1])
        cores.append(TTCore(tf.constant(0.5 * rng.standard_normal(shape), DTYPE)))
    return tuple(cores)


def _evaluate_full_tt(cores, basis, points: tf.Tensor) -> tf.Tensor:
    state = tf.ones([tf.shape(points)[0], 1], DTYPE)
    for axis, core in enumerate(cores):
        basis_values = basis.evaluate_axis(axis, points[:, axis])
        state = tf.einsum("nl,lkr,nk->nr", state, core.values, basis_values)
    return state[:, 0]


def _gauss_nodes_weights(order: int) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = np.polynomial.legendre.leggauss(order)
    return nodes, weights


def _coordinate_map(dimension: int) -> highdim.AffineCoordinateMap:
    return highdim.AffineCoordinateMap(
        offset=tf.constant([0.3] * dimension, DTYPE),
        matrix=tf.constant(np.diag([2.0 + 0.5 * i for i in range(dimension)]), DTYPE),
    )


def _build(
    dimension: int = 3,
    split_index: int = 1,
    tau: float = 0.0,
    ranks: tuple[int, ...] = (1, 3, 2, 1),
    degree: int = 5,
    seed: int = SEED,
):
    basis = _product_basis(dimension, degree)
    cores = _random_cores(dimension, degree, ranks, seed)
    prefix_basis = _product_basis(split_index, degree)
    retained = retained_quadratic_form_from_squared_tt(
        cores,
        basis,
        split_index=split_index,
        tau=tau,
        prefix_basis=prefix_basis,
        coordinate_map=_coordinate_map(split_index),
    )
    return basis, cores, retained


def test_u_marg_type_1_evaluator_matches_brute_force_and_gram_has_rank_gt_one() -> None:
    dimension, split_index = 3, 1
    basis, cores, retained = _build(dimension, split_index, tau=0.0)

    # Suffix Gram must genuinely have rank > 1 for this fixture, and the
    # object must expose it as a matrix (quadratic form), never a scalar
    # square (program veto V10).
    eigenvalues = np.linalg.eigvalsh(retained.suffix_gram.numpy())
    assert (eigenvalues > 1e-10).sum() > 1
    assert retained.suffix_gram.shape.rank == 2
    assert retained.boundary_rank == 3

    # Brute-force: integrate h(z_pref, u)^2 over the suffix block under the
    # normalized reference measure with dense Gauss-Legendre quadrature.
    order = 24
    nodes, weights = _gauss_nodes_weights(order)
    suffix_dim = dimension - split_index
    mesh = np.meshgrid(*([nodes] * suffix_dim), indexing="ij")
    suffix_points = np.stack([m.reshape(-1) for m in mesh], axis=1)
    weight_mesh = np.meshgrid(*([weights / 2.0] * suffix_dim), indexing="ij")
    suffix_weights = np.prod(np.stack([w.reshape(-1) for w in weight_mesh], axis=1), axis=1)

    rng = np.random.default_rng(7)
    prefix_points = rng.uniform(-1.0, 1.0, size=(11, split_index))
    for row in prefix_points:
        tiled = np.concatenate(
            [np.tile(row, (suffix_points.shape[0], 1)), suffix_points], axis=1
        )
        h_values = _evaluate_full_tt(cores, basis, tf.constant(tiled, DTYPE)).numpy()
        brute = float(np.sum(suffix_weights * h_values**2))
        unnormalized = float(
            retained.quadratic_form_values(tf.constant(row[None, :], DTYPE)).numpy()[0]
        )
        assert abs(unnormalized - brute) <= 1e-10 * max(1.0, abs(brute))

    # Normalizer Zc = Z_h at tau=0 must equal the full-block Gram integral.
    full_dim_mesh = np.meshgrid(*([nodes] * dimension), indexing="ij")
    full_points = np.stack([m.reshape(-1) for m in full_dim_mesh], axis=1)
    full_weight_mesh = np.meshgrid(*([weights / 2.0] * dimension), indexing="ij")
    full_weights = np.prod(
        np.stack([w.reshape(-1) for w in full_weight_mesh], axis=1), axis=1
    )
    h_full = _evaluate_full_tt(cores, basis, tf.constant(full_points, DTYPE)).numpy()
    z_h_brute = float(np.sum(full_weights * h_full**2))
    assert abs(float(retained.z_complete_ref.numpy()) - z_h_brute) <= 1e-10 * z_h_brute


def test_u_marg_deriv_1_tangents_match_centered_finite_differences() -> None:
    dimension, split_index, degree = 3, 2, 5
    ranks = (1, 2, 3, 1)
    basis = _product_basis(dimension, degree)
    cores = _random_cores(dimension, degree, ranks, SEED + 1)
    dot_cores = _random_cores(dimension, degree, ranks, SEED + 2)
    prefix_basis = _product_basis(split_index, degree)
    coordinate_map = _coordinate_map(split_index)

    def build(eps: float) -> RetainedQuadraticForm:
        perturbed = tuple(
            TTCore(core.values + eps * dot.values) for core, dot in zip(cores, dot_cores)
        )
        return retained_quadratic_form_from_squared_tt(
            perturbed,
            basis,
            split_index=split_index,
            tau=0.25,
            prefix_basis=prefix_basis,
            coordinate_map=coordinate_map,
        )

    retained = build(0.0)
    dot_prefix, dot_gram, dot_z_h = retained_quadratic_form_tangent_from_squared_tt(
        cores, dot_cores, basis, split_index=split_index
    )

    step = 1e-6
    plus, minus = build(step), build(-step)

    fd_gram = (plus.suffix_gram - minus.suffix_gram) / (2.0 * step)
    assert float(tf.reduce_max(tf.abs(dot_gram - fd_gram)).numpy()) <= 1e-6

    fd_z = float(((plus.z_complete_ref - minus.z_complete_ref) / (2.0 * step)).numpy())
    assert abs(float(dot_z_h.numpy()) - fd_z) <= 1e-6 * max(1.0, abs(fd_z))

    rng = np.random.default_rng(11)
    points = tf.constant(rng.uniform(-1.0, 1.0, size=(9, split_index)), DTYPE)
    analytic = retained.reference_log_density_tangent(
        points, dot_prefix, dot_gram, dot_z_h
    ).numpy()
    fd_log = (
        tf.math.log(plus.evaluate_reference_density(points))
        - tf.math.log(minus.evaluate_reference_density(points))
    ).numpy() / (2.0 * step)
    assert np.max(np.abs(analytic - fd_log)) <= 1e-5


def test_u_measure_1_conversion_identity_and_mass_preservation() -> None:
    dimension, split_index = 3, 2
    _basis, _cores, retained = _build(dimension, split_index, tau=0.5, ranks=(1, 2, 3, 1))

    order = 30
    nodes, weights = _gauss_nodes_weights(order)
    mesh = np.meshgrid(*([nodes] * split_index), indexing="ij")
    z_points = tf.constant(
        np.stack([m.reshape(-1) for m in mesh], axis=1), DTYPE
    )
    reference_weights = np.prod(
        np.stack(
            [w.reshape(-1) for w in np.meshgrid(*([weights / 2.0] * split_index), indexing="ij")],
            axis=1,
        ),
        axis=1,
    )
    lebesgue_weights = reference_weights * (2.0**split_index)

    # Mass under the reference measure: integral p_ref d mu == 1.
    p_ref = retained.evaluate_reference_density(z_points).numpy()
    assert abs(float(np.sum(reference_weights * p_ref)) - 1.0) <= 1e-10

    # Conversion identity at the same points, and physical mass == 1:
    # integral p_phys dx = sum w_leb * J_R * p_phys(R(z)).
    physical_points, log_abs_det = retained.coordinate_map.forward(z_points)
    p_phys = retained.evaluate_physical_density(physical_points).numpy()
    w_ref_density = np.exp(
        sum(-np.log(2.0) for _ in range(split_index))
    )  # w_ref = prod 1/length_i, lengths are 2
    expected_phys = p_ref * w_ref_density / np.exp(log_abs_det.numpy())
    assert np.max(np.abs(p_phys - expected_phys)) <= 1e-12 * max(1.0, np.max(np.abs(p_phys)))
    physical_mass = float(
        np.sum(lebesgue_weights * np.exp(log_abs_det.numpy()) * p_phys)
    )
    assert abs(physical_mass - 1.0) <= 1e-10


def test_u_tau_1_complete_normalizer_identity_and_tangent() -> None:
    dimension, split_index = 2, 1
    basis = _product_basis(dimension)
    cores = _random_cores(dimension, 5, (1, 3, 1), SEED + 3)
    dot_cores = _random_cores(dimension, 5, (1, 3, 1), SEED + 4)
    prefix_basis = _product_basis(split_index)
    coordinate_map = _coordinate_map(split_index)

    tau = 0.35
    retained = retained_quadratic_form_from_squared_tt(
        cores,
        basis,
        split_index=split_index,
        tau=tau,
        prefix_basis=prefix_basis,
        coordinate_map=coordinate_map,
    )
    baseline = retained_quadratic_form_from_squared_tt(
        cores,
        basis,
        split_index=split_index,
        tau=0.0,
        prefix_basis=prefix_basis,
        coordinate_map=coordinate_map,
    )
    # Complete normalizer: Zc(tau) = Z_h + tau * Z0_ref with Z0_ref = 1.
    assert abs(
        float((retained.z_complete_ref - baseline.z_complete_ref - tau).numpy())
    ) <= 1e-14

    # Evaluator uses the COMPLETE normalizer: p_ref = (q + tau)/Zc.
    rng = np.random.default_rng(3)
    points = tf.constant(rng.uniform(-1.0, 1.0, size=(7, split_index)), DTYPE)
    q = retained.quadratic_form_values(points)
    expected = (q + tau) / retained.z_complete_ref
    assert float(
        tf.reduce_max(tf.abs(retained.evaluate_reference_density(points) - expected)).numpy()
    ) <= 1e-15

    # Mass still 1 at tau > 0 (defensive component included).
    order = 40
    nodes, weights = _gauss_nodes_weights(order)
    grid = tf.constant(nodes[:, None], DTYPE)
    mass = float(np.sum((weights / 2.0) * retained.evaluate_reference_density(grid).numpy()))
    assert abs(mass - 1.0) <= 1e-10

    # dot_Zc == dot_Zh (tau, Z0 fixed): tangent of Zc has no tau term.
    _dp, _dg, dot_z_h = retained_quadratic_form_tangent_from_squared_tt(
        cores, dot_cores, basis, split_index=split_index
    )
    step = 1e-6
    def z_complete(eps: float) -> float:
        perturbed = tuple(
            TTCore(core.values + eps * dot.values) for core, dot in zip(cores, dot_cores)
        )
        return float(
            retained_quadratic_form_from_squared_tt(
                perturbed,
                basis,
                split_index=split_index,
                tau=tau,
                prefix_basis=prefix_basis,
                coordinate_map=coordinate_map,
            ).z_complete_ref.numpy()
        )

    fd = (z_complete(step) - z_complete(-step)) / (2.0 * step)
    assert abs(float(dot_z_h.numpy()) - fd) <= 1e-6 * max(1.0, abs(fd))


def test_symmetry_is_asserted_not_silently_symmetrized() -> None:
    basis = _product_basis(1)
    cores = _random_cores(1, 5, (1, 2, 1), SEED)
    bad_gram = tf.constant([[1.0, 0.5], [0.2, 1.0]], DTYPE)
    try:
        RetainedQuadraticForm(
            prefix_cores=(TTCore(tf.ones([1, 5, 2], DTYPE)),),
            suffix_gram=bad_gram,
            tau=tf.constant(0.0, DTYPE),
            z_complete_ref=tf.constant(1.0, DTYPE),
            prefix_basis=basis,
            coordinate_map=_coordinate_map(1),
        )
    except ValueError as error:
        assert "symmetric" in str(error)
    else:
        raise AssertionError("asymmetric suffix Gram must be rejected")
