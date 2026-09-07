"""Independent/reference tests for the C2 Gaussian-Hermite proposal."""

import math

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import eval_hermitenorm
import tensorflow as tf

from bayesfilter.highdim.c2_gaussian_hermite_proposal_tf import (
    GaussianHermiteRetainedProposal,
    normalized_hermite_incomplete_gram,
    retained_proposal_from_transition_snapshot,
    stateless_proposal_random_inputs,
)
from bayesfilter.highdim.squared_tt_engine_gaussian_xla_tf import (
    GaussianXLAFrozenTransitionSnapshot,
    GaussianXLARetainedProposalSnapshot,
)


DTYPE = tf.float64


def _reference_incomplete_gram(a: int, b: int, endpoint: float) -> float:
    normalization = math.sqrt(math.factorial(a) * math.factorial(b))

    def integrand(value: float) -> float:
        return (
            eval_hermitenorm(a, value)
            * eval_hermitenorm(b, value)
            * math.exp(-0.5 * value * value)
            / math.sqrt(2.0 * math.pi)
            / normalization
        )

    return quad(integrand, -np.inf, endpoint, epsabs=2e-13, epsrel=2e-13)[0]


def _rank_one_proposal(*, defensive_nu: float | None = 5.0):
    coefficients = tf.constant([1.0, 0.35, -0.15], DTYPE)
    gram = tf.constant([[1.7]], DTYPE)
    z_h = gram[0, 0] * tf.reduce_sum(tf.square(coefficients))
    return GaussianHermiteRetainedProposal(
        prefix_core_values=(tf.reshape(coefficients, [1, 3, 1]),),
        suffix_gram=gram,
        z_h=z_h,
        tau_abs=tf.constant(0.04, DTYPE),
        coordinate_offset=tf.constant([0.4], DTYPE),
        coordinate_matrix=tf.constant([[1.2]], DTYPE),
        defensive_nu=defensive_nu,
        time_index=1,
        source_snapshot_fingerprint="0" * 64,
    )


def test_incomplete_hermite_gram_matches_quadrature_through_degree_six() -> None:
    endpoints = (-2.1, -0.35, 0.0, 1.6)
    observed = normalized_hermite_incomplete_gram(
        tf.constant(endpoints, DTYPE), 6
    ).numpy()
    for endpoint_index, endpoint in enumerate(endpoints):
        for left_degree in range(7):
            for right_degree in range(7):
                expected = _reference_incomplete_gram(
                    left_degree, right_degree, endpoint
                )
                assert observed[endpoint_index, left_degree, right_degree] == pytest.approx(
                    expected, abs=3e-12, rel=3e-12
                )
    right_endpoint = normalized_hermite_incomplete_gram(
        tf.constant([24.0], DTYPE), 6
    )[0]
    tf.debugging.assert_near(right_endpoint, tf.eye(7, dtype=DTYPE), atol=2e-12)


def test_rank_one_conditional_cdf_is_monotone_and_density_normalizes() -> None:
    proposal = _rank_one_proposal()
    grid = tf.linspace(tf.constant(-8.0, DTYPE), tf.constant(8.0, DTYPE), 2001)
    incomplete = normalized_hermite_incomplete_gram(grid, proposal.degree)
    coefficients = tf.reshape(proposal.prefix_core_values[0], [3])
    cdf = tf.einsum("a,nab,b->n", coefficients, incomplete, coefficients)
    cdf = cdf / tf.reduce_sum(tf.square(coefficients))
    assert float(tf.reduce_min(cdf[1:] - cdf[:-1]).numpy()) >= -2e-14
    assert float(cdf[0].numpy()) < 1e-12
    assert float((1.0 - cdf[-1]).numpy()) < 1e-12

    coefficients_np = coefficients.numpy()
    z_h = float(proposal.z_h.numpy())
    tau = float(proposal.tau_abs.numpy())
    nu = float(proposal.defensive_nu)

    def density(value: float) -> float:
        hermite_values = np.array(
            [
                eval_hermitenorm(degree, value)
                / math.sqrt(math.factorial(degree))
                for degree in range(3)
            ]
        )
        polynomial = 1.7 * float(coefficients_np @ hermite_values) ** 2
        gaussian = math.exp(-0.5 * value * value) / math.sqrt(2.0 * math.pi)
        student = (
            math.gamma((nu + 1.0) / 2.0)
            / (math.gamma(nu / 2.0) * math.sqrt(nu * math.pi))
            * (1.0 + value * value / nu) ** (-(nu + 1.0) / 2.0)
        )
        return (polynomial * gaussian + tau * student) / (z_h + tau)

    integral = quad(density, -np.inf, np.inf, epsabs=2e-11, epsrel=2e-11)[0]
    assert integral == pytest.approx(1.0, abs=2e-10)


def test_kr_inverse_and_complete_mixture_density_eager_graph_parity() -> None:
    proposal = _rank_one_proposal()
    count = 128
    random_inputs = stateless_proposal_random_inputs(proposal, count, (713, 29))
    eager = proposal.sample_physical(*random_inputs)
    assert bool(eager["cdf_bracket_valid"].numpy())
    assert bool(eager["finite"].numpy())
    assert float(eager["maximum_inverse_cdf_residual"].numpy()) <= 2e-13
    assert float(eager["minimum_conditional_mass"].numpy()) > 0.0

    compiled = proposal.compiled_sampler(count, jit_compile=False)(*random_inputs)
    assert proposal.compiled_sampler(count, jit_compile=False) is proposal.compiled_sampler(
        count, jit_compile=False
    )
    for name in (
        "reference_points",
        "reference_log_density",
        "physical_points",
        "physical_log_density",
        "maximum_inverse_cdf_residual",
    ):
        tf.debugging.assert_near(compiled[name], eager[name], atol=2e-12, rtol=2e-12)

    direct_log_density = proposal.physical_log_density(eager["physical_points"])
    tf.debugging.assert_near(
        direct_log_density, eager["physical_log_density"], atol=2e-12, rtol=2e-12
    )
    reference_from_physical = (
        eager["physical_points"] - proposal.coordinate_offset[None, :]
    ) / proposal.coordinate_matrix[0, 0]
    tf.debugging.assert_near(
        proposal.reference_log_density(reference_from_physical)
        - tf.math.log(proposal.coordinate_matrix[0, 0]),
        eager["physical_log_density"],
        atol=2e-12,
        rtol=2e-12,
    )


def _synthetic_snapshot() -> GaussianXLAFrozenTransitionSnapshot:
    current = tf.constant([[[1.0], [0.25]]], DTYPE)
    branch = tf.constant([[[math.sqrt(0.6)], [math.sqrt(0.4)]]], DTYPE)
    previous = tf.constant([[[1.0], [0.0]]], DTYPE)
    z_h = tf.constant(1.0625, DTYPE)
    tau_relative = tf.constant(1e-4, DTYPE)
    corrected = tf.constant(-1.3, DTYPE)
    raw = corrected + tf.math.log1p(tau_relative)
    return GaussianXLAFrozenTransitionSnapshot(
        run_identity="synthetic-snapshot",
        time_index=1,
        state_dim=1,
        basis_degree=1,
        rank=1,
        row_count=4,
        sweeps=2,
        ridge=1e-10,
        configured_tau=1e-4,
        coordinate_half_width=3.0,
        config_seed=42,
        condition_number_veto=1e12,
        branch_gram_floor=1e-12,
        row_design="sobol",
        training_row_seed=(42, 101),
        defensive_nu=5.0,
        branch_count=2,
        basis_identity="hermite_reference_counting_branch_v1",
        mixed_shapes=((1, 2, 1), (1, 2, 1), (1, 2, 1)),
        prefix_values=(tf.constant([[[1.0], [0.0]]], DTYPE),),
        suffix_gram=tf.constant([[1.0]], DTYPE),
        tau_abs_previous=tf.constant(1e-4, DTYPE),
        z_complete_previous=tf.constant(1.0001, DTYPE),
        old_coordinate_offset=tf.constant([0.0], DTYPE),
        old_coordinate_matrix=tf.constant([[1.0]], DTYPE),
        joint_mean=tf.constant([0.3, -0.1], DTYPE),
        joint_chol=tf.constant([[1.2, 0.0], [0.2, 0.9]], DTYPE),
        observation=tf.constant([0.2], DTYPE),
        training_rows=tf.zeros([4, 2], DTYPE),
        training_weights=tf.ones([4], DTYPE) / 4.0,
        frozen_shift=tf.constant(-0.5, DTYPE),
        fitted_core_values=(current, branch, previous),
        z_h=z_h,
        raw_increment=raw,
        corrected_increment=corrected,
        worst_condition=tf.constant(10.0, DTYPE),
        weighted_fit_rms=tf.constant(0.01, DTYPE),
        u_old_max=tf.constant(1.0, DTYPE),
        target_summary={"all_finite": 1.0},
    )


def test_snapshot_constructor_recomputes_suffix_gram_tau_and_map() -> None:
    snapshot = _synthetic_snapshot()
    proposal = retained_proposal_from_transition_snapshot(snapshot)
    tf.debugging.assert_near(proposal.suffix_gram, tf.ones([1, 1], DTYPE), atol=1e-14)
    tf.debugging.assert_near(proposal.z_h, snapshot.z_h, atol=1e-14)
    tf.debugging.assert_near(
        proposal.tau_abs, snapshot.z_h * tf.constant(1e-4, DTYPE), atol=2e-14
    )
    tf.debugging.assert_near(proposal.coordinate_offset, snapshot.joint_mean[:1])
    tf.debugging.assert_near(proposal.coordinate_matrix, snapshot.joint_chol[:1, :1])
    assert proposal.time_index == snapshot.time_index
    assert len(proposal.proposal_id) == 64


def test_retained_snapshot_constructor_uses_captured_production_state() -> None:
    z_h = tf.constant(1.0625, DTYPE)
    tau_abs = tf.constant(1.0625e-4, DTYPE)
    corrected = tf.constant(-1.3, DTYPE)
    raw = corrected + tf.math.log1p(tau_abs / z_h)
    snapshot = GaussianXLARetainedProposalSnapshot(
        run_identity="synthetic-retained-snapshot",
        time_index=1,
        state_dim=1,
        basis_degree=1,
        rank=1,
        row_count=4,
        sweeps=2,
        ridge=1e-10,
        configured_tau=1e-4,
        coordinate_half_width=3.0,
        config_seed=42,
        row_design="sobol",
        defensive_nu=5.0,
        basis_identity="hermite_retained_quadratic_form_v1",
        prefix_core_values=(tf.constant([[[1.0], [0.25]]], DTYPE),),
        suffix_gram=tf.ones([1, 1], DTYPE),
        z_h=z_h,
        tau_abs=tau_abs,
        z_complete=z_h + tau_abs,
        coordinate_offset=tf.constant([0.3], DTYPE),
        coordinate_matrix=tf.constant([[1.2]], DTYPE),
        raw_increment=raw,
        corrected_increment=corrected,
    )
    proposal = retained_proposal_from_transition_snapshot(snapshot)
    tf.debugging.assert_equal(
        proposal.prefix_core_values[0], snapshot.prefix_core_values[0]
    )
    tf.debugging.assert_equal(proposal.suffix_gram, snapshot.suffix_gram)
    tf.debugging.assert_equal(proposal.tau_abs, snapshot.tau_abs)
    tf.debugging.assert_equal(
        proposal.coordinate_matrix, snapshot.coordinate_matrix
    )
    assert len(proposal.source_snapshot_fingerprint) == 64


def test_proposal_fails_closed_for_invalid_gram_and_floor() -> None:
    with pytest.raises(ValueError, match="positive semidefinite"):
        GaussianHermiteRetainedProposal(
            prefix_core_values=(tf.ones([1, 1, 1], DTYPE),),
            suffix_gram=tf.constant([[-1.0]], DTYPE),
            z_h=tf.constant(1.0, DTYPE),
            tau_abs=tf.constant(0.1, DTYPE),
            coordinate_offset=tf.zeros([1], DTYPE),
            coordinate_matrix=tf.eye(1, dtype=DTYPE),
            defensive_nu=None,
            time_index=1,
            source_snapshot_fingerprint="0" * 64,
        )
    with pytest.raises(ValueError, match="strictly positive"):
        GaussianHermiteRetainedProposal(
            prefix_core_values=(tf.ones([1, 1, 1], DTYPE),),
            suffix_gram=tf.ones([1, 1], DTYPE),
            z_h=tf.constant(1.0, DTYPE),
            tau_abs=tf.constant(0.0, DTYPE),
            coordinate_offset=tf.zeros([1], DTYPE),
            coordinate_matrix=tf.eye(1, dtype=DTYPE),
            defensive_nu=None,
            time_index=1,
            source_snapshot_fingerprint="0" * 64,
        )
