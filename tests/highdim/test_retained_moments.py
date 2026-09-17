"""U-MAP-MOM-1: exact retained moments vs dense quadrature reference.

Design note (adapted maps, 2026-08-20) Section 5 rung 1. The lemma's
Gram-chain moments must match brute-force tensor-quadrature moments of
the same retained density at n in {1, 2} to near machine precision
(both are exact integrals of polynomials; disagreement means the chain
or the M^(p) constants are wrong, not resolution).
"""

from __future__ import annotations

import os
from dataclasses import replace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.highdim.filtering import AffineCoordinateMap
from bayesfilter.highdim.retained_moments_tf import retained_reference_moments
from bayesfilter.highdim.retained_quadratic_form_tf import (
    RetainedQuadraticForm,
    retained_quadratic_form_from_squared_tt,
)
from bayesfilter.highdim.squared_tt_engine_v0_tf import _product_basis
from bayesfilter.highdim.tt import TTCore

DTYPE = tf.float64


def _random_retained(n: int, rank: int, deg: int, tau: float, seed: int) -> RetainedQuadraticForm:
    rng = np.random.default_rng(seed)
    basis = _product_basis(n + 1, deg)
    dims = [int(basis.bases[0].basis_dim)] * (n + 1)
    cores = tuple(
        TTCore(tf.constant(
            0.5 * rng.standard_normal(
                [1 if a == 0 else rank, dims[a], 1 if a == n else rank]
            ), DTYPE))
        for a in range(n + 1)
    )
    base = retained_quadratic_form_from_squared_tt(
        cores, basis, split_index=n, tau=0.0,
        prefix_basis=_product_basis(n, deg),
        coordinate_map=AffineCoordinateMap(
            offset=tf.zeros([n], DTYPE), matrix=tf.eye(n, dtype=DTYPE)
        ),
    )
    z_h = base.z_complete_ref
    return RetainedQuadraticForm(
        prefix_cores=base.prefix_cores, suffix_gram=base.suffix_gram,
        tau=tf.constant(tau, DTYPE) * z_h, z_complete_ref=(1.0 + tau) * z_h,
        prefix_basis=base.prefix_basis, coordinate_map=base.coordinate_map,
    )


def _dense_reference_moments(retained: RetainedQuadraticForm, order: int):
    n = len(retained.prefix_cores)
    nodes, weights = np.polynomial.legendre.leggauss(order)
    mesh = np.meshgrid(*([nodes] * n), indexing="ij")
    points = np.stack([m.reshape(-1) for m in mesh], axis=1)
    wmesh = np.meshgrid(*([weights / 2.0] * n), indexing="ij")
    w = np.prod(np.stack([m.reshape(-1) for m in wmesh], axis=1), axis=1)
    density = retained.evaluate_reference_density(tf.constant(points, DTYPE)).numpy()
    mass = float((w * density).sum())
    mean = (w * density)[:, None] * points
    mean = mean.sum(axis=0)
    centered = points - mean[None, :]
    cov = np.einsum("n,ni,nj->ij", w * density, centered, centered)
    return mass, mean, cov


def test_u_map_mom_1_moments_match_dense_reference() -> None:
    for n, rank, deg, tau, seed in ((1, 3, 8, 1e-4, 5), (2, 3, 6, 1e-3, 7), (2, 2, 8, 0.0, 11)):
        retained = _random_retained(n, rank, deg, tau, seed)
        mean, cov = retained_reference_moments(retained)
        mass, ref_mean, ref_cov = _dense_reference_moments(retained, deg + 4)
        assert abs(mass - 1.0) <= 1e-10, f"reference density mass {mass}"
        mean_err = float(np.max(np.abs(mean.numpy() - ref_mean)))
        cov_err = float(np.max(np.abs(cov.numpy() - ref_cov)))
        assert mean_err <= 1e-10, f"n={n} tau={tau}: mean err {mean_err}"
        assert cov_err <= 1e-10, f"n={n} tau={tau}: cov err {cov_err}"


def test_u_map_mom_1_covariance_psd() -> None:
    retained = _random_retained(2, 3, 6, 1e-3, 13)
    _mean, cov = retained_reference_moments(retained)
    eig = np.linalg.eigvalsh(cov.numpy())
    assert eig[0] > 0.0, f"covariance not PD: {eig}"


@pytest.mark.parametrize("jit", [False, True])
@pytest.mark.parametrize("mixed_degrees", [False, True])
def test_public_moments_compile_complete_result_and_reuse_live_inputs(jit, mixed_degrees):
    from bayesfilter.highdim.bases import LegendreBasis1D, ProductBasis
    from bayesfilter.highdim.retained_moments_tf import retained_moment_program
    from bayesfilter.highdim.retained_quadratic_form_tf import prefix_gram_matrix

    initial = _random_retained(2, 2, 3, 1e-3, 17)
    if mixed_degrees:
        first_basis, second_basis = initial.prefix_basis.bases
        basis = ProductBasis((LegendreBasis1D(first_basis.domain, 1), second_basis),
                             initial.prefix_basis.convention)
        cores = (TTCore(initial.prefix_cores[0].values[:, :2]), initial.prefix_cores[1])
        normalizer = tf.reduce_sum(prefix_gram_matrix(cores, basis) * initial.suffix_gram) + initial.tau
        initial = replace(initial, prefix_basis=basis, prefix_cores=cores,
                          z_complete_ref=normalizer)
    program, inputs = retained_moment_program(initial, jit_compile=jit)
    first = retained_reference_moments(initial, jit_compile=jit)
    cores = tuple(TTCore(core.values + 0.03) for core in initial.prefix_cores)
    gram = initial.suffix_gram * 1.1
    tau = initial.tau * 1.2
    normalizer = tf.reduce_sum(prefix_gram_matrix(cores, initial.prefix_basis) * gram) + tau
    changed = replace(initial, prefix_cores=cores, suffix_gram=gram,
                      tau=tau, z_complete_ref=normalizer)
    same_program, _ = retained_moment_program(changed, jit_compile=jit)
    assert same_program is program
    actual = retained_reference_moments(changed, jit_compile=jit)
    mass, mean, covariance = _dense_reference_moments(changed, 8)
    np.testing.assert_allclose(mass, 1.0, atol=1e-10)
    np.testing.assert_allclose(actual[0], mean, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(actual[1], covariance, atol=1e-10, rtol=1e-10)
    assert float(tf.reduce_max(tf.abs(first[0] - actual[0]))) > 1e-5
    assert program.experimental_get_tracing_count() == 1
    assert any(value.shape == (2, 3, 4, 4)
               for value in program.get_concrete_function().captured_inputs)
    if jit:
        assert "HloModule" in program.experimental_get_compiler_ir(*inputs)(stage="hlo")
