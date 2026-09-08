"""U-MAP-MOM-1: exact retained moments vs dense quadrature reference.

Design note (adapted maps, 2026-08-20) Section 5 rung 1. The lemma's
Gram-chain moments must match brute-force tensor-quadrature moments of
the same retained density at n in {1, 2} to near machine precision
(both are exact integrals of polynomials; disagreement means the chain
or the M^(p) constants are wrong, not resolution).
"""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
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
