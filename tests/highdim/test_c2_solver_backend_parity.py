"""U-SOLVE-PARITY: eager vs XLA weighted-ridge solver backend equivalence.

Gate-A attribution evidence (2026-08-25): the two backends must agree on
a single stress-conditioned solve under BOTH uniform and non-uniform
(Christoffel half-mixture) weights. Measured 7.9e-14 / 3.4e-14 relative;
gated at 1e-10. This is the backend-equivalence unit the end-to-end
swamp-regime lane comparison cannot provide (ALS amplifies rounding to
its convergence floor there — see the Gate A record). CPU-only
diagnostic; GPU intentionally hidden.
"""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import tensorflow as tf

from bayesfilter.highdim.fitting import _solve_scaled_augmented_ridge
from bayesfilter.highdim.squared_tt_engine_gaussian_tf import _christoffel_rows
from bayesfilter.highdim.squared_tt_engine_v0_tf import EngineConfig
from bayesfilter.highdim.squared_tt_engine_xla_tf import _solve_scaled_qr

DTYPE = tf.float64


def _stress_system():
    rng = np.random.default_rng(3)
    design = tf.constant(rng.standard_normal((14336, 468)), DTYPE)
    design = design * tf.constant(
        np.exp(rng.uniform(-6, 0, 468)), DTYPE
    )[None, :]
    target = tf.linalg.matvec(
        design, tf.constant(rng.standard_normal(468), DTYPE)
    ) + 1e-6 * tf.constant(rng.standard_normal(14336), DTYPE)
    return design, target


def _relative_gap(design, weights, target) -> float:
    eager = _solve_scaled_augmented_ridge(
        design=design, target_values=target, weights=weights, ridge=1e-10
    )
    xla = _solve_scaled_qr(design, weights, target, tf.constant(1e-10, DTYPE))
    a = tf.reshape(getattr(eager, "solution", eager), [-1])
    b = tf.reshape(xla[0] if isinstance(xla, tuple) else xla, [-1])
    return float(tf.norm(a - b) / tf.maximum(tf.norm(a), 1e-300))


def test_solver_backends_agree_under_christoffel_weights() -> None:
    design, target = _stress_system()
    config = EngineConfig(
        basis_degree=12, rank=6, row_count=2048, sweeps=8, ridge=1e-10,
        tau=1e-6, coordinate_half_width=3.0, seed=93026, row_design="sobol",
    )
    _rows, w, _ess = _christoffel_rows(config, 2048, 4, (93026, 101), 12)
    w7 = tf.reshape(tf.repeat(w, 7, axis=0), [-1])
    assert _relative_gap(design, w7, target) < 1e-10


def test_solver_backends_agree_under_uniform_weights() -> None:
    design, target = _stress_system()
    uniform = tf.fill([14336], tf.constant(1.0 / 14336, DTYPE))
    assert _relative_gap(design, uniform, target) < 1e-10
