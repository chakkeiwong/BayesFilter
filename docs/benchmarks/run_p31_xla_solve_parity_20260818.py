"""P3.1 probe: XLA (jit_compile) parity of the scaled augmented solve kernel.

Scoping note: docs/plans/bayesfilter-p3-xla-port-scoping-note-2026-08-18.md.
Standalone diagnostic (no repo changes): re-express the mathematical core of
`_solve_scaled_augmented_ridge` (column scales -> scaled augmented QR ->
triangular solves -> unscale) as a compilable function; compare eager vs
jit_compile=True on frozen random fixtures spanning benign and
ill-conditioned regimes. Gate: parity 1e-12 (plan P3). Also counts
retraces across repeated same-shape calls.
"""
import os, sys, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
sys.path.insert(0, "/home/chakwong/BayesFilter")

import numpy as np, tensorflow as tf
from bayesfilter.highdim.fitting import (
    _solve_scaled_augmented_ridge,
    _weighted_column_scales,
    _DEFAULT_COLUMN_SCALE_FLOOR,
)

DT = tf.float64


def solve_kernel(design, weights, target, ridge):
    """Compilable core: mirrors _solve_scaled_augmented_ridge's math."""
    scales, _n, _f = _weighted_column_scales(design, weights, _DEFAULT_COLUMN_SCALE_FLOOR)
    scaled = design / scales[None, :]
    sw = tf.sqrt(weights)
    n_cols = tf.shape(design)[1]
    augmented = tf.concat(
        [scaled * sw[:, None], tf.sqrt(ridge) * tf.eye(n_cols, dtype=DT) / scales[None, :]],
        axis=0,
    )
    rhs = tf.concat([target * sw, tf.zeros([n_cols], DT)], axis=0)
    q, r_factor = tf.linalg.qr(augmented)
    y = tf.linalg.matvec(q, rhs, transpose_a=True)
    z = tf.linalg.triangular_solve(r_factor, y[:, None], lower=False)[:, 0]
    return z / scales


solve_jit = tf.function(solve_kernel, jit_compile=True)
solve_eager = solve_kernel

rng = np.random.default_rng(7)
worst = 0.0
for trial, (rows, cols, cond_scale) in enumerate([
    (200, 30, 1.0), (200, 30, 1e-6), (500, 64, 1.0), (500, 64, 1e-8),
    (1000, 121, 1.0), (1000, 121, 1e-7),
]):
    a = rng.standard_normal((rows, cols))
    a[:, -3:] *= cond_scale  # near-degenerate columns for the ill-conditioned arms
    w = rng.uniform(0.5, 1.5, rows)
    g = rng.standard_normal(rows)
    args = (tf.constant(a, DT), tf.constant(w, DT), tf.constant(g, DT), tf.constant(1e-10, DT))
    ref = _solve_scaled_augmented_ridge(
        design=args[0], target_values=args[2], weights=args[1], ridge=1e-10
    ).solution
    for name, fn in (("eager-kernel", solve_eager), ("xla-kernel", solve_jit)):
        got = fn(*args)
        rel = float(tf.norm(got - ref) / tf.maximum(tf.norm(ref), 1.0))
        worst = max(worst, rel)
        print(f"trial {trial} ({rows}x{cols}, colscale {cond_scale:.0e}) {name}: rel vs repo solver {rel:.3e}", flush=True)

# retrace count across repeated same-shape calls
traced = solve_jit.experimental_get_tracing_count()
for _ in range(5):
    a = rng.standard_normal((500, 64))
    solve_jit(tf.constant(a, DT), tf.constant(rng.uniform(0.5, 1.5, 500), DT),
              tf.constant(rng.standard_normal(500), DT), tf.constant(1e-10, DT))
print(f"retraces for 5 same-shape calls: {solve_jit.experimental_get_tracing_count() - traced}", flush=True)
print(f"WORST rel: {worst:.3e}  gate 1e-12: {'PASS' if worst <= 1e-12 else 'FAIL'}", flush=True)
