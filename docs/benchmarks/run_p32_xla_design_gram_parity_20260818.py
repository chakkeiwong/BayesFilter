"""P3.2 probe: XLA parity of design assembly + prefix/suffix Gram chains.

Scoping note: docs/plans/bayesfilter-p3-xla-port-scoping-note-2026-08-18.md.
Standalone diagnostic (no repo changes): wrap the repo's own
`FixedTTFitter._build_design_matrix`, `prefix_gram_matrix`, and
`suffix_gram_matrix` in tf.function(jit_compile=True) on realistic
mixed-basis fixtures (n=2 transition step shape: 2n+1 axes with the
discrete branch axis) and compare against eager at the 1e-12 gate.
Compilability failures (host syncs inside the wrapped code) surface as
tracing errors and are reported, not hidden.
"""
import os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
sys.path.insert(0, "/home/chakwong/BayesFilter")

import numpy as np, tensorflow as tf
from bayesfilter.highdim.fitting import FixedTTFitter
from bayesfilter.highdim.retained_quadratic_form_tf import (
    prefix_gram_matrix, suffix_gram_matrix,
)
from bayesfilter.highdim.squared_tt_engine_v0_tf import (
    DiscreteIndicatorBasis1D, _product_basis,
)
from bayesfilter.highdim.bases import ProductBasis
from bayesfilter.highdim.tt import TTCore

DT = tf.float64
rng = np.random.default_rng(11)
n, deg, rank, branch = 2, 10, 2, 3
cb = _product_basis(n, deg)
bd = int(cb.bases[0].basis_dim)
mixed = ProductBasis(
    list(cb.bases) + [DiscreteIndicatorBasis1D(branch)] + list(_product_basis(n, deg).bases),
    cb.convention,
)
dims = [bd] * n + [branch] + [bd] * n
cores = tuple(
    TTCore(tf.constant(
        rng.standard_normal([1 if a == 0 else rank, dims[a], 1 if a == 2 * n else rank]), DT))
    for a in range(2 * n + 1)
)
pts_np = rng.uniform(-1, 1, (300, 2 * n + 1))
pts_np[:, n] = rng.integers(0, branch, 300)  # discrete branch coordinate
pts = tf.constant(pts_np, DT)
fitter = FixedTTFitter()

worst = 0.0
failures = []
for idx in (0, n, 2 * n):  # first, branch, last axis
    ref = fitter._build_design_matrix(mixed, pts, cores, idx)
    try:
        jit = tf.function(
            lambda p: fitter._build_design_matrix(mixed, p, cores, idx), jit_compile=True
        )(pts)
        rel = float(tf.norm(jit - ref) / tf.maximum(tf.norm(ref), 1.0))
        worst = max(worst, rel)
        print(f"design core {idx}: xla rel {rel:.3e}", flush=True)
    except Exception as e:
        failures.append(("design", idx, str(e)[:200]))
        print(f"design core {idx}: XLA-COMPILE-FAIL {str(e)[:160]}", flush=True)

for name, fn, args in (
    ("prefix_gram", prefix_gram_matrix, (cores[:n], mixed)),
    ("suffix_gram", lambda c, b: suffix_gram_matrix(c, b, axis_offset=n), (cores[n:], mixed)),
):
    ref = fn(*args)
    try:
        jit = tf.function(lambda: fn(*args), jit_compile=True)()
        rel = float(tf.norm(jit - ref) / tf.maximum(tf.norm(ref), 1.0))
        worst = max(worst, rel)
        print(f"{name}: xla rel {rel:.3e}", flush=True)
    except Exception as e:
        failures.append((name, None, str(e)[:200]))
        print(f"{name}: XLA-COMPILE-FAIL {str(e)[:160]}", flush=True)

print(f"WORST rel: {worst:.3e}  gate 1e-12: {'PASS' if worst <= 1e-12 and not failures else 'FAIL'}", flush=True)
if failures:
    print(f"compile failures: {len(failures)} (see above)", flush=True)
