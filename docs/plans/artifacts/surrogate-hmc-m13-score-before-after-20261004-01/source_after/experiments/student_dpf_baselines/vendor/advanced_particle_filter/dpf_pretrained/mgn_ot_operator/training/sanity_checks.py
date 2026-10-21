"""
Phase 0: Architecture sanity checks (no training).

Run these BEFORE any training to catch bugs in the model code.

Usage:
    python -m training.sanity_checks
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tensorflow as tf

from models.operator import AmortizedOTOperator


def _make_operator(d=2, seed=0):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    op = AmortizedOTOperator(d=d)
    # Build by calling once
    B, N = 2, 100
    x = tf.random.normal((B, N, d))
    w = tf.nn.softmax(tf.random.normal((B, N)), axis=-1)
    log_eps = tf.fill((B,), tf.math.log(0.01))
    _ = op((x, w, log_eps))
    return op


def _make_cloud(B=2, N=100, d=2, seed=1):
    rng = np.random.default_rng(seed)
    x = tf.constant(rng.normal(size=(B, N, d)).astype(np.float32))
    w_raw = rng.random(size=(B, N)).astype(np.float32)
    w = w_raw / w_raw.sum(axis=-1, keepdims=True)
    w = tf.constant(w)
    log_eps = tf.constant(np.full(B, np.log(0.01), dtype=np.float32))
    return x, w, log_eps


# ---------------------------------------------------------------------------
# Check 1: identity recovery at init
# ---------------------------------------------------------------------------
def check_identity_at_init():
    """At init with V=I good-init, the MGN output is approximately
        y_tilde ~= x_tilde + (modular perturbation of order W_std)
    After denormalization: y ~= x + scale * (modular perturbation).
    We check that the output is *finite and bounded*, and that the weighted
    mean of y matches weighted mean of x (the module contributions average
    out to near zero by symmetry of the Gaussian init of W_k).
    """
    print("\n[Check 1] Operator is finite and respects weighted-mean at init")
    d = 2
    op = _make_operator(d=d)
    x, w, log_eps = _make_cloud(d=d)

    y = op((x, w, log_eps)).numpy()
    x_np = x.numpy()
    w_np = w.numpy()

    # Finite
    assert np.all(np.isfinite(y)), "non-finite output at init"

    # Bounded: no particle moved further than 100x its own norm (absurd if bug)
    max_ratio = np.max(np.linalg.norm(y - x_np, axis=-1)
                      / (np.linalg.norm(x_np, axis=-1) + 1e-8))
    print(f"  max |y - x| / |x| per particle = {max_ratio:.3f}")
    assert max_ratio < 100, f"unreasonably large displacement: {max_ratio}"

    # Weighted mean of y should be close to weighted mean of x
    # (since good-init W_k is symmetric, module contributions average out).
    mean_x = (w_np[:, :, None] * x_np).sum(axis=1)
    mean_y = (w_np[:, :, None] * y).sum(axis=1)
    mean_err = float(np.max(np.abs(mean_y - mean_x)))
    print(f"  |weighted_mean(y) - weighted_mean(x)| = {mean_err:.3e}")
    assert mean_err < 1.0, f"weighted mean shifted too much: {mean_err}"
    print("  OK")


# ---------------------------------------------------------------------------
# Check 2: Jacobian PSD
# ---------------------------------------------------------------------------
def check_jacobian_psd():
    """The Jacobian of MGN w.r.t. each particle's input should be PSD."""
    print("\n[Check 2] Jacobian PSD (via autograd on normalized MGN)")
    d = 3
    op = _make_operator(d=d)

    # Use the MGN module directly with the hypernet output for one cloud.
    rng = np.random.default_rng(42)
    x_cloud = tf.constant(rng.normal(size=(1, 20, d)).astype(np.float32))
    w_cloud = tf.constant(np.full((1, 20), 1.0 / 20, dtype=np.float32))
    log_eps = tf.constant([np.log(0.01)], dtype=tf.float32)

    # Extract the MGN map as a function of a single particle location.
    # We do this by calling the operator and differentiating y w.r.t. x.
    # Pick one particle: i = 0, batch 0.
    x_probe = tf.Variable(x_cloud.numpy().copy())
    with tf.GradientTape() as tape:
        y = op((x_probe, w_cloud, log_eps))  # (1, 20, d)
        y_particle = y[0, 0, :]              # (d,)
    jac = tape.jacobian(y_particle, x_probe)   # (d, 1, 20, d)
    jac = jac.numpy()[:, 0, 0, :]              # (d, d), block for particle 0

    sym = 0.5 * (jac + jac.T)
    eig = np.linalg.eigvalsh(sym)
    print(f"  eigenvalues of symmetric part: {eig}")
    assert np.min(eig) >= -1e-3, f"non-PSD: {eig}"
    print("  OK (note: this only checks the DIAGONAL BLOCK of the jacobian, which "
          "corresponds to the single-particle MGN jacobian once we factor out the "
          "fact that all particles in a cloud share theta)")


# ---------------------------------------------------------------------------
# Check 3: Permutation invariance (of the output as a multiset)
# ---------------------------------------------------------------------------
def check_permutation_invariance():
    """If we permute the input particles, the output multiset should be the same."""
    print("\n[Check 3] Permutation invariance of output multiset")
    d = 2
    op = _make_operator(d=d)

    rng = np.random.default_rng(7)
    N = 50
    x = rng.normal(size=(1, N, d)).astype(np.float32)
    w_raw = rng.random(N).astype(np.float32)
    w = (w_raw / w_raw.sum())[None, :]
    log_eps = np.array([np.log(0.01)], dtype=np.float32)

    perm = rng.permutation(N)
    x_perm = x[:, perm, :].copy()
    w_perm = w[:, perm].copy()

    y1 = op((tf.constant(x), tf.constant(w), tf.constant(log_eps))).numpy()[0]
    y2 = op((tf.constant(x_perm), tf.constant(w_perm), tf.constant(log_eps))).numpy()[0]

    # Compare as sets: sort by lex order.
    def sort_rows(a):
        return a[np.lexsort(a.T[::-1])]
    y1_sorted = sort_rows(y1)
    y2_sorted = sort_rows(y2)
    err = np.max(np.abs(y1_sorted - y2_sorted))
    print(f"  max abs deviation between sorted output multisets: {err:.3e}")
    assert err < 1e-3, f"not permutation invariant: {err}"
    print("  OK")


# ---------------------------------------------------------------------------
# Check 4: Translation equivariance
# ---------------------------------------------------------------------------
def check_translation_equivariance():
    """Operator applied to (x + t, w) should give y + t."""
    print("\n[Check 4] Translation equivariance")
    d = 2
    op = _make_operator(d=d)

    rng = np.random.default_rng(11)
    x = rng.normal(size=(1, 30, d)).astype(np.float32)
    w_raw = rng.random(30).astype(np.float32)
    w = (w_raw / w_raw.sum())[None, :]
    log_eps = np.array([np.log(0.01)], dtype=np.float32)

    t = rng.normal(size=(1, 1, d)).astype(np.float32) * 2.0
    y0 = op((tf.constant(x), tf.constant(w), tf.constant(log_eps))).numpy()
    y1 = op((tf.constant(x + t), tf.constant(w), tf.constant(log_eps))).numpy()

    err = np.max(np.abs(y1 - (y0 + t)))
    print(f"  max abs deviation: {err:.3e}")
    assert err < 1e-3, f"not translation equivariant: {err}"
    print("  OK")


# ---------------------------------------------------------------------------
# Check 5: Scale equivariance
# ---------------------------------------------------------------------------
def check_scale_equivariance():
    """Operator applied to (s*x, w) around centered origin should give s*y."""
    print("\n[Check 5] Scale equivariance (approximate)")
    d = 2
    op = _make_operator(d=d)

    rng = np.random.default_rng(13)
    # Use zero-mean cloud to make scaling semantics cleanest.
    x = rng.normal(size=(1, 40, d)).astype(np.float32)
    w = np.full((1, 40), 1.0 / 40, dtype=np.float32)
    x = x - (w[:, :, None] * x).sum(axis=1, keepdims=True)  # force zero mean
    log_eps = np.array([np.log(0.01)], dtype=np.float32)

    s = 3.0
    y0 = op((tf.constant(x), tf.constant(w), tf.constant(log_eps))).numpy()
    y1 = op((tf.constant(s * x), tf.constant(w), tf.constant(log_eps))).numpy()

    err = np.max(np.abs(y1 - s * y0))
    rel = err / (np.max(np.abs(s * y0)) + 1e-8)
    print(f"  max abs / relative deviation: {err:.3e} / {rel:.3e}")
    assert rel < 1e-3, f"not scale equivariant: rel error {rel}"
    print("  OK")


def run_all():
    print("=" * 60)
    print("Phase 0 sanity checks")
    print("=" * 60)
    check_identity_at_init()
    check_jacobian_psd()
    check_permutation_invariance()
    check_translation_equivariance()
    check_scale_equivariance()
    print("\n" + "=" * 60)
    print("All sanity checks passed.")
    print("=" * 60)


if __name__ == "__main__":
    run_all()
