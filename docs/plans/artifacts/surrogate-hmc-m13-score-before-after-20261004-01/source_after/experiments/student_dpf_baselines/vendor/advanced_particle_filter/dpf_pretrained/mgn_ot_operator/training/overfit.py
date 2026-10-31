"""
Phase 1: Overfitting sanity checks.

Before committing to full-scale training, verify the network can fit:
  1. A single cloud (sanity: capacity exists at all).
  2. 10 clouds (sanity: hypernet actually differentiates between clouds).

Targets are computed in NORMALIZED space (matches the operator's internal
frame and the precomputed dataset format). Loss is MSE in normalized space.

Default eps = 0.5 matches Corenflos et al. (2021)'s default DPF regularization.

If these don't work at default eps, full training won't either.

Usage (from a Colab notebook):

    from training.overfit import overfit_one, overfit_many
    overfit_one(n_steps=500)
    overfit_many(n_clouds=10, n_steps=2000)
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tensorflow as tf

from data.cloud_sampler import sample_cloud
from data.sinkhorn_targets import etpf_targets_normalized
from models.operator import AmortizedOTOperator


def _prepare_cloud_batch(n_clouds, N, d, eps, seed=0, sinkhorn_iters=1000):
    """Sample clouds, compute normalized-space Sinkhorn targets, stack tensors."""
    rng = np.random.default_rng(seed)
    xs = np.empty((n_clouds, N, d), dtype=np.float32)
    ws = np.empty((n_clouds, N), dtype=np.float32)
    for i in range(n_clouds):
        xs[i], ws[i] = sample_cloud(N, d, rng)
    eps_vec = tf.fill((n_clouds,), np.float32(eps))
    y_tilde_targets, _ = etpf_targets_normalized(
        tf.constant(xs), tf.constant(ws), eps_vec,
        n_iters=sinkhorn_iters,
    )
    log_eps = tf.fill((n_clouds,), np.float32(np.log(eps)))
    return (tf.constant(xs), tf.constant(ws), log_eps, y_tilde_targets)


def overfit_one(N=200, d=2, eps=0.5, n_steps=500, lr=1e-3, verbose_every=50):
    """Overfit to a single cloud. MSE (normalized space) should go to near 0."""
    print(f"\n[Phase 1a] Overfit single cloud  (eps={eps}, lr={lr}, n_steps={n_steps})")
    x, w, log_eps, y_tilde_tgt = _prepare_cloud_batch(1, N, d, eps, seed=0)
    op = AmortizedOTOperator(d=d)
    _ = op((x, w, log_eps))
    optimizer = tf.keras.optimizers.Adam(lr)

    @tf.function
    def step():
        with tf.GradientTape() as tape:
            y_tilde_pred, _, _ = op.forward_normalized((x, w, log_eps),
                                                       training=True)
            loss = tf.reduce_mean(tf.square(y_tilde_pred - y_tilde_tgt))
        grads = tape.gradient(loss, op.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 1.0)
        optimizer.apply_gradients(zip(grads, op.trainable_variables))
        return loss

    init_loss = None
    for s in range(n_steps):
        loss = float(step().numpy())
        if init_loss is None:
            init_loss = loss
        if s % verbose_every == 0 or s == n_steps - 1:
            print(f"  step {s:4d}: loss = {loss:.6e}")
    print(f"  ratio loss_final / loss_init: {loss / init_loss:.3e}")
    assert loss / init_loss < 1e-2, \
        f"Failed to overfit: final/init ratio = {loss/init_loss}"
    print("  OK")


def overfit_many(n_clouds=10, N=200, d=2, eps=0.5, n_steps=2000,
                 lr=1e-3, verbose_every=100):
    """Overfit to a small set of clouds."""
    print(f"\n[Phase 1b] Overfit {n_clouds} clouds  (eps={eps}, lr={lr}, n_steps={n_steps})")
    x, w, log_eps, y_tilde_tgt = _prepare_cloud_batch(n_clouds, N, d, eps, seed=1)
    op = AmortizedOTOperator(d=d)
    _ = op((x, w, log_eps))
    optimizer = tf.keras.optimizers.Adam(lr)

    @tf.function
    def step(x_b, w_b, e_b, y_b):
        with tf.GradientTape() as tape:
            y_tilde_pred, _, _ = op.forward_normalized((x_b, w_b, e_b),
                                                       training=True)
            loss = tf.reduce_mean(tf.square(y_tilde_pred - y_b))
        grads = tape.gradient(loss, op.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 1.0)
        optimizer.apply_gradients(zip(grads, op.trainable_variables))
        return loss

    init_loss = None
    for s in range(n_steps):
        loss = float(step(x, w, log_eps, y_tilde_tgt).numpy())
        if init_loss is None:
            init_loss = loss
        if s % verbose_every == 0 or s == n_steps - 1:
            print(f"  step {s:5d}: loss = {loss:.6e}")
    print(f"  ratio loss_final / loss_init: {loss / init_loss:.3e}")
    assert loss / init_loss < 5e-2, \
        f"Failed to overfit {n_clouds} clouds: final/init ratio = {loss/init_loss}"
    print("  OK")


if __name__ == "__main__":
    overfit_one(n_steps=500)
    overfit_many(n_clouds=10, n_steps=2000)
    print("\nPhase 1 complete.")


# ---------------------------------------------------------------------------
# Option B (CouplingOperator) overfit tests
# ---------------------------------------------------------------------------
def overfit_one_option_b(N=200, d=2, eps=0.5, n_steps=2000, lr=1e-3,
                         verbose_every=100):
    """Overfit the Option B CouplingOperator to a single cloud."""
    from models.operator_b import CouplingOperator
    print(f"\n[Option B - 1a] Overfit single cloud  "
          f"(eps={eps}, lr={lr}, n_steps={n_steps})")
    x, w, log_eps, y_tilde_tgt = _prepare_cloud_batch(1, N, d, eps, seed=0)
    op = CouplingOperator(d=d)
    _ = op((x, w, log_eps))          # build
    optimizer = tf.keras.optimizers.Adam(lr)

    @tf.function
    def step():
        with tf.GradientTape() as tape:
            y_tilde_pred, _, _, _ = op.forward_normalized(
                (x, w, log_eps), training=True)
            loss = tf.reduce_mean(tf.square(y_tilde_pred - y_tilde_tgt))
        grads = tape.gradient(loss, op.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 1.0)
        optimizer.apply_gradients(zip(grads, op.trainable_variables))
        return loss

    init_loss = None
    for s in range(n_steps):
        loss = float(step().numpy())
        if init_loss is None:
            init_loss = loss
        if s % verbose_every == 0 or s == n_steps - 1:
            print(f"  step {s:5d}: loss = {loss:.6e}")
    print(f"  ratio loss_final / loss_init: {loss / init_loss:.3e}")
    return loss


def overfit_many_option_b(n_clouds=10, N=200, d=2, eps=0.5,
                          n_steps=3000, lr=1e-3, verbose_every=150):
    """Overfit the Option B CouplingOperator to a batch of clouds."""
    from models.operator_b import CouplingOperator
    print(f"\n[Option B - 1b] Overfit {n_clouds} clouds  "
          f"(eps={eps}, lr={lr}, n_steps={n_steps})")
    x, w, log_eps, y_tilde_tgt = _prepare_cloud_batch(n_clouds, N, d, eps, seed=1)
    op = CouplingOperator(d=d)
    _ = op((x, w, log_eps))
    optimizer = tf.keras.optimizers.Adam(lr)

    @tf.function
    def step(x_b, w_b, e_b, y_b):
        with tf.GradientTape() as tape:
            y_tilde_pred, _, _, _ = op.forward_normalized(
                (x_b, w_b, e_b), training=True)
            loss = tf.reduce_mean(tf.square(y_tilde_pred - y_b))
        grads = tape.gradient(loss, op.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 1.0)
        optimizer.apply_gradients(zip(grads, op.trainable_variables))
        return loss

    init_loss = None
    for s in range(n_steps):
        loss = float(step(x, w, log_eps, y_tilde_tgt).numpy())
        if init_loss is None:
            init_loss = loss
        if s % verbose_every == 0 or s == n_steps - 1:
            print(f"  step {s:5d}: loss = {loss:.6e}")
    print(f"  ratio loss_final / loss_init: {loss / init_loss:.3e}")
    return loss
