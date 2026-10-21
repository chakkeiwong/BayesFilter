"""
Single question: can M-MGN at (K, h) fit a single (x_tilde, y_tilde_target) pair
to near-zero MSE with unconstrained optimization?

No hypernet. No encoder. No identity-based init tricks. Just:
  theta = trainable variable
  y_pred = mmgn_forward(x_tilde, *unpack(theta))
  loss = MSE(y_pred, y_tilde_target)
  Adam, train until flat.

Final loss < 1e-5:  MGN has enough capacity; bottleneck is elsewhere (hypernet).
Final loss >> 1e-5: MGN at this (K, h) cannot express this target.

Usage:
    from training.diagnose_overfit import fit_mgn_to_target
    fit_mgn_to_target(K=4, h=32)       # current config
    fit_mgn_to_target(K=8, h=64)
    fit_mgn_to_target(K=16, h=128)
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tensorflow as tf

from data.cloud_sampler import sample_cloud
from data.sinkhorn_targets import etpf_targets_normalized, weighted_normalize
from models.mgn import mmgn_forward, unpack_params, param_count


def fit_mgn_to_target(N=200, d=2, K=4, h=32, eps=0.5,
                     n_steps=20000, lr=1e-3,
                     sinkhorn_iters=1000, seed=0, verbose_every=1000):
    """Fit a single MGN parameter vector to a single target cloud.

    Initialization: Xavier-scaled random.
    Optimizer: Adam with lr=1e-3. (Higher than phase 1's 1e-4; single cloud
    is well-conditioned enough.)

    Returns:
        final_loss (float)
    """
    P = param_count(d, K, h)
    print(f"\nK={K}, h={h}  ({P} params)")

    # Target.
    rng = np.random.default_rng(seed)
    x_np, w_np = sample_cloud(N, d, rng)
    x = tf.constant(x_np[None, :, :])
    w = tf.constant(w_np[None, :])
    eps_t = tf.constant([eps], dtype=tf.float32)
    y_tilde_target, _ = etpf_targets_normalized(
        x, w, eps_t, n_iters=sinkhorn_iters)
    x_tilde, _, _ = weighted_normalize(x, w)

    # Scaled init.
    # V uses standard Xavier (V^T V ~ I, well-behaved).
    # W uses cubic-root scaling: module output ~ h^2 * W_std^3 * sqrt(d),
    # so W_std = 1 / (h^2 * sqrt(d))^(1/3) gives O(1) module contribution.
    rng_tf = np.random.default_rng(seed)
    V_std = 1.0 / np.sqrt(d)
    W_std = 1.0 / (h * h * np.sqrt(d)) ** (1.0 / 3.0)
    V0 = rng_tf.normal(0, V_std, size=(d, d))
    a0 = np.zeros(d)
    W0 = rng_tf.normal(0, W_std, size=(K, h, d))
    b0 = np.zeros((K, h))
    theta_init = np.concatenate([
        V0.flatten(), a0.flatten(), W0.flatten(), b0.flatten()
    ]).reshape(1, -1).astype(np.float32)

    theta = tf.Variable(theta_init)
    opt = tf.keras.optimizers.Adam(lr)

    @tf.function
    def step():
        with tf.GradientTape() as tape:
            V, a, W, b = unpack_params(theta, d, K, h)
            y_pred = mmgn_forward(x_tilde, V, a, W, b)
            loss = tf.reduce_mean(tf.square(y_pred - y_tilde_target))
        grads = tape.gradient(loss, [theta])
        opt.apply_gradients(zip(grads, [theta]))
        return loss

    for s in range(n_steps):
        loss_val = float(step().numpy())
        if s % verbose_every == 0 or s == n_steps - 1:
            print(f"  step {s:6d}: loss = {loss_val:.4e}")

    print(f"  final: {loss_val:.4e}")
    return loss_val


def run_capacity_sweep(n_steps=20000):
    """Fit MGN to the same target at three capacity levels."""
    results = {}
    for K, h in [(4, 32), (8, 64), (16, 128)]:
        final = fit_mgn_to_target(K=K, h=h, n_steps=n_steps)
        results[(K, h)] = final

    print("\n" + "=" * 40)
    print(f"{'Config':15s}  {'Final loss':>14s}")
    print("=" * 40)
    for (K, h), v in results.items():
        print(f"K={K:2d}, h={h:3d}   {v:14.4e}")
    print("=" * 40)
    print("Reference: a capable model should reach < 1e-5.")
    return results


if __name__ == "__main__":
    run_capacity_sweep()
