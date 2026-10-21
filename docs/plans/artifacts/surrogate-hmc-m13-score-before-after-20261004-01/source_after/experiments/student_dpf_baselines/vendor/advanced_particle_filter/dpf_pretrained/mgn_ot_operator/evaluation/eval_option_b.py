"""
Evaluation entry point for Option B (CouplingOperator).

Mirrors evaluation/eval_on_ood.py but builds a CouplingOperator and uses
its 4-tuple forward_normalized signature.

Usage:
    python -m evaluation.eval_option_b \\
        --data /path/to/split.npz \\
        --ckpt_dir /path/to/checkpoints \\
        --out /path/to/metrics.json
"""

import argparse
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tensorflow as tf

from config import CONFIG
from models.operator_b import CouplingOperator
from data.dataset import load_split
from data.sinkhorn_targets import sinkhorn_log


def sinkhorn_divergence_pairs(x, y, eps=0.1, n_iters=100):
    """Sinkhorn divergence between two uniformly-weighted empirical measures
    (x, y), both shape (B, N, d).
    """
    B = tf.shape(x)[0]
    N = tf.shape(x)[1]
    N_f = tf.cast(N, x.dtype)
    u = tf.fill((B, N), 1.0 / N_f)
    eps_b = tf.fill((B,), tf.constant(eps, dtype=x.dtype))

    def _cost(a, b):
        C = tf.reduce_sum(
            tf.square(a[:, :, tf.newaxis, :] - b[:, tf.newaxis, :, :]),
            axis=-1,
        )
        T = sinkhorn_log(C, u, u, eps_b, n_iters=n_iters)
        return tf.reduce_sum(T * C, axis=[1, 2])

    return 2.0 * _cost(x, y) - _cost(x, x) - _cost(y, y)


def build_and_restore(ckpt_dir, d, N, model_cfg):
    op = CouplingOperator(
        d=d,
        d_model=model_cfg['d_model'],
        num_heads=model_cfg['num_heads'],
        num_inducing=model_cfg['num_inducing'],
        num_isab=model_cfg['num_isab'],
        d_head=model_cfg['d_head'],
        condition_on_log_eps=model_cfg.get('condition_on_log_eps', True),
    )
    dummy_x = tf.zeros((1, N, d))
    dummy_w = tf.fill((1, N), 1.0 / N)
    dummy_e = tf.fill((1,), tf.math.log(0.5))
    _ = op((dummy_x, dummy_w, dummy_e))

    # Critical: the training-time optimizer was constructed with a callable
    # learning_rate (CosineWithWarmup schedule). When you pass a scalar LR
    # to AdamW, Keras creates an extra Variable for `learning_rate`, which
    # shifts the variable indices and causes a shape-mismatch on restore.
    # Use a callable schedule here too so the variable layout matches.
    dummy_lr = tf.keras.optimizers.schedules.PolynomialDecay(1e-4, 1, 1e-4)
    dummy_opt = tf.keras.optimizers.AdamW(learning_rate=dummy_lr,
                                          weight_decay=1e-5)
    # Force the optimizer to materialize its slot vars by doing one zero step.
    # This makes its _trainable_variables list match the saved structure.
    with tf.GradientTape() as tape:
        y_dummy = op((dummy_x, dummy_w, dummy_e))
        loss_dummy = tf.reduce_sum(y_dummy * 0.0)
    grads_dummy = tape.gradient(loss_dummy, op.trainable_variables)
    dummy_opt.apply_gradients(zip(grads_dummy, op.trainable_variables))
    dummy_step = tf.Variable(0, dtype=tf.int64)

    best_dir = os.path.join(ckpt_dir, 'best')
    latest_dir = os.path.join(ckpt_dir, 'latest')
    ckpt = tf.train.Checkpoint(operator=op, optimizer=dummy_opt,
                               step=dummy_step)
    best_latest = tf.train.latest_checkpoint(best_dir)
    if best_latest is not None:
        print(f"Restoring from best: {best_latest}")
        ckpt.restore(best_latest).expect_partial()
        source = 'best'
    else:
        fallback = tf.train.latest_checkpoint(latest_dir)
        if fallback is None:
            raise FileNotFoundError(
                f"No checkpoints in {best_dir} or {latest_dir}")
        print(f"WARNING: best not found, using latest: {fallback}")
        ckpt.restore(fallback).expect_partial()
        source = 'latest'
    return op, source


def evaluate_file(data_path, ckpt_dir, out_path=None, batch_size=32,
                  model_cfg=None, split=None, N=None, d=None,
                  log10_eps_grid=None):
    """Evaluate on a precomputed split.

    data_path can be:
      - a single .npz file (just load that file), or
      - a directory (loader globs for `{split}_N{N}_d{d}__*.npz` and
        concatenates compatible files; split/N/d/log10_eps_grid must be given).
    """
    print(f"Loading data from {data_path}")
    if os.path.isdir(data_path):
        if any(x is None for x in (split, N, d, log10_eps_grid)):
            raise ValueError(
                "Directory mode requires split, N, d, log10_eps_grid kwargs.")
        split_dict = load_split(data_path, split=split, N=N, d=d,
                                log10_eps_grid=log10_eps_grid)
    else:
        split_dict = load_split(data_path)

    N_actual = split_dict['x'].shape[1]
    d_actual = split_dict['x'].shape[2]
    print(f"  n_clouds={split_dict['x'].shape[0]}, N={N_actual}, d={d_actual}, "
          f"n_eps={split_dict['y_tilde'].shape[1]}")

    if model_cfg is None:
        model_cfg = CONFIG['model_b']

    operator, source = build_and_restore(ckpt_dir, d_actual, N_actual, model_cfg)

    x_all = split_dict['x']
    w_all = split_dict['w']
    y_tilde_all = split_dict['y_tilde']
    grid = split_dict['log10_eps_grid']
    n_clouds = x_all.shape[0]

    print("Running evaluation ...")
    mses, l2s, tcs_p, tcs_t, sds = [], [], [], [], []
    per_eps = {}
    for e_idx, log10_eps in enumerate(grid):
        log_eps = np.float32(log10_eps * np.log(10.0))
        eps_mse, eps_l2 = [], []
        for start in range(0, n_clouds, batch_size):
            end = min(start + batch_size, n_clouds)
            x = tf.constant(x_all[start:end])
            w = tf.constant(w_all[start:end])
            y_tilde_tgt = tf.constant(y_tilde_all[start:end, e_idx])
            le = tf.fill((end - start,), log_eps)
            y_tilde_pred, center, scale, _ = operator.forward_normalized(
                (x, w, le), training=False)
            x_tilde = (x - center) / scale

            mse_b = float(tf.reduce_mean(tf.square(y_tilde_pred - y_tilde_tgt)))
            l2_b = float(tf.reduce_mean(tf.norm(y_tilde_pred - y_tilde_tgt, axis=-1)))
            mses.append(mse_b); l2s.append(l2_b)
            eps_mse.append(mse_b); eps_l2.append(l2_b)

            tcs_p.append(float(tf.reduce_mean(tf.reduce_sum(
                w[:, :, None] * tf.square(y_tilde_pred - x_tilde), axis=[1, 2]))))
            tcs_t.append(float(tf.reduce_mean(tf.reduce_sum(
                w[:, :, None] * tf.square(y_tilde_tgt - x_tilde), axis=[1, 2]))))

            sd = sinkhorn_divergence_pairs(y_tilde_pred, y_tilde_tgt,
                                           eps=0.1, n_iters=50)
            sds.append(float(tf.reduce_mean(sd)))
        per_eps[f'log10_eps_{float(log10_eps):.3f}'] = {
            'mse': float(np.mean(eps_mse)),
            'per_particle_l2': float(np.mean(eps_l2)),
        }

    result = {
        'data_path': data_path,
        'ckpt_dir': ckpt_dir,
        'checkpoint_source': source,
        'n_clouds': int(n_clouds),
        'N': int(N_actual),
        'd': int(d_actual),
        'aggregate': {
            'mse': float(np.mean(mses)),
            'per_particle_l2': float(np.mean(l2s)),
            'transport_cost_pred': float(np.mean(tcs_p)),
            'transport_cost_target': float(np.mean(tcs_t)),
            'sinkhorn_divergence': float(np.mean(sds)),
        },
        'per_eps': per_eps,
    }

    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Saved metrics to {out_path}")

    print("\nAggregate metrics:")
    for k, v in result['aggregate'].items():
        print(f"  {k:30s}: {v:.5f}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--ckpt_dir', type=str, required=True)
    parser.add_argument('--out', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()
    evaluate_file(args.data, args.ckpt_dir, args.out, args.batch_size)


if __name__ == '__main__':
    main()
