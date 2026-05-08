"""
Generic evaluation entry point.

Runs `evaluate_on_split` against any precomputed .npz that follows the
standard format (fields x, w, y, log10_eps_grid). This is intentionally
distribution-agnostic:

  - Point it at the in-distribution test set to reproduce Phase 3 numbers.
  - Point it at an OOD set (ring clouds, uniform clouds, etc.) once they've
    been precomputed with the same schema.
  - Point it at SVSSM-generated clouds once they've been converted to .npz.

The operator is restored from `best/` inside the checkpoint directory
(not `latest/`), so this always reports best-model metrics.

Usage:
    python -m evaluation.eval_on_ood \\
        --data /path/to/ood_split.npz \\
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
from models.operator import AmortizedOTOperator
from data.dataset import load_split
from evaluation.metrics import evaluate_on_split


def build_and_restore(ckpt_dir, d, N, model_cfg):
    """Build an operator matching the trained one and restore best weights."""
    operator = AmortizedOTOperator(
        d=d,
        K=model_cfg['K'],
        h=model_cfg['h'],
        d_model=model_cfg['d_model'],
        d_embed=model_cfg['d_embed'],
        num_heads=model_cfg['num_heads'],
        num_inducing=model_cfg['num_inducing'],
        num_isab=model_cfg['num_isab'],
        num_seeds=model_cfg['num_seeds'],
        hypernet_hidden=model_cfg['hypernet_hidden'],
    )
    # Build variables.
    dummy_x = tf.zeros((1, N, d))
    dummy_w = tf.fill((1, N), 1.0 / N)
    dummy_e = tf.fill((1,), tf.math.log(0.01))
    _ = operator((dummy_x, dummy_w, dummy_e))

    # Mirror the saved checkpoint structure (operator + optimizer + step).
    # The training-time optimizer used a callable LR schedule; we must use
    # one here too, otherwise an extra `learning_rate` Variable shifts the
    # variable layout and restoration fails with a shape-mismatch error.
    dummy_lr = tf.keras.optimizers.schedules.PolynomialDecay(1e-4, 1, 1e-4)
    dummy_opt = tf.keras.optimizers.AdamW(learning_rate=dummy_lr,
                                          weight_decay=1e-5)
    with tf.GradientTape() as tape:
        y_dummy = operator((dummy_x, dummy_w, dummy_e))
        loss_dummy = tf.reduce_sum(y_dummy * 0.0)
    grads_dummy = tape.gradient(loss_dummy, operator.trainable_variables)
    dummy_opt.apply_gradients(zip(grads_dummy, operator.trainable_variables))
    dummy_step = tf.Variable(0, dtype=tf.int64)

    best_dir = os.path.join(ckpt_dir, 'best')
    latest_dir = os.path.join(ckpt_dir, 'latest')

    ckpt = tf.train.Checkpoint(operator=operator, optimizer=dummy_opt,
                               step=dummy_step)
    # Prefer best; fall back to latest if best wasn't saved.
    best_latest = tf.train.latest_checkpoint(best_dir)
    if best_latest is not None:
        print(f"Restoring from best: {best_latest}")
        ckpt.restore(best_latest).expect_partial()
        source = 'best'
    else:
        fallback = tf.train.latest_checkpoint(latest_dir)
        if fallback is None:
            raise FileNotFoundError(
                f"No checkpoints found in {best_dir} or {latest_dir}."
            )
        print(f"WARNING: best checkpoint not found, falling back to latest: {fallback}")
        ckpt.restore(fallback).expect_partial()
        source = 'latest (fallback)'

    return operator, source


def evaluate_file(data_path, ckpt_dir, out_path, batch_size=32,
                  model_cfg=None, d=None):
    """Run evaluate_on_split on the given .npz and save results to JSON.

    If model_cfg is None, the defaults from config.CONFIG['model'] are used.
    If d is None, it is inferred from the .npz.
    """
    print(f"Loading data from {data_path}")
    split = load_split(data_path)
    N = split['x'].shape[1]
    d_inferred = split['x'].shape[2]
    if d is not None and d != d_inferred:
        raise ValueError(f"d mismatch: arg={d}, inferred from data={d_inferred}")
    d = d_inferred

    print(f"  n_clouds={split['x'].shape[0]}, N={N}, d={d}, "
          f"n_eps={split['y_tilde'].shape[1]}")

    if model_cfg is None:
        model_cfg = CONFIG['model']

    operator, source = build_and_restore(ckpt_dir, d, N, model_cfg)

    print("Running evaluate_on_split ...")
    metrics = evaluate_on_split(operator, split, batch_size=batch_size)

    # Include a per-epsilon breakdown. Reuses evaluate_on_split-style aggregation
    # but per-epsilon instead of marginalizing. All metrics in NORMALIZED space
    # to match the target storage format.
    print("Computing per-epsilon metrics ...")
    per_eps = {}
    x_all = split['x']
    w_all = split['w']
    y_tilde_all = split['y_tilde']
    grid = split['log10_eps_grid']
    n_clouds = x_all.shape[0]
    for e_idx, log10_eps in enumerate(grid):
        log_eps = np.float32(log10_eps * np.log(10.0))
        mses, l2s = [], []
        for start in range(0, n_clouds, batch_size):
            end = min(start + batch_size, n_clouds)
            x = tf.constant(x_all[start:end])
            w = tf.constant(w_all[start:end])
            y_tilde_tgt = tf.constant(y_tilde_all[start:end, e_idx])
            le = tf.fill((end - start,), log_eps)
            y_tilde_pred, _, _ = operator.forward_normalized(
                (x, w, le), training=False)
            mses.append(float(tf.reduce_mean(
                tf.square(y_tilde_pred - y_tilde_tgt))))
            l2s.append(float(tf.reduce_mean(
                tf.norm(y_tilde_pred - y_tilde_tgt, axis=-1))))
        per_eps[f'log10_eps_{float(log10_eps):.3f}'] = {
            'mse': float(np.mean(mses)),
            'per_particle_l2': float(np.mean(l2s)),
        }

    result = {
        'data_path': data_path,
        'ckpt_dir': ckpt_dir,
        'checkpoint_source': source,
        'n_clouds': int(split['x'].shape[0]),
        'N': int(N),
        'd': int(d),
        'aggregate': metrics,
        'per_eps': per_eps,
    }

    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Saved metrics to {out_path}")

    print("\nAggregate metrics:")
    for k, v in metrics.items():
        print(f"  {k:30s}: {v:.5f}")

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True,
                        help='Path to .npz with fields x, w, y, log10_eps_grid')
    parser.add_argument('--ckpt_dir', type=str, required=True,
                        help='Checkpoint root (containing best/ and latest/)')
    parser.add_argument('--out', type=str, default=None,
                        help='Where to save JSON metrics; skipped if omitted.')
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    evaluate_file(
        data_path=args.data,
        ckpt_dir=args.ckpt_dir,
        out_path=args.out,
        batch_size=args.batch_size,
    )


if __name__ == '__main__':
    main()
