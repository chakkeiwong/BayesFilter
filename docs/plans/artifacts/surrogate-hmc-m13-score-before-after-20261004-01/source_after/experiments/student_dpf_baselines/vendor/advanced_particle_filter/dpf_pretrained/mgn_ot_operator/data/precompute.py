"""
Offline dataset precompute.

Generates training/val/test clouds from Source B, computes Sinkhorn ETPF targets
for each cloud at every epsilon in the grid, and saves to disk as a single
numpy .npz file per split.

Targets are computed in NORMALIZED space (operator's internal frame). This
matches Corenflos et al.'s recommended workflow of rescaling the cost matrix
by cloud spread before running Sinkhorn at eps ~ O(1).

Output format (per split):
    x:       (n_clouds, N, d)          float32  raw positions
    w:       (n_clouds, N)              float32  raw weights
    y_tilde: (n_clouds, n_eps, N, d)   float32  ETPF targets in normalized space
    center:  (n_clouds, 1, d)           float32  per-cloud center
    scale:   (n_clouds, 1, 1)           float32  per-cloud scale
    log10_eps_grid:  (n_eps,)           float32

The operator's MGN output is in normalized space, so training loss compares
y_pred_tilde (from operator, pre-denorm) to y_tilde directly. No denorm in
the loss.

Run this OUTSIDE the main training session. Typically a one-time cost.

Usage:
    python -m data.precompute --split train --n_clouds 50000 --N 1000 --d 2 \
        --out_dir /content/drive/MyDrive/mgn_ot_operator/data
"""

import argparse
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tensorflow as tf

from data.cloud_sampler import sample_clouds_batch
from data.sinkhorn_targets import etpf_targets_normalized, summarize_convergence


def precompute_split(n_clouds, N, d, log10_eps_grid,
                     out_path, seed,
                     sinkhorn_batch=64, sinkhorn_iters=1000,
                     convergence_tol=1e-6, warn_frac=0.01):
    """
    Args:
        convergence_tol: threshold on (row_res + col_res) L1 for "converged"
            per cloud. Tight because normalized clouds + eps in [0.2, 2.0]
            with 1000 iters should give essentially machine-precision residuals.
        warn_frac: if more than this fraction of clouds at any eps fails to
            converge, a WARNING is printed.
    """
    rng = np.random.default_rng(seed)
    print(f"Sampling {n_clouds} clouds (N={N}, d={d}) ...")
    xs, ws = sample_clouds_batch(n_clouds, N, d, rng)
    print(f"  xs shape: {xs.shape},  ws shape: {ws.shape}")

    n_eps = len(log10_eps_grid)
    y_tilde_all = np.empty((n_clouds, n_eps, N, d), dtype=np.float32)
    center_all = np.empty((n_clouds, 1, d), dtype=np.float32)
    scale_all = np.empty((n_clouds, 1, 1), dtype=np.float32)

    conv_report = {}

    xs_tf = tf.constant(xs)
    ws_tf = tf.constant(ws)

    # Compute and cache per-cloud (center, scale) once. These don't depend on eps.
    # We could call weighted_normalize explicitly, but it's simpler to let
    # etpf_targets_normalized return them (they're deterministic, same each call).
    center_filled = False

    for e_idx, log10_eps in enumerate(log10_eps_grid):
        eps = np.float32(10.0 ** log10_eps)
        print(f"  Sinkhorn targets for log10(eps)={log10_eps:.3f} (eps={eps:.4g}) ...")

        row_res_all = np.empty(n_clouds, dtype=np.float32)
        col_res_all = np.empty(n_clouds, dtype=np.float32)

        for start in range(0, n_clouds, sinkhorn_batch):
            end = min(start + sinkhorn_batch, n_clouds)
            x_chunk = xs_tf[start:end]
            w_chunk = ws_tf[start:end]
            eps_chunk = tf.fill((end - start,), eps)

            # Returns 6 tensors: y_tilde, T, row_res, col_res, center, scale
            y_tilde, _, row_res, col_res, cent, scl = etpf_targets_normalized(
                x_chunk, w_chunk, eps_chunk,
                n_iters=sinkhorn_iters,
                return_residuals=True,
                return_scale_center=True,
            )
            y_tilde_all[start:end, e_idx] = y_tilde.numpy()
            row_res_all[start:end] = row_res.numpy()
            col_res_all[start:end] = col_res.numpy()
            if not center_filled:
                center_all[start:end] = cent.numpy()
                scale_all[start:end] = scl.numpy()

        center_filled = True

        summary = summarize_convergence(row_res_all, col_res_all,
                                        tol=convergence_tol)
        conv_report[f'log10_eps_{log10_eps:.3f}'] = summary

        frac_bad = summary['frac_not_converged']
        msg = (f"    max row_res={summary['max_row']:.2e}, "
               f"max col_res={summary['max_col']:.2e}, "
               f"frac_not_converged={frac_bad:.4f}")
        if frac_bad > warn_frac:
            print(f"    WARNING: {msg}")
            print(f"    Consider increasing sinkhorn_iters (currently "
                  f"{sinkhorn_iters}) for log10(eps)={log10_eps:.3f}.")
        else:
            print(f"    OK: {msg}")

    print(f"Saving to {out_path} ...")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(
        out_path,
        x=xs,
        w=ws,
        y_tilde=y_tilde_all,
        center=center_all,
        scale=scale_all,
        log10_eps_grid=np.array(log10_eps_grid, dtype=np.float32),
    )
    print(f"  saved. File size: {os.path.getsize(out_path) / 1e6:.1f} MB")

    conv_path = out_path.replace('.npz', '_convergence.json')
    with open(conv_path, 'w') as f:
        json.dump({
            'sinkhorn_iters': sinkhorn_iters,
            'convergence_tol': convergence_tol,
            'normalized_space': True,
            'per_eps': conv_report,
        }, f, indent=2)
    print(f"  convergence report: {conv_path}")


# ---------------------------------------------------------------------------
# Incremental ensure_split: check existing, compute only the shortfall
# ---------------------------------------------------------------------------
def _timestamp_suffix():
    from datetime import datetime
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def ensure_split(data_dir, split, N, d, log10_eps_grid,
                 n_clouds_target, base_seed,
                 sinkhorn_batch=256, sinkhorn_iters=1000):
    """Make sure at least n_clouds_target compatible clouds exist on disk.

    Scans data_dir for files matching `{split}_N{N}_d{d}__*.npz` whose
    stored log10_eps_grid matches the requested one. If the sum of their
    cloud counts >= n_clouds_target, returns without doing anything.
    Otherwise computes exactly the shortfall with a fresh timestamped file.

    Seeding: new clouds are drawn with `seed = base_seed + existing_count`,
    so the dataset is reproducible and non-overlapping with previous runs.

    Returns: list of all compatible file paths (existing + new).
    """
    # Avoid a circular import (precompute -> dataset -> precompute) by doing
    # the import at call time.
    from data.dataset import _find_compatible_files

    os.makedirs(data_dir, exist_ok=True)
    existing_paths = _find_compatible_files(
        data_dir, split, N, d, log10_eps_grid)
    existing_counts = []
    for p in existing_paths:
        d_ = np.load(p)
        existing_counts.append(int(d_['x'].shape[0]))
        d_.close()
    existing_total = sum(existing_counts)

    print(f"[{split}] target={n_clouds_target} clouds; "
          f"found {len(existing_paths)} compatible file(s) totaling "
          f"{existing_total} clouds")
    for p, c in zip(existing_paths, existing_counts):
        print(f"    {os.path.basename(p)}: {c}")

    shortfall = n_clouds_target - existing_total
    if shortfall <= 0:
        print(f"  already at or above target. Skipping precompute.")
        return existing_paths

    print(f"  shortfall: {shortfall} clouds. Computing new file ...")

    ts = _timestamp_suffix()
    new_path = os.path.join(
        data_dir, f'{split}_N{N}_d{d}__{ts}.npz')
    seed_for_new = base_seed + existing_total

    precompute_split(
        n_clouds=shortfall,
        N=N,
        d=d,
        log10_eps_grid=log10_eps_grid,
        out_path=new_path,
        seed=seed_for_new,
        sinkhorn_batch=sinkhorn_batch,
        sinkhorn_iters=sinkhorn_iters,
    )

    return existing_paths + [new_path]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=['train', 'val', 'test'], required=True)
    parser.add_argument('--n_clouds', type=int, required=True)
    parser.add_argument('--N', type=int, default=1000)
    parser.add_argument('--d', type=int, default=2)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--sinkhorn_batch', type=int, default=256)
    parser.add_argument('--sinkhorn_iters', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    seed_defaults = {'train': 1000, 'val': 2000, 'test': 3000}
    base_seed = args.seed if args.seed is not None else seed_defaults[args.split]

    # Corenflos-aligned three-point grid on normalized clouds.
    log10_eps_grid = [-0.5, -0.3, -0.12]

    # CLI routes through ensure_split so it does the right thing if files exist.
    ensure_split(
        data_dir=args.out_dir,
        split=args.split,
        N=args.N, d=args.d,
        log10_eps_grid=log10_eps_grid,
        n_clouds_target=args.n_clouds,
        base_seed=base_seed,
        sinkhorn_batch=args.sinkhorn_batch,
        sinkhorn_iters=args.sinkhorn_iters,
    )


if __name__ == '__main__':
    main()
