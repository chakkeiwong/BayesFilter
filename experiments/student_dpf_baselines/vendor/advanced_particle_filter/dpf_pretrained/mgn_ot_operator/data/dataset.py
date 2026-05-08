"""
tf.data.Dataset wrapper for precomputed clouds.

The .npz format from data/precompute.py has:
    x:       (n_clouds, N, d)         raw positions
    w:       (n_clouds, N)             raw weights
    y_tilde: (n_clouds, n_eps, N, d)  ETPF targets in NORMALIZED space
    center:  (n_clouds, 1, d)
    scale:   (n_clouds, 1, 1)
    log10_eps_grid: (n_eps,)

The loader yields (x, w, log_eps, y_tilde) for training. The operator is
given (x, w, log_eps) and produces y in world coords. The LOSS, however,
is computed in normalized space -- see training/train.py for the
y_pred_tilde = MGN output before denormalization.

Backward-compat: if the .npz has the old `y` field (un-normalized targets)
but no `y_tilde`, we raise a clear error. Regenerate via data/precompute.py.
"""

import glob
import os

import numpy as np
import tensorflow as tf


# Filename convention for timestamped files:
#   {split}_N{N}_d{d}__YYYYMMDD_HHMMSS.npz
# Loader globs `{split}_N{N}_d{d}__*.npz` in the given directory.


def _load_single_npz(path):
    """Load one .npz file, validate schema, return dict."""
    data = np.load(path)
    available = list(data.keys())
    if 'y_tilde' not in available:
        if 'y' in available:
            raise ValueError(
                f"{path} uses the OLD schema (un-normalized `y`). "
                f"This file predates the normalize-before-Sinkhorn change. "
                f"Delete it or move it out of the data directory. "
                f"Available keys: {available}"
            )
        raise ValueError(f"{path} missing y_tilde. Keys: {available}")
    return {
        'x': data['x'],
        'w': data['w'],
        'y_tilde': data['y_tilde'],
        'center': data['center'],
        'scale': data['scale'],
        'log10_eps_grid': data['log10_eps_grid'],
        '_source_path': path,
    }


def _find_compatible_files(data_dir, split, N, d, log10_eps_grid,
                           atol=1e-5):
    """Glob the directory for files matching the split/N/d, filter by
    log10_eps_grid equality.

    Returns sorted list of paths whose metadata matches.
    """
    pattern = os.path.join(data_dir, f'{split}_N{N}_d{d}__*.npz')
    candidates = sorted(glob.glob(pattern))
    matching = []
    expected_grid = np.asarray(log10_eps_grid, dtype=np.float32)
    for p in candidates:
        try:
            d_ = np.load(p)
            grid = d_['log10_eps_grid']
        except Exception as e:
            print(f"  (skipping unreadable {p}: {e})")
            continue
        if grid.shape != expected_grid.shape:
            continue
        if not np.allclose(grid, expected_grid, atol=atol):
            continue
        matching.append(p)
    return matching


def load_split(path_or_dir, split=None, N=None, d=None,
               log10_eps_grid=None):
    """Load a precomputed split, concatenating across multiple files if needed.

    Two modes:

      1. Single-file mode: `path_or_dir` points to a .npz file.
         Returns its contents directly.

      2. Directory mode: `path_or_dir` is a directory. Requires `split`,
         `N`, `d`, `log10_eps_grid` kwargs. Globs for files matching
         `{split}_N{N}_d{d}__*.npz` and whose stored log10_eps_grid matches.
         Concatenates along the cloud dimension.

    Returns dict with keys: x, w, y_tilde, center, scale, log10_eps_grid,
    plus (for directory mode) `_source_paths` (list of file paths loaded).
    """
    if os.path.isfile(path_or_dir):
        return _load_single_npz(path_or_dir)

    if not os.path.isdir(path_or_dir):
        raise FileNotFoundError(f"Not a file or dir: {path_or_dir}")

    if split is None or N is None or d is None or log10_eps_grid is None:
        raise ValueError(
            "Directory mode requires split, N, d, log10_eps_grid kwargs.")

    paths = _find_compatible_files(path_or_dir, split, N, d, log10_eps_grid)
    if not paths:
        raise FileNotFoundError(
            f"No compatible files in {path_or_dir} for "
            f"split={split}, N={N}, d={d}, grid={log10_eps_grid}")

    print(f"Loading {len(paths)} file(s) for split '{split}':")
    dicts = []
    for p in paths:
        dd = _load_single_npz(p)
        print(f"  {os.path.basename(p)}: {dd['x'].shape[0]} clouds")
        dicts.append(dd)

    # Concatenate along axis 0 (cloud dim) for x, w, y_tilde, center, scale.
    merged = {
        'x': np.concatenate([d_['x'] for d_ in dicts], axis=0),
        'w': np.concatenate([d_['w'] for d_ in dicts], axis=0),
        'y_tilde': np.concatenate([d_['y_tilde'] for d_ in dicts], axis=0),
        'center': np.concatenate([d_['center'] for d_ in dicts], axis=0),
        'scale': np.concatenate([d_['scale'] for d_ in dicts], axis=0),
        'log10_eps_grid': dicts[0]['log10_eps_grid'],  # same by construction
        '_source_paths': paths,
    }
    print(f"  Merged total: {merged['x'].shape[0]} clouds")
    return merged


def make_tf_dataset(split_dict, batch_size, shuffle=True, seed=0, training=True):
    """Build a tf.data.Dataset from a loaded split.

    For training: each draw picks a random epsilon from the grid.
    For val/test: deterministic iteration through (cloud, eps) pairs.

    Yields batches of (x, w, log_eps, y_tilde_target):
        x:              (B, N, d)    raw positions
        w:              (B, N)       raw weights
        log_eps:        (B,)         natural log of eps
        y_tilde_target: (B, N, d)    ETPF target in normalized space
    """
    n_clouds = split_dict['x'].shape[0]
    n_eps = split_dict['log10_eps_grid'].shape[0]

    x_all = split_dict['x']
    w_all = split_dict['w']
    y_tilde_all = split_dict['y_tilde']
    log10_eps_grid = split_dict['log10_eps_grid']

    if training:
        def gen():
            rng = np.random.default_rng(seed)
            indices = np.arange(n_clouds)
            while True:
                if shuffle:
                    rng.shuffle(indices)
                for i in indices:
                    e = rng.integers(n_eps)
                    log10_eps = log10_eps_grid[e]
                    log_eps = np.float32(log10_eps * np.log(10.0))
                    yield x_all[i], w_all[i], log_eps, y_tilde_all[i, e]
    else:
        def gen():
            for i in range(n_clouds):
                for e in range(n_eps):
                    log10_eps = log10_eps_grid[e]
                    log_eps = np.float32(log10_eps * np.log(10.0))
                    yield x_all[i], w_all[i], log_eps, y_tilde_all[i, e]

    N = x_all.shape[1]
    d = x_all.shape[2]
    output_signature = (
        tf.TensorSpec(shape=(N, d), dtype=tf.float32),
        tf.TensorSpec(shape=(N,), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32),
        tf.TensorSpec(shape=(N, d), dtype=tf.float32),
    )

    ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds
