"""
Amortized OT resampler — drop-in replacement for sinkhorn_resample.

Wraps the trained CouplingOperator (Set Transformer + attention coupling
head, see dpf_pretrained/mgn_ot_operator/) so it satisfies the same
interface as sinkhorn_resample:

    particles_r, log_w_r = amortized_resample(particles, log_w)
        particles:   [B, N, D]   fp64 (DPF convention)
        log_w:       [B, N]      fp64 normalized log-weights
        particles_r: [B, N, D]   fp64
        log_w_r:     [B, N]      fp64, uniform = -log(N)

The trained operator runs in fp32 internally (that is what the precompute
and training used). The adapter casts at the boundary so the rest of the
DPF pipeline can stay in fp64 for HMC numerical stability.

Construction loads weights from `ckpt_dir` (the path you'd see in the
training notebook, typically <drive>/checkpoints_option_b/best). The
dummy-optimizer trick from the eval scripts is required: in Keras 3 the
saved variable values live under `optimizer/_trainable_variables/...`,
so we need a parallel optimizer object during restore for the matcher
to find them.

Eps semantics
-------------
The operator was trained on a Corenflos-style epsilon (regularization
strength on normalized clouds, range ~0.2 to 2.0, default 0.5). This is
NOT the Sinkhorn epsilon (which has different semantics and only matters
for the sinkhorn resampler, where it goes through a different code path).
The operator's epsilon is fixed at construction time -- it conditions
the network's coupling head -- so per-step or adaptive eps is not yet
supported.

JIT
---
The adapter's __call__ does NOT wrap the operator forward in a fresh
@tf.function -- the operator's own forward path is already tf.function-
compiled internally via the encoder and coupling head. Wrapping again
here would force a retrace. Just call.
"""

import os
from typing import Optional

import numpy as np
import tensorflow as tf

from advanced_particle_filter.dpf_pretrained.mgn_ot_operator.models.operator_b import (
    CouplingOperator,
)


# ---------------------------------------------------------------------------
# Default checkpoint location (vendored)
# ---------------------------------------------------------------------------
def _default_ckpt_dir() -> str:
    """Resolve the bundled checkpoint location relative to this file.

    The codebase ships a copy of the trained operator at
    `dpf_pretrained/mgn_ot_operator/checkpoints_option_b/`. This lets users
    construct AmortizedOTResampler() without needing to manually download
    weights or set environment variables.

    Override by passing `ckpt_dir=...` explicitly (e.g. to use a fresh
    checkpoint trained in the operator-training repo).
    """
    here = os.path.dirname(os.path.abspath(__file__))
    pkg_root = os.path.dirname(here)  # advanced_particle_filter/
    return os.path.join(
        pkg_root, 'dpf_pretrained', 'mgn_ot_operator', 'checkpoints_option_b',
    )


# ---------------------------------------------------------------------------
# Default config matching what we trained
# ---------------------------------------------------------------------------
DEFAULT_MODEL_CFG = dict(
    d_model=64,
    num_heads=4,
    num_inducing=16,
    num_isab=2,
    d_head=64,
    condition_on_log_eps=True,
)


# ---------------------------------------------------------------------------
# Loader: build CouplingOperator and restore from checkpoint
# ---------------------------------------------------------------------------
def load_amortized_operator(
    ckpt_dir: str,
    d: int,
    N: int,
    model_cfg: Optional[dict] = None,
) -> CouplingOperator:
    """Build CouplingOperator and restore best (or latest) weights.

    Args:
        ckpt_dir: Path to the operator's checkpoint root. Must contain
                  either a `best/` or `latest/` subdir (or both); we
                  prefer `best/`.
        d:        State dimension. Must match the trained operator (=2 for
                  the current PoC).
        N:        Number of particles. Used only for building variables;
                  the operator handles arbitrary N at inference.
        model_cfg: Architecture config. Defaults to what we trained.

    Returns:
        Restored CouplingOperator. Note: built/loaded in tf.float32.
    """
    if model_cfg is None:
        model_cfg = DEFAULT_MODEL_CFG

    op = CouplingOperator(d=d, **model_cfg)

    # Build variables.
    dummy_x = tf.zeros((1, N, d), dtype=tf.float32)
    dummy_w = tf.fill((1, N), 1.0 / N)
    dummy_e = tf.fill((1,), tf.math.log(0.5))
    _ = op((dummy_x, dummy_w, dummy_e))

    # Mirror the saved checkpoint structure: optimizer + step. The
    # training-time optimizer used a callable LR schedule, so we use one
    # too (a constant Polynomial schedule) to match the variable layout.
    dummy_lr = tf.keras.optimizers.schedules.PolynomialDecay(1e-4, 1, 1e-4)
    dummy_opt = tf.keras.optimizers.AdamW(
        learning_rate=dummy_lr, weight_decay=1e-5,
    )
    # One zero step so the optimizer's _trainable_variables list populates
    # with the correct references and ordering.
    with tf.GradientTape() as tape:
        y_dummy = op((dummy_x, dummy_w, dummy_e))
        loss_dummy = tf.reduce_sum(y_dummy * 0.0)
    grads_dummy = tape.gradient(loss_dummy, op.trainable_variables)
    dummy_opt.apply_gradients(zip(grads_dummy, op.trainable_variables))
    dummy_step = tf.Variable(0, dtype=tf.int64)

    best_dir = os.path.join(ckpt_dir, 'best')
    latest_dir = os.path.join(ckpt_dir, 'latest')

    ckpt = tf.train.Checkpoint(
        operator=op, optimizer=dummy_opt, step=dummy_step,
    )
    best_latest = tf.train.latest_checkpoint(best_dir)
    if best_latest is not None:
        print(f"  amortized: restoring from best: {best_latest}")
        ckpt.restore(best_latest).expect_partial()
    else:
        fallback = tf.train.latest_checkpoint(latest_dir)
        if fallback is None:
            raise FileNotFoundError(
                f"No checkpoint found under {best_dir} or {latest_dir}")
        print(f"  amortized: WARNING best not found, using latest: {fallback}")
        ckpt.restore(fallback).expect_partial()

    return op


# ---------------------------------------------------------------------------
# Resampler class — callable, matches sinkhorn_resample interface
# ---------------------------------------------------------------------------
class AmortizedOTResampler:
    """Callable adapter wrapping CouplingOperator for use in a DPF.

    Construction loads the trained weights (one-time cost). At call time,
    handles the fp64<->fp32 boundary for the DPF and exposes the same
    (particles, log_w) -> (particles_r, log_w_r) signature as
    sinkhorn_resample.

    Args:
        ckpt_dir: Path to the operator's checkpoint root. If None (default),
                  uses the vendored checkpoint at
                  dpf_pretrained/mgn_ot_operator/checkpoints_option_b/.
                  Set explicitly to use a freshly-trained checkpoint.
        d:        State dimension. Default 2 (matches trained operator).
        N:        Number of particles. Default 1000.
        eps:      Regularization strength fed to the operator's eps
                  conditioning. Default 0.5 (Corenflos default,
                  what we trained around).
        dtype:    Outer DPF dtype (fp64 expected).
        model_cfg: Architecture config; defaults to trained config.
    """

    def __init__(
        self,
        ckpt_dir: Optional[str] = None,
        d: int = 2,
        N: int = 1000,
        eps: float = 0.5,
        dtype: tf.DType = tf.float64,
        model_cfg: Optional[dict] = None,
    ):
        if ckpt_dir is None:
            ckpt_dir = _default_ckpt_dir()
        self.operator = load_amortized_operator(
            ckpt_dir=ckpt_dir, d=d, N=N, model_cfg=model_cfg,
        )
        self.eps = float(eps)
        self.dtype = dtype
        self.N = N
        self.d = d
        # Precompute the constant log-eps tensor in fp32 (will be tiled per call).
        self._log_eps_scalar = tf.constant(np.log(eps), dtype=tf.float32)

    def __call__(
        self,
        particles: tf.Tensor,
        log_w: tf.Tensor,
    ):
        """Drop-in replacement for sinkhorn_resample.

        Args:
            particles: [B, N, D] fp64 (or whatever self.dtype is).
            log_w:     [B, N]   fp64 normalized log-weights.

        Returns:
            particles_r: [B, N, D] fp64
            log_w_r:     [B, N]   fp64, uniform = -log(N)
        """
        # 1. Cast to fp32 for the operator.
        x_fp32 = tf.cast(particles, tf.float32)
        w_fp32 = tf.cast(tf.exp(log_w), tf.float32)

        # 2. Build the per-cloud log_eps tensor.
        B = tf.shape(particles)[0]
        log_eps = tf.fill((B,), self._log_eps_scalar)

        # 3. Forward pass. operator.call denormalizes back to world coords
        #    so this is the right method (NOT forward_normalized).
        y_fp32 = self.operator((x_fp32, w_fp32, log_eps), training=False)

        # 4. Cast back to outer dtype.
        particles_r = tf.cast(y_fp32, self.dtype)

        # 5. Output weights are uniform (matches sinkhorn_resample convention).
        N_f = tf.cast(tf.shape(particles)[1], self.dtype)
        log_w_r = tf.fill(tf.shape(log_w), -tf.math.log(N_f))

        return particles_r, log_w_r
