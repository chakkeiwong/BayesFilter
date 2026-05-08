"""
Phase 2: Main training loop.

- tf.function-compiled train_step (one retrace per batch-size / N pair).
- AdamW with cosine LR schedule + linear warmup.
- Gradient clipping.
- Checkpoint/resume to Drive at end of every epoch:
    * `ckpt_dir/latest/`  -- rolling (last 3), used for session resume
    * `ckpt_dir/best/`    -- best val MSE so far, single slot
- Early stopping: patience-based on val MSE, with a min-delta to ignore noise.
  State is persisted alongside the log so it survives Colab disconnects.
- Validation metrics: MSE, per-particle L2 error, transport cost estimates.

At return time, the operator holds the *best* weights (by val MSE), not the
last-epoch weights.

Usage (inside a Colab notebook):

    from training.train import train
    operator = train(config, train_path, val_path, ckpt_dir, log_path)

Config fields used: see config.py. Paths must be valid (Drive mounted).
"""

import os
import json
import time
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tensorflow as tf

from models.operator import AmortizedOTOperator
from data.dataset import load_split, make_tf_dataset


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------
def make_train_step(operator, optimizer, grad_clip, N, d):
    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, N, d), dtype=tf.float32),
        tf.TensorSpec(shape=(None, N), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.float32),
        tf.TensorSpec(shape=(None, N, d), dtype=tf.float32),
    ])
    def train_step(x, w, log_eps, y_tilde_target):
        with tf.GradientTape() as tape:
            y_tilde_pred, _, _ = operator.forward_normalized(
                (x, w, log_eps), training=True)
            # MSE in NORMALIZED space -- matches how y_tilde_target was built.
            loss = tf.reduce_mean(tf.square(y_tilde_pred - y_tilde_target))
        grads = tape.gradient(loss, operator.trainable_variables)
        grads, grad_norm = tf.clip_by_global_norm(grads, grad_clip)
        optimizer.apply_gradients(zip(grads, operator.trainable_variables))
        return loss, grad_norm
    return train_step


def make_eval_step(operator, N, d):
    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, N, d), dtype=tf.float32),
        tf.TensorSpec(shape=(None, N), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.float32),
        tf.TensorSpec(shape=(None, N, d), dtype=tf.float32),
    ])
    def eval_step(x, w, log_eps, y_tilde_target):
        y_tilde_pred, center, scale = operator.forward_normalized(
            (x, w, log_eps), training=False)

        # Primary loss: MSE in normalized space.
        mse = tf.reduce_mean(tf.square(y_tilde_pred - y_tilde_target))

        # Per-particle L2 error in normalized space.
        per_part_l2 = tf.reduce_mean(
            tf.norm(y_tilde_pred - y_tilde_target, axis=-1))

        # Transport cost in normalized space.
        x_tilde = (x - center) / scale
        w_exp = w[:, :, tf.newaxis]
        tc_pred = tf.reduce_mean(
            tf.reduce_sum(w_exp * tf.square(y_tilde_pred - x_tilde),
                          axis=[1, 2]))
        tc_target = tf.reduce_mean(
            tf.reduce_sum(w_exp * tf.square(y_tilde_target - x_tilde),
                          axis=[1, 2]))
        return mse, per_part_l2, tc_pred, tc_target
    return eval_step


# ---------------------------------------------------------------------------
# Learning rate schedule
# ---------------------------------------------------------------------------
class CosineWithWarmup(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_lr, warmup_steps, total_steps, min_lr_ratio=0.01):
        super().__init__()
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = base_lr * min_lr_ratio

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup = tf.cast(self.warmup_steps, tf.float32)
        total = tf.cast(self.total_steps, tf.float32)
        # Linear warmup.
        lr_warmup = self.base_lr * step / tf.maximum(warmup, 1.0)
        # Cosine decay.
        progress = (step - warmup) / tf.maximum(total - warmup, 1.0)
        progress = tf.clip_by_value(progress, 0.0, 1.0)
        lr_cos = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * \
                 (1.0 + tf.cos(np.pi * progress))
        return tf.where(step < warmup, lr_warmup, lr_cos)


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------
def train(config, data_dir, ckpt_dir, log_path=None):
    """
    Args:
        config: dict with 'problem', 'model', 'training', 'data' sections.
        data_dir: directory containing train/val split files; loader
            scans `{split}_N{N}_d{d}__*.npz` and concatenates compatible files.
        ckpt_dir: directory to save checkpoints (typically on Drive).
        log_path: optional JSON path to append per-epoch metrics.
    """
    os.makedirs(ckpt_dir, exist_ok=True)

    N = config['data']['N']
    d = config['problem']['d']
    log10_eps_grid = config['data']['log10_eps_grid']

    # ---- Data -----------------------------------------------------------
    print(f"Loading train split from {data_dir}")
    train_split = load_split(data_dir, split='train', N=N, d=d,
                             log10_eps_grid=log10_eps_grid)
    print(f"Loading val split from {data_dir}")
    val_split = load_split(data_dir, split='val', N=N, d=d,
                           log10_eps_grid=log10_eps_grid)
    print(f"  train: {train_split['x'].shape[0]} clouds, "
          f"val: {val_split['x'].shape[0]} clouds, "
          f"N={N}, d={d}, n_eps={train_split['y_tilde'].shape[1]}")

    bs = config['training']['batch_size']
    train_ds = make_tf_dataset(train_split, bs, shuffle=True, training=True,
                               seed=config['training']['seed'])
    val_ds = make_tf_dataset(val_split, bs, shuffle=False, training=False)

    steps_per_epoch = train_split['x'].shape[0] // bs
    n_epochs = config['training']['n_epochs']
    warmup_steps = config['training']['warmup_epochs'] * steps_per_epoch
    total_steps = n_epochs * steps_per_epoch

    # ---- Model ----------------------------------------------------------
    tf.random.set_seed(config['training']['seed'])
    np.random.seed(config['training']['seed'])
    operator = AmortizedOTOperator(
        d=config['problem']['d'],
        K=config['model']['K'],
        h=config['model']['h'],
        d_model=config['model']['d_model'],
        d_embed=config['model']['d_embed'],
        num_heads=config['model']['num_heads'],
        num_inducing=config['model']['num_inducing'],
        num_isab=config['model']['num_isab'],
        num_seeds=config['model']['num_seeds'],
        hypernet_hidden=config['model']['hypernet_hidden'],
    )
    # Build the model so we have variables.
    N = train_split['x'].shape[1]
    d = config['problem']['d']
    dummy_x = tf.zeros((1, N, d))
    dummy_w = tf.fill((1, N), 1.0 / N)
    dummy_e = tf.fill((1,), tf.math.log(0.01))
    _ = operator((dummy_x, dummy_w, dummy_e))

    n_params = sum(np.prod(v.shape) for v in operator.trainable_variables)
    print(f"Model has {n_params:,} trainable parameters.")

    # ---- Optimizer ------------------------------------------------------
    lr_schedule = CosineWithWarmup(
        base_lr=config['training']['lr'],
        warmup_steps=warmup_steps,
        total_steps=total_steps,
    )
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=lr_schedule,
        weight_decay=config['training']['weight_decay'],
    )

    # ---- Checkpoint -----------------------------------------------------
    # Critical: in Keras 3, the optimizer's `_trainable_variables` list is
    # populated lazily on the first `apply_gradients` call. If we restore
    # a checkpoint before any apply_gradients has run, the list is empty
    # and the restorer cannot map saved values back into the model. We
    # force the list to populate by running one zero-loss step here.
    with tf.GradientTape() as _tape:
        _y = operator((dummy_x, dummy_w, dummy_e))
        _loss = tf.reduce_sum(_y * 0.0)
    _grads = _tape.gradient(_loss, operator.trainable_variables)
    optimizer.apply_gradients(zip(_grads, operator.trainable_variables))

    latest_dir = os.path.join(ckpt_dir, 'latest')
    best_dir = os.path.join(ckpt_dir, 'best')
    os.makedirs(latest_dir, exist_ok=True)
    os.makedirs(best_dir, exist_ok=True)

    ckpt = tf.train.Checkpoint(operator=operator, optimizer=optimizer,
                               step=tf.Variable(0, dtype=tf.int64))
    manager = tf.train.CheckpointManager(ckpt, latest_dir, max_to_keep=3)
    best_manager = tf.train.CheckpointManager(ckpt, best_dir, max_to_keep=1)

    start_epoch = 0
    if manager.latest_checkpoint:
        print(f"Restoring from {manager.latest_checkpoint}")
        ckpt.restore(manager.latest_checkpoint)
        start_epoch = int(ckpt.step.numpy()) // steps_per_epoch
        print(f"Resuming at epoch {start_epoch}")

    # ---- Early-stopping state (persisted across sessions) --------------
    # Sidecar JSON lives next to the log; survives Colab disconnects.
    # Keys: best_val_mse, best_epoch, epochs_since_improvement.
    es_state_path = None
    if log_path:
        es_state_path = log_path.replace('.json', '_es_state.json')
    es_state = {
        'best_val_mse': float('inf'),
        'best_epoch': -1,
        'epochs_since_improvement': 0,
    }
    if es_state_path and os.path.exists(es_state_path):
        with open(es_state_path) as f:
            es_state = json.load(f)
        print(f"Resumed early-stopping state: best_val_mse={es_state['best_val_mse']:.5f} "
              f"at epoch {es_state['best_epoch']}, "
              f"epochs_since_improvement={es_state['epochs_since_improvement']}")

    patience = config['training'].get('early_stop_patience', n_epochs)
    min_delta_rel = config['training'].get('early_stop_min_delta_rel', 1e-3)

    # ---- Compile steps --------------------------------------------------
    train_step = make_train_step(operator, optimizer,
                                 config['training']['grad_clip'], N, d)
    eval_step = make_eval_step(operator, N, d)

    # ---- Loop -----------------------------------------------------------
    log_entries = []
    if log_path and os.path.exists(log_path):
        with open(log_path) as f:
            log_entries = json.load(f)

    stopped_early = False
    for epoch in range(start_epoch, n_epochs):
        # --- Train ---
        t0 = time.time()
        train_losses = []
        for step_i, (x, w, log_eps, y_tgt) in enumerate(train_ds.take(steps_per_epoch)):
            loss, gn = train_step(x, w, log_eps, y_tgt)
            train_losses.append(float(loss.numpy()))
            ckpt.step.assign_add(1)
        train_mse = float(np.mean(train_losses))
        train_time = time.time() - t0

        # --- Validate ---
        t1 = time.time()
        v_mse, v_l2, v_tc_p, v_tc_t = [], [], [], []
        for x, w, log_eps, y_tgt in val_ds:
            mse, l2, tc_p, tc_t = eval_step(x, w, log_eps, y_tgt)
            v_mse.append(float(mse)); v_l2.append(float(l2))
            v_tc_p.append(float(tc_p)); v_tc_t.append(float(tc_t))
        val_time = time.time() - t1

        val_mse = float(np.mean(v_mse))
        val_l2 = float(np.mean(v_l2))
        val_tc_pred = float(np.mean(v_tc_p))
        val_tc_target = float(np.mean(v_tc_t))

        # --- Check for improvement ---
        # "Improvement" = val_mse dropped by at least min_delta_rel * best.
        # We compare against best_val_mse * (1 - min_delta_rel) to avoid
        # rewarding noise-level improvements near convergence.
        threshold = es_state['best_val_mse'] * (1.0 - min_delta_rel)
        improved = val_mse < threshold

        if improved:
            es_state['best_val_mse'] = val_mse
            es_state['best_epoch'] = epoch
            es_state['epochs_since_improvement'] = 0
            best_manager.save()     # snapshot the best-so-far model
            improvement_marker = ' *'
        else:
            es_state['epochs_since_improvement'] += 1
            improvement_marker = ''

        entry = {
            'epoch': epoch,
            'train_mse': train_mse,
            'val_mse': val_mse,
            'val_l2': val_l2,
            'val_tc_pred': val_tc_pred,
            'val_tc_target': val_tc_target,
            'train_time_s': train_time,
            'val_time_s': val_time,
            'is_best': bool(improved),
        }
        log_entries.append(entry)
        if log_path:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            with open(log_path, 'w') as f:
                json.dump(log_entries, f, indent=2)
        if es_state_path:
            with open(es_state_path, 'w') as f:
                json.dump(es_state, f, indent=2)

        print(f"[epoch {epoch:3d}] "
              f"train_mse={train_mse:.5f}  val_mse={val_mse:.5f}{improvement_marker}  "
              f"val_l2={val_l2:.4f}  tc_pred/target={val_tc_pred:.3f}/{val_tc_target:.3f}  "
              f"({train_time:.1f}s train, {val_time:.1f}s val)  "
              f"[patience {es_state['epochs_since_improvement']}/{patience}]")

        # --- Save latest ---
        manager.save()

        # --- Early stop check ---
        if es_state['epochs_since_improvement'] >= patience:
            print(f"Early stopping at epoch {epoch}: "
                  f"no improvement for {patience} epochs. "
                  f"Best val_mse={es_state['best_val_mse']:.5f} "
                  f"at epoch {es_state['best_epoch']}.")
            stopped_early = True
            break

    if not stopped_early:
        print("Training complete (reached n_epochs).")

    # --- Restore best checkpoint before returning ---
    # This is what the caller typically wants: the best model, not the last.
    if best_manager.latest_checkpoint:
        print(f"Restoring best checkpoint from epoch {es_state['best_epoch']} "
              f"(val_mse={es_state['best_val_mse']:.5f})")
        ckpt.restore(best_manager.latest_checkpoint)
    else:
        print("WARNING: no best checkpoint was saved. Returning latest model instead.")

    return operator
