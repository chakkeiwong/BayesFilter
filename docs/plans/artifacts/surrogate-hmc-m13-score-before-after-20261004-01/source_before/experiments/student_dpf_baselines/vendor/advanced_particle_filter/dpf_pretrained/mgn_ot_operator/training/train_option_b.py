"""
Phase 2: Main training loop for Option B (CouplingOperator).

Mirrors training/train.py but:
  - Builds CouplingOperator (Set Transformer + attention coupling head)
    instead of AmortizedOTOperator (Set Transformer + hypernet + M-MGN).
  - forward_normalized returns 4 items (y_tilde, center, scale, pi), so
    the train/eval steps unpack accordingly.
  - Uses a separate `model_b` section in config for Option B hyperparameters.

Usage (in Colab):
    from training.train_option_b import train_option_b
    op = train_option_b(config, train_path, val_path, ckpt_dir, log_path)
"""

import os
import json
import time
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tensorflow as tf

from models.operator_b import CouplingOperator
from data.dataset import load_split, make_tf_dataset
from training.train import CosineWithWarmup   # reuse the LR schedule


def make_train_step_b(operator, optimizer, grad_clip, N, d):
    """Build the tf.function-compiled train step with explicit signature.

    The signature pins batch size to None (allowing variable batch at
    training time but NOT at runtime-unknown-sized batches, since we use
    drop_remainder=True the batch is fixed). N and d are pinned so the
    function traces exactly once.
    """
    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, N, d), dtype=tf.float32),  # x
        tf.TensorSpec(shape=(None, N), dtype=tf.float32),      # w
        tf.TensorSpec(shape=(None,), dtype=tf.float32),        # log_eps
        tf.TensorSpec(shape=(None, N, d), dtype=tf.float32),  # y_tilde_target
    ])
    def train_step(x, w, log_eps, y_tilde_target):
        with tf.GradientTape() as tape:
            y_tilde_pred, _, _, _ = operator.forward_normalized(
                (x, w, log_eps), training=True)
            loss = tf.reduce_mean(tf.square(y_tilde_pred - y_tilde_target))
        grads = tape.gradient(loss, operator.trainable_variables)
        grads, grad_norm = tf.clip_by_global_norm(grads, grad_clip)
        optimizer.apply_gradients(zip(grads, operator.trainable_variables))
        return loss, grad_norm
    return train_step


def make_eval_step_b(operator, N, d):
    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, N, d), dtype=tf.float32),
        tf.TensorSpec(shape=(None, N), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.float32),
        tf.TensorSpec(shape=(None, N, d), dtype=tf.float32),
    ])
    def eval_step(x, w, log_eps, y_tilde_target):
        y_tilde_pred, center, scale, _ = operator.forward_normalized(
            (x, w, log_eps), training=False)
        mse = tf.reduce_mean(tf.square(y_tilde_pred - y_tilde_target))
        per_part_l2 = tf.reduce_mean(
            tf.norm(y_tilde_pred - y_tilde_target, axis=-1))
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


def build_operator_b(d, model_cfg):
    return CouplingOperator(
        d=d,
        d_model=model_cfg['d_model'],
        num_heads=model_cfg['num_heads'],
        num_inducing=model_cfg['num_inducing'],
        num_isab=model_cfg['num_isab'],
        d_head=model_cfg['d_head'],
        condition_on_log_eps=model_cfg.get('condition_on_log_eps', True),
    )


def train_option_b(config, data_dir, ckpt_dir, log_path=None):
    """Main training loop for Option B (CouplingOperator).

    Args:
        config: config dict with 'problem', 'data', 'training', 'model_b'.
        data_dir: directory containing train/val split files.
            The loader scans `data_dir/{split}_N{N}_d{d}__*.npz` and
            concatenates compatible files.
        ckpt_dir: directory for checkpoints.
        log_path: optional JSON path for per-epoch logs.
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
    operator = build_operator_b(d, config['model_b'])

    # Build variables.
    dummy_x = tf.zeros((1, N, d))
    dummy_w = tf.fill((1, N), 1.0 / N)
    dummy_e = tf.fill((1,), tf.math.log(0.5))
    _ = operator((dummy_x, dummy_w, dummy_e))
    n_params = sum(np.prod(v.shape) for v in operator.trainable_variables)
    print(f"CouplingOperator has {n_params:,} trainable parameters.")

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
    # Critical: in Keras 3 the optimizer's `_trainable_variables` list is
    # populated lazily on the first `apply_gradients` call. We force it to
    # populate by running one zero-loss step so that any subsequent
    # ckpt.restore() can correctly map saved values into the model.
    with tf.GradientTape() as _tape:
        _y, _, _, _ = operator.forward_normalized(
            (dummy_x, dummy_w, dummy_e), training=False)
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

    # ---- Early-stopping state ------------------------------------------
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
        print(f"Resumed ES state: best_val_mse={es_state['best_val_mse']:.5f} "
              f"@ epoch {es_state['best_epoch']}, "
              f"patience used {es_state['epochs_since_improvement']}")

    patience = config['training'].get('early_stop_patience', n_epochs)
    min_delta_rel = config['training'].get('early_stop_min_delta_rel', 1e-3)

    train_step = make_train_step_b(operator, optimizer,
                                   config['training']['grad_clip'], N, d)
    eval_step = make_eval_step_b(operator, N, d)

    # ---- Loop -----------------------------------------------------------
    log_entries = []
    if log_path and os.path.exists(log_path):
        with open(log_path) as f:
            log_entries = json.load(f)

    stopped_early = False
    for epoch in range(start_epoch, n_epochs):
        t0 = time.time()
        train_losses = []
        for step_i, (x, w, log_eps, y_tgt) in enumerate(
                train_ds.take(steps_per_epoch)):
            loss, _ = train_step(x, w, log_eps, y_tgt)
            train_losses.append(float(loss.numpy()))
            ckpt.step.assign_add(1)
        train_mse = float(np.mean(train_losses))
        train_time = time.time() - t0

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

        threshold = es_state['best_val_mse'] * (1.0 - min_delta_rel)
        improved = val_mse < threshold
        if improved:
            es_state['best_val_mse'] = val_mse
            es_state['best_epoch'] = epoch
            es_state['epochs_since_improvement'] = 0
            best_manager.save()
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

        manager.save()
        if es_state['epochs_since_improvement'] >= patience:
            print(f"Early stopping at epoch {epoch}: "
                  f"no improvement for {patience} epochs. "
                  f"Best val_mse={es_state['best_val_mse']:.5f} "
                  f"@ epoch {es_state['best_epoch']}.")
            stopped_early = True
            break

    if not stopped_early:
        print("Training complete (reached n_epochs).")

    if best_manager.latest_checkpoint:
        print(f"Restoring best checkpoint from epoch {es_state['best_epoch']}")
        ckpt.restore(best_manager.latest_checkpoint)
    else:
        print("WARNING: no best checkpoint was saved.")

    return operator
