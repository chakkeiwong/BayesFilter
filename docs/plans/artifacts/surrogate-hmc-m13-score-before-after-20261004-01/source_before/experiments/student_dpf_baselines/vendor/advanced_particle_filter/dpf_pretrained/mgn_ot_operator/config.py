"""
Centralized configuration for the amortized OT operator.

All hyperparameters live here. Accessed as:
    from config import CONFIG
    N = CONFIG['data']['N']

To run a modified experiment, copy this file and edit.
"""

import numpy as np


CONFIG = {
    # -------------------------------------------------------------------
    # Problem dimensions
    # -------------------------------------------------------------------
    'problem': {
        'd': 2,                   # state dimension
    },

    # -------------------------------------------------------------------
    # Data generation (Source B)
    # -------------------------------------------------------------------
    'data': {
        'N': 1000,                # particles per cloud (fixed for now)
        'n_train': 20_000,        # number of training clouds (20K; can add more later)
        'n_val': 1_000,
        'n_test': 1_000,
        'unimodal_prob': 0.5,
        'dirichlet_alpha_log10_range': (-1.0, 1.0),

        # Epsilon grid (Corenflos-aligned, three-point version).
        # Corenflos uses {0.25, 0.5, 0.75} with 0.5 as default. We use the
        # log10 values (-0.5, -0.3, -0.12) which map to eps ~ (0.316, 0.5, 0.758).
        'log10_eps_grid': [-0.5, -0.3, -0.12],

        # Sinkhorn settings for target precompute.
        # fp32 in the solver -- verified equivalent to fp64 for normalized
        # clouds at eps >= 0.1. Batch=256 to saturate A100 fp32 throughput.
        'sinkhorn_iters': 1000,
        'sinkhorn_batch': 256,
        'normalize_before_sinkhorn': True,
    },

    # -------------------------------------------------------------------
    # Architecture (Option B: CouplingOperator)
    # -------------------------------------------------------------------
    'model_b': {
        'd_model': 64,            # Set Transformer hidden dim
        'num_heads': 4,
        'num_inducing': 16,       # ISAB inducing points
        'num_isab': 2,            # number of ISAB blocks
        'd_head': 64,             # query/key projection dim for coupling head
        'condition_on_log_eps': True,
    },

    # -------------------------------------------------------------------
    # Architecture (Option A: AmortizedOTOperator, M-MGN-based, legacy)
    # -------------------------------------------------------------------
    'model': {
        'K': 4,                   # M-MGN modules
        'h': 32,                  # M-MGN hidden dim per module
        'd_model': 64,            # Set Transformer hidden dim
        'd_embed': 128,           # cloud embedding dim (output of encoder)
        'num_heads': 4,
        'num_inducing': 16,       # ISAB inducing points
        'num_isab': 2,            # number of ISAB blocks
        'num_seeds': 4,           # PMA seeds
        'hypernet_hidden': (256, 256),
    },

    # -------------------------------------------------------------------
    # Training (Phase 2)
    # -------------------------------------------------------------------
    'training': {
        'batch_size': 256,        # aggressive for A100
        'lr': 3e-4,
        'weight_decay': 1e-5,
        'n_epochs': 50,
        'warmup_epochs': 5,
        'grad_clip': 1.0,
        'seed': 0,
        # Early stopping: stop if val_mse hasn't improved for `patience` epochs.
        # Set patience >= n_epochs to effectively disable.
        'early_stop_patience': 10,
        # Minimum relative improvement to count as "improved".
        # e.g. 1e-3 means val_mse must drop by >=0.1% to reset patience.
        'early_stop_min_delta_rel': 1e-3,
    },

    # -------------------------------------------------------------------
    # Paths (Colab-specific; override at session start)
    # -------------------------------------------------------------------
    'paths': {
        'drive_root': '/content/drive/MyDrive/mgn_ot_operator',
        'local_root': '/content/mgn_ot_operator_local',
        'data_subdir': 'data',
        'ckpt_subdir': 'checkpoints',
    },
}
