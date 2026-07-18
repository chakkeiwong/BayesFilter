"""Deterministic prepared inputs for canonical Contract E LGSSM experiments."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

import tensorflow as tf

from bayesfilter.highdim.transport_chunk_policy import (
    TRANSPORT_CHUNK_POLICY_ID,
    validate_transport_chunks,
)


PREPARATION_ID = "contract_e_lgssm_philox_prepared_inputs_v1"
RESIDUAL_DESIGN_ID = "contract_e_residual_centered_population_scaled_v1"
MARGINAL_POLICY_ID = "contract_e_probability_marginals_tvcol_erow_v1"
TV_COLUMN_TOLERANCE = 1.0e-4
MAXIMUM_ROW_ERROR_TOLERANCE = 1.0e-2
RNG_ALGORITHM = "philox"
RAW_DRAW_DTYPE = tf.float64
INITIAL_DOMAIN_TAG = 101
TRANSITION_DOMAIN_TAG_BASE = 1000
RESIDUAL_DOMAIN_TAG_BASE = 2000
MAX_TIME_STEPS = 1000
STATE_DIMENSION = 3


def _tensor_hash(value: tf.Tensor) -> str:
    tensor = tf.convert_to_tensor(value)
    payload = {
        "dtype": tensor.dtype.name,
        "shape": tensor.shape.as_list(),
        "serialized_sha256": hashlib.sha256(
            tf.io.serialize_tensor(tensor).numpy()
        ).hexdigest(),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _seed_key(root_seed: int, domain_tag: int) -> tf.Tensor:
    for name, value in (("root_seed", root_seed), ("domain_tag", domain_tag)):
        if not -(2**31) <= int(value) < 2**31:
            raise ValueError(f"{name} must fit signed int32")
    return tf.constant([int(root_seed), int(domain_tag)], tf.int32)


def _stateless_normal(shape: Sequence[int], root_seed: int, domain_tag: int) -> tf.Tensor:
    return tf.random.stateless_normal(
        shape,
        seed=_seed_key(root_seed, domain_tag),
        dtype=RAW_DRAW_DTYPE,
        alg=RNG_ALGORITHM,
    )


def _centered_residual_design(raw: tf.Tensor) -> tf.Tensor:
    raw = tf.convert_to_tensor(raw, RAW_DRAW_DTYPE)
    particle_count = raw.shape[-2]
    if particle_count is None or particle_count <= 1:
        raise ValueError("residual design requires statically known N > 1")
    centered = raw - tf.reduce_mean(raw, axis=-2, keepdims=True)
    scale = tf.sqrt(
        tf.cast(particle_count, RAW_DRAW_DTYPE)
        / tf.cast(particle_count - 1, RAW_DRAW_DTYPE)
    )
    return scale * centered


def prepare_contract_e_lgssm_inputs(
    *,
    observations: Any,
    estimator_seeds: Sequence[int],
    num_particles: int,
    fixed_reset_mask: Any,
    prepared_ridge: Any,
    epsilon: Any,
    scaling: Any,
    sinkhorn_steps: int,
    balance_steps: int,
    row_chunk_size: int,
    col_chunk_size: int,
    dtype: tf.dtypes.DType,
) -> dict[str, Any]:
    """Construct fixed-noise inputs under the repository transport policy."""

    dtype = tf.dtypes.as_dtype(dtype)
    if dtype not in (tf.float32, tf.float64):
        raise ValueError("dtype must be float32 or float64")
    if num_particles <= 1:
        raise ValueError("num_particles must be greater than one")
    chunk_selection = validate_transport_chunks(
        num_particles,
        row_chunk_size=row_chunk_size,
        col_chunk_size=col_chunk_size,
    )
    if not estimator_seeds:
        raise ValueError("estimator_seeds must be nonempty")
    if len(set(int(seed) for seed in estimator_seeds)) != len(estimator_seeds):
        raise ValueError("estimator_seeds must be unique and ordered")
    if min(sinkhorn_steps, row_chunk_size, col_chunk_size) <= 0:
        raise ValueError("sinkhorn steps and chunk sizes must be positive")
    if balance_steps < 0:
        raise ValueError("balance_steps must be non-negative")

    observations64 = tf.convert_to_tensor(observations, RAW_DRAW_DTYPE)
    if observations64.shape.rank != 2 or observations64.shape[1] != STATE_DIMENSION:
        raise ValueError("observations must have shape [T, 3]")
    time_steps = observations64.shape[0]
    if time_steps is None or not 0 < time_steps <= MAX_TIME_STEPS:
        raise ValueError(f"time length must be in [1, {MAX_TIME_STEPS}]")
    batch_size = len(estimator_seeds)
    reset_mask = tf.convert_to_tensor(fixed_reset_mask, tf.bool)
    ridge = tf.convert_to_tensor(prepared_ridge, RAW_DRAW_DTYPE)
    expected_batch_time = tf.TensorShape([batch_size, time_steps])
    if reset_mask.shape != expected_batch_time:
        raise ValueError(f"fixed_reset_mask must have shape {expected_batch_time}")
    if ridge.shape != expected_batch_time:
        raise ValueError(f"prepared_ridge must have shape {expected_batch_time}")
    if not bool(tf.reduce_all(tf.math.is_finite(ridge)).numpy()):
        raise ValueError("prepared_ridge must be finite")
    if not bool(tf.reduce_all(ridge > 0.0).numpy()):
        raise ValueError("prepared_ridge must be strictly positive")
    epsilon64 = tf.convert_to_tensor(epsilon, RAW_DRAW_DTYPE)
    scaling64 = tf.convert_to_tensor(scaling, RAW_DRAW_DTYPE)
    if epsilon64.shape.rank != 0 or scaling64.shape.rank != 0:
        raise ValueError("epsilon and scaling must be scalar")
    if not bool(
        (tf.math.is_finite(epsilon64) & (epsilon64 > 0.0)).numpy()
    ):
        raise ValueError("epsilon must be finite and positive")
    if not bool(
        (tf.math.is_finite(scaling64) & (scaling64 > 0.0)).numpy()
    ):
        raise ValueError("scaling must be finite and positive")

    initial_by_seed = []
    transition_by_seed = []
    residual_by_seed = []
    for seed in estimator_seeds:
        root_seed = int(seed)
        initial_by_seed.append(
            _stateless_normal(
                [num_particles, STATE_DIMENSION], root_seed, INITIAL_DOMAIN_TAG
            )
        )
        transition_by_seed.append(
            tf.stack(
                [
                    _stateless_normal(
                        [num_particles, STATE_DIMENSION],
                        root_seed,
                        TRANSITION_DOMAIN_TAG_BASE + time_index,
                    )
                    for time_index in range(time_steps)
                ],
                axis=0,
            )
        )
        residual_raw = tf.stack(
            [
                _stateless_normal(
                    [num_particles, STATE_DIMENSION],
                    root_seed,
                    RESIDUAL_DOMAIN_TAG_BASE + time_index,
                )
                for time_index in range(time_steps)
            ],
            axis=0,
        )
        residual_by_seed.append(_centered_residual_design(residual_raw))

    tensors = {
        "observations": tf.cast(observations64, dtype),
        "initial_noise": tf.cast(tf.stack(initial_by_seed, axis=0), dtype),
        "transition_noise": tf.cast(tf.stack(transition_by_seed, axis=0), dtype),
        "fixed_reset_mask": reset_mask,
        "residual_design": tf.cast(tf.stack(residual_by_seed, axis=0), dtype),
        "prepared_ridge": tf.cast(ridge, dtype),
        "epsilon": tf.cast(epsilon64, dtype),
        "scaling": tf.cast(scaling64, dtype),
    }
    identity = {
        "preparation_id": PREPARATION_ID,
        "residual_design_id": RESIDUAL_DESIGN_ID,
        "tensorflow_version": tf.__version__,
        "rng_algorithm": RNG_ALGORITHM,
        "key_dtype": "int32",
        "key_encoding": "[root_seed, domain_tag]",
        "root_seeds_in_order": [int(seed) for seed in estimator_seeds],
        "domain_tags": {
            "initial": INITIAL_DOMAIN_TAG,
            "transition": "1000 + zero_based_time",
            "residual": "2000 + zero_based_time",
        },
        "raw_draw_dtype": RAW_DRAW_DTYPE.name,
        "final_dtype": dtype.name,
        "axis_convention": {
            "initial_noise": "[batch, particle, state]",
            "transition_noise": "[batch, time, particle, state]",
            "residual_design": "[batch, time, particle, state]",
        },
        "num_particles": int(num_particles),
        "transport_chunk_policy_id": TRANSPORT_CHUNK_POLICY_ID,
        "transport_block_grid": [
            chunk_selection.row_blocks,
            chunk_selection.col_blocks,
        ],
        "time_steps": int(time_steps),
        "sinkhorn_steps": int(sinkhorn_steps),
        "balance_steps": int(balance_steps),
        "marginal_policy_id": MARGINAL_POLICY_ID,
        "tv_column_tolerance": TV_COLUMN_TOLERANCE,
        "maximum_row_error_tolerance": MAXIMUM_ROW_ERROR_TOLERANCE,
        "row_chunk_size": int(row_chunk_size),
        "col_chunk_size": int(col_chunk_size),
        "tensor_sha256": {name: _tensor_hash(value) for name, value in tensors.items()},
    }
    return {"prepared": tensors, "identity": identity}


def prepared_values(result: Mapping[str, Any]) -> dict[str, tf.Tensor]:
    """Return the canonical factory payload from a preparation result."""

    return dict(result["prepared"])


__all__ = [
    "INITIAL_DOMAIN_TAG",
    "MAX_TIME_STEPS",
    "PREPARATION_ID",
    "RESIDUAL_DESIGN_ID",
    "RESIDUAL_DOMAIN_TAG_BASE",
    "RNG_ALGORITHM",
    "TRANSITION_DOMAIN_TAG_BASE",
    "prepare_contract_e_lgssm_inputs",
    "prepared_values",
]
