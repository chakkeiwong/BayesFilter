"""Frozen configuration for the Austria SIR T1 centered-density campaign."""

from __future__ import annotations

import math
from types import MappingProxyType
from typing import Mapping


ARM_TABLE: Mapping[str, Mapping[str, float | int]] = MappingProxyType(
    {
        "a01_r2_rad01_lr3e4_l1_1e9_g01": MappingProxyType(
            {"rank": 2, "radius": 0.01, "learning_rate": 3e-4, "l1_weight": 1e-9, "derivative_weight": 0.1}
        ),
        "a02_r4_rad01_lr3e4_l1_1e9_g01": MappingProxyType(
            {"rank": 4, "radius": 0.01, "learning_rate": 3e-4, "l1_weight": 1e-9, "derivative_weight": 0.1}
        ),
        "a03_r4_rad03_lr3e4_l1_1e9_g01": MappingProxyType(
            {"rank": 4, "radius": 0.03, "learning_rate": 3e-4, "l1_weight": 1e-9, "derivative_weight": 0.1}
        ),
        "a04_r4_rad01_lr1e4_l1_1e9_g01": MappingProxyType(
            {"rank": 4, "radius": 0.01, "learning_rate": 1e-4, "l1_weight": 1e-9, "derivative_weight": 0.1}
        ),
        "a05_r4_rad01_lr3e4_l1_0_g01": MappingProxyType(
            {"rank": 4, "radius": 0.01, "learning_rate": 3e-4, "l1_weight": 0.0, "derivative_weight": 0.1}
        ),
        "a06_r4_rad01_lr3e4_l1_1e8_g01": MappingProxyType(
            {"rank": 4, "radius": 0.01, "learning_rate": 3e-4, "l1_weight": 1e-8, "derivative_weight": 0.1}
        ),
        "a07_r4_rad01_lr3e4_l1_1e9_g0": MappingProxyType(
            {"rank": 4, "radius": 0.01, "learning_rate": 3e-4, "l1_weight": 1e-9, "derivative_weight": 0.0}
        ),
    }
)


INITIALIZER_AUDIT_TABLE: Mapping[
    str, Mapping[str, float | int | str]
] = MappingProxyType(
    {
        "i01_r2_amp1_pert05_lr3e4": MappingProxyType(
            {
                "family": "connected_random_score_prefit",
                "rank": 2,
                "amplitude_scale": 1.0,
                "perturbation_scale": 0.05,
                "learning_rate": 3e-4,
                "steps": 32,
            }
        ),
        "i02_r4_amp1_pert05_lr3e4": MappingProxyType(
            {
                "family": "connected_random_score_prefit",
                "rank": 4,
                "amplitude_scale": 1.0,
                "perturbation_scale": 0.05,
                "learning_rate": 3e-4,
                "steps": 32,
            }
        ),
        "i03_add_ridge1e4_global1": MappingProxyType(
            {
                "family": "exact_additive_score_ridge",
                "rank": 2,
                "ridge_fraction": 1e-4,
                "global_score_weight": 1.0,
                "steps": 0,
            }
        ),
        "i04_add_ridge1e4_global10": MappingProxyType(
            {
                "family": "exact_additive_score_ridge",
                "rank": 2,
                "ridge_fraction": 1e-4,
                "global_score_weight": 10.0,
                "steps": 0,
            }
        ),
    }
)


PREFIX_TANGENT_TABLE: Mapping[str, Mapping[str, float | int]] = MappingProxyType(
    {
        "p01_prefixw001": MappingProxyType({"prefix_weight": 0.01}),
        "p02_prefixw01": MappingProxyType({"prefix_weight": 0.1}),
        "p03_prefixw1": MappingProxyType({"prefix_weight": 1.0}),
        "p04_prefix16_g100_w001": MappingProxyType(
            {
                "prefix_weight": 0.001,
                "prefix_point_count": 16,
                "global_score_weight": 100.0,
            }
        ),
        "p05_prefix16_g100_w01": MappingProxyType(
            {
                "prefix_weight": 0.01,
                "prefix_point_count": 16,
                "global_score_weight": 100.0,
            }
        ),
        "p06_prefix64_g100_w0025": MappingProxyType(
            {
                "prefix_weight": 0.0025,
                "prefix_point_count": 64,
                "global_score_weight": 100.0,
            }
        ),
    }
)


PAIR_TANGENT_TABLE: Mapping[str, Mapping[str, float | int]] = MappingProxyType(
    {
        "r01_pair_ridge1e4": MappingProxyType(
            {
                "ridge_fraction": 1e-4,
                "prefix_point_count": 16,
                "global_score_weight": 100.0,
                "prefix_weight": 0.01,
            }
        ),
        "r02_pair_ridge1e3": MappingProxyType(
            {
                "ridge_fraction": 1e-3,
                "prefix_point_count": 16,
                "global_score_weight": 100.0,
                "prefix_weight": 0.01,
            }
        ),
    }
)


DIRECT_TT_TANGENT_TABLE: Mapping[str, Mapping[str, float | int]] = MappingProxyType(
    {
        "d01_r4_lr1e5": MappingProxyType(
            {"rank": 4, "learning_rate": 1e-5, "steps": 64}
        ),
        "d02_r4_lr3e5": MappingProxyType(
            {"rank": 4, "learning_rate": 3e-5, "steps": 64}
        ),
        "d03_r7_lr1e5": MappingProxyType(
            {"rank": 7, "learning_rate": 1e-5, "steps": 64}
        ),
        "d04_r7_lr3e5": MappingProxyType(
            {"rank": 7, "learning_rate": 3e-5, "steps": 64}
        ),
        "d05_r7_lr1e5_p100_g100": MappingProxyType(
            {
                "rank": 7,
                "learning_rate": 1e-5,
                "steps": 64,
                "point_weight": 100.0,
                "global_weight": 100.0,
                "prefix_weight": 1.0,
            }
        ),
    }
)


ROTATING_PREFIX_TANGENT_TABLE: Mapping[
    str, Mapping[str, float | int | str]
] = MappingProxyType(
    {
        "q01_r7_pool512_batch64_steps256": MappingProxyType(
            {
                "rank": 7,
                "learning_rate": 1e-5,
                "steps": 256,
                "point_weight": 100.0,
                "global_weight": 100.0,
                "prefix_weight": 1.0,
                "fit_pool_size": 512,
                "prefix_batch_size": 64,
                "calibration_size": 64,
                "checkpoint_interval": 8,
                "pool_partition_seed": 85901,
                "minibatch_seed": 85902,
            }
        ),
        "q02_r7_pool512_batch64_steps1024": MappingProxyType(
            {
                "rank": 7,
                "learning_rate": 1e-5,
                "steps": 1024,
                "point_weight": 100.0,
                "global_weight": 100.0,
                "prefix_weight": 1.0,
                "fit_pool_size": 512,
                "prefix_batch_size": 64,
                "calibration_size": 64,
                "checkpoint_interval": 8,
                "pool_partition_seed": 85901,
                "minibatch_seed": 85902,
            }
        ),
        "q03_r8_core_tangent_warm_start_steps256": MappingProxyType(
            {
                "initializer_id": "hash_verified_ungauged_core_tangent_s05_rank8_v1",
                "rank": 8,
                "learning_rate": 1e-5,
                "steps": 256,
                "point_weight": 100.0,
                "global_weight": 100.0,
                "prefix_weight": 1.0,
                "fit_pool_size": 512,
                "prefix_batch_size": 64,
                "calibration_size": 64,
                "checkpoint_interval": 8,
                "pool_partition_seed": 85901,
                "minibatch_seed": 85902,
            }
        ),
        "c01_core_affine_zero_lr1e3_steps256": MappingProxyType(
            {
                "initializer_id": "current_frozen_basis_core_affine_zero_v1",
                "rank": 8,
                "learning_rate": 1e-3,
                "steps": 256,
                "point_weight": 100.0,
                "global_weight": 100.0,
                "prefix_weight": 1.0,
                "fit_pool_size": 512,
                "prefix_batch_size": 64,
                "calibration_size": 64,
                "checkpoint_interval": 8,
                "pool_partition_seed": 85901,
                "minibatch_seed": 85902,
            }
        ),
        "c02_core_affine_zero_lr3e4_steps256": MappingProxyType(
            {
                "initializer_id": "current_frozen_basis_core_affine_zero_v1",
                "rank": 8,
                "learning_rate": 3e-4,
                "steps": 256,
                "point_weight": 100.0,
                "global_weight": 100.0,
                "prefix_weight": 1.0,
                "fit_pool_size": 512,
                "prefix_batch_size": 64,
                "calibration_size": 64,
                "checkpoint_interval": 8,
                "pool_partition_seed": 85901,
                "minibatch_seed": 85902,
            }
        ),
    }
)


CORE_AFFINE_LBFGS_TABLE: Mapping[str, Mapping[str, float | int | str]] = (
    MappingProxyType(
        {
            "l01_core_affine_fullpool_lbfgs": MappingProxyType(
                {
                    "initializer_id": "current_frozen_basis_core_affine_zero_v1",
                    "rank": 8,
                    "point_weight": 100.0,
                    "global_weight": 100.0,
                    "prefix_weight": 1.0,
                    "fit_pool_size": 512,
                    "calibration_size": 64,
                    "pool_partition_seed": 85901,
                    "num_correction_pairs": 20,
                    "max_iterations": 128,
                    "gradient_tolerance": 1e-8,
                    "relative_objective_tolerance": 1e-12,
                    "max_line_search_iterations": 50,
                }
            ),
        }
    )
)


CORE_AFFINE_CG_TABLE: Mapping[str, Mapping[str, float | int | str]] = (
    MappingProxyType(
        {
            "n01_core_affine_fullpool_cg_from_l01": MappingProxyType(
                {
                    "initializer_id": "hash_bound_l01_core_affine_child_v1",
                    "initializer_result_sha256": "051d7cb8f7f6a67a0d0e4bc0328042299348a42a4312406deb1607f495bbb074",
                    "initializer_manifest_sha256": "7c7803eb9aa77ef82706469209af73d6b036c875c0932f410c8fb79113b21dcb",
                    "initializer_child_identity": "c6e6334e7f711c13f1f115f4508aaaf21d33e8c7bb62e09050eea740bf444e00",
                    "rank": 8,
                    "point_weight": 100.0,
                    "global_weight": 100.0,
                    "prefix_weight": 1.0,
                    "fit_pool_size": 512,
                    "calibration_size": 64,
                    "pool_partition_seed": 85901,
                    "max_iterations": 512,
                    "residual_tolerance": 1e-10,
                    "trace_interval": 16,
                }
            ),
        }
    )
)


CORE_AFFINE_MINIMAX_TABLE: Mapping[str, Mapping[str, float | int | str]] = (
    MappingProxyType(
        {
            "m01_core_affine_gate_max_from_n01": MappingProxyType(
                {
                    "initializer_id": "hash_bound_n01_core_affine_child_v1",
                    "initializer_result_sha256": "5de920f96bea2b473d801e304a72a7e3a3f7a1277c31302d3c41542d1e4526db",
                    "initializer_manifest_sha256": "3744cb2da72c4feeac5538282f9f7b31be1c665edbc302a1dba31151fdf4dcd1",
                    "initializer_child_identity": "efff5cf4551335d4580cdc50387c483a78bd3669a7210a080d63efc25bf8cf47",
                    "rank": 8,
                    "fit_pool_size": 512,
                    "calibration_size": 64,
                    "pool_partition_seed": 85901,
                    "temperature": 64.0,
                    "num_correction_pairs": 20,
                    "max_iterations": 256,
                    "gradient_tolerance": 1e-8,
                    "relative_objective_tolerance": 1e-12,
                    "max_line_search_iterations": 50,
                }
            ),
        }
    )
)


FULL_TT_MINIMAX_TABLE: Mapping[str, Mapping[str, float | int | str]] = (
    MappingProxyType(
        {
            "f01_full_r8_gate_max_from_n01": MappingProxyType(
                {
                    "initializer_id": "hash_bound_n01_full_rank8_child_v1",
                    "initializer_result_sha256": "5de920f96bea2b473d801e304a72a7e3a3f7a1277c31302d3c41542d1e4526db",
                    "initializer_manifest_sha256": "3744cb2da72c4feeac5538282f9f7b31be1c665edbc302a1dba31151fdf4dcd1",
                    "initializer_child_identity": "efff5cf4551335d4580cdc50387c483a78bd3669a7210a080d63efc25bf8cf47",
                    "rank": 8,
                    "position_size": 32880,
                    "fit_pool_size": 512,
                    "calibration_size": 64,
                    "pool_partition_seed": 85901,
                    "temperature": 64.0,
                    "num_correction_pairs": 20,
                    "max_iterations": 256,
                    "gradient_tolerance": 1e-8,
                    "relative_objective_tolerance": 1e-12,
                    "max_line_search_iterations": 50,
                }
            ),
        }
    )
)


RANK12_MINIMAX_TABLE: Mapping[str, Mapping[str, float | int | str]] = (
    MappingProxyType(
        {
            "r12_rank12_gate_max_from_n01": MappingProxyType(
                {
                    "initializer_id": "hash_bound_n01_rank12_child_v1",
                    "initializer_result_sha256": "5de920f96bea2b473d801e304a72a7e3a3f7a1277c31302d3c41542d1e4526db",
                    "initializer_manifest_sha256": "3744cb2da72c4feeac5538282f9f7b31be1c665edbc302a1dba31151fdf4dcd1",
                    "initializer_child_identity": "efff5cf4551335d4580cdc50387c483a78bd3669a7210a080d63efc25bf8cf47",
                    "rank": 12,
                    "position_size": 73800,
                    "fit_pool_size": 512,
                    "calibration_size": 64,
                    "pool_partition_seed": 85901,
                    "temperature": 64.0,
                    "num_correction_pairs": 20,
                    "max_iterations": 256,
                    "gradient_tolerance": 1e-8,
                    "relative_objective_tolerance": 1e-12,
                    "max_line_search_iterations": 50,
                    # The rank-12 child is constructed by exact zero-padding
                    # from the admitted rank-8 child.  The old connected-
                    # expansion controls were never consumed by the terminal
                    # retry and are retained only in the historical plan.
                }
            ),
        }
    )
)


def rotating_prefix_checkpoint_key(
    maximum_residual: float, mean_squared_residual: float, update: int
) -> tuple[float, float, int]:
    """Order feasible checkpoints by the frozen calibration rule."""

    maximum = float(maximum_residual)
    mean_squared = float(mean_squared_residual)
    update_index = int(update)
    if not math.isfinite(maximum) or maximum < 0.0:
        raise ValueError("maximum_residual must be finite and nonnegative")
    if not math.isfinite(mean_squared) or mean_squared < 0.0:
        raise ValueError("mean_squared_residual must be finite and nonnegative")
    if update_index < 0:
        raise ValueError("update must be nonnegative")
    return maximum, mean_squared, update_index


def axis_theta_rows(radius: float) -> tuple[tuple[float, float, float], ...]:
    rows: list[tuple[float, float, float]] = [(0.0, 0.0, 0.0)]
    for axis in range(3):
        positive = [0.0, 0.0, 0.0]
        negative = [0.0, 0.0, 0.0]
        positive[axis] = float(radius)
        negative[axis] = -float(radius)
        rows.extend((tuple(positive), tuple(negative)))
    return tuple(rows)


def validation_theta_rows() -> tuple[tuple[float, float, float], ...]:
    return axis_theta_rows(0.01) + axis_theta_rows(0.03)[1:]


__all__ = [
    "ARM_TABLE",
    "INITIALIZER_AUDIT_TABLE",
    "PREFIX_TANGENT_TABLE",
    "PAIR_TANGENT_TABLE",
    "DIRECT_TT_TANGENT_TABLE",
    "FULL_TT_MINIMAX_TABLE",
    "RANK12_MINIMAX_TABLE",
    "CORE_AFFINE_CG_TABLE",
    "CORE_AFFINE_LBFGS_TABLE",
    "CORE_AFFINE_MINIMAX_TABLE",
    "ROTATING_PREFIX_TANGENT_TABLE",
    "axis_theta_rows",
    "rotating_prefix_checkpoint_key",
    "validation_theta_rows",
]
