#!/usr/bin/env python3
"""Run one bounded GPU/XLA T1 centered-density arm or untouched claim."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import tensorflow as tf  # noqa: E402
import tensorflow_probability as tfp  # noqa: E402

from bayesfilter.runtime.gpu_memory_policy import (  # noqa: E402
    configure_tensorflow_gpu_memory_growth,
)


MEMORY_POLICY = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)

from bayesfilter.highdim.zhao_cui_austria_sir_centered_density_tf import (  # noqa: E402
    CenteredThetaFeatures,
    LaneBCenteredResidualChild,
    load_centered_residual_child,
)
from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_artifact_compat import (  # noqa: E402
    load_lane_b_t1_artifact_v1_compat,
)
from bayesfilter.highdim.zhao_cui_austria_sir_parameter_density_training_tf import (  # noqa: E402
    CenteredResidualTrainer,
    CoreAffineTangentTrainer,
    T1ParameterDensityBatch,
    build_t1_parameter_density_batch,
    core_affine_released_coordinate_mask,
    core_tangent_banks_from_residual_components,
    core_tangent_to_residual_component,
    embed_residual_component_with_connected_channels,
    embed_residual_component_at_rank,
    estimate_t1_prefix_scores,
    estimate_t1_ratio_score,
    fixed_rank_initial_residual_components,
    make_compiled_absolute_train_step,
    make_compiled_core_affine_gate_minimax_value_and_gradient,
    make_compiled_full_tt_gate_minimax_value_and_gradient,
    make_compiled_core_affine_total_score_value_and_gradient,
    make_compiled_origin_score_prefit_step,
    make_compiled_origin_total_score_train_step,
    rotating_prefix_minibatch_indices,
    residual_components_from_position,
    residual_components_position,
    solve_quadratic_value_gradient_with_conjugate_gradient,
    target_informed_additive_score_initialization,
    target_informed_within_region_pair_score_initialization,
)
from bayesfilter.highdim.zhao_cui_austria_sir_parameter_density_campaign_config import (  # noqa: E402
    ARM_TABLE,
    CORE_AFFINE_CG_TABLE,
    CORE_AFFINE_MINIMAX_TABLE,
    INITIALIZER_AUDIT_TABLE,
    PREFIX_TANGENT_TABLE,
    PAIR_TANGENT_TABLE,
    DIRECT_TT_TANGENT_TABLE,
    FULL_TT_MINIMAX_TABLE,
    RANK12_MINIMAX_TABLE,
    CORE_AFFINE_LBFGS_TABLE,
    ROTATING_PREFIX_TANGENT_TABLE,
    axis_theta_rows as axis_theta_row_values,
    rotating_prefix_checkpoint_key,
    validation_theta_rows as validation_theta_row_values,
)


PLAN = Path(
    "docs/plans/"
    "bayesfilter-zhao-cui-austria-sir-parameter-density-t1-campaign-plan-"
    "2026-08-01.md"
)
PARENT_DIR = ROOT / (
    "docs/plans/artifacts/zhao-cui-austria-sir-lane-b-t1-20260730/"
    "pilot-final-02/p05_r4_b5_lr3e4_l1_1e9/artifact"
)
EXPECTED_PARENT_IDENTITY = (
    "e4b56526205eb50c3d2aa3b8a8ce6ce27539aa5ab50ad286380136db28ed2b59"
)
EXPECTED_PARENT_VALUE = -31.1290512231882
SELECTION_SCHEMA = "bayesfilter.zhao_cui_austria_sir_parameter_density_selection.v1"
MEMORY_CAP_BYTES = 6 * 1024**3
TRAIN_INITIAL_SEED = 85201
TRAIN_TRANSITION_SEED = 85202
VALIDATION_INITIAL_SEED = 85301
VALIDATION_TRANSITION_SEED = 85302
VALIDATION_PREFIX_SEED = 85401
UNTOUCHED_INITIAL_SEED = 85501
UNTOUCHED_TRANSITION_SEED = 85502
UNTOUCHED_PREFIX_SEED = 85601
RESIDUAL_SEED = 85701
TRAIN_COUNT = 4096
VALIDATION_COUNT = 8192
UNTOUCHED_COUNT = 65536
UNTOUCHED_PREFIX_COUNT = 32768
TRAIN_BATCH_SIZE = 64
TRAIN_STEPS = 96
TRAIN_PREFIX_SEED = 85251
CALIBRATION_PREFIX_SEED = 85252
TRAIN_PREFIX_COUNT = 8192
SELECTED_INITIALIZER_ID = "i04_add_ridge1e4_global10"
SELECTED_INITIALIZER_RIDGE_FRACTION = 1e-4
SELECTED_INITIALIZER_GLOBAL_SCORE_WEIGHT = 10.0
RANK_EXPANSION_EPSILON = 1e-3
CORE_TANGENT_WARM_START_DIR = ROOT / (
    "docs/plans/artifacts/zhao-cui-austria-sir-lane-b-t1-score-20260731/"
    "pilot-01/s05_lr1e3_l1_1e9/artifact"
)
CORE_TANGENT_WARM_START_MANIFEST_SHA256 = (
    "131ff990b7288e6e3cccc9ea517176e23e49527e359a64e25c00c4b2b07baae4"
)
CORE_TANGENT_WARM_START_IDENTITY = (
    "a6a6c68fc4939612d2c52e129e969dc8de96b2be06fe3a8db6a2ef5591d0dc91"
)
CORE_AFFINE_LBFGS_RESULT_DIR = ROOT / (
    "docs/plans/artifacts/zhao-cui-austria-sir-parameter-density-t1-20260801/"
    "core-affine-lbfgs-v1/l01_core_affine_fullpool_lbfgs"
)
CORE_AFFINE_CG_RESULT_DIR = ROOT / (
    "docs/plans/artifacts/zhao-cui-austria-sir-parameter-density-t1-20260801/"
    "core-affine-cg-v1/n01_core_affine_fullpool_cg_from_l01"
)
CORE_TANGENT_WARM_START_CHILD_IDENTITY = (
    "2e7a7ef8236d7c64961438b0df42bb48f4f086aa1405755f18d457d1655c1b25"
)

def axis_theta_rows(radius: float) -> tf.Tensor:
    return tf.constant(axis_theta_row_values(radius), tf.float64)


def validation_theta_rows() -> tf.Tensor:
    return tf.constant(validation_theta_row_values(), tf.float64)


def _noise(count: int, initial_seed: int, transition_seed: int) -> tuple[tf.Tensor, tf.Tensor]:
    return (
        tf.random.stateless_normal([int(count), 18], [int(initial_seed), 1], dtype=tf.float64),
        tf.random.stateless_normal(
            [int(count), 18], [int(transition_seed), 1], dtype=tf.float64
        ),
    )


def _prefix_authority_reproducibility(
    output_dir: Path,
    *,
    train_count: int,
    train_prefix_count: int,
    max_seconds: float,
    started: float,
) -> Mapping[str, object]:
    """Compare independent score authorities on fixed fit/calibration points."""

    if int(train_count) < 512 + 64:
        raise ValueError("training count is too small for the reproducibility audit")
    if int(train_prefix_count) != 8192:
        raise ValueError("reproducibility audit freezes the 8,192-row authority")
    parent = _require_parent()
    train_initial_a, train_transition_a = _noise(4096, 86001, 86002)
    train_initial_b, train_transition_b = _noise(4096, 86101, 86102)
    batch_a = build_t1_parameter_density_batch(
        parent=parent,
        theta=tf.zeros([1, 3], tf.float64),
        initial_noise=train_initial_a,
        transition_noise=train_transition_a,
        role="authority_reproducibility_global_a",
    )
    batch_b = build_t1_parameter_density_batch(
        parent=parent,
        theta=tf.zeros([1, 3], tf.float64),
        initial_noise=train_initial_b,
        transition_noise=train_transition_b,
        role="authority_reproducibility_global_b",
    )
    global_a = estimate_t1_ratio_score(batch_a, theta_index=0)
    global_b = estimate_t1_ratio_score(batch_b, theta_index=0)
    partition = tf.random.experimental.stateless_shuffle(
        tf.range(int(train_count), dtype=tf.int32),
        seed=tf.constant([85901, 1], tf.int32),
    )
    indices = tf.concat([partition[:16], partition[512:528]], axis=0)
    # Freeze the point set once; only the authority draws differ between A/B.
    physical_a = tf.gather(batch_a.physical_points[0, :, :18], indices)
    conditional_a = estimate_t1_prefix_scores(
        prefix_points=physical_a,
        global_score=global_a,
        sample_count=8192,
        seed=86051,
    )
    conditional_b = estimate_t1_prefix_scores(
        prefix_points=physical_a,
        global_score=global_b,
        sample_count=8192,
        seed=86151,
    )
    scores_a = tf.stack([row.score for row in conditional_a])
    scores_b = tf.stack([row.score for row in conditional_b])
    se_a = tf.stack([row.score_standard_error for row in conditional_a])
    se_b = tf.stack([row.score_standard_error for row in conditional_b])
    ess_a = tf.stack([row.effective_sample_size for row in conditional_a])
    ess_b = tf.stack([row.effective_sample_size for row in conditional_b])
    combined_se = tf.sqrt(tf.square(se_a) + tf.square(se_b))
    z = tf.math.divide_no_nan(tf.abs(scores_a - scores_b), combined_se)
    g = tf.math.divide_no_nan(tf.abs(scores_a - scores_b), 3.0 * combined_se + 1e-5)
    authority_valid = bool(
        tf.reduce_all(ess_a >= 4096.0).numpy()
        and tf.reduce_all(ess_b >= 4096.0).numpy()
        and tf.reduce_all(se_a <= tf.constant([2.0, 1.0, 0.5], tf.float64)).numpy()
        and tf.reduce_all(se_b <= tf.constant([2.0, 1.0, 0.5], tf.float64)).numpy()
    )
    elapsed = time.monotonic() - started
    memory = tf.config.experimental.get_memory_info("GPU:0")
    passed = (
        authority_valid
        and bool(tf.reduce_all(tf.math.is_finite(z)).numpy())
        and int(memory["peak"]) <= MEMORY_CAP_BYTES
        and elapsed <= max_seconds
    )
    return {
        "schema_version": "bayesfilter.zhao_cui_austria_sir_parameter_density_prefix_authority_reproducibility.v1",
        "status": "PASS_PREFIX_AUTHORITY_REPRODUCIBILITY" if passed else "BLOCK_PREFIX_AUTHORITY_REPRODUCIBILITY",
        "classification": "diagnostic_only",
        "parent_identity": parent.identity.hash.value,
        "point_indices": indices,
        "global_a": _score_payload(global_a),
        "global_b": _score_payload(global_b),
        "authority_a": [_score_payload(row) for row in conditional_a],
        "authority_b": [_score_payload(row) for row in conditional_b],
        "paired_z": z,
        "paired_g": g,
        "paired_z_squared_mean": tf.reduce_mean(tf.square(z)),
        "paired_z_median": tfp.stats.percentile(tf.reshape(z, [-1]), 50.0),
        "paired_z_maximum": tf.reduce_max(z),
        "paired_g_fraction_at_most_one": tf.reduce_mean(
            tf.cast(g <= 1.0, tf.float64)
        ),
        "paired_g_maximum": tf.reduce_max(g),
        "authority_valid": authority_valid,
        "gpu_allocator": {key: int(value) for key, value in memory.items()},
        "gates": {
            "authority_valid": authority_valid,
            "finite_paired_diagnostics": bool(tf.reduce_all(tf.math.is_finite(z)).numpy()),
            "memory_under_6_gib": int(memory["peak"]) <= MEMORY_CAP_BYTES,
            "wall_time": elapsed <= max_seconds,
            "passed": passed,
        },
        "run_manifest": _run_manifest(started),
        "nonclaims": [
            "this diagnostic does not establish target correctness or TT representation adequacy",
            "32 points do not certify simultaneous 576-point tail calibration",
            "no density, T1, T2, HMC, or production claim",
        ],
    }


def _prefix_authority_sample_growth(
    output_dir: Path,
    *,
    train_count: int,
    max_seconds: float,
    started: float,
) -> Mapping[str, object]:
    """Measure prefix-estimator drift as conditional sample count grows."""

    if int(train_count) < 512 + 64:
        raise ValueError("training count is too small for the sample-growth audit")
    parent = _require_parent()
    initial_noise, transition_noise = _noise(4096, 86401, 86402)
    batch = build_t1_parameter_density_batch(
        parent=parent,
        theta=tf.zeros([1, 3], tf.float64),
        initial_noise=initial_noise,
        transition_noise=transition_noise,
        role="authority_sample_growth_global",
    )
    global_score = estimate_t1_ratio_score(batch, theta_index=0)
    partition = tf.random.experimental.stateless_shuffle(
        tf.range(int(train_count), dtype=tf.int32),
        seed=tf.constant([85901, 1], tf.int32),
    )
    indices = tf.concat([partition[:4], partition[512:516]], axis=0)
    points = tf.gather(batch.physical_points[0, :, :18], indices)
    small = estimate_t1_prefix_scores(
        prefix_points=points,
        global_score=global_score,
        sample_count=8192,
        seed=86251,
    )
    large = estimate_t1_prefix_scores(
        prefix_points=points,
        global_score=global_score,
        sample_count=65536,
        seed=86351,
    )
    small_score = tf.stack([row.score for row in small])
    large_score = tf.stack([row.score for row in large])
    small_se = tf.stack([row.score_standard_error for row in small])
    large_se = tf.stack([row.score_standard_error for row in large])
    small_ess = tf.stack([row.effective_sample_size for row in small])
    large_ess = tf.stack([row.effective_sample_size for row in large])
    combined_se = tf.sqrt(tf.square(small_se) + tf.square(large_se))
    z = tf.math.divide_no_nan(tf.abs(small_score - large_score), combined_se)
    drift = tf.abs(small_score - large_score)
    elapsed = time.monotonic() - started
    memory = tf.config.experimental.get_memory_info("GPU:0")
    finite = bool(tf.reduce_all(tf.math.is_finite(z)).numpy())
    valid = bool(
        tf.reduce_all(small_ess >= 4096.0).numpy()
        and tf.reduce_all(large_ess >= 32768.0).numpy()
        and tf.reduce_all(small_se <= tf.constant([2.0, 1.0, 0.5], tf.float64)).numpy()
        and tf.reduce_all(large_se <= tf.constant([2.0, 1.0, 0.5], tf.float64)).numpy()
    )
    passed = valid and finite and int(memory["peak"]) <= MEMORY_CAP_BYTES and elapsed <= max_seconds
    return {
        "schema_version": "bayesfilter.zhao_cui_austria_sir_parameter_density_prefix_authority_sample_growth.v1",
        "status": "PASS_PREFIX_AUTHORITY_SAMPLE_GROWTH" if passed else "BLOCK_PREFIX_AUTHORITY_SAMPLE_GROWTH",
        "classification": "diagnostic_only",
        "parent_identity": parent.identity.hash.value,
        "point_indices": indices,
        "global_authority": _score_payload(global_score),
        "small_sample_count": 8192,
        "large_sample_count": 65536,
        "small_authority": [_score_payload(row) for row in small],
        "large_authority": [_score_payload(row) for row in large],
        "absolute_score_drift": drift,
        "paired_z": z,
        "paired_z_squared_mean": tf.reduce_mean(tf.square(z)),
        "paired_z_maximum": tf.reduce_max(z),
        "small_ess_minimum": tf.reduce_min(small_ess),
        "large_ess_minimum": tf.reduce_min(large_ess),
        "authority_valid": valid,
        "gpu_allocator": {key: int(value) for key, value in memory.items()},
        "gates": {
            "authority_valid": valid,
            "finite_paired_diagnostics": finite,
            "memory_under_6_gib": int(memory["peak"]) <= MEMORY_CAP_BYTES,
            "wall_time": elapsed <= max_seconds,
            "passed": passed,
        },
        "run_manifest": _run_manifest(started),
        "nonclaims": [
            "sample growth does not establish exactness of either self-normalized estimate",
            "no density, T1, T2, HMC, or production claim",
        ],
    }


def _jsonable(value: object) -> object:
    if isinstance(value, tf.Tensor):
        array = value.numpy()
        return array.item() if array.ndim == 0 else array.tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, float) and not (float("-inf") < value < float("inf")):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(args: list[str]) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=False, capture_output=True, text=True
    ).stdout.strip()


def _run_manifest(started: float) -> Mapping[str, object]:
    logical = tuple(tf.config.list_logical_devices("GPU"))
    if not logical:
        raise RuntimeError("parameter-density campaign requires a logical GPU")
    return {
        "git_commit": _git(["rev-parse", "HEAD"]),
        "git_status_short": _git(["status", "--short"]).splitlines(),
        "command": sys.argv,
        "environment": sys.prefix,
        "host": platform.node(),
        "python": platform.python_version(),
        "tensorflow": tf.__version__,
        "device": [item.name for item in logical],
        "dtype": "float64",
        "tf32_enabled": bool(tf.config.experimental.tensor_float_32_execution_enabled()),
        "jit_compile": True,
        "gpu_memory_policy": dict(MEMORY_POLICY),
        "plan": PLAN.as_posix(),
        "wall_time_seconds": time.monotonic() - started,
        "trust_basis": "owner_designated_managed_session_visible_gpu_trusted",
        "random_seeds": {
            "training_initial": TRAIN_INITIAL_SEED,
            "training_transition": TRAIN_TRANSITION_SEED,
            "training_prefix": TRAIN_PREFIX_SEED,
            "training_prefix_count": TRAIN_PREFIX_COUNT,
            "calibration_prefix": CALIBRATION_PREFIX_SEED,
            "validation_initial": VALIDATION_INITIAL_SEED,
            "validation_transition": VALIDATION_TRANSITION_SEED,
            "validation_prefix": VALIDATION_PREFIX_SEED,
            "untouched_initial": UNTOUCHED_INITIAL_SEED,
            "untouched_transition": UNTOUCHED_TRANSITION_SEED,
            "untouched_prefix": UNTOUCHED_PREFIX_SEED,
            "residual_initialization": RESIDUAL_SEED,
        },
        "source_sha256": {
            PLAN.as_posix(): _sha256(ROOT / PLAN),
            Path(__file__).resolve().relative_to(ROOT).as_posix(): _sha256(Path(__file__).resolve()),
            "bayesfilter/highdim/zhao_cui_austria_sir_parameter_density_training_tf.py": _sha256(
                ROOT / "bayesfilter/highdim/zhao_cui_austria_sir_parameter_density_training_tf.py"
            ),
            "bayesfilter/highdim/zhao_cui_austria_sir_centered_density_tf.py": _sha256(
                ROOT / "bayesfilter/highdim/zhao_cui_austria_sir_centered_density_tf.py"
            ),
            "bayesfilter/highdim/zhao_cui_austria_sir_parameter_density_campaign_config.py": _sha256(
                ROOT / "bayesfilter/highdim/zhao_cui_austria_sir_parameter_density_campaign_config.py"
            ),
            "scripts/select_zhao_cui_austria_sir_parameter_density_t1.py": _sha256(
                ROOT / "scripts/select_zhao_cui_austria_sir_parameter_density_t1.py"
            ),
            PARENT_DIR.relative_to(ROOT).joinpath("manifest.json").as_posix(): _sha256(
                PARENT_DIR / "manifest.json"
            ),
        },
    }


def _require_parent():
    parent = load_lane_b_t1_artifact_v1_compat(PARENT_DIR)
    if parent.identity.hash.value != EXPECTED_PARENT_IDENTITY:
        raise RuntimeError("admitted T1 parent identity drift")
    if abs(float(parent.value()) - EXPECTED_PARENT_VALUE) > 2e-13:
        raise RuntimeError("admitted T1 parent value drift")
    return parent


def _load_core_tangent_warm_start(
    parent,
) -> tuple[tuple[tuple[tf.Tensor, ...], ...], Mapping[str, object]]:
    manifest_path = CORE_TANGENT_WARM_START_DIR / "manifest.json"
    manifest_sha256 = _sha256(manifest_path)
    if manifest_sha256 != CORE_TANGENT_WARM_START_MANIFEST_SHA256:
        raise ValueError("core-tangent warm-start manifest hash mismatch")
    payload = json.loads(manifest_path.read_text())
    if (
        payload.get("schema_version")
        != "bayesfilter.zhao_cui_austria_sir_lane_b_t1_score_artifact.v1"
        or payload.get("parent_identity") != parent.identity.hash.value
        or payload.get("identity_sha256") != CORE_TANGENT_WARM_START_IDENTITY
        or payload.get("child_identity") != CORE_TANGENT_WARM_START_CHILD_IDENTITY
        or payload.get("config", {}).get("arm_id") != "s05_lr1e3_l1_1e9"
    ):
        raise ValueError("core-tangent warm-start manifest fields mismatch")
    tensors = payload.get("tensors")
    if not isinstance(tensors, Mapping):
        raise ValueError("core-tangent warm-start tensor ledger missing")
    banks = []
    tensor_ledger = []
    for axis, parent_core in enumerate(parent.cores):
        bank = []
        for parameter in range(3):
            name = f"tangent_{axis:02d}_{parameter}"
            row = tensors.get(name)
            if not isinstance(row, Mapping) or row.get("path") != f"{name}.tensor":
                raise ValueError(f"core-tangent warm-start tensor missing: {name}")
            path = CORE_TANGENT_WARM_START_DIR / str(row["path"])
            serialized = tf.io.read_file(path.as_posix())
            realized_sha256 = hashlib.sha256(bytes(serialized.numpy())).hexdigest()
            if realized_sha256 != row.get("sha256"):
                raise ValueError(f"core-tangent warm-start tensor hash mismatch: {name}")
            value = tf.io.parse_tensor(
                serialized, out_type=tf.dtypes.as_dtype(str(row["dtype"]))
            )
            value = tf.ensure_shape(value, row["shape"])
            if value.dtype != tf.float64 or value.shape != parent_core.shape:
                raise ValueError(f"core-tangent warm-start tensor shape mismatch: {name}")
            tf.debugging.assert_all_finite(value, f"core-tangent warm-start {name}")
            bank.append(value)
            tensor_ledger.append(
                {
                    "name": name,
                    "path": path.relative_to(ROOT).as_posix(),
                    "sha256": realized_sha256,
                    "shape": value.shape.as_list(),
                    "dtype": value.dtype.name,
                }
            )
        banks.append(tuple(bank))
    return tuple(banks), {
        "classification": "historical_same_parent_warm_start_only",
        "manifest_path": manifest_path.relative_to(ROOT).as_posix(),
        "manifest_sha256": manifest_sha256,
        "historical_identity": payload["identity_sha256"],
        "historical_child_identity": payload["child_identity"],
        "historical_source_closure_stale_and_not_admitted": True,
        "historical_gauge_used": False,
        "tensor_ledger": tensor_ledger,
    }


def _zero_residuals(parent, features: CenteredThetaFeatures):
    rows = []
    for _ in range(features.feature_count):
        component = [tf.identity(core) for core in parent.cores]
        component[0] = tf.zeros_like(component[0])
        rows.append(tuple(component))
    return tuple(rows)


def _selected_training_only_initial_components(
    *,
    parent,
    features: CenteredThetaFeatures,
    training: T1ParameterDensityBatch,
    target_rank: int,
):
    fitted = target_informed_additive_score_initialization(
        parent=parent,
        local_points=training.local_points[0],
        target_complete_data_score=training.complete_data_score[0],
        importance_log_weight=training.observation_log_density[0],
        ridge_fraction=SELECTED_INITIALIZER_RIDGE_FRACTION,
        global_score_weight=SELECTED_INITIALIZER_GLOBAL_SCORE_WEIGHT,
    )
    if int(target_rank) == 2:
        components = fitted.residual_components
    elif int(target_rank) > 2:
        components = tuple(
            embed_residual_component_with_connected_channels(
                component,
                target_rank=int(target_rank),
                seed=RESIDUAL_SEED + 104729 * component_index,
                seeded_channel_epsilon=RANK_EXPANSION_EPSILON,
            )
            for component_index, component in enumerate(fitted.residual_components)
        )
    else:
        raise ValueError("selected initializer supports residual rank at least two")
    return components, {
        "selected_initializer_id": SELECTED_INITIALIZER_ID,
        "ridge_fraction": fitted.ridge_fraction,
        "global_score_weight": fitted.global_score_weight,
        "target_rank": int(target_rank),
        "rank_expansion_epsilon": (
            RANK_EXPANSION_EPSILON if int(target_rank) > 2 else 0.0
        ),
        "realized_ridge": fitted.realized_ridge,
        "coefficient_rms": fitted.coefficient_rms,
    }


def _score_payload(estimate) -> Mapping[str, object]:
    return {
        "value": estimate.value,
        "score": estimate.score,
        "score_standard_error": estimate.score_standard_error,
        "effective_sample_size": estimate.effective_sample_size,
    }


def _compiled_child_evaluator(child: LaneBCenteredResidualChild, prefix_points: tf.Tensor):
    @tf.function(jit_compile=True, reduce_retracing=True)
    def evaluate() -> tuple[tf.Tensor, ...]:
        value, score = child.increment_and_score(tf.zeros([3], tf.float64))
        prefix_value, prefix_score = child.prefix_log_marginal_and_score(
            tf.zeros([3], tf.float64), prefix_points
        )
        return value, score, prefix_value, prefix_score

    return evaluate


def _evaluate_validation(
    trainer: CenteredResidualTrainer,
    validation: T1ParameterDensityBatch,
    *,
    prefix_count: int,
) -> Mapping[str, object]:
    child = trainer.freeze_child()
    baseline = CenteredResidualTrainer(
        trainer.parent,
        features=trainer.features,
        initial_residual_components=_zero_residuals(trainer.parent, trainer.features),
    )
    child_metrics = trainer.heldout_metrics(validation)
    baseline_metrics = baseline.heldout_metrics(validation)
    non_origin = slice(1, None)
    paired_shape_ratio = (
        child_metrics["normalized_log_density_rms"][non_origin]
        / tf.maximum(
            baseline_metrics["normalized_log_density_rms"][non_origin],
            tf.constant(1e-12, tf.float64),
        )
    )
    mass_tolerance = (
        3.0 * child_metrics["target_log_mass_standard_error"]
        + tf.constant(0.05, tf.float64)
    )
    mass_z = child_metrics["absolute_log_mass_error"] / mass_tolerance
    point = trainer.origin_point_score_metrics_arrays(
        validation.local_points[0],
        validation.complete_data_score[0],
        validation.observation_log_density[0],
    )
    fisher = estimate_t1_ratio_score(validation, theta_index=0)
    child_value, child_score = child.increment_and_score(tf.zeros([3], tf.float64))
    likelihood_tolerance = 3.0 * fisher.score_standard_error + tf.constant(1e-5, tf.float64)
    likelihood_z = tf.abs(child_score - fisher.score) / likelihood_tolerance
    prefix_authority = estimate_t1_prefix_scores(
        prefix_points=validation.physical_points[0, :1, :18],
        global_score=fisher,
        sample_count=int(prefix_count),
        seed=VALIDATION_PREFIX_SEED,
    )[0]
    prefix_value, prefix_score = child.prefix_log_marginal_and_score(
        tf.zeros([3], tf.float64), validation.local_points[0, :1, :18]
    )
    prefix_tolerance = 3.0 * prefix_authority.score_standard_error + tf.constant(
        1e-5, tf.float64
    )
    prefix_z = tf.abs(prefix_score[0] - prefix_authority.score) / prefix_tolerance
    shape_gate = tf.logical_and(
        tf.reduce_mean(paired_shape_ratio) <= tf.constant(0.95, tf.float64),
        tf.reduce_max(paired_shape_ratio) <= tf.constant(1.05, tf.float64),
    )
    point_gate = tf.reduce_all(
        point["normalized_score_residual_rms"] <= tf.constant(0.90, tf.float64)
    )
    prefix_valid = tf.logical_and(
        prefix_authority.effective_sample_size >= tf.constant(0.5 * prefix_count, tf.float64),
        tf.reduce_all(
            prefix_authority.score_standard_error
            <= tf.constant([2.0, 1.0, 0.5], tf.float64)
        ),
    )
    gates = {
        "origin_value": tf.abs(child_value - trainer.parent.value()) <= tf.constant(2e-13, tf.float64),
        "mass": tf.reduce_all(mass_z <= 1.0),
        "shape": shape_gate,
        "point_score": point_gate,
        "likelihood_score": tf.reduce_all(likelihood_z <= 1.0),
        "prefix_authority_valid": prefix_valid,
        "prefix_score": tf.reduce_all(prefix_z <= 1.0),
    }
    return {
        "child_origin_value": child_value,
        "child_origin_score": child_score,
        "heldout": child_metrics,
        "parent_baseline_heldout": baseline_metrics,
        "paired_nonorigin_shape_ratio": paired_shape_ratio,
        "mass_standardized_residual": mass_z,
        "point_score": point,
        "fisher": _score_payload(fisher),
        "likelihood_standardized_residual": likelihood_z,
        "prefix_child_log_value": prefix_value,
        "prefix_child_score": prefix_score,
        "prefix_authority": _score_payload(prefix_authority),
        "prefix_standardized_residual": prefix_z,
        "gates": gates,
        "selector_metrics": {
            "maximum_standardized_score_residual": tf.reduce_max(
                tf.concat([likelihood_z, prefix_z], axis=0)
            ),
            "mean_paired_shape_ratio": tf.reduce_mean(paired_shape_ratio),
            "maximum_mass_standardized_residual": tf.reduce_max(mass_z),
        },
    }


def _capacity(output_dir: Path, max_seconds: float, started: float) -> Mapping[str, object]:
    parent = _require_parent()
    features = CenteredThetaFeatures()
    initial, transition = _noise(64, 85101, 85102)
    batch = build_t1_parameter_density_batch(
        parent=parent,
        theta=axis_theta_rows(0.01),
        initial_noise=initial,
        transition_noise=transition,
        role="capacity_probe",
    )
    initial_components, initializer = _selected_training_only_initial_components(
        parent=parent,
        features=features,
        training=batch,
        target_rank=4,
    )
    trainer = CenteredResidualTrainer(
        parent,
        features=features,
        initial_residual_components=initial_components,
    )
    step = make_compiled_absolute_train_step(
        trainer,
        tf.keras.optimizers.Adam(learning_rate=3e-4),
        l1_weight=1e-9,
        l2_weight=1e-10,
        derivative_weight=0.1,
        gradient_clip_norm=100.0,
    )
    terms = step(
        batch.theta,
        batch.local_points,
        parent.shift_constant + batch.observation_log_density,
        batch.local_points[0],
        batch.complete_data_score[0],
        batch.observation_log_density[0],
    )
    elapsed = time.monotonic() - started
    memory = tf.config.experimental.get_memory_info("GPU:0")
    finite = tf.reduce_all(
        tf.stack([tf.reduce_all(tf.math.is_finite(value)) for value in terms])
    )
    passed = bool(finite.numpy()) and int(memory["peak"]) <= MEMORY_CAP_BYTES and elapsed <= max_seconds
    return {
        "schema_version": "bayesfilter.zhao_cui_austria_sir_parameter_density_capacity.v1",
        "status": "PASS_T1_PARAMETER_DENSITY_CAPACITY" if passed else "BLOCK_T1_PARAMETER_DENSITY_CAPACITY",
        "parent_identity": parent.identity.hash.value,
        "terms": terms,
        "initializer": initializer,
        "gpu_allocator": {key: int(value) for key, value in memory.items()},
        "gates": {
            "finite": finite,
            "memory_under_6_gib": int(memory["peak"]) <= MEMORY_CAP_BYTES,
            "wall_time": elapsed <= max_seconds,
            "passed": passed,
        },
        "run_manifest": _run_manifest(started),
    }


def _initializer_audit(
    audit_id: str,
    output_dir: Path,
    *,
    train_count: int,
    max_seconds: float,
    started: float,
) -> Mapping[str, object]:
    del output_dir
    if audit_id not in INITIALIZER_AUDIT_TABLE:
        raise ValueError(f"unknown initializer audit arm: {audit_id}")
    arm = INITIALIZER_AUDIT_TABLE[audit_id]
    parent = _require_parent()
    features = CenteredThetaFeatures()
    initial, transition = _noise(
        train_count, TRAIN_INITIAL_SEED, TRAIN_TRANSITION_SEED
    )
    training = build_t1_parameter_density_batch(
        parent=parent,
        theta=tf.zeros([1, 3], tf.float64),
        initial_noise=initial,
        transition_noise=transition,
        role="training_only_initializer_audit",
    )
    trace = []
    initialization_metadata: Mapping[str, object]
    family = str(arm["family"])
    if family == "connected_random_score_prefit":
        trainer = CenteredResidualTrainer(
            parent,
            features=features,
            initial_residual_components=fixed_rank_initial_residual_components(
                parent=parent,
                features=features,
                rank=int(arm["rank"]),
                seed=RESIDUAL_SEED,
                amplitude_scale=float(arm["amplitude_scale"]),
                perturbation_scale=float(arm["perturbation_scale"]),
            ),
        )
        before = trainer.origin_point_score_metrics_arrays(
            training.local_points[0],
            training.complete_data_score[0],
            training.observation_log_density[0],
        )
        step = make_compiled_origin_score_prefit_step(
            trainer,
            tf.keras.optimizers.Adam(learning_rate=float(arm["learning_rate"])),
            gradient_clip_norm=100.0,
        )
        for update in range(int(arm["steps"])):
            if time.monotonic() - started > max_seconds:
                raise TimeoutError("initializer audit exceeded its wall-time cap")
            terms = step(
                training.local_points[0],
                training.complete_data_score[0],
                training.observation_log_density[0],
            )
            if update in {0, int(arm["steps"]) - 1} or (update + 1) % 8 == 0:
                trace.append(
                    {
                        "update": update + 1,
                        "loss_before_update": terms[0],
                        "target_likelihood_score": terms[1],
                        "child_likelihood_score_before_update": terms[2],
                        "target_point_score_rms": terms[3],
                        "score_residual_rms_before_update": terms[4],
                        "normalized_score_residual_rms_before_update": terms[5],
                        "child_point_score_standard_deviation_before_update": terms[6],
                        "importance_effective_sample_size": terms[7],
                        "gradient_norm": terms[8],
                        "maximum_core_magnitude": terms[9],
                    }
                )
        initialization_metadata = {
            "family": family,
            "seed": RESIDUAL_SEED,
        }
    elif family == "exact_additive_score_ridge":
        baseline = CenteredResidualTrainer(
            parent,
            features=features,
            initial_residual_components=_zero_residuals(parent, features),
        )
        before = baseline.origin_point_score_metrics_arrays(
            training.local_points[0],
            training.complete_data_score[0],
            training.observation_log_density[0],
        )
        fitted = target_informed_additive_score_initialization(
            parent=parent,
            local_points=training.local_points[0],
            target_complete_data_score=training.complete_data_score[0],
            importance_log_weight=training.observation_log_density[0],
            ridge_fraction=float(arm["ridge_fraction"]),
            global_score_weight=float(arm["global_score_weight"]),
        )
        trainer = CenteredResidualTrainer(
            parent,
            features=features,
            initial_residual_components=fitted.residual_components,
        )
        initialization_metadata = {
            "family": family,
            "ridge_fraction": fitted.ridge_fraction,
            "global_score_weight": fitted.global_score_weight,
            "realized_ridge": fitted.realized_ridge,
            "target_likelihood_score": fitted.target_likelihood_score,
            "target_point_score_rms": fitted.target_point_score_rms,
            "design_column_rms": fitted.design_column_rms,
            "coefficient_rms": fitted.coefficient_rms,
        }
    else:
        raise ValueError(f"unsupported initializer audit family: {family}")
    after = trainer.origin_point_score_metrics_arrays(
        training.local_points[0],
        training.complete_data_score[0],
        training.observation_log_density[0],
    )
    fisher = estimate_t1_ratio_score(training, theta_index=0)
    child_value, child_score = trainer.freeze_child().increment_and_score(
        tf.zeros([3], tf.float64)
    )
    finite = tf.reduce_all(
        tf.stack(
            [
                tf.reduce_all(tf.math.is_finite(value))
                for value in (
                    before["loss"],
                    after["loss"],
                    after["normalized_score_residual_rms"],
                    after["child_point_score_standard_deviation"],
                    child_value,
                    child_score,
                    fisher.score_standard_error,
                )
            ]
        )
    )
    loss_reduction = after["loss"] / before["loss"]
    global_score_relative_error = tf.abs(
        after["child_likelihood_score"] - after["target_likelihood_score"]
    ) / tf.maximum(tf.abs(after["target_likelihood_score"]), tf.ones([3], tf.float64))
    global_score_standardized_residual = tf.abs(
        after["child_likelihood_score"] - fisher.score
    ) / (3.0 * fisher.score_standard_error + tf.constant(1e-5, tf.float64))
    fisher_mcse_informative = tf.reduce_all(
        fisher.score_standard_error
        <= tf.constant([2.0, 1.0, 0.5], tf.float64)
    )
    global_score_gate = tf.reduce_max(global_score_relative_error) <= tf.constant(
        0.50, tf.float64
    )
    rms_gate = tf.reduce_all(
        after["normalized_score_residual_rms"] <= tf.constant(0.90, tf.float64)
    )
    nonconstant_gate = tf.reduce_all(
        after["child_point_score_standard_deviation"]
        >= tf.constant(0.05, tf.float64) * after["target_point_score_rms"]
    )
    ess_gate = after["importance_effective_sample_size"] >= tf.constant(
        0.5 * int(train_count), tf.float64
    )
    value_gate = tf.abs(child_value - parent.value()) <= tf.constant(
        2e-13, tf.float64
    )
    elapsed = time.monotonic() - started
    memory = tf.config.experimental.get_memory_info("GPU:0")
    passed = (
        bool(finite.numpy())
        and float(loss_reduction) <= 0.80
        and bool(rms_gate.numpy())
        and bool(nonconstant_gate.numpy())
        and bool(global_score_gate.numpy())
        and bool(ess_gate.numpy())
        and bool(value_gate.numpy())
        and int(memory["peak"]) <= MEMORY_CAP_BYTES
        and elapsed <= max_seconds
    )
    return {
        "schema_version": "bayesfilter.zhao_cui_austria_sir_parameter_density_initializer_audit.v1",
        "status": (
            "VIABLE_TRAINING_ONLY_INITIALIZER"
            if passed
            else "REJECTED_TRAINING_ONLY_INITIALIZER"
        ),
        "audit_id": audit_id,
        "arm": dict(arm),
        "parent_identity": parent.identity.hash.value,
        "parent_value": parent.value(),
        "initial_metrics": before,
        "final_metrics": after,
        "origin_child_value": child_value,
        "origin_child_score": child_score,
        "loss_ratio": loss_reduction,
        "global_score_relative_error": global_score_relative_error,
        "global_score_standardized_residual": global_score_standardized_residual,
        "training_fisher": _score_payload(fisher),
        "initialization_metadata": initialization_metadata,
        "trace": trace,
        "training_role": training.role,
        "gpu_allocator": {key: int(value) for key, value in memory.items()},
        "gates": {
            "finite": finite,
            "loss_reduction": loss_reduction <= tf.constant(0.80, tf.float64),
            "coordinatewise_normalized_rms": rms_gate,
            "nonconstant_score_field": nonconstant_gate,
            "global_score": global_score_gate,
            "fisher_mcse_informative": fisher_mcse_informative,
            "importance_ess": ess_gate,
            "origin_value": value_gate,
            "memory_under_6_gib": int(memory["peak"]) <= MEMORY_CAP_BYTES,
            "wall_time": elapsed <= max_seconds,
            "passed": passed,
        },
        "run_manifest": _run_manifest(started),
        "nonclaims": [
            "training-only initializer audit is not validation or selection evidence",
            "no T1 score admission, T2, HMC, superiority, or production claim",
        ],
    }


def _selected_initializer_validation(
    output_dir: Path,
    *,
    train_count: int,
    validation_count: int,
    validation_prefix_count: int,
    max_seconds: float,
    started: float,
) -> Mapping[str, object]:
    del output_dir
    parent = _require_parent()
    features = CenteredThetaFeatures()
    train_initial, train_transition = _noise(
        train_count, TRAIN_INITIAL_SEED, TRAIN_TRANSITION_SEED
    )
    training = build_t1_parameter_density_batch(
        parent=parent,
        theta=tf.zeros([1, 3], tf.float64),
        initial_noise=train_initial,
        transition_noise=train_transition,
        role="training_only_selected_initializer_fit",
    )
    components, initializer = _selected_training_only_initial_components(
        parent=parent,
        features=features,
        training=training,
        target_rank=2,
    )
    trainer = CenteredResidualTrainer(
        parent, features=features, initial_residual_components=components
    )
    valid_initial, valid_transition = _noise(
        validation_count, VALIDATION_INITIAL_SEED, VALIDATION_TRANSITION_SEED
    )
    validation = build_t1_parameter_density_batch(
        parent=parent,
        theta=validation_theta_rows(),
        initial_noise=valid_initial,
        transition_noise=valid_transition,
        role="validation_only_selected_initializer_diagnostic",
    )
    result = _evaluate_validation(
        trainer, validation, prefix_count=int(validation_prefix_count)
    )
    elapsed = time.monotonic() - started
    memory = tf.config.experimental.get_memory_info("GPU:0")
    validation_gates = result["gates"]
    assert isinstance(validation_gates, Mapping)
    score_gates = all(
        bool(tf.convert_to_tensor(validation_gates[name]).numpy())
        for name in (
            "origin_value",
            "point_score",
            "likelihood_score",
            "prefix_authority_valid",
            "prefix_score",
        )
    )
    passed = (
        score_gates
        and int(memory["peak"]) <= MEMORY_CAP_BYTES
        and elapsed <= max_seconds
    )
    return {
        "schema_version": "bayesfilter.zhao_cui_austria_sir_parameter_density_initializer_validation.v1",
        "status": (
            "PASS_SELECTED_INITIALIZER_VALIDATION_SCORE_DIAGNOSTIC"
            if passed
            else "REJECT_SELECTED_INITIALIZER_VALIDATION_SCORE_DIAGNOSTIC"
        ),
        "parent_identity": parent.identity.hash.value,
        "initializer": initializer,
        "training_role": training.role,
        "validation_role": validation.role,
        "validation": result,
        "gpu_allocator": {key: int(value) for key, value in memory.items()},
        "gates": {
            "score_subset": score_gates,
            "memory_under_6_gib": int(memory["peak"]) <= MEMORY_CAP_BYTES,
            "wall_time": elapsed <= max_seconds,
            "passed": passed,
        },
        "run_manifest": _run_manifest(started),
        "nonclaims": [
            "validation diagnostic cannot select a scientific child or consume untouched data",
            "mass and off-origin shape remain separate gates",
            "no T1 admission, T2, HMC, superiority, or production claim",
        ],
    }


def _prefix_tangent_diagnostic(
    tangent_id: str,
    output_dir: Path,
    *,
    train_count: int,
    validation_count: int,
    train_prefix_count: int,
    validation_prefix_count: int,
    max_seconds: float,
    started: float,
) -> Mapping[str, object]:
    del output_dir
    if tangent_id not in PREFIX_TANGENT_TABLE:
        raise ValueError(f"unknown prefix tangent arm: {tangent_id}")
    arm = PREFIX_TANGENT_TABLE[tangent_id]
    prefix_point_count = int(arm.get("prefix_point_count", 3))
    global_score_weight = float(
        arm.get("global_score_weight", SELECTED_INITIALIZER_GLOBAL_SCORE_WEIGHT)
    )
    parent = _require_parent()
    features = CenteredThetaFeatures()
    train_initial, train_transition = _noise(
        train_count, TRAIN_INITIAL_SEED, TRAIN_TRANSITION_SEED
    )
    training = build_t1_parameter_density_batch(
        parent=parent,
        theta=tf.zeros([1, 3], tf.float64),
        initial_noise=train_initial,
        transition_noise=train_transition,
        role="training_only_prefix_tangent_fit",
    )
    fisher = estimate_t1_ratio_score(training, theta_index=0)
    physical_prefix = training.physical_points[0, :prefix_point_count, :18]
    local_prefix = training.local_points[0, :prefix_point_count, :18]
    authorities = estimate_t1_prefix_scores(
        prefix_points=physical_prefix,
        global_score=fisher,
        sample_count=int(train_prefix_count),
        seed=TRAIN_PREFIX_SEED,
    )
    prefix_score = tf.stack([row.score for row in authorities])
    prefix_se = tf.stack([row.score_standard_error for row in authorities])
    fitted = target_informed_additive_score_initialization(
        parent=parent,
        local_points=training.local_points[0],
        target_complete_data_score=training.complete_data_score[0],
        importance_log_weight=training.observation_log_density[0],
        ridge_fraction=SELECTED_INITIALIZER_RIDGE_FRACTION,
        global_score_weight=global_score_weight,
        prefix_local_points=local_prefix,
        prefix_target_score=prefix_score,
        prefix_score_standard_error=prefix_se,
        prefix_weight=float(arm["prefix_weight"]),
    )
    trainer = CenteredResidualTrainer(
        parent,
        features=features,
        initial_residual_components=fitted.residual_components,
    )
    training_point = trainer.origin_point_score_metrics_arrays(
        training.local_points[0],
        training.complete_data_score[0],
        training.observation_log_density[0],
    )
    child = trainer.freeze_child()
    _training_prefix_value, training_prefix_child_score = (
        child.prefix_log_marginal_and_score(tf.zeros([3], tf.float64), local_prefix)
    )
    training_global_tolerance = (
        3.0 * fisher.score_standard_error + tf.constant(1e-5, tf.float64)
    )
    training_global_residual = tf.abs(
        training_point["child_likelihood_score"] - fisher.score
    ) / training_global_tolerance
    prefix_validity = tf.stack(
        [
            tf.logical_and(
                row.effective_sample_size
                >= tf.constant(0.5 * int(train_prefix_count), tf.float64),
                tf.reduce_all(
                    row.score_standard_error
                    <= tf.constant([2.0, 1.0, 0.5], tf.float64)
                ),
            )
            for row in authorities
        ]
    )
    training_gates = {
        "point_score": tf.reduce_all(
            training_point["normalized_score_residual_rms"]
            <= tf.constant(0.90, tf.float64)
        ),
        "global_score": tf.reduce_all(training_global_residual <= 1.0),
        "prefix_authority_valid": tf.reduce_all(prefix_validity),
        "prefix_score": tf.reduce_all(
            fitted.training_prefix_score_standardized_residual <= 1.0
        ),
    }


def _pair_tangent_diagnostic(
    tangent_id: str,
    output_dir: Path,
    *,
    train_count: int,
    validation_count: int,
    train_prefix_count: int,
    validation_prefix_count: int,
    max_seconds: float,
    started: float,
) -> Mapping[str, object]:
    del output_dir
    if tangent_id not in PAIR_TANGENT_TABLE:
        raise ValueError(f"unknown pair tangent arm: {tangent_id}")
    arm = PAIR_TANGENT_TABLE[tangent_id]
    parent = _require_parent()
    features = CenteredThetaFeatures()
    train_initial, train_transition = _noise(
        train_count, TRAIN_INITIAL_SEED, TRAIN_TRANSITION_SEED
    )
    training = build_t1_parameter_density_batch(
        parent=parent,
        theta=tf.zeros([1, 3], tf.float64),
        initial_noise=train_initial,
        transition_noise=train_transition,
        role="training_only_pair_tangent_fit",
    )
    fisher = estimate_t1_ratio_score(training, theta_index=0)
    point_count = int(arm["prefix_point_count"])
    physical_prefix = training.physical_points[0, :point_count, :18]
    local_prefix = training.local_points[0, :point_count, :18]
    authorities = estimate_t1_prefix_scores(
        prefix_points=physical_prefix,
        global_score=fisher,
        sample_count=int(train_prefix_count),
        seed=TRAIN_PREFIX_SEED,
    )
    prefix_score = tf.stack([row.score for row in authorities])
    prefix_se = tf.stack([row.score_standard_error for row in authorities])
    fitted = target_informed_within_region_pair_score_initialization(
        parent=parent,
        local_points=training.local_points[0],
        target_complete_data_score=training.complete_data_score[0],
        importance_log_weight=training.observation_log_density[0],
        ridge_fraction=float(arm["ridge_fraction"]),
        global_score_weight=float(arm["global_score_weight"]),
        prefix_local_points=local_prefix,
        prefix_target_score=prefix_score,
        prefix_score_standard_error=prefix_se,
        prefix_weight=float(arm["prefix_weight"]),
    )
    trainer = CenteredResidualTrainer(
        parent,
        features=features,
        initial_residual_components=fitted.residual_components,
    )
    training_point = trainer.origin_point_score_metrics_arrays(
        training.local_points[0],
        training.complete_data_score[0],
        training.observation_log_density[0],
    )
    training_global_residual = tf.abs(
        training_point["child_likelihood_score"] - fisher.score
    ) / (3.0 * fisher.score_standard_error + tf.constant(1e-5, tf.float64))
    prefix_validity = tf.stack(
        [
            tf.logical_and(
                row.effective_sample_size
                >= tf.constant(0.5 * int(train_prefix_count), tf.float64),
                tf.reduce_all(
                    row.score_standard_error
                    <= tf.constant([2.0, 1.0, 0.5], tf.float64)
                ),
            )
            for row in authorities
        ]
    )
    training_gates = {
        "point_score": tf.reduce_all(
            training_point["normalized_score_residual_rms"] <= 0.90
        ),
        "global_score": tf.reduce_all(training_global_residual <= 1.0),
        "prefix_authority_valid": tf.reduce_all(prefix_validity),
        "prefix_score": tf.reduce_all(
            fitted.training_prefix_score_standardized_residual <= 1.0
        ),
    }


def _direct_tt_tangent_diagnostic(
    tangent_id: str,
    output_dir: Path,
    *,
    train_count: int,
    validation_count: int,
    train_prefix_count: int,
    validation_prefix_count: int,
    max_seconds: float,
    started: float,
) -> Mapping[str, object]:
    if tangent_id not in DIRECT_TT_TANGENT_TABLE:
        raise ValueError(f"unknown direct TT tangent arm: {tangent_id}")
    arm = DIRECT_TT_TANGENT_TABLE[tangent_id]
    parent = _require_parent()
    features = CenteredThetaFeatures()
    train_initial, train_transition = _noise(
        train_count, TRAIN_INITIAL_SEED, TRAIN_TRANSITION_SEED
    )
    training = build_t1_parameter_density_batch(
        parent=parent,
        theta=tf.zeros([1, 3], tf.float64),
        initial_noise=train_initial,
        transition_noise=train_transition,
        role="training_only_direct_tt_tangent",
    )
    fisher = estimate_t1_ratio_score(training, theta_index=0)
    fit_physical = training.physical_points[0, :64, :18]
    fit_local = training.local_points[0, :64, :18]
    calibration_physical = training.physical_points[0, 64:128, :18]
    calibration_local = training.local_points[0, 64:128, :18]
    fit_authorities = estimate_t1_prefix_scores(
        prefix_points=fit_physical,
        global_score=fisher,
        sample_count=int(train_prefix_count),
        seed=TRAIN_PREFIX_SEED,
    )
    calibration_authorities = estimate_t1_prefix_scores(
        prefix_points=calibration_physical,
        global_score=fisher,
        sample_count=int(train_prefix_count),
        seed=CALIBRATION_PREFIX_SEED,
    )
    fit_score = tf.stack([row.score for row in fit_authorities])
    fit_se = tf.stack([row.score_standard_error for row in fit_authorities])
    calibration_score = tf.stack([row.score for row in calibration_authorities])
    calibration_se = tf.stack(
        [row.score_standard_error for row in calibration_authorities]
    )
    base = target_informed_additive_score_initialization(
        parent=parent,
        local_points=training.local_points[0],
        target_complete_data_score=training.complete_data_score[0],
        importance_log_weight=training.observation_log_density[0],
        ridge_fraction=SELECTED_INITIALIZER_RIDGE_FRACTION,
        global_score_weight=100.0,
        prefix_local_points=training.local_points[0, :16, :18],
        prefix_target_score=tf.stack([row.score for row in fit_authorities[:16]]),
        prefix_score_standard_error=tf.stack(
            [row.score_standard_error for row in fit_authorities[:16]]
        ),
        prefix_weight=0.01,
    )
    rank = int(arm["rank"])
    initial_components = tuple(
        embed_residual_component_with_connected_channels(
            component,
            target_rank=rank,
            seed=RESIDUAL_SEED + 104729 * component_index,
            seeded_channel_epsilon=RANK_EXPANSION_EPSILON,
        )
        for component_index, component in enumerate(base.residual_components)
    )
    trainer = CenteredResidualTrainer(
        parent, features=features, initial_residual_components=initial_components
    )
    step = make_compiled_origin_total_score_train_step(
        trainer,
        tf.keras.optimizers.Adam(learning_rate=float(arm["learning_rate"])),
        point_weight=float(arm.get("point_weight", 1.0)),
        global_weight=float(arm.get("global_weight", 1.0)),
        prefix_weight=float(arm.get("prefix_weight", 1.0)),
        l2_weight=1e-10,
        gradient_clip_norm=100.0,
    )
    trace = []
    for update in range(int(arm["steps"])):
        if time.monotonic() - started > max_seconds:
            raise TimeoutError("direct TT tangent arm exceeded its wall-time cap")
        terms = step(
            training.local_points[0],
            training.complete_data_score[0],
            training.observation_log_density[0],
            fisher.score,
            fisher.score_standard_error,
            fit_local,
            fit_score,
            fit_se,
        )
        if update in {0, int(arm["steps"]) - 1} or (update + 1) % 8 == 0:
            calibration = trainer.origin_prefix_score_metrics_arrays(
                calibration_local, calibration_score, calibration_se
            )
            trace.append(
                {
                    "update": update + 1,
                    "total_loss_before_update": terms[0],
                    "point_loss_before_update": terms[1],
                    "global_loss_before_update": terms[2],
                    "fit_prefix_loss_before_update": terms[3],
                    "point_normalized_rms_before_update": terms[4],
                    "global_standardized_residual_before_update": terms[5],
                    "fit_prefix_maximum_standardized_residual_before_update": terms[6],
                    "gradient_norm": terms[7],
                    "maximum_core_magnitude": terms[8],
                    "calibration_prefix_maximum_standardized_residual_after_update": tf.reduce_max(
                        calibration["standardized_residual"]
                    ),
                }
            )
    point = trainer.origin_point_score_metrics_arrays(
        training.local_points[0],
        training.complete_data_score[0],
        training.observation_log_density[0],
    )
    global_metrics = trainer.origin_global_score_metrics_arrays(
        fisher.score, fisher.score_standard_error
    )
    fit_prefix = trainer.origin_prefix_score_metrics_arrays(
        fit_local, fit_score, fit_se
    )
    calibration_prefix = trainer.origin_prefix_score_metrics_arrays(
        calibration_local, calibration_score, calibration_se
    )
    training_authority_valid = all(
        float(row.effective_sample_size) >= 0.5 * int(train_prefix_count)
        and bool(
            tf.reduce_all(
                row.score_standard_error
                <= tf.constant([2.0, 1.0, 0.5], tf.float64)
            ).numpy()
        )
        for row in (*fit_authorities, *calibration_authorities)
    )
    training_gates = {
        "point_score": tf.reduce_all(point["normalized_score_residual_rms"] <= 0.90),
        "global_score": tf.reduce_all(global_metrics["standardized_residual"] <= 1.0),
        "fit_prefix": tf.reduce_all(fit_prefix["standardized_residual"] <= 1.0),
        "calibration_prefix": tf.reduce_all(
            calibration_prefix["standardized_residual"] <= 1.0
        ),
        "authority_valid": training_authority_valid,
    }
    valid_initial, valid_transition = _noise(
        validation_count, VALIDATION_INITIAL_SEED, VALIDATION_TRANSITION_SEED
    )
    validation = build_t1_parameter_density_batch(
        parent=parent,
        theta=validation_theta_rows(),
        initial_noise=valid_initial,
        transition_noise=valid_transition,
        role="validation_only_direct_tt_tangent_selection",
    )
    validation_result = _evaluate_validation(
        trainer, validation, prefix_count=int(validation_prefix_count)
    )
    validation_gates = validation_result["gates"]
    assert isinstance(validation_gates, Mapping)
    validation_score_subset = all(
        bool(tf.convert_to_tensor(validation_gates[name]).numpy())
        for name in (
            "origin_value",
            "point_score",
            "likelihood_score",
            "prefix_authority_valid",
            "prefix_score",
        )
    )
    child = trainer.freeze_child()
    artifact_dir = output_dir / "artifact"
    child.save(artifact_dir)
    reloaded = load_centered_residual_child(artifact_dir, parent=parent)
    eager = (
        *reloaded.increment_and_score(tf.zeros([3], tf.float64)),
        *reloaded.prefix_log_marginal_and_score(
            tf.zeros([3], tf.float64), validation.local_points[0, :1, :18]
        ),
    )
    compiled = _compiled_child_evaluator(
        reloaded, validation.local_points[0, :1, :18]
    )()
    xla_residual = tf.reduce_max(
        tf.abs(
            tf.concat(
                [tf.reshape(left - right, [-1]) for left, right in zip(eager, compiled)],
                axis=0,
            )
        )
    )
    training_pass = all(
        bool(tf.convert_to_tensor(value).numpy()) for value in training_gates.values()
    )
    elapsed = time.monotonic() - started
    memory = tf.config.experimental.get_memory_info("GPU:0")
    passed = (
        training_pass
        and validation_score_subset
        and reloaded.identity == child.identity
        and float(xla_residual) <= 3e-11
        and int(memory["peak"]) <= MEMORY_CAP_BYTES
        and elapsed <= max_seconds
    )
    return {
        "schema_version": "bayesfilter.zhao_cui_austria_sir_parameter_density_direct_tt_tangent.v1",
        "status": "VIABLE_DIRECT_TT_TANGENT" if passed else "REJECTED_DIRECT_TT_TANGENT",
        "tangent_id": tangent_id,
        "arm": dict(arm),
        "parent_identity": parent.identity.hash.value,
        "child_identity": child.identity.hash.value,
        "artifact_directory": artifact_dir.relative_to(ROOT).as_posix(),
        "fresh_reload": reloaded.identity == child.identity,
        "xla_eager_maximum_residual": xla_residual,
        "training_role": training.role,
        "training_fisher": _score_payload(fisher),
        "fit_prefix_authorities": [_score_payload(row) for row in fit_authorities],
        "calibration_prefix_authorities": [
            _score_payload(row) for row in calibration_authorities
        ],
        "training_point": point,
        "training_global": global_metrics,
        "fit_prefix": fit_prefix,
        "calibration_prefix": calibration_prefix,
        "training_gates": training_gates,
        "trace": trace,
        "validation_role": validation.role,
        "validation": validation_result,
        "gpu_allocator": {key: int(value) for key, value in memory.items()},
        "gates": {
            "training": training_pass,
            "validation_score_subset": validation_score_subset,
            "fresh_reload": reloaded.identity == child.identity,
            "xla_eager": float(xla_residual) <= 3e-11,
            "memory_under_6_gib": int(memory["peak"]) <= MEMORY_CAP_BYTES,
            "wall_time": elapsed <= max_seconds,
            "passed": passed,
        },
        "run_manifest": _run_manifest(started),
        "nonclaims": [
            "direct TT tangent validation cannot admit T1",
            "mass and off-origin shape remain required density gates",
            "no untouched, T2, HMC, superiority, or production claim",
        ],
    }
    valid_initial, valid_transition = _noise(
        validation_count, VALIDATION_INITIAL_SEED, VALIDATION_TRANSITION_SEED
    )
    validation = build_t1_parameter_density_batch(
        parent=parent,
        theta=validation_theta_rows(),
        initial_noise=valid_initial,
        transition_noise=valid_transition,
        role="validation_only_pair_tangent_selection",
    )
    validation_result = _evaluate_validation(
        trainer, validation, prefix_count=int(validation_prefix_count)
    )
    validation_gates = validation_result["gates"]
    assert isinstance(validation_gates, Mapping)
    validation_score_subset = all(
        bool(tf.convert_to_tensor(validation_gates[name]).numpy())
        for name in (
            "origin_value",
            "point_score",
            "likelihood_score",
            "prefix_authority_valid",
            "prefix_score",
        )
    )
    training_pass = all(
        bool(tf.convert_to_tensor(value).numpy()) for value in training_gates.values()
    )
    elapsed = time.monotonic() - started
    memory = tf.config.experimental.get_memory_info("GPU:0")
    passed = (
        training_pass
        and validation_score_subset
        and int(memory["peak"]) <= MEMORY_CAP_BYTES
        and elapsed <= max_seconds
    )
    return {
        "schema_version": "bayesfilter.zhao_cui_austria_sir_parameter_density_pair_tangent.v1",
        "status": (
            "VIABLE_TRAINING_PAIR_TANGENT"
            if passed
            else "REJECTED_TRAINING_PAIR_TANGENT"
        ),
        "tangent_id": tangent_id,
        "arm": dict(arm),
        "parent_identity": parent.identity.hash.value,
        "training_role": training.role,
        "training_fisher": _score_payload(fisher),
        "training_point": training_point,
        "training_global_standardized_residual": training_global_residual,
        "training_prefix_authorities": [_score_payload(row) for row in authorities],
        "training_prefix_standardized_residual": (
            fitted.training_prefix_score_standardized_residual
        ),
        "training_gates": training_gates,
        "validation_role": validation.role,
        "validation": validation_result,
        "initializer": {
            "feature_family": "additive_plus_within_region_si_pairs_rank7",
            "ridge_fraction": fitted.ridge_fraction,
            "global_score_weight": fitted.global_score_weight,
            "prefix_weight": fitted.prefix_weight,
            "prefix_point_count": point_count,
            "realized_ridge": fitted.realized_ridge,
            "coefficient_rms": fitted.coefficient_rms,
        },
        "gpu_allocator": {key: int(value) for key, value in memory.items()},
        "gates": {
            "training": training_pass,
            "validation_score_subset": validation_score_subset,
            "memory_under_6_gib": int(memory["peak"]) <= MEMORY_CAP_BYTES,
            "wall_time": elapsed <= max_seconds,
            "passed": passed,
        },
        "run_manifest": _run_manifest(started),
        "nonclaims": [
            "validation selects a tangent hypothesis but cannot admit T1",
            "mass and off-origin shape remain required density gates",
            "no untouched, T2, HMC, superiority, or production claim",
        ],
    }
    valid_initial, valid_transition = _noise(
        validation_count, VALIDATION_INITIAL_SEED, VALIDATION_TRANSITION_SEED
    )
    validation = build_t1_parameter_density_batch(
        parent=parent,
        theta=validation_theta_rows(),
        initial_noise=valid_initial,
        transition_noise=valid_transition,
        role="validation_only_prefix_tangent_selection",
    )
    validation_result = _evaluate_validation(
        trainer, validation, prefix_count=int(validation_prefix_count)
    )
    validation_gates = validation_result["gates"]
    assert isinstance(validation_gates, Mapping)
    validation_score_subset = all(
        bool(tf.convert_to_tensor(validation_gates[name]).numpy())
        for name in (
            "origin_value",
            "point_score",
            "likelihood_score",
            "prefix_authority_valid",
            "prefix_score",
        )
    )
    training_pass = all(
        bool(tf.convert_to_tensor(value).numpy()) for value in training_gates.values()
    )
    elapsed = time.monotonic() - started
    memory = tf.config.experimental.get_memory_info("GPU:0")
    passed = (
        training_pass
        and validation_score_subset
        and int(memory["peak"]) <= MEMORY_CAP_BYTES
        and elapsed <= max_seconds
    )
    return {
        "schema_version": "bayesfilter.zhao_cui_austria_sir_parameter_density_prefix_tangent.v1",
        "status": (
            "VIABLE_TRAINING_PREFIX_TANGENT"
            if passed
            else "REJECTED_TRAINING_PREFIX_TANGENT"
        ),
        "tangent_id": tangent_id,
        "arm": dict(arm),
        "parent_identity": parent.identity.hash.value,
        "training_role": training.role,
        "training_fisher": _score_payload(fisher),
        "training_point": training_point,
        "training_global_standardized_residual": training_global_residual,
        "training_prefix_authorities": [
            _score_payload(row) for row in authorities
        ],
        "training_prefix_child_score": training_prefix_child_score,
        "training_prefix_standardized_residual": (
            fitted.training_prefix_score_standardized_residual
        ),
        "training_gates": training_gates,
        "validation_role": validation.role,
        "validation": validation_result,
        "initializer": {
            "ridge_fraction": fitted.ridge_fraction,
            "global_score_weight": fitted.global_score_weight,
            "prefix_weight": fitted.prefix_weight,
            "prefix_point_count": prefix_point_count,
            "realized_ridge": fitted.realized_ridge,
            "coefficient_rms": fitted.coefficient_rms,
        },
        "gpu_allocator": {key: int(value) for key, value in memory.items()},
        "gates": {
            "training": training_pass,
            "validation_score_subset": validation_score_subset,
            "memory_under_6_gib": int(memory["peak"]) <= MEMORY_CAP_BYTES,
            "wall_time": elapsed <= max_seconds,
            "passed": passed,
        },
        "run_manifest": _run_manifest(started),
        "nonclaims": [
            "validation selects a tangent hypothesis but cannot admit T1",
            "mass and off-origin shape remain required density gates",
            "no untouched, T2, HMC, superiority, or production claim",
        ],
    }


def _rotating_prefix_tangent_diagnostic(
    tangent_id: str,
    output_dir: Path,
    *,
    train_count: int,
    validation_count: int,
    train_prefix_count: int,
    validation_prefix_count: int,
    max_seconds: float,
    started: float,
) -> Mapping[str, object]:
    if tangent_id not in ROTATING_PREFIX_TANGENT_TABLE:
        raise ValueError(f"unknown rotating-prefix tangent arm: {tangent_id}")
    arm = ROTATING_PREFIX_TANGENT_TABLE[tangent_id]
    fit_pool_size = int(arm["fit_pool_size"])
    prefix_batch_size = int(arm["prefix_batch_size"])
    calibration_size = int(arm["calibration_size"])
    checkpoint_interval = int(arm["checkpoint_interval"])
    required_rows = fit_pool_size + calibration_size
    if int(train_count) < max(required_rows, 16):
        raise ValueError("training count is too small for the frozen prefix partition")
    if fit_pool_size % prefix_batch_size != 0:
        raise ValueError("prefix batch size must divide the fit-pool size")
    if int(arm["steps"]) % checkpoint_interval != 0:
        raise ValueError("checkpoint interval must divide the update count")

    parent = _require_parent()
    features = CenteredThetaFeatures()
    train_initial, train_transition = _noise(
        train_count, TRAIN_INITIAL_SEED, TRAIN_TRANSITION_SEED
    )
    training = build_t1_parameter_density_batch(
        parent=parent,
        theta=tf.zeros([1, 3], tf.float64),
        initial_noise=train_initial,
        transition_noise=train_transition,
        role="training_only_rotating_prefix_tangent",
    )
    fisher = estimate_t1_ratio_score(training, theta_index=0)

    partition = tf.random.experimental.stateless_shuffle(
        tf.range(int(train_count), dtype=tf.int32),
        seed=tf.constant([int(arm["pool_partition_seed"]), 1], tf.int32),
    )
    fit_indices = partition[:fit_pool_size]
    calibration_indices = partition[fit_pool_size:required_rows]
    fit_physical = tf.gather(training.physical_points[0, :, :18], fit_indices)
    fit_local = tf.gather(training.local_points[0, :, :18], fit_indices)
    calibration_physical = tf.gather(
        training.physical_points[0, :, :18], calibration_indices
    )
    calibration_local = tf.gather(
        training.local_points[0, :, :18], calibration_indices
    )
    fit_authorities = estimate_t1_prefix_scores(
        prefix_points=fit_physical,
        global_score=fisher,
        sample_count=int(train_prefix_count),
        seed=TRAIN_PREFIX_SEED,
    )
    calibration_authorities = estimate_t1_prefix_scores(
        prefix_points=calibration_physical,
        global_score=fisher,
        sample_count=int(train_prefix_count),
        seed=CALIBRATION_PREFIX_SEED,
    )
    fit_score = tf.stack([row.score for row in fit_authorities])
    fit_se = tf.stack([row.score_standard_error for row in fit_authorities])
    calibration_score = tf.stack([row.score for row in calibration_authorities])
    calibration_se = tf.stack(
        [row.score_standard_error for row in calibration_authorities]
    )

    rank = int(arm["rank"])
    if arm.get("initializer_id") == "current_frozen_basis_core_affine_zero_v1":
        initializer_authorities = ()
        initializer_prefix_indices = []
        initializer_ledger = {
            "classification": "current_frozen_basis_core_affine_zero",
            "realized_block_rank": rank,
            "historical_gauge_used": False,
        }
        trainer = CoreAffineTangentTrainer(parent)
    elif arm.get("initializer_id") == "hash_verified_ungauged_core_tangent_s05_rank8_v1":
        tangent_banks, initializer_ledger = _load_core_tangent_warm_start(parent)
        initializer_authorities = ()
        initializer_prefix_indices: list[int] = []
        initial_components = tuple(
            core_tangent_to_residual_component(
                parent_cores=parent.cores,
                tangent_cores=tuple(bank[component_index] for bank in tangent_banks),
            )
            for component_index in range(3)
        )
        realized_rank = max(
            max(int(core.shape[0]), int(core.shape[2]))
            for component in initial_components
            for core in component
        )
        if realized_rank != rank:
            raise ValueError("core-tangent warm-start realized rank mismatch")
        trainer = CenteredResidualTrainer(
            parent, features=features, initial_residual_components=initial_components
        )
    else:
        # Match d05's p05 initializer exactly; q01/q02 change only coverage.
        initializer_authorities = estimate_t1_prefix_scores(
            prefix_points=training.physical_points[0, :16, :18],
            global_score=fisher,
            sample_count=int(train_prefix_count),
            seed=TRAIN_PREFIX_SEED,
        )
        base = target_informed_additive_score_initialization(
            parent=parent,
            local_points=training.local_points[0],
            target_complete_data_score=training.complete_data_score[0],
            importance_log_weight=training.observation_log_density[0],
            ridge_fraction=SELECTED_INITIALIZER_RIDGE_FRACTION,
            global_score_weight=100.0,
            prefix_local_points=training.local_points[0, :16, :18],
            prefix_target_score=tf.stack([row.score for row in initializer_authorities]),
            prefix_score_standard_error=tf.stack(
                [row.score_standard_error for row in initializer_authorities]
            ),
            prefix_weight=0.01,
        )
        initial_components = tuple(
            embed_residual_component_with_connected_channels(
                component,
                target_rank=rank,
                seed=RESIDUAL_SEED + 104729 * component_index,
                seeded_channel_epsilon=RANK_EXPANSION_EPSILON,
            )
            for component_index, component in enumerate(base.residual_components)
        )
        initializer_prefix_indices = list(range(16))
        initializer_ledger = {
            "classification": "p05_additive_connected_rank_expansion",
            "realized_rank": rank,
        }
        trainer = CenteredResidualTrainer(
            parent, features=features, initial_residual_components=initial_components
        )
    step = make_compiled_origin_total_score_train_step(
        trainer,
        tf.keras.optimizers.Adam(learning_rate=float(arm["learning_rate"])),
        point_weight=float(arm["point_weight"]),
        global_weight=float(arm["global_weight"]),
        prefix_weight=float(arm["prefix_weight"]),
        l2_weight=1e-10,
        gradient_clip_norm=100.0,
    )

    best_key: tuple[float, float, int] | None = None
    best_variables: tuple[tf.Tensor, ...] | None = None
    selected_update: int | None = None
    fallback_key: tuple[float, float, float, int] | None = None
    fallback_variables: tuple[tf.Tensor, ...] | None = None
    fallback_update: int | None = None
    trace: list[Mapping[str, object]] = []

    def record_checkpoint(update: int, terms: tuple[tf.Tensor, ...] | None) -> None:
        nonlocal best_key, best_variables, selected_update
        nonlocal fallback_key, fallback_variables, fallback_update
        point = trainer.origin_point_score_metrics_arrays(
            training.local_points[0],
            training.complete_data_score[0],
            training.observation_log_density[0],
        )
        global_metrics = trainer.origin_global_score_metrics_arrays(
            fisher.score, fisher.score_standard_error
        )
        calibration = trainer.origin_prefix_score_metrics_arrays(
            calibration_local, calibration_score, calibration_se
        )
        point_feasible = bool(
            tf.reduce_all(point["normalized_score_residual_rms"] <= 0.90).numpy()
        )
        global_feasible = bool(
            tf.reduce_all(global_metrics["standardized_residual"] <= 1.0).numpy()
        )
        maximum = float(tf.reduce_max(calibration["standardized_residual"]))
        mean_squared = float(calibration["loss"])
        key = rotating_prefix_checkpoint_key(maximum, mean_squared, update)
        feasibility_violation = float(
            tf.maximum(
                tf.reduce_max(point["normalized_score_residual_rms"] / 0.90),
                tf.reduce_max(global_metrics["standardized_residual"]),
            )
        )
        candidate_fallback_key = (
            feasibility_violation,
            maximum,
            mean_squared,
            int(update),
        )
        if fallback_key is None or candidate_fallback_key < fallback_key:
            fallback_key = candidate_fallback_key
            fallback_variables = tuple(
                tf.identity(variable) for variable in trainer.trainable_variables
            )
            fallback_update = int(update)
        became_best = False
        if point_feasible and global_feasible and (best_key is None or key < best_key):
            best_key = key
            best_variables = tuple(
                tf.identity(variable) for variable in trainer.trainable_variables
            )
            selected_update = int(update)
            became_best = True
        row: dict[str, object] = {
            "update": int(update),
            "point_normalized_rms": point["normalized_score_residual_rms"],
            "global_standardized_residual": global_metrics["standardized_residual"],
            "calibration_prefix_maximum_standardized_residual": maximum,
            "calibration_prefix_mean_squared_standardized_residual": mean_squared,
            "point_feasible": point_feasible,
            "global_feasible": global_feasible,
            "maximum_point_global_feasibility_ratio": feasibility_violation,
            "became_best_feasible_checkpoint": became_best,
        }
        if terms is not None:
            row.update(
                {
                    "last_minibatch_total_loss_before_update": terms[0],
                    "last_minibatch_point_loss_before_update": terms[1],
                    "last_minibatch_global_loss_before_update": terms[2],
                    "last_minibatch_prefix_loss_before_update": terms[3],
                    "last_minibatch_prefix_maximum_standardized_residual_before_update": terms[6],
                    "last_minibatch_gradient_norm": terms[7],
                    "maximum_core_magnitude": terms[8],
                }
            )
        trace.append(row)

    # The baseline is an eligible checkpoint; a repair that only worsens the
    # disjoint calibration pool must restore the unmodified p05 initialization.
    record_checkpoint(0, None)
    last_terms: tuple[tf.Tensor, ...] | None = None
    for update in range(int(arm["steps"])):
        if time.monotonic() - started > max_seconds:
            raise TimeoutError("rotating-prefix tangent arm exceeded its wall-time cap")
        minibatch = rotating_prefix_minibatch_indices(
            pool_size=fit_pool_size,
            batch_size=prefix_batch_size,
            update=update,
            seed=int(arm["minibatch_seed"]),
        )
        last_terms = step(
            training.local_points[0],
            training.complete_data_score[0],
            training.observation_log_density[0],
            fisher.score,
            fisher.score_standard_error,
            tf.gather(fit_local, minibatch),
            tf.gather(fit_score, minibatch),
            tf.gather(fit_se, minibatch),
        )
        if (update + 1) % checkpoint_interval == 0:
            record_checkpoint(update + 1, last_terms)

    feasible_checkpoint_found = (
        best_variables is not None and selected_update is not None and best_key is not None
    )
    if feasible_checkpoint_found:
        restored_variables = best_variables
        restored_update = selected_update
        restored_key: tuple[float, ...] = best_key
        checkpoint_classification = "best_point_global_feasible_calibration_checkpoint"
    else:
        if fallback_variables is None or fallback_update is None or fallback_key is None:
            raise RuntimeError("rotating-prefix checkpoint trace is empty")
        restored_variables = fallback_variables
        restored_update = fallback_update
        restored_key = fallback_key
        checkpoint_classification = "least_infeasible_explanatory_fallback"
    for variable, selected in zip(trainer.trainable_variables, restored_variables):
        variable.assign(selected)

    point = trainer.origin_point_score_metrics_arrays(
        training.local_points[0],
        training.complete_data_score[0],
        training.observation_log_density[0],
    )
    global_metrics = trainer.origin_global_score_metrics_arrays(
        fisher.score, fisher.score_standard_error
    )
    fit_prefix = trainer.origin_prefix_score_metrics_arrays(
        fit_local, fit_score, fit_se
    )
    calibration_prefix = trainer.origin_prefix_score_metrics_arrays(
        calibration_local, calibration_score, calibration_se
    )
    authority_rows = (*fit_authorities, *calibration_authorities)
    authority_valid = all(
        float(row.effective_sample_size) >= 0.5 * int(train_prefix_count)
        and bool(
            tf.reduce_all(
                row.score_standard_error
                <= tf.constant([2.0, 1.0, 0.5], tf.float64)
            ).numpy()
        )
        for row in authority_rows
    )
    training_gates = {
        "point_score": tf.reduce_all(point["normalized_score_residual_rms"] <= 0.90),
        "global_score": tf.reduce_all(global_metrics["standardized_residual"] <= 1.0),
        "fit_pool_prefix": tf.reduce_all(fit_prefix["standardized_residual"] <= 1.0),
        "calibration_prefix": tf.reduce_all(
            calibration_prefix["standardized_residual"] <= 1.0
        ),
        "authority_valid": authority_valid,
    }

    valid_initial, valid_transition = _noise(
        validation_count, VALIDATION_INITIAL_SEED, VALIDATION_TRANSITION_SEED
    )
    validation = build_t1_parameter_density_batch(
        parent=parent,
        theta=validation_theta_rows(),
        initial_noise=valid_initial,
        transition_noise=valid_transition,
        role="validation_only_rotating_prefix_tangent_selection",
    )
    validation_result = _evaluate_validation(
        trainer, validation, prefix_count=int(validation_prefix_count)
    )
    validation_gates = validation_result["gates"]
    assert isinstance(validation_gates, Mapping)
    validation_score_subset = all(
        bool(tf.convert_to_tensor(validation_gates[name]).numpy())
        for name in (
            "origin_value",
            "point_score",
            "likelihood_score",
            "prefix_authority_valid",
            "prefix_score",
        )
    )

    child = trainer.freeze_child()
    artifact_dir = output_dir / "artifact"
    child.save(artifact_dir)
    reloaded = load_centered_residual_child(artifact_dir, parent=parent)
    local_prefix = validation.local_points[0, :1, :18]
    eager = (
        *reloaded.increment_and_score(tf.zeros([3], tf.float64)),
        *reloaded.prefix_log_marginal_and_score(
            tf.zeros([3], tf.float64), local_prefix
        ),
    )
    compiled = _compiled_child_evaluator(reloaded, local_prefix)()
    xla_residual = tf.reduce_max(
        tf.abs(
            tf.concat(
                [tf.reshape(left - right, [-1]) for left, right in zip(eager, compiled)],
                axis=0,
            )
        )
    )
    training_pass = all(
        bool(tf.convert_to_tensor(value).numpy()) for value in training_gates.values()
    )
    elapsed = time.monotonic() - started
    memory = tf.config.experimental.get_memory_info("GPU:0")
    passed = (
        training_pass
        and validation_score_subset
        and reloaded.identity == child.identity
        and float(xla_residual) <= 3e-11
        and int(memory["peak"]) <= MEMORY_CAP_BYTES
        and elapsed <= max_seconds
    )
    return {
        "schema_version": "bayesfilter.zhao_cui_austria_sir_parameter_density_rotating_prefix_tangent.v1",
        "status": (
            "VIABLE_ROTATING_PREFIX_TANGENT"
            if passed
            else "REJECTED_ROTATING_PREFIX_TANGENT"
        ),
        "classification": "extension_or_invention",
        "tangent_id": tangent_id,
        "arm": dict(arm),
        "parent_identity": parent.identity.hash.value,
        "child_identity": child.identity.hash.value,
        "artifact_directory": artifact_dir.relative_to(ROOT).as_posix(),
        "fresh_reload": reloaded.identity == child.identity,
        "xla_eager_maximum_residual": xla_residual,
        "training_role": training.role,
        "training_fisher": _score_payload(fisher),
        "initializer": initializer_ledger,
        "initializer_prefix_indices": initializer_prefix_indices,
        "initializer_prefix_authorities": [
            _score_payload(row) for row in initializer_authorities
        ],
        "fit_pool_indices": fit_indices,
        "calibration_indices": calibration_indices,
        "fit_prefix_authorities": [_score_payload(row) for row in fit_authorities],
        "calibration_prefix_authorities": [
            _score_payload(row) for row in calibration_authorities
        ],
        "selected_update": restored_update,
        "selected_checkpoint_key": restored_key,
        "selected_checkpoint_classification": checkpoint_classification,
        "point_global_feasible_checkpoint_found": feasible_checkpoint_found,
        "trace": trace,
        "training_point": point,
        "training_global": global_metrics,
        "fit_pool_prefix": fit_prefix,
        "calibration_prefix": calibration_prefix,
        "training_gates": training_gates,
        "validation_role": validation.role,
        "validation": validation_result,
        "gpu_allocator": {key: int(value) for key, value in memory.items()},
        "gates": {
            "training": training_pass,
            "validation_score_subset": validation_score_subset,
            "fresh_reload": reloaded.identity == child.identity,
            "xla_eager": float(xla_residual) <= 3e-11,
            "memory_under_6_gib": int(memory["peak"]) <= MEMORY_CAP_BYTES,
            "wall_time": elapsed <= max_seconds,
            "passed": passed,
        },
        "run_manifest": _run_manifest(started),
        "nonclaims": [
            "rotating-prefix tangent validation cannot admit T1",
            "the assembled parameter-score route is not source-faithful Zhao-Cui",
            "mass and off-origin shape remain required density gates",
            "no untouched, T2, HMC, superiority, or production claim",
        ],
    }


def _core_affine_quadratic_diagnostic(
    solver_id: str,
    output_dir: Path,
    *,
    solver_family: str,
    train_count: int,
    validation_count: int,
    train_prefix_count: int,
    validation_prefix_count: int,
    max_seconds: float,
    started: float,
) -> Mapping[str, object]:
    tables = {
        "lbfgs": CORE_AFFINE_LBFGS_TABLE,
        "conjugate_gradient": CORE_AFFINE_CG_TABLE,
        "smooth_minimax": CORE_AFFINE_MINIMAX_TABLE,
        "full_tt_smooth_minimax": FULL_TT_MINIMAX_TABLE,
        "rank12_smooth_minimax": RANK12_MINIMAX_TABLE,
    }
    if solver_family not in tables or solver_id not in tables[solver_family]:
        raise ValueError(f"unknown core-affine {solver_family} arm: {solver_id}")
    arm = tables[solver_family][solver_id]
    fit_pool_size = int(arm["fit_pool_size"])
    calibration_size = int(arm["calibration_size"])
    if int(train_count) < fit_pool_size + calibration_size:
        raise ValueError("training count is too small for the quadratic-solve prefix partition")
    parent = _require_parent()
    train_initial, train_transition = _noise(
        train_count, TRAIN_INITIAL_SEED, TRAIN_TRANSITION_SEED
    )
    training = build_t1_parameter_density_batch(
        parent=parent,
        theta=tf.zeros([1, 3], tf.float64),
        initial_noise=train_initial,
        transition_noise=train_transition,
        role=f"training_only_core_affine_fullpool_{solver_family}",
    )
    fisher = estimate_t1_ratio_score(training, theta_index=0)
    partition = tf.random.experimental.stateless_shuffle(
        tf.range(int(train_count), dtype=tf.int32),
        seed=tf.constant([int(arm["pool_partition_seed"]), 1], tf.int32),
    )
    fit_indices = partition[:fit_pool_size]
    calibration_indices = partition[
        fit_pool_size : fit_pool_size + calibration_size
    ]
    fit_physical = tf.gather(training.physical_points[0, :, :18], fit_indices)
    fit_local = tf.gather(training.local_points[0, :, :18], fit_indices)
    calibration_physical = tf.gather(
        training.physical_points[0, :, :18], calibration_indices
    )
    calibration_local = tf.gather(
        training.local_points[0, :, :18], calibration_indices
    )
    fit_authorities = estimate_t1_prefix_scores(
        prefix_points=fit_physical,
        global_score=fisher,
        sample_count=int(train_prefix_count),
        seed=TRAIN_PREFIX_SEED,
    )
    calibration_authorities = estimate_t1_prefix_scores(
        prefix_points=calibration_physical,
        global_score=fisher,
        sample_count=int(train_prefix_count),
        seed=CALIBRATION_PREFIX_SEED,
    )
    fit_score = tf.stack([row.score for row in fit_authorities])
    fit_se = tf.stack([row.score_standard_error for row in fit_authorities])
    calibration_score = tf.stack([row.score for row in calibration_authorities])
    calibration_se = tf.stack(
        [row.score_standard_error for row in calibration_authorities]
    )
    authority_rows = (*fit_authorities, *calibration_authorities)
    authority_valid = all(
        float(row.effective_sample_size) >= 0.5 * int(train_prefix_count)
        and bool(
            tf.reduce_all(
                row.score_standard_error
                <= tf.constant([2.0, 1.0, 0.5], tf.float64)
            ).numpy()
        )
        for row in authority_rows
    )
    if not authority_valid:
        raise RuntimeError("core-affine quadratic-solve prefix authority is invalid")

    initializer_audit: Mapping[str, object]
    if solver_family == "lbfgs":
        trainer = CoreAffineTangentTrainer(parent)
        initializer_audit = {"initializer_id": arm["initializer_id"]}
    else:
        source_dir = (
            CORE_AFFINE_LBFGS_RESULT_DIR
            if solver_family == "conjugate_gradient"
            else CORE_AFFINE_CG_RESULT_DIR
        )
        source_result_path = source_dir / "result.json"
        source_manifest_path = source_dir / "artifact/manifest.json"
        if _sha256(source_result_path) != arm["initializer_result_sha256"]:
            raise ValueError("hash-bound source result mismatch")
        if _sha256(source_manifest_path) != arm["initializer_manifest_sha256"]:
            raise ValueError("hash-bound source child manifest mismatch")
        source_result = json.loads(source_result_path.read_text(encoding="ascii"))
        if source_result.get("child_identity") != arm["initializer_child_identity"]:
            raise ValueError("hash-bound source result child identity mismatch")
        source_child = load_centered_residual_child(
            source_dir / "artifact", parent=parent
        )
        if source_child.identity.hash.value != arm["initializer_child_identity"]:
            raise ValueError("hash-bound source child identity mismatch")
        if solver_family == "full_tt_smooth_minimax":
            trainer = CenteredResidualTrainer(
                parent,
                initial_residual_components=source_child.residual_components,
            )
        elif solver_family == "rank12_smooth_minimax":
            expanded = tuple(
                embed_residual_component_at_rank(
                    component,
                    target_rank=int(arm["rank"]),
                )
                for component in source_child.residual_components
            )
            trainer = CenteredResidualTrainer(
                parent, initial_residual_components=expanded
            )
        else:
            initial_banks = core_tangent_banks_from_residual_components(
                parent_cores=parent.cores,
                residual_components=source_child.residual_components,
            )
            trainer = CoreAffineTangentTrainer(
                parent, initial_tangent_banks=initial_banks
            )
        roundtrip_child = trainer.freeze_child()
        if solver_family in {"conjugate_gradient", "full_tt_smooth_minimax"}:
            if roundtrip_child.identity != source_child.identity:
                raise ValueError("source structured core-tangent round trip changed identity")
            roundtrip_value, roundtrip_score = roundtrip_child.increment_and_score(
                tf.zeros([3], tf.float64)
            )
            source_value, source_score = source_child.increment_and_score(
                tf.zeros([3], tf.float64)
            )
            tf.debugging.assert_near(roundtrip_value, source_value, atol=2e-12, rtol=2e-12)
            tf.debugging.assert_near(roundtrip_score, source_score, atol=2e-12, rtol=2e-12)
        else:
            source_value, source_score = source_child.increment_and_score(
                tf.zeros([3], tf.float64)
            )
            roundtrip_value, roundtrip_score = roundtrip_child.increment_and_score(
                tf.zeros([3], tf.float64)
            )
            tf.debugging.assert_near(roundtrip_value, source_value, atol=2e-12, rtol=2e-12)
            tf.debugging.assert_near(roundtrip_score, source_score, atol=2e-12, rtol=2e-12)
        initializer_audit = {
            "initializer_id": arm["initializer_id"],
            "result_path": source_result_path.relative_to(ROOT).as_posix(),
            "result_sha256": _sha256(source_result_path),
            "manifest_sha256": _sha256(source_manifest_path),
            "child_identity": source_child.identity.hash.value,
            "structured_roundtrip_identity": roundtrip_child.identity.hash.value,
            "structured_roundtrip_exact": solver_family in {
                "conjugate_gradient",
                "full_tt_smooth_minimax",
            },
            "origin_value_score_roundtrip": True,
        }
    callback_inputs = {
        "parent": parent,
        "point_local_points": training.local_points[0],
        "point_target_score": training.complete_data_score[0],
        "point_importance_log_weight": training.observation_log_density[0],
        "global_target_score": fisher.score,
        "global_score_standard_error": fisher.score_standard_error,
        "prefix_local_points": fit_local,
        "prefix_target_score": fit_score,
        "prefix_score_standard_error": fit_se,
    }
    if solver_family == "smooth_minimax":
        compiled_callback = make_compiled_core_affine_gate_minimax_value_and_gradient(
            **callback_inputs,
            temperature=float(arm["temperature"]),
            l2_weight=1e-10,
        )
    elif solver_family in {"full_tt_smooth_minimax", "rank12_smooth_minimax"}:
        reference_position = trainer.position()
        if int(reference_position.shape[0]) != int(arm["position_size"]):
            raise ValueError("full-TT position size does not match the frozen arm")
        compiled_callback = make_compiled_full_tt_gate_minimax_value_and_gradient(
            **callback_inputs,
            template_components=trainer.residual_variables,
            reference_position=reference_position,
            temperature=float(arm["temperature"]),
            l2_displacement_weight=1e-10,
        )
    else:
        compiled_callback = make_compiled_core_affine_total_score_value_and_gradient(
            **callback_inputs,
            point_weight=float(arm["point_weight"]),
            global_weight=float(arm["global_weight"]),
            prefix_weight=float(arm["prefix_weight"]),
            l2_weight=1e-10,
        )
    initial_position = trainer.position()
    initial_objective, initial_gradient = compiled_callback(initial_position)
    initial_memory = tf.config.experimental.get_memory_info("GPU:0")
    if int(initial_memory["peak"]) > MEMORY_CAP_BYTES:
        raise MemoryError("core-affine full-pool callback exceeded the 6 GiB cap")
    if time.monotonic() - started > max_seconds:
        raise TimeoutError("core-affine quadratic solve exceeded its wall cap before optimization")

    released_gradient_audit: Mapping[str, object] | None = None
    if solver_family in {"full_tt_smooth_minimax", "rank12_smooth_minimax"}:
        released_mask = (
            core_affine_released_coordinate_mask(parent.cores)
            if solver_family == "full_tt_smooth_minimax"
            else tf.ones_like(initial_gradient, dtype=tf.bool)
        )
        if released_mask.shape != initial_gradient.shape:
            raise ValueError("released-coordinate mask does not match full-TT position")
        released_gradient = tf.boolean_mask(initial_gradient, released_mask)
        released_gradient_audit = {
            "released_coordinate_count": tf.reduce_sum(tf.cast(released_mask, tf.int64)),
            "released_gradient_norm": tf.linalg.norm(released_gradient),
            "released_maximum_absolute_gradient": tf.reduce_max(
                tf.abs(released_gradient)
            ),
            "released_gradient_nonzero": tf.reduce_any(
                tf.abs(released_gradient) > tf.constant(1e-14, tf.float64)
            ),
        }
        if not bool(
            tf.convert_to_tensor(
                released_gradient_audit["released_gradient_nonzero"]
            ).numpy()
        ):
            raise RuntimeError("full-TT released coordinates have zero initial gradient")
    if solver_family in {
        "lbfgs",
        "smooth_minimax",
        "full_tt_smooth_minimax",
        "rank12_smooth_minimax",
    }:
        solver = tfp.optimizer.lbfgs_minimize(
            compiled_callback,
            initial_position=initial_position,
            num_correction_pairs=int(arm["num_correction_pairs"]),
            tolerance=float(arm["gradient_tolerance"]),
            f_relative_tolerance=float(arm["relative_objective_tolerance"]),
            max_iterations=int(arm["max_iterations"]),
            max_line_search_iterations=int(arm["max_line_search_iterations"]),
            parallel_iterations=1,
        )
        final_position = solver.position
        converged = bool(tf.convert_to_tensor(solver.converged).numpy())
        solver_failed = bool(tf.convert_to_tensor(solver.failed).numpy())
        final_objective = solver.objective_value
        final_gradient = solver.objective_gradient
        solver_payload = {
            "family": solver_family,
            "converged": converged,
            "failed": solver_failed,
            "num_iterations": solver.num_iterations,
            "num_objective_evaluations": solver.num_objective_evaluations,
            "objective_value": final_objective,
            "objective_gradient_norm": tf.linalg.norm(final_gradient),
            "objective_reduction": initial_objective - final_objective,
        }
    else:
        def timed_callback(position: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
            if time.monotonic() - started > max_seconds:
                raise TimeoutError("core-affine conjugate gradient exceeded its wall cap")
            return compiled_callback(position)

        cg = solve_quadratic_value_gradient_with_conjugate_gradient(
            timed_callback,
            initial_position=initial_position,
            tolerance=float(arm["residual_tolerance"]),
            max_iterations=int(arm["max_iterations"]),
            trace_interval=int(arm["trace_interval"]),
        )
        final_position = cg.position
        converged = bool(cg.converged)
        solver_failed = bool(cg.failed)
        final_objective, final_gradient = compiled_callback(final_position)
        solver_payload = {
            "family": solver_family,
            "converged": converged,
            "failed": solver_failed,
            "num_iterations": cg.num_iterations,
            "objective_value": final_objective,
            "objective_gradient_norm": tf.linalg.norm(final_gradient),
            "objective_reduction": initial_objective - final_objective,
            "initial_normal_equation_residual_norm": cg.initial_residual_norm,
            "normal_equation_residual_norm": cg.residual_norm,
            "relative_normal_equation_residual_norm": cg.relative_residual_norm,
            "minimum_search_curvature": cg.minimum_curvature,
            "trace": [
                {
                    "iteration": iteration,
                    "residual_norm": residual_norm,
                    "relative_residual_norm": relative_residual,
                }
                for iteration, residual_norm, relative_residual in cg.trace
            ],
        }
    tf.debugging.assert_all_finite(final_position, "core-affine solver position")
    tf.debugging.assert_all_finite(final_objective, "core-affine solver objective")
    tf.debugging.assert_all_finite(final_gradient, "core-affine solver gradient")
    trainer.assign_position(final_position)
    point = trainer.origin_point_score_metrics_arrays(
        training.local_points[0],
        training.complete_data_score[0],
        training.observation_log_density[0],
    )
    global_metrics = trainer.origin_global_score_metrics_arrays(
        fisher.score, fisher.score_standard_error
    )
    fit_prefix = trainer.origin_prefix_score_metrics_arrays(
        fit_local, fit_score, fit_se
    )
    calibration_prefix = trainer.origin_prefix_score_metrics_arrays(
        calibration_local, calibration_score, calibration_se
    )
    training_gates = {
        "point_score": tf.reduce_all(point["normalized_score_residual_rms"] <= 0.90),
        "global_score": tf.reduce_all(global_metrics["standardized_residual"] <= 1.0),
        "fit_pool_prefix": tf.reduce_all(fit_prefix["standardized_residual"] <= 1.0),
        "calibration_prefix": tf.reduce_all(
            calibration_prefix["standardized_residual"] <= 1.0
        ),
        "authority_valid": authority_valid,
    }

    valid_initial, valid_transition = _noise(
        validation_count, VALIDATION_INITIAL_SEED, VALIDATION_TRANSITION_SEED
    )
    validation = build_t1_parameter_density_batch(
        parent=parent,
        theta=validation_theta_rows(),
        initial_noise=valid_initial,
        transition_noise=valid_transition,
        role=f"validation_only_core_affine_{solver_family}_selection",
    )
    validation_result = _evaluate_validation(
        trainer, validation, prefix_count=int(validation_prefix_count)
    )
    validation_gates = validation_result["gates"]
    assert isinstance(validation_gates, Mapping)
    validation_score_subset = all(
        bool(tf.convert_to_tensor(validation_gates[name]).numpy())
        for name in (
            "origin_value",
            "point_score",
            "likelihood_score",
            "prefix_authority_valid",
            "prefix_score",
        )
    )
    child = trainer.freeze_child()
    artifact_dir = output_dir / "artifact"
    child.save(artifact_dir)
    reloaded = load_centered_residual_child(artifact_dir, parent=parent)
    local_prefix = validation.local_points[0, :1, :18]
    eager = (
        *reloaded.increment_and_score(tf.zeros([3], tf.float64)),
        *reloaded.prefix_log_marginal_and_score(
            tf.zeros([3], tf.float64), local_prefix
        ),
    )
    compiled = _compiled_child_evaluator(reloaded, local_prefix)()
    xla_residual = tf.reduce_max(
        tf.abs(
            tf.concat(
                [tf.reshape(left - right, [-1]) for left, right in zip(eager, compiled)],
                axis=0,
            )
        )
    )
    training_pass = all(
        bool(tf.convert_to_tensor(value).numpy()) for value in training_gates.values()
    )
    elapsed = time.monotonic() - started
    memory = tf.config.experimental.get_memory_info("GPU:0")
    passed = (
        converged
        and not solver_failed
        and training_pass
        and validation_score_subset
        and reloaded.identity == child.identity
        and float(xla_residual) <= 3e-11
        and int(memory["peak"]) <= MEMORY_CAP_BYTES
        and elapsed <= max_seconds
    )
    return {
        "schema_version": "bayesfilter.zhao_cui_austria_sir_parameter_density_core_affine_quadratic_solve.v1",
        "status": (
            "VIABLE_CORE_AFFINE_QUADRATIC_TANGENT"
            if passed
            else "REJECTED_CORE_AFFINE_QUADRATIC_TANGENT"
        ),
        "classification": "extension_or_invention",
        "solver_id": solver_id,
        "solver_family": solver_family,
        "arm": dict(arm),
        "initializer_audit": initializer_audit,
        "parent_identity": parent.identity.hash.value,
        "child_identity": child.identity.hash.value,
        "artifact_directory": artifact_dir.relative_to(ROOT).as_posix(),
        "fit_pool_indices": fit_indices,
        "calibration_indices": calibration_indices,
        "fit_prefix_authorities": [_score_payload(row) for row in fit_authorities],
        "calibration_prefix_authorities": [
            _score_payload(row) for row in calibration_authorities
        ],
        "initial_capacity_probe": {
            "objective": initial_objective,
            "gradient_norm": tf.linalg.norm(initial_gradient),
            "released_gradient_audit": released_gradient_audit,
            "gpu_allocator": {
                key: int(value) for key, value in initial_memory.items()
            },
        },
        "solver": solver_payload,
        "training_point": point,
        "training_global": global_metrics,
        "fit_pool_prefix": fit_prefix,
        "calibration_prefix": calibration_prefix,
        "training_gates": training_gates,
        "validation": validation_result,
        "fresh_reload": reloaded.identity == child.identity,
        "xla_eager_maximum_residual": xla_residual,
        "gpu_allocator": {key: int(value) for key, value in memory.items()},
        "gates": {
            "solver_converged": converged,
            "solver_not_failed": not solver_failed,
            "training": training_pass,
            "validation_score_subset": validation_score_subset,
            "fresh_reload": reloaded.identity == child.identity,
            "xla_eager": float(xla_residual) <= 3e-11,
            "memory_under_6_gib": int(memory["peak"]) <= MEMORY_CAP_BYTES,
            "wall_time": elapsed <= max_seconds,
            "passed": passed,
        },
        "run_manifest": _run_manifest(started),
        "nonclaims": [
            "quadratic-solver convergence and objective reduction are not score admission",
            "the assembled parameter-score route is not source-faithful Zhao-Cui",
            "mass and off-origin shape remain required density gates",
            "no untouched, T2, HMC, superiority, or production claim",
        ],
    }


def _core_affine_lbfgs_diagnostic(
    solver_id: str,
    output_dir: Path,
    **kwargs: object,
) -> Mapping[str, object]:
    return _core_affine_quadratic_diagnostic(
        solver_id,
        output_dir,
        solver_family="lbfgs",
        **kwargs,
    )


def _core_affine_cg_diagnostic(
    solver_id: str,
    output_dir: Path,
    **kwargs: object,
) -> Mapping[str, object]:
    return _core_affine_quadratic_diagnostic(
        solver_id,
        output_dir,
        solver_family="conjugate_gradient",
        **kwargs,
    )


def _core_affine_minimax_diagnostic(
    solver_id: str,
    output_dir: Path,
    **kwargs: object,
) -> Mapping[str, object]:
    return _core_affine_quadratic_diagnostic(
        solver_id,
        output_dir,
        solver_family="smooth_minimax",
        **kwargs,
    )


def _full_tt_minimax_diagnostic(
    solver_id: str,
    output_dir: Path,
    **kwargs: object,
) -> Mapping[str, object]:
    return _core_affine_quadratic_diagnostic(
        solver_id,
        output_dir,
        solver_family="full_tt_smooth_minimax",
        **kwargs,
    )


def _rank12_minimax_diagnostic(
    solver_id: str,
    output_dir: Path,
    **kwargs: object,
) -> Mapping[str, object]:
    return _core_affine_quadratic_diagnostic(
        solver_id,
        output_dir,
        solver_family="rank12_smooth_minimax",
        **kwargs,
    )


def _pilot(
    arm_id: str,
    output_dir: Path,
    *,
    train_count: int,
    validation_count: int,
    train_steps: int,
    validation_prefix_count: int,
    max_seconds: float,
    started: float,
) -> Mapping[str, object]:
    if arm_id not in ARM_TABLE:
        raise ValueError(f"unknown arm: {arm_id}")
    arm = ARM_TABLE[arm_id]
    parent = _require_parent()
    features = CenteredThetaFeatures()
    train_initial, train_transition = _noise(
        train_count, TRAIN_INITIAL_SEED, TRAIN_TRANSITION_SEED
    )
    training = build_t1_parameter_density_batch(
        parent=parent,
        theta=axis_theta_rows(float(arm["radius"])),
        initial_noise=train_initial,
        transition_noise=train_transition,
        role="training",
    )
    initial_components, initializer = _selected_training_only_initial_components(
        parent=parent,
        features=features,
        training=training,
        target_rank=int(arm["rank"]),
    )
    trainer = CenteredResidualTrainer(
        parent, features=features, initial_residual_components=initial_components
    )
    valid_initial, valid_transition = _noise(
        validation_count, VALIDATION_INITIAL_SEED, VALIDATION_TRANSITION_SEED
    )
    validation = build_t1_parameter_density_batch(
        parent=parent,
        theta=validation_theta_rows(),
        initial_noise=valid_initial,
        transition_noise=valid_transition,
        role="validation",
    )
    step = make_compiled_absolute_train_step(
        trainer,
        tf.keras.optimizers.Adam(learning_rate=float(arm["learning_rate"])),
        l1_weight=float(arm["l1_weight"]),
        l2_weight=1e-10,
        derivative_weight=float(arm["derivative_weight"]),
        gradient_clip_norm=100.0,
    )
    trace = []
    training_failure = None
    permutation = tf.range(int(train_count), dtype=tf.int32)
    for update in range(int(train_steps)):
        if time.monotonic() - started > max_seconds:
            raise TimeoutError("pilot exceeded its predeclared wall-time cap")
        if update % max(int(train_count) // TRAIN_BATCH_SIZE, 1) == 0:
            permutation = tf.random.experimental.stateless_shuffle(
                tf.range(int(train_count), dtype=tf.int32),
                seed=tf.constant([85801, update], tf.int32),
            )
        first = (update * TRAIN_BATCH_SIZE) % int(train_count)
        indices = permutation[first : first + TRAIN_BATCH_SIZE]
        try:
            terms = step(
                training.theta,
                tf.gather(training.local_points, indices, axis=1),
                tf.gather(
                    parent.shift_constant + training.observation_log_density,
                    indices,
                    axis=1,
                ),
                tf.gather(training.local_points[0], indices),
                tf.gather(training.complete_data_score[0], indices),
                tf.gather(training.observation_log_density[0], indices),
            )
        except tf.errors.InvalidArgumentError as error:
            training_failure = {
                "classification": "candidate_numerical_training_failure",
                "failed_update": update + 1,
                "error_type": type(error).__name__,
                "error": str(error),
            }
            break
        if update in {0, int(train_steps) - 1} or (update + 1) % 16 == 0:
            trace.append(
                {
                    "update": update + 1,
                    "total_loss": terms[0],
                    "absolute_density_loss": terms[1],
                    "derivative_matching_loss": terms[2],
                    "mean_child_mass": terms[3],
                    "mean_target_log_term": terms[4],
                    "mean_target_mass_estimate": terms[5],
                    "maximum_target_mass_standard_error": terms[6],
                    "minimum_rho": terms[7],
                    "gradient_norm": terms[8],
                    "maximum_core_magnitude": terms[9],
                }
            )
    if training_failure is not None:
        memory = tf.config.experimental.get_memory_info("GPU:0")
        return {
            "schema_version": "bayesfilter.zhao_cui_austria_sir_parameter_density_pilot.v1",
            "status": "REJECTED_T1_PARAMETER_DENSITY_ARM",
            "arm_id": arm_id,
            "arm": dict(arm),
            "parent_identity": parent.identity.hash.value,
            "parent_value": parent.value(),
            "child_identity": None,
            "training_trace": trace,
            "initializer": initializer,
            "training_failure": training_failure,
            "training_role": training.role,
            "validation_role": validation.role,
            "validation": None,
            "artifact_directory": None,
            "gpu_allocator": {key: int(value) for key, value in memory.items()},
            "gates": {
                "training_finite": False,
                "memory_under_6_gib": int(memory["peak"]) <= MEMORY_CAP_BYTES,
                "wall_time": time.monotonic() - started <= max_seconds,
                "passed": False,
            },
            "run_manifest": _run_manifest(started),
            "nonclaims": [
                "candidate numerical failure does not invalidate the harness or research direction",
                "no validation, score admission, T2, or HMC claim",
            ],
        }
    validation_result = _evaluate_validation(
        trainer, validation, prefix_count=int(validation_prefix_count)
    )
    child = trainer.freeze_child()
    artifact_dir = output_dir / "artifact"
    child.save(artifact_dir)
    reloaded = load_centered_residual_child(artifact_dir, parent=parent)
    local_prefix = validation.local_points[0, :1, :18]
    eager = (
        *reloaded.increment_and_score(tf.zeros([3], tf.float64)),
        *reloaded.prefix_log_marginal_and_score(
            tf.zeros([3], tf.float64), local_prefix
        ),
    )
    compiled = _compiled_child_evaluator(reloaded, local_prefix)()
    xla_residual = tf.reduce_max(
        tf.abs(tf.concat([tf.reshape(left - right, [-1]) for left, right in zip(eager, compiled)], axis=0))
    )
    memory = tf.config.experimental.get_memory_info("GPU:0")
    validation_gates = validation_result["gates"]
    assert isinstance(validation_gates, Mapping)
    validation_pass = all(bool(tf.convert_to_tensor(value).numpy()) for value in validation_gates.values())
    passed = (
        validation_pass
        and reloaded.identity == child.identity
        and float(xla_residual) <= 3e-11
        and int(memory["peak"]) <= MEMORY_CAP_BYTES
        and time.monotonic() - started <= max_seconds
    )
    return {
        "schema_version": "bayesfilter.zhao_cui_austria_sir_parameter_density_pilot.v1",
        "status": "VIABLE_T1_PARAMETER_DENSITY_ARM" if passed else "REJECTED_T1_PARAMETER_DENSITY_ARM",
        "arm_id": arm_id,
        "arm": dict(arm),
        "parent_identity": parent.identity.hash.value,
        "parent_value": parent.value(),
        "child_identity": child.identity.hash.value,
        "training_trace": trace,
        "initializer": initializer,
        "training_role": training.role,
        "validation_role": validation.role,
        "validation": validation_result,
        "artifact_directory": artifact_dir.relative_to(ROOT).as_posix(),
        "artifact_manifest_sha256": _sha256(artifact_dir / "manifest.json"),
        "fresh_reload": reloaded.identity == child.identity,
        "xla_eager_maximum_residual": xla_residual,
        "gpu_allocator": {key: int(value) for key, value in memory.items()},
        "gates": {
            "validation": validation_pass,
            "fresh_reload": reloaded.identity == child.identity,
            "xla_eager": float(xla_residual) <= 3e-11,
            "memory_under_6_gib": int(memory["peak"]) <= MEMORY_CAP_BYTES,
            "wall_time": time.monotonic() - started <= max_seconds,
            "passed": passed,
        },
        "run_manifest": _run_manifest(started),
        "nonclaims": [
            "validation is selection evidence only",
            "no T1 score admission before the untouched claim",
            "no T2 or HMC readiness",
        ],
    }


def _claim(
    selection_path: Path,
    output_dir: Path,
    *,
    sample_count: int,
    prefix_count: int,
    max_seconds: float,
    started: float,
) -> Mapping[str, object]:
    del output_dir
    selection = json.loads(selection_path.read_text())
    if (
        selection.get("schema_version") != SELECTION_SCHEMA
        or selection.get("status") != "SELECTED_T1_PARAMETER_DENSITY_ARM"
        or not isinstance(selection.get("selected"), Mapping)
    ):
        raise ValueError("claim requires a valid selector-frozen artifact")
    selected = selection["selected"]
    assert isinstance(selected, Mapping)
    artifact_dir = ROOT / str(selected["artifact_directory"])
    parent = _require_parent()
    child = load_centered_residual_child(artifact_dir, parent=parent)
    if child.identity.hash.value != selected.get("child_identity"):
        raise ValueError("selector child identity does not match the loaded artifact")
    initial, transition = _noise(
        sample_count, UNTOUCHED_INITIAL_SEED, UNTOUCHED_TRANSITION_SEED
    )
    batch = build_t1_parameter_density_batch(
        parent=parent,
        theta=tf.zeros([1, 3], tf.float64),
        initial_noise=initial,
        transition_noise=transition,
        role="untouched",
    )
    fisher = estimate_t1_ratio_score(batch, theta_index=0)
    child_value, child_score = child.increment_and_score(tf.zeros([3], tf.float64))
    tolerance = 3.0 * fisher.score_standard_error + tf.constant(1e-5, tf.float64)
    difference = tf.abs(child_score - fisher.score)
    coordinate_gate = difference <= tolerance
    informative = tf.reduce_all(
        fisher.score_standard_error <= tf.constant([2.0, 1.0, 0.5], tf.float64)
    )
    prefix_points = batch.physical_points[0, :3, :18]
    local_prefix = batch.local_points[0, :3, :18]
    prefix_authorities = estimate_t1_prefix_scores(
        prefix_points=prefix_points,
        global_score=fisher,
        sample_count=int(prefix_count),
        seed=UNTOUCHED_PREFIX_SEED,
    )
    prefix_value, prefix_score = child.prefix_log_marginal_and_score(
        tf.zeros([3], tf.float64), local_prefix
    )
    prefix_rows = []
    prefix_pass = True
    for index, authority in enumerate(prefix_authorities):
        prefix_tolerance = 3.0 * authority.score_standard_error + tf.constant(1e-5, tf.float64)
        prefix_difference = tf.abs(prefix_score[index] - authority.score)
        score_gate = tf.reduce_all(prefix_difference <= prefix_tolerance)
        validity = tf.logical_and(
            authority.effective_sample_size >= tf.constant(0.5 * prefix_count, tf.float64),
            tf.reduce_all(
                authority.score_standard_error <= tf.constant([2.0, 1.0, 0.5], tf.float64)
            ),
        )
        row_pass = bool(tf.logical_and(score_gate, validity).numpy())
        prefix_pass = prefix_pass and row_pass
        prefix_rows.append(
            {
                "index": index,
                "child_log_value": prefix_value[index],
                "child_score": prefix_score[index],
                "authority": _score_payload(authority),
                "absolute_difference": prefix_difference,
                "tolerance": prefix_tolerance,
                "score_gate": score_gate,
                "validity_gate": validity,
                "passed": row_pass,
            }
        )
    eager = (child_value, child_score, prefix_value, prefix_score)
    compiled = _compiled_child_evaluator(child, local_prefix)()
    xla_residual = tf.reduce_max(
        tf.abs(tf.concat([tf.reshape(left - right, [-1]) for left, right in zip(eager, compiled)], axis=0))
    )
    memory = tf.config.experimental.get_memory_info("GPU:0")
    passed = (
        bool(tf.reduce_all(coordinate_gate).numpy())
        and bool(informative.numpy())
        and prefix_pass
        and abs(float(child_value) - EXPECTED_PARENT_VALUE) <= 2e-13
        and float(xla_residual) <= 3e-11
        and int(memory["peak"]) <= MEMORY_CAP_BYTES
        and time.monotonic() - started <= max_seconds
    )
    return {
        "schema_version": "bayesfilter.zhao_cui_austria_sir_parameter_density_claim.v1",
        "status": "PASS_T1_PARAMETER_DENSITY_VALUE_AND_SCORE" if passed else "BLOCK_T1_PARAMETER_DENSITY_CLAIM",
        "parent_identity": parent.identity.hash.value,
        "child_identity": child.identity.hash.value,
        "child_origin_value": child_value,
        "child_origin_score": child_score,
        "fisher": _score_payload(fisher),
        "absolute_score_difference": difference,
        "score_tolerance": tolerance,
        "coordinate_gate": coordinate_gate,
        "informative_mcse_gate": informative,
        "prefix_rows": prefix_rows,
        "xla_eager_maximum_residual": xla_residual,
        "gpu_allocator": {key: int(value) for key, value in memory.items()},
        "gates": {
            "origin_value": abs(float(child_value) - EXPECTED_PARENT_VALUE) <= 2e-13,
            "origin_score": bool(tf.reduce_all(coordinate_gate).numpy()),
            "informative_mcse": bool(informative.numpy()),
            "prefix_score": prefix_pass,
            "xla_eager": float(xla_residual) <= 3e-11,
            "memory_under_6_gib": int(memory["peak"]) <= MEMORY_CAP_BYTES,
            "wall_time": time.monotonic() - started <= max_seconds,
            "passed": passed,
        },
        "artifact_directory": artifact_dir.relative_to(ROOT).as_posix(),
        "selection_path": selection_path.relative_to(ROOT).as_posix(),
        "selection_sha256": _sha256(selection_path),
        "selected_arm_id": selected["arm_id"],
        "run_manifest": _run_manifest(started),
        "decision_table": {
            "decision": "open_T2_total_score" if passed else "reject_selected_T1_centered_child",
            "primary_criterion_status": "passed" if passed else "failed",
            "veto_diagnostic_status": "passed" if prefix_pass and int(memory["peak"]) <= MEMORY_CAP_BYTES else "failed",
            "main_uncertainty": "iid ratio and conditional-ratio Monte Carlo standard error",
            "next_justified_action": "implement T2 retained-prefix recursion" if passed else "classify candidate versus authority failure",
            "not_concluded": "no T2/T20, HMC, exact-likelihood theorem, posterior, superiority, or production readiness",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--capacity-probe", action="store_true")
    group.add_argument("--initializer-audit", choices=tuple(INITIALIZER_AUDIT_TABLE))
    group.add_argument("--selected-initializer-validation", action="store_true")
    group.add_argument("--prefix-tangent", choices=tuple(PREFIX_TANGENT_TABLE))
    group.add_argument("--pair-tangent", choices=tuple(PAIR_TANGENT_TABLE))
    group.add_argument("--direct-tt-tangent", choices=tuple(DIRECT_TT_TANGENT_TABLE))
    group.add_argument(
        "--rotating-prefix-tangent", choices=tuple(ROTATING_PREFIX_TANGENT_TABLE)
    )
    group.add_argument("--core-affine-lbfgs", choices=tuple(CORE_AFFINE_LBFGS_TABLE))
    group.add_argument("--core-affine-cg", choices=tuple(CORE_AFFINE_CG_TABLE))
    group.add_argument(
        "--core-affine-minimax", choices=tuple(CORE_AFFINE_MINIMAX_TABLE)
    )
    group.add_argument("--full-tt-minimax", choices=tuple(FULL_TT_MINIMAX_TABLE))
    group.add_argument("--rank12-minimax", choices=tuple(RANK12_MINIMAX_TABLE))
    group.add_argument("--prefix-authority-reproducibility", action="store_true")
    group.add_argument("--prefix-authority-sample-growth", action="store_true")
    group.add_argument("--pilot-arm", choices=tuple(ARM_TABLE))
    group.add_argument("--claim-selection", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-count", type=int, default=TRAIN_COUNT)
    parser.add_argument("--validation-count", type=int, default=VALIDATION_COUNT)
    parser.add_argument("--train-steps", type=int, default=TRAIN_STEPS)
    parser.add_argument("--validation-prefix-count", type=int, default=8192)
    parser.add_argument("--train-prefix-count", type=int, default=TRAIN_PREFIX_COUNT)
    parser.add_argument("--claim-count", type=int, default=UNTOUCHED_COUNT)
    parser.add_argument("--claim-prefix-count", type=int, default=UNTOUCHED_PREFIX_COUNT)
    parser.add_argument("--max-seconds", type=float, required=True)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        if args.prefix_authority_reproducibility:
            result = _prefix_authority_reproducibility(
                output_dir,
                train_count=args.train_count,
                train_prefix_count=args.train_prefix_count,
                max_seconds=args.max_seconds,
                started=started,
            )
        elif args.prefix_authority_sample_growth:
            result = _prefix_authority_sample_growth(
                output_dir,
                train_count=args.train_count,
                max_seconds=args.max_seconds,
                started=started,
            )
        elif args.capacity_probe:
            result = _capacity(output_dir, args.max_seconds, started)
        elif args.initializer_audit is not None:
            result = _initializer_audit(
                args.initializer_audit,
                output_dir,
                train_count=args.train_count,
                max_seconds=args.max_seconds,
                started=started,
            )
        elif args.selected_initializer_validation:
            result = _selected_initializer_validation(
                output_dir,
                train_count=args.train_count,
                validation_count=args.validation_count,
                validation_prefix_count=args.validation_prefix_count,
                max_seconds=args.max_seconds,
                started=started,
            )
        elif args.prefix_tangent is not None:
            result = _prefix_tangent_diagnostic(
                args.prefix_tangent,
                output_dir,
                train_count=args.train_count,
                validation_count=args.validation_count,
                train_prefix_count=args.train_prefix_count,
                validation_prefix_count=args.validation_prefix_count,
                max_seconds=args.max_seconds,
                started=started,
            )
        elif args.pair_tangent is not None:
            result = _pair_tangent_diagnostic(
                args.pair_tangent,
                output_dir,
                train_count=args.train_count,
                validation_count=args.validation_count,
                train_prefix_count=args.train_prefix_count,
                validation_prefix_count=args.validation_prefix_count,
                max_seconds=args.max_seconds,
                started=started,
            )
        elif args.direct_tt_tangent is not None:
            result = _direct_tt_tangent_diagnostic(
                args.direct_tt_tangent,
                output_dir,
                train_count=args.train_count,
                validation_count=args.validation_count,
                train_prefix_count=args.train_prefix_count,
                validation_prefix_count=args.validation_prefix_count,
                max_seconds=args.max_seconds,
                started=started,
            )
        elif args.rotating_prefix_tangent is not None:
            result = _rotating_prefix_tangent_diagnostic(
                args.rotating_prefix_tangent,
                output_dir,
                train_count=args.train_count,
                validation_count=args.validation_count,
                train_prefix_count=args.train_prefix_count,
                validation_prefix_count=args.validation_prefix_count,
                max_seconds=args.max_seconds,
                started=started,
            )
        elif args.core_affine_lbfgs is not None:
            result = _core_affine_lbfgs_diagnostic(
                args.core_affine_lbfgs,
                output_dir,
                train_count=args.train_count,
                validation_count=args.validation_count,
                train_prefix_count=args.train_prefix_count,
                validation_prefix_count=args.validation_prefix_count,
                max_seconds=args.max_seconds,
                started=started,
            )
        elif args.core_affine_cg is not None:
            result = _core_affine_cg_diagnostic(
                args.core_affine_cg,
                output_dir,
                train_count=args.train_count,
                validation_count=args.validation_count,
                train_prefix_count=args.train_prefix_count,
                validation_prefix_count=args.validation_prefix_count,
                max_seconds=args.max_seconds,
                started=started,
            )
        elif args.core_affine_minimax is not None:
            result = _core_affine_minimax_diagnostic(
                args.core_affine_minimax,
                output_dir,
                train_count=args.train_count,
                validation_count=args.validation_count,
                train_prefix_count=args.train_prefix_count,
                validation_prefix_count=args.validation_prefix_count,
                max_seconds=args.max_seconds,
                started=started,
            )
        elif args.full_tt_minimax is not None:
            result = _full_tt_minimax_diagnostic(
                args.full_tt_minimax,
                output_dir,
                train_count=args.train_count,
                validation_count=args.validation_count,
                train_prefix_count=args.train_prefix_count,
                validation_prefix_count=args.validation_prefix_count,
                max_seconds=args.max_seconds,
                started=started,
            )
        elif args.rank12_minimax is not None:
            result = _rank12_minimax_diagnostic(
                args.rank12_minimax,
                output_dir,
                train_count=args.train_count,
                validation_count=args.validation_count,
                train_prefix_count=args.train_prefix_count,
                validation_prefix_count=args.validation_prefix_count,
                max_seconds=args.max_seconds,
                started=started,
            )
        elif args.pilot_arm is not None:
            result = _pilot(
                args.pilot_arm,
                output_dir,
                train_count=args.train_count,
                validation_count=args.validation_count,
                train_steps=args.train_steps,
                validation_prefix_count=args.validation_prefix_count,
                max_seconds=args.max_seconds,
                started=started,
            )
        else:
            assert args.claim_selection is not None
            result = _claim(
                args.claim_selection.resolve(),
                output_dir,
                sample_count=args.claim_count,
                prefix_count=args.claim_prefix_count,
                max_seconds=args.max_seconds,
                started=started,
            )
        _write_json(output_dir / "result.json", result)
    except Exception as error:
        failure = {
            "schema_version": "bayesfilter.zhao_cui_austria_sir_parameter_density_failure.v1",
            "status": "INFRASTRUCTURE_IMPLEMENTATION_OR_CANDIDATE_FAILURE",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "run_manifest": _run_manifest(started),
        }
        _write_json(output_dir / "failure.json", failure)
        raise
    if not bool(result["gates"]["passed"]):  # type: ignore[index]
        raise SystemExit(2)


if __name__ == "__main__":
    main()
