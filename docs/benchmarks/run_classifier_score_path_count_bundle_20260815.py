"""Run one independent-noise classifier-score path-count bundle."""

from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
import tensorflow as tf

from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth

CPU_ONLY_SMOKE = os.environ.get("BAYESFILTER_CPU_ONLY_SMOKE", "false").lower() in {
    "1",
    "true",
    "yes",
}
GPU_MEMORY_POLICY = configure_tensorflow_gpu_memory_growth(
    tf, require_gpu=not CPU_ONLY_SMOKE
)
tf.config.experimental.enable_tensor_float_32_execution(False)

from bayesfilter.independent_score import gaussian_observation_simulator_tf as gaussian  # noqa: E402
from bayesfilter.independent_score import sir_observation_simulator_tf as sir  # noqa: E402
from bayesfilter.independent_score.anchored_orthogonal_ratio_score_tf import (  # noqa: E402
    DELTAS,
    fit_anchored_classifier,
)


THETA = tf.zeros([3], tf.float64)
HORIZONS = (20, 40, 50)
COORDINATES = (0, 1, 2)
PATH_COUNTS = (8192, 16384, 32768)
SIMULATION_BLOCK = 8192
INVALID_PATH_RATE_THRESHOLD = 1.0e-3
INVALID_PATH_POLICIES = ("preserve", "remove_invalid_paths")
ROOT_SEED = 95170
V5_ROOT = ROOT / "docs/benchmarks/artifacts/sir_null_calibrated_predictive_consistency_20260814"
V6_ROOT = ROOT / "docs/benchmarks/artifacts/classifier_score_variance_reduction_20260815"
PLAN_PATH = ROOT / "docs/plans/bayesfilter-classifier-score-path-count-scaling-v7-plan-2026-08-15.md"
ESTIMATOR_PATH = ROOT / "bayesfilter/independent_score/anchored_orthogonal_ratio_score_tf.py"
GAUSSIAN_PATH = ROOT / "bayesfilter/independent_score/gaussian_observation_simulator_tf.py"
SIR_PATH = ROOT / "bayesfilter/independent_score/sir_observation_simulator_tf.py"
RUNNER_PATH = Path(__file__).resolve()


def safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(item) for item in value]
    if tf.is_tensor(value) or isinstance(value, tf.Variable):
        return safe(tf.convert_to_tensor(value).numpy().tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def seed(*parts: int) -> int:
    result = ROOT_SEED
    for part in parts:
        result = (result * 1009 + int(part) + 7919) % 2147483000
    return result


def profile(name: str) -> dict[str, int]:
    if name == "smoke":
        return {
            "validation": 32,
            "calibration": 32,
            "test": 64,
            "audit": 16,
            "batch": 128,
            "epochs": 3,
            "minimum": 2,
            "patience": 2,
        }
    if name in ("full", "full_cell"):
        return {
            "validation": 512,
            "calibration": 512,
            "test": 1024,
            "audit": 128,
            "batch": 2048,
            "epochs": 80,
            "minimum": 15,
            "patience": 10,
        }
    raise ValueError("profile must be smoke, full_cell, or full")


def model_code(kind: str) -> int:
    return 1 if kind == "gaussian" else 2


def make_noise(kind: str, count: int, key: tuple[int, ...]) -> tuple[tf.Tensor, ...]:
    first = seed(model_code(kind), *key)
    if kind == "gaussian":
        return (
            tf.random.stateless_normal(
                [count, 50, 9], [first, 71], dtype=tf.float64
            ),
        )
    return (
        tf.random.stateless_normal([count, 18], [first, 11], dtype=tf.float64),
        tf.random.stateless_normal(
            [count, 50, 18], [first, 13], dtype=tf.float64
        ),
        tf.random.stateless_normal([count, 50, 9], [first, 17], dtype=tf.float64),
    )


def simulate(
    simulator: Any,
    parameters: tf.Tensor,
    noise: tuple[tf.Tensor, ...],
    count: int,
) -> tf.Tensor:
    return simulator(parameters, *(value[:count] for value in noise))


def simulate_in_fixed_blocks(
    simulator: Any,
    parameters: tf.Tensor,
    noise: tuple[tf.Tensor, ...],
    count: int,
    *,
    block_size: int = SIMULATION_BLOCK,
) -> tf.Tensor:
    if int(count) % int(block_size) != 0:
        raise ValueError("path count must be divisible by the frozen simulation block")
    blocks = []
    for start in range(0, int(count), int(block_size)):
        stop = start + int(block_size)
        blocks.append(simulator(parameters, *(value[start:stop] for value in noise)))
    return tf.concat(blocks, axis=0)


def noise_hash(values: tuple[tf.Tensor, ...], count: int) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(tf.io.serialize_tensor(value[:count]).numpy())
    return digest.hexdigest()


def dataset_hash(values: tuple[tf.Tensor, tf.Tensor, tf.Tensor]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(tf.io.serialize_tensor(value).numpy())
    return digest.hexdigest()


def tensor_hash(value: tf.Tensor) -> str:
    return hashlib.sha256(tf.io.serialize_tensor(value).numpy()).hexdigest()


def finite_path_mask(paths: tf.Tensor) -> tf.Tensor:
    """Return one finite/non-finite decision per generated observation path."""

    values = tf.convert_to_tensor(paths)
    if values.shape.rank != 3:
        raise ValueError("observation paths must have shape [path,time,observation]")
    return tf.reduce_all(tf.math.is_finite(values), axis=(1, 2))


def filter_invalid_paths(
    paths: tf.Tensor,
    *,
    remove_invalid: bool,
) -> tuple[tf.Tensor, dict[str, Any]]:
    """Optionally remove non-finite paths, preserving an auditable count."""

    values = tf.convert_to_tensor(paths)
    mask = finite_path_mask(values)
    invalid = int(tf.reduce_sum(tf.cast(tf.logical_not(mask), tf.int32)).numpy())
    total = int(values.shape[0])
    if remove_invalid and invalid:
        values = tf.boolean_mask(values, mask)
    return values, {
        "generated_path_count": total,
        "invalid_path_count": invalid,
        "retained_path_count": int(values.shape[0]),
    }


def _filter_paired_cases(
    cases: list[tuple[tf.Tensor, tf.Tensor]],
    *,
    path_count: int,
    policy: str,
    threshold: float,
) -> tuple[list[tuple[tf.Tensor, tf.Tensor]], dict[str, Any]]:
    """Filter +/- cases together so conditional class balance remains valid."""

    if policy not in INVALID_PATH_POLICIES:
        raise ValueError(f"unknown invalid-path policy: {policy}")
    if float(threshold) < 0.0:
        raise ValueError("invalid-path threshold must be non-negative")
    masks = [(finite_path_mask(minus), finite_path_mask(plus)) for minus, plus in cases]
    invalid_rows = sum(
        int(tf.reduce_sum(tf.cast(tf.logical_not(mask), tf.int32)).numpy())
        for pair in masks
        for mask in pair
    )
    generated_rows = 2 * int(path_count) * len(cases)
    rate = invalid_rows / generated_rows if generated_rows else 0.0
    remove = policy == "remove_invalid_paths"
    filtered: list[tuple[tf.Tensor, tf.Tensor]] = []
    removed_pairs = 0
    for (minus, plus), (minus_mask, plus_mask) in zip(cases, masks):
        pair_mask = tf.logical_and(minus_mask, plus_mask)
        if remove:
            filtered.append((tf.boolean_mask(minus, pair_mask), tf.boolean_mask(plus, pair_mask)))
            removed_pairs += int(path_count) - int(tf.reduce_sum(tf.cast(pair_mask, tf.int32)).numpy())
        else:
            filtered.append((minus, plus))
    return filtered, {
        "policy": policy,
        "threshold": float(threshold),
        "generated_path_count": generated_rows,
        "invalid_path_count": invalid_rows,
        "invalid_path_rate": rate,
        "threshold_flagged": rate > float(threshold),
        "removal_applied": remove,
        "removed_pair_count": removed_pairs if remove else 0,
        "removed_path_row_count": 2 * removed_pairs if remove else 0,
    }


def trim_paired_training_rows(
    observations: tf.Tensor,
    deltas: tf.Tensor,
    labels: tf.Tensor,
    pair_ids: tf.Tensor,
    *,
    batch_size: int,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, int]:
    """Trim only complete pairs when filtering breaks batch divisibility."""

    rows = int(observations.shape[0])
    remainder = rows % int(batch_size)
    if remainder == 0:
        return observations, deltas, labels, pair_ids, 0
    target = rows - remainder
    if target < 2:
        raise ValueError("invalid-path filtering removed too many training rows")
    order = tf.argsort(pair_ids, stable=True)
    selected = order[:target]
    return (
        tf.gather(observations, selected),
        tf.gather(deltas, selected),
        tf.gather(labels, selected),
        tf.gather(pair_ids, selected),
        remainder,
    )


def conditional_dataset(
    kind: str,
    simulator: Any,
    *,
    coordinate: int,
    bundle: int,
    role: int,
    count: int,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    cases: list[tuple[tf.Tensor, tf.Tensor]] = []
    observations: list[tf.Tensor] = []
    delta_rows: list[tf.Tensor] = []
    labels: list[tf.Tensor] = []
    direction = tf.one_hot(coordinate, 3, dtype=tf.float64)
    for delta_index, delta in enumerate(DELTAS):
        minus_noise = make_noise(kind, count, (bundle, role, coordinate, delta_index, 0))
        plus_noise = make_noise(kind, count, (bundle, role, coordinate, delta_index, 1))
        minus = simulate(
            simulator, THETA - tf.cast(delta, tf.float64) * direction, minus_noise, count
        )
        plus = simulate(
            simulator, THETA + tf.cast(delta, tf.float64) * direction, plus_noise, count
        )
        cases.append((minus, plus))
        observations.extend((minus, plus))
        delta_rows.extend(
            (
                tf.fill([count], tf.cast(delta, tf.float32)),
                tf.fill([count], tf.cast(delta, tf.float32)),
            )
        )
        labels.extend((tf.zeros([count], tf.float32), tf.ones([count], tf.float32)))
    return (
        tf.concat(observations, axis=0),
        tf.concat(delta_rows, axis=0),
        tf.concat(labels, axis=0),
    )


def shared_splits(
    kind: str, simulator: Any, *, coordinate: int, bundle: int, cfg: dict[str, int]
) -> dict[str, tuple[tf.Tensor, tf.Tensor, tf.Tensor]]:
    result = {}
    for role_name, role_index in (
        ("validation", 20),
        ("calibration", 21),
        ("test", 22),
    ):
        result[role_name] = conditional_dataset(
            kind,
            simulator,
            coordinate=coordinate,
            bundle=bundle,
            role=role_index,
            count=cfg[role_name],
        )
    return result


def independent_training_dataset(
    kind: str,
    simulator: Any,
    *,
    coordinate: int,
    bundle: int,
    count: int,
    invalid_path_policy: str = "preserve",
    invalid_path_threshold: float = INVALID_PATH_RATE_THRESHOLD,
    batch_size: int = 2048,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, dict[str, Any], dict[str, Any]]:
    cases: list[tuple[tf.Tensor, tf.Tensor]] = []
    pair_id_cases: list[tuple[tf.Tensor, tf.Tensor]] = []
    pair_hashes: dict[str, Any] = {}
    direction = tf.one_hot(coordinate, 3, dtype=tf.float64)
    prefix_counts = tuple(value for value in PATH_COUNTS if value <= int(count))
    for delta_index, delta in enumerate(DELTAS):
        minus_noise = make_noise(kind, count, (bundle, 10, coordinate, delta_index, 0))
        plus_noise = make_noise(kind, count, (bundle, 10, coordinate, delta_index, 1))
        minus = simulate_in_fixed_blocks(
            simulator,
            THETA - tf.cast(delta, tf.float64) * direction,
            minus_noise,
            count,
        )
        plus = simulate_in_fixed_blocks(
            simulator,
            THETA + tf.cast(delta, tf.float64) * direction,
            plus_noise,
            count,
        )
        cases.append((minus, plus))
        ids = tf.range(count, dtype=tf.int64) + tf.cast(delta_index * count, tf.int64)
        pair_id_cases.append((ids, ids))
        pair_hashes[str(float(delta))] = {
            "minus_noise_sha256": noise_hash(minus_noise, count),
            "plus_noise_sha256": noise_hash(plus_noise, count),
            "noise_identical": all(
                bool(tf.reduce_all(left == right).numpy())
                for left, right in zip(minus_noise, plus_noise)
            ),
            "prefix_sha256": {
                str(prefix): {
                    "minus": noise_hash(minus_noise, prefix),
                    "plus": noise_hash(plus_noise, prefix),
                }
                for prefix in prefix_counts
            },
        }
    filtered, filtering = _filter_paired_cases(
        cases,
        path_count=count,
        policy=invalid_path_policy,
        threshold=invalid_path_threshold,
    )
    observations: list[tf.Tensor] = []
    delta_rows: list[tf.Tensor] = []
    labels: list[tf.Tensor] = []
    pair_ids: list[tf.Tensor] = []
    for delta_index, ((minus, plus), (minus_ids, plus_ids)) in enumerate(
        zip(filtered, pair_id_cases)
    ):
        if filtering["removal_applied"]:
            pair_mask = tf.logical_and(
                finite_path_mask(cases[delta_index][0]),
                finite_path_mask(cases[delta_index][1]),
            )
            minus_ids = tf.boolean_mask(minus_ids, pair_mask)
            plus_ids = tf.boolean_mask(plus_ids, pair_mask)
        retained = int(minus.shape[0])
        observations.extend((minus, plus))
        delta_rows.extend((tf.fill([retained], tf.cast(DELTAS[delta_index], tf.float32)),) * 2)
        labels.extend((tf.zeros([retained], tf.float32), tf.ones([retained], tf.float32)))
        pair_ids.extend((minus_ids, plus_ids))
    train_observations = tf.concat(observations, axis=0)
    train_deltas = tf.concat(delta_rows, axis=0)
    train_labels = tf.concat(labels, axis=0)
    train_pair_ids = tf.concat(pair_ids, axis=0)
    train_observations, train_deltas, train_labels, train_pair_ids, trimmed = trim_paired_training_rows(
        train_observations,
        train_deltas,
        train_labels,
        train_pair_ids,
        batch_size=batch_size,
    )
    filtering["trimmed_training_row_count"] = trimmed
    return (
        train_observations,
        train_deltas,
        train_labels,
        train_pair_ids,
        pair_hashes,
        filtering,
    )


def observation_paths(kind: str, simulator: Any, *, count: int) -> tf.Tensor:
    noise = make_noise(kind, count, (0, 30, 0, 0, 0))
    return simulate(simulator, THETA, noise, count)


def fixed_path(kind: str) -> tf.Tensor:
    return (
        gaussian.fixed_observed_path(50)
        if kind == "gaussian"
        else sir.fixed_observed_path(81120, 50)
    )[None, ...]


def gaussian_exact_scores(paths: tf.Tensor) -> tf.Tensor:
    return tf.stack(
        [
            tf.stack(
                [
                    gaussian.exact_score(THETA, paths[index, :horizon, :])[coordinate]
                    for horizon in HORIZONS
                    for coordinate in COORDINATES
                ]
            )
            for index in range(int(paths.shape[0]))
        ]
    )


def selected_controls(kind: str) -> dict[str, Any]:
    folder = "gaussian_full_attempt01" if kind == "gaussian" else "sir_full_attempt02"
    return json.loads(
        (V5_ROOT / folder / "selected_controls.json").read_text(encoding="utf-8")
    )["selected"]


def dependency_audit() -> dict[str, Any]:
    paths = (ESTIMATOR_PATH, GAUSSIAN_PATH, SIR_PATH, RUNNER_PATH)
    banned = {
        "highdim",
        "filtering",
        "filters",
        "particle",
        "particles",
        "smoothing",
        "simulation_score_tf",
    }
    violations = []
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            for module in modules:
                if set(module.lower().split(".")) & banned:
                    violations.append({"path": str(path), "module": module})
    loaded = sorted(
        name
        for name in sys.modules
        if name.startswith("bayesfilter.") and set(name.lower().split(".")) & banned
    )
    return {
        "source_violations": violations,
        "forbidden_loaded_modules": loaded,
        "passed": not violations and not loaded,
    }


def allocator_info() -> dict[str, Any]:
    if CPU_ONLY_SMOKE:
        return {"device": "CPU", "current_bytes": None, "peak_bytes": None}
    info = tf.config.experimental.get_memory_info("GPU:0")
    return {
        "device": "GPU:0",
        "current_bytes": int(info["current"]),
        "peak_bytes": int(info["peak"]),
    }


def run(
    output: Path,
    *,
    kind: str,
    bundle: int,
    path_count: int,
    profile_name: str,
    cell_filter: str | None,
    invalid_path_policy: str,
) -> None:
    cfg = profile(profile_name)
    if CPU_ONLY_SMOKE and profile_name != "smoke":
        raise ValueError("CPU-only mode is limited to smoke")
    if profile_name == "full" and cell_filter is not None:
        raise ValueError("cell filtering is not allowed for the full campaign")
    if int(path_count) not in PATH_COUNTS:
        raise ValueError("path count is not in the frozen ladder")
    if profile_name == "smoke" and int(path_count) != 8192:
        raise ValueError("CPU smoke is limited to the 8192 mechanics fixture")
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    dependencies = dependency_audit()
    if not dependencies["passed"]:
        raise RuntimeError(f"dependency audit veto: {dependencies}")

    simulator = (gaussian if kind == "gaussian" else sir).make_compiled_observation_simulator(50)
    controls = selected_controls(kind)
    audit_paths = observation_paths(kind, simulator, count=cfg["audit"])
    observed = fixed_path(kind)
    audit_outputs = tf.Variable(tf.zeros([cfg["audit"], 9], tf.float64), trainable=False)
    fixed_outputs = tf.Variable(tf.zeros([9], tf.float64), trainable=False)
    cells: dict[str, Any] = {}
    shared_hashes_by_coordinate: dict[str, Any] = {}

    for coordinate in COORDINATES:
        if cell_filter is not None and not cell_filter.endswith(f"_j{coordinate}"):
            continue
        shared_started = time.perf_counter()
        shared = shared_splits(
            kind, simulator, coordinate=coordinate, bundle=bundle, cfg=cfg
        )
        shared_seconds = time.perf_counter() - shared_started
        shared_hashes_by_coordinate[f"j{coordinate}"] = {
            name: dataset_hash(values) for name, values in shared.items()
        }
        train_started = time.perf_counter()
        train, train_deltas, train_labels, train_pair_ids, pair_hashes, training_filtering = (
            independent_training_dataset(
                kind,
                simulator,
                coordinate=coordinate,
                bundle=bundle,
                count=int(path_count),
                invalid_path_policy=invalid_path_policy,
                invalid_path_threshold=INVALID_PATH_RATE_THRESHOLD,
                batch_size=cfg["batch"],
            )
        )
        train_seconds = time.perf_counter() - train_started
        for horizon in HORIZONS:
            key = f"T{horizon}_j{coordinate}"
            if cell_filter is not None and key != cell_filter:
                continue
            control = controls[key]
            fit_started = time.perf_counter()
            fit = fit_anchored_classifier(
                train[:, :horizon, :],
                train_deltas,
                train_labels,
                validation_observations=shared["validation"][0][:, :horizon, :],
                validation_deltas=shared["validation"][1],
                validation_labels=shared["validation"][2],
                calibration_observations=shared["calibration"][0][:, :horizon, :],
                calibration_deltas=shared["calibration"][1],
                calibration_labels=shared["calibration"][2],
                test_observations=shared["test"][0][:, :horizon, :],
                test_deltas=shared["test"][1],
                test_labels=shared["test"][2],
                architecture=control["architecture"],
                seed=seed(40, bundle, horizon, coordinate),
                expected_deltas=DELTAS,
                epochs=cfg["epochs"],
                minimum_epochs=cfg["minimum"],
                patience=cfg["patience"],
                batch_size=cfg["batch"],
                l2=float(control["l2"]),
                jit_compile=True,
                train_pair_ids=train_pair_ids,
            )
            fit_seconds = time.perf_counter() - fit_started
            cell_index = HORIZONS.index(horizon) * 3 + coordinate
            audit_values = fit.score_at_observation(audit_paths[:, :horizon, :])
            fixed_value = fit.score_at_observation(observed[:, :horizon, :])[0]
            audit_outputs[:, cell_index].assign(audit_values)
            fixed_outputs[cell_index].assign(fixed_value)
            rows_per_epoch = int(train.shape[0])
            cells[key] = {
                "architecture": control["architecture"],
                "l2": control["l2"],
                "finite": bool(fit.finite.numpy())
                and bool(tf.reduce_all(tf.math.is_finite(audit_values)).numpy()),
                "temperature": float(fit.calibration_temperature.numpy()),
                "epochs_run": fit.epochs_run,
                "optimizer_updates": fit.epochs_run
                * (rows_per_epoch // int(cfg["batch"])),
                "optimizer_complete": fit.epochs_run < cfg["epochs"]
                or fit.final_ten_epoch_improvement < 1.0e-4,
                "test_log_loss": float(fit.test_log_loss.numpy()),
                "pair_hashes": pair_hashes,
                "training_filtering": training_filtering,
                "training_rows": rows_per_epoch,
                "timing_seconds": {
                    "shared_split_generation": shared_seconds,
                    "training_generation": train_seconds,
                    "fit": fit_seconds,
                },
            }
            del fit
            tf.keras.backend.clear_session()
            gc.collect()
        del train, train_deltas, train_labels, train_pair_ids, shared
        gc.collect()
        write(
            output / "partial.json",
            {
                "schema": "bayesfilter.classifier_score_path_count_bundle.partial.v1",
                "kind": kind,
                "bundle": bundle,
                "path_count": int(path_count),
                "profile": profile_name,
                "cells": cells,
                "shared_split_hashes": shared_hashes_by_coordinate,
                "audit_outputs": audit_outputs,
                "fixed_outputs": fixed_outputs,
            },
        )

    complete = cell_filter is None and profile_name == "full"
    result = {
        "schema": "bayesfilter.classifier_score_path_count_bundle.v1",
        "status": "COMPLETED" if complete else "DIAGNOSTIC_COMPLETED",
        "kind": kind,
        "bundle": bundle,
        "path_count": int(path_count),
        "training_rows_per_coordinate": int(next(iter(cells.values())).get("training_rows", 12 * int(path_count))) if cells else 0,
        "invalid_path_policy": invalid_path_policy,
        "invalid_path_rate_threshold": INVALID_PATH_RATE_THRESHOLD,
        "target_law": (
            "survivor_conditioned_training_law"
            if invalid_path_policy == "remove_invalid_paths"
            else "original_declared_simulation_law"
        ),
        "simulation_block_size": SIMULATION_BLOCK,
        "profile": profile_name,
        "cells": cells,
        "shared_split_hashes": shared_hashes_by_coordinate,
        "audit_outputs": audit_outputs,
        "fixed_outputs": fixed_outputs,
        "audit_path_sha256": tensor_hash(audit_paths),
        "fixed_path_sha256": tensor_hash(observed),
        "exact_audit_scores": gaussian_exact_scores(audit_paths)
        if kind == "gaussian"
        else None,
        "exact_fixed_score": gaussian_exact_scores(observed)[0]
        if kind == "gaussian"
        else None,
        "wall_time_seconds": time.perf_counter() - started,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "gpu_memory_policy": GPU_MEMORY_POLICY,
        "nonclaims": [
            "not exact SIR score",
            "not fixed-update sample-size scaling",
            "not filter/HMC/default evidence",
            "filtered arms are survivor-conditioned and are not the original SIR law",
        ],
    }
    write(output / "result.json", result)
    source_paths = (PLAN_PATH, ESTIMATOR_PATH, GAUSSIAN_PATH, SIR_PATH, RUNNER_PATH)
    write(
        output / "manifest.json",
        {
            "schema": "bayesfilter.classifier_score_path_count_bundle.manifest.v1",
            "status": result["status"],
            "git_commit": git_commit(),
            "command": [sys.executable, *sys.argv],
            "python": sys.executable,
            "environment": "tftwogpu",
            "kind": kind,
            "bundle": bundle,
            "path_count": int(path_count),
            "root_seed": ROOT_SEED,
            "profile": profile_name,
            "invalid_path_policy": invalid_path_policy,
            "invalid_path_rate_threshold": INVALID_PATH_RATE_THRESHOLD,
            "cpu_only_smoke": CPU_ONLY_SMOKE,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
            "selected_nvidia_smi_index": os.environ.get(
                "BAYESFILTER_SELECTED_NVIDIA_SMI_INDEX", "unset"
            ),
            "selected_gpu_uuid": os.environ.get(
                "BAYESFILTER_SELECTED_GPU_UUID", "unset"
            ),
            "selected_gpu_name": os.environ.get(
                "BAYESFILTER_SELECTED_GPU_NAME", "unset"
            ),
            "selected_gpu_utilization_percent_at_launch": os.environ.get(
                "BAYESFILTER_SELECTED_GPU_UTILIZATION", "unset"
            ),
            "selected_gpu_free_mib_at_launch": os.environ.get(
                "BAYESFILTER_SELECTED_GPU_FREE_MIB", "unset"
            ),
            "gpu_selection_reason": os.environ.get(
                "BAYESFILTER_GPU_SELECTION_REASON", "unset"
            ),
            "xla_flags": os.environ.get("XLA_FLAGS", "unset"),
            "tf32_enabled": False,
            "gpu_memory_policy": GPU_MEMORY_POLICY,
            "allocator_info": allocator_info(),
            "data_version": "v6_seed_bound_nested_prefix_ladder",
            "baseline_root": str(
                V6_ROOT
                / ("gaussian_full" if kind == "gaussian" else "sir_full")
            ),
            "audit_path_sha256": result["audit_path_sha256"],
            "fixed_path_sha256": result["fixed_path_sha256"],
            "plan": str(PLAN_PATH.relative_to(ROOT)),
            "source_hashes": {
                str(path.relative_to(ROOT)): sha(path) for path in source_paths
            },
            "dependency_audit": dependencies,
            "wall_time_seconds": result["wall_time_seconds"],
            "result_sha256": sha(output / "result.json"),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kind", choices=("gaussian", "sir"), required=True)
    parser.add_argument("--bundle", type=int, required=True)
    parser.add_argument("--path-count", type=int, choices=PATH_COUNTS, required=True)
    parser.add_argument(
        "--profile", choices=("smoke", "full_cell", "full"), default="full"
    )
    parser.add_argument(
        "--cell",
        choices=tuple(
            f"T{horizon}_j{coordinate}"
            for horizon in HORIZONS
            for coordinate in COORDINATES
        ),
    )
    parser.add_argument(
        "--invalid-path-policy",
        choices=INVALID_PATH_POLICIES,
        default="preserve",
        help="preserve the original law or run an explicitly survivor-conditioned filtered arm",
    )
    args = parser.parse_args()
    run(
        args.output,
        kind=args.kind,
        bundle=args.bundle,
        path_count=args.path_count,
        profile_name=args.profile,
        cell_filter=args.cell,
        invalid_path_policy=args.invalid_path_policy,
    )


if __name__ == "__main__":
    main()
