"""Run one paired four-arm classifier-score variance bundle."""

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

CPU_ONLY_SMOKE = os.environ.get("BAYESFILTER_CPU_ONLY_SMOKE", "false").lower() in {"1", "true", "yes"}
GPU_MEMORY_POLICY = configure_tensorflow_gpu_memory_growth(tf, require_gpu=not CPU_ONLY_SMOKE)
tf.config.experimental.enable_tensor_float_32_execution(False)

from bayesfilter.independent_score import gaussian_observation_simulator_tf as gaussian  # noqa: E402
from bayesfilter.independent_score import sir_observation_simulator_tf as sir  # noqa: E402
from bayesfilter.independent_score.anchored_orthogonal_ratio_score_tf import (  # noqa: E402
    DELTAS,
    fit_anchored_classifier,
)
from bayesfilter.independent_score.variance_reduction_tf import ARM_NAMES  # noqa: E402


THETA = tf.zeros([3], tf.float64)
HORIZONS = (20, 40, 50)
COORDINATES = (0, 1, 2)
MAX_TRAIN = 8192
ROOT_SEED = 95170
V5_ROOT = ROOT / "docs/benchmarks/artifacts/sir_null_calibrated_predictive_consistency_20260814"
PLAN_PATH = ROOT / "docs/plans/bayesfilter-classifier-score-variance-reduction-v6-plan-2026-08-15.md"
REVIEW_PATH = ROOT / "docs/plans/bayesfilter-classifier-score-variance-reduction-v6-plan-review-2026-08-15.md"
ESTIMATOR_PATH = ROOT / "bayesfilter/independent_score/anchored_orthogonal_ratio_score_tf.py"
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
    path.write_text(json.dumps(safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def seed(*parts: int) -> int:
    result = ROOT_SEED
    for part in parts:
        result = (result * 1009 + int(part) + 7919) % 2147483000
    return result


def profile(name: str) -> dict[str, int]:
    if name == "smoke":
        return {"n_small": 64, "n_large": 128, "validation": 32, "calibration": 32, "test": 64, "audit": 16, "batch": 128, "epochs": 3, "minimum": 2, "patience": 2}
    if name == "full":
        return {"n_small": 2048, "n_large": 8192, "validation": 512, "calibration": 512, "test": 1024, "audit": 128, "batch": 2048, "epochs": 80, "minimum": 15, "patience": 10}
    if name == "full_cell":
        return {"n_small": 2048, "n_large": 8192, "validation": 512, "calibration": 512, "test": 1024, "audit": 128, "batch": 2048, "epochs": 80, "minimum": 15, "patience": 10}
    if name == "capacity":
        return {"n_small": 2048, "n_large": 8192, "validation": 32, "calibration": 32, "test": 64, "audit": 16, "batch": 2048, "epochs": 3, "minimum": 2, "patience": 2}
    raise ValueError("profile must be smoke, capacity, or full")


def model_code(kind: str) -> int:
    return 1 if kind == "gaussian" else 2


def make_noise(kind: str, count: int, key: tuple[int, ...]) -> tuple[tf.Tensor, ...]:
    first = seed(model_code(kind), *key)
    if kind == "gaussian":
        return (tf.random.stateless_normal([count, 50, 9], [first, 71], dtype=tf.float64),)
    return (
        tf.random.stateless_normal([count, 18], [first, 11], dtype=tf.float64),
        tf.random.stateless_normal([count, 50, 18], [first, 13], dtype=tf.float64),
        tf.random.stateless_normal([count, 50, 9], [first, 17], dtype=tf.float64),
    )


def simulate(simulator: Any, parameters: tf.Tensor, noise: tuple[tf.Tensor, ...], count: int) -> tf.Tensor:
    return simulator(parameters, *(value[:count] for value in noise))


def conditional_dataset(
    kind: str,
    simulator: Any,
    *,
    coordinate: int,
    bundle: int,
    role: int,
    count: int,
    max_count: int,
    crn: bool,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, dict[str, Any]]:
    observations: list[tf.Tensor] = []
    delta_rows: list[tf.Tensor] = []
    labels: list[tf.Tensor] = []
    pair_ids: list[tf.Tensor] = []
    pair_hashes: dict[str, Any] = {}
    direction = tf.one_hot(coordinate, 3, dtype=tf.float64)
    for delta_index, delta in enumerate(DELTAS):
        minus_noise = make_noise(kind, max_count, (bundle, role, coordinate, delta_index, 0))
        plus_noise = minus_noise if crn else make_noise(kind, max_count, (bundle, role, coordinate, delta_index, 1))
        minus = simulate(simulator, THETA - tf.cast(delta, tf.float64) * direction, minus_noise, count)
        plus = simulate(simulator, THETA + tf.cast(delta, tf.float64) * direction, plus_noise, count)
        observations.extend((minus, plus))
        delta_rows.extend((tf.fill([count], tf.cast(delta, tf.float32)), tf.fill([count], tf.cast(delta, tf.float32))))
        labels.extend((tf.zeros([count], tf.float32), tf.ones([count], tf.float32)))
        ids = tf.range(count, dtype=tf.int64) + tf.cast(delta_index * max_count, tf.int64)
        pair_ids.extend((ids, ids))
        pair_hashes[str(float(delta))] = {
            "minus_noise_sha256": noise_hash(minus_noise, count),
            "plus_noise_sha256": noise_hash(plus_noise, count),
            "minus_prefix_sha256": noise_hash(minus_noise, min(count, 2048)),
            "plus_prefix_sha256": noise_hash(plus_noise, min(count, 2048)),
            "noise_identical": all(bool(tf.reduce_all(left[:count] == right[:count]).numpy()) for left, right in zip(minus_noise, plus_noise)),
        }
    return tf.concat(observations, axis=0), tf.concat(delta_rows, axis=0), tf.concat(labels, axis=0), tf.concat(pair_ids, axis=0), pair_hashes


def paired_training_datasets(
    kind: str,
    simulator: Any,
    *,
    coordinate: int,
    bundle: int,
    cfg: dict[str, int],
    arm_names: tuple[str, ...],
) -> dict[str, tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, dict[str, Any]]]:
    unknown = set(arm_names) - set(ARM_NAMES)
    if unknown:
        raise ValueError(f"unknown variance-reduction arms: {sorted(unknown)}")
    observations: dict[str, list[tf.Tensor]] = {arm: [] for arm in arm_names}
    delta_rows: dict[str, list[tf.Tensor]] = {arm: [] for arm in arm_names}
    labels: dict[str, list[tf.Tensor]] = {arm: [] for arm in arm_names}
    pair_ids: dict[str, list[tf.Tensor]] = {arm: [] for arm in arm_names}
    pair_hashes: dict[str, dict[str, Any]] = {arm: {} for arm in arm_names}
    need_crn = any(arm.startswith("crn") for arm in arm_names)
    need_independent = any(arm.startswith("independent") for arm in arm_names)
    direction = tf.one_hot(coordinate, 3, dtype=tf.float64)
    max_count = cfg["n_large"]
    for delta_index, delta in enumerate(DELTAS):
        minus_noise = make_noise(kind, max_count, (bundle, 10, coordinate, delta_index, 0))
        independent_plus_noise = make_noise(kind, max_count, (bundle, 10, coordinate, delta_index, 1)) if need_independent else None
        minus = simulate(simulator, THETA - tf.cast(delta, tf.float64) * direction, minus_noise, max_count)
        crn_plus = simulate(simulator, THETA + tf.cast(delta, tf.float64) * direction, minus_noise, max_count) if need_crn else None
        independent_plus = (
            simulate(simulator, THETA + tf.cast(delta, tf.float64) * direction, independent_plus_noise, max_count)
            if independent_plus_noise is not None
            else None
        )
        for arm in arm_names:
            count = cfg["n_large"] if arm.endswith("n8192") else cfg["n_small"]
            crn = arm.startswith("crn")
            plus_noise = minus_noise if crn else independent_plus_noise
            plus = crn_plus if crn else independent_plus
            if plus_noise is None or plus is None:
                raise RuntimeError(f"missing generated training bank for {arm}")
            observations[arm].extend((minus[:count], plus[:count]))
            delta_rows[arm].extend((tf.fill([count], tf.cast(delta, tf.float32)), tf.fill([count], tf.cast(delta, tf.float32))))
            labels[arm].extend((tf.zeros([count], tf.float32), tf.ones([count], tf.float32)))
            ids = tf.range(count, dtype=tf.int64) + tf.cast(delta_index * max_count, tf.int64)
            pair_ids[arm].extend((ids, ids))
            pair_hashes[arm][str(float(delta))] = {
                "minus_noise_sha256": noise_hash(minus_noise, count),
                "plus_noise_sha256": noise_hash(plus_noise, count),
                "minus_prefix_sha256": noise_hash(minus_noise, min(count, 2048)),
                "plus_prefix_sha256": noise_hash(plus_noise, min(count, 2048)),
                "noise_identical": all(
                    bool(tf.reduce_all(left[:count] == right[:count]).numpy())
                    for left, right in zip(minus_noise, plus_noise)
                ),
            }
    return {
        arm: (
            tf.concat(observations[arm], axis=0),
            tf.concat(delta_rows[arm], axis=0),
            tf.concat(labels[arm], axis=0),
            tf.concat(pair_ids[arm], axis=0),
            pair_hashes[arm],
        )
        for arm in arm_names
    }


def shared_splits(kind: str, simulator: Any, *, coordinate: int, bundle: int, cfg: dict[str, int]) -> dict[str, tuple[tf.Tensor, tf.Tensor, tf.Tensor]]:
    result = {}
    for role_name, role_index in (("validation", 20), ("calibration", 21), ("test", 22)):
        values, deltas, labels, _, _ = conditional_dataset(
            kind,
            simulator,
            coordinate=coordinate,
            bundle=bundle,
            role=role_index,
            count=cfg[role_name],
            max_count=cfg[role_name],
            crn=False,
        )
        result[role_name] = (values, deltas, labels)
    return result


def observation_paths(kind: str, simulator: Any, *, bundle: int, count: int) -> tf.Tensor:
    noise = make_noise(kind, count, (bundle, 30, 0, 0, 0))
    return simulate(simulator, THETA, noise, count)


def fixed_path(kind: str) -> tf.Tensor:
    return (gaussian.fixed_observed_path(50) if kind == "gaussian" else sir.fixed_observed_path(81120, 50))[None, ...]


def gaussian_exact_scores(paths: tf.Tensor) -> tf.Tensor:
    return tf.stack(
        [
            tf.stack(
                [
                    gaussian.exact_score(THETA, paths[path_index, :horizon, :])[coordinate]
                    for horizon in HORIZONS
                    for coordinate in COORDINATES
                ]
            )
            for path_index in range(int(paths.shape[0]))
        ]
    )


def selected_controls(kind: str) -> dict[str, Any]:
    folder = "gaussian_full_attempt01" if kind == "gaussian" else "sir_full_attempt02"
    path = V5_ROOT / folder / "selected_controls.json"
    return json.loads(path.read_text(encoding="utf-8"))["selected"]


def tensor_hash(value: tf.Tensor) -> str:
    return hashlib.sha256(tf.io.serialize_tensor(value).numpy()).hexdigest()


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


def dependency_audit() -> dict[str, Any]:
    paths = (
        ESTIMATOR_PATH,
        ROOT / "bayesfilter/independent_score/gaussian_observation_simulator_tf.py",
        ROOT / "bayesfilter/independent_score/sir_observation_simulator_tf.py",
        RUNNER_PATH,
    )
    banned = {"highdim", "filtering", "filters", "particle", "particles", "smoothing", "simulation_score_tf"}
    source_violations = []
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
                    source_violations.append({"path": str(path), "module": module})
    loaded = sorted(
        name
        for name in sys.modules
        if name.startswith("bayesfilter.") and set(name.lower().split(".")) & banned
    )
    return {
        "source_violations": source_violations,
        "forbidden_loaded_modules": loaded,
        "passed": not source_violations and not loaded,
    }


def allocator_info() -> dict[str, Any]:
    if CPU_ONLY_SMOKE:
        return {"device": "CPU", "current_bytes": None, "peak_bytes": None}
    try:
        info = tf.config.experimental.get_memory_info("GPU:0")
        return {"device": "GPU:0", "current_bytes": int(info["current"]), "peak_bytes": int(info["peak"])}
    except (ValueError, RuntimeError) as error:
        return {"device": "GPU:0", "error": str(error), "current_bytes": None, "peak_bytes": None}


def run(output: Path, *, kind: str, bundle: int, profile_name: str, arm_filter: str | None, cell_filter: str | None) -> None:
    cfg = profile(profile_name)
    if CPU_ONLY_SMOKE and profile_name != "smoke":
        raise ValueError("CPU-only mode is limited to smoke")
    if cell_filter is not None and profile_name == "full":
        raise ValueError("cell filtering is not allowed for the full campaign")
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    dependencies = dependency_audit()
    if not dependencies["passed"]:
        raise RuntimeError(f"dependency audit veto: {dependencies}")
    simulator = (gaussian if kind == "gaussian" else sir).make_compiled_observation_simulator(50)
    controls = selected_controls(kind)
    audit_paths = observation_paths(kind, simulator, bundle=0, count=cfg["audit"])
    observed = fixed_path(kind)
    audit_outputs = tf.Variable(tf.zeros([len(ARM_NAMES), cfg["audit"], 9], tf.float64), trainable=False)
    fixed_outputs = tf.Variable(tf.zeros([len(ARM_NAMES), 9], tf.float64), trainable=False)
    selected_arms = ARM_NAMES if arm_filter is None else (arm_filter,)
    arm_rows: dict[str, Any] = {
        arm_name: {
            "train_count": cfg["n_large"] if arm_name.endswith("n8192") else cfg["n_small"],
            "crn": arm_name.startswith("crn"),
            "cells": {},
            "shared_split_hashes": {},
        }
        for arm_name in selected_arms
    }
    for coordinate in COORDINATES:
        if cell_filter is not None and not cell_filter.endswith(f"_j{coordinate}"):
            continue
        shared_started = time.perf_counter()
        shared = shared_splits(kind, simulator, coordinate=coordinate, bundle=bundle, cfg=cfg)
        shared_seconds = time.perf_counter() - shared_started
        shared_hashes = {name: dataset_hash(values) for name, values in shared.items()}
        train_started = time.perf_counter()
        training_by_arm = paired_training_datasets(
            kind,
            simulator,
            coordinate=coordinate,
            bundle=bundle,
            cfg=cfg,
            arm_names=selected_arms,
        )
        train_generation_seconds = time.perf_counter() - train_started
        for arm_position, arm_name in enumerate(selected_arms):
            arm_index = ARM_NAMES.index(arm_name)
            arm_rows[arm_name]["shared_split_hashes"][f"j{coordinate}"] = shared_hashes
            train, train_deltas, train_labels, train_pair_ids, pair_hashes = training_by_arm[arm_name]
            for horizon in HORIZONS:
                key = f"T{horizon}_j{coordinate}"
                if cell_filter is not None and key != cell_filter:
                    continue
                control = controls[key]
                fit_started = time.perf_counter()
                fit = fit_anchored_classifier(
                    train[:, :horizon, :], train_deltas, train_labels,
                    validation_observations=shared["validation"][0][:, :horizon, :], validation_deltas=shared["validation"][1], validation_labels=shared["validation"][2],
                    calibration_observations=shared["calibration"][0][:, :horizon, :], calibration_deltas=shared["calibration"][1], calibration_labels=shared["calibration"][2],
                    test_observations=shared["test"][0][:, :horizon, :], test_deltas=shared["test"][1], test_labels=shared["test"][2],
                    architecture=control["architecture"], seed=seed(40, bundle, horizon, coordinate), expected_deltas=DELTAS,
                    epochs=cfg["epochs"], minimum_epochs=cfg["minimum"], patience=cfg["patience"], batch_size=cfg["batch"], l2=float(control["l2"]), jit_compile=True,
                    train_pair_ids=train_pair_ids,
                )
                fit_seconds = time.perf_counter() - fit_started
                cell_index = HORIZONS.index(horizon) * 3 + coordinate
                evaluation_started = time.perf_counter()
                audit_values = fit.score_at_observation(audit_paths[:, :horizon, :])
                fixed_value = fit.score_at_observation(observed[:, :horizon, :])[0]
                audit_outputs[arm_index, :, cell_index].assign(audit_values)
                fixed_outputs[arm_index, cell_index].assign(fixed_value)
                evaluation_seconds = time.perf_counter() - evaluation_started
                arm_rows[arm_name]["cells"][key] = {
                    "architecture": control["architecture"],
                    "l2": control["l2"],
                    "finite": bool(fit.finite.numpy()) and bool(tf.reduce_all(tf.math.is_finite(audit_values)).numpy()),
                    "temperature": float(fit.calibration_temperature.numpy()),
                    "epochs_run": fit.epochs_run,
                    "optimizer_complete": fit.epochs_run < cfg["epochs"] or fit.final_ten_epoch_improvement < 1.0e-4,
                    "test_log_loss": float(fit.test_log_loss.numpy()),
                    "timing_seconds": {
                        "shared_split_generation": shared_seconds,
                        "shared_split_cache_hit": arm_position > 0,
                        "training_generation": train_generation_seconds,
                        "training_bank_cache_hit": arm_position > 0,
                        "fit": fit_seconds,
                        "evaluation": evaluation_seconds,
                    },
                    "pair_hashes": pair_hashes,
                }
                del fit
                tf.keras.backend.clear_session()
                gc.collect()
            del train, train_deltas, train_labels, train_pair_ids
            gc.collect()
        del training_by_arm, shared
        gc.collect()
        partial = {
            "schema": "bayesfilter.classifier_score_variance_bundle.partial.v1",
            "kind": kind,
            "bundle": bundle,
            "profile": profile_name,
            "completed_arms": list(arm_rows),
            "arm_rows": arm_rows,
            "audit_outputs": audit_outputs,
            "fixed_outputs": fixed_outputs,
        }
        write(output / "partial.json", partial)

    result = {
        "schema": "bayesfilter.classifier_score_variance_bundle.v1",
        "status": "COMPLETED" if arm_filter is None and cell_filter is None else "SMOKE_COMPLETED",
        "kind": kind,
        "bundle": bundle,
        "profile": profile_name,
        "arm_names": ARM_NAMES,
        "completed_arms": list(arm_rows),
        "arm_rows": arm_rows,
        "audit_outputs": audit_outputs,
        "fixed_outputs": fixed_outputs,
        "audit_path_sha256": tensor_hash(audit_paths),
        "fixed_path_sha256": tensor_hash(observed),
        "exact_audit_scores": gaussian_exact_scores(audit_paths) if kind == "gaussian" else None,
        "exact_fixed_score": gaussian_exact_scores(observed)[0] if kind == "gaussian" else None,
        "wall_time_seconds": time.perf_counter() - started,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "gpu_memory_policy": GPU_MEMORY_POLICY,
        "nonclaims": ["not exact SIR score", "not natural path variance", "not filter/HMC/default evidence"],
    }
    write(output / "result.json", result)
    write(
        output / "manifest.json",
        {
            "schema": "bayesfilter.classifier_score_variance_bundle.manifest.v1",
            "status": result["status"],
            "git_commit": git_commit(),
            "command": [sys.executable, *sys.argv],
            "python": sys.executable,
            "environment": "tftwogpu",
            "kind": kind,
            "bundle": bundle,
            "root_seed": ROOT_SEED,
            "profile": profile_name,
            "cpu_only_smoke": CPU_ONLY_SMOKE,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
            "xla_flags": os.environ.get("XLA_FLAGS", "unset"),
            "tf32_enabled": False,
            "gpu_memory_policy": GPU_MEMORY_POLICY,
            "allocator_info": allocator_info(),
            "data_version": "simulator_and_seed_bound",
            "audit_path_sha256": result["audit_path_sha256"],
            "fixed_path_sha256": result["fixed_path_sha256"],
            "plan": str(PLAN_PATH.relative_to(ROOT)),
            "review": str(REVIEW_PATH.relative_to(ROOT)),
            "source_hashes": {
                str(path.relative_to(ROOT)): sha(path)
                for path in (PLAN_PATH, REVIEW_PATH, ESTIMATOR_PATH, RUNNER_PATH)
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
    parser.add_argument("--profile", choices=("smoke", "capacity", "full_cell", "full"), default="full")
    parser.add_argument("--arm", choices=ARM_NAMES)
    parser.add_argument("--cell", choices=tuple(f"T{horizon}_j{coordinate}" for horizon in HORIZONS for coordinate in COORDINATES))
    args = parser.parse_args()
    run(args.output, kind=args.kind, bundle=args.bundle, profile_name=args.profile, arm_filter=args.arm, cell_filter=args.cell)


if __name__ == "__main__":
    main()
