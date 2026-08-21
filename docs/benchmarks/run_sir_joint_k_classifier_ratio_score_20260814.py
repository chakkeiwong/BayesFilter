"""Run the V3 joint-k observation-only classifier-ratio score campaign."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
import tensorflow as tf

from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth

GPU_MEMORY_POLICY = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
tf.config.experimental.enable_tensor_float_32_execution(False)
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bayesfilter.independent_score import gaussian_observation_simulator_tf as gaussian_simulator  # noqa: E402
from bayesfilter.independent_score import sir_observation_simulator_tf as sir_simulator  # noqa: E402
from bayesfilter.independent_score.joint_k_classifier_ratio_score_tf import (  # noqa: E402
    ARCHITECTURES,
    DELTA_SCALE,
    fit_joint_k_classifier,
)


THETA = tf.zeros([3], tf.float64)
HORIZONS = (20, 40, 50)
DELTAS = (0.01, 0.02, 0.03, 0.04)
COORDINATES = (0, 1, 2)
REGULARIZATION = (0.0, 1.0e-5)
FINAL_REPLICATES = 3
ROOT_SEED = 89400
DEFAULT_OUTPUT = ROOT / "docs/benchmarks/artifacts/sir_joint_k_classifier_ratio_score_20260814"
TRAIN_COUNT = 2048
VALIDATION_COUNT = 512
CALIBRATION_COUNT = 512
TEST_COUNT = 1024
BATCH_SIZE = 2048


def _safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    if isinstance(value, tf.Tensor):
        return _safe(value.numpy().tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _seed(*parts: int) -> int:
    value = ROOT_SEED
    for part in parts:
        value = (value * 1009 + int(part) + 7919) % 2147483000
    return value


def _profile(profile: str) -> dict[str, int]:
    if profile == "full":
        return {"train": TRAIN_COUNT, "validation": VALIDATION_COUNT, "calibration": CALIBRATION_COUNT, "test": TEST_COUNT, "batch_size": BATCH_SIZE, "epochs": 80, "minimum_epochs": 15, "patience": 10}
    if profile == "smoke":
        return {"train": 64, "validation": 32, "calibration": 32, "test": 64, "batch_size": 128, "epochs": 4, "minimum_epochs": 2, "patience": 2}
    raise ValueError("profile must be full or smoke")


def _forbidden_loaded_modules() -> list[str]:
    forbidden_tokens = {"highdim", "filtering", "filters", "particle", "particles", "smoothing", "simulation_score_tf"}
    return [name for name in sorted(sys.modules) if name.startswith("bayesfilter.") and (set(name.lower().split(".")) & forbidden_tokens)]


def _source_audit() -> dict[str, Any]:
    paths = [
        ROOT / "bayesfilter/independent_score/joint_k_classifier_ratio_score_tf.py",
        ROOT / "bayesfilter/independent_score/gaussian_observation_simulator_tf.py",
        ROOT / "bayesfilter/independent_score/sir_observation_simulator_tf.py",
        Path(__file__).resolve(),
    ]
    violations = []
    imports: dict[str, list[str]] = {}
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        modules = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                modules.append(node.module or "")
        imports[path.name] = modules
        for module in modules:
            if set(module.lower().split(".")) & {"highdim", "filtering", "filters", "particle", "particles", "smoothing", "simulation_score_tf"}:
                violations.append({"path": str(path), "module": module})
    return {"imports": imports, "violations": violations, "passed": not violations}


def _noise(count: int, seed: int) -> tuple[tf.Tensor, ...]:
    return (
        tf.random.stateless_normal([count, 18], [_seed(seed, 1), 11], dtype=tf.float64),
        tf.random.stateless_normal([count, 50, 18], [_seed(seed, 2), 13], dtype=tf.float64),
        tf.random.stateless_normal([count, 50, 9], [_seed(seed, 3), 17], dtype=tf.float64),
    )


def _dataset(
    simulator_kind: str,
    *,
    coordinate: int,
    role: int,
    replicate: int,
    count_per_delta: int,
    data_domain: int,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    observations = []
    deltas = []
    labels = []
    for delta_index, delta in enumerate(DELTAS):
        seed_base = _seed(data_domain, coordinate, delta_index, role, replicate)
        if simulator_kind == "sir":
            simulator = sir_simulator.make_compiled_observation_simulator(50)
            minus_noise = _noise(count_per_delta, _seed(seed_base, 0))
            plus_noise = _noise(count_per_delta, _seed(seed_base, 1))
            direction = tf.one_hot(coordinate, 3, dtype=tf.float64)
            minus = simulator(THETA - tf.cast(delta, tf.float64) * direction, *minus_noise)
            plus = simulator(THETA + tf.cast(delta, tf.float64) * direction, *plus_noise)
        else:
            simulator = gaussian_simulator.make_compiled_observation_simulator(50)
            minus_noise = tf.random.stateless_normal([count_per_delta, 50, 9], [_seed(seed_base, 0), 31], dtype=tf.float64)
            plus_noise = tf.random.stateless_normal([count_per_delta, 50, 9], [_seed(seed_base, 1), 37], dtype=tf.float64)
            direction = tf.one_hot(coordinate, 3, dtype=tf.float64)
            minus = simulator(THETA - tf.cast(delta, tf.float64) * direction, minus_noise)
            plus = simulator(THETA + tf.cast(delta, tf.float64) * direction, plus_noise)
        observations.extend([minus, plus])
        deltas.extend([tf.fill([count_per_delta], tf.cast(delta, tf.float32)), tf.fill([count_per_delta], tf.cast(delta, tf.float32))])
        labels.extend([tf.zeros([count_per_delta], tf.float32), tf.ones([count_per_delta], tf.float32)])
    return tf.concat(observations, 0), tf.concat(deltas, 0), tf.concat(labels, 0)


def _splits(simulator_kind: str, *, coordinate: int, replicate: int, data_domain: int, config: dict[str, int]) -> dict[str, tuple[tf.Tensor, tf.Tensor, tf.Tensor]]:
    return {
        role_name: _dataset(simulator_kind, coordinate=coordinate, role=role, replicate=replicate, count_per_delta=config[role_name], data_domain=data_domain)
        for role_name, role in (("train", 1), ("validation", 2), ("calibration", 3), ("test", 4))
    }


def _fit_row(splits: dict[str, tuple[tf.Tensor, tf.Tensor, tf.Tensor]], *, stage: str, horizon: int, coordinate: int, replicate: int, architecture: str, l2: float, config: dict[str, int]) -> dict[str, Any]:
    train = splits["train"]
    validation = splits["validation"]
    calibration = splits["calibration"]
    test = splits["test"]
    fit = fit_joint_k_classifier(
        train[0][:, :horizon, :], train[1], train[2],
        validation_observations=validation[0][:, :horizon, :], validation_deltas=validation[1], validation_labels=validation[2],
        calibration_observations=calibration[0][:, :horizon, :], calibration_deltas=calibration[1], calibration_labels=calibration[2],
        test_observations=test[0][:, :horizon, :], test_deltas=test[1], test_labels=test[2],
        expected_deltas=DELTAS, architecture=architecture, seed=_seed(700 if stage == "exact_oracle" else 800, horizon, coordinate, replicate),
        epochs=config["epochs"], minimum_epochs=config["minimum_epochs"], patience=config["patience"], batch_size=config["batch_size"], l2=l2, jit_compile=True,
    )
    observed = (gaussian_simulator.fixed_observed_path(horizon) if stage == "exact_oracle" else sir_simulator.fixed_observed_path(81120, horizon))[None, ...]
    score = fit.score_at_observation(observed)[0]
    pooled_signal = float(fit.test_log_loss.numpy()) < math.log(2.0) - 2.0 * float(fit.test_log_loss_standard_error.numpy())
    calibration_not_worse = float(fit.calibration_log_loss_after.numpy()) <= float(fit.calibration_log_loss_before.numpy()) + 1.0e-4
    per_delta = {}
    for delta in DELTAS:
        key = str(float(delta))
        auc = float(fit.test_auc_by_delta[key].numpy())
        ece = float(fit.test_ece_by_delta[key].numpy())
        logit = float(fit.calibrated_logit(observed, tf.constant([delta], tf.float32))[0].numpy())
        lower = float(fit.test_logit_minimum_by_delta[key].numpy())
        upper = float(fit.test_logit_maximum_by_delta[key].numpy())
        expansion = 0.1 * max(upper - lower, 1.0)
        per_delta[key] = {"auc": auc, "ece": ece, "observed_logit": logit, "logit_support": [lower, upper], "support": lower - expansion <= logit <= upper + expansion}
    aucs = [per_delta[str(float(delta))]["auc"] for delta in DELTAS]
    eces = [per_delta[str(float(delta))]["ece"] for delta in DELTAS]
    admission = {
        "finite": bool(fit.finite.numpy()),
        "pooled_signal": pooled_signal,
        "calibration_not_worse": calibration_not_worse,
        "temperature_positive": float(fit.calibration_temperature.numpy()) > 0.0,
        "per_delta_ece": max(eces) <= 0.04,
        "at_least_two_informative_deltas": sum(auc > 0.52 for auc in aucs) >= 2,
        "auc_not_inverted": all(aucs[i + 1] >= aucs[i] - 0.03 for i in range(len(aucs) - 1)),
        "maximum_delta_not_separated": aucs[-1] <= 0.995,
        "support_all_deltas": all(item["support"] for item in per_delta.values()),
        "optimizer_complete": fit.epochs_run < config["epochs"] or fit.final_ten_epoch_improvement < 1.0e-4,
    }
    return {
        "stage": stage, "horizon": horizon, "coordinate": coordinate, "replicate": replicate,
        "architecture": architecture, "l2": l2, "best_epoch": fit.best_epoch, "epochs_run": fit.epochs_run,
        "final_ten_epoch_improvement": fit.final_ten_epoch_improvement,
        "validation_log_loss": float(fit.validation_log_loss.numpy()), "validation_log_loss_standard_error": float(fit.validation_log_loss_standard_error.numpy()),
        "calibration_log_loss_before": float(fit.calibration_log_loss_before.numpy()), "calibration_log_loss_after": float(fit.calibration_log_loss_after.numpy()),
        "calibration_temperature": float(fit.calibration_temperature.numpy()), "test_log_loss": float(fit.test_log_loss.numpy()), "test_log_loss_standard_error": float(fit.test_log_loss_standard_error.numpy()),
        "score_estimate": float(score.numpy()), "per_delta": per_delta, "admission": admission, "admitted": all(admission.values()),
    }


def _select(rows: list[dict[str, Any]], horizon: int, coordinate: int) -> dict[str, Any]:
    candidates = [row for row in rows if row["horizon"] == horizon and row["coordinate"] == coordinate]
    candidates.sort(key=lambda row: (row["validation_log_loss"], row["architecture"], row["l2"]))
    if not candidates:
        raise ValueError("missing selection rows")
    best = candidates[0]
    simpler = [row for row in candidates if row["architecture"] == "joint_linear_quadratic_odd5" and row["validation_log_loss"] <= best["validation_log_loss"] + best["validation_log_loss_standard_error"]]
    selected = simpler[0] if simpler else best
    return {"architecture": selected["architecture"], "l2": selected["l2"], "validation_log_loss": selected["validation_log_loss"], "candidates": candidates}


def _summarize(rows: list[dict[str, Any]], stage: str) -> tuple[dict[str, Any], bool]:
    summary = {}
    all_pass = True
    for horizon in HORIZONS:
        for coordinate in COORDINATES:
            selected = [row for row in rows if row["horizon"] == horizon and row["coordinate"] == coordinate and row["admitted"]]
            values = [row["score_estimate"] for row in selected]
            mean = sum(values) / len(values) if values else None
            se = (sum((v - mean) ** 2 for v in values) / (len(values) * (len(values) - 1))) ** 0.5 if len(values) > 1 else None
            range_value = max(values) - min(values) if values else None
            gates = {"three_admitted_replicates": len(values) == FINAL_REPLICATES, "finite": mean is not None and se is not None, "replicate_range": range_value is not None and range_value <= max(2.0, 4.0 * se) if se is not None else False, "precision": se is not None and se <= max(1.0, 0.25 * abs(mean)) if mean is not None else False}
            exact_score = None
            if stage == "exact_oracle":
                exact_score = float(gaussian_simulator.exact_score(THETA, gaussian_simulator.fixed_observed_path(horizon))[coordinate].numpy())
                gates["exact_score_error"] = mean is not None and se is not None and abs(mean - exact_score) <= max(0.5, 3.0 * se)
            admitted = all(gates.values())
            all_pass = all_pass and admitted
            summary[f"T{horizon}_j{coordinate}"] = {"replicate_scores": values, "mean": mean, "standard_error": se, "range": range_value, "exact_score": exact_score, "gates": gates, "reference_admitted": admitted, "status": "admitted" if admitted else "no_joint_k_ratio_reference"}
    return summary, all_pass


def run(output_root: Path, *, stage: str, profile: str, oracle_result: Path | None) -> None:
    if stage not in {"exact_oracle", "sir"}:
        raise ValueError("invalid stage")
    if stage == "sir":
        if oracle_result is None:
            raise ValueError("SIR requires --oracle-result")
        payload = json.loads(oracle_result.read_text())
        if payload.get("status") != "PASSED":
            raise ValueError("SIR requires a passed exact oracle")
    config = _profile(profile)
    audit = _source_audit()
    if not audit["passed"] or _forbidden_loaded_modules():
        raise RuntimeError("dependency veto")
    output_root.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    simulator_kind = "gaussian" if stage == "exact_oracle" else "sir"
    manifest = {"schema": "bayesfilter.joint_k_classifier_ratio_score.manifest.v1", "status": "RUNNING", "stage": stage, "profile": profile, "method": "conditional_joint_k_odd5_classifier_ratio", "score_identity": "calibrated_c1_at_observation/(2*delta_scale)", "delta_scale": DELTA_SCALE, "deltas": DELTAS, "horizons": HORIZONS, "coordinates": COORDINATES, "train_count_per_delta_per_class": config["train"], "pooled_training_rows": config["train"] * 2 * len(DELTAS), "batch_size": config["batch_size"], "selection_data_domain": 30, "final_data_domain": 40, "gpu_memory_policy": GPU_MEMORY_POLICY, "python": sys.executable, "python_version": platform.python_version(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"), "xla_flags": os.environ.get("XLA_FLAGS", "unset"), "source_audit": audit, "oracle_result": str(oracle_result) if oracle_result else None, "source_hashes": {path.name: _sha256(path) for path in [ROOT / "bayesfilter/independent_score/joint_k_classifier_ratio_score_tf.py", ROOT / "docs/plans/bayesfilter-sir-joint-k-classifier-ratio-score-v3-plan-2026-08-14.md", Path(__file__).resolve()]}}
    _write_json(output_root / "run_manifest.json", manifest)
    selection_rows = []
    for coordinate in COORDINATES:
        splits = _splits(simulator_kind, coordinate=coordinate, replicate=0, data_domain=30, config=config)
        for horizon in HORIZONS:
            for architecture in ARCHITECTURES:
                for l2 in REGULARIZATION:
                    selection_rows.append(_fit_row(splits, stage=stage, horizon=horizon, coordinate=coordinate, replicate=0, architecture=architecture, l2=l2, config=config))
    selected = {f"T{horizon}_j{coordinate}": _select(selection_rows, horizon, coordinate) for horizon in HORIZONS for coordinate in COORDINATES}
    _write_json(output_root / "selected_controls.json", {"selected": selected, "rows": selection_rows})
    final_rows = []
    for coordinate in COORDINATES:
        for replicate in range(FINAL_REPLICATES):
            splits = _splits(simulator_kind, coordinate=coordinate, replicate=replicate, data_domain=40, config=config)
            for horizon in HORIZONS:
                controls = selected[f"T{horizon}_j{coordinate}"]
                row = _fit_row(splits, stage=stage, horizon=horizon, coordinate=coordinate, replicate=replicate, architecture=controls["architecture"], l2=float(controls["l2"]), config=config)
                final_rows.append(row)
                _write_json(output_root / f"row_{len(final_rows)-1:04d}.json", row)
    summary, all_pass = _summarize(final_rows, stage)
    runtime = _forbidden_loaded_modules()
    result = {"schema": "bayesfilter.joint_k_classifier_ratio_score.result.v1", "stage": stage, "profile": profile, "status": "PASSED" if all_pass and profile == "full" and not runtime else ("SMOKE_COMPLETED" if profile == "smoke" else "FAILED"), "all_reference_cells_admitted": all_pass and not runtime, "selected": selected, "summary": summary, "rows": len(final_rows), "runtime_dependency_violations": runtime, "nonclaims": ["not exact SIR score", "not filter correctness or ranking", "not HMC/default readiness"]}
    _write_json(output_root / "result.json", result)
    manifest.update({"status": result["status"], "wall_time_seconds": time.perf_counter() - started, "finished_at": datetime.now(timezone.utc).isoformat(), "result_sha256": _sha256(output_root / "result.json")})
    _write_json(output_root / "run_manifest.json", manifest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--stage", choices=("exact_oracle", "sir"), required=True)
    parser.add_argument("--profile", choices=("full", "smoke"), default="full")
    parser.add_argument("--oracle-result", type=Path)
    args = parser.parse_args()
    run(args.output_root, stage=args.stage, profile=args.profile, oracle_result=args.oracle_result)


if __name__ == "__main__":
    main()
