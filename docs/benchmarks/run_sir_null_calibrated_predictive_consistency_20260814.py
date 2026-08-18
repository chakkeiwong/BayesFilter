"""V5 joint null-calibrated predictive-consistency campaign.

The primary result is same-parameter predictive coverage for a frozen
observation-only ratio estimator. Gaussian exact scores are diagnostics only.
"""

from __future__ import annotations

import argparse
import ast
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

from bayesfilter.independent_score import gaussian_observation_simulator_tf as gaussian_simulator  # noqa: E402
from bayesfilter.independent_score import sir_observation_simulator_tf as sir_simulator  # noqa: E402
from bayesfilter.independent_score.anchored_orthogonal_ratio_score_tf import (  # noqa: E402
    ARCHITECTURES,
    DELTA_SCALE,
    DELTAS,
    basis_diagnostics,
    fit_anchored_classifier,
)
from bayesfilter.independent_score.null_calibration_tf import (  # noqa: E402
    audit_failure,
    conformal_threshold,
    fit_svd_geometry,
    zero_mean_max_t_diagnostic,
)


THETA = tf.zeros([3], tf.float64)
HORIZONS = (20, 40, 50)
COORDINATES = (0, 1, 2)
REGULARIZATION = (0.0, 1.0e-5)
ROOT_SEED = 92170
SELECTION_DOMAIN = 50
TRAIN_DOMAIN = 60
NULL_FIT_DOMAIN = 70
NULL_CALIBRATION_DOMAIN = 80
NULL_AUDIT_DOMAIN = 90
OUTPUT_DEFAULT = ROOT / "docs/benchmarks/artifacts/sir_null_calibrated_predictive_consistency_20260814"

PLAN_PATH = ROOT / "docs/plans/bayesfilter-sir-null-calibrated-predictive-consistency-v5-plan-2026-08-14.md"
REVIEW_PATH = ROOT / "docs/plans/bayesfilter-sir-null-calibrated-predictive-consistency-v5-plan-review-2026-08-14.md"
REPAIR_PATH = ROOT / "docs/plans/bayesfilter-sir-null-calibrated-predictive-consistency-v5p1-repair-2026-08-14.md"
IMPLEMENTATION_PATH = ROOT / "bayesfilter/independent_score/null_calibration_tf.py"
ESTIMATOR_PATH = ROOT / "bayesfilter/independent_score/anchored_orthogonal_ratio_score_tf.py"
RUNNER_PATH = Path(__file__).resolve()


def safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(item) for item in value]
    if isinstance(value, tf.Tensor):
        return safe(value.numpy().tolist())
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
        return {"train": 64, "validation": 32, "calibration": 32, "test": 64, "batch": 128, "epochs": 4, "minimum": 2, "patience": 2, "null_fit": 16, "null_calibration": 16, "null_audit": 32, "bootstrap": 128}
    if name == "full":
        return {"train": 2048, "validation": 512, "calibration": 512, "test": 1024, "batch": 2048, "epochs": 80, "minimum": 15, "patience": 10, "null_fit": 500, "null_calibration": 200, "null_audit": 500, "bootstrap": 1000}
    raise ValueError("profile must be smoke or full")


def forbidden_loaded_modules() -> list[str]:
    tokens = ("highdim", "filtering", "filters", "particle", "particles", "smoothing", "simulation_score_tf")
    return sorted(name for name in sys.modules if name.startswith("bayesfilter.") and any(token in name.lower().split(".") for token in tokens))


def validate_domains() -> None:
    domains = {
        "selection": SELECTION_DOMAIN,
        "training": TRAIN_DOMAIN,
        "null_fit": NULL_FIT_DOMAIN,
        "null_calibration": NULL_CALIBRATION_DOMAIN,
        "null_audit": NULL_AUDIT_DOMAIN,
    }
    if len(set(domains.values())) != len(domains):
        raise ValueError(f"selection/training/null domains must be distinct: {domains}")


def source_audit() -> dict[str, Any]:
    paths = (IMPLEMENTATION_PATH, ESTIMATOR_PATH, ROOT / "bayesfilter/independent_score/gaussian_observation_simulator_tf.py", ROOT / "bayesfilter/independent_score/sir_observation_simulator_tf.py", RUNNER_PATH)
    banned = {"highdim", "filtering", "filters", "particle", "particles", "smoothing", "simulation_score_tf"}
    imports: dict[str, list[str]] = {}
    violations: list[dict[str, str]] = []
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        modules: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                modules.append(node.module or "")
        imports[path.name] = modules
        for module in modules:
            if set(module.lower().split(".")) & banned:
                violations.append({"path": str(path), "module": module})
    return {"imports": imports, "violations": violations, "passed": not violations}


def noise(count: int, value: int) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    return (
        tf.random.stateless_normal([count, 18], [seed(value, 1), 11], dtype=tf.float64),
        tf.random.stateless_normal([count, 50, 18], [seed(value, 2), 13], dtype=tf.float64),
        tf.random.stateless_normal([count, 50, 9], [seed(value, 3), 17], dtype=tf.float64),
    )


SIMULATORS = {50: None}


def initialize_simulators() -> None:
    # Static-horizon compiled functions are cached once to avoid retracing in
    # the nested delta/partition loops.
    SIMULATORS[50] = {
        "gaussian": gaussian_simulator.make_compiled_observation_simulator(50),
        "sir": sir_simulator.make_compiled_observation_simulator(50),
    }


def conditional_dataset(kind: str, *, coordinate: int, count: int, domain: int, role: int, replicate: int) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    xs: list[tf.Tensor] = []
    deltas: list[tf.Tensor] = []
    labels: list[tf.Tensor] = []
    direction = tf.one_hot(coordinate, 3, dtype=tf.float64)
    simulator = SIMULATORS[50][kind]
    for delta_index, delta in enumerate(DELTAS):
        base = seed(domain, role, replicate, coordinate, delta_index)
        if kind == "gaussian":
            minus = simulator(THETA - tf.cast(delta, tf.float64) * direction, tf.random.stateless_normal([count, 50, 9], [seed(base, 0), 31], dtype=tf.float64))
            plus = simulator(THETA + tf.cast(delta, tf.float64) * direction, tf.random.stateless_normal([count, 50, 9], [seed(base, 1), 37], dtype=tf.float64))
        else:
            minus = simulator(THETA - tf.cast(delta, tf.float64) * direction, *noise(count, seed(base, 0)))
            plus = simulator(THETA + tf.cast(delta, tf.float64) * direction, *noise(count, seed(base, 1)))
        xs.extend((minus, plus))
        deltas.extend((tf.fill([count], tf.cast(delta, tf.float32)), tf.fill([count], tf.cast(delta, tf.float32))))
        labels.extend((tf.zeros([count], tf.float32), tf.ones([count], tf.float32)))
    return tf.concat(xs, 0), tf.concat(deltas, 0), tf.concat(labels, 0)


def conditional_splits(kind: str, *, coordinate: int, domain: int, replicate: int, cfg: dict[str, int]) -> dict[str, tuple[tf.Tensor, tf.Tensor, tf.Tensor]]:
    return {name: conditional_dataset(kind, coordinate=coordinate, count=cfg[name], domain=domain, role=role, replicate=replicate) for name, role in (("train", 1), ("validation", 2), ("calibration", 3), ("test", 4))}


def null_paths(kind: str, *, count: int, domain: int, role: int) -> tf.Tensor:
    if kind == "gaussian":
        return SIMULATORS[50][kind](THETA, tf.random.stateless_normal([count, 50, 9], [seed(domain, role, 1), 71], dtype=tf.float64))
    return SIMULATORS[50][kind](THETA, *noise(count, seed(domain, role, 2)))


def fit_selection(kind: str, cfg: dict[str, int]) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for coordinate in COORDINATES:
        splits = conditional_splits(kind, coordinate=coordinate, domain=SELECTION_DOMAIN, replicate=0, cfg=cfg)
        for horizon in HORIZONS:
            candidates: list[dict[str, Any]] = []
            for architecture in ARCHITECTURES:
                for l2 in REGULARIZATION:
                    train, validation, calibration, test = (splits[name] for name in ("train", "validation", "calibration", "test"))
                    fit = fit_anchored_classifier(
                        train[0][:, :horizon, :], train[1], train[2],
                        validation_observations=validation[0][:, :horizon, :], validation_deltas=validation[1], validation_labels=validation[2],
                        calibration_observations=calibration[0][:, :horizon, :], calibration_deltas=calibration[1], calibration_labels=calibration[2],
                        test_observations=test[0][:, :horizon, :], test_deltas=test[1], test_labels=test[2], architecture=architecture,
                        seed=seed(100, horizon, coordinate, int(l2 * 1e7)), expected_deltas=DELTAS, epochs=cfg["epochs"], minimum_epochs=cfg["minimum"], patience=cfg["patience"], batch_size=cfg["batch"], l2=l2, jit_compile=True,
                    )
                    candidates.append({"architecture": architecture, "l2": l2, "validation_log_loss": float(fit.validation_log_loss.numpy()), "validation_se": float(fit.validation_log_loss_standard_error.numpy())})
            candidates.sort(key=lambda row: (row["validation_log_loss"], row["architecture"], row["l2"]))
            best = candidates[0]
            selected[f"T{horizon}_j{coordinate}"] = {"architecture": best["architecture"], "l2": best["l2"], "candidates": candidates}
    return selected


def head_admission(fit: Any, cfg: dict[str, int], fixed_observation: tf.Tensor) -> dict[str, Any]:
    aucs = [float(fit.test_auc_by_delta[str(float(delta))].numpy()) for delta in DELTAS]
    eces = [float(fit.test_ece_by_delta[str(float(delta))].numpy()) for delta in DELTAS]
    support = []
    for delta in DELTAS:
        key = str(float(delta))
        observed_logit = float(fit.calibrated_logit(fixed_observation, tf.constant([delta], tf.float32))[0].numpy())
        lower = float(fit.test_logit_minimum_by_delta[key].numpy())
        upper = float(fit.test_logit_maximum_by_delta[key].numpy())
        expansion = 0.1 * max(upper - lower, 1.0)
        support.append(lower - expansion <= observed_logit <= upper + expansion)
    checks = {
        "finite": bool(fit.finite.numpy()),
        "pooled_signal": float(fit.test_log_loss.numpy()) < math.log(2.0) - 2.0 * float(fit.test_log_loss_standard_error.numpy()),
        "calibration_not_worse": float(fit.calibration_log_loss_after.numpy()) <= float(fit.calibration_log_loss_before.numpy()) + 1.0e-4,
        "temperature_positive": float(fit.calibration_temperature.numpy()) > 0.0,
        "per_delta_ece": max(eces) <= 0.04,
        "informative_deltas": sum(auc > 0.52 for auc in aucs) >= 2,
        "auc_not_inverted": all(aucs[index + 1] >= aucs[index] - 0.03 for index in range(len(aucs) - 1)),
        "max_delta_not_separated": aucs[-1] <= 0.995,
        "fixed_observation_support": all(support),
        "optimizer_complete": fit.epochs_run < cfg["epochs"] or fit.final_ten_epoch_improvement < 1.0e-4,
    }
    hard_names = ("finite", "temperature_positive", "optimizer_complete")
    hard_checks = {name: checks[name] for name in hard_names}
    score_diagnostics = {name: value for name, value in checks.items() if name not in hard_names}
    return {
        "checks": checks,
        "hard_checks": hard_checks,
        "score_interpretability_diagnostics": score_diagnostics,
        "hard_admitted": all(hard_checks.values()),
        "score_interpretability_all_passed": all(score_diagnostics.values()),
    }


def fit_bundle(kind: str, cfg: dict[str, int], selected: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    fits: dict[str, Any] = {}
    admissions: dict[str, Any] = {}
    fixed_full = (gaussian_simulator.fixed_observed_path(50) if kind == "gaussian" else sir_simulator.fixed_observed_path(81120, 50))[None, ...]
    for coordinate in COORDINATES:
        splits = conditional_splits(kind, coordinate=coordinate, domain=TRAIN_DOMAIN, replicate=0, cfg=cfg)
        for horizon in HORIZONS:
            train, validation, calibration, test = (splits[name] for name in ("train", "validation", "calibration", "test"))
            control = selected[f"T{horizon}_j{coordinate}"]
            key = f"T{horizon}_j{coordinate}"
            fits[key] = fit_anchored_classifier(
                train[0][:, :horizon, :], train[1], train[2],
                validation_observations=validation[0][:, :horizon, :], validation_deltas=validation[1], validation_labels=validation[2],
                calibration_observations=calibration[0][:, :horizon, :], calibration_deltas=calibration[1], calibration_labels=calibration[2],
                test_observations=test[0][:, :horizon, :], test_deltas=test[1], test_labels=test[2], architecture=control["architecture"],
                seed=seed(200, horizon, coordinate), expected_deltas=DELTAS, epochs=cfg["epochs"], minimum_epochs=cfg["minimum"], patience=cfg["patience"], batch_size=cfg["batch"], l2=float(control["l2"]), jit_compile=True,
            )
            admissions[key] = head_admission(fits[key], cfg, fixed_full[:, :horizon, :])
    return fits, admissions


def evaluate_bundle(fits: dict[str, Any], observations: tf.Tensor) -> tf.Tensor:
    return tf.stack([fits[f"T{horizon}_j{coordinate}"].score_at_observation(observations[:, :horizon, :]) for horizon in HORIZONS for coordinate in COORDINATES], axis=1)


def summarize_exact(kind: str, observations: tf.Tensor, outputs: tf.Tensor) -> dict[str, Any] | None:
    if kind != "gaussian":
        return None
    exact = tf.stack([tf.stack([gaussian_simulator.exact_score(THETA, observations[index, :horizon, :])[coordinate] for horizon in HORIZONS for coordinate in COORDINATES]) for index in range(int(observations.shape[0]))])
    return {"mean_exact_score": tf.reduce_mean(exact, axis=0), "mean_estimate_minus_exact": tf.reduce_mean(outputs - exact, axis=0), "rmse": tf.sqrt(tf.reduce_mean(tf.square(outputs - exact), axis=0))}


def base_manifest(kind: str, profile_name: str, cfg: dict[str, int], audit: dict[str, Any], loaded: list[str], started: float) -> dict[str, Any]:
    return {"schema": "bayesfilter.null_calibrated_predictive_consistency.manifest.v1", "status": "RUNNING", "kind": kind, "profile": profile_name, "git_commit": git_commit(), "selection_domain": SELECTION_DOMAIN, "train_domain": TRAIN_DOMAIN, "null_fit_domain": NULL_FIT_DOMAIN, "null_calibration_domain": NULL_CALIBRATION_DOMAIN, "null_audit_domain": NULL_AUDIT_DOMAIN, "config": cfg, "cpu_only_smoke": CPU_ONLY_SMOKE, "gpu_memory_policy": GPU_MEMORY_POLICY, "python": sys.executable, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"), "xla_flags": os.environ.get("XLA_FLAGS", "unset"), "source_hashes": {str(path.relative_to(ROOT)): sha(path) for path in (PLAN_PATH, REVIEW_PATH, REPAIR_PATH, IMPLEMENTATION_PATH, ESTIMATOR_PATH, RUNNER_PATH)}, "source_audit": audit, "runtime_module_audit": {"forbidden_loaded_modules": loaded, "passed": not loaded}, "started_at": datetime.now(timezone.utc).isoformat(), "wall_time_seconds": time.perf_counter() - started}


def run(output_root: Path, *, kind: str, profile_name: str) -> None:
    if CPU_ONLY_SMOKE and profile_name != "smoke":
        raise ValueError("BAYESFILTER_CPU_ONLY_SMOKE is allowed only for smoke runs")
    validate_domains()
    cfg = profile(profile_name)
    audit = source_audit()
    loaded = forbidden_loaded_modules()
    if not audit["passed"]:
        raise RuntimeError("source dependency veto")
    if loaded:
        raise RuntimeError(f"runtime dependency veto: {loaded}")
    initialize_simulators()
    output_root.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    manifest = base_manifest(kind, profile_name, cfg, audit, loaded, started)
    write(output_root / "run_manifest.json", manifest)
    selected = fit_selection(kind, cfg)
    write(output_root / "selected_controls.json", {"selected": selected, "selection_domain": SELECTION_DOMAIN})
    fits, head_admissions = fit_bundle(kind, cfg, selected)
    if profile_name == "full" and not all(row["hard_admitted"] for row in head_admissions.values()):
        write(output_root / "head_admissions.json", {"heads": head_admissions, "all_hard_admitted": False, "score_interpretability_all_passed": all(row["score_interpretability_all_passed"] for row in head_admissions.values())})
        result = {"schema": "bayesfilter.null_calibrated_predictive_consistency.result.v1", "status": "BLOCKED_HEAD_ADMISSION", "kind": kind, "profile": profile_name, "claim": "joint_same_parameter_predictive_coverage_only", "head_admissions": head_admissions, "null_coverage_executed": False, "reason": "frozen_classifier_head_admission_veto", "nonclaims": ["not pathwise exact score validation", "not evidence against null calibration", "not filter correctness", "not ranking"]}
        write(output_root / "result.json", result)
        manifest.update({"status": result["status"], "finished_at": datetime.now(timezone.utc).isoformat(), "wall_time_seconds": time.perf_counter() - started, "result_sha256": sha(output_root / "result.json")})
        write(output_root / "run_manifest.json", manifest)
        return
    write(output_root / "head_admissions.json", {"heads": head_admissions, "all_hard_admitted": all(row["hard_admitted"] for row in head_admissions.values()), "score_interpretability_all_passed": all(row["score_interpretability_all_passed"] for row in head_admissions.values())})
    fit_paths = null_paths(kind, count=cfg["null_fit"], domain=NULL_FIT_DOMAIN, role=1)
    calibration_paths = null_paths(kind, count=cfg["null_calibration"], domain=NULL_CALIBRATION_DOMAIN, role=2)
    audit_paths = null_paths(kind, count=cfg["null_audit"], domain=NULL_AUDIT_DOMAIN, role=3)
    fit_outputs = evaluate_bundle(fits, fit_paths)
    calibration_outputs = evaluate_bundle(fits, calibration_paths)
    audit_outputs = evaluate_bundle(fits, audit_paths)
    geometry = fit_svd_geometry(fit_outputs)
    calibration_scores = geometry.distance(calibration_outputs)
    threshold = conformal_threshold(calibration_scores, coverage=0.95)
    audit_scores = geometry.distance(audit_outputs)
    failures = int(tf.reduce_sum(tf.cast(audit_scores > threshold, tf.int32)).numpy())
    audit_result = audit_failure(failures, int(audit_outputs.shape[0]))
    fixed = (gaussian_simulator.fixed_observed_path(50) if kind == "gaussian" else sir_simulator.fixed_observed_path(81120, 50))[None, ...]
    fixed_output = evaluate_bundle(fits, fixed)
    fixed_distance = geometry.distance(fixed_output)[0]
    all_null_observations = tf.concat((fit_paths, calibration_paths, audit_paths), axis=0)
    all_null_outputs = tf.concat((fit_outputs, calibration_outputs, audit_outputs), axis=0)
    result = {
        "schema": "bayesfilter.null_calibrated_predictive_consistency.result.v1",
        "status": "PASSED" if not audit_result["falsified"] and profile_name == "full" else ("SMOKE_COMPLETED" if profile_name == "smoke" else "FAILED"),
        "kind": kind,
        "profile": profile_name,
        "claim": "joint_same_parameter_predictive_coverage_only",
        "coverage_target": 0.95,
        "head_admissions": head_admissions,
        "score_interpretability_all_passed": all(row["score_interpretability_all_passed"] for row in head_admissions.values()),
        "calibration_rank": int(geometry.rank),
        "calibration_threshold": threshold,
        "audit": audit_result,
        "fixed_observation_output": fixed_output[0],
        "fixed_observation_distance": fixed_distance,
        "fixed_observation_accepted": bool(fixed_distance.numpy() <= threshold.numpy()),
        "zero_mean_diagnostic": zero_mean_max_t_diagnostic(all_null_outputs, bootstrap_replicates=cfg["bootstrap"]),
        "gaussian_exact_diagnostic": summarize_exact(kind, all_null_observations, all_null_outputs),
        "nonclaims": ["not pathwise exact score validation", "not filter correctness", "not ranking", "not HMC/default readiness"],
    }
    write(output_root / "null_fit_outputs.json", {"outputs": fit_outputs, "geometry": {"center": geometry.center, "singular_values": geometry.singular_values, "rank": geometry.rank, "omitted_variance": geometry.omitted_variance, "threshold": geometry.threshold}})
    write(output_root / "null_calibration_outputs.json", {"outputs": calibration_outputs, "scores": calibration_scores, "threshold": threshold})
    write(output_root / "null_audit_outputs.json", {"outputs": audit_outputs, "scores": audit_scores, "failures": failures, "audit": audit_result})
    write(output_root / "result.json", result)
    manifest.update({"status": result["status"], "finished_at": datetime.now(timezone.utc).isoformat(), "wall_time_seconds": time.perf_counter() - started, "result_sha256": sha(output_root / "result.json")})
    write(output_root / "run_manifest.json", manifest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--kind", choices=("gaussian", "sir"), required=True)
    parser.add_argument("--profile", choices=("smoke", "full"), default="full")
    args = parser.parse_args()
    run(args.output_root, kind=args.kind, profile_name=args.profile)


if __name__ == "__main__":
    main()
