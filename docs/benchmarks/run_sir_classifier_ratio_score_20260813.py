"""Run observation-only classifier likelihood-ratio score references."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import platform
import statistics
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

from bayesfilter.independent_score.classifier_ratio_score_tf import (  # noqa: E402
    ARCHITECTURES,
    central_score_from_calibrated_logit,
    epsilon_squared_extrapolation,
    fit_ratio_classifier,
)
from bayesfilter.independent_score import (  # noqa: E402
    gaussian_observation_simulator_tf as gaussian_simulator,
)
from bayesfilter.independent_score import (  # noqa: E402
    sir_observation_simulator_tf as sir_simulator,
)


THETA = tf.zeros([3], tf.float64)
HORIZONS = (20, 40, 50)
EPSILONS = (0.01, 0.02, 0.04, 0.08)
COORDINATES = (0, 1, 2)
REGULARIZATION = (0.0, 1.0e-5)
FINAL_REPLICATES = 3
ROOT_SEED = 89300
DEFAULT_OUTPUT = ROOT / "docs/benchmarks/artifacts/sir_classifier_ratio_score_20260813"
FORBIDDEN_MODULE_SEGMENTS = {
    "filtering",
    "filters",
    "particle",
    "particles",
    "smoothing",
    "simulation_score_tf",
}


def _safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, tf.Tensor):
        return _safe(value.numpy().tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _seed(*parts: int) -> int:
    value = int(ROOT_SEED)
    for part in parts:
        value = (value * 1009 + int(part) + 7919) % 2147483000
    return value


def _profile(profile: str) -> dict[str, int]:
    if profile == "full":
        return {
            "train": 2048,
            "validation": 512,
            "calibration": 512,
            "test": 1024,
            "epochs": 160,
            "minimum_epochs": 20,
            "patience": 12,
        }
    if profile == "smoke":
        return {
            "train": 128,
            "validation": 64,
            "calibration": 64,
            "test": 128,
            "epochs": 8,
            "minimum_epochs": 4,
            "patience": 3,
        }
    raise ValueError("profile must be full or smoke")


def _forbidden_loaded_modules() -> list[str]:
    forbidden = []
    for name in sorted(sys.modules):
        if not name.startswith("bayesfilter."):
            continue
        segments = set(name.lower().split("."))
        if segments & FORBIDDEN_MODULE_SEGMENTS or "highdim" in segments:
            forbidden.append(name)
    return forbidden


def _source_dependency_audit() -> dict[str, Any]:
    paths = (
        ROOT / "bayesfilter/independent_score/classifier_ratio_score_tf.py",
        ROOT / "bayesfilter/independent_score/sir_observation_simulator_tf.py",
        ROOT / "bayesfilter/independent_score/gaussian_observation_simulator_tf.py",
        Path(__file__).resolve(),
    )
    imported = {}
    violations = []
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        modules = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                modules.append(node.module or "")
        imported[path.name] = modules
        for module in modules:
            segments = set(module.lower().split("."))
            if segments & FORBIDDEN_MODULE_SEGMENTS or "highdim" in segments:
                violations.append({"path": str(path), "module": module})
    return {"imported_modules": imported, "violations": violations, "passed": not violations}


def _labels(count_per_class: int) -> tf.Tensor:
    return tf.concat(
        [
            tf.zeros([count_per_class], tf.float32),
            tf.ones([count_per_class], tf.float32),
        ],
        axis=0,
    )


def _sir_noise(count: int, seed: int) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    initial = tf.random.stateless_normal(
        [count, 18], [_seed(seed, 1), 11], dtype=tf.float64
    )
    transition = tf.random.stateless_normal(
        [count, 50, 18], [_seed(seed, 2), 13], dtype=tf.float64
    )
    observation = tf.random.stateless_normal(
        [count, 50, 9], [_seed(seed, 3), 17], dtype=tf.float64
    )
    return initial, transition, observation


def _sir_dataset(
    *,
    coordinate: int,
    epsilon: float,
    role: int,
    replicate: int,
    count_per_class: int,
    data_domain: int,
) -> tuple[tf.Tensor, tf.Tensor]:
    simulator = sir_simulator.make_compiled_observation_simulator(50)
    minus_noise = _sir_noise(
        count_per_class,
        _seed(data_domain, coordinate, int(round(epsilon * 10000)), role, replicate, 0),
    )
    plus_noise = _sir_noise(
        count_per_class,
        _seed(data_domain, coordinate, int(round(epsilon * 10000)), role, replicate, 1),
    )
    direction = tf.one_hot(coordinate, 3, dtype=tf.float64)
    minus = simulator(THETA - tf.cast(epsilon, tf.float64) * direction, *minus_noise)
    plus = simulator(THETA + tf.cast(epsilon, tf.float64) * direction, *plus_noise)
    return tf.concat([minus, plus], axis=0), _labels(count_per_class)


def _gaussian_dataset(
    *,
    coordinate: int,
    epsilon: float,
    role: int,
    replicate: int,
    count_per_class: int,
    data_domain: int,
) -> tuple[tf.Tensor, tf.Tensor]:
    simulator = gaussian_simulator.make_compiled_observation_simulator(50)
    minus_noise = tf.random.stateless_normal(
        [count_per_class, 50, 9],
        [
            _seed(data_domain, coordinate, int(round(epsilon * 10000)), role, replicate, 0),
            31,
        ],
        dtype=tf.float64,
    )
    plus_noise = tf.random.stateless_normal(
        [count_per_class, 50, 9],
        [
            _seed(data_domain, coordinate, int(round(epsilon * 10000)), role, replicate, 1),
            37,
        ],
        dtype=tf.float64,
    )
    direction = tf.one_hot(coordinate, 3, dtype=tf.float64)
    minus = simulator(THETA - tf.cast(epsilon, tf.float64) * direction, minus_noise)
    plus = simulator(THETA + tf.cast(epsilon, tf.float64) * direction, plus_noise)
    return tf.concat([minus, plus], axis=0), _labels(count_per_class)


def _datasets(
    dataset_fn: Callable[..., tuple[tf.Tensor, tf.Tensor]],
    *,
    coordinate: int,
    epsilon: float,
    replicate: int,
    data_domain: int,
    config: dict[str, int],
) -> dict[str, tuple[tf.Tensor, tf.Tensor]]:
    return {
        role_name: dataset_fn(
            coordinate=coordinate,
            epsilon=epsilon,
            role=role_number,
            replicate=replicate,
            count_per_class=config[role_name],
            data_domain=data_domain,
        )
        for role_name, role_number in (
            ("train", 1),
            ("validation", 2),
            ("calibration", 3),
            ("test", 4),
        )
    }


def _fit_one(
    datasets: dict[str, tuple[tf.Tensor, tf.Tensor]],
    *,
    stage: str,
    horizon: int,
    coordinate: int,
    epsilon: float,
    replicate: int,
    architecture: str,
    l2: float,
    data_domain: int,
    config: dict[str, int],
) -> dict[str, Any]:
    train_x, train_y = datasets["train"]
    validation_x, validation_y = datasets["validation"]
    calibration_x, calibration_y = datasets["calibration"]
    test_x, test_y = datasets["test"]
    fit = fit_ratio_classifier(
        train_x[:, :horizon, :],
        train_y,
        validation_observations=validation_x[:, :horizon, :],
        validation_labels=validation_y,
        calibration_observations=calibration_x[:, :horizon, :],
        calibration_labels=calibration_y,
        test_observations=test_x[:, :horizon, :],
        test_labels=test_y,
        architecture=architecture,
        seed=_seed(
            300 if stage == "exact_oracle" else 400,
            horizon,
            coordinate,
            int(round(epsilon * 10000)),
            replicate,
            99,
        ),
        epochs=config["epochs"],
        minimum_epochs=config["minimum_epochs"],
        patience=config["patience"],
        batch_size=config["train"] * 2,
        l2=l2,
        jit_compile=True,
    )
    if stage == "exact_oracle":
        observed = gaussian_simulator.fixed_observed_path(horizon)[None, ...]
    else:
        observed = sir_simulator.fixed_observed_path(81120, horizon)[None, ...]
    observed_logit = fit.calibrated_logit(observed)[0]
    score = central_score_from_calibrated_logit(observed_logit, epsilon)
    heldout_min = float(fit.test_logit_minimum.numpy())
    heldout_max = float(fit.test_logit_maximum.numpy())
    expanded = 0.1 * max(heldout_max - heldout_min, 1.0)
    support_pass = (
        heldout_min - expanded
        <= float(observed_logit.numpy())
        <= heldout_max + expanded
    )
    test_loss = float(fit.test_log_loss.numpy())
    test_loss_se = float(fit.test_log_loss_standard_error.numpy())
    auc = float(fit.test_auc.numpy())
    ece = float(fit.expected_calibration_error.numpy())
    calibration_before = float(fit.calibration_log_loss_before.numpy())
    calibration_after = float(fit.calibration_log_loss_after.numpy())
    admission = {
        "finite": bool(fit.finite.numpy()),
        "signal": test_loss < math.log(2.0) - 2.0 * test_loss_se,
        "auc_range": 0.505 <= auc <= 0.995,
        "ece": ece <= 0.03,
        "platt_slope": 0.5 <= float(fit.calibration_slope.numpy()) <= 2.0,
        "calibration_not_worse": calibration_after <= calibration_before + 1.0e-4,
        "observed_path_support": support_pass,
    }
    return {
        "stage": stage,
        "horizon": horizon,
        "coordinate": coordinate,
        "epsilon": epsilon,
        "replicate": replicate,
        "architecture": architecture,
        "l2": l2,
        "data_domain": data_domain,
        "best_epoch": fit.best_epoch,
        "train_log_loss": float(fit.train_log_loss.numpy()),
        "validation_log_loss": float(fit.validation_log_loss.numpy()),
        "validation_log_loss_standard_error": float(
            fit.validation_log_loss_standard_error.numpy()
        ),
        "calibration_log_loss_before": calibration_before,
        "calibration_log_loss_after": calibration_after,
        "test_log_loss": test_loss,
        "test_log_loss_standard_error": test_loss_se,
        "test_auc": auc,
        "expected_calibration_error": ece,
        "platt_slope": float(fit.calibration_slope.numpy()),
        "platt_intercept": float(fit.calibration_intercept.numpy()),
        "observed_calibrated_logit": float(observed_logit.numpy()),
        "observed_logit_support": [heldout_min, heldout_max],
        "score_estimate": float(score.numpy()),
        "admission": admission,
        "admitted": all(admission.values()),
    }


def _select_controls(
    rows: list[dict[str, Any]], horizon: int, coordinate: int
) -> dict[str, Any]:
    candidates = []
    for architecture in ARCHITECTURES:
        for l2 in REGULARIZATION:
            selected = [
                row
                for row in rows
                if row["horizon"] == horizon
                and row["coordinate"] == coordinate
                and row["architecture"] == architecture
                and row["l2"] == l2
            ]
            if len(selected) != 1:
                raise ValueError(
                    f"incomplete selection rows for {horizon}/{coordinate}/{architecture}/{l2}"
                )
            losses = [row["validation_log_loss"] for row in selected]
            mean = statistics.mean(losses)
            candidates.append(
                {
                    "architecture": architecture,
                    "l2": l2,
                    "validation_log_loss": mean,
                    "validation_log_loss_standard_error": selected[0][
                        "validation_log_loss_standard_error"
                    ],
                }
            )
    best = min(candidates, key=lambda row: row["validation_log_loss"])
    eligible_simpler = [
        row
        for row in candidates
        if row["architecture"] in ("linear_full_path", "linear_full_path_quadratic")
        and row["validation_log_loss"]
        <= best["validation_log_loss"]
        + best["validation_log_loss_standard_error"]
    ]
    if eligible_simpler:
        selected = min(
            eligible_simpler,
            key=lambda row: (
                0 if row["architecture"] == "linear_full_path" else 1,
                row["validation_log_loss"],
                row["l2"],
            ),
        )
    else:
        selected = best
    return {"horizon": horizon, "coordinate": coordinate, **selected, "candidates": candidates}


def _summarize(
    final_rows: list[dict[str, Any]], *, stage: str
) -> tuple[dict[str, Any], bool]:
    summary: dict[str, Any] = {"by_horizon_coordinate": {}}
    all_reference_gates = True
    for horizon in HORIZONS:
        for coordinate in COORDINATES:
            admitted_by_epsilon = {}
            for epsilon in EPSILONS:
                rows = [
                    row
                    for row in final_rows
                    if row["horizon"] == horizon
                    and row["coordinate"] == coordinate
                    and abs(row["epsilon"] - epsilon) < 1.0e-12
                    and row["admitted"]
                ]
                admitted_by_epsilon[epsilon] = [row["score_estimate"] for row in rows]
            extrapolation = epsilon_squared_extrapolation(
                admitted_by_epsilon, required_replicates=FINAL_REPLICATES
            )
            if stage == "exact_oracle":
                exact_score = float(
                    gaussian_simulator.exact_score(
                        THETA, gaussian_simulator.fixed_observed_path(horizon)
                    )[coordinate].numpy()
                )
                if extrapolation["reference_admitted"]:
                    tolerance = max(
                        0.5, 3.0 * float(extrapolation["intercept_standard_error"])
                    )
                    error = abs(float(extrapolation["intercept"]) - exact_score)
                    oracle_gate = error <= tolerance
                else:
                    tolerance = None
                    error = None
                    oracle_gate = False
                extrapolation["exact_score"] = exact_score
                extrapolation["absolute_error"] = error
                extrapolation["oracle_tolerance"] = tolerance
                extrapolation["exact_oracle_gate"] = oracle_gate
                extrapolation["reference_admitted"] = bool(
                    extrapolation["reference_admitted"] and oracle_gate
                )
                if not extrapolation["reference_admitted"]:
                    extrapolation["status"] = "exact_oracle_failed"
            all_reference_gates = all_reference_gates and bool(
                extrapolation["reference_admitted"]
            )
            summary["by_horizon_coordinate"][f"T{horizon}_j{coordinate}"] = extrapolation
    return summary, all_reference_gates


def _require_oracle(path: Path | None) -> dict[str, Any]:
    if path is None:
        raise ValueError("SIR execution requires --oracle-result")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("stage") != "exact_oracle" or payload.get("status") != "PASSED":
        raise ValueError("SIR execution requires a passed exact-oracle result")
    return {"path": str(path.resolve()), "sha256": _sha256(path)}


def run_campaign(
    output_root: Path,
    *,
    stage: str,
    profile: str,
    oracle_result: Path | None,
) -> None:
    if stage not in {"exact_oracle", "sir"}:
        raise ValueError("stage must be exact_oracle or sir")
    config = _profile(profile)
    if profile == "smoke" and stage == "sir":
        raise ValueError("SIR stage has no claim-bearing smoke mode")
    oracle_provenance = _require_oracle(oracle_result) if stage == "sir" else None
    source_audit = _source_dependency_audit()
    if not source_audit["passed"]:
        raise RuntimeError(f"source dependency veto: {source_audit['violations']}")
    initial_runtime_violations = _forbidden_loaded_modules()
    if initial_runtime_violations:
        raise RuntimeError(f"runtime dependency veto: {initial_runtime_violations}")

    started = time.perf_counter()
    output_root.mkdir(parents=True, exist_ok=False)
    dataset_fn = _gaussian_dataset if stage == "exact_oracle" else _sir_dataset
    manifest = {
        "schema": "bayesfilter.classifier_ratio_score.manifest.v2",
        "status": "RUNNING",
        "stage": stage,
        "profile": profile,
        "method": "balanced_observation_only_classifier_logit_ratio",
        "score_identity": "calibrated_logit_at_fixed_observation/(2*epsilon)",
        "git_revision": _git_revision(),
        "python": sys.executable,
        "python_version": platform.python_version(),
        "theta": THETA,
        "horizons": HORIZONS,
        "epsilons": EPSILONS,
        "coordinates": COORDINATES,
        "final_replicates": FINAL_REPLICATES,
        "data_counts_per_class": {
            key: config[key] for key in ("train", "validation", "calibration", "test")
        },
        "training": {
            "epochs": config["epochs"],
            "minimum_epochs": config["minimum_epochs"],
            "patience": config["patience"],
            "batch_size": config["train"] * 2,
            "batch_native": True,
            "jit_compile": True,
            "tf32": False,
            "samplewise_loop_or_scalar_fallback": False,
        },
        "selection_data_domain": 10,
        "final_data_domain": 20,
        "selection_scope": "stage_horizon_coordinate",
        "paired_prefix_generation_horizon": 50,
        "oracle_provenance": oracle_provenance,
        "gpu_memory_policy": GPU_MEMORY_POLICY,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
        "xla_flags": os.environ.get("XLA_FLAGS", "unset"),
        "source_dependency_audit": source_audit,
        "initial_loaded_bayesfilter_modules": sorted(
            name for name in sys.modules if name.startswith("bayesfilter")
        ),
        "source_hashes": {
            path.name: _sha256(path)
            for path in (
                ROOT / "bayesfilter/independent_score/classifier_ratio_score_tf.py",
                ROOT / "bayesfilter/independent_score/sir_observation_simulator_tf.py",
                ROOT / "bayesfilter/independent_score/gaussian_observation_simulator_tf.py",
                ROOT / "docs/plans/bayesfilter-sir-classifier-ratio-score-v2-plan-2026-08-13.md",
                ROOT / "docs/plans/bayesfilter-sir-classifier-ratio-score-v2-plan-review-2026-08-13.md",
                Path(__file__).resolve(),
            )
        },
    }
    _write_json(output_root / "run_manifest.json", manifest)

    selection_rows = []
    for coordinate in COORDINATES:
        datasets = _datasets(
            dataset_fn,
            coordinate=coordinate,
            epsilon=0.04,
            replicate=0,
            data_domain=10,
            config=config,
        )
        for horizon in HORIZONS:
            for architecture in ARCHITECTURES:
                for l2 in REGULARIZATION:
                    selection_rows.append(
                        _fit_one(
                            datasets,
                            stage=stage,
                            horizon=horizon,
                            coordinate=coordinate,
                            epsilon=0.04,
                            replicate=0,
                            architecture=architecture,
                            l2=l2,
                            data_domain=10,
                            config=config,
                        )
                    )
    selected = {
        f"T{horizon}_j{coordinate}": _select_controls(
            selection_rows, horizon, coordinate
        )
        for horizon in HORIZONS
        for coordinate in COORDINATES
    }
    _write_json(
        output_root / "selected_architectures.json",
        {"selected": selected, "selection_rows": selection_rows},
    )

    final_rows = []
    for coordinate in COORDINATES:
        for epsilon in EPSILONS:
            for replicate in range(FINAL_REPLICATES):
                datasets = _datasets(
                    dataset_fn,
                    coordinate=coordinate,
                    epsilon=epsilon,
                    replicate=replicate,
                    data_domain=20,
                    config=config,
                )
                for horizon in HORIZONS:
                    controls = selected[f"T{horizon}_j{coordinate}"]
                    row = _fit_one(
                        datasets,
                        stage=stage,
                        horizon=horizon,
                        coordinate=coordinate,
                        epsilon=epsilon,
                        replicate=replicate,
                        architecture=str(controls["architecture"]),
                        l2=float(controls["l2"]),
                        data_domain=20,
                        config=config,
                    )
                    final_rows.append(row)
                    _write_json(output_root / f"row_{len(final_rows)-1:04d}.json", row)

    summary, all_reference_gates = _summarize(final_rows, stage=stage)
    runtime_violations = _forbidden_loaded_modules()
    if runtime_violations:
        all_reference_gates = False
    status = "PASSED" if all_reference_gates and profile == "full" else "FAILED"
    if profile == "smoke":
        status = "SMOKE_COMPLETED"
    result = {
        "schema": "bayesfilter.classifier_ratio_score.result.v2",
        "stage": stage,
        "profile": profile,
        "status": status,
        "all_horizon_coordinate_references_admitted": all_reference_gates,
        "selected_architectures": selected,
        "summary": summary,
        "rows": len(final_rows),
        "runtime_dependency_audit": {
            "loaded_bayesfilter_modules": sorted(
                name for name in sys.modules if name.startswith("bayesfilter")
            ),
            "violations": runtime_violations,
            "passed": not runtime_violations,
        },
        "nonclaims": [
            "not an exact SIR score",
            "not correctness or ranking evidence for any state-estimation algorithm",
            "not HMC or default-readiness evidence",
        ],
    }
    _write_json(output_root / "result.json", result)
    manifest["status"] = status
    manifest["wall_time_seconds"] = time.perf_counter() - started
    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    manifest["result_sha256"] = _sha256(output_root / "result.json")
    _write_json(output_root / "run_manifest.json", manifest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--stage", choices=("exact_oracle", "sir"), required=True)
    parser.add_argument("--profile", choices=("smoke", "full"), default="full")
    parser.add_argument("--oracle-result", type=Path)
    args = parser.parse_args()
    run_campaign(
        args.output_root,
        stage=args.stage,
        profile=args.profile,
        oracle_result=args.oracle_result,
    )


if __name__ == "__main__":
    main()
