"""Estimate Austria-SIR observed-data scores from independent simulations."""

from __future__ import annotations

import argparse
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
from typing import Any

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
import tensorflow as tf

from bayesfilter.runtime.gpu_memory_policy import (  # noqa: E402
    configure_tensorflow_gpu_memory_growth,
)


GPU_MEMORY_POLICY = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bayesfilter.highdim.simulation_score_tf import (  # noqa: E402
    fisher_identity_simulation_score,
)
from bayesfilter.highdim.sir_latent_preclip_tf import (  # noqa: E402
    latent_preclip_zhao_cui_sir_austria_model,
)
from bayesfilter.highdim.sir_online_score_teacher_tf import (  # noqa: E402
    _transition_mean_and_parameter_tangent,
    initial_log_density_and_score,
    observation_log_density_and_score,
    static_spec_from_model,
    transition_log_density_and_score,
)
DTYPE = tf.float64
THETA = tf.zeros([3], DTYPE)
HORIZONS = (20, 40, 50)
OBSERVATION_SEED = 81120
BASE_SIMULATION_SEED = 86100
DEFAULT_OUTPUT = ROOT / "docs/benchmarks/artifacts/sir_simulation_score_20260813"


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
    path.write_text(
        json.dumps(_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _gpu_manifest(memory_policy: dict[str, Any]) -> dict[str, Any]:
    logical = tf.config.list_logical_devices("GPU")
    if not logical:
        raise RuntimeError("simulation score campaign requires a visible GPU")
    return {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
        "physical_devices": [str(d.name) for d in tf.config.list_physical_devices("GPU")],
        "logical_devices": [str(d.name) for d in logical],
        "device": "/device:GPU:0",
        "dtype": "float64",
        "jit_compile": True,
        "tf32_execution_enabled": False,
        "memory_policy": memory_policy,
        "trust_basis": "owner_designated_managed_session_visible_gpu_trusted",
    }


def _simulate_and_score_batch(
    spec: Any,
    theta: tf.Tensor,
    observed: tf.Tensor,
    initial_noise: tf.Tensor,
    transition_noise: tf.Tensor,
    observation_noise: tf.Tensor,
    horizon: int,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Simulate a batch and evaluate observation likelihood and complete score."""

    initial_chol = tf.linalg.cholesky(spec.initial_covariance)
    process_chol = tf.linalg.cholesky(spec.process_covariance)
    latent = spec.initial_mean + tf.linalg.matmul(
        initial_noise, initial_chol, transpose_b=True
    )
    log_observation = tf.zeros([tf.shape(initial_noise)[0]], DTYPE)
    _, complete_score = initial_log_density_and_score(theta, latent, spec=spec)
    previous = latent
    for time_index in range(1, horizon + 1):
        mean, _ = _transition_mean_and_parameter_tangent(
            previous, theta, time_index, spec
        )
        current = mean + tf.linalg.matmul(
            transition_noise[:, time_index - 1, :], process_chol, transpose_b=True
        )
        observation_log_weight, observation_score = observation_log_density_and_score(
            theta,
            current,
            observed[time_index - 1],
            time_index=time_index,
            spec=spec,
        )
        _, transition_score = transition_log_density_and_score(
            theta, previous, current, time_index=time_index, spec=spec
        )
        log_observation += observation_log_weight
        complete_score += transition_score + observation_score
        previous = current
    return log_observation, complete_score


def _compiled_batch_runner(spec: Any, theta: tf.Tensor, observed: tf.Tensor, horizon: int):
    @tf.function(jit_compile=True)
    def run(initial_noise: tf.Tensor, transition_noise: tf.Tensor, observation_noise: tf.Tensor):
        return _simulate_and_score_batch(
            spec, theta, observed, initial_noise, transition_noise, observation_noise, horizon
        )

    return run


def _replicate(
    model: Any,
    theta: tf.Tensor,
    observed: tf.Tensor,
    *,
    horizon: int,
    seed: int,
    paths: int,
    batch_size: int,
) -> dict[str, Any]:
    if paths <= 0 or batch_size <= 0:
        raise ValueError("paths and batch_size must be positive")
    spec = static_spec_from_model(model)
    runner = _compiled_batch_runner(spec, theta, observed, horizon)
    log_likelihoods: list[tf.Tensor] = []
    complete_scores: list[tf.Tensor] = []
    started = time.perf_counter()
    for offset in range(0, paths, batch_size):
        count = min(batch_size, paths - offset)
        batch_seed = int(seed + offset // batch_size)
        initial_noise = tf.random.stateless_normal(
            [count, model.state_dim()], [batch_seed, 101], dtype=DTYPE
        )
        transition_noise = tf.random.stateless_normal(
            [count, horizon, model.state_dim()], [batch_seed, 102], dtype=DTYPE
        )
        observation_noise = tf.random.stateless_normal(
            [count, horizon, model.observation_dim()], [batch_seed, 103], dtype=DTYPE
        )
        log_likelihood, complete_score = runner(initial_noise, transition_noise, observation_noise)
        log_likelihoods.append(log_likelihood)
        complete_scores.append(complete_score)
    estimate = fisher_identity_simulation_score(
        tf.concat(log_likelihoods, axis=0), tf.concat(complete_scores, axis=0)
    )
    payload = {
        "horizon": horizon,
        "seed": seed,
        "paths": paths,
        "batch_size": batch_size,
        "wall_time_seconds": time.perf_counter() - started,
        "log_marginal": estimate.log_marginal,
        "score": estimate.score,
        "effective_sample_size": estimate.effective_sample_size,
        "effective_sample_fraction": estimate.effective_sample_fraction,
        "maximum_normalized_weight": estimate.maximum_normalized_weight,
        "log_weight_range": estimate.log_weight_range,
        "finite": estimate.finite,
        "collapsed": estimate.collapsed,
    }
    return _safe(payload)


def run_campaign(
    *, output_root: Path, paths: int, replicates: int, batch_size: int,
    horizons: tuple[int, ...] = HORIZONS, replicate_start: int = 0,
) -> None:
    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    tf.config.experimental.enable_tensor_float_32_execution(False)
    model = latent_preclip_zhao_cui_sir_austria_model()
    _, all_observations = model.physical_model.base_model.simulate(
        final_time=max(HORIZONS), seed=OBSERVATION_SEED
    )
    output_root.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema": "bayesfilter.sir_simulation_score.manifest.v1",
        "status": "RUNNING",
        "started_at": started_at.isoformat(),
        "git_revision": _git_revision(),
        "python": sys.executable,
        "python_version": platform.python_version(),
        "model_target": model.manifest_payload(),
        "theta": THETA,
        "observation_seed": OBSERVATION_SEED,
        "horizons": list(horizons),
        "replicates": replicates,
        "replicate_start": replicate_start,
        "paths_per_replicate": paths,
        "batch_size": batch_size,
        "simulation_seed_formula": "86100 + 100*horizon + replicate_index",
        "gpu": _gpu_manifest(dict(GPU_MEMORY_POLICY)),
        "source_hashes": {
            "simulation_score_tf.py": _sha256(ROOT / "bayesfilter/highdim/simulation_score_tf.py"),
            "sir_latent_preclip_tf.py": _sha256(ROOT / "bayesfilter/highdim/sir_latent_preclip_tf.py"),
            "models.py": _sha256(ROOT / "bayesfilter/highdim/models.py"),
            "plan.md": _sha256(ROOT / "docs/plans/bayesfilter-sir-simulation-score-plan-2026-08-13.md"),
        },
    }
    _write_json(output_root / "run_manifest.json", manifest)
    rows: list[dict[str, Any]] = []
    try:
        for horizon in horizons:
            observed = tf.cast(all_observations[1 : horizon + 1], DTYPE)
            for replicate in range(replicate_start, replicate_start + replicates):
                row = _replicate(
                    model,
                    THETA,
                    observed,
                    horizon=horizon,
                    seed=BASE_SIMULATION_SEED + 100 * horizon + replicate,
                    paths=paths,
                    batch_size=batch_size,
                )
                row["replicate"] = replicate
                rows.append(row)
                _write_json(output_root / f"row_{horizon}_{replicate:02d}.json", row)
        summary: dict[str, Any] = {"schema": "bayesfilter.sir_simulation_score.summary.v1", "by_horizon": {}}
        for horizon in horizons:
            selected = [row for row in rows if row["horizon"] == horizon]
            summary["by_horizon"][str(horizon)] = {
                "replicate_count": len(selected),
                "score_mean": [
                    sum(row["score"][j] for row in selected) / len(selected)
                    for j in range(3)
                ],
                "score_replicates": [row["score"] for row in selected],
                "log_marginal_replicates": [row["log_marginal"] for row in selected],
                "ess_fraction_replicates": [row["effective_sample_fraction"] for row in selected],
                "maximum_weight_replicates": [row["maximum_normalized_weight"] for row in selected],
                "all_finite": all(row["finite"] for row in selected),
                "any_collapsed": any(row["collapsed"] for row in selected),
            }
        result = {
            "schema": "bayesfilter.sir_simulation_score.result.v1",
            "status": "PASS" if all(row["finite"] and not row["collapsed"] for row in rows) else "DIAGNOSTIC_VETO",
            "rows": len(rows),
            "summary": summary,
            "wall_time_seconds": time.perf_counter() - started,
            "nonclaims": list(manifest["model_target"]["what_is_not_claimed"]) + [
                "exact observed_data_score_oracle",
                "particle_algorithm_correctness_or_superiority",
            ],
        }
        _write_json(output_root / "result.json", result)
        manifest["status"] = result["status"]
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        manifest["wall_time_seconds"] = time.perf_counter() - started
        _write_json(output_root / "run_manifest.json", manifest)
    except Exception as exc:
        _write_json(output_root / "failure.json", {"status": "CAMPAIGN_FAILED", "type": type(exc).__name__, "message": str(exc)})
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT / "attempt01")
    parser.add_argument("--paths", type=int, default=8192)
    parser.add_argument("--replicates", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--horizons", type=int, nargs="+", choices=HORIZONS, default=list(HORIZONS)
    )
    parser.add_argument("--replicate-start", type=int, default=0)
    args = parser.parse_args()
    run_campaign(
        output_root=args.output_root,
        paths=args.paths,
        replicates=args.replicates,
        batch_size=args.batch_size,
        horizons=tuple(args.horizons),
        replicate_start=args.replicate_start,
    )


if __name__ == "__main__":
    main()
