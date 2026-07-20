#!/usr/bin/env python3
"""Bounded production-block timing canary for q-general SSL-LSTM forecasts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import tensorflow as tf


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bayesfilter.inference.cpu_forecast_pool import (  # noqa: E402
    CPUForecastPool,
    CPUForecastPoolConfig,
)
from bayesfilter.nonlinear.ssl_lstm_complexity_predictive_tf import (  # noqa: E402
    CALIBRATION_DRAWS_PER_CHAIN,
    FORECAST_HORIZON,
    FORECAST_REPLICATION_COUNT,
    ROOT_SEED,
    calibration_seed_roots,
)
from bayesfilter.nonlinear.ssl_lstm_complexity_target_tf import (  # noqa: E402
    PRIOR_CENTER,
    complexity_posterior_target,
)


SCHEMA = "bayesfilter.ssl_lstm.complexity_forecast_pool_timing.v1"
SCRIPT = Path(__file__).resolve().relative_to(ROOT)
PLAN = Path(
    "docs/plans/"
    "bayesfilter-ssl-lstm-neutra-hmc-state-complexity-ladder-plan-2026-07-19.md"
)
Q_VALUES = (1, 2, 5, 10, 20)
CHARTS = ("chart-a", "chart-b")
WORKERS_BY_Q = {1: 32, 2: 32, 5: 32, 10: 32, 20: 16}
BLOCK_DRAWS = 256
WARM_REPEATS = 2
MATERIAL_PREDICTIVE_DRAWS_PER_CHAIN = 12288
MATERIAL_TOTAL_BLOCKS = 388
MATERIAL_FRESH_POOL_STARTS = 3
MATERIAL_WARM_BLOCKS = MATERIAL_TOTAL_BLOCKS - MATERIAL_FRESH_POOL_STARTS
HOST_RAM_CAP_BYTES = 64 * 1024**3
BUDGET_MARGIN = 1.50
CANARY_ROOT_SEED = ROOT_SEED + 1001


class ForecastTimingError(RuntimeError):
    pass


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "numpy"):
        return json_safe(value.numpy())
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    return value


def canonical(payload: Any) -> bytes:
    return (
        json.dumps(
            json_safe(payload),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def payload_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical(payload)).hexdigest()


def repo_path(path: Path, *, label: str) -> Path:
    resolved = (ROOT / path).resolve()
    if not resolved.is_relative_to(ROOT):
        raise ForecastTimingError(f"{label} must remain inside the repository")
    return resolved


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ForecastTimingError(f"output already exists: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical(payload))
    temporary.replace(path)


def execution_source_signature() -> str:
    paths = (
        SCRIPT,
        Path("bayesfilter/inference/cpu_forecast_pool.py"),
        Path("bayesfilter/nonlinear/ssl_lstm_complexity_predictive_tf.py"),
        Path("bayesfilter/nonlinear/ssl_lstm_complexity_target_tf.py"),
    )
    return payload_sha256({path.as_posix(): sha256(ROOT / path) for path in paths})


def folded_seeds(root: tuple[int, int], count: int) -> np.ndarray:
    root_tensor = tf.constant(root, tf.int32)
    return np.asarray(
        [
            tf.random.experimental.stateless_fold_in(
                root_tensor, tf.constant(index, tf.int32), alg="philox"
            ).numpy()
            for index in range(int(count))
        ],
        dtype=np.int32,
    )


def material_seed_set(q: int) -> set[tuple[int, int]]:
    values = set()
    for root in calibration_seed_roots(q):
        values.update(
            tuple(row) for row in folded_seeds(root, CALIBRATION_DRAWS_PER_CHAIN).tolist()
        )
    maximum_extension_segments = (
        MATERIAL_PREDICTIVE_DRAWS_PER_CHAIN - 512
    ) // BLOCK_DRAWS
    for chart_index, _chart in enumerate(CHARTS):
        chart_offset = chart_index * 1000
        values.update(
            (ROOT_SEED, 50000 + 100 * q + chart_offset + index)
            for index in range(maximum_extension_segments)
        )
        forecast_offset = chart_index * 1_000_000
        values.update(
            (
                ROOT_SEED,
                60000 + 100 * q + forecast_offset + index,
            )
            for index in range(4 * MATERIAL_PREDICTIVE_DRAWS_PER_CHAIN)
        )
    return values


def canary_seeds(q: int) -> np.ndarray:
    seeds = np.stack(
        (
            np.full(BLOCK_DRAWS, CANARY_ROOT_SEED + q, dtype=np.int32),
            np.arange(BLOCK_DRAWS, dtype=np.int32) + 80000 + 1000 * q,
        ),
        axis=1,
    )
    if not {tuple(row) for row in seeds.tolist()}.isdisjoint(material_seed_set(q)):
        raise ForecastTimingError("timing-canary seeds overlap a material seed domain")
    return seeds


def projection_seconds(first_seconds: float, warm_seconds: float) -> float:
    return BUDGET_MARGIN * (
        MATERIAL_FRESH_POOL_STARTS * float(first_seconds)
        + MATERIAL_WARM_BLOCKS * float(warm_seconds)
    )


def contract_payload(q: int) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "mode": "contract-smoke",
        "status": "PASSED",
        "q": q,
        "block_draws": BLOCK_DRAWS,
        "warm_repeats": WARM_REPEATS,
        "worker_count": WORKERS_BY_Q[q],
        "forecast_replication_count": FORECAST_REPLICATION_COUNT,
        "forecast_horizon": FORECAST_HORIZON,
        "material_total_blocks": MATERIAL_TOTAL_BLOCKS,
        "material_fresh_pool_starts": MATERIAL_FRESH_POOL_STARTS,
        "material_warm_blocks": MATERIAL_WARM_BLOCKS,
        "budget_margin": BUDGET_MARGIN,
        "host_ram_cap_bytes": HOST_RAM_CAP_BYTES,
        "material_execution_authorized": False,
        "nonclaims": [
            "contract/import smoke only",
            "no forecast evaluation, calibration, HMC, or predictive decision",
        ],
    }


def run_canary(args: argparse.Namespace) -> dict[str, Any]:
    output = repo_path(args.output, label="output")
    source_signature = execution_source_signature()
    target = complexity_posterior_target(args.q, jit_compile=True)
    truth = np.repeat(
        np.asarray(PRIOR_CENTER.numpy(), dtype=np.float64)[None, :],
        BLOCK_DRAWS,
        axis=0,
    )
    seeds = canary_seeds(args.q)
    config = CPUForecastPoolConfig(
        worker_factory_path=(
            "bayesfilter.nonlinear.ssl_lstm_complexity_predictive_tf:"
            "complexity_forecast_worker_factory"
        ),
        worker_config={"q": args.q},
        worker_count=WORKERS_BY_Q[args.q],
        cores_per_worker=1,
        timeout_seconds=float(args.timeout_seconds),
    )
    started = time.perf_counter()
    outputs = []
    rows = []
    with CPUForecastPool(config) as pool:
        for index in range(1 + WARM_REPEATS):
            if execution_source_signature() != source_signature:
                raise ForecastTimingError("source drift during forecast timing canary")
            call_started = time.perf_counter()
            means, variances, observations, metadata = pool.evaluate(
                truth,
                seeds,
                request_id=f"q{args.q}-forecast-timing-{index}",
            )
            wall = time.perf_counter() - call_started
            aggregate = int(metadata["aggregate_parent_worker_ru_maxrss_bytes"])
            if aggregate > HOST_RAM_CAP_BYTES:
                raise ForecastTimingError("aggregate parent-worker RSS exceeded 64 GiB")
            if metadata["configured_worker_count"] != WORKERS_BY_Q[args.q]:
                raise ForecastTimingError("configured worker count drift")
            if len(metadata["startup_worker_pids"]) != WORKERS_BY_Q[args.q]:
                raise ForecastTimingError("startup worker PID count mismatch")
            if metadata["active_worker_count"] != WORKERS_BY_Q[args.q]:
                raise ForecastTimingError("production block did not use every worker")
            if any(
                row["cuda_visible_devices"] != "-1"
                or row["tensorflow_gpu_devices"] != []
                for row in metadata["worker_metadata"]
            ):
                raise ForecastTimingError("forecast timing worker GPU visibility veto")
            outputs.append((means, variances, observations))
            rows.append(
                {
                    "call_index": index,
                    "role": "first" if index == 0 else "warm_replay",
                    "wall_seconds": wall,
                    "output_sha256": {
                        "means": hashlib.sha256(np.ascontiguousarray(means).tobytes()).hexdigest(),
                        "variances": hashlib.sha256(
                            np.ascontiguousarray(variances).tobytes()
                        ).hexdigest(),
                        "observations": hashlib.sha256(
                            np.ascontiguousarray(observations).tobytes()
                        ).hexdigest(),
                    },
                    "worker_metadata": json_safe(metadata),
                }
            )
    for replay in outputs[1:]:
        for expected, actual in zip(outputs[0], replay, strict=True):
            if not np.array_equal(expected, actual):
                raise ForecastTimingError("forecast timing replay mismatch")
    first_seconds = rows[0]["wall_seconds"]
    warm_seconds = max(row["wall_seconds"] for row in rows[1:])
    projection = projection_seconds(first_seconds, warm_seconds)
    wall = time.perf_counter() - started
    payload = {
        "schema": SCHEMA,
        "mode": "timing-canary",
        "status": "PASSED",
        "q": args.q,
        "target_signature": target.target_signature(),
        "execution_source_signature": source_signature,
        "worker_count": WORKERS_BY_Q[args.q],
        "block_draws": BLOCK_DRAWS,
        "forecast_replication_count": FORECAST_REPLICATION_COUNT,
        "forecast_horizon": FORECAST_HORIZON,
        "seed_hash": hashlib.sha256(seeds.tobytes()).hexdigest(),
        "seed_domain_disjoint_from_material": True,
        "calls": rows,
        "first_block_seconds": first_seconds,
        "warm_block_max_seconds": warm_seconds,
        "phase6_forecast_budget_seconds_with_50pct_margin": projection,
        "phase6_forecast_budget_hours_with_50pct_margin": projection / 3600.0,
        "run_manifest": {
            "git_commit": subprocess.check_output(
                ("git", "rev-parse", "HEAD"), cwd=ROOT, text=True
            ).strip(),
            "command": " ".join(sys.argv),
            "environment": os.environ.get("CONDA_DEFAULT_ENV", "N/A"),
            "python": sys.version.split()[0],
            "tensorflow": tf.__version__,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "physical_gpus": [
                device.name for device in tf.config.list_physical_devices("GPU")
            ],
            "logical_gpus": [device.name for device in tf.config.list_logical_devices("GPU")],
            "jit_compile": True,
            "tf32": bool(tf.config.experimental.tensor_float_32_execution_enabled()),
            "wall_seconds": wall,
            "host_ru_maxrss_bytes": int(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
            ),
            "host_ram_cap_bytes": HOST_RAM_CAP_BYTES,
            "trust_basis": "cpu_hidden_forecast_timing_canary",
            "plan": PLAN.as_posix(),
            "output": args.output.as_posix(),
        },
        "inference_status": {
            "hard_veto_screen": "passed",
            "statistically_supported_ranking": "none",
            "descriptive_only_differences": ["first/warm wall and worker timing dispersion"],
            "default_readiness": "not_assessed",
            "next_evidence_needed": "complete Phase 3--6 numerical budget",
        },
        "nonclaims": [
            "timing and resource canary only",
            "no forecast accuracy or calibration claim",
            "no predictive equivalence, HMC, NeuTra, or model-adequacy claim",
        ],
    }
    write_json(output, payload)
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("contract-smoke", "timing-canary"), required=True)
    parser.add_argument("--q", type=int, choices=Q_VALUES, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout-seconds", type=float, default=7200.0)
    parser.add_argument("--authorize-timing-canary", action="store_true")
    args = parser.parse_args(argv)
    if args.mode == "timing-canary":
        if not args.authorize_timing_canary:
            parser.error("timing-canary requires --authorize-timing-canary")
        if args.output is None:
            parser.error("timing-canary requires --output")
        if not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0.0:
            parser.error("--timeout-seconds must be finite and positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = contract_payload(args.q) if args.mode == "contract-smoke" else run_canary(args)
    print(
        "JSON_SUMMARY "
        + json.dumps(
            {
                "mode": payload["mode"],
                "status": payload["status"],
                "q": payload["q"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
