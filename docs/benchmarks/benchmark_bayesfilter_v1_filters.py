"""CPU benchmark harness for BayesFilter v1 filtering candidates.

The harness records timing metadata for small fixed-shape TensorFlow fixtures.
It is intentionally conservative: CPU-only by default, explicit shapes,
first-call timing separated from later-call timing, and no client-specific
readiness claims.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-bayesfilter")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf  # noqa: E402

from bayesfilter import (  # noqa: E402
    StatePartition,
    TFLinearGaussianStateSpace,
    TFLinearGaussianStateSpaceDerivatives,
    make_affine_structural_tf,
    tf_qr_linear_gaussian_log_likelihood,
    tf_qr_linear_gaussian_score_hessian,
    tf_svd_cut4_log_likelihood,
    tf_svd_linear_gaussian_log_likelihood,
    tf_svd_sigma_point_log_likelihood,
)


@dataclass(frozen=True)
class BenchmarkConfig:
    repeats: int
    timesteps: int
    state_dim: int
    observation_dim: int
    parameter_dim: int
    dtype: str
    seed: int


@dataclass(frozen=True)
class BenchmarkResult:
    benchmark: str
    backend: str
    mode: str
    timesteps: int
    state_dim: int
    observation_dim: int
    stochastic_rank: int | None
    point_count: int | None
    parameter_dim: int | None
    first_call_seconds: float | None
    second_call_seconds: float | None
    mean_steady_seconds: float | None
    rss_before_mb: float | None
    rss_after_mb: float | None
    rss_delta_mb: float | None
    max_rss_before_mb: float | None
    max_rss_after_mb: float | None
    max_rss_delta_mb: float | None
    status: str
    value: float | None
    error: str | None


def _max_rss_mb() -> float:
    # Linux reports ru_maxrss in KiB.  The benchmark is currently Linux/WSL2.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _current_rss_mb() -> float | None:
    statm = Path("/proc/self/statm")
    if not statm.exists():
        return None
    try:
        fields = statm.read_text(encoding="utf-8").split()
        resident_pages = int(fields[1])
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (IndexError, OSError, ValueError):
        return None
    return resident_pages * page_size / (1024.0 * 1024.0)


def _linear_model(
    state_dim: int,
    observation_dim: int,
) -> tuple[TFLinearGaussianStateSpace, tf.Tensor]:
    dtype = tf.float64
    diag = tf.linspace(
        tf.constant(0.45, dtype),
        tf.constant(0.70, dtype),
        state_dim,
    )
    transition_matrix = tf.linalg.diag(diag)
    transition_covariance = tf.linalg.diag(
        tf.linspace(tf.constant(0.04, dtype), tf.constant(0.07, dtype), state_dim)
    )
    observation_matrix = tf.reshape(
        tf.linspace(
            tf.constant(0.15, dtype),
            tf.constant(0.95, dtype),
            observation_dim * state_dim,
        ),
        [observation_dim, state_dim],
    )
    observation_covariance = tf.linalg.diag(
        tf.linspace(
            tf.constant(0.08, dtype),
            tf.constant(0.12, dtype),
            observation_dim,
        )
    )
    model = TFLinearGaussianStateSpace(
        initial_mean=tf.zeros([state_dim], dtype=dtype),
        initial_covariance=tf.eye(state_dim, dtype=dtype),
        transition_offset=tf.linspace(
            tf.constant(0.0, dtype),
            tf.constant(0.02, dtype),
            state_dim,
        ),
        transition_matrix=transition_matrix,
        transition_covariance=transition_covariance,
        observation_offset=tf.zeros([observation_dim], dtype=dtype),
        observation_matrix=observation_matrix,
        observation_covariance=observation_covariance,
    )
    return model, observation_matrix


def _observations(timesteps: int, observation_dim: int) -> tf.Tensor:
    values = tf.linspace(
        tf.constant(-0.25, dtype=tf.float64),
        tf.constant(0.35, dtype=tf.float64),
        timesteps * observation_dim,
    )
    return tf.reshape(values, [timesteps, observation_dim])


def _derivatives(
    model: TFLinearGaussianStateSpace,
    parameter_dim: int,
) -> TFLinearGaussianStateSpaceDerivatives:
    state_dim = int(model.initial_mean.shape[0])
    observation_dim = int(model.observation_offset.shape[0])
    dtype = tf.float64
    d_transition_matrix = tf.zeros([parameter_dim, state_dim, state_dim], dtype=dtype)
    d_transition_matrix = tf.tensor_scatter_nd_update(
        d_transition_matrix,
        indices=tf.constant([[0, 0, 0]], dtype=tf.int32),
        updates=tf.constant([0.05], dtype=dtype),
    )
    d_observation_covariance = tf.zeros(
        [parameter_dim, observation_dim, observation_dim],
        dtype=dtype,
    )
    if parameter_dim > 1:
        d_observation_covariance = tf.tensor_scatter_nd_update(
            d_observation_covariance,
            indices=tf.constant([[1, 0, 0]], dtype=tf.int32),
            updates=tf.constant([0.02], dtype=dtype),
        )
    return TFLinearGaussianStateSpaceDerivatives(
        d_initial_mean=tf.zeros([parameter_dim, state_dim], dtype=dtype),
        d_initial_covariance=tf.zeros([parameter_dim, state_dim, state_dim], dtype=dtype),
        d_transition_offset=tf.zeros([parameter_dim, state_dim], dtype=dtype),
        d_transition_matrix=d_transition_matrix,
        d_transition_covariance=tf.zeros([parameter_dim, state_dim, state_dim], dtype=dtype),
        d_observation_offset=tf.zeros([parameter_dim, observation_dim], dtype=dtype),
        d_observation_matrix=tf.zeros(
            [parameter_dim, observation_dim, state_dim],
            dtype=dtype,
        ),
        d_observation_covariance=d_observation_covariance,
        d2_initial_mean=tf.zeros([parameter_dim, parameter_dim, state_dim], dtype=dtype),
        d2_initial_covariance=tf.zeros(
            [parameter_dim, parameter_dim, state_dim, state_dim],
            dtype=dtype,
        ),
        d2_transition_offset=tf.zeros([parameter_dim, parameter_dim, state_dim], dtype=dtype),
        d2_transition_matrix=tf.zeros(
            [parameter_dim, parameter_dim, state_dim, state_dim],
            dtype=dtype,
        ),
        d2_transition_covariance=tf.zeros(
            [parameter_dim, parameter_dim, state_dim, state_dim],
            dtype=dtype,
        ),
        d2_observation_offset=tf.zeros(
            [parameter_dim, parameter_dim, observation_dim],
            dtype=dtype,
        ),
        d2_observation_matrix=tf.zeros(
            [parameter_dim, parameter_dim, observation_dim, state_dim],
            dtype=dtype,
        ),
        d2_observation_covariance=tf.zeros(
            [parameter_dim, parameter_dim, observation_dim, observation_dim],
            dtype=dtype,
        ),
    )


def _structural_model():
    partition = StatePartition(
        state_names=("m", "lag_m"),
        stochastic_indices=(0,),
        deterministic_indices=(1,),
        innovation_dim=1,
    )
    return make_affine_structural_tf(
        partition=partition,
        initial_mean=tf.constant([0.1, -0.1], dtype=tf.float64),
        initial_covariance=tf.linalg.diag(tf.constant([1.2, 0.7], dtype=tf.float64)),
        transition_offset=tf.zeros([2], dtype=tf.float64),
        transition_matrix=tf.constant([[0.35, -0.10], [1.0, 0.0]], dtype=tf.float64),
        innovation_matrix=tf.constant([[0.25], [0.0]], dtype=tf.float64),
        innovation_covariance=tf.constant([[0.43]], dtype=tf.float64),
        observation_offset=tf.zeros([1], dtype=tf.float64),
        observation_matrix=tf.constant([[1.0, 0.25]], dtype=tf.float64),
        observation_covariance=tf.constant([[0.19]], dtype=tf.float64),
    )


def _materialize(value) -> float:
    if hasattr(value, "log_likelihood"):
        tensor = value.log_likelihood
    elif isinstance(value, tuple):
        tensor = value[0]
    else:
        tensor = value
    return float(tf.convert_to_tensor(tensor).numpy())


def _time_runner(
    name: str,
    backend: str,
    mode: str,
    runner: Callable[[], object],
    *,
    repeats: int,
    timesteps: int,
    state_dim: int,
    observation_dim: int,
    stochastic_rank: int | None = None,
    point_count: int | None = None,
    parameter_dim: int | None = None,
) -> BenchmarkResult:
    call = runner if mode == "eager" else tf.function(runner, reduce_retracing=True)
    timings: list[float] = []
    values: list[float] = []
    rss_before = _current_rss_mb()
    max_rss_before = _max_rss_mb()
    try:
        for _ in range(repeats):
            start = time.perf_counter()
            values.append(_materialize(call()))
            timings.append(time.perf_counter() - start)
    except Exception as exc:  # pragma: no cover - benchmark records failures.
        rss_after = _current_rss_mb()
        max_rss_after = _max_rss_mb()
        return BenchmarkResult(
            benchmark=name,
            backend=backend,
            mode=mode,
            timesteps=timesteps,
            state_dim=state_dim,
            observation_dim=observation_dim,
            stochastic_rank=stochastic_rank,
            point_count=point_count,
            parameter_dim=parameter_dim,
            first_call_seconds=timings[0] if timings else None,
            second_call_seconds=timings[1] if len(timings) > 1 else None,
            mean_steady_seconds=None,
            rss_before_mb=rss_before,
            rss_after_mb=rss_after,
            rss_delta_mb=(
                None if rss_before is None or rss_after is None else rss_after - rss_before
            ),
            max_rss_before_mb=max_rss_before,
            max_rss_after_mb=max_rss_after,
            max_rss_delta_mb=max_rss_after - max_rss_before,
            status="failed",
            value=values[-1] if values else None,
            error=f"{type(exc).__name__}: {exc}",
        )

    steady = timings[1:] if len(timings) > 1 else timings
    rss_after = _current_rss_mb()
    max_rss_after = _max_rss_mb()
    return BenchmarkResult(
        benchmark=name,
        backend=backend,
        mode=mode,
        timesteps=timesteps,
        state_dim=state_dim,
        observation_dim=observation_dim,
        stochastic_rank=stochastic_rank,
        point_count=point_count,
        parameter_dim=parameter_dim,
        first_call_seconds=timings[0],
        second_call_seconds=timings[1] if len(timings) > 1 else None,
        mean_steady_seconds=sum(steady) / len(steady),
        rss_before_mb=rss_before,
        rss_after_mb=rss_after,
        rss_delta_mb=(
            None if rss_before is None or rss_after is None else rss_after - rss_before
        ),
        max_rss_before_mb=max_rss_before,
        max_rss_after_mb=max_rss_after,
        max_rss_delta_mb=max_rss_after - max_rss_before,
        status="ok",
        value=values[-1],
        error=None,
    )


def run_benchmarks(config: BenchmarkConfig) -> list[BenchmarkResult]:
    model, _ = _linear_model(config.state_dim, config.observation_dim)
    observations = _observations(config.timesteps, config.observation_dim)
    derivatives = _derivatives(model, config.parameter_dim)
    structural = _structural_model()
    structural_observations = _observations(config.timesteps, 1)

    cases: list[tuple[str, str, Callable[[], object], dict[str, int | None]]] = [
        (
            "linear_qr_value",
            "tf_qr",
            lambda: tf_qr_linear_gaussian_log_likelihood(
                observations,
                model,
                backend="tf_qr",
                jitter=tf.constant(1e-9, dtype=tf.float64),
            ).log_likelihood,
            {"stochastic_rank": None, "point_count": None, "parameter_dim": None},
        ),
        (
            "linear_qr_score_hessian",
            "tf_qr_sqrt",
            lambda: tf_qr_linear_gaussian_score_hessian(
                observations,
                model,
                derivatives,
                backend="tf_qr_sqrt",
                jitter=tf.constant(1e-9, dtype=tf.float64),
            ).log_likelihood,
            {
                "stochastic_rank": None,
                "point_count": None,
                "parameter_dim": config.parameter_dim,
            },
        ),
        (
            "linear_svd_value",
            "tf_svd",
            lambda: tf_svd_linear_gaussian_log_likelihood(
                observations,
                model,
                backend="tf_svd",
                jitter=tf.constant(1e-9, dtype=tf.float64),
                singular_floor=tf.constant(1e-12, dtype=tf.float64),
            ).log_likelihood,
            {"stochastic_rank": None, "point_count": None, "parameter_dim": None},
        ),
        (
            "svd_cubature_value",
            "tf_svd_cubature",
            lambda: tf_svd_sigma_point_log_likelihood(
                structural_observations,
                structural,
                rule="cubature",
                innovation_floor=tf.constant(1e-12, dtype=tf.float64),
            )[0],
            {"stochastic_rank": 1, "point_count": 6, "parameter_dim": None},
        ),
        (
            "svd_ukf_value",
            "tf_svd_ukf",
            lambda: tf_svd_sigma_point_log_likelihood(
                structural_observations,
                structural,
                rule="unscented",
                innovation_floor=tf.constant(1e-12, dtype=tf.float64),
            )[0],
            {"stochastic_rank": 1, "point_count": 7, "parameter_dim": None},
        ),
        (
            "svd_cut4_value",
            "tf_svd_cut4",
            lambda: tf_svd_cut4_log_likelihood(
                structural_observations,
                structural,
                innovation_floor=tf.constant(1e-12, dtype=tf.float64),
            )[0],
            {"stochastic_rank": 1, "point_count": 14, "parameter_dim": None},
        ),
    ]
    results: list[BenchmarkResult] = []
    for mode in ("eager", "graph"):
        for name, backend, runner, metadata in cases:
            results.append(
                _time_runner(
                    name,
                    backend,
                    mode,
                    runner,
                    repeats=config.repeats,
                    timesteps=(
                        config.timesteps
                        if name.startswith("linear")
                        else int(structural_observations.shape[0])
                    ),
                    state_dim=config.state_dim if name.startswith("linear") else 2,
                    observation_dim=(
                        config.observation_dim if name.startswith("linear") else 1
                    ),
                    stochastic_rank=metadata["stochastic_rank"],
                    point_count=metadata["point_count"],
                    parameter_dim=metadata["parameter_dim"],
                )
            )
    return results


def _environment() -> dict[str, object]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "tensorflow": tf.__version__,
        "logical_devices": [
            {"name": device.name, "device_type": device.device_type}
            for device in tf.config.list_logical_devices()
        ],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device_policy": (
            "CPU-only benchmark harness; GPU/XLA-GPU claims require separate "
            "escalated probes and matching shapes."
        ),
        "memory_policy": (
            "Process-level RSS snapshots are recorded before and after each "
            "benchmark row.  max_rss is process high-water RSS, so row deltas "
            "are diagnostic metadata, not isolated allocation profiles."
        ),
    }


def _format_optional(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _markdown_report(payload: dict[str, object], json_path: Path | None) -> str:
    config = payload["config"]
    environment = payload["environment"]
    results = payload["results"]
    json_name = str(json_path) if json_path is not None else "stdout"
    rows = [
        "| Benchmark | Backend | Mode | First s | Steady s | RSS delta MB | Max RSS delta MB | Points | Status |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in results:
        rows.append(
            "| {benchmark} | {backend} | {mode} | {first} | {steady} | "
            "{rss_delta} | {max_delta} | {points} | {status} |".format(
                benchmark=row["benchmark"],
                backend=row["backend"],
                mode=row["mode"],
                first=_format_optional(row["first_call_seconds"]),
                steady=_format_optional(row["mean_steady_seconds"]),
                rss_delta=_format_optional(row["rss_delta_mb"]),
                max_delta=_format_optional(row["max_rss_delta_mb"]),
                points=_format_optional(row["point_count"]),
                status=row["status"],
            )
        )

    return "\n".join(
        [
            "# BayesFilter v1 Filter CPU Benchmark",
            "",
            "Purpose: record CPU-only timing and process-memory metadata for "
            "BayesFilter v1 filtering candidates.",
            "",
            "## Claim Scope",
            "",
            str(payload["claim_scope"]),
            "",
            "## Configuration",
            "",
            "```text",
            f"repeats = {config['repeats']}",
            f"timesteps = {config['timesteps']}",
            f"state_dim = {config['state_dim']}",
            f"observation_dim = {config['observation_dim']}",
            f"parameter_dim = {config['parameter_dim']}",
            f"dtype = {config['dtype']}",
            "```",
            "",
            "## Environment",
            "",
            "```text",
            f"python = {environment['python']}",
            f"platform = {environment['platform']}",
            f"tensorflow = {environment['tensorflow']}",
            f"cuda_visible_devices = {environment['cuda_visible_devices']}",
            f"logical_devices = {environment['logical_devices']}",
            "```",
            "",
            str(environment["device_policy"]),
            "",
            str(environment["memory_policy"]),
            "",
            "## Result",
            "",
            f"The JSON file is authoritative: `{json_name}`.",
            "",
            *rows,
            "",
            "## Interpretation",
            "",
            "Rows with `status = ok` completed for the declared fixed shape.  "
            "First-call timing includes tracing/initialization effects for graph "
            "mode.  Steady timing uses calls after the first observation.  Memory "
            "fields are process-level diagnostics and should not be interpreted as "
            "isolated per-backend allocation profiles.",
            "",
            "This artifact does not certify MacroFinance/DSGE switch-over, "
            "GPU/XLA-GPU readiness, or HMC readiness.",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timesteps", type=int, default=8)
    parser.add_argument("--state-dim", type=int, default=2)
    parser.add_argument("--observation-dim", type=int, default=2)
    parser.add_argument("--parameter-dim", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260510)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--markdown-output", type=Path, default=None)
    args = parser.parse_args()

    if args.repeats < 1:
        raise ValueError("--repeats must be positive")
    if args.state_dim < 1 or args.observation_dim < 1:
        raise ValueError("state and observation dimensions must be positive")
    if args.parameter_dim < 2:
        raise ValueError("--parameter-dim must be at least 2")

    config = BenchmarkConfig(
        repeats=args.repeats,
        timesteps=args.timesteps,
        state_dim=args.state_dim,
        observation_dim=args.observation_dim,
        parameter_dim=args.parameter_dim,
        dtype="float64",
        seed=args.seed,
    )
    payload = {
        "benchmark": "bayesfilter_v1_filters_cpu",
        "claim_scope": (
            "Benchmark artifact only.  Not a client switch-over, GPU, XLA-GPU, "
            "or HMC readiness claim."
        ),
        "config": asdict(config),
        "environment": _environment(),
        "results": [asdict(row) for row in run_benchmarks(config)],
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    if args.markdown_output is not None:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(
            _markdown_report(payload, args.output).rstrip() + "\n",
            encoding="utf-8",
        )
    print(text)


if __name__ == "__main__":
    main()
