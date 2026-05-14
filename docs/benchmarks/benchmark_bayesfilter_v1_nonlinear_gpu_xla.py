"""GPU/XLA diagnostic harness for BayesFilter v1 nonlinear sigma-point filters.

This script is intentionally benchmark-only.  It does not change production
behavior and does not certify broad GPU speedups.  Run with ``--device-scope
visible`` only after an escalated GPU/CUDA probe.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import resource
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable


_pre_parser = argparse.ArgumentParser(add_help=False)
_pre_parser.add_argument(
    "--device-scope",
    choices=("cpu", "visible"),
    default="cpu",
    help="Hide GPU by default. Use visible only after escalated GPU probes.",
)
_pre_args, _ = _pre_parser.parse_known_args()
if _pre_args.device_scope == "cpu":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-bayesfilter")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf  # noqa: E402

from bayesfilter.testing import (  # noqa: E402
    make_nonlinear_accumulation_model_tf,
    model_b_observations_tf,
    nonlinear_sigma_point_value_branch_summary,
    tf_nonlinear_sigma_point_value_filter,
)


BACKENDS = ("tf_svd_cut4", "tf_svd_cubature", "tf_svd_ukf")


@dataclass(frozen=True)
class NonlinearGpuXlaConfig:
    device_scope: str
    repeats: int
    warmup_calls: int
    timesteps: int
    backends: tuple[str, ...]
    modes: tuple[str, ...]
    devices: tuple[str, ...]
    dtype: str
    model: str
    claim_scope: str


@dataclass(frozen=True)
class NonlinearGpuXlaRow:
    backend: str
    mode: str
    requested_device: str
    device_name: str | None
    timesteps: int
    state_dim: int
    innovation_dim: int
    observation_dim: int
    point_count: int
    branch_ok_count: int
    branch_total_count: int
    branch_active_floor_count: int
    branch_weak_spectral_gap_count: int
    branch_nonfinite_count: int
    warmup_calls: int
    warmup_seconds: float | None
    first_call_seconds: float | None
    second_call_seconds: float | None
    mean_steady_seconds: float | None
    max_rss_before_mb: float | None
    max_rss_after_mb: float | None
    max_rss_delta_mb: float | None
    value: float | None
    status: str
    error: str | None


def _max_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _observations(timesteps: int) -> tf.Tensor:
    base = model_b_observations_tf()
    repeats = math.ceil(timesteps / int(base.shape[0]))
    return tf.tile(base, [repeats, 1])[:timesteps]


def _parse_csv(raw: str, *, allowed: set[str], name: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in raw.split(",") if item.strip())
    if not values:
        raise ValueError(f"--{name} must include at least one value")
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"unknown --{name} value(s): {', '.join(unknown)}")
    return values


def _logical_device_names(kind: str) -> tuple[str, ...]:
    return tuple(device.name for device in tf.config.list_logical_devices(kind.upper()))


def _selected_device_names(requested: tuple[str, ...]) -> dict[str, str | None]:
    mapping: dict[str, str | None] = {}
    cpu_devices = _logical_device_names("CPU")
    gpu_devices = _logical_device_names("GPU")
    for item in requested:
        if item == "cpu":
            mapping[item] = cpu_devices[0] if cpu_devices else None
        elif item == "gpu":
            mapping[item] = gpu_devices[0] if gpu_devices else None
    return mapping


def _branch_metadata(backend: str) -> dict[str, int]:
    grid = tf.constant(
        [[0.66, 0.23, 0.75], [0.70, 0.25, 0.80], [0.74, 0.27, 0.85]],
        dtype=tf.float64,
    )
    branch = nonlinear_sigma_point_value_branch_summary(
        model_b_observations_tf(),
        grid,
        lambda row: make_nonlinear_accumulation_model_tf(
            rho=row[0],
            sigma=row[1],
            beta=row[2],
        ),
        backend=backend,
    )
    return {
        "point_count": branch.max_point_count,
        "ok_count": branch.ok_count,
        "total_count": branch.total_count,
        "active_floor_count": branch.active_floor_count,
        "weak_spectral_gap_count": branch.weak_spectral_gap_count,
        "nonfinite_count": branch.nonfinite_count,
    }


def _materialize(value: object) -> float:
    return float(tf.convert_to_tensor(value, dtype=tf.float64).numpy())


def _runner(
    observations: tf.Tensor,
    *,
    backend: str,
    device_name: str,
) -> Callable[[], tf.Tensor]:
    model = make_nonlinear_accumulation_model_tf()

    def run() -> tf.Tensor:
        with tf.device(device_name):
            result = tf_nonlinear_sigma_point_value_filter(
                observations,
                model,
                backend=backend,
                return_filtered=False,
            )
            return result.log_likelihood

    return run


def _time_row(
    *,
    backend: str,
    mode: str,
    requested_device: str,
    device_name: str | None,
    observations: tf.Tensor,
    branch: dict[str, int],
    repeats: int,
    warmup_calls: int,
) -> NonlinearGpuXlaRow:
    state_dim = 2
    innovation_dim = 1
    observation_dim = 1
    if device_name is None:
        return NonlinearGpuXlaRow(
            backend=backend,
            mode=mode,
            requested_device=requested_device,
            device_name=None,
            timesteps=int(observations.shape[0]),
            state_dim=state_dim,
            innovation_dim=innovation_dim,
            observation_dim=observation_dim,
            point_count=branch["point_count"],
            branch_ok_count=branch["ok_count"],
            branch_total_count=branch["total_count"],
            branch_active_floor_count=branch["active_floor_count"],
            branch_weak_spectral_gap_count=branch["weak_spectral_gap_count"],
            branch_nonfinite_count=branch["nonfinite_count"],
            warmup_calls=warmup_calls,
            warmup_seconds=None,
            first_call_seconds=None,
            second_call_seconds=None,
            mean_steady_seconds=None,
            max_rss_before_mb=None,
            max_rss_after_mb=None,
            max_rss_delta_mb=None,
            value=None,
            status="skipped_device_unavailable",
            error=f"no logical {requested_device.upper()} device visible",
        )

    raw_call = _runner(observations, backend=backend, device_name=device_name)
    if mode == "eager":
        call = raw_call
    elif mode == "graph":
        call = tf.function(raw_call, reduce_retracing=True)
    elif mode == "xla":
        call = tf.function(raw_call, jit_compile=True, reduce_retracing=True)
    else:  # pragma: no cover - parser validates this.
        raise ValueError(f"unknown mode: {mode}")

    warmup_seconds = None
    timings: list[float] = []
    values: list[float] = []
    max_rss_before = _max_rss_mb()
    try:
        if warmup_calls > 0:
            start = time.perf_counter()
            for _ in range(warmup_calls):
                _materialize(call())
            warmup_seconds = time.perf_counter() - start
        for _ in range(repeats):
            start = time.perf_counter()
            values.append(_materialize(call()))
            timings.append(time.perf_counter() - start)
    except Exception as exc:  # pragma: no cover - diagnostic artifact path.
        max_rss_after = _max_rss_mb()
        return NonlinearGpuXlaRow(
            backend=backend,
            mode=mode,
            requested_device=requested_device,
            device_name=device_name,
            timesteps=int(observations.shape[0]),
            state_dim=state_dim,
            innovation_dim=innovation_dim,
            observation_dim=observation_dim,
            point_count=branch["point_count"],
            branch_ok_count=branch["ok_count"],
            branch_total_count=branch["total_count"],
            branch_active_floor_count=branch["active_floor_count"],
            branch_weak_spectral_gap_count=branch["weak_spectral_gap_count"],
            branch_nonfinite_count=branch["nonfinite_count"],
            warmup_calls=warmup_calls,
            warmup_seconds=warmup_seconds,
            first_call_seconds=timings[0] if timings else None,
            second_call_seconds=timings[1] if len(timings) > 1 else None,
            mean_steady_seconds=None,
            max_rss_before_mb=max_rss_before,
            max_rss_after_mb=max_rss_after,
            max_rss_delta_mb=max_rss_after - max_rss_before,
            value=values[-1] if values else None,
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
        )

    steady = timings[1:] if len(timings) > 1 else timings
    max_rss_after = _max_rss_mb()
    return NonlinearGpuXlaRow(
        backend=backend,
        mode=mode,
        requested_device=requested_device,
        device_name=device_name,
        timesteps=int(observations.shape[0]),
        state_dim=state_dim,
        innovation_dim=innovation_dim,
        observation_dim=observation_dim,
        point_count=branch["point_count"],
        branch_ok_count=branch["ok_count"],
        branch_total_count=branch["total_count"],
        branch_active_floor_count=branch["active_floor_count"],
        branch_weak_spectral_gap_count=branch["weak_spectral_gap_count"],
        branch_nonfinite_count=branch["nonfinite_count"],
        warmup_calls=warmup_calls,
        warmup_seconds=warmup_seconds,
        first_call_seconds=timings[0],
        second_call_seconds=timings[1] if len(timings) > 1 else None,
        mean_steady_seconds=sum(steady) / len(steady),
        max_rss_before_mb=max_rss_before,
        max_rss_after_mb=max_rss_after,
        max_rss_delta_mb=max_rss_after - max_rss_before,
        value=values[-1],
        status="ok",
        error=None,
    )


def _environment(device_scope: str) -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "tensorflow": tf.__version__,
        "device_scope": device_scope,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "logical_devices": [
            {"name": device.name, "device_type": device.device_type}
            for device in tf.config.list_logical_devices()
        ],
        "physical_devices": [
            {"name": device.name, "device_type": device.device_type}
            for device in tf.config.list_physical_devices()
        ],
        "policy": (
            "Rows are diagnostic timing evidence for one fixed Model B shape. "
            "They do not certify broad GPU/XLA speedups or numerical "
            "correctness."
        ),
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _format_optional(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _markdown(payload: dict[str, Any], json_path: Path | None) -> str:
    json_name = str(json_path) if json_path is not None else "stdout"
    rows = [
        "| Backend | Device | Mode | T | Points | Branch OK | Warmup s | First s | Steady s | Status |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in payload["rows"]:
        rows.append(
            "| {backend} | {device} | {mode} | {timesteps} | {points} | "
            "{branch_ok}/{branch_total} | {warmup} | {first} | {steady} | "
            "{status} |".format(
                backend=row["backend"],
                device=row["requested_device"],
                mode=row["mode"],
                timesteps=row["timesteps"],
                points=row["point_count"],
                branch_ok=row["branch_ok_count"],
                branch_total=row["branch_total_count"],
                warmup=_format_optional(row["warmup_seconds"]),
                first=_format_optional(row["first_call_seconds"]),
                steady=_format_optional(row["mean_steady_seconds"]),
                status=row["status"],
            )
        )
    return "\n".join(
        [
            "# BayesFilter V1 Nonlinear GPU/XLA Diagnostic",
            "",
            f"The JSON file is authoritative: `{json_name}`.",
            "",
            "## Claim Scope",
            "",
            str(payload["config"]["claim_scope"]),
            "",
            "## Environment",
            "",
            "```text",
            f"tensorflow = {payload['environment']['tensorflow']}",
            f"device_scope = {payload['environment']['device_scope']}",
            f"cuda_visible_devices = {payload['environment']['cuda_visible_devices']}",
            f"logical_devices = {payload['environment']['logical_devices']}",
            "```",
            "",
            "## Rows",
            "",
            *rows,
            "",
            "## Interpretation Rule",
            "",
            "Only rows with `status = ok` and matching branch metadata can be used "
            "for timing interpretation.  Failed or skipped rows are diagnostic "
            "evidence only.",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device-scope",
        choices=("cpu", "visible"),
        default=_pre_args.device_scope,
        help="Hide GPU by default. Use visible only after escalated GPU probes.",
    )
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--warmup-calls", type=int, default=1)
    parser.add_argument("--timesteps", type=int, default=24)
    parser.add_argument(
        "--backends",
        default="tf_svd_cut4",
        help="Comma-separated backends: tf_svd_cut4,tf_svd_cubature,tf_svd_ukf.",
    )
    parser.add_argument(
        "--modes",
        default="eager,graph,xla",
        help="Comma-separated modes: eager,graph,xla.",
    )
    parser.add_argument(
        "--devices",
        default="cpu,gpu",
        help="Comma-separated requested devices: cpu,gpu.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--markdown-output", type=Path, default=None)
    args = parser.parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be positive")
    if args.warmup_calls < 0:
        raise ValueError("--warmup-calls must be nonnegative")
    if args.timesteps < 1:
        raise ValueError("--timesteps must be positive")
    backends = _parse_csv(args.backends, allowed=set(BACKENDS), name="backends")
    modes = _parse_csv(args.modes, allowed={"eager", "graph", "xla"}, name="modes")
    devices = _parse_csv(args.devices, allowed={"cpu", "gpu"}, name="devices")

    config = NonlinearGpuXlaConfig(
        device_scope=args.device_scope,
        repeats=args.repeats,
        warmup_calls=args.warmup_calls,
        timesteps=args.timesteps,
        backends=backends,
        modes=modes,
        devices=devices,
        dtype="float64",
        model="model_b_nonlinear_accumulation",
        claim_scope=(
            "Diagnostic timing for one fixed nonlinear Model B shape.  "
            "Not a correctness certificate, not a broad speedup claim, and not "
            "a production behavior change."
        ),
    )
    observations = _observations(args.timesteps)
    device_names = _selected_device_names(devices)
    branch_by_backend = {backend: _branch_metadata(backend) for backend in backends}
    rows = [
        _time_row(
            backend=backend,
            mode=mode,
            requested_device=device,
            device_name=device_names[device],
            observations=observations,
            branch=branch_by_backend[backend],
            repeats=args.repeats,
            warmup_calls=args.warmup_calls if mode in {"graph", "xla"} else 0,
        )
        for backend in backends
        for device in devices
        for mode in modes
    ]
    payload = _json_safe(
        {
            "benchmark": "bayesfilter_v1_nonlinear_gpu_xla",
            "config": asdict(config),
            "environment": _environment(args.device_scope),
            "rows": [asdict(row) for row in rows],
        }
    )
    text = json.dumps(payload, allow_nan=False, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    if args.markdown_output is not None:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(_markdown(payload, args.output), encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
