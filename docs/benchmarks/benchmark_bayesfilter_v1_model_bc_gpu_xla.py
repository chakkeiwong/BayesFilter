"""Model B/C GPU/XLA scaling diagnostics for BayesFilter V1.

This benchmark is diagnostic infrastructure only.  It records shape-specific
timings for nonlinear sigma-point value filters and records score branch gates
for the same model/filter/horizon cells.  It does not certify broad GPU speedup
and does not change production behavior.
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
    make_nonlinear_accumulation_first_derivatives_tf,
    make_nonlinear_accumulation_model_tf,
    make_univariate_nonlinear_growth_first_derivatives_tf,
    make_univariate_nonlinear_growth_model_tf,
    model_b_observations_tf,
    model_c_observations_tf,
    nonlinear_sigma_point_score_branch_summary,
    nonlinear_sigma_point_value_branch_summary,
    tf_nonlinear_sigma_point_value_filter,
)


MODELS = ("model_b_nonlinear_accumulation", "model_c_autonomous_nonlinear_growth")
BACKENDS = ("tf_svd_cubature", "tf_svd_ukf", "tf_svd_cut4")


@dataclass(frozen=True)
class BranchMetadata:
    value_ok_count: int
    value_total_count: int
    value_active_floor_count: int
    value_weak_spectral_gap_count: int
    value_nonfinite_count: int
    score_ok_count: int
    score_total_count: int
    score_active_floor_count: int
    score_weak_spectral_gap_count: int
    score_nonfinite_count: int
    score_failure_labels: tuple[str, ...]
    point_count: int
    allow_fixed_null_support: bool


@dataclass(frozen=True)
class TimingRow:
    model: str
    backend: str
    mode: str
    requested_device: str
    device_name: str | None
    timesteps: int
    point_count: int
    allow_fixed_null_support: bool
    value_branch_ok_count: int
    value_branch_total_count: int
    score_branch_ok_count: int
    score_branch_total_count: int
    score_branch_failure_labels: tuple[str, ...]
    warmup_calls: int
    repeats: int
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


def _parse_csv(raw: str, *, allowed: set[str], name: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in raw.split(",") if item.strip())
    if not values:
        raise ValueError(f"--{name} must include at least one value")
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"unknown --{name} value(s): {', '.join(unknown)}")
    return values


def _parse_int_csv(raw: str, *, name: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    if not values:
        raise ValueError(f"--{name} must include at least one value")
    if any(value < 1 for value in values):
        raise ValueError(f"--{name} values must be positive")
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


def _base_observations(model: str) -> tf.Tensor:
    if model == "model_b_nonlinear_accumulation":
        return model_b_observations_tf()
    if model == "model_c_autonomous_nonlinear_growth":
        return model_c_observations_tf()
    raise ValueError(f"unknown model: {model}")


def _observations(model: str, timesteps: int) -> tf.Tensor:
    base = _base_observations(model)
    repeats = math.ceil(timesteps / int(base.shape[0]))
    return tf.tile(base, [repeats, 1])[:timesteps]


def _parameter_grid(model: str) -> tf.Tensor:
    if model == "model_b_nonlinear_accumulation":
        return tf.constant(
            [[0.62, 0.20, 0.70], [0.70, 0.25, 0.80], [0.78, 0.30, 0.90]],
            dtype=tf.float64,
        )
    if model == "model_c_autonomous_nonlinear_growth":
        return tf.constant(
            [[0.90, 1.00, 0.20], [1.00, 1.00, 0.20], [1.10, 1.10, 0.25]],
            dtype=tf.float64,
        )
    raise ValueError(f"unknown model: {model}")


def _allow_fixed_null_support(model: str) -> bool:
    return model == "model_c_autonomous_nonlinear_growth"


def _model_from_params(model: str, params: tf.Tensor):
    if model == "model_b_nonlinear_accumulation":
        return make_nonlinear_accumulation_model_tf(
            rho=params[0],
            sigma=params[1],
            beta=params[2],
        )
    if model == "model_c_autonomous_nonlinear_growth":
        return make_univariate_nonlinear_growth_model_tf(
            process_sigma=params[0],
            observation_sigma=params[1],
            initial_variance=params[2],
        )
    raise ValueError(f"unknown model: {model}")


def _derivatives_from_params(model: str, params: tf.Tensor):
    if model == "model_b_nonlinear_accumulation":
        return make_nonlinear_accumulation_first_derivatives_tf(
            rho=params[0],
            sigma=params[1],
            beta=params[2],
        )
    if model == "model_c_autonomous_nonlinear_growth":
        return make_univariate_nonlinear_growth_first_derivatives_tf(
            process_sigma=params[0],
            observation_sigma=params[1],
        )
    raise ValueError(f"unknown model: {model}")


def _default_model(model: str):
    if model == "model_b_nonlinear_accumulation":
        return make_nonlinear_accumulation_model_tf()
    if model == "model_c_autonomous_nonlinear_growth":
        return make_univariate_nonlinear_growth_model_tf()
    raise ValueError(f"unknown model: {model}")


def _branch_metadata(model: str, backend: str, observations: tf.Tensor) -> BranchMetadata:
    grid = _parameter_grid(model)
    value_summary = nonlinear_sigma_point_value_branch_summary(
        observations,
        grid,
        lambda row: _model_from_params(model, row),
        backend=backend,
        innovation_floor=tf.constant(1e-12, dtype=tf.float64),
    )
    score_summary = nonlinear_sigma_point_score_branch_summary(
        observations,
        grid,
        lambda row: _model_from_params(model, row),
        lambda row: _derivatives_from_params(model, row),
        backend=backend,
        innovation_floor=tf.constant(1e-12, dtype=tf.float64),
        spectral_gap_tolerance=tf.constant(1e-8, dtype=tf.float64),
        allow_fixed_null_support=_allow_fixed_null_support(model),
    )
    return BranchMetadata(
        value_ok_count=value_summary.ok_count,
        value_total_count=value_summary.total_count,
        value_active_floor_count=value_summary.active_floor_count,
        value_weak_spectral_gap_count=value_summary.weak_spectral_gap_count,
        value_nonfinite_count=value_summary.nonfinite_count,
        score_ok_count=score_summary.ok_count,
        score_total_count=score_summary.total_count,
        score_active_floor_count=score_summary.active_floor_count,
        score_weak_spectral_gap_count=score_summary.weak_spectral_gap_count,
        score_nonfinite_count=score_summary.nonfinite_count,
        score_failure_labels=score_summary.failure_labels,
        point_count=max(value_summary.max_point_count, score_summary.max_point_count),
        allow_fixed_null_support=_allow_fixed_null_support(model),
    )


def _materialize(value: object) -> float:
    return float(tf.convert_to_tensor(value, dtype=tf.float64).numpy())


def _runner(
    *,
    model_name: str,
    observations: tf.Tensor,
    backend: str,
    device_name: str,
) -> Callable[[], tf.Tensor]:
    model = _default_model(model_name)

    def run() -> tf.Tensor:
        with tf.device(device_name):
            result = tf_nonlinear_sigma_point_value_filter(
                observations,
                model,
                backend=backend,
                innovation_floor=tf.constant(1e-12, dtype=tf.float64),
                return_filtered=False,
            )
            return result.log_likelihood

    return run


def _skipped_row(
    *,
    model: str,
    backend: str,
    mode: str,
    requested_device: str,
    device_name: str | None,
    timesteps: int,
    branch: BranchMetadata,
    repeats: int,
    warmup_calls: int,
    status: str,
    error: str,
) -> TimingRow:
    return TimingRow(
        model=model,
        backend=backend,
        mode=mode,
        requested_device=requested_device,
        device_name=device_name,
        timesteps=timesteps,
        point_count=branch.point_count,
        allow_fixed_null_support=branch.allow_fixed_null_support,
        value_branch_ok_count=branch.value_ok_count,
        value_branch_total_count=branch.value_total_count,
        score_branch_ok_count=branch.score_ok_count,
        score_branch_total_count=branch.score_total_count,
        score_branch_failure_labels=branch.score_failure_labels,
        warmup_calls=warmup_calls,
        repeats=repeats,
        warmup_seconds=None,
        first_call_seconds=None,
        second_call_seconds=None,
        mean_steady_seconds=None,
        max_rss_before_mb=None,
        max_rss_after_mb=None,
        max_rss_delta_mb=None,
        value=None,
        status=status,
        error=error,
    )


def _time_row(
    *,
    model: str,
    backend: str,
    mode: str,
    requested_device: str,
    device_name: str | None,
    observations: tf.Tensor,
    branch: BranchMetadata,
    repeats: int,
    warmup_calls: int,
) -> TimingRow:
    timesteps = int(observations.shape[0])
    if device_name is None:
        return _skipped_row(
            model=model,
            backend=backend,
            mode=mode,
            requested_device=requested_device,
            device_name=None,
            timesteps=timesteps,
            branch=branch,
            repeats=repeats,
            warmup_calls=warmup_calls,
            status="skipped_device_unavailable",
            error=f"no logical {requested_device.upper()} device visible",
        )
    if branch.value_ok_count != branch.value_total_count:
        return _skipped_row(
            model=model,
            backend=backend,
            mode=mode,
            requested_device=requested_device,
            device_name=device_name,
            timesteps=timesteps,
            branch=branch,
            repeats=repeats,
            warmup_calls=warmup_calls,
            status="blocked_value_branch_gate",
            error="value branch gate did not pass for this shape",
        )
    if branch.score_ok_count != branch.score_total_count:
        return _skipped_row(
            model=model,
            backend=backend,
            mode=mode,
            requested_device=requested_device,
            device_name=device_name,
            timesteps=timesteps,
            branch=branch,
            repeats=repeats,
            warmup_calls=warmup_calls,
            status="blocked_score_branch_gate",
            error="score branch gate did not pass for this shape",
        )

    raw_call = _runner(
        model_name=model,
        observations=observations,
        backend=backend,
        device_name=device_name,
    )
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
        return TimingRow(
            model=model,
            backend=backend,
            mode=mode,
            requested_device=requested_device,
            device_name=device_name,
            timesteps=timesteps,
            point_count=branch.point_count,
            allow_fixed_null_support=branch.allow_fixed_null_support,
            value_branch_ok_count=branch.value_ok_count,
            value_branch_total_count=branch.value_total_count,
            score_branch_ok_count=branch.score_ok_count,
            score_branch_total_count=branch.score_total_count,
            score_branch_failure_labels=branch.score_failure_labels,
            warmup_calls=warmup_calls,
            repeats=repeats,
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
    return TimingRow(
        model=model,
        backend=backend,
        mode=mode,
        requested_device=requested_device,
        device_name=device_name,
        timesteps=timesteps,
        point_count=branch.point_count,
        allow_fixed_null_support=branch.allow_fixed_null_support,
        value_branch_ok_count=branch.value_ok_count,
        value_branch_total_count=branch.value_total_count,
        score_branch_ok_count=branch.score_ok_count,
        score_branch_total_count=branch.score_total_count,
        score_branch_failure_labels=branch.score_failure_labels,
        warmup_calls=warmup_calls,
        repeats=repeats,
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
            "Rows are diagnostic timing evidence for exact Model B/C shapes. "
            "They do not certify broad GPU/XLA speedups or production behavior."
        ),
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return value


def _format_optional(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _markdown(payload: dict[str, Any], json_path: Path) -> str:
    rows = [
        "| Model | Backend | T | Device | Mode | Branch | Steady s | Status |",
        "| --- | --- | ---: | --- | --- | ---: | ---: | --- |",
    ]
    for row in payload["rows"]:
        rows.append(
            "| {model} | {backend} | {timesteps} | {device} | {mode} | "
            "{score_ok}/{score_total} | {steady} | {status} |".format(
                model=row["model"],
                backend=row["backend"],
                timesteps=row["timesteps"],
                device=row["requested_device"],
                mode=row["mode"],
                score_ok=row["score_branch_ok_count"],
                score_total=row["score_branch_total_count"],
                steady=_format_optional(row["mean_steady_seconds"]),
                status=row["status"],
            )
        )
    return "\n".join(
        [
            "# BayesFilter V1 Model B/C GPU/XLA Scaling Diagnostic",
            "",
            f"The JSON file is authoritative: `{json_path}`.",
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
            "for timing interpretation.  Failed, skipped, or branch-blocked rows "
            "are diagnostic evidence only.",
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
    parser.add_argument("--timesteps", default="8,16")
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--backends", default=",".join(BACKENDS))
    parser.add_argument("--modes", default="graph,xla")
    parser.add_argument("--devices", default="cpu,gpu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be positive")
    if args.warmup_calls < 0:
        raise ValueError("--warmup-calls must be nonnegative")
    models = _parse_csv(args.models, allowed=set(MODELS), name="models")
    backends = _parse_csv(args.backends, allowed=set(BACKENDS), name="backends")
    modes = _parse_csv(args.modes, allowed={"eager", "graph", "xla"}, name="modes")
    devices = _parse_csv(args.devices, allowed={"cpu", "gpu"}, name="devices")
    timesteps = _parse_int_csv(args.timesteps, name="timesteps")

    device_names = _selected_device_names(devices)
    rows: list[TimingRow] = []
    for model in models:
        for timestep in timesteps:
            observations = _observations(model, timestep)
            for backend in backends:
                branch = _branch_metadata(model, backend, observations)
                for device in devices:
                    for mode in modes:
                        rows.append(
                            _time_row(
                                model=model,
                                backend=backend,
                                mode=mode,
                                requested_device=device,
                                device_name=device_names[device],
                                observations=observations,
                                branch=branch,
                                repeats=args.repeats,
                                warmup_calls=(
                                    args.warmup_calls
                                    if mode in {"graph", "xla"}
                                    else 0
                                ),
                            )
                        )

    payload = _json_safe(
        {
            "benchmark": "bayesfilter_v1_model_bc_gpu_xla",
            "config": {
                "device_scope": args.device_scope,
                "repeats": args.repeats,
                "warmup_calls": args.warmup_calls,
                "timesteps": timesteps,
                "models": models,
                "backends": backends,
                "modes": modes,
                "devices": devices,
                "dtype": "float64",
                "claim_scope": (
                    "Diagnostic timing for exact Model B/C nonlinear "
                    "sigma-point shapes only.  Not a broad speedup claim."
                ),
            },
            "environment": _environment(args.device_scope),
            "rows": [asdict(row) for row in rows],
        }
    )
    args.output.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.markdown_output.write_text(_markdown(payload, args.output) + "\n", encoding="utf-8")
    status_counts: dict[str, int] = {}
    for row in payload["rows"]:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
    print(json.dumps(status_counts, sort_keys=True))


if __name__ == "__main__":
    main()
