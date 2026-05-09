"""Common result contract for quarantined student DPF baseline adapters."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from enum import StrEnum
import json
from pathlib import Path
import sys
import time
from typing import Any, Iterator


class BaselineStatus(StrEnum):
    """Normalized status labels for student-baseline runs."""

    OK = "ok"
    FAILED = "failed"
    SKIPPED = "skipped"
    BLOCKED_MISSING_DEPENDENCY = "blocked_missing_dependency"
    BLOCKED_ENVIRONMENT_DRIFT = "blocked_environment_drift"
    BLOCKED_MISSING_ASSUMPTION = "blocked_missing_assumption"
    ADAPTER_ERROR = "adapter_error"


@dataclass(slots=True)
class BaselineResult:
    """Serializable result object used by all student-baseline adapters."""

    implementation_name: str
    source_commit: str | None
    fixture_name: str
    seed: int | None = None
    num_particles: int | None = None
    dtype: str | None = None
    status: BaselineStatus = BaselineStatus.OK
    failure_reason: str | None = None
    log_likelihood: float | None = None
    likelihood_surrogate: float | None = None
    filtered_means: Any | None = None
    filtered_covariances: Any | None = None
    particle_means: Any | None = None
    particle_covariances: Any | None = None
    ess_by_time: Any | None = None
    resampling_count: int | None = None
    runtime_seconds: float | None = None
    gradient_available: bool = False
    gradient_value: Any | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["status"] = str(self.status)
        return _json_safe(data)

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


def blocked_result(
    *,
    implementation_name: str,
    source_commit: str | None,
    fixture_name: str,
    status: BaselineStatus,
    failure_reason: str,
    seed: int | None = None,
    num_particles: int | None = None,
    dtype: str | None = None,
    runtime_seconds: float | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> BaselineResult:
    """Build a structured blocked/failed result without fabricating metrics."""

    return BaselineResult(
        implementation_name=implementation_name,
        source_commit=source_commit,
        fixture_name=fixture_name,
        seed=seed,
        num_particles=num_particles,
        dtype=dtype,
        status=status,
        failure_reason=failure_reason,
        runtime_seconds=runtime_seconds,
        diagnostics=diagnostics or {},
    )


def exception_result(
    *,
    implementation_name: str,
    source_commit: str | None,
    fixture_name: str,
    exc: BaseException,
    seed: int | None = None,
    num_particles: int | None = None,
    dtype: str | None = None,
    runtime_seconds: float | None = None,
) -> BaselineResult:
    """Convert an adapter exception into a serializable failed result."""

    return blocked_result(
        implementation_name=implementation_name,
        source_commit=source_commit,
        fixture_name=fixture_name,
        seed=seed,
        num_particles=num_particles,
        dtype=dtype,
        status=BaselineStatus.ADAPTER_ERROR,
        failure_reason=f"{type(exc).__name__}: {exc}",
        runtime_seconds=runtime_seconds,
    )


def write_json(path: str | Path, payload: Any) -> None:
    """Write a JSON payload, creating parent directories if needed."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


@contextmanager
def prepend_sys_path(*paths: str | Path) -> Iterator[None]:
    """Temporarily prepend import paths for vendored student code."""

    original = list(sys.path)
    for path in reversed([str(Path(p)) for p in paths]):
        if path not in sys.path:
            sys.path.insert(0, path)
    try:
        yield
    finally:
        sys.path[:] = original


@contextmanager
def elapsed_timer() -> Iterator[dict[str, float]]:
    """Measure elapsed wall-clock seconds for a run."""

    state: dict[str, float] = {}
    start = time.perf_counter()
    try:
        yield state
    finally:
        state["runtime_seconds"] = time.perf_counter() - start


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, StrEnum):
        return str(value)
    if hasattr(value, "numpy"):
        return _json_safe(value.numpy())
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)
