"""Shared host-side validation and diagnostic helpers for HMC preparation."""
from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Mapping

from bayesfilter.inference.hmc import PrecomputedMassArtifact
from bayesfilter.inference.hmc_artifact_identity import mass_artifact_signature


def _validate_band(values: Sequence[float], *, name: str) -> tuple[float, float]:
    values = tuple(values)
    if len(values) != 2:
        raise ValueError(f"{name} must contain exactly two values")
    lower, upper = tuple(float(item) for item in values)
    if not math.isfinite(lower) or not math.isfinite(upper):
        raise ValueError(f"{name} values must be finite")
    if not 0.0 < lower <= upper < 1.0:
        raise ValueError(f"{name} must satisfy 0 < lower <= upper < 1")
    return lower, upper


def _validate_seed(seed: Sequence[int]) -> tuple[int, int]:
    values = tuple(int(item) for item in seed)
    if len(values) != 2:
        raise ValueError("seed must contain exactly two integers")
    return values


def _validate_step_repair_multiplier(value: Any, *, name: str) -> float:
    multiplier = float(value)
    if not math.isfinite(multiplier) or multiplier <= 1.0:
        raise ValueError(f"{name} must be finite and greater than 1")
    return multiplier


def _string_tuple(values: Sequence[str] | str) -> tuple[str, ...]:
    if isinstance(values, str):
        raw = (values,)
    else:
        raw = tuple(values)
    return tuple(str(item) for item in raw if str(item))


def _mass_artifact_signature(mass_artifact: PrecomputedMassArtifact) -> str:
    return mass_artifact_signature(mass_artifact)


def _round_seed(base: tuple[int, int], index: int) -> tuple[int, int]:
    return int(base[0]), int(base[1]) + int(index)


def _scalar_or_none(value: Any) -> float | None:
    from bayesfilter.inference.hmc_budget_ladder import _last_metadata_entry

    present, scalar = _last_metadata_entry(value)
    if not present:
        return None
    try:
        return float(scalar)
    except (TypeError, ValueError):
        return None


def _seed_from_mapping(mapping: Mapping[str, Any], key: str) -> tuple[int, int] | None:
    value = mapping.get(key)
    if value is None:
        return None
    return _validate_seed(value)


def _bool_or_none(value: Any) -> bool | None:
    from bayesfilter.inference.hmc_budget_ladder import _last_metadata_entry

    present, scalar = _last_metadata_entry(value)
    return bool(scalar) if present else None


def _int_or_none(value: Any) -> int | None:
    scalar = _scalar_or_none(value)
    return None if scalar is None else int(scalar)


def _json_ready(value: Any) -> Any:
    if hasattr(value, "numpy"):
        value = value.numpy()
    if hasattr(value, "tolist"):
        value = value.tolist()
    elif hasattr(value, "item"):
        value = value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    return value


def _runtime_seconds_or_none(metadata: Mapping[str, Any]) -> float | None:
    for key in ("sample_chain_call_s", "first_call_s", "warm_call_s"):
        scalar = _scalar_or_none(metadata.get(key))
        if scalar is not None:
            return scalar
    return None


def _telemetry_payload(value: Any) -> Mapping[str, Any] | None:
    if value is None:
        return None
    payload = {key: _json_ready(item) for key, item in dict(value).items()}
    if "telemetry_failure_veto" in value:
        payload["telemetry_failure_veto_bool"] = bool(
            _bool_or_none(value["telemetry_failure_veto"])
        )
    return payload


def _finite_number(value: Any) -> bool:
    scalar = _scalar_or_none(value)
    return scalar is not None and math.isfinite(scalar)
