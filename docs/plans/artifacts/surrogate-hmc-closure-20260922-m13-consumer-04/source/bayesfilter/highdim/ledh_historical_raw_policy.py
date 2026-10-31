"""Fail-closed policy helpers for historical raw-barycentric LEDH routes."""

from __future__ import annotations

from typing import Any


HISTORICAL_RAW_BARYCENTRIC_STATUS = (
    "historical_raw_barycentric_diagnostic_only"
)


def require_historical_raw_diagnostic_opt_in(
    args_or_flag: Any,
    *,
    route_name: str,
) -> str:
    """Require an explicit diagnostic opt-in before executing a raw route."""

    enabled = (
        args_or_flag
        if isinstance(args_or_flag, bool)
        else getattr(args_or_flag, "historical_raw_diagnostic", False)
    )
    if enabled is not True:
        raise ValueError(
            f"{route_name} is a historical raw-barycentric diagnostic; "
            "set --historical-raw-diagnostic explicitly"
        )
    return HISTORICAL_RAW_BARYCENTRIC_STATUS
