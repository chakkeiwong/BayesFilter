"""Bind an accepted local covariance to the ordinary public HMC tuner.

The executable ``main`` uses stubs, so this listing can be checked without
estimating a covariance or launching HMC.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from bayesfilter.inference import (
    HMCKernelTuningConfig,
    estimate_sequential_map_covariance,
    tune_hmc_kernel,
)


def tune_from_fresh_local_covariance(
    *,
    adapter: Any,
    value_and_score_fn: Callable[[Any], tuple[Any, Any]],
    initial_positions: Sequence[Any],
    config: HMCKernelTuningConfig,
    covariance_kwargs: Mapping[str, Any] | None = None,
    covariance_estimator: Callable[..., Any] = estimate_sequential_map_covariance,
    tuner: Callable[..., Any] = tune_hmc_kernel,
) -> Any:
    """Estimate, accept, and bind one center/covariance pair in one coordinate system."""

    covariance_result = covariance_estimator(
        value_and_score_fn,
        initial_positions,
        **dict(covariance_kwargs or {}),
    )
    if not covariance_result.accepted:
        raise RuntimeError("local covariance was not accepted")
    if covariance_result.map_candidate is None or covariance_result.covariance is None:
        raise RuntimeError("accepted covariance result is incomplete")
    return tuner(
        adapter=adapter,
        initial_position=covariance_result.map_candidate,
        initial_covariance=covariance_result.covariance,
        config=config,
    )


def main() -> Mapping[str, Any]:
    """Exercise only the covariance-to-tuner argument binding."""

    accepted = SimpleNamespace(
        accepted=True,
        map_candidate=(0.25, -0.25),
        covariance=((2.0, 0.1), (0.1, 0.75)),
    )
    observed: dict[str, Any] = {}

    def covariance_stub(*_args: Any, **_kwargs: Any) -> Any:
        return accepted

    def tuner_stub(**kwargs: Any) -> Mapping[str, Any]:
        observed.update(kwargs)
        return {"status": "arguments_bound_without_hmc"}

    result = tune_from_fresh_local_covariance(
        adapter=object(),
        value_and_score_fn=lambda position: (position, position),
        initial_positions=((0.0, 0.0),),
        config=HMCKernelTuningConfig.smoke(
            target_scope="docs_covariance_first_binding"
        ),
        covariance_estimator=covariance_stub,
        tuner=tuner_stub,
    )
    assert observed["initial_position"] == accepted.map_candidate
    assert observed["initial_covariance"] == accepted.covariance
    assert result["status"] == "arguments_bound_without_hmc"
    return result


if __name__ == "__main__":
    print(main()["status"])
