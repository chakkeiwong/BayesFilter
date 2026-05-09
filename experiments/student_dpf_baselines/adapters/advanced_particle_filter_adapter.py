"""Adapter for the vendored `advanced_particle_filter` student snapshot."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any

import numpy as np

from .common import (
    BaselineResult,
    BaselineStatus,
    exception_result,
    prepend_sys_path,
)


IMPLEMENTATION_NAME = "advanced_particle_filter"
SOURCE_COMMIT = "d2a797c330e11befacbb736b5c86b8d03eb4a389"
VENDOR_ROOT = Path(__file__).resolve().parents[1] / "vendor"


def run_smoke_fixture(
    fixture: Any,
    *,
    seed: int,
    num_particles: int | None,
) -> BaselineResult:
    """Run a small linear-Gaussian fixture through the advanced snapshot."""

    start = time.perf_counter()
    try:
        with prepend_sys_path(VENDOR_ROOT):
            from advanced_particle_filter.filters import (  # type: ignore
                BootstrapParticleFilter,
                KalmanFilter,
            )
            from advanced_particle_filter.models import make_lgssm  # type: ignore

            model = make_lgssm(
                fixture.A,
                fixture.C,
                fixture.Q,
                fixture.R,
                fixture.m0,
                fixture.P0,
            )
            kf_result = KalmanFilter().filter(model, fixture.observations)

            diagnostics: dict[str, Any] = {
                "adapter_target": "KalmanFilter",
                "kalman_log_likelihood": float(kf_result.log_likelihood),
            }
            ess = None
            particle_means = None
            particle_covariances = None
            resampling_count = None

            if num_particles is not None:
                pf = BootstrapParticleFilter(
                    n_particles=num_particles,
                    seed=seed,
                )
                pf_result = pf.filter(
                    model,
                    fixture.observations,
                    return_particles=False,
                    rng=np.random.default_rng(seed),
                )
                ess = pf_result.ess
                particle_means = pf_result.means
                particle_covariances = pf_result.covariances
                resampling_count = (
                    int(np.sum(pf_result.resampled))
                    if pf_result.resampled is not None
                    else None
                )
                diagnostics.update(
                    {
                        "particle_filter_log_likelihood": (
                            float(pf_result.log_likelihood)
                            if pf_result.log_likelihood is not None
                            else None
                        ),
                        "particle_filter_average_ess": (
                            float(pf_result.average_ess())
                            if pf_result.ess is not None
                            else None
                        ),
                    }
                )

            return BaselineResult(
                implementation_name=IMPLEMENTATION_NAME,
                source_commit=SOURCE_COMMIT,
                fixture_name=fixture.name,
                seed=seed,
                num_particles=num_particles,
                dtype="float64",
                status=BaselineStatus.OK,
                log_likelihood=float(kf_result.log_likelihood),
                filtered_means=kf_result.means,
                filtered_covariances=kf_result.covariances,
                particle_means=particle_means,
                particle_covariances=particle_covariances,
                ess_by_time=ess,
                resampling_count=resampling_count,
                runtime_seconds=time.perf_counter() - start,
                gradient_available=False,
                diagnostics=diagnostics,
            )
    except Exception as exc:  # pragma: no cover - exercised by runners.
        return exception_result(
            implementation_name=IMPLEMENTATION_NAME,
            source_commit=SOURCE_COMMIT,
            fixture_name=getattr(fixture, "name", "unknown"),
            exc=exc,
            seed=seed,
            num_particles=num_particles,
            dtype="float64",
            runtime_seconds=time.perf_counter() - start,
        )
