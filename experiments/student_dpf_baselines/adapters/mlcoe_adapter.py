"""Adapter for the vendored `2026MLCOE` student snapshot."""

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


IMPLEMENTATION_NAME = "2026MLCOE"
SOURCE_COMMIT = "020cfd7f2f848afa68432e95e6c6e747d3d2402d"
SNAPSHOT_ROOT = Path(__file__).resolve().parents[1] / "vendor" / "2026MLCOE"


def run_smoke_fixture(
    fixture: Any,
    *,
    seed: int,
    num_particles: int | None,
) -> BaselineResult:
    """Run a small linear-Gaussian fixture through the MLCOE KF path."""

    del num_particles
    start = time.perf_counter()
    try:
        with prepend_sys_path(SNAPSHOT_ROOT):
            import tensorflow as tf  # type: ignore
            from src.filters.classical import DTYPE, KF  # type: ignore

            tf.random.set_seed(seed)
            x0 = tf.constant(fixture.m0[:, None], dtype=DTYPE)
            p0 = tf.constant(fixture.P0, dtype=DTYPE)
            filt = KF(
                tf.constant(fixture.A, dtype=DTYPE),
                tf.constant(fixture.C, dtype=DTYPE),
                tf.constant(fixture.Q, dtype=DTYPE),
                tf.constant(fixture.R, dtype=DTYPE),
                p0,
                x0,
            )

            means = [fixture.m0.astype(float)]
            covariances = [fixture.P0.astype(float)]
            log_likelihoods = []
            for obs in fixture.observations:
                x_pred, p_pred = filt.predict()
                log_likelihoods.append(
                    _gaussian_predictive_log_likelihood(
                        np.asarray(x_pred.numpy(), dtype=float).reshape(-1),
                        np.asarray(p_pred.numpy(), dtype=float),
                        obs,
                        fixture.C,
                        fixture.R,
                    )
                )
                x_upd, p_upd = filt.update(
                    tf.constant(obs[:, None], dtype=DTYPE)
                )
                means.append(np.asarray(x_upd.numpy(), dtype=float).reshape(-1))
                covariances.append(np.asarray(p_upd.numpy(), dtype=float))

            return BaselineResult(
                implementation_name=IMPLEMENTATION_NAME,
                source_commit=SOURCE_COMMIT,
                fixture_name=fixture.name,
                seed=seed,
                num_particles=None,
                dtype="float64",
                status=BaselineStatus.OK,
                log_likelihood=float(np.sum(log_likelihoods)),
                filtered_means=np.asarray(means),
                filtered_covariances=np.asarray(covariances),
                runtime_seconds=time.perf_counter() - start,
                gradient_available=False,
                diagnostics={
                    "adapter_target": "src.filters.classical.KF",
                    "log_likelihood_source": "adapter_predictive_gaussian",
                },
            )
    except Exception as exc:  # pragma: no cover - exercised by runners.
        return exception_result(
            implementation_name=IMPLEMENTATION_NAME,
            source_commit=SOURCE_COMMIT,
            fixture_name=getattr(fixture, "name", "unknown"),
            exc=exc,
            seed=seed,
            num_particles=None,
            dtype="float64",
            runtime_seconds=time.perf_counter() - start,
        )


def run_bpf_fixture(
    fixture: Any,
    *,
    seed: int,
    num_particles: int,
) -> BaselineResult:
    """Run a linear-Gaussian fixture through the MLCOE BPF path."""

    start = time.perf_counter()
    try:
        with prepend_sys_path(SNAPSHOT_ROOT):
            import tensorflow as tf  # type: ignore
            from src.filters.particle import BPF, DTYPE  # type: ignore

            np.random.seed(seed)
            tf.random.set_seed(seed)

            model = _LinearGaussianBpfModel(fixture, tf=tf, dtype=DTYPE)
            filt = BPF(model, N=num_particles)
            filt.init(
                tf.constant(fixture.m0, dtype=DTYPE),
                tf.constant(fixture.P0, dtype=DTYPE),
            )

            particle_means = []
            particle_covariances = []
            ess_by_time = []
            threshold_resampled = []

            _append_particle_summary(
                filt,
                particle_means=particle_means,
                particle_covariances=particle_covariances,
            )

            for obs in fixture.observations:
                filt.step(tf.constant(obs, dtype=DTYPE))
                ess = float(filt.ess.numpy())
                ess_by_time.append(ess)
                threshold_resampled.append(ess < num_particles * 0.1)
                _append_particle_summary(
                    filt,
                    particle_means=particle_means,
                    particle_covariances=particle_covariances,
                )

            return BaselineResult(
                implementation_name=IMPLEMENTATION_NAME,
                source_commit=SOURCE_COMMIT,
                fixture_name=fixture.name,
                seed=seed,
                num_particles=num_particles,
                dtype="float64",
                status=BaselineStatus.OK,
                log_likelihood=None,
                likelihood_surrogate=None,
                particle_means=np.asarray(particle_means, dtype=float),
                particle_covariances=np.asarray(particle_covariances, dtype=float),
                ess_by_time=np.asarray(ess_by_time, dtype=float),
                resampling_count=int(np.sum(threshold_resampled)),
                runtime_seconds=time.perf_counter() - start,
                gradient_available=False,
                diagnostics={
                    "adapter_target": "src.filters.particle.BPF",
                    "log_likelihood_status": "not_exposed_by_mlcoe_bpf",
                    "ess_semantics": "pre_resampling_step_ess",
                    "resampling_count_semantics": "threshold_inferred_ess_lt_0.1N",
                    "resampling_threshold": 0.1,
                    "threshold_resampled": threshold_resampled,
                },
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


def _gaussian_predictive_log_likelihood(
    x_pred: np.ndarray,
    p_pred: np.ndarray,
    obs: np.ndarray,
    C: np.ndarray,
    R: np.ndarray,
) -> float:
    mean_y = C @ x_pred
    innovation = obs - mean_y
    S = C @ p_pred @ C.T + R
    sign, logdet = np.linalg.slogdet(S)
    if sign <= 0:
        raise np.linalg.LinAlgError("predictive covariance is not positive definite")
    quad = float(innovation.T @ np.linalg.solve(S, innovation))
    ny = obs.shape[0]
    return -0.5 * (ny * np.log(2.0 * np.pi) + logdet + quad)


class _LinearGaussianBpfModel:
    """Adapter-local TensorFlow bridge for MLCOE's BPF model contract."""

    def __init__(self, fixture: Any, *, tf: Any, dtype: Any) -> None:
        self.state_dim = fixture.state_dim
        self.obs_dim = fixture.obs_dim
        self.F = tf.constant(fixture.A, dtype=dtype)
        self.Q_filter = tf.constant(fixture.Q, dtype=dtype)
        self.R_filter = tf.constant(fixture.R, dtype=dtype)
        self.R_inv_filter = tf.linalg.inv(self.R_filter)
        self._H = tf.constant(fixture.C, dtype=dtype)

    def h_func(self, x: Any) -> Any:
        return self._tf_matmul_observation(x)

    def _tf_matmul_observation(self, x: Any) -> Any:
        import tensorflow as tf  # type: ignore

        if len(x.shape) == 1:
            return tf.linalg.matvec(self._H, x)
        return tf.matmul(x, self._H, transpose_b=True)


def _append_particle_summary(
    filt: Any,
    *,
    particle_means: list[np.ndarray],
    particle_covariances: list[np.ndarray],
) -> None:
    particles = np.asarray(filt.X.numpy(), dtype=float)
    weights = np.asarray(filt.W.numpy(), dtype=float)
    weights = weights / np.sum(weights)
    mean = np.sum(particles * weights[:, None], axis=0)
    diff = particles - mean
    covariance = diff.T @ (diff * weights[:, None])
    covariance = 0.5 * (covariance + covariance.T)
    particle_means.append(mean)
    particle_covariances.append(covariance)
