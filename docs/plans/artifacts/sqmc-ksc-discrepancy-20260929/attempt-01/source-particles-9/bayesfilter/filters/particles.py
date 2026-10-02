"""Structural bootstrap particle-filter reference backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np

from bayesfilter.structural import (
    CompletionPolicy,
    FilterRunMetadata,
    StructuralFilterConfig,
    validate_filter_config,
)


ProposalSpace = Literal["innovation"]
ProposalCorrection = Literal["model_exact", "declared_approximation"]


class ParticleFilterNotAuditedError(NotImplementedError):
    """Compatibility error for older fail-closed particle-filter callers."""


@dataclass(frozen=True)
class ParticleFilterConfig:
    """Configuration for the audited structural bootstrap particle filter."""

    num_particles: int = 2048
    proposal_space: ProposalSpace = "innovation"
    deterministic_completion: CompletionPolicy = "required"
    resampling: Literal["systematic", "multinomial"] = "systematic"
    random_seed: int | None = 0
    deterministic_noise_scale: float = 0.0
    approximation_label: str | None = None
    proposal_correction: ProposalCorrection = "model_exact"

    def __post_init__(self) -> None:
        object.__setattr__(self, "num_particles", int(self.num_particles))
        object.__setattr__(
            self,
            "deterministic_noise_scale",
            float(self.deterministic_noise_scale),
        )
        if self.num_particles <= 0:
            raise ValueError("num_particles must be positive")
        if self.proposal_space != "innovation":
            raise ValueError("only innovation proposal_space is currently audited")
        if self.deterministic_completion not in {"required", "none", "approximate"}:
            raise ValueError(
                "deterministic_completion must be required, none, or approximate"
            )
        if self.resampling not in {"systematic", "multinomial"}:
            raise ValueError("resampling must be systematic or multinomial")
        if self.proposal_correction not in {"model_exact", "declared_approximation"}:
            raise ValueError(
                "proposal_correction must be model_exact or declared_approximation"
            )
        if self.approximation_label is not None and not self.approximation_label.strip():
            raise ValueError("approximation_label must be nonempty when supplied")
        if self.deterministic_noise_scale < 0.0:
            raise ValueError("deterministic_noise_scale must be nonnegative")
        if self.deterministic_noise_scale > 0.0 and (
            self.approximation_label is None
            or self.proposal_correction != "declared_approximation"
        ):
            raise ValueError(
                "deterministic-coordinate noise requires an approximation_label "
                "and declared_approximation correction policy"
            )


@dataclass(frozen=True)
class ParticleFilterResult:
    log_likelihood: float
    final_particles: np.ndarray | None
    metadata: FilterRunMetadata
    diagnostics: dict[str, float | int | str]


def _as_observation_matrix(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        return arr[:, None]
    return arr


def _symmetrize(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (matrix + matrix.T)


def _gaussian_logpdf_by_particle(
    values: np.ndarray,
    mean_by_particle: np.ndarray,
    covariance: np.ndarray,
) -> np.ndarray:
    covariance = _symmetrize(np.asarray(covariance, dtype=float))
    chol = np.linalg.cholesky(covariance)
    centered = np.asarray(values, dtype=float)[None, :] - mean_by_particle
    whitened = np.linalg.solve(chol, centered.T).T
    log_det = 2.0 * float(np.sum(np.log(np.diag(chol))))
    quad = np.sum(whitened * whitened, axis=1)
    dim = int(centered.shape[1])
    return -0.5 * (dim * np.log(2.0 * np.pi) + log_det + quad)


def _normalize_log_weights(log_weights: np.ndarray) -> tuple[float, np.ndarray]:
    max_log = float(np.max(log_weights))
    shifted = np.exp(log_weights - max_log)
    mean_weight = float(np.mean(shifted))
    if not np.isfinite(mean_weight) or mean_weight <= 0.0:
        raise FloatingPointError("particle likelihood weights are degenerate")
    normalized = shifted / float(np.sum(shifted))
    return max_log + np.log(mean_weight), normalized


def _systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    count = int(weights.shape[0])
    positions = (rng.random() + np.arange(count)) / count
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0
    return np.searchsorted(cumulative, positions, side="right")


def _resample(
    particles: np.ndarray,
    weights: np.ndarray,
    rng: np.random.Generator,
    method: str,
) -> np.ndarray:
    if method == "systematic":
        indices = _systematic_resample(weights, rng)
    elif method == "multinomial":
        indices = rng.choice(weights.shape[0], size=weights.shape[0], p=weights)
    else:
        raise ValueError("unsupported resampling method")
    return particles[indices]


def particle_filter_log_likelihood(
    model,
    observations: np.ndarray,
    *,
    config: ParticleFilterConfig | None = None,
    mask: np.ndarray | None = None,
    theta=None,
    return_particles: bool = False,
    identity_diagnostic: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
) -> ParticleFilterResult:
    """Evaluate a structural bootstrap particle likelihood.

    Particles are sampled in the declared innovation space and then propagated
    through the model's structural transition.  Deterministic-completion
    coordinates are therefore model outputs, not independently noised states.
    """

    cfg = config or ParticleFilterConfig()
    partition = model.partition
    validate_filter_config(
        partition,
        StructuralFilterConfig(
            integration_space=cfg.proposal_space,
            deterministic_completion=cfg.deterministic_completion,
            approximation_label=cfg.approximation_label,
        ),
    )
    rng = np.random.default_rng(cfg.random_seed)
    observations = _as_observation_matrix(observations)
    observation_cov = np.asarray(model.observation_cov(theta), dtype=float)
    if observations.shape[1] != observation_cov.shape[0]:
        raise ValueError("observations and observation_cov dimensions do not match")
    obs_mask = np.isfinite(observations) if mask is None else np.asarray(mask, dtype=bool)
    if obs_mask.shape != observations.shape:
        raise ValueError("mask must have the same shape as observations")

    particles = rng.multivariate_normal(
        np.asarray(model.initial_mean(theta), dtype=float),
        np.asarray(model.initial_cov(theta), dtype=float),
        size=cfg.num_particles,
    )
    innovation_cov = np.asarray(model.innovation_cov(theta), dtype=float)
    log_likelihood = 0.0
    min_effective_sample_size = float(cfg.num_particles)
    max_identity_residual = 0.0
    resample_count = 0

    for row, row_mask in zip(observations, obs_mask, strict=True):
        previous_particles = particles.copy()
        innovations = rng.multivariate_normal(
            np.zeros(partition.innovation_dim),
            innovation_cov,
            size=cfg.num_particles,
        )
        particles = np.asarray(
            [
                model.transition(previous, innovation, theta)
                for previous, innovation in zip(previous_particles, innovations, strict=True)
            ],
            dtype=float,
        )
        if cfg.deterministic_noise_scale > 0.0 and partition.deterministic_indices:
            noise = rng.normal(
                scale=cfg.deterministic_noise_scale,
                size=(cfg.num_particles, len(partition.deterministic_indices)),
            )
            particles[:, partition.deterministic_indices] += noise
        if identity_diagnostic is not None:
            residuals = np.asarray(
                identity_diagnostic(previous_particles, particles),
                dtype=float,
            )
            if residuals.size:
                max_identity_residual = max(
                    max_identity_residual,
                    float(np.max(np.abs(residuals))),
                )

        if not np.any(row_mask):
            continue
        observed_points = _as_observation_matrix(model.observe(particles, theta))
        log_weights = _gaussian_logpdf_by_particle(
            row[row_mask],
            observed_points[:, row_mask],
            observation_cov[np.ix_(row_mask, row_mask)],
        )
        increment, weights = _normalize_log_weights(log_weights)
        log_likelihood += float(increment)
        ess = 1.0 / float(np.sum(weights * weights))
        min_effective_sample_size = min(min_effective_sample_size, ess)
        particles = _resample(particles, weights, rng, cfg.resampling)
        resample_count += 1

    metadata = FilterRunMetadata(
        filter_name="structural_bootstrap_particle",
        partition=partition,
        integration_space=cfg.proposal_space,
        deterministic_completion=cfg.deterministic_completion,
        approximation_label=cfg.approximation_label,
        differentiability_status="monte_carlo_value_only",
        compiled_status="eager_numpy",
    )
    diagnostics = {
        "num_particles": cfg.num_particles,
        "resample_count": resample_count,
        "min_effective_sample_size": min_effective_sample_size,
        "max_identity_residual": max_identity_residual,
        "proposal_correction": cfg.proposal_correction,
    }
    return ParticleFilterResult(
        log_likelihood=float(log_likelihood),
        final_particles=particles.copy() if return_particles else None,
        metadata=metadata,
        diagnostics=diagnostics,
    )
