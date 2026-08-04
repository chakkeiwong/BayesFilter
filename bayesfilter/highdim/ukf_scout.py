"""UKF scout metadata and adapters for fixed-rank filtering design."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import tensorflow as tf

from bayesfilter.highdim.models import SpatialSIRSSM, p30_spatial_sir_fixture_model


P52_UKF_SCOUT_CLAIM = "scout_not_truth"


@dataclass(frozen=True)
class UKFScoutConfig:
    """Configuration for deterministic UKF scouting.

    The scout proposes centers, scales, and covariance summaries.  It is not a
    correctness oracle, exact likelihood, filtering-validation result, or HMC
    readiness result.
    """

    horizon: int = 1
    alpha: float = 1.0
    beta: float = 2.0
    kappa: float = 0.0
    jitter: float = 1e-9
    rank_tolerance: float = 1e-8
    claim_class: str = P52_UKF_SCOUT_CLAIM

    def __post_init__(self) -> None:
        if int(self.horizon) < 0:
            raise ValueError("horizon must be nonnegative")
        for name in ("alpha", "beta", "jitter", "rank_tolerance"):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if self.claim_class != P52_UKF_SCOUT_CLAIM:
            raise ValueError("UKF scout cannot promote stronger claims")
        object.__setattr__(self, "horizon", int(self.horizon))
        object.__setattr__(self, "alpha", float(self.alpha))
        object.__setattr__(self, "beta", float(self.beta))
        object.__setattr__(self, "kappa", float(self.kappa))
        object.__setattr__(self, "jitter", float(self.jitter))
        object.__setattr__(self, "rank_tolerance", float(self.rank_tolerance))


@dataclass(frozen=True)
class UKFScoutResult:
    """Finite UKF scout paths and summaries for one state-space target."""

    dimension: int
    compartments: int
    horizon: int
    sigma_point_count: int
    mean_path: tf.Tensor
    covariance_path: tf.Tensor
    scale_path: tf.Tensor
    covariance_eigenvalues: tf.Tensor
    effective_dimension_path: tf.Tensor
    max_abs_correlation_path: tf.Tensor
    process_covariance_shape: tuple[int, int]
    process_covariance_diagonal_range: tuple[float, float]
    observation_covariance_shape: tuple[int, int]
    observation_covariance_diagonal_range: tuple[float, float]
    initial_covariance_shape: tuple[int, int]
    initial_covariance_diagonal_range: tuple[float, float]
    status: str
    claim_class: str
    nonclaims: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.status != "PASS_P52_UKF_SCOUT":
            raise ValueError("unknown UKF scout status")
        if self.claim_class != P52_UKF_SCOUT_CLAIM:
            raise ValueError("UKF scout result cannot promote stronger claims")
        required = {
            "scout_not_truth",
            "no filtering correctness",
            "no exact likelihood",
            "no HMC readiness",
        }
        if not required <= set(self.nonclaims):
            raise ValueError("missing UKF scout nonclaims")

    def manifest_payload(self) -> Mapping[str, object]:
        final_center = self.mean_path[-1]
        final_scale = self.scale_path[-1]
        final_eigenvalues = self.covariance_eigenvalues[-1]
        return {
            "dimension": self.dimension,
            "compartments": self.compartments,
            "horizon": self.horizon,
            "sigma_point_count": self.sigma_point_count,
            "status": self.status,
            "claim_class": self.claim_class,
            "covariance_choices": {
                "source": "supplied SpatialSIRSSM covariance matrices",
                "process_covariance_shape": self.process_covariance_shape,
                "process_covariance_diagonal_range": (
                    self.process_covariance_diagonal_range
                ),
                "observation_covariance_shape": self.observation_covariance_shape,
                "observation_covariance_diagonal_range": (
                    self.observation_covariance_diagonal_range
                ),
                "initial_covariance_shape": self.initial_covariance_shape,
                "initial_covariance_diagonal_range": (
                    self.initial_covariance_diagonal_range
                ),
            },
            "final_center_dimension": int(final_center.shape[0]),
            "final_center_head": _tensor_list(final_center[: min(4, self.dimension)]),
            "final_center_tail": _tensor_list(final_center[-min(4, self.dimension) :]),
            "final_scale_dimension": int(final_scale.shape[0]),
            "final_scale_range": [
                float(tf.reduce_min(final_scale).numpy()),
                float(tf.reduce_max(final_scale).numpy()),
            ],
            "final_covariance_spectrum_dimension": int(final_eigenvalues.shape[0]),
            "final_covariance_eigenvalue_range": [
                float(tf.reduce_min(final_eigenvalues).numpy()),
                float(tf.reduce_max(final_eigenvalues).numpy()),
            ],
            "effective_dimension_path": _tensor_list(self.effective_dimension_path),
            "max_abs_correlation_path": _tensor_list(self.max_abs_correlation_path),
            "finite": bool(
                tf.reduce_all(tf.math.is_finite(self.mean_path)).numpy()
                and tf.reduce_all(tf.math.is_finite(self.covariance_path)).numpy()
                and tf.reduce_all(tf.math.is_finite(self.scale_path)).numpy()
            ),
            "lower_rung_sanity_comparator": _lower_rung_sanity_comparator(
                self.dimension
            ),
            "nonclaims": self.nonclaims,
        }


def ukf_scout_result_from_paths(
    mean_path: tf.Tensor,
    covariance_path: tf.Tensor,
    *,
    sigma_point_count: int,
    source: str,
) -> UKFScoutResult:
    """Package moments from an existing UKF route as scout metadata.

    This adapter does not implement another filter. It preserves the scout-only
    claim class while making target-specific UKF paths reusable by initializers.
    """

    means = tf.convert_to_tensor(mean_path, dtype=tf.float64)
    covariances = tf.convert_to_tensor(covariance_path, dtype=tf.float64)
    if means.shape.rank != 2 or covariances.shape.rank != 3:
        raise ValueError("UKF paths must have shapes [time, state] and [time, state, state]")
    time_count = int(means.shape[0])
    state_dim = int(means.shape[1])
    if covariances.shape != (time_count, state_dim, state_dim):
        raise ValueError("UKF covariance path shape does not match mean path")
    if time_count < 1 or int(sigma_point_count) < 1:
        raise ValueError("UKF paths require at least one time point and sigma point")
    if not bool(
        tf.reduce_all(tf.math.is_finite(means)).numpy()
        and tf.reduce_all(tf.math.is_finite(covariances)).numpy()
    ):
        raise ValueError("UKF paths must be finite")
    covariances = 0.5 * (covariances + tf.linalg.matrix_transpose(covariances))
    eigenvalues = tf.linalg.eigvalsh(covariances)
    scales = tf.sqrt(tf.maximum(tf.linalg.diag_part(covariances), 1e-15))
    max_eigen = tf.reduce_max(eigenvalues, axis=1)
    effective_dimension = tf.reduce_sum(
        tf.cast(eigenvalues > tf.maximum(1e-12, max_eigen[:, tf.newaxis] * 1e-8), tf.int32),
        axis=1,
    )
    correlations = covariances / (
        scales[:, :, tf.newaxis] * scales[:, tf.newaxis, :]
    )
    off_diagonal = correlations - tf.eye(state_dim, dtype=tf.float64)[tf.newaxis, :, :]
    return UKFScoutResult(
        dimension=state_dim,
        compartments=1,
        horizon=time_count - 1,
        sigma_point_count=int(sigma_point_count),
        mean_path=means,
        covariance_path=covariances,
        scale_path=scales,
        covariance_eigenvalues=eigenvalues,
        effective_dimension_path=effective_dimension,
        max_abs_correlation_path=tf.reduce_max(tf.abs(off_diagonal), axis=[1, 2]),
        process_covariance_shape=(state_dim, state_dim),
        process_covariance_diagonal_range=(0.0, 0.0),
        observation_covariance_shape=(1, 1),
        observation_covariance_diagonal_range=(0.0, 0.0),
        initial_covariance_shape=(state_dim, state_dim),
        initial_covariance_diagonal_range=(
            float(tf.reduce_min(tf.linalg.diag_part(covariances[0])).numpy()),
            float(tf.reduce_max(tf.linalg.diag_part(covariances[0])).numpy()),
        ),
        status="PASS_P52_UKF_SCOUT",
        claim_class=P52_UKF_SCOUT_CLAIM,
        nonclaims=(
            "scout_not_truth",
            "no filtering correctness",
            "no exact likelihood",
            "no HMC readiness",
            f"source_route={source}",
        ),
    )


def spatial_sir_ukf_scout(
    model: SpatialSIRSSM,
    *,
    config: UKFScoutConfig | None = None,
    observations: tf.Tensor | None = None,
) -> UKFScoutResult:
    """Run a deterministic UKF scout for a spatial SIR fixture."""

    cfg = UKFScoutConfig() if config is None else config
    state_dim = model.state_dim()
    obs_dim = model.observation_dim()
    if observations is None:
        observation_path = _nominal_observations(model, cfg.horizon)
    else:
        observation_path = tf.convert_to_tensor(observations, dtype=tf.float64)
    if observation_path.shape != (cfg.horizon + 1, obs_dim):
        raise ValueError("observations must have shape [horizon + 1, observation_dim]")

    mean = tf.convert_to_tensor(model.initial_mean, dtype=tf.float64)
    covariance = _symmetrize(tf.convert_to_tensor(model.initial_covariance, dtype=tf.float64))
    means = []
    covariances = []

    for time_index in range(cfg.horizon + 1):
        if time_index > 0:
            points, mean_weights, covariance_weights = _unscented_sigma_points(
                mean,
                covariance,
                cfg,
            )
            propagated = model.transition_mean(points)
            mean = _weighted_mean(propagated, mean_weights)
            covariance = (
                _weighted_covariance(propagated, mean, covariance_weights)
                + model.process_covariance
            )
            covariance = _stabilize_covariance(covariance, cfg.jitter)

        points, mean_weights, covariance_weights = _unscented_sigma_points(
            mean,
            covariance,
            cfg,
        )
        observed_points = model.infectious_components(points)
        observation_mean = _weighted_mean(observed_points, mean_weights)
        innovation_covariance = (
            _weighted_covariance(observed_points, observation_mean, covariance_weights)
            + model.observation_covariance
        )
        centered_state = points - mean[tf.newaxis, :]
        centered_obs = observed_points - observation_mean[tf.newaxis, :]
        cross_covariance = tf.einsum(
            "n,ni,nj->ij",
            covariance_weights,
            centered_state,
            centered_obs,
        )
        innovation_covariance = _stabilize_covariance(
            innovation_covariance,
            cfg.jitter,
        )
        gain = tf.transpose(
            tf.linalg.solve(
                innovation_covariance,
                tf.transpose(cross_covariance),
            )
        )
        residual = observation_path[time_index] - observation_mean
        mean = mean + tf.linalg.matvec(gain, residual)
        covariance = covariance - gain @ innovation_covariance @ tf.transpose(gain)
        covariance = _stabilize_covariance(covariance, cfg.jitter)
        means.append(mean)
        covariances.append(covariance)

    mean_path = tf.stack(means)
    covariance_path = tf.stack(covariances)
    scale_path = tf.sqrt(
        tf.maximum(
            tf.linalg.diag_part(covariance_path),
            tf.constant(cfg.jitter, dtype=tf.float64),
        )
    )
    eigenvalues = tf.linalg.eigvalsh(covariance_path)
    max_eigen = tf.reduce_max(eigenvalues, axis=1)
    threshold = tf.maximum(
        tf.constant(cfg.rank_tolerance, dtype=tf.float64),
        max_eigen * tf.constant(cfg.rank_tolerance, dtype=tf.float64),
    )
    effective_dimension = tf.reduce_sum(
        tf.cast(eigenvalues > threshold[:, tf.newaxis], tf.int32),
        axis=1,
    )
    max_abs_correlation = _max_abs_correlation_path(covariance_path, scale_path)
    if not bool(
        tf.reduce_all(tf.math.is_finite(mean_path)).numpy()
        and tf.reduce_all(tf.math.is_finite(covariance_path)).numpy()
        and tf.reduce_all(tf.math.is_finite(scale_path)).numpy()
        and tf.reduce_all(tf.math.is_finite(eigenvalues)).numpy()
    ):
        raise ValueError("UKF scout produced nonfinite diagnostics")
    return UKFScoutResult(
        dimension=state_dim,
        compartments=obs_dim,
        horizon=cfg.horizon,
        sigma_point_count=2 * state_dim + 1,
        mean_path=mean_path,
        covariance_path=covariance_path,
        scale_path=scale_path,
        covariance_eigenvalues=eigenvalues,
        effective_dimension_path=effective_dimension,
        max_abs_correlation_path=max_abs_correlation,
        process_covariance_shape=_matrix_shape(model.process_covariance),
        process_covariance_diagonal_range=_diagonal_range(model.process_covariance),
        observation_covariance_shape=_matrix_shape(model.observation_covariance),
        observation_covariance_diagonal_range=_diagonal_range(
            model.observation_covariance
        ),
        initial_covariance_shape=_matrix_shape(model.initial_covariance),
        initial_covariance_diagonal_range=_diagonal_range(model.initial_covariance),
        status="PASS_P52_UKF_SCOUT",
        claim_class=P52_UKF_SCOUT_CLAIM,
        nonclaims=(
            "scout_not_truth",
            "no filtering correctness",
            "no exact likelihood",
            "no HMC readiness",
        ),
    )


def p52_spatial_sir_ukf_scout_manifest(
    *,
    dimensions: Sequence[int] = (18, 50, 100),
    horizon: int = 1,
) -> Mapping[str, object]:
    """Build a summary manifest for P52-M3 spatial SIR UKF scouts."""

    rows = []
    for dimension in dimensions:
        if int(dimension) % 2:
            raise ValueError("spatial SIR state dimension must be even")
        model = p30_spatial_sir_fixture_model(int(dimension) // 2)
        result = spatial_sir_ukf_scout(model, config=UKFScoutConfig(horizon=horizon))
        rows.append(dict(result.manifest_payload()))
    return {
        "schema_version": "p52.ukf_scout.v1",
        "phase": "P52-M3",
        "status": "PASS_P52_M3_UKF_SCOUTING",
        "claim_class": P52_UKF_SCOUT_CLAIM,
        "dimensions": tuple(int(value) for value in dimensions),
        "rows": rows,
        "nonclaims": (
            "scout_not_truth",
            "no filtering correctness",
            "no exact likelihood",
            "no HMC readiness",
            "no production spatial SIR readiness",
            "no d=100 filtering correctness",
        ),
    }


def _nominal_observations(model: SpatialSIRSSM, horizon: int) -> tf.Tensor:
    state = tf.convert_to_tensor(model.initial_mean, dtype=tf.float64)
    observations = [model.infectious_components(state)[0]]
    for _ in range(int(horizon)):
        state = model.transition_mean(state)[0]
        observations.append(model.infectious_components(state)[0])
    return tf.stack(observations)


def _unscented_sigma_points(
    mean: tf.Tensor,
    covariance: tf.Tensor,
    config: UKFScoutConfig,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    mean = tf.convert_to_tensor(mean, dtype=tf.float64)
    covariance = _stabilize_covariance(covariance, config.jitter)
    dim = int(mean.shape[0])
    spread = config.alpha * config.alpha * (float(dim) + config.kappa)
    if spread <= 0.0:
        raise ValueError("alpha**2 * (dim + kappa) must be positive")
    lambda_value = spread - float(dim)
    eigenvalues, eigenvectors = tf.linalg.eigh(covariance)
    eigenvalues = tf.maximum(eigenvalues, tf.constant(config.jitter, dtype=tf.float64))
    factor = eigenvectors @ tf.linalg.diag(tf.sqrt(eigenvalues))
    eye = tf.eye(dim, dtype=tf.float64)
    offsets = tf.concat(
        [
            tf.zeros([1, dim], dtype=tf.float64),
            tf.sqrt(tf.constant(spread, dtype=tf.float64)) * eye,
            -tf.sqrt(tf.constant(spread, dtype=tf.float64)) * eye,
        ],
        axis=0,
    )
    points = mean[tf.newaxis, :] + offsets @ tf.transpose(factor)
    axis_weight = tf.constant(1.0 / (2.0 * spread), dtype=tf.float64)
    mean_weights = tf.concat(
        [
            tf.constant([lambda_value / spread], dtype=tf.float64),
            tf.fill([2 * dim], axis_weight),
        ],
        axis=0,
    )
    covariance_weights = tf.concat(
        [
            tf.constant(
                [lambda_value / spread + (1.0 - config.alpha**2 + config.beta)],
                dtype=tf.float64,
            ),
            tf.fill([2 * dim], axis_weight),
        ],
        axis=0,
    )
    return points, mean_weights, covariance_weights


def _weighted_mean(points: tf.Tensor, weights: tf.Tensor) -> tf.Tensor:
    return tf.reduce_sum(points * weights[:, tf.newaxis], axis=0)


def _weighted_covariance(points: tf.Tensor, mean: tf.Tensor, weights: tf.Tensor) -> tf.Tensor:
    centered = points - mean[tf.newaxis, :]
    return _symmetrize(tf.einsum("n,ni,nj->ij", weights, centered, centered))


def _stabilize_covariance(covariance: tf.Tensor, jitter: float) -> tf.Tensor:
    matrix = _symmetrize(tf.convert_to_tensor(covariance, dtype=tf.float64))
    dim = int(matrix.shape[0])
    return matrix + tf.constant(float(jitter), dtype=tf.float64) * tf.eye(
        dim,
        dtype=tf.float64,
    )


def _symmetrize(matrix: tf.Tensor) -> tf.Tensor:
    return 0.5 * (matrix + tf.transpose(matrix))


def _max_abs_correlation_path(covariance_path: tf.Tensor, scale_path: tf.Tensor) -> tf.Tensor:
    correlations = covariance_path / (
        scale_path[:, :, tf.newaxis] * scale_path[:, tf.newaxis, :]
    )
    dim = int(covariance_path.shape[-1])
    off_diag = correlations - tf.eye(dim, dtype=tf.float64)[tf.newaxis, :, :]
    return tf.reduce_max(tf.abs(off_diag), axis=[1, 2])


def _tensor_list(tensor: tf.Tensor) -> object:
    value = tf.convert_to_tensor(tensor)
    return value.numpy().tolist()


def _matrix_shape(matrix: tf.Tensor) -> tuple[int, int]:
    tensor = tf.convert_to_tensor(matrix, dtype=tf.float64)
    if tensor.shape.rank != 2:
        raise ValueError("covariance matrix must have rank 2")
    return (int(tensor.shape[0]), int(tensor.shape[1]))


def _diagonal_range(matrix: tf.Tensor) -> tuple[float, float]:
    diagonal = tf.linalg.diag_part(tf.convert_to_tensor(matrix, dtype=tf.float64))
    return (
        float(tf.reduce_min(diagonal).numpy()),
        float(tf.reduce_max(diagonal).numpy()),
    )


def _lower_rung_sanity_comparator(dimension: int) -> Mapping[str, object]:
    if int(dimension) < 18:
        return {
            "status": "not_required_for_subpaper_dimension",
            "claim_class": "none",
        }
    return {
        "status": "recorded_sanity_only_not_promotion",
        "claim_class": "lower_rung_value_moment_sanity_only",
        "source_artifacts": (
            "docs/plans/bayesfilter-highdim-zhao-cui-p50-m6-spatial-sir-predator-prey-ladder-manifest-2026-06-09.json",
            "docs/plans/bayesfilter-highdim-zhao-cui-p47-spatial-sir-filtering-target-manifest-2026-06-08.json",
        ),
        "source_row": "spatial_sir_j1_zhaocui_vs_dense_lower_rung",
        "source_dimension": 2,
        "source_evidence": "Zhao-Cui lower-rung value/moment diagnostic against dense reference",
        "nonclaims": (
            "not a d=18 dense reference",
            "not UKF correctness",
            "not production spatial SIR readiness",
            "not HMC readiness",
        ),
    }
