"""UKF-moment initializer for fixed-branch density training."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence

import tensorflow as tf

from bayesfilter.highdim.bases import ProductBasis
from bayesfilter.highdim.diagnostics import (
    HighDimStatus,
    MassMeasure,
    assert_density_matches_mass,
    assert_tf_float64,
)
from bayesfilter.highdim.tt import TTCore
from bayesfilter.highdim.ukf_scout import P52_UKF_SCOUT_CLAIM, UKFScoutResult


P76_UKF_INITIALIZER_RULE = "ukf_whitened_gaussian_sqrt_projection_v1"
P76_ROUTE_CLASSIFICATION = "extension_or_invention"
P76_SCHEMA_VERSION = "p76_ukf_initializer.v1"
P76_STATUS_OK = "P76_UKF_INITIALIZER_OK"
P76_NONCLAIMS = (
    "scout_not_truth",
    "not source-faithful Zhao-Cui",
    "not lower-gate repair evidence",
    "not validation evidence",
    "not HMC readiness evidence",
    "not large-pilot evidence",
)


@dataclass(frozen=True)
class P76UKFInitializerConfig:
    """Configuration for the P76 UKF square-root initializer."""

    product_basis: ProductBasis
    ranks: tuple[int, ...]
    time_index: int = 1
    gamma: float = 4.0
    covariance_abs_floor: float = 1e-9
    covariance_rel_floor: float = 1e-8
    quadrature_order: int = 32
    seed_epsilon: float = 1e-6
    require_curvature_degree: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.product_basis, ProductBasis):
            raise TypeError("product_basis must be a ProductBasis")
        assert_density_matches_mass(self.product_basis.convention)
        ranks = tuple(int(rank) for rank in self.ranks)
        dimension = self.product_basis.dimension
        if len(ranks) != dimension + 1:
            raise ValueError(f"ranks: {HighDimStatus.INVALID_SHAPE.value}")
        if ranks[0] != 1 or ranks[-1] != 1 or any(rank <= 0 for rank in ranks):
            raise ValueError(f"ranks: {HighDimStatus.INVALID_SHAPE.value}")
        internal = ranks[1:-1]
        if internal and any(rank != internal[0] for rank in internal):
            raise ValueError("P76 first implementation requires uniform internal rank")
        object.__setattr__(self, "ranks", ranks)
        object.__setattr__(self, "time_index", int(self.time_index))
        if self.time_index < 0:
            raise ValueError("time_index must be nonnegative")
        for name in ("gamma", "covariance_abs_floor", "covariance_rel_floor", "seed_epsilon"):
            value = float(getattr(self, name))
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        quadrature_order = int(self.quadrature_order)
        max_degree = max(int(basis.max_degree) for basis in self.product_basis.bases)
        if quadrature_order < max(8, 2 * max_degree + 4):
            raise ValueError("quadrature_order is too small for the declared basis")
        object.__setattr__(self, "quadrature_order", quadrature_order)
        if bool(self.require_curvature_degree):
            low_degree = [int(basis.max_degree) for basis in self.product_basis.bases if int(basis.max_degree) < 2]
            if low_degree:
                raise ValueError("degree must be at least 2 for a curvature-carrying UKF initializer")
        object.__setattr__(self, "require_curvature_degree", bool(self.require_curvature_degree))


@dataclass(frozen=True)
class P76AdjacentUKFMoments:
    """Adjacent physical moments assembled from UKF scout paths."""

    center: tf.Tensor
    covariance: tf.Tensor
    time_index: int
    previous_time_index: int
    claim_class: str

    def __post_init__(self) -> None:
        center = tf.convert_to_tensor(self.center, dtype=tf.float64)
        covariance = tf.convert_to_tensor(self.covariance, dtype=tf.float64)
        if center.shape.rank != 1:
            raise ValueError(f"center: {HighDimStatus.INVALID_SHAPE.value}")
        if covariance.shape.rank != 2 or covariance.shape[0] != covariance.shape[1]:
            raise ValueError(f"covariance: {HighDimStatus.INVALID_SHAPE.value}")
        if int(covariance.shape[0]) != int(center.shape[0]):
            raise ValueError(f"covariance: {HighDimStatus.INVALID_SHAPE.value}")
        _assert_finite("center", center)
        _assert_finite("covariance", covariance)
        if self.claim_class != P52_UKF_SCOUT_CLAIM:
            raise ValueError("UKF initializer requires scout_not_truth claim class")
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "covariance", covariance)
        object.__setattr__(self, "time_index", int(self.time_index))
        object.__setattr__(self, "previous_time_index", int(self.previous_time_index))
        object.__setattr__(self, "claim_class", str(self.claim_class))


@dataclass(frozen=True)
class P76StabilizedCovariance:
    """Symmetric positive covariance plus stabilization diagnostics."""

    covariance: tf.Tensor
    eigenvectors: tf.Tensor
    raw_eigenvalues: tf.Tensor
    floored_eigenvalues: tf.Tensor
    eigen_floor: tf.Tensor

    def __post_init__(self) -> None:
        covariance = tf.convert_to_tensor(self.covariance, dtype=tf.float64)
        eigenvectors = tf.convert_to_tensor(self.eigenvectors, dtype=tf.float64)
        raw = tf.convert_to_tensor(self.raw_eigenvalues, dtype=tf.float64)
        floored = tf.convert_to_tensor(self.floored_eigenvalues, dtype=tf.float64)
        floor = tf.convert_to_tensor(self.eigen_floor, dtype=tf.float64)
        if covariance.shape.rank != 2 or covariance.shape[0] != covariance.shape[1]:
            raise ValueError(f"covariance: {HighDimStatus.INVALID_SHAPE.value}")
        dim = int(covariance.shape[0])
        if eigenvectors.shape != (dim, dim) or raw.shape != (dim,) or floored.shape != (dim,):
            raise ValueError(f"eigendecomposition: {HighDimStatus.INVALID_SHAPE.value}")
        if floor.shape.rank != 0:
            raise ValueError(f"eigen_floor: {HighDimStatus.INVALID_SHAPE.value}")
        _assert_finite("stabilized_covariance", covariance)
        _assert_finite("eigenvectors", eigenvectors)
        _assert_finite("raw_eigenvalues", raw)
        _assert_finite("floored_eigenvalues", floored)
        _assert_finite("eigen_floor", floor)
        if not bool(tf.reduce_all(floored > 0.0).numpy()):
            raise ValueError("floored eigenvalues must be positive")
        object.__setattr__(self, "covariance", covariance)
        object.__setattr__(self, "eigenvectors", eigenvectors)
        object.__setattr__(self, "raw_eigenvalues", raw)
        object.__setattr__(self, "floored_eigenvalues", floored)
        object.__setattr__(self, "eigen_floor", floor)


@dataclass(frozen=True)
class P76UKFInitializerResult:
    """Result of the P76 UKF initializer."""

    cores: tuple[TTCore, ...]
    center: tf.Tensor
    linear_map: tf.Tensor
    stabilized_covariance: tf.Tensor
    raw_eigenvalues: tf.Tensor
    floored_eigenvalues: tf.Tensor
    projection_coefficients: tuple[tf.Tensor, ...]
    manifest: Mapping[str, object]
    status: str = P76_STATUS_OK
    nonclaims: tuple[str, ...] = P76_NONCLAIMS

    def __post_init__(self) -> None:
        cores = tuple(core if isinstance(core, TTCore) else TTCore(core) for core in self.cores)
        center = tf.convert_to_tensor(self.center, dtype=tf.float64)
        linear_map = tf.convert_to_tensor(self.linear_map, dtype=tf.float64)
        covariance = tf.convert_to_tensor(self.stabilized_covariance, dtype=tf.float64)
        raw = tf.convert_to_tensor(self.raw_eigenvalues, dtype=tf.float64)
        floored = tf.convert_to_tensor(self.floored_eigenvalues, dtype=tf.float64)
        coefficients = tuple(tf.convert_to_tensor(coef, dtype=tf.float64) for coef in self.projection_coefficients)
        if center.shape.rank != 1:
            raise ValueError(f"center: {HighDimStatus.INVALID_SHAPE.value}")
        dim = int(center.shape[0])
        if len(cores) != dim or len(coefficients) != dim:
            raise ValueError(f"initializer dimension: {HighDimStatus.INVALID_SHAPE.value}")
        for name, value in (
            ("linear_map", linear_map),
            ("stabilized_covariance", covariance),
        ):
            if value.shape != (dim, dim):
                raise ValueError(f"{name}: {HighDimStatus.INVALID_SHAPE.value}")
            _assert_finite(name, value)
        if raw.shape != (dim,) or floored.shape != (dim,):
            raise ValueError(f"eigenvalues: {HighDimStatus.INVALID_SHAPE.value}")
        _assert_finite("center", center)
        _assert_finite("raw_eigenvalues", raw)
        _assert_finite("floored_eigenvalues", floored)
        if self.status != P76_STATUS_OK:
            raise ValueError("unknown P76 initializer status")
        if "scout_not_truth" not in set(self.nonclaims):
            raise ValueError("missing scout_not_truth nonclaim")
        object.__setattr__(self, "cores", cores)
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "linear_map", linear_map)
        object.__setattr__(self, "stabilized_covariance", covariance)
        object.__setattr__(self, "raw_eigenvalues", raw)
        object.__setattr__(self, "floored_eigenvalues", floored)
        object.__setattr__(self, "projection_coefficients", coefficients)
        object.__setattr__(self, "manifest", MappingProxyType(dict(self.manifest)))
        object.__setattr__(self, "nonclaims", tuple(self.nonclaims))


def p76_adjacent_moments_from_scout(
    scout: UKFScoutResult,
    *,
    time_index: int,
) -> P76AdjacentUKFMoments:
    """Assemble block-diagonal adjacent moments from UKF scout paths."""

    if not isinstance(scout, UKFScoutResult):
        raise TypeError("scout must be a UKFScoutResult")
    if scout.claim_class != P52_UKF_SCOUT_CLAIM:
        raise ValueError("UKF initializer requires scout_not_truth claim class")
    index = int(time_index)
    if index < 1 or index > int(scout.horizon):
        raise ValueError("time_index outside scout horizon")
    mean_path = tf.convert_to_tensor(scout.mean_path, dtype=tf.float64)
    covariance_path = tf.convert_to_tensor(scout.covariance_path, dtype=tf.float64)
    if mean_path.shape.rank != 2 or covariance_path.shape.rank != 3:
        raise ValueError(f"scout paths: {HighDimStatus.INVALID_SHAPE.value}")
    current_mean = mean_path[index]
    previous_mean = mean_path[index - 1]
    current_covariance = covariance_path[index]
    previous_covariance = covariance_path[index - 1]
    state_dim = int(current_mean.shape[0])
    zeros = tf.zeros([state_dim, state_dim], dtype=tf.float64)
    covariance = tf.concat(
        [
            tf.concat([current_covariance, zeros], axis=1),
            tf.concat([zeros, previous_covariance], axis=1),
        ],
        axis=0,
    )
    return P76AdjacentUKFMoments(
        center=tf.concat([current_mean, previous_mean], axis=0),
        covariance=covariance,
        time_index=index,
        previous_time_index=index - 1,
        claim_class=scout.claim_class,
    )


def p76_initial_moments_from_scout(
    scout: UKFScoutResult,
) -> P76AdjacentUKFMoments:
    """Return one-state UKF moments for the initial SVX-ZC fit."""

    if not isinstance(scout, UKFScoutResult):
        raise TypeError("scout must be a UKFScoutResult")
    if scout.claim_class != P52_UKF_SCOUT_CLAIM:
        raise ValueError("UKF initializer requires scout_not_truth claim class")
    mean_path = tf.convert_to_tensor(scout.mean_path, dtype=tf.float64)
    covariance_path = tf.convert_to_tensor(scout.covariance_path, dtype=tf.float64)
    if mean_path.shape.rank != 2 or covariance_path.shape.rank != 3:
        raise ValueError(f"scout paths: {HighDimStatus.INVALID_SHAPE.value}")
    return P76AdjacentUKFMoments(
        center=mean_path[0],
        covariance=covariance_path[0],
        time_index=0,
        previous_time_index=-1,
        claim_class=scout.claim_class,
    )


def p76_stabilize_covariance(
    covariance: tf.Tensor,
    *,
    abs_floor: float,
    rel_floor: float,
) -> P76StabilizedCovariance:
    """Symmetrize and eigen-floor a covariance matrix."""

    matrix = _symmetrize(tf.convert_to_tensor(covariance, dtype=tf.float64))
    if matrix.shape.rank != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"covariance: {HighDimStatus.INVALID_SHAPE.value}")
    _assert_finite("covariance", matrix)
    if float(abs_floor) <= 0.0 or float(rel_floor) <= 0.0:
        raise ValueError("covariance floors must be positive")
    raw, eigenvectors = tf.linalg.eigh(matrix)
    _assert_finite("raw_eigenvalues", raw)
    max_abs = tf.reduce_max(tf.abs(raw))
    floor = tf.maximum(
        tf.constant(float(abs_floor), dtype=tf.float64),
        tf.constant(float(rel_floor), dtype=tf.float64) * max_abs,
    )
    floored = tf.maximum(raw, floor)
    stabilized = eigenvectors @ tf.linalg.diag(floored) @ tf.transpose(eigenvectors)
    stabilized = _symmetrize(stabilized)
    return P76StabilizedCovariance(
        covariance=stabilized,
        eigenvectors=eigenvectors,
        raw_eigenvalues=raw,
        floored_eigenvalues=floored,
        eigen_floor=floor,
    )


def p76_local_frame_from_moments(
    moments: P76AdjacentUKFMoments,
    config: P76UKFInitializerConfig,
) -> tuple[tf.Tensor, tf.Tensor, P76StabilizedCovariance]:
    """Build the UKF-whitened affine frame from adjacent moments."""

    if not isinstance(moments, P76AdjacentUKFMoments):
        raise TypeError("moments must be P76AdjacentUKFMoments")
    if not isinstance(config, P76UKFInitializerConfig):
        raise TypeError("config must be P76UKFInitializerConfig")
    if config.product_basis.dimension != int(moments.center.shape[0]):
        raise ValueError("product_basis dimension must match adjacent UKF dimension")
    stabilized = p76_stabilize_covariance(
        moments.covariance,
        abs_floor=config.covariance_abs_floor,
        rel_floor=config.covariance_rel_floor,
    )
    linear_map = (
        tf.constant(config.gamma, dtype=tf.float64)
        * stabilized.eigenvectors
        @ tf.linalg.diag(tf.sqrt(stabilized.floored_eigenvalues))
    )
    _assert_finite("linear_map", linear_map)
    return moments.center, linear_map, stabilized


def p76_gaussian_sqrt_projection_coefficients(
    product_basis: ProductBasis,
    *,
    gamma: float,
    quadrature_order: int,
    center: tf.Tensor | None = None,
    linear_map: tf.Tensor | None = None,
    reference_offset: tf.Tensor | None = None,
    reference_matrix: tf.Tensor | None = None,
) -> tuple[tf.Tensor, ...]:
    """Project the UKF Gaussian square root into the fixed reference basis.

    The guide is evaluated at physical points from the route's fixed reference
    map and standardized with the UKF center and marginal scale. This keeps the
    target coordinate map unchanged while making the UKF frame operational.
    Correlations are retained in the UKF diagnostics but the product guide uses
    its coordinatewise marginals, as required by this initializer's factorized
    projection contract.
    """

    if not isinstance(product_basis, ProductBasis):
        raise TypeError("product_basis must be a ProductBasis")
    if float(gamma) <= 0.0:
        raise ValueError("gamma must be positive")
    dimension = product_basis.dimension
    if center is None:
        center = tf.zeros([dimension], dtype=tf.float64)
    else:
        center = tf.convert_to_tensor(center, dtype=tf.float64)
    if linear_map is None:
        linear_map = tf.eye(dimension, dtype=tf.float64)
    else:
        linear_map = tf.convert_to_tensor(linear_map, dtype=tf.float64)
    if reference_offset is None:
        reference_offset = tf.zeros([dimension], dtype=tf.float64)
    else:
        reference_offset = tf.convert_to_tensor(reference_offset, dtype=tf.float64)
    if reference_matrix is None:
        reference_matrix = tf.eye(dimension, dtype=tf.float64)
    else:
        reference_matrix = tf.convert_to_tensor(reference_matrix, dtype=tf.float64)
    if center.shape != (dimension,) or linear_map.shape != (dimension, dimension):
        raise ValueError("UKF frame shape does not match product basis")
    if reference_offset.shape != (dimension,) or reference_matrix.shape != (dimension, dimension):
        raise ValueError("reference map shape does not match product basis")
    marginal_scale = tf.sqrt(tf.reduce_sum(tf.square(linear_map), axis=1))
    marginal_scale = tf.maximum(marginal_scale, tf.constant(1e-15, dtype=tf.float64))
    nodes, weights = _legendre_gauss_nodes_weights(int(quadrature_order))
    coefficients = []
    for basis in product_basis.bases:
        half_length = 0.5 * basis.domain.length
        midpoint = 0.5 * (basis.domain.left + basis.domain.right)
        points = midpoint + half_length * nodes
        axis = len(coefficients)
        physical = reference_offset[axis] + reference_matrix[axis, axis] * points
        standardized = (physical - center[axis]) / marginal_scale[axis]
        values = tf.exp(
            -0.25
            * tf.constant(float(gamma) ** 2, dtype=tf.float64)
            * tf.square(standardized)
        )
        active_weights = _active_1d_weights(
            product_basis.convention.mass_measure,
            half_length,
            weights,
        )
        guide_normalizer = tf.reduce_sum(
            active_weights
            * tf.exp(
                -0.5
                * tf.constant(float(gamma) ** 2, dtype=tf.float64)
                * tf.square(standardized)
            )
        )
        tf.debugging.assert_positive(
            guide_normalizer,
            message="UKF guide normalizer must be positive",
        )
        values = values / tf.sqrt(guide_normalizer)
        basis_values = basis.evaluate(points)
        rhs = tf.reduce_sum(active_weights[:, tf.newaxis] * values[:, tf.newaxis] * basis_values, axis=0)
        mass = basis.mass_matrix(product_basis.convention.mass_measure)
        coefficient = tf.reshape(tf.linalg.solve(mass, rhs[:, tf.newaxis]), [-1])
        _assert_finite("projection_coefficient", coefficient)
        coefficients.append(coefficient)
    return tuple(coefficients)


def p76_rank_one_ukf_sqrt_cores(
    product_basis: ProductBasis,
    *,
    gamma: float,
    quadrature_order: int,
) -> tuple[TTCore, ...]:
    """Build rank-one TT cores for the projected UKF square-root guide."""

    coefficients = p76_gaussian_sqrt_projection_coefficients(
        product_basis,
        gamma=gamma,
        quadrature_order=quadrature_order,
    )
    return tuple(TTCore(tf.reshape(coef, [1, int(coef.shape[0]), 1])) for coef in coefficients)


def p76_embed_rank_one_with_seeded_channels(
    rank_one_coefficients: Sequence[tf.Tensor],
    *,
    ranks: tuple[int, ...],
    seed_epsilon: float,
) -> tuple[TTCore, ...]:
    """Embed rank-one coefficient vectors into a uniform-rank seeded TT."""

    coefficients = tuple(tf.convert_to_tensor(coef, dtype=tf.float64) for coef in rank_one_coefficients)
    dim = len(coefficients)
    ranks = tuple(int(rank) for rank in ranks)
    if len(ranks) != dim + 1:
        raise ValueError(f"ranks: {HighDimStatus.INVALID_SHAPE.value}")
    if ranks[0] != 1 or ranks[-1] != 1 or any(rank <= 0 for rank in ranks):
        raise ValueError(f"ranks: {HighDimStatus.INVALID_SHAPE.value}")
    internal = ranks[1:-1]
    if internal and any(rank != internal[0] for rank in internal):
        raise ValueError("P76 first implementation requires uniform internal rank")
    if float(seed_epsilon) <= 0.0:
        raise ValueError("seed_epsilon must be positive")
    reference_scale = tf.maximum(
        tf.abs(coefficients[0][0]),
        tf.constant(1e-300, dtype=tf.float64),
    )
    max_rank = max(ranks)
    extra_count = max(max_rank - 1, 0)
    seeded_scale = (
        reference_scale * tf.constant(float(seed_epsilon) / float(extra_count), dtype=tf.float64)
        if extra_count > 0
        else tf.constant(0.0, dtype=tf.float64)
    )
    cores = []
    for axis, coefficient in enumerate(coefficients):
        if coefficient.shape.rank != 1:
            raise ValueError(f"coefficient: {HighDimStatus.INVALID_SHAPE.value}")
        _assert_finite("rank_one_coefficient", coefficient)
        left_rank = ranks[axis]
        right_rank = ranks[axis + 1]
        basis_dim = int(coefficient.shape[0])
        values = tf.zeros([left_rank, basis_dim, right_rank], dtype=tf.float64)
        indices = [[0, basis, 0] for basis in range(basis_dim)]
        updates = [coefficient[basis] for basis in range(basis_dim)]
        for channel in range(1, min(left_rank, right_rank)):
            basis_index = _seeded_basis_index(axis=axis, channel=channel, basis_dim=basis_dim)
            indices.append([channel, basis_index, channel])
            updates.append(tf.constant(1.0, dtype=tf.float64))
        if axis == 0:
            for channel in range(1, right_rank):
                basis_index = _seeded_basis_index(axis=axis, channel=channel, basis_dim=basis_dim)
                indices.append([0, basis_index, channel])
                updates.append(seeded_scale)
        if axis == dim - 1:
            for channel in range(1, left_rank):
                basis_index = _seeded_basis_index(axis=axis, channel=channel, basis_dim=basis_dim)
                indices.append([channel, basis_index, 0])
                updates.append(tf.constant(1.0, dtype=tf.float64))
        values = tf.tensor_scatter_nd_update(
            values,
            tf.constant(indices, dtype=tf.int64),
            tf.stack(updates),
        )
        cores.append(TTCore(values))
    return tuple(cores)


def p76_build_ukf_initializer(
    scout: UKFScoutResult,
    config: P76UKFInitializerConfig,
    *,
    reference_offset: tf.Tensor | None = None,
    reference_matrix: tf.Tensor | None = None,
) -> P76UKFInitializerResult:
    """Build UKF initializer cores and manifest for a fixed reference map."""

    if not isinstance(config, P76UKFInitializerConfig):
        raise TypeError("config must be P76UKFInitializerConfig")
    moments = (
        p76_initial_moments_from_scout(scout)
        if int(config.time_index) == 0
        else p76_adjacent_moments_from_scout(scout, time_index=config.time_index)
    )
    center, linear_map, stabilized = p76_local_frame_from_moments(moments, config)
    coefficients = p76_gaussian_sqrt_projection_coefficients(
        config.product_basis,
        gamma=config.gamma,
        quadrature_order=config.quadrature_order,
        center=center,
        linear_map=linear_map,
        reference_offset=reference_offset,
        reference_matrix=reference_matrix,
    )
    cores = p76_embed_rank_one_with_seeded_channels(
        coefficients,
        ranks=config.ranks,
        seed_epsilon=config.seed_epsilon,
    )
    manifest = p76_initializer_manifest_payload(
        config=config,
        moments=moments,
        linear_map=linear_map,
        stabilized=stabilized,
    )
    return P76UKFInitializerResult(
        cores=cores,
        center=center,
        linear_map=linear_map,
        stabilized_covariance=stabilized.covariance,
        raw_eigenvalues=stabilized.raw_eigenvalues,
        floored_eigenvalues=stabilized.floored_eigenvalues,
        projection_coefficients=coefficients,
        manifest=manifest,
    )


def p76_initializer_manifest_payload(
    *,
    config: P76UKFInitializerConfig,
    moments: P76AdjacentUKFMoments,
    linear_map: tf.Tensor,
    stabilized: P76StabilizedCovariance,
) -> Mapping[str, object]:
    """Return a JSON-friendly P76 initializer manifest."""

    linear_map = tf.convert_to_tensor(linear_map, dtype=tf.float64)
    return {
        "schema_version": P76_SCHEMA_VERSION,
        "initializer_rule": P76_UKF_INITIALIZER_RULE,
        "route_classification": P76_ROUTE_CLASSIFICATION,
        "status": P76_STATUS_OK,
        "claim_class": moments.claim_class,
        "time_index": moments.time_index,
        "previous_time_index": moments.previous_time_index,
        "dimension": config.product_basis.dimension,
        "basis_dim_tuple": config.product_basis.basis_dim_tuple(),
        "rank_tuple": config.ranks,
        "gamma": config.gamma,
        "covariance_abs_floor": config.covariance_abs_floor,
        "covariance_rel_floor": config.covariance_rel_floor,
        "eigen_floor": float(stabilized.eigen_floor.numpy()),
        "raw_eigenvalue_range": _range_payload(stabilized.raw_eigenvalues),
        "floored_eigenvalue_range": _range_payload(stabilized.floored_eigenvalues),
        "center_dimension": int(moments.center.shape[0]),
        "linear_map_shape": _shape_payload(linear_map),
        "linear_map_finite": bool(tf.reduce_all(tf.math.is_finite(linear_map)).numpy()),
        "stabilized_covariance_shape": _shape_payload(stabilized.covariance),
        "stabilized_covariance_finite": bool(
            tf.reduce_all(tf.math.is_finite(stabilized.covariance)).numpy()
        ),
        "quadrature_order": config.quadrature_order,
        "seed_epsilon": config.seed_epsilon,
        "degree_guard_status": "enforced" if config.require_curvature_degree else "not_required",
        "source_route_prefit_used": False,
        "audit_data_used": False,
        "default_behavior_changed": False,
        "nonclaims": P76_NONCLAIMS,
    }


def _legendre_gauss_nodes_weights(order: int) -> tuple[tf.Tensor, tf.Tensor]:
    if int(order) < 2:
        raise ValueError("order must be at least 2")
    k = tf.cast(tf.range(1, int(order), dtype=tf.int32), tf.float64)
    beta = k / tf.sqrt(4.0 * tf.square(k) - 1.0)
    jacobi = tf.linalg.diag(beta, k=1) + tf.linalg.diag(beta, k=-1)
    eigenvalues, eigenvectors = tf.linalg.eigh(jacobi)
    weights = 2.0 * tf.square(eigenvectors[0, :])
    return eigenvalues, weights


def _active_1d_weights(
    measure: MassMeasure,
    half_length: tf.Tensor,
    reference_weights: tf.Tensor,
) -> tf.Tensor:
    if measure is MassMeasure.REFERENCE_MEASURE:
        return 0.5 * reference_weights
    if measure is MassMeasure.REFERENCE_LEBESGUE:
        return half_length * reference_weights
    raise ValueError(HighDimStatus.MEASURE_MISMATCH.value)


def _symmetrize(matrix: tf.Tensor) -> tf.Tensor:
    return 0.5 * (matrix + tf.transpose(matrix))


def _seeded_basis_index(*, axis: int, channel: int, basis_dim: int) -> int:
    if int(basis_dim) <= 1:
        return 0
    return 1 + ((int(axis) + int(channel) - 1) % (int(basis_dim) - 1))


def _assert_finite(name: str, tensor: tf.Tensor) -> None:
    values = tf.convert_to_tensor(tensor, dtype=tf.float64)
    assert_tf_float64(name, values)
    if not bool(tf.reduce_all(tf.math.is_finite(values)).numpy()):
        raise ValueError(f"{name}: {HighDimStatus.NONFINITE_VALUE.value}")


def _shape_payload(tensor: tf.Tensor) -> tuple[int, ...]:
    return tuple(int(dim) for dim in tf.convert_to_tensor(tensor).shape)


def _range_payload(tensor: tf.Tensor) -> tuple[float, float]:
    values = tf.convert_to_tensor(tensor, dtype=tf.float64)
    return (
        float(tf.reduce_min(values).numpy()),
        float(tf.reduce_max(values).numpy()),
    )


__all__ = [
    "P76AdjacentUKFMoments",
    "P76_NONCLAIMS",
    "P76_ROUTE_CLASSIFICATION",
    "P76_SCHEMA_VERSION",
    "P76_STATUS_OK",
    "P76StabilizedCovariance",
    "P76UKFInitializerConfig",
    "P76UKFInitializerResult",
    "P76_UKF_INITIALIZER_RULE",
    "p76_adjacent_moments_from_scout",
    "p76_build_ukf_initializer",
    "p76_embed_rank_one_with_seeded_channels",
    "p76_gaussian_sqrt_projection_coefficients",
    "p76_initial_moments_from_scout",
    "p76_initializer_manifest_payload",
    "p76_local_frame_from_moments",
    "p76_rank_one_ukf_sqrt_cores",
    "p76_stabilize_covariance",
]
