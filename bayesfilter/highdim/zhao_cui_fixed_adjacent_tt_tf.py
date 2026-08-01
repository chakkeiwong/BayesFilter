"""Fixed adjacent-state squared-TT comparator for scalar state-space models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import tensorflow as tf

from bayesfilter.highdim.bases import ProductBasis
from bayesfilter.highdim.derivatives import (
    FiniteDifferenceTable,
    FixedBranchScoreResult,
    fixed_branch_compatibility_hash,
    make_finite_difference_row,
)
from bayesfilter.highdim.diagnostics import HighDimStatus, freeze_mapping
from bayesfilter.highdim.filtering import (
    FixedBranchFilterConfig,
    HighDimCoordinateMap,
    legendre_gauss_nodes_weights,
)
from bayesfilter.highdim.fixed_branch import BranchIdentity, BranchManifest
from bayesfilter.highdim.fitting import FixedTTFitSampleBatch, FixedTTFitter
from bayesfilter.highdim.models import TFHighDimStateSpaceModel
from bayesfilter.highdim.squared_tt import (
    SquaredTTDensity,
    TensorProductReferenceDensity,
)
from bayesfilter.highdim.tt import TTCore


ROUTE_ID = "zhao_cui_fixed_adjacent_state_squared_tt_v1"
ROUTE_CLASSIFICATION = "extension_or_invention"
ROUTE_SUBTYPE = "fixed_parameter_adjacent_state_squared_tt_extension"
AXIS_ORDER = ("x_t", "x_t_minus_1")
INITIALIZER_ID = "orthonormal_mode_diagonal_norm_balanced_v1"


@dataclass(frozen=True)
class ScalarAdjacentTTConfig:
    """Frozen one-axis initial and two-axis adjacent-state fit configs."""

    initial: FixedBranchFilterConfig
    adjacent: FixedBranchFilterConfig
    scalar_coordinate_map: HighDimCoordinateMap
    transition_before_first_observation: bool = False

    def __post_init__(self) -> None:
        if self.initial.product_basis is None or self.initial.fit_config is None:
            raise ValueError("initial config requires product_basis and fit_config")
        if self.adjacent.product_basis is None or self.adjacent.fit_config is None:
            raise ValueError("adjacent config requires product_basis and fit_config")
        if self.initial.product_basis.dimension != 1:
            raise ValueError("initial product basis must be one-dimensional")
        if self.adjacent.product_basis.dimension != 2:
            raise ValueError("adjacent product basis must be two-dimensional")
        if self.initial.measure_convention != self.adjacent.measure_convention:
            raise ValueError("initial and adjacent measure conventions must match")
        if not callable(getattr(self.scalar_coordinate_map, "forward", None)):
            raise TypeError("scalar_coordinate_map must implement forward")
        if not callable(getattr(self.scalar_coordinate_map, "manifest_payload", None)):
            raise TypeError("scalar_coordinate_map must implement manifest_payload")
        if not isinstance(self.transition_before_first_observation, bool):
            raise TypeError("transition_before_first_observation must be bool")

    def manifest_payload(self) -> Mapping[str, object]:
        return {
            "route_id": ROUTE_ID,
            "route_classification": ROUTE_CLASSIFICATION,
            "route_subtype": ROUTE_SUBTYPE,
            "axis_order": AXIS_ORDER,
            "initializer_id": INITIALIZER_ID,
            "transition_before_first_observation": self.transition_before_first_observation,
            "initial": self.initial.manifest_payload(),
            "adjacent": self.adjacent.manifest_payload(),
            "scalar_coordinate_map": self.scalar_coordinate_map.manifest_payload(),
        }


@dataclass(frozen=True)
class ScalarAdjacentTTStep:
    """One fitted initial or adjacent-state squared-TT update."""

    time_index: int
    target_kind: str
    log_increment: tf.Tensor
    scaled_normalizer: tf.Tensor
    log_scale_shift: tf.Tensor
    log_scale_shift_index: int
    fit_result: object
    density: SquaredTTDensity
    carried_keep_axes: tuple[int, ...]
    marginal_mass: tf.Tensor
    diagnostics: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "time_index", int(self.time_index))
        object.__setattr__(self, "log_scale_shift_index", int(self.log_scale_shift_index))
        for name in (
            "log_increment",
            "scaled_normalizer",
            "log_scale_shift",
            "marginal_mass",
        ):
            value = tf.convert_to_tensor(getattr(self, name), dtype=tf.float64)
            if value.shape.rank != 0:
                raise ValueError(f"{name} must be scalar")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "carried_keep_axes", tuple(self.carried_keep_axes))
        object.__setattr__(self, "diagnostics", freeze_mapping(self.diagnostics))


@dataclass(frozen=True)
class ScalarAdjacentTTResult:
    """Finite adjacent-state value result and frozen-branch identity."""

    log_likelihood: tf.Tensor
    log_increments: tf.Tensor
    steps: tuple[ScalarAdjacentTTStep, ...]
    branch_identity: BranchIdentity
    compatibility_hash: str
    diagnostics: Mapping[str, object]

    def __post_init__(self) -> None:
        log_likelihood = tf.convert_to_tensor(self.log_likelihood, dtype=tf.float64)
        log_increments = tf.convert_to_tensor(self.log_increments, dtype=tf.float64)
        if log_likelihood.shape.rank != 0 or log_increments.shape.rank != 1:
            raise ValueError("invalid value-result shapes")
        if len(self.steps) != int(log_increments.shape[0]):
            raise ValueError("step count must match log increments")
        if len(str(self.compatibility_hash)) != 64:
            raise ValueError("compatibility_hash must be a SHA-256 digest")
        object.__setattr__(self, "log_likelihood", log_likelihood)
        object.__setattr__(self, "log_increments", log_increments)
        object.__setattr__(self, "steps", tuple(self.steps))
        object.__setattr__(self, "diagnostics", freeze_mapping(self.diagnostics))


def scalar_adjacent_state_fixed_tt_value(
    model: TFHighDimStateSpaceModel,
    theta: tf.Tensor,
    observations: tf.Tensor,
    config: ScalarAdjacentTTConfig,
    *,
    fixture_id: str = "contract-e-tp.phase6.scalar-adjacent-tt.v1",
    branch_seed_prefix: str = "contract-e-tp-phase6-scalar-adjacent-tt",
) -> ScalarAdjacentTTResult:
    """Evaluate the fixed adjacent-state squared-TT finite likelihood."""

    if int(model.state_dim()) != 1:
        raise TypeError("scalar adjacent-state TT route requires state_dim == 1")
    theta_vector = _theta_vector(theta, int(model.parameter_dim()))
    observation_matrix = _observation_matrix(observations, int(model.observation_dim()))
    observation_count = int(observation_matrix.shape[0])
    if observation_count < 1:
        raise ValueError("observations must be nonempty")

    previous_density: SquaredTTDensity | None = None
    previous_keep_axes: tuple[int, ...] = ()
    adjacent_initial_cores = config.adjacent.initial_cores
    steps = []
    log_increments = []

    for time_index in range(observation_count):
        if time_index == 0 and not config.transition_before_first_observation:
            basis = _required_basis(config.initial)
            reference_points, weights = _reference_quadrature(
                basis,
                config.initial.fit_quadrature_order,
            )
            physical_points, current_log_abs_det = config.scalar_coordinate_map.forward(
                reference_points
            )
            log_target = (
                model.initial_log_density(theta_vector, physical_points)
                + model.observation_log_density(
                    theta_vector,
                    physical_points,
                    observation_matrix[time_index],
                    t=time_index,
                )
                + current_log_abs_det
                - _log_reference_density(basis)
            )
            active_config = config.initial
            initial_cores = _required_initial_cores(active_config)
            target_kind = "initial_state_observation"
            keep_axes = (0,)
        elif time_index == 0:
            basis = _required_basis(config.adjacent)
            reference_points, weights = _reference_quadrature(
                basis,
                config.adjacent.fit_quadrature_order,
            )
            current_reference = reference_points[:, 0:1]
            previous_reference = reference_points[:, 1:2]
            current_physical, current_log_abs_det = config.scalar_coordinate_map.forward(
                current_reference
            )
            previous_physical, previous_log_abs_det = config.scalar_coordinate_map.forward(
                previous_reference
            )
            log_target = (
                model.initial_log_density(theta_vector, previous_physical)
                + model.transition_log_density(
                    theta_vector,
                    previous_physical,
                    current_physical,
                    t=time_index,
                )
                + model.observation_log_density(
                    theta_vector,
                    current_physical,
                    observation_matrix[time_index],
                    t=time_index,
                )
                + current_log_abs_det
                + previous_log_abs_det
                - _log_reference_density(basis)
            )
            active_config = config.adjacent
            initial_cores = (
                adjacent_initial_cores
                if adjacent_initial_cores is not None
                else _required_initial_cores(active_config)
            )
            target_kind = "transitioned_initial_adjacent_state_update"
            keep_axes = (0,)
        else:
            if previous_density is None or previous_keep_axes != (0,):
                raise RuntimeError("missing previous fitted marginal")
            basis = _required_basis(config.adjacent)
            reference_points, weights = _reference_quadrature(
                basis,
                config.adjacent.fit_quadrature_order,
            )
            current_reference = reference_points[:, 0:1]
            previous_reference = reference_points[:, 1:2]
            current_physical, current_log_abs_det = config.scalar_coordinate_map.forward(
                current_reference
            )
            previous_physical, _ = config.scalar_coordinate_map.forward(
                previous_reference
            )
            previous_log_density = tf.math.log(
                previous_density.normalized_marginal_density_values(
                    previous_keep_axes,
                    previous_reference,
                )
            )
            log_target = (
                previous_log_density
                + model.transition_log_density(
                    theta_vector,
                    previous_physical,
                    current_physical,
                    t=time_index,
                )
                + model.observation_log_density(
                    theta_vector,
                    current_physical,
                    observation_matrix[time_index],
                    t=time_index,
                )
                + current_log_abs_det
                - _log_reference_density(config.initial.product_basis)
            )
            active_config = config.adjacent
            initial_cores = (
                adjacent_initial_cores
                if adjacent_initial_cores is not None
                else _required_initial_cores(active_config)
            )
            target_kind = "adjacent_state_update"
            keep_axes = (0,)

        log_scale_shift_index = int(tf.argmax(log_target).numpy())
        log_scale_shift = tf.reduce_max(log_target)
        sqrt_target = tf.exp(0.5 * (log_target - log_scale_shift))
        fit_result = FixedTTFitter().fit(
            product_basis=basis,
            samples=FixedTTFitSampleBatch(
                points=reference_points,
                target_values=sqrt_target,
                weights=weights,
            ),
            config=active_config.fit_config,
            initial_cores=initial_cores,
            branch_seed=f"{branch_seed_prefix}:t{time_index}:fit",
            measure_convention=active_config.measure_convention,
        )
        if fit_result.status is not HighDimStatus.OK:
            raise ValueError(fit_result.status.value)
        density = _density_from_fit(fit_result, active_config)
        scaled_normalizer = density.normalizer()
        log_increment = tf.math.log(scaled_normalizer) + log_scale_shift
        marginal_mass = _marginal_mass(density, keep_axes)
        step = ScalarAdjacentTTStep(
            time_index=time_index,
            target_kind=target_kind,
            log_increment=log_increment,
            scaled_normalizer=scaled_normalizer,
            log_scale_shift=log_scale_shift,
            log_scale_shift_index=log_scale_shift_index,
            fit_result=fit_result,
            density=density,
            carried_keep_axes=keep_axes,
            marginal_mass=marginal_mass,
            diagnostics={
                "axis_order": (
                    ("x_0",)
                    if time_index == 0
                    and not config.transition_before_first_observation
                    else AXIS_ORDER
                ),
                "integrated_axes": (
                    ()
                    if time_index == 0
                    and not config.transition_before_first_observation
                    else (1,)
                ),
                "fit_residual": fit_result.fit_residual,
                "fit_update_structure": _fit_update_structure(fit_result),
                "source_route_operation": (
                    "initial_same_target_fit"
                    if time_index == 0
                    and not config.transition_before_first_observation
                    else (
                        "transitioned_initial_adjacent_fit_then_previous_state_marginal"
                        if time_index == 0
                        else "algorithm2_adjacent_fit_then_previous_state_marginal"
                    )
                ),
            },
        )
        steps.append(step)
        log_increments.append(log_increment)
        previous_density = density
        previous_keep_axes = keep_axes
        if time_index > 0:
            adjacent_initial_cores = tuple(fit_result.fitted_tt.cores)

    increments = tf.stack(log_increments)
    log_likelihood = tf.reduce_sum(increments)
    compatibility_hash = _compatibility_hash(
        model=model,
        observation_matrix=observation_matrix,
        config=config,
        branch_seed_prefix=branch_seed_prefix,
        steps=steps,
    )
    manifest = BranchManifest(
        version="zhao_cui_fixed_adjacent_state_squared_tt_result.v1",
        payload={
            "route_id": ROUTE_ID,
            "route_classification": ROUTE_CLASSIFICATION,
            "route_subtype": ROUTE_SUBTYPE,
            "fixture_id": fixture_id,
            "model": model.manifest_payload(),
            "theta": theta_vector,
            "observations": observation_matrix,
            "config": _compatibility_config_payload(config),
            "log_increments": increments,
            "log_likelihood": log_likelihood,
            "compatibility_hash": compatibility_hash,
            "step_fit_hashes": tuple(
                step.fit_result.branch_identity.hash.value for step in steps
            ),
            "what_is_not_claimed": (
                "adaptive_tt_cross",
                "adaptive_ttsirt_reproduction",
                "source_faithful",
                "exact_filtering",
                "cross_method_equivalence",
                "hmc_readiness",
                "default_readiness",
            ),
        },
    )
    identity = BranchIdentity(manifest=manifest, hash=manifest.sha256())
    return ScalarAdjacentTTResult(
        log_likelihood=log_likelihood,
        log_increments=increments,
        steps=tuple(steps),
        branch_identity=identity,
        compatibility_hash=compatibility_hash,
        diagnostics={
            "route_id": ROUTE_ID,
            "route_classification": ROUTE_CLASSIFICATION,
            "route_subtype": ROUTE_SUBTYPE,
            "axis_order": AXIS_ORDER,
            "observation_count": observation_count,
            "autodiff_finite_program": True,
            "previous_marginal_derivative_owner": "tensorflow_total_derivative",
            "transition_before_first_observation": config.transition_before_first_observation,
        },
    )


def scalar_adjacent_state_fixed_tt_score(
    model: TFHighDimStateSpaceModel,
    theta: tf.Tensor,
    observations: tf.Tensor,
    config: ScalarAdjacentTTConfig,
    *,
    finite_difference_h: Sequence[float] = (1e-2, 3e-3, 1e-3, 3e-4),
    fixture_id: str = "contract-e-tp.phase6.scalar-adjacent-tt.score.v1",
    branch_seed_prefix: str = "contract-e-tp-phase6-scalar-adjacent-tt",
) -> FixedBranchScoreResult:
    """Differentiate the same fixed adjacent-state finite value program."""

    theta_vector = _theta_vector(theta, int(model.parameter_dim()))
    with tf.GradientTape() as tape:
        tape.watch(theta_vector)
        value_result = scalar_adjacent_state_fixed_tt_value(
            model,
            theta_vector,
            observations,
            config,
            fixture_id=fixture_id.replace("score", "value"),
            branch_seed_prefix=branch_seed_prefix,
        )
    score = tape.gradient(value_result.log_likelihood, theta_vector)
    if score is None:
        raise RuntimeError("TensorFlow did not produce a total gradient")
    score = tf.convert_to_tensor(score, dtype=tf.float64)
    if not bool(tf.reduce_all(tf.math.is_finite(score)).numpy()):
        raise ValueError(HighDimStatus.NONFINITE_VALUE.value)

    rows = []
    for parameter_index in range(int(model.parameter_dim())):
        for h in finite_difference_h:
            step = tf.constant(float(h), dtype=tf.float64)
            plus_theta = tf.tensor_scatter_nd_add(
                theta_vector,
                [[parameter_index]],
                [step],
            )
            minus_theta = tf.tensor_scatter_nd_add(
                theta_vector,
                [[parameter_index]],
                [-step],
            )
            plus = scalar_adjacent_state_fixed_tt_value(
                model,
                plus_theta,
                observations,
                config,
                fixture_id=fixture_id.replace("score", "fd-plus"),
                branch_seed_prefix=branch_seed_prefix,
            )
            minus = scalar_adjacent_state_fixed_tt_value(
                model,
                minus_theta,
                observations,
                config,
                fixture_id=fixture_id.replace("score", "fd-minus"),
                branch_seed_prefix=branch_seed_prefix,
            )
            rows.append(
                make_finite_difference_row(
                    parameter_index=parameter_index,
                    h=float(h),
                    value_plus=plus.log_likelihood,
                    value_minus=minus.log_likelihood,
                    branch_hash_plus=plus.compatibility_hash,
                    branch_hash_minus=minus.compatibility_hash,
                    branch_hash_base=value_result.compatibility_hash,
                    analytic_gradient=score[parameter_index],
                )
            )
    return FixedBranchScoreResult(
        log_likelihood=value_result.log_likelihood,
        score=score,
        branch_identity=value_result.branch_identity,
        replay_tape_hash=value_result.branch_identity.hash.value,
        finite_difference_table=FiniteDifferenceTable(tuple(rows)),
        status=HighDimStatus.OK,
        diagnostics={
            "route_id": ROUTE_ID,
            "route_classification": ROUTE_CLASSIFICATION,
            "route_subtype": ROUTE_SUBTYPE,
            "score_backend": "tensorflow_total_autodiff_same_finite_program",
            "previous_marginal_derivative_included": True,
            "finite_difference_h": tuple(float(h) for h in finite_difference_h),
            "compatibility_hash": value_result.compatibility_hash,
            "log_increments": value_result.log_increments,
            "step_evidence": tuple(
                {
                    "time_index": step.time_index,
                    "target_kind": step.target_kind,
                    "fit_dimension": step.density.sqrt_tt.product_basis.dimension,
                    "axis_order": step.diagnostics["axis_order"],
                    "integrated_axes": step.diagnostics["integrated_axes"],
                    "log_scale_shift_index": step.log_scale_shift_index,
                    "marginal_mass": step.marginal_mass,
                    "fit_residual": step.fit_result.fit_residual,
                    "fit_update_structure": step.diagnostics[
                        "fit_update_structure"
                    ],
                    "fit_condition_numbers": tuple(
                        update.get("condition_number")
                        for update in step.fit_result.core_update_statuses
                    ),
                    "fit_unscaled_normal_condition_numbers": tuple(
                        update.get("unscaled_normal_condition_number")
                        for update in step.fit_result.core_update_statuses
                    ),
                }
                for step in value_result.steps
            ),
        },
    )


def norm_balanced_initial_cores(
    product_basis: ProductBasis,
    ranks: Sequence[int],
) -> tuple[TTCore, ...]:
    """Build independent orthonormal-mode channels for one or two axes."""

    rank_tuple = tuple(int(rank) for rank in ranks)
    if product_basis.dimension == 1:
        if rank_tuple != (1, 1):
            raise ValueError("one-axis initializer requires ranks (1, 1)")
        values = tf.zeros(
            [1, product_basis.bases[0].basis_dim, 1],
            dtype=tf.float64,
        )
        return (
            TTCore(
                tf.tensor_scatter_nd_update(
                    values,
                    [[0, 0, 0]],
                    [tf.constant(1.0, dtype=tf.float64)],
                )
            ),
        )
    if product_basis.dimension != 2:
        raise ValueError("norm-balanced initializer currently supports one or two axes")
    if len(rank_tuple) != 3 or rank_tuple[0] != 1 or rank_tuple[2] != 1:
        raise ValueError("two-axis initializer requires ranks (1, r, 1)")
    active_rank = rank_tuple[1]
    if active_rank > min(basis.basis_dim for basis in product_basis.bases):
        raise ValueError("adjacent rank exceeds available independent polynomial modes")
    coefficient = tf.math.rsqrt(tf.cast(active_rank, tf.float64))
    left = tf.zeros(
        [1, product_basis.bases[0].basis_dim, active_rank],
        dtype=tf.float64,
    )
    right = tf.zeros(
        [active_rank, product_basis.bases[1].basis_dim, 1],
        dtype=tf.float64,
    )
    left_indices = [[0, channel, channel] for channel in range(active_rank)]
    right_indices = [[channel, channel, 0] for channel in range(active_rank)]
    updates = tf.fill([active_rank], coefficient)
    return (
        TTCore(tf.tensor_scatter_nd_update(left, left_indices, updates)),
        TTCore(tf.tensor_scatter_nd_update(right, right_indices, updates)),
    )


def _required_basis(config: FixedBranchFilterConfig) -> ProductBasis:
    if config.product_basis is None:
        raise ValueError("config requires a product basis")
    return config.product_basis


def _required_initial_cores(config: FixedBranchFilterConfig) -> tuple[TTCore, ...]:
    if config.initial_cores is None:
        raise ValueError("config requires frozen initial cores")
    return tuple(config.initial_cores)


def _theta_vector(theta: tf.Tensor, parameter_dim: int) -> tf.Tensor:
    value = tf.convert_to_tensor(theta, dtype=tf.float64)
    if value.shape.rank != 1 or value.shape[0] != int(parameter_dim):
        raise ValueError("theta has the wrong shape")
    return value


def _observation_matrix(observations: tf.Tensor, observation_dim: int) -> tf.Tensor:
    value = tf.convert_to_tensor(observations, dtype=tf.float64)
    if value.shape.rank == 1:
        value = value[:, tf.newaxis]
    if value.shape.rank != 2 or value.shape[1] != int(observation_dim):
        raise ValueError("observations have the wrong shape")
    return value


def _reference_quadrature(
    product_basis: ProductBasis,
    order: int,
) -> tuple[tf.Tensor, tf.Tensor]:
    nodes, weights = legendre_gauss_nodes_weights(int(order))
    axis_nodes = []
    axis_weights = []
    for basis in product_basis.bases:
        midpoint = 0.5 * (basis.domain.left + basis.domain.right)
        half_length = 0.5 * basis.domain.length
        axis_nodes.append(midpoint + half_length * nodes)
        axis_weights.append(0.5 * weights)
    node_mesh = tf.meshgrid(*axis_nodes, indexing="ij")
    weight_mesh = tf.meshgrid(*axis_weights, indexing="ij")
    points = tf.stack([tf.reshape(axis, [-1]) for axis in node_mesh], axis=1)
    product_weights = tf.ones([tf.shape(points)[0]], dtype=tf.float64)
    for axis_weight in weight_mesh:
        product_weights = product_weights * tf.reshape(axis_weight, [-1])
    return points, product_weights


def _log_reference_density(product_basis: ProductBasis) -> tf.Tensor:
    value = tf.constant(0.0, dtype=tf.float64)
    for basis in product_basis.bases:
        value = value - tf.math.log(tf.constant(basis.domain.length, tf.float64))
    return value


def _density_from_fit(
    fit_result: object,
    config: FixedBranchFilterConfig,
) -> SquaredTTDensity:
    basis = _required_basis(config)
    defensive = TensorProductReferenceDensity(basis, config.measure_convention)
    tau = tf.constant(config.density_tau, dtype=tf.float64)
    normalizer_floor = tf.constant(config.normalizer_floor, dtype=tf.float64)
    denominator_floor = tf.constant(config.denominator_floor, dtype=tf.float64)
    identity = SquaredTTDensity.expected_branch_identity(
        sqrt_tt=fit_result.fitted_tt,
        defensive_density=defensive,
        tau=tau,
        normalizer_floor=normalizer_floor,
        denominator_floor=denominator_floor,
        measure_convention=config.measure_convention,
    )
    return SquaredTTDensity(
        sqrt_tt=fit_result.fitted_tt,
        defensive_density=defensive,
        tau=tau,
        normalizer_floor=normalizer_floor,
        denominator_floor=denominator_floor,
        measure_convention=config.measure_convention,
        branch_identity=identity,
    )


def _marginal_mass(
    density: SquaredTTDensity,
    keep_axes: tuple[int, ...],
) -> tf.Tensor:
    basis = ProductBasis(
        [density.sqrt_tt.product_basis.bases[axis] for axis in keep_axes],
        density.measure_convention,
    )
    points, weights = _reference_quadrature(basis, 65)
    values = density.normalized_marginal_density_values(keep_axes, points)
    return tf.reduce_sum(weights * values)


def _fit_update_structure(fit_result: object) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            update.get("sweep_index"),
            update.get("core_index"),
            update.get("status"),
        )
        for update in fit_result.core_update_statuses
    )


def _compatibility_hash(
    *,
    model: TFHighDimStateSpaceModel,
    observation_matrix: tf.Tensor,
    config: ScalarAdjacentTTConfig,
    branch_seed_prefix: str,
    steps: Sequence[ScalarAdjacentTTStep],
) -> str:
    return fixed_branch_compatibility_hash(
        {
            "route_id": ROUTE_ID,
            "route_classification": ROUTE_CLASSIFICATION,
            "route_subtype": ROUTE_SUBTYPE,
            "model": model.manifest_payload(),
            "parameter_dim": int(model.parameter_dim()),
            "state_dim": int(model.state_dim()),
            "observation_dim": int(model.observation_dim()),
            "observation_shape": tuple(int(dim) for dim in observation_matrix.shape),
            "config": _compatibility_config_payload(config),
            "branch_seed_prefix": branch_seed_prefix,
            "step_count": len(steps),
            "step_target_kinds": tuple(step.target_kind for step in steps),
            "step_axis_orders": tuple(step.diagnostics["axis_order"] for step in steps),
            "step_integrated_axes": tuple(
                step.diagnostics["integrated_axes"] for step in steps
            ),
            "step_log_scale_shift_indices": tuple(
                step.log_scale_shift_index for step in steps
            ),
            "step_fit_update_structures": tuple(
                step.diagnostics["fit_update_structure"] for step in steps
            ),
        }
    )


def _compatibility_config_payload(
    config: ScalarAdjacentTTConfig,
) -> Mapping[str, object]:
    return {
        "route_id": ROUTE_ID,
        "route_classification": ROUTE_CLASSIFICATION,
        "route_subtype": ROUTE_SUBTYPE,
        "axis_order": AXIS_ORDER,
        "initializer_id": INITIALIZER_ID,
        "transition_before_first_observation": config.transition_before_first_observation,
        "initial": _fixed_filter_compatibility_payload(config.initial),
        "adjacent": _fixed_filter_compatibility_payload(config.adjacent),
        "scalar_coordinate_map": config.scalar_coordinate_map.manifest_payload(),
    }


def _fixed_filter_compatibility_payload(
    config: FixedBranchFilterConfig,
) -> Mapping[str, object]:
    fit = config.fit_config
    if fit is None:
        raise ValueError("fixed filter compatibility requires fit_config")
    return {
        "density_tau": config.density_tau,
        "normalizer_floor": config.normalizer_floor,
        "denominator_floor": config.denominator_floor,
        "coordinate_maps": tuple(
            coordinate_map.manifest_payload()
            for coordinate_map in config.coordinate_maps
        ),
        "measure_convention": {
            "density_measure": config.measure_convention.density_measure.value,
            "mass_measure": config.measure_convention.mass_measure.value,
            "reference_weight_name": config.measure_convention.reference_weight_name,
        },
        "deterministic_seed": config.deterministic_seed,
        "product_basis": _required_basis(config).manifest_payload(),
        "initial_cores": tuple(
            core.values for core in _required_initial_cores(config)
        ),
        "fit_quadrature_order": config.fit_quadrature_order,
        "fit": {
            "ranks": fit.ranks,
            "ridge": fit.ridge,
            "max_sweeps": fit.max_sweeps,
            "sweep_order": fit.sweep_order,
            "row_budget": fit.row_budget,
            "column_budget": fit.column_budget,
            "dense_matrix_byte_budget": fit.dense_matrix_byte_budget,
            "normal_matrix_byte_budget": fit.normal_matrix_byte_budget,
            "condition_number_warning": fit.condition_number_warning,
            "condition_number_veto": fit.condition_number_veto,
            "holdout_tolerance": fit.holdout_tolerance,
            "stabilization_policy_id": fit.stabilization_policy_id,
            "solver_backend": fit.solver_backend,
            "column_scale_floor": fit.column_scale_floor,
        },
    }


__all__ = [
    "AXIS_ORDER",
    "INITIALIZER_ID",
    "ROUTE_CLASSIFICATION",
    "ROUTE_ID",
    "ROUTE_SUBTYPE",
    "ScalarAdjacentTTConfig",
    "ScalarAdjacentTTResult",
    "ScalarAdjacentTTStep",
    "norm_balanced_initial_cores",
    "scalar_adjacent_state_fixed_tt_score",
    "scalar_adjacent_state_fixed_tt_value",
]
