"""Transition-route contracts for memory-bounded spatial SIR filtering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import tensorflow as tf


P52_ALLOWED_TRANSITION_INTERFACES = frozenset(
    {
        "streamed_local_neighborhood",
        "tt_mpo_factorized_contraction",
        "blocked_factorized_contraction",
    }
)
P52_FORBIDDEN_TRANSITION_INTERFACES = frozenset(
    {
        "pairwise_grid_transition",
        "all_grid_pairwise_transition",
        "multistate_grid_pairwise_transition",
        "dense_all_pairs_retained_grid",
    }
)

P53_LOWER_RUNG_ROUTE_CLASS = "lower_rung_dense_equivalent"
P53_LOWER_RUNG_CLAIM_CLASS = "interface_tieout_only_not_scaling"
P53_LOWER_RUNG_ROUTE_NONCLAIMS = (
    "no lower-rung dense equivalence proof",
    "no scaling-route readiness",
    "no d=18 spatial SIR readiness",
    "no d=50 or d=100 spatial SIR readiness",
    "no filtering correctness",
    "no HMC readiness",
)
P53_SCALING_ROUTE_CLASS = "scaling_route"
P53_LOCAL_SCALING_ROUTE_ID = "p53_spatial_sir_local_neighborhood_contraction"
P53_LOCAL_SCALING_SELECTED_DESIGN = "local-neighborhood sparse transition contraction"
P53_LOCAL_SCALING_ROUTE_NONCLAIMS = (
    "no lower-rung dense tie-out yet",
    "no scaling-route admission yet",
    "no rank-selection readiness",
    "no d=18 spatial SIR readiness",
    "no d=50 or d=100 spatial SIR readiness",
    "no HMC readiness",
    "no GPU readiness",
)


@dataclass(frozen=True)
class FactorizedTransitionRouteContract:
    """Contract for a production-eligible spatial SIR transition route."""

    route_id: str
    transition_interface: str
    materializes_dense_pairs: bool
    deterministic_replay: bool
    differentiable_backend: str
    exposes_reff_bound: bool
    effective_transition_rank_multiplier: int | None
    memory_metadata_available: bool
    status: str
    blocker: str | None = None
    claim_class: str = "route_contract_not_filtering_correctness"

    def __post_init__(self) -> None:
        if not str(self.route_id).strip():
            raise ValueError("route_id must be nonempty")
        interface = str(self.transition_interface)
        if interface in P52_FORBIDDEN_TRANSITION_INTERFACES:
            raise ValueError("dense all-pairs transition interface is forbidden")
        if interface not in P52_ALLOWED_TRANSITION_INTERFACES:
            raise ValueError("unknown transition interface")
        if self.differentiable_backend != "tensorflow":
            raise ValueError("differentiable_backend must be tensorflow")
        if self.claim_class != "route_contract_not_filtering_correctness":
            raise ValueError("route contract cannot promote filtering correctness")
        if self.status not in {
            "PASS_P52_FACTORIZED_ROUTE_CONTRACT",
            "BLOCK_P52_FACTORIZED_ROUTE_IMPLEMENTATION_MISSING",
        }:
            raise ValueError("unknown route contract status")
        if self.status.startswith("PASS"):
            if self.blocker is not None:
                raise ValueError("passing route contract cannot carry a blocker")
            if self.materializes_dense_pairs:
                raise ValueError("passing route cannot materialize dense pairs")
            if not self.deterministic_replay:
                raise ValueError("passing route must preserve deterministic replay")
            if not self.exposes_reff_bound:
                raise ValueError("passing route must expose an R_eff bound")
            if self.effective_transition_rank_multiplier is None:
                raise ValueError("passing route must declare an R_eff bound")
            if int(self.effective_transition_rank_multiplier) <= 0:
                raise ValueError("effective_transition_rank_multiplier must be positive")
            if not self.memory_metadata_available:
                raise ValueError("passing route must expose memory metadata")
        else:
            if not self.blocker:
                raise ValueError("blocking route contract requires a blocker")
        object.__setattr__(self, "route_id", str(self.route_id))
        object.__setattr__(self, "transition_interface", interface)

    def manifest_payload(self) -> Mapping[str, object]:
        return {
            "route_id": self.route_id,
            "transition_interface": self.transition_interface,
            "materializes_dense_pairs": self.materializes_dense_pairs,
            "deterministic_replay": self.deterministic_replay,
            "differentiable_backend": self.differentiable_backend,
            "exposes_reff_bound": self.exposes_reff_bound,
            "effective_transition_rank_multiplier": (
                self.effective_transition_rank_multiplier
            ),
            "memory_metadata_available": self.memory_metadata_available,
            "status": self.status,
            "blocker": self.blocker,
            "claim_class": self.claim_class,
            "nonclaims": (
                "no filtering correctness",
                "no production spatial SIR readiness",
                "no HMC readiness",
            ),
        }


def p52_current_spatial_sir_route_blocker_manifest() -> Mapping[str, object]:
    """Return the M4 blocker for the current dense all-pairs implementation."""

    blocked_route = {
        "route_id": "current_multistate_retained_grid",
        "transition_interface": "multistate_grid_pairwise_transition",
        "materializes_dense_pairs": True,
        "blocked_functions": (
            "multistate_nonlinear_transition_adjacent_target_batch",
            "_multistate_pairwise_transition_between_grids_log_density",
        ),
        "blocked_operations": (
            "tf.repeat(current, repeats=previous_count, axis=0)",
            "tf.tile(previous, [current_count, 1])",
        ),
        "blocker": "BLOCK_P52_FACTORIZED_ROUTE_IMPLEMENTATION_MISSING",
    }
    required_contract = FactorizedTransitionRouteContract(
        route_id="required_spatial_sir_factorized_transition_route",
        transition_interface="tt_mpo_factorized_contraction",
        materializes_dense_pairs=False,
        deterministic_replay=True,
        differentiable_backend="tensorflow",
        exposes_reff_bound=False,
        effective_transition_rank_multiplier=None,
        memory_metadata_available=False,
        status="BLOCK_P52_FACTORIZED_ROUTE_IMPLEMENTATION_MISSING",
        blocker=(
            "route interface specified but no implemented TensorFlow factorized "
            "transition application with R_eff and memory metadata exists yet"
        ),
    )
    return {
        "schema_version": "p52.factorized_transition_route.v1",
        "phase": "P52-M4",
        "status": "BLOCK_P52_FACTORIZED_TRANSITION_ROUTE",
        "current_route": blocked_route,
        "required_contract": dict(required_contract.manifest_payload()),
        "repair_required": (
            "implement streamed/local or TT-MPO factorized TensorFlow transition "
            "application before d=18 spatial SIR filtering can run"
        ),
        "tokens_emitted": ("BLOCK_P52_FACTORIZED_TRANSITION_ROUTE",),
        "nonclaims": (
            "no factorized route implementation",
            "no d=18 spatial SIR filtering",
            "no filtering correctness",
            "no production spatial SIR readiness",
            "no HMC readiness",
        ),
    }


@dataclass(frozen=True)
class LowerRungStreamingRouteConfig:
    """Configuration for the P53 lower-rung streaming dense-equivalent route.

    The route is exact for a retained finite grid up to floating-point
    reduction order, but it is still dense-equivalent because every current row
    sums over every previous retained row.  It is therefore an interface and
    tie-out route, not a scaling route.
    """

    current_block_size: int = 8
    previous_block_size: int = 64
    route_id: str = "p53_lower_rung_streaming_dense_equivalent"

    def __post_init__(self) -> None:
        if int(self.current_block_size) <= 0:
            raise ValueError("current_block_size must be positive")
        if int(self.previous_block_size) <= 0:
            raise ValueError("previous_block_size must be positive")
        if not str(self.route_id).strip():
            raise ValueError("route_id must be nonempty")
        object.__setattr__(self, "current_block_size", int(self.current_block_size))
        object.__setattr__(self, "previous_block_size", int(self.previous_block_size))
        object.__setattr__(self, "route_id", str(self.route_id))

    def manifest_payload(self) -> Mapping[str, object]:
        return {
            "route_id": self.route_id,
            "route_class": P53_LOWER_RUNG_ROUTE_CLASS,
            "claim_class": P53_LOWER_RUNG_CLAIM_CLASS,
            "current_block_size": int(self.current_block_size),
            "previous_block_size": int(self.previous_block_size),
            "materializes_full_dense_pairs": False,
            "uses_tensorflow_backend": True,
            "nonclaims": P53_LOWER_RUNG_ROUTE_NONCLAIMS,
        }


@dataclass(frozen=True)
class LowerRungStreamingRouteMetadata:
    """Replay and memory metadata emitted by the P53 lower-rung route."""

    route_id: str
    current_count: int
    previous_count: int
    state_dim: int
    time_index: int
    current_block_size: int
    previous_block_size: int
    max_transition_rows_per_call: int
    route_width_proxy: int
    memory_forecast_bytes: int
    replay_identity: Mapping[str, object]
    route_class: str = P53_LOWER_RUNG_ROUTE_CLASS
    claim_class: str = P53_LOWER_RUNG_CLAIM_CLASS
    materializes_full_dense_pairs: bool = False
    differentiable_backend: str = "tensorflow"
    exposes_reff_bound: bool = False
    effective_transition_rank_multiplier: int | None = None
    nonclaims: tuple[str, ...] = P53_LOWER_RUNG_ROUTE_NONCLAIMS

    def __post_init__(self) -> None:
        if self.route_class != P53_LOWER_RUNG_ROUTE_CLASS:
            raise ValueError("lower-rung route metadata has wrong route_class")
        if self.claim_class != P53_LOWER_RUNG_CLAIM_CLASS:
            raise ValueError("lower-rung route metadata has wrong claim_class")
        if self.materializes_full_dense_pairs:
            raise ValueError("lower-rung streaming route cannot materialize full dense pairs")
        if self.differentiable_backend != "tensorflow":
            raise ValueError("lower-rung route backend must be tensorflow")
        if self.exposes_reff_bound:
            raise ValueError("lower-rung dense-equivalent route cannot expose an R_eff bound")
        if self.effective_transition_rank_multiplier is not None:
            raise ValueError("lower-rung dense-equivalent route cannot declare an R_eff multiplier")
        for name in (
            "current_count",
            "previous_count",
            "state_dim",
            "current_block_size",
            "previous_block_size",
            "max_transition_rows_per_call",
            "route_width_proxy",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if int(self.memory_forecast_bytes) <= 0:
            raise ValueError("memory_forecast_bytes must be positive")
        object.__setattr__(self, "route_id", str(self.route_id))
        object.__setattr__(self, "current_count", int(self.current_count))
        object.__setattr__(self, "previous_count", int(self.previous_count))
        object.__setattr__(self, "state_dim", int(self.state_dim))
        object.__setattr__(self, "time_index", int(self.time_index))
        object.__setattr__(self, "current_block_size", int(self.current_block_size))
        object.__setattr__(self, "previous_block_size", int(self.previous_block_size))
        object.__setattr__(
            self,
            "max_transition_rows_per_call",
            int(self.max_transition_rows_per_call),
        )
        object.__setattr__(self, "route_width_proxy", int(self.route_width_proxy))
        object.__setattr__(self, "memory_forecast_bytes", int(self.memory_forecast_bytes))
        object.__setattr__(self, "replay_identity", dict(self.replay_identity))
        object.__setattr__(self, "nonclaims", tuple(self.nonclaims))

    def manifest_payload(self) -> Mapping[str, object]:
        return {
            "route_id": self.route_id,
            "route_class": self.route_class,
            "claim_class": self.claim_class,
            "materializes_full_dense_pairs": self.materializes_full_dense_pairs,
            "differentiable_backend": self.differentiable_backend,
            "exposes_reff_bound": self.exposes_reff_bound,
            "effective_transition_rank_multiplier": self.effective_transition_rank_multiplier,
            "current_count": int(self.current_count),
            "previous_count": int(self.previous_count),
            "state_dim": int(self.state_dim),
            "time_index": int(self.time_index),
            "current_block_size": int(self.current_block_size),
            "previous_block_size": int(self.previous_block_size),
            "max_transition_rows_per_call": int(self.max_transition_rows_per_call),
            "route_width_proxy": int(self.route_width_proxy),
            "memory_forecast_bytes": int(self.memory_forecast_bytes),
            "replay_identity": dict(self.replay_identity),
            "nonclaims": tuple(self.nonclaims),
        }


@dataclass(frozen=True)
class LowerRungStreamingRouteResult:
    """Predictive log-density result from the P53 lower-rung route."""

    predictive_log_density: tf.Tensor
    metadata: LowerRungStreamingRouteMetadata

    def __post_init__(self) -> None:
        values = tf.convert_to_tensor(self.predictive_log_density, dtype=tf.float64)
        if values.shape.rank != 1:
            raise ValueError("predictive_log_density must be a vector")
        object.__setattr__(self, "predictive_log_density", values)
        if not isinstance(self.metadata, LowerRungStreamingRouteMetadata):
            raise TypeError("metadata must be LowerRungStreamingRouteMetadata")


def p53_lower_rung_streaming_route_manifest(
    config: LowerRungStreamingRouteConfig | None = None,
) -> Mapping[str, object]:
    """Return the static P53-M2 route manifest without running a model."""

    route_config = LowerRungStreamingRouteConfig() if config is None else config
    return {
        "schema_version": "p53.lower_rung_streaming_route.v1",
        "phase": "P53-M2",
        "status": "P53_M2_ROUTE_IMPLEMENTATION_AVAILABLE_PENDING_REVIEW",
        "route": dict(route_config.manifest_payload()),
        "tokens_emitted": (),
        "nonclaims": P53_LOWER_RUNG_ROUTE_NONCLAIMS,
    }


def lower_rung_streaming_predictive_log_density(
    model: object,
    theta: tf.Tensor,
    current_physical_points: tf.Tensor,
    previous_physical_points: tf.Tensor,
    previous_log_terms: tf.Tensor,
    time_index: int,
    config: LowerRungStreamingRouteConfig | None = None,
) -> LowerRungStreamingRouteResult:
    """Apply the P53 lower-rung streaming dense-equivalent transition route.

    The returned vector has one entry per current point:

    ``logsumexp_j previous_log_terms[j] + log p(x_current_i | x_previous_j)``.

    The implementation streams over current/previous blocks and evaluates one
    current row against a bounded previous block at a time.  It therefore
    avoids the old full-grid pair materialization while preserving TensorFlow
    autodiff through the transition density.
    """

    route_config = LowerRungStreamingRouteConfig() if config is None else config
    state_dim = int(model.state_dim())
    current = _p53_route_matrix(current_physical_points, state_dim, "current_physical_points")
    previous = _p53_route_matrix(previous_physical_points, state_dim, "previous_physical_points")
    previous_terms = _p53_route_vector(previous_log_terms, "previous_log_terms")
    theta_tensor = tf.convert_to_tensor(theta, dtype=tf.float64)
    current_count = _p53_static_row_count(current, "current_physical_points")
    previous_count = _p53_static_row_count(previous, "previous_physical_points")
    if previous_terms.shape[0] != previous_count:
        raise ValueError("previous_log_terms length must match previous_physical_points")

    current_blocks = []
    for current_start in range(0, current_count, route_config.current_block_size):
        current_stop = min(current_count, current_start + route_config.current_block_size)
        current_block = current[current_start:current_stop]
        current_block_count = int(current_block.shape[0])
        accumulated = tf.fill(
            [current_block_count],
            tf.constant(float("-inf"), dtype=tf.float64),
        )
        for previous_start in range(0, previous_count, route_config.previous_block_size):
            previous_stop = min(previous_count, previous_start + route_config.previous_block_size)
            previous_block = previous[previous_start:previous_stop]
            previous_term_block = previous_terms[previous_start:previous_stop]
            previous_block_count = int(previous_block.shape[0])
            candidate_rows = []
            for row_index in range(current_block_count):
                next_rows = tf.broadcast_to(
                    current_block[row_index : row_index + 1],
                    [previous_block_count, state_dim],
                )
                transition_log = model.transition_log_density(
                    theta_tensor,
                    previous_block,
                    next_rows,
                    t=int(time_index),
                )
                transition_log = tf.reshape(
                    tf.convert_to_tensor(transition_log, dtype=tf.float64),
                    [previous_block_count],
                )
                candidate_rows.append(
                    tf.reduce_logsumexp(previous_term_block + transition_log)
                )
            block_candidate = tf.stack(candidate_rows)
            accumulated = _p53_logaddexp_vectors(accumulated, block_candidate)
        current_blocks.append(accumulated)
    predictive = tf.concat(current_blocks, axis=0)
    metadata = _p53_lower_rung_metadata(
        config=route_config,
        current_count=current_count,
        previous_count=previous_count,
        state_dim=state_dim,
        time_index=int(time_index),
    )
    return LowerRungStreamingRouteResult(
        predictive_log_density=predictive,
        metadata=metadata,
    )


def _p53_lower_rung_metadata(
    config: LowerRungStreamingRouteConfig,
    current_count: int,
    previous_count: int,
    state_dim: int,
    time_index: int,
) -> LowerRungStreamingRouteMetadata:
    max_current = min(int(config.current_block_size), int(current_count))
    max_previous = min(int(config.previous_block_size), int(previous_count))
    float_bytes = tf.float64.size
    memory_forecast_bytes = int(
        float_bytes
        * (
            max_current * int(state_dim)
            + 2 * max_previous * int(state_dim)
            + 3 * max_previous
            + 3 * max_current
        )
    )
    replay_identity = {
        "formula": "logsumexp_j(previous_log_terms[j] + transition_log_density(previous_j,current_i,t))",
        "block_order": "current-major then previous-major",
        "reduction": "streaming_logaddexp_over_previous_blocks",
        "route_class": P53_LOWER_RUNG_ROUTE_CLASS,
        "claim_class": P53_LOWER_RUNG_CLAIM_CLASS,
    }
    return LowerRungStreamingRouteMetadata(
        route_id=config.route_id,
        current_count=int(current_count),
        previous_count=int(previous_count),
        state_dim=int(state_dim),
        time_index=int(time_index),
        current_block_size=int(config.current_block_size),
        previous_block_size=int(config.previous_block_size),
        max_transition_rows_per_call=max_previous,
        route_width_proxy=int(previous_count),
        memory_forecast_bytes=memory_forecast_bytes,
        replay_identity=replay_identity,
    )


def _p53_route_matrix(value: tf.Tensor, state_dim: int, name: str) -> tf.Tensor:
    tensor = tf.convert_to_tensor(value, dtype=tf.float64)
    if tensor.shape.rank != 2 or tensor.shape[1] != int(state_dim):
        raise ValueError(f"{name} must have shape [n, {int(state_dim)}]")
    _p53_static_row_count(tensor, name)
    return tensor


def _p53_route_vector(value: tf.Tensor, name: str) -> tf.Tensor:
    tensor = tf.convert_to_tensor(value, dtype=tf.float64)
    if tensor.shape.rank != 1 or tensor.shape[0] is None:
        raise ValueError(f"{name} must be a statically shaped vector")
    return tensor


def _p53_static_row_count(tensor: tf.Tensor, name: str) -> int:
    if tensor.shape.rank != 2 or tensor.shape[0] is None:
        raise ValueError(f"{name} must have statically known row count")
    rows = int(tensor.shape[0])
    if rows <= 0:
        raise ValueError(f"{name} must have positive row count")
    return rows


def _p53_logaddexp_vectors(left: tf.Tensor, right: tf.Tensor) -> tf.Tensor:
    left_tensor = tf.convert_to_tensor(left, dtype=tf.float64)
    right_tensor = tf.convert_to_tensor(right, dtype=tf.float64)
    if left_tensor.shape != right_tensor.shape:
        raise ValueError("logaddexp vector shapes must match")
    return tf.reduce_logsumexp(tf.stack([left_tensor, right_tensor], axis=0), axis=0)


@dataclass(frozen=True)
class LocalNeighborhoodScalingRouteConfig:
    """Configuration for the P53-M4B local-neighborhood scaling route."""

    basis_order: int
    tt_rank_left: int = 1
    tt_rank_right: int = 1
    memory_cap_bytes: int = 32 * 1024**3
    route_id: str = P53_LOCAL_SCALING_ROUTE_ID
    dtype_name: str = "float64"
    branch_id: str = "p53-m4b-unfitted-local-route"

    def __post_init__(self) -> None:
        for name in ("basis_order", "tt_rank_left", "tt_rank_right", "memory_cap_bytes"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.route_id != P53_LOCAL_SCALING_ROUTE_ID:
            raise ValueError("P53-M4B local route_id mismatch")
        if self.dtype_name != "float64":
            raise ValueError("P53-M4B local route requires float64")
        if not str(self.branch_id).strip():
            raise ValueError("branch_id must be nonempty")
        object.__setattr__(self, "basis_order", int(self.basis_order))
        object.__setattr__(self, "tt_rank_left", int(self.tt_rank_left))
        object.__setattr__(self, "tt_rank_right", int(self.tt_rank_right))
        object.__setattr__(self, "memory_cap_bytes", int(self.memory_cap_bytes))
        object.__setattr__(self, "branch_id", str(self.branch_id))


@dataclass(frozen=True)
class LocalNeighborhoodScalingRouteMetadata:
    """Replay and route-width metadata for the P53-M4B scaling route."""

    route_id: str
    route_class: str
    selected_design: str
    rk4_substeps: int
    dependency_neighborhoods: tuple[tuple[int, ...], ...]
    max_dependency_width: int
    basis_order: int
    tt_rank_metadata: Mapping[str, int]
    R_eff: int
    memory_forecast_bytes: int
    covariance_scope: str
    dtype: str
    branch_id: str
    replay_identity: Mapping[str, object]
    nonclaims: tuple[str, ...] = P53_LOCAL_SCALING_ROUTE_NONCLAIMS

    def __post_init__(self) -> None:
        if self.route_id != P53_LOCAL_SCALING_ROUTE_ID:
            raise ValueError("local scaling metadata route_id mismatch")
        if self.route_class != P53_SCALING_ROUTE_CLASS:
            raise ValueError("local scaling metadata route_class mismatch")
        if self.selected_design != P53_LOCAL_SCALING_SELECTED_DESIGN:
            raise ValueError("local scaling metadata selected_design mismatch")
        if self.covariance_scope != "diagonal_process_covariance":
            raise ValueError("local scaling route is exact only for diagonal process covariance")
        for name in ("rk4_substeps", "max_dependency_width", "basis_order", "R_eff", "memory_forecast_bytes"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        object.__setattr__(
            self,
            "dependency_neighborhoods",
            tuple(tuple(int(index) for index in row) for row in self.dependency_neighborhoods),
        )
        object.__setattr__(self, "tt_rank_metadata", dict(self.tt_rank_metadata))
        object.__setattr__(self, "replay_identity", dict(self.replay_identity))
        object.__setattr__(self, "nonclaims", tuple(self.nonclaims))

    def manifest_payload(self) -> Mapping[str, object]:
        return {
            "route_id": self.route_id,
            "route_class": self.route_class,
            "selected_design": self.selected_design,
            "rk4_substeps": int(self.rk4_substeps),
            "dependency_neighborhoods": self.dependency_neighborhoods,
            "max_dependency_width": int(self.max_dependency_width),
            "basis_order": int(self.basis_order),
            "tt_rank_metadata": dict(self.tt_rank_metadata),
            "R_eff": int(self.R_eff),
            "memory_forecast_bytes": int(self.memory_forecast_bytes),
            "covariance_scope": self.covariance_scope,
            "dtype": self.dtype,
            "branch_id": self.branch_id,
            "replay_identity": dict(self.replay_identity),
            "nonclaims": tuple(self.nonclaims),
        }


@dataclass(frozen=True)
class LocalNeighborhoodCoordinateFactorResult:
    """One coordinate's local transition-factor values and route metadata."""

    coordinate_index: int
    log_factor: tf.Tensor
    dependency_neighborhood: tuple[int, ...]
    metadata: LocalNeighborhoodScalingRouteMetadata

    def __post_init__(self) -> None:
        log_factor = tf.convert_to_tensor(self.log_factor, dtype=tf.float64)
        if log_factor.shape.rank != 2:
            raise ValueError("log_factor must have shape [current_values, previous_rows]")
        if not isinstance(self.metadata, LocalNeighborhoodScalingRouteMetadata):
            raise TypeError("metadata must be LocalNeighborhoodScalingRouteMetadata")
        object.__setattr__(self, "coordinate_index", int(self.coordinate_index))
        object.__setattr__(self, "log_factor", log_factor)
        object.__setattr__(
            self,
            "dependency_neighborhood",
            tuple(int(index) for index in self.dependency_neighborhood),
        )


@dataclass(frozen=True)
class LocalNeighborhoodPredictiveResult:
    """Lower-rung tie-out result assembled from local coordinate factors."""

    transition_log_density: tf.Tensor
    predictive_log_density: tf.Tensor
    metadata: LocalNeighborhoodScalingRouteMetadata
    tieout_adapter_scope: str = "lower_rung_diagnostic_not_production_contraction"

    def __post_init__(self) -> None:
        transition = tf.convert_to_tensor(self.transition_log_density, dtype=tf.float64)
        predictive = tf.convert_to_tensor(self.predictive_log_density, dtype=tf.float64)
        if transition.shape.rank != 2:
            raise ValueError("transition_log_density must have shape [current, previous]")
        if predictive.shape.rank != 1:
            raise ValueError("predictive_log_density must be a vector")
        if transition.shape[0] != predictive.shape[0]:
            raise ValueError("transition and predictive current dimensions must match")
        if not isinstance(self.metadata, LocalNeighborhoodScalingRouteMetadata):
            raise TypeError("metadata must be LocalNeighborhoodScalingRouteMetadata")
        object.__setattr__(self, "transition_log_density", transition)
        object.__setattr__(self, "predictive_log_density", predictive)
        object.__setattr__(self, "tieout_adapter_scope", str(self.tieout_adapter_scope))


def spatial_sir_local_scaling_route_metadata(
    model: object,
    config: LocalNeighborhoodScalingRouteConfig,
) -> LocalNeighborhoodScalingRouteMetadata:
    """Build P53-M4B metadata for the selected local-neighborhood route."""

    structural_model = _local_route_structural_model(model)
    _assert_diagonal_process_covariance(structural_model)
    state_dim = int(structural_model.state_dim())
    neighborhoods = _spatial_sir_coordinate_reachability_neighborhoods(structural_model)
    if len(neighborhoods) != state_dim:
        raise ValueError("dependency_neighborhoods length mismatch")
    max_width = max(len(row) for row in neighborhoods)
    r_eff = int(config.basis_order) ** int(max_width)
    r_eff *= int(config.tt_rank_left) * int(config.tt_rank_right)
    memory_forecast_bytes = int(
        tf.float64.size
        * state_dim
        * (int(config.basis_order) ** int(max_width))
        * int(config.tt_rank_left)
        * int(config.tt_rank_right)
    )
    if memory_forecast_bytes > int(config.memory_cap_bytes):
        raise ValueError("local scaling route memory forecast exceeds cap")
    tt_rank_metadata = {
        "left": int(config.tt_rank_left),
        "right": int(config.tt_rank_right),
    }
    replay_identity = {
        "route_id": config.route_id,
        "route_class": P53_SCALING_ROUTE_CLASS,
        "selected_design": P53_LOCAL_SCALING_SELECTED_DESIGN,
        "rk4_substeps": int(getattr(structural_model, "_rk4_substeps")),
        "dependency_neighborhoods": neighborhoods,
        "basis_order": int(config.basis_order),
        "tt_rank_metadata": tt_rank_metadata,
        "R_eff": int(r_eff),
        "memory_forecast_bytes": int(memory_forecast_bytes),
        "covariance_scope": "diagonal_process_covariance",
        "dtype": config.dtype_name,
        "branch_id": config.branch_id,
    }
    return LocalNeighborhoodScalingRouteMetadata(
        route_id=config.route_id,
        route_class=P53_SCALING_ROUTE_CLASS,
        selected_design=P53_LOCAL_SCALING_SELECTED_DESIGN,
        rk4_substeps=int(getattr(structural_model, "_rk4_substeps")),
        dependency_neighborhoods=neighborhoods,
        max_dependency_width=max_width,
        basis_order=int(config.basis_order),
        tt_rank_metadata=tt_rank_metadata,
        R_eff=r_eff,
        memory_forecast_bytes=memory_forecast_bytes,
        covariance_scope="diagonal_process_covariance",
        dtype=config.dtype_name,
        branch_id=config.branch_id,
        replay_identity=replay_identity,
    )


def spatial_sir_local_coordinate_log_factor(
    model: object,
    theta: tf.Tensor,
    previous_physical_points: tf.Tensor,
    current_coordinate_values: tf.Tensor,
    coordinate_index: int,
    config: LocalNeighborhoodScalingRouteConfig,
) -> LocalNeighborhoodCoordinateFactorResult:
    """Evaluate one local current-coordinate transition factor.

    This is the M4B scaling-route primitive.  It evaluates
    ``log Normal(x_a; Phi_a(z), sigma_a^2)`` for one current coordinate
    against previous retained rows.  It does not enumerate global
    current-state/previous-state pairs.
    """

    metadata = spatial_sir_local_scaling_route_metadata(model, config)
    structural_model = _local_route_structural_model(model)
    state_dim = int(structural_model.state_dim())
    coordinate = int(coordinate_index)
    if coordinate < 0 or coordinate >= state_dim:
        raise ValueError("coordinate_index out of range")
    previous = _p53_route_matrix(previous_physical_points, state_dim, "previous_physical_points")
    current_values = tf.convert_to_tensor(current_coordinate_values, dtype=tf.float64)
    if current_values.shape.rank != 1 or current_values.shape[0] is None:
        raise ValueError("current_coordinate_values must be a statically shaped vector")
    transition_mean = tf.convert_to_tensor(
        _transition_mean_for_local_route(model, theta, previous),
        dtype=tf.float64,
    )
    mean_coordinate = transition_mean[:, coordinate]
    covariance_diag = tf.linalg.diag_part(
        tf.convert_to_tensor(structural_model.process_covariance, dtype=tf.float64)
    )
    variance = covariance_diag[coordinate]
    if not bool(tf.math.is_finite(variance).numpy()) or bool((variance <= 0.0).numpy()):
        raise ValueError("process covariance diagonal entry must be positive")
    residual = current_values[:, tf.newaxis] - mean_coordinate[tf.newaxis, :]
    log_two_pi = tf.math.log(tf.constant(2.0 * 3.141592653589793, dtype=tf.float64))
    log_factor = -0.5 * (log_two_pi + tf.math.log(variance) + tf.square(residual) / variance)
    return LocalNeighborhoodCoordinateFactorResult(
        coordinate_index=coordinate,
        log_factor=log_factor,
        dependency_neighborhood=metadata.dependency_neighborhoods[coordinate],
        metadata=metadata,
    )


def spatial_sir_local_pairwise_transition_log_density(
    model: object,
    theta: tf.Tensor,
    current_physical_points: tf.Tensor,
    previous_physical_points: tf.Tensor,
    config: LocalNeighborhoodScalingRouteConfig,
) -> LocalNeighborhoodPredictiveResult:
    """Assemble lower-rung pairwise transition logs from local factors.

    This diagnostic adapter supports P53-M4C tie-out on tiny retained grids. It
    validates the local factorization against dense references, but it is not
    the high-dimensional production contraction admitted by later phases.
    """

    state_dim = int(model.state_dim())
    current = _p53_route_matrix(current_physical_points, state_dim, "current_physical_points")
    previous = _p53_route_matrix(previous_physical_points, state_dim, "previous_physical_points")
    local_terms = []
    metadata = spatial_sir_local_scaling_route_metadata(model, config)
    for coordinate in range(state_dim):
        factor = spatial_sir_local_coordinate_log_factor(
            model=model,
            theta=theta,
            previous_physical_points=previous,
            current_coordinate_values=current[:, coordinate],
            coordinate_index=coordinate,
            config=config,
        )
        local_terms.append(factor.log_factor)
    transition = tf.add_n(local_terms)
    empty_predictive = tf.fill(
        [int(current.shape[0])],
        tf.constant(float("nan"), dtype=tf.float64),
    )
    return LocalNeighborhoodPredictiveResult(
        transition_log_density=transition,
        predictive_log_density=empty_predictive,
        metadata=metadata,
    )


def spatial_sir_local_predictive_log_density(
    model: object,
    theta: tf.Tensor,
    current_physical_points: tf.Tensor,
    previous_physical_points: tf.Tensor,
    previous_log_terms: tf.Tensor,
    time_index: int,
    config: LocalNeighborhoodScalingRouteConfig,
) -> LocalNeighborhoodPredictiveResult:
    """Lower-rung predictive log density from local transition factors.

    ``time_index`` is recorded for interface parity with the dense and
    lower-rung streaming routes. The current SpatialSIRSSM transition density
    is time-homogeneous, so this adapter does not otherwise use it.
    """

    del time_index
    previous_terms = _p53_route_vector(previous_log_terms, "previous_log_terms")
    pairwise = spatial_sir_local_pairwise_transition_log_density(
        model=model,
        theta=theta,
        current_physical_points=current_physical_points,
        previous_physical_points=previous_physical_points,
        config=config,
    )
    previous_count = int(pairwise.transition_log_density.shape[1])
    if previous_terms.shape[0] != previous_count:
        raise ValueError("previous_log_terms length must match previous_physical_points")
    predictive = tf.reduce_logsumexp(
        previous_terms[tf.newaxis, :] + pairwise.transition_log_density,
        axis=1,
    )
    return LocalNeighborhoodPredictiveResult(
        transition_log_density=pairwise.transition_log_density,
        predictive_log_density=predictive,
        metadata=pairwise.metadata,
    )


def p53_local_scaling_route_manifest(
    model: object,
    config: LocalNeighborhoodScalingRouteConfig,
) -> Mapping[str, object]:
    """Return a static manifest for the selected P53-M4B scaling route."""

    metadata = spatial_sir_local_scaling_route_metadata(model, config)
    return {
        "schema_version": "p53.local_scaling_route.v1",
        "phase": "P53-M4B",
        "status": "P53_M4B_SCALING_ROUTE_IMPLEMENTATION_AVAILABLE_PENDING_REVIEW",
        "route": dict(metadata.manifest_payload()),
        "tokens_emitted": (),
        "nonclaims": P53_LOCAL_SCALING_ROUTE_NONCLAIMS,
    }


def _assert_diagonal_process_covariance(model: object) -> None:
    covariance = tf.convert_to_tensor(model.process_covariance, dtype=tf.float64)
    if covariance.shape.rank != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("process_covariance must be square")
    diagonal = tf.linalg.diag(tf.linalg.diag_part(covariance))
    if not bool(tf.reduce_all(tf.abs(covariance - diagonal) <= 1e-12).numpy()):
        raise ValueError("local scaling route requires diagonal process covariance")


def _local_route_structural_model(model: object) -> object:
    """Return the theta-independent SIR structure used by the local route."""

    if _is_spatial_sir_structural_model(model):
        return model
    base_model = getattr(model, "base_model", None)
    if base_model is not None and _is_spatial_sir_structural_model(base_model):
        if not callable(getattr(model, "scaled_model", None)):
            raise TypeError("parameterized local route model must expose scaled_model")
        if not callable(getattr(model, "transition_mean", None)):
            raise TypeError("parameterized local route model must expose transition_mean")
        return base_model
    raise TypeError(
        "local scaling route supports SpatialSIRSSM or ParameterizedZhaoCuiSIRSSM-compatible wrappers"
    )


def _is_spatial_sir_structural_model(model: object) -> bool:
    return (
        callable(getattr(model, "state_dim", None))
        and callable(getattr(model, "observation_dim", None))
        and hasattr(model, "process_covariance")
        and hasattr(model, "neighbor_sets")
        and hasattr(model, "_rk4_substeps")
    )


def _transition_mean_for_local_route(
    model: object,
    theta: tf.Tensor,
    previous_physical_points: tf.Tensor,
) -> tf.Tensor:
    """Evaluate the local-route transition mean with explicit wrapper support."""

    if _is_spatial_sir_structural_model(model):
        return model.transition_mean(previous_physical_points)
    structural_model = _local_route_structural_model(model)
    del structural_model
    if callable(getattr(model, "transition_mean", None)):
        return model.transition_mean(theta, previous_physical_points)
    raise TypeError("parameterized local route model must expose transition_mean")


def _spatial_sir_coordinate_reachability_neighborhoods(
    model: object,
) -> tuple[tuple[int, ...], ...]:
    compartments = int(model.observation_dim())
    state_dim = int(model.state_dim())
    site_neighbors = tuple(tuple(int(site) for site in row) for row in model.neighbor_sets)
    one_step_site_dependencies = []
    for site in range(compartments):
        dependencies = {site}
        dependencies.update(site_neighbors[site])
        one_step_site_dependencies.append(dependencies)
    reachability = [set(row) for row in one_step_site_dependencies]
    for _ in range(max(0, int(getattr(model, "_rk4_substeps")) - 1)):
        updated = []
        for dependencies in reachability:
            expanded = set(dependencies)
            for site in dependencies:
                expanded.update(one_step_site_dependencies[site])
            updated.append(expanded)
        reachability = updated
    coordinate_neighborhoods = []
    for coordinate in range(state_dim):
        site = coordinate // 2
        coords = []
        for dependency_site in sorted(reachability[site]):
            coords.extend([2 * dependency_site, 2 * dependency_site + 1])
        coordinate_neighborhoods.append(tuple(coords))
    return tuple(coordinate_neighborhoods)
