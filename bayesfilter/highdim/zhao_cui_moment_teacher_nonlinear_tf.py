"""Nonlinear bootstrap/Contract-E particle filters with a TT moment teacher.

The particle lane owns the finite likelihood.  The independently fitted
adjacent squared-TT lane supplies only post-reset standardized shape targets.
Both lanes use the source-order event sequence ``x0 -> transition -> y1``.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Mapping

import tensorflow as tf

from bayesfilter.highdim.bases import BoundedInterval, LegendreBasis1D
from bayesfilter.highdim.cubature_genut_adapters import (
    exact_transformed_sv_candidate_adapter,
    parameterized_austria_sir_candidate_adapter,
    predator_prey_candidate_adapter,
)
from bayesfilter.highdim.cubature_genut_filter import CandidateModelAdapter
from bayesfilter.highdim.diagnostics import MassMeasure
from bayesfilter.highdim.higher_moment_contract_e import higher_moment_shape_jvp
from bayesfilter.highdim.ledh_contract_e_streaming_tf import (
    _contract_e_streaming_forward_jvp_core,
)
from bayesfilter.highdim.ledh_tuning_scope import LEDHTuningScope, require_scope_match
from bayesfilter.highdim.transport_chunk_policy import (
    TRANSPORT_CHUNK_POLICY_ID,
    select_transport_chunks,
    validate_transport_chunks,
)
from bayesfilter.highdim.zhao_cui_moment_teacher import (
    legendre_monomial_operator_matrix,
)
from bayesfilter.highdim.zhao_cui_moment_teacher_lgssm_tf import (
    MomentTeacherControls,
    _active_mask,
    _legendre_basis_values,
)
from bayesfilter.highdim.zhao_cui_moment_teacher_xla import (
    padded_fixed_teacher_recursion_shape_xla,
)


PREDATOR_PREY_ROUTE_ID = "zhao_cui_moment_teacher_contract_e_chol_predator_prey_v1"
AUSTRIA_SIR_ROUTE_ID = "zhao_cui_moment_teacher_contract_e_chol_austria_sir_v1"
ACTUAL_SV_ROUTE_ID = "zhao_cui_moment_teacher_contract_e_chol_actual_sv_v1"
CONTROL_FAMILY_ID = "zhao_cui_moment_teacher_nonlinear_controls_v1"
RESET_CONTRACT_ID = "contract_e_chol_v1"
TUNING_SCHEMA = "bayesfilter.zhao_cui_moment_teacher_nonlinear_tuning.v1"
EVENT_ORDER = "x0_then_transition_then_observe"
_TWO_PI = 6.283185307179586476925286766559

_ISSUED_TUNING_SEALS: set[object] = set()


@dataclass(frozen=True)
class NonlinearMomentTeacherTuningArtifact:
    """Repository-issued controls for one exact nonlinear execution scope."""

    scope: LEDHTuningScope
    controls: MomentTeacherControls
    calibration_data_id: str
    validation_data_id: str
    selection_record_id: str
    chart_id: str
    pair_set_id: str
    _seal: object

    def __post_init__(self) -> None:
        if self._seal not in _ISSUED_TUNING_SEALS:
            raise TypeError("nonlinear moment-teacher tuning must be repository-issued")
        if self.scope.route_id not in (
            PREDATOR_PREY_ROUTE_ID,
            AUSTRIA_SIR_ROUTE_ID,
            ACTUAL_SV_ROUTE_ID,
        ):
            raise ValueError("nonlinear moment-teacher tuning route is unsupported")
        if self.scope.control_family_id != CONTROL_FAMILY_ID:
            raise ValueError("nonlinear moment-teacher control family mismatch")
        if self.calibration_data_id == self.validation_data_id:
            raise ValueError("calibration and validation data must be disjoint")
        if not all(
            str(item).strip()
            for item in (
                self.calibration_data_id,
                self.validation_data_id,
                self.selection_record_id,
                self.chart_id,
                self.pair_set_id,
            )
        ):
            raise ValueError("nonlinear tuning identity fields must be nonempty")

    @property
    def artifact_id(self) -> str:
        payload = {
            "schema": TUNING_SCHEMA,
            "scope": self.scope.as_dict(),
            "scope_sha256": self.scope.scope_sha256,
            "controls": self.controls.as_dict(),
            "calibration_data_id": self.calibration_data_id,
            "validation_data_id": self.validation_data_id,
            "selection_record_id": self.selection_record_id,
            "chart_id": self.chart_id,
            "pair_set_id": self.pair_set_id,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
        return hashlib.sha256(encoded).hexdigest()

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": TUNING_SCHEMA,
            "artifact_id": self.artifact_id,
            "scope": self.scope.as_dict(),
            "scope_sha256": self.scope.scope_sha256,
            "controls": self.controls.as_dict(),
            "calibration_data_id": self.calibration_data_id,
            "validation_data_id": self.validation_data_id,
            "selection_record_id": self.selection_record_id,
            "chart_id": self.chart_id,
            "pair_set_id": self.pair_set_id,
        }


def issue_nonlinear_moment_teacher_tuning_artifact(
    *,
    scope: LEDHTuningScope,
    controls: MomentTeacherControls,
    calibration_data_id: str,
    validation_data_id: str,
    selection_record_id: str,
    chart_id: str,
    pair_set_id: str,
) -> NonlinearMomentTeacherTuningArtifact:
    if scope.reset_contract_id != RESET_CONTRACT_ID:
        raise ValueError("nonlinear moment-teacher reset contract mismatch")
    seal = object()
    _ISSUED_TUNING_SEALS.add(seal)
    return NonlinearMomentTeacherTuningArtifact(
        scope=scope,
        controls=controls,
        calibration_data_id=str(calibration_data_id),
        validation_data_id=str(validation_data_id),
        selection_record_id=str(selection_record_id),
        chart_id=str(chart_id),
        pair_set_id=str(pair_set_id),
        _seal=seal,
    )


def make_nonlinear_tuning_scope(
    *,
    model_id: str,
    target_id: str,
    route_id: str,
    horizon: int,
    prepared_data_id: str,
    particle_count: int,
    state_dimension: int,
    parameter_count: int,
    dtype: tf.dtypes.DType,
    tf32_enabled: bool,
    jit_compile: bool,
) -> LEDHTuningScope:
    if route_id not in (
        PREDATOR_PREY_ROUTE_ID,
        AUSTRIA_SIR_ROUTE_ID,
        ACTUAL_SV_ROUTE_ID,
    ):
        raise ValueError("unsupported nonlinear moment-teacher route")
    chunks = select_transport_chunks(particle_count)
    return LEDHTuningScope(
        model_id=model_id,
        target_id=target_id,
        route_id=route_id,
        reset_contract_id=RESET_CONTRACT_ID,
        horizon=horizon,
        prepared_data_id=prepared_data_id,
        particle_count=particle_count,
        state_dimension=state_dimension,
        parameter_count=parameter_count,
        dtype=tf.dtypes.as_dtype(dtype).name,
        tf32_enabled=bool(tf32_enabled),
        jit_compile=bool(jit_compile),
        chunk_policy_id=TRANSPORT_CHUNK_POLICY_ID,
        row_chunk_size=chunks.row_chunk_size,
        col_chunk_size=chunks.col_chunk_size,
        row_blocks=chunks.row_blocks,
        col_blocks=chunks.col_blocks,
        control_family_id=CONTROL_FAMILY_ID,
    )


def prepare_nonlinear_teacher_inputs(
    *,
    adapter: CandidateModelAdapter,
    observations: tf.Tensor,
    state_offset: tf.Tensor,
    state_scale: tf.Tensor,
    center_theta: tf.Tensor,
    initial_standard_deviation: float,
    process_standard_deviation: float,
    fit_rows: int,
    basis_size: int,
    rank: int,
    sweeps: int,
    defensive_weight: float,
    pair_indices: tf.Tensor,
    root_seed: int,
) -> dict[str, tf.Tensor]:
    """Prepare a fixed affine adjacent-state TT design in FP32."""

    observations = tf.convert_to_tensor(observations, tf.float32)
    state_offset = tf.reshape(tf.convert_to_tensor(state_offset, tf.float32), [-1])
    state_scale = tf.reshape(tf.convert_to_tensor(state_scale, tf.float32), [-1])
    center_theta = tf.reshape(tf.convert_to_tensor(center_theta, tf.float32), [-1])
    pair_indices = tf.convert_to_tensor(pair_indices, tf.int32)
    state_dimension = int(state_offset.shape[0])
    if state_scale.shape != state_offset.shape or tf.reduce_any(state_scale <= 0.0):
        raise ValueError("nonlinear teacher state scales must be positive and dimension-matched")
    if pair_indices.shape.rank != 2 or pair_indices.shape[1] != 2:
        raise ValueError("nonlinear teacher pair indices must have shape [P,2]")
    dimension = 2 * state_dimension
    if adapter.state_dimension != state_dimension:
        raise ValueError("nonlinear teacher adapter and chart dimensions differ")
    if initial_standard_deviation <= 0.0 or process_standard_deviation <= 0.0:
        raise ValueError("nonlinear teacher density scales must be positive")

    # A broad independent product design misses narrow transition manifolds.
    # Freeze a deterministic mixture of initial-local and transition-coupled
    # rows instead; these are fit rows, not particle or claim samples.
    candidate_count = max(16 * fit_rows, 256)
    initial_mean = adapter.initial_value(
        center_theta, tf.zeros([1, state_dimension], tf.float32)
    )[0]
    initial_reference_mean = (initial_mean - state_offset) / state_scale
    local_previous = initial_reference_mean[None, :] + tf.cast(
        initial_standard_deviation, tf.float32
    ) * tf.random.stateless_normal(
        [candidate_count, state_dimension],
        [int(root_seed), 3101],
        dtype=tf.float32,
    ) / state_scale[None, :]
    broad_previous = tf.random.stateless_uniform(
        [candidate_count, state_dimension],
        [int(root_seed), 3103],
        minval=-0.90,
        maxval=0.90,
        dtype=tf.float32,
    )

    def coupled(previous_reference: tf.Tensor, seed_offset: int) -> tf.Tensor:
        previous = state_offset[None, :] + previous_reference * state_scale[None, :]
        process_noise = tf.cast(process_standard_deviation, tf.float32) * (
            tf.random.stateless_normal(
                [candidate_count, state_dimension],
                [int(root_seed), seed_offset],
                dtype=tf.float32,
            )
        )
        current = adapter.transition_value(
            center_theta, previous, process_noise, tf.constant(0, tf.int32)
        )
        current_reference = (current - state_offset[None, :]) / state_scale[None, :]
        rows = tf.concat([current_reference, previous_reference], axis=1)
        valid = tf.reduce_all(tf.math.is_finite(rows), axis=1) & tf.reduce_all(
            tf.abs(rows) <= 0.95, axis=1
        )
        return tf.boolean_mask(rows, valid)

    local_rows = coupled(local_previous, 3109)
    broad_rows = coupled(broad_previous, 3113)
    local_count = max(1, fit_rows // 4)
    broad_count = fit_rows - local_count
    local_available = int(tf.shape(local_rows)[0].numpy())
    broad_available = int(tf.shape(broad_rows)[0].numpy())
    if local_available < local_count or broad_available < broad_count:
        raise ValueError("nonlinear chart does not contain enough transition-coupled fit rows")
    reference = tf.concat(
        [local_rows[:local_count], broad_rows[:broad_count]], axis=0
    )
    basis_values = tf.stack(
        [_legendre_basis_values(reference[:, axis], basis_size) for axis in range(dimension)]
    )
    query_basis_values = tf.stack(
        [
            _legendre_basis_values(
                reference[:, axis + state_dimension]
                if axis < state_dimension
                else reference[:, axis],
                basis_size,
            )
            for axis in range(dimension)
        ]
    )
    active_mask = _active_mask(dimension, rank, basis_size, tf.float32)
    initial_cores = tf.random.stateless_normal(
        [dimension, rank, basis_size, rank],
        seed=[int(root_seed), 3119],
        stddev=0.03,
        dtype=tf.float32,
    ) * active_mask
    initial_cores = tf.tensor_scatter_nd_update(
        initial_cores,
        tf.constant([[axis, 0, 0, 0] for axis in range(dimension)], tf.int32),
        tf.ones([dimension], tf.float32),
    )
    basis = LegendreBasis1D(BoundedInterval(-1.0, 1.0), basis_size - 1)
    operator_powers = tf.cast(
        tf.stack(
            [
                tf.stack(
                    [
                        legendre_monomial_operator_matrix(
                            basis, power, MassMeasure.REFERENCE_MEASURE
                        )
                        for power in range(5)
                    ]
                )
                for _ in range(dimension)
            ]
        ),
        tf.float32,
    )
    defensive_moments = tf.constant([1.0, 0.0, 1.0 / 3.0, 0.0, 1.0 / 5.0], tf.float32)
    state_matrix = tf.concat(
        [tf.linalg.diag(state_scale), tf.zeros([state_dimension, state_dimension], tf.float32)],
        axis=1,
    )
    return {
        "observations": observations,
        "reference_points": reference,
        "basis_values": basis_values,
        "active_mask": active_mask,
        "schedule": tf.tile(tf.range(dimension, dtype=tf.int32), [sweeps]),
        "weights": tf.fill([fit_rows], tf.cast(1.0 / fit_rows, tf.float32)),
        "initial_cores": initial_cores,
        "scale_shift_indices": tf.zeros([int(observations.shape[0])], tf.int32),
        "defensive_weights": tf.fill(
            [int(observations.shape[0])], tf.cast(defensive_weight, tf.float32)
        ),
        "query_basis_values": query_basis_values,
        "keep_mask": tf.concat(
            [tf.ones([state_dimension], tf.bool), tf.zeros([state_dimension], tf.bool)], axis=0
        ),
        "mass_operators": operator_powers[:, 0],
        "defensive_marginal_values": tf.ones(
            [int(observations.shape[0]), fit_rows], tf.float32
        ),
        "defensive_mass": tf.constant(1.0, tf.float32),
        "operator_powers": operator_powers,
        "defensive_power_moments": tf.tile(defensive_moments[None, :], [dimension, 1]),
        "state_offset": state_offset,
        "state_matrix": state_matrix,
        "pair_indices": pair_indices,
        "state_scale": state_scale,
        "center_theta": center_theta,
    }


def _normal_log_density_and_tangent(
    residual: tf.Tensor, residual_tangent: tf.Tensor, variance: float
) -> tuple[tf.Tensor, tf.Tensor]:
    dtype = residual.dtype
    dimension = tf.cast(tf.shape(residual)[1], dtype)
    variance_tensor = tf.cast(variance, dtype)
    value = -0.5 * (
        dimension * tf.math.log(tf.cast(_TWO_PI, dtype) * variance_tensor)
        + tf.reduce_sum(tf.square(residual), axis=1) / variance_tensor
    )
    tangent = -tf.reduce_sum(
        residual[:, :, None] * residual_tangent, axis=1
    ) / variance_tensor
    return value, tangent


def _teacher_base_log_targets(
    theta: tf.Tensor,
    prepared: Mapping[str, tf.Tensor],
    adapter: CandidateModelAdapter,
    *,
    initial_variance: float,
    process_variance: float,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    reference = prepared["reference_points"]
    dimension = adapter.state_dimension
    offset = prepared["state_offset"]
    scale = prepared["state_scale"]
    current = offset[None, :] + reference[:, :dimension] * scale[None, :]
    previous = offset[None, :] + reference[:, dimension:] * scale[None, :]
    zeros = tf.zeros_like(previous)
    zero_tangent = tf.zeros(
        [tf.shape(previous)[0], dimension, adapter.parameter_count], tf.float32
    )
    rows = []
    tangents = []
    for time_index in range(int(prepared["observations"].shape[0])):
        time_tensor = tf.constant(time_index, tf.int32)
        mean = adapter.transition_value(theta, previous, zeros, time_tensor)
        mean_tangent = adapter.transition_tangent(
            theta, previous, zeros, zero_tangent, time_tensor
        )
        transition, transition_tangent = _normal_log_density_and_tangent(
            current - mean, -mean_tangent, process_variance
        )
        observation = adapter.observation_value(
            theta, current, prepared["observations"][time_index], time_tensor
        )
        observation_tangent = adapter.observation_tangent(
            theta,
            current,
            tf.zeros(
                [tf.shape(current)[0], dimension, adapter.parameter_count], tf.float32
            ),
            prepared["observations"][time_index],
            time_tensor,
        )
        value = transition + observation
        tangent = transition_tangent + observation_tangent
        if time_index == 0:
            if adapter.initial_log_density is None:
                initial, initial_tangent = _normal_log_density_and_tangent(
                    previous - adapter.initial_value(theta, tf.zeros_like(previous)),
                    -adapter.initial_tangent(theta, tf.zeros_like(previous)),
                    initial_variance,
                )
            else:
                initial = adapter.initial_log_density(theta, previous)
                initial_tangent = adapter.initial_log_density_tangent(theta, previous)
            value += initial
            tangent += initial_tangent
        rows.append(value)
        tangents.append(tangent)
    chart_log_jacobian = 2.0 * tf.reduce_sum(tf.math.log(2.0 * scale))
    values = tf.stack(rows) + chart_log_jacobian
    tangent_values = tf.stack(tangents)
    valid = tf.reduce_all(tf.math.is_finite(theta)) & tf.reduce_all(
        tf.math.is_finite(values)
    ) & tf.reduce_all(tf.math.is_finite(tangent_values))
    return values, tangent_values, valid


def _teacher_targets(
    theta: tf.Tensor,
    prepared: Mapping[str, tf.Tensor],
    controls: MomentTeacherControls,
    adapter: CandidateModelAdapter,
    *,
    initial_variance: float,
    process_variance: float,
    setup_static: bool = False,
) -> dict[str, tf.Tensor]:
    base, base_tangents, physical_valid = _teacher_base_log_targets(
        theta,
        prepared,
        adapter,
        initial_variance=initial_variance,
        process_variance=process_variance,
    )
    recursion = (
        padded_fixed_teacher_recursion_shape_xla.python_function
        if setup_static
        else padded_fixed_teacher_recursion_shape_xla
    )
    results = []
    for parameter_index in range(adapter.parameter_count):
        results.append(
            recursion(
                prepared["basis_values"],
                prepared["active_mask"],
                prepared["schedule"],
                base,
                base_tangents[:, :, parameter_index],
                prepared["weights"],
                tf.zeros_like(prepared["weights"]),
                prepared["initial_cores"],
                tf.zeros_like(prepared["initial_cores"]),
                prepared["scale_shift_indices"],
                prepared["defensive_weights"],
                tf.zeros_like(prepared["defensive_weights"]),
                prepared["query_basis_values"],
                prepared["keep_mask"],
                prepared["mass_operators"],
                prepared["defensive_marginal_values"],
                tf.zeros_like(prepared["defensive_marginal_values"]),
                prepared["defensive_mass"],
                tf.zeros([], tf.float32),
                prepared["operator_powers"],
                prepared["defensive_power_moments"],
                prepared["state_offset"],
                tf.zeros_like(prepared["state_offset"]),
                prepared["state_matrix"],
                tf.zeros_like(prepared["state_matrix"]),
                prepared["pair_indices"],
                tf.cast(controls.tt_ridge, tf.float32),
                tf.cast(controls.column_scale_floor, tf.float32),
                tf.cast(controls.condition_number_veto, tf.float32),
                tf.cast(controls.fit_residual_veto, tf.float32),
            )
        )
    first = results[0]
    return {
        "marginal_values": first[2],
        "normalizers": first[4],
        "skew": first[5],
        "kurtosis": first[6],
        "co_skew": first[7],
        "co_kurtosis": first[8],
        "skew_tangent": tf.stack([item[9] for item in results], axis=-1),
        "kurtosis_tangent": tf.stack([item[10] for item in results], axis=-1),
        "co_skew_tangent": tf.stack([item[11] for item in results], axis=-1),
        "co_kurtosis_tangent": tf.stack([item[12] for item in results], axis=-1),
        "valid": physical_valid & tf.reduce_all(tf.stack([item[-1] for item in results])),
    }


def freeze_nonlinear_teacher_scale_shift_indices(
    teacher_prepared: Mapping[str, tf.Tensor],
    controls: MomentTeacherControls,
    adapter: CandidateModelAdapter,
    *,
    initial_variance: float,
    process_variance: float,
    maximum_iterations: int = 8,
) -> dict[str, tf.Tensor]:
    prepared = dict(teacher_prepared)
    theta = prepared["center_theta"]
    base, _, _ = _teacher_base_log_targets(
        theta,
        prepared,
        adapter,
        initial_variance=initial_variance,
        process_variance=process_variance,
    )
    # Start from a finite base-target shift before adding the carried marginal.
    # An arbitrary row can overflow exp(0.5 * (log_target - shift)) on nonlinear
    # charts before the fixed-point iteration has a chance to update it.
    indices = tf.argmax(base, axis=1, output_type=tf.int32)
    for _ in range(maximum_iterations):
        prepared["scale_shift_indices"] = indices
        targets = _teacher_targets(
            theta,
            prepared,
            controls,
            adapter,
            initial_variance=initial_variance,
            process_variance=process_variance,
            setup_static=True,
        )
        if not bool(targets["valid"].numpy()):
            raise ValueError("nonlinear teacher scale-shift freeze found an invalid fit")
        previous = tf.concat(
            [tf.ones_like(targets["marginal_values"][:1]), targets["marginal_values"][:-1]],
            axis=0,
        )
        augmented = base + tf.where(
            tf.range(tf.shape(base)[0])[:, None] > 0,
            tf.math.log(tf.maximum(previous, tf.constant(1.0e-30, tf.float32))),
            tf.zeros_like(base),
        )
        next_indices = tf.argmax(augmented, axis=1, output_type=tf.int32)
        if bool(tf.reduce_all(next_indices == indices).numpy()):
            prepared["scale_shift_indices"] = next_indices
            return prepared
        indices = next_indices
    raise ValueError("nonlinear teacher scale-shift branch did not stabilize")


def _normalize(logits: tf.Tensor, tangent: tf.Tensor) -> dict[str, tf.Tensor]:
    increment = tf.reduce_logsumexp(logits)
    log_weights = logits - increment
    weights = tf.exp(log_weights)
    increment_tangent = tf.reduce_sum(weights[:, None] * tangent, axis=0)
    log_weight_tangent = tangent - increment_tangent[None, :]
    return {
        "increment": increment,
        "increment_tangent": increment_tangent,
        "log_weights": log_weights,
        "log_weights_tangent": log_weight_tangent,
        "weights": weights,
        "weights_tangent": weights[:, None] * log_weight_tangent,
    }


def _geometry(particles: tf.Tensor, tangent: tf.Tensor) -> dict[str, tf.Tensor]:
    dtype = particles.dtype
    center = tf.reduce_mean(particles, axis=0, keepdims=True)
    center_tangent = tf.reduce_mean(tangent, axis=0, keepdims=True)
    centered = particles - center
    centered_tangent = tangent - center_tangent
    variance = tf.reduce_mean(tf.square(centered), axis=0)
    std = tf.sqrt(variance)
    safe_std = tf.maximum(std, tf.cast(1.0e-12, dtype))
    variance_tangent = 2.0 * tf.reduce_mean(
        centered[:, :, None] * centered_tangent, axis=0
    )
    std_tangent = variance_tangent / (2.0 * safe_std[:, None])
    diameter = tf.reduce_max(std)
    diameter_mask = std == diameter
    diameter_weights = tf.cast(diameter_mask, dtype) / tf.reduce_sum(
        tf.cast(diameter_mask, dtype)
    )
    diameter_tangent = tf.reduce_sum(diameter_weights[:, None] * std_tangent, axis=0)
    scale = tf.sqrt(tf.cast(tf.shape(particles)[1], dtype)) * diameter
    scale_tangent = tf.sqrt(tf.cast(tf.shape(particles)[1], dtype)) * diameter_tangent
    safe_scale = tf.maximum(scale, tf.cast(1.0e-12, dtype))
    scaled = centered / safe_scale
    scaled_tangent = (
        centered_tangent / safe_scale
        - centered[:, :, None] * scale_tangent[None, None, :] / tf.square(safe_scale)
    )
    maximum = tf.reduce_max(scaled)
    minimum = tf.reduce_min(scaled)
    maximum_mask = scaled == maximum
    minimum_mask = scaled == minimum
    maximum_tangent = tf.reduce_sum(
        tf.cast(maximum_mask, dtype)[:, :, None] * scaled_tangent,
        axis=[0, 1],
    ) / tf.reduce_sum(tf.cast(maximum_mask, dtype))
    minimum_tangent = tf.reduce_sum(
        tf.cast(minimum_mask, dtype)[:, :, None] * scaled_tangent,
        axis=[0, 1],
    ) / tf.reduce_sum(tf.cast(minimum_mask, dtype))
    coordinate_range = maximum - minimum
    range_squared = tf.square(coordinate_range)
    epsilon0 = tf.maximum(range_squared, tf.cast(1.0e-6, dtype))
    epsilon0_tangent = tf.where(
        range_squared >= tf.cast(1.0e-6, dtype),
        2.0 * coordinate_range * (maximum_tangent - minimum_tangent),
        tf.zeros_like(maximum_tangent),
    )
    return {
        "scaled": scaled,
        "scaled_tangent": scaled_tangent,
        "epsilon0": epsilon0,
        "epsilon0_tangent": epsilon0_tangent,
        "valid": tf.reduce_all(tf.math.is_finite(scaled)) & (diameter > 0.0),
    }


def _scatter_pairs(
    values: tf.Tensor, pair_indices: tf.Tensor, state_dimension: int
) -> tf.Tensor:
    return tf.scatter_nd(pair_indices, values, [state_dimension, state_dimension])


def _scatter_pair_tangents(
    values: tf.Tensor, pair_indices: tf.Tensor, state_dimension: int, parameter_count: int
) -> tf.Tensor:
    return tf.scatter_nd(
        pair_indices, values, [state_dimension, state_dimension, parameter_count]
    )


def latent_preclip_austria_sir_candidate_adapter() -> CandidateModelAdapter:
    """Adapt Austria SIR to the sealed latent-preclip source event order."""

    base = parameterized_austria_sir_candidate_adapter()

    def physical_state(
        particles: tf.Tensor, tangent: tf.Tensor, time_index: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor]:
        susceptible = particles[:, 0::2]
        infectious = particles[:, 1::2]
        clipped = tf.reshape(
            tf.stack([tf.maximum(susceptible, 0.0), infectious], axis=2),
            [tf.shape(particles)[0], base.state_dimension],
        )
        active = tf.cast(susceptible > 0.0, particles.dtype)
        clipped_tangent = tf.reshape(
            tf.stack(
                [
                    tangent[:, 0::2, :] * active[:, :, None],
                    tangent[:, 1::2, :],
                ],
                axis=2,
            ),
            [tf.shape(particles)[0], base.state_dimension, base.parameter_count],
        )
        use_clip = time_index > 0
        return (
            tf.where(use_clip, clipped, particles),
            tf.where(use_clip, clipped_tangent, tangent),
        )

    def transition_value(theta, particles, noise, time_index):
        zero_tangent = tf.zeros(
            [tf.shape(particles)[0], base.state_dimension, base.parameter_count],
            particles.dtype,
        )
        previous, _ = physical_state(particles, zero_tangent, time_index)
        return base.transition_value(theta, previous, noise, time_index)

    def transition_tangent(theta, particles, noise, tangent, time_index):
        previous, previous_tangent = physical_state(particles, tangent, time_index)
        return base.transition_tangent(
            theta, previous, noise, previous_tangent, time_index
        )

    return CandidateModelAdapter(
        state_dimension=base.state_dimension,
        parameter_count=base.parameter_count,
        initial_value=base.initial_value,
        initial_tangent=base.initial_tangent,
        transition_value=transition_value,
        transition_tangent=transition_tangent,
        observation_value=base.observation_value,
        observation_tangent=base.observation_tangent,
    )


def nonlinear_moment_teacher_value_and_score_core(
    theta: tf.Tensor,
    particle_prepared: Mapping[str, tf.Tensor],
    teacher_prepared: Mapping[str, tf.Tensor] | None,
    controls: MomentTeacherControls,
    adapter: CandidateModelAdapter,
    *,
    initial_variance: float,
    process_variance: float,
    row_chunk_size: int,
    col_chunk_size: int,
) -> dict[str, tf.Tensor]:
    """Evaluate one source-order bootstrap/Chol finite program and total score."""

    theta = tf.reshape(tf.convert_to_tensor(theta, tf.float32), [adapter.parameter_count])
    initial_noise = tf.convert_to_tensor(particle_prepared["initial_noise"], tf.float32)
    particle_count = int(initial_noise.shape[0])
    state_dimension = adapter.state_dimension
    parameter_count = adapter.parameter_count
    teacher = None
    if teacher_prepared is not None:
        teacher = _teacher_targets(
            theta,
            teacher_prepared,
            controls,
            adapter,
            initial_variance=initial_variance,
            process_variance=process_variance,
        )
    particles = adapter.initial_value(theta, initial_noise)
    particle_tangent = adapter.initial_tangent(theta, initial_noise)
    uniform_log_weight = -tf.math.log(tf.cast(particle_count, tf.float32))
    log_weights = tf.fill([particle_count], uniform_log_weight)
    log_weights_tangent = tf.zeros([particle_count, parameter_count], tf.float32)
    objective = tf.zeros([], tf.float32)
    score = tf.zeros([parameter_count], tf.float32)
    valid = tf.constant(True) if teacher is None else teacher["valid"]
    time_steps = int(particle_prepared["observations"].shape[0])
    shape_history = tf.TensorArray(tf.bool, size=time_steps, element_shape=[])
    mean_history = tf.TensorArray(tf.float32, size=time_steps, element_shape=[])
    covariance_history = tf.TensorArray(tf.float32, size=time_steps, element_shape=[])
    skew_history = tf.TensorArray(tf.float32, size=time_steps, element_shape=[])
    pair_indices = (
        tf.zeros([0, 2], tf.int32)
        if teacher_prepared is None
        else teacher_prepared["pair_indices"]
    )
    off_diagonal = 1.0 - tf.eye(state_dimension, dtype=tf.float32)

    def body(
        time_index,
        particles_value,
        particle_tangent_value,
        log_weights_value,
        log_weights_tangent_value,
        objective_value,
        score_value,
        valid_value,
        shape_values,
        mean_values,
        covariance_values,
        skew_values,
    ):
        noise = particle_prepared["process_noise"][time_index]
        proposed = adapter.transition_value(theta, particles_value, noise, time_index)
        proposed_tangent = adapter.transition_tangent(
            theta, particles_value, noise, particle_tangent_value, time_index
        )
        observation_log = adapter.observation_value(
            theta,
            proposed,
            particle_prepared["observations"][time_index],
            time_index,
        )
        observation_tangent = adapter.observation_tangent(
            theta,
            proposed,
            proposed_tangent,
            particle_prepared["observations"][time_index],
            time_index,
        )
        normalized = _normalize(
            log_weights_value + observation_log,
            log_weights_tangent_value + observation_tangent,
        )
        geometry = _geometry(proposed, proposed_tangent)
        reset = _contract_e_streaming_forward_jvp_core(
            geometry["scaled"][None, :, :],
            proposed[None, :, :],
            normalized["log_weights"][None, :],
            normalized["weights"][None, :],
            particle_prepared["residual_design"][time_index][None, :, :],
            particle_prepared["prepared_ridge"][time_index][None],
            geometry["scaled_tangent"][None, :, :, :],
            proposed_tangent[None, :, :, :],
            normalized["log_weights_tangent"][None, :, :],
            normalized["weights_tangent"][None, :, :],
            tf.zeros([1, particle_count, state_dimension, parameter_count], tf.float32),
            tf.zeros([1, parameter_count], tf.float32),
            geometry["epsilon0_tangent"][None, :],
            particle_prepared["epsilon"],
            geometry["epsilon0"][None],
            particle_prepared["scaling"],
            steps=controls.sinkhorn_steps,
            balance_steps=controls.balance_steps,
            row_chunk_size=row_chunk_size,
            col_chunk_size=col_chunk_size,
        )
        reset_valid = (
            geometry["valid"]
            & reset["quotient"]["valid_chart"][0]
            & reset["quotient"]["marginal_valid"][0]
            & reset["reset"]["finite"][0]
            & reset["reset"]["factor_diagonal_positive"][0]
        )
        if teacher is None:
            corrected_particles = reset["particles"][0]
            corrected_tangent = reset["particles_tangent"][0]
            correction_valid = tf.constant(True)
            skew_residual = tf.zeros([], tf.float32)
        else:
            corrected = higher_moment_shape_jvp(
                proposed,
                normalized["weights"],
                proposed_tangent,
                normalized["weights_tangent"],
                reset["particles"][0],
                reset["particles_tangent"][0],
                correction_steps=controls.correction_steps,
                strength=controls.correction_strength,
                floor=controls.correction_floor,
                pairwise_correction_steps=controls.pairwise_correction_steps,
                pairwise_strength=controls.pairwise_strength,
                pairwise_floor=controls.pairwise_floor,
                explicit_target_skew=teacher["skew"][time_index],
                explicit_target_kurtosis=teacher["kurtosis"][time_index],
                explicit_target_skew_tangent=teacher["skew_tangent"][time_index],
                explicit_target_kurtosis_tangent=teacher["kurtosis_tangent"][time_index],
                explicit_target_pairwise_co_skew=_scatter_pairs(
                    teacher["co_skew"][time_index], pair_indices, state_dimension
                ),
                explicit_target_pairwise_co_kurtosis=_scatter_pairs(
                    teacher["co_kurtosis"][time_index], pair_indices, state_dimension
                ),
                explicit_target_pairwise_co_skew_tangent=_scatter_pair_tangents(
                    teacher["co_skew_tangent"][time_index],
                    pair_indices,
                    state_dimension,
                    parameter_count,
                ),
                explicit_target_pairwise_co_kurtosis_tangent=_scatter_pair_tangents(
                    teacher["co_kurtosis_tangent"][time_index],
                    pair_indices,
                    state_dimension,
                    parameter_count,
                ),
                pairwise_co_skew_target_mask=off_diagonal,
                pairwise_co_kurtosis_target_mask=off_diagonal,
            )
            corrected_particles = corrected["particles"]
            corrected_tangent = corrected["particles_tangent"]
            correction_valid = corrected["valid"]
            skew_residual = tf.reduce_max(tf.abs(corrected["skew_residual"]))
        source_mean = tf.reduce_sum(normalized["weights"][:, None] * proposed, axis=0)
        source_centered = proposed - source_mean[None, :]
        source_covariance = tf.einsum(
            "n,ni,nj->ij", normalized["weights"], source_centered, source_centered
        )
        corrected_mean = tf.reduce_mean(corrected_particles, axis=0)
        corrected_centered = corrected_particles - corrected_mean[None, :]
        corrected_covariance = tf.einsum(
            "ni,nj->ij", corrected_centered, corrected_centered
        ) / tf.cast(particle_count, tf.float32)
        step_valid = (
            reset_valid
            & correction_valid
            & tf.reduce_all(tf.math.is_finite(observation_log))
            & tf.reduce_all(tf.math.is_finite(observation_tangent))
        )
        return (
            time_index + 1,
            corrected_particles,
            corrected_tangent,
            tf.fill([particle_count], uniform_log_weight),
            tf.zeros_like(log_weights_tangent_value),
            objective_value + normalized["increment"],
            score_value + normalized["increment_tangent"],
            valid_value & step_valid,
            shape_values.write(time_index, correction_valid),
            mean_values.write(time_index, tf.reduce_max(tf.abs(corrected_mean - source_mean))),
            covariance_values.write(
                time_index,
                tf.reduce_max(tf.abs(corrected_covariance - source_covariance)),
            ),
            skew_values.write(time_index, skew_residual),
        )

    result = tf.while_loop(
        lambda time_index, *_: time_index < time_steps,
        body,
        (
            tf.zeros([], tf.int32),
            particles,
            particle_tangent,
            log_weights,
            log_weights_tangent,
            objective,
            score,
            valid,
            shape_history,
            mean_history,
            covariance_history,
            skew_history,
        ),
        maximum_iterations=time_steps,
        parallel_iterations=1,
    )
    return {
        "objective": result[5],
        "score": result[6],
        "valid_chart": result[7],
        "final_particles": result[1],
        "final_particles_tangent": result[2],
        "teacher_valid": tf.constant(True) if teacher is None else teacher["valid"],
        "teacher_normalizers": (
            tf.ones([time_steps], tf.float32) if teacher is None else teacher["normalizers"]
        ),
        "shape_valid_history": result[8].stack(),
        "mean_residual_history": result[9].stack(),
        "covariance_residual_history": result[10].stack(),
        "skew_residual_history": result[11].stack(),
    }


def predator_prey_moment_teacher_value_and_score_core(
    theta: tf.Tensor,
    particle_prepared: Mapping[str, tf.Tensor],
    teacher_prepared: Mapping[str, tf.Tensor] | None,
    controls: MomentTeacherControls,
    *,
    row_chunk_size: int,
    col_chunk_size: int,
) -> dict[str, tf.Tensor]:
    return nonlinear_moment_teacher_value_and_score_core(
        theta,
        particle_prepared,
        teacher_prepared,
        controls,
        predator_prey_candidate_adapter(),
        initial_variance=1.0,
        process_variance=4.0,
        row_chunk_size=row_chunk_size,
        col_chunk_size=col_chunk_size,
    )


def austria_sir_moment_teacher_value_and_score_core(
    theta: tf.Tensor,
    particle_prepared: Mapping[str, tf.Tensor],
    teacher_prepared: Mapping[str, tf.Tensor] | None,
    controls: MomentTeacherControls,
    *,
    row_chunk_size: int,
    col_chunk_size: int,
) -> dict[str, tf.Tensor]:
    return nonlinear_moment_teacher_value_and_score_core(
        theta,
        particle_prepared,
        teacher_prepared,
        controls,
        latent_preclip_austria_sir_candidate_adapter(),
        initial_variance=1.0,
        process_variance=1.0,
        row_chunk_size=row_chunk_size,
        col_chunk_size=col_chunk_size,
    )


def actual_sv_moment_teacher_value_and_score_core(
    theta: tf.Tensor,
    particle_prepared: Mapping[str, tf.Tensor],
    teacher_prepared: Mapping[str, tf.Tensor] | None,
    controls: MomentTeacherControls,
    *,
    row_chunk_size: int,
    col_chunk_size: int,
) -> dict[str, tf.Tensor]:
    """Evaluate exact transformed SV with the stationary initial density."""

    return nonlinear_moment_teacher_value_and_score_core(
        theta,
        particle_prepared,
        teacher_prepared,
        controls,
        exact_transformed_sv_candidate_adapter(sigma=1.0),
        # The explicit adapter callback owns the parameter-dependent initial law.
        initial_variance=1.0,
        process_variance=1.0,
        row_chunk_size=row_chunk_size,
        col_chunk_size=col_chunk_size,
    )


def make_nonlinear_moment_teacher_value_and_score_tf(
    *,
    adapter: CandidateModelAdapter,
    particle_prepared: Mapping[str, tf.Tensor],
    teacher_prepared: Mapping[str, tf.Tensor] | None,
    tuning_artifact: NonlinearMomentTeacherTuningArtifact,
    expected_scope: LEDHTuningScope,
    initial_variance: float,
    process_variance: float,
    jit_compile: bool = True,
):
    if tuning_artifact._seal not in _ISSUED_TUNING_SEALS:
        raise TypeError("nonlinear tuning artifact has no repository issuance")
    require_scope_match(
        expected_scope,
        tuning_artifact.scope.as_dict(),
        label="nonlinear moment-teacher tuning artifact",
    )
    chunks = validate_transport_chunks(
        expected_scope.particle_count,
        row_chunk_size=expected_scope.row_chunk_size,
        col_chunk_size=expected_scope.col_chunk_size,
    )
    if int(particle_prepared["initial_noise"].shape[0]) != expected_scope.particle_count:
        raise ValueError("nonlinear particle count does not match tuning scope")
    if int(particle_prepared["observations"].shape[0]) != expected_scope.horizon:
        raise ValueError("nonlinear horizon does not match tuning scope")
    controls = tuning_artifact.controls

    @tf.function(
        input_signature=[tf.TensorSpec([adapter.parameter_count], tf.float32)],
        jit_compile=jit_compile,
        reduce_retracing=True,
    )
    def evaluate(theta: tf.Tensor) -> dict[str, tf.Tensor]:
        return nonlinear_moment_teacher_value_and_score_core(
            theta,
            particle_prepared,
            teacher_prepared,
            controls,
            adapter,
            initial_variance=initial_variance,
            process_variance=process_variance,
            row_chunk_size=chunks.row_chunk_size,
            col_chunk_size=chunks.col_chunk_size,
        )

    return evaluate


def route_identity_prepared_inputs(
    particle_prepared: Mapping[str, tf.Tensor],
    teacher_prepared: Mapping[str, tf.Tensor],
    tuning_artifact: NonlinearMomentTeacherTuningArtifact,
) -> dict[str, object]:
    """Bind every prepared tensor and tuned control used by the route."""

    if tuning_artifact._seal not in _ISSUED_TUNING_SEALS:
        raise TypeError("route identity requires repository-issued nonlinear tuning")
    result: dict[str, object] = {
        name: particle_prepared[name]
        for name in (
            "observations",
            "initial_noise",
            "process_noise",
            "residual_design",
            "prepared_ridge",
            "epsilon",
            "scaling",
        )
    }
    result["event_order_code"] = 1
    result["teacher_observations"] = teacher_prepared["observations"]
    for name in (
        "reference_points",
        "basis_values",
        "active_mask",
        "schedule",
        "weights",
        "initial_cores",
        "scale_shift_indices",
        "defensive_weights",
        "query_basis_values",
        "keep_mask",
        "mass_operators",
        "defensive_marginal_values",
        "defensive_mass",
        "operator_powers",
        "defensive_power_moments",
        "state_offset",
        "state_matrix",
        "state_scale",
        "pair_indices",
        "center_theta",
    ):
        result[name] = teacher_prepared[name]
    result.update(tuning_artifact.controls.as_dict())
    result["row_chunk_size"] = tuning_artifact.scope.row_chunk_size
    result["col_chunk_size"] = tuning_artifact.scope.col_chunk_size
    return result


__all__ = [
    "ACTUAL_SV_ROUTE_ID",
    "AUSTRIA_SIR_ROUTE_ID",
    "CONTROL_FAMILY_ID",
    "EVENT_ORDER",
    "NonlinearMomentTeacherTuningArtifact",
    "PREDATOR_PREY_ROUTE_ID",
    "austria_sir_moment_teacher_value_and_score_core",
    "actual_sv_moment_teacher_value_and_score_core",
    "freeze_nonlinear_teacher_scale_shift_indices",
    "issue_nonlinear_moment_teacher_tuning_artifact",
    "latent_preclip_austria_sir_candidate_adapter",
    "make_nonlinear_moment_teacher_value_and_score_tf",
    "make_nonlinear_tuning_scope",
    "nonlinear_moment_teacher_value_and_score_core",
    "predator_prey_moment_teacher_value_and_score_core",
    "prepare_nonlinear_teacher_inputs",
    "route_identity_prepared_inputs",
]
