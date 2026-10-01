"""Diagnostic LGSSM particle filter with an independent TT moment teacher.

The particle lane owns the finite likelihood.  The squared-TT lane supplies
only explicit standardized shape targets for the bounded post-reset
correction.  Both lanes carry their complete manual directional derivatives.
These are finite-program JVPs, not admitted canonical recursive LEDH scores.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from typing import Any, Mapping

import tensorflow as tf

from bayesfilter.highdim.higher_moment_contract_e import higher_moment_shape_jvp
from bayesfilter.highdim.ledh_contract_e_canonical_lgssm_tf import (
    OBSERVATION_DIMENSION,
    PARAMETER_COUNT,
    STATE_DIMENSION,
    _canonical_fused_step_core,
    _gaussian_log_density_jvp_core,
    _lgssm_component_tangents,
    _lgssm_components,
    _physical_chart,
)
from bayesfilter.highdim.ledh_tuning_scope import LEDHTuningScope, require_scope_match
from bayesfilter.highdim.moment_teacher_native_tf import (
    freeze_scale_shift_core,
    operator_power_program,
    teacher_directions,
    teacher_normal,
)
from bayesfilter.highdim.transport_chunk_policy import (
    TRANSPORT_CHUNK_POLICY_ID,
    select_transport_chunks,
    validate_transport_chunks,
)
from bayesfilter.ops.stateless_random_tf import philox_uniform_float64

ROUTE_ID = "zhao_cui_moment_teacher_contract_e_chol_lgssm_v1"
ROUTE_SPECIFICATION_ID = "contract_e_chol_zhao_cui_moment_teacher_lgssm_v1"
CONTROL_FAMILY_ID = "zhao_cui_moment_teacher_lgssm_controls_v1"
TUNING_ARTIFACT_SCHEMA = "bayesfilter.zhao_cui_moment_teacher_tuning.v1"
RESET_CONTRACT_ID = "contract_e_chol_v1"
CANONICAL_LEDH_ADMITTED = False
SCORE_PROVENANCE = "finite_program_manual_jvp_diagnostic_only"
PAIR_INDICES = ((0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1))

_TUNING_CONSTRUCTION_KEY = object()
_ISSUED_TUNING_SEALS: set[object] = set()


@dataclass(frozen=True)
class MomentTeacherControls:
    """Fixed controls for one exact tuning scope."""

    sinkhorn_steps: int
    balance_steps: int
    correction_steps: int
    correction_strength: float
    correction_floor: float
    pairwise_correction_steps: int
    pairwise_strength: float
    pairwise_floor: float
    tt_ridge: float
    column_scale_floor: float
    condition_number_veto: float
    fit_residual_veto: float

    def __post_init__(self) -> None:
        integer_values = (
            self.sinkhorn_steps,
            self.balance_steps,
            self.correction_steps,
            self.pairwise_correction_steps,
        )
        if any(isinstance(value, bool) or int(value) != value for value in integer_values):
            raise TypeError("moment-teacher step controls must be integers")
        if self.sinkhorn_steps <= 0 or self.balance_steps <= 0:
            raise ValueError("Sinkhorn and balance steps must be positive")
        if self.correction_steps < 0 or self.pairwise_correction_steps < 0:
            raise ValueError("moment correction step counts must be non-negative")
        positive = (
            self.correction_floor,
            self.pairwise_floor,
            self.tt_ridge,
            self.column_scale_floor,
            self.condition_number_veto,
            self.fit_residual_veto,
        )
        if any(float(value) <= 0.0 for value in positive):
            raise ValueError("moment-teacher floors, ridge, and vetoes must be positive")
        if self.correction_strength < 0.0 or self.pairwise_strength < 0.0:
            raise ValueError("moment correction strengths must be non-negative")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def sha256(self) -> str:
        payload = json.dumps(
            self.as_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("ascii")
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class MomentTeacherTuningArtifact:
    """Repository-issued selection for one exact LEDH/TT execution scope."""

    scope: LEDHTuningScope
    controls: MomentTeacherControls
    calibration_data_id: str
    validation_data_id: str
    selection_record_id: str
    _seal: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._seal not in _ISSUED_TUNING_SEALS:
            raise TypeError("moment-teacher tuning artifacts must be repository-issued")
        if not self.calibration_data_id or not self.validation_data_id:
            raise ValueError("calibration and validation IDs must be nonempty")
        if self.calibration_data_id == self.validation_data_id:
            raise ValueError("calibration and validation data must be disjoint")

    @property
    def artifact_id(self) -> str:
        payload = {
            "schema": TUNING_ARTIFACT_SCHEMA,
            "scope": self.scope.as_dict(),
            "scope_sha256": self.scope.scope_sha256,
            "controls": self.controls.as_dict(),
            "controls_sha256": self.controls.sha256,
            "calibration_data_id": self.calibration_data_id,
            "validation_data_id": self.validation_data_id,
            "selection_record_id": self.selection_record_id,
        }
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("ascii")
        return hashlib.sha256(encoded).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": TUNING_ARTIFACT_SCHEMA,
            "artifact_id": self.artifact_id,
            "scope": self.scope.as_dict(),
            "scope_sha256": self.scope.scope_sha256,
            "controls": self.controls.as_dict(),
            "controls_sha256": self.controls.sha256,
            "calibration_data_id": self.calibration_data_id,
            "validation_data_id": self.validation_data_id,
            "selection_record_id": self.selection_record_id,
        }


def issue_moment_teacher_tuning_artifact(
    *,
    scope: LEDHTuningScope,
    controls: MomentTeacherControls,
    calibration_data_id: str,
    validation_data_id: str,
    selection_record_id: str,
) -> MomentTeacherTuningArtifact:
    """Issue a fail-closed tuning selection from repository-owned fields."""

    if scope.route_id != ROUTE_ID:
        raise ValueError("moment-teacher tuning scope has the wrong route")
    if scope.reset_contract_id != RESET_CONTRACT_ID:
        raise ValueError("moment-teacher tuning scope has the wrong reset contract")
    if scope.control_family_id != CONTROL_FAMILY_ID:
        raise ValueError("moment-teacher tuning scope has the wrong control family")
    seal = object()
    _ISSUED_TUNING_SEALS.add(seal)
    return MomentTeacherTuningArtifact(
        scope=scope,
        controls=controls,
        calibration_data_id=str(calibration_data_id),
        validation_data_id=str(validation_data_id),
        selection_record_id=str(selection_record_id),
        _seal=seal,
    )


def _legendre_basis_values(points: tf.Tensor, basis_size: int) -> tf.Tensor:
    points = tf.convert_to_tensor(points)
    if basis_size < 1:
        raise ValueError("basis_size must be positive")
    flat = tf.reshape(points, [-1])
    values = tf.TensorArray(points.dtype, size=basis_size, element_shape=flat.shape).write(0, tf.ones_like(flat))
    if basis_size > 1:
        values = values.write(1, tf.sqrt(tf.cast(3.0, points.dtype)) * flat)

        def step(degree, previous, current, values):
            following = (
                tf.cast(2 * degree + 1, points.dtype) * flat * current
                - tf.cast(degree, points.dtype) * previous
            ) / tf.cast(degree + 1, points.dtype)
            values = values.write(degree + 1,
                tf.sqrt(tf.cast(2 * (degree + 1) + 1, points.dtype)) * following
            )
            return degree + 1, current, following, values

        _, _, _, values = tf.while_loop(
            lambda degree, *_: degree < basis_size - 1, step,
            (tf.constant(1), tf.ones_like(flat), flat, values),
            maximum_iterations=basis_size - 2, parallel_iterations=1,
        )
    return tf.reshape(tf.transpose(values.stack()), tf.concat([tf.shape(points), [basis_size]], 0))


def _active_mask(dimension: int, rank: int, basis_size: int, dtype: tf.dtypes.DType) -> tf.Tensor:
    middle = tf.ones([dimension, rank, basis_size, rank], dtype)
    left_mask = tf.one_hot(0, rank, dtype=dtype)[:, None, None]
    right_mask = tf.one_hot(0, rank, dtype=dtype)[None, None, :]
    first = middle[0] * left_mask
    last = middle[-1] * right_mask
    if dimension == 2:
        return tf.stack([first, last])
    return tf.concat([first[None], middle[1:-1], last[None]], axis=0)


def prepare_lgssm_teacher_inputs(
    *,
    observations: tf.Tensor,
    time_steps: int,
    fit_rows: int,
    basis_size: int,
    rank: int,
    sweeps: int,
    chart_scale: float,
    defensive_weight: float,
    root_seed: int,
    dtype: tf.dtypes.DType,
    center_theta: tf.Tensor,
    jit_compile: bool = True,
) -> dict[str, tf.Tensor]:
    """Create deterministic fixed rows, bases, operators, and branch choices."""

    dtype = tf.dtypes.as_dtype(dtype)
    if dtype not in (tf.float32, tf.float64):
        raise ValueError("teacher dtype must be float32 or float64")
    if min(time_steps, fit_rows, basis_size, rank, sweeps) <= 0:
        raise ValueError("teacher dimensions and sweep count must be positive")
    observations = tf.convert_to_tensor(observations, dtype)
    if observations.shape != (time_steps, OBSERVATION_DIMENSION):
        raise ValueError("teacher observations must match the declared LGSSM horizon")
    return _lgssm_preparation_program(
        time_steps, fit_rows, basis_size, rank, sweeps, dtype, bool(jit_compile),
    )(observations, tf.convert_to_tensor(center_theta, dtype),
      tf.cast(chart_scale, dtype), tf.cast(defensive_weight, dtype),
      tf.convert_to_tensor(root_seed, tf.int32))


@lru_cache(maxsize=16)
def _lgssm_preparation_program(time_steps, fit_rows, basis_size, rank, sweeps, dtype, jit_compile):
    operators = operator_power_program(basis_size, jit_compile=jit_compile)()

    @tf.function(input_signature=[tf.TensorSpec([time_steps, OBSERVATION_DIMENSION], dtype),
        tf.TensorSpec([PARAMETER_COUNT], dtype), tf.TensorSpec([], dtype),
        tf.TensorSpec([], dtype), tf.TensorSpec([], tf.int32)],
        jit_compile=jit_compile, autograph=False)
    def prepare(observations, center_theta, chart_scale, defensive_weight, root_seed):
        return _prepare_lgssm_teacher_inputs_core(
            observations, center_theta, chart_scale, defensive_weight, root_seed,
            time_steps=time_steps, fit_rows=fit_rows, basis_size=basis_size,
            rank=rank, sweeps=sweeps, operators=operators,
        )

    return prepare


def _prepare_lgssm_teacher_inputs_core(observations, center_theta, chart_scale,
    defensive_weight, root_seed, *, time_steps, fit_rows, basis_size, rank, sweeps, operators):
    dtype = observations.dtype
    dimension = 2 * STATE_DIMENSION
    reference_points64 = (tf.constant(-0.95, tf.float64) + tf.constant(1.9, tf.float64)
        * philox_uniform_float64([fit_rows, dimension], tf.stack([root_seed, 3107])))
    reference_points = tf.cast(reference_points64, dtype)
    basis_values = _legendre_basis_values(tf.transpose(reference_points), basis_size)
    query_basis_values = tf.concat([basis_values[STATE_DIMENSION:], basis_values[STATE_DIMENSION:]], axis=0)
    mask = _active_mask(dimension, rank, basis_size, dtype)
    raw_cores = tf.cast(0.03, dtype) * teacher_normal(
        [dimension, rank, basis_size, rank], tf.stack([root_seed, 3119]), dtype)
    initial_cores = raw_cores * mask
    constant_path = tf.concat([tf.range(dimension)[:, None], tf.zeros([dimension, 3], tf.int32)], axis=1)
    initial_cores = tf.tensor_scatter_nd_update(
        initial_cores, constant_path, tf.ones([dimension], dtype)
    )
    operator_powers64 = tf.broadcast_to(operators, [dimension, 5, basis_size, basis_size])
    defensive_moments = tf.constant(
        [1.0, 0.0, 1.0 / 3.0, 0.0, 1.0 / 5.0], dtype
    )
    scale = tf.cast(chart_scale, dtype)
    state_matrix = tf.concat(
        [scale * tf.eye(STATE_DIMENSION, dtype=dtype),
         tf.zeros([STATE_DIMENSION, STATE_DIMENSION], dtype)],
        axis=1,
    )
    initial_shift_indices = tf.zeros([time_steps], tf.int32)
    prepared = {
        "observations": observations,
        "reference_points": reference_points,
        "basis_values": basis_values,
        "active_mask": mask,
        "schedule": tf.tile(tf.range(dimension, dtype=tf.int32), [sweeps]),
        "weights": tf.fill([fit_rows], tf.cast(1.0 / fit_rows, dtype)),
        "initial_cores": initial_cores,
        "scale_shift_indices": initial_shift_indices,
        "defensive_weights": tf.fill([time_steps], tf.cast(defensive_weight, dtype)),
        "query_basis_values": query_basis_values,
        "keep_mask": tf.constant([True, True, True, False, False, False]),
        "mass_operators": tf.cast(operator_powers64[:, 0], dtype),
        "defensive_marginal_values": tf.ones([time_steps, fit_rows], dtype),
        "defensive_mass": tf.constant(1.0, dtype),
        "operator_powers": tf.cast(operator_powers64, dtype),
        "defensive_power_moments": tf.tile(defensive_moments[None, :], [dimension, 1]),
        "state_offset": tf.zeros([STATE_DIMENSION], dtype),
        "state_matrix": state_matrix,
        "pair_indices": tf.constant(PAIR_INDICES, tf.int32),
        "chart_scale": scale,
        "center_theta": tf.convert_to_tensor(center_theta, dtype),
    }
    base, _, _ = _teacher_base_log_targets(prepared["center_theta"], prepared)
    prepared["scale_shift_indices"] = tf.argmax(base, axis=1, output_type=tf.int32)
    return prepared


def _teacher_base_log_targets(
    theta: tf.Tensor, prepared: Mapping[str, tf.Tensor]
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Return adjacent log factors, their five directions, and chart validity."""

    dtype = prepared["reference_points"].dtype
    theta = tf.reshape(tf.convert_to_tensor(theta, dtype), [PARAMETER_COUNT])
    reference = prepared["reference_points"]
    scale = tf.cast(prepared["chart_scale"], dtype)
    current = scale * reference[:, :STATE_DIMENSION]
    previous = scale * reference[:, STATE_DIMENSION:]
    components = _lgssm_components(theta, 1)
    tangents = _lgssm_component_tangents(theta, 1)
    transition_mean = previous * components["phi"][None, :]
    transition_mean_tangent = previous[:, :, None] * tf.concat(
        [tf.eye(STATE_DIMENSION, dtype=dtype), tf.zeros([STATE_DIMENSION, 2], dtype)],
        axis=1,
    )[None, :, :]
    transition = _gaussian_log_density_jvp_core(
        (current - transition_mean)[None, :, :],
        components["transition_covariance"],
        (-transition_mean_tangent)[None, :, :, :],
        tangents["d_transition_covariance"],
    )
    predicted = tf.linalg.matmul(current, components["observation_matrix"], transpose_b=True)
    zero_residual_tangent = tf.zeros(
        [1, tf.shape(current)[0], OBSERVATION_DIMENSION, PARAMETER_COUNT],
        dtype,
    )
    horizon = prepared["observations"].shape[0]
    observation = _gaussian_log_density_jvp_core(
        predicted[None] - prepared["observations"][:, None],
        tf.broadcast_to(components["observation_covariance"], [horizon, OBSERVATION_DIMENSION, OBSERVATION_DIMENSION]),
        tf.broadcast_to(zero_residual_tangent, [horizon, tf.shape(current)[0], OBSERVATION_DIMENSION, PARAMETER_COUNT]),
        tf.broadcast_to(tangents["d_observation_covariance"], [horizon, OBSERVATION_DIMENSION, OBSERVATION_DIMENSION, PARAMETER_COUNT]),
    )
    per_time_values = transition["value"] + observation["value"]
    per_time_tangents = transition["tangent"] + observation["tangent"]
    initial_covariance = tf.linalg.diag(tf.square(components["initial_std"]))[None, :, :]
    initial_covariance_tangent = tf.einsum(
        "ij,ip->ijp", tf.eye(STATE_DIMENSION, dtype=dtype),
        2.0 * components["initial_std"][:, None] * tangents["d_initial_std"],
    )[None]
    initial = _gaussian_log_density_jvp_core(
        previous[None, :, :],
        initial_covariance,
        tf.zeros([1, tf.shape(previous)[0], STATE_DIMENSION, PARAMETER_COUNT], dtype),
        initial_covariance_tangent,
    )
    per_time_values = tf.tensor_scatter_nd_add(per_time_values, [[0]], initial["value"])
    per_time_tangents = tf.tensor_scatter_nd_add(per_time_tangents, [[0]], initial["tangent"])
    # Uniform reference density and the affine chart contribute this fixed factor.
    chart_log_factor = tf.cast(2 * STATE_DIMENSION, dtype) * tf.math.log(
        2.0 * scale
    )
    values = per_time_values + chart_log_factor
    tangent_values = per_time_tangents
    return values, tangent_values, _physical_chart(theta)


def _teacher_targets(
    theta: tf.Tensor,
    prepared: Mapping[str, tf.Tensor],
    controls: MomentTeacherControls,
    *,
    setup_static: bool = False,
) -> dict[str, tf.Tensor]:
    base, base_tangents, physical_valid = _teacher_base_log_targets(theta, prepared)
    return teacher_directions(base, base_tangents, physical_valid, prepared, controls,
                              setup_static=setup_static)


def freeze_teacher_scale_shift_indices(
    teacher_prepared: Mapping[str, tf.Tensor],
    controls: MomentTeacherControls,
    *,
    maximum_iterations: int = 8,
    jit_compile: bool = True,
) -> dict[str, tf.Tensor]:
    """Freeze center-parameter maximizing rows including carried marginals."""

    if maximum_iterations <= 0:
        raise ValueError("scale-shift fixed-point iterations must be positive")
    prepared = dict(teacher_prepared)
    specification = tuple((name, tf.TensorSpec(value.shape, value.dtype))
                          for name, value in prepared.items())
    indices, valid, stable = _lgssm_freeze_program(
        specification, controls, maximum_iterations, bool(jit_compile))(prepared)
    if not bool(valid):
        raise ValueError("teacher scale-shift freeze encountered an invalid TT fit")
    if not bool(stable):
        raise ValueError("teacher scale-shift maximizing-row fixed point did not stabilize")
    prepared["scale_shift_indices"] = indices
    return prepared


@lru_cache(maxsize=16)
def _lgssm_freeze_program(specification, controls, maximum_iterations, jit_compile):
    @tf.function(input_signature=[dict(specification)], jit_compile=jit_compile, autograph=False)
    def freeze(prepared):
        theta = prepared["center_theta"]
        base, _, _ = _teacher_base_log_targets(theta, prepared)

        def targets(indices):
            current = dict(prepared, scale_shift_indices=indices)
            return _teacher_targets(theta, current, controls, setup_static=True)

        return freeze_scale_shift_core(base, prepared["scale_shift_indices"], targets, maximum_iterations)

    return freeze


def _scatter_pair_values(values: tf.Tensor, pair_indices: tf.Tensor) -> tf.Tensor:
    return tf.scatter_nd(pair_indices, values, [STATE_DIMENSION, STATE_DIMENSION])


def _scatter_pair_tangents(values: tf.Tensor, pair_indices: tf.Tensor) -> tf.Tensor:
    return tf.scatter_nd(
        pair_indices,
        values,
        [STATE_DIMENSION, STATE_DIMENSION, PARAMETER_COUNT],
    )


def moment_teacher_lgssm_value_and_score_core(
    theta: tf.Tensor,
    particle_prepared: Mapping[str, tf.Tensor],
    teacher_prepared: Mapping[str, tf.Tensor],
    controls: MomentTeacherControls,
    *,
    row_chunk_size: int,
    col_chunk_size: int,
) -> dict[str, tf.Tensor]:
    """Evaluate the exact finite hybrid likelihood and its manual score."""

    initial_noise = particle_prepared["initial_noise"]
    if initial_noise.shape[0] != 1:
        raise ValueError("the first moment-teacher integration route requires batch size one")
    dtype = initial_noise.dtype
    theta = tf.reshape(tf.convert_to_tensor(theta, dtype), [PARAMETER_COUNT])
    teacher = _teacher_targets(theta, teacher_prepared, controls)
    components = _lgssm_components(theta, 1)
    tangents = _lgssm_component_tangents(theta, 1)
    particles = initial_noise * components["initial_std"][None, None, :]
    particles_tangent = (
        initial_noise[:, :, :, None] * tangents["d_initial_std"][None, None, :, :]
    )
    particle_count = int(initial_noise.shape[1])
    uniform_log_weight = -tf.math.log(tf.cast(particle_count, dtype))
    log_weights = tf.fill([1, particle_count], uniform_log_weight)
    log_weights_tangent = tf.zeros([1, particle_count, PARAMETER_COUNT], dtype)
    likelihood = tf.zeros([1], dtype)
    score = tf.zeros([1, PARAMETER_COUNT], dtype)
    valid = tf.fill([1], teacher["valid"])
    time_steps = int(particle_prepared["observations"].shape[0])
    shape_valid_history = tf.TensorArray(tf.bool, size=time_steps, element_shape=[])
    skew_residual_history = tf.TensorArray(
        dtype, size=time_steps, element_shape=[STATE_DIMENSION]
    )
    mean_residual_history = tf.TensorArray(dtype, size=time_steps, element_shape=[])
    covariance_residual_history = tf.TensorArray(
        dtype, size=time_steps, element_shape=[]
    )
    pair_indices = teacher_prepared["pair_indices"]
    off_diagonal = 1.0 - tf.eye(STATE_DIMENSION, dtype=dtype)

    def body(
        time_index,
        current_particles,
        current_particles_tangent,
        current_log_weights,
        current_log_weights_tangent,
        current_likelihood,
        current_score,
        current_valid,
        shape_history,
        residual_history,
        mean_history,
        covariance_history,
    ):
        step = _canonical_fused_step_core(
            time_index,
            current_particles,
            current_particles_tangent,
            current_log_weights,
            current_log_weights_tangent,
            components,
            tangents,
            particle_prepared,
            steps=controls.sinkhorn_steps,
            balance_steps=controls.balance_steps,
            row_chunk_size=row_chunk_size,
            col_chunk_size=col_chunk_size,
            execute_contract_e=True,
        )
        corrected = higher_moment_shape_jvp(
            step["weighted_source_particles"][0],
            step["normalized_weights"][0],
            step["weighted_source_particles_tangent"][0],
            step["normalized_weights_tangent"][0],
            step["particles"][0],
            step["particles_tangent"][0],
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
            explicit_target_pairwise_co_skew=_scatter_pair_values(
                teacher["co_skew"][time_index], pair_indices
            ),
            explicit_target_pairwise_co_kurtosis=_scatter_pair_values(
                teacher["co_kurtosis"][time_index], pair_indices
            ),
            explicit_target_pairwise_co_skew_tangent=_scatter_pair_tangents(
                teacher["co_skew_tangent"][time_index], pair_indices
            ),
            explicit_target_pairwise_co_kurtosis_tangent=_scatter_pair_tangents(
                teacher["co_kurtosis_tangent"][time_index], pair_indices
            ),
            pairwise_co_skew_target_mask=off_diagonal,
            pairwise_co_kurtosis_target_mask=off_diagonal,
        )
        source_mean = tf.reduce_sum(
            step["normalized_weights"][0, :, None]
            * step["weighted_source_particles"][0],
            axis=0,
        )
        source_centered = step["weighted_source_particles"][0] - source_mean[None, :]
        source_covariance = tf.einsum(
            "n,ni,nj->ij",
            step["normalized_weights"][0],
            source_centered,
            source_centered,
        )
        corrected_mean = tf.reduce_mean(corrected["particles"], axis=0)
        corrected_centered = corrected["particles"] - corrected_mean[None, :]
        corrected_covariance = tf.einsum(
            "ni,nj->ij", corrected_centered, corrected_centered
        ) / tf.cast(particle_count, dtype)
        mean_residual = tf.reduce_max(tf.abs(corrected_mean - source_mean))
        covariance_residual = tf.reduce_max(
            tf.abs(corrected_covariance - source_covariance)
        )
        step_valid = step["time_valid"] & corrected["valid"][None]
        return (
            time_index + 1,
            corrected["particles"][None, :, :],
            corrected["particles_tangent"][None, :, :, :],
            step["log_weights"],
            step["log_weights_tangent"],
            current_likelihood + step["increment"],
            current_score + step["increment_tangent"],
            current_valid & step_valid,
            shape_history.write(time_index, corrected["valid"]),
            residual_history.write(time_index, corrected["skew_residual"]),
            mean_history.write(time_index, mean_residual),
            covariance_history.write(time_index, covariance_residual),
        )

    result = tf.while_loop(
        lambda time_index, *_: time_index < time_steps,
        body,
        (
            tf.zeros([], tf.int32),
            particles,
            particles_tangent,
            log_weights,
            log_weights_tangent,
            likelihood,
            score,
            valid,
            shape_valid_history,
            skew_residual_history,
            mean_residual_history,
            covariance_residual_history,
        ),
        maximum_iterations=time_steps,
        parallel_iterations=1,
    )
    return {
        "objective": result[5][0],
        "score": result[6][0],
        "valid_chart": result[7],
        "final_particles": result[1],
        "final_particles_tangent": result[2],
        "teacher_valid": teacher["valid"],
        "teacher_normalizers": teacher["normalizers"],
        "shape_valid_history": result[8].stack(),
        "skew_residual_history": result[9].stack(),
        "mean_residual_history": result[10].stack(),
        "covariance_residual_history": result[11].stack(),
    }


@tf.function(jit_compile=True, reduce_retracing=True)
def moment_teacher_lgssm_value_and_score_tf(
    theta: tf.Tensor,
    particle_prepared: Mapping[str, tf.Tensor],
    teacher_prepared: Mapping[str, tf.Tensor],
) -> dict[str, tf.Tensor]:
    """Factory identity anchor using the reviewed mechanics control fixture."""

    chunks = int(particle_prepared["initial_noise"].shape[1])
    controls = MomentTeacherControls(
        sinkhorn_steps=2,
        balance_steps=2,
        correction_steps=1,
        correction_strength=0.05,
        correction_floor=1.0e-6,
        pairwise_correction_steps=1,
        pairwise_strength=0.02,
        pairwise_floor=1.0e-6,
        tt_ridge=1.0e-5,
        column_scale_floor=1.0e-6,
        condition_number_veto=1.0e8,
        fit_residual_veto=2.0,
    )
    return moment_teacher_lgssm_value_and_score_core(
        theta,
        particle_prepared,
        teacher_prepared,
        controls,
        row_chunk_size=chunks,
        col_chunk_size=chunks,
    )


def make_moment_teacher_lgssm_value_and_score_tf(
    particle_prepared: Mapping[str, tf.Tensor],
    teacher_prepared: Mapping[str, tf.Tensor],
    tuning_artifact: MomentTeacherTuningArtifact,
    *,
    expected_scope: LEDHTuningScope,
    jit_compile: bool = True,
):
    """Bind one exact prepared/tuned scope into a compiled value-score graph."""

    if not isinstance(tuning_artifact, MomentTeacherTuningArtifact):
        raise TypeError("moment-teacher route requires a repository-issued tuning artifact")
    if tuning_artifact._seal not in _ISSUED_TUNING_SEALS:
        raise TypeError("moment-teacher tuning artifact has no repository issuance")
    require_scope_match(
        expected_scope,
        tuning_artifact.scope.as_dict(),
        label="moment-teacher tuning artifact",
    )
    initial_noise = particle_prepared["initial_noise"]
    reset_activity = tf.get_static_value(
        tf.reduce_all(particle_prepared["fixed_reset_mask"])
    )
    if reset_activity is None or not bool(reset_activity):
        raise ValueError(
            "the first moment-teacher integration route requires a statically all-active reset mask"
        )
    chunks = validate_transport_chunks(
        int(initial_noise.shape[1]),
        row_chunk_size=expected_scope.row_chunk_size,
        col_chunk_size=expected_scope.col_chunk_size,
    )
    if expected_scope.dtype != initial_noise.dtype.name:
        raise ValueError("moment-teacher tuning dtype does not match prepared particles")
    if expected_scope.horizon != int(particle_prepared["observations"].shape[0]):
        raise ValueError("moment-teacher tuning horizon does not match prepared particles")
    if expected_scope.particle_count != int(initial_noise.shape[1]):
        raise ValueError("moment-teacher tuning particle count does not match prepared particles")
    controls = tuning_artifact.controls

    @tf.function(
        input_signature=[tf.TensorSpec([PARAMETER_COUNT], initial_noise.dtype)],
        jit_compile=jit_compile,
        reduce_retracing=True,
    )
    def evaluate(theta: tf.Tensor) -> dict[str, tf.Tensor]:
        return moment_teacher_lgssm_value_and_score_core(
            theta,
            particle_prepared,
            teacher_prepared,
            controls,
            row_chunk_size=chunks.row_chunk_size,
            col_chunk_size=chunks.col_chunk_size,
        )

    return evaluate


def make_moment_teacher_lgssm_prepared_particles_tf(
    teacher_prepared: Mapping[str, tf.Tensor],
    tuning_artifact: MomentTeacherTuningArtifact,
    *,
    expected_scope: LEDHTuningScope,
    jit_compile: bool = True,
):
    """Compile once per scope while accepting seed-specific particle inputs."""

    if tuning_artifact._seal not in _ISSUED_TUNING_SEALS:
        raise TypeError("moment-teacher tuning artifact has no repository issuance")
    require_scope_match(
        expected_scope,
        tuning_artifact.scope.as_dict(),
        label="moment-teacher tuning artifact",
    )
    chunks = validate_transport_chunks(
        expected_scope.particle_count,
        row_chunk_size=expected_scope.row_chunk_size,
        col_chunk_size=expected_scope.col_chunk_size,
    )
    dtype = tf.dtypes.as_dtype(expected_scope.dtype)
    particle_signature = {
        "observations": tf.TensorSpec([expected_scope.horizon, OBSERVATION_DIMENSION], dtype),
        "initial_noise": tf.TensorSpec([1, expected_scope.particle_count, STATE_DIMENSION], dtype),
        "transition_noise": tf.TensorSpec(
            [1, expected_scope.horizon, expected_scope.particle_count, STATE_DIMENSION], dtype
        ),
        "fixed_reset_mask": tf.TensorSpec([1, expected_scope.horizon], tf.bool),
        "residual_design": tf.TensorSpec(
            [1, expected_scope.horizon, expected_scope.particle_count, STATE_DIMENSION], dtype
        ),
        "prepared_ridge": tf.TensorSpec([1, expected_scope.horizon], dtype),
        "epsilon": tf.TensorSpec([], dtype),
        "scaling": tf.TensorSpec([], dtype),
    }
    controls = tuning_artifact.controls

    @tf.function(
        input_signature=[tf.TensorSpec([PARAMETER_COUNT], dtype), particle_signature],
        jit_compile=jit_compile,
        reduce_retracing=True,
    )
    def evaluate(
        theta: tf.Tensor, particle_prepared: Mapping[str, tf.Tensor]
    ) -> dict[str, tf.Tensor]:
        return moment_teacher_lgssm_value_and_score_core(
            theta,
            particle_prepared,
            teacher_prepared,
            controls,
            row_chunk_size=chunks.row_chunk_size,
            col_chunk_size=chunks.col_chunk_size,
        )

    return evaluate


def route_identity_prepared_inputs(
    particle_prepared: Mapping[str, tf.Tensor],
    teacher_prepared: Mapping[str, tf.Tensor],
    tuning_artifact: MomentTeacherTuningArtifact,
) -> dict[str, Any]:
    """Build the exact factory payload for this prepared/tuned finite route."""

    if not isinstance(tuning_artifact, MomentTeacherTuningArtifact):
        raise TypeError("route identity requires a repository-issued tuning artifact")
    controls = tuning_artifact.controls
    scope = tuning_artifact.scope
    result: dict[str, Any] = {
        name: particle_prepared[name]
        for name in (
            "observations",
            "initial_noise",
            "transition_noise",
            "fixed_reset_mask",
            "residual_design",
            "prepared_ridge",
            "epsilon",
            "scaling",
        )
    }
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
        "pair_indices",
        "chart_scale",
        "center_theta",
    ):
        result[name] = teacher_prepared[name]
    result.update(controls.as_dict())
    result["row_chunk_size"] = scope.row_chunk_size
    result["col_chunk_size"] = scope.col_chunk_size
    return result


def make_lgssm_tuning_scope(
    *,
    horizon: int,
    prepared_data_id: str,
    particle_count: int,
    dtype: tf.dtypes.DType,
    tf32_enabled: bool,
    jit_compile: bool,
) -> LEDHTuningScope:
    chunks = select_transport_chunks(particle_count)
    return LEDHTuningScope(
        model_id="canonical_lgssm_m3_v1",
        target_id="finite_particle_observed_data_log_likelihood",
        route_id=ROUTE_ID,
        reset_contract_id=RESET_CONTRACT_ID,
        horizon=horizon,
        prepared_data_id=prepared_data_id,
        particle_count=particle_count,
        state_dimension=STATE_DIMENSION,
        parameter_count=PARAMETER_COUNT,
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


__all__ = [
    "CONTROL_FAMILY_ID",
    "MomentTeacherControls",
    "MomentTeacherTuningArtifact",
    "ROUTE_ID",
    "ROUTE_SPECIFICATION_ID",
    "issue_moment_teacher_tuning_artifact",
    "freeze_teacher_scale_shift_indices",
    "make_lgssm_tuning_scope",
    "make_moment_teacher_lgssm_prepared_particles_tf",
    "make_moment_teacher_lgssm_value_and_score_tf",
    "moment_teacher_lgssm_value_and_score_core",
    "moment_teacher_lgssm_value_and_score_tf",
    "prepare_lgssm_teacher_inputs",
    "route_identity_prepared_inputs",
]
