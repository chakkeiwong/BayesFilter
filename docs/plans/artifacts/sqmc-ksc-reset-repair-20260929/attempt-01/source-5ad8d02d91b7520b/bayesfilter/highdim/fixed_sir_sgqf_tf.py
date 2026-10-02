"""Fixed-parameter Zhao--Cui Austria SIR SGQF value route."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import tensorflow as tf

from bayesfilter.highdim.models import zhao_cui_sir_austria_model
from bayesfilter.nonlinear.fixed_sgqf_tf import tf_fixed_sgqf_level2_axis_cloud


FIXED_SIR_SGQF_ROW_ID = "zhao_cui_spatial_sir_austria_j9_T20"
FIXED_SIR_SGQF_ROUTE_ID = (
    "fixed_sgqf_zhao_cui_sir_austria_j9_t20_transition_then_observe_value_only_v1"
)
FIXED_SIR_SGQF_TARGET_ID = (
    "zhao_cui_sir_austria_fixed_kappa_nu_tf_seed81103_y1_y20_v1"
)
FIXED_SIR_SGQF_SEED = 81103
FIXED_SIR_SGQF_HORIZON = 20
FIXED_SIR_SGQF_STATE_DIM = 18
FIXED_SIR_SGQF_OBSERVATION_DIM = 9
FIXED_SIR_SGQF_STATE_SHA256 = (
    "67de785712b3af3a464b7b318c2f0a517062d2ae65db73f62ba97c6e6bc63793"
)
FIXED_SIR_SGQF_OBSERVATION_SHA256 = (
    "311975f378572d577a4efff005156bf736eeb36ab08824323a7d36231beffbda"
)
FIXED_SIR_SGQF_CLOUD_SHA256 = (
    "a303685144ad6abeaa0a88f49bd0e5df0bc3af2c9193df33e9ee7d09054fc2c8"
)

_MODEL = zhao_cui_sir_austria_model()
_CLOUD = tf_fixed_sgqf_level2_axis_cloud(FIXED_SIR_SGQF_STATE_DIM)
_OBSERVATION_INDICES = tf.constant(tuple(range(1, FIXED_SIR_SGQF_STATE_DIM, 2)))
_LOG_TWO_PI = tf.constant(1.8378770664093453, dtype=tf.float64)
_MIN_VARIANCE = tf.constant(1.0e-12, dtype=tf.float64)


def _tensor_hash(value: tf.Tensor) -> str:
    tensor = tf.convert_to_tensor(value, dtype=tf.float64)
    return hashlib.sha256(bytes(tf.io.serialize_tensor(tensor).numpy())).hexdigest()


def _semantic_hash(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def generate_fixed_sir_source_dataset_tf() -> tuple[tf.Tensor, tf.Tensor]:
    """Replay ``x0:x20`` and ``y1:y20`` in author-program event order."""

    with tf.device("/CPU:0"):
        model = _MODEL
        generator = tf.random.Generator.from_seed(FIXED_SIR_SGQF_SEED)
        initial_factor = tf.linalg.cholesky(model.initial_covariance)
        process_noise = generator.normal([model.state_dim()], dtype=tf.float64)
        state = model.initial_mean + tf.linalg.matvec(initial_factor, process_noise)
        observation_factor = tf.linalg.cholesky(model.observation_covariance)
        states = [state]
        observations = []
        for time_index in range(FIXED_SIR_SGQF_HORIZON):
            state = model.transition_push_from_standard_normal(
                tf.zeros([0], dtype=tf.float64),
                state[tf.newaxis, :],
                generator.normal([model.state_dim()], dtype=tf.float64),
                time_index,
            )[0]
            observation = model.infectious_components(state)[0] + tf.linalg.matvec(
                observation_factor,
                generator.normal([model.observation_dim()], dtype=tf.float64),
            )
            states.append(state)
            observations.append(observation)

        state_path = tf.stack(states)
        observation_path = tf.stack(observations)
        if _tensor_hash(state_path) != FIXED_SIR_SGQF_STATE_SHA256:
            raise ValueError("fixed SIR source-order state hash mismatch")
        if _tensor_hash(observation_path) != FIXED_SIR_SGQF_OBSERVATION_SHA256:
            raise ValueError("fixed SIR source-order observation hash mismatch")
        return state_path, observation_path


def fixed_sir_sgqf_value_only_status(
    observations: tf.Tensor,
) -> tuple[tf.Tensor, Mapping[str, tf.Tensor]]:
    """Run prefix-capable fixed-parameter mechanics without issuing identity."""

    values = tf.convert_to_tensor(observations)
    if values.dtype != tf.float64:
        raise ValueError("fixed SIR SGQF observations must use float64")
    if values.shape.rank != 2 or values.shape[1] != FIXED_SIR_SGQF_OBSERVATION_DIM:
        raise ValueError("fixed SIR SGQF observations require shape [T, 9]")
    points = tf.convert_to_tensor(_CLOUD.points, dtype=tf.float64)
    weights = tf.convert_to_tensor(_CLOUD.weights, dtype=tf.float64)
    mean = tf.convert_to_tensor(_MODEL.initial_mean, dtype=tf.float64)
    covariance = tf.convert_to_tensor(_MODEL.initial_covariance, dtype=tf.float64)
    process_covariance = tf.convert_to_tensor(
        _MODEL.process_covariance, dtype=tf.float64
    )
    observation_covariance = tf.convert_to_tensor(
        _MODEL.observation_covariance, dtype=tf.float64
    )
    total_value = tf.constant(0.0, dtype=tf.float64)
    valid = tf.constant(True)
    min_predictive = tf.constant(float("inf"), dtype=tf.float64)
    min_innovation = tf.constant(float("inf"), dtype=tf.float64)
    min_filtered = tf.constant(float("inf"), dtype=tf.float64)

    def symmetrize(matrix: tf.Tensor) -> tf.Tensor:
        return 0.5 * (matrix + tf.transpose(matrix))

    def covariance_from(centered: tf.Tensor) -> tf.Tensor:
        return symmetrize(tf.einsum("r,ri,rj->ij", weights, centered, centered))

    def body(
        index: tf.Tensor,
        current_mean: tf.Tensor,
        current_covariance: tf.Tensor,
        value_total: tf.Tensor,
        current_valid: tf.Tensor,
        current_min_predictive: tf.Tensor,
        current_min_innovation: tf.Tensor,
        current_min_filtered: tf.Tensor,
    ):
        previous_factor = tf.linalg.cholesky(current_covariance)
        previous_points = current_mean[tf.newaxis, :] + points @ tf.transpose(
            previous_factor
        )
        transition_values = _MODEL.transition_mean(previous_points)
        predicted_mean = tf.einsum("r,ri->i", weights, transition_values)
        centered_predicted = transition_values - predicted_mean[tf.newaxis, :]
        predicted_covariance = symmetrize(
            process_covariance + covariance_from(centered_predicted)
        )
        predicted_factor = tf.linalg.cholesky(predicted_covariance)
        predictive_points = predicted_mean[tf.newaxis, :] + points @ tf.transpose(
            predicted_factor
        )
        observation_points = tf.gather(
            predictive_points, _OBSERVATION_INDICES, axis=1
        )
        observation_mean = tf.einsum("r,ri->i", weights, observation_points)
        centered_observation = observation_points - observation_mean[tf.newaxis, :]
        innovation_covariance = symmetrize(
            observation_covariance + covariance_from(centered_observation)
        )
        centered_state = predictive_points - predicted_mean[tf.newaxis, :]
        cross_covariance = tf.einsum(
            "r,ri,rj->ij", weights, centered_state, centered_observation
        )
        innovation = values[index] - observation_mean
        innovation_factor = tf.linalg.cholesky(innovation_covariance)
        innovation_precision = tf.linalg.cholesky_solve(
            innovation_factor,
            tf.eye(FIXED_SIR_SGQF_OBSERVATION_DIM, dtype=tf.float64),
        )
        innovation_solve = tf.linalg.matvec(innovation_precision, innovation)
        increment = -0.5 * (
            tf.cast(FIXED_SIR_SGQF_OBSERVATION_DIM, tf.float64) * _LOG_TWO_PI
            + 2.0
            * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(innovation_factor)))
            + tf.reduce_sum(innovation * innovation_solve)
        )
        gain = cross_covariance @ innovation_precision
        filtered_mean = predicted_mean + tf.linalg.matvec(gain, innovation)
        filtered_covariance = symmetrize(
            predicted_covariance
            - gain @ innovation_covariance @ tf.transpose(gain)
        )
        predictive_eigenvalue = tf.reduce_min(tf.linalg.eigvalsh(predicted_covariance))
        innovation_eigenvalue = tf.reduce_min(
            tf.linalg.eigvalsh(innovation_covariance)
        )
        filtered_eigenvalue = tf.reduce_min(tf.linalg.eigvalsh(filtered_covariance))
        step_valid = tf.logical_and(
            predictive_eigenvalue > _MIN_VARIANCE,
            tf.logical_and(
                innovation_eigenvalue > _MIN_VARIANCE,
                filtered_eigenvalue > _MIN_VARIANCE,
            ),
        )
        step_valid = tf.logical_and(step_valid, tf.math.is_finite(increment))
        return (
            index + 1,
            filtered_mean,
            filtered_covariance,
            value_total + increment,
            tf.logical_and(current_valid, step_valid),
            tf.minimum(current_min_predictive, predictive_eigenvalue),
            tf.minimum(current_min_innovation, innovation_eigenvalue),
            tf.minimum(current_min_filtered, filtered_eigenvalue),
        )

    result = tf.while_loop(
        lambda index, *_unused: index < tf.shape(values)[0],
        body,
        (
            tf.constant(0, dtype=tf.int32),
            mean,
            covariance,
            total_value,
            valid,
            min_predictive,
            min_innovation,
            min_filtered,
        ),
        parallel_iterations=1,
    )
    return result[3], {
        "status_code": tf.where(result[4], tf.constant(0), tf.constant(1)),
        "valid_value": result[4],
        "min_predictive_eigenvalue": result[5],
        "min_innovation_eigenvalue": result[6],
        "min_filtered_eigenvalue": result[7],
    }


@dataclass(frozen=True)
class FixedSIRSGQFRoute:
    """Sealed fixed-SIR T20 value route and repository-issued identity."""

    states: tf.Tensor
    observations: tf.Tensor
    route_identity: str
    manifest: Mapping[str, object]

    def __post_init__(self) -> None:
        states = tf.convert_to_tensor(self.states, dtype=tf.float64)
        observations = tf.convert_to_tensor(self.observations, dtype=tf.float64)
        if states.shape != (FIXED_SIR_SGQF_HORIZON + 1, FIXED_SIR_SGQF_STATE_DIM):
            raise ValueError("fixed SIR canonical states require shape [21, 18]")
        if observations.shape != (
            FIXED_SIR_SGQF_HORIZON,
            FIXED_SIR_SGQF_OBSERVATION_DIM,
        ):
            raise ValueError("fixed SIR canonical observations require shape [20, 9]")
        if _tensor_hash(states) != FIXED_SIR_SGQF_STATE_SHA256:
            raise ValueError("fixed SIR canonical state identity rejected")
        if _tensor_hash(observations) != FIXED_SIR_SGQF_OBSERVATION_SHA256:
            raise ValueError("fixed SIR canonical observation identity rejected")
        manifest = dict(self.manifest)
        if self.route_identity != _semantic_hash(manifest):
            raise ValueError("fixed SIR SGQF route identity rejected")
        object.__setattr__(self, "states", states)
        object.__setattr__(self, "observations", observations)
        object.__setattr__(self, "manifest", MappingProxyType(manifest))

    @property
    def parameter_dim(self) -> int:
        return 0

    @property
    def required_result_kind(self) -> str:
        return "value_only_no_free_theta"

    def value_only_status(self) -> tuple[tf.Tensor, Mapping[str, tf.Tensor]]:
        return fixed_sir_sgqf_value_only_status(self.observations)


def make_fixed_sir_sgqf_route() -> FixedSIRSGQFRoute:
    """Issue the canonical source-order T20 route; callers cannot stamp identity."""

    states, observations = generate_fixed_sir_source_dataset_tf()
    model = _MODEL
    cloud = _CLOUD
    cloud_hash = hashlib.sha256(
        bytes(tf.io.serialize_tensor(cloud.points).numpy())
        + bytes(tf.io.serialize_tensor(cloud.weights).numpy())
    ).hexdigest()
    if cloud_hash != FIXED_SIR_SGQF_CLOUD_SHA256:
        raise ValueError("fixed SIR SGQF cloud hash mismatch")
    manifest: dict[str, object] = {
        "schema": "bayesfilter.fixed_sir_sgqf_route.v1",
        "row_id": FIXED_SIR_SGQF_ROW_ID,
        "route_id": FIXED_SIR_SGQF_ROUTE_ID,
        "target_id": FIXED_SIR_SGQF_TARGET_ID,
        "result_kind": "value_only_no_free_theta",
        "parameter_dim": 0,
        "seed": FIXED_SIR_SGQF_SEED,
        "time_order": "x0_then_20_transition_then_observe_steps_y1_y20",
        "horizon": FIXED_SIR_SGQF_HORIZON,
        "state_dim": FIXED_SIR_SGQF_STATE_DIM,
        "observation_dim": FIXED_SIR_SGQF_OBSERVATION_DIM,
        "state_sha256": FIXED_SIR_SGQF_STATE_SHA256,
        "observation_sha256": FIXED_SIR_SGQF_OBSERVATION_SHA256,
        "cloud_sha256": FIXED_SIR_SGQF_CLOUD_SHA256,
        "cloud_level": 2,
        "cloud_point_count": cloud.point_count,
        "cloud_negative_weight_count": cloud.negative_weight_count,
        "kappa": [float(value) for value in model.kappa.numpy()],
        "nu": [float(value) for value in model.nu.numpy()],
        "delta": float(model.delta.numpy()),
        "rk4_internal_step": float(model.rk4_internal_step.numpy()),
        "process_covariance": "I18",
        "observation_covariance": "100I9",
        "initial_covariance": "I18",
        "dtype": "float64",
        "backend": "tensorflow_fixed_level2_sgqf_value_only",
        "source_anchor": (
            "third_party/audit/zhao_cui_tensor_ssm_p10/source/models/ssmodel.m:34"
        ),
        "nonclaims": [
            "not exact nonlinear likelihood",
            "not an inferred-parameter or score route",
            "not MATLAB random-stream replay",
            "not SGQF superiority or default-readiness evidence",
        ],
    }
    return FixedSIRSGQFRoute(
        states=states,
        observations=observations,
        route_identity=_semantic_hash(manifest),
        manifest=manifest,
    )


__all__ = [
    "FIXED_SIR_SGQF_CLOUD_SHA256",
    "FIXED_SIR_SGQF_HORIZON",
    "FIXED_SIR_SGQF_OBSERVATION_SHA256",
    "FIXED_SIR_SGQF_ROUTE_ID",
    "FIXED_SIR_SGQF_ROW_ID",
    "FIXED_SIR_SGQF_SEED",
    "FIXED_SIR_SGQF_STATE_SHA256",
    "FIXED_SIR_SGQF_TARGET_ID",
    "FixedSIRSGQFRoute",
    "fixed_sir_sgqf_value_only_status",
    "generate_fixed_sir_source_dataset_tf",
    "make_fixed_sir_sgqf_route",
]
