"""Packed-XLA material replay and FD tangents for Lane-B T2 training."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping

import tensorflow as tf

from bayesfilter.highdim.models import _xla_isotropic_mvn_log_prob
from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_t2_score_tf import (
    T2_SCORE_TARGET_ID,
    t2_target_log_value_and_manual_score,
)
from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_t2_tf import (
    LaneBT2Artifact,
    LaneBT2ProposalCloud,
    build_t2_training_batch,
)
from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_training_jvp_tf import (
    ADAM_BETA_1,
    ADAM_BETA_2,
    ADAM_EPSILON,
    DTYPE,
    FUNCTIONAL_SCREEN_COLUMNS,
    FUNCTIONAL_SCREEN_ORDER,
    MATERIAL_REPLAY_ATOL,
    MATERIAL_REPLAY_POLICY_ID,
    MATERIAL_REPLAY_RTOL,
    OFFLINE_ISSUER_DERIVATIVE,
    PARAMETER_DIM,
    ROOT,
    RUNTIME_SCORE_BACKEND,
    SHIFT_DERIVATIVE_POLICY,
    TAU_DERIVATIVE_POLICY,
    _claim_local_transition_mean_xla,
    _semantic_sha256,
    load_t1_training_jvp_child,
)
from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_tf import (
    balanced_initial_cores,
    lane_b_product_basis,
)
from bayesfilter.highdim.zhao_cui_austria_sir_parameter_child_tf import (
    LaneBParameterChild,
)
from bayesfilter.highdim.zhao_cui_austria_sir_packed_xla_tf import (
    PACKED_XLA_POLICY_ID,
    material_positive_value_metrics,
    material_replay_metrics,
    pack_cores,
    pack_tangent_banks,
    packed_adam_apply_gradients,
    packed_amplitude,
    packed_core_mask,
    packed_normalized_density,
    packed_normalized_prefix_density,
    packed_per_core_regularizers,
    packed_square_mass,
    precompute_basis_values,
    precompute_mass_matrices,
    unpack_cores,
)
from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_target_tf import (
    generate_sealed_lane_b_dataset,
)
from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_t2_score_tf import (
    physical_z1_to_parent_local_prefix,
)


T2_REPLAY_ID = "lane_b_t2_packed_xla_full_cloud_adam_centered_fd_tangent_v2"
T2_ISSUER_SCHEMA = "bayesfilter.zhao_cui_austria_sir_lane_b_t2_training_tangent.v3"
T2_ISSUER_ID = "repository_owned_material_t2_packed_xla_fd_tangent_issuer_v2"
T2_OFFLINE_ISSUER_DERIVATIVE = (
    "tensorflow_xla_centered_fd_shape_plus_scalar_radial_projection_h5e5_v1"
)
T2_TANGENT_FINITE_DIFFERENCE_STEP = 5e-5
T2_FINITE_DIFFERENCE_STEP = 1e-4
T2_FINITE_DIFFERENCE_ATOL = 3e-4
T2_FINITE_DIFFERENCE_RTOL = 3e-4
T2_MEMORY_CAP_BYTES = 6 * 1024**3
T2_GPU_MEMORY_LIMIT_MIB = 6 * 1024
T2_TRAINING_PREPARED_RESULT_PATH = Path(
    "docs/plans/artifacts/zhao-cui-austria-sir-lane-b-t2-20260731/"
    "attempt-06-training-prepared-final-closure/result.json"
)
T2_CALIBRATION_PREPARED_RESULT_PATH = Path(
    "docs/plans/artifacts/zhao-cui-austria-sir-lane-b-t2-20260731/"
    "attempt-08-calibration-prepared-final-closure/result.json"
)
REQUIRED_T2_ISSUER_SOURCE_PATHS = (
    "bayesfilter/highdim/zhao_cui_austria_sir_lane_b_t2_training_jvp_tf.py",
    "bayesfilter/highdim/zhao_cui_austria_sir_lane_b_t2_score_tf.py",
    "bayesfilter/highdim/zhao_cui_austria_sir_lane_b_t2_prepared_tf.py",
    "bayesfilter/highdim/zhao_cui_austria_sir_lane_b_training_jvp_tf.py",
    "bayesfilter/highdim/zhao_cui_austria_sir_parameter_child_tf.py",
    "bayesfilter/highdim/zhao_cui_austria_sir_packed_xla_tf.py",
    "docs/plans/bayesfilter-zhao-cui-austria-sir-material-replay-xla-repair-plan-2026-08-02.md",
    "docs/plans/bayesfilter-zhao-cui-austria-sir-t2-scalar-consistency-repair-note-2026-08-02.md",
    "scripts/run_zhao_cui_austria_sir_lane_b_t2_training_jvp.py",
)


@dataclass(frozen=True)
class T2ReplayCloudInputs:
    joint_points: tf.Tensor
    local_points: tf.Tensor
    origin_log_importance_weight: tf.Tensor
    origin_target_log_value: tf.Tensor
    basis_values: tf.Tensor
    t1_prefix_basis_values: tf.Tensor


@dataclass(frozen=True)
class T2ReplayInputs:
    training: T2ReplayCloudInputs
    calibration: T2ReplayCloudInputs
    training_manifest: Mapping[str, object]
    calibration_manifest: Mapping[str, object]
    mass_matrices: tf.Tensor
    initial_packed_cores: tf.Tensor
    parent_packed_cores: tf.Tensor
    packed_mask: tf.Tensor
    t1_parent_packed_cores: tf.Tensor
    t1_packed_tangents: tf.Tensor
    observation: tf.Tensor
    microbatch_indices: tf.Tensor


@dataclass(frozen=True)
class T2ReplayJVPResult:
    cores: tuple[tf.Tensor, ...]
    tangent_cores: tuple[tuple[tf.Tensor, ...], ...]
    increment: tf.Tensor
    increment_score: tf.Tensor
    cumulative_value: tf.Tensor
    cumulative_score: tf.Tensor
    maximum_core_residual: tf.Tensor
    maximum_normalized_core_residual: tf.Tensor
    functional_replay_metrics: tf.Tensor
    finite_difference_plus: tf.Tensor
    finite_difference_minus: tf.Tensor
    raw_core_tangent_increment_score: tf.Tensor
    scalar_consistency_radial_correction: tf.Tensor
    replay_id: str = T2_REPLAY_ID


def _project_tangents_to_scalar_derivative(
    raw_tangents: tf.Tensor,
    requested_score: tf.Tensor,
    parent_base: tf.Tensor,
    mass_matrices: tf.Tensor,
    tau: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Add a first-core radial component that enforces scalar consistency."""

    with tf.GradientTape() as mass_tape:
        mass_tape.watch(parent_base)
        parent_square_mass = packed_square_mass(parent_base, mass_matrices)
    mass_gradient = mass_tape.gradient(parent_square_mass, parent_base)
    parent_normalizer = parent_square_mass + tau
    raw_score = tf.einsum(
        "pabcd,abcd->p", raw_tangents, mass_gradient
    ) / parent_normalizer
    first_axis = tf.one_hot(0, 36, dtype=DTYPE)[
        :, tf.newaxis, tf.newaxis, tf.newaxis
    ]
    radial_direction = parent_base * first_axis
    radial_response = tf.reduce_sum(
        mass_gradient * radial_direction
    ) / parent_normalizer
    radial_correction = (requested_score - raw_score) / radial_response
    corrected = raw_tangents + radial_correction[
        :, tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis
    ] * radial_direction[tf.newaxis, ...]
    return corrected, raw_score, radial_correction


def prepare_t2_replay_cloud_inputs(
    *,
    artifact: LaneBT2Artifact,
    t1_child: LaneBParameterChild,
    cloud: LaneBT2ProposalCloud,
) -> T2ReplayCloudInputs:
    """Bind one serialized proposal cloud to the frozen T2 frame."""

    if t1_child.parent.identity != artifact.parent_artifact.identity:
        raise ValueError("T2 replay T1 child parent identity mismatch")
    batch = build_t2_training_batch(cloud, artifact.frame, artifact.shift_constant)
    origin = tf.zeros([PARAMETER_DIM], DTYPE)
    target = t2_target_log_value_and_manual_score(
        t1_child, origin, cloud.joint_points
    )
    basis = lane_b_product_basis(
        order=artifact.settings.basis_order,
        num_elems=artifact.settings.basis_num_elems,
    )
    local_z1 = physical_z1_to_parent_local_prefix(t1_child, cloud.joint_points[:, 18:])
    prefix_points = tf.concat(
        [local_z1, tf.zeros([tf.shape(local_z1)[0], 18], DTYPE)], axis=1
    )
    return T2ReplayCloudInputs(
        joint_points=cloud.joint_points,
        local_points=batch.points,
        origin_log_importance_weight=cloud.log_importance_weight,
        origin_target_log_value=target["log_value"],
        basis_values=precompute_basis_values(basis, batch.points),
        t1_prefix_basis_values=precompute_basis_values(basis, prefix_points),
    )


def make_t2_replay_inputs(
    *,
    artifact: LaneBT2Artifact,
    t1_child: LaneBParameterChild,
    training_cloud: LaneBT2ProposalCloud,
    calibration_cloud: LaneBT2ProposalCloud,
) -> T2ReplayInputs:
    training = prepare_t2_replay_cloud_inputs(
            artifact=artifact, t1_child=t1_child, cloud=training_cloud
        )
    calibration = prepare_t2_replay_cloud_inputs(
            artifact=artifact, t1_child=t1_child, cloud=calibration_cloud
        )
    basis = lane_b_product_basis(
        order=artifact.settings.basis_order,
        num_elems=artifact.settings.basis_num_elems,
    )
    initial = balanced_initial_cores(artifact.settings, basis)
    sample_count = int(training.local_points.shape[0])
    microbatch_size = int(artifact.settings.batch_size)
    microbatch_count = sample_count // microbatch_size
    microbatch_indices = tf.reshape(
        tf.range(sample_count, dtype=tf.int32), [microbatch_count, microbatch_size]
    )
    _states, observations, _all = generate_sealed_lane_b_dataset()
    return T2ReplayInputs(
        training=training,
        calibration=calibration,
        training_manifest=training_cloud.manifest_payload(),
        calibration_manifest=calibration_cloud.manifest_payload(),
        mass_matrices=precompute_mass_matrices(basis),
        initial_packed_cores=pack_cores(initial),
        parent_packed_cores=pack_cores(artifact.cores),
        packed_mask=packed_core_mask(tuple(core.shape for core in artifact.cores)),
        t1_parent_packed_cores=pack_cores(t1_child.parent_cores),
        t1_packed_tangents=pack_tangent_banks(t1_child.tangent_cores),
        observation=tf.ensure_shape(observations[1], [9]),
        microbatch_indices=tf.ensure_shape(
            microbatch_indices, [microbatch_count, microbatch_size]
        ),
    )


def _packed_t2_target_log_value(
    theta: tf.Tensor,
    cloud: T2ReplayCloudInputs,
    inputs: T2ReplayInputs,
    tau: tf.Tensor,
) -> tf.Tensor:
    """Evaluate the carried T1 marginal and T2 row target without host code."""

    parameters = tf.reshape(tf.convert_to_tensor(theta, DTYPE), [PARAMETER_DIM])
    conditioned_t1 = inputs.t1_parent_packed_cores + tf.einsum(
        "p,pabcd->abcd", parameters, inputs.t1_packed_tangents
    )
    previous_density = packed_normalized_prefix_density(
        conditioned_t1,
        cloud.t1_prefix_basis_values,
        inputs.mass_matrices,
        tau,
    )
    points = cloud.joint_points
    z2 = points[:, :18]
    z1 = points[:, 18:]
    transition_mean = _claim_local_transition_mean_xla(parameters, z1)
    transition = _xla_isotropic_mvn_log_prob(
        z2 - transition_mean, tf.constant(1.0, DTYPE)
    )
    observation_variance = tf.constant(100.0, DTYPE) * tf.exp(
        tf.constant(2.0, DTYPE) * parameters[2]
    )
    likelihood = _xla_isotropic_mvn_log_prob(
        inputs.observation[tf.newaxis, :] - z2[:, 1::2], observation_variance
    )
    return tf.math.log(previous_density) + transition + likelihood


def _make_t2_compiled_primal(
    t1_child: LaneBParameterChild,
    artifact: LaneBT2Artifact,
    inputs: T2ReplayInputs,
):
    """Build one complete packed T2 replay with TensorFlow-owned control flow."""

    settings = artifact.settings
    tau = tf.constant(settings.tau, DTYPE)
    l1 = tf.constant(settings.l1_weight, DTYPE)
    l2 = tf.constant(settings.l2_weight, DTYPE)
    learning_rate = tf.cast(tf.constant(settings.learning_rate, tf.float32), DTYPE)
    clip = tf.constant(settings.gradient_clip_norm, DTYPE)
    sample_count = int(inputs.training.local_points.shape[0])
    microbatch_size = int(settings.batch_size)
    if sample_count % microbatch_size:
        raise ValueError("T2 replay cloud must divide exactly into microbatches")
    microbatch_count = sample_count // microbatch_size
    train_steps = int(settings.train_steps)
    count = tf.constant(float(microbatch_count), DTYPE)
    inverse_count = tf.constant(1.0 / float(microbatch_count), DTYPE)
    t1_shift = tf.constant(t1_child.parent.shift_constant, DTYPE)
    t2_shift = tf.constant(artifact.shift_constant, DTYPE)

    @tf.function(jit_compile=True, reduce_retracing=True)
    def compiled_primal(theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        parameters = tf.reshape(tf.convert_to_tensor(theta, DTYPE), [PARAMETER_DIM])
        origin = tf.zeros_like(parameters)
        active_target = _packed_t2_target_log_value(
            parameters, inputs.training, inputs, tau
        )
        origin_target = _packed_t2_target_log_value(
            origin, inputs.training, inputs, tau
        )
        active_log_weight = (
            inputs.training.origin_log_importance_weight
            + active_target
            - origin_target
        )
        global_lse = tf.reduce_logsumexp(active_log_weight)
        initial = inputs.initial_packed_cores * inputs.packed_mask
        momentums = tf.zeros_like(initial)
        velocities = tf.zeros_like(initial)

        def train_condition(step, _cores, _momentums, _velocities):
            return step < train_steps

        def train_body(step, cores, active_m, active_v):
            gradient_sum = tf.zeros_like(cores)

            def micro_condition(microbatch, _gradient_sum):
                return microbatch < microbatch_count

            def micro_body(microbatch, accumulated):
                indices = inputs.microbatch_indices[microbatch]
                basis_values = tf.gather(inputs.training.basis_values, indices)
                log_weight = tf.gather(active_log_weight, indices)
                with tf.GradientTape() as tape:
                    tape.watch(cores)
                    amplitude = packed_amplitude(cores, basis_values)
                    rho = tf.square(amplitude) + tau
                    alpha = tf.exp(log_weight - global_lse)
                    cross_entropy = -count * tf.reduce_sum(
                        alpha * tf.math.log(rho)
                    )
                    active_l1, active_l2 = packed_per_core_regularizers(
                        cores, inputs.packed_mask
                    )
                    loss = (
                        cross_entropy
                        + tf.math.log(
                            packed_square_mass(cores, inputs.mass_matrices) + tau
                        )
                        + l1 * active_l1
                        + l2 * active_l2
                    )
                gradient = tape.gradient(loss, cores) * inputs.packed_mask
                return microbatch + 1, accumulated + gradient

            _, gradient_sum = tf.while_loop(
                micro_condition,
                micro_body,
                (tf.constant(0, tf.int32), gradient_sum),
                parallel_iterations=1,
                maximum_iterations=microbatch_count,
            )
            gradients = gradient_sum * inverse_count
            next_cores, next_m, next_v = packed_adam_apply_gradients(
                cores,
                active_m,
                active_v,
                gradients,
                inputs.packed_mask,
                step=step + 1,
                learning_rate=learning_rate,
                gradient_clip_norm=clip,
                beta_1=ADAM_BETA_1,
                beta_2=ADAM_BETA_2,
                epsilon=ADAM_EPSILON,
            )
            return step + 1, next_cores, next_m, next_v

        _, trained, _, _ = tf.while_loop(
            train_condition,
            train_body,
            (tf.constant(0, tf.int32), initial, momentums, velocities),
            parallel_iterations=1,
            maximum_iterations=train_steps,
        )
        active_calibration_target = _packed_t2_target_log_value(
            parameters, inputs.calibration, inputs, tau
        )
        origin_calibration_target = _packed_t2_target_log_value(
            origin, inputs.calibration, inputs, tau
        )
        active_calibration_weight = (
            inputs.calibration.origin_log_importance_weight
            + active_calibration_target
            - origin_calibration_target
        )
        shifted_log_mass = (
            tf.constant(artifact.calibration_estimate.log_shifted_normalizer, DTYPE)
            + tf.reduce_logsumexp(active_calibration_weight)
            - tf.reduce_logsumexp(inputs.calibration.origin_log_importance_weight)
        )
        target_mass = tf.exp(shifted_log_mass)
        scale = tf.sqrt(
            (target_mass - tau) / packed_square_mass(trained, inputs.mass_matrices)
        )
        axis_scale = tf.concat(
            [tf.reshape(scale, [1]), tf.ones([35], DTYPE)], axis=0
        )[:, tf.newaxis, tf.newaxis, tf.newaxis]
        calibrated = trained * axis_scale * inputs.packed_mask
        increment = (
            tf.math.log(packed_square_mass(calibrated, inputs.mass_matrices) + tau)
            - t2_shift
        )
        conditioned_t1 = inputs.t1_parent_packed_cores + tf.einsum(
            "p,pabcd->abcd", parameters, inputs.t1_packed_tangents
        )
        t1_value = (
            tf.math.log(packed_square_mass(conditioned_t1, inputs.mass_matrices) + tau)
            - t1_shift
        )
        return calibrated, increment, t1_value + increment

    return compiled_primal


def replay_t2_training_value(
    t1_child: LaneBParameterChild,
    artifact: LaneBT2Artifact,
    theta: tf.Tensor,
    *,
    inputs: T2ReplayInputs,
) -> tf.Tensor:
    _cores, _increment, cumulative = _make_t2_compiled_primal(
        t1_child, artifact, inputs
    )(theta)
    return cumulative


def replay_t2_training_jvp(
    t1_child: LaneBParameterChild,
    artifact: LaneBT2Artifact,
    *,
    inputs: T2ReplayInputs,
) -> T2ReplayJVPResult:
    """Materially replay T2 and issue centered-difference XLA core tangents."""

    primal = _make_t2_compiled_primal(t1_child, artifact, inputs)

    @tf.function(jit_compile=True, reduce_retracing=True)
    def compiled_bundle():
        origin = tf.zeros([PARAMETER_DIM], DTYPE)
        origin_cores, origin_increment, origin_cumulative = primal(origin)
        tangent_array = tf.TensorArray(
            DTYPE, size=PARAMETER_DIM, element_shape=origin_cores.shape
        )
        increment_array = tf.TensorArray(DTYPE, size=PARAMETER_DIM, element_shape=[])
        cumulative_array = tf.TensorArray(DTYPE, size=PARAMETER_DIM, element_shape=[])
        plus_array = tf.TensorArray(DTYPE, size=PARAMETER_DIM, element_shape=[])
        minus_array = tf.TensorArray(DTYPE, size=PARAMETER_DIM, element_shape=[])
        tangent_step = tf.constant(T2_TANGENT_FINITE_DIFFERENCE_STEP, DTYPE)
        fd_step = tf.constant(T2_FINITE_DIFFERENCE_STEP, DTYPE)

        def condition(parameter, *_arrays):
            return parameter < PARAMETER_DIM

        def body(parameter, tangents, increments, cumulative, plus, minus):
            direction = tf.one_hot(parameter, PARAMETER_DIM, dtype=DTYPE)
            plus_cores, plus_increment, plus_cumulative = primal(
                origin + tangent_step * direction
            )
            minus_cores, minus_increment, minus_cumulative = primal(
                origin - tangent_step * direction
            )
            _, _, fd_plus = primal(origin + fd_step * direction)
            _, _, fd_minus = primal(origin - fd_step * direction)
            denominator = tf.constant(2.0, DTYPE) * tangent_step
            return (
                parameter + 1,
                tangents.write(parameter, (plus_cores - minus_cores) / denominator),
                increments.write(parameter, (plus_increment - minus_increment) / denominator),
                cumulative.write(parameter, (plus_cumulative - minus_cumulative) / denominator),
                plus.write(parameter, fd_plus),
                minus.write(parameter, fd_minus),
            )

        _, tangents, increments, cumulative, plus, minus = tf.while_loop(
            condition,
            body,
            (
                tf.constant(0, tf.int32),
                tangent_array,
                increment_array,
                cumulative_array,
                plus_array,
                minus_array,
            ),
            parallel_iterations=1,
            maximum_iterations=PARAMETER_DIM,
        )
        corrected_tangents, raw_increment_score, radial_correction = (
            _project_tangents_to_scalar_derivative(
                tangents.stack(),
                increments.stack(),
                inputs.parent_packed_cores * inputs.packed_mask,
                inputs.mass_matrices,
                tf.constant(artifact.settings.tau, DTYPE),
            )
        )
        return (
            origin_cores,
            origin_increment,
            origin_cumulative,
            corrected_tangents,
            increments.stack(),
            cumulative.stack(),
            plus.stack(),
            minus.stack(),
            raw_increment_score,
            radial_correction,
        )

    (
        cores,
        increment,
        cumulative_value,
        packed_tangents,
        increment_score,
        cumulative_score,
        plus,
        minus,
        raw_increment_score,
        radial_correction,
    ) = compiled_bundle()
    _core_passed, residual, normalized = material_replay_metrics(
        cores, inputs.parent_packed_cores, inputs.packed_mask
    )
    functional_passed, metrics = _t2_functional_replay_metrics(
        cores, artifact, inputs
    )
    tf.debugging.assert_equal(
        functional_passed,
        True,
        "T2 material functional replay gate failed",
    )
    shapes = tuple(core.shape for core in artifact.cores)
    unpacked = unpack_cores(cores, shapes)
    banks = tuple(
        tuple(
            tf.ensure_shape(
                packed_tangents[parameter, axis, : shape[0], :, : shape[2]], shape
            )
            for parameter in range(PARAMETER_DIM)
        )
        for axis, shape in enumerate(shapes)
    )
    return T2ReplayJVPResult(
        cores=unpacked,
        tangent_cores=banks,
        increment=increment,
        increment_score=increment_score,
        cumulative_value=cumulative_value,
        cumulative_score=cumulative_score,
        maximum_core_residual=residual,
        maximum_normalized_core_residual=normalized,
        functional_replay_metrics=metrics,
        finite_difference_plus=plus,
        finite_difference_minus=minus,
        raw_core_tangent_increment_score=raw_increment_score,
        scalar_consistency_radial_correction=radial_correction,
    )


def _t2_functional_replay_metrics(
    cores: tf.Tensor,
    artifact: LaneBT2Artifact,
    inputs: T2ReplayInputs,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Compare T2 replay and parent on four gauge-invariant functional screens."""

    tau = tf.constant(artifact.settings.tau, DTYPE)
    observed_rows = (
        packed_normalized_density(cores, inputs.training.basis_values, inputs.mass_matrices, tau),
        packed_normalized_density(cores, inputs.calibration.basis_values, inputs.mass_matrices, tau),
        packed_normalized_prefix_density(cores, inputs.training.basis_values, inputs.mass_matrices, tau),
        packed_normalized_prefix_density(cores, inputs.calibration.basis_values, inputs.mass_matrices, tau),
    )
    reference_rows = (
        packed_normalized_density(inputs.parent_packed_cores, inputs.training.basis_values, inputs.mass_matrices, tau),
        packed_normalized_density(inputs.parent_packed_cores, inputs.calibration.basis_values, inputs.mass_matrices, tau),
        packed_normalized_prefix_density(inputs.parent_packed_cores, inputs.training.basis_values, inputs.mass_matrices, tau),
        packed_normalized_prefix_density(inputs.parent_packed_cores, inputs.calibration.basis_values, inputs.mass_matrices, tau),
    )
    metric_rows = tuple(
        material_positive_value_metrics(observed, reference)
        for observed, reference in zip(observed_rows, reference_rows)
    )
    passed = tf.reduce_all(tf.stack([row[0] for row in metric_rows]))
    metrics = tf.stack(
        [tf.stack([row[1], row[2], row[3]]) for row in metric_rows], axis=0
    )
    return passed, metrics


def _prepared_input_descriptor(path: Path) -> dict[str, object]:
    absolute = ROOT / path
    payload = json.loads(absolute.read_text())
    gates = payload.get("gates")
    if (
        payload.get("schema_version")
        != "bayesfilter.zhao_cui_austria_sir_lane_b_t2_prepared_cloud.v1"
        or payload.get("status") != "PREPARED_T2_CLOUD"
        or not isinstance(gates, dict)
        or not all(
            gates.get(name) is True
            for name in ("finite", "sample_count", "memory", "effective_sample_size")
        )
        or not isinstance(payload.get("cloud_manifest"), dict)
    ):
        raise ValueError(f"T2 prepared input is not admissible: {path}")
    return {
        "path": path.as_posix(),
        "result_sha256": hashlib.sha256(absolute.read_bytes()).hexdigest(),
        "cloud_manifest": payload["cloud_manifest"],
    }


def _current_t2_issuer_source_sha256() -> dict[str, str]:
    return {
        relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        for relative in REQUIRED_T2_ISSUER_SOURCE_PATHS
    }


def issue_t2_training_jvp_identity_payload(
    *,
    t1_issuer_identity: str,
    t1_child_identity: str,
    parent_t1_identity: str,
    parent_t2: LaneBT2Artifact,
    t2_child_identity: str,
    tangent_tensor_sha256: Mapping[str, str],
    evidence: Mapping[str, object],
) -> dict[str, object]:
    """Issue the repository-bound identity for one passed T2 replay."""

    training = _prepared_input_descriptor(T2_TRAINING_PREPARED_RESULT_PATH)
    calibration = _prepared_input_descriptor(T2_CALIBRATION_PREPARED_RESULT_PATH)
    sample_count = int(training["cloud_manifest"]["sample_count"])
    microbatch_size = int(parent_t2.settings.batch_size)
    if sample_count % microbatch_size:
        raise ValueError("T2 issuer input does not divide into full microbatches")
    return {
        "schema_version": T2_ISSUER_SCHEMA,
        "issuer_id": T2_ISSUER_ID,
        "replay_id": T2_REPLAY_ID,
        "score_target_id": T2_SCORE_TARGET_ID,
        "classification": "extension_or_invention",
        "parent_t1_identity": str(parent_t1_identity),
        "parent_t2_identity": parent_t2.identity.hash.value,
        "t1_issuer_identity": str(t1_issuer_identity),
        "t1_child_identity": str(t1_child_identity),
        "t2_child_identity": str(t2_child_identity),
        "training_prepared_input": training,
        "calibration_prepared_input": calibration,
        "shift_derivative_policy": SHIFT_DERIVATIVE_POLICY,
        "tau_derivative_policy": TAU_DERIVATIVE_POLICY,
        "optimizer": {
            "family": "keras3_adam_functional_exact_update_order",
            "gradient_accumulation": "full_cloud_equal_microbatch_mean_v1",
            "learning_rate": parent_t2.settings.learning_rate,
            "beta_1": ADAM_BETA_1,
            "beta_2": ADAM_BETA_2,
            "epsilon": ADAM_EPSILON,
            "gradient_clip_norm": parent_t2.settings.gradient_clip_norm,
            "train_steps": parent_t2.settings.train_steps,
            "sample_count": sample_count,
            "microbatch_size": microbatch_size,
            "microbatch_count": sample_count // microbatch_size,
            "jit_compile": True,
            "full_program_control_flow": "nested_tensorflow_while_loop",
            "python_numerical_loops": False,
        },
        "replay_gate": {
            "material_functional_atol": MATERIAL_REPLAY_ATOL,
            "material_functional_rtol": MATERIAL_REPLAY_RTOL,
            "maximum_normalized_functional_residual": 1.0,
            "functional_screen_order": list(FUNCTIONAL_SCREEN_ORDER),
            "functional_screen_columns": list(FUNCTIONAL_SCREEN_COLUMNS),
            "tangent_finite_difference_step": T2_TANGENT_FINITE_DIFFERENCE_STEP,
            "finite_difference_step": T2_FINITE_DIFFERENCE_STEP,
            "finite_difference_atol": T2_FINITE_DIFFERENCE_ATOL,
            "finite_difference_rtol": T2_FINITE_DIFFERENCE_RTOL,
            "memory_cap_bytes": T2_MEMORY_CAP_BYTES,
            "gpu_memory_limit_mib": T2_GPU_MEMORY_LIMIT_MIB,
        },
        "evidence": dict(evidence),
        "tangent_tensor_sha256": dict(tangent_tensor_sha256),
        "source_sha256": _current_t2_issuer_source_sha256(),
        "material_replay_policy_id": MATERIAL_REPLAY_POLICY_ID,
        "packed_xla_policy_id": PACKED_XLA_POLICY_ID,
        "runtime_score_backend": RUNTIME_SCORE_BACKEND,
        "offline_issuer_derivative": T2_OFFLINE_ISSUER_DERIVATIVE,
        "runtime_autodiff": False,
        "runtime_finite_difference": False,
        "hmc_authorized": False,
    }


def load_t2_training_jvp_child(
    directory: Path,
    *,
    t1_issuer_directory: Path,
    parent_t1: object,
    parent_t2: LaneBT2Artifact,
) -> tuple[LaneBParameterChild, LaneBParameterChild, dict[str, object]]:
    """Load a T2 child only through the strict T1 and T2 issuer chain."""

    if parent_t2.parent_artifact.identity.hash.value != parent_t1.identity.hash.value:
        raise ValueError("T2 training-JVP parent chain mismatch")
    t1_child, t1_issuer = load_t1_training_jvp_child(
        Path(t1_issuer_directory), parent=parent_t1
    )
    output = Path(directory)
    payload = json.loads((output / "result.json").read_text())
    if (
        payload.get("schema_version") != T2_ISSUER_SCHEMA
        or payload.get("status")
        != "PASS_T1_T2_MATERIAL_TRAINING_REPLAY_AND_FD_TANGENT"
        or payload.get("parent_t1_identity") != parent_t1.identity.hash.value
        or payload.get("parent_t2_identity") != parent_t2.identity.hash.value
        or payload.get("t1_issuer_identity") != t1_issuer.get("issuer_identity")
    ):
        raise ValueError("T2 training-JVP artifact schema/status/parent mismatch")
    identity = payload.get("issuer_identity_payload")
    if not isinstance(identity, dict):
        raise ValueError("T2 training-JVP identity payload missing")
    fixed_fields = {
        "issuer_id": T2_ISSUER_ID,
        "replay_id": T2_REPLAY_ID,
        "score_target_id": T2_SCORE_TARGET_ID,
        "classification": "extension_or_invention",
        "parent_t1_identity": parent_t1.identity.hash.value,
        "parent_t2_identity": parent_t2.identity.hash.value,
        "t1_issuer_identity": t1_issuer["issuer_identity"],
        "t1_child_identity": t1_child.identity.hash.value,
        "shift_derivative_policy": SHIFT_DERIVATIVE_POLICY,
        "tau_derivative_policy": TAU_DERIVATIVE_POLICY,
        "runtime_score_backend": RUNTIME_SCORE_BACKEND,
        "offline_issuer_derivative": T2_OFFLINE_ISSUER_DERIVATIVE,
        "material_replay_policy_id": MATERIAL_REPLAY_POLICY_ID,
        "packed_xla_policy_id": PACKED_XLA_POLICY_ID,
        "runtime_autodiff": False,
        "runtime_finite_difference": False,
        "hmc_authorized": False,
    }
    if (
        any(identity.get(key) != value for key, value in fixed_fields.items())
        or _semantic_sha256(identity) != payload.get("issuer_identity")
    ):
        raise ValueError("T2 training-JVP issuer identity mismatch")
    if identity.get("training_prepared_input") != _prepared_input_descriptor(
        T2_TRAINING_PREPARED_RESULT_PATH
    ) or identity.get("calibration_prepared_input") != _prepared_input_descriptor(
        T2_CALIBRATION_PREPARED_RESULT_PATH
    ):
        raise ValueError("T2 training-JVP prepared-input identity mismatch")
    training = _prepared_input_descriptor(T2_TRAINING_PREPARED_RESULT_PATH)
    sample_count = int(training["cloud_manifest"]["sample_count"])
    microbatch_size = int(parent_t2.settings.batch_size)
    expected_optimizer = {
        "family": "keras3_adam_functional_exact_update_order",
        "gradient_accumulation": "full_cloud_equal_microbatch_mean_v1",
        "learning_rate": parent_t2.settings.learning_rate,
        "beta_1": ADAM_BETA_1,
        "beta_2": ADAM_BETA_2,
        "epsilon": ADAM_EPSILON,
        "gradient_clip_norm": parent_t2.settings.gradient_clip_norm,
        "train_steps": parent_t2.settings.train_steps,
        "sample_count": sample_count,
        "microbatch_size": microbatch_size,
        "microbatch_count": sample_count // microbatch_size,
        "jit_compile": True,
        "full_program_control_flow": "nested_tensorflow_while_loop",
        "python_numerical_loops": False,
    }
    expected_gate = {
        "material_functional_atol": MATERIAL_REPLAY_ATOL,
        "material_functional_rtol": MATERIAL_REPLAY_RTOL,
        "maximum_normalized_functional_residual": 1.0,
        "functional_screen_order": list(FUNCTIONAL_SCREEN_ORDER),
        "functional_screen_columns": list(FUNCTIONAL_SCREEN_COLUMNS),
        "tangent_finite_difference_step": T2_TANGENT_FINITE_DIFFERENCE_STEP,
        "finite_difference_step": T2_FINITE_DIFFERENCE_STEP,
        "finite_difference_atol": T2_FINITE_DIFFERENCE_ATOL,
        "finite_difference_rtol": T2_FINITE_DIFFERENCE_RTOL,
        "memory_cap_bytes": T2_MEMORY_CAP_BYTES,
        "gpu_memory_limit_mib": T2_GPU_MEMORY_LIMIT_MIB,
    }
    if identity.get("optimizer") != expected_optimizer or identity.get(
        "replay_gate"
    ) != expected_gate:
        raise ValueError("T2 training-JVP optimizer/replay gate mismatch")
    if identity.get("source_sha256") != _current_t2_issuer_source_sha256():
        raise ValueError("T2 training-JVP source closure stale")
    evidence = identity.get("evidence")
    if not isinstance(evidence, dict):
        raise ValueError("T2 training-JVP evidence missing")
    functional_metrics = evidence.get("functional_replay_metrics")
    if (
        not isinstance(functional_metrics, list)
        or len(functional_metrics) != len(FUNCTIONAL_SCREEN_ORDER)
        or any(not isinstance(row, list) or len(row) != len(FUNCTIONAL_SCREEN_COLUMNS) for row in functional_metrics)
        or not all(math.isfinite(float(value)) for row in functional_metrics for value in row)
        or not all(float(row[1]) <= 1.0 for row in functional_metrics)
        or functional_metrics != payload.get("functional_replay_metrics")
    ):
        raise ValueError("T2 training tangent material functional evidence failed")
    scalar_normalized = float(evidence.get("scalar_normalized_residual", math.inf))
    if not math.isfinite(scalar_normalized) or scalar_normalized > 1.0:
        raise ValueError("T2 training tangent material scalar evidence failed")
    if int(evidence.get("gpu_allocator_peak_bytes", T2_MEMORY_CAP_BYTES + 1)) > T2_MEMORY_CAP_BYTES:
        raise ValueError("T2 training-JVP memory evidence failed")
    fd_rows = evidence.get("finite_difference_rows")
    if not isinstance(fd_rows, list) or len(fd_rows) != PARAMETER_DIM:
        raise ValueError("T2 training-JVP finite-difference evidence missing")
    for parameter, row in enumerate(fd_rows):
        if not isinstance(row, dict) or int(row.get("parameter", -1)) != parameter:
            raise ValueError("T2 training-JVP finite-difference row mismatch")
        observed = float(row.get("finite_difference", math.nan))
        issued = float(row.get("issued_tangent_score", math.nan))
        residual = float(row.get("absolute_residual", math.nan))
        if (
            float(row.get("step", math.nan)) != T2_FINITE_DIFFERENCE_STEP
            or not all(math.isfinite(value) for value in (observed, issued, residual))
            or residual != abs(observed - issued)
            or residual
            > T2_FINITE_DIFFERENCE_ATOL + T2_FINITE_DIFFERENCE_RTOL * abs(issued)
        ):
            raise ValueError("T2 training-JVP finite-difference evidence failed")
    gates = payload.get("hard_gates")
    required_gates = (
        "strict_t1_issuer_load",
        "strict_t2_prepared_cloud_load",
        "material_functional_replay",
        "material_scalar_replay",
        "manual_issued_tangent_parity",
        "independent_step_halving_fd_parity",
        "memory_under_6_gib",
    )
    if not isinstance(gates, dict) or not all(
        gates.get(name) is True for name in required_gates
    ):
        raise ValueError("T2 training-JVP hard gate missing or failed")
    tensors = payload.get("tensors")
    hashes = identity.get("tangent_tensor_sha256")
    names = {
        f"tangent_{axis:02d}_{parameter}"
        for axis in range(len(parent_t2.cores))
        for parameter in range(PARAMETER_DIM)
    }
    if not isinstance(tensors, dict) or not isinstance(hashes, dict) or set(tensors) != names or set(hashes) != names:
        raise ValueError("T2 training-JVP tensor ledger mismatch")

    def read(name: str, shape: tf.TensorShape) -> tf.Tensor:
        row = tensors[name]
        if (
            not isinstance(row, dict)
            or row.get("sha256") != hashes[name]
            or row.get("dtype") != "float64"
            or row.get("shape") != shape.as_list()
        ):
            raise ValueError(f"T2 training-JVP tensor identity mismatch: {name}")
        candidate = (output / str(row.get("path"))).resolve()
        try:
            candidate.relative_to(output.resolve())
        except ValueError as exc:
            raise ValueError("T2 training-JVP tensor path escapes artifact") from exc
        serialized = tf.io.read_file(candidate.as_posix())
        if hashlib.sha256(bytes(serialized.numpy())).hexdigest() != hashes[name]:
            raise ValueError(f"T2 training-JVP tensor hash mismatch: {name}")
        return tf.ensure_shape(
            tf.io.parse_tensor(serialized, out_type=tf.float64), shape
        )

    banks = tuple(
        tuple(
            read(f"tangent_{axis:02d}_{parameter}", core.shape)
            for parameter in range(PARAMETER_DIM)
        )
        for axis, core in enumerate(parent_t2.cores)
    )
    t2_child = LaneBParameterChild(parent_t2, banks)
    if t2_child.identity.hash.value != identity.get("t2_child_identity") or t2_child.identity.hash.value != payload.get("t2_child_identity"):
        raise ValueError("T2 training-JVP child identity mismatch")
    origin = tf.zeros([PARAMETER_DIM], DTYPE)
    t1_value, t1_score = t1_child.increment_and_score(origin)
    increment, increment_score = t2_child.increment_and_score(origin)
    tf.debugging.assert_near(
        increment,
        tf.constant(float(evidence["manual_increment"]), DTYPE),
        atol=2e-13,
        rtol=0.0,
    )
    tf.debugging.assert_near(
        increment_score,
        tf.constant(evidence["manual_increment_score"], DTYPE),
        atol=3e-10,
        rtol=3e-10,
    )
    tf.debugging.assert_near(
        t1_value + increment,
        tf.constant(float(evidence["manual_cumulative_value"]), DTYPE),
        atol=5e-13,
        rtol=0.0,
    )
    tf.debugging.assert_near(
        t1_score + increment_score,
        tf.constant(evidence["manual_cumulative_score"], DTYPE),
        atol=4e-10,
        rtol=4e-10,
    )
    return t1_child, t2_child, payload


__all__ = [
    "REQUIRED_T2_ISSUER_SOURCE_PATHS",
    "T2_CALIBRATION_PREPARED_RESULT_PATH",
    "T2_FINITE_DIFFERENCE_ATOL",
    "T2_FINITE_DIFFERENCE_RTOL",
    "T2_FINITE_DIFFERENCE_STEP",
    "T2_GPU_MEMORY_LIMIT_MIB",
    "T2_ISSUER_ID",
    "T2_ISSUER_SCHEMA",
    "T2_MEMORY_CAP_BYTES",
    "T2_OFFLINE_ISSUER_DERIVATIVE",
    "T2_TANGENT_FINITE_DIFFERENCE_STEP",
    "T2_REPLAY_ID",
    "T2_TRAINING_PREPARED_RESULT_PATH",
    "T2ReplayCloudInputs",
    "T2ReplayInputs",
    "T2ReplayJVPResult",
    "issue_t2_training_jvp_identity_payload",
    "load_t2_training_jvp_child",
    "make_t2_replay_inputs",
    "prepare_t2_replay_cloud_inputs",
    "replay_t2_training_jvp",
    "replay_t2_training_value",
]
