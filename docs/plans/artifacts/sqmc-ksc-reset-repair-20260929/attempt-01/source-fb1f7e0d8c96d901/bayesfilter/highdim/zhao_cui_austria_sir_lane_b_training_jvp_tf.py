"""Packed-XLA tangents of the frozen Lane-B T1 training/calibration program.

This module issues centered-difference core tangents from the same frozen XLA
primal that materially replays the admitted parent. Runtime value/score code
consumes only the issued tangents and uses manual paired-core contractions.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Sequence

import tensorflow as tf

from bayesfilter.highdim.sir_latent_preclip_tf import (
    latent_preclip_zhao_cui_sir_austria_model,
)
from bayesfilter.highdim.models import (
    _zhao_cui_sir_austria_adjacency_xla,
    _xla_isotropic_mvn_log_prob,
    _zhao_cui_sir_austria_initial_mean_xla,
    _zhao_cui_sir_austria_rhs_xla,
)
from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_target_tf import (
    LaneBT1ProposalCloud,
    generate_sealed_lane_b_dataset,
    generate_t1_proposal_cloud,
)
from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_tf import (
    LaneBT1Artifact,
    balanced_initial_cores,
    build_training_batch,
    lane_b_product_basis,
)
from bayesfilter.highdim.zhao_cui_austria_sir_parameter_child_tf import (
    LaneBParameterChild,
)
from bayesfilter.highdim.zhao_cui_austria_sir_packed_xla_tf import (
    MATERIAL_REPLAY_ATOL,
    MATERIAL_REPLAY_POLICY_ID,
    MATERIAL_REPLAY_RTOL,
    PACKED_XLA_POLICY_ID,
    material_replay_metrics,
    material_positive_value_metrics,
    pack_cores,
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


DTYPE = tf.float64
PARAMETER_DIM = 3
REPLAY_ID = "lane_b_t1_packed_xla_training_adam_centered_fd_tangent_v3"
ADAM_BETA_1 = 0.9
ADAM_BETA_2 = 0.999
ADAM_EPSILON = 1e-7
ISSUER_SCHEMA = "bayesfilter.zhao_cui_austria_sir_lane_b_training_tangent.v4"
ISSUER_ID = "repository_owned_material_replay_packed_xla_fd_tangent_issuer_v3"
SHIFT_DERIVATIVE_POLICY = "frozen_parent_shift_zero_derivative_v1"
TAU_DERIVATIVE_POLICY = "frozen_parent_tau_zero_derivative_v1"
RUNTIME_SCORE_BACKEND = "manual_paired_core_contractions_no_autodiff_v1"
OFFLINE_ISSUER_DERIVATIVE = "tensorflow_xla_centered_fd_core_tangent_h5e5_v1"
TANGENT_FINITE_DIFFERENCE_STEP = 5e-5
FINITE_DIFFERENCE_STEP = 1e-4
FINITE_DIFFERENCE_ATOL = 2e-4
FINITE_DIFFERENCE_RTOL = 2e-4
MEMORY_CAP_BYTES = 6 * 1024**3
GPU_MEMORY_LIMIT_MIB = 6 * 1024
FUNCTIONAL_SCREEN_ORDER = (
    "training_full_density",
    "calibration_full_density",
    "training_prefix_marginal",
    "calibration_prefix_marginal",
)
FUNCTIONAL_SCREEN_COLUMNS = (
    "maximum_absolute_residual",
    "maximum_normalized_residual",
    "maximum_log_residual",
)
ROOT = Path(__file__).resolve().parents[2]
REQUIRED_ISSUER_SOURCE_PATHS = (
    "bayesfilter/highdim/models.py",
    "bayesfilter/highdim/zhao_cui_austria_sir_packed_xla_tf.py",
    "bayesfilter/highdim/zhao_cui_austria_sir_lane_b_training_jvp_tf.py",
    "bayesfilter/highdim/zhao_cui_austria_sir_parameter_child_tf.py",
    "docs/plans/bayesfilter-zhao-cui-austria-sir-material-replay-xla-repair-plan-2026-08-02.md",
    "scripts/run_zhao_cui_austria_sir_lane_b_training_jvp.py",
)


def _claim_local_transition_mean_xla(
    theta: tf.Tensor, states: tf.Tensor
) -> tf.Tensor:
    """Evaluate the frozen four-substep Austria transition as a static XLA graph."""

    parameters = tf.reshape(tf.convert_to_tensor(theta, DTYPE), [PARAMETER_DIM])
    state = tf.convert_to_tensor(states, DTYPE)
    kappa = tf.fill([9], tf.constant(0.1, DTYPE) * tf.exp(parameters[0]))
    nu = tf.fill([9], tf.constant(18.0, DTYPE) * tf.exp(parameters[1]))
    adjacency = _zhao_cui_sir_austria_adjacency_xla()
    degree = tf.reduce_sum(adjacency, axis=1)
    step = tf.constant(0.005, DTYPE)

    def rk4_step(active_state):
        k1 = _zhao_cui_sir_austria_rhs_xla(
            active_state, kappa, nu, adjacency, degree
        )
        k2 = _zhao_cui_sir_austria_rhs_xla(
            active_state + tf.constant(0.5, DTYPE) * step * k1,
            kappa,
            nu,
            adjacency,
            degree,
        )
        k3 = _zhao_cui_sir_austria_rhs_xla(
            active_state + tf.constant(0.5, DTYPE) * step * k2,
            kappa,
            nu,
            adjacency,
            degree,
        )
        k4 = _zhao_cui_sir_austria_rhs_xla(
            active_state + tf.constant(0.5, DTYPE) * step * k3,
            kappa,
            nu,
            adjacency,
            degree,
        )
        next_state = active_state + (step / tf.constant(6.0, DTYPE)) * (
            k1 + tf.constant(2.0, DTYPE) * k2 + tf.constant(2.0, DTYPE) * k3 + k4
        )
        return next_state

    state = rk4_step(state)
    state = rk4_step(state)
    state = rk4_step(state)
    return rk4_step(state)


@dataclass(frozen=True)
class T1ReplayInputs:
    """Frozen arrays needed to replay the admitted T1 optimizer."""

    training_physical_points: tf.Tensor
    training_local_points: tf.Tensor
    training_origin_target_sqrt: tf.Tensor
    training_integration_weights: tf.Tensor
    calibration_physical_points: tf.Tensor
    calibration_origin_log_likelihood: tf.Tensor
    training_basis_values: tf.Tensor
    calibration_basis_values: tf.Tensor
    mass_matrices: tf.Tensor
    initial_packed_cores: tf.Tensor
    parent_packed_cores: tf.Tensor
    packed_mask: tf.Tensor
    observation: tf.Tensor
    training_batch_indices: tf.Tensor


@dataclass(frozen=True)
class T1ReplayJVPResult:
    """Origin replay and three core-tangent banks."""

    cores: tuple[tf.Tensor, ...]
    tangent_cores: tuple[tuple[tf.Tensor, ...], ...]
    value: tf.Tensor
    score: tf.Tensor
    maximum_core_residual: tf.Tensor
    maximum_normalized_core_residual: tf.Tensor
    functional_replay_metrics: tf.Tensor
    finite_difference_plus: tf.Tensor
    finite_difference_minus: tf.Tensor
    replay_id: str = REPLAY_ID


def _physical_log_target_and_score(
    theta: tf.Tensor,
    points: tf.Tensor,
    *,
    model: object | None = None,
    observation: tf.Tensor | None = None,
) -> tuple[tf.Tensor, tf.Tensor]:
    parameters = tf.reshape(tf.convert_to_tensor(theta, DTYPE), [PARAMETER_DIM])
    physical = tf.convert_to_tensor(points, DTYPE)
    z1 = physical[:, :18]
    z0 = physical[:, 18:]
    active_model = model or latent_preclip_zhao_cui_sir_austria_model()
    if observation is None:
        _states, observations, _all = generate_sealed_lane_b_dataset()
        active_observation = observations[0]
    else:
        active_observation = tf.ensure_shape(
            tf.convert_to_tensor(observation, DTYPE), [9]
        )
    log_target = (
        active_model.initial_log_density(parameters, z0)
        + active_model.transition_log_density(parameters, z0, z1, 1)
        + active_model.observation_log_density(
            parameters, z1, active_observation, 1
        )
    )
    score = (
        active_model.initial_log_density_parameter_score(parameters, z0)
        + active_model.transition_log_density_parameter_score(parameters, z0, z1, 1)
        + active_model.observation_log_density_parameter_score(
            parameters, z1, active_observation, 1
        )
    )
    return log_target, score


def _physical_log_target_value_xla(
    theta: tf.Tensor,
    points: tf.Tensor,
    observation: tf.Tensor,
) -> tf.Tensor:
    """Graph-native FP64 value of the same T1 complete-data row target."""

    parameters = tf.reshape(tf.convert_to_tensor(theta, DTYPE), [PARAMETER_DIM])
    physical = tf.convert_to_tensor(points, DTYPE)
    z1 = physical[:, :18]
    z0 = physical[:, 18:]
    active_observation = tf.ensure_shape(
        tf.convert_to_tensor(observation, DTYPE), [9]
    )
    initial = _xla_isotropic_mvn_log_prob(
        z0 - _zhao_cui_sir_austria_initial_mean_xla()[tf.newaxis, :],
        tf.constant(1.0, DTYPE),
    )
    transition_mean = _claim_local_transition_mean_xla(parameters, z0)
    transition = _xla_isotropic_mvn_log_prob(
        z1 - transition_mean,
        tf.constant(1.0, DTYPE),
    )
    observation_variance = tf.constant(100.0, DTYPE) * tf.exp(
        tf.constant(2.0, DTYPE) * parameters[2]
    )
    likelihood = _xla_isotropic_mvn_log_prob(
        active_observation[tf.newaxis, :] - z1[:, 1::2],
        observation_variance,
    )
    return initial + transition + likelihood


def _cloud_from_manifest(
    row: dict[str, object] | object, *, expected_role: str
) -> LaneBT1ProposalCloud:
    if not isinstance(row, dict) and not hasattr(row, "get"):
        raise ValueError("T1 replay cloud manifest is invalid")
    role = str(row.get("role"))
    if role != expected_role:
        raise ValueError(f"T1 replay requires role {expected_role}")
    cloud = generate_t1_proposal_cloud(
        sample_count=int(row.get("sample_count")),
        seed=int(row.get("seed")),
        role=role,
    )
    observed = cloud.manifest_payload()
    for key, value in observed.items():
        expected = row.get(key)
        if key == "joint_axis_order" and expected is not None:
            expected = tuple(expected)
        if expected != value:
            raise ValueError(f"T1 replay cloud mismatch: {key}")
    return cloud


def prepare_t1_replay_inputs(parent: LaneBT1Artifact) -> T1ReplayInputs:
    """Regenerate frozen clouds and precompute setup-static packed tensors."""

    training = _cloud_from_manifest(
        parent.training_cloud_manifest, expected_role="training_frame"
    )
    calibration_row = parent.calibration_estimate.manifest_payload()
    calibration = generate_t1_proposal_cloud(
        sample_count=int(calibration_row["sample_count"]),
        seed=int(calibration_row["seed"]),
        role=str(calibration_row["role"]),
    )
    if calibration.manifest_payload()["log_likelihood_sha256"] != calibration_row[
        "log_likelihood_sha256"
    ]:
        raise ValueError("T1 replay calibration cloud mismatch")
    batch = build_training_batch(training, parent.frame, parent.shift_constant)
    _states, observations, _all = generate_sealed_lane_b_dataset()
    basis = lane_b_product_basis(
        order=parent.settings.basis_order,
        num_elems=parent.settings.basis_num_elems,
    )
    initial = balanced_initial_cores(parent.settings, basis)
    shapes = tuple(core.shape for core in parent.cores)
    population = int(batch.points.shape[0])
    batch_size = int(parent.settings.batch_size)
    train_steps = int(parent.settings.train_steps)
    zero_steps = tf.range(train_steps, dtype=tf.int32)[:, tf.newaxis]
    offsets = tf.range(batch_size, dtype=tf.int32)[tf.newaxis, :]
    training_batch_indices = tf.math.floormod(
        zero_steps * tf.constant(batch_size, tf.int32) + offsets,
        tf.constant(population, tf.int32),
    )
    return T1ReplayInputs(
        training_physical_points=training.joint_points,
        training_local_points=batch.points,
        training_origin_target_sqrt=batch.target_sqrt_values,
        training_integration_weights=batch.integration_weights,
        calibration_physical_points=calibration.joint_points,
        calibration_origin_log_likelihood=calibration.log_likelihood,
        training_basis_values=precompute_basis_values(basis, batch.points),
        calibration_basis_values=precompute_basis_values(
            basis,
            build_training_batch(
                calibration, parent.frame, parent.shift_constant
            ).points,
        ),
        mass_matrices=precompute_mass_matrices(basis),
        initial_packed_cores=pack_cores(initial),
        parent_packed_cores=pack_cores(parent.cores),
        packed_mask=packed_core_mask(shapes),
        observation=tf.ensure_shape(observations[0], [9]),
        training_batch_indices=tf.ensure_shape(
            training_batch_indices, [train_steps, batch_size]
        ),
    )


def _origin_proposal_log_density(points: tf.Tensor) -> tf.Tensor:
    physical = tf.convert_to_tensor(points, DTYPE)
    z1 = physical[:, :18]
    z0 = physical[:, 18:]
    model = latent_preclip_zhao_cui_sir_austria_model()
    origin = tf.zeros([PARAMETER_DIM], DTYPE)
    return model.initial_log_density(origin, z0) + model.transition_log_density(
        origin, z0, z1, 1
    )


def _evaluate_cores(
    cores: Sequence[tf.Tensor], basis: object, points: tf.Tensor
) -> tf.Tensor:
    values = tf.convert_to_tensor(points, DTYPE)
    vector = tf.ones([tf.shape(values)[0], 1], DTYPE)
    for axis, core in enumerate(cores):
        basis_values = basis.evaluate_axis(axis, values[:, axis])
        vector = tf.einsum(
            "na,nab->nb", vector, tf.einsum("nl,alb->nab", basis_values, core)
        )
    return tf.reshape(vector, [tf.shape(values)[0]])


def _square_mass(cores: Sequence[tf.Tensor], basis: object) -> tf.Tensor:
    vector = tf.ones([1], DTYPE)
    measure = basis.convention.mass_measure
    for axis, core in enumerate(cores):
        mass = basis.bases[axis].mass_matrix(measure)
        paired = tf.einsum("alb,AmB,lm->aAbB", core, core, mass)
        vector = tf.einsum(
            "a,ab->b",
            vector,
            tf.reshape(
                paired,
                [int(core.shape[0]) ** 2, int(core.shape[2]) ** 2],
            ),
        )
    return tf.reshape(vector, [])


def _training_loss(
    theta: tf.Tensor,
    cores: Sequence[tf.Tensor],
    physical_points: tf.Tensor,
    local_points: tf.Tensor,
    origin_target_sqrt: tf.Tensor,
    integration_weights: tf.Tensor,
    origin_physical_log_target: tf.Tensor,
    *,
    tau: tf.Tensor,
    l1_weight: tf.Tensor,
    l2_weight: tf.Tensor,
    basis: object,
    observation: tf.Tensor,
) -> tf.Tensor:
    physical_log_target = _physical_log_target_value_xla(
        theta,
        physical_points,
        observation,
    )
    log_ratio = physical_log_target - origin_physical_log_target
    target_sqrt = origin_target_sqrt * tf.exp(0.5 * log_ratio)
    target_square = tf.square(target_sqrt)
    raw_alpha = integration_weights * (target_square + tau)
    alpha = raw_alpha / tf.reduce_sum(raw_alpha)
    amplitude = _evaluate_cores(cores, basis, local_points)
    rho = tf.square(amplitude) + tau
    cross_entropy = -tf.reduce_sum(alpha * tf.math.log(rho))
    log_normalizer = tf.math.log(_square_mass(cores, basis) + tau)
    l1 = tf.add_n([tf.reduce_sum(tf.abs(core)) for core in cores])
    l2 = tf.add_n([tf.reduce_sum(tf.square(core)) for core in cores])
    return cross_entropy + log_normalizer + l1_weight * l1 + l2_weight * l2


def functional_adam_step(
    theta: tf.Tensor,
    cores: Sequence[tf.Tensor],
    momentums: Sequence[tf.Tensor],
    velocities: Sequence[tf.Tensor],
    *,
    step: int,
    learning_rate: tf.Tensor,
    gradient_clip_norm: tf.Tensor,
    loss_fn,
) -> tuple[tuple[tf.Tensor, ...], tuple[tf.Tensor, ...], tuple[tf.Tensor, ...]]:
    """Apply the exact Keras-3 Adam update as a tensor-valued function."""

    tensors = tuple(tf.convert_to_tensor(core, DTYPE) for core in cores)
    with tf.GradientTape() as tape:
        tape.watch(tensors)
        loss = loss_fn(theta, tensors)
    gradients = tape.gradient(loss, tensors)
    if any(gradient is None for gradient in gradients):
        raise ValueError("T1 replay produced a missing core gradient")
    return functional_adam_apply_gradients(
        tensors,
        momentums,
        velocities,
        gradients,
        step=step,
        learning_rate=learning_rate,
        gradient_clip_norm=gradient_clip_norm,
    )


def functional_adam_apply_gradients(
    cores: Sequence[tf.Tensor],
    momentums: Sequence[tf.Tensor],
    velocities: Sequence[tf.Tensor],
    gradients: Sequence[tf.Tensor],
    *,
    step: int | tf.Tensor,
    learning_rate: tf.Tensor,
    gradient_clip_norm: tf.Tensor,
) -> tuple[tuple[tf.Tensor, ...], tuple[tf.Tensor, ...], tuple[tf.Tensor, ...]]:
    """Apply the exact clipped Keras-3 Adam state transition."""

    tensors = tuple(tf.convert_to_tensor(core, DTYPE) for core in cores)
    checked = tuple(tf.convert_to_tensor(gradient, DTYPE) for gradient in gradients)
    if not (
        len(tensors) == len(momentums) == len(velocities) == len(checked)
    ):
        raise ValueError("Adam cores, slots, and gradients must have equal lengths")
    checked, _ = tf.clip_by_global_norm(checked, gradient_clip_norm)
    # Keras backend constants are first materialized in float32 and then cast
    # to the variable dtype inside Adam.update_step.
    beta1 = tf.cast(tf.constant(ADAM_BETA_1, tf.float32), DTYPE)
    beta2 = tf.cast(tf.constant(ADAM_BETA_2, tf.float32), DTYPE)
    one_minus_beta1 = tf.constant(1.0 - ADAM_BETA_1, DTYPE)
    one_minus_beta2 = tf.constant(1.0 - ADAM_BETA_2, DTYPE)
    local_step = tf.cast(step, DTYPE)
    alpha = learning_rate * tf.sqrt(1.0 - tf.pow(beta2, local_step)) / (
        1.0 - tf.pow(beta1, local_step)
    )
    next_cores = []
    next_momentums = []
    next_velocities = []
    for core, gradient, momentum, velocity in zip(
        tensors, checked, momentums, velocities
    ):
        # Preserve Keras 3 operation order; equivalent rearrangements differ
        # by ulps and cannot reproduce a frozen artifact tensor-for-tensor.
        next_momentum = momentum + (gradient - momentum) * one_minus_beta1
        next_velocity = velocity + (tf.square(gradient) - velocity) * one_minus_beta2
        next_core = core - alpha * next_momentum / (
            tf.sqrt(next_velocity) + tf.constant(ADAM_EPSILON, DTYPE)
        )
        next_cores.append(next_core)
        next_momentums.append(next_momentum)
        next_velocities.append(next_velocity)
    return tuple(next_cores), tuple(next_momentums), tuple(next_velocities)


def _make_t1_compiled_primal(parent: LaneBT1Artifact, inputs: T1ReplayInputs):
    """Build the full packed replay with TensorFlow-owned control flow."""

    settings = parent.settings
    learning_rate = tf.cast(tf.constant(settings.learning_rate, tf.float32), DTYPE)
    clip = tf.constant(settings.gradient_clip_norm, DTYPE)
    tau = tf.constant(settings.tau, DTYPE)
    l1 = tf.constant(settings.l1_weight, DTYPE)
    l2 = tf.constant(settings.l2_weight, DTYPE)
    batch_size = int(settings.batch_size)
    train_steps = int(settings.train_steps)

    @tf.function(jit_compile=True, reduce_retracing=True)
    def compiled_primal(theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        parameters = tf.reshape(tf.convert_to_tensor(theta, DTYPE), [PARAMETER_DIM])
        initial = inputs.initial_packed_cores * inputs.packed_mask
        momentums = tf.zeros_like(initial)
        velocities = tf.zeros_like(initial)

        def condition(step, _cores, _momentums, _velocities):
            return step < tf.constant(train_steps, tf.int32)

        def body(step, cores, active_momentums, active_velocities):
            indices = inputs.training_batch_indices[step]
            physical = tf.gather(inputs.training_physical_points, indices)
            basis_values = tf.gather(inputs.training_basis_values, indices)
            origin_target_sqrt = tf.gather(
                inputs.training_origin_target_sqrt, indices
            )
            weights = tf.gather(inputs.training_integration_weights, indices)
            active_log = _physical_log_target_value_xla(
                parameters, physical, inputs.observation
            )
            origin_log = _physical_log_target_value_xla(
                tf.zeros_like(parameters), physical, inputs.observation
            )
            target_sqrt = origin_target_sqrt * tf.exp(
                tf.constant(0.5, DTYPE) * (active_log - origin_log)
            )
            raw_alpha = weights * (tf.square(target_sqrt) + tau)
            alpha = raw_alpha / tf.reduce_sum(raw_alpha)
            with tf.GradientTape() as tape:
                tape.watch(cores)
                amplitude = packed_amplitude(cores, basis_values)
                rho = tf.square(amplitude) + tau
                active_l1, active_l2 = packed_per_core_regularizers(
                    cores, inputs.packed_mask
                )
                loss = (
                    -tf.reduce_sum(alpha * tf.math.log(rho))
                    + tf.math.log(
                        packed_square_mass(cores, inputs.mass_matrices) + tau
                    )
                    + l1 * active_l1
                    + l2 * active_l2
                )
            gradients = tape.gradient(loss, cores)
            next_cores, next_m, next_v = packed_adam_apply_gradients(
                cores,
                active_momentums,
                active_velocities,
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
            condition,
            body,
            (tf.constant(0, tf.int32), initial, momentums, velocities),
            parallel_iterations=1,
            maximum_iterations=train_steps,
        )
        active_calibration = _physical_log_target_value_xla(
            parameters, inputs.calibration_physical_points, inputs.observation
        )
        origin_calibration = _physical_log_target_value_xla(
            tf.zeros_like(parameters),
            inputs.calibration_physical_points,
            inputs.observation,
        )
        active_log_weight = (
            inputs.calibration_origin_log_likelihood
            + active_calibration
            - origin_calibration
        )
        shifted_log_mass = (
            tf.constant(
                parent.calibration_estimate.log_shifted_normalizer, DTYPE
            )
            + tf.reduce_logsumexp(active_log_weight)
            - tf.reduce_logsumexp(inputs.calibration_origin_log_likelihood)
        )
        target_mass = tf.exp(shifted_log_mass)
        scale = tf.sqrt(
            (target_mass - tau) / packed_square_mass(trained, inputs.mass_matrices)
        )
        axis_scale = tf.concat(
            [tf.reshape(scale, [1]), tf.ones([35], DTYPE)], axis=0
        )[:, tf.newaxis, tf.newaxis, tf.newaxis]
        calibrated = trained * axis_scale * inputs.packed_mask
        value = (
            tf.math.log(
                packed_square_mass(calibrated, inputs.mass_matrices) + tau
            )
            - tf.constant(parent.shift_constant, DTYPE)
        )
        return calibrated, value

    return compiled_primal


def _make_t1_compiled_bundle(parent: LaneBT1Artifact, inputs: T1ReplayInputs):
    """Compile origin replay, core tangents, and independent scalar FD checks."""

    primal = _make_t1_compiled_primal(parent, inputs)

    @tf.function(jit_compile=True, reduce_retracing=True)
    def compiled_bundle():
        origin = tf.zeros([PARAMETER_DIM], DTYPE)
        origin_cores, origin_value = primal(origin)
        tangent_array = tf.TensorArray(
            DTYPE, size=PARAMETER_DIM, element_shape=origin_cores.shape
        )
        score_array = tf.TensorArray(DTYPE, size=PARAMETER_DIM, element_shape=[])
        plus_array = tf.TensorArray(DTYPE, size=PARAMETER_DIM, element_shape=[])
        minus_array = tf.TensorArray(DTYPE, size=PARAMETER_DIM, element_shape=[])
        tangent_step = tf.constant(TANGENT_FINITE_DIFFERENCE_STEP, DTYPE)
        fd_step = tf.constant(FINITE_DIFFERENCE_STEP, DTYPE)

        def condition(parameter, _tangents, _scores, _plus, _minus):
            return parameter < PARAMETER_DIM

        def body(parameter, tangents, scores, plus, minus):
            direction = tf.one_hot(parameter, PARAMETER_DIM, dtype=DTYPE)
            tangent_plus_cores, tangent_plus_value = primal(
                origin + tangent_step * direction
            )
            tangent_minus_cores, tangent_minus_value = primal(
                origin - tangent_step * direction
            )
            _, plus_value = primal(origin + fd_step * direction)
            _, minus_value = primal(origin - fd_step * direction)
            return (
                parameter + 1,
                tangents.write(
                    parameter,
                    (tangent_plus_cores - tangent_minus_cores)
                    / (tf.constant(2.0, DTYPE) * tangent_step),
                ),
                scores.write(
                    parameter,
                    (tangent_plus_value - tangent_minus_value)
                    / (tf.constant(2.0, DTYPE) * tangent_step),
                ),
                plus.write(parameter, plus_value),
                minus.write(parameter, minus_value),
            )

        _, tangent_array, score_array, plus_array, minus_array = tf.while_loop(
            condition,
            body,
            (
                tf.constant(0, tf.int32),
                tangent_array,
                score_array,
                plus_array,
                minus_array,
            ),
            parallel_iterations=1,
            maximum_iterations=PARAMETER_DIM,
        )
        return (
            origin_cores,
            origin_value,
            tangent_array.stack(),
            score_array.stack(),
            plus_array.stack(),
            minus_array.stack(),
        )

    return compiled_bundle


def _t1_functional_replay_metrics(
    replayed: tf.Tensor, inputs: T1ReplayInputs, tau: tf.Tensor
) -> tuple[tf.Tensor, tf.Tensor]:
    """Compare gauge-invariant full and prefix densities on frozen clouds."""

    observed_rows = (
        packed_normalized_density(
            replayed, inputs.training_basis_values, inputs.mass_matrices, tau
        ),
        packed_normalized_density(
            replayed, inputs.calibration_basis_values, inputs.mass_matrices, tau
        ),
        packed_normalized_prefix_density(
            replayed, inputs.training_basis_values, inputs.mass_matrices, tau
        ),
        packed_normalized_prefix_density(
            replayed, inputs.calibration_basis_values, inputs.mass_matrices, tau
        ),
    )
    reference_rows = (
        packed_normalized_density(
            inputs.parent_packed_cores,
            inputs.training_basis_values,
            inputs.mass_matrices,
            tau,
        ),
        packed_normalized_density(
            inputs.parent_packed_cores,
            inputs.calibration_basis_values,
            inputs.mass_matrices,
            tau,
        ),
        packed_normalized_prefix_density(
            inputs.parent_packed_cores,
            inputs.training_basis_values,
            inputs.mass_matrices,
            tau,
        ),
        packed_normalized_prefix_density(
            inputs.parent_packed_cores,
            inputs.calibration_basis_values,
            inputs.mass_matrices,
            tau,
        ),
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


def replay_t1_training_jvp(
    parent: LaneBT1Artifact,
    *,
    inputs: T1ReplayInputs | None = None,
) -> T1ReplayJVPResult:
    """Materially replay T1 and issue packed-XLA origin tangents."""

    prepared = inputs or prepare_t1_replay_inputs(parent)
    packed_cores, value, packed_tangents, score, plus, minus = (
        _make_t1_compiled_bundle(parent, prepared)()
    )
    _core_passed, residual, normalized = material_replay_metrics(
        packed_cores, prepared.parent_packed_cores, prepared.packed_mask
    )
    functional_passed, functional_metrics = _t1_functional_replay_metrics(
        packed_cores, prepared, tf.constant(parent.settings.tau, DTYPE)
    )
    tf.debugging.assert_equal(
        functional_passed, True, "T1 material functional replay gate failed"
    )
    shapes = tuple(core.shape for core in parent.cores)
    cores = unpack_cores(packed_cores, shapes)
    tangent_banks = tuple(
        tuple(
            tf.ensure_shape(
                packed_tangents[parameter, axis, : shape[0], :, : shape[2]],
                shape,
            )
            for parameter in range(PARAMETER_DIM)
        )
        for axis, shape in enumerate(shapes)
    )
    return T1ReplayJVPResult(
        cores=cores,
        tangent_cores=tangent_banks,
        value=value,
        score=score,
        maximum_core_residual=residual,
        maximum_normalized_core_residual=normalized,
        functional_replay_metrics=functional_metrics,
        finite_difference_plus=plus,
        finite_difference_minus=minus,
    )


def replay_t1_training_value(
    parent: LaneBT1Artifact,
    theta: tf.Tensor,
    *,
    inputs: T1ReplayInputs | None = None,
) -> tf.Tensor:
    """Evaluate the same frozen replay scalar for an offline diagnostic theta."""

    prepared = inputs or prepare_t1_replay_inputs(parent)
    _cores, value = _make_t1_compiled_primal(parent, prepared)(
        tf.reshape(tf.convert_to_tensor(theta, DTYPE), [PARAMETER_DIM])
    )
    tf.debugging.assert_all_finite(value, "frozen replay value")
    return value


def _semantic_sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "ascii"
    )
    return hashlib.sha256(encoded).hexdigest()


def load_t1_training_jvp_child(
    directory: Path,
    *,
    parent: LaneBT1Artifact,
) -> tuple[LaneBParameterChild, dict[str, object]]:
    """Load only a passed, repository-issued material-replay tangent artifact."""

    output = Path(directory)
    payload = json.loads((output / "result.json").read_text())
    if (
        payload.get("schema_version") != ISSUER_SCHEMA
        or payload.get("status")
        != "PASS_T1_MATERIAL_TRAINING_REPLAY_AND_FD_TANGENT"
        or payload.get("parent_identity") != parent.identity.hash.value
    ):
        raise ValueError("T1 training-JVP artifact schema/status/parent mismatch")
    identity_payload = payload.get("issuer_identity_payload")
    if not isinstance(identity_payload, dict):
        raise ValueError("T1 training-JVP identity payload missing")
    if (
        identity_payload.get("issuer_id") != ISSUER_ID
        or identity_payload.get("parent_identity") != parent.identity.hash.value
        or identity_payload.get("replay_id") != REPLAY_ID
        or identity_payload.get("classification") != "extension_or_invention"
        or identity_payload.get("shift_derivative_policy")
        != SHIFT_DERIVATIVE_POLICY
        or identity_payload.get("tau_derivative_policy") != TAU_DERIVATIVE_POLICY
        or identity_payload.get("runtime_score_backend") != RUNTIME_SCORE_BACKEND
        or identity_payload.get("offline_issuer_derivative")
        != OFFLINE_ISSUER_DERIVATIVE
        or identity_payload.get("material_replay_policy_id")
        != MATERIAL_REPLAY_POLICY_ID
        or identity_payload.get("packed_xla_policy_id") != PACKED_XLA_POLICY_ID
        or identity_payload.get("runtime_autodiff") is not False
        or identity_payload.get("runtime_finite_difference") is not False
        or identity_payload.get("hmc_authorized") is not False
        or _semantic_sha256(identity_payload) != payload.get("issuer_identity")
    ):
        raise ValueError("T1 training-JVP issuer identity mismatch")
    expected_replay_gate = {
        "material_functional_atol": MATERIAL_REPLAY_ATOL,
        "material_functional_rtol": MATERIAL_REPLAY_RTOL,
        "maximum_normalized_functional_residual": 1.0,
        "functional_screen_order": list(FUNCTIONAL_SCREEN_ORDER),
        "functional_screen_columns": list(FUNCTIONAL_SCREEN_COLUMNS),
        "tangent_finite_difference_step": TANGENT_FINITE_DIFFERENCE_STEP,
        "finite_difference_step": FINITE_DIFFERENCE_STEP,
        "finite_difference_atol": FINITE_DIFFERENCE_ATOL,
        "finite_difference_rtol": FINITE_DIFFERENCE_RTOL,
        "memory_cap_bytes": MEMORY_CAP_BYTES,
        "gpu_memory_limit_mib": GPU_MEMORY_LIMIT_MIB,
    }
    if identity_payload.get("replay_gate") != expected_replay_gate:
        raise ValueError("T1 training-JVP replay gate identity mismatch")
    expected_optimizer = {
        "family": "keras3_adam_functional_exact_update_order",
        "learning_rate": parent.settings.learning_rate,
        "beta_1": ADAM_BETA_1,
        "beta_2": ADAM_BETA_2,
        "epsilon": ADAM_EPSILON,
        "gradient_clip_norm": parent.settings.gradient_clip_norm,
        "train_steps": parent.settings.train_steps,
        "batch_size": parent.settings.batch_size,
        "jit_compile": True,
        "full_program_control_flow": "tensorflow_while_loop",
        "python_numerical_loops": False,
    }
    if identity_payload.get("optimizer") != expected_optimizer:
        raise ValueError("T1 training-JVP optimizer identity mismatch")
    if _semantic_sha256(identity_payload.get("training_cloud_manifest")) != _semantic_sha256(
        dict(parent.training_cloud_manifest)
    ):
        raise ValueError("T1 training-JVP training cloud identity mismatch")
    if _semantic_sha256(identity_payload.get("calibration_estimate")) != _semantic_sha256(
        parent.calibration_estimate.manifest_payload()
    ):
        raise ValueError("T1 training-JVP calibration identity mismatch")
    source_hashes = identity_payload.get("source_sha256")
    if (
        not isinstance(source_hashes, dict)
        or set(source_hashes) != set(REQUIRED_ISSUER_SOURCE_PATHS)
    ):
        raise ValueError("T1 training-JVP source closure missing")
    for relative_path, expected in source_hashes.items():
        candidate = (ROOT / str(relative_path)).resolve()
        try:
            candidate.relative_to(ROOT)
        except ValueError as exc:
            raise ValueError("T1 training-JVP source path escapes repository") from exc
        if not candidate.is_file():
            raise ValueError(f"T1 training-JVP source missing: {relative_path}")
        observed = hashlib.sha256(candidate.read_bytes()).hexdigest()
        if observed != expected:
            raise ValueError(f"T1 training-JVP source closure stale: {relative_path}")
    gates = payload.get("hard_gates")
    required = (
        "training_and_calibration_cloud_hashes",
        "material_functional_replay",
        "material_scalar_replay",
        "manual_issued_tangent_parity",
        "independent_step_halving_fd_parity",
        "memory_under_6_gib",
    )
    if not isinstance(gates, dict) or not all(gates.get(name) is True for name in required):
        raise ValueError("T1 training-JVP hard gate missing or failed")
    material_evidence = identity_payload.get("material_replay_evidence")
    if not isinstance(material_evidence, dict):
        raise ValueError("T1 training-JVP material replay evidence missing")
    functional_metrics = material_evidence.get("functional_replay_metrics")
    if (
        not isinstance(functional_metrics, list)
        or len(functional_metrics) != len(FUNCTIONAL_SCREEN_ORDER)
        or any(not isinstance(row, list) or len(row) != len(FUNCTIONAL_SCREEN_COLUMNS) for row in functional_metrics)
    ):
        raise ValueError("T1 training-JVP functional replay evidence malformed")
    flattened_metrics = [float(value) for row in functional_metrics for value in row]
    normalized_metrics = [float(row[1]) for row in functional_metrics]
    scalar_absolute = float(material_evidence.get("scalar_absolute_residual", math.inf))
    scalar_normalized = float(material_evidence.get("scalar_normalized_residual", math.inf))
    scalar_log = float(material_evidence.get("scalar_log_residual", math.inf))
    if (
        not all(math.isfinite(value) for value in flattened_metrics)
        or not all(value <= 1.0 for value in normalized_metrics)
        or functional_metrics != payload.get("functional_replay_metrics")
        or not all(math.isfinite(value) for value in (scalar_absolute, scalar_normalized, scalar_log))
        or scalar_normalized > 1.0
        or scalar_absolute != float(payload.get("scalar_absolute_residual", math.nan))
        or scalar_normalized != float(payload.get("scalar_normalized_residual", math.nan))
        or scalar_log != float(payload.get("scalar_log_residual", math.nan))
    ):
        raise ValueError("T1 training-JVP material replay evidence failed")
    derivative_evidence = identity_payload.get("derivative_evidence")
    fd_rows = (
        derivative_evidence.get("independent_finite_difference_rows")
        if isinstance(derivative_evidence, dict)
        else None
    )
    if (
        not isinstance(fd_rows, list)
        or len(fd_rows) != PARAMETER_DIM
        or fd_rows != payload.get("finite_difference_rows")
    ):
        raise ValueError("T1 training tangent derivative evidence missing")
    for parameter, row in enumerate(fd_rows):
        if not isinstance(row, dict) or int(row.get("parameter", -1)) != parameter:
            raise ValueError("T1 training tangent derivative row mismatch")
        step = float(row.get("step", math.nan))
        observed = float(row.get("finite_difference", math.nan))
        issued = float(row.get("issued_tangent_score", math.nan))
        residual = float(row.get("absolute_residual", math.nan))
        if (
            step != FINITE_DIFFERENCE_STEP
            or not all(math.isfinite(value) for value in (observed, issued, residual))
            or residual != abs(observed - issued)
            or residual
            > FINITE_DIFFERENCE_ATOL + FINITE_DIFFERENCE_RTOL * abs(issued)
        ):
            raise ValueError("T1 training tangent derivative evidence failed")
    tensors = payload.get("tensors")
    expected_hashes = identity_payload.get("tangent_tensor_sha256")
    if not isinstance(tensors, dict) or not isinstance(expected_hashes, dict):
        raise ValueError("T1 training-JVP tensor ledger missing")

    def read(name: str, shape: tf.TensorShape) -> tf.Tensor:
        row = tensors.get(name)
        if not isinstance(row, dict) or expected_hashes.get(name) != row.get("sha256"):
            raise ValueError(f"T1 training-JVP tensor identity mismatch: {name}")
        serialized = tf.io.read_file((output / str(row["path"])).as_posix())
        if hashlib.sha256(bytes(serialized.numpy())).hexdigest() != row.get("sha256"):
            raise ValueError(f"T1 training-JVP tensor hash mismatch: {name}")
        value = tf.io.parse_tensor(serialized, out_type=tf.float64)
        return tf.ensure_shape(value, shape)

    banks = []
    for axis, core in enumerate(parent.cores):
        banks.append(
            tuple(read(f"tangent_{axis:02d}_{parameter}", core.shape) for parameter in range(PARAMETER_DIM))
        )
    child = LaneBParameterChild(parent, tuple(banks))
    if child.identity.hash.value != payload.get("child_identity"):
        raise ValueError("T1 training-JVP child identity mismatch")
    value, score = child.increment_and_score(tf.zeros([PARAMETER_DIM], DTYPE))
    tf.debugging.assert_near(
        value, tf.constant(float(payload["manual_value"]), DTYPE), atol=2e-13, rtol=0.0
    )
    tf.debugging.assert_near(
        score,
        tf.constant(payload["manual_score"], DTYPE),
        atol=2e-10,
        rtol=2e-10,
    )
    return child, payload


__all__ = [
    "ISSUER_ID",
    "ISSUER_SCHEMA",
    "FINITE_DIFFERENCE_ATOL",
    "FINITE_DIFFERENCE_RTOL",
    "FINITE_DIFFERENCE_STEP",
    "FUNCTIONAL_SCREEN_COLUMNS",
    "FUNCTIONAL_SCREEN_ORDER",
    "GPU_MEMORY_LIMIT_MIB",
    "MEMORY_CAP_BYTES",
    "OFFLINE_ISSUER_DERIVATIVE",
    "REPLAY_ID",
    "REQUIRED_ISSUER_SOURCE_PATHS",
    "RUNTIME_SCORE_BACKEND",
    "SHIFT_DERIVATIVE_POLICY",
    "TAU_DERIVATIVE_POLICY",
    "TANGENT_FINITE_DIFFERENCE_STEP",
    "T1ReplayInputs",
    "T1ReplayJVPResult",
    "functional_adam_apply_gradients",
    "functional_adam_step",
    "load_t1_training_jvp_child",
    "prepare_t1_replay_inputs",
    "replay_t1_training_jvp",
    "replay_t1_training_value",
]
