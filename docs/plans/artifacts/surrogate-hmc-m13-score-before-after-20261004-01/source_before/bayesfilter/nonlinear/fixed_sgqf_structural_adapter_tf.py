from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf

from bayesfilter.highdim.models import PredatorPreySSM
from bayesfilter.nonlinear.fixed_sgqf_derivatives_tf import TFFixedSGQFDerivatives
from bayesfilter.nonlinear.fixed_sgqf_tf import (
    TFFixedSGQFAffineModel,
    TFFixedSGQFNonlinearModel,
)
from bayesfilter.structural_tf import TFStructuralStateSpace


_ADMISSION_STATUSES = {"exact_eligible", "approximate_eligible", "ineligible"}


@dataclass(frozen=True)
class TFFixedSGQFStructuralAdapterResult:
    eligible: bool
    reason: str | None
    model: TFFixedSGQFNonlinearModel | TFFixedSGQFAffineModel | None
    derivatives: TFFixedSGQFDerivatives | None = None
    admission_status: str = "ineligible"
    target_scope: str | None = None
    nonclaims: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "eligible", bool(self.eligible))
        if self.reason is not None:
            object.__setattr__(self, "reason", str(self.reason))
        admission_status = str(self.admission_status)
        if admission_status not in _ADMISSION_STATUSES:
            raise ValueError(f"unknown structural adapter admission status: {admission_status!r}")
        object.__setattr__(self, "admission_status", admission_status)
        if self.target_scope is not None:
            object.__setattr__(self, "target_scope", str(self.target_scope))
        object.__setattr__(self, "nonclaims", tuple(str(item) for item in self.nonclaims))

    @property
    def exact_eligible(self) -> bool:
        return self.admission_status == "exact_eligible"

    @property
    def approximate_eligible(self) -> bool:
        return self.admission_status == "approximate_eligible"


def tf_predator_prey_to_fixed_sgqf_model(
    model: PredatorPreySSM,
    theta: tf.Tensor,
    *,
    with_derivatives: bool = False,
) -> TFFixedSGQFStructuralAdapterResult:
    theta = tf.convert_to_tensor(theta, dtype=tf.float64)

    def transition_fn(points: tf.Tensor) -> tf.Tensor:
        values = tf.convert_to_tensor(points, dtype=tf.float64)
        return model.transition_mean(theta, values)

    def observation_fn(points: tf.Tensor) -> tf.Tensor:
        values = tf.convert_to_tensor(points, dtype=tf.float64)
        return values

    fixed_model = TFFixedSGQFNonlinearModel(
        initial_mean=model.initial_mean,
        initial_covariance=model.initial_covariance,
        process_covariance=model.process_covariance,
        observation_covariance=model.observation_covariance,
        transition_fn=transition_fn,
        observation_fn=observation_fn,
        name="fixed_sgqf_predator_prey_adapter",
    )

    if not with_derivatives:
        return TFFixedSGQFStructuralAdapterResult(
            eligible=True,
            reason=None,
            model=fixed_model,
            derivatives=None,
            admission_status="approximate_eligible",
            target_scope="declared_structural_gaussian_projection_predator_prey_adapter",
            nonclaims=(
                "not generic nonlinear-SSM admission",
                "not exact-target admission outside the reviewed predator-prey lane",
            ),
        )

    parameter_dim = int(theta.shape[0])
    state_dim = fixed_model.state_dim
    observation_dim = fixed_model.observation_dim

    def transition_state_jacobian_fn(points: tf.Tensor) -> tf.Tensor:
        values = tf.convert_to_tensor(points, dtype=tf.float64)
        if values.shape.rank == 1:
            values = values[tf.newaxis, :]
        with tf.GradientTape() as tape:
            tape.watch(values)
            outputs = transition_fn(values)
        return tape.batch_jacobian(outputs, values)

    def d_transition_fn(points: tf.Tensor) -> tf.Tensor:
        values = tf.convert_to_tensor(points, dtype=tf.float64)
        if values.shape.rank == 1:
            values = values[tf.newaxis, :]
        with tf.GradientTape() as tape:
            tape.watch(theta)
            outputs = model.transition_mean(theta, values)
        jacobian = tape.jacobian(outputs, theta)
        return tf.transpose(tf.convert_to_tensor(jacobian, dtype=tf.float64), perm=[2, 0, 1])

    def observation_state_jacobian_fn(points: tf.Tensor) -> tf.Tensor:
        values = tf.convert_to_tensor(points, dtype=tf.float64)
        if values.shape.rank == 1:
            values = values[tf.newaxis, :]
        point_count = tf.shape(values)[0]
        return tf.broadcast_to(tf.eye(observation_dim, dtype=tf.float64), [point_count, observation_dim, state_dim])

    def d_observation_fn(points: tf.Tensor) -> tf.Tensor:
        values = tf.convert_to_tensor(points, dtype=tf.float64)
        if values.shape.rank == 1:
            values = values[tf.newaxis, :]
        return tf.zeros([parameter_dim, tf.shape(values)[0], observation_dim], dtype=tf.float64)

    fixed_derivatives = TFFixedSGQFDerivatives(
        d_initial_mean=tf.zeros([parameter_dim, state_dim], dtype=tf.float64),
        d_initial_covariance=tf.zeros([parameter_dim, state_dim, state_dim], dtype=tf.float64),
        d_process_covariance=tf.zeros([parameter_dim, state_dim, state_dim], dtype=tf.float64),
        d_observation_covariance=tf.zeros([parameter_dim, observation_dim, observation_dim], dtype=tf.float64),
        transition_state_jacobian_fn=transition_state_jacobian_fn,
        d_transition_fn=d_transition_fn,
        observation_state_jacobian_fn=observation_state_jacobian_fn,
        d_observation_fn=d_observation_fn,
        name="fixed_sgqf_predator_prey_adapter_derivatives",
    )

    return TFFixedSGQFStructuralAdapterResult(
        eligible=True,
        reason=None,
        model=fixed_model,
        derivatives=fixed_derivatives,
        admission_status="approximate_eligible",
        target_scope="declared_structural_gaussian_projection_predator_prey_adapter",
        nonclaims=(
            "not generic nonlinear-SSM admission",
            "not exact-target admission outside the reviewed predator-prey lane",
        ),
    )


def tf_structural_to_fixed_sgqf_model(
    model: TFStructuralStateSpace,
    *,
    derivatives=None,
) -> TFFixedSGQFStructuralAdapterResult:
    if model.is_affine:
        affine_model = TFFixedSGQFAffineModel(
            initial_mean=model.initial_mean,
            initial_covariance=model.initial_covariance,
            transition_matrix=model.transition_matrix,
            process_covariance=(
                model.innovation_matrix
                @ model.innovation_covariance
                @ tf.transpose(model.innovation_matrix)
            ),
            observation_matrix=model.observation_matrix,
            observation_covariance=model.observation_covariance,
            transition_offset=model.transition_offset,
            observation_offset=model.observation_offset,
            name=f"fixed_sgqf_{model.name}_affine_structural_adapter",
        )
        return TFFixedSGQFStructuralAdapterResult(
            eligible=True,
            reason=None,
            model=affine_model,
            derivatives=None,
            admission_status="exact_eligible",
            target_scope="exact_affine_structural_lane",
            nonclaims=(
                "affine structural exactness does not imply generic nonlinear exact admission",
            ),
        )

    if model.name != "model_c_autonomous_nonlinear_growth":
        return TFFixedSGQFStructuralAdapterResult(
            eligible=False,
            reason="current fixed SGQF structural adapter is exact-eligible only for affine structural models and approximate-eligible only for the autonomous phase nonlinear growth fixture; model_b remains ineligible under the present additive-state lane",
            model=None,
            derivatives=None,
            admission_status="ineligible",
            target_scope="structural_admission_fail_closed",
            nonclaims=(
                "not generic nonlinear-SSM admission",
                "model-specific unsupported routes remain blocked",
            ),
        )

    process_variance = tf.convert_to_tensor(model.innovation_covariance[0, 0], dtype=tf.float64)
    observation_covariance = tf.convert_to_tensor(model.observation_covariance, dtype=tf.float64)
    initial_mean = tf.convert_to_tensor(model.initial_mean, dtype=tf.float64)
    initial_covariance = tf.convert_to_tensor(model.initial_covariance, dtype=tf.float64)

    def transition_fn(points: tf.Tensor) -> tf.Tensor:
        values = tf.convert_to_tensor(points, dtype=tf.float64)
        if values.shape.rank == 1:
            values = values[tf.newaxis, :]
        x_prev = values[:, 0]
        tau_prev = values[:, 1]
        x_next = (
            0.5 * x_prev
            + 25.0 * x_prev / (1.0 + tf.square(x_prev))
            + 8.0 * tf.math.cos(1.2 * tau_prev)
        )
        tau_next = tau_prev + 1.0
        return tf.stack([x_next, tau_next], axis=1)

    def observation_fn(points: tf.Tensor) -> tf.Tensor:
        values = tf.convert_to_tensor(points, dtype=tf.float64)
        if values.shape.rank == 1:
            values = values[tf.newaxis, :]
        return tf.square(values[:, :1]) / 20.0

    fixed_model = TFFixedSGQFNonlinearModel(
        initial_mean=initial_mean,
        initial_covariance=initial_covariance,
        process_covariance=tf.convert_to_tensor(
            [[process_variance, 0.0], [0.0, 0.0]],
            dtype=tf.float64,
        ),
        observation_covariance=observation_covariance,
        transition_fn=transition_fn,
        observation_fn=observation_fn,
        name="fixed_sgqf_model_c_structural_adapter",
    )

    if derivatives is None:
        return TFFixedSGQFStructuralAdapterResult(
            eligible=True,
            reason=None,
            model=fixed_model,
            derivatives=None,
            admission_status="approximate_eligible",
            target_scope="declared_structural_gaussian_projection_model_c_adapter",
            nonclaims=(
                "not exact-target admission",
                "not generic nonlinear-SSM admission outside the reviewed model-c fixture",
            ),
        )

    parameter_dim = int(derivatives.d_initial_mean.shape[0])
    fixed_derivatives = TFFixedSGQFDerivatives(
        d_initial_mean=tf.convert_to_tensor(derivatives.d_initial_mean, dtype=tf.float64),
        d_initial_covariance=tf.convert_to_tensor(derivatives.d_initial_covariance, dtype=tf.float64),
        d_process_covariance=tf.convert_to_tensor(
            [
                [[derivatives.d_innovation_covariance[i, 0, 0], 0.0], [0.0, 0.0]]
                for i in range(parameter_dim)
            ],
            dtype=tf.float64,
        ),
        d_observation_covariance=tf.convert_to_tensor(derivatives.d_observation_covariance, dtype=tf.float64),
        transition_state_jacobian_fn=lambda points: tf.convert_to_tensor(
            derivatives.transition_state_jacobian_fn(
                tf.convert_to_tensor(points, dtype=tf.float64),
                tf.zeros([tf.shape(tf.convert_to_tensor(points, dtype=tf.float64))[0], 1], dtype=tf.float64),
            ),
            dtype=tf.float64,
        ),
        d_transition_fn=lambda points: tf.convert_to_tensor(
            derivatives.d_transition_fn(
                tf.convert_to_tensor(points, dtype=tf.float64),
                tf.zeros([tf.shape(tf.convert_to_tensor(points, dtype=tf.float64))[0], 1], dtype=tf.float64),
            ),
            dtype=tf.float64,
        ),
        observation_state_jacobian_fn=lambda points: tf.convert_to_tensor(
            derivatives.observation_state_jacobian_fn(tf.convert_to_tensor(points, dtype=tf.float64)),
            dtype=tf.float64,
        ),
        d_observation_fn=lambda points: tf.convert_to_tensor(
            derivatives.d_observation_fn(tf.convert_to_tensor(points, dtype=tf.float64)),
            dtype=tf.float64,
        ),
        name="fixed_sgqf_model_c_structural_adapter_derivatives",
    )
    return TFFixedSGQFStructuralAdapterResult(
        eligible=True,
        reason=None,
        model=fixed_model,
        derivatives=fixed_derivatives,
        admission_status="approximate_eligible",
        target_scope="declared_structural_gaussian_projection_model_c_adapter",
        nonclaims=(
            "not exact-target admission",
            "not generic nonlinear-SSM admission outside the reviewed model-c fixture",
        ),
    )
