"""Model callbacks for initial-design LEDH-PFPF-GenUT experiments.

Target densities and parameter scores are delegated to repository model classes.
The callbacks in this module define only the Gaussian proposal surface used by
the LEDH flow and the deterministic Gaussian pushes used to construct particles.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Protocol

import tensorflow as tf

from bayesfilter.highdim import ledh_contract_e_tp_lgssm_tf as lgssm_score
from bayesfilter.highdim.cubature_genut_adapters import (
    diagonal_lgssm_candidate_adapter,
    exact_transformed_sv_candidate_adapter,
    generalized_sv_prior_mean_candidate_adapter,
    ksc_mixture_sv_candidate_adapter,
    parameterized_austria_sir_candidate_adapter,
    predator_prey_candidate_adapter,
)
from bayesfilter.highdim.cubature_genut_filter import CandidateModelAdapter
from bayesfilter.highdim.models import (
    GeneralizedSVPriorMeanSSM,
    PredatorPreySSM,
    StochasticVolatilitySSM,
)
from bayesfilter.highdim.sir_latent_preclip_tf import LatentPreclipSIRSSM
from bayesfilter.highdim import sir_online_score_teacher_tf as sir_score
from bayesfilter.highdim.sv_mixture_cut4 import (
    ExactTransformedSVSSM,
    KSCMixtureTransformedSVSSM,
)


Tensor = tf.Tensor
_LOG_TWO_PI = math.log(2.0 * math.pi)


class AnalyticalScoreModel(Protocol):
    def state_dim(self) -> int: ...
    def observation_dim(self) -> int: ...
    def parameter_dim(self) -> int: ...
    def initial_log_density(self, theta: Tensor, x0: Tensor) -> Tensor: ...
    def initial_log_density_parameter_score(
        self, theta: Tensor, x0: Tensor
    ) -> Tensor: ...
    def transition_log_density(
        self, theta: Tensor, x_prev: Tensor, x_next: Tensor, t: int
    ) -> Tensor: ...
    def transition_log_density_parameter_score(
        self, theta: Tensor, x_prev: Tensor, x_next: Tensor, t: int
    ) -> Tensor: ...
    def observation_log_density(
        self, theta: Tensor, x_t: Tensor, y_t: Tensor, t: int
    ) -> Tensor: ...
    def observation_log_density_parameter_score(
        self, theta: Tensor, x_t: Tensor, y_t: Tensor, t: int
    ) -> Tensor: ...


ObservationCallbacks = tuple[
    Callable[[Tensor], Tensor],
    Callable[[Tensor], Tensor],
    Callable[[Tensor, Tensor], Tensor],
]


@dataclass(frozen=True)
class LEDHGenUTModelCallbacks:
    """Fixed target and LEDH proposal definitions for one active model."""

    model_id: str
    model: AnalyticalScoreModel
    push_adapter: CandidateModelAdapter
    transition_before_first_observation: bool
    target_time_offset: int
    initial_covariance: Callable[[Tensor], Tensor]
    transition_mean: Callable[[Tensor, Tensor, int], Tensor]
    transition_covariance: Callable[[Tensor], Tensor]
    transition_matrix: Callable[[Tensor], Tensor]
    observation_covariance: Callable[[Tensor], Tensor]
    proposal_observation: Callable[[Tensor, Tensor], Tensor]
    observation_callbacks: Callable[[Tensor, int], ObservationCallbacks]

    def __post_init__(self) -> None:
        if self.model.state_dim() != self.push_adapter.state_dimension:
            raise ValueError("model and push-adapter state dimensions differ")
        if self.model.parameter_dim() != self.push_adapter.parameter_count:
            raise ValueError("model and push-adapter parameter dimensions differ")
        if self.target_time_offset not in (0, 1):
            raise ValueError("target time offset must be zero or one")

    @property
    def state_dimension(self) -> int:
        return self.model.state_dim()

    @property
    def observation_dimension(self) -> int:
        return self.model.observation_dim()

    @property
    def parameter_count(self) -> int:
        return self.model.parameter_dim()

    def target_time(self, observation_index: int | Tensor) -> int | Tensor:
        if tf.is_tensor(observation_index):
            return observation_index + tf.cast(
                self.target_time_offset, observation_index.dtype
            )
        return int(observation_index) + self.target_time_offset


def _normal_cdf(value: Tensor) -> Tensor:
    return 0.5 * (
        tf.constant(1.0, value.dtype)
        + tf.math.erf(value / tf.sqrt(tf.constant(2.0, value.dtype)))
    )


def _identity_residual(predicted: Tensor, observed: Tensor) -> Tensor:
    return observed[None, None, :] - predicted


def _linear_observation_callbacks(matrix: Tensor) -> ObservationCallbacks:
    matrix = tf.convert_to_tensor(matrix, tf.float32)

    def observation_fn(points: Tensor) -> Tensor:
        return tf.einsum("bnd,od->bno", points, matrix)

    def jacobian_fn(points: Tensor) -> Tensor:
        return tf.broadcast_to(
            matrix[None, None, :, :],
            [tf.shape(points)[0], tf.shape(points)[1], *matrix.shape],
        )

    return observation_fn, jacobian_fn, _identity_residual


def _scalar_linear_observation_callbacks(
    scale: Tensor, offset: Tensor
) -> ObservationCallbacks:
    scale = tf.reshape(tf.convert_to_tensor(scale, tf.float32), [])
    offset = tf.reshape(tf.convert_to_tensor(offset, tf.float32), [])

    def observation_fn(points: Tensor) -> Tensor:
        return scale * points + offset

    def jacobian_fn(points: Tensor) -> Tensor:
        return tf.ones(
            [tf.shape(points)[0], tf.shape(points)[1], 1, 1], points.dtype
        ) * scale

    return observation_fn, jacobian_fn, _identity_residual


def _normal_log_density(residual: Tensor, covariance: Tensor) -> Tensor:
    covariance = tf.convert_to_tensor(covariance, residual.dtype)
    chol = tf.linalg.cholesky(covariance)
    solved = tf.linalg.matrix_transpose(
        tf.linalg.cholesky_solve(chol, tf.linalg.matrix_transpose(residual))
    )
    dimension = tf.cast(tf.shape(residual)[-1], residual.dtype)
    logdet = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(chol)))
    return -0.5 * (
        dimension * tf.cast(_LOG_TWO_PI, residual.dtype)
        + logdet
        + tf.reduce_sum(residual * solved, axis=-1)
    )


@dataclass(frozen=True)
class DiagonalLGSSMAnalyticalModel:
    """Five-parameter diagonal LGSSM with repository standard-score equations."""

    observation_matrix: Tensor

    def state_dim(self) -> int:
        return 3

    def observation_dim(self) -> int:
        return 3

    def parameter_dim(self) -> int:
        return 5

    def initial_log_density(self, theta: Tensor, x0: Tensor) -> Tensor:
        variance = tf.square(theta[3]) / (1.0 - tf.square(theta[:3]))
        return _normal_log_density(x0, tf.linalg.diag(variance))

    def initial_log_density_parameter_score(
        self, theta: Tensor, x0: Tensor
    ) -> Tensor:
        return lgssm_score._initial_target_model_score_marks(theta, x0)  # noqa: SLF001

    def transition_log_density(
        self, theta: Tensor, x_prev: Tensor, x_next: Tensor, t: int
    ) -> Tensor:
        del t
        pairwise_log, _ = lgssm_score._target_transition_log_density_and_score(  # noqa: SLF001
            theta, x_next, x_prev
        )
        return tf.linalg.diag_part(pairwise_log)

    def transition_log_density_parameter_score(
        self, theta: Tensor, x_prev: Tensor, x_next: Tensor, t: int
    ) -> Tensor:
        del t
        _, pairwise_score = lgssm_score._target_transition_log_density_and_score(  # noqa: SLF001
            theta, x_next, x_prev
        )
        row = tf.range(tf.shape(x_prev)[0], dtype=tf.int32)
        return tf.gather_nd(pairwise_score, tf.stack([row, row], axis=1))

    def observation_log_density(
        self, theta: Tensor, x_t: Tensor, y_t: Tensor, t: int
    ) -> Tensor:
        del t
        predicted = tf.linalg.matmul(
            x_t, tf.cast(self.observation_matrix, x_t.dtype), transpose_b=True
        )
        return _normal_log_density(
            y_t[None, :] - predicted,
            tf.square(theta[4]) * tf.eye(3, dtype=theta.dtype),
        )

    def observation_log_density_parameter_score(
        self, theta: Tensor, x_t: Tensor, y_t: Tensor, t: int
    ) -> Tensor:
        del t
        return lgssm_score._target_observation_score(theta, x_t, y_t)  # noqa: SLF001


def diagonal_lgssm_callbacks() -> LEDHGenUTModelCallbacks:
    matrix = tf.constant(
        [[1.0, 0.25, -0.15], [0.2, 1.1, 0.3], [-0.1, 0.35, 0.9]],
        tf.float32,
    )
    model = DiagonalLGSSMAnalyticalModel(matrix)
    return LEDHGenUTModelCallbacks(
        model_id="lgssm_T50",
        model=model,
        push_adapter=diagonal_lgssm_candidate_adapter(observation_matrix=matrix),
        transition_before_first_observation=False,
        target_time_offset=0,
        initial_covariance=lambda theta: tf.linalg.diag(
            tf.square(theta[3]) / (1.0 - tf.square(theta[:3]))
        ),
        transition_mean=lambda theta, points, _time: points * theta[:3],
        transition_covariance=lambda theta: tf.square(theta[3])
        * tf.eye(3, dtype=theta.dtype),
        transition_matrix=lambda theta: tf.linalg.diag(theta[:3]),
        observation_covariance=lambda theta: tf.square(theta[4])
        * tf.eye(3, dtype=theta.dtype),
        proposal_observation=lambda _theta, observation: observation,
        observation_callbacks=lambda _theta, _time: _linear_observation_callbacks(
            matrix
        ),
    )


def _sv_callbacks(
    *, exact: bool
) -> LEDHGenUTModelCallbacks:
    model: ExactTransformedSVSSM | KSCMixtureTransformedSVSSM
    adapter: CandidateModelAdapter
    if exact:
        model = ExactTransformedSVSSM()
        adapter = exact_transformed_sv_candidate_adapter()
        model_id = "exact_sv_T10"
    else:
        model = KSCMixtureTransformedSVSSM()
        adapter = ksc_mixture_sv_candidate_adapter()
        model_id = "ksc_sv_T10"

    def gamma(theta: Tensor) -> Tensor:
        return _normal_cdf(theta[0])

    return LEDHGenUTModelCallbacks(
        model_id=model_id,
        model=model,
        push_adapter=adapter,
        transition_before_first_observation=False,
        target_time_offset=0,
        initial_covariance=lambda theta: tf.reshape(
            1.0 / (1.0 - tf.square(gamma(theta))), [1, 1]
        ),
        transition_mean=lambda theta, points, _time: gamma(theta) * points,
        transition_covariance=lambda theta: tf.eye(1, dtype=theta.dtype),
        transition_matrix=lambda theta: tf.reshape(gamma(theta), [1, 1]),
        observation_covariance=lambda theta: tf.reshape(
            tf.cast(math.pi * math.pi / 2.0, theta.dtype), [1, 1]
        ),
        proposal_observation=lambda _theta, observation: observation,
        observation_callbacks=lambda theta, _time: _scalar_linear_observation_callbacks(
            tf.constant(1.0, theta.dtype), 2.0 * theta[1]
        ),
    )


def exact_sv_callbacks() -> LEDHGenUTModelCallbacks:
    return _sv_callbacks(exact=True)


def ksc_sv_callbacks() -> LEDHGenUTModelCallbacks:
    return _sv_callbacks(exact=False)


def generalized_sv_callbacks() -> LEDHGenUTModelCallbacks:
    model = GeneralizedSVPriorMeanSSM()

    def components(theta: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        gamma = _normal_cdf(theta[0])
        tau = tf.exp(theta[1])
        mu = theta[2] * tau
        return gamma, tau, mu

    return LEDHGenUTModelCallbacks(
        model_id="generalized_sv_T10",
        model=model,
        push_adapter=generalized_sv_prior_mean_candidate_adapter(),
        transition_before_first_observation=True,
        target_time_offset=0,
        initial_covariance=lambda theta: tf.reshape(
            1.0 / (1.0 - tf.square(components(theta)[0])), [1, 1]
        ),
        transition_mean=lambda theta, points, _time: components(theta)[2]
        + components(theta)[0] * (points - components(theta)[2]),
        transition_covariance=lambda theta: tf.eye(1, dtype=theta.dtype),
        transition_matrix=lambda theta: tf.reshape(components(theta)[0], [1, 1]),
        observation_covariance=lambda theta: tf.reshape(
            tf.cast(2.0, theta.dtype), [1, 1]
        ),
        proposal_observation=lambda _theta, observation: tf.math.log(
            tf.square(observation) + tf.cast(1.0e-6, observation.dtype)
        ),
        observation_callbacks=lambda theta, _time: _scalar_linear_observation_callbacks(
            components(theta)[1], tf.constant(0.0, theta.dtype)
        ),
    )


def predator_prey_callbacks(model: PredatorPreySSM) -> LEDHGenUTModelCallbacks:
    identity = tf.eye(2, dtype=tf.float32)
    return LEDHGenUTModelCallbacks(
        model_id="predator_prey_T20",
        model=model,
        push_adapter=predator_prey_candidate_adapter(),
        transition_before_first_observation=True,
        target_time_offset=1,
        initial_covariance=lambda theta: tf.cast(model.initial_covariance, theta.dtype),
        transition_mean=lambda theta, points, _time: tf.cast(
            model.transition_mean(tf.cast(theta, model.dtype), tf.cast(points, model.dtype)),
            theta.dtype,
        ),
        transition_covariance=lambda theta: tf.cast(
            model.process_covariance, theta.dtype
        ),
        transition_matrix=lambda theta: tf.eye(2, dtype=theta.dtype),
        observation_covariance=lambda theta: tf.cast(
            model.observation_covariance, theta.dtype
        ),
        proposal_observation=lambda _theta, observation: observation,
        observation_callbacks=lambda _theta, _time: _linear_observation_callbacks(
            identity
        ),
    )


@dataclass(frozen=True)
class LatentPreclipSIRTeacherModel:
    """XLA-oriented repository standard-score surface for latent Austria SIR."""

    static_spec: sir_score.SIRTeacherStaticSpec

    def state_dim(self) -> int:
        return self.static_spec.state_dimension

    def observation_dim(self) -> int:
        return self.static_spec.observation_dimension

    def parameter_dim(self) -> int:
        return sir_score.PARAMETER_COUNT

    def initial_log_density(self, theta: Tensor, x0: Tensor) -> Tensor:
        return sir_score.initial_log_density_and_score(
            theta, x0, spec=self.static_spec
        )[0]

    def initial_log_density_parameter_score(
        self, theta: Tensor, x0: Tensor
    ) -> Tensor:
        return sir_score.initial_log_density_and_score(
            theta, x0, spec=self.static_spec
        )[1]

    def transition_log_density(
        self, theta: Tensor, x_prev: Tensor, x_next: Tensor, t: int
    ) -> Tensor:
        return sir_score.transition_log_density_and_score(
            theta, x_prev, x_next, time_index=t, spec=self.static_spec
        )[0]

    def transition_log_density_parameter_score(
        self, theta: Tensor, x_prev: Tensor, x_next: Tensor, t: int
    ) -> Tensor:
        return sir_score.transition_log_density_and_score(
            theta, x_prev, x_next, time_index=t, spec=self.static_spec
        )[1]

    def observation_log_density(
        self, theta: Tensor, x_t: Tensor, y_t: Tensor, t: int
    ) -> Tensor:
        return sir_score.observation_log_density_and_score(
            theta, x_t, y_t, time_index=t, spec=self.static_spec
        )[0]

    def observation_log_density_parameter_score(
        self, theta: Tensor, x_t: Tensor, y_t: Tensor, t: int
    ) -> Tensor:
        return sir_score.observation_log_density_and_score(
            theta, x_t, y_t, time_index=t, spec=self.static_spec
        )[1]


def austria_sir_callbacks(model: LatentPreclipSIRSSM) -> LEDHGenUTModelCallbacks:
    infectious_matrix = tf.stack(
        [tf.one_hot(2 * index + 1, 18, dtype=tf.float32) for index in range(9)],
        axis=0,
    )

    standard_score_model = LatentPreclipSIRTeacherModel(
        sir_score.static_spec_from_model(model)
    )

    def transition_mean(theta: Tensor, points: Tensor, time_index: int) -> Tensor:
        mean, _ = sir_score._transition_mean_and_parameter_tangent(  # noqa: SLF001
            tf.cast(points, tf.float64),
            tf.cast(theta, tf.float64),
            time_index,
            standard_score_model.static_spec,
        )
        return tf.cast(mean, theta.dtype)

    return LEDHGenUTModelCallbacks(
        model_id="austria_sir_T20",
        model=standard_score_model,
        push_adapter=parameterized_austria_sir_candidate_adapter(),
        transition_before_first_observation=True,
        target_time_offset=1,
        initial_covariance=lambda theta: tf.eye(18, dtype=theta.dtype),
        transition_mean=transition_mean,
        transition_covariance=lambda theta: tf.eye(18, dtype=theta.dtype),
        transition_matrix=lambda theta: tf.eye(18, dtype=theta.dtype),
        observation_covariance=lambda theta: tf.eye(9, dtype=theta.dtype)
        * tf.cast(100.0, theta.dtype)
        * tf.exp(2.0 * theta[2]),
        proposal_observation=lambda _theta, observation: observation,
        observation_callbacks=lambda _theta, _time: _linear_observation_callbacks(
            infectious_matrix
        ),
    )


__all__ = [
    "AnalyticalScoreModel",
    "DiagonalLGSSMAnalyticalModel",
    "LEDHGenUTModelCallbacks",
    "LatentPreclipSIRTeacherModel",
    "austria_sir_callbacks",
    "diagonal_lgssm_callbacks",
    "exact_sv_callbacks",
    "generalized_sv_callbacks",
    "ksc_sv_callbacks",
    "predator_prey_callbacks",
]
