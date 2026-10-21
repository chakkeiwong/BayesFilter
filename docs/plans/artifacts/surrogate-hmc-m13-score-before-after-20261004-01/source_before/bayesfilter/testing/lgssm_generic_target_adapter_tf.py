"""LGSSM generic SSM target fixtures for adapter-level tests.

These helpers expose the existing static QR LGSSM fixture through the generic
``SSMTargetContract`` and batch-native posterior adapter boundary.  They are
testing fixtures only: no HMC, transport binding, NeuTra training, or posterior
validation is performed here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import tensorflow as tf

from bayesfilter.ssm import (
    BayesianSSMProblem,
    FilterProgram,
    GenericSSMPosteriorAdapter,
    ParameterChart,
    ParameterPrior,
    SSMDataSignature,
    SSMStaticShape,
    SSMTargetContract,
    build_ssm_posterior_adapter,
)
from bayesfilter.testing.tf_hmc_readiness import QRStaticLGSSMTarget


LGSSM_GENERIC_TARGET_NONCLAIMS = (
    "LGSSM generic target adapter fixture only",
    "no HMC tuning or sampling claim",
    "no NeuTra training claim",
    "no posterior convergence claim",
    "no production readiness claim",
    "no scientific validity claim",
)

LGSSM_GENERIC_TARGET_XLA_HMC_NONCLAIMS = (
    "LGSSM generic target adapter XLA-HMC value/score opt-in only",
    "Phase 15 trusted GPU/XLA objective compile gate is the evidence anchor",
    "no fixed-transport HMC chain execution claim",
    "no HMC tuning or sampling claim",
    "no posterior convergence claim",
    "no production readiness claim",
    "no scientific validity claim",
)


@dataclass(frozen=True)
class LGSSMGenericTargetFixture:
    """Generic adapter fixture for the deterministic static QR LGSSM target."""

    source_target: QRStaticLGSSMTarget
    contract: SSMTargetContract
    adapter: GenericSSMPosteriorAdapter

    @property
    def initial_batch(self) -> tf.Tensor:
        return self.source_target.initial_parameters[tf.newaxis, :]


def make_lgssm_generic_target_fixture() -> LGSSMGenericTargetFixture:
    """Return a BayesFilter-owned LGSSM target via the generic SSM adapter."""

    source_target = QRStaticLGSSMTarget.default()
    contract = make_lgssm_generic_target_contract(source_target)
    adapter = build_ssm_posterior_adapter(
        contract=contract,
        prior_log_prob_and_grad=lambda theta: lgssm_gaussian_prior_log_prob_and_grad(
            theta,
            prior_scale=source_target.prior_scale,
        ),
        filter_log_likelihood_and_grad=lambda theta: (
            lgssm_qr_log_likelihood_and_grad(
                theta,
                source_target=source_target,
            )
        ),
        dtype=tf.float64,
        target_scope="lgssm-static-qr-generic-target-fixture",
        evidence_path=(
            "docs/plans/"
            "bayesfilter-lgssm-first-neutra-hmc-phase15-manual-score-xla-"
            "compile-gate-result-2026-07-08.md"
        ),
        xla_hmc_ready=True,
        full_chain_xla_diagnostic_ready=False,
        nonclaims=LGSSM_GENERIC_TARGET_XLA_HMC_NONCLAIMS,
    )
    return LGSSMGenericTargetFixture(
        source_target=source_target,
        contract=contract,
        adapter=adapter,
    )


def make_lgssm_generic_target_contract(
    source_target: QRStaticLGSSMTarget | None = None,
    *,
    data_hash: str = "sha256:lgssm-static-qr-observations-v1",
    model_hash: str = "sha256:lgssm-static-qr-model-v1",
    transform_hash: str = "sha256:identity-rho-log-measurement-noise-v1",
    prior_hash: str = "sha256:standard-gaussian-prior-scale-v1",
    filter_hash: str = "sha256:tf-qr-exact-lgssm-v1",
) -> SSMTargetContract:
    """Build a stable generic target contract for the QR static LGSSM fixture."""

    target = QRStaticLGSSMTarget.default() if source_target is None else source_target
    observations = tf.convert_to_tensor(target.observations, dtype=tf.float64)
    if observations.shape.rank != 2:
        raise ValueError("LGSSM observations must have shape [time, observation]")
    horizon = observations.shape[0]
    observation_dim = observations.shape[1]
    if horizon is None or observation_dim is None:
        raise ValueError("LGSSM observations must have static time and observation dims")
    parameter_dim = target.initial_parameters.shape[0]
    if parameter_dim is None:
        raise ValueError("LGSSM parameter dimension must be static")

    static_shape = SSMStaticShape(
        horizon=int(horizon),
        state_dim=1,
        observation_dim=int(observation_dim),
        innovation_dim=1,
        parameter_dim=int(parameter_dim),
    )
    problem = BayesianSSMProblem(
        problem_id="lgssm-static-qr-generic-target-fixture",
        static_shape=static_shape,
        data_signature=SSMDataSignature(
            dataset_id="lgssm-static-qr-fixture-observations",
            observation_shape=tuple(observations.shape.as_list()),
            data_hash=data_hash,
        ),
        target_coordinate_convention="unconstrained",
        model_manifest={
            "model_id": "lgssm-static-qr-fixture",
            "model_hash": model_hash,
            "capabilities": (
                "linear_gaussian_transition",
                "linear_gaussian_observation",
                "exact_kalman_log_likelihood",
            ),
            "source_fixture": "bayesfilter.testing.tf_hmc_readiness.QRStaticLGSSMTarget",
        },
    )
    chart = ParameterChart(
        parameter_names=("rho_unconstrained", "log_measurement_noise"),
        unconstrained_dim=int(parameter_dim),
        constrained_shape=(int(parameter_dim),),
        transform_manifest={
            "transform_id": "identity-unconstrained-chart",
            "transform_hash": transform_hash,
            "coordinate_order": ("rho_unconstrained", "log_measurement_noise"),
            "source_convention": "QRStaticLGSSMTarget.model_and_derivatives",
        },
        log_jacobian_convention="not_included",
    )
    prior = ParameterPrior(
        prior_manifest={
            "prior_id": "zero-mean-independent-gaussian",
            "prior_hash": prior_hash,
            "parameter_order": chart.parameter_names,
            "prior_scale": tuple(float(value) for value in target.prior_scale.numpy()),
        },
        support_policy="unbounded",
        log_density_authority="graph_native",
    )
    filter_program = FilterProgram(
        filter_id="tf-qr-exact-lgssm-loglikelihood",
        required_model_capabilities=(
            "linear_gaussian_transition",
            "linear_gaussian_observation",
            "exact_kalman_log_likelihood",
        ),
        deterministic_target_policy="deterministic",
        approximation_semantics="exact",
        filter_manifest={
            "filter_id": "tf-qr-exact-lgssm-loglikelihood",
            "filter_hash": filter_hash,
            "backend": "tensorflow",
            "value_source": "bayesfilter.linear.kalman_qr_tf",
            "score_source": (
                "bayesfilter.linear.kalman_qr_derivatives_tf "
                "analytical QR Kalman score"
            ),
        },
    )
    return SSMTargetContract(
        problem=problem,
        chart=chart,
        prior=prior,
        filter_program=filter_program,
        frozen_transport=None,
    )


def lgssm_gaussian_prior_log_prob_and_grad(
    theta: Any,
    *,
    prior_scale: Any,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Return rank-2 Gaussian prior value and score for unconstrained params."""

    theta_tensor = _rank2_theta(theta)
    scale = tf.convert_to_tensor(prior_scale, dtype=theta_tensor.dtype)
    scaled = theta_tensor / scale
    value = -0.5 * tf.reduce_sum(tf.square(scaled), axis=-1)
    score = -theta_tensor / tf.square(scale)
    return value, score


def lgssm_qr_log_likelihood_and_grad(
    theta: Any,
    *,
    source_target: QRStaticLGSSMTarget | None = None,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Return rank-2 exact QR LGSSM likelihood value and analytical score."""

    theta_tensor = _rank2_theta(theta)
    target = QRStaticLGSSMTarget.default() if source_target is None else source_target
    values = []
    scores = []
    for row in tf.unstack(theta_tensor, axis=0):
        result = target.analytic_score_hessian(row)
        value = result.log_likelihood
        score = result.score
        values.append(value)
        scores.append(score)
    return tf.stack(values, axis=0), tf.stack(scores, axis=0)


def _rank2_theta(theta: Any) -> tf.Tensor:
    theta_tensor = tf.convert_to_tensor(theta, dtype=tf.float64)
    if theta_tensor.shape.rank != 2:
        raise ValueError("LGSSM generic target requires rank 2 theta [B, D]")
    parameter_dim = theta_tensor.shape[-1]
    if parameter_dim is None:
        raise ValueError("LGSSM generic target requires static parameter dimension")
    if int(parameter_dim) != 2:
        raise ValueError("LGSSM generic target parameter dimension must be 2")
    return theta_tensor
