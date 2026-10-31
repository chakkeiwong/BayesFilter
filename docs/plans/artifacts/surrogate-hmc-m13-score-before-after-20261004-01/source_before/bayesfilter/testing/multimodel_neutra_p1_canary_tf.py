"""Complete analytic Gaussian posterior fixture for the P1 harness canary."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

import tensorflow as tf

from bayesfilter.inference.posterior_adapter import ValueScoreCapability
from bayesfilter.ssm import (
    BayesianSSMProblem,
    FilterProgram,
    ParameterChart,
    ParameterPrior,
    SSMDataSignature,
    SSMStaticShape,
    SSMTargetContract,
    stable_ssm_target_signature,
)


SYNTHETIC_CANARY_SCOPE = "P1-SYNTHETIC-GAUSSIAN"
SYNTHETIC_OBSERVATION = tf.constant((0.5, -0.25), tf.float64)
SYNTHETIC_PRIOR_VARIANCE = tf.constant((4.0, 4.0), tf.float64)
SYNTHETIC_LIKELIHOOD_VARIANCE = tf.constant((0.64, 1.21), tf.float64)


def synthetic_gaussian_prior_value_score(
    theta: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Independent prior term used by the P1 recomposition dossier."""

    values = tf.convert_to_tensor(theta, tf.float64)
    return (
        -0.5 * tf.reduce_sum(tf.square(values) / SYNTHETIC_PRIOR_VARIANCE, axis=-1),
        -values / SYNTHETIC_PRIOR_VARIANCE,
    )


def synthetic_gaussian_likelihood_value_score(
    theta: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Independent observation-likelihood term for the P1 dossier."""

    values = tf.convert_to_tensor(theta, tf.float64)
    residual = values - SYNTHETIC_OBSERVATION
    return (
        -0.5
        * tf.reduce_sum(
            tf.square(residual) / SYNTHETIC_LIKELIHOOD_VARIANCE, axis=-1
        ),
        -residual / SYNTHETIC_LIKELIHOOD_VARIANCE,
    )


def synthetic_exponential_chart_jacobian_value_score(
    theta: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Independent log-Jacobian for theta=log(constrained parameter)."""

    values = tf.convert_to_tensor(theta, tf.float64)
    return tf.reduce_sum(values, axis=-1), tf.ones_like(values)


def synthetic_gaussian_final_posterior_value_score(
    theta: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Production canary assembler, deliberately separate from dossier terms."""

    values = tf.convert_to_tensor(theta, tf.float64)
    residual = values - SYNTHETIC_OBSERVATION
    value = -0.5 * tf.reduce_sum(
        tf.square(values) / SYNTHETIC_PRIOR_VARIANCE
        + tf.square(residual) / SYNTHETIC_LIKELIHOOD_VARIANCE,
        axis=-1,
    ) + tf.reduce_sum(values, axis=-1)
    score = (
        -values / SYNTHETIC_PRIOR_VARIANCE
        - residual / SYNTHETIC_LIKELIHOOD_VARIANCE
        + tf.ones_like(values)
    )
    return value, score


class SyntheticGaussianCampaignAdapter:
    """Batch-native complete posterior used only to exercise the P1 harness."""

    parameter_dim = 2
    parameter_names = ("theta_0", "theta_1")
    dtype = tf.float64

    def __init__(self, contract: SSMTargetContract | None = None) -> None:
        self.contract = make_synthetic_gaussian_contract() if contract is None else contract
        payload = {
            "schema": "bayesfilter.testing.multimodel_neutra_p1_canary_adapter.v1",
            "mathematical_target_signature": stable_ssm_target_signature(self.contract),
            "dtype": self.dtype.name,
            "parameter_names": self.parameter_names,
            "route": "synthetic_gaussian_final_posterior_value_score",
        }
        self._adapter_signature = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def adapter_signature(self) -> str:
        return self._adapter_signature

    def value_score_capability(self) -> ValueScoreCapability:
        return ValueScoreCapability(
            value_score_authority="graph_native",
            xla_hmc_ready=True,
            full_chain_xla_diagnostic_ready=True,
            runtime_backend="tensorflow_synthetic_gaussian_p1_canary",
            evidence_path="bayesfilter/testing/multimodel_neutra_p1_canary_tf.py",
            target_scope=SYNTHETIC_CANARY_SCOPE,
            nonclaims=(
                "synthetic P1 harness canary only",
                "no nonlinear model or training quality claim",
            ),
        )

    def log_prob(self, theta: Any) -> tf.Tensor:
        return synthetic_gaussian_final_posterior_value_score(theta)[0]

    def log_prob_and_grad(self, theta: Any) -> tuple[tf.Tensor, tf.Tensor]:
        return synthetic_gaussian_final_posterior_value_score(theta)

    def neutra_batch_log_prob_and_grad_status(
        self, theta: Any
    ) -> tuple[tf.Tensor, tf.Tensor, Mapping[str, tf.Tensor]]:
        values = tf.convert_to_tensor(theta, tf.float64)
        value, score = synthetic_gaussian_final_posterior_value_score(values)
        leading = tf.shape(values)[:-1]
        return value, score, {
            "status_code": tf.zeros(leading, tf.int32),
            "valid_pre_regularized_score": tf.ones(leading, tf.bool),
            "floor_count_value": tf.zeros(leading, tf.int32),
            "min_innovation_eigenvalue": tf.ones(leading, tf.float64),
            "innovation_condition_estimate": tf.ones(leading, tf.float64),
        }

    def target_status_telemetry(self, theta: Any) -> Mapping[str, tf.Tensor]:
        values = tf.convert_to_tensor(theta, tf.float64)
        leading = tf.shape(values)[:-1]
        return {
            "status_code": tf.zeros(leading, tf.int32),
            "valid_pre_regularized_score": tf.ones(leading, tf.bool),
        }


def make_synthetic_gaussian_contract(
    *,
    data_hash: str = "sha256:p1-synthetic-observation-v1",
    prior_hash: str = "sha256:p1-synthetic-prior-v1",
    filter_hash: str = "sha256:p1-synthetic-gaussian-likelihood-v1",
    transform_hash: str = "sha256:p1-synthetic-exponential-chart-v1",
    model_hash: str = "sha256:p1-synthetic-gaussian-model-v1",
) -> SSMTargetContract:
    """Construct the complete mathematical metadata for the synthetic target."""

    shape = SSMStaticShape(
        horizon=1,
        state_dim=2,
        observation_dim=2,
        innovation_dim=2,
        parameter_dim=2,
    )
    problem = BayesianSSMProblem(
        problem_id="p1-synthetic-gaussian-campaign-canary",
        static_shape=shape,
        data_signature=SSMDataSignature(
            dataset_id="p1-synthetic-gaussian-observation",
            observation_shape=(1, 2),
            data_hash=data_hash,
        ),
        target_coordinate_convention="unconstrained",
        model_manifest={
            "model_id": "p1-synthetic-gaussian",
            "model_hash": model_hash,
            "capabilities": ("analytic_gaussian_likelihood",),
        },
    )
    chart = ParameterChart(
        parameter_names=("theta_0", "theta_1"),
        unconstrained_dim=2,
        constrained_shape=(2,),
        transform_manifest={
            "transform_id": "elementwise-exponential-chart",
            "transform_hash": transform_hash,
        },
        log_jacobian_convention="included_in_chart",
    )
    prior = ParameterPrior(
        prior_manifest={
            "prior_id": "independent-zero-mean-gaussian",
            "prior_hash": prior_hash,
        },
        support_policy="unbounded",
        log_density_authority="graph_native",
    )
    filter_program = FilterProgram(
        filter_id="p1-synthetic-analytic-gaussian-likelihood",
        required_model_capabilities=("analytic_gaussian_likelihood",),
        deterministic_target_policy="deterministic",
        approximation_semantics="exact",
        filter_manifest={
            "filter_id": "p1-synthetic-analytic-gaussian-likelihood",
            "filter_hash": filter_hash,
            "backend": "tensorflow",
        },
    )
    return SSMTargetContract(
        problem=problem,
        chart=chart,
        prior=prior,
        filter_program=filter_program,
        frozen_transport=None,
    )
