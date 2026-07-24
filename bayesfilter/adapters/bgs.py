"""BayesFilter-owned TensorFlow/TFP posterior adapter for the BGS target."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from typing import Any, NamedTuple

import tensorflow as tf
import tensorflow_probability as tfp

from bayesfilter.inference.posterior_adapter import ValueScoreCapability


DTYPE = tf.float64
PARAMETER_DIMENSION = 46
PARAMETER_NAMES = (
    "e_z", "e_u", "e_g", "e_i", "e_r", "e_p", "e_w", "e_lk",
    "e_qe_b", "e_qe_k", "e_cbl", "sig_c", "sig_l", "tpr_beta", "h",
    "phiss", "i_p", "i_w", "alpha", "zeta_p", "Phi_p", "psi",
    "phi_pi", "phi_y", "phi_dy", "phi_sprd", "rho", "rho_r", "rho_p",
    "rho_u", "rho_lk", "rho_cbl", "rootb1", "rootb2", "rootk1",
    "rootk2", "mu_p", "mu_w", "kap_tau", "pac", "theta", "LEV",
    "lamb_cbl", "trend", "mean_Pi", "mean_l",
)
PRIOR_FAMILIES = (
    "INV_GAMMA_PDF", "INV_GAMMA_PDF", "INV_GAMMA_PDF", "INV_GAMMA_PDF",
    "INV_GAMMA_PDF", "INV_GAMMA_PDF", "INV_GAMMA_PDF", "INV_GAMMA_PDF",
    "INV_GAMMA_PDF", "INV_GAMMA_PDF", "INV_GAMMA_PDF", "NORMAL_PDF",
    "NORMAL_PDF", "GAMMA_PDF", "BETA_PDF", "NORMAL_PDF", "BETA_PDF",
    "BETA_PDF", "NORMAL_PDF", "BETA_PDF", "NORMAL_PDF", "BETA_PDF",
    "NORMAL_PDF", "NORMAL_PDF", "NORMAL_PDF", "NORMAL_PDF", "BETA_PDF",
    "BETA_PDF", "BETA_PDF", "BETA_PDF", "BETA_PDF", "BETA_PDF",
    "BETA_PDF", "BETA_PDF", "BETA_PDF", "BETA_PDF", "BETA_PDF",
    "BETA_PDF", "GAMMA_PDF", "GAMMA_PDF", "BETA_PDF", "NORMAL_PDF",
    "GAMMA_PDF", "NORMAL_PDF", "GAMMA_PDF", "NORMAL_PDF",
)
PRIOR_LOWER = tf.constant((
    0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001,
    0.0001, 0.0001, 0.0001, 0.25, 0.25, 0.01, 0.001, 0.5, 0.01, 0.01,
    0.01, 0.01, 1.0, 0.01, 1.0, 0.001, 0.001, 0.001, 0.01, 0.01,
    0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.001, 0.001,
    0.001, 0.000001, 0.5, 0.5, 0.1, 0.01, 0.001, -10.0,
), DTYPE)
PRIOR_UPPER = tf.constant((
    100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0,
    100.0, 100.0, 5.0, 10.0, 2.0, 0.99, 15.0, 0.99, 0.99, 0.99, 0.99,
    3.0, 0.99, 3.0, 0.5, 0.5, 0.5, 0.999, 0.999, 0.999, 0.999,
    0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 2.0, 20.0,
    0.98, 10.0, 20.0, 2.0, 2.0, 10.0,
), DTYPE)
PRIOR_MEAN = tf.constant((
    0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.5, 2.0,
    0.25, 0.7, 4.0, 0.5, 0.5, 0.3, 0.5, 1.25, 0.5, 1.5, 0.125,
    0.125, 0.125, 0.75, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
    0.5, 0.5, 0.3, 2.0, 0.95, 3.0, 3.0, 0.44, 0.625, 0.0,
), DTYPE)
PRIOR_SD = tf.constant((
    0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
    0.375, 0.75, 0.1, 0.1, 1.5, 0.15, 0.15, 0.05, 0.1, 0.125, 0.15,
    0.25, 0.05, 0.05, 0.05, 0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
    0.2, 0.2, 0.2, 0.2, 0.1, 4.0, 0.05, 1.0, 3.0, 0.05, 0.1, 2.0,
), DTYPE)
SOURCE_HASHES = {
    "driver": "7ed8d84cb87bd5d6a2b2c4b3e612df5bea40d346cdea4cf79f387ab9e7b28a97",
    "mod": "e433ad5c8d7f1769f839996718ae608cb911bc3f781988de30bb231a6497c16d",
    "order": "333c36c9bcc6727bfd6f6e3ecfba5e670096f719eb77106c72909f2811e2ae8b",
}
_NORMAL = tf.constant([family == "NORMAL_PDF" for family in PRIOR_FAMILIES])
_BETA = tf.constant([family == "BETA_PDF" for family in PRIOR_FAMILIES])
_GAMMA = tf.constant([family == "GAMMA_PDF" for family in PRIOR_FAMILIES])
_INV_GAMMA = tf.constant(
    [family == "INV_GAMMA_PDF" for family in PRIOR_FAMILIES]
)
_TFD = tfp.distributions

BGS_POSTERIOR_DEBUG_NONCLAIMS = (
    "CPU graph parity adapter only",
    "no end-to-end XLA authority",
    "no HMC tuning or sampling claim",
    "no posterior convergence claim",
    "no GPU readiness claim",
    "no default-readiness claim",
)
BGS_POSTERIOR_XLA_NONCLAIMS = (
    "D296 synthetic-data target only",
    "no HMC tuning claim",
    "no posterior convergence claim",
    "no efficiency or superiority claim",
    "no production or default-readiness claim",
)
BGS_POSTERIOR_NONCLAIMS = BGS_POSTERIOR_DEBUG_NONCLAIMS
_CAPABILITY_MODES = {"debug_graph", "target_xla_graph_chain"}
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")

BGS_STATUS_NONFINITE_UNCONSTRAINED = 1
BGS_STATUS_TRANSFORM_OUTSIDE_OPEN_SUPPORT = 2
BGS_STATUS_PRIOR_OR_JACOBIAN_NONFINITE = 4
BGS_STATUS_DESCRIPTOR_FAILURE = 8
BGS_STATUS_STATE_SPACE_FAILURE = 16
BGS_STATUS_LIKELIHOOD_VALUE_NONFINITE = 32
BGS_STATUS_LIKELIHOOD_SCORE_NONFINITE = 64
BGS_STATUS_POSTERIOR_NONFINITE = 128


class BGSConstrainedLikelihoodResult(NamedTuple):
    signed_log_likelihood: tf.Tensor
    signed_score: tf.Tensor
    descriptor_success: Any = True
    numerical_state_space_success: Any = True
    likelihood_value_finite: Any = True
    likelihood_score_finite: Any = True


class BGSPosteriorComponents(NamedTuple):
    theta: tf.Tensor
    signed_log_likelihood: tf.Tensor
    constrained_log_prior: tf.Tensor
    log_abs_det_jacobian: tf.Tensor
    posterior_value: tf.Tensor
    posterior_score: tf.Tensor
    finite_unconstrained: tf.Tensor
    transform_in_open_support: tf.Tensor
    prior_and_jacobian_finite: tf.Tensor
    descriptor_success: tf.Tensor
    numerical_state_space_success: tf.Tensor
    likelihood_value_finite: tf.Tensor
    likelihood_score_finite: tf.Tensor
    composed_posterior_finite: tf.Tensor
    status_code: tf.Tensor
    valid: tf.Tensor


def theta_from_unconstrained(u: Any) -> tf.Tensor:
    values = tf.ensure_shape(
        tf.cast(tf.convert_to_tensor(u), DTYPE), (PARAMETER_DIMENSION,)
    )
    return PRIOR_LOWER + (PRIOR_UPPER - PRIOR_LOWER) * tf.math.sigmoid(values)


def unconstrained_from_theta(theta: Any) -> tf.Tensor:
    values = tf.ensure_shape(
        tf.cast(tf.convert_to_tensor(theta), DTYPE), (PARAMETER_DIMENSION,)
    )
    proportions = (values - PRIOR_LOWER) / (PRIOR_UPPER - PRIOR_LOWER)
    support = tf.reduce_all(
        tf.logical_and(proportions > 0.0, proportions < 1.0)
    )
    check = tf.debugging.assert_equal(
        support, True, message="theta is outside the open BGS support"
    )
    with tf.control_dependencies((check,)):
        return tf.math.log(proportions) - tf.math.log1p(-proportions)


def log_abs_det_jacobian(u: Any) -> tf.Tensor:
    values = tf.ensure_shape(
        tf.cast(tf.convert_to_tensor(u), DTYPE), (PARAMETER_DIMENSION,)
    )
    return tf.reduce_sum(
        tf.math.log(PRIOR_UPPER - PRIOR_LOWER)
        + tf.math.log_sigmoid(values)
        + tf.math.log_sigmoid(-values)
    )


def constrained_log_prior_and_score(theta: Any) -> tuple[tf.Tensor, tf.Tensor]:
    values = tf.ensure_shape(
        tf.cast(tf.convert_to_tensor(theta), DTYPE), (PARAMETER_DIMENSION,)
    )
    width = PRIOR_UPPER - PRIOR_LOWER
    z = (values - PRIOR_LOWER) / width
    beta_variance = tf.where(
        _BETA, tf.square(PRIOR_SD / width), tf.ones_like(PRIOR_SD)
    )
    beta_mean = tf.where(
        _BETA,
        (PRIOR_MEAN - PRIOR_LOWER) / width,
        tf.fill((PARAMETER_DIMENSION,), tf.constant(0.5, DTYPE)),
    )
    beta_concentration = beta_mean * (1.0 - beta_mean) / beta_variance - 1.0
    beta_a = beta_mean * beta_concentration
    beta_b = (1.0 - beta_mean) * beta_concentration
    gamma_mean = tf.where(_GAMMA, PRIOR_MEAN, tf.ones_like(PRIOR_MEAN))
    gamma_sd = tf.where(_GAMMA, PRIOR_SD, tf.ones_like(PRIOR_SD))
    gamma_shape = tf.square(gamma_mean / gamma_sd)
    gamma_scale = tf.square(gamma_sd) / gamma_mean
    inv_mean = tf.where(_INV_GAMMA, PRIOR_MEAN, tf.ones_like(PRIOR_MEAN))
    inv_sd = tf.where(_INV_GAMMA, PRIOR_SD, tf.ones_like(PRIOR_SD))
    inv_shape = 2.0 + tf.square(inv_mean / inv_sd)
    inv_scale = inv_mean * (inv_shape - 1.0)

    beta_z = tf.where(
        _BETA,
        z,
        tf.fill((PARAMETER_DIMENSION,), tf.constant(0.5, DTYPE)),
    )
    gamma_values = tf.where(_GAMMA, values, tf.ones_like(values))
    inv_gamma_values = tf.where(_INV_GAMMA, values, tf.ones_like(values))

    normal_log = _TFD.Normal(PRIOR_MEAN, PRIOR_SD).log_prob(values)
    beta_log = _TFD.Beta(beta_a, beta_b).log_prob(beta_z) - tf.math.log(width)
    gamma_log = _TFD.Gamma(
        gamma_shape, rate=1.0 / gamma_scale
    ).log_prob(gamma_values)
    inv_gamma_log = _TFD.InverseGamma(
        inv_shape, scale=inv_scale
    ).log_prob(inv_gamma_values)
    contributions = tf.where(
        _NORMAL,
        normal_log,
        tf.where(_BETA, beta_log, tf.where(_GAMMA, gamma_log, inv_gamma_log)),
    )

    normal_score = -(values - PRIOR_MEAN) / tf.square(PRIOR_SD)
    beta_score = (
        (beta_a - 1.0) / beta_z - (beta_b - 1.0) / (1.0 - beta_z)
    ) / width
    gamma_score = (gamma_shape - 1.0) / gamma_values - 1.0 / gamma_scale
    inv_gamma_score = (
        -(inv_shape + 1.0) / inv_gamma_values
        + inv_scale / tf.square(inv_gamma_values)
    )
    score = tf.where(
        _NORMAL,
        normal_score,
        tf.where(_BETA, beta_score, tf.where(_GAMMA, gamma_score, inv_gamma_score)),
    )
    support = tf.reduce_all(
        tf.logical_and(values > PRIOR_LOWER, values < PRIOR_UPPER)
    )
    value = tf.where(
        support,
        tf.reduce_sum(contributions),
        tf.constant(float("-inf"), DTYPE),
    )
    score = tf.where(
        support,
        score,
        tf.fill((PARAMETER_DIMENSION,), tf.constant(float("nan"), DTYPE)),
    )
    return value, score


class BGSPosteriorAdapter:
    """Scalar transformed posterior adapter with conservative capability."""

    def __init__(
        self,
        constrained_likelihood_value_score: Callable[[tf.Tensor], Any],
        *,
        evidence_path: str,
        capability_mode: str = "debug_graph",
        likelihood_signature: str | None = None,
    ) -> None:
        if not callable(constrained_likelihood_value_score):
            raise TypeError("constrained_likelihood_value_score must be callable")
        self._likelihood = constrained_likelihood_value_score
        self._evidence_path = str(evidence_path)
        mode = str(capability_mode)
        if mode not in _CAPABILITY_MODES:
            raise ValueError(
                "capability_mode must be 'debug_graph' or "
                "'target_xla_graph_chain'"
            )
        signature = None if likelihood_signature is None else str(likelihood_signature)
        if mode == "target_xla_graph_chain" and (
            signature is None or _SHA256_PATTERN.fullmatch(signature) is None
        ):
            raise ValueError(
                "target_xla_graph_chain mode requires a lowercase SHA-256 "
                "likelihood_signature"
            )
        self._capability_mode = mode
        self._likelihood_signature = signature
        self.supports_retained_value_score_status = True
        runtime_backend = (
            "tensorflow_tfp_bayesfilter_qr_target_xla_graph_chain"
            if mode == "target_xla_graph_chain"
            else "tensorflow_tfp_bayesfilter_qr_graph_debug"
        )
        payload = {
            "schema": "bayesfilter.bgs.posterior_adapter.v1",
            "parameter_names": PARAMETER_NAMES,
            "source_hashes": SOURCE_HASHES,
            "target_scope": "bgs_d296_synthetic_transformed_target",
            "runtime_backend": runtime_backend,
            "capability_mode": mode,
            "likelihood_signature": signature,
        }
        self._signature = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
        ).hexdigest()

    @property
    def parameter_dim(self) -> int:
        return PARAMETER_DIMENSION

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return PARAMETER_NAMES

    def adapter_signature(self) -> str:
        return self._signature

    @property
    def capability_mode(self) -> str:
        return self._capability_mode

    def value_score_capability(self) -> ValueScoreCapability:
        if self._capability_mode == "target_xla_graph_chain":
            return ValueScoreCapability(
                value_score_authority="reviewed_gradient_tape_xla_exception",
                xla_hmc_ready=False,
                full_chain_xla_diagnostic_ready=False,
                runtime_backend="tensorflow_tfp_bayesfilter_qr_target_xla_graph_chain",
                evidence_path=self._evidence_path,
                target_scope="bgs_d296_synthetic_transformed_target",
                nonclaims=BGS_POSTERIOR_XLA_NONCLAIMS,
            )
        return ValueScoreCapability(
            value_score_authority="debug_only",
            xla_hmc_ready=False,
            full_chain_xla_diagnostic_ready=False,
            runtime_backend="tensorflow_tfp_bayesfilter_qr_graph_debug",
            evidence_path=self._evidence_path,
            target_scope="bgs_d296_synthetic_transformed_target",
            nonclaims=BGS_POSTERIOR_DEBUG_NONCLAIMS,
        )

    def components(self, u: Any) -> BGSPosteriorComponents:
        values = tf.ensure_shape(
            tf.cast(tf.convert_to_tensor(u), DTYPE), (PARAMETER_DIMENSION,)
        )
        theta = theta_from_unconstrained(values)
        prior_value, prior_score = constrained_log_prior_and_score(theta)
        jacobian = log_abs_det_jacobian(values)
        finite_unconstrained = tf.reduce_all(tf.math.is_finite(values))
        transform_in_open_support = tf.reduce_all(tf.logical_and(
            tf.math.is_finite(theta),
            tf.logical_and(theta > PRIOR_LOWER, theta < PRIOR_UPPER),
        ))
        prior_and_jacobian_finite = tf.reduce_all(tf.stack((
            tf.math.is_finite(prior_value),
            tf.reduce_all(tf.math.is_finite(prior_score)),
            tf.math.is_finite(jacobian),
        )))
        evaluate_likelihood = tf.reduce_all(tf.stack((
            finite_unconstrained,
            transform_in_open_support,
            prior_and_jacobian_finite,
        )))

        def target_branch():
            likelihood = self._likelihood(theta)
            return (
                tf.convert_to_tensor(likelihood.signed_log_likelihood, DTYPE),
                tf.ensure_shape(
                    tf.convert_to_tensor(likelihood.signed_score, DTYPE),
                    (PARAMETER_DIMENSION,),
                ),
                tf.convert_to_tensor(likelihood.descriptor_success, tf.bool),
                tf.convert_to_tensor(
                    likelihood.numerical_state_space_success, tf.bool
                ),
                tf.convert_to_tensor(likelihood.likelihood_value_finite, tf.bool),
                tf.convert_to_tensor(likelihood.likelihood_score_finite, tf.bool),
            )

        def skipped_target_branch():
            return (
                tf.constant(0.0, DTYPE),
                tf.zeros((PARAMETER_DIMENSION,), DTYPE),
                tf.constant(True),
                tf.constant(True),
                tf.constant(True),
                tf.constant(True),
            )

        (
            likelihood_value,
            likelihood_score,
            descriptor_success,
            numerical_state_space_success,
            reported_likelihood_value_finite,
            reported_likelihood_score_finite,
        ) = tf.cond(evaluate_likelihood, target_branch, skipped_target_branch)
        likelihood_value_finite = tf.logical_and(
            reported_likelihood_value_finite,
            tf.math.is_finite(likelihood_value),
        )
        likelihood_score_finite = tf.logical_and(
            reported_likelihood_score_finite,
            tf.reduce_all(tf.math.is_finite(likelihood_score)),
        )
        sigmoid = tf.math.sigmoid(values)
        dtheta_du = (PRIOR_UPPER - PRIOR_LOWER) * sigmoid * (1.0 - sigmoid)
        raw_score = (
            (likelihood_score + prior_score) * dtheta_du + 1.0 - 2.0 * sigmoid
        )
        raw_value = likelihood_value + prior_value + jacobian
        composed_posterior_finite = tf.logical_and(
            tf.math.is_finite(raw_value),
            tf.reduce_all(tf.math.is_finite(raw_score)),
        )
        status_code = tf.constant(0, tf.int32)
        status_code += tf.where(
            finite_unconstrained,
            0,
            BGS_STATUS_NONFINITE_UNCONSTRAINED,
        )
        status_code += tf.where(
            transform_in_open_support,
            0,
            BGS_STATUS_TRANSFORM_OUTSIDE_OPEN_SUPPORT,
        )
        status_code += tf.where(
            prior_and_jacobian_finite,
            0,
            BGS_STATUS_PRIOR_OR_JACOBIAN_NONFINITE,
        )
        status_code += tf.where(
            descriptor_success,
            0,
            BGS_STATUS_DESCRIPTOR_FAILURE,
        )
        status_code += tf.where(
            numerical_state_space_success,
            0,
            BGS_STATUS_STATE_SPACE_FAILURE,
        )
        status_code += tf.where(
            likelihood_value_finite,
            0,
            BGS_STATUS_LIKELIHOOD_VALUE_NONFINITE,
        )
        status_code += tf.where(
            likelihood_score_finite,
            0,
            BGS_STATUS_LIKELIHOOD_SCORE_NONFINITE,
        )
        status_code += tf.where(
            composed_posterior_finite,
            0,
            BGS_STATUS_POSTERIOR_NONFINITE,
        )
        valid = status_code == tf.constant(0, tf.int32)
        value = tf.where(valid, raw_value, tf.constant(float("-inf"), DTYPE))
        # A zero invalid-point score is a rejection convention, not a derivative.
        score = tf.where(valid, raw_score, tf.zeros_like(raw_score))
        return BGSPosteriorComponents(
            theta,
            likelihood_value,
            prior_value,
            jacobian,
            value,
            score,
            finite_unconstrained,
            transform_in_open_support,
            prior_and_jacobian_finite,
            descriptor_success,
            numerical_state_space_success,
            likelihood_value_finite,
            likelihood_score_finite,
            composed_posterior_finite,
            status_code,
            valid,
        )

    def log_prob(self, u: Any) -> tf.Tensor:
        return self.components(u).posterior_value

    def log_prob_and_grad(self, u: Any) -> tuple[tf.Tensor, tf.Tensor]:
        components = self.components(u)
        return components.posterior_value, components.posterior_score

    def log_prob_and_grad_status(
        self, u: Any
    ) -> tuple[tf.Tensor, tf.Tensor, dict[str, tf.Tensor]]:
        components = self.components(u)
        return (
            components.posterior_value,
            components.posterior_score,
            self._status_telemetry(components),
        )

    def target_status_telemetry(self, u: Any) -> dict[str, tf.Tensor]:
        return self._status_telemetry(self.components(u))

    @staticmethod
    def _status_telemetry(
        components: BGSPosteriorComponents,
    ) -> dict[str, tf.Tensor]:
        zero = tf.constant(0.0, DTYPE)
        unavailable = tf.constant(float("nan"), DTYPE)
        innovation_sentinel = tf.where(components.valid, zero, unavailable)
        return {
            "status_code": components.status_code,
            "valid_pre_regularized_score": components.valid,
            "floor_count_value": tf.constant(0, tf.int32),
            "min_innovation_eigenvalue": innovation_sentinel,
            "innovation_condition_estimate": innovation_sentinel,
            "innovation_metrics_available": tf.constant(False),
            "finite_unconstrained": components.finite_unconstrained,
            "transform_in_open_support": components.transform_in_open_support,
            "prior_and_jacobian_finite": components.prior_and_jacobian_finite,
            "descriptor_success": components.descriptor_success,
            "numerical_state_space_success": (
                components.numerical_state_space_success
            ),
            "likelihood_value_finite": components.likelihood_value_finite,
            "likelihood_score_finite": components.likelihood_score_finite,
            "composed_posterior_finite": components.composed_posterior_finite,
        }


__all__ = [
    "BGSConstrainedLikelihoodResult",
    "BGSPosteriorAdapter",
    "BGSPosteriorComponents",
    "BGS_POSTERIOR_NONCLAIMS",
    "BGS_POSTERIOR_DEBUG_NONCLAIMS",
    "BGS_POSTERIOR_XLA_NONCLAIMS",
    "BGS_STATUS_DESCRIPTOR_FAILURE",
    "BGS_STATUS_LIKELIHOOD_SCORE_NONFINITE",
    "BGS_STATUS_LIKELIHOOD_VALUE_NONFINITE",
    "BGS_STATUS_NONFINITE_UNCONSTRAINED",
    "BGS_STATUS_POSTERIOR_NONFINITE",
    "BGS_STATUS_PRIOR_OR_JACOBIAN_NONFINITE",
    "BGS_STATUS_STATE_SPACE_FAILURE",
    "BGS_STATUS_TRANSFORM_OUTSIDE_OPEN_SUPPORT",
    "PARAMETER_DIMENSION",
    "PARAMETER_NAMES",
    "PRIOR_FAMILIES",
    "PRIOR_LOWER",
    "PRIOR_MEAN",
    "PRIOR_SD",
    "PRIOR_UPPER",
    "SOURCE_HASHES",
    "constrained_log_prior_and_score",
    "log_abs_det_jacobian",
    "theta_from_unconstrained",
    "unconstrained_from_theta",
]
