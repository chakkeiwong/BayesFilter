"""Batch-native proper temperature bridges for NeuTra targets.

The primary bridge used by the q=20 experiment is

``log g_beta(theta) = log g0(theta) + beta * log L(theta)``.

The target supplies the prior and likelihood terms from one numerical
program.  This module only combines those terms; it never reconstructs a
likelihood by subtracting two independently rounded posterior evaluations.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import tensorflow as tf

from bayesfilter.inference.posterior_adapter import (
    ValueScoreCapability,
    value_score_capability,
)


TEMPERED_BRIDGE_SCHEMA = "bayesfilter.tempered.gaussian_likelihood_bridge.v1"
TEMPERED_BRIDGE_NONCLAIMS = (
    "proper bridge and value-score interface only",
    "finite properness receipt is not a mode-discovery guarantee",
    "bridge stress draws are not posterior samples",
    "no HMC convergence or posterior-correctness claim",
)


class TemperedBridgeError(ValueError):
    """Raised when a bridge contract or properness receipt is invalid."""


@dataclass(frozen=True)
class BridgePropernessReceipt:
    """Source-bound sufficient proof that every ladder law is proper."""

    target_signature: str
    bridge_id: str
    horizon: int
    observation_dim: int
    augmented_state_dim: int
    parameter_dim: int
    prior_variance: float
    log_prior_kernel_integral: float
    observation_variance: float
    sigma_rule: str
    sigma_alpha: float
    sigma_beta: float
    sigma_kappa: float
    covariance_weights_nonnegative: bool
    covariance_weights_match_unscented_formula: bool
    covariance_weight_sum: float
    covariance_weight_sum_tolerance: float
    gaussian_innovation_factorization: bool
    likelihood_strictly_positive: bool
    log_likelihood_upper_bound: float
    likelihood_upper_bound: float
    log_normalizer_upper_bound: float
    proof: str
    source_facts_hash: str
    nonclaims: tuple[str, ...] = TEMPERED_BRIDGE_NONCLAIMS

    def payload(self) -> Mapping[str, Any]:
        payload = asdict(self)
        payload["nonclaims"] = list(self.nonclaims)
        payload["schema"] = "bayesfilter.tempered.bridge_properness_receipt.v1"
        return payload


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _finite_float(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise TemperedBridgeError(f"{name} must be finite")
    return result


def build_q20_properness_receipt(
    source_facts: Mapping[str, Any],
    *,
    target_signature: str,
    bridge_id: str,
) -> BridgePropernessReceipt:
    """Validate and record the Gaussian-innovation properness argument.

    The q=20 filter has a finite horizon and Gaussian innovations.  Nonnegative
    unscented covariance weights make the propagated covariance positive
    semidefinite; adding the fixed observation variance ``R > 0`` gives
    ``S_t >= R``.  Thus every likelihood factor is strictly positive and
    bounded by ``(2*pi*R)**(-d/2)``.  The runtime prior is an *unnormalized*
    Gaussian kernel, so its finite integral is included explicitly in the
    normalizer bound rather than silently treated as one.
    """

    facts = dict(source_facts)
    declared_target_signature = facts.get("target_signature")
    if declared_target_signature is not None and str(declared_target_signature) != str(
        target_signature
    ):
        raise TemperedBridgeError(
            "properness facts target_signature does not match the bridge target"
        )
    horizon = int(facts.get("horizon", 0))
    observation_dim = int(facts.get("observation_dim", 0))
    augmented_state_dim = int(facts.get("augmented_state_dim", 0))
    parameter_dim = int(facts.get("parameter_dim", 0))
    if (
        horizon <= 0
        or observation_dim != 1
        or augmented_state_dim <= 0
        or parameter_dim <= 0
    ):
        raise TemperedBridgeError(
            "q20 properness facts require positive dimensions, finite horizon, and one observation coordinate"
        )
    prior_variance = _finite_float(facts.get("prior_variance"), "prior_variance")
    if prior_variance <= 0.0:
        raise TemperedBridgeError("properness requires positive prior variance")
    observation_variance = _finite_float(
        facts.get("observation_variance"), "observation_variance"
    )
    if observation_variance <= 0.0:
        raise TemperedBridgeError("properness requires strictly positive observation variance")
    sigma_rule = str(facts.get("sigma_rule", ""))
    if sigma_rule != "unscented":
        raise TemperedBridgeError("q20 properness receipt requires the unscented rule")
    sigma_alpha = _finite_float(facts.get("sigma_alpha"), "sigma_alpha")
    sigma_beta = _finite_float(facts.get("sigma_beta"), "sigma_beta")
    sigma_kappa = _finite_float(facts.get("sigma_kappa"), "sigma_kappa")
    if sigma_alpha <= 0.0:
        raise TemperedBridgeError("sigma_alpha must be positive")
    weights = facts.get("covariance_weights")
    if not isinstance(weights, (tuple, list)) or not weights:
        raise TemperedBridgeError("properness facts require covariance weights")
    weight_values = tuple(
        _finite_float(value, "covariance weight") for value in weights
    )
    expected_count = 2 * augmented_state_dim + 1
    if len(weight_values) != expected_count:
        raise TemperedBridgeError(
            "unscented covariance weight count must equal 2*augmented_state_dim+1"
        )
    sigma_lambda = sigma_alpha * sigma_alpha * (
        augmented_state_dim + sigma_kappa
    ) - augmented_state_dim
    sigma_denominator = augmented_state_dim + sigma_lambda
    if not math.isfinite(sigma_denominator) or sigma_denominator <= 0.0:
        raise TemperedBridgeError(
            "unscented sigma-point denominator must be finite and positive"
        )
    expected_center_weight = (
        sigma_lambda / sigma_denominator
        + 1.0
        - sigma_alpha * sigma_alpha
        + sigma_beta
    )
    expected_off_center_weight = 1.0 / (2.0 * sigma_denominator)
    expected_weights = (
        expected_center_weight,
        *(expected_off_center_weight for _ in range(expected_count - 1)),
    )
    computed_weight_sum = math.fsum(weight_values)
    weight_sum_tolerance = (
        64.0
        * max(1, len(weight_values))
        * 2.220446049250313e-16
        * max(1.0, abs(computed_weight_sum))
    )
    formula_tolerance = max(
        weight_sum_tolerance,
        64.0
        * 2.220446049250313e-16
        * max(1.0, *(abs(value) for value in expected_weights)),
    )
    formula_match = all(
        math.isclose(
            observed,
            expected,
            rel_tol=0.0,
            abs_tol=formula_tolerance,
        )
        for observed, expected in zip(weight_values, expected_weights, strict=True)
    )
    if not formula_match:
        raise TemperedBridgeError(
            "covariance weights do not match the declared unscented rule"
        )
    nonnegative = bool(
        facts.get("covariance_weights_nonnegative", False)
    ) and all(value >= 0.0 for value in weight_values)
    if not nonnegative:
        raise TemperedBridgeError(
            "covariance weights are not certified nonnegative"
        )
    weight_sum = _finite_float(
        facts.get("covariance_weight_sum", sum(weight_values)),
        "covariance_weight_sum",
    )
    if weight_sum < 0.0 or not math.isclose(
        weight_sum,
        computed_weight_sum,
        rel_tol=0.0,
        abs_tol=weight_sum_tolerance,
    ):
        raise TemperedBridgeError(
            "declared covariance weight sum does not match the source weights"
        )
    innovation_factorization = bool(facts.get("gaussian_innovation_factorization", False))
    strictly_positive = bool(facts.get("likelihood_strictly_positive", False))
    if not innovation_factorization or not strictly_positive:
        raise TemperedBridgeError(
            "properness facts must certify a strictly positive Gaussian innovation factorization"
        )

    log_factor_bound = -0.5 * observation_dim * math.log(
        2.0 * math.pi * observation_variance
    )
    log_likelihood_upper_bound = horizon * log_factor_bound
    log_prior_kernel_integral = 0.5 * parameter_dim * math.log(
        2.0 * math.pi * prior_variance
    )
    # For 0 <= beta <= 1, L(theta)^beta <= max(1,M).  Multiplication by the
    # unnormalized Gaussian prior kernel contributes its finite integral.
    log_normalizer_upper_bound = log_prior_kernel_integral + max(
        0.0, log_likelihood_upper_bound
    )
    try:
        likelihood_upper_bound = math.exp(log_likelihood_upper_bound)
    except OverflowError as exc:
        raise TemperedBridgeError(
            "likelihood bound exceeds host representable range"
        ) from exc
    facts_hash = _canonical_hash(facts)
    proof = (
        "finite-horizon Gaussian innovation factorization; nonnegative "
        "unscented covariance weights imply propagated covariance PSD; fixed "
        "observation variance R>0 implies S_t>=R; hence "
        "0<L(theta)<=M and 0<Z_beta<=A_prior*max(1,M)<infinity for "
        "beta in [0,1], where A_prior=(2*pi*prior_variance)^(parameter_dim/2)"
    )
    return BridgePropernessReceipt(
        target_signature=str(target_signature),
        bridge_id=str(bridge_id),
        horizon=horizon,
        observation_dim=observation_dim,
        augmented_state_dim=augmented_state_dim,
        parameter_dim=parameter_dim,
        prior_variance=prior_variance,
        log_prior_kernel_integral=log_prior_kernel_integral,
        observation_variance=observation_variance,
        sigma_rule=sigma_rule,
        sigma_alpha=sigma_alpha,
        sigma_beta=sigma_beta,
        sigma_kappa=sigma_kappa,
        covariance_weights_nonnegative=nonnegative,
        covariance_weights_match_unscented_formula=formula_match,
        covariance_weight_sum=weight_sum,
        covariance_weight_sum_tolerance=weight_sum_tolerance,
        gaussian_innovation_factorization=innovation_factorization,
        likelihood_strictly_positive=strictly_positive,
        log_likelihood_upper_bound=log_likelihood_upper_bound,
        likelihood_upper_bound=likelihood_upper_bound,
        log_normalizer_upper_bound=log_normalizer_upper_bound,
        proof=proof,
        source_facts_hash=facts_hash,
    )


class GaussianLikelihoodBridge:
    """A fixed, batch-native Gaussian-prior/likelihood temperature bridge."""

    def __init__(
        self,
        component_target: Any,
        *,
        prior_center: Any,
        prior_variance: float,
        source_facts: Mapping[str, Any],
        bridge_id: str = "gaussian_prior_likelihood_v1",
        jit_compile: bool = True,
    ) -> None:
        method = getattr(component_target, "batch_prior_likelihood_value_score_status", None)
        if not callable(method):
            raise TemperedBridgeError(
                "component_target must expose batch_prior_likelihood_value_score_status"
            )
        center = tf.convert_to_tensor(prior_center, tf.float64)
        if center.shape.rank != 1 or center.shape[0] is None:
            raise TemperedBridgeError("prior_center must be a static rank-1 tensor")
        variance = _finite_float(prior_variance, "prior_variance")
        if variance <= 0.0:
            raise TemperedBridgeError("prior_variance must be positive")
        target_signature_fn = getattr(component_target, "target_signature", None)
        if not callable(target_signature_fn):
            raise TemperedBridgeError("component_target must expose target_signature")
        self.component_target = component_target
        self.prior_center = center
        self.prior_variance = variance
        self.bridge_id = str(bridge_id)
        self.jit_compile = bool(jit_compile)
        self._target_signature = str(target_signature_fn())
        facts = dict(source_facts)
        declared_dim = facts.get("parameter_dim")
        if declared_dim is None or int(declared_dim) != int(center.shape[0]):
            raise TemperedBridgeError(
                "source facts parameter_dim must match prior_center"
            )
        declared_variance = facts.get("prior_variance")
        if declared_variance is None or not math.isclose(
            float(declared_variance), variance, rel_tol=0.0, abs_tol=0.0
        ):
            raise TemperedBridgeError(
                "source facts prior_variance must match bridge prior_variance"
            )
        declared_target_signature = facts.get("target_signature")
        if declared_target_signature is not None and str(
            declared_target_signature
        ) != self._target_signature:
            raise TemperedBridgeError(
                "source facts target_signature does not match component target"
            )
        facts["target_signature"] = self._target_signature
        self._source_facts = facts
        self._compiled: dict[int, Any] = {}
        self._receipt = build_q20_properness_receipt(
            self._source_facts,
            target_signature=self._target_signature,
            bridge_id=self.bridge_id,
        )
        self._signature = _canonical_hash(self.signature_payload())

    @property
    def parameter_dim(self) -> int:
        return int(self.prior_center.shape[0])

    @property
    def target_signature(self) -> str:
        return self._target_signature

    @property
    def signature(self) -> str:
        return self._signature

    @property
    def properness_receipt(self) -> BridgePropernessReceipt:
        return self._receipt

    def source_facts(self) -> Mapping[str, Any]:
        return dict(self._source_facts)

    def signature_payload(self) -> Mapping[str, Any]:
        return {
            "schema": TEMPERED_BRIDGE_SCHEMA,
            "bridge_id": self.bridge_id,
            "target_signature": self._target_signature,
            "prior_center": self.prior_center.numpy().tolist(),
            "prior_variance": self.prior_variance,
            "parameter_dim": self.parameter_dim,
            "source_facts_hash": _canonical_hash(self._source_facts),
            "jit_compile": self.jit_compile,
            "batch_native": True,
            "log_domain": True,
            "nonclaims": list(TEMPERED_BRIDGE_NONCLAIMS),
        }

    def component_terms(
        self, theta: Any
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, Mapping[str, tf.Tensor]]:
        """Return ``(log L, grad log L, log g0, grad log g0, status)``."""
        values = self._validate_theta(theta)
        return self.component_target.batch_prior_likelihood_value_score_status(values)

    def value_score_status(
        self, theta: Any, beta: Any
    ) -> tuple[tf.Tensor, tf.Tensor, Mapping[str, tf.Tensor]]:
        values = self._validate_theta(theta)
        beta_tensor = tf.convert_to_tensor(beta, tf.float64)
        if beta_tensor.shape.rank != 0:
            raise TemperedBridgeError("beta must be a scalar")
        size = int(values.shape[0])
        compiled = self._compiled.get(size)
        if compiled is None:
            compiled = tf.function(
                self._value_score_status_impl,
                input_signature=(
                    tf.TensorSpec([size, self.parameter_dim], tf.float64),
                    tf.TensorSpec([], tf.float64),
                ),
                jit_compile=self.jit_compile,
                reduce_retracing=False,
            )
            self._compiled[size] = compiled
        return compiled(values, beta_tensor)

    def log_prob_and_grad_status(
        self, theta: Any, beta: Any
    ) -> tuple[tf.Tensor, tf.Tensor, Mapping[str, tf.Tensor]]:
        return self.value_score_status(theta, beta)

    def fixed_beta_adapter(self, beta: float) -> "FixedBetaBridgeAdapter":
        """Bind one ladder level to the repository value/score interface."""
        return FixedBetaBridgeAdapter(self, beta=beta)

    def _validate_theta(self, theta: Any) -> tf.Tensor:
        values = tf.convert_to_tensor(theta, tf.float64)
        if values.shape.rank != 2 or values.shape[0] is None:
            raise TemperedBridgeError(
                f"theta must have static rank-2 shape [batch,{self.parameter_dim}]"
            )
        if values.shape[-1] != self.parameter_dim:
            raise TemperedBridgeError(
                f"theta must have shape [batch,{self.parameter_dim}]"
            )
        if int(values.shape[0]) <= 0:
            raise TemperedBridgeError("theta batch must be nonempty")
        return values

    def _value_score_status_impl(
        self, theta: tf.Tensor, beta: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor, Mapping[str, tf.Tensor]]:
        likelihood, likelihood_score, prior, prior_score, raw_status = (
            self.component_target.batch_prior_likelihood_value_score_status(theta)
        )
        beta_valid = tf.logical_and(
            tf.math.is_finite(beta),
            tf.logical_and(beta >= tf.constant(0.0, tf.float64), beta <= tf.constant(1.0, tf.float64)),
        )
        value = prior + beta * likelihood
        score = prior_score + beta * likelihood_score
        finite = tf.logical_and(
            tf.math.is_finite(value),
            tf.reduce_all(tf.math.is_finite(score), axis=1),
        )
        target_valid = tf.convert_to_tensor(
            raw_status["valid_pre_regularized_score"], tf.bool
        )
        valid = tf.logical_and(target_valid, tf.logical_and(beta_valid, finite))
        status = dict(raw_status)
        status["bridge_valid"] = valid
        status["bridge_beta_valid"] = tf.fill(tf.shape(value), beta_valid)
        status["status_code"] = tf.where(valid, 0, 1)
        status["valid_pre_regularized_score"] = valid
        status["value_finite"] = tf.math.is_finite(value)
        status["score_finite"] = tf.reduce_all(tf.math.is_finite(score), axis=1)
        return (
            tf.ensure_shape(value, [theta.shape[0]]),
            tf.ensure_shape(score, [theta.shape[0], self.parameter_dim]),
            status,
        )


class FixedBetaBridgeAdapter:
    """Immutable value/score adapter for one level of a proper bridge.

    HMC and the fixed-transport tuner consume a target with no free temperature
    argument.  This adapter binds that argument, carries the exact bridge and
    target identities into its signature, and converts any invalid numerical
    row to a non-finite target/score pair so a Metropolis proposal cannot be
    silently accepted outside the admitted numerical target.
    """

    target_status_invalid_rows_become_nonfinite = True
    supports_retained_flat_batch = True
    supports_retained_value_score_status = True

    def __init__(self, bridge: GaussianLikelihoodBridge, *, beta: float) -> None:
        if not isinstance(bridge, GaussianLikelihoodBridge):
            raise TemperedBridgeError(
                "fixed-beta adapter requires a GaussianLikelihoodBridge"
            )
        beta_value = _finite_float(beta, "beta")
        if not 0.0 <= beta_value <= 1.0:
            raise TemperedBridgeError("beta must lie in [0,1]")
        self.bridge = bridge
        self.beta = beta_value
        self.parameter_dim = bridge.parameter_dim
        self.target_scope = (
            f"{getattr(bridge.component_target, 'target_scope', bridge.target_signature)}"
            f":bridge={bridge.bridge_id}:beta={beta_value.hex()}"
        )
        names = getattr(bridge.component_target, "parameter_names", None)
        self.parameter_names = tuple(names) if names is not None else tuple(
            f"theta.{index}" for index in range(self.parameter_dim)
        )
        self._signature = _canonical_hash(self.adapter_signature_payload())

    def adapter_signature_payload(self) -> Mapping[str, Any]:
        return {
            "schema": "bayesfilter.tempered.fixed_beta_bridge_adapter.v1",
            "bridge_signature": self.bridge.signature,
            "bridge_id": self.bridge.bridge_id,
            "target_signature": self.bridge.target_signature,
            "target_scope": self.target_scope,
            "beta": self.beta,
            "parameter_dim": self.parameter_dim,
            "dtype": "float64",
            "batch_native": True,
            "invalid_rows_become_nonfinite": True,
            "nonclaims": list(TEMPERED_BRIDGE_NONCLAIMS),
        }

    def adapter_signature(self) -> str:
        return self._signature

    def value_score_capability(self) -> ValueScoreCapability:
        base = value_score_capability(self.bridge.component_target)
        xla_ready = bool(
            self.bridge.jit_compile and base.is_accepted_xla_hmc_authority
        )
        return ValueScoreCapability(
            value_score_authority=base.value_score_authority,
            xla_hmc_ready=xla_ready,
            full_chain_xla_diagnostic_ready=False,
            runtime_backend="bayesfilter.inference.tempered_target_tf",
            evidence_path=base.evidence_path,
            target_scope=self.target_scope,
            nonclaims=tuple(base.nonclaims) + TEMPERED_BRIDGE_NONCLAIMS,
        )

    def initial_position(self) -> tf.Tensor:
        return tf.identity(self.bridge.prior_center)

    def log_prob(self, theta: Any) -> tf.Tensor:
        return self.log_prob_and_grad(theta)[0]

    def log_prob_and_grad(self, theta: Any) -> tuple[tf.Tensor, tf.Tensor]:
        value, score, _status = self.log_prob_and_grad_status(theta)
        return value, score

    def log_prob_and_grad_batch(
        self, theta: Any
    ) -> tuple[tf.Tensor, tf.Tensor]:
        return self.log_prob_and_grad(theta)

    def log_prob_and_grad_status(
        self, theta: Any
    ) -> tuple[tf.Tensor, tf.Tensor, Mapping[str, tf.Tensor]]:
        value, score, status = self.bridge.value_score_status(theta, self.beta)
        valid = tf.convert_to_tensor(status["bridge_valid"], tf.bool)
        invalid_value = tf.fill(
            tf.shape(value), tf.constant(float("nan"), tf.float64)
        )
        invalid_score = tf.fill(
            tf.shape(score), tf.constant(float("nan"), tf.float64)
        )
        return (
            tf.where(valid, value, invalid_value),
            tf.where(valid[:, tf.newaxis], score, invalid_score),
            dict(status),
        )

    def target_status_telemetry(self, theta: Any) -> Mapping[str, tf.Tensor]:
        _value, _score, status = self.bridge.value_score_status(theta, self.beta)
        return dict(status)


def make_q20_tempered_bridge(
    q: int = 20,
    *,
    jit_compile: bool = True,
    principal_sqrt_backend: str = "compiled_custom_op",
) -> GaussianLikelihoodBridge:
    """Construct the source-bound q=20 Gaussian-prior bridge."""
    from bayesfilter.nonlinear.ssl_lstm_complexity_batched_target_tf import (
        batch_native_complexity_posterior_target,
    )

    target = batch_native_complexity_posterior_target(
        int(q),
        jit_compile=jit_compile,
        principal_sqrt_backend=principal_sqrt_backend,
    )
    facts = target.bridge_source_facts()
    return GaussianLikelihoodBridge(
        target,
        prior_center=target.config.prior_center,
        prior_variance=float(target.config.prior_standard_deviation) ** 2,
        source_facts=facts,
        jit_compile=jit_compile,
        bridge_id=f"ssl_lstm_q{int(q)}_gaussian_likelihood_bridge_v1",
    )


__all__ = [
    "BridgePropernessReceipt",
    "FixedBetaBridgeAdapter",
    "GaussianLikelihoodBridge",
    "TEMPERED_BRIDGE_NONCLAIMS",
    "TEMPERED_BRIDGE_SCHEMA",
    "TemperedBridgeError",
    "build_q20_properness_receipt",
    "make_q20_tempered_bridge",
]
