"""Batch-native posterior target builder for generic Bayesian SSM contracts."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

import tensorflow as tf

from bayesfilter.inference.posterior_adapter import ValueScoreCapability
from bayesfilter.ssm.contracts import (
    SSMTargetContract,
    stable_ssm_target_signature,
    validate_ssm_target_contract,
)


BatchRankPolicy = Literal["rank2_required"]
BatchValueScoreFn = Callable[[tf.Tensor], tuple[Any, Any]]

TARGET_BUILDER_NONCLAIMS = (
    "generic SSM posterior target builder only",
    "no HMC tuning or sampling claim",
    "no XLA HMC readiness claim",
    "no NeuTra training claim",
    "no posterior convergence claim",
    "no scientific validity claim",
)

XLA_TARGET_BUILDER_NONCLAIMS = (
    "generic SSM posterior target builder XLA-HMC value/score opt-in only",
    "no HMC tuning or sampling claim",
    "no full-chain XLA diagnostic claim",
    "no posterior convergence claim",
    "no scientific validity claim",
)

_KNOWN_BATCH_RANK_POLICIES = {"rank2_required"}
_PROCESS_LOCAL_SIGNATURE_PATTERNS = (
    re.compile(r"\b0x[0-9a-fA-F]+\b"),
    re.compile(r"\bobject at\b"),
    re.compile(r"\bid\s*\("),
)


class InvalidSSMTargetBuilderContract(ValueError):
    """Raised when a target-builder boundary is incomplete or unsafe."""


@dataclass(frozen=True)
class SSMTargetBuilderMetadata:
    """Stable metadata for a generic SSM posterior adapter."""

    schema: str
    target_signature: str
    adapter_signature: str
    dtype: str
    parameter_dim: int
    parameter_names: tuple[str, ...]
    batch_rank_policy: BatchRankPolicy
    non_batch_static_shape: Mapping[str, int]
    value_score_authority: str
    runtime_backend: str
    target_scope: str
    nonclaims: tuple[str, ...] = TARGET_BUILDER_NONCLAIMS

    def manifest_payload(self) -> Mapping[str, Any]:
        return {
            "schema": self.schema,
            "target_signature": self.target_signature,
            "adapter_signature": self.adapter_signature,
            "dtype": self.dtype,
            "parameter_dim": self.parameter_dim,
            "parameter_names": self.parameter_names,
            "batch_rank_policy": self.batch_rank_policy,
            "non_batch_static_shape": self.non_batch_static_shape,
            "value_score_authority": self.value_score_authority,
            "runtime_backend": self.runtime_backend,
            "target_scope": self.target_scope,
            "nonclaims": self.nonclaims,
        }


class GenericSSMPosteriorAdapter:
    """Batch-native value/score adapter assembled from generic SSM contracts."""

    def __init__(
        self,
        *,
        contract: SSMTargetContract,
        prior_log_prob_and_grad: BatchValueScoreFn,
        filter_log_likelihood_and_grad: BatchValueScoreFn,
        dtype: Any = tf.float64,
        batch_rank_policy: BatchRankPolicy = "rank2_required",
        target_scope: str | None = None,
        evidence_path: str | None = None,
        runtime_backend: str = "tensorflow",
        xla_hmc_ready: bool = False,
        full_chain_xla_diagnostic_ready: bool = False,
        nonclaims: tuple[str, ...] = TARGET_BUILDER_NONCLAIMS,
    ) -> None:
        validated = validate_ssm_target_contract(
            contract,
            require_filter_hmc_target_ready=True,
        )
        if validated.frozen_transport is not None:
            raise InvalidSSMTargetBuilderContract(
                "Phase 2 target builder expects an untransported target contract"
            )
        if not callable(prior_log_prob_and_grad):
            raise TypeError("prior_log_prob_and_grad must be callable")
        if not callable(filter_log_likelihood_and_grad):
            raise TypeError("filter_log_likelihood_and_grad must be callable")
        policy = str(batch_rank_policy)
        if policy not in _KNOWN_BATCH_RANK_POLICIES:
            raise InvalidSSMTargetBuilderContract(
                f"unknown batch_rank_policy: {policy}"
            )
        dtype_obj = tf.as_dtype(dtype)
        target = str(target_scope or validated.problem.problem_id)
        if not target:
            raise InvalidSSMTargetBuilderContract("target_scope must be nonempty")
        runtime = str(runtime_backend)
        if not runtime:
            raise InvalidSSMTargetBuilderContract("runtime_backend must be nonempty")

        self.contract = validated
        self.prior_log_prob_and_grad = prior_log_prob_and_grad
        self.filter_log_likelihood_and_grad = filter_log_likelihood_and_grad
        self.dtype = dtype_obj
        self.batch_rank_policy = policy
        self.parameter_dim = int(validated.chart.unconstrained_dim)
        self.parameter_names = tuple(validated.chart.parameter_names)
        self.target_scope = target
        self.evidence_path = evidence_path
        self.runtime_backend = runtime
        self.xla_hmc_ready = bool(xla_hmc_ready)
        self.full_chain_xla_diagnostic_ready = bool(full_chain_xla_diagnostic_ready)
        if self.full_chain_xla_diagnostic_ready and not self.xla_hmc_ready:
            raise InvalidSSMTargetBuilderContract(
                "full_chain_xla_diagnostic_ready requires xla_hmc_ready"
            )
        if self.xla_hmc_ready and not evidence_path:
            raise InvalidSSMTargetBuilderContract(
                "xla_hmc_ready requires a reviewed evidence_path"
            )
        self.nonclaims = tuple(str(item) for item in nonclaims) or TARGET_BUILDER_NONCLAIMS
        self.target_signature = stable_ssm_target_signature(validated)
        self.non_batch_static_shape = dict(
            validated.problem.static_shape.manifest_payload()
        )
        self._adapter_signature = _stable_json_hash(self._signature_payload())

    def manifest_payload(self) -> Mapping[str, Any]:
        capability = self.value_score_capability()
        return {
            **self._signature_payload(),
            "contract_manifest": self.contract.manifest_payload(),
            "xla_hmc_ready": capability.xla_hmc_ready,
            "full_chain_xla_diagnostic_ready": (
                capability.full_chain_xla_diagnostic_ready
            ),
            "evidence_path": self.evidence_path,
            "nonclaims": self.nonclaims,
        }

    def metadata(self) -> SSMTargetBuilderMetadata:
        return SSMTargetBuilderMetadata(
            schema="bayesfilter.ssm.generic_posterior_adapter.v1",
            target_signature=self.target_signature,
            adapter_signature=self.adapter_signature(),
            dtype=self.dtype.name,
            parameter_dim=self.parameter_dim,
            parameter_names=self.parameter_names,
            batch_rank_policy=self.batch_rank_policy,
            non_batch_static_shape=self.non_batch_static_shape,
            value_score_authority="graph_native",
            runtime_backend=self.runtime_backend,
            target_scope=self.target_scope,
            nonclaims=self.nonclaims,
        )

    def adapter_signature(self) -> str:
        return self._adapter_signature

    def value_score_capability(self) -> ValueScoreCapability:
        return ValueScoreCapability(
            value_score_authority="graph_native",
            xla_hmc_ready=self.xla_hmc_ready,
            full_chain_xla_diagnostic_ready=self.full_chain_xla_diagnostic_ready,
            runtime_backend=self.runtime_backend,
            evidence_path=self.evidence_path,
            target_scope=self.target_scope,
            nonclaims=self.nonclaims,
        )

    def log_prob(self, theta: Any) -> tf.Tensor:
        value, _score = self.log_prob_and_grad(theta)
        return value

    def log_prob_and_grad(self, theta: Any) -> tuple[tf.Tensor, tf.Tensor]:
        theta_tensor = self._validate_theta(theta)
        prior_value, prior_score = self._call_value_score(
            self.prior_log_prob_and_grad,
            theta_tensor,
            "prior",
        )
        likelihood_value, likelihood_score = self._call_value_score(
            self.filter_log_likelihood_and_grad,
            theta_tensor,
            "filter likelihood",
        )
        value = prior_value + likelihood_value
        score = prior_score + likelihood_score
        value, score = self._validate_value_score(theta_tensor, value, score, "posterior")
        with tf.control_dependencies(
            [
                tf.debugging.assert_all_finite(
                    value,
                    "posterior target value must be finite",
                ),
                tf.debugging.assert_all_finite(
                    score,
                    "posterior target score must be finite",
                ),
            ]
        ):
            return tf.identity(value), tf.identity(score)

    def log_prob_and_grad_batch(self, theta: Any) -> tuple[tf.Tensor, tf.Tensor]:
        return self.log_prob_and_grad(theta)

    def _signature_payload(self) -> Mapping[str, Any]:
        return {
            "schema": "bayesfilter.ssm.generic_posterior_adapter.v1",
            "target_signature": self.target_signature,
            "dtype": self.dtype.name,
            "parameter_dim": self.parameter_dim,
            "parameter_names": self.parameter_names,
            "batch_rank_policy": self.batch_rank_policy,
            "non_batch_static_shape": self.non_batch_static_shape,
            "value_score_authority": "graph_native",
            "runtime_backend": self.runtime_backend,
            "target_scope": self.target_scope,
        }

    def _validate_theta(self, theta: Any) -> tf.Tensor:
        tensor = tf.convert_to_tensor(theta, dtype=self.dtype)
        if tensor.shape.rank != 2:
            raise ValueError("batch-native SSM target requires rank 2 theta [B, D]")
        trailing = tensor.shape[-1]
        if trailing is None:
            raise ValueError("batch-native SSM target requires static parameter dimension")
        if int(trailing) != self.parameter_dim:
            raise ValueError("theta trailing dimension must match chart unconstrained_dim")
        return tensor

    def _call_value_score(
        self,
        fn: BatchValueScoreFn,
        theta: tf.Tensor,
        label: str,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        result = fn(theta)
        if not isinstance(result, tuple) or len(result) != 2:
            raise TypeError(f"{label} function must return (value, score)")
        value, score = result
        return self._validate_value_score(theta, value, score, label)

    def _validate_value_score(
        self,
        theta: tf.Tensor,
        value: Any,
        score: Any,
        label: str,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        value_tensor = tf.convert_to_tensor(value, dtype=self.dtype)
        score_tensor = tf.convert_to_tensor(score, dtype=self.dtype)
        if value_tensor.shape.rank != 1:
            raise ValueError(f"{label} value must have rank 1 [B]")
        if score_tensor.shape.rank != 2:
            raise ValueError(f"{label} score must have rank 2 [B, D]")
        score_dim = score_tensor.shape[-1]
        if score_dim is None:
            raise ValueError(f"{label} score must have static parameter dimension")
        if int(score_dim) != self.parameter_dim:
            raise ValueError(f"{label} score trailing dimension must match parameter_dim")
        checks = [
            tf.debugging.assert_equal(
                tf.shape(value_tensor)[0],
                tf.shape(theta)[0],
                message=f"{label} value leading batch dimension mismatch",
            ),
            tf.debugging.assert_equal(
                tf.shape(score_tensor)[0],
                tf.shape(theta)[0],
                message=f"{label} score leading batch dimension mismatch",
            ),
        ]
        with tf.control_dependencies(checks):
            return tf.identity(value_tensor), tf.identity(score_tensor)


def build_ssm_posterior_adapter(
    *,
    contract: SSMTargetContract,
    prior_log_prob_and_grad: BatchValueScoreFn,
    filter_log_likelihood_and_grad: BatchValueScoreFn,
    dtype: Any = tf.float64,
    batch_rank_policy: BatchRankPolicy = "rank2_required",
    target_scope: str | None = None,
        evidence_path: str | None = None,
        runtime_backend: str = "tensorflow",
        xla_hmc_ready: bool = False,
        full_chain_xla_diagnostic_ready: bool = False,
        nonclaims: tuple[str, ...] = TARGET_BUILDER_NONCLAIMS,
) -> GenericSSMPosteriorAdapter:
    """Build a batch-native posterior adapter from a generic SSM contract."""

    return GenericSSMPosteriorAdapter(
        contract=contract,
        prior_log_prob_and_grad=prior_log_prob_and_grad,
        filter_log_likelihood_and_grad=filter_log_likelihood_and_grad,
        dtype=dtype,
        batch_rank_policy=batch_rank_policy,
        target_scope=target_scope,
        evidence_path=evidence_path,
        runtime_backend=runtime_backend,
        xla_hmc_ready=xla_hmc_ready,
        full_chain_xla_diagnostic_ready=full_chain_xla_diagnostic_ready,
        nonclaims=nonclaims,
    )


def stable_ssm_posterior_adapter_signature(adapter: GenericSSMPosteriorAdapter) -> str:
    """Return the persisted stable signature for a generic SSM adapter."""

    if not isinstance(adapter, GenericSSMPosteriorAdapter):
        raise TypeError("adapter must be a GenericSSMPosteriorAdapter")
    return adapter.adapter_signature()


def _stable_json_hash(payload: Mapping[str, Any] | Any) -> str:
    normalized = _json_safe(payload)
    blob = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    _reject_process_local_identity(blob)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "numpy"):
        return _json_safe(value.numpy())
    if hasattr(value, "tolist") and hasattr(value, "shape"):
        return _json_safe(value.tolist())
    if hasattr(value, "item") and not isinstance(value, Mapping):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _reject_process_local_identity(value: Any) -> None:
    text = json.dumps(_json_safe(value), sort_keys=True, default=str)
    if any(pattern.search(text) for pattern in _PROCESS_LOCAL_SIGNATURE_PATTERNS):
        raise InvalidSSMTargetBuilderContract(
            "SSM target-builder signatures must not contain process-local identity"
        )
