"""SSL-LSTM filter-HMC value/score protocol metadata.

This module defines a small fail-closed protocol layer for the SSL-LSTM
filter-HMC program.  It does not implement a filter and does not claim that any
candidate adapter is ready; it validates the metadata an adapter must provide
before later HMC phases can consume it.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from bayesfilter.inference.posterior_adapter import (
    InvalidNonlinearSSMContract,
    NonlinearSSMAdapterContract,
    NonlinearSSMStaticShape,
    ObservationSemantics,
    ParameterTransformMetadata,
    RegularizationConvention,
    ValueScoreCapability,
    stable_nonlinear_ssm_program_signature,
    validate_nonlinear_ssm_contract,
)


SSLLSTMFilterName = Literal[
    "fixed_sgqf",
    "svd_ukf",
    "zhaocui_fixed",
    "ledh_streaming_ot",
]
SSLLSTMGradientPath = Literal[
    "analytic_first_order_fixed_sgqf",
    "analytic_first_order_svd_ukf",
    "analytic_first_order_zhaocui_fixed",
    "manual_vjp_streaming_ot",
]
SSLLSTMArtifactRole = Literal["target", "debug_reference"]

SSL_LSTM_VALUE_SCORE_SCHEMA_VERSION = "ssl_lstm.filter_hmc.value_score.v1"
_GATES = ("input", "forget", "output", "candidate")
_KNOWN_FILTERS = {
    "fixed_sgqf",
    "svd_ukf",
    "zhaocui_fixed",
    "ledh_streaming_ot",
}
_KNOWN_GRADIENT_PATHS = {
    "analytic_first_order_fixed_sgqf",
    "analytic_first_order_svd_ukf",
    "analytic_first_order_zhaocui_fixed",
    "manual_vjp_streaming_ot",
}
_FORBIDDEN_TARGET_AUTHORITIES = {
    "gradient_tape_fallback",
    "reviewed_gradient_tape_xla_exception",
    "reviewed_tf_py_function_finite_reject_bridge",
    "debug_only",
    "unavailable",
}
_REQUIRED_ARTIFACT_FIELDS = (
    "schema_version",
    "artifact_role",
    "target_scope",
    "filter_name",
    "gradient_path",
    "value_score_authority",
    "compile_mode",
    "jit_compile",
    "device",
    "tf32_enabled",
    "seed_policy",
    "branch_or_randomness_policy",
    "log_likelihood",
    "score",
    "score_finite",
    "finite_difference_check",
    "diagnostic_roles",
    "nonclaims",
)


class InvalidSSLLSTMProtocol(InvalidNonlinearSSMContract):
    """Raised when SSL-LSTM filter-HMC metadata is incomplete or unsafe."""


@dataclass(frozen=True)
class SSLLSTMStaticConfig:
    """Static dimensions and parameter layout for the Phase-1 SSL-LSTM target."""

    horizon: int
    latent_dim: int
    hidden_dim: int
    observation_dim: int
    covariance_mode: str = "diagonal"

    def __post_init__(self) -> None:
        for name in ("horizon", "latent_dim", "hidden_dim", "observation_dim"):
            value = int(getattr(self, name))
            if value <= 0:
                raise InvalidSSLLSTMProtocol(f"{name} must be positive")
            object.__setattr__(self, name, value)
        if self.covariance_mode != "diagonal":
            raise InvalidSSLLSTMProtocol(
                "Phase-2 SSL-LSTM protocol currently admits diagonal covariance mode only"
            )

    @property
    def augmented_state_dim(self) -> int:
        return self.latent_dim + 2 * self.hidden_dim

    @property
    def innovation_dim(self) -> int:
        return self.latent_dim

    @property
    def parameter_names(self) -> tuple[str, ...]:
        names: list[str] = []
        k = self.latent_dim
        h = self.hidden_dim
        d = self.observation_dim
        n = self.augmented_state_dim
        for gate in _GATES:
            for row in range(h):
                for col in range(k):
                    names.append(f"lstm_input.{gate}.{row}.{col}")
        for gate in _GATES:
            for row in range(h):
                for col in range(h):
                    names.append(f"lstm_recurrent.{gate}.{row}.{col}")
        for gate in _GATES:
            for row in range(h):
                names.append(f"lstm_bias.{gate}.{row}")
        for row in range(k):
            for col in range(h):
                names.append(f"latent_mean_weight.{row}.{col}")
        for row in range(k):
            names.append(f"latent_mean_bias.{row}")
        for row in range(d):
            for col in range(k):
                names.append(f"observation_weight.{row}.{col}")
        for row in range(d):
            names.append(f"observation_bias.{row}")
        for row in range(n):
            names.append(f"initial_mean.{row}")
        for row in range(n):
            names.append(f"initial_std_unconstrained.{row}")
        for row in range(k):
            names.append(f"process_std_unconstrained.{row}")
        for row in range(d):
            names.append(f"observation_std_unconstrained.{row}")
        return tuple(names)

    @property
    def parameter_dim(self) -> int:
        return len(self.parameter_names)

    def static_shape(self) -> NonlinearSSMStaticShape:
        return NonlinearSSMStaticShape(
            horizon=self.horizon,
            state_dim=self.augmented_state_dim,
            observation_dim=self.observation_dim,
            innovation_dim=self.innovation_dim,
            parameter_dim=self.parameter_dim,
            constrained_parameter_shape=(self.parameter_dim,),
            unconstrained_parameter_shape=(self.parameter_dim,),
        )


@dataclass(frozen=True)
class SSLLSTMFilterProtocolSpec:
    """Required metadata for one candidate filter lane."""

    filter_name: SSLLSTMFilterName
    filter_implementation: str
    likelihood_term: str
    required_gradient_path: SSLLSTMGradientPath
    required_seed_policy: str
    branch_or_randomness_policy: str


_FILTER_SPECS: Mapping[str, SSLLSTMFilterProtocolSpec] = {
    "fixed_sgqf": SSLLSTMFilterProtocolSpec(
        filter_name="fixed_sgqf",
        filter_implementation="ssl_lstm_fixed_sgqf",
        likelihood_term="tf_fixed_sgqf_score",
        required_gradient_path="analytic_first_order_fixed_sgqf",
        required_seed_policy="not_used",
        branch_or_randomness_policy="fixed_sparse_grid_branch_manifest",
    ),
    "svd_ukf": SSLLSTMFilterProtocolSpec(
        filter_name="svd_ukf",
        filter_implementation="ssl_lstm_svd_ukf",
        likelihood_term="tf_svd_ukf_score",
        required_gradient_path="analytic_first_order_svd_ukf",
        required_seed_policy="not_used",
        branch_or_randomness_policy="deterministic_sigma_point_rule",
    ),
    "zhaocui_fixed": SSLLSTMFilterProtocolSpec(
        filter_name="zhaocui_fixed",
        filter_implementation="ssl_lstm_zhaocui_fixed",
        likelihood_term="zhaocui_fixed_analytic_score",
        required_gradient_path="analytic_first_order_zhaocui_fixed",
        required_seed_policy="stateless_required",
        branch_or_randomness_policy="fixed_hmc_adaptation_manifest",
    ),
    "ledh_streaming_ot": SSLLSTMFilterProtocolSpec(
        filter_name="ledh_streaming_ot",
        filter_implementation="ssl_lstm_ledh_streaming_ot_manual_vjp",
        likelihood_term="ledh_streaming_ot_manual_vjp_score",
        required_gradient_path="manual_vjp_streaming_ot",
        required_seed_policy="stateless_required",
        branch_or_randomness_policy="streaming_ot_manual_vjp_fixed_seed_manifest",
    ),
}


@dataclass(frozen=True)
class SSLLSTMAdapterProtocol:
    """A candidate adapter contract plus SSL-LSTM-specific gradient metadata."""

    filter_name: SSLLSTMFilterName
    gradient_path: SSLLSTMGradientPath
    contract: NonlinearSSMAdapterContract
    branch_or_randomness_policy: str
    artifact_schema_version: str = SSL_LSTM_VALUE_SCORE_SCHEMA_VERSION

    @property
    def spec(self) -> SSLLSTMFilterProtocolSpec:
        return ssl_lstm_filter_protocol_spec(self.filter_name)

    def signature_payload(self) -> Mapping[str, Any]:
        return {
            "filter_name": self.filter_name,
            "gradient_path": self.gradient_path,
            "branch_or_randomness_policy": self.branch_or_randomness_policy,
            "artifact_schema_version": self.artifact_schema_version,
            "contract_signature": stable_nonlinear_ssm_program_signature(self.contract),
        }

    def stable_signature(self) -> str:
        blob = json.dumps(
            _normalize_for_json(self.signature_payload()),
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def ssl_lstm_filter_protocol_spec(
    filter_name: str,
) -> SSLLSTMFilterProtocolSpec:
    """Return the protocol requirements for a known SSL-LSTM filter lane."""

    name = str(filter_name)
    try:
        return _FILTER_SPECS[name]
    except KeyError as exc:
        raise InvalidSSLLSTMProtocol(f"unknown SSL-LSTM filter: {name}") from exc


def build_expected_ssl_lstm_adapter_protocol(
    config: SSLLSTMStaticConfig,
    *,
    filter_name: SSLLSTMFilterName,
    evidence_path: str,
    target_scope: str | None = None,
    gradient_path: SSLLSTMGradientPath | None = None,
    value_score_authority: str = "graph_native",
    xla_hmc_ready: bool = True,
    compile_mode: str = "xla",
    nonclaims: Sequence[str] = (
        "protocol contract only",
        "no filter accuracy claim",
        "no HMC convergence claim",
        "no SSL-LSTM estimation claim",
    ),
) -> SSLLSTMAdapterProtocol:
    """Build an expected SSL-LSTM adapter contract for later implementations.

    The returned object is protocol metadata.  It is not evidence that the
    corresponding filter implementation exists or passes numerical checks.
    """

    spec = ssl_lstm_filter_protocol_spec(filter_name)
    gradient = gradient_path or spec.required_gradient_path
    scope = target_scope or f"ssl_lstm_filter_hmc:{filter_name}"
    capability = ValueScoreCapability(
        value_score_authority=value_score_authority,  # type: ignore[arg-type]
        xla_hmc_ready=xla_hmc_ready,
        runtime_backend="tensorflow",
        evidence_path=evidence_path,
        target_scope=scope,
        nonclaims=tuple(nonclaims),
    )
    contract = NonlinearSSMAdapterContract(
        parameter_names=config.parameter_names,
        static_shape=config.static_shape(),
        transform=ParameterTransformMetadata(
            orientation="unconstrained_to_constrained",
            inverse_orientation="constrained_to_unconstrained",
            log_det_jacobian_convention="included_in_prior_term",
            transform_source="ssl_lstm_filter_hmc_phase1_parameterization",
        ),
        observation_semantics=ObservationSemantics(
            mask_convention="none",
            missingness_convention="none",
            mask_shape=(config.horizon, config.observation_dim),
        ),
        regularization=RegularizationConvention(
            jitter=0.0,
            covariance_floor=1.0e-12,
            psd_repair="positive_diagonal_softplus_floor",
            symmetrize=True,
            logdet_convention="implemented_covariance",
            implemented_covariance="diagonal_positive_covariance_from_transform",
            repair_role="target",
        ),
        value_score=capability,
        prior_term="ssl_lstm_explicit_parameter_prior_with_transform_jacobian",
        likelihood_term=spec.likelihood_term,
        dtype="float64",
        backend="tensorflow",
        filter_implementation=spec.filter_implementation,
        compile_mode=compile_mode,  # type: ignore[arg-type]
        trace_policy="filter_trace_optional",
        return_filtered=True,
        seed_policy=spec.required_seed_policy,  # type: ignore[arg-type]
        map_source="not_used",
        mass_matrix_source="not_used",
        hessian_source="not_used",
    )
    protocol = SSLLSTMAdapterProtocol(
        filter_name=spec.filter_name,
        gradient_path=gradient,
        contract=contract,
        branch_or_randomness_policy=spec.branch_or_randomness_policy,
    )
    return validate_ssl_lstm_adapter_protocol(protocol)


def validate_ssl_lstm_adapter_protocol(
    protocol: SSLLSTMAdapterProtocol,
    *,
    require_xla_hmc_ready: bool = True,
) -> SSLLSTMAdapterProtocol:
    """Validate SSL-LSTM candidate metadata and fail closed on unsafe scores."""

    if not isinstance(protocol, SSLLSTMAdapterProtocol):
        raise TypeError("protocol must be an SSLLSTMAdapterProtocol")
    spec = ssl_lstm_filter_protocol_spec(protocol.filter_name)
    if protocol.gradient_path not in _KNOWN_GRADIENT_PATHS:
        raise InvalidSSLLSTMProtocol(
            f"unknown SSL-LSTM gradient path: {protocol.gradient_path}"
        )
    if protocol.gradient_path != spec.required_gradient_path:
        raise InvalidSSLLSTMProtocol(
            f"{protocol.filter_name} requires gradient path "
            f"{spec.required_gradient_path}, got {protocol.gradient_path}"
        )
    if protocol.contract.filter_implementation != spec.filter_implementation:
        raise InvalidSSLLSTMProtocol(
            f"{protocol.filter_name} requires filter_implementation "
            f"{spec.filter_implementation}"
        )
    if protocol.contract.likelihood_term != spec.likelihood_term:
        raise InvalidSSLLSTMProtocol(
            f"{protocol.filter_name} requires likelihood_term {spec.likelihood_term}"
        )
    if protocol.contract.seed_policy != spec.required_seed_policy:
        raise InvalidSSLLSTMProtocol(
            f"{protocol.filter_name} requires seed_policy {spec.required_seed_policy}"
        )
    if protocol.branch_or_randomness_policy != spec.branch_or_randomness_policy:
        raise InvalidSSLLSTMProtocol(
            f"{protocol.filter_name} requires branch_or_randomness_policy "
            f"{spec.branch_or_randomness_policy}"
        )
    authority = protocol.contract.value_score.value_score_authority
    if authority in _FORBIDDEN_TARGET_AUTHORITIES:
        raise InvalidSSLLSTMProtocol(
            f"{protocol.filter_name} target score cannot use {authority}"
        )
    if authority != "graph_native":
        raise InvalidSSLLSTMProtocol(
            f"{protocol.filter_name} target score requires graph_native authority"
        )
    validate_nonlinear_ssm_contract(
        protocol.contract,
        require_xla_hmc_ready=require_xla_hmc_ready,
    )
    if protocol.contract.value_score.evidence_path is None:
        raise InvalidSSLLSTMProtocol("SSL-LSTM value/score authority requires evidence_path")
    if not protocol.contract.value_score.target_scope:
        raise InvalidSSLLSTMProtocol("SSL-LSTM value/score authority requires target_scope")
    if not protocol.contract.value_score.nonclaims:
        raise InvalidSSLLSTMProtocol("SSL-LSTM value/score authority requires nonclaims")
    if protocol.artifact_schema_version != SSL_LSTM_VALUE_SCORE_SCHEMA_VERSION:
        raise InvalidSSLLSTMProtocol("unsupported SSL-LSTM artifact schema version")
    return protocol


def ssl_lstm_value_score_artifact_schema() -> Mapping[str, object]:
    """Return the required value/score artifact schema for this program."""

    return {
        "schema_version": SSL_LSTM_VALUE_SCORE_SCHEMA_VERSION,
        "required_fields": _REQUIRED_ARTIFACT_FIELDS,
        "diagnostic_roles": {
            "score_finite": "promotion_veto",
            "finite_difference_check": "promotion_veto_for_adapter_admission",
            "runtime": "explanatory",
            "score_norm": "explanatory",
        },
        "nonclaims": (
            "artifact schema only",
            "no filter accuracy claim",
            "no HMC convergence claim",
            "no SSL-LSTM estimation claim",
        ),
    }


def validate_ssl_lstm_value_score_artifact(
    artifact: Mapping[str, Any],
    *,
    protocol: SSLLSTMAdapterProtocol | None = None,
) -> Mapping[str, Any]:
    """Validate a value/score artifact dictionary against the Phase-2 schema."""

    missing = [field for field in _REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise InvalidSSLLSTMProtocol(
            "SSL-LSTM value/score artifact missing fields: " + ", ".join(missing)
        )
    schema_version = str(artifact["schema_version"])
    if schema_version != SSL_LSTM_VALUE_SCORE_SCHEMA_VERSION:
        raise InvalidSSLLSTMProtocol(
            f"unsupported SSL-LSTM value/score schema: {schema_version}"
        )
    role = str(artifact["artifact_role"])
    if role not in {"target", "debug_reference"}:
        raise InvalidSSLLSTMProtocol(f"unknown artifact_role: {role}")
    spec = ssl_lstm_filter_protocol_spec(str(artifact["filter_name"]))
    gradient_path = str(artifact["gradient_path"])
    if gradient_path != spec.required_gradient_path:
        raise InvalidSSLLSTMProtocol(
            f"artifact gradient_path must be {spec.required_gradient_path}"
        )
    authority = str(artifact["value_score_authority"])
    if authority != "graph_native":
        raise InvalidSSLLSTMProtocol(
            "SSL-LSTM target value/score artifact requires graph_native authority"
        )
    if str(artifact["seed_policy"]) != spec.required_seed_policy:
        raise InvalidSSLLSTMProtocol(
            f"artifact seed_policy must be {spec.required_seed_policy}"
        )
    if str(artifact["branch_or_randomness_policy"]) != spec.branch_or_randomness_policy:
        raise InvalidSSLLSTMProtocol(
            "artifact branch_or_randomness_policy does not match filter protocol"
        )
    nonclaims = tuple(str(item) for item in artifact["nonclaims"])
    if not nonclaims:
        raise InvalidSSLLSTMProtocol("artifact nonclaims must be nonempty")
    if role == "target":
        if str(artifact["compile_mode"]) != "xla":
            raise InvalidSSLLSTMProtocol("target artifact requires compile_mode xla")
        if bool(artifact["jit_compile"]) is not True:
            raise InvalidSSLLSTMProtocol("target artifact requires jit_compile true")
    if bool(artifact["score_finite"]) is not True:
        raise InvalidSSLLSTMProtocol("artifact score_finite hard veto failed")
    roles = artifact["diagnostic_roles"]
    if not isinstance(roles, Mapping):
        raise InvalidSSLLSTMProtocol("diagnostic_roles must be a mapping")
    if roles.get("finite_difference_check") not in {
        "promotion_veto_for_adapter_admission",
        "explanatory",
    }:
        raise InvalidSSLLSTMProtocol(
            "finite_difference_check role must be declared as veto or explanatory"
        )
    if protocol is not None:
        validate_ssl_lstm_adapter_protocol(protocol)
        if str(artifact["filter_name"]) != protocol.filter_name:
            raise InvalidSSLLSTMProtocol("artifact filter_name does not match protocol")
        if str(artifact["target_scope"]) != protocol.contract.value_score.target_scope:
            raise InvalidSSLLSTMProtocol("artifact target_scope does not match protocol")
    return dict(artifact)


def _normalize_for_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _normalize_for_json(value[key]) for key in sorted(value)}
    if isinstance(value, tuple):
        return [_normalize_for_json(item) for item in value]
    if isinstance(value, list):
        return [_normalize_for_json(item) for item in value]
    return value


__all__ = [
    "InvalidSSLLSTMProtocol",
    "SSL_LSTM_VALUE_SCORE_SCHEMA_VERSION",
    "SSLLSTMAdapterProtocol",
    "SSLLSTMArtifactRole",
    "SSLLSTMFilterName",
    "SSLLSTMFilterProtocolSpec",
    "SSLLSTMGradientPath",
    "SSLLSTMStaticConfig",
    "build_expected_ssl_lstm_adapter_protocol",
    "ssl_lstm_filter_protocol_spec",
    "ssl_lstm_value_score_artifact_schema",
    "validate_ssl_lstm_adapter_protocol",
    "validate_ssl_lstm_value_score_artifact",
]
