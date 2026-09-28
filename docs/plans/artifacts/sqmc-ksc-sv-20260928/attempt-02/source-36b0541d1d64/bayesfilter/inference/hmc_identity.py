"""Versioned semantic identities for retained fixed-kernel HMC execution."""

from __future__ import annotations

import hashlib
import json
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


CANONICAL_ARRAY_IDENTITY_SCHEMA_V1 = "bayesfilter.canonical_array_identity.v1"
CANONICAL_FLOAT64_SCHEMA_V1 = "bayesfilter.canonical_float64.v1"
DETERMINISTIC_LGSSM_TARGET_IDENTITY_SCHEMA_V1 = (
    "bayesfilter.deterministic_lgssm_target_identity.v1"
)
FROZEN_HMC_TRANSFORM_IDENTITY_SCHEMA_V1 = (
    "bayesfilter.frozen_hmc_transform_identity.v1"
)
FROZEN_HMC_TRANSITION_IDENTITY_SCHEMA_V1 = (
    "bayesfilter.frozen_hmc_transition_identity.v1"
)
FROZEN_HMC_EXECUTION_CONTRACT_SCHEMA_V1 = (
    "bayesfilter.frozen_hmc_execution_contract.v1"
)
SELECTION_STAGE_IDENTITY_SCHEMA_V1 = "bayesfilter.hmc_selection_stage_identity.v1"
SELECTION_PROVENANCE_IDENTITY_SCHEMA_V1 = (
    "bayesfilter.hmc_selection_provenance_identity.v1"
)

TFP_HMC_KERNEL_FAMILY_V1 = "tfp.mcmc.HamiltonianMonteCarlo"
TFP_HMC_INTEGRATOR_ROUTE_V1 = "tfp_hmc_leapfrog_one_step_tf_while_loop"
DETERMINISTIC_LGSSM_TARGET_ROUTE_V1 = (
    "bayesfilter.testing.multidim_triangular_lgssm_tf."
    "lower_triangular_lgssm_log_prob_score_status.v1"
)
DETERMINISTIC_LGSSM_SCORE_VALIDITY_ROUTE_V1 = (
    "svd_eigh_graph_status_pre_regularized_or_nan.v1"
)
PHASE7_SEED_DERIVATION_ROUTE_V1 = (
    "root_plus_stage_100000_10000_check_1009_101_worker_37_17.v1"
)
PHASE7_INITIAL_STATE_POLICY_V1 = (
    "linspace_minus_0p15_plus_0p15_alternating_sign_global_chain_order.v1"
)
PHASE7_DIAGNOSTIC_ROUTE_V1 = "bayesfilter.rank_normalized_hmc_diagnostics.v1"
PHASE7_SERIOUS_DIAGNOSTIC_GATE_V1 = "all_rank_normalized_thresholds_pass.v1"
PHASE7_SMOKE_DIAGNOSTIC_GATE_V1 = "finite_diagnostics_only_non_promoting.v1"
PHASE7_WORKER_HARD_VETO_POLICY_V1 = (
    "nonfinite_divergence_xla_retrace_worker_order_pid.v1"
)
PHASE7_CONTROLLER_POLICY_V1 = "extend_until_all_diagnostics_pass_or_cap.v1"
PHASE7_NO_RESUME_POLICY_V1 = "fresh_run_no_resume_no_manual_mutation.v1"
PHASE7_WORKER_PARTITION_POLICY_V1 = (
    "contiguous_global_chains_ascending_worker_response_stable_pid.v1"
)
PHASE7_DIAGNOSTIC_ACCUMULATION_POLICY_V1 = (
    "burnin_latest_window_retained_cumulative_draws.v1"
)

_HEX_DIGITS = frozenset("0123456789abcdef")
_ALLOWED_SELECTION_STAGES = frozenset(
    {
        "bootstrap",
        "geometry",
        "windowed_mass",
        "fixed_mass_step",
        "frozen_step_trajectory",
        "fresh_verification",
        "tune_verify_repair_loop",
    }
)


def _require_exact_keys(
    payload: Mapping[str, Any],
    *,
    required: Sequence[str],
    label: str,
) -> None:
    if not isinstance(payload, Mapping):
        raise TypeError(f"{label} must be a mapping")
    if any(not isinstance(key, str) for key in payload):
        raise ValueError(f"{label} keys must be strings")
    expected = frozenset(required)
    observed = frozenset(payload)
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    if missing or extra:
        raise ValueError(f"{label} fields mismatch: missing={missing}, extra={extra}")


def _require_nonempty(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonblank string")
    if value != value.strip():
        raise ValueError(f"{label} must not contain surrounding whitespace")
    return value


def _require_sha256(value: Any, *, label: str, tagged: bool = True) -> str:
    text = _require_nonempty(value, label=label)
    digest = text.removeprefix("sha256:") if tagged else text
    if tagged and not text.startswith("sha256:"):
        raise ValueError(f"{label} must use the sha256: prefix")
    if not tagged and text.startswith("sha256:"):
        raise ValueError(f"{label} must be a bare SHA-256 digest")
    if len(digest) != 64 or any(char not in _HEX_DIGITS for char in digest):
        raise ValueError(f"{label} must be a complete lowercase SHA-256")
    return text


def _require_int(value: Any, *, label: str, minimum: int | None = None) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{label} must be an integer")
    number = int(value)
    if minimum is not None and number < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    return number


def _require_bool(value: Any, *, label: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{label} must be boolean")
    return bool(value)


def _require_string_tuple(
    values: Any,
    *,
    label: str,
    nonempty: bool = True,
) -> tuple[str, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError(f"{label} must be a sequence")
    result = tuple(
        _require_nonempty(value, label=f"{label}[{index}]")
        for index, value in enumerate(values)
    )
    if nonempty and not result:
        raise ValueError(f"{label} must be non-empty")
    if len(set(result)) != len(result):
        raise ValueError(f"{label} must not contain duplicates")
    return result


def _strict_json_value(value: Any, *, label: str = "payload") -> Any:
    """Return a type-tagged canonical tree for one parsed JSON value."""

    if value is None:
        return ["null"]
    if isinstance(value, str):
        return ["string", value]
    if isinstance(value, (bool, np.bool_)):
        return ["boolean", bool(value)]
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        return ["integer", str(int(value))]
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return ["float64_ieee754", struct.pack(">d", number).hex()]
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError(f"{label} contains a non-string mapping key")
        return [
            "mapping",
            [
                [key, _strict_json_value(item, label=f"{label}.{key}")]
                for key, item in sorted(value.items())
            ],
        ]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [
            "sequence",
            [
                _strict_json_value(item, label=f"{label}[{index}]")
                for index, item in enumerate(value)
            ],
        ]
    raise TypeError(f"{label} contains unsupported type {type(value).__name__}")


def _canonical_json_hash(payload: Mapping[str, Any]) -> str:
    normalized = _strict_json_value(payload)
    blob = json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _identity_hash(payload: Mapping[str, Any]) -> str:
    return f"sha256:{_canonical_json_hash(payload)}"


@dataclass(frozen=True)
class CanonicalArrayIdentityV1:
    """Canonical numerical-array identity preserving dtype, shape, and bytes."""

    semantic_dtype: str
    shape: tuple[int, ...]
    byte_sha256: str
    canonical_byte_order: str = "big_endian"
    canonical_memory_order: str = "C"
    schema: str = CANONICAL_ARRAY_IDENTITY_SCHEMA_V1

    def __post_init__(self) -> None:
        if self.schema != CANONICAL_ARRAY_IDENTITY_SCHEMA_V1:
            raise ValueError("unsupported canonical array identity schema")
        dtype_text = _require_nonempty(self.semantic_dtype, label="semantic_dtype")
        dtype = np.dtype(dtype_text)
        if dtype.kind not in "biufc":
            raise ValueError("canonical array dtype must be numeric or boolean")
        shape = tuple(
            _require_int(item, label="shape entry", minimum=0) for item in self.shape
        )
        digest = _require_sha256(
            self.byte_sha256,
            label="canonical array byte_sha256",
            tagged=False,
        )
        if self.canonical_byte_order != "big_endian":
            raise ValueError("canonical array byte order must be big_endian")
        if self.canonical_memory_order != "C":
            raise ValueError("canonical array memory order must be C")
        object.__setattr__(self, "semantic_dtype", dtype.name)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "byte_sha256", digest)

    @classmethod
    def from_array(cls, value: Any) -> "CanonicalArrayIdentityV1":
        array = np.asarray(value)
        if array.dtype.kind not in "biufc":
            raise ValueError("canonical array input must be numeric or boolean")
        if array.dtype.kind in "fc" and not np.all(np.isfinite(array)):
            raise ValueError("canonical array input must be finite")
        canonical_dtype = array.dtype.newbyteorder(">")
        canonical = np.ascontiguousarray(array.astype(canonical_dtype, copy=False))
        return cls(
            semantic_dtype=array.dtype.name,
            shape=tuple(int(item) for item in array.shape),
            byte_sha256=hashlib.sha256(canonical.tobytes(order="C")).hexdigest(),
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "CanonicalArrayIdentityV1":
        _require_exact_keys(
            payload,
            required=(
                "schema",
                "semantic_dtype",
                "shape",
                "canonical_byte_order",
                "canonical_memory_order",
                "byte_sha256",
            ),
            label="canonical array identity",
        )
        return cls(
            schema=payload["schema"],
            semantic_dtype=payload["semantic_dtype"],
            shape=tuple(payload["shape"]),
            canonical_byte_order=payload["canonical_byte_order"],
            canonical_memory_order=payload["canonical_memory_order"],
            byte_sha256=payload["byte_sha256"],
        )

    def payload(self) -> Mapping[str, Any]:
        return {
            "schema": self.schema,
            "semantic_dtype": self.semantic_dtype,
            "shape": self.shape,
            "canonical_byte_order": self.canonical_byte_order,
            "canonical_memory_order": self.canonical_memory_order,
            "byte_sha256": self.byte_sha256,
        }

    @property
    def identity_hash(self) -> str:
        return _identity_hash(self.payload())


@dataclass(frozen=True)
class CanonicalFloat64V1:
    """Exact finite float64 identity using IEEE-754 big-endian bits."""

    ieee754_hex: str
    schema: str = CANONICAL_FLOAT64_SCHEMA_V1

    def __post_init__(self) -> None:
        if self.schema != CANONICAL_FLOAT64_SCHEMA_V1:
            raise ValueError("unsupported canonical float64 schema")
        bits = _require_nonempty(self.ieee754_hex, label="ieee754_hex")
        if len(bits) != 16 or any(char not in _HEX_DIGITS for char in bits):
            raise ValueError("float64 IEEE-754 identity must contain 16 lowercase hex digits")
        value = struct.unpack(">d", bytes.fromhex(bits))[0]
        if not np.isfinite(value):
            raise ValueError("canonical float64 value must be finite")
        object.__setattr__(self, "ieee754_hex", bits)

    @classmethod
    def from_value(cls, value: Any) -> "CanonicalFloat64V1":
        if isinstance(value, (bool, np.bool_)):
            raise ValueError("canonical float64 value must be numeric")
        number = float(np.float64(value))
        if not np.isfinite(number):
            raise ValueError("canonical float64 value must be finite")
        return cls(ieee754_hex=struct.pack(">d", number).hex())

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "CanonicalFloat64V1":
        _require_exact_keys(
            payload,
            required=("schema", "ieee754_hex"),
            label="canonical float64",
        )
        return cls(schema=payload["schema"], ieee754_hex=payload["ieee754_hex"])

    @property
    def value(self) -> float:
        return float(struct.unpack(">d", bytes.fromhex(self.ieee754_hex))[0])

    def payload(self) -> Mapping[str, Any]:
        return {"schema": self.schema, "ieee754_hex": self.ieee754_hex}


@dataclass(frozen=True)
class DeterministicLGSSMTargetIdentityV1:
    """Mathematical target inputs consumed by the deterministic LGSSM adapter."""

    observations: CanonicalArrayIdentityV1
    parameter_names: tuple[str, ...]
    state_dim: int
    observation_dim: int
    parameter_dim: int
    rho_max: CanonicalFloat64V1
    lower_scale: CanonicalFloat64V1
    truth_diag_a: CanonicalArrayIdentityV1
    truth_lower_a: CanonicalArrayIdentityV1
    truth_process_std: CanonicalArrayIdentityV1
    truth_observation_std: CanonicalArrayIdentityV1
    prior_scales: CanonicalArrayIdentityV1
    kalman_jitter: CanonicalFloat64V1
    singular_floor: CanonicalFloat64V1
    target_route: str = DETERMINISTIC_LGSSM_TARGET_ROUTE_V1
    score_validity_route: str = DETERMINISTIC_LGSSM_SCORE_VALIDITY_ROUTE_V1
    schema: str = DETERMINISTIC_LGSSM_TARGET_IDENTITY_SCHEMA_V1

    def __post_init__(self) -> None:
        if self.schema != DETERMINISTIC_LGSSM_TARGET_IDENTITY_SCHEMA_V1:
            raise ValueError("unsupported deterministic LGSSM target identity schema")
        if self.target_route != DETERMINISTIC_LGSSM_TARGET_ROUTE_V1:
            raise ValueError("unsupported deterministic LGSSM target route")
        if self.score_validity_route != DETERMINISTIC_LGSSM_SCORE_VALIDITY_ROUTE_V1:
            raise ValueError("unsupported deterministic LGSSM score-validity route")
        names = _require_string_tuple(self.parameter_names, label="parameter_names")
        state_dim = _require_int(self.state_dim, label="state_dim", minimum=1)
        observation_dim = _require_int(
            self.observation_dim,
            label="observation_dim",
            minimum=1,
        )
        parameter_dim = _require_int(
            self.parameter_dim,
            label="parameter_dim",
            minimum=1,
        )
        if parameter_dim != 18 or len(names) != parameter_dim:
            raise ValueError("deterministic LGSSM target requires 18 ordered parameters")
        if state_dim != 4 or observation_dim != 4:
            raise ValueError("deterministic LGSSM target requires 4D state and observation")
        if self.observations.semantic_dtype != "float64":
            raise ValueError("LGSSM observations must be float64")
        if len(self.observations.shape) != 2 or self.observations.shape[1] != observation_dim:
            raise ValueError("LGSSM observation shape must be (horizon, observation_dim)")
        if self.observations.shape[0] <= 1:
            raise ValueError("LGSSM observation horizon must be greater than one")
        expected_shapes = {
            "truth_diag_a": (4,),
            "truth_lower_a": (6,),
            "truth_process_std": (4,),
            "truth_observation_std": (4,),
            "prior_scales": (18,),
        }
        for field_name, expected_shape in expected_shapes.items():
            value = getattr(self, field_name)
            if value.semantic_dtype != "float64" or value.shape != expected_shape:
                raise ValueError(f"{field_name} must be float64 with shape {expected_shape}")
        if not (0.0 < self.rho_max.value < 1.0):
            raise ValueError("rho_max must lie strictly between zero and one")
        if self.lower_scale.value <= 0.0:
            raise ValueError("lower_scale must be positive")
        if self.kalman_jitter.value <= 0.0 or self.singular_floor.value <= 0.0:
            raise ValueError("LGSSM numerical floors must be positive")
        object.__setattr__(self, "parameter_names", names)
        object.__setattr__(self, "state_dim", state_dim)
        object.__setattr__(self, "observation_dim", observation_dim)
        object.__setattr__(self, "parameter_dim", parameter_dim)

    @classmethod
    def from_runtime_inputs(
        cls,
        *,
        observations: Any,
        parameter_names: Sequence[str],
        contract: Mapping[str, Any],
        prior_scales: Any | None = None,
        kalman_jitter: Any = 1.0e-9,
        singular_floor: Any = 1.0e-12,
    ) -> "DeterministicLGSSMTargetIdentityV1":
        if not isinstance(contract, Mapping):
            raise TypeError("LGSSM source contract must be a mapping")
        shape = contract.get("static_shape")
        transform = contract.get("transform")
        truth = contract.get("truth_template")
        if not all(isinstance(item, Mapping) for item in (shape, transform, truth)):
            raise ValueError("LGSSM source contract is missing mathematical fields")
        lower = truth.get("lower_A")
        if not isinstance(lower, Mapping):
            raise ValueError("LGSSM truth_template.lower_A must be a mapping")
        effective_prior_scales = (
            np.asarray([0.50] * 4 + [0.60] * 6 + [0.35] * 8, dtype=np.float64)
            if prior_scales is None
            else np.asarray(prior_scales, dtype=np.float64)
        )
        return cls(
            observations=CanonicalArrayIdentityV1.from_array(
                np.asarray(observations, dtype=np.float64)
            ),
            parameter_names=tuple(parameter_names),
            state_dim=_require_int(shape.get("state_dim"), label="state_dim", minimum=1),
            observation_dim=_require_int(
                shape.get("observation_dim"),
                label="observation_dim",
                minimum=1,
            ),
            parameter_dim=_require_int(
                shape.get("parameter_dim"),
                label="parameter_dim",
                minimum=1,
            ),
            rho_max=CanonicalFloat64V1.from_value(transform.get("rho_max")),
            lower_scale=CanonicalFloat64V1.from_value(transform.get("lower_scale")),
            truth_diag_a=CanonicalArrayIdentityV1.from_array(
                np.asarray(truth.get("diag_A"), dtype=np.float64)
            ),
            truth_lower_a=CanonicalArrayIdentityV1.from_array(
                np.asarray(
                    [lower.get(key) for key in ("a21", "a31", "a32", "a41", "a42", "a43")],
                    dtype=np.float64,
                )
            ),
            truth_process_std=CanonicalArrayIdentityV1.from_array(
                np.asarray(truth.get("process_std"), dtype=np.float64)
            ),
            truth_observation_std=CanonicalArrayIdentityV1.from_array(
                np.asarray(truth.get("observation_std"), dtype=np.float64)
            ),
            prior_scales=CanonicalArrayIdentityV1.from_array(effective_prior_scales),
            kalman_jitter=CanonicalFloat64V1.from_value(kalman_jitter),
            singular_floor=CanonicalFloat64V1.from_value(singular_floor),
        )

    @classmethod
    def from_adapter(cls, adapter: Any) -> "DeterministicLGSSMTargetIdentityV1":
        return cls.from_runtime_inputs(
            observations=getattr(adapter, "_observations", None),
            parameter_names=getattr(adapter, "_parameter_names", ()),
            contract=getattr(adapter, "_contract", None),
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "DeterministicLGSSMTargetIdentityV1":
        fields = (
            "schema",
            "target_route",
            "score_validity_route",
            "observations",
            "parameter_names",
            "state_dim",
            "observation_dim",
            "parameter_dim",
            "rho_max",
            "lower_scale",
            "truth_diag_a",
            "truth_lower_a",
            "truth_process_std",
            "truth_observation_std",
            "prior_scales",
            "kalman_jitter",
            "singular_floor",
        )
        _require_exact_keys(payload, required=fields, label="deterministic LGSSM target identity")
        return cls(
            schema=payload["schema"],
            target_route=payload["target_route"],
            score_validity_route=payload["score_validity_route"],
            observations=CanonicalArrayIdentityV1.from_payload(payload["observations"]),
            parameter_names=tuple(payload["parameter_names"]),
            state_dim=payload["state_dim"],
            observation_dim=payload["observation_dim"],
            parameter_dim=payload["parameter_dim"],
            rho_max=CanonicalFloat64V1.from_payload(payload["rho_max"]),
            lower_scale=CanonicalFloat64V1.from_payload(payload["lower_scale"]),
            truth_diag_a=CanonicalArrayIdentityV1.from_payload(payload["truth_diag_a"]),
            truth_lower_a=CanonicalArrayIdentityV1.from_payload(payload["truth_lower_a"]),
            truth_process_std=CanonicalArrayIdentityV1.from_payload(
                payload["truth_process_std"]
            ),
            truth_observation_std=CanonicalArrayIdentityV1.from_payload(
                payload["truth_observation_std"]
            ),
            prior_scales=CanonicalArrayIdentityV1.from_payload(payload["prior_scales"]),
            kalman_jitter=CanonicalFloat64V1.from_payload(payload["kalman_jitter"]),
            singular_floor=CanonicalFloat64V1.from_payload(payload["singular_floor"]),
        )

    def payload(self) -> Mapping[str, Any]:
        return {
            "schema": self.schema,
            "target_route": self.target_route,
            "score_validity_route": self.score_validity_route,
            "observations": self.observations.payload(),
            "parameter_names": self.parameter_names,
            "state_dim": self.state_dim,
            "observation_dim": self.observation_dim,
            "parameter_dim": self.parameter_dim,
            "rho_max": self.rho_max.payload(),
            "lower_scale": self.lower_scale.payload(),
            "truth_diag_a": self.truth_diag_a.payload(),
            "truth_lower_a": self.truth_lower_a.payload(),
            "truth_process_std": self.truth_process_std.payload(),
            "truth_observation_std": self.truth_observation_std.payload(),
            "prior_scales": self.prior_scales.payload(),
            "kalman_jitter": self.kalman_jitter.payload(),
            "singular_floor": self.singular_floor.payload(),
        }

    @property
    def identity_hash(self) -> str:
        return _identity_hash(self.payload())


@dataclass(frozen=True)
class FrozenHMCTransformIdentityV1:
    """Mechanical identity of one validated affine adapter layer."""

    layer_index: int
    runtime_route: str
    target_scope: str
    dimension: int
    factor_orientation: str
    log_jacobian_convention: str
    center: CanonicalArrayIdentityV1
    factor: CanonicalArrayIdentityV1
    schema: str = FROZEN_HMC_TRANSFORM_IDENTITY_SCHEMA_V1

    def __post_init__(self) -> None:
        if self.schema != FROZEN_HMC_TRANSFORM_IDENTITY_SCHEMA_V1:
            raise ValueError("unsupported HMC transform identity schema")
        index = _require_int(self.layer_index, label="layer_index", minimum=0)
        dimension = _require_int(self.dimension, label="dimension", minimum=1)
        route = _require_nonempty(self.runtime_route, label="transform runtime_route")
        scope = _require_nonempty(self.target_scope, label="transform target_scope")
        if self.factor_orientation != "row_right_transpose":
            raise ValueError("unsupported transform factor orientation")
        if self.log_jacobian_convention not in {"constant_omitted", "constant_included"}:
            raise ValueError("unsupported transform log-Jacobian convention")
        if self.center.shape != (dimension,) or self.factor.shape != (dimension, dimension):
            raise ValueError("transform arrays must match dimension")
        if self.center.semantic_dtype != "float64" or self.factor.semantic_dtype != "float64":
            raise ValueError("HMC transform arrays must be float64")
        object.__setattr__(self, "layer_index", index)
        object.__setattr__(self, "dimension", dimension)
        object.__setattr__(self, "runtime_route", route)
        object.__setattr__(self, "target_scope", scope)

    @classmethod
    def from_adapter(cls, adapter: Any, *, layer_index: int) -> "FrozenHMCTransformIdentityV1":
        transform = getattr(adapter, "transform", None)
        if transform is None:
            raise ValueError("HMC transform adapter is missing transform")
        route = getattr(adapter, "runtime_backend", None)
        if not route:
            raise ValueError("HMC transform adapter is missing runtime_backend")
        return cls(
            layer_index=layer_index,
            runtime_route=route,
            target_scope=getattr(adapter, "target_scope", None),
            dimension=getattr(adapter, "parameter_dim", None),
            factor_orientation=getattr(transform, "factor_orientation", None),
            log_jacobian_convention=getattr(transform, "log_jacobian_convention", None),
            center=CanonicalArrayIdentityV1.from_array(transform.center),
            factor=CanonicalArrayIdentityV1.from_array(transform.factor),
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "FrozenHMCTransformIdentityV1":
        fields = (
            "schema",
            "layer_index",
            "runtime_route",
            "target_scope",
            "dimension",
            "factor_orientation",
            "log_jacobian_convention",
            "center",
            "factor",
        )
        _require_exact_keys(payload, required=fields, label="HMC transform identity")
        return cls(
            schema=payload["schema"],
            layer_index=payload["layer_index"],
            runtime_route=payload["runtime_route"],
            target_scope=payload["target_scope"],
            dimension=payload["dimension"],
            factor_orientation=payload["factor_orientation"],
            log_jacobian_convention=payload["log_jacobian_convention"],
            center=CanonicalArrayIdentityV1.from_payload(payload["center"]),
            factor=CanonicalArrayIdentityV1.from_payload(payload["factor"]),
        )

    def payload(self) -> Mapping[str, Any]:
        return {
            "schema": self.schema,
            "layer_index": self.layer_index,
            "runtime_route": self.runtime_route,
            "target_scope": self.target_scope,
            "dimension": self.dimension,
            "factor_orientation": self.factor_orientation,
            "log_jacobian_convention": self.log_jacobian_convention,
            "center": self.center.payload(),
            "factor": self.factor.payload(),
        }


@dataclass(frozen=True)
class FrozenHMCTransitionIdentityV1:
    """Semantic identity of one retained fixed-kernel HMC transition."""

    target_scope: str
    target: DeterministicLGSSMTargetIdentityV1
    transforms: tuple[FrozenHMCTransformIdentityV1, ...]
    step_size: CanonicalFloat64V1
    num_leapfrog_steps: int
    kernel_family: str = TFP_HMC_KERNEL_FAMILY_V1
    integrator_route: str = TFP_HMC_INTEGRATOR_ROUTE_V1
    state_dtype: str = "float64"
    schema: str = FROZEN_HMC_TRANSITION_IDENTITY_SCHEMA_V1

    def __post_init__(self) -> None:
        if self.schema != FROZEN_HMC_TRANSITION_IDENTITY_SCHEMA_V1:
            raise ValueError("unsupported HMC transition identity schema")
        if self.kernel_family != TFP_HMC_KERNEL_FAMILY_V1:
            raise ValueError("unsupported HMC kernel family")
        if self.integrator_route != TFP_HMC_INTEGRATOR_ROUTE_V1:
            raise ValueError("unsupported HMC integrator route")
        scope = _require_nonempty(self.target_scope, label="target_scope")
        if self.state_dtype != "float64":
            raise ValueError("HMC transition V1 requires float64 state dtype")
        transforms = tuple(self.transforms)
        if len(transforms) != 2:
            raise ValueError("HMC transition V1 requires exactly two transforms")
        if tuple(item.layer_index for item in transforms) != (0, 1):
            raise ValueError("HMC transition transform order must be (0, 1)")
        if any(item.dimension != self.target.parameter_dim for item in transforms):
            raise ValueError("HMC transition transform dimensions must match target")
        if any(item.target_scope != scope for item in transforms):
            raise ValueError("HMC transition transform scopes must match target")
        leapfrog = _require_int(
            self.num_leapfrog_steps,
            label="num_leapfrog_steps",
            minimum=1,
        )
        if self.step_size.value <= 0.0:
            raise ValueError("step_size must be positive")
        object.__setattr__(self, "target_scope", scope)
        object.__setattr__(self, "state_dtype", "float64")
        object.__setattr__(self, "transforms", transforms)
        object.__setattr__(self, "num_leapfrog_steps", leapfrog)

    @classmethod
    def from_replay(cls, replay: Any) -> "FrozenHMCTransitionIdentityV1":
        final_adapter = getattr(replay, "adapter", None)
        final_kernel = getattr(replay, "final_kernel_payload", None)
        contract = getattr(replay, "contract", None)
        if final_adapter is None or not isinstance(final_kernel, Mapping):
            raise ValueError("replay is missing final adapter or kernel payload")
        if not isinstance(contract, Mapping):
            raise ValueError("replay is missing reconstruction contract")
        phase4_adapter = getattr(final_adapter, "base_adapter", None)
        base_adapter = getattr(phase4_adapter, "base_adapter", None)
        if phase4_adapter is None or base_adapter is None:
            raise ValueError("replay must contain exactly two adapter transforms")
        if getattr(base_adapter, "base_adapter", None) is not None:
            raise ValueError("replay contains unexpected adapter transform depth")
        target = DeterministicLGSSMTargetIdentityV1.from_adapter(base_adapter)
        scope = _require_nonempty(contract.get("target_scope"), label="target_scope")
        if _require_int(
            contract.get("target_dimension"),
            label="target_dimension",
            minimum=1,
        ) != target.parameter_dim:
            raise ValueError("replay target dimension does not match live target")
        capability_fn = getattr(base_adapter, "value_score_capability", None)
        capability = capability_fn() if callable(capability_fn) else None
        if capability is None:
            raise ValueError("base target is missing value/score capability")
        if getattr(capability, "runtime_backend", None) != (
            "tensorflow_manual_lgssm_svd_graph_status_score"
        ):
            raise ValueError("base target runtime backend mismatch")
        capability_scope = _require_nonempty(
            getattr(capability, "target_scope", None),
            label="base target capability scope",
        )
        if capability_scope != scope:
            raise ValueError("base target scope does not match replay")
        if getattr(base_adapter, "parameter_dim", None) != target.parameter_dim:
            raise ValueError("base adapter dimension does not match target")
        if "step_size" not in final_kernel or "num_leapfrog_steps" not in final_kernel:
            raise ValueError("replay final kernel mechanics are missing")
        return cls(
            target_scope=scope,
            target=target,
            transforms=(
                FrozenHMCTransformIdentityV1.from_adapter(phase4_adapter, layer_index=0),
                FrozenHMCTransformIdentityV1.from_adapter(final_adapter, layer_index=1),
            ),
            step_size=CanonicalFloat64V1.from_value(final_kernel["step_size"]),
            num_leapfrog_steps=final_kernel["num_leapfrog_steps"],
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "FrozenHMCTransitionIdentityV1":
        fields = (
            "schema",
            "kernel_family",
            "integrator_route",
            "target_scope",
            "state_dtype",
            "target",
            "transforms",
            "step_size",
            "num_leapfrog_steps",
        )
        _require_exact_keys(payload, required=fields, label="HMC transition identity")
        transforms = payload["transforms"]
        if not isinstance(transforms, Sequence) or isinstance(transforms, (str, bytes)):
            raise ValueError("HMC transition transforms must be a sequence")
        return cls(
            schema=payload["schema"],
            kernel_family=payload["kernel_family"],
            integrator_route=payload["integrator_route"],
            target_scope=payload["target_scope"],
            state_dtype=payload["state_dtype"],
            target=DeterministicLGSSMTargetIdentityV1.from_payload(payload["target"]),
            transforms=tuple(FrozenHMCTransformIdentityV1.from_payload(item) for item in transforms),
            step_size=CanonicalFloat64V1.from_payload(payload["step_size"]),
            num_leapfrog_steps=payload["num_leapfrog_steps"],
        )

    def payload(self) -> Mapping[str, Any]:
        return {
            "schema": self.schema,
            "kernel_family": self.kernel_family,
            "integrator_route": self.integrator_route,
            "target_scope": self.target_scope,
            "state_dtype": self.state_dtype,
            "target": self.target.payload(),
            "transforms": tuple(item.payload() for item in self.transforms),
            "step_size": self.step_size.payload(),
            "num_leapfrog_steps": self.num_leapfrog_steps,
        }

    @property
    def identity_hash(self) -> str:
        return _identity_hash(self.payload())


@dataclass(frozen=True)
class FrozenHMCExecutionContractV1:
    """Exact deterministic Phase 7 controller semantics for one transition."""

    transition_identity_hash: str
    run_mode: str
    global_initial_state: CanonicalArrayIdentityV1
    root_seed: tuple[int, int]
    worker_count: int
    chains_per_worker: int
    burnin_initial: int
    burnin_extension: int
    burnin_check_window: int
    burnin_maximum: int
    retained_initial: int
    retained_extension: int
    retained_check_interval: int
    retained_maximum: int
    max_chunk_results: int
    rhat_max: CanonicalFloat64V1
    bulk_ess_min: CanonicalFloat64V1
    tail_ess_min: CanonicalFloat64V1
    wall_time_cap_seconds: int
    thread_environment: tuple[tuple[str, str], ...]
    tensorflow_version: str
    tfp_version: str
    python_version: str
    initial_state_policy: str = PHASE7_INITIAL_STATE_POLICY_V1
    seed_derivation_route: str = PHASE7_SEED_DERIVATION_ROUTE_V1
    seed_dtype: str = "int32"
    burnin_stage_index: int = 1
    retained_stage_index: int = 2
    compile_probe_stage_index: int = 1
    compile_probe_check_index: int = 9999
    compile_probe_advances_state: bool = False
    multiprocessing_start_method: str = "spawn"
    persistent_workers: bool = True
    contiguous_global_chain_order: bool = True
    worker_partition_policy: str = PHASE7_WORKER_PARTITION_POLICY_V1
    compile_workers_sequentially: bool = True
    cuda_visible_devices: str = "-1"
    jit_compile: bool = True
    use_xla: bool = True
    chain_execution_mode: str = "tf_function"
    num_burnin_steps_per_chunk: int = 0
    trace_policy: str = "reduced"
    target_status_trace_policy: str = "none"
    diagnostic_coordinate_system: str = (
        "raw_lgssm_parameters_after_two_mass_transforms"
    )
    diagnostic_route: str = PHASE7_DIAGNOSTIC_ROUTE_V1
    diagnostic_gate_policy: str = PHASE7_SERIOUS_DIAGNOSTIC_GATE_V1
    diagnostic_accumulation_policy: str = (
        PHASE7_DIAGNOSTIC_ACCUMULATION_POLICY_V1
    )
    all_parameters_required: bool = True
    worker_hard_veto_policy: str = PHASE7_WORKER_HARD_VETO_POLICY_V1
    controller_policy: str = PHASE7_CONTROLLER_POLICY_V1
    no_resume_policy: str = PHASE7_NO_RESUME_POLICY_V1
    manual_thinning_allowed: bool = False
    manual_chain_exclusion_allowed: bool = False
    manual_extension_allowed: bool = False
    schema: str = FROZEN_HMC_EXECUTION_CONTRACT_SCHEMA_V1

    def __post_init__(self) -> None:
        if self.schema != FROZEN_HMC_EXECUTION_CONTRACT_SCHEMA_V1:
            raise ValueError("unsupported HMC execution contract schema")
        transition_hash = _require_sha256(
            self.transition_identity_hash,
            label="transition_identity_hash",
        )
        if self.run_mode not in {"smoke", "serious"}:
            raise ValueError("run_mode must be smoke or serious")
        worker_count = _require_int(self.worker_count, label="worker_count", minimum=1)
        chains_per_worker = _require_int(
            self.chains_per_worker,
            label="chains_per_worker",
            minimum=1,
        )
        root_seed = tuple(
            _require_int(item, label="root_seed entry", minimum=0) for item in self.root_seed
        )
        if len(root_seed) != 2:
            raise ValueError("root_seed must contain two integers")
        if (
            self.global_initial_state.semantic_dtype != "float64"
            or len(self.global_initial_state.shape) != 2
            or self.global_initial_state.shape[0] != worker_count * chains_per_worker
            or self.global_initial_state.shape[1] != 18
        ):
            raise ValueError("global initial state must bind every float64 chain")
        count_fields = (
            "burnin_initial",
            "burnin_extension",
            "burnin_check_window",
            "burnin_maximum",
            "retained_initial",
            "retained_extension",
            "retained_check_interval",
            "retained_maximum",
        )
        counts = {
            name: _require_int(getattr(self, name), label=name, minimum=1)
            for name in count_fields
        }
        if counts["burnin_maximum"] < counts["burnin_initial"] or (
            counts["retained_maximum"] < counts["retained_initial"]
        ):
            raise ValueError("controller maxima must cover initial chunks")
        if self.rhat_max.value <= 0.0 or self.bulk_ess_min.value <= 0.0 or (
            self.tail_ess_min.value <= 0.0
        ):
            raise ValueError("diagnostic thresholds must be positive")
        wall_cap = _require_int(
            self.wall_time_cap_seconds,
            label="wall_time_cap_seconds",
            minimum=1,
        )
        compile_probe = _require_int(
            self.compile_probe_check_index,
            label="compile_probe_check_index",
            minimum=0,
        )
        expected_constants = {
            "initial_state_policy": PHASE7_INITIAL_STATE_POLICY_V1,
            "seed_derivation_route": PHASE7_SEED_DERIVATION_ROUTE_V1,
            "seed_dtype": "int32",
            "multiprocessing_start_method": "spawn",
            "worker_partition_policy": PHASE7_WORKER_PARTITION_POLICY_V1,
            "cuda_visible_devices": "-1",
            "chain_execution_mode": "tf_function",
            "trace_policy": "reduced",
            "target_status_trace_policy": "none",
            "diagnostic_route": PHASE7_DIAGNOSTIC_ROUTE_V1,
            "diagnostic_accumulation_policy": (
                PHASE7_DIAGNOSTIC_ACCUMULATION_POLICY_V1
            ),
            "worker_hard_veto_policy": PHASE7_WORKER_HARD_VETO_POLICY_V1,
            "controller_policy": PHASE7_CONTROLLER_POLICY_V1,
            "no_resume_policy": PHASE7_NO_RESUME_POLICY_V1,
        }
        for name, expected in expected_constants.items():
            if getattr(self, name) != expected:
                raise ValueError(f"unsupported Phase 7 {name}")
        expected_gate = (
            PHASE7_SMOKE_DIAGNOSTIC_GATE_V1
            if self.run_mode == "smoke"
            else PHASE7_SERIOUS_DIAGNOSTIC_GATE_V1
        )
        if self.diagnostic_gate_policy != expected_gate:
            raise ValueError("diagnostic_gate_policy does not match run_mode")
        expected_stage_indices = {
            "burnin_stage_index": 1,
            "retained_stage_index": 2,
            "compile_probe_stage_index": 1,
        }
        for name, expected in expected_stage_indices.items():
            if _require_int(getattr(self, name), label=name, minimum=0) != expected:
                raise ValueError(f"unsupported Phase 7 {name}")
        max_chunk_results = _require_int(
            self.max_chunk_results,
            label="max_chunk_results",
            minimum=1,
        )
        expected_max_chunk = max(
            counts["burnin_initial"],
            counts["burnin_extension"],
            counts["retained_initial"],
            counts["retained_extension"],
        )
        if max_chunk_results != expected_max_chunk:
            raise ValueError("max_chunk_results does not match controller chunks")
        required_true = (
            "persistent_workers",
            "contiguous_global_chain_order",
            "compile_workers_sequentially",
            "jit_compile",
            "use_xla",
            "all_parameters_required",
        )
        required_false = (
            "compile_probe_advances_state",
            "manual_thinning_allowed",
            "manual_chain_exclusion_allowed",
            "manual_extension_allowed",
        )
        for name in required_true:
            if _require_bool(getattr(self, name), label=name) is not True:
                raise ValueError(f"Phase 7 requires {name}=true")
        for name in required_false:
            if _require_bool(getattr(self, name), label=name) is not False:
                raise ValueError(f"Phase 7 requires {name}=false")
        if _require_int(
            self.num_burnin_steps_per_chunk,
            label="num_burnin_steps_per_chunk",
            minimum=0,
        ) != 0:
            raise ValueError("Phase 7 chunks require num_burnin_steps=0")
        environment = tuple(
            (
                _require_nonempty(key, label="thread_environment key"),
                _require_nonempty(value, label=f"thread_environment[{key}]"),
            )
            for key, value in self.thread_environment
        )
        if len(dict(environment)) != len(environment):
            raise ValueError("thread_environment contains duplicate keys")
        if tuple(sorted(environment)) != environment:
            raise ValueError("thread_environment must use canonical key order")
        expected_environment_keys = {
            "CUDA_VISIBLE_DEVICES",
            "MKL_NUM_THREADS",
            "MPLCONFIGDIR",
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "TF_CPP_MIN_LOG_LEVEL",
            "TF_NUM_INTEROP_THREADS",
            "TF_NUM_INTRAOP_THREADS",
        }
        if set(dict(environment)) != expected_environment_keys:
            raise ValueError("thread_environment must bind the complete worker environment")
        for name in ("tensorflow_version", "tfp_version", "python_version"):
            _require_nonempty(getattr(self, name), label=name)
        object.__setattr__(self, "transition_identity_hash", transition_hash)
        object.__setattr__(self, "root_seed", root_seed)
        object.__setattr__(self, "worker_count", worker_count)
        object.__setattr__(self, "chains_per_worker", chains_per_worker)
        object.__setattr__(self, "wall_time_cap_seconds", wall_cap)
        object.__setattr__(self, "compile_probe_check_index", compile_probe)
        object.__setattr__(self, "max_chunk_results", max_chunk_results)
        object.__setattr__(self, "thread_environment", environment)
        for name, value in counts.items():
            object.__setattr__(self, name, value)

    @classmethod
    def from_phase7_config(
        cls,
        *,
        transition: FrozenHMCTransitionIdentityV1,
        config: Any,
        smoke: bool,
        tensorflow_version: str,
        tfp_version: str,
        python_version: str,
    ) -> "FrozenHMCExecutionContractV1":
        payload = getattr(config, "payload", None)
        if not isinstance(payload, Mapping):
            raise TypeError("Phase 7 config must expose a payload mapping")
        execution = payload.get("execution")
        burnin = payload.get("burnin")
        retained = payload.get("retained")
        diagnostics = payload.get("diagnostics")
        if not all(isinstance(item, Mapping) for item in (execution, burnin, retained, diagnostics)):
            raise ValueError("Phase 7 config is missing execution sections")
        worker_count = _require_int(execution.get("worker_count"), label="worker_count", minimum=1)
        chains_per_worker = _require_int(
            execution.get("chains_per_worker"),
            label="chains_per_worker",
            minimum=1,
        )
        chain_count = worker_count * chains_per_worker
        offsets = np.linspace(-0.15, 0.15, chain_count, dtype=np.float64)
        pattern = 1.0 - 2.0 * (np.arange(transition.target.parameter_dim) % 2)
        initial_state = offsets[:, None] * pattern[None, :]
        if smoke:
            counts = (4, 4, 4, 4, 8, 8, 8, 8)
            bulk_ess = 1.0
            tail_ess = 1.0
        else:
            counts = (
                burnin.get("initial_results_per_chain"),
                burnin.get("extension_results_per_chain"),
                burnin.get("check_window_results_per_chain"),
                burnin.get("max_results_per_chain"),
                retained.get("initial_results_per_chain"),
                retained.get("extension_results_per_chain"),
                retained.get("check_interval_results_per_chain"),
                retained.get("max_results_per_chain"),
            )
            bulk_ess = diagnostics.get("bulk_ess_min")
            tail_ess = diagnostics.get("tail_ess_min")
        return cls(
            transition_identity_hash=transition.identity_hash,
            run_mode="smoke" if smoke else "serious",
            global_initial_state=CanonicalArrayIdentityV1.from_array(initial_state),
            root_seed=tuple(execution.get("root_seed", ())),
            worker_count=worker_count,
            chains_per_worker=chains_per_worker,
            burnin_initial=counts[0],
            burnin_extension=counts[1],
            burnin_check_window=counts[2],
            burnin_maximum=counts[3],
            retained_initial=counts[4],
            retained_extension=counts[5],
            retained_check_interval=counts[6],
            retained_maximum=counts[7],
            max_chunk_results=max(counts[0], counts[1], counts[4], counts[5]),
            rhat_max=CanonicalFloat64V1.from_value(diagnostics.get("rhat_max")),
            bulk_ess_min=CanonicalFloat64V1.from_value(bulk_ess),
            tail_ess_min=CanonicalFloat64V1.from_value(tail_ess),
            wall_time_cap_seconds=execution.get("wall_time_cap_seconds"),
            thread_environment=tuple(
                sorted(
                    {
                        **dict(execution.get("thread_environment", {})),
                        "CUDA_VISIBLE_DEVICES": "-1",
                        "MPLCONFIGDIR": "/tmp/matplotlib-bayesfilter-phase7-worker",
                        "TF_CPP_MIN_LOG_LEVEL": "1",
                    }.items()
                )
            ),
            tensorflow_version=tensorflow_version,
            tfp_version=tfp_version,
            python_version=python_version,
            compile_workers_sequentially=execution.get("compile_workers_sequentially"),
            cuda_visible_devices=execution.get("cuda_visible_devices"),
            jit_compile=execution.get("jit_compile"),
            use_xla=execution.get("use_xla"),
            chain_execution_mode=execution.get("chain_execution_mode"),
            diagnostic_coordinate_system=diagnostics.get("coordinate_system"),
            diagnostic_gate_policy=(
                PHASE7_SMOKE_DIAGNOSTIC_GATE_V1
                if smoke
                else PHASE7_SERIOUS_DIAGNOSTIC_GATE_V1
            ),
            all_parameters_required=diagnostics.get("all_parameters_required"),
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "FrozenHMCExecutionContractV1":
        fields = tuple(cls.__dataclass_fields__)
        _require_exact_keys(payload, required=fields, label="HMC execution contract")
        array_fields = {"global_initial_state"}
        float_fields = {"rhat_max", "bulk_ess_min", "tail_ess_min"}
        values = dict(payload)
        for name in array_fields:
            values[name] = CanonicalArrayIdentityV1.from_payload(values[name])
        for name in float_fields:
            values[name] = CanonicalFloat64V1.from_payload(values[name])
        values["root_seed"] = tuple(values["root_seed"])
        values["thread_environment"] = tuple(tuple(item) for item in values["thread_environment"])
        return cls(**values)

    def payload(self) -> Mapping[str, Any]:
        result: dict[str, Any] = {}
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if isinstance(value, (CanonicalArrayIdentityV1, CanonicalFloat64V1)):
                value = value.payload()
            result[name] = value
        return result

    @property
    def identity_hash(self) -> str:
        return _identity_hash(self.payload())


@dataclass(frozen=True)
class SelectionStageIdentityV1:
    """One typed selection-lineage reference without embedding source payloads."""

    stage_id: str
    source_schema: str
    canonical_payload_hash: str
    selected_index: int | None = None
    schema: str = SELECTION_STAGE_IDENTITY_SCHEMA_V1

    def __post_init__(self) -> None:
        if self.schema != SELECTION_STAGE_IDENTITY_SCHEMA_V1:
            raise ValueError("unsupported selection stage identity schema")
        if self.stage_id not in _ALLOWED_SELECTION_STAGES:
            raise ValueError("unsupported selection stage_id")
        _require_nonempty(self.source_schema, label="selection stage source_schema")
        _require_sha256(
            self.canonical_payload_hash,
            label="selection stage canonical_payload_hash",
        )
        if self.selected_index is not None:
            object.__setattr__(
                self,
                "selected_index",
                _require_int(self.selected_index, label="selected_index", minimum=0),
            )

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "SelectionStageIdentityV1":
        _require_exact_keys(
            payload,
            required=tuple(cls.__dataclass_fields__),
            label="selection stage identity",
        )
        return cls(**dict(payload))

    def payload(self) -> Mapping[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class SelectionProvenanceIdentityV1:
    """Typed identity for full selection history and named lineage."""

    source_selection_schema: str
    source_selection_payload_hash: str
    tuning_config_hash: str
    final_status: str
    stage_lineage: tuple[SelectionStageIdentityV1, ...]
    selected_step_hash: str
    selected_trajectory_hash: str
    review_record_hashes: tuple[str, ...] = ()
    schema: str = SELECTION_PROVENANCE_IDENTITY_SCHEMA_V1

    def __post_init__(self) -> None:
        if self.schema != SELECTION_PROVENANCE_IDENTITY_SCHEMA_V1:
            raise ValueError("unsupported selection provenance schema")
        _require_nonempty(self.source_selection_schema, label="source_selection_schema")
        _require_sha256(
            self.source_selection_payload_hash,
            label="source_selection_payload_hash",
        )
        _require_sha256(self.tuning_config_hash, label="tuning_config_hash")
        _require_nonempty(self.final_status, label="final_status")
        lineage = tuple(self.stage_lineage)
        if not lineage:
            raise ValueError("selection stage_lineage must be non-empty")
        stage_ids = tuple(item.stage_id for item in lineage)
        if len(set(stage_ids)) != len(stage_ids):
            raise ValueError("selection stage_lineage contains duplicate stage IDs")
        _require_sha256(self.selected_step_hash, label="selected_step_hash", tagged=False)
        _require_sha256(
            self.selected_trajectory_hash,
            label="selected_trajectory_hash",
            tagged=False,
        )
        reviews = tuple(
            _require_sha256(item, label="review_record_hash")
            for item in self.review_record_hashes
        )
        object.__setattr__(self, "stage_lineage", lineage)
        object.__setattr__(self, "review_record_hashes", reviews)

    @classmethod
    def from_source_payload(
        cls,
        *,
        source_selection_payload: Mapping[str, Any],
        tuning_config_hash: str,
        stage_lineage: Sequence[SelectionStageIdentityV1],
        selected_step_hash: str,
        selected_trajectory_hash: str,
        review_record_hashes: Sequence[str] = (),
    ) -> "SelectionProvenanceIdentityV1":
        source_schema = source_selection_payload.get("schema")
        return cls(
            source_selection_schema=_require_nonempty(
                source_schema,
                label="source selection schema",
            ),
            source_selection_payload_hash=canonical_artifact_payload_hash(
                source_selection_payload
            ),
            tuning_config_hash=tuning_config_hash,
            final_status=_require_nonempty(
                source_selection_payload.get("final_status"),
                label="source selection final_status",
            ),
            stage_lineage=tuple(stage_lineage),
            selected_step_hash=selected_step_hash,
            selected_trajectory_hash=selected_trajectory_hash,
            review_record_hashes=tuple(review_record_hashes),
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "SelectionProvenanceIdentityV1":
        _require_exact_keys(
            payload,
            required=tuple(cls.__dataclass_fields__),
            label="selection provenance identity",
        )
        values = dict(payload)
        values["stage_lineage"] = tuple(
            SelectionStageIdentityV1.from_payload(item) for item in values["stage_lineage"]
        )
        values["review_record_hashes"] = tuple(values["review_record_hashes"])
        return cls(**values)

    def payload(self) -> Mapping[str, Any]:
        return {
            "schema": self.schema,
            "source_selection_schema": self.source_selection_schema,
            "source_selection_payload_hash": self.source_selection_payload_hash,
            "tuning_config_hash": self.tuning_config_hash,
            "final_status": self.final_status,
            "stage_lineage": tuple(item.payload() for item in self.stage_lineage),
            "selected_step_hash": self.selected_step_hash,
            "selected_trajectory_hash": self.selected_trajectory_hash,
            "review_record_hashes": self.review_record_hashes,
        }

    @property
    def identity_hash(self) -> str:
        return _identity_hash(self.payload())


def artifact_file_sha256(path: str | Path) -> str:
    """Return the exact SHA-256 for a serialized artifact file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_artifact_payload_hash(payload: Mapping[str, Any]) -> str:
    """Return strict canonical full-payload integrity without projection."""

    if not isinstance(payload, Mapping):
        raise TypeError("artifact payload must be a mapping")
    return f"sha256:{_canonical_json_hash(payload)}"
