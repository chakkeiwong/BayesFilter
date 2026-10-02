"""Fixed-shape metadata for promoted compiled nonlinear value paths."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import tensorflow as tf

from bayesfilter.inference import RegularizationConvention
from bayesfilter.structural_tf import TFStructuralStateSpace

CompiledValuePathClassification = Literal[
    "production_compiled_value",
    "testing_reference",
    "blocked",
]
CompiledValuePathMode = Literal["eager", "tf_function", "xla"]

_KNOWN_VALUE_BACKENDS = {"tf_svd_cubature", "tf_svd_ukf", "tf_svd_cut4"}
_KNOWN_CLASSIFICATIONS = {
    "production_compiled_value",
    "testing_reference",
    "blocked",
}
_KNOWN_COMPILE_MODES = {"eager", "tf_function", "xla"}
_FORBIDDEN_SOURCE_PATTERNS: Mapping[str, re.Pattern[str]] = {
    "tf.numpy_function": re.compile(r"\btf\.numpy_function\b"),
    "tf.py_function": re.compile(r"\btf\.py_function\b"),
    ".numpy(": re.compile(r"\.numpy\s*\("),
    "numpy": re.compile(r"\b(?:np|numpy)\."),
    "scipy": re.compile(r"\bscipy\."),
}
_PROCESS_LOCAL_SIGNATURE_PATTERNS = (
    re.compile(r"\b0x[0-9a-fA-F]+\b"),
    re.compile(r"\bobject at\b"),
    re.compile(r"\bid\s*\("),
)


class InvalidCompiledValuePathContract(ValueError):
    """Raised when a promoted value path has incomplete or unsafe metadata."""


@dataclass(frozen=True)
class NonlinearFilterValueStaticShape:
    """Fixed tensor dimensions for a promoted nonlinear filter value path."""

    horizon: int
    state_dim: int
    observation_dim: int
    innovation_dim: int
    augmented_dim: int
    observation_shape: tuple[int, int]
    return_filtered: bool

    def __post_init__(self) -> None:
        for name in (
            "horizon",
            "state_dim",
            "observation_dim",
            "innovation_dim",
            "augmented_dim",
        ):
            value = int(getattr(self, name))
            if value <= 0:
                raise InvalidCompiledValuePathContract(f"{name} must be positive")
            object.__setattr__(self, name, value)
        shape = tuple(int(dim) for dim in self.observation_shape)
        if shape != (self.horizon, self.observation_dim):
            raise InvalidCompiledValuePathContract(
                "observation_shape must equal (horizon, observation_dim)"
            )
        if self.augmented_dim != self.state_dim + self.innovation_dim:
            raise InvalidCompiledValuePathContract(
                "augmented_dim must equal state_dim + innovation_dim"
            )
        object.__setattr__(self, "observation_shape", shape)
        object.__setattr__(self, "return_filtered", bool(self.return_filtered))

    def signature_payload(self) -> Mapping[str, Any]:
        return {
            "horizon": self.horizon,
            "state_dim": self.state_dim,
            "observation_dim": self.observation_dim,
            "innovation_dim": self.innovation_dim,
            "augmented_dim": self.augmented_dim,
            "observation_shape": self.observation_shape,
            "return_filtered": self.return_filtered,
        }


@dataclass(frozen=True)
class NonlinearFilterValuePathContract:
    """Value-only certification metadata for fixed-shape compiled filters."""

    fixture_name: str
    model_name: str
    backend: str
    filter_implementation: str
    classification: CompiledValuePathClassification
    compile_mode: CompiledValuePathMode
    dtype: str
    static_shape: NonlinearFilterValueStaticShape
    regularization: RegularizationConvention
    source_paths: tuple[str, ...]
    value_authority: str = "graph_native_value_only"
    dynamic_horizon_status: str = "not_claimed"
    score_status: str = "not_claimed"
    hmc_status: str = "not_claimed"
    nonclaims: tuple[str, ...] = (
        "value-path compile is not score evidence",
        "value-path compile is not full-chain HMC evidence",
        "fixed-shape evidence is not dynamic-horizon evidence",
    )

    def __post_init__(self) -> None:
        backend = str(self.backend)
        classification = str(self.classification)
        compile_mode = str(self.compile_mode)
        if backend not in _KNOWN_VALUE_BACKENDS:
            raise InvalidCompiledValuePathContract(f"unknown value backend: {backend}")
        if classification not in _KNOWN_CLASSIFICATIONS:
            raise InvalidCompiledValuePathContract(
                f"unknown value-path classification: {classification}"
            )
        if compile_mode not in _KNOWN_COMPILE_MODES:
            raise InvalidCompiledValuePathContract(
                f"unknown value-path compile_mode: {compile_mode}"
            )
        if classification == "production_compiled_value" and compile_mode != "xla":
            raise InvalidCompiledValuePathContract(
                "production_compiled_value requires compile_mode='xla'"
            )
        source_paths = tuple(str(path) for path in self.source_paths)
        if not source_paths:
            raise InvalidCompiledValuePathContract("source_paths must be nonempty")
        for name in (
            "fixture_name",
            "model_name",
            "filter_implementation",
            "dtype",
            "value_authority",
            "dynamic_horizon_status",
            "score_status",
            "hmc_status",
        ):
            if not str(getattr(self, name)).strip():
                raise InvalidCompiledValuePathContract(f"{name} must be nonempty")
        object.__setattr__(self, "backend", backend)
        object.__setattr__(self, "classification", classification)
        object.__setattr__(self, "compile_mode", compile_mode)
        object.__setattr__(self, "source_paths", source_paths)
        object.__setattr__(self, "nonclaims", tuple(str(item) for item in self.nonclaims))
        _reject_process_local_identity(self.signature_payload())

    @property
    def is_promoted_fixed_shape_value_path(self) -> bool:
        return bool(
            self.classification == "production_compiled_value"
            and self.compile_mode == "xla"
            and self.dynamic_horizon_status == "not_claimed"
            and self.score_status == "not_claimed"
            and self.hmc_status == "not_claimed"
        )

    def signature_payload(self) -> Mapping[str, Any]:
        return {
            "fixture_name": self.fixture_name,
            "model_name": self.model_name,
            "backend": self.backend,
            "filter_implementation": self.filter_implementation,
            "classification": self.classification,
            "compile_mode": self.compile_mode,
            "dtype": self.dtype,
            "static_shape": self.static_shape.signature_payload(),
            "regularization": self.regularization.signature_payload(),
            "source_paths": self.source_paths,
            "value_authority": self.value_authority,
            "dynamic_horizon_status": self.dynamic_horizon_status,
            "score_status": self.score_status,
            "hmc_status": self.hmc_status,
            "nonclaims": self.nonclaims,
        }


def tensorflow_nonlinear_value_path_contract(
    *,
    fixture_name: str,
    observations: tf.Tensor,
    model: TFStructuralStateSpace,
    backend: str,
    return_filtered: bool,
    compile_mode: CompiledValuePathMode = "xla",
    classification: CompiledValuePathClassification = "production_compiled_value",
    innovation_floor: float = 1e-12,
    jitter: float = 0.0,
) -> NonlinearFilterValuePathContract:
    """Build fixed-shape value-only metadata for a promoted TF filter path."""

    obs = tf.convert_to_tensor(observations, dtype=tf.float64)
    if obs.shape.rank == 1:
        obs = obs[:, tf.newaxis]
    if obs.shape.rank != 2:
        raise InvalidCompiledValuePathContract("observations must be rank 1 or 2")
    horizon = obs.shape[0]
    observation_dim = obs.shape[1]
    if horizon is None or observation_dim is None:
        raise InvalidCompiledValuePathContract(
            "compiled value paths require static observation shape"
        )
    if model.state_dim is None or model.innovation_dim is None:
        raise InvalidCompiledValuePathContract("model dimensions must be static")
    if int(observation_dim) != int(model.observation_dim):
        raise InvalidCompiledValuePathContract(
            "observation shape must match model observation_dim"
        )
    backend = str(backend)
    implementation = (
        "tf_svd_cut4_filter" if backend == "tf_svd_cut4" else "tf_svd_sigma_point_filter"
    )
    return NonlinearFilterValuePathContract(
        fixture_name=fixture_name,
        model_name=str(model.name),
        backend=backend,
        filter_implementation=implementation,
        classification=classification,
        compile_mode=compile_mode,
        dtype=obs.dtype.name,
        static_shape=NonlinearFilterValueStaticShape(
            horizon=int(horizon),
            state_dim=int(model.state_dim),
            observation_dim=int(observation_dim),
            innovation_dim=int(model.innovation_dim),
            augmented_dim=int(model.state_dim + model.innovation_dim),
            observation_shape=(int(horizon), int(observation_dim)),
            return_filtered=return_filtered,
        ),
        regularization=RegularizationConvention(
            jitter=float(jitter),
            covariance_floor=float(innovation_floor),
            psd_repair="tf.linalg.eigh_floor",
            symmetrize=True,
            logdet_convention="implemented_regularized_covariance",
            implemented_covariance="post_floor_innovation_covariance",
            repair_role="target",
        ),
        source_paths=_source_paths_for_backend(backend),
    )


def stable_nonlinear_filter_value_path_signature(
    contract: NonlinearFilterValuePathContract,
) -> str:
    """Return a stable SHA-256 signature for fixed-shape value-path metadata."""

    if not isinstance(contract, NonlinearFilterValuePathContract):
        raise TypeError("contract must be a NonlinearFilterValuePathContract")
    payload = contract.signature_payload()
    blob = json.dumps(_normalize_for_json(payload), sort_keys=True, separators=(",", ":"))
    _reject_process_local_identity(blob)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def find_forbidden_compiled_value_tokens(source: str) -> tuple[str, ...]:
    """Return XLA-hostile host/Python bridge tokens found in source text."""

    text = str(source)
    return tuple(
        token for token, pattern in _FORBIDDEN_SOURCE_PATTERNS.items() if pattern.search(text)
    )


def _source_paths_for_backend(backend: str) -> tuple[str, ...]:
    if backend in {"tf_svd_cubature", "tf_svd_ukf"}:
        return (
            "bayesfilter/nonlinear/sigma_points_tf.py",
            "bayesfilter/linear/svd_factor_tf.py",
            "bayesfilter/structural_tf.py",
        )
    if backend == "tf_svd_cut4":
        return (
            "bayesfilter/nonlinear/svd_cut_tf.py",
            "bayesfilter/nonlinear/cut_tf.py",
            "bayesfilter/nonlinear/sigma_points_tf.py",
            "bayesfilter/linear/svd_factor_tf.py",
            "bayesfilter/structural_tf.py",
        )
    raise InvalidCompiledValuePathContract(f"unknown value backend: {backend}")


def _normalize_for_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _normalize_for_json(val) for key, val in value.items()}
    if isinstance(value, (tuple, list)):
        return [_normalize_for_json(item) for item in value]
    return value


def _reject_process_local_identity(value: Any) -> None:
    text = json.dumps(_normalize_for_json(value), sort_keys=True, default=str)
    if any(pattern.search(text) for pattern in _PROCESS_LOCAL_SIGNATURE_PATTERNS):
        raise InvalidCompiledValuePathContract(
            "compiled value-path signatures must not contain process-local identity"
        )
