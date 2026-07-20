"""Small generic HMC-runtime helpers with fail-closed authority checks."""

from __future__ import annotations

import hashlib
import json
import re
import secrets
import time
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np

from bayesfilter.inference.mass_matrix import (
    covariance_from_negative_hessian,
    whitening_from_covariance,
)
from bayesfilter.inference.hmc_tuning import (
    HMCTuningPolicy,
    normalize_hmc_tuning_policy,
    require_executable_tuning_policy,
)
from bayesfilter.inference.hmc_verification import (
    HMCAcceptancePolicy,
    _evaluate_retained_target_health,
    evaluate_hmc_acceptance_evidence,
    summarize_hmc_tuning_telemetry,
)
from bayesfilter.inference.batched_value_score import reviewed_value_score_target_fn
from bayesfilter.inference.posterior_adapter import value_score_capability


_PROCESS_LOCAL_SIGNATURE_PATTERNS = (
    re.compile(r"\b0x[0-9a-fA-F]+\b"),
    re.compile(r"\bobject at\b"),
    re.compile(r"\bid\s*\("),
)


@dataclass(frozen=True)
class PrecomputedMAP:
    """Shape- and adapter-scoped precomputed MAP/mass artifact metadata."""

    position: Any
    covariance: Any
    adapter_signature: str
    source: str = "precomputed_map"


@dataclass(frozen=True)
class PrecomputedMassArtifact:
    """Validated precomputed position/covariance/factor metadata for HMC.

    The artifact separates the role of the position from the mass matrix.  A
    caller may persist a true MAP, but diagnostic centers such as a known truth
    or prior center must use a non-MAP ``position_role`` so downstream gates do
    not silently promote initialization geometry into a MAP-quality claim.
    """

    position: Any
    covariance: Any
    factor: Any
    adapter_signature: str
    position_role: str = "map"
    covariance_source: str = "unspecified"
    matrix_used_for_square_root: str = "covariance"
    factor_orientation: str = "row_right_transpose"
    source: str = "precomputed_mass_artifact"
    eigen_summary: Mapping[str, Any] | None = None
    precision_eigen_summary: Mapping[str, Any] | None = None
    regularization_report: Mapping[str, Any] | None = None
    log_jacobian_convention: str = "constant_omitted"
    nonclaims: tuple[str, ...] = (
        "precomputed mass artifact only",
        "no posterior convergence claim",
        "no production MAP quality claim",
        "no sampler readiness claim",
    )
    reconstruction_rtol: float = 1.0e-10
    reconstruction_atol: float = 1.0e-10

    def __post_init__(self) -> None:
        position = np.asarray(self.position, dtype=float).copy()
        covariance = np.asarray(self.covariance, dtype=float).copy()
        factor = np.asarray(self.factor, dtype=float).copy()
        if position.ndim != 1:
            raise ValueError("precomputed mass position must be a one-dimensional vector")
        dim = int(position.shape[0])
        if covariance.shape != (dim, dim):
            raise ValueError("precomputed mass covariance shape must match position dimension")
        if factor.shape != (dim, dim):
            raise ValueError("precomputed mass factor shape must match position dimension")
        if not np.all(np.isfinite(position)):
            raise ValueError("precomputed mass position must be finite")
        if not np.all(np.isfinite(covariance)):
            raise ValueError("precomputed mass covariance must be finite")
        if not np.all(np.isfinite(factor)):
            raise ValueError("precomputed mass factor must be finite")
        if not np.allclose(covariance, covariance.T, rtol=self.reconstruction_rtol, atol=self.reconstruction_atol):
            raise ValueError("precomputed mass covariance must be symmetric")
        orientation = str(self.factor_orientation)
        if orientation != "row_right_transpose":
            raise ValueError("factor_orientation must be 'row_right_transpose'")
        reconstruction = factor @ factor.T
        if not np.allclose(
            reconstruction,
            covariance,
            rtol=float(self.reconstruction_rtol),
            atol=float(self.reconstruction_atol),
        ):
            raise ValueError(
                "precomputed mass factor must reconstruct covariance as factor @ factor.T"
            )
        convention = str(self.log_jacobian_convention)
        if convention not in {"constant_omitted", "constant_included"}:
            raise ValueError(
                "log_jacobian_convention must be 'constant_omitted' or 'constant_included'"
            )
        signature = _validate_persisted_signature(str(self.adapter_signature))
        position_role = str(self.position_role)
        if not position_role:
            raise ValueError("position_role must be non-empty")
        covariance_source = str(self.covariance_source)
        if not covariance_source:
            raise ValueError("covariance_source must be non-empty")
        matrix_used = str(self.matrix_used_for_square_root)
        if not matrix_used:
            raise ValueError("matrix_used_for_square_root must be non-empty")
        source = str(self.source)
        if not source:
            raise ValueError("source must be non-empty")
        actual_eigen_summary = _covariance_eigen_summary(covariance)
        if not bool(actual_eigen_summary["finite"]):
            raise ValueError("precomputed mass covariance eigenvalues must be finite")
        if not bool(actual_eigen_summary["positive"]):
            raise ValueError("precomputed mass covariance must be positive definite")
        eigen_summary = (
            actual_eigen_summary
            if self.eigen_summary is None
            else _normalize_eigen_summary(self.eigen_summary)
        )
        if not bool(eigen_summary["finite"]):
            raise ValueError("precomputed mass covariance eigenvalues must be finite")
        if not bool(eigen_summary["positive"]):
            raise ValueError("precomputed mass covariance must be positive definite")
        precision_eigen_summary = (
            None
            if self.precision_eigen_summary is None
            else _normalize_eigen_summary(self.precision_eigen_summary)
        )
        regularization_report = (
            {}
            if self.regularization_report is None
            else dict(self.regularization_report)
        )
        nonclaims = tuple(str(item) for item in self.nonclaims)
        if not nonclaims:
            raise ValueError("nonclaims must be non-empty")
        position.setflags(write=False)
        covariance.setflags(write=False)
        factor.setflags(write=False)
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "covariance", covariance)
        object.__setattr__(self, "factor", factor)
        object.__setattr__(self, "adapter_signature", signature)
        object.__setattr__(self, "position_role", position_role)
        object.__setattr__(self, "covariance_source", covariance_source)
        object.__setattr__(self, "matrix_used_for_square_root", matrix_used)
        object.__setattr__(self, "factor_orientation", orientation)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "eigen_summary", eigen_summary)
        object.__setattr__(self, "precision_eigen_summary", precision_eigen_summary)
        object.__setattr__(self, "regularization_report", regularization_report)
        object.__setattr__(self, "log_jacobian_convention", convention)
        object.__setattr__(self, "nonclaims", nonclaims)
        object.__setattr__(self, "reconstruction_rtol", float(self.reconstruction_rtol))
        object.__setattr__(self, "reconstruction_atol", float(self.reconstruction_atol))

    @classmethod
    def from_covariance(
        cls,
        *,
        position: Any,
        covariance: Any,
        adapter_signature: str,
        position_role: str = "map",
        covariance_source: str,
        matrix_used_for_square_root: str = "regularized_covariance",
        source: str = "precomputed_covariance",
        jitter: float = 1.0e-9,
        **kwargs: Any,
    ) -> "PrecomputedMassArtifact":
        regularized = np.asarray(covariance, dtype=float) + float(jitter) * np.eye(
            np.asarray(covariance, dtype=float).shape[0]
        )
        factor = whitening_from_covariance(covariance, jitter=jitter)
        return cls(
            position=position,
            covariance=regularized,
            factor=factor,
            adapter_signature=adapter_signature,
            position_role=position_role,
            covariance_source=covariance_source,
            matrix_used_for_square_root=matrix_used_for_square_root,
            source=source,
            **kwargs,
        )

    @classmethod
    def from_negative_hessian(
        cls,
        *,
        position: Any,
        negative_hessian: Any,
        adapter_signature: str,
        position_role: str = "map",
        covariance_source: str = "negative_hessian",
        matrix_used_for_square_root: str = "covariance_from_negative_hessian",
        source: str = "negative_hessian",
        jitter: float = 1.0e-9,
        eigenvalue_floor: float | None = None,
        max_condition_number: float | None = None,
        dense: bool = True,
        **kwargs: Any,
    ) -> "PrecomputedMassArtifact":
        mass = covariance_from_negative_hessian(
            negative_hessian,
            source=covariance_source,
            jitter=jitter,
            eigenvalue_floor=eigenvalue_floor,
            max_condition_number=max_condition_number,
            dense=dense,
        )
        factor = whitening_from_covariance(mass.covariance, jitter=0.0)
        return cls(
            position=position,
            covariance=mass.covariance,
            factor=factor,
            adapter_signature=adapter_signature,
            position_role=position_role,
            covariance_source=mass.source,
            matrix_used_for_square_root=matrix_used_for_square_root,
            source=source,
            eigen_summary=mass.covariance_eigen_summary,
            precision_eigen_summary=mass.precision_eigen_summary,
            regularization_report=mass.regularization_report,
            **kwargs,
        )

    @property
    def dimension(self) -> int:
        return int(self.position.shape[0])

    def validate_for_adapter(
        self,
        adapter: Any,
        *,
        expected_dim: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if expected_dim is not None and self.dimension != int(expected_dim):
            raise ValueError(
                f"precomputed mass dimension {self.dimension} does not match {expected_dim}"
            )
        expected_signature = stable_adapter_signature(adapter)
        if self.adapter_signature != expected_signature:
            raise ValueError("precomputed mass adapter signature mismatch")
        return self.position, self.covariance, self.factor

    def build_latent_transform(self) -> "LatentAffineHMCTransform":
        return LatentAffineHMCTransform(
            center=self.position,
            factor=self.factor,
            factor_orientation=self.factor_orientation,
            covariance_provenance=self.covariance_source,
            log_jacobian_convention=self.log_jacobian_convention,
            nonclaims=self.nonclaims,
        )

    def signature_payload(self) -> Mapping[str, Any]:
        return {
            "dimension": self.dimension,
            "adapter_signature": self.adapter_signature,
            "position_role": self.position_role,
            "covariance_source": self.covariance_source,
            "matrix_used_for_square_root": self.matrix_used_for_square_root,
            "factor_orientation": self.factor_orientation,
            "source": self.source,
            "eigen_summary": self.eigen_summary,
            "precision_eigen_summary": self.precision_eigen_summary,
            "regularization_report": self.regularization_report,
            "log_jacobian_convention": self.log_jacobian_convention,
            "nonclaims": self.nonclaims,
        }

    def to_payload(self, *, include_arrays: bool = False) -> Mapping[str, Any]:
        """Return a JSON-safe reload payload, optionally including arrays.

        ``signature_payload`` remains the stable metadata-only fingerprint
        surface.  This method is the durable handoff schema for callers that
        need to persist and rehydrate the validated mass artifact.
        """

        payload = dict(self.signature_payload())
        payload.update(
            {
                "artifact_type": "bayesfilter_precomputed_mass_artifact",
                "schema_version": 1,
                "include_arrays": bool(include_arrays),
                "reconstruction_rtol": self.reconstruction_rtol,
                "reconstruction_atol": self.reconstruction_atol,
            }
        )
        payload["eigen_summary"] = _json_safe_metadata(self.eigen_summary)
        payload["precision_eigen_summary"] = _json_safe_metadata(
            self.precision_eigen_summary
        )
        payload["regularization_report"] = _json_safe_metadata(
            self.regularization_report
        )
        payload["nonclaims"] = list(self.nonclaims)
        if include_arrays:
            payload.update(
                {
                    "position": np.asarray(self.position, dtype=float).tolist(),
                    "covariance": np.asarray(self.covariance, dtype=float).tolist(),
                    "factor": np.asarray(self.factor, dtype=float).tolist(),
                }
            )
        return payload

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
        *,
        expected_adapter_signature: str | None = None,
        expected_dim: int | None = None,
    ) -> "PrecomputedMassArtifact":
        """Rehydrate a validated artifact from an array-bearing payload."""

        if not isinstance(payload, Mapping):
            raise TypeError("precomputed mass payload must be a mapping")
        if payload.get("artifact_type") != "bayesfilter_precomputed_mass_artifact":
            raise ValueError("precomputed mass payload artifact_type mismatch")
        if int(payload.get("schema_version", -1)) != 1:
            raise ValueError("precomputed mass payload schema_version mismatch")
        if expected_adapter_signature is not None:
            expected = _validate_persisted_signature(str(expected_adapter_signature))
            observed = _validate_persisted_signature(str(payload.get("adapter_signature", "")))
            if observed != expected:
                raise ValueError("precomputed mass payload adapter signature mismatch")
        if expected_dim is not None:
            expected_dimension = int(expected_dim)
            observed_dimension = int(payload.get("dimension", -1))
            if observed_dimension != expected_dimension:
                raise ValueError("precomputed mass payload dimension mismatch")
        missing_arrays = tuple(
            key for key in ("position", "covariance", "factor") if key not in payload
        )
        if missing_arrays:
            raise ValueError(
                "precomputed mass payload missing required array fields: "
                + ", ".join(missing_arrays)
            )
        return cls(
            position=payload["position"],
            covariance=payload["covariance"],
            factor=payload["factor"],
            adapter_signature=str(payload.get("adapter_signature", "")),
            position_role=str(payload.get("position_role", "map")),
            covariance_source=str(payload.get("covariance_source", "")),
            matrix_used_for_square_root=str(
                payload.get("matrix_used_for_square_root", "")
            ),
            factor_orientation=str(
                payload.get("factor_orientation", "row_right_transpose")
            ),
            source=str(payload.get("source", "")),
            eigen_summary=None,
            precision_eigen_summary=payload.get("precision_eigen_summary"),
            regularization_report=payload.get("regularization_report"),
            log_jacobian_convention=str(
                payload.get("log_jacobian_convention", "constant_omitted")
            ),
            nonclaims=tuple(str(item) for item in payload.get("nonclaims", ())),
            reconstruction_rtol=float(payload.get("reconstruction_rtol", 1.0e-10)),
            reconstruction_atol=float(payload.get("reconstruction_atol", 1.0e-10)),
        )


@dataclass(frozen=True)
class LatentAffineHMCTransform:
    """Metadata-bearing row-vector affine transform for latent HMC states.

    The transform is

    ``theta = center + z @ factor.T``

    with latent score ``grad_z = factor.T @ grad_theta`` for a scalar target
    evaluated at ``theta``.  The orientation field is semantic authority, not a
    display label: callers must not reinterpret ``factor`` as a left-multiply or
    column-vector convention.
    """

    center: Any
    factor: Any
    factor_orientation: str = "row_right_transpose"
    covariance_provenance: str = "unspecified"
    log_jacobian_convention: str = "constant_omitted"
    nonclaims: tuple[str, ...] = (
        "affine coordinate transform only",
        "no posterior convergence claim",
        "no sampler readiness claim",
    )

    def __post_init__(self) -> None:
        center = np.asarray(self.center, dtype=float)
        factor = np.asarray(self.factor, dtype=float)
        if center.ndim != 1:
            raise ValueError("latent affine center must be a one-dimensional vector")
        dim = int(center.shape[0])
        if factor.shape != (dim, dim):
            raise ValueError("latent affine factor must be square and match center dimension")
        if not np.all(np.isfinite(center)):
            raise ValueError("latent affine center must be finite")
        if not np.all(np.isfinite(factor)):
            raise ValueError("latent affine factor must be finite")
        orientation = str(self.factor_orientation)
        if orientation != "row_right_transpose":
            raise ValueError("factor_orientation must be 'row_right_transpose'")
        convention = str(self.log_jacobian_convention)
        if convention not in {"constant_omitted", "constant_included"}:
            raise ValueError(
                "log_jacobian_convention must be 'constant_omitted' or 'constant_included'"
            )
        provenance = str(self.covariance_provenance)
        if not provenance:
            raise ValueError("covariance_provenance must be non-empty")
        nonclaims = tuple(str(item) for item in self.nonclaims)
        if not nonclaims:
            raise ValueError("nonclaims must be non-empty")
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "factor", factor)
        object.__setattr__(self, "factor_orientation", orientation)
        object.__setattr__(self, "covariance_provenance", provenance)
        object.__setattr__(self, "log_jacobian_convention", convention)
        object.__setattr__(self, "nonclaims", nonclaims)

    @property
    def dimension(self) -> int:
        return int(self.center.shape[0])

    def latent_to_position(self, z: Any) -> np.ndarray:
        """Map scalar or batched row-vector latent coordinates to positions."""

        z_array = self._validate_latent_shape(z)
        return self.center + z_array @ self.factor.T

    def position_to_latent(self, theta: Any) -> np.ndarray:
        """Invert the affine map for nonsingular factors."""

        theta_array = self._validate_position_shape(theta)
        delta = theta_array - self.center
        try:
            solved = np.linalg.solve(self.factor, np.moveaxis(delta, -1, 0))
            return np.moveaxis(solved, 0, -1)
        except np.linalg.LinAlgError as exc:
            raise ValueError("latent affine factor must be nonsingular to invert") from exc

    def theta_score_to_latent_score(self, grad_theta: Any) -> np.ndarray:
        """Apply the chain rule ``grad_z = factor.T @ grad_theta`` row-wise."""

        grad_array = self._validate_position_shape(grad_theta)
        return grad_array @ self.factor

    def value_and_score(
        self,
        z: Any,
        value_and_grad_fn: Callable[[np.ndarray], tuple[Any, Any]],
    ) -> tuple[Any, np.ndarray]:
        """Evaluate a theta-target and return the corresponding latent score."""

        theta = self.latent_to_position(z)
        value, grad_theta = value_and_grad_fn(theta)
        return value, self.theta_score_to_latent_score(grad_theta)

    def signature_payload(self) -> Mapping[str, Any]:
        return {
            "dimension": self.dimension,
            "factor_orientation": self.factor_orientation,
            "covariance_provenance": self.covariance_provenance,
            "log_jacobian_convention": self.log_jacobian_convention,
            "nonclaims": self.nonclaims,
        }

    def _validate_latent_shape(self, value: Any) -> np.ndarray:
        array = np.asarray(value, dtype=float)
        if array.shape[-1:] != (self.dimension,):
            raise ValueError("latent coordinate trailing dimension must match transform dimension")
        if not np.all(np.isfinite(array)):
            raise ValueError("latent coordinates must be finite")
        return array

    def _validate_position_shape(self, value: Any) -> np.ndarray:
        array = np.asarray(value, dtype=float)
        if array.shape[-1:] != (self.dimension,):
            raise ValueError("position/score trailing dimension must match transform dimension")
        if not np.all(np.isfinite(array)):
            raise ValueError("position/score values must be finite")
        return array


@dataclass(frozen=True)
class FullChainHMCConfig:
    """Tiny full-chain TFP HMC runtime configuration."""

    num_results: int
    num_burnin_steps: int
    step_size: float
    num_leapfrog_steps: int
    seed: tuple[int, int]
    use_xla: bool = False
    trace_policy: str = "standard"
    target_status_trace_policy: str = "none"
    adaptation_policy: str = "fixed_kernel_no_adaptation"
    tuning_policy: str | HMCTuningPolicy | None = None
    target_scope: str | None = None
    chain_execution_mode: str = "tf_function"

    def __post_init__(self) -> None:
        for name in ("num_results", "num_burnin_steps", "num_leapfrog_steps"):
            value = int(getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        step_size = float(self.step_size)
        if not np.isfinite(step_size) or step_size <= 0.0:
            raise ValueError("step_size must be positive and finite")
        object.__setattr__(self, "step_size", step_size)
        seed = tuple(int(item) for item in self.seed)
        if len(seed) != 2:
            raise ValueError("seed must contain exactly two integers")
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "use_xla", bool(self.use_xla))
        chain_execution_mode = str(self.chain_execution_mode)
        if chain_execution_mode not in {"tf_function", "eager"}:
            raise ValueError("chain_execution_mode must be 'tf_function' or 'eager'")
        if self.use_xla and chain_execution_mode != "tf_function":
            raise ValueError("XLA full-chain HMC requires chain_execution_mode='tf_function'")
        object.__setattr__(self, "chain_execution_mode", chain_execution_mode)
        policy = str(self.trace_policy)
        if policy not in {"standard", "reduced"}:
            raise ValueError("trace_policy must be 'standard' or 'reduced'")
        object.__setattr__(self, "trace_policy", policy)
        target_status_policy = str(self.target_status_trace_policy)
        if target_status_policy not in {"none", "per_chain_step"}:
            raise ValueError(
                "target_status_trace_policy must be 'none' or 'per_chain_step'"
            )
        if target_status_policy != "none" and policy != "standard":
            raise ValueError(
                "target-status telemetry requires trace_policy='standard'"
            )
        object.__setattr__(self, "target_status_trace_policy", target_status_policy)
        adaptation_policy = str(self.adaptation_policy)
        if self.tuning_policy is None:
            if adaptation_policy != "fixed_kernel_no_adaptation":
                raise ValueError(
                    "BayesFilter HMC adaptation is fail-closed; raw adaptation_policy "
                    "strings support only fixed_kernel_no_adaptation. Pass a reviewed "
                    "HMCTuningPolicy object for diagnostic adaptation."
                )
            tuning_policy = normalize_hmc_tuning_policy(adaptation_policy)
        else:
            if adaptation_policy != "fixed_kernel_no_adaptation":
                raise ValueError(
                    "adaptation_policy strings other than fixed_kernel_no_adaptation "
                    "remain unsupported; use tuning_policy=HMCTuningPolicy(...)"
                )
            tuning_policy = normalize_hmc_tuning_policy(self.tuning_policy)
            require_executable_tuning_policy(tuning_policy)
            if tuning_policy.uses_windowed_mass_adaptation:
                raise ValueError(
                    "windowed mass adaptation is executable only through the "
                    "Phase 4 windowed diagnostic runner, not run_full_chain_tfp_hmc"
                )
            if tuning_policy.uses_dual_averaging and tuning_policy.label != "fixed_mass_dual_averaging":
                raise ValueError(
                    "dual-averaging HMC execution requires fixed_mass_dual_averaging "
                    "in this phase"
                )
        if (
            tuning_policy.num_adaptation_steps > 0
            and tuning_policy.num_adaptation_steps > self.num_burnin_steps
        ):
            raise ValueError("tuning policy adaptation steps must not exceed burnin steps")
        if (
            tuning_policy.uses_dual_averaging
            and self.use_xla
            and tuning_policy.label != "fixed_mass_dual_averaging"
        ):
            raise ValueError(
                "XLA dual-averaging HMC execution requires fixed_mass_dual_averaging "
                "in this phase"
            )
        object.__setattr__(self, "tuning_policy", tuning_policy)
        object.__setattr__(self, "adaptation_policy", tuning_policy.adaptation_policy)
        if self.target_scope is not None:
            object.__setattr__(self, "target_scope", str(self.target_scope))

    def signature_payload(self) -> Mapping[str, Any]:
        return {
            "num_results": self.num_results,
            "num_burnin_steps": self.num_burnin_steps,
            "step_size": self.step_size,
            "num_leapfrog_steps": self.num_leapfrog_steps,
            "seed": self.seed,
            "use_xla": self.use_xla,
            "chain_execution_mode": self.chain_execution_mode,
            "trace_policy": self.trace_policy,
            "target_status_trace_policy": self.target_status_trace_policy,
            "adaptation_policy": self.adaptation_policy,
            "tuning_policy": self.tuning_policy.payload(),
            "target_scope": self.target_scope,
        }


@dataclass(frozen=True)
class FullChainHMCRunResult:
    """Tensor-valued full-chain HMC result plus engineering metadata."""

    samples: Any
    trace: Mapping[str, Any]
    diagnostics: Mapping[str, Any]
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class FixedSizeHMCChunkConfig:
    """Fixed-shape HMC chunk contract with runtime active sample count."""

    max_results: int
    num_burnin_steps: int
    step_size: float
    num_leapfrog_steps: int
    seed: tuple[int, int]
    use_xla: bool = False
    trace_policy: str = "reduced"
    target_status_trace_policy: str = "none"
    target_scope: str | None = None
    chain_execution_mode: str = "tf_function"

    def __post_init__(self) -> None:
        max_results = int(self.max_results)
        if max_results <= 0:
            raise ValueError("max_results must be positive")
        object.__setattr__(self, "max_results", max_results)
        burnin = int(self.num_burnin_steps)
        if burnin < 0:
            raise ValueError("num_burnin_steps must be nonnegative")
        object.__setattr__(self, "num_burnin_steps", burnin)
        leapfrog = int(self.num_leapfrog_steps)
        if leapfrog <= 0:
            raise ValueError("num_leapfrog_steps must be positive")
        object.__setattr__(self, "num_leapfrog_steps", leapfrog)
        step_size = float(self.step_size)
        if not np.isfinite(step_size) or step_size <= 0.0:
            raise ValueError("step_size must be positive and finite")
        object.__setattr__(self, "step_size", step_size)
        seed = tuple(int(item) for item in self.seed)
        if len(seed) != 2:
            raise ValueError("seed must contain exactly two integers")
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "use_xla", bool(self.use_xla))
        chain_execution_mode = str(self.chain_execution_mode)
        if chain_execution_mode not in {"tf_function", "eager"}:
            raise ValueError("chain_execution_mode must be 'tf_function' or 'eager'")
        if self.use_xla and chain_execution_mode != "tf_function":
            raise ValueError("XLA fixed-size HMC chunk requires chain_execution_mode='tf_function'")
        object.__setattr__(self, "chain_execution_mode", chain_execution_mode)
        trace_policy = str(self.trace_policy)
        if trace_policy not in {"reduced", "standard"}:
            raise ValueError("trace_policy must be 'reduced' or 'standard'")
        object.__setattr__(self, "trace_policy", trace_policy)
        target_status_policy = str(self.target_status_trace_policy)
        if target_status_policy != "none":
            raise ValueError("fixed-size HMC chunk runner does not collect target-status traces")
        object.__setattr__(self, "target_status_trace_policy", target_status_policy)
        if self.target_scope is not None:
            object.__setattr__(self, "target_scope", str(self.target_scope))

    def signature_payload(self) -> Mapping[str, Any]:
        return {
            "max_results": self.max_results,
            "num_burnin_steps": self.num_burnin_steps,
            "step_size": self.step_size,
            "num_leapfrog_steps": self.num_leapfrog_steps,
            "seed": self.seed,
            "use_xla": self.use_xla,
            "chain_execution_mode": self.chain_execution_mode,
            "trace_policy": self.trace_policy,
            "target_status_trace_policy": self.target_status_trace_policy,
            "target_scope": self.target_scope,
        }


@dataclass(frozen=True)
class FixedSizeHMCChunkRunResult:
    """Fixed-size HMC chunk result with valid-mask and final-state handoff."""

    samples: Any
    valid_mask: Any
    final_state: Any
    trace: Mapping[str, Any]
    diagnostics: Mapping[str, Any]
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class HMCSampleArchiveManifest:
    """Public metadata for a private retained HMC sample archive.

    The manifest records paths, hashes, shapes, dtypes, byte counts, and scalar
    diagnostics only.  The archive files themselves are private because they
    contain raw retained samples and state handoff tensors.
    """

    archive_dir: str
    manifest_path: str
    manifest_hash: str
    payload: Mapping[str, Any]

    def public_payload(self) -> Mapping[str, Any]:
        """Return the metadata-only payload suitable for public run artifacts."""

        return dict(self.payload)


class HMCStreamingSampleArchiveSink:
    """Incremental private sink for retained fixed-size HMC chunks.

    ``write_chunk`` serializes one chunk's sample, mask, and final-state tensors
    immediately and retains only metadata needed for the final public-safe
    manifest.  This is the bounded-retention contract callers should use when
    they need retained samples without accumulating all chunk tensors in Python.
    """

    def __init__(
        self,
        *,
        archive_dir: str | Path,
        config: FixedSizeHMCChunkConfig,
        archive_label: str,
        metadata: Mapping[str, Any] | None = None,
        overwrite: bool = False,
    ) -> None:
        label = str(archive_label).strip()
        if not label:
            raise ValueError("archive_label must be non-empty")
        root = Path(archive_dir)
        if root.exists() and not root.is_dir():
            raise ValueError("archive_dir exists and is not a directory")
        root.mkdir(parents=True, exist_ok=True)
        self.archive_dir = root
        self.config = config
        self.archive_label = label
        self.metadata = {} if metadata is None else _json_safe_metadata(dict(metadata))
        self.overwrite = bool(overwrite)
        self._chunk_payloads: list[dict[str, Any]] = []
        self._total_valid = 0
        self._total_nonfinite = 0
        self._finalized = False

    @property
    def chunk_count(self) -> int:
        return len(self._chunk_payloads)

    def write_chunk(
        self,
        chunk: FixedSizeHMCChunkRunResult,
        *,
        chunk_index: int | None = None,
    ) -> Mapping[str, Any]:
        """Flush one chunk to private tensor shards and return metadata only."""

        if self._finalized:
            raise RuntimeError("cannot write chunks after archive manifest finalization")
        index = self.chunk_count if chunk_index is None else int(chunk_index)
        if index != self.chunk_count:
            raise ValueError("chunk_index must match current sequential chunk_count")
        payload, valid_count, nonfinite_count = _write_hmc_archive_chunk(
            archive_dir=self.archive_dir,
            archive_label=self.archive_label,
            chunk=chunk,
            config=self.config,
            chunk_index=index,
            overwrite=self.overwrite,
        )
        self._chunk_payloads.append(payload)
        self._total_valid += valid_count
        self._total_nonfinite += nonfinite_count
        return dict(payload)

    def finalize(self) -> HMCSampleArchiveManifest:
        """Write and return the final public-safe manifest."""

        if self._finalized:
            raise RuntimeError("archive manifest already finalized")
        if not self._chunk_payloads:
            raise ValueError("cannot finalize an empty HMC sample archive")
        self._finalized = True
        return _finalize_hmc_sample_archive_manifest(
            archive_dir=self.archive_dir,
            archive_label=self.archive_label,
            config=self.config,
            chunk_payloads=tuple(self._chunk_payloads),
            total_valid=self._total_valid,
            total_nonfinite=self._total_nonfinite,
            metadata=self.metadata,
            overwrite=self.overwrite,
        )


@dataclass(frozen=True)
class InternalSegmentHMCRunnerConfig:
    """Single-call HMC runner with fixed internal progress segments.

    The runner executes ``segment_count * segment_length`` HMC transitions in
    one compiled function call.  In the initial ``summary_only`` output mode it
    returns only the final state plus fixed-shape scalar segment summaries.  It
    deliberately does not return sample tensors; retained-sample streaming is a
    separate contract so long-running clients do not accidentally reintroduce
    repeated compiled-call or returned-buffer memory growth.
    """

    segment_count: int
    segment_length: int
    num_burnin_steps: int
    step_size: float
    num_leapfrog_steps: int
    seed: tuple[int, int]
    use_xla: bool = False
    output_mode: str = "summary_only"
    target_scope: str | None = None
    chain_execution_mode: str = "tf_function"

    def __post_init__(self) -> None:
        for name in ("segment_count", "segment_length", "num_leapfrog_steps"):
            value = int(getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        burnin = int(self.num_burnin_steps)
        if burnin < 0:
            raise ValueError("num_burnin_steps must be nonnegative")
        object.__setattr__(self, "num_burnin_steps", burnin)
        step_size = float(self.step_size)
        if not np.isfinite(step_size) or step_size <= 0.0:
            raise ValueError("step_size must be positive and finite")
        object.__setattr__(self, "step_size", step_size)
        seed = tuple(int(item) for item in self.seed)
        if len(seed) != 2:
            raise ValueError("seed must contain exactly two integers")
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "use_xla", bool(self.use_xla))
        chain_execution_mode = str(self.chain_execution_mode)
        if chain_execution_mode not in {"tf_function", "eager"}:
            raise ValueError("chain_execution_mode must be 'tf_function' or 'eager'")
        if self.use_xla and chain_execution_mode != "tf_function":
            raise ValueError("XLA internal-segment HMC requires chain_execution_mode='tf_function'")
        object.__setattr__(self, "chain_execution_mode", chain_execution_mode)
        output_mode = str(self.output_mode)
        if output_mode != "summary_only":
            raise ValueError("internal-segment HMC currently supports only output_mode='summary_only'")
        object.__setattr__(self, "output_mode", output_mode)
        if self.target_scope is not None:
            object.__setattr__(self, "target_scope", str(self.target_scope))

    @property
    def total_transitions(self) -> int:
        return int(self.segment_count * self.segment_length)

    def signature_payload(self) -> Mapping[str, Any]:
        return {
            "segment_count": self.segment_count,
            "segment_length": self.segment_length,
            "total_transitions": self.total_transitions,
            "num_burnin_steps": self.num_burnin_steps,
            "step_size": self.step_size,
            "num_leapfrog_steps": self.num_leapfrog_steps,
            "seed": self.seed,
            "use_xla": self.use_xla,
            "chain_execution_mode": self.chain_execution_mode,
            "output_mode": self.output_mode,
            "target_scope": self.target_scope,
        }


@dataclass(frozen=True)
class RetainedSampleHMCArchiveConfig:
    """One-call retained HMC contract with private sample archive output.

    The runner executes all burn-in and retained HMC transitions in a single
    TensorFlow call and returns no sample tensor to the caller.  Retained
    samples are persisted privately after the call in exactly one physical
    retained-sample shard; any manifest or final-state files are sidecars, not
    additional sample shards.
    """

    num_results: int
    num_burnin_steps: int
    step_size: float
    num_leapfrog_steps: int
    seed: tuple[int, int]
    use_xla: bool = False
    target_scope: str | None = None
    chain_execution_mode: str = "tf_function"
    sampler_diagnostics_policy: str = "public_safe_summary"

    def __post_init__(self) -> None:
        results = int(self.num_results)
        if results <= 0:
            raise ValueError("num_results must be positive")
        object.__setattr__(self, "num_results", results)
        burnin = int(self.num_burnin_steps)
        if burnin < 0:
            raise ValueError("num_burnin_steps must be nonnegative")
        object.__setattr__(self, "num_burnin_steps", burnin)
        leapfrog = int(self.num_leapfrog_steps)
        if leapfrog <= 0:
            raise ValueError("num_leapfrog_steps must be positive")
        object.__setattr__(self, "num_leapfrog_steps", leapfrog)
        step_size = float(self.step_size)
        if not np.isfinite(step_size) or step_size <= 0.0:
            raise ValueError("step_size must be positive and finite")
        object.__setattr__(self, "step_size", step_size)
        seed = tuple(int(item) for item in self.seed)
        if len(seed) != 2:
            raise ValueError("seed must contain exactly two integers")
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "use_xla", bool(self.use_xla))
        chain_execution_mode = str(self.chain_execution_mode)
        if chain_execution_mode not in {"tf_function", "eager"}:
            raise ValueError("chain_execution_mode must be 'tf_function' or 'eager'")
        if self.use_xla and chain_execution_mode != "tf_function":
            raise ValueError(
                "XLA retained-sample HMC archive requires chain_execution_mode='tf_function'"
            )
        object.__setattr__(self, "chain_execution_mode", chain_execution_mode)
        if self.target_scope is not None:
            object.__setattr__(self, "target_scope", str(self.target_scope))
        sampler_diagnostics_policy = str(self.sampler_diagnostics_policy)
        if sampler_diagnostics_policy != "public_safe_summary":
            raise ValueError(
                "retained-sample HMC archive currently supports only "
                "sampler_diagnostics_policy='public_safe_summary'"
            )
        object.__setattr__(
            self,
            "sampler_diagnostics_policy",
            sampler_diagnostics_policy,
        )

    def signature_payload(self) -> Mapping[str, Any]:
        return {
            "num_results": self.num_results,
            "num_burnin_steps": self.num_burnin_steps,
            "step_size": self.step_size,
            "num_leapfrog_steps": self.num_leapfrog_steps,
            "seed": self.seed,
            "use_xla": self.use_xla,
            "chain_execution_mode": self.chain_execution_mode,
            "target_scope": self.target_scope,
            "sampler_diagnostics_policy": self.sampler_diagnostics_policy,
        }


@dataclass(frozen=True)
class SequentialRHatHMCVerificationConfig:
    """Sequential fixed-kernel HMC verifier with fixed checkpoints.

    The verifier runs a fixed-size TF/TFP HMC chunk repeatedly, computes
    dependence-aware acceptance evidence and rank-normalized split/folded R-hat
    on private traces after each checkpoint. Out-of-band acceptance returns a
    repair decision immediately; promotion requires both acceptance evidence
    and the R-hat gate after the minimum retained count.
    It is a tuning-verification gate, not a posterior-convergence certificate.
    """

    check_interval: int
    max_results: int
    num_burnin_steps: int
    step_size: float
    num_leapfrog_steps: int
    seed: tuple[int, int]
    min_retained_results_for_pass: int | None = None
    chain_count: int = 4
    rhat_threshold: float = 1.01
    acceptance_policy: HMCAcceptancePolicy = field(
        default_factory=HMCAcceptancePolicy
    )
    use_xla: bool = False
    target_scope: str | None = None
    chain_execution_mode: str = "tf_function"

    def __post_init__(self) -> None:
        check_interval = int(self.check_interval)
        if check_interval <= 0:
            raise ValueError("check_interval must be positive")
        object.__setattr__(self, "check_interval", check_interval)
        max_results = int(self.max_results)
        if max_results <= 0:
            raise ValueError("max_results must be positive")
        if max_results > 10000:
            raise ValueError("max_results must not exceed the 10000 retained-sample cap")
        if max_results < check_interval:
            raise ValueError("max_results must be at least check_interval")
        object.__setattr__(self, "max_results", max_results)
        min_retained = (
            check_interval
            if self.min_retained_results_for_pass is None
            else int(self.min_retained_results_for_pass)
        )
        if min_retained <= 0:
            raise ValueError("min_retained_results_for_pass must be positive")
        if min_retained > max_results:
            raise ValueError(
                "min_retained_results_for_pass must not exceed max_results"
            )
        object.__setattr__(
            self,
            "min_retained_results_for_pass",
            min_retained,
        )
        burnin = int(self.num_burnin_steps)
        if burnin < 0:
            raise ValueError("num_burnin_steps must be nonnegative")
        object.__setattr__(self, "num_burnin_steps", burnin)
        leapfrog = int(self.num_leapfrog_steps)
        if leapfrog <= 0:
            raise ValueError("num_leapfrog_steps must be positive")
        object.__setattr__(self, "num_leapfrog_steps", leapfrog)
        step_size = float(self.step_size)
        if not np.isfinite(step_size) or step_size <= 0.0:
            raise ValueError("step_size must be positive and finite")
        object.__setattr__(self, "step_size", step_size)
        seed = tuple(int(item) for item in self.seed)
        if len(seed) != 2:
            raise ValueError("seed must contain exactly two integers")
        object.__setattr__(self, "seed", seed)
        chain_count = int(self.chain_count)
        if chain_count < 2:
            raise ValueError("sequential R-hat verification requires at least two chains")
        object.__setattr__(self, "chain_count", chain_count)
        threshold = float(self.rhat_threshold)
        if not np.isfinite(threshold) or threshold <= 1.0:
            raise ValueError("rhat_threshold must be finite and greater than 1")
        object.__setattr__(self, "rhat_threshold", threshold)
        if not isinstance(self.acceptance_policy, HMCAcceptancePolicy):
            raise TypeError("acceptance_policy must be HMCAcceptancePolicy")
        object.__setattr__(self, "use_xla", bool(self.use_xla))
        chain_execution_mode = str(self.chain_execution_mode)
        if chain_execution_mode not in {"tf_function", "eager"}:
            raise ValueError("chain_execution_mode must be 'tf_function' or 'eager'")
        if self.use_xla and chain_execution_mode != "tf_function":
            raise ValueError(
                "XLA sequential R-hat HMC verification requires "
                "chain_execution_mode='tf_function'"
            )
        object.__setattr__(self, "chain_execution_mode", chain_execution_mode)
        if self.target_scope is not None:
            object.__setattr__(self, "target_scope", str(self.target_scope))

    def signature_payload(self) -> Mapping[str, Any]:
        return {
            "check_interval": self.check_interval,
            "max_results": self.max_results,
            "min_retained_results_for_pass": self.min_retained_results_for_pass,
            "num_burnin_steps": self.num_burnin_steps,
            "step_size": self.step_size,
            "num_leapfrog_steps": self.num_leapfrog_steps,
            "seed": self.seed,
            "chain_count": self.chain_count,
            "rhat_threshold": self.rhat_threshold,
            "acceptance_policy": self.acceptance_policy.payload(),
            "use_xla": self.use_xla,
            "chain_execution_mode": self.chain_execution_mode,
            "target_scope": self.target_scope,
        }


@dataclass(frozen=True)
class InternalSegmentHMCRunResult:
    """Summary-only internal-segment HMC result.

    ``segment_indices`` records the transition index at each segment boundary;
    ``segment_target_log_prob`` records the per-chain target log probability at
    those boundaries.  No raw samples are returned in ``summary_only`` mode.
    """

    final_state: Any
    final_target_log_prob: Any
    final_index: Any
    segment_index: Any
    segment_indices: Any
    segment_target_log_prob: Any
    diagnostics: Mapping[str, Any]
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class RetainedSampleHMCArchiveRunResult:
    """Public-safe result for one-call retained HMC archive execution.

    Raw retained samples are intentionally absent from this contract.  The
    caller receives only the final state for explicit handoff, finite/count
    diagnostics, public-safe metadata, and an aggregate archive summary.
    """

    final_state: Any
    final_target_log_prob: Any
    final_index: Any
    diagnostics: Mapping[str, Any]
    metadata: Mapping[str, Any]
    archive_summary: Mapping[str, Any]


@dataclass(frozen=True)
class SequentialRHatHMCVerificationResult:
    """Public-safe result for sequential R-hat tuning verification."""

    passed: bool
    cap_hit: bool
    retained_sample_count: int
    chunk_count: int
    max_finite_rhat: float | None
    finite_rhat_count: int
    nonfinite_rhat_count: int
    diagnostics: Mapping[str, Any]
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class SequentialRHatCheckpointWriterConfig:
    """Opt-in private checkpoint writer for sequential R-hat verification.

    The writer runs only after a fixed-size HMC chunk has returned to Python.
    It is not part of the TensorFlow/TFP target, transition kernel, or XLA
    compiled sampling graph.
    """

    checkpoint_dir: str | Path
    checkpoint_label: str = "sequential_rhat"
    overwrite: bool = False

    def __post_init__(self) -> None:
        root = Path(self.checkpoint_dir)
        if root.exists() and not root.is_dir():
            raise ValueError("checkpoint_dir exists and is not a directory")
        label = str(self.checkpoint_label).strip()
        if not label:
            raise ValueError("checkpoint_label must be non-empty")
        if "/" in label or "\\" in label:
            raise ValueError("checkpoint_label must not contain path separators")
        object.__setattr__(self, "checkpoint_dir", root)
        object.__setattr__(self, "checkpoint_label", label)
        object.__setattr__(self, "overwrite", bool(self.overwrite))


SEQUENTIAL_RHAT_CHECKPOINT_PUBLIC_REFERENCE_FIELDS: tuple[str, ...] = (
    "artifact_type",
    "schema_version",
    "checkpoint_kind",
    "checkpoint_id",
    "checkpoint_sha256",
    "contract_sha256",
    "private_paths_publicized",
    "public_summary_contains_paths",
    "public_summary_contains_raw_values",
    "public_summary_contains_tensor_descriptors",
    "public_summary_contains_kernel_payload",
    "nonclaims",
)

SEQUENTIAL_RHAT_CHECKPOINT_PUBLIC_NONCLAIMS: tuple[str, ...] = (
    "sequential R-hat checkpoint reference only",
    "private checkpoint contents are not publicized",
    "no posterior convergence claim",
    "no sampler superiority claim",
    "no scientific validity claim",
    "no resume claim",
)

SEQUENTIAL_RHAT_CHECKPOINT_INSPECTION_PUBLIC_FIELDS: tuple[str, ...] = (
    "artifact_type",
    "schema_version",
    "checkpoint_kind",
    "checkpoint_id",
    "checkpoint_sha256",
    "contract_sha256",
    "manifest_core_hash_verified",
    "private_payload_hashes_verified",
    "private_payload_count",
    "required_private_payloads_present",
    "retained_sample_count",
    "latest_chunk_index",
    "valid_sample_count",
    "nonfinite_valid_sample_count",
    "rhat_summary",
    "acceptance_summary",
    "finite_checks",
    "divergence_summary",
    "provenance",
    "resume",
    "privacy_contract",
    "nonclaims",
)

SEQUENTIAL_RHAT_CHECKPOINT_INSPECTION_NONCLAIMS: tuple[str, ...] = (
    "sequential R-hat private checkpoint inspection only",
    "private checkpoint contents are not publicized",
    "no posterior convergence claim",
    "no sampler superiority claim",
    "no scientific validity claim",
    "no resume claim",
)

SEQUENTIAL_RHAT_CHECKPOINT_KINDS: tuple[str, ...] = (
    "phase7_boundary_handoff",
    "pre_verification_handoff",
    "verification_chunk",
)

_SEQUENTIAL_RHAT_OPAQUE_CHECKPOINT_ID_PATTERN = (
    r"^srhat-v1-[0-9a-f]{32}$"
)
_SHA256_HEX_PATTERN = r"^[0-9a-f]{64}$"
_SEQUENTIAL_RHAT_FORBIDDEN_CHECKPOINT_ID_TOKENS = (
    "run",
    "chunk",
    "handoff",
    "sample",
    "state",
    "trace",
    "target",
    "accept",
    "step",
    "leapfrog",
    "mass",
    "kernel",
    "manifest",
    "path",
    "private",
    "result",
    "verification",
)


def sequential_rhat_verification_checkpoint_contract() -> Mapping[str, Any]:
    """Return the Phase 7 checkpoint public/private boundary contract.

    This is a schema contract, not a writer.  Public progress artifacts may
    carry only opaque checkpoint identifiers and hashes.  Private manifests are
    the only place where path-shaped references, tensor shard descriptors, raw
    HMC outputs, selected-kernel mechanics, and readback metadata may appear.
    """

    contract = {
        "artifact_type": "bayesfilter_sequential_rhat_checkpoint_contract",
        "schema_version": 1,
        "public_reference_artifact_type": (
            "bayesfilter_sequential_rhat_checkpoint_public_reference"
        ),
        "private_manifest_artifact_type": (
            "bayesfilter_private_sequential_rhat_checkpoint_manifest"
        ),
        "checkpoint_kinds": SEQUENTIAL_RHAT_CHECKPOINT_KINDS,
        "opaque_checkpoint_id_pattern": _SEQUENTIAL_RHAT_OPAQUE_CHECKPOINT_ID_PATTERN,
        "hash_pattern": _SHA256_HEX_PATTERN,
        "opaque_checkpoint_id_policy": {
            "format": "srhat-v1-<32 lowercase hex chars>",
            "semantic_payload_allowed": False,
            "generation_source": "private manifest or writer nonce/hash material",
            "forbidden_semantic_tokens": (
                _SEQUENTIAL_RHAT_FORBIDDEN_CHECKPOINT_ID_TOKENS
            ),
        },
        "public_reference_fields": SEQUENTIAL_RHAT_CHECKPOINT_PUBLIC_REFERENCE_FIELDS,
        "public_hash_fields": ("checkpoint_sha256", "contract_sha256"),
        "public_id_fields": ("checkpoint_id",),
        "forbidden_public_key_tokens": (
            "manifest_path",
            "archive_dir",
            "sample_path",
            "samples",
            "valid_mask",
            "final_state",
            "trace",
            "target_log_prob",
            "log_accept",
            "step_size",
            "leapfrog",
            "mass",
            "selected_kernel",
            "private_manifest",
        ),
        "forbidden_public_value_patterns": (
            "absolute_path",
            "relative_path",
            "private_directory_name",
            "serialized_tensor_filename",
        ),
        "private_manifest_required_fields": (
            "artifact_type",
            "schema_version",
            "checkpoint_kind",
            "checkpoint_id",
            "manifest_path",
            "manifest_core_sha256",
            "config_private_signature",
            "adapter_signature",
            "created_at_utc",
            "private_shards",
            "private_diagnostics",
            "privacy_contract",
            "nonclaims",
        ),
        "private_manifest_schema": {
            "artifact_type": "literal private_manifest_artifact_type",
            "schema_version": "integer 1",
            "checkpoint_kind": "one of checkpoint_kinds",
            "checkpoint_id": "matches opaque_checkpoint_id_pattern",
            "manifest_path": "private absolute or caller-private relative path",
            "manifest_core_sha256": (
                "lowercase sha256 hex over manifest core payload excluding this field"
            ),
            "config_private_signature": "lowercase sha256 hex over private config payload",
            "adapter_signature": "stable non-empty adapter signature string",
            "created_at_utc": "ISO-8601 UTC timestamp",
            "private_shards": "mapping of shard role to shard_record",
            "private_diagnostics": "private_diagnostics_record",
            "privacy_contract": "privacy_contract_record",
            "nonclaims": "non-empty sequence including no posterior convergence claim",
        },
        "private_shard_record_schema": {
            "role": (
                "one of samples, valid_mask, final_state, reduced_trace, "
                "target_log_prob_summary, log_accept_ratio_summary, "
                "rhat_summary, selected_kernel_private_payload, mass_payload, "
                "boundary_private_payload, config_private_payload, "
                "state_summary_private_payload"
            ),
            "path": "private shard path; never allowed in public progress",
            "sha256": "lowercase sha256 hex of shard bytes",
            "bytes": "non-negative integer byte count",
            "shape": "tuple/list of non-negative integer dimensions, when tensor-like",
            "dtype": "string dtype, when tensor-like",
            "serializer": "string serializer identifier",
        },
        "private_shard_roles": (
            "boundary_private_payload",
            "config_private_payload",
            "state_summary_private_payload",
            "samples",
            "valid_mask",
            "final_state",
            "reduced_trace",
            "target_log_prob_summary",
            "log_accept_ratio_summary",
            "rhat_summary",
            "selected_kernel_private_payload",
            "mass_payload",
        ),
        "private_manifest_required_shard_roles_by_kind": {
            "phase7_boundary_handoff": (
                "boundary_private_payload",
                "config_private_payload",
                "state_summary_private_payload",
            ),
            "pre_verification_handoff": (
                "selected_kernel_private_payload",
                "final_state",
                "mass_payload",
            ),
            "verification_chunk": (
                "samples",
                "valid_mask",
                "final_state",
                "reduced_trace",
                "target_log_prob_summary",
                "log_accept_ratio_summary",
                "rhat_summary",
            ),
        },
        "private_diagnostics_record_schema": {
            "chunk_index": "integer or null for boundary/pre-verification handoff",
            "retained_sample_count": "integer retained count available after this artifact",
            "valid_sample_count": "integer valid sample count or null",
            "nonfinite_valid_sample_count": "integer nonfinite valid sample count or null",
            "rhat_summary": "mapping with threshold, finite counts, max finite R-hat, pass flag",
            "acceptance_summary": "private aggregate acceptance/log-accept metadata",
            "target_log_prob_summary": "private finite/count target log-prob metadata",
            "divergence_summary": "private native divergence status/count metadata",
        },
        "rhat_summary_schema": {
            "rhat_threshold": "positive finite float greater than 1",
            "max_finite_rhat": "finite float or null",
            "finite_rhat_count": "non-negative integer",
            "nonfinite_rhat_count": "non-negative integer",
            "all_finite_rhat_at_or_below_threshold": "boolean",
        },
        "log_accept_ratio_summary_schema": {
            "finite_count": "non-negative integer",
            "nonfinite_count": "non-negative integer",
            "max_abs_finite": "finite float or null",
        },
        "target_log_prob_summary_schema": {
            "finite_count": "non-negative integer",
            "nonfinite_count": "non-negative integer",
            "min_finite": "finite float or null",
            "max_finite": "finite float or null",
        },
        "privacy_contract_record_schema": {
            "manifest_contains_private_paths": True,
            "manifest_contains_private_raw_tensors": False,
            "manifest_points_to_private_raw_tensors": True,
            "public_reference_contains_paths": False,
            "public_reference_contains_tensor_descriptors": False,
            "public_reference_contains_hmc_mechanics": False,
        },
        "private_handoff_fields": (
            "selected_kernel_private_payload",
            "selected_kernel_hash",
            "final_state_shard",
            "mass_payload_shard",
            "step_size",
            "num_leapfrog_steps",
        ),
        "private_boundary_handoff_fields": (
            "boundary_private_payload",
            "config_private_payload",
            "state_summary_private_payload",
            "stage",
            "target_scope",
            "target_dimension",
            "attempt_index",
            "public_budget_class",
            "private_raw_state_allowed",
        ),
        "private_verification_chunk_fields": (
            "chunk_index",
            "retained_sample_count",
            "samples_shard",
            "valid_mask_shard",
            "final_state_shard",
            "reduced_trace_diagnostics",
            "target_log_prob_summary",
            "log_accept_ratio_summary",
            "rhat_summary",
        ),
        "privacy_contract": {
            "public_progress_contains_private_paths": False,
            "public_progress_contains_raw_tensors": False,
            "public_progress_contains_tensor_descriptors": False,
            "public_progress_contains_hmc_mechanics": False,
            "private_manifest_contains_readback_authority": True,
        },
        "inspect_summary_artifact_type": (
            "bayesfilter_sequential_rhat_checkpoint_inspection_summary"
        ),
        "inspect_summary_public_fields": (
            SEQUENTIAL_RHAT_CHECKPOINT_INSPECTION_PUBLIC_FIELDS
        ),
        "inspect_resume_policy": {
            "resume_supported_by_current_schema": False,
            "resume_status": "deferred_until_continuation_contract_is_tested",
            "inspection_does_not_publicize_private_paths": True,
            "inspection_does_not_publicize_raw_tensors": True,
            "inspection_does_not_publicize_hmc_mechanics": True,
        },
        "nonclaims": SEQUENTIAL_RHAT_CHECKPOINT_PUBLIC_NONCLAIMS,
    }
    payload = dict(contract)
    payload["contract_sha256"] = program_signature(contract)
    return _json_safe_metadata(payload)


def build_sequential_rhat_checkpoint_public_reference(
    *,
    checkpoint_kind: str,
    checkpoint_id: str,
    checkpoint_sha256: str,
) -> Mapping[str, Any]:
    """Build the only checkpoint reference shape allowed in public progress."""

    payload = {
        "artifact_type": "bayesfilter_sequential_rhat_checkpoint_public_reference",
        "schema_version": 1,
        "checkpoint_kind": str(checkpoint_kind),
        "checkpoint_id": str(checkpoint_id),
        "checkpoint_sha256": str(checkpoint_sha256),
        "contract_sha256": sequential_rhat_verification_checkpoint_contract()[
            "contract_sha256"
        ],
        "private_paths_publicized": False,
        "public_summary_contains_paths": False,
        "public_summary_contains_raw_values": False,
        "public_summary_contains_tensor_descriptors": False,
        "public_summary_contains_kernel_payload": False,
        "nonclaims": SEQUENTIAL_RHAT_CHECKPOINT_PUBLIC_NONCLAIMS,
    }
    assert_sequential_rhat_checkpoint_public_reference_safe(payload)
    return _json_safe_metadata(payload)


def assert_sequential_rhat_checkpoint_public_reference_safe(
    payload: Mapping[str, Any],
) -> None:
    """Reject public checkpoint references that leak paths or HMC mechanics."""

    fields = set(payload)
    expected = set(SEQUENTIAL_RHAT_CHECKPOINT_PUBLIC_REFERENCE_FIELDS)
    if fields != expected:
        extra = sorted(fields - expected)
        missing = sorted(expected - fields)
        raise ValueError(
            "sequential R-hat public checkpoint reference fields mismatch: "
            f"extra={extra} missing={missing}"
        )
    if payload["artifact_type"] != (
        "bayesfilter_sequential_rhat_checkpoint_public_reference"
    ):
        raise ValueError("unexpected sequential R-hat checkpoint artifact_type")
    if int(payload["schema_version"]) != 1:
        raise ValueError("unexpected sequential R-hat checkpoint schema_version")
    if str(payload["checkpoint_kind"]) not in SEQUENTIAL_RHAT_CHECKPOINT_KINDS:
        raise ValueError("checkpoint_kind is not recognized")
    checkpoint_id = str(payload["checkpoint_id"])
    if re.fullmatch(_SEQUENTIAL_RHAT_OPAQUE_CHECKPOINT_ID_PATTERN, checkpoint_id) is None:
        raise ValueError("checkpoint_id must be an opaque srhat-v1 hex handle")
    checkpoint_id_body = checkpoint_id.removeprefix("srhat-v1-")
    if any(token in checkpoint_id_body for token in _SEQUENTIAL_RHAT_FORBIDDEN_CHECKPOINT_ID_TOKENS):
        raise ValueError("checkpoint_id must not encode semantic checkpoint metadata")
    for key in ("checkpoint_sha256", "contract_sha256"):
        if re.fullmatch(_SHA256_HEX_PATTERN, str(payload[key])) is None:
            raise ValueError(f"{key} must be a lowercase SHA-256 hex digest")
    for key in (
        "private_paths_publicized",
        "public_summary_contains_paths",
        "public_summary_contains_raw_values",
        "public_summary_contains_tensor_descriptors",
        "public_summary_contains_kernel_payload",
    ):
        if bool(payload[key]):
            raise ValueError(f"{key} must be false for public checkpoint references")
    if tuple(payload["nonclaims"]) != SEQUENTIAL_RHAT_CHECKPOINT_PUBLIC_NONCLAIMS:
        raise ValueError("public checkpoint reference nonclaims mismatch")

    forbidden_key_tokens = tuple(
        sequential_rhat_verification_checkpoint_contract()[
            "forbidden_public_key_tokens"
        ]
    )

    def walk(value: Any, path: tuple[str, ...]) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                key_text = str(key)
                lowered = key_text.lower()
                if key_text not in SEQUENTIAL_RHAT_CHECKPOINT_PUBLIC_REFERENCE_FIELDS:
                    if any(token in lowered for token in forbidden_key_tokens):
                        raise ValueError(
                            "public checkpoint reference contains forbidden key: "
                            + ".".join(path + (key_text,))
                        )
                walk(item, path + (key_text,))
        elif isinstance(value, (tuple, list)):
            for index, item in enumerate(value):
                walk(item, path + (str(index),))
        elif isinstance(value, str):
            if "/" in value or "\\" in value:
                raise ValueError(
                    "public checkpoint reference contains path-shaped value: "
                    + ".".join(path)
                )
            if value.startswith(".") or value.startswith("~"):
                raise ValueError(
                    "public checkpoint reference contains relative path value: "
                    + ".".join(path)
                )

    walk(payload, tuple())


def write_sequential_rhat_pre_verification_handoff_checkpoint(
    *,
    writer_config: SequentialRHatCheckpointWriterConfig,
    adapter: Any,
    config_private_payload: Mapping[str, Any],
    selected_kernel_private_payload: Mapping[str, Any],
    mass_payload: Mapping[str, Any],
    final_state: Any,
    retained_count: int = 0,
) -> Mapping[str, Any]:
    """Write the private selected-kernel handoff before verification starts."""

    import tensorflow as tf

    root = Path(writer_config.checkpoint_dir)
    root.mkdir(parents=True, exist_ok=True)
    label = str(writer_config.checkpoint_label)
    checkpoint_id = "srhat-v1-" + secrets.token_hex(16)
    artifact_prefix = f"{label}_{checkpoint_id}"
    checkpoint_kind = "pre_verification_handoff"

    final_state_tensor = tf.convert_to_tensor(final_state)
    selected_payload = _json_safe_metadata(selected_kernel_private_payload)
    mass_private_payload = _json_safe_metadata(mass_payload)
    shard_sources = {
        "selected_kernel_private_payload": (selected_payload, "json"),
        "final_state": (final_state_tensor, "tensorflow.serialize_tensor"),
        "mass_payload": (mass_private_payload, "json"),
    }
    shards: dict[str, Mapping[str, Any]] = {}
    for role, (payload, serializer) in shard_sources.items():
        shard_path = root / f"{artifact_prefix}_{role}.{_sequential_rhat_shard_suffix(serializer)}"
        if shard_path.exists() and not writer_config.overwrite:
            raise FileExistsError(f"sequential R-hat checkpoint shard exists: {shard_path}")
        if serializer == "json":
            sha256, byte_count = _write_json_shard(shard_path, payload)
            shape: tuple[int, ...] = ()
            dtype = "json"
        else:
            tensor = tf.convert_to_tensor(payload)
            sha256, byte_count = _write_sequential_rhat_serialized_tensor_shard(
                shard_path,
                tensor,
            )
            shape = tuple(int(dim) for dim in tensor.shape)
            dtype = tensor.dtype.name
        shards[role] = {
            "role": role,
            "path": str(shard_path),
            "sha256": sha256,
            "bytes": int(byte_count),
            "shape": shape,
            "dtype": dtype,
            "serializer": serializer,
        }

    required_roles = tuple(
        sequential_rhat_verification_checkpoint_contract()[
            "private_manifest_required_shard_roles_by_kind"
        ][checkpoint_kind]
    )
    missing_roles = [role for role in required_roles if role not in shards]
    if missing_roles:
        raise ValueError(f"missing sequential R-hat checkpoint shards: {missing_roles}")
    manifest_core = {
        "artifact_type": "bayesfilter_private_sequential_rhat_checkpoint_manifest",
        "schema_version": 1,
        "checkpoint_kind": checkpoint_kind,
        "checkpoint_id": checkpoint_id,
        "manifest_path": str(root / f"{artifact_prefix}_manifest.json"),
        "config_private_signature": program_signature(config_private_payload),
        "adapter_signature": stable_adapter_signature(adapter),
        "created_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "private_shards": shards,
        "private_diagnostics": {
            "chunk_index": None,
            "retained_sample_count": int(retained_count),
            "valid_sample_count": None,
            "nonfinite_valid_sample_count": None,
            "rhat_summary": {
                "rhat_threshold": _tensor_or_plain_to_metadata(
                    config_private_payload.get("rhat_threshold")
                ),
                "max_finite_rhat": None,
                "finite_rhat_count": 0,
                "nonfinite_rhat_count": 0,
                "all_finite_rhat_at_or_below_threshold": False,
            },
            "acceptance_summary": {
                "acceptance_rate": None,
                "acceptance_decision_count": None,
            },
            "target_log_prob_summary": {
                "finite_count": 0,
                "nonfinite_count": 0,
                "min_finite": None,
                "max_finite": None,
            },
            "divergence_summary": {
                "status": "not_collected",
                "count": None,
            },
        },
        "privacy_contract": {
            "manifest_contains_private_paths": True,
            "manifest_contains_private_raw_tensors": False,
            "manifest_points_to_private_raw_tensors": True,
            "public_reference_contains_paths": False,
            "public_reference_contains_tensor_descriptors": False,
            "public_reference_contains_hmc_mechanics": False,
        },
        "nonclaims": SEQUENTIAL_RHAT_CHECKPOINT_PUBLIC_NONCLAIMS,
    }
    manifest_core_text = json.dumps(
        _json_safe_metadata(manifest_core),
        indent=2,
        sort_keys=True,
    ) + "\n"
    manifest_payload = dict(manifest_core)
    manifest_payload["manifest_core_sha256"] = hashlib.sha256(
        manifest_core_text.encode("utf-8")
    ).hexdigest()
    manifest_path = Path(str(manifest_core["manifest_path"]))
    if manifest_path.exists() and not writer_config.overwrite:
        raise FileExistsError(f"sequential R-hat checkpoint manifest exists: {manifest_path}")
    _atomic_write_json(manifest_path, manifest_payload)
    return build_sequential_rhat_checkpoint_public_reference(
        checkpoint_kind=checkpoint_kind,
        checkpoint_id=checkpoint_id,
        checkpoint_sha256=manifest_payload["manifest_core_sha256"],
    )


def write_sequential_rhat_boundary_handoff_checkpoint(
    *,
    writer_config: SequentialRHatCheckpointWriterConfig,
    adapter: Any,
    config_private_payload: Mapping[str, Any],
    boundary_private_payload: Mapping[str, Any],
    state_summary_private_payload: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Write a private Phase 7 boundary handoff before a long HMC subcall.

    Boundary checkpoints are forced-stop observability artifacts only.  They are
    written outside the TensorFlow/TFP transition graph and deliberately contain
    JSON summary/provenance shards only: no raw samples, final/current states,
    traces, mass payloads, selected-kernel payloads, step sizes, or leapfrog
    counts.
    """

    root = Path(writer_config.checkpoint_dir)
    root.mkdir(parents=True, exist_ok=True)
    label = str(writer_config.checkpoint_label)
    checkpoint_id = "srhat-v1-" + secrets.token_hex(16)
    artifact_prefix = f"{label}_{checkpoint_id}"
    checkpoint_kind = "phase7_boundary_handoff"

    boundary_payload = _json_safe_metadata(
        {
            **dict(boundary_private_payload),
            "checkpoint_kind": checkpoint_kind,
            "private_raw_state_allowed": False,
            "private_raw_samples_allowed": False,
            "private_hmc_mechanics_allowed": False,
            "diagnostic_role": "forced_stop_observability_only",
            "reports_verifier_entry_durability": False,
            "reports_posterior_convergence": False,
            "reports_sampler_superiority": False,
            "reports_scientific_validity": False,
            "reports_resume_readiness": False,
        }
    )
    config_payload = _json_safe_metadata(config_private_payload)
    state_summary_payload = _json_safe_metadata(
        {
            **({} if state_summary_private_payload is None else dict(state_summary_private_payload)),
            "summary_only": True,
            "raw_state_included": False,
            "raw_samples_included": False,
            "tensor_payload_included": False,
        }
    )
    shard_sources = {
        "boundary_private_payload": (boundary_payload, "json"),
        "config_private_payload": (config_payload, "json"),
        "state_summary_private_payload": (state_summary_payload, "json"),
    }
    shards: dict[str, Mapping[str, Any]] = {}
    for role, (payload, serializer) in shard_sources.items():
        shard_path = root / f"{artifact_prefix}_{role}.{_sequential_rhat_shard_suffix(serializer)}"
        if shard_path.exists() and not writer_config.overwrite:
            raise FileExistsError(f"sequential R-hat checkpoint shard exists: {shard_path}")
        sha256, byte_count = _write_json_shard(shard_path, payload)
        shards[role] = {
            "role": role,
            "path": str(shard_path),
            "sha256": sha256,
            "bytes": int(byte_count),
            "shape": (),
            "dtype": "json",
            "serializer": serializer,
        }

    required_roles = tuple(
        sequential_rhat_verification_checkpoint_contract()[
            "private_manifest_required_shard_roles_by_kind"
        ][checkpoint_kind]
    )
    missing_roles = [role for role in required_roles if role not in shards]
    if missing_roles:
        raise ValueError(f"missing sequential R-hat checkpoint shards: {missing_roles}")
    manifest_core = {
        "artifact_type": "bayesfilter_private_sequential_rhat_checkpoint_manifest",
        "schema_version": 1,
        "checkpoint_kind": checkpoint_kind,
        "checkpoint_id": checkpoint_id,
        "manifest_path": str(root / f"{artifact_prefix}_manifest.json"),
        "config_private_signature": program_signature(config_private_payload),
        "adapter_signature": stable_adapter_signature(adapter),
        "created_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "private_shards": shards,
        "private_diagnostics": {
            "chunk_index": None,
            "retained_sample_count": 0,
            "valid_sample_count": None,
            "nonfinite_valid_sample_count": None,
            "rhat_summary": {
                "rhat_threshold": None,
                "max_finite_rhat": None,
                "finite_rhat_count": 0,
                "nonfinite_rhat_count": 0,
                "all_finite_rhat_at_or_below_threshold": False,
            },
            "acceptance_summary": {
                "acceptance_rate": None,
                "acceptance_decision_count": None,
            },
            "target_log_prob_summary": {
                "finite_count": 0,
                "nonfinite_count": 0,
                "min_finite": None,
                "max_finite": None,
            },
            "divergence_summary": {
                "status": "not_collected",
                "count": None,
            },
        },
        "privacy_contract": {
            "manifest_contains_private_paths": True,
            "manifest_contains_private_raw_tensors": False,
            "manifest_points_to_private_raw_tensors": False,
            "public_reference_contains_paths": False,
            "public_reference_contains_tensor_descriptors": False,
            "public_reference_contains_hmc_mechanics": False,
        },
        "nonclaims": SEQUENTIAL_RHAT_CHECKPOINT_PUBLIC_NONCLAIMS,
    }
    manifest_core_text = json.dumps(
        _json_safe_metadata(manifest_core),
        indent=2,
        sort_keys=True,
    ) + "\n"
    manifest_payload = dict(manifest_core)
    manifest_payload["manifest_core_sha256"] = hashlib.sha256(
        manifest_core_text.encode("utf-8")
    ).hexdigest()
    manifest_path = Path(str(manifest_core["manifest_path"]))
    if manifest_path.exists() and not writer_config.overwrite:
        raise FileExistsError(f"sequential R-hat checkpoint manifest exists: {manifest_path}")
    _atomic_write_json(manifest_path, manifest_payload)
    return build_sequential_rhat_checkpoint_public_reference(
        checkpoint_kind=checkpoint_kind,
        checkpoint_id=checkpoint_id,
        checkpoint_sha256=manifest_payload["manifest_core_sha256"],
    )


def _write_sequential_rhat_private_checkpoint(
    *,
    writer_config: SequentialRHatCheckpointWriterConfig,
    checkpoint_kind: str,
    adapter: Any,
    config: SequentialRHatHMCVerificationConfig,
    chunk_index: int,
    retained_count: int,
    chunk_result: FixedSizeHMCChunkRunResult,
    chunk_summary: Mapping[str, Any],
    rhat_summary: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Write one private verification checkpoint and return a public reference."""

    import tensorflow as tf

    if str(checkpoint_kind) != "verification_chunk":
        raise ValueError("Phase 2 writer supports verification_chunk checkpoints only")
    root = Path(writer_config.checkpoint_dir)
    root.mkdir(parents=True, exist_ok=True)
    label = str(writer_config.checkpoint_label)
    checkpoint_id = "srhat-v1-" + secrets.token_hex(16)
    artifact_prefix = f"{label}_{checkpoint_id}"

    sample_tensor = tf.convert_to_tensor(chunk_result.samples)
    valid_mask_tensor = tf.convert_to_tensor(chunk_result.valid_mask, dtype=tf.bool)
    final_state_tensor = tf.convert_to_tensor(chunk_result.final_state)
    reduced_trace = {
        "diagnostics": _json_safe_metadata(
            {
                key: _tensor_or_plain_to_metadata(value)
                for key, value in chunk_result.diagnostics.items()
            }
        ),
        "metadata": _json_safe_metadata(
            {
                key: _tensor_or_plain_to_metadata(value)
                for key, value in chunk_result.metadata.items()
            }
        ),
    }
    target_log_prob_summary = _sequential_rhat_target_log_prob_summary(
        chunk_result.diagnostics
    )
    log_accept_ratio_summary = _sequential_rhat_log_accept_ratio_summary(
        chunk_result.diagnostics
    )
    rhat_payload = _sequential_rhat_checkpoint_rhat_summary(config, rhat_summary)

    shard_sources = {
        "samples": (sample_tensor, "tensorflow.serialize_tensor"),
        "valid_mask": (valid_mask_tensor, "tensorflow.serialize_tensor"),
        "final_state": (final_state_tensor, "tensorflow.serialize_tensor"),
        "reduced_trace": (reduced_trace, "json"),
        "target_log_prob_summary": (target_log_prob_summary, "json"),
        "log_accept_ratio_summary": (log_accept_ratio_summary, "json"),
        "rhat_summary": (rhat_payload, "json"),
    }
    shards: dict[str, Mapping[str, Any]] = {}
    for role, (payload, serializer) in shard_sources.items():
        shard_path = root / f"{artifact_prefix}_{role}.{_sequential_rhat_shard_suffix(serializer)}"
        if shard_path.exists() and not writer_config.overwrite:
            raise FileExistsError(f"sequential R-hat checkpoint shard exists: {shard_path}")
        if serializer == "json":
            sha256, byte_count = _write_json_shard(shard_path, payload)
            shape: tuple[int, ...] = ()
            dtype = "json"
        else:
            tensor = tf.convert_to_tensor(payload)
            sha256, byte_count = _write_sequential_rhat_serialized_tensor_shard(
                shard_path,
                tensor,
            )
            shape = tuple(int(dim) for dim in tensor.shape)
            dtype = tensor.dtype.name
        shards[role] = {
            "role": role,
            "path": str(shard_path),
            "sha256": sha256,
            "bytes": int(byte_count),
            "shape": shape,
            "dtype": dtype,
            "serializer": serializer,
        }

    required_roles = tuple(
        sequential_rhat_verification_checkpoint_contract()[
            "private_manifest_required_shard_roles_by_kind"
        ][str(checkpoint_kind)]
    )
    missing_roles = [role for role in required_roles if role not in shards]
    if missing_roles:
        raise ValueError(f"missing sequential R-hat checkpoint shards: {missing_roles}")
    manifest_core = {
        "artifact_type": "bayesfilter_private_sequential_rhat_checkpoint_manifest",
        "schema_version": 1,
        "checkpoint_kind": str(checkpoint_kind),
        "checkpoint_id": checkpoint_id,
        "manifest_path": str(root / f"{artifact_prefix}_manifest.json"),
        "config_private_signature": program_signature(config.signature_payload()),
        "adapter_signature": stable_adapter_signature(adapter),
        "created_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "private_shards": shards,
        "private_diagnostics": {
            "chunk_index": int(chunk_index),
            "retained_sample_count": int(retained_count),
            "valid_sample_count": chunk_summary.get("valid_sample_count"),
            "nonfinite_valid_sample_count": chunk_summary.get(
                "nonfinite_valid_sample_count"
            ),
            "rhat_summary": rhat_payload,
            "acceptance_summary": _sequential_rhat_acceptance_summary(chunk_summary),
            "target_log_prob_summary": target_log_prob_summary,
            "divergence_summary": _sequential_rhat_divergence_summary(chunk_summary),
        },
        "privacy_contract": {
            "manifest_contains_private_paths": True,
            "manifest_contains_private_raw_tensors": False,
            "manifest_points_to_private_raw_tensors": True,
            "public_reference_contains_paths": False,
            "public_reference_contains_tensor_descriptors": False,
            "public_reference_contains_hmc_mechanics": False,
        },
        "nonclaims": SEQUENTIAL_RHAT_CHECKPOINT_PUBLIC_NONCLAIMS,
    }
    manifest_core_text = json.dumps(
        _json_safe_metadata(manifest_core),
        indent=2,
        sort_keys=True,
    ) + "\n"
    manifest_payload = dict(manifest_core)
    manifest_payload["manifest_core_sha256"] = hashlib.sha256(
        manifest_core_text.encode("utf-8")
    ).hexdigest()
    manifest_path = Path(str(manifest_core["manifest_path"]))
    if manifest_path.exists() and not writer_config.overwrite:
        raise FileExistsError(f"sequential R-hat checkpoint manifest exists: {manifest_path}")
    _atomic_write_json(manifest_path, manifest_payload)
    return build_sequential_rhat_checkpoint_public_reference(
        checkpoint_kind=str(checkpoint_kind),
        checkpoint_id=checkpoint_id,
        checkpoint_sha256=manifest_payload["manifest_core_sha256"],
    )


def inspect_sequential_rhat_private_checkpoint(
    *,
    manifest_path: str | Path,
    public_reference: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Verify a private sequential R-hat checkpoint and summarize it safely.

    The public checkpoint reference is intentionally path-free, so callers must
    supply the private manifest path from their private artifact catalog.  This
    helper verifies the manifest core hash and every private payload hash before
    returning aggregate diagnostics.  It does not deserialize raw retained
    draws, continuation positions, selected-kernel payloads, or mass payloads;
    resume remains explicitly deferred until a separate continuation contract is
    specified and tested.
    """

    path = Path(manifest_path)
    manifest = _load_sequential_rhat_private_checkpoint_manifest(path)
    contract = sequential_rhat_verification_checkpoint_contract()
    checkpoint_kind = str(manifest["checkpoint_kind"])
    checkpoint_id = str(manifest["checkpoint_id"])
    checkpoint_sha256 = str(manifest["manifest_core_sha256"])

    if public_reference is not None:
        assert_sequential_rhat_checkpoint_public_reference_safe(public_reference)
        mismatched_fields = [
            field
            for field in ("checkpoint_kind", "checkpoint_id", "checkpoint_sha256")
            if str(public_reference[field]) != str(
                checkpoint_sha256 if field == "checkpoint_sha256" else manifest[field]
            )
        ]
        if mismatched_fields:
            raise ValueError(
                "public checkpoint reference does not match private manifest: "
                f"{mismatched_fields}"
            )

    required_roles = tuple(
        contract["private_manifest_required_shard_roles_by_kind"][checkpoint_kind]
    )
    private_shards = manifest["private_shards"]
    missing_roles = [role for role in required_roles if role not in private_shards]
    if missing_roles:
        raise ValueError(f"missing sequential R-hat checkpoint shards: {missing_roles}")
    _verify_sequential_rhat_private_checkpoint_shards(path, private_shards)
    private_json_summaries = _load_sequential_rhat_private_checkpoint_json_summaries(
        path,
        private_shards,
        roles=(
            "target_log_prob_summary",
            "log_accept_ratio_summary",
            "rhat_summary",
        ),
    )

    diagnostics = manifest["private_diagnostics"]
    rhat_summary = _public_sequential_rhat_inspection_rhat_summary(
        private_json_summaries.get("rhat_summary", diagnostics.get("rhat_summary"))
    )
    target_value_health = _public_sequential_rhat_inspection_target_health(
        private_json_summaries.get(
            "target_log_prob_summary",
            diagnostics.get("target_log_prob_summary"),
        )
    )
    acceptance_log_health = _public_sequential_rhat_inspection_acceptance_log_health(
        private_json_summaries.get("log_accept_ratio_summary")
    )
    if checkpoint_kind == "phase7_boundary_handoff":
        finite_checks = {
            "retained_values_finite": None,
            "retained_values_status": "not_applicable_boundary_handoff",
            "private_target_value_health_passed": (
                target_value_health["nonfinite_count"] == 0
            ),
            "private_acceptance_log_health_passed": None,
            "private_acceptance_log_status": "not_applicable_boundary_handoff",
            "rhat_all_finite": int(rhat_summary["nonfinite_rhat_count"]) == 0,
        }
    else:
        finite_checks = {
            "retained_values_finite": (
                _int_or_none_metadata(diagnostics.get("nonfinite_valid_sample_count"))
                == 0
            ),
            "private_target_value_health_passed": (
                target_value_health["nonfinite_count"] == 0
            ),
            "private_acceptance_log_health_passed": (
                acceptance_log_health["nonfinite_count"] == 0
            ),
            "rhat_all_finite": int(rhat_summary["nonfinite_rhat_count"]) == 0,
        }
    summary = {
        "artifact_type": "bayesfilter_sequential_rhat_checkpoint_inspection_summary",
        "schema_version": 1,
        "checkpoint_kind": checkpoint_kind,
        "checkpoint_id": checkpoint_id,
        "checkpoint_sha256": checkpoint_sha256,
        "contract_sha256": contract["contract_sha256"],
        "manifest_core_hash_verified": True,
        "private_payload_hashes_verified": True,
        "private_payload_count": int(len(private_shards)),
        "required_private_payloads_present": True,
        "retained_sample_count": _int_or_none_metadata(
            diagnostics.get("retained_sample_count")
        ),
        "latest_chunk_index": _int_or_none_metadata(diagnostics.get("chunk_index")),
        "valid_sample_count": _int_or_none_metadata(
            diagnostics.get("valid_sample_count")
        ),
        "nonfinite_valid_sample_count": _int_or_none_metadata(
            diagnostics.get("nonfinite_valid_sample_count")
        ),
        "rhat_summary": rhat_summary,
        "acceptance_summary": _public_sequential_rhat_inspection_acceptance_summary(
            diagnostics.get("acceptance_summary")
        ),
        "finite_checks": finite_checks,
        "divergence_summary": _public_sequential_rhat_inspection_divergence_summary(
            diagnostics.get("divergence_summary")
        ),
        "provenance": {
            "config_private_signature": str(manifest["config_private_signature"]),
            "adapter_signature_sha256": hashlib.sha256(
                str(manifest["adapter_signature"]).encode("utf-8")
            ).hexdigest(),
            "created_at_utc": str(manifest["created_at_utc"]),
        },
        "resume": {
            "supported": False,
            "status": "deferred",
            "reason_code": "incomplete_continuation_contract",
            "required_future_contract": (
                "validated continuation-position schema",
                "validated random-seed continuation schedule",
                "retained-draw reconstruction policy",
                "checkpoint-to-verifier config replay contract",
            ),
        },
        "privacy_contract": {
            "private_paths_publicized": False,
            "public_summary_contains_paths": False,
            "public_summary_contains_raw_values": False,
            "public_summary_contains_tensor_descriptors": False,
            "public_summary_contains_kernel_payload": False,
            "public_summary_contains_hmc_mechanics": False,
        },
        "nonclaims": SEQUENTIAL_RHAT_CHECKPOINT_INSPECTION_NONCLAIMS,
    }
    _assert_sequential_rhat_checkpoint_inspection_public_safe(summary)
    return _json_safe_metadata(summary)


def _load_sequential_rhat_private_checkpoint_manifest(
    path: Path,
) -> Mapping[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"sequential R-hat checkpoint manifest not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("sequential R-hat checkpoint manifest must be a JSON object")
    contract = sequential_rhat_verification_checkpoint_contract()
    missing_fields = [
        field for field in contract["private_manifest_required_fields"] if field not in payload
    ]
    if missing_fields:
        raise ValueError(
            "sequential R-hat checkpoint manifest missing fields: "
            f"{missing_fields}"
        )
    if payload["artifact_type"] != contract["private_manifest_artifact_type"]:
        raise ValueError("unexpected sequential R-hat checkpoint manifest artifact_type")
    if int(payload["schema_version"]) != 1:
        raise ValueError("unexpected sequential R-hat checkpoint manifest schema_version")
    checkpoint_kind = str(payload["checkpoint_kind"])
    if checkpoint_kind not in SEQUENTIAL_RHAT_CHECKPOINT_KINDS:
        raise ValueError("checkpoint_kind is not recognized")
    checkpoint_id = str(payload["checkpoint_id"])
    if re.fullmatch(_SEQUENTIAL_RHAT_OPAQUE_CHECKPOINT_ID_PATTERN, checkpoint_id) is None:
        raise ValueError("checkpoint_id must be an opaque srhat-v1 hex handle")
    manifest_hash = str(payload["manifest_core_sha256"])
    if re.fullmatch(_SHA256_HEX_PATTERN, manifest_hash) is None:
        raise ValueError("manifest_core_sha256 must be a lowercase SHA-256 hex digest")
    core = dict(payload)
    observed_hash = str(core.pop("manifest_core_sha256"))
    core_text = json.dumps(
        _json_safe_metadata(core),
        indent=2,
        sort_keys=True,
    ) + "\n"
    expected_hash = hashlib.sha256(core_text.encode("utf-8")).hexdigest()
    if observed_hash != expected_hash:
        raise ValueError("sequential R-hat checkpoint manifest_core_sha256 mismatch")
    if tuple(payload["nonclaims"]) != SEQUENTIAL_RHAT_CHECKPOINT_PUBLIC_NONCLAIMS:
        raise ValueError("sequential R-hat checkpoint manifest nonclaims mismatch")
    private_shards = payload["private_shards"]
    if not isinstance(private_shards, Mapping):
        raise ValueError("sequential R-hat checkpoint private_shards must be a mapping")
    diagnostics = payload["private_diagnostics"]
    if not isinstance(diagnostics, Mapping):
        raise ValueError(
            "sequential R-hat checkpoint private_diagnostics must be a mapping"
        )
    return payload


def _verify_sequential_rhat_private_checkpoint_shards(
    manifest_path: Path,
    private_shards: Mapping[str, Any],
) -> None:
    valid_roles = set(
        sequential_rhat_verification_checkpoint_contract()["private_shard_roles"]
    )
    for role, shard in private_shards.items():
        role_text = str(role)
        if role_text not in valid_roles:
            raise ValueError(f"unrecognized sequential R-hat checkpoint shard: {role_text}")
        if not isinstance(shard, Mapping):
            raise ValueError("sequential R-hat checkpoint shard record must be a mapping")
        if str(shard.get("role")) != role_text:
            raise ValueError(
                "sequential R-hat checkpoint shard role mismatch: "
                f"{role_text}"
            )
        shard_hash = str(shard.get("sha256"))
        if re.fullmatch(_SHA256_HEX_PATTERN, shard_hash) is None:
            raise ValueError("checkpoint shard sha256 must be a lowercase SHA-256 digest")
        serializer = str(shard.get("serializer"))
        if serializer not in {"json", "tensorflow.serialize_tensor"}:
            raise ValueError(
                "sequential R-hat checkpoint shard serializer is not recognized"
            )
        shard_path = _resolve_sequential_rhat_private_checkpoint_path(
            str(shard.get("path")),
            manifest_path.parent,
        )
        if not shard_path.exists():
            raise FileNotFoundError(
                "sequential R-hat checkpoint shard not found: "
                f"{role_text}"
            )
        data = shard_path.read_bytes()
        expected_bytes = int(shard.get("bytes"))
        if len(data) != expected_bytes:
            raise ValueError(
                "sequential R-hat checkpoint shard byte count mismatch: "
                f"{role_text}"
            )
        observed_hash = hashlib.sha256(data).hexdigest()
        if observed_hash != shard_hash:
            raise ValueError(
                "sequential R-hat checkpoint shard hash mismatch: "
                f"{role_text}"
            )


def _load_sequential_rhat_private_checkpoint_json_summaries(
    manifest_path: Path,
    private_shards: Mapping[str, Any],
    *,
    roles: Sequence[str],
) -> Mapping[str, Any]:
    summaries: dict[str, Any] = {}
    for role in roles:
        shard = private_shards.get(role)
        if shard is None:
            continue
        if str(shard.get("serializer")) != "json":
            continue
        shard_path = _resolve_sequential_rhat_private_checkpoint_path(
            str(shard.get("path")),
            manifest_path.parent,
        )
        summaries[role] = json.loads(shard_path.read_text(encoding="utf-8"))
    return summaries


def _resolve_sequential_rhat_private_checkpoint_path(
    path_text: str,
    manifest_dir: Path,
) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    candidates = (
        path,
        manifest_dir / path,
        manifest_dir / path.name,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return path


def _public_sequential_rhat_inspection_rhat_summary(value: Any) -> Mapping[str, Any]:
    rhat_summary = value if isinstance(value, Mapping) else {}
    return {
        "rhat_threshold": _float_or_none_metadata(rhat_summary.get("rhat_threshold")),
        "max_finite_rhat": _float_or_none_metadata(
            rhat_summary.get("max_finite_rhat")
        ),
        "finite_rhat_count": _int_or_none_metadata(
            rhat_summary.get("finite_rhat_count")
        ) or 0,
        "nonfinite_rhat_count": _int_or_none_metadata(
            rhat_summary.get("nonfinite_rhat_count")
        ) or 0,
        "all_finite_rhat_at_or_below_threshold": bool(
            rhat_summary.get("all_finite_rhat_at_or_below_threshold", False)
        ),
    }


def _public_sequential_rhat_inspection_acceptance_summary(
    value: Any,
) -> Mapping[str, Any]:
    acceptance_summary = value if isinstance(value, Mapping) else {}
    return {
        "acceptance_rate": _float_or_none_metadata(
            acceptance_summary.get("acceptance_rate")
        ),
        "acceptance_decision_count": _int_or_none_metadata(
            acceptance_summary.get("acceptance_decision_count")
        ),
    }


def _public_sequential_rhat_inspection_target_health(value: Any) -> Mapping[str, Any]:
    target_summary = value if isinstance(value, Mapping) else {}
    min_finite = _float_or_none_metadata(target_summary.get("min_finite"))
    max_finite = _float_or_none_metadata(target_summary.get("max_finite"))
    return {
        "finite_count": _int_or_none_metadata(target_summary.get("finite_count")),
        "nonfinite_count": _int_or_none_metadata(
            target_summary.get("nonfinite_count")
        ),
        "finite_extrema_recorded": min_finite is not None and max_finite is not None,
    }


def _public_sequential_rhat_inspection_acceptance_log_health(
    value: Any,
) -> Mapping[str, Any]:
    acceptance_summary = value if isinstance(value, Mapping) else {}
    max_abs_finite = _float_or_none_metadata(acceptance_summary.get("max_abs_finite"))
    return {
        "finite_count": _int_or_none_metadata(acceptance_summary.get("finite_count")),
        "nonfinite_count": _int_or_none_metadata(
            acceptance_summary.get("nonfinite_count")
        ),
        "finite_extreme_recorded": max_abs_finite is not None,
    }


def _public_sequential_rhat_inspection_divergence_summary(
    value: Any,
) -> Mapping[str, Any]:
    divergence_summary = value if isinstance(value, Mapping) else {}
    return {
        "divergence_status": divergence_summary.get(
            "divergence_status",
            divergence_summary.get("status"),
        ),
        "divergence_count": _int_or_none_metadata(
            divergence_summary.get("divergence_count", divergence_summary.get("count"))
        ),
    }


def _assert_sequential_rhat_checkpoint_inspection_public_safe(
    payload: Mapping[str, Any],
) -> None:
    fields = set(payload)
    expected = set(SEQUENTIAL_RHAT_CHECKPOINT_INSPECTION_PUBLIC_FIELDS)
    if fields != expected:
        extra = sorted(fields - expected)
        missing = sorted(expected - fields)
        raise ValueError(
            "sequential R-hat checkpoint inspection fields mismatch: "
            f"extra={extra} missing={missing}"
        )
    forbidden_key_tokens = (
        "manifest_path",
        "sample_path",
        "state_path",
        "samples",
        "valid_mask",
        "final_state",
        "trace",
        "target_log_prob",
        "log_accept",
        "step_size",
        "leapfrog",
        "mass",
        "selected_kernel",
        "shape",
        "dtype",
        "serializer",
    )
    forbidden_value_tokens = (
        "/",
        "\\",
        ".tftensor",
        "target_log_prob",
        "log_accept",
        "step_size",
        "num_leapfrog_steps",
        "selected_kernel",
        "final_state",
        "valid_mask",
    )

    def walk(value: Any, path: tuple[str, ...]) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                key_text = str(key)
                lowered = key_text.lower()
                if any(token in lowered for token in forbidden_key_tokens):
                    raise ValueError(
                        "checkpoint inspection summary contains forbidden key: "
                        + ".".join(path + (key_text,))
                    )
                walk(item, path + (key_text,))
        elif isinstance(value, (tuple, list)):
            for index, item in enumerate(value):
                walk(item, path + (str(index),))
        elif isinstance(value, str):
            if any(token in value for token in forbidden_value_tokens):
                raise ValueError(
                    "checkpoint inspection summary contains forbidden value: "
                    + ".".join(path)
                )
            if value.startswith(".") or value.startswith("~"):
                raise ValueError(
                    "checkpoint inspection summary contains relative path value: "
                    + ".".join(path)
                )

    walk(payload, tuple())


def _sequential_rhat_shard_suffix(serializer: str) -> str:
    if serializer == "json":
        return "json"
    return "tftensor"


def _write_json_shard(path: Path, payload: Any) -> tuple[str, int]:
    text = json.dumps(_json_safe_metadata(payload), indent=2, sort_keys=True) + "\n"
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(path)
    return hashlib.sha256(text.encode("utf-8")).hexdigest(), len(text.encode("utf-8"))


def _write_sequential_rhat_serialized_tensor_shard(
    path: Path,
    tensor: Any,
) -> tuple[str, int]:
    """Atomically write one private sequential R-hat tensor shard."""

    import tensorflow as tf

    serialized = tf.io.serialize_tensor(tf.convert_to_tensor(tensor))
    data = bytes(serialized.numpy())
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_bytes(data)
    tmp_path.replace(path)
    return hashlib.sha256(data).hexdigest(), len(data)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    text = json.dumps(_json_safe_metadata(payload), indent=2, sort_keys=True) + "\n"
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(path)


def _sequential_rhat_checkpoint_rhat_summary(
    config: SequentialRHatHMCVerificationConfig,
    rhat_summary: Mapping[str, Any],
) -> Mapping[str, Any]:
    return {
        "rhat_threshold": float(config.rhat_threshold),
        "max_finite_rhat": rhat_summary.get("max_finite_rhat"),
        "finite_rhat_count": int(rhat_summary.get("finite_rhat_count", 0)),
        "nonfinite_rhat_count": int(rhat_summary.get("nonfinite_rhat_count", 0)),
        "all_finite_rhat_at_or_below_threshold": bool(rhat_summary.get("passed", False)),
    }


def _sequential_rhat_acceptance_summary(chunk_summary: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "acceptance_rate": chunk_summary.get("acceptance_rate"),
        "acceptance_decision_count": chunk_summary.get("acceptance_decision_count"),
    }


def _sequential_rhat_target_log_prob_summary(
    diagnostics: Mapping[str, Any],
) -> Mapping[str, Any]:
    finite_count = _int_or_none_metadata(diagnostics.get("target_log_prob_finite_count"))
    min_finite = _float_or_none_metadata(diagnostics.get("target_log_prob_min_finite"))
    max_finite = _float_or_none_metadata(diagnostics.get("target_log_prob_max_finite"))
    if finite_count is None or finite_count <= 0:
        min_finite = None
        max_finite = None
    if min_finite is not None and not np.isfinite(min_finite):
        min_finite = None
    if max_finite is not None and not np.isfinite(max_finite):
        max_finite = None
    return {
        "finite_count": finite_count,
        "nonfinite_count": _int_or_none_metadata(
            diagnostics.get("target_log_prob_nonfinite_count")
        ),
        "min_finite": min_finite,
        "max_finite": max_finite,
    }


def _sequential_rhat_log_accept_ratio_summary(
    diagnostics: Mapping[str, Any],
) -> Mapping[str, Any]:
    finite_count = _int_or_none_metadata(diagnostics.get("log_accept_ratio_finite_count"))
    max_abs_finite = _float_or_none_metadata(
        diagnostics.get("log_accept_ratio_max_abs_finite")
    )
    if finite_count is None or finite_count <= 0:
        max_abs_finite = None
    if max_abs_finite is not None and not np.isfinite(max_abs_finite):
        max_abs_finite = None
    return {
        "finite_count": finite_count,
        "nonfinite_count": _int_or_none_metadata(
            diagnostics.get("log_accept_ratio_nonfinite_count")
        ),
        "max_abs_finite": max_abs_finite,
    }


def _sequential_rhat_divergence_summary(chunk_summary: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "divergence_status": chunk_summary.get("divergence_status"),
        "divergence_count": chunk_summary.get("divergence_count"),
    }


class ReusableFullChainHMCRunner:
    """Reusable compiled TFP HMC runner for a fixed static HMC contract.

    The ordinary ``run_full_chain_tfp_hmc`` helper builds a TensorFlow/TFP
    ``sample_chain`` callable and immediately calls it once.  That is fine for
    tiny gates, but expensive model clients may need to amortize TensorFlow's
    first-call graph construction.  This runner owns the same BayesFilter HMC
    target, kernel, trace, authority checks, and result schema, but compiles the
    chain callable once for a static state shape.  Per-call ``current_state``,
    ``seed``, and scalar ``step_size`` are tensor arguments, so repeated calls
    can reuse the compiled graph without changing HMC semantics.

    The interface is deliberately narrow: ``num_results``, burn-in length,
    leapfrog count, trace policy, tuning policy, XLA flag, and target scope are
    fixed by ``FullChainHMCConfig``.  A different value for any of those fields
    requires a different runner.
    """

    def __init__(
        self,
        adapter: Any,
        initial_state_template: Any,
        config: FullChainHMCConfig,
        *,
        dynamic_num_leapfrog_steps: bool = False,
    ) -> None:
        self.adapter = adapter
        self.config = config
        self.dynamic_num_leapfrog_steps = bool(dynamic_num_leapfrog_steps)
        self.capability = _validate_full_chain_hmc_authority(adapter, config)

        import tensorflow as tf

        template = tf.cast(tf.convert_to_tensor(initial_state_template), tf.float64)
        if template.shape.rank is None:
            raise ValueError("reusable HMC runner requires static state rank")
        if any(dim is None for dim in template.shape):
            raise ValueError("reusable HMC runner requires fully static state shape")
        self._state_shape = tuple(int(dim) for dim in template.shape)
        self._state_dtype = template.dtype
        self._initial_state_template = template
        self._target_log_prob = _make_tfp_target_log_prob_fn(
            adapter,
            dtype=self._state_dtype,
        )
        self._trace_fn = _trace_fn_for_config(config, adapter=adapter)
        runner_build_start = time.perf_counter()
        self._runner = self._build_runner()
        self._runner_build_s = time.perf_counter() - runner_build_start
        self._call_count = 0
        self._first_call_s: float | None = None
        self._warm_call_s: float | None = None

    @property
    def state_shape(self) -> tuple[int, ...]:
        return self._state_shape

    @property
    def state_dtype(self) -> str:
        return self._state_dtype.name

    def run(
        self,
        *,
        current_state: Any | None = None,
        seed: tuple[int, int] | Any | None = None,
        step_size: float | Any | None = None,
        num_leapfrog_steps: int | Any | None = None,
    ) -> FullChainHMCRunResult:
        """Run one chain call with tensor-parameterized dynamic inputs."""

        import tensorflow as tf

        state = self._initial_state_template if current_state is None else current_state
        state_tensor = tf.convert_to_tensor(state, dtype=self._state_dtype)
        if tuple(state_tensor.shape.as_list()) != self._state_shape:
            raise ValueError("current_state shape must match reusable runner template")
        seed_value = self.config.seed if seed is None else seed
        seed_tensor = tf.convert_to_tensor(seed_value, dtype=tf.int32)
        if tuple(seed_tensor.shape.as_list()) != (2,):
            raise ValueError("seed must have static shape (2,)")
        step_value = self.config.step_size if step_size is None else step_size
        step_tensor = tf.convert_to_tensor(step_value, dtype=self._state_dtype)
        if step_tensor.shape.rank != 0:
            raise ValueError("step_size must be a scalar")
        leapfrog_value = (
            self.config.num_leapfrog_steps
            if num_leapfrog_steps is None
            else num_leapfrog_steps
        )
        leapfrog_tensor = tf.convert_to_tensor(leapfrog_value, dtype=tf.int32)
        if leapfrog_tensor.shape.rank != 0:
            raise ValueError("num_leapfrog_steps must be a scalar")
        if not self.dynamic_num_leapfrog_steps:
            static_l = tf.get_static_value(leapfrog_tensor)
            if static_l is None or int(static_l) != int(self.config.num_leapfrog_steps):
                raise ValueError(
                    "num_leapfrog_steps can vary only when the reusable runner was "
                    "built with dynamic_num_leapfrog_steps=True"
                )

        sample_chain_start = time.perf_counter()
        if self.dynamic_num_leapfrog_steps:
            samples, trace = self._runner(
                state_tensor,
                seed_tensor,
                step_tensor,
                leapfrog_tensor,
            )
        else:
            samples, trace = self._runner(state_tensor, seed_tensor, step_tensor)
        sample_chain_call_s = time.perf_counter() - sample_chain_start
        self._call_count += 1
        if self._call_count == 1:
            self._first_call_s = sample_chain_call_s
            self._warm_call_s = None
        else:
            self._warm_call_s = sample_chain_call_s

        trace_capture_start = time.perf_counter()
        trace_for_diagnostics = trace if self.config.trace_policy == "standard" else {}
        diagnostics = _full_chain_hmc_diagnostics(
            samples,
            trace_for_diagnostics,
            trace_policy=self.config.trace_policy,
        )
        trace_capture_s = time.perf_counter() - trace_capture_start
        metadata = self._metadata(
            sample_chain_call_s=sample_chain_call_s,
            trace_capture_s=trace_capture_s,
            trace=trace if isinstance(trace, Mapping) else None,
            num_leapfrog_steps=leapfrog_tensor,
        )
        return FullChainHMCRunResult(
            samples=samples,
            trace=trace,
            diagnostics=diagnostics,
            metadata=metadata,
        )

    __call__ = run

    def _build_runner(self) -> Callable[[Any, Any, Any], tuple[Any, Mapping[str, Any]]]:
        import tensorflow as tf
        import tensorflow_probability as tfp

        tfm = tfp.mcmc
        config = self.config
        target_log_prob = self._target_log_prob
        trace_fn = self._trace_fn

        def run_chain(
            current_state: Any,
            seed: Any,
            step_size: Any,
            num_leapfrog_steps: Any | None = None,
        ) -> tuple[Any, Mapping[str, Any]]:
            leapfrog_count = (
                config.num_leapfrog_steps
                if num_leapfrog_steps is None
                else num_leapfrog_steps
            )
            kernel = tfm.HamiltonianMonteCarlo(
                target_log_prob_fn=target_log_prob,
                step_size=step_size,
                num_leapfrog_steps=leapfrog_count,
            )
            if config.tuning_policy.uses_dual_averaging:
                kernel = tfm.DualAveragingStepSizeAdaptation(
                    inner_kernel=kernel,
                    num_adaptation_steps=config.tuning_policy.num_adaptation_steps,
                    target_accept_prob=tf.constant(
                        config.tuning_policy.target_accept_prob,
                        dtype=current_state.dtype,
                    ),
                )
            return tfm.sample_chain(
                num_results=config.num_results,
                num_burnin_steps=config.num_burnin_steps,
                current_state=current_state,
                kernel=kernel,
                trace_fn=trace_fn,
                seed=seed,
            )

        if config.chain_execution_mode == "eager":
            return run_chain
        input_signature = [
            tf.TensorSpec(shape=self._state_shape, dtype=self._state_dtype),
            tf.TensorSpec(shape=(2,), dtype=tf.int32),
            tf.TensorSpec(shape=(), dtype=self._state_dtype),
        ]
        if self.dynamic_num_leapfrog_steps:
            input_signature.append(tf.TensorSpec(shape=(), dtype=tf.int32))
        if config.use_xla:
            return tf.function(
                run_chain,
                input_signature=input_signature,
                jit_compile=True,
                reduce_retracing=True,
            )
        return tf.function(
            run_chain,
            input_signature=input_signature,
            reduce_retracing=True,
        )

    def _metadata(
        self,
        *,
        sample_chain_call_s: float,
        trace_capture_s: float,
        trace: Mapping[str, Any] | None,
        num_leapfrog_steps: Any,
    ) -> Mapping[str, Any]:
        import tensorflow as tf

        config = self.config
        capability = self.capability
        dynamic_inputs = ("current_state", "seed", "step_size")
        if self.dynamic_num_leapfrog_steps:
            dynamic_inputs = (*dynamic_inputs, "num_leapfrog_steps")
        if config.chain_execution_mode == "eager":
            timing_scope = "reusable_eager_runner_execute_only"
        elif config.use_xla:
            timing_scope = (
                "reusable_tf_function_xla_first_call_compile_plus_execute_then_warm_execute"
            )
        else:
            timing_scope = (
                "reusable_tf_function_first_call_trace_compile_plus_execute_then_warm_execute"
            )
        return {
            "runtime": "tfp.mcmc.sample_chain",
            "reusable_runner": True,
            "jit_compile": config.use_xla,
            "use_xla": config.use_xla,
            "chain_execution_mode": config.chain_execution_mode,
            "trace_policy": config.trace_policy,
            "target_status_trace_policy": config.target_status_trace_policy,
            "adaptation_policy": config.adaptation_policy,
            "tuning_policy": config.tuning_policy.payload(),
            "adaptation_policy_source": config.tuning_policy.source,
            "trace_unavailability": _trace_unavailability(
                config.trace_policy,
                trace,
                target_status_trace_policy=config.target_status_trace_policy,
            ),
            "value_score_authority": capability.value_score_authority,
            "target_scope": capability.target_scope,
            "requested_target_scope": config.target_scope,
            "program_signature": program_signature(
                {
                    "adapter": stable_adapter_signature(self.adapter),
                    "capability": {
                        "value_score_authority": capability.value_score_authority,
                        "xla_hmc_ready": capability.xla_hmc_ready,
                        "full_chain_xla_diagnostic_ready": (
                            capability.full_chain_xla_diagnostic_ready
                        ),
                        "runtime_backend": capability.runtime_backend,
                        "target_scope": capability.target_scope,
                        "evidence_path": capability.evidence_path,
                        "nonclaims": capability.nonclaims,
                    },
                    "config": config.signature_payload(),
                    "initial_state_shape": self._state_shape,
                    "initial_state_dtype": self._state_dtype.name,
                    "dynamic_inputs": dynamic_inputs,
                    "dynamic_num_leapfrog_steps": self.dynamic_num_leapfrog_steps,
                }
            ),
            "initial_state_shape": self._state_shape,
            "initial_state_dtype": self._state_dtype.name,
            "dynamic_inputs": dynamic_inputs,
            "seed_source": "runtime_tensor_argument",
            "current_state_source": "runtime_tensor_argument",
            "step_size_source": "runtime_tensor_argument",
            "num_leapfrog_steps_source": (
                "runtime_tensor_argument"
                if self.dynamic_num_leapfrog_steps
                else "static_runner_config"
            ),
            "dynamic_num_leapfrog_steps": self.dynamic_num_leapfrog_steps,
            "num_leapfrog_steps": int(tf.get_static_value(num_leapfrog_steps)),
            "sample_chain_call_s": sample_chain_call_s,
            "sample_chain_invocation_count": self._call_count,
            "sample_chain_timing_scope": timing_scope,
            "runner_build_s": self._runner_build_s,
            "first_call_s": self._first_call_s,
            "first_sample_chain_compile_execute_s": self._first_call_s,
            "warm_call_s": self._warm_call_s,
            "warm_sample_chain_execute_s": self._warm_call_s,
            "trace_capture_s": trace_capture_s,
            "trace_capture_timing_scope": (
                "post_sample_chain_public_safe_trace_diagnostics_capture"
            ),
            "warm_sample_shape": None,
            "warm_trace_keys": tuple(sorted(trace.keys())) if trace is not None else tuple(),
            "timing_buckets": {
                "runner_build_s": "explanatory_only_runner_callable_construction",
                "first_call_s": (
                    "explanatory_only_first_sample_chain_compile_plus_execute_when_tf_function"
                ),
                "warm_call_s": "explanatory_only_subsequent_sample_chain_execute",
                "trace_capture_s": (
                    "explanatory_only_post_sample_chain_trace_diagnostics_capture"
                ),
            },
            "nonclaims": (
                "reusable full-chain HMC engineering runner only",
                "no sampler convergence claim",
                "no posterior validity claim",
                "no GPU readiness claim",
                "no performance superiority claim",
            ),
        }


class IndependentChainHMCRunner:
    """Run independent scalar-chain XLA graphs serially or in threads.

    This is the production-chain boundary used by the local ``dsge_hmc``
    implementation: one reviewed TFP runner per chain remains inside the
    GPU-owning process, while Python threads schedule the independent compiled
    calls.  It is intentionally separate from CPU process pools used to
    evaluate independent NeuTra training rows.
    """

    def __init__(
        self,
        adapter: Any,
        initial_state_template: Any,
        config: FullChainHMCConfig,
    ) -> None:
        import tensorflow as tf

        template = tf.cast(tf.convert_to_tensor(initial_state_template), tf.float64)
        if template.shape.rank != 2:
            raise ValueError(
                "independent-chain HMC requires initial_state_template shape "
                "[chain, parameter]"
            )
        if any(dim is None for dim in template.shape):
            raise ValueError("independent-chain HMC requires a fully static state shape")
        self.adapter = adapter
        self.config = config
        self._initial_state_template = template
        self._chain_count = int(template.shape[0])
        self._parameter_dim = int(template.shape[1])
        if self._chain_count < 1:
            raise ValueError("independent-chain HMC requires at least one chain")
        self._runners = tuple(
            ReusableFullChainHMCRunner(adapter, template[index], config)
            for index in range(self._chain_count)
        )
        self._call_count = 0

    @property
    def chain_count(self) -> int:
        return self._chain_count

    @property
    def state_shape(self) -> tuple[int, int]:
        return (self._chain_count, self._parameter_dim)

    def run(
        self,
        *,
        current_state: Any | None = None,
        root_seed: tuple[int, int] | Any | None = None,
        step_size: float | Any | None = None,
        mode: str = "threaded",
    ) -> FullChainHMCRunResult:
        """Run every scalar chain and restore TFP's draw-chain-parameter shape."""

        import tensorflow as tf

        if mode not in {"serial", "threaded"}:
            raise ValueError("independent-chain HMC mode must be 'serial' or 'threaded'")
        state = self._initial_state_template if current_state is None else current_state
        state_tensor = tf.convert_to_tensor(state, dtype=tf.float64)
        if tuple(state_tensor.shape.as_list()) != self.state_shape:
            raise ValueError(
                "current_state shape must match independent-chain runner template"
            )
        seed_value = self.config.seed if root_seed is None else root_seed
        seed_tensor = tf.convert_to_tensor(seed_value, dtype=tf.int32)
        if tuple(seed_tensor.shape.as_list()) != (2,):
            raise ValueError("root_seed must have static shape (2,)")
        step_value = self.config.step_size if step_size is None else step_size
        step_tensor = tf.convert_to_tensor(step_value, dtype=tf.float64)
        if step_tensor.shape.rank != 0:
            raise ValueError("step_size must be a scalar")
        chain_seeds = tuple(
            tf.random.experimental.stateless_fold_in(seed_tensor, index)
            for index in range(self._chain_count)
        )

        def run_one(index: int) -> tuple[Any, Mapping[str, Any], float]:
            chain_started = time.perf_counter()
            samples, trace = self._runners[index]._runner(
                state_tensor[index],
                chain_seeds[index],
                step_tensor,
            )
            return samples, trace, time.perf_counter() - chain_started

        started = time.perf_counter()
        if mode == "threaded" and self._chain_count > 1:
            with ThreadPoolExecutor(max_workers=self._chain_count) as pool:
                chain_outputs = tuple(pool.map(run_one, range(self._chain_count)))
        else:
            chain_outputs = tuple(run_one(index) for index in range(self._chain_count))
        wall_s = time.perf_counter() - started
        self._call_count += 1

        samples = tf.stack(
            [tf.convert_to_tensor(output[0]) for output in chain_outputs],
            axis=1,
        )
        trace = _stack_independent_chain_trace(
            [output[1] for output in chain_outputs]
        )
        diagnostics = _full_chain_hmc_diagnostics(
            samples,
            trace,
            trace_policy=self.config.trace_policy,
        )
        metadata = {
            "runtime": "tfp.mcmc.sample_chain_independent_chain_ensemble",
            "reusable_runner": True,
            "execution_mode": mode,
            "chain_count": self._chain_count,
            "state_shape": self.state_shape,
            "root_seed": tuple(int(value) for value in seed_tensor.numpy()),
            "chain_seeds": tuple(
                tuple(int(value) for value in seed.numpy()) for seed in chain_seeds
            ),
            "ensemble_call_count": self._call_count,
            "ensemble_call_s": wall_s,
            "per_chain_call_s": tuple(
                float(output[2]) for output in chain_outputs
            ),
            "use_xla": self.config.use_xla,
            "trace_policy": self.config.trace_policy,
            "target_scope": self.config.target_scope,
            "source_topology": (
                "dsge_hmc_independent_scalar_chain_tf_function_thread_pool"
            ),
            "nonclaims": (
                "HMC execution-topology mechanics only",
                "no sampler convergence claim",
                "no posterior validity claim",
                "native divergence unavailability is not zero divergences",
            ),
        }
        return FullChainHMCRunResult(
            samples=samples,
            trace=trace,
            diagnostics=diagnostics,
            metadata=metadata,
        )

    __call__ = run


def _stack_independent_chain_trace(
    chain_traces: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Stack identical scalar-chain trace trees along the chain axis."""

    import tensorflow as tf

    if not chain_traces:
        raise ValueError("at least one chain trace is required")
    keys = tuple(chain_traces[0].keys())
    expected = set(keys)
    if any(set(trace) != expected for trace in chain_traces[1:]):
        raise ValueError("independent-chain trace keys must match")

    def stack_values(values: Sequence[Any]) -> Any:
        first = values[0]
        if isinstance(first, Mapping):
            child_keys = tuple(first.keys())
            expected_children = set(child_keys)
            for value in values[1:]:
                if not isinstance(value, Mapping) or set(value) != expected_children:
                    raise ValueError("independent-chain nested trace keys must match")
            return {
                key: stack_values([value[key] for value in values])
                for key in child_keys
            }
        return tf.stack([tf.convert_to_tensor(value) for value in values], axis=1)

    return {key: stack_values([trace[key] for trace in chain_traces]) for key in keys}


def build_independent_chain_tfp_hmc_runner(
    adapter: Any,
    initial_state_template: Any,
    config: FullChainHMCConfig,
) -> IndependentChainHMCRunner:
    """Build the independent scalar-chain serial/threaded HMC topology."""

    return IndependentChainHMCRunner(adapter, initial_state_template, config)


def build_reusable_full_chain_tfp_hmc_runner(
    adapter: Any,
    initial_state_template: Any,
    config: FullChainHMCConfig,
    *,
    dynamic_num_leapfrog_steps: bool = False,
) -> ReusableFullChainHMCRunner:
    """Build a reusable BayesFilter TFP HMC runner for a static contract."""

    return ReusableFullChainHMCRunner(
        adapter,
        initial_state_template,
        config,
        dynamic_num_leapfrog_steps=dynamic_num_leapfrog_steps,
    )


class FixedSizeHMCChunkRunner:
    """Reusable fixed-shape HMC chunk runner using ``tf.while_loop``.

    The compiled contract fixes ``max_results`` and the static state shape, but
    each call supplies a runtime scalar ``active_results``.  The graph executes
    exactly the active number of sample transitions after fixed burn-in,
    writes valid draws into a fixed-size tensor, and returns a valid mask so
    callers can checkpoint chunks without recompiling for each active count.
    """

    def __init__(
        self,
        adapter: Any,
        initial_state_template: Any,
        config: FixedSizeHMCChunkConfig,
    ) -> None:
        self.adapter = adapter
        self.config = config
        self.capability = _validate_full_chain_hmc_authority(adapter, config)

        import tensorflow as tf

        template = tf.cast(tf.convert_to_tensor(initial_state_template), tf.float64)
        if template.shape.rank is None:
            raise ValueError("fixed-size HMC chunk runner requires static state rank")
        if any(dim is None for dim in template.shape):
            raise ValueError(
                "fixed-size HMC chunk runner requires fully static state shape"
            )
        self._state_shape = tuple(int(dim) for dim in template.shape)
        self._state_dtype = template.dtype
        self._initial_state_template = template
        self._target_log_prob = _make_tfp_target_log_prob_fn(
            adapter,
            dtype=self._state_dtype,
        )
        runner_build_start = time.perf_counter()
        self._runner = self._build_runner()
        self._runner_build_s = time.perf_counter() - runner_build_start
        self._call_count = 0
        self._first_call_s: float | None = None
        self._warm_call_s: float | None = None

    @property
    def state_shape(self) -> tuple[int, ...]:
        return self._state_shape

    @property
    def state_dtype(self) -> str:
        return self._state_dtype.name

    def run(
        self,
        *,
        active_results: int | Any,
        current_state: Any | None = None,
        seed: tuple[int, int] | Any | None = None,
        step_size: float | Any | None = None,
    ) -> FixedSizeHMCChunkRunResult:
        """Run one fixed-size chunk with runtime active sample count."""

        import tensorflow as tf

        state = self._initial_state_template if current_state is None else current_state
        state_tensor = tf.convert_to_tensor(state, dtype=self._state_dtype)
        if tuple(state_tensor.shape.as_list()) != self._state_shape:
            raise ValueError("current_state shape must match fixed-size runner template")
        seed_value = self.config.seed if seed is None else seed
        seed_tensor = tf.convert_to_tensor(seed_value, dtype=tf.int32)
        if tuple(seed_tensor.shape.as_list()) != (2,):
            raise ValueError("seed must have static shape (2,)")
        step_value = self.config.step_size if step_size is None else step_size
        step_tensor = tf.convert_to_tensor(step_value, dtype=self._state_dtype)
        if step_tensor.shape.rank != 0:
            raise ValueError("step_size must be a scalar")
        active_tensor = tf.convert_to_tensor(active_results, dtype=tf.int32)
        if active_tensor.shape.rank != 0:
            raise ValueError("active_results must be a scalar")
        if hasattr(active_tensor, "numpy"):
            active_value = int(active_tensor.numpy())
            if active_value < 0 or active_value > self.config.max_results:
                raise ValueError("active_results must satisfy 0 <= active_results <= max_results")

        chunk_start = time.perf_counter()
        samples, valid_mask, final_state, trace = self._runner(
            state_tensor,
            seed_tensor,
            step_tensor,
            active_tensor,
        )
        chunk_call_s = time.perf_counter() - chunk_start
        self._call_count += 1
        if self._call_count == 1:
            self._first_call_s = chunk_call_s
            self._warm_call_s = None
        else:
            self._warm_call_s = chunk_call_s

        trace_capture_start = time.perf_counter()
        diagnostics = _fixed_size_hmc_chunk_diagnostics(
            samples,
            valid_mask,
            trace,
            trace_policy=self.config.trace_policy,
        )
        trace_capture_s = time.perf_counter() - trace_capture_start
        metadata = self._metadata(
            active_results=int(active_tensor.numpy()) if hasattr(active_tensor, "numpy") else None,
            chunk_call_s=chunk_call_s,
            trace_capture_s=trace_capture_s,
            trace=trace,
        )
        return FixedSizeHMCChunkRunResult(
            samples=samples,
            valid_mask=valid_mask,
            final_state=final_state,
            trace=trace,
            diagnostics=diagnostics,
            metadata=metadata,
        )

    __call__ = run

    def _build_runner(self) -> Callable[[Any, Any, Any, Any], tuple[Any, Any, Any, Mapping[str, Any]]]:
        import tensorflow as tf
        import tensorflow_probability as tfp

        tfm = tfp.mcmc
        config = self.config
        target_log_prob = self._target_log_prob
        state_shape = self._state_shape

        def run_chunk(
            current_state: Any,
            seed: Any,
            step_size: Any,
            active_results: Any,
        ) -> tuple[Any, Any, Any, Mapping[str, Any]]:
            active_results = tf.cast(active_results, tf.int32)
            with tf.control_dependencies(
                [
                    tf.debugging.assert_greater_equal(
                        active_results,
                        tf.constant(0, dtype=tf.int32),
                        message="active_results must be nonnegative",
                    ),
                    tf.debugging.assert_less_equal(
                        active_results,
                        tf.constant(config.max_results, dtype=tf.int32),
                        message="active_results must not exceed max_results",
                    ),
                ]
            ):
                active_results = tf.identity(active_results)

            kernel = tfm.HamiltonianMonteCarlo(
                target_log_prob_fn=target_log_prob,
                step_size=step_size,
                num_leapfrog_steps=config.num_leapfrog_steps,
            )
            kernel_results = kernel.bootstrap_results(current_state)
            initial_accepted = tf.convert_to_tensor(kernel_results.is_accepted)
            acceptance_count_by_chain = tf.zeros(
                tf.shape(initial_accepted),
                dtype=tf.int32,
            )
            acceptance_total_by_chain = tf.zeros(
                tf.shape(initial_accepted),
                dtype=tf.int32,
            )
            native_divergence_template = _extract_native_divergence_tensor(
                kernel_results
            )
            native_divergence_available = native_divergence_template is not None
            if native_divergence_available:
                divergence_count_by_chain = tf.zeros(
                    tf.shape(tf.convert_to_tensor(native_divergence_template)),
                    dtype=tf.int32,
                )
            else:
                divergence_count_by_chain = tf.zeros(
                    tf.shape(initial_accepted),
                    dtype=tf.int32,
                )
            collect_standard_trace = config.trace_policy == "standard"

            def burnin_condition(index: Any, _state: Any, _results: Any) -> Any:
                return index < tf.constant(config.num_burnin_steps, dtype=tf.int32)

            def burnin_body(index: Any, state: Any, results: Any) -> tuple[Any, Any, Any]:
                step_seed = tf.random.experimental.stateless_fold_in(seed, index)
                next_state, next_results = kernel.one_step(
                    state,
                    results,
                    seed=step_seed,
                )
                return index + 1, next_state, next_results

            _burnin_index, burnin_state, burnin_results = tf.while_loop(
                burnin_condition,
                burnin_body,
                (tf.constant(0, dtype=tf.int32), current_state, kernel_results),
                parallel_iterations=1,
            )

            samples = tf.zeros((config.max_results,) + state_shape, dtype=current_state.dtype)

            def sample_condition(
                index: Any,
                _state: Any,
                _results: Any,
                _samples: Any,
                _acceptance_count_by_chain: Any,
                _acceptance_total_by_chain: Any,
                _log_accept_finite_count: Any,
                _log_accept_nonfinite_count: Any,
                _log_accept_max_abs_finite: Any,
                _target_log_prob_finite_count: Any,
                _target_log_prob_nonfinite_count: Any,
                _target_log_prob_min_finite: Any,
                _target_log_prob_max_finite: Any,
                _divergence_count_by_chain: Any,
                _is_accepted_buffer: Any,
                _log_accept_buffer: Any,
                _target_log_prob_buffer: Any,
            ) -> Any:
                return index < active_results

            def sample_body(
                index: Any,
                state: Any,
                results: Any,
                sample_buffer: Any,
                accepted_count_by_chain: Any,
                total_count_by_chain: Any,
                log_accept_finite_count: Any,
                log_accept_nonfinite_count: Any,
                log_accept_max_abs_finite: Any,
                target_log_prob_finite_count: Any,
                target_log_prob_nonfinite_count: Any,
                target_log_prob_min_finite: Any,
                target_log_prob_max_finite: Any,
                divergence_by_chain: Any,
                is_accepted_buffer: Any,
                log_accept_buffer: Any,
                target_log_prob_buffer: Any,
            ) -> tuple[
                Any,
                Any,
                Any,
                Any,
                Any,
                Any,
                Any,
                Any,
                Any,
                Any,
                Any,
                Any,
                Any,
                Any,
                Any,
                Any,
                Any,
            ]:
                step_seed = tf.random.experimental.stateless_fold_in(
                    seed,
                    tf.constant(config.num_burnin_steps, dtype=tf.int32) + index,
                )
                next_state, next_results = kernel.one_step(
                    state,
                    results,
                    seed=step_seed,
                )
                updated = tf.tensor_scatter_nd_update(
                    sample_buffer,
                    tf.reshape(index, [1, 1]),
                    tf.expand_dims(next_state, axis=0),
                )
                accepted = tf.cast(next_results.is_accepted, tf.int32)
                accepted_count_by_chain = accepted_count_by_chain + accepted
                total_count_by_chain = total_count_by_chain + tf.ones_like(
                    accepted,
                    dtype=tf.int32,
                )
                log_accept = tf.convert_to_tensor(
                    next_results.log_accept_ratio,
                    dtype=tf.float64,
                )
                log_accept_finite = tf.math.is_finite(log_accept)
                log_accept_finite_count = log_accept_finite_count + tf.reduce_sum(
                    tf.cast(log_accept_finite, tf.int32)
                )
                log_accept_nonfinite_count = (
                    log_accept_nonfinite_count
                    + tf.reduce_sum(
                        tf.cast(tf.logical_not(log_accept_finite), tf.int32)
                    )
                )
                log_accept_max_abs_finite = tf.maximum(
                    log_accept_max_abs_finite,
                    tf.reduce_max(
                        tf.where(
                            log_accept_finite,
                            tf.abs(log_accept),
                            tf.zeros_like(log_accept, dtype=tf.float64),
                        )
                    ),
                )
                target_log_prob = tf.convert_to_tensor(
                    next_results.accepted_results.target_log_prob,
                    dtype=tf.float64,
                )
                target_log_prob_finite = tf.math.is_finite(target_log_prob)
                target_log_prob_finite_count = (
                    target_log_prob_finite_count
                    + tf.reduce_sum(tf.cast(target_log_prob_finite, tf.int32))
                )
                target_log_prob_nonfinite_count = (
                    target_log_prob_nonfinite_count
                    + tf.reduce_sum(
                        tf.cast(tf.logical_not(target_log_prob_finite), tf.int32)
                    )
                )
                target_log_prob_min_finite = tf.minimum(
                    target_log_prob_min_finite,
                    tf.reduce_min(
                        tf.where(
                            target_log_prob_finite,
                            target_log_prob,
                            tf.fill(
                                tf.shape(target_log_prob),
                                tf.constant(float("inf"), dtype=tf.float64),
                            ),
                        )
                    ),
                )
                target_log_prob_max_finite = tf.maximum(
                    target_log_prob_max_finite,
                    tf.reduce_max(
                        tf.where(
                            target_log_prob_finite,
                            target_log_prob,
                            tf.fill(
                                tf.shape(target_log_prob),
                                tf.constant(float("-inf"), dtype=tf.float64),
                            ),
                        )
                    ),
                )
                if native_divergence_available:
                    divergence = tf.cast(
                        _extract_native_divergence_tensor(next_results),
                        tf.int32,
                    )
                    divergence_by_chain = divergence_by_chain + divergence
                if collect_standard_trace:
                    is_accepted_buffer = tf.tensor_scatter_nd_update(
                        is_accepted_buffer,
                        tf.reshape(index, [1, 1]),
                        tf.expand_dims(
                            tf.cast(next_results.is_accepted, tf.bool),
                            axis=0,
                        ),
                    )
                    log_accept_buffer = tf.tensor_scatter_nd_update(
                        log_accept_buffer,
                        tf.reshape(index, [1, 1]),
                        tf.expand_dims(tf.cast(log_accept, tf.float64), axis=0),
                    )
                    target_log_prob_buffer = tf.tensor_scatter_nd_update(
                        target_log_prob_buffer,
                        tf.reshape(index, [1, 1]),
                        tf.expand_dims(
                            tf.cast(target_log_prob, tf.float64),
                            axis=0,
                        ),
                    )
                return (
                    index + 1,
                    next_state,
                    next_results,
                    updated,
                    accepted_count_by_chain,
                    total_count_by_chain,
                    log_accept_finite_count,
                    log_accept_nonfinite_count,
                    log_accept_max_abs_finite,
                    target_log_prob_finite_count,
                    target_log_prob_nonfinite_count,
                    target_log_prob_min_finite,
                    target_log_prob_max_finite,
                    divergence_by_chain,
                    is_accepted_buffer,
                    log_accept_buffer,
                    target_log_prob_buffer,
                )

            trace_shape = tf.concat(
                [
                    tf.constant([config.max_results], dtype=tf.int32),
                    tf.shape(initial_accepted),
                ],
                axis=0,
            )
            trace_acceptance = tf.zeros(trace_shape, dtype=tf.bool)
            trace_log_accept = tf.fill(
                trace_shape,
                tf.constant(float("nan"), dtype=tf.float64),
            )
            trace_target_log_prob = tf.fill(
                trace_shape,
                tf.constant(float("nan"), dtype=tf.float64),
            )
            (
                _sample_index,
                final_state,
                _final_results,
                samples,
                acceptance_count_by_chain,
                acceptance_total_by_chain,
                log_accept_finite_count,
                log_accept_nonfinite_count,
                log_accept_max_abs_finite,
                target_log_prob_finite_count,
                target_log_prob_nonfinite_count,
                target_log_prob_min_finite,
                target_log_prob_max_finite,
                divergence_count_by_chain,
                trace_acceptance,
                trace_log_accept,
                trace_target_log_prob,
            ) = tf.while_loop(
                sample_condition,
                sample_body,
                (
                    tf.constant(0, dtype=tf.int32),
                    burnin_state,
                    burnin_results,
                    samples,
                    acceptance_count_by_chain,
                    acceptance_total_by_chain,
                    tf.constant(0, dtype=tf.int32),
                    tf.constant(0, dtype=tf.int32),
                    tf.constant(0.0, dtype=tf.float64),
                    tf.constant(0, dtype=tf.int32),
                    tf.constant(0, dtype=tf.int32),
                    tf.constant(float("inf"), dtype=tf.float64),
                    tf.constant(float("-inf"), dtype=tf.float64),
                    divergence_count_by_chain,
                    trace_acceptance,
                    trace_log_accept,
                    trace_target_log_prob,
                ),
                parallel_iterations=1,
            )
            valid_mask = tf.range(config.max_results, dtype=tf.int32) < active_results
            trace = {
                "trace_collected": tf.constant(True),
                "accepted_count_by_chain": acceptance_count_by_chain,
                "acceptance_total_by_chain": acceptance_total_by_chain,
                "accepted_decision_count": tf.reduce_sum(acceptance_count_by_chain),
                "acceptance_decision_count": tf.reduce_sum(acceptance_total_by_chain),
                "acceptance_rate": tf.reduce_sum(
                    tf.cast(acceptance_count_by_chain, tf.float64)
                )
                / tf.maximum(
                    tf.reduce_sum(tf.cast(acceptance_total_by_chain, tf.float64)),
                    tf.constant(1.0, dtype=tf.float64),
                ),
                "acceptance_rate_by_chain": tf.cast(
                    acceptance_count_by_chain,
                    tf.float64,
                )
                / tf.maximum(
                    tf.cast(acceptance_total_by_chain, tf.float64),
                    tf.ones_like(
                        tf.cast(acceptance_total_by_chain, tf.float64),
                        dtype=tf.float64,
                    ),
                ),
                "log_accept_ratio_finite_count": log_accept_finite_count,
                "log_accept_ratio_nonfinite_count": log_accept_nonfinite_count,
                "log_accept_ratio_max_abs_finite": log_accept_max_abs_finite,
                "target_log_prob_finite_count": target_log_prob_finite_count,
                "target_log_prob_nonfinite_count": target_log_prob_nonfinite_count,
                "target_log_prob_min_finite": tf.cond(
                    target_log_prob_finite_count > 0,
                    lambda: target_log_prob_min_finite,
                    lambda: tf.constant(float("nan"), dtype=tf.float64),
                ),
                "target_log_prob_max_finite": tf.cond(
                    target_log_prob_finite_count > 0,
                    lambda: target_log_prob_max_finite,
                    lambda: tf.constant(float("nan"), dtype=tf.float64),
                ),
                "native_divergence_available": tf.constant(
                    native_divergence_available
                ),
                "divergence_count_by_chain": divergence_count_by_chain,
                "divergence_count": tf.reduce_sum(divergence_count_by_chain),
            }
            if collect_standard_trace:
                trace.update(
                    {
                        "is_accepted": trace_acceptance,
                        "log_accept_ratio": trace_log_accept,
                        "target_log_prob": trace_target_log_prob,
                    }
                )
            return samples, valid_mask, final_state, trace

        if config.chain_execution_mode == "eager":
            return run_chunk
        input_signature = [
            tf.TensorSpec(shape=self._state_shape, dtype=self._state_dtype),
            tf.TensorSpec(shape=(2,), dtype=tf.int32),
            tf.TensorSpec(shape=(), dtype=self._state_dtype),
            tf.TensorSpec(shape=(), dtype=tf.int32),
        ]
        if config.use_xla:
            return tf.function(
                run_chunk,
                input_signature=input_signature,
                jit_compile=True,
                reduce_retracing=True,
            )
        return tf.function(
            run_chunk,
            input_signature=input_signature,
            reduce_retracing=True,
        )

    def _metadata(
        self,
        *,
        active_results: int | None,
        chunk_call_s: float,
        trace_capture_s: float,
        trace: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        config = self.config
        capability = self.capability
        trace_count_fn = getattr(self._runner, "experimental_get_tracing_count", None)
        compile_trace_count = None if trace_count_fn is None else int(trace_count_fn())
        if config.chain_execution_mode == "eager":
            timing_scope = "fixed_size_hmc_chunk_eager_execute_only"
        elif config.use_xla:
            timing_scope = (
                "fixed_size_hmc_chunk_tf_function_xla_first_call_compile_plus_execute_then_warm_execute"
            )
        else:
            timing_scope = (
                "fixed_size_hmc_chunk_tf_function_first_call_trace_compile_plus_execute_then_warm_execute"
            )
        return {
            "runtime": "tfp.mcmc.HamiltonianMonteCarlo.one_step_tf_while_loop",
            "fixed_size_chunk_runner": True,
            "uses_sample_chain": False,
            "jit_compile": config.use_xla,
            "use_xla": config.use_xla,
            "chain_execution_mode": config.chain_execution_mode,
            "trace_policy": config.trace_policy,
            "target_status_trace_policy": config.target_status_trace_policy,
            "max_results": config.max_results,
            "active_results": active_results,
            "num_burnin_steps": config.num_burnin_steps,
            "step_size": config.step_size,
            "num_leapfrog_steps": config.num_leapfrog_steps,
            "trace_unavailability": _trace_unavailability(
                config.trace_policy,
                trace,
                target_status_trace_policy=config.target_status_trace_policy,
            ),
            "value_score_authority": capability.value_score_authority,
            "target_scope": capability.target_scope,
            "requested_target_scope": config.target_scope,
            "program_signature": program_signature(
                {
                    "adapter": stable_adapter_signature(self.adapter),
                    "capability": {
                        "value_score_authority": capability.value_score_authority,
                        "xla_hmc_ready": capability.xla_hmc_ready,
                        "full_chain_xla_diagnostic_ready": (
                            capability.full_chain_xla_diagnostic_ready
                        ),
                        "runtime_backend": capability.runtime_backend,
                        "target_scope": capability.target_scope,
                        "evidence_path": capability.evidence_path,
                        "nonclaims": capability.nonclaims,
                    },
                    "config": config.signature_payload(),
                    "initial_state_shape": self._state_shape,
                    "initial_state_dtype": self._state_dtype.name,
                    "dynamic_inputs": (
                        "current_state",
                        "seed",
                        "step_size",
                        "active_results",
                    ),
                }
            ),
            "initial_state_shape": self._state_shape,
            "initial_state_dtype": self._state_dtype.name,
            "dynamic_inputs": (
                "current_state",
                "seed",
                "step_size",
                "active_results",
            ),
            "seed_source": "runtime_tensor_argument",
            "current_state_source": "runtime_tensor_argument",
            "step_size_source": "runtime_tensor_argument",
            "active_results_source": "runtime_tensor_argument",
            "compile_trace_count": compile_trace_count,
            "chunk_call_s": chunk_call_s,
            "chunk_invocation_count": self._call_count,
            "chunk_timing_scope": timing_scope,
            "runner_build_s": self._runner_build_s,
            "first_call_s": self._first_call_s,
            "first_chunk_compile_execute_s": self._first_call_s,
            "warm_call_s": self._warm_call_s,
            "warm_chunk_execute_s": self._warm_call_s,
            "trace_capture_s": trace_capture_s,
            "trace_capture_timing_scope": (
                "post_fixed_size_hmc_chunk_public_safe_trace_diagnostics_capture"
            ),
            "warm_sample_shape": None,
            "warm_trace_keys": tuple(sorted(trace.keys())),
            "timing_buckets": {
                "runner_build_s": "explanatory_only_runner_callable_construction",
                "first_call_s": (
                    "explanatory_only_first_fixed_chunk_compile_plus_execute_when_tf_function"
                ),
                "warm_call_s": "explanatory_only_subsequent_fixed_chunk_execute",
                "trace_capture_s": (
                    "explanatory_only_post_fixed_chunk_trace_diagnostics_capture"
                ),
            },
            "nonclaims": (
                "fixed-size HMC chunk engineering runner only",
                "no sampler convergence claim",
                "no posterior validity claim",
                "no GPU readiness claim",
                "no performance superiority claim",
            ),
        }


def build_fixed_size_hmc_chunk_runner(
    adapter: Any,
    initial_state_template: Any,
    config: FixedSizeHMCChunkConfig,
) -> FixedSizeHMCChunkRunner:
    """Build a fixed-size BayesFilter HMC chunk runner for a static state shape."""

    return FixedSizeHMCChunkRunner(adapter, initial_state_template, config)


def write_hmc_sample_archive(
    *,
    archive_dir: str | Path,
    chunks: Sequence[FixedSizeHMCChunkRunResult],
    config: FixedSizeHMCChunkConfig,
    archive_label: str,
    metadata: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> HMCSampleArchiveManifest:
    """Write private fixed-size HMC chunk tensors and a public-safe manifest.

    This convenience wrapper writes a finite sequence of chunks through
    ``HMCStreamingSampleArchiveSink``.  Long-running callers should instantiate
    the sink directly and call ``write_chunk`` inside their HMC loop so raw
    sample tensors can be released before the next chunk.
    """

    if not isinstance(chunks, Sequence) or isinstance(chunks, (str, bytes)):
        raise TypeError("chunks must be a non-empty sequence of chunk results")
    if not chunks:
        raise ValueError("chunks must be non-empty")
    sink = HMCStreamingSampleArchiveSink(
        archive_dir=archive_dir,
        config=config,
        archive_label=archive_label,
        metadata=metadata,
        overwrite=overwrite,
    )
    for index, chunk in enumerate(chunks):
        sink.write_chunk(chunk, chunk_index=index)
    return sink.finalize()


def _write_hmc_archive_chunk(
    *,
    archive_dir: Path,
    archive_label: str,
    chunk: FixedSizeHMCChunkRunResult,
    config: FixedSizeHMCChunkConfig,
    chunk_index: int,
    overwrite: bool,
) -> tuple[dict[str, Any], int, int]:
    import tensorflow as tf

    if not isinstance(chunk, FixedSizeHMCChunkRunResult):
        raise TypeError("chunk must be a FixedSizeHMCChunkRunResult")
    sample_tensor = tf.convert_to_tensor(chunk.samples)
    mask_tensor = tf.convert_to_tensor(chunk.valid_mask, dtype=tf.bool)
    state_tensor = tf.convert_to_tensor(chunk.final_state)
    if sample_tensor.shape.rank is None or mask_tensor.shape.rank != 1:
        raise ValueError("sample and mask tensors must have static rank")
    max_results = int(sample_tensor.shape[0])
    if max_results != int(config.max_results):
        raise ValueError("sample leading dimension must match config.max_results")
    if int(mask_tensor.shape[0]) != int(config.max_results):
        raise ValueError("valid_mask length must match config.max_results")
    if tuple(state_tensor.shape.as_list()) != tuple(sample_tensor.shape.as_list()[1:]):
        raise ValueError("final_state shape must match sample trailing shape")

    valid_count_tensor = tf.reduce_sum(tf.cast(mask_tensor, tf.int32))
    trailing_axes = tf.range(1, tf.rank(sample_tensor), dtype=tf.int32)
    finite_by_row = tf.reduce_all(
        tf.math.is_finite(sample_tensor),
        axis=trailing_axes,
    )
    nonfinite_valid_tensor = tf.reduce_sum(
        tf.cast(
            tf.logical_and(mask_tensor, tf.logical_not(finite_by_row)),
            tf.int32,
        )
    )
    valid_count = int(valid_count_tensor.numpy())
    nonfinite_valid = int(nonfinite_valid_tensor.numpy())

    sample_path = archive_dir / f"{archive_label}_chunk_{chunk_index:04d}_samples.tftensor"
    mask_path = archive_dir / f"{archive_label}_chunk_{chunk_index:04d}_valid_mask.tftensor"
    state_path = archive_dir / f"{archive_label}_chunk_{chunk_index:04d}_final_state.tftensor"
    for path in (sample_path, mask_path, state_path):
        if path.exists() and not overwrite:
            raise FileExistsError(f"archive shard already exists: {path}")
    sample_hash, sample_bytes = _write_serialized_tensor(sample_path, sample_tensor)
    mask_hash, mask_bytes = _write_serialized_tensor(mask_path, mask_tensor)
    state_hash, state_bytes = _write_serialized_tensor(state_path, state_tensor)

    diagnostics = chunk.diagnostics
    metadata_payload = chunk.metadata
    payload = {
        "chunk_index": int(chunk_index),
        "sample_path": str(sample_path),
        "sample_sha256": sample_hash,
        "sample_bytes": sample_bytes,
        "sample_shape": tuple(int(dim) for dim in sample_tensor.shape),
        "sample_dtype": sample_tensor.dtype.name,
        "valid_mask_path": str(mask_path),
        "valid_mask_sha256": mask_hash,
        "valid_mask_bytes": mask_bytes,
        "valid_mask_shape": tuple(int(dim) for dim in mask_tensor.shape),
        "valid_mask_dtype": mask_tensor.dtype.name,
        "final_state_path": str(state_path),
        "final_state_sha256": state_hash,
        "final_state_bytes": state_bytes,
        "final_state_shape": tuple(int(dim) for dim in state_tensor.shape),
        "final_state_dtype": state_tensor.dtype.name,
        "valid_sample_count": valid_count,
        "nonfinite_valid_sample_count": nonfinite_valid,
        "compile_trace_count": metadata_payload.get("compile_trace_count"),
        "chunk_invocation_count": metadata_payload.get("chunk_invocation_count"),
        "chunk_call_s": metadata_payload.get("chunk_call_s"),
        "diagnostic_roles": {
            "valid_sample_count": "archive_accounting",
            "nonfinite_valid_sample_count": "hard_veto",
            "compile_trace_count": "smoke_scale_retrace_diagnostic",
            "chunk_call_s": "explanatory_only_runtime",
        },
        "trace_policy": metadata_payload.get("trace_policy"),
        "native_divergence_status": _tensor_or_plain_to_metadata(
            diagnostics.get("native_divergence_status")
        ),
    }
    return payload, valid_count, nonfinite_valid


def _finalize_hmc_sample_archive_manifest(
    *,
    archive_dir: Path,
    archive_label: str,
    config: FixedSizeHMCChunkConfig,
    chunk_payloads: Sequence[Mapping[str, Any]],
    total_valid: int,
    total_nonfinite: int,
    metadata: Mapping[str, Any],
    overwrite: bool,
) -> HMCSampleArchiveManifest:
    manifest_payload = {
        "artifact_type": "bayesfilter_hmc_sample_archive_manifest",
        "schema_version": 1,
        "archive_label": str(archive_label),
        "archive_dir": str(archive_dir),
        "chunk_count": len(chunk_payloads),
        "total_valid_sample_count": int(total_valid),
        "total_nonfinite_valid_sample_count": int(total_nonfinite),
        "streaming_contract": {
            "incremental_write_before_next_chunk_supported": True,
            "sink_retains_chunk_tensors_after_write": False,
            "finalize_writes_manifest_only": True,
        },
        "config_public_contract": {
            "max_results": int(config.max_results),
            "num_burnin_steps": int(config.num_burnin_steps),
            "use_xla": bool(config.use_xla),
            "chain_execution_mode": str(config.chain_execution_mode),
            "trace_policy": str(config.trace_policy),
            "target_status_trace_policy": str(config.target_status_trace_policy),
            "target_scope": config.target_scope,
        },
        "chunks": tuple(dict(item) for item in chunk_payloads),
        "metadata": _json_safe_metadata(dict(metadata)),
        "privacy_contract": {
            "archive_contains_private_raw_tensors": True,
            "public_manifest_contains_raw_values": False,
            "public_manifest_fields": (
                "paths",
                "hashes",
                "shapes",
                "dtypes",
                "byte_counts",
                "counts",
                "public_diagnostics",
            ),
            "forbidden_public_payloads": (
                "raw_sample_values",
                "raw_state_values",
                "mass_matrices",
                "selected_kernel_payloads",
                "step_sizes",
                "leapfrog_counts",
            ),
        },
        "nonclaims": (
            "retained HMC sample archive mechanics only",
            "no posterior convergence claim",
            "no serious sampler health claim",
            "no long-run retained-memory claim",
            "no scientific validity claim",
        ),
    }
    _assert_hmc_archive_manifest_public_safe(manifest_payload)
    manifest_path = archive_dir / f"{archive_label}_manifest.json"
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(f"archive manifest already exists: {manifest_path}")
    core_text = json.dumps(
        _json_safe_metadata(manifest_payload),
        indent=2,
        sort_keys=True,
    ) + "\n"
    payload_with_hash = dict(manifest_payload)
    payload_with_hash["manifest_path"] = str(manifest_path)
    payload_with_hash["manifest_core_sha256"] = hashlib.sha256(
        core_text.encode("utf-8")
    ).hexdigest()
    manifest_text = json.dumps(
        _json_safe_metadata(payload_with_hash),
        indent=2,
        sort_keys=True,
    ) + "\n"
    manifest_path.write_text(manifest_text, encoding="utf-8")
    manifest_file_hash = hashlib.sha256(manifest_text.encode("utf-8")).hexdigest()
    return HMCSampleArchiveManifest(
        archive_dir=str(archive_dir),
        manifest_path=str(manifest_path),
        manifest_hash=manifest_file_hash,
        payload=_json_safe_metadata(payload_with_hash),
    )


class InternalSegmentHMCRunner:
    """Reusable single-call HMC runner with fixed internal summaries.

    This runner is for long HMC calls where repeated Python calls to compiled
    chunk bodies are undesirable.  It runs burn-in and sample transitions inside
    one ``tf.while_loop`` and records compact per-segment target summaries.  The
    initial contract is summary-only: it does not allocate or return sample
    buffers, so clients must not use it as a retained posterior sample runner.
    """

    def __init__(
        self,
        adapter: Any,
        initial_state_template: Any,
        config: InternalSegmentHMCRunnerConfig,
    ) -> None:
        self.adapter = adapter
        self.config = config
        self.capability = _validate_full_chain_hmc_authority(adapter, config)

        import tensorflow as tf

        template = tf.cast(tf.convert_to_tensor(initial_state_template), tf.float64)
        if template.shape.rank is None:
            raise ValueError("internal-segment HMC runner requires static state rank")
        if any(dim is None for dim in template.shape):
            raise ValueError("internal-segment HMC runner requires fully static state shape")
        self._state_shape = tuple(int(dim) for dim in template.shape)
        self._state_dtype = template.dtype
        self._initial_state_template = template
        self._target_log_prob = _make_tfp_target_log_prob_fn(
            adapter,
            dtype=self._state_dtype,
        )
        runner_build_start = time.perf_counter()
        self._runner = self._build_runner()
        self._runner_build_s = time.perf_counter() - runner_build_start
        self._call_count = 0
        self._first_call_s: float | None = None
        self._warm_call_s: float | None = None

    @property
    def state_shape(self) -> tuple[int, ...]:
        return self._state_shape

    @property
    def state_dtype(self) -> str:
        return self._state_dtype.name

    def run(
        self,
        *,
        current_state: Any | None = None,
        seed: tuple[int, int] | Any | None = None,
        step_size: float | Any | None = None,
    ) -> InternalSegmentHMCRunResult:
        """Run one internally segmented HMC call."""

        import tensorflow as tf

        state = self._initial_state_template if current_state is None else current_state
        state_tensor = tf.convert_to_tensor(state, dtype=self._state_dtype)
        if tuple(state_tensor.shape.as_list()) != self._state_shape:
            raise ValueError("current_state shape must match internal-segment runner template")
        seed_value = self.config.seed if seed is None else seed
        seed_tensor = tf.convert_to_tensor(seed_value, dtype=tf.int32)
        if tuple(seed_tensor.shape.as_list()) != (2,):
            raise ValueError("seed must have static shape (2,)")
        step_value = self.config.step_size if step_size is None else step_size
        step_tensor = tf.convert_to_tensor(step_value, dtype=self._state_dtype)
        if step_tensor.shape.rank != 0:
            raise ValueError("step_size must be a scalar")

        call_start = time.perf_counter()
        (
            final_state,
            final_target_log_prob,
            final_index,
            segment_index,
            segment_indices,
            segment_target_log_prob,
        ) = self._runner(state_tensor, seed_tensor, step_tensor)
        call_s = time.perf_counter() - call_start
        self._call_count += 1
        if self._call_count == 1:
            self._first_call_s = call_s
            self._warm_call_s = None
        else:
            self._warm_call_s = call_s

        diagnostics = _internal_segment_hmc_diagnostics(
            final_state=final_state,
            final_target_log_prob=final_target_log_prob,
            final_index=final_index,
            segment_index=segment_index,
            segment_indices=segment_indices,
            segment_target_log_prob=segment_target_log_prob,
            config=self.config,
        )
        metadata = self._metadata(call_s=call_s)
        return InternalSegmentHMCRunResult(
            final_state=final_state,
            final_target_log_prob=final_target_log_prob,
            final_index=final_index,
            segment_index=segment_index,
            segment_indices=segment_indices,
            segment_target_log_prob=segment_target_log_prob,
            diagnostics=diagnostics,
            metadata=metadata,
        )

    __call__ = run

    def _build_runner(self) -> Callable[[Any, Any, Any], tuple[Any, Any, Any, Any, Any, Any]]:
        import tensorflow as tf
        import tensorflow_probability as tfp

        tfm = tfp.mcmc
        config = self.config
        target_log_prob = self._target_log_prob

        def run_segments(current_state: Any, seed: Any, step_size: Any) -> tuple[Any, Any, Any, Any, Any, Any]:
            kernel = tfm.HamiltonianMonteCarlo(
                target_log_prob_fn=target_log_prob,
                step_size=step_size,
                num_leapfrog_steps=config.num_leapfrog_steps,
            )
            kernel_results = kernel.bootstrap_results(current_state)

            def burnin_condition(index: Any, _state: Any, _results: Any) -> Any:
                return index < tf.constant(config.num_burnin_steps, dtype=tf.int32)

            def burnin_body(index: Any, state: Any, results: Any) -> tuple[Any, Any, Any]:
                step_seed = tf.random.experimental.stateless_fold_in(seed, index)
                next_state, next_results = kernel.one_step(
                    state,
                    results,
                    seed=step_seed,
                )
                return index + 1, next_state, next_results

            _burnin_index, burnin_state, burnin_results = tf.while_loop(
                burnin_condition,
                burnin_body,
                (tf.constant(0, dtype=tf.int32), current_state, kernel_results),
                parallel_iterations=1,
            )

            segment_indices = tf.zeros((config.segment_count,), dtype=tf.int32)
            segment_target_log_prob = tf.zeros(
                (config.segment_count,) + self._state_shape[:-1],
                dtype=current_state.dtype,
            )
            total = tf.constant(config.total_transitions, dtype=tf.int32)
            segment_length = tf.constant(config.segment_length, dtype=tf.int32)

            def sample_condition(
                index: Any,
                _segment_index: Any,
                _state: Any,
                _results: Any,
                _segment_indices: Any,
                _segment_target_log_prob: Any,
            ) -> Any:
                return index < total

            def sample_body(
                index: Any,
                segment_index: Any,
                state: Any,
                results: Any,
                indices: Any,
                target_log_probs: Any,
            ) -> tuple[Any, Any, Any, Any, Any, Any]:
                step_seed = tf.random.experimental.stateless_fold_in(
                    seed,
                    tf.constant(config.num_burnin_steps, dtype=tf.int32) + index,
                )
                next_state, next_results = kernel.one_step(
                    state,
                    results,
                    seed=step_seed,
                )
                next_index = index + 1
                at_segment_end = tf.equal(tf.math.floormod(next_index, segment_length), 0)
                in_bounds = segment_index < tf.constant(config.segment_count, dtype=tf.int32)
                should_record = tf.logical_and(at_segment_end, in_bounds)
                final_target_log_prob = tf.convert_to_tensor(
                    next_results.accepted_results.target_log_prob
                )
                updated_indices = tf.cond(
                    should_record,
                    lambda: tf.tensor_scatter_nd_update(
                        indices,
                        tf.reshape(segment_index, [1, 1]),
                        tf.reshape(next_index, [1]),
                    ),
                    lambda: indices,
                )
                updated_target_log_probs = tf.cond(
                    should_record,
                    lambda: tf.tensor_scatter_nd_update(
                        target_log_probs,
                        tf.reshape(segment_index, [1, 1]),
                        tf.expand_dims(final_target_log_prob, axis=0),
                    ),
                    lambda: target_log_probs,
                )
                updated_segment_index = tf.where(
                    should_record,
                    segment_index + 1,
                    segment_index,
                )
                return (
                    next_index,
                    updated_segment_index,
                    next_state,
                    next_results,
                    updated_indices,
                    updated_target_log_probs,
                )

            (
                final_index,
                final_segment_index,
                final_state,
                final_results,
                final_segment_indices,
                final_segment_target_log_prob,
            ) = tf.while_loop(
                sample_condition,
                sample_body,
                (
                    tf.constant(0, dtype=tf.int32),
                    tf.constant(0, dtype=tf.int32),
                    burnin_state,
                    burnin_results,
                    segment_indices,
                    segment_target_log_prob,
                ),
                parallel_iterations=1,
            )
            final_target_log_prob = tf.convert_to_tensor(
                final_results.accepted_results.target_log_prob
            )
            return (
                final_state,
                final_target_log_prob,
                final_index,
                final_segment_index,
                final_segment_indices,
                final_segment_target_log_prob,
            )

        if config.chain_execution_mode == "eager":
            return run_segments
        input_signature = [
            tf.TensorSpec(shape=self._state_shape, dtype=self._state_dtype),
            tf.TensorSpec(shape=(2,), dtype=tf.int32),
            tf.TensorSpec(shape=(), dtype=self._state_dtype),
        ]
        if config.use_xla:
            return tf.function(
                run_segments,
                input_signature=input_signature,
                jit_compile=True,
                reduce_retracing=True,
            )
        return tf.function(
            run_segments,
            input_signature=input_signature,
            reduce_retracing=True,
        )

    def _metadata(self, *, call_s: float) -> Mapping[str, Any]:
        config = self.config
        capability = self.capability
        trace_count_fn = getattr(self._runner, "experimental_get_tracing_count", None)
        compile_trace_count = None if trace_count_fn is None else int(trace_count_fn())
        if config.chain_execution_mode == "eager":
            timing_scope = "internal_segment_hmc_eager_execute_only"
        elif config.use_xla:
            timing_scope = (
                "internal_segment_hmc_tf_function_xla_first_call_compile_plus_execute"
            )
        else:
            timing_scope = (
                "internal_segment_hmc_tf_function_first_call_trace_compile_plus_execute"
            )
        return {
            "runtime": "tfp.mcmc.HamiltonianMonteCarlo.one_step_internal_segment_tf_while_loop",
            "internal_segment_runner": True,
            "uses_sample_chain": False,
            "returns_samples": False,
            "jit_compile": config.use_xla,
            "use_xla": config.use_xla,
            "chain_execution_mode": config.chain_execution_mode,
            "output_mode": config.output_mode,
            "target_status_trace_policy": "none",
            "value_score_authority": capability.value_score_authority,
            "target_scope": capability.target_scope,
            "requested_target_scope": config.target_scope,
            "program_signature": program_signature(
                {
                    "adapter": stable_adapter_signature(self.adapter),
                    "capability": {
                        "value_score_authority": capability.value_score_authority,
                        "xla_hmc_ready": capability.xla_hmc_ready,
                        "full_chain_xla_diagnostic_ready": (
                            capability.full_chain_xla_diagnostic_ready
                        ),
                        "runtime_backend": capability.runtime_backend,
                        "target_scope": capability.target_scope,
                        "evidence_path": capability.evidence_path,
                        "nonclaims": capability.nonclaims,
                    },
                    "config": config.signature_payload(),
                    "initial_state_shape": self._state_shape,
                    "initial_state_dtype": self._state_dtype.name,
                    "dynamic_inputs": ("current_state", "seed", "step_size"),
                }
            ),
            "initial_state_shape": self._state_shape,
            "initial_state_dtype": self._state_dtype.name,
            "dynamic_inputs": ("current_state", "seed", "step_size"),
            "seed_source": "runtime_tensor_argument",
            "current_state_source": "runtime_tensor_argument",
            "step_size_source": "runtime_tensor_argument",
            "segment_count": config.segment_count,
            "segment_length": config.segment_length,
            "total_transitions": config.total_transitions,
            "num_burnin_steps": config.num_burnin_steps,
            "num_leapfrog_steps": config.num_leapfrog_steps,
            "compile_trace_count": compile_trace_count,
            "call_s": call_s,
            "call_invocation_count": self._call_count,
            "call_timing_scope": timing_scope,
            "runner_build_s": self._runner_build_s,
            "first_call_s": self._first_call_s,
            "warm_call_s": self._warm_call_s,
            "timing_buckets": {
                "runner_build_s": "explanatory_only_runner_callable_construction",
                "first_call_s": (
                    "explanatory_only_first_internal_segment_compile_plus_execute_when_tf_function"
                ),
                "warm_call_s": "explanatory_only_subsequent_internal_segment_execute",
            },
            "nonclaims": (
                "internal-segment HMC engineering runner only",
                "summary-only mode returns no posterior samples",
                "no sampler convergence claim",
                "no posterior validity claim",
                "no GPU readiness claim",
                "no performance superiority claim",
            ),
        }


class RetainedSampleHMCArchiveRunner:
    """Reusable one-call HMC runner that archives retained samples privately.

    The runner owns the only HMC execution call: burn-in and all retained
    transitions occur inside one TensorFlow function.  Returned metadata is
    deliberately public-safe; raw retained samples, sample descriptors, kernel
    parameters, and archive paths remain confined to the private archive.
    """

    def __init__(
        self,
        adapter: Any,
        initial_state_template: Any,
        config: RetainedSampleHMCArchiveConfig,
    ) -> None:
        self.adapter = adapter
        self.config = config
        self.capability = _validate_full_chain_hmc_authority(adapter, config)

        import tensorflow as tf

        template = tf.cast(tf.convert_to_tensor(initial_state_template), tf.float64)
        if template.shape.rank is None:
            raise ValueError(
                "retained-sample HMC archive runner requires static state rank"
            )
        if any(dim is None for dim in template.shape):
            raise ValueError(
                "retained-sample HMC archive runner requires fully static state shape"
            )
        self._state_shape = tuple(int(dim) for dim in template.shape)
        self._state_dtype = template.dtype
        self._initial_state_template = template
        self._target_log_prob = _make_tfp_target_log_prob_fn(
            adapter,
            dtype=self._state_dtype,
        )
        runner_build_start = time.perf_counter()
        self._runner = self._build_runner()
        self._runner_build_s = time.perf_counter() - runner_build_start
        self._call_count = 0
        self._first_call_s: float | None = None
        self._warm_call_s: float | None = None

    @property
    def state_shape(self) -> tuple[int, ...]:
        return self._state_shape

    @property
    def state_dtype(self) -> str:
        return self._state_dtype.name

    def run(
        self,
        *,
        archive_dir: str | Path,
        archive_label: str,
        metadata: Mapping[str, Any] | None = None,
        current_state: Any | None = None,
        seed: tuple[int, int] | Any | None = None,
        step_size: float | Any | None = None,
        overwrite: bool = False,
    ) -> RetainedSampleHMCArchiveRunResult:
        """Execute one retained HMC call and write one private sample shard."""

        import tensorflow as tf

        label = str(archive_label).strip()
        if not label:
            raise ValueError("archive_label must be non-empty")
        state = self._initial_state_template if current_state is None else current_state
        state_tensor = tf.convert_to_tensor(state, dtype=self._state_dtype)
        if tuple(state_tensor.shape.as_list()) != self._state_shape:
            raise ValueError(
                "current_state shape must match retained-sample archive runner template"
            )
        seed_value = self.config.seed if seed is None else seed
        seed_tensor = tf.convert_to_tensor(seed_value, dtype=tf.int32)
        if tuple(seed_tensor.shape.as_list()) != (2,):
            raise ValueError("seed must have static shape (2,)")
        step_value = self.config.step_size if step_size is None else step_size
        step_tensor = tf.convert_to_tensor(step_value, dtype=self._state_dtype)
        if step_tensor.shape.rank != 0:
            raise ValueError("step_size must be a scalar")

        call_start = time.perf_counter()
        (
            samples,
            final_state,
            final_target_log_prob,
            final_index,
            sampler_telemetry,
        ) = self._runner(
            state_tensor,
            seed_tensor,
            step_tensor,
        )
        call_s = time.perf_counter() - call_start
        self._call_count += 1
        if self._call_count == 1:
            self._first_call_s = call_s
            self._warm_call_s = None
        else:
            self._warm_call_s = call_s

        diagnostics = _retained_sample_hmc_archive_diagnostics(
            samples=samples,
            final_state=final_state,
            final_target_log_prob=final_target_log_prob,
            final_index=final_index,
            config=self.config,
            hmc_execution_call_count=self._call_count,
            sampler_telemetry=sampler_telemetry,
        )
        metadata_payload = self._metadata(call_s=call_s)
        archive_summary = _write_retained_sample_hmc_archive_bundle(
            archive_dir=Path(archive_dir),
            archive_label=label,
            samples=samples,
            final_state=final_state,
            final_target_log_prob=final_target_log_prob,
            config=self.config,
            metadata={
                "caller_metadata": {} if metadata is None else dict(metadata),
                "runner_metadata": dict(metadata_payload),
            },
            diagnostics=diagnostics,
            overwrite=bool(overwrite),
        )
        return RetainedSampleHMCArchiveRunResult(
            final_state=final_state,
            final_target_log_prob=final_target_log_prob,
            final_index=final_index,
            diagnostics=diagnostics,
            metadata=metadata_payload,
            archive_summary=archive_summary,
        )

    __call__ = run

    def _build_runner(self) -> Callable[[Any, Any, Any], tuple[Any, Any, Any, Any, Any]]:
        import tensorflow as tf
        import tensorflow_probability as tfp

        tfm = tfp.mcmc
        config = self.config
        target_log_prob = self._target_log_prob
        state_shape = self._state_shape

        def run_retained_archive(
            current_state: Any,
            seed: Any,
            step_size: Any,
        ) -> tuple[Any, Any, Any, Any, Any]:
            kernel = tfm.HamiltonianMonteCarlo(
                target_log_prob_fn=target_log_prob,
                step_size=step_size,
                num_leapfrog_steps=config.num_leapfrog_steps,
            )
            kernel_results = kernel.bootstrap_results(current_state)
            initial_accepted = tf.convert_to_tensor(kernel_results.is_accepted)
            acceptance_count_by_chain = tf.zeros(
                tf.shape(initial_accepted),
                dtype=tf.int32,
            )
            acceptance_total_by_chain = tf.zeros(
                tf.shape(initial_accepted),
                dtype=tf.int32,
            )
            native_divergence_template = _extract_native_divergence_tensor(
                kernel_results
            )
            native_divergence_available = native_divergence_template is not None
            if native_divergence_available:
                divergence_count_by_chain = tf.zeros(
                    tf.shape(tf.convert_to_tensor(native_divergence_template)),
                    dtype=tf.int32,
                )
            else:
                divergence_count_by_chain = tf.zeros(
                    tf.shape(initial_accepted),
                    dtype=tf.int32,
                )

            def burnin_condition(index: Any, _state: Any, _results: Any) -> Any:
                return index < tf.constant(config.num_burnin_steps, dtype=tf.int32)

            def burnin_body(index: Any, state: Any, results: Any) -> tuple[Any, Any, Any]:
                step_seed = tf.random.experimental.stateless_fold_in(seed, index)
                next_state, next_results = kernel.one_step(
                    state,
                    results,
                    seed=step_seed,
                )
                return index + 1, next_state, next_results

            _burnin_index, burnin_state, burnin_results = tf.while_loop(
                burnin_condition,
                burnin_body,
                (tf.constant(0, dtype=tf.int32), current_state, kernel_results),
                parallel_iterations=1,
            )

            samples = tf.zeros(
                (config.num_results,) + state_shape,
                dtype=current_state.dtype,
            )
            total = tf.constant(config.num_results, dtype=tf.int32)

            def sample_condition(
                index: Any,
                _state: Any,
                _results: Any,
                _samples: Any,
                _acceptance_count_by_chain: Any,
                _acceptance_total_by_chain: Any,
                _log_accept_finite_count: Any,
                _log_accept_nonfinite_count: Any,
                _log_accept_max_abs_finite: Any,
                _target_log_prob_finite_count: Any,
                _target_log_prob_nonfinite_count: Any,
                _target_log_prob_min_finite: Any,
                _target_log_prob_max_finite: Any,
                _divergence_count_by_chain: Any,
            ) -> Any:
                return index < total

            def sample_body(
                index: Any,
                state: Any,
                results: Any,
                sample_buffer: Any,
                accepted_count_by_chain: Any,
                total_count_by_chain: Any,
                log_accept_finite_count: Any,
                log_accept_nonfinite_count: Any,
                log_accept_max_abs_finite: Any,
                target_log_prob_finite_count: Any,
                target_log_prob_nonfinite_count: Any,
                target_log_prob_min_finite: Any,
                target_log_prob_max_finite: Any,
                divergence_by_chain: Any,
            ) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any]:
                step_seed = tf.random.experimental.stateless_fold_in(
                    seed,
                    tf.constant(config.num_burnin_steps, dtype=tf.int32) + index,
                )
                next_state, next_results = kernel.one_step(
                    state,
                    results,
                    seed=step_seed,
                )
                updated = tf.tensor_scatter_nd_update(
                    sample_buffer,
                    tf.reshape(index, [1, 1]),
                    tf.expand_dims(next_state, axis=0),
                )
                accepted = tf.cast(next_results.is_accepted, tf.int32)
                accepted_count_by_chain = accepted_count_by_chain + accepted
                total_count_by_chain = total_count_by_chain + tf.ones_like(
                    accepted,
                    dtype=tf.int32,
                )
                log_accept = tf.convert_to_tensor(
                    next_results.log_accept_ratio,
                    dtype=tf.float64,
                )
                log_accept_finite = tf.math.is_finite(log_accept)
                log_accept_finite_count = log_accept_finite_count + tf.reduce_sum(
                    tf.cast(log_accept_finite, tf.int32)
                )
                log_accept_nonfinite_count = (
                    log_accept_nonfinite_count
                    + tf.reduce_sum(
                        tf.cast(tf.logical_not(log_accept_finite), tf.int32)
                    )
                )
                log_accept_max_abs_finite = tf.maximum(
                    log_accept_max_abs_finite,
                    tf.reduce_max(
                        tf.where(
                            log_accept_finite,
                            tf.abs(log_accept),
                            tf.zeros_like(log_accept, dtype=tf.float64),
                        )
                    ),
                )
                target_log_prob = tf.convert_to_tensor(
                    next_results.accepted_results.target_log_prob,
                    dtype=tf.float64,
                )
                target_log_prob_finite = tf.math.is_finite(target_log_prob)
                target_log_prob_finite_count = (
                    target_log_prob_finite_count
                    + tf.reduce_sum(tf.cast(target_log_prob_finite, tf.int32))
                )
                target_log_prob_nonfinite_count = (
                    target_log_prob_nonfinite_count
                    + tf.reduce_sum(
                        tf.cast(tf.logical_not(target_log_prob_finite), tf.int32)
                    )
                )
                target_log_prob_min_finite = tf.minimum(
                    target_log_prob_min_finite,
                    tf.reduce_min(
                        tf.where(
                            target_log_prob_finite,
                            target_log_prob,
                            tf.fill(tf.shape(target_log_prob), tf.constant(float("inf"), dtype=tf.float64)),
                        )
                    ),
                )
                target_log_prob_max_finite = tf.maximum(
                    target_log_prob_max_finite,
                    tf.reduce_max(
                        tf.where(
                            target_log_prob_finite,
                            target_log_prob,
                            tf.fill(tf.shape(target_log_prob), tf.constant(float("-inf"), dtype=tf.float64)),
                        )
                    ),
                )
                if native_divergence_available:
                    divergence = tf.cast(
                        _extract_native_divergence_tensor(next_results),
                        tf.int32,
                    )
                    divergence_by_chain = divergence_by_chain + divergence
                return (
                    index + 1,
                    next_state,
                    next_results,
                    updated,
                    accepted_count_by_chain,
                    total_count_by_chain,
                    log_accept_finite_count,
                    log_accept_nonfinite_count,
                    log_accept_max_abs_finite,
                    target_log_prob_finite_count,
                    target_log_prob_nonfinite_count,
                    target_log_prob_min_finite,
                    target_log_prob_max_finite,
                    divergence_by_chain,
                )

            (
                final_index,
                final_state,
                final_results,
                samples,
                acceptance_count_by_chain,
                acceptance_total_by_chain,
                log_accept_finite_count,
                log_accept_nonfinite_count,
                log_accept_max_abs_finite,
                target_log_prob_finite_count,
                target_log_prob_nonfinite_count,
                target_log_prob_min_finite,
                target_log_prob_max_finite,
                divergence_count_by_chain,
            ) = tf.while_loop(
                sample_condition,
                sample_body,
                (
                    tf.constant(0, dtype=tf.int32),
                    burnin_state,
                    burnin_results,
                    samples,
                    acceptance_count_by_chain,
                    acceptance_total_by_chain,
                    tf.constant(0, dtype=tf.int32),
                    tf.constant(0, dtype=tf.int32),
                    tf.constant(0.0, dtype=tf.float64),
                    tf.constant(0, dtype=tf.int32),
                    tf.constant(0, dtype=tf.int32),
                    tf.constant(float("inf"), dtype=tf.float64),
                    tf.constant(float("-inf"), dtype=tf.float64),
                    divergence_count_by_chain,
                ),
                parallel_iterations=1,
            )
            final_target_log_prob = tf.convert_to_tensor(
                final_results.accepted_results.target_log_prob
            )
            sampler_telemetry = {
                "accepted_count_by_chain": acceptance_count_by_chain,
                "acceptance_total_by_chain": acceptance_total_by_chain,
                "accepted_decision_count": tf.reduce_sum(acceptance_count_by_chain),
                "acceptance_decision_count": tf.reduce_sum(acceptance_total_by_chain),
                "acceptance_rate": tf.reduce_sum(
                    tf.cast(acceptance_count_by_chain, tf.float64)
                )
                / tf.maximum(
                    tf.reduce_sum(tf.cast(acceptance_total_by_chain, tf.float64)),
                    tf.constant(1.0, dtype=tf.float64),
                ),
                "acceptance_rate_by_chain": tf.cast(
                    acceptance_count_by_chain,
                    tf.float64,
                )
                / tf.maximum(
                    tf.cast(acceptance_total_by_chain, tf.float64),
                    tf.ones_like(
                        tf.cast(acceptance_total_by_chain, tf.float64),
                        dtype=tf.float64,
                    ),
                ),
                "log_accept_ratio_finite_count": log_accept_finite_count,
                "log_accept_ratio_nonfinite_count": log_accept_nonfinite_count,
                "log_accept_ratio_max_abs_finite": log_accept_max_abs_finite,
                "target_log_prob_finite_count": target_log_prob_finite_count,
                "target_log_prob_nonfinite_count": target_log_prob_nonfinite_count,
                "target_log_prob_min_finite": tf.cond(
                    target_log_prob_finite_count > 0,
                    lambda: target_log_prob_min_finite,
                    lambda: tf.constant(float("nan"), dtype=tf.float64),
                ),
                "target_log_prob_max_finite": tf.cond(
                    target_log_prob_finite_count > 0,
                    lambda: target_log_prob_max_finite,
                    lambda: tf.constant(float("nan"), dtype=tf.float64),
                ),
                "native_divergence_available": tf.constant(
                    native_divergence_available
                ),
                "divergence_count_by_chain": divergence_count_by_chain,
                "divergence_count": tf.reduce_sum(divergence_count_by_chain),
            }
            return (
                samples,
                final_state,
                final_target_log_prob,
                final_index,
                sampler_telemetry,
            )

        if config.chain_execution_mode == "eager":
            return run_retained_archive
        input_signature = [
            tf.TensorSpec(shape=self._state_shape, dtype=self._state_dtype),
            tf.TensorSpec(shape=(2,), dtype=tf.int32),
            tf.TensorSpec(shape=(), dtype=self._state_dtype),
        ]
        if config.use_xla:
            return tf.function(
                run_retained_archive,
                input_signature=input_signature,
                jit_compile=True,
                reduce_retracing=True,
            )
        return tf.function(
            run_retained_archive,
            input_signature=input_signature,
            reduce_retracing=True,
        )

    def _metadata(self, *, call_s: float) -> Mapping[str, Any]:
        config = self.config
        capability = self.capability
        trace_count_fn = getattr(self._runner, "experimental_get_tracing_count", None)
        compile_trace_count = None if trace_count_fn is None else int(trace_count_fn())
        if config.chain_execution_mode == "eager":
            timing_scope = "retained_sample_hmc_archive_eager_execute_only"
        elif config.use_xla:
            timing_scope = (
                "retained_sample_hmc_archive_tf_function_xla_first_call_compile_plus_execute"
            )
        else:
            timing_scope = (
                "retained_sample_hmc_archive_tf_function_first_call_trace_compile_plus_execute"
            )
        return {
            "runtime": (
                "tfp.mcmc.HamiltonianMonteCarlo.one_step_retained_sample_archive_tf_while_loop"
            ),
            "single_call_retained_archive_runner": True,
            "hmc_execution_call_count": self._call_count,
            "uses_sample_chain": False,
            "returns_samples": False,
            "macrofinance_visible_chunk_count": 0,
            "private_archive_shard_count": 1,
            "private_archive_shard_semantics": (
                "exactly_one_physical_persisted_retained_sample_shard_or_opaque_bundle"
            ),
            "jit_compile": config.use_xla,
            "use_xla": config.use_xla,
            "chain_execution_mode": config.chain_execution_mode,
            "value_score_authority": capability.value_score_authority,
            "target_scope": capability.target_scope,
            "requested_target_scope": config.target_scope,
            "program_signature": program_signature(
                {
                    "adapter": stable_adapter_signature(self.adapter),
                    "capability": {
                        "value_score_authority": capability.value_score_authority,
                        "xla_hmc_ready": capability.xla_hmc_ready,
                        "full_chain_xla_diagnostic_ready": (
                            capability.full_chain_xla_diagnostic_ready
                        ),
                        "runtime_backend": capability.runtime_backend,
                        "target_scope": capability.target_scope,
                        "evidence_path": capability.evidence_path,
                        "nonclaims": capability.nonclaims,
                    },
                    "config_private_signature": program_signature(
                        config.signature_payload()
                    ),
                    "initial_state_shape": self._state_shape,
                    "initial_state_dtype": self._state_dtype.name,
                    "dynamic_inputs": (
                        "current_state",
                        "seed",
                        "kernel_step_scalar",
                    ),
                }
            ),
            "initial_state_shape": self._state_shape,
            "initial_state_dtype": self._state_dtype.name,
            "dynamic_inputs": (
                "current_state",
                "seed",
                "kernel_step_scalar",
            ),
            "seed_source": "runtime_tensor_argument",
            "current_state_source": "runtime_tensor_argument",
            "kernel_step_source": "runtime_tensor_argument",
            "num_results": config.num_results,
            "num_burnin_steps": config.num_burnin_steps,
            "sampler_diagnostics_policy": config.sampler_diagnostics_policy,
            "sampler_health_telemetry": "public_safe_aggregate_counts",
            "native_divergence_telemetry": (
                "native_boolean_field_if_exposed_by_tfp_kernel_results"
            ),
            "kernel_parameters_publicized": False,
            "compile_trace_count": compile_trace_count,
            "call_s": call_s,
            "call_invocation_count": self._call_count,
            "call_timing_scope": timing_scope,
            "runner_build_s": self._runner_build_s,
            "first_call_s": self._first_call_s,
            "warm_call_s": self._warm_call_s,
            "timing_buckets": {
                "runner_build_s": "explanatory_only_runner_callable_construction",
                "first_call_s": (
                    "explanatory_only_first_retained_archive_compile_plus_execute_when_tf_function"
                ),
                "warm_call_s": "explanatory_only_subsequent_retained_archive_execute",
            },
            "nonclaims": (
                "retained-sample HMC archive engineering runner only",
                "raw retained samples are private and not returned",
                "no sampler convergence claim",
                "no posterior validity claim",
                "no GPU readiness claim",
                "no performance superiority claim",
            ),
        }


def build_internal_segment_hmc_runner(
    adapter: Any,
    initial_state_template: Any,
    config: InternalSegmentHMCRunnerConfig,
) -> InternalSegmentHMCRunner:
    """Build a summary-only BayesFilter HMC runner with internal segments."""

    return InternalSegmentHMCRunner(adapter, initial_state_template, config)


def build_retained_sample_hmc_archive_runner(
    adapter: Any,
    initial_state_template: Any,
    config: RetainedSampleHMCArchiveConfig,
) -> RetainedSampleHMCArchiveRunner:
    """Build a one-call HMC runner that writes retained samples privately."""

    return RetainedSampleHMCArchiveRunner(adapter, initial_state_template, config)


class SequentialRHatHMCVerifier:
    """Sequential fixed-kernel verifier with private acceptance evidence."""

    def __init__(
        self,
        adapter: Any,
        initial_state_template: Any,
        config: SequentialRHatHMCVerificationConfig,
    ) -> None:
        self.adapter = adapter
        self.config = config

        import tensorflow as tf

        base = tf.cast(tf.convert_to_tensor(initial_state_template), tf.float64)
        if base.shape.rank not in {1, 2}:
            raise ValueError(
                "sequential verification initial_state_template must be a latent "
                "vector or an explicit chain-by-parameter start bank"
            )
        if any(dim is None for dim in base.shape):
            raise ValueError(
                "sequential R-hat verification requires a fully static state shape"
            )
        if base.shape.rank == 1:
            initial_state = _sequential_rhat_initial_chain_state(base, config.chain_count)
            initial_state_policy = "historical_deterministic_dispersed_latent_starts"
        else:
            if int(base.shape[0]) != int(config.chain_count):
                raise ValueError(
                    "explicit sequential verification start bank must match chain_count"
                )
            initial_state = base
            initial_state_policy = "private_frozen_post_warmup_start_bank"
        initial_chunk_config = FixedSizeHMCChunkConfig(
            max_results=config.check_interval,
            num_burnin_steps=config.num_burnin_steps,
            step_size=config.step_size,
            num_leapfrog_steps=config.num_leapfrog_steps,
            seed=config.seed,
            use_xla=config.use_xla,
            trace_policy="standard",
            target_status_trace_policy="none",
            target_scope=config.target_scope,
            chain_execution_mode=config.chain_execution_mode,
        )
        continuation_chunk_config = FixedSizeHMCChunkConfig(
            max_results=config.check_interval,
            num_burnin_steps=0,
            step_size=config.step_size,
            num_leapfrog_steps=config.num_leapfrog_steps,
            seed=config.seed,
            use_xla=config.use_xla,
            trace_policy="standard",
            target_status_trace_policy="none",
            target_scope=config.target_scope,
            chain_execution_mode=config.chain_execution_mode,
        )
        self._initial_chunk_config = initial_chunk_config
        self._continuation_chunk_config = continuation_chunk_config
        self._initial_runner = FixedSizeHMCChunkRunner(
            adapter,
            initial_state,
            initial_chunk_config,
        )
        self._continuation_runner = (
            self._initial_runner
            if config.num_burnin_steps == 0
            else FixedSizeHMCChunkRunner(
                adapter,
                initial_state,
                continuation_chunk_config,
            )
        )
        self._initial_state = initial_state
        self._initial_state_policy = initial_state_policy
        self._retained_target_health_policy: str | None = None

    def _configure_retained_target_health_policy(
        self,
        target_status_trace_policy: str,
    ) -> None:
        """Enable the private Phase 7 retained value/score/status recheck."""

        policy = str(target_status_trace_policy)
        if policy not in {"none", "per_chain_step"}:
            raise ValueError(
                "target_status_trace_policy must be 'none' or 'per_chain_step'"
            )
        self._retained_target_health_policy = policy

    def run(
        self,
        *,
        checkpoint_writer_config: SequentialRHatCheckpointWriterConfig | None = None,
        checkpoint_reference_callback: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> SequentialRHatHMCVerificationResult:
        """Run sequential verification and return public-safe diagnostics."""

        import tensorflow as tf

        config = self.config
        start = time.perf_counter()
        current_state = self._initial_state
        retained_chunks: list[Any] = []
        log_accept_chunks: list[np.ndarray] = []
        is_accepted_chunks: list[np.ndarray] = []
        target_log_prob_chunks: list[np.ndarray] = []
        chunk_summaries: list[Mapping[str, Any]] = []
        retained_count = 0
        chunk_index = 0
        passed = False
        cap_hit = False
        final_rhat: Mapping[str, Any] = _empty_rhat_summary()
        acceptance_evidence = None
        hard_vetoes: list[str] = []
        engineering_invalidity_reasons: list[str] = []
        candidate_data_invalidity_reasons: list[str] = []
        target_health_evaluated_draw_count = 0
        target_value_health_passed = True
        target_score_health_passed = True
        target_status_failure_count = (
            0 if self._retained_target_health_policy == "per_chain_step" else None
        )
        target_status_count_available = (
            self._retained_target_health_policy == "per_chain_step"
        )
        checkpoint_references: list[Mapping[str, Any]] = []

        while retained_count < config.max_results:
            active = min(config.check_interval, config.max_results - retained_count)
            seed = (
                int(config.seed[0]),
                int(config.seed[1]) + 1009 * (chunk_index + 1),
            )
            runner = self._initial_runner if chunk_index == 0 else self._continuation_runner
            result = runner.run(
                active_results=active,
                current_state=current_state,
                seed=seed,
                step_size=config.step_size,
            )
            current_state = result.final_state
            valid_samples = tf.boolean_mask(
                tf.convert_to_tensor(result.samples),
                tf.convert_to_tensor(result.valid_mask, dtype=tf.bool),
            )
            actual_valid_count = int(
                np.sum(_private_tensor_to_numpy(result.valid_mask).astype(bool))
            )
            retained_chunks.append(valid_samples)
            retained_count += actual_valid_count
            chunk_summary = _sequential_rhat_chunk_summary(
                result,
                chunk_index=chunk_index,
                retained_count=retained_count,
                expected_active_results=active,
            )
            chunk_summaries.append(chunk_summary)
            hard_vetoes.extend(chunk_summary.get("hard_vetoes", ()))
            engineering_invalidity_reasons.extend(
                chunk_summary.get("engineering_invalidity_reasons", ())
            )
            if self._retained_target_health_policy is not None:
                target_health = _evaluate_retained_target_health(
                    adapter=self.adapter,
                    samples=_private_tensor_to_numpy(valid_samples),
                    target_status_trace_policy=self._retained_target_health_policy,
                )
                engineering_invalidity_reasons.extend(
                    target_health["shared_invalidity_reasons"]
                )
                candidate_data_invalidity_reasons.extend(
                    target_health["candidate_data_invalidity_reasons"]
                )
                target_health_evaluated_draw_count += int(
                    target_health["evaluated_draw_count"]
                )
                target_value_health_passed = bool(
                    target_value_health_passed
                    and target_health["target_value_finite"]
                )
                target_score_health_passed = bool(
                    target_score_health_passed
                    and target_health["target_score_finite"]
                )
                if self._retained_target_health_policy == "per_chain_step":
                    chunk_status_count = target_health[
                        "target_status_failure_count"
                    ]
                    if chunk_status_count is None:
                        target_status_count_available = False
                        target_status_failure_count = None
                    elif target_status_count_available:
                        target_status_failure_count = int(
                            int(target_status_failure_count or 0)
                            + int(chunk_status_count)
                        )
            retained = tf.concat(retained_chunks, axis=0)
            final_rhat = _rhat_summary_from_retained_samples(
                retained,
                threshold=config.rhat_threshold,
            )
            trace = result.trace if isinstance(result.trace, Mapping) else {}
            required_trace_keys = ("is_accepted", "log_accept_ratio", "target_log_prob")
            missing_trace_keys = tuple(
                key for key in required_trace_keys if key not in trace
            )
            if missing_trace_keys:
                shared_invalidity_reasons = (
                    "required_standard_acceptance_trace_missing",
                )
                active_log_accept = np.zeros(
                    (actual_valid_count, int(config.chain_count)), dtype=float
                )
                active_is_accepted = np.zeros_like(active_log_accept, dtype=bool)
                active_target_log_prob = np.zeros_like(active_log_accept, dtype=float)
            else:
                shared_invalidity_reasons = ()
                mask = _private_tensor_to_numpy(result.valid_mask).astype(bool)
                active_log_accept = _private_tensor_to_numpy(
                    result.trace["log_accept_ratio"]
                )[mask]
                active_is_accepted = np.asarray(
                    result.trace["is_accepted"].numpy()
                    if hasattr(result.trace["is_accepted"], "numpy")
                    else result.trace["is_accepted"],
                    dtype=bool,
                )[mask]
                active_target_log_prob = _private_tensor_to_numpy(
                    result.trace["target_log_prob"]
                )[mask]
            log_accept_chunks.append(active_log_accept)
            is_accepted_chunks.append(active_is_accepted)
            target_log_prob_chunks.append(active_target_log_prob)
            divergence_status, divergence_count, divergence_consistent = (
                _aggregate_divergence_provenance(chunk_summaries)
            )
            if not divergence_consistent:
                engineering_invalidity_reasons.append("shared_schema_invalid")
            acceptance_evidence = evaluate_hmc_acceptance_evidence(
                samples=_private_tensor_to_numpy(retained),
                log_accept_ratio=np.concatenate(log_accept_chunks, axis=0),
                is_accepted=np.concatenate(is_accepted_chunks, axis=0),
                target_log_prob=np.concatenate(target_log_prob_chunks, axis=0),
                policy=config.acceptance_policy,
                native_divergence_status=divergence_status,
                native_divergence_count=divergence_count,
                shared_invalidity_reasons=tuple(
                    dict.fromkeys(
                        (*shared_invalidity_reasons, *engineering_invalidity_reasons)
                    )
                ),
                candidate_local_health_failures=tuple(
                    dict.fromkeys(candidate_data_invalidity_reasons)
                ),
            )
            if checkpoint_writer_config is not None:
                checkpoint_reference = _write_sequential_rhat_private_checkpoint(
                    writer_config=checkpoint_writer_config,
                    checkpoint_kind="verification_chunk",
                    adapter=self.adapter,
                    config=config,
                    chunk_index=chunk_index,
                    retained_count=retained_count,
                    chunk_result=result,
                    chunk_summary=chunk_summary,
                    rhat_summary=final_rhat,
                )
                checkpoint_references.append(checkpoint_reference)
                if checkpoint_reference_callback is not None:
                    checkpoint_reference_callback(checkpoint_reference)
            decision_reached = (
                acceptance_evidence.evidence_validity != "valid"
                or acceptance_evidence.acceptance_decision
                != "inconclusive_evidence"
            )
            minimum_retained_satisfied = retained_count >= int(
                config.min_retained_results_for_pass
            )
            if decision_reached and not acceptance_evidence.promotion_eligible:
                break
            if (
                acceptance_evidence.promotion_eligible
                and minimum_retained_satisfied
                and bool(final_rhat["passed"])
            ):
                passed = True
                break
            chunk_index += 1

        if (
            not passed
            and retained_count >= config.max_results
            and acceptance_evidence is not None
            and acceptance_evidence.evidence_validity == "valid"
            and acceptance_evidence.acceptance_decision
            in {"passed", "inconclusive_evidence"}
        ):
            cap_hit = True
        runtime_s = time.perf_counter() - start
        diagnostics = {
            "sequential_rhat_verification": True,
            "passed": bool(passed),
            "cap_hit": bool(cap_hit),
            "retained_sample_count": int(retained_count),
            "check_interval": int(config.check_interval),
            "max_results": int(config.max_results),
            "min_retained_results_for_pass": int(
                config.min_retained_results_for_pass
            ),
            "chunk_count": len(chunk_summaries),
            "rhat_threshold": float(config.rhat_threshold),
            "rhat_role": "fixed_kernel_convergence_gate_not_candidate_ranking",
            "rhat_definition": final_rhat["rhat_definition"],
            "max_finite_rhat": final_rhat["max_finite_rhat"],
            "max_rank_normalized_split_rhat": final_rhat[
                "max_rank_normalized_split_rhat"
            ],
            "max_folded_rank_normalized_split_rhat": final_rhat[
                "max_folded_rank_normalized_split_rhat"
            ],
            "finite_rhat_count": int(final_rhat["finite_rhat_count"]),
            "nonfinite_rhat_count": int(final_rhat["nonfinite_rhat_count"]),
            "all_finite_rhat_at_or_below_threshold": bool(final_rhat["passed"]),
            "minimum_retained_pass_gate_satisfied": bool(
                retained_count >= int(config.min_retained_results_for_pass)
            ),
            "samples_all_finite": not any(
                "nonfinite_retained_samples" in item.get("hard_vetoes", ())
                for item in chunk_summaries
            ),
            "target_value_health_passed": not any(
                "nonfinite_private_target_value" in item.get("hard_vetoes", ())
                for item in chunk_summaries
            ) and (
                True
                if self._retained_target_health_policy is None
                else bool(target_value_health_passed)
            ),
            "target_score_health_passed": (
                None
                if self._retained_target_health_policy is None
                else bool(target_score_health_passed)
            ),
            "retained_target_health_policy": (
                "disabled"
                if self._retained_target_health_policy is None
                else self._retained_target_health_policy
            ),
            "retained_target_health_evaluated_draw_count": int(
                target_health_evaluated_draw_count
            ),
            "target_status_failure_count": target_status_failure_count,
            "acceptance_log_health_passed": not any(
                "nonfinite_private_acceptance_log_value" in item.get("hard_vetoes", ())
                for item in chunk_summaries
            ),
            "runtime_s": float(runtime_s),
            "runtime_finite": bool(np.isfinite(runtime_s)),
            "acceptance_rate": (
                None if acceptance_evidence is None else acceptance_evidence.pooled_mean
            ),
            "acceptance_rate_semantics": "mean_metropolis_acceptance_probability",
            "binary_acceptance_rate": _aggregate_acceptance_rate(chunk_summaries),
            "binary_acceptance_role": "explanatory_runtime_provenance_only",
            "acceptance_evidence": (
                None if acceptance_evidence is None else acceptance_evidence.payload()
            ),
            "acceptance_evidence_validity": (
                None
                if acceptance_evidence is None
                else acceptance_evidence.evidence_validity
            ),
            "acceptance_decision": (
                None
                if acceptance_evidence is None
                else acceptance_evidence.acceptance_decision
            ),
            "candidate_promotion_vetoes": (
                ()
                if acceptance_evidence is None
                else acceptance_evidence.candidate_promotion_vetoes
            ),
            "tuning_repair_triggers": (
                ()
                if acceptance_evidence is None
                else acceptance_evidence.tuning_repair_triggers
            ),
            "engineering_invalidity_reasons": (
                ()
                if acceptance_evidence is None
                else acceptance_evidence.engineering_invalidity_reasons
            ),
            "divergence_status": _aggregate_divergence_provenance(chunk_summaries)[0],
            "divergence_count": _aggregate_divergence_provenance(chunk_summaries)[1],
            "divergence_provenance_consistent": _aggregate_divergence_provenance(
                chunk_summaries
            )[2],
            "hard_vetoes": tuple(dict.fromkeys(hard_vetoes)),
            "chunk_summaries": tuple(chunk_summaries),
            "checkpointing_enabled": checkpoint_writer_config is not None,
            "checkpoint_count": len(checkpoint_references),
            "checkpoint_references": tuple(checkpoint_references),
            "privacy_contract": {
                "public_summary_contains_raw_values": False,
                "public_summary_contains_chain_states": False,
                "public_summary_contains_step_size": False,
                "public_summary_contains_leapfrog_count": False,
                "public_summary_contains_mass_matrix": False,
                "public_summary_contains_checkpoint_paths": False,
                "public_summary_contains_checkpoint_tensor_descriptors": False,
                "public_summary_contains_checkpoint_hmc_mechanics": False,
            },
            "nonclaims": (
                "sequential fixed-checkpoint HMC tuning verification only",
                "historical unsplit R-hat is explanatory only",
                "no posterior convergence claim",
                "no scientific validity claim",
                "no sampler superiority claim",
                "no default-readiness claim",
            ),
        }
        metadata = {
            "runtime": "sequential_fixed_size_hmc_chunk_acceptance_verification",
            "uses_sample_chain": False,
            "fixed_size_chunk_runner": True,
            "sequential_rhat_verification": True,
            "acceptance_evidence_decision": (
                None
                if acceptance_evidence is None
                else acceptance_evidence.acceptance_decision
            ),
            "acceptance_evidence_validity": (
                None
                if acceptance_evidence is None
                else acceptance_evidence.evidence_validity
            ),
            "jit_compile": config.use_xla,
            "use_xla": config.use_xla,
            "chain_execution_mode": config.chain_execution_mode,
            "check_interval": int(config.check_interval),
            "max_results": int(config.max_results),
            "min_retained_results_for_pass": int(
                config.min_retained_results_for_pass
            ),
            "initial_burnin_steps": int(config.num_burnin_steps),
            "continuation_burnin_steps": 0,
            "retained_sample_count": int(retained_count),
            "chunk_count": len(chunk_summaries),
            "chain_count": int(config.chain_count),
            "rhat_threshold": float(config.rhat_threshold),
            "rhat_definition": final_rhat["rhat_definition"],
            "max_rank_normalized_split_rhat": final_rhat[
                "max_rank_normalized_split_rhat"
            ],
            "max_folded_rank_normalized_split_rhat": final_rhat[
                "max_folded_rank_normalized_split_rhat"
            ],
            "checkpointing_enabled": checkpoint_writer_config is not None,
            "checkpoint_count": len(checkpoint_references),
            "checkpoint_references": tuple(checkpoint_references),
            "compile_trace_count": _max_compile_trace_count(chunk_summaries),
            "first_call_s": _first_non_none(
                item.get("first_call_s") for item in chunk_summaries
            ),
            "warm_call_s": _last_non_none(
                item.get("warm_call_s") for item in chunk_summaries
            ),
            "runtime_s": float(runtime_s),
            "initial_state_policy": self._initial_state_policy,
            "kernel_parameters_publicized": False,
            "sample_values_publicized": False,
            "chain_handoff_publicized": False,
            "nonclaims": diagnostics["nonclaims"],
        }
        _assert_sequential_rhat_public_safe(diagnostics)
        _assert_sequential_rhat_public_safe(metadata)
        return SequentialRHatHMCVerificationResult(
            passed=bool(passed),
            cap_hit=bool(cap_hit),
            retained_sample_count=int(retained_count),
            chunk_count=len(chunk_summaries),
            max_finite_rhat=final_rhat["max_finite_rhat"],
            finite_rhat_count=int(final_rhat["finite_rhat_count"]),
            nonfinite_rhat_count=int(final_rhat["nonfinite_rhat_count"]),
            diagnostics=_json_safe_metadata(diagnostics),
            metadata=_json_safe_metadata(metadata),
        )

    __call__ = run


def build_sequential_rhat_hmc_verifier(
    adapter: Any,
    initial_state_template: Any,
    config: SequentialRHatHMCVerificationConfig,
) -> SequentialRHatHMCVerifier:
    """Build a sequential R-hat fixed-kernel HMC verifier."""

    return SequentialRHatHMCVerifier(adapter, initial_state_template, config)


def _make_hmc_target_log_prob_fn(
    adapter: Any,
    *,
    use_xla: bool = False,
    target_scope: str | None = None,
) -> Callable[[Any], Any]:
    """Build a target value function, refusing XLA without reviewed authority."""

    capability = value_score_capability(adapter)
    if use_xla and not capability.is_accepted_xla_hmc_authority:
        raise ValueError(
            "XLA HMC requires graph-native or reviewed value/score authority; "
            f"got {capability.value_score_authority!r}"
        )
    if use_xla and capability.target_scope is not None and str(target_scope) != capability.target_scope:
        raise ValueError("value/score target_scope mismatch")

    def target_log_prob(theta: Any) -> Any:
        if hasattr(adapter, "log_prob"):
            return adapter.log_prob(theta)
        if hasattr(adapter, "log_prob_and_grad"):
            value, _grad = adapter.log_prob_and_grad(theta)
            return value
        raise TypeError("adapter must expose log_prob or log_prob_and_grad")

    return target_log_prob


def _validate_full_chain_hmc_authority(
    adapter: Any,
    config: FullChainHMCConfig,
) -> Any:
    """Return adapter value/score authority after full-chain HMC checks."""

    capability = value_score_capability(adapter)
    if config.use_xla and not capability.is_accepted_full_chain_xla_diagnostic_authority:
        raise ValueError(
            "XLA full-chain HMC requires explicit full-chain-XLA diagnostic "
            "authority; target-only XLA readiness is not sufficient; got "
            f"{capability.value_score_authority!r}"
        )
    if config.use_xla and capability.target_scope is not None:
        if config.target_scope != capability.target_scope:
            raise ValueError("value/score target_scope mismatch")
    return capability


def run_full_chain_tfp_hmc(
    adapter: Any,
    initial_state: Any,
    config: FullChainHMCConfig,
) -> FullChainHMCRunResult:
    """Run exact ``tfp.mcmc.sample_chain`` for a tiny engineering HMC gate.

    This wrapper deliberately materializes no tensors to NumPy inside the
    compiled sampling path.  It returns TensorFlow tensors and metadata only;
    callers may convert tensors after this function returns.
    """

    capability = _validate_full_chain_hmc_authority(adapter, config)

    import tensorflow as tf
    import tensorflow_probability as tfp

    tfm = tfp.mcmc
    state = tf.cast(tf.convert_to_tensor(initial_state), tf.float64)
    _ = _make_hmc_target_log_prob_fn(
        adapter,
        use_xla=config.use_xla,
        target_scope=config.target_scope,
    )
    target_log_prob = _make_tfp_target_log_prob_fn(
        adapter,
        dtype=state.dtype,
    )
    kernel = tfm.HamiltonianMonteCarlo(
        target_log_prob_fn=target_log_prob,
        step_size=tf.constant(config.step_size, dtype=state.dtype),
        num_leapfrog_steps=config.num_leapfrog_steps,
    )
    if config.tuning_policy.uses_dual_averaging:
        kernel = tfm.DualAveragingStepSizeAdaptation(
            inner_kernel=kernel,
            num_adaptation_steps=config.tuning_policy.num_adaptation_steps,
            target_accept_prob=tf.constant(
                config.tuning_policy.target_accept_prob,
                dtype=state.dtype,
            ),
    )
    trace_fn = _trace_fn_for_config(config, adapter=adapter)
    runner_build_start = time.perf_counter()
    runner = _build_sample_chain_runner(config, kernel, trace_fn, state)
    runner_build_s = time.perf_counter() - runner_build_start
    sample_chain_start = time.perf_counter()
    samples, trace = runner()
    sample_chain_call_s = time.perf_counter() - sample_chain_start
    trace_capture_start = time.perf_counter()
    trace_for_diagnostics = trace if config.trace_policy == "standard" else {}
    diagnostics = _full_chain_hmc_diagnostics(
        samples,
        trace_for_diagnostics,
        trace_policy=config.trace_policy,
    )
    trace_capture_s = time.perf_counter() - trace_capture_start
    metadata = {
        "runtime": "tfp.mcmc.sample_chain",
        "jit_compile": config.use_xla,
        "use_xla": config.use_xla,
        "chain_execution_mode": config.chain_execution_mode,
        "trace_policy": config.trace_policy,
        "target_status_trace_policy": config.target_status_trace_policy,
        "adaptation_policy": config.adaptation_policy,
        "tuning_policy": config.tuning_policy.payload(),
        "adaptation_policy_source": config.tuning_policy.source,
        "trace_unavailability": _trace_unavailability(
            config.trace_policy,
            trace if isinstance(trace, Mapping) else None,
            target_status_trace_policy=config.target_status_trace_policy,
        ),
        "value_score_authority": capability.value_score_authority,
        "target_scope": capability.target_scope,
        "requested_target_scope": config.target_scope,
        "program_signature": program_signature(
            {
                "adapter": stable_adapter_signature(adapter),
                "capability": {
                    "value_score_authority": capability.value_score_authority,
                    "xla_hmc_ready": capability.xla_hmc_ready,
                    "full_chain_xla_diagnostic_ready": (
                        capability.full_chain_xla_diagnostic_ready
                    ),
                    "runtime_backend": capability.runtime_backend,
                    "target_scope": capability.target_scope,
                    "evidence_path": capability.evidence_path,
                    "nonclaims": capability.nonclaims,
                },
                "config": config.signature_payload(),
                "initial_state_shape": tuple(int(dim) for dim in state.shape),
                "initial_state_dtype": state.dtype.name,
            }
        ),
        "sample_chain_call_s": sample_chain_call_s,
        "sample_chain_invocation_count": 1,
        "sample_chain_timing_scope": "compile_plus_execute_when_xla_first_call",
        "initial_state_shape": tuple(int(dim) for dim in state.shape),
        "initial_state_dtype": state.dtype.name,
        "runner_build_s": runner_build_s,
        "first_call_s": sample_chain_call_s,
        "first_sample_chain_compile_execute_s": sample_chain_call_s,
        "warm_call_s": None,
        "warm_sample_chain_execute_s": None,
        "trace_capture_s": trace_capture_s,
        "trace_capture_timing_scope": (
            "post_sample_chain_public_safe_trace_diagnostics_capture"
        ),
        "warm_sample_shape": None,
        "warm_trace_keys": tuple(),
        "timing_buckets": {
            "runner_build_s": "explanatory_only_runner_callable_construction",
            "first_call_s": "explanatory_only_first_sample_chain_call",
            "warm_call_s": "not_available_single_use_runner",
            "trace_capture_s": (
                "explanatory_only_post_sample_chain_trace_diagnostics_capture"
            ),
        },
        "nonclaims": (
            "tiny full-chain HMC engineering gate only",
            "no sampler convergence claim",
            "no posterior validity claim",
            "no GPU readiness claim",
            "no performance superiority claim",
        ),
    }
    return FullChainHMCRunResult(
        samples=samples,
        trace=trace,
        diagnostics=diagnostics,
        metadata=metadata,
    )


def stable_adapter_signature(adapter: Any) -> str:
    """Return stable adapter metadata without process-local object identity."""

    explicit = getattr(adapter, "adapter_signature", None)
    if explicit is not None:
        signature = explicit() if callable(explicit) else explicit
        return _validate_persisted_signature(str(signature))
    payload: dict[str, Any] = {
        "class": adapter.__class__.__qualname__,
        "module": adapter.__class__.__module__,
    }
    if hasattr(adapter, "parameter_dim"):
        payload["parameter_dim"] = int(getattr(adapter, "parameter_dim"))
    names = getattr(adapter, "parameter_names", None)
    if names is not None:
        values = names() if callable(names) else names
        payload["parameter_names"] = tuple(str(name) for name in values)
    return program_signature(payload)


def _validate_persisted_signature(signature: str) -> str:
    if not signature:
        raise ValueError("adapter_signature must be non-empty")
    if any(pattern.search(signature) for pattern in _PROCESS_LOCAL_SIGNATURE_PATTERNS):
        raise ValueError(
            "adapter_signature must not contain process-local object identity"
        )
    return signature


def _json_safe_metadata(value: Any) -> Any:
    if hasattr(value, "numpy"):
        return _json_safe_metadata(value.numpy())
    if isinstance(value, Mapping):
        return {str(key): _json_safe_metadata(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe_metadata(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe_metadata(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_serialized_tensor(path: Path, tensor: Any) -> tuple[str, int]:
    """Write one TensorFlow tensor shard and return SHA256 plus byte count."""

    import tensorflow as tf

    serialized = tf.io.serialize_tensor(tf.convert_to_tensor(tensor))
    data = bytes(serialized.numpy())
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest(), len(data)


def _tensor_or_plain_to_metadata(value: Any) -> Any:
    """Convert scalar TensorFlow/NumPy values to JSON-safe public metadata."""

    if value is None:
        return None
    if hasattr(value, "numpy"):
        value = value.numpy()
    return _json_safe_metadata(value)


def _sequential_rhat_initial_chain_state(base: Any, chain_count: int) -> Any:
    """Build private deterministic dispersed latent starts for R-hat checks."""

    import tensorflow as tf

    base_tensor = tf.convert_to_tensor(base, dtype=tf.float64)
    dim = int(base_tensor.shape[-1])
    offsets = tf.linspace(
        tf.constant(-0.15, dtype=tf.float64),
        tf.constant(0.15, dtype=tf.float64),
        int(chain_count),
    )
    parity = tf.cast(
        tf.math.floormod(tf.range(dim, dtype=tf.int32), 2),
        dtype=tf.float64,
    )
    pattern = tf.constant(1.0, dtype=tf.float64) - tf.constant(2.0, dtype=tf.float64) * parity
    return tf.expand_dims(base_tensor, axis=0) + tf.expand_dims(offsets, axis=-1) * pattern


def _empty_rhat_summary() -> Mapping[str, Any]:
    return {
        "passed": False,
        "rhat_definition": (
            "max(rank-normalized split R-hat, "
            "folded rank-normalized split R-hat)"
        ),
        "max_rank_normalized_split_rhat": None,
        "max_folded_rank_normalized_split_rhat": None,
        "max_finite_rhat": None,
        "finite_rhat_count": 0,
        "nonfinite_rhat_count": 0,
    }


def _rhat_summary_from_retained_samples(
    samples: Any,
    *,
    threshold: float,
) -> Mapping[str, Any]:
    """Compute the shared modern multi-chain R-hat summary."""

    array = _private_tensor_to_numpy(samples)
    if array.ndim < 3:
        raise ValueError("R-hat samples must have shape (draw, chain, parameter...)")
    draw_count = int(array.shape[0])
    chain_count = int(array.shape[1])
    if draw_count < 4:
        return _empty_rhat_summary()
    if chain_count < 2:
        raise ValueError("R-hat requires at least two chains")
    flat = np.reshape(array, (draw_count, chain_count, -1))
    from bayesfilter.inference.hmc_convergence import (
        rank_normalized_split_rhat_summary,
    )

    summary = rank_normalized_split_rhat_summary(flat, rhat_max=threshold)
    return {
        "passed": bool(summary["passed"]),
        "rhat_definition": summary["rhat_definition"],
        "max_rank_normalized_split_rhat": summary[
            "max_rank_normalized_split_rhat"
        ],
        "max_folded_rank_normalized_split_rhat": summary[
            "max_folded_rank_normalized_split_rhat"
        ],
        "max_finite_rhat": summary["max_finite_rhat"],
        "finite_rhat_count": int(summary["finite_rhat_count"]),
        "nonfinite_rhat_count": int(summary["nonfinite_rhat_count"]),
    }


def _sequential_rhat_chunk_summary(
    result: FixedSizeHMCChunkRunResult,
    *,
    chunk_index: int,
    retained_count: int,
    expected_active_results: int,
) -> Mapping[str, Any]:
    diagnostics = result.diagnostics
    metadata = result.metadata
    hard_vetoes: list[str] = []
    engineering_invalidity_reasons: list[str] = []
    valid_sample_count = _int_or_none_metadata(
        diagnostics.get("valid_sample_count")
    )
    observed_valid_count = int(
        np.sum(_private_tensor_to_numpy(result.valid_mask).astype(bool))
    )
    if (
        valid_sample_count != int(expected_active_results)
        or observed_valid_count != int(expected_active_results)
        or valid_sample_count != observed_valid_count
    ):
        hard_vetoes.append("chunk_retained_count_mismatch")
        engineering_invalidity_reasons.append("shared_schema_invalid")
    nonfinite = _int_or_none_metadata(diagnostics.get("nonfinite_valid_sample_count"))
    if nonfinite is None or nonfinite != 0:
        hard_vetoes.append("nonfinite_retained_samples")
        engineering_invalidity_reasons.append("nonfinite_retained_samples")
    final_state_finite = bool(
        np.all(np.isfinite(_private_tensor_to_numpy(result.final_state)))
    )
    if not final_state_finite:
        hard_vetoes.append("nonfinite_final_state")
        engineering_invalidity_reasons.append("nonfinite_final_state")
    log_accept_nonfinite = _int_or_none_metadata(
        diagnostics.get("log_accept_ratio_nonfinite_count")
    )
    if log_accept_nonfinite is None or log_accept_nonfinite != 0:
        hard_vetoes.append("nonfinite_private_acceptance_log_value")
    target_nonfinite = _int_or_none_metadata(
        diagnostics.get("target_log_prob_nonfinite_count")
    )
    if target_nonfinite is None or target_nonfinite != 0:
        hard_vetoes.append("nonfinite_private_target_value")
    return {
        "chunk_index": int(chunk_index),
        "retained_sample_count": int(retained_count),
        "expected_active_results": int(expected_active_results),
        "burnin_role": "initial" if int(chunk_index) == 0 else "continuation",
        "valid_sample_count": valid_sample_count,
        "observed_valid_mask_count": observed_valid_count,
        "nonfinite_valid_sample_count": nonfinite,
        "acceptance_rate": _float_or_none_metadata(diagnostics.get("acceptance_rate")),
        "acceptance_decision_count": _int_or_none_metadata(
            diagnostics.get("acceptance_decision_count")
        ),
        "acceptance_log_health_passed": log_accept_nonfinite == 0,
        "target_value_health_passed": target_nonfinite == 0,
        "divergence_status": diagnostics.get("divergence_status"),
        "divergence_count": _int_or_none_metadata(diagnostics.get("divergence_count")),
        "compile_trace_count": metadata.get("compile_trace_count"),
        "first_call_s": metadata.get("first_call_s"),
        "warm_call_s": metadata.get("warm_call_s"),
        "chunk_call_s": metadata.get("chunk_call_s"),
        "hard_vetoes": tuple(hard_vetoes),
        "engineering_invalidity_reasons": tuple(
            engineering_invalidity_reasons
        ),
        "candidate_data_invalidity_reasons": tuple(
            reason
            for reason in hard_vetoes
            if reason not in engineering_invalidity_reasons
        ),
    }


def _aggregate_acceptance_rate(chunks: Sequence[Mapping[str, Any]]) -> float | None:
    accepted = 0.0
    total = 0
    for chunk in chunks:
        rate = chunk.get("acceptance_rate")
        count = chunk.get("acceptance_decision_count")
        if rate is None or count is None:
            continue
        accepted += float(rate) * int(count)
        total += int(count)
    return None if total <= 0 else float(accepted / total)


def _aggregate_divergence_provenance(
    chunks: Sequence[Mapping[str, Any]],
) -> tuple[str, int | None, bool]:
    """Aggregate only complete, homogeneous native-divergence coverage."""

    if not chunks:
        return "not_collected", None, True
    allowed = {"available", "not_exposed_by_kernel", "not_collected"}
    statuses = tuple(str(chunk.get("divergence_status")) for chunk in chunks)
    if any(status not in allowed for status in statuses) or len(set(statuses)) != 1:
        return "not_collected", None, False
    status = statuses[0]
    counts = tuple(chunk.get("divergence_count") for chunk in chunks)
    if status == "available":
        if any(
            isinstance(count, (bool, np.bool_))
            or not isinstance(count, (int, np.integer))
            or int(count) < 0
            for count in counts
        ):
            return "not_collected", None, False
        return status, int(sum(int(count) for count in counts)), True
    if any(count is not None for count in counts):
        return "not_collected", None, False
    return status, None, True


def _max_compile_trace_count(chunks: Sequence[Mapping[str, Any]]) -> int | None:
    values = [
        int(chunk["compile_trace_count"])
        for chunk in chunks
        if chunk.get("compile_trace_count") is not None
    ]
    return None if not values else int(max(values))


def _first_non_none(values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _last_non_none(values: Any) -> Any:
    seen = None
    for value in values:
        if value is not None:
            seen = value
    return seen


def _float_or_none_metadata(value: Any) -> float | None:
    if value is None:
        return None
    converted = _tensor_or_plain_to_metadata(value)
    if converted is None:
        return None
    return float(converted)


def _int_or_none_metadata(value: Any) -> int | None:
    if value is None:
        return None
    converted = _tensor_or_plain_to_metadata(value)
    if converted is None:
        return None
    if isinstance(converted, bool) or not isinstance(converted, (int, np.integer)):
        return None
    return int(converted)


def _private_tensor_to_numpy(value: Any) -> np.ndarray:
    """Convert a private diagnostic tensor to NumPy without JSON list expansion."""

    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=float)


def _assert_sequential_rhat_public_safe(payload: Mapping[str, Any]) -> None:
    """Fail closed if sequential verifier metadata leaks raw HMC mechanics."""

    forbidden_tokens = (
        "raw_sample",
        "raw_state",
        "sample_path",
        "state_path",
        "mass_matrix",
        "selected_kernel",
        "target_log_prob",
        "log_accept",
        "step_size",
        "leapfrog",
        "final_state",
    )
    allowed_keys = {
        "public_summary_contains_step_size",
        "public_summary_contains_leapfrog_count",
        "public_summary_contains_mass_matrix",
        "min_log_accept_ratio",
        "max_log_accept_ratio",
        "max_abs_log_accept_energy_proxy",
        "signed_log_accept_ratio_tails",
    }

    def walk(value: Any, path: tuple[str, ...]) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                key_text = str(key)
                lowered = key_text.lower()
                if (
                    any(token in lowered for token in forbidden_tokens)
                    and key_text not in allowed_keys
                ):
                    raise ValueError(
                        "sequential R-hat public payload contains forbidden key: "
                        + ".".join(path + (key_text,))
                    )
                walk(item, path + (key_text,))
        elif isinstance(value, (tuple, list)):
            for index, item in enumerate(value):
                walk(item, path + (str(index),))

    walk(payload, tuple())


def _assert_hmc_archive_manifest_public_safe(payload: Mapping[str, Any]) -> None:
    """Fail closed if archive metadata exposes forbidden raw HMC payload keys."""

    forbidden_tokens = (
        "raw_sample",
        "raw_state",
        "mass_matrix",
        "selected_kernel_payload",
        "step_size",
        "leapfrog",
    )

    def walk(value: Any, path: tuple[str, ...]) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                key_text = str(key)
                lowered = key_text.lower()
                if any(token in lowered for token in forbidden_tokens):
                    if key_text == "forbidden_public_payloads":
                        walk(item, path + (key_text,))
                        continue
                    raise ValueError(
                        "HMC archive manifest contains forbidden public key: "
                        + ".".join(path + (key_text,))
                    )
                walk(item, path + (key_text,))
        elif isinstance(value, (tuple, list)):
            for index, item in enumerate(value):
                walk(item, path + (str(index),))

    walk(payload, tuple())


def _write_retained_sample_hmc_archive_bundle(
    *,
    archive_dir: Path,
    archive_label: str,
    samples: Any,
    final_state: Any,
    final_target_log_prob: Any,
    config: RetainedSampleHMCArchiveConfig,
    metadata: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    overwrite: bool,
) -> Mapping[str, Any]:
    """Persist one private retained-sample shard and public-safe summary.

    The returned summary intentionally omits file paths, tensor shapes, dtypes,
    and per-shard hashes.  Those details remain in the private manifest so a
    public client can prove the archive exists without seeing raw sample
    descriptors.
    """

    import tensorflow as tf

    root = Path(archive_dir)
    if root.exists() and not root.is_dir():
        raise ValueError("archive_dir exists and is not a directory")
    root.mkdir(parents=True, exist_ok=True)
    sample_tensor = tf.convert_to_tensor(samples)
    state_tensor = tf.convert_to_tensor(final_state)
    target_tensor = tf.convert_to_tensor(final_target_log_prob)
    expected_sample_shape = (int(config.num_results),) + tuple(
        int(dim) for dim in state_tensor.shape
    )
    if tuple(sample_tensor.shape.as_list()) != expected_sample_shape:
        raise ValueError("retained sample tensor shape must be (num_results,) + state shape")

    sample_path = root / f"{archive_label}_retained_samples.tftensor"
    state_path = root / f"{archive_label}_final_state.tftensor"
    target_path = root / f"{archive_label}_final_target_log_prob.tftensor"
    manifest_path = root / f"{archive_label}_private_manifest.json"
    for path in (sample_path, state_path, target_path, manifest_path):
        if path.exists() and not overwrite:
            raise FileExistsError(f"retained archive file already exists: {path}")

    sample_hash, sample_bytes = _write_serialized_tensor(sample_path, sample_tensor)
    state_hash, state_bytes = _write_serialized_tensor(state_path, state_tensor)
    target_hash, target_bytes = _write_serialized_tensor(target_path, target_tensor)
    finite_by_sample = tf.reduce_all(
        tf.math.is_finite(sample_tensor),
        axis=tf.range(1, tf.rank(sample_tensor), dtype=tf.int32),
    )
    nonfinite_count = int(
        tf.reduce_sum(
            tf.cast(tf.logical_not(finite_by_sample), tf.int32)
        ).numpy()
    )
    private_manifest = {
        "artifact_type": "bayesfilter_private_retained_sample_hmc_archive",
        "schema_version": 1,
        "archive_label": str(archive_label),
        "archive_dir": str(root),
        "sample_shards": (
            {
                "kind": "retained_samples",
                "path": str(sample_path),
                "sha256": sample_hash,
                "bytes": sample_bytes,
                "shape": tuple(int(dim) for dim in sample_tensor.shape),
                "dtype": sample_tensor.dtype.name,
            },
        ),
        "sidecars": {
            "final_state": {
                "path": str(state_path),
                "sha256": state_hash,
                "bytes": state_bytes,
                "shape": tuple(int(dim) for dim in state_tensor.shape),
                "dtype": state_tensor.dtype.name,
            },
            "final_target_log_prob": {
                "path": str(target_path),
                "sha256": target_hash,
                "bytes": target_bytes,
                "shape": tuple(int(dim) for dim in target_tensor.shape),
                "dtype": target_tensor.dtype.name,
            },
        },
        "sample_shard_count": 1,
        "sidecar_count": 2,
        "retained_sample_count": int(config.num_results),
        "nonfinite_retained_sample_count": int(nonfinite_count),
        "config_private_signature": program_signature(config.signature_payload()),
        "metadata": _json_safe_metadata(dict(metadata)),
        "diagnostics_private_metadata": _json_safe_metadata(
            {key: _tensor_or_plain_to_metadata(value) for key, value in diagnostics.items()}
        ),
        "privacy_contract": {
            "contains_private_raw_tensors": True,
            "public_summary_contains_paths": False,
            "public_summary_contains_raw_values": False,
            "public_summary_contains_tensor_descriptors": False,
        },
    }
    manifest_text = json.dumps(
        _json_safe_metadata(private_manifest),
        indent=2,
        sort_keys=True,
    ) + "\n"
    manifest_path.write_text(manifest_text, encoding="utf-8")
    manifest_hash = hashlib.sha256(manifest_text.encode("utf-8")).hexdigest()
    public_summary = {
        "artifact_type": "bayesfilter_retained_sample_hmc_archive_summary",
        "schema_version": 1,
        "archive_label": str(archive_label),
        "private_archive_shard_count": 1,
        "sample_shard_count": 1,
        "non_sample_sidecar_count": 3,
        "retained_sample_count": int(config.num_results),
        "total_valid_sample_count": int(config.num_results),
        "total_nonfinite_valid_sample_count": int(nonfinite_count),
        "aggregate_private_tensor_bytes": int(sample_bytes + state_bytes + target_bytes),
        "sample_shard_bytes": int(sample_bytes),
        "private_manifest_sha256": manifest_hash,
        "private_manifest_bytes": len(manifest_text.encode("utf-8")),
        "private_paths_publicized": False,
        "private_sample_or_state_descriptors_publicized": False,
        "public_summary_contains_raw_values": False,
        "public_summary_contains_kernel_payload": False,
        "private_manifest_is_sidecar_not_sample_shard": True,
        "final_state_sidecar_is_not_sample_shard": True,
        "privacy_contract": {
            "archive_contains_private_raw_tensors": True,
            "public_summary_contains_paths": False,
            "public_summary_contains_raw_values": False,
            "public_summary_contains_tensor_descriptors": False,
            "public_summary_contains_step_size": False,
            "public_summary_contains_leapfrog_count": False,
            "public_summary_contains_mass_matrix": False,
        },
        "nonclaims": (
            "retained HMC sample archive mechanics only",
            "no posterior convergence claim",
            "no serious sampler health claim",
            "no long-run retained-memory claim",
            "no scientific validity claim",
        ),
    }
    _assert_retained_archive_public_summary_safe(public_summary)
    return _json_safe_metadata(public_summary)


def _assert_retained_archive_public_summary_safe(payload: Mapping[str, Any]) -> None:
    """Fail closed on public retained-archive summary leakage."""

    forbidden_tokens = (
        "_path",
        "path",
        "shape",
        "dtype",
        "sample_sha",
        "state_sha",
        "raw_sample",
        "raw_state",
        "step_size",
        "leapfrog",
        "mass_matrix",
        "selected_kernel",
    )

    def walk(value: Any, path: tuple[str, ...]) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                key_text = str(key)
                lowered = key_text.lower()
                if any(token in lowered for token in forbidden_tokens):
                    allowed = key_text in {
                        "private_paths_publicized",
                        "sample_shard_count",
                        "sample_shard_bytes",
                        "private_manifest_is_sidecar_not_sample_shard",
                        "final_state_sidecar_is_not_sample_shard",
                        "public_summary_contains_paths",
                        "public_summary_contains_raw_values",
                        "public_summary_contains_step_size",
                        "public_summary_contains_leapfrog_count",
                        "public_summary_contains_mass_matrix",
                    }
                    if not allowed:
                        raise ValueError(
                            "retained archive public summary contains forbidden key: "
                            + ".".join(path + (key_text,))
                        )
                walk(item, path + (key_text,))
        elif isinstance(value, (tuple, list)):
            for index, item in enumerate(value):
                walk(item, path + (str(index),))

    walk(payload, tuple())


def _covariance_eigen_summary(covariance: np.ndarray) -> Mapping[str, Any]:
    eigenvalues = np.linalg.eigvalsh(0.5 * (covariance + covariance.T))
    finite = bool(np.all(np.isfinite(eigenvalues)))
    positive = bool(finite and float(np.min(eigenvalues)) > 0.0)
    return {
        "finite": finite,
        "positive": positive,
        "min": float(np.min(eigenvalues)) if finite else float("nan"),
        "max": float(np.max(eigenvalues)) if finite else float("nan"),
        "condition_number": (
            float(np.max(eigenvalues) / np.min(eigenvalues)) if positive else float("inf")
        ),
        "eigenvalues": tuple(float(value) for value in eigenvalues),
    }


def _normalize_eigen_summary(summary: Mapping[str, Any]) -> Mapping[str, Any]:
    eigenvalues = summary.get("eigenvalues", ())
    return {
        "finite": bool(summary.get("finite")),
        "positive": bool(summary.get("positive")),
        "min": float(summary.get("min")),
        "max": float(summary.get("max")),
        "condition_number": float(summary.get("condition_number")),
        "eigenvalues": tuple(float(value) for value in eigenvalues),
    }


def validate_precomputed_map(
    artifact: PrecomputedMAP,
    adapter: Any,
    *,
    expected_dim: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate reusable MAP/mass metadata before it can be consumed."""

    position = np.asarray(artifact.position, dtype=float)
    covariance = np.asarray(artifact.covariance, dtype=float)
    if position.ndim != 1:
        raise ValueError("precomputed MAP position must be a one-dimensional vector")
    dim = int(position.shape[0])
    if expected_dim is not None and dim != int(expected_dim):
        raise ValueError(f"precomputed MAP dimension {dim} does not match {expected_dim}")
    if covariance.shape != (dim, dim):
        raise ValueError(
            "precomputed covariance shape must match the MAP position dimension"
        )
    expected_signature = stable_adapter_signature(adapter)
    if str(artifact.adapter_signature) != expected_signature:
        raise ValueError("precomputed MAP adapter signature mismatch")
    return position, covariance


def latent_to_position(z: Any, center: Any, whitening_factor: Any) -> np.ndarray:
    """Map row-vector latent coordinates with theta = center + z @ W.T."""

    transform = LatentAffineHMCTransform(
        center=center,
        factor=whitening_factor,
        covariance_provenance="legacy_whitening_factor",
    )
    return transform.latent_to_position(z)


def latent_value_and_score(
    z: Any,
    center: Any,
    whitening_factor: Any,
    value_and_grad_fn: Callable[[np.ndarray], tuple[Any, Any]],
) -> tuple[Any, np.ndarray]:
    """Evaluate a transformed target and map theta-score to latent-score."""

    transform = LatentAffineHMCTransform(
        center=center,
        factor=whitening_factor,
        covariance_provenance="legacy_whitening_factor",
    )
    return transform.value_and_score(z, value_and_grad_fn)


def static_unroll_chain_value_and_score(
    adapter: Any,
    chain_state: Any,
    *,
    use_xla: bool = False,
    target_scope: str | None = None,
) -> tuple[Any, Any]:
    """Evaluate value/score for a statically sized chain batch."""

    import tensorflow as tf

    capability = value_score_capability(adapter)
    if use_xla and not capability.is_accepted_xla_hmc_authority:
        raise ValueError(
            "XLA chain-batched value/score requires graph-native or reviewed authority; "
            f"got {capability.value_score_authority!r}"
        )
    if use_xla and capability.target_scope is not None and str(target_scope) != capability.target_scope:
        raise ValueError("value/score target_scope mismatch")
    if not hasattr(adapter, "log_prob_and_grad"):
        raise TypeError("adapter must expose log_prob_and_grad for chain value/score")

    shape = getattr(chain_state, "shape", None)
    rank = getattr(shape, "rank", None)
    if rank is None and shape is not None:
        rank = len(shape)
    if rank != 2:
        raise ValueError("chain_state must have static rank 2 [chain, parameter]")
    chain_count = shape[0]
    parameter_dim = shape[1]
    if chain_count is None or parameter_dim is None:
        raise ValueError("chain_state must have static chain and parameter dimensions")

    values = tf.TensorArray(
        dtype=chain_state.dtype,
        size=int(chain_count),
        element_shape=tf.TensorShape([]),
    )
    scores = tf.TensorArray(
        dtype=chain_state.dtype,
        size=int(chain_count),
        element_shape=tf.TensorShape([int(parameter_dim)]),
    )

    def body(index: Any, values_ta: Any, scores_ta: Any) -> tuple[Any, Any, Any]:
        value, score = adapter.log_prob_and_grad(chain_state[index])
        return (
            index + tf.constant(1, dtype=tf.int32),
            values_ta.write(index, tf.cast(value, chain_state.dtype)),
            scores_ta.write(index, tf.cast(score, chain_state.dtype)),
        )

    _index, values, scores = tf.while_loop(
        lambda index, *_unused: index < tf.constant(int(chain_count), dtype=tf.int32),
        body,
        (tf.constant(0, dtype=tf.int32), values, scores),
        parallel_iterations=1,
    )
    return values.stack(), scores.stack()


def _make_tfp_target_log_prob_fn(
    adapter: Any,
    *,
    dtype: Any,
) -> Callable[[Any], Any]:
    if hasattr(adapter, "log_prob_and_grad"):
        return reviewed_value_score_target_fn(adapter, dtype=dtype)
    return _make_hmc_target_log_prob_fn(adapter)


def _broadcast_upstream_gradient_to_score(dy: Any, score: Any) -> Any:
    """Broadcast target upstream gradients over trailing parameter axes only."""

    import tensorflow as tf

    score_tensor = tf.convert_to_tensor(score)
    upstream = tf.cast(dy, score_tensor.dtype)
    upstream_shape = upstream.shape
    score_shape = score_tensor.shape
    if upstream_shape.rank is not None and score_shape.rank is not None:
        if upstream_shape.rank > score_shape.rank:
            raise ValueError("target upstream gradient rank exceeds score rank")
        for idx, dim in enumerate(upstream_shape.as_list()):
            score_dim = score_shape[idx]
            if dim is not None and score_dim is not None and int(dim) != int(score_dim):
                raise ValueError(
                    "target upstream gradient leading dimensions must match score"
                )
        if upstream_shape.rank == score_shape.rank:
            return upstream
        return tf.reshape(
            upstream,
            tf.concat(
                [
                    tf.shape(upstream),
                    tf.ones([score_shape.rank - upstream_shape.rank], dtype=tf.int32),
                ],
                axis=0,
            ),
        )
    rank_delta = tf.rank(score_tensor) - tf.rank(upstream)
    with tf.control_dependencies(
        [
            tf.debugging.assert_greater_equal(
                rank_delta,
                0,
                message="target upstream gradient rank exceeds score rank",
            ),
            tf.debugging.assert_equal(
                tf.shape(upstream),
                tf.shape(score_tensor)[: tf.rank(upstream)],
                message="target upstream gradient leading dimensions must match score",
            ),
        ]
    ):
        return tf.reshape(
            upstream,
            tf.concat([tf.shape(upstream), tf.ones([rank_delta], dtype=tf.int32)], axis=0),
        )


def _build_sample_chain_runner(
    config: FullChainHMCConfig,
    kernel: Any,
    trace_fn: Callable[[Any, Any], Mapping[str, Any]],
    initial_state: Any,
) -> Callable[[], tuple[Any, Mapping[str, Any]]]:
    import tensorflow as tf
    import tensorflow_probability as tfp

    tfm = tfp.mcmc

    def run_chain() -> tuple[Any, Mapping[str, Any]]:
        return tfm.sample_chain(
            num_results=config.num_results,
            num_burnin_steps=config.num_burnin_steps,
            current_state=initial_state,
            kernel=kernel,
            trace_fn=trace_fn,
            seed=tf.constant(config.seed, dtype=tf.int32),
        )

    if config.use_xla:
        return tf.function(run_chain, jit_compile=True, reduce_retracing=True)
    if config.chain_execution_mode == "eager":
        return run_chain
    return tf.function(run_chain, reduce_retracing=True)


def _trace_fn_for_config(
    config: FullChainHMCConfig,
    *,
    adapter: Any | None = None,
) -> Callable[[Any, Any], Mapping[str, Any]]:
    if config.target_status_trace_policy != "none":
        if adapter is None or not callable(getattr(adapter, "target_status_telemetry", None)):
            raise TypeError(
                "target_status_trace_policy='per_chain_step' requires adapter "
                "target_status_telemetry"
            )
    if config.trace_policy != "standard":
        return _reduced_trace_fn
    if config.target_status_trace_policy == "per_chain_step":
        if config.tuning_policy.uses_dual_averaging:
            return _adaptive_standard_trace_fn_with_target_status(adapter)
        return _standard_trace_fn_with_target_status(adapter)
    if config.tuning_policy.uses_dual_averaging:
        return _adaptive_standard_trace_fn
    return _standard_trace_fn


def _standard_trace_fn_with_target_status(adapter: Any) -> Callable[[Any, Any], Mapping[str, Any]]:
    def trace_fn(state: Any, kernel_results: Any) -> Mapping[str, Any]:
        trace = dict(_standard_trace_fn(state, kernel_results))
        trace["target_status_telemetry"] = adapter.target_status_telemetry(state)
        return trace

    return trace_fn


def _standard_trace_fn(_state: Any, kernel_results: Any) -> Mapping[str, Any]:
    trace = {
        "is_accepted": kernel_results.is_accepted,
        "log_accept_ratio": kernel_results.log_accept_ratio,
        "target_log_prob": kernel_results.accepted_results.target_log_prob,
        "proposed_target_log_prob": kernel_results.proposed_results.target_log_prob,
    }
    proposed_correction = getattr(
        kernel_results.proposed_results,
        "log_acceptance_correction",
        None,
    )
    if proposed_correction is not None:
        trace["log_acceptance_correction"] = proposed_correction
    trace.update(_native_divergence_trace(kernel_results))
    return trace


def _adaptive_standard_trace_fn_with_target_status(adapter: Any) -> Callable[[Any, Any], Mapping[str, Any]]:
    def trace_fn(state: Any, kernel_results: Any) -> Mapping[str, Any]:
        trace = dict(_adaptive_standard_trace_fn(state, kernel_results))
        trace["target_status_telemetry"] = adapter.target_status_telemetry(state)
        return trace

    return trace_fn


def _adaptive_standard_trace_fn(_state: Any, kernel_results: Any) -> Mapping[str, Any]:
    inner_results = kernel_results.inner_results
    trace = {
        "is_accepted": inner_results.is_accepted,
        "log_accept_ratio": inner_results.log_accept_ratio,
        "target_log_prob": inner_results.accepted_results.target_log_prob,
        "step_size": kernel_results.new_step_size,
        "target_accept_prob": kernel_results.target_accept_prob,
        "num_adaptation_steps": kernel_results.num_adaptation_steps,
    }
    trace.update(_native_divergence_trace(inner_results))
    return trace


def _native_divergence_trace(kernel_results: Any) -> Mapping[str, Any]:
    """Expose only native boolean divergence fields from kernel results."""

    divergence = _extract_native_divergence_tensor(kernel_results)
    if divergence is None:
        return {}
    return {"divergence": divergence}


def _extract_native_divergence_tensor(kernel_results: Any) -> Any | None:
    """Return a native divergence tensor, never an energy/log-accept proxy."""

    import tensorflow as tf

    result_objects = (
        kernel_results,
        getattr(kernel_results, "proposed_results", None),
        getattr(kernel_results, "accepted_results", None),
    )
    field_names = (
        "is_divergent",
        "has_divergence",
        "divergence",
        "divergences",
    )
    for result_object in result_objects:
        if result_object is None:
            continue
        for field_name in field_names:
            if not hasattr(result_object, field_name):
                continue
            value = getattr(result_object, field_name)
            if value is None:
                continue
            try:
                tensor = tf.convert_to_tensor(value)
            except (TypeError, ValueError):
                continue
            if tensor.dtype == tf.bool:
                return tensor
    return None


def _reduced_trace_fn(_state: Any, _kernel_results: Any) -> Mapping[str, Any]:
    import tensorflow as tf

    return {"trace_collected": tf.constant(True)}


def _trace_unavailability(
    trace_policy: str,
    trace: Mapping[str, Any] | None = None,
    *,
    target_status_trace_policy: str = "none",
) -> Mapping[str, str]:
    target_status_trace_policy = str(target_status_trace_policy)
    if trace_policy == "standard":
        unavailable = {}
        if trace is not None and "divergence" not in trace:
            unavailable.update(
                {
                    "divergence": (
                        "native boolean divergence field not exposed by "
                        "TensorFlow Probability HMC kernel results"
                    ),
                }
            )
        if (
            target_status_trace_policy == "per_chain_step"
            and trace is not None
            and "target_status_telemetry" not in trace
        ):
            unavailable["target_status_telemetry"] = "target-status trace missing"
        return unavailable
    unavailable = {
        "is_accepted": "reduced trace policy",
        "log_accept_ratio": "reduced trace policy",
        "target_log_prob": "reduced trace policy",
        "divergence": "reduced trace policy",
    }
    if target_status_trace_policy != "none":
        unavailable["target_status_telemetry"] = "reduced trace policy"
    return unavailable


def _native_divergence_status_for_trace(
    trace: Mapping[str, Any],
    *,
    trace_policy: str,
) -> str:
    """Classify native divergence telemetry availability without proxies."""

    if "divergence" in trace:
        return "available"
    if trace_policy == "standard":
        return "not_exposed_by_kernel"
    return "unavailable"


def _hmc_health_diagnostics(trace: Mapping[str, Any]) -> Mapping[str, Any]:
    """Summarize non-divergence sampler health traces separately."""

    import tensorflow as tf

    health: dict[str, Any] = {
        "diagnostic_role": "hmc_health_diagnostics_not_native_divergence",
        "nonclaims": (
            "acceptance/log-accept/target-log-prob diagnostics are not native divergence telemetry",
            "no sampler convergence claim",
        ),
    }
    if "is_accepted" in trace:
        accepted = tf.cast(trace["is_accepted"], tf.float64)
        health["acceptance_rate"] = tf.reduce_mean(accepted)
        health["acceptance_finite"] = tf.reduce_all(tf.math.is_finite(accepted))
    else:
        health["acceptance_rate"] = None
        health["acceptance_finite"] = None
    if "log_accept_ratio" in trace:
        log_accept = tf.convert_to_tensor(trace["log_accept_ratio"], dtype=tf.float64)
        finite = tf.math.is_finite(log_accept)
        finite_values = tf.boolean_mask(log_accept, finite)
        health["log_accept_ratio"] = {
            "available": True,
            "finite": tf.reduce_all(finite),
            "finite_count": tf.reduce_sum(tf.cast(finite, tf.int32)),
            "nonfinite_count": tf.reduce_sum(tf.cast(tf.logical_not(finite), tf.int32)),
            "max_abs_finite": tf.cond(
                tf.size(finite_values) > 0,
                lambda: tf.reduce_max(tf.abs(finite_values)),
                lambda: tf.constant(float("nan"), dtype=tf.float64),
            ),
        }
    else:
        health["log_accept_ratio"] = {"available": False}
    if "target_log_prob" in trace:
        target_log_prob = tf.convert_to_tensor(trace["target_log_prob"], dtype=tf.float64)
        health["target_log_prob"] = {
            "available": True,
            "finite": tf.reduce_all(tf.math.is_finite(target_log_prob)),
            "min": tf.reduce_min(target_log_prob),
            "max": tf.reduce_max(target_log_prob),
        }
    else:
        health["target_log_prob"] = {"available": False}
    if "proposed_target_log_prob" in trace:
        proposed_target_log_prob = tf.convert_to_tensor(
            trace["proposed_target_log_prob"],
            dtype=tf.float64,
        )
        health["proposed_target_log_prob"] = {
            "available": True,
            "finite": tf.reduce_all(tf.math.is_finite(proposed_target_log_prob)),
            "min": tf.reduce_min(proposed_target_log_prob),
            "max": tf.reduce_max(proposed_target_log_prob),
        }
    else:
        health["proposed_target_log_prob"] = {"available": False}
    if "log_acceptance_correction" in trace:
        correction = tf.convert_to_tensor(
            trace["log_acceptance_correction"],
            dtype=tf.float64,
        )
        finite = tf.math.is_finite(correction)
        finite_values = tf.boolean_mask(correction, finite)
        health["log_acceptance_correction"] = {
            "available": True,
            "finite": tf.reduce_all(finite),
            "finite_count": tf.reduce_sum(tf.cast(finite, tf.int32)),
            "nonfinite_count": tf.reduce_sum(tf.cast(tf.logical_not(finite), tf.int32)),
            "max_abs_finite": tf.cond(
                tf.size(finite_values) > 0,
                lambda: tf.reduce_max(tf.abs(finite_values)),
                lambda: tf.constant(float("nan"), dtype=tf.float64),
            ),
        }
    else:
        health["log_acceptance_correction"] = {"available": False}
    return health


def _full_chain_hmc_diagnostics(
    samples: Any,
    trace: Mapping[str, Any],
    *,
    trace_policy: str,
) -> Mapping[str, Any]:
    import tensorflow as tf

    finite_by_sample = tf.reduce_all(tf.math.is_finite(samples), axis=-1)
    diagnostics: dict[str, Any] = {
        "finite_sample_count": tf.reduce_sum(tf.cast(finite_by_sample, tf.int32)),
        "nonfinite_sample_count": tf.reduce_sum(
            tf.cast(tf.logical_not(finite_by_sample), tf.int32)
        ),
        "sample_shape": tuple(int(dim) for dim in samples.shape),
        "trace_policy": trace_policy,
        "native_divergence_status": _native_divergence_status_for_trace(
            trace,
            trace_policy=trace_policy,
        ),
        "divergence_status": _native_divergence_status_for_trace(
            trace,
            trace_policy=trace_policy,
        ),
        "divergence_count": None,
        "divergence_source": None,
        "hmc_health_diagnostics": _hmc_health_diagnostics(trace),
        "nonclaims": (
            "finite tiny-chain diagnostics only",
            "no sampler convergence claim",
            "native divergence unavailability is not zero divergences",
        ),
    }
    if "is_accepted" in trace:
        diagnostics["acceptance_rate"] = tf.reduce_mean(
            tf.cast(trace["is_accepted"], tf.float64)
        )
    else:
        diagnostics["acceptance_rate"] = None
    if "divergence" in trace:
        divergence = tf.convert_to_tensor(trace["divergence"], dtype=tf.bool)
        diagnostics["native_divergence_status"] = "available"
        diagnostics["divergence_status"] = "available"
        diagnostics["divergence_count"] = tf.reduce_sum(
            tf.cast(divergence, tf.int32)
        )
        diagnostics["divergence_source"] = "native_boolean_tfp_kernel_result"
    if "target_log_prob" in trace:
        diagnostics["min_target_log_prob"] = tf.reduce_min(trace["target_log_prob"])
        diagnostics["max_target_log_prob"] = tf.reduce_max(trace["target_log_prob"])
    if "target_status_telemetry" in trace:
        diagnostics["target_status_telemetry"] = _target_status_telemetry_diagnostics(
            trace["target_status_telemetry"]
        )
    if all(key in trace for key in ("is_accepted", "log_accept_ratio")):
        diagnostics["tuning_telemetry"] = summarize_hmc_tuning_telemetry(
            samples=samples,
            log_accept_ratio=trace["log_accept_ratio"],
            is_accepted=trace["is_accepted"],
        )
    if "step_size" in trace:
        step_size = tf.convert_to_tensor(trace["step_size"], dtype=tf.float64)
        diagnostics["final_step_size"] = step_size[-1]
        diagnostics["final_step_size_finite"] = tf.reduce_all(tf.math.is_finite(step_size))
    if "target_accept_prob" in trace:
        target_accept = tf.convert_to_tensor(
            trace["target_accept_prob"],
            dtype=tf.float64,
        )
        diagnostics["target_accept_prob"] = tf.reshape(target_accept, [-1])[-1]
    if "num_adaptation_steps" in trace:
        adaptation_steps = tf.convert_to_tensor(
            trace["num_adaptation_steps"],
            dtype=tf.int32,
        )
        diagnostics["num_adaptation_steps"] = tf.reshape(adaptation_steps, [-1])[-1]
    return diagnostics


def _fixed_size_hmc_chunk_diagnostics(
    samples: Any,
    valid_mask: Any,
    trace: Mapping[str, Any],
    *,
    trace_policy: str,
) -> Mapping[str, Any]:
    """Summarize only valid rows from a fixed-size HMC chunk buffer."""

    import tensorflow as tf

    mask = tf.convert_to_tensor(valid_mask, dtype=tf.bool)
    finite_by_row = tf.reduce_all(tf.math.is_finite(samples), axis=-1)
    valid_finite = tf.boolean_mask(finite_by_row, mask)
    valid_count = tf.reduce_sum(tf.cast(mask, tf.int32))
    invalid_count = tf.shape(mask, out_type=tf.int32)[0] - valid_count
    diagnostics: dict[str, Any] = {
        "valid_sample_count": valid_count,
        "invalid_sample_count": invalid_count,
        "finite_valid_sample_count": tf.reduce_sum(tf.cast(valid_finite, tf.int32)),
        "nonfinite_valid_sample_count": tf.reduce_sum(
            tf.cast(tf.logical_not(valid_finite), tf.int32)
        ),
        "sample_shape": tuple(int(dim) for dim in samples.shape),
        "valid_mask_shape": tuple(int(dim) for dim in mask.shape),
        "trace_policy": trace_policy,
        "native_divergence_status": _native_divergence_status_for_trace(
            trace,
            trace_policy=trace_policy,
        ),
        "divergence_status": _native_divergence_status_for_trace(
            trace,
            trace_policy=trace_policy,
        ),
        "divergence_count": None,
        "divergence_source": None,
        "hmc_health_diagnostics": _hmc_health_diagnostics(trace),
        "nonclaims": (
            "fixed-size chunk finite-sample diagnostics only",
            "invalid buffer rows are not counted as samples",
            "no sampler convergence claim",
            "native divergence unavailability is not zero divergences",
        ),
    }
    if "acceptance_rate" in trace:
        diagnostics["acceptance_rate"] = tf.convert_to_tensor(
            trace["acceptance_rate"],
            dtype=tf.float64,
        )
        diagnostics["acceptance_rate_by_chain"] = tf.convert_to_tensor(
            trace["acceptance_rate_by_chain"],
            dtype=tf.float64,
        )
        diagnostics["accepted_decision_count"] = tf.convert_to_tensor(
            trace["accepted_decision_count"],
            dtype=tf.int32,
        )
        diagnostics["acceptance_decision_count"] = tf.convert_to_tensor(
            trace["acceptance_decision_count"],
            dtype=tf.int32,
        )
    if "log_accept_ratio_finite_count" in trace:
        diagnostics["log_accept_ratio_finite_count"] = tf.convert_to_tensor(
            trace["log_accept_ratio_finite_count"],
            dtype=tf.int32,
        )
        diagnostics["log_accept_ratio_nonfinite_count"] = tf.convert_to_tensor(
            trace["log_accept_ratio_nonfinite_count"],
            dtype=tf.int32,
        )
        diagnostics["log_accept_ratio_max_abs_finite"] = tf.convert_to_tensor(
            trace["log_accept_ratio_max_abs_finite"],
            dtype=tf.float64,
        )
    if "target_log_prob_finite_count" in trace:
        diagnostics["target_log_prob_finite_count"] = tf.convert_to_tensor(
            trace["target_log_prob_finite_count"],
            dtype=tf.int32,
        )
        diagnostics["target_log_prob_nonfinite_count"] = tf.convert_to_tensor(
            trace["target_log_prob_nonfinite_count"],
            dtype=tf.int32,
        )
        diagnostics["target_log_prob_min_finite"] = tf.convert_to_tensor(
            trace["target_log_prob_min_finite"],
            dtype=tf.float64,
        )
        diagnostics["target_log_prob_max_finite"] = tf.convert_to_tensor(
            trace["target_log_prob_max_finite"],
            dtype=tf.float64,
        )
    if "divergence" in trace:
        divergence = tf.boolean_mask(
            tf.convert_to_tensor(trace["divergence"], dtype=tf.bool),
            mask,
        )
        diagnostics["native_divergence_status"] = "available"
        diagnostics["divergence_status"] = "available"
        diagnostics["divergence_count"] = tf.reduce_sum(
            tf.cast(divergence, tf.int32)
        )
        diagnostics["divergence_source"] = "native_boolean_tfp_kernel_result"
    elif bool(_tensor_or_plain_to_metadata(trace.get("native_divergence_available", False))):
        diagnostics["native_divergence_status"] = "available"
        diagnostics["divergence_status"] = "available"
        diagnostics["divergence_count"] = tf.convert_to_tensor(
            trace["divergence_count"],
            dtype=tf.int32,
        )
        diagnostics["divergence_count_by_chain"] = tf.convert_to_tensor(
            trace["divergence_count_by_chain"],
            dtype=tf.int32,
        )
        diagnostics["divergence_source"] = "native_boolean_tfp_kernel_result"
    elif "native_divergence_available" in trace:
        diagnostics["native_divergence_status"] = "not_exposed_by_kernel"
        diagnostics["divergence_status"] = "not_exposed_by_kernel"
        diagnostics["divergence_count"] = None
        diagnostics["divergence_count_by_chain"] = None
        diagnostics["divergence_source"] = None
    if all(key in trace for key in ("is_accepted", "log_accept_ratio")):
        diagnostics["tuning_telemetry"] = summarize_hmc_tuning_telemetry(
            samples=tf.boolean_mask(samples, mask),
            log_accept_ratio=tf.boolean_mask(
                tf.convert_to_tensor(trace["log_accept_ratio"], dtype=tf.float64),
                mask,
            ),
            is_accepted=tf.boolean_mask(
                tf.convert_to_tensor(trace["is_accepted"], dtype=tf.bool),
                mask,
            ),
        )
    return diagnostics


def _internal_segment_hmc_diagnostics(
    *,
    final_state: Any,
    final_target_log_prob: Any,
    final_index: Any,
    segment_index: Any,
    segment_indices: Any,
    segment_target_log_prob: Any,
    config: InternalSegmentHMCRunnerConfig,
) -> Mapping[str, Any]:
    """Summarize a summary-only internal-segment HMC call."""

    import tensorflow as tf

    final_state_tensor = tf.convert_to_tensor(final_state)
    final_target_log_prob_tensor = tf.convert_to_tensor(final_target_log_prob)
    final_index_tensor = tf.convert_to_tensor(final_index, dtype=tf.int32)
    segment_index_tensor = tf.convert_to_tensor(segment_index, dtype=tf.int32)
    segment_indices_tensor = tf.convert_to_tensor(segment_indices, dtype=tf.int32)
    segment_target_log_prob_tensor = tf.convert_to_tensor(segment_target_log_prob)
    expected_indices = tf.range(
        tf.constant(1, dtype=tf.int32),
        tf.constant(config.segment_count + 1, dtype=tf.int32),
        dtype=tf.int32,
    ) * tf.constant(config.segment_length, dtype=tf.int32)
    return {
        "final_state_all_finite": tf.reduce_all(tf.math.is_finite(final_state_tensor)),
        "final_target_log_prob_all_finite": tf.reduce_all(
            tf.math.is_finite(final_target_log_prob_tensor)
        ),
        "segment_target_log_prob_all_finite": tf.reduce_all(
            tf.math.is_finite(segment_target_log_prob_tensor)
        ),
        "final_index": final_index_tensor,
        "expected_final_index": tf.constant(config.total_transitions, dtype=tf.int32),
        "final_index_matches_total_transitions": tf.equal(
            final_index_tensor,
            tf.constant(config.total_transitions, dtype=tf.int32),
        ),
        "segment_index": segment_index_tensor,
        "expected_segment_count": tf.constant(config.segment_count, dtype=tf.int32),
        "segment_index_matches_segment_count": tf.equal(
            segment_index_tensor,
            tf.constant(config.segment_count, dtype=tf.int32),
        ),
        "segment_indices_match_expected": tf.reduce_all(
            tf.equal(segment_indices_tensor, expected_indices)
        ),
        "segment_indices_shape": tuple(int(dim) for dim in segment_indices_tensor.shape),
        "segment_target_log_prob_shape": tuple(
            int(dim) for dim in segment_target_log_prob_tensor.shape
        ),
        "returns_samples": False,
        "output_mode": config.output_mode,
        "nonclaims": (
            "summary-only internal-segment diagnostics only",
            "no retained posterior samples",
            "no sampler convergence claim",
            "native divergence telemetry is unavailable in this reduced summary mode",
        ),
    }


def _retained_sample_hmc_archive_diagnostics(
    *,
    samples: Any,
    final_state: Any,
    final_target_log_prob: Any,
    final_index: Any,
    config: RetainedSampleHMCArchiveConfig,
    hmc_execution_call_count: int,
    sampler_telemetry: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Summarize one-call retained HMC output without returning samples."""

    import tensorflow as tf

    sample_tensor = tf.convert_to_tensor(samples)
    final_state_tensor = tf.convert_to_tensor(final_state)
    final_target_log_prob_tensor = tf.convert_to_tensor(final_target_log_prob)
    final_index_tensor = tf.convert_to_tensor(final_index, dtype=tf.int32)
    finite_by_sample = tf.reduce_all(
        tf.math.is_finite(sample_tensor),
        axis=tf.range(1, tf.rank(sample_tensor), dtype=tf.int32),
    )
    diagnostics = {
        "retained_sample_count": tf.constant(config.num_results, dtype=tf.int32),
        "valid_sample_count": tf.constant(config.num_results, dtype=tf.int32),
        "finite_retained_sample_count": tf.reduce_sum(
            tf.cast(finite_by_sample, tf.int32)
        ),
        "nonfinite_retained_sample_count": tf.reduce_sum(
            tf.cast(tf.logical_not(finite_by_sample), tf.int32)
        ),
        "retained_samples_all_finite": tf.reduce_all(finite_by_sample),
        "final_state_all_finite": tf.reduce_all(tf.math.is_finite(final_state_tensor)),
        "final_target_log_prob_all_finite": tf.reduce_all(
            tf.math.is_finite(final_target_log_prob_tensor)
        ),
        "final_index": final_index_tensor,
        "expected_final_index": tf.constant(config.num_results, dtype=tf.int32),
        "final_index_matches_num_results": tf.equal(
            final_index_tensor,
            tf.constant(config.num_results, dtype=tf.int32),
        ),
        "returns_samples": False,
        "uses_sample_chain": False,
        "hmc_execution_call_count": int(hmc_execution_call_count),
        "macrofinance_visible_chunk_count": 0,
        "private_archive_shard_count": 1,
        "native_divergence_status": "not_collected",
        "divergence_status": "not_collected",
        "divergence_count": None,
        "divergence_source": None,
        "sampler_diagnostics_policy": config.sampler_diagnostics_policy,
        "sampler_health_diagnostics": {
            "available": False,
            "diagnostic_role": "hmc_health_diagnostics_not_native_divergence",
            "nonclaims": (
                "acceptance/log-accept/target-log-prob diagnostics are not native divergence telemetry",
                "no sampler convergence claim",
            ),
        },
        "nonclaims": (
            "one-call retained archive finite-sample diagnostics only",
            "raw retained samples are private and not returned",
            "no sampler convergence claim",
            "native divergence unavailability is not zero divergences",
        ),
    }
    if sampler_telemetry is None:
        return diagnostics
    telemetry = dict(sampler_telemetry)
    native_divergence_available = bool(
        tf.convert_to_tensor(telemetry["native_divergence_available"]).numpy()
    )
    if native_divergence_available:
        diagnostics["native_divergence_status"] = "available"
        diagnostics["divergence_status"] = "available"
        diagnostics["divergence_count"] = tf.convert_to_tensor(
            telemetry["divergence_count"],
            dtype=tf.int32,
        )
        diagnostics["divergence_count_by_chain"] = tf.convert_to_tensor(
            telemetry["divergence_count_by_chain"],
            dtype=tf.int32,
        )
        diagnostics["divergence_source"] = "native_boolean_tfp_kernel_result"
    else:
        diagnostics["native_divergence_status"] = "not_exposed_by_kernel"
        diagnostics["divergence_status"] = "not_exposed_by_kernel"
        diagnostics["divergence_count"] = None
        diagnostics["divergence_count_by_chain"] = None
        diagnostics["divergence_source"] = None
    diagnostics["acceptance_rate"] = tf.convert_to_tensor(
        telemetry["acceptance_rate"],
        dtype=tf.float64,
    )
    diagnostics["acceptance_rate_by_chain"] = tf.convert_to_tensor(
        telemetry["acceptance_rate_by_chain"],
        dtype=tf.float64,
    )
    diagnostics["accepted_decision_count"] = tf.convert_to_tensor(
        telemetry["accepted_decision_count"],
        dtype=tf.int32,
    )
    diagnostics["acceptance_decision_count"] = tf.convert_to_tensor(
        telemetry["acceptance_decision_count"],
        dtype=tf.int32,
    )
    diagnostics["sampler_health_diagnostics"] = {
        "available": True,
        "diagnostic_role": "hmc_health_diagnostics_not_native_divergence",
        "acceptance_rate": diagnostics["acceptance_rate"],
        "acceptance_rate_by_chain": diagnostics["acceptance_rate_by_chain"],
        "accepted_decision_count": diagnostics["accepted_decision_count"],
        "acceptance_decision_count": diagnostics["acceptance_decision_count"],
        "log_accept_ratio": {
            "available": True,
            "finite_count": tf.convert_to_tensor(
                telemetry["log_accept_ratio_finite_count"],
                dtype=tf.int32,
            ),
            "nonfinite_count": tf.convert_to_tensor(
                telemetry["log_accept_ratio_nonfinite_count"],
                dtype=tf.int32,
            ),
            "max_abs_finite": tf.convert_to_tensor(
                telemetry["log_accept_ratio_max_abs_finite"],
                dtype=tf.float64,
            ),
        },
        "target_log_prob": {
            "available": True,
            "finite_count": tf.convert_to_tensor(
                telemetry["target_log_prob_finite_count"],
                dtype=tf.int32,
            ),
            "nonfinite_count": tf.convert_to_tensor(
                telemetry["target_log_prob_nonfinite_count"],
                dtype=tf.int32,
            ),
            "min_finite": tf.convert_to_tensor(
                telemetry["target_log_prob_min_finite"],
                dtype=tf.float64,
            ),
            "max_finite": tf.convert_to_tensor(
                telemetry["target_log_prob_max_finite"],
                dtype=tf.float64,
            ),
        },
        "native_divergence": {
            "status": diagnostics["native_divergence_status"],
            "count": diagnostics["divergence_count"],
            "count_by_chain": diagnostics["divergence_count_by_chain"],
            "source": diagnostics["divergence_source"],
        },
        "nonclaims": (
            "acceptance/log-accept/target-log-prob diagnostics are not native divergence telemetry",
            "native divergence unavailability is not zero divergences",
            "no sampler convergence claim",
        ),
    }
    return diagnostics


def _target_status_telemetry_diagnostics(telemetry: Mapping[str, Any]) -> Mapping[str, Any]:
    import tensorflow as tf

    if not isinstance(telemetry, Mapping):
        raise TypeError("target_status_telemetry trace must be a mapping")
    required = (
        "status_code",
        "valid_pre_regularized_score",
        "floor_count_value",
        "min_innovation_eigenvalue",
        "innovation_condition_estimate",
    )
    missing = tuple(key for key in required if key not in telemetry)
    if missing:
        raise ValueError(
            "target_status_telemetry missing required fields: " + ", ".join(missing)
        )
    status = tf.convert_to_tensor(telemetry["status_code"], dtype=tf.int32)
    valid = tf.convert_to_tensor(telemetry["valid_pre_regularized_score"], dtype=tf.bool)
    floors = tf.convert_to_tensor(telemetry["floor_count_value"], dtype=tf.int32)
    min_eigen = tf.convert_to_tensor(telemetry["min_innovation_eigenvalue"], dtype=tf.float64)
    condition = tf.convert_to_tensor(
        telemetry["innovation_condition_estimate"],
        dtype=tf.float64,
    )
    status_nonvalid = tf.logical_or(
        tf.not_equal(status, tf.zeros_like(status)),
        tf.logical_not(valid),
    )
    return {
        "trace_entry_count": tf.size(status),
        "status_nonvalid_count": tf.reduce_sum(tf.cast(status_nonvalid, tf.int32)),
        "all_status_valid": tf.reduce_all(tf.logical_not(status_nonvalid)),
        "floor_count_total": tf.reduce_sum(floors),
        "max_floor_count_value": tf.reduce_max(floors),
        "min_min_innovation_eigenvalue": tf.reduce_min(min_eigen),
        "max_innovation_condition_estimate": tf.reduce_max(condition),
        "telemetry_failure_veto": tf.logical_not(
            tf.reduce_all(tf.logical_not(status_nonvalid))
        ),
    }


def program_signature(payload: Mapping[str, Any] | Any) -> str:
    """Build a stable SHA-256 signature for JSON-like metadata."""

    normalized = _normalize_for_json(payload)
    blob = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _stack_like_chain(chain_state: Any, rows: list[Any]) -> Any:
    if hasattr(chain_state, "dtype") and chain_state.__class__.__module__.startswith("tensorflow"):
        import tensorflow as tf

        return tf.stack(rows, axis=0)
    return np.stack(rows, axis=0)


def _normalize_for_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _normalize_for_json(val) for key, val in value.items()}
    if isinstance(value, (tuple, list)):
        return [_normalize_for_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return _normalize_for_json(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value
