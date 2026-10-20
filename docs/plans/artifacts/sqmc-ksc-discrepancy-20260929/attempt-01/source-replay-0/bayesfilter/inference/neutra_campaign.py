"""Fail-closed campaign integration for typed NeuTra posterior identities."""

from __future__ import annotations

import hashlib
import inspect
import json
import ast
import sys
import textwrap
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import tensorflow as tf

from bayesfilter.inference.batched_value_score import FixedTransportValueScoreAdapter
from bayesfilter.inference.neutra_artifacts import (
    LoadedFrozenNeuTraArtifact,
    load_frozen_neutra_artifact,
)
from bayesfilter.inference.neutra_batching import (
    NeuTraBatchTargetBinding,
    bind_batch_native_neutra_target,
)
from bayesfilter.inference.neutra_hmc import (
    BatchedHMCConfig,
    NEUTRA_SEQUENTIAL_HMC_POLICY_ID,
    SequentialNeuTraHMCConfig,
    run_batched_hmc,
    run_sequential_neutra_hmc,
)
from bayesfilter.inference.neutra_training import (
    PlainDenseIAFTrainingConfig,
    PlainDenseIAFTrainingResult,
    train_plain_dense_iaf,
)
from bayesfilter.runtime import append_jsonl, atomic_write_json
from bayesfilter.runtime.gpu_memory_policy import TF_GPU_MEMORY_POLICY_SCHEMA
from bayesfilter.ssm import stable_ssm_target_signature


POSTERIOR_RECOMPOSITION_SCHEMA = (
    "bayesfilter.neutra.posterior_recomposition_admission.v1"
)
TYPED_TARGET_IDENTITY_SCHEMA = "bayesfilter.neutra.typed_target_identity.v1"
CPU_SAMPLE_PARTITION_SCHEMA = "bayesfilter.neutra.cpu_sample_partition.v1"
CAMPAIGN_ARCHIVE_SCHEMA = "bayesfilter.neutra.separate_sample_archive.v1"
CAMPAIGN_CELL_LEDGER_SCHEMA = "bayesfilter.neutra.campaign_cell_ledger.v1"

_RECOMPOSITION_ISSUER = object()
_TARGET_IDENTITY_ISSUER = object()
_HEX = frozenset("0123456789abcdef")
_FORWARD_STATES = (
    "TARGET_FROZEN",
    "VALUE_SCORE_ADMITTED",
    "POSTERIOR_IDENTITY_ADMITTED",
    "COMPARATOR_ADMITTED",
    "TRAINING_SCREENED",
    "TRAINING_ADMITTED",
    "NEUTRA_CONFIRMED",
)
_SIDE_EXIT_STATES = frozenset(
    {
        "TARGET_BLOCKED",
        "IMPLEMENTATION_BLOCKED",
        "FILTER_CANDIDATE_REJECTED",
        "COMPARATOR_BLOCKED",
        "SAMPLER_BLOCKED",
        "EVIDENCE_BLOCKED",
        "CELL_CANDIDATE_REJECTED",
    }
)


class NeuTraCampaignError(RuntimeError):
    """Raised when a campaign boundary cannot be established honestly."""


@dataclass(frozen=True)
class PosteriorRecompositionAdmission:
    """Repository-issued proof that independent terms match one final target."""

    schema: str
    mathematical_target_signature: str
    adapter_signature: str
    dtype: str
    parameter_dim: int
    point_count: int
    points_sha256: str
    component_identities: tuple[Mapping[str, Any], ...]
    maximum_absolute_value_error: float
    maximum_absolute_score_error: float
    value_tolerance: float
    score_tolerance: float
    admission_signature: str
    passed: bool
    _adapter: Any = field(repr=False, compare=False)
    _issuer: object = field(repr=False, compare=False)

    def payload(self) -> Mapping[str, Any]:
        return {
            "schema": self.schema,
            "mathematical_target_signature": self.mathematical_target_signature,
            "adapter_signature": self.adapter_signature,
            "dtype": self.dtype,
            "parameter_dim": self.parameter_dim,
            "point_count": self.point_count,
            "points_sha256": self.points_sha256,
            "component_identities": self.component_identities,
            "maximum_absolute_value_error": self.maximum_absolute_value_error,
            "maximum_absolute_score_error": self.maximum_absolute_score_error,
            "value_tolerance": self.value_tolerance,
            "score_tolerance": self.score_tolerance,
            "admission_signature": self.admission_signature,
            "passed": self.passed,
        }


@dataclass(frozen=True)
class TypedNeuTraTargetIdentity:
    """Non-overridable identity for one complete campaign posterior surface."""

    schema: str
    program_id: str
    scope_kind: str
    scope_id: str
    mathematical_target_signature: str
    target_signature: str
    adapter_signature: str
    batch_execution_surface: Mapping[str, Any]
    posterior_execution_surface: Mapping[str, Any]
    status_execution_surface: Mapping[str, Any]
    dtype: str
    parameter_dim: int
    recomposition_signature: str
    registry_artifact_sha256: str | None
    nonclaims: tuple[str, ...]
    _adapter: Any = field(repr=False, compare=False)
    _posterior_function: Callable[..., Any] = field(repr=False, compare=False)
    _status_function: Callable[..., Any] = field(repr=False, compare=False)
    _recomposition: PosteriorRecompositionAdmission = field(
        repr=False, compare=False
    )
    _issuer: object = field(repr=False, compare=False)

    def identity_payload(self) -> Mapping[str, Any]:
        return {
            "schema": self.schema,
            "program_id": self.program_id,
            "scope_kind": self.scope_kind,
            "scope_id": self.scope_id,
            "mathematical_target_signature": self.mathematical_target_signature,
            "adapter_signature": self.adapter_signature,
            "batch_execution_surface": self.batch_execution_surface,
            "posterior_execution_surface": self.posterior_execution_surface,
            "status_execution_surface": self.status_execution_surface,
            "dtype": self.dtype,
            "parameter_dim": self.parameter_dim,
            "recomposition_signature": self.recomposition_signature,
            "registry_artifact_sha256": self.registry_artifact_sha256,
            "nonclaims": self.nonclaims,
        }

    def payload(self) -> Mapping[str, Any]:
        return {**self.identity_payload(), "target_signature": self.target_signature}


@dataclass(frozen=True)
class CPUSampleBatchSpec:
    """Fixed batch identity assigned to a CPU worker without changing its seed."""

    batch_index: int
    start_index: int
    sample_count: int
    worker_index: int
    seed: tuple[int, int]
    schema: str = CPU_SAMPLE_PARTITION_SCHEMA

    def payload(self) -> Mapping[str, Any]:
        return {
            "schema": self.schema,
            "batch_index": self.batch_index,
            "start_index": self.start_index,
            "sample_count": self.sample_count,
            "worker_index": self.worker_index,
            "seed": self.seed,
        }


def admit_independent_posterior_recomposition(
    *,
    adapter: Any,
    points: Any,
    prior_value_score_fn: Callable[[tf.Tensor], tuple[Any, Any]],
    likelihood_value_score_fn: Callable[[tf.Tensor], tuple[Any, Any]],
    jacobian_value_score_fn: Callable[[tf.Tensor], tuple[Any, Any]],
    value_tolerance: float = 1.0e-10,
    score_tolerance: float = 1.0e-10,
) -> PosteriorRecompositionAdmission:
    """Compare a final adapter with separately callable posterior terms."""

    contract = getattr(adapter, "contract", None)
    if contract is None:
        raise NeuTraCampaignError("posterior recomposition requires an SSM contract")
    mathematical_signature = _bare_sha256(
        stable_ssm_target_signature(contract), "mathematical target signature"
    )
    adapter_signature = _adapter_signature(adapter)
    dtype = tf.as_dtype(getattr(adapter, "dtype", tf.float64))
    theta = tf.cast(tf.convert_to_tensor(points), dtype)
    if theta.shape.rank != 2:
        raise NeuTraCampaignError("recomposition points must have shape [point, parameter]")
    point_count, parameter_dim = theta.shape.as_list()
    if point_count is None or point_count < 2 or parameter_dim is None:
        raise NeuTraCampaignError(
            "recomposition requires at least two points and a static parameter dimension"
        )
    components = (
        ("prior", prior_value_score_fn),
        ("filter_likelihood", likelihood_value_score_fn),
        ("unconstraining_jacobian", jacobian_value_score_fn),
    )
    forbidden_final_assemblers = _production_final_assembler_functions(adapter)
    identities = tuple(
        _independent_component_identity(
            adapter,
            label,
            function,
            forbidden_functions=forbidden_final_assemblers,
        )
        for label, function in components
    )
    if len({row["source_sha256"] for row in identities}) != len(identities):
        raise NeuTraCampaignError("posterior recomposition components must be distinct")

    production_value, production_score = _invoke_checked_value_score(
        adapter.log_prob_and_grad,
        theta=theta,
        label="production posterior",
    )
    component_results = tuple(
        _invoke_checked_value_score(function, theta=theta, label=label)
        for label, function in components
    )
    recomposed_value = tf.add_n(tuple(row[0] for row in component_results))
    recomposed_score = tf.add_n(tuple(row[1] for row in component_results))
    value_error = float(
        tf.reduce_max(tf.abs(production_value - recomposed_value)).numpy()
    )
    score_error = float(
        tf.reduce_max(tf.abs(production_score - recomposed_score)).numpy()
    )
    value_limit = float(value_tolerance)
    score_limit = float(score_tolerance)
    if value_limit < 0.0 or score_limit < 0.0:
        raise NeuTraCampaignError("recomposition tolerances must be nonnegative")
    if value_error > value_limit or score_error > score_limit:
        raise NeuTraCampaignError(
            "independent posterior recomposition mismatch: "
            f"value={value_error:g}, score={score_error:g}"
        )
    points_sha256 = hashlib.sha256(
        bytes(tf.io.serialize_tensor(theta).numpy())
    ).hexdigest()
    signature_payload = {
        "schema": POSTERIOR_RECOMPOSITION_SCHEMA,
        "mathematical_target_signature": mathematical_signature,
        "adapter_signature": adapter_signature,
        "dtype": dtype.name,
        "parameter_dim": int(parameter_dim),
        "point_count": int(point_count),
        "points_sha256": points_sha256,
        "component_identities": identities,
        "maximum_absolute_value_error": value_error,
        "maximum_absolute_score_error": score_error,
        "value_tolerance": value_limit,
        "score_tolerance": score_limit,
        "passed": True,
    }
    return PosteriorRecompositionAdmission(
        **signature_payload,
        admission_signature=_stable_hash(signature_payload),
        _adapter=adapter,
        _issuer=_RECOMPOSITION_ISSUER,
    )


def issue_typed_neutra_target_identity(
    *,
    program_id: str,
    scope_kind: str,
    scope_id: str,
    adapter: Any,
    recomposition: PosteriorRecompositionAdmission,
    registry_row: Mapping[str, Any] | None = None,
    registry_artifact_sha256: str | None = None,
) -> TypedNeuTraTargetIdentity:
    """Issue one campaign identity from inspected code and admitted mathematics."""

    _validate_recomposition(recomposition, adapter)
    kind = str(scope_kind)
    identifier = _nonblank(scope_id, "scope_id")
    if kind not in {"synthetic_canary", "model_cell"}:
        raise NeuTraCampaignError("scope_kind must be synthetic_canary or model_cell")
    mathematical_signature = recomposition.mathematical_target_signature
    if kind == "model_cell":
        if registry_row is None:
            raise NeuTraCampaignError("model-cell identity requires a registry row")
        _validate_model_registry_row(
            registry_row,
            scope_id=identifier,
            mathematical_target_signature=mathematical_signature,
        )
        registry_hash = _bare_sha256(
            registry_artifact_sha256, "registry artifact SHA-256"
        )
    else:
        if registry_row is not None or registry_artifact_sha256 is not None:
            raise NeuTraCampaignError(
                "synthetic canary identity must not claim model registry authority"
            )
        registry_hash = None

    provisional = bind_batch_native_neutra_target(
        adapter, target_signature=mathematical_signature
    )
    execution_surface = _binding_execution_surface(provisional)
    posterior_surface, posterior_function = _posterior_execution_surface(adapter)
    status_surface, status_function = _bound_method_execution_surface(
        adapter,
        method_name="target_status_telemetry",
        label="target status telemetry",
    )
    identity_payload = {
        "schema": TYPED_TARGET_IDENTITY_SCHEMA,
        "program_id": _nonblank(program_id, "program_id"),
        "scope_kind": kind,
        "scope_id": identifier,
        "mathematical_target_signature": mathematical_signature,
        "adapter_signature": recomposition.adapter_signature,
        "batch_execution_surface": execution_surface,
        "posterior_execution_surface": posterior_surface,
        "status_execution_surface": status_surface,
        "dtype": recomposition.dtype,
        "parameter_dim": recomposition.parameter_dim,
        "recomposition_signature": recomposition.admission_signature,
        "registry_artifact_sha256": registry_hash,
        "nonclaims": (
            "typed target identity and execution binding only",
            "no HMC convergence or posterior recovery claim",
            "no NeuTra training quality or production readiness claim",
        ),
    }
    target_signature = _stable_hash(identity_payload)
    identity = TypedNeuTraTargetIdentity(
        **identity_payload,
        target_signature=target_signature,
        _adapter=adapter,
        _posterior_function=posterior_function,
        _status_function=status_function,
        _recomposition=recomposition,
        _issuer=_TARGET_IDENTITY_ISSUER,
    )
    require_typed_neutra_target(identity, adapter=adapter)
    return identity


def require_typed_neutra_target(
    identity: TypedNeuTraTargetIdentity,
    *,
    adapter: Any,
) -> NeuTraBatchTargetBinding:
    """Revalidate issuer, target fields, adapter code, and dependency closure."""

    if not isinstance(identity, TypedNeuTraTargetIdentity):
        raise NeuTraCampaignError("campaign target identity must be repository-issued")
    if identity._issuer is not _TARGET_IDENTITY_ISSUER:
        raise NeuTraCampaignError("campaign target identity has an invalid issuer")
    if identity.schema != TYPED_TARGET_IDENTITY_SCHEMA:
        raise NeuTraCampaignError("campaign target identity schema mismatch")
    if identity._adapter is not adapter:
        raise NeuTraCampaignError("campaign target identity belongs to another adapter")
    _validate_recomposition(identity._recomposition, adapter)
    if identity.recomposition_signature != identity._recomposition.admission_signature:
        raise NeuTraCampaignError("campaign recomposition signature mismatch")
    expected = _stable_hash(identity.identity_payload())
    if identity.target_signature != expected:
        raise NeuTraCampaignError("campaign target signature mismatch")
    current = bind_batch_native_neutra_target(
        adapter, target_signature=identity.target_signature
    )
    if _binding_execution_surface(current) != identity.batch_execution_surface:
        raise NeuTraCampaignError("campaign batch execution surface changed")
    posterior_surface, posterior_function = _posterior_execution_surface(adapter)
    if posterior_function is not identity._posterior_function:
        raise NeuTraCampaignError("campaign posterior value/score callable changed")
    if posterior_surface != identity.posterior_execution_surface:
        raise NeuTraCampaignError("campaign posterior execution surface changed")
    status_surface, status_function = _bound_method_execution_surface(
        adapter,
        method_name="target_status_telemetry",
        label="target status telemetry",
    )
    if status_function is not identity._status_function:
        raise NeuTraCampaignError("campaign target status callable changed")
    if status_surface != identity.status_execution_surface:
        raise NeuTraCampaignError("campaign target status execution surface changed")
    return current


def train_campaign_neutra(
    *,
    identity: TypedNeuTraTargetIdentity,
    adapter: Any,
    config: PlainDenseIAFTrainingConfig,
    freeze_transport_id: str,
    gpu_memory_policy: Mapping[str, Any] | None = None,
) -> PlainDenseIAFTrainingResult:
    """Run the existing trainer only for the exact repository-issued identity."""

    require_typed_neutra_target(identity, adapter=adapter)
    if config.target_signature != identity.target_signature:
        raise NeuTraCampaignError("training config target signature mismatch")
    if config.dimension != identity.parameter_dim:
        raise NeuTraCampaignError("training config target dimension mismatch")
    if config.output_dir.exists():
        raise NeuTraCampaignError(
            "campaign training requires a fresh output directory"
        )
    if config.require_gpu:
        if not isinstance(gpu_memory_policy, Mapping):
            raise NeuTraCampaignError(
                "GPU campaign training requires verified memory-growth metadata"
            )
        if (
            gpu_memory_policy.get("schema") != TF_GPU_MEMORY_POLICY_SCHEMA
            or gpu_memory_policy.get("mode") != "memory_growth"
            or gpu_memory_policy.get("all_physical_devices_memory_growth") is not True
            or gpu_memory_policy.get("configured_before_logical_device_initialization")
            is not True
        ):
            raise NeuTraCampaignError(
                "GPU campaign training memory-growth policy is not verified"
            )
    return train_plain_dense_iaf(
        adapter=adapter,
        config=config,
        freeze_transport_id=_nonblank(freeze_transport_id, "freeze_transport_id"),
    )


def load_campaign_neutra_transport(
    *,
    identity: TypedNeuTraTargetIdentity,
    adapter: Any,
    payload: Mapping[str, Any],
) -> LoadedFrozenNeuTraArtifact:
    """Load a frozen transport only under its exact campaign target identity."""

    require_typed_neutra_target(identity, adapter=adapter)
    loaded = load_frozen_neutra_artifact(
        payload, expected_target_signature=identity.target_signature
    )
    if loaded.manifest.dimension != identity.parameter_dim:
        raise NeuTraCampaignError("frozen transport target dimension mismatch")
    return loaded


def campaign_fixed_transport_adapter(
    *,
    identity: TypedNeuTraTargetIdentity,
    adapter: Any,
    loaded_artifact: LoadedFrozenNeuTraArtifact,
) -> FixedTransportValueScoreAdapter:
    """Build the shared explicit-score NeuTra HMC target after identity checks."""

    require_typed_neutra_target(identity, adapter=adapter)
    if loaded_artifact.manifest.target_signature != identity.target_signature:
        raise NeuTraCampaignError("cross-target frozen transport is forbidden")
    return FixedTransportValueScoreAdapter(
        base_adapter=adapter,
        transport=loaded_artifact.transport,
        target_scope=f"{identity.scope_id}:fixed-neutra",
        evidence_path="bayesfilter/inference/neutra_campaign.py",
        xla_hmc_ready=True,
        full_chain_xla_diagnostic_ready=False,
        require_batch_native=True,
    )


def run_campaign_plain_hmc(
    *,
    identity: TypedNeuTraTargetIdentity,
    adapter: Any,
    initial_state: Any,
    config: BatchedHMCConfig,
) -> Mapping[str, Any]:
    """Run the shared plain-HMC smoke only after typed identity replay."""

    require_typed_neutra_target(identity, adapter=adapter)
    return run_batched_hmc(
        adapter=adapter, initial_state=initial_state, config=config
    )


def run_campaign_neutra_hmc(
    *,
    identity: TypedNeuTraTargetIdentity,
    adapter: Any,
    loaded_artifact: LoadedFrozenNeuTraArtifact,
    initial_state: Any,
    parameter_names: Sequence[str],
    config: SequentialNeuTraHMCConfig,
    archive_callback: Callable[..., Mapping[str, Any]] | None = None,
    retained_diagnostic_fn: Callable[[tf.Tensor], Mapping[str, Any]] | None = None,
) -> Mapping[str, Any]:
    """Run the existing sequential controller on the bound frozen transport."""

    transformed = campaign_fixed_transport_adapter(
        identity=identity, adapter=adapter, loaded_artifact=loaded_artifact
    )

    def model_transform(samples: tf.Tensor) -> tf.Tensor:
        values = tf.convert_to_tensor(samples, tf.float64)
        shape = tf.shape(values)
        flat = tf.reshape(values, (-1, identity.parameter_dim))
        mapped = loaded_artifact.transport.forward_batch(flat)
        return tf.reshape(mapped, shape)

    return run_sequential_neutra_hmc(
        adapter=transformed,
        initial_state=initial_state,
        model_transform=model_transform,
        parameter_names=parameter_names,
        config=config,
        archive_callback=archive_callback,
        retained_diagnostic_fn=retained_diagnostic_fn,
    )


def deterministic_cpu_sample_partitions(
    *,
    root_seed: tuple[int, int],
    sample_count: int,
    batch_size: int,
    worker_count: int,
    domain: str,
) -> tuple[CPUSampleBatchSpec, ...]:
    """Assign fixed stateless batches to workers without changing batch seeds."""

    seed = tuple(int(item) for item in root_seed)
    if len(seed) != 2:
        raise NeuTraCampaignError("root_seed must contain exactly two integers")
    total = int(sample_count)
    size = int(batch_size)
    workers = int(worker_count)
    if total <= 0 or size <= 0 or workers <= 0:
        raise NeuTraCampaignError(
            "sample_count, batch_size, and worker_count must be positive"
        )
    domain_text = _nonblank(domain, "domain")
    domain_word = int.from_bytes(
        hashlib.sha256(domain_text.encode("utf-8")).digest()[:4], "big"
    ) & 0x7FFFFFFF
    rows = []
    batch_index = 0
    for start in range(0, total, size):
        active = min(size, total - start)
        rows.append(
            CPUSampleBatchSpec(
                batch_index=batch_index,
                start_index=start,
                sample_count=active,
                worker_index=batch_index % workers,
                seed=(seed[0] ^ domain_word, seed[1] + 1009 * (batch_index + 1)),
            )
        )
        batch_index += 1
    return tuple(rows)


def generate_cpu_sample_batch(
    spec: CPUSampleBatchSpec,
    *,
    dimension: int,
    dtype: Any = tf.float64,
) -> tf.Tensor:
    """Generate one fixed batch in a single TensorFlow stateless operation."""

    if not isinstance(spec, CPUSampleBatchSpec):
        raise TypeError("spec must be a CPUSampleBatchSpec")
    dim = int(dimension)
    if dim <= 0:
        raise NeuTraCampaignError("sample dimension must be positive")
    with tf.device("/CPU:0"):
        return tf.random.stateless_normal(
            (spec.sample_count, dim),
            seed=tf.constant(spec.seed, tf.int32),
            dtype=tf.as_dtype(dtype),
        )


class SeparateCampaignArchive:
    """Write warm-up and retained tensors into disjoint immutable paths."""

    def __init__(
        self,
        *,
        output_root: str | Path,
        identity: TypedNeuTraTargetIdentity,
        adapter: Any,
    ) -> None:
        require_typed_neutra_target(identity, adapter=adapter)
        self.output_root = Path(output_root)
        self.identity = identity

    def __call__(
        self,
        *,
        stage: str,
        chunk_index: int | None,
        latent_samples: Any,
        model_samples: Any,
        seed: tuple[int, int] | None,
        cumulative: bool,
    ) -> Mapping[str, Any]:
        if stage not in {"warmup", "retained"}:
            raise NeuTraCampaignError("archive stage must be warmup or retained")
        label = "cumulative" if cumulative else f"chunk-{int(chunk_index):04d}"
        destination = self.output_root / stage / label
        if destination.exists():
            raise NeuTraCampaignError(f"archive destination already exists: {destination}")
        destination.mkdir(parents=True)
        latent = tf.convert_to_tensor(latent_samples, tf.float64)
        model = tf.convert_to_tensor(model_samples, tf.float64)
        if latent.shape != model.shape or latent.shape.rank != 3:
            raise NeuTraCampaignError(
                "archive tensors must share [draw, chain, parameter] shape"
            )
        latent_path = destination / "latent.tensor"
        model_path = destination / "model.tensor"
        tf.io.write_file(str(latent_path), tf.io.serialize_tensor(latent))
        tf.io.write_file(str(model_path), tf.io.serialize_tensor(model))
        metadata = {
            "schema": CAMPAIGN_ARCHIVE_SCHEMA,
            "stage": stage,
            "cumulative": bool(cumulative),
            "chunk_index": chunk_index,
            "seed": seed,
            "target_signature": self.identity.target_signature,
            "sample_shape": tuple(int(item) for item in latent.shape),
            "latent_path": str(latent_path),
            "model_path": str(model_path),
            "warmup_excluded_from_posterior": True,
        }
        atomic_write_json(destination / "metadata.json", metadata)
        return metadata


class CampaignCellLedger:
    """P1 state guard with recipe outcomes separate from cell rejection."""

    def __init__(
        self,
        registry: Mapping[str, Any],
        *,
        required_candidate_families: Sequence[str] = ("plain_dense_iaf", "enhanced"),
        event_path: str | Path | None = None,
    ) -> None:
        cells = registry.get("cells")
        if not isinstance(cells, Sequence) or isinstance(cells, (str, bytes)):
            raise NeuTraCampaignError("registry cells must be a sequence")
        self._states = {str(row["cell_id"]): str(row["state"]) for row in cells}
        if len(self._states) != len(cells):
            raise NeuTraCampaignError("registry contains duplicate cell IDs")
        self._families = tuple(_nonblank(item, "candidate family") for item in required_candidate_families)
        self._recipes: dict[str, dict[str, Mapping[str, Any]]] = {
            cell_id: {} for cell_id in self._states
        }
        self._events: list[Mapping[str, Any]] = []
        self._event_path = None if event_path is None else Path(event_path)

    def transition(
        self,
        *,
        cell_id: str,
        new_state: str,
        evidence_path: str,
        target_identity: TypedNeuTraTargetIdentity | None = None,
    ) -> Mapping[str, Any]:
        cell = self._require_cell(cell_id)
        current = self._states[cell]
        target = str(new_state)
        if current == "TARGET_BLOCKED":
            raise NeuTraCampaignError(
                "P0-blocked cells require a later target-repair phase before transition"
            )
        if target == "RECIPE_REJECTED":
            raise NeuTraCampaignError(
                "recipe rejection is a candidate-family outcome, not a cell transition"
            )
        if target in _FORWARD_STATES:
            expected_index = _FORWARD_STATES.index(current) + 1
            if expected_index >= len(_FORWARD_STATES) or _FORWARD_STATES[expected_index] != target:
                raise NeuTraCampaignError("cell forward transition skips a required gate")
            if target_identity is None or target_identity.scope_id != cell:
                raise NeuTraCampaignError("cell forward transition requires its typed identity")
        elif target not in _SIDE_EXIT_STATES:
            raise NeuTraCampaignError(f"unsupported campaign cell state: {target}")
        event = {
            "cell_id": cell,
            "from_state": current,
            "to_state": target,
            "evidence_path": _nonblank(evidence_path, "evidence_path"),
            "target_signature": (
                None if target_identity is None else target_identity.target_signature
            ),
        }
        self._states[cell] = target
        self._record_event("cell_transition", event)
        return event

    def record_recipe_rejection(
        self,
        *,
        cell_id: str,
        family: str,
        recipe_id: str,
        evidence_path: str,
    ) -> Mapping[str, Any]:
        cell = self._require_cell(cell_id)
        if self._states[cell] == "TARGET_BLOCKED":
            raise NeuTraCampaignError(
                "P0-blocked cells cannot execute or reject training recipes in P1"
            )
        family_name = _nonblank(family, "family")
        if family_name not in self._families:
            raise NeuTraCampaignError("recipe family is outside the frozen candidate set")
        record = {
            "status": "RECIPE_REJECTED",
            "family": family_name,
            "recipe_id": _nonblank(recipe_id, "recipe_id"),
            "evidence_path": _nonblank(evidence_path, "evidence_path"),
        }
        self._recipes[cell][family_name] = record
        self._record_event(
            "recipe_outcome", {"cell_id": cell, **record}
        )
        return record

    def reject_cell_candidates(
        self, *, cell_id: str, evidence_path: str
    ) -> Mapping[str, Any]:
        cell = self._require_cell(cell_id)
        if self._states[cell] == "TARGET_BLOCKED":
            raise NeuTraCampaignError(
                "P0-blocked cells cannot be classified as candidate-rejected"
            )
        tried = self._recipes[cell]
        missing = tuple(family for family in self._families if family not in tried)
        if missing or any(row["status"] != "RECIPE_REJECTED" for row in tried.values()):
            raise NeuTraCampaignError(
                f"cell rejection requires all candidate families rejected; missing={missing}"
            )
        current = self._states[cell]
        event = {
            "cell_id": cell,
            "from_state": current,
            "to_state": "CELL_CANDIDATE_REJECTED",
            "evidence_path": _nonblank(evidence_path, "evidence_path"),
            "candidate_families": tuple(sorted(tried)),
        }
        self._states[cell] = "CELL_CANDIDATE_REJECTED"
        self._record_event("cell_transition", event)
        return event

    def payload(self) -> Mapping[str, Any]:
        return {
            "schema": CAMPAIGN_CELL_LEDGER_SCHEMA,
            "states": dict(sorted(self._states.items())),
            "required_candidate_families": self._families,
            "recipe_outcomes": {
                key: dict(sorted(value.items()))
                for key, value in sorted(self._recipes.items())
            },
            "events": tuple(self._events),
        }

    def _require_cell(self, cell_id: str) -> str:
        cell = str(cell_id)
        if cell not in self._states:
            raise NeuTraCampaignError(f"unknown campaign cell: {cell}")
        return cell

    def _record_event(self, event_type: str, payload: Mapping[str, Any]) -> None:
        event = {
            "schema": "bayesfilter.neutra.campaign_cell_event.v1",
            "event_type": event_type,
            **dict(payload),
        }
        self._events.append(event)
        if self._event_path is not None:
            append_jsonl(self._event_path, event)


def load_validated_p0_registry(
    registry_path: str | Path,
    *,
    expected_file_sha256: str,
) -> Mapping[str, Any]:
    """Load the exact P0 registry and reject drift or issued model signatures."""

    path = Path(registry_path)
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed != _bare_sha256(expected_file_sha256, "registry file SHA-256"):
        raise NeuTraCampaignError("P0 target registry file hash mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    cells = payload.get("cells")
    if not isinstance(cells, list) or len(cells) != 11:
        raise NeuTraCampaignError("P0 registry must contain exactly eleven cells")
    identifiers = tuple(str(row.get("cell_id")) for row in cells)
    if len(set(identifiers)) != 11:
        raise NeuTraCampaignError("P0 registry cell IDs must be unique")
    for row in cells:
        if row.get("state") != "TARGET_BLOCKED":
            raise NeuTraCampaignError("P1 expects every P0 model cell to remain blocked")
        if row.get("target_signature") is not None:
            raise NeuTraCampaignError("P1 forbids a model target signature in this registry")
        if row.get("target_signature_status") != (
            "NOT_ISSUED_INCOMPLETE_POSTERIOR_CONTRACT"
        ):
            raise NeuTraCampaignError("P0 target-signature status is not fail-closed")
    return payload


def _validate_model_registry_row(
    row: Mapping[str, Any],
    *,
    scope_id: str,
    mathematical_target_signature: str,
) -> None:
    if row.get("cell_id") != scope_id:
        raise NeuTraCampaignError("model registry row cell mismatch")
    if row.get("state") not in {
        "VALUE_SCORE_ADMITTED",
        "POSTERIOR_IDENTITY_ADMITTED",
    }:
        raise NeuTraCampaignError("model cell is not eligible for typed identity issuance")
    if row.get("target_signature") != mathematical_target_signature:
        raise NeuTraCampaignError("model registry mathematical target mismatch")


def _validate_recomposition(
    recomposition: PosteriorRecompositionAdmission, adapter: Any
) -> None:
    if not isinstance(recomposition, PosteriorRecompositionAdmission):
        raise NeuTraCampaignError("posterior recomposition must be repository-issued")
    if recomposition._issuer is not _RECOMPOSITION_ISSUER:
        raise NeuTraCampaignError("posterior recomposition issuer mismatch")
    if recomposition._adapter is not adapter or not recomposition.passed:
        raise NeuTraCampaignError("posterior recomposition belongs to another target")
    payload = dict(recomposition.payload())
    observed = payload.pop("admission_signature")
    if observed != _stable_hash(payload):
        raise NeuTraCampaignError("posterior recomposition signature mismatch")
    current_math = _bare_sha256(
        stable_ssm_target_signature(adapter.contract), "mathematical target signature"
    )
    if current_math != recomposition.mathematical_target_signature:
        raise NeuTraCampaignError("posterior mathematical contract changed")
    if _adapter_signature(adapter) != recomposition.adapter_signature:
        raise NeuTraCampaignError("posterior adapter signature changed")


def _independent_component_identity(
    adapter: Any,
    label: str,
    function: Callable[..., Any],
    forbidden_functions: frozenset[Callable[..., Any]],
) -> Mapping[str, Any]:
    if not callable(function):
        raise NeuTraCampaignError(f"{label} recomposition component must be callable")
    if inspect.ismethod(function) and function.__self__ is adapter:
        raise NeuTraCampaignError(
            "independent recomposition cannot reuse a production adapter method"
        )
    raw_function = function.__func__ if inspect.ismethod(function) else function
    if raw_function in forbidden_functions:
        raise NeuTraCampaignError(
            "independent recomposition cannot reuse a production final assembler"
        )
    try:
        source = textwrap.dedent(inspect.getsource(function))
    except (OSError, TypeError) as exc:
        raise NeuTraCampaignError(
            f"{label} recomposition component source must be inspectable"
        ) from exc
    if ".log_prob_and_grad" in source or ".neutra_batch_log_prob" in source:
        raise NeuTraCampaignError(
            "independent recomposition cannot call the production final assembler"
        )
    return {
        "role": label,
        "module": str(getattr(function, "__module__", "")),
        "qualname": str(getattr(function, "__qualname__", "")),
        "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
    }


def _production_final_assembler_functions(
    adapter: Any,
) -> frozenset[Callable[..., Any]]:
    method = getattr(adapter, "log_prob_and_grad", None)
    if not inspect.ismethod(method) or method.__self__ is not adapter:
        raise NeuTraCampaignError(
            "production posterior must expose an adapter-bound log_prob_and_grad"
        )
    function = method.__func__
    source = _inspectable_source(function, "production final assembler")
    tree = ast.parse(source)
    dependencies = {
        function,
        *(
            dependency
            for name in {
                node.func.id
                for node in ast.walk(tree)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            }
            if callable(dependency := function.__globals__.get(name))
            and not inspect.isclass(dependency)
        ),
    }
    return frozenset(dependencies)


def _checked_value_score(
    result: Any, *, theta: tf.Tensor, label: str
) -> tuple[tf.Tensor, tf.Tensor]:
    if not isinstance(result, tuple) or len(result) != 2:
        raise NeuTraCampaignError(f"{label} must return (value, score)")
    value = tf.convert_to_tensor(result[0])
    score = tf.convert_to_tensor(result[1])
    if value.dtype != theta.dtype or score.dtype != theta.dtype:
        raise NeuTraCampaignError(
            f"{label} dtype mismatch: expected {theta.dtype.name}, "
            f"observed value={value.dtype.name}, score={score.dtype.name}"
        )
    if value.shape != theta.shape[:-1] or score.shape != theta.shape:
        raise NeuTraCampaignError(f"{label} value/score shape mismatch")
    if not bool(tf.reduce_all(tf.math.is_finite(value)).numpy()) or not bool(
        tf.reduce_all(tf.math.is_finite(score)).numpy()
    ):
        raise NeuTraCampaignError(f"{label} value/score must be finite")
    return value, score


def _invoke_checked_value_score(
    function: Callable[[tf.Tensor], Any],
    *,
    theta: tf.Tensor,
    label: str,
) -> tuple[tf.Tensor, tf.Tensor]:
    try:
        result = function(theta)
    except (TypeError, ValueError, tf.errors.OpError) as exc:
        raise NeuTraCampaignError(
            f"{label} dtype mismatch or incompatible implementation: "
            f"declared {theta.dtype.name}"
        ) from exc
    return _checked_value_score(result, theta=theta, label=label)


def _binding_execution_surface(
    binding: NeuTraBatchTargetBinding,
) -> Mapping[str, Any]:
    payload = dict(binding.payload())
    payload.pop("target_signature", None)
    return payload


def _posterior_execution_surface(
    adapter: Any,
) -> tuple[Mapping[str, Any], Callable[..., Any]]:
    return _bound_method_execution_surface(
        adapter,
        method_name="log_prob_and_grad",
        label="posterior value/score",
    )


def _bound_method_execution_surface(
    adapter: Any,
    *,
    method_name: str,
    label: str,
) -> tuple[Mapping[str, Any], Callable[..., Any]]:
    method = getattr(adapter, method_name, None)
    if not inspect.ismethod(method) or method.__self__ is not adapter:
        raise NeuTraCampaignError(
            f"campaign {label} must be an adapter-bound instance method"
        )
    function = method.__func__
    source = _inspectable_source(function, f"{label} method")
    tree = ast.parse(source)
    called_names = tuple(
        sorted(
            {
                node.func.id
                for node in ast.walk(tree)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            }
        )
    )
    dependencies = tuple(
        sorted(
            (
                name,
                function.__globals__[name],
            )
            for name in called_names
            if callable(function.__globals__.get(name))
            and not inspect.isclass(function.__globals__.get(name))
            and str(
                getattr(function.__globals__.get(name), "__module__", "")
            ).startswith("bayesfilter.")
        )
    )
    callable_sources = tuple(
        {
            "global_name": name,
            "module": str(dependency.__module__),
            "qualname": str(dependency.__qualname__),
            "source_sha256": hashlib.sha256(
                _inspectable_source(
                    dependency, f"{label} dependency {name}"
                ).encode("utf-8")
            ).hexdigest(),
        }
        for name, dependency in dependencies
    )
    module_names = tuple(
        sorted({str(dependency.__module__) for _name, dependency in dependencies})
    )
    module_sources = []
    for module_name in module_names:
        module = sys.modules.get(module_name)
        if module is None:
            raise NeuTraCampaignError(f"{label} dependency module is not loaded: {module_name}")
        module_source = _inspectable_source(
            module, f"{label} dependency module {module_name}"
        )
        module_sources.append(
            {
                "module": module_name,
                "source_sha256": hashlib.sha256(
                    module_source.encode("utf-8")
                ).hexdigest(),
            }
        )
    return (
        {
            "method_name": method_name,
            "callable_module": str(function.__module__),
            "callable_qualname": str(function.__qualname__),
            "callable_source_sha256": hashlib.sha256(
                source.encode("utf-8")
            ).hexdigest(),
            "dependency_callable_sources": callable_sources,
            "dependency_module_sources": tuple(module_sources),
        },
        function,
    )


def _inspectable_source(value: Any, label: str) -> str:
    try:
        return textwrap.dedent(inspect.getsource(value))
    except (OSError, TypeError) as exc:
        raise NeuTraCampaignError(f"{label} source must be inspectable") from exc


def _adapter_signature(adapter: Any) -> str:
    function = getattr(adapter, "adapter_signature", None)
    if not callable(function):
        raise NeuTraCampaignError("campaign adapter must expose adapter_signature()")
    return _bare_sha256(function(), "adapter signature")


def _bare_sha256(value: Any, label: str) -> str:
    text = str(value)
    if text.startswith("sha256:"):
        text = text.split(":", 1)[1]
    if len(text) != 64 or any(character not in _HEX for character in text):
        raise NeuTraCampaignError(f"{label} must be a lowercase SHA-256 digest")
    return text


def _nonblank(value: Any, label: str) -> str:
    text = str(value)
    if not text.strip() or text != text.strip():
        raise NeuTraCampaignError(f"{label} must be a nonblank trimmed string")
    return text


def _stable_hash(payload: Mapping[str, Any]) -> str:
    blob = json.dumps(
        _json_ready(payload), sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"unsupported campaign identity value: {type(value).__name__}")
