"""Thin identity, supervision, tuning, and dry-run helpers for neural-force HMC."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, NamedTuple

import tensorflow as tf

from bayesfilter.inference.neural_force_hmc import FrozenTargetPotential
from bayesfilter.inference.neural_force_training import FrozenScalarResidualForce
from bayesfilter.inference.neutra_artifacts import load_frozen_neutra_artifact


class NeuralForceCampaignError(RuntimeError):
    """Raised when a campaign identity or evidence boundary fails closed."""


class NeuralForceSupervision(NamedTuple):
    positions: tf.Tensor
    potentials: tf.Tensor
    forces: tf.Tensor


@dataclass(frozen=True)
class NeuralForceTargetBinding:
    """Complete transformed potential bound to one target and transport."""

    adapter: Any
    endpoint_potential_function: Callable[[tf.Tensor], tf.Tensor]
    target_signature: str
    transport_signature: str
    dimension: int

    def __post_init__(self) -> None:
        for value, name in (
            (self.target_signature, "target_signature"),
            (self.transport_signature, "transport_signature"),
        ):
            text = str(value)
            if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
                raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
        if int(self.dimension) <= 0:
            raise ValueError("dimension must be positive")
        if not callable(getattr(self.adapter, "log_prob_and_grad_batch", None)):
            raise NeuralForceCampaignError(
                "transformed adapter must expose batch-native log_prob_and_grad_batch"
            )

    def potential(self, position: tf.Tensor) -> tf.Tensor:
        return tf.convert_to_tensor(self.endpoint_potential_function(position), tf.float64)

    def potential_and_force(self, position: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        value, score = self.adapter.log_prob_and_grad_batch(position)
        return -tf.convert_to_tensor(value, tf.float64), -tf.convert_to_tensor(score, tf.float64)

    def hmc_target(self) -> FrozenTargetPotential:
        return FrozenTargetPotential(
            function=self.potential,
            identity=_stable_hash(
                {
                    "schema": "bayesfilter.neural_force_target_binding.v1",
                    "target_signature": self.target_signature,
                    "transport_signature": self.transport_signature,
                    "dimension": self.dimension,
                    "potential": "negative_complete_transformed_log_density",
                }
            ),
            coordinate_system="transformed",
            includes_chart_log_jacobian=True,
            deterministic=True,
        )


@dataclass(frozen=True)
class NeuralForceTuningCandidate:
    """Short-chain evidence used only to nominate a fixed kernel."""

    candidate_id: str
    step_size: float
    num_leapfrog_steps: int
    health_passed: bool
    modern_rhat: float
    maximum_absolute_delta_h: float
    acceptance_rate: float

    def __post_init__(self) -> None:
        if not str(self.candidate_id).strip():
            raise ValueError("candidate_id must be nonempty")
        if float(self.step_size) <= 0.0 or int(self.num_leapfrog_steps) <= 0:
            raise ValueError("candidate mechanics must be positive")


def bind_transformed_neural_force_target(
    *,
    adapter: Any,
    endpoint_potential_function: Callable[[tf.Tensor], tf.Tensor],
    target_signature: str,
    transport_signature: str,
    dimension: int,
) -> NeuralForceTargetBinding:
    """Bind the complete transformed log density, including chart log-Jacobian."""

    return NeuralForceTargetBinding(
        adapter=adapter,
        endpoint_potential_function=endpoint_potential_function,
        target_signature=str(target_signature),
        transport_signature=str(transport_signature),
        dimension=int(dimension),
    )


def validate_value_only_endpoint_parity(
    binding: NeuralForceTargetBinding,
    points: Any,
    *,
    absolute_tolerance: float = 1.0e-9,
) -> Mapping[str, Any]:
    """Prove the endpoint-only scalar equals the complete transformed target."""

    positions = tf.convert_to_tensor(points, tf.float64)
    if positions.shape.rank != 2 or positions.shape[-1] != binding.dimension:
        raise ValueError("parity points must have shape [point, binding.dimension]")
    endpoint = binding.potential(positions)
    reference_value, _reference_score = binding.adapter.log_prob_and_grad_batch(positions)
    reference = -tf.convert_to_tensor(reference_value, tf.float64)
    difference = tf.abs(endpoint - reference)
    maximum = tf.reduce_max(difference)
    passed = bool(
        tf.reduce_all(tf.math.is_finite(endpoint)).numpy()
        and tf.reduce_all(tf.math.is_finite(reference)).numpy()
        and (maximum <= tf.constant(absolute_tolerance, tf.float64)).numpy()
    )
    result = {
        "schema": "bayesfilter.neural_force_value_only_endpoint_parity.v1",
        "passed": passed,
        "point_count": int(positions.shape[0]),
        "maximum_absolute_error": float(maximum.numpy()),
        "absolute_tolerance": float(absolute_tolerance),
        "endpoint_uses_true_gradient": False,
    }
    if not passed:
        raise NeuralForceCampaignError("value-only endpoint does not match transformed target")
    return result


def generate_neural_force_supervision(
    binding: NeuralForceTargetBinding,
    positions: Any,
) -> NeuralForceSupervision:
    """Generate one batched TensorFlow supervision block without row loops."""

    if not isinstance(binding, NeuralForceTargetBinding):
        raise TypeError("binding must be a NeuralForceTargetBinding")
    value = tf.convert_to_tensor(positions, tf.float64)
    if value.shape.rank != 2 or value.shape[-1] != binding.dimension:
        raise ValueError("positions must have shape [row, binding.dimension]")
    potentials, forces = binding.potential_and_force(value)
    tf.debugging.assert_all_finite(potentials, "supervision potentials")
    tf.debugging.assert_all_finite(forces, "supervision forces")
    return NeuralForceSupervision(value, potentials, forces)


def require_force_target_binding(
    *,
    force: FrozenScalarResidualForce,
    target: NeuralForceTargetBinding,
) -> None:
    """Reject cross-target or cross-chart force substitution."""

    if force.target_signature != target.target_signature:
        raise NeuralForceCampaignError("frozen force target signature mismatch")
    if force.transport_signature != target.transport_signature:
        raise NeuralForceCampaignError("frozen force transport signature mismatch")


def select_health_aware_tuning_candidate(
    candidates: Sequence[NeuralForceTuningCandidate],
    *,
    rhat_max: float = 1.05,
    maximum_absolute_delta_h: float = 100.0,
) -> NeuralForceTuningCandidate:
    """Nominate only a healthy candidate; acceptance is a final tie-breaker."""

    viable = tuple(
        candidate
        for candidate in candidates
        if candidate.health_passed
        and candidate.modern_rhat <= rhat_max
        and candidate.maximum_absolute_delta_h <= maximum_absolute_delta_h
    )
    if not viable:
        raise NeuralForceCampaignError("no tuning candidate passes health and modern R-hat gates")
    return min(
        viable,
        key=lambda candidate: (
            candidate.modern_rhat,
            candidate.maximum_absolute_delta_h,
            abs(candidate.acceptance_rate - 0.8),
            candidate.candidate_id,
        ),
    )


def validate_disjoint_seed_domains(seed_domains: Mapping[str, Sequence[int]]) -> Mapping[str, Any]:
    """Require distinct stateless roots for training, tuning, warm-up, and retained data."""

    required = ("training_screen", "fresh_training", "tuning", "warmup", "retained")
    missing = tuple(name for name in required if name not in seed_domains)
    if missing:
        raise NeuralForceCampaignError(f"missing seed domains: {missing}")
    normalized = {name: tuple(int(value) for value in seed_domains[name]) for name in required}
    if any(len(seed) != 2 for seed in normalized.values()):
        raise NeuralForceCampaignError("every seed domain must contain two integers")
    if len(set(normalized.values())) != len(normalized):
        raise NeuralForceCampaignError("training/tuning/warmup/retained seeds must be disjoint")
    return {"passed": True, "seed_domains": normalized, "retained_used_for_tuning": False}


def dry_run_tier_a_registry(registry_path: str | Path) -> Mapping[str, Any]:
    """Replay all Tier A transport payloads against their frozen target identities."""

    path = Path(registry_path)
    registry = json.loads(path.read_text(encoding="utf-8"))
    rows = registry.get("cells")
    if not isinstance(rows, list) or len(rows) != 5:
        raise NeuralForceCampaignError("Tier A registry must contain exactly five cells")
    results = []
    for row in rows:
        transport_path = Path(str(row["transport_file"]))
        if not transport_path.is_absolute():
            transport_path = Path.cwd() / transport_path
        payload = json.loads(transport_path.read_text(encoding="utf-8"))
        loaded = load_frozen_neutra_artifact(
            payload,
            expected_target_signature=str(row["target_signature"]),
        )
        expected_semantic_hash = str(row["transport_semantic_hash"])
        actual_semantic_hash = str(loaded.manifest.transport_hash)
        if actual_semantic_hash != expected_semantic_hash:
            raise NeuralForceCampaignError(
                f"{row['cell_id']} transport semantic hash mismatch"
            )
        results.append(
            {
                "cell_id": row["cell_id"],
                "target_signature": row["target_signature"],
                "transport_signature": actual_semantic_hash,
                "artifact_signature": loaded.artifact_signature,
                "dimension": loaded.manifest.dimension,
                "passed": True,
            }
        )
    return {
        "schema": "bayesfilter.neural_force_tier_a_dry_run.v1",
        "passed": True,
        "cell_count": len(results),
        "cells": tuple(results),
        "claims": "target/chart identity resolution only",
    }


def _stable_hash(payload: Mapping[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


__all__ = [
    "NeuralForceCampaignError",
    "NeuralForceSupervision",
    "NeuralForceTargetBinding",
    "NeuralForceTuningCandidate",
    "bind_transformed_neural_force_target",
    "dry_run_tier_a_registry",
    "generate_neural_force_supervision",
    "require_force_target_binding",
    "select_health_aware_tuning_candidate",
    "validate_disjoint_seed_domains",
    "validate_value_only_endpoint_parity",
]
