"""Fixed-transport HMC mechanics binding helpers.

The helpers in this module build manifest-rich mechanics checks only.  They do
not run serious HMC, tune a new policy, or establish sampler convergence.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import tensorflow as tf

from bayesfilter.inference.batched_value_score import FixedTransportValueScoreAdapter
from bayesfilter.inference.hmc_tuning import HMCTuningPolicy, normalize_hmc_tuning_policy
from bayesfilter.inference.posterior_adapter import value_score_capability


FIXED_TRANSPORT_HMC_BINDING_NONCLAIMS = (
    "fixed-transport HMC mechanics binding only",
    "CPU-only smoke when execution_device is cpu",
    "no serious HMC convergence claim",
    "no posterior validity claim",
    "no sampler superiority claim",
    "no default HMC policy change",
)

_PROCESS_LOCAL_SIGNATURE_PATTERNS = (
    re.compile(r"\b0x[0-9a-fA-F]+\b"),
    re.compile(r"\bobject at\b"),
    re.compile(r"\bid\s*\("),
)


class InvalidFixedTransportHMCBinding(ValueError):
    """Raised when fixed-transport HMC mechanics binding is unsafe."""


@dataclass(frozen=True)
class FixedTransportHMCManifest:
    """Manifest for one fixed-transport HMC mechanics binding."""

    target_signature: str
    transport_hash: str
    hmc_policy_label: str
    hmc_policy_hash: str
    xla_hmc_ready: bool
    use_xla: bool
    mass_policy: str
    seed: int
    execution_device: str
    mechanics_only: bool
    adapter_signature: str
    nonclaims: tuple[str, ...] = FIXED_TRANSPORT_HMC_BINDING_NONCLAIMS

    def manifest_payload(self) -> Mapping[str, Any]:
        return {
            "target_signature": self.target_signature,
            "transport_hash": self.transport_hash,
            "hmc_policy_label": self.hmc_policy_label,
            "hmc_policy_hash": self.hmc_policy_hash,
            "xla_hmc_ready": self.xla_hmc_ready,
            "use_xla": self.use_xla,
            "mass_policy": self.mass_policy,
            "seed": self.seed,
            "execution_device": self.execution_device,
            "mechanics_only": self.mechanics_only,
            "adapter_signature": self.adapter_signature,
            "nonclaims": self.nonclaims,
        }


@dataclass(frozen=True)
class FixedTransportHMCMechanicsResult:
    """Value/score mechanics output plus manifest metadata."""

    adapter: FixedTransportValueScoreAdapter
    manifest: FixedTransportHMCManifest
    value: tf.Tensor
    score: tf.Tensor
    diagnostics: Mapping[str, Any]


def bind_fixed_transport_hmc_mechanics(
    *,
    base_adapter: Any,
    loaded_artifact: Any,
    initial_position: Any,
    target_scope: str,
    tuning_policy: str | HMCTuningPolicy | None = None,
    use_xla: bool = False,
    mass_policy: str = "identity_mass_mechanics_smoke",
    seed: int = 0,
    execution_device: str = "cpu",
) -> FixedTransportHMCMechanicsResult:
    """Bind a fixed transport to a base adapter and run one mechanics check."""

    if not hasattr(loaded_artifact, "transport") or not hasattr(loaded_artifact, "binding"):
        raise TypeError("loaded_artifact must expose transport and binding")
    target = _nonempty_text(target_scope, "target_scope")
    policy = normalize_hmc_tuning_policy(tuning_policy)
    policy_hash = _stable_json_hash(
        {
            "schema": "bayesfilter.fixed_transport_hmc_policy.v1",
            "policy": policy.payload(),
        }
    )
    mass = _nonempty_text(mass_policy, "mass_policy")
    device = _nonempty_text(execution_device, "execution_device")
    seed_value = int(seed)
    if seed_value < 0:
        raise InvalidFixedTransportHMCBinding("seed must be nonnegative")
    binding = loaded_artifact.binding
    target_signature = _nonempty_text(binding.target_signature, "target_signature")
    transport_manifest = binding.transport_manifest
    transport_hash = _nonempty_text(
        transport_manifest.get("transport_hash"),
        "transport_hash",
    )
    adapter = FixedTransportValueScoreAdapter(
        base_adapter=base_adapter,
        transport=loaded_artifact.transport,
        target_scope=target,
        xla_hmc_ready=bool(use_xla),
        full_chain_xla_diagnostic_ready=False,
        evidence_path="bayesfilter/inference/fixed_transport_hmc.py",
    )
    capability = adapter.value_score_capability()
    if use_xla and not capability.is_accepted_xla_hmc_authority:
        raise InvalidFixedTransportHMCBinding(
            "XLA fixed-transport HMC binding requires accepted value/score authority"
        )
    theta = tf.convert_to_tensor(initial_position, dtype=tf.float64)
    if theta.shape.rank != 2:
        raise ValueError("fixed-transport HMC mechanics initial_position must be [B, D]")
    value, score = adapter.log_prob_and_grad_batch(theta)
    value = tf.convert_to_tensor(value, dtype=tf.float64)
    score = tf.convert_to_tensor(score, dtype=tf.float64)
    if value.shape.rank != 1:
        raise ValueError("fixed-transport HMC mechanics value must be rank 1")
    if score.shape.rank != 2:
        raise ValueError("fixed-transport HMC mechanics score must be rank 2")
    with tf.control_dependencies(
        [
            tf.debugging.assert_all_finite(value, "mechanics value must be finite"),
            tf.debugging.assert_all_finite(score, "mechanics score must be finite"),
        ]
    ):
        value = tf.identity(value)
        score = tf.identity(score)
    manifest = FixedTransportHMCManifest(
        target_signature=target_signature,
        transport_hash=transport_hash,
        hmc_policy_label=policy.label,
        hmc_policy_hash=policy_hash,
        xla_hmc_ready=bool(capability.is_accepted_xla_hmc_authority),
        use_xla=bool(use_xla),
        mass_policy=mass,
        seed=seed_value,
        execution_device=device,
        mechanics_only=True,
        adapter_signature=adapter.adapter_signature(),
    )
    return FixedTransportHMCMechanicsResult(
        adapter=adapter,
        manifest=manifest,
        value=value,
        score=score,
        diagnostics={
            "mechanics_only": True,
            "value_shape": tuple(value.shape.as_list()),
            "score_shape": tuple(score.shape.as_list()),
            "hmc_policy_label": policy.label,
            "hmc_policy_hash": policy_hash,
            "execution_device": device,
            "nonclaims": FIXED_TRANSPORT_HMC_BINDING_NONCLAIMS,
        },
    )


def stable_fixed_transport_hmc_manifest_signature(
    manifest: FixedTransportHMCManifest,
) -> str:
    """Return a stable SHA-256 signature for a mechanics binding manifest."""

    if not isinstance(manifest, FixedTransportHMCManifest):
        raise TypeError("manifest must be a FixedTransportHMCManifest")
    return _stable_json_hash(
        {
            "schema": "bayesfilter.fixed_transport_hmc_manifest.v1",
            "manifest": manifest.manifest_payload(),
        }
    )


def _nonempty_text(value: Any, name: str) -> str:
    text = str(value)
    if not text.strip():
        raise InvalidFixedTransportHMCBinding(f"{name} must be nonempty")
    return text


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
        raise InvalidFixedTransportHMCBinding(
            "fixed-transport HMC manifests must not contain process-local identity"
        )
