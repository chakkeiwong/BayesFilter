"""Frozen NeuTra-style transport artifact loader.

This module loads small, reviewed frozen transport manifests only.  It does not
train NeuTra, import model-specific code, or establish sampler validity.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import tensorflow as tf


NEUTRA_ARTIFACT_NONCLAIMS = (
    "frozen transport artifact loader only",
    "no NeuTra training claim",
    "no HMC tuning or sampling claim",
    "no posterior convergence claim",
    "no scientific validity claim",
)

_SUPPORTED_SCHEMAS = {"bayesfilter.neutra.frozen_affine_diag.v1"}
_DENSE_IAF_SCHEMA = "bayesfilter.neutra.dense_iaf_frozen_transport.v1"
_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
_PROCESS_LOCAL_SIGNATURE_PATTERNS = (
    re.compile(r"\b0x[0-9a-fA-F]+\b"),
    re.compile(r"\bobject at\b"),
    re.compile(r"\bid\s*\("),
)


class InvalidNeuTraArtifact(ValueError):
    """Raised when a frozen NeuTra artifact is missing required bindings."""


@dataclass(frozen=True)
class FrozenNeuTraArtifactManifest:
    """Stable manifest for a loaded frozen transport artifact."""

    schema: str
    transport_id: str
    dimension: int
    target_signature: str
    transport_hash: str
    log_jacobian_available: bool
    training_state_hash: str | None
    topology_hash: str | None = None
    tensor_hash: str | None = None
    nonclaims: tuple[str, ...] = NEUTRA_ARTIFACT_NONCLAIMS

    def manifest_payload(self) -> Mapping[str, Any]:
        return {
            "schema": self.schema,
            "transport_id": self.transport_id,
            "dimension": self.dimension,
            "target_signature": self.target_signature,
            "transport_hash": self.transport_hash,
            "log_jacobian_available": self.log_jacobian_available,
            "training_state_hash": self.training_state_hash,
            "topology_hash": self.topology_hash,
            "tensor_hash": self.tensor_hash,
            "nonclaims": self.nonclaims,
        }


class FrozenAffineDiagonalTransport:
    """Frozen affine diagonal transport loaded from a small artifact manifest."""

    def __init__(
        self,
        *,
        manifest: FrozenNeuTraArtifactManifest,
        shift: tuple[float, ...],
        raw_scale: tuple[float, ...],
    ) -> None:
        self.manifest = manifest
        self.parameter_dim = manifest.dimension
        if len(shift) != self.parameter_dim:
            raise InvalidNeuTraArtifact("shift length must match artifact dimension")
        if len(raw_scale) != self.parameter_dim:
            raise InvalidNeuTraArtifact("raw_scale length must match artifact dimension")
        self.shift = tf.constant(shift, dtype=tf.float64)
        self.raw_scale = tf.constant(raw_scale, dtype=tf.float64)

    @property
    def scale(self) -> tf.Tensor:
        return tf.exp(self.raw_scale)

    @property
    def target_signature(self) -> str:
        return self.manifest.target_signature

    def manifest_payload(self) -> Mapping[str, Any]:
        return {
            **self.manifest.manifest_payload(),
            "parameter_dim": self.parameter_dim,
            "shift": tuple(float(item) for item in self.shift.numpy().tolist()),
            "raw_scale": tuple(float(item) for item in self.raw_scale.numpy().tolist()),
        }

    def forward(self, z: Any) -> tf.Tensor:
        values = tf.convert_to_tensor(z, dtype=tf.float64)
        return self.shift + self.scale * values

    def forward_batch(self, z_batch: Any) -> tf.Tensor:
        values = tf.convert_to_tensor(z_batch, dtype=tf.float64)
        if values.shape.rank != 2:
            raise ValueError("frozen transport batch input must have rank 2")
        return self.shift + self.scale * values

    def log_abs_det_jacobian(self, z: Any) -> tf.Tensor:
        values = tf.convert_to_tensor(z, dtype=tf.float64)
        return tf.zeros(tf.shape(values)[:-1], dtype=values.dtype) + tf.reduce_sum(
            self.raw_scale
        )

    def log_abs_det_jacobian_batch(self, z_batch: Any) -> tf.Tensor:
        values = tf.convert_to_tensor(z_batch, dtype=tf.float64)
        if values.shape.rank != 2:
            raise ValueError("frozen transport batch input must have rank 2")
        return tf.zeros(tf.shape(values)[:-1], dtype=values.dtype) + tf.reduce_sum(
            self.raw_scale
        )

    def pullback_score(self, z: Any, theta_score: Any) -> tf.Tensor:
        score = tf.convert_to_tensor(theta_score, dtype=tf.float64)
        return score * tf.cast(self.scale, score.dtype)

    def pullback_score_batch(self, z_batch: Any, theta_score_batch: Any) -> tf.Tensor:
        values = tf.convert_to_tensor(z_batch, dtype=tf.float64)
        if values.shape.rank != 2:
            raise ValueError("frozen transport batch input must have rank 2")
        score = tf.convert_to_tensor(theta_score_batch, dtype=values.dtype)
        if score.shape.rank != 2:
            raise ValueError("frozen transport batch score must have rank 2")
        return score * tf.cast(self.scale, score.dtype)

    def log_abs_det_jacobian_score(self, z: Any) -> tf.Tensor:
        values = tf.convert_to_tensor(z, dtype=tf.float64)
        return tf.zeros_like(values)

    def log_abs_det_jacobian_score_batch(self, z_batch: Any) -> tf.Tensor:
        values = tf.convert_to_tensor(z_batch, dtype=tf.float64)
        if values.shape.rank != 2:
            raise ValueError("frozen transport batch input must have rank 2")
        return tf.zeros_like(values)


class FrozenDenseIAFTransport:
    """Frozen composed dense-IAF transport loaded from a reviewed manifest."""

    def __init__(
        self,
        *,
        manifest: FrozenNeuTraArtifactManifest,
        components: tuple[Any, ...],
    ) -> None:
        self.manifest = manifest
        self.parameter_dim = manifest.dimension
        self.components = components

    @property
    def target_signature(self) -> str:
        return self.manifest.target_signature

    def manifest_payload(self) -> Mapping[str, Any]:
        return {
            **self.manifest.manifest_payload(),
            "parameter_dim": self.parameter_dim,
            "component_count": len(self.components),
        }

    def forward(self, z: Any) -> tf.Tensor:
        values = tf.convert_to_tensor(z, dtype=tf.float64)
        return self.forward_batch(_ensure_rank2(values, "dense IAF input"))[0]

    def forward_batch(self, z_batch: Any) -> tf.Tensor:
        values = _ensure_rank2(tf.convert_to_tensor(z_batch, dtype=tf.float64), "dense IAF batch input")
        output, _ = self._forward_and_logdet(values)
        return output

    def log_abs_det_jacobian(self, z: Any) -> tf.Tensor:
        values = tf.convert_to_tensor(z, dtype=tf.float64)
        return self.log_abs_det_jacobian_batch(_ensure_rank2(values, "dense IAF input"))[0]

    def log_abs_det_jacobian_batch(self, z_batch: Any) -> tf.Tensor:
        values = _ensure_rank2(tf.convert_to_tensor(z_batch, dtype=tf.float64), "dense IAF batch input")
        _, logdet = self._forward_and_logdet(values)
        return logdet

    def _forward_and_logdet(self, values: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        output = values
        logdet = tf.zeros(tf.shape(values)[:-1], dtype=values.dtype)
        for component in self.components:
            output, component_logdet = component.forward_and_logdet(output)
            logdet = logdet + component_logdet
        return output, logdet


class _DenseAutoregressiveIAFComponent:
    def __init__(
        self,
        *,
        dim: int,
        hidden_layers: tuple[int, ...],
        activation: str,
        s_max: float,
        weights: tuple[tf.Tensor, ...],
        biases: tuple[tf.Tensor, ...],
    ) -> None:
        self.dim = dim
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.s_max = s_max
        self.weights = weights
        self.biases = biases
        self.masks = _dense_iaf_masks(dim, hidden_layers)

    def forward_and_logdet(self, values: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        h = values
        for weight, bias, mask in zip(self.weights[:-1], self.biases[:-1], self.masks[:-1]):
            h = tf.matmul(h, weight * mask) + bias
            h = _apply_activation(h, self.activation)
        raw = tf.matmul(h, self.weights[-1] * self.masks[-1]) + self.biases[-1]
        scale_logits = raw[..., : self.dim]
        shift = raw[..., self.dim :]
        scale_log = self.s_max * tf.math.tanh(scale_logits / self.s_max)
        output = values * tf.exp(scale_log) + shift
        return output, tf.reduce_sum(scale_log, axis=-1)


class _MixingLinearComponent:
    def __init__(self, *, matrix: tf.Tensor) -> None:
        self.matrix = matrix
        sign, log_abs_det = tf.linalg.slogdet(matrix)
        if bool(tf.equal(sign, 0.0).numpy()):
            raise InvalidNeuTraArtifact("mixing_linear matrix must be nonsingular")
        self.log_abs_det = tf.convert_to_tensor(log_abs_det, dtype=tf.float64)

    def forward_and_logdet(self, values: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        output = tf.matmul(values, self.matrix)
        logdet = tf.zeros(tf.shape(values)[:-1], dtype=values.dtype) + self.log_abs_det
        return output, logdet


class _AffineComponent:
    def __init__(
        self,
        *,
        offset: tf.Tensor,
        scale: tf.Tensor | None = None,
        matrix: tf.Tensor | None = None,
    ) -> None:
        self.offset = offset
        self.scale = scale
        self.matrix = matrix
        if scale is not None:
            if bool(tf.reduce_any(tf.equal(scale, 0.0)).numpy()):
                raise InvalidNeuTraArtifact("affine scale must be nonzero")
            self.log_abs_det = tf.reduce_sum(tf.math.log(tf.abs(scale)))
        elif matrix is not None:
            sign, log_abs_det = tf.linalg.slogdet(matrix)
            if bool(tf.equal(sign, 0.0).numpy()):
                raise InvalidNeuTraArtifact("affine matrix must be nonsingular")
            self.log_abs_det = tf.convert_to_tensor(log_abs_det, dtype=tf.float64)
        else:
            raise InvalidNeuTraArtifact("affine component requires scale or matrix")

    def forward_and_logdet(self, values: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        if self.scale is not None:
            output = self.offset + values * self.scale
        else:
            output = self.offset + tf.matmul(values, self.matrix, transpose_b=True)
        logdet = tf.zeros(tf.shape(values)[:-1], dtype=values.dtype) + self.log_abs_det
        return output, logdet


class _ComposedComponent:
    def __init__(self, *, children: tuple[Any, ...]) -> None:
        if not children:
            raise InvalidNeuTraArtifact("composed component children must be nonempty")
        self.children = children

    def forward_and_logdet(self, values: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        output = values
        logdet = tf.zeros(tf.shape(values)[:-1], dtype=values.dtype)
        for child in self.children:
            output, child_logdet = child.forward_and_logdet(output)
            logdet = logdet + child_logdet
        return output, logdet


@dataclass(frozen=True)
class LoadedFrozenNeuTraArtifact:
    """Loaded frozen transport plus stable reuse metadata."""

    transport: Any
    manifest: FrozenNeuTraArtifactManifest
    binding: Any
    artifact_signature: str

    def manifest_payload(self) -> Mapping[str, Any]:
        return {
            "schema": "bayesfilter.neutra.loaded_frozen_artifact.v1",
            "artifact_signature": self.artifact_signature,
            "manifest": self.manifest.manifest_payload(),
            "binding": self.binding.manifest_payload(),
        }


def load_frozen_neutra_artifact(
    payload: Mapping[str, Any],
    *,
    expected_target_signature: str,
    binding_factory: Any | None = None,
) -> LoadedFrozenNeuTraArtifact:
    """Load a synthetic/reviewed frozen transport artifact fail-closed."""

    from bayesfilter.ssm import FrozenTransportBinding

    binding_type = FrozenTransportBinding if binding_factory is None else binding_factory
    normalized = _require_mapping(payload, "payload")
    expected = _nonempty_text(expected_target_signature, "expected_target_signature")
    schema = _nonempty_text(normalized.get("schema"), "schema")
    if schema == _DENSE_IAF_SCHEMA:
        return _load_dense_iaf_neutra_artifact(
            normalized,
            expected_target_signature=expected,
            binding_type=binding_type,
        )
    if schema not in _SUPPORTED_SCHEMAS:
        raise InvalidNeuTraArtifact(f"unsupported NeuTra artifact schema: {schema}")
    transport_id = _nonempty_text(normalized.get("transport_id"), "transport_id")
    dimension = int(normalized.get("dimension", 0))
    if dimension <= 0:
        raise InvalidNeuTraArtifact("dimension must be positive")
    target_signature = _nonempty_text(normalized.get("target_signature"), "target_signature")
    if target_signature != expected:
        raise InvalidNeuTraArtifact("target_signature mismatch")
    if not bool(normalized.get("log_jacobian_available", False)):
        raise InvalidNeuTraArtifact("log_jacobian_available is required")
    shift = _float_tuple(normalized.get("shift"), "shift")
    raw_scale = _float_tuple(normalized.get("raw_scale"), "raw_scale")
    if len(shift) != dimension:
        raise InvalidNeuTraArtifact("shift length must match dimension")
    if len(raw_scale) != dimension:
        raise InvalidNeuTraArtifact("raw_scale length must match dimension")
    training_state_hash = normalized.get("training_state_hash")
    if training_state_hash is not None:
        training_state_hash = _nonempty_text(training_state_hash, "training_state_hash")
    transport_hash = _stable_json_hash(
        {
            "schema": schema,
            "transport_id": transport_id,
            "dimension": dimension,
            "target_signature": target_signature,
            "shift": shift,
            "raw_scale": raw_scale,
            "training_state_hash": training_state_hash,
        }
    )
    manifest = FrozenNeuTraArtifactManifest(
        schema=schema,
        transport_id=transport_id,
        dimension=dimension,
        target_signature=target_signature,
        transport_hash=transport_hash,
        log_jacobian_available=True,
        training_state_hash=training_state_hash,
    )
    transport = FrozenAffineDiagonalTransport(
        manifest=manifest,
        shift=shift,
        raw_scale=raw_scale,
    )
    binding = binding_type(
        transport_id=transport_id,
        dimension=dimension,
        target_signature=target_signature,
        log_jacobian_available=True,
        transport_manifest={
            "transport_id": transport_id,
            "transport_hash": transport_hash,
            "target_signature": target_signature,
            "schema": schema,
        },
    )
    artifact_signature = _stable_json_hash(
        {
            "schema": "bayesfilter.neutra.loaded_frozen_artifact.v1",
            "manifest": manifest.manifest_payload(),
            "binding": binding.manifest_payload(),
        }
    )
    return LoadedFrozenNeuTraArtifact(
        transport=transport,
        manifest=manifest,
        binding=binding,
        artifact_signature=artifact_signature,
    )


def finalize_dense_iaf_neutra_artifact_payload(
    payload: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Return a normalized dense-IAF payload with stable component/top-level hashes."""

    normalized = _require_mapping(payload, "payload")
    schema = _nonempty_text(normalized.get("schema"), "schema")
    if schema != _DENSE_IAF_SCHEMA:
        raise InvalidNeuTraArtifact(f"unsupported dense IAF schema: {schema}")
    finalized = dict(normalized)
    components = _component_list(finalized.get("components"), "components")
    finalized["components"] = [_finalize_component_hashes(component) for component in components]
    hashes = _dense_iaf_top_level_hashes(finalized)
    finalized.update(hashes)
    return finalized


def stable_frozen_neutra_artifact_signature(
    artifact: LoadedFrozenNeuTraArtifact,
) -> str:
    """Return the stable persisted signature for a loaded frozen artifact."""

    if not isinstance(artifact, LoadedFrozenNeuTraArtifact):
        raise TypeError("artifact must be a LoadedFrozenNeuTraArtifact")
    return artifact.artifact_signature


def _load_dense_iaf_neutra_artifact(
    normalized: Mapping[str, Any],
    *,
    expected_target_signature: str,
    binding_type: Any,
) -> LoadedFrozenNeuTraArtifact:
    transport_id = _nonempty_text(normalized.get("transport_id"), "transport_id")
    dimension = int(normalized.get("dimension", 0))
    if dimension <= 0:
        raise InvalidNeuTraArtifact("dimension must be positive")
    target_signature = _nonempty_text(normalized.get("target_signature"), "target_signature")
    _require_sha256_hex(target_signature, "target_signature")
    if target_signature != expected_target_signature:
        raise InvalidNeuTraArtifact("target_signature mismatch")
    if not bool(normalized.get("log_jacobian_available", False)):
        raise InvalidNeuTraArtifact("log_jacobian_available is required")
    finalized = finalize_dense_iaf_neutra_artifact_payload(normalized)
    for key in ("topology_hash", "tensor_hash", "transport_hash"):
        supplied = _nonempty_text(normalized.get(key), key)
        _require_sha256_hex(supplied, key)
        if supplied != finalized[key]:
            raise InvalidNeuTraArtifact(f"{key} mismatch")
    component_by_id = {
        _nonempty_text(component.get("component_id"), "component_id"): component
        for component in _component_list(normalized.get("components"), "components")
    }
    component_order = tuple(
        _nonempty_text(item, "component_order item")
        for item in _sequence(normalized.get("component_order"), "component_order")
    )
    if set(component_order) != set(component_by_id) or len(component_order) != len(component_by_id):
        raise InvalidNeuTraArtifact("component_order must match component ids exactly")
    components = tuple(_parse_dense_component(component_by_id[item], dimension) for item in component_order)
    training_state_hash = normalized.get("training_state_hash")
    if training_state_hash is not None:
        training_state_hash = _nonempty_text(training_state_hash, "training_state_hash")
    manifest = FrozenNeuTraArtifactManifest(
        schema=_DENSE_IAF_SCHEMA,
        transport_id=transport_id,
        dimension=dimension,
        target_signature=target_signature,
        transport_hash=finalized["transport_hash"],
        log_jacobian_available=True,
        training_state_hash=training_state_hash,
        topology_hash=finalized["topology_hash"],
        tensor_hash=finalized["tensor_hash"],
    )
    transport = FrozenDenseIAFTransport(manifest=manifest, components=components)
    binding = binding_type(
        transport_id=transport_id,
        dimension=dimension,
        target_signature=target_signature,
        log_jacobian_available=True,
        transport_manifest={
            "transport_id": transport_id,
            "transport_hash": finalized["transport_hash"],
            "target_signature": target_signature,
            "schema": _DENSE_IAF_SCHEMA,
            "topology_hash": finalized["topology_hash"],
            "tensor_hash": finalized["tensor_hash"],
        },
    )
    artifact_signature = _stable_json_hash(
        {
            "schema": "bayesfilter.neutra.loaded_frozen_artifact.v1",
            "manifest": manifest.manifest_payload(),
            "binding": binding.manifest_payload(),
        }
    )
    return LoadedFrozenNeuTraArtifact(
        transport=transport,
        manifest=manifest,
        binding=binding,
        artifact_signature=artifact_signature,
    )


def _parse_dense_component(payload: Mapping[str, Any], dimension: int) -> Any:
    _assert_component_hashes(payload)
    kind = _nonempty_text(payload.get("kind"), "component kind")
    dim = int(payload.get("dim", dimension))
    if dim != dimension:
        raise InvalidNeuTraArtifact("component dim must match artifact dimension")
    if kind == "dense_autoregressive_iaf":
        hidden_layers = tuple(int(item) for item in _sequence(payload.get("hidden_layers"), "hidden_layers"))
        if not hidden_layers or any(item <= 0 for item in hidden_layers):
            raise InvalidNeuTraArtifact("hidden_layers must be positive")
        activation = _nonempty_text(payload.get("activation"), "activation")
        if activation not in {"elu", "tanh", "relu"}:
            raise InvalidNeuTraArtifact(f"unsupported dense IAF activation: {activation}")
        masks_policy = _nonempty_text(payload.get("masks_policy"), "masks_policy")
        if masks_policy != "legacy_degree_masks_v1":
            raise InvalidNeuTraArtifact("unsupported dense IAF masks_policy")
        s_max = float(payload.get("s_max", 0.0))
        if s_max <= 0.0:
            raise InvalidNeuTraArtifact("s_max must be positive")
        layer_sizes = (dimension, *hidden_layers, 2 * dimension)
        weights = _tensor_tuple(payload.get("weights"), "weights")
        biases = _tensor_tuple(payload.get("biases"), "biases")
        if len(weights) != len(layer_sizes) - 1 or len(biases) != len(layer_sizes) - 1:
            raise InvalidNeuTraArtifact("dense IAF parameter length mismatch")
        for index, (weight, bias) in enumerate(zip(weights, biases)):
            _require_shape(weight, (layer_sizes[index], layer_sizes[index + 1]), f"weights[{index}]")
            _require_shape(bias, (layer_sizes[index + 1],), f"biases[{index}]")
        return _DenseAutoregressiveIAFComponent(
            dim=dimension,
            hidden_layers=hidden_layers,
            activation=activation,
            s_max=s_max,
            weights=weights,
            biases=biases,
        )
    if kind == "mixing_linear":
        matrix = _tensor(payload.get("matrix"), "matrix")
        _require_shape(matrix, (dimension, dimension), "matrix")
        return _MixingLinearComponent(matrix=matrix)
    if kind == "affine":
        return _parse_affine_component(payload, dimension)
    if kind == "affine_dense":
        return _parse_affine_component(payload, dimension, dense=True)
    if kind == "composed":
        children = tuple(
            _parse_dense_component(child, dimension)
            for child in _component_list(payload.get("children"), "children")
        )
        return _ComposedComponent(children=children)
    raise InvalidNeuTraArtifact(f"unsupported component kind: {kind}")


def _parse_affine_component(
    payload: Mapping[str, Any],
    dimension: int,
    *,
    dense: bool = False,
) -> _AffineComponent:
    offset = _tensor(payload.get("offset"), "offset")
    _require_shape(offset, (dimension,), "offset")
    if "scale" in payload and payload.get("scale") is not None and not dense:
        scale = _tensor(payload.get("scale"), "scale")
        _require_shape(scale, (dimension,), "scale")
        return _AffineComponent(offset=offset, scale=scale)
    matrix_key = "matrix" if "matrix" in payload else "L_np"
    matrix = _tensor(payload.get(matrix_key), matrix_key)
    _require_shape(matrix, (dimension, dimension), matrix_key)
    return _AffineComponent(offset=offset, matrix=matrix)


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise InvalidNeuTraArtifact(f"{name} must be a mapping")
    normalized = _normalize_for_json(value)
    if not isinstance(normalized, Mapping):
        raise InvalidNeuTraArtifact(f"{name} must normalize to a mapping")
    _reject_process_local_identity(normalized)
    _reject_nonfinite_json_number(normalized)
    return normalized


def _nonempty_text(value: Any, name: str) -> str:
    text = str(value)
    if not text.strip() or text == "None":
        raise InvalidNeuTraArtifact(f"{name} must be nonempty")
    return text


def _float_tuple(value: Any, name: str) -> tuple[float, ...]:
    if value is None:
        raise InvalidNeuTraArtifact(f"{name} is required")
    try:
        result = tuple(float(item) for item in value)
    except TypeError as exc:
        raise InvalidNeuTraArtifact(f"{name} must be a numeric sequence") from exc
    if not result:
        raise InvalidNeuTraArtifact(f"{name} must be nonempty")
    if not all(tf.math.is_finite(tf.constant(result, dtype=tf.float64)).numpy().tolist()):
        raise InvalidNeuTraArtifact(f"{name} must be finite")
    return result


def _tensor(value: Any, name: str) -> tf.Tensor:
    if value is None:
        raise InvalidNeuTraArtifact(f"{name} is required")
    tensor = tf.constant(value, dtype=tf.float64)
    if not bool(tf.reduce_all(tf.math.is_finite(tensor)).numpy()):
        raise InvalidNeuTraArtifact(f"{name} must be finite")
    return tensor


def _tensor_tuple(value: Any, name: str) -> tuple[tf.Tensor, ...]:
    return tuple(_tensor(item, f"{name}[{index}]") for index, item in enumerate(_sequence(value, name)))


def _require_shape(tensor: tf.Tensor, shape: tuple[int, ...], name: str) -> None:
    actual = tuple(tensor.shape.as_list())
    if actual != shape:
        raise InvalidNeuTraArtifact(f"{name} shape mismatch: expected {shape}, got {actual}")


def _ensure_rank2(values: tf.Tensor, name: str) -> tf.Tensor:
    if values.shape.rank == 1:
        return values[tf.newaxis, :]
    if values.shape.rank != 2:
        raise ValueError(f"{name} must have rank 1 or 2")
    return values


def _sequence(value: Any, name: str) -> tuple[Any, ...]:
    if not isinstance(value, (tuple, list)):
        raise InvalidNeuTraArtifact(f"{name} must be a sequence")
    if not value:
        raise InvalidNeuTraArtifact(f"{name} must be nonempty")
    return tuple(value)


def _component_list(value: Any, name: str) -> tuple[Mapping[str, Any], ...]:
    items = _sequence(value, name)
    out = []
    for item in items:
        if not isinstance(item, Mapping):
            raise InvalidNeuTraArtifact(f"{name} entries must be mappings")
        out.append(_normalize_for_json(item))
    return tuple(out)


def _apply_activation(values: tf.Tensor, activation: str) -> tf.Tensor:
    if activation == "elu":
        return tf.nn.elu(values)
    if activation == "tanh":
        return tf.math.tanh(values)
    if activation == "relu":
        return tf.nn.relu(values)
    raise InvalidNeuTraArtifact(f"unsupported activation: {activation}")


def _dense_iaf_masks(dim: int, hidden_layers: tuple[int, ...]) -> tuple[tf.Tensor, ...]:
    degrees: list[list[int]] = [list(range(1, dim + 1))]
    max_degree = max(1, dim - 1)
    for width in hidden_layers:
        degrees.append([1 + (index % max_degree) for index in range(width)])
    degrees.append(list(range(1, dim + 1)) + list(range(1, dim + 1)))
    masks = []
    for layer_index, (deg_in, deg_out) in enumerate(zip(degrees[:-1], degrees[1:])):
        is_output = layer_index == len(degrees) - 2
        masks.append(
            tf.constant(
                [
                    [
                        1.0
                        if ((source < target) if is_output else (source <= target))
                        else 0.0
                        for target in deg_out
                    ]
                    for source in deg_in
                ],
                dtype=tf.float64,
            )
        )
    return tuple(masks)


def _finalize_component_hashes(component: Mapping[str, Any]) -> Mapping[str, Any]:
    finalized = dict(_normalize_for_json(component))
    if finalized.get("kind") == "composed":
        finalized["children"] = [
            _finalize_component_hashes(child)
            for child in _component_list(finalized.get("children"), "children")
        ]
    finalized["component_topology_hash"] = _stable_json_hash(_component_topology_payload(finalized))
    finalized["component_tensor_hash"] = _stable_json_hash(_component_tensor_payload(finalized))
    return finalized


def _assert_component_hashes(component: Mapping[str, Any]) -> None:
    for key, payload_fn in (
        ("component_topology_hash", _component_topology_payload),
        ("component_tensor_hash", _component_tensor_payload),
    ):
        supplied = _nonempty_text(component.get(key), key)
        _require_sha256_hex(supplied, key)
        expected = _stable_json_hash(payload_fn(component))
        if supplied != expected:
            raise InvalidNeuTraArtifact(f"{key} mismatch")
    if component.get("kind") == "composed":
        for child in _component_list(component.get("children"), "children"):
            _assert_component_hashes(child)


def _dense_iaf_top_level_hashes(payload: Mapping[str, Any]) -> Mapping[str, str]:
    topology_hash = _stable_json_hash(_transport_topology_payload(payload))
    tensor_hash = _stable_json_hash(_transport_tensor_payload(payload))
    transport_hash = _stable_json_hash(
        {
            "schema": _DENSE_IAF_SCHEMA,
            "transport_id": payload.get("transport_id"),
            "dimension": payload.get("dimension"),
            "target_signature": payload.get("target_signature"),
            "log_jacobian_available": bool(payload.get("log_jacobian_available", False)),
            "logdet_semantics": "sum_component_log_abs_det_jacobian_forward_order",
            "topology_hash": topology_hash,
            "tensor_hash": tensor_hash,
        }
    )
    return {
        "topology_hash": topology_hash,
        "tensor_hash": tensor_hash,
        "transport_hash": transport_hash,
    }


def _transport_topology_payload(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "schema": payload.get("schema"),
        "dimension": payload.get("dimension"),
        "component_order": payload.get("component_order"),
        "components": [
            _component_topology_payload(component)
            for component in _component_list(payload.get("components"), "components")
        ],
    }


def _transport_tensor_payload(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "schema": payload.get("schema"),
        "dimension": payload.get("dimension"),
        "component_order": payload.get("component_order"),
        "components": [
            _component_tensor_payload(component)
            for component in _component_list(payload.get("components"), "components")
        ],
    }


def _component_topology_payload(component: Mapping[str, Any]) -> Mapping[str, Any]:
    kind = component.get("kind")
    base: dict[str, Any] = {
        "component_id": component.get("component_id"),
        "kind": kind,
        "dim": component.get("dim"),
        "dtype": component.get("dtype"),
    }
    if kind == "dense_autoregressive_iaf":
        base.update(
            {
                "hidden_layers": component.get("hidden_layers"),
                "activation": component.get("activation"),
                "s_max": component.get("s_max"),
                "masks_policy": component.get("masks_policy"),
                "weight_shapes": [_shape_of_nested(item) for item in _sequence(component.get("weights"), "weights")],
                "bias_shapes": [_shape_of_nested(item) for item in _sequence(component.get("biases"), "biases")],
            }
        )
    elif kind == "mixing_linear":
        base["matrix_shape"] = _shape_of_nested(component.get("matrix"))
    elif kind in {"affine", "affine_dense"}:
        base["offset_shape"] = _shape_of_nested(component.get("offset"))
        if component.get("scale") is not None:
            base["scale_shape"] = _shape_of_nested(component.get("scale"))
        matrix_key = "matrix" if "matrix" in component else "L_np"
        if component.get(matrix_key) is not None:
            base["matrix_key"] = matrix_key
            base["matrix_shape"] = _shape_of_nested(component.get(matrix_key))
    elif kind == "composed":
        base["children"] = [
            _component_topology_payload(child)
            for child in _component_list(component.get("children"), "children")
        ]
    return base


def _component_tensor_payload(component: Mapping[str, Any]) -> Mapping[str, Any]:
    kind = component.get("kind")
    base: dict[str, Any] = {
        "component_id": component.get("component_id"),
        "kind": kind,
        "dim": component.get("dim"),
        "dtype": component.get("dtype"),
    }
    if kind == "dense_autoregressive_iaf":
        base["weights"] = component.get("weights")
        base["biases"] = component.get("biases")
    elif kind == "mixing_linear":
        base["matrix"] = component.get("matrix")
    elif kind in {"affine", "affine_dense"}:
        base["offset"] = component.get("offset")
        if component.get("scale") is not None:
            base["scale"] = component.get("scale")
        matrix_key = "matrix" if "matrix" in component else "L_np"
        if component.get(matrix_key) is not None:
            base[matrix_key] = component.get(matrix_key)
    elif kind == "composed":
        base["children"] = [
            _component_tensor_payload(child)
            for child in _component_list(component.get("children"), "children")
        ]
    return base


def _shape_of_nested(value: Any) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    if not value:
        return (0,)
    return (len(value), *_shape_of_nested(value[0]))


def _require_sha256_hex(value: str, name: str) -> None:
    if not _SHA256_HEX_RE.fullmatch(value):
        raise InvalidNeuTraArtifact(f"{name} must be a 64-character sha256 hex digest")


def _normalize_for_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _normalize_for_json(item)
            for key, item in sorted(value.items(), key=lambda entry: str(entry[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_normalize_for_json(item) for item in value]
    return value


def _stable_json_hash(payload: Mapping[str, Any] | Any) -> str:
    normalized = _normalize_for_json(payload)
    _reject_nonfinite_json_number(normalized)
    blob = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    _reject_process_local_identity(blob)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _reject_nonfinite_json_number(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise InvalidNeuTraArtifact("frozen NeuTra artifact manifests must contain finite numbers")
    if isinstance(value, Mapping):
        for item in value.values():
            _reject_nonfinite_json_number(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _reject_nonfinite_json_number(item)


def _reject_process_local_identity(value: Any) -> None:
    text = json.dumps(_normalize_for_json(value), sort_keys=True, default=str)
    if any(pattern.search(text) for pattern in _PROCESS_LOCAL_SIGNATURE_PATTERNS):
        raise InvalidNeuTraArtifact(
            "frozen NeuTra artifact manifests must not contain process-local identity"
        )
