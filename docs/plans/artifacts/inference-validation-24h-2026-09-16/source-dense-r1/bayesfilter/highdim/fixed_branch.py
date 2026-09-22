"""Canonical fixed-branch identity helpers for high-dimensional filters."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import struct
from typing import Any, Mapping

import tensorflow as tf

from bayesfilter.highdim.diagnostics import HighDimStatus


@dataclass(frozen=True)
class BranchHash:
    """SHA-256 digest over a full canonical branch manifest."""

    value: str

    def __post_init__(self) -> None:
        value = str(self.value)
        if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
            raise ValueError("BranchHash must be a lowercase SHA-256 hex digest")
        object.__setattr__(self, "value", value)


@dataclass(frozen=True)
class BranchManifest:
    """Full branch manifest with deterministic canonical serialization."""

    version: str
    payload: Mapping[str, object]

    def __post_init__(self) -> None:
        if not str(self.version).strip():
            raise ValueError("version must be nonempty")
        if not isinstance(self.payload, Mapping):
            raise TypeError("payload must be a mapping")

    def to_canonical_bytes(self) -> bytes:
        canonical = {
            "type": "BranchManifest",
            "version": str(self.version),
            "payload": _canonicalize(self.payload),
        }
        return json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")

    def sha256(self) -> BranchHash:
        return BranchHash(hashlib.sha256(self.to_canonical_bytes()).hexdigest())


@dataclass(frozen=True)
class BranchIdentity:
    """Validated identity tying a manifest to its full-manifest hash."""

    manifest: BranchManifest
    hash: BranchHash

    def __post_init__(self) -> None:
        if not isinstance(self.manifest, BranchManifest):
            raise TypeError("manifest must be a BranchManifest")
        if not isinstance(self.hash, BranchHash):
            raise TypeError("hash must be a BranchHash")
        expected = self.manifest.sha256()
        if self.hash != expected:
            raise ValueError(HighDimStatus.SELECTIVE_BRANCH_HASH_REJECTED.value)


def _canonicalize(value: Any) -> Any:
    if isinstance(value, tf.Tensor):
        return _canonicalize_tensor(value)
    if isinstance(value, Mapping):
        return {
            str(key): _canonicalize(value[key])
            for key in sorted(value, key=lambda item: str(item))
        }
    if isinstance(value, tuple):
        return {"type": "tuple", "items": [_canonicalize(item) for item in value]}
    if isinstance(value, list):
        return {"type": "list", "items": [_canonicalize(item) for item in value]}
    if value is None:
        return {"type": "none", "value": None}
    if isinstance(value, bool):
        return {"type": "bool", "value": value}
    if isinstance(value, int) and not isinstance(value, bool):
        return {"type": "int", "value": str(value)}
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("branch manifest floats must be finite")
        return {"type": "float64", "hex": struct.pack(">d", value).hex()}
    if isinstance(value, str):
        return {"type": "str", "value": value}
    raise TypeError(f"unsupported branch manifest value type: {type(value)!r}")


def _canonicalize_tensor(tensor: tf.Tensor) -> Mapping[str, object]:
    value = tf.convert_to_tensor(tensor)
    if not bool(tf.reduce_all(tf.math.is_finite(tf.cast(value, tf.float64))).numpy()):
        raise ValueError("branch manifest tensors must be finite")
    flat = tf.reshape(value, [-1])
    if value.dtype.is_floating:
        cast_flat = tf.cast(flat, tf.float64)
        encoded = [struct.pack(">d", float(item)).hex() for item in cast_flat.numpy()]
    elif value.dtype.is_integer:
        encoded = [str(int(item)) for item in flat.numpy()]
    elif value.dtype == tf.bool:
        encoded = [bool(item) for item in flat.numpy()]
    else:
        raise TypeError(f"unsupported tensor dtype for branch manifest: {value.dtype}")
    return {
        "type": "tensor",
        "dtype": value.dtype.name,
        "shape": list(value.shape.as_list()),
        "encoding": "big_endian_scalar_list",
        "values": encoded,
    }
