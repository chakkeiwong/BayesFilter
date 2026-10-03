"""Local research checkpoints: atomic committed bundles and replayable work units.

Uncommitted attempts are retained for diagnosis. A killed GPU call is replayed
from its inputs and stateless seed; live device instructions are not serialized.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import time
import uuid
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any


class CheckpointError(RuntimeError):
    """Committed evidence is corrupt or belongs to a different computation."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()


def payload_hash(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def durable_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial-" + uuid.uuid4().hex)
    with temporary.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    _sync_directory(path.parent)


def durable_json(path: Path, value: Any) -> None:
    durable_bytes(path, canonical_bytes(value) + b"\n")


class DurableTensorCheckpoint:
    """Serialize TensorFlow tensor trees without pickle or NumPy computation.

    One process owns a store at a time. Each committed unit binds its inputs and
    the store identity; incomplete attempts never become readable checkpoints.
    """

    def __init__(
        self,
        root: str | Path,
        identity: Mapping[str, Any],
        *,
        event_callback: Callable[[str, Mapping[str, Any]], None] | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = (self.root / ".writer.lock").open("a+b")
        try:
            fcntl.flock(self._lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._lock.close()
            raise CheckpointError("checkpoint store already has a live writer") from exc
        self.identity = json.loads(canonical_bytes(identity))
        self.event_callback = event_callback
        self.records: list[dict[str, Any]] = []
        identity_path = self.root / "identity.json"
        try:
            if identity_path.exists():
                if json.loads(identity_path.read_bytes()) != self.identity:
                    raise CheckpointError("checkpoint identity changed")
            else:
                durable_json(identity_path, self.identity)
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        if not self._lock.closed:
            fcntl.flock(self._lock.fileno(), fcntl.LOCK_UN)
            self._lock.close()

    def __enter__(self) -> "DurableTensorCheckpoint":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    @staticmethod
    def tensor_hash(value: Any) -> str:
        import tensorflow as tf

        with tf.device("/CPU:0"):
            return hashlib.sha256(tf.io.serialize_tensor(value).numpy()).hexdigest()

    def _path(self, key: str) -> Path:
        if not key or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in key):
            raise CheckpointError("invalid checkpoint key")
        return self.root / "committed" / key

    def contains(self, key: str) -> bool:
        return self._path(key).exists()

    def _event(self, attempt: Path, name: str, payload: Mapping[str, Any]) -> None:
        event = {**payload, "event": name, "pid": os.getpid(), "at_unix": time.time()}
        durable_json(attempt / f"{name}.json", event)
        if self.event_callback is not None:
            self.event_callback(name, event)

    def _encode(self, value: Any, directory: Path, receipts: list[Any]) -> Any:
        import tensorflow as tf

        if tf.is_tensor(value):
            filename = f"tensor-{len(receipts):04d}.bin"
            with tf.device("/CPU:0"):
                content = tf.io.serialize_tensor(value).numpy()
            durable_bytes(directory / filename, content)
            receipt = {"file": filename, "sha256": hashlib.sha256(content).hexdigest(),
                       "dtype": value.dtype.name, "shape": value.shape.as_list()}
            receipts.append(receipt)
            return {"tensor": receipt}
        if isinstance(value, Mapping):
            return {"mapping": {str(key): self._encode(item, directory, receipts) for key, item in value.items()}}
        if isinstance(value, (tuple, list)):
            return {"sequence": [self._encode(item, directory, receipts) for item in value]}
        canonical_bytes(value)
        return {"scalar": value}

    def _decode(self, tree: Any, directory: Path) -> Any:
        import tensorflow as tf

        if "tensor" in tree:
            receipt = tree["tensor"]
            path = directory / receipt["file"]
            if path.parent != directory or not path.is_file():
                raise CheckpointError("missing or invalid committed tensor path")
            content = path.read_bytes()
            if hashlib.sha256(content).hexdigest() != receipt["sha256"]:
                raise CheckpointError("committed tensor checksum mismatch")
            with tf.device("/CPU:0"):
                tensor = tf.io.parse_tensor(content, out_type=tf.as_dtype(receipt["dtype"]))
            if tensor.shape.as_list() != receipt["shape"]:
                raise CheckpointError("committed tensor shape mismatch")
            return tensor
        if "mapping" in tree:
            return {key: self._decode(item, directory) for key, item in tree["mapping"].items()}
        if "sequence" in tree:
            return [self._decode(item, directory) for item in tree["sequence"]]
        return tree["scalar"]

    def load(self, key: str, inputs: Mapping[str, Any]) -> tuple[Any, Mapping[str, Any]]:
        directory = self._path(key)
        try:
            content = (directory / "bundle.json").read_bytes()
            if hashlib.sha256(content).hexdigest() != (directory / "bundle.sha256").read_text():
                raise CheckpointError("committed bundle checksum mismatch")
            bundle = json.loads(content)
            if bundle["identity_hash"] != payload_hash(self.identity) or bundle["inputs_hash"] != payload_hash(inputs):
                raise CheckpointError("committed checkpoint input or identity mismatch")
            return self._decode(bundle["tree"], directory), bundle["metadata"]
        except (OSError, ValueError, KeyError) as exc:
            raise CheckpointError(f"invalid committed checkpoint {key}: {exc}") from exc

    def run(self, key: str, inputs: Mapping[str, Any], compute: Callable[[], Any]) -> Any:
        destination = self._path(key)
        if destination.exists():
            value, metadata = self.load(key, inputs)
            self.records.append({"key": key, "replayed": True, **metadata})
            return value
        attempt = self.root / "attempts" / f"{key}-{uuid.uuid4().hex}"
        attempt.mkdir(parents=True)
        info = {"key": key, "inputs": inputs, "identity_hash": payload_hash(self.identity)}
        self._event(attempt, "call-start", info)
        started = time.monotonic()
        value = compute()
        import tensorflow as tf

        for item in tf.nest.flatten(value):
            if tf.is_tensor(item):
                item.numpy()
        compute_seconds = time.monotonic() - started
        self._event(attempt, "call-end", {**info, "compute_seconds": compute_seconds})
        temporary = attempt / "bundle"
        temporary.mkdir()
        serialization_started = time.monotonic()
        receipts: list[Any] = []
        tree = self._encode(value, temporary, receipts)
        self._event(attempt, "tensors-written", info)
        metadata = {"compute_seconds": compute_seconds,
                    "serialization_seconds": time.monotonic() - serialization_started,
                    "completed_at_unix": time.time(), "attempt": str(attempt)}
        bundle = {"schema": "bayesfilter.durable_tensor_checkpoint.v1",
                  "identity_hash": payload_hash(self.identity), "inputs_hash": payload_hash(inputs),
                  "tree": tree, "metadata": metadata}
        content = canonical_bytes(bundle)
        durable_bytes(temporary / "bundle.json", content)
        durable_bytes(temporary / "bundle.sha256", hashlib.sha256(content).hexdigest().encode())
        destination.parent.mkdir(parents=True, exist_ok=True)
        self._event(attempt, "before-commit", info)
        os.rename(temporary, destination)
        _sync_directory(destination.parent)
        self.records.append({"key": key, "replayed": False, **metadata})
        self._event(attempt, "committed", info)
        return value
