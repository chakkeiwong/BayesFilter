"""Historical ordinary-tuner diagnostic recovery at completed attempt boundaries.

The public candidate-set procedure uses its own numerical checkpoints and
resume API. This helper does not select a public tuning route.

This is artifact I/O, not another numerical runtime. TensorFlow tensor shards
and validated dataclass fields retain the mass, start bank, repair history and
stage results. Compiled runner caches are rebuilt in the next process. No live
TensorFlow graph, interrupted transition, or posterior sample is resumed.
"""
from __future__ import annotations

import dataclasses
import math
import os
import sys
import time
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from bayesfilter.runtime.durable_tensor_checkpoint import (
    CheckpointError, DurableTensorCheckpoint, durable_json,
)


def _pack(value: Any) -> Any:
    import tensorflow as tf

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        kind = type(value)
        if not kind.__module__.startswith("bayesfilter."):
            raise CheckpointError(f"unsupported checkpoint class: {kind}")
        # A runner cache contains closures/graphs, not learned tuning state.
        fields = {field.name: _pack(getattr(value, field.name))
                  for field in dataclasses.fields(value)
                  if field.init and field.name != "private_runner_cache_handoff"}
        return {"kind": "dataclass", "module": kind.__module__,
                "name": kind.__name__, "fields": fields}
    if tf.is_tensor(value):
        return {"kind": "tensor", "value": value}
    if type(value).__module__.startswith("numpy"):
        return {"kind": "array", "value": tf.convert_to_tensor(value)}
    if isinstance(value, Mapping):
        return {"kind": "mapping", "items": [[_pack(k), _pack(v)] for k, v in value.items()]}
    if isinstance(value, (tuple, list)):
        return {"kind": "tuple" if isinstance(value, tuple) else "list",
                "items": [_pack(item) for item in value]}
    if isinstance(value, float) and not math.isfinite(value):
        return {"kind": "nonfinite", "value": str(value)}
    if value is None or isinstance(value, (str, bool, int, float)):
        return {"kind": "scalar", "value": value}
    raise CheckpointError(f"unsupported checkpoint value: {type(value)}")


def _unpack(tree: Any) -> Any:
    kind = tree["kind"]
    if kind == "dataclass":
        # Only already imported BayesFilter dataclasses can be constructed.
        # No pickle, executable data, or module imports come from the artifact.
        module = sys.modules.get(tree["module"])
        cls = getattr(module, tree["name"], None)
        if (not tree["module"].startswith("bayesfilter.")
                or not isinstance(cls, type) or not dataclasses.is_dataclass(cls)
                or cls.__module__ != tree["module"]):
            raise CheckpointError("checkpoint dataclass is unavailable")
        return cls(**{key: _unpack(value) for key, value in tree["fields"].items()})
    if kind == "mapping":
        return {_unpack(key): _unpack(value) for key, value in tree["items"]}
    if kind in {"tuple", "list"}:
        values = [_unpack(value) for value in tree["items"]]
        return tuple(values) if kind == "tuple" else values
    if kind == "array":
        return tree["value"].numpy()
    if kind == "nonfinite":
        return float(tree["value"])
    if kind in {"tensor", "scalar"}:
        return tree["value"]
    raise CheckpointError("unknown checkpoint value kind")


class HMCTuningCampaignCheckpoint:
    """One locked campaign with cumulative active wall time and attempt slots.

    Budget increases are explicit arguments; identity excludes only resource
    limits. An unclosed process must be reconciled with its supervisor's actual
    elapsed time before reuse. A heartbeat is progress, not proof of termination.
    """

    def __init__(self, root: str | Path, identity: Mapping[str, Any], *,
                 max_attempts: int, time_budget_s: float,
                 interrupted_elapsed_s: float | None = None) -> None:
        import json

        self.store = DurableTensorCheckpoint(root, identity)
        self.root = self.store.root
        self.started = time.monotonic()
        self.session = None
        try:
            self.ledger_path = self.root / "campaign.json"
            self.ledger = (json.loads(self.ledger_path.read_text()) if self.ledger_path.exists()
                           else {"schema": "bayesfilter.hmc_tuning_campaign.v1", "calls": []})
            calls = self.ledger["calls"]
            if calls and "elapsed_seconds" not in calls[-1]:
                if interrupted_elapsed_s is None:
                    raise CheckpointError("interrupted campaign call needs supervisor elapsed-time reconciliation")
                if not math.isfinite(interrupted_elapsed_s) or interrupted_elapsed_s < 0:
                    raise CheckpointError("invalid interrupted-call elapsed time")
                calls[-1] = {**calls[-1], "elapsed_seconds": interrupted_elapsed_s,
                             "status": "interrupted_reconciled"}
                durable_json(self.ledger_path, self.ledger)
            elif interrupted_elapsed_s is not None:
                raise CheckpointError("no interrupted campaign call to reconcile")
            self.elapsed_before = sum(call["elapsed_seconds"] for call in calls)
            if not math.isfinite(time_budget_s) or time_budget_s <= 0:
                raise ValueError("campaign time budget must be positive and finite")
            self.max_attempts = max_attempts
            self.time_budget_s = time_budget_s
            self.remaining_seconds = max(0.0, time_budget_s - self.elapsed_before)
            keys = sorted(path.name for path in (self.root / "committed").glob("attempt-*"))
            if keys != [f"attempt-{index:04d}" for index in range(len(keys))]:
                raise CheckpointError("committed attempt sequence has a gap")
            self.attempt_states = [self.load(key) for key in keys]
            if len(keys) > max_attempts:
                raise CheckpointError("campaign attempt limit is below completed work")
            if self.remaining_seconds <= 60:
                raise CheckpointError("campaign time budget exhausted (60-second closeout reserve)")
            self.session = {"id": uuid.uuid4().hex, "pid": os.getpid(), "started_at_unix": time.time(),
                            "max_attempts": max_attempts, "time_budget_seconds": time_budget_s,
                            "elapsed_before_seconds": self.elapsed_before,
                            "remaining_seconds": self.remaining_seconds}
            calls.append(self.session)
            durable_json(self.ledger_path, self.ledger)
        except BaseException:
            self.store.close()
            raise

    def __enter__(self) -> "HMCTuningCampaignCheckpoint":
        return self

    def __exit__(self, exc_type, _exc, _tb) -> None:
        try:
            self.session.update(elapsed_seconds=time.monotonic() - self.started,
                                status="returned" if exc_type is None else "exception")
            durable_json(self.ledger_path, self.ledger)
        finally:
            self.store.close()

    def load(self, key: str) -> Any:
        tree, _ = self.store.load(key, {"key": key})
        return _unpack(tree)

    def stage(self, key: str, compute) -> Any:
        if self.store.contains(key):
            return self.load(key)
        value = compute()
        self.store.run(key, {"key": key}, lambda: _pack(value))
        return value

    def commit_attempt(self, state: Mapping[str, Any]) -> None:
        index = state["attempt"].attempt_index
        if index != len(self.attempt_states) or index >= self.max_attempts:
            raise CheckpointError("attempt commit exceeds campaign sequence or limit")
        self.stage(f"attempt-{index:04d}", lambda: dict(state))
        self.attempt_states.append(dict(state))
        durable_json(self.root / "latest.json", {
            "completed_attempts": len(self.attempt_states),
            "next_attempt_index": index + 1,
            "elapsed_seconds": self.elapsed_before + time.monotonic() - self.started,
            "last_attempt_status": state["attempt"].final_status,
            "checkpoint_key": f"attempt-{index:04d}",
        })
