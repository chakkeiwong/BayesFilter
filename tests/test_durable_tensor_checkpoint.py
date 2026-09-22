"""CPU-only checkpoint and interruption mechanics; no scientific admission."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest
import tensorflow as tf

from bayesfilter.runtime.durable_tensor_checkpoint import CheckpointError, DurableTensorCheckpoint


@pytest.mark.parametrize("event", ("call-start", "call-end", "tensors-written", "before-commit", "committed"))
def test_hard_process_death_at_each_commit_boundary(tmp_path, event):
    root = tmp_path / "store"
    program = """
import os, signal, sys
import tensorflow as tf
from bayesfilter.runtime.durable_tensor_checkpoint import DurableTensorCheckpoint
def hook(event, payload):
    if event == sys.argv[2]:
        os.kill(os.getpid(), signal.SIGKILL)
with DurableTensorCheckpoint(sys.argv[1], {"source": "test"}, event_callback=hook) as store:
    store.run("unit", {"seed": [1, 2]}, lambda: {"state": tf.constant([[1., 2.]], tf.float64)})
"""
    result = subprocess.run([sys.executable, "-c", program, str(root), event],
                            env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1", "TF_NUM_INTRAOP_THREADS": "1",
                                 "TF_NUM_INTEROP_THREADS": "1"}, capture_output=True, timeout=60)
    assert result.returncode == -signal.SIGKILL, result.stderr.decode()
    calls = []
    with DurableTensorCheckpoint(root, {"source": "test"}) as store:
        value = store.run("unit", {"seed": [1, 2]}, lambda: calls.append(True) or {"state": tf.constant([[1., 2.]], tf.float64)})
        assert bool(tf.reduce_all(value["state"] == [[1., 2.]]).numpy())
        assert store.records[0]["replayed"] is (event == "committed")
        assert bool(calls) is (event != "committed")
    assert len(list((root / "committed").iterdir())) == 1
    assert list((root / "attempts").iterdir())


def test_corrupt_committed_tensor_never_recomputed(tmp_path):
    with DurableTensorCheckpoint(tmp_path, {"source": "test"}) as store:
        store.run("unit", {}, lambda: tf.constant(2.0))
    (tmp_path / "committed/unit/tensor-0000.bin").write_bytes(b"broken")
    with DurableTensorCheckpoint(tmp_path, {"source": "test"}) as store:
        with pytest.raises(CheckpointError, match="checksum"):
            store.run("unit", {}, lambda: pytest.fail("corruption cannot authorize regeneration"))


def test_identity_input_and_live_writer_are_checked(tmp_path):
    with DurableTensorCheckpoint(tmp_path, {"source": "test"}) as store:
        store.run("unit", {"seed": 1}, lambda: tf.constant(2.0))
        with pytest.raises(CheckpointError, match="live writer"):
            DurableTensorCheckpoint(tmp_path, {"source": "test"})
        with pytest.raises(CheckpointError, match="input"):
            store.run("unit", {"seed": 2}, lambda: None)
    with pytest.raises(CheckpointError, match="identity"):
        DurableTensorCheckpoint(tmp_path, {"source": "changed"})


def test_partial_metadata_does_not_mask_committed_evidence(tmp_path):
    with DurableTensorCheckpoint(tmp_path, {}) as store:
        store.run("unit", {}, lambda: {"tensor": tf.constant([1.0]), "optional": None})
    (tmp_path / "committed/unit/bundle.json").write_bytes(b"{")
    with DurableTensorCheckpoint(tmp_path, {}) as store:
        with pytest.raises(CheckpointError, match="checksum"):
            store.load("unit", {})
