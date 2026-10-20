"""Crash-only instrumentation of the unchanged frozen D3 prior reuse fixture."""

import ctypes
import hashlib
import json
import os
import time
from pathlib import Path

import tensorflow as tf

from tests import test_filter_repair_posterior_initializer_capacity as capacity
from tests.test_filter_repair_geometry_control import clean


def test_frozen_prior_reuse_native_stack(request, monkeypatch):
    directory = Path(request.config.getoption("xmlpath")).parent
    progress = directory / "prior-capacity-progress.jsonl"
    assert not progress.exists()
    names = ("arena", "ordblks", "smblks", "hblks", "hblkhd", "usmblks",
        "fsmblks", "uordblks", "fordblks", "keepcost")

    class Mallinfo(ctypes.Structure):
        _fields_ = [(name, ctypes.c_size_t) for name in names]

    library = ctypes.CDLL(None)
    malloc_info = getattr(library, "mallinfo2", None)
    if malloc_info is not None:
        malloc_info.argtypes = []
        malloc_info.restype = Mallinfo

    original_loader = capacity.original_module
    completed = []
    _, expected_covariance, *_ = capacity.fixture(3)
    expected_mean = .13 + .03 * tf.cast(tf.range(3), tf.float64)

    def record(stage, extra=None):
        info = malloc_info() if malloc_info is not None else None
        payload = {"stage": stage, "completed_calls": len(completed),
            "monotonic_seconds": time.monotonic(), "pid": os.getpid(),
            "memory": capacity.snapshot(False),
            "mallinfo2": {name: getattr(info, name) for name in names} if info is not None else None,
            **(extra or {})}
        with progress.open("a") as stream:
            stream.write(json.dumps(payload, allow_nan=False) + "\n")
            stream.flush()

    def load_reference():
        reference, hashes = original_loader()
        original_endpoint = reference.initialize_posterior_local_location_scale
        original_payload = reference.PosteriorLocalInitializerResult.payload

        def observe_endpoint(*args, **kwargs):
            record("before_public_call")
            return original_endpoint(*args, **kwargs)

        def observe_payload(self, *args, **kwargs):
            value = original_payload(self, *args, **kwargs)
            serialized = json.dumps(clean(value), sort_keys=True, allow_nan=False).encode()
            path = directory / f"completed-call-{len(completed):02d}.json"
            path.write_bytes(serialized + b"\n")
            previous = completed[len(completed) % 2] if len(completed) >= 2 else None
            replay = previous is None or previous == value
            completed.append(value)
            record("after_public_payload", {"result_file": path.name,
                "result_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "accepted": value["accepted"], "exact_replay": replay})
            assert value["accepted"] and replay
            tf.debugging.assert_near(tf.constant(value["center"], tf.float64), expected_mean, atol=1e-7, rtol=1e-7)
            tf.debugging.assert_near(tf.constant(value["covariance_theta"], tf.float64), expected_covariance, atol=1e-7, rtol=1e-7)
            return value

        monkeypatch.setattr(reference, "initialize_posterior_local_location_scale", observe_endpoint)
        monkeypatch.setattr(reference.PosteriorLocalInitializerResult, "payload", observe_payload)
        return reference, hashes

    monkeypatch.setattr(capacity, "original_module", load_reference)
    capacity.test_complete_initializer_twenty_call_reuse("prior", 3, request, monkeypatch)
