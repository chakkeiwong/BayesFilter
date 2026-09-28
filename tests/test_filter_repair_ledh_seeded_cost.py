"""Isolated diagnostic before/after LEDH endpoint costs; no scientific claims.

NumPy is used only for fixture construction and independent post-run checks.
The executable candidate callbacks and owner use TensorFlow exclusively.
"""

import gc
import json
import os
import resource
import threading
import time
import weakref
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.highdim.ledh_canonical_filter_tf import (
    canonical_value_and_diagnostics,
    make_seeded_canonical_value_program,
)
from scripts.filter_repair_cost_provenance import GPUProcessMonitor
from tests.test_filter_repair_ledh_seeded_public import (
    _fixture,
    _reference_inputs,
)
from tests.test_filter_repair_ledh_seeded_public import (
    authorities as _authorities,
)
from tests.test_filter_repair_ledh_value_native import (
    BASELINE,
    _compare,
    _frozen_reference,
)


@pytest.fixture(scope="module")
def authorities():
    yield from _authorities.__wrapped__()


def _callbacks_with_frozen_constants(callbacks):
    """Freeze independent fixture constants before timing; no runtime NumPy."""
    q = callbacks.process_noise_covariance
    r = callbacks.observation_covariance
    # Match the independent fixture's float64 constants exactly.
    q_normalizer = tf.constant(2 * np.log(2 * np.pi) + np.linalg.slogdet(q.numpy())[1], tf.float64)
    r_normalizer = tf.constant(2 * np.log(2 * np.pi) + np.linalg.slogdet(r.numpy())[1], tf.float64)

    def density(residual, covariance, normalizer):
        chol = tf.linalg.cholesky(covariance)
        solved = tf.linalg.triangular_solve(
            tf.broadcast_to(chol, [tf.shape(residual)[0], 2, 2]), residual[:, :, None])[:, :, 0]
        return -0.5 * (tf.reduce_sum(tf.square(solved), axis=1) + normalizer)

    return replace(callbacks,
        transition_log_density_fn=lambda points, ancestors, index: density(
            points - callbacks.transition_mean_fn(ancestors, index), q, q_normalizer),
        observation_log_density_fn=lambda points, observation, index: density(
            observation[None, :] - callbacks.observation_fn(points, index), r, r_normalizer))


def _rss():
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("RSS unavailable")


def _sync(result):
    for value in result.values():
        value.numpy()


@pytest.mark.parametrize("arm", ["prior_eager", "candidate_graph", "candidate_xla"])
def test_isolated_seeded_cost(authorities, arm, request):
    baseline, callbacks, observations, controls, source_sha = _fixture(authorities, "dual_trust")
    callbacks = _callbacks_with_frozen_constants(callbacks)
    args = (observations, tf.constant([123], tf.uint32), tf.constant([17], tf.uint32))
    gpu = os.environ.get("BAYESFILTER_TEST_DEVICE_SCOPE") == "visible"
    device = "GPU:0" if gpu else "CPU:0"

    def allocator():
        try:
            return tf.config.experimental.get_memory_info(device)
        except ValueError:
            if gpu:
                raise
            return None

    def sample():
        return {"rss_bytes": _rss(), "allocator": allocator(),
                "process_high_water_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}

    try:
        tf.config.experimental.reset_memory_stats(device)
    except ValueError:
        if gpu:
            raise
    samples = []
    stop = threading.Event()

    def sampler():
        while not stop.wait(.01):
            samples.append(_rss())

    thread = threading.Thread(target=sampler, daemon=True)
    owner = None
    before = sample()
    with GPUProcessMonitor(gpu) as monitor:
        thread.start()
        try:
            start = time.perf_counter()
            if arm == "prior_eager":
                def call(observed):
                    return baseline.canonical_value_and_diagnostics(
                        callbacks, observed, particle_count=8, seed=123, resample_seed=17, **controls)
            else:
                owner = make_seeded_canonical_value_program(
                    callbacks, tf.TensorSpec(observations.shape, tf.float64),
                    particle_count=8, jit_compile=arm == "candidate_xla", **controls)

                def call(observed):
                    return owner(observed, args[1], args[2])

            first = call(observations)
            _sync(first)
            cold = time.perf_counter() - start
            after_cold = sample()
            times = []
            for _ in range(15):
                start = time.perf_counter()
                last = call(observations)
                _sync(last)
                times.append(time.perf_counter() - start)
            after_warm = sample()
            reuse_peak = max([before["rss_bytes"], *samples, after_warm["rss_bytes"]])
            one_shot = []
            if arm != "prior_eager":
                for _ in range(2):
                    start = time.perf_counter()
                    public = canonical_value_and_diagnostics(callbacks, observations,
                        particle_count=8, seed=123, resample_seed=17,
                        jit_compile=arm == "candidate_xla", **controls)
                    _sync(public)
                    one_shot.append(time.perf_counter() - start)
            after_one_shot = sample()
            trace_count = owner.experimental_get_tracing_count() if owner else None
            owner_reference = weakref.ref(owner) if owner else None
            call = None
            owner = None
            gc.collect()
            owner_collected = owner_reference() is None if owner_reference else None
            after_release = sample()
        finally:
            stop.set()
            thread.join(timeout=5)
    assert not thread.is_alive()
    assert samples
    record = {"schema": "filter_repair_ledh_seeded_cost.v1", "arm": arm,
        "device": first["value"].device, "dtype": "float64", "reset_dtype": "float32",
        "shape": {"T": 3, "N": 8, "d": 2}, "seeds": [13, 123, 17],
        "controls": controls, "baseline_commit": BASELINE,
        "baseline_sha256": source_sha, "flow_baseline_sha256": baseline._flow_source_sha256,
        "cold_seconds": cold, "warm_seconds": times, "one_shot_seconds": one_shot,
        "before": before, "after_cold": after_cold, "after_warm": after_warm,
        "after_one_shot": after_one_shot, "after_release": after_release,
        "reuse_sampled_peak_rss_bytes": reuse_peak,
        "overall_sampled_peak_rss_bytes": max(samples), "rss_sample_count": len(samples),
        "trace_count": trace_count, "owner_collected": owner_collected,
        "jit_compile": arm == "candidate_xla",
        "exception": None if arm == "candidate_xla" else "explicit diagnostic/reference",
        "cost_provenance": monitor.payload(),
        "hlo_export": "omitted; endpoint tests cover HLO without contaminating cost samples",
        "validation_outside_cost_measurements": True,
        "nonclaims": ["No statistical speed ranking, capacity, scientific or canonical admission."]}
    directory = Path(request.config.getoption("xmlpath")).parent
    path = directory / "seeded-cost.json"
    path.write_text(json.dumps(record, indent=2) + "\n")
    expected = _frozen_reference(baseline, callbacks, observations,
                                 _reference_inputs(123, 17, tf.float64, stages=1), controls)
    assert bool(expected["numerical_valid"])
    if arm == "prior_eager":
        for key in first:
            np.testing.assert_array_equal(first[key], last[key], err_msg=key)
            if first[key].dtype.is_floating:
                np.testing.assert_allclose(first[key], expected[key], atol=1e-6, rtol=1e-6,
                                           equal_nan=True, err_msg=key)
            else:
                np.testing.assert_array_equal(first[key], expected[key], err_msg=key)
        errors = {"healthy_complete_original_record": True}
    else:
        errors = _compare(first, expected)
        for key in first:
            np.testing.assert_array_equal(first[key], last[key], err_msg=key)
        _compare(public, expected)
        assert trace_count == 1
        assert owner_collected
    record.update(numerical_passed=True, comparison=errors,
                   raw_first={k: v.numpy().tolist() for k, v in first.items() if k != "model_id"})
    path.write_text(json.dumps(record, indent=2) + "\n")
