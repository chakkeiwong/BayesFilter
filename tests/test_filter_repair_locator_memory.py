"""Fresh-process capacity diagnostic for optional compiled progress storage."""

import gc
import json
import resource
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.inference.sequential_batched_locator_tf import BufferedBatchedLocator
from tests.test_filter_repair_batched_locator import _batch
from tests.test_filter_repair_sequential_locator import _target

D = tf.float64


def _snapshot():
    status = dict(line.split(":", 1) for line in Path("/proc/self/status").read_text().splitlines())
    return {"gpu": tf.config.experimental.get_memory_info("GPU:0"),
        "host_rss_bytes": int(status["VmRSS"].split()[0]) * 1024,
        "host_proc_hwm_bytes": int(status["VmHWM"].split()[0]) * 1024,
        "host_resource_hwm_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}


def _observations(result):
    numeric = {name: value.numpy().tolist() for name, value in result.items()
        if name not in ("trace", "selected")}
    numeric["selected"] = {name: value.numpy().tolist() for name, value in result["selected"].items()}
    numeric["trace"] = result["trace"].numpy()[:int(result["trace_count"])].tolist()
    return numeric


@pytest.mark.parametrize("count,capacity", [(2, 128), (2, 4096), (4, 128), (4, 4096)])
def test_buffer_capacity_memory_and_complete_observations(count, capacity):
    assert tf.config.list_physical_devices("GPU"), "this capacity diagnostic requires the recorded GPU lane"
    starts = tf.reshape(tf.linspace(tf.constant(-.4, D), tf.constant(.3, D), 2 * count), [count, 2])
    scale = tf.constant([.7, 1.3], D)
    int(tf.size(starts))
    stages = {"before_construction": _snapshot()}
    owner = BufferedBatchedLocator(_target("quadratic"), _batch("quadratic"),
        count, 2, 4., 1e-8, 4, 7, "converged_all", device=starts.device, capacity=capacity)
    assert owner.rows.device == owner.calls.device == starts.device
    stages["constructed"] = _snapshot()
    concrete = owner.compiled.get_concrete_function()
    graph = concrete.graph.as_graph_def()
    graph_nodes = len(graph.node) + sum(len(function.node_def) for function in graph.library.function)
    stages["traced"] = _snapshot()
    result = owner(starts, scale)
    stages["cold_outputs_live"] = _snapshot()
    reference = _observations(result)
    assert 0 < reference["trace_count"] == reference["objective_calls"] < capacity
    assert not reference["trace_overflow"]
    del result
    gc.collect()
    stages["cold_outputs_released"] = _snapshot()
    tf.config.experimental.reset_memory_stats("GPU:0")
    warm = []
    for _ in range(20):
        result = owner(starts, scale)
        live = _snapshot()
        observed = _observations(result)
        assert observed == reference
        del result
        warm.append({"live": live, "released": _snapshot()})
    current = [value["released"]["gpu"]["current"] for value in warm]
    assert current[-1] <= max(current[:5])
    np.testing.assert_allclose(np.diff(current[5:]), 0, atol=0, rtol=0)
    report = {"role": "explanatory_progress_capacity_memory_only", "capacity": capacity,
        "starts": count, "dimension": 2, "buffer_numerical_bytes": capacity * count * 5 * 8,
        "jit_compile": True, "graph_nodes": graph_nodes, "stages": stages,
        "warm_samples": warm, "observations": reference,
        "limitations": ["One fresh process per capacity and extent; no terminal repeat claim.",
            "Allocator peaks include resources, temporaries and returned snapshots; not live tensor accounting.",
            "All capacities use identical observation materialization and output lifetimes."]}
    print("LOCATOR_CAPACITY_MEMORY " + json.dumps(report, sort_keys=True))
