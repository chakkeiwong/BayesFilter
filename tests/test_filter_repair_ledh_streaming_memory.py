"""Diagnostic cost/capacity fixtures; NumPy checks are outside timed kernels."""

import gc
import json
import os
import time
import weakref
from pathlib import Path

import pytest
import tensorflow as tf

import bayesfilter.highdim.ledh_canonical_filter_tf as current
import tests.test_filter_repair_ledh_seeded_cost as cost
from scripts.filter_repair_cost_provenance import GPUProcessMonitor
from tests.test_filter_repair_ledh_streaming import (
    BUFFERED_COMMIT,
    _graph_evidence,
    _numeric,
    _records_compare,
    authorities,
    buffered_authority,
)

# Re-export fixture registrations for isolated pytest nodes.
__all__ = ["authorities", "buffered_authority"]


@pytest.mark.parametrize("arm", ["buffered_xla", "streaming_graph", "streaming_xla"])
def test_streaming_owner_cost(authorities, buffered_authority, arm, request, monkeypatch):
    module = buffered_authority[0] if arm == "buffered_xla" else current
    factory = module.make_seeded_canonical_value_program
    references = []

    def observed_factory(*args, **kwargs):
        owner = factory(*args, **kwargs)
        references.append(weakref.ref(owner))
        return owner

    monkeypatch.setattr(module, "make_seeded_canonical_value_program", observed_factory)
    monkeypatch.setattr(cost, "make_seeded_canonical_value_program", observed_factory)
    monkeypatch.setattr(cost, "canonical_value_and_diagnostics", module.canonical_value_and_diagnostics)
    cost.test_isolated_seeded_cost(authorities,
        "candidate_graph" if arm == "streaming_graph" else "candidate_xla", request)
    gc.collect()
    collected = [reference() is None for reference in references]
    assert collected == [True, True, True]
    path = Path(request.config.getoption("xmlpath")).parent / "seeded-cost.json"
    record = json.loads(path.read_text())
    record.update(implementation_arm=arm, buffered_commit=BUFFERED_COMMIT,
                  all_three_owners_collected=collected,
                  owner_reuse_calls=15, extra_fresh_owners=2)
    path.write_text(json.dumps(record, indent=2) + "\n")


@pytest.mark.parametrize("arm", ["buffered", "streaming"])
@pytest.mark.parametrize("horizon,count", [(3, 8), (3, 64), (32, 64), (128, 64)])
def test_streaming_capacity(authorities, buffered_authority, arm, horizon, count, request):
    _, fixture, _ = authorities
    model = fixture._lgssm_model(13, horizon=horizon)
    callbacks = cost._callbacks_with_frozen_constants(fixture._callbacks_for_lgssm(model))
    observations = tf.constant(model["observations"], tf.float64)
    spec = tf.TensorSpec(observations.shape, tf.float64)
    args = (observations, tf.constant([123], tf.uint32), tf.constant([17], tf.uint32))
    controls = {"flow_substeps": 3, "sinkhorn_steps": 2, "balance_steps": 2,
                "dual_cap_enabled": True, "trust_region_enabled": True}
    gpu = os.environ.get("BAYESFILTER_TEST_DEVICE_SCOPE") == "visible"
    device = "GPU:0" if gpu else "CPU:0"

    def sample():
        try:
            allocator = tf.config.experimental.get_memory_info(device)
        except ValueError:
            if gpu:
                raise
            allocator = None
        return {"rss_bytes": cost._rss(), "allocator": allocator}

    try:
        tf.config.experimental.reset_memory_stats(device)
    except ValueError:
        if gpu:
            raise
    directory = Path(request.config.getoption("xmlpath")).parent
    module = buffered_authority[0] if arm == "buffered" else current
    before = sample()
    with GPUProcessMonitor(gpu) as monitor:
        start = time.perf_counter()
        owner = module.make_seeded_canonical_value_program(callbacks, spec, particle_count=count, **controls)
        value = owner(*args)
        cost._sync(value)
        cold = time.perf_counter() - start
        after_cold = sample()
        times = []
        for _ in range(3):
            start = time.perf_counter()
            repeated = owner(*args)
            cost._sync(repeated)
            times.append(time.perf_counter() - start)
        after_warm = sample()
    graph = _graph_evidence(owner, args, directory, f"{arm}-{horizon}-{count}", horizon, count)
    reference = weakref.ref(owner)
    del owner
    gc.collect()
    report = {"shape": {"T": horizon, "N": count, "d": 2}, "buffered_commit": BUFFERED_COMMIT,
        "buffered_process_bytes": horizon * count * 2 * 8,
        "streaming_draw_bytes": count * 2 * 8, "philox_state_bytes": 24,
        "arm": arm, "before": before, "after_cold": after_cold, "after_warm": after_warm,
        "cold_seconds": cold, "warm_seconds": times, "record": _numeric(value),
        "device": value["value"].device, "cost_provenance": monitor.payload(), "graph": graph,
        "owner_collected": reference() is None, "after_export_and_release": sample(),
        "fresh_process_per_arm_and_point": True,
        "nonclaims": ["After-export memory includes compiler IR inspection.",
                      "Rejected records cannot establish numerical speed ranking or scientific capacity."]}
    path = directory / f"capacity-{horizon}-{count}-{arm}.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    assert reference() is None
    assert bool(graph["process_buffer_shape_nodes"]) == (arm == "buffered")
    assert (graph["hlo"]["optimized_hlo"]["process_shape_occurrences"] > 0) == (arm == "buffered")
    # Build the other authority only after all primary memory/cost samples.
    other = current if arm == "buffered" else buffered_authority[0]
    comparison_owner = other.make_seeded_canonical_value_program(callbacks, spec, particle_count=count, **controls)
    comparison = comparison_owner(*args)
    report["comparison"] = _records_compare(comparison, value) if arm == "buffered" else _records_compare(value, comparison)
    path.write_text(json.dumps(report, indent=2) + "\n")
