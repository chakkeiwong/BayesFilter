"""Refinement must reuse exact nodes and reject stale or invalid baseline evidence."""
import copy
import json
from pathlib import Path

import pytest
import tensorflow as tf

from bayesfilter.testing import surrogate_hmc_m13_posterior_pilot as pilot
from bayesfilter.testing import surrogate_hmc_m13_posterior_refine as refine
from bayesfilter.inference.surrogate_hmc_program import file_hash, write_json


def make_baseline(tmp_path):
    grid = dict(axis=tf.linspace(tf.constant(-pilot.RADIUS, tf.float64),
        tf.constant(pilot.RADIUS, tf.float64), pilot.ORDER).numpy().tolist(),
        log_likelihood=tf.zeros([pilot.ORDER, pilot.ORDER, 8], tf.float64).numpy().tolist(),
        valid=tf.ones([pilot.ORDER, pilot.ORDER, 8], tf.bool).numpy().tolist(),
        seed=list(pilot.SEED))
    write_json(tmp_path/"grid.json", grid)
    write_json(tmp_path/"data.json", dict(test="mechanics"))
    write_json(tmp_path/"report.json", dict(source_sha256={},
        grid_sha256=file_hash(tmp_path/"grid.json"), data_sha256=file_hash(tmp_path/"data.json"),
        status="M13_POSTERIOR_REFERENCE_PILOT_COMPLETED", valid_node_count=pilot.ORDER**2,
        particles=pilot.PARTICLES, replicates=pilot.REPLICATES))
    return grid


def test_nested_nodes_preserve_existing_values_and_mark_new_missing(tmp_path):
    grid = make_baseline(tmp_path)
    before = copy.deepcopy(grid)
    nested = refine.nested_grid(grid, 65)
    assert nested["axis"][::2] == grid["axis"]
    assert nested["log_likelihood"][4][6] == grid["log_likelihood"][2][3]
    assert nested["log_likelihood"][5][6] == [None]*8
    assert nested["valid"][5][6] == [False]*8
    assert grid == before
    with pytest.raises(ValueError, match="successive"):
        refine.nested_grid(grid, 129)


def test_baseline_hash_and_validity_are_required(tmp_path):
    grid = make_baseline(tmp_path)
    _, loaded = refine.load_baseline(tmp_path)
    assert loaded == grid
    grid["valid"][0][0][0] = False
    write_json(tmp_path/"grid.json", grid)
    with pytest.raises(ValueError, match="grid changed"):
        refine.load_baseline(tmp_path)
    report = json.loads((tmp_path/"report.json").read_text())
    report["grid_sha256"] = file_hash(tmp_path/"grid.json")
    write_json(tmp_path/"report.json", report)
    with pytest.raises(ValueError, match="invalid likelihood"):
        refine.load_baseline(tmp_path)


def test_master_refinement_is_within_existing_remaining_reservation():
    from bayesfilter.inference import surrogate_hmc_closure_program as master
    jobs = master.jobs("m13-posterior-refine")
    assert jobs[1]["plan"] == refine.PLAN
    assert jobs[0]["cpu"]+jobs[1]["cpu"] <= 494.18754
    assert jobs[1]["timeout"] <= 545.14822213
    assert "tests/test_surrogate_hmc_m13_posterior_refine.py" in jobs[0]["command"]


def test_worker_calls_refinement(monkeypatch, tmp_path):
    from bayesfilter.testing import surrogate_hmc_closure_diagnostics as worker
    calls = []
    monkeypatch.setattr(refine, "refinement_stage", lambda output: calls.append(output) or
                        dict(status="MECHANICS", qualified=False))
    path = tmp_path/"report.json"
    worker.main(["--stage", "m13-posterior-refine", "--device", "cpu", "--output", str(path)])
    assert calls == [path]
