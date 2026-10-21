"""Mechanics and validity checks for nested same-data grid reuse."""
import copy
import json
import pytest
import tensorflow as tf
from bayesfilter.inference.surrogate_hmc_program import file_hash, write_json
from bayesfilter.inference.surrogate_hmc_closure_program import PASS_CAPS, jobs
from bayesfilter.testing.surrogate_hmc_m13_same_data_refine import (
    PLAN, RADIUS, checked_grid, refine_grid,
)


def fixture_grid():
    axis = [-RADIUS+i*RADIUS/2 for i in range(5)]
    return dict(axis=axis, log_likelihood=[[[x+y+k for k in range(8)] for y in axis] for x in axis],
        complete=True, invalid_nodes=[], data_seed=1704, observation_checksum="fixture",
        particles=4096, seed=[31, 32], replicates=8)


def test_refinement_calls_only_new_nodes_and_preserves_failures():
    grid = fixture_grid()
    old = copy.deepcopy(grid)
    calls, rows = [], []

    def evaluate(theta, seed):
        calls.append(tuple(theta.numpy().tolist()))
        value = tf.reduce_sum(theta)+tf.range(8, dtype=tf.float64)
        return dict(log_likelihood=value, valid=tf.fill([8], len(calls) != 3),
                    minimum_ess=tf.ones([8], tf.float64)*100.,
                    proposal_maximum_normalized_weight=tf.ones([8], tf.float64)*.02)

    result = refine_grid(grid, evaluate, lambda i, records, invalid: rows.extend(records))
    assert grid == old
    assert len(calls) == len(set(calls)) == len(rows) == 81-25
    assert result["reused_evaluations"] == 25
    assert result["new_evaluations"] == 56
    assert len(result["invalid_nodes"]) == 1
    assert len([r for r in rows if not all(r["valid"])]) == 1
    for i, row in enumerate(old["log_likelihood"]):
        for j, value in enumerate(row):
            assert result["log_likelihood"][2*i][2*j] == value
    for i, x in enumerate(result["axis"]):
        for j, y in enumerate(result["axis"]):
            assert result["log_likelihood"][i][j] == pytest.approx([x+y+k for k in range(8)])


def test_saved_grid_integrity_and_scope(tmp_path):
    path = tmp_path/"grid.json"
    grid = fixture_grid()
    write_json(path, grid)
    kwargs = dict(expected_hash=file_hash(path), checksum="fixture", particles=4096, seed=(31, 32), order=5)
    assert checked_grid(path, **kwargs) == grid
    changed = copy.deepcopy(grid)
    changed["log_likelihood"][0][0][0] += .1
    write_json(path, changed)
    with pytest.raises(ValueError, match="checksum"):
        checked_grid(path, **kwargs)
    for key, value in [("data_seed", 1701), ("particles", 16384), ("seed", [31, 33]),
                       ("complete", False), ("invalid_nodes", [[0, 0]]),
                       ("log_likelihood", grid["log_likelihood"][:-1]),
                       ("axis", [-RADIUS]*5)]:
        changed = copy.deepcopy(grid)
        changed[key] = value
        write_json(path, changed)
        with pytest.raises(ValueError):
            checked_grid(path, **(kwargs | dict(expected_hash=file_hash(path))))


def test_master_refinement_budget_and_plan():
    selected = jobs("m13-same-data-refine")
    assert [x["name"] for x in selected] == ["review", "m13-same-data-refine"]
    assert selected[1]["plan"] == PLAN
    assert selected[1]["device"] == "gpu"
    assert selected[1]["cpu"] == 4000.
    assert PASS_CAPS["m13-same-data-refine"] == dict(cpu_worker_seconds=4500., gpu_seconds=5000.)
