"""Focused harness tests for the material SSL-LSTM q=20 AIS campaign."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "docs/benchmarks/run_ssl_lstm_q20_physical_ais_material_2026_08_10.py"


def _load_runner():
    name = "test_ssl_lstm_q20_physical_ais_material_runner"
    spec = importlib.util.spec_from_file_location(name, RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_material_tasks_partition_one_hundred_cpus_without_overlap() -> None:
    runner = _load_runner()
    base = runner._load_base()
    tasks = runner._tasks(
        base,
        {"means": (), "precisions": ()},
        family="central",
        batch_index=0,
        num_steps=base.MATERIAL_STEPS,
    )
    assert len(tasks) == 25
    cpu_ids = [cpu_id for task in tasks for cpu_id in task["cpu_ids"]]
    assert cpu_ids == list(range(100))
    assert all(len(task["cpu_ids"]) == 4 for task in tasks)
    assert all(task["rejuvenation_interval"] == 8 for task in tasks)


def test_material_seed_streams_are_unique_and_family_disjoint() -> None:
    runner = _load_runner()
    base = runner._load_base()
    proposal = {"means": (), "precisions": ()}
    task_groups = [
        runner._tasks(
            base,
            proposal,
            family="central",
            batch_index=batch_index,
            num_steps=base.MATERIAL_STEPS,
        )
        for batch_index in range(base.CENTRAL_BATCHES)
    ] + [
        runner._tasks(
            base,
            proposal,
            family="sensitivity",
            batch_index=batch_index,
            num_steps=base.SENSITIVITY_STEPS,
        )
        for batch_index in range(base.SENSITIVITY_BATCHES)
    ]
    tasks = [task for group in task_groups for task in group]
    proposal_seeds = [task["seed"] for task in tasks]
    ais_seeds = [task["ais_seed"] for task in tasks]
    assert len(set(proposal_seeds)) == 250
    assert len(set(ais_seeds)) == 250
    assert set(proposal_seeds).isdisjoint(ais_seeds)


def test_material_runner_uses_versioned_non_canary_output() -> None:
    runner = _load_runner()
    assert runner.OUTPUT_ROOT.as_posix().endswith("/r3")
    assert runner.FINAL == runner.OUTPUT_ROOT / "material.json"
    assert runner.RUNNER_STOP_SECONDS < 7200.0
