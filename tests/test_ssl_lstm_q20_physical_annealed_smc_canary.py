"""Focused harness tests for the SSL-LSTM q=20 annealed-SMC canary."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "docs/benchmarks/run_ssl_lstm_q20_physical_annealed_smc_canary_2026_08_10.py"


def _load_runner():
    name = "test_ssl_lstm_q20_physical_annealed_smc_runner"
    spec = importlib.util.spec_from_file_location(name, RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_worker_tasks_cover_one_hundred_cpus_and_particles() -> None:
    runner = _load_runner()
    tasks = runner._base_tasks({"means": (), "precisions": ()}, "mutate")
    assert len(tasks) == 25
    assert runner.WORKERS * runner.ROWS_PER_WORKER == runner.PARTICLES == 100
    cpu_ids = [cpu_id for task in tasks for cpu_id in task["cpu_ids"]]
    assert cpu_ids == list(range(100))
    assert all(len(task["cpu_ids"]) == 4 for task in tasks)


def test_seed_domains_are_disjoint_for_all_declared_stages() -> None:
    runner = _load_runner()
    initial = {(20260810, 21000 + worker) for worker in range(runner.WORKERS)}
    resampling = {
        (20260810, 22000 + stage) for stage in range(runner.MAX_STAGES)
    }
    mutation = {
        (20260810, 23000 + stage * runner.WORKERS + worker)
        for stage in range(runner.MAX_STAGES)
        for worker in range(runner.WORKERS)
    }
    assert len(initial) == 25
    assert len(resampling) == 24
    assert len(mutation) == 600
    assert initial.isdisjoint(resampling)
    assert initial.isdisjoint(mutation)
    assert resampling.isdisjoint(mutation)


def test_canary_contract_is_global_bounded_and_terminal_preserving() -> None:
    runner = _load_runner()
    source = RUNNER.read_text(encoding="utf-8")
    assert runner.TARGET_ESS_FRACTION == 0.80
    assert runner.MAX_STAGES == 24
    assert runner.RUNNER_CAP_SECONDS < 3600.0
    assert "systematic_resample_indices" in source
    assert '"terminal_pre_resampling": terminal' in source
    assert 'if terminal:' in source
    terminal_block = source.split("if terminal:", maxsplit=1)[1].split(
        "parents = systematic_resample_indices", maxsplit=1
    )[0]
    assert '"resampled": False' in terminal_block
    assert "systematic_resample_indices" not in terminal_block


def test_stage_receipts_keep_pre_and_post_namespaces_separate() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert '"receipts": {"pre": pre_receipts, "post": {}}' in source
    assert 'stage_payload["receipts"]["post"]' in source
    assert "receipts.update" not in source
    assert "physical_annealed_smc.stage.v2" in source


def test_worker_wave_forces_fresh_process_before_tensorflow_import() -> None:
    runner = _load_runner()
    source = RUNNER.read_text(encoding="utf-8")
    assert "max_tasks_per_child=1" in source
    assert source.index("os.sched_setaffinity") < source.index("import tensorflow as tf")
    assert 'tuple(row["actual_cpu_ids"]) != tuple(row["cpu_ids"])' in source
