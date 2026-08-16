"""Focused contracts for the four-chain distributed replica checkpoint."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "docs/benchmarks/run_ssl_lstm_q20_physical_distributed_replica_checkpoint_2026_08_10.py"


def _load_runner():
    name = "test_ssl_lstm_q20_physical_distributed_replica_checkpoint_runner"
    spec = importlib.util.spec_from_file_location(name, RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_checkpoint_freezes_four_chain_kernel_topology_and_budget() -> None:
    runner = _load_runner()
    assert runner.CHAINS == 4
    assert runner.ROWS == runner.WORKERS == 24
    assert runner.TRANSITIONS == 25
    assert runner.CHUNK_SIZE == 5
    assert runner.LEAPFROG == 8
    assert runner.WORKER_CPU_IDS == tuple(range(32, 56))
    assert runner.CAP_SECONDS == 900.0
    assert runner.MATERIAL_MINIMUM_TRANSITIONS == 1300
    assert runner.PROJECTION_MARGIN == 1.5
    assert runner.ACCEPTANCE_LOWER == 0.35
    assert runner.ACCEPTANCE_UPPER == 0.99
    config = runner._pool_config()
    assert config.worker_count == 24
    assert config.batch_sizes == (1,)
    assert config.batch_per_worker == 1
    assert config.factory_config["jit_compile"] is True


def test_checkpoint_is_chunked_atomic_global_travel_and_nonpromotional() -> None:
    _load_runner()
    source = RUNNER.read_text(encoding="utf-8")
    assert "for transition_index in range(TRANSITIONS)" in source
    assert "replica_travel_diagnostics(identities)" in source
    assert "chunk terminal cache parity failed" in source
    assert "every_swap_matrix_is_permutation" in source
    assert "log_acceptance_finite_or_invalid_path_negative_infinity" in source
    assert "material_projection_within_20000_seconds" in source
    assert "sum(transition_wall) + sum(cache_check_wall)" in source
    assert 'with_suffix(absolute.suffix + ".tmp")' in source
    assert "FOUR_CHAIN_CHECKPOINT_HARNESS_FAILED" in source
    assert "no convergence, posterior, mass" in source
    assert "import numpy" not in source
