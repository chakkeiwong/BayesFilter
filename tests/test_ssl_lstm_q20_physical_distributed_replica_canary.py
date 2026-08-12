"""Focused contracts for the exact-target distributed replica canary."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "docs/benchmarks/run_ssl_lstm_q20_physical_distributed_replica_canary_2026_08_10.py"


def _load_runner():
    name = "test_ssl_lstm_q20_physical_distributed_replica_canary_runner"
    spec = importlib.util.spec_from_file_location(name, RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_canary_matches_physical_baseline_and_distributes_all_rows() -> None:
    runner = _load_runner()
    assert runner.BETAS == (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125)
    assert runner.LEAPFROG == 8
    assert runner.CHAINS == 2
    assert runner.ROWS == runner.WORKERS == 12
    assert runner.WORKER_CPU_IDS == tuple(range(32, 44))
    assert runner.OUTPUT_ROOT.as_posix().endswith("/r3-distributed-canary")
    assert runner.FAILED_R2_SHA256 == (
        "bfc3b2a4b4afcea87010cbf434d21b911171dfa155e86e8f979f799ac9c6b30f"
    )
    config = runner._pool_config()
    assert config.worker_count == 12
    assert config.batch_sizes == (1,)
    assert config.batch_per_worker == 1
    assert config.factory_config["jit_compile"] is True
    assert config.worker_cpu_ids == tuple(range(32, 44))


def test_canary_is_cpu_xla_bounded_atomic_and_proposal_safe() -> None:
    runner = _load_runner()
    source = RUNNER.read_text(encoding="utf-8")
    assert runner.CAP_SECONDS == 3600.0
    assert 'os.environ["CUDA_VISIBLE_DEVICES"] = "-1"' in source
    assert "distributed_replica_exchange_transition" in source
    assert "terminal-independent-check" in source
    assert '["log_prob"]' in source
    assert '["value"]' not in source
    assert "invalid_proposal_paths_self_rejected" in source
    assert "log_acceptance_finite_or_forced_negative_infinity" in source
    assert 'with_suffix(absolute.suffix + ".tmp")' in source
    assert "DISTRIBUTED_REPLICA_CANARY_HARNESS_FAILED" in source
    assert "np." not in source
    assert "import numpy" not in source
