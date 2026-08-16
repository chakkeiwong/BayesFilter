"""Focused contract tests for the physical replica timing campaign."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHILD = ROOT / "docs/benchmarks/run_ssl_lstm_q20_physical_replica_timing_2026_08_10.py"
SUPERVISOR = ROOT / "docs/benchmarks/run_ssl_lstm_q20_physical_replica_timing_supervisor_2026_08_10.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_child_matches_exact_physical_baseline_and_profiles_cache() -> None:
    child = _load(CHILD, "test_physical_replica_timing_child")
    source = CHILD.read_text(encoding="utf-8")
    assert child.BETAS == (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125)
    assert child.LEAPFROG == 8
    assert child.CHAINS == 2
    assert child.TRANSITIONS_PER_CALL == 1
    assert "make_replica_exchange_fixed_hmc_sampler" in source
    assert "experimental_get_tracing_count" in source
    assert "experimental_get_compiler_ir" in source
    assert "global_runner.THREADS = int(threads)" in source
    assert "historical target helper thread binding failed" in source
    assert 'os.environ["CUDA_VISIBLE_DEVICES"] = "-1"' in source
    assert "terminal accepted target state/status is invalid" in source


def test_supervisor_is_detachable_bounded_sequential_and_atomic() -> None:
    supervisor = _load(SUPERVISOR, "test_physical_replica_timing_supervisor")
    source = SUPERVISOR.read_text(encoding="utf-8")
    assert supervisor.CHILD_CAP_SECONDS == 2400.0
    assert supervisor.CAMPAIGN_CAP_SECONDS == 5000.0
    assert [spec["threads"] for spec in supervisor.TOPOLOGIES] == [4, 32]
    assert supervisor.TOPOLOGIES[0]["cpu_ids"] == tuple(range(32, 36))
    assert supervisor.TOPOLOGIES[1]["cpu_ids"] == tuple(range(32, 64))
    assert "for spec in TOPOLOGIES" in source
    assert "start_new_session=True" in source
    assert "os.killpg" in source
    assert 'with_suffix(absolute.suffix + ".tmp")' in source
    assert "PHYSICAL_REPLICA_TIMING_HARNESS_FAILED" in source
