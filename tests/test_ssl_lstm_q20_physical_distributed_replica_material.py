"""Focused contracts for the bounded physical replica material runner."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import tensorflow as tf


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / (
    "docs/benchmarks/"
    "run_ssl_lstm_q20_physical_distributed_replica_material_2026_08_11.py"
)


def _load_runner():
    name = "test_ssl_lstm_q20_physical_distributed_replica_material_runner"
    spec = importlib.util.spec_from_file_location(name, RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_material_budget_topology_and_diagnostics_are_frozen() -> None:
    runner = _load_runner()
    assert runner.WORKERS == 24
    assert runner.ROWS_PER_WORKER == 1
    assert runner.CHUNK_SIZE == 10
    assert runner.WARMUP_MIN == 300
    assert runner.WARMUP_MAX == 500
    assert runner.WARMUP_WINDOW == 300
    assert runner.WARMUP_RHAT_MAX == 1.05
    assert runner.RETAINED_MIN == 1000
    assert runner.RETAINED_MAX == 1500
    assert runner.RETAINED_RHAT_MAX == 1.01
    assert runner.RETAINED_BULK_ESS_MIN == 1000.0
    assert runner.RETAINED_TAIL_ESS_MIN == 400.0
    assert runner.HARD_WALL_CAP_SECONDS == 8 * 60 * 60
    assert runner.FINALIZATION_RESERVE_SECONDS == 300.0
    config = runner._pool_config()
    assert config.batch_sizes == (1,)
    assert config.batch_per_worker == 1
    assert config.factory_config["jit_compile"] is True


def test_window_round_trips_require_cold_hot_cold_inside_window() -> None:
    runner = _load_runner()
    # One chain, three identities. Identity zero completes cold-hot-cold.
    identities = tf.constant(
        [
            [[0], [1], [2]],
            [[1], [2], [0]],
            [[2], [1], [0]],
            [[0], [1], [2]],
        ],
        tf.int32,
    )
    result = runner._window_round_trips(tf, identities)
    assert int(result["round_trip_returns_by_chain"][0].numpy()) == 1
    assert bool(result["each_chain_has_required_round_trip"].numpy())


def test_adaptive_checks_use_frozen_milestones() -> None:
    runner = _load_runner()
    assert runner.WARMUP_MILESTONES == (300, 350, 400, 450, 500)
    assert runner.RETAINED_MILESTONES == (1000, 1250, 1500)
    assert runner._next_milestone(0, runner.WARMUP_MILESTONES) == 300
    assert runner._next_milestone(300, runner.WARMUP_MILESTONES) == 350
    assert runner._next_milestone(341, runner.WARMUP_MILESTONES) == 350
    assert runner._next_milestone(1000, runner.RETAINED_MILESTONES) == 1250
    assert runner._next_milestone(1500, runner.RETAINED_MILESTONES) == 1500


def test_material_runner_preserves_warmup_retained_and_nonclaims() -> None:
    runner = _load_runner()
    source = RUNNER.read_text(encoding="utf-8")
    assert runner.CHECKPOINT_SHA256 == (
        "8276947db5785786567c5194b469c0938907820faf8d1bafd0265b1d4f87adab"
    )
    assert runner.MATERIALITY_SHA256 == (
        "5de1e5d217abd9ae293aff81356955c799ed6328e6a66670b019220f6d27aad2"
    )
    assert "state_rows[warmup_count:]" in source
    assert "rank_normalized_hmc_diagnostics" in source
    assert "invalid_hmc_path_was_accepted" in source
    assert "swap_matrix_is_permutation" in source
    assert "retained_state_target_score_finite" in source
    assert "occupancy_role" in source
    assert "two-region travel does not prove exhaustive mode discovery" in source
    assert "launch_source_sha256" in source
    assert "launch_git_commit" in source
    assert "import numpy" not in source
    assert 'with_suffix(absolute.suffix + ".tmp")' in source
