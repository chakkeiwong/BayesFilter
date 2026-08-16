from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import pytest
import tensorflow as tf


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / (
    "docs/benchmarks/"
    "run_ssl_lstm_q20_fixed_hmc_api_cpu_xla_validation_2026_08_02.py"
)
SPEC = importlib.util.spec_from_file_location("q20_fixed_hmc_api_campaign", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
CAMPAIGN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CAMPAIGN)


def _worker_row(chain_index: int, *, draws: int = 4, dimension: int = 2):
    base = tf.reshape(
        tf.range(draws * dimension, dtype=tf.float64), (draws, dimension)
    )
    base = base + tf.cast(chain_index * 100, tf.float64)
    scalar = tf.range(draws, dtype=tf.float64) + tf.cast(chain_index * 10, tf.float64)
    return {
        "chain_index": chain_index,
        "samples": base,
        "trace": {
            "is_accepted": tf.ones((draws,), tf.bool),
            "log_accept_ratio": -scalar,
            "target_log_prob": scalar,
            "proposed_target_log_prob": scalar + 0.5,
            "target_score": base + 0.25,
            "delta_h": scalar,
            "target_status_code": tf.zeros((draws,), tf.int32),
            "target_valid_pre_regularized_score": tf.ones((draws,), tf.bool),
        },
    }


def test_chain_seed_folding_is_deterministic_and_disjoint() -> None:
    seeds = tuple(CAMPAIGN._fold_chain_seed((17, 23), index) for index in range(4))
    assert seeds == tuple(
        CAMPAIGN._fold_chain_seed((17, 23), index) for index in range(4)
    )
    assert len(set(seeds)) == 4
    with pytest.raises(CAMPAIGN.CampaignError, match="outside"):
        CAMPAIGN._fold_chain_seed((17, 23), 4)


def test_worker_rows_reassemble_exactly_on_chain_axis() -> None:
    rows = tuple(_worker_row(index) for index in reversed(range(4)))
    samples, trace = CAMPAIGN._reassemble_worker_chunk(
        rows, active_results=4, dimension=2
    )
    assert tuple(samples.shape) == (4, 4, 2)
    assert tuple(trace["target_score"].shape) == (4, 4, 2)
    assert tuple(trace["log_accept_ratio"].shape) == (4, 4)
    for chain_index in range(4):
        expected = _worker_row(chain_index)
        tf.debugging.assert_equal(samples[:, chain_index, :], expected["samples"])
        tf.debugging.assert_equal(
            trace["target_score"][:, chain_index, :],
            expected["trace"]["target_score"],
        )


def test_admitted_kernel_loader_fails_closed_and_preserves_payload(tmp_path) -> None:
    chart = tmp_path / "chart-a"
    chart.mkdir()
    kernel = {
        "step_size": 0.2,
        "num_leapfrog_steps": 2,
        "mass_policy": "fixed_identity_z",
        "use_xla": True,
        "shared_scalar_step_across_chain_bank": True,
    }
    (chart / "summary.json").write_text(
        json.dumps({"status": "KERNEL_ADMITTED"}), encoding="utf-8"
    )
    (chart / "tuning-result.json").write_text(
        json.dumps({"passed": True, "final_kernel_payload": kernel}),
        encoding="utf-8",
    )
    assert CAMPAIGN._load_admitted_kernel(tmp_path, "chart-a") == kernel

    kernel["num_leapfrog_steps"] = 1
    (chart / "tuning-result.json").write_text(
        json.dumps({"passed": True, "final_kernel_payload": kernel}),
        encoding="utf-8",
    )
    with pytest.raises(CAMPAIGN.CampaignError, match="at least two"):
        CAMPAIGN._load_admitted_kernel(tmp_path, "chart-a")


def test_reviewed_sequential_chunk_size_divides_all_policy_boundaries() -> None:
    chunk = CAMPAIGN.SEQUENTIAL_CHUNK_RESULTS
    assert chunk == 40
    for total in (1000, 2000, 10000):
        assert total % chunk == 0
