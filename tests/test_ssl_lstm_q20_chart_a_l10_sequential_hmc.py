from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest
import tensorflow as tf


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / (
    "docs/benchmarks/"
    "run_ssl_lstm_q20_chart_a_l10_sequential_hmc_2026_08_04.py"
)
PLAN = ROOT / (
    "docs/plans/"
    "bayesfilter-ssl-lstm-q20-chart-a-l10-sequential-hmc-plan-2026-08-04.md"
)


def _load_runner():
    name = "q20_chart_a_l10_sequential_hmc_runner"
    specification = importlib.util.spec_from_file_location(name, SCRIPT)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


RUNNER = _load_runner()


def _worker_row(chain_index: int, *, draws: int = 4):
    samples = tf.reshape(
        tf.range(draws * RUNNER.DIMENSION, dtype=tf.float64),
        (draws, RUNNER.DIMENSION),
    ) + tf.cast(chain_index * 100, tf.float64)
    scalar = tf.range(draws, dtype=tf.float64) + tf.cast(
        chain_index * 10, tf.float64
    )
    return {
        "chain_index": chain_index,
        "samples": samples,
        "trace": {
            "is_accepted": tf.ones((draws,), tf.bool),
            "log_accept_ratio": -scalar,
            "target_log_prob": scalar,
            "proposed_target_log_prob": scalar + 0.5,
            "target_score": samples + 0.25,
            "delta_h": scalar,
            "target_status_code": tf.zeros((draws,), tf.int32),
            "target_valid_pre_regularized_score": tf.ones((draws,), tf.bool),
        },
    }


def test_static_contract_is_policy_scale_and_cpu_only() -> None:
    assert RUNNER.POLICY_ID == "bayesfilter_neutra_sequential_hmc_v1"
    assert RUNNER.CHUNK_RESULTS == 500
    assert RUNNER.CHAIN_COUNT == 4
    assert RUNNER.THREADS_PER_CHAIN == 8
    assert RUNNER.SUPERVISOR_CPU == 32
    assert RUNNER.DEFAULT_CAP_SECONDS == pytest.approx(86400.0)
    assert tuple(cpu for group in RUNNER.CHAIN_CPUS for cpu in group) == tuple(
        range(32)
    )
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "-1"


def test_launcher_calls_shared_sequential_controller_and_has_no_numpy() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "run_sequential_neutra_hmc(" in source
    assert "NEUTRA_SEQUENTIAL_HMC_POLICY_ID" in source
    assert "result.payload()" not in source
    assert '"schema": "bayesfilter.neutra.sequential_hmc_result.v1"' in source
    assert "tfp.mcmc.sample_chain" not in source
    assert "import numpy" not in source
    assert "np." not in source
    assert source.index('os.environ["CUDA_VISIBLE_DEVICES"] = "-1"') < source.index(
        "def _configure_tensorflow"
    )


def test_plan_contains_required_evidence_and_audit_sections() -> None:
    text = PLAN.read_text(encoding="utf-8")
    for heading in (
        "## Research Intent Ledger",
        "## Evidence Contract",
        "## Default And Assumption Audit",
        "## Pre-Mortem",
        "## Skeptical Pre-Execution Audit",
    ):
        assert heading in text
    assert "2,000" in text
    assert "1,000" in text
    assert "bulk ESS `>=400`" in text
    assert "tail ESS `>=400`" in text
    assert "The sequential minima are never reduced" in text


def test_chain_seed_folding_is_deterministic_and_disjoint() -> None:
    first = tuple(RUNNER._fold_chain_seed((17, 23), i) for i in range(4))
    second = tuple(RUNNER._fold_chain_seed((17, 23), i) for i in range(4))
    assert first == second
    assert len(set(first)) == 4
    with pytest.raises(RUNNER.CampaignError, match="outside"):
        RUNNER._fold_chain_seed((17, 23), 4)


def test_worker_rows_reassemble_exactly_with_actual_status_trace() -> None:
    rows = tuple(_worker_row(index) for index in reversed(range(4)))
    samples, trace = RUNNER._reassemble_worker_chunk(rows, active_results=4)
    assert tuple(samples.shape) == (4, 4, 4)
    assert tuple(trace["target_score"].shape) == (4, 4, 4)
    assert tuple(trace["target_status_code"].shape) == (4, 4)
    assert tuple(trace["target_valid_pre_regularized_score"].shape) == (4, 4)
    for chain_index in range(4):
        expected = _worker_row(chain_index)
        tf.debugging.assert_equal(samples[:, chain_index], expected["samples"])


def test_forecast_requires_margin_and_archive_reserve() -> None:
    assert RUNNER._forecast_allows_next_chunk(
        elapsed_seconds=100.0,
        cap_seconds=1000.0,
        completed_chunk_seconds=(100.0, 120.0),
    )
    assert not RUNNER._forecast_allows_next_chunk(
        elapsed_seconds=300.0,
        cap_seconds=1000.0,
        completed_chunk_seconds=(100.0, 120.0),
    )


def test_kernel_loader_rejects_identity_drift(tmp_path, monkeypatch) -> None:
    source = json.loads(RUNNER.TUNING_ARTIFACT.read_text(encoding="utf-8"))
    path = tmp_path / "candidate.json"
    path.write_bytes(RUNNER._canonical_bytes(source))
    monkeypatch.setattr(RUNNER, "EXPECTED_TUNING_SHA256", RUNNER._sha256(path))
    assert RUNNER.load_frozen_kernel(path)["num_leapfrog_steps"] == 10

    source["final_kernel_payload"]["num_leapfrog_steps"] = 1
    path.write_bytes(RUNNER._canonical_bytes(source))
    monkeypatch.setattr(RUNNER, "EXPECTED_TUNING_SHA256", RUNNER._sha256(path))
    with pytest.raises(RUNNER.CampaignError, match="kernel hash mismatch"):
        RUNNER.load_frozen_kernel(path)


def test_material_cli_defaults_to_reviewed_cap() -> None:
    args = RUNNER.parse_args(["--mode", "run"])
    assert args.cap_seconds == pytest.approx(86400.0)
    with pytest.raises(SystemExit):
        RUNNER.parse_args(["--mode", "worker"])
