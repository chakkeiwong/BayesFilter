from __future__ import annotations

import importlib.util
import json
import os
import sys
from argparse import Namespace
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import pytest
import tensorflow as tf


ROOT = Path(__file__).resolve().parents[1]
RUNNER = (
    ROOT
    / "docs/benchmarks/run_ssl_lstm_neutra_complexity_predictive_validation_2026_07_19.py"
)


def load_runner():
    name = "ssl_lstm_neutra_complexity_predictive_validation_runner"
    spec = importlib.util.spec_from_file_location(name, RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runner = load_runner()


def test_contract_smoke_is_dynamic_and_materially_side_effect_free() -> None:
    args = runner.parse_args(["--mode", "contract-smoke", "--q", "20"])
    runner.validate_args(args)
    payload = runner.contract_payload(args)
    assert payload["phase5_admission_draws_per_chain"] == {
        "minimum": 512,
        "maximum": 4096,
        "derived_from_each_admitted_phase5_receipt": True,
    }
    assert payload["phase6_extension_segment_range"] == {
        "if_phase5_admits_at_4096": 32,
        "if_phase5_admits_at_512": 46,
    }
    assert payload["forecast_worker_count"] == 16
    assert payload["predictive_contract"]["ridge_ladder"] == [0.0]
    assert payload["path_plot_artifacts"] == ["json", "png", "pdf"]
    assert payload["material_execution_authorized"] is False


def test_calibration_mode_forbids_retained_inputs_and_validation_requires_receipts() -> None:
    args = runner.parse_args(
        [
            "--mode",
            "calibrate",
            "--q",
            "1",
            "--authorize-material-run",
            "--cap-seconds",
            "60",
            "--output-root",
            "docs/plans/artifacts/test-calibration",
            "--phase5-summary",
            "phase5.json",
        ]
    )
    with pytest.raises(runner.PredictiveValidationError, match="forbids"):
        runner.validate_args(args)
    args = runner.parse_args(
        [
            "--mode",
            "validate",
            "--q",
            "1",
            "--authorize-material-run",
            "--cap-seconds",
            "60",
            "--output-root",
            "docs/plans/artifacts/test-validation",
        ]
    )
    with pytest.raises(runner.PredictiveValidationError, match="requires Phase 5"):
        runner.validate_args(args)
    resume_args = runner.parse_args(
        [
            "--mode",
            "calibrate",
            "--q",
            "1",
            "--authorize-material-run",
            "--cap-seconds",
            "60",
            "--output-root",
            "docs/plans/artifacts/test-calibration",
            "--resume",
        ]
    )
    assert resume_args.resume is True


def test_calibration_chain_checkpoint_replays_hashes(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "CALIBRATION_DRAWS_PER_CHAIN", 2)
    observations = tf.reshape(tf.cast(tf.range(40), tf.float64), (2, 2, 10))
    tensor_path = tmp_path / "calibration-chain-00.tftensor"
    tensor_path.write_bytes(bytes(tf.io.serialize_tensor(observations).numpy()))
    root = runner.calibration_seed_roots(1)[0]
    seeds = runner.forecast_seeds_from_root(root, 2)
    seed_hash = runner.hashlib.sha256(seeds.tobytes()).hexdigest()
    observation_hash = runner.hashlib.sha256(
        np.ascontiguousarray(observations.numpy()).tobytes()
    ).hexdigest()
    signature = runner.payload_sha256(
        {
            "root": list(root),
            "seed_hash": seed_hash,
            "observation_hash": observation_hash,
        }
    )
    receipt = {
        "schema": runner.SCHEMA,
        "mode": "calibrate-chain",
        "q": 1,
        "chain_index": 0,
        "root_seed": list(root),
        "seed_hash": seed_hash,
        "observation_hash": observation_hash,
        "observation_tensor_sha256": runner.sha256(tensor_path),
        "forecast_signature": signature,
        "aggregate_parent_worker_ru_maxrss_bytes": 1,
        "execution_source_signature": "source",
    }
    (tmp_path / "calibration-chain-00.json").write_text(
        json.dumps(receipt), encoding="utf-8"
    )
    loaded = runner.load_calibration_chain(
        output=tmp_path,
        q=1,
        chain_index=0,
        root=root,
        source_signature="source",
    )
    assert loaded is not None
    tf.debugging.assert_equal(loaded[0], observations)
    assert loaded[1] == signature
    receipt["seed_hash"] = "wrong"
    (tmp_path / "calibration-chain-00.json").write_text(
        json.dumps(receipt), encoding="utf-8"
    )
    with pytest.raises(runner.PredictiveValidationError, match="seed hash"):
        runner.load_calibration_chain(
            output=tmp_path,
            q=1,
            chain_index=0,
            root=root,
            source_signature="source",
        )


def test_phase5_requires_admission_and_dynamic_draw_counts(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "sha256", lambda path: "hash")
    source_hashes = {
        key: "hash"
        for key in ("target", "adapter", "artifact_loader", "hmc", "diagnostics")
    }
    payload = {
        "schema": runner.PHASE5_SCHEMA,
        "status": "ADMITTED",
        "q": 2,
        "both_charts_admitted": True,
        "source_bindings": {"source_sha256": source_hashes},
        "charts": {
            chart: {
                "final_admission": {"admitted": True, "draw_count_per_chain": draws}
            }
            for chart, draws in (("chart-a", 512), ("chart-b", 2048))
        },
    }
    path = tmp_path / "phase5.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded, _binding = runner.load_phase5(2, Path("phase5.json"))
    assert loaded["charts"]["chart-a"]["final_admission"]["draw_count_per_chain"] == 512
    payload["charts"]["chart-b"]["final_admission"]["admitted"] = False
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(runner.PredictiveValidationError, match="admission missing"):
        runner.load_phase5(2, Path("phase5.json"))


def test_seed_domains_are_disjoint_and_q_specific() -> None:
    q1 = runner.seed_domain_contract(1)
    q20 = runner.seed_domain_contract(20)
    assert q1["pairwise_disjoint"] is True
    assert q20["pairwise_disjoint"] is True
    assert q1["sha256"] != q20["sha256"]


def test_extension_count_uses_actual_phase5_admission() -> None:
    assert runner.phase6_extension_segment_count(4096) == 32
    assert runner.phase6_extension_segment_count(2048) == 40
    assert runner.phase6_extension_segment_count(512) == 46
    with pytest.raises(runner.PredictiveValidationError, match="admission draw"):
        runner.phase6_extension_segment_count(7680)


def test_sampler_extension_screen_renews_movement_gate(monkeypatch) -> None:
    monkeypatch.setattr(
        runner,
        "compute_coordinate_diagnostics",
        lambda values: {
            "rank_normalized_split_rhat": {"maximum": [1.0] * 4},
            "rank_normalized_ess": {"bulk": [1000.0] * 4, "tail": [1000.0] * 4},
            "mean": {"mcse_sd_ratio": [0.01] * 4},
        },
    )
    values = tf.zeros((4, runner.PREDICTIVE_DRAWS_PER_CHAIN, 4), tf.float64)
    archive = {
        "manifest": {
            "diagnostics_private_metadata": {
                "native_divergence_status": "not_exposed_by_kernel",
                "divergence_count": None,
                "sampler_health_diagnostics": {
                    "log_accept_ratio": {"nonfinite_count": 0},
                    "target_log_prob": {"nonfinite_count": 0},
                },
            }
        }
    }
    result = runner.sampler_extension_screen(values, values, [archive])
    assert result["passed"] is False
    assert result["chain_moved"] == [False, True, True, True]
    assert "unmoved_chain" in result["hard_vetoes"]


def test_predictive_decision_accepts_identical_and_detects_shifted_fixtures() -> None:
    generator = tf.random.Generator.from_seed(20260719)
    base = generator.normal((4, 512, 2, 10), dtype=tf.float64)
    variances = tf.ones_like(base)
    identical = runner.predictive_decision(
        {"chart-a": base, "chart-b": base},
        {"chart-a": variances, "chart-b": variances},
    )
    assert identical["status"] == "PASS"
    shifted = runner.predictive_decision(
        {"chart-a": base, "chart-b": base + 2.0},
        {"chart-a": variances, "chart-b": variances},
    )
    assert shifted["status"] == "MATERIAL_DIFFERENCE"


def test_resume_requires_summary_and_material_contract_match(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(
        runner,
        "validate_calibration",
        lambda q, path: {
            "path": "calibration.json",
            "sha256": "calibration-hash",
            "calibration_signature": "calibration-signature",
        },
    )
    phase5 = {
        "charts": {
            chart: {
                "segments": [],
                "final_admission": {"draw_count_per_chain": 512},
            }
            for chart in runner.CHARTS
        }
    }
    monkeypatch.setattr(
        runner,
        "load_phase5",
        lambda q, path: (phase5, {"path": "phase5.json", "sha256": "phase5-hash"}),
    )
    monkeypatch.setattr(runner, "execution_source_signature", lambda: "source")
    args = Namespace(
        q=1,
        output_root=Path("out"),
        calibration_receipt=Path("calibration.json"),
        phase5_summary=Path("phase5.json"),
        resume=True,
        cap_seconds=1000.0,
        mode="validate",
    )
    with pytest.raises(runner.PredictiveValidationError, match="resume requires"):
        runner.run_validation(args)


def test_path_plot_writer_emits_json_png_and_pdf(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    output = tmp_path / "plots"
    output.mkdir()
    paths = {
        chart: tf.reshape(
            tf.cast(tf.range(4 * 12288 * 2 * 10), tf.float64),
            (4, 12288, 2, 10),
        )
        for chart in runner.CHARTS
    }
    receipt = runner.write_path_plot_artifacts(paths, output)
    assert receipt["path_type"] == "simulated_standardized_observation_path"
    for key in ("data", "png", "pdf"):
        assert (tmp_path / receipt[key]["path"]).is_file()
        assert receipt[key]["sha256"]


def test_forecast_memory_veto_uses_parent_plus_worker_high_water() -> None:
    with pytest.raises(runner.HostMemoryVeto, match="parent plus"):
        runner.enforce_forecast_memory(
            {"aggregate_parent_worker_ru_maxrss_bytes": runner.HOST_RAM_CAP_BYTES + 1}
        )
