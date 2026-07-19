from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from types import ModuleType

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import pytest
import tensorflow as tf


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "docs/benchmarks/"
    "run_ssl_lstm_neutra_phase8_predictive_design_canary_2026_07_17.py"
)


@pytest.fixture(scope="module")
def harness() -> ModuleType:
    spec = importlib.util.spec_from_file_location("phase8_predictive_canary", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_phase7_handoff_and_a0_points_are_exact(harness: ModuleType) -> None:
    binding = harness.validate_phase7_receipt()
    assert binding["sha256"] == harness.PHASE7_RECEIPT_SHA256
    assert binding["retained_samples_read_by_canary"] is False
    points = harness.a0_start_points()
    assert tuple(points.shape) == (4, 4)
    assert bool(tf.reduce_all(tf.math.is_finite(points)))


def test_canary_banks_are_replayable_and_domain_separated(harness: ModuleType) -> None:
    config = harness.SSLLSTMForecastConfig()
    first = harness.make_canary_banks(config)
    replay = harness.make_canary_banks(config)
    assert set(first) == {"shared", "independent-g", "independent-h"}
    for label in first:
        assert first[label].content_signature == replay[label].content_signature
        assert first[label].tensor_hashes() == replay[label].tensor_hashes()
    assert first["shared"].role == "paired_diagnostic_shared"
    assert first["independent-g"].arm_id == 1
    assert first["independent-h"].arm_id == 2


def test_canary_constants_do_not_freeze_a3_fixture_policy(harness: ModuleType) -> None:
    assert harness.SHARED_SEED[0] > 10000
    assert harness.INDEPENDENT_SEED[0] > 10000
    assert harness.RIDGE_LADDER == (0.0, 1.0e-12, 1.0e-10, 1.0e-8, 1.0e-6)
    assert harness.CONDITION_NUMBER_MAX == 1.0e8
    source = SCRIPT.read_text(encoding="utf-8")
    for forbidden in (
        "mean_margin_hex",
        "log_variance_margin_hex",
        "mmd_tolerance_hex",
        "coverage_replication_count",
        "bootstrap_count = 128",
    ):
        assert forbidden not in source


def test_write_json_is_strict_and_no_overwrite(
    harness: ModuleType, tmp_path: Path
) -> None:
    output = tmp_path / "receipt.json"
    harness._write_json(
        output,
        {
            "nan": float("nan"),
            "raw_status": b"VALID",
            "tensor_status": tf.constant("VALID"),
        },
    )
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "nan": "NaN",
        "raw_status": "VALID",
        "tensor_status": "VALID",
    }
    with pytest.raises(harness.Phase8CanaryError, match="refusing to overwrite"):
        harness._write_json(output, {"status": "second"})


def test_canary_source_does_not_read_phase7_private_shards(harness: ModuleType) -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "retained-private" not in source
    assert "retained_samples.tftensor" not in source
    assert "confirmatory_forecast_bank_opened\": False" in source


def test_canary_requires_all_compiled_surface_telemetry(harness: ModuleType) -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert '"terminal_output_devices": terminal_devices' in source
    assert '"output_devices": summary_devices' in source
    assert '"statistics": _trace_count(predictive_equivalence._summary_xla)' in source
    assert "predictive_equivalence._long_run_covariance_xla" in source
    assert "any(count != 1 for count in trace_counts.values())" in source

    harness._require_gpu_devices(
        {"means": "/job:localhost/replica:0/task:0/device:GPU:0"},
        surface="summary fixture",
    )
    with pytest.raises(harness.Phase8CanaryError, match="not GPU resident"):
        harness._require_gpu_devices(
            {"means": "/job:localhost/replica:0/task:0/device:CPU:0"},
            surface="summary fixture",
        )
