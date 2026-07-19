from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import ModuleType

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import pytest
import tensorflow as tf


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "docs/benchmarks/diagnose_ssl_lstm_neutra_phase8_raw_covariance_decompositions_2026_07_17.py"
)


@pytest.fixture(scope="module")
def harness() -> ModuleType:
    spec = importlib.util.spec_from_file_location("phase8_raw_decomposition", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_diagnostic_binds_exact_projection_failure(harness: ModuleType) -> None:
    receipt = harness.validate_diagnostic_receipt()
    assert tuple(receipt["gpu_xla_audit"]["failure_indices"]) == harness.EXPECTED_FAILURE_INDICES
    assert receipt["classification"]["projection_only"] is True


def test_decomposition_kernel_matches_cpu_spd_fixture(harness: ModuleType) -> None:
    base = tf.constant(
        ((2.0, 0.3, 0.1), (0.3, 1.0, -0.2), (0.1, -0.2, 0.7)), tf.float64
    )
    values = tf.broadcast_to(base, (256, 3, 3))
    outputs = harness._decomposition_kernel(values)
    for index in (2, 8, 14, 15, 19, 20, 21):
        assert float(tf.reduce_max(outputs[index])) < 8.0
    assert bool(tf.reduce_all(outputs[11] > 0.0))
    assert bool(tf.reduce_all(outputs[16] > 0.0))


def test_source_is_localization_only() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "forecast_ssl_lstm_paths" not in source
    assert '"forecast_executed": False' in source
    assert '"production_source_modified_by_diagnostic": False' in source
    assert '"confirmation_suffix_selected": False' in source


def test_write_json_refuses_overwrite(harness: ModuleType, tmp_path: Path) -> None:
    output = tmp_path / "receipt.json"
    harness._write_json(output, {"status": "first"})
    with pytest.raises(harness.RawDecompositionDiagnosticError, match="overwrite"):
        harness._write_json(output, {"status": "second"})
