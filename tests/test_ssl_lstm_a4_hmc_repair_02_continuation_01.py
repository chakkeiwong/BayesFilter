from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import pytest
import tensorflow as tf


ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / (
    "docs/benchmarks/run_ssl_lstm_a4_hmc_repair_02_continuation_01_2026_07_14.py"
)


@pytest.fixture(scope="module")
def continuation() -> ModuleType:
    spec = importlib.util.spec_from_file_location("ssl_lstm_a4_cont01", HARNESS)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_frozen_continuation_contract(continuation: ModuleType) -> None:
    assert continuation.CONTINUATION_DRAWS == 250
    assert continuation.CONTINUATION_BURNIN == 0
    assert continuation.CONTINUATION_SEED == (20260714, 1640)
    assert continuation.FROZEN_STEP_SIZE == pytest.approx(0.37613058552609946)
    assert continuation.NUM_LEAPFROG_STEPS == 4
    assert continuation.TRAJECTORY_LENGTH == pytest.approx(1.5045223421043978)


def test_handoff_hashes_shapes_kernel_and_budget(continuation: ModuleType) -> None:
    segment, samples, state, manifest, total = continuation.validate_handoff()
    assert segment["status"] == "NOT_ADMITTED"
    assert tuple(samples.shape) == (250, 4, 4)
    assert tuple(state.shape) == (4, 4)
    assert total == pytest.approx(2040.799946242012)
    assert manifest["retained_sample_count"] == 250
    tf.debugging.assert_all_finite(samples, "old retained samples")
    tf.debugging.assert_all_finite(state, "old final state")


def test_handoff_hash_drift_fails_closed(
    continuation: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, _digest = continuation.INPUT_HASHES[0]
    monkeypatch.setattr(continuation, "INPUT_HASHES", ((path, "0" * 64),))
    with pytest.raises(continuation.ContinuationError, match="SHA-256 drift"):
        continuation.validate_handoff()


def test_recorded_source_binding_drift_fails_closed(
    continuation: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(continuation, "ROOT", tmp_path)
    source = tmp_path / "source.txt"
    source.write_text("current", encoding="ascii")
    payload = {
        "source_files": [
            {
                "path": "source.txt",
                "bytes": 5,
                "sha256": "0" * 64,
            }
        ]
    }
    with pytest.raises(continuation.ContinuationError, match="binding drift"):
        continuation._assert_recorded_sources_current(payload, Path("receipt.json"))


def test_no_overwrite_guard_checks_public_and_private(
    continuation: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(continuation, "ROOT", tmp_path)
    monkeypatch.setattr(continuation, "OUTPUT_PATH", Path("segment-1.json"))
    monkeypatch.setattr(continuation, "CONTINUATION_PRIVATE", Path("private"))
    continuation._require_fresh()
    (tmp_path / "segment-1.json").write_text("occupied", encoding="ascii")
    with pytest.raises(continuation.ContinuationError, match="refusing overwrite"):
        continuation._require_fresh()


def test_new_namespace_does_not_overlap_repair02(continuation: ModuleType) -> None:
    assert continuation.CONTINUATION_ROOT != continuation.REPAIR02_ROOT
    assert continuation.CONTINUATION_PRIVATE != continuation.REPAIR02_PRIVATE
    assert "continuation-01" in continuation.OUTPUT_PATH.as_posix()
    assert all(
        "continuation-01" in path.as_posix()
        for path in continuation._private_output_paths()
    )
