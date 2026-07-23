from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "docs/benchmarks/run_ssl_lstm_two_architecture_loss_gate_2026_07_21.py"
ARTIFACT_ROOT = ROOT / (
    "docs/plans/artifacts/ssl-lstm-q20-two-architecture-loss-gate-2026-07-21"
)


def load_harness():
    name = "ssl_lstm_two_architecture_loss_gate"
    spec = importlib.util.spec_from_file_location(name, HARNESS)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


harness = load_harness()


def test_paired_interval_uses_student_t_critical_value() -> None:
    left = np.zeros(256)
    right = np.arange(256, dtype=np.float64) / 255.0
    row = harness.paired_interval(left.tolist(), right.tolist())
    differences = right - left
    expected_se = np.std(differences, ddof=1) / np.sqrt(256)
    assert row["standard_error"] == pytest.approx(expected_se)
    assert row["two_sided_95_upper"] == pytest.approx(
        np.mean(differences)
        + harness.TWO_SIDED_95_T_CRITICAL_DF_255 * expected_se
    )


def test_real_artifacts_pass_with_disclosed_terminal_snapshot_limitation() -> None:
    arms = [
        harness.load_arm(ARTIFACT_ROOT / "arch-32x32", (32, 32), stream)
        for stream in harness.STREAMS
    ] + [
        harness.load_arm(ARTIFACT_ROOT / "arch-64x64", (64, 64), stream)
        for stream in harness.STREAMS
    ]
    assert len({json.dumps(row["controller_config"], sort_keys=True) for row in arms}) == 1
    assert len({row["target_signature"] for row in arms}) == 1
    assert len({row["source_sha256"]["controller"] for row in arms}) == 2


def test_load_arm_fails_closed_on_manifest_architecture_mismatch(
    tmp_path: Path, monkeypatch
) -> None:
    copied = tmp_path / "arch-32x32"
    (copied / "seed-a").mkdir(parents=True)
    source = ARTIFACT_ROOT / "arch-32x32"
    summary = json.loads((source / "final-summary.json").read_text())
    summary["run_manifest"]["hidden_layers"] = [64, 64]
    (copied / "final-summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (copied / "seed-a" / "result.json").write_text(
        (source / "seed-a" / "result.json").read_text(), encoding="utf-8"
    )
    monkeypatch.setattr(harness, "ROOT", ROOT)
    with pytest.raises(RuntimeError, match="manifest contract mismatch"):
        harness.load_arm(copied, (32, 32), "seed-a")
