from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "docs/benchmarks/audit_ssl_lstm_neutra_confirmation_support_2026_07_16.py"


def load_runner():
    name = "ssl_lstm_neutra_confirmation_support_audit"
    spec = importlib.util.spec_from_file_location(name, RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runner = load_runner()


def test_support_audit_is_read_only_and_uses_existing_checkpoint_hashes() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    assert runner.STEPS == (100, 300, 600, 800, 1100)
    assert "trainer.train_step" not in source
    assert "checkpoint hash mismatch" in source
    assert "load_frozen_neutra_artifact" in source
    assert "_probe_diagnostics" in source
    assert "support_screen_passed" in source
    assert "CHECKPOINT_SELECTION_MISALIGNMENT_SUPPORTED" in source
    assert "SELECTED_CANDIDATE_SUPPORT_INSTABILITY_SUPPORTED" in source
    assert "no retraining or reselection authority" in source
