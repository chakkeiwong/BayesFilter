"""Focused contract tests for historical SMC receipt recovery."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "docs/benchmarks/recover_ssl_lstm_q20_physical_annealed_smc_receipts_2026_08_10.py"


def _load_runner():
    name = "test_ssl_lstm_q20_physical_annealed_smc_receipt_recovery_runner"
    spec = importlib.util.spec_from_file_location(name, RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_receipt_name_sets_reconstruct_full_stage_schema() -> None:
    runner = _load_runner()
    assert len(runner.INITIAL_NAMES) == 13
    assert len(runner.PRE_NAMES) == 9
    assert len(runner.POST_NAMES) == 13
    assert runner.PRE_NAMES & runner.POST_NAMES == {
        "proposal_log_prob",
        "roots",
        "sign",
        "target_log_prob",
        "theta",
        "z",
    }


def test_recovery_is_cpu_only_read_only_for_historical_evidence() -> None:
    runner = _load_runner()
    source = RUNNER.read_text(encoding="utf-8")
    assert runner.RECOVERY.name == "receipt-recovery-v1.json"
    assert 'os.environ["CUDA_VISIBLE_DEVICES"] = "-1"' in source
    assert "target_log_prob_fn" not in source
    assert "batch_native_complexity_posterior_target" not in source
    assert "refusing to overwrite recovery artifact" in source
    assert "rglob(\"*.tftensor\")" in source
    assert 'tf.io.read_file(str(_abs(path)))' in source
