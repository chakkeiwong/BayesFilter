from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "docs/benchmarks/run_ssl_lstm_neutra_capacity_32x32_diagnostic_2026_07_15.py"


def load_runner():
    name = "ssl_lstm_neutra_capacity_32x32_diagnostic_runner"
    spec = importlib.util.spec_from_file_location(name, RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runner = load_runner()


def test_capacity_diagnostic_reuses_exact_historical_streams() -> None:
    assert [stream.__dict__ for stream in runner.STREAMS] == [
        {
            "label": "seed-a",
            "initialization_seed": (20260715, 4101),
            "training_seed": (20260715, 5101),
            "validation_seed": (20260715, 5201),
        },
        {
            "label": "seed-b",
            "initialization_seed": (20260715, 4102),
            "training_seed": (20260715, 5102),
            "validation_seed": (20260715, 5202),
        },
    ]


def test_capacity_config_changes_width_and_family_only() -> None:
    class Target:
        def target_signature(self):
            return "a" * 64

        def adapter_signature(self):
            return "b" * 64

    config = runner.stream_config(Target(), runner.STREAMS[0])
    assert config.family == runner.SSL_LSTM_CAPACITY_NEUTRA_FAMILY
    assert config.hidden_layers == (32, 32)
    assert config.stages == 3
    assert config.activation == "elu"
    assert config.learning_rate == pytest.approx(0.01)
    assert config.learning_rate_schedule == "paper_piecewise"
    assert config.gradient_clip_mode == "per_variable"
    assert config.gradient_clip_norm == pytest.approx(10.0)
    assert config.jit_compile is True


def test_resource_and_timing_contract_is_frozen() -> None:
    assert runner.STEPS == 1200
    assert runner.CHECKPOINT_EVERY == 100
    assert runner.VALIDATION_EVERY == 100
    assert runner.PER_STREAM_SECONDS == 4500.0
    assert runner.SHARED_SECONDS == 9000.0
    assert runner.SATURATION_MAX == pytest.approx(0.05)
    assert runner.INVERSE_RADIUS_MAX == pytest.approx(4.30)


def test_runner_has_no_full_confirmation_or_hmc_path() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    assert "5000" not in source
    assert "HMC" not in source
    assert "R2_CAPACITY_REPAIR_NOMINATED" in source
    assert "R2_CAPACITY_REPAIR_NOT_NOMINATED" in source


def test_candidate_veto_does_not_suppress_paired_stream() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    assert 'if result["decision"] != "R2_CAPACITY_REPAIR_NOMINATED"' not in source


def test_program_classification_separates_invalid_evidence() -> None:
    passed = {"decision": "R2_CAPACITY_REPAIR_NOMINATED"}
    vetoed = {"decision": "R2_CAPACITY_REPAIR_NOT_NOMINATED"}
    invalid = {"decision": "INVALID_EVIDENCE"}
    assert runner.classify_program([passed, passed], 8999.0) == (
        "R2_CAPACITY_REPAIR_NOMINATED"
    )
    assert runner.classify_program([passed, vetoed], 8999.0) == (
        "R2_CAPACITY_REPAIR_NOT_NOMINATED"
    )
    assert runner.classify_program([passed, invalid], 8999.0) == "INVALID_EVIDENCE"
    assert runner.classify_program([passed], 100.0) == "INVALID_EVIDENCE"
    assert runner.classify_program([passed, passed], 9000.1) == "INVALID_EVIDENCE"


def test_stage_saturation_diagnostic_is_required() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    assert 'validation["stage_saturation"] = stage_saturation' in source
    assert "expected exactly three IAF stages" in source
