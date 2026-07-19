from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "docs/benchmarks/run_ssl_lstm_neutra_optuna_tuning_2026_07_15.py"


def load_runner():
    name = "ssl_lstm_neutra_optuna_tuning_runner"
    spec = importlib.util.spec_from_file_location(name, RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runner = load_runner()


def test_historical_streams_are_exact_and_no_confirmation_stream_is_present() -> None:
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


def test_rungs_and_search_contract_fail_closed() -> None:
    assert runner.parse_rungs("50,100,200,400") == (50, 100, 200, 400)
    for value in ("", "0,100", "100,100", "100,50", "a,2"):
        with pytest.raises(Exception):
            runner.parse_rungs(value)
    assert runner.fixed_timing_parameters() == runner.TrialParameters(1e-3, 0.01, 5.0)
    with pytest.raises(ValueError, match="learning_rate"):
        runner.TrialParameters(3e-3, 0.01, 5.0)
    with pytest.raises(ValueError, match="initialization_scale"):
        runner.TrialParameters(1e-3, 0.03, 5.0)
    with pytest.raises(ValueError, match="gradient_clip_norm"):
        runner.TrialParameters(1e-3, 0.01, 7.0)


def test_stream_a_is_evaluated_before_stream_b_and_reporting_is_intermediate() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    first = source.index("stream=STREAMS[0]")
    second = source.index("stream=STREAMS[1]")
    assert first < second
    assert "trial.report" in source
    assert "trial.should_prune" in source
    assert 'if a["status"] != "SURVIVED"' in source
    assert "common_worst = max" in source


def test_harness_requires_explicit_cap_and_has_no_hmc_or_confirmation_mode() -> None:
    with pytest.raises(SystemExit):
        runner.parse_args(["--mode", "study"])
    args = runner.parse_args(
        [
            "--mode",
            "timing-smoke",
            "--gpu-cap-seconds",
            "60",
            "--rungs",
            "1,2",
        ]
    )
    assert args.gpu_cap_seconds == pytest.approx(60.0)
    assert args.rungs == (1, 2)
    source = RUNNER_PATH.read_text(encoding="utf-8")
    assert 'choices=("timing-smoke", "study")' in source
    assert "run_hmc" not in source.lower()
    assert "fixed_transport_hmc" not in source.lower()


def test_terminal_vetoes_precede_scalar_objective_nomination() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    assert "heldout_loss_improvement_not_established" in source
    assert "moderate_shell_missing_support" in source
    assert "dense_scale_saturation_above_cap" in source
    assert "worst_historical_stream_terminal_heldout_mean_rkl" in source
    assert "does not statistically rank viable trials" in source


def test_resource_stop_is_not_classified_as_candidate_failure(tmp_path) -> None:
    budget = runner.Budget(10.0)
    error = runner.ResourceStop("fixture cap")
    runner.write_resource_stop_receipt(
        tmp_path,
        runner.STREAMS[0],
        runner.fixed_timing_parameters(),
        budget,
        error,
    )
    payload = runner.json.loads((tmp_path / "resource-stop.json").read_text())
    assert payload["status"] == "RESOURCE_STOP"
    assert payload["candidate_veto"] is False
    assert payload["scientific_interpretation"] == "none"
