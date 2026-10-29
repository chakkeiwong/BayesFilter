"""Rejected experimental execution settings cannot silently become serial."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys

import pytest


def test_experimental_worker_payload_fails_instead_of_changing_execution():
    from bayesfilter.inference import HMCCandidateExecutionConfig
    from tests.test_hmc_candidate_set_execution import execution_config
    payload = execution_config().payload()
    assert "trial_analysis_workers" not in payload
    assert HMCCandidateExecutionConfig.from_payload(payload).payload() == payload
    changed = deepcopy(payload)
    changed["trial_analysis_workers"] = 4
    with pytest.raises(TypeError, match="trial_analysis_workers"):
        HMCCandidateExecutionConfig.from_payload(changed)


def test_experimental_model_configuration_is_not_downgraded():
    from bayesfilter.testing.acceptance_release_validation import full_search_configuration
    from bayesfilter.testing.acceptance_decision_models import validate_configuration
    cfg = full_search_configuration("nonlinear", seed=(20261002, 2502),
        wall_seconds=1800, replicated_trial_batch_size=32)
    validate_configuration(cfg)
    cfg.update(schema="bayesfilter.acceptance_model_config.v4", trial_analysis_workers=4)
    with pytest.raises(ValueError, match="unsupported model configuration"):
        validate_configuration(cfg)


def test_withdrawn_cli_option_fails_before_any_worker_launch(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1]/"scripts/run_hmc_v7_release_prices.py"
    spec = importlib.util.spec_from_file_location("serial_release_prices", path)
    prices = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prices)
    monkeypatch.setattr(sys, "argv", ["prices", "--source", str(tmp_path/"source"),
        "--output", str(tmp_path/"run"), "--gpu", "-1", "--trial-analysis-workers", "4"])
    def unexpected(*args, **kwargs):
        pytest.fail("a withdrawn option must fail before source access or launch")
    monkeypatch.setattr(prices, "check_source", unexpected)
    with pytest.raises(SystemExit) as exc:
        prices.main()
    assert exc.value.code == 2
    assert not (tmp_path/"run").exists()
