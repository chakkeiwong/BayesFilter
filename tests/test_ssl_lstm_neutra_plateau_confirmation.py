from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "docs/benchmarks/run_ssl_lstm_neutra_support_eligible_confirmation_2026_07_16.py"


def load_runner():
    name = "ssl_lstm_neutra_plateau_confirmation_runner"
    spec = importlib.util.spec_from_file_location(name, RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runner = load_runner()


def test_fresh_streams_are_disjoint_from_historical_tuning_streams() -> None:
    seeds = [
        seed
        for stream in runner.FRESH_STREAMS
        for seed in (
            stream.initialization_seed,
            stream.training_seed,
            stream.validation_seed,
        )
    ]
    assert len(seeds) == len(set(seeds)) == 6
    assert not set(seeds) & runner.EXCLUDED_SEED_ROWS
    assert [stream.label for stream in runner.FRESH_STREAMS] == ["fresh-e", "fresh-f"]


def test_policy_load_binds_summary_hash_and_exact_fresh_streams(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    summary = tmp_path / "study-summary.json"
    summary.write_text("{}\n", encoding="utf-8")
    parent = tmp_path / "parent-policy.json"
    parent.write_text(
        json.dumps(
            {
                "selected_hyperparameters": {"learning_rate": 1.0e-3},
                "plateau_policy": {"patience_steps": 500},
            }
        ),
        encoding="utf-8",
    )
    audit = tmp_path / "support-audit.json"
    audit.write_text("{}\n", encoding="utf-8")
    policy = {
        "schema": runner.POLICY_SCHEMA,
        "study_summary_path": "study-summary.json",
        "study_summary_sha256": runner.sha256(summary),
        "parent_policy_path": "parent-policy.json",
        "parent_policy_sha256": runner.sha256(parent),
        "support_audit_path": "support-audit.json",
        "support_audit_sha256": runner.sha256(audit),
        "selected_hyperparameters": {"learning_rate": 1.0e-3},
        "plateau_policy": {
            "patience_steps": 500,
            "roundtrip_max_abs": 1.0e-9,
        },
        "fresh_streams": [stream.__dict__ for stream in runner.FRESH_STREAMS],
    }
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(policy), encoding="utf-8")
    assert runner.load_policy(path)["schema"] == runner.POLICY_SCHEMA
    policy["study_summary_sha256"] = "0" * 64
    path.write_text(json.dumps(policy), encoding="utf-8")
    with pytest.raises(runner.ConfirmationError, match="hash mismatch"):
        runner.load_policy(path)


def test_policy_repair_metadata_may_change_but_parent_schedule_may_not(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    summary = tmp_path / "study-summary.json"
    summary.write_text("{}\n", encoding="utf-8")
    audit = tmp_path / "support-audit.json"
    audit.write_text("{}\n", encoding="utf-8")
    parent_plateau = {
        key: index + 1 for index, key in enumerate(runner.PARENT_PLATEAU_SCHEDULE_KEYS)
    }
    parent_plateau["two_period_interpretation"] = "parent wording"
    parent = tmp_path / "parent-policy.json"
    parent_payload = {
        "selected_hyperparameters": {"learning_rate": 1.0e-3},
        "plateau_policy": parent_plateau,
    }
    parent.write_text(json.dumps(parent_payload), encoding="utf-8")
    repaired_plateau = dict(parent_plateau)
    repaired_plateau.update(
        {
            "two_period_interpretation": "support-eligible wording",
            "checkpoint_eligibility": "prospective repair metadata",
        }
    )
    policy = {
        "schema": runner.POLICY_SCHEMA,
        "study_summary_path": summary.name,
        "study_summary_sha256": runner.sha256(summary),
        "parent_policy_path": parent.name,
        "parent_policy_sha256": runner.sha256(parent),
        "support_audit_path": audit.name,
        "support_audit_sha256": runner.sha256(audit),
        "selected_hyperparameters": parent_payload["selected_hyperparameters"],
        "plateau_policy": repaired_plateau,
        "fresh_streams": [stream.__dict__ for stream in runner.FRESH_STREAMS],
    }
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(policy), encoding="utf-8")
    assert runner.load_policy(path)["schema"] == runner.POLICY_SCHEMA
    policy["plateau_policy"]["patience_steps"] += 1
    path.write_text(json.dumps(policy), encoding="utf-8")
    with pytest.raises(runner.ConfirmationError, match="parent plateau schedule"):
        runner.load_policy(path)


def test_confirmation_uses_controller_best_state_and_mutable_lr() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    assert "NeuTraPlateauController" in source
    assert "trainer.set_learning_rate" in source
    assert "joint_training_checkpoint_payload" in source
    assert "best_trainer.restore_state(best_state)" in source
    assert "terminal-state.json" in source
    assert "best-state.json" in source
    assert "best-frozen-payload.json" in source
    assert "resource-stop-checkpoint.json" in source
    assert source.index("resource-stop-checkpoint.json") < source.index(
        'raise ResourceStop("shared confirmation GPU-time cap exhausted")'
    )
    assert 'parser.add_argument("--resume", action="store_true")' in source
    assert "prior_charged_seconds" in source
    assert "declared total confirmation cap is already exhausted" in source
    assert "completed stream result hash mismatch on resume" in source
    assert "resumed trainer/controller learning rate mismatch" in source
    assert "checkpoint_probes" in source
    assert "controller_observation" in source
    assert "moderate_shell_max_inverse_radius" in source
    assert "serialized_best_support_diagnostics_mismatch" in source
    assert "no_support_eligible_checkpoint" in source
    assert "repaired policy changed selected hyperparameters" in source
    assert "repaired policy changed the parent plateau schedule" in source


def test_confirmation_has_no_hmc_execution_and_requires_both_fresh_runs() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    assert "run_hmc" not in source.lower()
    assert "fixed_transport_hmc" not in source.lower()
    assert "len(results) == len(FRESH_STREAMS)" in source
    assert 'row["decision"] == "CONFIRMATION_PASSED"' in source
    assert "maximum-step stop is not a convergence claim" in source


def test_replayed_support_diagnostics_use_tight_numeric_tolerance() -> None:
    expected = {
        "all_finite": True,
        "saturation_fraction": 0.01,
        "roundtrip_max_abs": 2.0e-15,
        "moderate_shell_max_inverse_radius": 3.2,
    }
    observed = dict(expected)
    observed["moderate_shell_max_inverse_radius"] += 1.0e-13
    assert runner.support_diagnostics_match(observed, expected)
    observed["moderate_shell_max_inverse_radius"] = 3.3
    assert not runner.support_diagnostics_match(observed, expected)
