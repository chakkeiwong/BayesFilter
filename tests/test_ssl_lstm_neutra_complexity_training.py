from __future__ import annotations

import importlib.util
import json
import os
import signal
import sys
from pathlib import Path

import pytest


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "docs/benchmarks/run_ssl_lstm_neutra_complexity_training_2026_07_19.py"


def load_runner():
    name = "ssl_lstm_neutra_complexity_training_runner"
    spec = importlib.util.spec_from_file_location(name, RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runner = load_runner()


def test_selected_worker_topology_and_search_contract() -> None:
    assert runner.WORKERS_BY_Q == {1: 32, 2: 32, 5: 32, 10: 32, 20: 16}
    assert runner.OPTUNA_RUNGS == (50, 100, 200, 400)
    assert runner.BATCH_SIZE == 480
    assert runner.parse_args(["--mode", "single-diagnostic", "--q", "20"]).batch_size == 480
    assert runner.parse_args(
        ["--mode", "single-diagnostic", "--q", "20", "--batch-size", "100"]
    ).batch_size == 100
    assert runner.VALIDATION_BATCH_SIZE == 64
    assert runner.MAX_STEPS == 2000
    assert runner.pool_config(20).worker_count == 16
    with pytest.raises(ValueError, match="selected topology"):
        runner.pool_config(20, 32)


def test_batch_size_must_be_positive() -> None:
    with pytest.raises(SystemExit):
        runner.parse_args(["--mode", "single-diagnostic", "--q", "20", "--batch-size", "0"])


def test_hidden_layer_contract_preserves_baseline_and_accepts_deep_diagnostic() -> None:
    assert runner.parse_args(["--mode", "single-diagnostic", "--q", "20"]).hidden_layers == (
        32,
        32,
    )
    assert runner.parse_args(
        ["--mode", "single-diagnostic", "--q", "20", "--hidden-layers", "32,32,32"]
    ).hidden_layers == (32, 32, 32)
    assert runner.parse_args(
        ["--mode", "single-diagnostic", "--q", "20", "--hidden-layers", "64,64"]
    ).hidden_layers == (64, 64)
    with pytest.raises(SystemExit):
        runner.parse_args(
            ["--mode", "single-diagnostic", "--q", "20", "--hidden-layers", "64,32"]
        )


def test_material_modes_fail_closed_without_explicit_authority() -> None:
    args = runner.parse_args(["--mode", "study", "--q", "1"])
    with pytest.raises(runner.ComplexityTrainingError, match="authorize-material-run"):
        runner.validate_material_args(args)
    args = runner.parse_args(
        [
            "--mode",
            "study",
            "--q",
            "1",
            "--authorize-material-run",
            "--gpu-cap-seconds",
            "60",
        ]
    )
    with pytest.raises(runner.ComplexityTrainingError, match="explicit output root"):
        runner.validate_material_args(args)

    args = runner.parse_args(["--mode", "single-diagnostic", "--q", "20"])
    with pytest.raises(runner.ComplexityTrainingError, match="authorize-material-run"):
        runner.validate_material_args(args)


def test_single_diagnostic_is_q20_only_and_cannot_change_final_seed_contract() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert 'args.mode == "single-diagnostic"' in source
    assert 'phase3_admission_status"] = "NOT_EVALUATED_ONE_SEED"' in source
    assert '"single_seed_mechanism_diagnostic_only"' in source
    assert "for stream in STREAMS" in source
    assert runner.STREAMS == (
        runner.Stream("seed-a", (20260719, 12101), (20260719, 13101), (20260719, 14101)),
        runner.Stream("seed-b", (20260719, 12102), (20260719, 13102), (20260719, 14102)),
    )


def test_spawned_cpu_worker_marker_overrides_parent_gpu_selection(monkeypatch) -> None:
    monkeypatch.setenv("BAYESFILTER_CPU_VALUE_SCORE_WORKER", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    assert runner._configure_visibility_before_tensorflow_import() == "-1"
    assert runner.os.environ["CUDA_VISIBLE_DEVICES"] == "-1"


def test_contract_smoke_records_external_boundary_and_best_state_repair() -> None:
    args = runner.parse_args(["--mode", "contract-smoke", "--q", "20"])
    runner.validate_material_args(args)
    payload = runner.contract_payload(args)
    assert payload["status"] == "PASSED"
    assert payload["selected_worker_count"] == 16
    assert payload["material_execution_authorized"] is False
    assert payload["plateau_config"]["validation_check_every"] == 250
    assert payload["plateau_config"]["patience_steps"] == 250
    assert payload["plateau_config"]["max_steps"] == 2000
    assert payload["plateau_config"]["learning_rate_factor"] == pytest.approx(0.5)
    assert payload["plateau_config"]["post_repair_no_improvement_cycles"] == 2
    assert payload["plateau_config"]["moderate_shell_max_inverse_radius"] == pytest.approx(
        4.30
    )
    assert payload["repair_order"] == [
        "observe paired validation plateau",
        "restore best trainer and Adam state",
        "halve learning rate without resetting controller patience",
        "stop after two additional validation cycles without improvement",
    ]
    assert payload["fresh_confirmation_contract"]["stream"]["label"] == "seed-c"
    assert payload["fresh_confirmation_contract"]["separate_receipt_required"] is True
    assert payload["source_bindings"]["source_sha256"]["runner"]


def test_loss_only_control_is_explicit_and_manifested() -> None:
    args = runner.parse_args(
        ["--mode", "single-diagnostic", "--q", "20", "--loss-only-control"]
    )
    assert args.loss_only_control is True
    params = runner.fixed_smoke_parameters()
    payload = runner.plateau_config(
        params, saturation_repair_enabled=not args.loss_only_control
    ).manifest_payload()
    assert payload["saturation_repair_enabled"] is False
    source = RUNNER.read_text(encoding="utf-8")
    assert "saturation_repair_enabled=not args.loss_only_control" in source
    assert '"audit": audit' in source
    assert "AUDIT_BATCH_SIZE = 256" in source
    assert "LOSS_ONLY_ARCHITECTURE_PLAN" in source
    assert runner.plan_for_args(args) == runner.LOSS_ONLY_ARCHITECTURE_PLAN
    assert "process_launch_after_import_before_execution" in source
    assert '"gpu_memory_growth_verified"' in source
    assert '"gpu_allocator_memory_bytes"' in source


def test_external_transport_audit_uses_frozen_batch_api() -> None:
    class FrozenTransport:
        def forward_batch(self, z):
            return z + 1.0

        def log_abs_det_jacobian_batch(self, z):
            return runner.tf.zeros((runner.AUDIT_BATCH_SIZE,), dtype=runner.tf.float64)

    class Pool:
        def evaluate_values(self, rows, *, request_id):
            assert rows.shape == (runner.AUDIT_BATCH_SIZE, 4)
            assert request_id == "audit-test"
            return [0.0] * runner.AUDIT_BATCH_SIZE, {}

    audit = runner._external_transport_audit(
        FrozenTransport(), Pool(), runner.tf.zeros((runner.AUDIT_BATCH_SIZE, 4), runner.tf.float64),
        request_id="audit-test",
    )
    assert audit["batch_size"] == runner.AUDIT_BATCH_SIZE
    assert audit["mean_loss"] == 0.0


def test_runner_uses_external_pool_paths_and_sequential_optuna_streams() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "train_step_with_external_value_score" in source
    assert "validation_batch_with_external_value" in source
    assert "pool.evaluate_values" in source
    assert "with CPUValueScorePool(pool_config(args.q)) as pool" in source
    assert source.index("stream=STREAMS[0]") < source.index("stream=STREAMS[1]")
    assert "trial.report" in source
    assert "trial.should_prune" in source
    assert "trainer.restore_state(best_state)" in source
    assert "trainer.set_learning_rate(action.current_learning_rate)" in source
    assert "validate_joint_training_checkpoint" in source
    assert '"candidate_veto": False' in source
    assert "write_trial_record" in source
    assert "HostMemoryVeto" in source


def test_trial_parameter_contract_fails_closed() -> None:
    assert runner.fixed_smoke_parameters() == runner.TrialParameters(4e-4, 0.01, 10.0)
    with pytest.raises(ValueError, match="learning_rate"):
        runner.TrialParameters(3e-3, 0.01, 10.0)
    with pytest.raises(ValueError, match="initialization_scale"):
        runner.TrialParameters(4e-4, 0.03, 10.0)
    with pytest.raises(ValueError, match="gradient_clip_norm"):
        runner.TrialParameters(4e-4, 0.01, 7.0)


def test_budget_resume_charges_prior_seconds() -> None:
    budget = runner.Budget(100.0, prior_seconds=40.0)
    assert budget.elapsed >= 40.0
    with pytest.raises(runner.ResourceStop):
        budget.require(61.0)


def _joint_checkpoint(step: int = 250):
    best = {
        "step": step,
        "config": {"family": "fixture"},
        "state_hash": "a" * 64,
    }
    current = {
        "step": step,
        "config": {"family": "fixture"},
        "state_hash": "b" * 64,
    }
    controller = {
        "last_observation_step": step,
        "best_trainer_state_hash": best["state_hash"],
    }
    return runner.joint_training_checkpoint_payload(
        trainer_state=current,
        controller_state=controller,
        best_trainer_state=best,
    )


def test_latest_verified_progress_checkpoint_rejects_tampering(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    stream = runner.STREAMS[0]
    checkpoint = tmp_path / "checkpoint-0250.json"
    runner.write_json(checkpoint, _joint_checkpoint())
    receipt = {
        "step": 250,
        "path": checkpoint.relative_to(tmp_path).as_posix(),
        "sha256": runner.sha256(checkpoint),
        "checkpoint_hash": json.loads(checkpoint.read_text())["checkpoint_hash"],
    }
    progress = json.loads(runner.canonical({
        "schema": runner.SCHEMA,
        "status": "RUNNING",
        "stream": runner.asdict(stream),
        "last_program_step": 250,
        "history": [{"step": 250}],
        "checkpoints": [receipt],
    }))
    joint, source = runner.latest_verified_progress_checkpoint(
        progress=progress,
        stream=stream,
    )
    assert joint["checkpoint_hash"] == receipt["checkpoint_hash"]
    assert source["step"] == 250

    tampered = dict(progress)
    tampered["last_program_step"] = 500
    with pytest.raises(runner.ComplexityTrainingError, match="step mismatch"):
        runner.latest_verified_progress_checkpoint(progress=tampered, stream=stream)

    checkpoint.write_text(checkpoint.read_text() + " ", encoding="utf-8")
    with pytest.raises(runner.ComplexityTrainingError, match="file hash mismatch"):
        runner.latest_verified_progress_checkpoint(progress=progress, stream=stream)


def test_signal_interruption_is_deferred_and_resettable() -> None:
    runner.reset_training_interruption()
    runner.request_training_interruption(signal.SIGTERM, None)
    assert runner._INTERRUPTION_SIGNAL == signal.SIGTERM
    runner.reset_training_interruption()
    assert runner._INTERRUPTION_SIGNAL is None


def test_runner_supports_interruption_receipts_and_orphan_resume() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert '"interruption-stop.json"' in source
    assert "latest_verified_progress_checkpoint" in source
    assert '"single-diagnostic orphan resume requires RUNNING progress"' in source
    assert "install_training_signal_handlers" in source


def test_confirmation_mode_is_trigger_bound_and_fail_closed(tmp_path, monkeypatch) -> None:
    args = runner.parse_args(["--mode", "confirmation", "--q", "1"])
    with pytest.raises(runner.ComplexityTrainingError, match="authorize-material-run"):
        runner.validate_material_args(args)

    monkeypatch.setattr(runner, "ROOT", tmp_path)
    failed = tmp_path / "failed.json"
    failed.write_text(
        runner.canonical(
            {
                "schema": runner.SCHEMA,
                "q": 1,
                "status": "VETOED",
                "stream": runner.asdict(runner.STREAMS[0]),
                "params": {
                    "learning_rate": 4.0e-4,
                    "initialization_scale": 0.01,
                    "gradient_clip_norm": 10.0,
                },
                "vetoes": ["heldout_loss_improvement_not_established"],
            }
        ).decode("ascii"),
        encoding="ascii",
    )
    admitted = tmp_path / "admitted.json"
    admitted.write_text(
        runner.canonical(
            {
                "schema": runner.SCHEMA,
                "q": 1,
                "status": "ADMITTED",
                "stream": runner.asdict(runner.STREAMS[1]),
                "params": {
                    "learning_rate": 4.0e-4,
                    "initialization_scale": 0.01,
                    "gradient_clip_norm": 10.0,
                },
            }
        ).decode("ascii"),
        encoding="ascii",
    )
    summary = tmp_path / "final-summary.json"
    summary.write_text(
        runner.canonical(
            {
                "schema": runner.SCHEMA,
                "mode": "final",
                "status": "COMPLETED",
                "q": 1,
                "params": {
                    "learning_rate": 4.0e-4,
                    "initialization_scale": 0.01,
                    "gradient_clip_norm": 10.0,
                },
                "fresh_confirmation_eligible": True,
                "results": [
                    {
                        "label": "seed-a",
                        "path": "failed.json",
                        "sha256": runner.sha256(failed),
                        "status": "VETOED",
                    },
                    {
                        "label": "seed-b",
                        "path": "admitted.json",
                        "sha256": runner.sha256(admitted),
                        "status": "ADMITTED",
                    },
                ],
            }
        ).decode("ascii"),
        encoding="ascii",
    )
    trigger = runner.load_confirmation_trigger(
        1,
        final_summary_path=Path("final-summary.json"),
        failed_result_path=Path("failed.json"),
    )
    assert trigger["failed_stream"]["label"] == "seed-a"
    assert trigger["failed_result_sha256"] == runner.sha256(failed)

    invalid = tmp_path / "invalid.json"
    invalid.write_text(
        runner.canonical(
            {
                "schema": runner.SCHEMA,
                "q": 1,
                "status": "ADMITTED",
                "stream": runner.asdict(runner.STREAMS[0]),
            }
        ).decode("ascii"),
        encoding="ascii",
    )
    with pytest.raises(runner.ComplexityTrainingError, match="does not match"):
        runner.load_confirmation_trigger(
            1,
            final_summary_path=Path("final-summary.json"),
            failed_result_path=Path("invalid.json"),
        )
