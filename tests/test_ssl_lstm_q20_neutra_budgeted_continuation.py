from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / (
    "docs/benchmarks/"
    "run_ssl_lstm_q20_neutra_budgeted_continuation_2026_08_06.py"
)
SUPERVISOR = ROOT / (
    "docs/benchmarks/"
    "run_ssl_lstm_q20_neutra_budgeted_continuation_supervisor_2026_08_06.py"
)
PLAN = ROOT / (
    "docs/plans/"
    "bayesfilter-ssl-lstm-q20-neutra-budgeted-continuation-plan-2026-08-06.md"
)
LOCALIZATION = ROOT / (
    "docs/benchmarks/"
    "diagnose_ssl_lstm_q20_neutra_target_validity_shard_2026_08_06.py"
)


def test_runner_is_syntax_valid_and_uses_no_numpy_runtime() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }

    assert "numpy" not in imports
    assert "np." not in source
    assert "tf.map_fn(" not in source
    assert "tf.vectorized_map(" not in source
    assert "NeuTraPlateauController" not in source
    assert "sample_chain(" not in source
    assert "HamiltonianMonteCarlo(" not in source


def test_contract_freezes_full_budget_and_500_row_banks() -> None:
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = "-1"
    completed = subprocess.run(
        (sys.executable, str(RUNNER), "--contract-only"),
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60.0,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["total_updates"] == 4000
    assert payload["checkpoint_every"] == 500
    assert payload["training_batch_size"] == 100
    assert payload["monitor_size"] == 500
    assert payload["selection_size"] == 500
    assert payload["audit_size"] == 500
    assert payload["monitor_controls_training"] is False
    assert payload["learning_rate_schedule"] == [
        [1, 2000, 2.0e-4],
        [2001, 3000, 1.0e-4],
        [3001, 4000, 5.0e-5],
    ]


def test_runner_binds_exact_resume_hashes_and_embedded_steps() -> None:
    source = RUNNER.read_text(encoding="utf-8")

    assert "c87ee24874705bb12296cc05b82310326579694cc04c2a3682792f9bf18fb9ff" in source
    assert "849e33855d87dc34644e15757942bf872937d9f4d4b00a4f03855661827d761d" in source
    assert '"embedded_step": 1500' in source
    assert '"embedded_step": 2250' in source
    assert 'checkpoint.get("best_trainer_state")' in source
    assert "trainer.restore_state(state)" in source


def test_monitor_has_no_training_control_and_selection_is_post_training() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    loop = source.index("for continuation_update in range(1, max_updates + 1):")
    selection = source.index("selection_z = batch(", loop)
    monitor_role = source.index("explanatory_telemetry_only_no_training_control")

    assert selection > loop
    assert monitor_role < loop
    assert '"monitor_controls_training": False' in source
    assert (
        "trainer.set_learning_rate(learning_rate_for_update(start_continuation_update + 1))"
        in source
    )
    assert "if continuation_update % CHECKPOINT_EVERY == 0:" in source
    assert "if action.should_stop" not in source
    assert "meaningful_improvement" not in source


def test_gpu_and_cpu_topology_fail_closed() -> None:
    runner = RUNNER.read_text(encoding="utf-8")
    supervisor = SUPERVISOR.read_text(encoding="utf-8")

    assert 'os.environ.get("CUDA_VISIBLE_DEVICES") != "1"' in runner
    assert 'multiprocessing.current_process().name == "MainProcess"' in runner
    assert 'spawned target workers require CUDA_VISIBLE_DEVICES=-1' in runner
    assert "configure_tensorflow_gpu_memory_growth(" in runner
    assert runner.index("GPU_MEMORY_POLICY = configure_tensorflow_gpu_memory_growth(") < runner.index(
        "from bayesfilter.inference.neutra_artifacts import"
    )
    assert "require_gpu=MATERIAL_PARENT" in runner
    assert "trainer variables are not all placed on logical GPU 0" in runner
    assert '"batch_native_tensorflow_status_no_row_mapping_v2"' in runner
    assert "training batch did not use every persistent worker" in runner
    assert "worker affinity telemetry is incomplete" in runner
    assert "unique_worker_vm_hwm_bytes_by_pid" in runner
    assert "raw_task_sum_is_not_process_memory_total" in runner
    assert 'Path(f"/proc/{int(pid)}/status")' in runner
    assert '"CUDA_VISIBLE_DEVICES": "1"' in supervisor
    assert '"TF_FORCE_GPU_ALLOW_GROWTH": "true"' in supervisor
    assert '"seed-a": {' in supervisor and '"seed-b": {' in supervisor
    assert "tuple(range(0, 25))" in supervisor
    assert "tuple(range(25, 50))" in supervisor


def test_target_validity_recovery_rejects_whole_batch_before_update() -> None:
    source = RUNNER.read_text(encoding="utf-8")

    recovery = source.index("recovery = bounded_target_validity_recovery(")
    update = source.index("trainer.train_step_with_external_value_score", recovery)
    assert recovery < update
    assert "MAX_TARGET_VALIDITY_RETRIES = 3" in source
    assert "pool.evaluate_with_status(" in source
    assert "target-validity-failure-" in source
    assert "target_validity_failure_observed" in source
    assert "TARGET_VALIDITY_RECOVERY_EXHAUSTED" in source
    assert 'write_json(output / "result.json", result)' in source
    assert "trainer or optimizer state changed on a rejected target batch" not in source
    assert "bounded_target_validity_recovery" in source


def test_checkpoint_migration_is_restricted_to_missing_empty_output_scale() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    continuation_loader = source[
        source.index("def load_continuation_resume(") : source.index(
            "def support_probe(", source.index("def load_continuation_resume(")
        )
    ]

    assert "migrate_legacy_empty_output_scale_state" in source
    assert 'expected.pop("fixed_output_scale", None) != []' in source
    assert "numerical_transform_changed" in source
    assert "historical_checkpoint_modified" in source
    assert "migrate_legacy_empty_output_scale_state(" in continuation_loader
    assert "trainer.restore_state(migrated_state)" in continuation_loader


def test_exact_shard_localization_is_xla_cpu_only_and_hash_bound() -> None:
    source = LOCALIZATION.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }

    assert "numpy" not in imports
    assert 'os.environ.get("CUDA_VISIBLE_DEVICES") != "-1"' in source
    assert "jit_compile=True" in source
    assert "EXPECTED_INPUT_SHA256" in source
    assert 'if (start, stop) != (16, 20):' in source
    assert "placement_classified_invalid_count" in source
    assert "innovation_classified_invalid_count" in source
    assert "threshold_relaxation_supported" in source


def test_plan_has_evidence_audit_and_nonclaims() -> None:
    text = PLAN.read_text(encoding="utf-8")

    assert "Research intent ledger" in text
    assert "Evidence contract" in text
    assert "Default and numeric assumption audit" in text
    assert "Skeptical plan audit" in text
    assert "Pre-mortem" in text
    assert "Decision table" in text
    assert "Inference-status table" in text
    assert "Monitoring is telemetry only and cannot change LR" in text
    assert "43,200 wall seconds" in text
    assert "no convergence or posterior claim" in text
