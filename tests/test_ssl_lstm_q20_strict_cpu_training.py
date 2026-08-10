from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "docs/benchmarks/run_ssl_lstm_q20_strict_cpu_training_2026_07_22.py"
PINNED_SCRIPT = ROOT / "docs/benchmarks/run_ssl_lstm_q20_strict_cpu_batch_native_training_2026_07_22.py"
PLAN = ROOT / "docs/plans/bayesfilter-ssl-lstm-q20-strict-cpu-training-plan-2026-07-22.md"


def test_strict_cpu_launcher_binds_requested_training_contract() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "THREAD_LIMIT = 50" in source
    assert "BATCH_SIZE = 100" in source
    assert "HIDDEN_LAYERS = (32, 32)" in source
    assert "dataclasses.replace(config, jit_compile=False)" in source
    assert "_tensorflow.config.threading.set_intra_op_parallelism_threads" in source
    assert "for stream in training.STREAMS" in source
    assert "saturation_repair_enabled=False" in source
    assert "batch_native_complexity_posterior_target" in source
    assert "_BatchNativeBoundary" in source
    assert "CPUValueScorePool(" not in source


def test_strict_cpu_launcher_cannot_promote_outputs() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert '"hmc_eligible": False' in source
    assert '"transport_promotion_eligible": False' in source
    assert '"posterior_claim_eligible": False' in source
    assert "CPU_DIAGNOSTIC_SCREEN_PASSED" in source


def test_batch_native_target_has_no_row_mapping_or_numpy_runtime() -> None:
    target = ROOT / "bayesfilter/nonlinear/ssl_lstm_complexity_batched_target_tf.py"
    source = target.read_text(encoding="utf-8")
    assert "tf.map_fn" not in source
    assert "tf.vectorized_map" not in source
    assert "import numpy" not in source
    assert "np." not in source
    assert 'evaluation_policy = "batch_native_tensorflow_no_row_mapping_v1"' in source


def test_strict_cpu_plan_has_budget_thread_and_stop_contract() -> None:
    text = PLAN.read_text(encoding="utf-8")
    assert "31,500 s" in text
    assert "2,000 program updates" in text
    assert "configured compute cores above 50" in text
    assert "PASS_FOR_BOUNDED_CPU_DIAGNOSTIC_EXECUTION" in text


def test_pinned_launcher_selects_one_stream_and_binds_batch4x25() -> None:
    source = PINNED_SCRIPT.read_text(encoding="utf-8")
    assert 'parser.add_argument("--stream"' in source
    assert 'parser.add_argument("--cpu-processes"' in source
    assert 'parser.add_argument("--batch-per-process"' in source
    assert 'parser.add_argument("--resume-checkpoint"' in source
    assert 'parser.add_argument("--prior-wall-seconds"' in source
    assert "for stream in selected_streams" in source
    assert '"one seed is diagnostic only"' in source
    assert '"hmc_eligible": False' in source


def test_pinned_launcher_resume_is_joint_and_budget_charged() -> None:
    source = PINNED_SCRIPT.read_text(encoding="utf-8")
    assert "validate_joint_training_checkpoint(checkpoint)" in source
    assert 'trainer.restore_state(checkpoint["trainer_state"])' in source
    assert 'controller.restore_state(checkpoint["controller_state"])' in source
    assert "range(program_step + 1, MAX_STEPS + 1)" in source
    assert "budget.require(180.0)" in source
    assert "budget.require(120.0)" in source
    assert 'write_json(output / "summary.json", payload, replace=True)' in source


def test_pinned_launcher_smoke_cannot_be_training_evidence() -> None:
    source = PINNED_SCRIPT.read_text(encoding="utf-8")
    assert 'parser.add_argument("--debug-stop-after-steps"' in source
    assert '"CPU_DEBUG_SMOKE_COMPLETED"' in source
    assert '"training_quality_eligible": False' in source
    assert '"debug smoke only"' in source
