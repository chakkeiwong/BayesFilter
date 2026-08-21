"""Static contracts for the paper d100 objective-specific training runner."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "docs/benchmarks/run_neutra_paper_d100_training_2026_08_13.py"
PROFILE = ROOT / "docs/benchmarks/profile_neutra_paper_d100_forward_training_2026_08_14.py"


def test_runner_preserves_exact_replay_and_gpu_xla_training_contract() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert 'choices=("reverse_kl", "forward_kl")' in source
    assert "MatchedReverseKLNeuTraTrainer" in source
    assert "WeightedForwardKLNeuTraTrainer" in source
    assert 'os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"' in source
    assert "configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)" in source
    assert "enable_tensor_float_32_execution(False)" in source
    assert "jit_compile=True" in source
    assert "sample_wise_loop_or_scalar_fallback" in source
    assert "external_exact_sample_generation_cpu_only" in source
    assert "audit_opened_after_checkpoint_freeze" in source
    assert "training replay row count must be divisible by batch size" in source
    assert "tf.map_fn" not in source
    assert "tf.vectorized_map" not in source
    assert "tf.while_loop" not in source


def test_runner_excludes_unsupported_budget_and_scalar_batch() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "batch size must exceed one" in source
    assert "capped at 10000 updates" in source
    assert "output root must be fresh" in source
    assert "replay target mismatch" in source
    assert "replay manifest artifact hash mismatch" in source


def test_forward_profile_is_diagnostic_and_records_xla_device_contract() -> None:
    source = PROFILE.read_text(encoding="utf-8")
    assert "tf.profiler.experimental.start" in source
    assert "tf.profiler.experimental.stop" in source
    assert '"execution_target": "gpu_xla_diagnostic_only"' in source
    assert '"jit_compile": True' in source
    assert '"tf32_enabled": False' in source
    assert "import numpy" not in source
