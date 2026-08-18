"""Static contract tests for matched German reverse-KL training."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "docs/benchmarks/run_weighted_neutra_german_reverse_2026_08_13.py"


def test_reverse_runner_preserves_gpu_xla_batch_native_contract() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert 'os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"' in source
    assert "configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)" in source
    assert "MatchedReverseKLNeuTraTrainer" in source
    assert "german_credit_log_prob_batch" in source
    assert "sample_wise_loop_or_scalar_fallback\": False" in source
    assert "jit_compile=True" in source
    assert '"dtype": "float64"' in source
    assert '"tf32_enabled": False' in source


def test_reverse_runner_uses_disjoint_selection_and_hash_bound_state() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "(20260813, 43002)" in source
    assert "(20260813, 43003)" in source
    assert "selection_audit_disjoint\": True" in source
    assert '"target_data_sha256": spec.data_sha256' in source
    assert '"target_reference_sha256": spec.reference_sha256' in source
    assert '"state_hash"' in source
    assert 'choices=(200, 1000, 3000)' in source
