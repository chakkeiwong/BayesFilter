"""Static contracts for the CPU-only German proposal diagnostic."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "docs/benchmarks/diagnose_weighted_neutra_german_proposal_2026_08_13.py"


def test_proposal_runner_is_cpu_only_batch_native_and_hash_bound() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert 'os.environ["CUDA_VISIBLE_DEVICES"] = "-1"' in source
    assert "german_credit_log_prob_batch(spec, physical[start:stop])" in source
    assert "sample_wise_loop_or_scalar_fallback\": False" in source
    assert "German proposal checkpoint data hash mismatch" in source
    assert "German proposal checkpoint reference hash mismatch" in source
    assert "proposal_hash" in source
    assert 'choices=("reverse_scale_mixture", "reference_augmented")' in source
    assert "sample_reference_augmented_proposal" in source


def test_proposal_runner_uses_predeclared_global_and_median_batch_ess() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "ESS_FRACTION_MIN = 0.0625" in source
    assert "BATCH_SIZE = 4096" in source
    assert "global_ess / tf.cast(args.sample_count" in source
    assert "median_batch_fraction >= ESS_FRACTION_MIN" in source
