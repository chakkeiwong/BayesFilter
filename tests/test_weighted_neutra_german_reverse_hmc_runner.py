"""Static contracts for corrected German reverse-comparator HMC."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "docs/benchmarks/run_weighted_neutra_german_reverse_hmc_2026_08_13.py"


def test_runner_uses_fixed_length_gpu_xla_canonical_hmc() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert 'os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"' in source
    assert "configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)" in source
    assert "leapfrog_grid=(3, 5, 10, 15, 20, 25, 32)" in source
    assert "if leapfrog < 2:" in source
    assert "NoUTurnSampler" not in source
    assert "SequentialNeuTraHMCConfig" in source
    assert "use_xla=True" in source


def test_runner_reports_reference_mcse_without_reopening_weighted_training() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "constrained_from_unconstrained" in source
    assert "mean_batch_means_mcse" in source
    assert "square_batch_means_mcse" in source
    assert '"reference_has_stored_mcse": False' in source
    assert '"weighted_training_reopened": False' in source
    assert "coordinatewise z values are not a joint equality test" in source
    assert 'path.relative_to(root).as_posix()' in source
