"""Focused static contract tests for the source-bound strong-smooth canary."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "docs/benchmarks/run_weighted_neutra_strong_smooth_canary_2026_08_12.py"
REPLAY = ROOT / "docs/benchmarks/generate_weighted_neutra_strong_smooth_replay_2026_08_12.py"


def test_runner_preserves_gpu_xla_and_frozen_affine_local_contract() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert 'os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"' in source
    assert "configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)" in source
    assert 'parser.add_argument("--replay-root", type=Path, required=True)' in source
    assert "jit_compile=True" in source
    assert "training_coordinates\": \"frozen_affine_lift_local_x\"" in source
    assert "theta = mu + L @ IAF(z)" in source
    assert "sample_wise_loop_or_scalar_fallback\": False" in source


def test_runner_uses_frozen_hash_bound_proposal_and_disjoint_replay_clouds() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    replay = REPLAY.read_text(encoding="utf-8")
    assert "frozen replay proposal hash mismatch" in source
    assert "strong-smooth-proposal-r7-reflected" in source
    assert "(20260812, 16001)" in replay
    assert "(20260812, 16002)" in replay
    assert "(20260812, 16000)" in replay
    assert "proposal_sha256" in source
    assert "replay tensor SHA-256 mismatch" in source
    assert 'os.environ["CUDA_VISIBLE_DEVICES"] = "-1"' in replay
    assert 'raw_probabilities = tf.constant(frozen["probabilities"], tf.float64)' in replay
    assert '"training_local_rows": training_rows' in replay
    assert 'rows = tf.gather(training_rows, indices)' in source
    assert "serious strong-smooth rung is frozen" in source
