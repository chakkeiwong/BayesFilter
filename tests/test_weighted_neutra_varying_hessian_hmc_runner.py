"""Static contract tests for the parameterized varying-Hessian HMC runner."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "docs/benchmarks/run_weighted_neutra_strong_smooth_hmc_2026_08_12.py"


def test_runner_requires_target_bound_training_inputs() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert 'parser.add_argument("--training-root", type=Path, required=True)' in source
    assert 'parser.add_argument("--plan", type=Path, required=True)' in source
    assert 'training_root / "run_manifest.json"' in source
    assert "training manifest artifact hash mismatch" in source
    assert "training checkpoint target name mismatch" in source
    assert "training checkpoint constants SHA-256 mismatch" in source
    assert "CONSTANTS" not in source


def test_runner_derives_target_scopes_and_preserves_fixed_hmc_policy() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert 'target_scope=f"weighted_neutra_varying_hessian:{target_name}:hmc_v1"' in source
    assert 'target_scope=f"weighted_neutra_varying_hessian:{target_name}:tuning_v1"' in source
    assert "leapfrog_grid=(3, 5, 10, 15, 20, 25)" in source
    assert "if leapfrog < 2:" in source
    assert "NoUTurnSampler" not in source
    assert "use_xla=True" in source
    assert 'os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"' in source
