"""Static role-boundary tests for Phase 3 modular diagnostics."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "docs/benchmarks/run_ssl_lstm_q20_particle_authority_modular_arms_2026_08_25.py"


def test_modular_runner_preserves_role_boundaries() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "finite_moment_transform_auxiliary" in source
    assert "symmetric_ut_sigma_point_auxiliary" in source
    assert "not_genut" in source
    assert "affine_density_scaffold" in source
    assert "approximate_etpf_comparator" in source
    assert "canonical LEDH-PFPF" in source


def test_modular_runner_is_cpu_hidden_and_numpy_free() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert 'CUDA_VISIBLE_DEVICES") != "-1"' in source
    assert "import numpy" not in source
    assert "from numpy" not in source
