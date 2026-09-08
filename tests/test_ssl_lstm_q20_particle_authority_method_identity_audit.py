"""Static contract checks for the Phase 17 method-identity audit."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "docs/benchmarks/run_ssl_lstm_q20_particle_authority_method_identity_audit_2026_08_25.py"


def test_method_identity_audit_is_cpu_hidden_and_numpy_free() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert 'os.environ.get("CUDA_VISIBLE_DEVICES") != "-1"' in source
    assert 'os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH") != "true"' in source
    assert "import numpy" not in source
    assert "tf.config.set_visible_devices([], \"GPU\")" in source


def test_method_identity_audit_uses_direct_scientific_classifications() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    for classification in (
        "wrong_relative_to_named_etpf_source_identity",
        "wrong_relative_to_named_genut_source_identity",
        "wrong_relative_to_named_ledh_pfpf_source_identity",
        "wrong_relative_to_named_full_etpf_identity",
    ):
        assert classification in source
    assert "maximum_third_moment_residual" in source
    assert "source_hashes" in source


def test_method_identity_audit_preserves_scaffold_nonclaims() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "scaffold_status" in source
    assert "does not reject ETPF, GenUT, LEDH-PFPF, or ET-PF" in source
    assert "absence in the bounded q20 runner" in source
