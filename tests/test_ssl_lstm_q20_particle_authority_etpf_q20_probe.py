"""Static contract checks for the bounded q20 ETPF probe."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "docs/benchmarks/run_ssl_lstm_q20_particle_authority_etpf_q20_probe_2026_08_25.py"


def test_probe_is_cpu_hidden_and_preserves_mode_axis() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert 'CUDA_VISIBLE_DEVICES") != "-1"' in source
    assert 'mode_axis = 2' in source
    assert "protocol_hash" in source
    assert "target_status_valid" in source


def test_probe_disallows_posterior_or_iid_claims() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "not IID" in source
    assert "no mode-discovery" in source
    assert "deterministic 32-row subset" in source
