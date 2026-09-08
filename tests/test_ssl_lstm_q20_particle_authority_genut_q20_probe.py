"""Static contract checks for the q20 GenUT feasibility probe."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "docs/benchmarks/run_ssl_lstm_q20_particle_authority_genut_q20_probe_2026_08_25.py"


def test_q20_genut_probe_has_feasibility_boundary() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "GENUT_Q20_INFEASIBLE_SCOPE" in source
    assert "genut_feasible" in source
    assert "protocol_hash" in source


def test_q20_genut_probe_preserves_quadrature_nonclaims() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "not IID posterior samples" in source
    assert "no global mode-discovery" in source
