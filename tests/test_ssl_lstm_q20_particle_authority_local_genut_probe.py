"""Static checks for the local GenUT feasibility probe."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "docs/benchmarks/run_ssl_lstm_q20_particle_authority_local_genut_probe_2026_08_25.py"


def test_local_probe_binds_mode_axis_and_preserves_nonclaims() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "points[:, 2]" in source
    assert "LOCAL_GENUT_INFEASIBLE_SCOPE" in source
    assert "not IID" in source


def test_local_probe_records_subset_indices_and_weight_mass() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert '"indices": indices' in source
    assert "weight_mass_before_renormalization" in source
