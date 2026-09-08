"""Static contract tests for the fresh q20 authority pilot wrapper."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "docs/benchmarks/run_ssl_lstm_q20_particle_authority_pilot_2026_08_25.py"


def test_pilot_is_cpu_hidden_and_hash_bound() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert 'os.environ.get("CUDA_VISIBLE_DEVICES") != "-1"' in source
    assert "canonical_protocol_hash" in source
    assert "identity_invariant_reference" in source
    assert "No historical six-bank particle" in source


def test_pilot_has_no_numpy_runtime_dependency() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "import numpy" not in source
    assert "from numpy" not in source


def test_fixed_schedule_is_not_online_adaptive() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "SCHEDULE_CANDIDATES" in source
    assert "_select_fixed_schedule" in source
    assert "every_nonterminal_fixed_stage" in source
    assert "online adaptive" not in source.lower()


def test_mutation_acceptance_is_particle_level_not_coordinate_level() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert 'tf.shape(current["z"])[0]' in source
    assert "proposal_count" in source
    assert "tf.size(current[\"z\"])" not in source


def test_mode_axis_is_explicit_in_protocol() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "MODE_AXIS = 2" in source
    assert '"mode_axis": MODE_AXIS' in source
