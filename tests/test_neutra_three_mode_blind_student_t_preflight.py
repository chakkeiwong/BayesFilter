"""Static contract checks for the mode-blind Student-t proposal preflight."""

from __future__ import annotations

from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "docs/benchmarks/preflight_neutra_three_mode_blind_student_t_2026_08_17.py"
)


def test_mode_blind_proposal_does_not_construct_from_component_parameters() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    proposal_block = source.split("proposal = tfd.Independent(", 1)[1].split(
        "rows = proposal.sample", 1
    )[0]
    assert 'loc=tf.zeros((4,), tf.float64)' in proposal_block
    assert 'target["means"]' not in proposal_block
    assert 'target["covariances"]' not in proposal_block
    assert 'target["probabilities"]' not in proposal_block


def test_preflight_uses_xla_and_derived_support_threshold() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "@tf.function(jit_compile=True" in source
    assert "ESS_FRACTION_MIN = 1.0 / 16.0" in source
    assert "proposal_support_failed_stop_before_training_and_hmc" in source
