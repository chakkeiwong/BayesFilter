"""CPU-only accounting checks for a localized restricted-comparison retry."""

import json

import pytest

from scripts.run_sir_restricted_comparison import remaining_after_attempt


def test_retry_deducts_failed_process(tmp_path):
    neural = tmp_path / "neural"
    ledger = dict(neural_study=str(neural), status="failed", allowance_seconds=100.,
                  charged_seconds=4, remaining_seconds=96.)
    path = tmp_path / "ledger.json"
    path.write_text(json.dumps(ledger))
    assert remaining_after_attempt(neural, 100., tmp_path) == 96.
    for field, value in (("status", "running"), ("neural_study", str(tmp_path)),
                         ("allowance_seconds", 101.), ("charged_seconds", 0),
                         ("remaining_seconds", 97.)):
        path.write_text(json.dumps(dict(ledger, **{field: value})))
        with pytest.raises(ValueError):
            remaining_after_attempt(neural, 100., tmp_path)


def test_retry_chain_preserves_all_charges_and_rejects_cycles(tmp_path):
    neural = tmp_path / "neural"
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    initial = dict(neural_study=str(neural), status="failed", allowance_seconds=100.,
                   charged_seconds=4, remaining_seconds=96.)
    later = dict(initial, allowance_seconds=96., charged_seconds=10, remaining_seconds=86.,
                 previous_attempt=str(first))
    (first / "ledger.json").write_text(json.dumps(initial))
    (second / "ledger.json").write_text(json.dumps(later))
    assert remaining_after_attempt(neural, 100., second) == 86.
    initial["previous_attempt"] = str(second)
    (first / "ledger.json").write_text(json.dumps(initial))
    with pytest.raises(ValueError, match="Cyclic"):
        remaining_after_attempt(neural, 100., second)
