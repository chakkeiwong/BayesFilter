from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_gap_closure_plan_declares_upstream_veto_and_nonclaims() -> None:
    text = (ROOT / "docs/plans/bayesfilter-ssl-lstm-q20-gap-closure-plan-2026-08-18.md").read_text()
    assert "NeuTra retraining" in text
    assert "posterior archive" in text
    assert "No NUTS" in text or "no NUTS" in text
    assert "two-known-region" in text
    assert "Skeptical Plan Audit" in text


def test_dense_material_runner_is_bound_to_fresh_canary_and_dense_mass() -> None:
    text = (ROOT / "docs/benchmarks/run_ssl_lstm_q20_gap_closure_dense_material_2026_08_18.py").read_text()
    assert "r1-dense-mass-step-0p35/canary.json" in text
    assert "CANARY_SHA256" in text
    assert "mass_matrix = tf.constant(canary[\"configuration\"][\"mass_matrix\"]" in text
    assert "master_seed = (20260818, 9101)" not in text
    assert "MASTER_SEED = (20260818, 9101)" in text
    assert "MATERIAL_REPLICA_WARMUP_NOT_READY" in text
