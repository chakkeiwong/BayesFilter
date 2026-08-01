from __future__ import annotations

import json
from pathlib import Path

import pytest

from docs.benchmarks import aggregate_lgssm_particle_bias_ladder as aggregate


ROOT = Path(__file__).resolve().parents[2]
N2000 = ROOT / "docs/benchmarks/artifacts/lgssm_particle_bias_ladder_20260720/n2000_scope_attempt01/campaign_result.json"
N5000 = ROOT / "docs/benchmarks/artifacts/lgssm_particle_bias_ladder_20260720/n5000_repair_scope_attempt01/campaign_result.json"
N5000_FAILED = ROOT / "docs/benchmarks/artifacts/lgssm_particle_bias_ladder_20260720/n5000_scope_attempt02/campaign_result.json"


def test_completed_scope_bindings_are_exact() -> None:
    _, n2000, binding2000 = aggregate._require_scope_campaign(
        N2000, num_particles=2000
    )
    _, n5000, binding5000 = aggregate._require_scope_campaign(
        N5000, num_particles=5000
    )
    assert binding2000["all_valid"]
    assert binding5000["all_valid"]
    assert n2000["preparation_identity"]["transport_block_grid"] == [1, 1]
    assert n5000["preparation_identity"]["transport_block_grid"] == [2, 2]


def test_binding_rejects_wrong_particle_identity(tmp_path: Path) -> None:
    payload = json.loads(N2000.read_text(encoding="utf-8"))
    payload["tuning_scope"]["particle_count"] = 5000
    path = tmp_path / "wrong.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid N=2000 tuning scope"):
        aggregate._require_scope_campaign(path, num_particles=2000)


def test_failed_claim_is_repair_evidence_only() -> None:
    summary = aggregate._failed_claim_summary(N5000_FAILED)
    assert summary["claim_status"] == "FAIL_DIRECT_GATE"
    assert summary["failed_seed_ids"] == [82024, 82027, 82030]
    assert summary["used_for_repair_selection"] is False
    assert summary["used_for_final_bias_screen"] is False


def test_frozen_scopes_reproduce_expected_screen_verdicts() -> None:
    n2000 = aggregate._scope_result(N2000, num_particles=2000)
    n5000 = aggregate._scope_result(N5000, num_particles=5000)
    assert n2000["screen"] == "screen_fail"
    assert n5000["screen"] == "screen_fail"
    assert n2000["relative_error_intervals"]["q_scale"]["mean"] < -0.4
    assert -0.11 < n5000["relative_error_intervals"]["q_scale"]["mean"] < -0.09
