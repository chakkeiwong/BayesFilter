from __future__ import annotations

import json
from pathlib import Path

from docs.benchmarks import run_canonical_lgssm_fused_ot_loop_campaign as campaign


def test_selection_gate_binds_count_campaign_and_disjoint_audit(tmp_path: Path) -> None:
    path = tmp_path / "selection.json"
    path.write_text(
        json.dumps(
            {
                "status": "selection_complete",
                "campaign_id": campaign.CAMPAIGN_ID,
                "selected_balance_steps": 2,
                "dtype": "float32",
                "tf32_enabled": True,
                "time_steps": 2,
                "num_particles": campaign.NUM_PARTICLES,
                "selection_uses_only_marginals": True,
                "design": [{"pass": True}],
                "audit": {"pass": True},
                "attempt_id": "test-selection",
            }
        ),
        encoding="utf-8",
    )
    assert campaign._selection_gate(path, 2)["pass"]
    assert not campaign._selection_gate(path, 3)["pass"]


def test_precision_gate_rejects_source_or_count_mismatch(tmp_path: Path) -> None:
    seeds = list(campaign.SEEDS_BY_HORIZON[2])
    base = {
        "campaign_id": campaign.CAMPAIGN_ID,
        "plan_path": campaign.PLAN_PATH,
        "time_steps": 2,
        "num_particles": campaign.NUM_PARTICLES,
        "balance_steps": 2,
        "estimator_seeds": seeds,
        "hard_valid": True,
        "aggregate_value": -8.0,
        "aggregate_physical_score": [1.0, -1.0, 2.0, 3.0, 4.0],
        "source_sha256": {"source.py": "same"},
    }
    reference = {
        "status": "node_complete",
        "result": {
            **base,
            "device": {"dtype": "float64", "tf32_enabled": False},
        },
    }
    path = tmp_path / "reference.json"
    path.write_text(json.dumps(reference), encoding="utf-8")
    candidate = {
        "status": "node_complete",
        "result": {
            **base,
            "device": {"dtype": "float32", "tf32_enabled": True},
        },
    }
    assert campaign._precision_gate(path, candidate, balance_steps=2)["pass"]
    candidate["result"]["source_sha256"] = {"source.py": "different"}
    assert not campaign._precision_gate(path, candidate, balance_steps=2)["pass"]
