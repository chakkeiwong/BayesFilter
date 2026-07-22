from __future__ import annotations

import hashlib
import json
from pathlib import Path

from docs.benchmarks import (
    run_genut_predator_prey_leaderboard_continuation as campaign,
)


def test_campaign_scope_and_partitions_are_valid() -> None:
    assert campaign.HORIZON == 20
    assert campaign.DGP_SEEDS["claim"] == (81104,)
    assert all(count % 6 == 0 for count in campaign.PARTICLE_COUNTS)
    partitions = [set(campaign.DGP_SEEDS[name]) for name in campaign.DGP_SEEDS]
    assert partitions[0].isdisjoint(partitions[1])
    assert partitions[0].isdisjoint(partitions[2])
    assert partitions[1].isdisjoint(partitions[2])


def test_uncertainty_helpers_distinguish_paired_and_independent() -> None:
    summary = campaign._summary([1.0, 2.0, 3.0, 4.0])  # noqa: SLF001
    assert summary["count"] == 4
    assert summary["ci95_lower"] < summary["mean"] < summary["ci95_upper"]

    left = [[2.0] * 7, [3.0] * 7, [4.0] * 7, [5.0] * 7]
    right = [[1.0] * 7, [2.0] * 7, [3.0] * 7, [4.0] * 7]
    paired = campaign._paired_difference(left, right)  # noqa: SLF001
    assert paired["value"]["mean"] == 1.0
    assert paired["value"]["sample_sd"] == 0.0

    independent = campaign._independent_mean_difference(  # noqa: SLF001
        campaign._summary([2.0, 3.0, 4.0, 5.0]),  # noqa: SLF001
        campaign._summary([1.0, 2.0, 3.0, 4.0]),  # noqa: SLF001
    )
    assert independent["mean"] == 1.0
    assert independent["standard_error"] > 0.0


def test_terminal_artifact_contract_when_present() -> None:
    root = Path(
        "docs/benchmarks/artifacts/"
        "genut_predator_prey_leaderboard_continuation_20260722/attempt01"
    )
    result_path = root / "result.json"
    if not result_path.exists():
        return
    result = json.loads(result_path.read_text(encoding="utf-8"))
    manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["result_sha256"] == hashlib.sha256(
        result_path.read_bytes()
    ).hexdigest()
    assert result["claim"]["criteria"]["score_truth_established"] is False
    assert result["decision"]["leaderboard_admitted"] is False
    assert result["decision"]["default_changed"] is False
    assert result["engineering_ledger"]["repository_identity"] is True
    assert set(result["claim"]["route_identities"]) == {
        str(count) for count in campaign.PARTICLE_COUNTS
    }
