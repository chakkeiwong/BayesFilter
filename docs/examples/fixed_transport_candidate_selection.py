"""Synthetic callback fixture; no training, tuning or HMC is executed.

For a real run, replace validate_rung with a qualified fixed-kernel runner,
archive its draws, and return fixed_transport_candidate_diagnostics using
those draws, matching-coordinate starts, and recorded mechanics telemetry.
"""

from __future__ import annotations

from bayesfilter.inference import (
    FixedTransportCandidateSelectionConfig,
    select_fixed_transport_candidate_set,
)


def validate_rung(candidate: dict[str, object], rung: int) -> dict[str, object]:
    del rung
    return {
        "all_finite": True,
        "target_status_valid": True,
        "all_chain_movement": True,
        "native_divergence_count": 0,
        "max_rhat": 1.005,
        "min_bulk_ess": 500.0,
        "min_tail_ess": 250.0,
        "max_mcse_sd_ratio": 0.05,
        "candidate_label": candidate["candidate_id"],
    }


result = select_fixed_transport_candidate_set(
    [
        {"candidate_id": "restart-a", "whitening_score": (10.0, 2.0)},
        {"candidate_id": "restart-b", "whitening_score": (12.0, 1.0)},
    ],
    validate_rung=validate_rung,
    score_candidate=lambda candidate, _: candidate["whitening_score"],
    config=FixedTransportCandidateSelectionConfig(rungs=2),
    score_metadata={"provenance": "fabricated_interface_fixture", "uncertainty_status": "not_estimated"},
)

assert [row["candidate_id"] for row in result["viable_candidates"]] == [
    "restart-a",
    "restart-b",
]
assert result["selected_candidate"]["candidate_id"] == "restart-a"
assert result["statistical_ranking_supported"] is False
print(result["nomination_status"])
