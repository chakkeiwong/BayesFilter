"""Focused policy tests for the generic robust broad-grid tuner."""

from __future__ import annotations

import pytest

import bayesfilter.inference.hmc_robust_broad_grid as robust_module
from bayesfilter.inference.hmc_robust_broad_grid import (
    DEFAULT_L_GRID,
    RobustBroadGridConfig,
    select_robust_candidate,
)


def _row(l: int, ess: float, *, rhat: float = 1.04, decision: str = "passed"):
    return {
        "l": l,
        "candidate_signature": f"candidate-{l}",
        "qualification": {
            "acceptance": {
                "evidence_validity": "valid",
                "acceptance_decision": decision,
                "candidate_promotion_vetoes": (),
                "cost_stop_reasons": (),
            },
            "convergence": {
                "diagnostics_all_finite": True,
                "max_rhat": rhat,
                "min_bulk_ess": ess,
            },
            "native_divergence_count": None,
        },
    }


def test_config_records_inherited_grid_and_500_step_rung_as_provenance() -> None:
    config = RobustBroadGridConfig()
    assert config.l_grid == DEFAULT_L_GRID
    assert config.qualification_results == 500
    assert config.use_xla is True
    assert "not_universal_default" in config.l_grid_provenance
    assert "not_posterior_verification" in config.qualification_rung_provenance
    assert config.acceptance_band == (0.65, 0.75)
    assert config.target_accept_prob == 0.70


def test_config_accepts_target_specific_grid_and_qualification_rung() -> None:
    config = RobustBroadGridConfig(
        l_grid=(2, 4, 8),
        qualification_results=320,
        qualification_burnin_steps=80,
        l_grid_provenance="target_specific_geometry_review",
        qualification_rung_provenance="target_specific_budget_review",
    )

    assert config.l_grid == (2, 4, 8)
    assert config.qualification_results == 320
    assert config.payload()["l_grid_provenance"] == "target_specific_geometry_review"


def test_config_rejects_duplicate_grid() -> None:
    with pytest.raises(ValueError, match="unique"):
        RobustBroadGridConfig(l_grid=(3, 3, 5))


def test_config_rejects_asymmetric_acceptance_band() -> None:
    with pytest.raises(ValueError, match="symmetric"):
        RobustBroadGridConfig(acceptance_band=(0.64, 0.75))


def test_campaign_delegates_mass_preparation_to_public_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def stop_after_public_helper(**kwargs):
        calls.append(kwargs)
        raise RuntimeError("bounded helper sentinel")

    monkeypatch.setattr(
        robust_module,
        "prepare_operational_windowed_mass_handoff",
        stop_after_public_helper,
    )
    adapter = object()
    result = robust_module.tune_hmc_kernel_robust_broad_grid(
        adapter=adapter,
        initial_position=(0.0, 0.0),
        config=RobustBroadGridConfig(
            l_grid=(3,),
            use_xla=False,
            chain_execution_mode="eager",
        ),
    )

    assert len(calls) == 1
    assert calls[0]["adapter"] is adapter
    assert calls[0]["config"].preset == "serious"
    assert result["status"] == "mass_preparation_failed"
    assert result["error_message"] == "bounded helper sentinel"


def test_selector_accepts_valid_inconclusive_tuning_evidence() -> None:
    selected = select_robust_candidate(
        [
            _row(3, 40.0),
            _row(5, 200.0, rhat=1.06),
            _row(9, 80.0),
            _row(13, 500.0, decision="inconclusive_evidence"),
        ]
    )
    assert selected is not None
    assert selected["l"] == 13


def test_selector_rejects_inconclusive_candidate_with_health_veto() -> None:
    row = _row(3, 999.0, decision="inconclusive_evidence")
    row["qualification"]["acceptance"]["candidate_promotion_vetoes"] = (
        "movement_gate_failed",
    )
    assert select_robust_candidate([row, _row(5, 10.0)])["l"] == 5


def test_selector_returns_none_when_all_candidates_fail() -> None:
    assert select_robust_candidate([_row(3, 40.0, rhat=1.10)]) is None


def test_selector_rejects_available_native_divergence() -> None:
    row = _row(3, 999.0)
    row["qualification"]["native_divergence_count"] = 1
    assert select_robust_candidate([row, _row(5, 10.0)]) ["l"] == 5
