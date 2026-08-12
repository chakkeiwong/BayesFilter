from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import pytest

from bayesfilter.inference.hmc_tuning_state import (
    HMCStepRepair,
    HMCTuningTransition,
    aggregate_bracketed_step_repair,
    aggregate_step_repair,
    aggregate_step_veto_bracket_repair,
)
from bayesfilter.inference.hmc_verification import (
    HMCAcceptancePolicy,
    evaluate_hmc_acceptance_evidence,
)


def _evidence(probability: float):
    draw = np.arange(64, dtype=float)[:, None, None]
    chain = np.arange(4, dtype=float)[None, :, None]
    values = np.full((64, 4), probability)
    return evaluate_hmc_acceptance_evidence(
        samples=draw + chain,
        log_accept_ratio=np.log(values),
        is_accepted=np.ones_like(values, dtype=bool),
        policy=HMCAcceptancePolicy(),
    )


def _inconclusive_evidence():
    draw = np.arange(64, dtype=float)[:, None, None]
    chain = np.arange(4, dtype=float)[None, :, None]
    block_probabilities = np.repeat((0.60, 0.80, 0.60, 0.80), 16)
    values = np.repeat(block_probabilities[:, None], 4, axis=1)
    return evaluate_hmc_acceptance_evidence(
        samples=draw + chain,
        log_accept_ratio=np.log(values),
        is_accepted=np.ones_like(values, dtype=bool),
        policy=HMCAcceptancePolicy(),
    )


def _local_veto(*reasons: str):
    values = np.full((64, 4), 0.70)
    return evaluate_hmc_acceptance_evidence(
        samples=np.arange(64, dtype=float)[:, None, None]
        + np.arange(4, dtype=float)[None, :, None],
        log_accept_ratio=np.log(values),
        is_accepted=np.ones_like(values, dtype=bool),
        policy=HMCAcceptancePolicy(),
        candidate_local_health_failures=tuple(reasons),
    )


def test_typed_transition_accepts_legal_and_rejects_illegal_edges() -> None:
    transition = HMCTuningTransition(
        source="metric_frozen",
        target="step_tuned",
        reason="exact-L tune complete",
    )
    assert transition.payload()["target"] == "step_tuned"
    with pytest.raises(ValueError, match="illegal"):
        HMCTuningTransition(source="initialized", target="passed", reason="skip")


@pytest.mark.parametrize(
    ("probability", "direction", "expected"),
    [(0.40, "lower_epsilon", 0.05), (0.90, "higher_epsilon", 0.20)],
)
def test_step_repair_is_directional_monotone_and_bounded(
    probability: float,
    direction: str,
    expected: float,
) -> None:
    repair = aggregate_step_repair(
        (_evidence(probability),) * 3,
        base_step_size=0.1,
        repair_factor=3.0,
        bracket=(0.01, 0.25),
        verification_reserved=True,
    )
    assert repair.disposition == "repair_step"
    assert repair.direction == direction
    assert repair.factor == pytest.approx(2.0)
    assert repair.repaired_step_size == pytest.approx(expected)


def test_repair_aggregation_is_permutation_invariant_and_detects_conflict() -> None:
    low = _evidence(0.40)
    high = _evidence(0.90)
    first = aggregate_step_repair(
        (low, high),
        base_step_size=0.1,
        repair_factor=2.0,
        verification_reserved=True,
    )
    second = aggregate_step_repair(
        (high, low),
        base_step_size=0.1,
        repair_factor=2.0,
        verification_reserved=True,
    )
    assert first == second
    assert first.disposition == "inconclusive_conflict"


def test_candidate_data_invalid_does_not_mask_supported_peer_repair() -> None:
    local_veto = evaluate_hmc_acceptance_evidence(
        samples=np.zeros((64, 4, 1)),
        log_accept_ratio=np.full((64, 4), np.nan),
        is_accepted=np.zeros((64, 4), dtype=bool),
        policy=HMCAcceptancePolicy(),
    )
    repair = aggregate_step_repair(
        (local_veto, _evidence(0.40)),
        base_step_size=0.1,
        repair_factor=2.0,
        verification_reserved=True,
    )
    assert repair.disposition == "repair_step"
    assert repair.direction == "lower_epsilon"


def test_step_aggregation_distinguishes_sticking_from_resonance() -> None:
    values = np.full((64, 4), 0.90)
    sticking = evaluate_hmc_acceptance_evidence(
        samples=np.zeros((64, 4, 1)),
        log_accept_ratio=np.log(values),
        is_accepted=np.ones_like(values, dtype=bool),
        policy=HMCAcceptancePolicy(),
    )

    repair = aggregate_step_repair(
        (sticking,),
        base_step_size=0.1,
        repair_factor=2.0,
        verification_reserved=True,
    )

    assert sticking.acceptance_decision == "repair_trajectory"
    assert "path_return_resonance_detected" not in sticking.candidate_promotion_vetoes
    assert repair.disposition == "inconclusive_trajectory"
    assert repair.direction is None


def test_resonance_alert_does_not_erase_supported_low_step_repair() -> None:
    chain_offsets = np.arange(4, dtype=float)[:, None]
    state_a = np.concatenate((chain_offsets, -chain_offsets), axis=1)
    state_b = state_a + np.array([1.0, -0.5])
    samples = np.stack((state_a, state_b) * 32, axis=0)
    values = np.full((64, 4), 0.40)
    evidence = evaluate_hmc_acceptance_evidence(
        samples=samples,
        log_accept_ratio=np.log(values),
        is_accepted=np.ones_like(values, dtype=bool),
        policy=HMCAcceptancePolicy(),
    )

    repair = aggregate_step_repair(
        (evidence,),
        base_step_size=0.1,
        repair_factor=2.0,
        verification_reserved=True,
    )

    assert "path_return_resonance_detected" in evidence.candidate_promotion_vetoes
    assert evidence.acceptance_decision == "repair_step_lower"
    assert repair.disposition == "repair_step"
    assert repair.direction == "lower_epsilon"
    assert repair.repaired_step_size == pytest.approx(0.05)


def test_repair_requires_reserved_reverification_and_detects_stall_oscillation() -> None:
    no_reservation = aggregate_step_repair(
        (_evidence(0.40),),
        base_step_size=0.1,
        repair_factor=2.0,
        verification_reserved=False,
    )
    assert no_reservation.disposition == "inconclusive_evidence"

    stalled = aggregate_step_repair(
        (_evidence(0.40),),
        base_step_size=0.1,
        repair_factor=2.0,
        direction_history=("lower_epsilon",),
        repaired_step_history=(0.05,),
        verification_reserved=True,
    )
    assert stalled.disposition == "inconclusive_stalled_or_oscillating"

    oscillating = aggregate_step_repair(
        (_evidence(0.40),),
        base_step_size=0.1,
        repair_factor=2.0,
        direction_history=(
            "lower_epsilon",
            "higher_epsilon",
            "lower_epsilon",
        ),
        verification_reserved=True,
    )
    assert oscillating.disposition == "inconclusive_stalled_or_oscillating"


def test_bracketed_repair_updates_empirical_bounds_and_uses_log_midpoint() -> None:
    higher = aggregate_bracketed_step_repair(
        (_evidence(0.90),) * 3,
        base_step_size=0.1,
        repair_factor=2.0,
        verification_reserved=True,
    )
    assert higher.disposition == "repair_step"
    assert higher.direction == "higher_epsilon"
    assert higher.bracket == pytest.approx((0.1, None))
    assert higher.repaired_step_size == pytest.approx(0.2)

    lower = aggregate_bracketed_step_repair(
        (_evidence(0.40),) * 2,
        base_step_size=0.2,
        repair_factor=2.0,
        empirical_bracket=higher.bracket,
        direction_history=("higher_epsilon",),
        repaired_step_history=(0.2,),
        verification_reserved=True,
    )
    assert lower.disposition == "repair_step"
    assert lower.direction == "lower_epsilon"
    assert lower.bracket == pytest.approx((0.1, 0.2))
    assert lower.repaired_step_size == pytest.approx(np.sqrt(0.1 * 0.2))
    assert lower.factor == pytest.approx(np.sqrt(2.0))


def test_bracketed_repair_fails_closed_on_repeated_proposal() -> None:
    stalled = aggregate_bracketed_step_repair(
        (_evidence(0.90),),
        base_step_size=0.1,
        repair_factor=2.0,
        direction_history=("higher_epsilon",),
        repaired_step_history=(0.2,),
        verification_reserved=True,
    )
    assert stalled.disposition == "inconclusive_stalled_or_oscillating"
    assert stalled.repaired_step_size is None


def test_bracketed_repair_refines_one_sided_mixed_neutral_evidence() -> None:
    higher = aggregate_bracketed_step_repair(
        (_evidence(0.90),) * 3,
        base_step_size=0.1,
        repair_factor=2.0,
        verification_reserved=True,
    )
    mixed = aggregate_bracketed_step_repair(
        (_evidence(0.40), _evidence(0.40), _inconclusive_evidence()),
        base_step_size=0.2,
        repair_factor=2.0,
        empirical_bracket=higher.bracket,
        direction_history=("higher_epsilon",),
        repaired_step_history=(0.2,),
        verification_reserved=True,
    )

    assert mixed.disposition == "repair_step"
    assert mixed.direction == "lower_epsilon"
    assert mixed.bracket == pytest.approx((0.1, 0.2))
    assert mixed.repaired_step_size == pytest.approx(np.sqrt(0.1 * 0.2))
    assert mixed.directional_evidence_count == 2
    assert mixed.neutral_evidence_count == 1
    assert mixed.one_sided_directional_support is True
    assert mixed.payload()["neutral_evidence_count"] == 1


def test_bracketed_repair_does_not_mask_conflict_pass_or_invalidity() -> None:
    low = _evidence(0.40)
    high = _evidence(0.90)
    neutral = _inconclusive_evidence()
    conflict = aggregate_bracketed_step_repair(
        (low, high, neutral),
        base_step_size=0.1,
        repair_factor=2.0,
        verification_reserved=True,
    )
    passed = aggregate_bracketed_step_repair(
        (low, _evidence(0.70), neutral),
        base_step_size=0.1,
        repair_factor=2.0,
        verification_reserved=True,
    )
    invalid = aggregate_bracketed_step_repair(
        (_local_veto("nonfinite_log_accept_ratio"), low, neutral),
        base_step_size=0.1,
        repair_factor=2.0,
        verification_reserved=True,
    )

    assert conflict.disposition == "inconclusive_conflict"
    assert conflict.repaired_step_size is None
    assert passed.disposition == "repair_step"
    assert passed.bracket == (None, None)
    assert passed.one_sided_directional_support is False
    assert invalid.disposition == "repair_step"
    assert invalid.direction == "lower_epsilon"
    assert invalid.bracket == (None, None)
    assert invalid.one_sided_directional_support is False


def test_bracketed_mixed_repair_requires_reserved_verification() -> None:
    repair = aggregate_bracketed_step_repair(
        (_evidence(0.40), _inconclusive_evidence()),
        base_step_size=0.2,
        repair_factor=2.0,
        empirical_bracket=(0.1, 0.3),
        verification_reserved=False,
    )

    assert repair.disposition == "inconclusive_evidence"
    assert repair.repaired_step_size is None
    assert repair.bracket == pytest.approx((0.1, 0.2))


@pytest.mark.parametrize(
    "reason",
    ("nonfinite_log_accept_ratio", "nonfinite_target_log_prob"),
)
def test_invalid_candidate_data_never_supplies_step_repair(reason: str) -> None:
    repair = aggregate_step_veto_bracket_repair(
        (_local_veto(reason),) * 3,
        base_step_size=0.2,
        repair_factor=2.0,
        empirical_bracket=(0.1, None),
        verification_reserved=True,
        lower_bound_source_attempt_index=0,
    )

    assert repair.disposition == "candidate_data_invalid"
    assert repair.direction is None
    assert repair.repaired_step_size is None
    assert repair.factor is None
    assert repair.lower_bound_source_attempt_index is None
    assert repair.source_health_failures == ((reason,),) * 3


@pytest.mark.parametrize(
    "evidence",
    (
        (_local_veto("unknown secret=/private/path"),) * 3,
        (
            _local_veto("nonfinite_candidate_state"),
            _local_veto("native_divergence_positive", "unknown=private"),
            _local_veto("nonfinite_target_log_prob"),
        ),
    ),
)
def test_step_veto_recovery_fails_closed_and_redacts_unknown_reasons(evidence) -> None:
    repair = aggregate_step_veto_bracket_repair(
        evidence,
        base_step_size=0.2,
        repair_factor=2.0,
        empirical_bracket=(0.1, None),
        verification_reserved=True,
        lower_bound_source_attempt_index=0,
    )

    assert repair.disposition in {"candidate_data_invalid", "shared_invalidity"}
    serialized = repr(repair.payload())
    assert "private" not in serialized
    assert "secret" not in serialized
    assert "unrecognized_health_failure" in serialized


@pytest.mark.parametrize(
    ("bracket", "source_index"),
    (((None, None), None), ((0.1, None), None)),
)
def test_step_veto_recovery_requires_provenanced_lower_bound(
    bracket,
    source_index,
) -> None:
    repair = aggregate_step_veto_bracket_repair(
        (_local_veto("nonfinite_candidate_state"),) * 3,
        base_step_size=0.2,
        repair_factor=2.0,
        empirical_bracket=bracket,
        verification_reserved=True,
        lower_bound_source_attempt_index=source_index,
    )
    assert repair.disposition == "candidate_data_invalid"
    assert repair.direction is None
    assert repair.repaired_step_size is None
    assert repair.lower_bound_source_attempt_index is None


def test_invalid_data_does_not_consume_reserved_full_matrix_retry() -> None:
    repair = aggregate_step_veto_bracket_repair(
        (_local_veto("nonfinite_log_accept_ratio"),) * 3,
        base_step_size=0.2,
        repair_factor=2.0,
        empirical_bracket=(0.1, None),
        verification_reserved=False,
        lower_bound_source_attempt_index=0,
    )
    assert repair.disposition == "candidate_data_invalid"
    assert repair.repaired_step_size is None
    assert repair.bracket == pytest.approx((0.1, None))


def test_unproven_divergence_reason_never_supplies_midpoint_repair() -> None:
    repair = aggregate_step_veto_bracket_repair(
        (_local_veto("native_divergence_positive"),) * 3,
        base_step_size=0.2,
        repair_factor=2.0,
        empirical_bracket=(0.1, None),
        repaired_step_history=(np.sqrt(0.1 * 0.2),),
        verification_reserved=True,
        lower_bound_source_attempt_index=0,
    )
    assert repair.disposition == "candidate_data_invalid"
    assert repair.repaired_step_size is None


def test_step_repair_rejects_a_factor_that_does_not_match_the_applied_step() -> None:
    with pytest.raises(ValueError, match="realized directional ratio"):
        HMCStepRepair(
            disposition="repair_step",
            direction="lower_epsilon",
            base_step_size=0.2,
            repaired_step_size=np.sqrt(0.1 * 0.2),
            factor=2.0,
            bracket=(0.1, 0.2),
            source_decisions=("repair_step_lower",),
            verification_reserved=True,
        )
