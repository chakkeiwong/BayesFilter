from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import pytest
import numpy as np

from bayesfilter.inference.hmc_operational_broad_grid import (
    GUARD_ROLE,
    MAX_L,
    MIN_GUARD_L,
    PRIMARY_L_GRID,
    OperationalBroadGridPolicy,
    OperationalBroadGridExecutionConfig,
    OperationalMassHandoff,
    OperationalPairEvidence,
    OperationalPrimaryCandidate,
    OperationalPrimaryRequest,
    OperationalStatisticalEpsilonRepairPolicy,
    SameEpsilonNeighborGuard,
    SameEpsilonNeighborGuardRequest,
    assemble_operational_broad_grid_result,
    classify_operational_pair_evidence,
    classify_operational_statistical_epsilon_evidence,
    advance_operational_statistical_epsilon_repair,
    expand_same_epsilon_neighbor_guards,
    operational_broad_seed,
    primary_requests,
    run_operational_broad_grid,
    run_operational_broad_grid_process_parallel,
)
from bayesfilter.inference.hmc_coordinates import (
    AffineCoordinateTransform,
    MomentumMetric,
    PositionCovarianceEstimate,
)


POLICY = OperationalBroadGridPolicy(
    root_seed=(20260720, 9100),
    confirmation_num_results=128,
)
REPAIR_POLICY = OperationalStatisticalEpsilonRepairPolicy(
    tuning_root_seed=(20260731, 9400),
    qualification_root_seed=(20260731, 9500),
)


def _handoff(*, disposition="dense_update", retained=False):
    prior = "metric-prior"
    frozen = prior if retained else "metric-frozen"
    return OperationalMassHandoff(
        update_disposition=disposition,
        prior_metric_signature=prior,
        frozen_metric_signature=frozen,
        coordinate_signature="coordinate",
        adapter_signature="adapter",
        target_signature="target",
        lineage_signature="lineage",
        canonical_covariance_signature="canonical-covariance",
        latent_metric_signature="latent-metric",
        metric_evidence_signature="evidence",
        retained_prior_metric=retained,
        latent_identity_equivalence_proven=disposition == "dense_update",
    )


def _fixed_identity_handoff():
    return OperationalMassHandoff(
        update_disposition="fixed_identity",
        prior_metric_signature="identity-metric",
        frozen_metric_signature="identity-metric",
        coordinate_signature="coordinate",
        adapter_signature="adapter",
        target_signature="target",
        lineage_signature="lineage",
        canonical_covariance_signature="identity-covariance",
        latent_metric_signature="identity-metric",
        metric_evidence_signature="fixed-identity-policy",
        retained_prior_metric=True,
        latent_identity_equivalence_proven=True,
    )


def _primary(l, *, epsilon=0.2, viable=True):
    request = OperationalPrimaryRequest(
        num_leapfrog_steps=l,
        tune_seed=operational_broad_seed(
            POLICY.root_seed,
            domain="primary_independent_epsilon_tune",
            num_leapfrog_steps=l,
        ),
        mass_handoff_signature=_handoff().signature,
    )
    evidence = classify_operational_pair_evidence(
        chain_run_means=((0.70,) * POLICY.evidence_unit_count),
        evidence_signature=f"evidence-{l}",
        policy=POLICY,
        hard_rejection_reasons=() if viable else ("candidate_invalid",),
    )
    return OperationalPrimaryCandidate(
        request=request,
        tuned_step_size=epsilon,
        evidence=evidence,
        metric_signature="metric-frozen",
        coordinate_signature="coordinate",
        lineage_signature="lineage",
        tune_evidence_signature=f"tune-{l}",
    )


def _guard(request, *, viable=True):
    evidence = classify_operational_pair_evidence(
        chain_run_means=((0.70,) * POLICY.evidence_unit_count),
        evidence_signature=f"guard-evidence-{request.num_leapfrog_steps}-{request.inherited_step_size}",
        policy=POLICY,
        hard_rejection_reasons=() if viable else ("candidate_invalid",),
    )
    return SameEpsilonNeighborGuard(request=request, evidence=evidence)


def test_exact_primary_grid_and_mass_handoff_gate():
    assert PRIMARY_L_GRID == (3, 5, 9, 13, 18, 25)
    assert primary_requests(POLICY, _handoff())
    assert primary_requests(POLICY, _fixed_identity_handoff())
    assert _fixed_identity_handoff().payload()["fixed_identity_metric_preserved"] is True
    assert primary_requests(POLICY, _handoff(disposition="diagonal_fallback", retained=True)) == ()


def test_classification_policy_freezes_replication_t90_interval():
    with pytest.raises(ValueError, match="frozen 90% df=2 interval"):
        OperationalBroadGridPolicy(
            root_seed=(20260720, 9100),
            confirmation_num_results=128,
            working_interval_level=0.95,
        )
    with pytest.raises(ValueError, match="frozen 90% df=2 interval"):
        OperationalBroadGridPolicy(
            root_seed=(20260720, 9100),
            confirmation_num_results=128,
            working_t_critical=4.302652729911275,
        )


def test_fixed_identity_handoff_rejects_metric_or_equivalence_drift():
    payload = dict(
        update_disposition="fixed_identity",
        prior_metric_signature="identity-metric",
        frozen_metric_signature="identity-metric",
        coordinate_signature="coordinate",
        adapter_signature="adapter",
        target_signature="target",
        lineage_signature="lineage",
        canonical_covariance_signature="identity-covariance",
        latent_metric_signature="identity-metric",
        metric_evidence_signature="fixed-identity-policy",
        retained_prior_metric=True,
        latent_identity_equivalence_proven=True,
    )
    with pytest.raises(ValueError, match="metric signature changed"):
        OperationalMassHandoff(**{**payload, "frozen_metric_signature": "drift"})
    with pytest.raises(ValueError, match="canonical/latent equivalence"):
        OperationalMassHandoff(
            **{**payload, "latent_identity_equivalence_proven": False}
        )


def test_primary_and_guard_boundaries_are_strict():
    with pytest.raises(ValueError):
        OperationalPrimaryRequest(
            num_leapfrog_steps=2,
            tune_seed=(1, 2),
            mass_handoff_signature="handoff",
        )
    with pytest.raises(ValueError):
        OperationalPrimaryRequest(
            num_leapfrog_steps=26,
            tune_seed=(1, 2),
            mass_handoff_signature="handoff",
        )
    with pytest.raises(ValueError):
        SameEpsilonNeighborGuardRequest(
            num_leapfrog_steps=1,
            inherited_step_size=0.2,
            parent_candidate_signatures=("parent",),
            parent_l_values=(3,),
            screen_seeds=((1, 2), (3, 4), (5, 6)),
            mass_handoff_signature="handoff",
            metric_signature="metric",
            coordinate_signature="coordinate",
            lineage_signature="lineage",
        )


def test_seed_domains_are_order_independent_and_disjoint():
    primary = operational_broad_seed(
        POLICY.root_seed,
        domain="primary_independent_epsilon_tune",
        num_leapfrog_steps=9,
        epsilon=0.2,
    )
    guard = operational_broad_seed(
        POLICY.root_seed,
        domain="same_epsilon_neighbor_guard_screen",
        num_leapfrog_steps=8,
        epsilon=0.2,
        replication_index=0,
    )
    assert primary != guard
    assert primary == operational_broad_seed(
        POLICY.root_seed,
        domain="primary_independent_epsilon_tune",
        num_leapfrog_steps=9,
        epsilon=0.2,
    )


def test_one_hop_guard_expansion_admits_l2_and_does_not_recurse():
    guards = expand_same_epsilon_neighbor_guards(
        (_primary(3), _primary(5), _primary(25)),
        policy=POLICY,
        handoff=_handoff(),
    )
    assert tuple(item.num_leapfrog_steps for item in guards) == (2, 4, 6, 24)
    assert all(item.num_leapfrog_steps >= MIN_GUARD_L for item in guards)
    assert all(item.num_leapfrog_steps <= MAX_L for item in guards)
    assert all(item.inherited_step_size == 0.2 for item in guards)
    assert all(item.payload()["epsilon_retuned"] is False for item in guards)
    assert all(item.payload()["recursive_expansion_allowed"] is False for item in guards)
    assert all(item.payload()["role"] == GUARD_ROLE for item in guards)
    assert all(
        item.payload()["scientific_role"] == "same_epsilon_neighbor_coverage"
        for item in guards
    )
    assert all(item.payload()["parent_promotion_veto"] is False for item in guards)


def test_guard_pair_deduplication_keeps_distinct_epsilon_pairs():
    guards = expand_same_epsilon_neighbor_guards(
        (_primary(3, epsilon=0.2), _primary(5, epsilon=0.3)),
        policy=POLICY,
        handoff=_handoff(),
    )
    identities = {(item.num_leapfrog_steps, item.inherited_step_size) for item in guards}
    assert identities == {(2, 0.2), (4, 0.2), (4, 0.3), (6, 0.3)}


def test_uncertainty_interval_can_preserve_noisy_candidate_and_hard_veto_wins():
    evidence = classify_operational_pair_evidence(
        chain_run_means=(0.64, 0.68, 0.70, 0.74, 0.76, 0.67, 0.71, 0.72, 0.69, 0.73, 0.66, 0.75),
        evidence_signature="noisy",
        policy=POLICY,
    )
    assert evidence.disposition in {"provisional_viable", "unresolved_budget"}
    veto = classify_operational_pair_evidence(
        chain_run_means=(0.70,) * POLICY.evidence_unit_count,
        evidence_signature="veto",
        policy=POLICY,
        hard_rejection_reasons=("nonfinite_target_state",),
    )
    assert veto.disposition == "hard_rejected"


def test_candidate_local_invalidity_is_a_pair_rejection_without_fake_means():
    evidence = classify_operational_pair_evidence(
        chain_run_means=(),
        evidence_signature="candidate-local-invalid",
        policy=POLICY,
        hard_rejection_reasons=("nonfinite_log_accept_ratio",),
    )
    assert evidence.disposition == "hard_rejected"
    assert evidence.chain_run_means == ()
    assert evidence.replication_means == ()
    assert evidence.grand_mean is None
    assert evidence.working_interval is None
    assert not evidence.viable


def test_missing_or_partial_pair_evidence_still_fails_closed():
    with pytest.raises(ValueError, match="require a hard rejection reason"):
        classify_operational_pair_evidence(
            chain_run_means=(),
            evidence_signature="missing",
            policy=POLICY,
        )
    with pytest.raises(ValueError, match="incomplete or invalid"):
        classify_operational_pair_evidence(
            chain_run_means=(0.70,),
            evidence_signature="partial",
            policy=POLICY,
            hard_rejection_reasons=("candidate_invalid",),
        )


def test_precise_mean_outside_practical_band_is_directional_not_viable():
    above = classify_operational_pair_evidence(
        chain_run_means=(0.7597853586512194,) * POLICY.evidence_unit_count,
        evidence_signature="above-point-mean",
        policy=POLICY,
    )
    below = classify_operational_pair_evidence(
        chain_run_means=(0.649,) * POLICY.evidence_unit_count,
        evidence_signature="below-point-mean",
        policy=POLICY,
    )
    assert above.disposition == "needs_higher_epsilon"
    assert below.disposition == "needs_lower_epsilon"


def test_band_crossing_is_unresolved_not_provisional_viable():
    evidence = classify_operational_pair_evidence(
        chain_run_means=(
            0.54,
            0.58,
            0.62,
            0.66,
            0.68,
            0.70,
            0.72,
            0.74,
            0.74,
            0.78,
            0.82,
            0.86,
        ),
        evidence_signature="boundary-crossing",
        policy=POLICY,
    )
    assert POLICY.practical_region[0] <= evidence.grand_mean <= POLICY.practical_region[1]
    assert evidence.working_interval[0] < POLICY.practical_region[0]
    assert evidence.working_interval[1] > POLICY.practical_region[1]
    assert evidence.disposition == "unresolved_budget"
    assert not evidence.viable


def test_uncertainty_uses_three_replication_means_not_twelve_chain_means():
    evidence = classify_operational_pair_evidence(
        chain_run_means=(
            0.7792324023,
            0.7792324023,
            0.7792324023,
            0.7792324023,
            0.7565863617,
            0.7565863617,
            0.7565863617,
            0.7565863617,
            0.7774131189,
            0.7774131189,
            0.7774131189,
            0.7774131189,
        ),
        evidence_signature="replication-unit",
        policy=POLICY,
    )
    assert evidence.replication_means == pytest.approx(
        (0.7792324023, 0.7565863617, 0.7774131189)
    )
    assert evidence.grand_mean > POLICY.practical_region[1]
    assert evidence.working_interval[0] < POLICY.practical_region[1]
    assert evidence.disposition == "unresolved_budget"
    assert evidence.payload()["working_interval_unit"] == (
        "fresh_seeded_replication_mean_across_chains"
    )


@pytest.mark.parametrize(
    "chain_run_means",
    (
        (
            0.65,
            0.65,
            0.65,
            0.65,
            0.70,
            0.70,
            0.70,
            0.70,
            0.70,
            0.70,
            0.70,
            0.70,
        ),
        (
            0.70,
            0.70,
            0.70,
            0.70,
            0.70,
            0.70,
            0.70,
            0.70,
            0.75,
            0.75,
            0.75,
            0.75,
        ),
    ),
)
def test_interval_that_crosses_one_practical_boundary_is_unresolved(chain_run_means):
    evidence = classify_operational_pair_evidence(
        chain_run_means=chain_run_means,
        evidence_signature="one-boundary-crossing",
        policy=POLICY,
    )

    assert evidence.disposition == "unresolved_budget"
    assert not evidence.viable


def test_interval_contained_in_practical_band_is_provisional_viable():
    evidence = classify_operational_pair_evidence(
        chain_run_means=(0.70,) * POLICY.evidence_unit_count,
        evidence_signature="contained",
        policy=POLICY,
    )

    assert evidence.working_interval == pytest.approx((0.70, 0.70))
    assert evidence.disposition == "provisional_viable"
    assert evidence.viable


def _statistical_evidence(replication_means, *, signature="statistical"):
    chain_means = tuple(
        value
        for replication_mean in replication_means
        for value in (replication_mean,) * REPAIR_POLICY.chain_count
    )
    return classify_operational_statistical_epsilon_evidence(
        chain_run_means=chain_means,
        evidence_signature=signature,
        policy=REPAIR_POLICY,
    )


def test_statistical_repair_policy_requires_disjoint_roots_and_five_units():
    with pytest.raises(ValueError, match="root seeds must be disjoint"):
        OperationalStatisticalEpsilonRepairPolicy(
            tuning_root_seed=(1, 2), qualification_root_seed=(1, 2)
        )
    with pytest.raises(ValueError, match="five replications"):
        OperationalStatisticalEpsilonRepairPolicy(
            tuning_root_seed=(1, 2), qualification_root_seed=(3, 4), replication_count=3
        )
    assert REPAIR_POLICY.evidence_unit_count == 20
    assert REPAIR_POLICY.working_t_critical == pytest.approx(2.1318467863266495)


def test_statistical_evidence_uses_five_replication_means_not_twenty_chains():
    evidence = _statistical_evidence((0.68, 0.69, 0.70, 0.71, 0.72))
    assert evidence.replication_means == pytest.approx((0.68, 0.69, 0.70, 0.71, 0.72))
    assert evidence.disposition == "candidate_nominated"
    assert evidence.candidate_nominated
    assert evidence.payload()["working_interval_limitations"] == (
        "five_replications_only", "shared_calibrated_start", "student_t_working_model",
        "no_familywise_claim", "no_convergence_claim",
    )


def test_statistical_evidence_direction_and_overlap_are_interval_based():
    high = _statistical_evidence((0.79, 0.80, 0.81, 0.80, 0.79), signature="high")
    low = _statistical_evidence((0.59, 0.60, 0.61, 0.60, 0.59), signature="low")
    overlap = _statistical_evidence((0.50, 0.60, 0.70, 0.80, 0.90), signature="overlap")
    assert high.disposition == "needs_higher_epsilon"
    assert low.disposition == "needs_lower_epsilon"
    assert overlap.disposition == "unresolved_budget"


def test_statistical_controller_uses_one_sided_move_then_log_midpoint():
    high = _statistical_evidence((0.79, 0.80, 0.81, 0.80, 0.79), signature="high")
    first = advance_operational_statistical_epsilon_repair(
        evidence=high, current_epsilon=0.2, attempt_index=0,
        bracket_before=(None, None), attempted_epsilons=(0.2,), policy=REPAIR_POLICY,
    )
    assert first.terminal_disposition == "repair_epsilon"
    assert first.direction == "higher_epsilon"
    assert first.bracket_after == pytest.approx((0.2, None))
    assert first.next_epsilon == pytest.approx(0.25)
    low = _statistical_evidence((0.59, 0.60, 0.61, 0.60, 0.59), signature="low")
    second = advance_operational_statistical_epsilon_repair(
        evidence=low, current_epsilon=0.25, attempt_index=1,
        bracket_before=first.bracket_after, attempted_epsilons=(0.2, 0.25), policy=REPAIR_POLICY,
    )
    assert second.direction == "lower_epsilon"
    assert second.bracket_after == pytest.approx((0.2, 0.25))
    assert second.next_epsilon == pytest.approx(np.sqrt(0.2 * 0.25))


def test_statistical_controller_freezes_compatible_candidate_and_stops_unresolved():
    admitted = _statistical_evidence((0.68, 0.69, 0.70, 0.71, 0.72))
    freeze = advance_operational_statistical_epsilon_repair(
        evidence=admitted, current_epsilon=0.22, attempt_index=0,
        bracket_before=(None, None), attempted_epsilons=(0.22,), policy=REPAIR_POLICY,
    )
    assert freeze.terminal_disposition == "freeze_for_qualification"
    assert freeze.next_epsilon is None
    broad_overlap = _statistical_evidence((0.50, 0.60, 0.70, 0.80, 0.90))
    assert broad_overlap.disposition == "unresolved_budget"
    stop = advance_operational_statistical_epsilon_repair(
        evidence=broad_overlap, current_epsilon=0.22, attempt_index=0,
        bracket_before=(None, None), attempted_epsilons=(0.22,), policy=REPAIR_POLICY,
    )
    assert stop.terminal_disposition == "tuning_unresolved"
    assert stop.direction is None


def test_statistical_candidate_nomination_is_compatibility_not_equivalence():
    evidence = _statistical_evidence((0.68, 0.69, 0.70, 0.71, 0.72))
    assert evidence.working_interval[0] >= REPAIR_POLICY.repair_region[0]
    assert evidence.working_interval[1] <= REPAIR_POLICY.repair_region[1]
    assert evidence.disposition == "candidate_nominated"
    assert evidence.payload()["working_interval_role"].endswith("not_equivalence_proof")


def test_statistical_candidate_nomination_rejects_interval_outside_repair_region():
    too_wide = _statistical_evidence((0.10, 0.50, 0.70, 0.90, 0.95))
    assert too_wide.working_interval[0] < REPAIR_POLICY.repair_region[0]
    assert too_wide.working_interval[1] > REPAIR_POLICY.repair_region[1]
    assert too_wide.disposition == "unresolved_budget"


def test_statistical_controller_fails_closed_at_attempt_budget():
    high = _statistical_evidence((0.79, 0.80, 0.81, 0.80, 0.79))
    decision = advance_operational_statistical_epsilon_repair(
        evidence=high, current_epsilon=0.4, attempt_index=4,
        bracket_before=(0.3, None), attempted_epsilons=(0.2, 0.25, 0.3, 0.35, 0.4), policy=REPAIR_POLICY,
    )
    assert decision.terminal_disposition == "attempt_budget_exhausted"
    assert decision.next_epsilon is None


def test_pair_local_guard_failure_does_not_erase_other_epsilon():
    primary = (_primary(3, epsilon=0.2), _primary(5, epsilon=0.3))
    requests = expand_same_epsilon_neighbor_guards(primary, policy=POLICY, handoff=_handoff())
    records = tuple(_guard(request, viable=not (request.num_leapfrog_steps == 4 and request.inherited_step_size == 0.2)) for request in requests)
    assert sum(item.viable for item in records) == len(records) - 1
    assert any(item.request.num_leapfrog_steps == 4 and item.request.inherited_step_size == 0.3 for item in records)


def test_complete_barrier_preserves_every_viable_pair_without_ranking():
    primaries = tuple(_primary(item) for item in PRIMARY_L_GRID)
    guards = tuple(
        _guard(request)
        for request in expand_same_epsilon_neighbor_guards(
            primaries, policy=POLICY, handoff=_handoff()
        )
    )
    result = assemble_operational_broad_grid_result(
        policy=POLICY,
        handoff=_handoff(),
        primary_candidates=primaries,
        guard_candidates=guards,
    )
    assert result.disposition == "viable_pair_set"
    assert result.primary_barrier.complete
    assert result.guard_barrier.complete
    assert result.payload()["representative"] is None
    assert result.public_payload()["epsilon_values_exposed"] is False


def test_next_round_union_keeps_compatible_primaries_and_coverage_points():
    primary_viability = {3: False, 5: True, 9: True, 13: True, 18: True, 25: True}
    primaries = tuple(
        _primary(item, viable=primary_viability[item]) for item in PRIMARY_L_GRID
    )
    coverage_requests = expand_same_epsilon_neighbor_guards(
        primaries, policy=POLICY, handoff=_handoff()
    )
    coverage_viability = {4: False, 6: False, 8: False, 10: False, 12: True, 14: True, 17: True, 19: True, 24: True}
    guards = tuple(
        _guard(request, viable=coverage_viability[request.num_leapfrog_steps])
        for request in coverage_requests
    )
    result = assemble_operational_broad_grid_result(
        policy=POLICY,
        handoff=_handoff(),
        primary_candidates=primaries,
        guard_candidates=guards,
    )
    assert result.next_round_l_values == (5, 9, 12, 13, 14, 17, 18, 19, 24, 25)
    assert tuple(
        item.request.num_leapfrog_steps for item in result.next_round_candidates
    ) == (5, 9, 12, 13, 14, 17, 18, 19, 24, 25)
    assert all(
        item.request.num_leapfrog_steps not in {4, 6, 8, 10}
        for item in result.next_round_candidates
    )
    assert result.public_payload()["stochastic_ranking_performed"] is False
    assert result.public_payload()["next_round_l_values"] == (
        5, 9, 12, 13, 14, 17, 18, 19, 24, 25
    )


def test_failed_coverage_probe_does_not_veto_compatible_parent():
    primaries = tuple(
        _primary(item, viable=item != 3) for item in PRIMARY_L_GRID
    )
    requests = expand_same_epsilon_neighbor_guards(
        primaries, policy=POLICY, handoff=_handoff()
    )
    guards = tuple(
        _guard(request, viable=request.num_leapfrog_steps not in {4, 6})
        for request in requests
    )
    result = assemble_operational_broad_grid_result(
        policy=POLICY,
        handoff=_handoff(),
        primary_candidates=primaries,
        guard_candidates=guards,
    )
    assert result.next_round_l_values == (5, 8, 9, 10, 12, 13, 14, 17, 18, 19, 24, 25)
    assert 5 in result.next_round_l_values
    assert 9 in result.next_round_l_values
    assert 4 not in result.next_round_l_values
    assert 6 not in result.next_round_l_values


def test_incomplete_primary_barrier_is_not_promotable():
    result = assemble_operational_broad_grid_result(
        policy=POLICY,
        handoff=_handoff(),
        primary_candidates=(_primary(3),),
    )
    assert result.disposition == "shared_execution_invalid"
    assert not result.primary_barrier.complete
    assert result.guard_barrier.planned_signatures == ()


def test_affine_whitening_locks_canonical_kinetic_metric_without_double_inversion():
    covariance = np.asarray(((4.0, 1.0), (1.0, 2.0)), dtype=float)
    estimate = PositionCovarianceEstimate(
        center=np.zeros(2),
        covariance=covariance,
        source_coordinate_signature="source",
        estimator_family="fixture",
        state_count=128,
        effective_rank=2,
        regularization_report={"method": "none"},
        adequacy_report={"passed": True},
    )
    transform = AffineCoordinateTransform.from_covariance_estimate(estimate)
    latent_metric = MomentumMetric.identity_for(transform)
    canonical_momentum_covariance = np.linalg.inv(covariance)
    canonical_kinetic_precision = covariance
    assert np.allclose(
        canonical_momentum_covariance @ canonical_kinetic_precision,
        np.eye(2),
    )
    canonical_momentum = np.asarray((0.7, -0.4))
    latent_momentum = canonical_momentum @ transform.factor
    assert np.isclose(
        latent_momentum @ latent_momentum,
        canonical_momentum @ canonical_kinetic_precision @ canonical_momentum,
    )
    assert np.allclose(latent_metric.momentum_covariance, np.eye(2))
    assert np.allclose(latent_metric.kinetic_precision, np.eye(2))


def test_serial_runner_records_failure_as_barrier_not_partial_promotion():
    calls = []

    def primary_runner(request):
        calls.append(request.num_leapfrog_steps)
        if request.num_leapfrog_steps == 9:
            raise RuntimeError("worker failed")
        return _primary(request.num_leapfrog_steps)

    def guard_runner(request):
        return _guard(request)

    result = run_operational_broad_grid(
        policy=POLICY,
        handoff=_handoff(),
        primary_runner=primary_runner,
        guard_runner=guard_runner,
    )
    assert result.disposition == "shared_execution_invalid"
    assert result.guard_candidates == ()
    assert not result.primary_barrier.complete


def _process_execution(*, guard_factory="guard_factory"):
    module = "bayesfilter.testing.hmc_operational_broad_grid_fixture"
    return OperationalBroadGridExecutionConfig(
        mode="process_parallel",
        primary_max_workers=2,
        guard_max_workers=2,
        primary_worker_factory_locator=f"{module}:primary_factory",
        guard_worker_factory_locator=f"{module}:{guard_factory}",
        worker_environment=(("CUDA_VISIBLE_DEVICES", "-1"),),
    )


def _pair_mechanics(result):
    return (
        tuple(item.payload() for item in result.primary_candidates),
        tuple(item.payload() for item in result.guard_candidates),
        result.disposition,
    )


def test_process_parallel_matches_serial_pair_semantics():
    from bayesfilter.testing.hmc_operational_broad_grid_fixture import (
        guard_factory,
        primary_factory,
    )

    serial = run_operational_broad_grid(
        policy=POLICY,
        handoff=_handoff(),
        primary_runner=lambda request: primary_factory(request, POLICY, _handoff()),
        guard_runner=lambda request: guard_factory(request, POLICY, _handoff()),
    )
    parallel = run_operational_broad_grid_process_parallel(
        policy=POLICY,
        handoff=_handoff(),
        execution=_process_execution(),
    )
    assert _pair_mechanics(parallel) == _pair_mechanics(serial)
    assert parallel.execution.mode == "process_parallel"
    assert parallel.primary_barrier.complete
    assert parallel.guard_barrier.complete


def test_process_parallel_guard_failure_invalidates_complete_barrier():
    result = run_operational_broad_grid_process_parallel(
        policy=POLICY,
        handoff=_handoff(),
        execution=_process_execution(guard_factory="failing_guard_factory"),
    )
    assert result.disposition == "shared_execution_invalid"
    assert result.primary_barrier.complete
    assert not result.guard_barrier.complete
    assert any("fixture guard worker failure" in item for item in result.guard_barrier.failure_reasons)


def test_process_configuration_requires_memory_growth_for_gpu_workers():
    module = "bayesfilter.testing.hmc_operational_broad_grid_fixture"
    with pytest.raises(ValueError, match="TF_FORCE_GPU_ALLOW_GROWTH=true"):
        OperationalBroadGridExecutionConfig(
            mode="process_parallel",
            primary_max_workers=6,
            guard_max_workers=6,
            primary_worker_factory_locator=f"{module}:primary_factory",
            guard_worker_factory_locator=f"{module}:guard_factory",
            worker_environment=(("CUDA_VISIBLE_DEVICES", "0"),),
        )
