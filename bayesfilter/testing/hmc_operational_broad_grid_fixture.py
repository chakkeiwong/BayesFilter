"""Spawn-worker fixtures for operational broad-grid semantic tests."""

from __future__ import annotations

from bayesfilter.inference.hmc_operational_broad_grid import (
    OperationalBroadGridPolicy,
    OperationalMassHandoff,
    OperationalPrimaryCandidate,
    OperationalPrimaryRequest,
    SameEpsilonNeighborGuard,
    SameEpsilonNeighborGuardRequest,
    classify_operational_pair_evidence,
)


def primary_factory(
    request: OperationalPrimaryRequest,
    policy: OperationalBroadGridPolicy,
    handoff: OperationalMassHandoff,
) -> OperationalPrimaryCandidate:
    evidence = classify_operational_pair_evidence(
        chain_run_means=(0.70,) * policy.evidence_unit_count,
        evidence_signature=f"process-primary-{request.num_leapfrog_steps}",
        policy=policy,
    )
    return OperationalPrimaryCandidate(
        request=request,
        tuned_step_size=0.2 + request.num_leapfrog_steps / 1000.0,
        evidence=evidence,
        metric_signature=handoff.frozen_metric_signature,
        coordinate_signature=handoff.coordinate_signature,
        lineage_signature=handoff.lineage_signature,
        tune_evidence_signature=f"process-tune-{request.num_leapfrog_steps}",
    )


def guard_factory(
    request: SameEpsilonNeighborGuardRequest,
    policy: OperationalBroadGridPolicy,
    handoff: OperationalMassHandoff,
) -> SameEpsilonNeighborGuard:
    del handoff
    evidence = classify_operational_pair_evidence(
        chain_run_means=(0.70,) * policy.evidence_unit_count,
        evidence_signature=(
            f"process-guard-{request.num_leapfrog_steps}-"
            f"{request.inherited_step_size}"
        ),
        policy=policy,
    )
    return SameEpsilonNeighborGuard(request=request, evidence=evidence)


def failing_guard_factory(
    request: SameEpsilonNeighborGuardRequest,
    policy: OperationalBroadGridPolicy,
    handoff: OperationalMassHandoff,
) -> SameEpsilonNeighborGuard:
    if request.num_leapfrog_steps == 2:
        raise RuntimeError("fixture guard worker failure")
    return guard_factory(request, policy, handoff)
