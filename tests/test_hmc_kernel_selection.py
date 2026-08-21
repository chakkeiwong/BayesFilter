from __future__ import annotations

import os
from dataclasses import replace
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import pytest

from bayesfilter.inference.hmc_kernel_selection import (
    FixedTrajectoryCandidate,
    FixedTrajectoryCandidateRetuneFailure,
    FixedTrajectoryCandidateResult,
    FixedTrajectoryReplication,
    FixedTrajectorySelection,
    FixedTrajectorySelectionRepairAttempt,
    _finalize_operational_selection_nomination,
    _signature,
    _candidate_execution_contract_signature,
    _exact_l_retune_signature,
    private_start_bank_content_signature,
    extend_operational_fixed_trajectory_evidence,
    fixed_trajectory_candidate_values,
    paired_candidate_seed,
    run_bounded_operational_fixed_trajectory_selection,
    run_operational_fixed_trajectory_selection,
    select_fixed_trajectory_representative,
    deterministic_candidate_order,
    build_verified_fixed_kernel_handoff,
)
from bayesfilter.inference.hmc_verification import (
    HMCAcceptancePolicy,
    evaluate_hmc_acceptance_evidence,
)


def _evidence(
    probability: float,
    *,
    stuck: bool = False,
    draw_count: int = 64,
    policy: HMCAcceptancePolicy | None = None,
):
    draws = np.arange(draw_count, dtype=float)[:, None, None]
    chains = np.arange(4, dtype=float)[None, :, None]
    samples = np.zeros((draw_count, 4, 2)) if stuck else draws + chains
    values = np.full((draw_count, 4), probability)
    return evaluate_hmc_acceptance_evidence(
        samples=samples,
        log_accept_ratio=np.log(values),
        is_accepted=np.ones_like(values, dtype=bool),
        policy=HMCAcceptancePolicy() if policy is None else policy,
    )


def _bank_signature(bank, coordinate_signature: str = "coordinate") -> str:
    return private_start_bank_content_signature(bank, coordinate_signature)


def _target_status_trace(
    shape: tuple[int, int],
    *,
    nonvalid: bool = False,
    malformed: bool = False,
):
    payload = {
        "status_code": np.ones(shape, dtype=np.int32)
        if nonvalid
        else np.zeros(shape, dtype=np.int32),
        "valid_pre_regularized_score": np.zeros(shape, dtype=bool)
        if nonvalid
        else np.ones(shape, dtype=bool),
        "floor_count_value": np.zeros(shape, dtype=np.int32),
        "min_innovation_eigenvalue": np.ones(shape),
        "innovation_condition_estimate": np.ones(shape),
    }
    if malformed:
        payload["status_code"] = payload["status_code"][:-1]
    return payload


class _FiniteTargetAdapter:
    def log_prob_and_grad(self, theta):
        values = np.asarray(theta, dtype=float)
        return -0.5 * np.sum(np.square(values), axis=-1), -values

    def target_status_telemetry(self, theta):
        return _target_status_trace(np.asarray(theta).shape[:-1])


def _execution_signature(candidate, index, seed, kwargs) -> str:
    return _candidate_execution_contract_signature(
        candidate=candidate,
        replication_index=index,
        seed=seed,
        frozen_step_size=kwargs["frozen_step_size"],
        num_results=kwargs["screen_num_results"],
        num_burnin_steps=kwargs["screen_num_burnin_steps"],
        target_scope=kwargs["target_scope"],
        acceptance_policy=kwargs["acceptance_policy"],
        target_status_trace_policy=kwargs.get("target_status_trace_policy", "none"),
        chain_execution_mode=kwargs["chain_execution_mode"],
        use_xla=kwargs["use_xla"],
    )


def _candidate(leapfrog: int, *, anchor: int = 10):
    return FixedTrajectoryCandidate(
        anchor_l=anchor,
        num_leapfrog_steps=leapfrog,
        max_leapfrog_steps=64,
        coordinate_signature="coordinate",
        metric_signature="metric",
        start_bank_signature="bank",
    )


def _result(leapfrog: int, probability: float = 0.70, *, retuned: bool = True):
    candidate = _candidate(leapfrog)
    replications = tuple(
        FixedTrajectoryReplication(
            candidate=candidate,
            replication_index=index,
            seed=paired_candidate_seed(
                (20260711, 700),
                candidate_signature=candidate.signature,
                replication_index=index,
            ),
            acceptance_evidence_payload=_evidence(probability).payload(),
            mean_esjd_per_gradient=float(10 - index),
        )
        for index in range(3)
    )
    return FixedTrajectoryCandidateResult(
        candidate=candidate,
        replications=replications,
        exact_l_retuned_step_size=0.1 if retuned else None,
        exact_l_retune_signature="retuned" if retuned else None,
    )


def _local_veto_evidence():
    return evaluate_hmc_acceptance_evidence(
        samples=np.zeros((64, 4, 2)),
        log_accept_ratio=np.full((64, 4), np.nan),
        is_accepted=np.zeros((64, 4), dtype=bool),
        policy=HMCAcceptancePolicy(),
    )


def _local_veto_with_reasons(*reasons: str):
    values = np.full((64, 4), 0.70)
    return evaluate_hmc_acceptance_evidence(
        samples=np.arange(64, dtype=float)[:, None, None]
        + np.arange(4, dtype=float)[None, :, None],
        log_accept_ratio=np.log(values),
        is_accepted=np.ones_like(values, dtype=bool),
        policy=HMCAcceptancePolicy(),
        candidate_local_health_failures=tuple(reasons),
    )


def _shared_invalidity_evidence():
    values = np.full((64, 4), 0.70)
    return evaluate_hmc_acceptance_evidence(
        samples=np.arange(64, dtype=float)[:, None, None]
        + np.arange(4, dtype=float)[None, :, None],
        log_accept_ratio=np.log(values),
        is_accepted=np.ones_like(values, dtype=bool),
        policy=HMCAcceptancePolicy(),
        shared_invalidity_reasons=("test_shared_invalidity",),
    )


def _native_divergence_veto_evidence(draw_count: int = 64):
    values = np.full((draw_count, 4), 0.70)
    return evaluate_hmc_acceptance_evidence(
        samples=np.arange(draw_count, dtype=float)[:, None, None]
        + np.arange(4, dtype=float)[None, :, None],
        log_accept_ratio=np.log(values),
        is_accepted=np.ones_like(values, dtype=bool),
        policy=HMCAcceptancePolicy(),
        native_divergence_status="available",
        native_divergence_count=1,
    )


def _inconclusive_evidence(
    *,
    draw_count: int = 64,
    policy: HMCAcceptancePolicy | None = None,
):
    if draw_count % 4:
        raise ValueError("inconclusive fixture requires four equal blocks")
    samples = np.arange(draw_count, dtype=float)[:, None, None] + np.arange(
        4, dtype=float
    )[None, :, None]
    block_probabilities = np.repeat(
        (0.60, 0.80, 0.60, 0.80),
        draw_count // 4,
    )
    probabilities = np.repeat(block_probabilities[:, None], 4, axis=1)
    evidence = evaluate_hmc_acceptance_evidence(
        samples=samples,
        log_accept_ratio=np.log(probabilities),
        is_accepted=np.ones_like(probabilities, dtype=bool),
        policy=HMCAcceptancePolicy() if policy is None else policy,
    )
    assert evidence.decision == "inconclusive_evidence"
    return evidence


def _result_from_evidence(leapfrog: int, evidence) -> FixedTrajectoryCandidateResult:
    candidate = _candidate(leapfrog)
    records = evidence if isinstance(evidence, tuple) else (evidence,) * 3
    return FixedTrajectoryCandidateResult(
        candidate=candidate,
        replications=tuple(
            FixedTrajectoryReplication(
                candidate=candidate,
                replication_index=index,
                seed=(20260711 + leapfrog, index + 1),
                acceptance_evidence_payload=record.payload(),
            )
            for index, record in enumerate(records)
        ),
    )


def test_candidate_values_are_bounded_distinct_and_sorted() -> None:
    assert fixed_trajectory_candidate_values(10, max_leapfrog_steps=64) == (5, 10, 20)
    assert fixed_trajectory_candidate_values(1, max_leapfrog_steps=1) == (1,)
    assert fixed_trajectory_candidate_values(40, max_leapfrog_steps=64) == (20, 40, 64)


def test_paired_seed_map_is_candidate_order_independent() -> None:
    candidates = [_candidate(value) for value in (5, 10, 20)]
    first = {
        candidate.signature: tuple(
            paired_candidate_seed(
                (20260711, 700),
                candidate_signature=candidate.signature,
                replication_index=index,
            )
            for index in range(3)
        )
        for candidate in candidates
    }
    second = {
        candidate.signature: tuple(
            paired_candidate_seed(
                (20260711, 700),
                candidate_signature=candidate.signature,
                replication_index=index,
            )
            for index in range(3)
        )
        for candidate in reversed(candidates)
    }
    assert first == second
    assert len({seed for seeds in first.values() for seed in seeds}) == 9


def test_representative_policy_is_permutation_invariant_and_not_efficiency_ranked() -> None:
    results = (_result(5), _result(10), _result(20))
    first = select_fixed_trajectory_representative(results, anchor_l=10)
    second = select_fixed_trajectory_representative(reversed(results), anchor_l=10)

    assert first.signature == second.signature
    assert first.representative.candidate.num_leapfrog_steps == 10
    assert first.payload()["stochastic_ranking_performed"] is False


def test_arbitrary_candidate_order_is_deterministic_and_ignores_metrics() -> None:
    candidates = (
        {"L": 5, "signature": "b", "acceptance": 0.99, "runtime": 1.0},
        {"L": 3, "signature": "c", "acceptance": 0.60, "runtime": 9.0},
        {"L": 5, "signature": "a", "acceptance": 0.10, "runtime": 2.0},
    )
    ordered = deterministic_candidate_order(candidates, anchor_l=4)
    assert tuple(item["signature"] for item in ordered) == ("c", "a", "b")
    reversed_metrics = tuple(
        {**item, "acceptance": 1.0 - item["acceptance"], "runtime": 100.0 - item["runtime"]}
        for item in reversed(candidates)
    )
    assert tuple(
        item["signature"] for item in deterministic_candidate_order(reversed_metrics, anchor_l=4)
    ) == ("c", "a", "b")


def test_verified_handoff_is_bayesfilter_owned_and_uses_policy_only() -> None:
    candidates = (
        {
            "L": 5,
            "step_size": 0.24,
            "signature": "l5",
            "lineage_signature": "lineage",
            "verification_checkpoint_sha256": "5" * 64,
            "qualification_passed": True,
            "acceptance": 0.99,
        },
        {
            "L": 3,
            "step_size": 0.29,
            "signature": "l3",
            "lineage_signature": "lineage",
            "verification_checkpoint_sha256": "3" * 64,
            "qualification_passed": True,
            "acceptance": 0.01,
        },
    )
    handoff = build_verified_fixed_kernel_handoff(candidates, anchor_l=4)
    assert handoff is not None
    assert handoff.num_leapfrog_steps == 3
    assert handoff.payload()["stochastic_ranking_performed"] is False
    assert handoff.payload()["retained_sampling_authorized"] is False


def test_candidate_result_uses_explicit_v3_schema_for_repair_role_semantics() -> None:
    result = _result(10)

    assert result.payload()["schema"] == (
        "bayesfilter.hmc_fixed_trajectory_candidate_result.v3"
    )
    assert result.signature == _signature(
        "bayesfilter.hmc_fixed_trajectory_candidate_result.v3",
        result.payload(),
    )
    assert "resonance_repair_detected" in result.payload()
    assert "trajectory_repair_detected" in result.payload()


def test_representative_requires_exact_final_l_epsilon_retune() -> None:
    selection = select_fixed_trajectory_representative(
        (_result(10, retuned=False),), anchor_l=10
    )
    assert selection.disposition == "representative_nominated"


def test_candidate_batch_conflict_and_sticking_are_scoped() -> None:
    low = _result(5, 0.40)
    high = _result(10, 0.90)
    conflict = select_fixed_trajectory_representative((low, high), anchor_l=10)
    assert conflict.disposition == "inconclusive_conflict"

    candidate = _candidate(10)
    sticking = FixedTrajectoryCandidateResult(
        candidate=candidate,
        replications=tuple(
            FixedTrajectoryReplication(
                candidate=candidate,
                replication_index=index,
                seed=(1, index + 1),
                acceptance_evidence_payload=_evidence(0.90, stuck=True).payload(),
            )
            for index in range(3)
        ),
    )
    assert select_fixed_trajectory_representative(
        (sticking,), anchor_l=10
    ).disposition == "inconclusive_trajectory"
    assert sticking.trajectory_repair_detected is True
    assert sticking.resonance_detected is False


def test_mixed_local_veto_does_not_erase_supported_peer_repair() -> None:
    local = _local_veto_evidence()
    lower = _evidence(0.40)
    mixed = _result_from_evidence(5, (local, lower, lower))
    exhausted_10 = _result_from_evidence(10, local)
    exhausted_20 = _result_from_evidence(20, local)

    selection = select_fixed_trajectory_representative(
        (exhausted_20, mixed, exhausted_10), anchor_l=10
    )
    assert selection.disposition == "repair_required"
    assert selection.representative is None


def test_resonance_alert_does_not_erase_supported_low_step_repair() -> None:
    candidate = _candidate(10)
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
    result = FixedTrajectoryCandidateResult(
        candidate=candidate,
        replications=tuple(
            FixedTrajectoryReplication(
                candidate=candidate,
                replication_index=index,
                seed=(2, index + 1),
                acceptance_evidence_payload=evidence.payload(),
            )
            for index in range(3)
        ),
    )

    selection = select_fixed_trajectory_representative((result,), anchor_l=10)

    assert evidence.acceptance_decision == "repair_step_lower"
    assert result.resonance_detected is True
    assert result.resonance_repair_detected is False
    assert result.trajectory_repair_detected is False
    assert selection.disposition == "repair_required"


def test_neutral_pass_blocks_global_repair_pending_evidence_extension() -> None:
    higher = _evidence(0.90)
    inconclusive = _inconclusive_evidence()
    partly_supported = _result_from_evidence(
        5,
        (higher, higher, higher),
    )
    passed = _evidence(0.70)
    assert passed.decision == "passed"
    nearest = _result_from_evidence(10, (passed, inconclusive, inconclusive))
    other = _result_from_evidence(20, inconclusive)

    selection = select_fixed_trajectory_representative(
        (partly_supported, nearest, other), anchor_l=10
    )
    assert selection.disposition == "inconclusive_evidence"
    assert selection.representative is None


def test_complete_inconclusive_candidate_blocks_peer_step_repair_for_extension() -> None:
    higher = _evidence(0.90)
    inconclusive = _inconclusive_evidence()
    directional = _result_from_evidence(5, higher)
    eligible = _result_from_evidence(10, inconclusive)
    peer = _result_from_evidence(20, higher)

    selection = select_fixed_trajectory_representative(
        (peer, eligible, directional), anchor_l=10
    )

    assert selection.disposition == "inconclusive_evidence"
    assert eligible.evidence_extension_eligible is True
    assert directional.evidence_extension_eligible is False
    assert peer.evidence_extension_eligible is False


def test_shared_invalidity_and_trajectory_pathology_precede_extension() -> None:
    inconclusive = _inconclusive_evidence()
    eligible = _result_from_evidence(10, inconclusive)
    shared = _result_from_evidence(5, _shared_invalidity_evidence())
    sticking = _result_from_evidence(20, _evidence(0.90, stuck=True))

    shared_selection = select_fixed_trajectory_representative(
        (eligible, shared), anchor_l=10
    )
    trajectory_selection = select_fixed_trajectory_representative(
        (eligible, sticking), anchor_l=10
    )

    assert eligible.evidence_extension_eligible is True
    assert shared_selection.disposition == "shared_invalidity"
    assert trajectory_selection.disposition == "inconclusive_trajectory"


def test_candidate_set_exhaustion_requires_every_replication_local_veto() -> None:
    local = _local_veto_evidence()
    selection = select_fixed_trajectory_representative(
        tuple(_result_from_evidence(leapfrog, local) for leapfrog in (5, 10, 20)),
        anchor_l=10,
    )
    assert selection.disposition == "candidate_set_exhausted"


def test_shared_invalidity_precedes_an_otherwise_viable_candidate() -> None:
    shared = _result_from_evidence(5, _shared_invalidity_evidence())
    viable = _result(10)
    selection = select_fixed_trajectory_representative(
        (viable, shared), anchor_l=10
    )
    assert selection.disposition == "shared_invalidity"
    assert selection.representative is None


@pytest.mark.parametrize(
    ("results", "disposition"),
    [
        (lambda: (_result(10, retuned=False),), "repair_required"),
        (lambda: (_result(5, 0.40), _result(10, 0.90)), "inconclusive_evidence"),
        (lambda: (_result(5, 0.40),), "inconclusive_conflict"),
    ],
)
def test_selection_type_rejects_disposition_inconsistent_with_evidence(
    results,
    disposition: str,
) -> None:
    with pytest.raises(ValueError, match="inconsistent with completed evidence"):
        FixedTrajectorySelection(
            anchor_l=10,
            candidate_results=results(),
            representative_signature=None,
            disposition=disposition,
        )


def test_passing_candidate_with_promotion_veto_cannot_be_selected() -> None:
    vetoed = _result_from_evidence(10, _native_divergence_veto_evidence())
    viable = _result(20)

    selection = select_fixed_trajectory_representative(
        (vetoed, viable), anchor_l=10
    )

    assert vetoed.decisions == ("passed",) * 3
    assert vetoed.viable is False
    assert selection.representative is viable
    assert selection.representative.candidate.num_leapfrog_steps == 20


def _scripted_selection(kwargs, probability: float, *, selected: bool):
    results = []
    for leapfrog in (5, 10, 20):
        candidate = FixedTrajectoryCandidate(
            anchor_l=10,
            num_leapfrog_steps=leapfrog,
            max_leapfrog_steps=64,
            coordinate_signature=kwargs["coordinate_signature"],
            metric_signature=kwargs["metric_signature"],
            start_bank_signature=kwargs["private_start_bank_signature"],
        )
        replications = tuple(
            FixedTrajectoryReplication(
                candidate=candidate,
                replication_index=index,
                seed=(seed := paired_candidate_seed(
                    kwargs["root_seed"],
                    candidate_signature=candidate.signature,
                    replication_index=index,
                )),
                acceptance_evidence_payload=_evidence(
                    probability,
                    draw_count=kwargs["screen_num_results"],
                ).payload(),
                execution_contract_signature=_execution_signature(
                    candidate,
                    index,
                    seed,
                    kwargs,
                ),
            )
            for index in range(3)
        )
        retuned_step = (
            kwargs["frozen_step_size"] if selected and leapfrog == 10 else None
        )
        results.append(
            FixedTrajectoryCandidateResult(
                candidate=candidate,
                replications=replications,
                exact_l_retuned_step_size=retuned_step,
                exact_l_retune_signature=(
                    _exact_l_retune_signature(
                        candidate=candidate,
                        root_seed=kwargs["root_seed"],
                        adaptation_steps=kwargs["final_tune_adaptation_steps"],
                        initial_step_size=kwargs["frozen_step_size"],
                        retuned_step_size=retuned_step,
                        target_scope=kwargs["target_scope"],
                        acceptance_policy=kwargs["acceptance_policy"],
                        target_status_trace_policy=kwargs.get(
                            "target_status_trace_policy", "none"
                        ),
                        chain_execution_mode=kwargs["chain_execution_mode"],
                        use_xla=kwargs["use_xla"],
                    )
                    if retuned_step is not None
                    else None
                ),
            )
        )
    return select_fixed_trajectory_representative(results, anchor_l=10)


def _matrix_selection(
    kwargs,
    evidence_by_l,
    *,
    retuned_l: int | None = None,
):
    results = []
    for leapfrog in (5, 10, 20):
        candidate = FixedTrajectoryCandidate(
            anchor_l=10,
            num_leapfrog_steps=leapfrog,
            max_leapfrog_steps=64,
            coordinate_signature=kwargs["coordinate_signature"],
            metric_signature=kwargs["metric_signature"],
            start_bank_signature=kwargs["private_start_bank_signature"],
        )
        evidence = evidence_by_l[leapfrog]
        evidence_records = evidence if isinstance(evidence, tuple) else (evidence,) * 3
        replications = tuple(
            FixedTrajectoryReplication(
                candidate=candidate,
                replication_index=index,
                seed=(seed := paired_candidate_seed(
                    kwargs["root_seed"],
                    candidate_signature=candidate.signature,
                    replication_index=index,
                    domain="candidate_selection",
                )),
                acceptance_evidence_payload=record.payload(),
                execution_contract_signature=_execution_signature(
                    candidate,
                    index,
                    seed,
                    kwargs,
                ),
            )
            for index, record in enumerate(evidence_records)
        )
        retuned_step = kwargs["frozen_step_size"] if leapfrog == retuned_l else None
        results.append(
            FixedTrajectoryCandidateResult(
                candidate=candidate,
                replications=replications,
                exact_l_retuned_step_size=retuned_step,
                exact_l_retune_signature=(
                    _exact_l_retune_signature(
                        candidate=candidate,
                        root_seed=kwargs["root_seed"],
                        adaptation_steps=kwargs["final_tune_adaptation_steps"],
                        initial_step_size=kwargs["frozen_step_size"],
                        retuned_step_size=retuned_step,
                        target_scope=kwargs["target_scope"],
                        acceptance_policy=kwargs["acceptance_policy"],
                        target_status_trace_policy=kwargs.get(
                            "target_status_trace_policy", "none"
                        ),
                        chain_execution_mode=kwargs["chain_execution_mode"],
                        use_xla=kwargs["use_xla"],
                    )
                    if retuned_step is not None
                    else None
                ),
            )
        )
    return select_fixed_trajectory_representative(results, anchor_l=10)


def _fake_extension_runner(
    *,
    probability_by_l=None,
    inconclusive: bool = False,
    local_veto: bool = False,
    shared_invalidity_seed: tuple[int, int] | None = None,
):
    calls = []

    def runner(_adapter, initial_state, config):
        calls.append((np.asarray(initial_state).copy(), config))
        if config.tuning_policy.uses_dual_averaging:
            return SimpleNamespace(
                samples=np.zeros((4, 4, 2)),
                trace={
                    "log_accept_ratio": np.zeros((4, 4)),
                    "is_accepted": np.ones((4, 4), dtype=bool),
                    "target_log_prob": np.zeros((4, 4)),
                    "step_size": np.full(4, 0.125),
                },
                diagnostics={
                    "final_step_size": 0.125,
                    "native_divergence_status": "not_exposed_by_kernel",
                    "divergence_count": None,
                },
            )

        draw_count = config.num_results
        draw = np.arange(draw_count, dtype=float)[:, None, None]
        samples = draw + np.asarray(initial_state)[None, :, :]
        if inconclusive:
            block = np.repeat(
                (0.60, 0.80, 0.60, 0.80),
                draw_count // 4,
            )
            probabilities = np.repeat(block[:, None], 4, axis=1)
        else:
            probability = (
                0.70
                if probability_by_l is None
                else probability_by_l[config.num_leapfrog_steps]
            )
            probabilities = np.full((draw_count, 4), probability)
        log_accept_ratio = np.log(probabilities)
        if local_veto:
            log_accept_ratio[:] = np.nan
        trace = {
            "log_accept_ratio": log_accept_ratio,
            "is_accepted": np.ones_like(probabilities, dtype=bool),
            "target_log_prob": np.zeros_like(probabilities),
        }
        if config.seed == shared_invalidity_seed:
            trace = {}
        return SimpleNamespace(
            samples=samples,
            trace=trace,
            diagnostics={
                "native_divergence_status": "not_exposed_by_kernel",
                "divergence_count": None,
            },
        )

    return runner, calls


def _bounded_extension_kwargs(*, selector, runner, evidence_extender=None):
    bank = np.arange(8, dtype=float).reshape(4, 2)
    kwargs = {
        "adapter": _FiniteTargetAdapter(),
        "private_start_bank": bank,
        "private_start_bank_signature": _bank_signature(bank),
        "coordinate_signature": "coordinate",
        "metric_signature": "metric",
        "anchor_l": 10,
        "max_leapfrog_steps": 64,
        "initial_step_size": 0.1,
        "root_seed": (20260711, 700),
        "target_scope": "test",
        "acceptance_policy": HMCAcceptancePolicy(),
        "max_attempts": 5,
        "screen_num_results": 256,
        "screen_num_burnin_steps": 16,
        "final_tune_adaptation_steps": 64,
        "run_full_chain": runner,
        "selector": selector,
        "evidence_extension_checkpoints": (512, 1024),
    }
    if evidence_extender is not None:
        kwargs["evidence_extender"] = evidence_extender
    return kwargs


def test_bounded_selection_uses_empirical_bracket_and_disjoint_lineage() -> None:
    calls = []
    bank = np.arange(8, dtype=float).reshape(4, 2)
    bank_signature = _bank_signature(bank)

    def selector(**kwargs):
        calls.append(kwargs)
        probabilities = (0.90, 0.40, 0.70)
        return _scripted_selection(
            kwargs,
            probabilities[len(calls) - 1],
            selected=len(calls) == 3,
        )

    result = run_bounded_operational_fixed_trajectory_selection(
        adapter=_FiniteTargetAdapter(),
        private_start_bank=bank,
        private_start_bank_signature=bank_signature,
        coordinate_signature="coordinate",
        metric_signature="metric",
        anchor_l=10,
        max_leapfrog_steps=64,
        initial_step_size=0.1,
        root_seed=(20260711, 700),
        target_scope="test",
        acceptance_policy=HMCAcceptancePolicy(),
        max_attempts=5,
        selector=selector,
    )

    assert result.terminal_disposition == "representative_selected"
    assert result.repair_direction_history == (
        "higher_epsilon",
        "lower_epsilon",
    )
    assert result.repaired_step_history == pytest.approx(
        (0.2, np.sqrt(0.1 * 0.2))
    )
    assert result.final_bracket == pytest.approx((0.1, 0.2))
    assert len({attempt.root_seed for attempt in result.attempts}) == 3
    all_seeds = tuple(
        seed for attempt in result.attempts for seed in attempt.replication_seeds
    )
    assert len(set(all_seeds)) == 27
    assert all(
        result_item.candidate.coordinate_signature == "coordinate"
        and result_item.candidate.metric_signature == "metric"
        and result_item.candidate.start_bank_signature == bank_signature
        for attempt in result.attempts
        for result_item in attempt.selection.candidate_results
    )


def test_bounded_selection_refines_midas_like_mixed_directional_midpoint() -> None:
    calls = []
    high = _evidence(0.90, draw_count=256)
    low = _evidence(0.40, draw_count=256)
    neutral = _inconclusive_evidence(draw_count=256)

    def selector(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return _matrix_selection(kwargs, {5: high, 10: high, 20: high})
        if len(calls) == 2:
            return _matrix_selection(kwargs, {5: low, 10: low, 20: low})
        if len(calls) == 3:
            return _matrix_selection(
                kwargs,
                {5: low, 10: (low, low, neutral), 20: low},
            )
        return _scripted_selection(kwargs, 0.70, selected=True)

    runner, _runner_calls = _fake_extension_runner(inconclusive=True)
    kwargs = _bounded_extension_kwargs(selector=selector, runner=runner)
    kwargs["evidence_extension_checkpoints"] = ()
    result = run_bounded_operational_fixed_trajectory_selection(**kwargs)

    first = 0.1
    doubled = 2.0 * first
    first_midpoint = np.sqrt(first * doubled)
    second_midpoint = np.sqrt(first * first_midpoint)
    assert result.terminal_disposition == "representative_selected"
    assert tuple(call["frozen_step_size"] for call in calls) == pytest.approx(
        (first, doubled, first_midpoint, second_midpoint)
    )
    assert result.repair_direction_history == (
        "higher_epsilon",
        "lower_epsilon",
        "lower_epsilon",
    )
    mixed_repair = result.attempts[2].repair
    assert mixed_repair is not None
    assert mixed_repair.directional_evidence_count == 8
    assert mixed_repair.neutral_evidence_count == 1
    assert mixed_repair.one_sided_directional_support is True
    assert mixed_repair.bracket == pytest.approx((first, first_midpoint))
    assert mixed_repair.repaired_step_size == pytest.approx(second_midpoint)


def test_mixed_high_neutral_bound_retains_append_only_source_provenance() -> None:
    calls = []
    high = _evidence(0.90, draw_count=256)
    low = _evidence(0.40, draw_count=256)
    neutral = _inconclusive_evidence(draw_count=256)

    def selector(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return _matrix_selection(
                kwargs,
                {5: high, 10: (high, high, neutral), 20: high},
            )
        if len(calls) == 2:
            return _matrix_selection(kwargs, {5: low, 10: low, 20: low})
        return _scripted_selection(kwargs, 0.70, selected=True)

    runner, _runner_calls = _fake_extension_runner(inconclusive=True)
    kwargs = _bounded_extension_kwargs(selector=selector, runner=runner)
    kwargs["evidence_extension_checkpoints"] = ()
    result = run_bounded_operational_fixed_trajectory_selection(**kwargs)

    assert result.terminal_disposition == "representative_selected"
    first_attempt = result.attempts[0]
    assert first_attempt.repair is not None
    assert first_attempt.repair.one_sided_directional_support is True
    assert first_attempt.lower_bound_source_attempt_index_after == 0
    assert result.attempts[1].lower_bound_source_attempt_index_before == 0
    assert result.final_bracket == pytest.approx((0.1, 0.2))


def test_bounded_selection_exhaustion_is_non_promoting_and_five_attempts() -> None:
    calls = []
    bank = np.arange(8, dtype=float).reshape(4, 2)

    def selector(**kwargs):
        calls.append(kwargs)
        return _scripted_selection(kwargs, 0.90, selected=False)

    result = run_bounded_operational_fixed_trajectory_selection(
        adapter=_FiniteTargetAdapter(),
        private_start_bank=bank,
        private_start_bank_signature=_bank_signature(bank),
        coordinate_signature="coordinate",
        metric_signature="metric",
        anchor_l=10,
        max_leapfrog_steps=64,
        initial_step_size=0.1,
        root_seed=(20260711, 700),
        target_scope="test",
        acceptance_policy=HMCAcceptancePolicy(),
        max_attempts=5,
        selector=selector,
    )

    assert result.terminal_disposition == "budget_exhausted_valid"
    assert len(result.attempts) == 5
    assert tuple(call["frozen_step_size"] for call in calls) == pytest.approx(
        (0.1, 0.2, 0.4, 0.8, 1.6)
    )
    assert result.representative is None


def test_operational_selection_uses_frozen_bank_paired_matrix_and_exact_l_retune() -> None:
    calls = []
    bank = np.arange(8, dtype=float).reshape(4, 2)

    class _Run:
        pass

    def runner(_adapter, initial_state, config):
        calls.append((np.asarray(initial_state).copy(), config))
        result = _Run()
        if config.tuning_policy.uses_dual_averaging:
            result.samples = np.zeros((4, 4, 2))
            result.trace = {
                "log_accept_ratio": np.zeros((4, 4)),
                "is_accepted": np.ones((4, 4), dtype=bool),
                "target_log_prob": np.zeros((4, 4)),
                "step_size": np.full(4, 0.125),
            }
            result.diagnostics = {
                "final_step_size": 0.125,
                "native_divergence_status": "not_exposed_by_kernel",
                "divergence_count": None,
            }
            return result
        draw = np.arange(config.num_results, dtype=float)[:, None, None]
        result.samples = draw + bank[None, :, :]
        probability = np.full((config.num_results, 4), 0.70)
        result.trace = {
            "log_accept_ratio": np.log(probability),
            "is_accepted": np.ones_like(probability, dtype=bool),
            "target_log_prob": np.zeros_like(probability),
        }
        result.diagnostics = {
            "native_divergence_status": "not_exposed_by_kernel",
            "divergence_count": None,
        }
        return result

    selection = run_operational_fixed_trajectory_selection(
        adapter=_FiniteTargetAdapter(),
        private_start_bank=bank,
        private_start_bank_signature=_bank_signature(bank),
        coordinate_signature="coordinate",
        metric_signature="metric",
        anchor_l=10,
        max_leapfrog_steps=64,
        frozen_step_size=0.1,
        root_seed=(20260711, 700),
        target_scope="test",
        acceptance_policy=HMCAcceptancePolicy(),
        run_full_chain=runner,
    )

    assert selection.disposition == "representative_selected"
    assert selection.representative.candidate.num_leapfrog_steps == 10
    assert selection.representative.exact_l_retuned_step_size == pytest.approx(0.125)
    assert len(calls) == 10
    assert all(np.array_equal(call_bank, bank) for call_bank, _config in calls)
    assert len({config.seed for _bank, config in calls}) == 10
    assert all(
        not config.tuning_policy.uses_dual_averaging for _bank, config in calls[:9]
    )
    assert calls[-1][1].num_leapfrog_steps == 10
    assert calls[-1][1].tuning_policy.uses_dual_averaging


def test_operational_selection_persists_period_two_resonance_and_blocks_retune() -> None:
    calls = []
    bank = np.arange(8, dtype=float).reshape(4, 2)

    def runner(_adapter, initial_state, config):
        calls.append(config)
        state_a = np.asarray(initial_state, dtype=float)
        state_b = state_a + np.array([1.0, -0.5])
        samples = np.stack((state_a, state_b) * (config.num_results // 2), axis=0)
        probability = np.full((config.num_results, 4), 0.70)
        return SimpleNamespace(
            samples=samples,
            trace={
                "log_accept_ratio": np.log(probability),
                "is_accepted": np.ones_like(probability, dtype=bool),
                "target_log_prob": np.zeros_like(probability),
            },
            diagnostics={
                "native_divergence_status": "not_exposed_by_kernel",
                "divergence_count": None,
            },
        )

    selection = run_operational_fixed_trajectory_selection(
        adapter=_FiniteTargetAdapter(),
        private_start_bank=bank,
        private_start_bank_signature=_bank_signature(bank),
        coordinate_signature="coordinate",
        metric_signature="metric",
        anchor_l=10,
        max_leapfrog_steps=64,
        frozen_step_size=0.1,
        root_seed=(20260711, 700),
        target_scope="test",
        acceptance_policy=HMCAcceptancePolicy(),
        screen_num_results=64,
        run_full_chain=runner,
    )

    assert selection.disposition == "inconclusive_resonance"
    assert selection.representative is None
    assert len(calls) == 9
    assert all(not config.tuning_policy.uses_dual_averaging for config in calls)
    assert all(
        replication.path_return_fraction == pytest.approx(1.0)
        and candidate.resonance_detected
        for candidate in selection.candidate_results
        for replication in candidate.replications
    )


def test_operational_candidate_does_not_truncate_malformed_divergence_count() -> None:
    bank = np.arange(8, dtype=float).reshape(4, 2)

    def runner(_adapter, _initial_state, config):
        draws = np.arange(config.num_results, dtype=float)[:, None, None]
        probabilities = np.full((config.num_results, 4), 0.70)
        return SimpleNamespace(
            samples=draws + bank[None, :, :],
            trace={
                "log_accept_ratio": np.log(probabilities),
                "is_accepted": np.ones_like(probabilities, dtype=bool),
                "target_log_prob": np.zeros_like(probabilities),
            },
            diagnostics={
                "native_divergence_status": "available",
                "divergence_count": 1.5,
            },
        )

    selection = run_operational_fixed_trajectory_selection(
        adapter=_FiniteTargetAdapter(),
        private_start_bank=bank,
        private_start_bank_signature=_bank_signature(bank),
        coordinate_signature="coordinate",
        metric_signature="metric",
        anchor_l=10,
        max_leapfrog_steps=64,
        frozen_step_size=0.1,
        root_seed=(20260711, 700),
        target_scope="test",
        acceptance_policy=HMCAcceptancePolicy(),
        run_full_chain=runner,
    )

    assert selection.disposition == "candidate_set_exhausted"
    assert all(
        replication.evidence.evidence_validity == "candidate_data_invalid"
        for candidate in selection.candidate_results
        for replication in candidate.replications
    )
    assert all(
        replication.evidence.engineering_invalidity_reasons
        == ("native_divergence_provenance_inconsistent",)
        for candidate in selection.candidate_results
        for replication in candidate.replications
    )


@pytest.mark.parametrize(
    ("target_status_mode", "expected_disposition", "expected_validity", "reason"),
    (
        (
            "valid",
            "representative_selected",
            "valid",
            None,
        ),
        (
            "missing",
            "shared_invalidity",
            "shared_execution_invalid",
            "required_target_status_telemetry_missing",
        ),
        (
            "malformed",
            "shared_invalidity",
            "shared_execution_invalid",
            "shared_schema_invalid",
        ),
        (
            "nonvalid",
            "candidate_set_exhausted",
            "candidate_data_invalid",
            "target_status_telemetry_failure",
        ),
    ),
)
def test_operational_selection_enforces_requested_target_status(
    target_status_mode: str,
    expected_disposition: str,
    expected_validity: str,
    reason: str | None,
) -> None:
    bank = np.arange(8, dtype=float).reshape(4, 2)

    def runner(_adapter, initial_state, config):
        if config.tuning_policy.uses_dual_averaging:
            trace = {
                "log_accept_ratio": np.zeros((4, 4)),
                "is_accepted": np.ones((4, 4), dtype=bool),
                "target_log_prob": np.zeros((4, 4)),
                "step_size": np.full(4, 0.125),
                "target_status_telemetry": _target_status_trace((4, 4)),
            }
            return SimpleNamespace(
                samples=np.zeros((4, 4, 2)),
                trace=trace,
                diagnostics={"final_step_size": 0.125},
            )
        draws = np.arange(config.num_results, dtype=float)[:, None, None]
        probability = np.full((config.num_results, 4), 0.70)
        trace = {
            "log_accept_ratio": np.log(probability),
            "is_accepted": np.ones_like(probability, dtype=bool),
            "target_log_prob": np.zeros_like(probability),
        }
        if target_status_mode != "missing":
            trace["target_status_telemetry"] = _target_status_trace(
                probability.shape,
                nonvalid=target_status_mode == "nonvalid",
                malformed=target_status_mode == "malformed",
            )
        return SimpleNamespace(
            samples=draws + np.asarray(initial_state)[None, :, :],
            trace=trace,
            diagnostics={},
        )

    selection = run_operational_fixed_trajectory_selection(
        adapter=_FiniteTargetAdapter(),
        private_start_bank=bank,
        private_start_bank_signature=_bank_signature(bank),
        coordinate_signature="coordinate",
        metric_signature="metric",
        anchor_l=10,
        max_leapfrog_steps=64,
        frozen_step_size=0.1,
        root_seed=(20260711, 701),
        target_scope="test",
        acceptance_policy=HMCAcceptancePolicy(),
        target_status_trace_policy="per_chain_step",
        run_full_chain=runner,
    )

    assert selection.disposition == expected_disposition
    assert all(
        replication.evidence.evidence_validity == expected_validity
        for candidate in selection.candidate_results
        for replication in candidate.replications
    )
    if reason is not None:
        assert all(
            reason in replication.evidence.engineering_invalidity_reasons
            for candidate in selection.candidate_results
            for replication in candidate.replications
        )


def test_exact_l_retune_rejects_requested_target_status_veto() -> None:
    bank = np.arange(8, dtype=float).reshape(4, 2)

    def runner(_adapter, initial_state, config):
        if config.tuning_policy.uses_dual_averaging:
            return SimpleNamespace(
                samples=np.zeros((4, 4, 2)),
                trace={
                    "log_accept_ratio": np.zeros((4, 4)),
                    "is_accepted": np.ones((4, 4), dtype=bool),
                    "target_log_prob": np.zeros((4, 4)),
                    "step_size": np.full(4, 0.125),
                    "target_status_telemetry": _target_status_trace(
                        (4, 4), nonvalid=True
                    ),
                },
                diagnostics={"final_step_size": 0.125},
            )
        draws = np.arange(config.num_results, dtype=float)[:, None, None]
        probability = np.full((config.num_results, 4), 0.70)
        return SimpleNamespace(
            samples=draws + np.asarray(initial_state)[None, :, :],
            trace={
                "log_accept_ratio": np.log(probability),
                "is_accepted": np.ones_like(probability, dtype=bool),
                "target_log_prob": np.zeros_like(probability),
                "target_status_telemetry": _target_status_trace(probability.shape),
            },
            diagnostics={},
        )

    selection = run_operational_fixed_trajectory_selection(
            adapter=_FiniteTargetAdapter(),
            private_start_bank=bank,
            private_start_bank_signature=_bank_signature(bank),
            coordinate_signature="coordinate",
            metric_signature="metric",
            anchor_l=10,
            max_leapfrog_steps=64,
            frozen_step_size=0.1,
            root_seed=(20260711, 702),
            target_scope="test",
            acceptance_policy=HMCAcceptancePolicy(),
            target_status_trace_policy="per_chain_step",
            run_full_chain=runner,
        )

    assert selection.disposition == "candidate_retune_failed"
    assert selection.representative is None
    assert selection.retune_failure_scope == "candidate_data_invalid"
    assert selection.retune_failure_reasons == ("target_status_telemetry_failure",)
    assert tuple(
        next(
            item.candidate.num_leapfrog_steps
            for item in selection.candidate_results
            if item.candidate.signature == failure.candidate_signature
        )
        for failure in selection.candidate_retune_failures
    ) == (10, 5, 20)
    assert tuple(
        failure.nomination_ordinal
        for failure in selection.candidate_retune_failures
    ) == (0, 1, 2)
    assert len({failure.seed for failure in selection.candidate_retune_failures}) == 3


def test_exact_l_retune_falls_back_after_candidate_local_failure() -> None:
    bank = np.arange(8, dtype=float).reshape(4, 2)
    tune_calls = []

    def runner(_adapter, initial_state, config):
        if config.tuning_policy.uses_dual_averaging:
            tune_calls.append(config)
            samples = np.zeros((4, 4, 2))
            if config.num_leapfrog_steps == 10:
                samples[0, 0, 0] = np.nan
            return SimpleNamespace(
                samples=samples,
                trace={
                    "log_accept_ratio": np.zeros((4, 4)),
                    "is_accepted": np.ones((4, 4), dtype=bool),
                    "target_log_prob": np.zeros((4, 4)),
                    "step_size": np.full(4, 0.125),
                },
                diagnostics={"final_step_size": 0.125},
            )
        draws = np.arange(config.num_results, dtype=float)[:, None, None]
        probability = np.full((config.num_results, 4), 0.70)
        return SimpleNamespace(
            samples=draws + np.asarray(initial_state)[None, :, :],
            trace={
                "log_accept_ratio": np.log(probability),
                "is_accepted": np.ones_like(probability, dtype=bool),
                "target_log_prob": np.zeros_like(probability),
            },
            diagnostics={},
        )

    root_seed = (20260713, 1)
    selection = run_operational_fixed_trajectory_selection(
        adapter=_FiniteTargetAdapter(),
        private_start_bank=bank,
        private_start_bank_signature=_bank_signature(bank),
        coordinate_signature="coordinate",
        metric_signature="metric",
        anchor_l=10,
        max_leapfrog_steps=64,
        frozen_step_size=0.1,
        root_seed=root_seed,
        target_scope="test",
        acceptance_policy=HMCAcceptancePolicy(),
        run_full_chain=runner,
    )

    assert [config.num_leapfrog_steps for config in tune_calls] == [10, 5]
    assert selection.disposition == "representative_selected"
    assert selection.representative.candidate.num_leapfrog_steps == 5
    assert selection.retune_failure_scope is None
    assert len(selection.candidate_retune_failures) == 1
    failure = selection.candidate_retune_failures[0]
    candidate_l10 = next(
        item.candidate
        for item in selection.candidate_results
        if item.candidate.num_leapfrog_steps == 10
    )
    assert failure.candidate_signature == candidate_l10.signature
    assert failure.nomination_ordinal == 0
    assert failure.reasons == ("nonfinite_candidate_state",)
    assert failure.seed == paired_candidate_seed(
        root_seed,
        candidate_signature=candidate_l10.signature,
        replication_index=0,
        domain="exact_final_l_epsilon_tune",
    )


def test_exact_l_retune_shared_invalidity_stops_fallback_immediately() -> None:
    bank = np.arange(8, dtype=float).reshape(4, 2)
    tune_calls = []

    def runner(_adapter, initial_state, config):
        if config.tuning_policy.uses_dual_averaging:
            tune_calls.append(config)
            samples = np.zeros((4, 4, 2))
            if config.num_leapfrog_steps == 10:
                samples[0, 0, 0] = np.nan
            elif config.num_leapfrog_steps == 5:
                samples = samples[:-1]
            return SimpleNamespace(
                samples=samples,
                trace={
                    "log_accept_ratio": np.zeros((4, 4)),
                    "is_accepted": np.ones((4, 4), dtype=bool),
                    "target_log_prob": np.zeros((4, 4)),
                    "step_size": np.full(4, 0.125),
                },
                diagnostics={"final_step_size": 0.125},
            )
        draws = np.arange(config.num_results, dtype=float)[:, None, None]
        probability = np.full((config.num_results, 4), 0.70)
        return SimpleNamespace(
            samples=draws + np.asarray(initial_state)[None, :, :],
            trace={
                "log_accept_ratio": np.log(probability),
                "is_accepted": np.ones_like(probability, dtype=bool),
                "target_log_prob": np.zeros_like(probability),
            },
            diagnostics={},
        )

    selection = run_operational_fixed_trajectory_selection(
        adapter=_FiniteTargetAdapter(),
        private_start_bank=bank,
        private_start_bank_signature=_bank_signature(bank),
        coordinate_signature="coordinate",
        metric_signature="metric",
        anchor_l=10,
        max_leapfrog_steps=64,
        frozen_step_size=0.1,
        root_seed=(20260713, 2),
        target_scope="test",
        acceptance_policy=HMCAcceptancePolicy(),
        run_full_chain=runner,
    )

    assert [config.num_leapfrog_steps for config in tune_calls] == [10, 5]
    assert selection.disposition == "shared_invalidity"
    assert selection.representative is None
    assert len(selection.candidate_retune_failures) == 1
    assert selection.retune_failure_scope == "shared_execution_invalid"
    assert selection.retune_failure_reasons == ("shared_schema_invalid",)
    candidate_l5 = next(
        item.candidate
        for item in selection.candidate_results
        if item.candidate.num_leapfrog_steps == 5
    )
    assert selection.retune_candidate_signature == candidate_l5.signature


def test_evidence_extension_preserves_matrix_when_exact_l_runner_raises() -> None:
    """An exact-L exception is a typed shared veto, not a lost extension ledger."""

    bank = np.arange(8, dtype=float).reshape(4, 2)
    policy = HMCAcceptancePolicy()

    def runner(_adapter, initial_state, config):
        if config.tuning_policy.uses_dual_averaging:
            raise RuntimeError("synthetic exact-L runner failure")
        draws = np.arange(config.num_results, dtype=float)[:, None, None]
        probability = np.full((config.num_results, 4), 0.60)
        if config.num_results > 64:
            probability.fill(0.70)
        return SimpleNamespace(
            samples=draws + np.asarray(initial_state)[None, :, :],
            trace={
                "log_accept_ratio": np.log(probability),
                "is_accepted": np.ones_like(probability, dtype=bool),
                "target_log_prob": np.zeros_like(probability),
            },
            diagnostics={},
        )

    common = {
        "coordinate_signature": "coordinate",
        "metric_signature": "metric",
        "private_start_bank_signature": _bank_signature(bank),
        "root_seed": (20260813, 901),
        "frozen_step_size": 0.1,
        "screen_num_results": 64,
        "screen_num_burnin_steps": 16,
        "final_tune_adaptation_steps": 64,
        "target_scope": "test",
        "acceptance_policy": policy,
        "chain_execution_mode": "tf_function",
        "use_xla": False,
    }
    initial = _matrix_selection(
        common,
        {5: _inconclusive_evidence(), 10: _inconclusive_evidence(), 20: _inconclusive_evidence()},
    )
    assert initial.disposition == "inconclusive_evidence"

    finalized, extension = extend_operational_fixed_trajectory_evidence(
        selection=initial,
        adapter=_FiniteTargetAdapter(),
        private_start_bank=bank,
        private_start_bank_signature=common["private_start_bank_signature"],
        coordinate_signature=common["coordinate_signature"],
        metric_signature=common["metric_signature"],
        frozen_step_size=common["frozen_step_size"],
        root_seed=common["root_seed"],
        target_scope=common["target_scope"],
        acceptance_policy=policy,
        checkpoint=128,
        extension_round_index=0,
        screen_num_burnin_steps=common["screen_num_burnin_steps"],
        final_tune_adaptation_steps=common["final_tune_adaptation_steps"],
        run_full_chain=runner,
    )

    assert extension.matrix_disposition == "representative_nominated"
    assert extension.finalized_disposition == "shared_invalidity"
    assert len(extension.slots) == 9
    assert finalized.disposition == "shared_invalidity"
    assert finalized.retune_failure_scope == "shared_execution_invalid"
    assert finalized.retune_failure_reasons == ("retune_runtime_error",)
    assert finalized.retune_candidate_signature is not None


def test_exact_l_retune_fallback_is_invariant_to_candidate_permutation() -> None:
    bank = np.arange(8, dtype=float).reshape(4, 2)
    candidate_results = tuple(_result(leapfrog, retuned=False) for leapfrog in (5, 10, 20))
    outcomes = []
    call_orders = []

    for ordered_results in (
        candidate_results,
        tuple(reversed(candidate_results)),
        (candidate_results[2], candidate_results[0], candidate_results[1]),
    ):
        calls = []

        def runner(_adapter, _initial_state, config):
            calls.append(config.num_leapfrog_steps)
            samples = np.zeros((4, 4, 2))
            if config.num_leapfrog_steps == 10:
                samples[0, 0, 0] = np.nan
            return SimpleNamespace(
                samples=samples,
                trace={
                    "log_accept_ratio": np.zeros((4, 4)),
                    "is_accepted": np.ones((4, 4), dtype=bool),
                    "target_log_prob": np.zeros((4, 4)),
                    "step_size": np.full(4, 0.125),
                },
                diagnostics={"final_step_size": 0.125},
            )

        nomination = select_fixed_trajectory_representative(
            ordered_results,
            anchor_l=10,
        )
        outcomes.append(
            _finalize_operational_selection_nomination(
                selection=nomination,
                adapter=_FiniteTargetAdapter(),
                bank=bank,
                bank_signature="bank",
                coordinate_signature="coordinate",
                metric_signature="metric",
                frozen_step_size=0.1,
                root_seed=(20260713, 3),
                target_scope="test",
                acceptance_policy=HMCAcceptancePolicy(),
                final_tune_adaptation_steps=64,
                chain_execution_mode="tf_function",
                use_xla=False,
                target_status_trace_policy="none",
                runner=runner,
            )
        )
        call_orders.append(tuple(calls))

    assert call_orders == [(10, 5)] * 3
    assert {outcome.signature for outcome in outcomes} == {outcomes[0].signature}
    assert all(
        outcome.representative.candidate.num_leapfrog_steps == 5
        for outcome in outcomes
    )


def test_selection_rejects_nonprefix_retune_failure_history() -> None:
    results = tuple(_result(leapfrog, retuned=False) for leapfrog in (5, 10, 20))
    candidate_l5 = next(
        item.candidate
        for item in results
        if item.candidate.num_leapfrog_steps == 5
    )
    failure = FixedTrajectoryCandidateRetuneFailure(
        candidate_signature=candidate_l5.signature,
        seed=(20260713, 10),
        reasons=("nonfinite_candidate_state",),
        nomination_ordinal=0,
    )

    with pytest.raises(ValueError, match="deterministic policy prefix"):
        FixedTrajectorySelection(
            anchor_l=10,
            candidate_results=results,
            representative_signature=None,
            disposition="candidate_retune_failed",
            candidate_retune_failures=(failure,),
            retune_failure_scope="candidate_data_invalid",
            retune_failure_reasons=failure.reasons,
            retune_candidate_signature=failure.candidate_signature,
        )


def test_selection_shared_screen_invalidity_precedes_retune_history() -> None:
    viable = _result(10, retuned=False)
    shared = _result_from_evidence(5, _shared_invalidity_evidence())
    failure = FixedTrajectoryCandidateRetuneFailure(
        candidate_signature=viable.candidate.signature,
        seed=(20260713, 11),
        reasons=("nonfinite_candidate_state",),
        nomination_ordinal=0,
    )

    with pytest.raises(ValueError, match="retune failure provenance is invalid"):
        FixedTrajectorySelection(
            anchor_l=10,
            candidate_results=(viable, shared),
            representative_signature=None,
            disposition="candidate_retune_failed",
            retune_failure_scope="candidate_data_invalid",
            retune_failure_reasons=failure.reasons,
            retune_candidate_signature=failure.candidate_signature,
            candidate_retune_failures=(failure,),
        )


def test_selection_attempt_rejects_fabricated_retune_failure_seed() -> None:
    root_seed = (20260711, 700)
    candidate_l10 = _result(10, retuned=False)
    candidate_l5 = _result(5, retuned=True)
    candidate_l20 = _result(20, retuned=False)
    failure = FixedTrajectoryCandidateRetuneFailure(
        candidate_signature=candidate_l10.candidate.signature,
        seed=(1234, 5678),
        reasons=("nonfinite_candidate_state",),
        nomination_ordinal=0,
    )
    selection = FixedTrajectorySelection(
        anchor_l=10,
        candidate_results=(candidate_l5, candidate_l10, candidate_l20),
        representative_signature=candidate_l5.signature,
        disposition="representative_selected",
        candidate_retune_failures=(failure,),
    )

    with pytest.raises(ValueError, match="failure seed lost its lineage"):
        FixedTrajectorySelectionRepairAttempt(
            attempt_index=0,
            root_seed=root_seed,
            input_step_size=0.1,
            selection=selection,
            bracket_before=(None, None),
        )


def test_bounded_selection_audits_every_fallback_retune_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bank = np.arange(8, dtype=float).reshape(4, 2)
    original_seed_fn = paired_candidate_seed

    def colliding_seed(root_seed, *, candidate_signature, replication_index, domain):
        if domain == "exact_final_l_epsilon_tune":
            return (987654321, 123456789)
        return original_seed_fn(
            root_seed,
            candidate_signature=candidate_signature,
            replication_index=replication_index,
            domain=domain,
        )

    def runner(_adapter, initial_state, config):
        if config.tuning_policy.uses_dual_averaging:
            samples = np.zeros((4, 4, 2))
            if config.num_leapfrog_steps == 10:
                samples[0, 0, 0] = np.nan
            return SimpleNamespace(
                samples=samples,
                trace={
                    "log_accept_ratio": np.zeros((4, 4)),
                    "is_accepted": np.ones((4, 4), dtype=bool),
                    "target_log_prob": np.zeros((4, 4)),
                    "step_size": np.full(4, 0.125),
                },
                diagnostics={"final_step_size": 0.125},
            )
        draws = np.arange(config.num_results, dtype=float)[:, None, None]
        probability = np.full((config.num_results, 4), 0.70)
        return SimpleNamespace(
            samples=draws + np.asarray(initial_state)[None, :, :],
            trace={
                "log_accept_ratio": np.log(probability),
                "is_accepted": np.ones_like(probability, dtype=bool),
                "target_log_prob": np.zeros_like(probability),
            },
            diagnostics={},
        )

    monkeypatch.setattr(
        "bayesfilter.inference.hmc_kernel_selection.paired_candidate_seed",
        colliding_seed,
    )
    with pytest.raises(ValueError, match="reused an execution seed"):
        run_bounded_operational_fixed_trajectory_selection(
            adapter=_FiniteTargetAdapter(),
            private_start_bank=bank,
            private_start_bank_signature=_bank_signature(bank),
            coordinate_signature="coordinate",
            metric_signature="metric",
            anchor_l=10,
            max_leapfrog_steps=64,
            initial_step_size=0.1,
            root_seed=(20260713, 4),
            target_scope="test",
            acceptance_policy=HMCAcceptancePolicy(),
            max_attempts=1,
            run_full_chain=runner,
        )


@pytest.mark.parametrize(
    ("malformation", "expected_disposition", "expected_validity", "expected_reason"),
    (
        (
            "sample_shape",
            "shared_invalidity",
            "shared_execution_invalid",
            "shared_schema_invalid",
        ),
        (
            "trace_shape",
            "shared_invalidity",
            "shared_execution_invalid",
            "shared_schema_invalid",
        ),
        (
            "acceptance_dtype",
            "shared_invalidity",
            "shared_execution_invalid",
            "shared_schema_invalid",
        ),
        (
            "nonfinite_samples",
            "candidate_set_exhausted",
            "candidate_data_invalid",
            "nonfinite_candidate_state",
        ),
    ),
)
def test_operational_candidate_runner_boundary_fails_closed(
    malformation: str,
    expected_disposition: str,
    expected_validity: str,
    expected_reason: str,
) -> None:
    bank = np.arange(8, dtype=float).reshape(4, 2)

    def runner(_adapter, _initial_state, config):
        draws = np.arange(config.num_results, dtype=float)[:, None, None]
        samples = draws + bank[None, :, :]
        probability = np.full((config.num_results, 4), 0.70)
        accepted = np.ones_like(probability, dtype=bool)
        if malformation == "sample_shape":
            samples = samples[:-1]
        elif malformation == "trace_shape":
            probability = probability[:-1]
            accepted = accepted[:-1]
        elif malformation == "acceptance_dtype":
            accepted = accepted.astype(float)
        elif malformation == "nonfinite_samples":
            samples[0, 0, 0] = np.nan
        return SimpleNamespace(
            samples=samples,
            trace={
                "log_accept_ratio": np.log(probability),
                "is_accepted": accepted,
                "target_log_prob": np.zeros_like(probability),
            },
            diagnostics={
                "native_divergence_status": "not_exposed_by_kernel",
                "divergence_count": None,
            },
        )

    selection = run_operational_fixed_trajectory_selection(
        adapter=_FiniteTargetAdapter(),
        private_start_bank=bank,
        private_start_bank_signature=_bank_signature(bank),
        coordinate_signature="coordinate",
        metric_signature="metric",
        anchor_l=10,
        max_leapfrog_steps=64,
        frozen_step_size=0.1,
        root_seed=(20260711, 700),
        target_scope="test",
        acceptance_policy=HMCAcceptancePolicy(),
        run_full_chain=runner,
    )

    assert selection.disposition == expected_disposition
    assert all(
        replication.evidence.evidence_validity == expected_validity
        and replication.evidence.engineering_invalidity_reasons
        == (expected_reason,)
        and replication.mean_esjd_per_gradient is None
        for candidate in selection.candidate_results
        for replication in candidate.replications
    )


def test_nonfinite_candidate_does_not_block_independent_peer_selection() -> None:
    bank = np.arange(8, dtype=float).reshape(4, 2)
    calls = []

    class _RejectNonfiniteTargetAdapter(_FiniteTargetAdapter):
        def log_prob_and_grad(self, theta):
            values = np.asarray(theta, dtype=float)
            if not np.all(np.isfinite(values)):
                raise AssertionError("invalid candidate state must not reach target")
            return super().log_prob_and_grad(values)

    def runner(_adapter, initial_state, config):
        calls.append(config)
        if config.tuning_policy.uses_dual_averaging:
            return SimpleNamespace(
                samples=np.zeros((4, 4, 2)),
                trace={
                    "log_accept_ratio": np.zeros((4, 4)),
                    "is_accepted": np.ones((4, 4), dtype=bool),
                    "target_log_prob": np.zeros((4, 4)),
                    "step_size": np.full(4, 0.125),
                },
                diagnostics={"final_step_size": 0.125},
            )
        draws = np.arange(config.num_results, dtype=float)[:, None, None]
        samples = draws + np.asarray(initial_state)[None, :, :]
        if config.num_leapfrog_steps == 5:
            samples[0, 0, 0] = np.nan
        probability = np.full((config.num_results, 4), 0.70)
        return SimpleNamespace(
            samples=samples,
            trace={
                "log_accept_ratio": np.log(probability),
                "is_accepted": np.ones_like(probability, dtype=bool),
                "target_log_prob": np.zeros_like(probability),
            },
            diagnostics={},
        )

    selection = run_operational_fixed_trajectory_selection(
        adapter=_RejectNonfiniteTargetAdapter(),
        private_start_bank=bank,
        private_start_bank_signature=_bank_signature(bank),
        coordinate_signature="coordinate",
        metric_signature="metric",
        anchor_l=10,
        max_leapfrog_steps=64,
        frozen_step_size=0.1,
        root_seed=(20260712, 703),
        target_scope="test",
        acceptance_policy=HMCAcceptancePolicy(),
        run_full_chain=runner,
    )

    failed = next(
        item for item in selection.candidate_results
        if item.candidate.num_leapfrog_steps == 5
    )
    assert all(
        replication.evidence.evidence_validity == "candidate_data_invalid"
        and replication.evidence.engineering_invalidity_reasons
        == ("nonfinite_candidate_state",)
        for replication in failed.replications
    )
    assert selection.disposition == "representative_selected"
    assert selection.representative.candidate.num_leapfrog_steps == 10
    assert selection.representative.exact_l_retuned_step_size == pytest.approx(0.125)
    assert len(calls) == 10


def test_operational_esjd_and_gradient_cost_match_tfp_hmc_accounting() -> None:
    bank = np.arange(8, dtype=float).reshape(4, 2)

    def runner(_adapter, initial_state, config):
        if config.tuning_policy.uses_dual_averaging:
            return SimpleNamespace(
                samples=np.zeros((4, 4, 2)),
                trace={
                    "log_accept_ratio": np.zeros((4, 4)),
                    "is_accepted": np.ones((4, 4), dtype=bool),
                    "target_log_prob": np.zeros((4, 4)),
                    "step_size": np.full(4, 0.125),
                },
                diagnostics={"final_step_size": 0.125},
            )
        draws = np.arange(config.num_results, dtype=float)[:, None, None]
        samples = draws + np.asarray(initial_state)[None, :, :]
        probability = np.full((config.num_results, 4), 0.70)
        return SimpleNamespace(
            samples=samples,
            trace={
                "log_accept_ratio": np.log(probability),
                "is_accepted": np.ones_like(probability, dtype=bool),
                "target_log_prob": np.zeros_like(probability),
            },
            diagnostics={},
        )

    selection = run_operational_fixed_trajectory_selection(
        adapter=_FiniteTargetAdapter(),
        private_start_bank=bank,
        private_start_bank_signature=_bank_signature(bank),
        coordinate_signature="coordinate",
        metric_signature="metric",
        anchor_l=10,
        max_leapfrog_steps=64,
        frozen_step_size=0.1,
        root_seed=(20260711, 700),
        target_scope="test",
        acceptance_policy=HMCAcceptancePolicy(),
        screen_num_results=64,
        screen_num_burnin_steps=16,
        run_full_chain=runner,
    )

    replication = next(
        item.replications[0]
        for item in selection.candidate_results
        if item.candidate.num_leapfrog_steps == 10
    )
    assert replication.mean_esjd_per_gradient == pytest.approx(2.0 / 10.0)
    assert replication.cost_gradient_evaluations == 4 * (1 + 80 * 10)


@pytest.mark.parametrize(
    ("malformation", "expected_disposition", "expected_scope", "expected_reason"),
    (
        (
            "missing_step_trace",
            "shared_invalidity",
            "shared_execution_invalid",
            "shared_schema_invalid",
        ),
        (
            "nonfinite_samples",
            "candidate_retune_failed",
            "candidate_data_invalid",
            "nonfinite_candidate_state",
        ),
        (
            "step_mismatch",
            "shared_invalidity",
            "shared_execution_invalid",
            "shared_schema_invalid",
        ),
        (
            "nonfinite_target_value",
            "candidate_retune_failed",
            "candidate_data_invalid",
            "nonfinite_target_log_prob",
        ),
        (
            "nonfinite_target_score",
            "candidate_retune_failed",
            "candidate_data_invalid",
            "nonfinite_target_score",
        ),
        (
            "malformed_target_value_shape",
            "shared_invalidity",
            "shared_execution_invalid",
            "target_value_score_shape_invalid",
        ),
        (
            "malformed_target_score_shape",
            "shared_invalidity",
            "shared_execution_invalid",
            "target_value_score_shape_invalid",
        ),
        (
            "target_callback_exception",
            "shared_invalidity",
            "shared_execution_invalid",
            "shared_callback_invalid",
        ),
    ),
)
def test_exact_l_retune_health_is_required_for_selection(
    malformation: str,
    expected_disposition: str,
    expected_scope: str,
    expected_reason: str,
) -> None:
    bank = np.arange(8, dtype=float).reshape(4, 2)

    class _RetuneHealthAdapter(_FiniteTargetAdapter):
        def log_prob_and_grad(self, theta):
            values = np.asarray(theta, dtype=float)
            if malformation == "target_callback_exception" and np.all(
                values == 1000.0
            ):
                raise RuntimeError("unexpected target callback failure")
            target_value, target_score = super().log_prob_and_grad(values)
            if not np.all(values == 1000.0):
                return target_value, target_score
            if malformation == "nonfinite_target_value":
                target_value = np.full_like(target_value, np.nan)
            elif malformation == "nonfinite_target_score":
                target_score = np.full_like(target_score, np.nan)
            elif malformation == "malformed_target_value_shape":
                target_value = target_value[:-1]
            elif malformation == "malformed_target_score_shape":
                target_score = target_score[:, :-1]
            return target_value, target_score

    def runner(_adapter, initial_state, config):
        if config.tuning_policy.uses_dual_averaging:
            samples = np.full((4, 4, 2), 1000.0)
            step_trace = np.full(4, 0.125)
            trace = {
                "log_accept_ratio": np.zeros((4, 4)),
                "is_accepted": np.ones((4, 4), dtype=bool),
                "target_log_prob": np.zeros((4, 4)),
                "step_size": step_trace,
            }
            final_step = 0.125
            if malformation == "missing_step_trace":
                trace.pop("step_size")
            elif malformation == "nonfinite_samples":
                samples[0, 0, 0] = np.nan
            elif malformation == "step_mismatch":
                final_step = 0.25
            return SimpleNamespace(
                samples=samples,
                trace=trace,
                diagnostics={"final_step_size": final_step},
            )
        draws = np.arange(config.num_results, dtype=float)[:, None, None]
        samples = draws + np.asarray(initial_state)[None, :, :]
        probability = np.full((config.num_results, 4), 0.70)
        return SimpleNamespace(
            samples=samples,
            trace={
                "log_accept_ratio": np.log(probability),
                "is_accepted": np.ones_like(probability, dtype=bool),
                "target_log_prob": np.zeros_like(probability),
            },
            diagnostics={},
        )

    selection = run_operational_fixed_trajectory_selection(
        adapter=_RetuneHealthAdapter(),
            private_start_bank=bank,
            private_start_bank_signature=_bank_signature(bank),
            coordinate_signature="coordinate",
            metric_signature="metric",
            anchor_l=10,
            max_leapfrog_steps=64,
            frozen_step_size=0.1,
            root_seed=(20260711, 700),
            target_scope="test",
            acceptance_policy=HMCAcceptancePolicy(),
        run_full_chain=runner,
    )

    assert selection.disposition == expected_disposition
    assert selection.representative is None
    assert selection.retune_failure_scope == expected_scope
    assert selection.retune_failure_reasons == (expected_reason,)


def _attempt5_like_selection(**kwargs):
    passed = _evidence(0.70, draw_count=256)
    higher = _evidence(0.90, draw_count=256)
    inconclusive = _inconclusive_evidence(draw_count=256)
    return _matrix_selection(
        kwargs,
        {
            5: higher,
            10: (passed, inconclusive, inconclusive),
            20: inconclusive,
        },
    )


def _phase1_mixed_selection(**kwargs):
    higher = _evidence(0.90, draw_count=256)
    inconclusive = _inconclusive_evidence(draw_count=256)
    return _matrix_selection(
        kwargs,
        {5: higher, 10: inconclusive, 20: higher},
    )


def _all_mapping_keys(value):
    if isinstance(value, dict):
        keys = set(value)
        for item in value.values():
            keys.update(_all_mapping_keys(item))
        return keys
    if isinstance(value, (tuple, list)):
        keys = set()
        for item in value:
            keys.update(_all_mapping_keys(item))
        return keys
    return set()


def test_default_disabled_extension_is_not_called_and_exhausts_truthfully() -> None:
    runner, runner_calls = _fake_extension_runner(probability_by_l={10: 0.70, 20: 0.40})

    def forbidden_extender(**_kwargs):
        raise AssertionError("default-disabled evidence extension was called")

    kwargs = _bounded_extension_kwargs(
        selector=_attempt5_like_selection,
        runner=runner,
        evidence_extender=forbidden_extender,
    )
    kwargs["evidence_extension_checkpoints"] = ()
    result = run_bounded_operational_fixed_trajectory_selection(**kwargs)

    assert result.terminal_disposition == "budget_exhausted_valid"
    assert result.selection.disposition == "inconclusive_evidence"
    assert result.evidence_extension_checkpoints == ()
    assert result.payload()["evidence_extension_count"] == 0
    assert result.payload()["evidence_extension_checkpoints"] == ()
    assert runner_calls == []


def test_extension_reruns_exact_inconclusive_slots_then_retunes_exact_l() -> None:
    runner, calls = _fake_extension_runner(
        probability_by_l={10: 0.70, 20: 0.40}
    )
    result = run_bounded_operational_fixed_trajectory_selection(
        **_bounded_extension_kwargs(
            selector=_attempt5_like_selection,
            runner=runner,
        )
    )

    assert result.terminal_disposition == "representative_selected"
    assert result.evidence_extension_checkpoints == (512, 1024)
    assert len(result.attempts) == 1
    attempt = result.attempts[0]
    assert len(attempt.evidence_extensions) == 1
    extension = attempt.evidence_extensions[0]
    assert extension.checkpoint == 512
    assert extension.frozen_step_size == pytest.approx(0.1)
    assert extension.screen_num_burnin_steps == 16
    assert tuple(
        (slot.candidate.num_leapfrog_steps, slot.replication_index)
        for slot in extension.slots
    ) == ((10, 1), (10, 2), (20, 0), (20, 1), (20, 2))
    assert all(slot.prior_decision == "inconclusive_evidence" for slot in extension.slots)
    assert all(
        slot.extended_replication.seed != slot.prior_seed
        for slot in extension.slots
    )
    assert extension.matrix_disposition == "representative_nominated"
    assert extension.finalized_disposition == "representative_selected"

    source_by_l = {
        item.candidate.num_leapfrog_steps: item
        for item in extension.source_selection.candidate_results
    }
    final_by_l = {
        item.candidate.num_leapfrog_steps: item
        for item in result.selection.candidate_results
    }
    assert final_by_l[10].replications[0].signature == source_by_l[10].replications[0].signature
    assert tuple(item.signature for item in final_by_l[5].replications) == tuple(
        item.signature for item in source_by_l[5].replications
    )
    assert result.representative.candidate.num_leapfrog_steps == 10
    assert result.representative.exact_l_retuned_step_size == pytest.approx(0.125)

    extension_calls = [config for _bank, config in calls if not config.tuning_policy.uses_dual_averaging]
    tune_calls = [config for _bank, config in calls if config.tuning_policy.uses_dual_averaging]
    assert len(extension_calls) == 5
    assert all(config.num_results == 512 for config in extension_calls)
    assert all(config.num_burnin_steps == 16 for config in extension_calls)
    assert all(config.step_size == pytest.approx(0.1) for config in extension_calls)
    assert len(tune_calls) == 1
    assert tune_calls[0].num_results == 4
    assert tune_calls[0].num_burnin_steps == 64
    assert tune_calls[0].num_leapfrog_steps == 10
    all_execution_seeds = (
        attempt.replication_seeds
        + attempt.extension_seeds
        + (attempt.exact_l_retune_seed,)
    )
    assert len(set(all_execution_seeds)) == 15

    payload = extension.payload()
    assert payload["complete_matrix_barrier"] is True
    assert payload["fresh_traces_concatenated"] is False
    assert all(
        slot["fresh_trace_replaces_prior_evidence"] is True
        and slot["traces_concatenated"] is False
        for slot in payload["slots"]
    )
    assert not {"samples", "trace", "start_bank", "raw_start_bank"}.intersection(
        _all_mapping_keys(payload)
    )


def test_phase1_mixed_matrix_extends_candidate_before_scalar_repair() -> None:
    runner, calls = _fake_extension_runner(probability_by_l={10: 0.70})
    result = run_bounded_operational_fixed_trajectory_selection(
        **_bounded_extension_kwargs(
            selector=_phase1_mixed_selection,
            runner=runner,
        )
    )

    assert result.terminal_disposition == "representative_selected"
    assert len(result.attempts) == 1
    assert result.repair_direction_history == ()
    attempt = result.attempts[0]
    assert attempt.input_step_size == pytest.approx(0.1)
    assert attempt.repair is None
    assert len(attempt.evidence_extensions) == 1
    extension = attempt.evidence_extensions[0]
    assert extension.frozen_step_size == pytest.approx(0.1)
    assert tuple(
        (slot.candidate.num_leapfrog_steps, slot.replication_index)
        for slot in extension.slots
    ) == ((10, 0), (10, 1), (10, 2))
    source_by_l = {
        item.candidate.num_leapfrog_steps: item
        for item in extension.source_selection.candidate_results
    }
    final_by_l = {
        item.candidate.num_leapfrog_steps: item
        for item in result.selection.candidate_results
    }
    for leapfrog in (5, 20):
        assert tuple(rep.signature for rep in final_by_l[leapfrog].replications) == tuple(
            rep.signature for rep in source_by_l[leapfrog].replications
        )
    extension_calls = [
        config for _bank, config in calls if not config.tuning_policy.uses_dual_averaging
    ]
    tune_calls = [
        config for _bank, config in calls if config.tuning_policy.uses_dual_averaging
    ]
    assert len(extension_calls) == 3
    assert all(config.step_size == pytest.approx(0.1) for config in extension_calls)
    assert len(tune_calls) == 1
    assert tune_calls[0].num_leapfrog_steps == 10


def test_candidate_extension_eligibility_excludes_vetoed_inconclusive_peer() -> None:
    healthy = _inconclusive_evidence(draw_count=256)
    probabilities = np.repeat(
        np.repeat((0.60, 0.80, 0.60, 0.80), 64)[:, None],
        4,
        axis=1,
    )
    vetoed = evaluate_hmc_acceptance_evidence(
        samples=np.arange(256, dtype=float)[:, None, None]
        + np.arange(4, dtype=float)[None, :, None],
        log_accept_ratio=np.log(probabilities),
        is_accepted=np.ones_like(probabilities, dtype=bool),
        policy=HMCAcceptancePolicy(),
        native_divergence_status="available",
        native_divergence_count=1,
    )
    assert vetoed.acceptance_decision == "inconclusive_evidence"
    assert vetoed.candidate_promotion_vetoes == ("native_divergence_positive",)

    def selector(**kwargs):
        return _matrix_selection(
            kwargs,
            {
                5: _evidence(0.90, draw_count=256),
                10: healthy,
                20: vetoed,
            },
        )

    runner, calls = _fake_extension_runner(probability_by_l={10: 0.70})
    result = run_bounded_operational_fixed_trajectory_selection(
        **_bounded_extension_kwargs(selector=selector, runner=runner)
    )

    extension = result.attempts[0].evidence_extensions[0]
    assert result.terminal_disposition == "representative_selected"
    assert {slot.candidate.num_leapfrog_steps for slot in extension.slots} == {10}
    assert all(
        config.num_leapfrog_steps == 10
        for _bank, config in calls
    )


def test_candidate_extension_eligibility_preserves_cost_stopped_peer() -> None:
    policy = HMCAcceptancePolicy(
        allowed_cost_stop_reasons=("persistent_candidate_cost_stop",)
    )
    probabilities = np.repeat(
        np.repeat((0.60, 0.80, 0.60, 0.80), 64)[:, None],
        4,
        axis=1,
    )
    cost_stopped = evaluate_hmc_acceptance_evidence(
        samples=np.arange(256, dtype=float)[:, None, None]
        + np.arange(4, dtype=float)[None, :, None],
        log_accept_ratio=np.log(probabilities),
        is_accepted=np.ones_like(probabilities, dtype=bool),
        policy=policy,
        cost_stop_reasons=("persistent_candidate_cost_stop",),
    )
    assert cost_stopped.acceptance_decision == "inconclusive_evidence"
    assert cost_stopped.cost_stop_scope == "exact_candidate_replication"

    def selector(**kwargs):
        return _matrix_selection(
            kwargs,
            {
                5: _evidence(0.90, draw_count=256, policy=policy),
                10: _inconclusive_evidence(draw_count=256, policy=policy),
                20: cost_stopped,
            },
        )

    runner, calls = _fake_extension_runner(probability_by_l={10: 0.70})
    kwargs = _bounded_extension_kwargs(selector=selector, runner=runner)
    kwargs["acceptance_policy"] = policy
    result = run_bounded_operational_fixed_trajectory_selection(**kwargs)

    extension = result.attempts[0].evidence_extensions[0]
    assert result.terminal_disposition == "representative_selected"
    assert {slot.candidate.num_leapfrog_steps for slot in extension.slots} == {10}
    source_cost_stopped = next(
        item
        for item in extension.source_selection.candidate_results
        if item.candidate.num_leapfrog_steps == 20
    )
    final_cost_stopped = next(
        item
        for item in result.selection.candidate_results
        if item.candidate.num_leapfrog_steps == 20
    )
    assert source_cost_stopped.evidence_extension_eligible is False
    assert tuple(rep.signature for rep in final_cost_stopped.replications) == tuple(
        rep.signature for rep in source_cost_stopped.replications
    )
    assert all(config.num_leapfrog_steps == 10 for _bank, config in calls)


def test_phase1_extension_policy_is_permutation_invariant() -> None:
    bank = np.arange(8, dtype=float).reshape(4, 2)
    kwargs = _bounded_extension_kwargs(
        selector=_phase1_mixed_selection,
        runner=None,
    )
    selection_kwargs = dict(kwargs)
    selection_kwargs["frozen_step_size"] = selection_kwargs.pop(
        "initial_step_size"
    )
    selection_kwargs["chain_execution_mode"] = "tf_function"
    selection_kwargs["use_xla"] = False
    selection = _phase1_mixed_selection(**selection_kwargs)
    permuted = select_fixed_trajectory_representative(
        tuple(reversed(selection.candidate_results)),
        anchor_l=10,
    )
    runner_a, _ = _fake_extension_runner(probability_by_l={10: 0.70})
    runner_b, _ = _fake_extension_runner(probability_by_l={10: 0.70})
    common = {
        "adapter": _FiniteTargetAdapter(),
        "private_start_bank": bank,
        "private_start_bank_signature": _bank_signature(bank),
        "coordinate_signature": "coordinate",
        "metric_signature": "metric",
        "frozen_step_size": 0.1,
        "root_seed": (20260711, 700),
        "target_scope": "test",
        "acceptance_policy": HMCAcceptancePolicy(),
        "checkpoint": 512,
        "extension_round_index": 0,
        "screen_num_burnin_steps": 16,
        "final_tune_adaptation_steps": 64,
    }

    finalized_a, extension_a = extend_operational_fixed_trajectory_evidence(
        selection=selection,
        run_full_chain=runner_a,
        **common,
    )
    finalized_b, extension_b = extend_operational_fixed_trajectory_evidence(
        selection=permuted,
        run_full_chain=runner_b,
        **common,
    )

    assert finalized_a.signature == finalized_b.signature
    assert extension_a.payload() == extension_b.payload()


def test_extension_can_trigger_directional_repair_then_reserved_attempt() -> None:
    selector_calls = []

    def selector(**kwargs):
        selector_calls.append(kwargs)
        if len(selector_calls) == 1:
            inconclusive = _inconclusive_evidence(draw_count=256)
            return _matrix_selection(
                kwargs,
                {5: inconclusive, 10: inconclusive, 20: inconclusive},
            )
        passed = _evidence(0.70, draw_count=256)
        return _matrix_selection(
            kwargs,
            {5: passed, 10: passed, 20: passed},
            retuned_l=10,
        )

    runner, calls = _fake_extension_runner(
        probability_by_l={5: 0.90, 10: 0.90, 20: 0.90}
    )
    result = run_bounded_operational_fixed_trajectory_selection(
        **_bounded_extension_kwargs(selector=selector, runner=runner)
    )

    assert result.terminal_disposition == "representative_selected"
    assert len(result.attempts) == 2
    assert result.repair_direction_history == ("higher_epsilon",)
    assert result.repaired_step_history == pytest.approx((0.2,))
    assert result.attempts[0].selection.disposition == "repair_required"
    assert len(result.attempts[0].evidence_extensions) == 1
    assert result.attempts[1].input_step_size == pytest.approx(0.2)
    assert len(calls) == 9
    assert all(config.num_results == 512 for _bank, config in calls)


@pytest.mark.parametrize(
    ("failure_kind", "expected_terminal"),
    (
        ("shared", "shared_invalidity"),
        ("local", "candidate_set_exhausted"),
    ),
)
def test_extension_propagates_shared_and_candidate_local_vetoes(
    failure_kind,
    expected_terminal,
) -> None:
    inconclusive = _inconclusive_evidence(draw_count=256)

    def selector(**kwargs):
        return _matrix_selection(
            kwargs,
            {5: inconclusive, 10: inconclusive, 20: inconclusive},
        )

    bounded_kwargs = _bounded_extension_kwargs(selector=selector, runner=None)
    source_candidate = FixedTrajectoryCandidate(
        anchor_l=10,
        num_leapfrog_steps=5,
        max_leapfrog_steps=64,
        coordinate_signature=bounded_kwargs["coordinate_signature"],
        metric_signature=bounded_kwargs["metric_signature"],
        start_bank_signature=bounded_kwargs["private_start_bank_signature"],
    )
    shared_seed = paired_candidate_seed(
        (20260711, 700),
        candidate_signature=source_candidate.signature,
        replication_index=0,
        domain="selection_evidence_extension_0_512",
    )
    runner, calls = _fake_extension_runner(
        inconclusive=failure_kind == "shared",
        local_veto=failure_kind == "local",
        shared_invalidity_seed=shared_seed if failure_kind == "shared" else None,
    )
    bounded_kwargs["run_full_chain"] = runner
    result = run_bounded_operational_fixed_trajectory_selection(
        **bounded_kwargs
    )

    assert result.terminal_disposition == expected_terminal
    assert result.selection.disposition == expected_terminal
    assert len(calls) == 9
    assert len(result.attempts[0].evidence_extensions[0].slots) == 9


def test_two_checkpoint_extension_exhausts_with_fresh_disjoint_traces() -> None:
    inconclusive = _inconclusive_evidence(draw_count=256)

    def selector(**kwargs):
        return _matrix_selection(
            kwargs,
            {5: inconclusive, 10: inconclusive, 20: inconclusive},
        )

    runner, calls = _fake_extension_runner(inconclusive=True)
    result = run_bounded_operational_fixed_trajectory_selection(
        **_bounded_extension_kwargs(selector=selector, runner=runner)
    )

    assert result.terminal_disposition == "budget_exhausted_valid"
    assert result.selection.disposition == "inconclusive_evidence"
    attempt = result.attempts[0]
    assert tuple(item.checkpoint for item in attempt.evidence_extensions) == (512, 1024)
    assert attempt.evidence_extensions[0].finalized_selection_signature == (
        attempt.evidence_extensions[1].source_selection_signature
    )
    assert tuple(config.num_results for _bank, config in calls) == (512,) * 9 + (1024,) * 9
    assert all(config.num_burnin_steps == 16 for _bank, config in calls)
    assert len(set(attempt.all_replication_execution_seeds)) == 27
    assert len(attempt.replication_seeds) == 9
    assert len(attempt.extension_seeds) == 18
    assert result.payload()["evidence_extension_checkpoints"] == (512, 1024)
    assert result.payload()["evidence_extension_count"] == 2


def test_extension_is_permutation_invariant() -> None:
    bank = np.arange(8, dtype=float).reshape(4, 2)
    bank_signature = _bank_signature(bank)
    kwargs = {
        "coordinate_signature": "coordinate",
        "metric_signature": "metric",
        "private_start_bank_signature": bank_signature,
        "root_seed": (20260711, 700),
        "frozen_step_size": 0.1,
        "screen_num_results": 256,
        "screen_num_burnin_steps": 16,
        "target_scope": "test",
        "acceptance_policy": HMCAcceptancePolicy(),
        "chain_execution_mode": "tf_function",
        "use_xla": False,
    }
    original = _attempt5_like_selection(**kwargs)
    permuted = select_fixed_trajectory_representative(
        tuple(reversed(original.candidate_results)),
        anchor_l=10,
    )
    runner_a, _calls_a = _fake_extension_runner(
        probability_by_l={10: 0.70, 20: 0.40}
    )
    runner_b, _calls_b = _fake_extension_runner(
        probability_by_l={10: 0.70, 20: 0.40}
    )
    common = {
        "adapter": _FiniteTargetAdapter(),
        "private_start_bank": bank,
        "private_start_bank_signature": bank_signature,
        "coordinate_signature": "coordinate",
        "metric_signature": "metric",
        "frozen_step_size": 0.1,
        "root_seed": (20260711, 700),
        "target_scope": "test",
        "acceptance_policy": kwargs["acceptance_policy"],
        "checkpoint": 512,
        "extension_round_index": 0,
        "screen_num_burnin_steps": 16,
        "final_tune_adaptation_steps": 64,
    }
    finalized_a, extension_a = extend_operational_fixed_trajectory_evidence(
        selection=original,
        run_full_chain=runner_a,
        **common,
    )
    finalized_b, extension_b = extend_operational_fixed_trajectory_evidence(
        selection=permuted,
        run_full_chain=runner_b,
        **common,
    )

    assert original.signature == permuted.signature
    assert finalized_a.signature == finalized_b.signature
    assert extension_a.payload() == extension_b.payload()


def test_extension_rejects_slot_seed_and_declared_epsilon_corruption() -> None:
    runner, _calls = _fake_extension_runner(
        probability_by_l={10: 0.70, 20: 0.40}
    )
    captured = {}

    def corrupting_extender(**kwargs):
        finalized, extension = extend_operational_fixed_trajectory_evidence(**kwargs)
        captured["extension"] = extension
        return finalized, replace(
            extension,
            frozen_step_size=2.0 * extension.frozen_step_size,
        )

    with pytest.raises(ValueError, match="extension (execution contract is invalid|contract)"):
        run_bounded_operational_fixed_trajectory_selection(
            **_bounded_extension_kwargs(
                selector=_attempt5_like_selection,
                runner=runner,
                evidence_extender=corrupting_extender,
            )
        )

    extension = captured["extension"]
    with pytest.raises(ValueError, match="every and only inconclusive slot"):
        replace(extension, slots=extension.slots[:-1])

    first = extension.slots[0]
    corrupted_replication = replace(
        first.extended_replication,
        seed=(1234567, 7654321),
    )
    corrupted_slot = replace(first, extended_replication=corrupted_replication)
    with pytest.raises(ValueError, match="seed does not match its domain"):
        replace(
            extension,
            slots=(corrupted_slot,) + extension.slots[1:],
        )


def test_extension_boundary_rejects_stale_v3_acceptance_policy_signature() -> None:
    runner, _calls = _fake_extension_runner(
        probability_by_l={10: 0.70, 20: 0.40}
    )

    def corrupting_extender(**kwargs):
        finalized, extension = extend_operational_fixed_trajectory_evidence(**kwargs)
        stale_signature = _signature(
            "bayesfilter.hmc_acceptance_policy.v3",
            kwargs["acceptance_policy"].payload(),
        )
        # Simulate a corrupt custom extender after dataclass construction so
        # the outer selection boundary, rather than __post_init__, is tested.
        object.__setattr__(
            extension,
            "acceptance_policy_signature",
            stale_signature,
        )
        return finalized, extension

    with pytest.raises(
        ValueError,
        match="evidence extender changed the declared extension contract",
    ):
        run_bounded_operational_fixed_trajectory_selection(
            **_bounded_extension_kwargs(
                selector=_attempt5_like_selection,
                runner=runner,
                evidence_extender=corrupting_extender,
            )
        )


def test_outer_loop_rejects_candidate_set_and_initial_seed_corruption() -> None:
    runner, _calls = _fake_extension_runner(inconclusive=True)

    def missing_candidate_selector(**kwargs):
        selection = _attempt5_like_selection(**kwargs)
        return select_fixed_trajectory_representative(
            tuple(
                item
                for item in selection.candidate_results
                if item.candidate.num_leapfrog_steps != 20
            ),
            anchor_l=10,
        )

    missing_kwargs = _bounded_extension_kwargs(
        selector=missing_candidate_selector,
        runner=runner,
    )
    missing_kwargs["evidence_extension_checkpoints"] = ()
    with pytest.raises(ValueError, match="changed the declared candidate set"):
        run_bounded_operational_fixed_trajectory_selection(**missing_kwargs)

    def corrupted_seed_selector(**kwargs):
        selection = _attempt5_like_selection(**kwargs)
        first_result = selection.candidate_results[0]
        first_replication = replace(
            first_result.replications[0],
            seed=(1234, 5678),
        )
        corrupted_result = replace(
            first_result,
            replications=(first_replication,) + first_result.replications[1:],
        )
        return select_fixed_trajectory_representative(
            (corrupted_result,) + selection.candidate_results[1:],
            anchor_l=10,
        )

    seed_kwargs = _bounded_extension_kwargs(
        selector=corrupted_seed_selector,
        runner=runner,
    )
    seed_kwargs["evidence_extension_checkpoints"] = ()
    with pytest.raises(ValueError, match="seed does not match its domain"):
        run_bounded_operational_fixed_trajectory_selection(**seed_kwargs)


def test_outer_loop_rejects_fabricated_exact_l_retune_signature() -> None:
    def selector(**kwargs):
        passed = _evidence(0.70, draw_count=256)
        results = []
        for leapfrog in (5, 10, 20):
            candidate = FixedTrajectoryCandidate(
                anchor_l=10,
                num_leapfrog_steps=leapfrog,
                max_leapfrog_steps=64,
                coordinate_signature=kwargs["coordinate_signature"],
                metric_signature=kwargs["metric_signature"],
                start_bank_signature=kwargs["private_start_bank_signature"],
            )
            replications = tuple(
                FixedTrajectoryReplication(
                    candidate=candidate,
                    replication_index=index,
                    seed=(seed := paired_candidate_seed(
                        kwargs["root_seed"],
                        candidate_signature=candidate.signature,
                        replication_index=index,
                    )),
                    acceptance_evidence_payload=passed.payload(),
                    execution_contract_signature=_execution_signature(
                        candidate,
                        index,
                        seed,
                        kwargs,
                    ),
                )
                for index in range(3)
            )
            results.append(
                FixedTrajectoryCandidateResult(
                    candidate=candidate,
                    replications=replications,
                    exact_l_retuned_step_size=(
                        kwargs["frozen_step_size"] if leapfrog == 10 else None
                    ),
                    exact_l_retune_signature=(
                        "fabricated-retune-signature" if leapfrog == 10 else None
                    ),
                )
            )
        return select_fixed_trajectory_representative(results, anchor_l=10)

    runner, _calls = _fake_extension_runner(inconclusive=True)
    kwargs = _bounded_extension_kwargs(selector=selector, runner=runner)
    kwargs["evidence_extension_checkpoints"] = ()
    with pytest.raises(ValueError, match="exact-final-L retune lineage is invalid"):
        run_bounded_operational_fixed_trajectory_selection(**kwargs)


def test_candidate_data_invalid_matrix_stops_without_reason_only_retry() -> None:
    calls = []
    local = _local_veto_with_reasons("nonfinite_log_accept_ratio")

    def selector(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return _scripted_selection(kwargs, 0.90, selected=False)
        if len(calls) == 2:
            return _matrix_selection(kwargs, {5: local, 10: local, 20: local})
        return _scripted_selection(kwargs, 0.70, selected=True)

    runner, _runner_calls = _fake_extension_runner(inconclusive=True)
    kwargs = _bounded_extension_kwargs(selector=selector, runner=runner)
    kwargs["evidence_extension_checkpoints"] = ()
    result = run_bounded_operational_fixed_trajectory_selection(**kwargs)

    assert result.terminal_disposition == "candidate_set_exhausted"
    assert len(calls) == 2
    assert tuple(call["frozen_step_size"] for call in calls) == pytest.approx(
        (0.1, 0.2)
    )
    assert result.step_veto_recovery_count == 0
    assert result.attempts[1].repair is None
    assert result.attempts[1].lower_bound_source_attempt_index_before == 0
    assert result.attempts[1].lower_bound_source_attempt_index_after == 0
    assert all(len(attempt.replication_seeds) == 9 for attempt in result.attempts)
    all_seeds = tuple(
        seed for attempt in result.attempts for seed in attempt.replication_seeds
    )
    assert len(set(all_seeds)) == 18
    payload = result.payload()
    assert payload["selection_repair_loop_exercised"] is True
    assert payload["repair_loop_validated"] is False
    assert payload["step_veto_recovery_count"] == 0
    first_l, first_candidate_evidence = result.attempts[1].payload()[
        "candidate_evidence"
    ][0]
    first_result = next(
        item
        for item in result.attempts[1].selection.candidate_results
        if item.candidate.num_leapfrog_steps == first_l
    )
    assert first_candidate_evidence[0] == {
        "replication_index": 0,
        "evidence_validity": "candidate_data_invalid",
        "acceptance_decision": "unavailable",
        "engineering_invalidity_reasons": ("nonfinite_log_accept_ratio",),
        "candidate_promotion_vetoes": (),
        "candidate_health_alerts": (),
        "execution_contract_signature": first_result.replications[
            0
        ].execution_contract_signature,
    }


def test_step_veto_recovery_rejects_unknown_reason_and_missing_lower_bound() -> None:
    local_unknown = _local_veto_with_reasons("exception secret=/private/path")

    def first_attempt_local(**kwargs):
        return _matrix_selection(
            kwargs,
            {5: local_unknown, 10: local_unknown, 20: local_unknown},
        )

    runner, _calls = _fake_extension_runner(inconclusive=True)
    kwargs = _bounded_extension_kwargs(selector=first_attempt_local, runner=runner)
    kwargs["evidence_extension_checkpoints"] = ()
    result = run_bounded_operational_fixed_trajectory_selection(**kwargs)

    assert result.terminal_disposition == "candidate_set_exhausted"
    assert len(result.attempts) == 1
    assert result.step_veto_recovery_count == 0
    serialized = repr(result.payload())
    assert "/private/path" not in serialized
    assert "exception secret" not in serialized
    assert "unrecognized_health_failure" in serialized


def test_v3_evidence_rejects_unknown_legacy_health_field() -> None:
    candidate = _candidate(10)
    payload = dict(
        _local_veto_with_reasons("nonfinite_log_accept_ratio").payload()
    )
    payload["hard_health_failures"] = ()

    with pytest.raises(ValueError, match="field set is inconsistent"):
        FixedTrajectoryReplication(
            candidate=candidate,
            replication_index=0,
            seed=(20260711, 1),
            acceptance_evidence_payload=payload,
        )


def test_mixed_high_matrix_does_not_create_step_veto_lower_bound() -> None:
    calls = []
    high = _evidence(0.90, draw_count=256)
    local = _local_veto_with_reasons("nonfinite_log_accept_ratio")

    def selector(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return _matrix_selection(
                kwargs,
                {
                    5: (high, high, local),
                    10: (high, high, local),
                    20: (high, high, local),
                },
            )
        return _matrix_selection(kwargs, {5: local, 10: local, 20: local})

    runner, _calls = _fake_extension_runner(inconclusive=True)
    kwargs = _bounded_extension_kwargs(selector=selector, runner=runner)
    kwargs["evidence_extension_checkpoints"] = ()
    result = run_bounded_operational_fixed_trajectory_selection(**kwargs)

    assert len(result.attempts) == 2
    assert result.attempts[0].repair.direction == "higher_epsilon"
    assert result.attempts[0].bracket_after == (None, None)
    assert result.attempts[0].lower_bound_source_attempt_index_after is None
    assert result.terminal_disposition == "candidate_set_exhausted"
    assert result.step_veto_recovery_count == 0


def test_native_divergence_veto_cannot_mutate_step_or_consume_retry() -> None:
    calls = []
    divergence_veto = _native_divergence_veto_evidence(draw_count=256)

    def selector(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return _scripted_selection(kwargs, 0.90, selected=False)
        if len(calls) == 2:
            return _matrix_selection(
                kwargs,
                {5: divergence_veto, 10: divergence_veto, 20: divergence_veto},
            )
        if len(calls) < 5:
            return _scripted_selection(kwargs, 0.90, selected=False)
        return _matrix_selection(
            kwargs,
            {5: divergence_veto, 10: divergence_veto, 20: divergence_veto},
        )

    runner, _calls = _fake_extension_runner(inconclusive=True)
    kwargs = _bounded_extension_kwargs(selector=selector, runner=runner)
    kwargs["evidence_extension_checkpoints"] = ()
    result = run_bounded_operational_fixed_trajectory_selection(**kwargs)

    assert len(calls) == 2
    assert result.terminal_disposition == "budget_exhausted_valid"
    assert result.attempts[-1].repair is None
    assert result.attempts[-1].payload()["step_veto_recovery_applied"] is False
    assert result.step_veto_recovery_count == 0


def test_outer_loop_rejects_bank_content_and_execution_contract_corruption() -> None:
    runner, _calls = _fake_extension_runner(inconclusive=True)
    kwargs = _bounded_extension_kwargs(
        selector=_attempt5_like_selection,
        runner=runner,
    )
    kwargs["evidence_extension_checkpoints"] = ()
    kwargs["private_start_bank_signature"] = "forged-bank-signature"
    with pytest.raises(ValueError, match="does not match its content"):
        run_bounded_operational_fixed_trajectory_selection(**kwargs)

    def corrupt_contract_selector(**selector_kwargs):
        selection = _scripted_selection(selector_kwargs, 0.90, selected=False)
        first_result = selection.candidate_results[0]
        corrupted = replace(
            first_result.replications[0],
            execution_contract_signature="forged-execution-contract",
        )
        return select_fixed_trajectory_representative(
            (
                replace(
                    first_result,
                    replications=(corrupted,) + first_result.replications[1:],
                ),
                *selection.candidate_results[1:],
            ),
            anchor_l=10,
        )

    valid_kwargs = _bounded_extension_kwargs(
        selector=corrupt_contract_selector,
        runner=runner,
    )
    valid_kwargs["evidence_extension_checkpoints"] = ()
    with pytest.raises(ValueError, match="execution contract is invalid"):
        run_bounded_operational_fixed_trajectory_selection(**valid_kwargs)


@pytest.mark.parametrize(
    ("corrupt_evidence", "message"),
    (
        (
            lambda: _evidence(
                0.90,
                draw_count=256,
                policy=HMCAcceptancePolicy(
                    target=0.71,
                    practical_region=(0.66, 0.76),
                    repair_region=(0.56, 0.86),
                ),
            ),
            "changed the acceptance policy",
        ),
        (
            lambda: _evidence(0.90, draw_count=64),
            "changed the screen draw budget",
        ),
    ),
)
def test_outer_loop_reconstructs_policy_and_draw_execution_contract(
    corrupt_evidence,
    message: str,
) -> None:
    def selector(**selector_kwargs):
        selection = _scripted_selection(selector_kwargs, 0.90, selected=False)
        first_result = selection.candidate_results[0]
        corrupted = replace(
            first_result.replications[0],
            acceptance_evidence_payload=corrupt_evidence().payload(),
        )
        return select_fixed_trajectory_representative(
            (
                replace(
                    first_result,
                    replications=(corrupted,) + first_result.replications[1:],
                ),
                *selection.candidate_results[1:],
            ),
            anchor_l=10,
        )

    runner, _calls = _fake_extension_runner(inconclusive=True)
    kwargs = _bounded_extension_kwargs(selector=selector, runner=runner)
    kwargs["evidence_extension_checkpoints"] = ()
    with pytest.raises(ValueError, match=message):
        run_bounded_operational_fixed_trajectory_selection(**kwargs)
