from __future__ import annotations

import json
import os
from dataclasses import replace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import pytest

from bayesfilter.inference.hmc_fixed_metric_grid_search import (
    DEFAULT_L_GRID,
    CandidateTuneRejected,
    FixedMetricCandidateEvidencePolicy,
    FixedMetricCandidateRecord,
    FixedMetricGridExecutionConfig,
    FixedMetricGridSearchConfig,
    FixedMetricScreenRecord,
    FixedMetricScreenOutcome,
    FixedMetricScreenRequest,
    FixedMetricSearchLineage,
    FixedMetricTuneOutcome,
    GridSearchResourceCloseout,
    GridSearchTargetVeto,
    SharedGridSearchInvalidity,
    aggregate_fixed_metric_candidate_evidence,
    confirm_fixed_metric_candidate,
    fixed_metric_search_seed,
    refinement_l_values,
    run_fixed_metric_candidate,
    run_fixed_metric_confirmation_screen,
    run_fixed_metric_grid_search,
)
from bayesfilter.inference.hmc_verification import (
    HMCAcceptancePolicy,
    evaluate_hmc_acceptance_evidence,
)


LINEAGE = FixedMetricSearchLineage(
    coordinate_signature="coordinate",
    metric_signature="metric",
    private_start_bank_content_signature="start-bank-content",
    common_state_signature="common-state",
)
POLICY = HMCAcceptancePolicy()
SPAWN_FIXTURE_MODULE = "bayesfilter.testing.hmc_fixed_metric_grid_search_fixture"


def _evidence(
    probability: float = 0.70,
    *,
    draw_count: int = 64,
    candidate_invalid: bool = False,
    policy: HMCAcceptancePolicy = POLICY,
):
    draws = np.arange(draw_count, dtype=float)[:, None, None]
    chains = np.arange(4, dtype=float)[None, :, None]
    samples = np.concatenate((draws + chains, 2.0 * draws + chains), axis=2)
    values = np.full((draw_count, 4), probability)
    if candidate_invalid:
        values[0, 0] = np.nan
    return evaluate_hmc_acceptance_evidence(
        samples=samples,
        log_accept_ratio=np.log(values),
        is_accepted=np.ones((draw_count, 4), dtype=bool),
        policy=policy,
    )


def _tune_runner(calls, *, reject=()):
    rejected = set(reject)

    def runner(request):
        calls.append(request)
        if request.num_leapfrog_steps in rejected:
            raise CandidateTuneRejected("nonfinite_adapted_step_size")
        return FixedMetricTuneOutcome(
            num_leapfrog_steps=request.num_leapfrog_steps,
            seed=request.seed,
            tuned_step_size=1.0 / (request.num_leapfrog_steps + 10.0),
            lineage=request.lineage,
        )

    return runner


def _screen_runner(calls, *, probability_by_l=None, invalid_l=()):
    probabilities = {} if probability_by_l is None else dict(probability_by_l)
    invalid = set(invalid_l)

    def runner(request):
        calls.append(request)
        evidence = _evidence(
            probabilities.get(request.num_leapfrog_steps, 0.70),
            draw_count=request.num_results,
            candidate_invalid=request.num_leapfrog_steps in invalid,
        )
        return FixedMetricScreenOutcome(
            num_leapfrog_steps=request.num_leapfrog_steps,
            replication_index=request.replication_index,
            seed=request.seed,
            tuned_step_size=request.tuned_step_size,
            lineage=request.lineage,
            acceptance_evidence_payload=evidence.payload(),
        )

    return runner


def _screen_record(
    *,
    replication_index,
    chain_means,
    stage="screen",
    num_results=64,
    leapfrog=9,
    step_size=0.25,
    stuck=False,
):
    probabilities = np.tile(np.asarray(chain_means, dtype=float), (num_results, 1))
    draws = np.arange(num_results, dtype=float)[:, None, None]
    chains = np.arange(4, dtype=float)[None, :, None]
    samples = np.zeros((num_results, 4, 2), dtype=float) if stuck else np.concatenate(
        (draws + chains, 2.0 * draws + chains), axis=2
    )
    evidence = evaluate_hmc_acceptance_evidence(
        samples=samples,
        log_accept_ratio=np.log(probabilities),
        is_accepted=np.ones((num_results, 4), dtype=bool),
        policy=POLICY,
    )
    request = FixedMetricScreenRequest(
        round_index=0,
        stage=stage,
        num_leapfrog_steps=leapfrog,
        replication_index=replication_index,
        seed=fixed_metric_search_seed(
            (20260719, 7300),
            domain=f"round_0_{stage}_{num_results}",
            num_leapfrog_steps=leapfrog,
            replication_index=replication_index,
        ),
        tuned_step_size=step_size,
        num_results=num_results,
        lineage=LINEAGE,
    )
    return FixedMetricScreenRecord(request=request, evidence_payload=evidence.payload())


def _candidate_from_chain_means(replication_chain_means, *, step_size=0.25):
    screens = tuple(
        _screen_record(
            replication_index=index,
            chain_means=means,
            step_size=step_size,
        )
        for index, means in enumerate(replication_chain_means)
    )
    return FixedMetricCandidateRecord(
        round_index=0,
        num_leapfrog_steps=9,
        tune_seed=fixed_metric_search_seed(
            (20260719, 7300), domain="round_0_tune", num_leapfrog_steps=9
        ),
        tuned_step_size=step_size,
        screens=screens,
    )


def _run(*, config=None, tune_runner=None, screen_runner=None):
    tune_calls = []
    screen_calls = []
    result = run_fixed_metric_grid_search(
        config=FixedMetricGridSearchConfig() if config is None else config,
        lineage=LINEAGE,
        acceptance_policy=POLICY,
        tune_runner=(
            _tune_runner(tune_calls) if tune_runner is None else tune_runner
        ),
        screen_runner=(
            _screen_runner(screen_calls) if screen_runner is None else screen_runner
        ),
    )
    return result, tune_calls, screen_calls


def _process_execution(factory: str, *, max_workers: int = 2):
    return FixedMetricGridExecutionConfig(
        mode="process_parallel",
        max_workers=max_workers,
        worker_factory_locator=f"{SPAWN_FIXTURE_MODULE}:{factory}",
        worker_environment=(("CUDA_VISIBLE_DEVICES", "-1"),),
    )


def _run_parallel(*, factory="deterministic_worker_factory", config=None):
    return run_fixed_metric_grid_search(
        config=FixedMetricGridSearchConfig() if config is None else config,
        lineage=LINEAGE,
        acceptance_policy=POLICY,
        execution=_process_execution(factory),
    )


def _search_mechanics(result):
    return {
        "round0_candidates": tuple(
            item.payload() for item in result.round0_candidates
        ),
        "refinement_candidates": tuple(
            item.payload() for item in result.refinement_candidates
        ),
        "survivors": tuple(item.payload() for item in result.survivors),
        "disposition": result.disposition,
        "shared_invalidity_reasons": result.shared_invalidity_reasons,
    }


def test_default_grid_is_preserved_and_custom_grid_is_supported() -> None:
    assert FixedMetricGridSearchConfig().l_grid == (3, 5, 9, 13, 18, 25)
    assert 1 not in DEFAULT_L_GRID and 2 not in DEFAULT_L_GRID
    assert FixedMetricGridSearchConfig(l_grid=(2, 4, 7, 11, 17, 25)).l_grid == (
        2,
        4,
        7,
        11,
        17,
        25,
    )
    assert FixedMetricGridSearchConfig(
        l_grid=(2, 4, 7, 11, 17, 21, 23, 25)
    ).l_grid == (2, 4, 7, 11, 17, 21, 23, 25)
    with pytest.raises(ValueError, match="at least three"):
        FixedMetricGridSearchConfig(l_grid=(2, 7))
    with pytest.raises(ValueError, match="distinct"):
        FixedMetricGridSearchConfig(l_grid=(2, 4, 7, 11, 17, 17))
    with pytest.raises(ValueError, match="L=25 bound"):
        FixedMetricGridSearchConfig(l_grid=(2, 4, 7, 11, 17, 26))
    assert FixedMetricGridSearchConfig(refinement_rounds=0).refinement_rounds == 0
    with pytest.raises(ValueError, match="zero or one refinement"):
        FixedMetricGridSearchConfig(refinement_rounds=2)


def test_round0_only_mode_never_launches_refinement() -> None:
    result, tune_calls, _ = _run(
        config=FixedMetricGridSearchConfig(refinement_rounds=0)
    )
    assert tuple(item.num_leapfrog_steps for item in tune_calls) == DEFAULT_L_GRID
    assert result.disposition == "survivor_set"
    assert result.refinement_candidates == ()
    assert result.public_summary()["planned_refinement_count"] == 0
    assert result.public_summary()["completed_refinement_count"] == 0


def test_refinement_uses_only_adjacent_tested_intervals() -> None:
    assert refinement_l_values(DEFAULT_L_GRID, (3,)) == (4,)
    assert refinement_l_values(DEFAULT_L_GRID, (9,)) == (7, 11)
    assert refinement_l_values(DEFAULT_L_GRID, (25,)) == (21, 22)
    assert refinement_l_values(DEFAULT_L_GRID, (3, 9, 25)) == (
        4,
        7,
        11,
        21,
        22,
    )
    custom = (2, 4, 7, 11, 17, 25)
    assert refinement_l_values(custom, (2, 11, 25)) == (3, 9, 14, 21)


def test_custom_grid_runs_requested_candidates_and_refines_active_grid() -> None:
    config = FixedMetricGridSearchConfig(l_grid=(2, 4, 7, 11, 17, 25))
    result, tune_calls, _ = _run(config=config)
    assert tuple(item.num_leapfrog_steps for item in tune_calls) == (
        2,
        4,
        7,
        11,
        17,
        25,
        3,
        5,
        6,
        9,
        14,
        21,
    )
    assert result.public_summary()["planned_initial_count"] == 6
    assert result.public_summary()["planned_refinement_count"] == 6


def test_grid_runner_does_not_invoke_confirmation() -> None:
    result, _, _ = _run(config=FixedMetricGridSearchConfig(refinement_rounds=0))
    assert all(
        screen.request.stage in {"screen", "evidence_extension"}
        for candidate in result.candidates
        for screen in candidate.screens
    )


def test_every_candidate_owns_an_independent_tune_and_exact_screen_step() -> None:
    result, tune_calls, screen_calls = _run()

    expected_refinement = (4, 7, 11, 15, 16, 21, 22)
    assert tuple(item.num_leapfrog_steps for item in tune_calls) == (
        *DEFAULT_L_GRID,
        *expected_refinement,
    )
    assert len({item.seed for item in tune_calls}) == len(tune_calls)
    assert result.disposition == "survivor_set"
    assert tuple(item.num_leapfrog_steps for item in result.survivors) == tuple(
        sorted((*DEFAULT_L_GRID, *expected_refinement))
    )
    tuned_by_l = {
        item.num_leapfrog_steps: item.tuned_step_size for item in result.candidates
    }
    assert len(set(tuned_by_l.values())) == len(tuned_by_l)
    for request in screen_calls:
        assert request.tuned_step_size == tuned_by_l[request.num_leapfrog_steps]
        assert request.lineage == LINEAGE
    assert not ({item.seed for item in tune_calls} & {item.seed for item in screen_calls})


def test_candidate_order_permutation_preserves_seeds_steps_and_survivors() -> None:
    forward, _, _ = _run()
    reverse, _, _ = _run(
        config=FixedMetricGridSearchConfig(l_grid=tuple(reversed(DEFAULT_L_GRID)))
    )

    def summary(result):
        return {
            item.num_leapfrog_steps: (item.tune_seed, item.tuned_step_size, item.survivor)
            for item in result.candidates
        }

    assert summary(forward) == summary(reverse)
    assert tuple(item.num_leapfrog_steps for item in forward.survivors) == tuple(
        item.num_leapfrog_steps for item in reverse.survivors
    )


def test_public_single_candidate_matches_serial_grid_candidate() -> None:
    config = FixedMetricGridSearchConfig(refinement_rounds=0)
    result, _, _ = _run(config=config)
    candidate = run_fixed_metric_candidate(
        round_index=0,
        num_leapfrog_steps=9,
        config=config,
        lineage=LINEAGE,
        acceptance_policy=POLICY,
        tune_runner=_tune_runner([]),
        screen_runner=_screen_runner([]),
    )

    expected = next(
        item for item in result.round0_candidates if item.num_leapfrog_steps == 9
    )
    assert candidate.payload() == expected.payload()


def test_spawn_parallel_round0_matches_serial_mechanics() -> None:
    config = FixedMetricGridSearchConfig(refinement_rounds=0)
    serial, _, _ = _run(config=config)
    parallel = _run_parallel(config=config)

    assert _search_mechanics(parallel) == _search_mechanics(serial)
    assert parallel.execution.mode == "process_parallel"
    assert parallel.public_summary()["execution_worker_count"] == 2


def test_spawn_parallel_refinement_matches_serial_mechanics() -> None:
    serial, _, _ = _run()
    parallel = _run_parallel()

    assert _search_mechanics(parallel) == _search_mechanics(serial)
    assert tuple(item.num_leapfrog_steps for item in parallel.refinement_candidates) == (
        4,
        7,
        11,
        15,
        16,
        21,
        22,
    )


def test_spawn_parallel_reversed_order_preserves_mechanics() -> None:
    forward = _run_parallel()
    reverse = _run_parallel(
        config=FixedMetricGridSearchConfig(l_grid=tuple(reversed(DEFAULT_L_GRID)))
    )

    assert _search_mechanics(reverse) == _search_mechanics(forward)


def test_spawn_parallel_completion_callbacks_follow_declared_order() -> None:
    completed = []
    config = FixedMetricGridSearchConfig(refinement_rounds=0)

    run_fixed_metric_grid_search(
        config=config,
        lineage=LINEAGE,
        acceptance_policy=POLICY,
        execution=_process_execution("out_of_order_worker_factory"),
        after_candidate=lambda candidate, index, total: completed.append(
            (candidate.num_leapfrog_steps, index, total)
        ),
    )

    assert completed == [
        (leapfrog, index, len(DEFAULT_L_GRID))
        for index, leapfrog in enumerate(DEFAULT_L_GRID)
    ]


def test_spawn_parallel_candidate_rejections_remain_local() -> None:
    result = _run_parallel(factory="candidate_rejection_worker_factory")

    tune_rejected = next(
        item for item in result.round0_candidates if item.num_leapfrog_steps == 9
    )
    screen_rejected = next(
        item for item in result.round0_candidates if item.num_leapfrog_steps == 13
    )
    assert tune_rejected.rejection_stage == "tune"
    assert screen_rejected.rejection_stage == "screen"
    assert result.shared_invalidity_reasons == ()
    assert result.disposition == "survivor_set"
    assert any(item.num_leapfrog_steps == 18 for item in result.survivors)


@pytest.mark.parametrize(
    "factory",
    (
        "bootstrap_failure_worker_factory",
        "missing_worker_factory",
    ),
)
def test_spawn_parallel_factory_failure_is_shared_invalidity(factory: str) -> None:
    result = _run_parallel(factory=factory)

    assert result.disposition == "shared_execution_invalid"
    assert result.shared_invalidity_reasons == ("untyped_callback_failure",)
    assert result.survivors == ()
    assert result.refinement_candidates == ()


@pytest.mark.parametrize(
    ("factory", "exception", "message"),
    (
        ("target_veto_worker_factory", GridSearchTargetVeto, "fixture target veto"),
        (
            "resource_closeout_worker_factory",
            GridSearchResourceCloseout,
            "fixture resource closeout",
        ),
    ),
)
def test_spawn_parallel_target_and_resource_vetoes_propagate(
    factory, exception, message
) -> None:
    with pytest.raises(exception, match=message):
        _run_parallel(factory=factory)


def test_process_execution_configuration_fails_closed() -> None:
    with pytest.raises(ValueError, match="module:factory"):
        FixedMetricGridExecutionConfig(
            mode="process_parallel",
            worker_environment=(("CUDA_VISIBLE_DEVICES", "-1"),),
        )
    with pytest.raises(ValueError, match="requires spawn"):
        FixedMetricGridExecutionConfig(
            mode="process_parallel",
            worker_factory_locator=f"{SPAWN_FIXTURE_MODULE}:deterministic_worker_factory",
            worker_environment=(("CUDA_VISIBLE_DEVICES", "-1"),),
            start_method="fork",
        )
    with pytest.raises(ValueError, match="explicit CUDA_VISIBLE_DEVICES"):
        FixedMetricGridExecutionConfig(
            mode="process_parallel",
            worker_factory_locator=f"{SPAWN_FIXTURE_MODULE}:deterministic_worker_factory",
        )
    with pytest.raises(ValueError, match="TF_FORCE_GPU_ALLOW_GROWTH=true"):
        FixedMetricGridExecutionConfig(
            mode="process_parallel",
            worker_factory_locator=f"{SPAWN_FIXTURE_MODULE}:deterministic_worker_factory",
            worker_environment=(("CUDA_VISIBLE_DEVICES", "0"),),
        )


def test_process_mode_rejects_parent_callbacks() -> None:
    with pytest.raises(ValueError, match="constructs callbacks inside workers"):
        run_fixed_metric_grid_search(
            config=FixedMetricGridSearchConfig(refinement_rounds=0),
            lineage=LINEAGE,
            acceptance_policy=POLICY,
            tune_runner=_tune_runner([]),
            execution=_process_execution("deterministic_worker_factory"),
        )


def test_observed_l9_threshold_crossings_require_confirmation() -> None:
    candidate = _candidate_from_chain_means(
        (
            (0.7163003803, 0.7969690089, 0.7745873438, 0.7022517138),
            (0.6574737035, 0.8134493364, 0.7451720450, 0.8179564750),
            (0.6468890152, 0.7550896739, 0.7271022691, 0.7645991318),
        ),
        step_size=0.26673749196885876,
    )

    aggregate = aggregate_fixed_metric_candidate_evidence(
        phase="nomination",
        num_leapfrog_steps=9,
        tuned_step_size=candidate.tuned_step_size,
        screens=candidate.screens,
        acceptance_policy=POLICY,
        evidence_policy=FixedMetricCandidateEvidencePolicy(),
    )

    assert candidate.survivor is False
    assert aggregate.disposition == "confirmation_required"
    assert aggregate.hard_rejection_reasons == ()
    assert aggregate.grand_mean == pytest.approx(0.7431533414)
    assert aggregate.working_interval[0] < 0.75
    assert aggregate.working_interval[1] > 0.65


def test_hard_health_veto_rejects_without_confirmation() -> None:
    candidate = FixedMetricCandidateRecord(
        round_index=0,
        num_leapfrog_steps=9,
        tune_seed=(1, 2),
        tuned_step_size=0.25,
        screens=tuple(
            _screen_record(
                replication_index=index,
                chain_means=(0.70,) * 4,
                stuck=index == 0,
            )
            for index in range(3)
        ),
    )
    calls = []

    result = confirm_fixed_metric_candidate(
        candidate=candidate,
        config=FixedMetricGridSearchConfig(refinement_rounds=0),
        lineage=LINEAGE,
        acceptance_policy=POLICY,
        evidence_policy=FixedMetricCandidateEvidencePolicy(),
        screen_runner=lambda request: calls.append(request),
    )

    assert result.disposition == "hard_rejected"
    assert "movement_gate_failed" in result.rejection_reasons
    assert result.confirmation_screens == ()
    assert calls == []


@pytest.mark.parametrize(
    ("means", "disposition"),
    [
        ((0.40, 0.41, 0.42, 0.43), "needs_lower_epsilon"),
        ((0.87, 0.88, 0.89, 0.90), "needs_higher_epsilon"),
    ],
)
def test_aggregate_interval_supports_only_clear_epsilon_repairs(
    means, disposition
) -> None:
    screens = tuple(
        _screen_record(replication_index=index, chain_means=means)
        for index in range(3)
    )

    result = aggregate_fixed_metric_candidate_evidence(
        phase="nomination",
        num_leapfrog_steps=9,
        tuned_step_size=0.25,
        screens=screens,
        acceptance_policy=POLICY,
        evidence_policy=FixedMetricCandidateEvidencePolicy(),
    )

    assert result.disposition == disposition


def test_only_fresh_confirmation_can_produce_provisional_viability() -> None:
    candidate = _candidate_from_chain_means(
        (
            (0.64, 0.76, 0.70, 0.72),
            (0.68, 0.74, 0.71, 0.73),
            (0.69, 0.72, 0.70, 0.71),
        )
    )
    calls = []

    def confirmation(request):
        calls.append(request)
        record = _screen_record(
            replication_index=request.replication_index,
            chain_means=(0.68, 0.70, 0.72, 0.74),
            stage="confirmation",
            num_results=request.num_results,
            step_size=request.tuned_step_size,
        )
        return FixedMetricScreenOutcome(
            num_leapfrog_steps=request.num_leapfrog_steps,
            replication_index=request.replication_index,
            seed=request.seed,
            tuned_step_size=request.tuned_step_size,
            lineage=request.lineage,
            acceptance_evidence_payload=record.evidence_payload,
        )

    result = confirm_fixed_metric_candidate(
        candidate=candidate,
        config=FixedMetricGridSearchConfig(refinement_rounds=0),
        lineage=LINEAGE,
        acceptance_policy=POLICY,
        evidence_policy=FixedMetricCandidateEvidencePolicy(),
        screen_runner=confirmation,
    )

    assert result.nomination.disposition == "confirmation_required"
    assert result.nomination.provisional_viable is False
    assert result.disposition == "provisional_viable"
    assert result.provisional_viable is True
    assert len(calls) == 3
    assert all(item.stage == "confirmation" and item.num_results == 256 for item in calls)


def test_broad_fresh_confirmation_remains_unresolved() -> None:
    candidate = _candidate_from_chain_means(((0.70,) * 4,) * 3)

    def confirmation(request):
        record = _screen_record(
            replication_index=request.replication_index,
            chain_means=(0.40, 0.45, 0.90, 0.95),
            stage="confirmation",
            num_results=request.num_results,
            step_size=request.tuned_step_size,
        )
        return FixedMetricScreenOutcome(
            num_leapfrog_steps=request.num_leapfrog_steps,
            replication_index=request.replication_index,
            seed=request.seed,
            tuned_step_size=request.tuned_step_size,
            lineage=request.lineage,
            acceptance_evidence_payload=record.evidence_payload,
        )

    result = confirm_fixed_metric_candidate(
        candidate=candidate,
        config=FixedMetricGridSearchConfig(refinement_rounds=0),
        lineage=LINEAGE,
        acceptance_policy=POLICY,
        evidence_policy=FixedMetricCandidateEvidencePolicy(),
        screen_runner=confirmation,
    )

    assert result.disposition == "unresolved_budget"
    assert result.provisional_viable is False


def test_confirmation_seed_domain_is_fresh_and_deterministic() -> None:
    root = (20260719, 7300)
    seeds = {
        fixed_metric_search_seed(
            root,
            domain="round_0_confirmation_256",
            num_leapfrog_steps=9,
            replication_index=index,
        )
        for index in range(3)
    }
    old = {
        fixed_metric_search_seed(
            root,
            domain=domain,
            num_leapfrog_steps=9,
            replication_index=index,
        )
        for domain in ("round_0_screen_64", "round_0_evidence_extension_128")
        for index in range(3)
    }

    assert len(seeds) == 3
    assert seeds.isdisjoint(old)


def test_confirmation_policy_payload_records_dependence_limitations() -> None:
    payload = FixedMetricCandidateEvidencePolicy().payload()

    assert payload["working_interval_unit"] == (
        "freshly_seeded_chain_run_mean_with_shared_start"
    )
    assert payload["working_interval_role"] == (
        "bounded_tuning_heuristic_not_confidence_guarantee"
    )
    assert set(payload["working_interval_dependence_limitations"]) == {
        "shared_initial_position",
        "within_chain_mcmc_autocorrelation",
        "chain_run_means_not_proven_independent_or_gaussian",
    }


def test_public_single_confirmation_screen_uses_fresh_identity() -> None:
    calls = []

    record = run_fixed_metric_confirmation_screen(
        round_index=0,
        num_leapfrog_steps=9,
        replication_index=2,
        tuned_step_size=0.25,
        config=FixedMetricGridSearchConfig(refinement_rounds=0),
        lineage=LINEAGE,
        acceptance_policy=POLICY,
        evidence_policy=FixedMetricCandidateEvidencePolicy(),
        screen_runner=_screen_runner(calls),
    )

    assert len(calls) == 1
    assert record.request.stage == "confirmation"
    assert record.request.num_results == 256
    assert record.request.seed == fixed_metric_search_seed(
        (20260719, 7300),
        domain="round_0_confirmation_256",
        num_leapfrog_steps=9,
        replication_index=2,
    )


def test_candidate_local_tune_failure_does_not_abort_other_grid_candidates() -> None:
    tune_calls = []
    screen_calls = []
    result = run_fixed_metric_grid_search(
        config=FixedMetricGridSearchConfig(),
        lineage=LINEAGE,
        acceptance_policy=POLICY,
        tune_runner=_tune_runner(tune_calls, reject=(9,)),
        screen_runner=_screen_runner(screen_calls),
    )

    failed = next(item for item in result.round0_candidates if item.num_leapfrog_steps == 9)
    assert failed.rejection_stage == "tune"
    assert failed.screens == ()
    assert tuple(item.num_leapfrog_steps for item in tune_calls[:6]) == DEFAULT_L_GRID
    assert any(item.num_leapfrog_steps == 13 for item in screen_calls)
    assert result.shared_invalidity_reasons == ()


def test_candidate_local_screen_invalidity_does_not_become_shared() -> None:
    tune_calls = []
    screen_calls = []
    result = run_fixed_metric_grid_search(
        config=FixedMetricGridSearchConfig(),
        lineage=LINEAGE,
        acceptance_policy=POLICY,
        tune_runner=_tune_runner(tune_calls),
        screen_runner=_screen_runner(screen_calls, invalid_l=(9,)),
    )

    failed = next(item for item in result.round0_candidates if item.num_leapfrog_steps == 9)
    assert failed.rejection_stage == "screen"
    assert failed.rejection_replication_index == 0
    assert failed.rejection_reasons == ("nonfinite_log_accept_ratio",)
    assert result.shared_invalidity_reasons == ()
    assert any(item.num_leapfrog_steps == 13 for item in screen_calls)


@pytest.mark.parametrize(
    ("field", "reason"),
    [
        ("coordinate_signature", "coordinate_signature_mismatch"),
        ("metric_signature", "metric_signature_mismatch"),
        (
            "private_start_bank_content_signature",
            "start_bank_signature_mismatch",
        ),
        ("common_state_signature", "common_state_signature_mismatch"),
    ],
)
def test_lineage_corruption_stops_barrier_and_clears_survivors(field, reason) -> None:
    calls = []

    def tune(request):
        calls.append(request)
        lineage = request.lineage
        if request.num_leapfrog_steps == 9:
            lineage = replace(lineage, **{field: "corrupted"})
        return FixedMetricTuneOutcome(
            num_leapfrog_steps=request.num_leapfrog_steps,
            seed=request.seed,
            tuned_step_size=0.05,
            lineage=lineage,
        )

    result, _, _ = _run(tune_runner=tune)
    assert result.disposition == "shared_execution_invalid"
    assert result.shared_invalidity_reasons == (reason,)
    assert tuple(item.num_leapfrog_steps for item in result.round0_candidates) == (3, 5)
    assert result.survivors == ()
    assert result.public_summary()["surviving_count"] == 0
    assert result.public_summary()["planned_refinement_count"] == 0
    assert result.refinement_candidates == ()


def test_interrupted_refinement_reports_planned_and_completed_counts() -> None:
    def tune(request):
        if request.round_index == 1 and request.num_leapfrog_steps == 7:
            raise SharedGridSearchInvalidity("shared_callback_invalid")
        return FixedMetricTuneOutcome(
            num_leapfrog_steps=request.num_leapfrog_steps,
            seed=request.seed,
            tuned_step_size=0.05,
            lineage=request.lineage,
        )

    result, _, _ = _run(tune_runner=tune)
    public = result.public_summary()
    assert result.disposition == "shared_execution_invalid"
    assert public["planned_refinement_count"] == 7
    assert public["completed_refinement_count"] == 1


def test_unknown_callback_failure_is_shared_invalidity() -> None:
    def broken(request):
        if request.num_leapfrog_steps == 5:
            raise RuntimeError("untyped failure")
        return FixedMetricTuneOutcome(
            num_leapfrog_steps=request.num_leapfrog_steps,
            seed=request.seed,
            tuned_step_size=0.05,
            lineage=request.lineage,
        )

    result, _, _ = _run(tune_runner=broken)
    assert result.disposition == "shared_execution_invalid"
    assert result.shared_invalidity_reasons == ("untyped_callback_failure",)


def test_evidence_extension_replaces_only_inconclusive_replication_identity() -> None:
    tune_calls = []
    screen_calls = []

    def screen(request):
        screen_calls.append(request)
        draw_count = request.num_results
        evidence = _evidence(0.70, draw_count=draw_count)
        return FixedMetricScreenOutcome(
            num_leapfrog_steps=request.num_leapfrog_steps,
            replication_index=request.replication_index,
            seed=request.seed,
            tuned_step_size=request.tuned_step_size,
            lineage=request.lineage,
            acceptance_evidence_payload=evidence.payload(),
        )

    result = run_fixed_metric_grid_search(
        config=FixedMetricGridSearchConfig(
            screen_num_results=32,
            extension_num_results=64,
        ),
        lineage=LINEAGE,
        acceptance_policy=POLICY,
        tune_runner=_tune_runner(tune_calls, reject=(5, 9, 13, 18, 25)),
        screen_runner=screen,
    )

    candidate = next(item for item in result.round0_candidates if item.num_leapfrog_steps == 3)
    assert len(candidate.evidence_extensions) == 3
    assert tuple(item.replication_index for item in candidate.evidence_extensions) == (0, 1, 2)
    assert all(item.replacement.request.stage == "evidence_extension" for item in candidate.evidence_extensions)
    assert all(item.replacement.request.num_results == 64 for item in candidate.evidence_extensions)
    assert candidate.survivor is True
    initial_seeds = {
        item.seed for item in screen_calls if item.stage == "screen" and item.num_leapfrog_steps == 3
    }
    extension_seeds = {
        item.seed
        for item in screen_calls
        if item.stage == "evidence_extension" and item.num_leapfrog_steps == 3
    }
    assert initial_seeds.isdisjoint(extension_seeds)


def test_acceptance_policy_mismatch_is_shared_invalidity() -> None:
    different_policy = HMCAcceptancePolicy(target=0.71, practical_region=(0.66, 0.76))

    def screen(request):
        return FixedMetricScreenOutcome(
            num_leapfrog_steps=request.num_leapfrog_steps,
            replication_index=request.replication_index,
            seed=request.seed,
            tuned_step_size=request.tuned_step_size,
            lineage=request.lineage,
            acceptance_evidence_payload=_evidence(policy=different_policy).payload(),
        )

    result, _, _ = _run(screen_runner=screen)
    assert result.disposition == "shared_execution_invalid"
    assert result.shared_invalidity_reasons == ("acceptance_policy_mismatch",)


def test_public_summary_is_aggregate_only_and_private_payload_has_no_raw_state() -> None:
    result, _, _ = _run()
    public = result.public_summary()
    public_text = json.dumps(public, sort_keys=True)
    private = result.payload()
    private_text = json.dumps(private, sort_keys=True)

    assert public["surviving_count"] == len(result.survivors)
    for forbidden in (
        "num_leapfrog_steps",
        "tuned_step_size",
        "root_seed",
        "coordinate_signature",
        "metric_signature",
        "start_bank",
        "common_state_signature",
        "candidate_signature",
        "representative_signature",
    ):
        assert forbidden not in public_text
    assert private["representative"] is None
    assert private["stochastic_ranking_performed"] is False
    assert private["raw_samples_exposed"] is False
    assert private["raw_states_exposed"] is False
    assert "effective_sample_size_by_coordinate" not in private_text


def test_seed_domains_are_order_independent_and_disjoint() -> None:
    root = (20260719, 7300)
    tune = fixed_metric_search_seed(
        root, domain="round_0_tune", num_leapfrog_steps=9
    )
    screens = {
        fixed_metric_search_seed(
            root,
            domain="round_0_screen_64",
            num_leapfrog_steps=9,
            replication_index=index,
        )
        for index in range(3)
    }
    extension = fixed_metric_search_seed(
        root,
        domain="round_0_evidence_extension_128",
        num_leapfrog_steps=9,
        replication_index=0,
    )
    assert len(screens) == 3
    assert tune not in screens
    assert extension not in screens and extension != tune


def test_candidate_callbacks_preserve_complete_boundaries_and_resource_scope() -> None:
    tune_calls = []
    screen_calls = []
    completed = []

    def before(round_index, candidate_index, leapfrog):
        if round_index == 0 and candidate_index == 2:
            raise GridSearchResourceCloseout("budget")

    with pytest.raises(GridSearchResourceCloseout, match="budget"):
        run_fixed_metric_grid_search(
            config=FixedMetricGridSearchConfig(),
            lineage=LINEAGE,
            acceptance_policy=POLICY,
            tune_runner=_tune_runner(tune_calls),
            screen_runner=_screen_runner(screen_calls),
            before_candidate=before,
            after_candidate=lambda candidate, *_: completed.append(candidate),
        )
    assert tuple(item.num_leapfrog_steps for item in completed) == (3, 5)


@pytest.mark.parametrize("stage", ("tune", "screen"))
def test_shared_target_veto_propagates_to_caller_closeout(stage: str) -> None:
    def tune(request):
        if stage == "tune":
            raise GridSearchTargetVeto("covariance target veto")
        return FixedMetricTuneOutcome(
            num_leapfrog_steps=request.num_leapfrog_steps,
            seed=request.seed,
            tuned_step_size=0.05,
            lineage=request.lineage,
        )

    def screen(request):
        if stage == "screen":
            raise GridSearchTargetVeto("covariance target veto")
        return _screen_runner([])(request)

    with pytest.raises(GridSearchTargetVeto, match="covariance target veto"):
        run_fixed_metric_grid_search(
            config=FixedMetricGridSearchConfig(),
            lineage=LINEAGE,
            acceptance_policy=POLICY,
            tune_runner=tune,
            screen_runner=screen,
        )
