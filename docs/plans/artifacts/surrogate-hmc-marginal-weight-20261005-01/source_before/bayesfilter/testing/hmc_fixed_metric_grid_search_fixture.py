"""Deterministic spawned-worker fixtures for fixed-metric grid-search tests.

These callbacks exercise scheduling and failure scope only.  They do not run
HMC and must not be used as numerical or sampler-validity evidence.
"""

from __future__ import annotations

import os
import time

import numpy as np

from bayesfilter.inference.hmc_fixed_metric_grid_search import (
    CandidateTuneRejected,
    FixedMetricCandidateRunners,
    FixedMetricCandidateWorkerRequest,
    FixedMetricScreenOutcome,
    FixedMetricTuneOutcome,
    GridSearchResourceCloseout,
    GridSearchTargetVeto,
)
from bayesfilter.inference.hmc_verification import evaluate_hmc_acceptance_evidence


def _acceptance_payload(request, policy, *, candidate_invalid: bool = False):
    draw_count = request.num_results
    draws = np.arange(draw_count, dtype=float)[:, None, None]
    chains = np.arange(4, dtype=float)[None, :, None]
    samples = np.concatenate((draws + chains, 2.0 * draws + chains), axis=2)
    probabilities = np.full((draw_count, 4), 0.70)
    if candidate_invalid:
        probabilities[0, 0] = np.nan
    return evaluate_hmc_acceptance_evidence(
        samples=samples,
        log_accept_ratio=np.log(probabilities),
        is_accepted=np.ones((draw_count, 4), dtype=bool),
        policy=policy,
    ).payload()


def _runners(
    worker_request: FixedMetricCandidateWorkerRequest,
    *,
    reject_tune: bool = False,
    reject_screen: bool = False,
    target_veto: bool = False,
    resource_closeout: bool = False,
    tune_delay_seconds: float = 0.0,
) -> FixedMetricCandidateRunners:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise RuntimeError("spawn fixture requires the declared CPU-only environment")

    def tune(request):
        if tune_delay_seconds:
            time.sleep(tune_delay_seconds)
        if target_veto:
            raise GridSearchTargetVeto("fixture target veto")
        if resource_closeout:
            raise GridSearchResourceCloseout("fixture resource closeout")
        if reject_tune:
            raise CandidateTuneRejected("nonfinite_adapted_step_size")
        return FixedMetricTuneOutcome(
            num_leapfrog_steps=request.num_leapfrog_steps,
            seed=request.seed,
            tuned_step_size=1.0 / (request.num_leapfrog_steps + 10.0),
            lineage=request.lineage,
        )

    def screen(request):
        return FixedMetricScreenOutcome(
            num_leapfrog_steps=request.num_leapfrog_steps,
            replication_index=request.replication_index,
            seed=request.seed,
            tuned_step_size=request.tuned_step_size,
            lineage=request.lineage,
            acceptance_evidence_payload=_acceptance_payload(
                request,
                worker_request.acceptance_policy,
                candidate_invalid=reject_screen,
            ),
        )

    return FixedMetricCandidateRunners(tune_runner=tune, screen_runner=screen)


def deterministic_worker_factory(
    request: FixedMetricCandidateWorkerRequest,
) -> FixedMetricCandidateRunners:
    return _runners(request)


def candidate_rejection_worker_factory(
    request: FixedMetricCandidateWorkerRequest,
) -> FixedMetricCandidateRunners:
    return _runners(
        request,
        reject_tune=request.num_leapfrog_steps == 9,
        reject_screen=request.num_leapfrog_steps == 13,
    )


def out_of_order_worker_factory(
    request: FixedMetricCandidateWorkerRequest,
) -> FixedMetricCandidateRunners:
    return _runners(
        request,
        tune_delay_seconds=0.25 if request.num_leapfrog_steps == 3 else 0.0,
    )


def bootstrap_failure_worker_factory(
    request: FixedMetricCandidateWorkerRequest,
) -> FixedMetricCandidateRunners:
    del request
    raise RuntimeError("private factory detail must not escape the child")


def target_veto_worker_factory(
    request: FixedMetricCandidateWorkerRequest,
) -> FixedMetricCandidateRunners:
    return _runners(request, target_veto=request.num_leapfrog_steps == 9)


def resource_closeout_worker_factory(
    request: FixedMetricCandidateWorkerRequest,
) -> FixedMetricCandidateRunners:
    return _runners(request, resource_closeout=request.num_leapfrog_steps == 9)
