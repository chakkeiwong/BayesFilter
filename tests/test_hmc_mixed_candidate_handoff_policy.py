from __future__ import annotations

import os
from dataclasses import replace
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import pytest

from bayesfilter.inference.hmc_kernel_selection import (
    FixedTrajectoryCandidate,
    FixedTrajectoryCandidateResult,
    FixedTrajectoryReplication,
    _candidate_execution_contract_signature,
    _candidate_handoff_policy,
    candidate_handoff_policy_payload,
    paired_candidate_seed,
    private_start_bank_content_signature,
    run_bounded_operational_fixed_trajectory_selection,
    select_fixed_trajectory_representative,
)
from bayesfilter.inference.hmc_kernel_tuning import HMCKernelTuningConfig
from bayesfilter.inference.hmc_verification import (
    HMCAcceptancePolicy,
    evaluate_hmc_acceptance_evidence,
)


def _evidence(probability: float, *, draw_count: int = 64, veto: bool = False):
    draws = np.arange(draw_count, dtype=float)[:, None, None]
    chains = np.arange(4, dtype=float)[None, :, None]
    samples = draws + chains
    values = np.full((draw_count, 4), probability)
    return evaluate_hmc_acceptance_evidence(
        samples=np.zeros_like(samples) if veto else samples,
        log_accept_ratio=np.full_like(values, np.nan) if veto else np.log(values),
        is_accepted=np.zeros_like(values, dtype=bool) if veto else np.ones_like(values, dtype=bool),
        policy=HMCAcceptancePolicy(),
    )


def _inconclusive(draw_count: int = 64):
    values = np.repeat((0.60, 0.80, 0.60, 0.80), draw_count // 4)
    return evaluate_hmc_acceptance_evidence(
        samples=np.arange(draw_count, dtype=float)[:, None, None]
        + np.arange(4, dtype=float)[None, :, None],
        log_accept_ratio=np.log(np.repeat(values[:, None], 4, axis=1)),
        is_accepted=np.ones((draw_count, 4), dtype=bool),
        policy=HMCAcceptancePolicy(),
    )


def _result(
    leapfrog: int,
    records,
    *,
    root_seed=(20260722, 800),
    start_bank_signature="bank",
    execution_kwargs=None,
):
    candidate = FixedTrajectoryCandidate(
        anchor_l=10,
        num_leapfrog_steps=leapfrog,
        max_leapfrog_steps=64,
        coordinate_signature="coordinate",
        metric_signature="metric",
        start_bank_signature=start_bank_signature,
    )
    if not isinstance(records, tuple):
        records = (records,) * 3
    return FixedTrajectoryCandidateResult(
        candidate=candidate,
        replications=tuple(
            FixedTrajectoryReplication(
                candidate=candidate,
                replication_index=index,
                seed=(seed := paired_candidate_seed(
                    root_seed,
                    candidate_signature=candidate.signature,
                    replication_index=index,
                )),
                acceptance_evidence_payload=record.payload(),
                execution_contract_signature=(
                    _candidate_execution_contract_signature(
                        candidate=candidate,
                        replication_index=index,
                        seed=seed,
                        frozen_step_size=execution_kwargs["frozen_step_size"],
                        num_results=execution_kwargs["screen_num_results"],
                        num_burnin_steps=execution_kwargs["screen_num_burnin_steps"],
                        target_scope=execution_kwargs["target_scope"],
                        acceptance_policy=execution_kwargs["acceptance_policy"],
                        target_status_trace_policy=execution_kwargs.get(
                            "target_status_trace_policy", "none"
                        ),
                        chain_execution_mode=execution_kwargs.get(
                            "chain_execution_mode", "tf_function"
                        ),
                        use_xla=execution_kwargs.get("use_xla", False),
                    )
                    if execution_kwargs is not None
                    else None
                ),
            )
            for index, record in enumerate(records)
        ),
    )


class _FiniteAdapter:
    def log_prob_and_grad(self, theta):
        values = np.asarray(theta, dtype=float)
        return -0.5 * np.sum(np.square(values), axis=-1), -values


def _all_mapping_keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _all_mapping_keys(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _all_mapping_keys(item)


def test_policy_contract_and_strict_default_are_explicit():
    assert _candidate_handoff_policy("strict") == "strict"
    contract = candidate_handoff_policy_payload(
        "mixed_evidence_requires_fresh_verification"
    )
    assert contract["schema"] == "bayesfilter.hmc_operational_candidate_handoff_policy.v1"
    assert contract["whole_matrix_veto_precedence"] is True
    strict = select_fixed_trajectory_representative(
        (_result(10, _evidence(0.70)),), anchor_l=10
    )
    assert strict.payload()["schema"] == "bayesfilter.hmc_fixed_trajectory_selection.v4"
    assert "candidate_handoff_policy" not in strict.payload()


def test_mixed_policy_requires_serious_one_doubling():
    with pytest.raises(ValueError, match="preset='serious'"):
        HMCKernelTuningConfig.standard(
            operational_candidate_handoff_policy="mixed_evidence_requires_fresh_verification",
        )
    with pytest.raises(ValueError, match="one_doubling"):
        HMCKernelTuningConfig.serious(
            operational_candidate_handoff_policy="mixed_evidence_requires_fresh_verification",
        )
    config = HMCKernelTuningConfig.serious(
        operational_evidence_policy="one_doubling",
        operational_candidate_handoff_policy="mixed_evidence_requires_fresh_verification",
    )
    assert config.payload()["operational_candidate_handoff_policy_schema_version"] == (
        "bayesfilter.hmc_operational_candidate_handoff_policy.v1"
    )


def test_mixed_matrix_nominates_only_at_terminal_boundary_and_is_permutation_invariant():
    passed = _evidence(0.70)
    inconclusive = _inconclusive()
    results = (
        _result(5, inconclusive),
        _result(10, (passed, inconclusive, inconclusive)),
        _result(20, inconclusive),
    )
    first = select_fixed_trajectory_representative(
        results,
        anchor_l=10,
        candidate_handoff_policy="mixed_evidence_requires_fresh_verification",
        terminal_candidate_handoff=True,
    )
    second = select_fixed_trajectory_representative(
        tuple(reversed(results)),
        anchor_l=10,
        candidate_handoff_policy="mixed_evidence_requires_fresh_verification",
        terminal_candidate_handoff=True,
    )
    assert first.disposition == "representative_nominated"
    assert first.candidate_handoff_disposition == (
        "mixed_evidence_provisional_nomination"
    )
    assert first.representative.candidate.num_leapfrog_steps == 10
    assert first.signature == second.signature
    assert first.payload()["schema"] == "bayesfilter.hmc_fixed_trajectory_selection.v5"


def test_mixed_policy_rejects_all_inconclusive_and_sibling_veto():
    inconclusive = _inconclusive()
    all_inconclusive = tuple(_result(item, inconclusive) for item in (5, 10, 20))
    rejected = select_fixed_trajectory_representative(
        all_inconclusive,
        anchor_l=10,
        candidate_handoff_policy="mixed_evidence_requires_fresh_verification",
        terminal_candidate_handoff=True,
    )
    assert rejected.candidate_handoff_disposition == "mixed_evidence_rejected"
    assert rejected.representative is None

    clean = _result(10, ( _evidence(0.70), inconclusive, inconclusive))
    vetoed_peer = _result(5, _evidence(0.70, veto=True))
    blocked = select_fixed_trajectory_representative(
        (clean, vetoed_peer, _result(20, inconclusive)),
        anchor_l=10,
        candidate_handoff_policy="mixed_evidence_requires_fresh_verification",
        terminal_candidate_handoff=True,
    )
    assert blocked.candidate_handoff_disposition == "mixed_evidence_rejected"
    assert blocked.representative is None


def test_bounded_mixed_policy_records_private_ledger_and_requires_fresh_retune():
    bank = np.arange(8, dtype=float).reshape(4, 2)
    candidate_handoff_calls = []

    def selector(**kwargs):
        passed = _evidence(0.70, draw_count=256)
        inconclusive = _inconclusive(256)
        candidate_handoff_calls.append(kwargs)
        candidates = (
            _result(
                5,
                inconclusive,
                root_seed=kwargs["root_seed"],
                start_bank_signature=kwargs["private_start_bank_signature"],
                execution_kwargs=kwargs,
            ),
            _result(
                10,
                (passed, inconclusive, inconclusive),
                root_seed=kwargs["root_seed"],
                start_bank_signature=kwargs["private_start_bank_signature"],
                execution_kwargs=kwargs,
            ),
            _result(
                20,
                inconclusive,
                root_seed=kwargs["root_seed"],
                start_bank_signature=kwargs["private_start_bank_signature"],
                execution_kwargs=kwargs,
            ),
        )
        # The selector itself remains strict; terminal promotion belongs to the outer boundary.
        return select_fixed_trajectory_representative(candidates, anchor_l=10)

    class Run:
        pass

    def runner(_adapter, initial_state, config):
        result = Run()
        if config.tuning_policy.uses_dual_averaging:
            result.samples = np.zeros((4, 4, 2))
            result.trace = {
                "log_accept_ratio": np.zeros((4, 4)),
                "is_accepted": np.ones((4, 4), dtype=bool),
                "target_log_prob": np.zeros((4, 4)),
                "step_size": np.full(4, 0.125),
            }
            result.diagnostics = {"final_step_size": 0.125}
            return result
        draw = np.arange(config.num_results, dtype=float)[:, None, None]
        result.samples = draw + np.asarray(initial_state)[None, :, :]
        pattern = np.repeat(
            (0.60, 0.80, 0.60, 0.80),
            config.num_results // 4,
        )
        values = np.repeat(pattern[:, None], 4, axis=1)
        result.trace = {
            "log_accept_ratio": np.log(values),
            "is_accepted": np.ones_like(values, dtype=bool),
            "target_log_prob": np.zeros_like(values),
        }
        result.diagnostics = {}
        return result

    kwargs = {
        "adapter": _FiniteAdapter(),
        "private_start_bank": bank,
        "private_start_bank_signature": private_start_bank_content_signature(bank, "coordinate"),
        "coordinate_signature": "coordinate",
        "metric_signature": "metric",
        "anchor_l": 10,
        "max_leapfrog_steps": 64,
        "initial_step_size": 0.1,
        "root_seed": (20260722, 800),
        "target_scope": "test",
        "acceptance_policy": HMCAcceptancePolicy(),
        "max_attempts": 1,
        "screen_num_results": 256,
        "screen_num_burnin_steps": 16,
        "final_tune_adaptation_steps": 64,
        "run_full_chain": runner,
        "selector": selector,
        "evidence_extension_checkpoints": (512,),
        "candidate_handoff_policy": "mixed_evidence_requires_fresh_verification",
    }
    # This fixture intentionally uses the existing callback contract only to
    # verify ledger shape; the real runner owns target validity in production.
    result = run_bounded_operational_fixed_trajectory_selection(**kwargs)
    ledger = result.private_evidence_ledger()
    assert ledger["schema"] == "bayesfilter.hmc_fixed_trajectory_private_evidence_ledger.v1"
    assert ledger["raw_samples_exposed"] is False
    assert ledger["raw_start_bank_exposed"] is False
    assert not {
        "samples",
        "start_bank",
        "raw_start_bank",
    }.intersection(_all_mapping_keys(ledger))
