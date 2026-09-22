"""Engineering regressions for intra-candidate acceptance disagreement."""

import numpy as np
import pytest

from bayesfilter.inference import hmc_kernel_tuning as tuning
from tests.test_hmc_kernel_tuning_outer_loop import (
    HMCAcceptancePolicy, _ToyGaussianAdapter, _loop_config,
    _phase4_direct_outcome, _phase7_direct_fixture,
    _sequential_verification_diagnostics, _tiny_budget_factory,
    evaluate_hmc_acceptance_evidence,
)


def conflicting_chain_diagnostics():
    diagnostics = dict(_sequential_verification_diagnostics(0.70))
    samples = (
        np.arange(64)[:, None, None] * np.array([1.0, -0.5])[None, None, :]
        + np.arange(4)[None, :, None]
    )
    probabilities = np.tile([0.55, 0.55, 0.85, 0.85], (64, 1))
    evidence = evaluate_hmc_acceptance_evidence(
        samples=samples, log_accept_ratio=np.log(probabilities),
        is_accepted=np.ones((64, 4), dtype=bool), policy=HMCAcceptancePolicy(),
        native_divergence_status="not_exposed_by_kernel",
    ).payload()
    assert evidence["acceptance_decision"] == "inconclusive_conflict"
    diagnostics.update(acceptance_evidence=evidence, passed=False)
    return diagnostics


@pytest.mark.parametrize("second", ["conflict", "lower", "passed", "shared_veto"])
def test_queue_preserves_within_candidate_conflict_and_terminal_precedence(second):
    geometry, _bootstrap, windowed, fixed, _handoff, _selected, _alternative = (
        _phase7_direct_fixture()
    )
    cases = iter(("conflict", second))

    def verifier(*, candidate_identity, **_kwargs):
        case = next(cases)
        diagnostics = (
            conflicting_chain_diagnostics() if case == "conflict"
            else _sequential_verification_diagnostics(0.60 if case == "lower" else 0.70)
        )
        return _phase4_direct_outcome(
            fixed=fixed, windowed=windowed, identity=candidate_identity,
            sequential_diagnostics=diagnostics,
            shared_error=RuntimeError("shared fixture failure") if case == "shared_veto" else None,
        )

    result = tuning._run_phase7_direct_candidate_queue(
        adapter=_ToyGaussianAdapter(), geometry=geometry, windowed_stage=windowed,
        fixed_mass_step_stage=fixed, config=_loop_config(max_attempts=1),
        budget_policy=_tiny_budget_factory(geometry.target_dimension, 0),
        attempt_index=0, target_scope="kernel_fixed_mass_step_toy_gaussian",
        verification_callback=None, checkpoint_writer_config=None,
        progress_callback=None, run_full_chain=lambda *_args, **_kwargs: None,
        verification_runner=verifier,
    )
    assert result.started_count == 2
    assert result.candidate_results[0]["acceptance_evidence"]["acceptance_decision"] == "inconclusive_conflict"
    if second == "passed":
        assert result.final_status == "passed"
        assert not result.repair_direction_conflict
    elif second == "shared_veto":
        assert result.final_status == "hard_veto"
        assert not result.repair_direction_conflict
    else:
        assert result.final_status == "repair_or_retry"
        assert result.diagnostic_role == "verification_acceptance_conflict"
        assert result.repair_direction_conflict
        assert len(result.repair_directions) <= 1
        assert "verification_acceptance_inconclusive_conflict" in result.repair_triggers
