"""MAMS proposer tests: engineering evidence, never posterior claims."""
from dataclasses import replace
import json
import math

import pytest

from bayesfilter.inference.hmc_mams_trajectory_proposals import (
    MAMSPilotConfig, _autocorrelation_summary, _bracketed_epsilon,
    propose_mams_trajectory_candidates,
)
from bayesfilter.inference.hmc_transition_spec import FrozenTransitionSpec
from tests.test_modern_hmc_pipeline import setup


def config_for(binding, **changes):
    config = MAMSPilotConfig(
        initial_epsilon=.2, initial_mean_duration=1.,
        epsilon_domain=binding.scope.epsilon_domain, max_steps_cap=20,
        epsilon_adaptation_windows=1, probe_transitions=4,
        fixed_settling_transitions=4,
        fixed_window_transitions=24, fixed_window_count=2,
        max_autocorrelation_lag=4, target_accept_probability=.9,
        log_epsilon_gain=.2, source_length_coefficient=.3,
        window_agreement_factor=3., duration_factors=(1., 2.),
        epsilon_factors=(1.,), max_requested_duration=2.,
        max_candidates=4, max_pilot_gradient_work=50000,
        max_pilot_wall_seconds=120., frozen_coordinate_scale=(1., 1.),
        frozen_projection_directions=((1., 1.),),
        seed=(310, 77))
    return replace(config, **changes)


def test_mams_pilot_uses_separate_continuous_fixed_policy_windows_and_fresh_tuner(tmp_path):
    from bayesfilter.inference import tune_hmc_kernel
    binding, search = setup(family="mams")
    result = propose_mams_trajectory_candidates(binding, config_for(binding),
                                                output_dir=tmp_path/"pilot")
    progress = json.loads((tmp_path/"pilot/pilot-progress.json").read_text())
    persisted = json.loads((tmp_path/"pilot/pilot-result.json").read_text())
    assert progress["status"] == "completed"
    assert progress["reserved_gradient_work"] == result["reserved_gradient_work"]
    assert progress["actual_gradient_work"] == result["actual_gradient_work"]
    assert all(attempt["status"] == "completed" for attempt in progress["attempts"])
    assert persisted["purpose_digest"] == result["purpose_digest"]
    assert result["actual_gradient_work"] <= result["reserved_gradient_work"]
    assert not result["tuning_artifact_authority"]
    assert not result["posterior_convergence_authority"]
    assert result["fresh_measurement_and_verification_required"]
    assert result["reserved_gradient_work"] <= 50000
    assert not binding._evidence
    assert len(result["duration_rows"]) >= 2
    assert any(origin == "declared_neighborhood" for _, origin in result["duration_hypotheses"])
    seeds = [tuple(r["seed"]) for r in result["seeds"]]
    assert len(seeds) == len(set(seeds))
    for row in result["duration_rows"]:
        windows = row["fixed_windows"]
        assert row["fixed_settling"]["excluded_from_autocorrelation"]
        assert len(windows) == 2
        assert windows[0]["epsilon"] == windows[1]["epsilon"]
        assert windows[0]["max_steps"] == windows[1]["max_steps"]
        for window in windows:
            diagnostic = window["diagnostics"]
            assert diagnostic["draws_per_chain"] == 24
            assert diagnostic["observable_roles"] == [
                "coordinate", "coordinate", "frozen_projection",
                "coordinate_square", "coordinate_square",
                "frozen_projection_square", "radius_square"]
            assert not diagnostic["posterior_convergence_authority"]
    for payload in result["candidates"]:
        spec = FrozenTransitionSpec.from_payload(payload)
        assert spec.profile.family == "mams"
        assert spec.max_steps <= 20
    pairs = result["epsilon_by_l"]
    assert pairs
    search = replace(search, primary_l_grid=tuple(k for k, _ in pairs),
                     epsilon_by_l=pairs, max_candidates=len(result["candidates"]),
                     total_budget_units=max(10, len(result["candidates"])*3),
                     replicated_acceptance_policy=replace(
                         search.replicated_acceptance_policy,
                         max_candidates=len(result["candidates"])))
    run = tune_hmc_kernel(adapter=binding._base_adapter,
                          initial_position=binding.initial_active_state,
                          config=search, candidate_set_adapter=binding.typed_adapter,
                          output_dir=tmp_path/"fresh")
    assert len(run.result.candidates) == len(result["candidates"])
    assert all(work.stage in {"measurement", "verification"}
               for work in run.result.work_items)
    qualification_seeds = [tuple(chunk["seed"]) for evidence in binding._evidence.values()
                           for chunk in evidence["chunks"]]
    assert qualification_seeds, "stream separation must check executed trial streams"
    assert set(seeds).isdisjoint(qualification_seeds)


def test_scoped_budget_stops_pilot_before_any_native_call(tmp_path):
    from bayesfilter.inference.hmc_mams_trajectory_proposals import MAMSPilotBudgetExceeded
    from bayesfilter.runtime.execution_budget import execution_budget
    binding, _ = setup(family="mams")
    with execution_budget(check=lambda: False, replace_wall_time_limit=True):
        with pytest.raises(MAMSPilotBudgetExceeded):
            propose_mams_trajectory_candidates(binding, config_for(binding), output_dir=tmp_path/"pilot")
    progress = json.loads((tmp_path/"pilot/pilot-progress.json").read_text())
    assert progress["attempts"] == []
    assert progress["actual_gradient_work"] == 0


def test_mams_pilot_rejects_hmc_and_infeasible_config_before_sampling():
    binding, _ = setup(family="hmc")
    with pytest.raises(ValueError, match="MAMS pilot requires"):
        propose_mams_trajectory_candidates(binding, config_for(binding))
    binding, _ = setup(family="mams")
    with pytest.raises(ValueError, match="count/domain|step-count cap"):
        propose_mams_trajectory_candidates(binding,
            config_for(binding, max_steps_cap=3))
    assert not binding._evidence


def test_fixed_window_diagnostic_detects_collapsed_and_slow_coordinate():
    import tensorflow as tf
    n = 40
    collapsed = tf.zeros([n, 4, 2], tf.float64)
    result = _autocorrelation_summary(collapsed, frozen_scale=(1., 1.), max_lag=4)
    assert result["source_harmonic_coordinate_iact"] is None
    assert result["constant_or_nonfinite_observable"] == [True]*5
    # Deliberately smooth one coordinate; alternating the other creates
    # apparent movement without comparable long-lag exploration.
    t = tf.cast(tf.range(n), tf.float64)
    smooth = tf.sin(t/12.)
    alternate = tf.where(tf.cast(tf.range(n)%2, tf.bool),
                         tf.constant(1., tf.float64), tf.constant(-1., tf.float64))
    chain = tf.stack((alternate, smooth), axis=-1)
    samples = tf.stack((chain, chain + .1, chain + .2, chain + .3), axis=1)
    result = _autocorrelation_summary(samples, frozen_scale=(1., 1.), max_lag=4)
    assert result["lag1_squared_displacement_by_coordinate"][0] > (
        100*result["lag1_squared_displacement_by_coordinate"][1])
    assert result["worst_coordinate_lag1_squared_displacement"] == pytest.approx(
        result["lag1_squared_displacement_by_coordinate"][1])
    assert result["observable_roles"].count("coordinate_square") == 2


def test_pilot_budget_stop_preserves_completed_attempt_and_partial_duration(tmp_path):
    binding, _ = setup(family="mams")
    config = config_for(binding, max_pilot_gradient_work=320)
    with pytest.raises(RuntimeError, match="gradient-work budget exhausted"):
        propose_mams_trajectory_candidates(binding, config, output_dir=tmp_path/"stopped")
    progress = json.loads((tmp_path/"stopped/pilot-progress.json").read_text())
    assert progress["status"] == "gradient_budget_exhausted"
    assert progress["reserved_gradient_work"] == 320
    assert len(progress["attempts"]) == 1
    assert progress["attempts"][0]["status"] == "completed"
    assert len(progress["active_duration"]["adaptation"]) == 1
    assert not (tmp_path/"stopped/pilot-result.json").exists()


def test_pilot_progress_directory_cannot_overwrite_an_attempt(tmp_path):
    binding, _ = setup(family="mams")
    occupied = tmp_path/"pilot"
    occupied.mkdir()
    with pytest.raises(FileExistsError):
        propose_mams_trajectory_candidates(binding, config_for(binding), output_dir=occupied)
    assert not binding._evidence


def test_epsilon_bracket_uses_geometric_middle_after_sign_change():
    domain = (.01, 2.)
    epsilon, low, high, proposed, hit = _bracketed_epsilon(
        .5, .99, .9, None, None, math.log(2), domain)
    assert (epsilon, low, high, proposed, hit) == (1., .5, None, 1., False)
    epsilon, low, high, proposed, hit = _bracketed_epsilon(
        epsilon, .2, .9, low, high, math.log(2), domain)
    assert epsilon == pytest.approx(math.sqrt(.5))
    assert (low, high, hit) == (.5, 1., False)


def test_fixed_window_correlation_matches_independent_scalar_calculation():
    import tensorflow as tf
    from bayesfilter.inference.hmc_pilot_diagnostics import pilot_diagnostic_program
    n = 80
    series = [math.sin(.63*t) + .24*math.sin(.13*t) for t in range(n)]
    samples = tf.constant(
        [[[series[t]+.11*chain] for chain in range(4)] for t in range(n)], tf.float64)
    result = _autocorrelation_summary(samples, frozen_scale=(1.,), max_lag=8)
    raw = pilot_diagnostic_program((n, 4, 1), 0, 8)(
        samples, tf.constant([1.], tf.float64), tf.zeros((0, 1), tf.float64))
    centered = [value - math.fsum(series)/n for value in series]
    variance = math.fsum(x*x for x in centered)/n
    for lag in range(9):
        expected = (math.fsum(centered[t]*centered[t+lag]
                              for t in range(n-lag))/(n-lag)/variance)
        assert float(raw["rho"][lag,0,0]) == pytest.approx(expected, abs=1e-10)
    assert result["observable_roles"] == ["coordinate", "coordinate_square", "radius_square"]
    assert result["posterior_convergence_authority"] is False


def test_pilot_recovery_reuses_completed_windows_and_charges_lost_work(tmp_path, monkeypatch):
    binding, _ = setup(family="mams")
    config = config_for(binding)
    original = binding._run
    calls = []
    def interrupted(*args):
        calls.append(args[-1])
        if len(calls) == 3:
            raise RuntimeError("injected infrastructure interruption")
        return original(*args)
    monkeypatch.setattr(binding, "_run", interrupted)
    with pytest.raises(RuntimeError, match="injected infrastructure"):
        propose_mams_trajectory_candidates(binding, config, output_dir=tmp_path/"attempt-1")
    failed = (tmp_path/"attempt-1/pilot-progress.json").read_bytes()
    before = json.loads(failed)
    assert before["status"] == "failed_call"
    assert len(before["attempts"]) == 3
    retry_seeds = []
    def retry(*args):
        retry_seeds.append(args[-1])
        return original(*args)
    monkeypatch.setattr(binding, "_run", retry)
    result = propose_mams_trajectory_candidates(binding, config,
        output_dir=tmp_path/"attempt-2", resume_from=tmp_path/"attempt-1")
    assert retry_seeds[0] == calls[-1]
    assert set(calls[:2]).isdisjoint(retry_seeds)
    progress = json.loads((tmp_path/"attempt-2/pilot-progress.json").read_text())
    assert sum(a["replayed"] for a in progress["attempts"]) == 2
    assert result["reserved_gradient_work"] == sum(a["reserved_gradient_work"]
                                                       for a in progress["attempts"])
    assert (tmp_path/"attempt-1/pilot-progress.json").read_bytes() == failed
    with pytest.raises(ValueError, match="identity mismatch"):
        propose_mams_trajectory_candidates(binding, replace(config, seed=(31, 78)),
            output_dir=tmp_path/"invalid", resume_from=tmp_path/"attempt-1")
    assert not (tmp_path/"invalid").exists()


def test_lag_limited_and_nonpositive_estimates_stay_unavailable():
    import tensorflow as tf
    t = tf.cast(tf.range(80), tf.float64)
    smooth = tf.broadcast_to(t[:, None, None], [80, 4, 1])
    result = _autocorrelation_summary(smooth, frozen_scale=(1.,), max_lag=4)
    assert all(result["iact_lag_limited"])
    assert result["source_harmonic_coordinate_iact"] is None
    assert result["worst_observable_iact"] is None
    assert result["iact_initial_positive_pairs"] == [None]*3


def test_adaptation_recovers_proposal_veto_without_hiding_shared_defect(tmp_path, monkeypatch):
    binding, _ = setup(family="mams")
    health = binding.health_failures
    calls = [0]
    def once(*args):
        calls[0] += 1
        return ("nonfinite_proposed_target_log_prob",) if calls[0] == 1 else health(*args)
    monkeypatch.setattr(binding, "health_failures", once)
    result = propose_mams_trajectory_candidates(binding, config_for(binding),
                                                output_dir=tmp_path/"backoff")
    first = result["duration_rows"][0]["adaptation"][0]
    assert first["acceptance"] is None
    assert first["bounded_next_epsilon"] < first["epsilon"]
    assert first["state_reused"]
    monkeypatch.setattr(binding, "health_failures", lambda *_: ("metropolis_state_mismatch",))
    with pytest.raises(ValueError, match="metropolis_state_mismatch"):
        propose_mams_trajectory_candidates(binding, config_for(binding), output_dir=tmp_path/"invalid-state")


def test_displacement_counts_first_transition_and_rejection_zeros():
    import tensorflow as tf
    from bayesfilter.inference.hmc_pilot_diagnostics import summarize_pilot_displacement
    initial = tf.constant([[0., 0.]], tf.float64)
    samples = tf.constant([[[2., 0.]], [[2., 0.]]], tf.float64)
    proposals = tf.constant([[[2., 0.]], [[6., 3.]]], tf.float64)
    result = summarize_pilot_displacement(initial, samples, proposals, frozen_scale=(2., 1.))
    assert result["proposal_squared_displacement_by_coordinate"] == pytest.approx([2.5, 4.5])
    assert result["realized_squared_displacement_by_coordinate"] == pytest.approx([.5, 0.])


def test_sub_resolution_variance_is_not_a_reliable_iact():
    import tensorflow as tf
    samples = tf.reshape(tf.cast(tf.range(40), tf.float64)*1e-20, [40, 1, 1])
    result = _autocorrelation_summary(samples, frozen_scale=(1.,), max_lag=4)
    assert result["source_harmonic_coordinate_iact"] is None
    assert result["usable_observable_iact"] == [False]*3


def test_epsilon_bracket_extreme_gain_stays_finite_and_in_domain():
    for acceptance, expected in [(1., 2.), (0., .01)]:
        result = _bracketed_epsilon(.5, acceptance, .9, None, None, 1000., (.01, 2.))
        assert result[0] == pytest.approx(expected)
        assert result[-1] is True
        assert result[3] is None or math.isfinite(result[3])


def test_processing_failure_is_terminal_and_existing_attempt_is_untouched(tmp_path, monkeypatch):
    import bayesfilter.inference.hmc_mams_trajectory_proposals as module
    binding, _ = setup(family="mams")
    def broken(*args, **kwargs):
        raise ValueError("injected diagnostic failure")
    monkeypatch.setattr(module, "_autocorrelation_summary", broken)
    output = tmp_path/"processing"
    with pytest.raises(ValueError, match="injected diagnostic"):
        propose_mams_trajectory_candidates(binding, config_for(binding), output_dir=output)
    before = (output/"pilot-progress.json").read_bytes()
    assert json.loads(before)["status"] == "failed_processing"
    with pytest.raises(FileExistsError):
        propose_mams_trajectory_candidates(binding, config_for(binding), output_dir=output)
    assert (output/"pilot-progress.json").read_bytes() == before


def test_recovery_accounts_committed_call_missing_from_progress(tmp_path, monkeypatch):
    from bayesfilter.runtime.durable_tensor_checkpoint import DurableTensorCheckpoint
    binding, _ = setup(family="mams")
    config = config_for(binding, duration_factors=(1.,))
    original = DurableTensorCheckpoint.run
    calls = []
    def fail_after_commit(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        calls.append(args[0])
        if len(calls) == 1:
            raise KeyboardInterrupt("simulated abrupt process loss after durable commit")
        return result
    monkeypatch.setattr(DurableTensorCheckpoint, "run", fail_after_commit)
    with pytest.raises(KeyboardInterrupt):
        propose_mams_trajectory_candidates(binding, config, output_dir=tmp_path/"first")
    monkeypatch.setattr(DurableTensorCheckpoint, "run", original)
    result = propose_mams_trajectory_candidates(binding, config, output_dir=tmp_path/"second",
                                                resume_from=tmp_path/"first")
    progress = json.loads((tmp_path/"second/pilot-progress.json").read_text())
    assert progress["attempts"][0]["status"] == "recovered_committed_call"
    assert progress["attempts"][0]["actual_gradient_work"] > 0
    assert result["actual_gradient_work"] == sum(a.get("actual_gradient_work", 0)
                                                     for a in progress["attempts"])
