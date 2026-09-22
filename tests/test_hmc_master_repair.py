"""Counterexamples from the completed inference-validation campaign."""
import json
from types import SimpleNamespace

import pytest

from bayesfilter.inference import HMCKernelTuningConfig, tune_hmc_kernel
from bayesfilter.inference.hmc_preparation import (
    HMCPreparationFailure, HMCPreparationProgress, expanded_preparation_bound,
    prepare_operational_windowed_mass_handoff,
)
from tests.test_hmc_candidate_set_execution import GaussianTarget


@pytest.mark.parametrize("preset", ["smoke", "standard", "diagnostic", "diagnostic_plus", "serious"])
def test_public_presets_reach_preparation(preset, monkeypatch):
    def reached(**kwargs):
        raise RuntimeError("reached preparation")
    monkeypatch.setattr("bayesfilter.inference.hmc_preparation.prepare_operational_windowed_mass_handoff", reached)
    cfg = getattr(HMCKernelTuningConfig, preset)(use_xla=False, target_scope="candidate-bridge-test")
    with pytest.raises(RuntimeError, match="reached preparation"):
        tune_hmc_kernel(adapter=GaussianTarget(), initial_position=[0., 0.], config=cfg)


@pytest.mark.parametrize("steps", [-1, True, 1.5, 6])
def test_expansion_config_rejects_invalid_steps(steps):
    with pytest.raises(ValueError, match="expansion_steps"):
        HMCKernelTuningConfig(candidate_search_bound_expansion_steps=steps)


def test_expansion_is_finite_explicit_and_serialized():
    assert expanded_preparation_bound(.6, factor=2., steps=0) == .6
    assert expanded_preparation_bound(.6, factor=2., steps=1) == 1.2
    with pytest.raises(ValueError, match="finite"):
        expanded_preparation_bound(1e308, factor=2., steps=2)
    with pytest.raises(ValueError, match="windowed_adaptive"):
        HMCKernelTuningConfig(mass_policy="fixed_identity", candidate_search_bound_expansion_steps=1)
    cfg = HMCKernelTuningConfig(candidate_search_bound_expansion_steps=1)
    assert cfg.payload()["candidate_search_bound_expansion_steps"] == 1


def test_preparation_veto_causes_persist_before_exception(tmp_path, monkeypatch):
    import bayesfilter.inference.hmc_kernel_tuning as legacy
    geometry = SimpleNamespace(artifact_hash="geometry")
    failed_round = legacy.HMCBootstrapRepairRound(
        round_index=0, seed=(1, 2), step_size=.5, num_leapfrog_steps=3,
        unclamped_num_leapfrog_steps=3, target_trajectory_length=1.5,
        leapfrog_clamped=False, clamp_direction=None, classification="hard_veto",
        diagnostic_role="bootstrap fixed-kernel screen only", screen_config_payload=None,
        hard_vetoes=("nonfinite_target_score",),
        diagnostics={"exception_type": "InvalidArgumentError", "exception_message": "target failed"})
    bootstrap = SimpleNamespace(artifact_hash="bootstrap", final_status="hard_veto",
        rounds=(failed_round,), payload=lambda: {"final_status": "hard_veto",
                                                "rounds": [failed_round.payload()]})
    monkeypatch.setattr("bayesfilter.inference.hmc_geometry.initialize_hmc_kernel_geometry", lambda **kw: geometry)
    monkeypatch.setattr("bayesfilter.inference.hmc_configuration._public_geometry_config", lambda cfg: None)
    monkeypatch.setattr("bayesfilter.inference.hmc_configuration._public_bootstrap_config", lambda *a, **kw: None)
    monkeypatch.setattr("bayesfilter.inference.hmc_bootstrap.run_hmc_bootstrap_screen", lambda **kw: bootstrap)
    with pytest.raises(HMCPreparationFailure):
        with HMCPreparationProgress(tmp_path) as progress:
            prepare_operational_windowed_mass_handoff(adapter=GaussianTarget(), initial_position=[0., 0.],
                config=HMCKernelTuningConfig.smoke(target_scope="test"), progress_callback=progress.phase)
    record = json.loads((tmp_path / "preparation_progress.json").read_text())
    assert record["failure"]["details"]["rounds"][0]["final_status"] == "hard_veto"
    assert record["failure"]["details"]["rounds"][0]["hard_vetoes"] == ["nonfinite_target_score"]
    assert record["failure"]["details"]["rounds"][0]["diagnostics"]["exception_message"] == "target failed"
    assert record["events"][-1]["phase"] == "bootstrap_failed"


def test_windowed_preparation_failure_preserves_stage_diagnostics(tmp_path, monkeypatch):
    import bayesfilter.inference.hmc_kernel_tuning as legacy
    monkeypatch.setattr("bayesfilter.inference.hmc_geometry.initialize_hmc_kernel_geometry", lambda **kw:
        SimpleNamespace(artifact_hash="geometry", target_dimension=2))
    monkeypatch.setattr("bayesfilter.inference.hmc_configuration._public_geometry_config", lambda cfg: None)
    monkeypatch.setattr("bayesfilter.inference.hmc_configuration._public_bootstrap_config", lambda *a, **kw: None)
    monkeypatch.setattr("bayesfilter.inference.hmc_bootstrap.run_hmc_bootstrap_screen", lambda **kw:
        SimpleNamespace(artifact_hash="bootstrap", final_status="passed",
                        payload=lambda: {"final_status": "passed"}))
    monkeypatch.setattr("bayesfilter.inference.hmc_mass_adaptation._bootstrap_preflight_passed", lambda result: True)
    monkeypatch.setattr("bayesfilter.inference.hmc_configuration._HMCAttemptBudgetPolicy", SimpleNamespace)
    monkeypatch.setattr("bayesfilter.inference.hmc_configuration._public_budget_policy_factory", lambda *a, **kw:
        lambda *args: SimpleNamespace(phase4_warmup_steps=32, payload=lambda: {}))
    monkeypatch.setattr("bayesfilter.inference.hmc_configuration._public_loop_config", lambda cfg: None)
    monkeypatch.setattr("bayesfilter.inference.hmc_configuration._phase7_windowed_stage_config", lambda *a, **kw: None)
    monkeypatch.setattr("bayesfilter.inference.hmc_mass_adaptation.run_hmc_windowed_mass_stage", lambda **kw:
        SimpleNamespace(passed=False, operational_warmup_result=None, final_status="hard_veto",
            hard_vetoes=("metric_update_failed",),
            diagnostics={"exception_type": "ValueError", "exception_message": "invalid covariance"},
            payload=lambda: {"final_status": "hard_veto", "hard_vetoes": ["metric_update_failed"]}))
    with pytest.raises(HMCPreparationFailure):
        with HMCPreparationProgress(tmp_path) as progress:
            prepare_operational_windowed_mass_handoff(adapter=GaussianTarget(), initial_position=[0., 0.],
                config=HMCKernelTuningConfig.smoke(target_scope="test"), progress_callback=progress.phase)
    record = json.loads((tmp_path / "preparation_progress.json").read_text())
    assert record["events"][-1]["phase"] == "windowed_mass_failed"
    assert record["failure"]["details"]["hard_vetoes"] == ["metric_update_failed"]
    assert record["failure"]["details"]["diagnostics"]["exception_message"] == "invalid covariance"


def test_nonfinite_preparation_diagnostics_do_not_mask_original_failure(tmp_path):
    details = {"stage": "windowed_mass", "hard_vetoes": ("nonfinite_target_score",),
               "diagnostics": {"raw": [float("nan"), float("inf"), -float("inf"), 1.5]}}
    failure = HMCPreparationFailure("original numerical failure", details=details)
    with pytest.raises(HMCPreparationFailure) as raised:
        with HMCPreparationProgress(tmp_path) as progress:
            progress.phase("windowed_mass_failed", details)
            raise failure
    assert raised.value is failure
    text = (tmp_path / "preparation_progress.json").read_text()
    def reject_nonfinite(value):
        raise AssertionError(f"Nonstandard JSON constant: {value}")
    record = json.loads(text, parse_constant=reject_nonfinite)
    assert record["status"] == "failed"
    assert record["failure"]["type"] == "HMCPreparationFailure"
    assert record["failure"]["reason"] == "original numerical failure"
    expected = [{"nonfinite": "nan"}, {"nonfinite": "inf"}, {"nonfinite": "-inf"}, 1.5]
    assert record["failure"]["details"]["diagnostics"]["raw"] == expected
    assert record["events"][-1]["details"]["diagnostics"]["raw"] == expected
    assert record["failure"]["details"]["hard_vetoes"] == ["nonfinite_target_score"]


@pytest.mark.parametrize("epsilon,cap,reason", [(.5, 1, "max_candidates"), (2., 4, "epsilon_domain")])
def test_repair_reports_the_actual_search_limit(epsilon, cap, reason):
    from tests.test_hmc_candidate_set_tuning import _scope
    from bayesfilter.inference.hmc_candidate_set_tuning import HMCControllerConfig, HMCTuningCandidateSetController
    result = HMCTuningCandidateSetController(_scope(), HMCControllerConfig(
        primary_l_grid=(3,), initial_epsilon=epsilon, max_candidates=cap)).run(
            lambda *args: {"decision": "repair_step_higher", "acceptance": .95})
    limits = [row for row in result.accounting_events if row["event"] == "repair_limit_exhausted"]
    assert limits[0]["reason"] == reason


def test_refinement_cap_counts_only_eligible_unadmitted_pairs():
    from tests.test_hmc_candidate_set_tuning import _scope, _pass
    from bayesfilter.inference.hmc_candidate_set_tuning import HMCControllerConfig, HMCTuningCandidateSetController
    result = HMCTuningCandidateSetController(_scope(), HMCControllerConfig(
        primary_l_grid=(3,), initial_epsilon=.5, max_candidates=2, refinement_rounds=1,
        epsilon_refinement_factors=(.8, 1.25, 100.), refinement_l_grid=(3, 5))).run(_pass)
    # .4 was admitted; .5 duplicates the parent; 50 is outside the domain.
    # Only (3, .625) and (5, .5) remain eligible at this round's cap.
    limits = [row for row in result.accounting_events if row["event"] == "refinement_cap_reached"]
    assert limits[0]["remaining_proposals"] == 2


def test_malformed_observation_does_not_poison_numerical_checkpoint(tmp_path):
    from tests.test_hmc_candidate_set_execution import make_binding
    from bayesfilter.inference.hmc_candidate_set_tuning import HMCControllerConfig, HMCTuningCandidateSetController
    from bayesfilter.inference.hmc_candidate_set_checkpoint import (
        write_numerical_tuning_checkpoint, load_numerical_tuning_checkpoint,
    )
    binding = make_binding()
    controller = HMCTuningCandidateSetController(binding.scope, HMCControllerConfig(
        primary_l_grid=(3,), initial_epsilon=1.3, evidence_rungs=(1,)))
    def malformed(work, candidate):
        observation = dict(binding.observe(work, candidate))
        observation["draw_range"] = ("invalid", 1)
        return observation
    with pytest.raises(ValueError):
        controller.run(malformed, checkpoint=lambda result:
            write_numerical_tuning_checkpoint(binding, result, tmp_path))
    assert not controller.result().observations
    restored, resumed = load_numerical_tuning_checkpoint(tmp_path / "tuning_checkpoint.json", adapter=GaussianTarget())
    result = resumed.run(restored.observe, checkpoint=lambda result:
        write_numerical_tuning_checkpoint(restored, result, tmp_path))
    assert result.verified_candidate_ids
    assert result.budget_used_units == 3  # The malformed attempt remains charged.
    _, reloaded = load_numerical_tuning_checkpoint(tmp_path / "tuning_checkpoint.json", adapter=GaussianTarget())
    assert reloaded.result().verified_candidate_ids == result.verified_candidate_ids


def test_later_shared_invalidity_disables_an_earlier_live_member(monkeypatch, tmp_path):
    import tensorflow as tf
    from tests.test_hmc_candidate_set_execution import make_binding
    from bayesfilter.inference import build_retained_bound_hmc_archive_runner_from_candidate_set_result
    from bayesfilter.inference.hmc_candidate_set_tuning import HMCControllerConfig, HMCTuningCandidateSetController
    binding = make_binding()
    controller = HMCTuningCandidateSetController(binding.scope, HMCControllerConfig(
        primary_l_grid=(3,), initial_epsilon=1.3, evidence_rungs=(1,)))
    earlier = controller.run(binding.observe)
    member = build_retained_bound_hmc_archive_runner_from_candidate_set_result(
        candidate_set_result=earlier, candidate_id=earlier.verified_candidate_ids[0], retained_binding=binding)
    controller.add_exploration_candidate(5, 1.3)
    original = binding._run
    def invalid_transition(*args):
        value = original(*args)
        return SimpleNamespace(samples=value.samples + tf.constant(.01, value.samples.dtype),
            trace=value.trace, metadata=value.metadata)
    monkeypatch.setattr(binding, "_run", invalid_transition)
    later = controller.run(binding.observe)
    assert later.completion_status == "shared_invalidity"
    assert later.observations[-1]["observation"]["evidence_validity"] == "shared_execution_invalid"
    with pytest.raises(ValueError, match="shared-invalid"):
        member.export(tmp_path / "stale-member.json")
    with pytest.raises(ValueError, match="shared-invalid"):
        build_retained_bound_hmc_archive_runner_from_candidate_set_result(
            candidate_set_result=earlier, candidate_id=earlier.verified_candidate_ids[0], retained_binding=binding)
