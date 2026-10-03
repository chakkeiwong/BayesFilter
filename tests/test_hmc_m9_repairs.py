"""Finite startup and rejected-interval counterexamples; CPU mechanics only."""
from dataclasses import replace
import json
import math

import pytest
import tensorflow as tf

from bayesfilter.inference import HMCKernelTuningConfig, tune_hmc_kernel
from bayesfilter.inference.hmc_bootstrap_initialization import initialize_bootstrap_step
from bayesfilter.inference.hmc_candidate_set_tuning import (
    HMCControllerConfig, HMCTuningCandidateSetController,
)
from bayesfilter.inference.hmc_preparation import HMCPreparationFailure
from tests.test_hmc_bootstrap_initialization import Gaussian, inputs
from tests.test_hmc_candidate_set_tuning import _scope


class NoStatusGaussian(Gaussian):
    supports_retained_value_score_status = False

    def target_status_telemetry(self, theta):
        raise AssertionError("absent telemetry must not be requested")

    def log_prob_and_grad(self, theta):
        return -.5 * self.precision * tf.reduce_sum(theta**2, axis=-1), -self.precision * theta


@pytest.mark.parametrize("xla", [False, True])
def test_scaled_startup_without_status_preserves_failed_probes(xla):
    adapter, geometry, config = inputs(NoStatusGaussian(precision=1.e6), xla=xla)
    config = replace(config, target_status_trace_policy="none")
    repaired = initialize_bootstrap_step(adapter=adapter, geometry=geometry, config=config)
    record = repaired.formula_report["bootstrap_initialization"]
    assert record["target_status_trace_policy"] == "none"
    assert record["initial"]["status"] == {}
    assert "status_valid" not in record["initial"]["flags"]
    assert repaired.mass_artifact is geometry.mass_artifact
    assert repaired.initial_step_size < 2. / math.sqrt(adapter.precision)
    assert len(record["rounds"]) > 1
    assert record["rounds"][0]["epsilon"] == geometry.initial_step_size
    assert record["rounds"][-1]["proposal_valid"]
    assert all(n == 1 for n in record["graph_traces"].values())
    json.dumps(record, allow_nan=False)


def test_none_policy_still_vetoes_nonfinite_initial_score():
    class BrokenScore(NoStatusGaussian):
        def log_prob_and_grad(self, theta):
            value, score = super().log_prob_and_grad(theta)
            return value, score * tf.constant(float("nan"), tf.float64)
    adapter, geometry, config = inputs(BrokenScore())
    with pytest.raises(HMCPreparationFailure, match="invalid_initial_target"):
        initialize_bootstrap_step(adapter=adapter, geometry=geometry,
                                  config=replace(config, target_status_trace_policy="none"))


@pytest.mark.parametrize("kind", ["missing", "dtype", "shape"])
def test_declared_status_is_required_and_validated(kind):
    class BadStatus(Gaussian):
        def target_status_telemetry(self, theta):
            status = super().target_status_telemetry(theta)
            if kind == "missing":
                status.pop("floor_count_value")
            elif kind == "dtype":
                status["status_code"] = tf.cast(status["status_code"], tf.float64)
            else:
                status["floor_count_value"] = tf.constant(0, tf.int32)
            return status
    adapter, geometry, config = inputs(BadStatus())
    with pytest.raises((ValueError, TypeError), match="status|telemetry|dtype"):
        initialize_bootstrap_step(adapter=adapter, geometry=geometry, config=config)


@pytest.mark.parametrize("rounds", [True, -1, 1.5, float("inf")])
def test_startup_budget_validation(rounds):
    with pytest.raises(ValueError, match="bootstrap_initialization_rounds"):
        HMCKernelTuningConfig(bootstrap_initialization_rounds=rounds)


def test_public_startup_option_reaches_fresh_bootstrap(monkeypatch):
    from bayesfilter.inference import hmc_bootstrap
    class ReachedBootstrap(Exception):
        pass
    def screen(**kwargs):
        assert kwargs["config"].acceptance_role == "warmup_startup_only"
        record = kwargs["geometry"].formula_report["bootstrap_initialization"]
        assert record["max_rounds"] == 20
        assert record["target_status_trace_policy"] == "none"
        assert len(record["rounds"]) > 1
        assert all(row["seed"] != list(kwargs["config"].seed) for row in record["rounds"])
        raise ReachedBootstrap
    monkeypatch.setattr(hmc_bootstrap, "run_hmc_bootstrap_screen", screen)
    config = HMCKernelTuningConfig.smoke(use_xla=False, bootstrap_initialization_rounds=20)
    assert config.payload()["bootstrap_initialization_rounds"] == 20
    with pytest.raises(ReachedBootstrap):
        tune_hmc_kernel(adapter=NoStatusGaussian(precision=1.e6),
                        initial_position=[0., 0.], config=config)
    with pytest.raises(ValueError, match="windowed_adaptive"):
        replace(config, mass_policy="fixed_identity")


def test_disabled_startup_keeps_old_bootstrap_role():
    from bayesfilter.inference.hmc_configuration import _public_bootstrap_config
    config = HMCKernelTuningConfig()
    assert config.bootstrap_initialization_rounds == 0
    assert _public_bootstrap_config(config).acceptance_role == "fixed_kernel_screen"


def test_startup_bootstrap_exhaustion_cannot_fall_through_to_adaptation(monkeypatch):
    from types import SimpleNamespace
    from bayesfilter.inference import hmc_bootstrap, hmc_mass_adaptation
    from bayesfilter.inference.hmc_preparation import prepare_operational_windowed_mass_handoff
    monkeypatch.setattr(hmc_bootstrap, "run_hmc_bootstrap_screen", lambda **kwargs:
        SimpleNamespace(passed=False, artifact_hash="failed", final_status="repair_budget_exhausted", rounds=(),
                        payload=lambda: {"passed": False, "final_status": "repair_budget_exhausted"}))
    monkeypatch.setattr(hmc_mass_adaptation, "run_hmc_windowed_mass_stage", lambda **kwargs:
        pytest.fail("exhausted startup must not run adaptation"))
    with pytest.raises(HMCPreparationFailure, match="startup floor"):
        prepare_operational_windowed_mass_handoff(adapter=NoStatusGaussian(), initial_position=[0., 0.],
            config=HMCKernelTuningConfig.smoke(use_xla=False, bootstrap_initialization_rounds=20))


def search_config(**overrides):
    values = dict(primary_l_grid=(3, 5), initial_epsilon=1., refinement_rounds=2,
                  epsilon_refinement_factors=(), explore_failed_intervals=True)
    return HMCControllerConfig(**(values | overrides))


def missed_interval(work, candidate):
    if candidate.leapfrog_steps == 5:
        return {"decision": "passed", "acceptance": .7}
    if candidate.epsilon == 1.:
        return {"decision": "repair_step_higher", "acceptance": .99}
    if candidate.epsilon > 1.3:
        return {"decision": "unavailable", "evidence_validity": "candidate_data_invalid",
                "hard_vetoes": ("nonfinite_log_accept_ratio",), "acceptance": None}
    return {"decision": "passed", "acceptance": .7}


def controller(config=None, **scope_changes):
    return HMCTuningCandidateSetController(replace(_scope(), max_repairs_per_family=5,
                                                   **scope_changes), config or search_config())


def test_optional_interval_recovers_member_and_preserves_every_failure():
    baseline = controller(search_config(explore_failed_intervals=False)).run(missed_interval)
    result = controller().run(missed_interval)
    assert [c.leapfrog_steps for c in baseline.candidates if c.candidate_id in baseline.verified_candidate_ids] == [5]
    verified = [c for c in result.candidates if c.candidate_id in result.verified_candidate_ids]
    assert sorted(c.leapfrog_steps for c in verified) == [3, 5]
    child = next(c for c in verified if c.leapfrog_steps == 3)
    assert child.epsilon == pytest.approx(2.**.25)
    anchor = next(c for c in result.candidates if c.candidate_id == child.parent_candidate_id)
    assert anchor.epsilon == 1.
    assert anchor.candidate_family_id == child.candidate_family_id
    assert anchor.mass_signature == child.mass_signature
    rejected = [r for r in result.verification_receipts if r.evidence_validity != "valid"]
    assert len(rejected) == 2
    for receipt in rejected:
        assert receipt.acceptance is None and not receipt.repair_eligible
        assert result.candidate_states[receipt.candidate_id] == "promotion_failed"
        with pytest.raises(ValueError):
            result.replay_candidate(receipt.candidate_id)
    receipts = [r for r in result.verification_receipts if r.candidate_id == child.candidate_id]
    assert [r.stage for r in receipts] == ["measurement", "verification"]
    assert receipts[0].stream_id != receipts[1].stream_id
    assert result.budget_used_units == len(result.observations)
    assert result.budget_used_units + result.remaining_budget_units == result.config.total_budget_units
    events = [e for e in result.accounting_events if e["event"] == "rejected_interval_proposed"]
    assert len(events) == 2 and all(not e["qualification_transferred"] for e in events)


@pytest.mark.parametrize("pause", [1, 3, 5, 6])
def test_interval_resume_preserves_lineage_budget_and_all_members(pause):
    expected = controller().run(missed_interval)
    partial = controller().run(missed_interval, max_work_items=pause)
    restored = HMCTuningCandidateSetController.from_result_payload(json.loads(json.dumps(partial.payload())))
    result = restored.run(missed_interval)
    assert result.candidates == expected.candidates
    assert result.verification_receipts == expected.verification_receipts
    assert result.repair_actions == expected.repair_actions
    assert result.verified_candidate_ids == expected.verified_candidate_ids
    assert result.budget_used_units == expected.budget_used_units
    assert len({c.candidate_record_hash for c in result.candidates}) == len(result.candidates)


@pytest.mark.parametrize("reason", ["required_target_status_telemetry_missing", "nonfinite_state",
                                   "unknown_failure", "native_divergence_provenance_inconsistent"])
def test_non_numerical_rejection_does_not_create_interval(reason):
    def outcome(work, candidate):
        value = missed_interval(work, candidate)
        if value.get("evidence_validity") == "candidate_data_invalid":
            value["hard_vetoes"] = (reason,)
        return value
    result = controller().run(outcome)
    assert not any(e["event"] == "rejected_interval_proposed" for e in result.accounting_events)


def test_invalid_root_has_no_direction_or_repair_authority():
    result = controller().run(lambda *args: {"decision": "unavailable",
        "evidence_validity": "candidate_data_invalid", "hard_vetoes": ("nonfinite_log_accept_ratio",)})
    assert len(result.candidates) == 2 and not result.verified_candidate_ids
    assert not result.repair_actions


def test_shared_failure_stops_scope_and_disables_verified_sibling():
    def outcome(work, candidate):
        if candidate.epsilon > 1.:
            return {"decision": "unavailable", "evidence_validity": "shared_execution_invalid"}
        return missed_interval(work, candidate)
    result = controller().run(outcome)
    assert result.completion_status == "shared_invalidity"
    assert not result.verified_candidate_ids
    assert not any(e["event"] == "rejected_interval_proposed" for e in result.accounting_events)


def test_nonmonotone_lower_direction_can_explore_without_mutating_anchor():
    def outcome(work, candidate):
        if candidate.epsilon == 1.:
            return {"decision": "repair_step_lower", "acceptance": .1}
        if candidate.epsilon == .5:
            return {"decision": "unavailable", "evidence_validity": "candidate_data_invalid",
                    "hard_vetoes": ("nonfinite_log_accept_ratio",)}
        return {"decision": "passed", "acceptance": .7}
    result = controller().run(outcome)
    # Both funded rounds explore; the first successful pair never removes its
    # sibling or prevents the remaining declared refinement.
    assert len(result.verified_candidate_ids) == 4
    for steps in (3, 5):
        values = [result.replay_candidate(cid).epsilon for cid in result.verified_candidate_ids
                  if result.replay_candidate(cid).leapfrog_steps == steps]
        assert math.sqrt(.5) == pytest.approx(values[0])
        assert all(.5 < value < 1. for value in values)


@pytest.mark.parametrize("config", [search_config(refinement_rounds=0), search_config(max_candidates=3),
                                   search_config(total_budget_units=6, candidate_reserve_units=2,
                                                 repair_reserve_units=1, allow_repair_from_free_pool=False)])
def test_interval_search_respects_explicit_caps(config):
    result = controller(config).run(missed_interval)
    assert len(result.candidates) <= config.max_candidates
    assert result.budget_used_units <= config.total_budget_units
    assert not any(result.replay_candidate(cid).leapfrog_steps == 3 for cid in result.verified_candidate_ids)


def test_interval_family_cap_and_config_roundtrip():
    result = HMCTuningCandidateSetController(replace(_scope(), max_repairs_per_family=1),
                                           search_config()).run(missed_interval)
    assert len(result.repair_actions) == 1
    cfg = search_config()
    assert HMCControllerConfig.from_payload(json.loads(json.dumps(cfg.payload()))) == cfg
    historical = dict(cfg.payload())
    historical.pop("explore_failed_intervals")
    assert not HMCControllerConfig.from_payload(historical).explore_failed_intervals
    with pytest.raises(ValueError, match="boolean"):
        replace(cfg, explore_failed_intervals=1)


def test_interval_artifact_roundtrip_and_closed_primary_cohort(tmp_path):
    from bayesfilter.inference.hmc_candidate_set_artifacts import (
        write_candidate_set_result, load_candidate_set_result_payload,
    )
    result = controller().run(missed_interval)
    path = tmp_path/"result.json"
    write_candidate_set_result(result, path)
    loaded = load_candidate_set_result_payload(path)
    assert loaded["verified_candidate_ids"] == list(result.verified_candidate_ids)
    initial = [c.candidate_id for c in result.candidates if c.parent_candidate_id is None]
    assert [o["candidate_id"] for o in result.observations[:2]] == initial
    resumed = HMCTuningCandidateSetController.from_result_payload(loaded).run(missed_interval)
    assert resumed.verified_candidate_ids == result.verified_candidate_ids
    assert resumed.budget_used_units == result.budget_used_units
