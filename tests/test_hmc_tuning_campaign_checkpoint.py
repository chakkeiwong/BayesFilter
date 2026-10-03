"""Historical recovery checks use typed handoffs and deterministic runners."""
import json

import pytest
import tensorflow as tf

from bayesfilter.inference import hmc_kernel_tuning as tuning
from bayesfilter.inference.hmc_tuning_checkpoint import HMCTuningCampaignCheckpoint
from bayesfilter.runtime.durable_tensor_checkpoint import CheckpointError
from tests.test_hmc_kernel_tuning_fixed_mass_step import _geometry, _bootstrap, _windowed_stage
from tests.test_hmc_kernel_tuning_outer_loop import _loop_config, _tiny_budget_factory


def test_checkpoint_roundtrip_retains_private_stage_state_and_rejects_corruption(tmp_path):
    geometry, bootstrap, windowed = _geometry(), _bootstrap(), _windowed_stage()
    handoff = tuning._HMCPhaseAttemptState(
        mass_artifact_payload=geometry.mass_artifact.to_payload(include_arrays=True),
        mass_artifact_signature=geometry.mass_artifact_signature,
        canonical_theta_state=tf.constant([1., 2.], tf.float64),
        private_start_bank_theta=tf.ones((4, 2), tf.float64), private_start_bank_signature="bank",
        selected_step_size=0.05, selected_step_hash="epsilon", handoff_stage="phase5_selected",
    )
    with HMCTuningCampaignCheckpoint(tmp_path, {"target": "same"}, max_attempts=10, time_budget_s=18000) as cp:
        cp.stage("fixture", lambda: (geometry, bootstrap, windowed, handoff))
    with HMCTuningCampaignCheckpoint(tmp_path, {"target": "same"}, max_attempts=10, time_budget_s=18000) as cp:
        restored = cp.stage("fixture", lambda: pytest.fail("completed work reran"))
        assert restored[0].artifact_hash == geometry.artifact_hash
        assert restored[1].artifact_hash == bootstrap.artifact_hash
        assert restored[2].artifact_hash == windowed.artifact_hash
        assert restored[3].payload() == handoff.payload()
        assert (restored[3].private_start_bank_theta == handoff.private_start_bank_theta).all()
    shard = next((tmp_path / "committed/fixture").glob("tensor-*.bin"))
    shard.write_bytes(b"corrupted")
    with HMCTuningCampaignCheckpoint(tmp_path, {"target": "same"}, max_attempts=10, time_budget_s=18000) as cp:
        with pytest.raises(CheckpointError, match="checksum"):
            cp.load("fixture")
    with pytest.raises(CheckpointError, match="identity changed"):
        HMCTuningCampaignCheckpoint(tmp_path, {"target": "different"}, max_attempts=10, time_budget_s=18000)


def test_time_is_cumulative_and_extension_preserves_completed_work(tmp_path, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr("bayesfilter.inference.hmc_tuning_checkpoint.time.monotonic", lambda: clock[0])
    with HMCTuningCampaignCheckpoint(tmp_path, {}, max_attempts=3, time_budget_s=1000) as cp:
        cp.stage("geometry", _geometry)
        clock[0] += 400
    clock[0] += 50000  # Offline time is excluded.
    with HMCTuningCampaignCheckpoint(tmp_path, {}, max_attempts=10, time_budget_s=18000) as cp:
        assert cp.remaining_seconds == 17600
        cp.stage("geometry", lambda: pytest.fail("budget extension lost geometry"))
        clock[0] += 17550
    with pytest.raises(CheckpointError, match="budget exhausted"):
        HMCTuningCampaignCheckpoint(tmp_path, {}, max_attempts=10, time_budget_s=18000)


def test_interrupted_session_requires_supervisor_time_and_prevents_double_writer(tmp_path):
    cp = HMCTuningCampaignCheckpoint(tmp_path, {}, max_attempts=10, time_budget_s=18000)
    with pytest.raises(CheckpointError, match="live writer"):
        HMCTuningCampaignCheckpoint(tmp_path, {}, max_attempts=10, time_budget_s=18000)
    cp.store.close()  # Model SIGKILL: no context-manager closeout receipt.
    with pytest.raises(CheckpointError, match="reconciliation"):
        HMCTuningCampaignCheckpoint(tmp_path, {}, max_attempts=10, time_budget_s=18000)
    with HMCTuningCampaignCheckpoint(tmp_path, {}, max_attempts=10, time_budget_s=18000,
                                     interrupted_elapsed_s=123) as resumed:
        assert resumed.remaining_seconds == 17877
    ledger = json.loads((tmp_path / "campaign.json").read_text())
    assert ledger["calls"][0]["status"] == "interrupted_reconciled"


def test_completed_attempts_are_reused_after_interruption_and_budget_extension(tmp_path):
    from tests.test_hmc_kernel_tuning_fixed_mass_step import _ToyGaussianAdapter, _fake_result
    from tests.test_hmc_kernel_tuning_outer_loop import _passed_phase7_stage_fixtures
    windowed, fixed, trajectory = _passed_phase7_stage_fixtures()
    kwargs = dict(adapter=_ToyGaussianAdapter(), geometry=_geometry(), bootstrap=_bootstrap(),
                  _budget_policy_factory=_tiny_budget_factory,
                  _windowed_stage_runner=lambda **_: windowed,
                  _fixed_mass_step_stage_runner=lambda **_: fixed,
                  _frozen_step_trajectory_stage_runner=lambda **_: trajectory)

    def execute(root, acceptances, max_attempts=2, interrupted=False):
        calls = []
        first_plan = tuning._phase7_direct_candidate_queue_plan(
            fixed_mass_step_stage=fixed, config=_loop_config(max_attempts=max_attempts),
            budget_policy=_tiny_budget_factory(2, 0), attempt_index=0,
        )
        first_seeds = set(first_plan.verification_seeds)
        def verify(_adapter, _state, config):
            calls.append(config.seed)
            acceptance = 0.80 if tuple(config.seed) in first_seeds else 0.70
            return _fake_result(num_results=config.num_results, acceptance=acceptance)
        with HMCTuningCampaignCheckpoint(root, {"case": "loop"}, max_attempts=max_attempts, time_budget_s=18000) as cp:
            if interrupted:
                original = cp.commit_attempt
                def stop_after_commit(state):
                    original(state)
                    raise KeyboardInterrupt("simulated process interruption")
                cp.commit_attempt = stop_after_commit
            result = tuning.run_hmc_tune_verify_repair_loop(
                **kwargs, config=_loop_config(max_attempts=max_attempts),
                run_full_chain=verify, _campaign_checkpoint=cp,
            )
        return result, calls

    reference, reference_calls = execute(tmp_path / "reference", [0.80, 0.70])
    assert reference.passed
    with pytest.raises(KeyboardInterrupt):
        execute(tmp_path / "resumed", [0.80], interrupted=True)
    resumed, resumed_calls = execute(tmp_path / "resumed", [0.70])
    assert resumed.passed, [(a.final_status, a.verification_diagnostics) for a in resumed.attempts]
    assert len(resumed.attempts) == 2
    assert resumed_calls == reference_calls[-len(resumed_calls):]
    assert resumed.attempts[0].handoff_state_payload == reference.attempts[0].handoff_state_payload
    assert resumed.attempts[1].incoming_state_payload == reference.attempts[1].incoming_state_payload
    assert resumed.seed_report == reference.seed_report
    capped, _ = execute(tmp_path / "extension", [0.80], max_attempts=1)
    assert capped.final_status == "budget_exhausted"
    extended, extended_calls = execute(tmp_path / "extension", [0.70], max_attempts=2)
    assert extended.passed and len(extended_calls) == 1


def test_direct_phase5_repair_survives_checkpoint_and_keeps_mass_l_and_fresh_seed(tmp_path):
    from tests.test_hmc_kernel_tuning_outer_loop import (
        _phase7_direct_fixture, _ToyGaussianAdapter, _sequential_verification_diagnostics,
    )
    geometry, bootstrap, windowed, fixed, _handoff, identity, _alternative = _phase7_direct_fixture()
    config = _loop_config(max_attempts=10)
    common = dict(adapter=_ToyGaussianAdapter(), geometry=geometry, windowed_stage=windowed,
                  fixed_mass_step_stage=fixed, config=config, target_scope=config.target_scope)
    original = tuning._phase7_direct_candidate_verification_input(
        **common, attempt_index=0, candidate_identity=identity,
    )
    outcome = tuning._HMCPhase7FixedKernelVerificationOutcome(
        verification_input=original, verification_config_payload={"max_results": 64},
        diagnostics=_sequential_verification_diagnostics(0.80, rhat_passed=False),
        callback_result=tuning.FixedMassHMCTuningBudgetCallbackResult(),
        final_status="repair_or_retry", diagnostic_role="verification_acceptance_repair_trigger",
        hard_vetoes=(), continuation_scope="repair_or_retry",
        repair_triggers=(tuning._PHASE7_VERIFICATION_ACCEPTANCE_REPAIR_TRIGGER,), repair_evidence=None,
    )
    state = tuning._phase7_attempt_state_from_direct_outcome(config=config, windowed_stage=windowed,
                                                          outcome=outcome, repair_verification_reserved=True)
    assert tuning._phase7_should_run_operational_repair_verification(state)
    with HMCTuningCampaignCheckpoint(tmp_path, {}, max_attempts=10, time_budget_s=18000) as cp:
        cp.stage("direct", lambda: (state, windowed, fixed))
    with HMCTuningCampaignCheckpoint(tmp_path, {}, max_attempts=10, time_budget_s=18000) as cp:
        restored, common["windowed_stage"], common["fixed_mass_step_stage"] = cp.load("direct")
        repaired = tuning._phase7_operational_repair_verification_input(
            **common, attempt_index=1, attempt_state=restored,
        )
    assert repaired.step_size == state.verification_repair_step_size > original.step_size
    assert repaired.num_leapfrog_steps == original.num_leapfrog_steps
    assert repaired.adapted_mass_artifact_signature == original.adapted_mass_artifact_signature
    assert repaired.candidate_record_hash == original.candidate_record_hash
    assert repaired.verification_seed != original.verification_seed


def test_real_operational_warmup_state_roundtrips_without_repeating_hmc(tmp_path):
    from tests.test_hmc_kernel_tuning_windowed_mass import _operational_inputs, _operational_budget
    adapter, geometry, bootstrap = _operational_inputs()
    with HMCTuningCampaignCheckpoint(tmp_path, {"case": "tf-warmup"}, max_attempts=10, time_budget_s=18000) as cp:
        original = cp.stage("warmup", lambda: tuning.run_hmc_windowed_mass_stage(
            adapter=adapter, geometry=geometry, bootstrap=bootstrap,
            config=tuning.HMCWindowedMassStageConfig(
                chain_execution_mode="tf_function", target_scope="kernel_windowed_mass_toy_gaussian",
            ), _attempt_budget_policy=_operational_budget(),
        ))
    assert original.operational_warmup_result is not None
    with HMCTuningCampaignCheckpoint(tmp_path, {"case": "tf-warmup"}, max_attempts=10, time_budget_s=18000) as cp:
        restored = cp.stage("warmup", lambda: pytest.fail("warmup repeated"))
    assert restored.artifact_hash == original.artifact_hash
    left = restored.operational_warmup_result.final_kernel_state
    right = original.operational_warmup_result.final_kernel_state
    assert left.epsilon == right.epsilon
    assert left.current_epsilon_context_signature == right.current_epsilon_context_signature
    assert (left.canonical_theta == right.canonical_theta).all()
    assert (restored.operational_warmup_result.private_start_bank_theta == original.operational_warmup_result.private_start_bank_theta).all()

    # A fresh interpreter must reconstruct the types without importing modules
    # named by an artifact or relying on the first process's runtime caches.
    import subprocess
    import sys
    subprocess.run([
        sys.executable, "-c",
        "from bayesfilter.inference import hmc_kernel_tuning\n"
        "from bayesfilter.inference.hmc_tuning_checkpoint import HMCTuningCampaignCheckpoint\n"
        "import sys\n"
        "with HMCTuningCampaignCheckpoint(sys.argv[1], {'case': 'tf-warmup'}, max_attempts=10, time_budget_s=18000) as cp:\n"
        "    restored = cp.load('warmup')\n"
        "    assert restored.artifact_hash == sys.argv[2]\n",
        str(tmp_path), original.artifact_hash,
    ], check=True, capture_output=True, text=True, timeout=60)


def test_historical_executor_reuses_geometry_bootstrap_and_binds_numerical_inputs(tmp_path, monkeypatch):
    from tests.test_hmc_kernel_tuning_public_api import _ToyGaussianAdapter, _loop_result
    calls = []
    def geometry_runner(**_):
        calls.append("geometry")
        return _geometry()
    def bootstrap_runner(**_):
        calls.append("bootstrap")
        return _bootstrap()
    def loop_runner(**kwargs):
        cp = kwargs["_campaign_checkpoint"]
        assert cp.remaining_seconds < cp.time_budget_s or len(cp.ledger["calls"]) == 1
        return _loop_result(passed=True)
    monkeypatch.setattr(tuning, "initialize_hmc_kernel_geometry", geometry_runner)
    monkeypatch.setattr(tuning, "run_hmc_bootstrap_screen", bootstrap_runner)
    monkeypatch.setattr(tuning, "run_hmc_tune_verify_repair_loop", loop_runner)
    kwargs = dict(adapter=_ToyGaussianAdapter(), initial_position=[0.0, 0.0],
                  campaign_checkpoint_dir=tmp_path / "campaign")
    for index, budget in enumerate((1000, 18000)):
        result = tuning._run_canonical_hmc_tuning(
            **kwargs, campaign_time_budget_s=budget, output_dir=tmp_path / f"call-{index}",
            config=tuning.HMCKernelTuningConfig.smoke(
                target_scope="kernel_fixed_mass_step_toy_gaussian", max_attempts=3 if index == 0 else 10,
            ),
        )
        assert result.passed
    assert calls == ["geometry", "bootstrap"]
    kwargs["initial_position"] = [0.0, 0.1]
    with pytest.raises(CheckpointError, match="identity changed"):
        tuning._run_canonical_hmc_tuning(**kwargs, campaign_time_budget_s=18000,
                        config=tuning.HMCKernelTuningConfig.smoke(
                            target_scope="kernel_fixed_mass_step_toy_gaussian", max_attempts=10))


@pytest.mark.parametrize("option,value", [
    ("campaign_checkpoint_dir", "historical-campaign"),
    ("campaign_time_budget_s", 18000),
    ("campaign_interrupted_elapsed_s", 123),
])
def test_public_dispatch_rejects_historical_campaign_options_before_execution(
    option, value, monkeypatch,
):
    from bayesfilter.inference import hmc_candidate_set_public, tune_hmc_kernel

    def unexpected_execution(**_):
        pytest.fail("retired campaign options must be rejected before execution")

    monkeypatch.setattr(
        hmc_candidate_set_public, "run_shared_ordinary_tuning", unexpected_execution,
    )
    monkeypatch.setattr(tuning, "_run_canonical_hmc_tuning", unexpected_execution)
    with pytest.raises(ValueError, match="resume_hmc_candidate_set_tuning"):
        tune_hmc_kernel(adapter=object(), initial_position=None, **{option: value})
