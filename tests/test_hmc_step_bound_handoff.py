"""Regression fixtures for bounded adaptation, not posterior evidence."""

from types import SimpleNamespace
import time

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.inference import hmc, hmc_budget_ladder, hmc_kernel_tuning
from bayesfilter.inference import HMCTuningPolicy, ValueScoreCapability


class GaussianAdapter:
    parameter_dim = 2

    def adapter_signature(self):
        return "bounded-adaptation-gaussian-regression-v1"

    def value_score_capability(self):
        return ValueScoreCapability(
            value_score_authority="graph_native",
            xla_hmc_ready=False,
            runtime_backend="tensorflow",
            evidence_path=__file__,
            target_scope="bounded-adaptation-regression",
            nonclaims=("test fixture only",),
        )

    def log_prob_and_grad(self, position):
        position = tf.convert_to_tensor(position, tf.float64)
        return -0.5 * tf.reduce_sum(tf.square(position), axis=-1), -position


def run_config(**overrides):
    payload = dict(
        num_results=8,
        num_burnin_steps=8,
        step_size=0.05,
        num_leapfrog_steps=3,
        seed=(20260908, 1),
        target_scope="bounded-adaptation-regression",
        tuning_policy=HMCTuningPolicy.fixed_mass_dual_averaging(
            num_adaptation_steps=8, target_accept_prob=0.70, source=__file__
        ),
    )
    return hmc.FullChainHMCConfig(**(payload | overrides))


@pytest.mark.parametrize("bound", [0.0, -1.0, float("inf"), float("nan"), 0.01])
def test_invalid_or_below_initial_bound_is_rejected(bound):
    with pytest.raises(ValueError, match="step_size_upper_bound"):
        run_config(step_size_upper_bound=bound)


@pytest.mark.parametrize("reusable", [False, True])
def test_actual_adaptation_and_exported_step_stay_bounded(reusable, monkeypatch):
    import tensorflow_probability as tfp

    config = run_config(step_size_upper_bound=0.05)
    original = tfp.mcmc.sample_chain

    def capture(**kwargs):
        original_trace = kwargs["trace_fn"]

        def diagnostic_trace(state, results):
            return dict(original_trace(state, results)) | {
                "consumed_step_size": results.inner_results.proposed_results.step_size
            }

        kwargs["trace_fn"] = diagnostic_trace
        kwargs["num_results"] += kwargs["num_burnin_steps"]
        kwargs["num_burnin_steps"] = 0
        return original(**kwargs)

    monkeypatch.setattr(tfp.mcmc, "sample_chain", capture)
    initial = tf.constant([[0.0, 0.0], [0.1, -0.1]], tf.float64)
    if reusable:
        runner = hmc.build_reusable_full_chain_tfp_hmc_runner(
            adapter=GaussianAdapter(), initial_state_template=initial, config=config
        )
        result = runner.run(current_state=initial)
    else:
        result = hmc.run_full_chain_tfp_hmc(GaussianAdapter(), initial, config)
    assert np.isfinite(result.samples.numpy()).all()
    assert result.samples.shape[0] == 16
    assert np.max(result.trace["step_size"].numpy()) <= 0.05
    assert np.max(result.trace["consumed_step_size"].numpy()) <= 0.05


def test_healthy_uncapped_path_agrees_within_float64_roundoff():
    initial = tf.constant([[0.1, -0.1], [-0.2, 0.2]], tf.float64)
    baseline = run_config()
    uncapped = hmc.run_full_chain_tfp_hmc(GaussianAdapter(), initial, baseline)
    high_cap = hmc.run_full_chain_tfp_hmc(
        GaussianAdapter(), initial, run_config(step_size_upper_bound=100.0)
    )
    np.testing.assert_allclose(uncapped.samples.numpy(), high_cap.samples.numpy(), rtol=1.0e-10, atol=1.0e-10)
    np.testing.assert_allclose(
        uncapped.trace["step_size"].numpy(), high_cap.trace["step_size"].numpy(),
        rtol=1.0e-10, atol=1.0e-10,
    )


@pytest.mark.parametrize("leapfrog_count", [3, 5, 9, 13, 18, 25])
def test_every_per_l_config_reaches_actual_adaptation(leapfrog_count):
    ladder = hmc_kernel_tuning._joint_l_epsilon_ladder_config(
        hmc_kernel_tuning.HMCFixedMassStepStageConfig(),
        initial_step=0.05,
        num_leapfrog_steps=leapfrog_count,
        target_scope="bounded-adaptation-regression",
        seed_offset=0,
        qualified_step_size_upper_bound=0.08,
    )
    assert ladder.step_repair_max_step_size == 0.08
    runtime = hmc_budget_ladder._tune_config(
        ladder, budget=8, seed=(20260908, 1), step=0.05,
        target_scope="bounded-adaptation-regression",
    )
    assert runtime.step_size_upper_bound == 0.08
    assert runtime.signature_payload()["step_size_upper_bound"] == 0.08


def test_final_metric_bound_is_used_and_stale_metric_is_rejected():
    final = SimpleNamespace(
        transform=SimpleNamespace(signature="final-coordinates"),
        momentum_metric=SimpleNamespace(signature="final-metric"), epsilon=0.05,
    )
    window = SimpleNamespace(
        next_coordinate_signature=None, next_metric_signature=None,
        coordinate_signature_used="final-coordinates", metric_signature_used="final-metric",
        step_size_upper_bound=0.08, next_reasonable_epsilon=None, metric_decision=None,
    )
    stage = SimpleNamespace(operational_warmup_result=SimpleNamespace(
        final_kernel_state=final, windows=(window,)
    ))
    assert hmc_kernel_tuning._fixed_mass_step_upper_bound(stage) == 0.08
    window.coordinate_signature_used = "old-coordinates"
    window.metric_signature_used = "old-metric"
    with pytest.raises(ValueError, match="metric"):
        hmc_kernel_tuning._fixed_mass_step_upper_bound(stage)
    window.metric_decision = SimpleNamespace(update_applied=True)
    window.next_coordinate_signature = "final-coordinates"
    window.next_metric_signature = "final-metric"
    with pytest.raises(ValueError, match="fresh qualified"):
        hmc_kernel_tuning._fixed_mass_step_upper_bound(stage)
    window.next_reasonable_epsilon = SimpleNamespace(passed=True, selected_step_size=0.06)
    assert hmc_kernel_tuning._fixed_mass_step_upper_bound(stage) == 0.06


def test_real_windowed_handoff_uses_last_consumed_metric():
    from tests.test_hmc_kernel_tuning_windowed_mass import (
        OPERATIONAL_WINDOWED_WARMUP_ALGORITHM_ID,
        _operational_budget, _operational_inputs, _stage_config,
    )

    adapter, geometry, bootstrap = _operational_inputs()
    stage = hmc_kernel_tuning.run_hmc_windowed_mass_stage(
        adapter=adapter, geometry=geometry, bootstrap=bootstrap,
        config=_stage_config(
            chain_execution_mode="tf_function",
            algorithm_id=OPERATIONAL_WINDOWED_WARMUP_ALGORITHM_ID,
        ),
        _attempt_budget_policy=_operational_budget(),
    )
    assert stage.passed
    window = stage.operational_warmup_result.windows[-1]
    assert window.next_coordinate_signature is None
    assert window.next_metric_signature is None
    assert hmc_kernel_tuning._fixed_mass_step_upper_bound(stage) == window.step_size_upper_bound


def test_grid_caller_threads_bound_to_all_candidates(monkeypatch):
    received = []

    def record_config(**kwargs):
        received.append(kwargs["config"])
        raise RuntimeError("injected candidate failure after recording the handoff")

    monkeypatch.setattr(hmc_kernel_tuning, "run_fixed_mass_hmc_tuning_budget_ladder", record_config)
    grid = (3, 5, 9, 13, 18, 25)
    result = hmc_kernel_tuning._run_joint_l_epsilon_grid_round(
        adapter=GaussianAdapter(), adapted_mass=None, initial_state_factory=None,
        config=hmc_kernel_tuning.HMCFixedMassStepStageConfig(), initial_step=0.05,
        qualified_step_size_upper_bound=0.08,
        target_scope="bounded-adaptation-regression", target_trajectory=0.3,
        anchor_l=3, max_leapfrog_steps=25, round_index=0, grid_stage="primary",
        grid_values=grid, fixed_mass_stage_start_perf_counter_s=time.perf_counter(),
        completed_candidate_elapsed_s=[], screen_callback=None,
        run_full_chain=hmc.run_full_chain_tfp_hmc, attempt_budget_policy=None,
        attempt_state=None, progress_callback=None, progress_attempt_index=0,
    )
    assert tuple(config.num_leapfrog_steps for config in received) == grid
    assert all(config.step_size_upper_bound == 0.08 for config in received)
    assert all(config.step_repair_max_step_size == 0.08 for config in received)
    assert len(result["candidates"]) == len(grid)


def test_cache_identity_includes_step_bound():
    payloads = [hmc_budget_ladder._reusable_static_contract_payload(
        run_config(step_size_upper_bound=bound),
        hmc_adapter_signature="gaussian", target_dimension=2, mass_signature="identity",
        initial_state=tf.zeros([2, 2], tf.float64), dynamic_num_leapfrog_steps=True,
    ) for bound in (0.05, 0.08)]
    assert payloads[0] != payloads[1]
    assert payloads[0]["static_config"]["step_size_upper_bound"] == 0.05


@pytest.mark.parametrize("algorithm_id,fresh_bracket", [
    (hmc_kernel_tuning.ORDINARY_BROAD_FIXED_METRIC_ALGORITHM_ID, True),
    (hmc_kernel_tuning.LEGACY_JOINT_L_EPSILON_ALGORITHM_ID, False),
])
def test_acceptance_bracket_is_not_transferred_between_leapfrog_counts(
    algorithm_id, fresh_bracket,
):
    bracket = {"next_step_size": 0.05, "high_acceptance_step_lower_bound": 0.04,
               "low_acceptance_step_upper_bound": 0.07}
    attempt = SimpleNamespace(selected_num_leapfrog_steps=3,
                              verification_repair_max_step_size=0.06,
                              fixed_mass_bracket_state=bracket)
    configurations = [hmc_kernel_tuning._joint_l_epsilon_ladder_config(
        hmc_kernel_tuning.HMCFixedMassStepStageConfig(algorithm_id=algorithm_id),
        initial_step=0.05,
        num_leapfrog_steps=count, target_scope="bounded-adaptation-regression",
        seed_offset=0, attempt_state=attempt, qualified_step_size_upper_bound=0.08,
    ) for count in (3, 5)]
    assert all(config.require_finite_trajectory_bracket == fresh_bracket for config in configurations)
    assert (configurations[0].initial_fixed_mass_bracket_state is None) == fresh_bracket
    assert configurations[0].step_repair_max_step_size == 0.06
    assert configurations[1].initial_fixed_mass_bracket_state is None
    assert configurations[1].step_repair_max_step_size == 0.08
