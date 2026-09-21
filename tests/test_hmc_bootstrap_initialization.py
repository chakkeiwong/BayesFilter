"""CPU/debugging checks of initialization, never posterior evidence."""
from dataclasses import replace
import json

import pytest
import tensorflow as tf

from bayesfilter.inference.hmc_bootstrap_initialization import (
    BootstrapMomentumProbe, initialize_bootstrap_step,
)
from bayesfilter.inference.hmc_kernel_tuning import (
    HMCBootstrapScreenConfig, HMCGeometryInitializationConfig,
    initialize_hmc_kernel_geometry, _bootstrap_leapfrog_payload,
)
from bayesfilter.inference.hmc_preparation import HMCPreparationFailure
from bayesfilter.inference.posterior_adapter import ValueScoreCapability


class Gaussian:
    parameter_dim = 2
    target_scope = "bootstrap_initialization_gaussian_reference"
    supports_retained_value_score_status = True

    def __init__(self, precision=1., domain=float("inf"), invalid=False):
        self.precision, self.domain, self.invalid = precision, domain, invalid

    def adapter_signature(self):
        return f"initialization-test:{self.precision}:{self.domain}:{self.invalid}"

    def value_score_capability(self):
        return ValueScoreCapability(value_score_authority="graph_native", xla_hmc_ready=True,
            full_chain_xla_diagnostic_ready=True, runtime_backend="tensorflow",
            target_scope=self.target_scope, evidence_path=__file__, nonclaims=("CPU toy only",))

    def target_status_telemetry(self, theta):
        valid = tf.reduce_all(tf.math.is_finite(theta) & (tf.abs(theta) < self.domain), axis=-1) & (not self.invalid)
        return {"valid_pre_regularized_score": valid, "status_code": tf.where(valid, 0, 1),
                "floor_count_value": tf.zeros_like(tf.cast(valid, tf.int32))}

    def log_prob_and_grad_status(self, theta):
        status = self.target_status_telemetry(theta)
        valid = status["valid_pre_regularized_score"]
        value = -.5 * self.precision * tf.reduce_sum(theta**2, axis=-1)
        score = -self.precision * theta
        nan = tf.constant(float("nan"), tf.float64)
        return tf.where(valid, value, nan), tf.where(valid[..., None], score, nan), status

    def log_prob_and_grad(self, theta):
        return self.log_prob_and_grad_status(theta)[:2]


def inputs(adapter=None, *, xla=False):
    adapter = adapter or Gaussian()
    geometry = initialize_hmc_kernel_geometry(adapter=adapter,
        initial_position=tf.zeros([2], tf.float64), parameter_scales=tf.ones([2], tf.float64),
        config=HMCGeometryInitializationConfig(geometry_scaling_c=.5, covariance_jitter=0.))
    config = HMCBootstrapScreenConfig(target_scope=adapter.target_scope, use_xla=xla,
        chain_execution_mode="tf_function", target_status_trace_policy="per_chain_step")
    return adapter, geometry, config


@pytest.mark.parametrize("xla", [False, True])
def test_healthy_start_preserves_target_mass_and_pair(xla):
    adapter, geometry, config = inputs(xla=xla)
    repaired = initialize_bootstrap_step(adapter=adapter, geometry=geometry, config=config)
    record = repaired.formula_report["bootstrap_initialization"]
    assert repaired.mass_artifact is geometry.mass_artifact
    assert repaired.adapter_signature == geometry.adapter_signature
    assert repaired.initial_step_size == geometry.initial_step_size
    pair = _bootstrap_leapfrog_payload(repaired.initial_step_size, repaired.target_trajectory_length,
                                      max_leapfrog_steps=config.max_leapfrog_steps)
    assert pair["num_leapfrog_steps"] == record["selected_num_leapfrog_steps"] == repaired.initial_num_leapfrog_steps
    assert record["artifact_authority"] is False
    assert list(record["graph_traces"].values()) == [1]
    assert record["rounds"][0]["seed"] != list(config.seed)


def test_rejected_nonfinite_proposal_shrinks_and_preserves_first_failure():
    adapter, geometry, config = inputs(Gaussian(precision=1.e4, domain=.2))
    events = []
    repaired = initialize_bootstrap_step(adapter=adapter, geometry=geometry, config=config,
        progress_callback=lambda phase, payload: events.append(phase))
    record = repaired.formula_report["bootstrap_initialization"]
    assert repaired.initial_step_size < geometry.initial_step_size
    first = record["first_invalid_proposal"]
    assert first["epsilon"] == geometry.initial_step_size
    assert first["retained_valid"] and not first["proposal_valid"]
    assert len(first["physical_state"]) == 4 and "score" in first and "status" in first
    json.dumps(record, allow_nan=False)
    assert "bootstrap_initialization.first_invalid_proposal" in events
    assert all(count == 1 for count in record["graph_traces"].values())
    assert record["rounds"][-1]["proposal_valid"]
    assert record["rounds"][-1]["mean_acceptance_probability"] >= config.repair_band[0]


@pytest.mark.parametrize("xla", [False, True])
def test_combined_diagnostics_do_not_replace_public_hmc_telemetry(xla):
    class RichStatusGaussian(Gaussian):
        def log_prob_and_grad_status(self, theta):
            value, score, status = super().log_prob_and_grad_status(theta)
            return value, score, {**status, "min_innovation_eigenvalue": tf.ones_like(value)}
    adapter, geometry, config = inputs(RichStatusGaussian(), xla=xla)
    repaired = initialize_bootstrap_step(adapter=adapter, geometry=geometry, config=config)
    initial = repaired.formula_report["bootstrap_initialization"]["initial"]
    assert initial["raw_status"]["min_innovation_eigenvalue"] == [1.]*4
    assert "min_innovation_eigenvalue" not in initial["status"]
    assert all(initial["healthy"])


def test_invalid_initial_target_is_fatal_before_any_transition(monkeypatch):
    adapter, geometry, config = inputs(Gaussian(invalid=True))
    monkeypatch.setattr(BootstrapMomentumProbe, "__call__", lambda *a: pytest.fail("transition on invalid initial target"))
    with pytest.raises(HMCPreparationFailure, match="invalid_initial_target") as error:
        initialize_bootstrap_step(adapter=adapter, geometry=geometry, config=config)
    assert not error.value.details["rounds"]


@pytest.mark.parametrize("kind", ["exception", "invalid_retained"])
def test_runtime_and_retained_failures_are_not_epsilon_repairs(monkeypatch, kind):
    adapter, geometry, config = inputs()
    original, count = BootstrapMomentumProbe.__call__, []
    def broken(self, *args):
        count.append(1)
        if kind == "exception":
            raise RuntimeError("injected runtime failure")
        result = original(self, *args)
        result["retained"]["healthy"] = tf.zeros([4], tf.bool)
        return result
    monkeypatch.setattr(BootstrapMomentumProbe, "__call__", broken)
    with pytest.raises(RuntimeError, match="runtime failure|invalid_retained_target"):
        initialize_bootstrap_step(adapter=adapter, geometry=geometry, config=config)
    assert len(count) == 1


def test_search_exhaustion_has_evidence_and_no_startup_nomination():
    adapter, geometry, config = inputs(Gaussian(precision=1.e4, domain=.2))
    with pytest.raises(HMCPreparationFailure, match="exhausted") as error:
        initialize_bootstrap_step(adapter=adapter, geometry=geometry, config=config, max_rounds=1)
    assert error.value.details["first_invalid_proposal"]
    assert "selected_epsilon" not in error.value.details


def test_initializer_precedes_fresh_bootstrap_with_exact_pair(monkeypatch):
    from bayesfilter.inference import hmc_kernel_tuning, hmc_bootstrap
    from bayesfilter.inference.hmc_preparation import prepare_operational_windowed_mass_handoff
    adapter = Gaussian(precision=1.e4, domain=.2)
    class ReachedFreshBootstrap(Exception):
        pass
    def bootstrap(**kwargs):
        geometry = kwargs["geometry"]
        record = geometry.formula_report["bootstrap_initialization"]
        assert record["selected_epsilon"] == geometry.initial_step_size
        pair = _bootstrap_leapfrog_payload(geometry.initial_step_size, geometry.target_trajectory_length,
                                          max_leapfrog_steps=kwargs["config"].max_leapfrog_steps)
        assert record["selected_num_leapfrog_steps"] == pair["num_leapfrog_steps"]
        assert all(row["seed"] != list(kwargs["config"].seed) for row in record["rounds"])
        raise ReachedFreshBootstrap
    monkeypatch.setattr(hmc_bootstrap, "run_hmc_bootstrap_screen", bootstrap)
    monkeypatch.setattr(hmc_kernel_tuning, "run_hmc_bootstrap_screen", bootstrap)
    config = hmc_kernel_tuning.HMCKernelTuningConfig.smoke(target_scope=adapter.target_scope,
        use_xla=False, chain_execution_mode="tf_function", target_status_trace_policy="per_chain_step")
    with pytest.raises(ReachedFreshBootstrap):
        prepare_operational_windowed_mass_handoff(adapter=adapter, initial_position=tf.zeros([2], tf.float64),
            parameter_scales=tf.ones([2], tf.float64), config=config, initialize_bootstrap=True)


@pytest.mark.parametrize("consumer", ["pricing", "classical_tuning"])
def test_both_q20_consumers_call_shared_repair(tmp_path, monkeypatch, consumer):
    from bayesfilter.inference import hmc_kernel_tuning, q20_production_hmc, q20_master_stages
    from tests.test_q20_production_repair import tiny_protocol, four_dimensional_bridge
    class ReachedPreparation(Exception):
        pass
    def prepare(**kwargs):
        assert kwargs["initialize_bootstrap"] is True
        from bayesfilter.inference.hmc_bootstrap_checkpoint import CheckpointedBootstrapRunner
        assert isinstance(kwargs["bootstrap_execution"], CheckpointedBootstrapRunner)
        assert callable(kwargs["progress_callback"])
        assert kwargs["initial_position"].shape == (4,)
        raise ReachedPreparation
    monkeypatch.setattr(hmc_kernel_tuning, "prepare_operational_windowed_mass_handoff", prepare)
    with pytest.raises(ReachedPreparation):
        if consumer == "pricing":
            q20_master_stages.price_preparation(tiny_protocol(), four_dimensional_bridge(),
                tmp_path / "pricing", beta=1., max_seconds=30.)
        else:
            q20_production_hmc.tune_scope(tiny_protocol(), four_dimensional_bridge(),
                tmp_path / "tune", method="classical", initial_position=tf.zeros([4, 4], tf.float64))
