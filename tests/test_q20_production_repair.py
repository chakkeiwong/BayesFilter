"""Tiny CPU/reference engineering tests, never q20 learning evidence."""
import copy
from dataclasses import replace
from pathlib import Path

import pytest
import tensorflow as tf

from bayesfilter.inference.q20_production_config import (
    protocol_template, validate_protocol, digest, scoped_seed, training_cohort,
)
from bayesfilter.inference.neutra_training_protocol import (
    TrainingSession, export_weighted_transport, transport_parity, HeldoutLoss,
    paired_loss_statistics, assess_training_rung,
)
from bayesfilter.inference.neutra_weighted_training import WeightedDenseIAFTransport, WeightedNeuTraConfig
from bayesfilter.inference.tempered_transport_ensemble_tf import (
    IndependentTemperedReverseKLTrainer, prepare_transport_initialization,
)
from tests.test_tempered_transport_ensemble import _bridge


def four_dimensional_bridge(jit_compile=False):
    from bayesfilter.inference.tempered_target_tf import GaussianLikelihoodBridge
    from tests.test_tempered_transport_ensemble import _AnalyticComponentTarget, _facts
    class FourDimensionalTarget(_AnalyticComponentTarget):
        def target_signature(self):
            return digest({"fixture": "four_dimensional_gaussian"})
        def batch_prior_likelihood_value_score_status(self, theta):
            center = tf.constant([.35, -.08, .65, .05], tf.float64)
            prior_score = -(theta-center)/16.
            delta = theta - tf.constant([.5, -.5, .25, 1.], tf.float64)
            valid = tf.reduce_all(tf.math.is_finite(theta), axis=1)
            return (-.5*tf.reduce_sum(delta**2, axis=1), -delta,
                    -.5*tf.reduce_sum((theta-center)**2, axis=1)/16., prior_score,
                    {"status_code": tf.where(valid, 0, 1), "valid_pre_regularized_score": valid,
                     "floor_count_value": tf.zeros_like(tf.cast(valid, tf.int32))})
    facts = {**_facts(), "parameter_dim": 4, "prior_variance": 16.}
    return GaussianLikelihoodBridge(FourDimensionalTarget(), prior_center=[.35, -.08, .65, .05],
                                    prior_variance=16., source_facts=facts, jit_compile=jit_compile)


def tiny_protocol():
    c = protocol_template()
    c.update(role="smoke", cpu_reference=True, jit_compile=False)
    c["training"].update(widths=[4], learning_rates=[.001], roots=[0, 1],
        rungs=[1, 2], cohort_min_updates=1, checkpoint_every=1, batch_size=8)
    c["validation"].update(bank_sizes=[16], reliability_rows=8)
    c["posterior"].update(warmup_min=64, warmup_window=64, warmup_chunk=64, warmup_max=128,
        warmup_rhat=1.5, retained_min=64, retained_chunk=64, retained_max=128,
        retained_rhat=1.5, bulk_ess=5., tail_ess=5., mean_mcse_sd=2., quantile_mcse_sd=2., event_mcse=1.)
    c["tuning"].update(l_grid=[3], initial_epsilon=.3, practical_region=[.5,.9], repair_region=[.41,.99],
        startup=8, pilot=64, measurement=64, verification=64, max_repairs_per_family=0,
        evidence_rungs=[1], total_budget_units=10, repair_reserve_units=3)
    validate_protocol(c)
    return c


def session(beta=.5):
    bridge = _bridge()
    config = WeightedNeuTraConfig(dimension=2, hidden_layers=(4, 4), stages=2,
        activation="tanh", initialization_seed=(16, 7), jit_compile=False)
    prepared = prepare_transport_initialization(WeightedDenseIAFTransport(config), bridge,
        component_id="test", seed=(71, 2), batch_size=8, beta=beta, repair_scales=(1.,),
        reference_center=[.3, -.4], reference_scale=[1.5, .7])
    trainer = IndependentTemperedReverseKLTrainer(config, bridge, beta=beta,
        component_id="test", batch_size=8, prepared_initialization=prepared)
    scope = {"data_identity": bridge.signature, "dtype": "float64", "backend": "CPU/reference",
             "jit_compile": False, "training_seed_derivation": {"root": [53, 1]},
             "validation_bank_ids": ["test-heldout"], "source": "test-source"}
    return TrainingSession(trainer=trainer, scope=scope, root_seed=(53, 1)), config, bridge


def test_protocol_shortcuts_fail_before_training():
    c = protocol_template()
    validate_protocol(c)
    assert len(training_cohort(c)) == 24
    for field, value in [("batch_size", 1), ("rungs", [2, 6]), ("beta_zero_updates", 2),
                         ("carry_optimizer_across_beta", False)]:
        broken = copy.deepcopy(c)
        broken["training"][field] = value
        with pytest.raises(ValueError):
            validate_protocol(broken)
    broken = copy.deepcopy(c)
    broken["cpu_reference"] = True
    with pytest.raises(ValueError):
        validate_protocol(broken)
    assert len({scoped_seed(c, "train", i) for i in range(1000)}) == 1000
    assert scoped_seed(c, "train", 0) != scoped_seed(c, "validation", 0)


def test_adam_exact_restart_and_beta_continuation():
    a, config, bridge = session()
    a.advance(3)
    saved = a.checkpoint()
    original_hash = saved["state_hash"]
    b = TrainingSession.restore(saved, bridge=bridge, config=config,
        expected_scope=a.scope, preflight_seed=(71, 2))
    a.advance(2)
    b.advance(2)
    for left, right in zip(a.trainer.variables, b.trainer.variables):
        tf.debugging.assert_equal(left, right)
    assert a.checkpoint()["optimizer"] == b.checkpoint()["optimizer"]
    assert a.rng_index == b.rng_index == 5
    assert digest({k:v for k,v in saved.items() if k != "state_hash"}) == original_hash
    before = a.checkpoint()
    next_level = a.next_beta(1., root_seed=(93, 1), preflight_seed=(71, 3))
    assert next_level.checkpoint()["optimizer"] == before["optimizer"]
    next_level.advance(1)
    assert int(next_level.trainer.optimizer.iterations.numpy()) == 6
    assert next_level.level_updates == 1
    assert next_level.history[-1]["clipping_applied"] in (True, False)


def test_checkpoint_corruption_and_scope_drift_reject():
    a, config, bridge = session()
    a.advance(1)
    saved = a.checkpoint()
    broken = copy.deepcopy(saved)
    broken["optimizer"][-1]["value"][0] += 1.
    with pytest.raises(ValueError, match="checksum"):
        TrainingSession.restore(broken, bridge=bridge, config=config,
            expected_scope=a.scope, preflight_seed=(71, 2))
    with pytest.raises(ValueError, match="scope"):
        TrainingSession.restore(saved, bridge=bridge, config=config,
            expected_scope={**a.scope, "source": "changed"}, preflight_seed=(71, 2))
    with pytest.raises(ValueError, match="configuration"):
        TrainingSession.restore(saved, bridge=bridge, config=replace(config, learning_rate=.04),
            expected_scope=a.scope, preflight_seed=(71, 2))


def test_post_update_nonfinite_rolls_back(monkeypatch):
    a, _, _ = session()
    before = a.checkpoint()
    original = a.trainer.train_step
    def broken(seed):
        result = original(seed)
        a.trainer.variables[0].assign(tf.fill(a.trainer.variables[0].shape, tf.constant(float("nan"), tf.float64)))
        return result
    monkeypatch.setattr(a.trainer, "train_step", broken)
    with pytest.raises(tf.errors.InvalidArgumentError):
        a.advance(1)
    after = a.checkpoint()
    assert after["map"]["variables"] == before["map"]["variables"]
    assert after["optimizer"] == before["optimizer"]
    assert after["history"][-1]["status"] == "rejected"


def test_nonzero_map_export_and_loss_assessment():
    a, config, bridge = session()
    baseline = HeldoutLoss(a.trainer.transport, bridge, .5, batch_size=8, jit_compile=False)(32, (83, 1))
    a.advance(5)
    payload, frozen = export_weighted_transport(a.trainer.transport,
        target_signature=digest({"fixture": "gaussian"}),
        training_state_hash=a.checkpoint()["state_hash"], transport_id="test-map")
    z = tf.random.stateless_normal([24, 2], (24, 5), dtype=tf.float64)
    result = transport_parity(a.trainer.transport, frozen, z, rtol=1e-10, atol=1e-11)
    assert result["passed"], result
    assert payload["components"][-1]["kind"] == "affine"
    after = HeldoutLoss(a.trainer.transport, bridge, .5, batch_size=8, jit_compile=False)(32, (83, 1))
    statistics = paired_loss_statistics(baseline, after, multiplier=1.96)
    assert statistics["rows"] == 32
    # Analytic pullback independently agrees with automatic differentiation of
    # the loaded finite map; GradientTape of summed independent rows uses no pfor.
    with tf.GradientTape() as tape:
        tape.watch(z)
        value = tf.reduce_sum(frozen.forward_batch(z)) + tf.reduce_sum(frozen.log_abs_det_jacobian_batch(z))
    score = frozen.pullback_score_batch(z, tf.ones_like(z)) + frozen.log_abs_det_jacobian_score_batch(z)
    tf.debugging.assert_near(score, tape.gradient(value, z), atol=1e-10)


@pytest.mark.parametrize("base,inc,precise,cap,previous,expected", [
    (-1., -.3, True, False, 0, "continue_training"),
    (-1., 0., True, False, 1, "plateau_nominee"),
    (-1., .3, True, False, 0, "deterioration_repair_trigger"),
    (0., 0., False, True, 0, "cap_learning_unresolved"),
    (-1., -.3, True, True, 0, "cap_learning_observed"),
])
def test_learning_dispositions(base, inc, precise, cap, previous, expected):
    def stats(value):
        h = .001 if precise else .1
        return {"half_width": h, "lower": value-h, "upper": value+h}
    result = assess_training_rung(baseline=stats(base), increment=stats(inc), reliability=True,
        prior_plateaus=previous, at_cap=cap, minimum_improvement=.04,
        maximum_half_width=.02, plateau_comparisons=2)
    assert result["status"] == expected
    assert result["posterior_qualified"] is False


def test_production_entrypoint_has_explicit_modes_and_nonpromotion():
    import subprocess,sys,json
    result=subprocess.run([sys.executable,"docs/benchmarks/run_ssl_lstm_q20_production_2026_09_15.py","validate"],
                          check=True,capture_output=True,text=True)
    report=json.loads(result.stdout)
    assert set(report["stages"]) >= {"train","tune","sample","reference","compare","confirmation"}
    assert report["promotion_eligible"] is False
    assert report["budget_required_before_numerical_work"] is True


def test_cohort_restart_runs_both_positive_temperatures(tmp_path):
    import json
    from bayesfilter.inference.q20_production_training import run_training_cohort
    config, bridge = tiny_protocol(), four_dimensional_bridge()
    kwargs = dict(memory_policy={"mode": "tiny_cpu_reference"}, max_seconds=120.)
    paused = run_training_cohort(config, bridge, tmp_path/"first", stop_after_rung=1, **kwargs)
    assert paused["status"] == "paused_at_requested_rung"
    resumed = run_training_cohort(config, bridge, tmp_path/"resumed", resume=paused["checkpoint"], **kwargs)
    assert resumed["cohort_complete"]
    checkpoint = json.loads(Path(resumed["checkpoint"]).read_text())
    for name,candidate in checkpoint["cohort"].items():
        assert candidate["session"]["map"]["beta"] == 1.
        direct=name.startswith("direct-")
        assert candidate["session"]["iteration"] == (2 if direct else 4)
        assert set(candidate["exports"]) == ({"1.0"} if direct else {"0.5", "1.0"})
        assert len(candidate["assessments"]) == (2 if direct else 4)
