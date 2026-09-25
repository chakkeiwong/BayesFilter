"""CPU/reference checks of endpoint diagnostics, never q20 quality evidence."""
import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import tensorflow as tf

from bayesfilter.inference.neutra_post_training import (
    PostTrainingProbe, assess_post_training, require_post_training_assessment, post_training_settings,
    POST_TRAINING_POINTS, PROBE_SCHEMA,
)
from bayesfilter.inference.q20_production_config import training_cohort
from bayesfilter.inference.q20_production_training import new_session, scope_for
from bayesfilter.inference.q20_training_validation import FrozenLossCache, ValidationBudgetExhausted
from tests.test_q20_production_repair import tiny_protocol, four_dimensional_bridge


def fixture():
    config, bridge = tiny_protocol(), four_dimensional_bridge()
    candidate = training_cohort(config, method="neutra")[0]
    scope = scope_for(config, bridge, candidate, sources={}, memory_policy={})
    session = new_session(config, bridge, candidate, scope=scope)
    return session, bridge


def nominee():
    return {"status": "hmc_trial_nominee", "hmc_trial_eligible": True,
        "continuing_improvement": True, "plateau_observed": False,
        "at_training_cap": True, "distinct_increment_observed": True}


@pytest.mark.parametrize("jit_compile", [False, True])
def test_exact_gaussian_and_bad_scaling_are_measured_without_mutating_training(jit_compile):
    session, bridge = fixture()
    before = session.checkpoint()
    probe = PostTrainingProbe(session.trainer.transport, bridge, 0., rows=8, jit_compile=jit_compile)
    exact = probe((61, 31))
    assert exact["finite"] and exact["valid_rows"] == 8
    assert exact["score_residual_rms"] < 1e-12
    assert exact["centered_log_density_rms"] < 1e-12
    probe((61, 32))
    assert probe.compiled.experimental_get_tracing_count() == 1
    assert session.checkpoint() == before
    concrete = probe.compiled.get_concrete_function()
    if jit_compile:
        assert concrete.function_def.attr["_XlaMustCompile"].b
    graph = concrete.graph.as_graph_def()
    nodes = list(graph.node) + [n for f in graph.library.function for n in f.node_def]
    assert not any(n.op in {"PyFunc", "EagerPyFunc"} for n in nodes)
    # The prior map used on the narrower beta-one Gaussian has analytic
    # residual -16*z plus its shifted-center term (up to the map permutation).
    bad = PostTrainingProbe(session.trainer.transport, bridge, 1., rows=8,
                           jit_compile=jit_compile)((61, 31))
    assert bad["finite"] and bad["score_residual_rms"] > 5.
    report = assess_post_training(state=before, decision=nominee(), parity={"passed": True},
                                 probe=bad, history=[])
    assert report["hmc_trial_eligible"]  # Finite residual size is not a veto.
    assert report["next_action"] == "fresh_fixed_transport_hmc_tuning"
    assert report["continuing_training_indicated"]
    assert not report["training_quality_established"]
    assert not report["posterior_qualified"]


def test_nonlinear_probe_includes_log_jacobian_score():
    class SinhMap:
        parameter_dim = 2
        stages = [SimpleNamespace(s_max=2., _network=lambda z: (tf.zeros_like(z), tf.zeros_like(z)),
            forward_and_logdet=lambda z: (tf.sinh(z), tf.reduce_sum(tf.math.log(tf.cosh(z)), axis=1)))]
        def forward_and_logdet(self, z):
            return self.stages[0].forward_and_logdet(z)
        def pullback_score_batch(self, z, score):
            return tf.cosh(z) * score
        def log_abs_det_jacobian_score_batch(self, z):
            return tf.tanh(z)
        def log_prob(self, x):
            z = tf.asinh(x)
            return -.5*tf.reduce_sum(z*z + tf.math.log(tf.constant(2.*3.141592653589793, tf.float64)), axis=1) - tf.reduce_sum(tf.math.log(tf.cosh(z)), axis=1)
    class StandardNormal:
        parameter_dim = 2
        def value_score_status(self, x, beta):
            return -.5*tf.reduce_sum(x*x, axis=1), -x, {"bridge_valid": tf.ones_like(x[:, 0], tf.bool)}
    seed = (61, 31)
    result = PostTrainingProbe(SinhMap(), StandardNormal(), 1., rows=8, jit_compile=True)(seed)
    z = tf.random.stateless_normal([8, 2], seed, dtype=tf.float64)
    residual = -tf.sinh(z)*tf.cosh(z) + tf.tanh(z) + z
    expected = tf.sqrt(tf.reduce_mean(tf.square(residual), axis=0))
    tf.debugging.assert_near(tf.constant(result["score_residual_rms_per_coordinate"], tf.float64),
                             expected, atol=1e-11, rtol=1e-11)


@pytest.mark.parametrize("nonfinite", [False, True])
def test_invalid_target_is_recorded_and_vetoes_trial(nonfinite):
    session, bridge = fixture()
    class InvalidBridge:
        parameter_dim = bridge.parameter_dim
        def value_score_status(self, x, beta):
            value, score, status = bridge.value_score_status(x, beta)
            if nonfinite:
                score = score * tf.constant(float("nan"), tf.float64)
            else:
                status = {**status, "bridge_valid": tf.zeros_like(status["bridge_valid"])}
            return value, score, status
    probe = PostTrainingProbe(session.trainer.transport, InvalidBridge(), 0., rows=8,
                              jit_compile=False)((61, 31))
    report = assess_post_training(state=session.checkpoint(), decision=nominee(),
        parity={"passed": True}, probe=probe, history=[])
    json.dumps(report, allow_nan=False)
    assert not report["numerical_check_passed"]
    assert not report["hmc_trial_eligible"]
    assert report["next_action"] == "repair_numerical_failure"


def test_saturated_scale_and_continuous_clipping_are_repair_signals():
    session, bridge = fixture()
    stage = session.trainer.transport.inner.stages[-1]
    stage.biases[-1].assign(tf.ones_like(stage.biases[-1])*20.)
    probe = PostTrainingProbe(session.trainer.transport, bridge, 0., rows=8,
                              jit_compile=False)((61, 31))
    report = assess_post_training(state=session.checkpoint(), decision=nominee(),
        parity={"passed": True}, probe=probe,
        history=[{"status": "accepted", "clipping_applied": True}]*4)
    assert "clipping_on_majority_of_updates" in report["repair_triggers"]
    assert "scale_parameterization_saturation" in report["repair_triggers"]
    assert report["numerical_check_passed"]  # Repair signals cannot reject the method.


def test_probe_cache_is_exact_map_bound_and_respects_budget(tmp_path):
    session, bridge = fixture()
    before = session.checkpoint()
    def cache():
        return FrozenLossCache(tmp_path, bridge=bridge, scope=session.scope,
                               batch_size=8, jit_compile=False)
    with pytest.raises(ValidationBudgetExhausted):
        cache().post_training_probe(before["map"], rows=8, seed=(61, 31), budget_check=lambda: False)
    first = cache().post_training_probe(before["map"], rows=8, seed=(61, 31))
    second = cache().post_training_probe(before["map"], rows=8, seed=(61, 31), budget_check=lambda: False)
    assert not first["cache_reused"] and second["cache_reused"]
    assert {**first, "cache_reused": True} == second
    session.advance(1)
    changed = session.checkpoint()
    assert not cache().post_training_probe(changed["map"], rows=8, seed=(61, 31))["cache_reused"]
    path = next(tmp_path.glob("*/geometry-*.json"))
    saved = json.loads(path.read_text())
    saved["result"]["score_residual_rms"] = 999.
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="checksum"):
        cache().post_training_probe(saved["identity"]["evaluation"]["map"], rows=8, seed=(61, 31))


@pytest.mark.parametrize("change", ["missing", "state", "map", "beta", "invalid", "short", "partial", "no_quantiles"])
def test_consumer_rejects_absent_stale_or_invalid_report(change):
    session, bridge = fixture()
    state = session.checkpoint()
    probe = PostTrainingProbe(session.trainer.transport, bridge, 0., rows=1000,
                              jit_compile=False)((61, 31))
    report = assess_post_training(state=state, decision=nominee(), parity={"passed": True},
                                  probe=probe, history=[])
    record = {"assessment": {"post_training": report, "beta": 0.,
        "current_map_hash": state["map"]["transport_state_hash"]},
        "training_checkpoint_hash": state["state_hash"],
        "frozen_transport": {"training_state_hash": state["state_hash"]}}
    assert require_post_training_assessment(record) is report
    changed = copy.deepcopy(record)
    r = changed["assessment"]["post_training"]
    if change == "missing":
        changed["assessment"].pop("post_training")
    elif change == "state":
        r["training_state_hash"] = "another-checkpoint"
    elif change == "map":
        r["map_hash"] = "another-map"
    elif change == "beta":
        r["beta"] = 1.
    elif change == "invalid":
        r["geometry"]["finite"] = False
    elif change == "short":
        r["geometry"].update(rows=32, valid_rows=32)
    elif change == "partial":
        r["geometry"]["complete"] = False
    else:
        r["geometry"]["score_residual_norm"].pop("p99")
    with pytest.raises(ValueError, match="post-training"):
        require_post_training_assessment(changed)


def test_standard_count_is_independent_of_small_map_parity_bank():
    config = tiny_protocol()
    config["role"] = "development"
    assert config["validation"]["reliability_rows"] == 8
    assert post_training_settings(config) == {"rows": 1000, "batch_size": 20}
    assert post_training_settings(config, sanity_only=True)["rows"] == 8


def test_full_1000_point_xla_report_matches_independent_statistics():
    session, bridge = fixture()
    probe = PostTrainingProbe(session.trainer.transport, bridge, 1., jit_compile=True)
    seed = (61, 31)
    bank = probe.latent_bank(seed)
    blocks = [probe.batch(bank[i:i+20]) for i in range(0, 1000, 20)]
    result = probe.summarize(blocks, seed)
    assert result["rows"] == POST_TRAINING_POINTS == 1000
    assert result["batch_size"] == 20 and result["batches"] == 50
    assert result["complete"] and result["finite"] and result["valid_rows"] == 1000
    assert result["traces"] == 1
    # Independent stdlib calculations from saved per-point observations.
    norm = [math.sqrt(math.fsum(v*v for v in row)) for block in blocks for row in block["score_residual"]]
    z_norm = [math.sqrt(math.fsum(v*v for v in row)) for block in blocks for row in block["latent"]]
    r = [v for block in blocks for v in block["log_ratio"]]
    stats = result["score_residual_norm"]
    ordered = sorted(norm)
    for field, expected in (("min", ordered[0]), ("max", ordered[-1]),
                            ("median", ordered[499]), ("p95", ordered[949]), ("p99", ordered[989]),
                            ("mean", math.fsum(norm)/1000),
                            ("rms", math.sqrt(math.fsum(v*v for v in norm)/1000))):
        assert stats[field] == pytest.approx(expected, rel=1e-12, abs=1e-12)
    assert stats["exceedance_fraction"]["1.0"] == pytest.approx(sum(v > 1 for v in norm)/1000)
    assert stats["fraction_larger_than_gaussian_score"] == pytest.approx(
        sum(a > b for a, b in zip(norm, z_norm, strict=True))/1000)
    assert result["r_log_target_over_gaussian_up_to_constant"]["range"] == pytest.approx(max(r)-min(r))
    mean_r = math.fsum(r)/1000
    assert result["centered_log_density_rms"] == pytest.approx(math.sqrt(math.fsum((x-mean_r)**2 for x in r)/1000))
    assert stats["rms"] == pytest.approx(2*result["score_residual_rms"])


def test_partial_bank_resumes_saved_batches_and_cannot_issue_summary(tmp_path):
    session, bridge = fixture()
    state = session.checkpoint()
    def cache():
        return FrozenLossCache(tmp_path, bridge=bridge, scope=session.scope, batch_size=8, jit_compile=False)
    def budget():
        return len(list(tmp_path.glob("*/geometry-*/batch-*.json"))) < 2
    with pytest.raises(ValidationBudgetExhausted, match="40/1000"):
        cache().post_training_probe(state["map"], seed=(61, 31), budget_check=budget)
    assert not list(tmp_path.glob("*/geometry-*.json"))
    paths = list(tmp_path.glob("*/geometry-*/batch-*.json"))
    preserved = {str(p): p.read_bytes() for p in paths}
    resumed = cache().post_training_probe(state["map"], seed=(61, 31))
    assert resumed["complete"] and resumed["rows"] == 1000
    assert resumed["traces"] == 1
    assert all(Path(p).read_bytes() == raw for p, raw in preserved.items())
    assert len(list(tmp_path.glob("*/geometry-*/batch-*.json"))) == 50
    direct = PostTrainingProbe(session.trainer.transport, bridge, 0., jit_compile=False)((61, 31))
    assert resumed["score_residual_norm"] == direct["score_residual_norm"]
    assert resumed["r_log_target_over_gaussian_up_to_constant"] == direct["r_log_target_over_gaussian_up_to_constant"]


def test_training_endpoint_uses_full_diagnostic_for_standard_role(tmp_path):
    """CPU analytic fixture isolates endpoint policy; no serious training run."""
    from bayesfilter.inference.q20_production_training import _evaluate_rung
    session, bridge = fixture()
    state = session.checkpoint()
    config = tiny_protocol()
    config["role"] = "development"  # Exercise production diagnostic sizing only.
    candidate = training_cohort(config, method="neutra")[0]
    cache = FrozenLossCache(tmp_path, bridge=bridge, scope=session.scope, batch_size=8, jit_compile=False)
    assessment, _ = _evaluate_rung(config, bridge, candidate, session, state, state, 0, cache=cache)
    report = assessment["post_training"]["geometry"]
    assert report["rows"] == 1000 and report["batches"] == 50
    assert report["r_log_target_over_gaussian_up_to_constant"]["range"] < 1e-12
    assert session.checkpoint() == state


def test_repair_summary_displays_full_distribution(tmp_path):
    import importlib.util
    script = Path(__file__).resolve().parents[1]/"scripts/q20_training_repair_watch.py"
    spec = importlib.util.spec_from_file_location("repair_watch_for_test", script)
    watcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(watcher)
    session, bridge = fixture()
    geometry = PostTrainingProbe(session.trainer.transport, bridge, 0., jit_compile=False)((61,31))
    report = assess_post_training(state=session.checkpoint(), decision=nominee(), parity={"passed": True},
                                  probe=geometry, history=[])
    result = {"arms":{"continue-clip":{"status":"repair_tranche_complete", "assessment":{"post_training":report}}},
        "aggregate_worker_seconds":1., "remaining_campaign_seconds":100., "remaining_diagnostic_seconds":50.}
    (tmp_path/"result.json").write_text(json.dumps(result))
    watcher.summary(tmp_path)
    text = (tmp_path/"tranche-summary.md").read_text()
    for fragment in ("1000", "Median", "p95", "p99", "Fraction >1", "Fraction >norm(z)", "Range of r"):
        assert fragment in text
