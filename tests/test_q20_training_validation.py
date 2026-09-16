"""CPU reference checks of cached numerical evidence and decision boundaries."""
import json
from pathlib import Path

import pytest
import tensorflow as tf

from bayesfilter.inference.neutra_training_protocol import HeldoutLoss, assess_training_rung
from bayesfilter.inference.q20_training_validation import FrozenLossCache, ValidationBudgetExhausted
from bayesfilter.inference.q20_production_training import run_training_cohort
from tests.test_q20_production_repair import session, tiny_protocol, four_dimensional_bridge


def cache(root, bridge, scope, jit_compile=False):
    return FrozenLossCache(root, bridge=bridge, scope=scope, batch_size=8, jit_compile=jit_compile)


@pytest.mark.parametrize("jit_compile", [False, True])
def test_cache_extends_exact_prefix_and_reuses_shared_maps_after_restart(tmp_path, jit_compile):
    current, _, bridge = session()
    frozen = current.checkpoint()["map"]
    bank = cache(tmp_path, bridge, current.scope, jit_compile)
    seed = (83, 1)
    reference = HeldoutLoss(current.trainer.transport, bridge, .5, batch_size=8, jit_compile=jit_compile)
    for n in (16, 32, 64):
        tf.debugging.assert_equal(bank.evaluate(frozen, n, seed), reference(n, seed))
    assert bank.evaluated_rows == 64
    expected = bank.evaluate(frozen, 64, seed)
    assert bank.evaluated_rows == 64
    restarted = cache(tmp_path, bridge, current.scope, jit_compile)
    current.advance(1)
    tf.debugging.assert_equal(restarted.evaluate(frozen, 64, seed), expected)
    assert restarted.evaluated_rows == 0
    new = current.checkpoint()["map"]
    restarted.evaluate(new, 16, seed)
    restarted.evaluate(frozen, 16, (83, 2))
    at_one = current.next_beta(1., root_seed=(93, 1), preflight_seed=(71, 3))
    restarted.evaluate(at_one.checkpoint()["map"], 16, seed)
    assert restarted.evaluated_rows == 48
    assert len(restarted.entries) == 4
    if jit_compile:
        for entry in bank.entries.values():
            graph = entry["evaluator"].compiled
            assert graph.experimental_get_tracing_count() == 1
            concrete = graph.get_concrete_function()
            assert concrete.function_def.attr["_XlaMustCompile"].b
            definition = concrete.graph.as_graph_def()
            nodes = list(definition.node) + [node for fn in definition.library.function for node in fn.node_def]
            assert not any("PyFunc" in node.op or "HostCompute" in node.op for node in nodes)


def test_partial_validation_saves_completed_blocks_and_resumes_only_suffix(tmp_path):
    current, _, bridge = session()
    frozen = current.checkpoint()["map"]
    bank = cache(tmp_path, bridge, current.scope)
    with pytest.raises(ValidationBudgetExhausted):
        bank.evaluate(frozen, 64, (83, 1), budget_check=lambda: bank.evaluated_rows < 16)
    restarted = cache(tmp_path, bridge, current.scope)
    actual = restarted.evaluate(frozen, 64, (83, 1))
    assert restarted.evaluated_rows == 48
    expected = HeldoutLoss(current.trainer.transport, bridge, .5, batch_size=8, jit_compile=False)(64, (83, 1))
    tf.debugging.assert_equal(actual, expected)


def test_corrupt_cached_tensor_fails_closed(tmp_path):
    current, _, bridge = session()
    bank = cache(tmp_path, bridge, current.scope)
    bank.evaluate(current.checkpoint()["map"], 16, (83, 1))
    path = next(tmp_path.glob("*/loss-*.tensor"))
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        cache(tmp_path, bridge, current.scope).evaluate(current.checkpoint()["map"], 16, (83, 1))


@pytest.mark.parametrize("base,inc,at_cap,minimum_met,resolved,status", [
    ((-2., -1.), (-1.5, -.5), False, True, True, "continue_training"),
    ((-2., -1.), (.5, 1.5), False, True, True, "deterioration_repair_trigger"),
    ((-2., -1.), (-.5, .5), False, True, False, "expand_validation_or_continue"),
    ((-2., -1.), (-.5, .5), True, True, False, "cap_learning_observed"),
    ((.1, .5), (-.5, .5), True, True, True, "cap_learning_unresolved"),
    ((-2., -1.), (-.5, .5), False, False, True, "continue_training"),
])
def test_validation_precision_is_required_only_for_unresolved_decisions(base, inc, at_cap, minimum_met, resolved, status):
    def stats(bounds):
        return {"lower": bounds[0], "upper": bounds[1], "half_width": (bounds[1]-bounds[0])/2}
    decision = assess_training_rung(baseline=stats(base), increment=stats(inc), reliability=True,
        prior_plateaus=1, at_cap=at_cap, minimum_improvement=.04, maximum_half_width=.02,
        plateau_comparisons=2, minimum_updates_met=minimum_met)
    assert decision["validation_resolved"] is resolved
    assert decision["status"] == status
    assert decision["development_eligible"] is False


def test_precise_pre_floor_plateau_is_preserved_without_early_promotion():
    base = {"lower": -.6, "upper": -.59, "half_width": .005}
    inc = {"lower": -.005, "upper": .005, "half_width": .005}
    decision = assess_training_rung(baseline=base, increment=inc, reliability=True,
        prior_plateaus=1, at_cap=False, minimum_improvement=.04, maximum_half_width=.02,
        plateau_comparisons=2, minimum_updates_met=False)
    assert decision["plateaus"] == 2
    assert decision["development_eligible"] is False


def test_calibration_resumes_into_full_protocol_without_rng_or_optimizer_drift(tmp_path):
    config, bridge = tiny_protocol(), four_dimensional_bridge()
    config["training"]["cohort_min_updates"] = 2
    config["validation"]["bank_sizes"] = [16, 32]
    kwargs = dict(memory_policy={"mode": "tiny_cpu_reference"}, max_seconds=180.)
    calibration = run_training_cohort(config, bridge, tmp_path / "calibration", calibration_only=True, **kwargs)
    assert calibration["calibration_complete"] and not calibration["cohort_complete"]
    first = json.loads(Path(calibration["checkpoint"]).read_text())
    assert len(first["cohort"]) == 2
    for name, item in first["cohort"].items():
        assert item["session"]["level_updates"] == 1
        assert item["session"]["map"]["beta"] == (1. if name.startswith("direct") else .5)
        assert item["assessments"][0]["validation_evaluated_rows"] == 32
        assert not item["assessments"][0]["decision"]["development_eligible"]
    resumed = run_training_cohort(config, bridge, tmp_path / "resumed", resume=calibration["checkpoint"], **kwargs)
    uninterrupted = run_training_cohort(config, bridge, tmp_path / "whole", **kwargs)
    a = json.loads(Path(resumed["checkpoint"]).read_text())["cohort"]
    b = json.loads(Path(uninterrupted["checkpoint"]).read_text())["cohort"]
    assert a.keys() == b.keys()
    for name in a:
        for key in ("optimizer", "rng_index", "iteration", "level_updates"):
            assert a[name]["session"][key] == b[name]["session"][key]
        assert a[name]["session"]["map"]["variables"] == b[name]["session"]["map"]["variables"]
        assert [row["seed"] for row in a[name]["session"]["history"]] == [row["seed"] for row in b[name]["session"]["history"]]
    assert (tmp_path / "resumed/validation-cache").is_dir()
