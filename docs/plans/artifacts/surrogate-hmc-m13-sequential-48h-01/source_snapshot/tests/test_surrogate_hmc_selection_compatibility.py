"""CPU diagnostic checks for the narrowly reviewed selection compatibility."""
import ast
import json
from types import SimpleNamespace

import pytest
import tensorflow as tf

from bayesfilter.inference import surrogate_hmc_selection_compatibility as compat
from bayesfilter.testing.surrogate_hmc_m13_four_chain_cost import chunk_health

SELECTION = compat.ROOT / ("docs/plans/artifacts/"
    "surrogate-hmc-closure-20260922-m13-materiality-repair-01/"
    "m13-materiality-repair/selection.json")


def inputs():
    saved = json.loads(SELECTION.read_text())
    expected = dict(saved["source_hashes"])
    expected[compat.VERIFIER] = compat.file_hash(compat.ROOT / compat.VERIFIER)
    return saved, expected


def test_real_saved_selection_and_decisions_survive_only_reviewed_change():
    saved, expected = inputs()
    loaded = compat.load_compatible_selection(SELECTION, expected_scope=saved["scope"],
                                              expected_source_hashes=expected)
    assert loaded["controls"] == saved["controls"]
    assert loaded["decisions"] == saved["decisions"]
    assert loaded["source_hashes"] == saved["source_hashes"]
    assert loaded["source_compatibility"]["decisions_recomputed"]
    assert loaded["posterior_qualified"] is False


@pytest.mark.parametrize("damage", ["force", "verifier", "scope", "snapshot", "evidence"])
def test_rejects_unreviewed_or_corrupt_selection(damage, monkeypatch):
    saved, expected = inputs()
    scope = dict(saved["scope"])
    if damage == "force":
        name = next(n for n in expected if n != compat.VERIFIER)
        expected[name] = "changed"
    elif damage == "verifier":
        expected[compat.VERIFIER] = "changed"
    elif damage == "scope":
        scope["horizon"] = 21
    else:
        original = compat.file_hash
        target = (SELECTION.parents[1] / "source" / compat.VERIFIER if damage == "snapshot"
                  else compat.Path(saved["evidence"]["calibration"]["path"]))
        monkeypatch.setattr(compat, "file_hash", lambda p: "changed" if compat.Path(p) == target else original(p))
    with pytest.raises(ValueError):
        compat.load_compatible_selection(SELECTION, expected_scope=scope,
                                         expected_source_hashes=expected)


@pytest.mark.parametrize("failure", ["none", "fallback", "nan", "endpoint", "divergence", "energy"])
def test_actual_control_caller_retains_identical_health_decision(failure):
    saved, _ = inputs()
    snapshot = SELECTION.parents[1] / "source" / compat.VERIFIER
    old_source, new_source = snapshot.read_text(), (compat.ROOT / compat.VERIFIER).read_text()
    compat.check_health_compatibility(old_source, new_source)
    node = next(n for n in ast.parse(old_source).body if isinstance(n, ast.FunctionDef) and n.name == "_health")
    namespace = {"tf": tf}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(snapshot), "exec"), namespace)
    states = tf.ones([1, 4, 2], tf.float64)
    if failure == "nan":
        states *= float("nan")
    status = tf.one_hot(tf.zeros([1, 4], tf.int32), 9, dtype=tf.int32)
    if failure == "endpoint":
        status = tf.zeros_like(status)
    chunk = SimpleNamespace(states=states, final_state=states[-1],
        log_accept_ratio=tf.zeros([1, 4], tf.float64),
        delta_h=tf.fill([1, 4], tf.constant(1001. if failure == "energy" else 0., tf.float64)),
        endpoint_status=status, divergence=tf.fill([1, 4], failure == "divergence"),
        force_fallback=tf.fill([1, 4], failure == "fallback"))
    start = tf.zeros([4, 2], tf.float64)
    old = namespace["_health"](chunk, start, tf.constant(1000., tf.float64))
    current = chunk_health(chunk, start)
    fallback = bool(tf.reduce_any(chunk.force_fallback))
    historical_decision = all(bool(v) for k,v in old.items() if k != "all_chains_moved") and not fallback
    current_decision = all(bool(v) for k,v in current.items() if k != "all_chains_moved") and not fallback
    assert historical_decision == current_decision == (failure == "none")


def test_health_change_beyond_redundant_guard_is_rejected():
    snapshot = SELECTION.parents[1] / "source" / compat.VERIFIER
    current = (compat.ROOT / compat.VERIFIER).read_text()
    with pytest.raises(ValueError, match="health semantics"):
        compat.check_health_compatibility(snapshot.read_text(), current.replace("<= energy_max", "< energy_max"))
