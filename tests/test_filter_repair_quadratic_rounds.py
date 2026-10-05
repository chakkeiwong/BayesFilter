"""Complete original-record diagnostics for the native paired round controller."""

import dataclasses
import json
import re
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.inference.batched_quadratic_center import (
    BatchedQuadraticCenterConfig,
    BatchedQuadraticCenterResult,
    refine_batched_quadratic_center,
)
from bayesfilter.inference.quadratic_rounds_tf import (
    ROLES,
    STATUS,
    make_paired_quadratic_controller,
    paired_quadratic_controller,
)
from tests.test_filter_repair_quadratic_batches import _equal_records
from tests.test_filter_repair_quadratic_numerics import inputs, original

D = tf.float64
CASES = ("centered", "move", "nonquadratic", "round_limit", "invalid_initial", "invalid_replay",
    "invalid_large", "invalid_small", "invalid_check", "invalid_proposal", "invalid_final_replay",
    "replay_value", "initial_evidence", "bad_evidence", "nonspd", "nonfinite_scaled_score", "factor_failure")


def fixture(dimension, case, *, counter=None):
    precision = inputs("trust", dimension)[0]
    centered = case in ("centered", "invalid_final_replay", "nonfinite_scaled_score", "factor_failure")
    center = tf.zeros([dimension], D) if centered else tf.linspace(tf.constant(-.4, D), tf.constant(.7, D), dimension)
    scale = tf.linspace(tf.constant(.7, D), tf.constant(1.3, D), dimension)
    if case == "nonfinite_scaled_score":
        scale = tf.fill([dimension], tf.constant(2., D))
    if case == "factor_failure":
        scale = tf.fill([dimension], tf.constant(1e155, D))
    if case == "nonspd":
        precision = tf.linalg.diag(tf.constant([-1.] + [1.] * (dimension - 1), D))
    per_design = (2 * dimension + 3) // 4
    invalid_call = {"invalid_initial": 1, "invalid_replay": 2, "invalid_large": 3,
        "invalid_small": 3 + per_design, "invalid_check": 3 + 2 * per_design,
        "invalid_proposal": 3 + 3 * per_design, "invalid_final_replay": 3 + 3 * per_design}.get(case, 0)

    def callback(points):
        call = tf.constant(0, tf.int64) if counter is None else counter.assign_add(1)
        score = -points @ precision
        value = -.5 * tf.reduce_sum(points * (points @ precision), axis=1)
        if case == "nonquadratic":
            score -= .04 * points ** 3
            value -= .01 * tf.reduce_sum(points ** 4, axis=1)
        if case == "nonfinite_scaled_score":
            score = tf.fill(tf.shape(points), tf.constant(1e308, D))
            value = tf.reduce_sum(points * score, axis=1)
        if case == "factor_failure":
            scaled = points / scale
            score = -scaled / scale
            value = -.5 * tf.reduce_sum(scaled * scaled, axis=1)
        if case == "replay_value":
            value += tf.where(call == 2, tf.constant(.01, D), tf.constant(0., D))
        eligible = tf.ones([4], tf.bool)
        if invalid_call:
            eligible &= (call != invalid_call) | (tf.range(4) != 1)
        return value, score, eligible

    config = BatchedQuadraticCenterConfig(pilot_method="paired_local", max_fit_rounds=1 if case == "round_limit" else 4,
        centeredness_cap=1e-8)
    has_evidence = case in ("initial_evidence", "bad_evidence")
    initial_value = -.5 * tf.reduce_sum(center * tf.linalg.matvec(precision, center))
    initial_score = -tf.linalg.matvec(precision, center)
    if has_evidence:
        initial_value += .001 if case == "bad_evidence" else 1e-12
        initial_score += 1e-12
    args = (center, scale, tf.constant(config.seed), tf.constant(has_evidence), initial_value, initial_score)
    return callback, config, args


def materialize(result, config, dimension, seed, *, bulk=True):
    """Format completed records only; execution metadata is compared separately."""
    status = STATUS[int(result["status"])]
    accepted = status == "local_center_candidate"
    history, rounds = result["history"], result["rounds"]
    if bulk:
        history, rounds = tf.nest.map_structure(lambda value: value.numpy().tolist(), (history, rounds))
    reports = []
    for index in range(int(result["iterations"])):
        if not bool(rounds["model_present"][index]):
            continue
        report = {key: rounds[key][index] for key in ("anchor", "anchor_value", "radius", "cloud_changed_incumbent")}
        report["model"] = {key: value[index] for key, value in rounds["models"].items()}
        if bool(rounds["centeredness_present"][index]):
            report["centeredness"] = rounds["centeredness"][index]
        if bool(rounds["step_present"][index]):
            report.update({key: rounds[key][index] for key in ("actual_improvement", "ratio", "trust_step_accepted")})
            report["step"] = {key: value[index] for key, value in rounds["steps"].items()}
        reports.append(report)
    batches = [{"role": ROLES[int(history["roles"][index])], "first_index": history["first_index"][index],
        **{key: history[key][index] for key in ("positions", "values", "scores", "valid")}}
        for index in range(int(result["callback_batches"]))]
    details = {key: result[key] for key in ("physical_rows", "padded_rows", "invalid_rows", "callback_batches")}
    details.update(planned_physical_rows=config.planned_rows(dimension), selected_evaluation_index=result["selected_index"],
        rounds=reports, candidate_batches=batches, seed=seed, pilot_method=config.pilot_method, paired_steps=config.paired_steps)
    return BatchedQuadraticCenterResult(accepted, status, result["center"], result["center_value"], result["center_score"],
        result["factor"] if accepted else None, result["precision"] if accepted else None, details).payload()


def reference(callback, config, args):
    baseline = original()[1]["trust"]
    options = dataclasses.asdict(config)
    options.update(seed=int(args[2]), jit_compile_trust=False)
    result = baseline.refine_batched_quadratic_center(callback, args[0], args[1],
        config=baseline.BatchedQuadraticCenterConfig(**options),
        _initial_evidence=(args[4], args[5]) if bool(args[3]) else None).payload()
    assert result["diagnostics"].pop("full_initializer_xla") is False
    assert result["diagnostics"].pop("jit_compile_trust") is False
    traces = {key: result["diagnostics"].pop(key) for key in ("fit_trace_count", "trust_trace_count")
              if key in result["diagnostics"]}
    return result, traces


def public_record(payload, *, jit, original_traces):
    """Assert changed execution metadata; return every original numerical field."""
    details = payload["diagnostics"]
    assert details.pop("full_initializer_xla") is False
    assert details.pop("jit_compile_refinement") is jit
    assert details.pop("jit_compile_trust") is jit
    assert details.pop("jit_compile_fit") is jit
    assert details.pop("refinement_trace_count") == 1
    calls = {"fit_trace_count": details.pop("fit_calls"), "trust_trace_count": details.pop("trust_calls")}
    if payload["accepted"]:
        assert details.pop("fit_trace_count") == details.pop("trust_trace_count") == 1
        assert original_traces == {key: int(value > 0) for key, value in calls.items()}
    else:
        assert "fit_trace_count" not in details and "trust_trace_count" not in details
    return payload


def compare_public_records(actual, expected, *, jit):
    """Check the public boundary and original metadata before numerical parity."""
    assert expected["diagnostics"].pop("full_initializer_xla") is False
    expected["diagnostics"].pop("jit_compile_trust")
    traces = {key: expected["diagnostics"].pop(key) for key in ("fit_trace_count", "trust_trace_count")
        if key in expected["diagnostics"]}
    _equal_records(public_record(actual, jit=jit, original_traces=traces), expected)


@pytest.mark.parametrize("dimension", [1, 3, 5])
@pytest.mark.parametrize("case", CASES)
def test_complete_original_round_records(dimension, case, request):
    counter = tf.Variable(0, dtype=tf.int64)
    callback, config, args = fixture(dimension, case, counter=counter)
    expected, traces = reference(callback, config, args)
    expected_status = {"centered": "local_center_candidate", "round_limit": "refinement_round_limit",
        "invalid_initial": "initial_target_invalid", "invalid_replay": "replay_mismatch",
        "invalid_large": "curvature_target_invalid", "invalid_small": "curvature_target_invalid",
        "invalid_check": "curvature_target_invalid", "invalid_final_replay": "replay_mismatch",
        "replay_value": "replay_mismatch", "bad_evidence": "localizer_refinement_replay_mismatch",
        "nonspd": "quadratic_model_rejected", "nonfinite_scaled_score": "nonfinite_scaled_score",
        "factor_failure": "pilot_factorization_failed"}.get(case)
    if expected_status is not None:
        assert expected["status"] == expected_status
    calls = int(counter)
    records = {}
    for mode in ("graph", "xla"):
        counter.assign(0)
        options = dataclasses.replace(config, jit_compile_trust=mode == "xla")
        public = refine_batched_quadratic_center(callback, args[0], args[1], config=options,
            _initial_evidence=(args[4], args[5]) if bool(args[3]) else None).payload()
        assert int(counter) == calls
        _equal_records(public_record(public, jit=mode == "xla", original_traces=traces), expected)
        # Exercise the actual cached public program again without recompiling
        # an identical fresh factory for this reporting/schema comparison.
        program = paired_quadratic_controller(callback, dimension, options, jit_compile=mode == "xla")
        counter.assign(0)
        output = program(*args)
        actual = materialize(output, config, dimension, int(args[2]))
        _equal_records(actual, materialize(output, config, dimension, int(args[2]), bulk=False))
        records[mode] = actual
        assert int(counter) == calls == int(output["callback_batches"])
        if traces:
            assert traces == {"fit_trace_count": int(output["fit_calls"] > 0), "trust_trace_count": int(output["trust_calls"] > 0)}
        assert program.experimental_get_tracing_count() == 1
    directory = Path(request.config.getoption("xmlpath")).parent
    with (directory / f"quadratic-rounds-{case}-{dimension}.json").open("x") as handle:
        json.dump({"original": expected, **records, "original_trace_counts": traces,
            "baseline": "3582b4ac", "source_sha256": original()[0].hashes(),
            "execution_metadata_note": "Complete candidate controller compiles as one boundary; original trust is the graph precision reference. Numerical fields and decisions are unmodified."}, handle, indent=2, allow_nan=False)
        handle.write("\n")
    for actual in records.values():
        _equal_records(actual, expected)


def test_round_controller_changed_inputs_and_enclosing_xla(request):
    callback, config, args = fixture(3, "move")
    program = make_paired_quadratic_controller(callback, 3, config)
    center = args[0] + .01
    values, scores, _ = callback(tf.repeat(center[None], 4, axis=0))
    changed = (center, args[1] * 1.1, args[2] + 1, tf.constant(True), values[0] + 1e-12, scores[0] + 1e-12)
    first, second = program(*args), program(*changed)
    assert not np.array_equal(first["history"]["positions"], second["history"]["positions"])
    expected, _ = reference(callback, config, changed)
    _equal_records(materialize(second, config, 3, int(changed[2])), expected)
    assert program.experimental_get_tracing_count() == 1
    concrete = program.get_concrete_function()
    graph = concrete.graph.as_graph_def()
    nodes = list(graph.node) + [node for fn in graph.library.function for node in fn.node_def]
    assert not {"PyFunc", "EagerPyFunc", "PyFuncStateless"} & {node.op for node in nodes}
    hlo = program.experimental_get_compiler_ir(*args)(stage="hlo")
    changed_hlo = program.experimental_get_compiler_ir(*changed)(stage="hlo")
    directory = Path(request.config.getoption("xmlpath")).parent
    for name, value in (("initial.hlo", hlo), ("changed.hlo", changed_hlo), ("graph.pbtxt", str(graph))):
        with (directory / name).open("x") as handle:
            handle.write(value)
    assert hlo == changed_hlo
    expected_operands = len(args) + len(concrete.captured_inputs)
    assert len(re.findall(r"\bparameter\((\d+)\)", hlo[hlo.rfind("\nENTRY "):])) == expected_operands
    outer = tf.function(lambda *values: program(*values), input_signature=program.input_signature,
        autograph=False, jit_compile=True)
    enclosed = outer(*changed)
    for left, right in zip(tf.nest.flatten(enclosed), tf.nest.flatten(second), strict=True):
        np.testing.assert_allclose(left, right, atol=1e-10, rtol=1e-10)
