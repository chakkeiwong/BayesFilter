"""Complete original-record qualification of the uniform-cloud controller."""

import json
import re
from pathlib import Path

import pytest
import tensorflow as tf

from bayesfilter.inference.batched_quadratic_center import BatchedQuadraticCenterConfig
from bayesfilter.inference.quadratic_rounds_tf import make_quadratic_controller
from tests.test_filter_repair_quadratic_batches import _equal_records
from tests.test_filter_repair_quadratic_numerics import inputs, original
from tests.test_filter_repair_quadratic_rounds import materialize, reference

D = tf.float64
CASES = ("centered", "move", "nonquadratic", "round_limit", "invalid_initial", "invalid_replay",
    "invalid_fit", "invalid_check", "invalid_proposal", "invalid_final_replay", "replay_value",
    "initial_evidence", "bad_evidence", "nonspd", "nonfinite_scaled_score", "factor_failure")


def fixture(dimension, case, *, counter=None, rows=None, seed=20260910, width=.1):
    matrix = inputs("trust", dimension)[0]
    centered = case in ("centered", "invalid_final_replay", "nonfinite_scaled_score", "factor_failure")
    center = tf.zeros([dimension], D) if centered else tf.linspace(tf.constant(-.4, D), tf.constant(.7, D), dimension)
    scale = tf.linspace(tf.constant(.7, D), tf.constant(1.3, D), dimension)
    if case == "nonfinite_scaled_score":
        scale = tf.fill([dimension], tf.constant(2., D))
    if case == "factor_failure":
        scale = tf.fill([dimension], tf.constant(1e155, D))
    if case == "nonspd":
        matrix = tf.linalg.diag(tf.constant([-1.] + [1.] * (dimension - 1), D))
    count = max(32, 4 * dimension) if rows is None else rows
    per_design = (count + 3) // 4
    bad = {"invalid_initial": 1, "invalid_replay": 2, "invalid_fit": 3,
        "invalid_check": 3 + per_design, "invalid_proposal": 3 + 2 * per_design,
        "invalid_final_replay": 3 + 2 * per_design}.get(case, 0)

    def callback(points):
        call = tf.constant(0, tf.int64) if counter is None else counter.assign_add(1)
        score = -points @ matrix
        value = -.5 * tf.reduce_sum(points * (points @ matrix), axis=1)
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
        if bad:
            eligible &= (call != bad) | (tf.range(4) != 1)
        return value, score, eligible

    config = BatchedQuadraticCenterConfig(pilot_method="uniform_cloud", rows_per_cloud=rows,
        seed=seed, fit_half_width=width, max_fit_rounds=1 if case == "round_limit" else 4,
        centeredness_cap=1e-8)
    has_evidence = case in ("initial_evidence", "bad_evidence")
    value = -.5 * tf.reduce_sum(center * tf.linalg.matvec(matrix, center))
    score = -tf.linalg.matvec(matrix, center)
    if has_evidence:
        value += .001 if case == "bad_evidence" else 1e-12
        score += 1e-12
    return callback, config, (center, scale, tf.constant(seed), tf.constant(has_evidence), value, score)


@pytest.mark.parametrize("dimension", [1, 3, 5])
@pytest.mark.parametrize("case", CASES)
def test_complete_uniform_original_records(dimension, case, request):
    counter = tf.Variable(0, dtype=tf.int64)
    callback, config, args = fixture(dimension, case, counter=counter)
    expected, traces = reference(callback, config, args)
    expected_status = {"centered": "local_center_candidate", "round_limit": "refinement_round_limit",
        "invalid_initial": "initial_target_invalid", "invalid_replay": "replay_mismatch",
        "invalid_fit": "curvature_target_invalid", "invalid_check": "curvature_target_invalid",
        "invalid_final_replay": "replay_mismatch", "replay_value": "replay_mismatch",
        "bad_evidence": "localizer_refinement_replay_mismatch", "nonspd": "quadratic_model_rejected",
        "nonfinite_scaled_score": "nonfinite_scaled_score", "factor_failure": "pilot_factorization_failed"}.get(case)
    if expected_status is not None:
        assert expected['status'] == expected_status
    calls = int(counter)
    results = {}
    for mode in ("graph", "xla"):
        counter.assign(0)
        program = make_quadratic_controller(callback, dimension, config, jit_compile=mode == "xla")
        computed = program(*args)
        actual = materialize(computed, config, dimension, int(args[2]))
        results[mode] = actual
        assert int(counter) == calls == int(computed['callback_batches'])
        assert int(computed['physical_rows']) <= config.planned_rows(dimension)
        assert program.experimental_get_tracing_count() == 1
        _equal_records(actual, materialize(computed, config, dimension, int(args[2]), bulk=False))
        if traces:
            assert traces == {"fit_trace_count": int(computed["fit_calls"] > 0),
                              "trust_trace_count": int(computed["trust_calls"] > 0)}
    directory = Path(request.config.getoption("xmlpath")).parent
    with (directory / f"uniform-rounds-{case}-{dimension}.json").open("x") as handle:
        json.dump({"original": expected, **results, "original_trace_counts": traces,
            "baseline": "3582b4ac", "source_sha256": original()[0].hashes()}, handle, indent=2, allow_nan=False)
        handle.write("\n")
    for actual in results.values():
        _equal_records(actual, expected)


@pytest.mark.parametrize("dimension", [3, 5])
def test_uniform_changed_inputs_partial_batches_and_hlo(dimension, request):
    callback, config, args = fixture(dimension, 'nonquadratic', rows=33, seed=-1729, width=.03)
    kernel = make_quadratic_controller(callback, dimension, config)
    changed = (args[0] + .01, args[1] * 1.1, args[2] + 1, *args[3:])
    outputs, hlos = [], []
    for values in (args, changed):
        actual = materialize(kernel(*values), config, dimension, int(values[2]))
        expected, _ = reference(callback, config, values)
        _equal_records(actual, expected)
        outputs.append(actual)
        hlos.append(kernel.experimental_get_compiler_ir(*values)(stage='hlo'))
    assert outputs[0] != outputs[1]
    assert hlos[0] == hlos[1]
    assert kernel.experimental_get_tracing_count() == 1
    concrete = kernel.get_concrete_function()
    # The callback legitimately captures its fixed target matrix. Dynamic
    # center/scale/seed/evidence must remain distinct runtime operands.
    assert len(concrete.captured_inputs) == 1
    tf.debugging.assert_equal(concrete.captured_inputs[0], inputs('trust', dimension)[0])
    operands = len(args) + len(concrete.captured_inputs)
    assert len(re.findall(r"\bparameter\((\d+)\)", hlos[0][hlos[0].rfind('\nENTRY '):])) == operands
    graph = concrete.graph.as_graph_def()
    nodes = [*graph.node, *(node for function in graph.library.function for node in function.node_def)]
    assert any(node.op in ('While', 'StatelessWhile') for node in nodes)
    assert not {'PyFunc', 'EagerPyFunc', 'PyFuncStateless'} & {node.op for node in nodes}
    directory = Path(request.config.getoption('xmlpath')).parent
    with (directory / f'uniform-runtime-{dimension}.json').open('x') as handle:
        json.dump({'records': outputs, 'trace_count': 1, 'runtime_operands': operands,
            'hlo_equal': True, 'graph_nodes': len(nodes),
            'baseline': '3582b4ac', 'source_sha256': original()[0].hashes()}, handle, indent=2, allow_nan=False)
        handle.write('\n')
    with (directory / f'uniform-runtime-{dimension}.hlo').open('x') as handle:
        handle.write(hlos[0])


@pytest.mark.parametrize("jit_compile", [False, True])
def test_uniform_cache_keeps_seed_dynamic_and_configuration_distinct(jit_compile):
    from dataclasses import replace

    from bayesfilter.inference.quadratic_rounds_tf import (
        clear_paired_quadratic_controller_cache,
        paired_quadratic_controller,
        quadratic_controller,
    )

    callback, config, _ = fixture(3, "centered")
    clear_paired_quadratic_controller_cache()
    first = quadratic_controller(callback, 3, config, jit_compile=jit_compile)
    assert quadratic_controller(callback, 3, replace(config, seed=-123), jit_compile=jit_compile) is first
    assert quadratic_controller(callback, 3, replace(config, rows_per_cloud=33), jit_compile=jit_compile) is not first
    paired = replace(config, pilot_method="paired_local")
    second = paired_quadratic_controller(callback, 3, paired, jit_compile=jit_compile)
    assert quadratic_controller(callback, 3, paired, jit_compile=jit_compile) is second
    assert quadratic_controller(callback, 3, config, jit_compile=not jit_compile) is not first
    clear_paired_quadratic_controller_cache()
