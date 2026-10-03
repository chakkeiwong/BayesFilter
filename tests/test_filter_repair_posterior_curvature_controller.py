"""Diagnostic frozen-statement comparisons for posterior curvature control."""

import ast
import gc
import hashlib
import weakref

import pytest
import tensorflow as tf

from bayesfilter.inference import fixed_center_curvature as fixed
from bayesfilter.inference import posterior_local_initializer as original
from bayesfilter.inference.dense_initializer_reporting import _raise_partition_error
from bayesfilter.inference.posterior_curvature_controller_tf import (
    STATUSES,
    make_posterior_curvature_program,
)
from tests.test_filter_repair_block_capture import stable_hlo
from tests.test_filter_repair_geometry_control import clean, save
from tests.test_filter_repair_posterior_movement import (
    REVISION,
    frozen_posterior_module,
    verified_reference_tree,
)
from tests.test_filter_repair_quadratic_batches import _equal_records
from tests.test_posterior_local_initializer import _thresholds

D = tf.float64
CASES = ((1, False, "stationary"), (3, True, "stationary"),
    (1, True, "recenter"), (1, True, "exhaustion"), (3, True, "nonlinear"),
    (1, False, "invalid_center"), (1, True, "invalid_partial"),
    (1, True, "budget"), (1, True, "mismatch"), (1, True, "overlap"))


def original_curvature(evaluator, config, thresholds, prepared):
    sources, hashes = verified_reference_tree()
    path = "bayesfilter/inference/posterior_local_initializer.py"
    source = sources[path]
    function = next(node for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == "initialize_posterior_local_location_scale")
    recorder = next(node for node in function.body
        if isinstance(node, ast.FunctionDef) and node.name == "record_candidate")
    loop = next(node for node in function.body if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name) and node.target.id == "curvature_attempt")
    recorder_source = "\n".join(source.splitlines()[recorder.lineno - 1:recorder.end_lineno])
    loop_source = "\n".join(source.splitlines()[loop.lineno - 1:loop.end_lineno])

    def sample(rows, dimension, *, radius, seed):
        attempt = seed[0] - config.seed[0]
        partition = seed[1] - config.seed[1] - 1000 * attempt
        return prepared[attempt, partition, :rows]

    def capture(**kwargs):
        curvature = kwargs["curvature"]
        marginal = kwargs.get("marginal")
        return {"accepted": kwargs["accepted"], "status": kwargs["status"],
            "center": kwargs["center"], "value": kwargs["center_value"], "score": kwargs["center_score"],
            "ledger": kwargs["ledger"], "attempts": kwargs["extra"]["curvature_attempts"],
            "curvature": None if curvature is None else curvature.payload(),
            "precision_z": kwargs.get("precision_z"), "covariance_z": kwargs.get("covariance_z"),
            "precision_theta": kwargs.get("precision_theta"), "covariance_theta": kwargs.get("covariance_theta"),
            "marginal": marginal, "scale_log": None if marginal is None else tf.math.log(marginal)}

    text = '''def run(initial, value, score, units):
    cfg = config
    candidates, ledger = [], []
    locator_payload, movement_rows = {}, [{}]
    move_cfg = LowRankSPDQuadraticGeometryConfig()
'''+recorder_source+'''
    record_candidate(initial, float(value), score, "initial")
    train_rows, selection_rows, audit_rows = _cloud_row_counts(cfg, int(initial.shape[0]))
    dimension = int(initial.shape[0])
    curvature_attempt_records = []
    seed = tuple(int(value) for value in cfg.seed)
'''+loop_source
    reference, _ = frozen_posterior_module()
    namespace = {**vars(reference), "evaluator": evaluator, "config": config,
        "curvature_thresholds": thresholds, "_sample_ball": sample, "_build_result": capture}
    exec(compile(text, f"{REVISION}:{path}:curvature", "exec"), namespace)  # noqa: S102
    return namespace["run"], {"revision": REVISION, "dependency_sha256": hashes,
        "recorder_sha256": hashlib.sha256(recorder_source.encode()).hexdigest(),
        "loop_sha256": hashlib.sha256(loop_source.encode()).hexdigest(),
        "adapter": "Prepared offsets and completed result capture only; original recorder/loop unchanged."}


def fixture(dimension, batched, case, config, *, precision=None):
    calls, row_count = tf.Variable(0, dtype=tf.int64), tf.Variable(0, dtype=tf.int64)
    positions = tf.Variable(tf.zeros([512, dimension], D))
    extents = tf.Variable(tf.zeros([512], tf.int64))
    if precision is None:
        precision = tf.linalg.diag(tf.cast(tf.range(dimension), D) + 2.) + .07

    def mathematical(cloud):
        delta = cloud - .13
        score = -tf.linalg.matmul(delta, precision)
        value = .5 * tf.reduce_sum(delta * score, axis=1)
        if case == "nonlinear":
            value -= .3 * tf.reduce_sum(delta ** 4, axis=1)
            score -= 1.2 * delta ** 3
        return value, score

    def batch(cloud):
        count = cloud.shape[0]
        call = calls.assign_add(1) - 1
        start = row_count.assign_add(count) - count
        updates = (positions.scatter_nd_update((start + tf.range(count, dtype=tf.int64))[:, None], cloud),
            extents.scatter_nd_update(call[None, None], tf.constant([count], tf.int64)))
        with tf.control_dependencies(updates):
            value, score = mathematical(tf.identity(cloud))
        if case == "invalid_center":
            value = tf.where(call == 0, tf.constant(float("nan"), D), value)
        if case == "invalid_partial":
            value = tf.where((call == 2) & (tf.range(count) == count - 1), tf.constant(float("nan"), D), value)
        return value, score

    def scalar(point):
        values, scores = batch(point[None])
        return values[0], scores[0]

    train, _, _ = original._cloud_row_counts(config, dimension)
    tracker = original._EligibilityTrackingEvaluator(scalar, dimension=dimension,
        max_rows=4 + train if case == "budget" else 1000, batched_fn=batch if batched else None,
        eligibility_fn=(lambda _: calls < 2) if case == "mismatch" else None,
        batched_eligibility_fn=(lambda cloud: tf.fill([cloud.shape[0]], calls < 2)) if case == "mismatch" else None)

    def reset():
        for resource in (calls, row_count, positions, extents, tracker.invalid_rows,
                tracker.mismatch_rows, tracker.budget_exhausted):
            resource.assign(tf.zeros_like(resource))
        # This stage must preserve work already charged by earlier stages.
        tracker.evaluated_rows.assign(3)

    def record():
        return clean({"positions": positions[:int(row_count)], "extents": extents[:int(calls)],
            "tracker": tracker.diagnostics()})

    return tracker, reset, record, mathematical


def complete(raw, scale, config, thresholds):
    replicates = config.replicate_count
    rows = original._cloud_row_counts(config, int(scale.shape[0]))
    names = [*[f"training[{i}]" for i in range(replicates)],
        *[f"selection[{i}]" for i in range(replicates)], "audit"]

    def seeds(attempt):
        return [(config.seed[0] + attempt, config.seed[1] + 1000 * attempt + i) for i in range(len(names))]

    records = [{"attempt": i, "center": raw["centers"][i], "partition_names": names,
        "partition_seeds": seeds(i), "random_stream": original.STREAM_ID,
        "partition_rows": [rows[0]] * replicates + [rows[1]] * replicates + [rows[2]],
        "center_moved": bool(raw["center_moved"][i]), "scaled_center_move": float(raw["scaled_moves"][i]),
        "objective_improvement": float(raw["improvements"][i]), "fit_attempted": bool(raw["fit_attempts"][i])}
        for i in range(int(raw["record_count"]))]
    ledger = []
    for i in range(int(raw["ledger_count"])):
        attempt, part = int(raw["ledger_attempts"][i]), int(raw["ledger_partitions"][i])
        name = "initial" if i == 0 else f"curvature[{attempt}]_" + (
            "center_replay" if part == -1 else f"{names[part]}_row")
        ledger.append({"ledger_index": i, "source": name, "position": raw["ledger_positions"][i],
            "value": float(raw["ledger_values"][i]), "score_l2": float(raw["ledger_scaled_score_l2"][i]),
            "promoted": bool(raw["ledger_promoted"][i])})
    curvature = None
    if bool(raw["fit_attempted"]):
        _raise_partition_error(raw["fit"]["validation"], replicates)
        curvature = fixed._fixed_center_result_from_native(raw["fit"]["fit"], raw["center"], raw["center_score_z"],
            dimension=int(scale.shape[0]), replicates=replicates, training_rows=rows[0], selection_rows=rows[1],
            audit_rows=rows[2], thresholds=thresholds, factor_max=config.factor_max,
            weights=config.shrinkage_weights, structured_target_family=config.structured_target_family,
            lineage={"role": "posterior_local_terminal_curvature", "curvature_attempt": int(raw["attempt_count"]) - 1,
                "partition_seeds": seeds(int(raw["attempt_count"]) - 1), "random_stream": original.STREAM_ID,
                "fit_center_equals_exact_incumbent": True})
    accepted = int(raw["status"]) == 1
    status = STATUSES[int(raw["status"])]
    if status == "curvature_result":
        status = f"curvature_{curvature.status}"
    return clean({"accepted": accepted, "status": status, "center": raw["center"],
        "value": raw["value"], "score": raw["score"], "ledger": ledger, "attempts": records,
        "curvature": None if curvature is None else curvature.payload(),
        "precision_z": raw["fit"]["fit"]["selection"]["precision"] if accepted else None,
        "covariance_z": raw["fit"]["fit"]["covariance"] if accepted else None,
        "precision_theta": raw["precision_theta"] if accepted else None,
        "covariance_theta": raw["covariance_theta"] if accepted else None,
        "marginal": raw["marginal"] if accepted else None, "scale_log": raw["scale_log"] if accepted else None})


def outcome(call, *args):
    try:
        return clean(call(*args))
    except ValueError as error:
        return {"error": {"type": type(error).__name__, "message": str(error)}}


@pytest.mark.parametrize("dimension,batched,case", CASES)
def test_prepared_curvature_records(dimension, batched, case, request):
    cfg = original.PosteriorLocalInitializerConfig(max_curvature_attempts=2, factor_max=1,
        training_rows_per_replicate=3 * dimension, selection_rows_per_replicate=2 * dimension,
        audit_rows=2 * dimension, seed=(42, 19))
    thresholds = _thresholds(dimension)
    rows = original._cloud_row_counts(cfg, dimension)
    counts = [rows[0]] * 2 + [rows[1]] * 2 + [rows[2]]
    capacity = max(counts)
    prepared = tf.stack([tf.stack([tf.pad(original._sample_ball(n, dimension, radius=.08,
        seed=(42 + a, 19 + 1000 * a + i)), [[0, capacity - n], [0, 0]])
        for i, n in enumerate(counts)]) for a in range(2)])
    tracker, reset, record, mathematical = fixture(dimension, batched, case, cfg)
    program = make_posterior_curvature_program(tracker, dimension, cfg, thresholds)
    observations = []
    for shift in (0., .01, 0.):
        initial = tf.fill([dimension], tf.constant(-.2 if case in ("recenter", "exhaustion") else .13, D))
        scale = tf.cast(tf.range(dimension), D) * .1 + .8 + shift
        value, score = mathematical(initial[None])
        value, score = value[0], score[0]
        clouds = tf.identity(prepared)
        if case == "recenter":
            clouds = tf.tensor_scatter_nd_update(clouds, [[0, 0, 0]], ((.13 - initial) / scale)[None])
        if case == "overlap":
            clouds = tf.tensor_scatter_nd_update(clouds, [[0, 1, 0]], clouds[0, 0, 0][None])
        reference, provenance = original_curvature(tracker, cfg, thresholds, clouds)
        reset()
        expected = outcome(reference, initial, value, score, scale)
        expected_calls = record()
        reset()
        with tf.GradientTape() as tape:
            tape.watch((initial, value, score, scale, clouds))
            raw = program(initial, value, score, scale, clouds)
            returned = raw["value"] + tf.reduce_sum(raw["center"] + raw["score"] + raw["marginal"])
        assert tape.gradient(returned, (initial, value, score, scale, clouds)) == (None,) * 5
        actual = outcome(complete, raw, scale, cfg, thresholds)
        observations.append({"expected": expected, "actual": actual,
            "expected_calls": expected_calls, "actual_calls": record(),
            "native_status": STATUSES[int(raw["status"])], "fit_attempted": bool(raw["fit_attempted"]),
            "attempt_count": int(raw["attempt_count"])})
    save(request, f"posterior-curvature-{dimension}-{batched}-{case}.json", {
        "comparisons": observations, "provenance": provenance, "public_installed": False})
    for observation in observations:
        _equal_records(observation["actual"], observation["expected"])
        _equal_records(observation["actual_calls"], observation["expected_calls"])
    assert observations[0] == observations[2]
    assert program.experimental_get_tracing_count() == 1
    hlo = stable_hlo(program.experimental_get_compiler_ir(initial, value, score, scale, clouds)(stage="hlo"))
    changed = stable_hlo(program.experimental_get_compiler_ir(initial, value, score, scale + .01, clouds)(stage="hlo"))
    assert hlo == changed
    if case == "recenter":
        assert observations[0]["attempt_count"] == 2 and observations[0]["fit_attempted"]
    if case == "exhaustion":
        assert observations[0]["native_status"] == "curvature_not_centered_within_attempt_budget"
    actual, calls = observations[0]["actual"], observations[0]["actual_calls"]
    if case == "invalid_partial":
        assert actual["status"] == "curvature_cloud_invalid"
        assert len(actual["ledger"]) == 2 + rows[0]
        assert calls["extents"] == [1, rows[0], rows[0]]
        assert calls["tracker"]["invalid_rows"] == 1
    if case == "budget":
        assert actual["status"] == "exact_evaluation_budget_exhausted"
        assert len(actual["ledger"]) == 2 + rows[0]
        assert calls["extents"] == [1, rows[0]]
        assert calls["tracker"]["evaluated_rows"] == 4 + rows[0]
    if case == "mismatch":
        assert actual["status"] == "eligibility_contract_mismatch"
        assert len(actual["ledger"]) == 2 and calls["tracker"]["mismatch_rows"] == rows[0]
    if case == "stationary":
        assert actual["accepted"] and observations[0]["attempt_count"] == 1


def test_compiler_error_and_owner_release(request):
    count = tf.Variable(0, dtype=tf.int64)
    eager = []

    def callback(point):
        if tf.executing_eagerly():
            eager.append(True)
        update = count.assign_add(1)
        with tf.control_dependencies([update]):
            value = tf.strings.to_number(tf.strings.as_string(point[0], precision=17), out_type=D)
            return -.5 * value ** 2, -point

    tracker = original._EligibilityTrackingEvaluator(callback, dimension=1, max_rows=100,
        batched_fn=None, eligibility_fn=None, batched_eligibility_fn=None)
    cfg = original.PosteriorLocalInitializerConfig(factor_max=1)
    program = make_posterior_curvature_program(tracker, 1, cfg, _thresholds(1))
    program_reference, scope_reference = weakref.ref(program), weakref.ref(program.dependency_scope)
    with pytest.raises(tf.errors.OpError, match="AsString|StringToNumber") as caught:
        program(tf.constant([.13], D), tf.constant(0., D), tf.constant([0.], D),
            tf.constant([.8], D), tf.zeros([2, 5, 2, 1], D))
    assert not eager and int(count) == int(tracker.evaluated_rows) == 0
    error_type = type(caught.value).__name__
    del caught, program
    gc.collect()
    assert program_reference() is None and scope_reference() is None
    save(request, "posterior-curvature-compiler-owner.json", {"error_type": error_type,
        "eager_calls": eager, "target_rows": int(count), "tracker": tracker.diagnostics(),
        "python_owner_collected": True, "native_executable_eviction_proved": False})
