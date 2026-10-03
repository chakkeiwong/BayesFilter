"""Pinned cloud and fitting records for the composed XLA attempt boundary."""

import gc
import hashlib
import weakref

import pytest
import tensorflow as tf

from bayesfilter.inference import dense_initializer_attempt_tf as native
from bayesfilter.inference import fixed_center_curvature as fixed
from bayesfilter.inference import fixed_center_fitting_tf as fitting
from tests.test_filter_repair_block_capture import stable_hlo
from tests.test_filter_repair_dense_initializer_cloud import original_partition_loop
from tests.test_filter_repair_dense_validated_fit import (
    _thresholds,
    normalized,
    original_fitter,
)
from tests.test_filter_repair_geometry_control import clean, save
from tests.test_filter_repair_quadratic_batches import _equal_records

D = tf.float64


@pytest.mark.parametrize("dimension", [1, 3])
@pytest.mark.parametrize("case", ["centered", "moved", "invalid", "fit_rejected"])
def test_dense_attempt_composition_preserves_cloud_and_fit(dimension, case, monkeypatch, request):
    thresholds = _thresholds(fixed, incomplete=case == "fit_rejected")
    rows = (3 * dimension, 2 * dimension, 2 * dimension + 1)
    partition_rows = [rows[0]] * 2 + [rows[1]] * 2 + [rows[2]]
    precision = tf.linalg.diag(tf.cast(tf.range(dimension), D) + 1.3) + .07
    mode = tf.fill([dimension], tf.constant(.8 if case == "moved" else .001, D))
    # The second positional argument to tf.Variable is trainable, not dtype.
    # int32 resources are host-only under GPU XLA; int64 stays on the device.
    counter_device = "/GPU:0" if tf.config.list_physical_devices("GPU") else "/CPU:0"
    with tf.device(counter_device):
        calls = tf.Variable(0, dtype=tf.int64, trainable=False)
        fit_calls = tf.Variable(0, dtype=tf.int64, trainable=False)
    assert calls.dtype == tf.int64 and calls.device.endswith(counter_device[1:])

    def callback(points):
        calls.assign_add(1)
        delta = points - mode
        scores = -tf.linalg.matmul(delta, precision, transpose_b=True)
        values = .5 * tf.reduce_sum(delta * scores, axis=1)
        valid = tf.ones([tf.shape(points)[0]], tf.bool)
        if case == "invalid":
            valid = tf.zeros_like(valid)
        return values, scores, valid

    factory = fitting.fit_program.__wrapped__

    def counted_factory(*args, **kwargs):
        inner = factory(*args, **kwargs)

        @tf.function(input_signature=inner.input_signature, jit_compile=True, autograph=False)
        def counted(*operands):
            with tf.control_dependencies([fit_calls.assign_add(1)]):
                return inner(*operands)
        return counted

    monkeypatch.setattr(fitting.fit_program, "__wrapped__", counted_factory)
    reference, excerpt = original_partition_loop()
    original, original_hashes = original_fitter()
    original_thresholds = _thresholds(original, incomplete=case == "fit_rejected")
    program = native.make_dense_initializer_attempt_program(callback, dimension, 2,
        *rows, thresholds=thresholds)
    clouds = tf.stack([tf.pad(.13 * tf.random.stateless_normal([n, dimension], [217, index], dtype=D),
        [[0, max(rows) - n], [0, 0]]) for index, n in enumerate(partition_rows)])
    reports, hlos = [], []
    for shift in (0., .00001, 0.):
        center = tf.fill([dimension], tf.constant(shift, D))
        scale = .8 + tf.cast(tf.range(dimension), D) * .2 + shift
        center_score = -tf.linalg.matvec(precision, center - mode)
        center_value = (.5 * tf.reduce_sum((center - mode) * center_score)
            if case != "moved" else tf.constant(-100., D))
        operands = (center, scale, center_value, center_score, clouds + shift)
        calls.assign(0)
        expected_cloud = reference(callback, center, scale, center_value, partition_rows, operands[4])
        expected_calls = int(calls)
        calls.assign(0)
        fit_calls.assign(0)
        with tf.GradientTape() as tape:
            tape.watch(operands)
            raw = program(*operands)
            output = tf.reduce_sum(raw["initial_output_scale_log"])
        frozen_derivatives = [value is None for value in tape.gradient(output, operands)]
        observed = clean(raw)
        partitions, archives = [], {}
        for index in range(observed["cloud"]["partition_count"]):
            n = partition_rows[index]
            for name, key in (("positions", "positions"), ("values", "values"),
                    ("scores", "scores"), ("valid", "row_validity")):
                archives[f"attempt_0_partition_{index}_{name}"] = observed["cloud"][key][index][:n]
            if observed["cloud"]["valid"] or index < observed["cloud"]["partition_count"] - 1:
                partitions.append((clean(operands[4][index, :n]), observed["cloud"]["scaled_scores"][index][:n]))
        actual_cloud = (observed["cloud"]["valid"], observed["candidate_center"],
            observed["candidate_value"], observed["exact_evaluation_rows"], partitions, archives)
        expected_fit = actual_fit = None
        valid, _, candidate_value, _, reference_partitions, _ = expected_cloud
        moved = valid and candidate_value > float(center_value)
        expected_status = 1 if not valid else 2 if moved else None
        if valid and not moved:
            train, select, audit = reference_partitions[:2], reference_partitions[2:4], reference_partitions[4]
            fit_inputs = (center.numpy(), (center_score * scale).numpy(),
                tf.stack([row[0] for row in train]).numpy(), tf.stack([row[1] for row in train]).numpy(),
                tf.stack([row[0] for row in select]).numpy(), tf.stack([row[1] for row in select]).numpy(),
                *audit)
            expected_fit = original.fit_fixed_center_curvature(*fit_inputs, thresholds=original_thresholds,
                factor_max=2, lineage={"role": "dense_attempt_composition"}).payload()
            actual_fit = fixed._fixed_center_result_from_native(raw["fit"]["fit"], center,
                center_score * scale, dimension=dimension, replicates=2, training_rows=rows[0],
                selection_rows=rows[1], audit_rows=rows[2], thresholds=thresholds, factor_max=2,
                weights=(0., .25, .5, .75, 1.), structured_target_family=None,
                lineage={"role": "dense_attempt_composition"}).payload()
            expected_status = 6 if expected_fit["accepted"] else 4
        report = {"shift": shift, "cloud": {"actual": clean(actual_cloud), "original": clean(expected_cloud)},
            "fit": {"actual": normalized({"result": actual_fit}) if actual_fit else None,
                "original": normalized({"result": expected_fit}) if expected_fit else None},
            "status": observed["status_code"], "expected_status": expected_status,
            "fit_ran": observed["fit_ran"], "fit_calls": int(fit_calls),
            "callbacks": int(calls), "expected_callbacks": expected_calls,
            "usable": observed["usable"], "frozen_derivatives": frozen_derivatives,
            "observed": observed}
        reports.append(report)
        hlos.append(stable_hlo(program.experimental_get_compiler_ir(*operands)(stage="hlo")))
    refs = {"program": weakref.ref(program), "graph": weakref.ref(program.get_concrete_function().graph),
        "callback": weakref.ref(callback)}
    traces = program.experimental_get_tracing_count()
    del program, callback
    gc.collect()
    released = {name: ref() is None for name, ref in refs.items()}
    report = {"case": case, "dimension": dimension, "records": reports,
        "reference_excerpt_sha256": hashlib.sha256(excerpt.encode()).hexdigest(),
        "original_fit_sources": original_hashes, "trace_count": traces,
        "hlo_unchanged": len(set(hlos)) == 1, "python_released": released,
        "nonclaims": ["Prepared clouds only; no RNG, locator, public initializer or DZ5 claim."]}
    save(request, f"dense-attempt-{dimension}-{case}.json", report)
    for row in reports:
        _equal_records(row["cloud"]["actual"], row["cloud"]["original"])
        _equal_records(row["fit"]["actual"], row["fit"]["original"])
        assert row["status"] == row["expected_status"]
        assert row["fit_calls"] == int(row["fit_ran"]) == int(row["expected_status"] in (4, 6))
        assert row["callbacks"] == row["expected_callbacks"]
        assert row["usable"] is (row["status"] == 6)
        assert all(row["frozen_derivatives"])
        observed = row["observed"]
        assert observed["exact_evaluation_rows"] == (sum(partition_rows) if case != "invalid" else rows[0])
        if row["usable"]:
            covariance = tf.constant(row["fit"]["original"]["result"]["selected_covariance_z"], D)
            scale = .8 + tf.cast(tf.range(dimension), D) * .2 + row["shift"]
            expected_scale = scale * tf.sqrt(tf.linalg.diag_part(covariance))
            _equal_records(observed["marginal_standard_deviations"], clean(expected_scale))
            _equal_records(observed["initial_output_scale_log"], clean(tf.math.log(expected_scale)))
            _equal_records(observed["initial_output_shift"], clean(tf.fill([dimension], tf.constant(row["shift"], D))))
        else:
            assert observed["initial_output_scale_log"] == [0.] * dimension
            assert observed["initial_output_shift"] == [0.] * dimension
    assert traces == 1 and report["hlo_unchanged"] and all(released.values()), report
