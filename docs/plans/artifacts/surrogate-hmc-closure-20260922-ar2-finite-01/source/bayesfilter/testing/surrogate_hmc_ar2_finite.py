"""Diagnostic preflight for the unchanged AR2 finite DCTR likelihood.

This module can nominate a quadrature design, never certify a posterior from
probe values. Numerical failures remain typed failures, not support exclusions.
"""
from pathlib import Path
import resource
import time

import tensorflow as tf

from bayesfilter.inference.surrogate_hmc_equivalence import finite_report
from bayesfilter.inference.surrogate_hmc_program import file_hash, write_json
from bayesfilter.testing.surrogate_hmc_ar2_reference import PRIOR_SD
from bayesfilter.testing.surrogate_hmc_ar2_sampling import load_reference, selected_consumer


def _cpu():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


def probe_design(arrays):
    axis = tf.constant([-6., -4., -2., 0., 2., 4., 6.], tf.float64) * PRIOR_SD
    prior = tf.reshape(tf.stack(tf.meshgrid(axis, axis, indexing="ij"), -1), [-1, 2])
    local = tf.constant([-2., 0., 2.], tf.float64)
    offsets = tf.reshape(tf.stack(tf.meshgrid(local, local, indexing="ij"), -1), [-1, 2])
    points = tf.concat([prior, arrays["means"] + offsets * arrays["scales"]], 0)
    labels = ["prior_center" if abs(i) <= 2 and abs(j) <= 2 else "prior_tail"
              for i in (-6, -4, -2, 0, 2, 4, 6) for j in (-6, -4, -2, 0, 2, 4, 6)]
    return points, labels + ["posterior_bulk"] * 9


def scan_batches(consumer, points):
    """Use the actual batch-four force/value and finite-endpoint call chain."""
    values, exact_values, statuses, timings = [], [], [], []
    wiring = True
    for start in range(0, int(points.shape[0]), 4):
        batch = points[start:start+4]
        count = int(batch.shape[0])
        batch = tf.concat([batch, tf.repeat(batch[:1], 4-count, 0)], 0)
        wall, cpu = time.monotonic(), _cpu()
        value, _, status = consumer.evaluate_force(batch)
        for tensor in tf.nest.flatten((value, status)):
            tensor.numpy()
        timings.append(dict(wall_seconds=time.monotonic()-wall, cpu_worker_seconds=_cpu()-cpu,
                            compilation_included=start == 0))
        endpoint = consumer.finite_endpoint.function(batch)
        wiring &= bool(tf.reduce_all(tf.equal(endpoint.potential, -tf.cast(value, tf.float64))))
        wiring &= set(endpoint.status) == set(status)
        wiring &= all(bool(tf.reduce_all(tf.equal(endpoint.status[key], status[key]))) for key in status)
        exact, _ = consumer.evaluate_reference(batch)
        values.append(tf.cast(value[:count], tf.float64))
        exact_values.append(exact[:count])
        statuses.append({key: tensor[:count] for key, tensor in status.items()})
    return (tf.concat(values, 0), tf.concat(exact_values, 0),
            {key: tf.concat([row[key] for row in statuses], 0) for key in statuses[0]},
            timings, wiring)


def equality_check(left, right):
    # Equality of failure placeholders is recorded, but cannot establish validity.
    return all(bool(tf.reduce_all(tf.equal(a, b)))
               for a, b in zip(tf.nest.flatten(left), tf.nest.flatten(right)))


def independence_checks(consumer, points):
    base = consumer.evaluate_force(points)
    repeated = consumer.evaluate_force(points)
    permutation = tf.constant([2, 0, 3, 1])
    permuted = consumer.evaluate_force(tf.gather(points, permutation))
    expected = tf.nest.map_structure(lambda tensor: tf.gather(tensor, permutation), base)
    independent = True
    for index in range(4):
        duplicated = consumer.evaluate_force(tf.repeat(points[index:index+1], 4, 0))
        same = tf.nest.map_structure(lambda tensor: tf.repeat(tensor[index:index+1], 4, 0), base)
        independent &= equality_check(duplicated, same)
    return dict(reproducible=equality_check(base, repeated),
                row_permutation_invariant=equality_check(permuted, expected),
                row_independent=independent)


def preflight_stage(output):
    directory = Path(output).parent
    directory.mkdir(parents=True, exist_ok=True)
    data, reference, arrays = load_reference()
    consumer = selected_consumer(data)
    points, labels = probe_design(arrays)
    values, exact, status, timings, wiring = scan_batches(consumer, points)
    mechanics = independence_checks(consumer, points[49:53])
    endpoint_valid = (status["value_valid"] & tf.math.is_finite(values)
                      & ~status["outside_declared_support"] & ~status["numerical_failure"])
    exact_valid = tf.math.is_finite(exact)
    group_reports = {}
    for label in ("prior_center", "prior_tail", "posterior_bulk"):
        mask = tf.constant([item == label for item in labels])
        valid = mask & endpoint_valid & exact_valid
        difference = tf.boolean_mask(values-exact, valid)
        group_reports[label] = dict(count=labels.count(label),
            valid_endpoints=int(tf.reduce_sum(tf.cast(mask & endpoint_valid, tf.int32))),
            valid_forces=int(tf.reduce_sum(tf.cast(mask & status["force_valid"], tf.int32))),
            log_density_difference_range=(
                [float(tf.reduce_min(difference)), float(tf.reduce_max(difference))]
                if int(tf.size(difference)) else None))
    steady = timings[1:]
    grid_points = sum((intervals+1)**2 for intervals in (64, 128, 256, 384))
    batches = (grid_points+3)//4
    forecast = {key: timings[0][key] + 1.25*batches*max(row[key] for row in steady)
                for key in ("cpu_worker_seconds", "wall_seconds")}
    records = {}
    for name, tensor in dict(points=points, finite_log_posterior=values, exact_log_posterior=exact,
                             **{"status_"+key: value for key, value in status.items()}).items():
        path = directory/(name+".tensor")
        tf.io.write_file(str(path), tf.io.serialize_tensor(tensor))
        records[name] = dict(path=path.name, sha256=file_hash(path),
                             dtype=tensor.dtype.name, shape=tensor.shape.as_list())
    write_json(directory/"consumer.json", consumer.manifest)
    write_json(directory/"data.json", data)
    passed = wiring and all(mechanics.values()) and bool(tf.reduce_all(endpoint_valid & exact_valid))
    report = dict(status="AR2_FINITE_PREFLIGHT_PASSED" if passed else "AR2_FINITE_PREFLIGHT_REPAIR_REQUIRED",
        qualified=passed, qualification_scope="domain probes, endpoint wiring and cost only",
        groups=group_reports, labels=labels, tensors=records, timings=timings,
        finite_endpoint_wiring=wiring, mechanics=mechanics,
        finite_invalid_indices=tf.reshape(tf.where(~endpoint_valid), [-1]).numpy().tolist(),
        exact_invalid_indices=tf.reshape(tf.where(~exact_valid), [-1]).numpy().tolist(),
        reference_qualified=reference["qualified"], data_sha256=file_hash(directory/"data.json"),
        consumer_sha256=file_hash(directory/"consumer.json"),
        grid_forecast=dict(points=grid_points, batch_size=4, safety_factor=1.25, **forecast),
        inference_launched=False, posterior_comparisons={"D/R": "NOT_RUN", "C/D": "NOT_RUN", "C/R": "NOT_RUN"},
        numerical_failures_are_support_boundaries=False, all_models_validated=False,
        next_action=("Existing master: qualify and allocate the finite-posterior reference within budget"
                     if passed else "Existing master: localize finite-endpoint failures before allocating the reference"))
    if tf.config.list_logical_devices("GPU"):
        report["allocator"] = tf.config.experimental.get_memory_info("GPU:0")
    return finite_report(report)
