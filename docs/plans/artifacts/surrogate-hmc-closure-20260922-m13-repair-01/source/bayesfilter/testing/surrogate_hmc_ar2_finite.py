"""Diagnostic preflight for the unchanged AR2 finite DCTR likelihood.

This module can nominate a quadrature design, never certify a posterior from
probe values. Numerical failures remain typed failures, not support exclusions.
"""
from pathlib import Path
import json
import resource
import time

import tensorflow as tf

from bayesfilter.inference.surrogate_hmc_equivalence import finite_report
from bayesfilter.inference.surrogate_hmc_program import ROOT, file_hash, write_json
from bayesfilter.testing.surrogate_hmc_ar2_reference import PRIOR_SD
from bayesfilter.testing.surrogate_hmc_ar2_sampling import load_reference, selected_consumer
from bayesfilter.highdim.ledh_failure_status_tf import FailureCode, FailureStage
from bayesfilter.highdim.ledh_structural_ar2_target_tf import make_ar2_target


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


def finite_endpoint_validity(values, status):
    """Match the public HMC endpoint's derivation from raw canonical status.

    numerical_failure is constructed by the HMC adapter; it is not a required
    field of the canonical evaluator. Force validity is a separate diagnostic.
    """
    return status["value_valid"] & tf.math.is_finite(values) & ~status["outside_declared_support"]


def preflight_stage(output):
    directory = Path(output).parent
    directory.mkdir(parents=True, exist_ok=True)
    data, reference, arrays = load_reference()
    consumer = selected_consumer(data)
    points, labels = probe_design(arrays)
    values, exact, status, timings, wiring = scan_batches(consumer, points)
    mechanics = independence_checks(consumer, points[49:53])
    endpoint_valid = finite_endpoint_validity(values, status)
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


FINITE_PREVIOUS_ROOT = ROOT / "docs/plans/artifacts/surrogate-hmc-closure-20260922-ar2-finite-02/ar2-finite-preflight"


def classify_precision(f32_valid, f64_valid, failed_probe_count):
    f32 = [bool(value) for value in f32_valid[:failed_probe_count]]
    f64 = [bool(value) for value in f64_valid[:failed_probe_count]]
    if all((not left) and right for left, right in zip(f32, f64)):
        return "FP32_PRECISION_SENSITIVITY"
    if any((not left) and (not right) for left, right in zip(f32, f64)):
        return "BOTH_PRECISIONS_CORRECTION_FAILURE"
    if all(f32):
        return "REPLAY_NO_LONGER_FAILS"
    return "MIXED_PRECISION_RESULT"


def _enum_name(enum, value):
    value = int(value)
    return next((member.name for member in enum if int(member) == value), f"UNKNOWN_{value}")


def _tensor_record(directory, name, tensor):
    path = directory / (name + ".tensor")
    tf.io.write_file(str(path), tf.io.serialize_tensor(tensor))
    return dict(path=path.name, sha256=file_hash(path), dtype=tensor.dtype.name,
                shape=tensor.shape.as_list())


def _target_diagnostics(target, points):
    """Evaluate one dtype arm with the full filter-moment observability path."""
    points = tf.cast(points, target.inputs["initial_states"].dtype)
    batch_size = 4
    function = target.compile(batch_size, filter_moments=True)
    values, statuses = [], []
    for start in range(0, int(points.shape[0]), batch_size):
        batch = points[start:start + batch_size]
        count = int(batch.shape[0])
        if count < batch_size:
            batch = tf.concat([batch, tf.repeat(batch[:1], batch_size-count, 0)], 0)
        value, _, status = function(batch)
        values.append(value[:count])
        statuses.append({key: tensor[:count] for key, tensor in status.items()})
    merged = {key: tf.concat([row[key] for row in statuses], 0)
              for key in statuses[0]}
    return tf.concat(values, 0), merged


def _min_eigenvalues(covariance):
    eigenvalues = tf.linalg.eigvalsh(tf.cast(covariance, tf.float64))
    return tf.reduce_min(eigenvalues, axis=list(range(1, eigenvalues.shape.rank))).numpy().tolist()


def _arm_rows(values, status, original_indices):
    rows = []
    min_filter = _min_eigenvalues(status["filtering_covariance"])
    min_post = _min_eigenvalues(status["post_reset_covariance"])
    for row, original in enumerate(original_indices):
        rows.append(dict(
            index=int(original),
            value=float(values[row]),
            value_finite=bool(tf.math.is_finite(values[row])),
            value_valid=bool(status["value_valid"][row]),
            force_valid=bool(status["force_valid"][row]),
            program_valid=bool(status["program_valid"][row]),
            value_failure_code=int(status["value_failure_code"][row]),
            value_failure_name=_enum_name(FailureCode, status["value_failure_code"][row]),
            force_failure_code=int(status["force_failure_code"][row]),
            force_failure_name=_enum_name(FailureCode, status["force_failure_code"][row]),
            first_failure_stage=int(status["first_failure_stage"][row]),
            first_failure_stage_name=_enum_name(FailureStage, status["first_failure_stage"][row]),
            first_failure_time=int(status["first_failure_time"][row]),
            first_failure_particle=int(status["first_failure_particle"][row]),
            filtering_min_eigenvalue=float(min_filter[row]),
            post_reset_min_eigenvalue=float(min_post[row]),
            filtering_ess_min=float(tf.reduce_min(status["filtering_ess"][row])),
            correction_cap_fraction_max=float(tf.reduce_max(
                status["higher_moment_fraction_coordinatewise_cap_active"][row])),
            correction_post_cap_absolute_max=float(tf.reduce_max(
                status["higher_moment_maximum_coordinatewise_post_cap_absolute"][row])),
            correction_cap_derivative_min=float(tf.reduce_min(
                status["higher_moment_minimum_coordinatewise_cap_derivative"][row])),
        ))
    return rows


def localization_stage(output):
    """Localize finite-endpoint failures without changing the claim target."""
    previous_report_path = FINITE_PREVIOUS_ROOT / "report.json"
    previous_points_path = FINITE_PREVIOUS_ROOT / "points.tensor"
    previous_report = json.loads(previous_report_path.read_text())
    if previous_report.get("status") != "AR2_FINITE_PREFLIGHT_REPAIR_REQUIRED":
        raise ValueError("finite-02 repair-required report is required")
    expected_hash = previous_report["tensors"]["points"]["sha256"]
    if file_hash(previous_points_path) != expected_hash:
        raise ValueError("finite-02 probe tensor hash changed")
    points = tf.io.parse_tensor(tf.io.read_file(str(previous_points_path)), tf.float64)
    invalid_indices = [int(i) for i in previous_report["finite_invalid_indices"]]
    valid_indices = [i for i in range(int(points.shape[0])) if i not in invalid_indices]
    # Three healthy controls expose whether the replay itself is stable.
    selected_indices = invalid_indices + valid_indices[:3]
    probe = tf.cast(tf.cast(tf.gather(points, tf.constant(selected_indices, tf.int32)),
                            tf.float32), tf.float64)
    data, reference, _ = load_reference()
    consumer = selected_consumer(data)
    f64_target = make_ar2_target(data, particle_count=64,
                                 controls=dict(consumer.fixture.controls), dtype=tf.float64)
    f32_value, f32_status = _target_diagnostics(consumer.fixture, probe)
    f64_value, f64_status = _target_diagnostics(f64_target, probe)

    same_design = {}
    for name in ("initial_states", "initial_covariances", "noises", "observations"):
        left, right = consumer.fixture.inputs[name], f64_target.inputs[name]
        same_design[name] = bool(tf.reduce_all(tf.equal(tf.cast(left, tf.float64), right)))
    same_design["reset_design"] = bool(tf.reduce_all(tf.equal(
        tf.cast(consumer.fixture.controls["reset_design"], tf.float64),
        f64_target.controls["reset_design"])))
    settings = {key: value for key, value in consumer.fixture.controls.items() if key != "reset_design"}
    f64_settings = {key: value for key, value in f64_target.controls.items() if key != "reset_design"}
    same_settings = settings == f64_settings
    base_values, _, base_status, _, endpoint_wiring = scan_batches(consumer, probe)
    status_agrees = all(bool(tf.reduce_all(tf.equal(base_status[key], f32_status[key])))
                        for key in base_status)
    same_values = tf.equal(base_values, tf.cast(f32_value, tf.float64))
    finite_pair = tf.math.is_finite(base_values) & tf.math.is_finite(f32_value)
    relative_error = tf.where(finite_pair,
        tf.abs(base_values-tf.cast(f32_value, tf.float64))/(1+tf.abs(base_values)), 0.)
    # An observability graph may change XLA rounding. Require a recorded FP32
    # rounding allowance and identical typed status, never validity fallback.
    value_agrees = bool(tf.reduce_all(same_values | (finite_pair & (relative_error <= 2e-5))))
    finite_observability = all(bool(tf.reduce_all(tf.math.is_finite(tensor)))
        for status in (f32_status, f64_status) for tensor in status.values()
        if tensor.dtype.is_floating)
    replay_failed = bool(tf.reduce_all(~f32_status["value_valid"][:len(invalid_indices)]))
    healthy_controls = bool(tf.reduce_all(f32_status["value_valid"][len(invalid_indices):]))
    f32_valid = f32_status["value_valid"] & tf.math.is_finite(f32_value)
    f64_valid = f64_status["value_valid"] & tf.math.is_finite(f64_value)
    category = classify_precision(f32_valid.numpy().tolist(), f64_valid.numpy().tolist(),
                                  len(invalid_indices))
    directory = Path(output).parent
    directory.mkdir(parents=True, exist_ok=True)
    tensors = {"rounded_probe_points": _tensor_record(directory, "rounded_probe_points", probe)}
    for arm, values, status in (("fp32", f32_value, f32_status), ("fp64", f64_value, f64_status)):
        tensors[arm + "_value"] = _tensor_record(directory, arm + "_value", values)
        for key, tensor in status.items():
            # Keep all typed status and observability arrays; they are small at ten steps.
            tensors[arm + "_" + key] = _tensor_record(directory, arm + "_" + key, tensor)
    rows32 = _arm_rows(f32_value, f32_status, selected_indices)
    rows64 = _arm_rows(f64_value, f64_status, selected_indices)
    report = dict(
        status="AR2_FINITE_LOCALIZATION_COMPLETED",
        qualified=bool(same_settings and all(same_design.values()) and endpoint_wiring
                       and status_agrees and value_agrees and finite_observability
                       and replay_failed and healthy_controls),
        qualification_scope="failure localization and precision comparison only; no finite posterior qualification",
        category=category,
        selected_indices=selected_indices,
        failed_probe_count=len(invalid_indices),
        fp32_rows=rows32,
        fp64_rows=rows64,
        same_particle_noise_observation_design=same_design,
        same_non_design_controls=same_settings,
        previous_preflight_report_sha256=file_hash(previous_report_path),
        previous_probe_tensor_sha256=expected_hash,
        current_consumer=consumer.manifest,
        fp64_manifest=dict(f64_target.manifest),
        tensors=tensors,
        reference_qualified=reference["qualified"],
        mechanics=dict(same_design=all(same_design.values()), same_controls=same_settings,
            endpoint_wiring=endpoint_wiring, instrumentation_status_agrees=status_agrees,
            instrumentation_value_agrees=value_agrees,
            instrumentation_max_scaled_value_error=float(tf.reduce_max(relative_error)),
            instrumentation_rounding_allowance=2e-5, finite_observability=finite_observability,
            failed_probe_reproduced=replay_failed, healthy_controls_valid=healthy_controls),
        finite_posterior_comparison="NOT_RUN",
        numerical_failures_are_support_boundaries=False,
        inference_launched=False,
        all_models_validated=False,
        next_action=(
            "Existing master: inspect correction margins and decide whether a scoped numerical repair "
            "or a justified integration-domain analysis is warranted"
        ),
    )
    if tf.config.list_logical_devices("GPU"):
        report["allocator"] = tf.config.experimental.get_memory_info("GPU:0")
    return finite_report(report)
