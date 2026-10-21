"""Resume the bounded M13 RWM comparator from its preserved archive.

The resume path is a resource repair. It does not change the endpoint, target,
proposal, seed schedule, looks, or assessment contract of the interrupted
comparator.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import tensorflow as tf

from bayesfilter.inference.surrogate_hmc_program import ROOT, file_hash, write_json
from bayesfilter.testing import surrogate_hmc_m13_endpoint_mechanics as mechanics
from bayesfilter.testing.surrogate_hmc_m13_endpoint_materiality import _load
from bayesfilter.testing.surrogate_hmc_m13_four_chain_cost import batched_endpoint
from bayesfilter.testing.surrogate_hmc_m13_posterior_assessment import assess, F64, LOOKS
from bayesfilter.testing.surrogate_hmc_m13_posterior_reference import load_reference, tensor_record
from bayesfilter.testing.surrogate_hmc_m13_rwm_baseline import (
    STARTS, SEED, PRIOR_SD, PLAN, archive, conditional_table, health, random_walk_graph,
)


EXPECTED = {
    "metric-warmup": 512,
    "step-warmup": 512,
    "fresh-warmup": 500,
    "retained-00": 250,
}


def _read_tensor(path: Path, dtype):
    value = tf.io.parse_tensor(tf.io.read_file(str(path)), dtype)
    if dtype != tf.bool:
        tf.debugging.assert_all_finite(value, "nonfinite archived RWM tensor")
    return value


def _load_chunk(directory: Path, expected_count: int) -> dict:
    if not directory.is_dir():
        raise ValueError(f"missing archived RWM chunk: {directory}")
    names = ("final_state", "final_potential", "initial_state", "initial_valid",
             "log_accept", "log_scale", "proposals", "states", "accepted", "valid")
    bool_names = {"initial_valid", "accepted", "valid"}
    values = {name: _read_tensor(directory / f"{name}.tfbin", tf.bool if name in bool_names else tf.float64) for name in names}
    if values["states"].shape.as_list() != [expected_count, 4, 2]:
        raise ValueError("archived RWM state shape mismatch")
    if values["proposals"].shape.as_list() != [expected_count, 4, 2]:
        raise ValueError("archived RWM proposal shape mismatch")
    for name in ("accepted", "valid", "log_accept"):
        if values[name].shape.as_list() != [expected_count, 4]:
            raise ValueError(f"archived RWM {name} shape mismatch")
    if values["initial_state"].shape.as_list() != [4, 2] or values["final_state"].shape.as_list() != [4, 2]:
        raise ValueError("archived RWM boundary shape mismatch")
    if values["final_potential"].shape.as_list() != [4] or values["initial_valid"].shape.as_list() != [4]:
        raise ValueError("archived RWM validity shape mismatch")
    if values["log_scale"].shape.as_list() != []:
        raise ValueError("archived RWM scale trace shape mismatch")
    if values["accepted"].dtype != tf.bool or values["valid"].dtype != tf.bool or values["initial_valid"].dtype != tf.bool:
        raise ValueError("archived RWM boolean dtype mismatch")
    tf.debugging.assert_equal(values["final_state"], values["states"][-1], message="archived RWM final state mismatch")
    return values


def _copy_and_index(previous: Path, output: Path) -> dict:
    old = previous / "worker" / "rwm"
    target = output / "rwm"
    target.mkdir(parents=True, exist_ok=False)
    records = {}
    for name, count in EXPECTED.items():
        source = old / name
        destination = target / name
        shutil.copytree(source, destination)
        chunk = _load_chunk(destination, count)
        records[name] = {key: tensor_record(destination / f"{key}.tfbin", value)
                         for key, value in chunk.items()}
    return records


def _same_tensor(a, b, message):
    tf.debugging.assert_equal(a, b, message=message)


def run_resume(output_dir, previous_dir, reference_path, coverage_path):
    output = Path(output_dir)
    previous = Path(previous_dir)
    runner_files = {"command.log", "run-manifest.json"}
    if output.exists() and any(item.name not in runner_files for item in output.iterdir()):
        raise FileExistsError(output)
    previous_manifest = json.loads((previous / "run-manifest.json").read_text())
    previous_report = json.loads((previous / "report.json").read_text())
    if previous_report.get("status") != "JOB_CPU_BUDGET_LIMIT":
        raise ValueError("resume source is not the declared CPU-budget stop")
    old_source = previous_manifest["source_hashes"]["bayesfilter/testing/surrogate_hmc_m13_rwm_baseline.py"]
    current_source = file_hash(ROOT / "bayesfilter/testing/surrogate_hmc_m13_rwm_baseline.py")
    if old_source != current_source:
        raise ValueError("interrupted RWM numerical source changed")
    reference, _, _ = load_reference(reference_path)
    calibration = json.loads(Path(coverage_path).read_text())
    if calibration.get("qualified") is not True or calibration.get("reference_sha256") != reference["reference_sha256"]:
        raise ValueError("RWM resume requires the same qualified reference coverage")
    _, fixture, _ = _load()
    if fixture.observation_checksum != reference["observation_checksum"]:
        raise ValueError("RWM resume/reference data mismatch")

    records = _copy_and_index(previous, output)
    old = output / "rwm"
    metric = _load_chunk(old / "metric-warmup", 512)
    step = _load_chunk(old / "step-warmup", 512)
    fresh = _load_chunk(old / "fresh-warmup", 500)
    first = _load_chunk(old / "retained-00", 250)
    if not all(health(chunk) for chunk in (metric, step, fresh, first)):
        raise ValueError("preserved RWM archive failed health validation")
    _same_tensor(step["initial_state"], metric["final_state"], "warmup state discontinuity")
    _same_tensor(fresh["initial_state"], tf.constant(STARTS, F64) * PRIOR_SD,
                 "fresh warmup start mismatch")
    _same_tensor(first["initial_state"], fresh["final_state"], "retained state discontinuity")

    _same_tensor(metric["initial_state"], tf.constant(STARTS, F64) * PRIOR_SD, "metric warmup start mismatch")
    _same_tensor(fresh["log_scale"], step["log_scale"], "fresh frozen scale mismatch")
    _same_tensor(first["log_scale"], step["log_scale"], "retained frozen scale mismatch")
    _, evaluator = mechanics.endpoint_components(fixture)
    endpoint = batched_endpoint(evaluator)
    metric_states = tf.reshape(metric["states"][-256:], [-1, 2])
    centered = metric_states - tf.reduce_mean(metric_states, 0)
    covariance = .9 * (tf.linalg.matmul(centered, centered, transpose_a=True) /
                       tf.cast(tf.shape(metric_states)[0] - 1, F64)) + \
                 .1 * PRIOR_SD ** 2 * tf.eye(2, dtype=F64)
    factor = tf.linalg.cholesky(covariance)
    scale = step["log_scale"]
    if not bool(tf.math.is_finite(scale)):
        raise ValueError("invalid frozen RWM scale")
    state = first["final_state"]
    chunks = [first]
    retained = random_walk_graph(endpoint, 250, adapt=False)
    assessments = []
    for index in range(1, 8):
        chunk = retained(state, scale, factor,
                         tf.random.experimental.stateless_fold_in(tf.constant(SEED), 10 + index))
        directory = old / f"retained-{index:02}"
        records[f"retained-{index:02}"] = archive(old, f"retained-{index:02}", chunk)
        if not health(chunk):
            raise ValueError("resumed RWM retained chunk health failed; archive preserved")
        _same_tensor(chunk["initial_state"], state, "resumed RWM state continuity failure")
        state = chunk["final_state"]
        chunks.append(chunk)
        count = 250 * (index + 1)
        if count in LOOKS:
            points = tf.transpose(tf.concat([c["states"] for c in chunks], 0), [1, 0, 2])
            assessments.append(assess(points, reference, calibration=calibration))
    acceptance = tf.reduce_mean(tf.cast(tf.concat([c["accepted"] for c in chunks], 0), F64), 0)
    report = dict(plan=PLAN, status="M13_RWM_BASELINE_COMPLETE", health_passed=True,
        qualified=assessments[-1]["status"] == "EQUIVALENT", assessments=assessments,
        acceptance_per_chain=acceptance.numpy().tolist(), conditional_behavior=conditional_table(chunks),
        heuristic_role="same frozen endpoint classical sampler comparator",
        reference_sha256=reference["reference_sha256"], coverage_sha256=file_hash(Path(coverage_path)),
        target=dict(data_seed=1704, observation_checksum=fixture.observation_checksum,
            endpoint_particles=mechanics.REFERENCE_PARTICLES, endpoint_replicates=mechanics.REFERENCE_REPLICATES,
            endpoint_seed=mechanics.REFERENCE_SEED.numpy().tolist(), prior_sd=PRIOR_SD,
            endpoint_dtype="float64", prior_support="R^2"),
        covariance=covariance.numpy().tolist(), scale=float(tf.exp(scale)),
        warmup_excluded=True, warmup_counts=[512, 512, 500], draws_per_chain=2000,
        seed=list(SEED), archives=records, resumed_from=str(previous.resolve()),
        wall_seconds=None, ranking_supported=False, numerical_backend="tensorflow_only",
        jit_compile=True, cpu_reference_exception=True,
        retained_trace_count=retained.experimental_get_tracing_count())
    write_json(old / "report.json", report)
    return report
