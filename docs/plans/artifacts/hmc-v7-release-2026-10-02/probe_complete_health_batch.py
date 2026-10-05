"""Saved-data diagnostic of existing complete trial reductions; no admission."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def worker(a):
    sys.path.insert(0, str(a.source))
    from scripts.run_hmc_v7_release_prices import check_source
    source_hash = check_source(a.source)
    import tensorflow as tf
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    memory = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    from bayesfilter.inference.hmc_candidate_set_execution import (
        _tensor_from_payload, _trace_from_payload)
    from bayesfilter.inference.hmc_candidate_health import (
        native_health_program, retained_finite_program, target_status_program)
    from bayesfilter.inference.hmc_verification import (
        _acceptance_summary_function, HMCAcceptancePolicy,
        TARGET_STATUS_TELEMETRY_OPTIONAL_CONDITIONING_FIELDS)
    from bayesfilter.inference.hmc_acceptance_statistics import _trial_summary_program

    raw_bytes = a.record.read_bytes()
    record = json.loads(raw_bytes)
    assert hashlib.sha256(json.dumps(record, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest() == a.record.stem
    started = time.monotonic()
    initial = _tensor_from_payload(record["initial_state"])
    data = [(_tensor_from_payload(c["samples"]), _trace_from_payload(c["trace"]))
            for c in record["chunks"]]
    assert len(data) == 32 and all(tuple(row[0].shape) == (68, 4, 2) for row in data)
    decode_seconds = time.monotonic() - started
    policy = HMCAcceptancePolicy()
    summary_args = (tf.constant(policy.block_count, tf.int32),
                    tf.constant(policy.min_decisions_per_chain, tf.int32),
                    tf.constant(policy.chain_count, tf.int32),
                    tf.constant(policy.max_abs_log_accept_energy_proxy, tf.float64))

    def reductions(row):
        samples, trace = row
        answer = {"native": native_health_program(True)(initial, samples,
            trace["proposed_state"], trace["initial_momentum"], trace["final_momentum"],
            trace["log_accept_ratio"], trace["target_log_prob"],
            trace["proposed_target_log_prob"], trace["log_acceptance_correction"],
            trace["is_accepted"], trace["target_score_finite"]),
            "retained": retained_finite_program()(samples[3:],
                trace["log_accept_ratio"][3:], trace["target_log_prob"][3:]),
            "summary": _acceptance_summary_function()(samples[3:],
                trace["log_accept_ratio"][3:], trace["is_accepted"][3:], *summary_args),
            "scores": _trial_summary_program(True)(trace["log_accept_ratio"][3:])}
        for key in ("target_status_telemetry", "proposed_target_status_telemetry"):
            status = trace[key]
            conditioning = tuple(status[name] for name in
                TARGET_STATUS_TELEMETRY_OPTIONAL_CONDITIONING_FIELDS if name in status)
            answer[key] = target_status_program(status["status_code"].dtype.name,
                status["floor_count_value"].dtype.name,
                tuple(value.dtype.name for value in conditioning), 2)(
                    status["status_code"], status["valid_pre_regularized_score"],
                    status["floor_count_value"], *conditioning)
        return answer

    started = time.monotonic()
    expected = [reductions(row) for row in data]
    expected_host = [tf.nest.map_structure(lambda value: value.numpy().tolist(), row)
                     for row in expected]
    scalar_first_seconds = time.monotonic() - started
    signature = tf.nest.map_structure(lambda value: tf.TensorSpec(value.shape, value.dtype), expected[0])
    inputs = tf.nest.map_structure(lambda value: tf.TensorSpec([None, *value.shape], value.dtype), data[0])

    @tf.function(input_signature=[inputs], autograph=False, jit_compile=False)
    def grouped(rows):
        return tf.map_fn(reductions, rows, fn_output_signature=signature, parallel_iterations=1)

    @tf.function(input_signature=[inputs], autograph=False, jit_compile=False)
    def chain_grouped(rows):
        samples, trace = rows
        batch = tf.shape(samples)[0]
        measured = tf.reshape(tf.transpose(samples[:, 3:], [1, 0, 2, 3]), [65, batch*4, 2])
        log_accept = tf.reshape(tf.transpose(trace["log_accept_ratio"][:, 3:], [1, 0, 2]), [65, batch*4])
        accepted = tf.reshape(tf.transpose(trace["is_accepted"][:, 3:], [1, 0, 2]), [65, batch*4])
        flat = _acceptance_summary_function()(measured, log_accept, accepted,
            summary_args[0], summary_args[1], batch*4, summary_args[3])
        result = tf.map_fn(reductions, rows, fn_output_signature=signature, parallel_iterations=1)
        # Diagnostic comparison only: reductions() above still establishes the
        # other outputs. This does not time an optimized complete-health path.
        means = tf.reshape(flat["chain_means"], [batch, 4])
        realized = tf.reshape(flat["realized_by_chain"], [batch, 4])
        result["summary"] = dict(chain_means=means, pooled=tf.reduce_mean(means, axis=1),
            block_means=tf.reshape(flat["block_means"], [batch, 4, 4]),
            realized_by_chain=realized, realized=tf.reduce_mean(realized, axis=1),
            movement=tuple(tf.reshape(value, [batch, 4]) for value in flat["movement"]),
            proxy=dict(negative=tf.reshape(flat["proxy"]["negative"], [batch, 4]),
                positive=tf.reshape(flat["proxy"]["positive"], [batch, 4]),
                minimum=tf.reduce_min(trace["log_accept_ratio"][:, 3:], axis=[1, 2]),
                maximum=tf.reduce_max(trace["log_accept_ratio"][:, 3:], axis=[1, 2]),
                max_abs=tf.reduce_max(tf.abs(trace["log_accept_ratio"][:, 3:]), axis=[1, 2])))
        return result

    def pack(rows):
        return tf.nest.map_structure(lambda *values: tf.stack(values), *rows)

    started = time.monotonic()
    program = chain_grouped if a.chain_grouped else grouped
    first = program(pack(data))
    first_host = tf.nest.map_structure(lambda value: value.numpy().tolist(), first)
    grouped_first_seconds = time.monotonic() - started
    for index, reference in enumerate(expected):
        actual = tf.nest.map_structure(lambda value: value[index], first)
        for left, right in zip(tf.nest.flatten(reference), tf.nest.flatten(actual)):
            tf.debugging.assert_equal(left, right)
    write(a.output / "worker-manifest.json", dict(source_manifest_sha256=source_hash,
        record_sha256=hashlib.sha256(raw_bytes).hexdigest(), memory_policy=memory,
        tensorflow_version=tf.__version__, tf32=tf.config.experimental.tensor_float_32_execution_enabled(),
        initial_device=initial.device, result_devices=sorted({x.device for x in tf.nest.flatten(first)}),
        outer_jit_compile=False, native_and_scores_jit_compile=True,
        health_summary_jit_compile=False, native_sampling=False,
        decode_seconds=decode_seconds, scalar_first_seconds=scalar_first_seconds,
        grouped_first_seconds=grouped_first_seconds))
    rows = []
    for size in (1, 8, 32):
        expected_groups = [tf.nest.map_structure(lambda value: value.numpy().tolist(),
            pack(expected[index:index+size])) for index in range(0, len(data), size)]
        for repeat in range(3):
            for mode in (("scalar", "grouped") if repeat % 2 == 0 else ("grouped", "scalar")):
                started = time.monotonic()
                if mode == "scalar":
                    actual = [tf.nest.map_structure(lambda value: value.numpy().tolist(), reductions(row))
                              for row in data]
                else:
                    actual = []
                    for index in range(0, len(data), size):
                        values = program(pack(data[index:index + size]))
                        actual.append(tf.nest.map_structure(lambda value: value.numpy().tolist(), values))
                elapsed = time.monotonic()-started
                assert actual == (expected_host if mode == "scalar" else expected_groups)
                rows.append(dict(size=size, repeat=repeat, mode=mode,
                    wall_seconds=elapsed, exact_parity=True))
                write(a.output / "calls.json", rows)
    write(a.output / "result.json", dict(status="exact_complete_reductions_passed",
        trials=len(data), rows=rows, tracing_count=program.experimental_get_tracing_count(),
        allocator=tf.config.experimental.get_memory_info("GPU:0"), release_ready=False,
        scope="saved-data reductions only; no full-payload, invalid-input or full-price claim",
        timing_scope=("chain-grouped diagnostic still computes original reductions; no optimized cost claim"
                      if a.chain_grouped else
                      "descriptive; calls include packing/materialization, parity checks excluded")))
    return 0


def main():
    p = argparse.ArgumentParser()
    for name in ("source", "record", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--gpu", required=True)
    p.add_argument("--worker", action="store_true")
    p.add_argument("--chain-grouped", action="store_true")
    a = p.parse_args()
    for name in ("source", "record", "output"):
        setattr(a, name, getattr(a, name).resolve())
    os.environ.update(CUDA_VISIBLE_DEVICES=a.gpu, TF_FORCE_GPU_ALLOW_GROWTH="true",
        TF_NUM_INTRAOP_THREADS="2", TF_NUM_INTEROP_THREADS="1", OMP_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="1", TF_CPP_MIN_LOG_LEVEL="2", BAYESFILTER_PRELOAD_CUSTOM_OP="0")
    os.sched_setaffinity(0, {24, 25, 26, 27})
    if a.worker:
        return worker(a)
    a.output.mkdir(parents=True, exist_ok=False)
    runner = a.output / "runner.py"
    runner.write_bytes(Path(__file__).read_bytes())
    command = [sys.executable, str(runner), *sys.argv[1:], "--worker"]
    manifest = dict(command=command, started_utc=datetime.now(timezone.utc).isoformat(),
        resource="gpu", gpu_uuid=a.gpu, environment=sys.executable,
        cpu_affinity=sorted(os.sched_getaffinity(0)),
        runner_sha256=hashlib.sha256(runner.read_bytes()).hexdigest(),
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        plan="docs/plans/bayesfilter-hmc-v7-release-plan-2026-10-02.md", native_sampling=False)
    write(a.output / "manifest.json", manifest)
    started = time.monotonic()
    with (a.output / "run.log").open("x") as log:
        try:
            code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=120).returncode
        except subprocess.TimeoutExpired:
            code = 124
    receipt = dict(**manifest, exit_code=code, wall_seconds=time.monotonic()-started)
    write(a.output / "receipt.json", receipt)
    print(json.dumps({k: receipt[k] for k in ("exit_code", "wall_seconds")}))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
