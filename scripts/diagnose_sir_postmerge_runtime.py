"""Diagnostic-only frozen consumer timing and optimized-HLO inspection."""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import selectors
import subprocess
import sys
import time

from replay_sir_postmerge_diagnostic import compare_numbers, file_hash, read_json


ARMS = ("plain", "conditional", "physical")
GPU_UUID = "GPU-68251639-fe82-8f81-3ccc-2953c32e805b"


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def emit(value):
    print(json.dumps(value, allow_nan=False), flush=True)


def worker(arguments):
    source = arguments.source.resolve()
    sys.path.insert(0, str(source))
    import tensorflow as tf
    import bayesfilter
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth

    if Path(bayesfilter.__file__).resolve().parent != source / "bayesfilter":
        raise ValueError("Source import isolation failed")
    memory = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.experimental.enable_tensor_float_32_execution(False)
    from bayesfilter.highdim.ledh_backward_guide_tf import build_backward_guided_ledh_kernel
    from bayesfilter.highdim.sqmc_nonlinear_tf import NonlinearSQMCSpec
    from bayesfilter.score_study.moment_training_tf import make_moment_endpoint, make_moment_objective
    from bayesfilter.score_study.run_source_snapshot import verify_source_snapshot

    source_index = read_json(source / "index.json")
    if not verify_source_snapshot(source, source_index)["unchanged"]:
        raise ValueError("Frozen source changed")
    auxiliary = read_json(source / "auxiliary-index.json")
    if not all(file_hash(source / name) == digest for name, digest in auxiliary.items()):
        raise ValueError("Frozen auxiliary source changed")
    study = arguments.repository / "docs/plans/artifacts/ledh-sir-neural-kdm-20261020"
    holdout = study / "attempt-16-finalized/final-holdout"
    original = read_json(holdout / "manifest.json")
    calibration = read_json(holdout / "calibration.json")
    records = {dataset: read_json(holdout / f"dataset-{dataset}.json")
               for dataset in arguments.datasets}
    moments = [tf.constant(value, tf.float64) for value in calibration["moment"]]
    objective = make_moment_objective(64, 18, *moments)
    spec = NonlinearSQMCSpec("sir_d18")
    theta = spec.default_theta(tf.float64)
    reset_design = tf.random.stateless_normal([64, 18], [263040002, 0], dtype=tf.float64)
    common = dict(horizon=5, particle_count=64, state_dimension=18, observation_dimension=9,
        parameter_dimension=3, dtype=tf.float64, jit_compile=True, return_trace=True,
        reset_design=reset_design, **original["controls"])
    factory = lambda parameter, direction: spec.model(parameter, direction)[0]
    upper_indices = [(row, column) for row in range(18) for column in range(row, 18)]

    def build_endpoint(beta):
        kernel = build_backward_guided_ledh_kernel(
            factory, None, **common, physical_mixture_weight=beta)
        endpoint = make_moment_endpoint(kernel, objective, 18)

        @tf.function(input_signature=list(endpoint.input_signature), jit_compile=True, autograph=False)
        def with_raw_moments(*inputs):
            result = endpoint.python_function(*inputs)
            center, scale, _ = moments
            normalized_mean = result["estimate"][:18]
            raw_mean = center + scale * normalized_mean
            row_indices = [row for row, _ in upper_indices]
            column_indices = [column for _, column in upper_indices]
            center_row, center_column = tf.gather(center, row_indices), tf.gather(center, column_indices)
            scale_row, scale_column = tf.gather(scale, row_indices), tf.gather(scale, column_indices)
            raw_second = (result["estimate"][18:] * scale_row * scale_column
                + center_row * scale_column * tf.gather(normalized_mean, column_indices)
                + center_column * scale_row * tf.gather(normalized_mean, row_indices)
                + center_row * center_column)
            return dict(result, raw_estimate=tf.concat([raw_mean, raw_second], axis=0))

        return with_raw_moments

    endpoints = {name: build_endpoint(beta) for name, beta in
                 (("plain", 0.), ("conditional", 1.), ("physical", .9))}

    @tf.function(input_signature=[tf.TensorSpec([], tf.int32), tf.TensorSpec([5, 9], tf.float64),
                                  tf.TensorSpec([18], tf.float64)], jit_compile=True, autograph=False)
    def row_inputs(particle_seed, observations, label):
        normals = tf.random.stateless_normal([64, 18], tf.stack([particle_seed, 0]), dtype=tf.float64)
        states, covariances, d_states, d_covariances = spec.initial_cloud(theta, normals)
        noises = tf.random.stateless_normal([5, 64, 18], tf.stack([particle_seed, 1]), dtype=tf.float64)
        uniforms = tf.random.stateless_uniform([5, 64], tf.stack([particle_seed, 2]), dtype=tf.float64)
        return (theta, tf.zeros([3], tf.float64), states, covariances, noises,
                observations, d_states, d_covariances, uniforms, label)

    input_sets, input_hashes = {}, {}
    for dataset, record in records.items():
        particle_seed = 283030000 + 4 * (dataset - 263043001)
        inputs = row_inputs(tf.constant(particle_seed), tf.constant(record["observations"], tf.float64),
                           tf.constant(record["label"], tf.float64))
        digest = hashlib.sha256()
        for tensor in (*inputs, reset_design):
            digest.update(tf.io.serialize_tensor(tensor).numpy())
        input_sets[dataset], input_hashes[dataset] = inputs, digest.hexdigest()
    archives = {}
    for filename in ("holdout.jsonl", "scores.jsonl"):
        for line in (arguments.reference / filename).read_text().splitlines():
            row = json.loads(line)
            if row["dataset"] in records and row["replicate"] == 0 and row["arm"] in ARMS:
                direction = row.get("direction", -1)
                if direction in (-1, 0):
                    if row["input_sha256"] != input_hashes[row["dataset"]]:
                        raise ValueError("Input differs from archived replay")
                    archives[(row["dataset"], row["arm"], direction)] = row
    if len(archives) != 6 * len(records):
        raise ValueError("Missing archived checks")
    manifest = dict(command=sys.argv, environment=sys.executable, tensorflow=tf.__version__,
        git_commit=source_index["commit"], source=str(source),
        source_index_sha256=file_hash(source / "index.json"), gpu_uuid=GPU_UUID,
        gpu_devices=[device.name for device in tf.config.list_physical_devices("GPU")],
        gpu_memory_policy=memory, dtype="float64", tf32=False, jit_compile=True,
        data_version="canonical_sir_d18_x0_then_transition_observe_v1", datasets=arguments.datasets,
        particle_seeds={dataset: 283030000 + 4 * (dataset - 263043001) for dataset in records},
        reset_seed=263040002, input_sha256=input_hashes,
        controls=original["controls"], output=str(arguments.output),
        custom_op_sha256=file_hash(arguments.repository / "bayesfilter/ops/_symmetric_sylvester_ops.so"),
        plan=arguments.plan,
        result=str(arguments.output / "rows.jsonl"), diagnostic_only=True)
    save(arguments.output / "manifest.json", manifest)
    emit(dict(stage="ready", input_sha256=input_hashes))

    def evaluate(dataset, arm, direction):
        inputs = input_sets[dataset]
        active_inputs = inputs if direction == -1 else (
            inputs[0], tf.one_hot(direction, 3, dtype=tf.float64), *inputs[2:])
        started = time.perf_counter()
        output = endpoints[arm](*active_inputs)
        materialized = tf.nest.map_structure(lambda tensor: tensor.numpy(), output)
        synchronized = time.perf_counter()
        result = tf.nest.map_structure(lambda value: value.tolist(), materialized)
        serialized = time.perf_counter()
        reference = archives[(dataset, arm, direction)]
        fields = ("loss", "estimate", "value", "score") if direction == -1 else ("value", "score")
        comparisons = {name: compare_numbers(reference[name], result[name]) for name in fields}
        if not result["valid"] or not all(item["passed"] for item in comparisons.values()):
            save(arguments.output / "tieout-failure.json", dict(dataset=dataset, arm=arm,
                 direction=direction, comparisons=comparisons, valid=result["valid"]))
            raise ValueError(f"Archived consumer tieout failed: {dataset}, {arm}, {direction}")
        row = dict(dataset=dataset, arm=arm, direction=direction, synchronized_seconds=synchronized-started,
                   serialization_seconds=serialized-synchronized,
                   total_seconds=serialized-started, comparisons=comparisons, output=result)
        with (arguments.output / "rows.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        return {name: value for name, value in row.items() if name != "output"}

    for line in sys.stdin:
        request = json.loads(line)
        if request["action"] == "stop":
            unchanged = verify_source_snapshot(source, source_index)["unchanged"]
            if not unchanged:
                raise ValueError("Source changed during diagnostic")
            emit(dict(stage="stopped", source_unchanged=unchanged,
                trace_counts={name: endpoint.experimental_get_tracing_count()
                              for name, endpoint in endpoints.items()},
                allocator=tf.config.experimental.get_memory_info("GPU:0")))
            return
        if request["action"] == "warm":
            rows, hlo_summary = [], {}
            for arm in ARMS:
                for dataset in records:
                    rows.append(evaluate(dataset, arm, -1))
                    rows.append(evaluate(dataset, arm, 0))
                hlo = endpoints[arm].experimental_get_compiler_ir(
                    *input_sets[arguments.datasets[0]])(stage="optimized_hlo")
                (arguments.output / f"{arm}.optimized-hlo.txt").write_text(hlo)
                targets = Counter(re.findall(r'custom_call_target="([^"]+)"', hlo))
                qr_lines = [line for line in hlo.splitlines()
                            if 'op_type="Qr"' in line or re.search(r"geqrf|orgqr|ormqr", line, re.I)]
                hlo_summary[arm] = dict(custom_call_targets=dict(targets), qr_line_count=len(qr_lines),
                    qr_lines_path=str(arguments.output / f"{arm}.qr-lines.txt"))
                (arguments.output / f"{arm}.qr-lines.txt").write_text("\n".join(qr_lines) + "\n")
            save(arguments.output / "hlo-summary.json", hlo_summary)
            emit(dict(stage="warm", rows=rows, hlo=hlo_summary))
        elif request["action"] == "block":
            emit(dict(stage="block", rows=[evaluate(dataset, arm, direction)
                 for dataset in request.get("datasets", arguments.datasets)
                 for arm in request["arms"] for direction in request.get("directions", (-1, 0))]))
        else:
            raise ValueError("Unknown worker request")


def paired_requests(datasets, arms, pairing):
    if pairing == "block":
        return [dict(action="block", datasets=datasets, arms=arms, directions=[-1, 0])]
    if pairing != "call":
        raise ValueError("Unknown timing pairing")
    return [dict(action="block", datasets=[dataset], arms=[arm], directions=[direction])
            for dataset in datasets for arm in arms for direction in (-1, 0)]


def supervisor(arguments):
    arguments.output.mkdir(parents=True, exist_ok=False)
    frozen = arguments.repository / "docs/plans/artifacts/ledh-sir-postmerge-replay-20261021/attempt-01"
    workers, logs, launch_times, charges, results = {}, {}, {}, {}, {}
    environment = dict(os.environ, TF_FORCE_GPU_ALLOW_GROWTH="true", CUDA_VISIBLE_DEVICES=GPU_UUID,
        XLA_FLAGS="--xla_gpu_enable_command_buffer=", PYTHONDONTWRITEBYTECODE="1",
        BAYESFILTER_OP_LIB_DIR=str(arguments.repository / "bayesfilter/ops"))

    def charged_seconds():
        return sum(charges.values()) + sum(time.monotonic()-started
            for lane, started in launch_times.items() if lane not in charges)

    def receive(lane):
        with selectors.DefaultSelector() as selector:
            selector.register(workers[lane].stdout, selectors.EVENT_READ)
            while True:
                if charged_seconds() >= arguments.budget_seconds:
                    raise TimeoutError("Diagnostic GPU-process budget exhausted")
                if selector.select(timeout=1):
                    line = workers[lane].stdout.readline()
                    if not line:
                        raise RuntimeError(f"{lane} worker exited: inspect its log")
                    return json.loads(line)

    def request(lane, value):
        workers[lane].stdin.write(json.dumps(value) + "\n")
        workers[lane].stdin.flush()
        return receive(lane)

    try:
        lanes = (("before", frozen / "before-source", frozen / "before-retry02"),
                 ("after", frozen / "merged-source", frozen / "after"))
        if arguments.baseline_source or arguments.candidate_source:
            if not all((arguments.baseline_source, arguments.candidate_source, arguments.reference)):
                raise ValueError("Repair comparison requires both source snapshots and the merged reference")
            lanes = (("before", arguments.baseline_source, arguments.baseline_reference or arguments.reference),
                     ("after", arguments.candidate_source, arguments.reference))
        for lane, snapshot, reference in lanes:
            destination = arguments.output / lane
            destination.mkdir()
            command = [sys.executable, str(Path(__file__).resolve()), "--worker",
                "--repository", str(arguments.repository), "--output", str(destination),
                "--source", str(snapshot), "--reference", str(reference), "--plan", arguments.plan,
                "--datasets", *map(str, arguments.datasets)]
            logs[lane] = (destination / "worker.log").open("w")
            launch_times[lane] = time.monotonic()
            workers[lane] = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=logs[lane], text=True, bufsize=1, env=environment, cwd=arguments.repository)
            results[lane] = dict(ready=receive(lane))
            emit(dict(lane=lane, stage="ready"))
            results[lane]["warm"] = request(lane, dict(action="warm"))
            save(arguments.output / "progress.json", results)
            emit(dict(lane=lane, stage="warm"))
        if results["before"]["ready"]["input_sha256"] != results["after"]["ready"]["input_sha256"]:
            raise ValueError("Cross-source input mismatch")
        results["blocks"] = []
        results["pairing"] = arguments.pairing
        for block in range(arguments.blocks):
            order = ("before", "after") if block % 2 == 0 else ("after", "before")
            offset = block % len(ARMS)
            arms = ARMS[offset:] + ARMS[:offset]
            dataset_offset = block % len(arguments.datasets)
            datasets = arguments.datasets[dataset_offset:] + arguments.datasets[:dataset_offset]
            block_rows = {lane: [] for lane in order}
            for request_index, paired_request in enumerate(paired_requests(datasets, arms, arguments.pairing)):
                call_order = order if request_index % 2 == 0 else tuple(reversed(order))
                for lane in call_order:
                    result = request(lane, paired_request)
                    block_rows[lane].extend(result["rows"])
            for lane in order:
                results["blocks"].append(dict(block=block, lane=lane, stage="block", rows=block_rows[lane]))
            save(arguments.output / "progress.json", results)
            emit(dict(stage="paired_block", block=block, charged_seconds=charged_seconds()))
        for lane in workers:
            results[lane]["terminal"] = request(lane, dict(action="stop"))
            workers[lane].wait(timeout=30)
            charges[lane] = time.monotonic() - launch_times[lane]
        if any(count != 1 for lane in workers
               for count in results[lane]["terminal"]["trace_counts"].values()):
            raise ValueError("Unexpected retracing")
        results["status"] = "complete"
    except Exception as error:
        results.update(status="failed", error=repr(error))
        raise
    finally:
        for lane, process in workers.items():
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
            charges.setdefault(lane, time.monotonic()-launch_times[lane])
            logs[lane].close()
        results["charges"] = charges
        results["charged_gpu_process_seconds"] = sum(charges.values())
        save(arguments.output / "result.json", results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--baseline-source", type=Path)
    parser.add_argument("--baseline-reference", type=Path)
    parser.add_argument("--candidate-source", type=Path)
    parser.add_argument("--datasets", type=int, nargs="+", default=[263043001])
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--pairing", choices=("block", "call"), default="block")
    parser.add_argument("--budget-seconds", type=float, default=1800.)
    parser.add_argument("--plan", default="docs/plans/ledh-sir-postmerge-runtime-20261021.md")
    parser.add_argument("--worker", action="store_true")
    arguments = parser.parse_args()
    if arguments.worker:
        worker(arguments)
    else:
        supervisor(arguments)


if __name__ == "__main__":
    main()
