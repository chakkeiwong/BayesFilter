"""Fresh N>=1008 SIR timing diagnostic; no historical result or weight reuse."""

import argparse
from dataclasses import asdict
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import runpy
import selectors
import statistics
import subprocess
import sys
import time

from replay_sir_postmerge_diagnostic import compare_numbers, file_hash


ROOT = Path(__file__).resolve().parents[1]
PLAN = "docs/plans/ledh-sir-n1008-runtime-20261023.md"
GPU = "GPU-68251639-fe82-8f81-3ccc-2953c32e805b"
ARMS = {"plain": 0., "conditional": 1., "physical": .9}
LANES = ("premerge", "checked", "current")
CONTROLS = dict(flow_substeps=24, reset_policy="contract_e", reset_epsilon=1.,
    reset_sinkhorn_steps=5, reset_balance_steps=32, reset_ridge=1e-5,
    reset_denominator_regularization=1e-7, correction_steps=1,
    correction_strength=.2, correction_lm_damping=.01,
    correction_lm_scale_floor=.0001, correction_trust_radius=.5,
    pairwise_steps=1, pairwise_strength=.02, pairwise_rms_cap=2.,
    coordinate_cap=.94, coordinate_cap_power=8., moment_safety=False)


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def emit(value):
    print(json.dumps(value, allow_nan=False), flush=True)


def compare_outputs(baseline, candidate, *, diagnostics=False):
    fields = ("value", "score", "estimate", "raw_estimate", "loss", "raw_blocks")
    numeric = {name: compare_numbers(baseline[name], candidate[name]) for name in fields}
    common = baseline["checks"].keys() & candidate["checks"].keys()
    changed = [name for name in sorted(common)
               if baseline["checks"][name] != candidate["checks"][name]]
    diagnostic_checks = {}
    if diagnostics:
        for name in baseline["diagnostics"].keys() & candidate["diagnostics"].keys():
            if "determinant_group" not in name:
                left, right = baseline["diagnostics"][name], candidate["diagnostics"][name]
                diagnostic_checks[name] = (dict(passed=left == right) if
                    isinstance(left, list) and left and isinstance(left[0], bool)
                    else compare_numbers(left, right))
    return dict(passed=bool(baseline["valid"] and candidate["valid"] and not changed
                and all(item["passed"] for item in (*numeric.values(), *diagnostic_checks.values()))),
                numeric=numeric, changed_flags=changed, diagnostics=diagnostic_checks,
                baseline_only_flags=sorted(baseline["checks"].keys() - common),
                candidate_only_flags=sorted(candidate["checks"].keys() - common))


def paired_interval(baseline, candidate, *, seed=263103099):
    if len(baseline) != len(candidate) or len(baseline) < 2:
        raise ValueError("At least two complete paired blocks are required")
    generator = random.Random(seed)
    ratios = []
    for _ in range(4096):
        indices = [generator.randrange(len(baseline)) for _ in baseline]
        ratios.append(sum(candidate[index] for index in indices)
                      / sum(baseline[index] for index in indices))
    ratios.sort()
    return dict(ratio=statistics.mean(candidate) / statistics.mean(baseline),
                interval95=[ratios[102], ratios[3993]],
                baseline_seconds=statistics.mean(baseline),
                current_seconds=statistics.mean(candidate))


def prepare(arguments):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    arguments.root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    sys.path.insert(0, str(ROOT))
    import tensorflow as tf
    from bayesfilter.score_study.supervised_guidance_tf import make_sir_supervision_programs
    from bayesfilter.score_study.moment_training_tf import calibrate_moments
    from bayesfilter.highdim.sqmc_nonlinear_tf import NonlinearSQMCSpec

    spec = NonlinearSQMCSpec("sir_d18")
    theta = spec.default_theta(tf.float64)
    calibration_simulator, _ = make_sir_supervision_programs(5, 256)
    paths, _ = calibration_simulator(theta, tf.constant(263103010))
    calibration = [tensor.numpy().tolist() for tensor in calibrate_moments(paths[:, -1])]
    simulator, _ = make_sir_supervision_programs(5, 1)
    datasets = []
    for index, seed in enumerate((263103001, 263103002)):
        paths, observations = simulator(theta, tf.constant(seed))
        datasets.append(dict(seed=seed, particle_seed=263103031 + index,
                             observations=observations[0].numpy().tolist(),
                             label=paths[0, -1].numpy().tolist()))
    save(arguments.root / "inputs.json", dict(calibration=calibration, datasets=datasets,
         calibration_seed=263103010, reset_seed=263103020,
         data_version="canonical_sir_d18_x0_then_transition_observe_v1",
         cpu_only=True, gpu_intentionally_hidden=True, jit_compile=True))
    capture = runpy.run_path(str(ROOT / "bayesfilter/score_study/run_source_snapshot.py"))
    destination = arguments.root / "current-source"
    capture["capture_source_snapshot"](ROOT, destination)
    auxiliary = {}
    for path in sorted((ROOT / "experiments").rglob("*.py")):
        relative = path.relative_to(ROOT)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
        auxiliary[str(relative)] = file_hash(target)
    save(destination / "auxiliary-index.json", auxiliary)
    artifacts = ROOT / "docs/plans/artifacts"
    save(arguments.root / "sources.json", dict(
        premerge=str(artifacts / "ledh-sir-postmerge-replay-20261021/attempt-01/before-source"),
        checked=str(artifacts / "ledh-sir-merge-repair-20261022/candidate-02-final-source"),
        current=str(destination)))
    save(arguments.root / "preparation.json", dict(command=sys.argv, environment=sys.executable,
         wall_seconds=time.monotonic()-started, plan=PLAN, cpu_only=True,
         gpu_intentionally_hidden=True, inputs_sha256=file_hash(arguments.root / "inputs.json")))
    emit(dict(stage="prepared", wall_seconds=time.monotonic()-started))


def worker(arguments, diagnostic_setup=None):
    source = Path(read(arguments.root / "sources.json")[arguments.lane])
    sys.path.insert(0, str(source))
    import tensorflow as tf
    import bayesfilter
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    from bayesfilter.score_study.run_source_snapshot import verify_source_snapshot

    if Path(bayesfilter.__file__).resolve().parent != source.resolve() / "bayesfilter":
        raise ValueError("Wrong imported source")
    memory = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.experimental.enable_tensor_float_32_execution(False)
    index = read(source / "index.json")
    auxiliary = read(source / "auxiliary-index.json")

    def unchanged():
        return (verify_source_snapshot(source, index)["unchanged"]
                and all(file_hash(source / name) == digest for name, digest in auxiliary.items()))

    if not unchanged():
        raise ValueError("Snapshot changed")
    diagnostic = diagnostic_setup(tf) if diagnostic_setup is not None else None
    from bayesfilter.highdim.ledh_backward_guide_tf import build_backward_guided_ledh_kernel
    from bayesfilter.highdim.sqmc_nonlinear_tf import NonlinearSQMCSpec
    from bayesfilter.highdim.transport_chunk_policy import select_transport_chunks
    from bayesfilter.score_study.moment_training_tf import make_moment_endpoint, make_moment_objective

    data = read(arguments.root / "inputs.json")
    particles = arguments.particles
    chunk = select_transport_chunks(particles)
    moments = [tf.constant(value, tf.float64) for value in data["calibration"]]
    objective = make_moment_objective(particles, 18, *moments)
    spec = NonlinearSQMCSpec("sir_d18")
    theta = spec.default_theta(tf.float64)
    reset = tf.random.stateless_normal([particles, 18], [263103020, 0], dtype=tf.float64)
    common = dict(horizon=5, particle_count=particles, state_dimension=18,
                  observation_dimension=9, parameter_dimension=3, dtype=tf.float64,
                  jit_compile=True, return_trace=True, reset_design=reset, **CONTROLS)
    def build_endpoint(beta):
        endpoint = make_moment_endpoint(build_backward_guided_ledh_kernel(
            lambda parameter, direction: spec.model(parameter, direction)[0], None,
            physical_mixture_weight=beta, **common), objective, 18)

        @tf.function(input_signature=list(endpoint.input_signature), jit_compile=True, autograph=False)
        def with_raw_moments(*active_inputs):
            result = endpoint.python_function(*active_inputs)
            center, scale, _ = moments
            normalized_mean = result["estimate"][:18]
            pairs = [(row, column) for row in range(18) for column in range(row, 18)]
            rows = [row for row, _ in pairs]
            columns = [column for _, column in pairs]
            center_row, center_column = tf.gather(center, rows), tf.gather(center, columns)
            scale_row, scale_column = tf.gather(scale, rows), tf.gather(scale, columns)
            raw_second = (result["estimate"][18:] * scale_row * scale_column
                + center_row * scale_column * tf.gather(normalized_mean, columns)
                + center_column * scale_row * tf.gather(normalized_mean, rows)
                + center_row * center_column)
            return dict(result, raw_estimate=tf.concat([center + scale * normalized_mean, raw_second], 0))

        return with_raw_moments

    endpoints = {name: build_endpoint(beta) for name, beta in ARMS.items()}

    @tf.function(input_signature=[tf.TensorSpec([], tf.int32),
                 tf.TensorSpec([5, 9], tf.float64), tf.TensorSpec([18], tf.float64)],
                 jit_compile=True, autograph=False)
    def make_inputs(seed, observations, label):
        normal = tf.random.stateless_normal([particles, 18], tf.stack([seed, 0]), dtype=tf.float64)
        states, covariances, tangent, covariance_tangent = spec.initial_cloud(theta, normal)
        noise = tf.random.stateless_normal([5, particles, 18], tf.stack([seed, 1]), dtype=tf.float64)
        uniform = tf.random.stateless_uniform([5, particles], tf.stack([seed, 2]), dtype=tf.float64)
        return (theta, tf.zeros([3], tf.float64), states, covariances, noise,
                observations, tangent, covariance_tangent, uniform, label)

    inputs, hashes = {}, {}
    for dataset in data["datasets"]:
        seed = dataset["seed"]
        inputs[seed] = make_inputs(tf.constant(dataset["particle_seed"]),
            tf.constant(dataset["observations"], tf.float64), tf.constant(dataset["label"], tf.float64))
        digest = hashlib.sha256()
        for tensor in (*inputs[seed], reset):
            digest.update(tf.io.serialize_tensor(tensor).numpy())
        hashes[str(seed)] = digest.hexdigest()
    save(arguments.output / "manifest.json", dict(command=sys.argv, environment=sys.executable,
        tensorflow=tf.__version__, git_commit=index["commit"], source=str(source),
        source_index_sha256=file_hash(source / "index.json"), input_sha256=hashes,
        input_file_sha256=file_hash(arguments.root / "inputs.json"), particles=particles,
        horizon=5, state_dimension=18, parameter_dimension=3, parameter_batch_size=1,
        guide=None, controls=CONTROLS, chunk=asdict(chunk), dtype="float64", tf32=False,
        jit_compile=True, xla_flags=os.environ.get("XLA_FLAGS"), gpu_memory_policy=memory,
        gpu_uuid=GPU, gpu_devices=[item.name for item in tf.config.list_physical_devices("GPU")],
        seeds=dict(calibration=263103010, reset=263103020,
                   data=[item["seed"] for item in data["datasets"]],
                   particle=[item["particle_seed"] for item in data["datasets"]]),
        custom_op_sha256=file_hash(ROOT / "bayesfilter/ops/_symmetric_sylvester_ops.so"),
        script_sha256=file_hash(Path(__file__)), plan=getattr(arguments, "plan", PLAN),
        diagnostic_only=True, diagnostic_intervention=diagnostic,
        result=str(arguments.output / "rows.jsonl")))
    emit(dict(stage="ready", input_sha256=hashes))
    warmed = {}
    for line in sys.stdin:
        request = json.loads(line)
        if request["action"] == "stop":
            terminal = dict(stage="stopped", source_unchanged=unchanged(),
                trace_counts={name: endpoint.experimental_get_tracing_count()
                              for name, endpoint in endpoints.items()},
                allocator=tf.config.experimental.get_memory_info("GPU:0"),
                diagnostic_intervention=diagnostic)
            save(arguments.output / "terminal.json", terminal)
            emit(terminal)
            return
        key = (request["dataset"], request["arm"], request["direction"])
        dataset, arm, direction = key
        active = inputs[dataset]
        if direction != -1:
            active = (active[0], tf.one_hot(direction, 3, dtype=tf.float64), *active[2:])
        if request["action"] == "export_hlo":
            destination = arguments.output / f"{arm}.optimized_hlo.txt"
            destination.write_text(endpoints[arm].experimental_get_compiler_ir(*active)(stage="optimized_hlo"))
            emit(dict(stage="hlo_exported", path=str(destination), sha256=file_hash(destination)))
            continue
        if request["action"] != "evaluate":
            raise ValueError("Unsupported worker request")
        started = time.perf_counter()
        output = endpoints[arm](*active)
        materialized = tf.nest.map_structure(lambda tensor: tensor.numpy(), output)
        synchronized = time.perf_counter()
        result = tf.nest.map_structure(lambda value: value.tolist(), materialized)
        row = dict(dataset=dataset, arm=arm, direction=direction, block=request["block"],
            synchronized_seconds=synchronized-started, serialization_seconds=time.perf_counter()-synchronized,
            output=result, allocator=tf.config.experimental.get_memory_info("GPU:0"))
        with (arguments.output / "rows.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        if not result["valid"] and arguments.lane == "current":
            raise ValueError(f"Invalid endpoint: {key}")
        if key in warmed and result != warmed[key]:
            raise ValueError(f"Nonrepeatable output: {key}")
        warmed[key] = result
        emit(row)


def gpu_snapshot():
    completed = subprocess.run(["nvidia-smi", "--query-gpu=uuid,memory.used,utilization.gpu",
        "--format=csv"], capture_output=True, text=True, timeout=15)
    return dict(returncode=completed.returncode, stdout=completed.stdout, stderr=completed.stderr)


def summarize(result):
    checks, summaries = [], []
    for comparator in ("premerge", "checked"):
        for arm, direction in itertools.product(ARMS, (-1, 0)):
            compatible = True
            for dataset in (263103001, 263103002):
                left = next(row["output"] for row in result["warm"][comparator]
                            if (row["dataset"], row["arm"], row["direction"]) == (dataset, arm, direction))
                right = next(row["output"] for row in result["warm"]["current"]
                             if (row["dataset"], row["arm"], row["direction"]) == (dataset, arm, direction))
                check = compare_outputs(left, right, diagnostics=comparator == "checked")
                compatible &= check["passed"]
                checks.append(dict(comparator=comparator, dataset=dataset, arm=arm,
                                   direction=direction, **check))
            for dataset in (None, 263103001, 263103002):
                times = {}
                for lane in (comparator, "current"):
                    times[lane] = [statistics.mean(row["synchronized_seconds"] for row in block[lane]
                        if row["arm"] == arm and row["direction"] == direction
                        and (dataset is None or row["dataset"] == dataset)) for block in result["blocks"]]
                interval = paired_interval(times[comparator], times["current"])
                summaries.append(dict(comparator=comparator, arm=arm, direction=direction,
                    dataset=dataset, parity_passed=compatible, **interval,
                    faster_supported=compatible and interval["interval95"][1] < 1.,
                    no_more_than_five_percent_slower=compatible and interval["interval95"][1] <= 1.05,
                    concurrent_device_use_during_timing_not_verified=True))
    return dict(checks=checks, timings=summaries, historical_results_used=False,
                scientific_accuracy_or_efficacy_claim=False, parameter_batch_tested=False)


def supervisor(arguments):
    arguments.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    workers, logs, launches, charges = {}, {}, {}, {}
    result = dict(status="running", plan=PLAN, command=sys.argv, warm={}, blocks=[],
                  budget_seconds=arguments.budget_seconds, gpu_initial=gpu_snapshot())
    environment = dict(os.environ, TF_FORCE_GPU_ALLOW_GROWTH="true", CUDA_VISIBLE_DEVICES=GPU,
        XLA_FLAGS="--xla_gpu_enable_command_buffer=", PYTHONDONTWRITEBYTECODE="1",
        BAYESFILTER_OP_LIB_DIR=str(ROOT / "bayesfilter/ops"))

    def receive(lane):
        with selectors.DefaultSelector() as selector:
            selector.register(workers[lane].stdout, selectors.EVENT_READ)
            while True:
                if time.monotonic()-started >= arguments.budget_seconds:
                    raise TimeoutError("Campaign elapsed-time budget exhausted")
                if selector.select(timeout=1.):
                    line = workers[lane].stdout.readline()
                    if not line:
                        raise RuntimeError(f"Worker {lane} exited; inspect its worker.log")
                    return json.loads(line)

    def request(lane, message):
        workers[lane].stdin.write(json.dumps(message) + "\n")
        workers[lane].stdin.flush()
        return receive(lane)

    cases = list(itertools.product((263103001, 263103002), ARMS, (-1, 0)))
    try:
        for lane in LANES:
            destination = arguments.output / lane
            destination.mkdir()
            command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--lane", lane,
                "--root", str(arguments.root), "--output", str(destination),
                "--particles", str(arguments.particles)]
            logs[lane] = (destination / "worker.log").open("w")
            launches[lane] = time.monotonic()
            workers[lane] = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=logs[lane], text=True, bufsize=1, env=environment, cwd=ROOT)
            result[lane] = dict(ready=receive(lane))
            result["warm"][lane] = []
            for dataset, arm, direction in cases:
                result["warm"][lane].append(request(lane, dict(action="evaluate", block=-1,
                    dataset=dataset, arm=arm, direction=direction)))
                emit(dict(stage="warm", lane=lane, dataset=dataset, arm=arm, direction=direction,
                          elapsed_seconds=time.monotonic()-started))
            save(arguments.output / "progress.json", result)
        hashes = [result[lane]["ready"]["input_sha256"] for lane in LANES]
        if any(value != hashes[0] for value in hashes):
            raise ValueError("Cross-source input mismatch")
        for block_index in range(arguments.blocks):
            block = {lane: [] for lane in LANES}
            offset = block_index % len(cases)
            ordered_cases = cases[offset:] + cases[:offset]
            for case_index, (dataset, arm, direction) in enumerate(ordered_cases):
                lane_offset = (block_index + case_index) % len(LANES)
                ordered_lanes = LANES[lane_offset:] + LANES[:lane_offset]
                if block_index % 2:
                    ordered_lanes = tuple(reversed(ordered_lanes))
                for lane in ordered_lanes:
                    block[lane].append(request(lane, dict(action="evaluate", block=block_index,
                        dataset=dataset, arm=arm, direction=direction)))
            result["blocks"].append(block)
            result.setdefault("gpu_blocks", []).append(gpu_snapshot())
            save(arguments.output / "progress.json", result)
            emit(dict(stage="paired_block", block=block_index, elapsed_seconds=time.monotonic()-started))
        for lane in LANES:
            result[lane]["terminal"] = request(lane, dict(action="stop"))
            workers[lane].wait(timeout=30)
            charges[lane] = time.monotonic()-launches[lane]
            terminal = result[lane]["terminal"]
            if not terminal["source_unchanged"] or set(terminal["trace_counts"].values()) != {1}:
                raise ValueError("Source drift or retracing")
        save(arguments.output / "analysis.json", summarize(result))
        result["status"] = "complete"
    except Exception as error:
        result.update(status="failed", error=repr(error))
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
            charges.setdefault(lane, time.monotonic()-launches[lane])
            logs[lane].close()
        result.update(wall_seconds=time.monotonic()-started, worker_seconds=charges,
                      charged_gpu_process_seconds=sum(charges.values()), gpu_final=gpu_snapshot())
        save(arguments.output / "result.json", result)
        emit(dict(stage=result["status"], wall_seconds=result["wall_seconds"],
                  charged_gpu_process_seconds=result["charged_gpu_process_seconds"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--lane", choices=LANES)
    parser.add_argument("--particles", type=int, default=1008)
    parser.add_argument("--blocks", type=int, default=16)
    parser.add_argument("--budget-seconds", type=float, default=2600.)
    arguments = parser.parse_args()
    if arguments.particles < 1008:
        parser.error("The owner disqualified runs with fewer than 1008 particles")
    if arguments.prepare:
        prepare(arguments)
    elif arguments.worker:
        worker(arguments)
    else:
        supervisor(arguments)


if __name__ == "__main__":
    main()
