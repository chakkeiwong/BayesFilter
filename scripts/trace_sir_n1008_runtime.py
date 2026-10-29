"""Diagnostic-only N1008 cost interventions; never algorithm or safety candidates."""

import argparse
import ast
import collections
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

import benchmark_sir_n1008_runtime as benchmark


ROOT = benchmark.ROOT
PLAN = "docs/plans/ledh-sir-n1008-slowdown-trace-20261023.md"
MODES = ("premerge", "current", "determinant_off", "innovation_off", "both_off")
ARMS = {mode: ("plain", "conditional") for mode in MODES}
ARMS["premerge"] += ("physical",)
ARMS["current"] += ("physical",)


def diagnostic_setup(mode):
    def setup(tf):
        record = dict(mode=mode, diagnostic_only=True, runtime_candidate=False, trace_calls=[])
        if mode == "premerge":
            return record
        from bayesfilter.highdim import ledh_canonical_score_tf as score

        original_determinant = score.checked_log_abs_det_tangent
        original_cholesky = score.safe_cholesky

        def determinant_placeholder(matrix, tangent):
            shape = tf.shape(matrix)[:-2]
            return (tf.zeros(shape, matrix.dtype), tf.zeros(shape, matrix.dtype),
                    tf.ones(shape, tf.bool), tf.ones(shape, tf.bool))

        def innovation_placeholder(matrix):
            return tf.ones(tf.shape(matrix)[:-2], tf.bool), tf.linalg.cholesky(matrix)

        def determinant(matrix, tangent):
            record["trace_calls"].append(dict(call="flow_determinant", shape=matrix.shape.as_list()))
            if mode in ("determinant_off", "both_off"):
                return determinant_placeholder(matrix, tangent)
            return original_determinant(matrix, tangent)

        def cholesky(matrix, name="cholesky"):
            if name == "flow_innovation":
                record["trace_calls"].append(dict(call=name, shape=matrix.shape.as_list()))
                if mode in ("innovation_off", "both_off"):
                    return innovation_placeholder(matrix)
            return original_cholesky(matrix, name)

        @tf.function(input_signature=[tf.TensorSpec([2, 2], tf.float64)] * 2,
                     jit_compile=True, autograph=False)
        def invalid_fixtures(singular, asymmetric):
            original_det = original_determinant(singular, tf.zeros_like(singular))[2]
            removed_det = determinant_placeholder(singular, tf.zeros_like(singular))[2]
            original_cov = original_cholesky(asymmetric, "flow_innovation")[0]
            removed_cov = innovation_placeholder(asymmetric)[0]
            return original_det, removed_det, original_cov, removed_cov

        fixture = invalid_fixtures(tf.constant([[1., 0.], [0., 0.]], tf.float64),
                                  tf.constant([[1., .25], [0., 1.]], tf.float64))
        record["guard_fixture_validity"] = [bool(value.numpy()) for value in fixture]
        if record["guard_fixture_validity"] != [False, True, False, True]:
            raise ValueError("Diagnostic safety-loss fixture failed")
        score.checked_log_abs_det_tangent = determinant
        score.safe_cholesky = cholesky
        return record

    return setup


def source_checks(root):
    sources = benchmark.read(root / "sources.json")
    current = Path(sources["current"])
    files = {
        "guide": "bayesfilter/highdim/ledh_backward_guide_tf.py",
        "score": "bayesfilter/highdim/ledh_canonical_score_tf.py",
        "physical": "bayesfilter/highdim/ledh_physical_proposal_tf.py",
        "moment": "bayesfilter/score_study/moment_training_tf.py",
    }
    trees = {name: ast.parse((current / path).read_text()) for name, path in files.items()}
    chain = (
        ("moment", "make_moment_endpoint", "python_function"),
        ("guide", "build_backward_guided_ledh_kernel", "_value_and_analytical_score_impl"),
        ("score", "_value_and_analytical_score_impl", "_flow_substeps_with_tangent"),
        ("score", "_value_and_analytical_score_impl", "physical_mixture_draw"),
        ("score", "_flow_substeps_with_tangent", "_flow_substep_body_impl"),
        ("score", "_flow_substeps_with_tangent", "checked_log_abs_det_tangent"),
        ("score", "_flow_substep_body_impl", "safe_cholesky"),
    )
    calls = []
    for file_key, function, callee in chain:
        node = next(node for node in ast.walk(trees[file_key])
                    if isinstance(node, ast.FunctionDef) and node.name == function)
        matching = [child.lineno for child in ast.walk(node) if isinstance(child, ast.Call)
                    and ((isinstance(child.func, ast.Name) and child.func.id == callee)
                         or (isinstance(child.func, ast.Attribute) and child.func.attr == callee))]
        if not matching:
            raise ValueError(f"Missing source call {function} -> {callee}")
        calls.append(dict(path=files[file_key], function=function, callee=callee, lines=matching))
    hashes = {name: benchmark.file_hash(Path(sources[name]) / files["physical"])
              for name in ("premerge", "current")}
    if len(set(hashes.values())) != 1:
        raise ValueError("Conditional proposal source is not identical")
    return dict(calls=calls, proposal_source_hashes=hashes)


def summarize_hlo(path):
    parser = runpy.run_path(str(ROOT / "docs/plans/artifacts/"
        "ledh-sir-remaining-overhead-trace-20261022/trace_hlo.py"))
    computations, entry = parser["parse_hlo"](path)
    reached = parser["reachable"](computations, entry)
    instructions = [row for name in reached for row in computations[name]]
    selected = []
    for row in instructions:
        operation = row["metadata"].get("op_name", "")
        if row["opcode"] == "while" or (row["opcode"] == "custom-call" and any(
                token in operation.lower() for token in ("qr", "triangular", "cholesky"))):
            selected.append({key: row[key] for key in ("opcode", "shape", "metadata", "line", "callees")})
    return dict(path=str(path), sha256=benchmark.file_hash(path),
                reachable_opcodes=dict(collections.Counter(row["opcode"] for row in instructions)),
                loops_and_factorizations=selected)


def compute_processes():
    completed = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                               "--format=csv"], capture_output=True, text=True, timeout=15)
    return dict(returncode=completed.returncode, stdout=completed.stdout, stderr=completed.stderr)


def warm_parity(result):
    parity = []
    for mode in MODES:
        for arm in ARMS[mode]:
            for dataset in (263103001, 263103002):
                baseline = result["warm"]["current"][f"{dataset}:{arm}"]["output"]
                observed = result["warm"][mode][f"{dataset}:{arm}"]["output"]
                verdict = benchmark.compare_outputs(baseline, observed, diagnostics=mode != "premerge")
                parity.append(dict(mode=mode, arm=arm, dataset=dataset, **verdict))
                if not verdict["passed"]:
                    raise ValueError(f"Healthy output mismatch: {mode} {arm} {dataset}")
    return parity


def analysis(result):
    rows, parity = [], warm_parity(result)
    for mode in MODES:
        if mode == "current":
            continue
        for arm in ARMS[mode]:
            for dataset in (None, 263103001, 263103002):
                def times(active_mode):
                    return [statistics.mean(row["synchronized_seconds"] for row in block[active_mode]
                            if row["arm"] == arm and (dataset is None or row["dataset"] == dataset))
                            for block in result["blocks"]]
                rows.append(dict(mode=mode, arm=arm, dataset=dataset,
                    ratio_definition="mode / unchanged current", **benchmark.paired_interval(times("current"), times(mode))))
    return dict(parity=parity, timing=rows, historical_n64_evidence_used=False,
                diagnostic_variants_runtime_eligible=False, parameter_batch_tested=False)


def supervisor(arguments):
    arguments.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    workers, logs, launched = {}, {}, {}
    result = dict(plan=PLAN, command=sys.argv, status="running", warm={}, blocks=[],
                  gpu_initial=benchmark.gpu_snapshot(), process_polls=[], source_checks=source_checks(arguments.root),
                  driver_sha256=benchmark.file_hash(Path(__file__)), budget_seconds=arguments.budget_seconds,
                  input_root=str(arguments.root), historical_n64_evidence_used=False)
    environment = dict(os.environ, TF_FORCE_GPU_ALLOW_GROWTH="true", CUDA_VISIBLE_DEVICES=benchmark.GPU,
        XLA_FLAGS="--xla_gpu_enable_command_buffer=", PYTHONDONTWRITEBYTECODE="1",
        BAYESFILTER_OP_LIB_DIR=str(ROOT / "bayesfilter/ops"))

    def receive(mode):
        with selectors.DefaultSelector() as selector:
            selector.register(workers[mode].stdout, selectors.EVENT_READ)
            while True:
                if time.monotonic() - started >= arguments.budget_seconds:
                    raise TimeoutError("Elapsed diagnostic budget exhausted")
                if selector.select(timeout=1.):
                    line = workers[mode].stdout.readline()
                    if not line:
                        raise RuntimeError(f"Worker {mode} exited; inspect worker.log")
                    return json.loads(line)

    def request(mode, message):
        workers[mode].stdin.write(json.dumps(message) + "\n")
        workers[mode].stdin.flush()
        return receive(mode)

    try:
        for mode in MODES:
            destination = arguments.output / mode
            destination.mkdir()
            command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--mode", mode,
                       "--root", str(arguments.root), "--output", str(destination), "--particles", "1008"]
            logs[mode] = (destination / "worker.log").open("w")
            launched[mode] = time.monotonic()
            workers[mode] = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=logs[mode], text=True, bufsize=1, env=environment, cwd=ROOT)
            ready = receive(mode)
            if ready.get("stage") != "ready":
                raise ValueError(f"Worker not ready: {mode}")
            result["warm"][mode] = {}
            for dataset, arm in itertools.product((263103001, 263103002), ARMS[mode]):
                row = request(mode, dict(action="evaluate", dataset=dataset, arm=arm, direction=0, block=-1))
                result["warm"][mode][f"{dataset}:{arm}"] = row
            benchmark.emit(dict(stage="warm", mode=mode, elapsed=time.monotonic()-started))
            benchmark.save(arguments.output / "progress.json", dict(stage="warm", mode=mode))
        input_hashes = [benchmark.read(arguments.output / mode / "manifest.json")["input_sha256"] for mode in MODES]
        if any(hashes != input_hashes[0] for hashes in input_hashes):
            raise ValueError("Unpaired input hashes")
        benchmark.save(arguments.output / "warm-parity.json", warm_parity(result))
        generator = random.Random(263103199)
        cases = [(mode, dataset, arm) for mode in MODES
                 for dataset, arm in itertools.product((263103001, 263103002), ARMS[mode])]
        for block_index in range(arguments.blocks):
            generator.shuffle(cases)
            block = {mode: [] for mode in MODES}
            result["process_polls"].append(dict(block=block_index, processes=compute_processes()))
            for mode, dataset, arm in cases:
                block[mode].append(request(mode, dict(action="evaluate", dataset=dataset, arm=arm,
                                                     direction=0, block=block_index)))
            result["blocks"].append(block)
            benchmark.save(arguments.output / "result.json", result)
            benchmark.emit(dict(stage="block", complete=block_index+1, elapsed=time.monotonic()-started))
        benchmark.save(arguments.output / "analysis.json", analysis(result))
        result["hlo"] = {}
        for mode in MODES:
            result["hlo"][mode] = {}
            for arm in ARMS[mode]:
                exported = request(mode, dict(action="export_hlo", dataset=263103001, arm=arm, direction=0))
                result["hlo"][mode][arm] = summarize_hlo(Path(exported["path"]))
            result[mode] = dict(terminal=request(mode, dict(action="stop")))
            if not result[mode]["terminal"]["source_unchanged"]:
                raise ValueError("Source drift")
            workers[mode].wait(timeout=15)
            result[mode]["resident_seconds"] = time.monotonic()-launched[mode]
            benchmark.emit(dict(stage="hlo", mode=mode, elapsed=time.monotonic()-started))
        result["status"] = "complete"
    except Exception as error:
        result["status"] = "failed"
        result["error"] = repr(error)
        raise
    finally:
        for mode, worker in workers.items():
            if worker.poll() is None:
                worker.terminate()
                try:
                    worker.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    worker.kill()
                    worker.wait(timeout=10)
            logs[mode].close()
            result.setdefault(mode, {}).setdefault("resident_seconds", time.monotonic()-launched[mode])
        result["wall_seconds"] = time.monotonic()-started
        result["gpu_final"] = benchmark.gpu_snapshot()
        result["process_final"] = compute_processes()
        benchmark.save(arguments.output / "result.json", result)
    benchmark.emit(dict(stage="complete", wall_seconds=result["wall_seconds"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--particles", type=int, default=1008)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--mode", choices=MODES)
    parser.add_argument("--blocks", type=int, default=12)
    parser.add_argument("--budget-seconds", type=float, default=1200.)
    arguments = parser.parse_args()
    if arguments.particles != 1008 or arguments.blocks < 2:
        parser.error("This diagnostic is fixed at N1008 and needs at least two blocks")
    if arguments.worker:
        if arguments.mode is None:
            parser.error("Worker needs a mode")
        arguments.lane = "premerge" if arguments.mode == "premerge" else "current"
        arguments.plan = PLAN
        benchmark.ARMS = {name: benchmark.ARMS[name] for name in ARMS[arguments.mode]}
        benchmark.worker(arguments, diagnostic_setup(arguments.mode))
    else:
        supervisor(arguments)


if __name__ == "__main__":
    main()
