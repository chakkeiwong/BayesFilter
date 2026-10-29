#!/usr/bin/env python3
"""Diagnostic-only component/force cost localization; no HMC or retuning.

First calls and warmed calls are separate. Components use the first observed
step; the scalar force uses the complete current T50 master binding. No array
backend other than TensorFlow is used. See the companion experiment plan.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PLAN = ROOT / "docs/plans/surrogate-hmc-five-model-cost-diagnosis-2026-10-11.md"
PREPARED = ROOT / "docs/plans/artifacts/surrogate-hmc-five-model-n4032-20261011-06"


def stamp():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def save(path, payload):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def worker(args):
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    manifest = dict(
        schema="bayesfilter.five_model_cost_diagnostic.v1", status="STARTING",
        model=args.worker, started_utc=stamp(), command=[sys.executable, *sys.argv],
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        plan=str(PLAN), result=str(args.output / "timings.json"),
        script_sha256=digest(__file__), python=sys.executable,
        gpu_uuid=args.gpu_uuid, device="GPU", cpu_only=False,
        evidence_role="engineering_cost_diagnostic_only", measurements={},
        limitations=["first-step component timings are not full-filter time shares",
                     "two warmed calls at one fixed parameter point",
                     "RTX 4080 SUPER differs from the earlier RTX 5080 pilot"],
    )
    save(args.output / "timings.json", manifest)
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_uuid
        os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        os.environ.setdefault("BAYESFILTER_PRELOAD_CUSTOM_OP", "0")
        os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "4")
        os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
        import tensorflow as tf
        from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
        memory = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
        tf.config.experimental.enable_tensor_float_32_execution(False)
        from bayesfilter.testing import surrogate_hmc_five_model as runtime
        from bayesfilter.highdim.covariance_proposal_tf import (
            build_maps, component_densities, draw_from_maps, observation_density, reset_cloud,
        )
        case = runtime.load_case(args.worker)
        selection = PREPARED / args.worker / "filter-tuning/filter-selection.json"
        runtime.load_selection(case, selection)
        devices = [tf.config.experimental.get_device_details(d)
                   for d in tf.config.list_physical_devices("GPU")]
        manifest.update(
            tensorflow=tf.__version__, memory_policy=memory, device_details=devices,
            gpu_trust_basis="escalated_local_GPU_execution", cpu_only=False,
            scope=case.scope(), controls=asdict(case.controls),
            beta=runtime.json_ready(case.beta), selection=str(selection),
            selection_sha256=digest(selection), particles=runtime.N,
            horizon=case.horizon, state_dimension=case.spec.dimension,
            observation_dimension=int(case.observations.shape[1]), parameter_count=case.p,
            dtype="float32", tf32=tf.config.experimental.tensor_float_32_execution_enabled(),
            endpoint_dtype="float64", jit_compile=True, status="MEASURING",
        )
        if manifest["tf32"]:
            raise RuntimeError("ordinary FP32 requires TF32 disabled")
        save(args.output / "timings.json", runtime.json_ready(manifest))

        def signature(values):
            return tf.nest.map_structure(lambda x: tf.TensorSpec(x.shape, x.dtype), values)

        def compile_function(fn, values):
            return tf.function(fn, input_signature=signature(values),
                               jit_compile=True, autograph=False)

        def graph_record(fn):
            concrete = fn.get_concrete_function()
            graph = concrete.graph.as_graph_def()
            counts = Counter(node.op for node in graph.node)
            for function in graph.library.function:
                counts.update(node.op for node in function.node_def)
            attributes = concrete.function_def.attr
            return dict(
                xla_must_compile=bool(attributes["_XlaMustCompile"].b)
                    if "_XlaMustCompile" in attributes else None,
                tracing_count=fn.experimental_get_tracing_count(),
                graph_and_function_nodes=sum(counts.values()),
                relevant_operations={name: counts[name] for name in (
                    "While", "StatelessWhile", "PyFunc", "EagerPyFunc",
                    "PartitionedCall", "StatefulPartitionedCall")},
            )

        def measure(label, fn, values, synchronize):
            record = dict(status="FIRST_CALL_RUNNING")
            manifest["measurements"][label] = record
            save(args.output / "timings.json", runtime.json_ready(manifest))
            start = time.monotonic()
            result = fn(*values)
            first = synchronize(result)
            record["first_call_seconds"] = time.monotonic() - start
            record["first_outputs"] = runtime.json_ready(first)
            record["status"] = "WARM_CALLS_RUNNING"
            record["warm_seconds"] = []
            save(args.output / "timings.json", runtime.json_ready(manifest))
            for _ in range(2):
                start = time.monotonic()
                result = fn(*values)
                last = synchronize(result)
                record["warm_seconds"].append(time.monotonic() - start)
                record["last_outputs"] = runtime.json_ready(last)
                save(args.output / "timings.json", runtime.json_ready(manifest))
            record.update(status="COMPLETE", graph=graph_record(fn),
                          warm_median_seconds=statistics.median(record["warm_seconds"]))
            save(args.output / "timings.json", runtime.json_ready(manifest))
            print(args.worker, label, record["first_call_seconds"], record["warm_seconds"], flush=True)
            return result

        with tf.device("/GPU:0"):
            theta = case.theta0
            direction = tf.one_hot(0, case.p, dtype=tf.float32)
            x, _, dx, _ = case.spec.initial_cloud(theta, case.fixed[0], direction)
            observation = case.observations[0]

            def maps_fn(theta, direction, x, dx, observation):
                model, _ = case.spec.model(theta, direction)
                return build_maps(model, theta, x, dx, observation, case.controls)

            inputs = (theta, direction, x, dx, observation)
            maps_graph = compile_function(maps_fn, inputs)
            maps = measure("first_step_build_maps", maps_graph, inputs,
                           lambda out: [out[12].numpy(), out[13].numpy(), out[14].numpy()])
            if not bool(maps[12]):
                raise RuntimeError("first-step map validity failed")
            draw_inputs = (maps, case.beta, case.fixed[2][0], case.fixed[1][0])
            draw_graph = compile_function(draw_from_maps, draw_inputs)
            child, dchild = measure("first_step_draw", draw_graph, draw_inputs,
                lambda out: [tf.reduce_sum(out[0]).numpy(), tf.reduce_sum(out[1]).numpy()])
            density_inputs = (child, dchild, *maps[:4])
            density_graph = compile_function(component_densities, density_inputs)
            logdens, ddens, valid = measure("first_step_three_mixture_densities",
                density_graph, density_inputs, lambda out: [out[2].numpy(), tf.reduce_sum(out[0]).numpy(), tf.reduce_sum(out[1]).numpy()])
            if not bool(valid):
                raise RuntimeError("mixture-density validity failed")

            def weight_fn(theta, direction, child, dchild, observation, logdens, ddens, beta):
                model, _ = case.spec.model(theta, direction)
                lg, dlg = observation_density(model, theta, child, dchild, observation)
                logmix = logdens + tf.math.log(beta)[None]
                lq = tf.reduce_logsumexp(logmix, -1)
                dlq = tf.reduce_sum(tf.nn.softmax(logmix, -1) * ddens, -1)
                lr, dlr = lg + logdens[:, 0] - lq, dlg + ddens[:, 0] - dlq
                ds = tf.reduce_sum(tf.nn.softmax(lr) * dlr)
                return lr - tf.reduce_logsumexp(lr), dlr - ds

            weight_inputs = (theta, direction, child, dchild, observation, logdens, ddens, case.beta)
            logw, dlogw = compile_function(weight_fn, weight_inputs)(*weight_inputs)
            reset_inputs = (child, dchild, logw, dlogw)

            def reset_fn(x, dx, logw, dlogw):
                return reset_cloud(x, dx, logw, dlogw, case.controls)

            reset_graph = compile_function(reset_fn, reset_inputs)
            reset = measure("first_step_reset", reset_graph, reset_inputs,
                            lambda out: [out[6].numpy(), *out[7].numpy().tolist(), tf.reduce_sum(out[0]).numpy(), tf.reduce_sum(out[1]).numpy()])
            if not bool(reset[6]):
                raise RuntimeError("reset validity failed")
            components = runtime.bindings(case)
            q = tf.zeros([case.p], tf.float64)
            force = measure("full_T50_scalar_force", components["scalar_force"], (q,),
                            lambda out: [out[0].numpy().tolist(), out[1].numpy()])
            if not bool(force[1]) or not all(math.isfinite(v) for v in force[0].numpy().tolist()):
                raise RuntimeError("full scalar force validity failed")
            endpoint = measure("independent_scalar_endpoint", components["independent"],
                    (tf.cast(theta, tf.float64),),
                    lambda out: [out[0].numpy(), out[1].numpy()])
            if not bool(endpoint[1]) or not math.isfinite(float(endpoint[0])):
                raise RuntimeError("independent endpoint validity failed")
            manifest["gpu_allocator_bytes"] = tf.config.experimental.get_memory_info("GPU:0")
        manifest["status"] = "COMPLETE_DIAGNOSTIC_ONLY"
    except Exception:
        manifest.update(status="DIAGNOSTIC_ERROR", traceback=traceback.format_exc())
        raise
    finally:
        manifest.update(wall_seconds=time.monotonic()-started, finished_utc=stamp())
        # Tensor values have already been materialized at diagnostic boundaries.
        save(args.output / "timings.json", manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--budget-seconds", type=float, default=360)
    parser.add_argument("--worker", choices=("lgssm", "sir_d18"))
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
    if not 0 < args.budget_seconds <= 360:
        raise ValueError("diagnostic plan permits at most 360 wall seconds")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    ledger = dict(status="RUNNING", started_utc=stamp(), command=[sys.executable, *sys.argv],
                  plan=str(PLAN), wall_budget_seconds=args.budget_seconds, models={})
    for model in ("lgssm", "sir_d18"):
        remaining = args.budget_seconds - (time.monotonic()-started)
        if remaining <= 0:
            ledger["models"][model] = dict(status="BUDGET_STOP_BEFORE_START")
            break
        command = [sys.executable, str(Path(__file__).resolve()), "--worker", model,
                   "--output", str(args.output/model), "--gpu-uuid", args.gpu_uuid]
        row = dict(status="RUNNING", command=command, started_utc=stamp())
        ledger["models"][model] = row
        save(args.output/"manifest.json", ledger)
        begin = time.monotonic()
        with (args.output/(model+".log")).open("w") as log:
            try:
                proc = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                      timeout=remaining, check=False)
                row.update(status="COMPLETE" if proc.returncode == 0 else "WORKER_ERROR",
                           returncode=proc.returncode)
            except subprocess.TimeoutExpired:
                row.update(status="BUDGET_STOP", returncode=None)
        row.update(wall_seconds=time.monotonic()-begin, finished_utc=stamp())
        save(args.output/"manifest.json", ledger)
        print(model, row["status"], row["wall_seconds"], flush=True)
    ledger.update(status="COMPLETE" if len(ledger["models"]) == 2 and all(
        row["status"] == "COMPLETE" for row in ledger["models"].values()) else "PARTIAL",
        wall_seconds=time.monotonic()-started, finished_utc=stamp())
    save(args.output/"manifest.json", ledger)


if __name__ == "__main__":
    main()
