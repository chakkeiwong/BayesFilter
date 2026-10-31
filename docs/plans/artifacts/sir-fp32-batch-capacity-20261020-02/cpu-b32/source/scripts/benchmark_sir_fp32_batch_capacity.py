#!/usr/bin/env python3
"""Bounded diagnostic of native SIR FP32 value/score batch memory and time.

No training, HMC, numerical admission, or production-default selection occurs.
The supervisor imports no numerical framework. CPU children hide GPUs before
import; GPU children require the supervisor to have trusted device access.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import shutil
import signal
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "docs/plans/sir-fp32-batch-capacity-benchmark-2026-10-20.md"
SELECTION = ROOT / (
    "docs/plans/artifacts/surrogate-hmc-cost-repair-24h-20261013-01/"
    "literature-prior-other-four-advance-01/sir_d18/filter-tuning/"
    "sir_d18/filter-selection.json")
GPU_UUID = "GPU-68251639-fe82-8f81-3ccc-2953c32e805b"
GIB = 1024 ** 3


def stamp():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def meminfo():
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        if key in ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree"):
            values[key] = int(value.split()[0]) * 1024
    return values


def process_memory(pid):
    try:
        lines = Path(f"/proc/{pid}/status").read_text().splitlines()
    except FileNotFoundError:
        return {}
    return {line.split(":")[0]: int(line.split()[1]) * 1024
            for line in lines if line.startswith(("VmRSS:", "VmHWM:", "VmSize:"))}


def gpu_snapshot():
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu",
         "--format=csv"], capture_output=True, text=True, timeout=15)
    return {"utc": stamp(), "returncode": completed.returncode,
            "csv": completed.stdout.strip(), "stderr": completed.stderr.strip()}


def source_hashes():
    # Include imported repository modules, including locally uncommitted code.
    paths = {Path(__file__).resolve(), PLAN, SELECTION}
    for module in tuple(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename:
            path = Path(filename).resolve()
            if path.suffix == ".py" and path.is_relative_to(ROOT) and path.is_file():
                paths.add(path)
    return {str(path.relative_to(ROOT)): digest(path) for path in sorted(paths)}


def worker(args):
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    record = dict(schema="bayesfilter.sir_fp32_batch_capacity.v1", status="STARTING",
                  device=args.device, batch_size=args.batch, started_utc=stamp(),
                  process_id=os.getpid(),
                  command=[sys.executable, *sys.argv], plan=str(PLAN),
                  evidence_role="engineering_diagnostic_only", measurements=[],
                  git_commit=subprocess.check_output(
                      ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip())

    def checkpoint(phase):
        record.update(phase=phase, elapsed_seconds=time.monotonic()-started)
        record["host_process_memory"] = process_memory(os.getpid())
        save(args.output / "result.json", record)

    checkpoint("framework_import")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1" if args.device == "cpu" else args.gpu
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "false"  # Explicit fixed-cap exception.
    os.environ["TF_NUM_INTRAOP_THREADS"] = "16" if args.device == "cpu" else "4"
    os.environ["TF_NUM_INTEROP_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = os.environ["TF_NUM_INTRAOP_THREADS"]
    os.environ["BAYESFILTER_PRELOAD_CUSTOM_OP"] = "0"
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    sys.path.insert(0, str(ROOT))
    try:
        import tensorflow as tf
        tf.config.experimental.enable_tensor_float_32_execution(False)
        tf.config.set_soft_device_placement(False)
        if args.device == "gpu":
            from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_limit
            memory_policy = configure_tensorflow_gpu_memory_limit(
                tf, memory_limit_mib=args.gpu_limit_mib, require_gpu=True)
            placement = "/GPU:0"
            with tf.device(placement):
                probe = tf.linalg.matmul(tf.eye(2), tf.eye(2))
                probe.numpy()
            if "GPU:0" not in probe.device:
                raise RuntimeError("trusted GPU probe did not execute on GPU")
            gpu_details = [tf.config.experimental.get_device_details(d)
                           for d in tf.config.list_physical_devices("GPU")]
            del probe
        else:
            if tf.config.list_physical_devices("GPU"):
                raise RuntimeError("CPU benchmark unexpectedly sees GPU devices")
            memory_policy = {"mode": "cpu_only_gpu_intentionally_hidden"}
            placement = "/CPU:0"
            gpu_details = []
        record.update(
            tensorflow=tf.__version__, python=sys.executable, cpu_only=args.device == "cpu",
            cpu_affinity=list(os.sched_getaffinity(0)), memory_policy=memory_policy,
            device_details=gpu_details, device_placement=placement,
            environment={k: os.environ.get(k) for k in (
                "CUDA_VISIBLE_DEVICES", "TF_FORCE_GPU_ALLOW_GROWTH", "TF_NUM_INTRAOP_THREADS",
                "TF_NUM_INTEROP_THREADS", "OMP_NUM_THREADS", "XLA_FLAGS", "TF_XLA_FLAGS")},
            dtype="float32", precision_policy="uniform", jit_compile=True,
            tf32=tf.config.experimental.tensor_float_32_execution_enabled(),
            gpu_trust_basis="escalated_local_GPU_execution" if args.device == "gpu" else None)
        if record["tf32"]:
            raise RuntimeError("FP32-only benchmark requires TF32 off")
        checkpoint("input_preparation_on_cpu")
        from bayesfilter.testing import surrogate_hmc_five_model as runtime
        from bayesfilter.highdim.covariance_proposal_tf import ProposalControls, VALIDITY_FIELDS
        from bayesfilter.highdim.covariance_proposal_score_batch_tf import make_batched_score_filter
        selected = json.loads(SELECTION.read_text())
        with tf.device("/CPU:0"):
            case = runtime.load_case("sir_d18")
            case = runtime.configure_prior(case, runtime.priors.PRIOR_ID)
            # This is explicitly a workload baseline, not exact-scope admission.
            case = replace(case, controls=ProposalControls(**selected["selected"]["controls"]),
                           beta=tf.constant(selected["selected"]["beta"], tf.float32))
            case = runtime.configure_precision(case, "uniform")
            qbank = tf.constant([[0., 0., 0.], [.05, 0., 0.], [0., .05, 0.], [0., 0., .05]], tf.float32)
            q = tf.gather(qbank, tf.range(args.batch) % 4)
            theta, _ = case.chart(q)
            cpu_inputs = (theta, case.observations, *case.fixed, case.beta)
            input_hashes = [hashlib.sha256(tf.io.serialize_tensor(x).numpy()).hexdigest()
                            for x in cpu_inputs]
        record.update(
            particles=runtime.N, horizon=case.horizon, state_dimension=case.spec.dimension,
            observation_dimension=int(case.observations.shape[1]), parameter_count=case.p,
            controls=asdict(case.controls), beta=case.beta.numpy().tolist(),
            target_id=case.spec.target_id, selection=str(SELECTION), selection_sha256=digest(SELECTION),
            selected_scope_transferred=False, data_path=str(case.data_path),
            data_sha256=digest(case.data_path), fixed_design_seed=261008301,
            parameter_bank_q=qbank.numpy().tolist(), parameter_rows=theta.numpy().tolist(),
            input_hashes=input_hashes, chunk_size=2016, record_history=False,
            source_sha256=source_hashes(), input_preparation_fp64_outside_timed_kernel=True)
        if any(x.dtype != tf.float32 for x in cpu_inputs):
            raise RuntimeError("non-FP32 kernel input")
        with tf.device(placement):
            inputs = tuple(tf.identity(x) for x in cpu_inputs)
            program = make_batched_score_filter(
                case.spec, runtime.N, case.horizon, case.controls, tf.float32,
                jit_compile=True, batch_size=args.batch, precision_policy="uniform",
                fixed_draw_design=(case.beta,case.fixed[2]))
            inputs = inputs[:4]
            record["draw_design"] = "exact_integer_indices_prepared_once_on_cpu_same_shared_authority"
            checkpoint("tracing")
            tick = time.monotonic()
            concrete = program.get_concrete_function()
            record["trace_seconds"] = time.monotonic()-tick
            graph = concrete.graph.as_graph_def()
            nodes = list(graph.node)
            for function in graph.library.function:
                nodes.extend(function.node_def)
            fp64_nodes = [n for n in nodes if any(
                a.type == tf.float64.as_datatype_enum or tf.float64.as_datatype_enum in a.list.type
                for a in n.attr.values())]
            # Model factories embed some FP64 literals followed by FP32 casts.
            # XLA constant-folds these; reject any actual FP64 numerical op,
            # placeholder, loop argument, or cast from a dynamic FP32 input.
            fp64_runtime = [n.name for n in fp64_nodes if not (n.op == "Const" or
                (n.op == "Cast" and n.attr["SrcT"].type == tf.float64.as_datatype_enum
                 and n.attr["DstT"].type == tf.float32.as_datatype_enum))]
            record["graph"] = dict(nodes_including_functions=len(nodes), serialized_bytes=graph.ByteSize(),
                                   function_count=len(graph.library.function), fp64_runtime_nodes=fp64_runtime,
                                   fp64_literal_and_cast_count=len(fp64_nodes)-len(fp64_runtime),
                                   fixed_signature=[str(s) for s in program.input_signature])
            record["graph"]["all_input_dtypes_including_captures"] = [x.dtype.name for x in concrete.inputs]
            record["source_sha256"] = source_hashes()
            for relative in record["source_sha256"]:
                snapshot = args.output / "source" / relative
                snapshot.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / relative, snapshot)
            del nodes, graph
            if fp64_runtime or any(x.dtype == tf.float64 for x in concrete.inputs):
                raise RuntimeError("FP64 arithmetic found in timed filter graph")

            def allocator():
                try:
                    return tf.config.experimental.get_memory_info(placement.strip("/"))
                except (ValueError, RuntimeError) as exc:
                    return {"unavailable": str(exc)}

            def reset_peak():
                try:
                    tf.config.experimental.reset_memory_stats(placement.strip("/"))
                except (ValueError, RuntimeError):
                    pass

            reset_peak()
            record["allocator_before_execution"] = allocator()
            for index in range(4):
                checkpoint("compile_and_first_execution" if index == 0 else f"warmed_execution_{index}")
                if index == 1:
                    reset_peak()
                tick = time.monotonic()
                cpu_tick = time.process_time()
                result = program(*inputs)
                values = result["value"].numpy().tolist()  # Synchronize, without copying particle clouds.
                elapsed = time.monotonic()-tick
                cpu_elapsed = time.process_time()-cpu_tick
                valid = result["valid"].numpy().tolist()
                completed = result["steps_completed"].numpy().tolist()
                score = result["score"].numpy().tolist()
                finite = all(math.isfinite(x) for x in values) and all(
                    math.isfinite(x) for row in score for x in row)
                full = all(valid) and all(x == case.horizon for x in completed) and finite
                failure_fields = []
                flags = result["validity_trace"].numpy().tolist()
                for row, count in enumerate(completed):
                    failure_fields.append([] if valid[row] or not count else
                        [name for name, ok in zip(VALIDITY_FIELDS, flags[row][count-1]) if not ok])
                measurement = dict(
                    kind="compile_plus_first_execution" if index == 0 else "warmed_execution",
                    seconds=elapsed, cpu_seconds=cpu_elapsed, cpu_seconds_per_wall_second=cpu_elapsed/elapsed,
                    full_horizon_valid=full, valid=valid, steps_completed=completed,
                    finite=finite, failure_fields=failure_fields, values=values, scores=score,
                    allocator=allocator(), host_process_memory=process_memory(os.getpid()),
                    full_filter_rows_per_second=args.batch/elapsed if full else None)
                record["measurements"].append(measurement)
                del result, flags
                checkpoint("measurement_saved")
            warmed = record["measurements"][1:]
            median = statistics.median(x["seconds"] for x in warmed)
            full = all(x["full_horizon_valid"] for x in warmed)
            record.update(status="COMPLETED_FULL_HORIZON" if full else "COMPLETED_TRUNCATED_OR_INVALID",
                          median_warmed_seconds=median,
                          full_filter_rows_per_second=args.batch/median if full else None,
                          minimum_warmed_seconds=min(x["seconds"] for x in warmed),
                          maximum_warmed_seconds=max(x["seconds"] for x in warmed))
            checkpoint("complete")
    except BaseException as exc:
        record.update(status="ERROR", failure_phase=record.get("phase"),
                      exception_type=type(exc).__name__, exception=str(exc)[-8000:])
        traceback.print_exc()
        checkpoint("failed")
        return 1
    return 0


def terminate_owned(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    manifest = dict(schema="bayesfilter.sir_fp32_batch_capacity_supervisor.v1", started_utc=stamp(),
                    command=[sys.executable, *sys.argv], plan=str(PLAN), results=[],
                    total_budget_seconds=args.total_seconds, per_arm_seconds=args.arm_seconds,
                    rss_limit_bytes=24*GIB, minimum_system_available_bytes=6*GIB,
                    gpu_allocator_limit_mib=args.gpu_limit_mib,
                    gpu_snapshot_before=gpu_snapshot(), memory_before=meminfo(),
                    cpu_info=subprocess.check_output(["lscpu"], text=True),
                    git_commit=subprocess.check_output(
                        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    script_sha256=digest(__file__), plan_sha256=digest(PLAN))
    arms = [("cpu", b) for b in (4, 8, 16, 32)] + [("gpu", b) for b in (4, 8, 16, 32, 64, 128)]
    if args.devices != "both":
        arms = [(d, b) for d, b in arms if d == args.devices]
    save(args.output / "manifest.json", manifest)
    for device, batch in arms:
        if time.monotonic()-started >= args.total_seconds:
            manifest["budget_exhausted"] = True
            break
        name = f"{device}-b{batch}"
        out = args.output / name
        command = [sys.executable, str(Path(__file__).resolve()), "worker", "--output", str(out),
                   "--device", device, "--batch", str(batch), "--gpu", args.gpu,
                   "--gpu-limit-mib", str(args.gpu_limit_mib)]
        before_gpu = gpu_snapshot() if device == "gpu" else None
        print(json.dumps({"event": "starting", "arm": name, "utc": stamp()}), flush=True)
        arm_started = time.monotonic()
        peak_rss = peak_hwm = 0
        available_min = meminfo()["MemAvailable"]
        reason = None
        with (args.output / f"{name}.log").open("w") as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            try:
                last_heartbeat = 0.
                while process.poll() is None:
                    usage = process_memory(process.pid)
                    peak_rss = max(peak_rss, usage.get("VmRSS", 0))
                    peak_hwm = max(peak_hwm, usage.get("VmHWM", 0))
                    available = meminfo()["MemAvailable"]
                    available_min = min(available_min, available)
                    if time.monotonic()-last_heartbeat >= 5:
                        save(args.output / "active.json", dict(arm=name,pid=process.pid,
                            elapsed_seconds=time.monotonic()-arm_started, memory=usage,
                            peak_rss_bytes_sampled=peak_rss, available_host_bytes=available))
                        last_heartbeat = time.monotonic()
                    if peak_rss >= 24*GIB:
                        reason = "HOST_RSS_GUARD_24_GIB"
                    elif available < 6*GIB:
                        reason = "SYSTEM_AVAILABLE_MEMORY_GUARD_6_GIB"
                    elif time.monotonic()-arm_started >= args.arm_seconds:
                        reason = "ARM_TIME_CAP"
                    elif time.monotonic()-started >= args.total_seconds:
                        reason = "TOTAL_TIME_CAP"
                    if reason:
                        terminate_owned(process)
                        break
                    time.sleep(.2)
            finally:
                terminate_owned(process)
        row = dict(arm=name, device=device, batch_size=batch, command=command,
                   exit_code=process.returncode, stop_reason=reason,
                   wall_seconds=time.monotonic()-arm_started, peak_rss_bytes_sampled=peak_rss,
                   peak_hwm_bytes_sampled=peak_hwm, minimum_available_host_bytes=available_min,
                   result=str(out / "result.json"), log=str(args.output / f"{name}.log"),
                   gpu_before=before_gpu, gpu_after=gpu_snapshot() if device == "gpu" else None)
        if (out / "result.json").exists():
            result = json.loads((out / "result.json").read_text())
            row.update(worker_status=result["status"], last_phase=result.get("phase"))
        save(args.output / f"{name}-watchdog.json", row)
        manifest["results"].append(row)
        manifest["wall_seconds"] = time.monotonic()-started
        save(args.output / "manifest.json", manifest)
        print(json.dumps({k: row.get(k) for k in (
            "arm", "exit_code", "stop_reason", "wall_seconds", "worker_status", "last_phase")}), flush=True)
    manifest.update(finished_utc=stamp(), wall_seconds=time.monotonic()-started,
                    memory_after=meminfo(), gpu_snapshot_after=gpu_snapshot())
    save(args.output / "manifest.json", manifest)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "worker"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--devices", choices=("both", "cpu", "gpu"), default="both")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--gpu", default=GPU_UUID)
    parser.add_argument("--gpu-limit-mib", type=int, default=12288)
    parser.add_argument("--total-seconds", type=int, default=6600)
    parser.add_argument("--arm-seconds", type=int, default=600)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.batch < 2 or min(args.gpu_limit_mib, args.total_seconds, args.arm_seconds) <= 0:
        parser.error("positive limits and batch size > 1 are required")
    return worker(args) if args.command == "worker" else run(args)


if __name__ == "__main__":
    raise SystemExit(main())
