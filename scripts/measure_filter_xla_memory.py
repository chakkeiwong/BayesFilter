"""Fresh-process XLA/graph memory diagnostic; never an admitted runtime.

Host Python loops serialize diagnostics and drive repeated measurements. The
same repository numerical implementation and output lifetime are used in both
arms. No NumPy numerical computation or autodiff is used.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import resource
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def host_memory():
    fields = {}
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith(("VmRSS:", "VmHWM:")):
            name, value, _ = line.split()
            fields[name.rstrip(":") + "_kib"] = int(value)
    fields["resource_ru_maxrss_kib"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return fields


def build_fixture(tf, name, jit):
    if name in {"factor", "rectangular"}:
        from bayesfilter.testing.compiled_filter_runtime_fixture_tf import (
            fixture_result,
            numerical_result,
        )

        theta = tf.reshape(tf.linspace(tf.constant(-0.2, tf.float64), 0.23, 24), [8, 3])

        def evaluate(position):
            return numerical_result(fixture_result(
                position, rectangular=name == "rectangular", joint=True,
                jit_compile=jit, horizon=96,
            ))

        return evaluate, (theta,), {"batch": 8, "parameters": 3, "state": 2, "observations": 2, "horizon": 96}

    from bayesfilter.linear.kalman_covariance_derivatives_tf import (
        tf_batched_covariance_kalman_value_and_score,
    )

    b, p, n, m, horizon = 8, 32, 24, 12, 96
    dtype = tf.float64
    theta = tf.reshape(tf.linspace(tf.constant(-0.1, dtype), 0.15, b * p), [b, p])
    grid = tf.reshape(tf.cast(tf.range(n * n), dtype), [n, n])
    base_a = 0.65 * tf.eye(n, dtype=dtype) + 0.02 * tf.sin(grid) / n
    h = tf.eye(m, n, dtype=dtype) + 0.03 * tf.cos(tf.reshape(tf.cast(tf.range(m * n), dtype), [m, n])) / n
    y = 0.05 * tf.sin(tf.reshape(tf.cast(tf.range(horizon * m), dtype), [horizon, m]))
    d_a = 0.002 * tf.sin(tf.reshape(tf.cast(tf.range(p * n * n), dtype), [p, n, n])) / n
    d_mean = 0.01 * tf.cos(tf.reshape(tf.cast(tf.range(p * n), dtype), [p, n]))

    def evaluate(position):
        a = base_a[None] + tf.einsum("bp,pij->bij", position, d_a)
        mean = tf.einsum("bp,pn->bn", position, d_mean)
        # The decorated public endpoint is XLA by default. Using its exact
        # Python body permits a genuinely non-XLA diagnostic of identical math.
        return tf_batched_covariance_kalman_value_and_score.python_function(
            observations=y,
            transition_offset=tf.zeros([b, n], dtype), transition_matrix=a,
            transition_covariance=tf.broadcast_to(0.07 * tf.eye(n, dtype=dtype), [b, n, n]),
            observation_offset=tf.zeros([b, m], dtype),
            observation_matrix=tf.broadcast_to(h, [b, m, n]),
            observation_covariance=tf.broadcast_to(0.13 * tf.eye(m, dtype=dtype), [b, m, m]),
            initial_state_mean=mean,
            initial_state_covariance=tf.broadcast_to(0.4 * tf.eye(n, dtype=dtype), [b, n, n]),
            d_initial_state_mean=tf.broadcast_to(d_mean, [b, p, n]),
            d_initial_state_covariance=tf.zeros([b, p, n, n], dtype),
            d_transition_offset=tf.zeros([b, p, n], dtype),
            d_transition_matrix=tf.broadcast_to(d_a, [b, p, n, n]),
            d_transition_covariance=tf.zeros([b, p, n, n], dtype),
            d_observation_offset=tf.zeros([b, p, m], dtype),
            d_observation_matrix=tf.zeros([b, p, m, n], dtype),
            d_observation_covariance=tf.zeros([b, p, m, m], dtype),
        )

    return evaluate, (theta,), {"batch": b, "parameters": p, "state": n, "observations": m, "horizon": horizon}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("CPU", "GPU"), required=True)
    parser.add_argument("--gpu", default="2")
    parser.add_argument("--fixture", choices=("factor", "rectangular", "covariance"), required=True)
    parser.add_argument("--jit", choices=("on", "off"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu if args.device == "GPU" else "-1"
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    output = {
        "schema": "filter_xla_memory_diagnostic.v2", "started_utc": datetime.now(timezone.utc).isoformat(),
        "device": args.device, "fixture_name": args.fixture, "jit_compile": args.jit == "on",
        "command": shlex.join([sys.executable, *sys.argv]), "cwd": str(ROOT),
        "environment": {key: os.environ.get(key) for key in ("CUDA_VISIBLE_DEVICES", "TF_FORCE_GPU_ALLOW_GROWTH", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS", "OPENBLAS_NUM_THREADS", "XLA_FLAGS", "TF_XLA_FLAGS")},
        "python": platform.python_version(), "platform": platform.platform(),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "seed": "deterministic trigonometric fixture; no random draws", "dtype": "float64",
        "plan": "docs/plans/filter_gradient_policy_memory_audit_20260917.md",
        "nonclaims": ["No repository-wide leak, speedup, posterior, tuning or scientific conclusion"],
        "stages": {"before_tf_import": host_memory()},
    }
    import tensorflow as tf

    from bayesfilter.runtime.gpu_memory_policy import (
        configure_tensorflow_gpu_memory_growth,
    )

    output["tensorflow"] = tf.__version__
    output["memory_policy"] = configure_tensorflow_gpu_memory_growth(tf, require_gpu=args.device == "GPU")
    output["stages"]["after_tf_import_before_fixture"] = host_memory()
    output["tf32"] = bool(tf.config.experimental.tensor_float_32_execution_enabled())
    output["global_auto_jit"] = tf.config.optimizer.get_jit()
    if output["global_auto_jit"]:
        raise AssertionError("global auto-JIT makes the off comparison invalid")
    if args.device == "GPU":
        output["hardware"] = subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,uuid,driver_version,memory.total", "--format=csv"], text=True).strip()
        output["trust_basis"] = "trusted_escalated_gpu_diagnostic"

    def snapshot():
        result = host_memory()
        if args.device == "GPU":
            result["gpu_allocator_bytes"] = tf.config.experimental.get_memory_info("GPU:0")
        return result

    def reset_peak():
        if args.device == "GPU":
            tf.config.experimental.reset_memory_stats("GPU:0")

    with tf.device(f"/{args.device}:0"):
        evaluate, inputs, dimensions = build_fixture(tf, args.fixture, output["jit_compile"])
        output["fixture"] = dimensions
        output["input_devices"] = [item.device for item in inputs]
        output["stages"]["after_fixture_and_inputs"] = snapshot()
        target = tf.function(evaluate, input_signature=[tf.TensorSpec(item.shape, item.dtype) for item in inputs], jit_compile=output["jit_compile"], autograph=False)
        reset_peak()
        trace_start = time.perf_counter()
        concrete = target.get_concrete_function()
        output["trace_seconds"] = time.perf_counter() - trace_start
        output["stages"]["after_trace"] = snapshot()
        graph = concrete.graph.as_graph_def()
        nodes = list(graph.node) + [node for fn in graph.library.function for node in fn.node_def]
        output["graph"] = {
            "nodes": len(nodes), "functions": len(graph.library.function),
            "callbacks": sorted({node.op for node in nodes} & {"PyFunc", "PyFuncStateless", "EagerPyFunc"}),
            "outer_xla_must_compile": bool(concrete.function_def.attr.get("_XlaMustCompile").b),
            "nested_xla_functions": [fn.signature.name for fn in graph.library.function if "_XlaMustCompile" in fn.attr and fn.attr["_XlaMustCompile"].b],
            "nested_xla_nodes": [node.name for node in nodes if "_XlaMustCompile" in node.attr and node.attr["_XlaMustCompile"].b],
            "explicit_xla_ops": sorted({node.op for node in nodes if "Xla" in node.op}),
        }
        if output["graph"]["callbacks"]:
            raise AssertionError("Python callback in measured graph")
        if not output["jit_compile"] and any(output["graph"][key] for key in ("outer_xla_must_compile", "nested_xla_functions", "nested_xla_nodes", "explicit_xla_ops")):
            raise AssertionError("Non-XLA diagnostic contains compilation")
        reset_peak()
        cold_start = time.perf_counter()
        values = target(*inputs)
        # Every returned numerical tensor is synchronized and serialized. No
        # device outputs survive between invocations in either arm.
        output["values"] = [item.numpy().tolist() for item in values]
        output["cold_seconds_after_trace"] = time.perf_counter() - cold_start
        output["output_devices"] = [item.device for item in values]
        output["output_shapes"] = [item.shape.as_list() for item in values]
        output["finite"] = all(bool(tf.reduce_all(tf.math.is_finite(item)).numpy()) for item in values)
        output["stages"]["after_cold_outputs_live"] = snapshot()
        del values
        gc.collect()
        output["stages"]["after_cold_outputs_released"] = snapshot()
        output["warm"] = []
        for index in range(20):
            reset_peak()
            warm_start = time.perf_counter()
            values = target(*inputs)
            serialized = [item.numpy().tolist() for item in values]
            elapsed = time.perf_counter() - warm_start
            if serialized != output["values"]:
                raise AssertionError("Fixed-input warm result changed")
            del values, serialized
            output["warm"].append({"iteration": index + 1, "seconds": elapsed, **snapshot()})
        gc.collect()
        output["stages"]["after_warm_outputs_released"] = snapshot()
        output["trace_count"] = target.experimental_get_tracing_count()
        if output["trace_count"] != 1 or not output["finite"]:
            raise AssertionError("Nonfinite outputs or retracing")
        # HLO export occurs after measurements so its text/cache is not charged
        # to just the on arm. The cold execution already proves compilation.
        if output["jit_compile"]:
            hlo = target.experimental_get_compiler_ir(*inputs)(stage="optimized_hlo")
            hlo_path = args.output.with_suffix(".optimized_hlo.txt")
            with hlo_path.open("x") as handle:
                handle.write(hlo)
            output["hlo"] = {"path": str(hlo_path), "sha256": hashlib.sha256(hlo.encode()).hexdigest()}
    sources = {str(Path(__file__).resolve().relative_to(ROOT)): hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    for module in tuple(sys.modules.values()):
        source = getattr(module, "__file__", None)
        if source and Path(source).suffix == ".py" and Path(source).resolve().is_relative_to(ROOT):
            sources[str(Path(source).resolve().relative_to(ROOT))] = hashlib.sha256(Path(source).read_bytes()).hexdigest()
    output["imported_repository_sources_sha256"] = dict(sorted(sources.items()))
    output["wall_seconds"] = time.perf_counter() - started
    with args.output.open("x") as handle:
        json.dump(output, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({key: output[key] for key in ("fixture_name", "jit_compile", "wall_seconds", "finite", "trace_count", "stages")}, indent=2))


if __name__ == "__main__":
    main()
