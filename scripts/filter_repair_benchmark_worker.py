"""Fresh-process engineering diagnostic for the filter repair campaign."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

from measure_filter_xla_memory import build_fixture as audit_fixture


def host_memory():
    return {row.split()[0].rstrip(":"): int(row.split()[1]) * 1024 for row in Path("/proc/self/status").read_text().splitlines() if row.startswith(("VmRSS:", "VmHWM:"))}


def fixture(tf, name, size, jit):
    if name in ("rectangular", "factor", "covariance", "sinkhorn_jvp"):
        if size != 1:
            raise ValueError("Legacy audit fixture has one frozen size")
        return audit_fixture(tf, name, jit)
    if name == "sqmc":
        from bayesfilter.highdim.sqmc_tf import hilbert_permutation
        count, dim, bits = 256 * size, 4 * size, 8
        points = tf.reshape(tf.sin(tf.cast(tf.range(count * dim), tf.float64) * 0.31), [count, dim])
        def evaluate(x):
            return hilbert_permutation(x, tf.zeros([dim], x.dtype), tf.ones([dim], x.dtype), bits=bits)
        return evaluate, (points,), {"particles": count, "dimension": dim, "bits": bits}
    if name == "dns":
        from bayesfilter.hardbound.dns_curve_tf import yield_curve
        factors = tf.reshape(tf.linspace(tf.constant(-0.04, tf.float64), 0.05, 96 * size), [32 * size, 3])
        maturities = tf.constant([0.25, 0.5, 1., 2., 5., 10.], tf.float64)
        def evaluate(x):
            return (yield_curve(x, maturities, 0.5, 0., 0.01, "softplus"),)
        return evaluate, (factors,), {"batch": 32 * size, "maturities": 6, "quadrature": 40}
    if name == "sgqf_derivatives":
        from bayesfilter.highdim.models import p30_predator_prey_fixture_model
        from bayesfilter.nonlinear.fixed_sgqf_structural_adapter_tf import (
            tf_predator_prey_to_fixed_sgqf_model,
        )
        model = p30_predator_prey_fixture_model()
        points = tf.reshape(tf.linspace(tf.constant(0.3, tf.float64), 1.6, 32 * size), [16 * size, 2])
        def evaluate(x):
            adapter = tf_predator_prey_to_fixed_sgqf_model(model, tf.constant([0.1, -0.2, 0.15, -0.1], tf.float64), with_derivatives=True)
            return adapter.derivatives.transition_state_jacobian_fn(x), adapter.derivatives.d_transition_fn(x)
        return evaluate, (points,), {"points": 16 * size, "state": 2, "parameters": 4}
    raise ValueError(f"Fixture {name} is not qualified yet; missing coverage blocks merge")


def measure(args, result):
    # Set source precedence before any numerical import, including fixture imports.
    source = args.source_root.resolve()
    sys.path.insert(0, str(source))
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    if args.device == "CPU":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    import tensorflow as tf

    from bayesfilter.runtime.gpu_memory_policy import (
        configure_tensorflow_gpu_memory_growth,
    )
    result["memory_policy"] = configure_tensorflow_gpu_memory_growth(tf, require_gpu=args.device == "GPU")
    result["tensorflow"] = tf.__version__
    result["tf32"] = tf.config.experimental.tensor_float_32_execution_enabled()
    result["environment"] = {key: os.environ.get(key) for key in ("CUDA_VISIBLE_DEVICES", "TF_FORCE_GPU_ALLOW_GROWTH", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS", "OPENBLAS_NUM_THREADS", "TF_XLA_FLAGS", "XLA_FLAGS")}
    if tf.config.optimizer.get_jit():
        raise RuntimeError("Global auto-JIT invalidates graph comparison")
    if args.device == "GPU":
        result["hardware"] = subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,uuid,driver_version", "--format=csv"], text=True).strip()
        result["trust_basis"] = "trusted_escalated_gpu_diagnostic"
    def snapshot():
        out = host_memory()
        if args.device == "GPU":
            out["gpu"] = tf.config.experimental.get_memory_info("GPU:0")
        return out
    def reset_peak():
        if args.device == "GPU":
            tf.config.experimental.reset_memory_stats("GPU:0")
    with tf.device(f"/{args.device}:0"):
        start = time.perf_counter()
        evaluate, inputs, dimensions = fixture(tf, args.fixture, args.size, args.jit == "on")
        # Legacy fixture loader prepends its own root. Source packages are already
        # resolved; restore baseline precedence for all subsequent lazy imports.
        sys.path.insert(0, str(source))
        result["preparation_seconds"] = time.perf_counter() - start
        result["dimensions"] = dimensions
        input_values = [value.numpy().tolist() for value in inputs]
        result["input_sha256"] = hashlib.sha256(json.dumps(input_values, allow_nan=False).encode()).hexdigest()
        result["input_shapes"] = [value.shape.as_list() for value in inputs]
        result["stages"]["prepared"] = snapshot()
        target = tf.function(evaluate, input_signature=[tf.TensorSpec(value.shape, value.dtype) for value in inputs], autograph=False, jit_compile=args.jit == "on")
        reset_peak()
        start = time.perf_counter()
        concrete = target.get_concrete_function()
        result["trace_seconds"] = time.perf_counter() - start
        result["stages"]["traced"] = snapshot()
        graph = concrete.graph.as_graph_def()
        nodes = list(graph.node) + [node for fn in graph.library.function for node in fn.node_def]
        result["graph"] = {"nodes": len(nodes), "functions": len(graph.library.function), "callbacks": sorted({node.op for node in nodes} & {"PyFunc", "PyFuncStateless", "EagerPyFunc"}), "must_compile": bool(concrete.function_def.attr.get("_XlaMustCompile").b), "nested_xla": [fn.signature.name for fn in graph.library.function if fn.attr.get("_XlaMustCompile") and fn.attr["_XlaMustCompile"].b], "xla_ops": sorted({node.op for node in nodes if "Xla" in node.op})}
        if result["graph"]["callbacks"] or (args.jit == "off" and (result["graph"]["nested_xla"] or result["graph"]["xla_ops"])):
            raise RuntimeError("Invalid callback/compilation boundary")
        def call():
            start = time.perf_counter()
            values = tf.nest.flatten(target(*inputs))
            copy_start = time.perf_counter()
            host = [value.numpy() for value in values]
            end = time.perf_counter()
            metrics = {"synchronized_seconds": end - start, "output_copy_seconds": end - copy_start}
            return values, host, metrics
        reset_peak()
        values, host, timing = call()
        result["cold"] = timing
        result["output_shapes"] = [value.shape.as_list() for value in values]
        result["output_devices"] = [value.device for value in values]
        result["values"] = [value.tolist() for value in host]
        result["stages"]["cold_outputs_live"] = snapshot()
        del values, host
        gc.collect()
        result["stages"]["cold_outputs_released"] = snapshot()
        result["warm"] = []
        for iteration in range(20):
            reset_peak()
            values, host, timing = call()
            serialized = [value.tolist() for value in host]
            if serialized != result["values"]:
                raise RuntimeError("Repeated fixed-input output changed")
            del values, host, serialized
            result["warm"].append({"iteration": iteration, **timing, **snapshot()})
        result["trace_count"] = target.experimental_get_tracing_count()
        if result["trace_count"] != 1:
            raise RuntimeError("Unbounded trace contract")
        if args.jit == "on":
            hlo = target.experimental_get_compiler_ir(*inputs)(stage="optimized_hlo")
            hlo_path = args.output.with_suffix(".hlo.txt")
            with hlo_path.open("x") as handle:
                handle.write(hlo)
            result["hlo_sha256"] = hashlib.sha256(hlo.encode()).hexdigest()
    imported = {}
    for module in tuple(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename and filename.endswith(".py"):
            path = Path(filename).resolve()
            if path.is_relative_to(source):
                imported[str(path.relative_to(source))] = hashlib.sha256(path.read_bytes()).hexdigest()
            elif getattr(module, "__name__", "").startswith(("bayesfilter.", "experiments.dpf_implementation.")):
                raise RuntimeError(f"Repository import escaped measured source: {filename}")
    result["imported_source_sha256"] = imported
    result["status"] = "passed"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--jit", choices=("on", "off"), required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--device", choices=("GPU", "CPU"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    started = time.perf_counter()
    result = {"schema": "filter_repair_measurement.v1", "fixture": args.fixture, "size": args.size, "jit": args.jit, "device": args.device, "source_root": str(args.source_root), "worker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "stages": {"start": host_memory()}, "status": "failed"}
    try:
        measure(args, result)
    except Exception:
        result["error"] = traceback.format_exc()
        raise
    finally:
        result["wall_seconds"] = time.perf_counter() - started
        with args.output.open("x") as handle:
            json.dump(result, handle, indent=2, allow_nan=False)
            handle.write("\n")
        print(json.dumps({key: result[key] for key in ("fixture", "size", "jit", "status", "wall_seconds")}))


if __name__ == "__main__":
    main()
