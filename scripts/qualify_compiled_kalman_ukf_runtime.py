"""Bounded CPU/GPU engineering qualification; no posterior or tuning claims."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["CPU", "GPU"], required=True)
    parser.add_argument("--gpu", default="2")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu if args.device == "GPU" else "-1"
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
    import tensorflow as tf

    from bayesfilter.runtime.gpu_memory_policy import (
        configure_tensorflow_gpu_memory_growth,
    )

    memory_policy = configure_tensorflow_gpu_memory_growth(
        tf, require_gpu=args.device == "GPU"
    )
    from audit_kalman_ukf_runtime import audit

    from bayesfilter.inference.hmc_transition_archive import (
        HMCTransitionArchiveConfig,
        build_hmc_transition_archive_runner,
    )
    from bayesfilter.linear.rectangular_factor_tf import _batched_qr_with_derivative
    from bayesfilter.linear.stack_qr_tf import batched_stack_qr_lower
    from bayesfilter.testing.compiled_filter_runtime_fixture_tf import (
        CompiledFilterMechanicsAdapter,
        fixture_result,
        numerical_result,
    )

    start = time.perf_counter()
    manifest = {
        "schema": "compiled_filter_runtime_qualification.v1",
        "command": sys.argv,
        "python": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "tensorflow": tf.__version__,
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "device": args.device,
        "memory_policy": memory_policy,
        "seed": [20260916, 37],
        "plan": "docs/plans/kalman_ukf_compiled_runtime_repair_20260916.md",
        "trust_basis": "owner_designated_managed_session_visible_gpu_trusted"
        if args.device == "GPU"
        else "CPU engineering reference",
        "tf32": bool(tf.config.experimental.tensor_float_32_execution_enabled()),
        "nonclaims": [
            "No DZ5 qualification",
            "No posterior, convergence, tuning or speedup claim",
            "No profiling-based synchronization claim",
        ],
        "source_sha256": {r["path"]: r["sha256"] for r in audit()["modules"]},
    }
    manifest["source_sha256"][
        "bayesfilter/testing/compiled_filter_runtime_fixture_tf.py"
    ] = hashlib.sha256(
        (
            ROOT / "bayesfilter/testing/compiled_filter_runtime_fixture_tf.py"
        ).read_bytes()
    ).hexdigest()

    def save():
        (args.output / "result.json").write_text(json.dumps(manifest, indent=2) + "\n")

    save()

    def sync(result):
        # Materialization is a measurement boundary, no NumPy numerical work.
        return (
            tf.reduce_sum(tf.cast(tf.nest.flatten(result)[-1], tf.float64))
            .numpy()
            .item()
        )

    def qualify(name, fn, call_args):
        begin = time.perf_counter()
        first = fn(*call_args)
        sync(first)
        cold = time.perf_counter() - begin
        warm = []
        for _ in range(2):
            begin = time.perf_counter()
            current = fn(*call_args)
            sync(current)
            warm.append(time.perf_counter() - begin)
        concrete = fn.get_concrete_function(*call_args)
        graph = concrete.graph.as_graph_def()
        nodes = list(graph.node) + [
            node for body in graph.library.function for node in body.node_def
        ]
        counts = Counter(node.op for node in nodes)
        forbidden = {"PyFunc", "PyFuncStateless", "EagerPyFunc"} & counts.keys()
        if forbidden:
            raise AssertionError(f"{name}: callbacks {forbidden}")
        hlo = fn.experimental_get_compiler_ir(*call_args)(stage="optimized_hlo")
        (args.output / f"{name}.hlo.txt").write_text(hlo)
        manifest[name] = {
            "cold_synchronized_seconds": cold,
            "warm_synchronized_seconds": warm,
            "trace_count": fn.experimental_get_tracing_count(),
            "graph_node_count": len(nodes),
            "reachable_operation_counts": dict(sorted(counts.items())),
            "hlo_sha256": hashlib.sha256(hlo.encode()).hexdigest(),
            "hlo_while_count": hlo.count("while("),
            "forbidden_callbacks": sorted(forbidden),
        }
        if manifest[name]["trace_count"] != 1:
            raise AssertionError(f"{name}: repeated tracing")
        save()
        return first

    def make_target(rectangular):
        @tf.function(
            input_signature=[tf.TensorSpec([2, 3], tf.float64)], jit_compile=True
        )
        def target(theta):
            return numerical_result(
                fixture_result(theta, rectangular=rectangular, joint=True)
            )

        return target

    manifest["hardware"] = (
        subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        if args.device == "GPU"
        else platform.processor()
    )
    manifest["environment"] = {
        key: os.environ.get(key)
        for key in (
            "CUDA_VISIBLE_DEVICES",
            "TF_FORCE_GPU_ALLOW_GROWTH",
            "TF_NUM_INTRAOP_THREADS",
            "TF_NUM_INTEROP_THREADS",
            "OPENBLAS_NUM_THREADS",
            "XLA_FLAGS",
        )
    }
    manifest["source_sha256"][str(Path(__file__).relative_to(ROOT))] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    with tf.device("/" + args.device + ":0"):
        stack = tf.random.stateless_normal([4, 9, 39], [20260916, 41], dtype=tf.float64)
        tangent = tf.random.stateless_normal(
            [4, 18, 9, 39], [20260916, 42], dtype=tf.float64
        )

        @tf.function(
            input_signature=[
                tf.TensorSpec([4, 9, 39], tf.float64),
                tf.TensorSpec([4, 18, 9, 39], tf.float64),
            ],
            jit_compile=True,
        )
        def stack_derivative(matrix, derivative):
            factor, d_factor, _ = batched_stack_qr_lower(
                matrix, derivative, compute_covariance_diagnostics=False
            )
            return factor, d_factor

        @tf.function(
            input_signature=[
                tf.TensorSpec([4, 39, 9], tf.float64),
                tf.TensorSpec([4, 18, 39, 9], tf.float64),
            ],
            jit_compile=True,
        )
        def rectangular_derivative(matrix, derivative):
            return _batched_qr_with_derivative(matrix, derivative)

        qualify("stack_qr_derivative", stack_derivative, (stack, tangent))
        qualify(
            "rectangular_qr_derivative",
            rectangular_derivative,
            (tf.linalg.matrix_transpose(stack), tf.linalg.matrix_transpose(tangent)),
        )
        manifest["primitive_fixture"] = {
            "batch": 4,
            "parameters": 18,
            "state": 9,
            "columns": 39,
            "seeds": [[20260916, 41], [20260916, 42]],
            "claim": "Repaired primitive timings only; no old/new speedup claim",
        }
        theta = tf.constant([[0.12, -0.17, 0.23], [-0.2, 0.19, -0.11]], tf.float64)
        for rectangular in [False, True]:
            name = "rectangular_target" if rectangular else "square_target"
            target = make_target(rectangular)
            out = qualify(name, target, (theta,))
            reference = numerical_result(
                fixture_result(
                    theta, rectangular=rectangular, joint=False, jit_compile=False
                )
            )
            if not all(
                bool(tf.reduce_all(tf.math.is_finite(value)).numpy())
                for value in (*out, *reference)
            ):
                raise AssertionError(f"{name}: nonfinite target output")
            errors = [
                float(tf.reduce_max(tf.abs(a - b)).numpy())
                for a, b in zip(out, reference)
            ]
            if max(errors) > 2e-8:
                raise AssertionError((name, errors))
            manifest[name]["reference_max_abs_errors"] = errors
            for i in range(3):
                delta = tf.one_hot(i, 3, dtype=tf.float64)[None] * 1e-5
                numerical = (target(theta + delta)[0] - target(theta - delta)[0]) / 2e-5
                if not bool(tf.reduce_all(tf.math.is_finite(numerical)).numpy()):
                    raise AssertionError(f"{name}: nonfinite finite difference")
                error = float(tf.reduce_max(tf.abs(numerical - out[1][:, i])).numpy())
                if error > 2e-6:
                    raise AssertionError((name, "score", error))
            manifest[name]["directional_score_passed"] = True
            manifest[name]["outputs"] = [x.numpy().tolist() for x in out]
            save()
        adapter = CompiledFilterMechanicsAdapter(rectangular=True)
        config = HMCTransitionArchiveConfig(
            max_results=3,
            step_size=0.03,
            num_leapfrog_steps=2,
            master_seed=(20260916, 37),
            use_xla=True,
            target_scope="compiled_filter_mechanics_diagnostic",
        )
        runner = build_hmc_transition_archive_runner(adapter, theta, config)
        tensors, valid, final = qualify(
            "native_hmc_archive",
            runner._runner,
            (theta, tf.constant(3, tf.int32), tf.constant(0, tf.int32)),
        )
        if not bool(tf.reduce_all(valid).numpy()):
            raise AssertionError("invalid archive transition")
        # Stateless RNG streams differ between TF and XLA backends. Replay
        # the exact archived momenta through independent leapfrog arithmetic,
        # rather than demand equality between different random trajectories.
        hmc_errors = {}
        for transition in range(3):
            q = tensors["pre_state"][transition]
            momentum = tensors["initial_momentum"][transition]
            value, grad = adapter.log_prob_and_grad(q)
            momentum = momentum + 0.5 * config.step_size * grad
            for leapfrog in range(config.num_leapfrog_steps):
                q = q + config.step_size * momentum
                value, grad = adapter.log_prob_and_grad(q)
                momentum = (
                    momentum
                    + (0.5 if leapfrog == config.num_leapfrog_steps - 1 else 1.0)
                    * config.step_size
                    * grad
                )
            for key, actual in [
                ("proposed_state", q),
                ("proposed_target_log_prob", value),
                ("proposed_grad_target_log_prob", grad),
                ("final_momentum", momentum),
            ]:
                error = float(
                    tf.reduce_max(tf.abs(actual - tensors[key][transition])).numpy()
                )
                hmc_errors[key] = max(hmc_errors.get(key, 0.0), error)
            expected_post = tf.where(
                tensors["is_accepted"][transition, :, None],
                q,
                tensors["pre_state"][transition],
            )
            error = float(
                tf.reduce_max(
                    tf.abs(expected_post - tensors["post_state"][transition])
                ).numpy()
            )
            hmc_errors["metropolis_state"] = max(
                hmc_errors.get("metropolis_state", 0.0), error
            )
        if max(hmc_errors.values()) > 2e-8:
            raise AssertionError(("HMC saved-momentum replay", hmc_errors))
        manifest["native_hmc_archive"]["saved_momentum_reference_max_abs_errors"] = (
            hmc_errors
        )
        manifest["native_hmc_archive"]["random_stream_comparison"] = (
            "TF and XLA streams differ; exact archived momentum replay used"
        )
        manifest["native_hmc_archive"]["final_state"] = final.numpy().tolist()
        manifest["native_hmc_archive"]["finite_active_tensors"] = all(
            bool(tf.reduce_all(tf.math.is_finite(v)).numpy())
            for v in tensors.values()
            if v.dtype.is_floating
        )
        manifest["native_hmc_archive"]["energy_identity_max_abs"] = float(
            tf.reduce_max(tf.abs(tensors["hamiltonian_identity_residual"])).numpy()
        )
        if not manifest["native_hmc_archive"]["finite_active_tensors"]:
            raise AssertionError("nonfinite transition")
        if manifest["native_hmc_archive"]["energy_identity_max_abs"] > 2e-8:
            raise AssertionError("Hamiltonian identity failed")
        if args.device == "GPU":
            manifest["allocator"] = tf.config.experimental.get_memory_info("GPU:0")
    manifest["peak_host_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    manifest["wall_seconds"] = time.perf_counter() - start
    manifest["status"] = "PASS_ENGINEERING_QUALIFICATION"
    save()
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "wall_seconds": manifest["wall_seconds"],
                "output": str(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()
