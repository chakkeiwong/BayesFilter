"""Bounded diagnostic of the frozen q20 target and public HMC execution.

This file changes no runtime defaults and issues no tuning/qualification receipt.
Run under the external timeout in the companion performance-audit plan.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source, output = args.source_root.resolve(), args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(source))
    record = {
        "schema": "bayesfilter.q20.hmc_performance_diagnostic.v1",
        "role": "explanatory_only_no_promotion",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "source_root": str(source),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip(),
        "numerical_baseline_commit": "71e0fba399489a8f25fcbdea1185f6bbb600e487",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "plan_file": "docs/plans/bayesfilter-ssl-lstm-q20-hmc-performance-audit-2026-09-16.md",
        "result_file": str(output / "result.json"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "tf_force_gpu_allow_growth": os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH"),
        "seeds": [20260915, 150001],
        "measurements": {},
        "status": "initializing",
    }

    def save():
        record["wall_seconds"] = time.monotonic() - started
        write_json(output / "result.json", record)

    save()
    try:
        if record["tf_force_gpu_allow_growth"] != "true":
            raise ValueError("TF_FORCE_GPU_ALLOW_GROWTH=true is required before import")
        import tensorflow as tf
        import tensorflow_probability as tfp
        from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth

        record["memory_policy"] = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
        record["tensorflow"], record["tfp"] = tf.__version__, tfp.__version__
        record["tf32"] = tf.config.experimental.tensor_float_32_execution_enabled()
        from bayesfilter.inference.q20_production_training import source_snapshot
        from bayesfilter.inference.q20_production_config import scoped_seed
        from bayesfilter.inference.q20_hmc_qualification import attach_qualification, check_full_chain_health
        from bayesfilter.inference.q20_production_hmc import draw_start_bank
        from bayesfilter.inference.tempered_target_tf import make_q20_tempered_bridge
        from bayesfilter.inference.hmc import (
            FullChainHMCConfig, ReusableFullChainHMCRunner,
            build_independent_chain_tfp_hmc_runner,
        )

        historical = source / "docs/plans/artifacts/ssl-lstm-q20-executable-master-2026-09-16/real-q20-r3"
        manifest = json.loads((historical / "attempts/00002-qualify-beta1/worker/manifest.json").read_text())
        if source_snapshot() != manifest["sources"]:
            raise ValueError("current source inventory differs from the executed master")
        record["source_inventory_matches_executed_master"] = True
        record["sources"] = manifest["sources"]
        config = manifest["request"]["config"]
        bridge = make_q20_tempered_bridge(20, jit_compile=True, principal_sqrt_backend="tensorflow_eigh_strict")
        attach_qualification(bridge, historical / "qualification.json", config)
        adapter = bridge.fixed_beta_adapter(1.0)
        starts, start_receipt = draw_start_bank(config, bridge, 1.0, "qualification")
        record.update(
            target_signature=bridge.target_signature, bridge_signature=bridge.signature,
            start_receipt=start_receipt, starts=starts.numpy().tolist(),
            target_dtype="float64", principal_sqrt_backend="tensorflow_eigh_strict",
            status="measuring", initialization_seconds=time.monotonic() - started,
        )
        save()

        def sync(value):
            for tensor in tf.nest.flatten(value):
                if tf.is_tensor(tensor):
                    tensor.numpy()

        def measure(label, call, validate):
            times = []
            last = None
            for index in range(3):
                begin = time.monotonic()
                last = call(index)
                sync(last)
                times.append(time.monotonic() - begin)
                validate(last)
            record["measurements"][label] = {"first_seconds": times[0], "warm_seconds": times[1:]}
            save()
            print(label, json.dumps(record["measurements"][label]), flush=True)
            return last

        def graph_inventory(compiled):
            concrete = compiled.get_concrete_function()
            definition = concrete.graph.as_graph_def()
            nodes = list(definition.node)
            for function in definition.library.function:
                nodes.extend(function.node_def)
            counts = Counter(node.op for node in nodes)
            callbacks = {op: count for op, count in counts.items() if "PyFunc" in op or "HostCompute" in op}
            result = {
                "xla_must_compile": concrete.function_def.attr["_XlaMustCompile"].b,
                "tracing_count": compiled.experimental_get_tracing_count(),
                "input_signature": str(compiled.input_signature),
                "function_count": len(definition.library.function),
                "op_counts": dict(counts), "callback_ops": callbacks,
            }
            if callbacks or not result["xla_must_compile"]:
                raise ValueError("compiled graph has a callback or lacks XLA")
            return result

        def checked_target(theta):
            value, score, status = adapter.log_prob_and_grad_status(theta)
            return value, score, status["valid_pre_regularized_score"]

        def valid_target(result):
            tf.debugging.assert_all_finite(result[0], "value")
            tf.debugging.assert_all_finite(result[1], "score")
            tf.debugging.assert_equal(tf.reduce_all(result[2]), True)

        targets = {size: tf.function(checked_target,
            input_signature=(tf.TensorSpec([size, 4], tf.float64),),
            jit_compile=True, reduce_retracing=False) for size in (1, 4)}
        scalar = measure("target_batch1", lambda _: targets[1](starts[:1]), valid_target)
        batched = measure("target_batch4", lambda _: targets[4](starts), valid_target)
        tf.debugging.assert_near(scalar[0], batched[0][:1], rtol=1e-9, atol=1e-10)
        tf.debugging.assert_near(scalar[1], batched[1][:1], rtol=1e-9, atol=1e-10)
        record["graphs"] = {f"target_batch{size}": graph_inventory(fn) for size, fn in targets.items()}
        record["target_batch_parity_first_row"] = True

        from bayesfilter.nonlinear.experimental_batched_svd_sigma_point_tf import (
            _checked_batched_principal_sqrt_factor_first_derivatives,
        )
        model, derivatives = bridge.component_target._batched_components(starts)
        covariance = tf.linalg.LinearOperatorBlockDiag([
            tf.linalg.LinearOperatorFullMatrix(model.initial_covariance),
            tf.linalg.LinearOperatorFullMatrix(model.innovation_covariance),
        ]).to_dense()
        dimension = int(covariance.shape[-1])
        direction = tf.broadcast_to(tf.eye(dimension, dtype=tf.float64), [4, 4, dimension, dimension])

        @tf.function(input_signature=(tf.TensorSpec([4, dimension, dimension], tf.float64),
                                      tf.TensorSpec([4, 4, dimension, dimension], tf.float64)),
                     jit_compile=True, reduce_retracing=False)
        def factor_program(matrix, directions):
            result = _checked_batched_principal_sqrt_factor_first_derivatives(
                matrix, directions, singular_floor=tf.constant(0., tf.float64),
                fixed_null_tolerance=tf.constant(1e-10, tf.float64),
                factor_backend="tensorflow_eigh_strict", label="diagnostic_initial_covariance")
            return result.factor, result.d_factor

        def valid_factor(result):
            for tensor in result:
                tf.debugging.assert_all_finite(tensor, "factor diagnostic")

        measure("checked_factor_initial_covariance_batch4", lambda _: factor_program(covariance, direction), valid_factor)
        record["placement_dimension"] = dimension
        record["state_dimension"] = int(model.initial_mean.shape[-1])
        record["innovation_dimension"] = int(model.innovation_covariance.shape[-1])
        record["horizon"] = int(bridge.component_target.config.static_config.horizon)
        record["factor_probe_directions"] = "four identity directions; diagnostic, not actual trajectory score"

        cfg = FullChainHMCConfig(num_results=2, num_burnin_steps=0,
            step_size=.01, num_leapfrog_steps=3, use_xla=True,
            seed=scoped_seed(config, "performance-audit", "hmc"), target_scope=adapter.target_scope,
            target_status_trace_policy="per_chain_step", capture_candidate_health=True)
        record["hmc_config"] = {"chains": 4, "transitions": 2, "L": 3, "epsilon": .01,
                                "root_seed": list(cfg.seed), "health_capture": True}
        serial_runner = build_independent_chain_tfp_hmc_runner(adapter, starts, cfg)
        batch_runner = ReusableFullChainHMCRunner(adapter, starts, cfg)

        def run(runner, index, serial):
            seed = scoped_seed(config, "performance-audit", "repeat", index)
            result = (runner.run(current_state=starts, root_seed=seed, mode="serial") if serial
                      else runner.run(current_state=starts, seed=seed))
            # Explicit synchronization includes every trace field, not just samples.
            sync((result.samples, result.trace))
            return result

        for label, runner, serial in (("public_serial_hmc", serial_runner, True),
                                      ("public_batched_hmc", batch_runner, False)):
            result = measure(label, lambda i: run(runner, i, serial), check_full_chain_health)
            record["measurements"][label]["endpoint_device"] = result.samples.device
            record["measurements"][label]["sample_shape"] = result.samples.shape.as_list()
            record["graphs"][label] = ([graph_inventory(child._runner) for child in runner._runners]
                                       if serial else graph_inventory(runner._runner))
            save()
        record["gpu_allocator"] = tf.config.experimental.get_memory_info("GPU:0")
        record["status"] = "completed"
    except BaseException as error:
        record.update(status="failed", failure_type=type(error).__name__,
                      failure=str(error), traceback=traceback.format_exc())
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
