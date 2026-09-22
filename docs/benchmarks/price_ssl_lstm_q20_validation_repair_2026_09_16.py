"""Bounded GPU pricing and prefix parity, never learned-map quality evidence."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(source))
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    record = {"schema": "bayesfilter.q20.validation_repair_pricing.v1",
        "status": "initializing", "role": "pricing_and_engineering_parity_only",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv], "python": sys.executable,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "plan_file": "docs/plans/bayesfilter-ssl-lstm-q20-validation-budget-repair-plan-2026-09-16.md",
        "result_file": str(output / "result.json"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "tf_force_gpu_allow_growth": os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH"),
        "production_qualified": False}

    def save():
        record["wall_seconds"] = time.monotonic() - started
        (output / "result.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")

    save()
    try:
        if record["tf_force_gpu_allow_growth"] != "true":
            raise ValueError("memory growth must be enabled before TensorFlow import")
        import tensorflow as tf
        import tensorflow_probability as tfp
        from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
        record["memory_policy"] = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
        record.update(tensorflow=tf.__version__, tfp=tfp.__version__,
                      tf32=tf.config.experimental.tensor_float_32_execution_enabled())
        from bayesfilter.inference.q20_production_config import protocol_template, scoped_seed
        from bayesfilter.inference.q20_production_training import (
            price_training, training_quote, new_session, scope_for, source_snapshot)
        from bayesfilter.inference.neutra_training_protocol import HeldoutLoss
        from bayesfilter.inference.q20_training_validation import FrozenLossCache
        from bayesfilter.inference.tempered_target_tf import make_q20_tempered_bridge
        config = protocol_template()
        bridge = make_q20_tempered_bridge(20, jit_compile=True, principal_sqrt_backend="tensorflow_eigh_strict")
        record.update(config=config, sources=source_snapshot(), seeds=config["seed"],
            target_signature=bridge.target_signature, bridge_signature=bridge.signature,
            data_version=bridge.target_signature, status="pricing")
        save()
        priced = price_training(config, bridge, output / "training", memory_policy=record["memory_policy"])
        if not all("GPU" in row["training_device"] for row in priced["rows"]):
            raise ValueError("pricing update variables must reside on GPU")
        record["pricing"] = priced
        record["budget_scenarios"] = training_quote(config, priced)
        record["status"] = "checking_cache_parity"
        save()
        candidate = {"id": "pricing-cache-parity", "width": config["training"]["widths"][0],
                     "learning_rate": config["training"]["learning_rates"][0], "root": 0}
        scope = scope_for(config, bridge, candidate, sources=record["sources"], memory_policy=record["memory_policy"])
        session = new_session(config, bridge, candidate, scope=scope)
        session = session.next_beta(1., root_seed=scoped_seed(config, "cache-parity-train"),
                                    preflight_seed=scoped_seed(config, "cache-parity-preflight"))
        session.advance(1)
        snapshot = session.checkpoint()["map"]
        cache = FrozenLossCache(output / "cache-parity", bridge=bridge, scope=scope,
                               batch_size=32, jit_compile=True)
        seed = scoped_seed(config, "cache-parity-bank")
        prefix = cache.evaluate(snapshot, 32, seed)
        extended = cache.evaluate(snapshot, 64, seed)
        replay = cache.evaluate(snapshot, 64, seed)
        uncached = HeldoutLoss(session.trainer.transport, bridge, 1., batch_size=32, jit_compile=True)(64, seed)
        tf.debugging.assert_equal(prefix, extended[:32])
        tf.debugging.assert_equal(replay, extended)
        tf.debugging.assert_near(extended, uncached, atol=1e-10, rtol=1e-9)
        if cache.evaluated_rows != 64:
            raise ValueError("incremental cache re-evaluated a prefix")
        graphs = [entry["evaluator"].compiled for entry in cache.entries.values()]
        for graph in graphs:
            concrete = graph.get_concrete_function()
            if graph.experimental_get_tracing_count() != 1 or not concrete.function_def.attr["_XlaMustCompile"].b:
                raise ValueError("cached evaluator lost single-trace XLA execution")
            definition = concrete.graph.as_graph_def()
            nodes = list(definition.node) + [node for fn in definition.library.function for node in fn.node_def]
            if any("PyFunc" in node.op or "HostCompute" in node.op for node in nodes):
                raise ValueError("host callback in validation graph")
        record["cache_parity"] = {"passed": True, "evaluated_rows": cache.evaluated_rows,
            "reused_rows": cache.reused_rows, "max_loss_difference": float(tf.reduce_max(tf.abs(extended-uncached)).numpy()),
            "traces": [graph.experimental_get_tracing_count() for graph in graphs],
            "jit_compile": True, "endpoint_device": extended.device, "atol": 1e-10, "rtol": 1e-9}
        record["gpu_allocator"] = tf.config.experimental.get_memory_info("GPU:0")
        if record["sources"] != source_snapshot():
            raise ValueError("source changed while pricing")
        record["status"] = "complete"
        save()
        print(json.dumps({"status": "complete", "wall_seconds": record["wall_seconds"],
                          "scenarios": record["budget_scenarios"]["scenarios"]}, indent=2))
    except BaseException as error:
        record.update(status="failed", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        save()
        raise


if __name__ == "__main__":
    main()
