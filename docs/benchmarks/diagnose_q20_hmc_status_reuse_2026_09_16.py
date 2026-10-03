"""Bounded paired q20 GPU mechanics/timing; no posterior or speed ranking claim."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
from unittest.mock import patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--request", type=Path, help="master worker request with the exact protocol")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    from bayesfilter.inference.q20_production_config import digest, protocol_template, validate_protocol
    from bayesfilter.inference.q20_gpu_runtime import (
        GPUResourceUnavailable, select_worker_gpu, check_gpu_contention)
    request = {} if args.request is None else json.loads(args.request.read_text())
    config = validate_protocol(request.get("config", protocol_template()))
    if config["cpu_reference"] or not config["jit_compile"]:
        raise ValueError("status-reuse comparison requires the declared GPU/XLA diagnostic")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    record = {"schema": "bayesfilter.q20.status_reuse_diagnostic.v1", "status": "started",
        "role": "paired_numerical_mechanics_and_descriptive_execution_cost_only",
        "command": [sys.executable, *sys.argv], "source_root": str(root),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "environment": sys.executable, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "plan_file": request.get("plan_file", "docs/plans/bayesfilter-ssl-lstm-q20-remaining-efficiency-roadmap-2026-09-16.md"),
        "config": config, "config_hash": digest(config),
        "result_file": str(output / "result.json"), "measurements": {},
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}

    def save():
        record["wall_seconds"] = time.monotonic()-started
        (output / "result.json").write_text(json.dumps(record, indent=2, allow_nan=False)+"\n")

    save()

    def interrupted(signum, _frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, interrupted)
    try:
        if os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH") != "true":
            raise ValueError("memory growth must be set before TensorFlow import")
        # Approval or scheduling can delay process creation after an outer
        # readiness check. Select capacity again before importing TensorFlow.
        record["launch_readiness"] = select_worker_gpu()
        selected = record["launch_readiness"]["selected_host_gpu"]
        record["cuda_visible_devices"] = str(selected)

        def check_contention():
            receipt = check_gpu_contention(record["launch_readiness"])
            record.setdefault("contention_checks", []).append({
                "wall_seconds": time.monotonic()-started, **receipt})

        check_contention()
        save()
        import tensorflow as tf
        import tensorflow_probability as tfp
        from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
        record["memory_policy"] = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
        record.update(tensorflow=tf.__version__, tfp=tfp.__version__,
                      tf32=tf.config.experimental.tensor_float_32_execution_enabled())
        from bayesfilter.inference.q20_campaign_runtime import source_snapshot
        from bayesfilter.inference.q20_production_config import scoped_seed
        from bayesfilter.inference.q20_hmc_qualification import qualify_bridge, attach_qualification, check_full_chain_health
        from bayesfilter.inference.q20_production_hmc import draw_start_bank
        from bayesfilter.inference.tempered_target_tf import make_q20_tempered_bridge
        from bayesfilter.inference.hmc import FullChainHMCConfig, ReusableFullChainHMCRunner
        import bayesfilter.inference.hmc as hmc

        record["sources"] = source_snapshot(root)
        bridge = make_q20_tempered_bridge(config["target"]["q"], jit_compile=config["jit_compile"],
                                         principal_sqrt_backend=config["target"]["principal_sqrt_backend"])
        record.update(target_signature=bridge.target_signature, bridge_signature=bridge.signature,
                      dtype=config["target"]["dtype"], horizon=config["target"]["horizon"],
                      q=config["target"]["q"], backend=config["target"]["principal_sqrt_backend"],
                      status="qualifying")
        save()
        begin = time.monotonic()
        qualify_bridge(config, bridge, output / "qualification")
        attach_qualification(bridge, output / "qualification/result.json", config)
        record["qualification_seconds"] = time.monotonic()-begin
        record["qualification"] = str(output / "qualification/result.json")
        check_contention()
        save()
        adapter = bridge.fixed_beta_adapter(1.)
        starts, receipt = draw_start_bank(config, bridge, 1., "qualification")
        cfg = FullChainHMCConfig(num_results=2, num_burnin_steps=0, step_size=.01,
            num_leapfrog_steps=3, use_xla=True, seed=scoped_seed(config, "status-reuse", "hmc"),
            target_scope=adapter.target_scope, target_status_trace_policy="per_chain_step",
            capture_candidate_health=True)
        record.update(starts=starts.numpy().tolist(), start_receipt=receipt,
                      hmc_config=cfg.signature_payload(), seeds=[])

        def sync(result):
            for tensor in tf.nest.flatten(result):
                tensor.numpy()

        def inventory(compiled):
            concrete = compiled.get_concrete_function()
            definition = concrete.graph.as_graph_def()
            nodes = list(definition.node)+[n for fn in definition.library.function for n in fn.node_def]
            counts = Counter(n.op for n in nodes)
            bad = [name for name in counts if "PyFunc" in name or "HostCompute" in name]
            xla = concrete.function_def.attr["_XlaMustCompile"].b
            if bad or not xla or compiled.experimental_get_tracing_count() != 1:
                raise ValueError("callback, retracing or non-XLA diagnostic graph")
            return {"xla": xla, "traces": compiled.experimental_get_tracing_count(),
                    "callbacks": bad, "op_counts": dict(counts)}

        runners = {}
        record["graph_construction_seconds"] = {}
        for name in ("uncached", "accepted_status_reuse"):
            begin = time.monotonic()
            runner = ReusableFullChainHMCRunner(adapter, starts, cfg)
            if name == "uncached":
                # Diagnostic comparator: trace the same public runner with the
                # transparent wrapper disabled. No file/default is changed.
                with patch.object(hmc, "cache_hmc_target_status", lambda kernel, *_a, **_kw: kernel):
                    runner._runner.get_concrete_function()
            else:
                runner._runner.get_concrete_function()
            runners[name] = runner
            record["graph_construction_seconds"][name] = time.monotonic()-begin
            record["measurements"][name] = []
        record["comparator_override"] = "disable only cache_hmc_target_status at trace construction"
        record["max_float_difference"] = 0.
        for repeat in range(3):
            seed = scoped_seed(config, "performance-audit", "repeat", repeat)
            record["seeds"].append(list(seed))
            results = {}
            order = ("uncached", "accepted_status_reuse") if repeat % 2 == 0 else ("accepted_status_reuse", "uncached")
            for name in order:
                check_contention()
                begin = time.monotonic()
                result = runners[name].run(current_state=starts, seed=seed)
                sync((result.samples, result.trace))
                seconds = time.monotonic()-begin
                check_contention()
                check_full_chain_health(result)
                results[name] = result
                record["measurements"][name].append({"repeat": repeat, "seconds": seconds,
                    "role": "first_compile_execute" if repeat == 0 else "warm_execute",
                    "device": result.samples.device})
                print(name, repeat, seconds, flush=True)
                save()
            left = (results["uncached"].samples, results["uncached"].trace)
            right = (results["accepted_status_reuse"].samples, results["accepted_status_reuse"].trace)
            tf.nest.assert_same_structure(left, right)
            for a, b in zip(tf.nest.flatten(left), tf.nest.flatten(right)):
                if a.dtype.is_floating:
                    tf.debugging.assert_near(a, b, rtol=1e-9, atol=1e-10)
                    record["max_float_difference"] = max(record["max_float_difference"],
                                                        float(tf.reduce_max(tf.abs(a-b))))
                else:
                    tf.debugging.assert_equal(a, b)
        record["graphs"] = {name: inventory(runner._runner) for name, runner in runners.items()}
        record["paired_health_and_transition_checks"] = "passed"
        record["allocator"] = tf.config.experimental.get_memory_info("GPU:0")
        record["source_unchanged"] = source_snapshot(root) == record["sources"]
        if not record["source_unchanged"]:
            raise ValueError("source changed during diagnostic")
        record["status"] = "completed"
    except GPUResourceUnavailable as error:
        record.update(status="waiting_for_gpu", resource_receipt=error.receipt)
    except BaseException as error:
        record.update(status="failed", failure=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
