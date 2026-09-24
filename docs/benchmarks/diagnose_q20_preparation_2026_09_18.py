#!/usr/bin/env python3
"""Replay one q20 preparation failure with the existing public progress hook.

Numerical code is imported from the recorded, unchanged execution checkout.
This diagnostic neither changes preparation settings nor issues tuning authority.
The caller supplies an existing Campaign supervisor and its external deadline.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--previous-attempt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source_root.resolve()))
    from bayesfilter.inference.q20_campaign_runtime import atomic_json, source_snapshot
    from bayesfilter.inference.q20_production_config import digest
    request = json.loads((args.previous_attempt / "request.json").read_text())
    campaign = json.loads((args.campaign / "campaign.json").read_text())
    config = request["config"]
    if source_snapshot(args.source_root) != campaign["sources"]:
        raise ValueError("diagnostic numerical sources changed")
    if digest(config) != campaign["config_hash"]:
        raise ValueError("diagnostic protocol changed")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    record = {"schema": "bayesfilter.q20.preparation_failure_diagnostic.v1",
        "started_at": datetime.now(timezone.utc).isoformat(), "status": "initializing",
        "git_commit": campaign["git_commit"], "sources": campaign["sources"],
        "source_root": str(args.source_root), "previous_attempt": str(args.previous_attempt),
        "request": request, "config_hash": digest(config), "command": [sys.executable, *sys.argv],
        "python": sys.executable, "pid": os.getpid(), "seeds": config["seed"],
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "plan_file": request["plan_file"], "result_file": str(args.output_dir / "result.json"),
        "production_qualified": False, "posterior_qualified": False}

    def save():
        record["wall_seconds"] = time.monotonic() - started
        atomic_json(args.output_dir / "result.json", record)

    save()
    try:
        if os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH", "").lower() != "true":
            raise ValueError("memory growth must be configured before TensorFlow import")
        from bayesfilter.inference.q20_gpu_runtime import select_worker_gpu, check_gpu_contention
        record["launch_readiness"] = select_worker_gpu()
        save()
        import tensorflow as tf
        from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
        record.update(memory_policy=configure_tensorflow_gpu_memory_growth(tf, require_gpu=True),
            tensorflow=tf.__version__, cuda_visible_devices=os.environ["CUDA_VISIBLE_DEVICES"],
            jit_compile=config["jit_compile"], tf32=tf.config.experimental.tensor_float_32_execution_enabled())
        from bayesfilter.inference.tempered_target_tf import make_q20_tempered_bridge
        from bayesfilter.inference.q20_hmc_qualification import attach_qualification
        from bayesfilter.inference.q20_production_hmc import draw_start_bank
        from bayesfilter.inference.hmc_kernel_tuning import HMCKernelTuningConfig, prepare_operational_windowed_mass_handoff
        from bayesfilter.inference.hmc_preparation import HMCPreparationProgress
        bridge = make_q20_tempered_bridge(config["target"]["q"], jit_compile=config["jit_compile"],
            principal_sqrt_backend=config["target"]["principal_sqrt_backend"])
        bridge = attach_qualification(bridge, request["qualification_path"], config)
        starts, _ = draw_start_bank(config, bridge, request["beta"], "pricing")
        previous_data = args.previous_attempt / "worker/data"
        expected = json.loads((previous_data / "starts.json").read_text())["positions"]
        tf.debugging.assert_equal(starts, tf.constant(expected, tf.float64))
        adapter = bridge.fixed_beta_adapter(request["beta"])
        cfg = HMCKernelTuningConfig.serious(target_scope=adapter.target_scope,
            use_xla=config["jit_compile"], chain_execution_mode="tf_function",
            target_status_trace_policy="per_chain_step", metric_update_requirement="require_operational_update",
            public_timeout_budget_s=min(config["tuning"]["max_wall_seconds"], request["max_seconds"]))
        if digest(cfg.payload()) != digest(json.loads((previous_data / "preparation-policy.json").read_text())):
            raise ValueError("preparation settings differ from failed attempt")
        record.update(status="running", starts_exactly_match=True, preparation_settings_exactly_match=True,
            target_signature=bridge.target_signature, bridge_signature=bridge.signature)
        save()
        with HMCPreparationProgress(args.output_dir, max_wall_time_seconds=request["max_seconds"]) as progress:
            prepare_operational_windowed_mass_handoff(adapter=adapter, initial_position=starts[0], config=cfg,
                parameter_scales=tf.fill([bridge.parameter_dim], tf.constant(4., tf.float64)),
                progress_callback=progress.phase)
        record["status"] = "preparation_completed_on_replay"
    except Exception as error:
        record.update(status="preparation_replay_failed", error_type=type(error).__name__,
            message=str(error), traceback=traceback.format_exc())
    finally:
        if record.get("launch_readiness"):
            try:
                record["terminal_capacity"] = check_gpu_contention(record["launch_readiness"])
            except Exception as error:
                record["terminal_capacity_error"] = str(error)
        record["preparation_progress_file"] = str(args.output_dir / "preparation_progress.json")
        save()
    print(json.dumps({k: record[k] for k in ("status", "wall_seconds", "result_file")}, indent=2))
    return 0 if record["status"] == "preparation_completed_on_replay" else 1


if __name__ == "__main__":
    raise SystemExit(main())
