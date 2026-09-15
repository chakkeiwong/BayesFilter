#!/usr/bin/env python3
"""Explicit q20 production protocol entrypoint.

``validate`` is metadata-only. ``price`` measures the configured batched
training graph. ``train`` executes the declared development cohort with fresh
versioned outputs and resumable complete checkpoints. Outputs remain
development evidence until heldout learning, public tuning, posterior
precision, and reference gates pass.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bayesfilter.inference.q20_production_config import (
    digest, load_protocol, protocol_template, validate_protocol, write_json,
)


def _load(path):
    return load_protocol(path) if path else validate_protocol(protocol_template())


def _bridge(config):
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    if config["cpu_reference"]:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    import tensorflow as tf
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    memory = configure_tensorflow_gpu_memory_growth(tf, require_gpu=not config["cpu_reference"])
    from bayesfilter.inference.tempered_target_tf import make_q20_tempered_bridge
    bridge = make_q20_tempered_bridge(config["target"]["q"], jit_compile=config["jit_compile"],
                                      principal_sqrt_backend=config["target"]["principal_sqrt_backend"])
    return bridge, memory


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("validate", "price", "train"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-dir", type=Path, required=False)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--stop-after-rung", type=int)
    parser.add_argument("--cpu-reference", action="store_true",
                        help="smoke-only CPU exception; cannot be development evidence")
    args = parser.parse_args(argv)
    config = _load(args.config)
    if args.cpu_reference:
        config = json.loads(json.dumps(config))
        config["role"], config["cpu_reference"], config["jit_compile"] = "smoke", True, False
        validate_protocol(config)
    if args.mode == "validate":
        print(json.dumps({"schema": config["schema"], "config_hash": digest(config),
                          "role": config["role"], "promotion_eligible": False,
                          "cohort_size": len(__import__("bayesfilter.inference.q20_production_config", fromlist=["training_cohort"]).training_cohort(config))}, indent=2))
        return 0
    if args.output_dir is None:
        parser.error("--output-dir is required for price/train")
    output = args.output_dir
    if output.exists():
        parser.error("output directory must be fresh and versioned")
    bridge, memory = _bridge(config)
    started = time.monotonic()
    if args.mode == "price":
        from bayesfilter.inference.q20_production_training import price_training
        result = price_training(config, bridge, output, memory_policy=memory)
    else:
        from bayesfilter.inference.q20_production_training import run_training_cohort
        result = run_training_cohort(config, bridge, output, memory_policy=memory,
            max_seconds=args.max_seconds or config["budget"]["repair_allocation_seconds"],
            resume=args.resume_checkpoint, stop_after_rung=args.stop_after_rung)
    result.update({"mode": args.mode, "config_hash": digest(config),
                   "wall_seconds": time.monotonic() - started,
                   "promotion_eligible": False,
                   "interpretation": "software/development evidence only; no q20 posterior claim"})
    write_json(output / "entrypoint-result.json", result, exclusive=False)
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
