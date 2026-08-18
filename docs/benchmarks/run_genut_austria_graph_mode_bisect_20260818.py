#!/usr/bin/env python3
"""Graph-mode value/JVP divergence bisection for the Austria GenUT endpoint.

Diagnostic lane under
docs/plans/bayesfilter-austria-genut-graph-mode-divergence-localization-plan-2026-08-18.md.
Reuses the frozen root-cause runner's endpoint arithmetic, identity guards,
and manifest by import; adds parameterized (horizon, correction_steps, mode)
cases and an optional grappler-disable probe. Promotes nothing.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Importing the base runner performs the early --device/--gpu-index
# environment and memory-growth configuration against THIS process argv.
from docs.benchmarks import (  # noqa: E402
    run_genut_austria_endpoint_root_cause_20260817 as base,
)

import tensorflow as tf  # noqa: E402


def _parse_case(text: str) -> tuple[int, int, str]:
    horizon_text, steps_text, mode = text.split(":")
    horizon = int(horizon_text)
    steps = int(steps_text)
    if mode not in ("eager", "graph", "xla"):
        raise ValueError(f"unknown mode in case {text!r}")
    if not 1 <= horizon <= 20:
        raise ValueError(f"horizon out of frozen range in case {text!r}")
    if steps < 0:
        raise ValueError(f"negative correction steps in case {text!r}")
    return horizon, steps, mode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--gpu-index", choices=("0", "1"), default="0")
    parser.add_argument(
        "--cases",
        nargs="+",
        required=True,
        help="cases as horizon:steps:mode, e.g. 2:4:graph",
    )
    parser.add_argument(
        "--grappler",
        choices=("default", "meta_off"),
        default="default",
        help="meta_off sets disable_meta_optimizer=True before any tracing",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    started = time.time()
    if args.device != base._EARLY_DEVICE:  # noqa: SLF001
        raise RuntimeError("parsed device does not match pre-import device selection")
    if args.grappler == "meta_off":
        tf.config.optimizer.set_experimental_options(
            {"disable_meta_optimizer": True}
        )
    grappler_options = dict(tf.config.optimizer.get_experimental_options())

    with tf.device("/CPU:0"):
        target = base.make_genut_neutra_target("austria_sir", particle_count=1008)
    target_hashes = {
        "observations": base._tensor_hash(target.observations),  # noqa: SLF001
        "initial_noise": base._tensor_hash(target.initial_noise),  # noqa: SLF001
        "process_noise": base._tensor_hash(target.process_noise),  # noqa: SLF001
        "design": base._tensor_hash(target.design),  # noqa: SLF001
    }
    if target.target_signature != base.EXPECTED_TARGET_SIGNATURE:
        raise RuntimeError("frozen target signature mismatch")
    if target.adapter_signature() != base.EXPECTED_ADAPTER_SIGNATURE:
        raise RuntimeError("frozen adapter signature mismatch")
    if target_hashes != base.EXPECTED_TARGET_HASHES:
        raise RuntimeError("frozen target tensor hash mismatch")

    source_paths = (
        ROOT / "bayesfilter/highdim/cubature_genut_batch_tf.py",
        ROOT / "bayesfilter/highdim/cubature_genut_neutra_targets.py",
        ROOT / "bayesfilter/highdim/cubature_genut_batch_adapters.py",
        ROOT / "bayesfilter/highdim/higher_moment_contract_e.py",
        ROOT / "bayesfilter/highdim/cubature_genut_filter.py",
        Path(__file__).resolve(),
        Path(base.__file__).resolve(),
    )
    payload: dict[str, Any] = base._manifest(  # noqa: SLF001
        output=output,
        device=args.device,
        memory_policy=base._MEMORY_POLICY,  # noqa: SLF001
        started=started,
    )
    payload.update(
        {
            "schema": "bayesfilter.genut_austria_graph_mode_bisect_result.v1",
            "plan": (
                "docs/plans/"
                "bayesfilter-austria-genut-graph-mode-divergence-"
                "localization-plan-2026-08-18.md"
            ),
            "source_sha256": {
                base._display_path(path): base._sha256_file(path)  # noqa: SLF001
                for path in source_paths
            },
            "target_signature": target.target_signature,
            "adapter_signature": target.adapter_signature(),
            "target_construction_device": "/CPU:0",
            "execution_device": "/GPU:0" if args.device == "gpu" else "/CPU:0",
            "target_hashes": target_hashes,
            "frozen_identity_guard": "PASS",
            "route_classification": "batch_diagonal_candidate",
            "grappler_mode": args.grappler,
            "grappler_experimental_options": grappler_options,
            "results": {"cases": []},
        }
    )
    base._write_json(output, payload | {"status": "RUNNING"})  # noqa: SLF001
    try:
        execution_device = "/GPU:0" if args.device == "gpu" else "/CPU:0"
        with tf.device(execution_device):
            for text in args.cases:
                horizon, steps, mode = _parse_case(text)
                case_started = time.time()
                record = base._endpoint(  # noqa: SLF001
                    target,
                    horizon=horizon,
                    correction_steps=steps,
                    mode=mode,
                )
                record["case_wall_seconds"] = time.time() - case_started
                payload["results"]["cases"].append(record)
                base._write_json(output, payload | {"status": "RUNNING"})  # noqa: SLF001
        payload["status"] = "COMPLETE"
    except Exception as exc:
        payload["status"] = "FAILED"
        payload["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        payload["wall_seconds"] = time.time() - started
        base._write_json(output, payload)  # noqa: SLF001


if __name__ == "__main__":
    main()
