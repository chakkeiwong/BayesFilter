#!/usr/bin/env python3
"""Diagnose the TF32 post-quotient marginal floor without retuning."""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf

from docs.benchmarks import run_canonical_lgssm_fused_ot_loop_repair as runner
from docs.benchmarks import run_canonical_lgssm_tf32_balance_selection as selection


SCHEMA_VERSION = "bayesfilter.canonical_lgssm_tf32_marginal_floor_diagnostic.v1"
CAMPAIGN_ID = "canonical-lgssm-tf32-balance-horizon-continuation-20260718"
DIAGNOSTIC_BALANCE_STEPS = 2


def _direct_pairwise_squared_cross(
    query: tf.Tensor, key: tf.Tensor
) -> tf.Tensor:
    difference = query[:, :, None, :] - key[:, None, :, :]
    return tf.reduce_sum(tf.square(difference), axis=-1)


def _direct_half_pairwise_squared_cross_jvp(
    query: tf.Tensor,
    key: tf.Tensor,
    d_query: tf.Tensor,
    d_key: tf.Tensor,
) -> tf.Tensor:
    difference = query[:, :, None, :] - key[:, None, :, :]
    tangent_difference = d_query[:, :, None, :, :] - d_key[:, None, :, :, :]
    return tf.reduce_sum(
        difference[:, :, :, :, None] * tangent_difference,
        axis=3,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--attempt-id", required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": CAMPAIGN_ID,
        "attempt_id": args.attempt_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "git_commit": runner._git_commit(),
        "role": "explanatory_diagnostic_only",
        "balance_steps": DIAGNOSTIC_BALANCE_STEPS,
        "seeds": list(selection.DESIGN_SEEDS),
        "forbidden_use": "must not select a new count or relax a gate",
        "mechanism": (
            "replace TF32-eligible dot-product squared distances with direct "
            "float32 coordinate differences for this process only"
        ),
    }
    started = time.perf_counter()
    try:
        payload["device"] = runner._configure_gpu(tf.float32)
        from bayesfilter.highdim import ledh_contract_e_streaming_tf as streaming
        from experiments.dpf_implementation.tf_tfp.resampling import (
            annealed_transport_tf,
        )

        annealed_transport_tf._pairwise_squared_cross = (  # noqa: SLF001
            _direct_pairwise_squared_cross
        )
        annealed_transport_tf._half_pairwise_squared_cross_jvp = (  # noqa: SLF001
            _direct_half_pairwise_squared_cross_jvp
        )
        streaming._pairwise_squared_cross = _direct_pairwise_squared_cross  # noqa: SLF001
        streaming._half_pairwise_squared_cross_jvp = (  # noqa: SLF001
            _direct_half_pairwise_squared_cross_jvp
        )
        payload["result"] = selection._evaluate(
            selection.DESIGN_SEEDS, DIAGNOSTIC_BALANCE_STEPS
        )
        payload["status"] = "diagnostic_complete"
    except Exception as error:
        payload["status"] = "diagnostic_failed"
        payload["exception"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
    payload["wall_time_seconds"] = time.perf_counter() - started
    runner._write_exclusive(output, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "wall_time_seconds": payload["wall_time_seconds"],
            },
            indent=2,
        )
    )
    if payload["status"] != "diagnostic_complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
