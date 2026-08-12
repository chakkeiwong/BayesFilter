#!/usr/bin/env python3
"""Bounded CPU-hidden T1 source-replica capacity gate."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from collections.abc import Mapping

if "--cpu-reference" in sys.argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/bayesfilter-mpl")

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

import tensorflow as tf  # noqa: E402

from bayesfilter.highdim.zhao_cui_austria_sir_fixed_variant_tf import (  # noqa: E402
    make_austria_sir_observed_data_target,
)
from bayesfilter.highdim.zhao_cui_austria_sir_source_replica_tf import (  # noqa: E402
    AuthorSIRSourceReplicaSpec,
    P70_HOLDOUT_REPLAY_NORMALIZED_RESIDUAL_VETO,
    fit_author_sir_t1_source_replica,
)


PLAN = "docs/plans/bayesfilter-zhao-cui-austria-sir-parameterized-source-replica-gap-closure-2026-07-30.md"
SCHEMA = "bayesfilter.zhao_cui_austria_sir_source_replica_t1_run.v1"


def _json_ready(value):
    if isinstance(value, Mapping):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(v) for v in value]
    if isinstance(value, tf.Tensor):
        raw = value.numpy()
        return raw.item() if value.shape.rank == 0 else raw.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cpu-reference", action="store_true")
    parser.add_argument("--fit-rank", type=int, default=1)
    parser.add_argument("--fit-sample-count", type=int, default=16)
    parser.add_argument("--holdout-sample-count", type=int, default=8)
    parser.add_argument("--train-steps", type=int, default=1)
    parser.add_argument("--optimizer-batch-size", type=int, default=8)
    parser.add_argument("--cdf-grid-size", type=int, default=9)
    parser.add_argument("--cdf-bisection-steps", type=int, default=4)
    parser.add_argument("--particle-count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=8615)
    args = parser.parse_args(argv)
    if args.output_root.exists():
        raise ValueError("output-root must be a fresh versioned directory")
    args.output_root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    target = make_austria_sir_observed_data_target()
    spec = AuthorSIRSourceReplicaSpec(
        fit_rank=args.fit_rank,
        fit_sample_count=args.fit_sample_count,
        holdout_sample_count=args.holdout_sample_count,
        train_steps=args.train_steps,
        optimizer_batch_size=args.optimizer_batch_size,
        cdf_grid_size=args.cdf_grid_size,
        cdf_bisection_steps=args.cdf_bisection_steps,
        kr_max_batch_working_bytes=64 * 1024 * 1024,
    )
    artifact = fit_author_sir_t1_source_replica(spec, seed=args.seed)
    summary = artifact.diagnostics["training_summary"]
    proposal = artifact.t1_algorithm3_diagnostic(
        particle_count=args.particle_count,
        seed=args.seed + 100,
    )
    fit_veto = bool(
        not tf.math.is_finite(tf.convert_to_tensor(summary["holdout_residual"])).numpy()
        or float(summary["holdout_residual"])
        > P70_HOLDOUT_REPLAY_NORMALIZED_RESIDUAL_VETO
    )
    proposal_pass = bool(
        proposal["finite"].numpy()
        and float(proposal["roundtrip_max_abs"].numpy()) <= 1e-4
        and float(proposal["ess_fraction"].numpy()) >= 0.5
    )
    primary_pass = bool(proposal_pass and not fit_veto)
    status = (
        "PASS_T1_SOURCE_REPLICA_ESS_ROUNDTRIP"
        if primary_pass
        else "BLOCK_T1_SOURCE_REPLICA_FIT_OR_PROPOSAL_GATE"
    )
    result = {
        "schema": SCHEMA,
        "plan": PLAN,
        "status": status,
        "primary_pass": primary_pass,
        "source_fit_artifact_status": "PASS_FINITE_SERIALIZED_AUTHOR_SHAPED_T1_CANDIDATE",
        "source_fit_backend": artifact.diagnostics["fit_backend"],
        "target_identity": target.target_identity,
        "target": target.manifest,
        "spec": spec.manifest_payload(),
        "memory_forecast": spec.memory_forecast(particle_count=args.particle_count),
        "artifact": artifact.payload(),
        "diagnostics": {
            "fit_residual": summary["fit_residual"],
            "holdout_residual": summary["holdout_residual"],
            "normalizer": summary["normalizer"],
            "runtime_seconds": summary["runtime_seconds"],
            "forward_adapter_status": "implemented_block_upper_author_order_upper_conditional",
            "fit_holdout_veto": fit_veto,
            "fit_holdout_residual_veto_threshold": (
                P70_HOLDOUT_REPLAY_NORMALIZED_RESIDUAL_VETO
            ),
            "fit_holdout_residual_veto_provenance": (
                "bayesfilter.highdim.source_route."
                "P70_HOLDOUT_REPLAY_NORMALIZED_RESIDUAL_VETO"
            ),
            "proposal": proposal,
        },
        "continuation_veto": (
            None if primary_pass else "do_not_continue_to_t2_t20_or_parameter_score"
        ),
        "nonclaims": (
            "no proposal-quality pass",
            "no exact likelihood or pseudo-marginal claim",
            "no source-faithful assembled forward route",
            "no parameter value/score claim",
            "no GPU/XLA or HMC claim",
        ),
        "command": " ".join(sys.argv),
        "git": {
            "commit": subprocess.run(
                ("git", "rev-parse", "HEAD"),
                cwd=REPOSITORY_ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip(),
            "worktree_dirty": bool(
                subprocess.run(
                    ("git", "status", "--short"),
                    cwd=REPOSITORY_ROOT,
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip()
            ),
        },
        "environment": {
            "python": sys.executable,
            "tensorflow": tf.__version__,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "intentional_cpu_reference": bool(args.cpu_reference),
        },
        "wall_time_seconds": time.monotonic() - started,
    }
    payload = _json_ready(result)
    (args.output_root / "result.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_root / "result.md").write_text(
        "# Zhao-Cui Austria SIR T1 Source-Replica Gate\n\n"
        f"Status: `{payload['status']}`\n\n"
        f"Artifact: `{args.output_root / 'result.json'}`\n",
        encoding="utf-8",
    )
    return 0 if primary_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
