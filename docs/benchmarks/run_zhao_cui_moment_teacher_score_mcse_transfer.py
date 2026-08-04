#!/usr/bin/env python3
"""Run one precision arm for the moment-teacher score/MCSE transfer diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import statistics

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf

from bayesfilter.runtime.gpu_memory_policy import (
    configure_tensorflow_gpu_memory_growth,
)


PLAN = "docs/plans/bayesfilter-zhao-cui-moment-teacher-score-mcse-transfer-plan-2026-07-30.md"
PLAN_N4096 = "docs/plans/bayesfilter-zhao-cui-moment-teacher-score-mcse-transfer-n4096-plan-2026-07-30.md"
SCHEMA = "bayesfilter.zhao_cui_moment_teacher_score_mcse_transfer_node.v1"
CAMPAIGN = "zhao-cui-moment-teacher-score-mcse-transfer-20260730"
SEEDS = tuple(range(81700, 81716))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _parse_seeds(text: str) -> tuple[int, ...]:
    seeds = tuple(int(token) for token in text.split(",") if token.strip())
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be a nonempty unique comma-separated list")
    return seeds


def _combine_batch_results(results: list[dict[str, object]]) -> dict[str, object]:
    first = results[0]
    seeds = [seed for result in results for seed in result["estimator_seeds"]]
    values = [value for result in results for value in result["per_seed_value"]]
    scores = [
        row for result in results for row in result["per_seed_physical_score"]
    ]
    parameter_count = len(scores[0])
    combined = {
        key: first[key]
        for key in (
            "device",
            "git_commit",
            "source_sha256",
            "campaign_id",
            "plan_path",
            "arm",
            "time_steps",
            "num_particles",
            "balance_steps",
            "sinkhorn_steps",
            "cache_same_cloud_geometry",
            "theta",
            "graph",
        )
    }
    combined.update(
        {
            "estimator_seeds": seeds,
            "preparation_identity": {
                "schema": "bayesfilter.paired_batched_preparation_identity.v1",
                "batch_identities": [
                    result["preparation_identity"] for result in results
                ],
                "combined_seeds": seeds,
            },
            "per_seed_value": values,
            "per_seed_physical_score": scores,
            "aggregate_value": statistics.fmean(values),
            "aggregate_physical_score": [
                statistics.fmean(row[index] for row in scores)
                for index in range(parameter_count)
            ],
            "finite": all(bool(result["finite"]) for result in results),
            "replay_checked": all(
                bool(result["replay_checked"]) for result in results
            ),
            "bitwise_replay": all(
                bool(result["bitwise_replay"]) for result in results
            ),
            "chart_valid": all(bool(result["chart_valid"]) for result in results),
            "marginal_valid": all(
                bool(result["marginal_valid"]) for result in results
            ),
            "reset_valid": all(bool(result["reset_valid"]) for result in results),
            "work_valid": all(bool(result["work_valid"]) for result in results),
            "hard_valid": all(bool(result["hard_valid"]) for result in results),
            "maximum_tv_column_error": max(
                float(result["maximum_tv_column_error"]) for result in results
            ),
            "maximum_row_error": max(
                float(result["maximum_row_error"]) for result in results
            ),
            "gpu_allocator_bytes": {
                "current": max(
                    int(result["gpu_allocator_bytes"]["current"])
                    for result in results
                ),
                "peak": max(
                    int(result["gpu_allocator_bytes"]["peak"])
                    for result in results
                ),
            },
            "batch_count": len(results),
            "batches": [
                {
                    "estimator_seeds": result["estimator_seeds"],
                    "hard_valid": result["hard_valid"],
                    "finite": result["finite"],
                    "chart_valid": result["chart_valid"],
                    "marginal_valid": result["marginal_valid"],
                    "reset_valid": result["reset_valid"],
                    "work_valid": result["work_valid"],
                    "maximum_tv_column_error": result[
                        "maximum_tv_column_error"
                    ],
                    "maximum_row_error": result["maximum_row_error"],
                    "timing_seconds": result["timing_seconds"],
                    "gpu_allocator_bytes": result["gpu_allocator_bytes"],
                }
                for result in results
            ],
        }
    )
    return combined


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, required=True)
    parser.add_argument("--num-particles", type=int, choices=(1024, 4096), default=1024)
    parser.add_argument(
        "--seeds", default=",".join(str(seed) for seed in SEEDS)
    )
    parser.add_argument("--batch-size", type=int, default=len(SEEDS))
    parser.add_argument("--warm-repetitions", type=int, default=1)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)

    started = time.perf_counter()
    memory_policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.experimental.enable_tensor_float_32_execution(args.tf32)
    logical = tf.config.list_logical_devices("GPU")
    if len(logical) != 1:
        raise RuntimeError(f"expected one logical GPU, got {logical}")

    from docs.benchmarks import run_canonical_lgssm_fused_ot_loop_repair as runner
    from bayesfilter.highdim import ledh_contract_e_canonical_lgssm_tf as canonical
    from bayesfilter.highdim.transport_chunk_policy import select_transport_chunks

    seeds = _parse_seeds(args.seeds)
    if args.batch_size <= 0 or len(seeds) % args.batch_size:
        raise ValueError("batch-size must be positive and divide the seed count")
    if args.warm_repetitions <= 0:
        raise ValueError("warm-repetitions must be positive")
    plan = PLAN_N4096 if args.num_particles == 4096 else PLAN
    campaign = (
        f"{CAMPAIGN}-n4096" if args.num_particles == 4096 else CAMPAIGN
    )
    chunks = select_transport_chunks(args.num_particles)
    callable_ = canonical.make_canonical_prepared_value_and_score_tf(
        batch_size=args.batch_size,
        time_steps=2,
        num_particles=args.num_particles,
        steps=20,
        balance_steps=2,
        row_chunk_size=chunks.row_chunk_size,
        col_chunk_size=chunks.col_chunk_size,
        jit_compile=True,
        dtype=tf.float32,
    )
    device = {
        "physical_devices": [
            device.name for device in tf.config.list_physical_devices("GPU")
        ],
        "logical_devices": [device.name for device in logical],
        "memory_policy": memory_policy,
        "memory_growth": True,
        "jit_compile": True,
        "dtype": "float32",
        "tf32_enabled": args.tf32,
        "trust_basis": "owner_designated_managed_session_visible_gpu_trusted",
    }
    batch_results = []
    for start in range(0, len(seeds), args.batch_size):
        batch_seeds = seeds[start : start + args.batch_size]
        node_args = argparse.Namespace(
            time_steps=2,
            num_particles=args.num_particles,
            seeds=",".join(str(seed) for seed in batch_seeds),
            arm="all_active_contract_e",
            balance_steps=2,
            sinkhorn_steps=20,
            warm_repetitions=args.warm_repetitions,
            cache_same_cloud_geometry=False,
            dtype="float32",
            include_replay_diagnostic=True,
            include_kalman_diagnostic=False,
            campaign_id=campaign,
            plan_path=plan,
        )
        batch_results.append(
            runner._execute(
                node_args,
                configured_device=device,
                compiled_prepared_callable=callable_,
            )
        )
    result = _combine_batch_results(batch_results)
    payload = {
        "schema": SCHEMA,
        "campaign_id": campaign,
        "classification": "complete_canonical_lgssm_score_transfer_diagnostic_only",
        "moment_teacher_final_score_tested": False,
        "precision_arm": "tf32" if args.tf32 else "fp32_no_tf32_reference",
        "plan": plan,
        "command": " ".join(sys.argv),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "source_sha256": {
            path: _sha256(ROOT / path)
            for path in (
                "bayesfilter/highdim/ledh_contract_e_canonical_lgssm_tf.py",
                "bayesfilter/highdim/ledh_contract_e_lgssm_preparation_tf.py",
                "bayesfilter/highdim/ledh_contract_e_streaming_tf.py",
                "bayesfilter/highdim/ledh_contract_e_reset_tf.py",
                "docs/benchmarks/run_canonical_lgssm_fused_ot_loop_repair.py",
                "docs/benchmarks/run_zhao_cui_moment_teacher_score_mcse_transfer.py",
            )
        },
        "result": result,
        "wall_time_seconds": time.perf_counter() - started,
        "nonclaims": [
            "no Zhao-Cui moment-teacher final score was computed",
            "no default-readiness or HMC-readiness evidence",
            f"no inference beyond this T=2 N={args.num_particles} canonical LGSSM scope",
        ],
    }
    result_path = output / "result.json"
    _write_json(result_path, payload)
    manifest = {
        "schema": "bayesfilter.run_manifest.v1",
        "git_commit": payload["git_commit"],
        "command": payload["command"],
        "environment": {
            "python": sys.version,
            "tensorflow": tf.__version__,
            "conda_env": os.environ.get("CONDA_DEFAULT_ENV", "unset"),
        },
        "device": device,
        "data_version": result["preparation_identity"],
        "seeds": list(seeds),
        "wall_time_seconds": payload["wall_time_seconds"],
        "output_artifact": str(output.relative_to(ROOT)),
        "plan": plan,
        "result": str(result_path.relative_to(ROOT)),
        "result_sha256": _sha256(result_path),
    }
    _write_json(output / "run_manifest.json", manifest)
    print(
        json.dumps(
            {
                "precision_arm": payload["precision_arm"],
                "hard_valid": result["hard_valid"],
                "wall_time_seconds": payload["wall_time_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
