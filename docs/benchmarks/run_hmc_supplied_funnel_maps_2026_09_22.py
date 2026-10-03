"""Bounded supplied-map funnel integration under the September 22 HMC plan.

This is a diagnostic development fit, not learned transport training, a
statistical comparison, or a new default. Run one explicit map and seed into a
fresh output directory; an external launcher bounds total worker time.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

PLAN = "docs/plans/bayesfilter-hmc-supplied-whitening-plan-2026-09-22.md"


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu_reference", "gpu"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--map", choices=("exact", "partial", "partial_half"), required=True)
    parser.add_argument("--seconds", type=float, required=True)
    parser.add_argument("--search", choices=("native", "intermediate_grid"), default="native")
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error("seconds must be positive")
    if args.search == "intermediate_grid" and args.map == "exact":
        parser.error("the follow-up grid is declared only for partial maps")
    start = time.monotonic()
    gpu = args.device == "gpu"
    if gpu:
        if os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH", "").lower() != "true":
            raise RuntimeError("GPU requires memory growth before framework import")
        if os.environ.get("CUDA_VISIBLE_DEVICES") in (None, "", "-1"):
            raise RuntimeError("GPU requires an explicit trusted visible device")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    for key in ("TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = "1"
    args.output.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo))
    manifest = {"command": [sys.executable, *sys.argv], "plan_file": PLAN,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "environment": sys.executable, "device": args.device, "gpu_intentionally_hidden": not gpu,
        "jit_compile": gpu, "cpu_threads": 1, "seed": args.seed,
        "data_version": "analytic scale-3 funnel, two children", "map": args.map,
        "search": args.search,
        "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source_hashes": {str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted((repo / "bayesfilter").rglob("*.py"))},
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "role": "supplied-map development integration; no training, ranking, or coverage claim"}
    write(args.output / "manifest.json", manifest)
    try:
        import tensorflow as tf
        from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
        from bayesfilter.testing.inference_validation.funnel_maps import supplied_funnel_map
        from bayesfilter.testing.inference_validation.designs import ScenarioSpec, ValidationDesign
        from bayesfilter.testing.inference_validation.procedures import execute_pipeline
        from bayesfilter.testing.inference_validation.engines.pipeline import check_inventory
        from bayesfilter.testing.inference_validation.storage import read_json

        manifest["memory_policy"] = configure_tensorflow_gpu_memory_growth(tf, require_gpu=gpu)
        write(args.output / "manifest.json", manifest)
        spec = supplied_funnel_map(args.map)
        write(args.output / "supplied_map.json", spec)
        extra_options = {}
        suffix = ""
        if args.search == "intermediate_grid":
            epsilons = ((.5, .625, .75, .875, 1., 1.125, 1.25) if args.map == "partial"
                        else (.25, .375, .5, .625, .75))
            extra_options["search"] = {"epsilon_by_l": tuple((l, epsilons) for l in (3, 5, 9, 13, 18, 25)),
                "pilot_enabled": False, "refinement_rounds": 2, "explore_failed_intervals": True}
            suffix = "-intermediate"
        design = ValidationDesign(design_id=f"supplied-funnel-{args.map}-{args.seed}{suffix}",
            engine="accuracy", scenario=ScenarioSpec(spec["target"], "fixed_transport", parameters=spec["parameters"]),
            replications=1, draws=500, seed=args.seed, budget_seconds=args.seconds,
            device=args.device, purpose="tuning given a supplied frozen map of the same funnel law",
            numerical_provenance=PLAN, step_size=.5, measurement_draws=128,
            posterior_cap=10000, mcse_tolerance=.05,
            options={**extra_options, "transport_payload": spec["transport_payload"], "posterior_members": "selected",
                "member_rule": "first_verified", "posterior_settings": {
                    "warmup_chunk_results": 500, "warmup_min_results": 2000,
                    "warmup_check_window_results": 1000, "warmup_max_results": 10000,
                    "retained_chunk_results": 500, "retained_min_results": 1000,
                    "retained_max_results": 10000}})
        write(args.output / "design.json", design.payload())
        pipeline = execute_pipeline(design, args.output / "fit", deadline=start + args.seconds)
        inventory = check_inventory(read_json(pipeline["tuning_path"]))
        if inventory["failures"]:
            raise RuntimeError("invalid candidate inventory: " + str(inventory["failures"]))
        assessed = [row for row in pipeline["members"] if row["status"] == "assessed"]
        rows = [{"candidate_id": row["candidate_id"], "L": row["L"], "epsilon": row["epsilon"],
                 "passed": row["posterior"]["passed"], "hard_vetoes": row["posterior"]["hard_vetoes"],
                 "warmup_results_per_chain": row["posterior"]["warmup_results_per_chain"],
                 "retained_results_per_chain": row["posterior"]["retained_results_per_chain"],
                 "warmup_cap_hit": row["posterior"]["warmup_cap_hit"],
                 "retained_cap_hit": row["posterior"]["retained_cap_hit"],
                 "warmup_exclusion_matches": row["warmup_exclusion_matches"]} for row in assessed]
        if any(not row["warmup_exclusion_matches"] for row in rows):
            raise RuntimeError("warmup exclusion failed")
        result = {"status": "complete", "map": args.map, "seed": args.seed, "search": args.search,
            "inventory": inventory, "tuning_completion": pipeline["completion"],
            "verified_candidate_ids": pipeline["verified_candidate_ids"],
            "tuning_usability": "verified_members" if pipeline["verified_candidate_ids"] else "no_verified_member",
            "posterior": rows, "unassessed_members": len(pipeline["members"]) - len(assessed),
            "ranking_supported": False, "scientific_gaps_closed": False,
            "elapsed_seconds": time.monotonic() - start}
        write(args.output / "result.json", result)
        print(json.dumps(result), flush=True)
    except Exception as exc:
        write(args.output / "failure.json", {"exception": type(exc).__name__, "message": str(exc),
            "traceback": traceback.format_exc(), "elapsed_seconds": time.monotonic() - start})
        raise


if __name__ == "__main__":
    main()
