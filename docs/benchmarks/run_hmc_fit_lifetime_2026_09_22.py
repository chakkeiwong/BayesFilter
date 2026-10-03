"""Bounded CPU reference study of persistent versus isolated complete fits."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("persistent", "isolated"), required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise RuntimeError("this study explicitly hides GPU before framework import")
    from bayesfilter.testing.inference_validation.designs import ValidationDesign
    from bayesfilter.testing.inference_validation.storage import read_json, write_json
    from bayesfilter.testing.inference_validation.execution import configure_worker, source_state
    from bayesfilter.testing.inference_validation.fit_process import resource_snapshot, run_isolated_replications

    design = ValidationDesign.from_payload(read_json(args.design))
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    manifest = {"source": source_state(), "command": [sys.executable, *sys.argv],
                "design": design.payload(), "mode": args.mode, "gpu_intentionally_hidden": True,
                "plan_file": design.numerical_provenance, "environment": sys.executable,
                "result_file": str(args.output / "result.json"),
                "data_version": "fixed declared analytic law/data; development only",
                "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    write_json(args.output / "manifest.json", manifest)
    if args.mode == "isolated":
        result = run_isolated_replications(design, args.output, deadline=started+design.budget_seconds)
        manifest["parent_resources"] = resource_snapshot()
    else:
        manifest["runtime"] = configure_worker(design)
        from bayesfilter.testing.inference_validation.engines.pipeline import run_replication, summarize_replications
        resources, records = [], []
        for replication in range(design.replications):
            fit_started = time.monotonic()
            resources.append({"replication": replication, "stage": "before", **resource_snapshot()})
            records.append(run_replication(design, args.output, replication,
                min(started+design.budget_seconds, fit_started+design.options["fit_process_timeout_seconds"])))
            resources.append({"replication": replication, "stage": "after", **resource_snapshot(),
                              "fit_seconds": time.monotonic()-fit_started})
            write_json(args.output / "resources.json", resources)
            if time.monotonic() >= started+design.budget_seconds:
                break
        result = summarize_replications(design, records)
        write_json(args.output / "assessment.json", result)
    manifest["elapsed_before_shutdown_seconds"] = time.monotonic()-started
    write_json(args.output / "manifest.json", manifest)
    write_json(args.output / "result.json", result)
    print(json.dumps({"completed": result["completed"], "planned": result["planned"],
                      "execution_failures": result["execution_failures"], "mode": args.mode,
                      "elapsed_before_shutdown_seconds": manifest["elapsed_before_shutdown_seconds"]}), flush=True)
    return int(result["completed"] != result["planned"] or bool(result["execution_failures"]))


if __name__ == "__main__":
    raise SystemExit(main())
