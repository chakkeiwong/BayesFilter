"""One bounded, isolated public-pipeline mechanics/activation fit for M32/M34."""
import argparse
import datetime
from pathlib import Path
import sys
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--design", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--strategy", choices=("static", "dynamic"), default="dynamic")
parser.add_argument("--seconds", type=float, required=True)
args = parser.parse_args()
from bayesfilter.testing.inference_validation.designs import ValidationDesign
from bayesfilter.testing.inference_validation.storage import read_json, write_json, file_hash
from bayesfilter.testing.inference_validation.execution import configure_worker, source_state

design = ValidationDesign.from_payload(read_json(args.design))
args.output.mkdir(parents=True, exist_ok=False)
manifest = {"command": sys.argv, "design": design.payload(), "design_identity": design.identity,
    "design_file_sha256": file_hash(args.design), "worker_sha256": file_hash(__file__),
    "seed": design.seed, "source": source_state(),
    "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "plan_file": "docs/plans/bayesfilter-hmc-repair-master-program-2026-09-16.md",
    "result_file": str(args.output / "result.json"), "runner_strategy": args.strategy,
    "role": "engineering_activation_only_not_power_coverage_or_performance"}
started = time.monotonic()
try:
    manifest["runtime"] = configure_worker(design)
    write_json(args.output / "manifest.json", manifest)
    from bayesfilter.testing.inference_validation.engines.pipeline import run_replication
    result = run_replication(design, args.output, 0, deadline=started+args.seconds,
                             reuse_leapfrog_graphs=args.strategy == "dynamic")
    if design.scenario.target == "normal_conjugate":
        from bayesfilter.testing.inference_validation.engines.normal_endpoint import endpoint_from_pipeline
        pipeline = read_json(args.output / "replication-0000/pipeline.json")
        if pipeline["verified_candidate_ids"]:
            result["endpoint"] = endpoint_from_pipeline(design.scenario.control, pipeline,
                tau=design.scenario.parameters["tau"], sigma=design.scenario.parameters["sigma"])
        else:
            result["endpoint"] = {"status": "unavailable", "reason": "no_verified_candidate"}
    write_json(args.output / "result.json", result)
finally:
    write_json(args.output / "exit.json", {"elapsed_seconds": time.monotonic()-started,
        "result_present": (args.output / "result.json").exists()})
