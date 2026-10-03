"""One isolated M30 full public-pipeline diagnostic with an execution-only option."""
from __future__ import annotations
import argparse
import datetime as dt
import json
from pathlib import Path
import sys
import time
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--strategy", choices=("static", "dynamic"), required=True)
    parser.add_argument("--seconds", type=float, required=True)
    args = parser.parse_args()
    from bayesfilter.testing.inference_validation.designs import resolve_suite
    from bayesfilter.testing.inference_validation.execution import configure_worker, source_state
    from bayesfilter.testing.inference_validation.storage import read_json, write_json, file_hash
    design, = resolve_suite(read_json(args.design))
    args.output.mkdir(parents=True, exist_ok=False)
    prefix = args.output / "full-fit"
    manifest = {"command": sys.argv, "python": sys.executable, "design": design.payload(),
        "design_identity": design.identity, "design_file_sha256": file_hash(args.design),
        "runner_strategy": args.strategy, "source": source_state(), "seed": design.seed,
        "data_version": "M29 unchanged declared target, prior and fixed data",
        "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "plan_file": "docs/plans/bayesfilter-hmc-post-m29-next-phase-2026-09-23.md",
        "result_file": "docs/plans/bayesfilter-hmc-m30-runner-reuse-result-2026-09-23.md",
        "budget_seconds": args.seconds, "execution_role": "same-source paired full-fit diagnostic"}
    started = time.monotonic()
    status = "failed"
    error = None
    try:
        manifest["runtime"] = configure_worker(design)
        write_json(str(prefix) + "-manifest.json", manifest)
        from bayesfilter.testing.inference_validation.fit_process import resource_snapshot
        from bayesfilter.testing.inference_validation.engines.pipeline import run_replication
        before = resource_snapshot()
        run_replication(design, args.output, 0, deadline=started + args.seconds,
                        reuse_leapfrog_graphs=args.strategy == "dynamic")
        write_json(str(prefix) + "-resources.json", {"before": before, "after": resource_snapshot()})
        status = "complete"
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        raise
    finally:
        receipt = {"status": status, "elapsed_seconds": time.monotonic() - started,
                   "error": error, "manifest_sha256": file_hash(str(prefix) + "-manifest.json")
                   if Path(str(prefix) + "-manifest.json").exists() else None}
        write_json(str(prefix) + "-exit.json", receipt)
        print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
