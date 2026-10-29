"""Run the canonical MP5 smoke check."""

from __future__ import annotations

import argparse
import sys
import time

from experiments.controlled_dpf_baseline.fixtures import make_fixture
from experiments.controlled_dpf_baseline.prototypes import run_clean_room_particle_flow
from experiments.controlled_dpf_baseline.results import (
    summarize_records,
    write_json,
    write_markdown_report,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num-particles", type=int, required=True)
    parser.add_argument("--flow-steps", type=int, required=True)
    parser.add_argument("--max-records", type=int, required=True)
    parser.add_argument("--max-wall-seconds", type=float, required=True)
    parser.add_argument("--records-json", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--report-md", required=True)
    args = parser.parse_args(argv)

    if args.max_records != 1:
        parser.error("MP5 smoke must use --max-records 1")
    start = time.perf_counter()
    fixture = make_fixture(args.fixture)
    record = run_clean_room_particle_flow(
        fixture,
        seed=args.seed,
        num_particles=args.num_particles,
        flow_steps=args.flow_steps,
        grid="smoke",
    )
    elapsed = time.perf_counter() - start
    records = [record]
    summary = summarize_records(
        records,
        runtime_warning_seconds=args.max_wall_seconds,
        expected_records=1,
    )
    summary["wall_seconds"] = elapsed
    write_json(args.records_json, {"records": records})
    write_json(args.summary_json, summary)
    decision = (
        "mp5_smoke_ok"
        if record["status"] == "ok" and elapsed <= args.max_wall_seconds
        else "mp5_smoke_structured_blocker"
    )
    write_markdown_report(
        args.report_md,
        title="Controlled DPF Baseline Smoke Result",
        decision=decision,
        summary=summary,
        records=records,
        notes=[
            "MP5 smoke only; this is not the MP6 fixed grid.",
            "Student implementation source was not imported or executed.",
        ],
    )
    return 0 if record["status"] == "ok" and elapsed <= args.max_wall_seconds else 1


if __name__ == "__main__":
    sys.exit(main())
