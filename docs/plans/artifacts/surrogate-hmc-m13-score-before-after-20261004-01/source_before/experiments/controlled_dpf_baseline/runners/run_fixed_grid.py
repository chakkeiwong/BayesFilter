"""Run the canonical MP6 fixed first-target grid."""

from __future__ import annotations

import argparse
import sys

from experiments.controlled_dpf_baseline.fixtures import make_fixture
from experiments.controlled_dpf_baseline.prototypes import run_clean_room_particle_flow
from experiments.controlled_dpf_baseline.results import (
    summarize_records,
    write_json,
    write_markdown_report,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", required=True)
    parser.add_argument("--fixtures", required=True)
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--num-particles", type=int, required=True)
    parser.add_argument("--low-noise-flow-steps", type=int, required=True)
    parser.add_argument("--moderate-flow-steps", required=True)
    parser.add_argument("--max-records", type=int, required=True)
    parser.add_argument("--per-record-warning-seconds", type=float, required=True)
    parser.add_argument("--records-json", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--report-md", required=True)
    args = parser.parse_args(argv)

    if args.grid != "first-target":
        parser.error("MP6 supports only --grid first-target")
    fixture_names = [item.strip() for item in args.fixtures.split(",") if item.strip()]
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    moderate_steps = [
        int(item.strip()) for item in args.moderate_flow_steps.split(",") if item.strip()
    ]
    planned = _build_first_target_plan(
        fixture_names=fixture_names,
        seeds=seeds,
        num_particles=args.num_particles,
        low_noise_flow_steps=args.low_noise_flow_steps,
        moderate_flow_steps=moderate_steps,
    )
    if len(planned) != args.max_records or args.max_records != 15:
        parser.error(
            f"MP6 first-target grid must produce exactly 15 records, got {len(planned)}"
        )
    records = []
    for fixture_name, seed, num_particles, flow_steps in planned:
        records.append(
            run_clean_room_particle_flow(
                make_fixture(fixture_name),
                seed=seed,
                num_particles=num_particles,
                flow_steps=flow_steps,
                grid="first-target",
            )
        )
    summary = summarize_records(
        records,
        runtime_warning_seconds=args.per_record_warning_seconds,
        expected_records=args.max_records,
    )
    write_json(args.records_json, {"records": records})
    write_json(args.summary_json, summary)
    decision = (
        "mp6_fixed_grid_ok"
        if summary["planned_records"] == 15 and summary["failed_records"] == 0
        else "mp6_fixed_grid_structured_blocker"
    )
    write_markdown_report(
        args.report_md,
        title="Controlled DPF Baseline Fixed-Grid Result",
        decision=decision,
        summary=summary,
        records=records,
        notes=[
            "MP6 fixed first-target grid only.",
            "Metrics are proxy diagnostics, not correctness certificates.",
        ],
    )
    return 0 if summary["planned_records"] == 15 and summary["failed_records"] == 0 else 1


def _build_first_target_plan(
    *,
    fixture_names: list[str],
    seeds: list[int],
    num_particles: int,
    low_noise_flow_steps: int,
    moderate_flow_steps: list[int],
) -> list[tuple[str, int, int, int]]:
    expected_fixtures = {
        "range_bearing_gaussian_low_noise",
        "range_bearing_gaussian_moderate",
    }
    if set(fixture_names) != expected_fixtures:
        raise ValueError(f"fixtures must be exactly {sorted(expected_fixtures)}")
    if seeds != [31, 43, 59, 71, 83]:
        raise ValueError("seeds must be exactly 31,43,59,71,83")
    if num_particles != 128:
        raise ValueError("num_particles must be exactly 128")
    if low_noise_flow_steps != 20:
        raise ValueError("low-noise flow steps must be exactly 20")
    if sorted(moderate_flow_steps) != [10, 20]:
        raise ValueError("moderate flow steps must be exactly 10,20")
    plan = [
        ("range_bearing_gaussian_low_noise", seed, num_particles, 20)
        for seed in seeds
    ]
    plan.extend(
        ("range_bearing_gaussian_moderate", seed, num_particles, flow_steps)
        for flow_steps in (10, 20)
        for seed in seeds
    )
    return plan


if __name__ == "__main__":
    sys.exit(main())
