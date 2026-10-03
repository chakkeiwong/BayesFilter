"""Freeze four supplied-map development designs; check the written count table."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

PLAN = "docs/plans/bayesfilter-hmc-post-m27-next-phase-2026-09-23.md"
COUNTS = {
    "posterior_settings": {"warmup_min_results": 10000, "warmup_check_window_results": 10000,
        "warmup_chunk_results": 5000, "warmup_max_results": 30000,
        "retained_min_results": 30000, "retained_chunk_results": 5000, "retained_max_results": 60000},
    "fixed_comparator": {"warmup_results": 10000, "retained_results": 60000},
    "posterior_count_budget": {"max_results_per_chain": 60000},
}


def check_counts(designs, plan_text):
    """Compare resolved JSON with the actual Markdown table, including omissions."""
    expected = {group + "." + key for group, values in COUNTS.items() for key in values}
    rows = re.findall(r"^\| ([a-z_]+\.[a-z_]+) \| ([0-9]+) \|$", plan_text, re.MULTILINE)
    if len(rows) != len(expected) or {key for key, _ in rows} != expected:
        raise ValueError("M28 plan count table is missing, incomplete or duplicated")
    if not designs:
        raise ValueError("M28 requires resolved designs")
    for design in designs:
        for field, count in rows:
            group, key = field.split(".")
            if design["options"].get(group, {}).get(key) != int(count):
                raise ValueError(f"M28 count mismatch: {design['design_id']} {field}")
        if design["posterior_cap"] != int(dict(rows)["posterior_settings.retained_max_results"]):
            raise ValueError("M28 posterior_cap mismatch")
        if not design["options"]["posterior_count_budget"].get("count_budget_reason"):
            raise ValueError("M28 requires a count budget reason")
    return {"checked_designs": len(designs), "counts": dict(rows), "passed": True}


def build_suite():
    from bayesfilter.testing.inference_validation.designs import ScenarioSpec, ValidationDesign
    from bayesfilter.testing.inference_validation.funnel_maps import supplied_funnel_map
    designs = []
    for kind in ("exact", "residual"):
        spec = supplied_funnel_map(kind)
        for seed in (2026092381, 2026092382):
            design = ValidationDesign(design_id=f"m28-{kind}-{seed}", engine="stopping",
                scenario=ScenarioSpec(spec["target"], "fixed_transport", parameters=spec["parameters"]),
                replications=1, draws=500, seed=seed, budget_seconds=1000, device="gpu",
                purpose="supplied residual scale and model-coordinate posterior delivery development",
                numerical_provenance=PLAN, l_grid=(3, 5, 9, 13, 18, 25), step_size=.5,
                measurement_draws=128, posterior_cap=60000, mcse_tolerance=.05,
                options={"transport_payload": spec["transport_payload"],
                    "plan_file": PLAN, "member_rule": "shortest_verified_l",
                    "posterior_member_count": 2, "posterior_members": "selected",
                    "posterior_precision_method": "lugsail", "isolate_fits": True,
                    "fit_process_timeout_seconds": 980,
                    "search": {"pilot_enabled": True, "refinement_rounds": 1,
                        "max_candidates": 100, "total_budget_units": 300, "repair_reserve_units": 40},
                    **{key: dict(value) for key, value in COUNTS.items()},
                    "posterior_count_budget": {**COUNTS["posterior_count_budget"],
                        "count_budget_reason": "M28 supplied-map development allocation; no adequacy or default claim"}})
            designs.append(design.payload())
    return {"schema": "bayesfilter.inference_validation_suite.v1", "suite_id": "m28-supplied-maps-r1",
            "profile": "numerical_pipeline", "profiles": {"numerical_pipeline": ["stopping"]},
            "designs": designs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", type=Path, default=Path(PLAN))
    args = parser.parse_args()
    from bayesfilter.testing.inference_validation.execution import plan_suite
    from bayesfilter.testing.inference_validation.storage import write_json
    suite = build_suite()
    resolved = plan_suite(suite)
    checked = check_counts([job["design"] for job in resolved["jobs"]], args.plan.read_text())
    if any(job["availability"] != "ready" for job in resolved["jobs"]):
        raise ValueError("M28 target unavailable")
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "suite.json", suite)
    write_json(args.output / "resolved-plan.json", resolved)
    write_json(args.output / "count-check.json", checked)
    for design in suite["designs"]:
        cell = {**suite, "suite_id": design["design_id"], "designs": [design]}
        write_json(args.output / (design["design_id"] + ".json"), cell)
    print(json.dumps(checked))


if __name__ == "__main__":
    main()
