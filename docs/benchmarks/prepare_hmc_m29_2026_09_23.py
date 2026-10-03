"""Freeze unchanged M27 numerical settings for M29 off/on profiling pairs."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import re

PLAN = "docs/plans/bayesfilter-hmc-post-m28-next-phase-2026-09-23.md"
NOTE = "docs/plans/bayesfilter-hmc-m29-execution-note-2026-09-23.md"
BASELINES = {"gaussian": "m27-gpu-gaussian-lugsail-2026092283.json",
             "beta_binomial": "m27-gpu-beta-binomial-lugsail-2026092284.json"}
SEEDS = {"gaussian": 2026092391, "beta_binomial": 2026092392}


def baselines(root):
    return {key: json.loads((Path(root) / name).read_text()) for key, name in BASELINES.items()}


def build_suite(baseline):
    designs = []
    for target, original in baseline.items():
        design = copy.deepcopy(original)
        design.update(design_id=f"m29-gpu-{target.replace('_', '-')}-{SEEDS[target]}",
                      seed=SEEDS[target], budget_seconds=1000, numerical_provenance=PLAN,
                      purpose="paired child profiling; no statistical or default promotion")
        design["options"].update(plan_file=PLAN, fit_process_timeout_seconds=980)
        designs.append(design)
    return {"schema": "bayesfilter.inference_validation_suite.v1", "suite_id": "m29-profile-pairs",
            "profile": "numerical", "profiles": {"numerical": ["stopping"]}, "designs": designs}


def check_designs(suite, baseline, note):
    allowed = {"design_id", "seed", "budget_seconds", "numerical_provenance", "purpose",
               "options.plan_file", "options.fit_process_timeout_seconds"}
    def leaves(value, prefix=""):
        result = {}
        for key, item in value.items():
            path = prefix + key
            if isinstance(item, dict):
                result.update(leaves(item, path + "."))
            else:
                result[path] = item
        return result
    table = {name: (int(a), int(b)) for name, a, b in re.findall(
        r"^\| ([a-z_]+) \| (\d+) \| (\d+) \|$", note, re.MULTILINE)}
    checked = []
    for design in suite["designs"]:
        target = design["scenario"]["target"]
        old, new = leaves(baseline[target]), leaves(design)
        changes = {k: {"before": old.get(k), "after": new.get(k)}
                   for k in old.keys() | new.keys() if old.get(k) != new.get(k)}
        if set(changes) != allowed:
            raise ValueError(f"unexpected baseline field changes: {sorted(changes)}")
        if design["seed"] != SEEDS[target] or design["budget_seconds"] != 1000:
            raise ValueError("M29 seed/budget does not match the plan")
        if design["options"]["fit_process_timeout_seconds"] != 980:
            raise ValueError("M29 child timeout does not match the plan")
        counts = {k: design[k] for k in ("draws", "measurement_draws", "posterior_cap")}
        counts.update(design["options"]["posterior_settings"])
        counts.update({"fixed_" + k: v for k, v in design["options"]["fixed_comparator"].items()})
        column = 0 if target == "gaussian" else 1
        if set(table) != set(counts) or any(table[k][column] != v for k, v in counts.items()):
            raise ValueError("resolved counts disagree with the written M29 count table")
        checked.append({"target": target, "counts": counts, "baseline_changes": changes})
    if {row["target"] for row in checked} != set(BASELINES) or len(checked) != 2:
        raise ValueError("M29 needs exactly the two declared target designs")
    return {"passed": True, "designs": checked,
            "pairing": "identical scientific design; only execution --profile-execution differs"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--note", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from bayesfilter.testing.inference_validation.execution import plan_suite
    from bayesfilter.testing.inference_validation.storage import write_json
    baseline = baselines(args.baseline_root)
    suite = build_suite(baseline)
    checked = check_designs(suite, baseline, args.note.read_text())
    resolved = plan_suite(suite)
    assert all(job["availability"] == "ready" for job in resolved["jobs"])
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "suite.json", suite)
    write_json(args.output / "resolved-plan.json", resolved)
    write_json(args.output / "count-and-baseline-check.json", checked)
    for design in suite["designs"]:
        write_json(args.output / (design["design_id"] + ".json"),
                   {**suite, "suite_id": design["design_id"], "designs": [design]})
    print(json.dumps({"passed": True, "designs": [d["design_id"] for d in suite["designs"]]}))


if __name__ == "__main__":
    main()
