"""Offline M15 aggregation; fresh datasets and pilot evidence remain separate."""
from collections import Counter
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "m14-r1/source-schedule-r1"
sys.path.insert(0, str(SOURCE))


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stopping_groups(directory):
    from bayesfilter.testing.inference_validation.designs import ValidationDesign, seed_for
    from bayesfilter.testing.inference_validation.engines.pipeline import summarize_replications

    index_path = directory / "run_index.json"
    index = read(index_path)
    groups, inputs = {}, {str(index_path): sha(index_path)}
    for job in index["plan"]["jobs"]:
        design = ValidationDesign.from_payload(job["design"])
        if design.engine == "stopping":
            key = design.scenario.target + "/" + design.scenario.start
            groups.setdefault(key, []).append(design)
    summaries = {}
    for key, designs in groups.items():
        records, rows, streams = [], [], set()
        comparable = None
        for design in designs:
            payload = design.payload()
            payload.pop("design_id")
            payload.pop("seed")
            payload["options"].pop("data")
            payload["options"].pop("data_provenance")
            if comparable is None:
                comparable = payload
            if payload != comparable or design.replications != 1:
                raise ValueError("incompatible stopping experiments")
            # Raw root seeds are intentionally shared. Domain-derived data and
            # fit streams, not the root integer, identify independent units.
            data_seed = tuple(design.options["data_provenance"]["seed"])
            fit_seed = tuple(seed_for(design.seed, design.design_id, 0, 0, "tuning"))
            if data_seed in streams or fit_seed in streams or data_seed == fit_seed:
                raise ValueError("duplicate data/fit stream")
            streams.update((data_seed, fit_seed))
            job = index["jobs"].get(design.design_id, {"status": "not_run"})
            row = {"design_id": design.design_id, "status": job["status"],
                   "data": design.options["data"], "data_seed": data_seed,
                   "tuning_seed": fit_seed}
            if job["status"] == "complete":
                path = Path(job["result"])
                if sha(path) != job["result_sha256"]:
                    raise ValueError("changed result: " + str(path))
                result = read(path)
                if result["design_identity"] != design.identity:
                    raise ValueError("result/design mismatch")
                inputs[str(path)] = sha(path)
                assessment = result["assessment"]
                for record in assessment["replications"]:
                    if record["inventory"]["failures"]:
                        raise ValueError("invalid tuning inventory")
                    records.append(dict(record, replication=len(records),
                                        design_id=design.design_id))
                row["finding"] = assessment["finding"]
            rows.append(row)
        # Each record already contains its own data-dependent exact truths.
        # This aggregator uses the first data only to enumerate quantity names.
        aggregate = summarize_replications(replace(designs[0], replications=len(designs)), records)
        members = [m for r in records for m in r["members"] if "assessment" in m]
        summaries[key] = {"aggregation": aggregate, "fits": rows,
            "runtime_passed": sum(bool(m["runtime_checks_passed"]) for m in members),
            "warmup_counts": dict(Counter(m["warmup_count"] for m in members)),
            "retained_counts": dict(Counter(m["retained_count"] for m in members)),
            "warmup_caps": sum(m["stopped_intervals"]["warmup_cap_hit"] for m in members),
            "retained_caps": sum(m["stopped_intervals"]["retained_cap_hit"] for m in members),
            "source_identity": index["source"]["identity"],
            "independent_unit": "new observed dataset and complete fit within this model/start group",
            "interpretation": "pointwise exploratory coverage and paired errors; no method ranking"}
    return {"groups": summaries, "input_sha256": inputs}


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
    from bayesfilter.testing.inference_validation.aggregation import aggregate_groups

    output = ROOT / "terminal-summary.json"
    if output.exists():
        raise FileExistsError(output)
    names = ("sbc-pilot-cpu-r1", "fresh-cpu-continuation-r3", "stopping-pilot-gpu-r1", "sbc-fresh-gpu-r2")
    for name in names:
        index = read(ROOT / name / "run_index.json")
        if any(j["status"] not in {"complete", "failed", "timed_out", "unfunded"}
               for j in index["jobs"].values()):
            raise ValueError("workers remain active: " + name)
    result = {"schema": "bayesfilter.hmc_m15_summary.v1", "sbc": {}, "stopping": {},
        "plan_file": "docs/plans/bayesfilter-hmc-repair-m15-design-2026-09-21.md",
        "command": sys.argv, "script_sha256": sha(Path(__file__)),
        "statistically_supported_ranking": False, "default_promoted": False}
    for name in names:
        directory = ROOT / name
        result["sbc"][name] = aggregate_groups(directory)
        result["stopping"][name] = stopping_groups(directory)
    with output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    for name, groups in result["sbc"].items():
        for row in groups:
            print(name, row["group_id"], row["completed"], "/", row["planned"], row["finding"])
    for name, group in result["stopping"].items():
        for key, row in group["groups"].items():
            print(name, key, "retained", row["retained_counts"], "caps", row["retained_caps"])


if __name__ == "__main__":
    main()
