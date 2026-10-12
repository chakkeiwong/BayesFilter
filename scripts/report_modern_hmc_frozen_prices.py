#!/usr/bin/env python3
"""Read-only cost report for frozen cohorts; no tuning or posterior authority."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.audit_mams_full_model_pricing import (
    check_hashes, check_tensor_tree, decode_metadata, digest, payload_digest, read,
)
from scripts.price_modern_hmc_frozen_design import cost_envelope


def check_design_reuse(priced, current):
    """Allow only the already documented, nonexecuted fixed-HMC repair fix."""
    expected = deepcopy(priced)
    correction = expected["method"] == "fixed_hmc" and expected != current
    if correction:
        search = expected["search"]
        if (search.get("schema") != "bayesfilter.hmc_controller_config.v3"
                or search.get("repair_trajectory_policy") != "preserve_mean_duration"
                or search.get("repair_mean_duration_max_steps") != 128):
            raise ValueError("unrecognized original fixed-HMC repair configuration")
        search.pop("repair_trajectory_policy")
        search.pop("repair_mean_duration_max_steps")
        search["schema"] = "bayesfilter.hmc_controller_config.v2"
    if expected != current:
        raise ValueError("price reuse changes more than the documented repair encoding")
    return "fixed_count_repair_correction" if correction else "identical_design"


def audit_call(directory, cell, root, call, *, dimension):
    key = f"root-{root['index']:03d}-repeat-{call['repeat']}"
    path = directory / "committed" / key / "bundle.json"
    if digest(path) != path.with_name("bundle.sha256").read_text().strip():
        raise ValueError("price checkpoint checksum mismatch")
    bundle = read(path)
    identity = read(directory / "identity.json")
    inputs = read(Path(bundle["metadata"]["attempt"]) / "call-start.json")["inputs"]
    candidate = inputs["candidate"]
    if (bundle["schema"] != "bayesfilter.durable_tensor_checkpoint.v1"
            or identity != dict(binding_hash=cell["binding_hash"], role="cost_only")
            or bundle["identity_hash"] != payload_digest(identity)
            or bundle["inputs_hash"] != payload_digest(inputs)
            or inputs["start_sha256"] != cell["initial_active_state_sha256"]
            or inputs["seed"] != call["seed"]
            or inputs["transitions"] != call["transitions"]
            or candidate["candidate_id"] != root["candidate_id"]
            or candidate["transition_spec"] != root["transition_spec"]
            or candidate["epsilon"] != root["epsilon"]
            or candidate["leapfrog_steps"] != root["leapfrog_steps"]):
        raise ValueError("price checkpoint differs from declared inputs")
    mapping = bundle["tree"]["mapping"]
    metadata = decode_metadata(mapping["metadata"])
    work = metadata["value_score_row_evaluations"] + metadata["target_status_row_evaluations"]
    if (metadata != call["native"] or metadata["use_xla"] is not True
            or "GPU" not in metadata["samples_device"]
            or metadata["transition_spec"] != root["transition_spec"]
            or not 0 < work <= metadata["maximum_reserved_gradient_work"]):
        raise ValueError("native price metadata or work disagrees with its receipt")
    shape = [call["transitions"], 4]
    if (mapping["samples"]["tensor"]["shape"] != [*shape, dimension]
            or mapping["samples"]["tensor"]["dtype"] != "float64"):
        raise ValueError("price samples have the wrong shape or dtype")
    trace = mapping["trace"]["mapping"]
    for name in ("trajectory_steps", "value_score_calls_per_chain",
                 "target_status_calls_per_chain", "log_accept_ratio"):
        if trace[name]["tensor"]["shape"] != shape:
            raise ValueError("missing full native trace: " + name)
    for name in ("status_code", "valid_pre_regularized_score", "floor_count_value"):
        if trace["target_status_telemetry"]["mapping"][name]["tensor"]["shape"] != shape:
            raise ValueError("missing full target-status trace: " + name)
    return dict(bundle_sha256=digest(path), actual_work=work,
                tensor_receipts_checked=check_tensor_tree(bundle["tree"], path.parent))


def build_report(prices_path, design_paths, *, available_seconds,
                 trial_overhead_seconds, fixed_overhead_seconds):
    if not math.isfinite(available_seconds) or available_seconds <= 0:
        raise ValueError("positive remaining allocation required")
    prices_path = Path(prices_path).resolve()
    prices = read(prices_path)
    if prices["status"] != "completed":
        raise ValueError("complete prices required; preserve and report missing cells separately")
    memory = prices["memory_policy"]
    if (prices["gpu_trust_basis"] != "trusted_escalated_gpu_execution"
            or prices["memory_growth_environment"].lower() != "true"
            or memory["all_physical_devices_memory_growth"] is not True
            or memory["configured_before_logical_device_initialization"] is not True):
        raise ValueError("price GPU/memory policy is not eligible")
    source = Path(prices["source_root"])
    if read(source / ".validation-source.json") != prices["numerical_source"]:
        raise ValueError("numerical source provenance changed")
    check_hashes({str(source / name): value for name, value in prices["compiled_operator_sha256"].items()})
    check_hashes({prices["command"][1]: prices["driver_sha256"]})
    if len(prices["rows"]) != len(design_paths):
        raise ValueError("report must preserve all priced methods")
    report = dict(prices_file=str(prices_path), prices_sha256=digest(prices_path),
        source=prices["numerical_source"], available_seconds=available_seconds,
        actual_price_worker_seconds=prices["supervisor_elapsed_seconds"],
        load_factors=[1., 2., 4.], rows=[], all_method_scenarios=[],
        qualification_authority=False, posterior_authority=False, method_ranking=None,
        probabilistic_upper_bound=False, launch_authorized_by_report=False)
    seeds, starts, cells = set(), set(), set()
    for index, (cell, design_path) in enumerate(zip(prices["rows"], design_paths)):
        design_path = Path(design_path).resolve()
        current, original = read(design_path), read(Path(cell["design_file"]))
        if digest(Path(cell["design_file"])) != cell["design_sha256"]:
            raise ValueError("priced design changed")
        reuse = check_design_reuse(original, current)
        key = (cell["target"], cell["method"], cell["seed"])
        if key in cells or key != (current["target"], current["method"], current["seed"]):
            raise ValueError("duplicate or mismatched method inventory")
        cells.add(key)
        if cell["status"] != "completed":
            raise ValueError("incomplete method price")
        source_count = check_hashes(cell["source_sha256"])
        geometry_dir = Path(current["geometry_directory"])
        if (digest(geometry_dir / "geometry.json") != cell["geometry_sha256"]
                or read(geometry_dir / "design.json")["metadata"]["data_sha256"] != cell["data_sha256"]):
            raise ValueError("data or geometry changed")
        starts.add(cell["initial_active_state_sha256"])
        dimension = len(read(geometry_dir / "geometry.json")["center"])
        calls = []
        for root_index, root in enumerate(cell["roots"]):
            if (root["index"] != root_index or root["status"] not in {"priced", "priced_with_veto"}
                    or [c["repeat"] for c in root["calls"]] != [0, 1]):
                raise ValueError("incomplete or reordered root prices")
            if (root["transition_spec"]["profile"] != current["execution"]["transition_profile"]
                    or root["transition_spec"]["max_steps"] != root["leapfrog_steps"]
                    or root["transition_spec"]["epsilon"] != root["epsilon"]):
                raise ValueError("root transition law differs from design")
            for call in root["calls"]:
                if tuple(call["seed"]) in seeds:
                    raise ValueError("price stream reused")
                seeds.add(tuple(call["seed"]))
                audit = audit_call(prices_path.parent / f"cell-{index:02d}", cell, root, call,
                                   dimension=dimension)
                calls.append(dict(root_index=root_index, repeat=call["repeat"],
                    seconds=call["elapsed_seconds"], native_seconds=call["native"]["sample_chain_call_s"],
                    hard_vetoes=call["hard_vetoes"], **audit))
        report["rows"].append(dict(target=cell["target"], method=cell["method"],
            root_count=len(cell["roots"]), design_file=str(design_path), design_sha256=digest(design_path),
            reuse=reuse, source_hashes_checked=source_count, calls=calls,
            preparation_seconds=cell["preparation_seconds"],
            scenarios=[cost_envelope(current, cell, load_multiplier=load,
                overhead_seconds=fixed_overhead_seconds + cell["preparation_seconds"],
                trial_overhead_seconds=trial_overhead_seconds) for load in report["load_factors"]]))
    if len(starts) != 1:
        raise ValueError("comparison prices used different initial states")
    for index, load in enumerate(report["load_factors"]):
        total = sum(row["scenarios"][index]["maximum_count_scenario_seconds"] for row in report["rows"])
        report["all_method_scenarios"].append(dict(load_multiplier=load, seconds=total,
            fits_remaining_allocation=total <= available_seconds))
    report.update(status="audited_cost_diagnostic", distinct_price_streams=len(seeds),
                  initial_active_state_sha256=next(iter(starts)))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prices", type=Path, required=True)
    parser.add_argument("--design", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--available-seconds", type=float, required=True)
    parser.add_argument("--trial-overhead-seconds", type=float, required=True)
    parser.add_argument("--fixed-overhead-seconds", type=float, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    report = build_report(args.prices, args.design, available_seconds=args.available_seconds,
        trial_overhead_seconds=args.trial_overhead_seconds, fixed_overhead_seconds=args.fixed_overhead_seconds)
    report.update(command=[sys.executable, *sys.argv], driver_sha256=digest(Path(__file__)),
                  elapsed_seconds=time.monotonic()-started, created_utc=datetime.now(timezone.utc).isoformat())
    with args.output.open("x") as output:
        output.write(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
