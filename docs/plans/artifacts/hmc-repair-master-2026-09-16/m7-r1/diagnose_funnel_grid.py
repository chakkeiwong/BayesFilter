"""Bounded GPU diagnosis of the saved funnel geometry using the public tuner.

No historical receipt grants qualification. The numerical package is the same
immutable M7 snapshot; only the supplied proposal grid and search identity differ.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
SNAPSHOT = ROOT/"source-r1"
SAVED = ROOT/"pilots-gpu-r1/m7-pilot-funnel/replication-0000/tuning"
PLAN = "docs/plans/bayesfilter-hmc-repair-master-program-2026-09-16.md"
# The first attempt failed configuration before transitions; keep its charge.
BUDGET = 600. - json.loads((ROOT/"funnel-grid-r1/gpu-diagnostic-run.json").read_text())["elapsed_seconds"]


def read(path):
    return json.loads(path.read_text())


def worker(output):
    # Configure device allocation before any TensorFlow import.
    sys.path.insert(0, str(SNAPSHOT))
    from bayesfilter.testing.inference_validation.execution import configure_worker, source_state
    from bayesfilter.testing.inference_validation.storage import write_json, file_hash
    runtime = configure_worker(SimpleNamespace(device="gpu"))
    write_json(output/"runtime.json", runtime)
    source = source_state()
    assert source["identity"] == read(SNAPSHOT/"source_snapshot.json")["source_identity"]

    from dataclasses import replace
    from bayesfilter.inference import HMCControllerConfig, HMCCandidateExecutionConfig, tune_hmc_kernel
    from bayesfilter.inference.hmc_candidate_set_execution import _issue_binding, _tensor_from_payload
    from bayesfilter.inference.hmc_candidate_set_tuning import _sha256
    from bayesfilter.inference.hmc_candidate_set_artifacts import load_candidate_set_result_payload
    from bayesfilter.testing.inference_validation.targets import ValidationTarget

    wrapper = read(SAVED/"execution_spec.json")
    spec = wrapper["execution"]
    assert _sha256(spec) == wrapper["binding_hash"]
    old = load_candidate_set_result_payload(SAVED/"candidate_set_result.json")
    assert not old["verified_candidate_ids"]
    lineage = spec["target_lineage"]
    assert lineage["model"] == "funnel" and lineage["control"] == "baseline"
    target = ValidationTarget(lineage["model"], lineage["prior"], lineage["data"], jit_compile=True)
    assert target.adapter_signature() == spec["scope"]["target_signature"]
    execution = replace(HMCCandidateExecutionConfig.from_payload(spec["config"]),
                        preparation_elapsed_seconds=0.)
    assert execution.use_xla
    original = HMCControllerConfig.from_payload(old["config"])
    epsilon = original.initial_epsilon
    multipliers = (.25, .5, 1., math.sqrt(2.), 2.)
    grid = tuple((length, tuple(epsilon*factor for factor in multipliers))
                 for length in original.primary_l_grid)
    domain = tuple(spec["scope"]["epsilon_domain"])
    assert all(domain[0] <= value <= domain[1] for _, values in grid for value in values)
    pair_count = sum(len(values) for _, values in grid)
    units = pair_count * 2 * sum(original.evidence_rungs)
    assert pair_count == 30 and units == 420
    search = replace(original, epsilon_by_l=grid, pilot_enabled=False, refinement_rounds=0,
                     max_candidates=pair_count, total_budget_units=units+1, repair_reserve_units=1,
                     max_wall_time_seconds=BUDGET)
    binding = _issue_binding(adapter=target, layers=spec["layers"],
        initial_active_state=_tensor_from_payload(spec["initial_active_state"]),
        target_scope=spec["target_scope"], target_lineage=lineage,
        preparation={"source": "saved_funnel_geometry_development_reconstruction",
                     "original_binding_hash": wrapper["binding_hash"], "multipliers": multipliers},
        config=execution, source_paths=[__file__], scope_id="saved-funnel-development",
        search_id="m7-explicit-grid-r1", epsilon_domain=domain,
        repair_factor=spec["scope"]["repair_factor"], max_repairs_per_family=0)
    assert binding.scope.mass_signature == spec["scope"]["mass_signature"]
    assert binding.scope.start_bank_signature == spec["scope"]["start_bank_signature"]
    manifest = {"source": source, "runtime": runtime, "command": sys.argv,
        "plan_file": PLAN, "result_file": str(output/"diagnostic.json"),
        "original_binding_hash": wrapper["binding_hash"], "new_binding_hash": binding.binding_hash,
        "saved_inputs": {name: file_hash(SAVED/name) for name in ("execution_spec.json", "candidate_set_result.json")},
        "same_mass_signature": binding.scope.mass_signature,
        "same_start_bank_signature": binding.scope.start_bank_signature,
        "search": search.payload(), "execution": execution.payload(), "root_seed": execution.seed,
        "scope": binding.scope.payload(), "budget_seconds": BUDGET,
        "data_version": "saved M7 funnel law and frozen preparation; no observed dataset",
        "original_failed_fit_preserved": True, "historical_qualification_reused": False}
    write_json(output/"manifest.json", manifest)
    result = tune_hmc_kernel(adapter=target, initial_position=binding.initial_active_state,
        config=search, candidate_set_adapter=binding.typed_adapter, output_dir=output/"tuning").result
    summary = {"completion": result.completion_status,
        "planned_pairs": pair_count, "observed_candidates": len(result.candidates),
        "verified_candidate_ids": result.verified_candidate_ids,
        "candidates": [{"id": c.candidate_id, "L": c.leapfrog_steps, "epsilon": c.epsilon,
                        "state": result.candidate_states[c.candidate_id]} for c in result.candidates],
        "finding": "verified_settings_found" if result.verified_candidate_ids else "no_verified_settings_observed",
        "mass_signature": binding.scope.mass_signature, "start_bank_signature": binding.scope.start_bank_signature,
        "posterior_assessed": False, "automatic_preparation_validated": False,
        "default_promoted": False, "ranking_supported": False,
        "interpretation": "Fresh supplied-grid measurements under saved geometry; original automatic failure remains in its denominator."}
    write_json(output/"diagnostic.json", summary)
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH", "").lower() != "true":
        raise RuntimeError("memory growth must be enabled before launch")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise RuntimeError("this reviewed attempt requires trusted GPU 1")
    if args.worker:
        worker(output)
        return
    output.mkdir(exist_ok=False)
    command = [sys.executable, str(Path(__file__).resolve()), "--output", str(output), "--worker"]
    started = time.monotonic()
    when = datetime.now(timezone.utc).isoformat()
    with (output/"worker.log").open("x") as log:
        try:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=BUDGET)
            code = completed.returncode
        except subprocess.TimeoutExpired:
            code = 124
    record = {"command": command, "started_utc": when, "elapsed_seconds": time.monotonic()-started,
        "returncode": code, "device": "gpu", "environment": sys.executable, "plan_file": PLAN,
        "result_file": str(output/"diagnostic.json"), "log": str(output/"worker.log"),
        "git_commit": read(SNAPSHOT/"source_snapshot.json")["git_commit"],
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "budget_seconds": BUDGET,
        "runtime": read(output/"runtime.json") if (output/"runtime.json").exists() else None}
    with (output/"gpu-diagnostic-run.json").open("x") as handle:
        json.dump(record, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(record, indent=2))
    raise SystemExit(code)


if __name__ == "__main__":
    main()
