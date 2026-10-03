"""Fresh public searches inside an omitted interval, at exactly saved geometry."""
import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT/"source-gpu-r6"
SAVED = ROOT/"posteriordb-geometry-hint-regression-pilot-0-gpu-r1/tuning"
PLAN = "docs/plans/bayesfilter-hmc-repair-master-program-2026-09-16.md"


def read(path):
    return json.loads(path.read_text())


def write(path, payload):
    with path.open("x") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")


def worker(replication, output):
    sys.path.insert(0, str(SOURCE))
    from bayesfilter.testing.inference_validation.execution import configure_worker, source_state
    runtime = configure_worker(SimpleNamespace(device="gpu"))
    write(output/"runtime.json", runtime)
    source = source_state()
    assert source["identity"] == read(SOURCE/"source_snapshot.json")["source_identity"]
    from bayesfilter.testing.inference_validation.posteriordb_targets import PosteriordbTarget
    from bayesfilter.testing.inference_validation.designs import seed_for
    from bayesfilter.testing.inference_validation.engines.pipeline import check_inventory
    from bayesfilter.inference import HMCControllerConfig, HMCCandidateExecutionConfig, tune_hmc_kernel
    from bayesfilter.inference.hmc_candidate_set_execution import _issue_binding, _tensor_from_payload
    from bayesfilter.inference.hmc_candidate_set_tuning import _sha256
    from bayesfilter.inference.hmc_candidate_set_artifacts import load_candidate_set_result_payload
    original = read(SAVED/"candidate_set_result.json")
    load_candidate_set_result_payload(SAVED/"candidate_set_result.json")
    wrapper = read(SAVED/"execution_spec.json")
    spec = wrapper["execution"]
    if _sha256(spec) != wrapper["binding_hash"] or original["verified_candidate_ids"]:
        raise ValueError("saved empty-search geometry identity mismatch")
    lineage = spec["target_lineage"]
    target = PosteriordbTarget(lineage["model"], lineage["data"], jit_compile=True)
    if target.adapter_signature() != spec["scope"]["target_signature"]:
        raise ValueError("saved target mismatch")
    # Infer the exact endpoints from immutable records, not rounded result prose.
    endpoints = sorted({c["epsilon"] for c in original["candidates"]})
    if len(endpoints) != 2 or endpoints[1] != 2.*endpoints[0]:
        raise ValueError("saved search does not have the predeclared doubled endpoints")
    grid = tuple(original["config"]["primary_l_grid"])
    interior = tuple(endpoints[0]*2.**(j/4.) for j in (1, 2, 3))
    seed = seed_for(2026091840, "regression-interior", replication)
    execution = replace(HMCCandidateExecutionConfig.from_payload(spec["config"]),
                        preparation_elapsed_seconds=0., seed=seed)
    binding = _issue_binding(adapter=target, layers=spec["layers"],
        initial_active_state=_tensor_from_payload(spec["initial_active_state"]),
        target_scope=spec["target_scope"], target_lineage=lineage,
        preparation={"source": "saved_regression_interval_diagnosis", "original_binding_hash": wrapper["binding_hash"]},
        config=execution, source_paths=[__file__, str(SAVED/"execution_spec.json"),
                                       str(SAVED/"candidate_set_result.json")],
        scope_id="regression-interior-diagnostic", search_id=f"fresh-{replication}",
        epsilon_domain=tuple(spec["scope"]["epsilon_domain"]),
        repair_factor=spec["scope"]["repair_factor"], max_repairs_per_family=0)
    for key in ("mass_signature", "start_bank_signature", "target_signature", "adapter_signature"):
        if getattr(binding.scope, key) != spec["scope"][key]:
            raise ValueError("reconstruction changed " + key)
    # Eighteen pairs, each with at most three measurement and three verification looks.
    search = HMCControllerConfig(primary_l_grid=grid, epsilon_by_l=tuple((l, interior) for l in grid),
        total_budget_units=109, repair_reserve_units=1, candidate_reserve_units=6,
        allow_repair_from_free_pool=False, pilot_enabled=False, evidence_rungs=(1, 2, 4),
        refinement_rounds=0, max_candidates=18, max_wall_time_seconds=840.)
    write(output/"manifest.json", {"command": sys.argv, "source": source, "runtime": runtime,
        "plan_file": PLAN, "result_file": str(output/"assessment.json"), "seed": list(seed),
        "baseline": str(SAVED), "input_sha256": {name: hashlib.sha256((SAVED/name).read_bytes()).hexdigest()
            for name in ("execution_spec.json", "candidate_set_result.json")},
        "endpoints": endpoints, "interior": interior, "search": search.payload(),
        "budget_provenance": "108 units = 18 pairs times 6 looks, plus 1 unused repair unit required by config; 60 seconds for setup/reporting",
        "original_receipts_reused": False, "mass_signature": binding.scope.mass_signature,
        "start_bank_signature": binding.scope.start_bank_signature})
    run = tune_hmc_kernel(adapter=target, initial_position=binding.initial_active_state,
        config=search, candidate_set_adapter=binding.typed_adapter, output_dir=output/"tuning")
    result = run.result
    native = read(output/"tuning/candidate_set_result.json")
    inventory = check_inventory(native)
    if inventory["failures"]:
        raise ValueError("candidate inventory failure")
    write(output/"assessment.json", {"replication": replication,
        "completion": result.completion_status, "verified_members": len(result.verified_candidate_ids),
        "verified_candidate_ids": list(result.verified_candidate_ids), "inventory": inventory,
        "candidate_states": native["candidate_states"], "posterior_assessed": False,
        "interpretation": "Existence of freshly verified interior pairs only; no posterior, automatic-fit or ranking claim."})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=int)
    args = parser.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "1"
    assert os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH", "").lower() == "true"
    if args.worker is not None:
        output = ROOT/f"regression-interior-{args.worker}-gpu-r2"
        try:
            worker(args.worker, output)
        except Exception as exc:
            write(output/"failure.json", {"exception": type(exc).__name__, "message": str(exc)})
            raise
        return
    for i in range(3):
        if not (ROOT/f"posteriordb-geometry-hint-regression-fresh-{i}-gpu-r1/gpu-diagnostic-run.json").exists():
            raise RuntimeError("finish the sequential geometry queue before this diagnosis")
    spent = sum(read(p)["elapsed_seconds"] for p in ROOT.glob("posteriordb-*-gpu-r1/gpu-diagnostic-run.json"))
    spent += sum(read(p)["elapsed_seconds"] for p in ROOT.glob("regression-interior-*-gpu-r1/gpu-diagnostic-run.json"))
    if spent+2700. > 16000.:
        raise RuntimeError("interval experiment exceeds matched-reference reservation")
    for i in range(3):
        output = ROOT/f"regression-interior-{i}-gpu-r2"
        output.mkdir(exist_ok=False)
        command = [sys.executable, str(Path(__file__).resolve()), "--worker", str(i)]
        started = time.monotonic()
        when = datetime.now(timezone.utc).isoformat()
        with (output/"worker.log").open("x") as log:
            try:
                code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=900.).returncode
            except subprocess.TimeoutExpired:
                code = 124
        write(output/"gpu-diagnostic-run.json", {"command": command, "started_utc": when,
            "elapsed_seconds": time.monotonic()-started, "returncode": code, "device": "gpu",
            "environment": sys.executable, "plan_file": PLAN, "result_file": str(output/"assessment.json"),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "git_commit": read(SOURCE/"source_snapshot.json")["git_commit"],
            "runtime": read(output/"runtime.json") if (output/"runtime.json").exists() else None})
        if code:
            raise RuntimeError("unresolved interval diagnostic failure: " + str(output))
        result = read(output/"assessment.json")
        print("INTERIOR", i, result["completion"], "verified", result["verified_members"], flush=True)


if __name__ == "__main__":
    main()
