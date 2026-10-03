"""M9 fresh disabled/enabled searches from an immutable M8 regression handoff."""
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
SAVED = ROOT.parent/"m8-r1/posteriordb-geometry-hint-regression-pilot-0-gpu-r1/tuning"
PLAN = "docs/plans/bayesfilter-hmc-repair-master-program-2026-09-16.md"


def read(path):
    return json.loads(path.read_text())


def write(path, payload):
    with path.open("x") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")


def worker(args, output):
    sys.path.insert(0, str(args.source.resolve()))
    from bayesfilter.testing.inference_validation.execution import configure_worker, source_state
    runtime = configure_worker(SimpleNamespace(device="gpu"))
    write(output/"runtime.json", runtime)
    source = source_state()
    assert source["identity"] == read(args.source/"source_snapshot.json")["source_identity"]
    from bayesfilter.testing.inference_validation.posteriordb_targets import PosteriordbTarget
    from bayesfilter.testing.inference_validation.designs import seed_for
    from bayesfilter.testing.inference_validation.engines.pipeline import check_inventory
    from bayesfilter.inference import (HMCControllerConfig, HMCCandidateExecutionConfig, tune_hmc_kernel,
        build_retained_bound_hmc_archive_runner_from_candidate_set_result, load_hmc_candidate_retained_runner)
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
    endpoints = sorted({c["epsilon"] for c in original["candidates"]})
    if len(endpoints) != 2 or endpoints[1] != 2.*endpoints[0]:
        raise ValueError("saved search does not have the declared doubled endpoints")
    # Identical fresh root seed, binding and initial candidate identities in both
    # arms: common initial work can be compared exactly, without a ranking claim.
    seed = seed_for(2026091940, "m9-saved-interval", 0)
    execution = replace(HMCCandidateExecutionConfig.from_payload(spec["config"]),
                        preparation_elapsed_seconds=0., seed=seed)
    binding = _issue_binding(adapter=target, layers=spec["layers"],
        initial_active_state=_tensor_from_payload(spec["initial_active_state"]),
        target_scope=spec["target_scope"], target_lineage=lineage,
        preparation={"source": "m9_saved_regression", "original_binding_hash": wrapper["binding_hash"]},
        config=execution, source_paths=[__file__, str(SAVED/"execution_spec.json"),
                                       str(SAVED/"candidate_set_result.json")],
        scope_id="m9-saved-regression", search_id="fresh-0",
        epsilon_domain=tuple(spec["scope"]["epsilon_domain"]),
        repair_factor=spec["scope"]["repair_factor"], max_repairs_per_family=5)
    for key in ("mass_signature", "start_bank_signature", "target_signature", "adapter_signature"):
        if getattr(binding.scope, key) != spec["scope"][key]:
            raise ValueError("reconstruction changed " + key)
    search = replace(HMCControllerConfig.from_payload(original["config"]),
        explore_failed_intervals=args.enabled, refinement_rounds=2, max_wall_time_seconds=args.seconds-60.)
    write(output/"manifest.json", {"command": sys.argv, "source": source, "runtime": runtime,
        "plan_file": PLAN, "result_file": str(output/"assessment.json"), "seed": list(seed),
        "baseline": str(SAVED), "input_sha256": {name: hashlib.sha256((SAVED/name).read_bytes()).hexdigest()
            for name in ("execution_spec.json", "candidate_set_result.json")},
        "endpoints": endpoints, "search": search.payload(), "original_receipts_reused": False,
        "mass_signature": binding.scope.mass_signature, "start_bank_signature": binding.scope.start_bank_signature})
    run = tune_hmc_kernel(adapter=target, initial_position=binding.initial_active_state,
        config=search, candidate_set_adapter=binding.typed_adapter, output_dir=output/"tuning")
    native = load_candidate_set_result_payload(output/"tuning/candidate_set_result.json")
    inventory = check_inventory(native)
    if inventory["failures"]:
        raise ValueError("candidate inventory failure")
    replayed = []
    for cid in run.result.verified_candidate_ids:
        member = build_retained_bound_hmc_archive_runner_from_candidate_set_result(
            candidate_set_result=run.result, candidate_id=cid, retained_binding=binding)
        path = output/(cid.rsplit(":", 1)[-1] + "-member.json")
        member.export(path)
        loaded = load_hmc_candidate_retained_runner(path, adapter=target)
        assert member.member_hash == loaded.member_hash
        candidate = run.result.replay_candidate(cid)
        replayed.append({"candidate_id": cid, "L": candidate.leapfrog_steps,
                         "epsilon": candidate.epsilon, "member_hash": member.member_hash})
    write(output/"assessment.json", {"enabled": args.enabled, "completion": run.result.completion_status,
        "verified_members": len(replayed), "verified": replayed, "inventory": inventory,
        "interval_proposals": [e for e in native["accounting_events"] if e["event"] == "rejected_interval_proposed"],
        "posterior_assessed": False, "ranking_supported": False, "default_promoted": False,
        "interpretation": "Fresh candidate reachability and checked replay from selected saved geometry only."})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=900.)
    parser.add_argument("--enabled", action="store_true")
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "0"
    assert os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH", "").lower() == "true"
    output = ROOT/("saved-enabled-gpu-r1" if args.enabled else "saved-disabled-gpu-r1")
    if args.worker:
        try:
            worker(args, output)
        except Exception as exc:
            write(output/"failure.json", {"exception": type(exc).__name__, "message": str(exc)})
            raise
        return
    costs = [read(p) for p in ROOT.glob("*/gpu-diagnostic-run.json")]
    if len(costs) >= 10 or sum(r["elapsed_seconds"] for r in costs) + args.seconds > 10000.:
        raise RuntimeError("M9 attempt/worker-wall budget exhausted")
    output.mkdir(exist_ok=False)
    command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--worker"]
    began = time.monotonic()
    when = datetime.now(timezone.utc).isoformat()
    with (output/"worker.log").open("x") as log:
        try:
            code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=args.seconds).returncode
        except subprocess.TimeoutExpired:
            code = 124
    write(output/"gpu-diagnostic-run.json", {"command": command, "started_utc": when,
        "elapsed_seconds": time.monotonic()-began, "returncode": code, "device": "gpu",
        "environment": sys.executable, "plan_file": PLAN, "result_file": str(output/"assessment.json"),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "git_commit": read(args.source/"source_snapshot.json")["git_commit"],
        "runtime": read(output/"runtime.json") if (output/"runtime.json").exists() else None})
    print(json.dumps(read(output/"gpu-diagnostic-run.json")))
    raise SystemExit(code)


if __name__ == "__main__":
    main()
