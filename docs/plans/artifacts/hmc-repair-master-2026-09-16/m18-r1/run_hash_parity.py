"""Matched GPU numerical parity with only the diagnostic hash hook changed."""
import argparse
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
PLAN = "docs/plans/bayesfilter-hmc-repair-m18-design-2026-09-21.md"


def read(path):
    return json.loads(path.read_text())


def worker(args, output):
    sys.path.insert(0, str(args.source))
    from bayesfilter.testing.inference_validation.execution import configure_worker, source_state
    from bayesfilter.testing.inference_validation.storage import write_json, write_tensor
    runtime = configure_worker(SimpleNamespace(device="gpu"))
    write_json(output / "runtime.json", runtime)
    source = source_state()
    if source["identity"] != read(args.source / "source_snapshot.json")["source_identity"]:
        raise ValueError("frozen numerical source changed")
    import tensorflow as tf
    from bayesfilter.inference import hmc_candidate_set_checkpoint as checkpoints
    from bayesfilter.inference.hmc_candidate_set_tuning import _sha256
    if args.arm == "generic":
        checkpoints._json_native_sha256 = _sha256
    from bayesfilter.testing.inference_validation.targets import ValidationTarget
    from bayesfilter.testing.inference_validation.procedures import initial_starts
    from bayesfilter.testing.inference_validation.designs import seed_for
    from bayesfilter.testing.inference_validation.engines.pipeline import check_inventory
    from bayesfilter.inference import (PrecomputedMassArtifact, HMCCandidateExecutionConfig, HMCAcceptancePolicy,
        HMCControllerConfig, bind_hmc_candidate_set_execution, tune_hmc_kernel,
        resume_hmc_candidate_set_tuning, build_retained_bound_hmc_archive_runner_from_candidate_set_result,
        load_hmc_candidate_retained_runner)
    data = [7, 12] if args.case == "beta_binomial" else None
    target = ValidationTarget(args.case, data=data, jit_compile=True)
    starts = initial_starts(target, "dispersed")
    mass = PrecomputedMassArtifact(position=[0.] * target.parameter_dim,
        covariance=tf.eye(target.parameter_dim, dtype=tf.float64),
        factor=tf.eye(target.parameter_dim, dtype=tf.float64),
        adapter_signature=target.adapter_signature(), position_role="fixed-parity-fixture",
        covariance_source="explicit identity; no automatic preparation claim")
    seed = seed_for(2026092198, "hash-parity", args.case)
    execution = HMCCandidateExecutionConfig(measurement_num_results=64, verification_num_results=64,
        pilot_num_results=64, num_warmup_steps=8, seed=seed, use_xla=True,
        target_status_trace_policy="none", acceptance_policy=HMCAcceptancePolicy())
    binding = bind_hmc_candidate_set_execution(adapter=target, initial_position=starts,
        mass_artifact=mass, config=execution, target_scope="inference_validation",
        scope_id="m18-hash-parity", search_id=args.case, epsilon_domain=(.005, 3.),
        repair_factor=2., max_repairs_per_family=5,
        target_lineage={"model": args.case, "data": data, "control": "baseline"},
        source_paths=[__file__, str(Path(sys.modules[ValidationTarget.__module__].__file__))])
    search = HMCControllerConfig(primary_l_grid=(3, 5, 9), initial_epsilon=.5,
        pilot_enabled=True, refinement_rounds=0, total_budget_units=72,
        repair_reserve_units=12, max_candidates=40, evidence_rungs=(1, 2, 4))
    write_json(output / "manifest.json", {"command": sys.argv, "runtime": runtime,
        "source": source, "seed": seed, "data_version": target.adapter_signature(),
        "plan_file": PLAN, "result_file": str(output / "numerical.json"),
        "arm": args.arm, "case": args.case, "search": search.payload(),
        "sole_intervention": "checkpoint JSON hashing hook; same source, seeds and numerical specification"})
    tune_hmc_kernel(adapter=target, initial_position=starts, config=search,
        candidate_set_adapter=binding.typed_adapter, output_dir=output / "tuning", max_work_items=2)
    run = resume_hmc_candidate_set_tuning(output / "tuning/tuning_checkpoint.json", adapter=target)
    binding = run.adapter._execution_binding
    payload = read(output / "tuning/candidate_set_result.json")
    inventory = check_inventory(payload)
    if inventory["failures"]:
        raise ValueError("candidate inventory failed")
    evidence = sorted(binding._evidence.values(), key=lambda record: record["work"]["ordinal"])
    projected = [{key: row[key] for key in
        ("candidate", "work", "initial_state", "seed", "samples", "trace", "analysis")} for row in evidence]
    numerical = {"binding_hash": binding.binding_hash, "scope": payload["scope"],
        "candidates": payload["candidates"], "states": payload["candidate_states"],
        "verified": payload["verified_candidate_ids"], "completion": payload["completion_status"],
        "evidence": projected}
    selected = sorted(run.result.verified_candidate_ids)[:1]
    if not selected:
        raise ValueError("parity fixture produced no verified retained member")
    member = build_retained_bound_hmc_archive_runner_from_candidate_set_result(
        candidate_set_result=run.result, candidate_id=selected[0], retained_binding=binding)
    member.export(output / "member.json")
    member = load_hmc_candidate_retained_runner(output / "member.json", adapter=target)
    discarded = member.run(num_results=64, seed=seed_for(*seed, "discarded"),
        output_dir=output / "discarded")
    retained = member.run(num_results=128, seed=seed_for(*seed, "retained"),
        output_dir=output / "retained", previous_archive=output / "discarded/retained_archive.json")
    for name, result in (("discarded", discarded), ("retained", retained)):
        write_tensor(output / (name + ".tensor"), result["position_samples"])
        numerical[name + "_draws"] = result["position_samples"].numpy().tolist()
    write_json(output / "numerical.json", numerical)
    write_json(output / "assessment.json", {"inventory": inventory,
        "selected": selected, "verified_count": len(run.result.verified_candidate_ids),
        "numerical_hash": _sha256(numerical), "timing_excluded_from_comparison": True,
        "posterior_accuracy_claim": False})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--case", choices=("gaussian", "beta_binomial"), required=True)
    parser.add_argument("--arm", choices=("generic", "native"), required=True)
    parser.add_argument("--seconds", type=float, default=400)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, "", "-1")
    assert os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH", "").lower() == "true"
    if not 0 < args.seconds <= 400:
        raise ValueError("unallocated GPU parity allowance")
    output = ROOT / ("hash-parity-" + args.case + "-" + args.arm + "-r1")
    if args.worker:
        try:
            worker(args, output)
        except Exception as exc:
            (output / "failure.json").write_text(json.dumps(
                {"exception": type(exc).__name__, "message": str(exc)}, indent=2))
            raise
        return
    output.mkdir(exist_ok=False)
    command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--worker"]
    started, when = time.monotonic(), datetime.now(timezone.utc).isoformat()
    with (output / "worker.log").open("x") as log:
        try:
            code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=args.seconds).returncode
        except subprocess.TimeoutExpired:
            code = 124
    record = {"command": command, "started_utc": when, "elapsed_seconds": time.monotonic() - started,
        "returncode": code, "device": "gpu", "environment": sys.executable,
        "plan_file": PLAN, "result_file": str(output / "assessment.json"),
        "git_commit": read(args.source / "source_snapshot.json")["git_commit"],
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "runtime": read(output / "runtime.json") if (output / "runtime.json").exists() else None}
    with (output / "diagnostic-run.json").open("x") as handle:
        json.dump(record, handle, indent=2)
    print(json.dumps(record))
    raise SystemExit(code)


if __name__ == "__main__":
    main()
