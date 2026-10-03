"""M17 public conditional position-field mechanics on several actual targets."""
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
PLAN = "docs/plans/bayesfilter-hmc-repair-m17-design-2026-09-21.md"
CASES = {"gaussian": None, "beta_binomial": [7, 12],
         "lgssm_location": [1., -1., .5], "banana": None}


def read(path):
    return json.loads(path.read_text())


def worker(args, output):
    sys.path.insert(0, str(args.source))
    from bayesfilter.testing.inference_validation.execution import configure_worker, source_state
    from bayesfilter.testing.inference_validation.storage import write_json
    runtime = configure_worker(SimpleNamespace(device=args.device))
    write_json(output / "runtime.json", runtime)
    source = source_state()
    if source["identity"] != read(args.source / "source_snapshot.json")["source_identity"]:
        raise ValueError("frozen source differs from plan")
    import tensorflow as tf
    from bayesfilter.testing.inference_validation.targets import ValidationTarget
    from bayesfilter.testing.inference_validation.procedures import initial_starts
    from bayesfilter.testing.inference_validation.designs import seed_for
    from bayesfilter.inference import (
        FrozenPositionOnlyForce, FrozenTargetPotential, FourChainMeanBandAcceptancePolicy,
        DETERMINISTIC_POSITION_ONLY_PROPOSAL_FIELD_SEMANTICS,
        TensorFlowHMCKernelTuningConfig, HMCControllerConfig,
        bind_neural_force_hmc_tuning_runner, tune_hmc_kernel,
        resume_position_field_candidate_tuning,
        build_retained_bound_hmc_archive_runner_from_candidate_set_result)
    jit = args.device == "gpu"
    target = ValidationTarget(args.case, data=CASES[args.case], jit_compile=jit)
    force = FrozenPositionOnlyForce(lambda q: -.8 * target.log_prob_and_grad(q)[1],
        identity="m17-force-" + target.adapter_signature(),
        semantics=DETERMINISTIC_POSITION_ONLY_PROPOSAL_FIELD_SEMANTICS)
    potential = FrozenTargetPotential(lambda q: -target.log_prob_and_grad(q)[0],
        identity="m17-potential-" + target.adapter_signature())
    binding = bind_neural_force_hmc_tuning_runner(force=force, target=potential,
        target_scope="inference_validation")
    seed = seed_for(2026092197, "position-field", args.device, args.case)
    config = TensorFlowHMCKernelTuningConfig(parameter_dimension=target.parameter_dim,
        evidence_role="diagnostic_only", mass_window_results=(32, 64),
        step_adaptation_results=64, verification_results=128, max_leapfrog_steps=25,
        initial_step_size=.5, budget_provenance="M17 bounded conditional-mechanics matrix",
        initial_step_size_provenance="explicit .5 hypothesis; candidate-specific pilot",
        geometry_provenance="identity parameter scales; no reference covariance",
        use_xla=jit, non_xla_reason=None if jit else "explicit CPU reference",
        target_scope="inference_validation", target_accept_prob=.70,
        verification_repair_rounds=3, step_repair_factor=2., mass_shrinkage=.10,
        covariance_jitter=1e-9, eigenvalue_floor=1e-9, max_condition_number=1e8, seed=seed,
        acceptance_policy=FourChainMeanBandAcceptancePolicy(
            overall_band=(.65, .75), per_chain_band=(.55, .85)))
    search = HMCControllerConfig(primary_l_grid=(3, 5, 9, 13, 18, 25),
        initial_epsilon=.5, pilot_enabled=True, refinement_rounds=1,
        max_candidates=100, total_budget_units=300, repair_reserve_units=40,
        max_wall_time_seconds=args.seconds - 15)
    write_json(output / "manifest.json", {"command": sys.argv, "source": source,
        "runtime": runtime, "seed": list(seed), "data_version": target.adapter_signature(),
        "plan_file": PLAN, "result_file": str(output / "assessment.json"),
        "case": args.case, "data": CASES[args.case], "force_multiplier": .8,
        "configuration": config.payload(), "search": search.payload(),
        "evidence_scope": "conditional position-field mechanics; no exact-score retained authority"})
    first = tune_hmc_kernel(adapter=target, initial_position=initial_starts(target, "dispersed"),
        parameter_scales=tf.ones([target.parameter_dim], tf.float64), config=config,
        search_config=search, runner_binding=binding, output_dir=output / "tuning",
        max_work_items=1)
    original = json.loads(json.dumps(first.result.observations))
    resumed = resume_position_field_candidate_tuning(output / "tuning/controller_checkpoint.json",
        adapter=target, runner_binding=binding)
    unchanged = list(resumed.result.observations[:len(original)]) == original
    denied = False
    if resumed.result.candidates:
        try:
            build_retained_bound_hmc_archive_runner_from_candidate_set_result(
                candidate_set_result=resumed.result,
                candidate_id=resumed.result.candidates[0].candidate_id, retained_binding=binding)
        except (TypeError, ValueError):
            denied = True
    conditional = (not resumed.numerical_handoff_authority and
        all(not row["observation"]["numerical_handoff_authority"] for row in resumed.result.observations))
    result = {"case": args.case, "completion": resumed.result.completion_status,
        "verified_candidates": list(resumed.result.verified_candidate_ids),
        "observations": len(resumed.result.observations),
        "executed_stages": sorted({row["stage"] for row in resumed.result.observations}), "resumed_prefix_unchanged": unchanged,
        "conditional_authority_preserved": conditional, "exact_score_member_denied": denied,
        "passed": unchanged and conditional and denied,
        "posterior_accuracy_assessed": False, "force_is_exact_score": False}
    write_json(output / "assessment.json", result)
    if not result["passed"]:
        raise ValueError("conditional route invariant failed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu_reference", "gpu"), required=True)
    parser.add_argument("--case", choices=tuple(CASES), required=True)
    parser.add_argument("--seconds", type=float, required=True)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if not 15 < args.seconds <= (200 if args.device == "gpu" else 100):
        raise ValueError("unallocated position-field budget")
    if args.device == "gpu":
        assert args.case in {"gaussian", "beta_binomial"}
        assert os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, "", "-1")
        assert os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH", "").lower() == "true"
    else:
        assert os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
    output = ROOT / ("position-field-" + args.device + "-" + args.case + "-r2")
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
        "returncode": code, "device": args.device, "gpu_intentionally_hidden": args.device == "cpu_reference",
        "environment": sys.executable, "plan_file": PLAN, "result_file": str(output / "assessment.json"),
        "git_commit": read(args.source / "source_snapshot.json")["git_commit"],
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "runtime": read(output / "runtime.json") if (output / "runtime.json").exists() else None}
    with (output / "diagnostic-run.json").open("x") as handle:
        json.dump(record, handle, indent=2)
    print(json.dumps(record))
    raise SystemExit(code)


if __name__ == "__main__":
    main()
