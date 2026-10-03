"""M20 bounded development probe, with explicit source and fresh output scopes."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

PLAN = "docs/plans/bayesfilter-hmc-repair-m20-design-2026-09-22.md"
BASE = Path(__file__).resolve().parent.parent
GRID = (3, 5, 9, 13, 18, 25)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu_reference", "gpu"), required=True)
    parser.add_argument("--case", choices=("saved_funnel", "fresh_funnel", "rotated", "noncentered"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    deadline = started + args.seconds
    args.output.mkdir(parents=True, exist_ok=False)
    if args.device == "gpu":
        if os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH", "").lower() != "true":
            raise RuntimeError("GPU launch requires memory growth before imports")
        if os.environ.get("CUDA_VISIBLE_DEVICES") in (None, "", "-1"):
            raise RuntimeError("GPU launch requires an explicitly visible device")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
    sys.path.insert(0, str(args.source.resolve()))
    import tensorflow as tf
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    memory = configure_tensorflow_gpu_memory_growth(tf, require_gpu=args.device == "gpu")
    from bayesfilter.testing.inference_validation.targets import ValidationTarget
    from bayesfilter.testing.inference_validation.procedures import initial_starts
    from bayesfilter.testing.inference_validation.designs import seed_for
    from bayesfilter.testing.inference_validation.execution import source_state
    from bayesfilter.inference import (
        HMCKernelTuningConfig, HMCCandidateExecutionConfig, HMCControllerConfig, HMCAcceptancePolicy,
        tune_hmc_kernel, load_numerical_tuning_checkpoint,
        bind_hmc_candidate_set_execution_from_preparation,
    )
    from bayesfilter.inference.hmc_candidate_set_execution import bind_hmc_candidate_set_execution_new_starts
    from bayesfilter.inference.hmc_candidate_set_public import _search
    from bayesfilter.inference.hmc_preparation import prepare_operational_windowed_mass_handoff
    from bayesfilter.inference.hmc_configuration import _fixed_mass_step_upper_bound

    gpu = args.device == "gpu"
    target_id = {"rotated": "rotated_gaussian", "noncentered": "funnel_noncentered"}.get(args.case, "funnel")
    parameters = {"angle": .6, "condition": 100.} if args.case == "rotated" else {"scale": 3.}
    target = ValidationTarget(target_id, parameters, jit_compile=gpu)
    manifest = {"command": sys.argv, "plan_file": PLAN, "source": source_state(),
        "environment": sys.executable, "device": args.device, "gpu_intentionally_hidden": not gpu,
        "memory_policy": memory, "jit_compile": gpu, "seed": args.seed, "data_version": "exact synthetic law",
        "target": target_id, "parameters": parameters, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "result_file": str(args.output / "result.json"), "cpu_threads": {"intra": 1, "inter": 1}}
    write(args.output / "manifest.json", manifest)
    results = []
    try:
        cfg = HMCKernelTuningConfig(preset="standard", use_xla=gpu, target_scope="inference_validation",
            seed=seed_for(args.seed, args.device, args.case, "preparation"),
            bootstrap_initialization_rounds=20, metric_evidence_policy="finite_window",
            metric_probe_num_results=16, preparation_max_restarts=3,
            candidate_search_bound_expansion_steps=1)
        evidence = HMCCandidateExecutionConfig(measurement_num_results=128, verification_num_results=128,
            num_warmup_steps=8, seed=seed_for(args.seed, args.device, args.case, "candidate"),
            acceptance_policy=HMCAcceptancePolicy(),
            use_xla=gpu, target_status_trace_policy="none",
            non_xla_reason=None if gpu else "M20 explicit CPU reference development")
        if args.case == "saved_funnel":
            tag = "gpu" if gpu else "cpu"
            old = BASE / "m17-r1" / f"matrix-{tag}-r1" / f"m17-{args.device}-centered" / "replication-0000/tuning/tuning_checkpoint.json"
            binding, old_controller = load_numerical_tuning_checkpoint(old, adapter=target)
            epsilon = old_controller.config.initial_epsilon
            original = old_controller.config
            # Same serialized start coordinates: the checked inverse/forward
            # path is checked again by the repository new-scope factory.
            physical_starts = binding.position_samples(binding.initial_active_state)
            preparation = None
            write(args.output / "geometry.json", {"parent_checkpoint": str(old),
                "parent_binding_hash": binding.binding_hash, "mass_signature": binding.scope.mass_signature,
                "epsilon": epsilon, "qualification_transferred": False})
        else:
            def progress(phase, details):
                with (args.output / "preparation-events.jsonl").open("a") as handle:
                    handle.write(json.dumps({"phase": phase, "elapsed_seconds": time.monotonic()-started}, allow_nan=False)+"\n")
                if time.monotonic() >= deadline:
                    raise TimeoutError("M20 preparation budget exhausted")
            preparation = prepare_operational_windowed_mass_handoff(adapter=target,
                initial_position=initial_starts(target, "dispersed")[0], config=cfg, progress_callback=progress)
            epsilon = preparation["windowed_stage"].operational_warmup_result.final_kernel_state.epsilon
            original = _search(None, epsilon=epsilon, grid=GRID)
            write(args.output / "geometry.json", {"epsilon": epsilon,
                "preparation_bound": _fixed_mass_step_upper_bound(preparation["windowed_stage"]),
                "final_adapter_signature": preparation["final_adapter_signature"],
                "mass_signature": preparation["adapted_mass_artifact_signature"], "qualification_transferred": False})
        arms = ("baseline", "epsilon_grid") if "funnel" in args.case else ("baseline",)
        for arm in arms:
            if time.monotonic() >= deadline:
                results.append({"arm": arm, "status": "unfunded"})
                continue
            search = replace(original, max_wall_time_seconds=max(1., deadline-time.monotonic()))
            if arm == "epsilon_grid":
                search = replace(search, epsilon_by_l=tuple((l, tuple(epsilon * factor for factor in (.125, .25, .5, 1.))) for l in GRID),
                    refinement_rounds=2, explore_failed_intervals=True)
            execution = replace(evidence, seed=seed_for(args.seed, args.case, args.device, arm, "candidate"))
            if preparation is None:
                arm_binding = bind_hmc_candidate_set_execution_new_starts(binding=binding,
                    initial_position=physical_starts, config=execution, scope_id="m20_saved_geometry",
                    search_id=arm, max_repairs_per_family=binding.scope.max_repairs_per_family)
                tf.debugging.assert_near(arm_binding.initial_active_state, binding.initial_active_state, atol=1e-12, rtol=1e-12)
                assert arm_binding.scope.mass_signature == binding.scope.mass_signature
            else:
                arm_binding = bind_hmc_candidate_set_execution_from_preparation(adapter=target, preparation=preparation,
                    target_lineage={"model": target_id, "prior": parameters, "data": None},
                    config=execution, source_paths=[str(Path(sys.modules[ValidationTarget.__module__].__file__))],
                    scope_id="m20_fresh_geometry", search_id=arm, epsilon_domain=(epsilon / 64., epsilon * 2.),
                    repair_factor=2., max_repairs_per_family=5, preparation_bound_expansion_steps=1)
            run = tune_hmc_kernel(adapter=target, initial_position=arm_binding.initial_active_state,
                config=search, candidate_set_adapter=arm_binding.typed_adapter, output_dir=args.output / arm / "tuning")
            result = run.result
            row = {"arm": arm, "status": result.final_status, "verified": list(result.verified_candidate_ids),
                "candidates": len(result.candidates), "candidate_states": result.candidate_states,
                "receipts": [r.payload() for r in result.verification_receipts], "budget_used_units": result.budget_used_units,
                "mass_signature": arm_binding.scope.mass_signature, "start_signature": arm_binding.scope.start_bank_signature}
            if args.case in ("rotated", "noncentered"):
                row["members"] = assess_members(args, target, arm_binding, result, deadline)
            results.append(row)
            write(args.output / arm / "summary.json", row)
        write(args.output / "result.json", {"status": "complete", "arms": results,
            "elapsed_seconds": time.monotonic()-started, "ranking_supported": False, "default_promoted": False})
    except Exception as exc:
        write(args.output / "failure.json", {"exception": type(exc).__name__, "message": str(exc),
            "traceback": traceback.format_exc(), "elapsed_seconds": time.monotonic()-started, "completed_arms": results})
        raise


def assess_members(args, target, binding, result, deadline):
    from bayesfilter.inference import (
        build_retained_bound_hmc_archive_runner_from_candidate_set_result,
        HMCPosteriorAssessmentPolicy, HMCPrecisionPolicy, HMCPrecisionTarget,
        SequentialNeuTraHMCConfig, run_hmc_posterior,
    )
    from bayesfilter.runtime.durable_tensor_checkpoint import DurableTensorCheckpoint
    from bayesfilter.testing.inference_validation.designs import seed_for
    from bayesfilter.testing.inference_validation.storage import write_tensor
    selected = {}
    for l in (3, 9, 18, 25):
        candidates = sorted(c for c in result.verified_candidate_ids if result.replay_candidate(c).leapfrog_steps == l)
        selected[str(l)] = candidates[0] if candidates else None
    write(args.output / "posterior-selection.json", {"selected": selected, "verified": result.verified_candidate_ids,
        "rule": "first verified identity per declared L; before posterior; missing L has no substitute"})
    targets = tuple(HMCPrecisionTarget(name, kind=kind, probability=.5 if kind=="quantile" else None, mcse_absolute_max=.05)
                    for name in target.spec.parameters for kind in ("mean", "quantile"))
    policy = HMCPosteriorAssessmentPolicy(precision=HMCPrecisionPolicy(targets, method="lugsail", jit_compile=args.device=="gpu"))
    rows = []
    for l, cid in selected.items():
        if cid is None or time.monotonic() >= deadline:
            rows.append({"L": int(l), "candidate_id": cid, "status": "unavailable" if cid is None else "unfunded"})
            continue
        directory = args.output / "members" / l
        directory.mkdir(parents=True, exist_ok=False)
        member = build_retained_bound_hmc_archive_runner_from_candidate_set_result(candidate_set_result=result,
            candidate_id=cid, retained_binding=binding)
        member.export(directory / "member.json")
        config = SequentialNeuTraHMCConfig(step_size=member.step_size, num_leapfrog_steps=member.num_leapfrog_steps,
            jit_compile=args.device=="gpu", warmup_seed=seed_for(args.seed,args.device,args.case,l,"warmup"),
            retained_seed=seed_for(args.seed,args.device,args.case,l,"retained"),
            warmup_chunk_results=500, warmup_min_results=2000, warmup_check_window_results=1000,
            warmup_max_results=10000, retained_chunk_results=500, retained_min_results=1000,
            retained_max_results=10000, assessment_policy=policy)
        with DurableTensorCheckpoint(directory / "chunks", {"member": member.member_hash, "config": config.payload()}) as store:
            post = run_hmc_posterior(member=member, config=config, parameter_names=target.spec.parameters,
                model_transform=target.to_model, checkpoint_store=store,
                budget_check=lambda _: time.monotonic() < deadline)
        write_tensor(directory / "retained.tensor", post["private_retained_raw"])
        write_tensor(directory / "warmup.tensor", post["private_warmup_raw"])
        public = {k:v for k,v in post.items() if not k.startswith("private_")}
        write(directory / "posterior.json", public)
        last = public["retained_checks"][-1]["modern_rhat"] if public["retained_checks"] else {}
        rows.append({"L": int(l), "candidate_id": cid, "epsilon": member.step_size, "status": "assessed",
            "passed": post["passed"], "warmup_draws": post["warmup_results_per_chain"],
            "retained_draws": post["retained_results_per_chain"], "hard_vetoes": post["hard_vetoes"],
            "precision": last.get("precision"), "max_rhat": last.get("max_finite_rhat")})
    return rows


if __name__ == "__main__":
    main()
