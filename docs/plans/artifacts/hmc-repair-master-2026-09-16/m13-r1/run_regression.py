"""M13 fresh no-hint fits with longer sequential metric-boundary probes."""
import argparse
from datetime import datetime,timezone
from dataclasses import replace
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
M8 = ROOT.parent/"m8-r1"
REPO = ROOT.parents[4]
CHECKOUT = REPO/".localresources/posteriordb-20260918"
PLAN = "docs/plans/bayesfilter-hmc-repair-master-program-2026-09-16.md"


def read(path):
    return json.loads(path.read_text())


def worker(args,output):
    sys.path.insert(0,str(args.source))
    from bayesfilter.testing.inference_validation.execution import configure_worker,source_state
    from bayesfilter.testing.inference_validation.storage import write_json,write_tensor
    runtime = configure_worker(SimpleNamespace(device="gpu"))
    write_json(output/"runtime.json",runtime)
    started = time.monotonic()
    deadline = started+args.seconds-30.
    source = source_state()
    assert source["identity"] == read(args.source/"source_snapshot.json")["source_identity"]
    from bayesfilter.testing.inference_validation.posteriordb_targets import PosteriordbTarget,load_case,UPSTREAM_COMMIT
    commit = subprocess.check_output(["git","rev-parse","HEAD"],cwd=CHECKOUT,text=True).strip()
    if commit != UPSTREAM_COMMIT:
        raise ValueError("upstream reference checkout changed")
    case = load_case(CHECKOUT,args.case)
    # Check inspected inputs against the pinned Git tree, not caller metadata.
    for record in case["files"].values():
        path = Path(record["path"])
        blob = subprocess.check_output(["git","show",commit+":"+str(path.relative_to(CHECKOUT))],cwd=CHECKOUT)
        if hashlib.sha256(blob).hexdigest() != record["sha256"]:
            raise ValueError("upstream file differs from pinned tree")
    import tensorflow as tf
    from bayesfilter.inference import (HMCKernelTuningConfig,HMCCandidateExecutionConfig,HMCAcceptancePolicy,
        tune_hmc_kernel,build_retained_bound_hmc_archive_runner_from_candidate_set_result,
        load_hmc_candidate_retained_runner,SequentialNeuTraHMCConfig,HMCPosteriorAssessmentPolicy,
        HMCPrecisionPolicy,HMCPrecisionTarget,run_hmc_posterior)
    from bayesfilter.inference.hmc_precision import mean_precision
    from bayesfilter.inference.hmc_convergence import rank_normalized_split_rhat_summary
    from bayesfilter.runtime.durable_tensor_checkpoint import DurableTensorCheckpoint
    from bayesfilter.testing.inference_validation.designs import seed_for
    from bayesfilter.testing.inference_validation.engines.pipeline import check_inventory
    target = PosteriordbTarget(args.case,case["data"],jit_compile=True)
    if args.case != "sblrc-blr":
        raise ValueError("this initialization diagnostic is regression-specific")
    initialization_path = M8/"regression-data-initialization.json"
    initialization = read(initialization_path)
    if (not initialization["passed"] or initialization["reference_draws_used"]
            or initialization["target_signature"] != target.adapter_signature()
            or initialization["data_sha256"] != case["files"]["data"]["sha256"]):
        raise ValueError("initialization does not match the observed-data target")
    initial_position = tf.constant(initialization["initial_position"],tf.float64)
    initial_value, initial_score = target.log_prob_and_grad(initial_position)
    if (not bool(tf.math.is_finite(initial_value))
            or float(tf.reduce_max(tf.abs(initial_score))) > initialization["gradient_tolerance"]):
        raise ValueError("GPU target does not reproduce the checked initial score")
    seed = seed_for(2026092080,"m13-preparation-recovery",args.case,args.phase,args.replication,"tuning")
    cfg = HMCKernelTuningConfig(preset="serious",metric_evidence_policy="finite_window",metric_probe_num_results=16,preparation_max_restarts=3,use_xla=True,target_scope="inference_validation",
        seed=seed,candidate_search_bound_expansion_steps=1,bootstrap_initialization_rounds=20,
        public_timeout_budget_s=args.seconds-90.)
    execution = HMCCandidateExecutionConfig(measurement_num_results=128,verification_num_results=128,
        num_warmup_steps=8,seed=seed,use_xla=True,target_status_trace_policy="none",
        acceptance_policy=HMCAcceptancePolicy())
    write_json(output/"manifest.json",{"source":source,"runtime":runtime,"command":sys.argv,
        "plan_file":PLAN,"result_file":str(output/"assessment.json"),"case":args.case,
        "upstream_commit":commit,"inputs":case["files"],"seed":list(seed),
        "target_signature":target.adapter_signature(),"budget_seconds":args.seconds,
        "data_version":case["files"]["data"]["sha256"],"preparation":cfg.payload(),
        "reference_used_for_tuning":False,"member_selection":"first verified identity before posterior sampling",
        "initialization":initialization,"initialization_sha256":hashlib.sha256(initialization_path.read_bytes()).hexdigest(),
        "geometry_hint":None,"search_policy":"serious preparation with finite-window proposals; M9 startup/search repairs"})
    from bayesfilter.inference.hmc_preparation import prepare_operational_windowed_mass_handoff, HMCPreparationProgress, expanded_preparation_bound
    from bayesfilter.inference.hmc_candidate_set_execution import bind_hmc_candidate_set_execution_from_preparation
    from bayesfilter.inference.hmc_candidate_set_public import _search, _domain
    from bayesfilter.inference.hmc_configuration import _fixed_mass_step_upper_bound
    from bayesfilter.inference.hmc_candidate_set_tuning import _sha256
    with HMCPreparationProgress(output/"preparation", max_wall_time_seconds=args.seconds-90.) as progress:
        preparation = prepare_operational_windowed_mass_handoff(adapter=target,
            initial_position=initial_position, config=cfg, progress_callback=progress.phase)
    from bayesfilter.inference.hmc_preparation import _progress_json_value
    write_json(output/"prepared-windows.json", _progress_json_value(preparation["windowed_stage"].operational_warmup_result.public_payload()))
    epsilon = preparation["windowed_stage"].operational_warmup_result.final_kernel_state.epsilon
    upper = expanded_preparation_bound(_fixed_mass_step_upper_bound(preparation["windowed_stage"]),
        factor=cfg.step_repair_factor, steps=cfg.candidate_search_bound_expansion_steps)
    search = replace(_search(None, epsilon=epsilon, grid=(3,5,9,13,18,25)),
        refinement_rounds=2, explore_failed_intervals=True, max_wall_time_seconds=args.seconds-90.)
    execution = replace(execution, preparation_elapsed_seconds=progress.elapsed_seconds)
    binding = bind_hmc_candidate_set_execution_from_preparation(adapter=target, preparation=preparation,
        target_lineage={"model":args.case,"data":case["data"],"source":commit}, config=execution,
        source_paths=[__file__,str(Path(sys.modules[PosteriordbTarget.__module__].__file__)),
            str(initialization_path),str(M8/"initialize_regression.py")], scope_id=preparation["target_scope"],
        search_id=_sha256(search.payload())[:20],
        epsilon_domain=_domain(search,cfg.step_repair_factor,cfg.max_attempts,upper),
        repair_factor=cfg.step_repair_factor,max_repairs_per_family=cfg.max_attempts,
        preparation_bound_expansion_steps=cfg.candidate_search_bound_expansion_steps)
    write_json(output/"resolved_search.json",search.payload())
    run = tune_hmc_kernel(adapter=target, initial_position=binding.initial_active_state,
        config=search, candidate_set_adapter=binding.typed_adapter, output_dir=output/"tuning")
    result = run.result
    inventory = check_inventory(read(output/"tuning/candidate_set_result.json"))
    if inventory["failures"]:
        raise ValueError("invalid native candidate inventory")
    selected = sorted(result.verified_candidate_ids)[:1]
    summary = {"case":args.case,"phase":args.phase,"replication":args.replication,
        "tuning_completion":result.completion_status,"verified_members":len(result.verified_candidate_ids),
        "selected":selected,"inventory":inventory,"reference_used_for_tuning":False,
        "unassessed_verified_members":max(0,len(result.verified_candidate_ids)-1),
        "finding":"member_selected_for_posterior" if selected else "no_verified_members",
        "posterior_available":False,"accuracy_screen_passed":False}
    write_json(output/"selection.json",summary)
    if selected and time.monotonic() < deadline:
        binding = run.adapter._execution_binding
        member = build_retained_bound_hmc_archive_runner_from_candidate_set_result(
            candidate_set_result=result,candidate_id=selected[0],retained_binding=binding)
        member.export(output/"member.json")
        member = load_hmc_candidate_retained_runner(output/"member.json",adapter=target)
        policy = HMCPosteriorAssessmentPolicy(precision=HMCPrecisionPolicy(
            tuple(HMCPrecisionTarget(name,mcse_absolute_max=.1) for name in target.model_names),method="lugsail"))
        config = SequentialNeuTraHMCConfig(step_size=member.step_size,num_leapfrog_steps=member.num_leapfrog_steps,
            jit_compile=True,warmup_chunk_results=500,warmup_min_results=2000,warmup_check_window_results=1000,
            warmup_max_results=10000,retained_chunk_results=500,retained_min_results=1000,retained_max_results=10000,
            warmup_seed=seed_for(*seed,"warmup"),retained_seed=seed_for(*seed,"retained"),assessment_policy=policy)
        with DurableTensorCheckpoint(output/"posterior_chunks",{"member":member.member_hash,
                "policy":policy.payload(),"model":target.adapter_signature(),"parameters":target.model_names}) as store:
            posterior = run_hmc_posterior(member=member,config=config,parameter_names=target.model_names,
                model_transform=target.to_model,checkpoint_store=store,budget_check=lambda _:time.monotonic()<deadline)
        draws = posterior["private_retained_raw"]
        write_tensor(output/"retained.tensor",draws)
        write_tensor(output/"warmup.tensor",posterior["private_warmup_raw"])
        summary["posterior"] = {k:v for k,v in posterior.items() if not k.startswith("private_")}
        summary["finding"] = "posterior_unavailable"
        if int(draws.shape[0]) >= 4:
            from scipy.stats import norm
            info = case["reference_info"]
            if not all(info["checks_made"].values()) or any(info["diagnostics"]["divergent_transitions"]):
                raise ValueError("reference reports failed diagnostics")
            chains = case["reference_chains"]
            if len(chains) != 10 or any(set(c) != set(target.model_names) for c in chains):
                raise ValueError("reference quantity/chain mismatch")
            reference = tf.stack([tf.stack([tf.constant(c[name],tf.float64) for name in target.model_names],axis=-1) for c in chains],axis=1)
            if tuple(reference.shape) != (1000,10,target.parameter_dim):
                raise ValueError("reference shape mismatch")
            ref_rhat = rank_normalized_split_rhat_summary(reference,rhat_max=1.01)
            ref = mean_precision(reference,method="lugsail",jit_compile=True)
            observed = mean_precision(draws,method="lugsail",jit_compile=True)
            z = float(norm.ppf(1-.05/(2*target.parameter_dim)))
            quantities = []
            for j,name in enumerate(target.model_names):
                delta = float(observed["estimate"][j]-ref["estimate"][j])
                se = float(tf.sqrt(tf.square(observed["mcse"][j])+tf.square(ref["mcse"][j])))
                sd = float(ref["posterior_sd"][j])
                valid = bool(observed["valid"][j]) and bool(ref["valid"][j]) and math.isfinite(se)
                quantities.append({"name":name,"difference":delta,"combined_mcse":se if valid else None,
                    "reference_mean":float(ref["estimate"][j]),"estimate":float(observed["estimate"][j]),
                    "reference_mcse":float(ref["mcse"][j]),"bayesfilter_mcse":float(observed["mcse"][j]),
                    "margin":.25*sd,"simultaneous_radius":z*se if valid else None,
                    "passed":valid and abs(delta)+z*se<=.25*sd})
            summary.update(posterior_available=True,quantities=quantities,reference_rhat=ref_rhat,
                reference_metadata=info,reference_sampling_structure="ten thinned Stan chains; not exact iid",
                reference_degrees_of_freedom="finite-chain lugsail approximation",
                z=z,pointwise_alpha=.05/target.parameter_dim,
                accuracy_screen_passed=bool(posterior["passed"]) and ref_rhat["passed"] and all(q["passed"] for q in quantities),
                finding="reference_assessed",default_promoted=False,ranking_supported=False)
    write_json(output/"assessment.json",summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--case",required=True)
    parser.add_argument("--phase",choices=("pilot","fresh"),required=True)
    parser.add_argument("--replication",type=int,required=True)
    parser.add_argument("--seconds",type=float,required=True)
    parser.add_argument("--worker",action="store_true")
    args = parser.parse_args()
    if args.phase != "fresh":
        raise ValueError("combined runner is the predeclared fresh-fit arm")
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "2"
    assert os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH","").lower() == "true"
    output = ROOT/f"regression-recovery-{args.phase}-{args.replication}-gpu-r1"
    if args.worker:
        try:
            worker(args,output)
        except Exception as exc:
            (output/"failure.json").write_text(json.dumps({"exception":type(exc).__name__,"message":str(exc)},indent=2)+"\n")
            raise
        return
    costs = [read(p) for p in ROOT.glob("*/gpu-diagnostic-run.json")]
    if args.replication not in (0, 1, 2) or not 30 < args.seconds <= 1500:
        raise ValueError("unallocated replication or time cap")
    if len(costs) >= 8 or sum(r["elapsed_seconds"] for r in costs) + args.seconds > 9000.:
        raise RuntimeError("M13 attempt/worker-wall budget exhausted")
    pilot = read(ROOT/"matched-m12-fit0-r1/gpu-diagnostic-run.json")
    if pilot["returncode"] != 0:
        raise RuntimeError("matched preparation needs diagnosis before fresh fits")
    output.mkdir(exist_ok=False)
    command = [sys.executable,str(Path(__file__).resolve()),*sys.argv[1:],"--worker"]
    started = time.monotonic()
    when = datetime.now(timezone.utc).isoformat()
    with (output/"worker.log").open("x") as log:
        try:
            result = subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=args.seconds)
            code = result.returncode
        except subprocess.TimeoutExpired:
            code = 124
    record = {"command":command,"started_utc":when,"elapsed_seconds":time.monotonic()-started,
        "returncode":code,"device":"gpu","environment":sys.executable,"plan_file":PLAN,
        "result_file":str(output/"assessment.json"),"script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "git_commit":read(args.source/"source_snapshot.json")["git_commit"],
        "runtime":read(output/"runtime.json") if (output/"runtime.json").exists() else None}
    (output/"gpu-diagnostic-run.json").write_text(json.dumps(record,indent=2)+"\n")
    print(json.dumps(record,indent=2))
    raise SystemExit(code)


if __name__ == "__main__":
    main()
