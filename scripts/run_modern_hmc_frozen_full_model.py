#!/usr/bin/env python3
"""Bounded full-model qualification/posterior consumer of an explicit design.

No default grid, automatic method ranking or adaptation claim is supplied.
Use a reviewed plan and a new output directory for each design execution.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bayesfilter.testing.inference_validation.modern_full_model_consumer import write_json


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_execution_mode(execution, *, cpu_reference):
    if not cpu_reference and not execution.use_xla:
        raise ValueError("GPU campaign execution requires XLA")


def posterior_factory(settings, *, target, use_xla):
    from bayesfilter.inference import (HMCPosteriorAssessmentPolicy, HMCPrecisionPolicy,
        HMCPrecisionTarget, SequentialNeuTraHMCConfig)
    from scripts.audit_mams_reference_compatibility import quantity_names
    from bayesfilter.testing.inference_validation.modern_full_model_consumer import validate_reference_controls
    validate_reference_controls(settings["reference_mcse_multiplier"],
                                settings["reference_normalized_tolerance"])
    required = {"warmup_chunk_results", "warmup_min_results", "warmup_check_window_results",
        "warmup_max_results", "warmup_rhat_max", "retained_chunk_results",
        "retained_min_results", "retained_max_results", "retained_rhat_max",
        "minimum_chain_count", "energy_error_log_accept_threshold",
        "max_results_per_chain", "count_budget_reason"}
    if not required <= set(settings["sequential"]):
        raise ValueError("declare all material sequential settings explicitly")
    forbidden = {"step_size", "num_leapfrog_steps", "warmup_seed", "retained_seed",
                 "assessment_policy", "jit_compile"}
    if forbidden & set(settings["sequential"]):
        raise ValueError("posterior design cannot override member settings or derived seeds")
    if set(settings["precision"]) != {"method", "batch_size", "min_batches", "lugsail_r", "lugsail_c"}:
        raise ValueError("declare all precision estimator settings explicitly")
    if set(settings["assessment"]) != {"warmup_bulk_ess_min", "warmup_tail_ess_min",
            "retained_bulk_ess_min", "retained_tail_ess_min", "warmup_consecutive_checks"}:
        raise ValueError("declare all posterior assessment settings explicitly")
    if settings["sequential"]["minimum_chain_count"] != 4:
        raise ValueError("full-model start bank contains exactly four chains")
    precision = HMCPrecisionPolicy(tuple(HMCPrecisionTarget(name,
        mcse_sd_ratio_max=settings["mcse_sd_ratio_max"]) for name in quantity_names(target)),
        **settings["precision"], jit_compile=use_xla)
    assessment = HMCPosteriorAssessmentPolicy(**settings["assessment"], precision=precision)
    first, second = settings["seed"]
    def factory(member, index):
        return SequentialNeuTraHMCConfig(step_size=member.step_size,
            num_leapfrog_steps=member.num_leapfrog_steps,
            warmup_seed=(first, second+2*index), retained_seed=(first, second+2*index+1),
            assessment_policy=assessment, jit_compile=use_xla, **settings["sequential"])
    return factory


def execution_source_paths():
    """The static library scanner does not traverse imports from scripts."""
    from scripts import audit_mams_reference_compatibility, validate_mams_adaptation
    return (__file__, audit_mams_reference_compatibility.__file__, validate_mams_adaptation.__file__)


def finalize_interruption(output, design, *, status):
    """Recover existing posterior dispositions before marking unfinished work."""
    from scripts.validate_mams_adaptation import finish_case
    posterior = output/"posterior/result.json"
    search = output/"search/result.json"
    result = read(search) if search.exists() else dict(rows=[])
    if posterior.exists():
        assessment = read(posterior)
        # Merge posterior dispositions by identity without losing tuning
        # parameters, repair ancestry or uncreated primary pairs.
        prior = {row["candidate_id"]: row for row in result["rows"]}
        result["rows"] = [dict(prior.get(row["candidate_id"], {}), **row)
                          for row in assessment["rows"]]
        result["posterior"] = assessment
    result.update(target=design["target"], method=design["method"],
        requested_primary_pairs=[(length,epsilon) for length,values in design["search"]["epsilon_by_l"]
                                 for epsilon in values])
    result = finish_case(output, result, status=status)
    inventory_path = result.get("candidate_inventory_source")
    if inventory_path is not None:
        checkpoint = read(inventory_path)
        checkpoint = checkpoint.get("result", checkpoint)
        created = {candidate["candidate_id"]: candidate for candidate in checkpoint["candidates"]}
        for row in result["rows"]:
            candidate = created[row["candidate_id"]]
            row.update({key: candidate[key] for key in (
                "parent_candidate_id", "candidate_family_id", "epsilon", "leapfrog_steps",
                "transition_spec", "creation_ordinal")})
        roots = {(candidate["leapfrog_steps"], candidate["epsilon"]) for candidate in created.values()
                 if candidate["parent_candidate_id"] is None}
        result.update(candidate_count=len(created), created_primary_count=len(roots),
            repair_descendant_count=len(created)-len(roots),
            verified_candidate_ids=checkpoint["verified_candidate_ids"],
            repair_actions=checkpoint["repair_actions"],
            uncreated_primary_pairs=[pair for pair in result["requested_primary_pairs"] if tuple(pair) not in roots])
    result["delivered_count"] = sum(row.get("delivered", False) for row in result["rows"])
    if posterior.exists():
        assessment.update(status=status, rows=result["rows"], delivered_count=result["delivered_count"])
        write_json(posterior, assessment)
        result["posterior"] = assessment
    write_json(output/"result.json", result)
    return result


def worker(args):
    import tensorflow as tf
    import tensorflow_probability as tfp
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    from bayesfilter.runtime.execution_budget import execution_budget, ExecutionBudgetExceeded
    from bayesfilter.inference import HMCCandidateExecutionConfig, HMCControllerConfig
    from bayesfilter.testing.inference_validation.modern_full_model_consumer import (
        full_model_binding, assess_full_model_members)
    from bayesfilter.testing.inference_validation.modern_tuning_consumer import (
        run_frozen_trajectory_search, frozen_search_evidence_preflight, require_attainable_qualification)
    from scripts.audit_mams_reference_compatibility import load_compatible_reference
    from scripts.validate_mams_adaptation import source_revision
    design = read(args.design)
    if design.get("schema") != "bayesfilter.modern_frozen_full_model_design.v1":
        raise ValueError("unsupported frozen full-model design")
    execution = HMCCandidateExecutionConfig.from_payload(design["execution"])
    search = HMCControllerConfig.from_payload(design["search"])
    validate_execution_mode(execution, cpu_reference=args.cpu_reference)
    # Validate posterior controls before spending the qualification allocation.
    factory = None
    if design.get("posterior") is not None:
        from types import SimpleNamespace
        factory=posterior_factory(design["posterior"],target=design["target"],use_xla=execution.use_xla)
        length,values=search.epsilon_by_l[0]
        factory(SimpleNamespace(step_size=values[0],num_leapfrog_steps=length),search.max_candidates-1)
        if not Path(design["reference_compatibility"]).is_file():
            raise ValueError("posterior execution requires an existing reference compatibility report")
    memory = configure_tensorflow_gpu_memory_growth(tf, require_gpu=not args.cpu_reference)
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    manifest = dict(status="preparing", **source_revision(), command=[sys.executable,*sys.argv],
        started_utc=datetime.now(timezone.utc).isoformat(), plan=design["plan"],
        design=design, design_file=str(args.design), design_sha256=digest(args.design),
        environment=sys.executable, tf_version=tf.__version__, tfp_version=tfp.__version__,
        cpu_reference=args.cpu_reference, gpu_intentionally_hidden=args.cpu_reference,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), memory_policy=memory,
        use_xla=execution.use_xla, tf32_enabled=tf.config.experimental.tensor_float_32_execution_enabled(),
        max_wall_seconds=args.max_wall_seconds, method_ranking=None, release_qualified=False)
    write_json(args.output/"manifest.json",manifest)
    try:
        preflight = frozen_search_evidence_preflight(execution, search)
        write_json(args.output/"qualification-preflight.json", preflight)
        manifest["qualification_preflight"] = preflight
        require_attainable_qualification(preflight)
        binding, fixture, metadata = full_model_binding(target=design["target"],
            geometry_directory=design["geometry_directory"], method=design["method"], seed=design["seed"],
            execution=execution, epsilon_domain=tuple(design["epsilon_domain"]),
            repair_factor=design["repair_factor"], max_repairs_per_family=design["max_repairs_per_family"],
            source_paths=execution_source_paths())
        inputs = {str(args.design):manifest["design_sha256"], **binding._spec["source_closure"]}
        for relative in ("bayesfilter/ops/_symmetric_sylvester_ops.so",
                         "bayesfilter/ops/symmetric_sylvester_op.cc"):
            path = ROOT/relative
            inputs[str(path)] = digest(path)
        for name in ("design.json", "geometry.json"):
            path=Path(design["geometry_directory"])/name
            inputs[str(path)]=digest(path)
        reference = None
        if design.get("posterior") is not None:
            reference = load_compatible_reference(design["reference_compatibility"], current_source=ROOT,
                target=design["target"], data_sha256=metadata["data_sha256"],
                data=fixture.observations.numpy().tolist())
            inputs.update(reference["reference_files_sha256"])
            inputs[design["reference_compatibility"]]=digest(design["reference_compatibility"])
        def check_inputs():
            for name, expected in inputs.items():
                if digest(name) != expected:
                    raise ValueError("execution input/source changed: "+name)
            binding.validate()
        manifest.update(status="qualifying", source_and_input_sha256=inputs,
            target=dict(data_sha256=metadata["data_sha256"], horizon=metadata["horizon"],
                        parameter_dim=metadata["parameter_dim"], geometry_sha256=metadata["geometry_sha256"],
                        global_coverage_established=False), binding_hash=binding.binding_hash,
            runtime=binding._runtime, reference=reference)
        write_json(args.output/"manifest.json",manifest)
        with execution_budget(check=lambda: time.monotonic()-started < args.max_wall_seconds,
                              replace_wall_time_limit=True):
            check_inputs()
            run, members=run_frozen_trajectory_search(binding, search, output_dir=args.output/"search",
                proposal_metadata=design["proposal_metadata"])
            manifest.update(tuning_completion_status=run.result.completion_status,
                            verified_candidate_ids=list(run.result.verified_candidate_ids))
            if run.result.completion_status in ("shared_invalidity", "paused_infrastructure"):
                raise RuntimeError("public qualification stopped with "+run.result.completion_status)
            posterior = None
            if reference is not None:
                manifest["status"]="posterior"
                write_json(args.output/"manifest.json",manifest)
                settings=design["posterior"]
                posterior=assess_full_model_members(run=run, members=members,
                    output_dir=args.output/"posterior", target=design["target"], reference=reference,
                    posterior_config_factory=factory,
                    mcse_multiplier=settings["reference_mcse_multiplier"],
                    normalized_tolerance=settings["reference_normalized_tolerance"],
                    use_xla=execution.use_xla,check_inputs=check_inputs)
            check_inputs()
        manifest["status"]=("resource_stopped" if run.result.completion_status=="partial_budget"
            or posterior is not None and posterior["status"]=="resource_stopped" else "completed")
        summary=read(args.output/"search/result.json")
        summary.update(target=design["target"],method=design["method"],posterior=posterior,
            status=manifest["status"],
            posterior_assessment="not_run" if posterior is None else posterior["status"])
        write_json(args.output/"result.json",summary)
    except ExecutionBudgetExceeded:
        manifest["status"] = "resource_stopped"
        finalize_interruption(args.output, design, status=manifest["status"])
    except BaseException as exc:
        manifest.update(status="repair_required",error_type=type(exc).__name__,error=str(exc))
        finalize_interruption(args.output, design, status=manifest["status"])
        raise
    finally:
        manifest.update(elapsed_seconds=time.monotonic()-started,finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(args.output/"manifest.json",manifest)
    return 0 if manifest["status"]=="completed" else 1


def supervise(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    command=[sys.executable,str(Path(__file__).resolve()),"--worker","--design",str(args.design),
        "--output",str(args.output),"--max-wall-seconds",str(args.max_wall_seconds)]
    if args.cpu_reference:
        command.append("--cpu-reference")
    started=time.monotonic()
    interrupted = None
    with subprocess.Popen(command,cwd=ROOT,start_new_session=True) as process:
        try:
            code=process.wait(timeout=args.max_wall_seconds)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGKILL)
            code=process.wait()
            interrupted = "supervisor_budget_exhausted"
        except BaseException:
            os.killpg(process.pid,signal.SIGKILL)
            code=process.wait()
            interrupted = "supervisor_interrupted"
    args.output.mkdir(parents=True,exist_ok=True)
    manifest_path = args.output/"manifest.json"
    manifest = read(manifest_path) if manifest_path.exists() else {}
    if interrupted is not None or not (args.output/"result.json").exists():
        status = interrupted or manifest.get("status", "worker_failed_before_manifest")
        if status not in {"repair_required", "resource_stopped"} and interrupted is None:
            status = "worker_interrupted"
        finalize_interruption(args.output,read(args.design),status=status)
        code = code or 1
        manifest.update(status=status, finished_utc=datetime.now(timezone.utc).isoformat(),
                        elapsed_seconds=time.monotonic()-started)
        write_json(manifest_path, manifest)
    write_json(args.output/"supervisor.json",dict(command=command,exit_code=code,
        elapsed_seconds=time.monotonic()-started,max_wall_seconds=args.max_wall_seconds,
        status=manifest.get("status", "worker_interrupted")))
    return code


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--max-wall-seconds",type=float,required=True)
    parser.add_argument("--cpu-reference",action="store_true")
    parser.add_argument("--worker",action="store_true",help=argparse.SUPPRESS)
    args=parser.parse_args()
    args.design,args.output=args.design.resolve(),args.output.resolve()
    if not math.isfinite(args.max_wall_seconds) or args.max_wall_seconds <= 0:
        raise ValueError("finite positive supervisor allocation required")
    if os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH", "").lower()!="true":
        raise ValueError("configure memory growth before framework import")
    if args.cpu_reference and os.environ.get("CUDA_VISIBLE_DEVICES")!="-1":
        raise ValueError("CPU reference must hide GPU devices")
    return worker(args) if args.worker else supervise(args)


if __name__=="__main__":
    sys.exit(main())
