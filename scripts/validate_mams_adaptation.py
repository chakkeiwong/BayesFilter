#!/usr/bin/env python3
"""Bounded D4R3 engineering/feasibility execution; no method ranking."""
from __future__ import annotations

import argparse
from dataclasses import asdict
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
PLAN = "docs/plans/bayesfilter-mams-adaptation-bounded-validation-2026-10-11.md"
TARGETS = ("gaussian", "rotated_gaussian", "student_t", "funnel_noncentered",
           "residual_funnel", "mixture", "LGSSM", "NL-ACC")


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n")
    temporary.replace(path)


def source_revision():
    provenance = ROOT / ".validation-source.json"
    if provenance.exists():
        return json.loads(provenance.read_text())
    return dict(git_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
                snapshot_kind="working_tree_with_recorded_source_hashes")


def recover_case_rows(output, result):
    """Keep all created policies in diagnostic denominators after interruption.

    This reads the durable public-tuner record for reporting only. It neither
    issues a retained member nor promotes incomplete verification evidence.
    """
    tuning = output / "search" / "tuning"
    for filename in ("tuning_checkpoint.json", "candidate_set_result.json", "controller_checkpoint.json"):
        path = tuning / filename
        if not path.exists():
            continue
        payload = json.loads(path.read_text())
        payload = payload.get("result", payload)
        existing = {row["candidate_id"]: row for row in result["rows"]}
        for candidate in payload["candidates"]:
            cid = candidate["candidate_id"]
            if cid not in existing:
                result["rows"].append(dict(candidate_id=cid,
                    tuning_state=payload["candidate_states"][cid],
                    status="posterior_not_started" if cid in payload["verified_candidate_ids"] else "not_verified"))
        result["verified_count"] = len(payload["verified_candidate_ids"])
        result["tuning_completion_status"] = payload["completion_status"]
        result["candidate_inventory_source"] = str(path)
        return


def finish_case(output, result, *, status, error=None):
    recover_case_rows(output, result)
    result["status"] = status
    if error is not None:
        result.update(error_type=type(error).__name__, error=str(error))
    for row in result["rows"]:
        if row["status"] in ("posterior_running", "posterior_not_started"):
            prefix = "posterior_interrupted" if row["status"] == "posterior_running" else "posterior_not_started"
            row["status"] = prefix + "_" + status
    pilot = output / "search" / "pilot" / "pilot-progress.json"
    if pilot.exists():
        payload = json.loads(pilot.read_text())
        result.update(pilot_status=payload["status"],
            pilot_actual_work=payload["actual_gradient_work"],
            pilot_reserved_work=payload["reserved_gradient_work"],
            pilot_hypotheses_before_qualification=payload["candidate_rows"])
    write_json(output / "result.json", result)
    return result


def finalize_interrupted_matrix(output, targets, *, status, error=None):
    """Parent-process finalization also covers SIGKILL and import failure."""
    rows = []
    for target in targets:
        directory = output / target
        terminal = directory / "result.json"
        progress = directory / "progress.json"
        if terminal.exists():
            result = json.loads(terminal.read_text())
        elif progress.exists():
            result = finish_case(directory, json.loads(progress.read_text()), status=status, error=error)
        else:
            result = dict(target=target, status="not_started_" + status, rows=[])
        rows.append(result)
    result = dict(status=status, rows=rows, ranking_supported=False, release_qualified=False)
    write_json(output / "result.json", result)
    return result


def execute_matrix(targets, output, seed, deadline, *, use_xla, source_hashes, case_runner=None):
    """Expected bounded outcomes continue; unknown/shared defects stop the matrix."""
    case_runner = run_case if case_runner is None else case_runner
    rows = [dict(target=target, status="not_started", rows=[]) for target in targets]
    write_json(output / "progress.json", dict(status="running", rows=rows))
    for index, target in enumerate(targets):
        if time.monotonic() >= deadline:
            return finalize_interrupted_matrix(output, targets, status="budget_exhausted")
        try:
            for name, digest in source_hashes.items():
                if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
                    raise ValueError("source changed during validation: " + name)
            rows[index] = case_runner(target, output / target, seed + 10000 * index, deadline,
                                     use_xla=use_xla, source_hashes=source_hashes)
        except Exception as exc:
            return finalize_interrupted_matrix(output, targets, status="repair_required", error=exc)
        write_json(output / "progress.json", dict(status="running", rows=rows))
    status = "completed" if all(row["status"] == "completed" for row in rows) else "completed_with_incomplete_targets"
    result = dict(status=status, rows=rows, ranking_supported=False, release_qualified=False)
    write_json(output / "result.json", result)
    return result


def fixture(target, use_xla):
    import tensorflow as tf
    if target in ("LGSSM", "NL-ACC"):
        from scripts.run_modern_hmc_bridge import _fixture
        f, starts, sources, metadata = _fixture(target, use_xla=use_xla)
        return f.adapter, starts, sources, metadata
    from bayesfilter.testing.inference_validation.targets import ValidationTarget
    from bayesfilter.testing.acceptance_uncertainty_models import ResidualFunnelTarget
    parameters = {"rotated_gaussian": {"condition": 25., "angle": .6},
                  "student_t": {"df": 5.}, "funnel_noncentered": {"scale": 1.},
                  "mixture": {"weight": .3, "separation": 5.}}.get(target, {})
    adapter = (ResidualFunnelTarget(jit_compile=use_xla) if target == "residual_funnel" else
               ValidationTarget(target, parameters, jit_compile=use_xla))
    d = adapter.parameter_dim
    starts = tf.constant([[-1., -.5, .4], [-.3, .2, -.6], [.4, -.2, .8], [1., .5, -.4]],
                         tf.float64)[:, :d]
    if target == "mixture":
        # Deliberately local bank: independent mode mass must expose the
        # limitation even if within-mode diagnostics look stationary.
        starts += tf.constant([5., 0.], tf.float64)
    sources = ["bayesfilter/testing/inference_validation/targets.py",
               "bayesfilter/testing/acceptance_uncertainty_models.py"]
    return adapter, starts, sources, dict(target_id=target, parameters=parameters,
        data_sha256="N/A: analytic law without observations",
        initialization="local positive-mode bank" if target == "mixture" else "fixed contrasting bank")


def run_case(target, output, seed, deadline, *, use_xla, source_hashes):
    from bayesfilter.inference.hmc_mams_trajectory_proposals import MAMSPilotBudgetExceeded, MAMSPilotUnavailable
    from bayesfilter.runtime.execution_budget import ExecutionBudgetExceeded
    output.mkdir()
    started = time.monotonic()
    result = dict(target=target, status="preparing", rows=[],
                  ranking_supported=False, release_qualified=False)
    write_json(output / "progress.json", result)
    try:
        _run_case(target, output, seed, deadline, result, use_xla=use_xla, source_hashes=source_hashes)
    except (MAMSPilotBudgetExceeded, MAMSPilotUnavailable, ExecutionBudgetExceeded) as exc:
        status = {MAMSPilotBudgetExceeded: "pilot_budget_exhausted",
                  MAMSPilotUnavailable: "pilot_unavailable",
                  ExecutionBudgetExceeded: "budget_exhausted"}[type(exc)]
        result["elapsed_seconds"] = time.monotonic() - started
        return finish_case(output, result, status=status, error=exc)
    except BaseException as exc:
        result["elapsed_seconds"] = time.monotonic() - started
        finish_case(output, result, status="repair_required", error=exc)
        raise
    result["elapsed_seconds"] = time.monotonic() - started
    return finish_case(output, result, status=result["status"])


def _run_case(target, output, seed, deadline, progress, *, use_xla, source_hashes):
    import tensorflow as tf
    from bayesfilter.inference import (HMCCandidateExecutionConfig, HMCControllerConfig,
        PrecomputedMassArtifact, bind_hmc_candidate_set_execution,
        HMCPosteriorAssessmentPolicy, HMCPrecisionPolicy, HMCPrecisionTarget,
        SequentialNeuTraHMCConfig, run_hmc_posterior)
    from bayesfilter.inference.hmc_acceptance_protocol import HMCReplicatedAcceptancePolicy
    from bayesfilter.inference.hmc_transition_spec import HMCTransitionProfile
    from bayesfilter.inference.hmc_mams_trajectory_proposals import MAMSPilotConfig
    from bayesfilter.inference.posterior_adapter import value_score_capability
    from bayesfilter.testing.inference_validation.mams_adaptation_consumer import run_adapted_mams_search
    from bayesfilter.runtime.durable_tensor_checkpoint import DurableTensorCheckpoint
    from bayesfilter.runtime.execution_budget import execution_budget, require_execution_budget

    started = time.monotonic()
    adapter, starts, sources, metadata = fixture(target, use_xla)
    d = adapter.parameter_dim
    identity = tf.eye(d, dtype=tf.float64)
    mass = PrecomputedMassArtifact(position=tf.zeros(d, tf.float64), covariance=identity,
        factor=identity, adapter_signature=adapter.adapter_signature(),
        position_role="reference_center", covariance_source="explicit supplied diagnostic identity")
    policy = HMCReplicatedAcceptancePolicy(trial_num_results=64, discarded_prefix=0,
        base_repetitions=32, max_repetitions=32, max_candidates=8,
        search_family_alpha=.05, verification_family_alpha=.05, diagnostic_family_alpha=.05,
        temporal_tolerance=.1, qualification_mode="lower_bound", qualification_region=(.5, 1.),
        min_normalized_return_displacement=0.)
    execution = HMCCandidateExecutionConfig(measurement_num_results=64, verification_num_results=64,
        num_warmup_steps=0, seed=(20261011, seed+100), acceptance_policy=policy,
        chain_mode="batched", chunk_max_results=64, use_xla=use_xla,
        non_xla_reason=None if use_xla else "explicit CPU diagnostic smoke",
        target_status_trace_policy="per_chain_step" if callable(getattr(adapter,"target_status_telemetry",None)) else "none",
        transition_profile=HMCTransitionProfile("mams", "uniform_duration"))
    binding = bind_hmc_candidate_set_execution(adapter=adapter, initial_position=starts,
        target_scope=value_score_capability(adapter).target_scope, target_lineage=metadata,
        config=execution, source_paths=list(source_hashes), scope_id="d4r3-"+target,
        search_id="seed-"+str(seed), epsilon_domain=(.001, 4.), repair_factor=1.5,
        max_repairs_per_family=1, mass_artifact=mass)
    pilot = MAMSPilotConfig(initial_epsilon=.1, initial_mean_duration=math.sqrt(d),
        epsilon_domain=binding.scope.epsilon_domain, max_steps_cap=128,
        epsilon_adaptation_windows=6, probe_transitions=4, fixed_settling_transitions=32,
        fixed_window_transitions=64, fixed_window_count=2, max_autocorrelation_lag=16,
        target_accept_probability=.9, log_epsilon_gain=math.log(2), source_length_coefficient=.3,
        window_agreement_factor=3., duration_factors=(1., 2.), epsilon_factors=(1.,),
        max_requested_duration=8*math.sqrt(d), max_candidates=4,
        max_pilot_gradient_work=2000000, max_pilot_wall_seconds=7200.,
        frozen_coordinate_scale=(1.,)*d, frozen_projection_directions=(tuple(1. for _ in range(d)),),
        seed=(20261011, seed+200))
    search = HMCControllerConfig(primary_l_grid=(3,), epsilon_by_l=((3, (.1,)),),
        total_budget_units=40, repair_reserve_units=12, max_candidates=8, evidence_rungs=(1,),
        max_gradient_work=40000000, max_wall_time_seconds=7200.,
        replicated_acceptance_policy=policy, repair_trajectory_policy="preserve_mean_duration",
        repair_mean_duration_max_steps=128)
    write_json(output/"design.json", dict(metadata=metadata, initial_state=starts.numpy().tolist(),
        target_signature=adapter.adapter_signature(), binding_hash=binding.binding_hash,
        pilot=asdict(pilot), search=search.payload(), source_hashes=source_hashes,
        scientific_promotion_authority=False))
    rows = progress["rows"]
    progress["status"] = "tuning"
    write_json(output / "progress.json", progress)
    with execution_budget(check=lambda: time.monotonic()<deadline, replace_wall_time_limit=True):
        proposals, run, members = run_adapted_mams_search(binding, pilot, search, output_dir=output/"search")
        for index, candidate in enumerate(run.result.candidates):
            row = dict(candidate_id=candidate.candidate_id, tuning_state=run.result.candidate_states[candidate.candidate_id],
                       status="posterior_not_started" if candidate.candidate_id in members else "not_verified")
            rows.append(row)
        progress.update(status="posterior", verified_count=len(members),
                        tuning_completion_status=run.result.completion_status)
        write_json(output/"progress.json", progress)
        if run.result.completion_status in ("shared_invalidity", "paused_infrastructure"):
            raise RuntimeError("public tuning stopped with " + run.result.completion_status)
        for index, row in enumerate(rows):
            if row["status"] == "not_verified":
                continue
            require_execution_budget()
            row["status"] = "posterior_running"
            write_json(output / "progress.json", progress)
            member = members[row["candidate_id"]]
            names = tuple("q"+str(i) for i in range(d))
            extra_names = tuple("q"+str(i)+"_square" for i in range(d))
            quantities = lambda x: {name: tf.square(x[...,i]) for i,name in enumerate(extra_names)}
            assessment = HMCPosteriorAssessmentPolicy(retained_bulk_ess_min=100., retained_tail_ess_min=100.,
                quantities_id="d4r3_coordinates_and_squares_v1",
                precision=HMCPrecisionPolicy(tuple(HMCPrecisionTarget(n,mcse_sd_ratio_max=.1)
                    for n in names+extra_names),method="lugsail",jit_compile=use_xla))
            config = SequentialNeuTraHMCConfig(step_size=member.step_size,
                num_leapfrog_steps=member.num_leapfrog_steps, warmup_seed=(20261011,seed+1000+index*2),
                retained_seed=(20261011,seed+1001+index*2), warmup_chunk_results=256,
                warmup_min_results=512, warmup_check_window_results=256, warmup_max_results=1024,
                retained_chunk_results=256, retained_min_results=256, retained_max_results=1024,
                assessment_policy=assessment,jit_compile=use_xla)
            destination=output/("posterior-%02d"%index);destination.mkdir()
            write_json(destination/"policy.json",config.payload(chain_count=4))
            with DurableTensorCheckpoint(destination/"chunks",dict(member_hash=member.member_hash,
                    policy=config.payload(chain_count=4))) as store:
                result=run_hmc_posterior(member=member,config=config,parameter_names=names,
                                         quantities_fn=quantities,checkpoint_store=store)
            samples=result.get("private_retained_raw")
            reference={}
            if samples is not None and int(samples.shape[0]):
                reference["observed_mean"]=tf.reduce_mean(samples,(0,1)).numpy().tolist()
                reference["observed_second_moment"]=tf.reduce_mean(tf.square(samples),(0,1)).numpy().tolist()
                if target in ("gaussian","funnel_noncentered","student_t"):
                    reference.update(exact_mean=[0.]*d,exact_second_moment=[5/3 if target=="student_t" else 1.]*d)
                if target=="rotated_gaussian":
                    c, s = math.cos(.6), math.sin(.6)
                    reference.update(exact_mean=[0.,0.],exact_second_moment=[c*c+25*s*s,s*s+25*c*c])
                if target=="mixture":
                    reference.update(exact_mean=[2.,0.],exact_second_moment=[26.,1.],
                                     observed_left_mass=float(tf.reduce_mean(tf.cast(samples[...,0]<0,tf.float64))),
                                     exact_left_mass=.3*.5*(1+math.erf(5/math.sqrt(2)))+.7*.5*math.erfc(5/math.sqrt(2)))
                tf.io.write_file(str(destination/"retained.tensor"),tf.io.serialize_tensor(samples))
                reference["quantity_coordinates"]="supplied adapter coordinates; not centered-funnel model quantities"
            public={k:v for k,v in result.items() if not k.startswith("private_")}
            write_json(destination/"result.json",public)
            row.update(status="diagnostic_posterior_checks_passed" if result["passed"] else "posterior_incomplete",
                       reference=reference,reference_interpretation="descriptive; no calibrated comparison")
            progress["elapsed_seconds"] = time.monotonic()-started
            write_json(output/"progress.json",progress)
    progress.update(status="completed" if run.result.completion_status == "complete" else "qualification_incomplete",
                    pilot_actual_work=proposals["actual_gradient_work"])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--targets",nargs="+",choices=TARGETS,required=True)
    parser.add_argument("--seed",type=int,required=True)
    parser.add_argument("--max-wall-seconds",type=float,required=True)
    parser.add_argument("--cpu-reference",action="store_true")
    parser.add_argument("--worker",action="store_true",help=argparse.SUPPRESS)
    args=parser.parse_args()
    if not math.isfinite(args.max_wall_seconds) or not 0<args.max_wall_seconds<=7200:
        raise ValueError("stage allocation must lie inside (0,7200] seconds")
    if len(set(args.targets))!=len(args.targets):
        raise ValueError("duplicate targets")
    if os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH","").lower()!="true":
        raise ValueError("memory growth must be configured before import")
    if args.cpu_reference and os.environ.get("CUDA_VISIBLE_DEVICES")!="-1":
        raise ValueError("CPU reference must intentionally hide GPUs")
    if not args.worker:
        return supervise(args)
    args.output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();deadline=started+args.max_wall_seconds
    manifest=dict(command=[sys.executable,*sys.argv],plan=PLAN,environment=sys.executable,
        **source_revision(),
        seed=args.seed,targets=args.targets,started_utc=datetime.now(timezone.utc).isoformat(),
        max_wall_seconds=args.max_wall_seconds,gpu_intentionally_hidden=args.cpu_reference,
        cpu_gpu_scope="CPU diagnostic" if args.cpu_reference else "trusted GPU/XLA diagnostic",
        status="starting",result_file=str(args.output/"result.json"))
    write_json(args.output/"manifest.json",manifest)
    try:
        import tensorflow as tf
        import tensorflow_probability as tfp
        from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
        memory=configure_tensorflow_gpu_memory_growth(tf,require_gpu=not args.cpu_reference)
        from bayesfilter.testing.inference_validation.modern_bridge_audit import source_dependencies
        entries=[__file__,ROOT/"scripts/run_modern_hmc_bridge.py",
                 ROOT/"bayesfilter/testing/simple_nonlinear_generic_target_adapter_tf.py",
                 ROOT/"bayesfilter/testing/acceptance_uncertainty_models.py"]
        paths=source_dependencies(entries,repository_root=ROOT)
        hashes={}
        for name in paths:
            path=Path(name);data=path.read_bytes();hashes[name]=hashlib.sha256(data).hexdigest()
            copy=args.output/"source"/path.relative_to(ROOT);copy.parent.mkdir(parents=True,exist_ok=True);copy.write_bytes(data)
        operators = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in (ROOT/"bayesfilter/ops").glob("*.so")}
        manifest.update(status="running",memory_policy=memory,source_hashes=hashes,
            compiled_operator_sha256=operators,tfp_version=tfp.__version__,
            framework_version=tf.__version__,python_version=sys.version,use_xla=not args.cpu_reference,
            cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
            tf32_enabled=tf.config.experimental.tensor_float_32_execution_enabled())
        write_json(args.output/"manifest.json",manifest)
        result=execute_matrix(args.targets,args.output,args.seed,deadline,
                              use_xla=not args.cpu_reference,source_hashes=hashes)
        manifest["status"]=result["status"]
    except BaseException as exc:
        manifest.update(status="repair_required",error_type=type(exc).__name__,error=str(exc))
        finalize_interrupted_matrix(args.output,args.targets,status="repair_required",error=exc)
        raise
    finally:
        manifest.update(elapsed_seconds=time.monotonic()-started,finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(args.output/"manifest.json",manifest)
    print(json.dumps(dict(status=manifest["status"],elapsed_seconds=manifest["elapsed_seconds"],output=str(args.output))))
    return 1 if manifest["status"] == "repair_required" else 0


def supervise(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    started = time.monotonic()
    # Grace is contained in, not added to, the declared aggregate allocation.
    grace = min(60., args.max_wall_seconds / 10.)
    command = [sys.executable, str(Path(__file__).resolve()), "--worker",
        "--output", str(args.output.resolve()), "--targets", *args.targets,
        "--seed", str(args.seed), "--max-wall-seconds", str(args.max_wall_seconds-grace)]
    if args.cpu_reference:
        command.append("--cpu-reference")
    with subprocess.Popen(command, cwd=ROOT, start_new_session=True) as worker:
        try:
            code = worker.wait(timeout=args.max_wall_seconds)
        except subprocess.TimeoutExpired as exc:
            os.killpg(worker.pid, signal.SIGKILL)
            code = worker.wait()
            args.output.mkdir(parents=True,exist_ok=True)
            finalize_interrupted_matrix(args.output,args.targets,status="supervisor_budget_exhausted",error=exc)
        except BaseException:
            os.killpg(worker.pid, signal.SIGKILL)
            worker.wait()
            raise
    args.output.mkdir(parents=True,exist_ok=True)
    if code and not (args.output/"result.json").exists():
        finalize_interrupted_matrix(args.output,args.targets,status="worker_failed")
    write_json(args.output/"supervisor.json",dict(command=command,exit_code=code,
        max_wall_seconds=args.max_wall_seconds,cooperative_grace_seconds=grace,
        elapsed_seconds=time.monotonic()-started))
    return code


if __name__=="__main__":
    sys.exit(main())
