"""Bounded local diagnostic jobs calling existing inference procedures.

The coordinator owns accounting. Every worker establishes device policy before
framework import. Native numerical artifacts and failed attempts are preserved.
"""
from __future__ import annotations
from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

from .catalog import get_target
from .designs import ValidationDesign,digest,resolve_suite
from .storage import read_json,write_json,file_hash

REPO=Path(__file__).resolve().parents[3]


def source_state():
    paths=list((REPO/"bayesfilter").rglob("*.py"))
    sources={str(p.relative_to(REPO)):file_hash(p) for p in sorted(paths)}
    snapshot=REPO/"source_snapshot.json"
    commit=(read_json(snapshot)["git_commit"] if snapshot.exists() else
            subprocess.run(["git","rev-parse","HEAD"],cwd=REPO,capture_output=True,text=True,check=True).stdout.strip())
    return {"commit":commit,
            "files":sources,"identity":digest(sources)}


def plan_suite(suite):
    designs=resolve_suite(suite)
    from .aggregation import validate_groups
    validate_groups(suite)
    jobs=[]
    for design in designs:
        spec=get_target(design.scenario.target)
        available=spec.available
        reason=spec.unavailable_reason
        if (available and not spec.pipeline_available
                and design.engine in {"search", "accuracy", "stopping", "sbc"}
                and design.scenario.route in {"ordinary", "prepared", "fixed_transport"}):
            available = False
            reason = spec.pipeline_unavailable_reason or "target lacks a shared posterior pipeline adapter"
        if design.scenario.route=="external":
            paths=[design.options.get(k) for k in ("external_bundle", "observations")]
            available=all(p and Path(p).is_file() for p in paths)
            reason=None if available else "external observation and reference bundles required"
        jobs.append({"design":design.payload(),"identity":design.identity,"coverage":design.coverage_key(),
                     "availability":"ready" if available else "unavailable","reason":reason})
    return {"schema":"bayesfilter.inference_validation_plan.v1","suite_id":suite["suite_id"],
            "suite_identity":digest(suite),"profile":suite["profile"],"jobs":jobs,
            # Inferred requirements describe every planned cell. An explicit
            # category requirement may intentionally ask for any matching cell.
            "required_coverage":suite.get("required_coverage",[
                {**j["coverage"], "design_id": j["design"]["design_id"]} for j in jobs]),
            "aggregate_groups":suite.get("aggregate_groups",[]),
            "maximum_worker_seconds":sum(d.budget_seconds for d in designs),
            "budget_by_device":{device:sum(d.budget_seconds for d in designs if d.device==device)
                                for device in ("gpu", "cpu_reference")},
            "no_numerical_execution":True}


def configure_worker(design):
    if "tensorflow" in sys.modules:
        raise RuntimeError("worker must configure device before TensorFlow import")
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]="true"
    if design.device=="cpu_reference": os.environ["CUDA_VISIBLE_DEVICES"]="-1"
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL","2")
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS","2")
    os.environ.setdefault("TF_NUM_INTEROP_THREADS","1")
    import tensorflow as tf
    import tensorflow_probability as tfp
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    memory=configure_tensorflow_gpu_memory_growth(tf,require_gpu=design.device=="gpu")
    placement = {}
    if design.device == "gpu":
        physical = tf.config.list_physical_devices("GPU")
        with tf.device("/GPU:0"):
            probe = tf.constant([1.], tf.float64) + tf.constant([2.], tf.float64)
        if "GPU" not in probe.device or float(probe[0]) != 3.:
            raise RuntimeError("GPU worker placement probe failed")
        placement = {"gpu_tensor_device": probe.device,
            "physical_gpu_details": [{"name": device.name,
                "details": tf.config.experimental.get_device_details(device)} for device in physical]}
    return {"python":platform.python_version(),"tensorflow":tf.__version__,"tfp":tfp.__version__,
            "device_scope":design.device,"gpu_intentionally_hidden":design.device=="cpu_reference",
            "memory_policy":memory,"jit_compile":design.device=="gpu",
            "tf32":tf.config.experimental.tensor_float_32_execution_enabled(),
            "cpu_threads":{"intra_op":os.environ["TF_NUM_INTRAOP_THREADS"],"inter_op":os.environ["TF_NUM_INTEROP_THREADS"]},
            "visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES"), **placement}


def worker(design_file,root,budget,attempt=1):
    root=Path(root); design=ValidationDesign.from_payload(read_json(design_file))
    started=time.monotonic()
    profile = None
    try:
        isolated = design.options.get("isolate_fits", False)
        runtime = ({"device_scope": design.device,
                    "framework_initialization": "isolated fit children only",
                    "child_manifests": "replication-*/process-attempt-*-manifest.json"}
                   if isolated else configure_worker(design))
        manifest={"design":design.payload(),"runtime":runtime,
            "source":source_state(),"command":sys.argv,"started_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
            "environment":sys.executable,"data_version":design.options.get("data_version", "declared synthetic law"),
            "plan_file":design.options.get("plan_file","docs/plans/bayesfilter-inference-validation-execution-2026-09-15.md"),
            "result_file":str(root/f"attempt-{attempt:03d}-result.json")}
        write_json(root/f"attempt-{attempt:03d}-manifest.json",manifest)
        deadline=started+budget
        if design.options.get("profile_execution", False):
            import cProfile
            profile = cProfile.Profile()
            profile.enable()
        if isolated:
            from .fit_process import run_isolated_replications
            assessment = run_isolated_replications(design, root, deadline=deadline)
        elif design.scenario.route=="external":
            from .references.external import load_reference
            from .engines.statistics import accuracy_assessment
            observed=read_json(design.options["observations"])
            reference,bundle=load_reference(design.options["external_bundle"],target_identity=observed["target_identity"],quantity_names=observed["quantity_names"])
            assessment=accuracy_assessment(observed["draws"],reference,tolerance=design.accuracy_tolerance,
                reference_iid=bundle.get("sampling_structure")=="iid",
                finite_variance=bundle.get("finite_variance") is True)
            assessment.update(reference_metadata=bundle,tested_procedure="external_recorded_observations")
        elif design.engine=="mechanics":
            from .engines.mechanics import run
            assessment=run(design,root,deadline)
        elif design.engine=="invariance":
            from .engines.invariance import run
            assessment=run(design,root,deadline)
        elif design.engine=="sbc":
            from .engines.sbc import run
            assessment=run(design,root,deadline)
        elif design.engine=="power":
            from .engines.power import run
            assessment=run(design,root,deadline)
        elif design.engine=="acceptance":
            from .engines.acceptance import run
            assessment=run(design,root,deadline)
        elif design.engine=="stopping" and design.scenario.route=="reference":
            from .engines.diagnostics import run
            assessment=run(design,root,deadline)
        else:
            from .engines.pipeline import run
            assessment=run(design,root,deadline)
        result={"schema":"bayesfilter.inference_validation_result.v1","design_identity":design.identity,
                "execution_status":("failed" if isolated and assessment.get("execution_failures") else "complete"),
                "assessment":assessment,"runtime":runtime,
                "elapsed_seconds":time.monotonic()-started,"coverage":design.coverage_key()}
        from .controls import response
        result["test_response"]=response(design,assessment)
        write_json(root/f"attempt-{attempt:03d}-result.json",result)
        write_json(root/"result.json",result)
        return 1 if isolated and assessment.get("execution_failures") else 0
    except Exception as exc:
        write_json(root/f"attempt-{attempt:03d}-failure.json",{"execution_status":"failed","exception":type(exc).__name__,
            "reason":str(exc),"traceback":traceback.format_exc(),"elapsed_seconds":time.monotonic()-started})
        traceback.print_exc()
        return 1
    finally:
        if profile is not None:
            profile.disable()
            try:
                profile.dump_stats(str(root/f"attempt-{attempt:03d}-host.prof"))
            except OSError as exc:
                # Profiling is explanatory. Preserve the engine outcome and its
                # failure record when a profile cannot be written.
                import warnings
                warnings.warn(f"host profile unavailable: {exc}", RuntimeWarning)


def run_suite(suite,root,*,resume=False,max_jobs=None,max_workers=1):
    """One ordinary advisory lock prevents accidental concurrent budget writers."""
    import fcntl
    root=Path(root).resolve()
    if root.exists() and not resume:
        raise ValueError("fresh output root required; use --resume for the identical design")
    if resume and not (root/"run_index.json").is_file():
        raise ValueError("resume requires an existing run index")
    if max_jobs is not None and (type(max_jobs) is not int or max_jobs < 1):
        raise ValueError("max_jobs must be positive")
    if type(max_workers) is not int or not 1 <= max_workers <= 32:
        raise ValueError("max_workers must be an integer in [1,32]")
    root.mkdir(parents=True,exist_ok=True)
    with (root/".coordinator.lock").open("a+b") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _run_suite(suite,root,resume=resume,max_jobs=max_jobs,max_workers=max_workers)


def _run_suite(suite,root,*,resume=False,max_jobs=None,max_workers=1):
    root=Path(root).resolve(); plan=plan_suite(suite); sources=source_state()
    index_path=root/"run_index.json"
    if index_path.exists():
        index=read_json(index_path)
        if index["suite_identity"]!=plan["suite_identity"] or index["source"]["identity"]!=sources["identity"]:
            raise ValueError("resume requires identical design and numerical source state")
    else:
        index={"schema":"bayesfilter.inference_validation_run.v1","suite_identity":plan["suite_identity"],
               "source":sources,"plan":plan,"jobs":{},"command":sys.argv}
    write_json(root/"resolved_plan.json",plan)
    pending=[]
    for job in plan["jobs"]:
        d=ValidationDesign.from_payload(job["design"]); key=d.design_id
        prior=index["jobs"].get(key,{"attempts":[]})
        if prior.get("status")=="complete": continue
        if prior.get("status")=="running":
            # The previous coordinator died without recording worker exit. We
            # cannot assume unused time or safely launch a duplicate worker.
            prior={**prior,"status":"interrupted","attempts":[*prior["attempts"],
                {"status":"coordinator_interrupted","elapsed_seconds":prior["reserved_seconds"],
                 "accounting":"full reservation charged; worker completion unknown"}]}
        if job["availability"]!="ready":
            index["jobs"][key]={**prior,"status":"unavailable","reason":job["reason"]}; continue
        if max_jobs is not None and len(pending)>=max_jobs: break
        consumed=sum(a["elapsed_seconds"] for a in prior["attempts"])
        remaining=d.budget_seconds-consumed
        if remaining<=0:
            index["jobs"][key]={**prior,"status":"unfunded","reason":"design budget exhausted"}; continue
        job_root=root/key; job_root.mkdir(exist_ok=True)
        design_path=job_root/"design.json"; write_json(design_path,d.payload())
        attempt=len(prior["attempts"])+1; log=job_root/f"attempt-{attempt:03d}.log"
        command=[sys.executable,"-m","bayesfilter.testing.inference_validation","_worker",str(design_path),str(job_root),str(remaining),str(attempt)]
        pending.append((d,prior,remaining,attempt,log,command,job_root))
    index["max_workers"]=max_workers
    # Only this coordinator writes the index. Threads supervise independent
    # processes; TensorFlow is initialized exclusively in those processes.
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        running={}
        cursor=0
        while cursor<len(pending) or running:
            while cursor<len(pending) and len(running)<max_workers:
                item=pending[cursor]; cursor+=1
                d,prior,remaining,attempt,log,command,job_root=item
                index["jobs"][d.design_id]={**prior,"status":"running","started_at":time.time(),
                    "log":str(log),"reserved_seconds":remaining}
                write_json(index_path,index)
                print(f"running {d.design_id}: {d.engine}/{d.scenario.target}/{d.scenario.route}",flush=True)
                running[pool.submit(_execute_job,d,command,log,remaining,attempt)]=item
            done,_=wait(running,return_when=FIRST_COMPLETED)
            for future in done:
                d,prior,remaining,attempt,log,command,job_root=running.pop(future)
                record=future.result()
                result_path=job_root/f"attempt-{attempt:03d}-result.json"
                index["jobs"][d.design_id]={"status":record["status"],"attempts":[*prior["attempts"],record],
                    "result":str(result_path) if record["exit_code"]==0 else None}
                if record["exit_code"]==0:
                    index["jobs"][d.design_id]["result_sha256"]=file_hash(result_path)
                write_json(index_path,index)
                print(f"{d.design_id}: {record['status']} ({record['elapsed_seconds']:.2f}s)",flush=True)
    write_json(index_path,index)
    from .reporting import report
    report(root)
    if plan["aggregate_groups"]:
        from .aggregation import aggregate_groups
        aggregate_groups(root)
    return index


def _execute_job(design,command,log,remaining,attempt):
    start=time.monotonic()
    with log.open("w") as handle:
        env=dict(os.environ, TF_FORCE_GPU_ALLOW_GROWTH="true", BAYESFILTER_PRELOAD_CUSTOM_OP="0")
        if design.device=="cpu_reference": env["CUDA_VISIBLE_DEVICES"]="-1"
        process=subprocess.Popen(command,cwd=REPO,stdout=handle,stderr=subprocess.STDOUT,
                                 env=env,start_new_session=True)
        try:
            code=process.wait(timeout=remaining)
            status="complete" if code==0 else "failed"
        except subprocess.TimeoutExpired:
            import signal
            os.killpg(process.pid,signal.SIGTERM)
            try: process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL); process.wait()
            code=None; status="timed_out"
    return {"attempt":attempt,"elapsed_seconds":time.monotonic()-start,"exit_code":code,
            "log":str(log),"status":status,"command":command}
