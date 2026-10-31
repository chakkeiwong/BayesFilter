"""Outcome-selected numerical diagnostics of the existing shared GPU consumer."""
from __future__ import annotations
import argparse
import bisect
import csv
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = "docs/plans/artifacts/iapf-tf32-tail-replay-20260923-01"
PREVIOUS = "docs/plans/artifacts/iapf-tf32-revalidation-20260923-01"
PLAN = "docs/plans/iapf-tf32-tail-replay-2026-09-23.md"
SCRIPT = "docs/benchmarks/diagnose_iapf_tf32_tail_replay.py"
PYTHON = "/home/chakwong/anaconda3/envs/tftwogpu/bin/python"
GPU = "GPU-68251639-fe82-8f81-3ccc-2953c32e805b"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def worker(mode, selection, output, warmup_original=False):
    import tensorflow as tf
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    policy = configure_tensorflow_gpu_memory_growth(tf)
    tf.config.set_soft_device_placement(False)
    tf.config.experimental.enable_tensor_float_32_execution(mode == "tf32")
    assert tf.config.experimental.tensor_float_32_execution_enabled() == (mode == "tf32")
    from bayesfilter.score_study.fitted_twist_tf import make_fitted_twist_kernel
    dtype = tf.float64 if mode == "fp64" else tf.float32
    design = json.loads(selection.read_text())
    fixture = Path(design["fixture"])
    output.mkdir()
    kernels, compilation, checks = {}, [], []

    def read(path, shape, common_fp32=True):
        values = [float(row[0]) for row in csv.reader(path.open())]
        return tf.reshape(tf.cast(tf.constant(values, tf.float32 if common_fp32 else dtype), dtype), shape)

    if warmup_original:
        # Diagnostic control: reproduce phase18's first six compilation calls.
        # This changes no model, guide, particle stream, or numerical setting.
        with tf.device("/GPU:0"):
            for d in (1,2,5):
                directory=fixture/"old"/f"d{d}"
                arrays=[read(directory/f"{name}.csv",shape,False) for name,shape in (
                    ("theta",[6]),("observations",[5,d]),("initial",[64,d]),
                    ("noise",[5,64,d]),("uniforms",[6,64]),("mixture",[5,64]),
                    ("centers",[5,d]),("covariances",[5,d,d]),("floors",[5]))]
                for constant in (False,True):
                    fn=make_fitted_twist_kernel(d,d,64,5,dtype.name,True,constant_twist=constant)
                    result=fn(*arrays)
                    assert all(bool(tf.reduce_all(tf.math.is_finite(x)).numpy()) for x in result)
                    (output/f"warmup-d{d}-{constant}.hlo.txt").write_text(
                        fn.experimental_get_compiler_ir(*arrays)(stage="optimized_hlo"))

    with (output / "traces.jsonl").open("w") as traces:
        for case in design["cases"]:
            d, N, T = (case[k] for k in ("d", "N", "T"))
            arm, seed = case["arm"], case["seed"]
            key = (d, N, T, arm == "constant")
            if key not in kernels:
                kernels[key] = tuple(make_fitted_twist_kernel(d,d,N,T,dtype.name,True,
                    constant_twist=arm == "constant",include_numerical_trace=flag) for flag in (False,True))
            fn, traced = kernels[key]
            directory = fixture / case["cell"]
            with tf.device("/GPU:0"):
                theta = read(directory / "theta.csv", [6])
                observations = read(directory / "observations.csv", [T,d])
                guides = [read(directory / f"{arm}-{suffix}.csv",shape) for suffix,shape in (
                    ("centers",[T,d]),("covariances",[T,d,d]),("floors",[T]))]
            with tf.device("/CPU:0"):
                initial = tf.random.stateless_normal([N,d],[seed,1],dtype=tf.float32)
                noise = tf.random.stateless_normal([T,N,d],[seed,2],dtype=tf.float32)
                uniform = tf.cast(tf.random.stateless_uniform([T+1,N],[seed,3],
                    minval=0,maxval=2**23,dtype=tf.int32),tf.float32)/2**23
                mixture = tf.cast(tf.random.stateless_uniform([T,N],[seed,4],
                    minval=0,maxval=2**23,dtype=tf.int32),tf.float32)/2**23
                draws = [tf.cast(x,dtype) for x in (initial,noise,uniform,mixture)]
            with tf.device("/GPU:0"):
                args = [theta,observations,*draws,*guides]
                result, traced_result = fn(*args), traced(*args)
                assert "GPU" in result[0].device
                assert all(bool(tf.reduce_all(tf.math.is_finite(x)).numpy()) for x in result)
                neutral = all(bool(tf.reduce_all(x == y).numpy()) for x,y in zip(result,traced_result[:3]))
                assert neutral, "Tracing changed the consumer"
                value = float(result[0].numpy())
                if not any(x["scope"] == list(key) for x in compilation):
                    hlo = fn.experimental_get_compiler_ir(*args)(stage="optimized_hlo")
                    hp = output / f"scope-{len(compilation):02d}.hlo.txt"
                    hp.write_text(hlo)
                    compilation.append(dict(scope=list(key),hlo_sha256=sha(hp),device=result[0].device,
                        trace_count=fn.experimental_get_tracing_count()))
            actual_trace = traced_result[-1]
            record = dict(cell=case["cell"],arm=arm,replicate=case["replicate"],seed=seed,N=N,T=T,
                log_likelihood=value,expected_log_likelihood=case["expected"][mode],
                exact_value_replay=value == case["expected"][mode],neutral=neutral,
                cloud=result[2].numpy().tolist(),score=result[1].numpy().tolist(),
                ancestors=actual_trace["ancestor_indices"].numpy().tolist(),
                cdf=actual_trace["ancestor_cdf"].numpy().tolist(),
                probabilities=actual_trace["gaussian_probability"].numpy().tolist(),
                mixture_choices=(tf.cast(mixture,dtype)<actual_trace["gaussian_probability"]).numpy().tolist(),
                uniforms=uniform.numpy().tolist(),mixture_uniforms=mixture.numpy().tolist())
            reconstructed = [[min(bisect.bisect_right(cdf,u),N-1) for u in us]
                             for cdf,us in zip(record["cdf"],record["uniforms"])]
            assert reconstructed == record["ancestors"], "CDF label reconstruction mismatch"
            assert [[u<p for u,p in zip(us,ps)] for us,ps in zip(record["mixture_uniforms"],
                record["probabilities"])] == record["mixture_choices"]
            traces.write(json.dumps(record,allow_nan=False)+"\n")
            traces.flush()
            checks.append({k:record[k] for k in ("cell","arm","replicate","exact_value_replay")})
            dump(output / "progress.json",dict(completed=len(checks),total=len(design["cases"])))
    dump(output / "result.json",dict(status="complete",mode=mode,checks=checks,
        tensorflow=tf.__version__,python=sys.version,build=tf.sysconfig.get_build_info(),
        memory_policy=policy,tf32_verified=tf.config.experimental.tensor_float_32_execution_enabled(),
        jit_compile=True,warmup_original=warmup_original,compilation=compilation,allocator=tf.config.experimental.get_memory_info("GPU:0"),
        gpu_uuid=os.environ["CUDA_VISIBLE_DEVICES"],
        consumer="bayesfilter.score_study.fitted_twist_tf.make_fitted_twist_kernel"))


def lattice_intervals(cdf):
    L = 2**23
    bounds = [0] + [math.ceil(L*min(1.,max(0.,x))) for x in cdf[:-1]] + [L]
    return list(zip(bounds[:-1],bounds[1:]))


def summarize(root, completed):
    traces = {}
    for mode,path in completed.items():
        records = [json.loads(x) for x in (Path(path)/"traces.jsonl").read_text().splitlines()]
        traces[mode] = {(x["cell"],x["arm"],x["replicate"]):x for x in records}
    assert all(set(x) == set(traces["tf32"]) for x in traces.values())
    rows = []
    for baseline in ("fp64","fp32"):
        for key,a in traces[baseline].items():
            b = traces["tf32"][key]
            assert a["uniforms"] == b["uniforms"] and a["mixture_uniforms"] == b["mixture_uniforms"]
            first = None
            pre,post = 0.,0.

            def ancestor_event(index):
                ca,cb=a["cdf"][index],b["cdf"][index]
                ia,ib=lattice_intervals(ca),lattice_intervals(cb)
                agreement=sum(max(0,min(v,y)-max(u,x)) for (u,v),(x,y) in zip(ia,ib))/(2**23)
                return dict(operation="ancestor",cdf_index=index,
                    can_affect_likelihood=index<a["T"],
                    max_cdf_difference=max(abs(x-y) for x,y in zip(ca,cb)),
                    conditional_lattice_disagreement_probability=1-agreement,
                    expected_changed_ancestors=a["N"]*(1-agreement),
                    observed_changed_ancestors=sum(x!=y for x,y in zip(a["ancestors"][index],b["ancestors"][index])))

            if a["ancestors"][0]!=b["ancestors"][0]:first=ancestor_event(0)
            for t in range(a["T"]):
                if first is None and a["mixture_choices"][t]!=b["mixture_choices"][t]:
                    first=dict(operation="mixture",time=t,can_affect_likelihood=True,
                        max_probability_difference=max(abs(x-y) for x,y in zip(a["probabilities"][t],b["probabilities"][t])))
                error=max(abs(x-y) for u,v in zip(a["cloud"][t],b["cloud"][t]) for x,y in zip(u,v))
                if first is None:pre=max(pre,error)
                else:post=max(post,error)
                if first is None and a["ancestors"][t+1]!=b["ancestors"][t+1]:first=ancestor_event(t+1)

            def minimum_ess(record):
                # Empirical CDF increments; an explanatory diagnostic only.
                values=[]
                for cdf in record["cdf"]:
                    weights=[y-x for x,y in zip([0.]+cdf[:-1],cdf)]
                    total=math.fsum(weights)
                    values.append(total*total/math.fsum(x*x for x in weights))
                return min(values)

            rows.append(dict(cell=key[0],arm=key[1],replicate=key[2],baseline=baseline,
                exact_replay=a["exact_value_replay"] and b["exact_value_replay"],first_changed_decision=first,
                max_cloud_error_before_changed_decision=pre,max_cloud_error_after_changed_decision=post,
                baseline_minimum_ess=minimum_ess(a),tf32_minimum_ess=minimum_ess(b)))
    dump(root / "localization.json",rows)
    decision=dict(selected_cases=len(traces["tf32"]),mode_replays=sum(len(x) for x in traces.values()),
        exact_value_replays=sum(x["exact_value_replay"] for group in traces.values() for x in group.values()),
        trace_neutral_and_choices_reconstructed=True,paired_cases=len(rows),
        changed_decision_pairs=sum(x["first_changed_decision"] is not None for x in rows),
        max_cloud_error_before_changed_decision=max(x["max_cloud_error_before_changed_decision"] for x in rows),
        max_cloud_error_after_changed_decision=max(x["max_cloud_error_after_changed_decision"] for x in rows),
        evidence_role="outcome_selected_explanatory_only",phase18_precision_veto_retained=True)
    dump(root / "decision.json",decision)
    return decision


def supervise(warmup_original=False):
    start=time.monotonic()
    repo=Path(__file__).resolve().parents[2]
    root,previous=repo/ROOT,repo/PREVIOUS
    root.mkdir(exist_ok=True)
    prior=json.loads((previous/"completed.json").read_text())
    ledger=root/"budget.json"
    budget=json.loads(ledger.read_text()) if ledger.exists() else dict(
        cpu_charged_seconds=0.,gpu_charged_seconds=0.,cpu_reserved_seconds=0.,gpu_reserved_seconds=0.,
        phase_cpu_cap_seconds=120.,phase_gpu_cap_seconds=2400.,attempts=0,
        **{k:v for k,v in json.loads((previous/"budget.json").read_text()).items() if k.startswith("remaining_campaign_")})
    assert not budget["cpu_reserved_seconds"] and not budget["gpu_reserved_seconds"]
    source=root/f"source-{len(list(root.glob('source-*')))+1:02d}"
    source.mkdir()
    files=[SCRIPT,PLAN,"bayesfilter/__init__.py","bayesfilter/runtime/__init__.py","bayesfilter/runtime/gpu_memory_policy.py"]
    files += [str(x.relative_to(repo)) for x in (repo/"bayesfilter/score_study").glob("*.py")]
    for name in files:
        dest=source/name; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(repo/name,dest)
    source_hashes={name:sha(source/name) for name in files}
    reference_manifest=json.loads((Path(prior["fp64"]).parent/"manifest.json").read_text())
    for name,digest in source_hashes.items():
        if name.startswith("bayesfilter/"):
            assert digest == reference_manifest["source_sha256"][name], "Shared source changed since phase18"
    selection=root/"selection.json"
    if not selection.exists():
        influence=json.loads((previous/"review-01/tail-influence.json").read_text())
        selected={(x["cell"],x["arm"],y["replicate"]) for x in influence if not x["precision_pass"] for y in x["top_three"]}
        metadata={(x["id"],x["arm"]):x for x in csv.DictReader((Path(prior["fixtures"])/"cells.csv").open())}
        values={mode:{(x["cell"],x["arm"],int(x["replicate"])):float(x["log_likelihood"])
            for x in csv.DictReader((Path(prior[mode])/"records.csv").open())} for mode in ("fp64","fp32","tf32")}
        cases=[]
        for cell,arm,rep in sorted(selected):
            m=metadata[cell,arm]
            cases.append(dict(cell=cell,arm=arm,replicate=rep,seed=int(m["particle_seed"])+rep,
                **{k:int(m[k]) for k in ("d","N","T")},expected={mode:v[cell,arm,rep] for mode,v in values.items()}))
        dump(selection,dict(fixture=prior["fixtures"],cases=cases,selection_source_sha256=sha(previous/"review-01/tail-influence.json"),
            original_csv_sha256={mode:sha(Path(prior[mode])/"records.csv") for mode in values}))
    for name,digest in prior["fixture_hashes"].items():assert sha(Path(prior["fixtures"])/name)==digest
    git=subprocess.check_output(["git","rev-parse","HEAD"],cwd=repo,text=True).strip()
    env=dict(os.environ,PYTHONPATH=str(source),CUDA_VISIBLE_DEVICES=GPU,TF_FORCE_GPU_ALLOW_GROWTH="true",
        BAYESFILTER_PRELOAD_CUSTOM_OP="0",OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",
        TF_NUM_INTRAOP_THREADS="2",TF_NUM_INTEROP_THREADS="2")
    if env.get("XLA_FLAGS",""):raise RuntimeError("Unexpected XLA_FLAGS")
    completed_path=root/"completed.json"
    completed=json.loads(completed_path.read_text()) if completed_path.exists() else {}
    gpu_start=budget["gpu_charged_seconds"]
    for mode in ("fp64","fp32","tf32"):
        if mode in completed:continue
        assert budget["attempts"]<6
        assert budget["gpu_charged_seconds"]+600<=budget["phase_gpu_cap_seconds"]
        assert budget["remaining_campaign_gpu_seconds"]>=600
        budget["attempts"]+=1; budget["gpu_reserved_seconds"]=600.; dump(ledger,budget)
        job=root/f"attempt{budget['attempts']:02d}-{mode}"; job.mkdir()
        command=[PYTHON,str(source/SCRIPT),"--mode",mode,"--selection",str(selection),"--output",str(job/"outputs")]
        if warmup_original:command.append("--warmup-original")
        manifest=dict(command=command,git_commit=git,source=str(source),source_sha256=source_hashes,
            input_selection_sha256=sha(selection),fixture_sha256=prior["fixture_hashes"],plan=PLAN,result_file=str(root/"result.md"),
            started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),environment={k:env[k] for k in (
                "CUDA_VISIBLE_DEVICES","TF_FORCE_GPU_ALLOW_GROWTH","PYTHONPATH","OPENBLAS_NUM_THREADS","OMP_NUM_THREADS",
                "MKL_NUM_THREADS","TF_NUM_INTRAOP_THREADS","TF_NUM_INTEROP_THREADS","BAYESFILTER_PRELOAD_CUSTOM_OP")},
            data_version="phase18_frozen_R_fixtures",seeds="selection.json; original phase18 streams",status="running")
        dump(job/"manifest.json",manifest); began=time.monotonic()
        try:
            with (job/"command.log").open("w") as log:
                result=subprocess.run(command,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=600)
            manifest.update(returncode=result.returncode,status="complete" if result.returncode==0 else "failed")
        except BaseException as exc:manifest.update(status="failed",error=repr(exc))
        finally:
            seconds=time.monotonic()-began; manifest["wall_seconds"]=seconds
            budget["gpu_reserved_seconds"]=0.; budget["gpu_charged_seconds"]+=seconds
            budget["remaining_campaign_gpu_seconds"]-=seconds
            manifest["output_sha256"]={str(x.relative_to(job)):sha(x) for x in job.rglob("*") if x.is_file() and x.name!="manifest.json"}
            dump(job/"manifest.json",manifest); dump(ledger,budget)
        if manifest["status"]!="complete":
            cpu=max(0.,time.monotonic()-start-(budget["gpu_charged_seconds"]-gpu_start))
            budget["cpu_charged_seconds"]+=cpu; budget["remaining_campaign_cpu_seconds"]-=cpu; dump(ledger,budget)
            raise RuntimeError(f"Repair required: {job}")
        completed[mode]=str(job/"outputs"); dump(completed_path,completed)
        print(json.dumps(dict(mode=mode,status=manifest["status"],seconds=seconds)),flush=True)
    decision=summarize(root,completed)
    cpu=max(0.,time.monotonic()-start-(budget["gpu_charged_seconds"]-gpu_start))
    budget["cpu_charged_seconds"]+=cpu; budget["remaining_campaign_cpu_seconds"]-=cpu; dump(ledger,budget)
    assert budget["cpu_charged_seconds"]<=budget["phase_cpu_cap_seconds"]
    dump(root/"summary-manifest.json",dict(source_sha256=sha(repo/SCRIPT),git_commit=git,CPU_only=True,
        gpu_intentionally_hidden_for_reporting=True,role="standard-library post-run diagnostic; no GPU imports",
        wall_seconds=cpu,plan=PLAN,selection_sha256=sha(selection),
        output_sha256={x:sha(root/x) for x in ("decision.json","localization.json")}))
    print(json.dumps(decision))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--execute",action="store_true")
    parser.add_argument("--warmup-original",action="store_true")
    parser.add_argument("--mode",choices=("fp64","fp32","tf32"))
    parser.add_argument("--selection",type=Path); parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    if args.execute:
        os.environ["CUDA_VISIBLE_DEVICES"]="-1"
        supervise(args.warmup_original)
    else:
        if not args.mode or not args.selection or not args.output:parser.error("worker requires mode, selection and output")
        worker(args.mode,args.selection,args.output,args.warmup_original)
