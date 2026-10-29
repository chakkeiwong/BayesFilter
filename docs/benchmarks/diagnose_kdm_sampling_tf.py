"""Diagnostic GPU/XLA source-replay and analytic moment checks for KDM sampling."""
import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]="true"
os.environ["BAYESFILTER_PRELOAD_CUSTOM_OP"]="0"
import argparse,datetime,hashlib,json,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path("/home/chakwong/BayesFilter");sys.path.insert(0,str(ROOT))

def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+"\n")

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=False)
    began=time.monotonic();gpu_start=None
    files=["docs/benchmarks/diagnose_kdm_sampling_tf.py","docs/benchmarks/reference_kdm_sampling.py",
           "docs/benchmarks/reference_kdm_neural.py","bayesfilter/replication/kdm_sampling_tf.py",
           "docs/plans/controlled-smc-kdm-native-sampling-20261006.md"]
    (out/"source_snapshot").mkdir()
    hashes={}
    for filename in files:
        raw=(ROOT/filename).read_bytes();hashes[filename]=hashlib.sha256(raw).hexdigest()
        (out/"source_snapshot"/Path(filename).name).write_bytes(raw)
    reference_python=ROOT/".localresources/envs/mdpf-reference-20260924/bin/python"
    if not reference_python.exists():
        raise RuntimeError("Pinned author reference interpreter is missing")
    cmd=[str(reference_python),"-B",str(ROOT/"docs/benchmarks/reference_kdm_sampling.py"),"--output",str(out/"author.json")]
    manifest=dict(schema="kdm_sampling_manifest_v1",command=[sys.executable,*sys.argv],
        author_command=cmd,git=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        branch=subprocess.check_output(["git","branch","--show-current"],cwd=ROOT,text=True).strip(),
        code_sha256=hashes,python=sys.executable,started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        seeds=[2026100611,2026100612,2026100613,2026100614],jit_compile=True,tf32=False,
        plan=files[-1],result_file="docs/plans/controlled-smc-kdm-native-sampling-result-20261006.md",
        environment="tftwogpu",data="synthetic source-anchored seven-regime fixture",cpu_wall_seconds=0.)
    write(out/"manifest.json",manifest)
    try:
        tick=time.monotonic()
        ref_env=os.environ.copy()
        ref_env["CUDA_VISIBLE_DEVICES"]="-1"
        ref_env["LD_LIBRARY_PATH"]="/home/chakwong/anaconda3/envs/tf-gpu2/lib"
        with (out/"author.log").open("w") as log:
            run=subprocess.run(cmd,env=ref_env,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT,timeout=300)
        manifest["cpu_wall_seconds"]=time.monotonic()-tick
        if run.returncode:raise RuntimeError("Source reference failed; see author.log")
        reference=json.loads((out/"author.json").read_text())
        gpu_start=time.monotonic()
        import tensorflow as tf
        from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
        memory=configure_tensorflow_gpu_memory_growth(tf,require_gpu=True)
        tf.config.experimental.enable_tensor_float_32_execution(False)
        tf.config.set_soft_device_placement(False)
        from bayesfilter.replication.kdm_sampling_tf import SourceKDMResampler
        checks=[];cases=[];moments=[]
        def check(name,passed,**extra):checks.append(dict(name=name,pass_check=bool(passed),**extra))
        with tf.device("/GPU:0"):
            for case in reference["cases"]:
                dtype=tf.as_dtype(case["dtype"]);label=case["dtype"]
                tensor=lambda x:tf.constant(x,dtype)
                p,w,bw=[tensor(case[k]) for k in ["locations","weights","bandwidth"]]
                replay=SourceKDMResampler(7,3,25,dtype=dtype)
                tick=time.monotonic()
                got=replay.replay(p,w,bw,tf.constant(case["ancestors"],tf.int32),
                                  tensor(case["normal_components"]),tf.constant(case["uniforms"],tf.float64))
                expected=tensor(case["replay"])
                tol=1e-9 if dtype==tf.float64 else 128*2**-23
                error=tf.abs(got["samples"]-expected)
                scaled=error/(tf.constant(tol,dtype)*(1+tf.abs(expected)))
                check(label+"/source_replay",tf.reduce_all(scaled<=1).numpy(),
                      max_absolute_error=float(tf.reduce_max(error).numpy()),
                      max_scaled_error=float(tf.reduce_max(scaled).numpy()))
                check(label+"/replay_valid",got["valid"].numpy())
                check(label+"/source_loop_count",int(got["iterations"].numpy())==case["iterations"])
                replay_seconds=time.monotonic()-tick
                sampler=SourceKDMResampler(7,3,32768,dtype=dtype)
                seed=tf.constant([2026100613,2026100614],tf.int64)
                tick=time.monotonic()
                native=sampler.sample(p,w,bw,seed)
                check(label+"/native_valid",native["valid"].numpy())
                native_seconds=time.monotonic()-tick
                check(label+"/gpu_placement","GPU:0" in native["samples"].device)
                # Sampling is deliberately detached, as in the source.
                variable=tf.Variable(p)
                with tf.GradientTape() as tape:
                    replayed=sampler.sample(variable,w,bw,seed)
                    scalar=tf.reduce_sum(replayed["samples"])
                check(label+"/stopped_sampling_gradient",tape.gradient(scalar,variable) is None)
                check(label+"/stateless_repeat",tf.reduce_all(native["samples"]==replayed["samples"]).numpy())
                check(label+"/one_native_trace",sampler.sample.experimental_get_tracing_count()==1)
                # Analytic calculations are diagnostic and evaluated in double.
                pd=tf.cast(p,tf.float64);wd=tf.cast(w,tf.float64)
                wd/=tf.reduce_sum(wd,-1,keepdims=True)
                bd=tf.cast(bw,tf.float64)
                mu=pd[...,:2];var=tf.square(bd[:,:2])[:,None,:]
                e1=tf.reduce_sum(wd[...,None]*mu,1)
                e2=tf.reduce_sum(wd[...,None]*(tf.square(mu)+var),1)
                e4=tf.reduce_sum(wd[...,None]*(tf.pow(mu,4)+6*tf.square(mu)*var+3*tf.square(var)),1)
                kappa=tf.cast(1/(bw[:,-1]+tf.constant(1e-8,dtype)),tf.float64)
                ratio=tf.math.bessel_i1e(kappa)/tf.math.bessel_i0e(kappa)
                circ=tf.stack([tf.sin(pd[...,2]),tf.cos(pd[...,2])],-1)
                ec=ratio[:,None]*tf.reduce_sum(wd[...,None]*circ,1)
                expected_m=tf.concat([e1,e2,ec],-1)
                variance=tf.concat([e2-tf.square(e1),e4-tf.square(e2),tf.ones([7,2],tf.float64)],-1)
                se=tf.sqrt(variance/32768.)
                for backend,draw in [("source",tensor(case["native"])),("tf",native["samples"])]:
                    x=tf.cast(draw,tf.float64)
                    feat=tf.concat([x[...,:2],tf.square(x[...,:2]),
                                   tf.sin(x[...,2:]),tf.cos(x[...,2:])],-1)
                    measured=tf.reduce_mean(feat,1)
                    z=tf.abs(measured-expected_m)/se
                    check(label+"/"+backend+"_analytic_moments",tf.reduce_all(z<=6).numpy(),
                          max_standard_errors=float(tf.reduce_max(z).numpy()),role="DIAGNOSTIC_SCREEN")
                    moments.append(dict(dtype=label,backend=backend,expectation=expected_m.numpy().tolist(),
                        measured=measured.numpy().tolist(),standard_errors=se.numpy().tolist(),
                        standardized_deviation=z.numpy().tolist()))
                counts=tf.reduce_sum(tf.one_hot(native["ancestors"],3,dtype=tf.float64),1)
                frequencies=counts/32768.
                category_se=tf.sqrt(wd*(1-wd)/32768.)
                category_bound=6*category_se+1e-12
                check(label+"/category_frequency",tf.reduce_all(tf.abs(frequencies-wd)<=category_bound).numpy(),
                      counts=counts.numpy().tolist(),role="DIAGNOSTIC_SCREEN")
                source_counts=tf.constant(case["native_counts"],tf.float64)
                check(label+"/source_category_frequency",
                      tf.reduce_all(tf.abs(source_counts/32768.-wd)<=category_bound).numpy(),
                      counts=source_counts.numpy().tolist(),role="DIAGNOSTIC_SCREEN")
                check(label+"/zero_weight_never_drawn",tf.reduce_all(tf.where(wd==0,counts==0,True)).numpy())
                bad_band=tf.concat([-tf.ones([7,1],dtype),bw[:,1:]],-1)
                invalid=replay.replay(p,w,bad_band,tf.constant(case["ancestors"],tf.int32),
                                  tensor(case["normal_components"]),tf.constant(case["uniforms"],tf.float64))
                check(label+"/invalid_band_rejected",not bool(invalid["valid"].numpy()))
                cases.append(dict(dtype=label,replay_seconds=replay_seconds,native_seconds=native_seconds,
                    source_iterations=case["iterations"],native_iterations=int(native["iterations"].numpy()),
                    native_trace_count=sampler.sample.experimental_get_tracing_count(),
                    device=native["samples"].device))
        failed=[x for x in checks if not x["pass_check"]]
        result=dict(status="PASS",scope="SOURCE_SAMPLING_IMPLEMENTATION_DIAGNOSTIC",replication="INCOMPLETE",
                    numerical_screen_passed=not failed,scientific_verdict="SAMPLING_SCREEN_PASS" if not failed else "SAMPLING_SCREEN_FAIL",
                    checks=checks,checks_total=len(checks),failed_checks=failed,cases=cases,moments=moments,
                    memory_policy=memory,allocator=tf.config.experimental.get_memory_info("GPU:0"),
                    physical_devices=[dict(name=d.name,details=tf.config.experimental.get_device_details(d))
                                      for d in tf.config.list_physical_devices("GPU")],
                    tensorflow=tf.__version__,torch=reference["torch"],source_commit=reference["source_commit"],
                    source_hashes=reference["source_hashes"],torch_sampler_sha256=reference["torch_sampler_sha256"])
        write(out/"summary.json",result);print(json.dumps(dict(output=str(out),checks=len(checks),failed=len(failed))))
        return 0
    except Exception:
        (out/"error.txt").write_text(traceback.format_exc())
        write(out/"summary.json",dict(status="ERROR",scope="SAMPLING_EXECUTION",replication="INCOMPLETE"))
        print(traceback.format_exc());return 1
    finally:
        if gpu_start is not None:manifest["gpu_wall_seconds"]=time.monotonic()-gpu_start
        manifest["wall_seconds"]=time.monotonic()-began
        manifest["output_sha256"]={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir()
                                  if p.is_file() and p.name!="manifest.json"}
        write(out/"manifest.json",manifest)
if __name__=="__main__":raise SystemExit(main())
