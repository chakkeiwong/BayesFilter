"""CPU/GPU independent diagnostic of a recorded finite-fit parity failure."""
import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]="true"
os.environ["BAYESFILTER_PRELOAD_CUSTOM_OP"]="0"
import argparse,csv,hashlib,json,math,subprocess,sys,time
from pathlib import Path
ROOT=Path("/home/chakwong/BayesFilter")

def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True)
    p.add_argument("--fixture",type=Path,required=True);a=p.parse_args()
    (a.output/"nvidia-smi.log").write_text(subprocess.check_output(["nvidia-smi"],text=True))
    sys.path.insert(0,str(ROOT))
    import tensorflow as tf
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    memory=configure_tensorflow_gpu_memory_growth(tf)
    if not tf.config.list_logical_devices("GPU"):raise RuntimeError("No trusted GPU")
    tf.config.experimental.enable_tensor_float_32_execution(False)
    from bayesfilter.score_study.iapf_finite_tf import make_finite_fit_kernel,make_finite_objective_kernel
    from bayesfilter.score_study.iapf_paper_tf import _log_normal,_add_floor
    def plain(x):
        if tf.is_tensor(x):return plain(x.numpy().tolist())
        if isinstance(x,dict):return {k:plain(v) for k,v in x.items()}
        if isinstance(x,(list,tuple)):return [plain(v) for v in x]
        if isinstance(x,float) and not math.isfinite(x):return str(x)
        return x
    def save(name,x):(a.output/name).write_text(json.dumps(plain(x),indent=2)+"\n")
    def read(folder,name,shape=None):
        with (folder/(name+".csv")).open() as stream:
            matrix=[[float(x) for x in row] for row in csv.reader(stream)]
        value=tf.constant(matrix,tf.float64)
        return tf.reshape(value,shape) if shape is not None else value
    source=ROOT/"bayesfilter/score_study/iapf_finite_tf.py"
    if source.read_bytes()!=(a.output/"sources"/source.name).read_bytes():
        raise RuntimeError("Runtime source drift")
    save("runtime.json",dict(tensorflow=tf.__version__,gpu_memory_policy=memory,
        dtype="float64",tf32=False,scope="diagnostic CPU/GPU graph/XLA replay",
        gpu_intentionally_hidden=False,trust="escalated_execution",
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
    root=a.fixture/"controller";ref=root/"backward2"
    cloud=read(root/"call2","clouds",[100,1000,5])
    prior=a.fixture.parent/"attempt24_iapf_paper_adaptive_gpu_xla"
    failed=json.loads((prior/"fixed-cloud-backward2.json").read_text())
    A=read(root,"A",[5,5]);C=read(root,"C",[5,5]);R=read(root,"R",[5,5])
    obs=read(root,"observations",[100,5]);Q=tf.eye(5,dtype=tf.float64)
    centers=tf.constant(failed["centers"],tf.float64)
    covs=tf.constant(failed["covariances"],tf.float64)
    floors=tf.constant(failed["floors"],tf.float64)
    results=[];start=time.monotonic()
    for mode,device,jit in [("cpu","/CPU:0",False),("gpu-graph","/GPU:0",False),
                             ("gpu-xla","/GPU:0",True)]:
        # Fresh factories prevent an earlier CPU placement being reused by a graph probe.
        make_finite_fit_kernel.cache_clear();make_finite_objective_kernel.cache_clear()
        with tf.device(device):
            fit_kernel=make_finite_fit_kernel(5,1000,"float64",jit)
            objective=make_finite_objective_kernel(5,1000,"float64",jit)
            specs=[tf.TensorSpec([1000,5],tf.float64),tf.TensorSpec([5],tf.float64),
                   tf.TensorSpec([5],tf.float64),tf.TensorSpec([5,5],tf.float64),
                   tf.TensorSpec([],tf.float64)]
            @tf.function(input_signature=specs,jit_compile=jit)
            def target(points,y,next_mean,next_cov,next_floor):
                return (_log_normal(tf.broadcast_to(y,[1000,5]),tf.matmul(points,C,transpose_b=True),R)+
                        _add_floor(_log_normal(tf.matmul(points,A,transpose_b=True),next_mean,Q+next_cov),next_floor))
            for t in [55,54,53]:
                x=cloud[t-1];ly=read(ref,"fit"+str(t)+"_targets",[1000])
                initial=read(ref,"fit"+str(t)+"_initial",[10])
                theta=read(ref,"fit"+str(t)+"_theta",[10])
                accepted=read(ref,"fit"+str(t)+"_accepted")
                before=time.monotonic();fit=fit_kernel(x,ly,tf.constant(False))
                fit=plain(fit);elapsed=time.monotonic()-before
                save(mode+"-time"+str(t)+"-fit.json",fit)
                error=max(abs(u-v) for u,v in zip(fit["theta"],plain(theta)))
                passed=all(abs(u-v)<=1e-5*(1+abs(v)) for u,v in zip(fit["theta"],plain(theta)))
                # Reference accepted rows: z[11], f, gradient[11].
                expected=plain(accepted);trajectory=[]
                iteration_rows={}
                for row in fit["trace"][:fit["trace_length"]]:
                    iteration_rows.setdefault(int(row[-2]),row)
                for i,row in enumerate(expected):
                    ev=objective(x,ly,initial,tf.constant(row[:11],tf.float64),tf.constant(False))
                    g=plain(ev[1]);z=iteration_rows.get(i+1,[0.]*11)[:11]
                    trajectory.append(dict(iteration=i+1,
                        parameter_error=max(abs(u-v) for u,v in zip(z,row[:11])),
                        gradient_error=max(abs(u-v) for u,v in zip(g,row[12:])),
                        gradient_relative_error=max(abs(u-v)/(1e-30+abs(v)) for u,v in zip(g,row[12:])),
                        objective_error=abs(float(ev[0].numpy())-row[11])))
                save(mode+"-time"+str(t)+"-replay.json",trajectory)
                entry=dict(mode=mode,time=t,endpoint_error=error,endpoint_pass=passed,
                    initializer_error=float(tf.reduce_max(tf.abs(tf.constant(fit["initial"],tf.float64)-initial)).numpy()),
                    code=fit["code"],iterations=fit["iterations"],seconds=elapsed,
                    max_replay_gradient_error=max(r["gradient_error"] for r in trajectory),
                    first_iterate_over_1e8=next((r for r in trajectory if r["parameter_error"]>1e-8),None))
                if t==54:
                    derived=target(x,obs[t-1],centers[t],covs[t],floors[t])
                    alternative=fit_kernel(x,derived,tf.constant(False))
                    save(mode+"-time54-propagated-fit.json",alternative)
                    entry["propagated_target_max_error"]=float(tf.reduce_max(tf.abs(derived-ly)).numpy())
                    entry["propagated_endpoint_max_error"]=float(tf.reduce_max(tf.abs(alternative["theta"]-theta)).numpy())
                results.append(entry);save("results.json",results)
    save("decision.json",dict(status="PASS",meaning="DIAGNOSTIC_COMPLETED_NOT_PARITY_PROMOTION",
                             results=results,seconds=time.monotonic()-start))
    with (a.output/"decision.csv").open("w") as stream:
        w=csv.DictWriter(stream,fieldnames=["status","scope"]);w.writeheader()
        w.writerow(dict(status="PASS",scope="FAILURE_LOCALIZATION_ONLY"))
    print(json.dumps(results),flush=True)

if __name__=="__main__":main()
