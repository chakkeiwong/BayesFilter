"""GPU/XLA diagnostic of evolving MDPF states under common base variates.

Author equations run in the existing CPU-only author environment. Only the
categorical random mapping is changed for coupling; all TF hot kernels below
have stable compiled signatures. This module is not an admitted runtime route.
"""
import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]="true"
os.environ["BAYESFILTER_PRELOAD_CUSTOM_OP"]="0"
os.environ["OMP_NUM_THREADS"]="1"
os.environ["TF_NUM_INTEROP_THREADS"]="1"
os.environ["TF_NUM_INTRAOP_THREADS"]="1"
import argparse, copy, gzip, hashlib, json, math, subprocess, sys, tempfile, time, traceback
from pathlib import Path
from multiprocessing.connection import Listener
ROOT=Path("/home/chakwong/BayesFilter")
sys.path.insert(0,str(ROOT))
BASE=ROOT/"docs/plans/artifacts/controlled-smc-kdm-replication-20260928"
SEED=202610123
def write(p,x):p.write_text(json.dumps(x,allow_nan=False)+"\n")
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main(out):
    out.mkdir(exist_ok=False)
    import tensorflow as tf
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    memory=configure_tensorflow_gpu_memory_growth(tf,require_gpu=True)
    tf.config.experimental.enable_tensor_float_32_execution(False)
    tf.config.set_soft_device_placement(False)
    from bayesfilter.replication.kdm_neural_replay_tf import BearingsNeuralReplay
    from bayesfilter.replication.kdm_source_stages_tf import SourceFullTraining
    def host(x):
        if tf.is_tensor(x):return x.numpy().tolist()
        if isinstance(x,dict):return {k:host(v) for k,v in x.items()}
        if isinstance(x,(tuple,list)):return [host(v) for v in x]
        return x
    # Diagnostic precision intervention: all Gaussian/categorical base arrays
    # are generated in FP32 and promoted exactly for FP64. VM rejection
    # uniforms remain native FP64 in both arms, as in the author algorithm.
    native_normal=tf.random.stateless_normal
    native_uniform=tf.random.stateless_uniform
    def quantized_normal(shape,seed,dtype=tf.float32,**kwargs):
        return tf.cast(native_normal(shape,seed,dtype=tf.float32,**kwargs),dtype)
    def quantized_uniform(shape,seed,dtype=tf.float32,**kwargs):
        if len(shape)==2 and list(shape)==[64,25]:
            return tf.cast(native_uniform(shape,seed,dtype=tf.float32,**kwargs),dtype)
        return native_uniform(shape,seed,dtype=dtype,**kwargs)
    tf.random.stateless_normal=quantized_normal
    tf.random.stateless_uniform=quantized_uniform
    # Each key has a fixed identity independent of rejection-loop termination.
    random_functions={}
    for dtype in (tf.float32,tf.float64):
        for kind,shape,uniform in (("proposal",[64,25,4],False),("normal",[25,64,25,2],False),
             ("initial_normal",[25,64,1,2],False),("categorical",[64,25],True)):
            def draw(seed,shape=shape,dtype=dtype,uniform=uniform):
                fn=tf.random.stateless_uniform if uniform else tf.random.stateless_normal
                return fn(shape,seed,dtype=dtype,alg="philox")
            random_functions[(kind,dtype.name)]=tf.function(draw,input_signature=[tf.TensorSpec([2],tf.int32)],jit_compile=True,autograph=False)
    for kind,shape in (("uniform",[3,25,64,25]),("initial_uniform",[3,25,64,1])):
        def draw(seed,shape=shape):return tf.random.stateless_uniform(shape,seed,dtype=tf.float64,alg="philox")
        random_functions[(kind,"float64")]=tf.function(draw,input_signature=[tf.TensorSpec([2],tf.int32)],jit_compile=True,autograph=False)
    fold=tf.random.experimental.stateless_fold_in
    key_functions={}
    for kind in ("proposal","normal","initial_normal","categorical","uniform","initial_uniform","resampler"):
        def make_key(root,t,iteration,kind=kind):
            if kind.startswith("initial_"):
                key=fold(root,0);return fold(key,0 if kind=="initial_normal" else iteration+1)
            seq=fold(root,1)
            if kind=="proposal":return fold(seq,1000+t)
            key=fold(seq,t)
            if kind=="resampler":return key
            return fold(key,{"normal":1,"categorical":0}.get(kind,iteration+2))
        key_functions[kind]=tf.function(make_key,input_signature=[tf.TensorSpec([2],tf.int64),
            tf.TensorSpec([],tf.int32),tf.TensorSpec([],tf.int32)],jit_compile=True,autograph=False)
    def base_key(kind,index,t=0,iteration=0):
        return key_functions[kind](tf.constant([SEED,index],tf.int64),tf.constant(t,tf.int32),tf.constant(iteration,tf.int32))
    def base(kind,dtype,index,t=0,iteration=0):
        dtype="float64" if "uniform" in kind else dtype
        return random_functions[(kind,dtype)](base_key(kind,index,t,iteration))
    categorical_functions={}
    for dtype in (tf.float32,tf.float64):
        def categorical(w,seed,dtype=dtype):
            u=tf.random.stateless_uniform([64,25],fold(seed,0),dtype=dtype,alg="philox")
            cdf=tf.cumsum(w/tf.reduce_sum(w,axis=-1,keepdims=True),axis=-1)
            return tf.minimum(tf.reduce_sum(tf.cast(u[...,None]>=cdf[:,None,:],tf.int32),axis=-1),24)
        categorical_functions[dtype.name]=tf.function(categorical,input_signature=[tf.TensorSpec([64,25],dtype),
            tf.TensorSpec([2],tf.int32)],jit_compile=True,autograph=False)
    def coupled_sample(sampler):
        def sample(p,w,bw,seed):
            key=lambda tag:fold(seed,tag)
            ancestors=categorical_functions[sampler.dtype.name](w,seed)
            z=tf.random.stateless_normal(sampler.shape+[2],key(1),dtype=sampler.dtype,alg="philox")
            normal=p[None,:,:,:2]+bw[None,:,None,:2]*z
            def uniform(i):return tf.random.stateless_uniform([3]+sampler.shape,key(i+2),dtype=tf.float64,alg="philox")
            return sampler._assemble(p,w,bw,ancestors,normal,uniform)
        return sample
    def difference(a,b,dtype,angle=False):
        a=tf.cast(a,tf.float64) if tf.is_tensor(a) else tf.constant(a,tf.float64)
        b=tf.cast(b,tf.float64) if tf.is_tensor(b) else tf.constant(b,tf.float64)
        d=a-b
        if angle:d=tf.atan2(tf.sin(d),tf.cos(d))
        error=tf.abs(d);tol=2e-5 if dtype=="float32" else 2e-9
        return dict(max_abs=float(tf.reduce_max(error)),rms=float(tf.sqrt(tf.reduce_mean(d*d))),
                    max_scaled=float(tf.reduce_max(error/(tol*(1+tf.abs(b))))),
                    finite=bool(tf.reduce_all(tf.math.is_finite(a))&tf.reduce_all(tf.math.is_finite(b))))
    def group(actual,expected,dtype):
        items={n:difference(actual[n],expected[n],dtype) for n in actual}
        return dict(max_abs=max(v["max_abs"] for v in items.values()),
            max_scaled=max(v["max_scaled"] for v in items.values()),
            failed_arrays=[n for n,v in items.items() if v["max_scaled"]>1],arrays=items)
    def tf_record(model,result,index):
        names=model.names;o=result["outputs"]
        # The real native sequence returns its live resampling draws and noises.
        ancestors=[]
        for t in range(2,17):
            ancestors.append(host(tf.cast(o[9][t-1],tf.int32)) if len(o)>9 else
                             host(categorical_functions[model.dtype.name](o[3][t-2],base_key("resampler",index,t))))
        for t in range(1,17):
            assert bool(tf.reduce_all(o[7][t-1]==base("proposal",model.dtype.name,index,t))),"Proposal base-input mismatch"
        return dict(loss=float(result["loss"]),gradient_norm=float(result["gradient_norm"]),
            gradients=dict(zip(names,host(result["raw_gradients"]))),
            parameters=dict(zip(names,host(result["parameters_after"]))),
            moments={n:dict(exp_avg=host(m),exp_avg_sq=host(v)) for n,m,v in zip(names,result["first_moments"],result["second_moments"])},
            captured=dict(raw=host(o[5]),ancestors=ancestors,iterations=host(o[8][1:]),samples=host(o[6][1:]),
                outputs=[dict(particles=host(o[0][j]),particle_weights=host(o[1][j]),bandwidths=host(o[2][j])) for j in range(16)]))
    def compare(a,b,dtype):
        ac,bc=a["captured"],b["captured"]
        ancestry=[]
        for t,(x,y) in enumerate(zip(ac["ancestors"],bc["ancestors"]),2):
            ancestry.append(dict(time=t,disagreements=int(tf.reduce_sum(tf.cast(tf.not_equal(x,y),tf.int32)))))
        times=[]
        for t,(x,y,raw) in enumerate(zip(ac["outputs"],bc["outputs"],ac["raw"]),1):
            p=tf.constant(x["particles"],tf.float64);q=tf.constant(y["particles"],tf.float64)
            r=tf.sqrt(tf.reduce_sum(tf.square(tf.constant(raw,tf.float64)[...,-2:]),-1))
            times.append(dict(time=t,position=difference(p[...,:2],q[...,:2],dtype),
                angle=difference(p[...,2],q[...,2],dtype,True),minimum_angle_radius=float(tf.reduce_min(r)),
                weights=difference(x["particle_weights"],y["particle_weights"],dtype)))
        return dict(loss_difference=a["loss"]-b["loss"],gradient_norm_difference=a["gradient_norm"]-b["gradient_norm"],
            parameters=group(a["parameters"],b["parameters"],dtype),gradients=group(a["gradients"],b["gradients"],dtype),
            first_moments=group({n:v["exp_avg"] for n,v in a["moments"].items()},{n:v["exp_avg"] for n,v in b["moments"].items()},dtype),
            second_moments=group({n:v["exp_avg_sq"] for n,v in a["moments"].items()},{n:v["exp_avg_sq"] for n,v in b["moments"].items()},dtype),
            ancestors=ancestry,times=times,
            max_sampling_iterations=max(ac["iterations"]+bc["iterations"]))
    def flags(comparison):
        fields=("gradients","parameters","first_moments","second_moments")
        return [f for f in fields if comparison[f]["max_scaled"]>1]+(
            ["ancestors"] if any(x["disagreements"] for x in comparison["ancestors"]) else [])
    def save_gzip(path,value):
        with gzip.open(path,"wt",compresslevel=1) as f:json.dump(value,f,allow_nan=False)

    started=time.monotonic()
    prior=BASE/"causal-localization-20261012/attempt06/diagnostic"
    case_path=prior/"unit01-tf-float32-first-flag.json.gz"
    with gzip.open(case_path,"rt") as f:case=json.load(f)
    source=case["source"];index=source["index"];state=source["before"]
    fixture=dict(variant="mdpf",dtype="float32",batch_size=64,parameters=state["parameters"],
        trainable_names=state["trainable_names"],
        frozen_names=[n for n in state["parameters"] if n not in state["trainable_names"]],
        learning_rates=state["rates"],constants=state["constants"])
    inputs=[tf.constant(source["data"][k],tf.bool if k=="ground_truth_mask" else tf.float32)
            for k in ("states","observations","ground_truth_mask")]
    seed=tf.constant([SEED,index],tf.int64)
    controls=[]
    def check(name,passed):
        controls.append(dict(name=name,passed=bool(passed)))
        assert passed,name
    source_anc=tf.constant(source["captured"]["ancestors"],tf.int32)
    resample_keys=tf.stack([base_key("resampler",index,t) for t in range(2,17)])
    def observed_sampler(sampler,mode,event):
        def sample(p,w,bw,seed):
            key=lambda tag:fold(seed,tag)
            u=tf.random.stateless_uniform([64,25],key(0),dtype=sampler.dtype,alg="philox")
            cdf=tf.cumsum(w/tf.reduce_sum(w,-1,keepdims=True),-1)
            ancestors=tf.minimum(tf.reduce_sum(tf.cast(u[...,None]>=cdf[:,None,:],tf.int32),-1),24)
            j=tf.argmax(tf.cast(tf.reduce_all(resample_keys==seed[None,:],-1),tf.int32),output_type=tf.int32)
            if mode=="all":
                ancestors=tf.gather(source_anc,j)
            elif mode=="first":
                t,row,slot=event
                ancestors=tf.cond(j==t-2,
                    lambda:tf.tensor_scatter_nd_update(ancestors,[[row,slot]],[source_anc[t-2,row,slot]]),
                    lambda:ancestors)
            z=tf.random.stateless_normal(sampler.shape+[2],key(1),dtype=sampler.dtype,alg="philox")
            normal=p[None,:,:,:2]+bw[None,:,None,:2]*z
            def uniform(i):return tf.random.stateless_uniform([3]+sampler.shape,key(i+2),dtype=tf.float64,alg="philox")
            result=sampler._assemble(p,w,bw,ancestors,normal,uniform)
            return dict(result,categorical_cdf=cdf,categorical_uniform=u)
        return sample
    # The repaired crossed-start endpoint calls this same sampler factory.
    from docs.benchmarks.diagnose_kdm_crossed_starts_tf import make_observed_coupled_sampler
    def run_tf(label,capture,mode="natural",event=None):
        model=BearingsNeuralReplay(fixture,capture_sampling_diagnostics=capture)
        model.load_state_dict(state)
        model.sampler._sample=(make_observed_coupled_sampler(tf,model.sampler) if capture and mode=="natural"
            else observed_sampler(model.sampler,mode,event) if capture else coupled_sample(model.sampler))
        stage=SourceFullTraining(model)
        result=stage.update(*inputs,seed)
        check(label+"/valid",bool(result["valid"]) and bool(result["committed"]))
        check(label+"/one_trace",stage.update.experimental_get_tracing_count()==1)
        record=tf_record(model,result,index)
        comparison=compare(record,source,"float32")
        save_gzip(out/(label+".json.gz"),dict(record=record,result=host(result),comparison=comparison))
        return result,record,comparison
    baseline,base_record,base_compare=run_tf("capture-off",False)
    check("capture_off_saved_result",host(baseline)==case["same_state"])
    natural,natural_record,natural_compare=run_tf("capture-on",True)
    capture_change=dict(loss=float(natural["loss"]-baseline["loss"]),
        parameters=group(natural_record["parameters"],base_record["parameters"],"float32"),
        gradients=group(natural_record["gradients"],base_record["gradients"],"float32"),
        exact_numerical_outputs=all(host(natural[k])==host(baseline[k]) for k in
            ("loss","raw_gradients","parameters_after","first_moments","second_moments")))
    # Direct choices from the instrumented kernel, not recomputed ancestors.
    actual=natural_record["captured"]["ancestors"]
    event=next(((t+2,row,slot) for t,(a,b) in enumerate(zip(actual,source["captured"]["ancestors"]))
                for row,(ar,br) in enumerate(zip(a,b)) for slot,(x,y) in enumerate(zip(ar,br)) if x!=y),None)
    interventions={"natural":natural_compare}
    boundary=None
    if event is not None:
        t,row,slot=event;o=natural["outputs"]
        cdf=host(o[10][t-1,row]);u=float(o[11][t-1,row,slot])
        sa=source["captured"]["ancestors"][t-2][row][slot];ta=actual[t-2][row][slot]
        source_cdf=source["captured"]["cdf"][t-2][row]
        boundary=dict(time=t,row=row,slot=slot,source_ancestor=sa,tf_ancestor=ta,
            uniform=u,source_uniform=source["captured"]["uniforms"][t-2][row][slot],
            source_cdf=source_cdf,tf_cdf=cdf,
            source_lower=0. if sa==0 else source_cdf[sa-1],source_upper=source_cdf[sa],
            tf_lower=0. if ta==0 else cdf[ta-1],tf_upper=cdf[ta])
        check("boundary_same_uniform",u==boundary["source_uniform"])
        check("tf_consumed_interval",boundary["tf_lower"]<=u<boundary["tf_upper"] or ta==24)
        check("source_consumed_interval",boundary["source_lower"]<=u<boundary["source_upper"] or sa==24)
        for mode in ("first","all"):
            _,_,comparison=run_tf("force-"+mode,True,mode,event)
            interventions[mode]=comparison
    write(out/"tf-intervention.json",dict(capture_change=capture_change,event=event,boundary=boundary,
        comparisons=interventions,scope="conditional single-case diagnostic, not qualification"))

    # Replay the original source independently; perturb only the selected
    # categorical uniform at the directly observed boundary.
    temp=tempfile.TemporaryDirectory(prefix="kdm-boundary-",dir="/tmp")
    listener=Listener(str(Path(temp.name)/"reference.sock"),family="AF_UNIX")
    listener._listener._socket.settimeout(60)
    log=(out/"author.log").open("w")
    env=dict(os.environ,CUDA_VISIBLE_DEVICES="-1",LD_LIBRARY_PATH="/home/chakwong/anaconda3/envs/tf-gpu2/lib")
    proc=subprocess.Popen(["/home/chakwong/anaconda3/envs/tf-gpu2/bin/python","-I","-B",
        str(ROOT/"docs/benchmarks/reference_kdm_crossed_starts.py"),"--socket",listener.address],
        cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    conn=None;cache={};source_cases=[];hashes=json.loads((prior/"random-hashes.json").read_text())
    def receive():
        assert conn.poll(120),"Author timeout"
        r=conn.recv()
        if r["type"]=="error":raise RuntimeError(r["error"])
        return r
    def replay(amplitude=0.,sign=1):
        conn.send(dict(op="replay",arm="source32",index=index,state=state))
        perturbation=None
        while True:
            r=receive()
            if r["type"]!="random":return r,perturbation
            identity=(r["kind"],r["dtype"],r["index"],r["time"],r["iteration"])
            if identity not in cache:
                value=base(r["kind"],r["dtype"],r["index"],r["time"],r["iteration"])
                cache[identity]=value
            value=cache[identity]
            common="|".join(map(str,(r["kind"],r["index"],r["time"],r["iteration"])))
            check("common_base/"+common,hashlib.sha256(tf.cast(value,tf.float64).numpy().tobytes()).hexdigest()==hashes[common])
            if event is not None and r["kind"]=="categorical" and r["time"]==event[0]:
                row,slot=event[1:]
                before=float(value[row,slot])
                after=tf.constant(before+sign*amplitude,tf.float32)
                value=tf.tensor_scatter_nd_update(value,[[row,slot]],[after])
                perturbation=dict(requested=sign*amplitude,realized=float(after)-before,before=before,after=float(after))
                check("perturbation_domain",0.<float(after)<1.)
            conn.send(dict(type="random",bytes=value.numpy().tobytes(),dtype=value.dtype.name,shape=value.shape.as_list()))
    try:
        conn=listener.accept();metadata=receive();check("source_cpu_reference",metadata["cpu_only"])
        conn.send(dict(op="setup",role="unit01-tf"));setup=receive();check("source_setup",setup["type"]=="setup")
        repeated,_=replay()
        check("author_exact_saved_replay",all(repeated[k]==source[k] for k in
            ("loss","gradients","parameters","moments","captured","data","before","after")))
        if event is not None:
            for amplitude in (0.,1e-8,1e-7,1e-6,1e-5):
                for sign in ((1,) if amplitude==0. else (-1,1)):
                    perturbed,perturbation=replay(amplitude,sign)
                    comparison=compare(perturbed,source,"float32")
                    row=dict(amplitude=amplitude,sign=sign,perturbation=perturbation,comparison=comparison)
                    source_cases.append(row)
                    write(out/("source-perturb-%02d.json"%len(source_cases)),row)
                    # Full tensor evidence for every perturbation; retain zero effects.
                    save_gzip(out/("source-perturb-%02d-tensors.json.gz"%len(source_cases)),perturbed)
        conn.send(dict(op="stop"));closed=receive();check("source_closed",closed["type"]=="closed")
        proc.wait(timeout=30);check("source_exit",proc.returncode==0)
        write(out/"source-provenance.json",closed)
        traces={**{"key/"+k:f.experimental_get_tracing_count() for k,f in key_functions.items()},
                **{"random/"+"/".join(k):f.experimental_get_tracing_count() for k,f in random_functions.items()},
                **{"categorical/"+k:f.experimental_get_tracing_count() for k,f in categorical_functions.items()}}
        check("bounded_helper_traces",all(n<=1 for n in traces.values()))
        write(out/"checks.json",controls)
        write(out/"environment.json",dict(tf=tf.__version__,author=metadata,memory_growth=memory,
            devices=[x.name for x in tf.config.list_logical_devices("GPU")],tf32=False,jit_compile=True,
            helper_traces=traces,allocator=tf.config.experimental.get_memory_info("GPU:0"),
            case_path=str(case_path),case_sha256=sha(case_path)))
        write(out/"result.json",dict(status="COMPLETED",scope="SAVED_CASE_DIAGNOSTIC_ONLY",
            replication="INCOMPLETE",checks=len(controls),passed=sum(c["passed"] for c in controls),
            capture_change=capture_change,event=event,boundary=boundary,comparisons=interventions,
            source_perturbations=source_cases,wall_seconds=time.monotonic()-started))
        print(json.dumps(dict(status="COMPLETED",output=str(out),event=event,checks=len(controls))),flush=True)
    finally:
        tf.random.stateless_normal=native_normal;tf.random.stateless_uniform=native_uniform
        if proc.poll() is None:proc.terminate();proc.wait(timeout=15)
        if conn:conn.close()
        listener.close();log.close();temp.cleanup()
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);main(p.parse_args().output)
