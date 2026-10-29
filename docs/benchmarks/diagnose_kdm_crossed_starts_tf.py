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
def make_observed_coupled_sampler(tf, sampler):
    """Diagnostic inverse-CDF sampler retaining the choices consumed by XLA.

    Called inside SourceFullTraining's stable compiled update/evaluation graph.
    This changes only the diagnostic sampler; native categorical RNG is separate.
    """
    def sample(p,w,bw,seed):
        key=lambda tag:tf.random.experimental.stateless_fold_in(seed,tag)
        u=tf.random.stateless_uniform([64,25],key(0),dtype=sampler.dtype,alg="philox")
        cdf=tf.cumsum(w/tf.reduce_sum(w,-1,keepdims=True),-1)
        ancestors=tf.minimum(tf.reduce_sum(tf.cast(u[...,None]>=cdf[:,None,:],tf.int32),-1),24)
        z=tf.random.stateless_normal(sampler.shape+[2],key(1),dtype=sampler.dtype,alg="philox")
        normal=p[None,:,:,:2]+bw[None,:,None,:2]*z
        def uniform(i):
            return tf.random.stateless_uniform([3]+sampler.shape,key(i+2),dtype=tf.float64,alg="philox")
        result=sampler._assemble(p,w,bw,ancestors,normal,uniform)
        return dict(result,categorical_cdf=cdf,categorical_uniform=u)
    return sample

def main(out):
    out.mkdir(exist_ok=False)
    # The controller replay is independent and runs before expensive GPU work.
    env=dict(os.environ,CUDA_VISIBLE_DEVICES="-1",LD_LIBRARY_PATH="/home/chakwong/anaconda3/envs/tf-gpu2/lib")
    # Replay actual saved histories in the independent CPU-only source env.
    controller_out=out/"controller"
    controller_env=dict(env,CUDA_VISIBLE_DEVICES="-1")
    with (out/"controller.log").open("w") as controller_log:
        subprocess.run(["/home/chakwong/anaconda3/envs/tf-gpu2/bin/python","-I","-B",
            str(ROOT/"docs/benchmarks/diagnose_kdm_validation_control.py"),"--output",str(controller_out)],
            env=controller_env,cwd=ROOT,stdout=controller_log,stderr=subprocess.STDOUT,check=True,timeout=300)
    controller=json.loads((controller_out/"result.json").read_text())
    assert controller["status"]=="PASS"
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
    def coupled_sample(sampler):
        return make_observed_coupled_sampler(tf,sampler)
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
        # Consume the choices returned from this update, never recompute them.
        assert len(o)==12,"Missing in-kernel sampling diagnostics"
        ancestors=host(tf.cast(o[9][1:],tf.int32))
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
    controls=[];records=[];provenance=[];started=time.monotonic()
    temp=tempfile.TemporaryDirectory(prefix="kdm-coupled-",dir="/tmp")
    socket=str(Path(temp.name)/"reference.sock");listener=Listener(socket,family="AF_UNIX")
    log=(out/"author.log").open("w")
    env=dict(os.environ,CUDA_VISIBLE_DEVICES="-1")
    # Remove TF-specific library lookup from the independent author environment.
    env["LD_LIBRARY_PATH"]="/home/chakwong/anaconda3/envs/tf-gpu2/lib"
    proc=subprocess.Popen(["/home/chakwong/anaconda3/envs/tf-gpu2/bin/python","-I","-B",
        str(ROOT/"docs/benchmarks/reference_kdm_crossed_starts.py"),"--socket",socket],
        cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    listener._listener._socket.settimeout(60)
    conn=None;cache={};random_hashes={};evaluations=[]
    def receive():
        assert conn.poll(120),"Author response timeout"
        r=conn.recv()
        if r["type"]=="error":raise RuntimeError(r["error"])
        return r
    def author_update(arm,index,op="update",state=None,perturbation=None):
        message=dict(op=op,arm=arm,index=index)
        if state is not None:message["state"]=state
        if perturbation is not None:message["perturbation"]=perturbation
        conn.send(message)
        while True:
            r=receive()
            if r["type"]!="random":return r
            identity=(r["kind"],r["dtype"],r["index"],r["time"],r["iteration"])
            if identity not in cache:
                value=base(r["kind"],r["dtype"],r["index"],r["time"],r["iteration"])
                # Serialization only; no NumPy numerical path.
                cache[identity]=dict(type="random",bytes=value.numpy().tobytes(),dtype=value.dtype.name,shape=value.shape.as_list())
                h=hashlib.sha256(tf.cast(value,tf.float64).numpy().tobytes()).hexdigest()
                common=(r["kind"],r["index"],r["time"],r["iteration"])
                if common in random_hashes:assert random_hashes[common]==h,"Precision base arrays differ"
                else:random_hashes[common]=h
            conn.send(cache[identity])
    try:
        conn=listener.accept();metadata=receive();assert metadata["type"]=="ready";write(out/"environment.json",dict(author=metadata,
            tf=tf.__version__,gpu_devices=[d.name for d in tf.config.list_logical_devices("GPU")],
            memory_growth=memory,tf32=False,jit_compile=True,seed=SEED))
        for role in ("unit01-author","unit01-tf","unit02-author","unit02-tf",
                     "unit01-authorbest","unit01-tfbest","unit02-authorbest","unit02-tfbest"):
            cache.clear();first_flag={};first_any=None
            conn.send(dict(op="setup",role=role));setup=receive();assert setup["type"]=="setup"
            provenance.append({k:setup[k] for k in ("path","sha256")})
            state=setup["state"];models={};stages={};sampler_checks={}
            for dtype in (tf.float32,tf.float64):
                fixture=dict(variant="mdpf",dtype=dtype.name,batch_size=64,parameters=state["parameters"],
                    trainable_names=state["trainable_names"],frozen_names=[n for n in state["parameters"] if n not in state["trainable_names"]],
                    learning_rates=state["rates"],constants=setup["constants"])
                model=BearingsNeuralReplay(fixture,capture_sampling_diagnostics=True);converted=copy.deepcopy(state);converted["dtype"]=dtype.name
                model.load_state_dict(converted)
                restored=model.state_dict()
                for key in ("parameters","first_moments","second_moments","step","rates"):
                    assert restored[key]==state[key],(role,dtype.name,key)
                model.sampler._sample=coupled_sample(model.sampler)
                stage=SourceFullTraining(model);models[dtype.name]=model;stages[dtype.name]=stage
                sampler_checks[dtype.name]=tf.function(model.sampler._sample,input_signature=[
                    tf.TensorSpec([64,25,3],dtype),tf.TensorSpec([64,25],dtype),tf.TensorSpec([64,3],dtype),
                    tf.TensorSpec([2],tf.int32)],jit_compile=True,autograph=False)
            # Fixed development panel, eight common draws; descriptive only.
            for draw in range(8):
                index=10000+draw
                for dtype in (tf.float32,tf.float64):
                    a=author_update("source32" if dtype==tf.float32 else "source64",index,op="evaluate")
                    model=models[dtype.name];before=model.state_dict()
                    args=[tf.constant(a["data"][k],tf.bool if k=="ground_truth_mask" else dtype) for k in ("states","observations","ground_truth_mask")]
                    result=stages[dtype.name].evaluate(*args,tf.constant([SEED,index],tf.int64))
                    assert bool(result["valid"]) and before==model.state_dict()
                    check=difference(result["loss"],a["loss"],dtype.name)
                    predictions={}
                    for engine in ("source","tf"):
                        values=[]
                        for j in (3,7,11,15):
                            if engine=="source":
                                particles=tf.constant(a["captured"]["outputs"][j]["particles"],dtype)
                                weights=tf.constant(a["captured"]["outputs"][j]["particle_weights"],dtype)
                            else:particles=result["outputs"][0][j];weights=result["outputs"][1][j]
                            values.append(tf.reduce_sum(particles[...,:2]*weights[...,None],axis=1))
                        prediction=tf.stack(values,axis=1)
                        truth=tf.gather(tf.constant(a["data"]["states"],dtype),[4,8,12,16],axis=1)[...,:2]
                        predictions[engine]=dict(xy_rmse=float(tf.sqrt(tf.reduce_mean(tf.reduce_sum(tf.square(prediction-truth),axis=-1)))),
                            posterior_xy=host(prediction))
                    item=dict(role=role,draw=draw,dtype=dtype.name,rows=64,source_loss=a["loss"],
                        tf_loss=float(result["loss"]),comparison=check,state_unchanged=True,
                        predictions=predictions,rmse_scope="development_xy_times_4_8_12_16")
                    evaluations.append(item)
                    with (out/"evaluation.jsonl").open("a") as f:f.write(json.dumps(item)+"\n")
                    write(out/"progress.json",dict(role=role,evaluation_draw=draw,dtype=dtype.name,wall_seconds=time.monotonic()-started))
            if role.endswith("best"):
                assert all(stage.evaluate.experimental_get_tracing_count()==1 for stage in stages.values())
                continue
            for index in range(16):
                cache.clear();arms={}
                for arm in ("source32","source64")+(("repeat32",) if index==0 else ()):
                    arms[arm]=author_update(arm,index)
                if index==0:
                    a,b=arms["source32"],arms["repeat32"]
                    fields=("loss","gradient_norm","gradients","parameters","moments","captured")
                    assert all(a[k]==b[k] for k in fields),"Source repeat mismatch"
                    controls.append(dict(name=role+"/source_repeat",pass_check=True))
                    for dtype in (tf.float32,tf.float64):
                        a=arms["source32" if dtype==tf.float32 else "source64"]
                        for j,inputs in enumerate(a["captured"]["sampler_inputs"]):
                            seed=base_key("resampler",0,j+2)
                            result=sampler_checks[dtype.name](*[tf.constant(inputs[k],dtype) for k in ("p","w","bw")],seed)
                            expected=tf.constant(a["captured"]["samples"][j],dtype)
                            d=difference(result["samples"][...,:2],expected[...,:2],dtype.name)
                            angle=difference(result["samples"][...,2],expected[...,2],dtype.name,True)
                            agree=bool(result["valid"]) and int(result["iterations"])==a["captured"]["iterations"][j] and host(result["ancestors"])==a["captured"]["ancestors"][j]
                            controls.append(dict(name=f"{role}/{dtype.name}/sampler/{j+2}",pass_check=agree and max(d["max_scaled"],angle["max_scaled"])<=1,
                                position=d,angle=angle,iterations=int(result["iterations"])))
                        inputs=a["captured"]["sampler_inputs"][0];seed=base_key("resampler",0,2)
                        p0,w0,b0=[tf.constant(inputs[k],dtype) for k in ("p","w","bw")]
                        original=sampler_checks[dtype.name](p0,w0,b0,seed)
                        shift=tf.constant([.25,0.,0.],dtype)
                        changed=sampler_checks[dtype.name](p0+shift,w0,b0,seed)
                        delta=difference(changed["samples"],original["samples"]+shift,dtype.name)
                        controls.append(dict(name=role+"/"+dtype.name+"/own_state_response",
                            pass_check=bool(changed["valid"]) and delta["max_scaled"]<=1,
                            difference=delta))
                        def initial_control(truth,root):
                            return stages[dtype.name].initial._sample(truth,fold(root,0))
                        initial_check=tf.function(initial_control,input_signature=[tf.TensorSpec([64,3],dtype),tf.TensorSpec([2],tf.int64)],jit_compile=True,autograph=False)
                        init=initial_check(tf.constant(a["data"]["states"],dtype)[:,0,:],tf.constant([SEED,0],tf.int64))
                        d=difference(init["particles"],a["captured"]["initial"]["particles"],dtype.name)
                        controls.append(dict(name=role+"/"+dtype.name+"/initial",pass_check=bool(init["valid"]) and d["max_scaled"]<=1,difference=d))
                    write(out/"adapter-controls.json",controls)
                    assert all(x["pass_check"] for x in controls),"Adapter controls failed"
                for dtype in (tf.float32,tf.float64):
                    a=arms["source32" if dtype==tf.float32 else "source64"]
                    args=[tf.constant(a["data"][k],tf.bool if k=="ground_truth_mask" else dtype) for k in ("states","observations","ground_truth_mask")]
                    tf_before=models[dtype.name].state_dict()
                    result=stages[dtype.name].update(*args,tf.constant([SEED,index],tf.int64))
                    assert bool(result["valid"]) and bool(result["committed"])
                    assert "GPU:0" in result["loss"].device,result["loss"].device
                    record_tf=tf_record(models[dtype.name],result,index)
                    record_tf["before"]=tf_before
                    record_tf["after"]=models[dtype.name].state_dict()
                    # Preserve actual uniform/CDF margins, not just ancestors.
                    cdf_records=[]
                    for j,t in enumerate(range(2,17)):
                        u=result["outputs"][11][t-1]
                        cdf=result["outputs"][10][t-1]
                        ancestors=tf.constant(record_tf["captured"]["ancestors"][j],tf.int32)
                        lower=tf.gather(tf.concat([tf.zeros_like(cdf[:,:1]),cdf[:,:-1]],-1),ancestors,batch_dims=1)
                        upper=tf.gather(cdf,ancestors,batch_dims=1)
                        assert bool(tf.reduce_all((u>=lower)&((u<upper)|(ancestors==24)))),"Consumed CDF interval mismatch"
                        cdf_records.append(dict(cdf=host(cdf),uniforms=host(u),
                            provenance="consumed_inside_update",
                            min_margin=float(tf.reduce_min(tf.minimum(tf.abs(u-lower),tf.abs(upper-u))))))
                    record_tf["captured"]["cdf_diagnostics"]=cdf_records
                    arms["tf32" if dtype==tf.float32 else "tf64"]=record_tf
                comparisons={}
                for label,a,b,dtype in (("tf32_source32","tf32","source32","float32"),
                    ("tf64_source64","tf64","source64","float64")):
                    comparisons[label]=compare(arms[a],arms[b],dtype)
                record=dict(role=role,update=index,comparisons=comparisons,
                    losses={k:v["loss"] for k,v in arms.items()})
                for dtype in (tf.float32,tf.float64):
                    label="tf32_source32" if dtype==tf.float32 else "tf64_source64"
                    flag=flags(comparisons[label])
                    if flag and dtype.name not in first_flag:
                        source_arm=arms["source32" if dtype==tf.float32 else "source64"]
                        model=models[dtype.name];live=model.state_dict()
                        model.load_state_dict(source_arm["before"])
                        args=[tf.constant(source_arm["data"][k],tf.bool if k=="ground_truth_mask" else dtype) for k in ("states","observations","ground_truth_mask")]
                        replay=stages[dtype.name].update(*args,tf.constant([SEED,index],tf.int64))
                        assert bool(replay["valid"]) and bool(replay["committed"])
                        same=compare(tf_record(model,replay,index),source_arm,dtype.name)
                        model.load_state_dict(live)
                        first_flag[dtype.name]=dict(update=index,flags=flag,same_state=same)
                        source_repeat=author_update("source32" if dtype==tf.float32 else "source64",index,
                            op="replay",state=source_arm["before"])
                        assert all(source_repeat[k]==source_arm[k] for k in ("loss","gradients","parameters","moments","captured"))
                        save_gzip(out/(role+"-"+dtype.name+"-first-flag.json.gz"),
                            dict(source=source_arm,tf=arms["tf32" if dtype==tf.float32 else "tf64"],same_state=host(replay),comparison=same))
                        record.setdefault("first_flags",{})[dtype.name]=first_flag[dtype.name]
                        if first_any is None:first_any=(index,source_arm)

                records.append(record);write(out/f"{role}-update{index:02d}.json",record)
                if index in (0,15):
                    save_gzip(out/f"{role}-tensors{index:02d}.json.gz",{k:{f:v[f] for f in ("parameters","moments","captured","before","after")} for k,v in arms.items() if k!="repeat32"})
                write(out/"progress.json",dict(role=role,completed_updates=index+1,wall_seconds=time.monotonic()-started))
                print(json.dumps(dict(role=role,update=index,loss_delta=comparisons["tf32_source32"]["loss_difference"],
                    ancestor_differences=sum(x["disagreements"] for x in comparisons["tf32_source32"]["ancestors"]))),flush=True)
            write(out/(role+"-first-flags.json"),first_flag)
            for dtype in (tf.float32,tf.float64):
                assert stages[dtype.name].evaluate.experimental_get_tracing_count()==1
                assert stages[dtype.name].update.experimental_get_tracing_count()==1
                write(out/(role+"-"+dtype.name+"-final.json"),models[dtype.name].state_dict())
        conn.send(dict(op="stop"));closed=receive();assert closed["type"]=="closed"
        proc.wait(timeout=30);assert proc.returncode==0
        write(out/"author-provenance.json",closed)
        write(out/"random-hashes.json",{"|".join(map(str,k)):v for k,v in random_hashes.items()})
        write(out/"result.json",dict(status="COMPLETED",scope="DIAGNOSTIC_ONLY",replication="INCOMPLETE",
            adapter_checks=len(controls),adapter_passes=sum(x["pass_check"] for x in controls),
            updates=len(records),evaluations=len(evaluations),controller_checks=len(controller["checks"]),
            controller_status=controller["status"],provenance=provenance,wall_seconds=time.monotonic()-started,
            training_trace_count_per_signature=1,allocator=tf.config.experimental.get_memory_info("GPU:0"),
            common_realized_particles=False,coupling="common base variates, own evolving state"))
    finally:
        tf.random.stateless_normal=native_normal;tf.random.stateless_uniform=native_uniform
        if proc.poll() is None:proc.terminate();proc.wait(timeout=15)
        if conn:conn.close()
        listener.close();log.close();temp.cleanup()
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);main(p.parse_args().output)
