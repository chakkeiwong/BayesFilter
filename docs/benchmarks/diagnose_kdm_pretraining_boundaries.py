"""Diagnostic only: author/TF pretraining parity at saved checkpoint boundaries."""
import argparse, gzip, json, math, os, subprocess, sys, tempfile, time
from pathlib import Path
from multiprocessing.connection import Listener
ROOT=Path("/home/chakwong/BayesFilter")
sys.path.insert(0,str(ROOT/"docs/benchmarks"))
from diagnose_kdm_fresh_pretraining import read, write, save, digest, SEED

def main(out):
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]="true"
    os.environ["BAYESFILTER_PRELOAD_CUSTOM_OP"]="0"
    sys.path.insert(0,str(ROOT))
    import tensorflow as tf
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    memory=configure_tensorflow_gpu_memory_growth(tf,require_gpu=True)
    tf.config.experimental.enable_tensor_float_32_execution(False)
    tf.config.set_soft_device_placement(False)
    from bayesfilter.replication.kdm_neural_replay_tf import BearingsNeuralReplay
    from bayesfilter.replication.kdm_source_stages_tf import SourceWeightPretraining
    out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    def host(x):
        if tf.is_tensor(x):return x.numpy().tolist()
        if isinstance(x,dict):return {k:host(v) for k,v in x.items()}
        if isinstance(x,(list,tuple)):return [host(v) for v in x]
        return x
    def difference(a,b,dtype):
        a=tf.reshape(tf.cast(a,tf.float64) if tf.is_tensor(a) else tf.constant(a,tf.float64),[-1])
        b=tf.reshape(tf.constant(b,tf.float64),[-1]);assert a.shape==b.shape
        assert bool(tf.reduce_all(tf.math.is_finite(a))&tf.reduce_all(tf.math.is_finite(b)))
        tol=2e-5 if dtype=="float32" else 2e-9
        return dict(max_abs=float(tf.reduce_max(tf.abs(a-b))),max_scaled=float(tf.reduce_max(tf.abs(a-b)/(tol*(1+tf.abs(b))))))
    def group(a,b,dtype):
        assert set(a)==set(b)
        d={n:difference(a[n],b[n],dtype) for n in a}
        return dict(max_abs=max(v["max_abs"] for v in d.values()),max_scaled=max(v["max_scaled"] for v in d.values()),
            flagged=[n for n,v in d.items() if v["max_scaled"]>1])
    def compare(result,reference,model,dtype):
        o=result["outputs"];c=reference["captured"]
        # Author encoders flatten the leading batch/particle dimensions; reshape
        # only at this diagnostic boundary, not in either numerical consumer.
        obs=tf.reshape(tf.constant(c["observation_features"],model.dtype),[-1,8])
        obs=tf.broadcast_to(obs[:,None,:],[len(reference["inputs"]["cloud"]),50,8])
        values={"loss":difference(result["loss"],reference["loss"],dtype)}
        for n in ("particle_features","logits","particle_weights","nll"):
            values[n]=difference(o[n],c[n],dtype)
        values["observation_features"]=difference(o["observation_features"],host(obs),dtype)
        batch=reference["inputs"]["data"]
        selected_states=[batch["states"][i][t] for t in range(0,17,4) for i in range(len(batch["states"]))]
        selected_obs=[batch["observations"][i][t] for t in range(0,17,4) for i in range(len(batch["states"]))]
        assert host(o["selected_states"])==selected_states and host(o["selected_observations"])==selected_obs
        if "gradients" in reference:
            values["gradients"]=group(dict(zip(model.names,result["raw_gradients"])),reference["gradients"],dtype)
            values["optimizer_gradients"]=group(dict(zip(model.names,result["optimizer_gradients"])),reference["optimizer_gradients"],dtype)
            for field,key in (("parameters_after","parameters"),("first_moments","first_moments"),("second_moments","second_moments")):
                expected={n:reference["state"][key][n] for n in model.names}
                values[key]=group(dict(zip(model.names,result[field])),expected,dtype)
        return values
    checks=[];records=[];aggregates=[];traces={};precision_queue=[];known_inputs={}
    def check(name,value):
        checks.append(dict(name=name,passed=bool(value)))
        if not value:raise AssertionError(name)
    with tf.device("/GPU:0"):
        probe=tf.matmul(tf.ones([2,2]),tf.eye(2))
    check("framework_probe_gpu", "GPU" in probe.device)
    tmp=tempfile.TemporaryDirectory(prefix="kdm-boundaries-",dir="/tmp")
    listener=Listener(str(Path(tmp.name)/"source.sock"),family="AF_UNIX")
    listener._listener._socket.settimeout(90)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES="-1",LD_LIBRARY_PATH="/home/chakwong/anaconda3/envs/tf-gpu2/lib")
    log=(out/"author.log").open("w")
    proc=subprocess.Popen(["/home/chakwong/anaconda3/envs/tf-gpu2/bin/python","-I","-B",
        str(ROOT/"docs/benchmarks/diagnose_kdm_fresh_pretraining.py"),"--socket",str(Path(tmp.name)/"source.sock")],
        cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    conn=None
    def receive():
        assert conn.poll(180),"Author reference timeout"
        result=conn.recv()
        if result["type"]=="error":raise RuntimeError(result["error"])
        return result
    try:
        conn=listener.accept();ready=receive()
        write(out/"environment.json",dict(source=ready,tensorflow=tf.__version__,
            gpu=[x.name for x in tf.config.list_logical_devices("GPU")],probe_device=probe.device,
            memory_growth=memory,tf32=False,jit_compile=True,seed=SEED,
            cpu_reference_gpu_intentionally_hidden=True))
        def run_cases(unit,checkpoint,dtype,cases,model=None,stages=None):
            conn.send(dict(op="setup",unit=unit,dtype=dtype,checkpoint=checkpoint))
            setup=receive()
            label=unit+"-"+checkpoint+"-"+dtype
            save(out/(label+"-setup.json.gz"),setup)
            if model is None:
                with tf.device("/GPU:0"):
                    model=BearingsNeuralReplay(setup["fixture"],batch_size=64)
            if stages is None:stages={}
            totals=[];row_ids=[]
            for split,index in cases:
                name=label+"-"+split+"-"+str(index)
                conn.send(dict(op="replay" if split=="training" else "evaluate",split=split,index=index))
                r=receive();check(name+"/source_step",r["type"]=="step")
                check(name+"/input_hash",digest(r["inputs"])==r["input_hash"])
                identity=(unit,checkpoint,split,index)
                if dtype=="float32":known_inputs[identity]=r["input_hash"]
                else:check(name+"/precision_inputs",known_inputs[identity]==r["input_hash"])
                save(out/(name+"-source.json.gz"),r)
                b=len(r["inputs"]["rows"])
                check(name+"/effective_count",r["effective_batch"]==b*5)
                with tf.device("/GPU:0"):
                    model.load_state_dict(r["before"])
                    state=model.state_dict()
                    for key in ("parameters","first_moments","second_moments","rates","step"):
                        check(name+"/restored_"+key,state[key]==r["before"][key])
                    if b not in stages:stages[b]=SourceWeightPretraining(model,b)
                    stage=stages[b];data=r["inputs"]["data"]
                    inputs=[tf.constant(data[k],tf.bool if k=="ground_truth_mask" else model.dtype)
                            for k in ("states","observations","ground_truth_mask")]
                    inputs.append(tf.constant(r["inputs"]["cloud"],model.dtype))
                    function=stage.update if split=="training" else stage.evaluate
                    result=function(*inputs)
                    check(name+"/finite_valid",bool(result["valid"]))
                    if split=="training":check(name+"/committed",bool(result["committed"]))
                    else:check(name+"/validation_no_mutation",state==model.state_dict())
                    comparisons=compare(result,r,model,dtype)
                    if split=="training" and index==0:
                        after=model.state_dict();first=host(result)
                        model.load_state_dict(state);repeat=function(*inputs)
                        check(name+"/exact_repeat",after==model.state_dict() and first==host(repeat))
                        check(name+"/author_instrumentation",r["uninstrumented_control"])
                    save(out/(name+"-tf.json.gz"),dict(result=host(result),state=model.state_dict()))
                flags=[key for key,value in comparisons.items() if value["max_scaled"]>1]
                record=dict(unit=unit,checkpoint=checkpoint,dtype=dtype,split=split,index=index,
                    rows=b,adam_step=r["before"]["step"],input_hash=r["input_hash"],
                    comparisons=comparisons,flags=flags)
                records.append(record)
                with (out/"steps.jsonl").open("a") as f:f.write(json.dumps(record,allow_nan=False)+"\n")
                if flags and dtype=="float32":precision_queue.append((unit,checkpoint,split,index))
                if split=="validation":
                    totals.append((r["effective_batch"],float(result["loss"]),r["loss"]))
                    row_ids.extend(r["inputs"]["rows"])
                print(json.dumps(dict(case=name,flags=flags)),flush=True)
            if len(totals)==4:
                check(label+"/validation_coverage",row_ids==list(range(1000)))
                count=sum(n for n,_,_ in totals)
                got=math.fsum(n*x for n,x,_ in totals)/count
                expected=math.fsum(n*x for n,_,x in totals)/count
                aggregates.append(dict(unit=unit,checkpoint=checkpoint,dtype=dtype,
                    effective_count=count,tf=got,author=expected,abs_difference=abs(got-expected)))
            for b,stage in stages.items():
                for kind in ("update","evaluate"):
                    n=getattr(stage,kind).experimental_get_tracing_count()
                    if n:
                        traces[unit+"/"+dtype+"/"+str(b)+"/"+kind]=n
                        check(label+"/trace/"+str(b)+"/"+kind,n==1)
            return model,stages
        for unit in ("unit01","unit02"):
            model=None;stages=None
            for checkpoint in ("tf_best","author_best"):
                model,stages=run_cases(unit,checkpoint,"float32",
                    [("training",0),("training",78)]+[("validation",i) for i in range(4)],model,stages)
        for unit,checkpoint in sorted({(u,c) for u,c,_,_ in precision_queue}):
            cases=[(s,i) for u,c,s,i in precision_queue if (u,c)==(unit,checkpoint)]
            run_cases(unit,checkpoint,"float64",cases)
        conn.send(dict(op="stop"));closed=receive()
        check("author_source_closure",closed["type"]=="closed" and bool(closed["author_hashes"]))
        write(out/"source-provenance.json",closed)
        proc.wait(timeout=30);check("author_exit",proc.returncode==0)
        flagged=[r for r in records if r["flags"]]
        result=dict(schema="kdm_pretraining_boundary_diagnostic_v1",
            status="LOCALIZATION_REQUIRED" if flagged else "PASS",checks=checks,records=records,
            validation_aggregates=aggregates,trace_counts=traces,
            memory=tf.config.experimental.get_memory_info("GPU:0"),wall_seconds=time.monotonic()-started,
            replication="INCOMPLETE",scope="SAVED_PRETRAINING_BOUNDARIES_ONLY",
            flags_are_replication_veto=False,statistical_ranking="NOT_TESTED",
            checkpoint_roles=dict(tf_best="saved parameters and full saved Adam state",
                author_best="author selected parameters with fresh diagnostic Adam state"))
        write(out/"result.json",result)
        print(json.dumps(dict(status=result["status"],checks=len(checks),cases=len(records),
            flags=len(flagged),wall_seconds=result["wall_seconds"])),flush=True)
    finally:
        if conn is not None:conn.close()
        listener.close()
        if proc.poll() is None:
            proc.terminate()
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()
        log.close();tmp.cleanup()

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--output",type=Path,required=True)
    main(parser.parse_args().output)
