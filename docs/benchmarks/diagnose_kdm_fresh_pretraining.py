"""Fresh-constructor source/TF diagnostic only; CPU author and GPU/XLA TF."""
import argparse, copy, gzip, hashlib, json, os, subprocess, sys, tempfile, time, traceback
from pathlib import Path
from multiprocessing.connection import Client, Listener
from types import SimpleNamespace, MethodType
ROOT=Path("/home/chakwong/BayesFilter")
BASE=ROOT/"docs/plans/artifacts/controlled-smc-kdm-replication-20260928"
SOURCE=BASE/"kdm-source-variability-20261008-01"
CAL=BASE/"paired-calibration-20261011/calibration-01"
SEED=202610123
def read(p): return json.loads(p.read_text())
def write(p,x): p.write_text(json.dumps(x,allow_nan=False)+"\n")
def digest(x): return hashlib.sha256(json.dumps(x,sort_keys=True,allow_nan=False).encode()).hexdigest()
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):
    with gzip.open(p,"wt",compresslevel=1) as f: json.dump(x,f,allow_nan=False)
def source_worker(socket):
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"
    sys.path.insert(0,str(ROOT/"docs/benchmarks"))
    import reference_kdm_source_stages as s
    ref,torch=s.ref,s.torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    wp=ref.AUTHOR/"src/trainers/specific_trainers/weight_model_trainer.py"
    forward=s.member(wp,"WeightModelTrainer","do_forward_pass")
    masked=s.member(wp,"WeightModelTrainer","get_true_masked_data")
    cloudcls=s.extract_clouds(wp)
    observation=s.member(ref.AUTHOR/"src/problems/bearings_only/bearings_only_problem.py","BearingsOnlyObservationTransformer")
    conn=Client(socket,family="AF_UNIX")
    conn.send(dict(type="ready",torch=torch.__version__,source_commit=ref.PIN,cpu_only=True,gpu_intentionally_hidden=True))
    source_hashes={};data=None
    try:
        while True:
            cmd=conn.recv()
            if cmd["op"]=="stop": break
            if cmd["op"]=="setup":
                unit,dtype_name=cmd["unit"],cmd["dtype"];dtype=getattr(torch,dtype_name)
                root=SOURCE/(unit+"-full")/"diagnostic"
                cfg=next(x for x in read(root/"configurations.json")["executed_stages"] if x["name"]=="weights")
                fixture=read(root/"weights-constructor.json")
                peer=read(CAL/unit/"training/diagnostic/weights-starting-state.json")
                assert fixture["parameters"]==peer["parameters"],"Fresh constructors differ"
                fixture["dtype"]=dtype_name
                for path in (root/"configurations.json",root/"weights-constructor.json",CAL/unit/"training/diagnostic/weights-starting-state.json"):
                    source_hashes[str(path)]=sha(path)
                native_config=ref.yaml.safe_load(ref.verify(ref.EXPERIMENT/"individual_training/config.yaml"))
                native_spec=next(v for e in native_config["experiments"] for n,v in e.items() if n=="full_dpf_weight_training_fixed_bands")
                architecture=ref.yaml.safe_load(ref.verify(ref.EXPERIMENT/"configs/models.yaml"))["models"]
                architecture.update(native_config["models"])
                assert json.loads(json.dumps(architecture))==cfg["architecture"]
                assert json.loads(json.dumps(native_spec["model_params"]))==cfg["configuration"]["model_params"]
                assert all(type(k) is int for k in architecture["dpf_training"]["kde"]["dims"])
                torch.set_default_dtype(torch.float32)
                model=ref.KDEParticleFilter(native_spec["model_params"],architecture).to(dtype=dtype)
                for module in model.modules():
                    for attr in ("min_bandwidths","max_bandwidths","residual_scale_factor"):
                        v=getattr(module,attr,None)
                        if torch.is_tensor(v):setattr(module,attr,v.to(dtype))
                params=dict(model.named_parameters())
                with torch.no_grad():
                    for n,p in params.items():p.copy_(torch.tensor(fixture["parameters"][n],dtype=dtype))
                for p in params.values():p.requires_grad_(False)
                native_config=ref.yaml.safe_load(ref.verify(ref.EXPERIMENT/"individual_training/config.yaml"))
                native_spec=next(v for e in native_config["experiments"] for n,v in e.items() if n=="full_dpf_weight_training_fixed_bands")
                training=native_spec["training_params"]
                saved_training=copy.deepcopy(cfg["configuration"]["training_params"])
                assert saved_training.pop("num_cpu_cores_for_dataloader",None)==1
                assert json.loads(json.dumps(training))==saved_training,"Source numerical pretraining config differs"
                class Collector:
                    def __init__(self):self.optimizers=[];self.rates={}
                    def add_optimizer_and_lr_scheduler(self,modules,names,lr):
                        if lr=="Freeze":return
                        group=[p for module in modules for p in module.parameters()]
                        for p in group:p.requires_grad_(True)
                        self.optimizers.append(torch.optim.Adam(group,lr=lr,weight_decay=training["weight_decay"]))
                        for n,p in params.items():
                            if any(p is v for v in group):self.rates[n]=lr
                collector=Collector();model.create_and_add_optimizers(training,collector,"weight")
                names=[n for n,p in params.items() if p.requires_grad]
                assert names==fixture["trainable_names"] and collector.rates==fixture["learning_rates"]
                assert training["weight_decay"]==.001
                checkpoint=cmd.get("checkpoint","constructor")
                checkpoint_provenance=dict(kind=checkpoint)
                if checkpoint=="tf_best":
                    path=CAL/unit/"training/diagnostic/weights-best.json"
                    saved=read(path)["model"];source_hashes[str(path)]=sha(path)
                    with torch.no_grad():
                        for n,p in params.items():p.copy_(torch.tensor(saved["parameters"][n],dtype=dtype))
                    for opt in collector.optimizers:
                        group=opt.param_groups[0]
                        group_names=[n for n,p in params.items() if any(p is q for q in group["params"])]
                        group_rates={saved["rates"][n] for n in group_names}
                        assert len(group_rates)==1
                        group["lr"]=next(iter(group_rates))
                        for n in group_names:
                            opt.state[params[n]]=dict(step=torch.tensor(float(saved["step"]),dtype=torch.float32),
                                exp_avg=torch.tensor(saved["first_moments"][n],dtype=dtype),
                                exp_avg_sq=torch.tensor(saved["second_moments"][n],dtype=dtype))
                            collector.rates[n]=saved["rates"][n]
                    checkpoint_provenance.update(path=str(path),sha256=sha(path),optimizer="saved_TF_Adam_state",step=saved["step"])
                elif checkpoint=="author_best":
                    config=next(x for x in read(root/"configurations.json")["executed_stages"] if x["name"]=="mdpf")
                    settings=config["configuration"]["pre_trained_models"]
                    for path in settings.values():
                        source_hashes[path]=sha(Path(path))
                    model.load_pretrained(settings,torch.device("cpu"))
                    checkpoint_provenance.update(paths=settings,optimizer="fresh_Adam_diagnostic_only",step=0)
                else:assert checkpoint=="constructor"
                cloudgen=cloudcls(training["particle_cloud_params"])
                trainer=SimpleNamespace(main_model=model,device=torch.device("cpu"),
                    problem=SimpleNamespace(observation_transformer=observation()),
                    particle_cloud_generator=cloudgen,loss_function=ref.KDE_NLL(training["loss_params"],model))
                trainer.get_true_masked_data=MethodType(masked,trainer)
                model.train()
                datasets={}
                for split in ("training","validation"):
                    p=root/(split+".json");source_hashes[str(p)]=sha(p)
                    datasets[split]=read(p)["data"]
                gen=torch.Generator().manual_seed(SEED+int(unit[-2:]))
                order=torch.randperm(5000,generator=gen).tolist()
                conn.send(dict(type="setup",fixture=fixture,order=order,source_hashes=source_hashes,checkpoint=checkpoint_provenance))
                continue
            if cmd["op"] == "cloud_law":
                # Observe the actual native consumer without replacing its draws.
                index=cmd["index"];rows=list(range(index*256,min((index+1)*256,1000)))
                assert dtype_name=="float32" and rows
                batch={k:torch.tensor([v[i] for i in rows],dtype=torch.bool if k=="ground_truth_mask" else torch.float32)
                       for k,v in datasets["validation"].items()}
                selected=trainer.get_true_masked_data(batch)[0]
                model.eval();before={n:p.detach().clone() for n,p in params.items()}
                captures=[];output_cloud=[]
                original_sample=cloudgen.sample_from_dist
                original_loss=trainer.loss_function.compute_loss
                def observe_sample(*args,**kwargs):
                    value=original_sample(*args,**kwargs);captures.append(value.detach().clone());return value
                def observe_loss(output,truth):
                    output_cloud.append(output["particles"].detach().clone());return original_loss(output,truth)
                torch.manual_seed(cmd["seed"])
                cloudgen.sample_from_dist=observe_sample
                trainer.loss_function.compute_loss=observe_loss
                try:
                    with torch.no_grad():objective,effective=forward(trainer,batch,None,0)
                finally:
                    cloudgen.sample_from_dist=original_sample
                    trainer.loss_function.compute_loss=original_loss
                assert len(captures)==2 and len(output_cloud)==1
                assert bool(torch.isfinite(objective))
                unchanged=all(torch.equal(p,before[n]) for n,p in params.items())
                assert unchanged
                control=None
                if cmd.get("control",False):
                    torch.manual_seed(cmd["seed"])
                    with torch.no_grad():plain,count=forward(trainer,batch,None,0)
                    control=bool(torch.equal(objective,plain)) and count==effective
                    assert control and all(torch.equal(p,before[n]) for n,p in params.items())
                conn.send(dict(type="cloud_law",data=ref.host(batch),rows=rows,
                    truth=ref.host(selected),centers=ref.host(captures[0][:,0,:]),
                    particles=ref.host(output_cloud[0]),sampled_particles=ref.host(captures[1]),
                    loss=float(objective),effective_batch=effective,unchanged=unchanged,
                    instrumentation_control=control,seed=cmd["seed"]))
                continue
            assert cmd["op"] in ("update","evaluate","replay")
            split=cmd.get("split","training");index=cmd["index"];training_step=cmd["op"]!="evaluate"
            model.train(training_step)
            batchsize=64 if split=="training" else 256
            rows=order[index*batchsize:(index+1)*batchsize] if split=="training" else list(range(index*batchsize,min((index+1)*batchsize,1000)))
            torch.set_default_dtype(torch.float32)
            batch32={k:torch.tensor([v[i] for i in rows],dtype=torch.bool if k=="ground_truth_mask" else torch.float32) for k,v in datasets[split].items()}
            selected32=trainer.get_true_masked_data(batch32)[0]
            torch.manual_seed(SEED+int(unit[-2:])*1000+(100 if split=="validation" else 0)+index)
            cloud32=cloudgen.create_cloud(selected32)
            inputs=dict(data=ref.host(batch32),cloud=ref.host(cloud32),rows=rows)
            cloud=cloud32.to(dtype);batch={k:v.to(dtype) if k!="ground_truth_mask" else v for k,v in batch32.items()}
            torch.set_default_dtype(dtype)
            def state():
                moments={}
                optimizer_states={p:st for opt in collector.optimizers for p,st in opt.state.items()}
                for n in names:
                    p=params[n];st=optimizer_states.get(p,{})
                    moments[n]=dict(exp_avg=ref.host(st.get("exp_avg",torch.zeros_like(p))),
                        exp_avg_sq=ref.host(st.get("exp_avg_sq",torch.zeros_like(p))))
                steps={int(v["step"]) for opt in collector.optimizers for v in opt.state.values()}
                assert len(steps)<=1
                return dict(schema="kdm_neural_training_state_v2",adam_scalar_policy="pytorch_noncapturable_binary64_v1",variant=fixture["variant"],dtype=dtype_name,
                    trainable_names=names,constants={k:v for k,v in fixture["constants"].items() if v is not None},parameters=ref.host(params),
                    rates=collector.rates,first_moments={n:v["exp_avg"] for n,v in moments.items()},
                    second_moments={n:v["exp_avg_sq"] for n,v in moments.items()},step=next(iter(steps),0))
            before=state()
            optimizer_before=[copy.deepcopy(opt.state_dict()) for opt in collector.optimizers]
            def restore():
                with torch.no_grad():
                    for n,p in params.items():p.copy_(torch.tensor(before["parameters"][n],dtype=dtype))
                for opt,st in zip(collector.optimizers,optimizer_before):opt.load_state_dict(copy.deepcopy(st))
            def perform(instrument):
                captured={};handles=[]
                original_cloud=cloudgen.create_cloud
                original_loss=trainer.loss_function.compute_loss
                cloudgen.create_cloud=lambda states:cloud
                if instrument:
                    def hook(name):
                        def capture(module,ins,result):captured[name]=ref.host(result)
                        return capture
                    handles=[model.observation_encoder.register_forward_hook(hook("observation_features")),
                        model.particle_encoder_for_weights_model.register_forward_hook(hook("particle_features")),
                        model.weight_model.encoder.register_forward_hook(hook("logits"))]
                    def loss_capture(output,truth):
                        nll=original_loss(output,truth)
                        captured.update(particle_weights=ref.host(output["particle_weights"]),nll=ref.host(nll))
                        return nll
                    trainer.loss_function.compute_loss=loss_capture
                model.zero_grad(set_to_none=True)
                try:
                    objective,effective=forward(trainer,batch,None,0)
                    if training_step:objective.backward()
                finally:
                    cloudgen.create_cloud=original_cloud;trainer.loss_function.compute_loss=original_loss
                    for h in handles:h.remove()
                assert bool(torch.isfinite(objective))
                result=dict(loss=float(objective.detach()),effective_batch=effective,captured=captured)
                if training_step:
                    assert all(params[n].grad is not None and bool(torch.isfinite(params[n].grad).all()) for n in names)
                    result["gradients"]=ref.host({n:params[n].grad for n in names})
                    result["optimizer_gradients"]=ref.host({n:params[n].grad+.001*params[n] for n in names})
                    for opt in collector.optimizers:opt.step()
                result["state"]=state()
                return result
            result=perform(True)
            if index==0 and split=="training":
                restore();plain=perform(False)
                assert all(result[k]==plain[k] for k in ("loss","gradients","optimizer_gradients","state")),"Instrumentation changes source"
                result["uninstrumented_control"]=True
            if cmd["op"]=="replay":restore()
            if not training_step:assert before==result["state"],"Validation changed source"
            result.update(type="step",before=before,inputs=inputs,input_hash=digest(inputs))
            conn.send(result)
        for module in list(sys.modules.values()):
            p=getattr(module,"__file__",None)
            if p and Path(p).suffix==".py" and Path(p).is_relative_to(ref.AUTHOR):ref.verify(Path(p))
        conn.send(dict(type="closed",author_hashes=ref.HASHES,input_hashes=source_hashes))
    except BaseException:
        conn.send(dict(type="error",error=traceback.format_exc()))
        raise
    finally:conn.close()

def main(out, gradient_replay=False):
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]="true";os.environ["BAYESFILTER_PRELOAD_CUSTOM_OP"]="0"
    sys.path.insert(0,str(ROOT))
    import tensorflow as tf
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    memory=configure_tensorflow_gpu_memory_growth(tf,require_gpu=True)
    tf.config.experimental.enable_tensor_float_32_execution(False)
    tf.config.set_soft_device_placement(False)
    from bayesfilter.replication.kdm_neural_replay_tf import BearingsNeuralReplay
    from bayesfilter.replication.kdm_source_stages_tf import SourceWeightPretraining
    out.mkdir(exist_ok=False)
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
    tmp=tempfile.TemporaryDirectory(prefix="kdm-causal-",dir="/tmp")
    socket=str(Path(tmp.name)/"source.sock");listener=Listener(socket,family="AF_UNIX")
    log=(out/"author.log").open("w")
    env=dict(os.environ,CUDA_VISIBLE_DEVICES="-1",LD_LIBRARY_PATH="/home/chakwong/anaconda3/envs/tf-gpu2/lib")
    proc=subprocess.Popen(["/home/chakwong/anaconda3/envs/tf-gpu2/bin/python","-I","-B",str(Path(__file__).resolve()),"--socket",socket],
        cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    listener._listener._socket.settimeout(90)
    conn=None
    def receive():
        assert conn.poll(180),"Source timeout"
        r=conn.recv()
        if r["type"]=="error":raise RuntimeError(r["error"])
        return r
    records=[];controls=[];hashes={};started=time.monotonic()
    prior_root=BASE/"causal-localization-20261012/attempt05/diagnostic"
    prior_records={}
    if gradient_replay:
        for line in (prior_root/"steps.jsonl").read_text().splitlines():
            item=json.loads(line)
            if item["unit"]=="unit01" and item["dtype"]=="float32" and item["split"]=="training":
                prior_records[item["index"]]=item
    exact_reproduction_checks=0
    try:
        conn=listener.accept();ready=receive()
        write(out/"environment.json",dict(source=ready,tensorflow=tf.__version__,memory_growth=memory,
            gpu=[d.name for d in tf.config.list_logical_devices("GPU")],tf32=False,jit_compile=True,seed=SEED))
        for unit in (("unit01",) if gradient_replay else ("unit01","unit02")):
            for dtype in (("float32",) if gradient_replay else ("float32","float64")):
                conn.send(dict(op="setup",unit=unit,dtype=dtype));setup=receive()
                save(out/(unit+"-"+dtype+"-setup.json.gz"),setup)
                with tf.device("/GPU:0"):model=BearingsNeuralReplay(setup["fixture"],batch_size=64)
                stages={};flagged=False;first_flag=False
                totals={}
                for split,count in ((("training",68),) if gradient_replay else (("training",79),("validation",4))):
                    if split=="validation" and (flagged or (dtype=="float64" and (unit,split,0) not in hashes)):break
                    total=0.;source_total=0.;rows=0
                    for index in range(count):
                        if split=="training" and index==16 and flagged:break
                        # FP64 never extends beyond the matched FP32 input scope.
                        if dtype=="float64" and (unit,split,index) not in hashes:break
                        conn.send(dict(op="update" if split=="training" else "evaluate",split=split,index=index))
                        r=receive();assert r["type"]=="step"
                        key=(unit,split,index)
                        assert digest(r["inputs"])==r["input_hash"]
                        if dtype=="float32":
                            hashes[key]=r["input_hash"];save(out/(unit+"-"+split+"-%03d-input.json.gz"%index),r["inputs"])
                        else:assert hashes[key]==r["input_hash"],"Precision inputs differ"
                        b=len(r["inputs"]["rows"])
                        if b not in stages:
                            with tf.device("/GPU:0"):stages[b]=SourceWeightPretraining(model,b)
                        stage=stages[b]
                        with tf.device("/GPU:0"):
                            data=r["inputs"]["data"]
                            inputs=[tf.constant(data[k],tf.bool if k=="ground_truth_mask" else model.dtype) for k in ("states","observations","ground_truth_mask")]
                            inputs.append(tf.constant(r["inputs"]["cloud"],model.dtype))
                            before=model.state_dict()
                            result=(stage.update if split=="training" else stage.evaluate)(*inputs)
                            assert bool(result["valid"])
                            if split=="training":assert bool(result["committed"])
                            comparisons=compare(result,r,model,dtype)
                            if split=="validation":assert before==model.state_dict()
                            if split=="training" and index==0:
                                after=model.state_dict();model.load_state_dict(before)
                                repeat=stage.update(*inputs)
                                assert after==model.state_dict() and host(result)==host(repeat)
                                assert r["uninstrumented_control"]
                                controls.append(dict(unit=unit,dtype=dtype,source_uninstrumented=True,tf_exact_repeat=True))
                            failed=[n for n,v in comparisons.items() if v["max_scaled"]>1]
                            item=dict(unit=unit,dtype=dtype,split=split,index=index,input_hash=r["input_hash"],comparisons=comparisons,flags=failed)
                            if (failed and not first_flag) or (gradient_replay and index==67):
                                first_flag=True;after=model.state_dict()
                                model.load_state_dict(r["before"])
                                same_state=(stage.update if split=="training" else stage.evaluate)(*inputs)
                                item["same_source_state"]=compare(same_state,r,model,dtype)
                                suffix="-gradient67" if gradient_replay and index==67 else "-first-flag"
                                save(out/(unit+"-"+dtype+suffix+".json.gz"),dict(reference=r,tf_before=before,
                                    tf_evolving=host(result),tf_same_state=host(same_state)))
                                model.load_state_dict(after)
                            flagged|=bool(failed)
                        item.update(loss=float(result["loss"]),source_loss=r["loss"])
                        if gradient_replay:
                            old=prior_records[index]
                            assert all(item[k]==old[k] for k in ("input_hash","loss","source_loss","comparisons","flags")),("Prior trajectory changed",index)
                            exact_reproduction_checks+=5
                        records.append(item)
                        with (out/"steps.jsonl").open("a") as f:f.write(json.dumps(item)+"\n")
                        total+=b*float(result["loss"]);source_total+=b*r["loss"];rows+=b
                        write(out/"progress.json",dict(unit=unit,dtype=dtype,split=split,index=index,flags=sum(bool(x["flags"]) for x in records),elapsed=time.monotonic()-started))
                    totals[split]=dict(rows=rows,loss=total/rows,source_loss=source_total/rows) if rows else {}
                traces={str(b):dict(update=s.update.experimental_get_tracing_count(),evaluate=s.evaluate.experimental_get_tracing_count()) for b,s in stages.items()}
                assert all(v<=1 for t in traces.values() for v in t.values()),"Retracing"
                write(out/(unit+"-"+dtype+"-summary.json"),dict(totals=totals,traces=traces,flags=flagged,final_state=model.state_dict()))
        conn.send(dict(op="stop"));closed=receive();assert closed["type"]=="closed";write(out/"source-provenance.json",closed)
        proc.wait(timeout=30);assert proc.returncode==0
        result=dict(status="COMPLETED",comparisons=len(records),controls=controls,
            flagged_steps=[{k:r[k] for k in ("unit","dtype","split","index","flags")} for r in records if r["flags"]],
            max_scaled={k:max(r["comparisons"][k]["max_scaled"] for r in records if k in r["comparisons"]) for k in records[0]["comparisons"]},
            gradient_replay=gradient_replay,exact_reproduction_checks=exact_reproduction_checks,
            selected_replays=[r for r in records if "same_source_state" in r],
            wall_seconds=time.monotonic()-started,replication="INCOMPLETE",role="Fresh pretraining localization, not final quality")
        write(out/"result.json",result)
        print(json.dumps(result),flush=True)
    finally:
        if conn:conn.close()
        listener.close()
        if proc.poll() is None:
            proc.terminate()
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()
        log.close();tmp.cleanup()
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--socket");p.add_argument("--output",type=Path)
    p.add_argument("--gradient-replay",action="store_true")
    a=p.parse_args()
    if a.socket:source_worker(a.socket)
    else:main(a.output,a.gradient_replay)
