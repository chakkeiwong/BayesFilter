"""Independent CPU author worker for common-base-variate diagnostics only."""
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
os.environ["OMP_NUM_THREADS"]="1"
import argparse, copy, hashlib, importlib.util, json, sys, time, traceback
from pathlib import Path
from types import SimpleNamespace, MethodType
from multiprocessing.connection import Client
ROOT=Path("/home/chakwong/BayesFilter")
BASE=ROOT/"docs/plans/artifacts/controlled-smc-kdm-replication-20260928"
CAL=BASE/"paired-calibration-20261011/calibration-01"
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main(socket):
    spec=importlib.util.spec_from_file_location("author",ROOT/"docs/benchmarks/reference_kdm_neural_exact_replay.py")
    ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
    torch=ref.torch;torch.set_num_threads(1);torch.set_num_interop_threads(1)
    trainer_path=ref.AUTHOR/"src/trainers/specific_trainers/full_sequence_trainer.py"
    forward=ref.source_member(trainer_path,"FullSequenceTrainer","do_forward_pass")
    convert=ref.source_member(trainer_path,"FullSequenceTrainer","convert_to_model_dtype")
    transformer=ref.source_member(ref.AUTHOR/"src/problems/bearings_only/bearings_only_problem.py","BearingsOnlyObservationTransformer")
    source=BASE/"kdm-source-variability-20261008-01/unit01-full/diagnostic"
    cfg=next(x for x in read(source/"configurations.json")["executed_stages"] if x["name"]=="mdpf")
    architecture={}
    for p in (ref.EXPERIMENT/"configs/models.yaml",ref.EXPERIMENT/"experiment0002_importance/config.yaml"):
        architecture.update(ref.yaml.safe_load(ref.verify(p))["models"])
    assert json.loads(json.dumps(architecture))==cfg["architecture"]
    data=read(source/"training.json")["data"]
    conn=Client(socket,family="AF_UNIX")
    conn.send(dict(type="ready",torch=torch.__version__,cpu_only=True,gpu_intentionally_hidden=True,
        source_commit=ref.PIN,data_sha256=sha(source/"training.json"),data_path=str(source/"training.json")))
    models={}
    while True:
        command=conn.recv()
        if command["op"]=="stop":break
        try:
            if command["op"]=="setup":
                role=command["role"];unit,origin=role.split("-")
                source=BASE/"kdm-source-variability-20261008-01"/(unit+"-full")/"diagnostic"
                cfg=next(x for x in read(source/"configurations.json")["executed_stages"] if x["name"]=="mdpf")
                assert json.loads(json.dumps(architecture))==cfg["architecture"]
                data=read(source/"training.json")["data"]
                validation=read(source/"validation.json")["data"]
                p=CAL/unit/"training/diagnostic/mdpf-starting-state.json"
                state=read(p)
                if origin.startswith("author"):
                    p=source/("mdpf/models/full_dpf_model_best.pt" if origin.endswith("best") else "mdpf/models/full_dpf_model_untrained.pt")
                    blob=torch.load(p,map_location="cpu",weights_only=True)
                    assert set(state["parameters"]).issubset(blob)
                    state["parameters"]={n:ref.host(blob[n]) for n in state["parameters"]}
                elif origin=="tfbest":
                    p=CAL/unit/"training/diagnostic/mdpf-best.json"
                    state["parameters"]=read(p)["model"]["parameters"]
                assert state["step"]==0
                assert all(not torch.count_nonzero(torch.tensor(v)) for field in ("first_moments","second_moments") for v in state[field].values())
                models={}
                for arm,dtype_name in (("source32","float32"),("repeat32","float32"),("source64","float64")):
                    dtype=getattr(torch,dtype_name);torch.set_default_dtype(dtype)
                    # Match saved float32 constants before converting the reference to binary64.
                    torch.set_default_dtype(torch.float32)
                    model=ref.KDEParticleFilter(cfg["configuration"]["model_params"],architecture).to(dtype=dtype)
                    torch.set_default_dtype(dtype)
                    for module in model.modules():
                        for attr in ("min_bandwidths","max_bandwidths","residual_scale_factor"):
                            v=getattr(module,attr,None)
                            if torch.is_tensor(v):setattr(module,attr,v.to(dtype))
                    params=dict(model.named_parameters())
                    with torch.no_grad():
                        for name,v in params.items():v.copy_(torch.tensor(state["parameters"][name],dtype=dtype))
                    class Collector:
                        def __init__(self):self.optimizers=[]
                        def add_optimizer_and_lr_scheduler(self,modules,names,lr):
                            for module in modules:
                                if lr=="Freeze":
                                    for v in module.parameters():v.requires_grad_(False)
                                else:self.optimizers.append(torch.optim.Adam(module.parameters(),lr=lr,weight_decay=0.))
                    collector=Collector();model.create_and_add_optimizers(cfg["configuration"]["training_params"],collector,"full")
                    names={id(v):n for n,v in params.items()}
                    for opt in collector.optimizers:
                        for group in opt.param_groups:
                            rates={state["rates"][names[id(v)]] for v in group["params"]};assert len(rates)==1
                            group["lr"]=rates.pop()
                            for v in group["params"]:
                                n=names[id(v)]
                                opt.state[v]=dict(step=torch.tensor(state["step"],dtype=torch.float32),
                                    exp_avg=torch.tensor(state["first_moments"][n],dtype=dtype),
                                    exp_avg_sq=torch.tensor(state["second_moments"][n],dtype=dtype))
                    assert [n for n,v in params.items() if v.requires_grad]==state["trainable_names"]
                    trainer=SimpleNamespace(main_model=model,device=torch.device("cpu"),problem=SimpleNamespace(
                        observation_transformer=transformer()),number_of_particles=25,skip_first_step_for_loss=False,
                        max_sequence_length=None,truncated_bptt_modulo=4,loss_shaper=None,
                        loss_function=ref.KDE_NLL(dict(kde_params=architecture["dpf_full_training"]["kde"]),model))
                    trainer.convert_to_model_dtype=MethodType(convert,trainer)
                    model.train();models[arm]=(model,params,collector,trainer,dtype)
                constants=dict(residual_scale=ref.host(model.proposal_model.residual_scale_factor),
                    min_bandwidths=ref.host(model.weighted_bandwidth_predictor.min_bandwidths),resampling_min_bandwidths=None)
                conn.send(dict(type="setup",state=state,constants=constants,path=str(p),sha256=sha(p),
                    data_hashes={str(source/(split+".json")):sha(source/(split+".json")) for split in ("training","validation")}))
                continue
            assert command["op"] in ("update","evaluate","replay")
            arm,index=command["arm"],command["index"]
            model,params,collector,trainer,dtype=models[arm];torch.set_default_dtype(dtype)
            evaluating=command["op"]=="evaluate";replaying=command["op"]=="replay"
            def snapshot():
                moments={v:st for opt in collector.optimizers for v,st in opt.state.items()}
                names=[n for n,v in params.items() if v.requires_grad]
                rates={n:g["lr"] for opt in collector.optimizers for g in opt.param_groups for n,p in params.items() if any(p is v for v in g["params"])}
                steps={int(st["step"]) for st in moments.values()}
                assert len(steps)<=1
                return dict(schema=state["schema"],adam_scalar_policy=state["adam_scalar_policy"],
                    variant="mdpf",dtype=str(dtype).split(".")[-1],trainable_names=names,constants=state["constants"],
                    parameters=ref.host(params),rates=rates,step=next(iter(steps),0),
                    first_moments={n:ref.host(moments[params[n]]["exp_avg"]) for n in names},
                    second_moments={n:ref.host(moments[params[n]]["exp_avg_sq"]) for n in names})
            live=snapshot();saved_opts=[copy.deepcopy(o.state_dict()) for o in collector.optimizers]
            if replaying:
                requested=command["state"]
                with torch.no_grad():
                    for n,p in params.items():p.copy_(torch.tensor(requested["parameters"][n],dtype=dtype))
                for opt in collector.optimizers:
                    for n,p in params.items():
                        if p in opt.state:
                            opt.state[p]=dict(step=torch.tensor(requested["step"],dtype=torch.float32),
                                exp_avg=torch.tensor(requested["first_moments"][n],dtype=dtype),
                                exp_avg_sq=torch.tensor(requested["second_moments"][n],dtype=dtype))
            before=snapshot()
            model.eval() if evaluating else model.train()
            lo=0 if evaluating else 384+64*index
            panel=validation if evaluating else data
            batch={k:torch.tensor(v[lo:lo+64],dtype=torch.bool if k=="ground_truth_mask" else dtype) for k,v in panel.items()}
            ctx=dict(stage="initial",time=0,normal=0,iteration=0)
            captured=dict(samples=[],outputs=[],raw=[],ancestors=[],iterations=[],sampler_inputs=[],cdf=[],uniforms=[],cdf_margin=[])
            raw_refs=[]
            perturbation=None
            def base(kind,time_index=0,iteration=0):
                conn.send(dict(type="random",kind=kind,time=time_index,iteration=iteration,index=index,dtype=str(dtype).split(".")[-1]))
                msg=conn.recv();assert msg["type"]=="random"
                return torch.frombuffer(bytearray(msg["bytes"]),dtype=getattr(torch,msg["dtype"])).reshape(msg["shape"])
            original_normal=torch.distributions.Normal.sample
            original_cat=torch.distributions.Categorical.sample
            original_rand=torch.rand
            original_init=model.create_initial_dpf_state
            original_sample=ref.KernelDensityEstimator.sample
            def initialize(*a,**kw):
                ctx.update(stage="initial",time=0,normal=0,iteration=0)
                result=original_init(*a,**kw);captured["initial"]=ref.host(result)
                captured["initial_iterations"]=ctx["iteration"]
                return result
            def sample(kde,*a,**kw):
                ctx.update(stage="resample",time=len(captured["samples"])+2,normal=0,iteration=0)
                captured["sampler_inputs"].append(ref.host(dict(p=kde.particles,w=kde.weights,bw=kde.bandwidths)))
                result=original_sample(kde,*a,**kw)
                captured["samples"].append(ref.host(result));captured["iterations"].append(ctx["iteration"])
                return result
            def normal(dist,sample_shape=torch.Size()):
                assert ctx["normal"]<2
                z=base("initial_normal" if ctx["stage"]=="initial" else "normal",ctx["time"])
                z=(z[:,:,0,ctx["normal"]].T if ctx["stage"]=="initial" else z[...,ctx["normal"]])
                ctx["normal"]+=1
                assert tuple(z.shape)==tuple(sample_shape)+tuple(dist.batch_shape)
                return dist.loc+dist.scale*z
            def categorical(dist,sample_shape=torch.Size()):
                assert list(sample_shape)==[25]
                u=base("categorical",ctx["time"])
                cdf=dist.probs.cumsum(-1)
                ancestors=(u[...,None]>=cdf[:,None,:]).sum(-1).clamp(max=24)
                lower=torch.cat([torch.zeros_like(cdf[:,:1]),cdf[:,:-1]],dim=-1).gather(1,ancestors)
                upper=cdf.gather(1,ancestors)
                assert bool(((u>=lower)&((u<upper)|(ancestors==24))).all()),"CDF interval mismatch"
                captured["ancestors"].append(ref.host(ancestors))
                captured["cdf"].append(ref.host(cdf));captured["uniforms"].append(ref.host(u))
                captured["cdf_margin"].append(float(torch.minimum((u-lower).abs(),(upper-u).abs()).min()))
                return ancestors.T
            def rand(shape,**kw):
                assert ctx["iteration"]<128
                u=base("initial_uniform" if ctx["stage"]=="initial" else "uniform",ctx["time"],ctx["iteration"])
                if ctx["stage"]=="initial":u=u[:,:,:,0].permute(0,2,1)
                assert tuple(u.shape)==tuple(shape) and u.dtype==kw["dtype"]
                ctx["iteration"]+=1
                return u
            def proposal_inputs(module,inputs):
                nonlocal perturbation
                old,encoded,obs=inputs;t=len(captured["raw"])+1
                noise=base("proposal",t)
                if arm=="perturb32" and index==0 and t==1:
                    flat=noise.flatten();coordinate=int(flat.abs().argmin());before=float(flat[coordinate])
                    flat[coordinate]+=1e-8;after=float(flat[coordinate]);noise=flat.reshape(noise.shape)
                    perturbation=dict(coordinate=coordinate,before=before,after=after,requested=1e-8,realized=after-before)
                    assert after!=before
                return old,encoded,obs,noise
            def raw_output(module,inputs,result):
                captured["raw"].append(ref.host(result[0]))
                if result[0].requires_grad:
                    result[0].retain_grad();raw_refs.append(result[0])
            def output(module,inputs,result):captured["outputs"].append(ref.host({k:v for k,v in result.items() if torch.is_tensor(v)}))
            hooks=[model.proposal_model.register_forward_pre_hook(proposal_inputs),
                model.proposal_model.register_forward_hook(raw_output),model.register_forward_hook(output)]
            torch.distributions.Normal.sample=normal;torch.distributions.Categorical.sample=categorical;torch.rand=rand
            model.create_initial_dpf_state=initialize;ref.KernelDensityEstimator.sample=sample
            tick=time.monotonic()
            try:
                for opt in collector.optimizers:opt.zero_grad(set_to_none=True)
                loss,effective=forward(trainer,batch,None,0)
                assert effective==64 and bool(torch.isfinite(loss))
                if not evaluating:loss.backward()
            finally:
                torch.distributions.Normal.sample=original_normal;torch.distributions.Categorical.sample=original_cat;torch.rand=original_rand
                model.create_initial_dpf_state=original_init;ref.KernelDensityEstimator.sample=original_sample
                for h in hooks:h.remove()
            trainable={n:v for n,v in params.items() if v.requires_grad}
            if not evaluating:
                assert all(v.grad is not None and bool(torch.isfinite(v.grad).all()) for v in trainable.values())
                gradients=ref.host({n:v.grad for n,v in trainable.items()})
                captured["raw_adjoints"]=[ref.host(v.grad) if v.grad is not None else None for v in raw_refs]
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),100.)
                for opt in collector.optimizers:opt.step()
            else:
                gradients={};norm=torch.tensor(0.)
            moments={}
            for opt in collector.optimizers:
                for n,v in trainable.items():
                    if v in opt.state:moments[n]=ref.host(opt.state[v])
            assert all(bool(torch.isfinite(v).all()) for v in params.values())
            assert len(captured["samples"])==15 and len(captured["raw"])==16
            after=snapshot()
            if evaluating:assert before==after,"Evaluation mutated author state"
            response=dict(type="update",arm=arm,index=index,loss=float(loss.detach()),gradient_norm=float(norm),
                gradients=gradients,parameters=ref.host(trainable),moments=moments,captured=captured,
                data=ref.host(batch),perturbation=perturbation,wall_seconds=time.monotonic()-tick,before=before,after=after)
            if replaying:
                with torch.no_grad():
                    for n,p in params.items():p.copy_(torch.tensor(live["parameters"][n],dtype=dtype))
                for opt,st in zip(collector.optimizers,saved_opts):opt.load_state_dict(st)
                assert snapshot()==live
            conn.send(response)
        except BaseException:
            conn.send(dict(type="error",error=traceback.format_exc()));break
    for module in list(sys.modules.values()):
        p=getattr(module,"__file__",None)
        if p and Path(p).suffix==".py" and Path(p).is_relative_to(ref.AUTHOR):ref.verify(Path(p))
    conn.send(dict(type="closed",author_commit=ref.PIN,author_hashes=ref.HASHES))
    conn.close()
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--socket",required=True);main(p.parse_args().socket)
