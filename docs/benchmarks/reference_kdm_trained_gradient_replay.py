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
                role=command["role"]
                p=CAL/"unit01/training/diagnostic"/("mdpf-starting-state.json" if role=="starting" else "mdpf-best.json")
                state=read(p);state=state if role=="starting" else state["model"]
                models={}
                for arm,dtype_name in (("source32","float32"),("perturb32","float32"),("repeat32","float32"),("source64","float64")):
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
                conn.send(dict(type="setup",state=state,constants=constants,path=str(p),sha256=sha(p)))
                continue
            assert command["op"]=="update"
            arm,index=command["arm"],command["index"]
            model,params,collector,trainer,dtype=models[arm];torch.set_default_dtype(dtype)
            before=copy.deepcopy(state)
            before["dtype"]=str(dtype).split(".")[-1]
            before["parameters"]=ref.host(params)
            optimizer_states={p:v for opt in collector.optimizers for p,v in opt.state.items()}
            before["first_moments"]={n:ref.host(optimizer_states[p]["exp_avg"]) for n,p in params.items() if p.requires_grad}
            before["second_moments"]={n:ref.host(optimizer_states[p]["exp_avg_sq"]) for n,p in params.items() if p.requires_grad}
            steps={int(v["step"]) for v in optimizer_states.values()}
            assert len(steps)==1
            before["step"]=next(iter(steps))

            lo=384+64*index
            batch={k:torch.tensor(v[lo:lo+64],dtype=torch.bool if k=="ground_truth_mask" else dtype) for k,v in data.items()}
            ctx=dict(stage="initial",time=0,normal=0,iteration=0)
            captured=dict(samples=[],outputs=[],raw=[],ancestors=[],iterations=[],sampler_inputs=[])
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
            def raw_output(module,inputs,result):captured["raw"].append(ref.host(result[0]))
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
                loss.backward()
            finally:
                torch.distributions.Normal.sample=original_normal;torch.distributions.Categorical.sample=original_cat;torch.rand=original_rand
                model.create_initial_dpf_state=original_init;ref.KernelDensityEstimator.sample=original_sample
                for h in hooks:h.remove()
            trainable={n:v for n,v in params.items() if v.requires_grad}
            assert all(v.grad is not None and bool(torch.isfinite(v.grad).all()) for v in trainable.values())
            gradients=ref.host({n:v.grad for n,v in trainable.items()})
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),100.)
            for opt in collector.optimizers:opt.step()
            moments={}
            for opt in collector.optimizers:
                for n,v in trainable.items():
                    if v in opt.state:moments[n]=ref.host(opt.state[v])
            assert all(bool(torch.isfinite(v).all()) for v in params.values())
            assert len(captured["samples"])==15 and len(captured["raw"])==16
            conn.send(dict(type="update",arm=arm,index=index,loss=float(loss.detach()),gradient_norm=float(norm),
                gradients=gradients,parameters=ref.host(trainable),moments=moments,captured=captured,
                before=before,data=ref.host(batch),perturbation=perturbation,wall_seconds=time.monotonic()-tick))
        except BaseException:
            conn.send(dict(type="error",error=traceback.format_exc()));break
    for module in list(sys.modules.values()):
        p=getattr(module,"__file__",None)
        if p and Path(p).suffix==".py" and Path(p).is_relative_to(ref.AUTHOR):ref.verify(Path(p))
    conn.send(dict(type="closed",author_commit=ref.PIN,author_hashes=ref.HASHES))
    conn.close()
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--socket",required=True);main(p.parse_args().socket)
