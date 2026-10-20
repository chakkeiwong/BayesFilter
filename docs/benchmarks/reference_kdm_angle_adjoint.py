"""Independent CPU author adjoints on a saved conditional draw tape; diagnostic only."""
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
os.environ["OMP_NUM_THREADS"]="1"
import argparse, gzip, hashlib, importlib.util, json, sys
from pathlib import Path
from types import SimpleNamespace, MethodType
ROOT=Path("/home/chakwong/BayesFilter")
BASE=ROOT/"docs/plans/artifacts/controlled-smc-kdm-replication-20260928"
def read(p):
    if str(p).endswith(".gz"):
        with gzip.open(p,"rt") as f:return json.load(f)
    return json.loads(p.read_text())
def save(p,x):
    with gzip.open(p,"wt",compresslevel=1) as f:json.dump(x,f,allow_nan=False)
def main(inp,out):
    spec=importlib.util.spec_from_file_location("author",ROOT/"docs/benchmarks/reference_kdm_neural_exact_replay.py")
    ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
    torch=ref.torch;torch.set_num_threads(1);torch.set_num_interop_threads(1)
    forward=ref.source_member(ref.AUTHOR/"src/trainers/specific_trainers/full_sequence_trainer.py","FullSequenceTrainer","do_forward_pass")
    convert=ref.source_member(ref.AUTHOR/"src/trainers/specific_trainers/full_sequence_trainer.py","FullSequenceTrainer","convert_to_model_dtype")
    transformer=ref.source_member(ref.AUTHOR/"src/problems/bearings_only/bearings_only_problem.py","BearingsOnlyObservationTransformer")
    cfg=next(x for x in read(BASE/"kdm-source-variability-20261008-01/unit01-full/diagnostic/configurations.json")["executed_stages"] if x["name"]=="mdpf")
    architecture={}
    for p in (ref.EXPERIMENT/"configs/models.yaml",ref.EXPERIMENT/"experiment0002_importance/config.yaml"):
        architecture.update(ref.yaml.safe_load(ref.verify(p))["models"])
    assert json.loads(json.dumps(architecture))==cfg["architecture"]
    data=read(inp);results={}
    for role,request in data.items():
        case=read(Path(request["case"]));expected=case["source"];state=expected["before"];cap=expected["captured"]
        dtype=torch.float32;torch.set_default_dtype(dtype)
        model=ref.KDEParticleFilter(cfg["configuration"]["model_params"],architecture)
        params=dict(model.named_parameters())
        with torch.no_grad():
            for name,p in params.items():p.copy_(torch.tensor(state["parameters"][name],dtype=dtype))
        for name,p in params.items():p.requires_grad_(name in state["trainable_names"])
        assert [n for n,p in params.items() if p.requires_grad]==state["trainable_names"]
        model.train()
        trainer=SimpleNamespace(main_model=model,device=torch.device("cpu"),
            problem=SimpleNamespace(observation_transformer=transformer()),number_of_particles=25,
            skip_first_step_for_loss=False,max_sequence_length=None,truncated_bptt_modulo=4,loss_shaper=None,
            loss_function=ref.KDE_NLL(dict(kde_params=architecture["dpf_full_training"]["kde"]),model))
        trainer.convert_to_model_dtype=MethodType(convert,trainer)
        batch={k:torch.tensor(v,dtype=torch.bool if k=="ground_truth_mask" else dtype) for k,v in expected["data"].items()}
        original_sample=ref.KernelDensityEstimator.sample
        original_norm=model.particle_transformer.apply_norm
        original_angle=model.particle_transformer.forward_tranform
        def run(coordinate=None,epsilon=0.):
            raw=[];normal=[];angle=[];noises=[];samples=[];outputs=[];perturbation=None
            def init(*args,**kwargs):
                return {k:torch.tensor(v,dtype=dtype) if isinstance(v,list) else v for k,v in cap["initial"].items()}
            def sample(kde,*args,**kwargs):
                value=torch.tensor(cap["samples"][len(samples)],dtype=dtype);samples.append(value);return value
            def proposal(module,inputs):
                noise=torch.tensor(request["noise"][len(noises)],dtype=dtype);noises.append(noise)
                return (*inputs,noise)
            def raw_hook(module,inputs,result):
                nonlocal perturbation
                value=result[0]
                if coordinate is not None and len(raw)==coordinate[0]:
                    delta=torch.zeros_like(value);idx=tuple(coordinate[1:]);delta[idx]=epsilon
                    before=float(value[idx].detach());value=value+delta
                    perturbation=dict(coordinate=coordinate,requested=epsilon,realized=float(value[idx].detach())-before)
                value.retain_grad();raw.append(value)
                return (value,*result[1:])
            def norm(value):
                result=original_norm(value)
                if value.shape[-1]==4:
                    result.retain_grad();normal.append(result)
                return result
            def transform(value):
                result=original_angle(value)
                if value.requires_grad:
                    result.retain_grad();angle.append(result)
                return result
            def output(module,inputs,result):outputs.append(ref.host({k:v for k,v in result.items() if torch.is_tensor(v)}))
            model.create_initial_dpf_state=init;ref.KernelDensityEstimator.sample=sample
            model.particle_transformer.apply_norm=norm;model.particle_transformer.forward_tranform=transform
            hooks=[model.proposal_model.register_forward_pre_hook(proposal),
                model.proposal_model.register_forward_hook(raw_hook),model.register_forward_hook(output)]
            model.zero_grad(set_to_none=True)
            try:
                loss,effective=forward(trainer,batch,None,0);assert effective==64
                loss.backward()
            finally:
                for h in hooks:h.remove()
                ref.KernelDensityEstimator.sample=original_sample
                model.particle_transformer.apply_norm=original_norm;model.particle_transformer.forward_tranform=original_angle
            assert len(raw)==len(normal)==len(angle)==16
            grads={n:ref.host(p.grad) for n,p in params.items() if p.requires_grad}
            assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in params.values() if p.requires_grad)
            def adjoint(values):return ref.host([torch.zeros_like(v) if v.grad is None else v.grad for v in values])
            return dict(loss=float(loss.detach()),gradients=grads,raw=ref.host(raw),normal=ref.host(normal),
                angle=ref.host(angle),raw_adjoint=adjoint(raw),normal_adjoint=adjoint(normal),
                angle_adjoint=adjoint(angle),adjoint_present=dict(raw=[v.grad is not None for v in raw],
                normal=[v.grad is not None for v in normal],angle=[v.grad is not None for v in angle]),
                outputs=outputs,perturbation=perturbation)
        baseline=run()
        controls=dict(loss=baseline["loss"]==expected["loss"],gradients=baseline["gradients"]==expected["gradients"],
            raw=baseline["raw"]==cap["raw"],outputs=baseline["outputs"]==cap["outputs"])
        assert all(controls.values()),(role,controls)
        r=torch.tensor(baseline["raw"])[...,2:];radius=torch.linalg.vector_norm(r,dim=-1)
        flat=int(radius.flatten().argmin());location=[flat//(64*25),(flat//25)%64,flat%25]
        panel=[]
        for component in request.get("components",(2,3)):
            for epsilon in request.get("epsilons",(0.,-1e-8,1e-8,-1e-7,1e-7,-1e-6,1e-6,-1e-5,1e-5)):
                trial=run(location+[component],epsilon)
                if epsilon==0.:
                    assert trial["loss"]==baseline["loss"] and trial["gradients"]==baseline["gradients"],"zero perturbation changed source"
                delta=max(float((torch.tensor(trial["gradients"][n])-torch.tensor(baseline["gradients"][n])).abs().max()) for n in state["trainable_names"])
                panel.append(dict(perturbation=trial["perturbation"],loss_difference=trial["loss"]-baseline["loss"],
                    gradient_max_difference=delta,gradients=trial["gradients"]))
        results[role]=dict(controls=controls,baseline=baseline,minimum_radius=float(radius.min()),
            location=location,perturbations=panel)
        print(json.dumps(dict(role=role,controls=controls,minimum_radius=float(radius.min()),panel_size=len(panel))),flush=True)
    for module in list(sys.modules.values()):
        p=getattr(module,"__file__",None)
        if p and Path(p).suffix==".py" and Path(p).is_relative_to(ref.AUTHOR):ref.verify(Path(p))
    save(out,dict(results=results,provenance=dict(torch=torch.__version__,cpu_only=True,
        gpu_intentionally_hidden=True,source_commit=ref.PIN,source_hashes=ref.HASHES,
        input_sha256=hashlib.sha256(inp.read_bytes()).hexdigest())))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--input",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();main(a.input,a.output)
