"""Independent CPU author-stage reference and dataset export; not runtime code."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace, MethodType
sys.path.insert(0, str(Path(__file__).resolve().parent))
import reference_kdm_neural as ref
torch, yaml = ref.torch, ref.yaml
ROOT, AUTHOR, EXPERIMENT, HISTORY = ref.ROOT, ref.AUTHOR, ref.EXPERIMENT, ref.HISTORY
host, verify, member = ref.host, ref.verify, ref.source_member
VM = sys.modules["models.kde_particle_filter.kde_particle_filter"].VonMisesFullDist

def write(path, data):
    path.write_text(json.dumps(data, allow_nan=False)+"\n")

def extract_clouds(path):
    tree=ast.parse(verify(path))
    nodes=[x for x in tree.body if isinstance(x,ast.ClassDef) and
           x.name in ("ParticleCloudGenerator","DistributionCloudGenerator")]
    ns=dict(torch=torch,D=torch.distributions,VonMisesFullDist=VM)
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(path),"exec"),ns)
    return ns["DistributionCloudGenerator"]

def build_model(model_params, architecture, checkpoint, dtype):
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(20261007)
    model=ref.KDEParticleFilter(model_params,architecture)
    model.load_pretrained({"dpf_model":str(checkpoint)},torch.device("cpu"))
    model.to(dtype=dtype)
    for module in model.modules():
        for attr in ("min_bandwidths","max_bandwidths","residual_scale_factor"):
            value=getattr(module,attr,None)
            if torch.is_tensor(value):
                setattr(module,attr,value.to(dtype=dtype))
    torch.set_default_dtype(dtype)
    model.train()
    return model

def constants(model):
    return dict(residual_scale=host(model.proposal_model.residual_scale_factor),
                min_bandwidths=host(model.weighted_bandwidth_predictor.min_bandwidths),
                resampling_min_bandwidths=host(model.resampling_weighted_bandwidth_predictor.min_bandwidths)
                   if model.resampling_weighted_bandwidth_predictor is not None else None)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    checkpoint=HISTORY/"attempt16/full_training/models/full_dpf_model_best.pt"
    digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    contract=json.loads((HISTORY/"attempt17/contract.json").read_text())
    if digest != contract["checkpoint_sha256"]:
        raise RuntimeError("Source checkpoint checksum mismatch")
    baseline=ROOT/"docs/plans/artifacts/controlled-smc-kdm-replication-20260928/kdm-neural-batch64-20261006-02/source/author-batch64.json"
    baseline_hash=hashlib.sha256(baseline.read_bytes()).hexdigest()
    data_config=yaml.safe_load(verify(EXPERIMENT/"configs/dataset_params_training.yaml"))
    common=yaml.safe_load(verify(EXPERIMENT/"configs/common_parameters_training.yaml"))
    architecture=yaml.safe_load(verify(EXPERIMENT/"configs/models.yaml"))["models"]
    historical=json.loads((HISTORY/"attempt16/diagnostic-configurations.json").read_text())
    if common["model_params"] != historical["configurations"]["full_training"]["model_params"]:
        raise RuntimeError("Source model wiring changed")
    item=member(AUTHOR/"src/datasets/bearings_only_dataset.py","BearingsOnlyDataset","__getitem__")
    datasets={}
    training_rows=None
    for split,size in (("training",5000),("validation",1000)):
        source=HISTORY/("attempt16/datasets/"+split+".pt")
        saved=torch.load(source,map_location="cpu",weights_only=False)
        owner=SimpleNamespace(car_states=saved["car_states"],sensor_obs=saved["sensor_obs"],
                              sparse_ground_truth_keep_modulo=4)
        rows=[item(owner,i) for i in range(size)]
        data={k:torch.stack([r[k] for r in rows]) for k in ("states","observations","ground_truth_mask")}
        if tuple(data["states"].shape)!=(size,17,3) or tuple(data["observations"].shape)!=(size,17,1):
            raise RuntimeError("Unexpected source dataset shape")
        if saved["car_states"].shape[0]!=size:
            raise RuntimeError("Dataset row-count mismatch")
        target=args.output/(split+".json")
        write(target,dict(schema="kdm_source_dataset_export_v1",split=split,rows=size,
                         data=host(data),row_ids=list(range(size)),source=str(source),
                         source_sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
        datasets[split]=dict(path=str(target.resolve()),sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                             source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),rows=size)
        if split=="training":
            training_rows={k:v[:64] for k,v in data.items()}
    observation_cls=member(AUTHOR/"src/problems/bearings_only/bearings_only_problem.py",
                           "BearingsOnlyObservationTransformer")
    weight_path=AUTHOR/"src/trainers/specific_trainers/weight_model_trainer.py"
    cloud_cls=extract_clouds(weight_path)
    forward=member(weight_path,"WeightModelTrainer","do_forward_pass")
    masked=member(weight_path,"WeightModelTrainer","get_true_masked_data")
    config=yaml.safe_load(verify(EXPERIMENT/"individual_training/config.yaml"))
    pre_spec=next(value for entry in config["experiments"] for name,value in entry.items()
                  if name=="full_dpf_weight_training_fixed_bands")
    pretraining=pre_spec["training_params"]
    initial_cases=[]
    for variant,directory in (("mdpf","experiment0002_importance"),
                              ("am_dpf","experiment0003_importance_init")):
        arch=copy.deepcopy(architecture)
        arch.update(yaml.safe_load(verify(EXPERIMENT/directory/"config.yaml"))["models"])
        for dtype_name in ("float64","float32"):
            dtype=getattr(torch,dtype_name)
            model=build_model(common["model_params"],arch,checkpoint,dtype)
            noise_parts=[]
            normal_original=torch.distributions.Normal.sample
            circular_original=VM.sample
            def normal_capture(dist,*args,**kwargs):
                value=normal_original(dist,*args,**kwargs); noise_parts.append(value.clone()); return value
            def circular_capture(dist,*args,**kwargs):
                value=circular_original(dist,*args,**kwargs); noise_parts.append(value.clone()); return value
            torch.manual_seed(20261007)
            torch.distributions.Normal.sample=normal_capture
            VM.sample=circular_capture
            data={k:v.to(dtype=dtype) if k!="ground_truth_mask" else v
                  for k,v in training_rows.items()}
            try:
                initial=model.create_initial_dpf_state(data["states"][:,0,:],data["observations"],25)
            finally:
                torch.distributions.Normal.sample=normal_original; VM.sample=circular_original
            if len(noise_parts)!=3:
                raise RuntimeError("Unexpected source initialization sample count")
            initial_cases.append(dict(variant=variant,dtype=dtype_name,noise=host(torch.stack(noise_parts,-1)),
                output=host(initial),data=host(data),constants=constants(model),batch_size=64))
    pre_cases=[]
    for dtype_name in ("float64","float32"):
        dtype=getattr(torch,dtype_name)
        arch=copy.deepcopy(architecture); arch.update(config["models"])
        model=build_model(pre_spec["model_params"],arch,checkpoint,dtype)
        for param in model.parameters(): param.requires_grad_(False)
        class Collector:
            def __init__(self): self.rates={}; self.optimizers=[]
            def add_optimizer_and_lr_scheduler(self,modules,names,lr):
                if lr=="Freeze": return
                params=[p for module in modules for p in module.parameters()]
                for param in params: param.requires_grad_(True)
                self.optimizers.append(torch.optim.Adam(params,lr=lr,weight_decay=pretraining["weight_decay"]))
                for name,param in model.named_parameters():
                    if any(param is p for p in params):
                        if name in self.rates: raise RuntimeError("Duplicate optimizer coverage")
                        self.rates[name]=lr
        collector=Collector()
        model.create_and_add_optimizers(pretraining,collector,"weight")
        params=dict(model.named_parameters()); trained={k:v for k,v in params.items() if v.requires_grad}
        if set(trained)!=set(collector.rates):
            raise RuntimeError("Pretraining optimizer coverage mismatch")
        data={k:v.to(dtype=dtype) if k!="ground_truth_mask" else v
              for k,v in training_rows.items()}
        loss_fn=ref.KDE_NLL(pretraining["loss_params"],model)
        cloud=cloud_cls(pretraining["particle_cloud_params"])
        trainer=SimpleNamespace(main_model=model,device=torch.device("cpu"),
            problem=SimpleNamespace(observation_transformer=observation_cls()),
            particle_cloud_generator=cloud,loss_function=loss_fn)
        trainer.get_true_masked_data=MethodType(masked,trainer)
        parameters_before=host(params); updates=[]
        original_cloud=cloud.create_cloud
        for i in range(2):
            torch.manual_seed(20261007+i)
            capture={}
            def cloud_capture(states):
                value=original_cloud(states);capture["cloud"]=host(value);return value
            cloud.create_cloud=cloud_capture
            model.zero_grad(set_to_none=True)
            tick=time.monotonic()
            objective,effective=forward(trainer,data,None,0)
            objective.backward()
            gradients={name:host(p.grad) for name,p in trained.items()}
            if any(p.grad is None for p in trained.values()): raise RuntimeError("Missing source gradient")
            optimizer_gradients={name:host(p.grad+pretraining["weight_decay"]*p) for name,p in trained.items()}
            norm=torch.sqrt(sum(torch.sum(p.grad*p.grad) for p in trained.values()))
            for opt in collector.optimizers: opt.step()
            moments={}
            for opt in collector.optimizers:
                for name,param in trained.items():
                    if param in opt.state: moments[name]=host(opt.state[param])
            selected_states,selected_obs,_=trainer.get_true_masked_data(data)
            updates.append(dict(data=host(data),captured=capture,effective_batch=effective,
                selected_states=host(selected_states),selected_observations=host(selected_obs),
                loss=host(objective),gradient_norm=host(norm),raw_gradients=gradients,
                optimizer_gradients=optimizer_gradients,parameters_after=host(trained),
                adam_after=moments,wall_seconds=time.monotonic()-tick))
            cloud.create_cloud=original_cloud
        pre_cases.append(dict(variant="mdpf",dtype=dtype_name,batch_size=320,
            parameters=parameters_before,trainable_names=list(trained),learning_rates=collector.rates,
            constants=constants(model),updates=updates,
            fixture_role="source fitted checkpoint, development mechanics only"))
    from reference_kdm_stage_transfer import export_transfers
    transfers = export_transfers(ref, pre_cases, json.loads(baseline.read_text()),
        architecture, common, checkpoint, args.output)
    for module in list(sys.modules.values()):
        filename=getattr(module,"__file__",None)
        if filename and Path(filename).suffix==".py" and Path(filename).is_relative_to(AUTHOR):
            verify(Path(filename))
    for relative in ("src/trainers/trainer.py","src/trainers/trainer_utils.py"):
        verify(AUTHOR/relative)
    result=dict(schema="kdm_source_stages_author_v1",status="PASS",replication="INCOMPLETE",
        cpu_only=True,gpu_intentionally_hidden=True,torch_version=torch.__version__,
        author_commit=ref.PIN,author_hashes=ref.HASHES,checkpoint=str(checkpoint),
        checkpoint_sha256=digest,baseline=str(baseline),baseline_sha256=baseline_hash,
        dataset_config=data_config,datasets=datasets,initial_cases=initial_cases,
        pretraining_cases=pre_cases,stage_transfers=transfers,pretraining_config=pretraining,
        full_training_config=historical["configurations"]["full_training"]["training_params"],
        wall_seconds=time.monotonic()-start,
        not_concluded=["complete training","trained-method parity","default readiness","HMC"])
    write(args.output/"result.json",result)
    print(json.dumps(dict(status="PASS",initial_cases=len(initial_cases),pretraining_cases=len(pre_cases),
                          wall_seconds=result["wall_seconds"])))
    return 0
if __name__=="__main__":
    raise SystemExit(main())
