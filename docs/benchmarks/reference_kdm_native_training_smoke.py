"""Independent author CPU reference for a bounded autonomous-training smoke."""
import argparse, hashlib, importlib.util, json, os, sys, time
from pathlib import Path
from types import SimpleNamespace, MethodType
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
os.environ["OMP_NUM_THREADS"]="1"
ROOT=Path("/home/chakwong/BayesFilter")
BASE=ROOT/"docs/plans/artifacts/controlled-smc-kdm-replication-20260928"
SEED=202610111
BATCH=64
UPDATES=64

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,allow_nan=False)+"\n")

def main(out,reference):
    assert sys.flags.isolated
    out.mkdir(exist_ok=False)
    spec=importlib.util.spec_from_file_location("reference_helpers",ROOT/"docs/benchmarks/reference_kdm_neural_exact_replay.py")
    ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
    torch=ref.torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.set_default_dtype(torch.float32)
    fixtures=[c for c in json.loads(reference.read_text())["cases"] if c["dtype"]=="float32"]
    cfgpath=BASE/"kdm-complete-training-20261007-03/diagnostic/configurations.json"
    configs={x["name"]:x for x in json.loads(cfgpath.read_text())["executed_stages"]}
    get_item=ref.source_member(ref.AUTHOR/"src/datasets/bearings_only_dataset.py","BearingsOnlyDataset","__getitem__")
    datasets={};data_hashes={}
    order=torch.randperm(5000,generator=torch.Generator().manual_seed(SEED))[:BATCH*UPDATES].tolist()
    for split,indexes in (("training",order),("validation",list(range(256)))):
        path=BASE/("kdm-complete-training-20261007-03/diagnostic/datasets/"+split+".pt")
        saved=torch.load(path,map_location="cpu",weights_only=False);data_hashes[str(path)]=sha(path)
        owner=SimpleNamespace(car_states=saved["car_states"],sensor_obs=saved["sensor_obs"],sparse_ground_truth_keep_modulo=4)
        rows=[get_item(owner,i) for i in indexes]
        datasets[split]={k:torch.stack([r[k] for r in rows]) for k in ("states","observations","ground_truth_mask")}
        assert datasets[split]["states"].shape==(len(indexes),17,3)
        assert torch.equal(datasets[split]["ground_truth_mask"],(torch.arange(17)%4==0).expand(len(indexes),17))
    compact=[{k:c[k] for k in ("variant","dtype","batch_size","parameters","trainable_names","frozen_names","learning_rates","constants")} for c in fixtures]
    write(out/"inputs.json",dict(cases=compact,datasets=ref.host(datasets),training_row_ids=order,validation_row_ids=list(range(256)),seed=SEED,
        initialization="paired difficult checkpoint; fresh optimizer moments; not fresh complete training",
        sampling="native own-state Torch and TF streams; no realized-particle replay"))
    transformer=ref.source_member(ref.AUTHOR/"src/problems/bearings_only/bearings_only_problem.py","BearingsOnlyObservationTransformer")
    trainer_path=ref.AUTHOR/"src/trainers/specific_trainers/full_sequence_trainer.py"
    forward=ref.source_member(trainer_path,"FullSequenceTrainer","do_forward_pass")
    convert=ref.source_member(trainer_path,"FullSequenceTrainer","convert_to_model_dtype")
    records=[]
    for vi,case in enumerate(fixtures):
        tick=time.monotonic();cfg=configs[case["variant"]];architecture={}
        directory={"mdpf":"experiment0002_importance","am_dpf":"experiment0003_importance_init"}[case["variant"]]
        for path in (ref.EXPERIMENT/"configs/models.yaml",ref.EXPERIMENT/directory/"config.yaml"):
            architecture.update(ref.yaml.safe_load(ref.verify(path))["models"])
        assert json.loads(json.dumps(architecture))==cfg["architecture"]
        model=ref.KDEParticleFilter(cfg["configuration"]["model_params"],architecture)
        params=dict(model.named_parameters())
        assert set(params)==set(case["parameters"])
        with torch.no_grad():
            for n,p in params.items():p.copy_(torch.tensor(case["parameters"][n]))
        class Collector:
            def __init__(self):self.optimizers=[];self.rates={}
            def add_optimizer_and_lr_scheduler(self,modules,names,lr):
                for module in modules:
                    ps=list(module.parameters())
                    if lr=="Freeze":
                        for p in ps:p.requires_grad_(False)
                    else:
                        self.optimizers.append(torch.optim.Adam(ps,lr=lr,weight_decay=0.0))
                        for n,p in params.items():
                            if any(p is q for q in ps):
                                assert n not in self.rates
                                self.rates[n]=lr
        collector=Collector();model.create_and_add_optimizers(cfg["configuration"]["training_params"],collector,"full")
        assert collector.rates==case["learning_rates"]
        assert {n for n,p in params.items() if p.requires_grad}==set(case["trainable_names"])
        assert ref.host(model.proposal_model.residual_scale_factor)==case["constants"]["residual_scale"]
        trainer=SimpleNamespace(main_model=model,device=torch.device("cpu"),problem=SimpleNamespace(observation_transformer=transformer()),
            number_of_particles=25,skip_first_step_for_loss=False,max_sequence_length=None,truncated_bptt_modulo=4,loss_shaper=None,
            loss_function=ref.KDE_NLL(dict(kde_params=architecture["dpf_full_training"]["kde"]),model))
        trainer.convert_to_model_dtype=MethodType(convert,trainer)
        def batch(split,i):return {k:v[i*BATCH:(i+1)*BATCH] for k,v in datasets[split].items()}
        def evaluate():
            vals=[];model.eval()
            with torch.random.fork_rng(devices=[]),torch.no_grad():
                torch.manual_seed(SEED+100+vi)
                for i in range(4):
                    loss,effective=forward(trainer,batch("validation",i),None,0)
                    assert effective==BATCH and torch.isfinite(loss)
                    vals.append(float(loss))
            return vals
        before=evaluate();model.train();torch.manual_seed(SEED+vi)
        losses=[];norms=[]
        for i in range(UPDATES):
            for optimizer in collector.optimizers:optimizer.zero_grad(set_to_none=True)
            loss,effective=forward(trainer,batch("training",i),None,0)
            assert effective==BATCH and torch.isfinite(loss)
            loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params.values() if p.requires_grad)
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),100.)
            assert torch.isfinite(norm)
            for optimizer in collector.optimizers:optimizer.step()
            assert all(torch.isfinite(p).all() for p in params.values())
            for optimizer in collector.optimizers:
                assert all(torch.isfinite(v).all() for state in optimizer.state.values() for v in state.values() if torch.is_tensor(v))
            losses.append(float(loss.detach()));norms.append(float(norm))
            if (i+1)%16==0:print(json.dumps(dict(variant=case["variant"],completed_updates=i+1,loss=losses[-1])),flush=True)
        after=evaluate()
        assert all(torch.equal(params[n],torch.tensor(case["parameters"][n])) for n in case["frozen_names"])
        changed=sum(not torch.equal(params[n],torch.tensor(case["parameters"][n])) for n in case["trainable_names"])
        assert changed>0
        checkpoint=out/(case["variant"]+"-final.pt")
        torch.save(dict(model=model.state_dict(),optimizers=[o.state_dict() for o in collector.optimizers]),checkpoint)
        records.append(dict(variant=case["variant"],updates=UPDATES,losses=losses,gradient_norms=norms,
            validation_before=before,validation_after=after,changed_parameters=changed,frozen_parameters_unchanged=True,
            checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),wall_seconds=time.monotonic()-tick))
    write(out/"result.json",dict(status="PASS",scope="NATIVE_TRAINING_SMOKE_NOT_QUALIFICATION",cases=records,seed=SEED,
        cpu_only=True,gpu_intentionally_hidden=True,torch_version=torch.__version__,source_commit=ref.PIN,source_hashes=ref.HASHES,
        inputs_sha256=sha(out/"inputs.json"),reference_sha256=sha(reference),configuration_sha256=sha(cfgpath),
        data_hashes=data_hashes,native_source_sampling=True))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);p.add_argument("--reference",type=Path,required=True)
    a=p.parse_args();main(a.output,a.reference)
