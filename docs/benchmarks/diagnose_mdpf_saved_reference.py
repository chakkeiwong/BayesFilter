"""CPU-only independent author-reference saved-trial diagnostics.
PyTorch is used only to inspect and compare the source reference, not as an
eligible BayesFilter algorithm backend. No training or checkpoint selection.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
os.environ["OMP_NUM_THREADS"]="1"
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
ROOT=Path("/home/chakwong/BayesFilter")
PARENT=ROOT/".localresources/worktrees/ledh-iapf-execution-20260922/docs/plans/artifacts/iapf-kdm-replication-20260924-02/stage0"
EVAL=PARENT/"attempt17"
TRAIN=PARENT/"attempt16"

def digest(p):
    h=hashlib.sha256()
    with p.open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):h.update(block)
    return h.hexdigest()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();out=args.output;start=time.monotonic()
    sys.path.insert(0,str(EVAL/"author-src"))
    import torch
    torch.set_num_threads(1)
    def load(p):return torch.load(p,map_location="cpu",weights_only=False)
    contract=json.loads((EVAL/"contract.json").read_text())
    checkpoint=Path(contract["checkpoint"])
    if digest(checkpoint)!=contract["checkpoint_sha256"]:raise RuntimeError("checkpoint mismatch")
    data=load(EVAL/"diagnostic-evaluation-data.pt")
    train=load(TRAIN/"datasets/training.pt")
    raw=load(EVAL/"evaluation/raw_all_sequence_evaluation_run_data.pt")
    metrics=load(EVAL/"evaluation/raw_metric_files/raw_rmse.pt")
    nll=load(EVAL/"evaluation/raw_metric_files/raw_nll.pt")
    truth=data["car_states"].double()
    n,t,d=truth.shape
    if (n,t,d)!=(5000,150,3) or sorted(raw)!=list(range(n)):raise RuntimeError("data scope mismatch")
    def wrap(x):return torch.atan2(torch.sin(x),torch.cos(x))
    predictions=[]
    for i in range(n):
        particles=raw[i]["particles"];weights=raw[i]["particle_weights"]
        xy=(particles[...,:2]*weights[...,None]).sum(-2)
        angle=torch.atan2((torch.sin(particles[...,2])*weights).sum(-1),
                          (torch.cos(particles[...,2])*weights).sum(-1))
        predictions.append(torch.cat([xy,angle[:,None]],-1).double())
    learned=torch.stack(predictions)
    del raw
    def errors(pred):
        delta=pred-truth
        delta[...,2]=wrap(delta[...,2])
        return delta.square().sum(-1),delta[...,:2].square().sum(-1)
    rmse,xy=errors(learned)
    source=torch.empty_like(rmse)
    source[metrics["indices"].long()]=metrics["raw_metric"].double()
    source_error=(rmse.sqrt()-source.sqrt()).abs().max().item()
    if source_error>1e-4:raise RuntimeError(f"source metric parity failed: {source_error}")
    training=train["car_states"].double()
    speed=torch.linalg.vector_norm(training[:,1:,:2]-training[:,:-1,:2],dim=-1).median()
    center=torch.cat([training[...,:2].mean((0,1)),
        torch.atan2(torch.sin(training[...,2]).mean(),torch.cos(training[...,2]).mean()).reshape(1)])
    sensor=torch.tensor([-5.,0.],dtype=torch.float64)
    tr_angle=torch.atan2(training[...,1]-sensor[1],training[...,0]-sensor[0])
    threshold=torch.quantile(wrap(train["sensor_obs"][...,0].double()-tr_angle).abs(),.9)
    angle=torch.atan2(truth[...,1]-sensor[1],truth[...,0]-sensor[0])
    unusual=wrap(data["sensor_obs"][...,0].double()-angle).abs()>threshold
    initial=learned[:,0]
    persistence=initial[:,None,:].expand(-1,t,-1).clone()
    motion=persistence.clone()
    step=torch.arange(t,dtype=torch.float64)[None,:]
    motion[...,0]+=speed*step*torch.cos(initial[:,2,None])
    motion[...,1]+=speed*step*torch.sin(initial[:,2,None])
    central=center[None,None,:].expand(n,t,d).clone()
    cases={"MDPF_saved":(rmse,xy),
           "initial_persistence":errors(persistence),
           "constant_motion":errors(motion),
           "training_center":errors(central)}
    summaries=[];per_trajectory={}
    periods={"all":(0,150),"early":(0,17),"middle":(17,50),"late":(50,150)}
    observed_losses=[]
    for period,(lo,hi) in periods.items():
        time_mask=torch.zeros(n,t,dtype=torch.bool);time_mask[:,lo:hi]=True
        for residual,mask in [("all",torch.ones_like(unusual)),
                              ("ordinary",~unusual),("unusual",unusual)]:
            selected=time_mask&mask;counts=selected.sum(1);valid=counts>0
            key=period+"_"+residual
            conditional={}
            for name,(total,position) in cases.items():
                if not torch.isfinite(total).all() or not torch.isfinite(position).all():
                    raise RuntimeError("nonfinite comparison")
                means=((total*selected).sum(1)/counts.clamp_min(1)).sqrt()
                pmeans=((position*selected).sum(1)/counts.clamp_min(1)).sqrt()
                conditional[name]=means[valid].mean().item()
                summaries.append(dict(period=period,bearing_residual=residual,method=name,
                    trajectories=int(valid.sum()),timepoints=int(selected.sum()),
                    source_rmse_mean=conditional[name],position_error_mean=pmeans[valid].mean().item()))
                per_trajectory[key+"_"+name]=dict(source_rmse=means,position_error=pmeans,counts=counts)
            for name,value in conditional.items():
                if name!="MDPF_saved" and conditional["MDPF_saved"]>value:
                    observed_losses.append(dict(period=period,residual=residual,heuristic=name,
                        mdpf=conditional["MDPF_saved"],heuristic_value=value))
    torch.save(per_trajectory,out/"conditional-per-trajectory.pt")
    with (out/"conditional.csv").open("w") as stream:
        w=csv.DictWriter(stream,fieldnames=list(summaries[0]));w.writeheader();w.writerows(summaries)
    controls={"source_metric_max_absolute_error":source_error,"checkpoint_sha256":digest(checkpoint),
        "evaluation_data_sha256":digest(EVAL/"diagnostic-evaluation-data.pt"),
        "training_data_sha256":digest(TRAIN/"datasets/training.pt"),
        "source_rmse_mean":source.mean(1).sqrt().mean().item(),
        "source_aggregation":"mean over trajectories of sqrt(mean over time of sum of squared state errors)","source_nll_mean":nll["raw_metric"].double().mean().item(),
        "training_median_displacement":speed.item(),"training_center":center.tolist(),
        "training_q90_bearing_error":threshold.item(),"cpu_only":True,"gpu_intentionally_hidden":True}
    (out/"controls.json").write_text(json.dumps(controls,indent=2)+"\n")
    result={"status":"COMPLETE","primary_criterion":"SAVED_METRIC_PARITY_PASS",
        "heuristic_dominance":"OBSERVED_CONDITIONAL_LOSS_POINT_ESTIMATION_PROMOTION_VETO" if observed_losses else
            "CHEAP_POINT_SCREEN_PASSED_FULL_DENSITY_GATE_INCOMPLETE",
        "observed_losses":observed_losses,"statistically_supported_ranking":False,
        "inference_unit":"one saved training seed; descriptive conditional trajectory means",
        "default_ready":False,"replication":"INCOMPLETE",
        "remaining":["Appendix B full learning experiment","fitted UKF and BPF density baselines",
                     "NLL versus MSE refinement","independent training-seed comparisons","TF GPU consumer parity"],
        "wall_seconds":time.monotonic()-start}
    (out/"result.json").write_text(json.dumps(result,indent=2)+"\n")
    with (out/"decision.csv").open("w") as stream:
        w=csv.DictWriter(stream,fieldnames=["status","heuristic_dominance","statistically_supported_ranking"])
        w.writeheader();w.writerow({k:result[k] for k in w.fieldnames})
    print(json.dumps({"status":result["status"],"heuristic_dominance":result["heuristic_dominance"],
                      "conditional_losses":len(observed_losses),"wall_seconds":result["wall_seconds"]}))

if __name__=="__main__":
    main()
