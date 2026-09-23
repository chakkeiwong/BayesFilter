"""Post-run paired training comparisons; cannot admit posterior estimates."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    request=read(args.request)
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS","2")
    os.environ.setdefault("TF_NUM_INTEROP_THREADS","2")
    os.environ.setdefault("OMP_NUM_THREADS","2")
    began=time.monotonic()
    import tensorflow as tf
    args.output.mkdir(parents=True,exist_ok=False)
    baseline=read(request["baseline"])
    if baseline["role"]!="untouched-final-bank" or baseline["rows"]!=2048:
        raise ValueError("baseline must use the declared independent final bank")
    if len(request["candidates"])>10:
        raise ValueError("candidate inventory exceeds the predeclared multiplicity allowance")
    loss0=tf.constant(baseline["blocks"]["loss"],tf.float64)
    score0=tf.reduce_sum(tf.constant(baseline["blocks"]["residual"],tf.float64)**2,axis=1)
    n=baseline["rows"]
    resamples=4096
    probability=.05/(2.*10.)
    # Both loss and score must pass to assert improvement (intersection test).
    # Each is adjusted across the ten predeclared candidate comparisons.
    @tf.function(input_signature=[tf.TensorSpec([n,2],tf.float64)],jit_compile=True,autograph=False)
    def paired(delta):
        indices=tf.random.stateless_uniform([resamples,n],(20260924,491),maxval=n,dtype=tf.int32)
        draws=tf.reduce_mean(tf.gather(delta,indices),axis=1)
        ordered=tf.sort(draws,axis=0)
        bounds=tf.gather(ordered,[int(math.floor(probability*(resamples-1))),
            int(math.floor((1.-probability)*(resamples-1)))])
        return {"mean":tf.reduce_mean(delta,axis=0),"bootstrap_interval":bounds,
            "standard_error":tf.math.reduce_std(delta,axis=0)/tf.sqrt(tf.constant(n-1,tf.float64))}
    rows=[]
    for candidate in request["candidates"]:
        final=read(candidate["measurement"])
        checkpoint=read(candidate["checkpoint"])
        finalized=read(candidate["finalized"])
        probe=finalized["post_training"]
        parameter_hash=hashlib.sha256(json.dumps(checkpoint["parameters"],sort_keys=True).encode()).hexdigest()
        if (final["transport_config"]!=checkpoint["transport_config"]
                or final["map_parameters_sha256"]!=parameter_hash
                or checkpoint["checkpoint_hash"]!=finalized["checkpoint"]["checkpoint_hash"]
                or probe["rows"]!=1000 or probe["valid_rows"]!=1000
                or not probe["finite"] or not probe["complete"]
                or not finalized["inverse_tail_check"]["passed"]):
            raise ValueError("final measurement/checkpoint or required diagnostic mismatch")
        if final["role"]!=baseline["role"] or final["blocks"]["latent"]!=baseline["blocks"]["latent"]:
            raise ValueError("candidate and baseline do not share the untouched bank")
        if not all(final["blocks"]["valid"]) or not all(baseline["blocks"]["valid"]):
            raise ValueError("invalid target rows cannot support a training conclusion")
        loss=tf.constant(final["blocks"]["loss"],tf.float64)
        score=tf.reduce_sum(tf.constant(final["blocks"]["residual"],tf.float64)**2,axis=1)
        statistics={k:v.numpy().tolist() for k,v in paired(tf.stack((loss-loss0,score-score0),axis=1)).items()}
        for value in tf.nest.flatten(statistics):
            if not math.isfinite(value):
                raise ValueError("nonfinite final statistics")
        row={**candidate,"statistics":statistics,
            "fixed_checkpoint_loss_improvement":statistics["bootstrap_interval"][1][0]<0.,
            "fixed_checkpoint_score_improvement":statistics["bootstrap_interval"][1][1]<0.,
            "standard_1000_point_probe":probe,
            "geometry":{k:v for k,v in final.items() if k not in ("blocks",)},
            "measurement_sha256":hashlib.sha256(Path(candidate["measurement"]).read_bytes()).hexdigest()}
        row["fixed_checkpoint_training_improvement"]=row["fixed_checkpoint_loss_improvement"] and row["fixed_checkpoint_score_improvement"]
        rows.append(row)
    families={}
    for family in sorted({r["family"] for r in rows}):
        members=[r for r in rows if r["family"]==family]
        seeds={r["root_seed"] for r in members}
        families[family]={"seeds":sorted(seeds),"independent_seed_count":len(seeds),
            "replicated_local_improvement":len(seeds)==3 and len(members)==3 and all(
                r["fixed_checkpoint_training_improvement"] for r in members),
            "interpretation":"conditional evidence for these seeded fits versus the saved baseline; no general superiority claim"}
    result={"status":"training_comparison_complete","baseline":request["baseline"],
        "baseline_sha256":hashlib.sha256(Path(request["baseline"]).read_bytes()).hexdigest(),
        "baseline_geometry":{k:v for k,v in baseline.items() if k!="blocks"},
        "comparisons":rows,"families":families,"paired_rows":n,"bootstrap_resamples":resamples,
        "bootstrap_seed":[20260924,491],"interval_probability":probability,"multiplicity_allowance":10,
        "statistics_order":["reverse_kl_loss_difference","squared_score_residual_difference"],
        "inference":"paired percentile bootstrap conditional on fixed maps; uncertainty across training seeds shown separately",
        "method_ranking":"not_estimated","posterior_qualified":False,"tf_version":tf.__version__,
        "wall_seconds":time.monotonic()-began,"command":sys.argv,"request":request,
        "cpu_reference":"post-run statistical analysis; GPUs deliberately hidden", "finished_at":datetime.now(timezone.utc).isoformat()}
    (args.output/"result.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"families":families,"wall_seconds":result["wall_seconds"]}),flush=True)


if __name__=="__main__":
    main()
