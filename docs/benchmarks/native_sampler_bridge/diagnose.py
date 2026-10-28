"""Post-run diagnostic reader; never controls or retunes a sampler."""
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import argparse
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))

def main():
    parser=argparse.ArgumentParser();parser.add_argument("directory",type=Path);args=parser.parse_args()
    start=time.monotonic()
    import tensorflow as tf
    from bayesfilter.inference.hmc_posterior_diagnostics import rank_normalized_split_rhat,rank_normalized_bulk_tail_ess
    result=json.loads((args.directory/"results.json").read_text());out=[]
    for item in result["fits"]:
        engine,kind=item["engine"],item["kind"];d=item["dimension"];c=item["chains"];w=item["warmup"];s=item["retained"]
        raw=tf.io.decode_raw((args.directory/f"{engine}-{kind}-rows.bin").read_bytes(),tf.float64,little_endian=sys.byteorder=="little")
        draws=tf.reshape(raw,[c,w+s,d+7])[:,w:,7:]
        # Existing post-run TF diagnostics are a CPU reference exception.
        rh=rank_normalized_split_rhat(draws)["maximum"]
        ess=rank_normalized_bulk_tail_ess(draws)
        out.append({"engine":engine,"kind":kind,"rhat_max":float(tf.reduce_max(rh)),
            "bulk_ess_min":float(tf.reduce_min(ess["bulk"])),"tail_ess_min":float(tf.reduce_min(ess["tail"])),
            "means":tf.reduce_mean(draws,axis=[0,1]).numpy().tolist(),
            "variances":tf.math.reduce_variance(draws,axis=[0,1]).numpy().tolist(),
            "funnel_neck_fraction_below_minus3":item["funnel_neck_fraction_below_minus3"],
            "divergences":item["divergences"],"role":"explanatory only; no posterior promotion"})
    artifact={"command":sys.argv,"source":str(args.directory),"jit_compile":False,
        "exception":"CPU post-run reporting; no algorithmic decisions","GPU_intentionally_hidden":True,
        "wall_seconds":time.monotonic()-start,"diagnostics":out}
    path=args.directory/"posterior_diagnostics.json"
    with path.open("x") as f:json.dump(artifact,f,indent=2)
    print(json.dumps(artifact,indent=2))

if __name__=="__main__":main()
