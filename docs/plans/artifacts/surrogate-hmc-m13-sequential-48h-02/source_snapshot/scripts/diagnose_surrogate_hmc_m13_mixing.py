"""Post-run CPU-only diagnostic for archived surrogate-HMC warmup; no admission."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
import argparse
import hashlib
import json
import math
from pathlib import Path
import tensorflow as tf

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    root=args.root
    progress=json.loads((root/"fresh-progress.json").read_text())
    nomination=json.loads((root/"nomination/manifest.json").read_text())
    def tensor(directory,record):
        path=directory/record["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest()!=record["sha256"]:
            raise ValueError("archive checksum mismatch: "+str(path))
        return tf.io.parse_tensor(tf.io.read_file(str(path)),tf.as_dtype(record.get("dtype","float64")))
    arrays={}; previous=None; initial=[]
    for chunk in progress["chunks"]:
        start=tensor(root,chunk["archive"]["initial"])
        if previous is not None:tf.debugging.assert_equal(start,previous)
        ts={k:tensor(root,v) for k,v in chunk["archive"]["tensors"].items()}
        tf.debugging.assert_equal(ts["final_state"],ts["states"][-1])
        initial.append(start)
        previous=ts["final_state"]
        for k,v in ts.items():
            if k not in ("final_state","final_step_size"):
                arrays.setdefault(k,[]).append(v)
    joined={k:tf.concat(v,axis=0) for k,v in arrays.items()}
    states=joined["states"]
    previous_states=tf.concat([initial[0][None,...],states[:-1]],axis=0)
    delta=states-previous_states
    recent=states[-1000:]
    def values(t):return t.numpy().tolist()
    def correlation(lag):
        a,b=states[:-lag],states[lag:]
        a=a-tf.reduce_mean(a,axis=0); b=b-tf.reduce_mean(b,axis=0)
        return values(tf.math.divide_no_nan(tf.reduce_mean(a*b,axis=0),
            tf.sqrt(tf.reduce_mean(a*a,axis=0)*tf.reduce_mean(b*b,axis=0))))
    payload=dict(role="post-run explanatory diagnostic only",cpu_only=True,
        gpu_intentionally_hidden=True,checksum_and_state_boundary_validation="PASS",
        backend="TensorFlow eager CPU diagnostic exception",draws_per_chain=int(states.shape[0]),
        acceptance_by_chain=values(tf.reduce_mean(tf.cast(joined["is_accepted"],tf.float64),axis=0)),
        squared_jump_by_chain_coordinate=values(tf.reduce_mean(delta*delta,axis=0)),
        lag_correlation_by_chain_coordinate={str(lag):correlation(lag) for lag in (1,10,100)},
        last1000_mean_by_chain_coordinate=values(tf.reduce_mean(recent,axis=0)),
        last1000_sd_by_chain_coordinate=values(tf.math.reduce_std(recent,axis=0)),
        nomination_candidate_fields=nomination["candidate_summary_fields"],
        nomination_candidates=[[v if math.isfinite(v) else None for v in row]
            for row in values(tensor(root/"nomination",nomination["tensors"]["candidate_summary"]))],
        fresh_force_fallbacks=int(tf.reduce_sum(tf.cast(joined["force_fallback"],tf.int32))),
        fresh_divergences=int(tf.reduce_sum(tf.cast(joined["divergence"],tf.int32))),
        source_root=str(root),posterior_qualified=False,ranking_supported=False)
    args.output.write_text(json.dumps(payload,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:payload[k] for k in ["draws_per_chain","acceptance_by_chain",
        "squared_jump_by_chain_coordinate","nomination_candidates","fresh_force_fallbacks"]}))

if __name__=="__main__":main()
