import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import hashlib, json
from pathlib import Path
import tensorflow as tf
ROOT = Path("docs/plans/artifacts/surrogate-hmc-m13-sequential-48h-02/m13-public-tuning/tuning")
OUT = Path(__import__("sys").argv[1])
def load(record, root=ROOT):
    path=root/record["path"]
    if hashlib.sha256(path.read_bytes()).hexdigest()!=record["sha256"]:
        raise ValueError("archive checksum mismatch: "+str(path))
    return tf.io.parse_tensor(tf.io.read_file(str(path)),out_type=tf.as_dtype(record.get("dtype","float64")))
def host(x): return x.numpy().tolist()
p=json.loads((ROOT/"fresh-progress.json").read_text())
r=p["chunks"][-1]["archive"]
v={k:load(a) for k,a in r["tensors"].items()}
delta=v["delta_h"]; accept=v["is_accepted"]; states=v["states"]
limit=1000.
indices=tf.where(tf.abs(delta)>limit)
previous=tf.concat([load(r["initial"])[None],states[:-1]],axis=0)
sort=tf.sort(tf.reshape(tf.abs(delta),[-1]))
n=int(tf.size(sort))
worst=[]
for row,chain in host(indices):
    worst.append(dict(transition=row+1,chain=chain,delta_h=float(delta[row,chain]),
        accepted=bool(accept[row,chain]),previous_state=host(previous[row,chain]),
        retained_state=host(states[row,chain])))
nomroot=ROOT/"nomination"
nom=json.loads((nomroot/"manifest.json").read_text())
nt={k:load(nom["tensors"][k],nomroot) for k in (
    "verification_log_accept_ratio","candidate_summary","health_summary",
    "step_size","num_leapfrog_steps","adaptation_divergence_count",
    "adaptation_force_fallback_count")}
logar=nt.pop("verification_log_accept_ratio")
report=dict(status="ARCHIVE_ANALYSIS_COMPLETE",cpu_only=True,
    gpu_devices_intentionally_hidden=True,target_unchanged=True,source=str(ROOT),
    question="Which archived health check stopped attempt1, and was the same bound applied during nomination?",
    counts=p["counts"],events=p["events"],checksum_validation="passed",
    fresh=dict(energy_error_max=limit,breaches=worst,
        abs_energy_quantiles={str(f):float(sort[min(n-1,int(f*(n-1)))]) for f in (.5,.9,.95,.99,1.)},
        max_abs_energy_by_chain=host(tf.reduce_max(tf.abs(delta),axis=0)),
        accepted_rate_by_chain=host(tf.reduce_mean(tf.cast(accept,tf.float64),axis=0)),
        expected_acceptance_by_chain=host(tf.reduce_mean(tf.exp(tf.minimum(v["log_accept_ratio"],0.)),axis=0)),
        accepted_energy_max=float(tf.reduce_max(tf.where(accept,tf.abs(delta),tf.zeros_like(delta)))),
        force_fallbacks=int(tf.reduce_sum(tf.cast(v["force_fallback"],tf.int32))),
        divergence_count=int(tf.reduce_sum(tf.cast(v["divergence"],tf.int32)))),
    nomination={**{k:host(a) for k,a in nt.items()},
        "candidate_summary_fields":nom["candidate_summary_fields"],
        "health_summary_fields":nom["health_summary_fields"],
        "max_abs_log_accept_ratio":float(tf.reduce_max(tf.abs(logar))),
        "energy_1000_breaches":int(tf.reduce_sum(tf.cast(tf.abs(logar)>limit,tf.int32))),
        "expected_acceptance_by_chain":host(tf.reduce_mean(tf.exp(tf.minimum(logar,0.)),axis=0))},
    nonclaims=["no convergence or posterior qualification","no statistical ranking","no exact original likelihood claim"])
OUT.write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps({k:report[k] for k in ("status","counts","fresh","nomination")},indent=2))
