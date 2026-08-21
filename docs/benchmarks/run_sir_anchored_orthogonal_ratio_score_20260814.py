"""Run V4 anchored-orthogonal observation-only ratio-score references."""
from __future__ import annotations
import argparse, ast, hashlib, json, math, os, platform, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
import tensorflow as tf
from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
GPU_MEMORY_POLICY = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
tf.config.experimental.enable_tensor_float_32_execution(False)
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from bayesfilter.independent_score import gaussian_observation_simulator_tf as gaussian_simulator  # noqa: E402
from bayesfilter.independent_score import sir_observation_simulator_tf as sir_simulator  # noqa: E402
from bayesfilter.independent_score.anchored_orthogonal_ratio_score_tf import (  # noqa: E402
    ARCHITECTURES, DELTA_SCALE, DELTAS, basis_diagnostics, fit_anchored_classifier,
)

THETA = tf.zeros([3], tf.float64)
HORIZONS = (20, 40, 50); COORDINATES = (0, 1, 2); REGULARIZATION = (0.0, 1.0e-5)
ROOT_SEED = 89500; FINAL_REPLICATES = 3; BATCH_SIZE = 2048
OUTPUT_DEFAULT = ROOT / "docs/benchmarks/artifacts/sir_anchored_orthogonal_ratio_score_20260814"
PLAN_PATH = ROOT / "docs/plans/bayesfilter-sir-anchored-orthogonal-ratio-score-v4-plan-2026-08-14.md"
REVIEW_PATH = ROOT / "docs/plans/bayesfilter-sir-anchored-orthogonal-ratio-score-v4-plan-review-2026-08-14.md"
IMPLEMENTATION_PATH = ROOT / "bayesfilter/independent_score/anchored_orthogonal_ratio_score_tf.py"
RUNNER_PATH = Path(__file__).resolve()

def safe(x: Any) -> Any:
    if isinstance(x, dict): return {str(k): safe(v) for k,v in x.items()}
    if isinstance(x, (list,tuple)): return [safe(v) for v in x]
    if isinstance(x, tf.Tensor): return safe(x.numpy().tolist())
    if isinstance(x,float) and not math.isfinite(x): return None
    return x
def write(path: Path, payload: dict[str,Any]) -> None: path.write_text(json.dumps(safe(payload),indent=2,sort_keys=True)+"\n",encoding="utf-8")
def sha(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()
def _forbidden_loaded_modules() -> list[str]:
    """Reject state-estimation imports even when source AST remains clean."""
    tokens = ("highdim", "filtering", "filters", "particle", "particles", "smoothing", "simulation_score_tf")
    return sorted(
        name for name in sys.modules
        if name.startswith("bayesfilter.") and any(token in name.lower().split(".") for token in tokens)
    )


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.STDOUT
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def seed(*parts: int) -> int:
    value=ROOT_SEED
    for part in parts: value=(value*1009+int(part)+7919)%2147483000
    return value
def profile(name: str) -> dict[str,int]:
    if name=="full": return {"train":2048,"validation":512,"calibration":512,"test":1024,"batch":2048,"epochs":80,"minimum":15,"patience":10}
    if name=="smoke": return {"train":64,"validation":32,"calibration":32,"test":64,"batch":128,"epochs":4,"minimum":2,"patience":2}
    raise ValueError("profile must be full or smoke")
def audit_source() -> dict[str,Any]:
    paths=[ROOT/"bayesfilter/independent_score/anchored_orthogonal_ratio_score_tf.py",ROOT/"bayesfilter/independent_score/gaussian_observation_simulator_tf.py",ROOT/"bayesfilter/independent_score/sir_observation_simulator_tf.py",Path(__file__).resolve()]
    violations=[]; imports={}
    banned={"highdim","filtering","filters","particle","particles","smoothing","simulation_score_tf"}
    for path in paths:
        tree=ast.parse(path.read_text()); modules=[]
        for node in ast.walk(tree):
            if isinstance(node,ast.Import): modules.extend(a.name for a in node.names)
            elif isinstance(node,ast.ImportFrom): modules.append(node.module or "")
        imports[path.name]=modules
        for module in modules:
            if set(module.lower().split(".")) & banned: violations.append({"path":str(path),"module":module})
    return {"imports":imports,"violations":violations,"passed":not violations}
def noise(count:int, s:int):
    return (tf.random.stateless_normal([count,18],[seed(s,1),11],dtype=tf.float64),tf.random.stateless_normal([count,50,18],[seed(s,2),13],dtype=tf.float64),tf.random.stateless_normal([count,50,9],[seed(s,3),17],dtype=tf.float64))
def dataset(kind:str, *, coordinate:int, role:int, replicate:int, count:int, domain:int):
    xs=[]; ds=[]; ys=[]; direction=tf.one_hot(coordinate,3,dtype=tf.float64)
    for index,delta in enumerate(DELTAS):
        base=seed(domain,coordinate,index,role,replicate)
        if kind=="sir":
            sim=sir_simulator.make_compiled_observation_simulator(50); minus=sim(THETA-tf.cast(delta,tf.float64)*direction,*noise(count,seed(base,0))); plus=sim(THETA+tf.cast(delta,tf.float64)*direction,*noise(count,seed(base,1)))
        else:
            sim=gaussian_simulator.make_compiled_observation_simulator(50); mn=tf.random.stateless_normal([count,50,9],[seed(base,0),31],dtype=tf.float64); pn=tf.random.stateless_normal([count,50,9],[seed(base,1),37],dtype=tf.float64); minus=sim(THETA-tf.cast(delta,tf.float64)*direction,mn); plus=sim(THETA+tf.cast(delta,tf.float64)*direction,pn)
        xs.extend([minus,plus]); ds.extend([tf.fill([count],tf.cast(delta,tf.float32)),tf.fill([count],tf.cast(delta,tf.float32))]); ys.extend([tf.zeros([count],tf.float32),tf.ones([count],tf.float32)])
    return tf.concat(xs,0),tf.concat(ds,0),tf.concat(ys,0)
def splits(kind:str, *, coordinate:int, replicate:int, domain:int, cfg:dict[str,int]):
    return {name:dataset(kind,coordinate=coordinate,role=role,replicate=replicate,count=cfg[name],domain=domain) for name,role in (("train",1),("validation",2),("calibration",3),("test",4))}
def fit_row(sp, *, stage:str, horizon:int, coordinate:int, replicate:int, architecture:str, l2:float, cfg:dict[str,int]):
    tr,va,ca,te=(sp[k] for k in ("train","validation","calibration","test"))
    fit=fit_anchored_classifier(tr[0][:,:horizon,:],tr[1],tr[2],validation_observations=va[0][:,:horizon,:],validation_deltas=va[1],validation_labels=va[2],calibration_observations=ca[0][:,:horizon,:],calibration_deltas=ca[1],calibration_labels=ca[2],test_observations=te[0][:,:horizon,:],test_deltas=te[1],test_labels=te[2],architecture=architecture,seed=seed(900 if stage=="exact_oracle" else 1000,horizon,coordinate,replicate),expected_deltas=DELTAS,epochs=cfg["epochs"],minimum_epochs=cfg["minimum"],patience=cfg["patience"],batch_size=cfg["batch"],l2=l2,jit_compile=True)
    obs=(gaussian_simulator.fixed_observed_path(horizon) if stage=="exact_oracle" else sir_simulator.fixed_observed_path(81120,horizon))[None,...]; score=float(fit.score_at_observation(obs)[0].numpy())
    per={}; direction=tf.one_hot(coordinate,3,dtype=tf.float64)
    for delta in DELTAS:
        key=str(float(delta)); auc=float(fit.test_auc_by_delta[key].numpy()); ece=float(fit.test_ece_by_delta[key].numpy()); logit=float(fit.calibrated_logit(obs,tf.constant([delta],tf.float32))[0].numpy()); lo=float(fit.test_logit_minimum_by_delta[key].numpy()); hi=float(fit.test_logit_maximum_by_delta[key].numpy()); ex=.1*max(hi-lo,1.); per[key]={"auc":auc,"ece":ece,"observed_logit":logit,"support":lo-ex<=logit<=hi+ex,"logit_support":[lo,hi]}
    aucs=[per[str(float(d))]["auc"] for d in DELTAS]; eces=[per[str(float(d))]["ece"] for d in DELTAS]
    admission={"finite":bool(fit.finite.numpy()),"pooled_signal":float(fit.test_log_loss.numpy())<math.log(2)-2*float(fit.test_log_loss_standard_error.numpy()),"calibration_not_worse":float(fit.calibration_log_loss_after.numpy())<=float(fit.calibration_log_loss_before.numpy())+1e-4,"temperature_positive":float(fit.calibration_temperature.numpy())>0,"per_delta_ece":max(eces)<=.04,"informative_deltas":sum(a>.52 for a in aucs)>=2,"auc_not_inverted":all(aucs[i+1]>=aucs[i]-.03 for i in range(len(aucs)-1)),"max_delta_not_separated":aucs[-1]<=.995,"support_all_deltas":all(v["support"] for v in per.values()),"optimizer_complete":fit.epochs_run<cfg["epochs"] or fit.final_ten_epoch_improvement<1e-4}
    return {"stage":stage,"horizon":horizon,"coordinate":coordinate,"replicate":replicate,"architecture":architecture,"l2":l2,"score_estimate":score,"best_epoch":fit.best_epoch,"epochs_run":fit.epochs_run,"final_ten_epoch_improvement":fit.final_ten_epoch_improvement,"validation_log_loss":float(fit.validation_log_loss.numpy()),"validation_log_loss_standard_error":float(fit.validation_log_loss_standard_error.numpy()),"calibration_temperature":float(fit.calibration_temperature.numpy()),"test_log_loss":float(fit.test_log_loss.numpy()),"test_log_loss_standard_error":float(fit.test_log_loss_standard_error.numpy()),"per_delta":per,"admission":admission,"admitted":all(admission.values())}
def select(rows,horizon,coordinate):
    c=[r for r in rows if r["horizon"]==horizon and r["coordinate"]==coordinate]; c.sort(key=lambda r:(r["validation_log_loss"],r["architecture"],r["l2"])); best=c[0]; simple=[r for r in c if r["architecture"]=="anchored_linear_quadratic" and r["validation_log_loss"]<=best["validation_log_loss"]+best["validation_log_loss_standard_error"]]; s=simple[0] if simple else best; return {"architecture":s["architecture"],"l2":s["l2"],"validation_log_loss":s["validation_log_loss"],"candidates":c}
def summarize(rows,stage):
    out={}; allpass=True
    for t in HORIZONS:
        for j in COORDINATES:
            vals=[r["score_estimate"] for r in rows if r["horizon"]==t and r["coordinate"]==j and r["admitted"]]; mean=sum(vals)/len(vals) if vals else None; se=(sum((v-mean)**2 for v in vals)/(len(vals)*(len(vals)-1)))**.5 if len(vals)>1 else None; rg=max(vals)-min(vals) if vals else None; gates={"three_replicates":len(vals)==3,"finite":mean is not None and se is not None,"range":rg is not None and rg<=max(2,4*se) if se is not None else False,"precision":se is not None and se<=max(1,.25*abs(mean)) if mean is not None else False}; exact=None
            if stage=="exact_oracle": exact=float(gaussian_simulator.exact_score(THETA,gaussian_simulator.fixed_observed_path(t))[j].numpy()); gates["exact_error"]=mean is not None and se is not None and abs(mean-exact)<=max(.5,3*se)
            admitted=all(gates.values()); allpass=allpass and admitted; out[f"T{t}_j{j}"]={"replicate_scores":vals,"mean":mean,"standard_error":se,"range":rg,"exact_score":exact,"gates":gates,"reference_admitted":admitted,"status":"admitted" if admitted else "no_anchored_ratio_reference"}
    return out,allpass
def run(outroot:Path,*,stage:str,profile_name:str,oracle:Path|None):
    if stage=="sir":
        if oracle is None:
            raise ValueError("SIR requires passed exact oracle")
        oracle_payload = json.loads(oracle.read_text())
        if (oracle_payload.get("status") != "PASSED"
                or oracle_payload.get("stage") != "exact_oracle"
                or oracle_payload.get("profile") != "full"
                or not oracle_payload.get("all_reference_cells_admitted", False)):
            raise ValueError("SIR requires a PASSED full exact oracle with all cells admitted")
    cfg=profile(profile_name); audit=audit_source(); banned=_forbidden_loaded_modules()
    if not audit["passed"]: raise RuntimeError("source dependency veto")
    if banned: raise RuntimeError(f"runtime dependency veto: {banned}")
    outroot.mkdir(parents=True,exist_ok=False); kind="gaussian" if stage=="exact_oracle" else "sir"; started=time.perf_counter(); bdiag=basis_diagnostics()
    static_paths = (PLAN_PATH, REVIEW_PATH, IMPLEMENTATION_PATH, RUNNER_PATH)
    manifest={"schema":"bayesfilter.anchored_orthogonal_ratio_score.manifest.v1","status":"RUNNING","stage":stage,"profile":profile_name,"method":"anchored_discrete_orthogonal_conditional_ratio","score_identity":"calibrated_c0/(2*delta_scale)","deltas":DELTAS,"delta_scale":DELTA_SCALE,"basis":bdiag,"batch_size":cfg["batch"],"pooled_training_rows":cfg["train"]*2*len(DELTAS),"selection_domain":50,"final_domain":60,"gpu_memory_policy":GPU_MEMORY_POLICY,"python":sys.executable,"cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES","unset"),"xla_flags":os.environ.get("XLA_FLAGS","unset"),"git_commit":_git_commit(),"source_hashes":{str(path.relative_to(ROOT)):sha(path) for path in static_paths},"source_audit":audit,"runtime_module_audit":{"forbidden_loaded_modules":banned,"passed":not banned},"oracle_provenance":{"exact_gaussian":str(gaussian_simulator.__file__),"sir_observation_simulator":str(sir_simulator.__file__),"sir_fixed_observation_seed":81120,"sir_oracle_status":"not_an_oracle"}}
    write(outroot/"run_manifest.json",manifest); selection=[]
    for j in COORDINATES:
        sp=splits(kind,coordinate=j,replicate=0,domain=50,cfg=cfg)
        for t in HORIZONS:
            for a in ARCHITECTURES:
                for l2 in REGULARIZATION: selection.append(fit_row(sp,stage=stage,horizon=t,coordinate=j,replicate=0,architecture=a,l2=l2,cfg=cfg))
    selected={f"T{t}_j{j}":select(selection,t,j) for t in HORIZONS for j in COORDINATES}; write(outroot/"selected_controls.json",{"selected":selected,"rows":selection}); final=[]
    for j in COORDINATES:
        for rep in range(FINAL_REPLICATES):
            sp=splits(kind,coordinate=j,replicate=rep,domain=60,cfg=cfg)
            for t in HORIZONS:
                c=selected[f"T{t}_j{j}"]; row=fit_row(sp,stage=stage,horizon=t,coordinate=j,replicate=rep,architecture=c["architecture"],l2=float(c["l2"]),cfg=cfg); final.append(row); write(outroot/f"row_{len(final)-1:04d}.json",row)
    summary,allpass=summarize(final,stage); result={"schema":"bayesfilter.anchored_orthogonal_ratio_score.result.v1","stage":stage,"profile":profile_name,"status":"PASSED" if allpass and profile_name=="full" else ("SMOKE_COMPLETED" if profile_name=="smoke" else "FAILED"),"all_reference_cells_admitted":allpass,"basis":bdiag,"selected":selected,"summary":summary,"rows":len(final),"nonclaims":["not exact SIR score","not filter correctness or ranking","not HMC/default readiness"]}; write(outroot/"result.json",result); manifest.update({"status":result["status"],"wall_time_seconds":time.perf_counter()-started,"finished_at":datetime.now(timezone.utc).isoformat(),"result_sha256":sha(outroot/"result.json")}); write(outroot/"run_manifest.json",manifest)
def main():
    p=argparse.ArgumentParser(); p.add_argument("--output-root",type=Path,required=True); p.add_argument("--stage",choices=("exact_oracle","sir"),required=True); p.add_argument("--profile",choices=("full","smoke"),default="full"); p.add_argument("--oracle-result",type=Path); a=p.parse_args(); run(a.output_root,stage=a.stage,profile_name=a.profile,oracle=a.oracle_result)
if __name__=="__main__": main()
