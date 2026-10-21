from pathlib import Path
import dataclasses, hashlib, json, math, os, sys, time
from datetime import datetime, timezone
root=Path(__file__).resolve().parent
source=root/"source"
if os.environ.get("CUDA_VISIBLE_DEVICES")!="-1":raise ValueError("CPU diagnostic requires hidden GPUs")
sys.path.insert(0,str(source))
started=time.monotonic()
import tensorflow as tf
import tensorflow_probability as tfp
from bayesfilter.inference.hmc_precision import HMCPrecisionPolicy, HMCPrecisionTarget
from bayesfilter.inference.hmc_posterior_assessment import HMCPosteriorAssessmentPolicy, assess_posterior
from bayesfilter.inference.hmc_convergence import rank_normalized_split_rhat_summary
from bayesfilter.testing.inference_validation.modern_bridge_audit import source_dependencies
out=root/"mams-sensitivity-01"
out.mkdir(exist_ok=False)
origin=root.parent/"continuation-02/mams-run/posterior"
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())
paths=source_dependencies([source/"bayesfilter/inference/hmc_posterior_assessment.py",source/"bayesfilter/inference/hmc_convergence.py"],repository_root=source)
hashes={p:sha(Path(p)) for p in paths}
inputs={}
report=dict(status="running",started_utc=datetime.now(timezone.utc).isoformat(),
    command=[sys.executable,*sys.argv],environment=sys.executable,
    source=read(source/".validation-source.json"),source_sha256=hashes,input_sha256=inputs,
    driver_sha256=sha(Path(__file__)),versions=dict(tensorflow=tf.__version__,tfp=tfp.__version__),
    gpu_intentionally_hidden=True,use_xla=True,wall_ceiling_seconds=500.,
    plan="docs/plans/bayesfilter-modern-hmc-comparator-precision-continuation-2026-10-13.md",
    data="unchanged archived full-target MAMS model quantities",
    fresh_randomness=False,seed_role="preserved original retained seeds; no new transitions",
    parity_tolerance=dict(relative=1e-9,absolute=1e-12,provenance="diagnostic CPU/GPU float64 rounding allowance, not posterior tolerance"),
    method_ranking=None,default_promotion=False,posterior_reclassification=False,rows=[])
def save():
    (out/"result.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
save()
try:
    for directory in sorted(origin.glob("member-*")):
        if time.monotonic()-started>=490:raise TimeoutError("sensitivity allocation exhausted")
        for name in ("result.json","policy.json","retained_raw.tensor"):
            inputs[str(directory/name)]=sha(directory/name)
        old=read(directory/"result.json");payload=read(directory/"policy.json")
        values=tf.io.parse_tensor(tf.io.read_file(str(directory/"retained_raw.tensor")),out_type=tf.float64)
        settings=payload["assessment_policy"];precision=settings["precision"]
        targets=tuple(HMCPrecisionTarget(**t) for t in precision["targets"])
        fields=HMCPrecisionPolicy.__dataclass_fields__
        original=HMCPrecisionPolicy(targets,**{k:v for k,v in precision.items() if k in fields and k!="targets"})
        guarded=dataclasses.replace(original,method="lugsail_max_batch_means")
        names=tuple(t.name for t in targets)
        assert values.shape==(old["retained_results_per_chain"],4,len(names))
        assessment=HMCPosteriorAssessmentPolicy(**{k:v for k,v in settings.items()
            if k in HMCPosteriorAssessmentPolicy.__dataclass_fields__ and k!="precision"})
        row=dict(member=directory.name,retained_seed=payload["retained_seed"],sample_shape=values.shape.as_list(),
            original_first_passing_saved_look=None,maximum_first_passing_saved_look=None,looks=[])
        report["rows"].append(row)
        for check in old["retained_checks"]:
            if time.monotonic()-started>=490:raise TimeoutError("sensitivity allocation exhausted")
            n=check["completed_results_per_chain"]
            rhat=rank_normalized_split_rhat_summary(values[:n],rhat_max=payload["retained_rhat_max"])
            policies={"original":original,"maximum":guarded}
            results={key:assess_posterior(values[:n],names,stage="retained",rhat=rhat,
                policy=dataclasses.replace(assessment,precision=policy)) for key,policy in policies.items()}
            archived=check["modern_rhat"]
            max_error=0.
            for a,b in zip(archived["precision"]["targets"],results["original"]["precision"]["targets"]):
                if a["valid"]!=b["valid"] or a["passed"]!=b["passed"]:raise ValueError("original precision verdict changed")
                for key in ("estimate","mcse","mcse_sd_ratio"):
                    if a[key] is None or b[key] is None:
                        if a[key]!=b[key]:raise ValueError("original availability changed")
                    elif not math.isclose(a[key],b[key],rel_tol=1e-9,abs_tol=1e-12):
                        raise ValueError("original precision parity failed")
                    else:max_error=max(max_error,abs(a[key]-b[key]))
            if archived["passed"]!=results["original"]["passed"]:raise ValueError("original posterior assessment changed")
            eligible=n>=payload["retained_min_results"] and check["health"]["health_passed"]
            for key in policies:
                label="original" if key=="original" else "maximum"
                if eligible and results[key]["passed"] and row[label+"_first_passing_saved_look"] is None:
                    row[label+"_first_passing_saved_look"]=n
            row["looks"].append(dict(draws=n,original_parity_max_absolute_error=max_error,
                original=results["original"],maximum=results["maximum"]))
            save()
    for name,digest in {**hashes,**inputs}.items():
        if sha(Path(name))!=digest:raise ValueError("source/input changed: "+name)
    report["status"]="completed"
except BaseException as exc:
    report.update(status="repair_required",error=str(exc));raise
finally:
    report.update(elapsed_seconds=time.monotonic()-started,finished_utc=datetime.now(timezone.utc).isoformat())
    save()
