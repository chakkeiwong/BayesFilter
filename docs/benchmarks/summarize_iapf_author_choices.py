"""Post-run diagnostic analysis of independent R hypotheses; standard library only."""
from __future__ import annotations
from collections import defaultdict, Counter
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys

OUT=Path(sys.argv[1])
REPORT=Path(sys.argv[2]) if len(sys.argv)>2 else OUT
if REPORT!=OUT:REPORT.mkdir(parents=True,exist_ok=False)
path=Path(__file__).with_name("run_iapf_author_choices.py")
spec=importlib.util.spec_from_file_location("choice_campaign",path)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.OUT=OUT
records=m.records();attempts=m.attempts()

def mean_ci(values):
    rng=random.Random(9260001)
    samples=[statistics.mean(rng.choices(values,k=len(values))) for _ in range(2000)]
    return [m.quantile(samples,.025),m.quantile(samples,.975)]

def mse(log_error):
    try:return math.expm1(float(log_error))**2
    except OverflowError:return math.inf

def sanitize(x):
    if isinstance(x,float) and not math.isfinite(x):return str(x)
    if isinstance(x,dict):return {k:sanitize(v) for k,v in x.items()}
    if isinstance(x,list):return [sanitize(v) for v in x]
    return x

def label(r):return ":".join(r[k] for k in ("method","floor_rule","sd_mode"))

# Verify executed sources and identical observations across all paired methods.
for source in OUT.glob("source-*"):
    hashes=json.loads((source/"sha256.json").read_text())
    assert all(m.digest(source/name)==value for name,value in hashes.items())
observations={};fixtures=[];heldout=[]
for e in attempts:
    base=Path(e["output"])
    if (base/"observations.csv").exists():
        seed=e["data_seed"];h=m.digest(base/"observations.csv")
        assert observations.setdefault(seed,h)==h,"different paired observations"
    fixtures+=m.read_csv(base/"fixtures.csv")
    heldout+=m.read_csv(base/"heldout.csv")

cells=defaultdict(list)
for r in records:
    cells[(r["stage"]=="validation",r["dimension"],r["data_seed"],label(r))].append(r)
summaries=[]
for (validation,d,seed,config),rs in sorted(cells.items()):
    reps=[r["replication"] for r in rs];assert len(reps)==len(set(reps)),"duplicated replicate"
    result=m.screen(rs,int(d),16 if validation else 4)
    good=[r for r in rs if r["status"]=="complete"]
    summaries.append(dict(validation=validation,dimension=int(d),data_seed=int(seed),configuration=config,
      screen=result,failures=dict(Counter(r["message"] for r in rs if r["status"]!="complete")),
      mean_algorithm_seconds=statistics.mean(float(r["algorithm_seconds"]) for r in good) if good else None,
      mean_resampling=statistics.mean(float(r["resampling_count"]) for r in good) if good else None,
      terminal_mse=statistics.mean((float(r["ratio"])-1)**2 for r in good) if good else None))

prefix_by_result={};floor_probability=[];histories=[]
for e in attempts:
    if e["mode"]!="run":continue
    base=Path(e["output"]);by=defaultdict(list)
    for p in m.read_csv(base/"prefixes.csv"):
        by[(p["method"],p["replication"],p["situation"])].append(mse(p["log_prefix_error"]))
    for key,values in by.items():prefix_by_result[(e["job_id"],*key)]=statistics.mean(values)
    fs=m.read_csv(base/"floor-probabilities.csv")
    if fs:floor_probability.append(dict(job_id=e["job_id"],configuration=e["arm"]+":"+e.get("floor","tail8")+":"+e.get("sd","sample"),
      dimension=e["dimension"],data_seed=e["data_seed"],
      median_of_time_medians=statistics.median(float(r["median"]) for r in fs),
      maximum=float(max(fs,key=lambda r:float(r["max"]))["max"])))
    hs=m.read_csv(base/"history.csv")
    histories.append(dict(job_id=e["job_id"],eligible_windows=len(hs),changed_threshold_decisions=sum(
      float(h["cv_population"])<.5<=float(h["cv_sample"]) for h in hs)))

heuristics=[]
for (validation,d,seed,config),rs in sorted(cells.items()):
    if rs[0]["method"] in ("bpf","fully_adapted","sis"):continue
    for baseline in ("bpf","fully_adapted","sis"):
        br=[r for r in records if (r["stage"]=="validation")==validation and r["data_seed"]==seed
          and r["method"]==baseline and r["status"]=="complete"]
        bs={r["replication"]:r for r in br}
        for situation in ("terminal","ordinary","large_innovation"):
            diffs=[];candidate_values=[];baseline_values=[]
            for r in rs:
                if r["status"]!="complete" or r["replication"] not in bs:continue
                b=bs[r["replication"]]
                if situation=="terminal":a_value=(float(r["ratio"])-1)**2;b_value=(float(b["ratio"])-1)**2
                else:
                    ak=(r["job_id"],r["method"],r["replication"],situation)
                    bk=(b["job_id"],b["method"],b["replication"],situation)
                    if ak not in prefix_by_result or bk not in prefix_by_result:continue
                    a_value=prefix_by_result[ak];b_value=prefix_by_result[bk]
                candidate_values.append(a_value);baseline_values.append(b_value);diffs.append(a_value-b_value)
            if not diffs:continue
            finite=all(math.isfinite(x) for x in diffs)
            ci=mean_ci(diffs) if finite else None
            heuristics.append(dict(validation=validation,dimension=int(d),data_seed=int(seed),
              configuration=config,baseline=baseline,situation=situation,paired_replicates=len(diffs),
              candidate_mse=statistics.mean(candidate_values),baseline_mse=statistics.mean(baseline_values),
              mean_difference=statistics.mean(diffs),difference_ci95=ci,
              observed_underperformance=statistics.mean(candidate_values)>statistics.mean(baseline_values),
              evidence="pointwise_bootstrap_diagnostic_no_multiplicity_corrected_ranking"))

controller_pairs=[]
for (validation,d,seed,config),rs in cells.items():
    if not config.endswith(":population"):continue
    control=cells.get((validation,d,seed,config.removesuffix("population")+"sample"),[])
    control={r["replication"]:r for r in control if r["status"]=="complete"}
    paired=[(r,control[r["replication"]]) for r in rs if r["status"]=="complete" and r["replication"] in control]
    controller_pairs.append(dict(dimension=int(d),data_seed=int(seed),configuration=config,
      paired=len(paired),changed_final_particles=sum(r["particles"]!=c["particles"] for r,c in paired),
      changed_iterations=sum(r["iterations"]!=c["iterations"] for r,c in paired),
      changed_terminal_value=sum(r["log_ratio"]!=c["log_ratio"] for r,c in paired)))

source_version=Path(__file__).parents[2].name.removeprefix("source-")
selection_path=OUT/f"validation-selection-{source_version}.json"
if not selection_path.exists():selection_path=OUT/"validation-selection.json"
selection=json.loads(selection_path.read_text()) if selection_path.exists() else {}
fit_selection=json.loads((OUT/"fit-selection.json").read_text()) if (OUT/"fit-selection.json").exists() else {}
validations=[r for r in records if r["stage"]=="validation"]
document=dict(status="completed_bounded_hypothesis_tests",attempt_status=dict(Counter(e["status"] for e in attempts)),
  source_hashes_verified=True,paired_observation_hashes=observations,
  superseded_records=[e["supersedes"] for e in attempts if "supersedes" in e and e["status"]=="finished"],
  worker_timeouts=[e["job_id"] for e in attempts if e["status"]=="timeout"],
  replica_status=dict(Counter(r["status"] for r in records)),
  fixture_status=dict(Counter(r["status"] for r in fixtures)),
  fit_selection=fit_selection,validation_selection=selection,cells=summaries,
  fixtures=fixtures,heldout=heldout,floor_probabilities=floor_probability,
  sd_history_diagnostics=histories,controller_pairs=controller_pairs,
  heuristic_comparisons=heuristics,heuristic_dominance_verdict={
    "calibration_observed_losses":sum(h["observed_underperformance"] for h in heuristics if not h["validation"]),
    "validation_observed_losses":sum(h["observed_underperformance"] for h in heuristics if h["validation"]),
    "ranking":"no overall or multiplicity-corrected ranking established",
    "roles":"terminal comparison veto; prefix losses veto general filtering promotion, not continuation of terminal reconstruction"},
  paper_replication=False,author_identity_established=False,default_readiness=False)
m.write_json(REPORT/"summary.json",sanitize(document))

fitlines=[];missing_fit_records=0
for arm in ("f1_strict",*m.FIT_ORDER):
    rs=[r for r in records if r["stage"]=="fit_probe" and r["method"]==arm]
    missing_fit_records+=16-len(rs)
    fitlines.append(f"| {arm} | {sum(r['status']=='complete' for r in rs)}/16 | {16-len(rs)} | "+
      "; ".join(f"{n} {msg}" for msg,n in Counter(r["message"] for r in rs if r["status"]!="complete").items())+" |")
floorlines=[]
for x in selection.get("calibration_screens",[]):
    if not any(c.get("complete",0) for c in x["cells"]):continue
    floorlines.append(f"| {x['arm']} / {x['floor']} / {x['sd']} | "+
      f"{sum(c['passed'] for c in x['cells'])}/6 | {'eligible' if x['passed'] else 'not eligible'} |")
observed_losses=[h for h in heuristics if h["observed_underperformance"]]
veto_text=(f"There are {len(observed_losses)} observed conditional losses against the constructed heuristic comparators. "
  "These block the corresponding promotion under the frozen conservative screen; small samples and rare tails prevent a general inferiority claim.")
selected_text=", ".join(x["arm"]+"/"+x["floor"]+"/"+x["sd"] for x in selection.get("candidates",[])) or "none"
timeout_count=sum(e["status"]=="timeout" for e in attempts)
provenance_link=os.path.relpath(OUT/"manifest.json",REPORT)
plan_link=os.path.relpath(OUT.parents[1]/"iapf-r-author-choice-hypotheses-2026-09-22.md",REPORT)
text=f"""# Tests of plausible iAPF numerical choices

The bounded author-choice campaign executed. The chosen fit for floor/controller
tests was `{fit_selection.get('selected','unresolved')}`. Untouched-validation
candidates: **{selected_text}**. {len(validations)} validation records were obtained.
These are independent CPU R reconstructions; the authors' choices are still unknown.

{veto_text}

## Fitting procedures

| Fit | Complete full-filter probes / scheduled | Without a final record | Recorded failures |
|---|---:|---:|---|
{chr(10).join(fitlines)}

{timeout_count} workers reached their predeclared time limit. Their {missing_fit_records} scheduled
replications without final records are resource-censored, not numerical failures.
Every affected fitting arm also has recorded failures, so filling those missing
records cannot restore its all-probes-complete qualification in this campaign.

The fixed-cloud study includes exact-Gaussian controls and independent predictive
and smoothing evaluation draws. F2 cannot test previous-iteration initialization
on a filter that fails before finishing its first backward sweep. Low absolute
Equation15 loss does not establish a useful guide. Weighted log fits explicitly
change the objective. Complete diagnostics are in summary.json and worker CSV/RDS.

## Positive floors and stopping convention

| Fit / floor / SD | Calibration cells passing / six | Validation disposition |
|---|---:|---|
{chr(10).join(floorlines)}

All floor variants rerun guide learning and filtering. Floor probabilities refer
to proposals actually used. Sample/population SD comparisons retain six estimates;
history replays explain threshold crossings and actual reruns measure the changed
trajectory. Paired effects and floor-probability summaries are in summary.json.
Every dataset remains fixed across methods and repetitions; source and observation
hashes were checked. No dataset was selected for resemblance to the paper.

## Decision

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Preserve tested fitting hypotheses and failures | Per-cell Kalman likelihood screens retained | Numeric/completion failures remain visible | Local optimizer termination versus useful fit | Inspect scale/shape diagnostics before a new fit hypothesis | Failure of iAPF theory |
| Validation selection follows frozen order | All six calibration cells required | Missing/failed cells cannot pass | Four-replica intervals miss rare tails | Validate only fully eligible combinations | Table matching or author identity |
| General filtering remains separately assessed | Original-prefix conditional MSE | Observed heuristic losses retained | Rare-weight tails and few datasets | Fresh paired evidence needed for any promotion | Terminal success implies accurate prefixes |

## Inference status

| Item | Status |
|---|---|
| Hard veto screen | Structured nonconvergence, boundary, nonconcavity, nonfinite and incomplete results retained |
| Statistically supported ranking | No overall or multiplicity-corrected ranking established |
| Descriptive differences | Resampling, N, runtime, four-replica calibration differences and controller effects |
| Default readiness | Not evaluated; CPU R reference only |
| Next evidence needed | Untouched multi-dataset replication of a numerically viable method; larger samples for rare tails |

Strongest alternative explanation is small-sample or observation-specific luck.
The weakest statistical evidence is calibration with four repetitions; published
tables are descriptive comparators, never selection losses. A fresh validation
failure would overturn calibration viability. Gaussian-limit checks concern a
zero-floor approximation and cannot prove divergence of the positive-floor filter.

Execution provenance, exact commands, versions, CPU settings, seeds, source hashes,
attempt wall times, failures and remaining budget are in the
[campaign manifest]({provenance_link}) and its adjacent attempts/source files.
See the [plan]({plan_link}). Deadline and budget were not renewed.
"""
(REPORT/"result.md").write_text(text)
print(json.dumps(dict(records=len(records),validation_records=len(validations),
  selected=selected_text,source_hashes_verified=True,summary=str(REPORT/"summary.json"))))
