#!/usr/bin/env python3
"""Read-only numerical reporting; never changes campaign decisions or settings."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time


def read(path):
    return json.loads(Path(path).read_text())


def write(path, text):
    temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def summarize(root):
    campaign=read(root/"campaign.json")
    now=datetime.now(timezone.utc)
    attempts=campaign["attempts"]
    spent=(campaign.get("prior_consumed_seconds",0.)+campaign.get("migration_additional_seconds",0.)
           +sum(a.get("wall_seconds",0.) for a in attempts))
    for a in attempts:
        if a["status"]=="RUNNING":
            spent+=max(0.,(now-datetime.fromisoformat(a["started_utc"].replace("Z","+00:00"))).total_seconds())
    completed=[]
    for attempt in attempts:
        if attempt.get("result"):
            result=read(attempt["result"])
            completed.append((attempt,result))
    scopes=[]
    for attempt,result in completed:
        if attempt["kind"]!="scope": continue
        scopes.append(dict(scope=attempt["scope"],particles=attempt["particles"],
             outcome=result.get("outcome",result.get("status")),datasets=result.get("datasets",0),
             summaries=result.get("summaries",[]),result=attempt["result"],
             source_revision=attempt.get("source_revision",campaign["configuration"]["harness_revision"]),
             source_campaign=attempt.get("source_campaign",str(root)),
             inherited=attempt.get("evidence_status")=="retained_previous_source_result_not_rerun"))
    grid_jobs=len(campaign["configuration"]["scopes"])*len(campaign["configuration"]["particles"])
    engineering=next((r for a,r in completed if a["kind"]=="engineering"),None)
    precision=next((r for a,r in completed if a["kind"]=="precision"),None)
    pilot=next((r for a,r in completed if a["kind"]=="pilot"),None)
    partial=[]
    saved_datasets={}
    for folder in sorted((root/"scopes").glob("*")):
        files=list(folder.glob("dataset-*.json"))
        if not files: continue
        rows=[read(f) for f in files]
        saved_datasets[folder.name]=rows
        looks=sorted(folder.glob("look-*.json"),key=lambda p:int(p.stem.split("-")[-1]))
        latest_look=read(looks[-1]) if looks else None
        partial.append(dict(scope=folder.name,complete_datasets=sum(bool(d["complete"]) for d in rows),
                            paired_design_rows=sum(len(p["rows"]) for d in rows for p in d["points"]),
                            latest_planned_look=int(looks[-1].stem.split("-")[-1]) if looks else None,
                            look_artifact=str(looks[-1]) if looks else None,
                            summaries=latest_look["summaries"] if latest_look else []))
    # Correct the generic producer label using saved validity, never discard a
    # failed row or rewrite the original scientific result.
    for scope in scopes:
        records=saved_datasets.get(f"{scope['scope']}-n{scope['particles']}",[])
        if not records and scope["inherited"]:
            records=[read(p) for p in sorted((Path(scope["source_campaign"])/"scopes"/
                f"{scope['scope']}-n{scope['particles']}").glob("dataset-*.json"))]
        for cell in scope["summaries"]:
            if cell.get("valid",True):
                continue
            counts={"ancestor":0,"marginal_mixture":0,"reference":0,"incomplete":0}
            for dataset in records:
                counts["reference"]+=int(not dataset["reference"]["valid"])
                rows=dataset["points"][cell["point"]]["rows"]
                counts["incomplete"]+=int(len(rows)!=32)
                for row in rows:
                    for method in ("ancestor","marginal_mixture"):
                        output=row[method]
                        numbers=[output["value"],*output["score"]]
                        healthy=output["valid"] and all(isinstance(x,(int,float))
                            and math.isfinite(x) for x in numbers)
                        counts[method]+=int(not healthy)
            cell["producer_verdict"]=cell["verdict"]
            cell["validity_attribution"]=counts
            if counts["reference"]:
                cell["verdict"]="INVALID_REFERENCE"
            elif counts["ancestor"] and counts["marginal_mixture"]:
                cell["verdict"]="INVALID_BOTH_METHODS"
            elif counts["ancestor"]:
                cell["verdict"]="INVALID_BASELINE"
            elif counts["marginal_mixture"]:
                cell["verdict"]="INVALID_CANDIDATE"
            else:
                cell["verdict"]="INCOMPLETE_PAIRED_COMPARISON"
    completed_by_key={(s["scope"],s["particles"]):s for s in scopes}
    progress_by_key={p["scope"]:p for p in partial}
    coverage=[]
    for n in campaign["configuration"]["particles"]:
        for scope in campaign["configuration"]["scopes"]:
            key=f"scope-{scope}-n{n}"
            done=completed_by_key.get((scope,n))
            progress=progress_by_key.get(f"{scope}-n{n}")
            relevant=[a for a in attempts if a["key"]==key]
            latest=relevant[-1] if relevant else {}
            if done:
                if not done["summaries"]:
                    label="controls rejected"
                else:
                    label=("retained" if done["inherited"] else "complete")+f" ({done['datasets']})"
            elif progress:
                label=("running" if latest.get("status")=="RUNNING" else "partial")+f" ({progress['complete_datasets']})"
            else:
                label={"RUNNING":"calibrating","FAILED":"failed attempt"}.get(latest.get("status"),"pending")
            coverage.append(dict(scope=scope,particles=n,status=label,
                                 source_campaign=done["source_campaign"] if done else str(root)))
    interim=[p for p in partial if p["summaries"] and
             (p["scope"].rsplit("-n",1)[0],int(p["scope"].rsplit("-n",1)[1])) not in completed_by_key]
    data=dict(timestamp=now.isoformat(),campaign_status=campaign["status"],gpu_work_seconds=spent,
              budget_seconds=campaign["configuration"]["budget_seconds"],grid_jobs=grid_jobs,
              completed_scope_jobs=len(scopes),scopes=scopes,partial=partial,coverage=coverage,interim_looks=interim,
              retained_previous_source_jobs=sum(s["inherited"] for s in scopes),
              current_source_scope_jobs=sum(not s["inherited"] for s in scopes),
              engineering=bool(engineering and engineering.get("passed")),
              precision=bool(precision and precision.get("passed")),
              dimension_extension="pending model definition for nonlinear dimensions",
              hmc_test_complete=False,default_ready=False)
    lines=["# Expanded marginal-weight verification — running results",
           "",
           "Purpose: test the likelihood/analytical-score component needed for surrogate HMC. "
           "This campaign does not yet test posterior sampling.",
           "",
           f"Updated: {now.isoformat()}. Campaign: {campaign['status']}.",
           f"Measured worker time so far: {spent/3600:.2f} hours of the "
           f"{campaign['configuration']['budget_seconds']/3600:.1f}-hour ceiling.",
           f"Completed scope jobs: {len(scopes)}/{grid_jobs}. A rejected control pair counts as "
           "a completed attempt, never as a successful accuracy comparison.",
           f"Retained prior-source comparisons: {data['retained_previous_source_jobs']}; "
           f"current-source completed comparisons: {data['current_source_scope_jobs']}. "
           "Retained results describe their original implementation and are not new-source replications.",
           "",
           "| Decision | Primary criterion | Veto diagnostics | Main uncertainty | Next justified action | Not concluded |",
           "|---|---|---|---|---|---|",
           f"| Engineering | {'Passed' if data['engineering'] else 'Pending/failed'} | Batch parity, finite-program score, T=1 reference checks | Larger scopes | Continue the same master | Model accuracy or HMC convergence |",
           f"| Precision | {'Passed' if data['precision'] else 'Pending/failed'} | Nonfinite evaluations | Different datasets/scopes | Keep FP32 and inspect larger scopes | Agreement with true model likelihood |",
           f"| Fresh-data replication | {len(scopes)}/{grid_jobs} scope jobs complete | See per-scope records | Dataset sampling and reference error | Complete planned 4→8→16 looks within budget | Universal improvement |",
           "| Nonlinear dimension extension | Pending | Model equations undefined | KSC is scalar; original range-bearing is 4D | Resolve requested new model definitions | Requested nonlinear dimension grid covered |",
           "",
           "| Inference status | Current interpretation |",
           "|---|---|",
           "| Hard veto screen | Engineering and precision verdicts above; accuracy scopes below |",
           "| Statistically supported ranking | Only the predeclared per-cell dataset intervals can support a ranking |",
           "| Descriptive-only differences | Timing, precision deltas and heuristic point estimates; FP64 timing from attempt 01 is superseded |",
           "| Default-readiness | Not established; ancestor remains the default |",
           "| Next evidence needed | Fresh datasets, reference refinement, particle-count and matched-FP32 cost comparisons, then actual surrogate-HMC tests |",
           "",
           "Independent accuracy authorities use FP64. In the active continuation simple comparators and LEDH use FP32/TF32/GPU/XLA. "
           "The equal-runtime bootstrap comparison measures likelihood accuracy; the bootstrap produces no score, "
           "while LEDH timing includes its analytical score. It is not an equivalent HMC-force workload. Source04 synchronizes value, score and validity and excludes report serialization from timed inference. Retained source03 timings include unequal report-serialization overhead. Actual cost mismatches remain explicit; no overall computational-value ranking is established.",
           ""]
    supervisor_path=root/"covariance-probe-supervisor.json"
    if supervisor_path.exists():
        supervisor=read(supervisor_path)
        lines += [f"Automatic covariance/timing diagnostics: {supervisor['status']}. "
                  "The supervisor drains the active short worker and resumes master dispatch afterward. "
                  "See covariance-probe-supervisor.json for the exact state.", ""]
    repair_path=root/"covariance-repair-supervisor.json"
    if repair_path.exists():
        repair=read(repair_path)
        lines += [f"Bounded covariance-product repair evaluation: {repair['status']}. "
                  "The unchanged master resumes automatically after this diagnostic.", ""]
    invalid=[(s,c) for s in scopes for c in s["summaries"] if not c.get("valid",True)]
    if invalid:
        lines += ["Invalid comparison attribution (original evidence is preserved):", ""]
        for scope,cell in invalid:
            lines.append(f"- {scope['scope']} N={scope['particles']} point {cell['point']}: "
                f"{cell['verdict']}; counts {cell['validity_attribution']}.")
        lines.append("")
    if campaign.get("previous_campaign"):
        lines += ["The selective covariance repair passed all 68 same-process public-consumer checks: "
                  "healthy values/scores were unchanged and 12 formerly invalid high-dimensional rows recovered. "
                  "An unchanged-source comparison against a saved earlier process nevertheless exceeded the "
                  "no-harm margin on three M13 rows. Cross-process FP32/TF32 reproducibility remains unresolved; "
                  "neither repeat checks within one graph nor this component campaign establish HMC readiness.",
                  ""]
    if engineering:
        checks=[c for row in engineering["rows"] for c in row["checks"]]
        lines += ["The GPU engineering fixtures used distinct parameter rows for all three models and both policies. "
           f"Maximum batch/separate-call discrepancy: {max(c['parity_relative'] for c in checks):.3g}; "
           f"maximum relative score/finite-difference discrepancy: {max(d['relative_error'] for c in checks for d in c['fd']):.6g}. "
           "The T=1 range-bearing quadrature/PF comparison passed its recorded uncertainty check.",
           ""]
    if precision:
        lines += ["| N=4096 precision arm | Parameter point | Absolute log-value delta | Scaled score delta | Repeat spread |",
                  "|---|---:|---:|---:|---:|"]
        for r in precision["rows"]:
            lines.append(f"| {r['policy']} | {r['point']} | {r['precision_delta']['value']:.6g} | "
                         f"{r['precision_delta']['score']:.6g} | {r['repeat_value_spread']:.6g} |")
        lines.append("")
    if pilot:
        lines += ["Pilot result: "+str({k:v for k,v in pilot.items() if k not in ("calibration","controls")}),""]
        lower=pilot.get("four_dataset_minimum_candidate_seconds",0.)
        if lower>campaign["configuration"]["worker_scope_cap_seconds"]:
            lines += ["The four-dataset pilot time estimate exceeds one worker's time cap. "
                      "Long scopes may therefore remain partial at the cap; saved rows are retained. "
                      "Numerical pilot validity is not a completed accuracy comparison.",""]
    if partial:
        lines += ["| Scope | Complete fresh datasets | Completed paired parameter/design rows |",
                  "|---|---:|---:|"]
        for row in partial:
            lines.append(f"| {row['scope']} | {row['complete_datasets']} | {row['paired_design_rows']} |")
        lines.append("")
    if interim:
        lines += ["Latest predeclared interim looks are shown below. These are unfinished comparisons; "
                  "the master extends inconclusive cells to the next planned dataset count.",
                  "",
                  "| Current scope | Datasets in look | Point | Mean absolute value error, ancestor → marginal | Mean scaled score error, ancestor → marginal | Value interval | Score interval | Decision |",
                  "|---|---:|---:|---|---|---|---|---|"]
        for progress in interim:
            for cell in progress["summaries"]:
                metrics=cell.get("metrics",{})
                def metric_text(name):
                    m=metrics.get(name)
                    if not m:return "not assessed","not assessed"
                    interval=m["reference_adjusted_interval"]
                    return (f"{m['baseline_mean']:.5g} → {m['candidate_mean']:.5g}",
                            f"[{interval[0]:.4g}, {interval[1]:.4g}]")
                value,vi=metric_text("value");score,si=metric_text("score")
                decision=cell["verdict"]
                if metrics:
                    decision="/".join(metrics[k]["verdict"] for k in ("value","score"))
                lines.append(f"| {progress['scope']} | {progress['latest_planned_look']} | {cell['point']} | "
                             f"{value} | {score} | {vi} | {si} | {decision} |")
        lines += ["", "Intervals concern marginal minus ancestor error, with the predeclared multiplicity "
                  "and reference allowances. Lower observed means alone do not establish improvement.", ""]
    lines += ["Grid coverage; numbers in parentheses are completed datasets. A retained entry identifies "
              "earlier-source evidence, and a complete entry identifies a finished current-source comparison, "
              "regardless of whether its accuracy criterion passed.", "",
              "| Model | N | T=10 | T=20 | T=50 | T=120 | T=240 |",
              "|---|---:|---|---|---|---|---|"]
    families=list(dict.fromkeys(s.rsplit("-t",1)[0] for s in campaign["configuration"]["scopes"]))
    coverage_map={(c["scope"],c["particles"]):c["status"] for c in coverage}
    for n in campaign["configuration"]["particles"]:
        for family in families:
            columns=[coverage_map.get((f"{family}-t{t}",n),"not defined") for t in (10,20,50,120,240)]
            lines.append(f"| {family} | {n} | "+" | ".join(columns)+" |")
    lines.append("")
    if scopes:
        lines += ["| Scope | Source | N | Datasets | Parameter point | Value verdict | Score verdict | Heuristic screen |",
                  "|---|---|---:|---:|---:|---|---|---|"]
        for scope in scopes:
            source_label=("retained "+Path(scope["source_campaign"]).name.rsplit("-",1)[-1]
                          if scope["inherited"] else "current "+root.name.rsplit("-",1)[-1])
            if not scope["summaries"]:
                lines.append(f"| {scope['scope']} | {source_label} | {scope['particles']} | 0 | — | {scope['outcome']} | Not assessed | Not assessed |")
            for r in scope["summaries"]:
                metrics=r.get("metrics",{})
                lines.append(f"| {scope['scope']} | {source_label} | {scope['particles']} | {scope['datasets']} | {r['point']} | "
                    f"{metrics.get('value',{}).get('verdict',r['verdict'])} | "
                    f"{metrics.get('score',{}).get('verdict',r['verdict'])} | "
                    f"{r.get('heuristic_dominance_verdict','Not assessed')} |")
        lines.append("")
    valid_cells=[(scope,cell) for scope in scopes for cell in scope["summaries"] if cell.get("metrics")]
    if valid_cells:
        lines += ["Paired accuracy results at the terminal predeclared look for each completed scope. "
                  "Entries show ancestor → marginal mean absolute error. Score error is 0.1 times "
                  "physical-score RMS error. Intervals concern marginal minus ancestor and include the "
                  "declared multiplicity and reference allowances.", "",
                  "| Scope / N | Source | Point | Value error | Value difference interval | Scaled score error | Score difference interval | Supported lower error |",
                  "|---|---|---:|---|---|---|---|---|"]
        for scope,cell in valid_cells:
            source=("retained "+Path(scope["source_campaign"]).name.rsplit("-",1)[-1]
                    if scope["inherited"] else "current "+root.name.rsplit("-",1)[-1])
            v,g=(cell["metrics"][name] for name in ("value","score"))
            vi,gi=v["reference_adjusted_interval"],g["reference_adjusted_interval"]
            supported=", ".join(name for name in ("value","score")
                                if cell["metrics"][name]["supported_lower_error"]) or "none"
            lines.append(f"| {scope['scope']} / {scope['particles']} | {source} | {cell['point']} | "
                f"{v['baseline_mean']:.5g} → {v['candidate_mean']:.5g} | [{vi[0]:.4g}, {vi[1]:.4g}] | "
                f"{g['baseline_mean']:.5g} → {g['candidate_mean']:.5g} | [{gi[0]:.4g}, {gi[1]:.4g}] | {supported} |")
        lines += ["", "Simple-filter likelihood comparisons at the same particle count. These are "
                  "descriptive mean absolute errors, not statistically established rankings. The exact "
                  "Kalman comparator naturally supplies the zero-error LGSSM reference. A component "
                  "non-inferiority result relative to ancestor weights does not establish that particle "
                  "filtering is preferable to Kalman filtering for a linear Gaussian model.", "",
                  "| Scope / N | Source | Point | Marginal error | Bootstrap error | Prior-trajectory error | Gaussian-filter error |",
                  "|---|---|---:|---:|---:|---:|---:|---:|"]
        for scope,cell in valid_cells:
            source=("retained "+Path(scope["source_campaign"]).name.rsplit("-",1)[-1]
                    if scope["inherited"] else "current "+root.name.rsplit("-",1)[-1])
            h=cell.get("heuristic",{})
            def cheap_text(name):
                entry=h.get(name)
                return f"{entry['comparator_mean_error']:.5g}" if entry else "missing"
            candidate=cell["metrics"]["value"]["candidate_mean"]
            lines.append(f"| {scope['scope']} / {scope['particles']} | {source} | {cell['point']} | "
                         f"{candidate:.5g} | {cheap_text('bootstrap')} | {cheap_text('prior')} | {cheap_text('gaussian')} |")
        lines += ["", "Equal-runtime coverage is shown separately. Partial matches describe only the "
                  "matched subset, and cannot establish an overall runtime ranking. Old-source timings "
                  "include unequal serialization overhead; current-source timings exclude it.", "",
                  "| Scope / N | Source | Point | Matched rows / expected | Marginal error on matched rows | Bootstrap error on matched rows | Complete |",
                  "|---|---|---:|---:|---:|---:|---|"]
        for scope,cell in valid_cells:
            source=("retained "+Path(scope["source_campaign"]).name.rsplit("-",1)[-1]
                    if scope["inherited"] else "current "+root.name.rsplit("-",1)[-1])
            h=cell.get("heuristic",{}).get("bootstrap_equal_runtime")
            if h:
                lines.append(f"| {scope['scope']} / {scope['particles']} | {source} | {cell['point']} | "
                             f"{h['usable_rows']} / {h['expected_rows']} | {h['candidate_mean_error']:.5g} | "
                             f"{h['comparator_mean_error']:.5g} | {h['complete']} |")
            else:
                lines.append(f"| {scope['scope']} / {scope['particles']} | {source} | {cell['point']} | "
                             f"0 / {scope['datasets']*32} | not assessed | not assessed | False |")
        lines.append("")
    lines += ["Post-run red-team note: exact finite-program derivatives do not establish accuracy relative "
              "to the statistical model. Four datasets can leave large uncertainty; particle-reference refinement "
              "and MCSE do not bound all remaining bias. The diagonal/pairwise correction controls remain "
              "scope-specific hypotheses. Low process-noise range-bearing performance must be reported separately "
              "from the nominal parameter point. The current parameter regimes are likelihood-evaluation points "
              "on data generated at the recorded truth; they are not a factorial grid of generating parameters.",
              "",
              "The next phase is part of the same authorized master. Consult next-phase.json and campaign.json "
              "for current execution and unfinished work. An active worker is not a completed result.",
              ""]
    write(root/"analysis.json",json.dumps(data,indent=2,allow_nan=False))
    write(root/"results.md","\n".join(lines))
    return campaign["status"]


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--watch",action="store_true")
    a=p.parse_args()
    while True:
        state=summarize(a.output)
        if not a.watch or state!="RUNNING":
            print(state, a.output/"results.md")
            return
        time.sleep(60)

if __name__=="__main__": main()
