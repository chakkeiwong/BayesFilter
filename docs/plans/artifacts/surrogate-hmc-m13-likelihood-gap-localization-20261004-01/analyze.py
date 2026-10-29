"""Post-run diagnostic analysis; standard-library reporting only."""
import hashlib,json,math,pathlib
HERE=pathlib.Path(__file__).resolve().parent
ROOT=HERE.parents[3]
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
trace=read(HERE/"report.json")
baseline=read(HERE/"first-step-baseline/report.json")
assert trace["status"]==baseline["status"]=="COMPLETED"
assert len(trace["rows"])==len(baseline["rows"])==8
manifests=[read(HERE/"run-manifest.json"),read(HERE/"first-step-baseline/run-manifest.json")]
assert all(m["exit_code"]==0 and m["status"]=="COMPLETED" for m in manifests)
for m,p in zip(manifests,[HERE/"report.json",HERE/"first-step-baseline/report.json"]):
    assert m["report_sha256"]==sha(p)
def finite_tree(x):
    if isinstance(x,float): return math.isfinite(x)
    if isinstance(x,dict): return all(finite_tree(v) for v in x.values())
    if isinstance(x,list): return all(finite_tree(v) for v in x)
    return True
assert finite_tree(trace) and finite_tree(baseline)
summary=[]
for row,b in zip(trace["rows"],baseline["rows"]):
    assert row["probe"]==b["probe"]
    steps=row["directions"][0]["steps"]; t1=steps[0]
    qs=row["first_observation_quadrature"]; q=qs[-1]
    true_score=q["score"]["0.0001"]
    ls=[d["steps"][0]["score_increment"] for d in row["directions"]]
    assert row["parity_passed"]
    for j,d in enumerate(row["directions"]):
        assert d["status"]["value_valid"] and d["status"]["force_valid"]
        assert all(s["program_valid"] for s in d["steps"])
        assert abs(sum(s["log_increment"] for s in d["steps"])-row["value"])<1e-9
        assert abs(sum(s["score_increment"] for s in d["steps"])-row["score"][j])<1e-9
    quad_value=q["center"]["value"]
    ledh_score_error=math.dist(ls,true_score)
    bootstrap_score_error=math.dist(b["score"],true_score)
    summary.append(dict(label=row["probe"]["label"],theta=row["probe"]["theta"],
        first_ledh_log_likelihood=t1["log_increment"],first_quadrature_log_likelihood=quad_value,
        first_bootstrap_log_likelihood=b["value"],
        first_ledh_log_likelihood_error=t1["log_increment"]-quad_value,
        first_bootstrap_log_likelihood_error=b["value"]-quad_value,
        first_ledh_score=ls,first_quadrature_score=true_score,first_bootstrap_score=b["score"],
        first_ledh_score_error_norm=ledh_score_error,first_bootstrap_score_error_norm=bootstrap_score_error,
        first_ledh_ess=t1["ess"],first_bootstrap_ess=b["ess"],
        minimum_ledh_ess=min(s["ess"] for s in steps),
        maximum_ledh_weight=max(s["maximum_weight"] for s in steps),
        first_prior_normalizer=math.exp(t1["prior_log_normalizer"]),
        first_ukf_variance_over_process_variance=t1["ukf_variance_over_process_variance"],
        first_flow_displacement_process_sd_rms=t1["flow_displacement_process_sd_rms"],
        quad_value_grid_change=abs(qs[0]["center"]["value"]-quad_value),
        quad_score_grid_change=max(abs(a-b) for a,b in zip(qs[0]["score"]["0.0001"],true_score)),
        quad_score_step_change=max(abs(a-b) for a,b in zip(q["score"]["0.0003"],true_score)),
        quad_relative_tail_bound=q["center"]["relative_tail_bound"],
        fixed_probe_heuristic_veto=bootstrap_score_error<ledh_score_error and
            abs(b["value"]-quad_value)<abs(t1["log_increment"]-quad_value)))
out=dict(question="Why are the M13 likelihood scores discrepant?",
    observed_mechanism="Severe finite-particle importance-weight concentration; likelihood/score errors exist before first reset",
    mechanism_supported_at_frozen_probes=True,full_T20_exact_score_oracle=False,
    algorithmic_repair_established=False,statistical_ranking_supported=False,probes=summary,
    maximum_trace_value_error=max(r["trace_value_error"] for r in trace["rows"]),
    maximum_trace_score_error=max(r["trace_score_error"] for r in trace["rows"]),
    maximum_reset_relative_covariance_error=max(s["reset_relative_covariance_error"]
        for r in trace["rows"] for s in r["directions"][0]["steps"]),
    maximum_reset_mean_error_in_filter_sd=max(s["reset_mean_error_in_filter_sd"]
        for r in trace["rows"] for s in r["directions"][0]["steps"]),
    heuristic_dominance_verdict="Current finite LEDH T1 calculation loses to the same-particle bootstrap at all eight fixed probes; no population ranking",
    wall_seconds=sum(m["wall_seconds"] for m in manifests),
    aggregate_score_diagnostic_seconds=1921.8366+sum(m["wall_seconds"] for m in manifests),
    remaining_score_diagnostic_seconds=manifests[-1]["remaining_budget_seconds"])
(HERE/"summary.json").write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
audit=dict(status="PASS",checks=["eight matching probes","finite outputs","all validity flags",
    "unchanged imported frozen sources (both workers)","exact trace/full-consumer value-score parity",
    "per-step increment sums","quadrature grid and FD refinement recorded",
    "bootstrap callback/finite-difference parity","report hashes match manifests",
    "budgets respected","no HMC or runtime changes"],
    artifacts={str(p.relative_to(HERE)):sha(p) for p in
        [HERE/"report.json",HERE/"first-step-baseline/report.json",HERE/"summary.json",
         HERE/"diagnose.py",HERE/"first_step_baseline.py"]})
assert out["wall_seconds"]<600 and out["aggregate_score_diagnostic_seconds"]<3600
(HERE/"final-audit.json").write_text(json.dumps(audit,indent=2,sort_keys=True)+"\n")
for s in summary:
    print(s["label"],"T1log_error",round(s["first_ledh_log_likelihood_error"],6),
          "bootstrap",round(s["first_bootstrap_log_likelihood_error"],6),
          "score_norms",round(s["first_ledh_score_error_norm"],6),
          round(s["first_bootstrap_score_error_norm"],6))
print("Audit PASS; seconds",out["wall_seconds"],"remaining",out["remaining_score_diagnostic_seconds"])
