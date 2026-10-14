"""Terminal integrity/decision assembly for the independent iAPF study."""
from pathlib import Path
import collections,csv,hashlib,json,math
ROOT=Path(__file__).resolve().parents[1]
P=ROOT/"docs/plans/artifacts/iapf-eq15-choice-calibration-20260925-01"
def rows(name,attempt="attempt01"):return list(csv.DictReader((P/attempt/name).open()))
def truth(v):return v=="TRUE"
fits=rows("fits.csv");floors=rows("floors.csv");selection=rows("selection.csv")
audit=rows("stationarity.csv","attempt02")
heur=rows("conditional-heuristics.csv","attempt02")
assert len(fits)==266 and len({(x["fixture"],x["arm"]) for x in fits})==266
assert len(floors)==324 and len({(x["d"],x["regime"],x["offset"],x["policy"]) for x in floors})==324
assert len(audit)==26 and all(truth(x["parity_pass"]) for x in audit)
assert len(selection)==14 and not any(truth(x["eligible"]) for x in selection)
assert all(x["quadrature_message"] in ("OK","analytically_divergent","not_evaluated_high_dimension") for x in floors)
assert len(rows("controls.csv"))==2
manifests=[json.loads((P/a/"manifest.json").read_text()) for a in ("attempt01","attempt02")]
verified=0
for a,m in zip(("attempt01","attempt02"),manifests):
    assert m["status"]=="COMPLETED" and m["exit_code"]==0
    for source,expected in m["source_sha256"].items():
        name=Path(source).name
        if name=="fixtures.rds":name="input-"+name
        actual=hashlib.sha256((P/a/"sources"/name).read_bytes()).hexdigest()
        assert actual==expected,(a,name,actual,expected)
        verified+=1
assert verified==13
assert len(heur)==243 and sum(truth(x["loses"]) for x in heur)==100
assert len({x["policy"] for x in heur if truth(x["loses"])})==9
assert collections.Counter(x["quadrature_message"] for x in floors)=={
    "OK":78,"analytically_divergent":3,"not_evaluated_high_dimension":243}
for a,m in zip(("attempt01","attempt02"),manifests):
    assert Path(m["result"]).is_file()
    terminal=(P/a/"terminal.txt").read_text()
    assert "scientific_replication=INCOMPLETE" in terminal
assert (P/"result.md").is_file()
cost=sum(x["wall_seconds"] for x in manifests)
assert cost<1200
decision=dict(status="COMPLETE_NO_QUALIFIED_FIT",plan=str(ROOT/"docs/plans/iapf-eq15-choice-calibration-20260925.md"),
    controls="PASS",fit_calibration="COMPLETE",floor_study="COMPLETE",
    terminal_review="PASS",review_kind="skeptical_self_review_with_executable_checks",
    confirmation="NOT_ADMITTED_NO_CALIBRATION_NOMINEE",
    full_filter_pilot="NOT_ADMITTED_FIT_AND_FLOOR_GATES",
    solver_fits=266,terminal_replays=26,total_solver_fits=292,floor_cells=324,
    source_snapshots_verified=verified,nominated_solvers=[],
    cpu_wall_seconds=cost,GPU_seconds=0,
    scientific_replication="INCOMPLETE",runtime_promotion="NOT_ELIGIBLE_INDEPENDENT_R_REFERENCE",
    heuristic_dominance=dict(verdict="VETO_ALL_POLICIES_HAVE_CONDITIONAL_LOSSES",
        comparisons=len(heur),loss_comparisons=sum(truth(x["loses"]) for x in heur),
        per_policy=dict(collections.Counter(x["policy"] for x in heur if truth(x["loses"])))),
    inference_status=dict(hard_veto="Supported numerical, shape, boundary and heuristic failures",
        statistical_ranking="Not established; fixed fixtures and deterministic local integrals",
        default_readiness="Not established",descriptive_only="Runtime, finite fixture coverage, high-dimensional behavior"),
    next_action="Calibrate scale-aware termination and test a declared useful local branch on the original objective before any adaptive pilot; no objective substitution.",
    limits=["No author optimizer recovered","No untouched confirmation admitted",
            "No high-dimensional second-moment integral evaluated",
            "Saved inputs lack independent held-out target callables",
            "Positive floor does not repair an inaccurate fit"])
(P/"decision.json").write_text(json.dumps(decision,indent=2)+"\n")
print(json.dumps({k:decision[k] for k in ("status","total_solver_fits","floor_cells","cpu_wall_seconds","source_snapshots_verified")}))
