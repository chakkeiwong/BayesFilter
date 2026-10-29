"""Diagnostic-only inspection of saved N1008 particle-design variability."""
import hashlib,json,math,statistics
from pathlib import Path
ROOT=Path("/home/chakwong/BayesFilter-ledh-graceful-failure-execution-20260917")
OUT=ROOT/"docs/plans/artifacts/ledh-range-bearing-covariance-20261009-07"
report=json.loads((OUT/"report.json").read_text())
rows=[json.loads(line) for line in (OUT/"compare.jsonl").read_text().splitlines()]
cells=[]
for cell in report["cells"]:
    ref=cell["reference"]
    arms={}
    for arm in ("new","old","prior"):
        group=[r for r in rows if (r["data_seed"],r["point"],r["arm"])==(cell["data_seed"],cell["point"],arm)]
        assert len(group)==8 and len({r["seed"] for r in group})==8 and all(r["valid"] for r in group)
        assert all(math.isfinite(v) for r in group for v in [r["value"],*r["score"]])
        centre=[statistics.mean(r["score"][j] for r in group) for j in range(2)]
        sd=[statistics.stdev(r["score"][j] for r in group) for j in range(2)]
        errors=[math.dist(r["score"],ref["score"]) for r in group]
        spread=math.hypot(*sd)
        centre_error=math.dist(centre,ref["score"])
        rmse=math.sqrt(statistics.mean(e*e for e in errors))
        assert math.isclose(rmse**2,centre_error**2+7/8*spread**2,rel_tol=1e-11,abs_tol=1e-11)
        stats=dict(n=8,score_mean=centre,score_coordinate_sd=sd,score_rms_spread=spread,
            score_centroid_reference_error=centre_error,score_rmse=rmse,
            mean_score_error=statistics.mean(errors),sd_score_error=statistics.stdev(errors),
            max_score_error=max(errors),error_fraction_of_reference_score_norm=statistics.mean(errors)/math.hypot(*ref["score"]),
            mean_loglik=statistics.mean(r["value"] for r in group),
            loglik_mean_minus_reference=statistics.mean(r["value"] for r in group)-ref["value"],
            loglik_sd=statistics.stdev(r["value"] for r in group))
        fields=group[0].get("trace_fields",[])
        if "ess" in fields:
            index=fields.index("ess")
            ess=[step[index] for r in group for step in r["trace"] if step[index] is not None]
            if ess:stats["minimum_particle_ess"]=min(ess)
        arms[arm]=stats
    cells.append(dict(data_seed=cell["data_seed"],point=cell["point"],reference_score=ref["score"],
        reference_score_mcse=ref["mcse"][1:],arms=arms))
result=dict(role="post-run explanatory diagnostic only; not HMC, a tuning decision, or a particle-count sufficiency test",
    source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (OUT/"compare.jsonl",OUT/"report.json")},
    interpretation="SD is across eight fixed particle designs. Centroid-reference discrepancy is not an unbiased bias estimate; both eight-design uncertainty and finite-particle reference bias remain. RMSE squared equals centroid discrepancy squared plus (n-1)/n times sum of coordinate sample variances.",
    cells=cells)
path=OUT/"variance-diagnostics.json"
content=json.dumps(result,indent=2,allow_nan=False)+"\n"
if path.exists():
    assert path.read_text()==content,"Refuse to overwrite differing evidence"
else:
    path.write_text(content)
print(json.dumps(dict(artifact=str(path))))
for c in cells:
    print(json.dumps(dict(data=c["data_seed"],point=c["point"],arms={arm:{k:round(v,4) if isinstance(v,(int,float)) else v for k,v in s.items() if k in ("mean_score_error","sd_score_error","score_rms_spread","score_centroid_reference_error","max_score_error","loglik_mean_minus_reference","loglik_sd","minimum_particle_ess")} for arm,s in c["arms"].items()})))
