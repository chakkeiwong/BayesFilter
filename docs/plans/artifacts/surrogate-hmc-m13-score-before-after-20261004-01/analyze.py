"""Post-run diagnostic arithmetic only; no algorithm decisions or tuning."""
import hashlib,json,math,pathlib,statistics,struct
R=pathlib.Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text())
def norm(x):return math.sqrt(sum(v*v for v in x))
def sub(a,b):return [x-y for x,y in zip(a,b)]
def f32(x):return struct.unpack("f",struct.pack("f",x))[0]
common=read(R/"inputs.json")
L=common["geometry"]["factor"]
center=common["geometry"]["center"]
C=common["geometry"]["covariance"]
prior_var=math.log(2.)**2
def white(g):return [L[0][0]*g[0]+L[1][0]*g[1],L[1][1]*g[1]]
def white_radius(e):return [abs(L[0][0])*e[0]+abs(L[1][0])*e[1],abs(L[1][1])*e[1]]
def metrics(a,b):
    aa,bb=white(a),white(b)
    return dict(raw_l2_error=norm(sub(a,b)),raw_reference_norm=norm(b),
        whitened_l2_error=norm(sub(aa,bb)),whitened_reference_norm=norm(bb),
        whitened_scaled_error=norm(sub(aa,bb))/max(1.,norm(bb)),
        whitened_cosine=sum(x*y for x,y in zip(aa,bb))/(norm(aa)*norm(bb)) if norm(aa)*norm(bb)>1e-20 else None)
reports={}
for arm in ("before","after"):
    for dtype in ("float32","float64"):
        p=(R if (arm,dtype)==("before","float32") else R/"attempt-02")/(arm+"-"+dtype+".json")
        d=read(p)
        assert d["status"]=="COMPLETED",(p,d["status"])
        reports[arm,dtype]=d
        assert d["input_manifest_sha256"]==hashlib.sha256((R/"inputs.json").read_bytes()).hexdigest()
def flat(arm,dtype,key):
    return [v for row in reports[arm,dtype]["base"] for v in row[key]]
def fd(arm):
    curves={}
    validity=[]
    for item in reports[arm,"float64"]["differences"]:
        h,dim=item["step"],item["parameter"]
        values={s:[v for b in item["values"][s] for v in b["value"]] for s in ("-1","1")}
        validity.extend([v for s in ("-1","1") for b in item["values"][s] for v in b["valid"]])
        curves[h,dim]=[(p-m)/(2*h) for p,m in zip(values["1"],values["-1"])]
    return curves,all(validity)
rows=[]
for i,probe in enumerate(common["probes"]):
    theta=probe["theta"]
    row=dict(probe=probe,arms={},heuristics={})
    for arm in ("before","after"):
        own,fd_valid=fd(arm)
        finest=[own[1e-4,j][i] for j in (0,1)]
        coarser=[own[3e-4,j][i] for j in (0,1)]
        own_unc=[abs(x-y) for x,y in zip(finest,coarser)]
        own_stable=all(e<=1e-4*(1+abs(x)) for e,x in zip(own_unc,finest))
        endpoint=reports[arm,"float64"]["endpoint"][i]
        ref=endpoint["steps"][-1]["coordinates"]
        refprev=endpoint["steps"][-2]["coordinates"]
        exact=[v["derivative"] for v in ref]
        end_unc=[abs(v["derivative"]-w["derivative"]) for v,w in zip(ref,refprev)]
        end_stable=all(e<=1e-4*(1+abs(x)) for e,x in zip(end_unc,exact))
        endpoint_valid=endpoint["valid"] and all(v["minus_valid"] and v["plus_valid"]
            for h in endpoint["steps"] for v in h["coordinates"])
        score64=flat(arm,"float64","score")[i]
        score32=flat(arm,"float32","score")[i]
        posterior_ref=[s-x/prior_var for s,x in zip(exact,theta)]
        # Match the observed XLA constant-division lowering (multiply rounded reciprocal).
        # Verified against the actual runtime wrapper in both versions. A Python
        # correctly-rounded division differed by one ULP in one component.
        runtime_score=[f32(s-f32(x*f32(1./f32(prior_var)))) for s,x in zip(score32,theta)]
        if i<4:
            actual=reports[arm,"float32"]["runtime_force"][i]
            assert [-x for x in runtime_score]==actual
        base_valid=flat(arm,"float32","valid")[i] and flat(arm,"float64","valid")[i]
        row["arms"][arm]=dict(
            likelihood_value_fp32=flat(arm,"float32","value")[i],
            likelihood_value_fp64=flat(arm,"float64","value")[i],
            score_fp32=score32,score_fp64=score64,
            own_finite_difference=finest,own_fd_uncertainty=own_unc,own_fd_stable=own_stable,
            own_fd_curve={str(h):[own[h,j][i] for j in (0,1)] for h in (1e-3,3e-4,1e-4)},
            endpoint_value=endpoint["value"],endpoint_likelihood_score=exact,
            endpoint_posterior_score=posterior_ref,endpoint_fd_uncertainty=end_unc,endpoint_fd_stable=end_stable,
            runtime_posterior_score=runtime_score,
            own_derivative_error=metrics(score64,finest),
            runtime_score_vs_own_derivative=metrics(score32,finest),
            runtime_posterior_score_vs_endpoint=metrics(runtime_score,posterior_ref),
            valid=base_valid and fd_valid and endpoint_valid,
            interpretable=base_valid and fd_valid and endpoint_valid and own_stable and end_stable)
    b,a=row["arms"]["before"],row["arms"]["after"]
    own_change=b["own_derivative_error"]["raw_l2_error"]-a["own_derivative_error"]["raw_l2_error"]
    own_resolution=3*(norm(b["own_fd_uncertainty"])+norm(a["own_fd_uncertainty"]))
    endpoint_change=b["runtime_posterior_score_vs_endpoint"]["whitened_l2_error"]-a["runtime_posterior_score_vs_endpoint"]["whitened_l2_error"]
    endpoint_resolution=3*(norm(white_radius(b["endpoint_fd_uncertainty"]))+norm(white_radius(a["endpoint_fd_uncertainty"])))
    verdict=lambda v,r:"improved_at_probe" if v>r else "worsened_at_probe" if v< -r else "unresolved_at_fd_resolution"
    row["comparison"]=dict(score_change_fp32=sub(a["score_fp32"],b["score_fp32"]),
        score_change_fp64=sub(a["score_fp64"],b["score_fp64"]),
        own_error_change=own_change,own_change_resolution=own_resolution,
        own_change_verdict=verdict(own_change,own_resolution),
        endpoint_error_change=endpoint_change,endpoint_change_resolution=endpoint_resolution,
        endpoint_change_verdict=verdict(endpoint_change,endpoint_resolution),
        endpoint_value_change=a["endpoint_value"]-b["endpoint_value"],
        endpoint_score_change=sub(a["endpoint_likelihood_score"],b["endpoint_likelihood_score"]))
    delta=sub(theta,center)
    det=C[0][0]*C[1][1]-C[0][1]*C[1][0]
    gaussian=[-(C[1][1]*delta[0]-C[0][1]*delta[1])/det,
              -(-C[1][0]*delta[0]+C[0][0]*delta[1])/det]
    heuristics={"zero":[0.,0.],"prior_only":[-x/prior_var for x in theta],"saved_gaussian":gaussian}
    row["heuristics"]={k:dict(score=v,error=metrics(v,a["endpoint_posterior_score"])) for k,v in heuristics.items()}
    row["heuristic_dominance_veto_at_probe"]=[
        k for k,v in row["heuristics"].items()
        if v["error"]["whitened_l2_error"]+endpoint_resolution
           < a["runtime_posterior_score_vs_endpoint"]["whitened_l2_error"]]
    rows.append(row)
summary={}
for arm in ("before","after"):
    rr=[r["arms"][arm] for r in rows]
    summary[arm]=dict(
        maximum_own_derivative_component_error=max(max(abs(v) for v in sub(x["score_fp64"],x["own_finite_difference"])) for x in rr),
        maximum_fp32_score_relative_error_to_fp64=max(norm(sub(x["score_fp32"],x["score_fp64"]))/norm(x["score_fp64"]) for x in rr),
        maximum_own_derivative_absolute_error=max(x["own_derivative_error"]["raw_l2_error"] for x in rr),
        maximum_own_derivative_scaled_error=max(x["own_derivative_error"]["raw_l2_error"]/max(1.,norm(x["own_finite_difference"])) for x in rr),
        maximum_fp32_vs_fp64_score_l2=max(norm(sub(x["score_fp32"],x["score_fp64"])) for x in rr),
        median_whitened_endpoint_scaled_error=statistics.median(x["runtime_posterior_score_vs_endpoint"]["whitened_scaled_error"] for x in rr),
        maximum_whitened_endpoint_scaled_error=max(x["runtime_posterior_score_vs_endpoint"]["whitened_scaled_error"] for x in rr),
        maximum_fd_refinement_error=max(max(x["own_fd_uncertainty"]) for x in rr),
        minimum_endpoint_cosine=min(x["runtime_posterior_score_vs_endpoint"]["whitened_cosine"] for x in rr))
summary["maximum_fp32_score_component_change"]=max(abs(v) for r in rows for v in r["comparison"]["score_change_fp32"])
summary["maximum_fp64_score_component_change"]=max(abs(v) for r in rows for v in r["comparison"]["score_change_fp64"])
summary["maximum_endpoint_value_change"]=max(abs(r["comparison"]["endpoint_value_change"]) for r in rows)
summary["maximum_endpoint_score_component_change"]=max(abs(v) for r in rows for v in r["comparison"]["endpoint_score_change"])
summary["all_points_valid"]=all(r["arms"][a]["valid"] for r in rows for a in ("before","after"))
summary["all_fd_checks_stable"]=all(r["arms"][a]["interpretable"] for r in rows for a in ("before","after"))
summary["own_derivative_verdicts"]=[r["comparison"]["own_change_verdict"] for r in rows]
summary["endpoint_agreement_verdicts"]=[r["comparison"]["endpoint_change_verdict"] for r in rows]
repeatability={}
for arm in ("before","after"):
    ob=read(R/"observability-01"/(arm+".json"))
    scores=[v for b in ob["base"] for v in b["score"]]
    forces=[v for b in ob["base"] for v in b["runtime_force"]]
    force_valid=[v for b in ob["base"] for v in b["runtime_force_valid"]]
    expected=[[-f32(s-f32(x*f32(1./f32(prior_var)))) for s,x in zip(g,p["theta"])]
              for g,p in zip(scores,common["probes"])]
    assert expected==forces,(arm,"all-eight observed runtime-force parity")
    assert all(force_valid),(arm,"runtime force invalid")
    repeatability[arm]=dict(
        maximum_repeat_score_component_change=max(abs(a-b) for g,h in zip(scores,flat(arm,"float32","score")) for a,b in zip(g,h)),
        maximum_repeat_score_relative_change=max(norm(sub(g,h))/norm(h)
            for g,h in zip(scores,flat(arm,"float32","score"))),
        maximum_repeat_value_change=max(abs(a-b) for a,b in zip(
            [v for b in ob["base"] for v in b["value"]],flat(arm,"float32","value"))),
        all_eight_runtime_force_parity_exact=True,
        all_eight_runtime_force_valid=all(force_valid))
summary["fp32_repeatability"]=repeatability
summary["finite_difference_curves"]={
    arm:{str(h):max(abs(v) for r in rows for v in sub(r["arms"][arm]["score_fp64"],r["arms"][arm]["own_fd_curve"][str(h)]))
         for h in (1e-3,3e-4,1e-4)}
    for arm in ("before","after")}
result=dict(schema="bayesfilter.m13.score_before_after_summary.v1",question="Did synchronization improve M13 score accuracy?",
    parameters=["lambda_Q","lambda_R"],score_quantity="log likelihood derivative; posterior comparison additionally includes Gaussian prior",
    summary=summary,probes=rows,statistical_ranking_supported=False,
    conditional_deterministic_measurement=True,new_hmc_transitions=0,
    heuristic_dominance_verdict={r["probe"]["label"]:r["heuristic_dominance_veto_at_probe"] for r in rows})
(R/"report.json").write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n")
print(json.dumps(summary,indent=2))
