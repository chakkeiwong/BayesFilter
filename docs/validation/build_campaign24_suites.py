"""Declarative suite authoring for the reviewed 24-hour validation campaign.

This creates design JSON only. Sampling always uses the shared validation CLI.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

PLAN="docs/plans/bayesfilter-inference-validation-24h-campaign-2026-09-16.md"
COUNTS={"warmup_chunk_results":500,"warmup_min_results":2000,"warmup_check_window_results":1000,
        "warmup_max_results":10000,"retained_chunk_results":500,"retained_min_results":1000,
        "retained_max_results":10000}


def design(name, engine, target, route, device, seconds, *, replications=1, **kwargs):
    options=dict(plan_file=str(Path(PLAN).resolve()),member_rule="first_verified",posterior_members="selected",
                 posterior_settings=COUNTS,preparation_preset="standard",
                 search={"pilot_enabled":True,"refinement_rounds":1,"max_candidates":100,
                         "total_budget_units":300,"repair_reserve_units":40})
    options.update(kwargs.pop("options",{}))
    result=dict(design_id=name,engine=engine,scenario=dict(target=target,route=route,control="baseline",
        start="dispersed",parameters={}),replications=replications,draws=500,seed=2026091601,
        budget_seconds=seconds,purpose="Funded independent validation of the public inference procedure",
        numerical_provenance=PLAN+"; native thresholds and fixed predeclared counts; no default promotion",
        device=device,alpha=.05,null_draws=1999,rank_draws=7,step_size=.6,leapfrog_steps=5,member_l=3,
        l_grid=[3,5,9,13,18,25],measurement_draws=128,posterior_cap=10000,mcse_tolerance=.1,
        accuracy_tolerance=.25,multiplicity=1,phase="development",options=options)
    result.update(kwargs)
    return result


def pilot(device):
    tag="gpu" if device=="gpu" else "cpu"
    return [design(tag+"-sbc-pilot","sbc","normal_conjugate","ordinary",device,1800,
                   replications=2,rank_draws=2),
            design(tag+"-simplex-pilot","accuracy","dirichlet","prepared",device,600,
                   options={"posterior_members":"all"}),
            design(tag+"-stopping-pilot","stopping","gaussian","prepared",device,900,
                   replications=2,options={"fixed_comparator":{"warmup_results":2000,"retained_results":4000}})]


def main_campaign(device):
    gpu=device=="gpu"
    tag="gpu" if gpu else "cpu"
    shards=8 if gpu else 16
    rows=[design(f"{tag}-normal-sbc-{i:02d}","sbc","normal_conjugate","ordinary",device,
                 7000 if gpu else 4000,replications=4,seed=2026091603,
                 phase="development" if gpu else "confirmation") for i in range(shards)]
    group={"group_id":tag+"-normal-sbc","design_ids":[r["design_id"] for r in rows],
           "replications":4*shards}
    for target in ("gaussian","mixture"):
        for part in range(2):
            row=design(f"{tag}-{target}-stopping-{part}","stopping",target,"prepared",device,
                       (2100 if target=="gaussian" else 3000) if gpu else
                       (1600 if target=="gaussian" else 2200),replications=4,seed=2026091604,
                       options={"fixed_comparator":{"warmup_results":2000,"retained_results":4000}})
            if target=="mixture": row["scenario"]["start"]="single_mode"
            rows.append(row)
    for target in ("beta_binomial","lgssm_location"):
        rows.append(design(f"{tag}-{target}-sbc","sbc",target,"ordinary",device,2000 if gpu else 1800,
                           replications=2,rank_draws=2,seed=2026091605))
    for epsilon in (.3,.6,1.):
        child=design("kernel-child","invariance","gaussian","frozen",device,300,
                     replications=128,step_size=epsilon,rank_draws=7,seed=2026091606)
        rows.append(design(f"{tag}-kernel-power-eps{str(epsilon).replace('.','p')}","power",
                           "gaussian","frozen",device,1000 if gpu else 900,replications=32,
                           seed=2026091606,options={"calibration_design":child,
                             "power_controls":["baseline","noop","wrong_energy"]}))
    rows.append(design(tag+"-affine-rotated","accuracy","rotated_gaussian","fixed_transport",
                       device,1800 if gpu else 1200,replications=2,seed=2026091607))
    targets=("gaussian","rotated_gaussian","banana","funnel","student_t","cauchy","mixture",
             "gamma","beta","dirichlet","normal_conjugate","beta_binomial","lgssm_location")
    for target in targets:
        rows.append(design(tag+"-mechanics-"+target,"mechanics",target,"frozen",device,
                           150 if gpu else 60,replications=32,seed=2026091608))
    if gpu:
        rows.append(design("gpu-simplex-selected","accuracy","dirichlet","prepared",device,1800,
                           replications=2,seed=2026091609))
        rows.append(design("gpu-funnel-ordinary","accuracy","funnel","ordinary",device,1500,
                           seed=2026091610))
    else:
        rows.append(design("cpu-gamma-prepared","accuracy","gamma","prepared",device,600,
                           seed=2026091609))
        rows.append(design("cpu-cauchy-ordinary","accuracy","cauchy","ordinary",device,900,
                           seed=2026091610))
    return {"schema":"bayesfilter.inference_validation_suite.v1","suite_id":"campaign24-main-"+tag,
            "profile":"campaign","profiles":{"campaign":["sbc","stopping","power","accuracy","mechanics"]},
            "designs":rows,"aggregate_groups":[group]}


def repair_campaign(device):
    tag="gpu" if device=="gpu" else "cpu"
    rows=[]
    for i in range(4):
        row=design(f"{tag}-native-search-{i}","sbc","normal_conjugate","ordinary",device,
                   1500 if device=="gpu" else 1000,replications=2,rank_draws=2,
                   seed=2026091611,options={"native_search":True})
        row["options"].pop("search")
        rows.append(row)
    group={"group_id":tag+"-native-search","design_ids":[r["design_id"] for r in rows],"replications":8}
    if device=="gpu":
        rows.extend(r for r in main_campaign(device)["designs"] if not r["design_id"].startswith("gpu-normal-sbc-"))
    return {"schema":"bayesfilter.inference_validation_suite.v1","suite_id":"campaign24-repair-"+tag,
            "profile":"campaign","profiles":{"campaign":["sbc","stopping","power","accuracy","mechanics"]},
            "designs":rows,"aggregate_groups":[group]}


def prepared_campaign(pilot_only, datasets=64):
    rows=[design(f"cpu-prepared-sbc-{i:02d}","sbc","normal_conjugate","prepared","cpu_reference",
                 600 if pilot_only else 2400,replications=2 if pilot_only else datasets//16,
                 rank_draws=2 if pilot_only else 7,seed=2026091612 if pilot_only else 2026091613+int(datasets==32),
                 phase="development" if pilot_only else "confirmation") for i in range(1 if pilot_only else 16)]
    return {"schema":"bayesfilter.inference_validation_suite.v1",
            "suite_id":"campaign24-prepared-pilot" if pilot_only else "campaign24-prepared-confirmation",
            "profile":"statistical","profiles":{"statistical":["sbc"]},"designs":rows,
            "aggregate_groups":[] if pilot_only else [{"group_id":"cpu-prepared-sbc",
                "design_ids":[r["design_id"] for r in rows],"replications":datasets}]}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--phase",choices=("pilot","calibration","calibration32","main","repair","prepared-pilot","prepared-confirmation","prepared-confirmation32"),default="pilot")
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    if args.phase.startswith("prepared-"):
        suite=prepared_campaign(args.phase=="prepared-pilot",32 if args.phase.endswith("32") else 64)
        path=args.output/(args.phase+".json")
        with path.open("x") as f: json.dump(suite,f,indent=2)
        print(path, sum(d["budget_seconds"] for d in suite["designs"]))
        return
    if args.phase in {"main","repair"}:
        for device in ("cpu_reference","gpu"):
            suite=(main_campaign if args.phase=="main" else repair_campaign)(device)
            path=args.output/(args.phase+("-gpu.json" if device=="gpu" else "-cpu.json"))
            with path.open("x") as f: json.dump(suite,f,indent=2)
            print(path, sum(d["budget_seconds"] for d in suite["designs"]))
        return
    if args.phase in {"calibration","calibration32"}:
        rows=[]
        for n in ((32,) if args.phase=="calibration32" else (64,128)):
            for severity in ((1.,) if args.phase=="calibration32" else (.25,.5,1.)):
                rows.append(design(f"rank-power-n{n}-bias{str(severity).replace('.','p')}",
                    "power","normal_conjugate","reference","cpu_reference",300 if n==32 else 600,replications=256,
                    draws=n,seed=2026091615 if n==32 else 2026091602,options={"location_severity":severity,"normal_data_count":6,
                        "include_radius":True,"include_ks":False}))
        suite={"schema":"bayesfilter.inference_validation_suite.v1","suite_id":"campaign24-rank-power",
               "profile":"statistical","profiles":{"statistical":["power"]},"designs":rows}
        path=args.output/("rank-power32.json" if args.phase=="calibration32" else "rank-power.json")
        with path.open("x") as f: json.dump(suite,f,indent=2)
        print(path)
        return
    for device in ("cpu_reference","gpu"):
        tag="gpu" if device=="gpu" else "cpu"
        suite={"schema":"bayesfilter.inference_validation_suite.v1","suite_id":"campaign24-pilot-"+tag,
               "profile":"numerical","profiles":{"numerical":["accuracy","stopping","sbc"]},"designs":pilot(device)}
        path=args.output/("pilot-"+tag+".json")
        with path.open("x") as f: json.dump(suite,f,indent=2)
        print(path)


if __name__=="__main__": main()
