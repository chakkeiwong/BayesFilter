"""Diagnostic-only reporting of captured independent R runs; no runtime decisions.

NumPy accelerates replicate-bootstrap reporting only. R filter execution and
all campaign scheduling remain independent of these arrays. Python's fixed
Random index stream preserves the preceding report's bootstrap convention.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import random
import statistics

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/plans/artifacts/iapf-r-24hour-campaign-20260921-01"
PAPER = {5: (.09,1000,6.93), 10: (.14,1000,15.11), 20: (.19,1000,27.61),
         40: (.23,1033,42.41), 80: (.35,1142,71.88)}
PAPER_BASELINES={"bpf":{5:(.51,99),10:(6.4,99)},
                 "fully_adapted":{5:(.10,26.04),10:(.17,52.71),20:(.53,84.98)}}
HEURISTICS = ("bpf", "fully_adapted", "sis")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def rows(path):
    if path.exists():
        with path.open() as stream:
            yield from csv.DictReader(stream)


@lru_cache(maxsize=4)
def indices(n):
    rng = random.Random(9212026)
    return np.fromiter((rng.randrange(n) for _ in range(4000*n)),
                       dtype=np.int32, count=4000*n).reshape(4000,n)


def interval(values, sd=False):
    a = np.asarray(values, dtype=np.float64)
    if len(a) < 2 or not np.isfinite(a).all():
        return None
    samples = a[indices(len(a))]
    estimates = np.std(samples,axis=1,ddof=1) if sd else np.mean(samples,axis=1)
    if not np.isfinite(estimates).all():
        return None
    return np.quantile(estimates,[.025,.975]).tolist()


def expected_methods(arm):
    return (["qr","short_qr",*HEURISTICS] if arm == "controller" else
            ["qr",*HEURISTICS] if arm == "baselines" else [arm])


def load_cell(phase, cell, attempts):
    selected = [a for a in attempts if a["phase"] == phase["phase"] and a["cell_id"] == cell["cell_id"]]
    expected = {(str(i),m) for i in range(cell["first"],cell["first"]+cell["repeats"])
                for m in expected_methods(cell["arm"])}
    records, prefixes = {}, {}
    provenance, hashes = [], defaultdict(set)
    fit_count = tail_count = 0
    fits_ok = tails_ok = True
    for attempt in selected:
        path = OUT / attempt["name"] / "results"
        for name in ("observations.csv", "kalman.csv"):
            value = digest(path/name)
            if value != attempt[name+"_sha256"]:
                raise RuntimeError("data changed after worker inspection")
            hashes[name].add(value)
        local = {}
        for row in rows(path/"replicates.csv"):
            key = row["replication"],row["method"]
            if key in records or key not in expected:
                raise RuntimeError("duplicate or unexpected method/replica")
            seed = 53000000+100000*cell["dimension"]+10*int(key[0])+dict(bpf=2,fully_adapted=3,sis=4).get(key[1],1)
            if int(row["seed"]) != seed:
                raise RuntimeError("filter seed mismatch")
            if row["status"] == "complete" and not all(math.isfinite(float(row[k])) for k in ("log_ratio","ratio","wall_seconds")):
                raise RuntimeError("non-finite completed record")
            records[key] = local[key] = row
        good = {k:r for k,r in local.items() if r["status"] == "complete"}
        fitted = {k:r for k,r in good.items() if k[1] not in HEURISTICS}
        seen_prefix, seen_fit, seen_tail = defaultdict(set),defaultdict(set),defaultdict(set)
        for row in rows(path/"prefixes.csv"):
            key = row["replication"],row["method"]
            if key not in good:
                continue
            t = int(row["time"])
            if t in seen_prefix[key] or not math.isfinite(float(row["log_prefix_error"])):
                raise RuntimeError("invalid prefix record")
            if row["situation"] not in ("ordinary","large_innovation"):
                raise RuntimeError("unknown innovation situation")
            seen_prefix[key].add(t)
            prefixes[key,t] = row
        for row in rows(path/"fits.csv"):
            key = row["replication"],row["method"]
            if key not in fitted:
                continue
            identity = int(row["iteration"]),int(row["time"])
            if identity in seen_fit[key]:
                raise RuntimeError("duplicate fit record")
            seen_fit[key].add(identity)
            fit_count += 1
            fits_ok = fits_ok and int(row["convergence"]) == 0
        for row in rows(path/"tails.csv"):
            key = row["replication"],row["method"]
            if key not in fitted:
                continue
            t = int(row["time"])
            if t in seen_tail[key]:
                raise RuntimeError("duplicate tail record")
            seen_tail[key].add(t)
            tail_count += 1
            tails_ok = tails_ok and row["passed"] == "TRUE"
        for key,row in good.items():
            if seen_prefix[key] != set(range(1,101)):
                raise RuntimeError("incomplete completed-replica prefixes")
            if key in fitted:
                expected_fit = {(iteration,t) for iteration in range(1,int(row["iterations"])) for t in range(1,101)}
                if seen_fit[key] != expected_fit or seen_tail[key] != set(range(1,101)):
                    raise RuntimeError("incomplete completed-replica fit/tail records")
                if not (path/f"{key[1]}-{key[0]}.rds").is_file():
                    raise RuntimeError("missing fitted-guide history")
        provenance.append(dict(attempt=attempt["name"],exit_code=attempt["exit_code"],
            files={name:digest(path/name) for name in ("replicates.csv","prefixes.csv","fits.csv","tails.csv") if (path/name).exists()}))
    if any(len(hashes[n]) != 1 for n in ("observations.csv","kalman.csv")):
        raise RuntimeError("different data or Kalman authority within a cell")
    return records,prefixes,dict(expected_pairs=len(expected),recorded_pairs=len(records),
        complete_pairs=sum(r["status"] == "complete" for r in records.values()),
        missing_pairs=sorted(expected-records.keys()),fit_rows=fit_count,tail_rows=tail_count,
        fits_ok=fits_ok,tails_ok=tails_ok,source_and_record_identity_checked=True,
        data_hashes={k:next(iter(v)) for k,v in hashes.items()},parents=provenance)


def summarize_cell(cell, records, prefixes, validity):
    methods = {}
    for method in expected_methods(cell["arm"]):
        raw = [r for k,r in records.items() if k[1] == method]
        good = sorted((r for r in raw if r["status"] == "complete"),key=lambda r:int(r["replication"]))
        result = dict(n=len(good),planned=cell["repeats"],failures=[r for r in raw if r["status"] != "complete"])
        methods[method] = result
        if not good:
            continue
        ratios = [float(r["ratio"]) for r in good]
        log_ratios = [float(r["log_ratio"]) for r in good]
        result.update(mean=statistics.mean(ratios),sd=statistics.stdev(ratios) if len(good)>1 else None,
            mean_ci95=interval(ratios),sd_ci95=interval(ratios,sd=True),
            minimum=min(ratios),maximum=max(ratios),minimum_log_ratio=min(log_ratios),
            maximum_log_ratio=max(log_ratios),ratio_underflow_count=sum(v==0 for v in ratios))
        for field in ("particles","wall_seconds","iterations","resampling_count"):
            result["mean_"+field] = statistics.mean(float(r[field]) for r in good)
        result["complete"] = len(good) == cell["repeats"]
        if method not in HEURISTICS:
            published=PAPER[cell["dimension"]]
            result["paper_comparison"]=dict(sd=published[0],mean_particles=published[1],mean_resampling=published[2])
        elif cell["dimension"] in PAPER_BASELINES.get(method,{}):
            published=PAPER_BASELINES[method][cell["dimension"]]
            result["paper_comparison"]=dict(sd=published[0],mean_particles=10000 if method=="bpf" else 5000,mean_resampling=published[1])
        if method not in HEURISTICS:
            paper_sd,paper_n,paper_resampling = PAPER[cell["dimension"]]
            mean_ci,sd_ci = result["mean_ci95"],result["sd_ci95"]
            result["practical_primary_pass"] = bool(result["complete"] and mean_ci and sd_ci and
                mean_ci[0]>=.8 and mean_ci[1]<=1.2 and sd_ci[1]<=2*paper_sd and result["mean_particles"]<=1.5*paper_n)
            result["literal_pattern_pass"] = bool(result["complete"] and sd_ci and
                sd_ci[0]>=.5*paper_sd and sd_ci[1]<=2*paper_sd and
                .5*paper_resampling<=result["mean_resampling_count"]<=2*paper_resampling)
    per_rep = defaultdict(lambda:defaultdict(list))
    for (key,t),row in prefixes.items():
        per_rep[key[1],row["situation"]][key[0]].append(math.expm1(float(row["log_prefix_error"]))**2)
    means = {k:{i:statistics.mean(v) for i,v in reps.items()} for k,reps in per_rep.items()}
    conditional = []
    for method in methods:
        if method in HEURISTICS:
            continue
        for situation in ("ordinary","large_innovation"):
            own = means.get((method,situation),{})
            for comparator in (*HEURISTICS, "qr"):
                if comparator == method:
                    continue
                other = means.get((comparator,situation),{})
                item = dict(method=method,comparator=comparator,situation=situation)
                if set(own) != set(other) or len(own) != cell["repeats"]:
                    item["status"] = "incomplete"
                else:
                    delta = [own[i]-other[i] for i in sorted(own,key=int)]
                    ci = interval(delta)
                    item.update(status="complete",mean_mse=statistics.mean(own.values()),
                        comparator_mean_mse=statistics.mean(other.values()),difference=statistics.mean(delta),
                        difference_ci95=ci,heuristic_promotion_veto=comparator in HEURISTICS and statistics.mean(delta)>0,
                        pointwise_difference_supported=bool(ci and (ci[0]>0 or ci[1]<0)))
                conditional.append(item)
        relevant = [r for r in conditional if r["method"]==method and r["comparator"] in HEURISTICS]
        verdict = ("incomplete" if any(r["status"]!="complete" for r in relevant) else
                   "veto" if any(r["heuristic_promotion_veto"] for r in relevant) else "passed observed screen")
        methods[method]["heuristic_dominance"] = verdict
        methods[method]["reference_screen_pass"] = bool(methods[method].get("practical_primary_pass") and
            verdict=="passed observed screen" and validity["fits_ok"] and validity["tails_ok"] and not validity["missing_pairs"])
    pairs = {}
    qr = {k[0]:r for k,r in records.items() if k[1]=="qr" and r["status"]=="complete"}
    short = {k[0]:r for k,r in records.items() if k[1]=="short_qr" and r["status"]=="complete"}
    if len(qr)==cell["repeats"] and qr.keys()==short.keys():
        for field in ("particles","wall_seconds","squared_likelihood_error"):
            def value(r):
                return (float(r["ratio"])-1)**2 if field=="squared_likelihood_error" else float(r[field])
            delta=[value(short[i])-value(qr[i]) for i in sorted(qr,key=int)]
            pairs[field]=dict(mean_difference=statistics.mean(delta),difference_ci95=interval(delta))
    return dict(**cell,validity=validity,methods=methods,conditional_comparisons=conditional,controller_pairs=pairs)


def write_report(result, directory):
    lines=["# Independent R campaign phase result", "", f"Phase: {result['phase']}. Fixed-count CPU reference analysis.", "",
        "| Data | d | Method | Complete | Mean ratio (95% interval) | SD (95% interval) | Mean N | Practical | Heuristic | Literal pattern |",
        "|---|---:|---|---:|---|---|---:|---|---|---|"]
    def ci_text(value):
        return "unavailable" if value is None else f"[{value[0]:.4g}, {value[1]:.4g}]"
    for cell in result["cells"]:
        for method,v in cell["methods"].items():
            if not v["n"]:
                continue
            sd_text="unavailable" if v["sd"] is None else f"{v['sd']:.5g}"
            lines.append(f"| {cell['data_seed']} | {cell['dimension']} | {method} | {v['n']}/{v['planned']} | "
                f"{v['mean']:.5g} {ci_text(v['mean_ci95'])} | {sd_text} {ci_text(v['sd_ci95'])} | "
                f"{v['mean_particles']:.5g} | {v.get('practical_primary_pass','N/A')} | "
                f"{v.get('heuristic_dominance','comparator')} | {v.get('literal_pattern_pass','N/A')} |")
    lines += ["", "A heuristic veto means an observed conditional loss to a simple comparator; it does not",
        "establish population inferiority when the paired interval includes zero. Intervals are",
        "pointwise, conditional on each fixed dataset, and are not simultaneous or cross-dataset inference.",
        "Finite log ratios with exponentiation underflow are counted in summary.json. A displayed zero",
        "sample SD for an underflowed baseline is not zero theoretical variance.", "",
        "| Data | d | Method | Observed / paper SD | Observed / paper mean N | Observed / paper resampling | Whole-run seconds |",
        "|---|---:|---|---|---|---|---:|"]
    for cell in result["cells"]:
        for method,v in cell["methods"].items():
            if not v["n"]:
                continue
            p=v.get("paper_comparison",{})
            sd_text="unavailable" if v["sd"] is None else f"{v['sd']:.5g}"
            lines.append(f"| {cell['data_seed']} | {cell['dimension']} | {method} | {sd_text} / {p.get('sd','unreported')} | "
                f"{v['mean_particles']:.5g} / {p.get('mean_particles','unreported')} | "
                f"{v['mean_resampling_count']:.5g} / {p.get('mean_resampling','unreported')} | {v['mean_wall_seconds']:.5g} |")
    lines += ["", "Paper values come from the preserved GJL first-study tables. Different datasets and",
        "QR's different fitting objective prevent literal equivalence claims. Whole-run timing",
        "includes learning and final evaluation and is not matched-cost or cross-machine evidence.", "",
        "| Decision | Primary criterion | Veto diagnostics | Main uncertainty | Next justified action | Not concluded |",
        "|---|---|---|---|---|---|"]
    for cell in result["cells"]:
        for method,v in cell["methods"].items():
            if method in HEURISTICS:
                continue
            lines.append(f"| {cell['data_seed']} {method}: {'passes reference screen' if v.get('reference_screen_pass') else 'not promoted'} | "
                f"{v.get('practical_primary_pass',False)} | heuristic={v.get('heuristic_dominance')}; "
                f"fits={cell['validity']['fits_ok']}; tails={cell['validity']['tails_ok']} | "
                "Finite replicas and fixed data | Continue the predeclared replication/analysis sequence | Author identity or production readiness |")
    lines += ["", "| Inference status | Finding |", "|---|---|",
        "| Hard veto screen | Exact record identity, source hashes, seeds, fit/tail and missing-pair status are preserved per cell. |",
        "| Statistically supported ranking | No global ranking. Paired pointwise intervals in summary.json support only their specified contrast and fixed data. |",
        "| Descriptive-only differences | Raw means, extreme ratios, resampling counts and unmatched-cost timing do not establish superiority. |",
        "| Default-readiness | No algorithm or production default changes. QR differs from paper Equation15. |",
        "| Next evidence needed | Complete the remaining frozen campaign; exact original settings/data and multivariate full-filter R/TF parity remain open. |", "",
        "Post-run skeptical review: the strongest alternative explanation is data-specific Monte Carlo",
        "variation or unobserved rare large weights. Larger independent fixed-count runs could overturn",
        "a small-sample screen. The weakest evidence is literal matching to unpublished original data",
        "and implementation choices. These results do not prove correctness or failure of LEDH, KDM or HMC.", ""]
    (directory/"result.md").write_text("\n".join(lines))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("phase_file",type=Path)
    parser.add_argument("--output-name",default="analysis-v1")
    args=parser.parse_args()
    phase=json.loads(args.phase_file.read_text())
    manifest=json.loads((OUT/"manifest.json").read_text())
    captured=manifest["phases"][phase["phase"]]
    for name,expected in captured["source_hashes"].items():
        if digest(OUT/phase["phase"]/"source-snapshot"/name)!=expected:
            raise RuntimeError("source snapshot mismatch")
    attempts=[json.loads(line) for line in (OUT/"attempts.jsonl").read_text().splitlines()]
    directory=OUT/phase["phase"]/args.output_name
    directory.mkdir(exist_ok=False)
    result=dict(phase=phase["phase"],plan=phase["plan"],bootstrap_repeats=4000,bootstrap_seed=9212026,
        analysis_source_sha256=digest(Path(__file__)),cells=[],
        uncertainty="pointwise replicate bootstrap conditional on each fixed dataset; no optional stopping")
    (directory/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    for cell in phase["cells"]:
        records,prefixes,validity=load_cell(phase,cell,attempts)
        result["cells"].append(summarize_cell(cell,records,prefixes,validity))
    result["complete_coverage"]=all(not c["validity"]["missing_pairs"] for c in result["cells"])
    (directory/"summary.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    write_report(result,directory)
    print(json.dumps(dict(phase=phase["phase"],complete_coverage=result["complete_coverage"],result=str(directory/"result.md"))))


if __name__=="__main__":
    main()
