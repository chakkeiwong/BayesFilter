#!/usr/bin/env python3
"""Post-run diagnostic reporting only; no TensorFlow, GPU, or selection mutation."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def finite_max(values):
    numeric = [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
    return max(numeric) if numeric else None


def report(root, status_only=False):
    root = Path(root)
    state = read(root/"result.json")
    summaries, detail, health = [], [], []
    for path in sorted(root.glob("*/result.json")):
        r = read(path)
        summaries.append(dict(model=r["model"], status=r["status"], active=r.get("active"),
            completed_rows=len(r["rows"]), wall_seconds=r.get("wall_seconds"),
            summary=r.get("summary"), timings=r.get("timing_median_seconds"),
            time_ratio=r.get("candidate_over_baseline_time"), source_mismatches=r.get("source_mismatches")))
        for row in r["rows"]:
            detail.append(dict(model=r["model"], **row))
        for p in path.parent.glob("*-health.json"):
            h = read(p)
            for arm, run in h["arms"].items():
                for field in ("sinkhorn_row_error", "sinkhorn_column_error", "reset_covariance_relative_error", "reset_margin"):
                    i = h["trace_fields"].index(field)
                    entries = [row[i] for trajectory in run["trace"] for row in trajectory]
                    health.append(dict(model=r["model"], arm=arm, field=field,
                        group=p.stem, maximum=finite_max(entries)))
    if status_only:
        print(json.dumps(dict(status=state["status"], active_model=state.get("active_model"),
            seconds=state["diagnostic_used_seconds"], models=summaries), indent=2))
        return
    if state["status"] == "RUNNING":
        raise ValueError("cannot issue terminal report while workers are running")
    numerical = [s for s in summaries if s["status"] in (
        "WITHIN_ONE_PERCENT_ON_INSPECTED_INPUTS", "MATERIAL_DIFFERENCE")]
    payload = dict(status=state["status"], source_result=str(root/"result.json"),
        source_sha256=hashlib.sha256((root/"result.json").read_bytes()).hexdigest(),
        diagnostic_used_seconds=state["diagnostic_used_seconds"], models=summaries,
        rows=detail, health=health,
        hard_veto_models=[s["model"] for s in summaries if s not in numerical],
        material_models=[s["model"] for s in numerical if s["status"] == "MATERIAL_DIFFERENCE"],
        no_material_difference_models=[s["model"] for s in numerical if s["status"] == "WITHIN_ONE_PERCENT_ON_INSPECTED_INPUTS"],
        statistically_supported_ranking=False, default_changed=False, posterior_sampling_complete=False)
    (root/"report.json").write_text(json.dumps(payload, indent=2, allow_nan=False)+"\n")
    lines = ["# Epsilon 1, five Sinkhorn iterations: paired materiality results", "",
        "Baseline: current corrected-source selected controls. Candidate: epsilon=1, iterations=5; all other controls and mixture weights held fixed.",
        "N4032/K2016, GPU/XLA, TF32 off; FP32 except SIR FP64 recursion/FP32 pairs.",
        "Saved plus two fresh datasets, six local parameter positions per model. These are filter log likelihoods and physical-parameter scores, not log priors or posterior samples.", "",
        "Changes below use 100*abs(candidate-baseline)/max(1,abs(baseline)). The 1% screen is componentwise; max score change is the largest component over all inspected inputs.", "",
        "| Model | Status | Max log-likelihood change % | Max score change % | Max absolute log-likelihood change | Max centered change | Baseline / candidate seconds per B2 call |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for s in summaries:
        q, t = s.get("summary") or {}, s.get("timings") or {}
        def f(key, scale=1):
            return f"{scale*q[key]:.6g}" if key in q else "N/A"
        timing = f"{t['baseline']:.3f} / {t['candidate']:.3f}" if t else "N/A"
        lines.append(f"| {s['model']} | {s['status']} | {f('max_likelihood_scaled_error',100)} | {f('max_score_scaled_error',100)} | {f('max_abs_log_likelihood_delta')} | {f('max_abs_centered_log_likelihood_delta')} | {timing} |")
    lines += ["", "Centered change subtracts each arm's nominal-parameter log likelihood within the same dataset. It is explanatory, not an additional predeclared veto.",
        "", "## Nominal parameter, saved dataset", "",
        "| Model | Baseline log likelihood | Candidate log likelihood | Baseline score | Candidate score |",
        "|---|---:|---:|---|---|"]
    for row in detail:
        if row["dataset"] == "saved" and row["position"] == 0:
            b, c = row["baseline"], row["candidate"]
            def vector(xs):
                return ", ".join(f"{x:.8g}" if isinstance(x,(float,int)) else str(x) for x in xs)
            lines.append(f"| {row['model']} | {b['value']} | {c['value']} | {vector(b['score'])} | {vector(c['score'])} |")
    lines += ["", "## Transport and reset diagnostics", "",
        "| Model | Arm | Max Sinkhorn row residual | Max column residual | Max reset covariance relative error |",
        "|---|---|---:|---:|---:|"]
    for s in summaries:
        for arm in ("baseline", "candidate"):
            values=[]
            for field in ("sinkhorn_row_error", "sinkhorn_column_error", "reset_covariance_relative_error"):
                v=finite_max([h["maximum"] for h in health if h["model"]==s["model"] and h["arm"]==arm and h["field"]==field])
                values.append(f"{v:.6g}" if v is not None else "N/A")
            lines.append(f"| {s['model']} | {arm} | "+" | ".join(values)+" |")
    lines += ["", "## Decision and inference status", "",
        "| Decision | Primary criterion | Veto diagnostic status | Main uncertainty | Next justified action | Not concluded |",
        "|---|---|---|---|---|---|",
        "| Retain existing selections while reporting this ablation | Per-model 1% comparison above | See invalid-arm/source status | Limited data/parameter coverage; both epsilon and iterations change | Assess model-specific materiality before any new tuning | No default or HMC promotion |", "",
        "| Inference status | Finding |", "|---|---|",
        "| Hard veto screen | Kernel validity, finite output, full horizon and source identity retained; failures shown explicitly |",
        "| Statistically supported ranking | None; no runtime or method-quality superiority claim |",
        "| Descriptive-only differences | Exact differences on these frozen inputs and three warmed timing repeats |",
        "| Default-readiness | No changes to selected controls or default precision |",
        "| Next evidence needed | Broader target-specific assessment for any proposed default change; separate HMC tuning and posterior checks |", "",
        "Post-run red team: the selected baseline is not a model oracle. Small relative changes can coexist with an important absolute or centered log-likelihood change. Frozen beta isolates the requested intervention but may not be optimal for it. The weakest evidence is coverage outside six local positions and three datasets. A different result on a new dataset or posterior region would overturn a broad similarity claim. This experiment does not estimate likelihood MCSE, because its fresh replications change datasets as well as particle designs.", "",
        f"Charged worker-wall GPU time: {state['diagnostic_used_seconds']/3600:.4f} h. All per-position values, scores and health traces are retained beside this report."]
    (root/"report.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({k:payload[k] for k in ("status","diagnostic_used_seconds","hard_veto_models","material_models","no_material_difference_models")}, indent=2))


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("output",type=Path)
    p.add_argument("--status",action="store_true")
    a=p.parse_args()
    report(a.output,a.status)
