"""Post-run assembly and provenance checks for independent R validation."""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import subprocess

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "docs/plans/artifacts/iapf-r-independent-validation-20260922-01"
PLAN = "docs/plans/iapf-r-independent-validation-2026-09-22.md"


def read(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def number(text):
    return None if text in ("", "NA", None) else float(text)


def fmt(text):
    value = number(text)
    return "—" if value is None else f"{value:.4g}"


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |",
                       "|" + "---|" * len(headers)] +
                      ["| " + " | ".join(map(str, row)) + " |" for row in rows])


def main():
    records = []
    for path in sorted(ROOT.glob("attempt*/manifest.json")):
        m = json.loads(path.read_text())
        assert m["status"] != "running", path
        for rel, expected in m["source_sha256"].items():
            assert sha(path.parent / "source" / rel) == expected, (path, rel)
        for rel, expected in m.get("input_sha256", {}).items():
            assert sha(REPO / rel) == expected, (path, rel)
        for rel, expected in m["output_sha256"].items():
            assert sha(path.parent / rel) == expected, (path, rel)
        if m["stage"] == "pair" and m["status"] == "complete":
            a = read(path.parent / "results/learning.csv")
            b = read(path.parent / "results/probes.csv")
            assert len(a) == 2 and len({r["filter_seed"] for r in a}) == 1
            assert len({r["filter_seed"] for r in b if r["N"] == "1000"}) == 1
            oracle = [r for r in b if r["method"] == "exact_full"]
            assert len(oracle) == 1 and abs(float(oracle[0]["terminal_error"])) <= 1e-7
            for row in a:
                if row["status"] == "complete":
                    assert math.isclose(float(row["training_seconds"]) + float(row["original_final_seconds"]),
                                        float(row["wall_seconds"]), abs_tol=1e-8)
                    assert {int(r["N"]) for r in b if r["method"] == row["method"]} == {250, 1000, 4000}
        records.append((path, m))
    reports = [(p, m) for p, m in records if m["stage"] == "report" and m["status"] == "complete"]
    assert reports, "No completed statistical report"
    report = reports[-1][0].parent / "results"
    primary = read(report / "primary-comparison.csv")
    reliability = read(report / "reliability.csv")
    grid = read(report / "accuracy-cost-grid.csv")
    costs = read(report / "precision-matched-cost.csv")
    heuristics = read(report / "conditional-heuristics.csv")
    learning = read(report / "learning.csv")
    probes = read(report / "probes.csv")
    cell_index = ROOT / ("cells-completion.csv" if (ROOT / "cells-completion.csv").exists() else "cells.csv")
    cells = read(cell_index)
    checks = read(report / "checks.csv")
    completed = sum(r["status"] == "complete" for r in learning)
    failed = [r for r in learning if r["status"] != "complete"]
    vetoes = [r for r in heuristics if r["observed_promotion_veto"] == "TRUE"]
    candidate_vetoes = [r for r in vetoes if r["method"] == "ridge_bounded1"]
    fit_summary = []
    for dimension in (5, 10, 20, 40, 80):
        fits = []
        for cell in cells:
            if int(cell["dimension"]) != dimension or cell["status"] != "complete":
                continue
            fits.extend(row for row in read(ROOT / cell["attempt"] / "results/fit-diagnostics.csv")
                        if row["method"] == "ridge_bounded1" and number(row["kkt"]) is not None)
        fit_summary.append(dict(dimension=dimension, recorded_fits=len(fits),
            positive_ridge_fits=sum(float(row["lambda"]) > 0 for row in fits),
            active_curvature_bound_fits=sum(float(row["active_constraints"]) > 0 for row in fits),
            active_set_fallback_fits=sum(float(row["active_set_iterations"]) > 0 for row in fits),
            max_KKT=max((float(row["kkt"]) for row in fits), default=None),
            min_weight_ESS=min((float(row["weight_ess"]) for row in fits), default=None)))
    with (ROOT / "fit-summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fit_summary[0]))
        writer.writeheader()
        writer.writerows(fit_summary)
    ranking = {r["dimension"]: r["verdict"] for r in primary}
    calibration = [r for r in grid if r["calibration_interval_excludes_one"] == "TRUE"]
    worker_seconds = sum(m["wall_seconds"] for _, m in records)
    summary = dict(status="complete" if all(r["status"] == "complete" for r in cells) else "incomplete",
        planned_cells=80, complete_cells=sum(r["status"] == "complete" for r in cells),
        planned_learners=160, learning_records=len(learning), learning_complete=completed,
        learning_failures=len(failed), probes=len(probes), primary_rankings=ranking,
        explanatory_fit_summary=fit_summary,
        observed_heuristic_vetoes=len(vetoes), repaired_heuristic_vetoes=len(candidate_vetoes),
        nominal_calibration_screen_flags=len(calibration),
        qualifying_precision_rungs=sum(r["precision_qualified"] == "TRUE" for r in grid
            if r["method"] in ("qr", "ridge_bounded1")),
        report_checks=checks, worker_seconds=worker_seconds, attempts=len(records),
        cell_index=cell_index.name,
        resource_censored_attempts=sum(m["status"] == "resource_censored" for _, m in records),
        remaining_prior_allocation_seconds=4513.601569557097-worker_seconds,
        cumulative_campaign_seconds=110076.1948335229+worker_seconds,
        default_readiness=False, full_paper_replication=False,
        inference_scope="Paired stochastic learning, conditional on two fresh fixed datasets per dimension; approximate bootstrap intervals.",
        statistical_report=str(report.relative_to(REPO)))
    labels = {"no_supported_ranking": "Unresolved",
        "repair_lower_MAE_conditional_99pct_interval": "Repaired lower MAE",
        "QR_lower_MAE_conditional_99pct_interval": "QR lower MAE",
        "incomplete_descriptive_only": "Incomplete; descriptive"}
    primary_table = table(["Dimension", "Pairs", "QR mean absolute log error",
        "Repaired mean absolute log error", "Repaired − QR", "99% paired interval", "Accuracy inference"],
        [[r["dimension"], r["paired_complete"], fmt(r["qr_MAE"]), fmt(r["repair_MAE"]),
          fmt(r["difference"]), f'[{fmt(r["lower99"])}, {fmt(r["upper99"])}]',
          labels[r["verdict"]]] for r in primary])
    cost_table = table(["Dimension", "Method", "Qualifying N", "Relative RMSE",
        "Simultaneous upper RMSE", "Mean total seconds"],
        [[r["dimension"], r["method"], r["N"] if r["status"] == "qualifying_measured_rung" else "None",
          fmt(r["relative_RMSE"]), fmt(r["upper_RMSE"]), fmt(r["mean_total_seconds"])] for r in costs])
    reliability_table = table(["Dimension", "Method", "Complete / planned", "Observed failures",
        "One-sided 95% upper failure probability"],
        [[r["dimension"], r["method"], f'{r["complete"]}/{r["planned"]}', r["failures"],
          fmt(r["failure_probability_upper95"])] for r in reliability])
    best_rows = sorted(candidate_vetoes, key=lambda r: float(r["mean_difference"]), reverse=True)[:8]
    heuristic_table = table(["Dimension", "Dataset", "Control", "Repaired mean absolute log error",
        "Control mean absolute log error"], [[r["dimension"], r["data_index"], r["comparator"],
        fmt(r["candidate_MAE"]), fmt(r["comparator_MAE"])] for r in best_rows])
    failure_text = ("No failure was observed among the recorded learners." if not failed else
        "Candidate failures remain in the comparison; a dimension with missing pairs cannot support confirmatory ranking.")
    fit_table = table(["Dimension", "Recorded repaired fits", "Positive ridge", "Active curvature bound",
        "Active-set fallback", "Maximum KKT residual", "Minimum weight ESS"],
        [[row["dimension"], row["recorded_fits"], row["positive_ridge_fits"],
          row["active_curvature_bound_fits"], row["active_set_fallback_fits"],
          fmt(row["max_KKT"]), fmt(row["min_weight_ESS"])] for row in fit_summary])
    text = f'''# Independent R iAPF validation: numerical completion and accuracy remain distinct

The campaign completed {summary['complete_cells']}/80 paired cells and
{completed}/160 scheduled full learners. {failure_text}
The repaired method has {len(candidate_vetoes)} observed conditional heuristic
underperformance records, which veto promotion. The dimension-specific paired
accuracy intervals below determine which QR/repaired differences are supported;
unresolved intervals are not evidence of equivalence. Full paper replication
remains open, because this weighted-log/ridge learner changes Equation 15 and
the original author implementation/settings have not been recovered.

This is the authorized independent CPU R reference. Both fitters, the shared
learning controller, numerical safeguards and floor convention were frozen at
the preceding repair manifest. No TensorFlow/GPU, LEDH, KDM, score, HMC or
production default was changed.

## What was tested

The first-study linear Gaussian model has T=100 and dimensions 5, 10, 20, 40
and 80. Two new datasets per dimension and eight paired learning repetitions
per dataset give 16 paired cases per dimension. The code uses the actual shared
controller and its fresh final estimate. Thirteen focused checks established
that timing instrumentation preserves the numerical outputs and RNG state;
exact full-Gaussian controls agree with Kalman under the unchanged 1e-7 gate.
All commands, seeds, frozen source identities and output hashes are preserved.

Intervals resample whole paired learning runs within each fixed dataset,
retaining the coupling between methods. The 99% intervals address the five
primary dimensional comparisons through the predeclared Bonferroni adjustment.
They are approximate bootstrap intervals conditional on these ten datasets;
two datasets per dimension do not establish general data-population performance.

## Numerical completion and paired terminal accuracy

{reliability_table}

These bounds show how much failure risk remains compatible with a small
validation sample. Completion is an engineering result; it is not a proof of
small error or rare-event reliability.

{primary_table}

The difference is mean absolute error in terminal log likelihood, repaired
minus QR. Negative values favor the repaired method. A confidence interval
overlapping zero leaves ranking unresolved. Prefix diagnostics were preserved
as explanatory quantities and were not substituted for the paper's terminal
likelihood target. Any candidate failures, caps or omissions remain visible
in the complete records and are not silently dropped to obtain a ranking.

{fit_table}

These retained fit diagnostics explain numerical behavior. The active-set
fallback solves the same frozen convex objective when the initial solver
fails its KKT check. A small KKT residual verifies the accepted optimizer's
first-order conditions; it does not establish that the fitted guide captures
the true future likelihood. Ridge and curvature-bound counts likewise do not
substitute for terminal accuracy.

## Heuristic comparisons and calibration

Every completed cell evaluates constant, observation-only, moment-diagonal,
precision-diagonal and full Gaussian guides. For comparison, both learned
guides are evaluated again at N=1000 using fresh common seeds. The two analytic
diagonal guides have privileged access to future Gaussian information; they
diagnose learning loss and do not establish a general-purpose alternative.
The exact full guide is an oracle check, not a stochastic competitor.

The full conditional table contains {len(vetoes)} observed underperformance
records across both learners. The following are up to eight largest observed
differences for the repaired learner; they are descriptive findings, not
statistical rankings of the controls.

{heuristic_table}

The accuracy/cost table also records mean likelihood ratio and the fraction
below .01. There are {len(calibration)} nominal calibration interval flags
across all methods/rungs. These screens do not establish estimator bias:
small samples can miss rare positive likelihood tails, and the calibration
intervals are not multiplicity-adjusted tests. In particular, a collapsed
bootstrap filter can have deceptively small observed variance. The report
never uses that variance alone as an efficiency claim.

## Cost at a stated precision threshold

After each learner stops, its guide is frozen and independent filters run at
N=250, 1000 and 4000. Total time includes the entire guide-learning cost, less
the original fresh-final filter, plus the measured new final-filter time.
Common data/model/Kalman setup is excluded. Runs are serial with one BLAS/OMP
thread and counterbalanced method order; timings remain machine-specific.

The target is relative likelihood RMSE <=.5. The listed rungs satisfy the
predeclared bootstrap upper-limit screen, with Bonferroni adjustment across
five dimensions, two learners and three rungs. The table reports the least
measured mean cost among qualifying tested rungs, without changing any default.
An absent rung means that this grid has not established the target. These are
costs at a shared precision ceiling, not evidence of identical MSE or an
optimal particle count. Bootstrap bounds do not protect against unseen tails.

{cost_table}

## Decision and limitations

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Record frozen-reference validation | {completed}/160 scheduled learners complete | See failure and oracle tables | Rare failures and new datasets | Preserve both frozen references and all failed/censored cells | General numerical correctness |
| Assess QR versus repaired accuracy by dimension | Predeclared paired 99% intervals above | Heuristic and validity vetoes remain binding | Small conditional sample and unseen tails | Use supported intervals only for the stated conditional comparison | Broad superiority or equivalence |
| Record measured cost at the precision ceiling | Independent fixed-guide grid and simultaneous upper RMSE screen | No qualifying rung means unresolved cost | Coarse grid, bootstrap coverage and machine load | Retain full cost/error curves; confirm any consequential cost claim independently | Optimal or exact matched-MSE efficiency |
| Keep original-paper replication open | Author identity and original objective differ | Replication gap remains | Unrecovered numerical choices | Keep the extension separate from Equation 15 | Full reproduction of the published implementation |

| Inference status | Finding |
|---|---|
| Hard veto screen | {len(failed)} recorded learner failures; oracle/check status retained in checks.csv |
| Statistically supported ranking | Only dimensions whose paired interval excludes zero, conditional on the tested datasets |
| Descriptive-only differences | Heuristic comparisons, individual errors, maxima, fit diagnostics and raw timings |
| Default readiness | No promotion; observed heuristic underperformance remains a veto |
| Next evidence needed | Resolve any new numerical failures from their saved cases; use independent data and adequate replication for wider accuracy/cost claims |

Post-run skeptical review: the strongest alternative explanation for stable
completion is that ridge regularization preserves an unweighted guide similar
to QR, rather than improving learning of future information. Exact and analytic
diagonal controls expose the remaining gap. The weakest evidence is the small
number of independent learned guides and fixed datasets, especially for rare
likelihood tails. New failures or a verified author implementation would
change the diagnosis. Numerical completion and conditional accuracy evidence
must not be promoted into full paper or production claims.

Numerical execution used {worker_seconds:.6f} aggregate worker seconds in
{len(records)} launches. The original deadline and 4,100-second phase cap were
preserved. Remaining prior allocation: {summary['remaining_prior_allocation_seconds']:.6f}
seconds; cumulative campaign use: {summary['cumulative_campaign_seconds']:.6f}/172800.
Per-attempt manifests distinguish successful process completion from candidate
success. The statistical tables are in `{report.relative_to(ROOT)}`.
The initial 120-second closure reserve left two cells unfinished. A bounded
scheduling repair reduced that reserve to 30 seconds and used fresh attempts,
without extending the original deadline or numerical budget. The original
index and report remain intact; `{cell_index.name}` identifies the selected
attempts. All timed-out work is included in the total cost of the campaign.
'''
    (ROOT / "result.md").write_text(text)
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    parent = json.loads((REPO / "docs/plans/artifacts/iapf-r-targeted-fitting-repair-20260922-01/manifest.json").read_text())
    outputs = [ROOT / "result.md", ROOT / "summary.json", ROOT / "cells.csv",
               ROOT / "execution-summary.json", ROOT / "fit-summary.csv"]
    if cell_index.name != "cells.csv":
        outputs.extend([cell_index, ROOT / "execution-summary-completion.json"])
    manifest = dict(status=summary["status"], git_commit=records[0][1]["git_commit"],
        git_branch=subprocess.check_output(["git", "branch", "--show-current"], cwd=REPO, text=True).strip(),
        dirty_worktree=True, plan=PLAN, result=str((ROOT / "result.md").relative_to(REPO)),
        assembled_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        CPU_only=True, GPU_intentionally_hidden=True, environment=records[0][1]["environment"],
        R_version=records[0][1]["R_version"],
        data_version="Snapshotted first-study linear Gaussian model; independent seeds in plan and every result row",
        worker_seconds=worker_seconds, cap_seconds=4100, launch_cap=90,
        remaining_prior_allocation_seconds=summary["remaining_prior_allocation_seconds"],
        deadline_utc="2026-09-21T20:04:26Z", deadline_renewed=False,
        frozen_reference_source_sha256={k: v for k, v in parent["final_source_sha256"].items()
            if Path(k).name in ("reference_iapf_paper.R", "reference_iapf_author_choices.R",
                               "reference_iapf_constrained_diagnostic.R")},
        attempts=[dict(path=str(p.relative_to(REPO)), sha256=sha(p), command=m["command"],
            status=m["status"], stage=m["stage"], wall_seconds=m["wall_seconds"]) for p, m in records],
        report_assembly=dict(command=["python", str(Path(__file__).relative_to(REPO))],
            source_sha256=sha(Path(__file__)), numeric_report=str(report.relative_to(REPO))),
        output_sha256={str(p.relative_to(REPO)): sha(p) for p in outputs},
        hash_verification="All executed sources, recorded inputs and outputs verified before assembly",
        summary=summary)
    with (ROOT / "manifest.json").open("x") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
