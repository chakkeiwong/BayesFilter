"""Post-run diagnostic reporting only; never imported by the experiment runtime."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "docs/plans/artifacts"
OUT = BASE / "surrogate-hmc-fresh-scope-advance-20261012-01"
LEDGER = BASE / "surrogate-hmc-repair-continuation-20261012-01/budget.json"
CURRENT = {
    "lgssm": "surrogate-hmc-guard-advance-lgssm-20261013-01",
    "ksc": "surrogate-hmc-guard-advance-ksc-20261013-01",
    "predator_prey": "surrogate-hmc-guard-advance-predator-20261013-01",
    "sir_d18": "surrogate-hmc-guard-advance-sir-20261013-01",
    "range_bearing": "surrogate-hmc-guard-advance-range-20261013-01",
}
LABELS = dict(lgssm="LGSSM", ksc="KSC SV", predator_prey="Predator-prey",
              sir_d18="SIR", range_bearing="Range bearing")

def read(path):
    return json.loads(path.read_text())

def relative(path):
    return str(path.relative_to(ROOT))

def changed_sources(scope):
    return [name for name, digest in scope.get("source_sha256", {}).items()
            if not (ROOT/name).is_file()
            or hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != digest]

def independent_comparison(selection):
    references = selection["independent_reference_diagnostics"]
    rows = []
    for check in selection["selected"]["precision_checks"]:
        for index, (candidate, reference) in enumerate(zip(
                check.get("candidate_values_and_scores", []),
                check.get("independent_reference_value_and_score", []))):
            assert len(candidate) == len(reference)
            differences = [x-y for x,y in zip(candidate, reference)]
            scaled = [abs(d)/max(1.,abs(y)) for d,y in zip(differences, reference)]
            rows.append(dict(seed=check["validation_seed"], row=index,
                candidate_value_and_score=candidate, reference_value_and_score=reference,
                signed_differences=differences, absolute_scaled_differences=scaled))
    if not rows:
        return dict(status="NOT_AVAILABLE", reference=references.get("reference"),
                    role="explanatory; independent accuracy not established")
    return dict(status="LOCAL_REFERENCE_COMPARISON", reference=references["reference"],
        rows=rows, max_absolute_loglik_difference=max(abs(r["signed_differences"][0]) for r in rows),
        max_scaled_loglik_difference=max(r["absolute_scaled_differences"][0] for r in rows),
        max_scaled_score_difference=max(v for r in rows for v in r["absolute_scaled_differences"][1:]),
        scale="max(1,abs(independent reference component))",
        role="explanatory local approximation accuracy; not a posterior criterion")

def latest_attempt(model, root_name):
    if root_name is None:
        return None
    root = BASE/root_name
    if not (root/"advance.json").is_file():
        return dict(status="RUNNING_OR_NOT_RECORDED", completed_candidates=[])
    advance = read(root/"advance.json")
    model_row = next((r for r in advance.get("models", []) if r["model"] == model), {})
    status = model_row.get("status", advance["status"])
    if model_row.get("filter-tuning", {}).get("status") == "TIMEOUT":
        status = "TUNING_TIME_LIMIT_REACHED"
    candidate_paths = sorted((root/model/"filter-tuning"/model).glob("filter-candidate-*.json"))
    candidates = []
    for path in candidate_paths:
        row = read(path)
        errors = [v for check in row.get("precision_checks", [])
                  for values in check["scaled_errors"] for v in values]
        candidates.append(dict(path=relative(path), index=row["index"],
            valid=row["valid"], precision_pass=row["precision_pass"],
            max_scaled_arithmetic_error=max(errors) if errors else None))
    return dict(status=status, original_status=advance["status"],
        completed_candidates=candidates, artifact=relative(root/"advance.json"))

def main():
    ledger = read(LEDGER)
    active = [a for a in ledger["attempts"] if a.get("status") == "RUNNING"]
    rows = []
    for model, root_name in CURRENT.items():
        roots = ([BASE/root_name] if root_name else []) + [OUT]
        root = next((r for r in roots if (r/model/"mechanics"/model/"pilot.json").is_file()), None)
        if root is None:
            rows.append(dict(model=model, status="NO_COMPLETED_PILOT"))
            continue
        pilot_path = root/model/"mechanics"/model/"pilot.json"
        selection_path = root/model/"filter-tuning"/model/"filter-selection.json"
        pilot, selection = read(pilot_path), read(selection_path)
        selected = selection["selected"]
        errors = [row for check in selected["precision_checks"] for row in check["scaled_errors"]]
        source_changes = changed_sources(pilot["scope"])
        result_path = pilot_path.parent/"result.json"
        result = read(result_path) if result_path.is_file() else {}
        rows.append(dict(
            model=model, label=LABELS[model], pilot_status=pilot["status"],
            parent_result_status=result.get("status", "MISSING"),
            pilot_path=relative(pilot_path), selection_path=relative(selection_path),
            source_matches_current=not source_changes,
            changed_source_paths=source_changes,
            selection_pilot_scope_equal=selection["scope"] == pilot["scope"],
            scope=pilot["scope"], selected_index=selected["index"],
            controls=selected["controls"], beta=selected["beta"],
            arithmetic_precision_pass=selected["precision_pass"],
            precision_max_scaled_value_error=max(row[0] for row in errors),
            precision_max_scaled_score_error=max(abs(x) for row in errors for x in row[1:]),
            precision_scale="max(1,abs(FP64 component)); same finite program, not a model oracle",
            independent_reference_diagnostics=selection["independent_reference_diagnostics"],
            local_reference_comparison=independent_comparison(selection),
            latest_attempt_interpretation=latest_attempt(model, root_name),
            current_attempt_path=relative(BASE/root_name/"advance.json") if root_name else None,
            current_attempt=read(BASE/root_name/"advance.json") if root_name and (BASE/root_name/"advance.json").is_file() else None,
            endpoint=pilot["endpoint"], steps_per_chain=pilot["steps_per_chain"],
            acceptance_per_chain=pilot["acceptance_per_chain"],
            finite=pilot["finite"], divergences=pilot["divergences"],
            force_fallback_count=pilot["force_fallback_count"],
            cold_run_health=pilot["cold_run_health"], warmed_run_health=pilot["warmed_run_health"],
            max_abs_delta_h=pilot["max_abs_delta_h"],
            seconds_per_four_chain_transition=pilot["seconds_per_four_chain_transition"],
            illustrative_2000_transition_hours=pilot["minimum_2000_warmup_hours_at_measured_rate"],
            timing_limitation=pilot["timing_limitation"],
            posterior_sampling_complete=pilot["posterior_sampling_complete"],
        ))
    summary = dict(
        schema="surrogate_hmc_repair_diagnostic_summary_v1",
        generated_utc=datetime.now(timezone.utc).isoformat(),
        status="IN_PROGRESS" if active else "BOUNDED_COMPUTE_STOP_PARTIAL_REPAIR_POSTERIOR_INCOMPLETE",
        next_phase_plan=read(OUT/"next-phase-plan.json") if (OUT/"next-phase-plan.json").is_file() else None,
        question="Repair the shared five-model surrogate-HMC implementation and test the resulting consumer.",
        plan="docs/plans/surrogate-hmc-five-model-performance-repair-master-2026-10-11.md",
        models=rows, ledger_path=relative(LEDGER),
        budget={k: ledger[k] for k in ["authorized_gpu_wall_seconds","used_seconds","remaining_seconds","max_attempts"]},
        attempts=len(ledger["attempts"]), active_attempts=active,
        default=dict(dtype="float32",tf32=False,particles=4032),
        canonical_ledh_admission=False, full_posterior_complete=False,
        statistically_supported_ranking=None,
    )
    (OUT/"repair-evidence-summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    lines = [
        "# Shared surrogate-HMC repair: consolidated evidence",
        "",
        "The shared analytical score and sampler repairs have executable parity evidence. "
        "Full posterior sampling is incomplete. Short mechanics runs do not establish convergence, "
        "model-score accuracy, or economical inference. FP32 remains the default; SIR and range "
        "bearing currently need an optional FP64 recursion with FP32 pair kernels for the checked scopes.",
        "",
        "This report preserves the exact source snapshot for every row. An earlier-snapshot result "
        "cannot issue a fresh tuning authority, even when a later repair has separate parity evidence.",
        "",
        "| Model | Scope | Snapshot | Arithmetic score discrepancy | Mechanics | Warm seconds/transition | Acceptance |",
        "|---|---|---|---:|---|---:|---|",
    ]
    for r in rows:
        if "pilot_status" not in r:
            lines.append("| "+LABELS[r["model"]]+" | Unavailable | — | — | No completed pilot | — | — |")
            continue
        scope = r["scope"]
        snapshot = "Current source" if r["source_matches_current"] else "Earlier source"
        acceptance = ", ".join(format(v,".3g") for v in r["acceptance_per_chain"])
        lines.append("| %s | T%d, state %d, parameter %d | %s | %.6g%% | %s | %.3f | %s |" % (
            r["label"], scope["horizon"], scope["state_dimension"], scope["parameter_dimension"],
            snapshot,100*r["precision_max_scaled_score_error"],r["pilot_status"],
            r["seconds_per_four_chain_transition"],acceptance))
    lines += [
        "",
        "| Model | Independent reference | Max absolute log-likelihood difference | Max scaled score difference |",
        "|---|---|---:|---:|",
    ]
    for r in rows:
        comparison = r.get("local_reference_comparison", {})
        if comparison.get("status") == "LOCAL_REFERENCE_COMPARISON":
            lines.append("| %s | %s | %.6g | %.4g%% |" % (
                r["label"], comparison["reference"], comparison["max_absolute_loglik_difference"],
                100*comparison["max_scaled_score_difference"]))
        else:
            lines.append("| "+LABELS[r["model"]]+" | Not established in this local selection | — | — |")
    lines += [
        "",
        "These reference comparisons use two independent validation datasets and two local parameter "
        "rows per dataset. They measure local approximation differences, with no statistical ranking "
        "or posterior bias conclusion. Near-zero score components use an absolute unit scale.",
        "",
        "Latest current-source attempts:",
        "",
    ]
    for r in rows:
        attempt = r.get("latest_attempt_interpretation")
        lines.append("- "+LABELS[r["model"]]+": "+(
            attempt.get("status", "UNKNOWN") if attempt else "RUNNING_OR_NOT_RECORDED")+
            "; "+str(r.get("current_attempt_path")))
        if attempt and attempt["status"] == "TUNING_TIME_LIMIT_REACHED":
            for candidate in attempt["completed_candidates"]:
                error = candidate["max_scaled_arithmetic_error"]
                lines.append("  - Saved candidate%d: valid=%s, precision_pass=%s, "
                             "max scaled arithmetic discrepancy %s. No final selection." % (
                    candidate["index"], candidate["valid"], candidate["precision_pass"],
                    "%.6g%%" % (100*error) if error is not None else "not available"))
    lines += [
        "",
        "Arithmetic discrepancies use max(1, absolute FP64 component) as the denominator and "
        "compare the same finite filter. They are not discrepancies from a model oracle. "
        "Timing covers a warmed transition of all four chains at L=1, epsilon=0.002. "
        "Acceptance uses only two proposals per chain; these are descriptive observations.",
        "",
        "The SIR pilot has one chain with no accepted proposal in its two-step warmed run. "
        "It passes the limited numerical mechanics criterion, but does not pass a claim of "
        "established movement or convergence. R-hat and posterior ESS are not available.",
        "",
        "| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |",
        "|---|---|---|---|---|---|",
        "| Retain verified engineering repairs | Paired value/score/state checks and focused tests | See linked per-repair results | Limited fixtures and local parameter coverage | Continue the master only within available compute | Statistical speed superiority |",
        "| Keep viable finite-filter candidates exploratory | Own-scope precision and short numerical health | Rejected precision candidates remain rejected | Force accuracy against the actual MH endpoint and posterior region | Improve cost and independent reference coverage before long chains | Full posterior inference or canonical LEDH admission |",
        "| Keep full posterior phase open | Adaptive warmup and cumulative sampling not completed | Required convergence/reference evidence absent | Current measured cost and explicit prior/target choice | Resume only under an adequate reviewed budget and target scope | Five-model HMC campaign completed |",
        "",
        "| Inference status | Finding |",
        "|---|---|",
        "| Hard veto screen | See each numerical mechanics status; source mismatches prohibit reuse as current tuning authority |",
        "| Statistically supported ranking | None |",
        "| Descriptive differences | Timing, short acceptance, energy errors and three-pair compact-covariance timings |",
        "| Default readiness | FP32 stays default; mixed precision remains an optional checked scope; no inference/default promotion |",
        "| Next evidence needed | Practical cost, justified endpoint/prior, target-specific tuning, adaptive warmup, cumulative R-hat/ESS and posterior reference agreement |",
        "",
        "The LGSSM MH endpoint is exact Kalman under the benchmark prior. KSC uses a fixed "
        "1201-node grid whose refinement must be checked over the posterior region. Predator-prey "
        "and SIR use fixed bootstrap likelihood approximations; range bearing uses a fixed "
        "trajectory-importance approximation. These endpoint limitations are separate from "
        "the precision of the proposal force.",
        "",
        "At the measured L=1 rate, even 2000 transitions per chain cost many hours. "
        "This is an illustrative projection before retained sampling, not a tuned-HMC forecast. "
        "Tuning may change step size, trajectory length and efficiency.",
        "",
        "Post-run red team: local precision and a healthy two-step chain can coexist with "
        "a poor force, biased approximate endpoint, inadequate parameter coverage, or an "
        "unaffordable long run. Independent posterior comparisons and longer cumulative "
        "diagnostics could overturn the present viability judgment. Heuristic/practical "
        "dominance has not been established, so practical-method promotion is unsupported.",
        "",
        "Budget charged: %.3f / %.3f seconds; ledger remainder %.3f seconds. "
        "Running reservations: %d. The ledger remains authoritative." % (
            ledger["used_seconds"],ledger["authorized_gpu_wall_seconds"],
            ledger["remaining_seconds"],len(active)),
        "",
        "Evidence paths:",
        "",
    ]
    for r in rows:
        if "pilot_path" in r:
            lines.append("- "+r["label"]+": "+r["pilot_path"]+"; "+r["selection_path"])
    lines += [
        "- Engineering results in this directory: compact-covariance-result.md, "
        "force-reuse-result.md, precision-guard-result.md and component-cost-result.md.",
        "- Full structured values and source differences: repair-evidence-summary.json.",
        "",
    ]
    (OUT/"repair-evidence-summary.md").write_text("\n".join(lines))
    print(json.dumps(dict(status=summary["status"],model_count=len(rows),
                          source_current=[r["model"] for r in rows if r.get("source_matches_current")],
                          active=len(active),output=str(OUT/"repair-evidence-summary.md"))))

if __name__ == "__main__":
    main()
