# Phase 7 Subplan: Fresh Stability And Target-Specific Protocol

Date: 2026-07-14

## Phase Objective

Rerun all four predeclared recipe smokes through the batch-binding-v2 route,
run one 100-step source-anchor stability job, and use its measured time and veto
status to decide whether the four-recipe 500-step screen fits a refreshed
campaign budget. Do not reuse stale row-mapped smoke artifacts.

## Entry Conditions Inherited From Phase 6

- Correctness and trusted GPU performance gates pass at `B=128`.
- Five compiled steps average `0.7431 s/step` including first compilation.
- Historical protocol smokes lack batch-native binding identity and are
  ineligible for continuation.
- 5,000-step final runs remain outside the current phase budget until refreshed
  from measured screen/long-run evidence.

## Required Artifacts

- Fresh versioned root under
  `docs/plans/artifacts/neutra-batch-native-training-2026-07-14/phase7/`.
- Four five-step recipe smoke results with binding-v2 provenance.
- One 100-step source-anchor stability result.
- A screen budget decision based on 100-step wall/program time.
- If admitted, four fresh 500-step screen jobs plus a selection result whose
  role remains proxy nomination only.
- Phase 7 result and reviewed Phase 8 institutionalization/closeout subplan.

## Execution Ladder And Budget

1. Run four five-step smokes sequentially: source anchor, lower learning rate,
   shallow two-stage, and wide two-times capacity.
2. Continue if every smoke has finite losses/gradients, valid exact-target
   status, binding schema v2, one compiled invocation, and GPU placement.
3. Run one 100-step source-anchor stability job using `step_override=100` under
   the screen recipe budget. Maximum authorized wall time: 10 minutes.
4. Continue to the four-recipe 500-step screen only if the 100-step run passes
   all hard vetoes and projects total four-arm screen live time at no more than
   30 minutes. The screen itself has a 35-minute aggregate live-time stop.
5. Do not launch 5,000-step final jobs in this phase. Write a refreshed budget
   and handoff after screen interpretation.

## Required Checks

- Fresh output roots; no overwrite or stale result resolution.
- All training/Adam/output devices GPU; XLA true; batch 128; no fallback.
- Binding schema v2 and exact dependency closure recorded.
- Every recorded status valid, no floors/nonfinite values.
- Loss and gradient histories finite; descriptive only.
- Heldout screen batches use common predeclared seeds and remain nomination
  metrics, not downstream promotion criteria.
- Record command, Git/environment/device, seeds, wall/program time, artifacts,
  and hashes.
- Preserve failed candidates; candidate rejection is not research-direction
  rejection unless a continuation veto fires.

## Evidence Contract

| Item | Phase contract |
| --- | --- |
| Question | Is the certified batch-native training stable enough for target-specific recipe screening, and which recipes remain viable? |
| Baseline | Four predeclared recipes under identical target, batch, heldout seeds, optimizer family, and budget. |
| Smoke pass | Engineering mechanics/status only. |
| Stability pass | 100 finite/valid steps with no resource or state failure; loss descriptive. |
| Screen nomination | Lowest common-heldout mean with the predeclared source-anchor paired-MCSE preference; proxy nomination only. |
| Hard veto | Invalid target status, nonfinite loss/gradient, wrong device/XLA, missing binding v2, artifact corruption, or budget exhaustion. |
| Continuation veto | Harness/target invalidity, broken assumptions, missing required diagnostics, or exhausted budget. A recipe failure alone is not a continuation veto. |
| Nonclaims | Smokes, stability, and heldout reverse-KL screen do not establish posterior correctness, HMC convergence, sampler superiority, robustness, generalization, or default-readiness. |

## Statistical Discipline

All continuous losses, gradient norms, timings, and heldout values are
descriptive unless the predeclared paired-MCSE rule applies to source-anchor
preference. The four-recipe screen nominates one recipe for fresh long-budget
validation; it does not establish superiority. Viable recipes are otherwise
statistically indistinguishable under current evidence.

## Default And Assumption Audit

| Choice | Provenance | Failure mode | Early diagnostic | Status |
| --- | --- | --- | --- | --- |
| Four existing recipes | target-specific protocol | inherited search may be too narrow | all are hypotheses; screen cannot establish global optimum | predeclared hypothesis set |
| Source-anchor 100-step stability | prior protocol anchor | another recipe may be more stable | all four get smokes; stability is route check, not selection | diagnostic baseline |
| 500 steps | existing screen | may be too short/noisy for transport quality | heldout role explicitly nomination only | proxy budget |
| Batch 128 | fair continuation and Phase 6 decision row | not throughput-optimal | no batch-size promotion from Phase 6 | frozen campaign baseline |

## Skeptical Subplan Audit

- Stale evidence: old row-mapped smokes are explicitly excluded.
- Proxy promotion: heldout reverse KL only nominates; downstream validation is
  still required after long training.
- Under-budgeting: 100-step measured time gates the 500-step screen budget.
- Candidate versus direction: a failed recipe is rejected; later recipes and
  repair continue unless target/harness validity fails.
- Unfair comparison: common target, batch, seeds, heldout batches, and budget;
  recipe differences are exactly the predeclared architecture/LR hypotheses.

Audit verdict: **PASS**. The ladder is now practical under the measured rate,
preserves scientific evidence roles, and stops before unbudgeted final training.

## Forbidden Claims And Actions

- Do not reuse old smoke/screen artifacts or mix result roots.
- Do not select from five-step or 100-step loss.
- Do not call the 500-step nominee the best or default.
- Do not launch 5,000-step training or downstream HMC without the refreshed
  handoff budget and required downstream plan.

## Exact Next-Phase Handoff Conditions

Phase 8 starts after Phase 7 writes the viability/nomination result, a refreshed
long-training budget or blocker, and a subplan covering reusable API promotion,
template/policy tests, stale-route demotion, documentation, and terminal
engineering/numerical/scientific ledgers.

## Stop Conditions

Stop for a true continuation veto or aggregate budget exhaustion. Repair and
retry localized infrastructure/artifact failures inside the same target,
method, device class, and budget.

## Phase-End Procedure

1. Run required local checks.
2. Write Phase 7 result/close record.
3. Draft or refresh Phase 8 subplan.
4. Review Phase 8 suitability and continue when no real blocker exists.

