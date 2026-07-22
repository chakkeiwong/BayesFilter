# LGSSM NeuTra Gap Closure Phase 5 - Post-Tuning Repair Proposal

> **Superseded 2026-07-15:** The apparent repair choice arose from the invalid
> fixed 1,000-draw admission design. The active repair is
> `docs/plans/bayesfilter-lgssm-neutra-sequential-hmc-repair-plan-2026-07-15.md`;
> it keeps the already selected kernel fixed and executes the missing
> sequential warm-up/retained policy.

Date: 2026-07-15  
Status: `BLOCKED_REQUIRES_NEW_SCIENTIFIC_CONTRACT`  
Parent: `docs/plans/bayesfilter-lgssm-neutra-scientific-gap-closure-plan-2026-07-15.md`

## Objective

Design a new bounded experiment that distinguishes HMC-family inadequacy from
transport-quality inadequacy after both current frozen candidates failed the
modern R-hat admission gate.

This is a proposal only. It is not part of the executed Phase 4 contract and
must not be launched without a refreshed plan and explicit continuation.

## Entry Conditions

- Phase 4 result is `BLOCK_PHASE4_NO_ADMITTED_HMC_CANDIDATE`.
- Seed1201 failed max folded R-hat `1.01569` at `a22_raw`.
- Seed1202 failed max rank R-hat `1.01721` at `a42_raw`.
- Both had acceptance near `0.70`, finite samples, valid status telemetry, and
  zero energy-error divergence screens.
- The prior step-size repair grid is exhausted.

## Candidate Repair Questions

The next experiment must choose and predeclare one of these bounded mechanisms:

1. increase verification length while keeping the selected step size and
   leapfrog count fixed, testing whether the two R-hat failures are finite-chain
   uncertainty rather than persistent mixing;
2. run a small, predeclared leapfrog-count comparison at the already viable
   step-size region, testing whether trajectory length is the bottleneck; or
3. train/evaluate a new transport recipe or objective, which is a material
   training-direction change and needs a new training budget.

The present evidence cannot select among these explanations. Do not silently
choose one from the observed R-hat values.

## Required Artifacts And Checks

Any approved repair must specify fresh seeds, total compute budget, exact
baseline, the unchanged R-hat/ESS/health gates, a no-weakening rule, and a
terminal result with a decision table. It must preserve current Phase 1-4
artifacts and use a new versioned output root. If a longer verification is
chosen, it must report uncertainty/MCSE or another declared justification; if
leapfrog count is changed, it must freeze a new kernel before any confirmation;
if training changes, it must repeat target-specific training evidence.

## Evidence Contract

The repair may establish only that a declared repair candidate passes the same
modern rank/folded R-hat admission gate. It must not weaken `1.01`, substitute
acceptance, or treat a short-chain improvement as posterior correctness.

## Forbidden Claims And Actions

- do not run Phase 5/confirmatory HMC from the rejected Phase 4 records;
- do not promote either current kernel because acceptance was near `0.70`;
- do not widen grids, change leapfrog count, extend chains, or retrain without
  a new reviewed contract;
- do not overwrite Phase 4 artifacts or relabel the candidates as admitted;
- do not claim the NeuTra research direction failed.

## Exact Handoff Conditions

The parent plan must be refreshed with the selected repair mechanism, its
baseline and assumption audit, compute/attempt budget, seeds, evidence
contract, and stop conditions. Only then may a new Phase 5 execution subplan
be marked ready. If no repair is approved, close the campaign as a negative
candidate result with the current nonclaims.

## Stop Conditions

Stop for no human direction on the materially different repair choice, any
proposal that weakens the modern R-hat gate, exhausted compute budget, or a
repair result with nonfinite/status/identity failure.

## Suitability Review

This subplan preserves the scientific question and current evidence, names the
three materially different explanations, and refuses to smuggle in an
unreviewed extra experiment after a declared continuation veto. It is suitable
as a handoff proposal but intentionally not executable yet.
