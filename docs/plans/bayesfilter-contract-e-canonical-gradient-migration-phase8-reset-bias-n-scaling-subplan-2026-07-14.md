# Phase 8 Repair Subplan: Reset-Bias Versus Finite-N Diagnostic

Date: 2026-07-14

Program ID: `contract-e-canonical-gradient-migration-20260713`

Continuation ID: `contract-e-canonical-gradient-migration-continuation-20260714-115526`

Status: `REVIEW_REQUIRED_BEFORE_EXECUTION`

## Objective

Determine whether the repaired lower-rung Kalman discrepancy is more consistent
with finite-particle error shared by the filter or with bias introduced by the
Contract E reset. Compare the frozen selected Contract E tuple with the
no-reset weighted LEDH baseline at `T=2` over a small predeclared increasing-`N`
ladder.

## Entry Conditions

- The valid `T=2,N=32` lower rung selected ridge
  `7.301568984985351e-09`, steps `20`, and chunks `16/16`.
- Same-program FD passed; Kalman differences are therefore not classified as FD
  wiring failure.
- The production identity factory remains empty, so every arm is diagnostic
  only.
- The original two-hour continuation budget and owner-selected `delta_grad=0.05`
  remain unchanged. Audit count `16` is not used here.

## Evidence Contract

Question: as particle count increases, do Contract E and no-reset value/gradient
errors move toward Kalman, and what part of their difference is attributable to
the active reset at this fixed data prefix and estimator seed?

Arms:

```text
particle counts N = 32, 64, 128
reset policies = all_active_contract_e, no_reset_weighted
T = 2, dataset seed = 81100, estimator seed = 80920
ridge/steps/chunks/epsilon/scaling frozen from the valid lower rung
```

Primary diagnostic: componentwise signed Contract-E-minus-Kalman and
no-reset-minus-Kalman value/HMC-gradient errors at each `N`, plus the paired
Contract-E-minus-no-reset difference. This is explanatory only.

Predeclared mechanism readout uses six quantities equally: relative value error
and the five componentwise relative HMC-gradient errors. For arm `a`, quantity
`k`, and adjacent counts `N_i<N_{i+1}`, define improvement as

```text
abs(error[a,k,N_{i+1}]) < abs(error[a,k,N_i]).
```

Define the paired reset-effect magnitude as

```text
R[k,N] = abs(metric[ContractE,k,N] - metric[NoReset,k,N]) / scale[k],
```

where `scale[value]=abs(Kalman value)` and
`scale[gradient k]=abs(Kalman HMC-gradient component k)`. All six Kalman scales
must be nonzero or the diagnostic stops for a revised scale.

The only predeclared descriptive classifications are:

- `shared_finite_N_pattern`: for all six quantities, both arms improve on both
  adjacent `N` edges and `R[k,N]` is nonincreasing on both edges;
- `reset_specific_pattern`: for all six quantities, the no-reset arm improves
  on both edges, while Contract E fails to improve on at least one edge, and
  `R[k,N]` is nondecreasing on at least one edge;
- `mixed_or_nonmonotone_inconclusive`: every other pattern, including any
  disagreement across components.

These labels are deterministic summaries of one-seed trajectories, not
statistical or causal conclusions.

Hard vetoes: source/prepared-input drift, nonfinite output, invalid chart,
branch/identity failure, timeout, dense allocation, or any changed numerical
setting other than the predeclared particle count/reset policy.

Interpretation:

- only the exact all-six-quantity rules above may emit
  `shared_finite_N_pattern` or `reset_specific_pattern`;
- any mixed or non-monotone pattern is automatically inconclusive and a
  multi-seed or larger-`N` plan is required;
- one seed cannot support a statistical ranking or equivalence claim.

## Required Artifacts And Checks

- exclusive result per arm and `N`;
- source/prepared-input hashes and CPU-hidden/XLA provenance;
- exact value, all physical/HMC gradients, signs, and telemetry;
- aggregate paired table and explicit descriptive-only classification;
- focused tests for reset policy, horizon, particle-count ladder, and no tuning;
- Python compilation, JSON parse, and scoped diff checks;
- bounded material review or documented reviewer unavailability; and
- a result note separating implementation, tuning, diagnostic, and scientific
  interpretations.

Review iteration 1: Claude returned `VERDICT: REVISE` because “moves similarly”
and “improves” were subjective for three counts and one seed. The rules above
replace that eyeballed interpretation with exact per-quantity monotonicity and
paired reset-effect definitions; mixed components are now automatically
inconclusive.

Attempt 1 repair: the no-reset comparator correctly leaves the canonical
`minimum_mass` diagnostic at `inf` because no Contract E reset executes. The
worker had incorrectly included that inactive-only sentinel in its all-output
finiteness veto. The repair excludes only `minimum_mass`/`minimum_mass_history`
from the executed-output finiteness scan and records the sentinel allowance
explicitly; all values, gradients, charts, and executed telemetry remain
finite requirements. The target, arms, settings, and budget are unchanged.

## Budget

At most six executed nodes, each capped at 300 seconds. Stop before launch if
the remaining two-hour continuation budget cannot cover the next node. No FD is
rerun because the same program at the repaired horizon already passed and this
diagnostic changes only prepared reset masks and `N`.

## Forbidden Claims And Actions

- Do not tune ridge, steps, chunks, `delta_grad`, or particle counts from output.
- Do not treat `N=128` as primary shape or scientific equivalence.
- Do not use audit count `16`, calibration/audit seeds, GPU, HMC, nonlinear
  migration, leaderboard regeneration, or factory admission.
- Do not rank the two stochastic arms from a single seed.

## Handoff And Stop Conditions

Write a diagnostic result after all six nodes or the first hard veto. A clear
descriptive mechanism signal may justify a separately reviewed repair or
increasing-`N`/multi-seed plan; it cannot promote Contract E. Stop on budget,
invalid artifact, or any need to change the frozen scientific contract.
