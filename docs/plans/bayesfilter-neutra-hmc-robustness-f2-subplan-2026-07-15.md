# NeuTra HMC Robustness Phase F2 Subplan

Date: 2026-07-15  
Status: `EXECUTED_AND_CLOSED`

## Objective And Entry Conditions

Tune, admit, and independently confirm the F1 frozen NeuTra candidate on the
new fixture through the canonical shared controller and compare it with the F0
plain-HMC comparator. F0 and F1 passed, and the exact target/payload/comparator
identities are frozen.

## Evidence Contract

Short fixed-budget HMC probes may nominate one fixed kernel by healthy
acceptance in `0.60-0.90`, closest to `0.75`; acceptance cannot promote it.
Admission requires retained warm-up, recent-window modern R-hat `<=1.05`, and
cumulative retained modern R-hat `<=1.01`, each capped at 10,000 per chain.

Independent confirmation uses fresh seeds, at least 2,000 warm-up samples per
chain, and at least 4,000 retained samples per chain. Promotion requires every
parameter to pass modern R-hat `<=1.01`, bulk ESS `>=1000`, tail ESS `>=400`,
finite/status/movement/energy-error health, plain-HMC mean agreement within four
combined MCSE, and truth recovery within three posterior SD.

Warm-up must be archived but excluded. The F0 comparator is valid only for
target signature `312d...d283`; any mismatch vetoes the run. No descriptive
runtime, acceptance, loss, ESS, or tail difference may become a superiority
claim. A pass supports one candidate on one additional fixture. A failure
rejects this candidate/fixture arm, not NeuTra universally.

## Defaults, Pre-Mortem, And Artifacts

The initial probe points and step grid are mechanics choices inherited from the
same-dimensional S1 arm, not promoted defaults. Start with
`(0.025,0.05,0.1,0.2,0.4,0.8)` and permit one fresh bracket repair only if
healthy probes fail to resolve the acceptance band. A health-vetoed kernel is
rejected individually; it does not invalidate distinct healthy configurations.

The run could pass misleadingly through comparator mismatch, recycled draws,
warm-up leakage, proxy promotion, or artifact overwrite. Exact hashes, disjoint
seed roots, fresh output directories, private/public tensor separation, and
posterior comparison checks veto those cases.

Required artifacts: payload/comparator identity ledger, probe/admission result,
separate warm-up/retained archives, independent confirmation, convergence and
posterior tables, command/environment/seed/wall-time manifest, F2 result, and
terminal Phase A audit subplan.

## Budget, Stop, And Handoff

CPU-hidden/XLA F2 budget is four hours with one fresh-directory retry only for
localized infrastructure or bracket-resolution failure. Warm-up and retained
caps are 10,000 per chain. Stop for target/payload/comparator mismatch,
health/convergence/recovery failure at cap, corrupted evidence, or budget
exhaustion. Whether F2 passes or validly fails, hand off to Phase A to audit
scope, claims, omissions, and drift.

Skeptical audit verdict: `PASS`. Downstream HMC is the primary criterion,
acceptance/loss remain proxy diagnostics, seeds and caps are explicit, and a
candidate failure cannot be silently upgraded into a universal research veto.
