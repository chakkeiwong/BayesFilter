# Claude Read-Only Review: SIR Remaining-Gap Closure Plan, Iteration 1

Date: 2026-07-16

Reviewed path:
`docs/plans/bayesfilter-sir-remaining-gap-closure-master-plan-2026-07-16.md`

Role: Claude Code read-only scientific reviewer; Codex supervisor/executor.

## Verdict

`VERDICT: REVISE`

## Material Findings

1. The plan could detect a statistically supported difference or remain
   inconclusive, but it had no predeclared positive equivalence criterion. Any
   later practical-accuracy admission would therefore be post hoc.
2. The Phase 2 requirements that bias "does not worsen with N" and interval
   width is "small enough" were noisy or undefined rather than self-contained
   statistical gates.

## Repairs Applied

1. The scientific question is narrowed to mismatch detection versus
   inconclusive evidence. No interval-includes-zero result can emit a positive
   practical-equivalence, HMC, or leaderboard accuracy token.
2. Phase 2 now freezes numeric dense-reference uncertainty from the predecessor
   result and requires simultaneous interval half-widths no larger than those
   observed refinement differences, so Monte Carlo precision must be at least
   as fine as reference numerical uncertainty.
3. Particle-count trends are explanatory only. Failure to reach the numeric
   precision gate within budget emits
   `BLOCK_TEACHER_PRECISION_UNDER_BUDGET`.

## Iteration 2 Finding And Repair

Claude agreed the first two blockers were resolved, then found that the plan
overstated what a teacher certified only at `J=1` can prove at `J=2` and `d=18`.
The later phases now support only LEDH--teacher disagreement or inconclusive
comparison. A disagreement is a conservative LEDH promotion veto but does not
identify which method is closer to the latent filtering target.

## Iteration 3 Finding And Repair

Claude found that Phase 3 used an undefined teacher-stability condition and that
claim-bearing comparisons preceded GPU/identity route binding. The plan now:

- defines teacher refinement as a Bonferroni-adjusted paired `N=128` versus
  `N=256` no-detected-shift diagnostic, with no convergence claim; and
- executes exact-source GPU/XLA and canonical identity before `J=2` and `d=18`
  comparison phases.

## Iteration 4 Verdict

`VERDICT: AGREE`

Claude found no remaining material mathematical, statistical, feasibility,
default, source-boundary, overclaim, or phase-ordering defect. The review
specifically accepted the paired `N=128` versus `N=256` no-detected-shift screen
as adequate for the plan's deliberately limited later-rung conclusions and
confirmed GPU/identity precede claim-bearing comparisons.
