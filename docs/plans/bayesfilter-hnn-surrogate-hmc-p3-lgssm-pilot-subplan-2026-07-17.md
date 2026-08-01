# P3 Subplan: LGSSM Exact-Kalman Pilot

Phase objective: determine on the best-understood exact target whether the
corrected HNN kernel is valid and sufficiently viable to justify nonlinear
model runs.

Entry conditions: P2 harness passes; exact-Kalman target/chart identities replay;
pilot command, seeds, training/tuning grids, and 12-GPU-hour ceiling are frozen.

Refreshed entry evidence: P2 passes 13 CPU/reference and three trusted GPU/XLA
checks. The registered target/chart identities replay. Preserved target-matched
chart archives contain 2,000 warm-up and 4,000 retained draws per chain. P3
uses disjoint archived coordinate slices for training and heldout supervision,
but fresh seeds for HNN tuning, warm-up, and retained sampling.

Required artifacts:

- tuned raw-coordinate plain HMC, zero-residual, learned residual, and
  true-gradient NeuTra-coordinate arms; matching preserved raw-HMC evidence may
  be reused after target and diagnostic replay;
- target-specific training screen and fresh selected force;
- disjoint tuning verification, retained warm-up, retained samples, tuned
  plain-HMC posterior/truth comparison, full trace, and cost ledger;
- phase decision separating validity from performance.

Required checks/tests/reviews:

- exact Kalman likelihood target replay, preserved tuned plain-HMC posterior
  moment/interval comparison, and generating-truth tail rule; the posterior is
  not falsely described as closed-form analytic;
- modern rank/folded R-hat, bulk/tail ESS, divergence/energy/status vetoes;
- full endpoint kinetic correction and target call-count verification;
- matched hardware/chart/mass/retained policy across arms;
- from-scratch, reuse-scenario, and sampling-only cost tables;
- apply the master program's descriptive performance criterion and issue
  `DESCRIPTIVE_PERFORMANCE_SCREEN_PASS` or
  `PERFORMANCE_NOT_DEMONSTRATED` separately from validity.
- prove that the separate value-only endpoint equals the complete transformed
  value/score target before timing or sampling; a wrapper that computes an
  unused true gradient is a P3 performance-boundary failure.

Evidence contract: a pass establishes the corrected kernel on one exact-Kalman
LGSSM fixture. It can unlock later cells but cannot prove nonlinear validity.

Forbidden claims/actions: no speed superiority from descriptive one-seed
metrics; no reuse of a learned LGSSM force on later targets; no dismissal of a
valid but slow method as mathematically wrong.

Exact P4 handoff: learned or zero-residual corrected arm passes validity and
sampler gates, shared harness remains valid, and remaining GPU budget is
sufficient. If learned force is rejected but the zero-residual arm passes,
repair training within two attempts before deciding whether HNN learning is
blocked; record map validity separately.

Stop conditions: shared kernel invalidity, exact-target disagreement, severe
truth-tail failure, irreparable target/chart mismatch, or exhausted P3 budget.
Performance-not-demonstrated alone does not veto later scientific validity
tests if budget remains. It must be reported directly and cannot be described
as a speedup, but P3 validity can still justify testing generality later.

Phase-end duties: run checks; write P3 result; refresh P4 from measured costs;
review P4; continue if no real blocker.

Skeptical audit, refreshed 2026-07-17: passed after repairing P2's endpoint
boundary and correcting the analytic-posterior wording. The prior NeuTra kernel
`(step_size=0.8, leapfrog_steps=10)` is a warm-start hypothesis only. P3 must
screen a bounded neighborhood and select on target health, modern R-hat, and
energy diagnostics before acceptance tie-breaking.
