# P4 R4 Subplan: Predator-Prey NeuTra Confirmation

Date: 2026-07-16

Program ID: `multimodel-neutra-filter-posterior-20260715`

Status: `READY_FOR_R4_EXECUTION`

## Phase Objective And Entry Conditions

Confirm or reject the fresh target-specific plain dense-IAF transports for
`PP-UKF` and `PP-SGQF` by tuning and running HMC in each transport's latent
coordinates. Each cell must remain bound to its own admitted target, frozen
transport, and admitted same-target plain-HMC comparator.

Entry requires, independently per cell:

- exact typed target signatures
  `036948f0faaf028d159d7b70337214f01514d732112c2d10e9f7eea1e13b8e30`
  for `PP-UKF` and
  `8e0a9582fd30643b2e77e7615a21c0d44cc6c1827865ea52c841cc6dbfdde1ad`
  for `PP-SGQF`;
- an admitted 4-chain plain-HMC comparator with separate warm-up and retained
  archives and recursive hashes;
- a fresh selected 5,000-step GPU/XLA training result with valid target status,
  frozen/trainable value-score parity, and recursive hashes; and
- no reuse of screen weights or HMC seeds in the confirmation run.

`PP-ZC` remains `TARGET_BLOCKED_SOURCE_ROUTE_MISMATCH` and is outside R4.

## Research Intent And Evidence Contract

| Field | Frozen R4 contract |
| --- | --- |
| Question | Does each frozen target-specific dense-IAF transport support valid HMC samples whose six physical posterior means are practically equivalent to its own admitted same-target plain-HMC comparator? |
| Baseline | The cell's admitted tuned plain-HMC retained archive, not another filter or an exact-model posterior |
| Candidate | HMC on the exact target pulled back through the cell's fresh frozen dense-IAF transport |
| Kernel nomination | Among health-valid finite probes, maximize minimum rank-normalized bulk ESS; grid order breaks ties |
| Sampler pass | Recent-window warm-up modern R-hat `<=1.05`; cumulative retained modern R-hat `<=1.01`; minimum bulk ESS `>=1000`; minimum tail ESS `>=400`; health and target status valid |
| Agreement pass | For all six physical posterior means simultaneously, the Bonferroni 95% upper confidence bound `abs(mean_N-mean_H)+z*sqrt(MCSE_N^2+MCSE_H^2)` is at most `0.10` comparator posterior SD |
| Promotion | Sampler pass and physical-mean agreement pass move only that cell to `NEUTRA_CONFIRMED`; this state is expressly mean-level, not full-distribution equivalence |
| Vetoes | Target/transport/comparator hash drift, invalid target status, nonfinite state/energy/diagnostic, divergence, no moved chains, warm-up or retained cap, failed convergence/ESS, or failed simultaneous agreement |
| Explanatory only | Acceptance, probe ordering, source-coordinate summaries, physical truth distance, training/heldout loss, runtime, and SGQF-versus-UKF differences |
| Not concluded | Full-distribution equivalence, tail/covariance/mode agreement, filter exactness, SGQF/UKF superiority, calibration, robustness across datasets, production readiness, or a universal NeuTra recipe |

## Statistical Design Repair

P0 recorded posterior equivalence margins as
`NOT_FROZEN_TARGET_BLOCKER`; therefore the master/runbook phrase
"P0-frozen simultaneous uncertainty/equivalence rule" is stale. No prior P4
target, comparator, or training result used an agreement margin, so this is a
pre-R4 design repair rather than a reinterpretation of R4 output.

The primary estimands are the posterior means of physical
`(r,K,a,s,u,v)`, obtained from the frozen six-probit chart. The practical margin
is `0.10` times the comparator posterior SD for the same estimand. This defines
a discrepancy small relative to posterior uncertainty without using the
unknown synthetic truth. Mean MCSE is posterior SD divided by the square root
of split-chain cross-chain ESS. Family-wise alpha is `0.05` over six two-sided intervals, with
the critical value computed from the standard normal quantile
`1-alpha/(2*6)`. Equivalence requires confidence-interval containment; failure
to reject equality and overlapping marginal intervals are not passes.

The normal MCSE interval is used only after the hard convergence gate supplies
at least four chains, bulk ESS `>=1000`, and finite diagnostics. At every
retained checkpoint the joint diagnostic requires both convergence and
agreement. A convergence-only pass therefore extends sampling until agreement
also passes or the 10,000-draw cap is reached. At the cap, a simultaneous lower
bound beyond the practical margin supports material mean disagreement; an
interval crossing the margin is classified as insufficient precision, not
agreement or disagreement. The comparison does not establish full-distribution
or exact posterior correctness because means do not determine a distribution
and both samplers share the same approximate filter target.

## Fresh Kernel And Sampling Design

- Four latent chains start at zero and three fixed small dispersed offsets.
- Six fresh step sizes: `0.025`, `0.05`, `0.10`, `0.20`, `0.40`, `0.80`.
- Ten leapfrog steps, `128` burn-in transitions, and `64` retained draws per
  probe. Probe R-hat, acceptance, and loss cannot promote a kernel.
- Warm-up uses 1,000-draw chunks, at least 2,000 draws, a recent 1,000-draw
  rank/folded-R-hat window, and a 10,000-draw cap per chain.
- Retained sampling uses 2,000-draw chunks, at least 4,000 draws, and a
  10,000-draw cap per chain.
- Warm-up and retained roots and seeds are disjoint. All warm-up samples are
  retained as evidence and excluded from posterior summaries.
- The UKF and SGQF probe, warm-up, and retained roots are mutually disjoint and
  disjoint from target admission, comparator, screen, heldout, and final
  training seeds.

## Default And Assumption Audit

| Choice | Provenance | Failure mode | Early diagnostic | Status |
| --- | --- | --- | --- | --- |
| transported latent HMC | NeuTra mechanism and repository shared controller | learned geometry is worse or invalid | fresh probe ladder and health/status checks | candidate mechanism |
| six-step logarithmic grid | prior LGSSM NeuTra protocol | viable scale outside grid | no eligible probe or cap failure | bounded target-specific hypothesis |
| ten leapfrog steps | prior NeuTra validation protocol | poor trajectory length | probe ESS and sequential convergence | warm-start hypothesis |
| four chains and 10,000 caps | owner/runbook policy | unresolved slow mixing | cap is an honest blocked state | reviewed policy |
| physical posterior means | declared predator-prey parameters | misses tail, covariance, or mode disagreement | convergence/tail ESS remain hard gates; SD/quantile/covariance summaries are explanatory and the claim is narrowed | primary agreement estimands |
| 0.10 comparator SD margin | negligible-relative-to-posterior-scale design | too loose for a downstream decision or too tight under MC error | simultaneous interval and per-parameter table | R4-only practical equivalence margin |
| Bonferroni normal intervals | transparent six-estimand family control | nonnormal MC error | ESS/R-hat prerequisites and explicit nonclaim | reviewed bounded approximation |

## Required Artifacts And Checks

1. Verify every declared source hash and reconstruct the repository-issued typed
   identity before loading the frozen transport.
2. Require exact equality of target signature, training state hash, transport
   hash, artifact signature, recipe, step count, and final-training seed.
3. Run a compiled frozen-transport target canary and reject target/status or
   device drift before the probe ladder.
4. Execute the probe ladder and the shared adaptive sequential controller on
   trusted GPU/XLA with memory growth configured before logical initialization.
5. Archive every warm-up and retained chunk plus cumulative tensors in disjoint
   directories, preserving typed target identity and seeds.
6. Compute full rank-normalized split and folded rank-normalized split R-hat,
   bulk ESS, and tail ESS in source coordinates. The maximum of rank and folded
   R-hat controls every R-hat gate.
7. Load the comparator retained tensor only after its recursive hash ledger and
   target signature pass, then compute the predeclared physical-mean simultaneous
   equivalence table using TensorFlow/TFP without NumPy.
8. Retain physical SD, marginal quantile, and correlation differences as
   explanatory diagnostics; do not silently promote them into equivalence or
   full-distribution claims.
9. Write the result, decision table, inference-status table, manifest, ledger,
   and recursive hash record. Run focused tests and scoped `git diff --check`.

## Repair, Handoff, And Stop Conditions

- A serialization, reporting, path, or localized harness defect is repaired in
  a fresh attempt under unchanged target, transport, grid, seeds, thresholds,
  and compute budget.
- No eligible probe, warm-up cap, retained cap, or failed modern diagnostics is
  `SAMPLER_BLOCKED` for that cell. It does not reject NeuTra generally.
- A converged run whose simultaneous lower bound exceeds the margin is
  `SAMPLER_BLOCKED_SAME_TARGET_MEAN_DISAGREEMENT`; investigate implementation,
  multimodality, and comparator validity before another recipe. A cap-hit
  interval that still crosses the margin is
  `EVIDENCE_BLOCKED_AGREEMENT_PRECISION`, not candidate disagreement.
- A target identity or status failure reopens target/transport binding and
  cannot be repaired by tuning.
- Independent cells continue after a cell-local block. Close P4 only after both
  eligible cells have terminal R4 states and `PP-ZC` remains explicitly blocked.
- At P4 close, run focused checks, write a phase result/close record, refresh P5
  from actual evidence, audit P5 suitability, and continue unless a shared
  harness validity veto or program-budget veto fired.

## Skeptical Pre-Execution Audit

Decision: `PASS`.

The comparator is target-identical, acceptance and short probes are not
promotion criteria, warm-up is retained but excluded, sampling grows to fixed
caps, modern rank and folded R-hat plus ESS and health control admission, and
the agreement criterion is frozen before any R4 samples. The audit found and
repaired the stale P0-margin claim. Claude's bounded material review then found
and this revision corrected an `SD/ESS` prose error, narrowed confirmation to
the tested physical means, and made precision part of the sequential stop rule.
Both final training artifacts subsequently passed recursive-hash, target,
GPU/XLA, status, frozen-parity, and heldout validation. Their result hashes are
`1650d256577f91d54e6c351545e9a7ef0cb208844dc859f19eecc3b496af27c9`
for `PP-UKF` and
`de5f7cc35f606fe6d07177d1059d24acc1187e80b4bda42963f9e2823bf64bd4`
for `PP-SGQF`. R4 may execute under the frozen contract above.
