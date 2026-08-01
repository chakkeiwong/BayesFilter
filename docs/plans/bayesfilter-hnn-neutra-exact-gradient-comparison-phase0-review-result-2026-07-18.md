# HNN-NeuTra Exact-Gradient Comparison Phase 0 Review Result

Decision: `PASS_PHASE0_REVIEWED_READY_FOR_HARNESS_REPAIR`.

## Scope And Finding

The prior P4/P5 subplans required exact-gradient same-chart NeuTra-HMC, but the
four nonlinear executable experiments omitted that arm and compared HNN only
with a zero-residual Gaussian-force ablation. That is wrong relative to the
stated performance question. Existing nonlinear validity artifacts remain
useful, but they cannot support an HNN-versus-NeuTra-HMC speed or accuracy
comparison.

## Skeptical Audit

| Audit item | Result |
| --- | --- |
| Exact baseline | Same frozen chart and exact transformed filtering gradient; pass |
| Proxy promotion | Zero residual is historical explanatory evidence only; pass |
| Matched mechanics | Same positions, seeds, step size, leapfrog count, transitions, endpoint, dtype, GPU, and XLA; pass |
| Tuned fairness | Same candidate/sample budget, independent healthy selection; pass |
| Timing | Explicit device synchronization, cold/warm separation, three alternating repeats; pass after repair |
| Accuracy | Modern convergence, truth/reference, intervals, and pooled-MCSE direct comparison; pass after repair |
| Cost accounting | Supervision, screening, final fit, tuning, sampling, reuse, common chart, break-even, and guarded from-scratch ledgers; pass |
| Defaults | Every material inherited choice classified with failure diagnostic; pass |
| Stop/repair | Candidate failure separated from continuation veto; per-cell and total ceilings present; pass |
| Artifact coverage | Fresh versioned roots, samples, manifests, hashes, phase results, synthesis; pass |

## Advisory Review Limitation

Claude health and one-file read probes succeeded with `CLAUDE_PROBE_OK` and
`PLAN_READ_OK`. The bounded substantive review and a fixed-token baseline/timing
review returned no output despite successful process exit. The limitation is
recorded and execution continues under the repository's proportional advisory
review rule. No Claude approval is treated as execution authority.

## Evidence Contract

| Field | Status |
| --- | --- |
| Question | Does HNN force replacement preserve posterior accuracy and reduce useful-sample cost relative to exact-gradient NeuTra-HMC on the same chart? |
| Baseline | Exact-gradient same-chart NeuTra-HMC |
| Primary criterion | Both valid; HNN lower warm seconds/minimum bulk ESS and lower matched seconds/transition |
| Vetoes | Identity, parity, finite/energy, modern R-hat/ESS, truth/reference, direct agreement, mechanics, and budget gates |
| Explanatory only | Acceptance, force loss, raw ESS differences, zero residual, and one-seed effect sizes |
| Not concluded | Universal superiority, calibration, latent-model exactness, default readiness, or statistically supported ranking |
| Artifact | `docs/plans/bayesfilter-hnn-neutra-exact-gradient-comparison-repair-plan-2026-07-18.md` |

## Phase 1 Handoff

Implement the exact-gradient arms, synchronized timing, matched benchmark,
direct posterior comparison, complete cost ledger, and regression tests. Run
CPU-hidden focused checks before any trusted GPU canary. No real continuation
veto fired.
