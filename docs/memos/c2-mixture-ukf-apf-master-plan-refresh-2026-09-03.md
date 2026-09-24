# C2 Mixture-UKF/APF Master-Plan Refresh Memo

Date: 2026-09-03
Status: `PHASE3_FIXED_SPLITS_COMPLETE_PHASE4_READY`
Governing plan: [bayesfilter-c2-mixture-ukf-apf-master-program-2026-09-02.md](../plans/bayesfilter-c2-mixture-ukf-apf-master-program-2026-09-02.md)

## Decision

The master program has completed the bounded Phase 0 preflight, the Phase 1 CPU
smoke, the Phase 2 entry pilot, the twelve-branch `N=8192` serious expansion,
and the twelve-branch Phase 3 fixed-split expansion. The Phase 3 run passed all
finite-program validity gates. K=1, K=2, and K=4 lose the declared ESS
comparison to several cheap adversaries, so none is promoted; the unchanged
contract now advances to the smooth-gate and defensive-tail repair.

The first two entry launches (`phase2-entry-attempt01` and `attempt02`) failed
before fitting because the local TensorFlow build initialized logical GPUs
before the growth helper could configure them. Attempt 01 exposed the
environment ordering; attempt 02 showed that restoring the override before the
high-dimensional package imports was still too early. Both are localized
infrastructure-ordering failures. The driver now defers the override and all
high-dimensional algorithm imports until after explicit policy configuration;
the unchanged-contract retry `phase2-entry-attempt03` passed.

## Change

The governing plan now requires every phase to close with a versioned result
note that classifies each failure, repairs local failures under the unchanged
scientific contract, runs a focused regression, refreshes the next phase's
settings and command, and records either `CONTINUE_NO_REAL_BLOCKER` or a precise
continuation veto. A phase-specific matrix covers preflight, UKF/APF, mixture,
defensive-gate, recursive-map, basis, reference-law, global-mixture, and
integration failures.

Low ESS, a heuristic-dominance veto, a poor held-out residual, or rejection of
one candidate is a repair or promotion event, not an automatic stop. A target
or measure mismatch, nonfinite or unsupported finite program, failed exact
identity or call-chain parity, corrupted evidence, exhausted budget, or a
change requiring new authority is a true continuation veto.

## Execution update

Phase 0 produced a passing fresh artifact at
`docs/benchmarks/artifacts/c2_mixture_ukf_apf_20260902/phase0-attempt02`.
Phase 1 smoke attempts at `N=256` and `N=1024` passed the required K=1
validity, score, denominator, observation-response, and XLA-parity checks;
the focused regression reported `24 passed`. The K=1 minimum ESS was lower
than bootstrap and transformed Student in both one-branch smoke rows. This is
descriptive candidate evidence and triggers Phase 2 comparison/repair; it is
not a continuation veto or a superiority ranking.

The Phase 1 close note is
`docs/benchmarks/artifacts/c2_mixture_ukf_apf_20260902/phase1-smoke-close-20260903.md`.
The MathDevMCP/Lean record is
`docs/benchmarks/artifacts/c2_mixture_ukf_apf_20260902/phase0-attempt02/formal-audit.md`.
The Phase 2 entry close note is
`docs/benchmarks/artifacts/c2_mixture_ukf_apf_20260902/phase2-entry-close-20260903.md`.
The Phase 2 serious close note is
`docs/benchmarks/artifacts/c2_mixture_ukf_apf_20260902/phase2-serious-close-20260903.md`.

The Phase 3 close note is
`docs/benchmarks/artifacts/c2_mixture_ukf_apf_20260902/phase3-fixed-split-close-20260903.md`.
It records `96/96` valid records, the independent six-row offset calibration,
the formal split audit, and the descriptive ESS/cost veto for both fixed
mixture sizes. Repeated TensorFlow retracing warnings are preserved as a
performance/lifecycle repair trigger; they did not invalidate the finite
program.

## Next action

Implement and execute Phase 4's fixed-size smooth softmax/sigmoid gate with a
full-support Student defensive component on the same C2 target and paired
contract. Use disjoint calibration data, freeze topology and selected controls,
evaluate the complete mixture density in every denominator, and preserve the
exact frozen analytical-score contract. Larger rows remain closed until the
Phase 4 validity and cost record is reviewed. Reviewer or formal-tool
unavailability remains a recorded limitation; a material mathematical or
implementation finding must be resolved before the affected phase proceeds.

## Integrity checks

On 2026-09-03 the plan passed Pandoc GFM parsing, balanced-math checks (16/16
display and 57/57 inline delimiters), linked-path checks, ASCII/whitespace
checks, and an untracked-file diff check. These checks do not constitute a
mathematical proof or an execution result.
