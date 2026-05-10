# Audit: BayesFilter v1 Phase-completion Plan

## Date

2026-05-11

## Scope

This audit reviews:

```text
docs/plans/bayesfilter-v1-phase-completion-plan-2026-05-11.md
```

The audit is written from the stance of another developer checking whether the
plan can be executed safely in the BayesFilter v1 external-compatibility lane.

## Verdict

Approved for scoped execution with one required strengthening:

- Phase D must add benchmark memory metadata before treating the CPU benchmark
  harness as closed for this phase.

The plan correctly avoids immediate MacroFinance and DSGE switch-over.  It also
correctly separates local BayesFilter CI gates from optional live external
compatibility evidence.

## What The Plan Gets Right

1. Lane ownership is explicit.
   The plan uses the lane-specific reset memo and forbids staging the shared
   monograph reset memo or unrelated sidecar files.

2. The MacroFinance pivot is correct.
   MacroFinance is treated as an external compatibility target until
   BayesFilter v1 stabilizes.  This avoids forcing client-project churn while
   BayesFilter is still defining its public surface.

3. DSGE containment is correct.
   SGU remains blocked for production filtering, while Rotemberg and EZ are
   only future optional live fixtures.  This is the right interpretation of the
   read-only DSGE inventory.

4. The evidence hierarchy is sensible.
   Local tests can become CI gates.  Live MacroFinance/DSGE checks remain
   optional compatibility evidence.  GPU/XLA-GPU and HMC claims require
   separate target-specific artifacts.

5. The hypotheses are falsifiable.
   The plan maps API stability, local compatibility, benchmark metadata,
   optional external checks, DSGE containment, GPU/HMC blockers, and SVD/eigen
   derivative deferral to explicit tests or evidence.

## Required Strengthening

The benchmark harness previously recorded timing, shapes, device scope, and
point counts, but not process memory.  Since the active May 11 plan lists
memory metadata as a remaining gap, Phase D should not be closed until the
harness records either:

- process RSS metadata for each row; or
- an explicit platform caveat explaining why memory metadata is unavailable.

This audit therefore approves execution only if Phase D adds or explicitly
vetoes memory metadata before creating final benchmark artifacts.

## Non-blocking Risks

1. Process RSS is not an isolated per-backend allocation profile.
   It is still useful as diagnostic metadata if clearly labeled.

2. Medium-shape benchmarks may be slow because QR score/Hessian tracing is
   expensive.
   If runtime becomes unreasonable, the execution pass should record a veto
   with the attempted shape rather than weakening the claim.

3. Optional live MacroFinance checks can be informative but should not become
   a CI dependency.
   The plan already handles this correctly by allowing `not run by policy`.

4. GPU claims remain blocked.
   This machine requires escalated commands for GPU/CUDA checks, and no GPU
   benchmark should be inferred from CPU-hidden TensorFlow runs.

## Execution Approval

Proceed through the plan if these veto diagnostics remain false:

- no shared monograph reset memo is staged;
- no `Zone.Identifier` sidecars, images, or unrelated DSGE request notes are
  staged;
- no MacroFinance or DSGE source files are edited;
- benchmark artifacts keep CPU-only scope;
- SGU is not promoted to production filtering;
- GPU, HMC, and SVD/eigen derivative claims do not exceed evidence.
