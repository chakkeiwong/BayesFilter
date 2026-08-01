# P4 R4 Plan Review Record

Date: 2026-07-16

## Codex Skeptical Audit

The first audit found that P0 had deliberately left target-specific posterior
equivalence margins unfrozen, so the runbook's `P0-frozen` R4 dependency was
stale. The R4 subplan repaired this before any transported-HMC samples by
freezing physical posterior means, a practical margin, simultaneous uncertainty
control, precision classifications, and explicit nonclaims.

## Claude Read-Only Review

Claude was responsive to a tiny health probe. The first one-path review found
three material issues:

1. the prose incorrectly stated mean MCSE as `SD/ESS` rather than
   `SD/sqrt(ESS)`;
2. the confirmation claim was broader than the six mean estimands; and
3. convergence-only stopping could leave the equivalence test underpowered.

The same subplan was visibly revised to correct the formula, narrow confirmation
to physical-mean agreement, and require the retained controller to continue
until convergence and resolved agreement both pass or the 10,000-draw cap
distinguishes supported disagreement from insufficient precision.

The second bounded one-path review returned:

`No remaining material issues on the three checked findings.`

`VERDICT: AGREE`

Claude was read-only and advisory. Codex remains supervisor and executor.

## Local Checks

- New harness and test modules compile.
- Focused equivalence/classification, shared campaign, sequential HMC, and
  convergence test suites pass.
- Static scan finds no NumPy import, host callback, or Python sample/time/training
  loop in the active computation path.
- Scoped `git diff --check` passes.

Decision: `PASS_FOR_R4_EXECUTION`.
