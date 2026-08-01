# Phase 8 Terminal Result Review

Date: 2026-07-14

Program ID: `contract-e-canonical-gradient-migration-20260713`

Reviewer: fresh bounded Codex substitute reviewer after the bounded Claude
one-path review returned no substantive output

## First Verdict

`VERDICT: REVISE`

The first review found two material scientific-wording defects in the numerical-
design instrumentation result:

1. `exact derivative` overstated the evidence, which was zero-ULP agreement
   between the manual JVP and forward autodiff on the frozen fixture; and
2. the handoff retained the superseded position that rigorous FD certification
   was pending callable error bounds, whereas the later FD result correctly
   classified that certificate as unconditionally unsupported and passed only
   the owner-directed heuristic screen.

## Repairs

- Replaced the exact-derivative claim with the measured zero-ULP manual-JVP /
  forward-autodiff agreement.
- Retired rigorous callable-error-bound FD certification as unsupported and not
  an active prerequisite.
- Preserved the seven-step same-program FD result as heuristic implementation
  evidence only.
- Preserved the independent owner numerical-design and primary-shape
  statistical blockers and all Kalman, target-shape, HMC, admission,
  leaderboard, release, and scientific nonclaims.

## Follow-Up Verdict

The repaired instrumentation result and representable-step FD result are
scientifically consistent. No material findings remain.

`VERDICT: AGREE`
