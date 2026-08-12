# GenUT Score-Variance Repair Validation Plan

Date: 2026-07-31
Status: `EXECUTED_DIAGNOSTIC_COMPLETE`

Terminal artifact:
`docs/benchmarks/artifacts/genut_score_variance_repair_validation_20260731/attempt08/`

Terminal result:
`docs/plans/bayesfilter-genut-score-variance-repair-validation-result-2026-07-31.md`

## Research Intent

| Item | Predeclared answer |
|---|---|
| Main question | Do the audit's corrected claims and proposed first repairs agree with measured finite-program behavior? |
| Mechanism diagnostic | Compare finite-time directional tangent growth with score SD for Austria diagonal-only, Austria pairwise, and LGSSM control arms. |
| Design diagnostic | Verify exact mean/covariance identities and distinguish raw `m22` from studentized co-kurtosis under fixed full whitening. |
| Primary criteria | All diagnostics finite; D1 is interpreted only as explanatory evidence; whitening identities pass at numerical tolerance. |
| Promotion veto | Non-finite route, invalid reset, score-increment mismatch, or failed exact whitening identities. |
| Statistical status | Three fixed particle seeds per arm are descriptive only; no superiority or asymptotic Lyapunov claim. |
| Nonclaims | No proof of `Var(score)=O(T)`, no exact Austria score oracle, no default/HMC/leaderboard promotion. |

## Scope And Controls

- Austria SIR: `d=18`, `N=1008`, `T=20`; diagonal controls from the July 30
  selected arm and pairwise controls from the July 30 selected arm.
- LGSSM: `d=3`, `N=1008`, `T=50`; the scope-matched diagonal controls from the
  July 30 cross-model campaign.
- Particle seeds: `98201..98203` for Austria and `98201..98203` for LGSSM;
  probe seed is fixed per particle seed and arm.
- Probe columns: `K=8`, propagated with zero explicit parameter source,
  renormalized per column at each time step.
- Backend: TensorFlow GPU/XLA, `float32`, TF32 enabled only because this is a
  diagnostic of the existing July 30 route; memory growth is required before
  logical-device initialization.
- Fresh output root:
  `docs/benchmarks/artifacts/genut_score_variance_repair_validation_20260731/`.

## Skeptical Audit

- D1 measures realized finite-horizon directional growth, not the asymptotic
  top Lyapunov exponent. A larger D1 value can be transient or directional.
- Score SD from three seeds is descriptive and cannot rank methods.
- The pairwise route changes the finite scalar, so a lower score SD does not
  establish lower score bias.
- The whitening check is deterministic algebraic evidence only; it does not
  establish downstream variance reduction.
- The run uses existing selected controls and does not retune them, so it is a
  diagnostic of the proposed mechanism, not claim-bearing tuning evidence.

## Artifacts

The runner must write `result.json`, `result.md`, and a run manifest containing
the Git commit, command, environment, GPU/memory policy, controls, seeds, and
source hashes. Existing artifacts must not be overwritten.

## Execution Closeout

The skeptical audit passed because the run treats directional growth and
three-seed score dispersion as explanatory diagnostics only. Attempt 08 passed
all hard validity and exact-whitening gates on the GPU/XLA route. The original
run manifest was too sparse for the complete serious-run metadata policy; an
append-only `run_manifest_supplement.json` records the missing fields and marks
which command details were reconstructed after the run. The terminal result
keeps the original manifest limitation explicit.
