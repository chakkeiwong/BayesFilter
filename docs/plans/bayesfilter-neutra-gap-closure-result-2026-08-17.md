# NeuTra Gap-Closure Result (2026-08-17)

## Provenance Correction

Every diagnostic in this result used checkpoint SHA-256 `57b21cc99778b0e24e6c5809ebbb6137709edf8177e7faeeac9d259deb2e7b12`,
the obsolete `(64,64)`, three-stage, 1,000-update baseline. The mode-trapping
diagnosis is correct for that checkpoint. It is not an unresolved failure of
the reviewed six-stage capacity repair, SHA-256 `b39c682030fb3ba8bafe863c747674db40b5d7c13e164c8445ddfab649ad93f6`,
which already passed downstream three-mode HMC and exact-law screens.

This result records execution of
`bayesfilter-neutra-gap-closure-plan-2026-08-17.md`.

## Decision Table

| Gap/hypothesis | Test | Result | Classification | Next action |
|---|---|---|---|---|
| H4 target/value/score mismatch | Analytic score vs GradientTape near all three modes and transformed score parity | Physical score max error `4.4e-15`; transformed score max error `7.1e-14`; all finite | Closed: not an implementation mismatch | Do not change target/score code based on this failure |
| H2 local frozen-transport geometry | Per-component physical clouds, inverse/forward reconstruction, log-determinant and cross-mode interpolation | Reconstruction max error `2.2e-15`; logdet error `4.4e-16`; component latent means differ strongly; interpolation reaches latent coordinate `-6.28` | Support geometry is finite but globally nonuniform; local mechanics pass | Treat as transport/support hypothesis, not a code bug |
| H1 initialization/mode coverage | Component-aware, balanced, and local starts with recorded `L=20/25` kernels | Chains remain in starting modes; transitions are zero in nearly all 512-draw chains | Confirmed mode-trapping/initialization sensitivity | Repair transport global support or use a reviewed multimodal training strategy |
| H3 verification-window artifact | Two seeds, two kernels, three starts | Same no-transition pattern across starts and seeds | Short-window noise alone is insufficient explanation | Longer verification is not the first repair; diagnose/retrain transport |

## Evidence

Focused contracts passed: `12 passed`. Diagnostic artifacts:

- H2/H4: `docs/plans/artifacts/neutra-gap-closure-2026-08-17-r1/h2-h4-gpu/`
- H1/H3: `docs/plans/artifacts/neutra-gap-closure-2026-08-17-r1/h1-h3-gpu/`

The H2/H4 probe used 64 draws per mixture component and the frozen checkpoint
SHA-256 `57b21cc99778b0e24e6c5809ebbb6137709edf8177e7faeeac9d259deb2e7b12`.
All target values, scores, transformed scores, inverse/forward maps, and
log-determinants were finite. The component latent means were approximately:

```text
mode 0: (-0.838,  0.010, -0.036, -0.136)
mode 1: ( 1.207, -0.087, -0.223,  0.189)
mode 2: ( 0.270, -0.074,  0.012,  0.187)
```

The H1/H3 probe used 512 draws per chain, 64 burn-in steps, `L=20` and `L=25`,
two seeds, and three initial-mode assignments. Component-aware and balanced
starts showed zero mode transitions in nearly every chain. Local starts stayed
in mode 0 for all draws. Chains moved within their local neighborhoods, so the
movement flag does not imply global mode exploration.

## What Is Closed

- The current evidence does not support changing the analytic three-mode target
  or score implementation.
- The inverse/forward transport and log-determinant mechanics are locally
  correct to numerical precision.
- The three-mode tuning veto is not explained by a single failed canary or a
  missing chain movement flag; it is consistent with mode trapping.

## What Remains Open

The capacity hypothesis is no longer open for this target under the
component-aware protocol: the original and two fresh `(128,128)`, six-stage
seeds passed support, tuned sequential HMC, and exact-law screens. The tested
zero-centered Student-t mode-blind proposal failed support before training.
What remains open is a target-query-driven proposal that discovers separated
regions without exact component centers. The evidence still does not justify
relaxing R-hat, using acceptance as a proxy, or launching HMC from the failed
small checkpoint.

## Inference Status

| Evidence class | Status |
|---|---|
| Hard veto screen | R-hat tuning veto applies to the obsolete small checkpoint only |
| Statistically supported ranking | None |
| Descriptive-only differences | Acceptance, local mode counts, latent geometry, logdet, runtime |
| Default-readiness | Not established |
| Next evidence needed | Target-query-driven mode-discovery proposal/support validation |
