# Phase 8 Result: Reset-Bias Versus Finite-N Diagnostic

Date: 2026-07-14

Program ID: `contract-e-canonical-gradient-migration-20260713`

Status: `MIXED_OR_NONMONOTONE_INCONCLUSIVE`

## Outcome

The reviewed six-node `T=2`, one-seed diagnostic completed for Contract E and
the no-reset weighted baseline at `N=32,64,128`. All nodes passed source,
prepared-input, chart, finiteness, Host-XLA, and allocation checks. The exact
predeclared classifier returned
`mixed_or_nonmonotone_inconclusive`.

Neither a shared finite-`N` pattern nor a reset-specific pattern is supported by
the all-six-quantity rule. Both arms improved from `N=32` to `64` for value,
`phi1`, `phi3`, and `r_scale`, then worsened at `N=128`; `phi2` worsened on both
edges in both arms. The paired reset effects were small for value, `phi1`,
`q_scale`, and `r_scale`, but mixed and non-monotone across components.

This is exactly the pattern the reviewed rule classifies as inconclusive. It
cannot be used to rank Contract E against no reset, reject Contract E, or tune
the particle count.

## Relative Error Table

| Quantity | CE N32 | CE N64 | CE N128 | No-reset N32 | No-reset N64 | No-reset N128 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| value | `0.01365` | `0.00908` | `0.02024` | `0.01383` | `0.00887` | `0.02067` |
| `phi1` | `0.18375` | `0.16275` | `0.22418` | `0.18562` | `0.16213` | `0.22937` |
| `phi2` | `0.09627` | `0.63111` | `0.97761` | `0.14583` | `0.75245` | `1.02120` |
| `phi3` | `1.26255` | `0.35673` | `1.03137` | `1.39995` | `0.27150` | `1.07774` |
| `q_scale` | `0.02841` | `0.03169` | `0.04518` | `0.01802` | `0.04352` | `0.03647` |
| `r_scale` | `0.04533` | `0.03844` | `0.11104` | `0.04079` | `0.03536` | `0.10861` |

These are one-seed descriptive values. The apparent `N=64` improvement in some
rows and worsening in others is not a supported optimum or ranking.

## Attempt Record

- Attempt 1 stopped on a localized no-reset harness defect: the inactive reset
  branch intentionally leaves `minimum_mass=inf`, which was incorrectly
  included in an all-output finiteness veto. Three Contract E nodes and the
  comparator artifact were preserved.
- Attempt 2 excluded only that inactive diagnostic sentinel from executed-output
  finiteness. All six nodes completed in `447.704355722999` seconds.

## Evidence And Decision

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Do not classify mechanism from one seed | Exact all-six monotonicity rule returned mixed | No implementation veto | Monte Carlo variability and weak oracle components | Paired multi-seed audit at a fixed predeclared `N` | Shared finite-N or reset-specific cause |
| Freeze rather than retune tuple | Numerical tuple already selected independently | No setting change allowed | Shape generalization | Use largest predeclared diagnostic `N=128`, not observed-best `N=64` | Particle-count optimum |

Aggregate artifact:
`docs/plans/logs/contract-e-canonical-gradient-migration-continuation-20260714-115526/phase8/reset-bias-n-scaling/attempt2/result.json`
with SHA-256
`e2f24a619524d6ffaac36482d53d48ef162a138f147f626df062fd3eae74749f`.

## Post-Run Red Team

Strongest alternative explanation: the non-monotone paths are ordinary
single-seed Monte Carlo fluctuations, not evidence that increasing `N` worsens
either route. What would overturn this interpretation is a paired multi-seed
analysis showing a stable reset effect or systematic arm-specific loss change.

Weakest evidence: there is only one estimator seed, and the `phi3` Kalman
component is small enough that relative errors are volatile.
