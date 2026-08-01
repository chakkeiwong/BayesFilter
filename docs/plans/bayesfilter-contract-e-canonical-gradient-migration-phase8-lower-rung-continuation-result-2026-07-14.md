# Phase 8 Continuation Result: Minimal Identifying Lower Rung

Date: 2026-07-14

Program ID: `contract-e-canonical-gradient-migration-20260713`

Continuation ID: `contract-e-canonical-gradient-migration-continuation-20260714-115526`

Status: `NUMERICAL_TUPLE_SELECTED_LOWER_RUNG_KALMAN_SCREEN_FAILED_DIAGNOSTIC_ONLY`

## Outcome

The originally approved `T=1,N=32` design was wrong for the stated downstream
selection target. In the canonical recursion, each likelihood increment and its
gradient are accumulated before the same-time Contract E reset. With only one
observation, reset settings can change final particles and transport telemetry
but cannot change the measured likelihood or gradient. Attempt 2 confirmed this
exactly: steps `10,20,40,80` had bitwise-identical objective and score hashes
while final-particle hashes and row residuals differed.

The minimal target-preserving repair was `T=2,N=32`. The first reset is then
upstream of the second likelihood increment. This repaired graph selected the
smallest hard-valid ridge and a stable transport/chunk tuple:

```text
ridge = 0.1225 * 2^-24 = 7.301568984985351e-09
finite Sinkhorn steps = 20
row/column chunks = 16/16
epsilon = 0.5
scaling = 0.9
```

The selected tuple passed repeated-call identity, all chart/finiteness/source
checks, Cholesky positivity, the predeclared step/chunk edges, and the complete
same-program FD screen. It then failed the descriptive one-seed `N=32` Kalman
screen. This is candidate evidence at a feasibility rung, not a statistically
valid equivalence decision and not evidence that the Contract E idea is wrong.

## Exact Results

At `T=2,N=32`, estimator seed `80920`:

```text
Contract E value = -8.741205744876314
Kalman value     = -8.862150494354594
relative error   = 0.013647336451273853
value boundary  = 0.001
```

HMC-coordinate gradients:

| Parameter | Contract E | Kalman | Relative error | `<=5%` |
| --- | ---: | ---: | ---: | --- |
| `phi1` | `2.18232730999` | `1.84357139790` | `18.37%` | no |
| `phi2` | `-0.29376342814` | `-0.26796628591` | `9.63%` | no |
| `phi3` | `-0.16496031918` | `-0.07290903832` | `126.25%` | no |
| `q_scale` | `1.58994861489` | `1.54602404507` | `2.84%` | yes |
| `r_scale` | `4.78497313071` | `5.01216030114` | `4.53%` | yes |

There were no sign reversals. The FD-only maximum relative error was
`3.736230387439299e-08`, well below `0.05*sqrt(5) =
0.1118033988749895`. Therefore the gradient is the derivative of the executed
finite Contract E program; the Kalman disagreement is not an FD wiring failure.

Numerical diagnostics for the selected tuple included row residual
`0.004710369445179219`, column residual `1.7763568394002505e-15`, raw covariance
residual Frobenius norm `6.549862359719101e-09`, mean residual
`2.220446049250313e-16`, and finite moderate Cholesky condition proxies. These
are explanatory diagnostics, not scientific promotion criteria.

## Attempts

| Attempt | Classification | Result |
| --- | --- | --- |
| 1 | Localized harness failure | Four valid nodes preserved; driver used unequal strict-zip slices and stopped before edge evaluation |
| 2 | Invalid experimental design | Proved `T=1` objective/gradient invariance to reset settings; also exposed reversed edge residual interpretation |
| 3 | Valid repaired lower rung | Selected ridge/steps/chunks, passed FD, failed one-seed `N=32` Kalman screen |

## Decision Table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Retain selected numerical tuple for diagnostics | Step/chunk stability and FD passed | No engineering/chart/identity veto | Generalization beyond `T=2,N=32` | Freeze tuple and run a small no-reset versus Contract E increasing-`N` diagnostic | Canonical admission or scientific equivalence |
| Reject `T=1` downstream selection design | Reset has no later likelihood increment | Design invalidity proven in code and hashes | None for this timing claim | Require `T>=2` for reset-effect diagnostics | Candidate rejection |
| Treat Kalman discrepancy as repair trigger | Value and three gradient components failed descriptive thresholds | No sign reversal; FD passed | Finite-particle error versus reset-induced bias | Baseline/N ladder with same data and settings | Contract E research-direction rejection |

## Inference Status

| Inference | Status |
| --- | --- |
| Hard veto screen | Engineering and numerical-selection checks passed; descriptive Kalman screen failed |
| Statistically supported ranking | None; one estimator seed and no uncertainty interval |
| Descriptive-only differences | All Kalman value/gradient errors and transport telemetry above |
| Default-readiness | Blocked; production factory remains empty |
| Next evidence needed | Selected Contract E versus no-reset baseline across an increasing-`N` diagnostic, then a reviewed primary-shape plan only if warranted |

## Artifacts

- Aggregate result:
  `docs/plans/logs/contract-e-canonical-gradient-migration-continuation-20260714-115526/phase8/lower-rung/attempt3-t2/ladder-result.json`
  (SHA-256 `dbed2ba1007fc0d32d2e0ead85ce5584ec653c36e973b149cee9c91b60624de0`).
- Final selected-tuple/FD result:
  `docs/plans/logs/contract-e-canonical-gradient-migration-continuation-20260714-115526/phase8/lower-rung/attempt3-t2/node-06-final-verification-d89836a2b3c4a21f-fd/result.json`
  (SHA-256 `0509f808c2957cc46582d3e8f6cb33983ee8e99124486373594718dc47ec821f`).
- Wall time: `250.2407164920005` seconds for valid attempt 3.

## Post-Run Red Team

Strongest alternative explanation: the apparent Kalman failure is dominated by
single-seed finite-particle error at `N=32`, especially for the weak `phi3`
oracle component. A competing explanation is reset-induced finite-ensemble
bias. The current artifact cannot distinguish them.

What would overturn the interpretation: failure of the no-reset/Contract E
baseline ladder to show distinct, reproducible trends; source/hash drift; or a
same-program derivative failure at the repaired horizon.

Weakest evidence: all Kalman error magnitudes are one-seed descriptive values.
They must not be used to rank routes, tune `delta_grad`, or claim equivalence.
