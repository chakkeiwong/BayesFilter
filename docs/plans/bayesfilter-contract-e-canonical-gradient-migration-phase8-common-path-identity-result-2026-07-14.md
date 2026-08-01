# Phase 8 Common Proposal/Weight Identity Result

Date: 2026-07-14

Program ID: `contract-e-canonical-gradient-migration-20260713`

Continuation ID: `contract-e-canonical-gradient-migration-continuation-20260714-115526`

Status: `COMMON_PATH_IDENTITIES_PASSED_FINITE_PARTICLE_ACCURACY_GATE_REMAINS_BLOCKED`

## Decision

The fixed-particle common LGSSM proposal, importance-weight, normalization,
and manual derivative paths pass their local identities. No localized bug was
found in the affine flow/Jacobian/density wiring or in the carried `T=2`
no-reset weighted recursion. The `delta_grad=0.05` scientific comparison
therefore remains failed at the existing `T=2,N=128` lower rung; the identity
passes do not convert finite-particle error into Kalman agreement.

Contract E remains diagnostic-only. The production factory is empty. Phase 9,
leaderboard regeneration, HMC, default-readiness, and release remain blocked.

## Owner Decisions Bound For This Continuation

- gradient boundary: `delta_grad=0.05`;
- paired audit count: `16`, exploratory with no power claim;
- FD-only implementation screen remains separate at `0.05*sqrt(5) =
  0.1118033988749895`;
- LGSSM value boundary remains `0.001` relative;
- Kalman is the LGSSM oracle.

## Evidence Contract Executed

At the frozen center `theta=(0.72,0.55,0.35,0.35,0.45)`, fixed float64 tensors
were used with `B=2,N=4,d=3`. The following were compared:

1. the canonical per-particle transition/observation/proposal/Jacobian
   correction;
2. the analytic Gaussian predictive density;
3. TensorFlow autodiff Jacobians of both expressions;
4. the canonical manual JVP; and
5. the canonical `T=2` no-reset recursion against an independently written
   optimal-proposal sequential importance recurrence, including value and
   derivative.

The implementation identity tolerance was `atol=2e-11, rtol=2e-11` in CPU-
hidden float64. This is a local roundoff tolerance, not a Kalman accuracy or
statistical promotion threshold.

## Results

| Check | Maximum absolute error | Maximum relative error | Status |
| --- | ---: | ---: | --- |
| Single-step correction value | `4.440892098500626e-16` | `2.2144923690072852e-16` | pass |
| Single-step correction autodiff | `3.552713678800501e-15` | `8.149426500088737e-15` | pass |
| Manual JVP vs code autodiff | `3.1086244689504383e-15` | `6.177791056518894e-15` | pass |
| Manual JVP vs analytic autodiff | `1.7763568394002505e-15` | `1.9716354435698558e-15` | pass |
| Single-step normalized value | `4.440892098500626e-16` | `2.0224984637636798e-16` | pass |
| Single-step normalized autodiff | `1.3322676295501878e-15` | `1.2076121111545095e-15` | pass |
| Two-step recursion value | `4.440892098500626e-16` | `1.2731269551597425e-16` | pass |
| Two-step recursion autodiff | `1.7763568394002505e-15` | `1.104629085166033e-15` | pass |
| Two-step manual JVP | `4.440892098500626e-15` | `1.2793575629455792e-15` | pass |

All outputs were finite. The focused test passed `1` test with `2` dependency
warnings. The earlier paired-audit/common-path test bundle passed `6` tests.

Artifacts:

- [common-path attempt 1](/home/chakwong/BayesFilter/docs/plans/logs/contract-e-canonical-gradient-migration-continuation-20260714-115526/phase8/common-path-identity/attempt1/result.json), SHA-256 `c81fb4597334915a1d113efd160171152ad7d15b980a89cf2ec3759e00cccc97`;
- [common-path attempt 2](/home/chakwong/BayesFilter/docs/plans/logs/contract-e-canonical-gradient-migration-continuation-20260714-115526/phase8/common-path-identity/attempt2/result.json), SHA-256 `01597282a8232843adbdbb99633a441812f8912f18978be1343d963908bbf646`;
- [float64 interval correction](/home/chakwong/BayesFilter/docs/plans/logs/contract-e-canonical-gradient-migration-continuation-20260714-115526/phase8/paired-reset-audit16/float64-correction/result.json), SHA-256 `2bba7163daa12001b12d130b28bc5aa7c63145c24934cf6e044a9c25c93b4e56`.

## Paired-Audit Reporting Correction

The preserved 16-seed audit was recomputed with explicit float64 Student-
distribution tensors. The critical value changed from `3.0362837314605713` to
`3.036283222821165` (difference `-5.086394061493138e-07`). All component
directions, Contract E equivalence decisions, and the overall
`mixed_or_inconclusive` classification were unchanged. No stochastic arm was
rerun and the original artifact remains preserved.

The only supported reset-specific effect remains a small `r_scale` worsening:
mean paired absolute-loss increase `0.0011934373797213703`, corrected 95%
simultaneous interval `[0.00024797033356117166, 0.0021389044258815692]` under
the predeclared Bonferroni/Student model. The other five paired effects remain
inconclusive. This has no power guarantee and does not identify a general reset
mechanism.

## Root-Cause Classification

| Question | Classification | Basis |
| --- | --- | --- |
| Is the affine conditional flow correction wired to the wrong predictive density? | `not found` | Exact fixed-particle value identity at float64 roundoff. |
| Is the manual single-step JVP inconsistent with autodiff? | `not found` | Exact JVP/autodiff identity at float64 roundoff. |
| Is normalization or its derivative double-counted at one step? | `not found` | Exact normalized value/Jacobian identity. |
| Is the carried `T=2` no-reset value/derivative recursion inconsistent with an independent recurrence? | `not found` | Exact value/autodiff/manual-JVP tie-out at float64 roundoff. |
| Does this prove Kalman-gradient equivalence? | `no` | The checks are conditional finite-particle implementation identities, not a population-limit theorem. |
| Does this justify Contract E admission? | `no` | The `T=2,N=128` 16-seed Kalman screen still fails and the reset audit is mixed/inconclusive. |

The evidence shifts the leading explanation toward finite-particle estimator
bias/variance in the shared recursion, with a possible secondary reset-specific
effect. It does not prove finite-particle error is the only cause; the next
study must vary `N` and replicate seeds under a frozen design.

## Required Checks At Close

- focused common-path and paired-audit tests: pass (`6 passed` in the combined
  run; latest common-path test `1 passed`);
- Python compilation: pass for all three changed harnesses;
- JSON parsing: pass for all three new artifacts; artifact hashes are recorded
  above;
- `git diff --check`: pass;
- no canonical factory registration or leaderboard/HMC execution: confirmed.

## Handoff

The next phase is a separately budgeted fixed-`N`/multi-seed finite-particle
study, with `N` and seed count frozen before inspecting output. It must use the
same Kalman oracle, `delta_grad=0.05`, value boundary `0.001`, and explicit
uncertainty intervals. It must not be launched inside this continuation after
the deadline, and it must not tune settings from the existing failures.

Until that study passes the predeclared center-scoped Kalman criteria, keep the
factory empty and do not run Phase 9, HMC, leaderboard regeneration, or release.

## Nonclaims

- no claim of primary-shape validity or `T=50,N=10000` feasibility;
- no claim of reset correctness or Contract E scientific equivalence;
- no claim of statistical power, normality, distribution-free coverage, or
  method superiority;
- no claim of HMC, default, leaderboard, production, or release readiness.
