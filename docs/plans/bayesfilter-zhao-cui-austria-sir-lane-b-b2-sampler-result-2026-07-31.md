# Zhao-Cui Austria SIR Lane-B B2 Sampler Result

Date: 2026-07-31

Status: `PASS_B2_RETAINED_SAMPLER_ADMISSION`

## Verdict

The selected T1 artifact now has a deterministic, finite, correctly scored
retained-state sampler eligible to supply the Lane-B T2 research baseline. The
sampler uses exact Gauss-3 interval masses for the selected piecewise-quadratic
basis, inverts the resulting piecewise-linear CDF, reports the exact
piecewise-constant proposal density of that inversion law, and carries
`log(p_TT/q_grid)` explicitly.

This closes the missing B2 sampler-law gate. It does not change the admitted T1
artifact or its identity, and it does not establish production KR closure,
source-faithful numerical CDF construction, T2 value, score, T20, HMC,
posterior correctness, or scientific superiority.

## Primary Evidence

| Field | Untouched result |
|---|---:|
| Source T1 identity | `e4b56526205eb50c3d2aa3b8a8ce6ce27539aa5ab50ad286380136db28ed2b59` |
| Untouched reference seed | `73703` |
| Sample count | `64` |
| Maximum inverse/forward residual | `1.249000902703301e-16` |
| Maximum proposal/Jacobian residual | `0.0` |
| Maximum conditional-mass residual | `2.9398705692074145e-13` |
| Correction log-weight range | `[-0.07919093235988584, 0.11636193900336167]` |
| Correction ESS | `63.876166659701646 / 64` |
| Static workspace estimate | `20,673,536` bytes |
| CPU peak RSS | `734,306,304` bytes |
| CPU process cap | `12 GiB` |
| Fresh replay | all six hashes exact |

The untouched claim is
`docs/plans/artifacts/zhao-cui-austria-sir-lane-b-b2-sampler-20260731/attempt-04-cpu-untouched-claim/result.json`,
SHA-256
`c7bd8d61268e36b42e3794a52b76b5b24339da090cac800e324a85ce448d1b2d`.

The fresh replay is
`docs/plans/artifacts/zhao-cui-austria-sir-lane-b-b2-sampler-20260731/attempt-05-cpu-fresh-replay/result.json`,
SHA-256
`1649792818cdcf2e7621ea9497dce26cc557fac3582dc6e568133ef94a2906df`.

## Repairs And Negative Evidence

1. The first candidate used a trapezoid grid on trusted GPU. It failed with
   `NONFINITE_VALUE` in an FP64 marginal contraction.
2. A bounded GPU localization showed non-finite contractions at inconsistent
   axes and batch sizes. It also exposed raw trapezoid conditional-mass
   residuals as large as `1.2356145994159466`.
3. These observations reject the trapezoid/GPU candidate. They do not reject
   the T1 TT or the research direction because the passed B3 independent FP64
   CPU contraction remained valid and the next planned repair targeted the
   exact polynomial interval mass.
4. For the selected order-two Lagrange basis, the squared conditional is degree
   at most four inside each element. The 65-node partition includes the element
   breakpoint, so Gauss-3 exactly integrates every interval up to FP64 error.
5. Calibration seed `73702` passed before untouched seed `73703` was read.

Failed artifacts remain under attempts 01 and 02. No failed artifact was
overwritten or promoted.

## Source Support Summary

- `PRIMARY_TECHNICAL_SUPPORT`: Zhao-Cui Eq. (20)-(23) and Algorithm 3 for the
  conditional proposal and correction; Proposition 2 for TT marginalization.
- `IMPLEMENTATION_EVIDENCE`: author
  `@TTSIRT/eval_irt_reference.m:43-71`, `@TTSIRT/marginalise.m:19-85`, and
  `Polynomials/LagrangepCDF.m` confirm upper conditional order, paired-core
  marginalization, and polynomial CDF treatment.
- `PROJECT_DERIVATION`: the exact density of the repository's finite
  piecewise-linear CDF is the interval probability divided by interval width.
- `extension_or_invention`: the repository exact-interval piecewise-constant
  proposal is not claimed as the author's numerical CDF implementation.

No network metadata lookup or snowballing was needed: B2 depended on one
already cached direct-method paper and its locally stored author code. No
publication/retraction status claim is made here.

## Decision Table

| Field | Decision |
|---|---|
| Decision | Admit the finite retained sampler and open refreshed B4. |
| Primary criterion | Passed: fresh-process exact replay and exact-law scoring. |
| Veto diagnostics | Passed: identity, frame prefix, mass, roundtrip, finite correction, and memory. |
| Main uncertainty | The retained density remains a finite TT approximation; B2 validates its sampler law, not exact filtering truth. |
| Next justified action | Scope-specific T2 target, tuning, fit, reload, and same-scalar fixed-value gate. |
| Not concluded | No production KR, T2/T20 score, HMC, posterior, or superiority claim. |

## Inference Status

| Evidence class | Status |
|---|---|
| Hard veto screen | Passed for the finite B2 sampler law. |
| Statistically supported ranking | None; no candidates are ranked. |
| Descriptive-only differences | Weight range, ESS, runtime, and memory. |
| Default readiness | Not established. |
| Next evidence needed | B4 T2 untouched same-scalar value admission. |

## Post-Run Red Team

The strongest alternative explanation is that the 64 untouched references do
not expose every tail-conditioned numerical problem. The exact interval-mass
identity removes quadrature bias for this basis, but it does not prove future
T2 proposal quality. T2 must retain finite correction and validation vetoes on
its own disjoint data. A future non-finite T2 conditional would reject that T2
candidate; it would not retroactively change this B2 finite-program result.

Focused regression: `7 passed` for the B2 sampler plus B3 boundary suites.

