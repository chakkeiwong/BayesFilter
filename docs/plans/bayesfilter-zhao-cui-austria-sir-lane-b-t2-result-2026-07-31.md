# Zhao-Cui Austria SIR Lane-B T2 Result

Date: 2026-07-31

Status: `PASS_NEW_FIXED_VARIANT_T1_T2_VALUE_BASELINE`

## Result

The fixed Lane-B finite program now has an admitted value through T2:

- selected arm: `t2_p05_r4_b5_lr3e4_l1_1e9`;
- selected T2 identity:
  `f51bb12bb6ab1a16cd843b350bb53a69cd449d602007278b8c5ef306a82e9f5e`;
- T1 value: `-31.1290512231882`;
- T2 increment: `-35.154752282413156`;
- cumulative T1:T2 value: `-66.28380350560136`.

The one-shot untouched estimate was `-35.17572642147189` with delta-method
log standard error `0.04658564052719933`. The absolute difference was
`0.020974139058736796`, below the frozen combined tolerance
`0.23312555986694752`.

## Hard Gates

| Gate | Result |
|---|---|
| Fresh T1/T2 identity reload | pass |
| Untouched same-scalar value | pass |
| Direct TT log-mass residual | `1.0081e-13`, pass |
| Cumulative value identity | exact, pass |
| GPU/XLA versus eager tie-out | exact, pass |
| Peak TensorFlow GPU allocation | `10,068,736` bytes, pass |

Claim artifact:
`docs/plans/artifacts/zhao-cui-austria-sir-lane-b-t2-20260731/attempt-13-selected-untouched-value-claim-xla-repair/result.json`

SHA-256:
`289565b59455a59e31190a5240ef98cbd885cfe4213677ecde1f22c31e206244`

## Tail Repair

The original untouched preparation failed because retained column `12287`
mapped to a finite previous state for which the author RK4 polynomial exceeded
FP64 state range. A source-bound signed-log evaluator showed the real polynomial
remained finite but reached components near `10^2573`. The fresh repaired
artifact retained all 16,384 rows and certified one FP64 extended-real
zero-density row with minimum standardized-residual overflow margin
`5548.1683` log units. Ordinary-row signed-log parity residual was
`4.94e-13`. No row was clipped, dropped, resampled, or removed from the Monte
Carlo denominator.

This tail result is correct for the explicitly declared FP64 extended-real
finite program. It is not a claim that the exact-real Gaussian density is zero.
The later score phase must separately prove zero derivative contribution or use
a wider stable score algebra.

## Inference Status

| Field | Verdict |
|---|---|
| Hard veto screen | passed |
| Statistically supported arm ranking | no |
| Descriptive-only differences | all six arms were viable; RMS ordering is descriptive |
| Default readiness | no |
| Next evidence | exact zero-parameter slice and analytical total score through T2 |

## Decision

Admit the T1:T2 fixed-value baseline and open the score slice. Do not run HMC.
Do not claim analytical score, T5/T10/T20, source-faithful assembled route,
production KR, posterior correctness, or scientific validity from this result.
