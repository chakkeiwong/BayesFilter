# Phase 4 Terminal Result Reconciliation

Date: `2026-07-20`

## Decision

`PHASE4_COMPLETE_TWO_SEED_DIAGNOSTIC`

The authoritative second-seed result completed successfully after the earlier
runtime note was written. The earlier
`phase4-second-seed-replay-result.md` is preserved as a historical snapshot,
but its `BLOCKED_RUNTIME_BEFORE_RETAINED_SAMPLING` decision is superseded by
the completed result below.

## Evidence

| Run | Result | Sampler screen | Truth tail |
| --- | --- | --- | --- |
| Attempt 01 | `docs/plans/artifacts/bayesfilter-public-tuner-fixed-identity-mass-20260719/phase4-lgssm-attempt01/LGSSM-EXACT/result.json` | passed; max R-hat `1.00945`, bulk ESS `1083.29`, tail ESS `1192.16`, acceptance `0.691`, zero energy divergences | `MARGINAL_RERUN`, minimum `p_truth=0.0457386` for `q2` |
| Second seed | `docs/plans/artifacts/bayesfilter-public-tuner-fixed-identity-mass-20260720/phase4-second-seed-replay-background-2/LGSSM-EXACT/result.json` | passed; max R-hat `1.00615`, bulk ESS `1407.43`, tail ESS `1677.90`, acceptance `0.70725`, zero energy divergences | `PASS`, minimum `p_truth=0.0527368` for `q2` |

Both runs used the same admitted mechanics: step size
`0.7779889586003162`, six leapfrog steps, fixed identity mass signature
`25eb272b3f8b1e742173a12ea1ae6a07ba8a203dfdba3e6f67deebc30a7598fe`, and the
same frozen transport SHA-256
`b0b89656b2503146556f50b4e5e3e0e6b9b63daf0673380043ccb046dd14877e`.

## Interpretation

The second seed is valid one-seed truth-tail evidence under the unchanged
contract. Together, the two runs provide a two-seed diagnostic in which both
sampler screens pass and the second seed passes the truth-tail threshold. The
evidence does not establish universal NeuTra validity, distributional
equivalence, sampler superiority, default readiness, or production readiness.

The apparent failure was an observation error: the blocked note was inspected
before the detached process wrote its terminal `result.json`, `run_manifest`,
and sample archives. The new runner-state repair is intended to prevent this
classification error in Phase 5.
