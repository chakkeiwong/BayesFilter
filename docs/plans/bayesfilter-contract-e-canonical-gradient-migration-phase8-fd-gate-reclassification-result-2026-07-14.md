# Phase 8 Result: Same-Program FD Gate Reclassification

Date: 2026-07-14

Program ID: `contract-e-canonical-gradient-migration-20260713`

Status: `FORMAL_FD_CERTIFICATE_UNSUPPORTED_HEURISTIC_SCREEN_INCONCLUSIVE_INVALID_ENDPOINT`

## Outcome

The prior demand for a rigorous callable-error-bound FD certificate is
unconditionally reclassified `unsupported`: no checked theorem or artifact
supplies absolute forward-error bounds for the compiled TensorFlow/XLA scalar
and score. This verdict is independent of the heuristic result.

The separate owner-directed seven-step same-program FD screen is
`FD_HEURISTIC_INCONCLUSIVE_INVALID_ENDPOINT`. All source, fixture, prepared-
input, center, branch, chart, one-concrete-callable, CPU-hidden, and JIT checks
passed. However, only 13 of 35 endpoint pairs had bitwise-equal representable
plus/minus actual steps. The remaining 22 were invalid under the predeclared
exact-symmetry rule, so the all-step screen cannot pass or fail.

## Descriptive Evidence

- valid symmetric endpoint pairs: `13/35`;
- invalid asymmetric endpoint pairs: `22/35`;
- valid relative-denominator-eligible pairs: `13/13`;
- valid pairs passing `0.05*sqrt(5) = 0.1118033988749895`: `13/13`;
- maximum valid relative error: `1.1488920760348987e-9`;
- center objective/score hex and branch hash reproduce Phase 5 v2;
- one XLA value-and-score callable evaluated center and every endpoint;
- all endpoint branch hashes and charts matched the center.

These valid-pair errors are explanatory only. Selecting only those 13 pairs
after observing the result would violate the all-seven-step contract.

## Attempt Record

Attempt 1 failed before endpoint output because `tf.one_hot` received
`tf.float64` as a positional `on_value` rather than the `dtype` keyword. The
failure is preserved. A one-line API repair and dtype regression test passed.
The single authorized retry used unchanged fixture, callable, ladder,
threshold, and environment and wrote the attempt-2 result.

No third attempt or alternate ladder ran.

## Decision Table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Rigorous FD certificate | Callable error bounds required | `unsupported` regardless of heuristic | executed-kernel forward error | Do not claim rigorous FD proof | Derivative proof |
| Seven-step FD heuristic | all 35 valid and passing | Inconclusive: 22 asymmetric representable pairs | endpoint construction | Keep screen inconclusive; separate reviewed redesign only | FD failure |
| Manual score evidence | zero-ULP forward autodiff plus checked identities | Passed on tiny fixture | general target behavior | Retain as primary engineering evidence | Kalman equivalence |

## Inference Status

| Inference | Status |
| --- | --- |
| Hard veto screen | Source/center/branch/chart identity passed; endpoint symmetry veto blocked FD heuristic pass |
| Statistically supported ranking | None |
| Descriptive-only differences | All valid-pair FD relative errors and Richardson diagnostics |
| Default-readiness | Not established |
| Next evidence needed | If still desired, a separately reviewed representable symmetric-step construction fixed before output; owner numerical and primary statistical designs remain independent blockers |

## Artifacts

- Harness: `docs/benchmarks/emit_contract_e_canonical_lgssm_phase8_fd_reclassification.py`
- Tests: `tests/highdim/test_contract_e_phase8_fd_reclassification.py`
- Attempt-1 failure: `docs/plans/logs/contract-e-canonical-gradient-migration-2026-07-13/phase8/fd-reclassification-attempt1/failure.json`
- Attempt-2 result: `docs/plans/logs/contract-e-canonical-gradient-migration-2026-07-13/phase8/fd-reclassification-attempt2/result.json`
- Attempt-2 SHA-256: `5261f5a627b14951f15a39d1e7ef5a8db2916f6e3ce413a25f33a5a74377f1c7`

## Nonclaims

The heuristic does not prove or disprove the derivative, supply a confidence
interval, or establish target-shape FD, Kalman equivalence, target numerical
adequacy, GPU/HMC/default/leaderboard/release/integrity readiness.
