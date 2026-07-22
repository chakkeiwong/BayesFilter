# Phase 2: Structural Support Regression Result

Date: 2026-07-15

Status: `PASS_CLOSED_HANDOFF_READY`

## Verdict

The exact two-state structural fixture is loop-native and passes local,
CPU/XLA, and trusted GPU/XLA support and total-tangent gates. At every step, the
carried points are fixed-index selections from a parent-by-innovation teacher;
the deterministic completion residual and its full carried-state total tangent
remain inside predeclared fixture-specific floating implementation guards.

This proves the stated finite fixture only. It is not an executable NAWM/SIR
likelihood or evidence for general structural filtering accuracy.

## Preparation And Mathematics

The inherited one-step chart `[1,4,6,11]` failed at step 1 with minimum weight
`-0.1599078`. Before result execution, an offline exhaustive 4-of-12 preparation
froze five per-step charts by maximum minimum weight with lexicographic tie-break.
There were 79 to 92 positive full-rank choices per step; selected minimum weights
range from `0.1899892` to `0.2314368`.

Final preparation regenerated after all XLA repairs:
`docs/benchmarks/artifacts/contract_e_tp_all_model_clean_xla_validation_20260715/phase-02/structural/final-preparation-20260715/preparation.json`,
SHA-256 `7e0cae35770f1e9620c8f22997ecdafd9f828e4c993dc5990e5f4376541f0a07`.
All scientific fields and selected charts are identical to the controlling run
preparation; only source provenance and generation metadata changed.

The tangent check explicitly carries all four parameter directions through the
parent state and transition, then gathers fixed selected point tangents. Reverse
autodiff independently owns the scalar score. The roundoff guards cover only
algebraic recombination conditional on recorded TensorFlow `tanh` values and
Jacobians; no transcendental-kernel accuracy claim is made.

## Evidence

| Check | Result |
| --- | --- |
| `T=1,2,5` loop/unrolled value, score, increments, final state/weights | pass at roundoff |
| same-scalar central FD | pass |
| maximum valid `T=5` support residual | `8.23e-17`, bound `3.35e-15` |
| maximum valid `T=5` support tangent | `2.22e-16`, bound `3.35e-14` |
| minimum selected weight | `0.1899892` |
| functional graph loops | 2: forward `While`, reverse `StatelessWhile` |
| same-factory `1e-3` invalid input | `valid=false`; value, score, state, weights nonfinite |
| final focused suite | `21 passed, 2 dependency warnings` |

CPU/XLA `T=5` artifact:
`docs/benchmarks/artifacts/contract_e_tp_all_model_clean_xla_validation_20260715/phase-02/structural/attempt-05-local-manual-tangent-20260715/result.json`,
SHA-256 `12fce6a08a40e942d8d03f5852dacfd3ba8a488129c3ab4723b015a80d517541`.

Exact CPU/GPU comparison uses `T=2` and the same preparation prefix. Value
absolute difference is `4.44e-16`; maximum score-component difference is
`3.33e-16`.

- CPU `T=2` SHA-256:
  `0423a35b557acb4c632663c6f5a098742722f16806c37c97a5a089bb609f2a2c`.
- Trusted GPU `T=2` SHA-256:
  `5800c59af582d69970ca08c93e89561286f3268611d212ab3eddea56a3b1aaa8`.

GPU compile plus first call was `3.406 s`; warmed valid and same-factory invalid
calls were `0.0159/0.0163 s`. The GPU was an RTX 4080 SUPER under the owner-
designated managed-session trust basis.

## Repair Record

1. Repeated one-step chart failed step 1; fixed offline per-step charts repaired
   preparation without runtime selection.
2. CPU/XLA attempt 1 exposed variant `TensorArray` histories; replaced with
   fixed-shape tensor carries.
3. Attempt 2 localized remaining TensorLists to reverse history Jacobians.
4. Attempt 3 exposed dynamic repeat/tile gradient shapes; static broadcast and
   reshape preserved parent-major ordering.
5. Attempt 4 exposed higher-order gather gradients from forward autodiff;
   explicit four-direction tangent carry replaced higher-order autodiff.
6. Attempt 5 CPU/XLA passed. The first trusted GPU attempt passed.

All failed attempts are preserved. No CPU diagnostic consumed a trusted fixture
attempt or eligible-model full-horizon attempt.

## Decision And Inference Status

| Decision | Primary status | Veto status | Next action | Not concluded |
| --- | --- | --- | --- | --- |
| close structural shared gate | support/tangent/local/XLA/GPU gates pass | clear | scalar-SV loop core | no general structural validity |
| retain explicit tangent carry | exact identity plus reverse-score/FD checks pass | none | regression guard | no generic tangent API |
| preserve NAWM/SIR blockers | fixture is not a client scalar | row veto unchanged | Phase 7 re-audit | no client admission |

Hard veto screen: passed for this fixture. Statistical ranking: not applicable;
the program is deterministic. Default readiness: false. The strongest risk is
that this small fixture does not cover client geometry or transcendental-kernel
error. Client-specific support evidence remains mandatory.

Phase 2 used conservatively 0.10 CPU core-hours and 0.01 trusted GPU-hours.
Remaining campaign budget: 95.87 CPU core-hours, 31.99 trusted GPU-hours, and
three full-horizon attempts per eligible model.

## NEXT_PHASE_READINESS

| Clause | Status | Evidence |
| --- | --- | --- |
| result/check consistency | `PASS` | hashes and `21 passed` |
| legal row classifications | `PASS` | actual/KSC only; generalized excluded |
| exact identities | `PASS` | Phase 3 target/time-order anchors |
| assumptions audited | `PASS` | Phase 3 defaults table |
| criteria/vetoes/nonclaims/fresh paths | `PASS` | Phase 3 subplan |
| executable next commands | `PASS` | prefix/loop/reference checks |
| no unresolved material boundary | `PASS` | shared structural veto clear |
| budget | `PASS` | Phase 3 minimum plus reserve below 95.87 CPU |

Overall: `READY` for
`docs/plans/bayesfilter-contract-e-tp-all-model-clean-xla-validation-phase3-scalar-sv-loop-core-subplan-2026-07-15.md`.
