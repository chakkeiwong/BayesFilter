# Phase 4: Scalar-SV Trusted Full-Horizon GPU/XLA Result

Date: 2026-07-16

Status: `PASS_CLOSED_HANDOFF_READY`

## Verdict

KSC-SV passes the complete center-scoped `T=1000` CPU and trusted GPU/XLA
engineering ladder. Actual non-Gaussian SV is valid and finite at the frozen
center, has bounded XLA topology, and fails closed at the invalid control, but
its frozen `T=1000` square chart is invalid at two of the four central-FD
endpoints. Actual SV therefore fails Phase 4 admission under the predeclared
same-scalar derivative gate and receives no full-horizon GPU attempt.

This is not a shared XLA defect. It is a row-specific chart-region failure:
Actual SV center execution is valid while the negative gamma endpoint and
positive log-beta endpoint return `valid=false` and poison their objectives.
The plan prohibited changing the chart or FD step after observing the result.

## Implemented Repairs

- added `T=1000` offline preparation support;
- added a fresh-process CPU/trusted-GPU harness with the exact loop factory,
  full array evidence, source/graph audit, same-scalar FD, warm replay, and
  same-factory invalid control;
- added a trusted GPU preflight that configures and verifies memory growth
  before logical-device initialization;
- made the compiled scalar-SV invalid branch poison every increment in addition
  to objective, score, particles, and final weights;
- rebuilt the `T=100` comparison graph from its hashed preparation using the
  current factory, avoiding a stale-code graph comparison; and
- repaired failure reporting so nonfinite FD endpoints emit a structured failed
  gate rather than crashing the harness.

No target law, observation transform, chart, order, lookahead, scalar, FD step,
or threshold was changed after execution began.

## Evidence

| Check | Actual SV | KSC-SV |
| --- | ---: | ---: |
| `T=1000` preparation | pass, 999 charts | pass, 999 charts |
| minimum prepared weight | `3.59e-7` | `5.32e-6` |
| maximum condition number | `4.47e4` | `5.69e4` |
| CPU center value | `-2286.2238500567` | `-2284.0985319720` |
| CPU center score | `[5.6549640321,-2.5482162620]` | `[2.4568313518,-4.4208764004]` |
| four central-FD endpoints valid | fail: two invalid | pass |
| maximum FD relative error | unavailable because endpoints invalid | CPU `3.45e-8`; GPU `3.19e-8` |
| top-node ratio `T=1000/T=100` | `1.00677` | `1.00448` |
| function-node ratio | `1.0` | `1.0` |
| GraphDef-byte ratio | `0.99990` | `0.99972` |
| compiled invalid control | pass, all claim outputs poisoned | pass, all claim outputs poisoned |
| trusted GPU full-horizon result | not eligible | pass |

Controlling artifacts and SHA-256:

- Actual preparation:
  `docs/benchmarks/artifacts/contract_e_tp_all_model_clean_xla_validation_20260715/phase-04/scalar-sv/attempt-01-preparation-20260715/actual_t1000_preparation.json`,
  `c50882fe7413e5ca7a7813aaa8eaf206255bed23af147be38fe18a4287eaad80`.
- KSC preparation:
  `docs/benchmarks/artifacts/contract_e_tp_all_model_clean_xla_validation_20260715/phase-04/scalar-sv/attempt-01-preparation-20260715/ksc_t1000_preparation.json`,
  `b299cd38d0d9fc8c9b233855c0b3630e9d89aac3dce3d0fc4ff62745abaf09c8`.
- Actual CPU result:
  `docs/benchmarks/artifacts/contract_e_tp_all_model_clean_xla_validation_20260715/phase-04/scalar-sv/attempt-02-cpu-20260715/actual_result.json`,
  `a9721100af8a94e7f533656b9319ad6b00c99d6eba82eca2ecee794fb37988f2`.
- KSC CPU result:
  `docs/benchmarks/artifacts/contract_e_tp_all_model_clean_xla_validation_20260715/phase-04/scalar-sv/attempt-03-cpu-ksc-20260715/ksc_result.json`,
  `73a0c76f794fa6b8e0062dd0532c4b8122a638a6176fa5adab8c9ae6f5309114`.
- GPU probe:
  `docs/benchmarks/artifacts/contract_e_tp_all_model_clean_xla_validation_20260715/phase-04/scalar-sv/attempt-01-gpu-probe-20260715/probe.json`,
  `44abc5cc771bc705afc20d6e954bc4471124a28d956a32cb725b118ac8e317a9`.
- KSC trusted GPU result:
  `docs/benchmarks/artifacts/contract_e_tp_all_model_clean_xla_validation_20260715/phase-04/scalar-sv/attempt-01-gpu-ksc-20260715/ksc_result.json`,
  `a3b0ddf53e12dd64b25615d054e2339774c0157f452991748ac244abcd74438a`.

KSC GPU value/score are `-2284.098531972001` and
`[2.456831351828945,-4.420876400418747]`. Compile-and-first took `37.10 s`,
warm execution `16.18 s`, and allocator peak was `8,436,073,472` bytes. The
largest descriptive CPU/GPU absolute differences were `1.36e-12` for value,
`1.39e-12` for score, and `7.94e-13` per increment. These observations do not
establish cross-device equivalence.

Focused local checks passed: `35 passed, 2 dependency warnings`; compileall,
JSON parse/hash checks, source/graph gates, and `git diff --check` passed.

## Failed Attempt And Repair

The first Actual CPU attempt executed the full factory but the harness passed
nonfinite FD values to a utility that accepts only finite vectors. The attempt
is preserved with a structured failure record. The reporting-only repair emits
an explicit failed FD policy with endpoint validity. A fresh retry confirmed
the underlying row veto without changing any scientific setting.

## Decision And Inference Status

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| reject Actual SV Phase 4 candidate | fails same-scalar FD endpoint-validity gate | row-specific promotion veto | whether a separately designed chart family can cover a region | preserve result; future chart research requires a new reviewed phase | no rejection of XLA or the target model |
| retain KSC-SV engineering candidate | every CPU/GPU gate passes | clear at frozen center and endpoints | center-only and approximate observation target | enter cross-model regression after model phases | no exact-SV, HMC, or default claim |
| continue master program | no shared continuation veto fired | clear | predator loop implementation remains absent | execute reviewed Phase 5 repair | no all-model completion claim |

Hard veto evidence: Actual FD endpoint invalidity rejects that row's Phase 4
candidate. KSC has no hard veto at the stated center scope. Statistical ranking
is not applicable and unsupported. Descriptive CPU/GPU differences do not rank
devices or methods. Neither row is default-ready.

## Budget

Recorded wall time is below `0.10` CPU hours and `0.05` trusted GPU hours;
conservative campaign accounting charges `0.20 CPU core-hours` and `0.05 GPU
hours`. KSC consumed one full-horizon GPU attempt; Actual consumed none.
Remaining campaign budget is `95.42 CPU core-hours`, `31.94 trusted GPU-hours`,
three attempts for predator/SIR/other eligible rows, and two KSC attempts.

## Post-Run Red Team

The strongest alternative explanation for KSC success is favorable center-only
chart selection. No nonzero-radius parameter region was tested. The strongest
alternative explanation for Actual failure is a square chart with insufficient
local positivity margin, not a wrong score or compiler defect. A new chart
family could test that hypothesis, but doing so after seeing this result would
change the frozen candidate and is outside Phase 4.

## NEXT_PHASE_READINESS

| Clause | Status | Evidence |
| --- | --- | --- |
| result/check consistency | `PASS` | controlling hashes, structured gates, `35 passed` |
| legal row classifications | `PASS` | Actual rejected; KSC retained; no proxy substitution |
| exact identities | `PASS` | row/preparation/data/factory identities bound |
| assumptions audited | `PASS` | center-only and inherited hypotheses remain explicit |
| criteria/vetoes/nonclaims/fresh paths | `PASS` | Phase 5 draft binds predator boundaries |
| executable next commands | `PASS` | focused implementation/test ladder precedes `T=20` |
| no unresolved material boundary | `PENDING` | Phase 5 bounded review not yet complete |
| budget | `PASS` | remaining budget exceeds Phase 5 minimum plus reserve |

Overall: `PENDING_PHASE5_REVIEW`; Phase 4 becomes handoff-ready when the Phase
5 review converges.
