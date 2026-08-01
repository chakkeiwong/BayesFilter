# Phase 3: Scalar-SV Loop-Native Core Result

Date: 2026-07-15

Status: `PASS_CLOSED_HANDOFF_READY`

## Verdict

Actual non-Gaussian SV and KSC-SV now use the same loop-native TensorFlow
orchestration while retaining distinct observation targets, transforms,
preparations, teacher orders, and target-specific continuation windows. Both
rows pass center-scoped CPU/XLA value, increment, total-score, state, validity,
same-scalar FD, source-reachability, graph-growth, and fail-closed gates through
`T=100`.

This is a center-scoped finite-program engineering result. The registry has no
reviewed nonlinear parameter region. Therefore neither row has a nonzero-radius
parameter-region certificate, full-box readiness, HMC readiness, filtering
accuracy proof, or `T=1000` result.

## Implemented Repair

- added loop-native target continuation and fixed-state filter recursion for
  Actual SV and KSC-SV;
- retained static `T=1` and terminal dispatch while using `tf.while_loop` for
  intermediate time and continuation recursion;
- replaced dynamic repeat/tile Cartesian products with static
  broadcast/reshape;
- made transformed-SV wrappers graph-native by binding one validated immutable
  `StochasticVolatilitySSM` instead of reconstructing it inside traced calls;
- made shared tensor finiteness validation preserve eager `ValueError` behavior
  and use TensorFlow assertions under tracing;
- used one masked continuation-loop body for all positive lookaheads, avoiding
  zero-iteration reverse TensorLists while preserving the scalar; and
- added exact-factory source guards, historical-route rejection, immutable
  structured harness output, and finite off-center fail-closed controls.

No target law, transform, time order, chart selection rule, FD threshold, or
scientific comparator was changed.

## Evidence

| Check | Actual SV | KSC-SV |
| --- | ---: | ---: |
| certified horizons | `1,2,3,10,100` | `1,2,3,10,100` |
| `T=100` teacher/lookahead | `25/16` | `41/8` |
| `T=100` value | `-226.4735505123` | `-226.2889233996` |
| `T=100` score | `[1.0188616582,2.9593282725]` | `[0.7820828289,3.1248416595]` |
| maximum `T=100` same-scalar FD relative error | `1.09e-8` | `7.06e-8` |
| top-node ratio `T=100/T=3` | `1.0` | `1.0` |
| function-node ratio `T=100/T=3` | `1.0` | `1.0` |
| GraphDef-byte ratio `T=100/T=3` | `1.0012233` | `1.0010435` |
| finite off-center theta `[4,-0.91629...]` | invalid and poisoned | invalid and poisoned |

Controlling results:

- Actual SV:
  `docs/benchmarks/artifacts/contract_e_tp_all_model_clean_xla_validation_20260715/phase-03/scalar-sv/attempt-05-t100-harness-20260715/actual_result.json`,
  SHA-256 `6772e3191d0b417509243c86f22bb3d4d5560081f8319af9fc807180ccdd1416`.
- KSC-SV:
  `docs/benchmarks/artifacts/contract_e_tp_all_model_clean_xla_validation_20260715/phase-03/scalar-sv/attempt-05-t100-harness-20260715/ksc_result.json`,
  SHA-256 `3098c70b1494abcc530cd0e883eb0d67541b5ee73b14dea57cab29a0569c166a`.

Fresh `T=100` preparations:

- Actual SV SHA-256
  `e9395d07294f31f7c2422e333e1b86d661a23b3a688cbb0a756dcf7556142b48`;
  minimum weight `1.03e-5`, maximum condition number `1.43e4`.
- KSC-SV SHA-256
  `577b68ccb060a999d637fc105aefada0b898b988d916e92b44506746bd3759b6`;
  minimum weight `2.66e-5`, maximum condition number `5.69e4`.

The final cross-model focused suite passed: `98 passed, 2 dependency warnings`
in `81.55 s`. Separate final loop/source tests passed `23/23`; compileall, JSON
parse/hash, and `git diff --check` passed.

## Failed Attempts And Repairs

1. `T=1` parity tests received JSON empty lists with shape `[0]`; the fixture
   now restores the required `[0,4]` chart shapes.
2. The source guard rejected an unresolved local lambda; a named nested poison
   helper made exact reachability auditable.
3. First XLA trace failed because transformed-SV wrappers reconstructed eager-
   validated models inside the graph. Immutable base-model delegation repaired
   the same equations.
4. The shared and transformed-SV row-matrix helpers used eager `.numpy()` in
   traced calls. Dual eager/graph validation repaired the runtime boundary.
5. XLA reverse autodiff rejected a zero-iteration continuation loop with a
   zero-length TensorList. Static masked positive-iteration dispatch repaired
   it without changing the scalar.
6. Structured attempt 2 passed all numerical gates but failed graph growth
   because lookahead 1 statically removed the continuation loop. A common
   masked continuation body repaired topology; attempt 3 passed.
7. Skeptical close audit found the reviewed subplan omitted the master
   program's `T=100` and parameter-region decisions. The plan was amended,
   fresh `T=100` charts were generated, and attempt 5 passed. No parameter box
   was invented.

All failed artifacts are preserved. No trusted GPU work or full-horizon attempt
was consumed in Phase 3.

## Decision And Inference Status

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| retain Actual SV for center-only Phase 4 | all Phase 3 gates pass through `T=100` | clear at center and FD endpoints | no region certificate; `T=1000` untested | target-specific `T=1000` preparation and trusted GPU/XLA | no full-box/HMC readiness |
| retain KSC-SV for center-only Phase 4 | all Phase 3 gates pass through `T=100` | clear at center and FD endpoints | fresh chart conditioning and `T=1000` untested | target-specific `T=1000` preparation and trusted GPU/XLA | no exact-SV or equivalence claim |
| reject historical unrolled route for compilation | exact reachable-source guard rejects it | permanent engineering veto | retained only as finite parity oracle | regression only | no removal of historical evidence |

Hard veto screen: both candidates pass at the stated center scope. Statistical
ranking: not applicable; the deterministic differences do not support method
ranking. Default readiness: false. Descriptive target-reference gaps from prior
work are not promoted into this engineering gate.

## Budget

Phase 3 records conservatively `0.25 CPU core-hours`, zero trusted GPU-hours,
and zero full-horizon attempts. Remaining campaign budget is `95.62 CPU
core-hours`, `31.99 trusted GPU-hours`, and three full-horizon attempts per
eligible row.

## Post-Run Red Team

The strongest alternative explanation is favorable center-only chart
selection. The exact same chart may become invalid elsewhere, and nothing here
defines a scientifically justified parameter box. The `T=100` graph result
does establish bounded graph construction for the captured finite program, but
it does not prove `T=1000` compilation, memory feasibility, warmed performance,
or filtering accuracy. A trusted full-horizon invalid chart, nonfinite score,
fallback, or graph-growth failure rejects only that candidate/configuration and
triggers the bounded Phase 4 repair loop.

## NEXT_PHASE_READINESS

| Clause | Status | Evidence |
| --- | --- | --- |
| result/check consistency | `PASS` | two controlling hashes; `98 passed` |
| legal row classifications | `PASS` | only Actual/KSC center-scoped candidates scheduled |
| exact identities | `PASS` | row-specific preparation hashes and target policies |
| assumptions audited | `PASS` | `T=100` hypotheses and missing region stated |
| criteria/vetoes/nonclaims/fresh paths | `PASS` | drafted Phase 4 subplan |
| executable next commands | `PASS` | preparation then trusted GPU/XLA ladder |
| no unresolved material boundary | `PASS` | Phase 4 focused re-review found no material issue and returned `VERDICT: AGREE` |
| budget | `PASS` | available `95.62 CPU,31.99 GPU,3 attempts/row`; Phase 4 minimum plus reserve declared below availability |

Overall: `PASS`; automatic handoff to reviewed Phase 4 execution is ready.
