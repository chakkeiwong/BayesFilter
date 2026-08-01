# Phase 4 Result: Streaming Quotient And Contract E Composition

Date: 2026-07-14

Program ID: `contract-e-canonical-gradient-migration-20260713`

Status:
`ENGINEERING_COMPOSITION_AND_PRODUCTION_SHAPE_FEASIBILITY_PASSED_NUMERICAL_SCIENTIFIC_PROMOTION_BLOCKED`

## Outcome

The streaming finite transport now computes the canonical row quotient
`Y=Q/M`, not the unnormalized numerator `Q`. Its analytic JVP carries `dQ,dM`
and applies

```text
dY = (dQ - Y*dM)/M,
```

while its VJP supplies both

```text
barQ = barY/M,
barM = -sum(barY*Y)/M.
```

No mass floor, clip, stop-gradient, second transport pass, or production
`N x N` matrix was introduced. A constant-one payload lets the existing
streaming kernel emit numerator and mass in one pass. The generic transport VJP
now separates geometry dimension from payload dimension, enabling the augmented
`d+1` payload without corrupting geometry cotangents.

The complete local Contract E composition adds the transported source path to
the direct source-moment path and converts the direct probability-weight
cotangent before adding it to the normalized-log-weight transport cotangent:

```text
G_X_total = G_X_direct + G_X_transport
G_logw_moment = w * G_w_probability
G_logw_total = G_logw_transport + G_logw_moment.
```

Phase 5 still owns the corrected-logit normalization pullback, exactly once.

## Claimed And Computed Quantities

| Item | Classification |
| --- | --- |
| Claimed engineering target | Finite streaming `Y=Q/M` plus complete local fixed-ridge Contract E direct-and-transport JVP/VJP composition |
| Quantity computed | Exact quotient identities and duality; small-fixture direct/transport path decomposition; descriptive dense/autodiff/chunk comparisons; CPU-XLA wrappers; one trusted-GPU production-shape forward and one repaired analytic VJP |
| Equality status | Quotient formula and checked dualities are correct; general dense/autodiff/chunk adequacy remains not checked to a justified bound |
| Production-shape feasibility | Supported for the selected one-step forward/VJP graphs at `B=1,N=10000,d=3`, float32, TF32, XLA, chunks `1024` |
| Numerical/scientific promotion | Blocked by row/Sinkhorn/chunk and inherited reset-adequacy requirements |
| Full filter/Kalman relation | Not computed in Phase 4 |

## Root Causes And Repairs

| Defect | Root cause | Repair | Evidence |
| --- | --- | --- | --- |
| Streaming cloud was wrong relative to Contract E target | Existing helper accumulated `Q` and `M` but returned `Q` unchanged | Augment payload with one, return explicit `Q,M,Y`, and veto invalid charts | Nonunit masses `0.684` to `1.465` on frozen local chart; exact quotient tests |
| Streaming derivative omitted quotient dependence | JVP propagated only `dQ`; VJP accepted only `barQ` | Carry `dM` and `barM` through the same streaming pass/pullback | Both paths nonzero; two-direction duality difference exactly zero |
| Augmented VJP width mismatch | Existing VJP reused geometry width for payload-adjoint buffers | Split `geometry_dim` and `payload_dim` | Unequal-width `d` versus `d+1` autodiff regression |
| Probability/log-weight coordinate risk | Direct reset weight cotangent was in probability coordinates | Convert with `w * G_w_probability` before addition | Exact composition identities and additive-constant test |
| GPU analytic VJP compiler abort | XLA GEMM-fusion autotuner could not combine layouts for a `10000x3` by `3x3` Contract E covariance pullback dot | Replace only that VJP-local row action by the algebraically identical broadcast multiply/reduction | `134 + 7 + 1` CPU checks; unchanged persisted diagnostics; final trusted-GPU VJP exit zero |

## Local Evidence

- Exact standalone quotient identities passed with no tolerance.
- Mass tangent and mass cotangent are nonzero.
- Two JVP/VJP duality differences are exactly `0.0` in binary64.
- Direct and transport source paths are separately nonzero.
- Direct and transport weight paths are separately nonzero.
- Phase 3 exact certificates and Phase 0-3 compatibility: `134 passed`.
- Phase 4 eager/reference checks: `7 passed`.
- Isolated Phase 4 CPU-XLA wrapper check: `1 passed`.

The observed general differences are binary64-roundoff scale but remain
descriptive because no justified kernel or chunk-accumulation error bound was
frozen. The local artifact status remains:

```text
EXACT_QUOTIENT_AND_DUALITY_CHECKS_PASSED_GENERAL_DENSE_AUTODIFF_CHUNK_PARITY_INCONCLUSIVE
```

## Production-Shape Feasibility

Trusted device: NVIDIA RTX 4080 SUPER, TensorFlow `2.19.1`, CUDA build `12.4`,
compute capability `8.9`, float32, TF32 enabled, XLA JIT.

| Graph | Hard-veto outcome | Peak GPU allocator | Compile/execute | Additional evidence |
| --- | --- | ---: | ---: | --- |
| Forward | Passed; finite positive masses and Contract E chart | `67,960,576` bytes | `6.907 s` | Warm `0.175 s`; repeated host diagnostics bitwise equal |
| Analytic VJP after repair | Passed; valid quotient chart and all cotangents finite | `84,859,392` bytes | `19.556 s` | Direct/transport source and weight paths nonzero |

This supports one-step production-shape graph feasibility. It does not prove
full-time feasibility, performance superiority, or numerical correctness.

## Failed Attempt Discipline

Two failures occurred before numerical execution and were repaired visibly:
repository-root import setup and an invalid VJP diagnostic-key lookup. They are
not candidate evidence.

The first actual compiled VJP attempt aborted in XLA GEMM-fusion autotuning with
exit `134`. That is real implementation/feasibility evidence. The HLO localized
the compiler failure to `_uniform_covariance_vjp`; the target-preserving repair
was independently reviewed and fully revalidated before the second and final
production-shape attempt passed.

## Unresolved Promotion Blockers

| Blocker | Status |
| --- | --- |
| Row-mass adequacy | Unresolved promotion blocker |
| Row marginal finite-Sinkhorn convergence | Unresolved promotion blocker |
| Column marginal finite-Sinkhorn convergence | Unresolved promotion blocker |
| Chunk accumulation error budget | Unresolved promotion blocker |
| Residual-design centering error requirement | Inherited unresolved promotion blocker |
| Mean-restoration error requirement | Inherited unresolved promotion blocker |
| Executed-kernel ridged-identity backward-error requirement | Inherited unresolved promotion blocker |
| Raw ridge-bias scientific requirement | Inherited unresolved promotion blocker |
| Conditioning/downstream-error budget | Inherited unresolved promotion blocker |
| Ridge magnitude/domain adequacy | Inherited unresolved promotion blocker |

Observed mass or residual magnitudes do not create post-result thresholds.

## Decision Table

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Close Phase 4 local engineering composition | Pass | No quotient, duality, coordinate, source-boundary, or CPU-XLA veto remains | General numerical error | Begin reviewed Phase 5 one-graph engineering work | General derivative accuracy |
| Support production-shape forward/VJP feasibility | Pass for selected graphs | Trusted GPU/XLA/TF32, chart, finite, and observed-memory vetoes pass | Full-time graph and wider fixtures | Preserve measured artifacts; do not generalize | Production readiness |
| Pass numerical/scientific promotion | Blocked | Ten adequacy requirements unresolved | Sinkhorn, chunk, ridge, and downstream error | Resolve only under pre-result criteria | Reset validity/admission |
| Register or admit Contract E v2 | Ineligible | Factory remains empty | Phases 5-9 incomplete | Keep fail closed | Default/HMC/leaderboard readiness |

## Inference-Status Table

| Inference | Status |
| --- | --- |
| Hard veto screen | Exact quotient and duality checks pass; CPU-XLA passes; selected production-shape forward/VJP execute on trusted GPU with valid chart and finite outputs/cotangents. |
| Statistically supported ranking | None; no stochastic method comparison ran. |
| Descriptive-only differences | Dense/autodiff/chunk differences, mass ranges, residuals, cotangent magnitudes, timings, and memory. |
| Default-readiness | Not established; factory remains empty and promotion blockers remain. |
| Next evidence needed | A single canonical callable owning candidate initialization, likelihood, corrected weights, quotient transport, Contract E reset, and JVP/FD primal identity. |

## Engineering, Numerical, And Scientific Ledgers

| Ledger | Phase 4 verdict |
| --- | --- |
| Engineering correctness | Passed for bounded quotient/composition identities and checked local derivatives |
| Production-shape feasibility | Passed for one selected forward and analytic VJP graph |
| Numerical adequacy | Blocked |
| Scientific interpretation | No claim evaluated |

## Artifacts

- `bayesfilter/highdim/ledh_contract_e_streaming_tf.py`;
- narrow generic-payload repair in
  `experiments/dpf_implementation/tf_tfp/resampling/annealed_transport_tf.py`;
- narrowed XLA repair in `bayesfilter/highdim/ledh_contract_e_reset_tf.py`;
- `tests/highdim/test_ledh_contract_e_streaming_phase4.py`;
- local structured diagnostics and frozen fixture;
- complexity/graph audit;
- trusted-GPU forward/VJP artifacts, fatal-attempt record, focused checks, and
  manifest under the Phase 4 log directory; and
- three bounded review records.

## Phase 5 Handoff

Phase 5 may begin local engineering work. It must build one callable whose
returned primal is the literal target for its JVP and every FD center/endpoint.
It must reconstruct candidate-dependent initialization inside that callable,
apply corrected-logit normalization once, consume `valid_chart` as a hard veto,
use only Contract E-Chol, and preserve fixed residual/ridge prepared-input
identity. The particle-only Phase 4 convenience wrapper is not an admission
gate.

GPU/full-filter, Kalman, nonlinear, HMC, artifact admission, and leaderboard
claims remain outside Phase 5 local entry.

## Post-Run Red Team

Strongest alternative explanation: the one smooth production-shape fixture is
unusually easy. That is why it establishes feasibility only and cannot clear
row/Sinkhorn/reset adequacy.

What would overturn the engineering close: a reproducible quotient identity or
duality failure, a coordinate double projection, hidden dense production state,
or source drift without artifact regeneration.

Weakest evidence: general numerical accuracy. The binary64-scale descriptive
differences are not promoted into an acceptance threshold.
