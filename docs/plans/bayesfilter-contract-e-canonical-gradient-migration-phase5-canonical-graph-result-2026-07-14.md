# Phase 5 Result: Canonical One-Graph LGSSM Value And Gradient

Date: 2026-07-14

Program ID: `contract-e-canonical-gradient-migration-20260713`

Status:
`CANONICAL_ONE_GRAPH_ZERO_ULP_ENGINEERING_CERTIFIED_NUMERICAL_SCIENTIFIC_PROMOTION_BLOCKED`

## Outcome

One repository-owned TensorFlow concrete value-and-score graph now contains a
finite LGSSM LEDH primal traversal and a separate five-parameter manual-JVP
traversal. The manual traversal matches TensorFlow forward autodiff of the
private primal traversal at `0 ULP` on the checked fixtures. This is executed
pointwise evidence, not a claim that the manual traversal is mechanically
generated from the primal or cannot drift after future edits.
Candidate-dependent stationary initialization is inside the callable. The
recursion is transition-first, corrected logits are normalized without a
probability floor, active resets use the Phase 4 row quotient followed by
Contract E--Chol, and inactive resets carry normalized weights. The parameter
direction axis is final and has size five.

The primary predeclared derivative gate passed exactly: the manual per-batch
and aggregate scores are bitwise identical (`0 ULP`) to TensorFlow forward
autodiff of the same private primal core on the frozen binary64 fixture. A
separate CPU-XLA certificate used one concrete value-and-score callable at the
center and all 30 finite-difference endpoints. Repeated center values were
bitwise identical, all charts were valid, and every endpoint branch identity
matched the center after a branch-only fixture repair.

This is an engineering certificate for the checked tiny finite program. It is
not a Kalman-equivalence, numerical-adequacy, production-admission, nonlinear,
HMC, leaderboard, or release result.

## Claimed And Computed Quantities

| Item | Classification |
| --- | --- |
| Claimed Phase 5 target | Total derivative of the literal finite Contract E LGSSM value program returned by one owned callable |
| Quantity computed | Separate manual five-direction JVP traversal and TensorFlow `ForwardAccumulator` JVP of the private primal traversal, packaged in one concrete value-and-score graph; same concrete graph called for FD |
| Equality verdict | Correct on the checked tiny binary64 fixtures: per-batch and aggregate manual-versus-autodiff comparisons are `0 ULP`; future-drift immunity is not established |
| FD relation | Central differences approach the same score with the expected approximately fourfold reduction after halving the step; explanatory only |
| Kalman relation | Not checked in Phase 5 |
| Production relation | Not checked for the full `T=50,N=10000` graph |

## Exact Derivative Evidence

The frozen `B=2,N=4,T=2,d=3,p=5` v2 fixture produced

```text
objective = -5.333391762198579
score = [-0.4468525807379741,
         -0.22403176890798465,
          0.05615837148813346,
         -3.5522223205276537,
         -4.746981132539255]
```

Every one of the ten per-batch score components and all five aggregate score
components had ULP distance zero against forward autodiff. Both batch charts
were valid and the minimum row masses were positive. The focused suite also
checks a separate `B=1,T=1` active-reset fixture, mixed active/inactive reset
history, exact initialization dependence, floor-free normalization JVP/VJP,
time-sum/batch-mean aggregation, an executable invalid physical chart, and
all-five-parameter objective sensitivity.

The exact certificate is
`docs/plans/logs/contract-e-canonical-gradient-migration-2026-07-13/phase5/exact-same-core-derivative-v2.json`
with SHA-256
`f6e18da00197425ef8850f23ddc815474397ff093c000920bb7e6b671e578c9d`.

## Same-Callable FD Evidence

The CPU-XLA certificate has exactly one concrete value-and-score callable and
executes score computation at every endpoint. Its hard checks are all true:

- repeated center bitwise identity;
- one concrete value-and-score callable;
- valid center and endpoint charts; and
- identical center/endpoint branch hashes.

At the finest frozen step, `h=1/512`, the componentwise symmetric relative
differences are approximately:

| Parameter | Relative difference |
| --- | ---: |
| `phi1` | `7.45e-6` |
| `phi2` | `3.03e-6` |
| `phi3` | `1.97e-5` |
| `q_scale` | `2.57e-6` |
| `r_scale` | `1.22e-6` |

These values are descriptive only. They neither define nor replace the exact
same-core gate, and the owner-directed `0.05*sqrt(p)` FD rule is not a Kalman
or general gradient-accuracy tolerance.

The certificate is
`docs/plans/logs/contract-e-canonical-gradient-migration-2026-07-13/phase5/cpu-xla-same-callable-certificate-v2.json`
with SHA-256
`20ec133bc5aee47f5daf3dc54d4c3593189202b1c305640cbd30b5e33b4ca709`.

## Failed Fixture And Repair

The original v1 fixture passed exact same-core derivatives and all chart
checks, but failed the predeclared FD branch-identity gate. At active reset
`batch=1,time=1`, the two leading scaled-geometry values were separated by
only about `7.27e-4`, less than the largest FD step `1/128`. Several endpoints
therefore changed the maximum index.

Before any v2 objective or score was read, a branch-only screen tried positive
dyadic changes to one prepared transition-noise entry. The smallest tested
change, `1/64`, passed every branch and chart endpoint check, changing the
entry from `1/8` to `9/64`. The original fixture and failed artifact remain
preserved and are not reinterpreted:

- v1 artifact SHA-256:
  `c39d3f6dae0ba3702895f6bf641f6a6fefa55418ae7e106f6f0318bc8c2fddd5`;
- v2 fixture SHA-256:
  `f6b6e2895208d7cd5cba0f57b05d4de7fb0de79e50ba62b7e6c70b06879942f4`.

This repair establishes branch-stable evidence for this fixture only. It is
not evidence of a general branch margin.

## Implementation Repairs

The exact gate required the manual tangent to mirror TensorFlow's literal
finite operation sequence, not merely an algebraically equivalent real-number
formula. The final implementation mirrors:

- TensorFlow's transposed registered Cholesky-gradient operation order;
- both literal triangular solves in `cholesky_solve`;
- the triangular-solve JVP for `LQ^{-1}`;
- stationary-initialization square/subtract/sqrt/division operations;
- Gaussian-density solve/product/reduction operations; and
- five explicit final-axis stabilized `reduce_logsumexp` reductions.

These were target-preserving finite-operation repairs made to the frozen
`0 ULP` certificate. No post-result tolerance was introduced.

## Checks

- Final CPU-hidden Phase 0-5 compatibility union:
  `150 passed, 2 warnings in 100.20s`.
- Final focused Phase 5 suite before the union:
  `8 passed, 2 warnings in 58.89s`.
- Python compilation: passed.
- JSON parsing, hashes, source prohibitions, and scoped `git diff --check`:
  passed.
- CPU-XLA certificate: passed with `jit_compile=true` and GPU deliberately
  hidden.
- No Phase 5 GPU, HMC, nonlinear, leaderboard, or release run occurred.

The warnings are the existing TensorFlow Probability `distutils.version`
deprecations.

## Unresolved Promotion Blockers

| Blocker | Status |
| --- | --- |
| General dense/autodiff/chunk error adequacy | Unresolved |
| Row-mass and finite-Sinkhorn marginal adequacy | Unresolved |
| Residual centering, mean, covariance, ridge, and conditioning adequacy | Unresolved |
| Full-time `T=50,N=10000` GPU/XLA/TF32 feasibility | Unresolved |
| Exact Kalman value and gradient equivalence | Not checked |
| Justified Phase 8 gradient-equivalence margin | Not frozen |
| Canonical v2 registration or admission | Factory remains empty |
| Nonlinear state/support validity | Not checked |
| HMC, leaderboard, default, or release readiness | Not established |

## Decision Table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Certify tiny one-graph packaging and derivative wiring | Passed at `0 ULP` on checked fixtures | No graph, initialization, normalization, reset, chart, or branch veto remains on those fixtures | General numerical behavior and future manual/primal drift | Mechanical historical-route cleanup | Mechanical derivation, Kalman, or production correctness |
| Treat FD as corroborating evidence | Descriptive convergence observed | Identity and branch prerequisites passed | Truncation/roundoff outside fixture | Preserve ladder; do not promote it | Statistical confidence or oracle agreement |
| Admit canonical v2 | Blocked | Factory is empty and later gates are open | Full-time and scientific evidence | Keep fail closed | Default/HMC/leaderboard readiness |

## Inference-Status Table

| Inference | Status |
| --- | --- |
| Hard veto screen | Passed for exact tiny same-core derivative and CPU-XLA same-callable identity |
| Statistically supported ranking | None; no stochastic method comparison ran |
| Descriptive-only differences | FD ladder values, timings, masses, and tiny-fixture magnitudes |
| Default-readiness | Not established |
| Next evidence needed | Fail-closed historical cleanup, documentation reconciliation, then a separately justified Kalman oracle design |

## Engineering, Numerical, And Scientific Ledgers

| Ledger | Phase 5 verdict |
| --- | --- |
| Engineering correctness | Passed pointwise for the checked canonical tiny primal/manual-JVP traversals packaged in one concrete graph |
| Numerical adequacy | Blocked outside the exact fixture certificates |
| Scientific interpretation | Kalman agreement and stochastic accuracy not checked |

## Artifacts

- `bayesfilter/highdim/ledh_contract_e_canonical_lgssm_tf.py`;
- `tests/highdim/test_ledh_contract_e_canonical_lgssm_phase5.py`;
- the v1, v2, and one-step frozen fixtures;
- the exact-derivative and same-callable certificate harnesses;
- structured artifacts, manifest, and focused-check record under the Phase 5
  log directory; and
- the branch-repair note.

## Phase 6 Handoff

Phase 6 may perform only mechanical fail-closed cleanup. It may remove obsolete
raw default/admission outcomes, require explicit historical-diagnostic opt-in,
emit `historical_raw_barycentric_diagnostic_only`, and test that no Contract E
failure falls back to raw reset. It may not register the Phase 5 callable,
change historical mathematics, begin nonlinear canonical implementation, or
claim that this tiny certificate resolves any scientific blocker.

## Post-Run Red Team

Strongest alternative explanation: exact agreement was obtained because the
manual JVP was deliberately written to reproduce TensorFlow's finite operation
order on a very small smooth fixture. That explanation is compatible with the
result and is why the conclusion is narrowly an engineering certificate.

What would overturn the close: a reproducible nonzero ULP on the frozen inputs,
a hidden value-only FD trace, prepared-input drift, a branch mismatch, or raw
reset reachability from the owned callable.

Weakest evidence: extrapolation. Nothing in Phase 5 establishes full-time,
float32/TF32, Kalman, nonlinear, or HMC behavior.
