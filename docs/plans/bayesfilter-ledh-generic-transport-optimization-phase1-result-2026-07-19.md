# Generic LEDH Transport Optimization Phase 1 Result

Date: 2026-07-19  
Plan: `docs/plans/bayesfilter-ledh-generic-transport-optimization-master-plan-2026-07-19.md`  
Status: `PHASE1_CACHE_PROMOTED_PHASE2_BATCHING_REJECTED`

## Research Question And Decision

Question: can the shared LEDH forward-JVP transport reuse same-cloud pairwise
geometry for arbitrary parameter dimension without changing the finite
transport or Contract E total derivative, and does that improve GPU/XLA
steady-state time?

Decision: yes for an opt-in same-cloud cache. The cache is promoted as a generic
capability with the streamed route retained as the fallback. It is not enabled
universally because its dense tangent storage scales as `B*N^2*P`. A separate
exact parameter-axis batching candidate for the Contract E Cholesky reset was
rejected because it was slower on the paired witness.

## Implementation

- Added cached same-cloud softmin JVP support in
  `experiments/dpf_implementation/tf_tfp/resampling/annealed_transport_tf.py`.
- Cached one pairwise cost `[B,N,N]` and tangent `[B,N,N,P]` per fixed OT solve.
- Reused both through finite annealed Sinkhorn, terminal balancing, and final
  transport assembly.
- Threaded an opt-in `cache_same_cloud_geometry` flag through the canonical
  forward-JVP route and the versioned benchmark harness.
- Preserved the repository chunk policy, iteration counts, fixed finite target,
  row quotient, Contract E moment dependencies, and total tangent composition.
- Added generic parity tests for `B=2`, nontrivial geometry/payload dimensions,
  float32/float64, and `P=1,3,5,18`.

## Evidence Contract Result

| Gate | Result |
| --- | --- |
| Generic value/JVP parity | Pass: `9/9` cache tests |
| Existing reset certificates | Pass: `16/16` |
| Existing streaming tests | Pass independently: `12/12` |
| Canonical LGSSM tests | Pass independently: `16/16` |
| GPU/XLA smoke | Pass, one `StatelessWhile`, hard gates true |
| T=50 B=1 performance | Pass: median `3.013398 s` to `0.302396 s`, `9.965x` |
| T=50 B=16 performance | Pass: median `25.921660 s` to `2.591018 s`, `10.004x` |
| 8 GiB memory cap | Pass |
| Parameter-axis reset batching | Reject: `0.302396 s` to `0.413242 s` |

The B=1 cache comparison used five synchronized warm calls. The B=16
comparison used three. Runtime results are descriptive, not a statistical
superiority claim.

## Numerical Comparison

| Witness | Value difference, cached minus baseline | Relative score L2 difference | Maximum absolute score-coordinate difference |
| --- | ---: | ---: | ---: |
| `T=50,N=1024,B=1,P=5` | `-3.0517578125e-05` | `3.2395e-05` | `2.1362e-04` |
| `T=50,N=1024,B=16,P=5` | `0.0` | `1.7771e-05` | `9.2506e-05` |

All paired arms passed finite, bitwise within-arm replay, chart, marginal,
reset, work-accounting, and XLA graph gates. The cross-arm differences are
consistent with float32/TF32 evaluation-order changes. Exact-arithmetic parity
is established by the direct generic tests; bitwise cross-arm equality is not
claimed.

## Memory Boundary

The cache requires approximately

`scalar_bytes * B * N^2 * (1 + P)`

bytes for the cost and its tangent, before compiler liveness and other filter
state. This supports any positive `P` algebraically but is not universally
memory-feasible. No arbitrary memory threshold was introduced. Every model,
horizon, batch, particle, dtype, and parameter scope must preserve its own
memory/performance witness before enabling the cache.

## Artifacts

Artifact root:
`docs/benchmarks/artifacts/ledh_generic_transport_optimization_20260719/`

Primary artifacts:

- `lgssm_t50_n1024_b1_baseline_attempt02.json`
- `lgssm_t50_n1024_b1_cached_attempt02.json`
- `lgssm_t50_n1024_b16_baseline_attempt01.json`
- `lgssm_t50_n1024_b16_cached_attempt01.json`
- `lgssm_t50_n1024_b1_cached_batched_reset_attempt03.json`
- `smoke_t2_n128_cached_attempt01.json`

The JSON artifacts record commit
`9fd0b97fccd8ba216407eb8ff0a727bdc5a2709b`, trusted RTX 4080 SUPER GPU
execution, XLA JIT, TF32/float32, seeds, controls, timing, memory, graph
identity, diagnostics, plan path, and result paths. The campaign run manifest
adds the verified TensorFlow `2.19.1` conda environment and artifact checksums:
`docs/benchmarks/artifacts/ledh_generic_transport_optimization_20260719/run_manifest.md`.

## Known Separate Gap

Running the legacy float32 production tests and float64 VJP reference tests in
one pytest process exposes mutable module-global `DTYPE` contamination in the
historical reverse-mode helpers. The affected suites pass independently. This
is an engineering test-order defect, not evidence against the cache or the
forward-JVP mathematics, and it was not expanded into this scoped change.

## Decision Table

| Decision | Primary criterion | Veto diagnostics | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Retain generic opt-in geometry cache | Pass | No parity, XLA, memory, marginal, reset, or replay veto | Cross-model runtime and memory behavior | Add route-specific switches and paired witnesses as nonlinear routes adopt the shared forward-JVP core | No universal speedup, production readiness, HMC readiness, or leaderboard completion |
| Reject batched reset JVP | Correct but slower | Performance promotion gate failed | Other hardware/model shapes may differ | Keep serial reset JVP; reconsider only with a new scoped benchmark | No claim that parameter batching is inherently bad |

## Post-Run Red Team

Strongest alternative explanation: XLA may optimize the dense cached expression
especially well for the measured one-tile `N=1024` shapes, so the observed
speedup need not transfer to larger multi-tile or much larger-`P` scopes.

Result that would overturn the decision: a route-specific paired test showing
memory-cap breach, failed numerical gates, or slower warm execution for the
cache. Such a result should disable the cache for that scope, not reject the
generic algorithm or alter the finite transport target.

Weakest evidence: no nonlinear model route currently calls the same fused
forward-JVP primitive, so cross-model speedup remains not checked.
