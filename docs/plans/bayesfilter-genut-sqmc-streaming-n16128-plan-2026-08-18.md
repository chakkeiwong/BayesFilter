# GenUT SQMC Exact Streaming and N=16128 Plan

Date: 2026-08-18  
Status: `REVIEWED_FOR_BOUNDED_EXECUTION`

## Research Intent Ledger

| Field | Frozen decision |
|---|---|
| Main question | Can the current Austria-SIR `T=20` SQMC trust-region filter be evaluated at `N=16128` by changing only the evaluation order of its finite multiplicative Sinkhorn reset, while preserving the dense algorithm's value, analytical score, validity diagnostics, and route semantics? |
| Baseline | The completed dense trust-region route in `claim_attempt01`, solver `genut_column_scaled_lm_smooth_rms_trust_v1`, particle counts `1008,2016,4032`, four SQMC variants, and seeds `97701..97716`. |
| Candidate | Exact tiled evaluation of the same squared-distance scale, kernel action, 16 alternating left/right multiplicative updates, row quotient, barycentric numerator, raw marginals, and post-quotient marginals. No dense `[N,N,d]`, kernel, coupling, or quotient-coupling tensor may be retained. |
| Target run | Austria-SIR `T=20,N=16128`, seed `97701`, all four SQMC variants. A larger seed campaign is not authorized by this plan. |
| Chunk policy | Repository-owned `dpf_transport_exact_divisor_cap3000_v1`: `N=1008 -> K=1008`, `N=4032 -> K=2016`, `N=16128 -> K=2688`. Selection occurs before TensorFlow tracing. |
| Score | Preserve the repository analytical all-parent backward score with child block size `126`. Streaming changes only the reset transport evaluation order. |
| Primary criterion | Dense/streamed parity on the same fixed finite program, followed by finite/program-valid GPU/XLA `N=16128` rows with the trust-region solver identity and required marginal/ancestry gates. |
| Promotion criterion | None. Successful execution establishes feasibility and finite-program parity in the tested scope only. |
| Continuation veto | Wrong finite update equations, failed dense parity beyond declared numerical tolerances, altered observations/input hashes/event order, nonfinite output, invalid reset, marginal gate failure, state-map saturation, ancestry invariant failure, source mismatch, artifact corruption, or campaign budget exhaustion. |
| Repair trigger | XLA compilation/resource failure caused by tile state, TensorArray layout, or bounded harness integration while the finite equations and campaign scope remain unchanged. |
| Nonclaims | No exact SIR observed-data score, no formal SQMC rate, no algorithm ranking, no 16-seed `N=16128` conclusion, no HMC/NeuTra/default readiness, and no claim that the annealed streaming OT route is equivalent to this multiplicative finite program. |

## Evidence Contract

| Role | Requirement |
|---|---|
| Engineering parity | Direct dense-versus-streamed reset tests preserve particles and every shared marginal diagnostic. A full small Austria route preserves value, analytical score, final particles, and validity. |
| Structural memory | Source inspection/test rejects full-`N` pair-difference, kernel, coupling, and quotient-coupling construction in the streaming helper. Retained tile state is `O(B K^2 d + B N d)`. |
| GPU/XLA | One visible GPU, TF32 enabled, XLA required, memory growth configured and verified before initialization, output device recorded. |
| N=4032 replay | Streamed rows use the same observations, controls, seeds, point sets, and hashes as the dense artifact. Differences are reported before `N=16128`. |
| N=16128 feasibility | All four seed-`97701` rows must be finite/program-valid with `K=2688`, score block `126`, zero saturation, valid route identity, required ancestry invariants, and TV residual at or below `1e-4`. |
| Artifact | Fresh versioned output root with raw rows, dense comparison, source hashes, environment/device/memory manifest, wall time, and terminal result/reset memo. |

## Exact Streaming Definition

For particles `x_i`, weights `w_j`, uniform row target `u_i=1/N`, and
`c=max(mean_ij ||x_i-x_j||^2, 1e-3)`, preserve the dense kernel

```text
K_ij = exp(-||x_i-x_j||^2 / (c * epsilon)).
```

Starting from `l=r=1`, preserve exactly `sinkhorn_steps + balance_steps`
alternating updates:

```text
l_i <- u_i / (sum_j K_ij r_j + 1e-7)
r_j <- w_j / (sum_i K_ij l_i + 1e-7)
```

After the last update, stream tiles again to compute row mass, barycentric
numerator, raw column mass, and row-quotient column mass. The full kernel and
coupling are never materialized. Each tile uses the dense baseline's exact
subtraction-square-reduction expression. This transiently materializes one
`[K,K,state]` tile, never the full `[N,N,state]` tensor. For `N=16128` and
policy `K=2688`, one FP32 difference tile is about `0.48 GiB`.

The existing `ledh_contract_e_streaming_tf` machinery is an implementation
reference for chunking, stable tensor shapes, marginal accumulation, and
tests. Its annealed log-potential solver is not substituted for the equations
above.

## Default and Assumption Audit

| Choice | Provenance | Justification | Failure mode | Early diagnostic | Status |
|---|---|---|---|---|---|
| `K` from exact-divisor policy | Active repository DPF transport policy | Required active transport policy and exact tiling | caller-selected small tiles silently change policy/performance | selector and manifest identity tests | reviewed default |
| Multiplicative finite updates | Current dense `_restore_cloud_primal` | Required same-program comparison | reuse of annealed potentials changes finite output | equation/source audit and dense parity | frozen baseline |
| FP32/TF32 GPU/XLA | Completed dense artifact and repository execution default | Same comparison backend | reduction-order drift can amplify over `T=20` | reset parity, then full-route `N=1008/4032` replay | baseline |
| Toleranced, not bitwise, parity | Tiled reductions change summation order | Equality of the finite mathematical operations is the target | tolerance too loose hides route drift | report absolute/relative errors for particles, value, and score | hypothesis with veto bounds |
| `N=16128`, one seed | User request plus bounded first feasibility rung | Answers whether the route can run without implying variance estimates | one seed cannot support variance/ranking claims | explicit nonclaim | feasibility target |
| Four variants | Existing comparison set | Exposes ancestry-specific integration defects | campaign may be expensive | compile/one-row stop before remaining variants if budget threatens | bounded diagnostic |
| Seed batch size 1 initially | Current analytical score and memory uncertainty | Isolates per-row memory before optimization | underuses GPU and increases wall time | allocator/runtime record; microbatch is a later optimization | convenience baseline |

## Parity Tolerances

The implementation first measures error at FP64 and FP32. Initial veto bounds
are deliberately tight for a deterministic evaluation-order change:

- direct FP64 reset: `atol=rtol=2e-10`;
- direct FP32 reset/XLA: `atol=rtol=2e-5`;
- full small Austria value and score: `atol=rtol=5e-4`;
- `N=4032` dense-artifact replay: report differences; veto `|value delta| >
  0.05` or any score-component relative delta above `5e-3` when scaled by
  `max(1, |dense score|)`.

These are same-program numerical tolerances, not scientific error margins. If
the direct tests show a smaller stable envelope, the result memo reports it;
the bounds are not relaxed after observing a failure.

## Skeptical Plan Audit

1. **Wrong baseline:** rejected. The comparator is the completed dense
   multiplicative GenUT reset, not the separate annealed streaming OT route or
   the batch GenUT finite-score implementation.
2. **Proxy promotion:** rejected. Memory, runtime, ESS, and finite execution are
   feasibility diagnostics; they do not establish score correctness or method
   superiority.
3. **Hidden objective change:** guarded by preserving cost scale, update count,
   update order, `1e-7` denominator floor, row quotient, and trust-region
   correction, with direct and full-route parity tests.
4. **Unfair comparison:** observations, route controls, input hashes, seed,
   event order, score recursion, TF32/XLA, and GPU class remain fixed.
5. **Batch confusion:** the existing batch GenUT route is dense and computes a
   different finite-program score. It is not used as parity evidence. Seed
   microbatching is deferred until scalar streamed correctness is established.
6. **Missing stop condition:** `N=16128` is not launched until unit/XLA parity
   and one streamed `N=4032` replay pass. OOM or invalidity at `N=16128` is
   retained as a feasibility result; the particle count is not silently lowered.
7. **Misleading pass:** a finite `N=16128` result could still compute the wrong
   SIR score. Existing oracle and finite-program correctness vetoes remain.

Audit decision: `PASS_AFTER_REJECTING_ANNEALED_SOLVER_SUBSTITUTION`. The plan
answers the requested memory/feasibility question while preserving the actual
dense finite program.

### Execution audit addendum after the first N=4032 replay

The first streamed `N=4032`, repaired-permutation, seed-`97701` row was valid
but failed the declared comparison tolerance against the archived dense row.
That comparison also exposed a provenance flaw in the original ladder: the
archived and current source hashes differ for the transport/reset module, route
wrapper, and harness. The archived row therefore cannot serve as the primary
same-program parity gate. It remains a descriptive historical comparison.

Before `N=16128`, execution now requires both:

1. compiled FP32 multi-tile dense/streaming unit parity from the same current
   source and inputs; and
2. a current-source dense/streaming route replay with matching input hashes,
   controls, compiler/device settings, and source hashes.

The original numerical tolerances remain frozen. A failure of the current-source
comparison is a continuation veto. A pass does not erase the historical drift;
the terminal result must report both comparisons and must not attribute the
historical difference to streaming.

The fresh current-source dense `N=1008` replay then completed on GPU1. It
matched the input hashes and controls but differed from the earlier streamed
one-block row. The direct transport diagnostic showed that the one-block
streaming wrapper changed the XLA reduction graph even though its equations
were algebraically identical. The implementation therefore treats the active
`K=N` policy case as the dense arithmetic baseline; genuinely tiled cases
(`N>3000`) retain the bounded streaming kernel. A new exact one-block equality
unit test covers this rule. The `N=16128` gate remains closed until the fresh
streamed `N=1008` replay passes against the new baseline.

## Execution Ladder and Budget

1. Refactor dense Sinkhorn row-quotient work behind a structured result and add
   an exact streaming implementation selected explicitly by plan mode.
2. Add focused tests for direct reset parity, diagnostics parity, route
   identity, chunk-policy enforcement, source memory structure, full Austria
   parity, and CPU/XLA compilation.
3. Run CPU-hidden focused tests and a GPU/XLA small smoke.
4. Run one streamed `N=4032` repaired-permutation seed and compare it with the
   completed dense row for seed `97701`. If it passes, run the other three
   variants at that seed only if needed to expose route integration defects.
5. Run `N=16128`, seed `97701`, all four variants on the available preferred
   GPU. Use a fresh output attempt and checkpoint every row.
6. Analyze value/score differences, validity, ancestry, ESS, TV residual,
   allocator peak, compilation/steady time, and write the terminal memo.

Compute ceiling: 45 minutes for implementation tests and small smokes, 90
minutes for `N=4032` streamed parity, and 8 GPU hours for the four `N=16128`
rows. At most two localized infrastructure repairs are permitted without
changing equations, particle count, variants, seed, or evidence gates.

## Planned Artifacts

Output root:
`docs/benchmarks/artifacts/genut-sqmc-streaming-n16128-20260818/`

Terminal note:
`docs/plans/bayesfilter-genut-sqmc-streaming-n16128-result-2026-08-18.md`
