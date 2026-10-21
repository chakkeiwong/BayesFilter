# Repairing native batching in the SIR likelihood and analytical score

## Current checkpoint

2026-10-21: baseline/design committed as f6bb90a00. The owner authorized
implementation and testing. Follow the
[execution plan](sir-native-batching-execution-2026-10-21.md) for the current
checkpoint, bounded compute and results. Native batching, memory contractions
and optional batched sharding are implemented; numerical and capacity findings
are recorded there. The earlier audit below describes the saved baseline,
not the repaired source. Full training-step integration remains in the parent
SIR NeuTra plan; callback memory fit alone does not complete it.

## What the audit established

The completed kernel is XLA compiled, but its batch implementation is partial.
The executed call chain is `benchmark_sir_fp32_batch_capacity.py` ->
`make_batched_score_filter` -> `fixed_program` -> `program`. Both functions
are compiled; the scalar endpoint also delegates to this recurrence with B=1.
The saved source agrees with the currently inspected batch module.

| Finding | Inspected implementation | Consequence |
|---|---|---|
| One model and proposal construction per row | `covariance_proposal_score_batch_tf.py:248`, `:266`, `:274` | Python expands a separate graph fragment for every B at trace time |
| Recolouring and optional correction remain row-specific | Same file `:211`, `:282` | Removing only the model-construction tuple leaves duplication |
| Model callbacks accept one theta and one directional derivative | `covariance_proposal_score_tf.py:45`; `ledh_canonical_models_tf.py:29` | SIR transition callbacks replay local primal integration for separate derivative directions, although the filter itself has one all-parameter recurrence |
| Density direct contraction builds a pair-by-state residual | `covariance_proposal_shared_gaussian_tf.py:96` | Logical temporary [B,K,K,D] |
| Transport has the same residual and an explicit pairwise sensitivity | `covariance_proposal_score_batch_tf.py:96`, `:145`, `:146` | Density contraction alone cannot remove the main classes of large temporary |
| OT retains every cost tile | Same file `:92` | Total logical cost cache is [B,N,N], even though individual operations are tiled |
| Shared map matrices/covariances are broadcast over ancestors | `covariance_proposal_score_tf.py:131`, `:141` | Shared structure becomes a large logical tensor before later consumers select the shared entry |

Saved graph node counts obey **G(B)=3410+5009 B** exactly for all six measured
GPU sizes. This is graph-construction evidence, not a profiler attribution of
runtime or allocated memory. The GPU warmed measurements were 49.7 seconds
at B=4, 96.8 at B=8, and 171.9 at B=16; the last is one observation. They show
little descriptive throughput gain, but do not support a statistical ranking.

At N=4032, the required particle chunk is K=2016. Logical FP32 sizes are:

| B | One [B,K,K,18] residual | One [B,K,K] scalar tile | All cached OT costs [B,N,N] |
|---:|---:|---:|---:|
| 32 | 8.721 GiB | 0.484 GiB | 1.938 GiB |
| 64 | 17.442 GiB | 0.969 GiB | 3.876 GiB |
| 128 | 34.884 GiB | 1.938 GiB | 7.752 GiB |

These are individual logical tensor sizes, not additive peak predictions.
XLA may fuse, eliminate, reuse or duplicate buffers. Actual evidence is the
21.04 GiB allocation failure at B=32. No final VRAM requirement was measured
for B=64/128. See [the benchmark report](sir-fp32-batch-capacity-results-2026-10-20.md).

## Repair 1: one model/proposal graph with a real batch axis

Use a shared model numerical authority accepting theta [B,P], states [B,N,D],
and analytical state sensitivities [P,B,N,D]. Model adapters specify their
physical laws; they must not contain separate filter implementations. Existing
single-row APIs should wrap this authority at B=1. Port every claim-bearing
consumer before claiming that the generality repair is complete.

Replace tuples of scalar `Models`, `build_maps`, observation-density calls and
recolour operations with batch-native operations. Preserve one tf.while_loop
body for each genuine recursion: observations, SIR integration, flow steps,
Sinkhorn iterations and particle tiles. Keep fixed signatures and XLA. A
separate executable for each chosen static B is acceptable; replicating the
entire model computation B times in its graph is the defect.

For each model step, propagate the value and all derivatives together:

    X_next[b] = f(theta[b], X[b])
    D_next[p,b] = f_x(theta[b], X[b]) D[p,b] + f_theta_p(theta[b], X[b]).

Implement the analytical products directly, without constructing an unnecessary
full state Jacobian. The existing Jet algebra and packed solve already support
leading batch axes and all P directions; reuse them. For SIR, share each primal
integration stage across its three derivatives. Preserve the existing source
stage convention, observation law and covariance derivatives exactly.

The current `checked_chol` reduces validity over every leading axis. Passing
an entire batch to it unchanged would let one failed row invalidate healthy
rows. Generalize the shared factorization result to retain row validity and
margin, reducing only the relevant particle/component axes at consumers.
Also generalize optional moment correction before claiming all-control support.

Do not substitute `tf.map_fn`, `tf.vectorized_map`, or a while loop over scalar
filters. The older `ledh_canonical_batch_fused_tf.py` is not a solution to this
call-chain problem: it maps evaluations and is not the current proposal kernel.

## Repair 2: retain compact structure and reduce pairwise temporaries

Keep shared Gaussian covariance and affine matrices compact, such as
[B,component,1,D,D]. Gather or broadcast them only at the mathematical
operation that needs them. Ancestor-dependent matrices must retain their true
ancestor axis; SIR's constant observation Jacobian cannot justify a shared
covariance shortcut for range bearing.

For shared covariance, write Q=C^{-1}, z=x-o and m=mu-o. Then

    (x-mu)' Q (x-mu) = z'Qz + m'Qm - 2 z'Qm.

This gives pairwise scalar tiles using batched matrix products, without a
[B,K,K,D] residual. The existing centered `contraction="matrix"` implementation
already does this for shared Gaussian values, and `shared_weighted_score`
already contracts the complete location and covariance score without a
[P,B,K,K] score tensor. Reuse and audit these functions rather than copying
them. The unshared-covariance path needs its own exact contractions or streamed
coordinate accumulation; shared-covariance algebra is not valid there.

The identity is exact in real arithmetic but changes FP32 reduction order and
can suffer cancellation for nearby points. It must not be silently promoted
from a small primitive test to this SIR scope. Keep direct subtraction as the
reference and evaluate a coordinate-streaming alternative if centered matrix
products fail. Do not silently clamp negative quadratic values or add ridges.

Apply the same analysis to transport cost and barycentre derivatives. With
normalized column weights w_ij, y_j=sum_i w_ij x_i, and unnormalized weights
proportional to exp(alpha_i+c_ij), the exact derivative is

    d y_j = sum_i w_ij [d x_i + (x_i-y_j)(d alpha_i + d c_ij)],
    d c_ij = -2/epsilon (x_i-x_j)'(d x_i-d x_j).

The derivative of the column normalizer cancels in this centered expression.
Weighted first, second and cross moments can evaluate these contractions
without retaining every pair's parameter derivative. This formula still
includes the derivative of alpha from the finite Sinkhorn iterations and
the moving target particle x_j; dropping either changes the score. Test the
actual normalized finite program, not an infinite-iteration OT derivative.
Keep the direct formula as an independent diagnostic. The memory/runtime cost
of packed features must itself be measured; algebra alone does not establish
an acceleration.

Even after these repairs, caching all costs scales as B*N*N. Evaluate on-demand
tile recomputation if a large native B is essential, preserving epsilon,
iteration count and exact divisor chunk policy. This exchanges memory for
work and cannot be assumed faster. Do not promise B=128 fits merely because
the density residual was removed.

## Repair 3: separate training batch size from execution batch size

If the full native batch still cannot fit, evaluate a logical batch of 64 or
128 in batched shards, for example 8 or 16 rows per shard. Each shard calls
the same batch-native compiled kernel and returns one value/score per row.
A compiled loop over batched shards bounds memory without restoring scalar
sample evaluation. Shard size 8 is a measured filter-capacity starting point,
not a demonstrated full-training-step default.

For a sample-separable training objective L=(1/B) sum_b L_b, accumulate
(1/B) sum_shards sum_b grad L_b at frozen weights, then perform **one** optimizer
update. An update after each shard is a different optimization procedure.
Match per-sample noise; apply regularization once; accumulate before clipping.
Check that the actual objective has no unaccounted cross-sample terms. Preserve
the trainer's whole-update invalid-row policy. Both the target callback and
complete optimizer step need parity checks; no sharded implementation exists
in the benchmarked route yet.

## Implementation order and evidence contract

This is a staged repair inside the existing integration work, not a change to
the scientific model, prior or surrogate-HMC objective.

1. Generalize the shared model/proposal/reset interfaces and compact matrices.
   Use the current direct pair contractions to isolate the batching change.
2. Establish row-wise parity and graph scaling before another expensive ladder.
   Then evaluate density and transport memory repairs separately, preserving
   the failed and successful variants as diagnostic evidence.
3. Measure complete callback and training-step memory. Add batched sharding
   only if needed for the desired logical batch; test its update equivalence.
4. Repeat the requested CPU/GPU ladder with final choices frozen, identical
   inputs and source capture, both timing clocks, and explicit resource limits.
   Reissue any required tuning scope before scientific use of changed controls.

Baseline: attempt 02's fixed data, design, parameter bank, N4032/T50, uniform
FP32, TF32 off, 16 flow steps, five reset iterations, epsilon 1, correction 0,
K2016. Its code snapshot is the implementation comparator, not an oracle.
Use matched direct FP64 calculations and step-size sweeps of same-program
directional differences as diagnostics where feasible; neither replaces the
analytical score in the production path.

Required engineering evidence: the exact benchmark and training consumers reach
the shared implementation; graph operation count has no proportional-to-B
model-body replication; no dynamic FP64 arithmetic, scalar fallback, pfor,
hidden retracing or cross-row validity reduction is introduced. Document shape
specialization. Check row permutations, duplicated/distinct theta rows, one bad
row among healthy rows, all parameter scores, completed observations and
intermediate particles/covariances. Cover SIR first and the other four model
adapters before a cross-model completion claim. Test optional correction too.

Numerical promotion criterion: preserve the intended finite value and total
score; use tight precision-scaled primitive checks and the owner's existing
1% full-output materiality rule with vector-scaled score differences and an
explicit near-zero floor. Freeze the precise comparison scales before running;
do not select them after seeing results. The observed 8.274% CPU/GPU score gap
must be localized against references rather than adopting CPU output as truth.
Final-posterior correctness is outside this capacity repair's evidence.

Capacity promotion criterion: full-horizon execution with measured host/device
peaks within the configured limits. Timeouts are incomplete measurements, not
OOMs. Report compilation and warmed latency separately. Runtime rankings need
matched repetitions and uncertainty; small descriptive changes do not select
a numerical method. Test the small native batch and plain batched sharding
alongside more elaborate algebra so complexity must earn its cost.

Candidate numerical mismatch, poor throughput or insufficient memory triggers
the next planned repair; it does not reject NeuTra or LEDH. Corrupt evidence,
wrong target/data, loss of required diagnostics, or exhausted declared compute
are reasons to stop execution. Before implementation benchmarking, fill in
the exact executable commands, versioned output root and remaining CPU/GPU
budget. No additional compute has been spent on this design beyond source and
saved-result inspection.

Skeptical review: graph shrinkage alone cannot solve pairwise storage;
matrix identities alone cannot establish FP32 accuracy; a tiny-Gaussian test
cannot admit the SIR callback; per-shard optimizer updates cannot stand in
for a large training batch; and a successful filter call cannot establish
complete NeuTra-step memory fit. The staged checks above address each of these
failure modes. Implementation and downstream numerical evidence remain missing.
