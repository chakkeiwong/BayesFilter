# Phase 2B Batched Reduction Derivation

**Date:** 2026-09-11  
**Branch:** `surrogate-hmc`  
**Contract:** [Phase 2B Binding Refactor Contract](phase2b-refactor-contract-20260911.md)  
**Status:** DERIVED — implementation may proceed

## Decision

Flattening is safe only for pointwise operations. Cloud reductions will retain
explicit tensors `[B,N,...]`; pointwise model calls may use `[B*N,...]` and
reshape back before any normalization, moment, transport, cap, or diagnostic
reduction. Tangents use `[K,B,N,...]`. This eliminates dependence on implicit
segment IDs and makes row isolation structural.

Let `b=1..B`, `i,j=1..N`, direction `k=1..K`, state dimension `d`, particles
`X[b,i,:]`, normalized weights `w[b,i]`, and tangents `dX[k,b,i,:]`,
`dw[k,b,i]`. Every formula below is evaluated independently for each `b`.

## Pointwise stages

UKF sigma propagation, transition/observation callbacks, LEDH flow updates,
and density evaluations act on individual `(b,i)` elements. They may flatten
`[B,N] -> [B*N]`. Direction inputs must first transpose
`[B,K,N,P] -> [K,B,N,P]` before flattening, as repaired in
`ledh_canonical_batch_fused_tf.py`; direct reshape interleaves B and K.

After pointwise work, reshape values to `[B,N,...]` and tangents to
`[K,B,N,...]` before reductions.

## Per-row normalizer and weights

For logits `l[b,i]`:

`a[b] = logsumexp_i l[b,i]`

`p[b,i] = exp(l[b,i] - a[b])`

For direction `k`:

`da[k,b] = sum_i p[b,i] dl[k,b,i]`

`dp[k,b,i] = p[b,i] (dl[k,b,i] - da[k,b])`

All reductions name `axis=N`; no reduction may include B or K.

## Sinkhorn transport

For each row:

`Delta[b,i,j] = X[b,i] - X[b,j]`

`C[b,i,j] = ||Delta[b,i,j]||^2`

`dC[k,b,i,j] = 2 Delta[b,i,j]^T (dX[k,b,i]-dX[k,b,j])`

The mean-cost scale is `s[b]=max(mean_{i,j} C[b,i,j], 1e-3)` with the existing
piecewise tangent. Kernel `G=exp(-C/(epsilon*s))` and its tangent use the
ordinary product/quotient rule. Sinkhorn vectors have shape `[B,N]`; matrix
vector products are batched (`einsum` or `matvec`) and never flatten B into N.

With coupling `Q[b,i,j]=u[b,i] G[b,i,j] v[b,j]`, row mass
`r[b,i]=sum_j Q[b,i,j]`, and transport `T[b,i,j]=Q[b,i,j]/r[b,i]`:

`dT = dQ/r - Q*dr/r^2`, where `dr=sum_j dQ`.

Barycentres and tangents are:

`Y[b,i] = sum_j T[b,i,j] X[b,j]`

`dY[k,b,i] = sum_j dT[k,b,i,j] X[b,j] + T[b,i,j] dX[k,b,j]`.

## Contract-E moments and affine restoration

Target weighted moments are per row:

`mu[b] = sum_i w[b,i] X[b,i]`

`dmu[k,b] = sum_i dw[k,b,i] X[b,i] + w[b,i] dX[k,b,i]`

`P[b] = sym(sum_i w[b,i](X[b,i]-mu[b])(X[b,i]-mu[b])^T)`.

Its tangent is the sum of the weight term and both centered-particle product
terms. Uniform barycentric and injected moments use `reduce_mean(axis=N)`.
Cholesky factors and triangular solves operate on batch matrices `[B,d,d]`;
the Cholesky differential operates on `[K,B,d,d]` by broadcasting the primal
factor over K. The final affine map and its tangent are unchanged algebraically,
with B retained.

Covariance carry uses the same transport:

`Pcarry[b,i] = sum_j T[b,i,j] Ppost[b,j]`

`dPcarry[k,b,i] = sum_j dT[k,b,i,j] Ppost[b,j] + T[b,i,j] dPpost[k,b,j]`.

No second transport solve is permitted.

## Higher-moment correction

Weighted targets, standardization, diagonal skew/kurtosis, pairwise moments,
trust-region norms, particle RMS caps, coordinate caps, affine restoration, and
all diagnostics reduce only over N within each B row. Their batched shapes are:

- target vectors `[B,d]`, tangents `[K,B,d]`;
- target pair matrices `[B,d,d]`, tangents `[K,B,d,d]`;
- standardized particles `[B,N,d]`, tangents `[K,B,N,d]`;
- scalar diagnostics `[B]` (or `[K,B]` only when direction-specific).

Algorithm iteration counters are shared scalar controls, but loop state carries
B and K. A failure in one row must set that row's validity false without
collapsing or masking another row.

Configuration branching remains Python/static at trace construction because the
controls define the finite program. Disabled parent mechanisms bypass nested
work exactly, preserving the no-effect identities in the regression matrix.

## Annealed composition

Each stage computes `[B,N]` logits and `[K,B,N]` tangents. Systematic resampling
must use independent deterministic offsets per `(time, stage, b)` and gather
within each row. Indices are fixed-realization, piecewise-constant tangents as in
the existing single-cloud oracle contract. A gather index for row `b` may never
address particles from another row.

## Adapter mapping

- Single cloud: promote theta/value state to B=1 and one score direction K=1,
  then unwrap the existing scalar and `[1]` result.
- Fused batch: preserve `[B,P]` theta and `[B,K,P]` directions.
- Rank-2 directions: promote to K=1 and squeeze only the K axis on return.
- Reference batch: call the same engine, not a Python row loop.

## Required implementation tests

1. B=1 stage parity against the old single-cloud route after each ported stage.
2. B/K row isolation after each reduction-bearing stage.
3. JVP versus autodiff oracle for Sinkhorn, Contract-E restoration/carry,
   diagonal/trust correction, pairwise correction, coordinate cap, and annealing.
4. Full configuration-regression matrix after every stage.
5. Shape and validity tests for rank-2 and rank-3 direction inputs.

The derivation finds no mathematical incompatibility with a unified engine,
provided reductions retain explicit B and tangent storage retains explicit K.
The unsafe direct-flatten approach is vetoed.
