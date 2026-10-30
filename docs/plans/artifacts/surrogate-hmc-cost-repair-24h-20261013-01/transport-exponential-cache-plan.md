# Fixed transport-kernel cache: bounded computation candidate

This is phase 2 of the authorized computation extension. The working scientific
source stays frozen while a primitive diagnostic tests the next repair. The
question is whether caching exp(C) removes repeated pairwise exponentials from
the finite Sinkhorn reset without material arithmetic or derivative changes.
Here C_ij=-||x_i-x_j||^2/epsilon <=0 is fixed within one reset, not across
observations or HMC positions. N4032/K2016, FP32/TF32-off and all iteration counts
remain fixed. The earlier full tangent cache is not repeated.

For any potential a, let s=max_j a_j, K_ij=exp(C_ij), u_j=exp(a_j-s).
Then Z_i=sum_j K_ij u_j, L_i=s+log(Z_i), W_ij=K_ij u_j/Z_i. In real
arithmetic these equal logsumexp_j(C_ij+a_j) and its softmax weights.
The analytical derivative remains dL_i=sum_j W_ij(dC_ij+da_j). The derivative
of the common shift cancels; no derivative term is omitted. Existing packed
weighted contractions can consume W unchanged. This caches only B*K*K primals,
not P*B*K*K tangents. Both row and column normalizations use the same identity.

Floating-point underflow needs an explicit check. Each of the kernel, potential
and product stages can flush a term below the smallest normal number tau. Since
C<=0 and a-s<=0, the sum of such losses is bounded by 3*K*tau before rounding.
Use the fast expression only when Z_i is finite and positive and
3*K*tau/Z_i <=1e-6 for every row. Otherwise use the existing log-domain expression
for that tile, and emit the fallback diagnostic. This bound concerns flushed
mass only; it does not bound rounding or its propagation into scores. The 1e-6
guard is a conservative hypothesis, four orders tighter than the owner 1%
screen, not a tuned scientific control. Invalid/non-nonpositive costs also fall
back. Empirical FP64/derivative and full-horizon comparisons remain required.

Stages: (1) CPU diagnostic tests of values, all parameter directions and extreme
potential fallback; (2) a GPU/XLA primitive cost screen using the actual tile,
batch and packed-width dimensions, at most300 seconds per model; (3) only if
useful, a separately recorded full-filter candidate with unchanged controls,
full-horizon and off-centre parity before actual consumer timing. No primitive
result promotes the filter or changes selected source. Fresh source-bound
tuning is required if a later implementation is retained.

Comparator: existing logsumexp plus softmax and the same packed matrix product.
Primary primitive screen: finite outputs; FP64 value/derivative equivalence;
FP32 <=1% scaled gap; safe fallback on underflow. Costs and fallback rates are
explanatory only. Stop this candidate on correctness failure, increased memory
or no useful measured cost reduction; continue other existing master work.
Total hard budget is the existing86400 seconds/120 attempts with one GPU worker.
Save the source, parameters, hashes, GPU/memory/TF32/XLA provenance, cold and warm
times and individual outputs in transport-cache-probe-* under this campaign.

Skeptical audit: a microbenchmark can be faster while whole-filter fusion is
slower, so it only nominates a candidate. Synthetic costs cannot prove coverage
of any model. Potential-dependent underflow can erase meaningful terms, so a
cache without the denominator guard is rejected before execution. The guard's
loss bound does not certify score accuracy. Few timing repetitions establish
no statistical speed ranking. No posterior, canonical LEDH or default claim.
