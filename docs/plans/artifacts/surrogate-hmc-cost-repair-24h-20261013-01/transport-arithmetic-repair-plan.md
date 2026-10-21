# Local transport arithmetic repair: continuation of master phase 2

Question: can a shared algebraic rearrangement restore FP32 SIR scores while
avoiding the cost of FP64 recursion? Corrected inverse-CDF indices are frozen.
The reset decomposition at corrected saved inputs localizes the dominant local
error to transport (maximum scaled tangent error about4% versus at most0.054%
in moment calculations). Staging and fused reset agree exactly at those inputs.

For C_ij=-||x_i-x_j||^2/epsilon,
dC_ij=-(2/epsilon)(x_i-x_j)^T(dx_i-dx_j).
The existing packed contraction uses the algebraically equal expanded radial
formula E[dx_i^T x_i+dx_j^T x_j-dx_i^T x_j-x_i^T dx_j]. It can cancel large
summands. The first candidate forms dC by direct pair differences contracted
with centered tangents before softmax averaging, recomputing one tile at a
time. It must not store a full P*N*N tangent cache. The output derivative
already uses that direct contraction; all parameter directions remain together.

The second candidate removes Sinkhorn gauge drift. For any differentiable c,
(alpha+c,gamma-c) gives the identical transport plan exp(C+alpha+gamma).
Its total derivative is unchanged because dc-dc cancels. After each finite
iteration set c=gamma_0 and use gamma-c, alpha+c, differentiating c as well.
Before final normalization also center alpha and adjust gamma oppositely.
Inductively the next finite iterate differs only by this gauge; thus finite
iteration count and all parameter derivatives are preserved in real arithmetic.
Rounding can differ, so it remains an unpromoted numerics candidate.

Compare baseline FP32, gauge FP32, direct-pair FP32, and their combination with
baseline FP64 on identical checksummed saved inputs at observations2-6. Use
exact source copies with asserted single replacements; these are diagnostic
candidate implementations only, never runtime or admission artifacts. Reuse
the actual general shared transport function with no omitted capability.
Stable XLA signatures, GPU memory growth, TF32off, N4032/K2016, two rows.
A tiny CPU XLA comparison to baseline FP64 precedes the GPU run.

Local nomination: at least5x reduction in maximum scaled tangent error at
observations4,5,6, no nonfinite output, and no increase in value error above1%.
Record norm errors and warm timings3 paired repeats at observation5. These are
explanatory, not superiority evidence. Candidates without that reduction are
not promoted by good runtime. None becomes a default on this experiment.
A nominated candidate must pass all-five full horizons/off-centre final value
and score within1%, current mathematical/derivative tests, fresh tuning and
actual consumer comparisons. No likelihood oracle or posterior claim follows.

Baseline/adversary: exact same finite transport in FP64 is the arithmetic
reference; unchanged FP32 is the engineering comparator. Scientific heuristic
filter comparisons are a separate question and cannot be inferred here.

Budget: at most1800 worker-wall seconds/attempt, at mosttwo attempts for this
local test, inside the existing24h ledger. A candidate miss triggers another
planned local repair only when budget permits; corruption or nonfinite
comparison invalidates the particular test. Preserve unique output directories.
Manifest includes source hashes, exact commands, checksums, device, environment,
wall time and memory. No model, data, epsilon, iteration, or particle change.

Default audit: inherited controls and selected observations are localization
baselines, not tuning choices. Gauge pivot is a computational coordinate with
an explicit invariance proof, not a statistical/default change. Direct-pair
contractions can consume more memory/latency; measure rather than assume.
Skeptical audit passed: the candidates retain total derivatives, source changes
are explicit, intermediate errors only nominate, and full-filter amplification
is checked before any retained change. This is the existing master repair phase.
