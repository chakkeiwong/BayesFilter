# Native batching implementation review

Reviewed against the saved executed baseline in
artifacts/sir-fp32-batch-capacity-20261020-02/gpu-b4/source. This is a local
implementation review, not an independent reviewer verdict or posterior result.
The execution plan is sir-native-batching-execution-2026-10-21.md.

## Mathematical target and call chain

The target remains the finite covariance-proposal likelihood and its total
analytical parameter derivative, with fixed observations, random design,
mixture weights, flow/reset schedule, and covariance restoration. A batch
contains independent parameter vectors; it does not change the filter target.
Values have shape [B,...], and derivatives have shape [P,B,...]. Ordinary Jet
operations implement the product, quotient, linear-solve and Cholesky rules.
The SIR integrator retains the source's half-step fourth stage. Changing that
stage to textbook RK4 would change the model and is outside this repair.

The benchmark calls make_batched_score_filter, which constructs NativeModels
once and executes one observation recurrence. The scalar make_score_filter
delegates to the same authority with B=1. The optional sharded factory loops
over groups of native batches, each larger than one. A fixed input signature
and jit_compile=True enclose the complete numerical callback in all three
entry points. Python handles configuration, static tensor-axis metadata,
source capture and reporting; it does not dispatch one target per sample.
Executable evidence includes full-filter comparisons for five model families,
seven model-callback variants, permutations, all-output shard comparisons,
and a direct scalar-public-entry test. The NeuTra training consumer is not yet
wired, so no training call-chain or optimizer-update parity is claimed.

## Transport contraction

For a source i and destination j, write the log kernel as
C_ij = -||x_i-x_j||^2/epsilon. In a parameter direction, with state tangent
d_i, its total derivative is

    dC_ij = -2/epsilon [x_i.d_i + x_j.d_j - d_i.x_j - x_i.d_j].

Subtract a common position origin and its tangent from every particle first.
Both differences, and hence this identity, are unchanged. The primal cost is
still accumulated from coordinate differences, one coordinate at a time;
there is no [B,K,K,D] residual. Log-sum-exp derivatives are softmax-weighted
averages. Packing d_i, x_i.d_i and the potential derivatives into feature
columns allows a matrix multiplication to compute these averages without a
[P,B,K,K] derivative tensor. Finite Sinkhorn iterations and their derivatives
remain unchanged in real arithmetic.

For the barycentre, let T_ij = exp(C_ij + alpha_i - normalizer_j),
m_j = sum_i T_ij and y_j = sum_i T_ij x_i. Then

    dm_j = sum_i T_ij (dC_ij + d alpha_i - d normalizer_j),
    dy_j = sum_i T_ij [d_i + x_i(dC_ij + d alpha_i - d normalizer_j)],
    d(y_j/m_j) = dy_j/m_j - y_j dm_j/m_j^2.

The implementation retains every term, including destination motion,
potential/normalizer derivatives, column-mass differentiation, and the common
origin derivative. Packed moments include x_i d_i^T and x_i x_i^T. This is an
algebraic rearrangement, not a stop-gradient approximation. Source-cloud
moments and weights continue through Cholesky covariance restoration.

The OT cost cache is still [B,N,N]; it has not become a fully streamed
linear-memory algorithm. Packing replaces pair-by-state storage with
O(B*N*P*D^2) feature storage, which is material at high D. Range-bearing's
nonshared covariance mixture retains pairwise state/parameter temporaries;
the SIR capacity result does not establish that model's capacity.

## Gaussian contraction and numerical limitations

For shared covariance C and precision A=C^-1, the log-density derivative is
-u^T(dx-dmu) + (u^T dC u - tr(A dC))/2, with u=A(x-mu).
Weighted moment contractions preserve both location and covariance terms.
The direct primal quadratic accumulates coordinate contributions and avoids
forming the pair-by-state tensor. Centering reduces cancellation in common
position and tangent translations. It does not prove stability over a long
nonlinear recurrence.

The full SIR T50/N4032 comparison supports real-arithmetic equivalence: old
and native FP64 scores differ by at most 4.35e-11 in relative vector norm on
the four fixed parameter rows. It does not establish FP32 equivalence. Row 3
differs from the common FP64 reference by 5.769% for the old program and
8.057% for the native program. Native versus old FP32 differs by 2.274%,
above the owner's 1% criterion. Small likelihood discrepancies cannot waive
a materially inaccurate score. No clipping, damping, altered target, relaxed
tolerance, or FP64 default has been introduced to hide this finding.

## Defects found by review and tests

1. The first graph test counted size-dependent Const/Fill allocation nodes as
   duplicated model computation. The corrected test compares numerical-body
   counts, which are constant for B=1,2,4,8,16,32,64,128.
2. Invalid rows received different later ancestor history depending on whether
   another row kept the loop running. History after row failure now consistently
   contains NaN. Healthy values and scores are unchanged; every shard output
   is compared, including a padded invalid final row.
3. The scalar wrapper indexed scalar covariance_tolerance. It now indexes only
   nonscalar outputs; a public-entry XLA regression checks every output.

The optional shard callback is an evaluation adapter. It preserves the full
logical output batch while bounding the temporary working batch. A future
training adapter must still accumulate the mean objective/gradient at frozen
weights, apply regularization and clipping once, and perform one optimizer
update. This review does not claim that machinery has been implemented.

## Reproducibility and final capacity check

The capacity runner no longer imports the unrelated five-model HMC master
to prepare SIR inputs. It uses the same repository model specification, frozen
dataset, selected controls and stateless design directly. All six prepared
input hashes match the saved baseline; the focused preparation tests pass.
This removes a dependency on unrelated uncommitted campaign code.

An isolated tree starts from baseline commit f6bb90a00 and overlays only the
intended source/test prerequisites. Its manifest records every overlay and the
hash of the existing locally built TensorFlow custom op. Initial collection
failures are preserved: the custom op needed copying as an environment
prerequisite, and an older compact-map campaign integration test imported
unrelated uncommitted HMC machinery. That older integration test remains
outside this scoped commit; it passed in the working-tree suite. The isolated
suite covers the repaired map algebra through independent native-model,
full-filter, shared-Gaussian, fixed-design and sharding checks. Its final log
is the source of the reported clean-tree test count.

GPU B128 exceeds the declared 12 GiB allocator cap as a single native batch.
The logical B128/native S4 adapter completes all 50 observations for every row
in 333.77 seconds including compilation, using 0.761 GiB allocator peak. Its
likelihoods and scores match repeated native B4 outputs exactly on this design.
This is one diagnostic evaluation, not a steady-state timing estimate or a
training-step memory measurement. B64 fits at about 8 GiB and completes two
evaluations, but its arm reaches the time cap before the full repeat set.

## Skeptical verdict

Native graph structure, algebra, failed-row isolation and public-entry wiring
pass the executed checks. Large-batch memory measurements are meaningful as
engineering diagnostics. Uniform FP32 numerical promotion fails the stated
score criterion, so neither default readiness nor serious NeuTra training is
established. The strongest alternative explanation for the apparent performance
gain is varying device/CPU contention; report timings descriptively. The
weakest correctness evidence is full-horizon FP32 stability. A measured,
target-preserving FP32 repair and a fresh matched full-horizon comparison are
needed before closing that numerical issue.
