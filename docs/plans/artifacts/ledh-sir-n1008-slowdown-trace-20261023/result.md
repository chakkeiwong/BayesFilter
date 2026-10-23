# Trace of the SIR slowdown at 1008 particles

23 October 2026. Completed on `ledh-refactor-with-policy-fix` at
`90a3acebd3965da5bdd64cb165dfdfe2d8cf0649`. This is an engineering trace of
the fresh N=1008 endpoint. All earlier N64 measurements and scientific or
performance conclusions remain disqualified and are not used here.

## Finding

The slowdown is caused by the merged numerical-validity path becoming live in
the analytical LEDH recurrence for the plain and beta=1 conditional consumers.
It is not caused by KDM training, controlled SMC, or the conditional proposal
module: this benchmark constructs the endpoint with `guide=None`, and the
proposal source hash is unchanged between the compared snapshots.

The benchmark reaches the shared kernel through
`scripts/benchmark_sir_n1008_runtime.py:170-177`. The SIR model is an additive
Gaussian transition in this call, so
`bayesfilter/highdim/ledh_canonical_score_tf.py:252-256` selects the Gaussian
transition route. At each time point, the kernel calls the flow recurrence at
`bayesfilter/highdim/ledh_canonical_score_tf.py:661-686` and records the flow's
value and force validity in the absorbing status. The flow's per-substep state
and tangent are then masked by those flags at
`bayesfilter/highdim/ledh_canonical_score_tf.py:1465-1477`.

That dependency is the key change. In the plain beta=0 and conditional beta=1
weight branches, the Gaussian marginal-ratio calculation at
`bayesfilter/highdim/ledh_canonical_score_tf.py:715-724` does not use the
accumulated flow log determinant. The validity flags still use the determinant
and its tangent, however, so XLA cannot discard the determinant computation
without changing the accepted/rejected recurrence. The pre-merge optimized HLO
has no corresponding determinant pivot loop for these two endpoints; the
current HLO does. This is a live recurrence dependency, not merely extra trace
data emitted at the end.

The current flow does two repeated pieces of work:

1. `safe_cholesky` checks the particle-specific innovation covariance at
   `ledh_canonical_score_tf.py:1355-1383`. It checks finiteness, symmetry,
   positive Cholesky diagonals, and the factor result. The implementation is in
   `ledh_numerical_safety_tf.py:87-121`. The innovation covariance depends on
   the current particle covariance and flow step, so the general implementation
   cannot hoist this check outside the substep loop.
2. Each substep builds a determinant matrix and tangent at
   `ledh_canonical_score_tf.py:1454-1464`. For SIR, the observation dimension is
   9 and the state dimension is 18, so the determinant is reduced to a 9 by 9
   matrix. The grouped loop at `ledh_canonical_score_tf.py:1591-1646` stores
   the matrices and invokes `checked_log_abs_det_tangent`; that routine performs
   nine serial pivot iterations and a triangular tangent solve
   (`ledh_numerical_safety_tf.py:25-84`). The measured endpoint has 5 time
   points and 24 flow substeps, so this path is repeated 120 times per call.

The determinant dimension reduction is a real optimization, but it does not
restore the pre-merge behavior for plain or beta=1 conditional endpoints:
before the merge the full determinant calculation was dead in their optimized
graphs, whereas now the smaller checked determinant is required by the masks.
The HLO intervention graphs support this explanation: removing determinant
validation removes the nested determinant loop, while removing innovation
validation leaves that loop present. The intervention fixtures also show that
the removed checks reject singular determinants and asymmetric innovation
covariances that the unchecked variants accept. Those variants are therefore
diagnostics, not candidate fixes.

The physical beta=0.9 arm behaves differently. Its mixture density uses the
flow log determinant in the base component at
`bayesfilter/highdim/ledh_physical_proposal_tf.py:126-143`, so the determinant
was live before and after the merge. The current low-rank 9 by 9 determinant
replaces the pre-merge full-state 18 by 18 QR determinant for this SIR geometry.
That reduction can dominate the added validity checks, which explains why the
same merge makes this arm faster while making plain and conditional arms
slower.

## Measurements and limits

The separate fresh N=1008 comparison measured current/pre-merge ratios of
1.2133 for plain, 1.3585 for beta=1 conditional, and 0.7282 for the physical
mixture. Its full results are in
`../ledh-sir-n1008-runtime-20261023/result.md`. The slowdown trace itself
completed in 747.5 seconds, but its five workers shared the target GPU. The
diagnostic intervention ratios are consequently not a valid decomposition of
the slowdown into determinant and innovation percentages. They are used only
for HLO liveness and safety-loss evidence.

The trace does not show that every LEDH workload is slower. It covers one SIR
endpoint, `T=5`, FP64, TF32 disabled, one analytical direction, a single
parameter cloud, and N=1008. It also does not evaluate a learned KDM or native
parameter batching. The measured result establishes the cause of this endpoint
regression, not a score-accuracy or scientific-efficacy conclusion.

## Decision table

| Decision | Criterion | Status | Next justified action | Not concluded |
| --- | --- | --- | --- | --- |
| Accept the source-level cause | Current HLO retains determinant validity as a recurrence dependency; source call chain agrees | Pass | Design a guard-preserving optimization | Exact percentage cost of each guard |
| Reject unchecked guard removal | Negative fixtures must still reject invalid factors | Pass for the safety veto | Keep both guard families in the canonical route | That the current implementation is optimally fast |
| Explain the physical-arm reversal | Physical density needs the determinant; current geometry lowers its order from 18 to 9 | Supported by source and HLO structure | Preserve the physical route while profiling other scopes | Production-wide speed superiority |

| Inference class | Status |
| --- | --- |
| Hard veto screen | Healthy outputs, source integrity, parity, and focused tests pass |
| Statistically supported runtime comparison | Fresh N=1008 comparison supports the stated matched-scope ratios; intervention attribution is not supported |
| Descriptive diagnostics | HLO loop counts, worker timings, and allocator observations |
| Default readiness | Not evaluated |
| Scientific efficacy | Not evaluated; no learned guide or score oracle is involved |

## Post-run red team

The strongest alternative explanation for the absolute intervention timings is
GPU sharing and scheduling noise; this is why no guard-specific percentage is
reported. It does not explain the source and HLO distinction between the
pre-merge and current plain/conditional graphs. The weakest generalization is
from this FP64 single-cloud endpoint to FP32, TF32, long-horizon, or native
parameter-batch execution.

Raw evidence is preserved in `attempt-01/`, including
`analysis.json`, `result.json`, the optimized HLO files, worker manifests,
the launch log, and the 8-test harness log. No runtime source or numerical
guard was changed by this trace.
