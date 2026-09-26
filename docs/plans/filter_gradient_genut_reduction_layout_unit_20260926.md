# GenUT CPU reduction-layout attribution

The N=10,000,d=18 CPU timing probe 04139 reproduces a 3.21x native/original
warm-time ratio despite lower HLO/RSS. The native implementation replaced
TF32-sensitive matrix products with FP32 broadcast products and reductions.
Question: is reducing over the first particle axis causing unnecessary CPU
cost, and can an equivalent last-axis layout reduce that cost without changing
the accepted numerical program or reintroducing the original GPU GEMM defect?

First evaluate one diagnostic-only source transform: retain all recurrences,
controls, moments, validity checks, caps and report fields, but transpose the
particle axis to the last dimension in weighted/uniform covariance, pair
moments and the pairwise cross moment. Keep factor order unchanged. Matrix and
vector results return in the same orientation. Use the exact saved FP32 inputs
from 04122 at N=10,000,d=18, plus N=1,000,d=3 from 04114. Compare current native
graph/XLA and proposed-layout graph/XLA at existing atol=rtol=2e-5, including
every reporting field, exact replay and no retracing. Same-mode equality may
fail from reduction rounding; preserve that as a veto, never mask it as speed.
Compare against saved original FP64 results 04135/04137 as an independent
diagnostic; every non-report field must pass, and the known mode-dependent
cap-active report remains explicitly open.

Measure both XLA owners in one process with alternating paired calls and
materialized outputs, 12 pairs per size. Export optimized HLO and record its
size and source transform. This test does not provide isolated peak memory;
if the candidate passes numerics and has consistently lower CPU timing,
qualify it on GPU and measure fresh-process allocation before installation.
A performance gain alone cannot authorize a runtime change. Runtime sources
remain untouched during this diagnostic, and no alternative score or backend
is introduced. CPU is a reference/capacity lane; GPU/XLA remains the default.

Local skeptical review: a transpose can be optimized away, and FP32 reduction
order can change the thresholded report. Saving the full output comparisons
and HLO distinguishes a real layout change from a no-op or a hidden semantic
change. Neither one fixture nor one timing cohort supports general ranking or
canonical LEDH/HMC claims. A slower or numerically rejected candidate is
preserved, and the existing runtime remains the authority.

Use `genut_transitive_layout_cpu` with explicit CPU and 300-second timeout;
reserve four CPU and four GPU attempts, at most 900 seconds per backend,
inside the unchanged 56/52-hour campaign. Versioned campaign run directories
preserve all attempts. Run CPU first; only a viable CPU candidate proceeds to
GPU. Stop on source/input drift, invalid reference or exhausted budget.

04161 passes numerical comparisons but is slower at d=18: 113.3 versus
86.6 ms median, with every paired ratio above 1.15. Reject the layout candidate;
no GPU validation or runtime installation follows. Next use a diagnostic
`XlaDotV2` with `PrecisionConfig.HIGHEST` on both FP32 operands, retaining the
explicit symmetric projection where the original GPU fusion failed. This keeps
FP32 inputs/results and global TF32 enabled while requesting full precision
for those products. Only the installed TensorFlow raw op and protobuf schema
are used; the convenience xla.py wrapper imports NumPy and is not used.
Raw-op derivative availability must be recorded before any adoption; a missing
gradient blocks runtime installation until a separately verified TensorFlow
pullback exists. Compare complete current-XLA and FP64 records, exact replay,
stable traces and 12 alternating cost pairs under the existing bounds. Run
`genut_transitive_highest_dot_cpu` first; GPU follows only if CPU improves.
These candidate groups are explicitly explanatory, not terminal runtime gates.

04162 CPU and 04163 GPU pass highest-dot numerical comparisons. CPU d=18
median is 34.8 versus 80.7 ms; GPU d=3 rises to 1.78 versus 1.38 ms. TensorFlow
raises a missing-gradient LookupError for XlaDotV2. Do not install this raw op
in a gradient-bearing runtime. The bounded next repair is a reusable
TensorFlow custom pullback: test all matrix contraction orientations in FP32
and FP64 against independent matmul VJPs and finite differences, preserving
the complete reduced-program derivative. Then qualify the nested compiled
dot's graph/debug behavior, owner lifetime, fresh-process memory, and d=3/18
GPU cost under predeclared criteria. A shape-specific dispatch requires its
own measured size study, never an invented universal threshold. This work
remains inside the campaign caps but needs a concise new unit allocation before
execution. No public/default numerical change is authorized solely by this
diagnostic result, and no new user permission is required for that bounded
implementation/qualification work under the existing repair authorization.
