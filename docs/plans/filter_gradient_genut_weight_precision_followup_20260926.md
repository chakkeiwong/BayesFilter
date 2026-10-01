# Residual GenUT weight-gradient precision

The Gram/triangular-pullback candidate passes all tested FP32 GPU XLA cells,
but dimension-18 FP32 CPU and GPU graph weight derivatives retain failures
against identical-input FP64. The original CPU implementation also fails.
This follow-up answers whether the remaining discrepancy comes from identifiable
rounding/cancellation or a precision limitation that needs an explicit error
result. A small derivative beside larger derivatives is not enough to call the
problem ill-conditioned. Keep the existing bounds and preserve failed cells.

Use exactly the saved 72-by-18, four-iteration inputs and frozen coefficients
from 04190/04192, with the pre-precision numerical source pinned at `5c9aa438e`
and the selected diagnostic transform `solve_gram_primal`. No candidate is
installed and the default TF32 setting, controls, floors and caps are unchanged.
FP64 value/gradient and directional finite-difference checks remain independent
references. Record original and changed input cases separately.

First expose the four weight-use branches (source moments, skew, kurtosis,
pair moments) as separate tensor operands carrying identical weight values.
Compare every untouched forward output with the unsplit program before using
the instrumentation. The sum of FP64 branch derivatives must reproduce the
unsplit total derivative. Save FP32/FP64 contributions, summation errors, and
the cancellation ratio `sum(abs(branch))/abs(total)` at all coordinates,
especially coordinate 18. Compare the observed error with the floating-point
bound for the final branch sum. A larger upstream error cannot be explained
away by the final sum's condition number.

If rounding remains upstream, use saved intermediates and derivative
contributions to isolate the first divergence. A bounded comparison with the
already-qualified, uninstalled highest-precision dot/pullback candidate may
test reduction rounding, but must retain the same target and every value field.
No FP64 runtime promotion, threshold waiver, stopped derivative, global TF32
change, or shape-specific dispatch is authorized by a diagnostic improvement.
A hybrid graph with compiled dot subprograms must be labeled explicitly.

Primary criteria are unchanged forward values, independent full derivatives
and meaningful localization of the error. If an error bound establishes that
the requested precision is unresolved, propose a fail-closed status or error
with that quantitative basis; never claim an unproved ill-conditioning diagnosis.
Only then resume the separately prepared memory/performance cohort and actual
runtime consumer qualification. No canonical LEDH analytical-score admission
follows from any autodiff diagnostic.

Reserve at most eight CPU and four GPU workers, each 300 seconds, inside the
unchanged 56 CPU/52 GPU process-hour caps. Begin with CPU contribution analysis;
GPU follows a concrete discriminating result. Use the existing campaign runner,
explicit devices, trusted GPU selection/growth and unique run directories.
Stop on input/source drift, invalid instrumentation/reference, or exhausted
allocation. Ordinary candidate failures remain repair triggers. New test code
must be registered before execution and sources frozen while workers run.

Skeptical review: splitting operands can change compiler fusion and gradient
summation order, so exact forward equivalence and an FP64 total-derivative
tie-out are prerequisite checks. A final-sum cancellation ratio is explanatory,
not a complete bound on nonlinear upstream rounding. A faster candidate with
an inaccurate gradient remains rejected. There is no supported performance
ranking in the current evidence.

04208 passes all instrumentation checks on CPU: graph/XLA, FP32/FP64 and both
original/changed operands retain bitwise-identical forward outputs, one trace
and the same FP64 total derivative. At the failing original coordinate 18,
the four FP64 branches have cancellation ratio about 673. The actual final
FP32 summation error is only 2.62e-6 (graph) or -1.19e-6 (XLA), below its
7.45e-6 bound. Most of the total 3.52e-5 / -3.33e-5 error lies upstream,
especially the source-moment and kurtosis branches. This supports cancellation
amplification but does not blame the final sum alone or establish a complete
runtime precision bound.

Next run the planned CPU high-precision-dot comparison with the selected
Gram/triangular precision candidate. Preserve the same reduced algorithm,
frozen scalar coefficients and original/changed dimension-18 inputs. The
qualified dot-family pullback supplies derivatives; no raw missing-gradient
operation is admitted. Compare all raw fields, record known cap-report failures,
and require every smooth field/gradient to pass the unchanged independent FP64
bound. The explicit graph parent contains XLA dot subprograms and must be
labeled that way. Do not infer timing or memory from this same-process check.

04209 rejects the combined highest-dot trial: it passes the original graph
case and changed XLA case, but fails changed graph and original XLA at unchanged
weight-gradient bounds. All outputs/derivatives, exact replay and one-trace
records are preserved. No GPU trial or performance ranking follows this failure.

Next test a local analytical regrouping of the source weighted-moment pullback,
one of the two largest upstream error contributors. For `m=sum(w*x)`,
`y=x-m`, `C=sum(w*y*y^T)`, incoming symmetric covariance adjoint `G` and mean
adjoint `g`, set `s=sum(w*y)`, `h=g-2G*s`. Then
`d_x[i]=w[i]*(h+2G*y[i])` and
`d_w[i]=x[i]^T*h+y[i]^T*G*y[i]`. These follow by differentiating the two
displayed moment equations and collecting the shared mean term. Retain `s`;
do not assume the realized weights sum exactly to one. Symmetrize the incoming
covariance adjoint to match the original forward symmetrization.

The diagnostic keeps the exact forward equations and combines the two weight
contributions before their final coordinate reduction. First check primitive
FP32/FP64 derivatives against independent original FP64 autodiff and centered
finite differences, including deliberately non-normalized weights. Only then
evaluate the complete original/changed dimension-18 cases in graph/XLA. Require
bitwise unchanged forward outputs and the original independent gradient bounds.
This is a tested first-derivative identity, not a replacement canonical LEDH
recursive score. The two CPU groups remain inside the eight-worker allocation;
a rejection is preserved and cannot justify a tolerance change.

04210 passes the FP32/FP64 primitive checks. 04211 rejects the complete
moment-regrouping candidate: all four original/changed graph/XLA cases retain
weight-gradient failures, with bitwise unchanged forward records. No candidate
is installed. The two rejected regroupings do not justify further arbitrary
changes to reduction order.

Next cut the existing program at its six source features: mean, Cholesky factor,
skew, kurtosis, pair skew and pair kurtosis. Obtain preparation and continuation
functions mechanically from the frozen source; preserve all controls. Require
the recomposed forward records to be bitwise identical to the complete program
and the FP64 chain-rule derivative to agree with the complete reference. Save
the FP32 recomposed derivative and explicitly check whether the original failure
survives this instrumentation. A changed or non-reproducing cut cannot explain
the original failure.

For preparation F, continuation H, original inputs x and rounded FP32 feature
vector b32, decompose the observed derivative error into three telescoping terms:
FP32 preparation pullback error with its observed continuation cotangent;
continuation cotangent error at the same b32; and propagation of the source
feature rounding, comparing DH64(b32) with DH64(F64(x)). All FP64 preparation
pullbacks use the same original rounded source inputs. This is a finite-program
localization, not an a priori error bound or ill-conditioning certificate. Save
each term and verify their sum against the observed total. Start on CPU in one
300-second worker from the remaining four-worker allocation. GPU work requires
a discriminating CPU result; there is no tolerance or precision-policy change.

Skeptical review: an explicit cut can change XLA fusion and the final adjoint
addition order. Exact forward checks, preservation of the failed gradient cell,
and an independent FP64 chain-rule tie-out are therefore prerequisites. The
telescoping decomposition attributes error only in a valid recomposed program;
it must not be generalized to a changed complete program. This check answers
where to repair next and cannot qualify memory, speed, or canonical LEDH scores.

04212 stopped before numerical comparison: the diagnostic cut omitted the
target covariance consumed by the unchanged covariance-residual report. Carry
that seventh feature as well; its adjoint for the declared particle loss is
zero. The repaired harness retains the same scientific contract and uses the
next reserved CPU worker. Preserve 04212 as a harness failure.

04213 rejects the cut as an explanation of the complete program. All four
forward comparisons change bits, and three of four cut cases remove the
original gradient failure. The FP64 chain rule and telescoping identities pass,
but neither fact restores the missing witness. Do not interpret its error
components as causes of the original failures.

Use one of the last two CPU workers for a whole-program directional check at
weight coordinates 18 and 21, using TensorFlow ForwardAccumulator only as an
independent diagnostic. Compare these directional derivatives to the frozen
FP64 gradient and its checked finite difference on the same original/changed
inputs. Keep directions as explicit stable-signature operands, avoiding pfor.
Save all primal records and require exact equality to the reverse-mode program
before attributing any difference to derivative propagation. Passing forward
derivatives would implicate reverse evaluation rather than certify the full
gradient or a canonical score. Failure to retain the original forward values
invalidates that attribution. This is a discriminating derivative-order check,
not a new runtime candidate or tolerance adjustment.

04214 fails while constructing the forward derivative: TensorFlow attempts to
capture the triangular custom pullback's loop-local Cholesky tensor in the
parent graph (`InaccessibleTensorError`). There is no derivative result.
Use the last reserved CPU worker to encapsulate the identical triangular
callable in a fixed-signature graph function. The original unwrapped program
remains the comparator; retain the exact-primal and derivative checks. The
outer XLA arm compiles the enclosed numerical operations, while graph mode
remains an explicit diagnostic. This is a harness compatibility retry, not a
runtime change. If the retry fails, preserve that limitation and close this
local allocation with precision unresolved, continuing independent execution
repairs and descriptive cost work under their existing allocations.

The eight-worker allocation closes through 04215 at 154.985868 CPU seconds.
The wrapped diagnostic preserves all graph forward records but still fails
original coordinate 18; its XLA path cannot compile the higher-order loop
derivative. See the [result](filter_gradient_genut_weight_precision_result_20260926.md).
No GPU trial, runtime candidate promotion or ill-conditioning finding follows.
