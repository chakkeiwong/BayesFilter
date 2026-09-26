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
