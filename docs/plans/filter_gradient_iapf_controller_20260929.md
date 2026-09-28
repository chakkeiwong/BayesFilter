# Adaptive iAPF controller repair

Source review at5eb694352 confirms three connected obligations, beyond the
now-qualified fixed fitted-APF owner:

1. `iapf_adapter.py::iteration_decision` computes scaled likelihood CV,
   monotonicity and particle doubling using Python arithmetic/loops.
2. `execute_iapf` runs filtering, decisions, fitting, precision casts and
   convergence/capacity vetoes in a Python recurrence. The first pass uses a
   constant twist; later passes use the fitted coefficients. The fitter runs
   at the preceding particle count even when the next pass doubles N.
3. `iapf_scope.py::validate_adaptive_ledger` recomputes numerical decisions for
   every prefix in Python and requires exact decision fields, including CV.
   It is part of result selection, so it cannot become a diagnostic exception.

Preserve all three call chains, the independently seeded iteration schedule,
mixed fit/filter precision,23-column diagnostics, cast/convergence vetoes and
actual particle-time accounting. No fit retuning, ridge, tolerance relaxation,
stopping simplification or probability-measure padding is permitted. Current
fit and final timing fields describe two separate stages; any eventual enclosing
design must preserve truthful measurements rather than invent substage times.

First implement a native decision primitive with stable padded history inputs
and explicit valid-prefix length. Padding represents absent history only and
must never enter CV or monotonicity. Retain Python3.11's sequential summation
order with native TensorFlow loops, FP64 controller arithmetic (the old host
materializes filter scalars as Python doubles), strict l>k and CV<tau tests,
sample variance divisor k, equal-count check and the finite doubling/cap veto.
Return native status and complete-window fields; no runtime fallback to Python.
Keep the original adapter/validator untouched during this primitive phase.

Freeze the original controller source and deterministic cases before launch.
Test incomplete/first-complete windows, strict stopping, count growth and
suppression after growth, monotone histories, capacity rejection, large common
log shifts, invalid histories and changed operands. Test CV thresholds at the
reference value and its immediately neighboring FP64 values; a changed action
is a veto requiring localization, not permission to relabel it as roundoff.
Require identical actions/particle counts/window flags, CV error<=1e-14 and
exact replay, one trace per declared signature and enclosing HLO without host
callbacks. Keep failed arrays and decisions before assertions.

Use max_history8, k2, max_particles64; N8/16/64 histories exercise the controller
without filtering or fitting. These are deterministic mechanics checks, not
tuned statistical controls or method-quality comparisons. Test tau0.01 for
ordinary cases; boundary tau values are adversarial implementation checks.
Invalid finite-history conditions return a false validity status; the eventual
host boundary must map that to the existing ValueError/DiagnosticFailure.

Allocate4 sequential workers/600 CPU and600 GPU seconds within the unchanged
global budget: CPU, GPU, readback/policy, one localization. Use groups
`iapf_controller_*`,300-second timeouts, existing tf-gpu environment and unique
campaign run directories. CPU is reference/debug; GPU uses trusted access and
memory growth. Failures are preserved and trigger the smallest discriminating
repair. Budget exhaustion, changed algorithm/streams or missing provenance stop
the unit. No training/HMC, subagents, external/model changes, packages/system/
cache changes, canonical LEDH claims or main merge.

After the primitive qualifies, enclose the offline recurrence using a finite
particle ladder and TensorFlow shape dispatch to the existing filter and fitter.
Document the output-shape and final-stage timing design before writing it.
Do not build padded particle distributions or count fixed loops as adaptation.
Update numerical ledger validation to the same shared controller, with an
independent archived comparator. Then qualify actual healthy/refused public
consumers, mixed precision, all histories, frozen-fit derivatives and costs.
No primitive pass closes any of those endpoint gates.

Skeptical review: standard parallel reductions could change a threshold decision
even when CV values are close. Sequential arithmetic and boundary tests address
this before an expensive full-filter owner is built. GPU transcendental rounding
may still differ; that is an explicit veto/localization question. Comparing only
values far from tau would miss it. The original validator's exact CV comparison
is an actual dependency, not a reporting detail. This bounded decision phase
passes review and does not alter the active adaptive route yet.

04863 preserves the anticipated strict-boundary failure: reference CV is
0.3303714383398673, candidate CV0.33037143833986743. At tau equal to the next
FP64 value above the reference, the original stops and the candidate fits.
Ordinary cases passed before that veto. Do not proceed to endpoint migration.
Use the reserved localized worker to compare shifted logs, exponentials,
sequential totals, mean, squared errors, variance, square root and CV. Compare
constant versus runtime divisors to distinguish reciprocal-strength-reduction
effects; diagnostics must state whether exposing intermediates changes the
actual owner's CV. No numerical implementation or threshold change is assumed.


04864 localizes the CPU discrepancy: shifted logs, exponentials and sequential
sum match exactly. Constant division by3 changes the mean from
0.7717720189564637 to0.7717720189564636. Runtime divisors restore every reported
intermediate through the square root, but final CV still differs by one ULP,
consistent with algebraic reassociation across the nested quotients. Exposing
intermediates preserves the original candidate CV in the constant arm, so this
probe reproduces the actual owner discrepancy.

Test local XLA optimization barriers around each scalar division's operands,
using the existing TensorFlow operator already used in factor geometry. The
barrier preserves the original scalar division boundaries; it does not change
the formula, threshold or dtype, or set a global compiler flag. Keep the
unrepaired candidate source as diagnostic evidence. Qualify the ordinary owner
again on CPU and GPU, including all three boundary thresholds; a remaining
difference still vetoes migration. Expand the attempt allowance to6 workers
with unchanged600CPU/600GPU seconds, within the same global budget. This is a
localized repair of the same scientific/execution contract, not new direction.

04865 rejects barriers alone: the strict decision mismatch is unchanged.
04866 shows that isolated runtime scalar divisions match Python exactly.
Optimized HLO confirms two mechanisms: constant mean division is replaced by
multiplication by0.33333333333333331, and the unbarriered dynamic final quotient
becomes `(root * window_count) / total`. The barrier owner preserves the latter
division boundary but still contains the reciprocal constant for its mean.

Use runtime window cardinality together with the local barriers. Every valid
complete window has exactly k+1 count entries>=2; its native count therefore
equals the original denominator, and count-1 equals the sample-variance divisor
k. Invalid histories never enter this branch. This changes neither the window,
statistical formula nor comparison threshold. It removes the compiler's static
reciprocal opportunity without a global compiler option or numerical clipping.
Repeat the unchanged decision gates. Allow8 workers with the same600CPU/600GPU
seconds to finish this localized repair, GPU check and readback; preserve all
failed variants. This remains a bounded primitive investigation, not admission.

04867 passes the unchanged CPU decision gates.04868 preserves a remaining GPU
failure: CV0.3303714383398674 equals the immediate upper-neighbor tau, whereas
the original CV is0.3303714383398673 and stops. Use the seventh worker for the
same arithmetic localization on GPU, including dynamic divisors plus barriers.
Do not silently accept the changed branch or extrapolate CPU qualification.
