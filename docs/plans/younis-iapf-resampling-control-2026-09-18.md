# iAPF resampling controls: reviewed execution plan

Date: 2026-09-18. Owner instruction: refresh and continue the master program.
This is a new bounded diagnostic campaign. The Fisher comparison is complete;
its final streams and exhausted budget remain closed.

## Question and mathematical target

Can the randomness introduced by ancestor selection explain and reduce the
variance of the analytical terminal-genealogy Fisher score on the existing
short-horizon nonlinear models? The previous comparison found acceptable
N4096 mean-bias screens but large variance and losses to UKF on weak curvature.
The candidate is a control variate applied to that same Fisher statistic.
General backward smoothing and LEDH integration remain deferred.

Write the terminal statistic as S_N = sum_i W_T^i a_T^i, where a_T contains
the analytical partial derivative of the complete physical joint density,
including the initial distribution, along the actual sampled genealogy.
This self-normalized statistic estimates the physical marginal score. At
finite N it can be biased and is not the derivative of the reported finite
particle likelihood.

Immediately before each nonterminal resampling, let F_t contain the current
particles, normalized CDF, additive scores a_t^i and all earlier randomness.
Let q_t^i be the ACTUAL probability of selecting i with that CDF. Define

    C_t = sqrt(N) [ N^{-1} sum_j a_t^{J_t^j} - sum_i q_t^i a_t^i ].

Conditional on F_t, E[a_t^{J_t^j}] = sum_i q_t^i a_t^i. Therefore
E[C_t | F_t] = 0 and E[C_t] = 0. Independence between C_t and later score
terms is neither assumed nor needed. Concatenate initial and intermediate
controls, excluding the final resampling, which occurs after S_N is computed.
For T=2 this gives 12 columns, with four structurally zero initial columns.

Fit B on independent calibration particle streams, conditional on the fixed
dataset, parameter and fitted twist. Freeze it before final streams. Then

    S_cv = S_N - C B,
    E[S_cv | B, data, twist] = E[S_N | data, twist].

The correction is OUTSIDE the normalized ratio; no ratio of expectations is
substituted for an expectation of a ratio. This preserves any finite-N bias.
The centering identity is a local derivation, not an attribution to author
code or a new claim about iAPF source faithfulness.

## Actual ancestor law and numerical contract

The experiment explicitly generates integer U in {0,...,2^23-1} and supplies
U/2^23 as FP32 ancestor uniforms. This power-of-two lattice is declared rather
than inferred from an undocumented random-number implementation. All paired
methods receive the same draws. Given the exact CDF passed to searchsorted
with side=right, define b_0=0, b_N=1 and b_i=clip(CDF_i,0,1) for i<N.
For M=2^23,

    q_i = [ceil(M b_i) - ceil(M b_{i-1})] / M.

This counts grid points in [b_{i-1},b_i), including ties, duplicate CDF entries
and the existing last-index clamp. It differs slightly from nominal softmax
weights. Exhaustive small-lattice enumeration must verify this law. An
explicit uniform-bit argument binds the optional controls to the caller's
sampling law; controls must not silently assume a law for arbitrary inputs.

The existing filter and additive score remain FP32/TF32/XLA. Control centering
and offline regression/application use FP64 TensorFlow to limit cancellation
and coefficient error. This is a declared diagnostic precision exception,
not FP32 default-readiness evidence. The expectation identity is exact in
real arithmetic on the represented pre-resampling scores; implementation
checks bound floating-point residuals. No resampling or filter protection is
changed, and no cross-compilation FP32 trajectory-equality claim is required.

## Evidence contract and intent ledger

- Primary comparator: raw Fisher score from the SAME kernel invocation.
- Mechanism criterion: at N4096, negative paired mean squared-error differences
  on each of four nonlinear datasets, with approximate bootstrap 99.75%
  intervals below zero (Bonferroni family 99% across these four comparisons).
  A pass nominates this control for broader validation only.
- Promotion vetoes: failed oracle/reference agreement; non-finite values;
  demonstrably incorrect centering or genealogy; active fitting bounds;
  unsupported mean-bias screen; observed conditional MSE loss to any heuristic.
- Continuation vetoes: invalid numerical reference, wrong sampling law,
  corrupted artifacts, data/calibration/final stream overlap, or exhausted
  budget. A noisy or ineffective control is a candidate rejection, not a
  continuation veto for the master research direction.
- Repair triggers: localized driver failure, singular/invalid coefficient
  computation, or finite but ineffective correction. Do not refit on final
  streams. A different candidate needs fresh calibration/confirmation.
- Explanatory diagnostics: retained SVD rank/conditioning, control means,
  coefficient sizes, calibration variance, final variance, bias and MCSE,
  fit iterations/bounds, tracing counts, numerical precision and runtime.
- Not concluded: unbiasedness, an HMC force, LEDH correctness, source-faithful
  author CV implementation, long-horizon validity, universal ranking or a
  new default. This stage does not repair earlier failed fitting holdouts.
- Evidence: versioned manifest, complete per-stream values/scores/controls,
  frozen coefficients, references, conditional comparisons, result.md and
  the active master checkpoint.

## Baselines, situations, and defaults

Constructed heuristic set: (1) bootstrap Fisher, removing fitted lookahead;
(2) no-resampling importance score, removing categorical genealogy noise;
(3) EKF, using first-order Gaussian conditioning; (4) UKF, using deterministic
nonlinear moment integration. Exact grid scores certify this T=2 diagnostic;
Kalman additionally certifies the affine sanity case. The fixed-label iAPF
derivative is retained as a different-target diagnostic, not an unbiased score
baseline. Evaluate each comparator separately on every dataset and size.

Fresh datasets: 1490 affine; 1500/1501 weak; 1510/1511 curved. Existing model,
theta and curvature definitions are retained to isolate the score mechanism.
They are diagnostic fixtures, not representative production claims. Check
grid refinement/tails and affine Kalman agreement before score interpretation.
Use N1024 and N4096 for nonlinear data, N4096 only for affine. These are a
bounded variance ladder, not tuned particle defaults.

Per dataset, the existing actual iAPF consumer fits the Gaussian-plus-floor
twist with the previously calibrated floor .001 and frozen fitter protocol.
This is a warm-start hypothesis, not a promoted setting: record validity,
bounds, objective and iterations. No final score is used to select the fit.
Use 192 independent calibration streams per scope and 128 untouched final
streams. The sample count gives 24 calibration observations per potentially
active control column; it is a budget choice whose risk is unstable regression.
Record SVD rank, singular values and calibration-versus-final variance.

Use the existing centered SVD least-squares control fit in FP64. No numerical
ridge is introduced: rank truncation at machine epsilon times matrix extent
is the existing roundoff-level pseudoinverse definition, and exact zero columns
are expected. Reject non-finite coefficients; do not reject structural rank
deficiency or tune the cutoff using final performance. Application subtracts
the known zero center, never an estimated calibration mean. No oracle score
enters fitting. Bootstrap intervals are approximate conditional uncertainty,
not exact confidence guarantees or population-wide rankings.

## Execution and budget

Output root: docs/plans/artifacts/younis-iapf-resampling-control-20260918-01/.
Use unique attemptNN directories; never overwrite prior evidence. Maximum four
attempts, 8000 filter-call charges, eight adaptive fits, 1800 cumulative driver
seconds and 600 CPU-test/GPU-probe seconds. Nominal charges: 5184 particle
calls plus 40 fitting and 60 moment-filter charges = 5284. Count any actual
fit calls above the conservative eight-call allowance. Repair retries share
these totals. Check remaining budget before every phase.

Environment: /home/chakwong/anaconda3/envs/tftwogpu/bin/python. CPU tests hide
GPU with CUDA_VISIBLE_DEVICES=-1. GPU commands use escalated access, verified
memory growth, RTX5080 UUID GPU-d54fdcfc-c6ed-dbe7-25c7-93f737e0f93a,
TF_FORCE_GPU_ALLOW_GROWTH=true, FP32/TF32/XLA filter kernels. Record commit,
dirty source hashes, actual command, seeds, hardware, precision exceptions,
wall time, all paths and trace counts. Shared GPU runtime is not a ranking.

1. Implement optional controls in the existing iAPF call chain and add exact
   enumeration plus independent genealogy checks. Run focused CPU tests.
2. Run the bounded diagnostic driver with --output-root <root>/attempt01.
   Its fresh calibration/final seed groups must differ. Save references,
   fit and calibration coefficients before starting final streams.
3. Assemble conditional MSE/variance/bias and heuristic verdicts, audit source
   hashes and inference, and update result, master and checkpoint.

## Skeptical pre-execution review

PASS for this bounded diagnostic, 2026-09-18. The original naive centering by
softmax weights would miss finite-CDF/lattice probabilities; the count formula
above repairs that flaw before execution. Exact centering outside the ratio
preserves the finite-N target. Independent coefficient calibration avoids
adaptive bias. The same invocation provides the right raw comparator; grid
and simple filters prevent an internal-only success criterion. No held-out
stream is reused, statistical families and stop conditions are explicit, and
GPU/FP64 exceptions are recorded. The strongest pre-mortem risks are weak
control correlation, overfit coefficients and fit-bound failure; the saved
conditional tables distinguish these from a mathematical identity failure.

## Attempt record and localized repair

Attempt01 stopped after 9.419787 seconds, before any fit/filter charges: the
driver named a nonexistent Kalman factory. Repaired the harness to call the
existing make_gaussian_kernel and select its value/score outputs. The focused
CPU repair check executes that factory and verifies all 11520 calibration/final
seed pairs are distinct. Sixteen focused CPU tests pass, including exhaustive
lattice centering and independent genealogy. Attempt01's manifest and terminal
error are preserved; initial startup console lines were not saved verbatim.
Attempt02 will resume the recorded budget, with a full redirected log. No
numerical kernel, target, criterion, dataset, precision or budget changed.
The localized repair passes skeptical review; no final streams were consumed.

## Post-run conditioning safety diagnostic (before execution)

The terminal comparison passes its four nonlinear primary checks, but the
affine calibration retains an eighth singular direction of size 8.64e-7
against a largest value 21.98, producing a coefficient of size 216232. This
triggers a dedicated Class C safety evaluation, not retuning on final scores.
For a scalar affine one-observation path, all six additive score components
are quadratic polynomials in (x0,x1). After subtracting their expectation,
their span has dimension at most five: x0, x1, x0^2, x0*x1, x1^2. Together
with the two initial controls, exact rank is at most seven. The eighth
direction is therefore roundoff contamination, not identifiable affine
score information. FP64 regression on FP32-produced scores used the wrong
precision when deciding whether this direction was resolved.

Evaluate the ordinary SVD threshold eps_input * max(rows,columns) * sigma_max
with eps_input=2^-23, solely on the nine SAVED CALIBRATION matrices. The
candidate is a numerical-rank safety hypothesis grounded in input precision;
it is not a universally proved error bound for arbitrary score computations.
Acceptance question: does it leave every resolved nonlinear coefficient and
calibration correction identical, while removing the affine spurious direction
and producing finite bounded coefficients/corrections? Do not compare final
MSE, select a threshold grid, alter saved campaign outputs or promote a runtime
default. Save both singular spectra, ranks, coefficients, correction norms
and exact non-harm verdicts. Use CPU FP64/XLA as an explicit independent
arithmetic diagnostic, no fits or filter calls; budget at most 60 additional
test/probe seconds (within the original 600). A pass nominates the rule for
fresh downstream validation before runtime adoption.

Skeptical review PASS: the affine polynomial rank bound supplies a falsifiable
failure explanation; the nonlinear controls are the known-good non-harm
cases. The criterion is conditioning and unchanged healthy outputs, not
primary-metric improvement. Final streams stay closed. This bounded safety
check does not activate another scientific campaign.

## Fresh downstream validation of the conditioning safeguard

The calibration safety probe passed: nonlinear coefficients/corrections were
identical; affine rank fell from eight to seven and its maximum coefficient
fell from 216232 to .01945. Continue within the remaining campaign capacity.
Use the five existing datasets and frozen twists at N4096, but new independent
particle groups `conditioning_confirmation_20260918_calibration_N4096` and
`conditioning_confirmation_20260918_final_N4096`. The existing fixed data are
appropriate here because the question is numerical protection of those same
controls, not performance on a new observation population. No old particle
stream, final error or oracle score enters coefficient fitting.

Add an optional input-precision declaration to the existing combination
factory; its default remains unchanged. Fit both FP64-threshold and declared
FP32-input-threshold coefficients on 192 fresh streams, apply both to the
same 128 fresh final score/control outputs, and save both. Baselines are the
uncorrected Fisher score and original correction in that invocation. Reuse
the deterministic EKF/UKF and numerical-reference values for these identical
datasets; stochastic heuristic superiority is not evaluated in this safety
stage and cannot be promoted from this evidence.

Acceptance is non-harm: all four resolved nonlinear coefficient arrays and
final corrected arrays must be exactly identical under the two cutoff rules;
the affine retained rank must be seven with the near-null direction removed,
finite coefficients/corrections, and a passing predeclared six-component 99%
Bonferroni mean-bias screen against Kalman. Numerical validity and unchanged
particle/Fisher outputs are hard vetoes. MSE and variance are explanatory,
never acceptance or tuning objectives. A pass validates this explicit optional
safeguard on these five scopes; default promotion and general rank-robustness
remain unsupported. No threshold grid, ridge or final-data retuning is allowed.

Charges: at most 1600 additional filter calls, zero new fits, one launch
`conditioning-confirmation01` (third campaign launch), 600 additional driver
seconds within the remaining 1698.483437, and 60 test seconds within 480.
Save a separate manifest, source closure, both coefficient matrices, all fresh
rows, safety decision and result supplement. Preserve prior results unchanged.

Skeptical review PASS before implementation/execution: the target-preserving
centering proof survives any independently fitted finite coefficient; the
change affects regression rank only. New seed groups separate both phases
from each other and the old final streams. The affine rank derivation, exact
nonlinear equality and Kalman check answer numerical safety directly. Reusing
fixed fits does not assert new-data generalization or promote fitter settings.
