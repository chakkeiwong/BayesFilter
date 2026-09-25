# SGQF initialization: result and next decision

The unchanged SGQF joint has lower observed audit discrepancy than the fitted
d4 TT on both first-transition targets. That is the headline: TT must not be presumed
to improve an already available approximation. SGQF initialization is useful
in the controlled comparison—its median audit error is lower than the generic
start in each d4 case—but it does not remove this early-transition deficit.
Later recursive targets show a different pattern: fitted corrections have
lower observed error than SGQF alone. These findings justify retaining SGQF
as an explicit candidate and testing its initialization, rather than always
discarding it after constructing regression rows.

This completes the initialization diagnostic in
[A04](observation-aware-tt-master-amendment-04-sgqf-initialization-20260915.md),
reviewed before execution. The research run completed all 24 cases in
98.306 seconds on RTX 5080, float64, GPU/XLA with verified memory growth.
Eighteen CPU-only tests passed; the separate 8.247-second GPU smoke used
RTX 4080 SUPER because CUDA ordinals differed from nvidia-smi. The full run
selected RTX 5080 by UUID. The smoke is not RTX 5080 timing evidence.

## What was compared

The physical target is the observation likelihood times transition density
times either the preceding SGQF Gaussian or the **saved preceding TT marginal**.
The latter is the actual recursive fitting target from campaign-02, not the
exact filtering law. Charts, observations and saved retained densities are
unchanged and input/target-source hashes are recorded. Each of three fresh row
designs uses 1024 training, 4096 validation and 8192 audit rows. Both fitting
initializations use the same target, rows, weights, degree/rank 3, training-only
scalar normalization and L1/optimization budget. L1 is selected separately by
validation signed-amplitude RMS; audit never selects it. These new row designs
do not turn the exposed observations into independent validation sequences.

The initializer is the analytic Hermite projection of sqrt(q_SGQF/rho),
compressed to pair rank 3. Its conversion error is measured before fitting.
The exact SGQF comparator keeps the original correlated Gaussian; the product
of marginal guides and the predictive Gaussian joint provide two additional
cheap comparators. The saved old TT is also evaluated, but comparisons with
that old fit include changed row designs and normalization; they do not isolate
initialization. The generic-versus-SGQF comparison does isolate the two starts
under the current fixed protocol, including their different coefficient gauges.

## Common-target audit results

The table reports median empirical squared Hellinger discrepancy over the
three row designs. Smaller values mean closer density agreement with the
specified target. These are descriptive values, not statistically supported
rankings. TT densities in this table include the inherited 5% product-Gaussian
defensive component.

| Dimension, time, incoming law | Exact SGQF joint | SGQF converted to TT, before fitting | Generic-start fit | SGQF-start fit | Validation safeguard choices |
| --- | ---: | ---: | ---: | ---: | --- |
| d1, t1, Gaussian | .001068 | .001390 | .000937 | .000937 | fitted TT, 3/3 |
| d1, t1, recursive TT | .002419 | .002805 | .001204 | .001204 | fitted TT, 3/3 |
| d1, t17, recursive TT | .003634 | .003900 | .000855 | .000855 | fitted TT, 3/3 |
| d1, t18, recursive TT | .005025 | .005029 | .000798 | .000798 | fitted TT, 3/3 |
| d4, t1, Gaussian | .001866 | .005384 | .009101 | .007671 | exact SGQF, 3/3 |
| d4, t1, recursive TT | .005701 | .009427 | .010610 | .008716 | exact SGQF, 3/3 |
| d4, t17, recursive TT | .010001 | .011566 | .005630 | .005092 | generic 1/3, SGQF-start 2/3 |
| d4, t18, recursive TT | .204194 | .204999 | .121157 | .090554 | generic-start fit, 3/3 |

The d4 first-transition example separates three losses. Against the
Gaussian-incoming target, SGQF's median audit discrepancy is .001866;
conversion without the defensive component gives .003206; adding 5% product
Gaussian gives .005384; SGQF-initialized fitting gives .007671. These are
distinct observed effects. The conversion's exact signed-amplitude L2 error
against the Gaussian is .002887: .002850 comes from polynomial truncation and
.0000371 from rank compression. Most conversion error here is therefore due
to the degree restriction, not rank-3 compression. This does not establish the
best attainable polynomial approximation to the nonlinear target.

At d4 t18, the Gaussian comparator is far from the **saved recursive TT target**.
That does not establish that SGQF is far from the true filter. Earlier TT error
and current observation non-Gaussianity have not been separated there. Fresh
fits are much closer to this target than the saved TT, whose median audit
discrepancy is .509100; this comparison includes new row designs and cannot be
credited entirely to SGQF initialization.

## What the safeguard establishes

The selector keeps the exact SGQF joint, the converted initializer, and the two
base-budget fitted families as candidates. Its empirical validation discrepancy
is no worse than SGQF in **24/24 cases**, by construction. The choice is written
before audit rows are generated. It is no worse on audit in **23/24 cases**.
The exception is d1/t1/Gaussian, row design 1: audit discrepancy increases by
0.000017008 relative to SGQF. This directly rules out an unconditional claim
that validation selection guarantees better unseen accuracy.

The selector also misses a descriptively better audit candidate: at d4 t18 it
selects the generic start in all three designs, although the SGQF start has
lower audit discrepancy. Thus the empirical rule preserves its validation
baseline, but does not reliably identify the best audit candidate. Do not
retroactively select using audit rows or tune the selector to these cases.
Against the constructed three-Gaussian heuristic set, the SGQF-start fit passes
the empirical audit screen in 11/12 d1 designs and 6/12 d4 designs. The d4
failures are all six first-transition designs; d1's failure is the audit
exception above. The other heuristic arms and their conditional scores are
preserved in every case, rather than omitted from the comparison.

A pointwise density lower bound is the wrong mathematical requirement. If two
normalized densities satisfy q1>=q0 everywhere, integrating gives zero integral
for the nonnegative difference, so they coincide almost everywhere. The useful
requirement is a declared accuracy or efficiency criterion with a preserved
SGQF option. No filter-level fallback has been installed by this diagnostic.

## Solver, calibration and validity

The Gaussian coefficients agree with independent low-dimensional Gauss–Hermite
quadrature; the identity Gaussian limit, full-rank reconstruction and joint
conditional algebra pass. Explicit-start fitting reaches the same compiled
fitter, and the extracted default initializer reproduces its previous values.
All per-case graph/XLA coefficient comparisons and finite/SPD/objective checks
pass. No numerical continuation veto fired; no infrastructure retry was needed.
The initializer remains a diagnostic full coefficient tensor, exponential in
dimension, and is not promoted as a scalable runtime implementation.

More optimization is not a general repair. The doubled schedule restarts from
the same original initialization at the selected L1. In d4 t18 it reduces the
generic fit's median audit discrepancy from .121157 to .077415; in the Gaussian
first-transition case it changes .009101 to .010072. SGQF-start KKT residuals
are smaller, but KKT does not certify global fit quality, and the longer runs
were excluded from selection. L1 depends on coefficient gauge, so neither
initialization is established as a gauge-independent optimum.

The 5% defensive component adds observed first-transition error. It is an
inherited comparison setting with no calibration evidence, not a newly accepted
default. Removing it solely to improve this metric would violate the planned
coverage/non-harm calibration requirement. That dedicated phase remains open.

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Retain SGQF initialization as a candidate | Lower median audit H2 than generic start in each d4 case | No validity failure; first-transition heuristic underperformance blocks promotion | Three row designs on exposed targets; gauge and finite-panel effects | Carry both initialization and exact SGQF baseline into the next reviewed protocol | Universal improvement or fitting convergence |
| Preserve exact SGQF as an explicit candidate | Chosen in all six d4 first-transition designs | Audit safeguard fails once; no unconditional accuracy guarantee | Validation coverage, inherited TT target error | Design and test a filter consumer with declared validation/fallback semantics | A filter fallback is already implemented or validated |
| Do not promote the TT configuration | d4 first-transition fits lose to SGQF | Empirical heuristic promotion veto, not a continuation veto | True downstream accuracy and independent-sequence uncertainty | Continue phase 8 calibration, then phase 9 only after freezing a candidate | Rejection of the SGQF/TT research direction |

| Inference status | Result |
| --- | --- |
| Hard validity screen | All executed identities, covariance/finite/projection guards and artifact checks pass. |
| Statistically supported ranking | None; no independent-sequence comparison or uncertainty analysis was run. |
| Descriptive-only differences | All H2/RMS/KKT/runtime differences and row-design medians above. |
| Default-readiness | Not established; exact SGQF wins salient cases, mass calibration and downstream validation remain open. |
| Next evidence needed | Dedicated defensive-mass coverage/non-harm calibration, a frozen implemented consumer, then untouched sequences with the complete reference/heuristic ladder and paired uncertainty. |

## Reproducibility, review and next program phase

The actual command, environment, sources, fixture hashes, seed policy, hardware,
allocator peaks and times are in
`../benchmarks/artifacts/observation_tt_sgqf_initialization_20260915/attempt-01/run_manifest.json`.
`result.json` contains the 24 case decisions; `summary.json` records medians and
ranges; each case JSON preserves L1 arms, initial and fitted cores, diagnostics
and the frozen choice. Full logs and 18-test outputs are in
`artifacts/observation-tt-sgqf-initialization-20260915-01/`.
The code changes only expose the unchanged generic initializer and correctly
label supplied starts; the analytic initializer and runner are diagnostic code.
The touched runtime module was already untracked in this checkout. Its pre-edit
source was reconstructed by reversing only these edits and verified against
the campaign-02 source hash; the exact before-copy and patch are preserved in
the execution record. All six recorded launch dependencies still match at
terminal verification. No unrelated working-tree edits were reverted.

Post-run red team: the strongest alternative explanation is that finite
training/validation coverage and coefficient-gauge regularization dominate
initialization, while previous TT errors change the target being approximated.
The weakest evidence is inference from three row designs on exposed sequences.
A consistent independent-sequence downstream failure would overturn the case
for adopting the candidate, even if its fitting losses improve.

A04's initialization trial is complete and its single research launch is closed.
[Terminal interpretation review](../reviews/observation-aware-tt-sgqf-initialization-result-review-20260915.md)
returned AGREE after a timeout and a bounded clarification of time scope and
the predeclared promotion rule. This does not certify uninspected code.
No extra attempt is justified merely to change these outcomes. The next master
action is phase 8's dedicated defensive-mass calibration and a reviewed design
for consuming the preserved SGQF candidate, followed by the unchanged phase-9
independent-sequence test. This requires a concrete protocol amendment before
new numerical work; it does not require another grant of the already authorized
remaining five-hour budget. Remaining time is tracked in
`artifacts/observation-tt-sgqf-initialization-20260915-01/budget.json`.
