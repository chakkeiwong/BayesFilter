# Fitted backward guides: bounded Gaussian transfer experiment

## Question and authority

Continue the user's requested execution on `ledh-refactor-with-policy-fix`.
The completed integration and Jacobian tests do not establish score accuracy.
This stage asks whether a quadratic policy fitted by the replicated controlled
SMC code can supply future-only proposal information to the actual shared LEDH
executor, and whether that mechanism merits nonlinear-model experiments.

This is an exploratory scalar Gaussian reference experiment. It does not
admit a canonical route, change a default, or establish results for SIR, range
bearing, KSC SV or predator–prey. The subsequent nonlinear experiments still
require their own fitting, tuning, references and untouched evaluations.
The old integration plan's exclusion of this stage is superseded by the
user's current instruction to continue, not by an inferred approval.

## Mathematical and implementation contract

Use x_t = alpha x_(t-1) + N(0,Q), y_t = H x_t + N(0,R), with
Q=0.09, R=0.25, H=1, initial mean zero and variance one. T=6 and N=32.
The existing six-parameter Gaussian adapter defines coordinates exactly.
The primary coordinate is log process standard deviation; all six analytical
score coordinates are retained. Compare to the exact Kalman total score.

Fit one backward quadratic update with the existing scalar controlled-SMC
implementation using 64 pilot particles and zero initial policy. Its policy
psi_t includes the current observation. Export only M_(t+1) psi_(t+1):
if psi_(t+1)(z)=exp(-a z^2-beta z-c), the future precision and linear term
are 2 alpha_ref^2 a/(1+2 Q_ref a) and
-alpha_ref beta/(1+2 Q_ref a). The terminal guide is constant.
The pilot reference and coefficients remain frozen during score evaluation.
Use an independently coded Gaussian backward recursion to check the export.
Fit failure, negative exported precision, projection, or material oracle
disagreement invalidates the fit; do not clip it into an accepted result.

The guided consumer must call the shared full LEDH executor, including UKF,
physical importance correction, OT, Contract E and dual-cap GenUT. Dynamic
guide tensors permit one stable graph across datasets. Verify this call chain
and analytical finite-program derivatives with focused executable tests.
A one-component Gaussian-mixture conversion checks compatibility with the
existing mixture guide. It is a representation test, not learned KDM,
multimodal fitting, or an independent KDM accuracy claim.

## Evidence and decision rules

Conditions, evaluated separately: independent states (alpha=0), informative
future (alpha=0.9), and a mismatched pilot (alpha=0.9, alpha_ref=0.2).
The last condition tests weak/misleading guidance rather than invalid data.
Constructed heuristic adversaries are:

| Comparator | Rationale |
| --- | --- |
| Plain shared LEDH | Measures the incremental effect of guidance |
| Current-observation quadratic guide | Tests whether looking ahead adds value |
| Bootstrap particle filter | Cheapest standard resampled particle estimator |
| Fully adapted SIS | Uses the exact current conditional Gaussian without resampling |
| Exact Kalman filter | Certifying oracle on this model, not a competitor to beat |

All particle arms use equal N, observations, initial/transition Gaussian
streams and declared uniform streams. Fit cost is reported separately and
included in guided total runtime. PF's derivative holds categorical ancestry
fixed; label it as the a.e. finite-program estimator, not an unbiased score.

Four calibration datasets per condition compare frozen future strengths
{0.25,1}; select on calibration mean squared log process-noise SD score error,
then evaluate the selected arm on four separate validation datasets without
reselection. Freeze before twelve
untouched datasets with two independent particle streams per dataset.
These tiny partitions nominate a guide only. They do not issue the existing
canonical scope-specific tuning artifact or support promotion/default claims.
Do not tune on heuristic outcomes or the untouched partition.

Primary descriptive metric: squared error of the log process-noise SD score against Kalman.
Preserve every coordinate and value error. Report paired dataset-level error
differences and 2,000 paired bootstrap 95% intervals, averaging the two particle
streams within each dataset. No cross-condition pooling. These small-sample
intervals are exploratory; no superiority/default claim is authorized.
Any heuristic loss in a condition is a promotion veto and the headline there.
Report a machine-readable heuristic verdict, including uncertainty.

Hard continuation vetoes: wrong target/indexing, missing analytical terms,
failed call-chain/zero-guide parity, nonfinite results, failed covariance or
reset validity, unverified device policy, missing evidence, or budget exhaustion.
Candidate rejection is not a continuation veto. The preplanned repair for an
overstrong full guide is the separately evaluated quarter-strength guide.
Runtime, ESS, fit conditioning/residuals and correction diagnostics explain
results; they cannot promote a candidate. No unbiasedness of the reset filter,
HMC readiness or nonlinear-model improvement follows.

## Defaults, safety and skeptical audit

Gaussian closure makes one quadratic fit sufficient in exact arithmetic:
the fitting basis spans the exact backward recursion. This is a correctness
probe, not evidence of adequate capacity for nonlinear models. Pilot size 64
is a bounded convenience; conditioning and oracle error expose poor support.
T=6/N=32 are small reference scopes, not settings transferred to other models.
The alpha conditions deliberately include the zero-information limit.

Use GPU/XLA FP64 as an explicit reference exception to the TF32 production
direction, isolating fitting and score error from roundoff. CPU-only tests
hide GPU devices. Memory growth is required before GPU initialization.
OT epsilon=1 and five Sinkhorn steps are the user's untuned comparator.
Balance steps=32 target marginal residuals; record them. Exact chunk policy
gives K=N. No global defaults change.

The full executor's correction controls are explicit reference choices:
24 Euler flow steps; ridge 1e-6 (about 1e-5 of Q, checked against a tenfold
smaller value); correction/pairwise steps 1/1; strengths 0.2/0.02;
LM damping 0.01 and scale floor 1e-4; trust radius 0.5; pairwise RMS cap 2;
coordinate cap 0.94, power 8, identity radius 0; moment safety false.
These retain the complete existing correction. Damping regularizes its
scaled normal equations; the step fractions and caps limit a single correction
relative to the standardized cloud. They remain hypotheses, not calibrated
defaults. Before interpreting scores, compare 12/24/48 flow steps and ridge
1e-6/1e-7 on one calibration fixture and preserve correction diagnostics.
Instability blocks numerical interpretation and triggers a bounded repair;
it is not evidence against future guidance.

Skeptical audit passed for this bounded question: the oracle shares the
physical model and parameterization; future-only export avoids duplicating
the current observation by mistake; coefficients are frozen, not refitted
inside differentiation; finite-program derivatives are distinguished from
model scores; independent conditions and cheap baselines prevent misleading
pooled comparisons. Learned KDM and nonlinear model fitting remain open.
The weak point is the short, scalar, exactly representable target. A successful
run can justify the next experiment only.

## Execution and budget

1. Implement export, stable shared consumer and focused correctness tests.
2. Trusted GPU/memory probe; tiny XLA smoke and numerical-safety diagnostic.
3. Execute calibration, validation and untouched reference experiment.
4. Preserve results and uncertainty, update mathematical note, run bounded
   MathDevMCP audit, build document and update active checkpoint.

Budget: 1,200 seconds CPU tests/build and 1,800 seconds GPU work including
compilation; at most three localized repair attempts, with fresh output roots.
No neural training, new external agent, environment mutation or branch change.
Artifacts: `docs/plans/artifacts/ledh-fitted-guide-20261016/`, with versioned
attempt directories. Each attempt records command, commit/source hashes,
environment, seeds/data definition, device policy, elapsed time and outcome.
Stop at the budget or a genuine continuation veto; retain failed evidence.

## Recorded numerical repair, October 16

Attempt01 used 37.247 GPU seconds and stopped before any fitting comparison.
The 12/24-step log process-noise SD scores differed by 0.222332, exceeding the predeclared
0.0998874 sensitivity tolerance. Both finite-program evaluations were valid;
this is unresolved discretization, not evidence against future guidance.
The initial harness stopped at the coarse comparison and never measured the
finer 48-step approximation. That stopping order cannot distinguish convergent
Euler error from an unresolved calculation. Preserve the failed attempt.

Repair 2 performs a bounded 24/48/96/192 refinement on the same calibration
fixture, before selection or heldout evaluation. Freeze the first finer
resolution whose adjacent log process-noise SD score change is below the unchanged
0.05*(1+abs(exact log process-noise SD score)) tolerance and does not increase relative to
the previous refinement change. This is a finite sensitivity screen, not a
proof of convergence or a score-accuracy tuning objective. Apply the frozen
resolution to every LEDH arm and test the original tenfold ridge perturbation
at that resolution with the unchanged 0.001*(1+abs(exact score)) threshold.
Failure through 192 steps or of the ridge check stops interpretation again.
No other controls, target, data partitions, comparisons or claims change.

Skeptical repair audit: a coarse Euler difference is expected and is not itself
a reason to abandon refinement; testing only the fine approximation would
hide that evidence, so all resolutions are retained. Selection uses numerical
sensitivity, never oracle MSE or heldout performance. A one-fixture screen
cannot establish uniform accuracy; every comparison still checks finite
values, factorization/correction validity, covariance and transport diagnostics.
The total budget remains 1,800 GPU seconds, with 1,762.753 seconds remaining
before attempt02; about 165.35 CPU test seconds have been used. Repair 1 fixed
a test-only TensorFlow dtype argument; two of three repairs are now allocated.

## Attempt02 result and decision

The repair completed on the same LEDH checkout in 375.893 GPU seconds. The
24-to-48 flow change in the log process-noise SD score was 0.0988435, below the unchanged
0.0998874 threshold, and the tenfold ridge perturbation changed that score by
2.3336e-6, below 0.00199775. The bounded reference sensitivity screen passed
and froze 48 flow substeps for this experiment. This is local numerical
stability evidence, not a convergence proof; the original 12-to-24 difference
of 0.222332 remains preserved in attempt01.

The complete run produced 624 finite rows: 48 calibration, 40 validation and
120 untouched rows per condition. The exact scalar fit agreed with the
independent backward Gaussian recursion with maximum discrepancy 1.55e-12
over all conditions (7.35e-14 for the mismatched pilot). The untouched exploratory mean squared log process-noise SD score errors were,
respectively, 0.24583/0.55760/8.31328/6.84017/9.08690 for adapted SIS/
bootstrap PF/current guide/future strength 1/plain LEDH in the informative
condition, and 0.24066/1.01846/10.34456/10.41046/10.40836 for adapted SIS/
bootstrap PF/current guide/future strength 0.25/plain LEDH in the mismatched-
pilot condition. In the independent condition the future guide and plain LEDH
were identical to the recorded precision. Every condition triggered the
predeclared heuristic promotion veto because a simple comparator had lower
error. Bootstrap intervals are exploratory; no confirmatory ranking is established.

Decision: the replicated scalar fit, future-only export, shared full-consumer
call chain, numerical validity and artifact production pass their engineering
checks. Future guidance is not promoted as a better score estimator, default,
canonical route, HMC route, or nonlinear-model solution. The informative
condition is a nomination signal, but its guide still loses descriptively to
adapted SIS and bootstrap PF. The next smallest question is why the plain and guided LEDH scores remain
inaccurate relative to the Gaussian oracle and simple particle comparators.
Diagnose that baseline error under a separately specified reference scope
before expanding to nonlinear fitting. Nonlinear work then needs fresh
scope-specific tuning and equal-compute heuristic comparisons; no tuning or
claim may reuse this scalar setting.

Post-run label correction: coordinate 1 is log process-noise standard
deviation, as declared above and in `gaussian_tf.parameterized_model`, where
Q=exp(2*theta[1]). The attempt JSON keys containing `log_q` are legacy names
for this coordinate, not derivatives with respect to log variance. A log-Q
variance score would be half this score and its squared error one quarter.
The calculation, selection and thresholds are unchanged. The original attempt
files and source hashes are preserved. `postrun-audit.json` maps the names,
checks all 624 rows and reports validation, all six coordinates and total
reference timing; see the result note under the versioned artifact root.
