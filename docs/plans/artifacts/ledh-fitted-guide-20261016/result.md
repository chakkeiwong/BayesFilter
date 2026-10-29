# Fitted backward-guide pilot: result

The selected future guide fails the predeclared heuristic promotion check in
all three conditions. The scalar controlled-SMC fit and its integration into
the shared LEDH executor pass the engineering checks, but these results do
not justify promoting guidance as a better model-score estimator. The
quarter-strength repair was included in calibration and selected where its
calibration error was smaller; this negative result does not reject the
research direction.

## Target and computed quantity

The scientific reference is the exact Kalman likelihood score. The LEDH
implementation computes the analytical total derivative of its finite value
program, conditional on frozen guide coefficients and realized random inputs.
These are different quantities at finite particle count. The focused
finite-difference check supports the implemented derivative; the score-error
experiment measures its discrepancy from the model score.

The primary coordinate is **log process-noise standard deviation**:
Q=exp(2 theta[1]), so the score in coordinate 1 equals twice the derivative
with respect to log process variance. The original JSON keys containing
`log_q` are misnamed aliases for this coordinate. Its MSE would be divided by
four if expressed in log-variance coordinates. All calculations, selection
rules and thresholds used coordinate 1 consistently; no values have been
rescaled or recomputed. `postrun-audit.json` records the corrected labels.

The actual replicated scalar controlled-SMC forward/backward fit supplied the
future-only guide. Its export agreed with the independent Gaussian information
recursion within 1.55e-12 over all recorded fits. Both concern the pilot model.
The evaluation adapter's inherited float32 construction of the constant 0.15
makes nominally unit loading and initial variance differ from one by about
5.2e-9 in FP64. Data, Kalman oracle and particle arms all share that adapter;
pilot exactness therefore refers to the nominal pilot, not bitwise equality
with the evaluation model. This tiny discrepancy has not been isolated as a
cause of score error.

The consumer retains analytical directions, the complete transition and
observation weighting, OT, Contract E and the dual-cap correction. A
one-component mixture check covers KDM representation compatibility only;
no KDM was trained. The bootstrap comparator differentiates its realized
program with categorical ancestry fixed, without an unbiased expectation-
gradient claim.

## Execution and numerical validity

The run stayed on `ledh-refactor-with-policy-fix` at commit
`ea1018de1da55ed02a29a8d60f08a6827fce8a42`, using the full shared executor,
FP64 TensorFlow/XLA and verified GPU memory growth on an RTX 4080 SUPER.
This is a nondefault Gaussian reference experiment. The GPU was nonexclusive.

Attempt01 stopped after 37.247 seconds: the 12/24 flow sensitivity difference
was 0.222332, above the 0.0998874 threshold. It produced no comparison rows.
The bounded repair in attempt02 selected 48 substeps after a 24/48 change of
0.0988435, narrowly below that threshold. The tenfold ridge perturbation
changed the score by 2.3336e-6, below 0.00199775. The local screen passed;
it does not establish uniform numerical accuracy or convergence.
Attempt02 completed in 375.893 seconds; total GPU campaign time was 413.140
of the 1,800 seconds budgeted. Original attempt files are preserved.

There are 624 finite rows: 48 calibration, 40 validation and 120 untouched
rows per condition. Calibration selected future strength 0.25 for the
independent and mismatched-pilot conditions and 1 for the informative case.
Selection was frozen before validation and untouched data. No continuation
veto fired in attempt02. The post-run audit verifies unique rows, disjoint
partitions, shared data/random streams across arms, recorded errors, valid
unprojected fits, zero terminal guides, exact independent-case parity, and
unchanged hashes for every source recorded by the runner.

## Score comparisons

Untouched mean squared error of the log process-noise SD score:

| Condition | Adapted SIS | Bootstrap PF | Current guide | Selected future guide | Plain LEDH |
| --- | ---: | ---: | ---: | ---: | ---: |
| Independent, alpha=0 | 2.78e-31 | 0.03403 | 0.54130 | 2.4319e-6 | 2.4319e-6 |
| Informative, alpha=0.9 | 0.24583 | 0.55760 | 8.31328 | 6.84017 | 9.08690 |
| Pilot alpha=0.2, evaluation alpha=0.9 | 0.24066 | 1.01846 | 10.34456 | 10.41046 | 10.40836 |

Each condition records `PROMOTION_VETO_DESCRIPTIVE_HEURISTIC_LOSS`.
In the informative case, the future-minus-plain paired MSE difference is
-2.24673, with exploratory 95% bootstrap interval [-4.01378, -0.69188].
This is a scoped signal favoring further investigation, not a confirmatory
ranking or a pass against the simpler particle methods. Its difference from
adapted SIS is +6.59434 [0.89049, 14.06430]. For the mismatched pilot, the
future-minus-plain difference is +0.00210 [-0.07445, 0.08405].

Validation was reported without reselection:

| Condition | Adapted SIS | Bootstrap PF | Current guide | Selected future guide | Plain LEDH |
| --- | ---: | ---: | ---: | ---: | ---: |
| Independent | 0 | 0.05346 | 0.30896 | 6.7e-6 | 6.7e-6 |
| Informative | 0.15221 | 1.46021 | 0.32699 | 0.62713 | 0.84987 |
| Mismatched pilot | 4.14535 | 5.27732 | 285.08762 | 258.33234 | 255.69577 |

The large variation between validation and untouched partitions is a major
uncertainty. Conditions use different datasets, so their differences do not
identify a causal effect of pilot mismatch. All six score coordinates and
value errors remain in the rows and the post-run aggregate.

Guided untouched mean evaluation-plus-fit time ranges from 0.817 to 0.927
seconds per row. The included 0.0069–0.0081 seconds of fit timing also contains
the independent oracle and validity checks. These are explanatory timings,
not equal-compute performance evidence.

## Decision and inference

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Retain scalar integration as an engineering probe | Fit/oracle agreement and shared consumer checks pass | No attempt02 continuation veto | Scalar exact closure only | Diagnose Gaussian baseline score error | No nonlinear or model-score correctness |
| Do not promote future guidance | Heuristic dominance fails | Promotion veto in every condition | Small partitions and large error variation | Separate baseline error from incremental guide effects | No superiority, default, canonical or HMC status |
| Keep controls exploratory | Local flow/ridge screen passes | Earlier coarse-flow failure retained | One fixture; threshold narrowly passed | Recheck under each new scope | No transfer of 48 steps or guide strengths |

| Evidence class | Status |
| --- | --- |
| Hard veto screen | Attempt01 failed numerical sensitivity; attempt02 passed the recorded continuation checks. |
| Statistically supported ranking | No confirmatory ranking; paired intervals are exploratory and do not establish broad superiority. |
| Descriptive-only differences | Informative guidance has lower observed error than plain LEDH; simple SIS/PF comparators remain more accurate. Mismatched-pilot guidance gives no clear gain. |
| Default readiness | Not ready; no canonical scope-specific tuning artifact was issued. |
| Next evidence needed | A separately planned Gaussian baseline-error diagnostic, then fresh nonlinear fitting/tuning with untouched data, equal-compute heuristics and repeated-seed uncertainty. |

The main alternative explanations are finite-particle/reset approximation
error, flow discretization and dataset-dependent variance; this experiment
does not separate them. The weakest evidence is the short scalar scope and
the large partition variation. A fresh, properly tuned run that clears every
heuristic condition with adequate uncertainty could overturn the promotion
decision. The current result rejects these candidate settings, not the
algebraic export or the possibility of useful future guidance.

## Verification and audit limits

Seven new tests and 37 shared regression tests pass. The new tests exercise
actual replicated fitting, invalid-integral rejection, KDM representation
parity, the shared consumer call chain, zero-guide parity and a total
finite-difference derivative. The initial test-only dtype-argument failure
and its repair remain in the logs.

The manuscript includes Proposition 21 and its square-completion and
backward-induction proof. It builds to 65 pages; final compilation has no
unresolved references or warnings, and the changed proof pages were visually
inspected. MathDev's derivation audit is inconclusive: it certified none of
six obligations. Its later structural audit treats prose words as missing
code terms and does not establish a mathematical defect or a proof. The
bounded local equation/call-chain audit and these tool limitations are in
`math-code-audit.md`. Independent mathematical certification remains open.

## Reproducibility paths

- `attempt01/manifest.json`, `attempt01/safety.json`, `attempt01.log`
- `attempt02/manifest.json`, `attempt02/safety.json`, `attempt02/selection.json`
- `attempt02/rows.jsonl`, `attempt02/result.json`, `attempt02.log`
- `audit_results.py`, `postrun-audit.json`
- `tests-attempt02.xml`, `regression.xml`
- `latex/ledh_younis_kdm_score.pdf`, `latex/pass4-final.log`
- `mathdev-fitted-future.json`, `mathdev-structural-audit.json`, `math-code-audit.md`
