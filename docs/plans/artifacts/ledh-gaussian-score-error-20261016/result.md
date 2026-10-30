# Gaussian LEDH score-error diagnostic — 16 October 2026

The Gaussian baseline is not ready for a new guide-improvement claim. The
diagnostic completed 255 of 280 planned evaluations before the shared LEDH
validity guard rejected a case. On the completed cases, analytical recursion
agrees with finite differences of the same finite program, but persistent
Gaussian model-score errors remain large and fail the heuristic promotion
screen. These are separate findings: a correct derivative of a numerical
likelihood need not accurately estimate the model likelihood score.

## Scope and provenance

Execution used branch `ledh-refactor-with-policy-fix`, worktree
`/home/chakwong/BayesFilter/.localresources/worktrees/ledh-iapf-execution-20260922`,
base commit `ea1018de1da55ed02a29a8d60f08a6827fce8a42`, and the source hashes in
`attempt01/manifest.json` and `attempt02/manifest.json`. The
[pre-run plan](../../ledh-gaussian-score-error-20261016.md) defines the evidence
contract, assumptions, conditional heuristic comparisons and stop conditions.
The reference execution was TensorFlow FP64/GPU/XLA, not the TF32 production
target. GPU memory growth, devices, commands, environment, data identity, seeds,
controls and wall times are recorded in the manifests. CPU replays deliberately
hide GPUs with `CUDA_VISIBLE_DEVICES=-1`.

The scalar model has six observations, process variance 0.09 and observation
variance 0.25. Independent and persistent conditions use transition coefficient
0 and 0.9 respectively; the stress condition perturbs the third observation.
There are six ordinary datasets per condition and two stress datasets, two
particle streams, particle counts 32/128, and flow resolutions 48/192. All six
parameter derivatives are checked on the finite-difference fixtures; the
predeclared primary score coordinate is log process-noise standard deviation.
The exact Kalman model score is the reference. Prior-proposal SIS, bootstrap PF
and fully adapted SIS are the constructed simple comparators.

The owner-requested development pair remains epsilon 1 / five Sinkhorn
iterations. Terminal balancing remains 32 in these runs. Neither these values
nor the inherited ridge, correction and cap controls are scope-tuned defaults.
No learned component is trained in this diagnostic.

## Execution and validity failure

Attempt01 selected the busy RTX 4080 SUPER through CUDA index 0. It was
interrupted after 493.339 seconds with 76 valid rows preserved. Its old
`running_grid` manifest status is stale after exit 130; it is not a live run.
Attempt02 used the explicit RTX 5080 UUID, checked numerical dependencies and
replayed two saved cases before reusing those rows. Maximum scaled value/score
differences in that cross-device replay were below 1e-15. It ran another
275.717 seconds and stopped at the validity veto, leaving 255 unique rows: 102
LEDH evaluations and 153 heuristic evaluations. The remaining 25 rows were not
executed. No failed score is included as a zero-valued observation.

The next missing cell identifies the rejected case as persistent, dataset 3,
stream 1, N=128 and 48 flow substeps. The original GPU harness did not save its
rejected tensors. A CPU reconstruction using the saved observations, parameters
and stream seed reproduces rejection with both XLA and non-XLA execution.
GPU/CPU equality of the failed random tensors was not independently checked;
the exact guard localization below applies to the archived CPU replay.

At zero-based time 5, transport-column total-variation residual is
0.00027054666504, exceeding the unchanged 0.0001 guard.
`reset_transport_valid` and cumulative `program_valid` become false in all six
directions. The UKF prediction/update, reset numerical, higher-moment, ancestry,
proposal and callback flags remain valid; the traced numerical quantities are
finite and the recorded covariance minima are positive. The shared executor
then intentionally returns minus-infinity value and zero derivatives. Those
zeros are rejection sentinels, not score estimates.

This is a genuine continuation veto for the grid, not evidence that the
analytical derivative is wrong or that backward guidance cannot work. The
observed residual motivates a separately calibrated terminal-balancing repair;
it does not justify weakening the guard or silently changing controls.

## Model-score error on completed cases

The table reports primary-coordinate mean squared error against the exact
Kalman score. Each row uses the same cases for all five arms. These are
descriptive sample averages, not uncertainty-supported rankings.

| Condition | N | Cases/arm | LEDH 48 | LEDH 192 | Prior SIS | Bootstrap PF | Adapted SIS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Independent | 32 | 12 | 1.3247e-6 | 8.3327e-8 | 0.22922 | 0.07224 | 3.86e-31 |
| Independent | 128 | 12 | 1.4662e-6 | 9.2199e-8 | 0.07401 | 0.02156 | 3.86e-31 |
| Persistent | 32 | 12 | 42.33438 | 42.24491 | 5.04072 | 0.57595 | 0.26080 |
| Persistent | 128 | 7 | 65.24796 | 65.39210 | 3.79138 | 0.64134 | 0.21888 |
| Persistent stress | 32 | 4 | 163.86941 | 163.85305 | 9.00616 | 3.86330 | 1.14375 |
| Persistent stress | 128 | 4 | 203.37961 | 202.07075 | 1.07287 | 2.68591 | 0.19761 |

Every tested LEDH configuration fails the predeclared conditional heuristic
promotion screen: adapted SIS has lower observed MSE in the independent
condition, and all three heuristics have lower observed MSE in the persistent
conditions. The adapted independent case is an exact Gaussian limiting case,
not evidence of universal superiority of adapted SIS.

The persistent N=128 group contains only seven cases. Comparing its average
directly with the twelve-case N=32 average would confound particle count with
coverage. On the seven matched cases, 48-step LEDH MSE changes from 54.64921 to
65.24796, and 192-step MSE from 54.56743 to 65.39210. Thus this small diagnostic
does not show that increasing N repairs the persistent-condition error. It
does not establish an asymptotic particle-count trend.

Increasing flow resolution reduces independent-condition MSE by about sixteen
fold, but leaves persistent-condition errors nearly unchanged. This makes
flow discretization alone an inadequate explanation for the observed persistent
errors; it does not prove convergence of the continuum-flow approximation.

## What the derivative and reset checks establish

Across two six-coordinate fixtures, the maximum scaled finite-difference error
of the analytical finite-program score is 2.458481807e-9. The corresponding
independent Kalman derivative check is 1.947649868e-9. These checks support the
implemented derivative on those fixtures only. They cannot certify model-score
accuracy, all parameter values, or a rejected program.

The reset restores the weighted input mean and variance to machine precision
on completed cases: maximum differences are approximately 1.34e-15 and
8.33e-17. Nevertheless, the weighted child variance is already too small
before the first reset. In the persistent condition, first-step ESS is about
13.7% of N and the weighted variance is approximately 0.149 of the Kalman
variance at N=32 and 0.328 at N=128 on their recorded case sets. The stress
condition also shows first-step concentration. This rules out attributing the
entire discrepancy to later reset covariance restoration.

Small finite clouds, the physical-prior/proposal weighting relationship, and
shape changes after reset remain candidate explanations. Frequent cap
activation is explanatory only: it does not by itself identify a cause or
justify changing a cap. A discriminating next diagnostic should inspect the
first-step proposal and weights against the exact Gaussian conditional.

## Observability repair and regression evidence

The shared score trace now retains the already-computed transport and proposal
validity flags. Control diagnostics preserve validity/finite flags and report
the transport-column total-variation residual. This adds seven lines across
the shared score and diagnostics modules; it changes neither calculations nor
acceptance thresholds. The GPU runner now archives rejected inputs, outputs
and diagnostics before raising its existing veto. That capture path was not
used retroactively to invent missing GPU evidence.

`observability-repair.patch` preserves the shared-module change.
`cpu-veto-observability/manifest.json` records before/after source hashes and
shows zero difference in every previously retained diagnostic on both CPU
replays; rejected values and score sentinels are unchanged. Eight focused
tests pass before and after repair, and all seven shared-consumer regression
tests pass (`tests.xml`, `observability-tests.xml`,
`shared-consumer-tests.xml`). These are engineering checks, not promotion
evidence. No numerical repair has yet been validated.

`postrun-audit.json` was generated before the observability edits and verifies
the run-time source hashes, unique coverage, matched comparator cases,
increment reconstruction and finite-difference results. Its hash-match field
refers to that audit time, not the subsequently edited working tree. The
historical analyzer deliberately fails its unchanged-source precondition in
the newer tree. `run-source.patch` preserves the run-time changes against the
recorded Git commit: all 253 source hashes were verified, including exact
reconstruction of the three files edited after execution. `run-source.json`
records that check and the patch hash. Reproduce the historical analyzer from
those versions; do not relabel repaired sources as the GPU campaign sources.

The manuscript includes the results and now places balancing and first-step
weighting before subsequent score comparisons. Its 66-page PDF builds without
undefined citations/references or layout warnings in the final pass. The
updated numerical discussion and provenance on pages 60 and 63 were visually
inspected. Build logs and the PDF are in `latex/`. No new independent
mathematical audit was performed; the earlier MathDev audit remains
inconclusive.

`final-validation.json` records the 15 passing post-repair regression checks,
non-finite rejection serialization check, Python syntax checks, clean
whitespace diff, preserved-source hash and final PDF hash. The small report
checker initially treated the saved row count as an array; correcting that
report-only assertion produced the passing verification without changing any
experimental evidence.

## Decision and inference status

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Stop this grid | Derivative fixtures pass; attribution is partial | Transport validity failure is a continuation veto | Exact failed GPU tensors were not archived | Preserve evidence and calibrate terminal balancing separately | Full grid completion or canonical validity |
| Do not promote tested LEDH controls | Conditional model-score error fails heuristic screen | Heuristic underperformance vetoes promotion | Few datasets, shared streams and incomplete coverage | Inspect first-step proposal/weighting before new guide comparison | Statistical superiority of any stochastic method |
| Keep the augmentation research direction open | No new guide comparison was attempted | No direction-level impossibility is established | Baseline approximation error remains unresolved | Repair baseline, then fresh frozen-guide comparison | Trained KDM, nonlinear transfer, HMC or default readiness |

| Inference item | Status |
| --- | --- |
| Hard veto screen | Shared transport guard rejects the reconstructed case; heuristic promotion screen fails in all conditions |
| Statistically supported ranking | None; no uncertainty analysis supports a stochastic ranking |
| Descriptive-only differences | Table MSE, flow/N contrasts, ESS, moment ratios and cap activity |
| Default-readiness | Not established; controls are untuned and the grid is incomplete |
| Next evidence needed | Guard-preserving balancing calibration, first-step Gaussian comparison, then untouched paired comparisons with uncertainty and scope tuning |

Post-run red team: the strongest alternative explanation for the persistent
score discrepancy is a proposal/weighting or approximation mismatch rather
than insufficient flow resolution. The weakest evidence is attribution from
small fixed particle clouds, particularly the incomplete N=128 group and the
unarchived GPU rejection. Exact first-step proposal/weight agreement would
weaken that explanation and shift attention to later reset shape and finite-N
effects. The observed results reject the tested candidate's promotion, not
the general idea of using future information.

## Budget and next action

The two permitted GPU launches used 769.056/1,200 wall seconds. Although
430.944 seconds remain arithmetically, no third launch is authorized by this
plan. Timed CPU tests, audit, localizations and document builds used
228.703/600 seconds. Short shell inspections and source preservation are not
included, so the remaining CPU allocation is strictly less than 371.297 seconds.

The next research stage is a separately bounded, scope-specific calibration
of terminal OT balancing with epsilon 1 / five Sinkhorn iterations unchanged.
It must preserve the current validity threshold, inspect value/score and
moment non-harm on healthy fixtures, and use separate calibration and heldout
cases. It must not optimize against the heuristic screen or model-score MSE.
Then localize first-step proposal/weighting error against the Gaussian oracle.
Only after those checks should a fresh controlled-SMC/KDM augmentation
comparison resume. The updated master and checkpoint reflect that order.
