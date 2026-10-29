# Expanded marginal-weight campaign: review and interim interpretation

Scientific goal: testing surrogate HMC. Current work verifies its likelihood
and analytical-score component; it supplies no new HMC samples.

Plan: [expanded verification](surrogate-hmc-marginal-expanded-execution-2026-10-06.md).
Live results: [source04](artifacts/surrogate-hmc-marginal-expanded-20261006-04/results.md).
Archived results: [source03](artifacts/surrogate-hmc-marginal-expanded-20261006-03/results.md).
This review records the archived source03 N1024 evidence and subsequent
integration repair. A fresh source04 continuation preserves those results under
their actual source identity; it uses new data for failed and unfinished scopes.
The full requested grid is not complete.

Source04 continuation checks passed: all six engineering fixtures, repeated
M13 T20 N4096 precision, and a valid LG20 T240 N1024 pilot. The pilot estimates
10,827 seconds for the minimum paired candidate calls, excluding references,
compilation and comparator costs. This exceeds the 7,200-second scope allowance;
it establishes feasibility of valid calls, not completion or accuracy. The
master preserves partial evidence and proceeds through the declared grid under
the unchanged total ceiling. See the [active checkpoint](artifacts/surrogate-hmc-marginal-expanded-20261006-04/checkpoint.md).

The first repaired-source LG10 T10 comparison has reached its eight-dataset
look: all 1,536 filter calls are valid; five of six likelihood/score
non-inferiority cells pass. Both errors are lower under the declared intervals
at parameter point (1.2,0.3). Value non-inferiority at (0.8,-0.5) remains
inconclusive, so the master is extending to sixteen datasets. This is an interim
result: [quantitative interpretation](artifacts/surrogate-hmc-marginal-expanded-20261006-04/lg10-t10-interim-eight.md).
The live report now includes all requested grid cells, actual error intervals,
simple-filter errors, and equal-runtime coverage. The executed frozen Python
sources are archived and verified against the run manifest.

## Skeptical review

The comparison changes only the weight law within each paired scope. The
baseline is ancestor weighting through the same consumer, with identical
data, innovations, reset designs, numerical controls, dtype and hardware.
Kalman values and scores are exact LGSSM references; KSC integration is
refined; range-bearing quadrature checks the one-step reference, while
replicated larger PF/Fisher references carry explicit uncertainty at longer
horizons. Derivative parity checks the actual consumer and does not establish
model-score accuracy.

The two control sets are declared hypotheses, calibrated separately for every
scope and frozen before claim data. They do not establish the best tuned
version of either method. The 0.05 margins and 0.1 score scale remain component
materiality hypotheses; they cannot certify a posterior. The data-generating
truth is fixed within each model; the reported regimes are likelihood
evaluation points. Shared data and designs remove an avoidable confound, but
do not remove dataset uncertainty.

The dataset, not its 32 particle designs, is the independent replication unit.
The three planned looks have simultaneous Student intervals, bootstrap and
leave-one-out sensitivity, and reference allowances. Sixteen clusters still
give approximate coverage. Unresolved intervals remain inconclusive.

The simple comparators are constructed from model structure: prior-trajectory
importance sampling, bootstrap filtering and a Gaussian filter (Kalman,
KSC moment approximation, or range-bearing EKF). They are reported
conditionally, not pooled across models or regimes. A smaller observed
comparator error is a conservative promotion veto, not itself a statistical
ranking. Incomplete runtime coverage cannot pass the comparison.

The plan passes as a bounded component investigation. It does not pass as
a complete all-model HMC validation or a literal nonlinear dimension grid.
KSC and range-bearing retain native dimensions 1 and 4 pending equations for
the requested extensions. The 24-hour cap may leave work unfinished.
Invalid references, corrupt provenance and exhausted resources stop execution;
candidate rejection triggers the next planned scope or particle rung.

## Repairs and checks

Two harness defects were repaired: reference construction inside traced graphs
and FP64 comparator timing against FP32/TF32 LEDH. Dynamic reference consumers
then passed parity on two datasets with one trace per fixed signature.
The initialization repair uses FP64 only for small checked stationary
matrix products, retaining input-dtype solves and FP32/TF32 filtering.
No residual threshold, ridge, damping, model law or margin was relaxed.
All 24 isolated stationary checks passed, 17 CPU regressions passed, and the
valid LG1/LG3/KSC full-consumer changes were at most 3.58e-6 scaled.

The new engineering run passes actual multiple-row versus separate-call parity
for both policies in LG3/KSC/M13, with zero observed discrepancy. The largest
finite-difference discrepancy is 1.77863e-5. The fresh T=1 quadrature/PF check
passes. Repeated M13 N4096 precision checks are finite and exactly repeatable,
with maximum FP32/FP64 deltas 0.00529032 in value and 0.00339039 in scaled
score. They are below the declared warning/materiality scale.

After initialization was repaired, d10/d20 still fail in marginal-mixture
weight factorization under both tested control sets. These are hard numerical
validity failures of the tested implementation/settings. No paired accuracy
result exists for those scopes. They do not disprove the algebraic identity,
and they do not justify relaxing the factorization guard.

## Interim results

All valid completed scopes below use N1024 and 32 paired designs per dataset.
LGSSM d3/T20 resolved at eight datasets; the others used sixteen.

| Scope | Primary value/score verdict | Statistically supported lower error versus ancestor |
|---|---|---|
| LGSSM d1 T10 | 3/3 non-inferior | Score at 2 points |
| LGSSM d3 T10 | 3/3 non-inferior | Value and score at 1 point |
| LGSSM d1 T20 | 3/3 non-inferior | Value at 1 point; score at 2 |
| LGSSM d3 T20 | 3/3 non-inferior | Score at 1 point |
| KSC d1 T20 | 2/3 value; 3/3 score non-inferior; point-1 value inconclusive | None |
| KSC d1 T10 | 2/3 non-inferior; point 1 inconclusive | None |
| M13 d4 T10 | 2/2 non-inferior | Value and score at both points |
| LGSSM d10/d20 T10 | No valid shared controls | Not assessed |

These statistical rankings are conditional on the stated approximate
interval procedure. A heuristic veto blocks promotion but does not erase a
supported comparison with the ancestor method.

| M13 T10 regime | Ancestor value error | Marginal value error | Paired error-difference interval | Bootstrap error | EKF error |
|---|---:|---:|---|---:|---:|
| Nominal | 2.6931 | 0.4757 | [-4.3441, -0.0908] | 0.2516 | 0.1896 |
| Low process noise | 23.6137 | 3.0285 | [-35.1356, -6.0347] | 0.5689 | 0.2658 |

Scaled score errors decrease from 0.6170 to 0.1190 and from 5.9560 to
0.9635. Corresponding adjusted intervals are [-0.9348,-0.0612] and
[-8.0395,-1.9455]. The candidate repairs much of the ancestor error, yet the
observed bootstrap/EKF likelihood errors are smaller. Those simple-method
means are descriptive; under the declared rule they veto promotion.

At the inconclusive KSC point, mean value error is 0.1927 versus 0.1158,
but the adjusted interval [-0.3001,0.1463] crosses the 0.05 deterioration
margin. Score error is 0.1077 versus 0.0503, with interval
[-0.2210,0.1062]. Smaller means do not settle non-inferiority.

All rows have complete same-N prior/bootstrap/Gaussian comparisons.
Equal-runtime matching is incomplete: M13 has only 16/512 and 17/512 matched
rows, and KSC 16–27/512. Those subsets cannot establish overall computational
value. Bootstrap timing measures likelihood alone; LEDH also computes score.

## Decision table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Continue optional candidate | Most completed cells non-inferior; M13 lower errors | Heuristic losses prevent promotion | Longer T, larger N, KSC point 1 | Continue existing master | Universal improvement |
| Keep current FP32 campaign | Precision/repeat checks pass | No precision validity veto | High-d factorization unresolved | Diagnose separately; preserve guards | Exact-model accuracy |
| Promote over simple filters | Not met | Observed Gaussian/bootstrap/EKF losses | Descriptive comparator uncertainty | Finish conditional cost/accuracy ladder | Cost superiority |
| Interpret d10/d20 | No valid comparison | WEIGHTS factorization fails | Implementation precision or conditioning | Bounded localization, then fresh calibration if repaired | Rejection of marginal identity |
| Claim surrogate-HMC success | Not tested | No posterior evidence | Endpoint-target and sampling behavior | Later HMC phase with supported tuner | Acceptance, R-hat, ESS, posterior correctness |

## Inference-status table

| Evidence class | Interpretation |
|---|---|
| Hard veto screen | High-d candidate factorization failure; low-d engineering passes |
| Statistically supported ranking | M13 versus ancestor in both metrics/points; selected LG cells; subject to approximate cluster intervals |
| Descriptive-only differences | Simple comparator means, timings, and smaller KSC point-1 means |
| Default readiness | Not established; optional path, no public tuning admission or HMC confirmation |
| Next evidence | Remaining horizons, particle ladder, full runtime coverage, high-d repair, then surrogate-HMC posterior testing |

The strongest alternative explanation is control-dependent proposal quality:
the shared settings isolate the law but are not each method's fully tuned
optimum. Reference bias and a few influential datasets remain possible.
A replicated valid deterioration result would overturn preservation in the
affected scope; complete cost/error comparisons could strengthen or reverse
the current computational verdict. The weakest evidence remains long-horizon
coverage, high-d validity and runtime matching.

## Further runtime review

The recorded seconds include full tensor-to-report serialization, whose cost
differs substantially between the methods' diagnostic returns. Consequently,
these measurements cannot yet establish relative kernel efficiency. Only
69/1536 KSC and 33/1024 M13 rows match the runtime budget at T10; the median
cost ratios are 0.65 and 0.63. The cap explains only one M13 mismatch.
A separate synchronized-call/serialization diagnostic is queued between master
workers, alongside the high-dimensional covariance diagnostic. The supervisor
resumes the master automatically and charges both within its original budget.

The accuracy observations and comparisons above remain available. No speed
ranking, complete equal-runtime comparison or surrogate-HMC claim is established.


## Diagnostic outcomes

The planned bounded diagnostics have now run. The covariance probe used the
actual first-observation inputs of the unchanged LG10/LG20 consumer. With
TF32 enabled, the two failing LG10 parameter points had relative asymmetry
\(1.5321\times10^{-5}\) and \(1.2550\times10^{-4}\), exceeding the
\(9.5367\times10^{-6}\) guard tolerance; the failing LG20 point had
\(1.2581\times10^{-4}\), exceeding \(1.9073\times10^{-5}\). Finiteness and
positive symmetric-part eigenvalues held. TF32-off made all points valid,
and an FP64 \(BQB^\top\) product cast to FP32 was valid at every point.
The untouched consumer and instrumented consumer agreed on validity. This
is a precision-sensitive factorization diagnosis, not evidence against the
marginal weight law. No Class C numerical change has been promoted.

The synchronized timing probe measured warmed XLA output separately from
report serialization. KSC T=10 N=1024 medians were 0.05497 s versus 0.08846
s for marginal-mixture synchronization and report-ready timing, compared
with 0.00292 s and 0.00294 s for bootstrap. M13 medians were 0.16341 s and
0.19733 s, compared with 0.00240 s and 0.00242 s. The approximately 0.033 s
LEDH serialization component is absent from the bootstrap payload. These
measurements explain why the existing equal-runtime matching is sparse; they
do not establish a computational ranking. A later comparison must calibrate
repeated-call kernel and end-to-end workloads separately.

The master resumed after both diagnostics and remains incomplete. The current
terminal interpretation is: low-dimensional accuracy evidence is usable;
the high-dimensional candidate is invalid at the current TF32 guard for
specific points; computational value is unresolved; and surrogate-HMC
posterior behavior has not been tested.

## Continuation through 12:50 UTC

LG1T50 (16datasets) and LG3T50 (8datasets) pass all three value/score
non-inferiority checks. Lower score error is supported at LG1 points1/2 and
LG3 point1; no lower value-error ranking is supported. KSC T50 uses16datasets:
points0/2 pass both checks; point1 remains inconclusive for both metrics.

M13T20 nominal: lower value and score errors are supported, with paired
reference-adjusted intervals[-5.9883,-.6416] and[-1.5036,-.1846].
The simple-bootstrap/Gaussian observed value errors remain lower, a promotion
veto without a statistical ranking against those methods. At low noise one
ancestor run fails, versus512/512 valid marginal runs. The reporter labels
this INVALID_BASELINE and preserves the producer's original mislabeled verdict
and failure row. No paired ranking is reported for that regime.

The unconditional FP64 covariance-product candidate recovers all24 tested
higher-dimensional full-filter rows but fails the .005 no-harm screen on
three healthy low-noise M13 rows (maximum value/score changes .02729/.11323).
It is not adopted. Score FD and FP64 parity pass, but cannot waive that failure.
A selective precision candidate, triggered only by the existing symmetry
check, is being evaluated under the same remaining budget.

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Retain valid N1024 likelihood/score results | LG and M13 nominal comparisons resolved as above | Simple-filter observed underperformance blocks promotion | KSC point1 intervals; longer horizons/N | Continue existing grid | HMC or universal improvement |
| Reject unconditional precision repair for broad adoption | High-d rescue passes; no-harm fails | M13 numerical changes exceed screen | Whether changes improve accuracy is not tested | Evaluate narrower repair | Statistical deterioration |
| Repair runtime measurement | Reporting overhead confirmed | Sparse matching prevents complete comparison | Stable matched cost across grid | Bounded repeated synchronized timing | Computational superiority |

| Inference status | Current finding |
|---|---|
| Hard veto screen | Original high-dimensional covariance failures and one M13 baseline failure are preserved |
| Statistically supported ranking | Selected LG score and M13 nominal comparisons, conditional on stated approximate intervals |
| Descriptive-only differences | Heuristic errors, timing, and precision-repair output changes |
| Default-readiness | Not established |
| Next evidence needed | Valid higher-dimensional candidates, unresolved dataset intervals, complete larger-N and runtime comparisons |

Strongest alternative explanation: numerical changes and approximate particle
references can drive apparent gains. The paired exact-LG checks, independent
reference refinements and protected failures constrain that explanation but do
not remove the longer-horizon/nonlinear uncertainty. No result here replaces
the subsequent surrogate-HMC test.

## Integrated repair decision and saved-run reproducibility

The actual-consumer paired check
artifacts/surrogate-hmc-marginal-integrated-nonharm-20261006-01/result.json
passed all68 rows. Every previously valid value and score was exactly unchanged;
all12 previously invalid high-dimensional rows became valid with positive
persisted precision-repair counts (eight d10, four d20). This supports using
the selective repair in the optional component experiment. It does not prove
HMC smoothness at a precision-switch boundary.

Earlier comparison against saved prior-process numbers failed on two healthy
M13 rows. Removing new diagnostic outputs did not cure the difference.
The subsequent repeatability check showed that the unchanged archived source
also differed from the saved original by up to .0153022 nats and .198898 scaled
score, beyond the same .005 no-harm screen on three rows. In that process,
two separately traced archived originals and the integrated candidate agreed
exactly on all eight rows; repeated calls also agreed exactly.
Thus a repair-specific deterioration is not established by the saved-run
comparison. Cross-process FP32/TF32 reproducibility at that scale is unresolved.
The mechanism may involve graph compilation or numerical execution, but this
diagnostic has not identified the cause. All failed reproduction artifacts
remain preserved. No criterion has been loosened.

The final six-scope check compares actual archived/current public consumers,
rather than only a prototype function. Fresh source04 engineering parity and
N4096 precision are required before further scope dispatch. Controls are
calibrated on new dataset2; untouched claim dataset indices start at100.
All previous worker and diagnostic seconds count toward the original86400 cap.
The latest prelaunch charge is32320.94seconds, leaving54079.06 (15.02hours).

Revised synchronized timing with bounded stabilization matched8/8 KSC and7/8
M13 test rows; one M13 ratio .767 failed [.8,1.2]. Preserve it and classify cost
matches per row. This supports the measurement repair, not an efficiency claim.
Warm-up, compilation and report serialization are excluded from steady-state
cost; score work remains included for LEDH and absent for bootstrap.

| Decision | Primary criterion | Veto diagnostic | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Continue optional selective repair |68/68 paired no-harm/validity checks pass | None in same-process check | Saved-run reproducibility and switch boundary | Fresh-source master under original budget | HMC/default admission |
| Global efficiency claim | Incomplete matching | One timing-probe mismatch; incomplete grid | Variance of actual costs | Preserve calibrated and realized costs | Speed superiority |
| Full requested dimension grid | LG dimensions defined; nonlinear native only | Missing nonlinear equations | Meaning of nonlinear extension | Continue defined scopes; await equations | Invented nonlinear models |

| Inference status | Finding |
|---|---|
| Hard veto screen | Earlier saved-run reproduction failed; no paired integration validity failure |
| Statistically supported ranking | Earlier source03 scope rankings remain conditional on their own uncertainty analysis |
| Descriptive-only differences | All precision, timing and reproducibility probe differences |
| Default-readiness | Not established |
| Next evidence needed | Fresh-source full-grid accuracy, precision and runtime results; nonlinear model definitions |

Post-run red team: same-process agreement could conceal compilation-dependent
variation shared by both methods. The original saved-run failure is therefore
a live limitation, not a solved discrepancy. Independent references and paired
fresh-dataset results remain necessary; no posterior conclusion follows here.
