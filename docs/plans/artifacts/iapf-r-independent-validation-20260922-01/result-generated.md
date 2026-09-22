# Independent R iAPF validation: numerical completion and accuracy remain distinct

The campaign completed 80/80 paired cells and
160/160 scheduled full learners. No failure was observed among the recorded learners.
The repaired method has 14 observed conditional heuristic
underperformance records, which veto promotion. The dimension-specific paired
accuracy intervals below determine which QR/repaired differences are supported;
unresolved intervals are not evidence of equivalence. Full paper replication
remains open, because this weighted-log/ridge learner changes Equation 15 and
the original author implementation/settings have not been recovered.

This is the authorized independent CPU R reference. Both fitters, the shared
learning controller, numerical safeguards and floor convention were frozen at
the preceding repair manifest. No TensorFlow/GPU, LEDH, KDM, score, HMC or
production default was changed.

## What was tested

The first-study linear Gaussian model has T=100 and dimensions 5, 10, 20, 40
and 80. Two new datasets per dimension and eight paired learning repetitions
per dataset give 16 paired cases per dimension. The code uses the actual shared
controller and its fresh final estimate. Thirteen focused checks established
that timing instrumentation preserves the numerical outputs and RNG state;
exact full-Gaussian controls agree with Kalman under the unchanged 1e-7 gate.
All commands, seeds, frozen source identities and output hashes are preserved.

Intervals resample whole paired learning runs within each fixed dataset,
retaining the coupling between methods. The 99% intervals address the five
primary dimensional comparisons through the predeclared Bonferroni adjustment.
They are approximate bootstrap intervals conditional on these ten datasets;
two datasets per dimension do not establish general data-population performance.

## Numerical completion and paired terminal accuracy

| Dimension | Method | Complete / planned | Observed failures | One-sided 95% upper failure probability |
|---|---|---|---|---|
| 5 | qr | 16/16 | 0 | 0.1707 |
| 5 | ridge_bounded1 | 16/16 | 0 | 0.1707 |
| 10 | qr | 16/16 | 0 | 0.1707 |
| 10 | ridge_bounded1 | 16/16 | 0 | 0.1707 |
| 20 | qr | 16/16 | 0 | 0.1707 |
| 20 | ridge_bounded1 | 16/16 | 0 | 0.1707 |
| 40 | qr | 16/16 | 0 | 0.1707 |
| 40 | ridge_bounded1 | 16/16 | 0 | 0.1707 |
| 80 | qr | 16/16 | 0 | 0.1707 |
| 80 | ridge_bounded1 | 16/16 | 0 | 0.1707 |

These bounds show how much failure risk remains compatible with a small
validation sample. Completion is an engineering result; it is not a proof of
small error or rare-event reliability.

| Dimension | Pairs | QR mean absolute log error | Repaired mean absolute log error | Repaired − QR | 99% paired interval | Accuracy inference |
|---|---|---|---|---|---|---|
| 5 | 16 | 0.02195 | 0.03398 | 0.01203 | [-0.004683, 0.03091] | Unresolved |
| 10 | 16 | 0.03813 | 0.05922 | 0.02109 | [-0.01941, 0.06044] | Unresolved |
| 20 | 16 | 0.08487 | 0.09116 | 0.006292 | [-0.05378, 0.07188] | Unresolved |
| 40 | 16 | 0.09017 | 0.1117 | 0.02157 | [-0.04625, 0.09697] | Unresolved |
| 80 | 16 | 0.1524 | 0.2802 | 0.1279 | [-0.0293, 0.3108] | Unresolved |

The difference is mean absolute error in terminal log likelihood, repaired
minus QR. Negative values favor the repaired method. A confidence interval
overlapping zero leaves ranking unresolved. Prefix diagnostics were preserved
as explanatory quantities and were not substituted for the paper's terminal
likelihood target. Any candidate failures, caps or omissions remain visible
in the complete records and are not silently dropped to obtain a ranking.

| Dimension | Recorded repaired fits | Positive ridge | Active curvature bound | Active-set fallback | Maximum KKT residual | Minimum weight ESS |
|---|---|---|---|---|---|---|
| 5 | 9600 | 0 | 0 | 0 | 3.393e-13 | 2.486 |
| 10 | 9600 | 0 | 0 | 0 | 9.91e-13 | 1.009 |
| 20 | 9600 | 32 | 0 | 0 | 7.935e-13 | 1 |
| 40 | 10200 | 1379 | 3 | 2 | 5.792e-07 | 1 |
| 80 | 12100 | 1920 | 1 | 1 | 6.537e-13 | 1 |

These retained fit diagnostics explain numerical behavior. The active-set
fallback solves the same frozen convex objective when the initial solver
fails its KKT check. A small KKT residual verifies the accepted optimizer's
first-order conditions; it does not establish that the fitted guide captures
the true future likelihood. Ridge and curvature-bound counts likewise do not
substitute for terminal accuracy.

## Heuristic comparisons and calibration

Every completed cell evaluates constant, observation-only, moment-diagonal,
precision-diagonal and full Gaussian guides. For comparison, both learned
guides are evaluated again at N=1000 using fresh common seeds. The two analytic
diagonal guides have privileged access to future Gaussian information; they
diagnose learning loss and do not establish a general-purpose alternative.
The exact full guide is an oracle check, not a stochastic competitor.

The full conditional table contains 20 observed underperformance
records across both learners. The following are up to eight largest observed
differences for the repaired learner; they are descriptive findings, not
statistical rankings of the controls.

| Dimension | Dataset | Control | Repaired mean absolute log error | Control mean absolute log error |
|---|---|---|---|---|
| 80 | 2 | qr | 0.2562 | 0.1889 |
| 80 | 2 | moment_diagonal | 0.2562 | 0.2017 |
| 40 | 2 | precision_diagonal | 0.1693 | 0.1174 |
| 80 | 2 | precision_diagonal | 0.2562 | 0.2199 |
| 10 | 1 | moment_diagonal | 0.05943 | 0.03453 |
| 5 | 1 | qr | 0.03811 | 0.01911 |
| 20 | 2 | precision_diagonal | 0.1218 | 0.1052 |
| 40 | 1 | precision_diagonal | 0.1267 | 0.1158 |

The accuracy/cost table also records mean likelihood ratio and the fraction
below .01. There are 10 nominal calibration interval flags
across all methods/rungs. These screens do not establish estimator bias:
small samples can miss rare positive likelihood tails, and the calibration
intervals are not multiplicity-adjusted tests. In particular, a collapsed
bootstrap filter can have deceptively small observed variance. The report
never uses that variance alone as an efficiency claim.

## Cost at a stated precision threshold

After each learner stops, its guide is frozen and independent filters run at
N=250, 1000 and 4000. Total time includes the entire guide-learning cost, less
the original fresh-final filter, plus the measured new final-filter time.
Common data/model/Kalman setup is excluded. Runs are serial with one BLAS/OMP
thread and counterbalanced method order; timings remain machine-specific.

The target is relative likelihood RMSE <=.5. The listed rungs satisfy the
predeclared bootstrap upper-limit screen, with Bonferroni adjustment across
five dimensions, two learners and three rungs. The table reports the least
measured mean cost among qualifying tested rungs, without changing any default.
An absent rung means that this grid has not established the target. These are
costs at a shared precision ceiling, not evidence of identical MSE or an
optimal particle count. Bootstrap bounds do not protect against unseen tails.

| Dimension | Method | Qualifying N | Relative RMSE | Simultaneous upper RMSE | Mean total seconds |
|---|---|---|---|---|---|
| 5 | qr | 250 | 0.06402 | 0.08591 | 1.968 |
| 5 | ridge_bounded1 | 250 | 0.05489 | 0.07513 | 1.632 |
| 10 | qr | 250 | 0.1397 | 0.2007 | 2.982 |
| 10 | ridge_bounded1 | 250 | 0.1731 | 0.236 | 2.566 |
| 20 | qr | 250 | 0.1933 | 0.2674 | 5.231 |
| 20 | ridge_bounded1 | 250 | 0.2271 | 0.3188 | 4.725 |
| 40 | qr | 1000 | 0.1724 | 0.2205 | 10.49 |
| 40 | ridge_bounded1 | 1000 | 0.1737 | 0.2514 | 10.41 |
| 80 | qr | 4000 | 0.1004 | 0.134 | 27.98 |
| 80 | ridge_bounded1 | 1000 | 0.2865 | 0.4565 | 34.1 |

## Decision and limitations

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Record frozen-reference validation | 160/160 scheduled learners complete | See failure and oracle tables | Rare failures and new datasets | Preserve both frozen references and all failed/censored cells | General numerical correctness |
| Assess QR versus repaired accuracy by dimension | Predeclared paired 99% intervals above | Heuristic and validity vetoes remain binding | Small conditional sample and unseen tails | Use supported intervals only for the stated conditional comparison | Broad superiority or equivalence |
| Record measured cost at the precision ceiling | Independent fixed-guide grid and simultaneous upper RMSE screen | No qualifying rung means unresolved cost | Coarse grid, bootstrap coverage and machine load | Retain full cost/error curves; confirm any consequential cost claim independently | Optimal or exact matched-MSE efficiency |
| Keep original-paper replication open | Author identity and original objective differ | Replication gap remains | Unrecovered numerical choices | Keep the extension separate from Equation 15 | Full reproduction of the published implementation |

| Inference status | Finding |
|---|---|
| Hard veto screen | 0 recorded learner failures; oracle/check status retained in checks.csv |
| Statistically supported ranking | Only dimensions whose paired interval excludes zero, conditional on the tested datasets |
| Descriptive-only differences | Heuristic comparisons, individual errors, maxima, fit diagnostics and raw timings |
| Default readiness | No promotion; observed heuristic underperformance remains a veto |
| Next evidence needed | Resolve any new numerical failures from their saved cases; use independent data and adequate replication for wider accuracy/cost claims |

Post-run skeptical review: the strongest alternative explanation for stable
completion is that ridge regularization preserves an unweighted guide similar
to QR, rather than improving learning of future information. Exact and analytic
diagonal controls expose the remaining gap. The weakest evidence is the small
number of independent learned guides and fixed datasets, especially for rare
likelihood tails. New failures or a verified author implementation would
change the diagnosis. Numerical completion and conditional accuracy evidence
must not be promoted into full paper or production claims.

Numerical execution used 2209.885535 aggregate worker seconds in
84 launches. The original deadline and 4,100-second phase cap were
preserved. Remaining prior allocation: 2303.716034
seconds; cumulative campaign use: 112286.080369/172800.
Per-attempt manifests distinguish successful process completion from candidate
success. The statistical tables are in `attempt084-report/results`.
The initial 120-second closure reserve left two cells unfinished. A bounded
scheduling repair reduced that reserve to 30 seconds and used fresh attempts,
without extending the original deadline or numerical budget. The original
index and report remain intact; `cells-completion.csv` identifies the selected
attempts. All timed-out work is included in the total cost of the campaign.
