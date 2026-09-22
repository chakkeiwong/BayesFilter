# Fresh iAPF calibration result and master refresh, 2026-09-17

The frozen-control iAPF baseline has larger observed score error than both EKF
and UKF on every one of the six claim datasets. The wider fitting candidates
fail objective-underflow checks on 14 source rows, so neither regime can issue
a selected-candidate tuning artifact. Nothing is promoted. The campaign itself
completed successfully, including the previously missing curved heuristic table
and frozen-control baseline on untouched claim data.

This is Phase 0E of the score master: a scalar nonlinear, two-observation pilot.
It does not complete the master, establish a canonical LEDH result, reject the
iAPF research direction, or establish a statistical ranking.

## Execution and verification

The owner authorized refresh and execution on 17 September. The
[fresh plan](../../../../younis-score-iapf-fit-calibration-fresh-2026-09-17.md)
received a skeptical audit before execution. It specifies three candidates,
fresh disjoint data partitions, conditional heuristics, explicit numerical
vetoes, and continuation of independent studies when selection is blocked.

The underflow guard is committed as `282e37b4`; the driver, focused tests and
pre-run plan are committed as `418e538825462e466a313c088fd8f838d2fbea74` in
`.localresources/worktrees/younis-score-iapf-underflow-guard-20260916`.
That checkout was clean at launch and remains the executed source authority.
The driver and tests were also copied to the main checkout without disturbing
other work. Seven new driver tests pass in each checkout; the commit hooks
pass. The unchanged fitter's earlier 19-test suites are linked in the master.

One GPU launch completed in 68.00 seconds of measured process wall time,
using 76.15 process CPU seconds. It attempted 68 rows: 54 completed numerical
results and 14 failed fits. Twelve selected-candidate claim rows were blocked
by incomplete source studies. Accounting charged 171 of 280 row-or-fit attempts;
failed rows conservatively cost their maximum four charges. Two of the three
allowed campaign launches and 109 charges remain unused. Repeating the same
rejected candidates has no scientific purpose.

The GPU probe and launch used the **RTX 5080**, PCI `0000:01:00.0`, with FP32
filtering, TF32, XLA and FP64 offline fitting. Memory growth was verified.
`CUDA_VISIBLE_DEVICES=1` did not select nvidia-smi index 1; the initial intended
RTX 4080 SUPER selection was corrected from TensorFlow's actual device logs.
Both probe and campaign used the same device. Future launches should select
by GPU UUID. No runtime ranking is made in this shared environment.

Device wall accounting is 71.30 seconds including the probe. CPU accounting
includes the explicit 580-second prelaunch charge, which contains a conservative
allowance for earlier unmeasured commit-hook work; it must not be described as
measured CPU consumption. Exact logs and aggregate accounting are saved beside
this note. The limits of 3000 GPU seconds and 7200 CPU seconds were respected.

The read-only [saved-result audit](saved-run-audit.json) checked eight study
fingerprints, all 54 result digests, matched observations and references,
reference refinement, complete heuristic tables, failure records and independent
claim streams with identical fitted coefficients. The largest mesh and domain
relative discrepancies were approximately `4.20e-16` and `1.48e-15`; the largest
reported reference tail mass was `4.45e-16`, within the predeclared tolerances.
These checks establish consistency of this pilot, not general filter validity.

## Conditional claim evidence

Entries are squared Euclidean error of the six-parameter observed-likelihood
score against the refined grid reference. The iAPF entry averages two final
streams with the same frozen fitted coefficients. Bootstrap and local-linear
entries each use one stream at N=16. EKF and UKF are deterministic. Adaptive
iAPF uses the reported N=16 or 32. These are descriptive pilot comparisons,
not equal-cost or statistically powered rankings.

| Regime / dataset | EKF | UKF | Bootstrap | Local-linear | Frozen iAPF | iAPF N | Boundary veto |
|---|---:|---:|---:|---:|---:|---:|---|
| Weak / 1120 | 0.004128 | 0.003112 | 0.180842 | 0.042779 | 6.447256 | 32 | Yes |
| Weak / 1122 | 0.002163 | 0.003746 | 0.363833 | 0.317321 | 0.723864 | 16 | Yes |
| Weak / 1124 | 0.002792 | 0.001283 | 4.040375 | 0.204123 | 0.267490 | 32 | Yes |
| Curved / 1121 | 0.051323 | 0.063988 | 0.995484 | 0.540883 | 0.404214 | 16 | No |
| Curved / 1123 | 5.896282 | 2.675860 | 54.443273 | 3.596901 | 28.045148 | 32 | Yes |
| Curved / 1125 | 0.031905 | 0.046100 | 1.152232 | 0.039698 | 0.723840 | 16 | Yes |

The predeclared heuristic screen vetoes promotion in every dataset because
EKF and UKF each have smaller observed error. Ten of twelve baseline claim
rows also have fitting-boundary activity. The accepted baseline's largest
normalized shape residual is approximately 0.9983. A finite, non-underflowed
fit therefore does not establish an accurate twist or a useful score.

The two wider candidates fail on all four weak source datasets and three of
four curved source datasets each: 14 rows containing 18 underflowed fitting
steps. Every rejected row reports solver convergence and a valid coefficient
cast, but the new underflow guard correctly sets fit validity false. Selection
retains those failures instead of dropping them from the declared candidate
family. Both baseline-only source studies remain complete and issue their own
repository-owned selection artifacts; all twelve baseline claim rows finish.

## Why the fitting problem remains

The checked local code in `bayesfilter/score_study/iapf_fit_tf.py`,
`_density_profile`, computes a profiled density least-squares objective. Write
`a = max_i log q_eta(x_i)`, `d_i = exp(log q_eta(x_i) - a)`, normalized target
values `t_i`, and profiled relative scale `c`. Its objective is

`L(eta) = exp(2a) * mean_i (d_i - c t_i)^2`.

The parameter-dependent amplitude `exp(2a)` is part of the implemented
objective. A Gaussian placed far from the fitting cloud can make that amplitude
very small while the relative shape residual remains large. On the first
rejected weak row, the saved values include log amplitude about -2046.76,
normalized shape residual about 0.9880, and floating-point objective exactly
zero. Widening the allowed means and scales exposes this failure more often.

The guard fixes admission, not this objective's finite-cloud geometry. Simply
removing the amplitude factor changes the objective. A log-objective solver
would avoid some underflow but can still prefer an extremely small positive
density-scale loss. Neither change may be silently called an equivalent
scientific repair. The next step must reconcile the fitting objective,
constraints and stopping criterion with the source specification, and assess
whether a declared adaptation is needed.

## Decision

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Preserve completed campaign; no candidate promotion | Selected-versus-baseline claim comparison unavailable | 14 invalid fits block selection; baseline fails conditional heuristic screen | Whether a suitable fitting formulation can produce a useful twist | Source-grounded density-fit repair, then fresh calibration and claims | Rejection of iAPF or completion of LEDH |
| Keep underflow guard | Rejects all 14 invalid fits before downstream score evaluation | No admitted underflow observed | Near-zero positive objectives can still satisfy absolute stopping tolerances | Evaluate scale and shape in the repair contract | Guard alone fixes fitting quality |

| Inference status | Finding |
|---|---|
| Hard veto screen | 14 underflow failures; selected claims blocked; boundary activity on 10 baseline claim rows; conditional heuristic promotion veto on all six datasets |
| Statistically supported ranking | None; two final streams and three datasets per regime are a pilot |
| Descriptive-only differences | All reported squared errors, adaptive N, and runtime |
| Default-readiness | Not established; no new default or HMC-facing admission |
| Next evidence needed | Explicit source-grounded fit repair; focused numerical checks; fresh scope-specific tuning and untouched claims; larger paired replication before ranking |

The harness, source snapshots, data matching and reference checks passed. The
current fitting family failed, which is a repair trigger. It is not a
continuation veto for the research direction. The next Phase 0E action is the
density-fit repair described in the master; comprehensive control calibration,
iAPF-moment integration into LEDH, wider models and powered replication remain
open. The failed claim data must not be reused for selecting the repair.

Post-run red-team: the strongest alternative explanation for poor baseline
scores is limited particles and stochastic variation, not a defect in twisting
as a general idea. The fitting failures themselves are direct numerical
evidence independent of that ranking uncertainty. A source-grounded repaired
fit with valid, fresh downstream comparisons could overturn the negative
candidate assessment. The weakest evidence here is the tiny claim replication
and unequal realized work across methods.

Complete execution evidence is in [run-manifest.json](launch01/run-manifest.json),
[consumer-evidence.json](launch01/consumer-evidence.json),
[invalid-fits.json](launch01/invalid-fits.json),
[conditional-heuristics.json](launch01/conditional-heuristics.json),
[attempt-accounting.json](launch01/attempt-accounting.json), and
[launch01.log](launch01.log). The executable audit is
[audit_saved_run.py](audit_saved_run.py).
