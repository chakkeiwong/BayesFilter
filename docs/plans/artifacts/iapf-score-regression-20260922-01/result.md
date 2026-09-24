# Score regression is numerically correct but still loses to a simple guide

The candidate fails the declared heuristic screen at dimensions5 and10. Its
observed mean squared log-likelihood error is much larger than the exact
current-observation Gaussian guide. All576 final evaluations complete and all
fits pass their declared validity checks. This rejects promotion of the current
diagonal, positive-floor candidate; it does not invalidate the experiment or
reject the score-fitting mechanism.

| Dimension | Bootstrap | Current observation | Full oracle | Eq15 QR/peak | Diagonal QR | Score projection |
|---|---:|---:|---:|---:|---:|---:|
| 2 | 0.149455 | 0.0143977 | 0.00657498 | 0.0236812 | 0.0134707 | 0.00990623 |
| 5 | 1.90722 | 0.0310369 | 0.0125107 | 1.46554 | 0.47286 | 1.06441 |
| 10 | 29.0323 | 0.0692813 | 0.0392852 | 21.2091 | 21.6079 | 15.2295 |

Each entry averages eight final particle streams on each of four fresh data
sets, T8,N256. Comparisons are paired within a data set and conditional on
dimension. These are descriptive observations, not a statistically supported
ranking. The full oracle uses a negligible floor and supplies a diagnostic
ceiling; its gap includes covariance and floor effects. Equal particle counts
do not mean equal computation. All three learned methods also lose to the full
oracle in every dimension under the predeclared descriptive screen.

The new rule regresses the analytical backward target log-gradient after full
cloud whitening, fits a symmetric precision and projects its Gaussian to a
diagonal covariance. Full Gaussian fixtures recover means and covariances at
all five tested dimensions2,5,10,40,80, using N256. Independent R recovery agrees
within4.89e-15. Rank-deficient and nonpositive-precision controls are rejected.
The positive-floor target gradient passes finite differences and independent R.
For all12 fresh cases, R reproduces the recursive candidate coefficients within
5.33e-15 and verifies exact Kalman, current-observation and
finite-initial-integration oracle calculations. Every fixed kernel traces once.
No exact backward oracle coefficient enters the candidate fitter.

The comparison intentionally uses one backward fit on the same bootstrap pilot
cloud. It isolates the fitting rule; it does not execute Algorithm4 to adaptive
convergence and does not reproduce the paper's different linear study. The new
fitting objective is an explicit local extension, not Eq15 or original-author
code. Its diagonal projection and .01 floor are still consequential choices.
No runtime default or frozen R reference was changed.

| Decision | Primary status | Veto status | Main uncertainty | Next action | Limit |
|---|---|---|---|---|---|
| Do not promote current candidate | Fresh likelihood error computed | Loses conditional heuristic screen | Correlation removal versus floor versus one-pass approximation | Replay full/diagonal by .01/negligible-floor factorial | No paper or default claim |

| Inference status | Evidence |
|---|---|
| Hard veto screen | All numerical checks pass; declared heuristic promotion veto fires |
| Statistically supported ranking | None: four data sets per dimension |
| Descriptive differences | Error table and preserved paired per-data differences |
| Default readiness | Not ready; extension remains diagnostic |
| Next evidence needed | Controlled covariance/floor mechanism isolation, then fresh adaptive downstream validation if justified |

Post-run red team: successful exact-Gaussian recovery cannot guarantee accuracy
for recursively floored non-Gaussian targets, and diagonal projection discards
correlations that the regression has correctly recovered. A full-covariance,
negligible-floor control should recover the exact Gaussian recursion; if it does
not, it reveals an implementation or reference error. This is the next smallest
discriminating check. The current-observation heuristic uses full covariance,
so the present result cannot attribute its advantage solely to the objective.

Attempt03 failed before data generation because the recursive factory disallowed
zero steps. A diagnostic wrapper now calls the same bounded initializer directly
with zero steps. No data, method or comparison changed; attempt04 completes the
frozen design. The failed5.001s is charged. Independent preflight functions were
AST-checked unchanged across the repair. Old snapshots and all failures remain.

Plan: docs/plans/iapf-score-regression-2026-09-22.md. Exact commands, environment,
source snapshots, GPU growth, seeds, data, outputs and timings are preserved in
launch manifests and versioned attempt directories.

Terminal checks:446. Phase CPU 1.490s; GPU process 98.721s. Remaining 45.832301 CPU/47.767242 GPU hours.
