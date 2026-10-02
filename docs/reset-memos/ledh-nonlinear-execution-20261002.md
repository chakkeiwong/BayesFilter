# Completed nonlinear screening — closed 2026-10-03

Status: the T=20/N=1008 predator–prey and SIR-d18 screening, independent-reference
ladder, precision replay and first-failure localization are complete. This closes
the diagnostic campaign, not the outstanding SIR repair or accuracy validation.
The executed code, raw results and LaTeX updates are archived in commit
`77be36b951de9b9a98225758554fe895cdf5a505` on `sqmc-development`.

Question: how stable and accurate are the likelihood and full analytical score
under the five correction arms? Evidence: [plan](../plans/ledh-nonlinear-execution-20261002.md),
[results and decision tables](../benchmarks/ledh-nonlinear-execution-results-20261002.md),
[master-program final summary](../benchmarks/ledh-nonlinear-master.md#completed-campaign),
and [execution manifest](../plans/artifacts/ledh-nonlinear-execution-20261002/execution-manifest.json).

Compute used: 1289.335 of 3600 GPU wall seconds; six of eight permitted launches.
The unused budget is 2310.665 seconds and two launches; no further run is pending
in this completed campaign. Preserve all failed rows and original artifacts.

Checked findings: predator–prey was finite in 70/70 evaluations and valid in
69/70; one original-arm trace-score check failed. SIR FP32 was invalid in 10/10.
Same-input SIR FP64 was finite in 4/4 and valid in 3/4, with log likelihoods
between -1041.829818 and -969.027666 versus the approximate reference -678.077466
(MCSE 0.014864). The range includes the finite row with a trace-score veto.
The first SIR failure is observation 4 in Contract E reset: upstream UKF/flow
checks pass, covariance-only reset states are finite but tangents are nonfinite,
and the guarded correction receives invalid input. The exact failing reset
factorization or derivative operation has not yet been identified.

Reference: independent bootstrap/Fisher estimates at N=8192, 32768, 131072 and
524288, four replications per rung. All likelihoods, score coordinates and
replication uncertainties are retained. These are approximate references;
finite-particle bias remains unresolved. There was no scope-specific tuning or
untouched claim run, and no statistically supported ranking or admission.
The original Zhao–Cui algorithm was not run on these saved datasets. Its inspected
author-code `lml` averages finite log importance weights, not log mean importance
weights; a same-target comparison needs a separately verified normalizer and score.
See the source anchors in the master-program summary.

Validation completed: 25 focused CPU tests passed (including the three reference
tests; do not double-count them), with GPUs intentionally hidden. The scientific
commit also passed the three oracle-contract hook tests. Both monograph chapter
copies agree; the 606-page build has no unresolved references/citations, and the
changed pages were rendered and inspected. Build records are under
`../plans/artifacts/ledh-nonlinear-execution-20261002/monograph-build/`.

Closeout audit: checked the summary against saved numerical results, run accounting
and the author source. Completion does not turn finite outputs or small descriptive
errors into promotion evidence. Git delivery status is determined from live refs.
Next scientific action: diagnose the observation-4 reset factorization and tangent,
evaluate any numerical protection for non-harm, then use fresh scope-specific
calibration and holdout partitions. Retain the separate same-data Zhao–Cui
comparison gap. Do not rerun completed screens or relax trace tolerances.
