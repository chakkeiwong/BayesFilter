# Observation-aware TT active checkpoint, 2026-09-17

## Active question and authority

Does SGQF-initialized pair TT provide useful exact-importance-corrected
Zhao-Cui filtering after covariance collapse is prevented? The owner requested
LaTeX propositions/proofs, a MathDevMCP audit, a reviewed plan and execution.
**A10 is complete; A11 is active.** Follow the [master](observation-aware-tt-repair-complete-program-20260913.md),
[A10 result](observation-aware-tt-robust-guide-20260916-result.md), and
[A11 plan](observation-aware-tt-master-amendment-11-protected-fitting-20260917.md).

Checkout /home/chakwong/BayesFilter, branch surrogate-hmc. A10 started at
50f93709d8a4c0b9481d49b07ec5373b73b23473; an external commit advanced HEAD
to 2baa30b65f40d3f22ecfe2daa58e451f0c943d5b during the campaign.
Per-run commits and source snapshots preserve provenance. Preserve unrelated
iAPF/LEDH work. No commit/push requested and no subagents authorized.

## Checked results

Nine propositions/proofs and fresh results are in LaTeX Section 17. The final
64-page PDF builds and rendered pages were inspected. All protected baseline
lines remain. MathDevMCP audit coverage is partial, with no proof certificate.
Optional implementation and actual-consumer wiring have 38 passing tests.
The exposed negative initialization scale is repaired without changing healthy
positive-scale arithmetic; calibration-02's entire selection ledger is unchanged.

Final confirmation completes all TT arms on 24 fresh cases with zero numerical,
CDF or log-evidence failures. One d4 reference is insufficiently precise:
exclude it from every admitted accuracy comparison. Scalar full protection has
19.0% lower MSE under the declared exploratory paired analysis. On eleven valid
d4 cases, guide repair MSE is .002395 versus A09 .002412; full protection is
.002747 (+13.9%, descriptive), mean ESS 303.89/512. There is no d4 or repaired-arm
statistical ranking. All candidates have a conditional heuristic promotion veto;
this is not a continuation veto or rejection of TT. Three unresolved guide
updates explicitly use the predictive fallback. The physical density bound
passes all 1,920 checked mixture steps; no realized ESS floor is proved.

Evidence root: artifacts/observation-tt-robust-guide-20260916-01/.
Terminal review: result-review.md; math audit: math-review.md; final comparison:
attempt-confirmation-02/comparison.json and terminal-checks.json. PDF:
latex/observation-aware-tt-a10.pdf. Numerical root:
../benchmarks/artifacts/observation_tt_robust_guide_20260916/.
Final manifest: attempt-confirmation-02/run_manifest.json (COMPLETE).
No A10 run is active. Preserve earlier failed attempts and source snapshots.

## Budget and exact next action

Sole ledger: artifacts/observation-tt-continuation-24h-20260915-01/budget.json.
A10 closed at 2026-09-16T18:24:33Z; charge 20313 seconds including
120-second closeout allowance. Numerical total 4192.169143 seconds
(including 150-second smoke-01 estimate) is part of this charge, not additional.
All three smoke, two calibration and two confirmation slots are consumed.
Prior reservations remain. Owner added another 86400 seconds after A10 on
2026-09-17 (local date). The extension is credited once in the same ledger,
with a 300-second allowance for this status review and explanation. Available
balance before A11: **106807.892609 seconds (29 hours 40 minutes)**.
Do not charge A10 or credit this extension again.

A11 began at 2026-09-16T19:24:00Z; 12 active-hour cap including six numerical
hours, charged once against the above balance. HEAD was 8a5c23ab1172884ffca63cac390623bf7735afe7
at A11 plan creation. The skeptical plan review passes. Protected
manuscript/budget copies and new evidence are under
artifacts/observation-tt-protected-fitting-20260917-01/.

A11 Section 18 (propositions 39–45) is written; the 69-page PDF builds and
the new pages were visually inspected. MathDevMCP coverage is partial because
document retrieval/extraction failed; its mixture-score scalar check is proved,
and its exact covariance-label check is inconclusive. See math-review.md.
The optional covariance blend, driver and diagnostics are implemented; 34
focused tests pass. GPU smoke-01 completed in 118.649815 seconds with valid
references and no candidate failures. Calibration-01 is running on GPU 1:
../benchmarks/artifacts/observation_tt_protected_fitting_20260917/attempt-calibration-01/.
All six scalar cases passed references and fitting; four-dimensional fitting
is underway. Process session 37801; run_manifest.json/result.json and per-case
summary.json preserve progress. Do not change the driver dependency files or
the hashed A11 plan until confirmation finishes. A11 has not yet been charged.

Next: finish the bounded math-review record and required regression checks,
monitor calibration, then launch the planned fresh confirmation from its
frozen controls. Preserve exact weights, A10 comparator, seed partitions,
and unrelated work. Final work: result note, LaTeX results/build, master and
checkpoint closeout, one A11 budget charge. No new permission is required.
