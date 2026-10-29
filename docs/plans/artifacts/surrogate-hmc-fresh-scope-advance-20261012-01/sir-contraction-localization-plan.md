# SIR Gaussian contraction localization

Question: is the SIR FP32 score discrepancy materially caused by the centered
matrix identity used for Gaussian log densities and location derivatives?
Both the direct pairwise difference and matrix contraction are algebraically
equivalent and implemented in the same shared routine. Neither changes the
statistical target, particle design, epsilon, or number of reset iterations.

Saved full T50 traces cross the 1% cumulative score scale at step5/6, with
minimum reset relative eigenvalue margins about0.009. This rejects a story of
an observed near-threshold Cholesky failure, not other sensitivity mechanisms.
See sir-saved-trace-localization.json. Four freshly tuned matrix candidates
(epsilon0.5,1,2,4) failed precision; none may enter HMC.

Run the existing precision command for sir_d18 with direct contraction on the
same saved data/design/baseline controls as the completed matrix comparison.
FP32 and FP64 use identical FP32-quantized inputs, N4032,T50,B2,TF32off,XLA.
Budget at most2400s total, each process at most1200s, zero timing repeats;
charged to the existing36000s ledger. No concurrent GPU experiment.
Root: artifacts/surrogate-hmc-sir-direct-precision-20261012-01.

Primary diagnostic criterion: finite complete trajectories and final value and
every score component within1% of FP64, scaled by max(1,abs(reference)).
Both contractions are engineering baselines, not exact-model oracles. This is
localization, not promotion. A pass triggers fresh own-scope tuning on new
calibration/validation seeds; a fail motivates numerical sensitivity analysis.
Do not tune on these comparisons or reinterpret small likelihood errors as
small score errors. Runtime, per-step score discrepancies, tangent magnitudes,
and covariance margins are explanatory. A timeout consumes budget and leaves
the comparison unresolved. Invalid inputs or corrupt evidence veto this run.

Default audit: FP32/N4032 are owner choices; direct contraction is the preserved
algebraic baseline, not presumed faster; FP64 is an explicit precision reference;
existing data are appropriate for localization only. Same inputs/controls and
all score components prevent a confounded comparison. No posterior readiness
or method superiority is implied. Heuristic dominance is inapplicable to this
same-finite-filter arithmetic diagnostic; model comparisons remain separate.

Skeptical audit passed: a different contraction can remove cancellation but
cannot cure sensitivity intrinsic to the finite reset. A negative result is
useful and does not terminate the campaign. After the active range sequence,
first finish the predeclared LG consumer timing, then this localization and
paired force-cache validation under the remaining allocation.
