# Active surrogate-HMC checkpoint: warmup covariance comparison completed

Scientific goal: test surrogate HMC on the frozen M13/data1704/T20
importance-likelihood posterior. Current question: how would the matrix change
if estimated from the archived 9,000 warmup steps? COMPLETED 2026-10-02, CPU
saved-data diagnostic, zero new samples and no runtime changes.

New finding: pooled warmup empirical covariance is
[[0.07749974,-0.02292811],[-0.02292811,0.03124836]], versus frozen
[[0.09739122,-0.01087504],[-0.01087504,0.06986349]]. But the actual regularizer,
0.9*S + 0.1*diag(0.48045302), yields
[[0.11779507,-0.02063529],[-0.02063529,0.07616882]], increasing original diagonal
entries 21.0%/9.0%. The prior contribution alone is 0.0480453 per diagonal.
Thus short covariance adaptation is not the whole explanation; broad-prior
shrinkage maintains the scale mismatch. No removal/weakening is promoted.
All 12,750 retained draws under empirical warmup factor: SD 1.117/0.991,
correlation 0.106, condition 1.377; existing shrinkage gives 0.906/0.578,
correlation -0.136, condition 2.567. Original condition 3.083. These are algebraic
geometry comparisons, not evidence that changed sampling would converge faster.

Evidence: docs/plans/artifacts/surrogate-hmc-m13-warmup-covariance-check-20261002-02/{result.md,report.json,run-manifest.json}.
Plan: docs/plans/surrogate-hmc-m13-warmup-covariance-check-2026-10-02.md.
Checks: warmup chunk/aggregate equality, source/seed/state continuity, original
affine parity, sample archive equality, Welford parity 7.22e-16 and actual saved
regularizer all pass. Success 2.54s; attempt 01 static-signature failure preserved.

Prior whitening result: runtime affine parity passed; original metric estimated
in 128/256 windows with 10% prior shrinkage, then frozen through 9,000 warmup.
Selected leapfrog count 1, epsilon 0.401. Chain 2 lowest empirical process 5%
acceptance 19.60%, squared whitened movement 0.000835 vs 0.13216 in upper 75%.
Local curvature/force mismatch and short trajectories remain possible causes.
Evidence: docs/plans/artifacts/surrogate-hmc-m13-whitening-check-20261002-02/result.md.

Master repair complete: bayesfilter/testing/surrogate_hmc_m13_cumulative.py is
shared by deadline/resume/recovery masters. All eligible post-warmup draws pool
cumulatively; no independent validation set or zero-count posterior restart.
65 focused CPU tests passed. R-hat <= 1.01 plus health is the stopping criterion;
low ESS warns about precision. Earlier 9,000 retained/chain passed 1.00742/1.00547.
Old master added 3,750; all 12,750/chain has R-hat 1.05418/1.00884, bulk ESS 77/321,
tail ESS 30/483. Keep both facts visible; no new criterion or hidden sample removal.
Evidence: docs/plans/surrogate-hmc-cumulative-master-repair-result-2026-10-02.md;
docs/plans/artifacts/surrogate-hmc-m13-reclassified-20261002-02/result.md;
docs/plans/artifacts/surrogate-hmc-m13-saved-diagnostics-20261002-01/result.md.

Budget: sampling deadline expired 2026-10-01T16:00:00Z. 12,750 of 20,000 draws/chain
used; 7,250 sample capacity remains but no unexpired sampling time allocation.
Exact next action: report completed warmup comparison. If repair is requested,
evaluate mass-adaptation windows and prior shrinkage jointly, with regional
curvature/force checks and step/trajectory retuning. That is a proposed repair
branch; unchanged master continuation would use the same frozen chains and
start with all saved draws, next look 13,000. No new sampling or retuning launched.
All-model campaign remains incomplete; no Claude review pending.
