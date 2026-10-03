# R iAPF: diagnose the first failed fit

Status: COMPLETE after the pre-run audit, saved-cloud checks, adaptive
initialization-path check and terminal skeptical review. See the
[mathematical findings and next tests](artifacts/iapf-r-root-cause-audit-20260922-01/result.md).
This is a bounded source/code diagnosis in response to the owner's request to
step back, not another replication sweep. No runtime fitting policy changed.

## Question and evidence contract

Does the observed failure originate in the twisted-filter identities, the
printed fitting objective, or the finite-cloud numerical approximation?
Trace the worker through the shared controller, filter, backward target and
fitter. Compare against GJL's local arXiv v2 source, Sections 2, 3.1--3.4,
5.1--5.2 and the variance appendix. The earlier source reconciliation supplies
the version/provenance ledger; final published technical text and original
author code remain unrecovered. The public Sempreteamo R file is a separate,
authorship-unverified implementation. No claim about its identity is allowed.

Use two already saved d80, iteration-zero, t99 failures: dataset 92100180,
replicates 1 and 3 of the wlog1/tail8 run. They exhibit respectively a
full-rank nonconcave fit and a numerical rank failure. This purposive selection
diagnoses those failures; it cannot estimate failure rates or rank methods.

The primary checks are deterministic: replay the exact saved cloud and target;
independently derive the Gaussian product; decompose fitted quadratic
coefficients into the true diagonal contribution and omitted cross terms;
compare exact diagonal controls with the same design and weights. A mismatched
replay, failed algebraic identity, nonfinite check or source discrepancy is a
continuation veto requiring investigation. A positive fitted curvature rejects
that Gaussian fit, not the whole iAPF research direction. Ill-conditioning,
weight ESS, floor contribution and objective profiles explain mechanisms and
must not alone certify an algorithm or a stochastic performance ranking.

Preserve all inputs by path and SHA-256, executed code, environment, command,
run time, CSV checks and a mathematical result note under
`artifacts/iapf-r-root-cause-audit-20260922-01/`. No defaults change.

## Smallest discriminating checks

1. Replay only the initial bootstrap pass with the recorded seed. Check both
   saved t99 clouds exactly; fit t100 through the actual shared fitter and
   reconstruct the saved t99 target. This checks the claim-bearing call chain
   and verifies the cloud is predictive, before the time-t observation weights
   are used for resampling.
2. Independently compute at t99
   `b(x)=N(y99;x,I)[N(Ax;m100,I+S100)+c100]`.
   With c removed the precision is `P=I+A'(I+S100)^(-1)A`, strictly positive.
   Quantify the actual floor contribution and the cross-coordinate curvature.
3. For the identical points and targets, inspect weight exponents
   `a=0,.25,.5,1,2`. This is a concentration diagnostic, not tuning or nomination.
   Fit using the current QR routine and an SVD reference; report rank, singular
   values, curvature and decomposition. A small weighted ESS does not imply
   exact algebraic rank deficiency. The numerical threshold is explicit.
4. Use an exactly diagonal quadratic response on the same design with the same
   weights. This isolates rank policy and finite precision from omitted cross
   terms. Compute the cross-term projection separately, without clipping or
   regularizing coefficients.
5. Evaluate the printed profiled Equation15 on a variance-inflation path
   starting from QR, comparing its absolute loss to its scale-free residual.
   This is a counterexample/geometry check, not proof of every optimizer's path.
6. Run the shared filter with the analytic full Gaussian guide at N=64 on the
   same fixed data and compare terminal likelihood to Kalman. This is a cheap
   deterministic oracle check, not learned-fit performance evidence.
7. Code inspection exposed a separate confound: F2 calls strict QR before
   inspecting the previous fit. Summarize the saved F1/F2 failures and replay
   one F2 rejection with an optimizer-call counter and the saved valid previous
   guide. This adaptive follow-up tests a newly identified execution-path
   concern, within the same budget; it does not change a fitting policy.

## Defaults, baselines, budget and skeptical audit

All data, seeds, N1000, T100, alpha .42, model matrices and floor tail8 are
inherited solely to reproduce existing failures. They are not newly selected
settings. The exact Gaussian algebra is the certifying reference. The cheap
adversary set is constructed from this structure: unweighted diagonal QR
(removes weight concentration), exact diagonal response (removes omitted
interactions), analytic full Gaussian guide (removes learning), and Kalman
(removes Monte Carlo). Evaluate conditionally on each of the two failure
types and each weight exponent; no method superiority claim is sought.

SVD is an independent base-R diagnostic, never a silent runtime replacement.
Use the ordinary machine-epsilon/dimension singular-value threshold and report
the spectrum; distinguish it from lm.wfit's inherited 1e-7 threshold. Identity
checks use double precision and explicit absolute/relative tolerances;
conditioning-sensitive coefficient comparisons report residuals and errors.
No ridge, clipping, learned parameter selection, package installation or new
observation data is introduced.

Run CPU only: `CUDA_VISIBLE_DEVICES=-1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
Rscript docs/benchmarks/diagnose_iapf_root_causes.R <fresh-output-directory>`.
At most 600 worker seconds and three attempts including repairs; no run past
the existing 2026-09-21T20:04:26Z campaign deadline. Debit actual time from the
remaining 5011.37557939 worker seconds. Preserve each attempt separately.

Pre-mortem: smoothing-cloud fixtures could pass while the actual bootstrap
learning cloud fails; replaying the actual failure removes that mismatch.
Exact Gaussian controls could hide a learned-fit defect, so they establish
only filter identities. Poor rank could be blamed on roundoff while omitted
interactions cause a real unconstrained regression defect; response
decomposition distinguishes these. Model-specific analytic fits cannot be
promoted as a general method.

Skeptical audit: PASS for this limited diagnosis. Baselines target the actual
failure, proxy diagnostics do not promote candidates, failure and continuation
vetoes differ, the environment is an explicit independent CPU/R reference,
and all inherited fitting assumptions are under examination. Matching the
paper's table, identifying author choices, and proving general iAPF failure
are explicitly outside this run's conclusions. Further repair design follows
the evidence rather than another optimizer/floor sweep.
