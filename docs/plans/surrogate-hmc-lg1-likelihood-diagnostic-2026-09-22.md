# LG1 likelihood and MLE diagnostic — 2026-09-22

The owner asks whether the small finite-posterior discrepancy actually means
surrogate HMC is failing, and requests likelihood percentages and an MLE check.
The scientific goal remains testing **DCTR-force HMC with an exact acceptance
endpoint** (B/R). Finite-likelihood accuracy (D/R) is a separate question.

Use the frozen LG1 attempt-4 claim data, T=10, N=1024 and selected controls.
Compare the saved 1,025-point finite physical log-density grid with a fresh
FP64 Kalman likelihood on exactly the FP32 observations consumed by DCTR.
Remove the uniform-prior constant before calling a value a likelihood. Report
100*expm1(log L_DCTR - log L_Kalman), grid MLEs, exact-likelihood loss at
the DCTR MLE, posterior normalization, and discrepancy over parameter space.
Grid maxima have resolution 0.00185546875; no continuous DCTR optimization is
claimed. Independently check the Kalman curve using the dense Gaussian
observation covariance, phi^abs(i-j)/(1-phi^2) + 0.5 I.

One fresh GPU/XLA batch checks the canonical DCTR physical endpoint at four
points (both grid MLEs and the nearest grid points to the exact 10% and 90%
posterior quantiles). A second evaluation repeats those same points. Check the
force and value status and agreement with the archived physical log density.
This is a deterministic likelihood/score diagnostic, not a new posterior
chain, independent data replication, or surrogate-HMC promotion.

Evidence contract: same-law Kalman is the exact comparator. Checksums,
matching scope/source identity, finite/valid force and endpoint, exact
Kalman/dense-Gaussian agreement within 1e-10, and fresh/saved DCTR agreement
within 5e-4 log units are validity checks. The latter allows FP32/XLA batch
rounding; report the realized error and never use this allowance as an
accuracy tolerance against Kalman. MLE proximity and likelihood percentages
are explanatory diagnostics only. No old equivalence look, margin, sampler
verdict or inference default is changed. A failure of input identity or value
validity stops this calculation; it does not reject the research direction.

Assumptions and skeptical audit: retain the tuned controls as the frozen
candidate under examination, not newly selected defaults. Preserve the same
prior, observations, initial-state law and transition timing. A near-constant
likelihood error cancels after posterior normalization; peak agreement alone
cannot establish agreement of posterior width or tails. The cheap comparator
set is exact Kalman, independent dense Gaussian, and the constant likelihood
rescaling counterexample. Evaluate center and tails separately. No learned
method is selected. The prior-removal check prevents comparing log posterior
with log likelihood. These checks answer the stated numerical question; a
new short HMC chain would not establish posterior equivalence.

Budget: at most 600 GPU-wall seconds and 600 CPU-worker seconds from the
remaining authorized campaign budget. One launch and one infrastructure-only
repair retry within the same total cap. Use a versioned output directory,
external 600-second wall timeout and process CPU limit. No new tuning or
retained-chain attempt is authorized by this diagnostic plan.

Environment: tftwogpu, GPU with memory growth, TF32 on, XLA numerical kernels;
host plotting/reporting are diagnostic. Script:
`scripts/diagnose_surrogate_hmc_lg1_likelihood.py`. Output:
`docs/plans/artifacts/surrogate-hmc-all-models-20260922-lg1-likelihood-01`.
The run manifest records the exact command, sources, archived inputs, hardware,
and measured wall/CPU time. Result note:
`docs/plans/surrogate-hmc-lg1-likelihood-result-2026-09-22.md`.

Replay repair: the first launch stopped before numerical evaluation because
the selection issuer added model/attempt validation after the frozen artifact
was issued. All 17 stored numerical source hashes match. The diagnostic loads
the archived issuer as an isolated module and calls its original selection
validator; it does not relax scope, evidence or numerical source checks. The
same consumer/reference and endpoint identities must still match. The first
attempt has no exact process resource log; reserve 60 CPU / 60 GPU seconds
conservatively. Retry wall cap is 540 seconds and process CPU hard cap is 530.
This is the one infrastructure retry. Earlier read-only CPU calculations are
accounted separately as a conservative 120 CPU-second reservation, since they
did not capture process CPU time. No old result is overwritten.
