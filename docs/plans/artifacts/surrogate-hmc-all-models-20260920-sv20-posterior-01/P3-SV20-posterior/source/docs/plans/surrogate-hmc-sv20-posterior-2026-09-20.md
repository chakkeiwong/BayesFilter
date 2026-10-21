# Actual-SV20 full-prior posterior reference

The goal remains testing surrogate HMC driven by an imperfect full canonical
DCTR force. This stage qualifies the independent posterior comparator R;
no HMC or learned component runs. The preceding
[likelihood check](surrogate-hmc-sv20-reference-result-2026-09-20.md) qualifies
twelve probes, which cannot stand in for integration over the prior.

Use the frozen attempt1 claim observations from
`surrogate-hmc-all-models-20260920-sv20-reference-01/P3-SV20-reference/claim/data.json`.
Verify the data and reference-source hashes before evaluation. Preserve the
stationary actual-SV law, sigma=1, independent Uniform(.1,.9) priors and probit
chart. This numerical reference may inspect claim data; DCTR control selection
must continue to use separate calibration and validation partitions.

Integrate likelihood times the constant physical prior over the full
rectangle [.1,.9]^2. Use parameter trapezoid grids with 16,32,64,128 intervals
per axis, initially with 512 latent-state intervals over [-16,16]. Reevaluate
the finest parameter grid using 1024 intervals over [-16,16] and then 1280
over [-20,20]. The last two latent grids share spacing 1/32. Do not silently
renormalize truncated transitions or initial densities. Compute next-state
predictive mass with the same unnormalized Gaussian transition quadrature.

The finest reference freezes means, standard deviations and marginal
.1/.5/.9 quantile cuts. Report thirteen functions: two standardized parameter
means, two centered second moments, their cross moment, six marginal CDF
values and Pr(|y21|<=.25), Pr(|y21|<=1) under the actual observation law.
For fixed x21 the last functions are
`erf(c/(sqrt(2)*beta*exp(x21/2)))`; integrate over the filtering posterior and
one-step transition. Unlike a sign probability under a symmetric observation
law, these functions depend on the latent and parameter uncertainty.
Integrate CDF cuts within the piecewise-linear marginal cells. Means/scales
and cuts are shared across every refinement comparison.

Primary qualification: every empirical numerical allowance must be at most
one tenth of the eventual function margin (.1 for five moments, .02 for six
CDF values, .03 for two predictive probabilities). The allowance is twice
the sum of (i) the maximum of the last two parameter-refinement changes and
(ii) the maximum of the two latent-grid changes, with a 1e-12 floor. Require
the last parameter change <= half the previous change +1e-5. Require the
maximum log-likelihood change on each full-prior latent refinement <=1e-6,
all predicted domain losses <=1e-9 on the two finest latent grids, and finite
positive normalization/scales. Predictive probabilities must lie in [0,1]
up to 1e-12 rounding. These allowances are empirical, not rigorous error bounds.

Diagnostic roles: failed integration, nonfinite values, wrong hashes/law,
or exhausted budget are continuation vetoes for this reference stage. Failure
of the numerical qualification blocks posterior comparisons and triggers a
separately planned refinement under remaining budget; it does not reject
surrogate HMC. Timing and observed posterior summaries are explanatory only.
No sampler ranking, convergence, DCTR accuracy or default readiness follows.

Constructed heuristic set for later surrogate-HMC evaluation: direct
two-dimensional quadrature is the tractable oracle; an independent Gaussian
proposal fitted to reference mean/covariance tests whether HMC earns its cost;
a zero-force position proposal with endpoint correction tests whether the
surrogate field adds useful movement. Evaluate centrally and in each marginal
tail using the frozen reference quantiles. Their performance is NOT_RUN here,
and missing heuristic comparisons continue to veto efficiency/default promotion.

Default audit: the existing prior/chart and stationary law define this target,
not a universal SV model. Trapezoid grids are a numerical hypothesis tested
by two independent refinements. The uniform parameter grid could miss narrow
structure; the refinement contraction and full-domain likelihood comparison
are early diagnostics, not proofs of uniform accuracy. Predictive thresholds
.25 and 1 are fixed test functions spanning smaller and larger returns in
the model's units, not thresholds selected to favor a sampler. FP64 GPU/XLA
is an independent-reference exception; no TF32, NumPy or pfor. Host loops
only shard fixed 32-row batches; filtering remains native TensorFlow/XLA.

Skeptical audit: using KSC, dropping stationary dependence, or normalizing
truncation would answer the wrong question. The already-qualified source and
fresh full-domain checks address these risks. Sharing the finest-grid scale
prevents a moving comparison target. Tiny probe errors alone are not the
promotion criterion. The audit passes this bounded reference preparation.

Budget: at most 600 GPU seconds and 900 CPU-worker seconds including one
infrastructure retry, plus 60 CPU-worker seconds for focused tests, within the
SV family's initial 2 GPU / .5 CPU-worker hours and total 63/15-hour campaign.
Before the ladder, compile and time fixed batches, then extrapolate all
remaining batches with a factor-three allowance plus 60 seconds for compilation
and reporting. Do not launch the ladder if either measured wall/CPU forecast
exceeds remaining stage budget. Peak allocator usage must remain below 4 GiB.
The master enforces wall/CPU bounds even if the estimate is optimistic.
Stop and preserve the artifact on a veto; do not expand grids after results.

Exact command, trusted GPU1 and verified memory growth:

```bash
CUDA_VISIBLE_DEVICES=GPU-68251639-fe82-8f81-3ccc-2953c32e805b TF_FORCE_GPU_ALLOW_GROWTH=true BAYESFILTER_PRELOAD_CUSTOM_OP=0 TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=1 /home/chakwong/anaconda3/envs/tftwogpu/bin/python scripts/run_surrogate_hmc_program.py --stage P3-SV20-posterior --scope SV20 --reference-root docs/plans/artifacts/surrogate-hmc-all-models-20260920-sv20-reference-01/P3-SV20-reference --output docs/plans/artifacts/surrogate-hmc-all-models-20260920-sv20-posterior-01
```

Preserve source/data hashes and copies, all numerical grids and predictive
arrays, timing forecast, commands/environment, seeds, accounting and report.
Result note: `docs/plans/surrogate-hmc-sv20-posterior-result-2026-09-20.md`.
Next action on qualification: actual-SV DCTR consumer mechanics and scoped
calibration. Prospective MCSE repair remains required before new inference.
