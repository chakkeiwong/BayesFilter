# AR2 uncertainty repair within the surrogate-HMC closure master

The research question remains whether full canonical analytical DCTR can drive
endpoint-corrected HMC on AR2. This step qualifies uncertainty estimates before
the primary B/R posterior test; it does not test a new sampler. LG1 stays closed.
The owner authorized continued execution of the closure master. Claude review
is deferred. This is its existing uncertainty-repair stage.

## Evidence and proposed repair

The original AR2 calibration (validation-03) failed only the unconditional
family interval-coverage bound for refresh correlation .9: 16/512 failures,
Wilson upper .050156. A single fresh confirmation (validation-04) also failed:
38/1024, upper .050524. Its admitted-chain failures were 30/1016, upper .041839.
Both failures are preserved. These bounds do not establish true undercoverage,
but they do not qualify the estimator under the declared .05 criterion.
There will be no further unchanged confirmation.

For each chain and batch length b, let S_b be b times the sample variance of
batch means, centered on that chain's full-sample mean. The candidate is the
scalar lugsail estimator S_L = 2 S_b - S_floor(b/3). Average S over four chains
and divide by 4n to estimate the variance of the pooled mean. Keep the original
40/20/10 long-batch counts, and take the maximum MCSE over those three choices.
A local guard also takes the maximum with each original batch-means variance;
the repair therefore cannot narrow an interval. Short batches discard only the
incomplete final batch and retain the full-sample centering used by author code.
All original validity tests remain necessary. Finite, positive repaired MCSEs
and the existing <=2 batch-sensitivity screen are also required for admission.

## Source and assumption audit

Vats and Flegal, *Lugsail lag windows for estimating time-average covariance
matrices*, arXiv:1809.04541v3, sections 4.1–4.3, equations (6)–(7), Theorems
3–4, and section 5 motivate r=3,c=1/2: for Bartlett/batch means the first-order
bias factor becomes -1, countering the usual negative bias under positive
correlation. Section 6.2's VAR coverage experiment and Appendix B were checked.
This is an asymptotic argument, not a finite-sample coverage guarantee.

Local primary source: `.localresources/papers/vats-flegal-lugsail-lag-windows.pdf`
(SHA256 be0fbb7189b2417f3142c32280ef7fa9b12d48cab94c11c624b3bfa5797eef4e).
The author repository `dvats/LugsailPaperCode`, `Examples/VAR/var_coverage.R`,
uses mcmcse with r=1,2,3. The author mcmcse implementation
`R/mcse_multi.R` lines 110–123 and 218–220 selects c=.5 for r=3 and implements
the displayed formula. Its `src/mbmc.cpp` centers on all n observations and
uses floor(n/b) complete batches. These files are stored alongside the paper
as `vats-flegal-lugsail-author-coverage.R`, `vats-mcmcse-author-multi.R`, and
`vats-mcmcse-author-mbmc.cpp`. Web search/status checks returned service errors;
no claim is made that publication/correction metadata was exhaustively checked.
This is a bounded method audit, not a complete literature review.

| Choice | Provenance / justification | Failure / early check | Status |
|---|---|---|---|
| r=3,c=.5 | Published MCMC recommendation; counter negative first-order bias | May overinflate or fail on AR2 functionals; fresh known-law coverage | Candidate |
| Maximum with original variance | Local Class C non-harm guard | Must never narrow an interval; independent arithmetic and monotonicity tests | Local extension |
| 40/20/10 batches, t36 critical value | Existing frozen assessment | Fixed batch counts do not establish asymptotic consistency; scope calibration required | Baseline hypothesis |
| Existing two looks and 13 functions | Measured cost plan and primary B/R family | Multiplicity, dependence, skew and prediction functions; calibrate the actual functions | Frozen scope |
| Refresh-chain calibration | Exact finite AR2 grid law with known expectations | May not represent HMC dependence; retained R-hat/ESS/variance checks still required | Diagnostic, not convergence proof |

## Evidence contract and budget

Comparator: original assessment on exactly the same inputs in focused tests;
truth for calibration is the finite AR2 reference grid law. The sampler's
continuous reference allowance remains separate. Promotion requires the same
per-regime Wilson bounds as before: unconditional and admitted family failures
<=.05, false equivalence <=.05, >=64 admitted replications, IID veto upper
<=.05, and sticky/offset/drift both-look rejection lower >=.95.

Fresh seed base 2028092300; IID/.5/.9 have 512 replications each; sticky,
offsets and drift have 128 each. Both 4000/8000 looks and every functional are
evaluated. No earlier trials are pooled or reused for qualification. One
scientific candidate evaluation; localized infrastructure repair is allowed
only within the same reservation. A calibration miss blocks this candidate's
posterior promotion and triggers inspection, not rejection of surrogate HMC.
Nonfinite outputs, changed target/criteria, corrupted evidence, or exhaustion
of the reservation stop this execution.

The constructed diagnostic adversaries are ordinary batch means (undercoverage
baseline), IID draws (false-veto check), refresh chains at .5 and .9 (dependence),
and sticky, chain-offset and drift negatives (invalid admission). The interval
width guard is checked on all input classes. Acceptance rates are not a
convergence test. No runtime, ESS or statistical superiority ranking is sought.

Before this stage the campaign has charged 31184.495488 CPU-worker seconds and
28792.499017 GPU-wall seconds. Caps remain 54000 and 226800. Reserve 900 CPU /
1500 GPU for the worker, 180 CPU for focused checks, and retain 200 CPU for the
posterior preflight. The conservative B forecast of 20517.538932 CPU seconds
still fits. The existing AR2-validation pass has 1957.873414 CPU /3656.408325 GPU
remaining, enough for this allocation. No budget increase or new hardware.

Command (trusted GPU, tftwogpu):

```sh
TF_FORCE_GPU_ALLOW_GROWTH=true /home/chakwong/anaconda3/envs/tftwogpu/bin/python -m bayesfilter.inference.surrogate_hmc_closure_program --stages ar2-calibration-repair --output docs/plans/artifacts/surrogate-hmc-closure-20260922-ar2-validation-05
```

The master preserves sources, manifests, receipts, per-replication JSONL and
group summaries. The new AR2 assessment module is used by both calibration and
retained assessment. Existing cost/force sources remain fixed: the MCSE change
does not change the measured HMC transition. The old precision forecast is a
nominal asymptotic forecast, not a guarantee of equivalence with wider intervals.

Skeptical audit: the main risk is repeatedly selecting a diagnostic to pass the
same calibration data. A source-motivated fixed correction and wholly fresh
trials address that risk; no threshold, reference or posterior margin changes.
The smallest tests check author-formula parity, interval non-narrowing, rejection
of constant inputs, and calibration/retained call-chain identity. The run may
still pass while HMC mixes poorly; actual retained diagnostics and independent
posterior agreement remain necessary. Passing authorizes the existing master's
public tuning and B/R test, followed by A/R only if the remaining budget fits.
