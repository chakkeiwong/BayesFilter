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

## Result and bounded continuation

Validation-05 completed all 1,920 fresh trials. IID, refresh-.5, sticky,
offset and drift controls passed, but refresh-.9 still failed the unchanged
unconditional bound: 17/512 failures, Wilson upper .052527. The admitted
subset passed (13/504, upper .043627). The guarded r=3 estimator increased
MCSE by roughly 6–9% in the strong-dependence group, so this is a substantive
uncertainty-estimator shortfall rather than an implementation crash. It does
not authorize posterior chains.

The next action remains inside the master: one bounded fresh evaluation of the
published over-lugsail alternative r=2, c=2/(1+r)=2/3, combined with the r=3
estimate and the original maximum guard. The paper states that smaller r
increases variance and that other r values are possible, while r=3 is its
general balance for MCMC; this candidate therefore needs its own qualification
and is not silently promoted. The same six groups, 1,920 fresh trials, exact
 thresholds, margins and two looks remain fixed. If it fails, uncertainty is an
 explicit AR2 blocker and the master records the candidate as unqualified.

## Validation-06 pre-run audit: over-lugsail r=2 candidate

The r=3 repair has now failed its fresh strong-correlation screen, so the
existing master permits one source-grounded candidate evaluation under the
same reservation. The candidate is
`ar2_guarded_over_lugsail_r2_c_two_thirds_plus_r3_v2`: for each planned batch
size it takes the maximum of the ordinary variance, the published r=3,
`c=1/2` lugsail variance, and the published r=2, `c=2/3` lugsail variance.
The r=2 formula is `3 S_b - 2 S_floor(b/2)`, with the same full-chain
centering and complete-batch truncation as the inspected author code. This
changes only the uncertainty diagnostic candidate; the AR2 target, generator,
functions, margins, looks, Wilson bounds, controls and stop conditions remain
frozen. It is not a new surrogate-HMC direction.

The pre-run audit found no unexamined implementation default: the independent
TensorFlow test compares both scalar formulas and the non-narrowing guard on
truncated batches; the calibration-stage test proves the r2 stage is selected
as a distinct master job; and the call chain uses the same wrapper for
calibration and retained assessment. The paper's discussion that smaller r
increases variance supports trying r=2, but does not promise finite-sample
coverage. A failed r2 candidate will therefore close this uncertainty-repair
branch as an AR2 blocker without launching posterior chains.

Validation-06 uses seed base 2029092300 and 1,920 wholly fresh trials across
the same six groups. The current AR2-validation reservation has 1,437.671937
CPU-worker seconds and 2,912.085978 GPU-wall seconds remaining. The focused
review allowance is 180 CPU seconds and the worker allowance is 900 CPU / 1500
GPU seconds, leaving the existing 200-CPU posterior-preflight reserve intact.
The exact master command is:

```sh
TF_FORCE_GPU_ALLOW_GROWTH=true /home/chakwong/anaconda3/envs/tftwogpu/bin/python -m bayesfilter.inference.surrogate_hmc_closure_program --stages ar2-calibration-repair-r2 --output docs/plans/artifacts/surrogate-hmc-closure-20260922-ar2-validation-06
```
