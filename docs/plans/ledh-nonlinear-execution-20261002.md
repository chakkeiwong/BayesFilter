# Nonlinear LEDH execution — 2026-10-02

Authorization: owner asked to continue execution after delivery of the nonlinear
master tester. This is a bounded local research screening campaign, not default
selection or admission. Base commit: 74bdc7777. Existing implementation checks
are in docs/benchmarks/ledh-nonlinear-master-results-20261002.md.

## Question and evidence contract

Does the revised marginal/pairwise correction and optional safety guard remain
numerically usable over T=20 in predator-prey and SIR d=18, and how do its actual
log likelihood and complete analytical score compare with simpler corrections?
Use one shared canonical LEDH executor. The five configurations are covariance
only, original repeated-axis correction, richer marginal, richer pairwise, and
guarded pairwise. Compare matched data/design seeds; report each model, dataset,
route and coordinate separately. All inherited controls are untuned hypotheses.

Numerical viability requires finite likelihood/all score coordinates, identical
scalar values across score directions, and value/trace parity. These establish
execution only. Accuracy would require same-target independently checked
references and uncertainty; moment fitting and low replicate variance are
explanatory, never accuracy criteria. No setting will be promoted or treated as
tuned without disjoint scope-specific calibration and validation plus a later
untouched claim. Screening data are not future untouched claim data.

Constructed simple adversaries: covariance-only reset (higher-moment fitting
must justify complexity); original capped reset (the repair's immediate
baseline); richer marginal only (mixed moments must justify their cost).
Evaluate them conditionally, never pool the two models/data/routes. A supported
accuracy loss to any simple arm vetoes promotion, not continuation of research.
Independent bootstrap particle filtering is a reference candidate, not an oracle
until its target, particle ladder and replicate uncertainty have been checked.
The existing predator-prey bootstrap reference observes x0 directly and is wrong
for this x0-transition-y1 target; do not reuse its likelihood values.

Candidate nonfinite values or invalid covariance are promotion vetoes and repair
triggers. Save all rows/traces, localize the first failed step, then execute the
planned narrower repair diagnostic. They are not continuation vetoes by themselves.
Stop affected comparisons for mismatched targets/data, broken invariants or
missing required evidence. Stop the entire campaign on resource conflict or
exhaustion of the total budget. Do not overwrite an attempt or drop failed seeds.

## Stages and budget

Total new allowance: 3600 GPU seconds (including compilation and failed attempts),
600 CPU seconds for reference tests/analysis, at most eight GPU launches. One
RTX 4080 SUPER; sequential GPU workers, growth enabled, XLA. No training/HMC,
package mutation, external compute or additional agent. Reserve at least 600 GPU
seconds for failure localization/reference checks rather than consume everything
on the comparison grid. Twenty minutes for any needed manuscript rebuild.

1. T20/N1008 first screen, both models, IID, all five arms, fresh dataset 260401,
   design seeds 260501 and 260502, FP32/TF32, full traces for the first design.
   Limit 1200 seconds overall / 580 per model. Actual values are descriptive.
2. Inspect failures before expansion. If invalid, compare the identical case in
   FP64, or a focused covariance/guard trace, within 600 seconds. Preserve the
   target/data and report precision scope changes. Do not tune to reference
   accuracy. A failed candidate does not cancel the other model or repair stage.
3. Spend the remaining comparison allowance on a second fresh screening dataset
   260402 and matched replicated designs, extending to inverse-CDF/permutation
   only if core numerical viability allows. Freeze the tested controls before
   this dataset. Record any incomplete matrix exactly; no implied coverage.
4. Audit a same-target independent bootstrap reference. If existing code cannot
   represent both timing and full score, implement a diagnostic TensorFlow
   bootstrap/Fisher-score reference with written derivation and a cheap linear
   Gaussian check before its GPU use. Use independent seeds and at least two
   particle counts under the remaining budget. Report finite-particle references
   and uncertainty honestly; failure to resolve a score is an unresolved oracle,
   not permission to rank filters using moment loss.
5. Save per-coordinate likelihood/score tables, failures, budget, interpretation,
   decision and inference-status tables. Update the checkpoint and monograph
   with substantive findings when available. No automatic expensive rerun.

Versioned output root: docs/plans/artifacts/ledh-nonlinear-execution-20261002/.
Result: docs/benchmarks/ledh-nonlinear-execution-results-20261002.md.
Checkpoint: docs/reset-memos/ledh-nonlinear-execution-20261002.md.
The master uses the earlier implementation plan in its internal manifest;
this campaign record and exact commands identify the new execution authority.

## Assumptions and pre-mortem audit

T20/N1008 is the documented diagnostic starting scope; N satisfies both 2d
residual and exact chunk policies. It is not a justified universal particle
count. Inspect ESS/covariance and use reference-count refinement to expose risk.
The KSC-derived numerical controls, radius 8, damping .01 and trust radius .5
are warm starts only. All choices remain visible in manifests; the guarded arm
gets a non-harm/safety comparison with its otherwise identical unguarded arm.
No inherited cap, including disabling its identity region, is promoted here.
FP32/TF32 and GPU/XLA are engineering defaults, with FP64 explicitly labeled
reference/localization. Differences may arise from roundoff or reference bias.
Data use a parameter-independent N(mean,I) initial law, transition then y1,
and the current canonical model covariances. Preserve exact observation hashes.
Two seeds are an early screen; later means/SEs with few designs remain descriptive.

Skeptical audit passed for screening: wrong historical timing was identified;
no existing nonlinear exact oracle is assumed; no proxy substitutes for accuracy;
failed candidates have a planned localization path; three simple arms and
conditional outputs are present. A full accuracy/default comparison would be
premature. The next command answers viability at the longer horizon, with
explicit limits, and cannot be reported as successful calibration or admission.

## First screen and localized repair — execution update

Screen-01 used 323.524 GPU-controller seconds: PP 10/10 finite, SIR 0/10.
SIR first failures include covariance-only at t=4 and richer/original at t=3.
This triggers precision localization, not rejection of pairwise repair. Replay
adds checked `--dataset-file` and `--random-input-dtype float32`: FP64 consumes
the exact FP32 observations and particle random inputs, keeping scope/data
separate. Run covariance-only and guarded pairwise first, both design seeds,
600-second maximum. Controls are unchanged. No accuracy tuning is performed.
Skeptical repair audit: regenerating in FP64 would change both data and RNG;
that flaw is repaired before launch by explicit source-dtype replay and hashes.

## Independent reference derivation and pre-run audit

For the exact target, Z(theta)=integral p_theta(x0:T,y1:T) dx0:T. Differentiating
under this finite-horizon Gaussian integral gives
`grad log Z = E[sum_t grad log f_theta(x_t|x_(t-1)) + grad log g_theta(y_t|x_t) | y]`.
Initial N(mean,I) contributes zero. This is Fisher's identity by direct quotient
rule; states are held fixed in the complete-data partial derivatives. For a
Gaussian factor with residual r=x-mu(theta), covariance C(theta), derivative is
`r' C^-1 dmu + (r' C^-1 dC C^-1 r - tr(C^-1 dC))/2`. Canonical model adapters
supply analytical RK derivatives and the SIR observation-variance derivative.

The independent bootstrap algorithm samples x0, propagates before every y,
weights by the observation density, accumulates complete-data derivatives along
each particle ancestry, reports their weighted average, and resamples particles
and accumulated scores together. Its likelihood is the product of average
unnormalized observation weights. The score is a finite-particle approximation
to the exact target score, NOT a derivative of the discontinuous finite random
PF program and NOT an alternative canonical LEDH score path. Path degeneracy
can make the score noisy; preserve ESS, largest weight, distinct x0 ancestors,
and replicate dispersion. No result will be labeled an exact oracle.

The new diagnostic uses TensorFlow FP64/XLA, fixed source observations/physical
theta, fresh independent seeds, and N=8192/32768 (initial four replicates each).
Two counts are a refinement diagnostic only; no automatic convergence status.
Report mean log likelihood plus SE, log-mean-exp likelihood estimate, mean
Fisher score plus SE, and likelihood-weighted pooled score. Validate the complete
Gaussian score against finite differences and a one-observation linear Gaussian
marginal before GPU execution. A fixed transition before y1 is part of the test.
The score loop uses one traced tf.while_loop body; resampling is inverse-CDF iid,
with linear storage. No NxN categorical sampling allocation, pfor or NumPy path.

Skeptical audit: this reference shares the model definition but no filter/reset
algorithm. Agreement cannot validate model physics; fixed-point derivative tests
check the callback connection. Independent count/seed variation is required;
four replicates provide descriptive uncertainty, not a statistical method rank.
A covariance-only or original LEDH arm remains a comparator, never an oracle.

Reference validation passed: three CPU tests (3.84 seconds); fixed-state full
score finite-difference maximum errors 2.29e-8 (PP), 5.25e-7 (SIR). The independent
linear Gaussian transition-first fixture agreed within its predeclared 6-MCSE
plus 0.01 tolerance. The reference reports log of the replication mean
likelihood and likelihood-weighted Fisher score; delete-one-replication
jackknife MCSE describes replication noise, **not finite-particle bias**.
Four replications are a diagnostic ladder, not convergence certification.
The source parameter values are replayed from the saved LEDH rows exactly.

Expanded PP screen finished in 495.270 seconds: 60/60 values and score vectors
finite; 59/60 passed all checks. Original/IID/260501 trace score differed by
0.000646591, above its tolerance. This remains a parity veto; no tolerance was
relaxed. Total charged GPU wall time so far: 899.003 seconds across three
launches. Stage 4 may use up to 900 seconds for 8192/32768-particle references on
PP data260401/260402 and SIR data260401, four independent replications each.

Stage 4 reference ladder returned in 23.529 seconds (GPU total 922.532/3600,
4/8 launches). SIR likelihood is near -678 at both particle counts, far from
finite FP64 LEDH values (-969 to -1042). SIR Fisher scores still have large
replication MCSE (25.5 for log-kappa at N32768); PP log-likelihood MCSE is about
0.026. These trigger the planned particle-sensitivity diagnostic, not promotion.
A fresh reference output extends to N131072 and N524288, the same three saved
datasets and four seeds, capped at 900 seconds. Question/target/method/hardware
and total budget are unchanged. Skeptical audit: larger N reduces stochastic
error but cannot prove absence of reference bias; report all rungs and do not
select a rung by agreement with LEDH. No controls are tuned on these data.

The extended reference ladder completed in 274.517 seconds. Total GPU wall is
1197.049/3600, 5/8 launches. SIR N524288 logL=-678.0775, jackknife MCSE=.0149;
score=(106.31,-65.96,5.586), MCSE=(5.54,2.42,.035). Finite-particle bias remains
unquantified. Additional stage validity flags now preserve already-computed
UKF predict/update, flow, callback and reset checks in the optional trace
(Class A observability; no numerical changes). A same-data FP32 SIR rerun
(covariance-only and guarded, design260501, T20/N1008, <=600 seconds) localizes
the first failure. Cheap CPU endpoint parity is required before launching.
Skeptical audit: cumulative validity alone could misattribute a UKF/flow failure
to moment repair; stage flags answer that narrower question without altering
controls, validity conditions, values or derivatives.

## Terminal execution review

The six bounded launches used 1289.335/3600 GPU wall seconds.
All planned screening/reference/localization stages completed. The failures are
candidate/reset failures, not a continuation veto against the research direction.
FP32 SIR first fails at observation4 in Contract E reset (upstream checks pass);
FP64 outputs remain far from the approximate reference. Next work requires the
reset-suboperation diagnostic and fresh scope calibration, not more unchanged
screening. No settings were promoted or tolerances relaxed. Full per-coordinate
results, all 25 conditional simple-arm comparisons, uncertainty, decision and
inference tables, and terminal red-team review are in the result note.
25 focused tests passed; compile/render verification is preserved separately.
