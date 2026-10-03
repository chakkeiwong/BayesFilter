# iAPF fitting-cloud and stopping-rule continuation

Owner authorization, 2026-09-19: “why can't we continue the exeuction? if there
is no need to stop, don't step.” Continue locally under this bounded allocation;
the prior eight-fit campaign stays archived. No independent review is claimed.

Status: all four scientific stages and the post-run observability repair are
complete. See the [terminal result](artifacts/younis-iapf-fitting-protocol-20260919-01/result.md).
Used 4/4 launches, 28/32 fits, 7988/8000 charged calls, 255.037560/1800 driver
seconds and a conservative 160/600 CPU check seconds. No run remains active.
The sections below preserve the pre-run contracts and sequential amendments;
the result and master program identify the remaining 1900 fitting question.

## Research intent and evidence contract

Question: does insufficient fitting-cloud coverage or premature likelihood-CV
stopping explain the weak-case physical-score failures? Use the same scalar
sine-transition/quadratic-observation model, T=2, final N=4096, six physical
parameters, and analytical normalized complete-data Fisher statistic. This is
an iAPF diagnostic, not a claim-bearing LEDH experiment.

Use a 2x2 comparison: initial fitting N=16 versus 256, and inherited stopping
(k=1, tau=100, four iterations maximum) versus an informative likelihood
stability screen (k=3, tau=.05, twelve iterations maximum). Particle caps are
eight times the initial count. The existing shared adapter and kernels execute
all four arms. Keep the original mean/SD box, floor=.001, relative-shape
objective and FP64 fitting so coverage and stopping can be distinguished from
the already-tested bound change. Labels: small_early (baseline), large_early,
small_stable, large_stable. The stopping change includes its window and cap.

The new threshold asks for <5% sample relative variation among four successive
likelihood estimates. Unlike tau=100 it can reject an unstable window. It is a
declared diagnostic accuracy hypothesis, not a convergence theorem or tuned
default. Adaptive likelihood estimates reflect both proposal changes and
Monte Carlo noise. Their CV cannot certify the fitted twist's predictive shape.
The previously derived bound CV <= sqrt(k+1) makes the inherited threshold
vacuous; preserve that arm explicitly as a baseline only.

Calibration observations: IDs 2000/2001 (weak), 2010/2011 (curved).
Untouched validation observations: 2200/2201 (weak), 2210/2211 (curved).
Check these IDs against prior reference records before launch. Within each
dataset, use shared offline seeds across fitting arms for a paired comparison.
Use 96 independent control-calibration streams and 64 separate score-assessment
streams; share these across arms, never between partitions or datasets. Each
control correction uses the existing 12 ancestor controls and six Gaussian
innovation contrasts, centered with 16 independent same-law reference clouds.
All control coefficients are frozen before assessment. All calibration scores
are tuning data; all prior failed finals remain excluded.

Select one repaired arm among those producing valid fits on all four
calibration datasets, using mean squared error of the innovation-corrected
physical score against the refined grid reference, equally weighted across
datasets. Baseline is retained separately, and no superiority is claimed from
selection. Bound contact, shape mismatch and heuristic errors are reported but
do not tune this selection. Freeze the selected protocol before generating
validation observations. Refit baseline and selected protocol independently
on each new validation dataset; these observation-specific offline fits are
part of the algorithm, not retuning of protocol controls.

Primary validation criterion: on EACH of four validation datasets, the paired
99.75% percentile-bootstrap upper bound (40,000 resamples) for corrected-score
MSE(candidate) minus MSE(baseline) is below zero. This is an approximate 99%
four-comparison family. Other arm/metric comparisons are descriptive. Intervals
condition on the realized observations, proposal fits and control fits; they
do not integrate selection, fitting or population uncertainty.

Promotion vetoes: any observed candidate loss to EKF (local linearization),
UKF (nonlinear moment propagation), or no-resampling (removes ancestor
randomness), evaluated separately on every validation dataset; failed
simultaneous mean-bias screen; any fitting-bound contact; invalid fit. Keep
raw Fisher and ancestor-only controls as essential method comparators. A
candidate failure does not invalidate the research direction.

Continuation vetoes: wrong device, nonfinite accepted numerical output,
failed refined reference, accidental seed collision, source drift, corrupted
or missing required records, or resource exhaustion. An adapter capacity or
optimizer failure rejects that arm/dataset and is recorded; it does not halt
the other predefined arms. Do not use a last unconverged iterate as a fit.

Repair trigger: if every repaired calibration arm is invalid because its
iteration/particle/optimizer cap is exhausted, run the predeclared reserve arm
N=1024, cap=8192, k=3, tau=.05, max_iterations=20 on the same calibration
datasets and then validate if all four fits are valid. This consumes four
reserved fits. No final-score-driven restart or new arm is permitted.

Explanatory diagnostics: per-iteration likelihood and CV, particle counts,
optimizer convergence and bound contacts, predictive-grid shape residual,
pointwise floor fraction, variance, regression rank, numerical bias screen,
trace count, device placement and timing. Preserve diagnostics from rejected
fits. Shape/likelihood/CV improvements alone cannot promote a score method.

Do not conclude general iAPF/KDM superiority, unbiased finite-N scores,
population performance, twist convergence from CV, longer-horizon or
multidimensional validity, tuning admission, a default change, LEDH admission
or HMC readiness. The fixed-label derivative remains a distinct diagnostic.

## Defaults, risks and skeptical review

| Choice/status | Reason and provenance | Failure mode / early check |
|---|---|---|
| N=256 hypothesis, cap=2048 | 16-fold cloud increase tests poor coverage within the same hardware budget | Rare regions may remain missed; independent predictive-shape check and validation score |
| k=3/tau=.05 hypothesis | Four-estimate window and explicit 5% relative variation requirement | Noisy/premature stop or cap exhaustion; full controller history and rejected fits |
| Original box and floor, diagnostic baseline | Isolates coverage/stopping; positive floor preserves support | Known bound/floor pathology; record contact/floor share, prohibit promotion when veto fires |
| Relative-shape objective / FP64 fitting | Existing checked shared implementation, avoids absolute-density scale underflow | Optimizer/local-family mismatch; gradient/convergence and independent shape residual |
| 96 calibration/64 assessment streams | Same cost/evidence scale as prior stage | Noisy coefficient fit/selection; frozen independent assessment and conditional uncertainty |
| N4096/T2, FP32/TF32/XLA | Existing final evaluation scope, no horizon transfer | Narrow conclusions; exact scope recorded |
| Six score coordinates equally weighted | Existing Euclidean model-score criterion | Parameterization dependence; retain per-coordinate bias/variance |
| Zero new damping/ridge changes | Existing checked input-precision rank safeguard remains active | Rank/conditioning artifacts checked; no numerical policy alteration |

Pre-mortem: a smaller CV could hide a poor twist or a noisy normalized score;
therefore only downstream untouched score accuracy is primary. Larger clouds
could fail merely through fitting caps; preserve the explicit failure and use
the reserve only for the specified cap failures. Averages could hide weak-case
failure; evaluate and veto per dataset, with three constructed heuristics.

Skeptical pre-execution review PASS: target/comparators remain matched; no
fitting diagnostic substitutes for score accuracy; tuning and validation
observations/streams are separated; CPU references and intended GPU are
explicit; invalid candidates do not veto a planned repair; resource/stop
conditions and statistical limitations are stated. Codex self-review only.

## Execution and budget

New continuation ceiling: four launches, 32 adaptive fits, 8000 conservatively
charged filter/fit calls, 1800 cumulative driver seconds, 600 CPU check seconds.
Planned work: 16 calibration fits, eight validation fits; four reserve fits
and remaining capacity cover the specified cap repair or infrastructure retry.
Charge up to twice each maximum iteration count per fit before launch; do not
refund rejected fits. Prior campaign expenses remain separately archived.

Environment: `/home/chakwong/anaconda3/envs/tftwogpu/bin/python`; pinned physical
RTX5080 UUID `GPU-d54fdcfc-c6ed-dbe7-25c7-93f737e0f93a`; escalated execution,
TF_FORCE_GPU_ALLOW_GROWTH=true and verified pre-initialization growth;
OMP_NUM_THREADS=4, TF_NUM_INTRAOP_THREADS=4, TF_NUM_INTEROP_THREADS=2.
Particle numerical kernels use stable signatures and XLA; CPU FP64 grid
references and FP64 diagnostic regression are explicitly separate exceptions.
CPU tests set CUDA_VISIBLE_DEVICES=-1 before TensorFlow import.

Driver: `docs/benchmarks/diagnose_younis_iapf_fitting_protocol.py`, stages
`calibration` and `validation`, explicit fresh `--output` under
`docs/plans/artifacts/younis-iapf-fitting-protocol-20260919-01/`.
Validation reads the completed calibration selection; it cannot override it.
Every launch snapshots source, command, environment, GPU identity, parameters,
observations, seeds, raw scores, frozen fits/coefficients, budget and timing.
Preserve prior attempts. Update master/checkpoint after each completed stage;
terminal result includes decision/inference tables and strongest alternative
explanation. Continue between authorized stages without another permission gate.

## Post-validation mechanism test, declared before execution

Calibration and validation are complete and immutable: 24 fits, 4672 calls,
161.918923 driver seconds. Validation produced 0/4 significant primary gains,
with an EKF/UKF loss on dataset2200 despite valid interior fits and small shape
residuals. The exact sample identity MSE=(63/64)trace(sample covariance)+squared
sample-mean error assigns 96.6--99.8% of observed error to variance. This is
descriptive attribution, not a proof of negligible true bias.

Continue under the same total ceiling with a separate precision experiment;
this amendment does not reopen fitting selection, change its criteria or
retune on validation. Freeze large_stable exactly as selected. Fit once on each
new dataset2400/2401 (weak),2410/2411 (curved), then evaluate final N=4096 and
16384 using the SAME observation-specific proposal. Fourfold N is the explicit
Monte Carlo hypothesis that variance falls by about four and standard error by
about two. This ratio is chosen to discriminate the variance mechanism, not to
optimize against a heuristic. No constant factor is assumed as a theorem.

Use 96 fresh control-calibration and64 fresh assessment replicates per N, with
independently fitted control coefficients for each N; share seed pairs across
particle counts for paired evaluation. Every count has its correct marginal
draw law; common seeds do not assert exact prefix equality across XLA shapes.
Offline fitting, control fitting and assessment remain mutually disjoint.
Use matching-N no-resampling, plus EKF and UKF on each dataset, and retain raw
Fisher/ancestor controls. Record particle work and timing: this changes compute
cost, so it cannot establish equal-cost superiority or a new default.

Primary precision criterion: negative upper endpoint of paired99.75% MSE
difference intervals, N16384 minus N4096, separately on all four datasets.
Mean-bias, heuristic, bound, reference, numerical, source/device and budget
rules above remain applicable. Improvement against N4096 does not override
an observed heuristic loss. Variance ratio, MSE ratio, ESS and timing explain
the mechanism; they cannot independently promote a method. This is a new
four-comparison family, not pooled campaign-wide inference.

Skeptical amendment audit PASS: fresh observations prevent validation reuse;
fixed selected proposal protocol prevents fitting retuning; control coefficients
are recalibrated for each changed particle scope; all comparators are retained;
the increased computational cost and conditional uncertainty are explicit.
No framework, kernel, reset, score target or production default changes.
Planned extra charges: four fits,1936 calls,one launch. Capacity remaining after
completion is only for a localized infrastructure repair. Driver stage is
`precision`; output is a fresh `precision01/` in the same campaign root.

## Frozen failure replay, declared after precision and before execution

Precision01 passed all four conditional primary comparisons and heuristic
screens. All its new datasets also cleared heuristics at N4096, so those
results alone do not answer the earlier UKF failures. To avoid reporting only
favorable observation situations, use the last launch for a diagnostic replay
of datasets1900,1901,2200. These are intentionally selected known failures;
the replay cannot provide untouched or population-level confirmation.

Freeze EXACTLY the previous fresh01 proposal for1900/1901, and the selected
large_stable validation01 proposal for2200. Do not refit, retune, alter bounds,
or recompute their adaptive stopping decisions. Evaluate N4096 versus16384
with96 new control-calibration and64 new assessment replicates, matching-N
no-resampling, EKF and UKF. Reserve fresh seeds. Proposal identity and executed
observations must match the archived input records exactly. The old final
scores remain immutable and are not reused as the new Monte Carlo sample.

This isolates final-particle precision conditional on the problematic frozen
fits, including the known poor/bound-contact fit for1900. The existing bound
contact remains a promotion veto whatever the replay scores show. Score
comparisons and heuristic failures are diagnostic; the declared paired99.75%
intervals cover a separate three-comparison family (at least the nominal99%
Bonferroni level, subject to bootstrap approximation). Mean-bias screens retain
the existing conservative critical value. No repair selection follows from
this replay. A remaining loss identifies unfinished work; it does not reject
the iAPF research direction.

Skeptical replay audit PASS: deliberate failure-case selection is disclosed;
fresh randomness and exact frozen-fit provenance isolate the particle-count
effect; no historical score sample becomes a fresh holdout; three adversaries
remain evaluated conditionally. Add1380 calls and zero fits, giving planned
campaign totals4 launches,28 fits,7988 calls under the unchanged ceilings.
Stage `challenge`, output `challenge01/`. This fourth scientific launch uses
the remaining launch allocation; no further repeat is planned.

## Post-run observability repair

All four scientific stages are complete. Code inspection found that
`bounded_density_fit` computes the fitting cloud's coordinate standard
deviations, but the recursive fitter drops them before the adapter writes
diagnostics. Preserve their minimum and maximum as two appended diagnostic
columns. This measures the physical scale of the standardized fitting box;
it does not establish predictive coverage or change the fit, stopping,
proposal, or score. Existing eleven columns retain their order. Historical
artifacts keep their executed eleven-column schema and source snapshots.

Skeptical engineering audit PASS: this is computed-but-discarded observability,
not a new numerical policy or experiment. Verify the recursive call chain on
a two-dimensional cloud with independently known spread, and run existing
fitting/adapter tests with GPU deliberately hidden. The check must preserve
fitted values and existing diagnostic semantics. New artifacts use the
expanded schema; the existing strict scope validator must not silently upgrade
old results. Charge checks to the remaining 600-second CPU allowance. No
additional research fit or GPU launch is authorized by this repair.
