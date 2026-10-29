# SIR batching: reference comparison and materiality

## Active checkpoint

Owner correction, 2026-10-21: a fixed 1% difference from the old implementation
or its FP64 recurrence is not the scientific acceptance criterion. Compare old
and new likelihoods and scores with the Zhao–Cui adapted numerical reference,
accounting for estimation uncertainty. FP32, TF32 off and N4032 remain the
requested settings. The nominal-point batching change is accepted as immaterial
for continuing integration. This does not certify every parameter point.

This policy supersedes the percentage-based numerical blocker in the batching
execution, review, result and parent SIR NeuTra integration documents. Historical
measurements and the original summary remain preserved. The active summary is
`artifacts/sir-native-batching-20261021-01/summary-materiality.json`.

Reporting repair completed: the assembler checks data, observations, horizon,
parameter coordinates and old/new design identity, then emits the revised
summary. Focused assertions confirm all precision measurements, 15 benchmark
arms and budget totals equal the historical summary; that summary's hash is
unchanged. Only row 0 has a matched adapted reference; rows 1–3 and paired-change
MCSE remain explicitly unmeasured. No new GPU or stochastic computation ran.
The next implementation task remains the parent Phase 1 adapter and complete
update check; this correction does not require an intervening precision campaign.

## Question, evidence contract and skeptical audit

The question is whether native batching materially worsens likelihood or score
accuracy against the same model reference, while preserving the analytical
recurrence and improving feasible batch execution. Comparators are the saved
old and new FP32 kernels on identical data, parameter rows and particle designs,
and the saved Zhao–Cui adapted reference for SIR seed 26100831 at theta=(0,0,0).
The same finite filter in FP64 remains an arithmetic diagnostic, not the model
oracle. No new filter, reference, training or HMC run is part of this
reclassification; only existing artifacts are analyzed.

The previous plan had a material flaw: it promoted a convenient arithmetic
percentage to a scientific veto without calibrating it to estimator uncertainty
or inferential consequences. The revised plan preserves algebra, derivative,
shape, row-isolation and validity checks, and reports approximation error
against the model reference separately from implementation parity. The old
summary is preserved so the changed interpretation cannot erase a measurement.
This audit passes for saved-evidence reclassification, not a new accuracy claim.

| Diagnostic or decision | Role |
|---|---|
| Paired old/new changes in likelihood and score error against the adapted reference, with uncertainty | Primary materiality evidence; no universal percentage cutoff |
| Changed mathematical target, omitted derivative, invalid arithmetic, cross-row coupling | Implementation veto; repair before using the affected route |
| Wrong data/parameter coordinates, broken provenance, missing required inputs | Comparison invalidity; do not interpret that comparison |
| Full-horizon FP32/FP64 or CPU/GPU differences | Explanatory diagnostics; percentage alone cannot block integration |
| OOM or runtime cap | Capacity limit on the tested batch, not rejection of a smaller native shard |
| Too few independent replications or no reference at a parameter row | Uncertainty warning; no equivalence or inferiority claim for that row |
| Training gradient, complete update memory and downstream HMC checks | Existing later integration checks; not established by a filter benchmark |

The cheap engineering comparators remain the scalar analytical reference, old
batched kernel and repeated small native batches. Distinct/duplicate rows,
invalid-row isolation and different batch sizes remain separate checks. No
statistical filter ranking or posterior conclusion follows from passing them.

## How randomness enters the comparison

For each parameter point, generate independently randomized particle designs
indexed by r and use the same design in old and new implementations. Let their
scores be g_old,r and g_new,r. Independent Zhao–Cui fits provide g_ref,k. Retain
all component values and the likelihoods; do not select points or components
after seeing which comparison is favorable. Repeated execution with a fixed
design measures arithmetic/performance repeatability, not Monte Carlo error.

Report each mean score minus the reference mean. For independent LEDH designs
and reference fits, the covariance of that estimated discrepancy is

    Cov(mean(g_method) - mean(g_ref)) = S_method/R + S_ref/K,

where S denotes the sample covariance across independent replicates. For the
paired batching change delta_r = g_new,r - g_old,r, its mean covariance is

    Cov(mean(delta)) = S_delta/R.

The common reference cancels from this paired difference. Adding its variance
twice would inflate uncertainty and could hide a systematic implementation
change. To compare errors relative to the reference, also report paired changes
in absolute component errors and squared vector error. Any interval calculation
must preserve old/new pairing and reuse the same sampled reference for both;
it must not treat the shared reference as two independent estimates.

Assess materiality against the measured stochastic variation and consequences
for inference, rather than testing whether a component percentage exceeds 1%.
An interval containing zero is not proof of equivalence; statistical
detectability alone is not proof of material harm. Do not label a persistent
bias as randomness. Reference rank, support, local-fit and particle bias require
separate refinement evidence and are not bounded by Monte Carlo standard errors.
Replication counts and interval precision must be budgeted before a future
replication run; no universal count or uncalibrated new cutoff is introduced
by this reporting repair.

## Available evidence and assumptions

| Choice or assumption | Provenance and status | Failure mode / smallest check |
|---|---|---|
| FP32, TF32 off, N4032, T50 | Owner workload; retained | Arithmetic sensitivity; preserve full scores and FP64 diagnostics |
| Identical old/new inputs and controls | Saved benchmark manifests | Unfair comparison; compare hashes, seeds, rows and controls |
| Zhao–Cui adapted score | Two saved rank-20 fits, 100,000 paths each; approximate reference | Fit bias and few replications; retain between-fit SE and construction limits |
| Nominal reference only | Saved reference theta=(0,0,0) | Extrapolation; mark the other three benchmark rows unmatched |
| Reference FP64 data versus FP32 kernel data | Existing rounding audit, maximum SIR observation change 7.24e-6 | Precision mismatch; disclose that its score effect is unmeasured |
| Nominal change is immaterial for continuation | Owner judgment, supported by the displayed small change | Not a global equivalence claim; retain regional and downstream checks |

At the nominal point, score changes are approximately (-0.0235901, 0.0010681,
0.0003853). Dividing their magnitudes by the reference's reported between-fit
standard errors gives (0.02198, 0.000644, 0.006818). These are descriptive scales,
not z-tests: they do not estimate variance across LEDH designs. The existing
reference gap remains about 14.15% versus 14.16% in relative vector norm.
Neither gap is an automatic veto. The saved bootstrap reference also disagrees
with the adapted reference, so neither supplies an exact SIR oracle here.

## Execution and completion

Update the active plan, result, review and parent checkpoint; extend the
post-run assembler to emit `summary-materiality.json` while preserving
`summary.json`. Validate matched inputs and theta, preserve every old numerical
measurement, and explicitly mark unmeasured uncertainties and unmatched points.
Run only standard-library artifact assembly and focused consistency assertions.
Budget: no new GPU time, no new stochastic evaluations, at most five minutes of
CPU reporting work. No package or environment change.

Command: `python3 docs/plans/artifacts/sir-native-batching-20261021-01/assemble_summary.py`.
The new summary records the analysis commit, command, input hashes and elapsed
time. The baseline and benchmark source snapshots continue to identify the
executed kernels; the current working tree is not retrospectively substituted.

| Decision | Criterion status | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Remove fixed-percentage blocker | Owner materiality correction applied | Percentage-only veto retired | Replicated error comparison absent | Continue parent Phase 1 adapter and full-update checks | All-region equivalence or posterior accuracy |
| Retain batching repair | Checked algebra, graph and isolation evidence retained | No new implementation defect established | Nonlinear FP32 sensitivity away from nominal point | Carry reference/uncertainty reporting into planned accuracy checks | Estimator bias is zero |

| Inference status | Finding |
|---|---|
| Hard veto screen | No new scientific veto; full-native B128 capacity limit remains |
| Statistically supported ranking | None from a single LEDH design and two reference fits |
| Descriptive-only differences | Nominal old/new change, reference gap and standard-error ratios |
| Default readiness | FP32 remains requested; complete training/inference readiness is not established |
| Next evidence needed | Adapter/update checks; paired designs and matched reference refinement for broader accuracy claims |

Post-run red team: reference uncertainty can mask systematic bias if used as a
blanket excuse. Conversely, a rounding difference from the same finite filter
can be scientifically negligible. Retaining both comparisons, with their
distinct meanings and missing uncertainty, avoids either unsupported conclusion.
