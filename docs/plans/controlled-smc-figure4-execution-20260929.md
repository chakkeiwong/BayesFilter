# Controlled SMC: execute the Figure 4 right experiment matrix

Owner authorization is the instruction to review and execute the authoritative
master. The source and scalar consumer validation passed. The cost pilot and
native-repetition parity study now make this bounded empirical phase executable.
This phase collects and analyzes the source experiment; it does not close the
other C experiments, classical comparator work or K track.

## Frozen scientific question and evidence contract

Recompute the published neuroscience normalizing-constant experiment using the
author data and method settings. Published section 6.1, Figure 4 right; author
StateSpaceModels/NeuroScience/NormConst_cSMC.m, NormConst_BPF.m and
NormConst_Results.m define the exact experiment. The pinned source commit and
data hashes are in the master and every run manifest.

Use all 3000 observations, binomial n=50, alpha=.99, initial N(0,1), transition
variances .01,.02,...,.20, 100 independent repetitions per variance, controlled
N128 with three refinements, and bootstrap N5529 with zero refinements. Preserve
all four controlled forward outputs; the last is the source experiment's
estimator. The first supplies an additional bootstrap N128 control. Reset the
policy for each independent repetition. Retain the author-provided 100-by-20
normalizer arrays as separate reference evidence, never as new local runs.

The primary experiment output is exactly the source statistic
log10(sample_variance(logZ)/mean(logZ)^2), using divisor 99. It is neither
variance(Z) nor relative variance of an unbiased log-likelihood estimator.
Compute means, sample variances and this statistic for local and author arrays.
Report pointwise 95% percentile-bootstrap intervals using 3999 resamples for
each curve and local-minus-author difference, and for the local cSMC-minus-BPF
log-relative-variance contrast. These intervals are pointwise, not simultaneous
20-cell guarantees. Seeds and bootstrap streams are fixed before execution.

Primary execution pass: exactly 100 valid independent outputs per method and
variance, matching frozen settings and call chain; complete manifest, source
snapshots and uncertainty tables; one trace per invoked configuration signature.
Wrong data/settings, nonfinite values, invalid regression/proposals, missing
required rows, budget exhaustion or corrupt artifacts stop the affected run.
Do not drop failed seeds, change precision treatment, tune on outputs or silently
reduce repetition counts. Numerical failure triggers a separately recorded repair
under the unchanged campaign; it is not evidence against the research direction.

Scientific interpretation is a separate decision. An execution pass or a
confidence interval covering zero does not prove equality with the author curve.
There is no invented factor-of-two equivalence tolerance. Report uncertainty
and any source discrepancy plainly. Individual logZ values, ESS, regression
conditioning, projection counts and timings are explanatory diagnostics.

## Comparators, conditions and claims still withheld

The constructed simple comparator set is: BPF N128 (the unadapted starting
filter), the author's larger BPF N5529 (its original CPU-cost comparator), an
observation-guided extended-Kalman proposal PF, and a three-point sigma-point
proposal PF. The first two execute here; the latter two still need normalized
proposal implementations and analytic checks under the master. Their absence is
an explicit heuristic-dominance promotion veto. It does not prevent collecting
the exact source matrix. No broad superiority or default-readiness claim is
permitted from this phase.

The 20 process-noise settings are evaluated separately. Temporal informative/
extreme-observation comparisons remain unfinished. In particular, controlled
SMC's twisted incremental potentials are not ordinary filtering predictive
densities, so grouping them by observation magnitude would compare different
quantities. Those later comparisons require original-target weights or an
independent filtering reference; this phase must not substitute raw twisted
increments. No state-estimation, smoothing, Lorenz, ToyMultimodal, gradient,
HMC, KDM-learning or whole-paper claim is supported.

## Defaults, batching and budget

| Choice | Provenance/status | Risk and smallest check |
|---|---|---|
| Source settings and normalizer statistic | Published figure and pinned source | Demo/settings/statistic confusion; assertions plus preserved source files |
| FP64, source projection, normal equations | Validated source-comparison route | Poor rank or invalid precision; record every refinement's minimum eigenvalue ratio and projection count; fail closed |
| Ten independent cSMC replicas per tensor batch | Engineering choice; full-horizon parity passed against preserved scalar core | Shared weights/RNG or layout errors; batch validation checks every short replica and full-horizon reference outputs |
| One BPF replica per call | Measured engineering choice | Batched BPF was slower per replica; use the shared scalar wrapper, not a new implementation |
| 3999 pointwise bootstrap resamples | Explicit uncertainty convention | Tail instability and multiplicity; no simultaneous/equivalence or rare-event guarantee |
| Source BPF particle count | Source replication only | It does not equalize GPU cost; report all adaptation/compilation/execution cost separately |

The batching study passed its preserved-scalar numerical checks. Observed warm
cSMC time was 2.426 seconds per replica in a batch of ten versus 7.258 seconds
in the sequential pilot. BPF batch four was slower per replica (.821 vs .425),
so it is not used. These are descriptive observations, not a timing ranking
with statistical support. A conservative extrapolation of the selected design
is about 2.4 GPU hours, including a factor of 1.5. It is not a guarantee.

Allocate at most 10800 GPU-charged seconds for this phase, within the inherited
remaining ~36.3 GPU hours. Extend the current execution allocation to 12000
GPU seconds total, including prior work; this is not new compute authorization.
At most two phase attempts, with charges across retries capped at 10800 seconds.
Stop at that cap. Preserve every partial run and completed repetition. A timeout
cannot count as full replication. No additional numerical CPU allocation;
the isolated MAT reference reader's small wall time is charged to this GPU run.

Claim streams start at 1000000 (controlled) and 3000000 (bootstrap); previous
pilots used 100000/200000 families. Tensor batches contain independent random
draws and the row records bind batch stream and replica index. Bootstrap-analysis
streams start at 3900000. Global seed 20260929. There is no tuning on these data.

## Skeptical review and execution

Pre-mortem: reproducing a curve may still conceal proposal bias, source data
mismatch or an unfair GPU cost claim. Prior exact/source controls, immutable
inputs, correct statistic extraction and complete cost reporting address these
risks. A run may disagree with the source through stochastic variability, rare
regression failures or actual implementation error; preserve the evidence and
separate these explanations rather than changing thresholds. The incomplete
heuristic and other-model checks remain promotion vetoes.

Audit PASS for bounded collection and uncertainty analysis of this exact source
matrix. Scope, numerical guards, disjoint streams and stopping conditions are
explicit. It is not a PASS for completion of the master or scientific promotion.

    /home/chakwong/anaconda3/envs/tftwogpu/bin/python -B scripts/run_controlled_smc_kdm.py run --phase c3_figure4

Run with trusted GPU access and memory growth. Preserve rows.jsonl after each
completed batch/replica and progress.json after each cell. Required terminal
outputs: summary.json, curves.json, author-reference.json, observations.json,
manifest.json, command.log and a readable result note. The supervisor owns the
timeout, resource accounting and process cleanup. No message/publication is sent.

## Terminal rendering

The diagnostic report_controlled_smc_figure4.py displays the already frozen
curve statistics and pointwise intervals in PNG/PDF after all 4000 rows finish.
It runs on CPU with GPU devices deliberately hidden, makes no algorithm call
or new statistical choice, preserves input/code hashes and writes a separate
versioned report directory. Counting intervals that include or exclude zero
is descriptive; it does not issue a simultaneous or equivalence verdict.
