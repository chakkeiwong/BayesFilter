# Replication continuation result, 2026-09-24

The selected continuation diagnostics have finished. **Neither iAPF nor
MDPF/KDM–IWSG has completed its paper replication.** The executable master
still dispatches prerequisite diagnostics; the full experimental phases remain
specifications, not implemented and budgeted campaigns.

Authority: [master](/home/chakwong/BayesFilter/docs/plans/iapf-kdm-replication-master-program-2026-09-24.md)
and [continuation plan](/home/chakwong/BayesFilter/docs/plans/iapf-kdm-replication-continuation-2026-09-24.md).
The owner's “if not, continue” instruction authorized this separate bounded
tranche. The prior twelve-attempt ledger was preserved.

## Completed execution and evidence

| Attempt | Diagnostic | Outcome | Charged seconds |
| --- | --- | --- | --- |
| 01 | Actual IWSG resampling → normalization → NLL, eight fixed-sample cases | PASS against independent values/derivatives | 7.081 CPU |
| 02 | Eq15 covariance expansion on ten distinct saved inputs, plus exact Gaussian controls | PASS: confirms amplitude escape; does not repair the fit | 0.265 CPU |
| 03 | Source training cost, full 5,000/1,000 datasets, one pretraining/two full epochs | PASS; 1,185 optimizer steps, 11 checkpoint reloads | 81.914 GPU |
| 04 | Actual source evaluation, 5,000 trajectories, T=150, N=25 | PASS on a short-trained checkpoint; cost evidence only | 36.403 GPU |
| 05 | Sequence-gradient diagnostic | FAILED observer assertion: first transition correctly skips KDE resampling | 4.525 CPU |
| 06 | Repaired observer; source truncated sequence gradient versus independent densities | PASS, six cases | 4.976 CPU |

The second tranche used 16.846 CPU-worker seconds, 118.317 GPU-worker seconds
and six launches. It retains 1,783.154 CPU seconds, 781.683 GPU seconds and six
launches under its original caps. Unused diagnostic time does not authorize
multi-day paper experiments. No child remains running.

Every attempt preserves its actual command, source snapshot, interpreter,
seeds, hardware role, elapsed time and outputs through its recorded manifest
and [ledger](ledger.json). [Terminal evidence review](terminal-evidence-review.json)
verified four source snapshots and unchanged pinned author dependencies.
Earlier results remain evidence under their own manifests. Adding dispatcher
jobs changes conservative dependency fingerprints; those results are not
automatically reused as current-manifest checks. Twelve focused CPU harness
tests pass. The repaired assertion was checked by the actual six-case replay.

## What the mathematical checks establish

The iAPF target is the inspected paper's Equation (15),
sum_i [Normal(x_i; m, Sigma) - lambda*y_i]^2. On a finite cloud, expanding all
diagonal variances makes the density vanish. The profiled positive scale
lambda = (p dot y)/(y dot y) also vanishes, so the residual tends to zero.
This is a defect in using unrestricted scalar minimization to certify a useful
shape, not evidence that every local optimizer or the iAPF algorithm fails.

On all ten preserved distinct inputs, the squared relative shape residual
increased to 0.930–0.999 while the scalar objective fell by factors between
approximately 1.5e-29 and 6.7e-134. The calculation is correct for the stated
finite-cloud objective. A tiny objective or a successful optimizer exit cannot
promote a useful fit. Exact Gaussian controls passed. The original optimizer,
finite local-solution selection and Equation (16) positive-floor policy remain
unresolved. No substitute objective, ridge, bound or new floor was promoted.
Evidence: [escape summary](stage0/attempt02/escape-summary.csv) and
[Gaussian controls](stage0/attempt02/gaussian-controls.csv).

The MDPF target checked here is the **actual pinned source estimator**. Its
sampler uses raw mixture weights, density evaluation smooths weights by 1e-8
and normalizes them, and injected weights add another 1e-8 after exp(logp -
stop_gradient(logp)). Angular concentration is 1/(bandwidth+1e-8) during density
evaluation but 1/bandwidth in the source NLL, which also adds a density floor.
These quantities differ from an unsmoothed matched-proposal idealization.
The tests match the source quantities explicitly; they do not prove that the
source's epsilon-modified estimator is exactly unbiased for the paper target.

Eight pointwise cases cover overlap, separation, tiny weights, angle wrapping
and a single component in FP64/FP32, with fixed-denominator finite differences
and constant-loss controls. The sequence check then executes the actual
FullSequenceTrainer.do_forward_pass at B=2,T=17,N=25 with source labels and
truncated BPTT modulo four. For three sampling seeds at both random and
short-trained parameters, sampled locations match exactly and the largest
relative L2 network-gradient discrepancy is 1.308e-6, below 3e-4.
The first transition bypasses resampling as specified by source initialization;
each arm has 16 transition calls and 15 density calls.

This supports source truncated-gradient parity on the inspected fixtures.
The comparator independently expresses Gaussian/von-Mises mixture densities
and NLL; the rest of the model architecture is shared. It is not a separate
independent implementation of the whole filter. Sequence fixtures explicitly
construct trainer fields; the real constructor, optimizer and loader path was
checked separately in attempt03. Untruncated total gradients, repeated-draw
estimator bias, learning quality and paper experiments remain unproved.
Evidence: [endpoint result](stage0/attempt01/result.json),
[source/paper differences](stage0/attempt01/source-target-differences.json),
[sequence comparisons](stage0/attempt06/comparisons.json).

## Measured cost and reproducibility issues

The unprofiled training check preserved source T=17, N=25, batch sizes 64/256,
architecture, loss and learning rates, using one persistent loader worker.
Two full train+validation epochs took 33.769 and 32.495 seconds; inclusive
Trainer.train averaged 34.952 seconds per epoch. Source train/validation dataset
generation took 2.164 seconds. One pretraining epoch plus validation took
3.133 seconds, or 4.496 seconds inclusive of training-side persistence/plots.
These timings are nested; adding all reported components double-counts work.

At unchanged short-run rates, eleven trials at the 800-epoch cap would consume
about 81–85 GPU-hours for full MDPF training alone. One 400-epoch pretraining
would add about 0.35–0.50 hours. These arithmetic scenarios are **not uncertainty
intervals or a complete campaign budget**. Long-run plotting, early stopping,
other methods, the synthetic experiment, iAPF and retries are unmeasured.
The old profiled estimate of about 177 hours is superseded for this scope.
See [cost arithmetic](cost-projection.json).

The source T=150 evaluation took 29.785 seconds inside setup_evaluation,
including 13.363 seconds generating separate diagnostic data, and 36.403
seconds for the whole child. It wrote 549 MB of raw outputs, with peak allocated
GPU memory of 634 MB. Five sequence batches and 745 transition calls were
observed. NLL/RMSE are finite but use an undertrained checkpoint and cannot be
interpreted as replication accuracy or compared for superiority.

The author EvaluationBase prints a configured random_seed but actually seeds
the RNGs from its default argument zero. This check preserved and recorded
effective evaluation seed zero; the separate data-generation seed is 924131.
A final protocol must explicitly decide and disclose seed semantics. The
original author checkout remains unchanged.

Reporting erratum: attempt03's legacy `unmeasured` string includes “complete
dataset generation.” Its structured dataset sizes and sparse events actually
measure the full 5,000/1,000 source datasets. The unresolved item is full-campaign
generation/reuse and long-training behavior, not those measured datasets.

The R and PyTorch/NumPy computations are independent references. They do not
establish the TensorFlow/TFP GPU/XLA implementation. No new method-level XLA,
retracing, GPU numerical-equivalence or TensorFlow consumer result was produced.

## Decision and inference

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Retain source IWSG mechanics | Fixed-sample and truncated-sequence parity PASS | No remaining mismatch on inspected fixtures; sampler/density epsilon distinction disclosed | Repeated-draw behavior and full learning | Specify estimator expectations and source synthetic endpoint | Unbiasedness or learning quality |
| Reject scalar-loss-based iAPF fit promotion | Useful finite fit UNRESOLVED | Demonstrated amplitude escape | Original local optimizer and positive floor | Recover source choices or define an explicitly qualified reconstruction before a fresh pilot | Rejection of iAPF itself |
| Retain cost measurements | Full-sized short training/evaluation paths PASS | No infrastructure or finite-artifact failure | Long-run scaling and other methods | Inventory comparators and build complete budget/stop rules | Full campaign authorization |
| Keep joint replication incomplete | I3/K2/K3/P not complete | Missing paper results, uncertainty and TF conformance | Empirical reproduction | Finish prerequisites before full campaigns | Superiority, default readiness or HMC readiness |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | iAPF shape-fit promotion remains vetoed; one observer failure was repaired without changing the target |
| Statistically supported ranking | None; no method ranking attempted |
| Descriptive differences | Short-run timings, losses and fixture errors only |
| Default readiness | Not established by independent reference diagnostics |
| Next evidence needed | Frozen fit/floor policy, repeated-draw/source-target analysis, synthetic and bearings repeats with uncertainty and conditional heuristic comparisons, actual TF consumers |

Post-run skeptical review: the strongest alternative explanation is that these
small shared-architecture checks pass while a larger sequence, learned model,
tail or source experiment fails. New mismatched gradients or changed data/source
conventions would overturn their narrow conclusion. The weakest evidence is
the long-run cost extrapolation from only two full epochs. The iAPF result
invalidates a fit-selection criterion, not the filter, model or research
direction. No continuation veto rejects the research direction; dependent
paper runs remain blocked by unresolved prerequisites and absent full budgets.

## Subsequent K1b continuation

Attempt07 adds the [expectation result](expectation-review.md). Six diagnostic
jobs now pass in seven attempts. Cumulative consumption is 25.488099 CPU s
and 118.317269 GPU s; five launches remain. The original five-job tables above
remain evidence at their recorded stage. See terminal-evidence-review02.json
for the updated source/outcome audit. Both replications remain incomplete.
