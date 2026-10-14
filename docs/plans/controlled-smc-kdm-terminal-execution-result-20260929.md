# Controlled-SMC/KDM execution and terminal review — 29 September 2026

The reviewed executable tranche is complete. The full neuroscience Figure 4
right experiment ran successfully: 20 process variances, 100 independent
repetitions per method and variance, and 4000 final estimates. The exact
Appendix B mixture-Kalman controls, the constructed sequential KDM/IWSG
sampling-to-SGD consumer, and the Lorenz normalization identity also pass.
**The two full replications remain incomplete.** iAPF remains deferred and
incomplete; controlled-SMC progress does not close it.

The master is [here](controlled-smc-kdm-replication-master-program-20260928.md).
All numerical attempts, including failures, are preserved in
[the campaign directory](artifacts/controlled-smc-kdm-replication-20260928).
No numerical process from this campaign remains running.

## What the review repaired

The starting master described phases but did not provide an executable
implementation for them. It also needed precise source-to-experiment mappings,
a reconciled compute balance, and separate correctness and empirical criteria.
The first execution specification, source map and dispatcher supplied these;
the [initial result](controlled-smc-kdm-first-execution-result-20260929.md)
preserves the reference/environment repairs and initial validation.

The neuroscience source uses negative-log quadratic regression and
multiplicative policy refinement. Its fitting procedure is explicit and has
now been executed and checked against the TensorFlow port. That removes the
specific uncertainty about iAPF's reconstructed nonlinear Gaussian-plus-floor
fit; it does not prove anything about the deferred iAPF implementation.

The TF port currently supports scalar affine-Gaussian state dynamics with the
declared observation potential. It is not a multivariate Lorenz implementation
or a natural-scale RBF policy implementation. The master now states those
boundaries. Source historical CPU-cost particle counts are not equal GPU costs.

The source audit found two additional substantive issues. Lorenz source drivers
omit Gaussian observation constants even though the paper defines normalized
densities; the executable identity below confirms their effect. The multimodal
example uses nonnegative least squares over RBFs, not the neuroscience quadratic
fit, and its text/source disagree between 100 and 101 observations. The named
repetition driver is absent. These require explicit implementation and source
conventions, not repeated tuning of the scalar quadratic code.
See [paper/source anchors](../reference/controlled-smc-remaining-source-audit-20260929.md).

## Execution evidence

| Phase / artifact | Result | Supported scope |
|---|---|---|
| Initial source and scalar validation, attempts03–12 | Source demo; 49 short and 34 full source-parity checks; 12 exact Kalman controls; 15 KDM/IWSG primitive checks passed | Declared source operations and scalar/primitive correctness |
| Native repetition batching, attempt14 | 137/137 passed | Same computation as the preserved scalar core, including full-horizon checks |
| [Figure 4, attempt15](artifacts/controlled-smc-kdm-replication-20260928/attempt15_c3_figure4/summary.json) | 4000/4000 valid estimates; 8309.624204 charged GPU seconds | Complete execution of the frozen Figure 4 right source matrix |
| [Exact mixture-Kalman, attempt16](artifacts/controlled-smc-kdm-replication-20260928/attempt16_k1_mixture_kalman/summary.json) | 12/12 passed; 5.278258 GPU seconds | Exact terminal density, normalizer and seven-parameter gradient controls for the constructed Appendix B model |
| [Sequential KDM, attempt18](artifacts/controlled-smc-kdm-replication-20260928/attempt18_k2_sequential/summary.json) | 38/38 passed; 13.494694 GPU seconds; maximum recorded parity error 4.44e-16 | Actual author resampling/weighting/KDE reference versus native-batch TF filtering, loss, gradient and one SGD update |
| [Lorenz normalization, attempt19](artifacts/controlled-smc-kdm-replication-20260928/attempt19_c1_lorenz_normalization/summary.json) | 8/8 paired checks, twelve forwards; 1.667372 CPU seconds | Unchanged author APF consumer, constant-restoration identity without refitting |

Attempt17's KDM failure was a diagnostic error. All 34 within-mode source
checks passed, but the final cross-mode comparison called tf.cast on Python
lists, rounding their values through float32. It reported 5.41922e-8 although
the saved FP64 values differed by only 4.44e-16. Attempt18 uses explicit FP64
conversion and adds a host-value roundtrip regression. Filter code, samples,
tolerances and scientific criteria were unchanged. The failed 14.452126 GPU
seconds remain charged. The [phase plan](kdm-sequential-consumer-validation-20260929.md)
records the repair and review.

GPU runs used TensorFlow 2.20.0-dev0+selfbuilt, FP64, TF32 disabled, XLA on,
and verified memory growth before initialization. Figure 4 ran on GPU:0
(RTX 4080 SUPER). Each invoked filter/fit configuration traced once; reporting
kernels also traced once. Twenty different process variances create twenty
configured kernels, so this is not a claim of one graph for the whole grid.
Repeated time/particle computations use native TensorFlow operations and loops;
outer Python loops schedule configurations, repetitions and artifact writes.
There is no pfor or NumPy numerical runtime in these TF paths.

The Figure 4 TensorFlow allocator peak was 602,937,856 bytes. The smallest
recorded regression Gram eigenvalue ratio was 2.59218e-9; no scalar precision
projection fired. Minimum final ESS was 81.5192 of 128. These are recorded
diagnostics, not separate accuracy or performance criteria. The [batch study](controlled-smc-repetition-batch-validation-20260929.md)
also preserves larger validation-memory measurements; the terminal peak does
not replace those.

## Figure 4 interpretation

[Rendered comparison and uncertainty intervals](artifacts/controlled-smc-kdm-replication-20260928/figure4-review-20260929-01/figure4-source-comparison.png)
([PDF](artifacts/controlled-smc-kdm-replication-20260928/figure4-review-20260929-01/figure4-source-comparison.pdf))
were inspected after rendering. Raw arrays, every repetition, fitted policies
and diagnostic tensors remain in attempt15. The display uses the frozen
3999-resample pointwise 95% bootstrap intervals.

The computed statistic is exactly the source's
log10(sample_variance(logZ)/mean(logZ)^2), with sample-variance divisor 99.
It is not relative variance of Z and does not establish unbiased log likelihoods.
The TF and saved author curves have similar overall shapes. Local-minus-author
intervals include zero in 17 of 20 cSMC settings and18 of 20 BPF settings.
The excluded-zero cSMC settings are .08 and .20 above the source statistic,
and .15 below; BPF is above at .12 and below at .08. This is an observed
pointwise pattern, not a simultaneous discrepancy test or equivalence proof.

For the predeclared local cSMC-minus-BPF contrast, every pointwise interval
lies below zero. Its estimates range from -3.9631 to -.4307 in the displayed
log10 statistic. This supports a lower value of that statistic for cSMC at
each individually assessed source noise setting. It does not supply a
simultaneous 20-setting guarantee, an equal-GPU-cost result, state-estimation
accuracy, or broad method superiority. The observation-guided and sigma-point
PFs and valid temporal comparisons remain absent, so the master heuristic
promotion veto remains open.

Total allocated batch/call time was 6055.54 seconds for cSMC and 2247.81 for BPF,
including each configuration's first compilation and host boundaries.
These totals are descriptive; compilation was not independently isolated and
no statistically controlled GPU speed comparison was run.

## KDM and Lorenz interpretation

The exact mixture-Kalman reference enumerates 32 Gaussian branches at five time
steps and passes independent branch-recursion, collapsed-Gaussian and gradient
checks. The KDM consumer preserves a native batch of four through sampling,
five updates, terminal KDE loss, seven-parameter gradient and one plain SGD
update. Both IWSG and truncated-gradient modes match actual author operations;
their forward densities agree, while their learning gradients differ as
expected. This closes the constructed consumer wiring question.

The coefficients, initial prior/timing and learning rate in this check are
explicit diagnostic choices. The paper specifies a 1000-sequence, length 5,
batch 64 training problem, but exact coefficients, perturbations and optimizer
schedule were not recovered. No full training curve, learned quality, gradient
estimator expectation, normalized bearings comparison or repeated-training
claim follows from source parity for fixed sampled inputs.

For Lorenz, adding the missing Gaussian observation constant leaves the tested
trajectories exactly unchanged and preserves ancestry and normalized weights.
The final log normalizer shifts by

    -H*p/2 * log(2*pi*observation_variance).

With H=101, p=6 this is +2233.85638 at variance 1e-4 and +838.48982 at 1e-2.
The maximum shift error was 1.77351e-11. These tests used fixed analytic policies,
I=0, two seeds and the supplied d=8 data. They do not establish invariance after
refitting, multivariate fit validity, or the Lorenz empirical curves. Future
code must preserve both source and normalized conventions explicitly.

## Decisions and uncertainty

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Accept Figure 4 execution | Complete frozen 4000-row matrix and uncertainty outputs | Numerical/source settings passed; heuristic promotion remains incomplete | Pointwise source differences and omitted comparators | Complete guided/temporal comparator contract and executable checks | Full controlled-SMC replication or broad superiority |
| Accept K exact reference and constructed consumer | 12 exact and 38 sequential checks passed | No checked numerical/wiring failure; exact source training settings absent | Learning behavior and source hyperparameters | Recover settings or freeze a labeled reconstruction and its target-specific training protocol | KDM learning replication |
| Accept Lorenz constant identity | All 8 paired checks passed | Normalized likelihood must not silently use source convention | Full quadratic conditioning and Table 1 data provenance | Full-covariance/RK4 source parity before a cost pilot | Lorenz empirical replication |
| Keep master active and incomplete | Current implemented tranche complete | Other source experiments, heuristics and learning remain open | Cost of full scoped empirical work | Develop the next bounded phase using the remaining inherited budget | Days spent or successful jobs imply completion |

| Inference status | Verdict |
|---|---|
| Hard veto screen | Completed phases pass their declared numerical checks. Attempt17's harness failure was repaired without changing criteria. |
| Statistically supported ranking | Pointwise source-statistic contrasts favor cSMC over the specified BPF in each individually assessed setting; no simultaneous or whole-method ranking is issued. |
| Descriptive-only differences | Source-curve resemblance, counts of intervals covering zero, ESS, conditioning, GPU times and KDM fixed-fixture values. |
| Default-readiness | Not established by this replication work; conditional heuristic and empirical training requirements remain open. |
| Next evidence needed | Guided/temporal and remaining model comparisons; recovered or explicitly reconstructed K training with independent repetitions and uncertainty. |

Post-run skeptical review: execution success did not validate the whole
research program. The strongest alternative explanations for remaining
source-curve differences are finite Monte Carlo variation and unrecovered
source random streams; deterministic source parity supports the implementation
but cannot prove distributional equivalence. A reproducible fixed-input
mismatch or a systematic source discrepancy under a separately frozen
comparison would overturn the current correctness assessment. The weakest
evidence remains KDM learning, which has not been executed here. This was a
source/code self-review; no independent reviewer or external message was launched.

## Budget and next stage

Nineteen numerical attempts consumed 74.495458 CPU seconds and 8591.712086 GPU
seconds in this successor campaign, including failures. Remaining inherited
balances are 678.671478 CPU seconds and 122359.588265 GPU seconds
(**33.988775 GPU hours**). CPU plot rendering took .543654 seconds and is
recorded separately as post-run reporting, not numerical campaign execution.
The current allocation remains CPU300/GPU12000 seconds; the attempt cap was
reassessed to 20 solely for the bounded K repair and Lorenz check. This is not
a fresh 48-hour authorization. Exact commands, commits, source/data hashes,
environment, seeds, hardware, wall times and output paths are in each manifest.

The next stage starts with the neuroscience guided/temporal comparator plan,
then full-covariance Lorenz source parity and the separate NNLS/RBF multimodal
reference. K work can proceed independently by resolving Appendix B settings
or predeclaring reconstruction hypotheses, validating normalized bearings
comparators and measuring a batch-native training trial before committing to
repetitions. The previous 8.589-hour training cost would require 85.89 GPU hours
for ten further trials, so that unchanged design cannot fit the remaining
balance. A measured faster route or an explicitly partial scope is necessary;
reducing trials cannot be labeled full replication.
