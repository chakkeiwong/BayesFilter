# Gaussian LEDH score-error diagnostic — 16 October 2026

## Question and evidence contract

Why did the full scalar LEDH development baseline have much larger model-score
error than simple particle comparators in the fitted-guide pilot? Distinguish
an incorrect analytical derivative from approximation/discretization error and
from dataset sensitivity before adding more learned guidance.

The target is the six-coordinate Gaussian model score, with log process-noise
SD as the primary diagnostic coordinate. The computed LEDH score is the total
analytical derivative of its finite value program. These are different targets
at finite particle count. Exact Kalman value/score is the model oracle. The
unmodified shared LEDH executor with zero guide is the baseline; no reset,
correction, covariance or flow feature is removed. Diagnostic summaries inspect
its real trace. No alternative LEDH implementation is introduced.

Pass/fail for engineering correctness: all six finite-program derivatives
agree with common-noise central differences at step sizes 1e-4 and 3e-5 on
two fresh fixtures, with best relative error <=2e-5 after scaling by
1+abs(analytical derivative). Independently check Kalman analytical derivatives
against the same stencil. Values must be direction-invariant. Per-time value
and score increments must sum to the returned total within 1e-9 scaled error.
The same checks also apply to the observed derivative of reset mean/variance
on the focused CPU fixture. These are correctness checks, never promotion.

Nonfinite values, covariance/correction/reset validity failures, failed
derivative checks, missing diagnostics or corrupted provenance are continuation
vetoes for the affected comparison. Stop to localize/repair within the budget.
Large oracle error, heuristic loss or lack of convergence on this small grid
is a promotion veto and a repair trigger, not an execution veto. All observed
MSE, flow/N differences, tail examples, ESS, covariance and runtime differences
are explanatory/descriptive; no ranking or default promotion is supported.

## Design and constructed adversaries

Use the existing scalar adapter (Q=.09, R=.25, initial variance nominally 1,
six parameters), T=6. Generate six new datasets at alpha=0 and six at alpha=.9.
For the first two persistent datasets add a fixed +3 predictive-SD innovation
at observation index 2, leaving other observations fixed. These two synthetic
stress cases diagnose propagation and are not draws from the model population.
Keep ordinary/stress cases and independent/persistent cases separate.

Compare all four cells N={32,128} x flow_substeps={48,192}, with two common-noise
streams per dataset. Generate N=128 noises and take the first N=32 rows, using
the same process-noise columns per time. This preserves each marginal law.
Record errors at every time, before and after reset, not just the final score.
The 112 LEDH rows are a two-factor diagnostic, not a tuning grid; select no
winner and do not change any default. Check finite differences on the first
persistent ordinary and first stress case at N=32/48 steps before the grid.

Construct and evaluate three cheap comparators at each N and in every situation:

| Comparator | Rationale |
| --- | --- |
| Prior SIS | Reveals what transition sampling alone achieves without flow/reset |
| Bootstrap PF | Tests ordinary likelihood weighting and discrete resampling |
| Fully adapted Gaussian SIS | Uses the tractable optimal current-observation proposal |

Kalman is the exact certifying oracle. Keep bootstrap ancestry locally fixed
in its analytical derivative; this does not establish unbiased differentiation
of its expectation. Record a machine-readable conditional heuristic verdict.
Heuristic loss vetoes promotion but does not stop error attribution. No fitting,
guide-strength selection, KDM training or use of earlier holdouts is allowed.

## Assumptions, safety and skeptical pre-run audit

| Choice / provenance | Status and rationale | Failure mode / earliest check |
| --- | --- | --- |
| epsilon=1, Sinkhorn=5, owner development instruction | Untuned baseline; held fixed to investigate the failed candidate | Marginal imbalance; preserve row/column errors and executor validity |
| Flow 48/192, prior 24/48 sensitivity result | Diagnostic refinement hypothesis, not convergence certification | Persistent discretization bias; report paired per-time differences at both N |
| N32/128, T6 scalar pilot scope | Affordable fourfold particle intervention; fresh datasets | Insufficient asymptotics; do not infer consistency from two N values |
| Ridge1e-6, diagonal LM .01, scale floor1e-4, trust .5, strength .2, pairwise .02/cap2, coordinate cap .94/power8 | Frozen *suspect* development settings, not endorsed defaults; preserving them identifies error in the actual earlier candidate | These alter the value program. Existing calibration-only ridge sensitivity is limited; record reset moments and cap/condition diagnostics. A protection change requires its own non-harm calibration, not score-MSE optimization |
| Moment safety off, coordinate identity radius0 | Inherited suspect baseline; no safety justification or default-readiness inferred | Preserve moment feasibility, cap activity and all correction diagnostics; a fired guard stops validity. This stage promptly evaluates the existing protected trajectory without silently changing protection |
| Fixed alternating residual design | Existing Contract-E finite program, deliberately unchanged | May distort shape despite moment restoration; compare weighted child and reset moments with Kalman per time |
| FP64/GPU/XLA, TF32 off, memory growth | Reference exception isolates precision; stable signatures and full shared consumer | No production TF32 conclusion; verify device/memory policy before work |
| Seeds [20261016,710000+...] | Fresh diagnostic design, deterministic and reproducible | Few datasets and synthetic tails; descriptive only, no distributional ranking |
| FD steps/tolerance | FP64 debugging scale, two steps expose cancellation/truncation | Branch sensitivity or nonsmoothness; preserve both estimates and stop on mismatch |

Skeptical audit PASSES for this diagnostic scope: the exact oracle and actual
full baseline answer the question; same-program parity is not confused with
model-score accuracy; no guide tuning uses previous holdouts; inexpensive
adversaries and conditional stress reporting are explicit. No setting is
promoted. Reset moment observations cannot by themselves prove that reset
causes model-score error; a causal intervention would need a separate plan.
Unresolved inherited protection settings remain risks, not reviewed defaults.

Pre-mortem: the grid could look better through seed luck, cancellation between
flow and particle errors, or distortion invisible to mean/variance. Report
all cases, per-time increments, ESS, shape/cap diagnostics and both factors.
Passing FD can coexist with a poor estimator of the model score.

## Execution and bounded resources

Branch/worktree: as in the refreshed original score master. Use tftwogpu Python,
trusted GPU access, CUDA_VISIBLE_DEVICES=0, TF_FORCE_GPU_ALLOW_GROWTH=true,
TF_NUM_INTRAOP_THREADS=2 and TF_NUM_INTEROP_THREADS=2. CPU tests explicitly hide
GPU. One launch plus at most one localized retry; total GPU wall cap 1,200s
from the previous stage's unused 1,386.860s. CPU tests/reporting cap 600s.
Stop at budget exhaustion with preserved partial evidence; no old replication
allocation or additional training is authorized by this plan.

Command (fresh attempt directory, never overwrite):

```sh
CUDA_VISIBLE_DEVICES=0 TF_FORCE_GPU_ALLOW_GROWTH=true TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=2 /home/chakwong/anaconda3/envs/tftwogpu/bin/python scripts/run_ledh_gaussian_score_error.py --output docs/plans/artifacts/ledh-gaussian-score-error-20261016/attempt01 --budget-seconds 1200
```

Store manifest (commit, dirty source hashes, exact command, environment, data
seeds, hardware/memory policy, wall time, trace counts), rows, finite differences,
conditional summaries and decision under that directory. The surrounding
`result.md` must include decision/inference tables, strongest alternative
explanation and next repair. Update the master and checkpoint after execution.

## Local resource repair, before attempt02

Attempt01's CUDA index 0 resolved to the busy RTX 4080 SUPER, despite
`nvidia-smi` listing the mostly idle RTX 5080 as index 0. Nonexclusive 192-step
rows cost about 13s; projected completion would exceed the allocation. Interrupt
only our process after 493.339s, preserving 76 completed rows and both passing
six-coordinate finite-difference fixtures. Exit code 130 and the unchanged
source hashes are retained. Its old manifest's `running_grid` state is stale
because KeyboardInterrupt bypassed the original exception handler.

Use the single allowed repair launch on the RTX 5080, addressed explicitly by
UUID `GPU-d54fdcfc-c6ed-dbe7-25c7-93f737e0f93a`; remain in the same local NVIDIA
GPU/FP64/XLA reference hardware class. Runtime comparisons across devices are
not scientific evidence. The revised harness verifies every numerical source
hash, replays two saved ordinary cases to 1e-9 scaled value/score parity and
1e-12 observation parity, copies the completed rows unchanged and evaluates
only missing keys. Numerical algorithms, seeds, factors and criteria are
unchanged. Keep the derivative checks from the unchanged program after replay
passes. Treat a mismatch as a continuation veto. Catch interruption explicitly
in future manifests. Maximum remaining GPU wall allocation: 706.660s.

```sh
CUDA_VISIBLE_DEVICES=GPU-d54fdcfc-c6ed-dbe7-25c7-93f737e0f93a TF_FORCE_GPU_ALLOW_GROWTH=true TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=2 /home/chakwong/anaconda3/envs/tftwogpu/bin/python scripts/run_ledh_gaussian_score_error.py --output docs/plans/artifacts/ledh-gaussian-score-error-20261016/attempt02 --budget-seconds 706.660 --resume-from docs/plans/artifacts/ledh-gaussian-score-error-20261016/attempt01
```

Skeptical repair audit: this addresses a measured resource/selection problem,
preserves valid completed evidence and does not change the scientific target,
budget or default. Cross-device replay and dependency checks are required before
reuse. No additional repair launch is available under this plan.
