# SIR moment-restoration numerical repair

## Active question and execution

Owner request: trace the failure, refresh and review the repair plan, execute.
Worktree: ledh-iapf-execution-20260922; branch: ledh-refactor-with-policy-fix.
Why does the unguided SIR endpoint first become nonfinite at time 4 for
dataset 262023017 / particle seed 274000034, and can the shared implementation
be repaired without harming healthy computations? This is numerical debugging,
not a repeat guide-efficacy experiment or permission to promote a default.

1. Trace the common correction call chain and capture the first failing
   operation, its inputs, factors and analytical tangents. Preserve the
   original failure and record affected source hashes before any repair.
2. Derive the smallest shared repair from that evidence; append its exact
   mathematical meaning, assumptions and skeptical review before execution.
3. Test mechanics, total-derivative parity and healthy-case non-harm. Replay
   the actual endpoint on GPU/XLA, including all score directions.
4. Record the result, limitations and next scientific action; refresh the
   master program and concise checkpoint. Do not reuse failed holdout data
   for guide selection or report a survivor-only efficacy ranking.

## Evidence contract and research intent

Baseline: the unchanged shared runtime and saved failed campaign inputs, plus
healthy cases from the same frozen numerical scope. Primary repair criterion:
the identified numerical defect is eliminated in the actual consumer endpoint,
outputs and analytical tangents remain finite, and healthy outputs are
unchanged to stated scale-aware numerical tolerance. A mere finite scalar is
insufficient: check restoration residuals and analytical finite differences.
Explanatory diagnostics: covariance spectra, scale, conditioning, factor and
tangent finiteness, cap activity and runtime. These are not efficacy metrics.
Promotion vetoes: wrong restored target, failed derivative parity, healthy
regression, or bypass of the shared executor. Continuation vetoes: corrupted
inputs, unrecoverable environment, exhausted budget or a changed scientific
target requiring owner direction. A rejected repair triggers another bounded
localization/repair within budget, not rejection of moment training.
No score-accuracy, guide-quality, default-readiness or HMC claim follows.
Cheap adversaries for this numerical question: unchanged implementation,
identity/healthy affine restoration, and independent float64 moment/factor
reference. Evaluate separately on healthy, concentrated and failing clouds.

## Defaults, assumptions and pre-mortem

Keep original T=5, N=64, continuous SIR d18, FP64 GPU/XLA/TF32-off diagnostic
scope and OT (1,5), 32 balancing steps, K=N. These are frozen comparators,
not proven optimal settings. The observed failure is not evidence to tune OT.
Existing relative PSD floor 1e-12 and disabled moment safety are hypotheses
to inspect, not guarantees of positive factors or harmless trajectories.
Relative regularization already changes covariance: do not call it a purely
fail-closed guard. Any additional damping, fallback or ridge is Class C and
requires a dedicated non-harm evaluation and explicit change-of-target note.
Failure risks: diagnosing only a downstream NaN, comparing different inputs,
repairing a diagnostic fork rather than the consumer, and hiding rank collapse
behind a finite ridge. Capture intermediate matrices before choosing a repair.

## Resources and artifacts

Python: /home/chakwong/anaconda3/envs/tftwogpu/bin/python. CPU-only unit/reference
checks use CUDA_VISIBLE_DEVICES=-1; GPU commands require escalation and
TF_FORCE_GPU_ALLOW_GROWTH=true with verified growth before initialization.
Artifacts: docs/plans/artifacts/ledh-moment-restoration-repair-20261020, unique
attempt directories, full logs, commands, git and source hashes, seeds, device
settings, wall times and result notes. Initial stage cap: 7200 GPU-process wall
seconds, at most 8 launches; each early localization capped at 600 seconds.
Inherited allowance before this stage: 27261.957119700 seconds. No external
compute, dependencies, commits or pushes. Record charges after each launch.

## Skeptical review before diagnosis

PASS for bounded localization only. The failed unguided baseline invalidates
the earlier efficacy comparison, not the supervised loss identity. The exact
operation remains unproved; the existing post-cap affine restore is a suspect,
not a conclusion. Original runner was subsequently edited for reporting only;
its archived source map must be respected rather than falsely claiming all
current files match. This plan does not authorize a numerical candidate until
its derivation and non-harm checks are recorded below.

## Isolated root cause and reviewed repair

diagnostic-01 and diagnostic-02 reproduce the first failure at time 4. The
largest weight rounds to one (all 64 weights remain positive); source
covariance trace is 4.771005654e-29, while the mean reaches 433.68. The capped
cloud has covariance trace 2.349868257e-29 in offsets from that mean. Adding
the mean rounds every particle to the same stored vector. The next affine
restoration recomputes covariance from those rounded positions: exactly zero.
The existing ridge delta*trace(C)/d is also zero; the current-cloud Cholesky
is the first nonfinite operation. Its NaNs spread through the analytical
tangent and returned states. OT and the guide optimizer did not cause that
factorization defect. Concentrated weights and infeasible high moments remain
scientific limitations even after the numerical defect is fixed.

Candidate: retain offsets before adding the target mean, and feed those
offsets to the SAME affine restoration. For P_i=mu+O_i, in exact arithmetic
P_i-mean(P)=O_i-mean(O). Hence C(P)=C(O) and
d(P_i-mean(P))=dO_i-mean(dO). The existing target factor, current covariance
ridge delta*trace(C)/d, its derivative, triangular solves and final addition
of the weighted source mean are unchanged. This is a stable evaluation of
the same real-arithmetic regularized map, not a larger ridge or a fallback
distribution. The optional offset argument has an explicit tangent partner.
All consumer lanes still call the one shared implementation.

Skeptical review: PASS for implementation and bounded validation. A new
absolute floor would change the target and cannot restore a lost cloud; it
is not this repair. Reassociation can change low bits and later nonlinear
trajectories, so validate the local map before interpreting whole-filter
differences. Final materialized positions can still lose sub-ULP spread;
retain restoration residuals and require explicit reporting of undefined
shape diagnostics instead of calling finite scores proof of moment fidelity.
Primary healthy tolerance: scaled error <=1e-10 in well-conditioned FP64
fixtures; ill-conditioned replay differences are reported individually.
Analytical JVP checks use centered finite differences at multiple steps and
independent autodiff only as a diagnostic, never a claim-bearing score path.
Check original failing cloud, first three pre-failure clouds, translations
that cross floating-point resolution, and shared batch-to-single parity.
Replay all three score directions through the full GPU endpoint and compare
a separate healthy sequence. At most four candidate/retest launches remain
after these two diagnostic launches. Diagnosis charged 45.423792154 seconds
plus the wall time in diagnostic-02/result.json; no training was repeated.

Validation command: GPU environment above, python -m
bayesfilter.score_study.moment_restoration_validation_diagnostic --diagnosis
docs/plans/artifacts/ledh-moment-restoration-repair-20261020/diagnostic-02
--source docs/plans/artifacts/ledh-moment-training-20261020/campaign-01
--output docs/plans/artifacts/ledh-moment-restoration-repair-20261020/validation-01.
The second endpoint is the previously valid dataset 262023001 / particle
274000002, a numerical comparator only. Endpoint finite differences use both
1e-4 and 1e-5 with normalized error <=1e-3 in every parameter, with independent
initial-cloud dependence checked explicitly. Failure blocks score-parity
claims and triggers localization; it cannot be excused by a finite endpoint.

## Validation-01 result and final diagnostic review

The exact saved failure now yields finite state/tangent/value/score, and all
12 full-endpoint central-difference checks pass (largest normalized error
7.54e-5, below the predeclared 1e-3 limit). Healthy local changes remain
below 1e-10, including the ill-conditioned third step. This fixes the
identified Cholesky defect, but the stronger all-trace-finite check FAILED.
At time4 final positions still round to one vector, so their standardized
third/fourth moments are undefined. No value for those moments is fabricated.
The final observability patch explicitly emits materialized_shape_available
and retains both target and materialized covariance traces. The tiny absolute
residual is not evidence of relative covariance restoration: variance loss
is 100 percent in this exceptional cloud. This remains a scientific-promotion
veto; it is not an authorization to drop a diagnostic or resume efficacy.

Final review PASS for the narrow numerical repair and availability reporting,
not for complete moment fidelity. Validation-02 repeats the same protocol
after these reporting changes, checks the exact diagnostic flag reaches the
consumer, and leaves the stronger trace-finiteness verdict unchanged if the
mathematical moments are still undefined. The fix adds neither a new ridge
nor synthetic spread. A persistent centered cloud representation across the
whole filter would be a separate numerical scope, not silently added here.

## Completed bounded execution

All four planned stages completed. Final regression: 100 tests pass.
Validation-02 confirms finite endpoints, local non-harm and all 12 derivative
checks (maximum normalized error 7.53520e-5). The stronger trace-finiteness
criterion remains FAIL; the time4 cloud loses all materialized variance.
The explicit availability flag reaches the consumer. This is not a waiver
of moment fidelity or permission to resume the interrupted efficacy study.

Terminal self-review accepts only the narrow factorization repair, with no
external review claimed. See [result and decision tables](artifacts/ledh-moment-restoration-repair-20261020/result.md).
Four GPU launches charged 251.675993930 seconds. Stage remaining
6948.324006070 seconds; inherited campaign remaining 27010.281125770.
Next scientific action: review a whole-filter representation repair before
refreshing scope-specific tuning and an independent efficacy comparison.
