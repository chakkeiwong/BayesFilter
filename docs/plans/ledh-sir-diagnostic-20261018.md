# Fresh SIR score diagnostic, 18 October 2026

Scope correction after owner feedback, 18 October 2026: this plan answered
the wrong question. The owner fixed OT at (1,5) and asked about SIR score
accuracy of the proposed LEDH improvements against the Zhao–Cui benchmark.
The agent introduced the five-versus-40 comparison. Preserve the plan and
results as a closed diagnostic, not completion of the requested SIR test;
do not extend its iteration comparison. The active checkpoint records the
corrected next action.

## Question and evidence contract

Does the current shared covariance-mixture LEDH consumer, at the owner's
requested OT setting `(epsilon, iterations) = (1, 5)`, reduce score error
relative to the same consumer with 40 iterations? Compare both with the saved
October Zhao–Cui numerical reference. This tests the existing consumer, not
SIR-fitted cSMC/KDM guidance: that fitting adapter is not implemented.

Use the LEDH worktree `ledh-iapf-execution-20260922`, branch
`ledh-refactor-with-policy-fix`, commit `ea1018de1`, preserving dirty changes.
The callable is `covariance_proposal_tf.make_filter`, with its analytical
directional score recurrence. This optional finite-program consumer is not an
admitted canonical LEDH route. Do not label this run canonical, tuned, HMC-ready,
source-faithful replication, or evidence of cSMC/KDM improvement.

The target is `canonical_sir_d18_x0_then_transition_observe_v1`, theta zero,
18 states, 9 observations and 3 score parameters. Reuse the independently saved
T=50 data sets 26100831 and 26100832, not prior LEDH outputs. Use N=1008, exact
chunk K=N, GPU/XLA float64 with TF32 disabled as an explicit numerical-reference
exception to the FP32/TF32 production direction. All arms use flow_steps=16,
correction_steps=0 and epsilon=1.

Arms, with beta ordered prior/global/local:

| Arm | Beta | Reset iterations | Rationale |
|---|---|---|---|
| requested5 | (0.1, 0.04913405758312983, 0.8508659424168701) | 5 | Requested diagnostic setting |
| baseline40 | same | 40 | Matched frozen-scope comparator, not a broken legacy baseline |
| prior-map | (1, 0, 0) | 5 | No observational proposal map; retains the same OT reset |
| defensive-global | (0.1, 0.9, 0) | 5 | Simple globally conditioned proposal |
| defensive-local | (0.1, 0, 0.9) | 5 | Simple locally conditioned proposal |
| equal-mixture | (1/3, 1/3, 1/3) | 5 | Agnostic equal density mixture |

The last four arms construct the heuristic adversary set. Evaluate each
conditionally on data set and score coordinate; do not pool away difficult
coordinates. The prior-map arm is not a conventional resampling bootstrap PF.

Reference: `ledh-independent-validation-20261008-01/report-20261009-011148/`
`comparison.json`, each SIR data set's `reference.zhao_rank20`. This is a saved
two-fit, rank-20 importance/quadratic numerical reference, not an exact oracle.
Its between-fit standard errors do not bound rank, support or approximation
bias, or replace conditional Monte Carlo uncertainty. No author algorithm or
source route is changed in this diagnostic.

## Research intent and diagnostic roles

- Mechanism under test: the effect of fewer finite Sinkhorn iterations, with
  incoming algorithm, mixture weights and all other controls held fixed.
- Primary comparison: paired per-run L2 score error against the fixed reference,
  requested5 minus baseline40, separately for each data set.
- Promotion criterion: negative simultaneous paired-bootstrap interval in both
  data sets, plus no heuristic veto. Passing would support only this bounded,
  approximate-reference comparison; default admission still remains excluded.
- Promotion veto: requested5 has larger observed coordinate RMSE or mean L2
  error than any simple adversary in either data set. Report such losses even
  when the stochastic ranking itself is unresolved.
- Continuation veto: non-finite values/scores, invalid maps/reset, incomplete
  horizon, direction-dependent primal value, or failed analytical/FD parity.
- Repair trigger: localized harness/infrastructure failure. Repair and retry
  within the same budget in a fresh attempt directory; do not relax numerics
  or retune on these test data.
- Explanatory only: likelihood error, ESS, weight concentration, Sinkhorn mass
  errors, displacement, moment residuals, roundoff warnings, runtime and
  parameter-wise signed errors/MCSE/RMSE.
- Candidate rejection is not rejection of future guidance. No result here can
  establish true-score unbiasedness, exact-oracle agreement, source fidelity,
  cross-model transfer, canonical admission or cSMC/KDM effectiveness.

Use eight fresh paired particle seeds per data set: 261018301–261018308 and
261018401–261018408, respectively. Freeze every control before execution. Report
mean scores with replication MCSE, per-coordinate RMSE and paired L2 contrasts.
Use 20,000 paired bootstrap resamples with fixed analysis seed 26101899 and
Bonferroni 99.5% percentile intervals for the ten planned data-set-level L2
contrasts (two primary plus eight heuristic contrasts). These intervals are
conditional on the saved approximate reference and exploratory with eight
replications; do not rank from point estimates alone. Coordinate comparisons
remain descriptive veto diagnostics, not supported stochastic rankings.

## Defaults and assumptions audit

| Choice/provenance | Status and justification | Failure mode / earliest check |
|---|---|---|
| Owner's (1,5) | Untuned diagnostic hypothesis | Poor transport; record mass residuals and score error |
| Frozen October beta and flow16/reset40 | Matched comparator hypothesis, not new tuning | Stale or scope-specific setting; compare constructed simple arms |
| No correction, radius/rate inactive | Frozen comparator to isolate iteration count; not a safety endorsement | Invalid moments; retain current fail-closed relative guards and traces |
| T50/N1008 and two saved data sets | Matches available reference/data; bounded diagnostic | Narrow target coverage; report per data set/coordinate only |
| FP64 GPU/XLA, no TF32 | Explicit reference exception | Compile/resource mismatch; trusted GPU probe and tiny smoke first |
| Eight seeds and bootstrap | Convenience budget, not confirmatory power | Wide/unstable intervals; no superiority claim without support |
| Approximate Zhao reference | Existing fixed comparator, not ground truth | Unknown approximation bias; preserve source/uncertainty and qualify all errors |

## Execution, budget and skeptical audit

New campaign, separate from the closed KSC campaign: at most 1,200 GPU worker
seconds total, three launches, and 180 CPU analysis/reference seconds. A trusted
device/memory-growth probe precedes the worker. Output root:
`docs/plans/artifacts/ledh-sir-diagnostic-20261018/`; never overwrite an attempt.
Stop scheduling rows before the time budget, save partial evidence, and report
an incomplete design rather than silently promote it. Resource failure may be
repaired without changing the target, methods, data or budget.

1. Run the existing small CPU-only SIR analytical-recursion regression, with
   GPU devices intentionally hidden, and save its log.
2. Escalated NVIDIA and TensorFlow memory-growth/device probe.
3. GPU/XLA N32/T2 smoke at requested controls, all three analytical directions,
   central FD steps 1e-4 and 5e-5. Require final FD error <=
   2e-4 * max(1, abs(score)); require all perturbed runs valid.
4. Run the frozen T50/N1008 paired design through the shared implementation.
   Preserve every row and trace, source/data hashes and device provenance.
5. Analyze with standard-library diagnostic code; write result, decision and
   inference-status tables, then refresh the concise active checkpoint.

Commands: `CUDA_VISIBLE_DEVICES=-1 <tftwogpu-python> -m pytest -q
tests/highdim/test_covariance_proposal.py -k
'complete_recursive_score_all_parameters and sir_d18'`; then
`<tftwogpu-python> docs/benchmarks/run_ledh_sir_diagnostic.py run --output
docs/plans/artifacts/ledh-sir-diagnostic-20261018/attempt-01`; then the same
script's `analyze --output <attempt-path>` command. GPU commands are escalated;
set `TF_FORCE_GPU_ALLOW_GROWTH=true` before TensorFlow import. The worker
requires and verifies GPU memory growth before logical initialization.

Pre-mortem: an apparent gain could come from the approximate reference, seed
luck or averaging opposite coordinate errors; the saved-reference disclosure,
paired intervals and coordinate RMSE expose but cannot eliminate these risks.
An apparent failure could instead be a derivative or invalid-reset bug; the
small regression/FD smoke and full validity traces distinguish those cases.

Skeptical audit before execution: PASS for a bounded diagnostic, not admission.
The baseline is the same current callable rather than a known-broken historical
lane; score error, not ESS or transport residual, is the comparison metric.
Default transfer, reference uncertainty, environment exception, repair/stop
conditions and absence of SIR-fitted guidance are explicit. A successful command
answers the consumer/iteration question only, not the full augmentation question.

Attempt 01 infrastructure repair: the N32/T2 GPU analytical/FD check passed
(maximum absolute difference 2.46e-6), then the native worker exited 139 before
any full row. WSL recorded a null-pointer fault in `libcuda.so.1.1`; no numerical
candidate verdict follows. Its manifest remains preserved as written before
the native crash. Charge a conservative 180 GPU-worker seconds (exact native
exit timing was not captured), leaving 1,020 seconds and two launches. Retry
on the same 4080 SUPER identified by UUID, hiding the second GPU; enable Python
fault traces and pre-row stage logging. A supervising process will retain exit
code and elapsed time even on native failure. This changes device visibility
and observability only; data, methods, precision, XLA, criteria and seeds stay
frozen. Skeptical repair audit PASS: the same scientific question is answered,
and no claim can be promoted from the tiny derivative check alone.

Attempt 02 infrastructure result: pinning the 4080 SUPER did not repair the
failure. The small GPU/XLA derivative smoke passed again, then the first
T50/N1008 requested5 evaluation terminated with SIGSEGV (-11), producing no
full rows. The external supervisor measured 47.42244011495495 seconds. There
is no numerical candidate verdict. The cause remains unconfirmed; no OOM was
observed. The remaining campaign budget is 972.577559885045 GPU-worker seconds
and one launch, plus the unused CPU analysis allowance.

Final localized repair hypothesis: disable CUDA command buffers using the
process-local `XLA_FLAGS=--xla_gpu_enable_command_buffer=` while retaining XLA,
FP64, the same GPU and all scientific controls. OpenXLA's upstream
`xla/debug_options_flags.cc`, `SetterForRepeatedEnum`, explicitly clears the
enabled commands for an empty string; the command-buffer scheduling header
identifies this list as the enabled command types. Source inspected at
https://github.com/openxla/xla/blob/main/xla/debug_options_flags.cc and saved at
`/tmp/sir-xla-debug-options-flags.cc`. This is an infrastructure hypothesis,
not an established crash diagnosis. Record the flag in both manifests and the
supervisor, run the same derivative smoke under it, and use GDB to preserve a
bounded native backtrace if the final attempt fails. No package or environment
installation is authorized or needed. Fresh output: `attempt-03`.

Skeptical repair audit PASS for one bounded diagnostic retry: neither the
reference nor data, seeds, score implementation, numerical precision, OT
controls or comparison criteria change. Disabling command recording may
affect execution speed, so this attempt cannot establish default runtime
performance. Small-smoke success alone is not full-SIR evidence. If the last
launch fails before full rows, report infrastructure failure and unmeasured
score improvement; do not manufacture results or launch a fourth worker.

## Terminal result, 18 October 2026

Attempt 03 exited normally after 26/96 planned arm evaluations, with three
primary pairs on data 26100831 and two on 26100832. Mean per-run L2 score errors
for requested five versus 40 iterations were 58.1107 versus 72.8357 and
70.8590 versus 77.6318. These are descriptive differences, not statistically
supported improvement; the eight-pair design was not completed and no
confidence intervals are issued. On the two common pairs for data 26100832,
the requested arm also has higher observed error than prior-map (65.2092),
defensive-local (69.9671) and equal-mixture (45.7755). No promotion follows.

All completed rows passed the recorded validity checks. Moment errors of
about 1e-13 are roundoff, not a blocker. Large finite global-arm scores remain
an accuracy problem. No SIR-fitted controlled-SMC/KDM guide was evaluated.
The Zhao–Cui comparator remains an approximate numerical reference.

Attempt 03 consumed 913.0771416550269 supervisor seconds; total conservative
GPU-worker charge is 1140.4995817699818/1200 seconds, leaving 59.5004182300182
seconds and zero launches. Worker wall time was 911.2626162549714 seconds;
CPU-only analysis took 0.006994791037868708 seconds. The GDB wrapper returned
1 after `bt` found no stack following normal worker exit; this is not a
worker crash. No further launch or scope expansion is authorized by this
diagnostic. Disabling command buffers does not establish the crash cause.

The terminal result, decision/inference tables, repair ledger and exact
provenance are in
[result.md](artifacts/ledh-sir-diagnostic-20261018/result.md).
The runner's post-run edit changes analysis completeness/promotion reporting
only; the original executed source hash remains in the run manifest. No
canonical, default, HMC or augmentation claim is made. This requested SIR
diagnostic is closed; a larger fitted-guide experiment requires its own
SIR-specific scientific plan and bounded budget.
