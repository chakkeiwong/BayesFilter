# Guided mixture score comparison — 16 October 2026

## Active amendment: proportionate moment checks

The owner's correction supersedes the terminal reset-repair instruction below.
Rejecting an FP64 relative covariance residual of about 2e-13 against
`100 * machine_epsilon * dimension` was disproportionate. That cutoff had no
checked forward-error bound or connection to the score accuracy sought here.
Finite scores, successful factorizations, and the saved CPU/GPU replay do not
demonstrate scientific harm. Preserve the old attempts as evidence of an
overly strict guard; do not repair the numerical algorithm merely to satisfy it.

Separate the old roundoff warning from numerical invalidity. For this optional
mixture implementation, use a dimensionless moment tolerance
`sqrt(machine_epsilon)`: about 1.49e-8 in FP64. This retains half the available
significand bits (26 of 52 in FP64) and rejects loss of that precision in the
restored moments. It is an explicit engineering accuracy budget, not a theorem
bounding score error or an empirically tuned score threshold. Normalize mean
error by the larger of the target mean magnitude and the largest marginal
standard deviation; covariance error already uses a relative Frobenius norm.
This scaling avoids a units-dependent absolute floor. Retain the old cutoff
and its warning in diagnostics. Nonfinite values, rank/factorization failure,
correction-cap violations, and moment error above the new budget remain vetoes.
No particle, derivative, OT setting, guide, beta objective, or correction changes.

Skeptical audit: the flaw was upgrading a roundoff warning into a continuation
veto. The new budget is a provisional numerical screen for this diagnostic,
not canonical/default admission or proof that small moment errors cannot be
amplified downstream. First check the actual consumer with injected small and
material moment errors, then replay the saved failure. Repeat the same frozen
seven-arm experiment under the revised guard and compare previously accepted
rows for non-harm. Do not change model settings or select on score errors.
Because the validity rule changed after a final case was inspected, classify
this completion as exploratory; it cannot be called untouched confirmation.

Use fresh `cpu07/`, `cpu08/`, and `attempt02/` directories under the existing
artifact root. At most one further GPU campaign, bounded by 600 wall seconds,
and 80 further CPU validation seconds: prior charges 252 GPU / 96 CPU seconds
leave these within the original 900 / 180 second totals. The earlier two-launch
count is superseded by this repair allowance; total compute remains bounded.
No independent reset-stabilization campaign is warranted by the observed error.
Stop on a real numerical veto or budget exhaustion; retain every case. Record
the guard policy and maximum residuals in the manifest and terminal result.

Completed amendment:12 focused checks pass; saved-input replay in cpu09
completes after a3.157-second import-path failure preserved in cpu08. GPU
attempt02 completes392 rows in411.314 seconds, with all336 prior accepted
results bitwise unchanged. Only the original stress identity case emits the
old-cutoff warning. CPU analysis uses21.70 seconds. Conservative cumulative
charges156/180 CPU and664/900 GPU seconds. The fitted-guide nomination fails;
fitted-minus-current/global intervals include zero in every condition. See
[completed result and next discriminating question](artifacts/ledh-guided-mixture-score-20261016/proportionality-result.md).

## Question and authorization

The owner asks to refresh the master and continue. Test whether replicated
controlled-SMC future messages help model-score estimation through the existing
identity/global/local density-mixture consumer, rather than the earlier
single-flow consumer. This is a bounded FP64 GPU/XLA diagnostic, not canonical
LEDH admission, per-scope tuning, KDM training, or default promotion. Preserve
the existing dirty tree on `ledh-refactor-with-policy-fix`.

## Evidence contract and skeptical audit

Use `LGSSMSpec('diagonal_ar', 2)`, raw prior N(0,I), predict before every
observation, process SD .3 and observation SD .5. Independent coordinates
permit the existing scalar cSMC fitter and exact future-information recursion
to be applied coordinatewise without changing their statistical model. Drift
is .2 or .9 in both coordinates. Cross each with unshifted observations and
an impulse of three marginal observation SDs at the middle observation in
coordinate zero. The latter is an explicit stress case, not ordinary model
sampling. Horizon5, particles32, flow16, reset epsilon1/steps5, correction2.

Seven arms share the same sampler, full mixture density, finite OT reset,
moment correction, and analytical recursion:

| Arm | Rationale |
| --- | --- |
| identity (1,0,0) | Physical predictive/bootstrap proposal |
| defensive global (.1,.9,0) | Simple global observation bridge |
| defensive local (.1,0,.9) | Simple conditional-covariance bridge |
| equal mixture | Diversification without optimization |
| calibrated current-only | Same consumer, no future information |
| calibrated fitted future | Replicated cSMC fitter, frozen future-only message |
| calibrated exact future | Gaussian positive control, not a learned candidate |

Fit guides using independent64-particle pilots at frozen nominal parameters.
Reuse the observation sequence in its own future message, as the proposal
requires, but never reuse evaluation particles as fitting particles. Fit beta
on two calibration sequences per condition using independent pilot draws
from the equal mixture and contexts produced by the same guide/consumer.
The existing normalized second-moment objective averages conditional relative
second moments; it is not the variance of the whole likelihood or its score.
Use beta0>=.1 and solver tolerance1e-7; freeze beta before two validation
sequences and twelve untouched final sequences per condition. No selection
or retuning uses validation or final errors. Each final sequence has its own
data, fitting and evaluation random streams, paired across arms.

Primary measurement: squared error of log-process-SD score against exact
Kalman score. Report every coordinate (drifts and log observation SD included),
likelihood error, paired error differences, and fit/calibration/evaluation
cost. Dataset-cluster bootstrap intervals with fixed10000 resamples are
diagnostic uncertainty estimates; twelve clusters and four conditions cannot
establish broad superiority. Nomination requires a negative upper95% paired
interval against current-only and all four heuristics in every condition.
Losing descriptively to a heuristic is a promotion veto and the headline,
not an automatic continuation veto. Exact-guide failure is an explanatory
diagnostic separating fitting from the downstream proposal/reset.

Continuation vetoes: nonfinite/missing scores, invalid sampler/reset, failed
fitter or beta convergence, broken derivative/reference/wiring check, budget
exhaustion. Stop the affected phase; retain failures, do not drop rows or
weaken checks. Local harness faults may be repaired within budget. A candidate
losing the score comparison triggers mechanism localization, not rejection
of the research direction. Runtime, ESS and solver gap are explanatory,
except invalidity/nonconvergence as specified.

Audit verdict: PASS for a diagnostic only. The prior single-flow baseline
was wrong for this question; all seven arms now call the same mixture factory.
Exact Kalman uses the same predict-first law and parameterization. The four
simple adversaries are evaluated separately in each regime. No proxy fit loss
can promote a score claim. Full Contract-E/UKF conformance, TF32 readiness,
nonlinear transfer and trained KDM remain outside this experiment.

## Defaults and failure explanations

| Choice / provenance | Status and justification | Failure / earliest check |
| --- | --- | --- |
| (1,5) owner development OT pair | Frozen diagnostic setting, not tuned | Incomplete finite transport; retain mass/reset diagnostics |
| N32,T5,64 fit particles | Budget-limited mechanism fixture | Monte Carlo/finite-reset error; exact score and simple adversaries |
| flow16/correction2 | Existing integration fixture, hypothesis only | Discretization/correction bias; finite derivative test, oracle arm |
| beta0=.1 | Diagnostic defense: qbeta>=.1q0 bounds p/q by10 | May overconstrain optimum; report coefficients and solver gap |
| No additional damping/clipping | Preserve verified consumer and reject invalid results | No stability claim beyond fixture; existing margins fail closed |
| cSMC scalar precision projection | Replicated fitting route, not population equivalence | Compare fitted with exact message, flag fit failure |
| FP64, XLA, GPU | Explicit reference exception to TF32 production target | Record actual GPU and memory-growth policy |

Pre-mortem: a small finite-program error can coexist with a wrong model score;
fit quality and beta objective cannot answer the score question. Shared random
numbers do not eliminate Monte Carlo error. Impulse cases do not represent
population frequencies. Fixed settings may fail without invalidating the idea.

## Execution and budget

1. Add shared-consumer diagnostic wrappers for dynamic frozen coefficients,
   direction assembly and guided beta pilot; test wiring and scalar conversion.
2. CPU-only focused tests (GPU hidden), then escalated GPU preflight and one
   bounded campaign. At most two GPU attempts including localized repair,
   total900 GPU wall seconds and180 CPU validation seconds. These are new
   continuation-stage caps, not borrowed remaining repair allowance.
3. Preserve command, HEAD, source hashes, environment, actual hardware, seeds,
   timings, controls, raw per-sequence results and failure status. Artifacts
   under `artifacts/ledh-guided-mixture-score-20261016/attemptNN/`; never overwrite.
4. Write decision and inference-status tables, red-team result, update this
   master/checkpoint. If a scientific continuation veto prevents completion,
   record exactly what failed and the next smallest discriminating check.

Environment: `/home/chakwong/anaconda3/envs/tftwogpu/bin/python`. GPU command
sets TF_FORCE_GPU_ALLOW_GROWTH=true and selects a verified GPU UUID; memory
growth is configured before initialization. Test commands set
CUDA_VISIBLE_DEVICES=-1 and are explicitly CPU diagnostic exceptions.

## Progress

Shared-wrapper call-chain regression passes (CPU, GPU intentionally hidden):
all four analytical directions equal the direct consumer; pilot density and
physical target equal direct map evaluation. Log: `artifacts/ledh-guided-mixture-score-20261016/cpu01/pytest.log`.
Conservatively charge14 CPU seconds (pytest reports13.95; external timer6.61
is inconsistent, so retain both and use the larger value). GPU preflight
selects idle RTX4080SUPER UUID `GPU-68251639-fe82-8f81-3ccc-2953c32e805b`;
the RTX5080 was88% utilized and is not used. GPU preflight is charged1 second.
Attempt01 runs with840-second internal budget and880-second external timeout.

Attempt01 stopped at the declared numerical-validity veto: condition3
(persistent drift plus impulse), final seed100204, identity arm. It preserved
336 valid rows and took230.850 seconds. The grid remains incomplete; no score
ranking or bootstrap analysis is admissible. Charge232 GPU seconds including
preflight, leaving668. Six result-analysis tests also pass; total CPU charge15.

### Failure localization, same campaign budget

The smallest next question is which existing validity condition rejected this
exact seed, and whether CPU reference agrees. Preserve all controls and data.
Reproduce only the failed identity evaluation, returning all four directions'
existing traces and validity flags (pure observability). Compare GPU/XLA with
CPU graph execution; do not change guard thresholds or use invalid derivatives
as score evidence. Cap this second GPU attempt at120 seconds, CPU reference
at90 seconds within the remaining165. Stop if the guard cannot be identified
without a numerics-changing repair. Success is an identified guard and a
reproducible case, not acceptance of the failed row. A failure of this simplest
arm invalidates the complete comparison; it does not reject the guide idea.

Audit: PASS for localization. Exact data/random streams and physical parameters
are retained; no hidden retuning, replacement baseline or new claim. The
attempt01 manifest's data-stream label says1, but the shared simulator actually
uses719; code/source hashes preserve the original command. The label is fixed
for future runs and the attempt01 evidence is not overwritten.

GPU02 reproduces failure in19.345 seconds; CPU04 replays its saved inputs in
8.424 seconds. All four scores remain finite; both devices fail the moment
check at observation3. GPU covariance error2.138e-13 and CPU4.825e-14 exceed
4.441e-14. Mean error is also about2e-13. This is not a NaN or transport
column-mass failure. The traces previously discarded the already computed
pre-correction moment errors. Append those errors, the actual mean threshold,
and moment/cap flags as pure observability; no numerical operation or guard is
changed. A final CPU-only replay distinguishes reset from correction error.
GPU attempts are now2/2; no further GPU launch in this campaign. Charge252
GPU seconds,44 CPU seconds before this final bounded CPU check.

Completed: CPU06 confirms the mean/covariance exceedance is already present
before correction; CPU05 passes eight numerical/wiring checks, CPU02 six
analysis-validity checks. Final conservative totals96 CPU/252 GPU seconds;
both GPU attempts used. See [terminal result](artifacts/ledh-guided-mixture-score-20261016/result.md)
and its decision table. The comparison remains incomplete; no score ranking
is supported. Next is a separately bounded precision/stability evaluation,
not a relaxed threshold or a restart using this failed final dataset.
