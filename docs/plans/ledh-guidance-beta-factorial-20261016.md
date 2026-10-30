# Future guidance versus mixture selection: bounded diagnostic

## Question and research intent

Does an exact frozen Gaussian future message change model-score error when
the proposal-mixture probabilities are held fixed, and does separately
calibrating those probabilities change that effect? The previous completed
comparison confounded the two interventions. Exact and fitted messages gave
scores within 1.24e-11 there, so more fitting capacity is not the next test.
This stage tests the existing optional covariance-guided density mixture,
not canonical Contract-E/UKF LEDH admission or trained KDM.

## Design and evidence contract

- Use the same shared TensorFlow analytical-score consumer, N=32, T=5,
  two-dimensional diagonal Gaussian AR model; alpha=0.2 or 0.9, each ordinary
  or with a three-marginal-SD impulse at observation 2 in coordinate 0.
- Compare current-only and exact-future guidance at each of five frozen beta
  policies: defensive global (0.1,0.9,0), defensive local (0.1,0,0.9), equal,
  independently calibrated current-only, and independently calibrated future.
  Add physical identity (1,0,0) as the eleventh arm. Cross both calibrated
  betas with both guidance states; do not recalibrate on evaluation data.
- Constructed heuristic adversaries: identity (unmodified predictive draw),
  defensive global (pooled covariance), defensive local (ancestor-conditional
  covariance), equal (no optimizer). Evaluate separately in all four regimes.
- Generate fresh seeds 217000+1000*condition+100*partition+index, with 2
  calibration, 2 validation and 12 final datasets per regime. Preserve the
  existing independent random streams and pair evaluation draws across arms.
  Validation is a validity check, not a selection or tuning partition.
- The certifying comparator for score error is the exact Kalman model score,
  expressed in drift0, drift1, log-process-SD, log-observation-SD coordinates.
  Frozen guide coefficients have zero derivative, as in the existing consumer;
  this is not differentiation through a refitted message.
- Primary diagnostic contrast: future-global minus current-global squared
  error in log-process-SD, separately by regime. Report a paired dataset
  bootstrap percentile 95% interval (10000 resamples, fixed analysis seeds).
  No multiplicity-adjusted or universal superiority claim is planned.
- Additional prespecified contrasts: guidance at every other beta; beta
  calibration changes within each guidance state; the interaction between
  guide state and current/future-calibrated beta. Report all four coordinates.
- Promotion veto: any descriptive primary-coordinate loss to a constructed
  heuristic blocks nomination; an interval crossing zero cannot support a
  ranking. A possible nomination requires all four primary intervals below
  zero and no heuristic loss. Even that only nominates longer validation.
- Continuation veto: nonfinite outputs, invalid density/factorization/caps or
  material moment failure, incomplete/mispaired rows, wrong oracle or scope,
  changed frozen beta within a paired contrast, or budget exhaustion.
- Repair trigger: localized harness failure may be repaired within budget;
  negative score results trigger mechanism localization, not rejection of the
  research direction. Tiny historical moment-cutoff warnings are explanatory.
- ESS, weight concentration, likelihood error, moment residuals, beta values
  and runtime are explanatory, never substitutes for score accuracy.
- No canonical admission, default change, nonlinear transfer, tuned optimality,
  HMC readiness, learned-guide quality, bias/variance decomposition or strong
  stochastic ranking follows from this small diagnostic.

## Defaults, assumptions and early checks

| Choice / provenance | Justification and status | Failure mode / earliest check |
|---|---|---|
| epsilon=1, Sinkhorn=5; owner development setting | Frozen comparison baseline, not a tuned default | Transport/reset invalidity; existing validity traces |
| 16 flow / 2 correction steps; prior shared consumer | Hold mechanism fixed, hypothesis only | Score error may reflect discretization; no generality claim |
| FP64, TF32 off, GPU/XLA | Explicit reference diagnostic, isolates mechanisms from low precision | Wrong device/backend; trusted GPU and manifest check |
| Existing cap/ridge/reset protections unchanged | No numerics-altering intervention in this stage | Factorization or material-moment failure; fail closed |
| 2 calibration datasets, beta floor 0.1, solver tolerance 1e-7 | Existing pilot objective, not a score-optimal policy | Calibration instability; report betas/gap and cross policies |
| Exact Gaussian message, frozen during score propagation | Positive control, removes fit error for this scoped model | Wrong interface or terminal message; existing parity tests |
| 12 final datasets per regime | Bounded mechanism diagnostic, not confirmatory power | Wide/unstable intervals; report uncertainty and no ranking |
| Current-only/future paired at fixed beta | Separates intervention from beta calibration | Pairing/beta mismatch; executable analysis rejection |

## Skeptical audit and pre-mortem

PASS for a bounded mechanism diagnostic: the baseline is the same consumer
and exact model oracle, not a finite-program derivative surrogate. Every
guidance comparison holds beta and random inputs fixed. Every optimized arm
faces four constructed heuristics separately in ordinary and persistent/stress
regimes. Fresh seeds avoid selecting on the previous final sample. Expected
failures are poor proposal geometry or a likelihood-variance objective not
improving score error; neither is a continuation veto. Cheap tests verify arm
construction, crossed beta reuse, missing/invalid rows and paired contrasts.
Passing while misleading remains possible with twelve clusters or this easy
Gaussian family; no default or cross-model promotion is allowed.

## Execution and budget

Worktree: `.localresources/worktrees/ledh-iapf-execution-20260922`, branch
`ledh-refactor-with-policy-fix`, starting commit `ea1018de1`. Preserve dirty work.
The owner's current request authorizes this next bounded master-program stage.
New stage cap: 900 aggregate GPU wall seconds, 180 CPU diagnostic seconds,
at most two GPU attempts. Record attempt times and remaining budget. No
environment changes, external agents, commits, pushes or destructive actions.

Environment: `/home/chakwong/anaconda3/envs/tftwogpu/bin/python`, `PYTHONPATH=.`,
`TF_FORCE_GPU_ALLOW_GROWTH=true`; trusted/escalated GPU with verified memory
growth before initialization. Explicitly hide GPU for focused CPU tests.
Shared-device timings are descriptive, not speed comparisons.

Command: `python scripts/run_ledh_guided_mixture_score.py --guidance-factorial
--output docs/plans/artifacts/ledh-guidance-beta-factorial-20261016/attempt01
--budget-seconds 780`. The supervisor imposes the remaining wall-time cap.
Analyze using `scripts/analyze_ledh_guidance_factorial.py`; save full logs,
source hashes, run manifest, calibration probabilities, per-row diagnostics,
decision JSON, result note and concise active checkpoint in this artifact root.

## Next decision

Attempt01 stopped after154 valid rows: current-only calibration in condition1
did not meet the solver certificate. No score ranking is permitted from this
incomplete run. Preserve it. Charge151.485 GPU seconds (the larger of the
internal151.484 and supervisor139.79 clocks). Before retry, reproduce only
that independent calibration on CPU, save objective/gap/iteration count and
inputs, and test whether a higher iteration cap satisfies the unchanged
1e-7 relative certificate. This is a solver resource diagnostic, not changing
the objective, tolerance, data, heuristic or promotion criterion. Budget the
CPU diagnostic at100 seconds; it fits the remaining149.4 CPU seconds. The
runner now preserves failed calibration diagnostics instead of discarding them.

The CPU/XLA reproduction took14.11 seconds. The original2000-step cap stops
at gap4.4003e-6; at4022 steps the same algorithm reaches gap1.01755e-7 below
the unchanged required1.01862e-7. This identifies an iteration-budget failure,
not invalid density/score math. Retry the same data and design with the
existing solver's cap20000, exposed as `--beta-max-steps 20000`; this does not
change its objective, update rule, tolerance or early exit. Leave the shared
solver default2000 unchanged. Attempt02 uses `--budget-seconds 700`, external
timeout740, fresh output `attempt02`; remaining GPU allowance748.515 seconds.
The failed first attempt remains incomplete evidence, not a candidate ranking.

Terminal-preparation math/code check: the manuscript's
`prop:bg-mixture-guided-score` fixes guide coefficients and beta, matching this
consumer; its preceding paragraph explicitly warns that conditional
weight-variance minimization need not reduce model-score error. The crossed
calibration policies here are deliberate interventions, not claims that one
guide's selected beta is tuned for another guide. The unchanged shared
consumer call-chain/derivative tests and two additional beta simplex/gap and
convexity tests pass. No manuscript theorem or numerical default changes.

If a fixed-beta guide helps but its calibrated arm does not, investigate the
calibration objective/contexts before adding KDM capacity. If fixed-beta
guidance also lacks evidence of benefit, inspect proposal concentration and
finite-N score error before nonlinear transfer. Do not silently infer which
cause dominates from non-significant differences. Update the master program
after the completed stage; a failed candidate is not a failed research direction.

## Completed stage

Attempt02 completed all 616 evaluations across 11 arms and four regimes.
All 154 accepted attempt01 score vectors and likelihood errors replay bitwise;
23 focused tests pass. The larger cap reaches the unchanged calibration
certificate at 4022 iterations on CPU and GPU. All final evaluations are valid,
with one retained roundoff warning and maximum covariance residual 9.411910e-15.
No numerical continuation veto remains.

The [result](artifacts/ledh-guidance-beta-factorial-20261016/result.md) and
[decision](artifacts/ledh-guidance-beta-factorial-20261016/decision.json)
preserve every comparison. All 20 fixed-beta guidance intervals for
log-process-SD error include zero. Every guided arm has at least one descriptive
loss to a heuristic, so none is nominated for promotion. Stress-case beta
effects and particle concentration motivate the planned finite-N localization;
they do not establish overall inferiority or reject backward guidance.

Conservative consumption is 729/900 GPU seconds and 54/180 CPU seconds;
both GPU attempts have finished. The next stage needs a bounded plan for
repeated particle clouds at fresh fixed datasets, followed by particle-count
localization with independent calibration. Preserve these final datasets as
evaluation evidence, and keep nonlinear/KDM expansion pending that diagnosis.
