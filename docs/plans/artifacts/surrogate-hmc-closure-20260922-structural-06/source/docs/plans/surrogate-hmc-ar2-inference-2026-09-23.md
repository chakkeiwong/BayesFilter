# AR2 surrogate-HMC posterior test — 2026-09-23

This is the next part of the existing closure master. The question is whether
the full canonical analytical DCTR proposal force samples the AR2 model
posterior when independent Kalman endpoint values supply the Metropolis
correction (B/R). A uses the exact Kalman gradient as a comparator. The force
is allowed to differ from that gradient. C/D and D/R require a separate finite
posterior reference and do not block this model-endpoint test. LG1 stays closed.

## Evidence and execution

The model, initial law, unbounded normal priors, observations and independent
reference are frozen in structural-03. Structural-04 selected N64/T10 controls
on calibration/validation and qualified FP32 against FP64 with the same draws.
Load the repository-issued selection and check source/data hashes before use.
Retain the full canonical structural flow, UKF lifecycle, correction and reset.

First measure four-chain complete transitions at L=1 and L=8 for A and B,
including compilation separately from five steady calls. Also measure each
force and endpoint. The cost probe uses claim posterior mean and one-SD
offsets, step .05 and inverse mass equal to reference marginal variances;
these are cost hypotheses only, not a tuning artifact or inference evidence.
Predict function-wise required effective samples using the grid variances,
the fixed 13-function margins and the empirical reference error allowance.
Report IAT=1,4,10 forecasts as hypotheses, not mixing guarantees. Select two
prospective looks from (2000,4000), (4000,8000) using IAT=4;
if the last is insufficient, label the planned precision under-budgeted.
No margins or reference errors may be changed to fit a forecast.

The shared prospective assessment must be calibrated on AR2 reference-law
functions at those exact looks before retained results are interpreted.
Its 13-function simultaneous intervals use Student-t(36), family alpha .05,
one B/R family and two looks; A/R is a descriptive comparator. Modern
parameter/function R-hat, ESS and batch-dependence checks qualify uncertainty.
Observed differences within practical margins are reported separately from
formal equivalence and do not assign owner acceptance automatically.

Use only the public tune_hmc_kernel with the repository-issued typed
position-only-force binding in raw log-scale coordinates. Both A and B need
their own mass, epsilon and trajectory tuning and fresh verification. Archive
warmup and verification separately; retain fresh samples only after admission.
The sampling schedule and exact tuning settings will be frozen in an addendum
after the cost probe, before any chains run.

## Budget and stops

Cost stage: at most 600 CPU-worker / 900 GPU seconds from the remaining
structural reservation (1578.790136 CPU / 3270.006149 GPU at entry).
Calibration and inference receive separate bounded reservations inside the
unchanged global 54000 CPU / 226800 GPU cap after cost measurement. A stage
finishing is not a stopping condition; continue the existing master when its
prerequisites and remaining global budget permit. At most three attempts per
stage; preserve every failure and charge all consumed time.

Invalid sources, law/parity failures, nonfinite current states, broken HMC
mechanics or missing evidence are continuation vetoes pending repair. Poor
acceptance, mixing or precision rejects this candidate or triggers the planned
retuning/precision repair; it does not reject surrogate HMC as a direction.
Do not rerun old chains under new criteria.

## Default audit and pre-mortem

| Choice | Provenance and reason | Failure mode / earliest diagnostic | Status |
|---|---|---|---|
| N64/T10 force | Structural-04 scoped protection and precision selection | Inaccurate force causes costly low movement; endpoint-corrected tuner checks this | Viable proposal candidate, no finite-posterior accuracy claim |
| Normal priors and x0 N(0,I) | Declared AR2 scope and direct-Gaussian/Kalman parity | Changing initial law silently changes posterior; source/data identity check | Frozen target |
| 13 functions / .1 and .02 margins | Closure roadmap practical posterior criteria | Good means hide wrong tails or predictions; conditional CDF/predictive functions | Pilot accuracy contract, not universal defaults |
| IAT4 forecast | Explicit cost-planning hypothesis | Underestimates required sampling; report IAT1/10 and realized MCSE | Hypothesis |
| t36 and batch diagnostics | Previously qualified KSC candidate, shared implementation | Transfer fails on AR2; fresh AR2 reference-law calibration | Unqualified until scoped test |
| .05 step / reference diagonal geometry in cost probe | Conservative local integration and actual parameter scales | Nonlocal proposals cost more or fail; complete L1/L8 status telemetry | Cost hypothesis only |

Cheap adversaries are the direct Gaussian observation calculation (likelihood
law), Kalman evaluation (same exact likelihood), exact-gradient HMC (sampler),
and independent grid draws (known-law diagnostic). Check central, marginal
10/50/90% regions, second moments and predictive probabilities separately.
Any observed efficiency loss against exact Kalman HMC must be reported, and
prevents an acceleration claim. This simple model can establish proposal-force
viability without demonstrating a speed advantage. The finite-target error,
all-model validity, default readiness and statistically supported method ranking
remain separate questions.

Skeptical pre-run audit: correct comparator is the declared model posterior;
force/reference score agreement and raw log-likelihood offsets are explanatory
only. Full-transition costs include endpoint work. Fixed paths, source checks,
independent reference and actual typed consumer tests guard against a passing
but wrong-target run. The principal remaining risks are underestimated mixing
cost and unqualified uncertainty transfer; the next two stages test those risks.

Commands use the tftwogpu environment, escalated GPU access, memory growth,
FP32/TF32 canonical force and FP64 reference/HMC coordinates, XLA kernels.
Entry: scripts/run_surrogate_hmc_program.py --stage closure --closure-stages
ar2-cost. Fresh artifact prefix: surrogate-hmc-closure-20260922-structural-NN.
All worker logs, tensors, source snapshot and accounting live under that root.

Cost attempt structural-05: a new fail-closed test used 80 draws, outside the
diagnostic's supported sizes. Nineteen other checks passed; GPU cost work was
correctly blocked. Repair uses the supported 1000-draw fixture. The forecast
now nominates only schedules the shared diagnostic supports. Charged 6.691610
CPU seconds, zero GPU; retry remains inside the unchanged structural budget.
