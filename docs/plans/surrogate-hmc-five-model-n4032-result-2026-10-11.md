# Five-model surrogate-HMC master result, 2026-10-11

The master program was reviewed and executed with ordinary FP32, TF32
disabled, N=4032, and XLA on the repository GPU path. The requested horizons
are LGSSM, KSC SV, predator-prey and SIR at T=50; the saved range-bearing
fixture is T=20. The result is a completed endpoint/filter-preparation phase
and a bounded mechanics/cost phase. **No model completed adaptive warmup or
retained posterior sampling.**

The scientific question for the full campaign is whether the analytical
covariance-proposal score can act as a deterministic surrogate force inside
Metropolis-corrected HMC while an independently defined endpoint likelihood
and prior determine the MH potential. In this run the primary mechanics
criterion was finite endpoint and force evaluation with no force fallback or
divergence. R-hat, ESS, MCSE, posterior-reference agreement, and efficiency
were deliberately not used because the two-transition pilot is too short.

## Provenance and execution

- Branch: ledh-graceful-failure-execution-20260917
- Git HEAD at launch: 10884528ca8b1a7463895256d30b9655d69c0955
- Preparation command: scripts/run_surrogate_hmc_five_model_master.py campaign --through filter-tuning --budget-seconds 2300 --gpu 1 --output docs/plans/artifacts/surrogate-hmc-five-model-n4032-20261011-06
- Mechanics command: scripts/run_surrogate_hmc_five_model_master.py campaign --through mechanics --priors benchmark-normal-v1 --prepared-root docs/plans/artifacts/surrogate-hmc-five-model-n4032-20261011-06 --budget-seconds 1200 --gpu 1 --output docs/plans/artifacts/surrogate-hmc-five-model-n4032-20261011-07
- Preparation wall time: 928.79 s. Mechanics wall budget: 1200 s; the master stopped at 1200.19 s.
- TensorFlow memory growth was configured before device initialization. The worker manifest records an NVIDIA GeForce RTX 5080, GPU/XLA, float32 filter tensors, and tf32=false. The transport policy selected equal 2016 row/column chunks for N=4032.
- CPU review after the code repairs: 28 tests passed in 17.78 s with two dependency deprecation warnings. The log and machine-readable record are under docs/plans/artifacts/surrogate-hmc-five-model-n4032-20261011-03/.

## Endpoint and filter preparation

The preparation artifact is docs/plans/artifacts/surrogate-hmc-five-model-n4032-20261011-06. All five
endpoint checks were finite and exactly replayed with fixed inputs; each
selection passed source, data, scope and selection-hash checks before the
mechanics continuation.

| model | independent endpoint | value at nominal point | validation loss | selected filter controls | selected beta |
|---|---|---:|---:|---|---|
| lgssm | exact_Kalman | -138.358 | 1.06785 | flow=32, reset=80, eps=1.0 | 0.1,0.219288,0.680713 |
| ksc | fixed_1201_node_grid_refinement_required | -111.975 | 1.36216 | flow=16, reset=40, eps=0.5 | 0.333333,0.333333,0.333333 |
| predator_prey | independent_fixed_bootstrap_4x4032_approximate_posterior | -267.514 | 1.02819 | flow=16, reset=40, eps=1.0 | 0.1,0.893673,0.00632727 |
| sir_d18 | independent_fixed_bootstrap_4x4032_approximate_posterior | -1717.13 | 1.17463 | flow=16, reset=40, eps=0.5 | 0.1,0.0691144,0.830886 |
| range_bearing | fixed_full_support_trajectory_importance_4x4032_approximate_posterior | 39.8541 | 1.76493 | flow=8, reset=40, eps=0.5 | 0.333333,0.333333,0.333333 |

LGSSM uses the exact Kalman endpoint. KSC uses the fixed 1201-node grid with a
1601-node refinement at the nominal point; the observed refinement difference
was approximately -2.84e-14. Predator-prey and SIR use frozen four-replicate
N=4032 bootstrap endpoints, and range bearing uses the frozen four-replicate
full-support trajectory-importance endpoint. Those nonlinear endpoints are
finite approximate posterior targets; they are not exact likelihood or
posterior oracles. For SIR, the four-replicate log-likelihood MCSE was 0.222,
so that uncertainty remains material for posterior-accuracy interpretation.
The range-bearing endpoint is finite/replay-checked but has no regional
precision reference in this phase.

The filter controls are own-scope exploratory selections. They are not
canonical LEDH admission or evidence that the filter score is the gradient of
the independent MH endpoint.

## Mechanics and cost

The fresh mechanics artifact is docs/plans/artifacts/surrogate-hmc-five-model-n4032-20261011-07. It used four
chains, two transitions, q-coordinate benchmark prior N(0,I), a fixed step
size of 0.002 and one leapfrog step. The prior is an explicit mechanics
fixture only; it does not settle the final scientific prior. All healthy
postwarmup draws would count as inference draws in a real run, and no separate
validation phase is inserted.

| model | mechanics status | seconds per four-chain transition | minimum 2000-warmup hours at measured rate | acceptance per chain | max delta H | divergences | force fallbacks |
|---|---|---:|---:|---|---:|---:|---:|
| ksc | PASS | 19.9274 | 11.0708 | 1,1,1,1 | 0.000171854 | 0 | 0 |
| range_bearing | PASS | 27.7916 | 15.4398 | 1,1,1,1 | 0.00318506 | 0 | 0 |
| lgssm | PASS | 128.046 | 71.1366 | 1,1,1,1 | 2.53127e-05 | 0 | 0 |
| predator_prey | PASS | 87.8024 | 48.7791 | 0.5,1,1,1 | 0.351272 | 0 | 0 |
| SIR | TIME_BUDGET_STOP | not measured | not measured | not produced | not produced | not produced | not produced |

The timing includes first graph compilation and is therefore a feasibility
diagnostic, not a steady-state runtime forecast. It nevertheless shows that a
naive full run is expensive: the nominal minimum-warmup extrapolation is about
11.1 hours for KSC, 15.4 hours for range bearing, 48.8 hours for predator-prey,
and 71.1 hours for LGSSM. SIR exceeded its allocated 492.7-second share without
a durable pilot result. The fair-share scheduler prevented the first model from
consuming the complete campaign budget.

The four completed mechanics pilots had finite states and endpoint values,
zero divergences, and zero force fallbacks. Their acceptance values are
descriptive only because there were only two transitions. The SIR timeout is a
resource/feasibility result, not evidence that the SIR force is mathematically
invalid.

## Decision and inference status

| decision | primary criterion | veto diagnostics | uncertainty | next justified action | not concluded |
|---|---|---|---|---|---|
| Endpoint/filter preparation | PASS for all five | no nonfinite endpoint; fixed-input replay passed; scope hashes passed | SIR endpoint MCSE 0.222; nonlinear endpoints approximate | retain fresh selections | exact nonlinear posterior correctness |
| Mechanics pilot | PASS for KSC, range, LGSSM, predator-prey; SIR budget stop | completed pilots had zero fallback/divergence; SIR produced no pilot | two transitions; compile-inclusive timings | feasibility repair / explicit budget decision | convergence or superiority |
| Full adaptive HMC | NOT RUN | no warmup, retained draws, R-hat or ESS | target prior and compute budget unresolved; SIR cost unresolved | settle prior and budget, then rerun inference on fresh root | original-posterior validity or default readiness |

Inference-status classification:

| row | status |
|---|---|
| hard-veto screen | No hard numerical veto for the four completed pilots; SIR has a declared time-budget stop |
| statistically supported ranking | None; no comparative posterior experiment ran |
| descriptive-only differences | acceptance, delta H, and compile-inclusive cost estimates |
| default readiness | Not established |
| next evidence needed | target-prior decision, adequate compute budget, SIR feasibility repair, adaptive warmup and cumulative retained draws, posterior-reference/comparator checks |

The current master correctly reports INCOMPLETE_SEE_MODEL_STATUS and writes
next-phase-plan.json with automatic_continuation=false; it does not claim
that the requested HMC campaign finished. Full inference requires an explicit
choice between the proposed benchmark-normal q prior and the existing
model-specific posterior setup, plus a total wall budget. This matters because
the saved KSC setup uses an independent Uniform(.1,.9) physical-parameter prior,
while the mechanics pilot used q~N(0,I). The current measured costs also make
an unmodified all-five run unlikely to finish within 24 hours and uncertain
within 48 hours.

## Red-team note

The strongest alternative explanation for the long timings is first-use XLA
compilation and the deliberately serial score-coordinate loop; the pilot does
not identify the steady-state cost. A result that would overturn the
feasibility conclusion is a separately measured compiled steady-state
implementation with materially lower per-transition cost. A result that would
overturn the mechanics pass is any nonfinite force, fallback, or divergence in
a longer run. None of these pilots tests posterior agreement with the
independent endpoint.

## Subsequent performance diagnosis (2026-10-11)

A separate exact-call-chain and GPU cost diagnostic is recorded in docs/plans/surrogate-hmc-five-model-cost-result-2026-10-11.md. Its bounded run found warmed full T50 scalar-force times of 15.07/15.59 seconds for LGSSM and 35.21/35.61 seconds for SIR on an RTX 4080 SUPER, with both forces finite and valid. SIR still has no completed four-chain mechanics pilot. The compile-inclusive per-transition figures and 2000-warmup arithmetic in this historical result are not steady-state runtime forecasts; no new posterior samples or convergence evidence were produced.
