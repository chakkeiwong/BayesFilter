# Why LGSSM and SIR are slow, 2026-10-11

The current master spends its time repeatedly computing the analytical
particle-filter force. XLA is enabled; a Python loop per particle, observation,
or leapfrog step is not the explanation. SIR's earlier mechanics stage ended
at its wall deadline, not at a recorded numerical failure. The new full-T50
force evaluations and independent endpoints are finite and valid for both
models at the nominal parameter point. This establishes local evaluability,
not HMC convergence or validity throughout the posterior.

## Measured cost

The bounded diagnostic completed in 211.61 monotonic wall seconds out of 360.
It used the idle RTX 4080 SUPER, TensorFlow 2.20.0-dev0+selfbuilt, ordinary
FP32, TF32 disabled, N=4032, T=50, and verified memory growth. The previous
pilot used an RTX 5080; these are not paired timings
across the two runs. The unchanged preparation-attempt06 selections were
validated against the current source and data scope before evaluation.

| Quantity | LGSSM | SIR |
|---|---:|---:|
| State / observation / parameter dimensions | 3 / 3 / 4 | 18 / 9 / 3 |
| Selected flow / reset iterations | 32 / 80 | 16 / 40 |
| Full T50 scalar force, first call | 26.044 s | 47.089 s |
| Full T50 scalar force, two warmed calls | 15.074, 15.589 s | 35.209, 35.611 s |
| Independent scalar endpoint, warmed | 0.0165, 0.0351 s | 0.1075, 0.1041 s |
| First-step map construction, warmed | 0.0098, 0.0099 s | 0.0691, 0.0679 s |
| First-step three mixture densities, warmed | 0.1353, 0.0747 s | 0.0623, 0.0619 s |
| First-step transport/reset, warmed | 0.0582, 0.0584 s | 0.2761, 0.2391 s |
| TensorFlow allocator peak across this diagnostic | 276,041,984 bytes | 783,169,280 bytes |

The force includes all parameter directions for one chain, not one directional
filter call. The endpoint is exact Kalman for LGSSM and the current four-seed
FP64 bootstrap estimate for SIR. These are different computational quantities,
so their timings explain the force/acceptance split, not method superiority.
First-step isolated component times are not exact percentages of a fused full
filter. LGSSM's two density timings are variable. Two repetitions do not
support tight runtime intervals or a model-to-model performance ranking.

Both full force graphs have `_XlaMustCompile=true`, tracing count 1, and zero
PyFunc/EagerPyFunc nodes, including their function libraries. The same checks
pass for every timed component and endpoint. The force graphs contain 20
(LGSSM) and 22 (SIR) StatelessWhile definitions. These are graph-structure
counts, not executed iteration counts. The earlier completed LGSSM mechanics
pilot calls an explicitly must-compile outer HMC function. SIR's earlier log
only establishes that some XLA cluster compiled; it does not establish that
the outer pilot graph finished compiling before timeout.

## Actual call chain and repeated work

`surrogate_hmc_five_model.py:311` constructs a four-chain, one-leapfrog pilot.
The leapfrog implementation in `neural_force_hmc.py:598` records L+1 force
calls per transition. In the master binding (`surrogate_hmc_five_model.py:249`),
the force uses a serial TensorFlow while-loop over chains, and each scalar
force uses another serial TensorFlow while-loop over p parameter directions.
Each direction invokes the full T50 particle filter again. Thus a four-chain
one-leapfrog transition invokes 32 full LGSSM filters or 24 full SIR filters.
It also evaluates four independent endpoints; for SIR those contain sixteen
bootstrap-filter runs in total. The measured endpoints are relatively cheap.

Within each directional filter (`covariance_proposal_tf.py:226`), every
observation builds affine proposal maps, draws proposals, evaluates all three
Gaussian mixture densities, and resets the cloud by finite Sinkhorn transport
and covariance restoration. The outer time loop is a TensorFlow while-loop.

The mixture calculation (`ledh_marginal_weights_tf.py:22`) evaluates every
particle against every component. With 4032 particles, N squared is 16,257,024.
Three families over T50 require 2,438,553,600 point/component Gaussian
evaluations per directional filter. Dense covariance quadratic forms give
an arithmetic scaling of order T N squared d squared for this part. The
four-chain LGSSM transition therefore requests about 78.0 billion such
Gaussian pairs before counting transport work. The source-level count does
not claim a one-to-one correspondence with GPU kernels or FLOPs.

At N4032 the repository chunk policy selects K2016, a 2-by-2 grid. In
`covariance_proposal_transport_tf.py:22`, each tile constructs squared
distances and their analytical tangents. Both row and column reductions call
this tile code from inside every Sinkhorn iteration. That is eight source
tile evaluations per iteration, with 80 iterations for LGSSM and 40 for SIR,
plus final reductions and barycentres. The clouds and their tangents do not
change within that Sinkhorn solve. The dense N<=3000 route instead forms
these costs before its iteration loop (`covariance_proposal_tf.py:174`).
This is a concrete reuse opportunity; whether XLA already hoists any of this
work is not established without optimized-HLO or kernel-profile inspection.

SIR is the nine-region, 18-state model, not a scalar epidemic model. Its
transition and tangent each execute four RK substeps in TensorFlow loops
(`ledh_canonical_models_tf.py:111` and `:131`). Map construction also solves
18-by-18 systems per particle per flow step. It costs roughly 0.068 seconds
in the isolated first-step measurement, while reset costs 0.24-0.28 seconds.
Thus describing the problem solely as a slow ODE would miss the largest
component observed in this diagnostic. Larger state dimension increases
pairwise distance, covariance, and matrix work throughout the filter.

There are Python loops: model/phase scheduling, tuning candidates, diagnostic
repetitions, and serialization. The three-family `for b in range(3)` in
`component_densities` executes while tracing and creates graph operations.
The fixed four bootstrap endpoints are also constructed at tracing time.
These are not Python callbacks during compiled particle evolution. XLA
compiles the repeated work; it does not make the chosen complexity disappear.

## Two master-program issues exposed by the audit

First, LGSSM's filter-control selector doubled flow/reset iterations from
16/40 to 32/80 for a validation-objective change from 1.068379402 to
1.067853928, about 0.0492 percent. Selection uses the minimum descriptive
mixture loss (`surrogate_hmc_five_model.py:440`) and has no measured HMC
cost/benefit criterion. The result explicitly disclaims statistical ranking.
There is no demonstrated downstream benefit sufficient to justify this
additional work. Do not silently reduce iterations in an existing frozen
selection; a cost-aware comparison must preserve numerical and posterior
criteria and use a new selection.

Second, the pilot times the first call of the outer HMC graph and divides by
only two transitions (`surrogate_hmc_five_model.py:327`). It also evaluates
the force separately before this timed region, potentially compiling another
enclosing graph. The recorded 128.046 seconds per LGSSM transition and the
derived 71.1-hour warmup number are compile-inclusive arithmetic, not measured
steady-state forecasts. SIR used the remaining 492.7 seconds of the 1200-second
mechanics budget and left no pilot result. Its log does not localize exactly
which nested computation was active when killed. The new diagnostic removes
that ambiguity for scalar-force evaluability, not for completed SIR HMC.

## Repairs justified by this diagnosis

These are part of making the existing surrogate-HMC master executable at a
reasonable cost, not a change in the scientific question. The requested
FP32 and N4032 remain fixed. No production numerical implementation was
changed during this diagnosis.

1. Add durable pilot phase telemetry and distinct first-call/warmed timings;
   stop using compile-inclusive extrapolations to decide long-run budgets.
2. Inspect optimized transport work, then test caching fixed cost/tangent
   tiles without changing K2016, finite Sinkhorn iterations, or normalization.
3. Share the primal filter trajectory across analytical score directions;
   the current p separate calls duplicate that work. Preserve the total
   derivative and check every parameter direction against the current path.
4. Exploit structurally shared observation Jacobians/covariances for LGSSM
   and SIR, through a general supported property rather than a model-name
   fork. Factor/transform once where mathematically identical, instead of
   repeating covariance matrix work for every Gaussian pair or particle.
5. Evaluate genuine batched chains within measured memory bounds, and
   compare control settings on downstream validity plus computational cost.

Every arithmetic rearrangement needs value/score parity, finite/moment checks,
and the actual master call-chain check before a new HMC run. Replacing the
surrogate force with Kalman would make LGSSM cheap but would test a different
force; use Kalman as a control, not as a silent replacement. This diagnostic
does not establish a speedup for any unimplemented repair.

## Decision and inference status

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Diagnosis complete | Current scalar force/endpoints measured, first/warm separated | Both finite and valid; must-compile, single trace, no Python callbacks | Single point; isolated components; different GPU from pilot | Repair repeated work and pilot timing | Convergence or a completion-time forecast |
| SIR remains viable for repair | T50 scalar force and endpoint complete | Prior pilot has time-budget stop, no observed numerical error | SIR outer HMC pilot still incomplete | Optimize and repeat bounded mechanics | SIR posterior validity or full HMC success |

| Inference status | Finding |
|---|---|
| Hard veto screen | No new numerical veto at the nominal point; no allocator OOM observed |
| Statistically supported ranking | None |
| Descriptive differences | Timings in the table, first-step cost concentration |
| Default readiness | No change; not established by this diagnostic |
| Next evidence needed | Numerical equivalence and warmed end-to-end timing after each performance repair, then adaptive HMC |

Post-run red team: first-step components may differ from later observations;
optimized XLA may fuse or hoist work differently inside the full force; two
warm measurements are insufficient for a precise performance distribution.
A kernel profile or exact-reuse prototype with no timing benefit would
overturn the corresponding optimization hypothesis, without invalidating
the measured full-force cost. No method or convergence ranking follows.

Evidence: `docs/plans/artifacts/surrogate-hmc-five-model-cost-20261011-01/`
contains the command manifest, complete worker logs, and each model's
`timings.json` with Git/source/data hashes, controls, device/memory policy,
graph metadata, timing repetitions and validity. Plan:
`docs/plans/surrogate-hmc-five-model-cost-diagnosis-2026-10-11.md`. Harness:
`scripts/diagnose_surrogate_hmc_five_model_cost.py`.
