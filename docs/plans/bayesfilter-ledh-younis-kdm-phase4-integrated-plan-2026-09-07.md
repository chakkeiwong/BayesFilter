# Phase 4: Integrated Younis-KDM Score Investigation

Date: 2026-09-07  
Status: `PHASE4A_GPU_REPAIRED; TF32_VETO_CONFIRMED; ORACLE_AND_RUNNER_PASS; TIMING_SCOPE_REVISED; SMALL_CELL_PILOT_COMPLETED; NO_PROMOTION_EVIDENCE; BROAD_LADDER_PAUSED; PHASE4B_MATH_BLOCKED`  
Governing reset:
`docs/plans/bayesfilter-ledh-younis-kdm-score-research-reset-2026-09-07.md`

## Research intent

The question is whether a continuous kernel-mixture representation can reduce
the error or variance of a score estimate for the state-space model when the
filtering machinery is LEDH--PF-PF followed by Contract-E, GenUT moment
restoration, and the owner dual-cap trust-region correction.

This question is separate from whether a positive-bandwidth program has the
same finite value as the canonical atom program. It does not. The canonical
baseline is

```text
ATOM-FINITE:  S_0^N(theta; xi) = d L_0^N(theta; xi) / d theta.
```

The score oracle on a linear-Gaussian state-space model is

```text
MODEL:  s(theta) = d log p_theta(y_1:T) / d theta.
```

KDM candidates are judged by error relative to `MODEL`. They are not admitted
by agreement with a different finite scalar.

## Two different KDM programs

### Phase 4A: integrated kernelized observation weighting

Let the ordinary pre-observation PF--PF logits be

```text
b_ti = log w^-_ti + log f_theta(x_ti | x^+_{t-1,i})
       + log |J_ti| - log q_theta(x^0_ti | x^+_{t-1,i}, y_t).
```

For a Gaussian state kernel with covariance `B_ti` and a linear-Gaussian
observation law, define

```text
gbar_ti = integral g_theta(y_t | x_ti + u) N(du; 0, B_ti)
         = N(y_t; C_theta x_ti,
             R_theta + C_theta B_ti C_theta^T),

a^h_ti = b_ti + log gbar_ti,
Z^h_t  = sum_i exp(a^h_ti),
alpha^h_ti = exp(a^h_ti) / Z^h_t.
```

The executed Phase 4A program uses `alpha^h_t` in the existing Contract-E
reset and the existing GenUT/dual-cap correction. Later transition, flow,
weight, covariance, and reset operations therefore consume the changed state
and must carry its tangent. Its scalar and score are

```text
L_obs-kdm^N = sum_t log Z^h_t,
S_obs-kdm^N = d L_obs-kdm^N / d theta.
```

This is `KDM-FINITE`. It is a fully differentiated finite program with KDM
feedback, but it is not the complete Bayesian update of a continuous Gaussian
mixture. In particular, it does not propagate each component's conditional
posterior mean and covariance. It must therefore be named
`kernelized_observation_weighting`, not `Younis MDPF` or `full KDM posterior`.
The distinction is part of the result, not a minor implementation caveat.

The Contract-E reset also carries the particle-local UKF covariance with the
same target-by-source transport used for the state.  If `A_ri` is the
normalized reset transport, the executed recurrence is

```text
P'_r  = sum_i A_ri P_i
dP'_r = sum_i dA_ri P_i + sum_i A_ri dP_i.
```

The carried covariance, rather than the pre-reset source ordering, is consumed
by the next UKF prediction.  This was a real call-chain bug found and repaired
on 2026-09-07; the focused endpoint tests now exercise it.  The covariance is
the particle-local conditional UKF covariance, so no between-cloud scatter is
added.

The zero-bandwidth branch uses `B=0` directly in the effective observation
covariance. It must reproduce `ATOM-FINITE` value, score, posterior weights,
reset states, and later-step trajectory. A positive semidefinite, rank-
deficient `B` is valid because the Gaussian integral is with respect to a
probability measure and `R + C B C^T` remains positive definite; no ambient
Lebesgue density for `N(0,B)` is asserted.

### Phase 4B: complete mixture-resampling importance reference

Younis and Sudderth sample from a complete continuous mixture and use a
proposal frozen at the differentiation point. A full BayesFilter reference
must likewise specify all of the following before code is written:

1. the exact cloud whose members are mixture component means;
2. component covariances and their support measure;
3. the categorical component law and continuous sampling map;
4. whether ancestry is retained as part of the state or marginalized;
5. the complete proposal density, not only the sampled component density;
6. the matching model numerator, including the sum over ancestors when
   ancestry is marginalized;
7. the LEDH map and Jacobian if sampling occurs before the flow;
8. the sequential anchor/replay rule across all time steps; and
9. the posterior particle set, weights, and total reset tangent passed to
   Contract-E and the dual-cap correction.

The previous conditional expression `q_KDM(x | J,y)` did not define items
1--4. Choosing those objects arbitrarily would yield a valid importance ratio
for an invented proposal, not a faithful implementation of the intended
research mechanism. Phase 4B is therefore mathematically blocked pending a
separate derivation. Existing density and anchored-weight primitives remain
diagnostic only.

For any fixed-anchor `MODEL-IS` implementation, the denominator and sampling
law are constant in the derivative. The finite estimator of a normalizer and
its derivative may be unbiased under the usual support and interchange
conditions, but the ratio `d hat Z / hat Z` is generally a biased finite-sample
score estimator. Proposal or selection log-density tangents must not be
subtracted from the anchored estimator. A moving-proposal finite program would
need a separate pathwise/measure derivative and may not reuse the anchored
formula.

## Evidence contract

| Field | Frozen requirement |
|---|---|
| Main question | Does Phase 4A or a later fully specified Phase 4B reduce paired model-score error at equal or explicitly reported compute? |
| Exact oracle | Independent Kalman value and analytical score for a nondegenerate LGSSM. |
| Primary baseline | The actual single-cloud canonical analytical endpoint with `reset_policy=contract_e` and both owner dual-cap mechanisms active. |
| Candidate 4A | Full-feedback `kernelized_observation_weighting` sharing the canonical executor; label `KDM-FINITE`. |
| Candidate 4B | Not yet authorized; requires the nine-item proposal derivation above; label `MODEL-IS`. |
| Engineering pass | Zero-bandwidth full-trajectory identity, positive-bandwidth finite-difference agreement, Gaussian-factor identity, PSD/support checks, and executable call-chain identity. |
| Research promotion criterion | On untouched paired paths, the predeclared confidence interval for the change in squared score error is below zero and below the practical threshold, with MCSE small relative to the effect. |
| Promotion veto | Any wrong target label, missing tangent, model-factor mismatch, nonfinite value, invalid covariance/support, changed canonical default, stale tuning scope, failed full-trajectory zero-bandwidth identity, or heuristic underperformance in a salient condition. |
| Continuation veto | Invalid oracle, invalid shared canonical engine, corrupt artifacts, exhausted campaign budget, or a mathematical contradiction in the declared candidate. Candidate underperformance is a repair/promotion veto, not automatically a direction veto. |
| Explanatory only | ESS, cap activity, value shift, bandwidth sensitivity, runtime, peak memory, mixture overlap, and descriptive tail errors. |
| What is not concluded | Derivative parity does not show lower model-score error. Lower variance does not show lower bias. An LGSSM result does not establish nonlinear or DSGE validity. |
| Artifacts | Unique directories below `docs/benchmarks/artifacts/ledh_younis_kdm_phase4_20260907/`, with plan, Git state, exact command, environment, device/XLA/TF32/memory policy, seeds, settings, rows, uncertainty summary, and checksums. |

## Heuristic adversary set and conditional situations

The candidate must be compared with, not tuned against:

- the exact Kalman score (oracle, not a heuristic);
- canonical `ATOM-FINITE` analytical LEDH score;
- bootstrap particle-filter score on the same LGSSM;
- the Phase 3 no-feedback KDM auxiliary while retaining the canonical score;
- zero-bandwidth Phase 4A, which must be exactly canonical; and
- a fixed diagonal innovation-kernel rule with no data-driven bandwidth
  selection beyond scale normalization.

Comparisons are conditional on short versus long horizons, small versus larger
particle counts, weak versus concentrated observations, active versus inactive
dual caps, and overlapping versus separated particle clouds. Losing to a
cheap eligible baseline in a salient condition blocks promotion there.

## Defaults and assumptions

| Choice | Provenance | Justification | Failure mode | Earliest diagnostic | Status |
|---|---|---|---|---|---|
| Contract-E plus both dual caps | Owner canonical policy | This is the named executed algorithm | A diagnostic no-reset slice could look artificially good | Wiring test and trace fields | Frozen baseline requirement |
| `B=diag((rho*s_d)^2)` | September 3 sidecar | Dimensionless scale family for a first comparison | Large `rho` changes the target; transferred scales mislead | Zero branch and bandwidth ladder | Hypothesis, never a universal default |
| Fixed `s_d` within a tuning scope | Calibration artifact | Makes `B` a declared numerical setting with `dB=0` | Poor coverage away from calibration scope | Scope mismatch and value-shift curve | Hypothesis |
| Explicit `dB` support | Project derivation | Required when a later bandwidth policy depends on theta | Silent partial derivative | Separate `dB` finite difference | Required capability |
| Linear-Gaussian convolution | Exact local derivation | Supplies a clean oracle gate | Cannot represent a non-Gaussian observation factor | Runtime atom-factor identity | Phase 4A scope limit |
| All-pairs mixture for Phase 4B | Younis Eqs. (4), (14)--(15) | Avoids a hidden sampled-component shortcut | Quadratic time/memory | Pair-count and peak-memory artifact | Required reference, not production default |

The September 3 selected value `rho=0.8` is not transferred. It was a boundary
selection from a different route and underpowered scope. It may justify
including values above 0.8 in an initial calibration grid, but cannot nominate
or default a bandwidth.

## Skeptical pre-execution audit

The plan was challenged for the required failure classes.

- **Wrong baseline:** the canonical comparator is the actual Contract-E and
  dual-cap endpoint, not the old GenUT sidecar and not `reset_policy=none`.
- **Proxy promotion:** finite differences, ESS, cap activity, runtime, and
  smoothness can veto or explain only. The primary comparison is paired error
  against the Kalman score with uncertainty.
- **Hidden target change:** positive `B` is always `KDM-FINITE`; only the
  explicit zero branch may claim canonical identity.
- **Incomplete KDM posterior:** Phase 4A is named kernelized observation
  weighting and is not represented as a full Younis MDPF. Phase 4B remains
  blocked until the complete proposal law is derived.
- **Unfair comparison:** paths, fixed streams, particle count, horizon, dtype,
  flow/reset settings, and reported compute are paired. Equal wall time is
  reported as a separate efficiency comparison rather than silently changing
  `N`.
- **Stale settings:** test-fixture values are mechanics settings only. A
  serious row requires a repository-issued tuning artifact for its exact
  model, horizon, particle count, dtype/backend, and complete LEDH/KDM controls.
- **Environment mismatch:** CPU-only checks are reference evidence. A serious
  run uses the trusted GPU path, memory growth, XLA where the complete route is
  compatible, and a manifest that says exactly what compiled.
- **Artifacts that cannot answer the question:** the result preserves
  path-by-stream score vectors, not only averages, so paired intervals, bias,
  variance, and MCSE can be recomputed.

The audit passes for Phase 4A implementation and focused engineering checks.
It does not authorize Phase 4B code or a broad multi-cell research campaign
yet.  The
existing `batch_fused` file is separately registered but its current endpoint
returns `children` directly after the UKF update and does not execute
Contract-E, GenUT, or the dual caps.  It is therefore not a full canonical
lane and is excluded from the Phase 4A experiment and NeuTra claims until a
dedicated repair replaces that reduced recurrence and its parity gates pass.

The repaired GPU evidence is now recorded.  Float64 and float32 without TF32
pass the complete endpoint smoke; the repaired float32 TF32 arm still fails the
declared value/score/state/weight parity thresholds, so TF32 remains vetoed for
this diagnostic route.  The first current-shape timing probe (`N=128,T=20`,
float64, XLA, no TF32) required `296.10 s` for compile plus first execution,
then `0.232 s` per warm execution and `31,457,280` peak GPU bytes.  The older
six-row timing note claiming 30--40 second compiles and a 0.10 GPU-hour total
does not describe the repaired code and is not used for a budget.

## Execution sequence

### 4A.1 Shared-engine refactor

Extract one internal analytical executor from
`canonical_value_and_analytical_score`. The registered canonical function
must call it with the ordinary observation factor. A repository-owned Phase 4A
wrapper calls the same executor with the convolved observation factor. No
copy of the flow, UKF, PF--PF, Contract-E, GenUT, or dual-cap recurrence is
permitted.

Required checks:

- canonical outputs before/after the refactor are identical on existing tests;
- every registered entry point still resolves;
- the Phase 4A endpoint reaches the same reset and higher-moment functions;
- the canonical route cannot acquire a KDM label or positive bandwidth by
  caller-stamped metadata.

### 4A.2 Complete Gaussian factor and total tangent

Implement component values and tangents for

```text
S_i = R + C B_i C^T,
mu_i = C x_i,
log gbar_i = log N(y; mu_i, S_i),
```

including `dx`, `dC`, `dR`, and `dB`. Check the supplied Gaussian
representation against the model's actual point-observation factor and
tangent at `B=0` on every step. Reject a mismatch rather than silently
convolving the wrong likelihood.

### 4A.3 Full-feedback gates

Use at least two time steps so the test proves that changed KDM weights feed
through Contract-E and dual caps into later states. Verify:

- exact zero-bandwidth value, score, weights, reset states, and trace parity;
- positive-bandwidth value shift is reported and nonzero on a discriminating
  fixture;
- analytical score matches a central finite difference of the same complete
  finite program;
- individual `dx`, `dC`, `dR`, and `dB` component tests;
- a singular PSD bandwidth in a declared support subspace is accepted, while
  an indefinite bandwidth is rejected; and
- no NumPy, pfor, sample-wise map, or separate reduced LEDH recurrence is added
  to the runtime path.

### 4A.4 Static graph/GPU readiness

After the reference gates pass, add a fixed-shape TensorFlow factory for the
complete Phase 4A endpoint. XLA compatibility, numerical parity, compile cost,
steady-state time, host/device peak memory, TF32 status, and memory growth must
be recorded before it may support a serious GPU run. A factor-only XLA test is
not evidence that the complete endpoint compiled.

### 4A.5 Calibration and untouched validation

Use the current bounded campaign runner only after a small-cell compile and
variance pilot.  Any larger cell requires its own compile probe and a fresh
versioned output directory; no N=512/T=50 claim follows from the superseded
timing note.  Calibration and validation use disjoint paths and streams:

```text
horizon:       5, 20, 50
particle N:    32, 128, 512
rho grid:      0, .025, .05, .10, .20, .40, .80, 1.20, 1.60
calibration:   disjoint paths and streams, selection by paired score MSE
validation:    untouched paths and streams, paired interval plus MCSE
```

The actual replication counts must be chosen from a pilot variance estimate
or a predeclared minimum-detectable-effect calculation. “Enough power” cannot
be replaced by an arbitrary seed count. Holdout data never tune `rho`.  The
The campaign runner is implemented.  A bounded one-cell pilot has now run with
disjoint calibration and validation paths.  The powered `N=32,T=5`,
float32/no-TF32 cell selected `rho=0.8`, but KDM-FINITE had 4.84% higher
validation score MSE than ATOM-FINITE; its paired bootstrap interval crossed
zero, so the ranking is descriptive only and the promotion criterion failed.
The complete result and row-level artifacts are in
`docs/plans/results/bayesfilter-ledh-younis-kdm-phase4a-campaign-result-20260908.md`
and `docs/benchmarks/artifacts/ledh_younis_kdm_phase4_20260908/campaign-attempt03-n32t5-powered-f32/`.
The old six-row timing estimate is superseded by the repaired `N=128,T=20`
compile probe and is not a current campaign budget.

### 4B. Complete mixture reference

Write the nine-item mathematical proposal specification, update the LaTeX
note, and repeat the skeptical audit. Only then implement the exact all-pairs
mixture sampler/density, target numerator, anchored derivative, Contract-E
reset, and dual-cap continuation. Sparse, nearest-neighbor, low-rank, or
sampled-component-only variants are later approximations and cannot stand in
for this reference.

### 5. DSGE extension

Proceed only after the LGSSM oracle gate. Phase 4A may use a singular PSD
innovation-space covariance because no ambient KDM density is formed. Phase 4B
must define its density with respect to the innovation/chart measure and
include any parameter-dependent chart Jacobian. Each DSGE target receives its
own tuning scope and observation-factor derivation.

## Current decision

Retain Phase 4A as a fully differentiated, full-feedback diagnostic using the
shared canonical engine, but do not promote it: the powered small-cell pilot
did not meet the score-error criterion.  The next discriminating diagnostic is
same-stream comparison of the complete rho curve against both ATOM-FINITE and
the Kalman oracle before any larger compile budget is spent.  Do not call the
route a full Younis filter.  Do not code Phase 4B until its proposal law is
complete.  The registered `batch_fused`/NeuTra lane remains a separate
implementation-conformance blocker because it still bypasses Contract-E,
GenUT, and the dual caps; no KDM result repairs that blocker.
