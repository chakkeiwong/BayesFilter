# Zhao-Cui Moment-Teacher Integration Campaign Result

Date: 2026-07-31
Plan: `docs/plans/bayesfilter-zhao-cui-moment-teacher-integration-campaign-plan-2026-07-30.md`
Selected execution route: GPU, FP32, TF32 disabled, XLA enabled
Classification: `extension_or_invention`
Status: campaign executed; Austria SIR candidate blocked by tuning/representation veto

## Outcome

The canonical particle/OT/Contract-E-Chol moment-teacher finite program is now
implemented and tested with a same-program analytical score. Deterministic
derivative parity, factory-bound route identity, GPU/XLA execution, and LGSSM
claims at `T=2,10,50` pass their validity gates. The repaired `T=50` evidence
uses six separately identified claim shards.

The conditional nonlinear phase has a split result:

- predator-prey `T=20`, `N=1024` passes as a one-seed descriptive feasibility
  probe; and
- score-admissible latent-preclip Austria SIR `T=20`, `N=1024` does not reach
  claim execution because both predeclared tuning arms fail to produce a valid
  recursively carried TT teacher.

This is a complete execution of the bounded campaign, not a complete
all-model pass. The Austria failure is a candidate tuning/representation hard
veto. It does not invalidate the particle likelihood, streaming transport,
Contract-E-Chol reset, higher-moment correction, LGSSM implementation,
predator-prey implementation, or Zhao-Cui research direction.

## Claimed Target And Computed Quantity

| Item | Result |
|---|---|
| Claimed target | Total analytical derivative of the exact finite hybrid particle/TT likelihood program |
| Likelihood owner | Particle normalized-weight increment at each observation |
| TT role | Independently fitted adjacent squared-TT standardized shape targets only |
| Reset | `contract_e_chol_v1` using total direct moment/weight plus streaming-transport derivative composition |
| Quantity actually computed | The same finite particle scalar and its manual total score, including carried TT and particle state |
| Equality verdict | Correct on deterministic finite-difference fixtures; LGSSM and predator-prey execute that program; Austria stops before a claim program is issued |
| Source classification | Zhao-Cui squared-TT fitting/marginal operations are source-anchored; the moment-teacher/Contract-E composition is `extension_or_invention` |

No TT scale shift or TT normalizer is added to the particle likelihood. A
frozen teacher score, raw-barycentric reset, Contract-E-TP route, or fixed APF
program would compute a different quantity and was not substituted.

## Implementation

The campaign added or completed:

- `bayesfilter/highdim/zhao_cui_moment_teacher_lgssm_tf.py` for the canonical
  LGSSM particle/TT composition and total score;
- `bayesfilter/highdim/zhao_cui_moment_teacher_nonlinear_tf.py` for source-order
  bootstrap proposals, canonical streaming Contract-E-Chol, and TT shape
  correction on nonlinear adapters;
- exact exposure of pre-reset particle moments and tangents from the canonical
  LGSSM step;
- repository-issued scope-specific tuning artifacts;
- repository-factory identities for LGSSM, predator-prey, and Austria SIR,
  including fixed model dimensions and cross-model substitution rejection;
- transition-coupled TT fit rows that are fixed offline and independent of
  claim particles;
- source-order `x0 -> transition -> observation` event binding; and
- an exact latent-preclip Austria adapter: the first transition uses the latent
  state, while later transitions clip susceptible coordinates and apply the
  corresponding piecewise tangent.

The new runtime route imports no NumPy. Tensor materialization occurs only at
setup validation or artifact/reporting boundaries. Concrete graphs contain
TensorFlow control flow and no `PyFunc` or `EagerPyFunc`.

## Deterministic Verification

Final CPU-hidden command:

```bash
CUDA_VISIBLE_DEVICES=-1 python -m pytest -q \
  tests/highdim/test_zhao_cui_moment_teacher_nonlinear.py \
  tests/highdim/test_zhao_cui_moment_teacher_integration.py \
  tests/highdim/test_cubature_genut_adapters.py \
  tests/highdim/test_ledh_contract_e_latent_sir_identity.py \
  tests/highdim/test_ledh_contract_e_schema_v2_factory.py
```

Result: `52 passed`, two TensorFlow Probability deprecation warnings, in
82.97 seconds.

The checks cover:

- tuning-scope mismatch rejection;
- zero-correction exact tie-out to the matching empirical-target canonical
  finite program;
- nonzero total analytical score versus centered finite differences;
- first-step source event order and Austria latent-preclip tangent behavior;
- mean/covariance restoration;
- exact-divisor chunk selection;
- repository factory identity and cross-model substitution rejection; and
- TensorFlow control flow with no host callback.

## Integrated GPU Gate

Artifact:
`docs/benchmarks/artifacts/zhao_cui_moment_teacher_integration_20260730/attempt02_gpu_xla_fp32_no_tf32/result.json`

SHA-256:
`51786c2e5f629db5283ffe281eaa6f4da38bf632269a5d2d295ec2f33c34a367`

| Check | Result |
|---|---:|
| GPU | NVIDIA GeForce RTX 4080 SUPER |
| TensorFlow | 2.19.1 |
| FP32 / TF32 / XLA | yes / disabled / compiled |
| Memory growth | verified before logical initialization |
| Maximum FP32/FP64 absolute error | `3.1167e-7` |
| Maximum relative error | `8.1467e-7` |
| Host callbacks | none |
| Peak TensorFlow allocation | 444,160 bytes |
| Warm mechanics execution | 1.56 seconds |

Attempt 01 is a preserved setup dtype failure before candidate tracing.

## LGSSM Claims

All claims use `N=1024`, `K=1024`, FP32, TF32 disabled, GPU/XLA, six frozen
claim seeds `81910..81915`, disjoint calibration/validation data, and a frozen
scope-specific tuning artifact.

| Horizon | Value error / MCSE | Maximum score error / MCSE | Coordinate | Hard validity | Signs |
|---|---:|---:|---|---|---|
| `T=2` | 0.708 | 2.262 | `phi2` | pass | every value/score series mixed |
| `T=10` | 0.422 | 1.793 | `phi1` | pass | every value/score series mixed |
| `T=50` | 0.769 | 2.299 | `phi2` | pass | every value/score series mixed |

Per-coordinate absolute mean error / MCSE:

| Horizon | `phi1` | `phi2` | `phi3` | `q_scale` | `r_scale` |
|---|---:|---:|---:|---:|---:|
| `T=2` | 0.801 | 2.262 | 0.368 | 0.382 | 1.172 |
| `T=10` | 1.793 | 1.677 | 0.210 | 1.290 | 0.848 |
| `T=50` | 1.099 | 2.299 | 0.778 | 0.686 | 0.481 |

The maximum mean/covariance restoration residuals are at FP32 roundoff. No
coordinate is all one-sided at any horizon. Six seeds therefore show no
all-one-sided displacement pattern, but they do not prove zero bias.

The campaign audit originally failed to state a numerical stochastic
continuation threshold. Before nonlinear execution, the plan was amended to
use `abs(mean error)/MCSE <= 3`, mixed signs, and all hard validity gates as a
feasibility-only anomaly screen across 18 horizon/quantity checks. All three
horizons pass. Because this threshold was added after the LGSSM claims ran, it
must not be represented as a prospectively calibrated unbiasedness or
promotion test.

Artifacts and SHA-256 values:

| Horizon | Artifact | SHA-256 |
|---|---|---|
| `T=2` | `lgssm_t2_claim_provenance_repair_attempt01/result.json` | `52b2ad33d0f4603be94f48c9075b487c197b4446234ab86ad1584f094abfd1ba` |
| `T=10` | `lgssm_t10_claim_provenance_repair_attempt01/result.json` | `e02ab45a3cc46b537ea7f67dd0852c87c637578803c26fd4f727d3e073dac526` |
| `T=50` | `lgssm_t50_claim_aggregate_attempt01.json` | `bf83998eda7cbe713b2e386f29496ee1771d60fe250ec75e08ecccd95ecb4e3b` |

The `T=50` aggregate binds six distinct per-seed route identities. The earlier
combined result remains informative but is provenance-ineligible because its
claim rows lack actual per-seed route identity.

## Predator-Prey Feasibility

Terminal artifact:
`docs/benchmarks/artifacts/zhao_cui_moment_teacher_integration_20260730/nonlinear_predator_prey_t20_attempt02_terminal/result.json`

SHA-256:
`94610c09f5448b29fa5f1bc29c0ce05e7f9c6bdd4c1313e3296074aebd45b4d4`

Route identity:
`29817d3b7fd48f8786ffc9837fdb58dfe7b14bbe3e16ad03f7d24901908e7596`

Scope: source-order predator-prey, `T=20`, `N=1024`, `K=1024`, physical six-
parameter coordinates, FP32, TF32 disabled, GPU/XLA, claim seed `82910`.

Candidate 0 failed its early chart-support diagnostic because it did not
contain enough transition-coupled fixed fit rows. Candidate 1 passed
calibration and validation and was frozen before claim execution:

- diagonal correction strength `0.01`, one step;
- no pairwise correction;
- TT ridge `1e-4`;
- Sinkhorn/balance counts `20/8`; and
- tuning artifact
  `522a6dbc4007945f9ab85425c219a5894b47389d4a63b088cd651518ed3c32ec`.

Untouched one-seed claim:

| Metric | TT teacher | Empirical-target Contract E | Paired difference |
|---|---:|---:|---:|
| Value | -102.760483 | -102.761208 | +0.000725 |
| `r` score | -30.629938 | -30.634918 | +0.004980 |
| `K` score | 0.179871 | 0.179810 | +0.000061 |
| `a` score | -0.070838 | -0.070782 | -0.000056 |
| `s` score | 0.180626 | 0.182615 | -0.001988 |
| `u` score | 14.002126 | 13.986257 | +0.015869 |
| `v` score | -18.376564 | -18.356934 | -0.019630 |

All hard gates pass. Maximum mean/covariance residuals are `7.63e-6` and
`9.54e-7`; peak TensorFlow allocation is 68,612,608 bytes. Total harness wall
time is 174.59 seconds and the claim pair takes 39.01 seconds. The graph has
XLA while control flow and no host callback.

These differences are descriptive only. One seed cannot rank the TT teacher
against empirical-target Contract E or establish nonlinear improvement.

## Austria SIR Tuning Veto

Terminal artifact:
`docs/benchmarks/artifacts/zhao_cui_moment_teacher_integration_20260730/nonlinear_austria_sir_t20_attempt03_terminal/result.json`

SHA-256:
`ad58416a9162a937f6f795ddddfff04e4d52818bd5047e9d6c90def5c91b181a`

The target is the sealed source-order latent-preclip Austria program at
`T=20`, with three log-scale parameters and observed `y1:y20`. The first
transition consumes the latent initial state; later transitions clip
susceptible coordinates and propagate the corresponding piecewise tangent.

Both predeclared rank-1/basis-2/chart arms fail during offline recursive TT
branch freezing with `nonlinear teacher scale-shift freeze found an invalid
fit`. The claim seed `82910` is not evaluated. GPU visibility, memory growth,
TF32-off, and XLA compilation are recorded, but there is no route identity
because no valid tuned finite program was issued.

The bounded repair ladder keeps the target, data, event order, particle
contract, and backend unchanged:

- corrected the initially discovered wrong non-preclip adapter;
- tested two independently declared fixed charts;
- tested TT ranks 1 and 2;
- tested basis sizes 2 and 3;
- tested 96, 144, and 192 fit rows;
- tested one and two ALS sweeps;
- tested TT ridges `1e-4` and `1e-3`; and
- tested the original scale-consistent defensive convention from `1e-6` to
  `1.0`.

Every target-preserving arm remains invalid before particle execution. A
diagnostic per-time log-centering plus defensive weight could make the finite
algebra run, but it changes the physical defensive-mixture coefficient unless
the entire scale convention is rederived. That experiment was rejected and
removed; it is not evidence.

Failure classification: tuning/representation failure of the current Austria
TT teacher. It is not an implementation failure of the particle likelihood,
streaming OT, Contract-E-Chol reset, or score because none of those claim steps
ran. It is not evidence against the moment-teacher research direction because
LGSSM and predator-prey passed.

## Decision Table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Retain the canonical integrated route as a tested opt-in finite program | derivative parity, identity, GPU/XLA, and LGSSM pass | clear for tested LGSSM scopes | only six seeds and post-hoc continuation threshold | preserve route and artifacts | no HMC or posterior readiness |
| Treat predator-prey as feasible | one untouched `N=1024`, `T=20` claim passes | clear | one seed; no ranking uncertainty analysis | optional multi-seed replication only if a predator-prey claim is needed | no superiority or default change |
| Reject current Austria candidate | no valid tuned recursive TT teacher | tuning/representation hard veto | whether a new stabilized parameterization can preserve the exact density in FP32 | write a new plan for log-domain/normalized TT recursion or another exact scale parameterization | no rejection of Contract E or Zhao-Cui direction |
| Keep empirical-target Contract E unchanged | no broad nonlinear/default evidence | clear | TT benefit not statistically established | no default promotion | no leaderboard or production claim |

## Inference Status

| Item | Status |
|---|---|
| Hard veto screen | LGSSM and predator-prey pass; Austria tuning fails before claim |
| Statistically supported ranking | none |
| Descriptive-only differences | all paired TT-versus-empirical nonlinear differences and six-seed continuous summaries |
| Viable candidates | integrated LGSSM route and predator-prey opt-in route |
| Default readiness | not established |
| Next evidence needed | target-preserving Austria TT scale/normalization repair; multi-seed nonlinear evidence before any ranking |

## Run Manifest

| Field | Value |
|---|---|
| Git commit | `fb9a0679adb7c731ff2ac42551f39bdcc15222a1` with dirty source hashes in artifacts |
| Environment | TensorFlow 2.19.1, conda `tf-gpu` |
| GPU | NVIDIA GeForce RTX 4080 SUPER |
| GPU policy | memory growth verified before logical initialization |
| Numerical route | FP32, TF32 disabled, XLA enabled |
| Particle/chunk | `N=1024`, exact `K=1024`, `1 x 1` grid |
| Nonlinear seeds | calibration `82900`, validation `82901`, claim `82910` |
| Output root | `docs/benchmarks/artifacts/zhao_cui_moment_teacher_integration_20260730/` |
| Plan | `docs/plans/bayesfilter-zhao-cui-moment-teacher-integration-campaign-plan-2026-07-30.md` |
| Result | this file |

Exact commands are recorded in each result JSON. Failed attempts are preserved
under unique directories and were not overwritten.

## Post-Run Red Team

Strongest alternative explanation for the LGSSM result: the six-seed MCSE
estimates are noisy, and the continuation threshold was added after observing
the claims. Mixed signs rule out only an all-one-sided pattern in these six
seeds; they do not establish unbiasedness.

Strongest alternative explanation for predator-prey: the bounded correction
is small, so its validity may say little about whether materially stronger TT
shape correction would remain stable or improve a downstream task. A one-seed
small difference is not evidence of improvement.

Strongest alternative explanation for Austria: the failure may reflect an
inadequate fixed affine chart, fit design, or scale parameterization rather
than intrinsic failure of squared TTs. The repair ladder is broad enough to
reject the present candidate, not the idea.

Evidence that would overturn the closeout:

- a derivation and implementation of a target-preserving normalized/log-domain
  TT recursion that passes Austria derivative and shape gates in FP32; or
- evidence that the current Austria target/event-order binding is itself wrong.

Weakest evidence: nonlinear scientific value. Predator-prey has only one
claim seed, and Austria has no claim. No nonlinear ranking, default change,
HMC readiness, posterior correctness, production readiness, or source-faithful
Zhao-Cui filtering claim follows.
