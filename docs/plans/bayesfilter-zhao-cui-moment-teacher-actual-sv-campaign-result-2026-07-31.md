# Zhao-Cui Moment-Teacher Actual-SV Campaign Result

Date: 2026-07-31
Plan: `docs/plans/bayesfilter-zhao-cui-moment-teacher-actual-sv-campaign-plan-2026-07-31.md`
Status: terminal pass; no statistically supported improvement
Target: exact transformed SV, `z_t=log(y_t^2)`
Route: GPU, FP32, TF32 disabled, XLA enabled
Classification: `fixed_hmc_adaptation` for the SV adapter plus
`extension_or_invention` for moment-teacher/Contract-E composition

## Outcome

The integrated moment-teacher algorithm runs successfully on exact transformed
stochastic volatility at `T=20`, `N=1024`. The dense reference, deterministic
score checks, GPU/XLA graph checks, memory policy, route identity,
mean/covariance restoration, and all six untouched candidate/baseline pairs
pass their hard gates.

The candidate does not demonstrate a statistically supported accuracy
improvement over empirical-target Contract E. Relative to the dense exact
reference, it is descriptively closer in value and `log_beta` score, and
slightly farther in `z_gamma` score. Every paired 95% interval for absolute-
error gain includes zero.

KSC-SV was not run. Its seven-component Gaussian-mixture observation model is
a different target and was excluded from this exact transformed-SV campaign.

## Target And Quantity

| Item | Result |
|---|---|
| Source model | `X0 ~ N(0, 1/(1-gamma^2))`, `Xt=gamma Xt-1+eta`, `Yt=beta exp(Xt/2) epsilon` |
| Physical point | `gamma=0.6`, `beta=0.4`, `sigma=1` |
| Coordinates | `(z_gamma=Phi^-1(gamma), log_beta)` |
| Event order | `x0 -> transition -> y1` |
| Likelihood owner | Normalized particle-weight increment |
| TT role | Independently fitted adjacent squared-TT shape targets only |
| Reset | `contract_e_chol_v1`, streaming Contract-E-Chol |
| Computed score | Total derivative of the same finite particle/TT/transport/reset program, including stationary initial density value/tangent |
| Source status | SV adapter is a fixed-HMC adaptation; moment-teacher composition is an extension/invention |

The TT normalizer does not enter the likelihood. The raw-to-transformed
Jacobian is parameter-independent, so parameter scores are unchanged by the
observation transform; values are reported on the transformed target.

## Deterministic And Reference Gates

CPU-hidden focused suite: `18 passed`, two TFP deprecation warnings, 68.88 s.
It includes source-order data, stationary initial-density tangent parity,
zero-correction tie-out, same-program score finite differences, identity and
cross-model rejection, and a regression proving scalar and multivariate static
XLA shapes do not share an invalid relaxed trace.

Reference artifact:
`docs/benchmarks/artifacts/zhao_cui_moment_teacher_actual_sv_20260731/reference_attempt01/result.json`

SHA-256:
`8137eb1020f223ea98cf39a85f038199ed20af98df3e2625aa96c01964d138ad`

| Dense reference | Result |
|---|---:|
| Value | `-42.20936045983713` |
| `z_gamma` score | `0.3932603193192895` |
| `log_beta` score | `1.7217809572935232` |
| Maximum refinement value gap | `6.90e-11` |
| Maximum refinement score gap | `6.87e-10` |
| Maximum score-increment sum residual | `8.88e-16` |

The authority is sequential FP64 TensorFlow dense quadrature with
`(order,radius)=(257,8),(401,8),(401,10)`. GPU was intentionally hidden.

## GPU And Tuning

Terminal artifact:
`docs/benchmarks/artifacts/zhao_cui_moment_teacher_actual_sv_20260731/gpu_attempt02_repair_balance/result.json`

SHA-256:
`18eddf1f47adce1e2c210e7263d8b18aa26f31ace38132881283c2c24f88ddfa`

Attempt 01 is preserved at:
`docs/benchmarks/artifacts/zhao_cui_moment_teacher_actual_sv_20260731/gpu_attempt01/result.json`

SHA-256: `7a44423c7f033f29e4a947e25e7bff306b2740fdac496be43d33b33e02ef3afc`

Attempt 01 had shared calibration invalidity in candidate and baseline at seed
`83900`; validation passed and no claim seed ran. The predeclared repair raised
terminal balance counts from 8 to 20/32. Attempt 02 then passed.

Selected tuning artifact: `00b795604afdc60a80f8eec6c7e4876fbfdd1d74531ca29c3e191d2d8e7b6e73`

Selected controls: `sinkhorn_steps=20`, `balance_steps=20`,
`correction_steps=1`, `correction_strength=0.005`, `tt_ridge=1e-3`, no pairwise
correction.

| GPU manifest field | Result |
|---|---|
| Device | NVIDIA GeForce RTX 4080 SUPER |
| TensorFlow | 2.19.1, conda `tf-gpu` |
| Memory growth | Verified before logical initialization |
| TF32/XLA | disabled/enabled |
| Particle/chunk | `N=1024`, exact `K=1024`, `1 x 1` grid |
| Peak TensorFlow allocation | `17,814,016` bytes |
| Wall time | `233.80` s |
| Graph | `While` present; no `PyFunc` or `EagerPyFunc` |

All hard-veto flags are false. Six claim rows have six distinct repository-
issued route identities.

## Claim Values And Scores

Previous means empirical-target Contract E under identical prepared randomness.
Candidate minus baseline is a paired displacement, not an accuracy metric.

| Seed | Candidate value | Baseline value | Delta | Candidate `z_gamma` | Baseline `z_gamma` | Delta | Candidate `log_beta` | Baseline `log_beta` | Delta |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 83910 | -42.1413460 | -42.1407661 | -0.0005798 | 0.1916023 | 0.1926047 | -0.0010023 | 1.2957623 | 1.2932909 | +0.0024714 |
| 83911 | -42.0308304 | -42.0299301 | -0.0009003 | 0.5106979 | 0.5115545 | -0.0008566 | 1.4778738 | 1.4751115 | +0.0027623 |
| 83912 | -42.1791420 | -42.1785202 | -0.0006218 | 0.5078294 | 0.5086419 | -0.0008125 | 1.5952737 | 1.5922042 | +0.0030695 |
| 83913 | -42.1929512 | -42.1923904 | -0.0005608 | 0.4309026 | 0.4313917 | -0.0004891 | 1.7428296 | 1.7401980 | +0.0026315 |
| 83914 | -42.2467880 | -42.2461624 | -0.0006256 | 0.3678896 | 0.3688023 | -0.0009128 | 1.7252034 | 1.7229842 | +0.0022192 |
| 83915 | -42.1944733 | -42.1939201 | -0.0005531 | 0.3341592 | 0.3355372 | -0.0013780 | 1.6431264 | 1.6412060 | +0.0019203 |

## Accuracy Comparison

Positive gain means lower candidate absolute error relative to the dense
reference. Intervals are paired six-seed Student-t 95% intervals.

| Quantity | Candidate mean error | Baseline mean error | Candidate MAE | Baseline MAE | Mean gain | 95% gain interval | Closer seeds | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Value | +0.0451053 | +0.0457456 | 0.0575812 | 0.0580129 | +0.0004317 | [-0.0001289,+0.0009923] | 5/6 | Descriptive candidate advantage; unsupported |
| `z_gamma` score | -0.0027468 | -0.0018383 | 0.0926298 | 0.0924406 | -0.0001891 | [-0.0012547,+0.0008765] | 3/6 | Descriptive baseline advantage; unsupported |
| `log_beta` score | -0.1417694 | -0.1442818 | 0.1499264 | 0.1508219 | +0.0008955 | [-0.0018365,+0.0036275] | 4/6 | Descriptive candidate advantage; unsupported |

Candidate/reference error ratios are `1.50 MCSE` (value), `0.056 MCSE`
(`z_gamma`), and `2.05 MCSE` (`log_beta`). These are descriptive finite-
particle errors, not unbiasedness tests.

## Decision And Inference Status

| Decision | Criterion | Veto | Uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Retain actual-SV route as tested opt-in | All hard validity/reference/GPU/identity gates pass | Clear | One point, six seeds | Preserve route; expand only under a new plan | No default/HMC/posterior readiness |
| Claim improvement over empirical Contract E | Gain interval excludes zero | Not met for any quantity | Six-seed paired power | Do not rank; use more seeds/scopes if needed | No superiority |
| Treat route as feasible | All six pairs finite and valid | Clear | Accuracy remains finite-particle limited | Descriptive feasibility only | No exact filtering/scientific validity |
| Run KSC-SV | Same exact target required | Not run; target differs | KSC is a surrogate | Separate campaign only if requested | No KSC evidence |

| Inference item | Status |
|---|---|
| Hard veto screen | Pass after balance repair; attempt 01 preserved |
| Statistically supported ranking | None |
| Descriptive differences | Value and `log_beta` MAE slightly favor candidate; `z_gamma` slightly favors baseline |
| Default readiness | Not established |
| Next evidence | More paired seeds and/or parameter/data scopes if improvement is required |

## Run Manifest And Red Team

Git commit: `fb9a0679adb7c731ff2ac42551f39bdcc15222a1` with dirty source hashes.
Data: `source_order_exact_transformed_sv_seed_83120_y1_y20_v1`.
Seeds: calibration `83900`, validation `83901`, claims `83910..83915`.
Output root:
`docs/benchmarks/artifacts/zhao_cui_moment_teacher_actual_sv_20260731/`.

Strongest alternative explanation is that both methods are similarly noisy
finite-particle estimators and six seeds lack power to detect small gains. The
scalar one-point result cannot establish high-dimensional scaling, KSC-SV
behavior, HMC readiness, posterior correctness, or source-faithful complete
Zhao-Cui filtering. A future prespecified multi-seed or multi-scope interval
excluding zero would overturn the current no-improvement verdict without
changing the present feasibility result.
