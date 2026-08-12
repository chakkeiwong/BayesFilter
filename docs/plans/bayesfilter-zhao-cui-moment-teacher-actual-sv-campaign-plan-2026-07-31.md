# Zhao-Cui Moment-Teacher Actual-SV Campaign

Date: 2026-07-31
Status: executed; terminal pass with no statistically supported improvement
Route under test: `zhao_cui_moment_teacher_gpu_fp32_no_tf32_xla_v1`
Target classification: exact transformed SV is `fixed_hmc_adaptation`; the
moment-teacher/Contract-E composition is `extension_or_invention`
Parent result:
`docs/plans/bayesfilter-zhao-cui-moment-teacher-integration-campaign-result-2026-07-31.md`

## Research Intent Ledger

| Item | Predeclared answer |
|---|---|
| Main question | Is the integrated squared-TT moment teacher valid and useful on exact transformed stochastic volatility at `T=20`, `N=1024`? |
| Candidate | Exact transformed-SV bootstrap particle likelihood, canonical streaming Contract-E-Chol, and independently fitted squared-TT shape targets. |
| Baseline | The same prepared particle program with empirical-target Contract E and no TT target substitution. |
| Accuracy authority | Converged sequential FP64 TensorFlow dense-grid value and autodiff score for the same exact transformed-SV target. |
| Expected failure mode | Scalar TT chart/scale failure, incomplete parameter tangent, route-identity mismatch, dense-reference nonconvergence, or no accuracy gain over empirical Contract E. |
| Promotion criterion | Deterministic derivative/identity tests and all six untouched GPU/XLA claim rows pass hard validity gates. |
| Promotion veto | Wrong target/event order, invalid dense reference, non-finite value/score, derivative mismatch, invalid reset, tuning-scope mismatch, TF32 enabled, missing XLA control flow, host callback, or failed mean/covariance restoration. |
| Continuation veto | Invalid target/harness, inability to issue route identity, exhausted repair budget, or all tuning candidates invalid after bounded repairs. |
| Repair trigger | Local shape, trace, chart, tuning, identity, serialization, or XLA failure with target and budget unchanged. |
| Explanatory diagnostics | TT residuals, correction residuals, paired displacements, runtime, allocator peak, and signs of errors. |
| Nonclaims | No KSC-SV result, source-faithful Zhao-Cui filter, HMC readiness, posterior correctness, high-dimensional scalability, default change, or broad SV validity. |

## Exact Target And Source Anchors

The source model is

\[
X_0\sim N\!\left(0,(1-\gamma^2)^{-1}\right),\quad
X_t=\gamma X_{t-1}+\eta_t,\quad
Y_t=\beta\exp(X_t/2)\epsilon_t,
\]

with `sigma=1`, `gamma=0.6`, and `beta=0.4`. The executed unconstrained
coordinates are `z_gamma=Phi^{-1}(gamma)` and `log_beta`. The fixed event order
is `x0 -> transition -> y1`.

Checked anchors:

- Zhao and Cui paper text
  `.localresources/papers/zhao-cui-tensor-train-sequential-learning-jmlr-2024.txt`:
  the adjacent recursion and squared-TT integration around lines 693--719, and
  the SV setup around lines 1994--2028;
- author source
  `third_party/audit/zhao_cui_tensor_ssm_p10/source/models/full_sol.m:21`
  and `:132`, which push the state before constructing transition/likelihood
  adjacent density;
- author transition and observation equations
  `third_party/audit/zhao_cui_tensor_ssm_p10/source/models/sv/transition.m:5`
  and `third_party/audit/zhao_cui_tensor_ssm_p10/source/models/sv/like.m:4`;
- author stationary initial state
  `third_party/audit/zhao_cui_tensor_ssm_p10/source/models/sv/priorpdf.m:8`; and
- author squared-TT marginal contraction
  `third_party/audit/zhao_cui_tensor_ssm_p10/source/deep-tensor.dev/src/@TTSIRT/marginalise.m:25`.

The paper/author implementation also infer parameters jointly in the TT. This
campaign instead freezes the parameter coordinate for each finite value/score
evaluation and differentiates the complete fixed program, which is a
`fixed_hmc_adaptation`. Supplying TT moments to Contract E is not in those
sources and remains `extension_or_invention`.

The primary observation target is exact `z=log(y^2)` with exact log-chi-square
noise. KSC-SV uses a seven-component Gaussian mixture and is a different target;
it is excluded from the primary comparison. The transform Jacobian depends only
on observed `y`, not `(gamma,beta)`, so transformed and raw-data parameter
scores are equal. Values are reported on the transformed target only.

## Evidence Contract

The exact finite candidate scalar is the sum of normalized particle-weight
increments. The TT normalizer never enters that scalar. The TT lane can affect
future increments only through the explicit higher-moment correction and
carried corrected cloud. The analytical score differentiates that same finite
program, including the stationary initial distribution, transition, exact
log-chi-square observation density, TT fitting/marginal recursion, streaming
transport, and Contract-E-Chol correction.

The baseline consumes identical observations, innovations, residual designs,
transport settings, and particle count but uses empirical-target Contract E.
The independent reference is FP64 sequential dense quadrature for the exact
same transformed scalar. Reference refinement uses `(order,radius)` arms
`(257,8)`, `(401,8)`, and `(401,10)`. It must have maximum value disagreement
`<=5e-5`, maximum score disagreement `<=2e-4`, and score-increment sum residual
`<=1e-10` before it can be used as an accuracy authority.

For each of value, `z_gamma` score, and `log_beta` score, define per-seed
absolute-error gain as

\[
G_s=|E^{\rm baseline}_s|-|E^{\rm candidate}_s|.
\]

Positive gain favors the candidate. With six paired untouched seeds, a ranking
is supported only when the two-sided 95% Student-t interval for mean gain
excludes zero. Otherwise differences are descriptive. This uncertainty rule is
secondary to hard validity vetoes.

Artifacts use unique directories under
`docs/benchmarks/artifacts/zhao_cui_moment_teacher_actual_sv_20260731/` and
record exact commands, source hashes, Git commit, environment, seeds, route and
tuning identities, GPU/memory policy, dtype, TF32/XLA status, and wall time.

## Default And Assumption Audit

| Choice | Provenance/status | Justification | Failure mode and early diagnostic |
|---|---|---|---|
| Actual transformed SV, not KSC | Exact local model and prior P41 audit; primary target | Avoids changing the likelihood to a Gaussian-mixture surrogate | Dense exact-target refinement and target hashes |
| `T=20` | New bounded feasibility scope | Matches predator-prey horizon and keeps first SV run affordable | Run deterministic `T=2` mechanics before GPU campaign |
| `N=1024`, `K=1024` | Existing nonlinear campaign and chunk policy | Non-fixture particle count with exact one-block transport | Configuration-time chunk validation |
| FP32, TF32 off, GPU/XLA | Selected moment-teacher route | TF32 previously failed lane parity for modest speed gain | Device/graph gate fails closed |
| Six claim seeds | Existing LGSSM seed policy; feasibility-level statistical arm | Allows paired sign and uncertainty analysis | Report wide intervals rather than ranking |
| Two charts and two control arms | Bounded tuning hypotheses, not defaults | Smallest scope-specific ladder consistent with prior nonlinear harness | Calibration/validation validity and heldout skew residual |
| Dense-grid reference | Independent TensorFlow diagnostic authority | Scalar SV permits a converged reference unavailable for predator-prey | Three-arm refinement gate |

No LGSSM, predator-prey, KSC-SV, or prior exact-SV control is silently treated
as tuned for this scope.

## Skeptical Plan Audit

The audit checked wrong baselines, proxy promotion, missing stop conditions,
unfair comparisons, silent defaults, stale event order, environment mismatch,
and whether the artifacts answer the stated question.

Findings and repairs:

1. Adding one `sv` row to the nonlinear feasibility harness would establish
   feasibility only, not improvement. The campaign therefore uses six paired
   untouched seeds and a predeclared paired uncertainty analysis.
2. KSC-SV has convenient conditional-Gaussian comparators but changes the exact
   observation target. Exact transformed SV is primary; KSC is excluded.
3. An older local exact-SV data helper observes `x0` at its first index, while
   this route and the author source use `x0 -> x1 -> y1`. This campaign creates
   a source-order dataset explicitly and hashes it.
4. A finite particle score can agree with its own centered difference while
   both candidate and baseline are inaccurate. The converged dense reference
   is therefore required before interpreting accuracy.
5. Dense quadrature is diagnostic FP64 TensorFlow and not a candidate runtime
   path. It cannot establish GPU/XLA or production readiness.
6. Six seeds provide weak ranking power. Only a paired interval excluding zero
   supports improvement; all other continuous differences remain descriptive.
7. The scalar SV result cannot establish the high-dimensional motivation of the
   TT teacher.

Audit verdict: `PASS_FOR_STAGED_EXECUTION`.

### Execution Amendment After Attempt 01

GPU attempt 01 initialized the trusted RTX 4080 SUPER, verified memory growth,
disabled TF32, compiled with XLA, and passed the dense reference. Both declared
TT arms passed validation seed `83901` but both the candidate and empirical
Contract-E baseline were invalid on calibration seed `83900`. No claim seed was
evaluated. Artifact:
`docs/benchmarks/artifacts/zhao_cui_moment_teacher_actual_sv_20260731/gpu_attempt01/result.json`,
SHA-256 `7a44423c7f033f29e4a947e25e7bff306b2740fdac496be43d33b33e02ef3afc`.

Because the invalidity is shared by teacher and no-teacher arms across both
charts, it is a transport/reset tuning repair trigger rather than evidence
against the TT representation. The inherited `balance_steps=8` was a
convenience transfer from predator-prey and was not independently varied in the
initial actual-SV grid. Repair attempt 02 uses the two remaining
target-preserving candidate slots: it repeats candidate 0 with
`balance_steps=20` and candidate 1 with `balance_steps=32`. All data, charts,
other controls, selection criteria, claim seeds, backend, target, and budget
remain unchanged. Component validity flags are added to the artifact for
failure classification only.

Amendment verdict: `PASS_FOR_LOCALIZED_REPAIR_ATTEMPT_02`.

## Execution Phases

1. Add a fixed exact transformed-SV wrapper and repository-issued route
   identity. Add tests for event order, analytical score parity, tuning-scope
   isolation, identity binding, and cross-model substitution rejection.
2. Generate one frozen source-order synthetic dataset at physical
   `(gamma,beta,sigma)=(0.6,0.4,1)` and `T=20`.
3. Run CPU-hidden deterministic `T=2` tests and the FP64 dense refinement.
4. Tune two declared chart/control arms on particle seeds `83900` and `83901`,
   select by hard validity then heldout skew residual, and freeze the artifact.
5. Run paired candidate/baseline claims on untouched seeds `83910..83915` with
   `N=1024` on trusted GPU/XLA.
6. Compute paired error gains against the frozen dense reference and write a
   terminal result with decision, inference-status, run-manifest, and post-run
   red-team tables.

## Budget And Stop Conditions

- One implementation attempt plus two localized repair attempts.
- At most three trusted GPU launches and two GPU-hours total.
- At most two tuning candidates in the initial ladder. A localized repair may
  add at most two target-preserving candidates without changing claim data.
- Dense reference expected under five minutes; stop if the refinement gate
  fails after one numerical repair.
- Never overwrite an existing artifact directory.
- Stop before claim execution on wrong event order, derivative mismatch,
  invalid reference, tuning hard veto, identity failure, memory-policy failure,
  persistent XLA failure, or exhausted budget.

## Planned Commands

CPU-hidden checks:

```bash
CUDA_VISIBLE_DEVICES=-1 python -m pytest -q \
  tests/highdim/test_zhao_cui_moment_teacher_actual_sv.py \
  tests/highdim/test_zhao_cui_moment_teacher_nonlinear.py
```

Trusted GPU campaign:

```bash
TF_FORCE_GPU_ALLOW_GROWTH=true \
  /home/chakwong/anaconda3/bin/conda run -n tf-gpu \
  python docs/benchmarks/run_zhao_cui_moment_teacher_actual_sv.py \
  --output docs/benchmarks/artifacts/zhao_cui_moment_teacher_actual_sv_20260731/attempt01
```
