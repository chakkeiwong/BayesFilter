# Revised LEDH graceful-failure repair and validation program

**Status:** proposed program for Claude to adopt; implementation and campaign execution have not begun under this document.

**Owner request:** turn the completed Codex review into a detailed, executable program. **Starting revision:** `surrogate-hmc`, `8a5c23ab1172884ffca63cac390623bf7735afe7`, checked 2026-09-17. All sources recorded in the [review manifest](artifacts/ledh-graceful-failure-review-20260917-01/review-manifest.json) were unchanged when this program was prepared.

**Primary evidence:** [Codex technical review](artifacts/ledh-graceful-failure-codex-review.md), including its deterministic CPU/GPU reproductions. **Original proposal:** [comprehensive testing plan](ledh-graceful-failure-comprehensive-testing-plan.md). Once adopted, this document replaces the original proposal's active execution instructions; retain the original and its results as historical records.

The objective is to make numerical failures observable and safely contain them through the actual canonical LEDH consumer, preserve healthy calculations, and determine which filtering and posterior claims survive independent validation. Completing a run, returning finite retained samples, and passing a toy rejection test are not sufficient evidence.

## 1. Outcomes, scope, and authority

Claude should work toward three separately reported outcomes:

1. **Engineering containment:** supported numerical failures produce the declared status and finite force convention at every consumer; a valid starting chain can handle them without silently producing NaNs, changing output shape, or corrupting another batch member. Healthy accepted calculations are preserved.
2. **Mathematical and numerical correctness:** the analytical score differentiates the declared finite value program through initialization, model parameters, filtering, transport, and reset. The implementation uses the stated model equations and the eligible canonical route.
3. **Statistical validation by scope:** filtering agrees with an appropriate reference within a predeclared error allowance, and HMC agrees with the posterior of its stated implemented target at adequate Monte Carlo precision. Approximation to the exact state-space posterior is assessed separately.

Do not collapse these outcomes into one “production ready” flag. Report each model, parameterization, horizon, particle count, dtype, backend, and target/force pairing explicitly. A completed investigation may conclude that a candidate failed or a scope is under-budgeted; that is not a successful validation of the scope.

This request commissions the proposal, not a running campaign or an external message to Claude. A later plain-language direction to execute this program within its stated budget is sufficient campaign authorization. Once execution is authorized, routine repairs and retries within the unchanged scientific contract and budget do not require renewed permission. Do not introduce approval tokens, hash-bound authorization phrases, or a review chain for every phase.

No new LEDH algorithm, reduced lane, raw-barycentric fallback, or autodiff score route is proposed. The only claim-bearing algorithm remains the owner's canonical LEDH-PF-PF OT route with Contract E--Chol, the UKF per-particle covariance lifecycle, GenUT dual-cap correction, and analytical recursive derivatives. Preserve canonical route identity and repository-issued tuning authority. Prior LEDH results invalidated by the 2026-08-21 policy cannot serve as baselines or warm starts.

The governing algorithm sources are the [canonical chapter](../chapters/ch19c_dpf_implementation_literature.tex), [canonical rebuild plan](bayesfilter-ledh-canonical-rebuild-plan-2026-08-21.md), and [conformance plan](bayesfilter-ledh-conformance-test-plan-2026-08-21.md), read under the current owner policies and [results invalidation notice](bayesfilter-ledh-results-invalidation-notice-2026-08-21.md). Before editing a stage, inspect its mathematical specification and the live executable conformance checks. Older implementation-status statements in those plans are historical; verify the current call chain rather than inheriting them as facts.

## 2. Research intent and evidence contract

| Item | Binding interpretation |
|---|---|
| Main question | Does the repaired failure handling preserve the healthy canonical computation and contain invalid evaluations through the real consumer, and what target does a chain actually sample when failures occur? |
| Mechanism under test | Explicit validity/status propagation, safe inactive arithmetic, persistent trajectory failure state, and a deterministic finite force extension where needed. |
| Expected failure modes | Lost `-inf` sentinel, invalid factor derivatives, nonfinite primal/score mismatch, status reduction across unrelated batch members, swallowed configuration errors, wrong initialization derivatives, and numerical exclusion of valid posterior support. |
| Engineering comparator | The same current canonical mathematics on inputs known to be healthy. Use a minimal test-only guard-bypass comparison or an independent expression of the same operations; do not revive an old algorithmic lane. |
| Mathematical comparator | Closed-form identities, exact Kalman filtering, and independent directional derivatives of the same finite program. |
| Statistical comparator | A controlled reference posterior for the same finite likelihood, plus a separate exact/independent state-space reference where available. |
| Primary safety criterion | No alteration of healthy results beyond the declared floating-point allowance; deterministic, bounded, flagged behavior for supported failure fixtures. |
| Promotion vetoes | Wrong target or derivative, unexplained nonfinite consumer output, failed healthy equivalence, missing exact scope tuning, inadequate reference, unresolved significant target exclusion, failed retained-chain diagnostics, or uncleared heuristic comparisons for comparative claims. |
| Continuation vetoes | Corrupted artifacts, unavailable required source/target definition, unsupported execution contract, an invalid reference on which the next phase depends, exhausted compute/attempt budget, or a nonlocal resource/privacy/cost change. |
| Repair triggers | A failed candidate, a malformed test fixture, a localized infrastructure error, failed calibration, or an expected numerical pathology covered by a later repair stage. Preserve the failed evidence and continue the applicable repair. |
| Explanatory diagnostics | Failure counts and locations, covariance margins/condition indicators, acceptance, ordinary energy-error distributions, runtime, compilation, memory, and descriptive ESS differences. Severe energy/nonfinite events are vetoes as specified in Phase 6; retained R-hat/ESS/MCSE requirements are admission criteria, not optional explanation. |
| Conclusions prohibited | “All failures are harmless,” “no crash proves correct HMC,” “zero NaNs in retained samples proves finite proposals,” “stationarity proves identification,” or “seven passing cases validate an eighth failing one.” |
| Preserved result | Versioned JSON decisions, exact inputs and samples, source/environment provenance, full logs, and a Markdown result note with the decision and inference-status tables in §13. |

**Skeptical audit of the revised design.** The original baseline is invalid because its four-state matrix is unstable, parameter inference is unspecified, and the real failure sentinel is broken. This program repairs those prerequisites before research interpretation. It replaces proxy promotion criteria with direct consumer tests and reference comparisons; separates mathematical support from evaluator failure; specifies proper priors and initialization derivatives; treats CPU/float64 work as reference evidence; and makes budgets and phase dependencies explicit. Remaining model adapters, numerical thresholds, tuning feasibility, and runtime estimates are hypotheses to resolve in the named early stages, not assumed capabilities.

## 3. Define the failure semantics before editing the implementation

### 3.1 Three targets must remain distinguishable

Use distinct names and identifiers for:

- `model_posterior`: the prior times the exact state-space marginal likelihood, when that likelihood is available or has a controlled independent approximation.
- `finite_program_posterior`: the prior times one fixed, deterministic canonical LEDH likelihood program, including its fixed particles/base randomness and numerical settings.
- `masked_finite_program_posterior`: that finite program additionally restricted to the region where its value evaluator succeeds.

An endpoint value labeled “exact” by the dual target means exact for its chosen executed value configuration; it does not mean exact nonlinear marginalization. A numerical failure at a mathematically valid point can change the implemented posterior. With failed region `A`, conditioning a target on `A^c` changes its total-variation distance by exactly `pi(A)`. Failure frequency among proposals does not directly estimate that mass.

For clarity, if the unmasked density is `p` and `a=pi(A)<1`, the masked density is `p 1_(A^c)/(1-a)`. Half the integrated absolute difference is `(a+a)/2=a`: one contribution is the removed mass on `A`, the other is the redistributed mass on its complement. When `a=1`, the masked posterior is not normalizable. This derivation presupposes a defined unmasked target; if it cannot be evaluated or characterized, the exclusion assessment is unresolved.

### 3.2 Required value, force, and status behavior

The internal evaluation should expose `value`, `force`, and tensor-valued diagnostics. Preserve existing public call signatures where possible; add a supported diagnostic/evaluation method rather than silently changing tuple arity for uninspected consumers. At minimum the diagnostics need:

```text
value_valid                 [...batch] bool
force_valid                 [...batch] bool
outside_declared_support    [...batch] bool
value_failure_code          [...batch] integer
force_failure_code          [...batch] integer
first_failure_time          [...batch] integer, -1 when absent
first_failure_stage         [...batch] integer
first_failure_particle      [...batch] integer, -1 when inapplicable
```

Use a small stable integer reason registry inside compiled kernels: nonfinite input, failed factor, invalid solve/tangent, reset/correction failure, model callback failure, or nonfinite proposal state. Strings, provenance, and serialization stay outside the numerical graph. Keep reason counts even if multiple failures follow the first. Do not let a caller stamp canonical identity through this diagnostic structure.

| Situation | Required behavior |
|---|---|
| Healthy value and force | Return the original finite value and analytical score/declared proposal field; all relevant status flags healthy. |
| Point outside the declared mathematical support | Return `-inf` and a finite zero force convention with `outside_declared_support=true`. This is distinct from evaluator failure. |
| Value evaluator fails numerically at an otherwise allowed point | Return `-inf`, finite zero force, and a numerical-value-failure reason. Containment may pass, but posterior equivalence is unvalidated until the excluded region is assessed. |
| Endpoint value is valid, biased-force evaluator fails | Preserve the finite endpoint value and flag the force failure. Use the deterministic zero extension described below only after its mechanics are checked. Do not set the valid endpoint value to `-inf` merely because a cheaper force failed. |
| Nonfinite proposed coordinates or momenta | Reject/terminate the invalid trajectory through a declared integrator-health path and retain a valid current state; record the event. It vetoes scientific admission until its cause is repaired. Do not call this ordinary support rejection. |
| Invalid chain initialization | Fail before sampling with a readable diagnostic. Starting at `-inf` is not a graceful-rejection test. |
| Wrong shape, unsupported configuration, stale tuning, or programming error | Raise/fail before the numerical kernel when possible. Do not convert these errors into innocuous rejected proposals. |

If the current code cannot distinguish value validity from force validity, implement and test that distinction before declaring the table satisfied. In particular, a directional derivative failure must not automatically invalidate an otherwise successfully computed primal value. Where a shared failed operation also invalidates the primal, report both.

The proposed finite force extension is `g_ext(theta)=g(theta)` where the requested force succeeds and zero otherwise. It changes the proposal on failed-force points, while keeping a valid endpoint target intact. Its rationale is explicit: a kick `(q,p) -> (q,p+a g_ext(q))` is a translation of momentum at fixed position, and a drift `(q,p) -> (q+a M^-1 p,p)` is a translation of position at fixed momentum. Their inverses use `-a`; a symmetric leapfrog composition is reversible with momentum reversal and volume-preserving for a fixed mass and a deterministic, position-only finite field. The endpoint Metropolis correction must use the complete potential-plus-kinetic difference. This argument does not cover stateful/random fallback fields, asymmetric early stopping, nonfinite arithmetic, or adaptation during retained sampling.

Thus an intermediate force failure is not automatically an endpoint support violation. A trajectory returning to a finite endpoint may be accepted under the checked extended-field proposal. If Claude chooses an abort rule instead, it must establish forward/reverse symmetry for that rule before using it. The zero extension is the recommended bounded repair; it needs a dedicated healthy-equivalence and reversal test, not a search for better ESS.

### 3.3 Batch and time semantics

Maintain separate status for each independent parameter/chain row. Reduce over particles or tangent directions only when they belong to one evaluation and the mathematical contract requires all of them. Never `reduce_all` across unrelated chains and thereby invalidate a healthy row because its neighbor failed.

Carry cumulative validity in the filter loop:

```text
value_alive_next = value_alive AND step_value_valid
force_alive_next = force_alive AND step_force_valid
```

Once an evaluation becomes invalid, later finite arithmetic must not revive it. Record the first failing time/stage. Skip the remaining inactive work or give it finite inert inputs, while preserving status. Mask final consumer outputs after the loop as well as at the stage boundary. Mixed-validity batches must preserve the outputs of healthy rows.

Do not feed a zero Cholesky factor into a triangular solve. Use an identity factor and zero tangent only on an already invalid inactive branch, or skip the solve with a bounded TensorFlow conditional. This is not a ridge or alteration of an accepted covariance. Do not catch arbitrary exceptions, drop particles, renormalize away failures, or retry with a different model/reset/backend inside HMC.

## 4. Phase 0 — establish the active state and truthful status

**Purpose:** eliminate stale context and preserve a valid comparison before repairs.

1. Read the review, applicable `AGENTS.md`/`CLAUDE.md`, and the current checkpoint. Verify checkout, branch, HEAD, and relevant dirty files. Preserve unrelated work. If the reviewed sources changed, inspect only the changes affecting the cited defects.
2. Create a unique campaign directory `docs/plans/artifacts/ledh-graceful-failure-<UTC timestamp>-<attempt>/`. Save the source commit and ordinary hashes of relevant dirty sources, environment, exact commands, and this program's path. Never overwrite the review's evidence directory.
3. Amend the active status/completion documents to say that sentinel propagation is broken and broader validation is pending. Keep the old result counts dated. Link the review and this adopted program; do not silently rewrite past results as though the repair had already passed.
4. Record an execution endpoint ledger: direct canonical value/score, both batch implementations, dual target, graph callable, actual HMC runner/tuner binding, and each selected model adapter. For each, identify callable, tensor ranks, dtype, device, XLA mode, derivative mode, reset identity, and validity output.
5. Preserve a small healthy fixture from the current eligible algorithm with fixed inputs and all canonical features exercised. If an existing fixture disables required features, label it primitive-only and add the full-feature fixture. Do not use pre-invalidation results as the baseline.

**Exit evidence:** active status corrected; reviewed defect IDs mapped to source paths and consumer endpoints; clean comparison inputs preserved; budget ledger initialized. **Failure response:** resolve source/checkout ambiguity locally; do not launch a campaign whose target cannot be identified.

## 5. Phase 1 — repair containment through the complete call chain

Work in the dependency order below. Existing names are anchors, not instructions to duplicate their logic in new modules.

| Work item | Current source | Required implementation and executable check |
|---|---|---|
| P1.1: factor validity | `bayesfilter/highdim/ledh_numerical_safety_tf.py` | Preserve leading batch shape; reject nonfinite input/factor and nonpositive factor diagonal under the declared covariance contract. Include upper-triangle corruption when the input is a full symmetric covariance. A lower-triangle-only API, if retained, must explicitly say so. |
| P1.2: safe tangent arithmetic | `ledh_canonical_score_stages_tf.py`, `ledh_unified_reset_tf.py` | Guard triangular solves and Cholesky differentials before invalid factors enter them. Preserve value and tangent status independently. Exercise reset gap, target, injected covariance, and higher-moment correction. |
| P1.3: numerical operation inventory | `ledh_canonical_score_tf.py` and the same stage/reset modules | Audit process/observation factors, UKF innovation, flow innovation, log determinants, normalization/division, and callback results. Each operation must either be validated configuration or emit a checked proposal-dependent status. Replace proposal-dependent assertions with status handling; retain genuine configuration assertions. |
| P1.4: persistent validity | Canonical time loop | Carry cumulative value/force status; test failures at first, middle, and final observations. No NaN arithmetic may leak from an inactive branch into a healthy batch row. |
| P1.5: sentinel preservation | Both paths in `ledh_canonical_batch_fused_tf.py` | Compare directional primal values only on finite rows and check status consistency separately. Remove the unconditional nonfinite-to-NaN conversion. Test both implementations through real wrapper calls. |
| P1.6: consumer semantics | `bayesfilter/inference/ledh_dual_parameter_target.py` | Consume statuses; distinguish exact endpoint and biased-force failures; expose diagnostics to the HMC binding. Cover scalar/batch custom-gradient shapes and the force-extension policy. |
| P1.7: compilation/shape defects | Graph callable and enclosing runner | Give repeated kernels stable signatures and verified XLA execution. Localize the reviewed empty CPU graph outputs and ordinary GPU graph timeout. Fix or explicitly classify a reference-mode limitation; no silently empty return values. |

Use meaningful tests, not assertions that merely mirror a new helper's implementation. Required tests include these independently observable properties:

- A healthy covariance reconstructs from its factor; a known-invalid covariance does not reach an unsafe downstream solve; validity shape equals the input's leading batch shape, including `[2,1]`.
- An injected inner `(-inf, zero score)` passes through the **real** batch wrapper unchanged. A second test runs the **unmocked** canonical engine and dual consumer with a deliberate internal failure fixture. Neither test may mock the wrapper whose behavior it claims to verify.
- Healthy/invalid rows evaluated together agree, row by row, with separate evaluations. A row's status cannot depend on unrelated batch composition or direction ordering.
- An earlier failure cannot be overwritten by a later healthy time step. Final value/force/status match the declared policy regardless of failure position.
- A finite exact endpoint with failed biased force retains that value and exposes the force failure. A NaN is never relabeled as successful support rejection.
- Configuration errors remain actionable errors before sampling. Low-level injected negative covariance is a mechanics fixture; a statically negative model covariance should be rejected at preflight.

**Mandatory matrix of execution modes:** primitive checks in CPU float64 eager/graph/XLA and GPU float32/float64 XLA; complete-consumer healthy/failure checks in GPU float32 with TF32 enabled and GPU float64 reference. Use CPU graph mode only as an explicitly labeled diagnostic. Any GPU command requires trusted/escalated access and pre-initialization memory growth. Record actual GPU identity rather than assuming CUDA and `nvidia-smi` ordinals coincide.

**Exit evidence:** all R1/R2/R6 containment reproductions have corresponding passing regression evidence; malformed integration fixtures are fixed; every active endpoint preserves status/shape; default GPU/XLA execution is confirmed; healthy equivalence passes. If an ordinary-graph diagnostic remains unsupported, its API must fail clearly and its scope remain unvalidated. Do not use that exception to excuse an unresolved default GPU/XLA failure.

## 6. Phase 2 — establish the parameter-to-value-to-score contract

The scalar canonical function already accepts initial-state and initial-covariance tangents; that does not prove the fused consumer forwards them. Extend the shared model/consumer interface where needed, without implementing a second reduced algorithm.

For every inferred parameter, complete this dependency audit and an executable directional test:

| Dependency | Required derivative |
|---|---|
| Initial mean/covariance | `dm0`, `dP0`, and the derivative of the fixed-base initial cloud |
| Transition mean/Jacobian | Direct parameter terms plus state-chain terms |
| Process covariance | `dQ`, including any parameterization chain rule |
| Observation mean/Jacobian/density | All direct parameter and state terms; use the actual density for SV |
| Observation covariance | `dR`, including scale-to-covariance factors such as `2 exp(2 theta)` |
| Transport and Contract E reset | Direct source moments/weights and transported-cloud terms, plus declared residual-design/ridge dependence |
| Prior and coordinates | Physical prior, transformation Jacobian, and their analytical derivatives |

If a callback cannot represent one of these terms, fail the model's admission until the shared interface is repaired. Do not freeze a parameter-dependent quantity to make a missing derivative disappear while retaining the original target label.

For a stable LGSSM, implement stationary covariance with TensorFlow:

\[
(I-F\otimes F)\operatorname{vec}(P_0)=\operatorname{vec}(Q),
\]

and its directional derivative with the same operator:

\[
dP_0-F(dP_0)F^\top
=(dF)P_0F^\top+FP_0(dF)^\top+dQ.
\]

Use a solve, not a matrix inverse. Dimension 20 gives a `400×400` system, suitable for an initial bounded implementation. Check stability and positive definiteness separately from the equation residual. The original unstable case has a tiny residual and a negative covariance eigenvalue; preserve it as a negative test.

For fixed base draws `z_i`, construct `x0_i=m0+L0 z_i` and propagate `dx0_i=dm0+(dL0)z_i`. Define separately what each particle's UKF covariance represents; do not assume that the global initial-cloud covariance and every conditional particle covariance can be interchanged. Follow the canonical initialization/lifecycle specification and test its moments. Fixed random inputs are part of the finite target identity.

Compare analytical directional scores with central differences over a predeclared step ladder on well-conditioned interior points, and with independent diagnostic autodiff where useful. Neither diagnostic becomes the claim-bearing score. Check several coordinate and mixed directions, not only one scalar parameter. Detect piecewise branch/cap crossings and use declared one-sided checks there; do not silently exclude failing points after seeing the result. Check value invariance across directions and parameter/batch permutations.

The default model/runtime path must remain TensorFlow/TFP and GPU/XLA. Existing NumPy imports or float64-only adapters are migration debt on any newly admitted path. Migrate only the dependencies needed by the chosen consumer; keep independent NumPy/SciPy references in explicitly diagnostic modules. Do not introduce pfor through Jacobian helpers without the repository's required approval.

**Exit evidence:** a dependency table for every proposed inference case, actual-endpoint wiring tests, analytical directional agreement on the same finite program, stationary covariance/value checks, and correctly classified missing capabilities. R5 is not closed by a function's existence alone.

## 7. Phase 3 — fix the model specifications and construct the reference problems

All specifications below are proposed **synthetic validation fixtures**, not universal model defaults or claims about real data. Their bounded priors deliberately make the benchmark target explicit. Do not narrow a prior after encountering failures or select a dataset because its posterior is easy. Validate the entire declared support, including its difficult regions.

For all LGSSMs: observe `y_t` after the transition from `x_(t-1)` to `x_t`, for `t=1,...,T`; use `m0=0` and the stationary `P0(theta)`; fix offsets to zero. `Q` and `R` below are covariances. The base `LinearGaussianSSM` has no inferred parameters, so these inference fixtures need a parameter-to-specification adapter into the existing general engine, with the complete analytical dependencies from Phase 2.

| Case | Exact synthetic specification | Parameters and proper prior in physical/model coordinates | Validation scope |
|---|---|---|---|
| LG1 | `d=1,T=10,F=phi,Q=1,H=1,R=0.5`; truth `phi=0.8` | `phi ~ Uniform(-0.95,0.95)` | Scalar finite-program and exact-Kalman posterior; stationary-initialization derivative; filtering. |
| LG2 | `d=2,T=20,F=a A`, `A=[[0.7,0.2],[0.1,0.6]]`; `Q=exp(2b)[[1,0.3],[0.3,0.8]]`, `H=I2,R=0.5 I2`; truth `(a,b)=(1,0)` | Independent `a ~ Uniform(0.5,1.2)`, `b ~ Uniform(-1,1)` | Two-parameter posterior, covariance-scale derivative, filtering. |
| LG4 | `d=4,T=50,F=a A4`; `Ad` has diagonal `0.6`, adjacent off-diagonals `0.1`, other entries zero; `Q=exp(2b) I4`; `H` selects coordinates 1 and 2; `R=0.5 I2`; truth `(1,0)` | Same two physical-coordinate priors as LG2 | Partially observed filtering and two-parameter posterior; no all-matrix identification claim. |
| LG20 | `d=20,T=120,F=a A20`; `Q=exp(2b) diag(q_i)`, `q_i=0.5+1.5(i-1)/19`, `i=1,...,20`; `H` selects the first ten coordinates; `R=0.5 I10`; truth `(1,0)` | Same two priors as LG2 | Filtering/capacity and, when feasible within budget, the same two-parameter posterior. Unfinished posterior validation remains explicitly pending. |
| SV20 | `T=20`, existing basic SV density: `x_t=gamma x_(t-1)+epsilon_t`, `y_t given x_t ~ N(0,beta² exp(x_t))`; stationary initial law; truth `(gamma,beta)=(0.95,0.5)` | Model coordinates `u=Phi^-1(gamma) ~ Uniform(0,2.5)`, `b=log(beta) ~ Uniform(-2,0.5)`, independent; process scale fixed at 1 | Actual SV likelihood/score, filtering, and two-parameter finite-program posterior. |
| SV40 | Identical model/priors to SV20, `T=40`; independently generated data | Same as SV20 | A distinct tuning and validation scope; no automatic transfer of settings or claim that every posterior is tighter. |
| PP20 | `T=20`, corrected predator–prey equations below; initial mean `(50,5)`, covariance `I2`; `Q=4 I2`, `R=4 I2`, observe both coordinates; interval `2.0`, classical RK4 step `0.1`; truth `(0.6,114,25,0.3,0.5,0.5)` | First infer `r ~ Uniform(0.1,1.1)` and `v ~ Uniform(0,1)`, fixing `K=114,a=25,s=0.3,u=0.5` | Declared two-parameter diagnostic posterior and filtering; six-parameter readiness remains untested until the extension below. |
| SIR3 | `J=3,T=30`, chain graph `1--2--3`; fixed `kappa_j=0.5,nu_j=0.2`; initial mean `(50,5,50,5,50,5)`, covariance `I6`; `Q=I6`, `R=100 I3`; infectious coordinates observed; interval `0.02`, classical RK4 step `0.005` | No inferred parameters in `SpatialSIRSSM` | Filtering, partial observation, and numerical containment only; parameter ESS/recovery are inapplicable. |

For bounded continuous coordinates `p in (l,u)`, the proposed HMC coordinate is `p=l+(u-l)sigmoid(z)`. A uniform physical prior requires its density and the log-Jacobian in the target; together they give `log sigmoid(z)+log sigmoid(-z)` per coordinate. The derivative is `1-2 sigmoid(z)`. Apply this transformation to the listed model coordinates, not twice to derived quantities such as `gamma=Phi(u)` or `beta=exp(b)`. Record the full chain rule and Jacobian convention. Ensure the adapter uses these declared priors exactly once; do not accidentally multiply by an inherited model-class prior as well. Stable log-sigmoid arithmetic avoids an unnecessary overflow; it does not justify changing the prior support.

The LG2 eigenvalues are `0.8a,0.5a`, below 1 on the declared support. The symmetric tridiagonal matrices have eigenvalues in `[0.4,0.8]`, so multiplying by `a<=1.2` remains stable. Nonzero nearest-neighbor coupling makes the first-coordinate observability matrix full rank in exact arithmetic: its successive rows first reach successive coordinates with nonzero coefficients. Numerical observability can still be poorly conditioned; report singular values and a dtype-relative rank threshold. This does not establish practical identification from one short dataset.

For each inferred parameterization, inspect likelihood profiles, joint contours, sensitivities, and any exact symmetries using the declared reference. Report weakly informed directions and prior-dominated posteriors. A proper bounded prior makes a target normalizable under the appropriate likelihood assumptions; it does not prove the parameters are identified. Weak identification is not a failure of a sampler that correctly reproduces the reference, and apparent recovery from one synthetic truth is not an identification proof. Save latent synthetic states for evaluation only; the filtering/inference target receives the declared observations, not those latent answers.

The predator–prey model must use the local implementation:

\[
J(x,y)=\frac{xy}{a+x},\qquad
\dot x=rx(1-x/K)-sJ(x,y),\qquad
\dot y=uJ(x,y)-vy.
\]

The SIR model must use local infection and diffusion:

\[
\dot S_j=-\kappa_jS_jI_j+\tfrac12(\sum_{k\in N_j}S_k-d_jS_j),
\quad
\dot I_j=\kappa_jS_jI_j-\nu_jI_j+\tfrac12(\sum_{k\in N_j}I_k-d_jI_j).
\]

Preserve the existing `diagnose_negative_after_noise` domain policy for these Gaussian-noise fixtures. Record negative populations; do not resample observations, clip states, or truncate transitions to conceal them. These are local numerical benchmarks, not certified nonnegative population models. If the declared domain policy makes the intended likelihood undefined, classify that as a model-contract failure and resolve it before inference, rather than treating every negative state as a harmless rejected proposal.

For SV, supply the actual state-dependent observation density and its analytical derivatives. A mean `beta exp(x/2)` with additive fixed covariance is a different model. The zero observation mean also gives a zero ordinary mean-based UKF cross-covariance; establish the actual proposal construction and its relation to the canonical specification before admitting this adapter. A changed proposal/density must not be hidden behind the word “SV.” If canonical support is genuinely missing, record the capability gap and continue eligible independent LGSSM work; do not invent a reduced SV lane.

Optional extensions are separately scoped: full six-parameter predator–prey with the existing parameter-box priors; `GeneralizedSVPriorMeanSSM` to exercise distinct mean/scale dependencies; and the existing three-parameter SIR wrapper for log infection, recovery, and observation-noise scales. They cannot be substituted for a failed mandatory case or added beyond budget without direction. New Zhao–Cui source-route behavior needs paper and original-author-code anchors before implementation; local equation agreement alone is not source-faithfulness. Reuse inspected sources where available and store any decision-bearing paper locally.

**Exit evidence:** machine-readable specifications, parameter/prior/Jacobian checks, exact observation timing, domain policies, structural calculations, and callable/source anchors. There are eight primary filtering scopes, seven potential parameter-inference scopes, and no parameter-inference claim for fixed SIR.

## 8. Phase 4 — calibrate numerical checks and freeze each LEDH scope

### 8.1 Exercise failure deliberately, without requiring healthy small scales to fail

Use the following deterministic matrix families before larger model runs. Record the represented matrix, dtype, symmetry error, factor status, factor residual, and tangent status. A mathematical construction and its rounded representation can have different ranks.

| Family | Construction and question |
|---|---|
| Healthy scaling | `c I`, with `c` in `{1e-20,1e-10,1,1e10,1e20}` where subsequent arithmetic is representable. These matrices have condition number one; small scale alone must not be diagnosed as loss of positive definiteness. |
| Conditioning | `diag(1,10^(-k))`, `k in {0,4,8,12,16}`, plus a fixed orthogonal rotation. Distinguish factorization success, representability, and derivative amplification. |
| Cancellation | `[[1,1-delta],[1-delta,1]]` for `delta=0.01,32 eps,4 eps,0`, using the working dtype's machine epsilon. Check the actually represented eigenvalues; do not claim a universal failure threshold. |
| Invalid covariance | Zero, a negative diagonal entry, an indefinite symmetric matrix, and NaN or infinity independently in upper and lower triangles. Test nonsymmetric input under the declared full-covariance contract. |
| Shape | Unbatched, `[1]`, `[2,1]`, and mixed healthy/invalid leading batches. The validity tensor must retain every leading dimension. |
| Derivative | Zero and nonzero covariance tangents, including a valid primal with an independently invalid tangent. No invalid solve may contaminate an unrelated valid row. |
| Full filter | Inject a checked failure at the first, middle, and last observation, and in each proposal-dependent factor/reset stage. An early failure must remain visible at the final consumer. |

The infinite/nonfinite and invalid-diagonal cases are exact status tests. Conditioning families are diagnostics until a rejection margin is justified. Record dimensionless residuals, for example `||LL^T-P||_F / max(||P||_F,tiny)` with a documented underflow treatment, rather than using one absolute threshold for all scales. Conditioning diagnostics may warn before they become an acceptance rule. A scale-relative margin must also have a rounding-error or calibration argument; “relative” alone is not a justification.

Computed diagnostics belong in the result. Adopt pure observability and checks that leave accepted values unchanged after a healthy no-fire regression. Any ridge, damping, clipping, change to correction caps, or other alteration of accepted arithmetic is a separate numerical candidate. Give it a derivation or a measured bias-versus-robustness/trust curve, declare its target and tuning consequences, and evaluate non-harm. The choice of zero damping or no ridge needs the same justification. Do not reject a useful protection merely because it fails to improve a primary accuracy or speed metric.

### 8.2 Freeze numerical acceptance rules before untouched runs

Exact requirements are finite outputs where promised, correct status/reason/shape, exact zero force on the defined failure branch, and preservation of the `-inf` sentinel. Healthy runs should be bitwise equal when the guarded and baseline executions use the same compiled arithmetic. If compiler scheduling prevents bitwise equality, preserve the difference and use the following **proposed fixture tolerances**, subject to a documented calibration before any claim data are examined:

| Check | Proposed reference budget | Meaning and early calibration |
|---|---|---|
| Healthy guarded versus unguarded value | Absolute difference `1e-9` in float64; `1e-4` in float32 | For identical fixed inputs in the same backend/mode. Report any tolerance dependence on horizon or accumulated rounding; do not hide a systematic likelihood shift. |
| Healthy moments and analytical score | Scaled error `1e-8` in float64; `1e-4` in float32 | Scale each quantity using a declared physical scale or reference magnitude, with an explicit floor in those units. These are engineering budgets, not universal numerical laws. |
| Analytical score versus finite differences | A stable error plateau consistent with truncation and rounding; proposed scaled errors `1e-6` and `1e-2` for float64 and float32, respectively | Use coordinate and mixed unit directions. Start with `h=eps^(1/3) max(1,norm_2(theta))` and multipliers `{1/8,1/4,1/2,1,2,4,8}`. Inspect cancellation and branch changes. Float64 provides the sharper check; a loose float32 check alone cannot establish the score. |
| LGSSM filtering approximation | Whitened mean RMS error at most `0.05`; whitened covariance RMS error at most `0.05` at every checked observation | For reference covariance `P=LL^T`, use `norm_2(L^-1(m_hat-m))/sqrt(d)` and `norm_F(L^-1 P_hat L^-T-I)/sqrt(d)`, computed by solves. Compare at identical parameters and initialization; report worst time and conditional behavior. |
| LGSSM likelihood approximation | Centered log-likelihood-ratio discrepancy at most `0.02` on the frozen parameter design | Subtract the value at a fixed central parameter from each method before comparison. Constant offsets do not change a posterior; parameter-dependent discrepancies do. This is a screen, not a posterior proof. |

The proposed accuracy budgets express the intended resolution of these small validation problems. They are convenience hypotheses until calibration establishes feasibility and sensitivity. Claude may revise them during calibration with a written scientific rationale and an updated specification, within the authorized scope. After freezing, a failed check is a failure or a repair trigger; raising a tolerance to pass a held-out result is forbidden. Preserve both a proposed and a realized threshold record. Do not compare the LEDH score with the Kalman score as if equality were required for a finite-particle program: derivative correctness and approximation to the exact model are different checks.

### 8.3 Data, randomness, and tuning partitions

Generate four calibration datasets, two validation datasets, and three untouched claim datasets for each primary model scope. Use the exact declared model, including its initial law. Record every generated dataset, even a difficult or unsuccessful one. Data generation is batch-native TensorFlow/TFP, normally distributed across bounded CPU workers, with `CUDA_VISIBLE_DEVICES=-1` set before imports. This exception is data generation, not the production filter or a GPU performance comparison.

Use a recorded two-integer master seed derived from `20260917` and deterministic fold-ins for model, horizon, partition, attempt, replicate, and purpose. Store the actual resulting seed pairs; do not use Python's process-randomized `hash()`. Keep data-generation, initial-cloud, transport, HMC momentum, and diagnostic-reference streams separate. A retry that needs fresh claim data receives a new recorded attempt partition; a failed claim dataset never becomes tuning data.

Freeze base particles and all finite-program randomness for each target. Repeated evaluation at the same parameter must be deterministic under its declared backend. Extra random filter replicates may measure approximation variability, but they must not be refreshed inside ordinary HMC. Averaging several likelihood estimates defines another finite target and must be specified before evaluation; it is not an invisible variance-reduction patch.

Use all three claim datasets for the planned filtering checks. For each parameter-inference scope, attempt posterior validation on them in recorded seed order, subject to the feasibility budget. Report completed and pending dataset-level results separately. One completed dataset can support its own scoped computational check, not completion of the three-dataset case. Three datasets do not establish frequentist coverage or a population-level performance ranking.

### 8.4 LEDH tuning is separate from HMC tuning

Before a claim run, issue a repository-owned LEDH tuning artifact for that exact scope: model/parameter/prior adapter, horizon/data regime, particle count, state and parameter dimensions, reset/route identity, backend/dtype/TF32/XLA mode, chunk policy, derivative composition, and all applicable numerical controls. Bind the calibrated input identities and the untouched evaluation regime according to the existing tuning interface. A hand-written JSON file carrying a plausible scope name is not tuning authority.

Inventory the actual route's controls before searching them: flow integration schedule, transport regularization and iteration/balance schedule, reset residual/ridge policy, and GenUT correction/trust controls where present. Read their implementation, units, mathematical role, and allowed ranges. Reuse a current, post-invalidation setting only as an explicitly recorded first candidate. Build bounded candidate ranges from convergence, residual, and trust/calibration curves on the calibration data. Freeze the candidate list and selection rule before validation; validate once, then freeze the selected controls before claims. If no valid authority or principled control range exists, repair that prerequisite locally or mark the case unready; do not invent an artifact or silently inherit settings.

Canonical features and reset semantics remain fixed. Turning off required correction, replacing the UKF lifecycle, or changing reset families is not ordinary tuning. A deliberately cheaper force requires its own honestly labeled optional proposal-field contract in Phase 5; it cannot become a second claim-bearing canonical score.

The proposed particle ladder is `N=256`, then `1024`, then `4096` only when a preceding rung fails its approximation screen and the remaining budget permits fresh tuning. Each rung is a new scope. Apply `dpf_transport_exact_divisor_cap3000_v1`: the corresponding chunks are `256`, `1024`, and `2048`. Validate the policy before tracing. These are validation workloads, not recommendations for a production particle count. `N=4096` additionally exercises a genuine `2×2` block grid; a tiny fixture chunk must not substitute for it.

For a frozen rung, choose controls by the predeclared approximation/validity requirements and safety calibration. Runtime can break a tie only after validity, uncertainty, and the stipulated selection rule permit it. Do not tune on the heuristic sanity checks in Phase 6 or on the untouched posterior-agreement result. A failed candidate rejects that scope's candidate; it does not reject LEDH as a research direction.

**Exit evidence:** calibrated status and error rules; immutable-in-practice saved datasets and seed streams; the selected settings and their provenance; valid matching LEDH tuning artifacts; reference filtering comparisons; a scope ledger that identifies ready, failed, and pending cases. Ordinary versioned files, source provenance, and checksums are sufficient.

## 9. Phase 5 — integrate the real HMC consumer and test rejection mechanics

### 9.1 Deterministic mechanics before stochastic chains

Use the production endpoint and runner wiring. A primitive integrator fixture is useful, but it cannot replace a real LEDH consumer test. Keep the current state valid and force the following encounters with fixed initial momenta and a short trajectory:

1. A finite state reaches a true support boundary. A one-dimensional flat target on `q>0`, with `q=0.1`, momentum `-1`, identity mass, one drift of size `0.2`, is a simple independent mechanics fixture. The endpoint has zero density; the Metropolis step must reject and retain the original state and cached value. Check the actual acceptance decision, not just a returned `-inf`.
2. The real canonical value path fails at a declared injected inner stage. Both fused and while-loop endpoints must return their correct sentinel/status through the dual target and runner. Healthy neighboring chains must remain unaffected. Injection marks this as a mechanics test, not an estimate of failure probability under a scientific posterior.
3. Only the proposal force fails while the endpoint likelihood remains valid. Preserve the endpoint value, use the checked zero extension, and compare forward/reverse trajectories with nonidentity mass as well as identity mass. Check the complete potential-plus-kinetic acceptance ratio independently. A finite returned endpoint may be accepted.
4. A first or middle filter failure is followed by otherwise healthy later observations. Check sticky status at the final target and the sampler's retained state.
5. A bad shape/configuration, invalid initial state, or nonfinite position/momentum takes the specific error/health path in §3. An exception from malformed input is not evidence of numerical containment.

Record exact positions, momenta, step sizes, mass, endpoint energies, status events, log acceptance, and accepted/rejected flags. Test state-cache consistency after rejection and across batched chains. A test must assert that the intended failure stage was actually reached. Preserve small failing inputs when a regression fails. Keep finite-value/force failures visible even if a later Metropolis rejection would otherwise conceal them.

### 9.2 Use the public tuner for the actual mathematical target

Before implementation, reread [the HMC tuning interface](../reference/hmc-tuning-interface.md) and inspect `HMC_TUNING_INTERFACE_CAPABILITIES` in `bayesfilter/inference/tuning_contract.py`. At the reviewed revision:

- `tune_hmc_kernel` with `HMCKernelTuningConfig` is the public artifact-authority tuner for an ordinary exact value/score adapter in its declared unconstrained coordinates.
- A frozen position-only field that is not the exact score requires `bind_neural_force_hmc_tuning_runner` and its repository-issued typed binding, passed to `tune_hmc_kernel`. The binding supplies the true endpoint potential in the same coordinates and consumes the tuner's affine mass transformation. The binding factory itself has no tuning-artifact authority.
- `tune_fixed_transport_hmc_kernel` is for an actual frozen nonlinear transport with a Jacobian-corrected transformed target. A cheaper LEDH force alone does not meet that prerequisite.

Check the live capability record rather than assuming these records remain unchanged. A bare callback, chain helper, direct identity-mass fallback, or historical tuner cannot issue the required artifact. If a touched eligible path still uses runtime NumPy, migrate that execution path within a bounded repair before admitting it; diagnostic NumPy comparisons cannot excuse the runtime violation.

Start with the same fully enabled canonical value and analytical force configuration. Qualify this route before introducing an optional cheaper force. If the force extension is activated at invalid-score points, label the field accordingly and use the typed route; do not call an extended non-gradient field an exact score. Bind target, coordinates, prior/Jacobian, fixed random inputs, model/data identity, force identity, settings, and source dependencies through repository-owned factories.

The optional cheaper-field experiment must state exactly which computation changes, prove the field is deterministic and position-only, and retain the full canonical endpoint value. It receives independent force and HMC tuning scopes, healthy-equivalence/mechanics checks, and a separate result. It cannot satisfy a claim that analytical canonical scores are correct. No position-dependent retuning is allowed during retained sampling.

Tuning HMC on the actual inference dataset during adaptation is legitimate when all warm-up draws are excluded and the final kernel is frozen. It is distinct from retuning the LEDH approximation on untouched claim data, which remains forbidden. Confirm this distinction in the two artifact identities so a driver cannot conflate them.

**Exit evidence:** deterministic real-consumer tests, a public-tuner binding check, matching HMC/LEDH artifacts, a frozen kernel specification, and a small actual-route smoke with recorded target status. A smoke can admit the next validation stage; it cannot establish posterior agreement or convergence.

## 10. Phase 6 — validate the sampled posterior with uncertainty

### 10.1 Construct the reference before interpreting a chain

The core inference fixtures have one or two parameters so that an independent posterior calculation is feasible in principle. Integrate in the bounded physical/model coordinates from Phase 3, including the declared prior. When comparing with HMC, transform its draws back to those coordinates. A separate reference implementation must check the transformation/Jacobian convention.

Compute two references for each LGSSM:

1. The posterior formed from the exact Kalman likelihood and the proper prior, with the specified stationary initialization.
2. The posterior formed from the same frozen canonical LEDH value program that supplies HMC endpoint values.

The first measures approximation to the state-space model; the second tests whether the HMC consumer samples its actual finite target. Agreement with the second cannot erase disagreement with the first. Do not substitute the Kalman posterior for the finite-target sampling check or call a finite-particle likelihood exact marginalization.

For SV and predator–prey, compute the finite-program reference similarly. An independent high-particle bootstrap particle filter with replicated likelihood estimates can nominate a model-posterior reference, but its Monte Carlo uncertainty and parameter-dependent approximation error must be controlled before calling that reference adequate. A log of an unbiased likelihood estimator is not an unbiased log likelihood. If a trustworthy nonlinear model reference is unaffordable, report finite-target sampling validation separately and leave model-posterior agreement pending.

Start bounded one-/two-dimensional quadrature with a coarse physical-coordinate design, then refine deterministically under a predeclared error rule. A suggested ladder is 33, 65, 129, and at most 257 nodes per axis, with local refinement near identified concentration if the reference method supports error control. Forecast the number of expensive value evaluations first. Require reference mean uncertainty below `0.01` posterior SD and relevant quantile uncertainty below `0.025` posterior SD; record normalization and mesh-refinement errors as well. These budgets allocate a small part of the comparison margins below to the reference. Two similar grids alone are not proof that a narrow missed mode is absent: check boundaries, multiple dispersed modes/starts, and posterior concentration between nodes. If the method cannot bound its error sufficiently, mark the reference unresolved instead of declaring agreement.

Independent NumPy/SciPy quadrature or reporting is permissible only in an explicitly diagnostic reference module that is not imported by runtime/tuning paths. Evaluate the claim-bearing LEDH value with the eligible TensorFlow route. Store parameter nodes, log values, normalization calculations, uncertainty estimates, and the exact source/configuration identity so the comparison can be reconstructed.

### 10.2 Diagnose numerical exclusion as a target change

Map support and numerical status separately over the reference parameter design, supplemented by prior draws, concentrated posterior probes, boundaries, and the deliberately difficult regions identified during calibration. The reference must attempt to evaluate a flagged point using an independently checked, mathematically equivalent route or precision when available. Higher precision with a different finite program is informative, but equality of targets must be established before its values are substituted.

Estimate or bound posterior mass excluded solely by evaluator failure using the unmasked reference and its integration uncertainty. Do not estimate it from the fraction of HMC proposals rejected. If the reference is itself evaluated with the same mask, it cannot reveal the missing mass. Report `unknown` when the unmasked finite target is not defined or cannot be evaluated well enough to support the comparison.

The proposed approximate-agreement budget is an upper bound of `1e-4` on excluded posterior mass, including reference uncertainty. Its rationale is to make truncation smaller than the posterior comparison resolution; it does not prove exact invariance to numerical failure. Any nonzero exclusion must remain labeled approximate. Exact equality requires an argument that the relevant mathematical target is not being restricted, not merely zero failures in a finite sample. An unknown or larger mass blocks an unqualified posterior-correctness conclusion while leaving successful engineering containment reportable.

Do not select the bound after seeing exclusions, narrow a prior to hide them, or treat successful rejection of the excluded points as validation. A failure that occurs only in an intentionally invalid mechanics fixture has a different evidentiary role from failure inside the declared scientific target.

### 10.3 Sampling protocol and decisions

Use four independent chains with dispersed valid initial points, derived from recorded prior quantiles or a predeclared dispersed initialization distribution. For multiple parameters, use different coordinate permutations and independent recorded jitter rather than placing all chains on one diagonal. Verify initialization values, force status, and coordinate conversions before tuning. Exclude all adaptation/warm-up draws from posterior estimates, but preserve them and their diagnostics.

Use the public tuner's supported bounded adaptation and trajectory search, with at most four nominated kernel candidates and a proposed total adaptation cap of 4,000 transitions per chain across stages/candidates. Confirm that this cap can accommodate the tuner's mandatory stages before launching; otherwise revise the resource estimate or mark the scope under-budgeted. A trajectory-count search may start with `{4,8,16,32}` only if compatible with the live tuner, with step size and mass selected through that tuner. These are proposed search/budget hypotheses, not inherited scientific defaults. Do not silently truncate a mandatory stage to fit them.

After independent pilot/tuning diagnostics, freeze the retained sample count, initially targeting 4,000 and capped at 10,000 transitions per chain. The cost forecast must include reference evaluation, adaptation, all candidates, leapfrog calls, and endpoint calls. Choose the retained length using pilot autocorrelation/MCSE estimates before examining final posterior agreement. Preserve a failed final result. Any subsequent repair/retry is a new attempt under the rules in §12, rather than repeated checking until a desired answer appears.

For every retained chunk, check finite states, finite accepted-state target values, status, movement in each chain, and the kernel's declared energy/divergence diagnostics. Freeze the live tuner's divergence definition in the specification. If it supplies none, use the proposed diagnostic convention `abs(delta_H)>1000` on trajectories with mathematically valid finite endpoints, together with any nonfinite state/momentum/energy, as a hard numerical-admission veto; calibrate this convention before claims and report the full energy-error distribution. It is a catastrophic-error screen, not proof of accurate integration. True support proposals may reject normally; their infinite endpoint potential is not automatically an integrator divergence. Numerical-value exclusions, force-extension events, and nonfinite trajectories must remain separate counters. A finite accepted state whose cached value is invalid is an implementation failure. A rare force-extension event is not automatically a biased target, but it requires the qualified proposal mechanics and adequate mixing evidence.

Use the maximum of rank-normalized split and folded rank-normalized split R-hat, with threshold `<=1.01`. Require bulk ESS at least 1,600 and tail ESS at least 400 for each reported parameter, plus directly estimated MCSE no larger than `0.025` posterior SD for means and `0.05` posterior SD for the stated tail quantiles. The ESS floors nominate adequate precision; the actual MCSE and autocorrelation-aware uncertainty are decisive. These criteria concern the completed case, not a claim that four short chains can rule out every missed mode.

Compare means and 5%, 50%, and 95% quantiles against the finite-program reference. Proposed equivalence margins are `0.10` reference posterior SD for means and `0.20` for those quantiles. Construct simultaneous 95% uncertainty intervals for the discrepancies across the declared parameters and summaries, accounting for chain autocorrelation and reference error. State the method in advance; conservative Bonferroni-adjusted, batch-means/quantile-MCSE intervals are an acceptable initial diagnostic method when their assumptions are checked. An interval must lie wholly inside its equivalence margin to pass. An interval overlapping a margin is inconclusive; absence of a significant difference is not equivalence.

For LGSSMs, repeat the comparison against the Kalman reference as a separate approximation decision, with its own uncertainty accounting. For nonlinear models, do so only when an adequate model reference exists. Report posterior SDs and scale definitions explicitly. Severe multimodality, unstable MCSE, inadequate reference resolution, or failed R-hat cannot be cured by reporting a favorable mean. Truth inside two posterior SDs is a descriptive event, not a correctness or coverage criterion.

For the seven proposed inference scopes, report each dataset-level decision and whether all three planned datasets finished. Fixed SIR has no inference decision. Do not use a “seven out of eight” pass rule. A case that is infeasible, unsupported, or under-budgeted remains pending or failed for its stated reason; it does not become a pass by aggregation.

### 10.4 Construct the heuristic comparisons and evaluate difficult situations

Before interpreting the complex method, implement the applicable cheap alternatives on the same observations and declared model. This table is the starting adversary set, with eligibility tied to the actual model:

| Alternative | Construction and purpose |
|---|---|
| Unconditional/initial estimate | Use the stationary unconditional mean/covariance where available; otherwise propagate the declared initial mean using the deterministic model. This tests whether observations and filtering add useful information. |
| Prediction-only filter | Propagate the declared process without updating on observations. Use exact moment propagation for LGSSMs and a clearly labeled deterministic or ensemble approximation otherwise. This exposes updates that make a state estimate worse. |
| Direct observation estimate | On directly observed coordinates, invert the actual linear observation map and use a prediction-only estimate for the others. Record the resulting noise. This applies to LGSSM, predator–prey, and infectious SIR coordinates; it does not pretend that a zero-mean SV observation is invertible. |
| Plain UKF | Use the same model and initialization with the ordinary UKF when its observation interface represents the true density. For basic SV, a constant observation mean gives no ordinary state update; report that degeneracy rather than replacing the density. |
| Bootstrap particle filter | Implement the prior-transition proposal with actual observation weights and resampling, using independent recorded random streams. Compare both a declared practical particle budget and, separately, a larger independent reference budget when affordable. |

Use at least three well-defined alternatives per applicable case; the table supplies three even when direct inversion/UKF is inapplicable. Keep the exact Kalman filter as an additional oracle for LGSSMs. Report state RMSE and uncertainty calibration on synthetic truth, filtering mean/covariance error to an adequate oracle, and likelihood/posterior discrepancies where references permit. No heuristic is automatically an exact reference.

Predeclare conditional situations from the model: weak/strong observation information and observed/unobserved coordinates; low/high persistence at fixed prior quantiles; small/large process scale; early/late observation times; benign/ill-conditioned covariance fixtures; low/high SV volatility; and low-population/rapid-growth epidemic or predator–prey states. Freeze the bin definitions from calibration data or model quantities. Record counts and Monte Carlo uncertainty; an empty or underpopulated bin supplies no reassuring evidence. Outcome-based diagnostics such as weight collapse may explain a result but must not be selected post hoc as the only conditions reported.

Distinguish conditions visited within the base datasets from changes to the generating parameters. Evaluating a different parameter on a fixed dataset is a likelihood/score stress check, not a filtering comparison generated in that regime. If making claims across parameter regimes, add an explicitly budgeted stress design with generating parameters at the 10th and 90th prior quantiles, changing one parameter at a time, with three independent datasets per setting and matching scope-specific tuning. Keep other parameters at their declared truths. This extension has to fit the remaining budget or stay pending; the three base claim datasets do not establish it. Base-scope conclusions can cover their predeclared within-dataset conditions without pretending to cover unvisited regimes.

Heuristic checks are falsification instruments, not tuning objectives. Any observed loss to an applicable heuristic in a salient situation must be a headline and leaves comparative/default promotion uncleared. Report uncertainty: an imprecise observed loss does not establish statistical inferiority, but it cannot be presented as a cleared dominance check. Passing the heuristic set establishes neither superiority nor posterior correctness. It also does not prevent adopting a separately verified harmless containment repair.

**Exit evidence:** separate finite-target and model-posterior decisions, retained samples and uncertainty analysis, numerical-exclusion assessment, conditional heuristic tables and machine-readable verdicts, and honest pending cases. No ranking of viable stochastic methods is justified by descriptive means, tails, or runtimes alone.

## 11. Phase 7 — measure capacity and cost after correctness

Measure the default GPU, float32, TF32-enabled, XLA-compiled route with an explicit stable input signature. Configure and verify memory growth before logical GPU initialization; record physical GPU UUID and actual TensorFlow placement. CUDA ordinal zero alone is insufficient provenance on this multi-GPU machine. All GPU probes and runs need trusted/escalated execution under the local policy. Do not launch a large job on an occupied device without the task's resource-sharing conditions being satisfied.

Use float64 and float32-without-TF32 only as labeled reference/comparison scopes, with their own tuning when they support claim-bearing comparisons. CPU or non-XLA runs are explicit debugging/reference exceptions. A different execution mode can localize a defect; it does not qualify the GPU/XLA default. The review's empty CPU graph result and GPU ordinary-graph timeout remain unresolved diagnostics until investigated or clearly excluded by the supported API; do not relabel them as passes.

For each ready scope, separate tracing/compilation, first execution, steady-state value/score cost, reference cost, tuning, and retained sampling. Start with ten paired steady-state repetitions on identical healthy inputs and synchronize by materializing the actual returned result. Compare guarded and same-mathematics healthy baselines only after healthy equivalence. Report paired differences and uncertainty, not an unsupported claim that overhead is below a universal percentage. A necessary non-harmful guard is not rejected because it misses an invented 5% threshold.

Exercise at least one ready `N=4096,K=2048` scope for exact multi-block transport if affordable. Record allocator current/peak bytes, peak host memory, graph/compilation behavior, and each failure. TensorFlow reservation or a device memory limit is not live tensor memory. A capacity failure is a resource finding, not posterior evidence or permission to shrink chunk sizes outside policy. Stop before an estimated job exceeds available memory or its allocated budget.

Performance conclusions are limited to the measured device and exact tuned scope. No default algorithm/settings change follows from this phase. Optional cheaper-field efficiency comparisons require the same endpoint target, valid chains, comparable uncertainty, total tuning costs, and a predeclared statistical comparison; raw ESS/second from one short run is descriptive.

**Exit evidence:** synchronized cost and capacity records, hardware/memory-policy provenance, paired timing uncertainty, and a feasible cost envelope for only the validated scopes.

## 12. Execution order, assumption audit, and bounded resources

### 12.1 Follow dependencies without waiting for every model

First complete Phase 0 and the shared Phase 1 repair. Then take **LG1 through the full sequence**: parameter/initialization derivatives, model specification, LEDH calibration/tuning, actual HMC binding, and one controlled posterior comparison. This is the earliest complete test of whether the program's consumer and reference design work. A mechanics smoke may precede expensive tuning, but it retains its non-claim-bearing status.

Expand next to LG2 and LG4, then the eligible nonlinear adapters, with LG20 posterior validation last because of its cost. Run SIR filtering after its adapter qualifies. Work on independent cases may continue when another case has a documented capability or reference blocker. Shared sentinel/target corruption blocks all dependent inference. Do not wait until eight expensive runs finish to discover that the scalar target or the reporting contract is wrong.

Phases 2–6 repeat per scope where their assumptions change. The final driver must express prerequisites explicitly; a `run-all` convenience option must not bypass them. Within an authorized campaign, use ordinary local repairs and scoped regression checks; no fresh reviewer approval is required for each step. A material scientific or direction change still requires resolution before the dependent action.

### 12.2 Defaults and assumptions to record before execution

| Choice | Provenance and justification | Failure mode / earliest check | Promotion status |
|---|---|---|---|
| Canonical reset, UKF lifecycle, correction, analytical score, chunk policy | Current owner directives and their canonical specification | A reduced consumer silently bypasses a capability; executable route/identity checks in Phases 0–2 | Binding target requirements; not proof of implementation correctness |
| Failure sentinel and finite zero force extension | Proposed repair plus the deterministic reversible-proposal argument in §3 | Primal incorrectly discarded, asymmetric abort, contaminated derivative, wrong mass coordinates; full-consumer mechanics and reversal tests | Candidate safety semantics until tests pass |
| Synthetic truths, horizons, bounded priors, partial-inference choices | Explicit fixtures in Phase 3 chosen for tractable independent references | Weak identification, concentrated/multimodal posterior, unintended support restriction; structural/prior/reference checks | Fixture hypotheses; no universal priors or real-data claim |
| Stationary initialization and fixed particle randomness | Mathematical target definition | Missing `dP0`/cloud derivative or refreshed noise; analytic/finite-difference tests and repeated values | Required for the declared fixtures |
| Predator–prey/SIR integration and domain policy | Existing local model settings, preserved to test that model | Discretization or negative-state behavior dominates; equation, timing, domain, and step-refinement diagnostics | Local baseline; no continuous-time/source-faithfulness claim without its own evidence |
| `N=256,1024,4096` | Bounded validation ladder; last rung exercises multi-block transport | Approximation inadequate or cost prohibitive; Kalman/reference screen and early timing | Convenience candidates, each freshly tuned |
| Flow/transport/correction/ridge controls, including off/zero choices | Actual current route plus principled calibration/derivation | Silent scientific-object change, collapse, biased resets; residual/trust/non-harm curves | Hypotheses until scope-specific tuning succeeds |
| GPU/TF32/XLA and stable signatures | Repository execution direction | CPU evidence substituted for GPU, retracing or unsupported kernels; explicit default-route smoke | Execution requirement, not a scientific performance claim |
| Numerical and posterior margins | Proposed resolutions in Phases 4/6 | Loose tolerance conceals error or strict tolerance makes work infeasible; calibration and sensitivity analysis before claims | Provisional until frozen; never adjusted to pass holdout |
| Seed/partition counts and four chains | Reproducible bounded comparison | Small-sample rankings, missed modes, contaminated holdouts; uncertainty and partition audit | Minimum design, not universal adequacy |
| HMC candidates, transition caps, retained counts | Feasibility hypotheses subordinate to the public tuner's stages | Stage truncation or inadequate MCSE; live capability check and pilot forecast | Budget choices, never convergence evidence |
| Reference quadrature/particle budget | Low parameter dimension and independent error control | Missed modes, shared mask, insufficient precision; refinement, dispersed checks, uncertainty accounting | Reference candidate until validated |
| Excluded-mass budget | Proposed truncation allowance below comparison resolution | Proposal failure rate mistaken for posterior mass; independent unmasked integration | Approximate-agreement allowance only |

An inherited setting may be a first candidate only if current policy allows it; invalidated historical LEDH results are never such candidates. Resolve undocumented assumptions in this table before their first consequential run, rather than attaching a disclaimer after execution.

### 12.3 Proposed campaign budget

This is a **proposed ceiling**, not authorization to launch or a forecast that every case will finish. The current task spends none of it. A later execution request can adopt these limits as one campaign. Use one GPU at a time; count actual device-hours across all compilation, probes, failures, tuning, reference evaluation, and sampling. Count CPU worker-hours in aggregate, not just elapsed wall time. Do not run independent GPU campaigns in parallel under this allowance.

| Stage | GPU device-hours | CPU worker-hours | Scope of allocation |
|---|---:|---:|---|
| Phase 0 | 0 | 1 | Provenance, route/status inventory, fixture preservation |
| Phase 1 | 2 | 3 | Containment and consumer regression checks |
| Phases 2–3 | 4 | 4 | Derivative/model checks and bounded data preparation |
| Phase 4 | 14 | 3 | Numerical calibration, scope tuning, filtering references |
| Phase 5 | 2 | 1 | Real-runner mechanics and tuner binding |
| Phase 6 | 30 | 3 | Reference posteriors, HMC adaptation and retained checks |
| Phase 7 | 4 | 0.5 | Capacity and repeated timing |
| Phase 8 / closeout | 0 | 0.5 | Artifact validation, interpretation, status updates |
| Local repair reserve | 8 | 0 | Reproductions/retries within the same contract |
| **Total ceiling** | **64** | **16** | No paid or expanded compute implied |

The planning times are allocation limits, not measured throughput. Redistribute unused allocations within the total and unchanged contract with a recorded rationale, preserving a repair reserve until the major prerequisites pass. Cap a single launched job at four GPU-hours initially; split longer planned work into restartable, documented chunks without altering a frozen target. Allow at most three substantive attempts per case/contract (initial attempt plus two repairs); ordinary tiny unit-test iterations do not consume a campaign attempt but do consume time. A new scientific candidate or a fresh held-out attempt must be counted and identified honestly.

Before each expensive stage, measure a small representative compiled call and forecast:

```text
total cost = compile + all tuning candidates + reference nodes
             + chains * retained transitions * actual per-transition cost
             + diagnostics/artifact work + repair allowance
```

The per-transition cost includes leapfrog counts, force/value calls, batching, endpoint correction, and synchronization. Apply a planning multiplier of `1.5` to the measured variable cost, and compare with the remaining allocation and device availability. Record the measured quantities, not just the resulting estimate. A reference grid with 257 nodes per axis can require 66,049 expensive values; low parameter dimension alone does not make that affordable.

If the estimate does not fit, classify the intended claim as **under-budgeted**. Continue cheaper independent repair or validation work that still answers its own question; do not reduce convergence requirements, transfer tuning, remove reference checks, or call a smoke a completed posterior study. Full nonlinear/model-reference validation may need a later explicitly enlarged program. Optional six-parameter predator–prey and parameterized SIR are outside the core allocations unless spare budget and the explicit extension contract cover them; they cannot silently displace mandatory evidence.

### 12.4 Stop, repair, or continue

| Event | Required decision |
|---|---|
| Shared wrong value/score, sentinel corruption, bad shape semantics, or invalid cache | Stop dependent inference; preserve the smallest reproduction; repair and rerun the focused check. Continue unrelated static work. |
| A calibrated LEDH candidate fails its approximation/non-harm criterion | Reject that candidate and try the planned repair/rung within budget and fresh tuning. Do not reject the research direction. |
| Held-out failure | Preserve it as final evidence for that attempt. Repair using permitted calibration information and, if justified, a fresh versioned tuning/claim partition. Do not tune on the failed holdout. |
| Reference or target definition is invalid/unavailable | Block that comparison; repair it or report the scientific claim unresolved. A finite chain does not repair the reference. |
| HMC has a supported true-support rejection | Continue with normal accounting if accepted states and kernel mechanics remain valid. |
| Numerical exclusion, force extension, divergence, or nonfinite proposal occurs | Apply its distinct Phase 6 rule; do not merge these into a harmless rejection total. Localize and repair where the applicable veto fires. |
| Infrastructure/resource error within unchanged contract | Record classification, spent time, remaining budget, and focused repair; retry in a fresh directory within the attempt cap. |
| Compute/attempt ceiling, new paid resources, material target/privacy/direction change | Stop that action and present the completed evidence plus the concrete new decision needed. No automatic scope expansion. |

Before ending the investigation, state whether the evidence invalidated the implementation, harness, reference, target definition, current candidate, or scientific idea. A failure of the current candidate alone is not evidence against every later planned repair.

## 13. Artifacts, executable deliverables, and closeout

### 13.1 Keep the implementation small and the evidence reconstructable

Prefer extending the existing general modules and focused tests listed in Phases 1–2. Do not create eight algorithm implementations. Model adapters may provide model mathematics and tangents; filtering/reset/transport/score recursion stays in the shared canonical implementation.

Reuse the existing tests in `tests/highdim`, `tests/inference`, and `tests/integration`. In particular, repair `tests/integration/test_ledh_graceful_failure_integration.py` and `tests/integration/test_ledh_hmc_graceful_failure.py`, and expand actual-consumer coverage in `tests/inference/test_ledh_dual_parameter_target.py` and `tests/highdim/test_ledh_canonical_batch_fused.py`. Use `tests/test_neural_force_hmc.py` and `tests/test_hmc_kernel_tuning_public_api.py` for the typed binding. New tests should prove a missing invariant or real consumer behavior, not merely repeat implementation expressions.

If no suitable existing driver covers this program, the proposed additions are:

- `scripts/run_ledh_graceful_failure_program.py`: a thin host-side driver selecting eligible shared implementations, consuming repository-issued artifacts, enforcing prerequisites/budgets, and writing results.
- `docs/benchmarks/ledh_graceful_failure_reference_diagnostics.py`: independent quadrature, finite-difference, and reporting/reference code, with its diagnostic status explicit and no import from runtime/tuning paths.
- Versioned machine-readable model/run specifications under the campaign artifact root. Keep source definitions and calibrated values separate enough to distinguish defaults from realized settings.

These paths are proposed deliverables, not existing commands at the time this program is written. The driver should expose phase/case/spec/output/attempt/seed-manifest/time-budget controls; default to GPU and JIT on; validate chunk/route/tuning identity before tracing; and make any CPU or `--no-jit-compile` exception explicit in its output. It must obtain canonical identities from the repository factory, never from a `--canonical` self-attestation flag. Reuse existing configuration and artifact APIs rather than adding a competing authority system.

### 13.2 Required contents of each versioned attempt

```text
<campaign>/<case>/<scope>/<attempt>/
  spec.json                 # actual model, priors, modes, controls, thresholds
  seeds.json                # exact streams and partitions
  manifest.json             # command, environment, source and hardware identity
  inputs/                   # data, fixed random inputs, hashes, failure fixtures
  tuning/                   # LEDH and HMC artifacts from their actual authorities
  samples/                  # warm-up and retained draws, distinctly labeled
  diagnostics/              # statuses, reference nodes/results, uncertainty
  logs/                     # complete stdout/stderr and exit/timeout information
  decision.json             # separate engineering, numerical, statistical decisions
  result.md                 # interpretation, limitations, next justified action
```

Not every phase needs every directory; absence must agree with its phase/status. Record `N/A` only when a field genuinely does not apply. In the manifest include commit, relevant dirty-source hashes, exact command and working directory, environment/versions, CPU/GPU choice, GPU UUID, verified memory-growth policy, dtype/TF32/XLA/input signatures, model/data version, seeds, scope/tuning identities, elapsed time/resource charges, this plan, and all output paths. Preserve full logs on disk; return only key status and paths to the conversation.

The decision schema must distinguish at least:

```text
engineering_containment: passed | failed | not_checked
healthy_equivalence: passed | failed | not_checked
analytical_score: passed | failed | not_checked
filtering_reference: passed | failed | inconclusive | not_checked
finite_target_posterior: passed | failed | inconclusive | not_checked
model_posterior: passed | failed | inconclusive | not_checked
numerical_exclusion: bounded | exceeds_budget | unknown | not_applicable
heuristic_dominance: cleared | uncleared | not_checked
execution_status: completed | failed | blocked | under_budgeted
```

Keep reason codes, scope identities, per-dataset statuses, reference uncertainty, conditional sample counts, and pending requirements beside these fields. `passed` always refers to the declared scoped criterion, not global correctness. Encode nonfinite numerical quantities in JSON as null plus an explicit status, or another documented standards-compliant representation; do not emit nonstandard `NaN`/`Infinity` tokens. These result fields report evidence; they cannot issue canonical/tuning identity.

### 13.3 Suggested initial checks and command recording

After implementation begins, the existing focused CPU reference tests can start with the following command, subject to verifying their current paths and dependencies:

```bash
CUDA_VISIBLE_DEVICES=-1 /home/chakwong/anaconda3/envs/tftwogpu/bin/python -m pytest -q \
  tests/highdim/test_ledh_numerical_safety_tf.py \
  tests/highdim/test_ledh_canonical_batch_fused.py \
  tests/inference/test_ledh_dual_parameter_target.py \
  tests/integration/test_ledh_graceful_failure_integration.py \
  tests/integration/test_ledh_hmc_graceful_failure.py
```

This is explicitly a CPU reference/debug check with GPU devices hidden. It does not replace the same critical actual-route checks on trusted GPU/XLA. For GPU checks, select the recorded device, set `TF_FORCE_GPU_ALLOW_GROWTH=true` before import, call and verify the repository memory-policy helper before initialization, and save the exact resolved command. Run those commands with escalated/trusted GPU permission. Do not embed invented driver options or assume test success from this template.

Before every serious command, materialize the full command and specification in the attempt manifest, including the bounded phase/case, timeout, actual driver interface, seed artifact, and output directory. A launch must consume those same settings. Do not rely on a human remembering settings that appear only in this prose. Tests need no exhaustive extra campaign once the appropriate checks pass; broaden only for a new change, failure, or unresolved risk.

### 13.4 Phase 8 — terminal review and truthful status

Reconcile every case/attempt against the budget, files, and criteria. The result note must contain a decision table with columns **decision, primary criterion status, veto status, main uncertainty, next justified action, and what is not concluded**. For stochastic results, add this inference-status table:

| Required row | Content |
|---|---|
| Hard veto screen | Exact implementation/numerical/reference failures and whether each was resolved |
| Statistically supported ranking | The supported comparison with its uncertainty method, or explicitly “none” |
| Descriptive-only differences | Means, tails, ESS, acceptance, and runtime differences lacking ranking evidence |
| Default-readiness | Scoped admission/pending status; no blanket promotion from these synthetic fixtures |
| Next evidence needed | The smallest justified repair or additional reference/replication |

Add a short red-team paragraph: strongest alternative explanation, weakest evidence, and what would overturn the conclusion. Preserve failed candidates and unsupported scopes. Check that every successful claim is linked to an actual consumer, matching tuning artifact, eligible execution mode, and adequate reference.

Update active status documents to the actual outcome. A valid completion can say “containment fixed; analytical score qualified for these scopes; nonlinear model-posterior agreement pending.” It cannot say “production ready” merely because all scripts returned zero. One terminal substantive review is sufficient when useful; do not create approval paperwork at every phase or let a missing advisory reviewer erase executable evidence. This program itself does not instruct anyone to launch another agent.

## 14. Review-finding closure and Claude's first actions

| Finding | Required work | Evidence that closes the finding |
|---|---|---|
| R1: sentinel lost by real consumer | Phases 1 and 5 | Actual canonical → both batch wrappers → dual target → runner tests preserve status, `-inf`, zero force, and healthy rows. |
| R2: invalid derivative arithmetic and incomplete failure coverage | Phases 1, 2, and 4 | Every factor/reset/tangent stage has safe inactive arithmetic, independent primal/force status, and sticky first/middle/last failure tests. |
| R3: unstable/unobservable Case 3 | Phase 3 | Replaced, explicitly stable LG4; checked observability conditioning and valid stationary covariance, with the old case retained only as a negative fixture. |
| R4: wrong nonlinear equations | Phases 2–3 | Exact local equations, noise/domain/integration conventions, generated data, and analytical adapters match; source-faithfulness claimed only with required anchors. |
| R5: identification and missing parameter derivatives | Phases 2–3 and 6 | Explicit identifiable-or-limited parameter hypotheses, proper priors, total initialization/noise derivatives, endpoint wiring, and actual posterior reference; no state-observability shortcut. |
| R6: incomplete finite and shape checks | Phases 1 and 4 | Nonfinite/upper-triangle/diagonal tests and batch shapes pass, while healthy scale tests do not falsely reject. |
| R7: unsupported HMC/posterior conclusions | Phases 5–6 and 8 | Checked proposal mechanics, independent finite/model references, uncertainty-aware equivalence, and an explicit numerical-exclusion assessment. |
| R8: inadequate integration tests | Phases 1 and 5 | Fixed shapes and deliberately reached real-engine failures, with true runner rejection/cache checks; mocks/toys remain primitive evidence only. |
| R9: unexecutable/underspecified campaign | Phases 0, 3–8 | Frozen specifications, scoped LEDH/HMC authority, recorded assumptions, bounded resource/attempt ledger, conditional baselines, and reproducible decisions. |

Claude's first substantive action after authorization is to verify the reviewed source state and reproduce **R1/R2/R6 on the smallest actual consumer**, then fix the shared containment path and its regression tests. The next action is the LG1 end-to-end validation described in §12.1. Broad nonlinear HMC and performance campaigns come only after those prerequisites.

The program is complete as an investigation when every required scope has an evidence-linked decision, all attempted work is accounted for, and pending work has a concrete reason and next action. The validation objective succeeds only for the scopes that actually pass their stated engineering, mathematical, and statistical criteria. Keep those two meanings of completion distinct.
