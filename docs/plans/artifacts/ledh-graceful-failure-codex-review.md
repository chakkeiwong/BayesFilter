# LEDH graceful-failure plan: technical review

**Verdict: Option C — major revisions required.**

Reviewed on 2026-09-17 against branch `surrogate-hmc`, commit `8a5c23ab1172884ffca63cac390623bf7735afe7`, and the working-tree sources recorded with the diagnostic artifacts. The inputs are the [review request](ledh-graceful-failure-codex-review-memo.md) and [comprehensive testing plan](../ledh-graceful-failure-comprehensive-testing-plan.md).

The proposed validation is necessary, but the plan cannot establish its stated conclusions as written. More immediately, the current implementation does **not** propagate the promised `(-inf, zero score)` result through the real HMC-facing consumer. This is a reproduced implementation defect, not merely missing evidence from larger models. The four-state LGSSM is also unstable and unobservable in one direction, two nonlinear model descriptions disagree with the repository, and the posterior success criteria would accept misleading results.

The “production ready” statements in the [status document](../ledh-graceful-failure-STATUS.md) and [Phase 4–5 completion report](../ledh-graceful-failure-phase45-completion.md) are unsupported. They should be replaced with the narrower status: **partial numerical-failure handling implemented; consumer propagation is broken; validation and repair are required**. This finding concerns graceful-failure readiness. It does not change the owner's GPU/XLA LEDH production direction or reject the LEDH research program.

This review changes no implementation, model, numerical default, or original plan. It includes bounded diagnostic scripts and evidence, and specifies the repairs needed before the proposed research campaign.

## 1. Findings in priority order

| ID | Priority | Finding | Evidence and required disposition |
|---|---|---|---|
| R1 | P1 | The actual batch wrapper destroys the failure sentinel before the dual target can consume it. | An injected `-inf` raises an equality assertion in ordinary execution and becomes `NaN` under XLA. The unpatched GPU/XLA dual target returns `NaN` value and gradient on an invalid-covariance fixture. Repair the entire consumer call chain. |
| R2 | P1 | A zero Cholesky factor is immediately used in derivative solves; validity flags do not prevent invalid arithmetic. | Failed sigma-point factors produce NaN tangents. Reset derivatives and other factorizations have analogous unguarded operations. Make inactive computations safe and preserve failure across the full trajectory. |
| R3 | P1 | LGSSM Case 3 cannot have the stationary covariance claimed by the plan. | Its eigenvalues are `1.2, 0.8, 0.8, 0.8`; its observability rank is 3. A successful Lyapunov linear solve returns an indefinite matrix. Replace the specification. |
| R4 | P1 | Predator–prey and SIR equations describe different models from the local implementations. | Direct equation-to-code comparison below. Correct the equations and generate data from the precisely declared model. |
| R5 | P1 | State observability is being used as a certificate of parameter identifiability; stationary-initialization derivatives are missing from the proposed consumer wiring. | Similarity transformations give a counterexample to the first claim. The fused consumer does not expose initial-cloud/covariance tangents or parameter-dependent covariance tangents. Declare the inferred parameters and verify their complete score path. |
| R6 | P1 | NaN-only Cholesky validation is not a finite-input or finite-factor guarantee. | Some infinite and upper-triangle-NaN inputs are reported valid; singleton batch dimensions are erased. Repair the validity and shape contract. |
| R7 | P1 | Crash avoidance and the proposed posterior screens do not establish posterior correctness. | Numerical rejection can truncate the implemented target; truth-within-two-posterior-SDs, ESS above 100, and a 7/8 aggregate pass rule do not resolve that problem. Use a stated reference target and uncertainty-aware criteria. |
| R8 | P2 | Existing integration evidence does not test the claimed route adequately. | The focused suite has 27 passes, 2 failures, and 1 skip. The two real-model tests fail on input shapes; other tests mock out the problematic wrapper or exercise a toy target. |
| R9 | P2 | The campaign lacks the settings, admission path, budgets, and stop conditions required to execute and interpret it. | Specify model/parameter/prior contracts, canonical route controls, scope-specific tuning, HMC authority, execution modes, and total compute/attempt limits. |

P1 denotes a defect that blocks the proposed correctness or readiness conclusion. P2 denotes a material execution or evidence gap. These are review priorities, not estimates of failure frequency in an actual posterior run.

## 2. What was checked

The review inspected the memo and plan, the numerical-safety helper, UKF tangent stages, reset implementation, canonical score loop, both batch wrappers, the dual target, relevant tests, local model equations, and the [HMC tuning interface](../../reference/hmc-tuning-interface.md) with its capability registry. Mathematical claims were checked by derivation and small TensorFlow diagnostics.

All executable evidence is in [ledh-graceful-failure-review-20260917-01](ledh-graceful-failure-review-20260917-01/). The [checkpoint](ledh-graceful-failure-review-20260917-01/checkpoint.md) records the pre-run skeptical audit and diagnostic evidence contract. The campaign itself was not run.

| Check | Result | Evidence |
|---|---|---|
| Six focused existing test files, deliberate CPU reference mode | **27 passed, 2 failed, 1 skipped**; pytest exit 1 | [Full log](ledh-graceful-failure-review-20260917-01/focused-pytest.log), [command/environment manifest](ledh-graceful-failure-review-20260917-01/focused-pytest-manifest.json) |
| Deterministic matrix, factor, tangent, wrapper, and actual-engine probes | Completed; several contracts disproved as detailed below | [CPU results](ledh-graceful-failure-review-20260917-01/probe-cpu.json), [script](ledh-graceful-failure-review-20260917-01/review_probe.py) |
| Initial GPU probe | Factor, tangent, and injected-wrapper checks completed; actual ordinary-graph check exceeded the 240-second bound | [Partial results](ledh-graceful-failure-review-20260917-01/probe-gpu.json), [timeout record](ledh-graceful-failure-review-20260917-01/probe-gpu-invocation.json) |
| Fresh-process unpatched dual target, CPU XLA | Healthy value/gradient finite; invalid covariance returns NaNs | [CPU results](ledh-graceful-failure-review-20260917-01/isolated-cpu.json) |
| Fresh-process unpatched dual target, RTX 5080 GPU XLA | Same failure; healthy control agrees with CPU to the reported precision | [GPU results](ledh-graceful-failure-review-20260917-01/isolated-gpu.json), [script](ledh-graceful-failure-review-20260917-01/isolated_target_probe.py) |

The environment was `/home/chakwong/anaconda3/envs/tftwogpu/bin/python`, TensorFlow `2.20.0-dev0+selfbuilt`, and TensorFlow Probability `0.25.0`. CPU runs set `CUDA_VISIBLE_DEVICES=-1` before import. GPU runs used escalated access and verified memory growth before device initialization. The initial GPU script's CUDA ordinal 0 resolved to an RTX 4080 SUPER; the final isolated script selected the RTX 5080 by UUID. Device names and placement are recorded rather than inferred from ordinal labels.

Aggregate framework-test/probe process time was **326.38 seconds**, including the 240-second timeout, within a predeclared 900-second diagnostic bound. These durations are resource accounting, not performance comparisons. Every invocation has a saved command and log. The main probe records source hashes; the [review manifest](ledh-graceful-failure-review-20260917-01/review-manifest.json) extends that provenance to the other inspected sources and artifacts.

The actual-consumer fixture used `N=4`, state dimension 1, one observation, fixed clouds/noises, Contract E, and enabled higher-moment/pairwise/coordinate-cap controls. It is an untuned mechanics fixture, not claim-bearing tuning or posterior evidence. Its derivative is the consumer's analytical custom score; `GradientTape` only reads that registered custom gradient. The complete-consumer checks used float64. The primitive checks included float32 and float64. **No full float32/TF32 production-route qualification was performed.**

Two limitations are retained rather than hidden: ordinary CPU graph execution returned empty outputs even for the healthy unpatched target in a fresh process, and the initial ordinary GPU graph check timed out. Their causes are unresolved. The independent XLA reproductions are sufficient for R1 and do not depend on explaining these additional graph defects.

## 3. Numerical failure does not yet reach the consumer safely

### 3.1 The broken sentinel path

The checked call chain is:

```text
DualParameterLEDHTarget.__call__
  -> canonical_batch_fused_value_score
     -> canonical_value_and_analytical_score
        -> UKF / flow / Contract-E reset and correction
```

In [ledh_canonical_score_tf.py](../../../bayesfilter/highdim/ledh_canonical_score_tf.py), lines 572–582, the per-step numerical/callback flags can select `-inf` and zero. But the consumer does not call this function directly:

1. [ledh_canonical_batch_fused_tf.py](../../../bayesfilter/highdim/ledh_canonical_batch_fused_tf.py), lines 260–266, asserts that directional primal values are near their common value. Comparing `-inf` to `-inf` through a numerical closeness check fails. The deterministic wiring fixture reproduced the assertion with a single direction.
2. Lines 279–284 classify every nonfinite value or score as invalid, then replace **both** with `NaN`. The while-loop alternative repeats the same behavior at lines 428–455. Under the tested XLA builds, the assertion does not stop execution, and this conversion destroys the sentinel.
3. [ledh_dual_parameter_target.py](../../../bayesfilter/inference/ledh_dual_parameter_target.py), lines 116–150, discards the returned diagnostics. Its custom gradient masks only a negative-infinite exact value. A NaN exact value does not satisfy that predicate; an invalid biased score is also unhandled when the exact value is finite.

The isolated unpatched consumer gives the following results:

| Initial particle covariance | CPU XLA value / gradient | RTX 5080 GPU XLA value / gradient |
|---|---|---|
| `0.2 I`, healthy control | `-1.024530361114084` / `[0.09100694457665404]` | `-1.024530361114084` / `[0.09100694457665406]` |
| `-I`, deliberate invalid-input fixture | `NaN` / `[NaN]` | `NaN` / `[NaN]` |

The negative covariance is a deterministic failure-injection fixture; it is not presented as a physically admissible model covariance or evidence of how often posterior proposals fail. It establishes that the advertised response to a numerical failure does not survive the actual route.

The custom-gradient unit test patches the entire batch wrapper to return the desired sentinel. That checks the last layer's behavior under a favorable mock, while bypassing the layer that breaks it. Keep such a test as a small unit check, but add a consumer test that preserves the real wrapper and another that preserves the whole engine.

**Required repair:** propagate explicit value validity, force validity, and failure reason through both batch implementations and the dual target. Compare directional values only where finite, while separately checking that their validity/status agrees. Preserve `-inf` and zero for a declared graceful numerical failure. Treat invalid exact values and invalid biased forces separately; do not silently admit a finite value with a NaN force. Programming errors and incompatible shapes should remain explicit errors rather than being disguised as rejected proposals.

### 3.2 A zero factor is not safe input to a Cholesky derivative

The helper returns a zero matrix when it detects failure. [ledh_canonical_score_stages_tf.py](../../../bayesfilter/highdim/ledh_canonical_score_stages_tf.py), lines 336–390, immediately feeds that factor into two triangular solves when differentiating sigma points. For two one-dimensional negative covariances, the probe returned finite zero sigma-point displacements, validity `[false, false]`, and **NaN tangents** in eager, graph, and XLA execution, on CPU and GPU.

The same structural issue exists in [ledh_unified_reset_tf.py](../../../bayesfilter/highdim/ledh_unified_reset_tf.py), lines 215–254: gap, target, and injected factors are differentiated before their flags can prevent invalid solves. Combining flags after arithmetic does not make that arithmetic safe.

Other relevant gaps remain:

- Canonical-score lines 189–190 factor process and observation covariances directly; line 825 factors a flow innovation covariance directly. The UKF update has another direct factorization at stage-module line 598. A complete operation-to-status audit is needed, including solves, log determinants, normalizations, and domain-sensitive scalar functions.
- Canonical-score lines 490–494 assert that the higher-moment correction is valid before the following sentinel selection. This raises on the actual invalid fixture in ordinary execution. A proposal-dependent numerical failure cannot simultaneously mean “return a sentinel” and “raise an assertion before returning it.”
- The final `overall_valid` at lines 572–582 is recomputed from the current step. No explicit cumulative validity flag is carried as loop state. Later arithmetic after an earlier failure is therefore not protected by a sticky failure state. The exact multi-step failure outcome was **not checked**; tests must force failure at the first, middle, and final observation and verify the final value and force, not only the current-step flag.

**Required repair:** skip invalid solves, or supply a benign finite factor and zero tangent only to an already invalid, inactive branch, then mask and preserve its status. An identity placeholder confined to an invalid branch does not authorize replacing an accepted covariance. Carry cumulative validity through the time loop and stop or safely neutralize subsequent invalid arithmetic. Do not silently ridge, clip, or otherwise change accepted covariances to make these tests pass; that would be a separate numerics-altering proposal requiring its own non-harm evaluation.

### 3.3 What `safe_cholesky` does and does not guarantee

[ledh_numerical_safety_tf.py](../../../bayesfilter/highdim/ledh_numerical_safety_tf.py), lines 84–112, detects `is_nan(chol_attempt)`, not all nonfinite inputs or outputs. It also finishes with an unrestricted `tf.squeeze`.

The diagnostic results expose three distinct problems:

| Input | Observed behavior | Consequence |
|---|---|---|
| Diagonal matrix containing positive infinity | Eager/ordinary graph report valid and retain an infinite factor; XLA reports invalid in this environment | NaN-only validity is incomplete and differs by backend. |
| Identity lower triangle with NaN in the upper triangle | Reported valid with a finite factor | Cholesky consumes the lower triangle. Its success cannot certify that a full covariance tensor is finite and symmetric. |
| Input shape `[2, 1, 2, 2]` | Validity shape becomes `[2]` instead of `[2, 1]` | Singleton batch dimensions disappear, permitting incorrect broadcasting. |

The installed TensorFlow Cholesky documentation specifies a symmetric positive-definite input and use of the lower triangle. It does not promise a universal “all numerical failures become NaN, without any exception” contract across backends. The review used the installed source documentation because online documentation retrieval was unavailable.

Define validity explicitly: compatible shape, finite relevant input, the covariance symmetry contract, finite factor, and strictly positive factor diagonal. Preserve the exact leading batch shape by reducing only over matrix axes. A relative factorization residual and conditioning/margin diagnostics are useful for validation; successful Cholesky alone does not establish accurate derivatives or good conditioning.

Do not invent a universal condition-number cutoff such as `1e12`, or a universal minimum covariance such as `1e-10`. Their meaning changes with dtype, scaling, and dimension. The implementation-plan statement that this is an `O(1)` check is also wrong: scanning an `n × n` factor is `O(n²)`, in addition to the factorization. The claim that Cholesky necessarily returns NaN on ill-conditioned matrices is false; many such matrices factor successfully.

## 4. The LGSSM mathematics needs correction

### 4.1 State observability is not parameter identifiability

The plan's conditions are neither generally sufficient nor generally necessary for identification of an unspecified parameter vector. They concern a state representation, its excitation, and one initialization choice. Parameter identification asks whether distinct allowed parameter values produce distinct observation laws.

For example, for any invertible state-coordinate change `z = Sx`, set

\[
F_S=SFS^{-1},\quad H_S=HS^{-1},\quad Q_S=SQS^\top,
\quad m_{0,S}=Sm_0,\quad P_{0,S}=SP_0S^\top.
\]

The transformed system has exactly the same observation distribution. Observability, controllability, and stationary initialization survive the transformation. Consequently, treating all those matrices as unknown does not identify their entries without further restrictions. Even a scalar unknown observation loading can be confounded with the latent-state scale.

Conversely, a model can contain an unobserved state independent of an identifiable parameter in its observed subsystem. Full-state observability is then unnecessary for identifying that parameter. With `Q` positive definite, the input matrix `Q^(1/2)` already has full row rank, so the stated controllability check supplies little additional discrimination. Positive-definite `R` is observation noise, not “sufficient excitation”; adding noise does not create information about an otherwise invisible parameter.

For each case, list the entries or transformations in `theta`, what is fixed, the prior/support, the parameter-to-observation-law map, and the remaining symmetries. Distinguish structural identification from the amount of information in a short dataset. A ten-observation sample need not recover its generating parameters precisely even when the model and sampler are correct.

### 4.2 The Lyapunov formula is correct on its stated stable domain

If `rho(F) < 1` and `Q` is positive semidefinite, the convergent series

\[
P_0=\sum_{j=0}^{\infty}F^jQ(F^\top)^j
\]

satisfies `P_0 - F P_0 F^T = Q`. Vectorization gives

\[
(I-F\otimes F)\operatorname{vec}(P_0)=\operatorname{vec}(Q).
\]

Solve the linear system; do not form its inverse. TensorFlow's row-major reshape is consistent here because the left and right multipliers are `F` and `F^T`, giving the same `F ⊗ F` representation. A solver succeeding outside the stable domain does not establish that its result is a covariance.

For dimension 20 the matrix is `400 × 400`, with 160,000 entries: 1.28 MB in float64 or 0.64 MB in float32 before solver workspace. A dense solve scales as `O(n^6)` in state dimension, but at this size it is a reasonable first implementation to measure. Repetition over HMC evaluations and tangent directions, conditioning near instability, and compilation matter more than the matrix's storage alone.

**Recommendation:** use a TensorFlow linear solve for this bounded first implementation, with stable signatures and recorded XLA checks. The installed TensorFlow has no `tf.linalg.solve_lyapunov`. A SciPy discrete-Lyapunov result is permitted as an independent diagnostic reference, not as the runtime parameter-to-covariance path. Consider a dedicated Schur-based or doubling solver only if measured cost or conditioning justifies the added implementation. An iteration stopped by a parameter-dependent numerical criterion also needs a precisely defined differentiated program; it is not an automatic improvement.

Check stability, symmetry, positive definiteness where required, and a scale-relative residual such as

\[
\frac{\|P_0-FP_0F^\top-Q\|_F}
{\|P_0\|_F+\|FP_0F^\top\|_F+\|Q\|_F}.
\]

Handle an all-zero denominator explicitly if semidefinite zero-noise cases are admitted. Report distance to instability and conditioning as diagnostics. Validate each proposed parameter value's allowed domain, not merely the generating truth.

### 4.3 Verification of the four cases

| Case | Checked verdict |
|---|---|
| 1: scalar `phi=0.8`, `Q=1`, `H=1` | Stable and observable; `P_0=1/(1-0.8²)=25/9=2.777777…`. The rounded `2.778` is correct for exposition, but should not be the numerical implementation. Identification/recovery still depends on which parameters are inferred. |
| 2: the stated bivariate `F` | Characteristic polynomial `lambda²-1.3 lambda+0.4`, hence eigenvalues `0.8` and `0.5`. `H=I` gives full observability; `Q` is positive definite since its leading diagonal entry is positive and determinant is `0.71`. A stationary covariance exists. |
| 3: `F=0.8 I+0.1 11^T` | **Wrong as stated.** Along the all-ones direction the eigenvalue is `0.8+0.1×4=1.2`; the other eigenvalues are `0.8`. With `Q=I`, the formal Lyapunov solution has eigenvalues `-25/11, 25/9, 25/9, 25/9`, so it is not a covariance. |
| 4: unspecified tridiagonal `F`, varying `Q`, random `H` | **Not checkable.** Exact coefficients, scales, projection construction/seed, parameterization, and all noise/initialization settings are required. |

Case 3 also has an unobservable direction `v=(0,0,1,-1)^T`: `Hv=0`, `Fv=0.8v`, and therefore `HF^kv=0` for every `k`. The numerical observability rank is 3, with smallest singular value approximately `2.78e-17`. Simply reducing the rank-one coupling can fix stability while leaving this invisibility intact.

The diagnostic Lyapunov solve for the invalid Case 3 had a residual norm of `8.74e-16`. This is a useful counterexample: an excellent equation residual does not turn an indefinite solution into a stationary covariance.

A concrete replacement worth specifying is a symmetric tridiagonal `F` with diagonal `0.6` and adjacent off-diagonals `0.1`. Its eigenvalues lie within `[0.4,0.8]` by the row-sum bound. Observing the first coordinate already reaches successive coordinates through the nonzero adjacent couplings: row `e_1^T F^k` first reaches coordinate `k+1` with coefficient `0.1^k`, giving full exact observability. Observing the first two coordinates in dimension 4, or the first ten in dimension 20, retains that property. This is a **proposed replacement**, not a change made by this review; numerical conditioning must still be measured.

### 4.4 Stationary initialization remains part of the likelihood and score

Stationary initialization replaces an arbitrary transient distribution with a specified model assumption. It does not remove initial-distribution dependence from a finite-horizon likelihood:

\[
p_\theta(y_{1:T})=\int p_\theta(x_0)
\prod_{t=1}^{T}p_\theta(x_t\mid x_{t-1})p_\theta(y_t\mid x_t)\,dx_{0:T}.
\]

If `F`, `Q`, or the initial mean depend on `theta`, their effects on `p_theta(x_0)` must be included. Differentiating the Lyapunov equation in a direction gives

\[
dP_0-F(dP_0)F^\top
=(dF)P_0F^\top+FP_0(dF)^\top+dQ.
\]

Thus the tangent uses the same linear operator with a different right-hand side. For fixed base particles `z`, an initial cloud `x_0=m_0+L_0z` requires `dx_0=dm_0+(dL_0)z`, as well as the correct per-particle covariance tangent. Resampling fresh base particles at each HMC evaluation would define a different stochastic target unless an explicitly valid extended-state method were used.

The single-cloud function accepts `initial_state_tangent` and `initial_covariance_tangent` at canonical-score lines 692–693. The reviewed fused wrapper does not expose/forward those arguments. Its `PerPointScoreModel` fields at batch-module lines 30–43 contain covariance tensors without the corresponding parameter-covariance tangent callbacks. This is a concrete consumer capability gap for stationary `F/Q` inference and variance-parameter models. An adapter existing elsewhere does not establish that this endpoint can use its capabilities.

Require an executable derivative check from the actual parameterized consumer to the same finite value program, including initialization, `Q`, `R`, observation dependence, reset moments/weights, and transport. Finite differences and autodiff may serve as independent diagnostic comparisons; the claim-bearing LEDH score must remain analytical. Freezing `P_0` while claiming the stationary model's total score is wrong relative to that claim.

For numerical observability, use a relative SVD rank diagnostic, for example `tau = c × max(rows, columns) × eps(dtype) × s_max`, with the multiplier and scaling declared. Report singular values and the smallest-to-largest ratio rather than only an integer rank. An absolute `1e-10` has no universal meaning. Exact symbolic reasoning is valuable for these constructed small examples; floating-point rank alone is neither structural parameter identification nor practical estimability.

## 5. The nonlinear model specifications and routes

### 5.1 Predator–prey: fix the equations before generating data

The plan's equations at lines 170–177 are wrong relative to [models.py](../../../bayesfilter/highdim/models.py), lines 1905–1919. The local implementation uses

\[
J(x,y)=\frac{xy}{a+x},\qquad
\dot x=rx(1-x/K)-sJ(x,y),\qquad
\dot y=uJ(x,y)-vy.
\]

The plan instead places `s` in the denominator and `a` in both interaction numerators, including an extra factor of `a` in predator growth. These are different parameter meanings and different dynamics. The canonical predator–prey adapter at [ledh_canonical_models_tf.py](../../../bayesfilter/highdim/ledh_canonical_models_tf.py), lines 364–389, agrees with the local model.

Specify the observation interval, RK4 substeps/variant, covariance versus standard-deviation convention, initial distribution, support, and transformation/prior Jacobians. Gaussian process noise and “populations must stay positive” do not by themselves define a consistent support rule: record the actual domain policy, and do not silently clip or truncate the model.

For the first parameter-inference smoke, a bounded two-parameter problem such as `(r,v)` with `(K,a,s,u)` fixed is easier to diagnose than six-dimensional recovery at `T=20`. These parameters directly affect prey growth and predator loss, respectively. This is a proposed diagnostic restriction, not a proof that they are well identified. Inspect their observation sensitivities and reference posterior. Preserve a later six-parameter validation if the intended claim concerns the full model; a restricted test cannot stand in for it.

### 5.2 SIR: this first case is a filtering test

The local SIR equations at model lines 902–919 are

\[
\begin{aligned}
\dot S_j&=-\kappa_jS_jI_j+
\tfrac12\left(\sum_{k\in N_j}S_k-d_jS_j\right),\\
\dot I_j&=\kappa_jS_jI_j-\nu_jI_j+
\tfrac12\left(\sum_{k\in N_j}I_k-d_jI_j\right).
\end{aligned}
\]

They combine local infection with diffusion of susceptible and infectious states. The plan's neighborhood-pooled, degree-normalized infection equations omit these diffusion terms and define another model.

`SpatialSIRSSM.parameter_dim()` returns **0** at model lines 705–706. Therefore the proposed fixed-`kappa`, fixed-`nu` case is a **filtering, partial-observation, and numerical-validity test**, with no parameter posterior, parameter ESS, or parameter-recovery criterion. For `J=3`, there are three locations and six tracked state coordinates, of which the three infectious coordinates are observed. Calling this eight parameter-recovery configurations is incorrect.

The existing `ParameterizedZhaoCuiSIRSSM`, lines 935–976, has three parameters: log infection-rate scale, log recovery-rate scale, and log observation-noise standard-deviation scale. The covariance scales by `exp(2 theta_3)`. If SIR inference is needed, use this existing definition as the starting point rather than inventing an unidentified wrapper. First verify complete analytical derivatives and that the actual consumer accepts its parameter-dependent observation covariance. Add its prior, initial state, and interval/noise settings explicitly.

This review checks **local-code agreement only**. It does not certify Zhao–Cui source-faithfulness. Any new Zhao–Cui source-route behavior or such claim still needs the applicable paper and author-source anchors under repository policy.

### 5.3 SV needs its actual state-dependent observation density

The stated basic SV observation is zero-mean with variance `beta² exp(x_t)`. It is not a nonlinear mean `beta exp(x_t/2)` plus constant additive noise. A generic additive-noise model with that mean would be a different likelihood.

Moreover, with zero conditional observation mean, the ordinary UKF state/observation cross-covariance from that mean is zero. Merely selecting a scalar nonlinear model does not establish that the flow proposal assimilates volatility information correctly. The plan must identify the actual density/tangent callbacks and the proposal construction, then check their connection to the consumer. A mixture approximation to transformed SV observations, where available, is a separately defined approximation and cannot silently replace the stated density.

At `gamma=0.95`, `sigma=1`, the stationary latent variance is approximately `10.2564`, with standard deviation about `3.20`. This makes latent scale and exponential tails relevant, but does not prove that overflow will occur in `T=20` or `T=40`. Nor does a longer particular sample guarantee a tighter posterior for every parameter. Those are empirical questions.

`GeneralizedSVPriorMeanSSM` is a useful later extension if it exercises a distinct initial-mean or scale derivative path. It is not required merely to increase the model count. Match the exact class, parameter dimension, and likelihood; similarly named canonical generalized-SV adapters are not automatically interchangeable.

Some existing model/adapter modules also contain legacy NumPy dependencies or fixed float64 assumptions. Their existence is not proof of eligibility for the default TensorFlow float32/TF32 GPU route. Audit and migrate the touched consumer path as needed; keep any independent reference implementation explicitly diagnostic.

## 6. A discriminating failure-test design

### 6.1 Exact pathological fixtures

There is no universal small positive `Q` or `R` that must make Cholesky fail. In the executed checks, `Q=1e-20 I` factored successfully in both tested dtypes and all three execution modes. Its condition number is 1; uniformly shrinking a well-conditioned matrix is not the same as making it ill-conditioned.

Use separate fixture families with explicit expectations:

| Family | Exact proposed inputs | Required interpretation |
|---|---|---|
| Healthy scale controls | `I`, `1e-20 I`, `1e20 I` | Finite factorizations and accurate scale-relative residuals where representable. These are no-fire controls, not guaranteed failures. |
| Conditioning ladder | `diag(1, 10^-k)` for `k = 0, 4, 8, 12, 16`, plus a fixed orthogonal rotation | Measure residuals, tangent error, and validity by dtype/backend. Large condition number alone is not an expected sentinel. |
| Near-singular cancellation | `[[1, 1-delta], [1-delta, 1]]`, with `delta` chosen from `1e-2`, `32 eps`, `4 eps`, and a value that rounds to zero | Record the realized matrix and eigenvalue margin. Mathematically positive inputs can become singular after rounding. |
| Deliberately invalid factor inputs | `diag(1,0)`, `diag(1,-1)`, `[[1,2],[2,1]]`, NaN/Inf entries in both relevant triangles | Deterministic invalid status with safe downstream behavior, or an explicitly documented unsupported-input error before sampling. These are mechanics fixtures, not admissible positive-definite model covariances. |
| Actual-engine injection | A known-good fixture with one invalid predicted, innovation, reset-gap, or correction covariance | Verify reason, location, propagation, final sentinel, and zero returned force. Include first/middle/final observations and mixed-validity batch rows. |
| Physical parameter stress | Declared prior-supported low noise, near-stationarity, stiff dynamics, or extreme volatility | Compare against a reliable reference. A failure is not automatically a correct rejection if the intended model remains valid. |

This ladder specifies concrete inputs without pretending that an arbitrary dimensional scale is a mathematical boundary. Priors and physically meaningful ranges must be model-specific. Deterministic fixtures remain separate from the data reserved for scientific claims.

### 6.2 Force an invalid proposal; do not wait for chance

Start from a valid state with finite target and force. For a one-dimensional mechanics target whose valid domain is known, choose a fixed initial momentum and step size that place a one-leapfrog-step endpoint outside that domain. For example, for support `theta > 0`, a flat interior force, `theta_0=0.1`, unit mass, momentum `-1`, and step size `0.2` give an endpoint `-0.1`.

The test must assert that the invalid evaluation actually occurred, the endpoint target is `-inf`, the returned force is finite zero under the declared convention, the complete HMC acceptance calculation rejects, and the previous valid state is retained. Record the reason/counter. A test that merely allows either a finite value or `-inf` can pass without testing failure handling.

Then repeat through the **real LEDH consumer** using a deterministic failure injection with the real wrapper intact, and finally a model parameter fixture whose relevant numerical boundary has been demonstrated. Test independently: invalid exact-value route, valid exact value with invalid biased-force route, and both invalid. Starting the chain at `-inf` is not a substitute; initialization must reject an invalid starting state before ordinary sampling.

Do not infer that every trajectory visiting an invalid interior point must be rejected solely from its endpoint. If a deterministic finite force extension lets a trajectory return to a finite endpoint, an endpoint Metropolis correction can be legitimate only under a reversible, volume-preserving proposal. Conversely, an early-abort rule needs symmetric forward/reverse handling. The proposed behavior must be stated and tested; stopping at whichever invalid intermediate operation happens first is not automatically valid HMC.

### 6.3 Distinguish mathematical support from numerical truncation

Returning `-inf` outside the true model support is appropriate when that support is part of the declared target. Returning `-inf` because the numerical evaluator failed at a mathematically valid parameter excludes a different set.

For a deterministic finite-program likelihood `L_hat(theta)` and prior `p(theta)`, a numerical validity mask `V(theta)` executes

\[
\widetilde\pi(\theta)\ \propto\ p(\theta)\widehat L(\theta)
\mathbf 1\{V(\theta)=1\}.
\]

This is generally a truncated version of the unmasked finite-program target. If `A` is the failed region and `pi(A)<1`, then the total-variation distance between `pi` and `pi(. | A^c)` equals `pi(A)`: the lost mass on `A` and the renormalization on its complement each contribute half that amount. Thus “HMC continued” cannot establish that numerical failures are harmless. Proposal failure frequency alone does not estimate the excluded posterior mass.

Keep three targets distinct: the exact state-space posterior, the posterior using a specified finite LEDH likelihood program, and the numerically masked version of that program. “Exact value” inside `DualParameterLEDHTarget` means its chosen endpoint value configuration; it does not establish exact marginalization of a nonlinear state-space model. Add the declared prior and any coordinate-transform Jacobian explicitly when constructing a posterior.

Zero force on an invalid region is a computational convention, not the derivative of a finite log density there. Inside the healthy region, preserve the analytical derivative of the stated finite program. If a cheaper biased force is intentionally used, correctness requires the corresponding deterministic proposal and endpoint correction conditions, not a claim that this force equals the exact score.

The installed TFP Metropolis–Hastings implementation combines target and proposal/kinetic correction terms, then uses a nonfinite-safe sum. In the inspected version, nonfinite sums become `-inf`. Therefore a chain can reject a NaN-corrupted proposal and keep running while concealing broken derivatives. Check intermediate target/force statuses; “no crash” is a mechanics observation, not a numerical-validity certificate.

## 7. Replace the success criteria with evidence that answers the question

### 7.1 Separate engineering, numerical validity, and posterior inference

| Question | Primary criterion | Veto or repair trigger | Explanatory diagnostics |
|---|---|---|---|
| Does graceful handling work? | Every deterministic failure fixture reaches the declared consumer result and rejection behavior; healthy outputs are unchanged within a justified tolerance | Exception on a supported numerical-failure case, NaN output, wrong shape, nonzero invalid force, missing failure reason, or alteration of accepted healthy results | Counts by operation, time index, batch row, and reason |
| Does the healthy analytical program compute its stated quantity? | Actual consumer agrees with independent value/derivative checks for the same finite program, including initialization and all parameter paths | Missing total-derivative term, incorrect model density, wrong reset route, inconsistent value/force status | Scale-relative residuals, sensitivity conditioning, FP32/FP64 differences |
| Does filtering behave correctly? | Predeclared agreement with exact Kalman filtering on LGSSM, and justified nonlinear references where available | Wrong likelihood/state moments beyond declared numerical and Monte Carlo uncertainty; invalid covariance or weights | State RMSE, interval coverage over replications, likelihood error, proposal/weight diagnostics |
| Does HMC sample the intended implemented target? | Agreement with a tractable reference posterior plus convergence and Monte Carlo precision criteria | Target/status failures, lack of chain movement, unexplained divergences, or materially excluded valid posterior mass | Acceptance, rejection reasons, energy errors, autocorrelation, runtime |
| Is a broader scientific or default claim justified? | Scope-specific untouched validation and the repository's canonical conformance/admission requirements | Any relevant numerical, source, tuning, or posterior gate fails | Performance and comparisons that remain descriptive without uncertainty |

For safety adoption, the central criterion is **non-harm on healthy trajectories and bounded, flagged behavior on failures**, not improved ESS or posterior accuracy. An arbitrary `<5%` overhead threshold must not reject a necessary low-cost safety check. Measure compilation and steady-state costs separately after correctness, with repeated paired timings and uncertainty; diagnose material overhead before making a performance promise.

### 7.2 Posterior checks

“Posterior mean within two posterior standard deviations of the generating truth” is not a correctness criterion. A diffuse or prior-dominated posterior can pass it; a correct posterior from a particular dataset can fail it. Changing two to one or three standard deviations does not fix the problem. Multiple parameter/model checks also change the chance of at least one failure.

For a one- or two-parameter LGSSM, compute a reference posterior using the exact Kalman likelihood and a controlled quadrature/grid or another independently validated method. Compare the implemented-target sampler against its own reference first, and assess the LEDH-versus-Kalman approximation separately. Set tolerances using reference error and Monte Carlo standard errors. Repeated synthetic datasets can assess coverage/calibration, with binomial or other appropriate uncertainty; a single generating truth in a single short dataset cannot do so.

ESS above 100 alone is weak. Even under favorable asymptotics, ESS 100 gives a posterior-mean standard error around `0.1` posterior standard deviations, with no general assurance for tails. Use multiple dispersed valid chains, maximum rank-normalized split and folded R-hat at the repository admission threshold `<=1.01`, bulk and tail ESS, and an MCSE target tied to the quantities being reported. R-hat `<1.1` is too weak for the proposed correctness claim. None of these diagnostics proves convergence in isolation.

Intentional support rejection is not automatically a Hamiltonian divergence. Report both with explicit definitions. Zero NaNs in retained samples can coexist with NaNs in every rejected proposal, so retained-sample checks are insufficient.

Drop “at least 7/8 pass.” All predeclared mandatory correctness cases must pass. A separately identified capacity or exploratory case may fail and motivate repair, but its failed scope remains unvalidated. The fixed-parameter SIR case has different criteria and must not be counted as successful parameter recovery.

### 7.3 Filtering references and simple comparators

Add filtering validation. LGSSM supplies an exact Kalman oracle for log likelihood, filtered means, and covariances. Compare the numerical approximation with that oracle under stated particle/tuning error, rather than demanding unexplained exact equality from a finite-particle algorithm.

When interpreting a complex filtering method, construct simple comparators from each model and evaluate them conditionally. A suitable initial set is:

- **Prediction-only estimator:** propagates the declared state model without assimilating observations; detects whether the update adds useful information.
- **Stationary or unconditional mean:** where defined, exposes failure to outperform a very simple forecast in ordinary regimes.
- **Observation plug-in or last-observation estimator:** defined only for the observed, invertible coordinates; do not manufacture an inverse for hidden SIR states or variance-only SV observations.
- **Plain UKF:** checks the contribution of the more complex particle-flow machinery where its observation assumptions apply.
- **Bootstrap particle filter:** a general nonlinear comparison; a high-particle reference needs its own replication/Monte Carlo error assessment.

Specify the exact formula for each comparator in each applicable case, and the conditional situations before running: healthy interior, near support boundaries, high/low observation noise, stiff dynamics, and concentrated weights as relevant. For LGSSM, retain the exact Kalman oracle alongside this set. Do not tune to the sanity-check cases. Inferiority to a simple comparator in a declared salient situation is a promotion veto and a headline result, not grounds to conceal a failing configuration in an average.

No stochastic ranking was attempted in this review. These requirements govern a future claim-bearing comparison, not the interpretation of the deterministic defect reproductions above.

## 8. A practical revised sequence and resource contract

1. **Repair the mechanics first.** Correct R1/R2/R6, fix malformed tests, and add finite healthy controls plus deterministic consumer failure cases. Check scalar and batched inputs, singleton batch axes, multiple tangent directions, mixed-validity rows, first/middle/final observations, CPU reference modes, and GPU/XLA. Keep changes localized.
2. **Freeze correct model specifications.** Replace Case 3, fully specify Case 4, correct predator–prey/SIR equations, declare every inferred parameter/prior/coordinate transform, and connect stationary-initialization and covariance tangents. Verify value/analytical-score agreement through the actual endpoint before HMC.
3. **Validate small filtering and posterior problems.** Begin with scalar/bivariate LGSSM exact references and actual-consumer forced rejection. Establish that healthy results do not change. Add the nonlinear density/derivative and filtering checks before attempting broad parameter recovery.
4. **Run the declared nonlinear and scale ladder.** Each changed horizon, particle count, dimension, backend, or other tuning-scope field requires its own eligible offline calibration/validation and frozen artifact before an untouched claim run. Keep the fixed SIR case filtering-only unless a separately specified inference case is added.
5. **Assemble the result and decide by scope.** Preserve failures and repairs, uncertainty, reason counts, and unvalidated scopes. A rejected candidate can motivate the next planned repair; only a true continuation veto or exhausted budget stops the campaign.

Before step 3 becomes a serious campaign, add one concise execution plan containing:

- Exact model/data versions, generating truths, inferred parameters, priors, support/transforms, and synthetic-data, particle, calibration, validation, and HMC seeds. Separate tuning and untouched claim partitions.
- The actual canonical callable and settings: Contract E reset identity, UKF covariance lifecycle, analytical total score, required GenUT/dual-cap controls, fixed random/base designs, ridge policy and rationale, flow/transport settings, and repository-issued scope-matching tuning evidence. Defaults such as `reset_policy="none"` or zero correction steps cannot silently define a claim-bearing canonical case.
- Particle counts and the enforced `dpf_transport_exact_divisor_cap3000_v1` chunk rule. No independent tiny chunk defaults. For fixtures below the cap, use `K=N`.
- GPU float32/TF32/XLA as the intended execution route, with memory-growth provenance. Label float64, CPU, and non-JIT checks as references or diagnostics. `DualParameterLEDHTarget.as_graph_callable` currently uses `tf.function` without `jit_compile=True` at lines 180–196; verify the enclosing execution route rather than equating “traced” with XLA.
- HMC chain count, valid dispersed initialization, mass/preconditioner, step-size/leapfrog tuning ranges, adaptation, maximum warm-up and retained transitions, chunk/monitoring schedule, MCSE/ESS/R-hat criteria, and the target/force combination. Prefer at least four chains for the initial validation, subject to an explicit resource calculation. “500 warm-up and 1000 draws” is a proposal to test, not a universal adequacy guarantee.
- Per-attempt and total GPU-time limits, maximum attempts, artifact root, and continuation vetoes. Include compilation, tuning, repairs, and reference construction in the budget. Conduct a bounded timing smoke before assigning a wall-clock schedule; no measured evidence supports either “3–4 days” or “5–7 days” yet.

The reviewed [tuning capability registry](../../../bayesfilter/inference/tuning_contract.py) distinguishes artifact-authority tuners from chain runners and diagnostic prototypes. Use the supported ordinary-coordinate `tune_hmc_kernel` authority for the appropriate target contract. If using an intentionally different endpoint-corrected force, bind the supported typed neural-force runner/contract rather than presenting it as an exact target gradient. A fixed-transport tuner is appropriate only when there is an actual supported transport and its Jacobian contract. A diagnostic TensorFlow prototype does not become an artifact-authority tuner by being callable. Existing NumPy migration debt on a chosen legacy route must be resolved within scope before claiming conformity to the repository's runtime policy.

The NeuTra sequential-controller policy applies if the campaign actually uses NeuTra; ordinary LEDH HMC should follow its own supported interface rather than inheriting an unrelated runner by name. No NeuTra training or new backend exception is required by this review.

A rough work estimate should count chains, warm-up/retained transitions, leapfrog force calls, particle count, horizon, flow/reset work, and analytical directions. The dual target evaluates an endpoint-value configuration and a biased-score configuration; its cost is not that of a single scalar likelihood. The dimension-20, `T=120` case should follow measured small-case capacity checks, not be the first debugging job. Current GPU availability is not a reservation or evidence that concurrent long runs will fit.

**Continuation vetoes** should include corrupted/missing evidence, wrong target or derivative, missing required tuning, invalid reference, unsupported consumer route, exceeded budget, and resource failure that cannot be repaired within the same contract. A failed posterior-recovery screen is a **candidate failure/repair trigger**, not proof that all canonical LEDH scores are broken. Diagnose the implementation, model specification, parameter information, tuning, and sampler separately.

## 9. Direct answers to every memo question

### Mathematical correctness

| Question | Answer |
|---|---|
| **Q1.1** | No. Observability/controllability and stationarity do not identify an unspecified parameter vector. State-coordinate transformations give an explicit counterexample; positive-definite observation noise is not excitation. See §4.1. |
| **Q1.2** | The discrete Lyapunov formula is correct for stable `F`; solve rather than invert. Use TensorFlow's dense linear solve initially at these sizes, checking stability, PSD/PD, residual, and the analytical tangent. SciPy is an independent reference only. See §4.2–4.4. |
| **Q1.3** | Case 1 is correct; Case 2 has eigenvalues `0.8,0.5`; Case 3 is unstable and rank-3 observable; Case 4 is underspecified. See §4.3. |
| **Q1.4** | Stationarity specifies the initial law; it does not remove it from the finite-horizon likelihood. Its parameter dependence must be differentiated, including fixed-base initial particle construction. See §4.4. |

### Implementation feasibility

| Question | Answer |
|---|---|
| **Q2.1** | A `400×400` dense solve is a reasonable bounded starting point, with 1.28 MB of float64 matrix storage before workspace. Benchmark repeated solves and derivatives before replacing it. The installed TensorFlow lacks `tf.linalg.solve_lyapunov`. |
| **Q2.2** | Use scaled numerical rank with dtype-relative tolerance, singular-value reporting, and exact structural reasoning where possible. A fixed `1e-10` threshold is not justified across scales/dtypes. |
| **Q2.3** | No small positive noise level guarantees failure. `1e-20 I` succeeds here. Use the explicit healthy, conditioning, cancellation, invalid-matrix, and actual-engine fixtures in §6.1; keep mathematical degeneracy separate from numerical failure on a valid target. |
| **Q2.4** | Yes. Specify the HMC tuning authority, search/adaptation protocol, coordinate system, chain policy, stopping limits, diagnostics, and total budget in the plan. Let the eligible tuner select permitted settings; do not bury unexplained constants in individual tests. |

### Test coverage

| Question | Answer |
|---|---|
| **Q3.1** | Add a model only for a missing mechanism. Generalized SV can exercise mean/scale derivatives; parameterized SIR is appropriate if SIR inference is part of the claim. First establish valid adapters and the correct likelihood. Neither is mandatory merely to enlarge the list. |
| **Q3.2** | No universal transition occurs at dimension 10. More useful coverage includes mixed/singleton batches, parameter directions, observation rank, conditioning, and reset behavior. Use an intermediate dimension as a capacity bridge if measurements justify it. |
| **Q3.3** | There is no universal critical horizon. Force failure at first/middle/last times and assess accumulated numerical error. Extend to `T=100` or `200` only under a question, scope-specific tuning, and budget; `T=40` is merely longer than `20`, not broadly “long horizon.” |
| **Q3.4** | Yes. Check likelihood and filtered moments against Kalman references, then nonlinear references with their own uncertainty. Track healthy-region equivalence and rejected numerical paths separately. |

### Success criteria

| Question | Answer |
|---|---|
| **Q4.1** | Neither 1, 2, nor 3 posterior SDs solves the problem. Compare against a reference posterior and use repeated-data calibration if claiming recovery/coverage. Distinguish posterior spread from Monte Carlo error. |
| **Q4.2** | ESS 100 alone is insufficient. Use modern split/folded R-hat `<=1.01`, bulk/tail ESS, MCSE requirements, valid initialization, and movement/status checks. R-hat `<1.1` is too permissive here. |
| **Q4.3** | The 7/8 rule has no stated justification. Require every mandatory correctness case; predeclare exploratory/capacity cases and report their failures without awarding their scope a pass. SIR filtering has separate criteria. |
| **Q4.4** | Measure overhead after correctness, but do not invent a 5% safety rejection threshold. Report compile cost and repeated paired steady-state timings with uncertainty. Necessary guards are judged first on non-harm and containment. |

### Risk assessment

These ratings concern the likelihood of blocking **this plan as written**, not estimated posterior failure probabilities. Empirical rates remain unknown.

| Question | Rating | Assessment and mitigation |
|---|---|---|
| **Q5.1** | **High** | Baseline execution/validation is already blocked by reproduced wiring defects and an invalid model case. Baseline recovery failure would not uniquely implicate the canonical score: wrong priors, poor information, missing derivatives, poor tuning, or inadequate sampling can also cause it. Repair mechanics, use exact LGSSM references, and classify the failure before continuing. |
| **Q5.2** | **High** | The plan has no test of numerical exclusion of valid posterior support. Log failures by reason/location and parameter region; compare suspicious points against higher-precision or independent references; distinguish true support violations from evaluator failures. Rejection rates alone cannot prove negligible target distortion. |
| **Q5.3** | **High** | Natural HMC exploration can leave the failure mechanism untested, allowing a vacuous pass. Use the deterministic valid-start/forced-invalid-endpoint tests in §6.2 and require a positive encounter count. Do not initialize the ordinary chain at an invalid state. |
| **Q5.4** | **High for the present helper; medium for model-specific conditioning calibration** | Infinite-input acceptance, invalid tangents, and shape loss are reproduced, so the current uniform NaN-only contract is inadequate. Repair finite/shape/status handling uniformly first; then study conditioning by dtype/scale/model without ad hoc per-model thresholds or silent regularization. |

### Timeline and resources

| Question | Answer |
|---|---|
| **Q6.1** | Neither 3–4 nor 5–7 days is justified yet. Repairs, target adapters, tuning, reference construction, and compilation precede the larger runs. Estimate from a bounded small-case timing study and separate engineering time from total GPU budget. |
| **Q6.2** | GPU/XLA is the project execution target; CPU/float64 checks are explicit references. Plan GPU sharing, memory growth, and per-run limits. Reserve a compute window only after measuring demand; a review request does not authorize an unbounded campaign. |
| **Q6.3** | Use structured JSON for configuration, diagnostics, checks, and machine-readable decisions, with a Markdown interpretation and links to full logs/sample arrays. LaTeX adds no necessary value to this engineering validation. |

### Integration

| Question | Answer |
|---|---|
| **Q7.1** | Put deterministic, fast mechanics and shape/route regressions in the ordinary suite. Mark GPU/XLA integration separately. Put long posterior campaigns behind an explicit runner and plan, not in routine pytest execution. |
| **Q7.2** | Start with a declared low-dimensional diagnostic such as `(r,v)` and fixed `(K,a,s,u)` after correcting the equations, then expand if the claim needs all six parameters. Check sensitivity and posterior information; do not describe the restriction as full-model validation. |
| **Q7.3** | The proposed base SIR case is filtering-only: it has zero parameters. Use the existing three-parameter SIR definition for a separately specified inference case after derivative and source-route checks. |
| **Q7.4** | No long HMC suite in pre-commit. Run fast deterministic checks there if consistent with repository practice; schedule GPU/campaign validation on demand or in a suitable bounded CI job. |

### Reproducibility

| Question | Answer |
|---|---|
| **Q8.1** | Yes. Preserve synthetic observations/truth, fixed particle/base randomness, tuning results, warm-up and retained samples with distinct roles, diagnostics, and failure records. Use lossless array artifacts and hashes rather than embedding every draw in Markdown. |
| **Q8.2** | Yes, for data, particles/reset designs, tuning partitions, HMC initialization/momenta, and replications. Record seed derivation and actual generated data because kernels/platforms can differ. Avoid reusing a claim dataset for repair tuning. |
| **Q8.3** | The current memo is a review request. A concise updated implementation/checkpoint note should record the accepted repaired plan and exact next action; after execution, the result note should state what actually happened. No new chain of approval-only memos is needed. |
| **Q8.4** | Yes. Use unique versioned directories under `docs/plans/artifacts/`, Git commit plus relevant dirty-source hashes, exact commands/environment, device and memory policy, seeds, wall time, plan/result paths, and immutable prior attempts. Do not overwrite failed evidence. |

## 10. Existing-test disposition and final decision

The focused run included:

```text
tests/highdim/test_ledh_numerical_safety_tf.py
tests/highdim/test_ledh_reset_validity.py
tests/inference/test_dual_parameter_target_invalid.py
tests/highdim/test_ledh_canonical_score_ukf_tangent.py
tests/integration/test_ledh_hmc_graceful_failure.py
tests/integration/test_ledh_graceful_failure_integration.py
```

The final file was not listed in the memo's integration inventory. Its normal/pathological model tests provide initial states shaped `[1,50,2]` to an interface expecting `[N,d]`, with similarly incompatible covariance/noise dimensions. Both fail before answering the graceful-failure question. Repair those fixtures; do not remove them from the count and then advertise the remaining toy tests as end-to-end validation.

The passing reset tests establish only their exercised healthy cases. The toy HMC tests and a hand-computed rejection probability do not verify the actual consumer, its kinetic correction, or a guaranteed invalid encounter. The 27 passing tests remain useful local evidence, but do not contradict the reproduced call-chain defects. This review does not claim the memo's earlier 26-test run never occurred; it reports the present checked suite and its limits.

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | What is not concluded |
|---|---|---|---|---|---|
| **Request major revisions** | Real-consumer sentinel contract fails; proposed model/identification criteria are insufficient | Reproduced NaNs/assertions; invalid Case 3; mismatched model equations; incomplete campaign specification | Frequency and posterior impact of numerical failures are unmeasured; ordinary-graph anomalies remain unexplained | Repair the local failure chain and test fixtures, correct/specify models and score paths, then run a bounded reference-led campaign | No claim of production readiness, posterior correctness, broad safety, or rejection of LEDH as a research direction |

| Inference status | Review conclusion |
|---|---|
| Hard veto screen | Specific deterministic engineering failures are established. |
| Statistically supported ranking | None attempted or supported. |
| Descriptive-only differences | Timings and tiny healthy CPU/GPU numeric differences are descriptive; they rank no algorithm or backend. |
| Default-readiness | Not established for graceful handling. The owner-designated GPU/XLA production direction remains a separate policy decision. |
| Next evidence needed | Repaired endpoint failure/no-fire checks, complete analytical derivative wiring, exact/reference filtering and posterior comparisons, and untouched scope-specific validation with uncertainty. |

**Post-review red-team assessment.** The strongest alternative explanation for some observations is behavior specific to the installed self-built TensorFlow/TFP stack; the ordinary-graph empty outputs and timeout particularly need localization. That explanation does not remove the source-level conversion of every nonfinite value to NaN, and the same sentinel failure was reproduced in isolated CPU and GPU XLA processes. A repaired real-consumer test returning `-inf` with a finite zero force, preserving healthy outputs, and producing a correct full-HMC rejection would overturn the narrow engineering finding. It would not by itself resolve target truncation, omitted derivatives, or posterior validity. The weakest part of present evidence is the absence of a multi-step, fully tuned, float32/TF32 model campaign; this review makes no claims requiring one.

**Actionable verdict:** revise the plan and withdraw the unsupported graceful-failure readiness claim. The next work is a localized correctness repair followed by the staged validation above, not a larger run of the current eight-case checklist.
