# Claude Fable Review Packet: LGSSM Value And `q_scale` Bias

Date: 2026-07-20

Review role: read-only mathematical and scientific code review. Claude Fable
must not edit files, run experiments, launch agents, or make repository changes.
The packet is intended to be sufficient for a deep review while Codex runs the
next time-local diagnostic in a separate execution lane.

## Review Request

Please analyze the persistent disagreement between the canonical LEDH Contract
E--Chol finite-particle value/score program and the exact differentiated Kalman
LGSSM target. The central question is:

> Which hypotheses can mathematically produce the observed value and `q_scale`
> bias, which hypotheses are already ruled out by proof or evidence, and which
> numerical controls were insufficiently tuned?

Please independently trace the finite program and its derivative, audit the
target comparison, and assess whether the proposed next experiment is capable
of distinguishing common finite-particle likelihood bias from reset-specific,
transport-specific, precision-specific, or derivative-composition errors.

End the review with:

1. a ranked hypothesis table (`supported`, `plausible`, `unlikely`, or `ruled out`);
2. any mathematical or implementation error not already identified;
3. controls that must be tuned or held fixed in a new scope;
4. the smallest discriminating experiment and its expected outcomes; and
5. exactly one final line:

```text
VERDICT: AGREE
```

or

```text
VERDICT: REVISE
```

`AGREE` means the packet's target distinction, evidence classification, and
next diagnostic are technically adequate. It does not mean the LEDH route has
passed the Kalman value/score gate.

## Scope And Nonclaims

The benchmark is a synthetic three-dimensional stationary LGSSM:

```text
phi       = (0.72, 0.55, 0.35)
q_scale   = 0.35
r_scale   = 0.45
H         = [[1.0, 0.25, -0.15],
             [0.2,  1.1,  0.3 ],
             [-0.1, 0.35, 0.9 ]]
dataset seed = 81100
horizon      = 50
```

The DGP initializes the latent state from

```text
N(0, diag(q_scale^2 / (1 - phi_j^2)))
```

and generates observations from the initial state followed by 49 transitions.
The exact comparator is the TensorFlow Cholesky Kalman likelihood and its
automatic derivative at the same physical parameter point, using the
float32-rounded observation tensor consumed by the production-shaped LEDH
route and differentiated in float64 after that cast.

The candidate is not the exact Kalman filter. It is the finite program

```text
Contract E-Chol + finite annealed Sinkhorn/terminal balance
+ LEDH Gaussian flow + finite importance normalization
+ equal-weight reset + fixed prepared residual design and ridge.
```

Do not infer from this packet:

- posterior or HMC readiness;
- equivalence from failure to reject a bias interval;
- a `1/N` convergence rate;
- nonlinear-model validity or transfer;
- method superiority;
- a universal `sinkhorn_steps` or `balance_steps` setting; or
- rejection of Contract E as a research direction.

## Current Empirical Result

The claim artifacts are engineering-valid but not Kalman-certified.

| Scope | Selected controls | Mean value bias | Mean `q_scale` bias | Simultaneous 95% CI for value | Simultaneous 95% CI for `q_scale` | Status |
| --- | --- | ---: | ---: | --- | --- | --- |
| `T=50,N=1024,K=1024` | `(20,8)` | `+0.0847%` | `-31.65%` | `[+0.0276%,+0.1419%]` | `[-43.59%,-19.71%]` | score fail |
| `T=50,N=2000,K=2000` | `(20,5)` | `+0.1003%` | `-45.75%` | value screen fail | `[-58.79%,-32.70%]` | screen fail |
| `T=50,N=5000,K=2500` | `(20,5)` | `+0.1482%` | `-9.91%` | `[+0.1116%,+0.1848%]` | `[-17.72%,-2.11%]` | screen fail |
| `T=50,N=10000,K=2500` | `(20,8)` | `+0.1735%` | `-15.90%` | `[+0.1502%,+0.1968%]` | `[-22.00%,-9.79%]` | screen fail |

The newer primary statistical diagnostic asks whether each mean relative-bias
interval contains zero. Zero is rejected for value and `q_scale` at both
`N=5000` and `N=10000`. This establishes persistent mean bias under the tested
finite programs, not the mechanism.

Artifacts:

- `docs/benchmarks/artifacts/lgssm_kalman_zero_bias_ci_20260720/aggregate.json`
- `docs/benchmarks/artifacts/lgssm_particle_bias_ladder_20260720/aggregate_final.json`
- `docs/benchmarks/artifacts/lgssm_n10000_tuned_kalman_20260720/attempt01/aggregate.json`
- `docs/plans/bayesfilter-lgssm-selected-controls-kalman-certification-result-2026-07-19.md`
- `docs/plans/bayesfilter-lgssm-particle2000-particle5000-kalman-bias-ladder-result-2026-07-20.md`

The particle-count result is nonmonotone: `N=2000` is worse than `N=1024`,
`N=5000` is descriptively better, and `N=10000` is descriptively worse than
`N=5000`. The scopes use independent seeds and different realized extrema in
the geometry-dependent annealing schedule, so this is not a clean paired
particle-count convergence study.

## Executed Finite Program

The implementation path is:

1. Initial particles:

   ```text
   particles = initial_noise * q / sqrt(1 - phi^2)
   ```

   Code: `bayesfilter/highdim/ledh_contract_e_canonical_lgssm_tf.py:697-724`.

2. At each time `t`, propagate first:

   ```text
   prior_mean = particles @ diag(phi)
   pre_flow   = prior_mean + q * transition_noise[t]
   ```

   Then apply the exact linear-Gaussian flow using transition covariance
   `q^2 I`, observation covariance `r^2 I`, and observation matrix `H`.

   Code: `bayesfilter/highdim/ledh_contract_e_canonical_lgssm_tf.py:778-811`.

3. The finite likelihood increment is:

   ```text
   logits_i = log_weight_i
            + log p(flow_i - prior_mean_i | q^2 I)
            + log p(H flow_i - y_t | r^2 I)
            - log q_proposal(flow_i | prior_mean_i)
            + log_abs_det(flow)

   increment_t = logsumexp(logits_i)
   normalized_log_weights_i = logits_i - increment_t
   ```

   Code: `bayesfilter/highdim/ledh_contract_e_canonical_lgssm_tf.py:792-816`.

4. The particle geometry is centered and scaled. `epsilon0` is derived from
   the maximum and minimum scaled particle coordinates, then the finite
   annealed schedule is clamped to terminal `epsilon`:

   ```text
   epsilon = 0.5
   scaling = 0.9
   ```

   Code: `bayesfilter/highdim/ledh_contract_e_canonical_lgssm_tf.py:564-585`
   and `:673-694`.

5. The streaming transport creates a barycentric equal-weight cloud. Contract
   E then uses weighted source moments, transported-cloud moments, fixed
   residual design, a covariance gap, and an affine Cholesky map. Active reset
   replaces the cloud and sets all log weights to `-log(N)`; inactive reset
   carries the normalized weights forward.

   Code: `bayesfilter/highdim/ledh_contract_e_canonical_lgssm_tf.py:817-856`;
   `bayesfilter/highdim/ledh_contract_e_reset_tf.py:44-102`.

6. The fused XLA recursion repeats the same finite step through a
   `tf.while_loop`, accumulating the finite value and score increments:

   Code: `bayesfilter/highdim/ledh_contract_e_canonical_lgssm_tf.py:1258-1705`.

## Derivative Path Already Checked

The manual JVP initializes tangents for:

- stationary initial standard deviations;
- transition matrix and transition covariance;
- observation covariance;
- flow particles and flow log determinant;
- Gaussian transition, proposal, and observation densities;
- log-normalization and normalized weights;
- geometry and `epsilon0`;
- streaming transport;
- Contract E moments, residual injection, affine map, and reset; and
- subsequent particle tangents after each active reset.

Code: `bayesfilter/highdim/ledh_contract_e_canonical_lgssm_tf.py:1080-1255`
and fused composition at `:1275-1456`.

The total score is accumulated only from the likelihood-increment tangent:

```text
per_batch_score += normalization.increment_tangent
```

The reset tangent affects later particles and later increments; it does not
retroactively alter the already accumulated current increment.

This is supported by same-scalar finite-difference evidence at the lower rung:

- `T=2,N=32` relative finite-difference error was `3.74e-08`;
- primitive Contract E JVP/VJP checks passed;
- finite differences used the same prepared scalar, ridge, controls, and
  endpoints.

Artifact/result context:
`docs/plans/bayesfilter-contract-e-canonical-gradient-migration-phase8-lower-rung-continuation-result-2026-07-14.md`.

This evidence proves, within its tested scope, that the score is the derivative
of the executed finite scalar. It does **not** prove that the finite scalar or
its derivative equals the Kalman likelihood or Kalman score.

## Mathematical Distinction To Audit

There are three different objects:

1. exact Kalman likelihood and score;
2. finite weighted-particle / LEDH importance-normalizer likelihood and score;
3. finite barycentric-transport plus Contract E reset likelihood and score.

The current code is intended to compute object 3. Comparing object 3 with object
1 is scientifically useful, but a discrepancy can be caused by finite-particle
or reset approximation even when the derivative of object 3 is exactly correct.

The value is a sum of logarithms of finite normalizer estimates. Unbiasedness of
an unlogged normalizing-constant estimate would not imply unbiasedness of its
logarithm. The `q_scale` score is the parameter derivative of that finite
log-bias function. In this benchmark `q_scale` also affects both initial and
transition covariance scales, and the exact `T=50` score increments have
substantial positive/negative cancellation.

## Contract E Mathematics

For source particles `X`, normalized weights `w`, and equal-target transport
coupling `Pi`, the barycentric output is

```text
tilde_x_j = (1 / u_j) sum_i Pi_ij x_i
```

and for equal target weights `u_j=1/N`:

```text
tilde_x_j = N sum_i Pi_ij x_i.
```

This output is a conditional mean of the source cloud under the transport
coupling. It preserves the weighted mean but contracts covariance by the
conditional covariance lost inside each output column.

Contract E adds residual spread and a Cholesky affine map. With prepared ridge
`lambda > 0`:

```text
B_lambda = chol(G_plus + lambda I)
tilde_Y  = Y_plus + Xi B_lambda^T
A_lambda = chol(Sigma_w + lambda I)
           chol(tilde_Sigma + lambda I)^(-1)
Y_star   = mean_w + centered(tilde_Y) A_lambda^T
```

The exact identity is:

```text
A_lambda (tilde_Sigma + lambda I) A_lambda^T
    = Sigma_w + lambda I
```

but the raw covariance residual is:

```text
Cov(Y_star) - Sigma_w = lambda (I - A_lambda A_lambda^T).
```

Therefore calling the raw covariance exactly restored when `lambda > 0` is
wrong. The reset is a declared finite target, not an exact Kalman filtering
step.

## LaTeX Source Anchors For Fable

These are the most relevant project derivations. Please inspect them directly
and check whether the implementation and this packet interpret them correctly.

### `docs/chapters/ch32c_entropic_ot_sinkhorn.tex`

- `:511-525`, equations `bf-eot-barycentric` and
  `bf-eot-barycentric-equal`: the equal-weight cloud is a barycentric
  conditional-mean projection.
- `:531-584`, Proposition
  `bf-eot-barycentric-covariance-contraction`: mean is preserved, covariance is
  reduced by expected conditional covariance.
- `:586-624`: explains why the next LGSSM predictive likelihood can change even
  when transport marginal residuals are small, and why a same-scalar gradient
  can be correct without being a Kalman gradient.
- `:690-725`, equations `bf-eot-weighted-covariance`,
  `bf-eot-transform-covariance`, and the second-order transform contract.
- `:731-781`, Proposition `bf-eot-barycentric-not-second-order`: formal proof
  that generic diffuse barycentric Sinkhorn output is not second-order exact.
- `:783-790`: explicitly states that tighter Sinkhorn convergence does not
  remove barycentric covariance loss.
- `:792-918`: Contracts A through D. In particular, Contract A accepts the
  barycentric finite scalar as its own target, while Contract C/D change the
  reset object and require new value and gradient gates.
- `:920-1268`: Contract E's residual-spread and affine-restoration construction,
  including its finite-target and rank/conditioning caveats.
- `:1299-1325`, `:1341-1420`: Contract E--Chol ridge gauge, prepared-ridge
  semantics, ridged restoration proposition, and raw covariance residual.
- `:1422-1456`: canonical eligibility, row quotient, fixed prepared controls,
  and Contract F/no-reset diagnostic meaning.

### `docs/chapters/ch32c2_ledh_pfpf_ot_custom_gradient.tex`

- `:49-69`, definition `bf-ledh-ot-same-scalar`: a custom gradient must be the
  derivative of the exact finite scalar returned by the primal.
- `:74-112`, equation `bf-ledh-ot-forward-composition`: transition-first
  sequence; the likelihood increment is computed before active reset.
- `:117-192`, equations `bf-ledh-contract-e-row-quotient`,
  `bf-ledh-contract-e-affine-reset`,
  `bf-ledh-contract-e-total-source-pullback`, and
  `bf-ledh-contract-e-total-logweight-pullback`: Contract E depends directly on
  source moments and weights as well as transported-cloud derivatives.
- `:532-563`, Proposition `bf-ledh-ot-finite-sinkhorn-vjp`: finite reverse scan
  gives the VJP of the finite Sinkhorn program under its fixed branches.
- `:589-646`, Proposition `bf-ledh-same-cloud-cache`: a geometry cache can
  preserve the finite JVP in exact arithmetic, but does not change the finite
  target.

The key review question is whether any code path omits a term required by the
total-derivative equations, especially the direct moment/weight terms in
`bf-ledh-contract-e-total-source-pullback` and the normalized-log-weight
composition in `bf-ledh-contract-e-total-logweight-pullback`.

## Previous Attempts And What They Established

### 1. Lower-rung derivative and reset work

The initial Contract E migration and lower-rung tests repaired the canonical
route, fixed the prepared ridge, and obtained same-scalar finite-difference
agreement at `T=2,N=32` (`3.74e-08` relative error). Kalman disagreement
remained. This rejected a gross local derivative wiring explanation, but not a
finite-program approximation error.

The selected lower-rung controls were:

```text
ridge   = 7.301568984985351e-09
epsilon = 0.5
scaling = 0.9
```

These values were later carried as warm starts, not independently tuned for
`T=50,N=5000` or `T=50,N=10000` value/score accuracy.

### 2. Contract E versus no-reset short-horizon attempts

The paired `T=2` particle ladder at `N=128,256,512,1024` ran both active
Contract E and no-reset weighted arms with the same observations and prepared
streams. Both arms were engineering-valid and their Kalman errors were nearly
the same. Paired reset-minus-no-reset intervals were mostly inconclusive.

This failed to support the simple hypothesis that Contract E alone causes the
Kalman discrepancy. It did not prove no reset effect at `T=50`, where reset
errors may accumulate recursively.

Artifacts/results:

- `docs/plans/bayesfilter-contract-e-canonical-gradient-migration-phase8-paired-reset-audit16-result-2026-07-14.md`
- `docs/plans/bayesfilter-canonical-lgssm-balancing-kalman-repair-phase2-result-2026-07-17.md`

### 3. TF32 balancing and mass-accounting repair

At `T=2,N=1024`, early TF32 balance counts failed a transport column-marginal
gate. A direct squared-distance diagnostic did not remove the failure, which
falsified the distance-GEMM hypothesis. The actual issue was a TF32-sensitive
all-ones payload GEMM in mass accounting; explicit mass reductions repaired the
engineering marginal gate.

At `T=10`, a 16-seed claim still failed the row-error gate, so the route did not
proceed to the old `T=50` continuation under that plan. This was a transport
engineering/tuning failure, not proof of the Kalman score bias mechanism.

Result:
`docs/plans/bayesfilter-canonical-lgssm-tf32-balance-horizon-continuation-result-2026-07-18.md`.

### 4. `T=50,N=1024` selected-control certification

Independent marginal tuning selected `(sinkhorn_steps=20,balance_steps=8)`.
All engineering gates passed, but the `q_scale` HMC-coordinate score had mean
relative error `-31.65%`, simultaneous CI `[-43.59%,-19.71%]`.

A same-scope no-reset diagnostic was worse on several coordinates. This rules
out the simplistic statement “remove Contract E and the finite filter becomes
Kalman-correct.” It does not identify whether Contract E partially repairs a
shared error or whether reset contributes at long horizon.

Result:
`docs/plans/bayesfilter-lgssm-selected-controls-kalman-certification-result-2026-07-19.md`.

### 5. `N=2000` and `N=5000` ladder

Each particle count received its own blind marginal calibration and untouched
16-seed claim. The first `N=5000` attempt selected `(20,3)` but failed the row
gate on three claim seeds. A localized multi-block harness repair was then
made, and a fresh repair scope selected `(20,5)` and passed all engineering
gates.

The scientific result remained negative: `N=2000` was worse than `N=1024`,
while `N=5000` had smaller `q_scale` bias but a value interval wholly above the
old value region. No `1/N` rate or cross-`N` ranking was established.

Result:
`docs/plans/bayesfilter-lgssm-particle2000-particle5000-kalman-bias-ladder-result-2026-07-20.md`.

### 6. Single-seed `N=10000` attempt

`T=50,N=10000,K=2500`, singleton execution was engineering-feasible. The
cross-scope warm start `(20,5)` produced one-seed `q_scale` relative error
`+22.63%`, versus `+5.02%` at `N=5000` for that paired seed. This was only one
seed and was not used as bias evidence.

Result:
`docs/plans/bayesfilter-lgssm-n10000-single-seed-kalman-diagnostic-result-2026-07-20.md`.

### 7. Proper `N=10000` scope tuning and claim

The exact `N=10000` scope independently tuned balance/Sinkhorn counts. `(20,5)`
failed validation row error; `(20,8)` was the first blind direct-gate pass.
The untouched 16-seed claim passed engineering gates but had mean value bias
`+0.1735%` and mean `q_scale` bias `-15.90%`; simultaneous intervals rejected
zero for both outputs.

Result:
`docs/plans/bayesfilter-lgssm-n10000-tuned-kalman-certification-plan-2026-07-20.md`
and its aggregate artifact under
`docs/benchmarks/artifacts/lgssm_n10000_tuned_kalman_20260720/attempt01/`.

### 8. `q*` reparameterization diagnostic

The proposed transformation was:

```text
q*^2 = q_scale^2 / 3 *
       [1/(1-phi1^2) + 1/(1-phi2^2) + (1-phi3^2)]
```

At the DGP, `q*=0.4232735346`. With `q=q*/sqrt(A(phi))`:

```text
q* (dL/dq*) = q (dL/dq)
```

for the physical q direction, and value is invariant at the same physical
point. The transformed `q*` log-score had exactly the same relative bias as
the existing log-`q_scale` score. This rules out reparameterization as an
independent repair, though it may change optimizer geometry or phi-coordinate
correlations.

Result:
`docs/plans/bayesfilter-lgssm-qstar-reparameterization-diagnostic-2026-07-20.md`.

## Tuning Audit

The exact-scope tuner searched only:

```text
sinkhorn_steps
balance_steps
```

and stopped at the first calibration/validation pair passing direct marginal
gates. The following were fixed or inherited:

- terminal `epsilon=0.5`;
- annealing `scaling=0.9`;
- geometry-derived `epsilon0` schedule;
- reset cadence, active at every period;
- residual-design construction and centering;
- prepared ridge magnitude and policy;
- float32/TF32 backend;
- the selection objective, which did not use value/score accuracy or a
  Kalman-blind score-stability diagnostic.

The repository registry makes the split explicit:
`bayesfilter/highdim/ledh_tuning_registry.py:25-38` declares only
`sinkhorn_steps` and `balance_steps` tunable for canonical LGSSM and treats
`epsilon`, `scaling`, `prepared_ridge`, and particle count as fixed. Particle
count, dtype/backend, and chunk policy should remain scope identity fields, but
the other numerical controls need either scope-specific tuning evidence or a
strong reviewed justification before value/score claims.

Important coupling: `epsilon0` is computed from finite sample extrema. Changing
`N` changes the realized geometry and annealing path, so `N=5000` versus
`N=10000` is not simply the same finite algorithm with more particles.

## Ranked Hypotheses To Review

| Rank | Hypothesis | Current evidence |
| ---: | --- | --- |
| 1 | Common finite-particle proposal/importance-normalizer error | Leading class. Short-horizon Contract E/no-reset errors were nearly shared. |
| 2 | Log-normalizer bias and long-horizon score cancellation | Mathematically intrinsic candidate; `q_scale` terminal score is cancellation-sensitive. |
| 3 | Contract E reset as long-horizon amplifier | Plausible, not reset-only at `T=2`, and not yet isolated at `T=50`. |
| 4 | Untuned `epsilon`, `scaling`, and geometry path | Plausible and directly under-tuned. |
| 5 | Ridge, residual design, or covariance conditioning | Plausible; direct ridge magnitude alone looks too small for the full bias. |
| 6 | FP32/TF32 recursive drift | Plausible secondary cause; replay does not establish accuracy. |
| 7 | Missing total-derivative term | Lower probability because local and same-scalar evidence passed; full time-local closure remains absent. |
| 8 | Transition/observation off-by-one | Not confirmed here because the DGP is stationary and time-homogeneous. |
| 9 | RNG, seed batching, or prepared-input mismatch | Largely ruled out by stateless preparation, hashes, singleton claims, and replay. |
| 10 | `q*` versus `q_scale`, `q^2`, or `log q` parameterization | Ruled out as an independent repair by exact chain rule. |

## Proposed Next Experiment

Use identical observations and prepared random streams for:

1. active Contract E;
2. no-reset weighted recursion;
3. exact Kalman increments.

At each time step, emit value and `q_scale` score contributions separated into:

- stationary initial-covariance term;
- transition/proposal term;
- observation density and likelihood-normalization term;
- carried previous-weight term;
- transport term; and
- Contract E moment/weight/reset term.

Each partial score must have a same-scalar finite-difference check. The
diagnostic should answer:

- Does the discrepancy appear before the reset?
- Does Contract E amplify or reduce it over time?
- Is the error primarily in the finite log-normalizer or in the score tangent?
- Does FP32/TF32 versus FP64/no-TF32 change the time-local pattern?
- Are the errors small local terms whose terminal sum is cancellation-sensitive?

Only after this decomposition should a small paired sensitivity test vary
terminal `epsilon`, `scaling`, ridge, reset cadence, or precision. Tuning those
controls directly against Kalman on claim data would be invalid; calibration
must remain disjoint and Kalman-blind.

## Review Checklist For Fable

Please check all of the following explicitly:

- Is the finite likelihood increment mathematically the correct scalar for the
  declared LEDH proposal and flow Jacobian?
- Is the initial stationary covariance derivative with respect to `phi` and
  `q_scale` correct?
- Are transition covariance, observation covariance, proposal density, and
  flow log-determinant derivatives all included exactly once?
- Does normalized-log-weight differentiation include the carried-weight tangent
  and avoid double normalization?
- Does Contract E's total derivative include both direct source-moment and
  transported-cloud terms, as required by the LaTeX total-pullback equations?
- Does the reset correctly stop current-increment retroactive influence while
  propagating reset tangents into subsequent increments?
- Is the transition-first ordering harmless only because this DGP is stationary?
- Does the finite-particle/log-normalizer argument explain a small value bias but
  much larger relative `q_scale` score bias?
- Does `epsilon0`'s sample-extrema dependence invalidate naive cross-`N`
  convergence interpretation?
- Which controls should be fixed as execution identity and which should be
  tuned per scope?
- Is the proposed time-local decomposition sufficient, or is an additional
  exact-recursion or independent Kalman bridge required?
- Are any claims in this packet too strong relative to the artifacts?

## Artifact And Environment Context

The accepted claims ran under:

```text
conda environment: tf-gpu
TensorFlow:        2.19.1
GPU:               NVIDIA GeForce RTX 4080 SUPER
execution:         float32 + TF32 + GPU + XLA
logical GPU cap:   8192 MiB
chunk policy:      dpf_transport_exact_divisor_cap3000_v1
N=5000:            K=2500, 2 x 2 blocks
N=10000:           K=2500, 4 x 4 blocks
claim microbatch:  singleton seeds for correctness
```

The worktree contains unrelated shared uncommitted changes. Review the source
paths and recorded artifacts as evidence, but do not assume the current dirty
worktree is identical to every historical artifact. Relevant current code:

- `bayesfilter/highdim/ledh_contract_e_canonical_lgssm_tf.py`
- `bayesfilter/highdim/ledh_contract_e_reset_tf.py`
- `bayesfilter/highdim/ledh_contract_e_streaming_tf.py`
- `bayesfilter/highdim/ledh_contract_e_lgssm_preparation_tf.py`
- `docs/benchmarks/run_canonical_lgssm_fused_ot_loop_repair.py`
- `docs/benchmarks/run_ledh_offline_ot_tuning_campaign.py`
- `bayesfilter/highdim/ledh_tuning_registry.py`

## Codex Handoff

This packet was prepared from the prior audit:
`docs/plans/bayesfilter-lgssm-value-qscale-bias-hypothesis-audit-2026-07-20.md`.
The next Codex execution lane should implement the time-local decomposition
under a fresh versioned artifact directory. Claude Fable's role is independent
analysis of the mathematics, target identity, derivative completeness, and
diagnostic design; it is not execution authority and does not need to wait for
the next run to begin review.

## Review Execution Status

Codex requested the bounded read-only Fable review with the command shape:

```text
claude -p "READ-ONLY BOUNDED PACKET REVIEW ... docs/plans/
bayesfilter-lgssm-value-qscale-bias-claude-fable-review-packet-2026-07-20.md ..."
```

The request did not produce a review or verdict. Claude reported:

- the workspace was not trusted, so local permission allow entries were
  ignored and the trust dialog would need to be accepted interactively; and
- the Claude credit balance was too low.

A minimal health probe (`claude -p "Return exactly CLAUDE_PROBE_OK."`) produced
the same trust and credit errors. This is an external reviewer-availability
limitation, not a scientific verdict and not evidence that the packet passed or
failed. A future agent should rerun the same bounded packet review after the
workspace trust and credit prerequisites are resolved, without changing the
packet contents solely to obtain an `AGREE`.

## MathDevMCP Probe

A local MathDevMCP smoke test succeeded for `doctor` but failed for repo-root
LaTeX search:

```text
PYTHONPATH=/home/chakwong/MathDevMCP/src python -m mathdevmcp.cli doctor
```
returned a valid capability report with `sympy==1.14.0` and `mcp==1.27.0`
available in the active Python.

```text
PYTHONPATH=/home/chakwong/MathDevMCP/src python -m mathdevmcp.cli search-latex "bf-ledh-contract-e-total-source-pullback" --root /home/chakwong/BayesFilter --limit 3
```
failed in the equation locator with:

```text
ValueError: invalid brace depth in LaTeX display environment
```

The most likely trigger is [docs/chapters/ch37_highdim_fixed_branch_likelihoods_and_same_scalar_gradients.tex](docs/chapters/ch37_highdim_fixed_branch_likelihoods_and_same_scalar_gradients.tex#L392-L399), where an `array` block uses `\\[1mm]` row breaks inside a display environment. That is a tooling issue, not a scientific conclusion.
