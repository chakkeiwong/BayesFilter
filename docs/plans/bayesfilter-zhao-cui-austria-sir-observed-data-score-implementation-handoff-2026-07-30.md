# Zhao-Cui Austria SIR Observed-Data Score Implementation Handoff

Status: `SUPERSEDED_HISTORICAL_APF_HANDOFF_NOT_ACTIVE`

> Superseded on 2026-07-30 by
> `docs/plans/bayesfilter-zhao-cui-austria-sir-fixed-variant-parameter-extension-master-plan-2026-07-30.md`.
> Do not use this APF handoff to govern the current extension of the exact P88
> training-base fixed-variant artifact. P76/P77 UKF was a separate experiment,
> not proved baseline behavior.

Date: 2026-07-30
Recipient: Zhao-Cui implementation agent
Task: implement and validate the missing full observed-data Zhao-Cui SIR value/score cell
Historical snapshot status: `IMPLEMENTATION_REQUIRED_NOT_LOST`

## Executive Verdict

The Zhao-Cui SIR score work was not lost, but it was not completed for the
then-targeted leaderboard row. At the time, the repository contained:

- the parameterized Austria SIR model (`J=9`, state dimension `d=18`,
  observation dimension `9`, parameter dimension `3`);
- analytical transition-density and observation-density parameter scores;
- a horizon-0 retained-grid score smoke;
- source-route retained samples, previous-marginal value mechanics, and
  fixed-TTSIRT/paired-core conditional machinery;
- a generic frozen-proposal APF value/recursive-score evaluator; and
- a small blockwise fitted-TTSIRT APF engineering rung.

What is still missing is an Austria-specific fixed-variant route that computes
the full observed-data/filtering scalar and the analytical recursive score of
that same finite scalar at `T=20`. The existing local complete-data score is not
that quantity. The historical all-axes retained-grid evaluator is not an
acceptable repair because its first transition at `d=18` is computationally
infeasible and repository policy demotes that route.

This handoff is an implementation request, not permission to relabel an old
artifact. The recipient must implement the missing route, audit its mathematics
and source relation, run the focused checks, and either produce an admissible
same-target score artifact or return a precise blocker with the unimplemented
derivative/measure/complexity item.

## Historical Target Contract

The implementation must target the same row used by the current GenUT/SGQF
comparison. Do not substitute a reduced SIR fixture, a local complete-data
sidecar, or an initial-observation-first data set.

| Field | Required value |
| --- | --- |
| Current comparison row | `austria_sir_T20` |
| Historical Zhao-Cui row id | `zhao_cui_spatial_sir_austria_j9_T20` |
| Model | `bayesfilter.highdim.models.parameterized_zhao_cui_sir_austria_model()` |
| Compartments | `J=9` |
| State ordering | `(S1,I1,S2,I2,...,S9,I9)` |
| State dimension | `18` |
| Observation | infectious components, dimension `9`, observation covariance `100 I_9` at truth |
| Parameters | `(log_kappa_scale, log_nu_scale, log_observation_noise_scale)` |
| Truth theta | `(0, 0, 0)` |
| Event order | draw `x0`, then for `t=1,...,20`: transition `x_t` and observe `y_t` |
| Horizon | `T=20` |
| Claim particle count | `N=1008` initially; a larger `N>1000` run may be added after the focused rung |
| Runtime target | TensorFlow, FP32, TF32 enabled, XLA enabled, GPU memory growth |
| Score requirement | manual analytical recursive score; no runtime autodiff or finite differences |
| Source observation SHA-256 | `cd794ad6e90a74f7cf6dc06b33550bff4bef6fbf66bb0917846d0691b5910f07` |
| Runtime FP32 observation SHA-256 | `40c793fb374e84fcd347c66b189352b5997740cc753ea0be03441ecf32828009` |

The sealed observation hash and target scope used by this historical campaign
are recorded in:

`docs/benchmarks/artifacts/moment_retuned_genut_whole_leaderboard_20260723/attempt05_final/result.json`

The current GenUT/SGQF comparison and its nonclaims are recorded in:

`docs/plans/bayesfilter-moment-retuned-genut-whole-leaderboard-result-2026-07-23.md`

The final Zhao-Cui cell must use the same observation tensor, event order,
parameterization, and target hash. A different hash is a different experiment.

## Research Intent And Evidence Contract

| Field | Contract |
| --- | --- |
| Main question | Can a fixed-variant Zhao-Cui-derived proposal produce a finite, memory-bounded Austria SIR observed-data likelihood scalar and its analytical score for the same scalar? |
| Candidate | Offline fixed TTSIRT/squared-TT conditional proposal, frozen for HMC, used in a source-order fixed-randomness APF with complete importance correction. |
| Baseline ladder | Existing GenUT result; existing SGQF result; a simple fixed proposal/APF smoke; the Zhao-Cui candidate. |
| Primary promotion criterion | Finite value and all three score coordinates for the exact declared finite program, with branch identity, target hash, and analytical-score provenance bound together. |
| Hard vetoes | Wrong event order/hash; reduced or local-complete-data target; retained-grid fallback; parameter-dependent runtime proposal/refit; omitted proposal/normalizer/previous-marginal term; nonfinite output; invalid density/measure; runtime FD/autodiff; unbounded memory; failed XLA/GPU claim artifact. |
| Explanatory diagnostics | ESS, weight entropy, proposal fit, conditional-density residual, round-trip error, score/FD differences at representative points, per-time score increments, compile/warm time, allocator peak. These do not prove exactness or rank methods. |
| What will not be concluded | No exact nonlinear likelihood theorem, posterior correctness, HMC convergence, source-faithful assembled-route claim, superiority over GenUT/SGQF, default promotion, NAWM readiness, or broad high-dimensional readiness. |
| Terminal artifacts | implementation result note, JSON run manifest/result, branch/program identity, focused-test output, and a reset memo if route status changes. |

The score/FD comparison is a diagnostic only. It may nominate or reject a
candidate implementation, but the runtime score must remain the manual
recursive derivative path.

## Skeptical Plan Audit

This handoff was audited against the current code, July 23 same-target result,
P81 retained-grid tests, P89/P90 source-route derivative records, P91 local
score result, Zhao-Cui paper, and pinned author source. The initial informal
claim that "all machinery exists" needed qualification: enough substrate exists
to make implementation realistic, but the Austria proposal branch, source-order
composition, and full observed-data score are not already implemented.

The audit identified and corrected the following misleading execution paths:

| Risk | Audit finding | Handoff correction |
| --- | --- | --- |
| Wrong baseline | Reduced `J=1` SIR and local complete-data scores were previously called Zhao-Cui SIR scores | Both are explicit hard vetoes for this row |
| Wrong target timing | The generic APF core observes at its first state, while the sealed target transitions before `y1` | Phase 1 requires an explicit source-order adapter and identity field |
| Proxy promoted to result | Horizon-0 and synthetic blockwise rungs test mechanics only | Claim requires untouched Austria `T=20,N=1008` evidence |
| Hidden derivative assumption | Frozen proposal terms are zero only when every proposal/ancestor object is actually theta independent | Branch identity and derivative ownership are promotion gates |
| Stale defaults | Degree/rank/defensive mass from synthetic or older source runs are not tuned for the sealed Austria scope | Phase 4 requires disjoint, scope-specific tuning |
| Non-answering artifact | A finite local score or source-route value alone would leave the leaderboard cell unresolved | Terminal artifact must contain same-program observed-data value and three-coordinate score |
| Environment mismatch | CPU/FP64 source checks do not establish the required GPU FP32/TF32/XLA path | Phase 5 requires trusted GPU execution and allocator evidence |

The revised plan passes this skeptical audit as an implementation handoff: it
uses the correct target and baselines, separates mechanics diagnostics from
promotion evidence, includes stop/repair conditions, and requires artifacts
that directly answer the missing-cell question. This audit does not establish
that the candidate will fit in memory, attain adequate ESS, or pass its score
identity checks.

## Default And Assumption Audit

No numerical setting below is an Austria default until target-specific tuning
and untouched validation support it.

| Choice | Provenance | Current status | Failure mode | Early diagnostic |
| --- | --- | --- | --- | --- |
| Frozen-proposal APF architecture | July 22 generic/rung-1 engineering work | implementation hypothesis | poor high-dimensional proposal causes weight collapse | `T=1/2` ESS and log-weight spread |
| Squared-TT/TTSIRT proposal | Zhao-Cui Eq. (13), Proposition 2, Algorithm 3 and author source | source-grounded operation, not admitted assembly | rank/degree insufficient or conditional density inaccurate | held-out density and paired-core checks |
| `N=1008` | Current same-target leaderboard scope and user policy `N>1000` | comparison baseline | insufficient score precision | multi-branch/seed SD and increment variance |
| FP32, TF32, XLA | Repository execution policy and current leaderboard | required claim backend | loss of score/normalizer precision | CPU FP64 tiny tie-out followed by GPU tolerance audit |
| Degree, rank, defensive mass | Older source experiments and synthetic engineering rungs | warm starts only | underfit, tail failure, or memory growth | target-specific calibration/validation ladder |
| Coordinate/block ordering | Existing generic prefix compiler and SIR state structure | hypothesis | destroys conditional structure or changes source semantics | direct conditional-density and source-order tests |
| Uniform or auxiliary ancestors | Generic APF implementation | comparator/hypothesis | incorrect normalization or low ESS | direct finite-scalar tie-out and categorical normalization |
| Fixed theta reference | Needed for offline HMC proposal freezing | hypothesis | poor performance away from reference theta | representative-point ESS/value/score stability diagnostics |
| Observation-noise and clipping measure | Local Austria model callbacks | target contract requiring audit | density differs from actual clipped transition law | support/measure audit before claim run |

Settings transferred from another model or scope may seed the first candidate
only. They are not promotion evidence and must not be caller-stamped as tuned.

## What Exists And Must Be Reused

### Austria SIR model and local analytical terms

`bayesfilter/highdim/models.py` contains
`ParameterizedZhaoCuiSIRSSM` and
`parameterized_zhao_cui_sir_austria_model()`.

The existing methods include:

- `transition_mean_parameter_jacobian`;
- `transition_log_density_parameter_score`;
- `observation_log_density_parameter_score`;
- `initial_log_density_parameter_score`; and
- the model transition/observation densities.

These methods provide the local density scores that enter a filtering-score
recursion. They do not, by themselves, integrate out the latent state path.

### Generic frozen-branch APF evaluator

`bayesfilter/highdim/zhao_cui_frozen_proposal_apf_tf.py` provides:

- `PreparedFrozenProposalBranch`;
- `compile_fixed_ttsirt_proposal_branch`;
- `prepare_frozen_proposal_apf_program`; and
- `FrozenProposalAPFProgram._evaluate_core`.

The generic evaluator already implements the important normalized-weight score
identity for a frozen branch:

```text
local marks -> log-sum-exp normalizer -> increment score
           -> centered derivative marks -> next time step
```

It also binds branch/program hashes and has Gaussian parity tests. Reuse the
working score recursion and identity pattern, but do not assume its current
observation timing is suitable for Austria SIR.

### Existing source-route mechanics

`bayesfilter/highdim/source_route.py` contains source-route value helpers for:

- retained sample generation;
- previous retained marginal evaluation;
- sequential negative log physical density; and
- fixed-HMC replay scaffolding.

The paired-core/conditional transport substrate is in the high-dimensional
transport modules and is exercised by:

`tests/highdim/test_zhao_cui_frozen_ttsirt_apf_compiler.py`

### Engineering APF rung

The July 22 blockwise TTSIRT APF rung demonstrated finite value/score mechanics
at a smaller synthetic scope. Its terminal report is:

`docs/plans/bayesfilter-zhao-cui-blockwise-ttsirt-apf-rung1-result-2026-07-22.md`

That result is an engineering candidate, not Austria SIR evidence. Reuse its
finite-branch, proposal-density, score, and memory patterns; do not copy its
independent Gaussian block assumptions into the coupled SIR target without a
target-specific fit and validation.

## What Is Missing

### 1. Austria-specific fixed proposal branch

Construct a fixed proposal branch for the actual observations and `T=20`.
The branch must contain, at minimum:

- fixed initial proposal samples or a fixed initial proposal map;
- fixed ancestor indices or a fixed auxiliary-ancestor law;
- fixed state samples generated by a proposal map;
- pointwise initial and transition proposal log densities;
- observation tensor and target hash;
- all coordinate maps, TT cores, ranks, defensive mass, basis domains,
  quadrature/CDF controls, seeds, and schedules; and
- a branch identity issued from the actual callable/settings, not caller text.

The proposal may be fitted offline at a reference parameter `theta_ref` and
then frozen for HMC. If so, record `theta_ref` and make the runtime proposal
and all proposal-density terms independent of the current HMC theta. Runtime
refitting or theta-dependent branch construction is forbidden.

### 2. Source-order evaluator

The generic APF core currently assumes an initial observation followed by later
transitions. Austria SIR requires:

```text
x0 ~ q0
for t = 1,...,T:
    choose/freeze ancestor A_t
    xt ~ qt(. | x_{t-1}^{A_t}, y_t)
    evaluate transition density and observation density at t
```

Do not pass an Austria data set through the generic `y0` slot. Either extend
the evaluator with an explicit event-order mode or add a SIR-specific wrapper
that shares the same TensorFlow/XLA score core. The event-order identifier must
be part of the route identity.

### 3. Complete same-scalar score

The runtime target is the score of the finite observed-data APF scalar, not
the score of one latent path. For a fixed branch and parameter-independent
proposal, define:

\[
\ell_0^i(\theta)=\log p_\theta(x_0^i)-\log q_0(x_0^i),
\]

and for `t=1,...,T`,

\[
\ell_t^i(\theta)=
\log W_{t-1}^{A_t^i}(\theta)
 +\log f_{t,\theta}(x_t^i\mid x_{t-1}^{A_t^i})
 +\log g_{t,\theta}(y_t\mid x_t^i)
 -\log a_{t-1}^{A_t^i}
 -\log q_t(x_t^i\mid x_{t-1}^{A_t^i},y_t).
\]

The last two proposal/auxiliary terms are constants in the runtime theta
evaluation when the branch is correctly frozen. Define

\[
Z_t=\sum_i \exp(\ell_t^i),\qquad
\Delta_t=\log Z_t-\log N,
\qquad \log\widehat L=\Delta_0+\sum_{t=1}^T\Delta_t.
\]

Let `D_{t-1}^j=\nabla_\theta\log W_{t-1}^j`. Define the unnormalized current
mark

\[
M_t^i=D_{t-1}^{A_t^i}
 +\nabla_\theta\log f_{t,\theta}(x_t^i\mid x_{t-1}^{A_t^i})
 +\nabla_\theta\log g_{t,\theta}(y_t\mid x_t^i).
\]

Then:

\[
\nabla_\theta\Delta_t=\sum_i W_t^i M_t^i,
\qquad
D_t^i=M_t^i-\nabla_\theta\Delta_t.
\]

The total score is

\[
\nabla_\theta\log\widehat L=
\nabla_\theta\Delta_0+\sum_{t=1}^T\nabla_\theta\Delta_t.
\]

If the proposal or auxiliary law is not parameter-independent, its derivative
must be derived and included. Do not silently stop gradients or call a partial
derivative a total score. The preferred first implementation freezes the
proposal branch so those terms are exactly zero in the runtime program.

### 4. Proposal density and measure closure

For every generated state, the implementation must be able to evaluate the
same proposal density used in the importance correction. Include:

- squared-TT defensive density and its normalizer;
- paired-core marginal/conditional density;
- physical-coordinate Jacobian or algebraic-map Jacobian;
- any block/product factorization; and
- ancestor-law normalization.

Austria SIR is a full-state Gaussian-noise model in the local model contract,
but the recipient must still verify that no singular or clipped coordinate has
silently changed the reference measure. The `clip_susceptible_after_noise`
policy is part of the model contract and must be represented consistently in
the transition simulation, density, and score. If the transition is actually a
mixed-measure map after clipping, stop and derive an innovation/mixed-measure
route rather than using a Lebesgue density incorrectly.

## Source And Classification Gate

The recipient must inspect and cite both the paper and pinned author source
before implementation. The following anchors are mandatory:

### Paper anchors

- `.localresources/papers/zhao-cui-tensor-train-sequential-learning-jmlr-2024.txt`,
  Section 3.1, Eq. (13): squared-TT plus defensive density;
- the same file, Proposition 2 and the KR conditional construction:
  paired-core marginalization and conditional density;
- the same file, Eq. (20), Eq. (21), Eq. (23), and Algorithm 3: forward
  conditional KR proposal and importance correction;
- the same file, Algorithm 5 / Section 5.4 for marginal carry and
  preconditioning;
- the same file, Section 6.3: Austrian SIR target, `J=9`, `T=20`, rank ladder;
- the same file, Section 6.4 only as a contrast for the predator-prey route.

### Pinned author-source anchors

- `third_party/audit/zhao_cui_tensor_ssm_p10/source/models/full_sol.m:21-43`
  for the sequential push, reapproximation, conditional sampling, and weight
  correction loop;
- `full_sol.m:46-129` for prior/marginal carry, preconditioning, SIRT fit, and
  normalizer update;
- `full_sol.m:132-136` for the transition/likelihood target assembly;
- `third_party/audit/zhao_cui_tensor_ssm_p10/source/deep-tensor.dev/src/@TTSIRT/marginalise.m:19-85`
  for paired-core marginalization;
- `.../@TTSIRT/eval_irt_reference.m:16-71` for the conditional inverse and
  proposal-density evaluation;
- `.../@TTSIRT/eval_irt_reference.m:73-181` for the derivative-capable map
  evaluation surface;
- `.../@TTSIRT/eval_rt_jac_reference.m:17-113` for the transport Jacobian;
- `third_party/audit/zhao_cui_tensor_ssm_p10/source/models/sir_austria/`
  (`setup.mlx`, `odefun.mlx`, `sir_step.mlx`, `transition.mlx`, `like.mlx`,
  `st_process.mlx`, `ob_process.mlx`, `priorpdf.mlx`, `priorsam.mlx`) for the
  model callback and measure conventions.

Classify each operation exactly once:

| Operation | Allowed classification |
| --- | --- |
| Squared-TT positive density and defensive term | `source_faithful` operation only if equations/source match |
| Paired-core marginal and KR conditional | `source_faithful` operation only if equations/source match |
| Frozen uniforms, seeds, ranks, and schedules | `fixed_hmc_adaptation` |
| Austria-specific fixed APF assembly | likely `extension_or_invention` unless source route is reproduced exactly |
| Fixed proposal used while evaluating another theta | `fixed_hmc_adaptation` or `extension_or_invention`; document the target change |
| Analytical recursive score of the assembled finite scalar | extension unless present in the author route |

Do not call the assembled route “source-faithful Zhao-Cui” without proving the
whole route against both anchors. The existing generic compiler explicitly
classifies its assembled route as `extension_or_invention`; preserve that
honesty unless a new audit proves otherwise.

## Implementation Phases

### Phase 0: freeze target and inspect source

1. Re-read this memo, the current leaderboard result, the P89/P90/P91 Zhao-Cui
   result notes, and the source anchors above.
2. Build a target manifest containing observation hash, theta, event order,
   model manifest, dimensions, dtype, and controls.
3. Add a test that rejects the reduced `J=1` route and the local complete-data
   route for the declared full-filtering row.
4. Record the route classification before code changes.

Exit: source anchors and target manifest are complete; no stale target is used.

### Phase 1: tiny same-scalar and event-order adapter

1. Implement an explicit source-order finite scalar for a tiny SIR horizon
   using the fixed-branch APF core.
2. Add an Austria-model adapter exposing the existing manual density-score
   methods in the expected batched shape.
3. Verify `x0 -> transition -> y1` against a direct scalar assembly.
4. Check value and recursive score against central FD only in tests at several
   representative theta points. FD is not runtime provenance.
5. Test fixed-branch replay bitwise/within FP32 tolerance.

Exit: tiny source-order value/score and score additivity pass.

### Phase 2: fixed-TTSIRT proposal branch

1. Fit or compile the actual adjacent proposal using disjoint calibration and
   validation observations/particles.
2. Reuse paired-core conditional and physical Jacobian machinery.
3. Freeze proposal maps, samples, ancestor law, ranks, defensive mass, and all
   random/discrete choices before the claim run.
4. Add pointwise proposal-density, normalization, support, and inverse/forward
   round-trip checks.
5. Measure retained tensors, compile graph size, peak allocator bytes, and
   per-particle working memory.

Exit: a finite `T=1` or `T=2` Austria branch exists without retained-grid
fallback and with a declared proposal measure.

### Phase 3: full recursive score and derivative ownership

1. Implement the normalized-weight derivative recursion above.
2. Include any nonzero derivative of proposal, auxiliary law, Jacobian,
   normalizer, or previous marginal if the proposal is not fully frozen.
3. Prefer a frozen parameter-independent online proposal for the first admitted
   route; document that this is a fixed-HMC adaptation and not an exact
   parameter-adaptive Zhao-Cui proposal.
4. Add per-time increment scores and a score additivity invariant.
5. Add same-scalar FD tests at representative interior theta points and a
   small multi-seed score-variance diagnostic. No FD in runtime.

Exit: the analytical score is demonstrably the total derivative of the exact
finite value program used by the evaluator.

### Phase 4: target-specific tuning

Use a new tuning scope bound to Austria SIR `T=20`, `N=1008`, `d=18`, `p=3`,
FP32/TF32/XLA, event order, proposal family, and all proposal controls.

Tune only on disjoint calibration/validation data. Candidate controls may
include basis degree, TT rank, defensive mass, coordinate scales, transport
iterations, inverse-CDF iterations, regularization, and optional auxiliary-law
controls. Select using a declared objective combining:

- finite/support/normalization validity;
- held-out adjacent-target fit;
- minimum ESS or weight-entropy floor;
- proposal-density/round-trip diagnostics; and
- score stability at representative tuning points, with FD used only as a
  diagnostic.

Do not use an analytical oracle score as a tuning objective. For this model,
the available SGQF score is a comparator, not an oracle. Do not tune on the
untouched claim observation or claim particle seeds.

Exit: a repository-issued tuning artifact exactly matches the claim scope.

### Phase 5: untouched claim run

The historical plan called for running the sealed target with `N=1008`,
`T=20`, FP32, TF32, XLA, and GPU memory
growth. Record:

- value and all three score coordinates;
- per-seed values/scores if using multiple fixed branches;
- ESS, normalized-weight residuals, support/measure checks;
- branch/program/target/tuning identities;
- current/peak TensorFlow allocator bytes;
- device, TF32, XLA, and memory policy;
- command, environment, git commit, seeds, wall time, and artifact paths.

Compare against the existing GenUT and SGQF cells descriptively. Do not claim a
ranking from one frozen target or one small branch.

Exit: either an executed Zhao-Cui Austria SIR `value+score` cell with complete
provenance, or a terminal blocker naming the exact missing mathematical or
resource condition.

### Phase 6: regression and leaderboard integration

Run the focused Zhao-Cui and common high-dimensional tests, then regenerate
only the same-target leaderboard artifact. At minimum preserve regression
checks for:

- LGSSM `T=50`;
- KSC-SV `T=10`;
- exact transformed SV `T=10`;
- generalized SV `T=10`;
- predator-prey `T=20`; and
- Austria SIR `T=20` GenUT/SGQF cells.

Do not overwrite the July 23 artifact. Write a new versioned attempt directory
and record whether the new Zhao-Cui cell is `executed_value_score` or remains
blocked.

## Required Tests

### Mathematical/score tests

- Manual transition and observation density scores agree with diagnostic
  `GradientTape` on tiny batched inputs.
- Source-order finite scalar equals a direct hand assembly.
- Recursive score equals central FD of that same scalar at representative
  interior theta points.
- Sum of per-time increment scores equals returned total score.
- Proposal and auxiliary terms are included or explicitly proven theta
  independent.
- A branch replay produces the same value/score within the declared FP32
  tolerance.

### Proposal/measure tests

- Conditional proposal density agrees with paired-core marginal formula.
- Forward/inverse transport round-trip passes the declared tolerance.
- Defensive density is positive on all generated points.
- Proposal and auxiliary categorical laws normalize.
- Physical-coordinate Jacobian is included exactly once.
- Susceptible clipping policy is consistent between simulation, density, and
  score; if not, fail closed.

### Complexity/runtime tests

- No generic full tensor-product retained grid for `d=18,T=20`.
- No NumPy in the runtime or XLA path.
- No Python sample loop, `tf.map_fn`, or scalar fallback inside the XLA kernel;
  fixed time loops may be compiled as a static TensorFlow graph.
- GPU memory growth is configured and verified before TensorFlow device init.
- `N=1008` claim run uses FP32/TF32/XLA and reports allocator peak.
- Proposal compilation and online evaluation memory are reported separately.

## Expected Failure Modes And Repairs

| Failure | Interpretation | Required response |
| --- | --- | --- |
| Horizon-0 passes but `T=1` fails | Not full filtering score; previous-marginal/proposal transition missing | implement Phase 2/3 ownership; do not promote horizon-0 result |
| Score differs from FD | Implementation or omitted derivative term | localize proposal, Jacobian, normalizer, and carry terms; keep candidate blocked |
| High score variance | finite-branch/APF Monte Carlo instability | increase fixed branches or particles only under a new budget; do not widen gates |
| Low ESS | proposal mismatch | retune proposal/rank/defensive mass on fresh calibration data |
| Memory explosion | architecture infeasible at target scope | stream/shard proposal or reduce rank/block structure; never revive retained grid |
| Clipping changes support/measure | density formula wrong relative to model | derive mixed-measure/innovation route or block |
| Source audit finds route invention | Whole route is not source-faithful | label `extension_or_invention`; do not silently promote source status |
| Generic evaluator event order mismatch | Wrong target even if finite | add explicit event-order mode and invalidate affected artifact |

## Definition Of Done

The implementation is complete only if all conditions hold:

1. The sealed Austria SIR target is identified by exact observation hash,
   event-order contract, theta convention, and model manifest.
2. A fixed proposal branch exists for `T=20` without the historical retained
   tensor-product grid.
3. The finite observed-data value program is documented and finite.
4. The returned score is the manual analytical total derivative of that same
   finite value program.
5. Proposal, ancestor, Jacobian, previous-marginal, and normalizer derivatives
   are either included or mathematically zero because the branch is frozen.
6. Runtime contains no autodiff or finite-difference score path.
7. Focused tests pass, including source-order and score/FD diagnostics.
8. The untouched `N=1008`, FP32/TF32/XLA GPU run has a complete manifest and
   allocator/memory evidence.
9. The result explicitly classifies the assembled route and does not call it
   source-faithful without a completed source audit.
10. The leaderboard cell is regenerated only in a new artifact directory and
    is compared descriptively to GenUT/SGQF.

If any item fails, report the precise blocker. Do not substitute the existing
local complete-data SIR score or the reduced `J=1` diagnostic to make the cell
appear complete.

## Required Handoff Outputs

The recipient must return:

1. implementation changes under the relevant `bayesfilter/highdim` modules;
2. focused and regression tests;
3. a target-specific plan/result note with the source-anchor ledger;
4. a structured JSON run manifest/result for the Austria SIR claim or blocker;
5. a score/value comparison against the existing same-target GenUT and SGQF
   rows; and
6. a reset memo recording route identity, classification, controls, artifacts,
   remaining gaps, and explicit nonclaims.

The current July 23 leaderboard remains the baseline until these outputs exist.
