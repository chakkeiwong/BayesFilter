# Claude Algorithm Review Memo: Zhao-Cui Austria SIR Fixed-Variant Parameter Extension

Date: 2026-07-30

Status: `READY_FOR_CLAUDE_READ_ONLY_ALGORITHM_REVIEW`

Review target:
`docs/plans/bayesfilter-zhao-cui-austria-sir-fixed-variant-parameter-extension-master-plan-2026-07-30.md`

Required review result path:
`docs/plans/bayesfilter-zhao-cui-austria-sir-fixed-variant-parameter-extension-claude-algorithm-review-result-2026-07-30.md`

## Role And Output Protocol

You are the independent mathematical and algorithmic reviewer. Perform a
read-only review. Do not edit files, run commands, launch agents, execute code,
or start experiments. Read this memo first, then inspect only the exact paths
and cited line regions that this memo explicitly names.

Write a complete Markdown review back to Codex in your response. Codex will
save your response verbatim at the required review result path above. Do not
claim that you wrote the repository file yourself.

Your response must:

1. begin exactly with
   `# Claude Review: Zhao-Cui Austria SIR Fixed-Variant Parameter Extension`;
2. be self-contained and suitable for direct persistence as the result file;
3. put blocking or correctness findings first, ordered by severity;
4. cite exact repository paths and line numbers for every material finding;
5. distinguish `correct`, `wrong relative to the stated target`, `unsupported`,
   `not checked`, and `heuristic only`;
6. distinguish a plan defect from an implementation gap, missing evidence, or
   expected future work;
7. propose precise plan edits for every blocking finding;
8. state whether Phase 0 may execute as written;
9. state what you inspected and what remains uninspected; and
10. end with exactly `VERDICT: AGREE` or `VERDICT: REVISE`.

`VERDICT: AGREE` means the plan is mathematically coherent, starts from the
right baseline, makes only supported claims, has implementable gates, and is
safe to authorize for Phase 0 only. It does not mean later phases, value/score
correctness, T20, GPU, production, or HMC are established.

## Owner Direction That Must Be Preserved

The project direction is to extend the BayesFilter fixed-variant line, not to
return to the original Zhao-Cui random TT-cross/ALS estimator and not to
substitute the July 30 APF/source-replica experiments.

The intended later delta is:

```text
exact fixed-variant parent
    + three external log-scale parameters
    + parameter-conditioned TT representation
    + analytical total score of the same finite scalar
    + streaming extension to T20
```

HMC is out of scope. The only currently active work is Phase 0 baseline
reconstruction and identity freeze.

Do not return `AGREE` merely because the plan is conservative. Verify that its
baseline, scalar, mathematics, derivative ownership, memory model, and phase
ordering are internally and externally supported.

## Mandatory Review Sources

Read the complete canonical plan:

- `docs/plans/bayesfilter-zhao-cui-austria-sir-fixed-variant-parameter-extension-master-plan-2026-07-30.md`.

Then inspect these exact evidence surfaces as needed to answer the review:

- Exact P88 artifact:
  `docs/plans/bayesfilter-highdim-zhao-cui-p88-phase2-degree-order3-rank4-lr3e-4-l1-0-fit-2026-06-27.json`.
  Inspect at least lines 1-105, 280-340, and 22015-22180.
- P88 fit construction:
  `scripts/p86_author_lagrangep_phase5_budget_fit.py:3380-3525` and the artifact
  serialization/loading-related code reached from that region.
- Squared-TT runtime:
  `bayesfilter/highdim/squared_tt.py`, especially the density, defensive-mass,
  normalizer, core-shape, identity, and serialization semantics used by P88.
- Fixed TTSIRT runtime:
  `bayesfilter/highdim/transport.py`, especially `FixedTTSIRTTransport`,
  marginalization, inverse transport, proposal density, and normalizer
  semantics.
- Retained-object scalar and recursion:
  `bayesfilter/highdim/source_route.py:454-490`, `:1852-1892`,
  `:8406-8505`, `:8532-8635`, and `:8792-8905`.
- P90 comparator contract and executed fixture:
  `docs/plans/bayesfilter-highdim-zhao-cui-p90-value-bridge-contract-2026-06-28.md`
  and
  `docs/plans/bayesfilter-highdim-zhao-cui-p90-phase3-value-bridge-execution-result-2026-06-28.md`.
- Parameterized Austria SIR model:
  `bayesfilter/highdim/models.py:935-1005` and `:1339-1365`.
- Existing derivative blockers and ownership surfaces:
  `bayesfilter/highdim/source_route.py:1400-1585`,
  `docs/plans/bayesfilter-highdim-zhao-cui-p90-phase5-derivative-implementation-result-2026-06-28.md`, and
  `docs/plans/bayesfilter-highdim-zhao-cui-p91-phase3-fd-consistency-result-2026-06-29.md`.
- Paper source:
  `.localresources/papers/zhao-cui-tensor-train-sequential-learning-jmlr-2024.txt:680-765`,
  `:880-924`, and `:1325-1369`.
- Author code:
  `third_party/audit/zhao_cui_tensor_ssm_p10/source/models/full_sol.m:21-135`,
  `third_party/audit/zhao_cui_tensor_ssm_p10/source/eg3_sir/mainscript.m:14-56`,
  `third_party/audit/zhao_cui_tensor_ssm_p10/source/deep-tensor.dev/src/@TTSIRT/marginalise.m:1-87`, and
  `third_party/audit/zhao_cui_tensor_ssm_p10/source/deep-tensor.dev/src/@TTSIRT/eval_cirt_reference.m:43-100`.

If one of these sources points to a directly required helper, inspect only that
helper and cite it. Do not broaden into a repository-wide review.

## Questions That Must Be Answered

### 1. Baseline Identity And Phase 0 Feasibility

Determine exactly what P88 proves and serializes.

- Is P88 correctly characterized as a `time_index=1`, 36-axis, rank-4,
  order-3, TensorFlow `training_base_optimizer` fit artifact?
- Does it contain enough frame, basis, measure, defensive-density, shift,
  normalizer, core, seed, and callable metadata to reconstruct the exact T1
  finite program?
- Is the stated SHA-256/provenance binding sufficient and correctly scoped?
- Is it correct that P88 contains no July 30 observation-hash binding and no T2
  trained cores?
- Can its serialized cores actually be loaded by `SquaredTTDensity` and
  `FixedTTSIRTTransport` without changing basis ordering, reference measure,
  frame, `tau`, shift, or branch identity?
- Can the exact P88 T1 retained object be carried to the T2 previous-marginal
  target boundary without constructing a T2 transport?
- Does any selected local API require at least two completed steps, making the
  Phase 0 T1-only wording or proposed artifact type infeasible?
- Can the P90 independent comparator architecture be instantiated on the exact
  P88 T1 branch, or does it rely on a different dimension, branch, fixture, or
  binding that makes the Phase 0 criterion unsupported?

Classify the Phase 0 exit criterion as `correct`, `wrong relative to the stated
target`, `unsupported`, or `implementable with named missing plumbing`.

### 2. Fixed Scalar And Probability Measure

Audit the exact finite scalar rather than its label.

The plan claims

\[
L_0=\sum_{t=1}^T(\log Z_t^0-c_t^0).
\]

Check:

- the sign and ownership of `c_t`/the shift against the P88 fit and author
  `log(sirt.z)-const` code;
- whether `Z_t` is the normalizer of the physical target, reference target,
  fitted squared-TT approximation, or another object;
- every affine determinant and reference-measure Jacobian;
- whether proposal-correction weights affect the claimed scalar directly,
  indirectly through the next retained object, or not at all;
- the exact relationship between this scalar and an observed-data log
  likelihood;
- whether clipping/push mechanics and Gaussian transition-density evaluation
  define a coherent common measure; and
- whether the plan's `operational finite approximation` nonclaim is precise
  enough or hides a target mismatch.

State the claimed quantity, the actually computed quantity, and whether they
are equal, approximately related, different, or not yet checked.

### 3. Defensive Squared-TT Algebra

The plan models the density as

\[
\rho_t=\phi_t^2+\tau_t\lambda_t.
\]

Verify this against `SquaredTTDensity`, P88 metadata, the reference measure,
and author squared-TT construction.

- Is `tau=1e-8` the correct field and interpretation?
- Is `lambda` explicitly represented, implicit in the basis measure, or
  different from the plan's notation?
- Is the normalizer exactly the integral of `phi^2 + tau*lambda`?
- Does extending the amplitude by
  `phi_t(u,z)=phi_t^0(z)+C_t(u,z)` preserve the exact origin density when
  `C_t(0,z)=0`?
- Are additional constraints required on ranks, gauges, parameter-core
  placement, defensive mass, or state cores to make the origin identity exact?
- Can the parent state cores truly remain byte-identical while adding three
  parameter axes and nontrivial parameter-state coupling, or must the plan
  specify a block-rank embedding whose parent slice is functionally identical
  but whose stored state cores differ?

Do not accept `C_t(0,z)=0` as sufficient unless the proposed TT architecture
can enforce it algebraically.

### 4. Conditioning Semantics

The target is an outer likelihood function of
`theta=(log_kappa_scale, log_nu_scale, log_obs_noise_scale)`. Theta must be
conditioned, not marginalized and not sampled as an inner particle coordinate.

Check:

- whether evaluating parameter cores first and integrating only state axes is
  mathematically correct;
- whether calling theta a `fixed prefix` conflicts with the inherited
  `[x_t, theta, x_{t-1}]` TT ordering or the local transport API;
- whether the three added axes should be a prefix, middle block, separate
  coefficient network, or another representation to preserve the fixed route;
- whether the conditional KR map generates only state coordinates; and
- whether the plan accidentally changes the probability measure or estimator
  when converting the author's joint parameter-estimation construction into an
  externally conditioned likelihood program.

Classify this as an explicit BayesFilter extension, not source-faithful Zhao-Cui
parameter estimation.

### 5. Total Score Of The Same Program

The plan claims a total derivative through retained state:

\[
D_aL=\sum_t\left(Z_t^{-1}D_aZ_t-D_ac_t\right),
\]

\[
S_{t,a}=\partial_a\Phi_t+D_R\Phi_t S_{t-1,a},
\qquad
D_aZ_t=\partial_aZ_t+D_RZ_tS_{t-1,a}.
\]

Audit whether this abstraction covers the actual program. Identify every
required derivative owner, including:

- transition, observation, and initial density;
- parameter chart and basis;
- TT amplitude, squared density, defensive mass, and normalizer;
- affine frame and determinant;
- shift selection/freeze policy, including the nondifferentiability of a
  runtime minimum if applicable;
- previous retained marginal and its marginalization/evaluation;
- normalized correction weights;
- inverse conditional transport and proposal density;
- frozen random numbers, deterministic resampling/genealogy, and any branch
  selection;
- retained samples, retained weights, and compact retained identities; and
- all paths by which an earlier theta changes a later `Z_t`.

Decide whether `R_t` is a sufficiently explicit finite-dimensional state for
manual forward sensitivity. If not, state the exact retained fields and
sensitivities the plan must name.

Explain which transport/proposal terms enter the value directly, which enter
only through `R_t`, and which are truly theta independent. Do not permit a
local transition-plus-observation score, stopped-gradient retained object, or
runtime autodiff/finite difference to be labeled the manual total score.

### 6. Memory And Computational Complexity

Audit the claim that the method can remain TT-linear and horizon-streaming.

- Verify the core-memory expression and whether 39 axes is the correct count.
- Account for basis dimensions, ranks, defensive terms, three score
  coordinates, marginal contractions, inverse-CDF workspaces, retained samples,
  normalized weights, and derivative state.
- Determine whether keeping only the current and previous retained objects is
  sufficient for the total derivative, or whether the proposed forward
  sensitivity must carry additional history-dependent tensors.
- Identify any operation that would materialize a theta-state tensor grid,
  sample-by-axis Jacobian, full transport Jacobian, or all-time computation
  graph.
- Check whether Phase 3's batch-native training requirement is compatible with
  the proposed parameterized target evaluation and memory caps.
- Decide whether the 12 GiB and 512 MiB caps are justified bounds, provisional
  hypotheses requiring Phase 0/1 measurement, or unsupported numbers.
- State a corrected asymptotic and practical memory model if the plan's model
  is incomplete.

### 7. Phase Ordering, Evidence Gates, And Failure Modes

Review all phases and the anti-drift guards.

- Are guards assigned to the first phase that can implement them?
- Does Phase 0 avoid training, UKF insertion, parameter work, GPU, and HMC?
- Does Phase 1 preserve the trainer family while correctly treating rank,
  degree, LR, L1, basis, and budgets as scope-specific hypotheses?
- Can Phase 2 establish an exact origin slice before Phase 3 trains coupling
  channels?
- Does Phase 3 define enough target-specific training and downstream validation
  to avoid promoting heldout loss alone?
- Does Phase 4 close every previous-marginal and transport derivative blocker
  before any T20/GPU claim?
- Is the T1/T2/T5/T10/T20 ladder ordered correctly for early memory and
  numerical failure detection?
- Are continuation vetoes, repair triggers, promotion vetoes, and explanatory
  diagnostics correctly separated?
- Could the plan pass every written gate while computing the wrong value or
  score? Give the strongest pass-while-wrong scenario.
- Could it fail because of an implementation/default choice rather than the
  scientific idea? Give the smallest discriminating diagnostic.

### 8. Forbidden Drift And Comparison Scope

Confirm that the plan cannot silently switch to:

- original-author random TT-cross/ALS;
- P76/P77 UKF as alleged inherited baseline behavior;
- July 30 source replica;
- frozen-proposal APF;
- generic retained-grid evaluation;
- theta marginalization or theta particles;
- runtime autodiff/finite-difference score; or
- HMC before correct value and score.

Also check that later GenUT, SGQF, and UKF comparisons use the same observations,
target, parameter point/domain, finite-program interpretation, dtype, and
uncertainty discipline. These comparisons are not Phase 0 authorization.

## Required Result Structure

Return one complete Markdown document with these sections in this order:

1. `# Claude Review: Zhao-Cui Austria SIR Fixed-Variant Parameter Extension`
2. `## Verdict Summary`
3. `## Blocking Findings`
4. `## Major Nonblocking Findings`
5. `## Mathematical Audit`
6. `## Baseline And Source Audit`
7. `## Score Ownership Audit`
8. `## Memory And Complexity Audit`
9. `## Phase And Failure Audit`
10. `## Required Plan Edits`
11. `## Phase 0 Authorization Decision`
12. `## Inspected Sources`
13. `## Residual Uncertainty And Nonclaims`
14. final verdict line.

For each finding use:

```text
Severity: BLOCKER | MAJOR | MINOR
Classification: correct | wrong relative to the stated target | unsupported | not checked | heuristic only
Evidence: exact path:line anchors
Problem: precise mismatch
Consequence: what would be computed or claimed incorrectly
Required repair: exact plan or gate change
```

If there are no blocking findings, write `None` under `## Blocking Findings`.
Do not omit a section. Do not soften a mathematical mismatch into a stylistic
suggestion. End exactly with one of:

```text
VERDICT: AGREE
```

or

```text
VERDICT: REVISE
```

## Exact Initial Claude Prompt

Use this prompt for the first bounded review call:

```text
READ-ONLY BOUNDED REVIEW. Review exactly this path and nothing else unless the
file itself explicitly asks you to inspect a cited path or line:
docs/plans/bayesfilter-zhao-cui-austria-sir-fixed-variant-parameter-extension-claude-algorithm-review-memo-2026-07-30.md.
Do not edit, run commands, launch agents, or review the whole repo. Perform the
source-grounded mathematical, algorithmic, derivative, memory, and phase audit
specified by the memo. Return the complete Markdown review in your response so
Codex can save it verbatim to the result path named in the memo. End exactly
with VERDICT: AGREE or VERDICT: REVISE.
```
