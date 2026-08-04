# Zhao-Cui Austria SIR Score Completion Plan

Date: 2026-08-02

Status: `PROPOSED_PENDING_MATERIAL_PLAN_REVIEW`

## Decision

The Austria score computation is not complete. The active evidence closes
material replay and local finite-difference tangent mechanics at `theta=0`
through T2, but it does not provide a reusable total score at nonzero
parameters or at the sealed `T=20` horizon.

Close the executable score gap with a **frozen-branch, manual analytical
importance-score recursion**. Offline Zhao-Cui-derived squared-TT/KR objects
may construct and freeze proposal branches. At runtime, the proposal samples,
ancestor choices, proposal densities, shifts, maps, and all discrete choices
are theta-independent. The runtime score is the manual total derivative of
the exact same frozen finite likelihood scalar and contains no derivative of
TT training, no finite differences, no autodiff, and no radial projection.

This route is classified `fixed_hmc_adaptation` for freezing the paper's
proposal randomness and `extension_or_invention` for the external
three-parameter likelihood-score assembly. It must be called a
**Zhao-Cui-derived Austria finite-program score**, not a source-faithful Zhao-Cui
Austria parameter score.

### Target-Replacement Disclosure

The selected scalar is not the current T1/T2 trained-TT normalizer
`sum_t(log(sirt.z_t)-const_t)`. It is a new frozen importance-filter likelihood
scalar that uses TT/KR objects as proposal components. The two quantities may
be close, but they are not equal by construction. Therefore:

- the admitted T1/T2 values and FD scores are historical/regression evidence,
  not parents of the new claim;
- the new route must recompute value and score together from T1 onward;
- it must issue a new route ID and a new leaderboard row label;
- it must not call a difference from the old T1/T2 value a replay residual;
  that difference is a cross-program diagnostic; and
- executing this plan is a material target choice for the score cell. Fable
  review is required, and the user's later request to execute the reviewed
  plan will constitute acceptance of this explicitly disclosed target.

If the intended deliverable is instead the derivative of the trained-TT
normalizer itself, this plan must not execute. That alternative still requires
an exact/manual derivative of the complete training map or a separately
defined differentiable joint-theta TT; the current centered-FD/radial route
does not close it.

The distinction is forced by the sources. Zhao and Cui's general Algorithm 2
supports joint parameter-state learning, but their Austria SIR experiment
fixes `kappa_j=0.1` and `nu_j=18` and performs state inference. The author
Austria script sets `d=0`. The repository's external parameters

```text
(log_kappa_scale, log_nu_scale, log_obs_noise_scale)
```

are a BayesFilter extension of that fixed-parameter example.

## Current Evidence And Exact Gaps

| Area | Current evidence | Remaining gap | Closable in this plan? |
|---|---|---|---|
| T1/T2 value programs | Frozen trained-TT T1 and T2 fixed-value artifacts exist and passed their declared value gates | Nothing beyond T2 is admitted, and these values are not the selected frozen-filter scalar | Yes for a new value-plus-score target, not by differentiating or extending the old scalar |
| Local T1/T2 score mechanics | T1/T2 centered-FD tangents passed material replay and step-halving checks at `theta=0` | They are not exact/manual derivatives; T2 uses a radial projection | Yes, by replacing them as claim provenance with a manual recursive score |
| Same-scalar identity | T1/T2 scalar residuals are about `4.26e-14` | T2 scalar consistency is enforced by construction; no off-origin same-scalar validation exists | Yes, at multiple untouched interior theta points |
| Parameter scope | Only `theta=0` is supported by the active issuer chain | No validated parameter domain or arbitrary-theta score | Partly: close a predeclared compact validation domain; do not claim all theta |
| Horizon | T2 cumulative score is issued | T3 through sealed T20 are absent | Yes, subject to staged validity/resource gates |
| Proposal law | T1 retained sampler law passed exact interval-mass, density, roundtrip, and high correction ESS checks; a T2 value route passed | No frozen, correctly scored branch chain through T20; historical APF proposal failed T1 ESS | Possibly; proposal quality is the main empirical risk |
| Tail algebra | T2 value has a signed-log certificate for one nonrepresentable row | Its score contribution was not independently derived; later horizons may expose more tails | Yes only if stable value-and-score algebra is derived; otherwise a terminal blocker |
| XLA route | T1/T2 numerical replay is TensorFlow/XLA native | No T20 compiled manual score kernel or graph audit exists | Yes |
| NumPy/Python numerical control | Absent from active T1/T2 claim kernels | Later-horizon code could reintroduce it | Yes, by static graph inspection and runtime tests |
| Physical likelihood accuracy | Independent T1 importance authorities exist with MCSE; active scores are local finite-program mechanics | No T20 independent same-target authority or uncertainty-qualified comparison | Partly; obtain independent reference evidence, but do not call a finite branch the exact likelihood |
| Source status | Individual adjacent target, squared-TT marginal, and KR operations have paper/code anchors | The external three-parameter Austria score is not in the author example | No. Label honestly; only new author evidence could change this classification |
| HMC readiness | None | Smoothness, determinism, domain coverage, and posterior diagnostics are absent | No. HMC is outside this plan |

## Research Intent Ledger

| Field | Definition |
|---|---|
| Main question | Can a frozen Zhao-Cui-derived proposal program produce a finite, deterministic, memory-bounded `T=20` Austria observed-data likelihood scalar and its manual total score over a predeclared compact parameter-validation domain? |
| Claimed target | The value and manual total derivative of one repository-defined frozen finite importance-filter scalar with sealed Austria observations and event order `z0 -> transition -> y1 -> ... -> transition -> y20`. |
| Candidate mechanism | Offline scope-tuned squared-TT retained densities and exact-law conditional samplers; frozen references and ancestry; compiled normalized-weight score recursion using local analytical density scores. |
| Expected failure mode | Proposal ESS collapses with horizon or theta; a tail row lacks stable derivative algebra; the fixed branch leaves the validated domain; score carry omits a term; or XLA memory/control-flow limits are exceeded. |
| Primary promotion criterion | On the untouched T20 claim branches, finite value and three-coordinate manual score, score additivity, fresh replay, strict source/tuning identity, and same-scalar diagnostic derivative agreement at the predeclared validation points. |
| Promotion veto | Wrong target/event order; nonfinite value/score; omitted proposal/ancestor/Jacobian/normalizer/carry term; proposal law mismatch; score check failure; ESS/weight-degeneracy failure; graph fallback; stale tuning; source-identity failure; or memory breach. |
| Continuation veto | The mathematical proposal measure is undefined, stable tail score cannot be derived, no candidate passes the staged T1/T2/T3 proposal screen within budget, an independent authority is invalid, or target-specific tuning budget is exhausted. |
| Repair trigger | A bounded rank, basis, regularization, branch-count, chunking, XLA lowering, serialization, or proposal-quality failure with unchanged target and campaign budget. |
| Explanatory diagnostics | Training loss, density residuals, ESS above its veto floor, weight tails, runtime, allocator peak, FD residuals after passing, and descriptive GenUT/SGQF/UKF differences. |
| Must not be concluded | Exact physical likelihood, source-faithful Austria parameter inference, arbitrary-theta correctness, HMC readiness, posterior correctness, default readiness, production readiness, or superiority. |

## Source-Support And Classification Ledger

The source check was performed from the locally stored published paper and
pinned author code. This is a direct-method implementation audit, not a broad
novelty survey. Network metadata, citation counts, venue ranks, and forward
snowballing are not needed for the route classification and are not claimed.

| Claim or operation | Support class | Anchor | Allowed conclusion |
|---|---|---|---|
| General joint parameter/state posterior and parameter marginal | `PRIMARY_TECHNICAL_SUPPORT` | Zhao-Cui Eqs. (5)-(6), Section 2.1, Algorithm 1 | The general method includes theta as joint TT coordinates and obtains `p(theta|y)` by marginalization |
| Adjacent target `previous marginal * transition * likelihood` | `PRIMARY_TECHNICAL_SUPPORT` | Zhao-Cui Eq. (15), Algorithm 2(a) | This target assembly is source-faithful when implemented as written |
| Squared-TT nonnegative approximation and marginal carry | `PRIMARY_TECHNICAL_SUPPORT` | Zhao-Cui Eqs. (13), (16), Proposition 2, Algorithm 2(b-c) | Squared-TT marginal mechanics are source-grounded |
| Conditional KR proposal and importance correction | `PRIMARY_TECHNICAL_SUPPORT` | Zhao-Cui Eqs. (20)-(23), Algorithm 3 | A correctly evaluated conditional proposal and correction are source-grounded operations |
| Author sequential target and normalizer assembly | `IMPLEMENTATION_EVIDENCE` | `third_party/audit/zhao_cui_tensor_ssm_p10/source/models/full_sol.m:21`, `full_sol.m:46`, `full_sol.m:72`, `full_sol.m:124`, `full_sol.m:132` | Confirms operation order, prior carry, fit, and `log(sirt.z)-const` accumulation |
| Author marginal and conditional implementation | `IMPLEMENTATION_EVIDENCE` | `third_party/audit/zhao_cui_tensor_ssm_p10/source/deep-tensor.dev/src/@TTSIRT/marginalise.m:19`, `third_party/audit/zhao_cui_tensor_ssm_p10/source/deep-tensor.dev/src/@TTSIRT/eval_irt_reference.m:16` | Confirms contraction direction and conditional inversion structure |
| Austria SIR target | `PRIMARY_TECHNICAL_SUPPORT` | Zhao-Cui Section 6.3, Eq. (37), especially paper text at lines 2274-2276 and 2332-2364 in the local text | Supports `J=9`, `T=20`, fixed kappa/nu, state inference, basis/rank context |
| Author Austria scope | `IMPLEMENTATION_EVIDENCE` | `third_party/audit/zhao_cui_tensor_ssm_p10/source/eg3_sir/mainscript.m:12-17`, especially `d=0` | Blocks a claim that the author's Austria example performs parameter inference |
| External three-parameter Austria likelihood score | `PROJECT_DERIVATION` plus `IMPLEMENTATION_EVIDENCE` | `bayesfilter/highdim/models.py` `ParameterizedZhaoCuiSIRSSM`; derivation below | `extension_or_invention`, not source-faithful Austria parameter inference |
| Exact interval-mass retained sampler | `PROJECT_DERIVATION` plus source-mapped operation | Existing B2 sampler result and `bayesfilter/highdim/zhao_cui_austria_sir_lane_b_sampler_tf.py` | Correct finite proposal law; not the author's numerical CDF implementation |
| Frozen proposal branch | `fixed_hmc_adaptation` | Project finite-program definition | Proposal derivative is zero only for the frozen finite scalar, not for an adaptive Zhao-Cui trainer |
| Manual recursive importance score | `PROJECT_DERIVATION` | Derivation below | Total derivative of the declared frozen finite scalar if every owned term is included |

### Literature Audit Boundary

```text
decision: sufficient for source classification, not a completeness survey
metadata_date: 2026-08-02
seed_papers: Zhao and Cui, JMLR 2024
source_support_summary: published PDF/text and pinned author implementation inspected
citation_venue_summary: not needed and not checked
backward_snowball_summary: not required for this direct operation mapping
forward_snowball_summary: not checked; no novelty claim
quarantined_sources: none known; formal retraction/erratum search not performed
top_omission_risks: later parameter-score extensions of the method may exist but cannot alter what the inspected Austria example computes
claim_support_gaps: no inspected source derives the repository's external three-coordinate Austria likelihood score
next_required_actions: Fable must verify the listed paper and author-code anchors before agreeing with source classifications
what_is_not_concluded: literature completeness, novelty, or source-faithful Austria parameter inference
```

## Claimed Finite Scalar And Manual Score

Let branch `b` contain fixed initial states, reference variates, proposal maps,
ancestor indices, proposal log densities, and auxiliary ancestor
probabilities. None depends on runtime `theta`.

For `N` particles, define initial log marks

\[
  a_0^i(\theta)=\log p_\theta(z_0^i)-\log q_0^i,
\]

and for `t=1,...,T`,

\[
  a_t^i(\theta)=
  \log W_{t-1}^{A_t^i}(\theta)
  +\log f_\theta(z_t^i\mid z_{t-1}^{A_t^i})
  +\log g_\theta(y_t\mid z_t^i)
  -\log \alpha_{t-1}^{A_t^i}
  -\log q_t^i.
\]

Here `q_t`, `alpha_t`, the states, and the ancestry are literal fixed branch
data. The finite value is

\[
  \Delta_t(\theta)=\operatorname{LSE}_i a_t^i(\theta)-\log N,
  \qquad
  \widehat\ell_T(\theta)=\sum_{t=0}^{T}\Delta_t(\theta),
\]

with normalized weights

\[
  W_t^i(\theta)=\operatorname{softmax}_i a_t^i(\theta).
\]

Let `D_t^i = nabla_theta log W_t^i`. With local manual model scores
`s_init`, `s_f`, and `s_g`, define

\[
  M_0^i=s_{\mathrm{init}}^i,
  \qquad
  \nabla_\theta\Delta_0=\sum_i W_0^iM_0^i,
  \qquad
  D_0^i=M_0^i-\nabla_\theta\Delta_0,
\]

and

\[
  M_t^i=D_{t-1}^{A_t^i}+s_{f,t}^i+s_{g,t}^i,
\]

\[
  \nabla_\theta\Delta_t=\sum_i W_t^iM_t^i,
  \qquad
  D_t^i=M_t^i-\nabla_\theta\Delta_t.
\]

The returned score is

\[
  \nabla_\theta\widehat\ell_T(\theta)
  =\sum_{t=0}^{T}\nabla_\theta\Delta_t.
\]

This is equal to the derivative of the declared finite scalar only because the
branch is theta-independent. If any proposal map, density, ancestor law,
Jacobian, shift, state, or branch selection depends on theta, its derivative
must be included or that object must be frozen before claim execution. A
stopped gradient is not a proof of independence.

Multiple branches may be combined only through one predeclared scalar, for
example

\[
  \widehat\ell_{B,T}(\theta)
  =\log\left(B^{-1}\sum_b\exp\widehat\ell_T^{(b)}(\theta)\right),
\]

whose score is the likelihood-weighted branch-score average. Averaging branch
scores without differentiating the same combined scalar is forbidden.

## Score Accuracy Contract

The five-significant-digit material rule remains appropriate for replayed
functional values:

```text
abs(replayed - reference) <= 5e-12 + 5e-6 * abs(reference)
```

It is not automatically the right absolute rule for a near-zero derivative.
For same-scalar score diagnostics use, per coordinate,

```text
abs(manual - diagnostic) <= score_atol + score_rtol * max(abs(manual), abs(diagnostic))
```

where `score_atol` and `score_rtol` are calibrated before untouched claims from
deterministic FP64 step-halving and backend-rounding evidence. Initial
hypotheses are `score_atol=5e-6`, `score_rtol=5e-5`; they are not frozen
defaults until Phase 1 calibration shows that they distinguish known
omitted-term mutants from valid numerical drift. Tolerance calibration may
tighten or reject the route, but may not be loosened after untouched claim
results are read. Independent stochastic-reference comparisons use their own
MCSE/uncertainty screen and must not be folded into this deterministic
same-scalar tolerance.

Central finite differences and `GradientTape` are independent diagnostics in
tests only. They must not appear in the runtime or issuer provenance. At each
validation point, require step halving (`h`, `h/2`, `h/4`) to identify a stable
range; a single FD step cannot promote the score.

## Parameter Domain Contract

The historical `[-0.5,0.5]^3` box is a prior diagnostic convention, not a
proved HMC domain. Phase 1 selects from the predeclared nested half-width
ladder `{0.03, 0.10, 0.25, 0.50}` and freezes the largest symmetric box whose
model, frozen branches, and reference diagnostics remain finite and
informative. These half-widths are calibration hypotheses, not established
defaults. Domain selection uses calibration points only.

At minimum, the validation design includes:

- the origin;
- both signs on every coordinate axis at inner and outer radii;
- all eight box corners;
- at least eight fixed mixed-coordinate interior points; and
- untouched points drawn before final tuning is frozen.

Passing supports only that compact validation domain. It does not establish
arbitrary-theta correctness or an HMC prior.

## Baseline And Comparator Ladder

| Rung | Purpose | Promotion role |
|---|---|---|
| Direct tiny scalar and autodiff/FD derivative | Mathematical authority at `T=1`, small `N` | Hard implementation veto only |
| Bootstrap/fixed Gaussian frozen branch | Naive baseline for recursion and variance | Required baseline, not Zhao-Cui promotion evidence |
| Existing T1/T2 material FD artifacts | Cross-program regression context at the origin | Explanatory only unless identical scalar/branch identity is independently proved; projected T2 tangent is not derivative authority |
| Frozen Zhao-Cui-derived exact-law proposal branch | Plain proposed finite program | Claim candidate |
| Scope-tuned multi-branch or enhanced proposal | Enhanced candidate only if plain route fails a declared proposal-quality gate | Must retain the same scalar and score contract |
| Independent conditional/reference filter | Same-target external check with MCSE/ESS | Veto/uncertainty evidence, not exact oracle unless derived |
| GenUT/SGQF/UKF | Existing same-model comparators | Descriptive unless uncertainty supports ranking |

## Phased Execution

### Phase 0: Source, Target, And Historical-Evidence Seal

1. Recheck the paper and author-code anchors in the ledger.
2. Seal the observation tensor/hash, event order, latent-preclip target,
   parameter convention, horizon, dtype, model version, local score callables,
   and the fact that the frozen-filter scalar replaces rather than
   differentiates the trained-TT normalizer for the new score row.
3. Preserve the current T1/T2 issuer identities and classify them as
   `finite_difference_local_mechanics_only`.
4. Build a source/classification test that rejects these labels:
   `source_faithful_austria_parameter_score`, `exact_autodiff`, and
   `arbitrary_theta`.

Exit: one target manifest and one operation-classification ledger; no new
experiment.

### Phase 1: Tiny Frozen-Scalar Derivation And Domain Calibration

1. Implement a direct tiny frozen-branch scalar and the manual recursion above
   for `T=1` and `T=2`.
2. Verify local model scores against diagnostic autodiff on batch-native FP64
   tensors.
3. Verify manual total score against `GradientTape` and three-step centered FD
   of the same frozen scalar at the calibration-domain design points.
4. Add deliberate omitted-carry, omitted-observation-score, omitted-proposal-
   correction, and wrong-event-order mutants; the gates must reject them.
5. Calibrate `score_atol`, `score_rtol`, and the compact parameter-validation
   domain before untouched points are opened.
6. Derive stable value and derivative treatment for rows whose RK4 state or
   Gaussian residual is outside ordinary FP64 range. If a row's finite value
   is declared zero, prove whether its derivative contribution is zero in the
   same finite program; do not infer this from the T2 value certificate.

Exit: exact finite-program derivative identity on tiny cases, meaningful
tolerance, sealed validation domain, and stable tail-score contract.

### Phase 2: XLA-Native Manual Score Kernel

1. Implement the batched scalar and score recursion with TensorFlow operations
   and `tf.while_loop` over every iterative numerical axis not eliminated by
   batch vectorization, including time and RK4 substeps.
2. Use XLA JIT by default. New claim-owned modules and runners must not import
   NumPy. No `PyFunc`, eager decision, Python numerical loop (including a
   statically unrolled RK4 loop), `tf.map_fn`, or scalar fallback may influence
   the numerical path.
3. Keep Python only at static configuration, artifact I/O, hashing, manifest,
   and reporting boundaries. Tensor materialization at those boundaries must
   not feed a numerical runtime decision.
4. Inspect the concrete graph for `While`/XLA ownership and absence of host
   callbacks. Add CPU-hidden parity tests and trusted-GPU smoke tests.
5. Configure the reviewed 6,144 MiB logical-device cap before TensorFlow GPU
   initialization and record the memory-policy exception correctly.

Exit: tiny and T2 XLA kernels reproduce the FP64 reference within calibrated
tolerance and fail closed on graph fallback.

### Phase 3: Frozen Proposal Chain And Scope-Specific Tuning

This phase does not differentiate or replay optimizer training. It consumes
offline proposal artifacts.

1. Reuse the passed exact interval-mass T1 retained sampler and its correctly
   scored finite law where compatible.
2. Build fresh proposal artifacts sequentially for `T=2,3,5,10,20`. Each
   horizon is a distinct target-specific tuning scope under this plan and the
   repository's scientific-default discipline; a lower-horizon setting is
   only a warm start.
3. Tune rank, basis, defensive mass, L1 weight, training budget, proposal
   chart, grid/CDF controls, chunking, and branch count on disjoint
   calibration/validation streams. The Zhao-Cui L1 policy applies.
4. Freeze all references, proposal maps, densities, ancestry, shifts, and
   branch-combination rule before untouched scoring.
5. At every staged horizon require finite proposal density, normalization,
   roundtrip, correct Jacobian, branch replay, retained-marginal identity,
   static workspace, and allocator gates.
6. Use proposal-quality vetoes based on predeclared ESS/weight-tail and
   independent-reference evidence. The exact floor must be reviewed and
   calibrated before claim data; do not automatically inherit the historical
   `ESS/N >= 0.5` threshold.

Exit: a repository-issued tuning artifact and immutable frozen branch chain
for each promoted horizon, or a precise proposal-quality/resource blocker.

### Phase 4: Staged Score Admission

1. Run `T=1`, then `T=2`, `T=3`, `T=5`, `T=10`, and `T=20`; never skip a failed
   horizon.
2. At each horizon evaluate the origin, sealed off-origin validation points,
   and independent untouched points inside the frozen domain.
3. Require finite value/score, increment-score additivity, same-scalar
   diagnostic derivative agreement, strict artifact reload, and deterministic
   replay.
4. Compare the new T1/T2 origin score descriptively against the existing
   material-FD values. This becomes a parity diagnostic only if scalar and
   branch identity prove both routes evaluate the identical finite program.
   Otherwise a discrepancy is an expected different-program diagnostic, not
   a failure and not a reason to fit the manual route to the old score.
5. Require an independent conditional/reference run at T1/T2 and at least one
   later horizon with reported MCSE, ESS, seeds, and uncertainty-aware score
   residuals. If full T20 reference cost is infeasible, record that as a limit
   and do not upgrade finite-program evidence to physical-likelihood evidence.

Exit: every staged horizon passes or the first failing horizon becomes the
terminal result.

### Phase 5: Untouched T20 Claim

Run at least two predeclared frozen branches, or justify statistically why one
branch is sufficient. The primary artifact must contain:

- the exact finite scalar definition and branch-combination rule;
- value, total score, and per-time increments for all three coordinates;
- per-branch values/scores and uncertainty diagnostics;
- ESS, weight entropy/tails, finite/support/measure checks;
- target, model, observation, source, branch, and tuning identities;
- dtype, TF32, XLA, GPU device, logical memory cap, allocator current/peak;
- environment/conda env, git commit, command, seeds, wall time, and paths;
- comparisons with independent reference and existing methods, classified by
  hard-veto, descriptive, and statistical evidence; and
- the explicit `extension_or_invention` route classification.

Exit status is one of:

- `PASS_T20_ZHAO_CUI_DERIVED_FROZEN_FINITE_SCORE`;
- `BLOCK_T20_PROPOSAL_QUALITY`;
- `BLOCK_T20_SCORE_IDENTITY`;
- `BLOCK_T20_TAIL_DERIVATIVE`;
- `BLOCK_T20_RESOURCE_OR_XLA`; or
- `BLOCK_T20_REFERENCE_INVALID`.

A passing artifact must also record
`physical_reference_status=SUPPORTED|LIMITED|NOT_RUN`. `LIMITED` or `NOT_RUN`
does not invalidate same-finite-program derivative correctness, but it forbids
any exact-physical-likelihood or scientific-accuracy claim.

### Phase 6: Closeout Without HMC

1. Run focused and relevant high-dimensional regressions.
2. Write a result note, run manifest, inference-status table, post-run
   red-team, and reset memo.
3. Update a new versioned leaderboard artifact only if the exact row label is
   `zhao_cui_derived_frozen_finite_score`; do not overwrite historical rows.
4. Stop. HMC requires a separate plan covering posterior target, prior/domain,
   deterministic branch policy, value/score smoothness, tuning, and sequential
   HMC diagnostics.

## Default And Assumption Audit

| Choice | Provenance/status | Failure mode | Earliest diagnostic | Promotion status |
|---|---|---|---|---|
| Frozen proposal branch | Simplest route whose proposal derivative is exactly zero for its finite scalar | High variance or poor off-origin support | T1/T2 domain and ESS screens | Hypothesis |
| Manual score recursion | Project derivation of the exact frozen scalar | Missing carry or normalization term | Tiny autodiff/FD plus mutant rejection | Candidate authority after pass |
| Existing T1/T2 TT parents | Passed bounded artifacts | Local fits do not extrapolate to later horizons or theta | Staged horizon/domain screens | Warm starts only |
| Exact interval-mass sampler | Passed T1 finite-law evidence | Rare prefix/tail issue at larger scope | Per-horizon normalization, roundtrip, and tail screens | Baseline sampler |
| Nested box half-widths `{0.03,0.10,0.25,0.50}` | Historical `0.5` diagnostic box plus smaller predeclared calibration ladder | Proposal collapse/nonfinite dynamics at corners; arbitrary prior implication | Phase 1 domain calibration and untouched interior checks | Hypothesis, not HMC default |
| `N=1008` | Historical leaderboard scope | High MC error or weight degeneracy | Sample/branch growth diagnostic with uncertainty | Baseline claim scope, not automatically sufficient |
| Initial score tolerance | Five-digit intent plus FP64 diagnostic hypothesis | Near-zero blindness or false rejection | Step-halving and omitted-term mutants | Calibration hypothesis |
| Rank/basis/ALS/Adam/L1 | Target- and scope-dependent | Underfit, overfit, or invalid transfer | Disjoint per-horizon tuning | No inherited default |
| 6,144 MiB cap | Existing reviewed shared-GPU policy | Candidate cannot fit | Compile/allocator preflight | Fixed resource limit |
| FP64 reference, FP32/TF32 claim | Existing backend ladder | Claim score loses material digits | Same-branch backend parity | Reference and production-target roles respectively |
| Two frozen branches | Minimal stochastic replication hypothesis | Understates branch uncertainty | Branch-growth diagnostic | Must be justified or increased |

## Skeptical Pre-Execution Audit

Verdict: `PASS_FOR_REVIEW; DO_NOT_EXECUTE_BEFORE_MATERIAL_FINDINGS_ARE_RESOLVED`.

| Required audit | Finding and repair |
|---|---|
| Wrong baseline | The active T1/T2 FD issuers answer local replay mechanics, not full score completion. They remain regression evidence only. |
| Proxy promoted | Training loss, density residual, step halving, and high ESS cannot independently promote the score. The primary object is the same frozen finite scalar and its manual total derivative. |
| Missing stop conditions | Mathematical measure, tail derivative, score identity, proposal quality, reference validity, XLA, memory, and budget vetoes are explicit. |
| Unfair comparison | Comparators must share observation/target conventions. No ranking follows without uncertainty support. |
| Hidden assumptions | Parameter domain, branch count, particle count, tolerance, rank, basis, L1, proposal chart, and tail algebra are hypotheses with early diagnostics. |
| Stale context | The historical APF path failed proposal quality; the newer exact-law retained sampler and T1/T2 value artifacts change feasibility but do not prove T20. |
| Environment mismatch | Reference is CPU-hidden FP64; claim path is trusted GPU/XLA with the logical-device cap and recorded allocator evidence. |
| Non-answering artifact | A T2 origin score cannot close a T20 score cell. Phase 5 requires T20 value, score, increments, uncertainty, and full identity. |
| False source claim | Paper Section 6.3 and author `d=0` explicitly block a source-faithful Austria parameter-score label. |
| Derivative-of-training detour | XLA higher-order optimizer autodiff is unnecessary for the selected scalar. Removing it closes the engineering problem without changing the claim into raw-core derivative correctness. |

## Pre-Mortem

- The manual score could match FD because both omit the same proposal term.
  Direct scalar assembly, proposal-independence assertions, and deliberate
  omitted-term mutants distinguish this.
- A fixed branch could be correct but useless away from the origin. Domain
  calibration, untouched off-origin points, and proposal-quality vetoes expose
  this before T20.
- A T20 run could return finite values while one branch dominates. Per-branch
  likelihoods, ESS, entropy, tails, and branch-growth uncertainty are required.
- A signed-log value repair could silently erase a nonzero score. Phase 1
  derives the derivative limit or blocks before horizon expansion.
- A Python runner could report XLA while performing numerical loops on the
  host. Concrete-graph inspection and forbidden-op tests veto this.
- A pass could be mislabeled source-faithful. Identity/status tests bind the
  extension classification.

## Compute And Attempt Budget

The campaign is authorized only after review and a plain-language execution
request. Proposed total budget:

| Work | Attempt/time limit |
|---|---|
| Phase 0-2 implementation and focused tests | 6 focused launches; 4 engineer-hours wall-clock |
| Phase 3 tuning at T1/T2/T3 | At most 8 candidate launches total; 4 GPU-hours |
| Phase 3 tuning at T5/T10/T20 | At most 12 candidate launches total; 10 GPU-hours |
| Phase 4 staged admissions | At most 2 claim-shaped launches per horizon; 8 GPU-hours total |
| Phase 5 untouched T20 | 1 primary launch plus 1 infrastructure-only retry; 4 GPU-hours |
| Independent references | 6 CPU-hours or 2 GPU-hours, whichever is used |

Versioned output root:

```text
docs/plans/artifacts/zhao-cui-austria-sir-score-completion-20260802/
```

Every retry uses a fresh directory and records failure classification, repair,
wall time, and remaining budget. A method/tolerance/domain/target expansion
requires a revised plan; localized infrastructure repair does not.

## Command And Environment Contract

Phase 0 must verify and seal the current baseline environment
`/home/chakwong/anaconda3/envs/tf-gpu`, including TensorFlow/TFP/CUDA versions.
If it is unsuitable, changing environments is an explicit environment decision
and package mutation remains outside this plan. Future implementation must
provide these stable runner surfaces before any serious launch:

```text
scripts/run_zhao_cui_austria_sir_frozen_score_preflight.py
scripts/run_zhao_cui_austria_sir_frozen_score_tuning.py
scripts/run_zhao_cui_austria_sir_frozen_score_claim.py
```

Reference/focused command shape:

```bash
CUDA_VISIBLE_DEVICES=-1 /home/chakwong/anaconda3/bin/conda run -p /home/chakwong/anaconda3/envs/tf-gpu \
  python -m pytest -q <focused-test-paths>
```

Trusted GPU preflight shape:

```bash
/home/chakwong/anaconda3/bin/conda run -p /home/chakwong/anaconda3/envs/tf-gpu \
  python scripts/run_zhao_cui_austria_sir_frozen_score_preflight.py \
  --horizon <1|2|3|5|10|20> \
  --output-dir docs/plans/artifacts/zhao-cui-austria-sir-score-completion-20260802/<fresh-attempt>/
```

Scope-specific tuning shape:

```bash
/home/chakwong/anaconda3/bin/conda run -p /home/chakwong/anaconda3/envs/tf-gpu \
  python scripts/run_zhao_cui_austria_sir_frozen_score_tuning.py \
  --horizon <2|3|5|10|20> \
  --scope-config <reviewed-scope-json> \
  --output-dir docs/plans/artifacts/zhao-cui-austria-sir-score-completion-20260802/<fresh-attempt>/
```

Untouched claim shape:

```bash
/home/chakwong/anaconda3/bin/conda run -p /home/chakwong/anaconda3/envs/tf-gpu \
  python scripts/run_zhao_cui_austria_sir_frozen_score_claim.py \
  --horizon 20 \
  --tuning-artifact <strictly-reloaded-t20-tuning-artifact> \
  --branch-manifest <sealed-branch-manifest> \
  --output-dir docs/plans/artifacts/zhao-cui-austria-sir-score-completion-20260802/<fresh-claim>/
```

All GPU/CUDA commands run with trusted/escalated device access. Each GPU runner
must configure the 6,144 MiB logical-device limit before initialization,
reject a missing GPU, and record the command exactly. CPU-only commands hide
CUDA before TensorFlow import and label the artifact as reference/testing only.

## Required Result Tables

The terminal result must include:

1. decision table: decision, primary criterion, veto status, uncertainty, next
   action, and nonclaims;
2. inference-status table: hard vetoes, statistically supported ranking,
   descriptive-only differences, default readiness, and next evidence;
3. engineering/numerical/scientific ledgers kept separate;
4. per-horizon value/score/additivity/proposal/tail/backend results;
5. per-branch and cross-branch uncertainty evidence; and
6. post-run red team: strongest alternative explanation, overturning evidence,
   and weakest evidence.

## Definition Of Done

The Austria Zhao-Cui-derived score computation is complete for this finite
program only when all conditions hold:

1. `T=20`, `J=9`, state dimension 18, observation dimension 9, and the exact
   three-parameter convention are sealed.
2. A frozen, correctly scored proposal branch chain exists through T20 without
   the historical full retained tensor-product grid.
3. The runtime value is one explicit finite scalar and the runtime score is
   its manual total derivative.
4. No runtime FD, autodiff, NumPy numerical path, Python numerical loop, host
   callback, or T2 radial projection contributes to the score.
5. Origin and off-origin same-scalar checks pass over the sealed validation
   domain, including step-halving diagnostics and omitted-term mutants.
6. Proposal, ancestor, Jacobian, normalizer, previous-weight carry, shift, and
   branch-combination derivatives are included or proved zero by literal
   frozen identity.
7. Stable tail value-and-score handling is proved for every encountered row.
8. Every staged horizon passes before T20.
9. The untouched GPU/XLA T20 artifact passes strict reload and records complete
   manifest and uncertainty evidence.
10. The result is labeled `extension_or_invention` and makes the nonclaims
    above.

Passing these items closes the **finite-program score computation**. It does
not close source-faithful Austria parameter inference, exact physical
likelihood, HMC, posterior correctness, or production readiness.
