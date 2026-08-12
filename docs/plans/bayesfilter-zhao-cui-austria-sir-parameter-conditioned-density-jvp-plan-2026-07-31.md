# Zhao-Cui Austria SIR Parameter-Conditioned Density/JVP Plan

Date: 2026-07-31

Status: `ACTIVE_REPLACEMENT_PLAN_AUDITED_BEFORE_EXECUTION`

This plan replaces the score direction in
`bayesfilter-zhao-cui-austria-sir-lane-b-t1-score-plan-2026-07-31.md`.
The old tangent child and its artifacts remain historical diagnostics. They are
not score authorities and must not be recalibrated or admitted by weakening the
failed untouched threshold.

The admitted fixed values remain closed evidence:

| Scope | Identity | Value |
|---|---|---:|
| T1 | `e4b56526205eb50c3d2aa3b8a8ce6ce27539aa5ab50ad286380136db28ed2b59` | `-31.1290512231882` |
| T2 | `f51bb12bb6ab1a16cd843b350bb53a69cd449d602007278b8c5ef306a82e9f5e` | increment `-35.154752282413156`; cumulative `-66.28380350560136` |

No phase below may reopen or replace these values without new contradictory
evidence.

## Decision

Use an **offline centered residual TT family trained at nonzero symmetric
parameter points with an absolute-scale objective**. Do not revive author
TT-cross/ALS, replace the working Adam training family, or train an independent
origin tangent.

The alternative considered was to replay a deterministic fixed trainer and
propagate its complete JVP. That is the clean construction used by the LGSSM
moment teacher, but it is not selected here because the admitted Austria parent
was produced by the repository Adam training-base route, not by the book's
fixed-ALS map. Replacing it with ALS changes the parent. Replaying optimizer
training at every runtime parameter is also not the admitted fixed finite
program, and exact reproduction of the selected origin artifact is not
established. The centered residual construction gives exact origin equality
algebraically while defining the missing off-origin map explicitly.

This decision is an `extension_or_invention`. It follows the fixed-branch
same-scalar differentiation discipline in the BayesFilter book, but Zhao and
Cui do not derive this outer-parameter likelihood score.

## Research Intent Ledger

| Field | Definition |
|---|---|
| Main question | Can a parameter-conditioned extension of the admitted Austria Lane-B squared-TT values define a unique finite value map and analytical total score while preserving the fixed parent exactly and remaining memory bounded through time? |
| Candidate mechanism | A parent amplitude plus a finite sum of centered parameter features times residual state TTs, trained on absolute target density at nonzero symmetric theta points. |
| Expected failure mode | The fixed parent shape is too inaccurate for a low-rank residual family to recover off-origin density, normalizer, origin score, or recursively carried prefix score. |
| Primary promotion criterion | Exact origin parent equality plus an untouched origin score agreement with an independent Fisher-identity estimate under its predeclared MCSE tolerance. |
| Promotion veto | Off-origin heldout absolute-density or normalizer failure; origin point-score failure; origin prefix-score failure; fresh-reload mismatch; GPU/XLA mismatch; or memory-cap breach. |
| Continuation veto | Target/measure inconsistency, failure of exact algebraic origin preservation, inability to represent the cross-component marginal without exponential storage, corrupted parent evidence, or invalid independent score authority. |
| Repair trigger | A candidate rank, feature, loss-weight, optimizer, radius, or budget failure with valid harness and target triggers the next predeclared candidate or a revised target-specific training protocol. |
| Explanatory diagnostics | Training loss, shape residual, per-coordinate score residual, Gram conditioning, feature contribution, runtime, peak allocation, and descriptive comparisons with GenUT/SGQF/UKF. |
| Must not be concluded | A mechanics or T1 pass does not prove T2/T20, exact nonlinear likelihood, whole-route source faithfulness, HMC readiness, posterior correctness, superiority, or production readiness. |

## Source And Derivation Boundary

The required source anchors were rechecked before this plan was written.

| Operation | Classification | Anchor and interpretation |
|---|---|---|
| Adjacent target `previous marginal * transition * likelihood` | `source_faithful` operation | Zhao-Cui Eqs. (9)-(12), Eq. (15), Algorithm 2(a); author `models/full_sol.m:72-80,132-135`. |
| Squared-TT nonnegative representation and marginalization | `source_faithful` operation | Zhao-Cui Algorithm 2(b-c), Proposition 2; author `@TTSIRT/marginalise.m:25-85`. |
| Increment `log(sirt.z)-const` | `source_faithful` operation | Author `models/full_sol.m:84-124`. |
| Fixed ranks, bases, points, schedules, streams, and branches | `fixed_hmc_adaptation` | Freezes source-route choices to define one finite program; it does not authorize HMC. |
| Existing Lane-B Adam training-base parent | `extension_or_invention` | Repository implementation and admitted artifacts; it is not author TT-cross/ALS. |
| External theta, centered residual TTs, absolute-scale training, and manual JVP | `extension_or_invention` | Project derivation below. Theta is conditioned on and never integrated. |

The inspected Zhao-Cui paper provides joint parameter-state posterior
approximation, not the external-theta observed-data derivative claimed here.
The assembled parameter-score route must not be called source-faithful Zhao-Cui.

Literature-audit scope is narrow because the plan does not make a novelty or
survey-completeness claim. The local published JMLR PDF/text and pinned author
code are sufficient for the operation classifications above. Citation counts,
venue ranking, forward snowballing, and broad omission review are not needed to
decide this local implementation boundary and are not being claimed.

## Claimed Mathematical Object

Let the fixed parent amplitude at time `t` be `h_t^0(r)` and let its exact
configured density be

\[
  \rho_t^0(r) = h_t^0(r)^2 + \tau_t q_{0,t}(r).
\]

The parent measure, frame, basis, defensive density, `tau`, shift, coordinate
order, normalizer rule, and retained-marginal rule remain unchanged.

Choose centered scalar parameter features `psi_k(theta)` satisfying
`psi_k(0)=0`. Define independent residual state TTs `H_{t,k}` and

\[
  h_t(\theta,r)
  = h_t^0(r) + \sum_{k=1}^{K}\psi_k(\theta)H_{t,k}(r),
\]

\[
  \rho_t(\theta,r)
  = h_t(\theta,r)^2 + \tau_t q_{0,t}(r).
\]

The initial feature ladder is:

1. linear features `(theta_1, theta_2, theta_3)`;
2. only if heldout off-origin curvature requires them, centered pure-quadratic
   and pairwise features;
3. no neural feature map and no parameter TT integration coordinate in this
   campaign.

Feature choice is a hypothesis selected on validation data, not an inherited
default. Linear features are the smallest mechanism that identifies the origin
score. Quadratic/cross features are capacity repairs, not automatic promotion.

Define `H_{t,0}=h_t^0`, `psi_0=1`, and the exact component Gram matrix

\[
  G_{t,ij}=\int H_{t,i}(r)H_{t,j}(r)\,d\nu_t(r).
\]

Then the normalizer and its analytical derivative are

\[
  Z_t(\theta)
  = \psi(\theta)^\top G_t\psi(\theta)
    +\tau_t\int q_{0,t}\,d\nu_t,
\]

\[
  D_a Z_t(\theta)
  = 2\,[D_a\psi(\theta)]^\top G_t\psi(\theta).
\]

At a point `r`,

\[
  D_a\log\rho_t
  =\frac{2h_tD_a h_t}{\rho_t},
  \qquad
  D_a h_t=\sum_k D_a\psi_k H_{t,k}.
\]

The increment and score are

\[
  \ell_t(\theta)=\log Z_t(\theta)-c_t,
  \qquad
  D_a\ell_t(\theta)=\frac{D_a Z_t(\theta)}{Z_t(\theta)},
\]

with the admitted fixed shifts held theta-independent. A parameter-dependent
calibration or post-fit gauge correction is forbidden. If a future design makes
the shift or defensive term parameter dependent, that defines a new reviewed
finite program and must include its derivative.

Because every non-parent feature vanishes at zero,

\[
  h_t(0,r)=h_t^0(r),\quad
  \rho_t(0,r)=\rho_t^0(r),\quad
  Z_t(0)=Z_t^0,
\]

exactly, independent of the fitted residual TTs. This is stronger than a
numerical origin penalty.

### Retained Marginal And Time Recursion

For retained coordinates `x_t` and marginalized coordinates `x_{t-1}`, define
cross-component contractions

\[
  A_{t,ij}(x_t)
  =\int H_{t,i}(x_t,x_{t-1})H_{t,j}(x_t,x_{t-1})
    \,d\nu(x_{t-1}).
\]

The unnormalized retained numerator is

\[
  a_t(\theta,x_t)
  =\sum_{i,j}\psi_i(\theta)\psi_j(\theta)A_{t,ij}(x_t)
   +\tau_t a_{0,t}(x_t),
\]

and the carried filter and score are

\[
  \widehat p_t=\frac{a_t}{Z_t},\qquad
  D_a\log\widehat p_t
  =\frac{D_a a_t}{a_t}-\frac{D_a Z_t}{Z_t}.
\]

At time `t+1`, the target is

\[
  Q_{t+1,\theta}(x_{t+1},x_t)
  =\widehat p_t(\theta,x_t)
   f_\theta(x_{t+1}\mid x_t)
   g_\theta(y_{t+1}\mid x_{t+1}).
\]

Its total target score includes all three terms, including the carried prefix
score. A correct current-step scalar with a missing prefix score is a partial
derivative and is wrong relative to the total-score claim.

## Why The Failed Child Is Not This Program

The historical `LaneBParameterChild` perturbs every parent core linearly and was
trained only as an origin tangent. It proves manual differentiation of its own
chosen child, but the fixed parent does not identify that child's tangent. Its
training-only amplitude calibration can change the scalar score without fixing
the pointwise or retained-prefix derivative. The untouched score failure is
therefore a candidate rejection, not a threshold problem.

This plan instead defines `theta -> rho_t(theta)` at nonzero theta, trains its
absolute scale, and validates the same family away from the origin. The old
child remains available only for regression and negative-evidence tests.

## T1 Absolute-Scale Training Definition

Use the fixed admitted T1 frame and shift `c_1^0`. For a physical adjacent-state
point `z=(z_1,z_0)`, define the target density relative to the parent's reference
measure,

\[
  Q_{1,\theta}^{\rm ref}(r)
  =e^{c_1^0}
    p_\theta(z_0)f_\theta(z_1\mid z_0)g_\theta(y_1\mid z_1)
    |J(r)|/w_{\rm ref}(r).
\]

For each training theta, generate common-random-number rows from
`p_theta(z_0) f_theta(z_1|z_0)`. The absolute unnormalized density loss is the
generalized KL/I-divergence up to a target-only constant,

\[
  \mathcal J_{\rm abs}(\theta)
  =Z_1^{\rm child}(\theta)
   -\int Q_{1,\theta}^{\rm ref}(r)
          \log\rho_1^{\rm child}(\theta,r)\,d\nu(r).
\]

The first term is an exact TT contraction. The second is estimated by
importance sampling; under the stated proposal its absolute weight is
`exp(c_1^0) g_theta(y_1|z_1)`. No normalization of those weights is allowed in
the absolute loss. Normalized weights may be used only for explicitly labeled
shape and score diagnostics.

The candidate objective is

\[
  \sum_{\theta_j}\mathcal J_{\rm abs}(\theta_j)
  +\gamma\sum_a
    \mathbb E_{Q_{1,0}/Z_{1,0}}
    \left[
      \frac{D_a\log\rho_1^{\rm child}(0,R)
             -D_a\log Q_{1,0}^{\rm ref}(R)}{s_a}
    \right]^2
  +\lambda_1\|H\|_1+\lambda_2\|H\|_2^2.
\]

The derivative-matching term is auxiliary. Its weight `gamma` must include a
zero comparator and be tuned on validation data. It cannot replace nonzero
theta absolute-density training or the untouched downstream score gate.

Symmetric theta radii, residual ranks, basis capacity, learning rate, L1/L2,
batch size, steps, and `gamma` are target-specific hypotheses. The first cheap
diagnostic uses a small symmetric radius ladder to expose under-resolution or
nonlinearity. A serious candidate requires disjoint training, validation, and
untouched streams and a reviewed finite budget before launch.

## Evidence Contract

| Field | Requirement |
|---|---|
| Scientific question | Does the centered residual family define a stable parameter-conditioned approximation whose origin score is supported by independent evidence? |
| Exact baseline | Admitted Lane-B T1 parent above; later the admitted T2 parent above. |
| Primary T1 criterion | Untouched origin child score versus independent Fisher estimate, coordinatewise within `absolute_floor + sigma * MCSE`; thresholds frozen before the untouched run. |
| Promotion vetoes | Any failed origin identity; nonzero-theta absolute-normalizer failure; heldout absolute log-density failure; origin point-score failure; origin retained-prefix score failure; nonfinite/ill-conditioned contraction; reload/XLA/memory failure. |
| Explanatory only | Training loss, normalized shape loss, raw tangent norms, runtime, and comparisons against GenUT/SGQF/UKF. |
| Nonclaim | Passing establishes the score of the declared finite child at T1 and evidence of local target fidelity; it does not establish exact-model or later-horizon correctness. |
| Artifact | Fresh versioned directories under `docs/plans/artifacts/zhao-cui-austria-sir-parameter-density-jvp-20260731/` plus a result note and run manifest. |

The independent Fisher estimate at T1 is valid because the proposal is
`p_0(z_0)f_0(z_1|z_0)` and the observation likelihood supplies the normalized
weights. It is an authority for the exact T1 model score with reported Monte
Carlo uncertainty, not for T2's recursively approximate child. T2 requires a
separate retained-prefix authority.

The T1 point-score authority at an adjacent-state point is the analytical
complete-data score minus the independent Fisher normalizer score. The T1
retained-prefix authority at a fixed `z_1` is

\[
  D_a\log p_0(z_1\mid y_1)
  =D_a\log g_0(y_1\mid z_1)
   +\mathbb E_0[
      D_a\log p_0(Z_0)+D_a\log f_0(z_1\mid Z_0)
      \mid z_1]
   -D_a\log p_0(y_1).
\]

Phase 2 must implement this as an independent conditional-ratio diagnostic with
reported ESS and MCSE (or a stronger independently checked evaluator). Prefix
points whose conditional estimator fails its predeclared ESS/MCSE validity gate
cannot be silently dropped or called passed; failure blocks the prefix-score
authority and therefore Phase 3 promotion.

## Default And Assumption Audit

| Choice | Provenance/status | Failure mode | Earliest diagnostic |
|---|---|---|---|
| Admitted parent cores/frame/shift/tau | Closed local evidence; baseline | Parent drift makes every comparison meaningless | Bitwise core hashes and exact origin contractions before training |
| Additive residual amplitudes | Project derivation; selected mechanism | Insufficient capacity or destructive cancellation off origin | Small T1 symmetric-theta fit and minimum rho/Gram checks |
| Linear theta features first | Minimal hypothesis, not default | Curvature aliases into the origin tangent | Positive/negative radii and heldout radius-transfer residuals |
| Quadratic/cross features | Capacity repair only | Overfit and `K^2` contraction cost | Validation gain plus allocation projection before promotion |
| Fixed parent frame | Baseline preservation | Off-origin mass moves into poorly resolved chart regions | Reference-coordinate tail and heldout density diagnostics by radius/sign |
| Fixed shift and tau | Baseline-preserving reviewed choice | Poor conditioning away from zero | Target mass and rho dynamic-range diagnostics; no posthoc calibration |
| Adam/L1 training family | Existing Lane-B family and lane policy | Optimizer/capacity failure mistaken for method failure | Tiny multi-theta overfit test, zero-L1 comparator, loss scaling check |
| Absolute I-divergence | Project derivation; selected objective | High-variance absolute importance weights | ESS/MCSE and duplicate-stream normalizer estimates before training ladder |
| Derivative loss weight | Tuned hypothesis including zero | Proxy fit passes while downstream score fails | Independent untouched score remains primary criterion |
| T1 first | Smallest discriminating scope | T1 success may not transfer recursively | Explicit nonclaim; T2 prefix gate is separate |
| FP64 reference, then GPU/XLA | Repository numerical policy | Backend mismatch | Eager/XLA parity before serious candidate |

## Memory And Complexity Contract

Do not materialize a single block-sum TT with rank `(K+1)R`, a state grid, a
theta-state tensor product, or full time-history tangents.

Store the parent TT and `K` independent residual TTs. Evaluate amplitudes in a
streaming loop and normalizers through pairwise component contractions. For
maximum component rank `R`, basis size `p`, dimension `D`, and batch `B`:

\[
  \text{stored cores}=O((K+1)DR^2p),
\]

\[
  \text{point workspace}=O(BR+BR^2),
  \qquad
  \text{normalizer-pair workspace}=O(R^4),
  \qquad
  \text{prefix-pair workspace}=O(BR^4),
\]

with `O(K^2D)` pair operations executed serially or in bounded chunks. Retained
marginal cross contractions have the same `K^2` component structure. Only the
current time-step workspace and compact carried representation are live during
training/evaluation.

Those are inference/contraction bounds. A reverse-mode training step also owns
optimizer slots and either retained activations or recomputation checkpoints.
Without checkpointing, a conservative activation bound is
`O((K+1)DBR^2)` in addition to `O((K+1)DR^2p)` cores and optimizer state.
Phase 2 must implement and measure a batch-native bounded strategy (checkpointed
component evaluation or an equivalently bounded custom gradient) before any
serious training launch. It must not infer training feasibility from the
smaller Phase 1 inference peak.

Every candidate must estimate bytes before allocation and record TensorFlow
allocator peak bytes. Serious GPU runs use memory growth configured and verified
before initialization. The initial hard cap remains 6 GiB. A feature expansion
that projects above the cap is rejected before launch.

## Execution Phases

### Phase 1: Centered Mechanics

Implement a new module rather than mutating the historical tangent child.

Required mechanics:

1. immutable parent plus residual component TTs;
2. centered linear and optional polynomial feature maps with analytical JVP;
3. point amplitude/density value and score;
4. exact Gram normalizer and score using cross-TT mass contractions;
5. retained-prefix cross contractions and quotient score;
6. repository-issued identity, serialization, reload, and tamper rejection;
7. static allocation estimate and XLA-compatible TensorFlow kernels.

Exit: exact T1 and T2 origin value/core/prefix equality, manual-versus-diagnostic
autodiff tie-out on synthetic residuals, and storage scaling tests. This is an
engineering gate only.

### Phase 2: T1 Target And Absolute-Loss Harness

Add batch-native parameterized T1 target evaluation and common-random-number
proposal generation. Verify the absolute importance identity and analytical
complete-data score against diagnostic autodiff on tiny batches. Implement the
absolute loss and its batch-native GPU/XLA training step.

Exit: tiny synthetic recovery test where a known residual family is recovered,
plus a tiny real-target overfit diagnostic showing that loss and scale respond
to residual parameters. Do not use these as promotion evidence.

### Phase 3: T1 Target-Specific Training Protocol

Write the serious experiment subplan before launching. It must freeze the
radius/feature/rank/loss/optimizer/L1 ladder, streams, attempts, total compute,
promotion thresholds, versioned outputs, and stop conditions. Use validation to
select one candidate, then consume untouched data once.

Exit: either `PASS_T1_PARAMETER_DENSITY_SCORE` under the full evidence contract
or a classified candidate/training/direction blocker. No gauge calibration.

### Phase 4: T2 Recursive Closure

Only after Phase 3 passes, attach the centered family to the admitted T2 parent.
Construct the T2 target from the parameterized T1 retained marginal and its
score. Validate the carried-prefix score independently before fitting T2, then
repeat the absolute-density and untouched increment/cumulative-score gates.

Exit: exact admitted T1:T2 value at zero and a supported cumulative analytical
score. A correct T1 scalar score without the retained-prefix score does not open
this phase.

### Phase 5: Streaming Horizon Ladder

Build fixed parents and centered residuals sequentially for `T=3,5,10,20`, one
time step at a time. Each step has its own retained-prefix, absolute-normalizer,
score, identity, reload, XLA, and memory gates. Stop at the first invalid step;
do not extrapolate a shorter-horizon score.

### Phase 6: Terminal Comparison

Compare the final declared Zhao-Cui finite value/score with GenUT, SGQF, and UKF
where a same-target executed result exists. Report continuous differences as
descriptive unless the comparison has predeclared uncertainty support. A
same-scalar derivative tie-out establishes internal correctness, not superiority
or equality to the exact observed-data score.

HMC remains outside this plan until the value and score program passes through
the required horizon.

## Skeptical Pre-Execution Audit

The plan was audited against the repository policy before Phase 1 execution.

| Audit risk | Finding and repair |
|---|---|
| Wrong baseline | Repaired: the admitted Lane-B T1/T2 artifacts, not P88, APF, source replica, retained grid, or ALS, are the value parents. |
| Proxy promoted to criterion | Repaired: training/shape losses are explanatory or veto diagnostics; untouched downstream score is primary. |
| Missing scale definition | Repaired: nonzero-theta absolute I-divergence uses unnormalized importance weights; posthoc gauge calibration is forbidden. |
| Origin tangent not identified | Repaired: a complete nonzero-theta family is defined and trained; origin equality is algebraic. |
| Hidden recursive partial derivative | Repaired: T2 cannot open without retained-prefix value and score validation. |
| Trainer drift | Repaired: author/fixed ALS is not substituted for the admitted Adam parent; the residual trainer remains the local training family. |
| Environment mismatch | Repaired: TensorFlow/TFP only, batch-native target, XLA default, GPU memory growth, CPU reference labeled diagnostic. |
| Memory blow-up | Repaired: additive components and `K^2` streaming contractions replace a materialized block-rank sum or theta-state grid. |
| Unfair comparator | Repaired: GenUT/SGQF/UKF are terminal same-target comparators, not score authorities or tuning inputs. |
| Missing stop conditions | Repaired: origin, target/measure, prefix, numerical, identity, backend, and memory continuation vetoes are explicit. |
| Misleading successful command | Repaired: Phase 1 mechanics cannot promote the score; Phase 2 overfit cannot promote the model; only the untouched contract can. |

Verdict: the plan is adequate to begin Phase 1. It is not yet authorization for
a serious T1 training campaign; Phase 3 requires its own bounded experiment
subplan after the mechanics and loss harness are verified.
