# Zhao-Cui Austria SIR Parameter Density T1 Campaign Plan

Date: 2026-08-01

Status: `CLOSED_REPRESENTATION_BRANCH_TARGET_REPAIR_REQUIRED`

Parent plan:
`docs/plans/bayesfilter-zhao-cui-austria-sir-parameter-conditioned-density-jvp-plan-2026-07-31.md`.

Phase 2 result:
`docs/plans/bayesfilter-zhao-cui-austria-sir-parameter-density-jvp-phase2-result-2026-08-01.md`.

## Research Intent Ledger

| Field | Frozen definition |
|---|---|
| Main question | Can a low-rank, externally parameter-conditioned centered residual family preserve the admitted T1 parent exactly and produce an origin likelihood and retained-prefix score supported by fresh independent estimates? |
| Candidate mechanism | Immutable admitted parent plus linear centered theta features and independently stored fixed-rank residual TTs trained with absolute I-divergence; optional origin derivative loss. |
| Expected failure mode | Off-origin target changes are outside the fixed frame/basis or residual rank, absolute weights are too noisy, or the auxiliary point-score fit does not recover the retained-prefix derivative. |
| Promotion criterion | One validation-selected frozen child passes the fresh untouched origin Fisher score coordinatewise within `3*MCSE + 1e-5`, exact origin value, and all declared vetoes. |
| Promotion veto | Failed off-origin mass/shape, origin point score, retained-prefix score, ESS/MCSE validity, identity/reload/XLA parity, finite/conditioning, or 6 GiB memory gate. |
| Continuation veto | Invalid target/measure, parent identity drift, origin centering failure, invalid independent authority, corrupted artifacts, or inability to run rank 4 batch 64 under 6 GiB. |
| Repair trigger | A valid candidate failure triggers the next predeclared arm or feature-capacity repair. It does not reject the fixed values or the centered-density direction. |
| Explanatory only | Training loss, gradient norm, runtime, raw metric ordering, and historical GenUT/SGQF/UKF differences. |
| Must not be concluded | T1 passing does not establish exact-model likelihood, T2/T20, HMC, posterior correctness, source-faithful assembled score, superiority, or production readiness. |

## Fixed Baseline And Source Boundary

The only value baseline is parent identity
`e4b56526205eb50c3d2aa3b8a8ce6ce27539aa5ab50ad286380136db28ed2b59`
with T1 value `-31.1290512231882`.

The source anchors remain Zhao-Cui Eqs. (9)-(12), Eq. (15), Algorithm 2,
Proposition 2, author `models/full_sol.m:72-135`, and author
`@TTSIRT/marginalise.m:25-85`. They ground adjacent-target construction,
squared-TT density, marginalization, and `log(sirt.z)-const`. External theta,
centered residuals, Adam, the absolute loss, and this score assembly are
`extension_or_invention`. No result may call the assembled route source-faithful
Zhao-Cui.

## Evidence Contract

| Field | Requirement |
|---|---|
| Exact comparator | The immutable admitted rank-4 T1 parent evaluated against the same validation target rows. |
| Primary untouched criterion | `abs(child_score - Fisher_score) <= 3*Fisher_MCSE + 1e-5` in all three coordinates. |
| Origin identity | Child value equals `-31.1290512231882` within `2e-13`; parent identity and core hashes remain unchanged. |
| Off-origin mass veto | For every validation theta, absolute child/target log-mass difference is at most `3*target_log_mass_MCSE + 0.05`. The `0.05` term is a frozen approximation allowance, not Monte Carlo uncertainty. |
| Off-origin shape veto | Mean non-origin normalized log-density RMS is at most `0.95` times the immutable-parent baseline and no non-origin row exceeds `1.05` times its paired parent baseline. |
| Origin point-score veto | Every coordinate's normalized score-residual RMS is at most `0.90`; the parent-zero-score baseline is one by construction. |
| Retained-prefix veto | At each frozen prefix point and coordinate, `abs(child-authority) <= 3*authority_MCSE + 1e-5`; conditional ESS is at least half the requested rows and MCSE is below `(2.0,1.0,0.5)`. No point may be dropped. |
| Backend/identity veto | Fresh reload identity equals the saved identity; eager and XLA origin value, likelihood score, and prefix score differ by at most `3e-11`. |
| Resource veto | TensorFlow allocator peak is at most 6 GiB with verified memory growth. No block-sum TT, theta-state grid, row mapping, or scalar target fallback. |
| Artifact | Fresh versioned directories under `docs/plans/artifacts/zhao-cui-austria-sir-parameter-density-t1-20260801/`, plus selector and result notes. |

Training and validation scores are selection evidence only. The untouched
stream is consumed once after the selector freezes one child identity.

## Data, Graph, And Seed Contract

All arms use the same common-random-number streams for fair paired validation.
The untouched seeds are not used by training, pilot reporting, or selection.

| Role | Count | Seed(s) | Theta rows |
|---|---:|---|---|
| Capacity probe | 64 | initial `85101`, transition `85102` | origin plus `+/-0.01` on each axis |
| Training | 4,096 | initial `85201`, transition `85202` | origin plus `+/-r_train` on each axis |
| Validation | 8,192 | initial `85301`, transition `85302` | origin, `+/-0.01`, and `+/-0.03` on each axis |
| Validation prefix authority | 8,192 per point | `85401` | first frozen validation-origin prefix point |
| Untouched origin/Fisher | 65,536 | initial `85501`, transition `85502` | origin only |
| Untouched prefix authority | 32,768 per point | `85601` | first three untouched-origin prefix points, fixed before evaluation |

The training graph shape is fixed at seven theta rows and batch size 64 for
every arm. Validation uses thirteen theta rows. This prevents graph-shape and
compilation differences from masquerading as arm performance.

## Candidate Ladder

### 2026-08-01 Reset Before Execution

The earlier pre-execution verdict is superseded.  Two defects were found before
any claim run:

1. runtime `eigvalsh` basis construction differed between CPU and GPU and made
   pre-repair child identities and capacity artifacts ineligible; and
2. the residual initializer's `1e-3` first-core amplitude and nearly constant
   random TT structure have no target-specific justification and seed an origin
   score near zero, far below both the global Fisher and retained-prefix score
   scales.

The basis defect is repaired by
`setup_static_cpu_nodes_barycentric_weights_v1`, whose fixed-node hashes are
bound into child identity.  The initializer defect remains open.  Do not run
the Stage A table below until a training-only target-informed prefit has passed
focused mechanics and a fresh skeptical audit.  Validation and untouched rows
must remain inaccessible to initializer design and prefit.

### Training-Only Initializer Audit

The audit question is whether exact finite-child origin-score pretraining can
turn a connected fixed-rank random TT into a nonconstant score field before the
absolute-density campaign.  It optimizes the already defined normalized score
of the same finite child,

\[
 L_{\rm pre}=\sum_{k=1}^3
 E_{\widehat p_0}\left[
 \left\{s_{\rm child,k}(r)-s_{\rm target,k}(r)\right\}^2/
 \widehat\sigma_k^2\right],
\]

including the child's exact cross-mass normalizer derivative.  It is not a new
density, score estimator, or promotion criterion.

| Audit arm | Rank | First-core scale | Connected perturbation | LR | Full-batch steps |
|---|---:|---:|---:|---:|---:|
| `i01_r2_amp1_pert05_lr3e4` | 2 | 1.0 | 0.05 | `3e-4` | 32 |
| `i02_r4_amp1_pert05_lr3e4` | 4 | 1.0 | 0.05 | `3e-4` | 32 |

Both arms use only the 4,096 training-origin rows and residual seed `85701`.
The order-one scale is a hypothesis prompted by the observed order-one target
point-score RMS; `0.05` is a connected-variation hypothesis rather than a
default.  Rank 2 is the compact baseline and rank 4 is the parent-rank
hypothesis.  The learning rate is a warm start from the known-family mechanics
test, not target-specific evidence.

An audit arm is viable only if all values and gradients are finite, the final
exact score loss is at most `0.80` of its initial value, every coordinate's
normalized score-residual RMS is at most `0.90`, training importance ESS is at
least half the row count, the admitted origin value remains within `2e-13`, the
GPU allocator peak is at most 6 GiB, and its wall cap is 8 minutes.  Among
viable arms the smaller rank is preferred, then smaller final loss.  This rule
selects an initialization policy using training data only; it cannot select a
scientific candidate.

The result root is
`docs/plans/artifacts/zhao-cui-austria-sir-parameter-density-t1-20260801/initializer-audit-frozen-basis-v1/`.
At most two initial arms and two repair arms may run, with a 20-minute total
budget.  Failure with finite gradients is a repair trigger for a learning-rate
or step-count diagnostic.  Non-finite evaluation, invalid ESS, origin drift,
memory failure, or a basis/identity mismatch is a continuation veto.

Skeptical audit: the baseline and score target are the admitted finite child;
no validation, untouched, prefix, GenUT, SGQF, or UKF quantity is used; the
loss is an initializer diagnostic and cannot be promoted into the T1 claim;
rank, scale, perturbation, optimizer, steps, seeds, gates, artifact, and budget
are explicit; and the run distinguishes a vanishing initialization from rank
failure.  Verdict: execute these two bounded training-only arms, then refresh
Stage A from their result.

The two initial arms were both valid but rejected. Rank 2 reduced exact score
loss only from `2.9869` to `2.9493`; rank 4 reduced it from `3.0131` to
`2.9207`. Their final coordinatewise normalized RMS values remained near one,
and rank 4 child point-score standard deviations
`[0.150, 0.118, 0.085]` remained far below target RMS
`[73.43, 19.26, 0.785]`. ESS, finite, origin-value, memory, and wall gates
passed. This rejects more random-TT rank as the immediate repair; it does not
reject the centered finite density.

The remaining two predeclared repair arms solve a 180-column training-only
weighted ridge problem for additive residuals. For each parameter coordinate
the design is the exact finite-child origin-score operator

\[
 D(r)c = \frac{2h_0(r)H_c(r)}{h_0(r)^2+\tau}
 -\frac{2\langle h_0,H_c\rangle}{\langle h_0,h_0\rangle+\tau},
 \qquad H_c(r)=\sum_{a=1}^{36}\sum_{j=1}^{5}c_{aj}\phi_{aj}(r_a).
\]

An additional exact row constrains
`2<h0,H_c>/(<h0,h0>+tau)` to the training Fisher score, closing the normalized
point-score gauge. The additive function is encoded exactly as a rank-2 TT;
there is no retained grid, ALS, TT-cross, or validation use.

| Repair arm | Ridge fraction | Global-score row weight | Rank | Optimizer steps |
|---|---:|---:|---:|---:|
| `i03_add_ridge1e4_global1` | `1e-4` | 1 | 2 | 0 |
| `i04_add_ridge1e4_global10` | `1e-4` | 10 | 2 | 0 |

The ridge is a scale-relative numerical hypothesis that passed a 256-row
mechanics check; it is not a promoted default. The two global weights test the
only remaining balance question without changing rank or basis. A repair is
viable only if the earlier initializer gates pass and the maximum training
Fisher relative error, scaled by `max(abs(score),1)`, is at most `0.50`.
Before Stage A, the provisional selection rule was rejected as too weak: a
50-percent relative global-score gate is not aligned with the later Fisher
claim and would select weight 1 despite a third-coordinate training score of
about `-4.30` against `-4.92`. The refreshed artifact therefore also reports
the training Fisher MCSE and coordinatewise standardized residual
`abs(child-target)/(3*MCSE+1e-5)`. Among otherwise viable repairs, minimize the
maximum of (a) normalized point-score RMS divided by `0.90` and (b) this global
score standardized residual, then arm id. This is still training-only
initializer selection and cannot admit T1; it prevents a convenience relative
tolerance from overriding the actual score scale.

The MCSE-aware refresh selected `i04_add_ridge1e4_global10`: its maximum
global-score standardized residual was `1.6844`, versus `15.3018` for weight 1,
and its maximum point-RMS gate ratio was below `0.55`. The selected training
initializer uses ridge fraction `1e-4`, global-score weight 10, and rank 2.
Rank-4 Stage A arms embed the same rank-2 function and add two deterministic
complete channels with endpoint scale `1e-3`; the induced score-loss drift is
below `1e-3` in the focused test, and every added channel has nonzero first-order
gradient. This expansion is a trainable-capacity hypothesis, not a claim or
default beyond this T1 scope.

All Stage A arms use linear features `(theta_0,theta_1,theta_2)`, the selected
training-only initializer, 96 updates, batch size 64, L2 `1e-10`, gradient clip
100, and deterministic expansion seed `85701`. L1 is explicitly tuned as
required by lane policy.

| Arm | Rank | Train radius | Learning rate | L1 | Derivative weight | Purpose |
|---|---:|---:|---:|---:|---:|---|
| `a01_r2_rad01_lr3e4_l1_1e9_g01` | 2 | 0.01 | `3e-4` | `1e-9` | 0.1 | Compact selected-initializer baseline. |
| `a02_r4_rad01_lr3e4_l1_1e9_g01` | 4 | 0.01 | `3e-4` | `1e-9` | 0.1 | Connected rank-expansion candidate. |
| `a03_r4_rad03_lr3e4_l1_1e9_g01` | 4 | 0.03 | `3e-4` | `1e-9` | 0.1 | Radius-transfer hypothesis. |
| `a04_r4_rad01_lr1e4_l1_1e9_g01` | 4 | 0.01 | `1e-4` | `1e-9` | 0.1 | Optimizer-rate repair. |
| `a05_r4_rad01_lr3e4_l1_0_g01` | 4 | 0.01 | `3e-4` | `0` | 0.1 | Required zero-L1 comparator. |
| `a06_r4_rad01_lr3e4_l1_1e8_g01` | 4 | 0.01 | `3e-4` | `1e-8` | Stronger L1 hypothesis. |
| `a07_r4_rad01_lr3e4_l1_1e9_g0` | 4 | 0.01 | `3e-4` | `1e-9` | Required zero-derivative comparator. |

The selector first rejects any arm failing a veto. Among viable arms it
minimizes, in order: maximum standardized validation likelihood/prefix score
residual, mean paired shape ratio, maximum mass z-residual, rank, then arm id.
Observed continuous differences are descriptive; this deterministic selection
rule nominates one arm and does not establish statistical superiority.

### Predeclared Capacity Repair

If no linear arm is viable and the harness/authority/resource gates remain
valid, inspect failure classification:

- curvature/radius-transfer failure with adequate rank opens pure-quadratic
  features at rank 4, using the same axis theta rows;
- rank failure with acceptable radius transfer opens rank 6 only after a fresh
  one-step 6 GiB capacity probe;
- interaction features require a new corner-theta training/validation design
  and a refreshed subplan because the Stage A axis rows do not identify them;
- ESS/MCSE failure triggers more independent authority rows within the total
  budget, not relaxed tolerances;
- point-score pass with prefix failure rejects the candidate and does not
  permit gauge calibration.

No repair may tune on untouched data.

### Refreshed Stage-A Audit

The selected initializer and rank-expansion mechanics pass focused tests, the
full CPU boundary is `27 passed, 4 deselected`, and the fresh GPU/XLA rank-4
batch-64 capacity artifact passes with allocator peak `17,635,072` bytes. The
capacity artifact is
`docs/plans/artifacts/zhao-cui-austria-sir-parameter-density-t1-20260801/capacity-frozen-basis-additive-v1/result.json`.

The Stage-A evidence contract remains unchanged: validation may select or veto,
untouched data remain unopened, and only a frozen/reloaded child passing the
later untouched Fisher and prefix gates can open T2. The first refreshed arm
`a01` is the smallest discriminator for whether absolute-density training from
the selected score-informed rank-2 initializer preserves score quality while
improving off-origin density. Review its failure classification before running
the remaining arms; do not execute a ladder whose mechanism has already been
invalidated. Verdict: run refreshed `a01` in a fresh output root.

`a01` is valid negative evidence. Fresh reload, XLA parity (`5.68e-14`), finite
host summaries, prefix-authority validity, origin value, memory (`833,642,496`
bytes), and wall gates passed. Point-score normalized RMS improved to
`[0.629, 0.596, 0.426]`, but the child global score was
`[-3.310, 1.149, -3.369]` versus validation Fisher
`[-5.052, 1.793, -4.892]`; the precise third coordinate failed by `53.46`
standardized units and the retained-prefix third coordinate by `70.67`.
Off-origin mass and shape also failed.

The failure is consistent with linear-amplitude curvature: for
`h(theta)=h0+theta_k H_k`, the exact mass contains the unavoidable positive
term `theta_k^2 <H_k,H_k>`. The large score-informed tangent made the child
log mass about `2.16` at radius `0.01`, while the target change was about
`+/-0.05`. Learning-rate and L1 changes do not remove this algebraic term.
Run `a02` next because rank 4 can test whether a lower-norm tangent
representation repairs the issue. If rank 4 retains the same mass/score
failure, skip the stale LR/L1 linear arms and trigger the predeclared quadratic
feature repair.

`a02` also produced valid negative evidence. Rank 4 reduced the radius-0.01
child log masses from about `2.16` to `0.58-0.62`, but mass, shape, point,
global-score, and prefix gates still failed. The third global and prefix
standardized residuals were `63.41` and `57.85`; XLA parity was `3.55e-14` and
peak allocation was `833,697,024` bytes. More linear rank is therefore not the
next repair.

Before opening quadratic amplitude features, evaluate the selected weight-10
rank-2 initializer on validation without absolute-density updates. This uses the
same already-open validation stream and cannot select or admit a child. It
determines whether the linear tangent should be frozen while quadratic
zero-derivative components repair mass/shape, or whether the tangent/prefix
representation itself must be repaired first.

The no-update validation diagnostic passed normalized point-score transfer at
`[0.343, 0.328, 0.533]`, but global score was just outside its gate at a
maximum `1.556` standardized units and prefix score failed at
`[8.31, 16.98, 121.76]`. This proves the joint point/global additive fit does
not define the required retained-prefix tangent. Quadratic features have zero
origin derivative and cannot repair that score gap.

### Training-Prefix Tangent Repair

Augment the same 180-column additive ridge with the exact origin prefix-score
operator at the first three fixed training-origin prefix points. For additive
coefficient vector `c`, each row is

\[
 D_{\rm prefix}(x)c=
 \frac{2\int h_0(x,u)H_c(x,u)\,d\mu(u)}
      {\int h_0(x,u)^2\,d\mu(u)+\tau}
 -\frac{2\langle h_0,H_c\rangle}{\langle h_0,h_0\rangle+\tau}.
\]

The target rows come from independent conditional-ratio estimates with 8,192
rows per point, seed `85251`, and the training Fisher score. They use physical
`z1` for the authority and the matching first 18 local coordinates for the
finite child. No validation or untouched prefix authority enters the fit.

Freeze ridge `1e-4` and global-score weight 10. Test prefix standardized-row
weights `0.01`, `0.1`, and `1.0`. These are hypotheses because inverse-MCSE
scaling can otherwise let the precise third coordinate dominate the joint
point-score field. An arm must pass finite, training point RMS `<=0.90`,
training global score within `3*MCSE+1e-5`, every training prefix coordinate
within `3*MCSE+1e-5`, prefix ESS at least half the requested rows, and the
existing origin/memory gates. Among viable arms, validation selects the minimum
maximum of likelihood and prefix standardized residuals, then point RMS, then
smaller prefix weight. Validation mass/shape are explanatory for this tangent
repair and cannot be relaxed for the later density candidate.

At most three training-prefix fits and validation diagnostics may run, each
under 15 minutes and under the existing 6 GiB cap. Invalid prefix authority,
operator parity failure, origin drift, non-finite solve, or memory failure is a
continuation veto. A valid but non-transferring tangent triggers feature/rank
repair; it does not authorize fitting validation or consuming untouched data.

Skeptical audit: this repair targets the exact missing claimed operator, uses a
new training-only authority, keeps validation as selection/veto only, leaves
the independent untouched claim sealed, and stores only a 180-column design
plus three prefix rows. It does not treat point loss as prefix evidence or add a
retained state grid. Verdict: implement operator parity first, then run the
three bounded prefix-weight arms.

Prefix weight `0.01` passed every training-prefix row with maximum standardized
residual below `0.08` and kept training/validation point-score RMS below `0.58`.
It nevertheless missed the precise third training global score by `2.32`
standardized units and the validation prefix by up to `59.93`. The prefix rows
are already nearly interpolated, so a larger weight may only worsen the global
tradeoff; run weight `0.1` as the discriminating transfer test. If it does not
materially reduce validation prefix error, skip weight `1.0` and repair prefix
point coverage/features rather than over-weighting already fitted rows.

Weight `0.1` reduced the maximum validation prefix residual only from `59.93`
to `54.65`, while worsening the third validation global residual from `2.46`
to `5.40` and the third training global residual from `2.32` to `4.41`.
Weight `1.0` is therefore dominated and must not run.

The next repair tests coverage rather than more weight. Use the first 16 fixed
training-origin prefix points with the same independent 8,192-row conditional
authority and seed. Raise the global row weight from 10 to 100 to preserve the
precise third likelihood score, and test per-prefix-row weights `0.001` then
`0.01`; the smaller value approximately offsets the increase from 3 to 16
points. Everything else remains fixed. Run the smaller prefix weight first. If
training prefix fit is valid and validation prefix error materially improves,
run the second weight; otherwise the additive feature family, not coverage
weight, is the repair target. This is still training-only fitting with the
already-open validation stream used only for selection/veto.

The 16-point `0.001` arm repaired validation global score (maximum standardized
residual `0.715`) and reduced validation prefix error from `54.65` to `11.56`.
It passed 47 of 48 training-prefix coordinates; the sole miss was `1.059`.
This is material coverage improvement with a narrowly underweighted training
row, so the predeclared `0.01` arm is justified. If it still cannot pass
validation prefix, the next repair must expand beyond additive univariate
features rather than add more row weight.

The 16-point `0.01` arm passes every training tangent gate and validation
point/global gates. Its validation prefix residuals are
`[2.52, 0.067, 9.86]`. Coverage has reduced the maximum prefix residual
monotonically from `121.76` (no prefix rows), through `59.93` and `54.65`
(three points), to `11.56` and `9.86` (16 points). The earlier immediate
feature-expansion trigger is therefore refreshed: run one 64-point arm before
adding pairwise spatial features. Use global weight 100 and per-row prefix
weight `0.0025`, preserving the total prefix weight `16*0.01=64*0.0025` while
changing coverage only. If validation prefix does not pass or materially
improve, additive coverage is exhausted and interaction/TT features become the
next repair. The 64 authorities remain training-only, use the same seed/count,
and fit the same 180-column solve; validation and untouched roles are unchanged.

The 64-point arm exhausted additive coverage. Validation prefix worsened to
`13.70`; training point RMS failed on the third coordinate at `1.051`, and
multiple training prefix rows failed, while validation global score remained
valid. More additive rows or weight are not justified.

### Within-Region Pair Repair

Extend the linear feature family by the 18 disjoint within-region pairs
`(S_i,I_i)`: nine pairs in `z1` axes `0:18` and the matching nine pairs in `z0`
axes `18:36`. Each pair uses the 25 products of the existing five cardinal
basis functions. Together with 180 additive columns this gives 630 columns,
still solved by bounded weighted ridge. The pair set is physically grounded in
the SIR reaction coupling and the frozen `[S1,I1,...,S9,I9]` order; arbitrary
cross-region adjacency and temporal pairs are deferred.

The disjoint adjacent-pair sum has an exact rank-7 TT automaton: start and
accumulated channels plus five one-axis carrier channels. This avoids a block
sum over 450 terms and keeps pair contractions polynomial and bounded. Additive
and pair point, global, and prefix operators must tie out to the exact finite
child before execution.

Use the best valid 16-point training-prefix design, global weight 100, prefix
weight `0.01`, ridge fractions `1e-4` and `1e-3`. The ridge ladder tests the
material new conditioning risk from redundant cardinal-product columns. A
candidate must pass every training tangent gate and the validation point,
global, and prefix score subset. Among viable candidates minimize maximum
validation likelihood/prefix standardized residual, then point RMS, then
smaller ridge. Mass and shape remain explanatory until zero-derivative
quadratic amplitude repair; they are not relaxed for the eventual density
candidate.

At most two pair arms may run, each under 20 minutes and 6 GiB. Operator parity,
finite Cholesky, authority validity, origin identity, and memory are
continuation vetoes. A valid pair tangent failure triggers a higher TT feature
family or a reassessment of the fixed parent basis; it does not authorize
validation fitting, retained grids, ALS/TT-cross, or untouched data.

Skeptical audit: the repair targets the observed non-transfer of retained-prefix
score, uses model-grounded interactions, preserves the exact same finite score
operators and data roles, and has explicit conditioning and memory checks. It
does not infer density validity from tangent validity. Verdict: implement and
test exact rank-7 representation/operator parity, then run ridge `1e-4` first.

The ridge-`1e-4` pair arm passes every training tangent gate, with point RMS
`[0.401, 0.367, 0.452]` and maximum training-prefix residual below `0.03`.
Validation global and point gates pass, but validation prefix worsens to
`[7.70, 14.10, 48.65]`, versus additive `p05` at
`[2.52, 0.067, 9.86]`. This is training-prefix overfit, not operator or memory
failure. Run the predeclared stronger ridge `1e-3` because it targets precisely
this conditioning/generalization risk. If it does not beat the additive
maximum `9.86`, reject the within-region pair family and do not add more pair
weight or columns.

Ridge `1e-3` still fails validation prefix at
`[6.01, 3.41, 34.10]`; it does not beat additive `p05` at
`[2.52, 0.067, 9.86]`. The within-region pair family is rejected. Both pair
arms pass every training tangent gate, so this is feature generalization
failure, not implementation, authority, or memory failure.

### Direct TT Tangent Repair

Stop adding hand-designed linear columns. Optimize the residual TT cores
directly against the exact finite-child origin operators, starting from the
best additive 16-point tangent `p05`. Preserve that rank-2 function exactly,
then add complete seeded channels at target rank 4 or 7. The objective combines:

1. normalized joint point-score loss on the 4,096 training-origin rows;
2. global Fisher residual scaled by `3*MCSE+1e-5`; and
3. mean retained-prefix residual scaled by each authority's
   `3*MCSE+1e-5`.

Use the first 64 training-origin prefix points for fitting with authority seed
`85251`, and the next 64 for training-calibration with independent seed
`85252`. Validation retains seed `85401` and remains selection/veto only;
untouched seeds remain sealed. No prefix point may be dropped.

Initial arms are rank 4 and 7, Adam learning rates `1e-5` and `3e-5`, 64
full-batch steps, global and prefix objective weights 1, no L1, L2 `1e-10`,
and clip norm 100. Begin with rank 4 at `1e-5`; review loss and calibration
before the next arm. The 4,096 joint point/global rows remain the established
training stream; the disjoint fit/calibration split applies to retained-prefix
points and authorities. An arm is viable only if training point/global/prefix,
calibration prefix, ESS/MCSE, finite, origin-value, memory, and wall gates pass.
Validation then applies the same score subset. Training and prefix calibration
choose steps/LR; validation may nominate rank but cannot admit T1.

At most four direct-TT tangent arms may run, each under 30 minutes and 6 GiB.
Every attempt stores its complete fit/calibration authorities, trace, final
cores through a child artifact, fresh reload, and XLA parity. A missing
gradient, operator parity failure, invalid authority, origin drift, non-finite
state, or memory failure is a continuation veto. A valid direct-TT failure is
evidence against the tested rank/optimizer, not against the fixed parent value.

Skeptical audit: this repair addresses the failure of restricted feature
families while retaining the same claimed score operators and disjoint data
roles. The fit/calibration split prevents selecting optimizer depth on
validation. Rank, LR, steps, weights, seeds, gates, memory, and nonclaims are
explicit. The run could still overfit 64 prefix points; the independent
training-calibration and validation prefix gates expose that. Verdict:
implement exact prefix/global loss and a compiled direct-TT tangent step, prove
mechanics on a focused test, then run rank 4 at `1e-5`.

Rank-4 `d01` is valid negative evidence. Fresh reload, XLA parity `2.84e-14`,
authority validity, memory (`790,232,576` bytes), and wall gates passed. Fit
prefix maximum residual fell from `95.88` to `28.99`, and calibration from
`87.87` to `67.25`, but the third point RMS rose from `0.706` to `1.140` and
the third global residual from `0.516` to `2.165`. Validation prefix remained
`28.65`. The optimizer is trading point/global validity for prefix fit at rank
4. A larger learning rate at the same rank would accelerate the same failed
trajectory, so `d02` is dominated and must not run. Run rank-7 `d03` at
`1e-5` next; it is the capacity discriminator for simultaneous operator fit.

Rank-7 `d03` follows rank 4 almost exactly: fit prefix falls to `28.70`,
calibration to `67.15`, but third point RMS reaches `1.142` and global residual
`2.134`; validation prefix is `28.85`. Extra rank did not repair the tradeoff.
The remaining higher-learning-rate arms are dominated under this objective.

The phase audit found a loss-scaling defect in the plan, not in the score
operators. Initial block losses were point `1.125`, global `0.089`, and prefix
`238.3`; raw weights `(1,1,1)` let prefix fitting dominate by roughly two
orders of magnitude and forced the point/global vetoes. Replace `d02/d04` with
feasibility-balanced rank-7 arms: point weight 100, global weight 100, prefix
weight 1, LR `1e-5`, 64 steps; if global alone crosses while point remains
valid, test global weight 1000. These are constraint-preservation hypotheses,
not defaults. They preserve the same data, target, authorities, rank, and
budget. Promotion gates remain unchanged; improved total loss cannot override
any point/global/prefix veto. Verdict: run the `(100,100,1)` arm first.

The `(100,100,1)` rank-7 arm `d05` is valid negative evidence. It preserves
the training and validation point/global gates, with training point RMS
`[0.560,0.488,0.826]`, training global residual
`[0.076,0.072,0.0013]`, and validation likelihood residual
`[0.694,0.819,0.818]`. It does not generalize the retained-prefix score:
training-fit maximum residual is `38.30`, independent training-calibration
maximum is `76.21`, and the three validation-prefix residuals are
`[3.07,0.851,14.64]`. Fresh reload, XLA parity, authority validity, wall time,
and memory (`832,969,984` allocator peak bytes) pass.

Do not run `d06`. Raising the global weight from 100 to 1000 does not target
the remaining prefix-generalization failure because `d05` already passes both
point and global feasibility gates. It would answer the wrong question and is
retired before execution.

### Rotating Prefix-Pool Repair

The fixed-first-index prefix design realized the pre-mortem risk recorded in
the default audit: direct TT optimization learned selected prefix points but
did not transfer to disjoint points. The next smallest discriminator changes
coverage and checkpoint selection only. It does not change the finite score
operators, parent, basis, rank, validation data, target density, or gates.

Generate one setup-static training-only pool as follows:

1. deterministically shuffle the 4,096 origin training rows with seed `85901`;
2. use the first 512 shuffled rows as the prefix-fit pool and the next 64 as a
   disjoint training-calibration pool;
3. estimate every conditional-ratio authority with 8,192 draws, fit seed
   `85251`, and calibration seed `85252`;
4. require every authority to have ESS at least 4,096 and coordinate MCSE at
   most `[2.0,1.0,0.5]`; and
5. store every selected row index and complete authority in the result.

Start from the same `p05` additive tangent embedded with connected channels at
rank 7. Run 256 Adam updates at learning rate `1e-5`, rotating deterministic
64-point prefix minibatches so that each eight-update epoch visits all 512 pool
points. This is 32 exact pool epochs and four times `d05`'s total prefix
point-exposures. Preserve point/global feasibility with objective weights
`(point,global,prefix)=(100,100,1)`, L2 `1e-10`, and clip norm 100. The 4,096
point rows and global authority remain full-batch because they are already
bounded and stable.

Every eight updates, evaluate the disjoint 64-point calibration pool. Among
checkpoints satisfying point RMS `<=0.90` coordinatewise and global residual
`<=1.0` coordinatewise, select the checkpoint with the smallest maximum
calibration-prefix standardized residual, breaking ties by calibration mean
squared residual and then earlier update. Restore that checkpoint before all
training, validation, reload, and XLA gates. Calibration selects optimizer
depth only; it is never added to the gradient. Existing validation seed
`85401` remains a selection/veto diagnostic and cannot be fit. Untouched seeds
remain sealed.

The repair passes only if the restored tangent passes the full 512-point fit
pool, the disjoint 64-point calibration pool, and the unchanged validation
point/global/prefix score subset, plus authority, finite-state, reload, XLA,
memory, and wall gates. Passing does not admit T1: it licenses freezing the
linear tangent and adding zero-origin-derivative quadratic theta features for
off-origin mass/shape repair. Failure is valid evidence against rotating
rank-7 TT optimization under this coverage and objective, not against the
fixed parent value or score-operator algebra.

Evidence contract: the question is whether wider iid prefix coverage repairs
generalization relative to `d05`; the primary discriminator is disjoint
calibration and validation prefix residual, while point/global failures are
vetoes. Training losses and the selected update are explanatory. No result can
establish T1 admission, T2, HMC readiness, source-faithful Zhao-Cui score
estimation, exact nonlinear likelihood, posterior correctness, or superiority.
The arm writes a fresh result and child under
`docs/plans/artifacts/zhao-cui-austria-sir-parameter-density-t1-20260801/rotating-prefix-tangent-v1/`.

Exact trusted GPU/XLA command:

```bash
TF_FORCE_GPU_ALLOW_GROWTH=true MPLCONFIGDIR=/tmp/matplotlib-cache \
/home/chakwong/anaconda3/envs/tf-gpu/bin/python \
scripts/run_zhao_cui_austria_sir_parameter_density_t1.py \
--rotating-prefix-tangent q01_r7_pool512_batch64_steps256 \
--output-dir docs/plans/artifacts/zhao-cui-austria-sir-parameter-density-t1-20260801/rotating-prefix-tangent-v1/q01_r7_pool512_batch64_steps256 \
--max-seconds 1800
```

`c02` completed in `233.69` seconds with a `794,677,504` byte peak and no
engineering veto. The restored final checkpoint had point RMS
`[0.975,0.913,0.411]`, global residual below `0.180`, feasibility ratio
`1.0834`, calibration maximum `22.27`, and validation prefix
`[10.46,11.04,2.77]`. Lower LR removed the large global oscillation but still
did not reach point feasibility, and the first point coordinate moved only
from `1.0` to `0.975` in 256 updates. `c01/c02` close the Adam LR audit; do not
run a third Adam arm.

### Deterministic Core-Affine Quadratic Solve

For fixed parent and frozen basis, all three origin score operators are linear
in the core-affine tangent coefficients. With weighted squared point, global,
and prefix residuals plus L2, the complete 512-point fit objective is a convex
quadratic in 8,280 coefficients. Rotating Adam is therefore an avoidable
stochastic optimizer for this phase.

Implement a functional flattened-position evaluator that reconstructs the
three parent-shaped tangent banks, builds the exact rank-8 product-rule
components, and returns the same `(100,100,1)` full-pool score loss and gradient
under XLA. Prove value/gradient parity against `CoreAffineTangentTrainer` and a
directional finite-difference diagnostic. Run one full-pool capacity probe
before optimization; veto nonfinite gradients, XLA failure, or allocator peak
above 6 GiB.

Use TensorFlow Probability L-BFGS with zero initialization, 20 correction
pairs, at most 128 iterations, gradient tolerance `1e-8`, relative objective
tolerance `1e-12`, and at most 50 line-search iterations. These are bounded
solver hypotheses, not defaults. L2 remains `1e-10`; no L1 or gauge is added.
The disjoint 64-point calibration and existing validation are evaluated only
after the solver terminates and cannot alter its objective or coefficients.

The solver arm passes only if it converges without a solver failure and the
same point/global, 512 fit-prefix, 64 calibration-prefix, validation score,
authority, reload, XLA, memory, and wall gates pass. Nonconvergence with a
finite candidate is a solver failure/repair trigger, not a representation
rejection. Convergence with failed prefix gates rejects mean-squared fitting of
this core-affine family and triggers a minimax/IRLS objective audit rather than
more optimizer tuning.

Evidence contract: the baseline is `c01/c02`; the question is whether solving
the declared convex full-pool objective, rather than stochastic Adam, reaches
the existing score gates. Objective decrease and solver convergence are
explanatory/validity diagnostics; only the unchanged score gates can promote
the tangent to off-origin density repair. No untouched data, T2, or HMC is
opened by this phase.

Skeptical audit: the target, representation, data, weights, and gates are held
fixed; the changed mechanism is precisely the optimizer failure identified by
`c01/c02`; full-pool fitting removes minibatch noise; solver and memory limits
are explicit; and validation remains gradient-free. Verdict: implement and
test the functional XLA value/gradient, run the capacity probe, then one L-BFGS
arm.

The first focused callback test found a setup/trace defect before any solver
run: `core_affine_origin_total_score_loss_arrays` rebuilt the product basis
inside `tf.function`, but basis construction deliberately performs eager
validation. Hoist the frozen centered basis outside the trace and capture it in
the callback. This is a localized implementation repair and enforces, rather
than changes, the setup-static basis contract.

The repaired functional GPU/XLA value/gradient and directional finite-
difference test passed. The L-BFGS runner performs the full-pool callback as a
hard capacity check before optimization and records its allocator peak. The
30-minute harness cap is checked before solver launch and after final
evaluation; the in-solver bound is 128 iterations and 50 line-search steps per
iteration.

Exact trusted GPU/XLA command:

```bash
TF_FORCE_GPU_ALLOW_GROWTH=true MPLCONFIGDIR=/tmp/matplotlib-cache \
/home/chakwong/anaconda3/envs/tf-gpu/bin/python \
scripts/run_zhao_cui_austria_sir_parameter_density_t1.py \
--core-affine-lbfgs l01_core_affine_fullpool_lbfgs \
--output-dir docs/plans/artifacts/zhao-cui-austria-sir-parameter-density-t1-20260801/core-affine-lbfgs-v1/l01_core_affine_fullpool_lbfgs \
--max-seconds 1800
```

`l01` completed in `187.46` seconds without an authority, XLA, memory,
serialization, or finite-state veto.  The initial callback used `283,041,280`
peak bytes and the complete run used `807,786,496` peak bytes.  L-BFGS reduced
the objective from `499286.4499` to `100.81785`, but reached the frozen
128-iteration cap without convergence; its final gradient norm was `70.7336`.
The finite candidate passed point RMS (`[0.851,0.372,0.176]`) and global-score
gates but failed the fit-prefix (`20.57` maximum), calibration-prefix (`21.03`
maximum), and validation-prefix (`[3.36,0.422,0.963]`) gates.  This is the
predeclared solver-repair trigger, not a representation rejection.  The result
SHA-256 is
`051d7cb8f7f6a67a0d0e4bc0328042299348a42a4312406deb1607f495bbb074`;
the child manifest SHA-256 is
`7c7803eb9aa77ef82706469209af73d6b036c875c0932f410c8fb79113b21dcb`,
and its child identity is
`c6e6334e7f711c13f1f115f4508aaaf21d33e8c7bb62e09050eea740bf444e00`.

### Matrix-Free Normal-Equation Repair

The score operators are linear in the 8,280 core-tangent coefficients, so the
same full-pool objective has gradient `g(x)=H x+b`.  The core coordinates also
contain gauge-like directions; their only strict curvature is the frozen
`1e-10` L2 term.  This explains why a large objective reduction can coexist
with a nonconverged quasi-Newton solve, but it does not establish that the
finite function family can pass the prefix gates.

Run one matrix-free conjugate-gradient repair on the unchanged normal equation
`H x=-b`.  Obtain Hessian actions from the exact compiled quadratic gradient,
not a materialized `8280 x 8280` matrix.  Warm-start from the hash-bound `l01`
child after proving that extracting its structured core tangents and rebuilding
the child reproduces its identity exactly.  Use at most 512 conjugate-gradient
iterations and a residual tolerance of `1e-10` relative to `||b||`; retain L2
`1e-10`, the full 512-point fit pool, the 64-point calibration pool, the same
authorities and `(100,100,1)` weights, and every existing scientific gate.

Evidence contract: the engineering question is whether `l01` stopped because
of quasi-Newton conditioning.  A finite matrix-free action, positive curvature,
relative normal-equation residual, and objective decrease are solver validity
diagnostics.  The unchanged point/global/fit-prefix/calibration-prefix and
validation score gates remain the only tangent promotion criteria.  Validation
cannot enter the solve.  A passed tangent still opens only zero-origin-
derivative off-origin density repair; it does not admit T1, T2, HMC, source-
faithfulness, posterior correctness, superiority, or production readiness.

Continuation rules: a capacity/XLA/nonfinite/invalid-curvature failure is an
implementation or conditioning blocker to diagnose.  A finite solve that does
not reach its normal-equation tolerance is a terminal solver failure for this
objective.  A converged solve that fails any prefix gate rejects mean-squared
fitting for the current core-affine family and triggers a minimax/IRLS loss
audit.  Do not run another L-BFGS duration, Adam learning rate, CG tolerance, or
iteration arm.

Skeptical audit: the baseline is the actual hash-bound `l01` candidate; the
target, parent, representation, data roles, weights, L2, authorities, and gates
are unchanged; no proxy replaces prefix validation; the 512-iteration,
30-minute, and 6-GiB stops are explicit; and an exact block-TT round trip guards
against silently solving from a different position.  The run can still
converge while proving only the least-squares optimum is inadequate, so solver
convergence is explicitly non-promotional.  Verdict: implement the extraction
round trip and matrix-free solve, pass focused tests, then run one `n01` arm.

Exact trusted GPU/XLA command:

```bash
TF_FORCE_GPU_ALLOW_GROWTH=true MPLCONFIGDIR=/tmp/matplotlib-cache \
/home/chakwong/anaconda3/envs/tf-gpu/bin/python \
scripts/run_zhao_cui_austria_sir_parameter_density_t1.py \
--core-affine-cg n01_core_affine_fullpool_cg_from_l01 \
--output-dir docs/plans/artifacts/zhao-cui-austria-sir-parameter-density-t1-20260801/core-affine-cg-v1/n01_core_affine_fullpool_cg_from_l01 \
--max-seconds 1800
```

`n01` completed in `186.96` seconds with exact hash and structured-tangent
round-trip checks and a `795,647,232` byte allocator peak.  It remained finite
and positive-curvature but reached the 512-iteration cap: relative normal-
equation residual `1.36e-5`, not the frozen `1e-10` tolerance.  It reduced the
same objective from the `l01` value `100.81785` to `23.63120`.  The candidate
passed point RMS (`[0.329,0.244,0.148]`), global score, and the independent
three-point validation prefix (`[0.698,0.311,0.338]`), but failed the broad
fit-prefix and calibration-prefix gates with maxima `10.25` and `16.60`.
The result SHA-256 is
`5de920f96bea2b473d801e304a72a7e3a3f7a1277c31302d3c41542d1e4526db`;
the child manifest SHA-256 is
`3744cb2da72c4feeac5538282f9f7b31be1c665edbc302a1dba31151fdf4dcd1`.
This closes further mean-squared solver tuning under the frozen continuation
rule.  It does not reject the analytical score operators or parent value.

### Gate-Scaled Smooth-Minimax Audit

The least-squares objective averages 1,536 prefix residual coordinates, while
the actual gate is their maximum.  The broad-pool failures therefore trigger
the planned objective audit.  Define the training gate vector

\[
 z(x)=\left(
   \frac{R_{\mathrm{point},1}(x)^2}{0.9^2},\ldots,
   \frac{R_{\mathrm{point},3}(x)^2}{0.9^2},
   R_{\mathrm{global},1}(x)^2,\ldots,
   R_{\mathrm{prefix},512,3}(x)^2
 \right),
\]

where every quantity is the existing normalized or standardized residual.  Fit
the convex smooth upper approximation

\[
 L_{64}(x)=\frac{1}{64}\log\sum_j\exp(64 z_j(x))
            +10^{-10}\lVert x\rVert_2^2.
\]

This targets the declared maximum gates directly and treats point, global, and
prefix constraints in their own existing threshold units.  Temperature 64 is
a bounded optimization hypothesis; with 1,542 terms its log-sum-exp gap above
the largest squared gate ratio is at most `log(1542)/64 = 0.115`.  It is not a
new score definition or admission threshold.

Warm-start from the hash-bound `n01` child after the same exact structured
round trip.  Use one TFP L-BFGS arm with 20 correction pairs, at most 256
iterations, gradient tolerance `1e-8`, relative objective tolerance `1e-12`,
and 50 line-search iterations.  The 512 fit points alone enter gradients; the
64 calibration points and unchanged validation remain evaluation-only.  No
second temperature, duration, or optimizer arm is allowed under this audit.

Evidence contract: the question is whether the current core-affine function
family can simultaneously satisfy the broad score gates when optimized for
their maximum rather than their mean square.  Solver convergence and smooth-
max reduction are validity/explanatory diagnostics.  Promotion still requires
every unchanged point/global/fit-prefix/calibration-prefix and validation score
gate, plus authority, reload, XLA, memory, and wall gates.  Passing opens only
zero-origin-derivative off-origin density repair.  It does not establish T1,
T2, HMC, source-faithfulness, posterior correctness, superiority, or production
readiness.

Continuation rules: nonfinite state, invalid source hashes, XLA/capacity, or a
solver failure before a finite improving candidate is an implementation
blocker.  A finite terminal candidate that fails the fit-prefix gate rejects
this smooth-minimax rank-8 core-affine candidate and triggers a representation
audit.  A fit pass with calibration or validation failure rejects its broad-
prefix generalization and likewise triggers a representation audit.  A full
score-gate pass freezes the tangent and begins quadratic-theta density repair.

Skeptical audit: this phase changes exactly the mismatch exposed by `l01/n01`
between mean-square optimization and maximum-residual promotion.  It binds the
actual strongest prior candidate, keeps calibration/validation out of the
gradient, normalizes all terms by existing gates rather than inventing a proxy,
and retains explicit 30-minute and 6-GiB stops.  A successful command cannot
silently promote because every original gate is reevaluated.  Verdict:
implement and test the smooth-minimax callback, then run one `m01` arm.

Exact trusted GPU/XLA command:

```bash
TF_FORCE_GPU_ALLOW_GROWTH=true MPLCONFIGDIR=/tmp/matplotlib-cache \
/home/chakwong/anaconda3/envs/tf-gpu/bin/python \
scripts/run_zhao_cui_austria_sir_parameter_density_t1.py \
--core-affine-minimax m01_core_affine_gate_max_from_n01 \
--output-dir docs/plans/artifacts/zhao-cui-austria-sir-parameter-density-t1-20260801/core-affine-minimax-v1/m01_core_affine_gate_max_from_n01 \
--max-seconds 1800
```

`m01` completed in `217.31` seconds without source-hash, authority, XLA,
finite-state, reload, wall, or memory (`808,418,816` peak bytes) vetoes.  It did
not converge in 256 L-BFGS iterations; objective fell from `105.1096` to
`39.6967`, while gradient norm rose to `2821.3`.  The fit-prefix maximum fell
from `10.25` to `6.30`, but calibration remained `15.71`, validation prefix
worsened to `[2.01,2.16,2.67]`, and two validation likelihood coordinates
failed.  The result SHA-256 is
`cb99df75c5e764d2df9b0321bbd578d4641c36171551a0e6c54083db1dc38397`;
the child manifest SHA-256 is
`da08d5195cdd40305d13e7c197d3bbd9a2d8da8a055381657fa1c71999652f3c`.
The frozen continuation rule closes further temperature, duration, and
optimizer tuning in the 8,280-dimensional core-affine manifold.

### Full Rank-8 Residual-TT Representation Audit

The exact product-rule encoding places each score component in a rank-8 TT but
constrains its cores to repeated parent blocks, structural zeros, and one
parent-shaped tangent block per axis.  This uses 8,280 free tangent
coefficients.  The same three rank-8 residual TTs contain 32,880 core entries
when those block constraints are released.  The larger family contains the
hash-bound `n01` function exactly and does not increase TT rank, create a
retained grid, add theta coordinates, or change the score operators.

Run one full-rank-8 arm from the `n01` child, not the overfit `m01` child.  Verify
the `n01` result and manifest hashes and an exact load/freeze identity round
trip.  Flatten all three residual-TT core sequences, reconstruct them inside a
compiled callback, and minimize the same temperature-64 gate-scaled smooth
maximum on the 512 fit-prefix points.  Use L2 `1e-10` on displacement from the
`n01` position rather than on the absolute block encoding; otherwise the
representation audit would newly penalize the repeated fixed parent blocks.
Use TFP L-BFGS with 20 correction pairs, at most 256 iterations, gradient
tolerance `1e-8`, relative objective tolerance `1e-12`, and at most 50 line-
search iterations.  Calibration and validation remain evaluation-only.

Before the campaign run, prove flattened-position round-trip parity, compiled
value/gradient directional parity, and nonzero gradient in at least one
released (non-product-rule-tangent) coordinate at a nonzero product-rule warm
start.  The full callback must pass a 6-GiB capacity check before optimization.

Evidence contract: the question is whether the broad conditional-prefix score
requires rank-8 core freedom outside the core-affine product-rule manifold.
Initial released-coordinate gradient and solver diagnostics are explanatory;
only the unchanged point/global/fit-prefix/calibration-prefix and validation
score gates promote.  Passing opens only zero-origin-derivative off-origin
density repair.  It does not admit T1, T2, HMC, source-faithfulness, posterior
correctness, superiority, or production readiness.

Continuation rules: invalid source hashes, round-trip/parity failure,
nonfinite state, XLA failure, or memory above 6 GiB is an implementation or
capacity veto.  A finite candidate that cannot improve the fit maximum is
evidence that released directions are ineffective.  A fit improvement without
fit/calibration/validation passage rejects this rank-8 full-TT candidate and
triggers an authority/target-identifiability audit before any higher rank.  Do
not run another rank, temperature, duration, or optimizer arm automatically.

Skeptical audit: the baseline is the strongest broad-pool candidate `n01`, not
the overfit `m01`; rank, data, authorities, gates, temperature, solver class,
and resource stops are held fixed; the parameter-count increase is explicit;
and validation cannot enter the gradient.  Earlier `d03/d05` do not answer this
question because they started from the additive `p05` function and used short
Adam mean-square trajectories rather than the `n01` broad-pool tangent and
gate-scaled objective.  A successful command cannot promote without every
original gate.  Verdict: implement and test the functional full-TT callback,
then run one `f01` arm.

Exact trusted GPU/XLA command:

```bash
TF_FORCE_GPU_ALLOW_GROWTH=true MPLCONFIGDIR=/tmp/matplotlib-cache \
/home/chakwong/anaconda3/envs/tf-gpu/bin/python \
scripts/run_zhao_cui_austria_sir_parameter_density_t1.py \
--full-tt-minimax f01_full_r8_gate_max_from_n01 \
--output-dir docs/plans/artifacts/zhao-cui-austria-sir-parameter-density-t1-20260801/full-tt-minimax-v1/f01_full_r8_gate_max_from_n01 \
--max-seconds 1800
```

`f01` completed in `215.03` seconds with exact source and full-position round
trips, a `857,519,616` byte allocator peak, and nonzero released-direction
gradient: norm `2691.79` over 24,600 released coordinates.  It did not converge
in 256 L-BFGS iterations.  The objective fell from `105.1096` to `32.3196` and
the fit-prefix maximum from `10.25` to `5.68`, but calibration remained `15.32`
and validation prefix was `[0.855,2.33,3.59]`.  The result SHA-256 is
`80c627307b4baa092af60a8be364f6543b5bae93f489777da0e6fafe7f2965a0`;
the child manifest SHA-256 is
`7f02418953e1f94ced71771efbe2fd1a69ea7c42dd116a85d177d3d95ef49912`.
Both constrained and full rank-8 families improve fit while missing disjoint
prefix points, so no higher rank or optimizer arm is authorized before the
predeclared authority/target-identifiability audit.

### Prefix-Authority Reproducibility Audit

The conditional-ratio authority reports ordinary influence-function MCSE for
each prefix point.  Its third-coordinate MCSE is identical across all points
because at T1 the third parameter contributes only the deterministic
point-specific observation score; its conditional score variance is zero, and
the prefix uncertainty is therefore the shared global-score MCSE.  This is an
algebraic explanation, not evidence that the MCSE is calibrated.

Select 32 fixed points without inspecting their scores: the first 16 shuffled
fit-pool indices and the first 16 shuffled calibration indices under partition
seed `85901`.  Construct two independent authorities `A/B`.  Each uses 4,096
fresh global-ratio rows and 8,192 conditional rows per point.  Freeze global
noise seeds `(86001,86002)` and `(86101,86102)` and conditional seeds `86051`
and `86151`.  For each point and coordinate compute

\[
 z_{AB}=\frac{|s_A-s_B|}
 {\sqrt{\mathrm{SE}_A^2+\mathrm{SE}_B^2}},
 \qquad
 g_{AB}=\frac{|s_A-s_B|}
 {3\sqrt{\mathrm{SE}_A^2+\mathrm{SE}_B^2}+10^{-5}}.
\]

Report every score/SE/ESS pair, coordinatewise mean squared `z_AB`, median and
maximum `z_AB`, the fraction with `g_AB<=1`, and the simultaneous maximum.  Do
not use this diagnostic to tune or replace any score gate.  It asks whether the
authority uncertainty used by all prior prefix gates is reproducible at its
claimed scale.

Evidence contract: independent paired differences are the primary calibration
diagnostic.  Nonfinite values, ESS below 4,096, or either conditional MCSE above
`[2,1,0.5]` are authority vetoes.  `mean(z_AB^2)` materially above one or many
`g_AB>1` values diagnose underestimated uncertainty or an invalid independence
assumption and block interpretation of prior prefix failures.  A calibrated
result does not prove the TT representation or score correct; it only keeps
the authority usable and shifts the next audit to target identifiability and
optimization.  No result opens density repair, untouched data, T2, or HMC.

Skeptical audit: points are fixed by indices rather than chosen by residual;
both global and conditional samples are independent across replicates; the
same sample sizes and estimator used by the campaign are tested; and raw paired
evidence is preserved.  Thirty-two points are enough to detect gross MCSE
failure like the observed 5--15 standardized residuals, but not to certify
tail calibration or simultaneous 576-point coverage.  Therefore a pass is
scoped to absence of gross miscalibration and cannot promote.  Verdict: add a
read-only authority diagnostic mode and run it once under a 20-minute and
6-GiB cap.

Exact trusted GPU command:

```bash
TF_FORCE_GPU_ALLOW_GROWTH=true MPLCONFIGDIR=/tmp/matplotlib-cache \
/home/chakwong/anaconda3/envs/tf-gpu/bin/python \
scripts/run_zhao_cui_austria_sir_parameter_density_t1.py \
--prefix-authority-reproducibility \
--output-dir docs/plans/artifacts/zhao-cui-austria-sir-parameter-density-t1-20260801/prefix-authority-reproducibility-v1/a01_two_seed_32points \
--max-seconds 1200
```

The first `a01` launch exposed a diagnostic harness defect before any paired
statistic was computed: independent global-noise batches were also used to
source the fixed prefix points, so the purported A/B points differed.  The
attempt is preserved as infrastructure evidence; freeze the points once from
the declared partition and retry unchanged in a fresh directory.  This repair
does not alter the scientific contract or campaign budget.

The repaired audit passed: paired `z^2` mean `0.601`, maximum `1.326`, all 32
`g_AB<=1`, authority validity passed, and peak allocation was `96.9 MB`.  This
rules out gross MCSE underestimation, but ESS and replicate agreement do not
detect bias from the linearized Gaussian conditional proposal.

### Prefix-Authority Sample-Growth Diagnostic

Evaluate eight fixed points (four fit and four calibration indices from the
same partition) with the unchanged Laplace proposal at `N=8192` and `N=65536`,
using independent conditional seeds `86251` and `86351` and an independent
global score cloud of 8,192 rows.  For each coordinate report the
small/large-estimate difference divided by
`sqrt(SE_8192^2+SE_65536^2)`, both ESS/MCSE values, and raw score drift.

This is target-identifiability evidence only.  Large drift or paired z blocks
promotion-grade use of the 8,192-row authorities and triggers a proposal or
independent-reference repair; small drift shifts the next decision back to
representation/optimization.  A larger sample is not automatically exact.
No density, T1, T2, HMC, or comparator claim is opened.  Stop at 20 minutes or
6 GiB.

Exact trusted GPU command:

```bash
TF_FORCE_GPU_ALLOW_GROWTH=true MPLCONFIGDIR=/tmp/matplotlib-cache \
/home/chakwong/anaconda3/envs/tf-gpu/bin/python \
scripts/run_zhao_cui_austria_sir_parameter_density_t1.py \
--prefix-authority-sample-growth \
--output-dir docs/plans/artifacts/zhao-cui-austria-sir-parameter-density-t1-20260801/prefix-authority-sample-growth-v1/a01_n8192_vs_n65536_8points \
--max-seconds 1200
```

The sample-growth audit passed: across eight fixed points, the 8,192-to-65,536
conditional estimate drift had maximum paired `z=0.236` and mean `z^2=0.0131`;
ESS minima were `8188` and `65504`, all MCSE/finite gates passed, and peak
allocation was `622 MB`.  This is evidence against gross proposal bias or
underestimated MCSE at the tested points.  The broad-prefix failures therefore
remain representation evidence, not an authority veto.

### Rank-12 Capacity Discriminator (Closed)

The terminal rank-12 retry has run and is rejected.  Its child is an exact
zero-padded expansion of `n01`; the connected-expansion controls described in
the earlier draft were not consumed and are historical hypotheses only.  The
child preserved the admitted origin value and passed the point-score gate, but
failed the untouched likelihood score, off-origin mass, shape, and retained-
prefix gates.  The largest retained-prefix standardized residual was `3.7249`
and the third likelihood-score residual was `1.9976`.  Its identity is
`553442f49ddd59b99bafbf0b3e7fde39d6791aa24835413b6e59e36ae93f8368`; the full
payload is under
`rank12-minimax-v1/r12_retry02_rank12_gate_max_from_n01/result.json`.

This is a valid representation failure, not an authority failure: the
two-seed reproducibility and sample-growth audits passed, with the latter
reaching ESS minima `8188` and `65504`.  Rank 12 is the terminal predeclared
discriminator.  Do not run rank 16, another optimizer, another temperature, an
off-origin density repair, or a score claim under this child.  This branch is
closed; the next action is the conditional-reference repair plan cited below.

Historical proposal text follows for provenance only and is not executable.

The constrained rank-8 and unrestricted rank-8 full-TT families both improved
the 512-point fit but failed disjoint calibration (`15.32` maximum for `f01`),
while the authority audits passed.  Run exactly one fresh rank-12 arm from the
hash-bound `n01` child.  Expand each of the three residual TTs by deterministic
connected channels from rank 8 to rank 12, preserving the `n01` function at the
initial point.  Use the same temperature-64 gate-scaled smooth maximum, the
same 512/64 split and authorities, displacement L2 `1e-10`, and one TFP L-BFGS
run with 20 correction pairs, 256 iterations, gradient tolerance `1e-8`,
relative objective tolerance `1e-12`, and 50 line-search iterations.

Before the solve, run a full callback capacity gate and require peak allocator
below 6 GiB; no dense Hessian or retained grid is allowed.  Validation remains
evaluation-only.  A fit/calibration/validation pass opens only off-origin
density repair.  A finite failure rejects rank-12 under this gate-scaled
objective and closes automatic rank escalation; the next decision is a fresh
target/representation design, not rank 16 or another optimizer arm.

Skeptical audit: target, estimator, authority sample sizes, seeds, objective,
solver, data roles, and gates are unchanged; only TT rank changes from 8 to 12.
The sample-growth artifact supports the authority but does not prove the
representation.  The rank-12 capacity probe is the earliest memory diagnostic,
and a successful command still cannot admit T1 without every original score
gate.  Verdict: implement the connected rank expansion, focused parity, and
one bounded `r12` arm.

Exact trusted GPU command:

```bash
TF_FORCE_GPU_ALLOW_GROWTH=true MPLCONFIGDIR=/tmp/matplotlib-cache \
/home/chakwong/anaconda3/envs/tf-gpu/bin/python \
scripts/run_zhao_cui_austria_sir_parameter_density_t1.py \
--rank12-minimax r12_rank12_gate_max_from_n01 \
--output-dir docs/plans/artifacts/zhao-cui-austria-sir-parameter-density-t1-20260801/rank12-minimax-v1/r12_rank12_gate_max_from_n01 \
--max-seconds 1800
```

The first `r12` launch stopped before callback construction on a runner import
defect (`embed_residual_component_at_rank` was not imported).  The second launch
then reached the rank-12 branch but exposed the same missing import before any
scientific computation.  Both are preserved infrastructure failures; add the
import and retry unchanged in a fresh directory.  No rank, target, authority,
solver, or gate has changed.

The first `c01` launch completed all 256 updates but hit a result-harness defect:
no point/global-feasible checkpoint caused a `RuntimeError` before the trace and
final candidate could be serialized. This is not scientific evidence for or
against `c01`; the only preserved facts are finite XLA execution and a
`191.54` second wall time. Repair the harness to retain a least-infeasible
checkpoint for explanatory rejected-candidate output. Order fallback
checkpoints first by the maximum of coordinatewise `point_RMS/0.90` and global
standardized residual, then calibration maximum, calibration mean squared
residual, and update. A fallback can never satisfy the training gate or be
selected as viable. Retry `c01` unchanged in a fresh `retry01` directory; the
scientific target, data, method, settings, gates, hardware, and total campaign
boundary do not change.

`retry01` successfully preserved a least-infeasible checkpoint but then exposed
a second wrapper-interface defect: final validation called `heldout_metrics`,
which the structured trainer had not delegated to the exact centered child
operator. Add this delegation and retry unchanged as `retry02`. Neither failed
attempt produced a scientific result; both remain preserved infrastructure
evidence.

`retry02` completed the first valid current-basis core-affine result in
`236.30` seconds with a `794,685,696` byte peak. No point/global-feasible
checkpoint appeared. The least-infeasible checkpoint was update 248 with
point RMS `[0.967,0.968,0.384]`, global residual
`[0.0088,0.175,0.908]`, feasibility ratio `1.0755`, calibration maximum
`22.94`, and validation prefix `[10.23,14.85,1.34]`. The first two point
coordinates decreased slowly, while the third/global coordinate oscillated
across otherwise finite updates. Authority, reload, XLA, memory, and wall gates
passed. This is a valid `c01` tuning failure and triggers the predeclared
lower-LR arm `c02`, not a rejection of the current-basis core-affine family.

`c02` changes only learning rate from `1e-3` to `3e-4`; every other field is
identical to `c01`. Its question is whether lower LR suppresses the global
oscillation enough to reach simultaneous point/global feasibility. If it does
not, do not run a third Adam LR arm; audit a deterministic solver for the
convex quadratic current-basis tangent objective.

Exact trusted GPU/XLA command:

```bash
TF_FORCE_GPU_ALLOW_GROWTH=true MPLCONFIGDIR=/tmp/matplotlib-cache \
/home/chakwong/anaconda3/envs/tf-gpu/bin/python \
scripts/run_zhao_cui_austria_sir_parameter_density_t1.py \
--rotating-prefix-tangent c02_core_affine_zero_lr3e4_steps256 \
--output-dir docs/plans/artifacts/zhao-cui-austria-sir-parameter-density-t1-20260801/core-affine-prefix-tangent-v1/c02_core_affine_zero_lr3e4_steps256 \
--max-seconds 1800
```

`q03` completed in `227.56` seconds with a `795,611,392` byte peak and all
engineering/resource gates passed, but it is not a clean test of the intended
historical function. Under the current frozen basis its starting point RMS was
`[0.958,0.849,2.032]` and global residual
`[0.681,1.097,39.56]`, materially different from the old artifact's reported
point-only behavior. The source-closure staleness was therefore substantive:
the pre-repair GPU basis coefficients do not define the same function under
the frozen CPU basis. Only updates 240, 248, and 256 satisfied point/global
feasibility. The selected final checkpoint had calibration maximum `28.58`,
validation prefix `[10.50,2.89,5.45]`, point RMS
`[0.897,0.703,0.516]`, and global residual below `0.053`. The arm is rejected,
and its historical warm start must not be extended or promoted.

### Current-Basis Core-Affine Repair

Retain the product-rule parameterization but remove the incompatible warm
start. Train fresh current-basis core tangents `D_{j,p}` directly. At the
origin, point, global, and prefix scores are linear functions of these tangent
cores even though their exact block-TT representation has rank 8. The combined
score objective is therefore a convex quadratic in `D`; this removes the free
rank-8 TT factorization nonconvexity and preserves the closest finite-program
extension of the working fixed parent.

Implement a `CoreAffineTangentTrainer` that owns only three tangent banks with
the parent-core shapes. It must use the frozen centered basis for every point,
mass, and prefix contraction; convert to the exact rank-8 block TT only at the
operator/freeze boundary; and expose the same analytical point/global/prefix
metrics as the centered trainer. Prove zero-slice value, operator parity,
finite nonzero gradients from a fresh zero initialization, fresh child reload,
and XLA parity. The historical gauge method is not called and no constant
normalizer correction is applied.

The first arm `c01` uses the same 512/64 prefix partition, 64-point rotation,
authorities, weights `(100,100,1)`, checkpoint rule, validation, 256 updates,
and resource caps. Learning rate `1e-3` is a warm-start hypothesis from the
same-model historical core-tangent point fit, not a default; all other settings
come from `q01`. If `c01` is finite but does not reach point/global feasibility,
run `c02` at `3e-4`. If it reaches feasibility but calibration is still moving
at the endpoint, review one bounded duration extension; otherwise reject the
core-affine family. Calibration and validation cannot enter the gradient or
change gates.

Evidence contract: compare with `q01/q03` to ask whether the correct
current-basis core-affine parameterization repairs the observed initialization
and generalization failure. Point/global infeasibility, invalid authorities,
nonfinite gradients, reload/XLA mismatch, memory, and wall are vetoes. Prefix
fit/calibration/validation are the primary discriminator. Passing still opens
only off-origin density repair, not T1 admission or HMC.

Skeptical audit: this phase removes a demonstrated basis mismatch rather than
retrying stale coefficients; it holds the target, data roles, score operators,
and gates fixed; it tests the working fixed-pipeline derivative structure; and
its only tuning ladder is an explicit same-model LR hypothesis. Verdict:
implement current-basis mechanics and run `c01` after focused tests.

Exact trusted GPU/XLA command:

```bash
TF_FORCE_GPU_ALLOW_GROWTH=true MPLCONFIGDIR=/tmp/matplotlib-cache \
/home/chakwong/anaconda3/envs/tf-gpu/bin/python \
scripts/run_zhao_cui_austria_sir_parameter_density_t1.py \
--rotating-prefix-tangent c01_core_affine_zero_lr1e3_steps256 \
--output-dir docs/plans/artifacts/zhao-cui-austria-sir-parameter-density-t1-20260801/core-affine-prefix-tangent-v1/c01_core_affine_zero_lr1e3_steps256 \
--max-seconds 1800
```

Skeptical audit: the baseline is the actual `d05` rank-7 child, not a weak
random initializer; the prefix gate remains the promotion discriminator and is
not replaced by training loss; calibration and validation remain disjoint from
the gradient; no stale rank/global-weight hypothesis is run; the authority
sample count, seeds, point schedule, checkpoint rule, wall cap, 6 GiB cap, and
continuation vetoes are explicit. The run could still pass 64 calibration and
three validation points without proving global prefix accuracy, so even a pass
opens density repair rather than T1 admission. Verdict: the plan answers the
observed failure and is fit to execute after focused schedule/checkpoint tests.

`q01` completed in `229.31` seconds with a `833,680,640` byte allocator peak;
all authority, reload, XLA, memory, wall, point, and global gates passed. The
512 fit and 64 calibration indices were unique and disjoint. It is rejected at
the prefix gate, but wider coverage improved the independent validation-prefix
residual from `d05` `[3.07,0.851,14.64]` to
`[3.50,5.11,6.87]`. Calibration maximum fell monotonically from `154.26` at
the initializer to `38.17` at update 256; calibration mean squared residual
fell from `413.91` to `81.84`. Point RMS improved to
`[0.496,0.453,0.599]`, global residual remained below `0.10`, and the final
update was the selected feasible checkpoint. The fit-pool maximum remained
`59.61`, so no prefix gate passed.

This is an under-budgeted candidate trajectory, not a rotating-coverage or
rank-7 rejection: the independent criterion had not plateaued or reversed when
the frozen update budget ended. Run one exact budget extension `q02` from the
same initializer, optimizer, pool, schedule, weights, and seeds with 1,024
total updates. Rerun from the initializer rather than restarting Adam from the
saved child, because the latter would silently discard optimizer state and
define a different trajectory. The existing calibration checkpoint rule makes
the longer arm safe against late deterioration.

`q02` retains the 30-minute and 6 GiB caps. It is the terminal same-objective
budget discriminator: if its best feasible calibration checkpoint still fails
the fit, calibration, or validation prefix gates, do not add more steps under
this objective. A valid failure then triggers an objective/family audit rather
than a third duration arm. This extension is justified by the observed
monotone endpoint, not promoted as a default. All evidence roles and nonclaims
remain unchanged.

Exact trusted GPU/XLA command:

```bash
TF_FORCE_GPU_ALLOW_GROWTH=true MPLCONFIGDIR=/tmp/matplotlib-cache \
/home/chakwong/anaconda3/envs/tf-gpu/bin/python \
scripts/run_zhao_cui_austria_sir_parameter_density_t1.py \
--rotating-prefix-tangent q02_r7_pool512_batch64_steps1024 \
--output-dir docs/plans/artifacts/zhao-cui-austria-sir-parameter-density-t1-20260801/rotating-prefix-tangent-v1/q02_r7_pool512_batch64_steps1024 \
--max-seconds 1800
```

`q02` exactly replayed every `q01` metric through update 256, establishing
deterministic trajectory identity. It completed in `381.89` seconds with a
`833,901,568` byte allocator peak and selected update 864. Point RMS
`[0.354,0.346,0.501]`, global residual below `0.047`, authority, reload, XLA,
memory, and wall gates passed. Fit-pool maximum prefix residual remained
`40.77`, calibration remained `24.05`, and validation prefix was
`[4.57,7.82,9.89]`. The calibration maximum plateaued around 24 after update
864 while validation worsened from `q01`. More updates under this initializer
and mean-squared objective are rejected.

### Core-Tangent Rank-8 Initialization Repair

The phrase "rank 7" overstated the tested capacity. The additive rank-2
component was zero-padded and given only five small diagonal rank-one paths;
the zero off-diagonal channels were not a target-informed generic rank-7
state. The observed rank-4/rank-7 trajectory equality therefore does not prove
that a fully coupled rank-7 or rank-8 tangent is ineffective.

Use the ungauged historical core-tangent artifact
`zhao-cui-austria-sir-lane-b-t1-score-20260731/pilot-01/s05_lr1e3_l1_1e9`
as a warm start only. Its point-only training is historical and ineligible for
admission, but it is same-parent evidence: before the later forbidden gauge it
gave validation point RMS `[0.963,0.849,0.0839]` and preserved the exact T1
parent. Verify the historical manifest parent identity and every tangent tensor
hash before use; record the manifest and tensor hashes in the new result. Do
not invoke its stale artifact loader and do not treat its old identity or
metrics as current evidence.

For parent cores `C_j` and one parameter's core tangents `D_{j,p}`, define

\[
 H_p(r)=\sum_{j=1}^{36}
 C_1(r_1)\cdots C_{j-1}(r_{j-1})D_{j,p}(r_j)
 C_{j+1}(r_{j+1})\cdots C_{36}(r_{36}).
\]

This is exactly the origin derivative of the working core-affine fixed-TT
pipeline. Encode it as a rank-`2R=8` block TT: first core `[C_1,D_1]`, middle
cores `[[C_j,D_j],[0,C_j]]`, and last core `[D_{36};C_{36}]`. Prove exact
point evaluation and global/prefix cross-operator parity against explicit
product-rule sweeps before execution. This is a meaningful fully coupled
rank-8 initializer, not a new score definition or a post-fit gauge.

Run `q03` with exactly the `q01` 512/64 prefix partition, 64-point rotating
schedule, 8,192-draw authorities, `(100,100,1)` weights, LR `1e-5`, 256
updates, calibration checkpoint rule, validation data, 30-minute cap, and
6 GiB cap. Only the initializer and resulting exact rank 8 change. The
historical gauge operation remains forbidden; the same finite child must pass
point, global, fit-prefix, calibration-prefix, and validation-prefix gates.

Evidence contract: compare `q03` with `q01` to ask whether target-informed
fully coupled tangent initialization repairs prefix generalization. A pass
still licenses only quadratic off-origin density repair. A valid failure
rejects this warm-started rank-8 hypothesis, not the analytical score
operators, parent value, or all structured core-affine families. No rank,
learning-rate, loss, authority, or validation change may be introduced inside
the arm.

Skeptical audit: the historical child is used only as a hash-verified warm
start; its stale identity and point-only metrics cannot promote. The block-TT
identity is algebraically testable. The baseline and all data roles are fixed,
and the changed variable is precisely the initialization defect exposed by the
previous arms. Verdict: implement and test the block construction, then run
one `q03` arm.

Exact trusted GPU/XLA command:

```bash
TF_FORCE_GPU_ALLOW_GROWTH=true MPLCONFIGDIR=/tmp/matplotlib-cache \
/home/chakwong/anaconda3/envs/tf-gpu/bin/python \
scripts/run_zhao_cui_austria_sir_parameter_density_t1.py \
--rotating-prefix-tangent q03_r8_core_tangent_warm_start_steps256 \
--output-dir docs/plans/artifacts/zhao-cui-austria-sir-parameter-density-t1-20260801/rotating-prefix-tangent-v1/q03_r8_core_tangent_warm_start_steps256 \
--max-seconds 1800
```

## Compute And Attempt Budget

- one rank-4 batch-64 capacity probe, wall cap 10 minutes;
- at most seven Stage A arms, wall cap 35 minutes each;
- at most three predeclared capacity-repair arms, wall cap 45 minutes each;
- one validation selector;
- one untouched claim, wall cap 60 minutes;
- total campaign wall budget 8 hours inside the active 10-hour goal;
- one RTX 4080 SUPER, FP64 TensorFlow, XLA JIT, verified memory growth, 6 GiB
  TensorFlow allocator peak cap; and
- every attempt writes a fresh directory and preserves failures.

A localized harness/serialization failure may be repaired and retried within
this budget if target, data roles, method, thresholds, hardware, and privacy
remain unchanged. A capacity veto, invalid authority, corrupted parent, or
budget exhaustion stops the campaign.

## Default And Assumption Audit

| Choice | Provenance/status | Failure mode | Early diagnostic |
|---|---|---|---|
| Parent frame/basis/tau/shift | Closed admitted baseline | Drift invalidates value and all comparisons | Identity/core/value check before every run. |
| Rank 2 and 4 | Compact baseline and parent-rank hypothesis | Rank 2 underfits; rank 4 still inadequate | Paired validation shape and score fields. |
| Radii 0.01 and 0.03 | Local finite-family hypotheses | Too small identifies only noise; too large aliases curvature | Symmetric signs and common validation at both radii. |
| Linear features | Minimal origin-identifying hypothesis | Cannot model curvature | Radius-transfer veto; pure quadratics only as repair. |
| Adam `1e-4/3e-4` | Existing parent family plus Phase 2 controlled diagnostic | Divergence or slow fit | First-step finite/gradient checks and loss trace. |
| L1 `0/1e-9/1e-8` | Lane policy and parent tuning scale | Zero overfits; large L1 suppresses residual | Disjoint validation and zero-L1 comparator. |
| Derivative weight `0/0.1` | Auxiliary hypothesis | Proxy dominates density or zero arm misses score | Same untouched criterion; zero comparator mandatory. |
| 96 steps, batch 64 | Bounded first campaign hypothesis | Undertraining mistaken for capacity failure | Loss/gradient trace; failed valid arm may trigger longer repair within budget. |
| Random TT amplitude `1e-3` | Rejected convenience choice | Nearly constant residuals seed a near-zero score and can make a valid rank look incapable | Replace with a training-only exact finite-child score prefit; record its objective, steps, learning rate, ESS, and before/after score residual. |
| Mass allowance 0.05 | Frozen approximation tolerance | Allows a biased normalizer | Report MCSE and raw error separately; primary score and shape/prefix gates remain. |
| Prefix points by fixed first indices | Convenience choice frozen before evaluation | Unrepresentative points | No dropping; result is scoped to declared points and does not prove global prefix accuracy. |
| Rotating 512-point prefix pool | Target-specific coverage hypothesis after fixed-point non-transfer | Still too sparse, correlated, or hard for rank 7 | Disjoint 64-point calibration and unchanged validation prefix veto. |
| 8,192 conditional draws per prefix | Inherited informative-authority setting; retained as a reviewed baseline | MCSE may hide error or make 576 authorities costly | ESS/MCSE validity veto and 30-minute wall cap. |
| 256 rotating updates, LR `1e-5` | 32 pool epochs and fourfold total prefix point-exposures versus `d05` | Undertraining or checkpoint noise | Eight-update calibration trace and restored best feasible checkpoint. |
| Calibration-minimum checkpoint | Explicit optimizer-depth selection rule | Overfits calibration or favors one outlier | Validation stays gradient-free; report maximum and mean residual; untouched stays sealed. |

## Pre-Mortem

The campaign could pass while misleading us if the three prefix points are easy,
the likelihood Fisher interval is wide, or the fixed frame hides tail errors.
The result therefore reports MCSE/ESS, paired off-origin shape at both radii,
and explicit scope nonclaims. It could fail for optimizer or rank reasons rather
than the centered-density idea; the staged ladder and repair triggers separate
those cases. It could fail for an invalid authority; ESS, MCSE, independent
seeds, and no-point-dropping rules distinguish that from candidate rejection.

## Skeptical Pre-Execution Audit

| Risk | Audit verdict and repair |
|---|---|
| Wrong baseline | Passed: only the admitted Lane-B T1 parent is used; the failed historical tangent, APF, source replica, retained grid, ALS, GenUT, SGQF, and UKF are excluded. |
| Proxy promoted | Passed: training/density metrics select or veto; untouched Fisher score is primary. |
| Missing stop condition | Passed: parent, authority, origin, prefix, XLA, memory, and budget continuation vetoes are explicit. |
| Unfair comparison | Passed: every arm shares training/validation streams and common validation theta rows. |
| Hidden defaults | Repaired: rank, radius, features, LR, L1/L2, derivative weight, steps, batch, seeds, and tolerances are recorded as hypotheses. |
| Stale context | Passed: the historical calibrated tangent failed untouched and is negative evidence only; no artifact or tuning is reused. |
| Environment mismatch | Passed subject to capacity probe: TensorFlow FP64, GPU/XLA, memory growth, and one fixed graph shape per stage. |
| Memory blow-up | Passed subject to capacity probe: independent components and streaming contractions; no block-rank sum, theta grid, or time history. |
| Successful command answers wrong question | Repaired: only a frozen, reloaded child passing untouched Fisher and prefix gates can open T2. |
| Validation leakage | Passed: fresh role-separated seeds; untouched streams are inaccessible to pilot/selector modes. |

Audit verdict after reset: do not run the stale capacity probe or Stage A.
First implement and test a training-only target-informed initializer/prefit for
the exact normalized origin score of the same finite child.  Then refresh the
candidate table and capacity command because prefit policy and cost are new
material controls.  The active campaign authorization remains sufficient once
that revised plan passes its skeptical audit.
