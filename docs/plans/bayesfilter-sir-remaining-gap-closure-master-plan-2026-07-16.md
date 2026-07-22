# SIR Remaining-Gap Closure Master Plan

Date: 2026-07-16

Status: `REVIEWED_EXECUTION_IN_PROGRESS_AMENDED_AFTER_SKEPTICAL_AUDIT`

Predecessor:
`docs/plans/bayesfilter-sir-latent-preclip-law-score-repair-result-2026-07-16.md`

## Research Intent Ledger

| Field | Binding intent |
| --- | --- |
| Main question | After repairing the latent pre-clipping SIR law and same-finite-scalar Contract E score, do LEDH and an independent filtering teacher show statistically supported spatial or `d=18` disagreement, or is their comparison inconclusive; and can the separate GPU, identity, hash, and Zhao--Cui engineering gaps be closed? |
| Candidate | `contract_e_chol_latent_preclip_sir_candidate_v1` on the latent simulator-law target. |
| Independent mechanism | An online all-pairs `O(N^2)` particle filtering-score recursion using normalized backward kernels and local complete-data scores, with replicated fixed-seed runs and uncertainty intervals. |
| Expected failure mode | LEDH is internally differentiable but biased in value or score; the `J=2` spatial teacher is under-resolved; route identity omits a dependency; exact-source XLA fails; or fixed-TTSIRT differentiation changes the author route. |
| Promotion criterion | Phase-specific engineering gates may pass. Accuracy phases may reject the current LEDH candidate or remain inconclusive; they cannot positively establish practical equivalence because no target-specific scientific equivalence margin has been justified. Canonical identity is issued only after the exact executed route and prepared-input closure pass. |
| Promotion veto | Nonfinite output, invalid clipping chart, failed teacher lower-rung validation, a statistically supported LEDH--teacher disagreement at a required rung, forgeable/incomplete identity, CPU/GPU mismatch, or source-ungrounded Zhao--Cui behavior. Teacher disagreement is a conservative promotion veto, not proof that LEDH rather than the teacher is wrong. |
| Continuation veto | The independent teacher fails its `J=1` oracle validation; artifacts are corrupted; the scientific target changes; or the total campaign budget is exhausted. A failed LEDH candidate is a promotion veto and repair trigger, not automatically a research-direction veto. |
| Repair trigger | Localized score-term omission, proposal/weight error, graph/XLA defect, identity-closure omission, deterministic-fit derivative omission, or stale serialization evidence. |
| Forbidden conclusion | Same-scalar FD alone does not prove filtering accuracy. A `J=1` or `J=2` pass does not prove `d=18`. A finite fixed-TTSIRT extension is not source-faithful unless paper and author-code anchors match the executed operations. No phase alone proves HMC or leaderboard readiness. |

## Target And Independent Score Mathematics

The target remains the latent pre-clipping model

\[
z_0\sim \mu_\theta,\qquad
z_t\mid z_{t-1}\sim
f_\theta\!\left(\cdot\mid C_{t-1}(z_{t-1})\right),\qquad
y_t\mid z_t\sim g_\theta\!\left(\cdot\mid C_t(z_t)\right),
\]

where `C_0` is the identity and `C_t` clips susceptible coordinates for
`t>=1`.

For an independently generated weighted particle cloud
`{z_t^i,W_t^i}_{i=1}^N`, define the backward kernel

\[
B_t(i,j)=
\frac{W_{t-1}^j f_\theta(z_t^i\mid C_{t-1}(z_{t-1}^j))}
{\sum_{k=1}^N W_{t-1}^k
 f_\theta(z_t^i\mid C_{t-1}(z_{t-1}^k))}.
\]

The online `O(N^2)` score mark is

\[
T_0^i=\nabla_\theta\log\mu_\theta(z_0^i)
      +\nabla_\theta\log g_\theta(y_0\mid C_0(z_0^i)),
\]

\[
T_t^i=\nabla_\theta\log g_\theta(y_t\mid C_t(z_t^i))
 +\sum_{j=1}^N B_t(i,j)
 \left[T_{t-1}^j+
 \nabla_\theta\log f_\theta(z_t^i\mid C_{t-1}(z_{t-1}^j))\right].
\]

The score estimate is `sum_i W_t^i T_t^i`. This is an independent
filtering-score estimator, not the derivative of the LEDH finite scalar. Its
Monte Carlo uncertainty must therefore be reported rather than hidden behind a
same-scalar FD threshold.

The teacher uses a bootstrap particle filter. At `t=0`, particles are drawn from
the initial law and weighted by `g_theta(y_0|z_0)`. For `t>=1`, ancestors are
systematically resampled from `W_{t-1}`, propagated through the latent Gaussian
transition, and weighted by the observation density. The likelihood estimate is
the sum of log mean incremental weights. Resampling uniforms and Gaussian noises
are stateless and seed-bound. LEDH and teacher runs use matched root seeds where
their random inputs have common semantics, but each method retains its own exact
algorithm.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Scientific question | At `J=1`, does the independent teacher reproduce the dense reference within quantified uncertainty? At `J=2` and `d=18`, do LEDH and that teacher show statistically supported disagreement under particle/replicate refinement? |
| Baselines | Refined dense `J=1` quadrature; independent `O(N^2)` score teacher; no-reset LEDH ablation; historical raw-barycentric route only as a wrong-target diagnostic; Zhao--Cui author paper and source for its separate comparator lane. |
| Primary accuracy criterion | At `J=1`, teacher bias is smaller than its 95% interval plus dense-reference numerical uncertainty and the simultaneous interval widths meet the numeric resolution gate below. At later rungs, matched-seed LEDH-minus-teacher simultaneous intervals determine whether a difference is statistically supported. No equivalence or practical-accuracy pass is claimed without a separately justified target-specific margin. |
| Engineering criteria | Same-scalar AD/FD, fail-closed charts, exact prepared-input hashes, functional loops, XLA compilation, and CPU/GPU agreement. These cannot replace the accuracy criterion. |
| Hard vetoes | Invalid teacher lower rung, nonfinite output, zero/negative normalization, clipping-boundary violation, missing score term, source-anchor failure, identity forgery, or exact-source GPU mismatch. |
| Explanatory only | Runtime, ESS, particle-count trends, observed mean differences without intervals, FD error once derivative wiring already passes, and graph size. |
| Artifacts | Versioned files under `docs/benchmarks/artifacts/sir_remaining_gap_closure_20260716/`, phase results under `docs/plans`, and review records under `docs/reviews`. |

## Horizon Convention

Throughout this plan, `T` is the number of observations, matching the production
SIR preparation and execution harnesses. Thus `T=1` contains only `t=0`, while
`T=2` contains `t=0,1` and one transition. The dense-reference helper's
`time_steps` argument counts transitions, so Phase 2 must call it with
`time_steps=T-1`. Artifacts must record both names and reject a mismatch.

## Execution Amendment After Horizon Audit

The pre-Phase-2 skeptical audit found that the reviewed text had inherited the
dense helper's transition-count convention while the teacher and production
harness use observation count. Recomputing the deterministic refinement at the
correct horizons also showed that the former width gate was not scientifically
meaningful: it required particle Monte Carlo half-widths to be as small as
deterministic quadrature refinement differences. At `T=1` those differences are
about `3e-9` for value and `1.7e-8` for score, which is not a justified accuracy
target for a finite replicated particle estimator.

The amendment preserves the dense-reference containment check as a mismatch
screen, reports interval widths without silently converting them into a
promotion threshold, and forbids a positive equivalence or certification claim.
A largest-rung interval excluding the dense reference is evidence of a teacher
mismatch and a continuation veto. An interval containing the dense reference
means only that no disagreement was detected at current precision. The latter
keeps the teacher viable as an independent disagreement comparator under the
later phases' already limited epistemic claims; it does not prove unbiasedness,
convergence, practical equivalence, HMC readiness, or leaderboard accuracy.

Correct-horizon deterministic refinement diagnostics are frozen before teacher
outcomes are inspected:

| Horizon | Dense value at order 33/radius 7 | Dense score | `u_value` | `u_score` |
| ---: | ---: | --- | ---: | ---: |
| `T=1` | `-0.37337136725883546` | `[0,0,-0.516089350201728]` | `2.9808919776996845e-9` | `1.6652841994257983e-8` |
| `T=2` | `-0.8570589548006784` | `[-0.00013728907649588102,0.0043448641619948086,-1.0261056979707353]` | `8.866663376849715e-7` | `1.323006956610584e-5` |

Here each `u` is the maximum of the order-29-to-33 and radius-6-to-7
refinement differences. These are diagnostics, not rigorous error bounds.

## Default And Assumption Audit

| Choice | Provenance | Justification | Failure mode | Early diagnostic | Status |
| --- | --- | --- | --- | --- | --- |
| `O(N^2)` backward-score recursion | Poyiadjis et al. (2011) and the particle derivative literature already cached locally | Retains progressive filtering-score information without smoothing and is independent of LEDH differentiation | Path/weight convention implemented incorrectly | `J=1,T=1,2` against dense manual score and local-term ablations | hypothesis until Phase 2 passes |
| Replicates `R=16` | Owner previously selected audit count 16 | Enables paired Student intervals without pretending one seed is accuracy evidence | Low power or unstable tails | report interval width and effective finite replicate count; increase only within budget if inconclusive | reviewed campaign baseline |
| Particle ladder `N=64,128,256` | bounded `O(N^2)` cost hypothesis | tests bias/variance trend before expensive rows | insufficient for `d=18` | `J=1` convergence and wall-time measurement | hypothesis, not default |
| `J=2` instead of dense 4D tensor quadrature | skeptical feasibility audit | order-33 Cartesian filtering would require prohibitive pairwise state grids | teacher itself biased | validate same implementation at `J=1`, then particle/replicate refinement at `J=2` | reviewed replacement for infeasible dense grid |
| 95% Student intervals | standard repeated-run uncertainty summary | matches the stated statistical confidence target | correlation from reused randomness | independent replicate seeds; paired common observations only where declared | reviewed statistical procedure |
| Bonferroni family | three score coordinates plus value at each promoted rung | controls simultaneous comparisons within one rung | over-conservative intervals | report adjusted and unadjusted intervals | reviewed diagnostic family |
| Canonical identity after exact-route engineering | Contract E repository policy | identity proves route binding, not scientific correctness | identity mistaken for admission evidence | separate identity and numerical ledgers; identity phase continues even if accuracy promotion fails | reviewed sequencing |
| XLA/GPU attempts `<=3` | predecessor exhausted its old budget; this is a new authorized successor campaign | enough for one run and localized retries | compiler repair consumes all attempts | CPU compile/smoke and graph audit before GPU | reviewed budget |
| Fixed-TTSIRT differentiation | Zhao--Cui Algorithms 2/5 and author `full_sol.m`/`pre_sol.m` | comparator must preserve recursive marginal, fit, map, and proposal operations | clean-room implementation silently changes method | source-operation ledger and lower-rung source replay | hypothesis; source gate precedes code |

## Skeptical Pre-Execution Audit

The audit checked wrong baselines, proxy promotion, hidden defaults, stop
conditions, unfair comparisons, stale context, environment mismatch, and
whether artifacts answer the stated question.

Material revisions made before execution:

1. The predecessor's proposed dense `J=2` tensor grid is computationally
   unsuitable: a four-dimensional grid followed by pairwise filtering grows far
   beyond a bounded reference run. It is replaced by the independently validated
   `O(N^2)` teacher with replicated uncertainty.
2. Same-scalar FD remains an engineering check only. Accuracy promotion uses an
   independent estimator and intervals.
3. Canonical identity is moved after exact-source GPU engineering but is not
   conditional on an accuracy pass. Identity binds the route; it does not certify
   the route's approximation quality.
4. The generic retained-grid multistate route is forbidden for Zhao--Cui
   production comparison. The phase must trace Algorithms 2/5 and exact author
   operations in `full_sol.m`, `pre_sol.m`, `precond.m`, and TTSIRT map calls.
5. A failing LEDH accuracy rung triggers feature/particle/reset repair. It does
   not justify altering the teacher or widening an agreement threshold after the
   result is seen.
6. The historical aggregate tensor hash failure is audited before regenerating
   anything. Matching displayed scalars is not sufficient reason to overwrite a
   certificate.
7. The plan cannot validly claim positive practical equivalence because no
   SIR-specific value/score margin is scientifically justified. The accuracy
   question is therefore mismatch detection versus inconclusive evidence. HMC
   and leaderboard accuracy admission remain blocked even if intervals include
   zero.
8. Teacher accuracy is externally certified only at `J=1`. At `J=2` and
   `d=18`, intervals support statements about LEDH--teacher disagreement only.
   They do not identify which method is closer to the latent filtering target
   and cannot by themselves establish target-level bias.

Audit verdict before external review: `PASS_AFTER_REVISION`.

Claude read-only review converged on iteration 4 with `VERDICT: AGREE` after
repairs to the equivalence boundary, numeric `J=1` gate, later-rung teacher
epistemic boundary, teacher-refinement diagnostic, and phase ordering. Review
record:
`docs/reviews/bayesfilter-sir-remaining-gap-closure-plan-claude-review-iter1-2026-07-16.md`.

## Source-Support Ledger For Zhao--Cui Lane

| Source | Classification | Inspected anchors | Allowed support | Forbidden support |
| --- | --- | --- | --- | --- |
| Zhao and Cui, JMLR 2024, local PDF SHA-256 `c547b9af...` | direct method | equations (15)--(23), Algorithms 2--3, Proposition 2, Algorithm 5, SIR equations (37)--(39) and numerical setup | recursive squared-TT density, marginalization, conditional KR proposal, importance correction, SIR benchmark setup | local derivative implementation correctness or HMC readiness |
| `third_party/audit/.../eg3_sir/mainscript.m` SHA-256 `3b0460ad...` | author executable | lines 12--17, 37--56, 63--65 | `d=0,m=18,T=20`, basis/TT settings, `full_sol` route | parameter-gradient route; the author SIR example fixes parameters |
| `models/full_sol.m` SHA-256 `486b14cb...` | author executable | lines 21--43, 46--130, especially prior marginal at 72--80, fitting at 90--124, `eval_irt`/`eval_pdf` correction | actual recursive marginal, fixed-fit adaptation target, map and proposal operations | differentiating through adaptive randomness without freezing it |
| `models/pre_sol.m` SHA-256 `575c40ec...` | author executable | lines 16--127 and preconditioning continuation | optional source preconditioned route | claiming plain `full_sol` and `pre_sol` are identical |
| `models/tensordot/precond.m` SHA-256 `4157e9b5...` | author executable | setup and coordinate-map operations | source preconditioning semantics | scientific correctness by code authority alone |

Network metadata and forward-citation checks are not needed for this bounded
implementation decision. Retraction/erratum status is not checked online in
this campaign; therefore the paper supports inspected algorithm operations only,
not a literature-completeness claim.

## Campaign Budget

- CPU/reference wall time: at most 6 hours total.
- Trusted GPU/XLA: at most 3 attempts and 2 hours total.
- Teacher replicates: start `R=16`; at most `R=32` only when a declared interval
  is inconclusive and remaining CPU/GPU budget permits.
- Particle ladder: start `N=64,128,256`; `N=512` is allowed only after measured
  cost and a documented need.
- Every serious launch uses a fresh output directory and records command,
  commit, environment, device, seeds, wall time, plan, and result.

## Phase 0: Historical Hash Audit

Objective: determine why the persisted Contract E Phase 3 aggregate tensor hash
changed while seven numerical diagnostic fields remained equal.

Required artifacts:

- a byte-level tensor-field hash comparison;
- source/artifact commit and serialization-method comparison;
- classification as stale artifact, serialization nondeterminism, source change,
  or unresolved corruption.

Gate: do not rewrite the old artifact unless the exact cause and compatibility
policy are documented. This phase cannot block new SIR numerical work unless it
reveals a shared Contract E numerical defect.

## Phase 1: `O(N^2)` Teacher Implementation

Objective: implement a TensorFlow float64 all-pairs filtering value/score teacher
with functional time loops and fixed-seed replicate support.

Required checks:

- transition, observation, and initial score terms independently match autodiff
  of their local densities;
- backward kernels are row-normalized and finite;
- `T=1` reduces to direct normalized local-score averaging;
- stopped previous-mark and stopped transition-score controls fail at `T=2`;
- no LEDH or Contract E implementation is called by the teacher.

Gate: focused tests pass before any cross-method comparison.

## Phase 2: `J=1` Teacher Certification

Objective: screen the independent teacher against the refined dense `J=1`
value/score reference at observation counts `T=1,2`, without claiming positive
equivalence.

Execution ladder: `N=64,128,256`, `R=16`, with independent replicate seeds.

For each horizon, use the frozen correct-horizon dense value, score, `u_value`,
and `u_score` from the execution amendment above.

Gate at the largest executed `N`:

- all replicate estimates, normalizers, and backward kernels are finite;
- each Bonferroni-adjusted 95% teacher interval, expanded by the applicable
  diagnostic `u_value` or `u_score`, is reported against the dense reference;
- an expanded interval excluding the dense reference emits
  `BLOCK_TEACHER_J1_DISAGREEMENT` and is a continuation veto;
- if all expanded intervals contain the dense reference, emit only
  `NO_TEACHER_J1_DISAGREEMENT_DETECTED_AT_CURRENT_PRECISION`.

Interval half-widths and particle-count trends are mandatory diagnostics but do
not become a positive equivalence gate without a separately justified
SIR-specific margin. `R=32` or `N=512` may be used only to resolve an observed
largest-rung disagreement or numerical instability within budget, not to chase
quadrature-scale Monte Carlo width. An interval containing the reference cannot
emit a teacher-certification, practical-equivalence, HMC, or leaderboard token.

## Phase 3: Exact-Source GPU/XLA Certificate

Objective: run the exact current candidate source on the bound preparation with
XLA JIT on trusted GPU, after an escalated device probe.

Gate: XLA compilation logged; `/GPU:0` recorded; one compiled functional route;
finite value/score; all charts valid; same prepared hashes; CPU/GPU value and
score agree to dtype-appropriate numerical precision. This gate certifies
execution only, not accuracy.

## Phase 4: Canonical SIR Route Identity

Objective: register a repository-owned non-overridable Contract E--Chol SIR
route specification bound to reset semantics, derivative composition, row-mass
normalization, residual design, ridge policy and realized input, prepared tensor
identity, JIT setting, and source dependency closure.

Gate: forgery, monkeypatch, callable substitution, omitted prepared field, wrong
JIT, and source-drift tests fail closed. Identity issuance is still distinct from
scientific admission and may complete even when the accuracy ledger blocks
promotion.

## Phase 5: `J=2` Spatial Comparison

Objective: test the first spatially coupled latent SIR target at `T=1,2,5` using
the `J=1`-certified teacher as an independent comparator and paired LEDH runs.

Teacher refinement rule: use matched replicate roots at `N=128` and `N=256`.
For value and each score coordinate, compute a Bonferroni-adjusted paired 95%
interval for `teacher_N256 - teacher_N128`. If any interval excludes zero, emit
`TEACHER_N_REFINEMENT_SHIFT_DETECTED`; the LEDH comparison remains descriptive
only and cannot emit a disagreement veto. If every interval contains zero, the
result may say `NO_TEACHER_N_REFINEMENT_SHIFT_DETECTED_AT_CURRENT_PRECISION`,
not that the teacher is unbiased or converged.

Gate: the no-detected-shift token is present and matched-seed LEDH-minus-teacher
simultaneous intervals are reported. An interval excluding
zero is statistical evidence of method disagreement and triggers feature,
particle, reset, flow, and teacher-bias diagnostics. It is a conservative LEDH
promotion veto but does not identify which method is wrong. An interval
including zero means the methods are not
distinguishable at the current precision; it is not an equivalence claim. No
threshold is relaxed after observing the data. This phase has no positive
accuracy-admission token.

## Phase 6: `d=18` Method-Disagreement Ladder

Objective: quantify LEDH-versus-teacher value/score disagreement at `T=2,5,20`.

Execution order: `T=2` first, then `T=5`, then `T=20` only if earlier rungs are
valid and budget remains. Use paired observations and declared independent
teacher seeds. Report value and each score coordinate with simultaneous
intervals.

Gate: no HMC or leaderboard promotion if a statistically supported score
disagreement persists. A failed rung must produce separate LEDH and teacher-bias
diagnostics, a focused retry, or a scientific blocker record. The result may say
only that the methods disagree; it may not assign target error without another
oracle. Inconclusive intervals are reported as inconclusive, not as accuracy
passes. Exact-source GPU and identity gates are inherited from Phases 3--4.
This phase has no positive practical-equivalence token.

## Phase 7: Zhao--Cui Fixed-TTSIRT Total Derivative

Objective: implement or precisely block the total derivative of the actual
fixed-randomness/fixed-fit adaptation of the author source route.

Required sequence:

1. freeze the exact source operation ledger from Algorithms 2/5 and the cited
   author files;
2. prove lower-rung value replay for prior marginal, reapproximation target,
   fixed linear solves, `eval_irt`/`eval_pdf`, coordinate maps, and correction;
3. differentiate every frozen fit solve using
   `dc=A^{-1}(db-dA c)` and propagate previous-marginal, map, proposal-density,
   and correction derivatives;
4. compare the total score to AD/FD of that same frozen finite source program;
5. only then compare descriptively with Contract E on the same latent target.

Gate: paper and author-code anchors plus total same-scalar score pass. If the
author operation cannot be reproduced without an unapproved route change, emit
`BLOCK_SOURCE_FAITHFUL_FIXED_TTSIRT_TOTAL_DERIVATIVE` and preserve the gap.

## Phase 8: Terminal Synthesis

Write a result with separate engineering, accuracy, identity, GPU, source-
faithfulness, and scientific ledgers; decision and inference-status tables;
manifests; post-run red team; exact remaining gaps; and prohibited claims.

## Phase Close Protocol

At each phase:

1. run focused local checks;
2. write or refresh the phase result;
3. audit whether the result invalidates the harness, target, math, artifact, or
   only the current candidate;
4. refresh the next phase entry conditions;
5. continue automatically when the campaign contract and budget are unchanged.

Stop only for a true continuation veto, exhausted budget, external/irreversible
boundary, or a source/scientific decision requiring owner direction.
