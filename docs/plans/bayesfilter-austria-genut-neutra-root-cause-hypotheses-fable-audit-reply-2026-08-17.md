# Austria GenUT NeuTra Root-Cause Hypotheses: Fable Audit Reply

Date: 2026-08-17

Status: `READ_ONLY_AUDIT_COMPLETE`

Auditor: Fable (claude-fable-5), independent mathematical and code reviewer

Audited artifact:
`docs/plans/bayesfilter-austria-genut-neutra-root-cause-hypotheses-fable-handoff-2026-08-17.md`

Audit basis: read-only inspection of the cited source at commit
`dae37183bf4421682b2ad991e2dc0d0f3c53f260` with a dirty worktree that includes
`cubature_genut_batch_tf.py`, `cubature_genut_neutra_targets.py`,
`higher_moment_contract_e.py`, and `cubature_genut_filter.py`. All line anchors
below refer to the current dirty working-tree state, not the committed state;
Phase 0 source hashing remains mandatory before execution. No source file was
edited, no command was run against a GPU, and no experiment was executed.

---

## VERDICT: REVISE

The hypothesis set is substantially correct. R0 and the H1 asymmetry are
**confirmed by direct source inspection**, the H1 forward/reverse intervention
design is causally sound as labeled, and the dependency order is right. REVISE
rather than AGREE because three material items require plan changes before
execution:

1. H4's expected semantics are wrong relative to the current source: the
   endpoint already fails closed under tangent-only invalidity (finding F3).
   Phase 5's interpretation table must be rewritten from "detect exposure" to
   "regression-guard the existing mask".
2. One additional (minor) primal-program asymmetry exists upstream of the
   higher-moment block and is missing from the hypothesis set (finding F4).
3. The Phase 1 requirement to capture intermediates under graph and XLA modes
   conflicts with the plan's own continuation veto on intrusive
   instrumentation (finding F5). Capture must be eager-only; graph/XLA arms
   must be endpoint-only.

None of these invalidate the research direction. This is candidate-plan
revision, not direction rejection.

---

## Findings Ordered By Severity

### F1 (severity: highest, scientific claim scope) — R0 CONFIRMED: the batch NeuTra target is diagonal-only, not the promoted dual-cap program

Classification: the handoff's route-identity concern is **correct**.

Evidence chain, verified by inspection:

- `bayesfilter/highdim/cubature_genut_batch_tf.py` contains no import of
  `bayesfilter.highdim.higher_moment_contract_e` at all
  (`cubature_genut_batch_tf.py:13-22`; imports are the Contract-E *reset* core
  `ledh_contract_e_reset_tf` and the diagonal LM helpers `genut_shape_lm_tf`
  only).
- `batch_finite_value` (`cubature_genut_batch_tf.py:1094`) and
  `batch_finite_value_score` (`cubature_genut_batch_tf.py:1304`) call only the
  local diagonal routines `_higher_moment_batch_value`
  (`cubature_genut_batch_tf.py:936`) and `_higher_moment_batch_jvp`
  (`cubature_genut_batch_tf.py:793`).
- `GenUTControls` (`cubature_genut_neutra_targets.py:48-58`) exposes only
  diagonal correction controls plus optional LM/trust controls. It has **no
  field** for pairwise steps, pairwise strength, `pairwise_particle_rms_cap`,
  `coordinatewise_standardized_cap`, or cap power. `_core_kwargs`
  (`cubature_genut_neutra_targets.py:315-333`) therefore cannot even express
  the dual-cap configuration.
- The dual-cap stages required by
  `docs/plans/bayesfilter-genut-dual-cap-monograph-ready-spec-2026-08-07.md`
  (pairwise co-skew/co-kurtosis, pairwise row-RMS radial cap 2, standardized
  coordinate cap `b=0.98, p=8`) exist only in
  `higher_moment_contract_e.higher_moment_shape_jvp`
  (`higher_moment_contract_e.py:913-951`, cap controls at
  `higher_moment_contract_e.py:927-934`) and are wired only into the
  non-batch route `cubature_genut_filter.finite_value_score`
  (`cubature_genut_filter.py:916-936`).

Two labeling precisions the revised plan should adopt:

- The batch route **does** perform a final affine restoration
  (`output = mean + standardized @ target_cholᵀ`,
  `cubature_genut_batch_tf.py:1068-1070` and `:901-903`) and a Contract-E
  mean/covariance reset. What is absent is the pairwise stage, the pairwise
  radial cap, and the coordinate cap. The route-identity ledger should say
  "diagonal + affine restoration, no pairwise, no dual caps", not "no affine
  restoration".
- `higher_moment_trust_radius` (`smooth_rms_cap_*`) caps the *diagonal
  displacement* RMS and is set to `0.0` for Austria; it is not the dual-cap
  spec's pairwise row-RMS radial cap. The ledger must not conflate them.

Consequence, stated plainly: the earlier Austria admission failure and the
current value/score mismatch are properties of a
`batch_diagonal_candidate` route. Reporting them as a failure of the promoted
dual-cap algorithm would be **wrong relative to that claim**. The handoff's
two-lane split is the correct response. Note also that the repository default
policy (project `CLAUDE.md`) names the dual-cap family as the promoted
production algorithm, so lane 2 (batch-native dual-cap endpoint) is what an
eventual Austria dual-cap NeuTra claim actually needs; lane 1 repairs shared
infrastructure.

### F2 (severity: high, engineering root cause) — H1 asymmetry CONFIRMED at source level, with exact anchors

Classification: the redundant-restandardization asymmetry is **correct** as
described in E5, and it is the first unequal primal operation on the particle
path visible by inspection.

Verified program shapes:

- **JVP path.** `_higher_moment_batch_jvp` standardizes the reset cloud once
  before its loop (`cubature_genut_batch_tf.py:855-870`). Then **each** call to
  `_shape_iteration_batch_jvp` recomputes mean, covariance, Cholesky, and a
  fresh standardization of its already-standardized input
  (`cubature_genut_batch_tf.py:608-618`) before forming `m3/m4`, the
  directions, and the shape system, and restandardizes again at iteration end
  (`cubature_genut_batch_tf.py:774-790`). Per iteration: two standardizations.
- **Value path.** `_higher_moment_batch_value` standardizes once before its
  loop (`cubature_genut_batch_tf.py:994-997`), forms `m3/m4` and the shape
  system directly from the incoming standardized cloud
  (`cubature_genut_batch_tf.py:1002-1020`), and restandardizes only after
  applying each correction (`cubature_genut_batch_tf.py:1062-1067`). Per
  iteration: one standardization.

In exact arithmetic the extra start-of-iteration standardization is the
identity (the input cloud has exactly zero mean and identity covariance by
construction of the previous restandardization). In FP32 it is not bitwise
idempotent: the recomputed mean is not exactly zero, the recomputed covariance
is not exactly identity, and the Cholesky/triangular-solve round-trip perturbs
every particle. The two functions therefore define different finite arithmetic
programs from the first correction iteration onward, before any tangent
enters. This confirms E5 and is consistent with E4's zero-tangent
first-iteration mismatch.

Element-by-element comparison of the remaining per-iteration primal operations
(m3/m4, residuals, direction3/4, j33–j44, jacobian/residual stacking, the
`lm_damping=0` normal-equation branch, displacement, end-of-iteration
restandardization, and the outer affine restoration) shows **identical
formulas and identical op sequences** between
`cubature_genut_batch_tf.py:1002-1067` (value) and
`cubature_genut_batch_tf.py:619-790` (JVP primal). The uniform/weighted moment
helpers apply `_sym` identically in both paths. I found no second pre-solve
primal asymmetry inside the higher-moment block. H1's forward intervention
(skip the redundant input standardization in the JVP iteration) therefore has
a well-defined target and, if E3's upstream parity holds, should align every
pre-solve primal tensor exactly. H3 remains worth testing only for graph/XLA
fusion-induced reduction reordering, not for source-level formula differences.

Sufficiency remains empirical: the asymmetry is confirmed as the first unequal
primal operation, but "ULP-scale perturbation → ~24 max cloud difference in
one iteration" is a *plausible, not checked* amplification claim. With the
Austria controls (`_AUSTRIA_CONTROLS`,
`cubature_genut_neutra_targets.py:116-126`: `correction_steps=4`,
`strength=0.2`, `lm_damping=0.0`, `trust_radius=0.0`, floor `1e-5`), the
unscaled normal-equation branch (`cubature_genut_batch_tf.py:705-714` JVP,
`:1035-1044` value) is active, and E6's condition estimate `~1.2e5` makes the
amplification arithmetic consistent (relative input perturbation `~1e-7`
times `~1e5` on coefficients of magnitude `~1e3` gives `O(10)` displacement
differences). The Phase 3 paired intervention is the correct and sufficient
discriminator. The reverse intervention is correctly labeled
target-changing/diagnostic-only.

### F3 (severity: medium, changes Phase 5 expectations) — H4 is largely resolved by inspection: the score endpoint fails closed

Classification: the plan's implicit expectation that tangent-only invalidity
might expose a finite value from a different recursion is **wrong relative to
the current source** at the endpoint boundary. Exact behavior:

- In the score loop, `stage_valid` includes tangent finiteness
  (`cubature_genut_batch_tf.py:1449-1460`), and `step_valid` additionally
  includes `restored["valid"]` and `higher["valid"]`, both of which include
  tangent finiteness (`cubature_genut_batch_tf.py:557-575` for the reset,
  `:915-917` for the higher-moment block).
- On any invalid step, particles and tangents are carried unchanged
  (`cubature_genut_batch_tf.py:1507-1512`) and `valid_value & step_valid`
  latches false permanently (`cubature_genut_batch_tf.py:1519`).
- Both returned scalars are masked: value and score are NaN whenever any step
  was invalid (`cubature_genut_batch_tf.py:1612-1615`; value route
  equivalently at `:1301`). The adapter layer re-masks on
  `valid_pre_regularized_score` (`cubature_genut_neutra_targets.py:412-419`).

Therefore a tangent-only failure cannot emit a finite score-bearing value for
a recursion that differs from the endpoint recursion: the pair fails closed as
required. Three genuine residual caveats, which the revised Phase 5 should
target instead:

1. **Endpoint validity asymmetry is by design but must be documented.** The
   value endpoint's `stage_valid` has no tangent checks
   (`cubature_genut_batch_tf.py:1191-1197`), so there exist θ where
   `batch_value_status` is finite while
   `neutra_batch_log_prob_and_grad_status` is NaN. Any consumer that mixes the
   two endpoints across that boundary sees an apparent value mismatch that is
   actually a validity-domain mismatch.
2. **Internal branch divergence exists but is masked.** The Sinkhorn
   `marginal_valid` in the JVP route additionally requires tangent finiteness
   (`cubature_genut_batch_tf.py:269-274`), so the `safe_barycentric` fallback
   (`:473-484`) can select different branches between routes; every such path
   feeds `reset_valid → step_valid → NaN`. The injected-tangent fixture is
   still worth running — as a **regression guard** with predicted outcome
   "NaN pair", not as a candidate root cause for the θ=0 mismatch.
3. **Diagnostics are not NaN-masked.** The diagnostics dict reflects the
   fallback recursion even when the scalar is masked. Any consumer treating
   diagnostics from invalid rows as evidence would be reading a different
   recursion; the status `valid_pre_regularized_score` flag must gate that.

### F4 (severity: low, missing hypothesis) — one additional primal-program asymmetry upstream of the higher-moment block, validity-flag only

New finding, absent from the handoff. In the reset stage:

- `_restore_cloud_batch_value` computes the covariance gap as
  `_sym(target_covariance - transported_covariance)` with **both** covariances
  unsymmetrized (`cubature_genut_batch_tf.py:382-392`).
- `_restore_cloud_batch_jvp` takes `target_cov` from `_weighted_moments_jvp`,
  which applies `_sym` **before** the subtraction
  (`cubature_genut_batch_tf.py:146`, used at `:457-465`).

In exact arithmetic these are equal; in FP32 the rounding sequences differ, so
`minimum_gap_eigenvalue` can differ between routes by roughly ULP scale. This
touches **no particle tensor** — it feeds only the `gap_valid` threshold
(`min_gap + ridge > 0`) — so it cannot explain the observed finite-value
mismatch, and a knife-edge flip fails closed via `pre_valid → step_valid →
NaN`. It should be added to the Phase 2 boundary list (between steps 5 and 6)
and folded into the H3/H4 scope, both for completeness and because it is a
second instance of the same defect class as H1: separately maintained primal
formulas drifting at the rounding level.

### F5 (severity: medium, plan-validity) — Phase 1/2 instrumentation must be eager-only; graph/XLA arms must be endpoint-only

The plan requires the runner to support eager, graph, and XLA modes *and* to
capture every intermediate tensor without changing operations before the
first mismatch. Under `tf.function` and especially under
`jit_compile=True`, adding fetch outputs changes fusion and can change the
compared arithmetic — which fires the plan's own continuation veto
("instrumentation changes the compared arithmetic before the first observed
divergence"). Required revision:

- First-unequal-tensor localization (Phase 2) and the H1 causal arms
  (Phase 3): eager mode with deterministic ops only.
- Graph and XLA arms: compare **endpoint scalars and final particle clouds
  only**, as H3/H6 replication evidence, with no interior capture.

This preserves every discriminating comparison while removing the arm that
could not have answered its question.

### F6 (severity: informational, lowers H5 prior) — the Austria direct derivative terms named in review question 6 are present and correct by derivation

Verified in `parameterized_austria_sir_batch_adapter`
(`cubature_genut_batch_adapters.py:189-293`), with
`κ = 0.1·e^{θ_0}`, `ν = 18·e^{θ_1}`, `σ² = 100·e^{2θ_2}`
(`cubature_genut_batch_adapters.py:201-206`):

- **∂κ/∂θ₀ = κ.** The infection tangent adds
  `infection · e₀` (`cubature_genut_batch_adapters.py:217-219`); since
  `infection = κ S I` and `∂κ/∂θ₀ = κ`, the direct term
  `∂(κSI)/∂θ₀ = κSI = infection` is exactly what is coded. Correct.
- **∂ν/∂θ₁ = ν.** `d_rhs_i` adds `-(ν·I) · e₁`
  (`cubature_genut_batch_adapters.py:223-229`); `∂(-νI)/∂θ₁ = -νI`. Correct.
- **∂σ²/∂θ₂ = 2σ².** Per observation coordinate,
  `ℓ = -½[log 2π + log σ² + r²/σ²]`, so
  `∂ℓ/∂θ₂ = -½[2 - 2r²/σ²] = r²/σ² - 1`, matching
  `direct = Σ(r²/σ² - 1)` times `e₂`
  (`cubature_genut_batch_adapters.py:279-282`). Correct.
- The state-tangent observation term `Σ r·t/σ²` and the RK4 tangent recursion
  (forward JVP of the four-substep RK4 map,
  `cubature_genut_batch_adapters.py:235-244`) are structurally correct;
  additive process noise is θ-independent, so its zero tangent is correct, and
  the θ-independent initial condition correctly has zero `initial_tangent`.

This is a *local formula* verification, not a validation of the composed
20-step tangent recursion; H5/Phase 6 (autodiff block comparison plus the h²
regression) remains required, but its prior probability as the mismatch cause
is now lower. Note also both loops obtain the primal from
`adapter.transition_value` (the score loop does not reuse the tangent call's
primal), consistent with E2.

### F7 (severity: low, test economy) — redundant or confirmatory tests

- Phase 2 steps 1–5 (initial particles through Contract-E restored particles)
  re-verify what E2/E3 already observed and what F2's inspection confirms are
  identical op sequences. Keep them — they are cheap and pin the upstream
  baseline — but classify them as confirmatory regression, not discovery.
- The H4 injected-tangent exposure test discriminates, but its predicted
  outcome under the current source is "masked NaN pair" (F3). Rewrite the
  Phase 5 decision-rule row accordingly: `CONFIRM_H4_BRANCH_DEFECT` should
  fire only if a finite scalar escapes the mask, which inspection predicts
  will not happen.
- No proposed test is non-discriminating for its stated hypothesis except the
  graph/XLA interior-capture arms removed under F5.

---

## Answers To The Ten Required Review Questions

1. **Redundant restandardization asymmetry?** Confirmed at source level.
   `_shape_iteration_batch_jvp` restandardizes its already-standardized input
   at `cubature_genut_batch_tf.py:608-618`; `_higher_moment_batch_value`
   standardizes once (`:994-997`) and restandardizes only post-correction
   (`:1062-1067`). See F2.
2. **Sufficient, or earlier unequal primal tensor?** No earlier unequal
   *particle* tensor is visible by inspection; the only earlier unequal primal
   arithmetic is the validity-only covariance-gap symmetrization (F4), which
   cannot move a particle. The asymmetry is confirmed as the first unequal
   primal operation; its quantitative sufficiency for the observed `~24`
   one-iteration cloud difference is plausible but not checked — exactly what
   the Phase 3 paired intervention discriminates.
3. **Conditioning as amplifier, not initiating mismatch?** Correctly
   classified. The unscaled normal solve with floor `1e-5`
   (`cubature_genut_batch_tf.py:705-714`, `:1035-1044`) squares κ(J) and is
   *identical code* in both routes; identical inputs would give identical
   FP32 outputs under deterministic ops. It can only amplify an upstream
   difference, not create one. H2's falsification criteria are sound.
4. **H1 interventions causal and labeled correctly?** Yes. The forward
   intervention (JVP iteration consumes the same standardized tensor the
   value route consumes) preserves the declared value program; the reverse
   intervention changes the value program and is correctly restricted to
   diagnostic use. One precision: at `correction_steps=1` the JVP path
   performs two pre-solve standardizations against the value path's one, so
   the forward intervention must remove exactly the *start-of-iteration*
   standardization, leaving the shared outer standardization intact.
5. **Can tangent validity change an externally accepted scalar recursion?**
   No, at the endpoint boundary, under the current source. Internal recursion
   state (carried particles, Sinkhorn fallback branch) can differ, but
   `valid` latches false and both value and score are NaN-masked
   (`cubature_genut_batch_tf.py:1612-1615`; adapter re-mask at
   `cubature_genut_neutra_targets.py:412-419`). Fail-closed as a pair holds.
   Residual exposure surfaces: endpoint validity-domain asymmetry between the
   two endpoints, and unmasked diagnostics on invalid rows (F3, items 1 and
   3).
6. **Austria direct derivative terms present?** Yes — all three
   (`∂κ/∂θ₀ = κ`, `∂ν/∂θ₁ = ν`, `∂σ²/∂θ₂ = 2σ²`) are present and correct by
   derivation (F6).
7. **Does the batch target execute any dual-cap stage?** No pairwise
   correction, no pairwise radial RMS cap, no coordinate cap `.98/8`. It does
   execute a final affine restoration and the Contract-E mean/covariance
   reset. `GenUTControls` cannot express the missing stages (F1). Direct
   answer to the handoff's headline question: **the current batch NeuTra
   target is not the promoted dual-cap GenUT finite program; it is a
   diagonal-only batch candidate.**
8. **Smallest correct architecture for a batch-native dual-cap endpoint?**
   A single shared-primal core per stage: refactor each correction stage into
   one function `stage_core(inputs, tangents: Tensor | None)` in which every
   primal op executes exactly once on one code path and tangent ops are added
   under a Python-level `if tangents is not None` guard. The value endpoint
   is `stage_core(x, None)`; the score endpoint is `stage_core(x, t)`. This
   makes value/score primal identity hold **by construction** (identical op
   sequence, not two maintained copies), leaving only compiler-mode variation
   for H3/H6 to check. Then port the contract-E stage order (diagonal →
   pairwise + row-RMS radial cap → coordinate cap → final affine restoration)
   from `higher_moment_contract_e.py:913-951` onto leading-batch tensors
   inside that shared core, and extend `GenUTControls` with the dual-cap
   fields so `_core_kwargs` can bind them. Do not implement value-only and
   JVP variants as separate functions again — that architecture is the root
   cause class being repaired.
9. **Redundant, non-discriminating, or arithmetic-perturbing tests?** See F5
   (graph/XLA interior capture perturbs the compared arithmetic — remove;
   endpoint-only under graph/XLA) and F7 (Phase 2 steps 1–5 confirmatory;
   H4 exposure test retained with corrected expected outcome).
10. **Additional hypotheses required?** One minor: the covariance-gap
    symmetrization asymmetry (F4), folded into the Phase 2 boundary list and
    the H3 scope. Otherwise the set is complete and causally distinguishable
    for the diagonal-lane question. For the dual-cap lane, the hypothesis set
    transfers, but every H1–H4 equivalent must be re-derived against the
    actual dual-cap stage order once a batch-native implementation exists —
    the diagonal-lane findings must not be assumed to enumerate that route's
    defects.

---

## Corrected Execution Order

The proposed order is correct in substance. Amendments:

```text
Phase 0  Route/evidence freeze (unchanged; ledger wording per F1:
         "diagonal + affine restoration, no pairwise, no dual caps")
Phase 1  Harness (amended per F5: interior capture eager-only;
         graph/XLA endpoint-only)
Phase 2  First-unequal-tensor localization (add F4 gap-symmetrization
         boundary between steps 5 and 6)
Phase 3  H1 causal interventions (unchanged)
Phase 4  H2 solver diagnosis (unchanged)
Phase 5  Branch/batch invariants (rewritten per F3: regression-guard the
         fail-closed mask, expected outcome NaN pair; document endpoint
         validity-domain asymmetry; H7 unchanged — by inspection no
         cross-row coupling exists, so it is a cheap invariant check)
Phase 6  Tangent audit (unchanged; F6 lowers the prior but does not
         replace the phase)
Phase 7  Cross-model + dual-cap-lane regression (unchanged)
Phase 8  Post-repair boundary (unchanged; H8 retuning stands — any H1
         repair changes implementation identity, and under the per-scope
         tuning rule the existing Austria artifact
         `genut_austria_antithetic_ensemble_20260803/tuning_attempt01`
         cannot admit the repaired route)
```

H2 must still not be tested as a replacement for H1; a stable solver on two
different finite programs reduces the gap without repairing the identity
(E8's interpretation is correct).

---

## Decision Table

| Field | Status |
|---|---|
| Decision | REVISE: execute after amending Phase 1 (F5), Phase 2 (F4), Phase 5 (F3), and the route-ledger wording (F1) |
| R0 route identity | `CONFIRM_R0_WRONG_ROUTE_IDENTITY` — batch target is `batch_diagonal_candidate`, correct by inspection |
| H1 asymmetry | Confirmed at source level; initiating-cause status remains for Phase 3 to establish causally |
| H2 classification | Amplifier framing correct; identical solver code in both routes |
| H4 semantics | Endpoint fails closed under current source; test retained as regression guard |
| H5 prior | Lowered by F6 local-derivative verification; phase still required |
| Main uncertainty | Quantitative sufficiency of the FP32 restandardization perturbation for the observed one-iteration magnitude (plausible, not checked) |
| Next justified action | Implementing agent revises the handoff plan per F3/F4/F5, then executes Phase 0 |
| Not concluded | No posterior correctness, exact Austria likelihood, exact nonlinear score, dual-cap correctness or applicability of the diagonal diagnosis to dual-cap, NeuTra/HMC readiness, solver promotion, tuning reuse, or any statistical ranking |

## Audit Non-Claims And Limits

- All conclusions are from static source inspection of the current dirty
  worktree; no numerical value in the handoff (the `5.6596` gap, the `~24`
  cloud difference, the `1.2e5` condition estimate) was independently
  reproduced.
- "Identical op sequence" claims predict bitwise parity only under identical
  device, deterministic ops, and identical graph construction; graph/XLA
  fusion remains an empirical H3/H6 question.
- This audit authorizes nothing beyond plan revision: no campaign launch, no
  default change, no source edit, no tuning reuse, and no Austria NeuTra
  admission.
