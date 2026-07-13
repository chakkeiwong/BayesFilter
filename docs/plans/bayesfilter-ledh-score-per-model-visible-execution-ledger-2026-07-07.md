# LEDH Score Per-Model Visible Execution Ledger

Date: 2026-07-07

Status: `DRAFT_LAUNCH_PACKAGE_CREATED`

Master program:

- `docs/plans/bayesfilter-ledh-score-per-model-master-program-2026-07-07.md`

Runbook:

- `docs/plans/bayesfilter-ledh-score-per-model-visible-gated-execution-runbook-2026-07-07.md`

## Ledger

### 2026-07-07 - Launch Package - PRECHECK

Evidence contract:

- Question: Can each admitted LEDH value row produce a no-tape total
  derivative of the same finite-`N` `log_likelihood` scalar?
- Baseline/comparator: Phase 8 value integration artifact, Phase 2-7 admitted
  value artifacts, exact derivatives where available, and same-scalar finite
  differences with fixed randomness otherwise.
- Primary criterion: a score row is admitted only after same-target value
  admission, no-tape total derivative implementation, tiny correctness pass,
  `N=10000` correctness/memory pass, and replayable score artifact validation.
- Veto diagnostics: score before value, value/score row-set mismatch,
  diagnostic SIR promotion, tape/autodiff, stopped partial derivative, wrong
  scalar, wrong parameter vector, nonfinite score, FD/exact mismatch, memory
  failure, or runtime-only promotion.
- Nonclaims: no score admission at launch; no HMC readiness, posterior
  correctness, scientific superiority, runtime ranking, or all-algorithm
  comparison.

Actions:

- Created draft master program, visible runbook, stop handoff, Phase 0 subplan,
  and launch review bundle.

Artifacts:

- `docs/plans/bayesfilter-ledh-score-per-model-master-program-2026-07-07.md`
- `docs/plans/bayesfilter-ledh-score-per-model-visible-gated-execution-runbook-2026-07-07.md`
- `docs/plans/bayesfilter-ledh-score-per-model-visible-stop-handoff-2026-07-07.md`
- `docs/plans/bayesfilter-ledh-score-per-model-phase0-baseline-governance-subplan-2026-07-07.md`
- `docs/reviews/bayesfilter-ledh-score-per-model-launch-review-bundle-2026-07-07.md`

Gate status:

- `IN_PROGRESS_PENDING_LOCAL_CHECKS_AND_REVIEW`

Next action:

- Run launch package local checks, then bounded read-only review.

### 2026-07-07 - Launch Package - REVIEW_ACCEPTED_PHASE0_MAY_START

Evidence contract:

- Question: Is it safe to launch Phase 0 as governance/baseline freeze only?
- Primary criterion: read-only review agrees Phase 0 admits no score, performs
  no score implementation, uses the Phase 8 value artifact as row-set anchor,
  bans tape/autodiff/stopped partials for admitted evidence, excludes the
  parameterized SIR diagnostic row, and preserves the KSC finite-mixture target
  boundary.
- Nonclaims: no score admission, score correctness, HMC readiness, posterior
  correctness, scientific superiority, runtime ranking, or all-algorithm
  comparison.

Actions:

- Ran launch local checks:
  - `py_compile` for upstream value integration builder/test passed;
  - Phase 8 value integration replay passed: `3 passed, 2 warnings`;
  - launch package diff hygiene passed.
- Direct Claude health probe returned `CLAUDE_PROBE_OK`.
- Broader launch packet review produced no output within repeated polls and
  was stopped.
- Narrowed verdict-only packet review returned `VERDICT: AGREE`.
- Review precision note: Phase 0 should explicitly freeze score as derivative
  of the realized finite-`N` estimator, not the true likelihood.

Artifacts:

- `docs/reviews/bayesfilter-ledh-score-per-model-launch-review-bundle-2026-07-07.md`

Review:

- Narrowed packet review: `VERDICT=AGREE`.

Gate status:

- `PASSED_LAUNCH_GATE_PHASE0_MAY_START`

Next action:

- Execute Phase 0 baseline and score-governance freeze.

### 2026-07-07 - Phase 0 - CHECK_REPAIR_STATIC_NO_TAPE_INVENTORY

Evidence contract:

- Question: Does the Phase 0 no-tape inventory check distinguish admitted
  score helper code from test guard text?
- Primary criterion: the static check must inspect actual LGSSM/fixed-SIR score
  helper functions for banned autodiff symbols without failing merely because
  tests define the banned-token guard set.
- Veto diagnostics: allowing tape/autodiff in helper functions; treating a
  broad token hit in a guard test as a score blocker without inspection.
- Nonclaims: no score admission, no score correctness, no Phase 1 execution.

Actions:

- Phase 0 broad static token check failed on
  `tests/test_ledh_lgssm_manual_score_phase4.py:140` because the test declares
  `forbidden_attrs = {"GradientTape", "ForwardAccumulator"}` as part of its
  guard.
- Inspected LGSSM and fixed-SIR test contexts and confirmed these hits are
  guard strings, not score helper route usage.
- Patched Phase 0 subplan to use AST inspection of named score helper
  functions instead of broad file-token search.

Artifacts:

- Patched:
  `docs/plans/bayesfilter-ledh-score-per-model-phase0-baseline-governance-subplan-2026-07-07.md`

Gate status:

- `IN_PROGRESS_RERUN_PHASE0_CHECKS`

Next action:

- Rerun targeted Phase 0 no-tape inventory check and continue Phase 0.

### 2026-07-07 - Phase 0 - REVIEW_REPAIR_PHASE1_FIELD_PRESERVATION

Evidence contract:

- Question: Does the Phase 1 schema subplan preserve the admitted value
  artifact's target-policy and parameter-coordinate identity?
- Primary criterion: read-only review finding is patched so Phase 1 requires
  row id, target scalar, output field, target observation policy, theta
  coordinate system, and parameter names/order to match the admitted value
  artifact.
- Nonclaims: no Phase 1 execution yet and no score admission.

Actions:

- Claude Phase 0/1 handoff review returned `VERDICT=REVISE`.
- Finding: Phase 1 validator outline matched row id and target scalar/output
  but did not explicitly require target observation policy, theta coordinate
  system, and parameter names/order preservation.
- Patched Phase 1 subplan primary criterion, veto diagnostics, step-by-step
  validator outline, negative tests, handoff conditions, and stop conditions.
- Patched Phase 0 result handoff and review bundle to state the preservation
  requirement.

Artifacts:

- Patched:
  `docs/plans/bayesfilter-ledh-score-per-model-phase1-score-schema-subplan-2026-07-07.md`
- Patched:
  `docs/plans/bayesfilter-ledh-score-per-model-phase0-baseline-governance-result-2026-07-07.md`
- Patched:
  `docs/reviews/bayesfilter-ledh-score-per-model-phase0-result-phase1-subplan-review-bundle-2026-07-07.md`

Gate status:

- `IN_PROGRESS_PENDING_REPAIR_CHECKS_AND_REVIEW`

Next action:

- Rerun focused checks and a narrowed read-only repair review.

### 2026-07-07 - Phase 0 - REVIEW_REPAIR_ACCEPTED

Evidence contract:

- Question: Did the Phase 1 field-preservation repair close the handoff
  blocker?
- Primary criterion: read-only review agrees Phase 1 schema-only execution may
  start after the subplan explicitly requires row id, target scalar, output
  field, target observation policy, theta coordinate system, and parameter
  names/order to match the admitted value artifact.
- Nonclaims: no Phase 1 implementation yet and no score admission.

Actions:

- Reran focused checks:
  - diff hygiene passed;
  - Phase 8 + LGSSM/fixed-SIR diagnostic replay passed:
    `14 passed, 2 warnings`.
- Ran focused Claude repair review.

Review:

- Focused repair review: `VERDICT=AGREE`.

Gate status:

- `PASSED_PHASE0_PHASE1_HANDOFF_PHASE1_MAY_START`

Next action:

- Execute Phase 1 score artifact schema and guards.

### 2026-07-07 - Phase 1 - LOCAL_PASS_PENDING_REVIEW

Evidence contract:

- Question: Can the repo reject score artifacts that are not no-tape
  same-scalar total derivatives of admitted value artifacts?
- Primary criterion: validator accepts only artifacts with admitted source
  value artifact, same row id, same target scalar, same output field, same
  target observation policy, same theta coordinate system, same parameter
  names/order, finite score vector, no-tape derivative provenance, same-route
  value/score identity, tiny/full admission distinction, memory fields for
  `N=10000`, and no forbidden autodiff flags.
- Veto diagnostics: score/value mismatch, tape/ForwardAccumulator/stopped
  partial route, diagnostic SIR promotion, KSC exact-SV overclaim, tiny-as-full
  admission, missing full-row memory gate, or nonfinite score.
- Nonclaims: no model score admission and no HMC/posterior/scientific/runtime
  claim.

Actions:

- Added score contract implementation:
  `bayesfilter/highdim/ledh_score_contract.py`.
- Added score contract tests:
  `tests/highdim/test_ledh_score_contract_phase1.py`.
- First focused schema run failed because the provenance guard banned the
  substring `autodiff`, which also appears in approved `no_autodiff` route
  labels.
- Repaired the guard to keep explicit route allowlisting and concrete
  tape/ForwardAccumulator/stopped-partial vetoes without rejecting approved
  no-tape route names.
- Wrote Phase 1 result, Phase 2 LGSSM subplan, and Phase 1 review bundle.

Local checks:

- Compile check passed.
- Focused schema tests passed: `19 passed, 2 warnings`.
- Combined value/schema replay passed: `22 passed, 2 warnings`.
- Diff hygiene passed for changed Phase 1 code/tests.

Artifacts:

- `docs/plans/bayesfilter-ledh-score-per-model-phase1-score-schema-result-2026-07-07.md`
- `docs/plans/bayesfilter-ledh-score-per-model-phase2-lgssm-subplan-2026-07-07.md`
- `docs/reviews/bayesfilter-ledh-score-per-model-phase1-review-bundle-2026-07-07.md`

Gate status:

- `LOCAL_PASS_PENDING_READONLY_REVIEW`

Next action:

- Run bounded read-only review of Phase 1 result and Phase 2 LGSSM subplan.

### 2026-07-07 - Phase 1 - REVIEW_ACCEPTED_PHASE2_MAY_START

Evidence contract:

- Question: Is it safe to start Phase 2 LGSSM score work after the Phase 1
  schema result?
- Primary criterion: read-only review agrees Phase 1 guards same-target
  score/value identity and no-tape provenance, and Phase 2 treats the stale
  LGSSM `N=1000` full-row identity as a preflight blocker before any N=10000
  score admission.
- Nonclaims: no LGSSM score admission yet; no full GPU score run yet; no
  HMC/posterior/scientific/runtime claim.

Actions:

- Tried the material Claude review gate. It returned:
  `REVIEW_STATUS=probe_timeout`, `VERDICT=NONE`.
- Ran tiny Claude health probe. It returned `CLAUDE_PROBE_OK`.
- Narrowed the review prompt to the compact packet and cited fixed paths.

Review:

- Narrowed packet review returned `VERDICT=AGREE`.
- Claude findings agreed that:
  - same-target identity is guarded at the admission boundary;
  - the no-tape repair is boundary-safe;
  - stale LGSSM `FULL_ROW_NUM_PARTICLES = 1000` is correctly treated as a
    preflight repair/blocker;
  - score admission is gated on
    `validate_ledh_score_artifact(..., require_admitted=True)`;
  - stop/handoff conditions are sufficient for Phase 2.

Artifacts:

- `docs/reviews/bayesfilter-ledh-score-per-model-phase1-review-bundle-2026-07-07.md`
- `.claude_reviews/20260708-013612-bayesfilter-ledh-score-phase1-phase2-handoff-20260707/status.json`

Gate status:

- `PASSED_PHASE1_PHASE2_HANDOFF_PHASE2_MAY_START`

Next action:

- Execute Phase 2 LGSSM preflight/tiny/schema work. Do not run the full
  trusted GPU `N=10000` score command until the preflight identifies and
  repairs or overrides the stale active full-row identity.

### 2026-07-07 - Phase 2 LGSSM - PREFLIGHT_PASS_FULL_RUN_MAY_START

Evidence contract:

- Question: Can the active LGSSM score route be made to target the admitted
  N=10000 value row before the trusted full score run?
- Primary criterion: active full-row identity uses N=10000, stale N=1000 is
  rejected as non-full-row evidence, raw score output can be normalized into
  the Phase 1 score artifact schema, and tiny/no-tape score tests still pass.
- Veto diagnostics: stale N=1000 admission, schema bypass, wrong score
  provenance, CPU/runtime-only full admission, or lost no-tape sentinel.
- Nonclaims: no full LGSSM score admission yet and no HMC/posterior/scientific
  claim.

Actions:

- Repaired active LGSSM full-row identity:
  `FULL_ROW_NUM_PARTICLES = 10000`.
- Added `_lgssm_score_artifact_from_result(...)` to normalize raw LGSSM score
  output into the Phase 1 score schema.
- Added Phase 2 tests that:
  - assert N=10000 is the only active full-row identity;
  - reject stale N=1000 raw results;
  - validate a full raw score fixture against the admitted value artifact;
  - keep CPU/runtime misses as tiny diagnostics, not admitted scores.

Local checks:

- Compile check passed.
- Existing LGSSM tiny score + Phase 1 schema + Phase 2 adapter tests passed:
  `29 passed, 2 warnings`.
- Diff hygiene passed for changed Phase 2 code/tests and plan artifacts.

Artifacts:

- `docs/benchmarks/benchmark_ledh_same_target_lgssm_m3_t50_value.py`
- `tests/highdim/test_ledh_lgssm_score_phase2_contract.py`

Gate status:

- `PASSED_PHASE2_PREFLIGHT_FULL_GPU_RUN_MAY_START`

Next action:

- Run trusted GPU/CUDA `N=10000`, `T=50` LGSSM compact no-tape score command
  with stdout/stderr captured to the Phase 2 log.

### 2026-07-07 - Phase 2 LGSSM - FULL_RUN_COMMAND_REPAIR

Evidence contract:

- Question: Did the failed full-run launch reveal a model/score failure or an
  invocation/route-name defect?
- Primary criterion: repair only the invocation defect if the active code path
  still dispatches to the full total-VJP route for `transport_ad_mode=full`.
- Veto diagnostics: using the historical stopped partial route for admitted
  score evidence, or hiding a route mismatch behind a renamed artifact.
- Nonclaims: no full score evidence yet.

Actions:

- Initial trusted GPU command failed with exit code `2`.
- Captured log showed an argparse error:
  `--transport-gradient-mode manual_streaming_finite` was not a valid choice.
- Inspected the active route. The CLI constant is the legacy string
  `manual_streaming_finite_sinkhorn_stopped_scale_keys`, but with
  `transport_ad_mode=full` the value path dispatches to
  `_filterflow_manual_streaming_finite_transport_total_vjp`, and the compact
  score helper uses
  `_filterflow_manual_streaming_finite_transport_value_and_jvp_total`.
- Patched the Phase 2 subplan command to use the actual CLI choice.
- Added a Phase 2 test that asserts full-mode dispatch reaches the total-VJP
  route and that the compact total-JVP helper source contains no
  `tf.stop_gradient` or `GradientTape`.

Local checks:

- Compile check passed.
- Phase 2 LGSSM contract + existing LGSSM manual score tests passed:
  `11 passed, 2 warnings`.
- Diff hygiene passed.

Gate status:

- `REPAIRED_INVOCATION_FULL_GPU_RUN_MAY_RETRY`

Next action:

- Retry the trusted GPU/CUDA `N=10000`, `T=50` LGSSM compact score command
  using `manual_streaming_finite_sinkhorn_stopped_scale_keys` with
  `transport_ad_mode=full`.

### 2026-07-07 - Phase 2 LGSSM - FULL_RUN_BLOCKED_REPAIR_LOOP_OPEN

Evidence contract:

- Question: Did the full LGSSM T=50,N=10000 score command produce admissible
  evidence?
- Primary criterion: raw JSON exists and can be normalized into a Phase 1
  score artifact with `require_admitted=True`.
- Veto diagnostics: no artifact, wrong T/N identity, hidden use of old T=2
  evidence, tape/stopped partial route, or unbounded execution.
- Nonclaims: no LGSSM score admission.

Actions:

- Trusted full run remained active for a long visible window but produced no
  raw JSON artifact.
- Trusted GPU telemetry showed high memory allocation and nonzero utilization.
- The run was interrupted with Ctrl-C to preserve visible execution control.
- The log traceback showed execution was inside the compact JVP finite-Sinkhorn
  softmin path during score/FD computation.
- Inspected prior score-memory artifacts and confirmed the old LGSSM
  `N=10000` artifact used `T=2`, so it cannot be admitted against the current
  `T=50` value artifact.
- Wrote a blocker result and repair subplan.

Artifacts:

- `docs/plans/bayesfilter-ledh-score-per-model-phase2-lgssm-n10000-run-2026-07-07.log`
- `docs/plans/bayesfilter-ledh-score-per-model-phase2-lgssm-full-run-blocker-result-2026-07-07.md`
- `docs/plans/bayesfilter-ledh-score-per-model-phase2-lgssm-repair-subplan-2026-07-07.md`

Gate status:

- `BLOCKED_FIXABLE_PHASE2_REPAIR_LOOP_OPEN`

Next action:

- Execute the Phase 2 repair subplan: build a bounded full-row score artifact
  path that computes the compact score once and uses a predeclared directional
  FD diagnostic instead of per-coordinate full FD.

### 2026-07-07 - Phase 2 LGSSM - REPAIR_REVIEW_REVISE_DIRECTIONAL_FD_TOO_WEAK

Evidence contract:

- Question: Is a single directional FD check sufficient to admit a 5D LGSSM
  full score artifact?
- Primary criterion: read-only review must agree any evidence-contract repair
  preserves full-score correctness.
- Nonclaims: no LGSSM score admission.

Review:

- Claude focused repair review returned `VERDICT=REVISE`.
- Finding: one predeclared directional FD checks only one projection of the 5D
  score and is too weak as the sole full-admission diagnostic.
- Finding: old T=2 evidence is forbidden in the plan, but an explicit
  implemented rejection test is still needed.

Repair:

- Patched the Phase 2 repair subplan so directional FD is diagnostic only.
- Full admission now still requires coordinate-wise same-scalar finite
  differences for all five parameters, exact/reference all-parameter score, or
  proof-backed reviewed tests.
- If only directional FD evidence is feasible, Phase 2 must write a blocker
  result instead of an admitted score artifact.

Gate status:

- `REPAIR_SUBPLAN_REVISED_PENDING_CHECKS_AND_REVIEW`

Next action:

- Rerun local checks and focused read-only review of the revised repair
  subplan.

### 2026-07-07 - Phase 2 LGSSM - CLOSED_BLOCKED_PHASE3_MAY_START_AFTER_REVIEW

Evidence contract:

- Question: Can Phase 2 close without admitting LGSSM while preserving the
  runbook boundary?
- Primary criterion: result explicitly blocks LGSSM, preserves the stronger
  all-parameter correctness gate, rejects old T=2 and stale N=1000 evidence,
  and drafts Phase 3 fixed-SIR subplan.
- Nonclaims: no LGSSM score admission and no scientific/runtime/HMC claim.

Actions:

- Added old T=2 rejection tests.
- Local checks passed:
  `32 passed, 2 warnings`.
- Wrote Phase 2 result and Phase 3 fixed-SIR subplan.

Artifacts:

- `docs/plans/bayesfilter-ledh-score-per-model-phase2-lgssm-result-2026-07-07.md`
- `docs/plans/bayesfilter-ledh-score-per-model-phase3-fixed-sir-subplan-2026-07-07.md`

Gate status:

- `PENDING_PHASE2_RESULT_PHASE3_SUBPLAN_REVIEW`

Next action:

- Review Phase 2 blocker result and Phase 3 fixed-SIR subplan. If review
  agrees, execute Phase 3.

### 2026-07-07 - Phase 2/3 Handoff - REVIEW_ACCEPTED_PHASE3_MAY_START

Evidence contract:

- Question: Can Phase 2 close as LGSSM blocked and Phase 3 fixed-SIR start
  without crossing score-admission boundaries?
- Primary criterion: read-only review agrees LGSSM old T=2/stale N=1000
  evidence is not admitted, all-parameter correctness boundary is preserved,
  fixed-SIR targets the main row, and directional FD remains diagnostic only.
- Nonclaims: no LGSSM score admission and no fixed-SIR score admission yet.

Review:

- Claude handoff review returned `VERDICT=AGREE`.
- Findings:
  - Phase 2 can close as blocked/not admitted.
  - Old T=2 and stale N=1000 are rejected by tests.
  - LGSSM all-parameter correctness boundary is preserved.
  - Phase 3 preserves fixed-SIR main-row identity and rejects
    parameterized/no-free-theta promotion.
  - Phase 3 stop/handoff conditions are sufficient.
- Minor nonblocking note: fixed-SIR adapter docstring and older tests mention
  Phase 4 naming, but substantive boundaries are correct.

Gate status:

- `PASSED_PHASE2_PHASE3_HANDOFF_PHASE3_MAY_START`

Next action:

- Execute Phase 3 fixed-SIR preflight.

### 2026-07-07 - Phase 3 Fixed-SIR - CLOSED_BLOCKED_PENDING_REVIEW

Evidence contract:

- Question: Can the fixed-SIR main row produce an admitted no-tape total
  derivative of the same finite-`N` LEDH `log_likelihood` scalar as the value
  artifact?
- Primary criterion: full admission requires all-parameter same-scalar
  correctness for
  `[log_kappa_scale, log_nu_scale, log_obs_noise_scale]`, full-row
  `T=20,N=10000` identity, memory pass, no-tape provenance, and
  `validate_ledh_score_artifact(..., require_admitted=True)`.
- Nonclaims: no fixed-SIR score admission; no rejection of the manual
  total-VJP mathematics; no exact nonlinear likelihood, HMC, posterior,
  source-faithfulness, runtime, or scientific claim.

Actions:

- Inspected the fixed-SIR score-memory artifact:
  `docs/plans/ledh-phase5-fixed-sir-score-memory-n10000-2026-07-06.json`.
- Confirmed it is main-row `N=10000` finite-score and memory evidence, but
  only directional same-scalar FD correctness.
- Patched the fixed-SIR adapter so `require_all_parameter_correctness=True`
  requires an explicit `all_parameter_score_correctness` record; a Boolean
  flag can no longer promote directional-only evidence.
- Added a test rejecting flag-only all-parameter promotion.
- Wrote Phase 3 result and Phase 4 predator-prey subplan.

Local checks:

- `CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest tests/highdim/test_ledh_fixed_sir_score_phase3_contract.py tests/test_ledh_fixed_sir_manual_score_phase4.py tests/highdim/test_ledh_score_contract_phase1.py -q`
- Result: `29 passed, 2 warnings`.

Artifacts:

- `docs/plans/bayesfilter-ledh-score-per-model-phase3-fixed-sir-result-2026-07-07.md`
- `docs/plans/bayesfilter-ledh-score-per-model-phase4-predator-prey-subplan-2026-07-07.md`
- `docs/reviews/bayesfilter-ledh-score-per-model-phase3-result-phase4-subplan-review-bundle-2026-07-07.md`
- `docs/benchmarks/benchmark_ledh_same_target_fixed_sir_score.py`
- `tests/highdim/test_ledh_fixed_sir_score_phase3_contract.py`

Gate status:

- `PENDING_PHASE3_RESULT_PHASE4_SUBPLAN_REVIEW`

Next action:

- Run bounded read-only review. If review agrees, execute Phase 4
  predator-prey preflight.

### 2026-07-07 - Phase 3/4 Handoff - REVIEW_ACCEPTED_PHASE4_MAY_START

Evidence contract:

- Question: Can Phase 3 close as fixed-SIR blocked and Phase 4 predator-prey
  start without crossing score-admission boundaries?
- Primary criterion: read-only review agrees fixed-SIR directional-only
  evidence is not admitted, the guard repair prevents flag-only promotion, and
  Phase 4 preserves same scalar, physical parameter order, no-tape provenance,
  and all-parameter correctness boundaries.
- Nonclaims: no fixed-SIR score admission and no predator-prey score admission
  yet.

Review:

- Claude review gate wrapper timed out at probe stage:
  `REVIEW_STATUS=probe_timeout`, `VERDICT=NONE`.
- Direct trusted Claude health probe returned `CLAUDE_PROBE_OK`.
- First narrowed packet review returned `VERDICT=REVISE` because the packet
  did not quote enough of the Phase 4 subplan to make Q3/Q4 self-sufficient.
- Patched the review bundle with the Phase 4 target scalar, output field,
  row id, source value artifact, observation policy, physical parameter order,
  admitted-route label, full-row identity, all-parameter correctness gate, and
  forbidden evidence/actions.
- Second narrowed packet review returned `VERDICT=AGREE`.

Artifacts:

- `docs/reviews/bayesfilter-ledh-score-per-model-phase3-result-phase4-subplan-review-bundle-2026-07-07.md`

Gate status:

- `PASSED_PHASE3_PHASE4_HANDOFF_PHASE4_MAY_START`

Next action:

- Execute Phase 4 predator-prey preflight and score-route inventory.

### 2026-07-07 - Phase 4 Predator-Prey - ROUTE_INVENTORY_BLOCKED_REPAIR_LOOP_OPEN

Evidence contract:

- Question: Can the predator-prey main row currently produce an admitted
  no-tape total derivative of the same finite-`N` LEDH `log_likelihood` scalar
  as the value artifact?
- Primary criterion: full admission requires a manual total-score adapter,
  all-parameter same-scalar correctness for `[r,K,a,s,u,v]`, full-row
  `T=20,N=10000` identity, memory pass, no-tape provenance, and score artifact
  validation.
- Nonclaims: no predator-prey score admission; no rejection of adapter
  feasibility; no exact nonlinear likelihood, HMC, posterior,
  source-faithfulness, runtime, or scientific claim.

Actions:

- Ran Phase 4 CPU-hidden preflight and value/schema replay.
- Inventoried the predator-prey value runner and model-local score helpers.
- Found value-only streaming LEDH callbacks and local-density parameter score
  methods, but no total reverse-scan LEDH score adapter.
- Added a Phase 4 score boundary test rejecting value-only and directional-only
  score promotion.
- Wrote a fixable blocker result and repair subplan.

Local checks:

- Forward value/schema replay: `21 passed, 2 warnings`.
- Phase 4 score boundary replay: `23 passed, 2 warnings`.

Artifacts:

- `docs/plans/bayesfilter-ledh-score-per-model-phase4-predator-prey-route-inventory-blocker-result-2026-07-07.md`
- `docs/plans/bayesfilter-ledh-score-per-model-phase4-predator-prey-repair-subplan-2026-07-07.md`
- `docs/reviews/bayesfilter-ledh-score-per-model-phase4-predator-prey-repair-review-bundle-2026-07-07.md`
- `tests/highdim/test_ledh_predator_prey_score_phase4_contract.py`

Gate status:

- `BLOCKED_FIXABLE_PHASE4_REPAIR_SUBPLAN_PENDING_REVIEW`

Next action:

- Run bounded read-only review of the Phase 4 predator-prey repair subplan.
  If review agrees, execute the repair subplan. If review revises, patch and
  rerun focused checks/review.

### 2026-07-07 - Phase 4 Predator-Prey Repair - REVIEW_ACCEPTED_EXECUTION_MAY_START

Evidence contract:

- Question: Is it boundary-safe to execute the predator-prey manual total-VJP
  score adapter repair plan?
- Primary criterion: read-only review agrees the blocker is fixable but not
  admitted, the repair plan includes transport/normalization/LEDH-flow/pre-flow
  and RK4 VJP pieces, and proxy evidence cannot be promoted.
- Nonclaims: no predator-prey score admission and no implementation evidence
  yet.

Review:

- Focused Claude review returned `VERDICT=AGREE`.
- Findings:
  - blocker result correctly refuses score admission;
  - repair subplan includes required reverse-scan pieces;
  - local-density-only, value-only, and directional-only evidence remain
    forbidden for full score admission;
  - execution is boundary-safe after tiny/schema gates and before any full
    `N=10000,T=20` run.

Gate status:

- `PASSED_PHASE4_REPAIR_REVIEW_EXECUTE_REPAIR`

Next action:

- Implement the repair in small steps, starting with predator-prey RHS/RK4 VJP
  unit tests before full filter reverse-scan wiring.

### 2026-07-07 - Phase 4 Predator-Prey Repair - RUNG1_DYNAMICS_VJP_PASSED

Evidence contract:

- Question: Can the predator-prey dynamics VJP pieces needed for a no-tape
  total-score adapter be implemented and checked?
- Primary criterion: RHS and transition-mean VJPs match central finite
  differences for all six physical parameters and state components.
- Nonclaims: no predator-prey score admission; no full LEDH total derivative;
  no full-row score/memory gate.

Actions:

- Added `docs/benchmarks/benchmark_ledh_same_target_predator_prey_score.py`.
- Implemented no-tape manual predator-prey RHS VJP, RK4 step VJP, and
  transition-mean VJP.
- Added finite-difference tests for all six physical parameters and state
  components.

Local checks:

- Focused dynamics/guard test: `4 passed, 2 warnings`.
- Broader Phase 4 repair check: `25 passed, 2 warnings`.

Artifacts:

- `docs/plans/bayesfilter-ledh-score-per-model-phase4-predator-prey-repair-rung1-dynamics-vjp-result-2026-07-07.md`
- `docs/benchmarks/benchmark_ledh_same_target_predator_prey_score.py`
- `tests/highdim/test_ledh_predator_prey_score_phase4_contract.py`

Gate status:

- `PHASE4_REPAIR_RUNG1_PASSED_RUNG2_PENDING`

Next action:

- Continue the Phase 4 repair with fixed-randomness forward replay and
  reverse-scan skeleton. Do not run full `N=10000,T=20` score/memory execution
  until tiny total-score/schema checks pass.

### 2026-07-07 - Phase 4 Predator-Prey Repair - RUNG2_TINY_TOTAL_SCORE_PASSED

Evidence contract:

- Question: Can the predator-prey no-tape total-score route run at tiny scale
  and match all-coordinate same-scalar finite differences?
- Primary criterion: route runs under runtime no-autodiff sentinel, matches
  coordinate-wise FD, and score artifacts remain diagnostic until full memory
  and all-parameter gates pass.
- Nonclaims: no predator-prey full score admission; no full-row score/memory
  evidence.

Actions:

- Wired a tiny fixed-randomness predator-prey total-score route.
- Added coordinate-wise same-scalar FD tests.
- Added tiny score artifact normalization and guards against full admission
  without explicit all-parameter correctness plus memory pass.

Local checks:

- Focused tiny route checks: `8 passed, 2 warnings`.
- Combined Phase 4 checks: `29 passed, 2 warnings`.

Artifacts:

- `docs/plans/bayesfilter-ledh-score-per-model-phase4-predator-prey-repair-rung2-tiny-total-score-result-2026-07-07.md`
- `docs/benchmarks/benchmark_ledh_same_target_predator_prey_score.py`
- `tests/highdim/test_ledh_predator_prey_score_phase4_contract.py`

Gate status:

- `PHASE4_REPAIR_RUNG2_PASSED_FULL_RUN_GATE_PENDING`

Next action:

- Review and prepare the full `N=10000,T=20` predator-prey score/memory gate.
  Do not claim score admission until a full artifact validates.

### 2026-07-07 - Phase 4 Predator-Prey Repair - RUNG3_GPU_SMOKE_MIXED_BLOCKED

Evidence contract:

- Question: Is the predator-prey no-tape score route ready for full
  `N=10000,T=20` score admission?
- Primary criterion: bounded GPU smoke should produce trustworthy
  all-parameter correctness evidence before full-row launch.
- Nonclaims: no predator-prey score admission and no full-row score/memory
  evidence.

Actions:

- Added score-runner CLI and tiny artifact writer.
- Fixed FD dtype handling.
- Ran tiny CPU-hidden CLI smoke.
- Ran trusted GPU float32/TF32 smoke and trusted GPU FP64 smoke.
- Tightened diagnostic status so failed FD checks write
  `blocked_score_not_run` instead of false `pass`.

Results:

- Tiny CPU-hidden CLI smoke passed as diagnostic-only.
- GPU float32/TF32 smoke failed strict FD correctness:
  `max_abs_error=0.2005905956029892`,
  `max_rel_error=0.9131697416305542`.
- GPU FP64 smoke passed tightly:
  `max_abs_error=3.948576079437771e-06`,
  `max_rel_error=1.5018796503683548e-08`.
- Focused tests after guard repair: `9 passed, 2 warnings`.
- Combined Phase 4 checks: `29 passed, 2 warnings`.

Artifacts:

- `docs/plans/bayesfilter-ledh-score-per-model-phase4-predator-prey-repair-rung3-gpu-smoke-result-2026-07-07.md`
- `/tmp/predator-prey-score-tiny.json`
- `/tmp/predator-prey-score-gpu-smoke.json`
- `/tmp/predator-prey-score-gpu-smoke-fp64.json`

Gate status:

- `PHASE4_REPAIR_FULL_RUN_BLOCKED_PENDING_CORRECTNESS_CALIBRATION_SUBPLAN`

Next action:

- Draft and review a Phase 4 full-row correctness calibration subplan before
  any full `N=10000,T=20` score admission run.

### 2026-07-07 - Phase 4 Predator-Prey Repair - CALIBRATION_SUBPLAN_DRAFTED

Evidence contract:

- Question: What correctness evidence is required before predator-prey
  full-row score admission after FP64 passes but FP32/TF32 FD is noisy?
- Primary criterion: any full score admission must preserve same-scalar
  all-parameter correctness and memory gates without proxy promotion.
- Nonclaims: no predator-prey score admission and no full score run.

Actions:

- Drafted the Phase 4 full-row correctness calibration subplan.
- Drafted a bounded read-only review bundle.

Artifacts:

- `docs/plans/bayesfilter-ledh-score-per-model-phase4-predator-prey-full-row-correctness-calibration-subplan-2026-07-07.md`
- `docs/reviews/bayesfilter-ledh-score-per-model-phase4-predator-prey-calibration-review-bundle-2026-07-07.md`

Gate status:

- `PENDING_PHASE4_CALIBRATION_SUBPLAN_REVIEW`

Next action:

- Run bounded read-only review of the calibration subplan.

### 2026-07-07 - Phase 4 Predator-Prey Calibration - REVIEW_ACCEPTED_EXECUTION_MAY_START

Evidence contract:

- Question: Is the Phase 4 predator-prey full-row correctness calibration
  subplan safe to execute before any full score admission run?
- Primary criterion: read-only review agrees the plan refuses full admission
  from tiny FP64 evidence alone, separates FP64 correctness from FP32/TF32
  runtime/memory, and prevents post-hoc tolerance loosening.
- Nonclaims: no predator-prey score admission and no full score run.

Review:

- First narrowed Claude review returned `VERDICT=REVISE` because the packet did
  not quote enough of the subplan.
- Patched the packet with the concrete calibration ladder, non-promotion
  rules, and stop conditions.
- Second narrowed Claude review returned `VERDICT=AGREE`.

Gate status:

- `PASSED_PHASE4_CALIBRATION_REVIEW_EXECUTE_CALIBRATION`

Next action:

- Execute the calibration ladder, starting with bounded FP64 GPU correctness
  rungs before any full score admission.

### 2026-07-07 - Phase 4 Predator-Prey Calibration - CLOSED_BLOCKED_PHASE5_REVIEW_PENDING

Evidence contract:

- Question: Can predator-prey full-row score be admitted after bounded FP64
  correctness passes but FP32/TF32 finite-difference diagnostics fail?
- Primary criterion: score admission requires a validating full artifact with
  same-scalar all-parameter correctness and `N=10000,T=20` memory evidence.
- Veto diagnostics: failed FP32/TF32 FD, absent full validating score artifact,
  and no reviewed bridge from bounded FP64 diagnostics to production full-row
  admission.
- Nonclaims: no predator-prey score admission; no rejection of the FP64
  manual total-VJP mathematics; no HMC, posterior, runtime, source-faithfulness,
  exact-likelihood, or scientific claim.

Actions:

- Inspected the completed FP64 and FP32/TF32 calibration artifacts.
- Wrote Phase 4 calibration result closing predator-prey as blocked/not
  admitted.
- Drafted Phase 5 actual-SV score subplan.
- Wrote read-only review bundle for Phase 4 result and Phase 5 subplan.

Artifacts:

- `docs/plans/bayesfilter-ledh-score-per-model-phase4-predator-prey-full-row-correctness-calibration-result-2026-07-07.md`
- `docs/plans/bayesfilter-ledh-score-per-model-phase5-actual-sv-subplan-2026-07-07.md`
- `docs/reviews/bayesfilter-ledh-score-per-model-phase4-calibration-result-phase5-subplan-review-bundle-2026-07-07.md`

Gate status:

- `LOCAL_ARTIFACTS_WRITTEN_PENDING_CHECKS_AND_REVIEW`

Next action:

- Run focused local checks and bounded read-only review. If review agrees,
  execute Phase 5 actual-SV score preflight and inventory.

### 2026-07-07 - Phase 4/5 Handoff - REVIEW_ACCEPTED_PHASE5_MAY_START

Evidence contract:

- Question: Is it boundary-safe to close predator-prey as blocked/not admitted
  and start Phase 5 actual-SV score from the drafted subplan?
- Primary criterion: local replay/schema checks pass and read-only review
  agrees the Phase 4 result refuses proxy promotion while the Phase 5 subplan
  preserves exact transformed actual-SV target boundaries.
- Nonclaims: no actual-SV score admission and no full score run.

Local checks:

```text
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/highdim/test_ledh_phase5_actual_sv_forward_scalar_artifact.py \
  tests/highdim/test_ledh_phase5_actual_sv_forward_scalar_tiny_artifact.py \
  tests/highdim/test_ledh_score_contract_phase1.py -q
```

Result:

```text
24 passed, 2 warnings
```

Review:

- `claude_review_gate.sh` returned `REVIEW_STATUS=probe_timeout`; this was
  not treated as a review result.
- Direct tiny Claude probe returned `CLAUDE_PROBE_OK`.
- Narrowed direct packet-only review returned `VERDICT: AGREE`.

Artifacts:

- Review gate status:
  `/home/chakwong/BayesFilter/.claude_reviews/20260708-050055-ledh-score-phase4-calibration-phase5-subplan/status.json`
- Review bundle:
  `docs/reviews/bayesfilter-ledh-score-per-model-phase4-calibration-result-phase5-subplan-review-bundle-2026-07-07.md`

Gate status:

- `PASSED_PHASE4_PHASE5_HANDOFF_PHASE5_MAY_START`

Next action:

- Execute Phase 5 actual-SV score preflight and route inventory. Do not launch
  full `N=10000,T=1000` score admission until tiny all-coordinate correctness
  passes and a reviewed full-row score/memory subplan exists.

### 2026-07-07 - Phase 5 Actual-SV Score - TINY_DIAGNOSTIC_PASSED_FULL_NOT_ADMITTED

Evidence contract:

- Question: Can actual-SV produce a no-tape total derivative of the same
  finite-`N` exact transformed LEDH `log_likelihood` scalar admitted by the
  value artifact?
- Primary criterion: tiny all-coordinate same-scalar FD and no-tape sentinel
  must pass before any full score/memory plan; full admission requires a
  separate full `N=10000,T=1000` artifact.
- Nonclaims: no full actual-SV score admission and no full score run.

Actions:

- Added actual-SV same-target score adapter.
- Added Phase 5 actual-SV score contract tests.
- Ran the tiny score CLI diagnostic at `T=2,N=64`.
- Wrote Phase 5 result and a full-row score/memory repair subplan.

Artifacts:

- `docs/benchmarks/benchmark_ledh_same_target_actual_sv_score.py`
- `tests/highdim/test_ledh_actual_sv_score_phase5_contract.py`
- `docs/plans/bayesfilter-ledh-score-per-model-phase5-actual-sv-tiny-score-diagnostic-2026-07-07.json`
- `docs/plans/bayesfilter-ledh-score-per-model-phase5-actual-sv-tiny-score-diagnostic-2026-07-07.md`
- `docs/plans/bayesfilter-ledh-score-per-model-phase5-actual-sv-result-2026-07-07.md`
- `docs/plans/bayesfilter-ledh-score-per-model-phase5-actual-sv-full-row-score-subplan-2026-07-07.md`

Local checks:

- Focused Phase 5 score tests: `7 passed, 2 warnings`.
- Combined Phase 5 replay/schema checks: `28 passed, 2 warnings`.
- Tiny score diagnostic:
  `max_abs_error=8.720567601372409e-10`,
  `max_rel_error=3.3272540035606193e-09`,
  `score_admission_status=tiny_score_diagnostic_not_admitted`.

Gate status:

- `PHASE5_TINY_PASSED_FULL_ROW_SUBPLAN_PENDING_REVIEW`

Next action:

- Review the Phase 5 result and full-row score/memory subplan before any
  trusted GPU full-row actual-SV score command.

### 2026-07-07 - Phase 5 Actual-SV Full-Row Subplan - REVIEW_ACCEPTED_MEMORY_AUDIT_MAY_START

Evidence contract:

- Question: Is it safe to continue from the actual-SV tiny score result to the
  full-row score/memory subplan?
- Primary criterion: read-only review agrees tiny diagnostic is not promoted,
  exact transformed actual-SV target boundaries are preserved, and full-row
  execution must start with a memory-risk audit.
- Nonclaims: no full actual-SV score admission and no full score run.

Review:

- Direct bounded Claude packet review returned `VERDICT: AGREE`.
- Review caveat: boundary-safe to continue to the full-row score/memory
  subplan, but not safe to skip the memory-risk audit and run the full row
  directly.

Artifacts:

- `docs/reviews/bayesfilter-ledh-score-per-model-phase5-actual-sv-result-full-row-subplan-review-bundle-2026-07-07.md`

Gate status:

- `PASSED_PHASE5_FULL_ROW_SUBPLAN_REVIEW_MEMORY_AUDIT_MAY_START`

Next action:

- Execute the full-row subplan memory-risk audit. Do not run full
  `N=10000,T=1000` unless the audit supports a reviewed bounded command.

### 2026-07-07 - Phase 5 Actual-SV Full-Row Gate - BLOCKED_BY_SAME_ALGORITHM_PARITY

Evidence contract:

- Question: Can the tiny-passing actual-SV score route proceed to a full-row
  score/memory run?
- Primary criterion: full-row execution requires same-forward-scalar and
  same-algorithm parity with the admitted value route before memory/runtime
  evidence matters.
- Nonclaims: no full actual-SV score admission and no full score run.

Actions:

- Audited retained reverse-scan records and estimated full payload at about
  `1.42 GiB` per seed before TensorFlow/object overhead.
- Ran a targeted tiny parity probe between the admitted value route and current
  score forward route at `T=2,N=64`.
- Found score route uses the matrix aux flow primitive while the admitted value
  route uses the streaming flow primitive.
- Wrote blocker result and streaming-flow parity repair subplan.

Parity result:

```text
value_route   = [-3.603378542893297]
score_forward = [-3.6034006908781437]
diff          = [-2.214798484656555e-05]
```

Artifacts:

- `docs/plans/bayesfilter-ledh-score-per-model-phase5-actual-sv-full-row-score-result-2026-07-07.md`
- `docs/plans/bayesfilter-ledh-score-per-model-phase5-actual-sv-streaming-flow-parity-repair-subplan-2026-07-07.md`

Gate status:

- `PHASE5_FULL_ROW_BLOCKED_STREAMING_FLOW_PARITY_REPAIR_PENDING_REVIEW`

Next action:

- Review and then execute the streaming-flow parity repair subplan. Do not run
  full actual-SV score/memory until parity and tiny FD pass on the repaired
  route.

### 2026-07-07 - Phase 5 Actual-SV Streaming-Flow Parity Repair - TINY_PASSED_FULL_NOT_ADMITTED

Evidence contract:

- Question: Can the actual-SV score route differentiate the exact same
  finite-`N` streaming-flow scalar used by the admitted value route at tiny
  diagnostic scale?
- Primary criterion: score-route forward scalar must match the value route
  before all-coordinate FD score evidence is used.
- Nonclaims: no full `N=10000,T=1000` score admission, no HMC readiness, no
  posterior correctness, no runtime ranking.

Review:

- `claude_review_gate.sh` returned `REVIEW_STATUS=probe_timeout`.
- Direct tiny Claude probe returned `CLAUDE_PROBE_OK`.
- Direct bounded read-only review of
  `docs/reviews/bayesfilter-ledh-score-per-model-phase5-streaming-flow-parity-repair-review-bundle-2026-07-07.md`
  returned `VERDICT: AGREE`.

Actions:

- Repaired the score adapter's LEDH flow primal to mirror the admitted
  streaming value route's chunking, padding, core arithmetic, and initial
  particle law.
- Repaired the score adapter's transport primal to use raw streaming transport
  for forward particles/log weights while retaining manual no-tape VJP for
  reverse cotangents.
- Added explicit same-forward-scalar parity and particle-chunk padding tests.
- Regenerated a repaired tiny score diagnostic artifact.

Checks:

```text
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/highdim/test_ledh_actual_sv_score_phase5_contract.py -q
```

Result: `9 passed, 2 warnings`.

```text
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/highdim/test_ledh_actual_sv_score_phase5_contract.py \
  tests/highdim/test_ledh_phase5_actual_sv_forward_scalar_artifact.py \
  tests/highdim/test_ledh_score_contract_phase1.py -q
```

Result: `30 passed, 2 warnings`.

Repaired tiny diagnostic:

```text
score = [-0.13676240070260542, 0.38478843496586546]
fd_score = [-0.1367604994584326, 0.38480405004648327]
max_abs_error = 1.561508061781458e-05
max_rel_error = 4.057930423530709e-05
score_admission_status = tiny_score_diagnostic_not_admitted
```

Artifacts:

- `docs/benchmarks/benchmark_ledh_same_target_actual_sv_score.py`
- `tests/highdim/test_ledh_actual_sv_score_phase5_contract.py`
- `docs/plans/bayesfilter-ledh-score-per-model-phase5-actual-sv-streaming-flow-parity-repair-result-2026-07-07.md`
- `docs/plans/bayesfilter-ledh-score-per-model-phase5-actual-sv-streaming-parity-tiny-score-diagnostic-2026-07-07.json`
- `docs/plans/bayesfilter-ledh-score-per-model-phase5-actual-sv-streaming-parity-tiny-score-diagnostic-2026-07-07.md`
- `docs/plans/bayesfilter-ledh-score-per-model-phase5-actual-sv-full-row-score-subplan-refresh-2026-07-07.md`

Gate status:

- `PHASE5_STREAMING_PARITY_REPAIR_PASSED_FULL_ROW_REFRESHED_SUBPLAN_PENDING_REVIEW`

Next action:

- Review the refreshed Phase 5 full-row score/memory subplan before any
  trusted GPU ladder or full `N=10000,T=1000` score command.

### 2026-07-07 - Phase 5 Actual-SV Full-Row Refresh - BLOCKED_BY_RUNTIME_MEMORY_SCALING

Evidence contract:

- Question: Can the repaired actual-SV no-tape total score route scale to full
  `N=10000,T=1000` with replayable correctness and memory evidence?
- Primary criterion: full score artifact must validate with full admission;
  ladder runs are diagnostic only.
- Nonclaims: no full actual-SV score admission, no HMC readiness, no posterior
  correctness, no runtime ranking.

Review:

- Direct bounded read-only review of
  `docs/reviews/bayesfilter-ledh-score-per-model-phase5-actual-sv-full-row-score-refresh-review-bundle-2026-07-07.md`
  returned `VERDICT: AGREE`.

Memory audit:

- Repaired stored-record reverse scan estimate:
  `1.464 MiB` per time step and `1.430 GiB` per seed at
  `T=1000,N=10000,float64` before TensorFlow/object overhead.

Trusted GPU ladder:

- `nvidia-smi` passed before ladder.
- Rung `T=5,N=256` passed as diagnostic evidence only:
  `elapsed_seconds=101.32655131816864`,
  `peak_mib=32.68115234375`,
  `max_abs_error=4.1991519822121015e-05`,
  `score_admission_status=tiny_score_diagnostic_not_admitted`.
- Rung `T=20,N=1024` was manually interrupted after roughly 15 minutes with
  near-budget GPU memory pressure (`nvidia-smi` showed about
  `15711 MiB / 16376 MiB` in use). No JSON artifact was written.
- Traceback showed the run was inside the manual streaming finite transport
  total pullback, not in a target-scalar parity failure.

Artifacts:

- `docs/plans/bayesfilter-ledh-score-per-model-phase5-actual-sv-full-row-score-refresh-result-2026-07-07.md`
- `docs/plans/bayesfilter-ledh-score-per-model-phase5-actual-sv-gpu-ladder-t5-n256-2026-07-07.json`
- `docs/plans/bayesfilter-ledh-score-per-model-phase5-actual-sv-gpu-ladder-t5-n256-2026-07-07.md`
- `docs/plans/bayesfilter-ledh-score-per-model-phase5-actual-sv-full-row-scaling-repair-subplan-2026-07-07.md`

Gate status:

- `PHASE5_FULL_ROW_BLOCKED_RUNTIME_MEMORY_SCALING_REPAIR_SUBPLAN_PENDING_REVIEW`

Next action:

- Review the Phase 5 full-row scaling repair subplan. Do not rerun full
  `N=10000,T=1000`; first add no-FD value/score-only diagnostics and
  per-stage instrumentation or a reviewed exact-reference correctness path.

### 2026-07-07 - Phase 5 Actual-SV Scaling Repair - BLOCKED_BY_SINGLE_TRANSPORT_VJP_SCALING

Evidence contract:

- Question: Can we produce a validator-compatible full-row actual-SV score
  evidence path without repeated full-row all-coordinate FD blowup?
- Primary criterion: identify a path that makes full correctness feasible or
  supplies a reviewed exact-reference route without weakening boundaries.
- Nonclaims: no full actual-SV score admission, no HMC readiness, no posterior
  correctness.

Review:

- Direct bounded read-only review of
  `docs/reviews/bayesfilter-ledh-score-per-model-phase5-actual-sv-full-row-scaling-repair-review-bundle-2026-07-07.md`
  returned `VERDICT: AGREE`.

Actions:

- Added `--diagnostic-mode value-score-only` to
  `docs/benchmarks/benchmark_ledh_same_target_actual_sv_score.py`.
- Added tests that value-score-only artifacts are blocked/non-admission.
- Ran focused tests: `29 passed, 2 warnings`.

Diagnostics:

- CPU-hidden tiny value-score-only smoke at `T=2,N=64` completed in
  `5.361246170010418` seconds and wrote a blocked artifact.
- Trusted GPU value-score-only diagnostic at `T=20,N=1024` was manually
  interrupted after about 9 minutes. No JSON artifact was written.
- During the no-FD run, `nvidia-smi` showed about
  `15763 MiB / 16376 MiB` in use.
- Traceback identified the blocker inside manual streaming finite transport
  total pullback / column-normalizer VJP.

Artifacts:

- `docs/plans/bayesfilter-ledh-score-per-model-phase5-actual-sv-full-row-scaling-repair-result-2026-07-07.md`
- `docs/plans/bayesfilter-ledh-score-per-model-phase5-actual-sv-value-score-only-tiny-smoke-2026-07-07.json`
- `docs/plans/bayesfilter-ledh-score-per-model-phase5-actual-sv-value-score-only-tiny-smoke-2026-07-07.md`
- `docs/plans/bayesfilter-ledh-score-per-model-phase5-actual-sv-transport-vjp-scaling-subplan-2026-07-07.md`

Gate status:

- `PHASE5_TRANSPORT_VJP_SCALING_SUBPLAN_PENDING_REVIEW`

Next action:

- Review the transport VJP scaling subplan. Do not run larger actual-SV score
  ladders until the transport VJP total-pullback blocker is repaired or
  explicitly blocked.

### 2026-07-07 - Phase 5 Actual-SV Transport VJP Scaling Subplan - REVIEW_ACCEPTED

Evidence contract:

- Question: Is it boundary-safe to target the manual streaming transport VJP
  total-pullback implementation before any larger actual-SV score ladder?
- Primary criterion: review agrees the subplan preserves same-forward-scalar,
  no-tape, no-stopped-partial, and non-admission boundaries.

Review:

- Direct bounded read-only review of
  `docs/reviews/bayesfilter-ledh-score-per-model-phase5-actual-sv-transport-vjp-scaling-review-bundle-2026-07-07.md`
  returned `VERDICT: AGREE`.
- Review caveat: blockwise VJP must not be silently treated as admissible
  merely because it is cheaper; any semantic change must stop for review.

Artifacts:

- `docs/plans/bayesfilter-ledh-score-per-model-phase5-actual-sv-transport-vjp-scaling-subplan-2026-07-07.md`
- `docs/reviews/bayesfilter-ledh-score-per-model-phase5-actual-sv-transport-vjp-scaling-review-bundle-2026-07-07.md`

Gate status:

- `PHASE5_TRANSPORT_VJP_SCALING_SUBPLAN_REVIEW_ACCEPTED_READY_TO_EXECUTE`

Next action:

- Execute the transport VJP scaling subplan: map the current total-pullback
  call chain, add tiny standalone transport VJP diagnostics, and compare only
  reviewed semantics-preserving candidates before any moderate GPU diagnostic.
