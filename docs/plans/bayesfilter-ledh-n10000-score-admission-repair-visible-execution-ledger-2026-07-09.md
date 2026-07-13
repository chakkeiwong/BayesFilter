# LEDH N10000 Score Admission Repair Visible Execution Ledger

Date: 2026-07-09

Status: `PHASE0_REVIEW_PACKET_REPAIR_IN_PROGRESS`

## Ledger

### 2026-07-09 - Launch Package - DRAFTED

Evidence contract:

- Question: Can all main LEDH rows obtain validator-admitted compact
  `N=10000` score artifacts without changing the value target scalar?
- Primary criterion: launch package includes master program, visible runbook,
  Phase 0 subplan, review bundle, explicit admission contract, and stop
  conditions before full runs.
- Non-claims: no score admitted, no full score run, no leaderboard rebuilt.

Actions:

- Drafted successor launch package.
- Claude review gate rejected external Claude review as data-disclosure risk.
- Fresh Codex substitute launch review returned `VERDICT: AGREE`.

### 2026-07-09 - Phase 0 - PASSED_INVENTORY_GATE_PENDING_REVIEW

Evidence contract:

- Question: What exact artifact and route gaps prevent current `N=10000`
  score evidence from being admitted?
- Primary criterion: every main LEDH row is listed with value artifact, score
  runner, current score evidence, admission status, and smallest next action.
- Non-claims: no score admission, no memory pass, no full-row correctness.

Actions:

- Ran static route/admission inventory.
- Confirmed required value artifacts exist.
- Parsed value and current score evidence summaries.
- Ran focused score-contract and leaderboard tests.
- Wrote Phase 0 result and Phase 1 subplan.

Local checks:

- Required artifact existence checks passed.
- `CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest tests/highdim/test_ledh_score_contract_phase1.py tests/test_two_lane_highdim_ledh_leaderboard.py -q`
  passed: `34 passed, 2 warnings`.

Artifacts:

- `docs/plans/bayesfilter-ledh-n10000-score-admission-repair-phase0-launch-inventory-result-2026-07-09.md`
- `docs/plans/bayesfilter-ledh-n10000-score-admission-repair-phase1-shared-emitter-subplan-2026-07-09.md`

Gate status:

- `PASSED_PENDING_REVIEW`

### 2026-07-09 - Phase 0 Review Round 1 - REVISE_REVIEW_VISIBILITY

Claude was again rejected by local policy. A fresh Codex substitute review
returned `VERDICT: REVISE` because it could not see uncommitted July 9 plan
files and reviewed stale file content. This is classified as a
`fixable_review_packet_visibility_issue`, not a substantive Phase 0 blocker.

Repair action:

- Recreate missing plan/result artifacts.
- Use a packet-only review containing the relevant excerpts directly.

### 2026-07-09 - Phase 0 Review Round 2 - PACKET_AGREE

Evidence contract:

- Question: Did Phase 0 close inventory and is Phase 1 safe to execute before
  full `N=10000` score runs?
- Primary criterion: packet-only read-only review finds no material blocker.
- Veto diagnostics: proxy raw/tiny/manual evidence admitted; missing row
  inventory; Phase 1 not appropriate before full runs.
- Non-claims: packet-only review is not full code inspection and admits no
  score rows.

Actions:

- Closed stale substitute reviewer.
- Recreated missing July 9 plan/result artifacts.
- Ran a packet-only fresh Codex read-only review with Phase 0 and Phase 1
  excerpts embedded directly in the prompt.
- Review returned `VERDICT: AGREE`.

Review note:

- Reviewer recommended preserving a per-row certification ledger in Phase 1
  result, with value artifact path, score artifact path, validation status, and
  exact mismatch reason for failures.

Gate status:

- `PASSED`

Next action:

- Execute Phase 1 shared emitter/certification precheck.

### 2026-07-09 - Phase 1 - PASSED_SHARED_EMITTER_GATE_PENDING_REVIEW

Evidence contract:

- Question: Is there now a shared, tested way to emit full-admission score
  artifacts so future full runs cannot accidentally produce raw legacy
  evidence?
- Primary criterion: a shared helper or certified wrapper builds artifacts
  that pass `validate_ledh_score_artifact(..., require_admitted=True)` only
  when all full-admission inputs are present and compact.
- Veto diagnostics: historical route admitted; raw legacy JSON admitted;
  missing memory pass admitted; row or parameter mismatch admitted; target
  scalar changed; per-model builders bypass validation.
- Non-claims: no full `N=10000` score run, no score leaderboard completion,
  no HMC readiness, no posterior correctness, no runtime ranking.

Actions:

- Added `bayesfilter/highdim/ledh_score_artifact.py`.
- Added `tests/highdim/test_ledh_score_artifact_emitter_phase1.py`.
- Ran compile checks and focused score-contract/per-model tests.
- Wrote Phase 1 result and Phase 2 LGSSM subplan.

Local checks:

- `python -m py_compile ...` passed.
- Focused pytest suite passed: `104 passed, 2 warnings`.

Artifacts:

- `bayesfilter/highdim/ledh_score_artifact.py`
- `tests/highdim/test_ledh_score_artifact_emitter_phase1.py`
- `docs/plans/bayesfilter-ledh-n10000-score-admission-repair-phase1-shared-emitter-result-2026-07-09.md`
- `docs/plans/bayesfilter-ledh-n10000-score-admission-repair-phase2-lgssm-subplan-2026-07-09.md`
- `docs/reviews/bayesfilter-ledh-n10000-score-admission-repair-phase1-result-phase2-subplan-review-bundle-2026-07-09.md`

Gate status:

- `PASSED_PENDING_REVIEW`

Next action:

- Packet-only Codex substitute review of Phase 1 result and Phase 2 subplan.

### 2026-07-09 - Phase 1 Review - PACKET_AGREE

Evidence contract:

- Question: Is Phase 1 acceptable, and is Phase 2 LGSSM safe to execute before
  a trusted full `N=10000` run?
- Primary criterion: read-only packet review finds no material blocker.
- Veto diagnostics: helper admits independently of validator; negative tests
  miss raw/tiny/historical cases; Phase 2 promotes July 6 raw `primary_pass`;
  Phase 2 lacks trusted GPU/validator stop conditions.
- Non-claims: review does not admit LGSSM score and does not authorize
  non-LGSSM claims.

Actions:

- Packet-only Codex substitute review returned `VERDICT: AGREE`.

Review summary:

- Helper preserves validator as admission authority.
- Negative tests cover the important failure modes.
- Phase 2 avoids promoting the July 6 raw `primary_pass` JSON.
- No blocker before trusted LGSSM full run, assuming trusted/escalated GPU
  access and post-run validator admission.

Gate status:

- `PASSED`

Next action:

- Execute Phase 2 LGSSM precheck and trusted full run.

### 2026-07-09 - Phase 2 - BLOCKED_FIXABLE_RUNTIME_MEMORY_REPAIR_REQUIRED

Evidence contract:

- Question: Can LGSSM produce a schema-valid compact `N=10000,T=50` score
  artifact admitted by the shared score validator?
- Primary criterion: output contains a score artifact admitted by
  `validate_ledh_score_artifact(..., require_admitted=True)` with compact
  LGSSM provenance and `memory_diagnostics.n10000_memory_pass = true`.
- Veto diagnostics: wrong scalar; row/value mismatch; raw legacy JSON reused;
  manual route; missing memory pass; nonfinite score; FD failure;
  CPU/sandbox result misreported as trusted GPU full run.
- Non-claims: no LGSSM score admission and no conclusion about compact score
  mathematics.

Actions:

- Precheck confirmed admitted LGSSM value artifact:
  `benchmark_lgssm_exact_oracle_m3_T50 10000 50 [81120, 81121, 81122, 81123, 81124]`.
- Launched reviewed trusted GPU LGSSM full command.
- Trusted status showed GPU device creation and XLA compilation.
- Trusted memory polling showed about `15.7 GiB` used, above the 14 GiB budget.
- No output artifact was emitted after prolonged execution.
- Codex interrupted the run to avoid unbounded GPU burn under an already
  over-budget memory observation.
- Wrote Phase 2 blocker result and Phase 2R repair subplan.

Artifacts:

- `docs/plans/logs/bayesfilter-ledh-n10000-score-admission-repair-phase2-lgssm-run-2026-07-09.log`
- `docs/plans/bayesfilter-ledh-n10000-score-admission-repair-phase2-lgssm-blocker-result-2026-07-09.md`
- `docs/plans/bayesfilter-ledh-n10000-score-admission-repair-phase2r-lgssm-memory-repair-subplan-2026-07-09.md`
- `docs/reviews/bayesfilter-ledh-n10000-score-admission-repair-phase2-blocker-phase2r-review-bundle-2026-07-09.md`

Gate status:

- `BLOCKED_FIXABLE_PENDING_REVIEW`

Next action:

- Review Phase 2 blocker and Phase 2R smaller-chunk/reduce-only repair plan.

### 2026-07-09 - Phase 2R Review - PACKET_AGREE

Evidence contract:

- Question: Is the Phase 2 blocker fairly classified, and is a smaller-chunk
  trusted retry a safe minimal repair?
- Primary criterion: read-only review agrees no material blocker before the
  smaller-chunk retry.
- Veto diagnostics: overclaiming math failure; changing target/admission
  criteria; raising memory budget; arbitrary chunk sweeps.
- Non-claims: review does not admit LGSSM score.

Actions:

- Packet-only Codex substitute review returned `VERDICT: AGREE`.

Review summary:

- Blocker classification is fair as runtime/memory.
- Smaller chunks are a safe minimal repair.
- If smaller chunks fail, stop for reduce-only streaming implementation plan.

Gate status:

- `PASSED`

Next action:

- Execute trusted smaller-chunk LGSSM retry.

### 2026-07-09 - Phase 2R - BLOCKED_FIXABLE_PROCEDURAL_MEMORY_REPAIR_REQUIRED

Evidence contract:

- Question: Can smaller chunks make the LGSSM compact score run emit an
  admitted `N=10000,T=50` score artifact within memory budget?
- Primary criterion: artifact validates with
  `validate_ledh_score_artifact(..., require_admitted=True)` and records a
  score-memory pass.
- Veto diagnostics: no artifact; no validator admission; no trusted full
  score-memory evidence.
- Non-claims: no LGSSM score admission, no compact-math rejection, no
  leaderboard completion.

Actions:

- Executed the reviewed trusted smaller-chunk retry with row/column/particle
  chunks of 128.
- Log confirmed trusted GPU device creation, CUDA/XLA initialization, cuDNN
  load, and XLA compilation.
- No JSON score artifact was emitted in the reviewed window.
- Closed Phase 2R as a fixable procedural/runtime blocker.
- Drafted Phase 2S score-procedure repair subplan.

Artifacts:

- `docs/plans/logs/bayesfilter-ledh-n10000-score-admission-repair-phase2r-lgssm-smaller-chunks-2026-07-09.log`
- `docs/plans/bayesfilter-ledh-n10000-score-admission-repair-phase2r-lgssm-memory-repair-result-2026-07-09.md`
- `docs/plans/bayesfilter-ledh-n10000-score-admission-repair-phase2s-lgssm-score-procedure-repair-subplan-2026-07-09.md`
- `docs/reviews/bayesfilter-ledh-n10000-score-admission-repair-phase2r-result-phase2s-subplan-review-bundle-2026-07-09.md`

Root-cause refinement:

- The runner's coordinate-wise same-scalar FD currently calls the compact
  score/JVP route for each plus/minus scalar instead of a value-only scalar
  route.
- The score artifact memory gate currently reads value-route memory measured
  before the score diagnostic, not score-specific memory.
- Transport JVP TensorArray stacking remains a possible implementation hot
  spot, but Phase 2S will repair and measure the procedural issues first.

Gate status:

- `BLOCKED_FIXABLE_PENDING_REVIEW`

Next action:

- Run read-only review for Phase 2R result and Phase 2S subplan. If Claude is
  blocked by local external-disclosure policy, use fresh Codex packet-only
  review and record the limitation.

### 2026-07-09 - Phase 2S Review - CODEX_PACKET_AGREE_AFTER_CLAUDE_POLICY_BLOCK

Evidence contract:

- Question: Is Phase 2R fairly closed and is Phase 2S safe before code edits?
- Primary criterion: read-only review finds no material blocker to the
  procedural repair phase.
- Veto diagnostics: continuing without validator admission; target drift;
  missing tests for FD-via-score-route and score-memory route copying.
- Non-claims: review does not admit LGSSM score and does not establish
  scientific or HMC readiness.

Actions:

- Attempted Claude review gate with bounded repo-local bundle.
- Local policy rejected the Claude call as external repository disclosure
  despite prior user approval.
- Did not route around the policy rejection.
- Spawned a fresh Codex packet-only read-only reviewer.
- Fresh review returned `VERDICT: AGREE`.

Review condition to carry into implementation:

- Add tests or smoke checks that fail if same-scalar FD calls
  `_compact_value_and_score_from_components`.
- Add tests that score artifact memory fields come from score-specific memory,
  not copied value-route memory.

Gate status:

- `PASSED`

Next action:

- Implement Phase 2S scoped repair and run focused tests.

### 2026-07-09 - Phase 2S - BLOCKED_FIXABLE_TF32_CORRECTNESS_GATE

Evidence contract:

- Question: Can LGSSM score admission be unblocked by repairing score procedure
  and score-memory measurement before another full run?
- Primary criterion: full Phase 2S artifact validates and reports
  score-specific memory pass.
- Veto diagnostics: default TF32 trusted GPU smoke same-scalar FD failure.
- Non-claims: no LGSSM score admission, no compact-math rejection, no
  leaderboard completion.

Actions:

- Implemented value-only same-scalar FD route.
- Implemented score-specific GPU memory reset and
  `score_gpu_memory_info_before/after`.
- Changed score artifact memory diagnostics to use score memory, not value
  memory.
- Added nested score artifact emission for full-row score runs.
- Added tests for FD not calling the score route and score memory not being
  copied from value-route memory.
- Ran focused CPU tests: `52 passed, 2 warnings`.
- Ran CPU prefix smoke: passed as non-admitted diagnostic.
- Ran trusted GPU smoke with TF32 enabled: memory instrumentation worked but
  same-scalar FD failed.
- Ran trusted GPU smoke with TF32 disabled: same-scalar FD passed.
- Ran bounded TF32 FD-step diagnostic: steps `0.001`, `0.003`, and `0.01`
  remained failing; larger-step sweep was interrupted as too slow.

Artifacts:

- `docs/plans/bayesfilter-ledh-n10000-score-admission-repair-phase2s-lgssm-score-procedure-repair-result-2026-07-09.md`
- `docs/plans/bayesfilter-ledh-n10000-score-admission-repair-phase2t-lgssm-tf32-correctness-policy-subplan-2026-07-09.md`
- `docs/plans/logs/bayesfilter-ledh-n10000-score-admission-repair-phase2s-prefix-smoke-2026-07-09.log`
- `docs/plans/logs/bayesfilter-ledh-n10000-score-admission-repair-phase2s-gpu-smoke-2026-07-09.log`
- `docs/plans/logs/bayesfilter-ledh-n10000-score-admission-repair-phase2s-gpu-smoke-no-tf32-2026-07-09.log`

Gate status:

- `BLOCKED_FIXABLE_PENDING_REVIEW`

Next action:

- Review Phase 2S result and Phase 2T correctness-policy subplan before any
  full `N=10000,T=50` LGSSM score run.

### 2026-07-09 - Phase 2T Review - CODEX_PACKET_AGREE_AFTER_CLAUDE_POLICY_BLOCK

Evidence contract:

- Question: Is Phase 2T safe as the next repair after default-TF32 smoke
  correctness failure?
- Primary criterion: read-only review agrees Phase 2T can execute before any
  full `N=10000,T=50` run.
- Veto diagnostics: tolerance loosening without mismatch sensitivity; exact
  Kalman overclaim; undisclosed no-TF32 correctness arm.
- Non-claims: review does not admit LGSSM score.

Actions:

- Attempted Claude review gate and hit the same local external-disclosure
  rejection.
- Used fresh Codex packet-only review.
- Review returned `VERDICT: AGREE`.

Review constraints to carry forward:

- Forbid any "just loosen tolerance" policy without route-mismatch sensitivity.
- Forbid exact Kalman/LGSSM reference claims that certify only ideal-model score
  rather than the same finite LEDH estimator derivative.
- Forbid silently using no-TF32 correctness evidence to admit TF32 production
  behavior without disclosure and reviewed rationale.

Gate status:

- `PASSED`

Next action:

- Execute Phase 2T correctness-policy repair.

### 2026-07-09 - Phase 2T - PASSED_POLICY_SMOKE_PENDING_PHASE2U_REVIEW

Evidence contract:

- Question: What correctness policy can validly certify the compact no-tape
  LGSSM score when default TF32 FD fails?
- Primary criterion: reviewed policy passes trusted GPU smoke and discloses the
  precision split.
- Non-claims: no full LGSSM score admission yet.

Actions:

- Added `--score-fd-tf32-mode`.
- Implemented disclosed separate-precision correctness-arm metadata.
- Added tests for no-TF32 FD-arm disclosure.
- Ran focused tests: `42 passed, 2 warnings`.
- Ran trusted GPU smoke with production TF32 enabled and FD TF32 disabled:
  same-scalar FD passed with max abs error `0.006190299987792969`, max relative
  error `0.0004953071475028992`; metadata disclosed the precision arm.
- Drafted Phase 2U full-run subplan.

Artifacts:

- `docs/plans/bayesfilter-ledh-n10000-score-admission-repair-phase2t-lgssm-tf32-correctness-policy-result-2026-07-09.md`
- `docs/plans/bayesfilter-ledh-n10000-score-admission-repair-phase2u-lgssm-full-run-subplan-2026-07-09.md`
- `docs/plans/logs/bayesfilter-ledh-n10000-score-admission-repair-phase2t-gpu-smoke-2026-07-09.log`

Gate status:

- `PASSED_PENDING_PHASE2U_REVIEW`

Next action:

- Review Phase 2U full-run subplan before launching the full trusted run.

### 2026-07-09 - Phase 2U Review - CODEX_PACKET_AGREE_AFTER_CLAUDE_POLICY_BLOCK

Evidence contract:

- Question: May the full trusted LGSSM `N=10000,T=50` score run launch?
- Primary criterion: read-only review agrees the command preserves the
  admission contract and stop conditions.
- Non-claims: review does not admit LGSSM score.

Actions:

- Attempted Claude review gate; local policy rejected external disclosure.
- Used fresh Codex packet-only review.
- Review returned `VERDICT: AGREE`.

Gate status:

- `PASSED`

Next action:

- Launch full trusted GPU run and monitor memory/artifact creation.

### 2026-07-09 - Phase 2U - BLOCKED_FIXABLE_FULL_RUNTIME_NO_ARTIFACT

Evidence contract:

- Question: Can LGSSM emit a validator-admitted compact `N=10000,T=50` score
  artifact under the repaired score procedure and reviewed correctness policy?
- Primary criterion: nested `score_artifact` validates with
  `validate_ledh_score_artifact(..., require_admitted=True)`.
- Veto diagnostic observed: no artifact was emitted.
- Non-claims: no LGSSM score admission; no compact-math rejection; no exact
  Kalman score claim.

Actions:

- Launched the reviewed trusted GPU full-row command.
- TensorFlow initialized `/GPU:0` and compiled XLA.
- During the bounded monitoring window, trusted GPU memory was about
  `15744-15754 MiB / 16376 MiB`.
- No output artifact existed at
  `docs/plans/bayesfilter-ledh-n10000-score-admission-repair-phase2u-lgssm-score-artifact-2026-07-09.json`.
- Interrupted and killed shell PID `229799` and Python PID `229800`.
- Trusted post-kill checks showed no remaining compute app and GPU memory back
  near `2057 MiB / 16376 MiB`.

Artifacts:

- `docs/plans/logs/bayesfilter-ledh-n10000-score-admission-repair-phase2u-lgssm-full-run-2026-07-09.log`
- `docs/plans/bayesfilter-ledh-n10000-score-admission-repair-phase2u-lgssm-full-run-result-2026-07-09.md`
- `docs/plans/bayesfilter-ledh-n10000-score-admission-repair-phase2v-lgssm-sharded-admission-subplan-2026-07-09.md`

Gate status:

- `BLOCKED_FIXABLE_PENDING_PHASE2V_REVIEW`

Next action:

- Review Phase 2U result and Phase 2V seed-sharded admission subplan before
  implementing sharded aggregation or launching new full shards.

### 2026-07-09 - Phase 2V Review - CODEX_PACKET_AGREE_AFTER_CLAUDE_POLICY_BLOCK

Evidence contract:

- Question: Is Phase 2V a safe next procedural repair after the Phase 2U
  monolithic no-artifact blocker?
- Primary criterion: read-only review agrees the subplan preserves the
  same-target score contract and does not promote partial shards.
- Non-claims: review does not admit LGSSM score.

Actions:

- Attempted the bounded Claude review gate using
  `docs/reviews/bayesfilter-ledh-n10000-score-admission-repair-phase2u-result-phase2v-subplan-review-bundle-2026-07-09.md`.
- Local policy rejected the Claude call as external repository disclosure.
- Did not route around the policy rejection.
- Used a fresh Codex packet-only read-only review.
- Review returned `VERDICT: AGREE`.

Review condition to carry into implementation:

- The aggregate artifact must explicitly represent the full fixed seed set.
- Per-seed raw shards must remain diagnostic and must not be normalized into
  admitted artifacts.

Gate status:

- `PASSED`

Next action:

- Implement Phase 2V shard aggregation, run local checks, then trusted GPU
  shard smoke before any full shard launch.

### 2026-07-09 - Phase 2V Local Checks And Smoke - PASSED_PENDING_FULL_SHARDS

Evidence contract:

- Question: Does the Phase 2V implementation preserve exact shard aggregation
  semantics before full `N=10000,T=50` shards?
- Primary criterion: focused tests pass and trusted GPU shard smoke passes.
- Non-claims: no LGSSM score admission yet; synthetic shard aggregation is not
  real score evidence.

Actions:

- Added a seed-sharded aggregate builder and CLI aggregation mode to
  `docs/benchmarks/benchmark_ledh_same_target_lgssm_m3_t50_value.py`.
- Added tests for full fixed-seed coverage, missing/duplicate/admitted-shard
  rejection, identity/runtime mismatch rejection, and tiny direct-batch versus
  shard-aggregate parity.
- Ran focused CPU-hidden tests:
  `46 passed, 2 warnings`.
- Ran `py_compile` on the LGSSM runner.
- Ran synthetic CLI aggregation smoke; it emitted a validator-valid synthetic
  aggregate with segmented execution disclosure.
- Ran trusted GPU one-seed prefix shard smoke at `N=256,T=3`; FD status passed
  and shard stayed non-admitted as expected.

Gate status:

- `PASSED_PENDING_FULL_SHARDS`

Skeptical audit before full shards:

- Target is still `observed_data_log_likelihood_estimator` / `log_likelihood`,
  not exact Kalman likelihood.
- Full shard commands must use fixed full-row seed values, `N=10000`, `T=50`,
  active-all transport, Sinkhorn iterations `10`, epsilon `0.5`, compact
  no-tape score route, production TF32 enabled, and disclosed no-TF32 FD arm.
- Per-seed shard completion is not admission. Only aggregate validator
  admission can admit LGSSM.
- Segmented max-shard memory is the memory claim; no monolithic batch memory
  claim is allowed.

Next action:

- Launch full trusted GPU shards one seed at a time, preserving durable raw
  shard artifacts and logs.

### 2026-07-09 - Phase 2V - BLOCKED_FIXABLE_SINGLE_SHARD_FULL_RUNTIME_NO_ARTIFACT

Evidence contract:

- Question: Can exact seed-sharded aggregation admit LGSSM without changing the
  target scalar?
- Primary criterion: aggregate nested score artifact validates with
  `validate_ledh_score_artifact(..., require_admitted=True)`.
- Veto observed: first full shard did not emit a raw shard artifact.
- Non-claims: no LGSSM admission; no rejection of compact score mathematics.

Actions:

- Implemented seed-sharded aggregation and CLI aggregation mode.
- Added tests for exact seed coverage, shard non-admission, identity/runtime
  rejection, and tiny direct-batch versus shard-aggregate parity.
- Ran focused tests: `46 passed, 2 warnings`.
- Ran synthetic aggregation CLI smoke: passed.
- Ran trusted GPU prefix shard smoke at `N=256,T=3`: passed and remained
  non-admitted.
- Launched full trusted GPU shard for seed `81120`.
- The shard initialized trusted `/GPU:0`, compiled XLA, and held about
  `15738 MiB / 16376 MiB`, but emitted no raw artifact after about six minutes.
- Stopped PID `251240`; post-stop trusted checks showed no compute app and GPU
  memory near `1879 MiB / 16376 MiB`.

Artifacts:

- `docs/plans/bayesfilter-ledh-n10000-score-admission-repair-phase2v-lgssm-sharded-admission-result-2026-07-09.md`
- `docs/plans/bayesfilter-ledh-n10000-score-admission-repair-phase2w-lgssm-score-fd-split-subplan-2026-07-09.md`
- `docs/plans/logs/bayesfilter-ledh-n10000-score-admission-repair-phase2v-lgssm-seed-81120.log`

Gate status:

- `BLOCKED_FIXABLE_PENDING_PHASE2W_REVIEW`

Next action:

- Review Phase 2V result and Phase 2W score/FD split subplan before additional
  code edits or full shard attempts.

### 2026-07-09 - Phase 2W Review - CODEX_PACKET_AGREE_AFTER_CLAUDE_POLICY_BLOCK

Evidence contract:

- Question: Is Phase 2W the correct next diagnostic after Phase 2V's
  single-shard no-artifact blocker?
- Primary criterion: read-only review agrees score/FD splitting preserves the
  admission contract and cannot admit split-stage artifacts.
- Non-claims: review does not admit LGSSM score.

Actions:

- Attempted the bounded Claude review gate.
- Local policy rejected the Claude call as external repository disclosure.
- Used fresh Codex packet-only review.
- Review returned `VERDICT: AGREE`.

Gate status:

- `PASSED`

Next action:

- Implement score/FD split diagnostics.

### 2026-07-09 - Phase 2W - BLOCKED_FIXABLE_COMPACT_SCORE_PASS_FULL_RUNTIME_NO_ARTIFACT

Evidence contract:

- Question: Is the full-shard no-artifact blocker caused by compact score
  execution, FD correctness execution, or the unsplit combination?
- Primary criterion: full trusted GPU score-only shard emits or writes a
  precise blocker.
- Veto observed: full score-only shard did not emit.
- Non-claims: no LGSSM admission; no compact-math rejection.

Actions:

- Added `--score-diagnostic-stage {score-and-fd,score-only,fd-only}` and
  `--score-reference-json`.
- Split score-only and FD-only diagnostics while preserving default
  score-and-FD behavior.
- Added tests ensuring score-only is non-admitted and FD-only uses the
  value-only scalar route without calling compact score/JVP.
- Ran focused tests: `48 passed, 2 warnings`.
- Ran trusted GPU score-only smoke at `N=256,T=3`: passed and non-admitted.
- Ran trusted GPU FD-only smoke at `N=256,T=3`: passed and non-admitted.
- Launched full trusted GPU score-only shard for seed `81120`.
- The full score-only shard initialized trusted `/GPU:0`, compiled XLA, and
  held about `15712 MiB / 16376 MiB`, but emitted no artifact after about
  6.75 minutes.
- Stopped PID `262046`; post-stop trusted checks showed no compute app and GPU
  memory near `1862 MiB / 16376 MiB`.

Artifacts:

- `docs/plans/bayesfilter-ledh-n10000-score-admission-repair-phase2w-lgssm-score-fd-split-result-2026-07-09.md`
- `docs/plans/bayesfilter-ledh-n10000-score-admission-repair-phase2x-lgssm-compact-score-kernel-repair-subplan-2026-07-09.md`
- `docs/plans/logs/bayesfilter-ledh-n10000-score-admission-repair-phase2w-score-only-seed-81120.log`

Gate status:

- `BLOCKED_FIXABLE_PENDING_PHASE2X_REVIEW`

Next action:

- Review Phase 2W result and Phase 2X compact score kernel repair subplan.

### 2026-07-09 - Phase 2X Review - CODEX_PACKET_AGREE_AFTER_CLAUDE_POLICY_BLOCK

Evidence contract:

- Question: Is Phase 2X the correct next kernel repair after Phase 2W narrowed
  the blocker to the full compact score pass?
- Primary criterion: read-only review agrees the subplan preserves the
  same-target score contract and has adequate tests/boundaries before
  implementation.
- Non-claims: review does not admit LGSSM score and does not approve exact
  Kalman substitution.

Actions:

- Attempted the bounded Claude review gate.
- Local policy rejected the Claude call as external repository disclosure.
- Used fresh Codex packet-only review.
- Review returned `VERDICT: AGREE`.

Review watch item:

- The generic score validator allows `exact_reference`, but the LGSSM builder
  hardcodes same-scalar FD and Phase 2X explicitly forbids exact-Kalman
  substitution. Carry this boundary through implementation.

Gate status:

- `PASSED`

Next action:

- Execute Phase 2X compact score kernel/tensor-lifetime repair.
