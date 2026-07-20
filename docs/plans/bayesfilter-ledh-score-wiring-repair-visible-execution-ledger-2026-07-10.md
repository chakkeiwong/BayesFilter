# LEDH Score Wiring Repair Visible Execution Ledger

Date: 2026-07-10

## Status

`PHASE9_FD_POLICY_CORRECTED_PARTIAL_CONTINUATION_REVIEW_REQUIRED`

## Role Contract

Codex is supervisor and executor. Claude is read-only reviewer only.

## Ledger

### 2026-07-10 - Phase 0 - PRECHECK

Evidence contract:

- Question: Does the master program correctly target the score wiring failures
  and prevent relabeling old routes as compact computation?
- Baseline/comparator: current code inventory and model-by-model
  classification from 2026-07-10.
- Primary criterion: master program, runbook, Phase 0 result, and Phase 1
  subplan exist and state compact/default route, precision, FD, review, and
  stop-condition gates.
- Veto diagnostics: missing model phase; hidden default float64 route; plan
  allowing historical route full admission; plan treating score-only memory as
  correctness; plan launching full GPU ladder before wiring tests.
- Non-claims: no code repair, no model score admission, no leaderboard
  completion, no HMC/scientific claim.

Actions:

- Created master program and Phase 0 subplan.
- Read Claude review gate guide and visible runbook template.

Artifacts:

- `docs/plans/bayesfilter-ledh-score-wiring-repair-master-program-2026-07-10.md`
- `docs/plans/bayesfilter-ledh-score-wiring-repair-phase0-launch-inventory-subplan-2026-07-10.md`

Gate status:

- `PASSED`

Next action:

- Advance to Phase 1 shared contract and precision gate.

### 2026-07-10 - Phase 0 - REVIEW

Actions:

- Claude review gate attempted and rejected by execution policy as external
  data disclosure.
- No workaround attempted.
- Fresh Codex read-only substitute review requested and completed.

Artifacts:

- `docs/reviews/bayesfilter-ledh-score-wiring-repair-launch-review-bundle-2026-07-10.md`
- `docs/plans/bayesfilter-ledh-score-wiring-repair-phase0-launch-inventory-result-2026-07-10.md`

Gate status:

- `PASSED`

Review verdict:

- `VERDICT: AGREE`

Next action:

- Begin Phase 1 precheck.

### 2026-07-10 - Phase 1 - ASSESS_GATE

Evidence contract:

- Question: Does the shared score contract prevent historical route full
  admission and provide reusable gates for compact route and precision defaults?
- Baseline/comparator: Phase 0 inventory and existing shared score contract.
- Primary criterion: tests prove historical routes cannot full-admit, compact
  route constants are the only full-admissible no-tape provenance, and
  production precision expectations are testable per model.
- Veto diagnostics: historical route full-admits; precision defaults not
  testable; stopped/partial/autodiff tokens accepted.
- Non-claims: no model runner repair; no GPU score memory; no leaderboard
  admission.

Actions:

- Added production score precision validator.
- Required `score_precision` metadata for full score admission.
- Added tests for missing precision, `float64`, and TF32-disabled full
  admission rejection.

Artifacts:

- `bayesfilter/highdim/ledh_score_contract.py`
- `bayesfilter/highdim/ledh_score_artifact.py`
- `tests/highdim/test_ledh_score_contract_phase1.py`
- `tests/highdim/test_ledh_score_artifact_emitter_phase1.py`
- `docs/plans/bayesfilter-ledh-score-wiring-repair-phase1-shared-contract-result-2026-07-10.md`
- `docs/plans/bayesfilter-ledh-score-wiring-repair-phase2-lgssm-subplan-2026-07-10.md`

Local checks:

- `python -m py_compile bayesfilter/highdim/ledh_score_contract.py bayesfilter/highdim/ledh_score_artifact.py`: passed.
- `pytest -q tests/highdim/test_ledh_score_contract_phase1.py tests/highdim/test_ledh_score_artifact_emitter_phase1.py`: `55 passed, 2 warnings`.

Gate status:

- `PASSED_AFTER_REPAIR`

Next action:

- Begin Phase 2 LGSSM compact default cleanup.

### 2026-07-10T05:10:53+08:00 - Phase 1 - REVIEW_REPAIR

Actions:

- Substitute review returned `VERDICT: REVISE`.
- Required explicit `score_precision.active_dtype` and
  `score_precision.tf_dtype` instead of defaulting them from `dtype`.
- Required full-admission compact provenance to match the score artifact row.
- Added focused tests for missing explicit precision fields and wrong-row
  compact provenance.

Artifacts:

- `bayesfilter/highdim/ledh_score_contract.py`
- `tests/highdim/test_ledh_score_contract_phase1.py`
- `docs/plans/bayesfilter-ledh-score-wiring-repair-phase1-shared-contract-result-2026-07-10.md`
- `docs/plans/logs/bayesfilter-ledh-score-wiring-repair-phase1-pycompile-r1-2026-07-10.log`
- `docs/plans/logs/bayesfilter-ledh-score-wiring-repair-phase1-shared-tests-r1-2026-07-10.log`

Local checks:

- `python -m py_compile bayesfilter/highdim/ledh_score_contract.py bayesfilter/highdim/ledh_score_artifact.py`: passed.
- `pytest -q tests/highdim/test_ledh_score_contract_phase1.py tests/highdim/test_ledh_score_artifact_emitter_phase1.py`: `57 passed, 2 warnings`.

Gate status:

- `PASSED_AFTER_REPAIR`

Next action:

- Continue to Phase 2 after recording the Phase 2 skeptical audit.

### 2026-07-10T05:20:42+08:00 - Phase 2 - ASSESS_GATE

Evidence contract:

- Question: Is the LGSSM default score path unambiguously compact, with
  historical reverse route demoted and score timing/precision metadata ready
  for downstream admission?
- Baseline/comparator: Phase 1 repaired shared contract and prior LGSSM compact
  `N=10000,T=50` score-only artifact.
- Primary criterion: tests prove compact dispatch, historical route demotion,
  score timing fields, and production precision in admitted artifacts.
- Veto diagnostics: default path calls full-history reverse; historical route
  full-admits; missing score precision in admitted artifact.
- Non-claims: no new GPU score run, no exact Kalman score claim, no all-model
  readiness.

Actions:

- Replaced the stale LGSSM raw full-row score status trigger with
  `admitted_same_target_compact_score`.
- Demoted old `admitted_same_target_memory_style_score` to legacy/wrong status
  that cannot full-admit.
- Added score precision propagation for monolithic and seed-sharded LGSSM
  score artifacts.
- Added `score_call_seconds` and `score_materialize_seconds` fields.
- Drafted Phase 3 fixed-SIR subplan.

Artifacts:

- `docs/benchmarks/benchmark_ledh_same_target_lgssm_m3_t50_value.py`
- `tests/highdim/test_ledh_lgssm_score_phase2_contract.py`
- `tests/test_ledh_lgssm_manual_score_phase4.py`
- `docs/plans/bayesfilter-ledh-score-wiring-repair-phase2-lgssm-result-2026-07-10.md`
- `docs/plans/bayesfilter-ledh-score-wiring-repair-phase3-fixed-sir-subplan-2026-07-10.md`
- `docs/plans/logs/bayesfilter-ledh-score-wiring-repair-phase2-lgssm-pycompile-2026-07-10.log`
- `docs/plans/logs/bayesfilter-ledh-score-wiring-repair-phase2-lgssm-tests-2026-07-10.log`
- `docs/plans/logs/bayesfilter-ledh-score-wiring-repair-phase2-lgssm-route-precision-rg-2026-07-10.log`

Local checks:

- `python -m py_compile docs/benchmarks/benchmark_ledh_same_target_lgssm_m3_t50_value.py`: passed.
- `pytest -q tests/test_ledh_lgssm_manual_score_phase4.py tests/highdim/test_ledh_lgssm_score_phase2_contract.py tests/highdim/test_ledh_score_contract_phase1.py`: `77 passed, 2 warnings`.

Gate status:

- `REVISE_REPAIRED_REVIEW_PENDING`

Next action:

- Run focused substitute re-review for the Phase 2 relabeling repair and Phase
  3 adversarial mismatch requirement.

### 2026-07-10 - Phase 2 - REVIEW_REPAIR

Actions:

- Substitute review returned `VERDICT: REVISE`.
- Repaired LGSSM artifact adapter so nested `manual_score_diagnostic` must
  disclose compact provenance, compact score route, no full-history reverse
  route, compact execution style, and matching nested score values.
- Added adversarial LGSSM tests for historical nested provenance relabeling and
  outer/nested score mismatch.
- Updated Phase 3 fixed-SIR subplan to require an adversarial mismatch fixture
  for the same relabeling class.
- Clarified that `score_materialize_seconds` is not a clean post-call tensor
  materialization split because the score diagnostic materializes internally.

Artifacts:

- `docs/benchmarks/benchmark_ledh_same_target_lgssm_m3_t50_value.py`
- `tests/highdim/test_ledh_lgssm_score_phase2_contract.py`
- `docs/plans/bayesfilter-ledh-score-wiring-repair-phase2-lgssm-result-2026-07-10.md`
- `docs/plans/bayesfilter-ledh-score-wiring-repair-phase3-fixed-sir-subplan-2026-07-10.md`
- `docs/plans/logs/bayesfilter-ledh-score-wiring-repair-phase2-lgssm-pycompile-r1-2026-07-10.log`
- `docs/plans/logs/bayesfilter-ledh-score-wiring-repair-phase2-lgssm-tests-r1-2026-07-10.log`
- `docs/plans/logs/bayesfilter-ledh-score-wiring-repair-phase2-lgssm-review-repair-rg-2026-07-10.log`

Local checks:

- `python -m py_compile docs/benchmarks/benchmark_ledh_same_target_lgssm_m3_t50_value.py`: passed.
- `pytest -q tests/test_ledh_lgssm_manual_score_phase4.py tests/highdim/test_ledh_lgssm_score_phase2_contract.py tests/highdim/test_ledh_score_contract_phase1.py`: `83 passed, 2 warnings`.

Gate status:

- `PASSED_AFTER_REPAIR`

Next action:

- Begin Phase 3 fixed-SIR compact default repair.

### 2026-07-11T00:02:15+08:00 - Phase 9 Gate B - FIXED_SIR_XLA_REPAIR_PENDING_REVIEW

Evidence classification:

- Trusted GPU/XLA preflight passed on GPU 0 before nonlinear Gate B.
- Attempt-1 fixed-SIR score-only completed with finite GPU output and a reset
  peak of `80.04736328125 MiB` at `T=1,N=4`, seed `81120`.
- Attempt-1 FD-only emitted a terminal failed artifact before computing any FD
  value because a graph-time callback constructor called `.numpy()` on a
  `SymbolicTensor`.
- This is a harness/XLA extraction failure, not an FD numerical mismatch and
  not evidence against the compact recurrence or another row.

Repair:

- Archived the attempt-1 score/FD artifacts and logs under distinct
  `attempt-1-fixed-sir-pre-repair` directories with hashes recorded in the
  repair result.
- Replaced graph-time `_dpf_sir_callbacks()` covariance reconstruction with
  Cholesky of the already-prepared fixed `transition_covariance[0]` tensor.
- Added a source guard against `_dpf_sir_callbacks`/`.numpy` and an actual
  CPU-hidden XLA/eager value-parity test.

Local checks:

- New focused repair tests: `2 passed, 76 deselected, 2 warnings in 13.80s`.
- Harness plus fixed-SIR: `97 passed, 2 warnings in 92.99s`.
- Combined harness/cross-model/shared contract: `151 passed, 2 warnings in
  27.58s`.
- LGSSM/fixed-SIR shard: `53 passed, 2 warnings in 92.11s`.
- Syntax, exact-command currentness, and `git diff --check`: passed.

Artifact:

- `docs/plans/bayesfilter-ledh-score-wiring-repair-phase9-gate-b-fixed-sir-repair-result-2026-07-11.md`

Gate status:

- `REPAIR_VALIDATED_REVIEW_PENDING_NO_GPU_RETRY_AUTHORIZED`

Next action:

- Perform a fresh bounded local substitute review. If and only if it returns
  `VERDICT: AGREE`, rerun both exact fixed-SIR Gate B commands. All other Gate B
  rows, Gate C/D, aggregation, and LGSSM remain blocked.

### 2026-07-11T00:18:00+08:00 - Phase 9 Gate B - FIXED_SIR_REPAIR_REVIEW

Review iteration 1:

- `VERDICT: REVISE` because retry shards would have named and hashed only the
  older Gate A authorization, not a fresh repair review.

Provenance repair:

- Added a dedicated repair-review path to the governance hash set, run
  manifest, and common shard validator.
- Added adversarial rejection tests for both a changed repair-review path and a
  changed repair-review hash.
- Kept the frozen exact command argv and all numerical settings unchanged.

Review iteration 2:

- `VERDICT: AGREE` after the final review-bound combined suite passed `152`
  tests and the final path/hash-focused suite passed `34` tests.

Gate status:

- `FIXED_SIR_SCORE_THEN_FD_RETRY_AUTHORIZED`

Next action:

- Run the exact trusted fixed-SIR score-only command. Run its exact FD-only
  command only if the new score shard passes hard provenance, device, finite,
  and memory checks. Later gates remain blocked.

### 2026-07-11T00:47:00+08:00 - Phase 9 Gate B - CROSS_ROW_EXTRACTION_REPAIR_REVIEW

Finding:

- Fixed-SIR passed its intermediate trusted retry, but predator-prey score
  tracing then exposed the same eager-model-construction defect class.
- Predator-prey failed before computing a score; its FD command did not run.
- Proactive CPU-hidden XLA probes found no analogous defect in actual-SV,
  generalized-SV, or KSC-SV.

Repair and checks:

- Froze predator-prey's established `delta=2.0`, `20`-substep RK4 schedule at
  eager module initialization for score, value-only, and historical transition
  helpers.
- Added actual XLA score/value compilation and eager parity for all five
  nonlinear rows.
- Predator-prey contract: `22 passed`; combined all-row suite: `158 passed`;
  final review-bound subset: `40 passed`.
- Archived both failed attempts and the successful intermediate fixed-SIR
  retry. Final live shards must share the new source/review identity.

Review verdict:

- `VERDICT: AGREE`

Gate status:

- `ALL_TEN_NONLINEAR_GATE_B_COMMANDS_AUTHORIZED_SEQUENTIALLY`

Next action:

- Rerun all five score/FD pairs in frozen order under the cross-row review hash.
  Gate C/D, aggregation, and LGSSM remain blocked.

### 2026-07-11T01:03:44+08:00 - Phase 9 Gate B - FINAL_RESULT

Final common-identity row decisions:

- fixed-SIR: passed by relative FD tolerance; tiny peak `80.0473633 MiB`.
- predator-prey: failed both FD branches (`max_abs=0.3162194`,
  `max_relative=1.0`); blocked from the current ladder.
- actual-SV: passed by absolute FD tolerance; transformed target preserved.
- generalized-SV: passed by absolute FD tolerance; raw source-route target
  preserved.
- KSC-SV: passed both FD branches; KSC surrogate target preserved.

Evidence discipline:

- All five score shards and four passing FD shards pass the runner's own raw
  validators. Predator-prey is a terminal `failed_fd` artifact as intended.
- No candidate ranking is statistically supported. Tiny peaks and runtimes are
  descriptive only.
- The predator-prey candidate failure does not reject the compact-score
  research direction or another row.

Artifact:

- `docs/plans/bayesfilter-ledh-score-wiring-repair-phase9-gate-b-result-2026-07-11.md`

Gate status:

- `GATE_B_COMPLETE_REVIEW_PENDING_GATE_C_BLOCKED`

Next action:

- Perform a fresh bounded local substitute review of the Gate B result. Do not
  start Gate C or the separate LGSSM lane before that verdict.

### 2026-07-11T01:12:00+08:00 - Phase 9 Gate B - RESULT_REVIEW

Review iteration 1:

- `VERDICT: REVISE`: future Gate C shards would not bind the Gate B result or
  its row-eligibility review.

Provenance repair:

- Added Gate B result and result-review paths to governance hashes, manifests,
  and common shard validation.
- Added independent adversarial tests for both paths and both hashes.
- Exact Gate C argv and all numerical settings remain unchanged.

Review iteration 2:

- `VERDICT: AGREE` after `161` full-suite tests and `35` final hash-sensitive
  tests passed.

Gate status:

- `GATE_C_AUTHORIZED_FIXED_SIR_ACTUAL_SV_GENERALIZED_SV_KSC_SV_ONLY`

Next action:

- Execute each eligible row's Gate C prefixes in frozen ascending order,
  score-before-FD. Predator-prey, Gate D, aggregation, and LGSSM remain blocked.

### 2026-07-10T05:41:59+08:00 - Phase 3 - ASSESS_GATE

Evidence contract:

- Question: Is fixed-SIR wired so compact forward-sensitivity is the only
  full-admissible score path, with old memory/manual result normalization
  diagnostic-only?
- Baseline/comparator: existing fixed-SIR compact helper/tests and the
  historical fixed-SIR score-memory artifact.
- Primary criterion: tests prove compact score no-autodiff execution,
  same-scalar tiny FD, production precision for full admission, and historical
  memory/manual demotion.
- Veto diagnostics: historical route full-admits; compact full artifact lacks
  `score_precision`; nested historical/manual route can be relabeled compact.
- Non-claims: no new N=10000 GPU fixed-SIR score run, no exact nonlinear
  likelihood claim, no leaderboard completion.

Actions:

- Added compact score precision propagation from diagnostic precision metadata.
- Required compact diagnostic base to declare compact route,
  `no_autodiff_score_route`, and same-route value/score status.
- Blocked full admission from `_fixed_sir_score_artifact_from_memory_result`.
- Preserved historical memory/manual provenance on legacy normalized artifacts.
- Added tests for production precision, TF32-disabled rejection, compact full
  fixture admission, and nested historical/manual relabeling rejection.
- Drafted Phase 4 predator-prey subplan.

Artifacts:

- `docs/benchmarks/benchmark_ledh_same_target_fixed_sir_score.py`
- `tests/highdim/test_ledh_fixed_sir_score_phase3_contract.py`
- `docs/plans/bayesfilter-ledh-score-wiring-repair-phase3-fixed-sir-result-2026-07-10.md`
- `docs/plans/bayesfilter-ledh-score-wiring-repair-phase4-predator-prey-subplan-2026-07-10.md`
- `docs/plans/logs/bayesfilter-ledh-score-wiring-repair-phase3-fixed-sir-pycompile-r1-2026-07-10.log`
- `docs/plans/logs/bayesfilter-ledh-score-wiring-repair-phase3-fixed-sir-tests-r1-2026-07-10.log`
- `docs/plans/logs/bayesfilter-ledh-score-wiring-repair-phase3-fixed-sir-route-precision-rg-2026-07-10.log`

Local checks:

- `python -m py_compile docs/benchmarks/benchmark_ledh_same_target_fixed_sir_score.py`: passed.
- `pytest -q tests/highdim/test_ledh_fixed_sir_score_phase3_contract.py tests/highdim/test_ledh_score_contract_phase1.py`: `67 passed, 2 warnings`.

Gate status:

- `PASSED`

Next action:

- Begin Phase 4 predator-prey compact default repair.

### 2026-07-10T19:44:36+08:00 - Phase 5 - REVIEW

Actions:

- Confirmed the earlier bounded Claude review attempt was rejected by execution
  policy as external repository data disclosure; no workaround was attempted.
- Fresh Codex substitute read-only review inspected the Phase 5 result, Phase 6
  subplan, actual-SV adapter and tests, and shared score contract.

Review verdict:

- `VERDICT: AGREE`

Review summary:

- Phase 5 physically wires the default diagnostic score to the compact route
  and uses a value-only same-scalar finite-difference comparator.
- Full admission rejects nested historical/manual provenance and delegates
  production precision and trusted memory validation to the shared contract.
- The transformed actual-SV target and exact-native-likelihood nonclaim remain
  intact.
- Phase 6 correctly scopes generalized-SV as compact-route precision and
  full-admission hardening.

Artifacts:

- `docs/reviews/bayesfilter-ledh-score-wiring-repair-phase5-result-phase6-subplan-codex-substitute-review-2026-07-10.md`

Gate status:

- `PASSED`

Next action:

- Begin Phase 6 generalized-SV compact precision and admission-boundary repair.

### 2026-07-10T19:57:36+08:00 - Phase 6 - ASSESS_GATE

Evidence contract:

- Question: does the existing generalized-SV compact score route satisfy the
  shared production precision and full-admission boundaries?
- Baseline: compact computation already existed, but defaults were
  `float64`/TF32-disabled and artifact construction lacked precision and
  nested full-row guards.
- Primary criterion: direct compact score base, value-only same-scalar FD,
  no-autodiff tiny execution, target preservation, production precision, and
  rejection of forged route/shape/seed metadata.
- Vetoes: KSC/actual-SV target substitution, non-production precision,
  non-compact nested provenance, or wrong full-row particles/time/seeds.
- Nonclaims: no trusted full generalized-SV GPU score-memory run, no full score
  admission, leaderboard, HMC, posterior, runtime-ranking, or scientific claim.

Actions:

- Set generalized-SV score defaults to `float32` with TF32 enabled.
- Made the coordinate-FD diagnostic call the compact score helper directly and
  use a value-only objective for perturbations.
- Added score precision and compact base metadata.
- Hardened full artifact construction against nested route relabeling and
  mismatched particles, time steps, or seeds.
- Preserved `source_route_prior_mean_generalized_sv` target semantics.
- Repaired a committed import-chain syntax defect by removing one duplicated
  `maximum_iterations` keyword from `bayesfilter/linear/kalman_qr_tf.py`.
- Drafted the Phase 7 KSC-SV subplan and review bundle.

Checks:

- CPU-hidden `py_compile`: passed.
- Initial pytest collection: blocked before Phase 6 tests by the committed
  duplicate-keyword syntax error.
- Targeted fixture repair check: `5 passed, 14 deselected, 2 warnings`.
- Final focused pytest: `68 passed, 2 warnings in 36.69s`.

Artifacts:

- `docs/plans/bayesfilter-ledh-score-wiring-repair-phase6-generalized-sv-result-2026-07-10.md`
- `docs/plans/bayesfilter-ledh-score-wiring-repair-phase7-ksc-sv-subplan-2026-07-10.md`
- `docs/reviews/bayesfilter-ledh-score-wiring-repair-phase6-result-phase7-subplan-review-bundle-2026-07-10.md`

Gate status:

- `LOCAL_CHECKS_PASSED_REVIEW_PENDING`

Next action:

- Review the Phase 6 result and Phase 7 subplan; begin Phase 7 only if review
  agrees.

### 2026-07-10T20:01:05+08:00 - Phase 6 - REVIEW

Actions:

- Fresh Codex substitute read-only review inspected the Phase 6 result, Phase
  7 subplan, generalized-SV implementation/tests, shared score contract, and
  prerequisite QR syntax repair.
- No Claude retry or workaround was attempted because Claude calls remain
  policy-blocked as external repository data disclosure.
- Ran the focused CPU-hidden QR module suite after the one-line prerequisite
  repair.

Review verdict:

- `VERDICT: AGREE`

Review summary:

- The default generalized-SV diagnostic physically calls the compact score
  route and uses a value-only finite-difference objective.
- Production precision, nested provenance, row shape, seeds, and target
  semantics are enforced for full admission.
- The Phase 6 result keeps local checks at wiring-evidence scope.
- The prerequisite syntax repair is correctly separated from generalized-SV
  numerical evidence; its focused suite passed `8 passed, 2 warnings`.
- Phase 7 preserves the KSC surrogate target and exact-native actual-SV
  nonclaim with complete gates and stop conditions.

Artifacts:

- `docs/reviews/bayesfilter-ledh-score-wiring-repair-phase6-result-phase7-subplan-codex-substitute-review-2026-07-10.md`

Gate status:

- `PASSED`

Next action:

- Begin Phase 7 KSC-SV compact precision and admission-boundary repair.

### 2026-07-10T20:07:11+08:00 - Phase 6/7 - REVIEW_REPAIR

Finding:

- A cross-model audit found that directly calling the component helper with all
  five full-row seeds would change the prior sequential seed execution schedule
  and could multiply peak memory. The scalar and compact recurrence were not
  invalidated, but the batching change was outside the reviewed requirement.

Repair:

- Added `_compact_value_and_score_across_seeds` to generalized-SV and KSC-SV.
- The wrapper invokes the compact component sequentially per seed and returns
  only the aggregated scalar/score metadata.
- Kept `_manual_value_and_score_across_seeds` as a compatibility alias, but the
  default diagnostic does not call it.
- Strengthened monkeypatch tests to use two seeds and require one compact
  component call per seed plus value-only FD calls.
- Marked the first Phase 6 substitute verdict as superseded pending re-review.

Gate status:

- `REPAIR_TESTS_AND_REVIEW_PENDING`

Next action:

- Rerun Phase 6 and Phase 7 focused checks, then perform a fresh substitute
  re-review before advancing.

### 2026-07-10T20:11:57+08:00 - Phase 7 - ASSESS_GATE_AFTER_REPAIR

Actions:

- Reran Phase 6 after the sequential-seed repair: `68 passed, 2 warnings in
  39.86s`.
- Reran Phase 7 after the same invariant repair: `68 passed, 2 warnings in
  23.16s`.
- Wrote the Phase 7 result, Phase 8 cross-model subplan, and fresh review
  bundle.

Evidence classification:

- Hard wiring/admission vetoes: none fired.
- Viable routes: compact generalized-SV and compact KSC-SV sequential-seed
  routes remain viable for later full-row GPU checks.
- Ranking: none supported or attempted.
- Descriptive-only evidence: CPU-hidden tiny equality/FD diagnostics.
- Next evidence: cross-model regression gate, then trusted GPU memory ladder.

Gate status:

- `LOCAL_CHECKS_PASSED_FRESH_REVIEW_PENDING`

Next action:

- Fresh substitute review of the batching repair, Phase 7 result, and Phase 8
  subplan.

### 2026-07-10T20:11:57+08:00 - Phase 6/7 - FOCUSED_REVIEW

Actions:

- Fresh substitute review inspected the sequential-seed batching repair,
  Phase 6 and Phase 7 post-repair tests, Phase 7 result, and the revised bounded
  Phase 8 subplan.

Review verdict:

- `VERDICT: AGREE`

Review summary:

- Both long SV adapters invoke compact components sequentially per seed and do
  not call the compatibility alias from the default diagnostic.
- Both preserve row-specific targets and full-admission guards.
- Phase 8 is CPU-hidden wiring regression only and is split into bounded test
  shards.

Artifact:

- `docs/reviews/bayesfilter-ledh-score-wiring-repair-phase6-repair-phase7-result-phase8-subplan-codex-substitute-review-2026-07-10.md`

Gate status:

- `PASSED_AFTER_BATCHING_REPAIR`

Next action:

- Begin Phase 8 cross-model score wiring regression gate.

### 2026-07-10T20:16:00+08:00 - Phase 8 - PRECHECK_REPAIR

Finding:

- Cross-model source audit found fixed-SIR FD perturbations called
  `_compact_value_and_score_from_components`, recomputing tangents instead of
  evaluating the value-only same scalar. This was wrong relative to the master
  program's binding value-only FD invariant.

Repair:

- Added `_value_objective_from_components` using the established
  `p8p._objective_from_components` streaming value core.
- Kept the diagnostic score base compact; switched only plus/minus FD calls to
  the value-only objective.
- Added a monkeypatch call-count test requiring one compact score call and two
  value-only calls per parameter.

Gate status:

- `FIXED_SIR_REPAIR_TEST_PENDING`

Next action:

- Rerun fixed-SIR/shared focused tests before continuing Phase 8.

### 2026-07-10T20:19:00+08:00 - Phase 8 - FIXED_SIR_REPAIR_RESULT

Checks:

- Fixed-SIR/shared CPU-hidden focused suite: `68 passed, 2 warnings in
  73.26s`.

Assessment:

- Score base remains compact forward sensitivity.
- FD perturbations now use the same value-only streaming scalar.
- The call-count test proves one score call and two value-only calls per
  coordinate.
- No full-row GPU, HMC, posterior, leaderboard, or scientific claim follows.

Gate status:

- `PASSED_FIXED_SIR_VALUE_ONLY_FD_REPAIR`

Next action:

- Resume Phase 8 cross-model invariant tests.

### 2026-07-10T20:51:40+08:00 - Phase 8 - SEQUENTIAL_SEED_REPAIR_RESULT

Skeptical audit finding:

- Fixed-SIR, predator-prey, and actual-SV still passed all fixed seeds into one
  compact component call. This could multiply peak GPU memory and conflicted
  with the repaired generalized-SV/KSC-SV sequential-seed baseline.
- The issue was an execution-policy repair trigger, not evidence against the
  target scalar or compact sensitivity recurrence.

Repair:

- Added explicit compact sequential-seed wrappers to all three rows.
- Added sequential value-only FD aggregation to fixed-SIR.
- Routed actual-SV `value-score-only` mode through the sequential wrapper.
- Added two-seed singleton invocation tests, exact order assertions, and
  historical-route sentinels.
- Expanded the cross-model schedule gate to all five nonlinear rows.

Checks:

- Py-compile: passed.
- Focused schedule tests: `9 passed, 2 warnings in 5.81s`.
- Shared/cross-model: `73 passed, 2 warnings in 2.74s`.
- LGSSM/fixed-SIR: `39 passed, 2 warnings in 77.47s`.
- Predator-prey/actual-SV: `45 passed, 2 warnings in 175.38s`.
- Generalized-SV/KSC-SV: `38 passed, 2 warnings in 57.46s`.

Gate status:

- `PASSED_AFTER_ALL_NONLINEAR_SEQUENTIAL_SEED_REPAIR`

Phase 9 readiness finding:

- LGSSM has explicit XLA and one-seed trusted score-only memory evidence.
- Current nonlinear score CLIs lack explicit XLA JIT, managed-session trust,
  device validation, reset score-memory, terminal progress artifacts, and
  score-only/FD-only shard separation.
- These omissions veto nonlinear GPU evidence from the current CLIs. Phase 9
  must start with a harness compliance gate.

Artifacts:

- `docs/plans/bayesfilter-ledh-score-wiring-repair-phase8-cross-model-tests-result-2026-07-10.md`
- `docs/plans/bayesfilter-ledh-score-wiring-repair-phase9-gpu-score-memory-subplan-2026-07-10.md`
- `docs/reviews/bayesfilter-ledh-score-wiring-repair-phase8-result-phase9-subplan-review-bundle-2026-07-10.md`

Next action:

- Perform a fresh bounded local substitute review. Do not edit Phase 9 harness
  code or launch GPU commands unless the review returns `VERDICT: AGREE`.

### 2026-07-10T21:05:00+08:00 - Phase 8/9 - REVIEW_ITERATION_1

Review verdict:

- `VERDICT: REVISE`

Blocking findings:

- The Phase 9 draft could not supply exact GPU commands before its shared
  harness existed.
- Finite-difference promotion criteria were referred to indirectly rather than
  frozen per row, leaving ambiguity about steps, tolerances, and per-seed
  versus aggregate pass requirements.

Revision:

- Limited current authorization to Gate A harness implementation with named
  files and CPU-hidden checks.
- Kept all GPU commands separately blocked pending a post-harness exact command
  manifest and fresh review.
- Froze per-row FD steps/tolerances and required every seed plus the aggregate
  to pass the existing absolute-or-relative rule.

Artifact:

- `docs/reviews/bayesfilter-ledh-score-wiring-repair-phase8-result-phase9-subplan-codex-substitute-review-iter1-2026-07-10.md`

Next action:

- Fresh bounded re-review of the revised Phase 9 Gate A scope. Implement Gate A
  only if that review returns `VERDICT: AGREE`; do not launch GPU commands.

### 2026-07-10T21:12:00+08:00 - Phase 9 - GATE_A_REVIEW_ITERATION_2

Review verdict:

- `VERDICT: AGREE`

Review scope:

- Gate A shared evidence-harness implementation only.
- No GPU/CUDA/XLA runtime command authorized.

Why the gate passes:

- Named files and CPU-hidden checks make the implementation scope executable.
- Existing component helpers can be made tensor-input compatible without
  changing the compact equations or row targets.
- FD criteria are frozen before results.
- GPU execution remains separately gated by a post-harness result, exact
  command manifest, and fresh review.

Artifact:

- `docs/reviews/bayesfilter-ledh-score-wiring-repair-phase8-result-phase9-subplan-codex-substitute-review-iter2-2026-07-10.md`

Next action:

- Implement Gate A only, run CPU-hidden checks, and write the Gate A result plus
  exact GPU execution manifest. Do not launch GPU commands.

### 2026-07-10T06:03:45+08:00 - Phase 4 - ASSESS_GATE

Evidence contract:

- Question: Is predator-prey wired so compact forward-sensitivity is the only
  full-admissible score path, with reverse/manual route demoted and precision
  enforced?
- Baseline/comparator: previous predator-prey tests asserted reverse/manual
  default; the compact helper was already present from the earlier tiny compact
  port.
- Primary criterion: tests prove compact score no-autodiff execution,
  same-scalar tiny FD, production precision for full admission, and rejection
  of nested historical/manual relabeling.
- Veto diagnostics: historical route full-admits; full artifact lacks
  `score_precision`; CLI/default production score remains `float64` or TF32
  disabled; nested manual route can be relabeled compact.
- Non-claims: no trusted `N=10000,T=20` GPU score-memory run, no full
  predator-prey score admission, no leaderboard/HMC/posterior/scientific claim.

Actions:

- Changed predator-prey score defaults to `float32` with TF32 enabled.
- Changed `_coordinate_fd_score_diagnostic` to use the compact score route as
  its score base and a value-only same-scalar objective for FD.
- Added `score_precision` metadata to score artifacts.
- Required nested compact base metadata before full admission.
- Preserved reverse/manual route as historical diagnostic-only.
- Adjusted low-level historical reverse/manual VJP FD tests to use a
  float32-stable finite-difference step.
- Drafted Phase 5 actual-SV subplan.

Artifacts:

- `docs/benchmarks/benchmark_ledh_same_target_predator_prey_score.py`
- `tests/highdim/test_ledh_predator_prey_score_phase4_contract.py`
- `docs/plans/bayesfilter-ledh-score-wiring-repair-phase4-predator-prey-result-2026-07-10.md`
- `docs/plans/bayesfilter-ledh-score-wiring-repair-phase5-actual-sv-subplan-2026-07-10.md`

Local checks:

- `python -m py_compile docs/benchmarks/benchmark_ledh_same_target_predator_prey_score.py tests/highdim/test_ledh_predator_prey_score_phase4_contract.py`: passed.
- `pytest -q tests/highdim/test_ledh_predator_prey_score_phase4_contract.py tests/highdim/test_ledh_score_contract_phase1.py`: `70 passed, 2 warnings`.

Gate status:

- `LOCAL_CHECKS_PASSED_REVIEW_PENDING`

Next action:

- Review Phase 4 result and Phase 5 subplan, then begin Phase 5 if review
  agrees.

### 2026-07-10T06:08:01+08:00 - Phase 4 - REVIEW

Actions:

- Claude review gate was attempted with
  `bash ~/python/claudecodex/scripts/claude_review_gate.sh ...`.
- Execution policy rejected the Claude call as external repository data
  disclosure. No workaround was attempted.
- Fresh Codex substitute read-only review inspected the Phase 4 result, Phase 5
  subplan, predator-prey adapter/tests, and shared score contract.

Review verdict:

- `VERDICT: AGREE`

Review summary:

- No issues found.
- Phase 4 closes predator-prey compact score wiring without relabeling
  historical reverse/manual routes as compact.
- Phase 4 result stays within CPU-hidden wiring evidence and avoids full
  GPU/admission, HMC, posterior, leaderboard, and scientific claims.
- Phase 5 actual-SV subplan carries the right dependency and boundaries.

Artifacts:

- `docs/reviews/bayesfilter-ledh-score-wiring-repair-phase4-result-phase5-subplan-review-bundle-2026-07-10.md`

Gate status:

- `PASSED`

Next action:

- Begin Phase 5 actual-SV compact default repair.

### 2026-07-10T06:16:42+08:00 - Phase 5 - ASSESS_GATE

Evidence contract:

- Question: Is actual-SV wired so compact forward-sensitivity is the only
  full-admissible score path for the transformed actual-SV same scalar?
- Baseline/comparator: previous actual-SV tests and code routed coordinate FD
  through memory-style reverse/manual score.
- Primary criterion: tests prove compact score no-autodiff execution,
  same-scalar tiny FD, transformed target preservation, production precision
  for full admission, and rejection of nested historical/manual relabeling.
- Veto diagnostics: historical route full-admits; target shifts to KSC/native
  exact likelihood; full artifact lacks `score_precision`; CLI/default
  production score remains `float64` or TF32 disabled.
- Non-claims: no trusted `N=10000,T=1000` GPU score-memory run, no full
  actual-SV score admission, no exact native likelihood, no
  leaderboard/HMC/posterior/scientific claim.

Actions:

- Changed actual-SV score defaults to `float32` with TF32 enabled.
- Changed `_coordinate_fd_score_diagnostic` to use the compact score route as
  its score base and a value-only same-scalar objective for FD.
- Added `score_precision` metadata to score artifacts.
- Required nested compact base metadata before full admission.
- Preserved reverse/manual route as historical diagnostic-only.
- Preserved transformed actual-SV target policy and exact-native likelihood
  nonclaim.
- Drafted Phase 6 generalized-SV subplan.

Artifacts:

- `docs/benchmarks/benchmark_ledh_same_target_actual_sv_score.py`
- `tests/highdim/test_ledh_actual_sv_score_phase5_contract.py`
- `docs/plans/bayesfilter-ledh-score-wiring-repair-phase5-actual-sv-result-2026-07-10.md`
- `docs/plans/bayesfilter-ledh-score-wiring-repair-phase6-generalized-sv-subplan-2026-07-10.md`

Local checks:

- `python -m py_compile docs/benchmarks/benchmark_ledh_same_target_actual_sv_score.py tests/highdim/test_ledh_actual_sv_score_phase5_contract.py`: passed.
- `pytest -q tests/highdim/test_ledh_actual_sv_score_phase5_contract.py tests/highdim/test_ledh_score_contract_phase1.py`: `70 passed, 2 warnings`.

Gate status:

- `LOCAL_CHECKS_PASSED_REVIEW_PENDING`

Next action:

- Review Phase 5 result and Phase 6 subplan, then begin Phase 6 if review
  agrees.

### 2026-07-10T05:46:49+08:00 - Phase 3 - REVIEW

Actions:

- Substitute review inspected Phase 3 result, Phase 4 subplan, fixed-SIR code,
  fixed-SIR tests, and shared score contract.

Review verdict:

- `VERDICT: AGREE`

Gate status:

- `PASSED`

Next action:

- Begin Phase 4 predator-prey compact default repair.

### 2026-07-10T05:30:51+08:00 - Phase 2 - FOCUSED_REVIEW

Actions:

- Focused substitute re-review checked the LGSSM nested-provenance repair,
  adversarial tests, Phase 3 mismatch-fixture requirement, and timing-field
  clarification.

Review verdict:

- `VERDICT: AGREE`

Gate status:

- `PASSED_AFTER_REPAIR`

Next action:

- Begin Phase 3 fixed-SIR compact default repair.

### 2026-07-11T01:29:30+08:00 - Phase 9 Gate C - FIXED_SIR_TERMINAL

Evidence:

- Fixed-SIR passed trusted score execution and memory at
  `T=1,5,20`, seed `81120`, `N=10000`.
- Reset score peaks were `185.3544922`, `348.3242188`, and
  `414.4467773 MiB`, all below `14000 MiB`.
- FD passed `T=1` and `T=5` by the frozen relative branch.
- Full `T=20` FD failed both branches:
  `max_abs=7.853515625 > 0.01` and
  `max_rel=0.0566700101 > 0.05`.

Artifacts:

- `docs/plans/bayesfilter-ledh-score-wiring-repair-phase9-gate-c-fixed-sir-result-2026-07-11.md`
- `docs/reviews/bayesfilter-ledh-score-wiring-repair-phase9-gate-c-fixed-sir-result-codex-substitute-review-2026-07-11.md`

Review verdict: `VERDICT: AGREE`.

Gate status: `FIXED_SIR_GATE_D_BLOCKED`.

### 2026-07-11T01:36:14+08:00 - Phase 9 Gate C - ACTUAL_SV_TERMINAL

Evidence:

- Actual-SV `T=4,N=10000`, seed `81120` score completed on trusted GPU/XLA
  with a `35.2270508 MiB` reset peak.
- FD failed both frozen branches:
  `max_abs=0.0094842315 > 0.005` and
  `max_rel=0.0602924675 > 0.005`.
- The transformed `log(y^2)` target was preserved; this is not native
  actual-SV likelihood evidence.

Artifacts:

- `docs/plans/bayesfilter-ledh-score-wiring-repair-phase9-gate-c-actual-sv-result-2026-07-11.md`
- `docs/reviews/bayesfilter-ledh-score-wiring-repair-phase9-gate-c-actual-sv-result-codex-substitute-review-2026-07-11.md`

Review verdict: `VERDICT: AGREE`.

Gate status: `ACTUAL_SV_T50_AND_GATE_D_BLOCKED`.

### 2026-07-11T01:46:57+08:00 - Phase 9 Gate C - GENERALIZED_SV_TERMINAL

Evidence:

- Generalized-SV `T=4,N=10000`, seed `81120` score completed on trusted
  GPU/XLA with a `35.2309570 MiB` reset peak.
- FD failed both frozen branches:
  `max_abs=0.0151546374 > 0.005` and
  `max_rel=0.4427539706 > 0.005`.
- The raw source-route prior-mean generalized-SV target was preserved.

Artifacts:

- `docs/plans/bayesfilter-ledh-score-wiring-repair-phase9-gate-c-generalized-sv-result-2026-07-11.md`
- `docs/reviews/bayesfilter-ledh-score-wiring-repair-phase9-gate-c-generalized-sv-result-codex-substitute-review-2026-07-11.md`

Review verdict: `VERDICT: AGREE`.

Gate status: `GENERALIZED_SV_T50_AND_GATE_D_BLOCKED`.

### 2026-07-11T01:53:32+08:00 - Phase 9 Gate C - KSC_SV_TERMINAL

Evidence:

- KSC-SV `T=4,N=10000`, seed `81120` score completed on trusted GPU/XLA with
  a `35.2260742 MiB` reset peak.
- FD failed both frozen branches:
  `max_abs=0.0102410018 > 0.005` and
  `max_rel=0.0369351506 > 0.005`.
- The KSC log-chi-square Gaussian-mixture surrogate target was preserved and
  was not relabeled native actual-SV.

Artifacts:

- `docs/plans/bayesfilter-ledh-score-wiring-repair-phase9-gate-c-ksc-sv-result-2026-07-11.md`
- `docs/reviews/bayesfilter-ledh-score-wiring-repair-phase9-gate-c-ksc-sv-result-codex-substitute-review-2026-07-11.md`

Review verdict: `VERDICT: AGREE`.

Gate status: `KSC_SV_T50_AND_GATE_D_BLOCKED`.

### 2026-07-11T02:12:32+08:00 - Phase 9 - CONSOLIDATED_STOP_REVIEW

Decision:

- All five nonlinear candidates have terminal row-local FD vetoes.
- No shared harness veto fired.
- No nonlinear row is eligible for Gate D or aggregation.
- No Gate D artifact exists, as required by the reviewed stop rules.
- The separate LGSSM lane was not run and remains non-admitted.
- Phase 10 has no scoped subplan and is not authorized by this result.

Artifacts:

- `docs/plans/bayesfilter-ledh-score-wiring-repair-phase9-gpu-score-memory-result-2026-07-10.md`, SHA-256
  `a99ec35f4196cfc57b4b0c82e63f509540fe6fd27aad5808cff3a403a776103f`
- `docs/reviews/bayesfilter-ledh-score-wiring-repair-phase9-gpu-score-memory-result-codex-substitute-review-2026-07-11.md`,
  SHA-256
  `00b917a73ae20cec30e398d63d4d41ee398470c7e357da689fd7916d84cef310`

Review verdict: `VERDICT: AGREE`.

Gate status: `STOPPED_AT_PHASE9_NO_NONLINEAR_SCORE_ADMISSION`.

Next action:

- Write and review a predeclared discrepancy-diagnostic subplan or an explicit
  closeout/leaderboard subplan. No additional frozen Phase 9 GPU command is
  authorized.

### 2026-07-11 - Phase 9 - FD_POLICY_CORRECTION

Correction trigger:

- Owner rejected the inherited `0.005` FD setting as arbitrary.
- The first correction was wrong relative to the owner's stated target: it
  used `2%`, combined directions with RSS/RMS, and described an FD-only rule as
  an HMC-oriented screen.
- Owner clarified the intended base constant is `5%`, selected to mirror the
  conventional 95% threshold, and that individual parameter directions control
  the FD decision.

Binding FD-only policy:

- `r_j = abs(score_j - FD_j) / max(abs(score_j), abs(FD_j), 1e-12)`;
- `threshold = 0.05 * sqrt(p)`;
- pass iff `max_j(r_j) <= threshold`;
- no RSS/RMS aggregation and no absolute-error escape branch;
- this is only an FD diagnostic, not a general score, HMC, posterior, or
  scientific-validity screen;
- the 95% analogy fixes the intended constant but does not turn this arithmetic
  into a calibrated confidence interval or coverage statement.

Actions:

- Amended and skeptically audited the correction subplan before repairing code.
- Replaced HMC/RSS policy names and fields with an FD-only maximum-coordinate
  implementation while preserving the historical `1e-12` denominator floor.
- Bumped future shared-runner shards from v2 to v3.
- Updated the framework-free reclassifier and all focused harness/policy tests.
- Reclassified all 11 completed live Gate B/Gate C comparisons with SHA-256
  source and FD-to-score binding checks.
- Preserved all original Phase 9 JSON shards byte-for-byte.
- Kept GPU execution blocked; no GPU command was run.

Corrected decisions:

- nine of 11 stored comparisons pass and two fail;
- predator-prey fails Gate B `T=1,N=2`;
- generalized-SV passes Gate B and fails Gate C `T=4,N=10000`;
- fixed-SIR passes its historical full-time Gate C `T=20,N=10000` FD check;
- Actual-SV (`p=2`) passes Gate C `T=4,N=10000` because
  `0.0602924688125 <= 0.0707106781187`;
- KSC-SV passes Gate C `T=4,N=10000`.

Verification:

- policy/reclassifier tests: `9 passed`;
- shared CPU-hidden harness: `89 passed, 2 warnings in 90.13s`;
- syntax and `git diff --check`: passed;
- all original Phase 9 source hashes matched before/after correction.

Artifacts:

- `docs/plans/bayesfilter-ledh-score-wiring-repair-phase9-fd-policy-correction-subplan-2026-07-11.md`;
- `docs/plans/ledh-score-wiring-repair-phase9-fd-reclassification-inputs-2026-07-11.json`;
- `docs/plans/artifacts/ledh-score-wiring-repair-phase9-fd-policy-correction/phase9-fd-policy-reclassification-2026-07-11.json`;
- `docs/plans/bayesfilter-ledh-score-wiring-repair-phase9-fd-policy-correction-result-2026-07-11.md`;
- `docs/reviews/bayesfilter-ledh-score-wiring-repair-phase9-fd-policy-correction-codex-review-2026-07-11.md`.

Final authority hashes:

- correction result:
  `12108b9dc32283c2a42ddcb72937a87853ba381cd97416a53d718e52a327bbaf`;
- reclassification JSON:
  `1ffa3fd9fdf74050d667b4205c8545e56657f0102b81fb28933894bd3644a4dd`;
- correction review:
  `5c3e983baea6d283fa9e8b1590ff5e518f4715f7958b9a881f60339ba24bfaee`,
  `VERDICT: AGREE`.

Gate status:
`PHASE9_FD_POLICY_CORRECTED_PARTIAL_CONTINUATION_REVIEW_REQUIRED`.

Next action:

- Do not replay the historical exact-command manifest; it targets v1 source
  paths and predates the v3 policy schema.
- Write and review a new continuation subplan/manifest before resuming
  fixed-SIR Gate D or Actual-SV/KSC-SV Gate C.
- Predator-prey and generalized-SV require a separate reviewed derivative-
  resolution diagnostic if their failed candidates are pursued.
- This correction authorizes no GPU execution, aggregation, HMC, or Phase 10.
