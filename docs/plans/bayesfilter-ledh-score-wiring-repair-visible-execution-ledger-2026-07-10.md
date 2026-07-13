# LEDH Score Wiring Repair Visible Execution Ledger

Date: 2026-07-10

## Status

`IN_PROGRESS`

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
