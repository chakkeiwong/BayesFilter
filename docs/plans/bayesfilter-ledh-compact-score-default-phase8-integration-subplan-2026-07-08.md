# Phase 8 Subplan: Compact Score Integration

Date: 2026-07-08

Status: `DRAFT_READY_FOR_REVIEW`

## Phase Objective

Integrate the compact score routes into the LEDH leaderboard score workflow so compact forward-sensitivity is the default style for all implemented model score rows and historical `manual_total_vjp*` routes remain diagnostic-only.

This phase must not claim full score admission for any row unless the row has explicit full-row memory evidence and passes the shared score artifact validator with `require_admitted=True`.

## Entry Conditions Inherited From Previous Phase

- Phase 1 shared contract admits only compact provenance for full score admission.
- Phases 2-7 have compact tiny score artifacts or tests for:
  - LGSSM;
  - actual-SV;
  - fixed-SIR;
  - predator-prey;
  - generalized-SV;
  - KSC-SV.
- Historical `manual_total_vjp*` route strings are allowed only for non-admitted diagnostics.
- Full `N=10000` score rows remain gated by memory evidence.

## Required Artifacts

- Integration implementation updates, expected candidates:
  - `docs/benchmarks/benchmark_two_lane_highdim_leaderboard.py`
  - any LEDH score runner or score-artifact aggregation script referenced by the current leaderboard workflow.
- Integration tests:
  - new or updated tests under `tests/` or `tests/highdim/` that verify compact score provenance is default and historical routes cannot be admitted.
- Phase 8 result or blocker result:
  `docs/plans/bayesfilter-ledh-compact-score-default-phase8-integration-result-2026-07-08.md`
- Optional leaderboard candidate artifact, only if a reviewed command exists and local checks pass:
  `docs/plans/bayesfilter-ledh-compact-score-default-phase8-integration-candidate-2026-07-08.json`
- Review bundle:
  `docs/reviews/bayesfilter-ledh-compact-score-default-phase8-integration-review-bundle-2026-07-08.md`

## Required Checks, Tests, And Reviews

Precheck commands:

```bash
rg -n "manual_total_vjp|compact_forward_sensitivity|score_derivative_provenance|score_admission_status|validate_ledh_score_artifact" \
  docs/benchmarks tests bayesfilter/highdim
```

Focused checks after implementation must include:

```bash
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/highdim/test_ledh_score_contract_phase1.py \
  tests/highdim/test_ledh_generalized_sv_score_phase6_contract.py \
  tests/highdim/test_ledh_ksc_sv_score_phase7_contract.py -q
```

Add or run any leaderboard integration tests discovered during precheck.

Review:

- Claude read-only review of integration diffs, result, and any candidate artifact when available.
- If Claude is unavailable or policy-blocked, write a Codex substitute review.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Does the leaderboard score workflow default to compact forward-sensitivity score routes and block historical `manual_total_vjp*` full admission? |
| Baseline/comparator | Shared score contract, compact per-model score runners, existing leaderboard assembly/tests, and tiny compact score artifacts. |
| Primary criterion | Integration code and tests select compact provenance for default score rows, never promote historical routes to full admission, preserve same target/value artifact identity, and pass focused tests. |
| Veto diagnostics | Historical route admitted; value and score target mismatch; score runner uses tape/autodiff/stopped partial; full row claimed without memory evidence; KSC exact native overclaim; generalized-SV/KSC target substitution; no integration test covers route selection. |
| Explanatory diagnostics | Number of rows wired, rows still blocked, tiny artifact coverage, runtime/memory notes, and leaderboard candidate diff. |
| Not concluded | Full leaderboard score completion, HMC readiness, posterior correctness, runtime ranking, scientific superiority, or public benchmark readiness. |
| Artifact | Phase 8 result/blocker, tests, optional candidate leaderboard artifact. |

## Required Implementation Steps

1. Inventory current leaderboard score wiring and artifact aggregation.
2. Identify all places where score route strings or score artifacts are read, written, or promoted.
3. Ensure compact route constants are the only full-admissible default routes.
4. Ensure any historical route remains explicitly diagnostic-only and fails full admission.
5. Add integration tests that cover:
   - compact provenance allowlist;
   - historical route rejection;
   - same value/score target identity;
   - KSC exact-native overclaim rejection;
   - generalized-SV and KSC target boundary preservation.
6. Run focused CPU-only checks.
7. If a leaderboard candidate can be generated without a full `N=10000` score run, generate it as a non-admitted integration candidate with explicit nonclaims.
8. Write Phase 8 result or blocker.
9. Review the result and stop or proceed according to the review verdict.

## Forbidden Claims And Actions

- Do not run or claim a full `N=10000` score row without explicit reviewed memory-evidence authorization.
- Do not promote tiny score artifacts to admitted leaderboard scores.
- Do not relabel historical `manual_total_vjp*` routes as compact.
- Do not use tape/autodiff or stopped partial derivatives in score routes.
- Do not change model target definitions or parameter coordinate systems.
- Do not claim HMC readiness, posterior correctness, runtime ranking, or scientific superiority.

## Exact Next-Phase Handoff Conditions

After Phase 8:

- If integration passes, write a final runbook closeout with remaining full-row memory gates and exact commands for future full-row score admission.
- If integration blocks, write a blocker result with the smallest next repair phase and stop for human direction if target or admission policy must change.

## Stop Conditions

Stop and ask for direction or write a blocker result if:

- leaderboard wiring cannot be located or has conflicting row semantics;
- passing integration requires changing target scalar, target policy, or coordinate system;
- tests reveal an admitted historical route that cannot be patched locally;
- a full-row run is needed before integration can be tested;
- implementation would require unrelated dirty-worktree changes;
- review does not converge after five rounds on the same material blocker.

## Skeptical Audit Before Execution

Risks checked before launch:

- Wrong baseline: integration must compare against the shared score contract and compact per-model ports, not historical score artifacts.
- Proxy metrics: tiny score artifacts cannot become full leaderboard admission.
- Hidden assumption: “default compact” means default score route style, not completed full-row score evidence.
- Environment mismatch: CPU-only integration tests must hide GPU; full GPU score runs require trusted execution and a separate gate.
- Useless artifact risk: candidate leaderboard artifacts must clearly label blocked/tiny/non-admitted rows.

Audit status: ready for read-only review after Phase 7 result review.
