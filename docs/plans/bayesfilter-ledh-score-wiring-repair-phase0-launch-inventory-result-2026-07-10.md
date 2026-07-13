# Phase 0 Result: Launch Inventory And Governance Freeze

Date: 2026-07-10

## Decision Table

| Decision item | Status |
| --- | --- |
| Master program created | Passed |
| Visible runbook created | Passed |
| Phase 0 subplan created | Passed |
| Phase 1 subplan drafted | Passed |
| Local inventory completed | Passed |
| Py-compile check | Passed |
| Claude review | Blocked by execution policy; substitute Codex review completed |
| Substitute Codex review | Passed: `VERDICT: AGREE` |
| Code repair | Not attempted in Phase 0 |
| Score admission | Not claimed |

## Evidence Contract Status

| Field | Status |
| --- | --- |
| Question | Answered for launch: the program targets compact default score wiring and blocks relabel-only repairs. |
| Baseline/comparator | Current code inventory and model-by-model 2026-07-10 classification. |
| Primary criterion | Met except review verdict pending: launch artifacts exist and state compact/default route, precision, FD, review, and stop-condition gates. |
| Veto diagnostics | No veto in local plan artifacts. Full GPU ladders are deferred; historical full admission is forbidden; score-only memory is non-admission. |
| Not concluded | No model code repair, no score admission, no leaderboard completion, no HMC/posterior/scientific claim. |

## Local Checks

Inventory command:

```bash
rg -n -- "manual_total_vjp|memory_style|COMPACT_SCORE_ROUTE_ID|score_derivative_provenance|--dtype|--tf32-mode" docs/benchmarks bayesfilter/highdim tests/highdim > docs/plans/logs/bayesfilter-ledh-score-wiring-repair-phase0-inventory-rg-2026-07-10.log
```

Result: passed; `918` matching lines recorded.

Py-compile command:

```bash
python -m py_compile \
  bayesfilter/highdim/ledh_score_contract.py \
  bayesfilter/highdim/ledh_score_artifact.py \
  docs/benchmarks/benchmark_ledh_same_target_lgssm_m3_t50_value.py \
  docs/benchmarks/benchmark_ledh_same_target_fixed_sir_score.py \
  docs/benchmarks/benchmark_ledh_same_target_predator_prey_score.py \
  docs/benchmarks/benchmark_ledh_same_target_actual_sv_score.py \
  docs/benchmarks/benchmark_ledh_same_target_generalized_sv_score.py \
  docs/benchmarks/benchmark_ledh_same_target_ksc_sv_score.py \
  > docs/plans/logs/bayesfilter-ledh-score-wiring-repair-phase0-pycompile-2026-07-10.log 2>&1
```

Result: passed; log is empty.

## Inventory Findings

| Model | Finding | Required follow-up |
| --- | --- | --- |
| LGSSM | Compact default route is present; naming residue includes `RAW_MEMORY_STYLE_ADMITTED_STATUS` and `HISTORICAL_COMPACT_SCORE_ROUTE_ID`. Defaults are `float32`; TF32 mode argument exists. | Phase 2 cleanup and timing instrumentation. |
| fixed-SIR | Compact score helper and artifact builder exist. Historical memory-result normalizer remains import-only diagnostic. | Phase 3 ensure current/default route is compact and legacy normalizer cannot be confused with current computation. |
| predator-prey | Compact helper exists, but current default coordinate-FD path uses `_manual_value_and_score_across_seeds` and memory-style route. CLI defaults to `float64` and TF32 disabled. | Phase 4 switch default to compact and fix precision defaults. |
| actual-SV | Compact helper exists, but current coordinate-FD/default path uses memory-style route. CLI defaults to `float64` and TF32 disabled. | Phase 5 switch default to compact and fix precision defaults. |
| generalized-SV | Default path calls compact route. CLI defaults to `float64` and TF32 disabled. | Phase 6 fix precision defaults and add source tests. |
| KSC-SV | Default path calls compact route. CLI defaults to `float64` and TF32 disabled. | Phase 7 fix precision defaults and add source tests. |

## Review Status

Claude review gate command was attempted:

```bash
bash /home/chakwong/python/claudecodex/scripts/claude_review_gate.sh \
  --cwd /home/chakwong/BayesFilter \
  --review-name bayesfilter-ledh-score-wiring-repair-launch-2026-07-10 \
  --bundle /home/chakwong/BayesFilter/docs/reviews/bayesfilter-ledh-score-wiring-repair-launch-review-bundle-2026-07-10.md \
  --probe-timeout 90 \
  --timeout-seconds 120 \
  --max-retries 1 \
  --allow-bounded-fallback
```

The command was rejected by the execution policy as external data disclosure.
No workaround was attempted. A fresh Codex read-only substitute review was
requested with the same review questions.

Substitute review status: `VERDICT: AGREE`.

Substitute review finding:

> No blocking findings. The launch artifacts consistently require the default
> LEDH score path to physically call the compact no-time-history recurrence,
> not merely relabel historical routes.

Minor non-blocking note: the prior LGSSM compact artifact is cited as baseline
context without inline artifact path in the launch docs. This is not promoted to
a pass criterion and is guarded by later physical-call tests.

## Plain-Language Gate

- Target: repair default score wiring, not prove score correctness.
- Computed quantity in Phase 0: inventory and syntax checks only.
- Direct classification: launch artifacts are sufficient for Phase 1 planning;
  model score wiring remains `not checked/repaired` beyond inventory.
- Unsupported claims: no score admission, no GPU memory claim, no leaderboard
  completion.

## Next Phase Handoff

Phase 1 may begin. Phase 1 is limited to shared score contract and precision
gate tests; it must not change per-model numerical score algorithms except for
trivial import/name compatibility required by shared tests.
