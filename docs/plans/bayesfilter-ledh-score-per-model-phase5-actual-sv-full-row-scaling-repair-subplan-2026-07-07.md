# Phase 5 Repair Subplan: Actual-SV Full-Row Score Scaling

metadata_date: 2026-07-07
status: `DRAFT_PENDING_REVIEW`
master_program: `docs/plans/bayesfilter-ledh-score-per-model-master-program-2026-07-07.md`
phase: 5-repair-full-row-score-scaling

## Phase Objective

Resolve the actual-SV full-row score scaling blocker without weakening the
same-target, no-tape, and score-correctness evidence contract.

## Entry Conditions Inherited From Previous Phase

- Streaming-flow parity repair passed tiny diagnostics.
- Required local checks passed with `30 passed, 2 warnings`.
- GPU rung `T=5,N=256` passed as diagnostic evidence only.
- GPU rung `T=20,N=1024` was manually interrupted after roughly 15 minutes with
  near-budget GPU memory pressure and no JSON artifact.
- Full `N=10000,T=1000` score remains not admitted.

## Required Artifacts

Input artifacts:

- `docs/plans/bayesfilter-ledh-score-per-model-phase5-actual-sv-streaming-flow-parity-repair-result-2026-07-07.md`
- `docs/plans/bayesfilter-ledh-score-per-model-phase5-actual-sv-full-row-score-refresh-result-2026-07-07.md`
- `docs/plans/bayesfilter-ledh-score-per-model-phase5-actual-sv-gpu-ladder-t5-n256-2026-07-07.json`

Code/tests:

- `docs/benchmarks/benchmark_ledh_same_target_actual_sv_score.py`
- `bayesfilter/highdim/ledh_score_contract.py`
- `tests/highdim/test_ledh_actual_sv_score_phase5_contract.py`
- `tests/highdim/test_ledh_score_contract_phase1.py`

Expected artifacts:

- scaling repair result:
  `docs/plans/bayesfilter-ledh-score-per-model-phase5-actual-sv-full-row-scaling-repair-result-2026-07-07.md`;
- any diagnostic JSON/Markdown outputs from reviewed probes;
- refreshed full-row admission subplan or explicit blocker;
- review bundle:
  `docs/reviews/bayesfilter-ledh-score-per-model-phase5-actual-sv-full-row-scaling-repair-review-bundle-2026-07-07.md`.

## Required Checks/Tests/Reviews

Before implementation:

- bounded read-only review of this scaling repair subplan.

After implementation:

```text
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/highdim/test_ledh_actual_sv_score_phase5_contract.py \
  tests/highdim/test_ledh_score_contract_phase1.py -q
```

Any TensorFlow GPU probe must use trusted/escalated execution.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Can we produce a validator-compatible full-row actual-SV score evidence path without repeated full-row all-coordinate FD blowup? |
| Baseline/comparator | Existing repaired tiny FD diagnostics, admitted actual-SV value artifact, Phase 1 score validator, rung `T=5,N=256`, interrupted rung `T=20,N=1024`. |
| Primary criterion | A reviewed path exists that either makes full all-coordinate correctness feasible or supplies a validator-compatible `exact_reference` correctness artifact without weakening no-tape or same-target constraints. |
| Veto diagnostics | Tape/autodiff, stopped partials, wrong scalar, target substitution, runtime-only promotion, memory-only promotion, validator over-relaxation, unsupported exact-reference claim. |
| Explanatory diagnostics | Per-stage timing, memory peak, shape scaling, transport VJP cost, record payload estimate, no-FD value/score-only runtime. |
| Not concluded | Full score admission, HMC readiness, posterior correctness, scientific superiority, runtime ranking, or other-model score readiness. |
| Artifact | Scaling repair result plus refreshed full-row admission subplan or blocker. |

## Step-By-Step Plan

1. Add a CLI/debug mode or helper that runs the repaired no-tape value/score
   route without coordinate FD and writes timing/memory diagnostics as
   non-admission evidence.
2. Add optional per-stage timing around:
   - forward record construction;
   - transport VJP;
   - flow VJP;
   - target transition/observation VJP;
   - coordinate FD objective replays.
3. Add tests that:
   - confirm no-FD diagnostic artifacts cannot be admitted;
   - confirm `exact_reference` cannot be claimed without required metadata;
   - preserve existing tiny FD and no-autodiff sentinels.
4. Run a trusted GPU no-FD memory/timing ladder:
   - `T=20,N=1024`;
   - if safe, one larger shape selected by the result evidence, not by optimism.
5. Decide between two reviewed admission routes:
   - make full FD feasible through checkpointing/recompute and instrumentation;
   - or define a strict `exact_reference` artifact contract for a manual
     derivation/reference replay that avoids full FD.
6. If code or validator semantics change, review before any full admission
   attempt.
7. Write the scaling repair result and refresh the full-row admission subplan
   or blocker.

## Forbidden Claims/Actions

- Do not rerun full `N=10000,T=1000` score admission before this subplan
  converges.
- Do not claim admission from no-FD timing or memory probes.
- Do not add or use tape, `ForwardAccumulator`, hidden autodiff, or stopped
  partials.
- Do not substitute KSC, raw Gaussian, augmented-noise, or matrix-flow evidence.
- Do not broaden `exact_reference` validation without a reviewed strict
  artifact contract.
- Do not treat ladder runtime/memory as correctness evidence.

## Exact Next-Phase Handoff Conditions

A refreshed full-row actual-SV score admission subplan may start only if:

- this scaling repair result exists;
- local checks pass;
- the result identifies a concrete validator-compatible correctness path;
- read-only review agrees the next full-row handoff is boundary-safe.

## Stop Conditions

Stop and write a blocker result if:

- no-FD value/score-only execution also shows unsafe memory/runtime scaling at
  modest shapes;
- per-stage instrumentation cannot isolate the blocker;
- exact-reference semantics would require unsupported mathematical or
  validator claims;
- review finds a material issue that does not converge after five rounds.

## Skeptical Plan Audit

| Risk | Control |
| --- | --- |
| Wrong baseline | Keep admitted value artifact and repaired same-forward-scalar tests as anchors. |
| Proxy promotion | No-FD and timing probes are explicitly non-admission evidence. |
| Missing stop condition | Stop on unsafe no-FD scaling, unsupported exact-reference semantics, or unresolved review issues. |
| Hidden assumption | Instrument the actual transport/flow VJP stages rather than guessing. |
| Stale context | Starts from the interrupted `T=20,N=1024` rung and current validator constraints. |
| Environment mismatch | GPU diagnostics require trusted execution. |
| Useless artifact | Result must either refresh an admission subplan or explicitly block. |
