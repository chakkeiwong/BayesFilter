# Phase 2W Subplan: LGSSM Full-Shard Score/FD Split Repair

Date: 2026-07-09

Status: `DRAFT_READY_FOR_REVIEW`

## Phase Objective

Repair the Phase 2V single-shard no-artifact blocker by splitting full-shard
score computation from finite-difference correctness. The next implementation
must be able to emit a durable score-only raw shard immediately after compact
score completion, then run finite-difference correctness as a separate bounded
stage. Full score admission remains impossible until both score and correctness
are present and aggregated through the existing validator.

## Entry Conditions Inherited From Previous Phase

- Phase 2S repaired same-scalar FD and score-specific memory measurement.
- Phase 2T established the disclosed no-TF32 FD correctness policy.
- Phase 2V implemented exact seed-sharded aggregation and passed focused tests,
  synthetic CLI aggregation smoke, and trusted GPU `N=256,T=3` shard smoke.
- Phase 2V first full shard seed `81120` did not emit a raw artifact after
  about six minutes, despite trusted GPU initialization and XLA compilation.
- No LGSSM `N=10000,T=50` score artifact is admitted yet.

## Required Artifacts

- Phase 2W result:
  `docs/plans/bayesfilter-ledh-n10000-score-admission-repair-phase2w-lgssm-score-fd-split-result-2026-07-09.md`
- Score-only shard artifact:
  `docs/plans/artifacts/ledh-n10000-score-admission-repair-phase2w-lgssm-score-fd-split/lgssm-seed-81120-score-only.json`
- FD-only shard artifact or blocker:
  `docs/plans/artifacts/ledh-n10000-score-admission-repair-phase2w-lgssm-score-fd-split/lgssm-seed-81120-fd-only.json`
- Combined diagnostic shard artifact:
  `docs/plans/artifacts/ledh-n10000-score-admission-repair-phase2w-lgssm-score-fd-split/lgssm-seed-81120-combined-diagnostic.json`
- Logs:
  `docs/plans/logs/bayesfilter-ledh-n10000-score-admission-repair-phase2w-*.log`
- Review bundle:
  `docs/reviews/bayesfilter-ledh-n10000-score-admission-repair-phase2v-result-phase2w-subplan-review-bundle-2026-07-09.md`

## Required Checks, Tests, And Reviews

Implementation checks:

```bash
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/highdim/test_ledh_lgssm_score_phase2_contract.py \
  tests/test_ledh_lgssm_manual_score_phase4.py \
  tests/highdim/test_ledh_score_contract_phase1.py -q
```

Focused tests to add or refresh:

- score-only mode emits compact no-tape score and per-seed score but marks
  correctness as missing and cannot be admitted;
- FD-only mode uses the value-only scalar route and cannot call the compact
  score/JVP route;
- combined shard assembly rejects missing score-only artifact, missing FD-only
  artifact, stale seed, stale `N`, stale `T`, wrong target, wrong parameter
  order, non-GPU score route, and failed FD;
- aggregation rejects score-only shards without correctness;
- no source in score-only or FD-only helpers contains `GradientTape`,
  `ForwardAccumulator`, or historical `manual_total_vjp` admission language.

Trusted GPU smoke:

```bash
MPLCONFIGDIR=/tmp python docs/benchmarks/benchmark_ledh_same_target_lgssm_m3_t50_value.py \
  --num-particles 256 \
  --time-steps 3 \
  --batch-seeds 81120 \
  --transport-policy active-all \
  --sinkhorn-iterations 2 \
  --sinkhorn-epsilon 0.5 \
  --transport-ad-mode full \
  --transport-gradient-mode manual_streaming_finite_sinkhorn_stopped_scale_keys \
  --row-chunk-size 64 \
  --col-chunk-size 64 \
  --particle-chunk-size 64 \
  --score-mode compact-sensitivity \
  --score-diagnostic-stage score-only \
  --history-mode value-only \
  --dtype float32 \
  --tf32-mode enabled \
  --device /GPU:0 \
  --device-scope visible \
  --expect-device-kind gpu \
  --output /tmp/phase2w-lgssm-score-only-smoke.json
```

Review:

- Run read-only review of the Phase 2V result plus this Phase 2W subplan before
  implementation.
- Claude may be used only if local policy permits bounded review. If local
  policy rejects Claude external disclosure, use a fresh Codex packet-only
  review and record the limitation.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Is the full-shard no-artifact blocker caused by compact score execution, FD correctness execution, or the unsplit combination? |
| Baseline/comparator | Phase 2V single-shard full command with no artifact after bounded window. |
| Primary criterion | A trusted GPU score-only `N=10000,T=50` shard either emits a durable raw score-only artifact or writes a precise score-pass blocker. |
| Secondary criterion | If score-only emits, FD-only emits a matching correctness artifact or writes a precise FD blocker. |
| Admission criterion | No full admission until combined shards aggregate and `validate_ledh_score_artifact(..., require_admitted=True)` passes. |
| Veto diagnostics | Score-only artifact admitted; FD-only calls compact score/JVP route; target drift; missing seed identity; historical route; tape/autodiff; stopped partials; non-GPU score route; failed FD; hidden no-TF32 FD policy. |
| Explanatory diagnostics | Score-only runtime, score-only peak memory, FD-only runtime, FD residuals, and whether the bottleneck is score or FD. |
| Not concluded | LGSSM score admission, HMC readiness, posterior correctness, exact Kalman score equality, runtime ranking, or non-LGSSM admission. |

## Forbidden Claims And Actions

- Do not admit score-only artifacts.
- Do not treat FD-only correctness as score evidence without a matching
  score-only artifact for the same seed and settings.
- Do not change target scalar, `N`, `T`, seeds, parameter order, source value
  artifact, transport policy, Sinkhorn settings, production TF32 policy, or
  disclosed no-TF32 FD arm.
- Do not use `GradientTape`, `ForwardAccumulator`, stopped partials, or
  historical `manual_total_vjp*` evidence.
- Do not proceed to fixed-SIR until LGSSM is admitted or Phase 2W writes a
  reviewed blocker and the master-program handoff condition is explicitly
  updated.

## Exact Next-Phase Handoff Conditions

Phase 3 fixed-SIR may begin only if:

- Phase 2W writes a result record;
- ledger is updated;
- either LGSSM score is admitted through the aggregate validator, or the result
  records a precise blocker and the next smallest repair;
- Phase 3 subplan is refreshed and reviewed.

## Stop Conditions

Stop if:

- local tests fail and cannot be fixed without changing the admission contract;
- trusted GPU score-only full shard fails to emit under the reviewed bounded
  window;
- FD-only correctness fails or cannot be matched to score-only identity;
- aggregate validator admission fails;
- review does not converge after five rounds;
- continuing requires package installation, network/data fetches, credentials,
  destructive git actions, or changing pass/fail criteria after seeing results.

## Skeptical Audit Before Execution

- Wrong baseline checked: this phase diagnoses the same finite-`N` LEDH score
  path, not exact Kalman likelihood.
- Proxy metric checked: score-only artifact emission is a diagnostic milestone,
  not admission.
- Hidden assumption checked: FD-only must use the same value-only scalar route
  from Phase 2S and must not call compact score/JVP.
- Artifact sufficiency checked: split artifacts make late FD failure visible
  without losing completed score evidence.
- Boundary checked: no later model phase starts unless LGSSM is admitted or a
  reviewed handoff explicitly permits it.

Audit status: ready for read-only review before implementation.
