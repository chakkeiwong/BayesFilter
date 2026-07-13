# Phase 2V Subplan: LGSSM Seed-Sharded N10000 Score Admission Repair

Date: 2026-07-09

Status: `DRAFT_READY_FOR_REVIEW`

## Phase Objective

Repair the Phase 2U full-run artifact blocker by implementing and testing a
seed-sharded LGSSM score admission procedure. The procedure must compute the
same full-row target score as the five-seed batch mean, but it may execute each
fixed seed in a separate trusted GPU process and aggregate durable raw shard
artifacts afterward.

## Entry Conditions Inherited From Previous Phase

- Phase 1 shared score artifact validator passed.
- Phase 2S repaired value-only same-scalar FD and score-specific memory
  measurement.
- Phase 2T selected the disclosed correctness policy: production TF32 enabled,
  FD correctness arm TF32 disabled and disclosed.
- Phase 2U reviewed and launched the monolithic full LGSSM run, but no artifact
  was emitted under the bounded runtime/memory window.
- No LGSSM `N=10000,T=50` score artifact is admitted yet.

## Required Artifacts

- Phase 2V result:
  `docs/plans/bayesfilter-ledh-n10000-score-admission-repair-phase2v-lgssm-sharded-admission-result-2026-07-09.md`
- Per-seed raw shard JSONs:
  `docs/plans/artifacts/ledh-n10000-score-admission-repair-phase2v-lgssm-shards/lgssm-seed-<seed>-raw-score.json`
- Per-seed raw shard logs:
  `docs/plans/logs/bayesfilter-ledh-n10000-score-admission-repair-phase2v-lgssm-seed-<seed>.log`
- Aggregate score artifact:
  `docs/plans/bayesfilter-ledh-n10000-score-admission-repair-phase2v-lgssm-score-artifact-2026-07-09.json`
- Aggregate Markdown companion:
  `docs/plans/bayesfilter-ledh-n10000-score-admission-repair-phase2v-lgssm-score-artifact-2026-07-09.md`
- Review bundle:
  `docs/reviews/bayesfilter-ledh-n10000-score-admission-repair-phase2u-result-phase2v-subplan-review-bundle-2026-07-09.md`

## Required Checks, Tests, And Reviews

Local implementation checks:

```bash
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/highdim/test_ledh_lgssm_score_phase2_contract.py \
  tests/test_ledh_lgssm_manual_score_phase4.py \
  tests/highdim/test_ledh_score_contract_phase1.py -q
```

Focused tests to add or refresh:

- shard aggregation rejects missing full fixed-seed coverage;
- shard aggregation rejects stale `N`, stale `T`, wrong row id, wrong target
  scalar, wrong parameter order, historical route, nonpassing FD, and non-GPU
  shard runtime gate for admitted aggregation;
- shard aggregation computes score as the arithmetic mean of per-seed scores;
- shard aggregation computes same-scalar FD as the arithmetic mean of per-seed
  FD values and recomputes aggregate residuals;
- shard aggregation records segmented execution and max score-memory peak;
- partial shard output remains diagnostic and is not validator-admitted.

Trusted GPU smoke before full shards:

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
  --score-fd-tf32-mode disabled \
  --history-mode value-only \
  --dtype float32 \
  --tf32-mode enabled \
  --device /GPU:0 \
  --device-scope visible \
  --expect-device-kind gpu \
  --output /tmp/phase2v-lgssm-shard-smoke.json
```

Review:

- Run read-only review of the Phase 2U result plus this Phase 2V subplan before
  implementation.
- Claude may be used only if local policy permits bounded review. If local
  policy rejects Claude external disclosure, use a fresh Codex packet-only
  review and record the limitation.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Can the full LGSSM score be admitted by exact fixed-seed sharding and aggregation without changing the target scalar? |
| Baseline/comparator | Phase 2U monolithic no-artifact blocker, Phase 2S/2T smoke gates, and admitted LGSSM value artifact. |
| Primary criterion | Aggregate nested score artifact validates with `validate_ledh_score_artifact(..., require_admitted=True)` against the admitted LGSSM value artifact. |
| Comparator check | On a small deterministic case, direct multi-seed batch score equals seed-sharded aggregate within stated float32 tolerance. |
| Veto diagnostics | Missing seed; duplicate seed; stale `N`/`T`; wrong row/target/parameter order; FD fail; non-GPU shard runtime gate; memory peak above budget; historical route; tape/autodiff; stopped partial derivative; partial shard promoted as admitted; undisclosed segmented execution. |
| Explanatory diagnostics | Per-seed runtime, per-seed peak memory, aggregate residuals, score variance across seeds, shard logs, and max shard memory. |
| Not concluded | Monolithic five-seed memory pass, runtime ranking, HMC readiness, posterior correctness, exact Kalman score equality, or non-LGSSM admission. |

## Forbidden Claims And Actions

- Do not claim the monolithic five-seed process passes memory or runtime if the
  admitted artifact is produced by segmented execution.
- Do not change the score target: it remains the derivative of the realized
  finite-`N` LEDH `observed_data_log_likelihood_estimator`, reported as
  `log_likelihood`.
- Do not change full-row seeds, `N`, `T`, theta coordinate system, parameter
  order, transport policy, Sinkhorn settings, source value artifact, or
  production precision policy.
- Do not admit a partial seed shard.
- Do not use `GradientTape`, `ForwardAccumulator`, stopped partials, or
  historical `manual_total_vjp*` evidence.
- Do not hide the separate no-TF32 FD correctness arm.
- Do not proceed to fixed-SIR until LGSSM is admitted or Phase 2V writes a
  reviewed blocker result.

## Exact Next-Phase Handoff Conditions

Phase 3 fixed-SIR may begin only if:

- Phase 2V writes a result record;
- ledger is updated;
- either the aggregate LGSSM score artifact validates with
  `require_admitted=True`, or the result records a precise blocker and next
  smallest repair;
- Phase 3 subplan is refreshed and reviewed.

## Stop Conditions

Stop if:

- the aggregation contract cannot be made exact for the full-row batch mean;
- small deterministic direct-batch versus shard-aggregate comparison fails;
- trusted GPU smoke fails;
- any required shard fails FD, runtime, finite-output, row identity, or memory
  gates;
- aggregate validator admission fails;
- local review does not converge after five rounds;
- continuing requires package installation, network/data fetches, credentials,
  destructive git actions, or changing pass/fail criteria after seeing results.

## Skeptical Audit Before Execution

- Wrong baseline checked: the target is not exact Kalman likelihood and not a
  new scalar; it is the admitted finite-`N` LEDH log-likelihood estimator.
- Proxy metric checked: per-seed completion is not admission; only aggregate
  validator admission can admit the score.
- Hidden assumption checked: seed-sharding is valid only because the full-row
  scalar is the arithmetic mean over fixed seeds and the filter dynamics do not
  couple batch elements.
- Memory claim checked: segmented max-shard memory is disclosed and is not a
  monolithic batch-memory claim.
- Artifact sufficiency checked: raw shard JSONs and logs are durable before
  aggregation, so a late failure cannot erase completed evidence.

Audit status: ready for read-only review before implementation.
