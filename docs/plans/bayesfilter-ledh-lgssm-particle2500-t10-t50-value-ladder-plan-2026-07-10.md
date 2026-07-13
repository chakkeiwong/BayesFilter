# LEDH LGSSM Particle2500 T10/T50 Value Ladder Plan

date: 2026-07-10
status: READY_TO_RUN

## Question

With OT chunks fixed at `row_chunk_size=col_chunk_size=2500`, does the larger
`particle_chunk_size=2500` LGSSM value route remain feasible as the horizon
increases from the completed `T=1` sanity check to `T=10` and then `T=50`?

## Evidence Contract

| Field | Contract |
|---|---|
| Question | Run a sequential one-seed LGSSM value-only ladder for `T=10` then `T=50`, using `N=10000`, `row/col/particle=2500`, GPU/XLA, TF32, streaming transport, and no score path. |
| Comparator | Prior `T=1` particle-chunk sanity result in `docs/plans/bayesfilter-ledh-lgssm-particle-chunk-sanity-result-2026-07-10.md`. |
| Primary criterion | Each launched command exits 0 and writes JSON with GPU output devices, finite value output, `score_mode=none`, `history_mode=value-only`, `N=10000`, requested `T`, `row/col/particle=2500`, and `value_materialize_completed`. |
| Veto diagnostics | CPU placement, timeout before `value_materialize_completed`, exception, non-finite value, score path launched, dense transport path, or missing terminal artifact. |
| Explanatory only | Compile plus first-call time, warm-call time, allocator peak, value delta to exact comparator, row/column residuals, and elapsed wall time. |
| Not concluded | Full five-seed leaderboard readiness, score readiness, FD agreement, HMC readiness, posterior correctness, scientific validity, or optimal chunk size for nonlinear/SIR models. |
| Artifacts | Candidate JSON files under `docs/plans/artifacts/`; result note under `docs/plans/`. |

## Skeptical Plan Audit

- Wrong baseline risk: avoided by treating `T=1` as a feasibility predecessor,
  not a performance baseline for promotion.
- Proxy metric risk: timing is explanatory only; finite GPU value completion is
  the pass/fail criterion.
- Missing stop condition risk: run `T=10` first and stop before `T=50` if it
  fails or times out before materialization.
- Hidden assumption risk: one seed does not represent the full leaderboard row.
  The plan records that explicitly.
- Artifact relevance risk: the runner emits stage markers into the JSON, so a
  timeout still preserves the stage reached.

Audit result: PASS for this bounded ladder.

## Commands

Run with trusted GPU permissions.

### T=10

```bash
MPLCONFIGDIR=/tmp /usr/bin/timeout 1200 /home/chakwong/anaconda3/envs/tf-gpu/bin/python docs/benchmarks/benchmark_ledh_same_target_lgssm_m3_t50_value.py --device-scope visible --expect-device-kind gpu --device /GPU:0 --batch-seeds 81120 --num-particles 10000 --time-steps 10 --history-mode value-only --score-mode none --transport-policy active-all --transport-gradient-mode manual_streaming_finite_sinkhorn_stopped_scale_keys --transport-ad-mode stabilized --sinkhorn-iterations 10 --sinkhorn-epsilon 0.5 --row-chunk-size 2500 --col-chunk-size 2500 --particle-chunk-size 2500 --dtype float32 --tf32-mode enabled --warmups 0 --repeats 1 --output docs/plans/artifacts/ledh-lgssm-particle2500-value-ladder-n10000-t10-row2500-col2500-2026-07-10.json
```

### T=50

Only run after the `T=10` command passes.

```bash
MPLCONFIGDIR=/tmp /usr/bin/timeout 2400 /home/chakwong/anaconda3/envs/tf-gpu/bin/python docs/benchmarks/benchmark_ledh_same_target_lgssm_m3_t50_value.py --device-scope visible --expect-device-kind gpu --device /GPU:0 --batch-seeds 81120 --num-particles 10000 --time-steps 50 --history-mode value-only --score-mode none --transport-policy active-all --transport-gradient-mode manual_streaming_finite_sinkhorn_stopped_scale_keys --transport-ad-mode stabilized --sinkhorn-iterations 10 --sinkhorn-epsilon 0.5 --row-chunk-size 2500 --col-chunk-size 2500 --particle-chunk-size 2500 --dtype float32 --tf32-mode enabled --warmups 0 --repeats 1 --output docs/plans/artifacts/ledh-lgssm-particle2500-value-ladder-n10000-t50-row2500-col2500-2026-07-10.json
```

## Stop Conditions

- Stop before `T=50` if `T=10` fails, times out, or does not reach
  `value_materialize_completed`.
- Stop after `T=50` completes or times out.
- Do not run score mode, FD, five seeds, or leaderboard integration inside this
  ladder.
