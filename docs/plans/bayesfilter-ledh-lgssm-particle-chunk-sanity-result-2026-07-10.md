# LEDH LGSSM Particle Chunk Sanity Result

date: 2026-07-10
status: COMPLETE

## Decision Table

| Field | Result |
|---|---|
| Decision | Both `particle_chunk_size=2500` and `particle_chunk_size=10000` pass the narrow `N=10000,T=1` LGSSM GPU/XLA value sanity check with `row/col=2500`. `2500` is the better candidate from this tiny timing check. |
| Primary criterion status | PASS for both candidates: each run exited 0, wrote JSON, used GPU outputs, reported finite value output, `score_mode=none`, `history_mode=value-only`, `row_chunk_size=col_chunk_size=2500`, and the requested particle chunk. |
| Veto diagnostic status | PASS: no CPU placement, no score path, no dense transport materialization, no exception, and both reached `value_materialize_completed`. |
| Main uncertainty | This is `T=1`, one seed, one warm call. It is only a chunk-shape sanity check, not full `T=50` readiness or score evidence. |
| Next justified action | Prefer `particle_chunk_size=2500` over the historical `512` for the next bounded LGSSM value-stage diagnostic, then test a longer horizon before changing defaults broadly. |
| Not concluded | Full leaderboard readiness, score readiness, FD agreement, HMC readiness, posterior correctness, or optimal particle chunk size for nonlinear/SIR models. |

## Results

| Particle chunk | Blocks | Compile + first call (s) | Value call (s) | Warm call (s) | TF allocator peak (MiB) | Finite output |
|---:|---:|---:|---:|---:|---:|---|
| 2500 | 4 | 10.097530146827921 | 10.089800033019856 | 0.15199913503602147 | 83.86474609375 | true |
| 10000 | 1 | 9.973582784179598 | 9.962811592034996 | 0.23606225801631808 | 83.86474609375 | true |

Both candidates had `max_column_residual=0.0`; `max_row_residual` was
`0.5272895097732544` for `2500` and `0.527414083480835` for `10000`.

## Interpretation

The sanity check supports the user's concern: for LGSSM value mode, large
particle chunks are feasible and the historical `512` is not justified by raw
memory pressure. TensorFlow reported the same allocator peak for `2500` and
`10000` in this short diagnostic.

The `10000` one-shot particle flow is not obviously better, however. In this
single warm-call run, it was slower than `2500` despite similar compile time and
the same allocator peak. That suggests the remaining question is kernel/XLA
runtime behavior, not basic tensor payload memory.

## Artifacts

| Artifact | Path |
|---|---|
| Plan | `docs/plans/bayesfilter-ledh-lgssm-particle-chunk-sanity-plan-2026-07-10.md` |
| `particle_chunk_size=2500` JSON | `docs/plans/artifacts/ledh-lgssm-particle-chunk-sanity-n10000-t1-row2500-col2500-particle2500-2026-07-10.json` |
| `particle_chunk_size=10000` JSON | `docs/plans/artifacts/ledh-lgssm-particle-chunk-sanity-n10000-t1-row2500-col2500-particle10000-2026-07-10.json` |

## Commands Actually Run

```bash
MPLCONFIGDIR=/tmp /usr/bin/timeout 600 /home/chakwong/anaconda3/envs/tf-gpu/bin/python docs/benchmarks/benchmark_ledh_same_target_lgssm_m3_t50_value.py --device-scope visible --expect-device-kind gpu --device /GPU:0 --batch-seeds 81120 --num-particles 10000 --time-steps 1 --history-mode value-only --score-mode none --transport-policy active-all --transport-gradient-mode manual_streaming_finite_sinkhorn_stopped_scale_keys --transport-ad-mode stabilized --sinkhorn-iterations 10 --sinkhorn-epsilon 0.5 --row-chunk-size 2500 --col-chunk-size 2500 --particle-chunk-size 2500 --dtype float32 --tf32-mode enabled --warmups 0 --repeats 1 --output docs/plans/artifacts/ledh-lgssm-particle-chunk-sanity-n10000-t1-row2500-col2500-particle2500-2026-07-10.json
```

```bash
MPLCONFIGDIR=/tmp /usr/bin/timeout 600 /home/chakwong/anaconda3/envs/tf-gpu/bin/python docs/benchmarks/benchmark_ledh_same_target_lgssm_m3_t50_value.py --device-scope visible --expect-device-kind gpu --device /GPU:0 --batch-seeds 81120 --num-particles 10000 --time-steps 1 --history-mode value-only --score-mode none --transport-policy active-all --transport-gradient-mode manual_streaming_finite_sinkhorn_stopped_scale_keys --transport-ad-mode stabilized --sinkhorn-iterations 10 --sinkhorn-epsilon 0.5 --row-chunk-size 2500 --col-chunk-size 2500 --particle-chunk-size 10000 --dtype float32 --tf32-mode enabled --warmups 0 --repeats 1 --output docs/plans/artifacts/ledh-lgssm-particle-chunk-sanity-n10000-t1-row2500-col2500-particle10000-2026-07-10.json
```

## Local Checks

- `git diff --check -- docs/plans/bayesfilter-ledh-lgssm-particle-chunk-sanity-plan-2026-07-10.md`: passed.

## Post-Run Red-Team Note

The main misleading-pass risk is extrapolating from `T=1` to `T=50` or from
LGSSM `D=3` to SIR `D=18`. This result only shows that the one-shot LGSSM flow
chunk is feasible at `T=1`; it does not prove that `10000` is a good default or
that score memory issues are solved.
