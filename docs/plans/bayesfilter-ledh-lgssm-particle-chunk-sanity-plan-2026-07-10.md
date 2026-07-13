# LEDH LGSSM Particle Chunk Sanity Plan

date: 2026-07-10
status: READY_TO_RUN

## Question

For the LGSSM same-target LEDH value path at `N=10000`, does replacing the
historical `particle_chunk_size=512` with larger exact particle chunks produce a
basic GPU/XLA value result when OT row/column chunks are fixed at the previously
preferred `2500 x 2500`?

## Evidence Contract

| Field | Contract |
|---|---|
| Question | Compare `particle_chunk_size=2500` and `particle_chunk_size=10000` on the LGSSM value path with fixed `row_chunk_size=col_chunk_size=2500`. |
| Comparator | The two candidates are compared only to each other in this sanity check. Historical `512` remains a safe prior, not an optimality claim. |
| Primary criterion | Each command exits 0 and writes JSON with GPU output devices, finite value output, `N=10000`, `T=1`, one seed, `score_mode=none`, `history_mode=value-only`, `row/col=2500`, and the requested particle chunk. |
| Veto diagnostics | CPU placement, missing/failed JSON artifact, exception, non-finite value, unexpected dense transport path, score computation launched, or timeout before `value_materialize_completed`. |
| Explanatory only | Compile plus first-call time, warm-call time, GPU allocator stats, value delta to exact prefix target, and row/column residuals. |
| Not concluded | Full `T=50` readiness, score readiness, FD agreement, HMC readiness, leaderboard admission, or scientific correctness. |
| Artifacts | Candidate JSON files under `docs/plans/artifacts/`; result note under `docs/plans/`. |

## Skeptical Plan Audit

- Wrong baseline risk: avoided by calling this a two-candidate sanity check, not
  a default-setting promotion.
- Proxy metric risk: timing is explanatory only; the primary gate is finite GPU
  value execution.
- Missing stop condition risk: each candidate uses a bounded timeout and stops on
  first material execution failure.
- Hidden assumption risk: the plan assumes the current LGSSM runner's `T=1`
  prefix mode exercises the same compiled value core as longer horizons. This is
  enough for chunk-shape sanity but not for full-row readiness.
- Artifact relevance risk: the JSON stage markers directly answer whether the
  call returned/materialized or timed out.

Audit result: PASS for this narrow diagnostic.

## Commands

Run both with trusted GPU permissions.

```bash
MPLCONFIGDIR=/tmp /usr/bin/timeout 600 /home/chakwong/anaconda3/envs/tf-gpu/bin/python docs/benchmarks/benchmark_ledh_same_target_lgssm_m3_t50_value.py --device-scope visible --expect-device-kind gpu --device /GPU:0 --batch-seeds 81120 --num-particles 10000 --time-steps 1 --history-mode value-only --score-mode none --transport-policy active-all --transport-gradient-mode manual_streaming_finite_sinkhorn_stopped_scale_keys --transport-ad-mode stabilized --sinkhorn-iterations 10 --sinkhorn-epsilon 0.5 --row-chunk-size 2500 --col-chunk-size 2500 --particle-chunk-size 2500 --dtype float32 --tf32-mode enabled --warmups 0 --repeats 1 --output docs/plans/artifacts/ledh-lgssm-particle-chunk-sanity-n10000-t1-row2500-col2500-particle2500-2026-07-10.json
```

```bash
MPLCONFIGDIR=/tmp /usr/bin/timeout 600 /home/chakwong/anaconda3/envs/tf-gpu/bin/python docs/benchmarks/benchmark_ledh_same_target_lgssm_m3_t50_value.py --device-scope visible --expect-device-kind gpu --device /GPU:0 --batch-seeds 81120 --num-particles 10000 --time-steps 1 --history-mode value-only --score-mode none --transport-policy active-all --transport-gradient-mode manual_streaming_finite_sinkhorn_stopped_scale_keys --transport-ad-mode stabilized --sinkhorn-iterations 10 --sinkhorn-epsilon 0.5 --row-chunk-size 2500 --col-chunk-size 2500 --particle-chunk-size 10000 --dtype float32 --tf32-mode enabled --warmups 0 --repeats 1 --output docs/plans/artifacts/ledh-lgssm-particle-chunk-sanity-n10000-t1-row2500-col2500-particle10000-2026-07-10.json
```

## Stop Conditions

- Stop after both candidates complete.
- Stop early if the first candidate fails in a way that makes the second unsafe.
- Do not run score mode or full `T=50` inside this sanity check.
