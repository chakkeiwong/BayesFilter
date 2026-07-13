# LEDH LGSSM Particle2500 Score-Only Ladder Plan

date: 2026-07-10
status: READY_TO_RUN

## Question

With `row_chunk_size=col_chunk_size=particle_chunk_size=2500`, can the current
LGSSM memory-style score route execute on GPU/XLA for `N=10000` without invoking
the historical full-history diagnostic or finite-difference stage?

## Evidence Contract

| Field | Contract |
|---|---|
| Question | Run `score_mode=manual-reverse`, `score_diagnostic_stage=score-only` for `T=10` then `T=50`, using `N=10000`, one seed, GPU/XLA, TF32, streaming transport, and chunks `2500/2500/2500`. |
| Comparator | The completed value ladder `docs/plans/bayesfilter-ledh-lgssm-particle2500-t10-t50-value-ladder-result-2026-07-10.md`, plus the score route metadata emitted by the runner. |
| Primary criterion | Each launched command exits 0 and writes JSON with GPU score output devices, finite score values, `score_route` equal to the memory-style route, `old_full_history_route_status=historical_forward_sensitivity_route_not_used`, `score_execution_style=memory_style_reverse_vjp_no_particle_param_axis`, `diagnostic_stage=score-only`, `N=10000`, requested `T`, and chunks `2500/2500/2500`. |
| Veto diagnostics | CPU score placement, exception, timeout before `score_completed`, non-finite score, historical route marked used, FD/admission claimed, dense transport materialization, or missing terminal artifact. |
| Explanatory only | Score values, objective, compile/value timing, GPU memory stats, and value finite-output status. |
| Not concluded | Same-scalar FD correctness, score admission, full five-seed leaderboard readiness, HMC readiness, posterior correctness, scientific validity, or nonlinear/SIR chunk policy. |
| Artifacts | Candidate JSON files under `docs/plans/artifacts/`; result note under `docs/plans/`. |

## Skeptical Plan Audit

- Wrong baseline risk: avoided by making this score-only feasibility, not score
  correctness or admission.
- Proxy metric risk: timing/memory are explanatory only; finite GPU score
  completion with memory-style route metadata is the pass/fail criterion.
- Missing stop condition risk: run `T=10` first and stop before `T=50` if it
  fails, times out, or does not reach `score_completed`.
- Hidden assumption risk: the runner currently requires `transport_ad_mode=full`
  for score. The plan uses that requirement and records that this differs from
  the value-only stabilized run.
- Artifact relevance risk: the JSON includes score route metadata and memory
  stats, which directly answer whether the memory-style route ran.

Audit result: PASS for this bounded score-only ladder.

## Commands

Run with trusted GPU permissions.

### T=10 Score-Only

```bash
MPLCONFIGDIR=/tmp /usr/bin/timeout 1800 /home/chakwong/anaconda3/envs/tf-gpu/bin/python docs/benchmarks/benchmark_ledh_same_target_lgssm_m3_t50_value.py --device-scope visible --expect-device-kind gpu --device /GPU:0 --batch-seeds 81120 --num-particles 10000 --time-steps 10 --history-mode value-only --score-mode manual-reverse --score-diagnostic-stage score-only --transport-policy active-all --transport-gradient-mode manual_streaming_finite_sinkhorn_stopped_scale_keys --transport-ad-mode full --sinkhorn-iterations 10 --sinkhorn-epsilon 0.5 --row-chunk-size 2500 --col-chunk-size 2500 --particle-chunk-size 2500 --dtype float32 --tf32-mode enabled --warmups 0 --repeats 1 --output docs/plans/artifacts/ledh-lgssm-particle2500-score-only-n10000-t10-row2500-col2500-2026-07-10.json
```

### T=50 Score-Only

Only run after the `T=10` command passes.

```bash
MPLCONFIGDIR=/tmp /usr/bin/timeout 3600 /home/chakwong/anaconda3/envs/tf-gpu/bin/python docs/benchmarks/benchmark_ledh_same_target_lgssm_m3_t50_value.py --device-scope visible --expect-device-kind gpu --device /GPU:0 --batch-seeds 81120 --num-particles 10000 --time-steps 50 --history-mode value-only --score-mode manual-reverse --score-diagnostic-stage score-only --transport-policy active-all --transport-gradient-mode manual_streaming_finite_sinkhorn_stopped_scale_keys --transport-ad-mode full --sinkhorn-iterations 10 --sinkhorn-epsilon 0.5 --row-chunk-size 2500 --col-chunk-size 2500 --particle-chunk-size 2500 --dtype float32 --tf32-mode enabled --warmups 0 --repeats 1 --output docs/plans/artifacts/ledh-lgssm-particle2500-score-only-n10000-t50-row2500-col2500-2026-07-10.json
```

## Stop Conditions

- Stop before `T=50` if `T=10` fails, times out, or does not reach
  `score_completed`.
- Stop after `T=50` completes or times out.
- Do not run FD, score admission, five seeds, or leaderboard integration inside
  this ladder.
