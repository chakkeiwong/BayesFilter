# LEDH LGSSM XLA Value-Stage Instrumentation Result

## Decision Table

| Field | Result |
|---|---|
| Decision | Stage instrumentation is working and the full-shape wait localizes inside the first compiled value call, before host materialization and before score. |
| Primary criterion status | Passed: tiny trusted GPU smoke emitted all value markers and completed. |
| Veto diagnostic status | No score/leaderboard claim made; CPU regression passed; GPU runs used trusted execution. |
| Main uncertainty | Whether the first full-shape call is spending all remaining time in compiled loop execution, or whether there is later hidden XLA work after the first `Compiled cluster` log. |
| Next justified action | Re-run the bounded value-stage diagnostic with the prior N10000 chunk-sizing policy, especially `row_chunk_size=col_chunk_size=2500`, before drawing runtime conclusions from the `128`-chunk timeout. |
| Not concluded | No score correctness, same-scalar FD correctness, leaderboard admission, posterior correctness, HMC readiness, or runtime ranking. |

## Commands And Results

CPU regression:

```bash
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp \
python -m pytest tests/test_ledh_lgssm_manual_score_phase4.py -q
```

Result: `12 passed, 2 warnings`.

Tiny trusted GPU marker smoke:

```bash
python docs/benchmarks/benchmark_ledh_same_target_lgssm_m3_t50_value.py \
  --batch-seeds 81120 --num-particles 256 --time-steps 3 \
  --transport-policy active-all --sinkhorn-iterations 2 \
  --sinkhorn-epsilon 0.5 \
  --transport-gradient-mode manual_streaming_finite_sinkhorn_stopped_scale_keys \
  --transport-ad-mode full --row-chunk-size 128 --col-chunk-size 128 \
  --particle-chunk-size 128 --history-mode value-only --score-mode none \
  --warmups 0 --repeats 1 --dtype float32 --tf32-mode enabled \
  --device /GPU:0 --device-scope visible --expect-device-kind gpu \
  --output docs/plans/artifacts/ledh-lgssm-xla-stage-marker-smoke-n256-t3-s2-2026-07-10.json \
  --markdown-output docs/plans/artifacts/ledh-lgssm-xla-stage-marker-smoke-n256-t3-s2-2026-07-10.md
```

Result: completed on GPU.  It emitted all markers:
`value_call_started`, `value_call_returned`, `value_materialize_started`,
`value_materialize_completed`.

Small-N T ladder, all `N=256`, `S=2`, row/col chunks `128`:

| T | value call seconds | materialize seconds | warm call seconds |
|---:|---:|---:|---:|
| 1 | 8.405173191102222 | 0.0036787588614970446 | 0.01741289207711816 |
| 3 | 8.434305171947926 | 0.0030835370998829603 | 0.05805846699513495 |
| 10 | 9.028688888065517 | 0.003052724990993738 | 0.18359913700260222 |
| 20 | 9.257218755083159 | 0.0029116219375282526 | 0.34576715086586773 |

Interpretation: at small `N`, the first-call number is mostly fixed setup/XLA
cost plus a smaller execution component, while warm execution scales with `T`.
Materialization is negligible for these runs.

Full-shape trusted GPU marker probe:

```bash
timeout 180s python docs/benchmarks/benchmark_ledh_same_target_lgssm_m3_t50_value.py \
  --batch-seeds 81120 --num-particles 10000 --time-steps 50 \
  --transport-policy active-all --sinkhorn-iterations 10 \
  --sinkhorn-epsilon 0.5 \
  --transport-gradient-mode manual_streaming_finite_sinkhorn_stopped_scale_keys \
  --transport-ad-mode full --row-chunk-size 128 --col-chunk-size 128 \
  --particle-chunk-size 128 --history-mode value-only --score-mode none \
  --warmups 0 --repeats 1 --dtype float32 --tf32-mode enabled \
  --device /GPU:0 --device-scope visible --expect-device-kind gpu \
  --output docs/plans/artifacts/ledh-lgssm-xla-stage-marker-fullshape-n10000-t50-s10-timeout180-2026-07-10.json
```

Result: timed out with exit `124`.  The terminal log showed XLA service
initialization and `Compiled cluster using XLA!`; the JSON remained at
`artifact_status=value_call_started`, with only `value_call_started` in
`value_stage_markers_emitted`.  Therefore the call did not reach
`value_call_returned`, and did not reach materialization or score.

An initial attempt used `--repeats 0`, which the runner correctly rejected with
`ValueError: warmups must be nonnegative and repeats must be positive`; that
attempt is procedural only and not XLA evidence.

## Current Interpretation

The current evidence does not support a Python block-pair loop hypothesis for
the active LGSSM streaming benchmark.  The hot path uses TensorFlow loops.  The
full-shape blocker is before `value_call_returned`, and the process had already
logged XLA compilation.  However, the `row=col=128` timeout should be treated as
a chunk-policy diagnostic, not as a fair full-row runtime estimate.  The `128`
chunk size was inherited from a score-memory repair attempt, while the prior
N10000 XLA chunk-sizing work identified exact large chunks such as `2500 x 2500`
as the relevant throughput setting.

For `N=10000` and chunk size `128`, there are `ceil(10000/128)=79` row blocks
and 79 column blocks, or 6241 block pairs per softmin-like pass.  With chunk
`2500`, there are exactly 4 row blocks and 4 column blocks, or 16 block pairs,
with no padding.  The total pair slots are similar, but the TensorFlow loop and
kernel schedule is radically different.

## Next Step

Run a bounded execution-cost ladder using the prior chunk-sizing evidence:

1. `N=10000,T=1,S=10`, chunks `2500/2500/512`, to test the prior exact tiling
   policy on the LGSSM value-stage marker route.
2. If needed, compare `2048`, `2500`, and `3334`, matching the June 24
   chunk-sizing plan rather than the July 9 score-memory fallback.
3. Only after that, extrapolate toward `T=50`; do not rerun full score until
   the value wall is understood.

Prior chunk-sizing anchors:

- `docs/plans/bayesfilter-ledh-pfpf-ot-autodiff-free-adjoint-n10000-xla-chunk2500-plan-2026-06-24.md`
- `docs/plans/bayesfilter-highdim-zhao-cui-p82-phase7r-xla-chunk2500-actual-gradient-remediation-result-2026-06-24.md`
