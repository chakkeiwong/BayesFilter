# LEDH LGSSM Particle2500 T10/T50 Value Ladder Result

date: 2026-07-10
status: COMPLETE

## Decision Table

| Field | Result |
|---|---|
| Decision | `particle_chunk_size=2500` remains feasible for one-seed LGSSM value-only GPU/XLA runs at both `T=10` and `T=50` with `N=10000` and `row/col=2500`. |
| Primary criterion status | PASS for both launched commands: each exited 0, wrote terminal JSON, used GPU outputs, reported finite value output, `score_mode=none`, `history_mode=value-only`, `N=10000`, requested `T`, and `row/col/particle=2500`. |
| Veto diagnostic status | PASS: no CPU placement, no score path, no dense transport materialization, no exception, and both reached `value_materialize_completed`. |
| Main uncertainty | One seed and one warm call only; this does not admit the full five-seed leaderboard row or the score path. |
| Next justified action | Use `row/col/particle=2500` for the next LGSSM score-path or five-seed value gating step, rather than reverting to historical `particle_chunk_size=512`. |
| Not concluded | Score readiness, FD agreement, HMC readiness, posterior correctness, full leaderboard admission, or optimal chunk size for nonlinear/SIR models. |

## Results

| Horizon | Compile + first call (s) | Value call (s) | Warm call (s) | TF allocator peak (MiB) | Finite output | Average relative error |
|---:|---:|---:|---:|---:|---|---:|
| 10 | 11.044563086004928 | 11.03466550912708 | 1.001461003208533 | 83.86474609375 | true | 0.008329782508417849 |
| 50 | 14.599952113814652 | 14.592213436029851 | 4.35889402590692 | 83.865234375 | true | 0.0008515580464820113 |

Additional diagnostics:

- `T=10`: `max_row_residual=0.889252245426178`, `max_column_residual=0.0`.
- `T=50`: `max_row_residual=0.963551938533783`, `max_column_residual=0.0`.
- `T=10`: total log likelihood by seed `[-32.31960678100586]`.
- `T=50`: total log likelihood by seed `[-135.96009826660156]`.

## Interpretation

This closes the immediate value-stage concern: the prior `T=50` stall was not an
inherent `N=10000,T=50` value-path blocker once `row/col/particle=2500` is used.
The TensorFlow allocator peak remained around 84 MiB in both runs, reinforcing
that `particle_chunk_size=512` was not needed for LGSSM value memory.

This result uses the active LEDH execution policy: `dtype=float32` with
TensorFlow TF32 execution enabled. Do not replace it with a float64 LEDH
diagnostic for production/admission evidence; see
`docs/plans/bayesfilter-ledh-tf32-only-execution-policy-reset-2026-07-10.md`.

This is still not score evidence. The score path has separate tangent/VJP
lifetime risks and must be tested under its own evidence contract.

## Artifacts

| Artifact | Path |
|---|---|
| Plan | `docs/plans/bayesfilter-ledh-lgssm-particle2500-t10-t50-value-ladder-plan-2026-07-10.md` |
| `T=10` JSON | `docs/plans/artifacts/ledh-lgssm-particle2500-value-ladder-n10000-t10-row2500-col2500-2026-07-10.json` |
| `T=50` JSON | `docs/plans/artifacts/ledh-lgssm-particle2500-value-ladder-n10000-t50-row2500-col2500-2026-07-10.json` |

## Commands Actually Run

```bash
MPLCONFIGDIR=/tmp /usr/bin/timeout 1200 /home/chakwong/anaconda3/envs/tf-gpu/bin/python docs/benchmarks/benchmark_ledh_same_target_lgssm_m3_t50_value.py --device-scope visible --expect-device-kind gpu --device /GPU:0 --batch-seeds 81120 --num-particles 10000 --time-steps 10 --history-mode value-only --score-mode none --transport-policy active-all --transport-gradient-mode manual_streaming_finite_sinkhorn_stopped_scale_keys --transport-ad-mode stabilized --sinkhorn-iterations 10 --sinkhorn-epsilon 0.5 --row-chunk-size 2500 --col-chunk-size 2500 --particle-chunk-size 2500 --dtype float32 --tf32-mode enabled --warmups 0 --repeats 1 --output docs/plans/artifacts/ledh-lgssm-particle2500-value-ladder-n10000-t10-row2500-col2500-2026-07-10.json
```

```bash
MPLCONFIGDIR=/tmp /usr/bin/timeout 2400 /home/chakwong/anaconda3/envs/tf-gpu/bin/python docs/benchmarks/benchmark_ledh_same_target_lgssm_m3_t50_value.py --device-scope visible --expect-device-kind gpu --device /GPU:0 --batch-seeds 81120 --num-particles 10000 --time-steps 50 --history-mode value-only --score-mode none --transport-policy active-all --transport-gradient-mode manual_streaming_finite_sinkhorn_stopped_scale_keys --transport-ad-mode stabilized --sinkhorn-iterations 10 --sinkhorn-epsilon 0.5 --row-chunk-size 2500 --col-chunk-size 2500 --particle-chunk-size 2500 --dtype float32 --tf32-mode enabled --warmups 0 --repeats 1 --output docs/plans/artifacts/ledh-lgssm-particle2500-value-ladder-n10000-t50-row2500-col2500-2026-07-10.json
```

## Local Checks

- `git diff --check -- docs/plans/bayesfilter-ledh-lgssm-particle2500-t10-t50-value-ladder-plan-2026-07-10.md`: passed.
- `git diff --check -- docs/plans/bayesfilter-ledh-lgssm-particle2500-t10-t50-value-ladder-plan-2026-07-10.md docs/plans/bayesfilter-ledh-lgssm-particle2500-t10-t50-value-ladder-result-2026-07-10.md`: passed.

## Post-Run Red-Team Note

The strongest alternative explanation for success is that value-only LGSSM
`D=3` is too easy to expose the score-path issue. That is acceptable for this
phase because the question was only whether larger particle chunks unblock the
LGSSM value path at longer horizons. The next discriminating artifact must test
score mode with the same chunk policy.
