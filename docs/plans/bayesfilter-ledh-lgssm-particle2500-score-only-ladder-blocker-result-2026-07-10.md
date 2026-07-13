# LEDH LGSSM Particle2500 Score-Only Ladder Blocker Result

date: 2026-07-10
status: BLOCKED

## Decision Table

| Field | Result |
|---|---|
| Decision | Do not continue to `T=50` score with the current route. The `T=10` score-only run reached `score_started` but consumed nearly all GPU memory and did not complete promptly. |
| Primary criterion status | FAIL/BLOCKED: terminal score JSON was not produced; partial JSON remained at `artifact_status=score_started`, `last_completed_stage=score_started`, `score_completed=false`. |
| Veto diagnostic status | VETO: trusted `nvidia-smi` showed about `15.7 GiB / 16.4 GiB` GPU memory in use while the score process was still running. |
| Main uncertainty | The run was manually interrupted/stopped before timeout, so this is a resource-behavior blocker rather than a formal timeout result. |
| Next justified action | Fix the score route before further `N=10000` score runs: avoid full-history storage and avoid forward-mode transport JVP materialization. |
| Not concluded | Score mathematical correctness, FD agreement, full leaderboard readiness, HMC readiness, or nonlinear/SIR feasibility. |

## What Happened

Command launched:

```bash
MPLCONFIGDIR=/tmp /usr/bin/timeout 1800 /home/chakwong/anaconda3/envs/tf-gpu/bin/python docs/benchmarks/benchmark_ledh_same_target_lgssm_m3_t50_value.py --device-scope visible --expect-device-kind gpu --device /GPU:0 --batch-seeds 81120 --num-particles 10000 --time-steps 10 --history-mode value-only --score-mode manual-reverse --score-diagnostic-stage score-only --transport-policy active-all --transport-gradient-mode manual_streaming_finite_sinkhorn_stopped_scale_keys --transport-ad-mode full --sinkhorn-iterations 10 --sinkhorn-epsilon 0.5 --row-chunk-size 2500 --col-chunk-size 2500 --particle-chunk-size 2500 --dtype float32 --tf32-mode enabled --warmups 0 --repeats 1 --output docs/plans/artifacts/ledh-lgssm-particle2500-score-only-n10000-t10-row2500-col2500-2026-07-10.json
```

Partial artifact:

- `docs/plans/artifacts/ledh-lgssm-particle2500-score-only-n10000-t10-row2500-col2500-2026-07-10.json`
- `artifact_status=score_started`
- `last_completed_stage=score_started`
- `score_completed=false`
- `score_mode=manual-reverse`
- `score_diagnostic_stage=score-only`
- `particle_chunk_size=2500`

Observed trusted GPU telemetry while the process was active:

- GPU memory used: approximately `15.7 GiB / 16.4 GiB`.
- GPU utilization: approximately `27%` to `38%`.

The process was then stopped because it was an interrupted diagnostic still
holding nearly all GPU memory.

## Root-Cause Trace

The value route was cheap because it used the streaming value transport and did
not need to propagate parameter tangents. In contrast, the score route currently
does two memory-heavy things.

First, `_manual_value_and_score_from_components` stores a forward history in
TensorArrays for the reverse pass. It records per-time scalar tensors and LEDH
flow tensors, including particle-indexed matrices such as
`post_covariance`, `post_chol`, and `affine_transform` with shape roughly
`[B, N, D, D]`.

Second, the forward transport inside that score route uses
`_filterflow_manual_streaming_finite_transport_total_vjp` with
`transport_ad_mode='full'`. The custom-gradient wrapper is reverse-mode with
respect to TensorFlow, but the forward score route still stores the transport
value history and the reverse pass calls the full pullback. The alternative
compact sensitivity path explicitly materializes transport tangents such as
`d_transported` with shape `[B, N, D, P]`, and the transport JVP helper also
allocates block-pair tangent tensors with shapes involving
`row_chunk_size x col_chunk_size x param_dim`.

For `N=10000`, `row_chunk_size=col_chunk_size=2500`, and `P=5`, one block-pair
transport tangent payload of shape `[1, 2500, 2500, 5]` is about `119 MiB`.
There are multiple such tensors inside each row/column block computation, and
XLA can retain buffers aggressively across the loop. This is qualitatively
different from the value-only path, whose TensorFlow allocator peak was only
about `84 MiB`.

## Interpretation

The `2500/2500/2500` chunk policy fixes LGSSM value-stage feasibility, but it
does not fix the current score route. The score route still has a structural
memory issue: it is not a true streaming reduce-only reverse score recurrence.

This blocker was observed on the active LEDH execution policy: `dtype=float32`
with TensorFlow TF32 execution enabled. The repair must target that route. Do
not treat older float64 LEDH score-memory diagnostics as current
production/admission evidence; see
`docs/plans/bayesfilter-ledh-tf32-only-execution-policy-reset-2026-07-10.md`.

The right fix is not to lower `particle_chunk_size` back to `512`. The right fix
is to change the score algorithm so it avoids:

- full forward-history TensorArrays of all LEDH flow internals;
- forward-mode transport tangent materialization;
- full `transport_ad_mode='full'` requirements when the intended score route is
memory-style reverse accumulation.

## Local Checks

- Code trace inspected:
  - `docs/benchmarks/benchmark_ledh_same_target_lgssm_m3_t50_value.py`
  - `experiments/dpf_implementation/tf_tfp/resampling/annealed_transport_tf.py`
- GPU process was stopped after interruption and GPU memory dropped back to
  approximately `2.2 GiB / 16.4 GiB`.

## Post-Run Red-Team Note

This blocker result should not be read as evidence against LEDH score
mathematics. It is evidence against the current implementation route for
`N=10000` score. The value path already showed that the model and chunks are
feasible when the score-specific tangent/history payload is absent.
