# Phase 2 Blocker Result: LGSSM Full Score Run Runtime/Memory

Date: 2026-07-09

Status: `BLOCKED_FIXABLE_RUNTIME_MEMORY_REPAIR_REQUIRED`

## Decision Table

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Do not admit LGSSM score from the attempted full run. | Failed: no score artifact was emitted, and the run was interrupted after prolonged execution. | Memory veto warning: trusted `nvidia-smi` polling showed about 15.7 GiB GPU memory use, above the 14 GiB admission budget recorded in previous score-memory gates. | Whether smaller chunks are enough, or whether the streaming transport value+JVP still materializes too much state for the full score diagnostic. | Run a reviewed Phase 2R repair: first a smaller-chunk trusted rerun, then if needed implement a true reduce-only streaming score path that avoids stacking row-block tensors. | No LGSSM score admission, no leaderboard score completion, no evidence against compact-score mathematics, no non-LGSSM conclusion. |

## Evidence Contract Result

| Field | Result |
| --- | --- |
| Question | Can LGSSM produce a schema-valid compact `N=10000,T=50` score artifact admitted by the shared score validator? |
| Baseline/comparator | Admitted LGSSM value artifact, July 6 raw compact score-memory JSON as non-admitted legacy evidence, and Phase 1 shared emitter/validator tests. |
| Primary criterion | Not met. No JSON score artifact was produced by the attempted full rerun. |
| Veto diagnostics | Trusted GPU memory polling showed approximately `15703-15795 MiB` used during the run, above the 14 GiB memory budget. The process ran about 18 minutes without emitting the target artifact and was interrupted by Codex. |
| Explanatory diagnostics | The log shows GPU creation, XLA compilation, and interruption inside TensorFlow slicing/TensorArray work in `experiments/dpf_implementation/tf_tfp/resampling/annealed_transport_tf.py` around the streaming softmin value+JVP row-block loop. |
| Not concluded | The compact score formula is not rejected; this is a runtime/memory artifact blocker for the selected command/chunks. |

## Command Attempted

Trusted GPU command:

```bash
MPLCONFIGDIR=/tmp python docs/benchmarks/benchmark_ledh_same_target_lgssm_m3_t50_value.py \
  --num-particles 10000 \
  --time-steps 50 \
  --batch-seeds 81120,81121,81122,81123,81124 \
  --transport-policy active-all \
  --sinkhorn-iterations 10 \
  --sinkhorn-epsilon 0.5 \
  --transport-ad-mode full \
  --transport-gradient-mode manual_streaming_finite_sinkhorn_stopped_scale_keys \
  --row-chunk-size 512 \
  --col-chunk-size 512 \
  --particle-chunk-size 256 \
  --score-mode compact-sensitivity \
  --history-mode value-only \
  --dtype float32 \
  --tf32-mode enabled \
  --device /GPU:0 \
  --device-scope visible \
  --expect-device-kind gpu \
  --output docs/plans/bayesfilter-ledh-n10000-score-admission-repair-phase2-lgssm-score-artifact-2026-07-09.json \
  --markdown-output docs/plans/bayesfilter-ledh-n10000-score-admission-repair-phase2-lgssm-score-artifact-2026-07-09.md \
  > docs/plans/logs/bayesfilter-ledh-n10000-score-admission-repair-phase2-lgssm-run-2026-07-09.log 2>&1
```

Outcome:

- Exit code: `130` after Codex interrupt.
- Output artifact: not present.
- Log artifact:
  `docs/plans/logs/bayesfilter-ledh-n10000-score-admission-repair-phase2-lgssm-run-2026-07-09.log`

Trusted status observations:

- GPU device created: NVIDIA GeForce RTX 4080 SUPER.
- XLA service initialized for CUDA and compiled a cluster.
- Live GPU memory observations:
  - `15742 MiB / 16376 MiB`;
  - `15703 MiB / 16376 MiB`;
  - `15704 MiB / 16376 MiB`;
  - `15795 MiB / 16376 MiB`.

## Root-Cause Hypothesis

The selected full command reached the streaming transport value+JVP path, but
the implementation still stores row-block values and tangents in TensorArrays
and stacks them before returning. The interruption stack pointed into
`annealed_transport_tf.py` row-body TensorArray work around:

```text
value_ta.write(block_index, result_block)
tangent_ta.write(block_index, tangent_block)
stacked_value = value_blocks.stack()
stacked_tangent = tangent_blocks.stack()
```

This suggests the full diagnostic may be chunked but not sufficiently
memory-reducing for `N=10000` with the selected row/column chunks.

## Boundary Notes

- The run was trusted GPU execution; non-escalated CUDA sandbox evidence is not
  being used.
- No score artifact was admitted.
- The old July 6 raw LGSSM `primary_pass=True` JSON remains non-admitted.
- This blocker is fixable under the runbook: reduce chunks first, then repair
  streaming accumulation if needed.

## Handoff To Phase 2R

Proceed to a dedicated repair subplan before any retry. The repair should:

1. run a trusted smaller-chunk retry with predeclared memory budget and timeout;
2. if still too slow or over budget, implement a true reduce-only streaming
   score diagnostic path that avoids stacking all row-block value/tangent
   tensors when only scalar log-likelihood and score are required;
3. preserve the same target scalar, same admitted value artifact, same seeds,
   `N=10000`, and `T=50`;
4. require validator admission before any leaderboard integration.
