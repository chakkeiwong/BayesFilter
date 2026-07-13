# Phase 2R Result: LGSSM Smaller-Chunk Memory Repair

Date: 2026-07-09

Status: `BLOCKED_FIXABLE_PROCEDURAL_MEMORY_REPAIR_REQUIRED`

## Decision Table

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Do not admit LGSSM score from Phase 2R. | Failed: the smaller-chunk trusted retry emitted no score artifact. | Runtime/memory gate still active: trusted GPU/XLA run created device state but did not reach JSON emission in the reviewed window. | Whether the dominant blocker is transport JVP tensor lifetime, score finite-difference recomputation, or score-memory measurement/procedure. | Draft and execute a reviewed Phase 2S repair focused on score-only emission, score-specific memory measurement, value-only FD checks, and then a deeper transport streaming repair only if still required. | No LGSSM score admission; no rejection of compact score mathematics; no non-LGSSM conclusion; no leaderboard completion. |

## Evidence Contract Result

| Field | Result |
| --- | --- |
| Question | Can smaller chunks make the LGSSM compact score run emit an admitted `N=10000,T=50` score artifact within memory budget? |
| Baseline/comparator | Phase 2 interrupted trusted run, admitted LGSSM value artifact, Phase 1 score artifact contract, and Phase 2R smaller-chunk subplan. |
| Primary criterion | Not met. No `phase2r` score JSON artifact exists. |
| Veto diagnostics | No artifact; no validator admission; trusted run did not complete in the reviewed window. |
| Explanatory diagnostics | The log shows TensorFlow GPU device creation, CUDA/XLA initialization, cuDNN load, and XLA compilation. Reducing row/column/particle chunks to 128 did not produce a prompt artifact. |
| Not concluded | This does not prove the compact derivative is wrong. It only blocks admission until the score procedure emits validated evidence. |

## Command Attempted

Trusted GPU retry:

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
  --row-chunk-size 128 \
  --col-chunk-size 128 \
  --particle-chunk-size 128 \
  --score-mode compact-sensitivity \
  --history-mode value-only \
  --dtype float32 \
  --tf32-mode enabled \
  --device /GPU:0 \
  --device-scope visible \
  --expect-device-kind gpu \
  --output docs/plans/bayesfilter-ledh-n10000-score-admission-repair-phase2r-lgssm-score-artifact-2026-07-09.json \
  --markdown-output docs/plans/bayesfilter-ledh-n10000-score-admission-repair-phase2r-lgssm-score-artifact-2026-07-09.md \
  > docs/plans/logs/bayesfilter-ledh-n10000-score-admission-repair-phase2r-lgssm-smaller-chunks-2026-07-09.log 2>&1
```

Outcome:

- Output artifact: absent.
- Log artifact:
  `docs/plans/logs/bayesfilter-ledh-n10000-score-admission-repair-phase2r-lgssm-smaller-chunks-2026-07-09.log`
- Process: trusted run was killed after the reviewed stop window because Ctrl-C
  did not promptly return control.

Log tail confirms:

```text
Created device /job:localhost/replica:0/task:0/device:GPU:0 with 13495 MB memory: NVIDIA GeForce RTX 4080 SUPER
XLA service initialized for platform CUDA
Loaded cuDNN version 91700
Compiled cluster using XLA
```

Artifact existence check:

```text
docs/plans/bayesfilter-ledh-n10000-score-admission-repair-phase2r-lgssm-score-artifact-2026-07-09.json: absent
```

## Root-Cause Update

Phase 2R showed that chunk-size reduction alone is not a sufficient repair.
Additional code inspection found two procedural risks that must be repaired
before another full blind run:

1. The LGSSM runner's compact score diagnostic computes the compact score and
   then performs coordinate-wise finite differences using
   `_compact_value_and_score_from_components` for each plus/minus value. That
   means FD checks recompute the full score/JVP path instead of a cheaper
   value-only scalar, multiplying runtime and tensor pressure before any JSON
   artifact is written.
2. The score artifact memory gate is derived from `gpu_memory_info_after`
   measured around the value route before the score diagnostic runs. This is
   not a score-memory measurement and is insufficient for score admission.

The transport JVP helpers also still assemble full row-block outputs with
TensorArrays and `stack()`. That may be acceptable where full transported
particles/tangents are required, but it remains a candidate implementation
repair only after the score procedure is split and measured correctly.

## Handoff To Phase 2S

Proceed to a dedicated Phase 2S subplan before any code edit or full rerun.
Phase 2S must:

1. add score-specific memory measurement and artifact fields;
2. make same-scalar FD use a value-only forward scalar route, not the full
   score/JVP route;
3. allow a score-only full run to emit a blocked artifact if correctness is not
   yet complete, while requiring full correctness and score-memory pass before
   admission;
4. run small/prefix parity checks before any trusted full `N=10000,T=50` retry;
5. keep the target scalar, admitted value artifact, seeds, `N=10000`, `T=50`,
   parameter order, and admission criteria fixed.
