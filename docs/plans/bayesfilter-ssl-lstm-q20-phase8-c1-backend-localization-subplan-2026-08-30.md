# Phase 8 C1 principal-square-root backend localization

Date: 2026-08-30  
Parent result:
`docs/plans/bayesfilter-ssl-lstm-q20-phase8-c1-graph-repair-result-2026-08-30.md`  
Status: `PASS_TARGET_LOCALIZATION_ATTEMPT_02`

## Question

Can the reviewed `tensorflow_eigh_strict` principal-square-root backend reduce
the q=20 target cost enough to rescue the small `B=8` graph, compared with the
compiled-custom-op backend used in the prior attempts?

## Evidence contract

The baseline is the same q=20 target and bridge with
`compiled_custom_op`. The prior localization measured finite `B=8` target
calls at 54.843 seconds (`beta=0`) and 45.801 seconds (`beta=0.5`) in a fresh
process. The candidate is the same target, data, beta values, XLA setting,
device, and stateless inputs with only the principal-square-root backend
changed to `tensorflow_eigh_strict`.

The diagnostic passes only if both `B=8` target calls return finite values,
finite analytic scores, valid statuses, and stage timings before the 300-second
cap. It may nominate a follow-up cost plan; it cannot select a transport,
establish whitening, or reopen C2--C5.

Hard vetoes are target-signature drift, invalid/nonfinite rows, failed GPU or
memory-growth setup, wrong backend identity, forbidden row mapping/pfor, or a
timeout before both small calls. A timeout is a backend feasibility failure,
not evidence against the bridge or transport mathematics.

## Scope and budget

One fresh launcher attempt is authorized with a 300-second wall cap and a new
output directory under:

```text
docs/plans/artifacts/ssl-lstm-q20-tempered-rkl-transport-ensemble-2026-08-30/c1-backend-localization/
```

The launcher uses GPU 0, `TF_FORCE_GPU_ALLOW_GROWTH=true` before TensorFlow
import, XLA, and no idle-GPU or Luna probe. The attempt is diagnostic only and
does not consume the closed C1 training allocation or any Phase 9 reserve.

## Skeptical pre-run audit

| Risk | Control |
|---|---|
| Wrong comparison | All target, bridge, beta, input, dtype, device, and XLA settings remain frozen; only the backend changes. |
| Large graph contaminates the question | The cap and stage markers isolate the two `B=8` calls; any later `B=256` start is recorded as uncompleted. |
| Proxy promotion | Timing is a feasibility diagnostic; no quality, whitening, or sampler criterion is changed. |
| Hidden scalar route | The existing localization runner uses static rank-2 `B=8` target batches and its static scan remains active. |
| Missing stop condition | One 300-second attempt, fresh output root, explicit timeout classification. |
| Memory misinterpretation | Pre-import memory growth and the runner's allocator receipt remain hard requirements. |

No material flaw remains in this bounded comparison. The result must be written
before any backend default or campaign-budget change.

## Execution and interpretation

Run:

```text
BAYESFILTER_PHASE8_MODE=target-localization \
BAYESFILTER_PHASE8_TIMEOUT_SECONDS=300 \
BAYESFILTER_PHASE8_PRINCIPAL_SQRT_BACKEND=tensorflow_eigh_strict \
BAYESFILTER_PHASE8_OUTPUT_ROOT=docs/plans/artifacts/ssl-lstm-q20-tempered-rkl-transport-ensemble-2026-08-30/c1-backend-localization \
BAYESFILTER_PHASE8_ATTEMPT_LABEL=attempt-01-eigh-strict \
bash scripts/run_ssl_lstm_q20_tempered_rkl_phase8_gpu_default.sh
```

Record the two localization stage files, timeout or manifest, backend identity,
GPU/memory receipts, target call status, and wall time. A finite small-call
result only nominates a new graph-cost plan; it does not make the backend a
default. A timeout closes this hypothesis and leaves the C1 continuation veto
active.

## Completion

Attempt 1 exposed a result-serialization defect after the target calls. The
focused list-serialization repair passed 42 relevant tests. Attempt 2 then
completed the full localization sequence in `69.98432411497924` seconds with
finite and valid B=8 and B=256 target calls, both transport preflights, the
256-row pullback diagnostic, and one batched optimizer update. The complete
decision and nonclaims are recorded in
`docs/plans/bayesfilter-ssl-lstm-q20-phase8-c1-backend-localization-result-2026-08-30.md`.
