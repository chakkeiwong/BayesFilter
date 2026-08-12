# SSL-LSTM q=20 NeuTra target-validity recovery plan (2026-08-06)

Status: `PASS_FOR_FOCUSED_IMPLEMENTATION_AND_CANARY`

## Research intent ledger

| Field | Frozen intent |
|---|---|
| Main question | Can a target-invalid NeuTra proposal batch be diagnosed exactly and handled without either applying a biased optimizer update or crashing the campaign? |
| Mechanism under test | Preserve per-row SSL-LSTM UKF status through the CPU/XLA pool, archive invalid attempted batches, skip their optimizer update, and retry with a deterministically fresh latent batch. |
| Expected failure mode | A transported tail proposal enters an invalid UKF covariance region; repeated retries may show that the current transport is numerically unusable. |
| Promotion criterion | None. A canary may establish only that the recovery harness behaves as specified. |
| Promotion veto | Any target-invalid event remains a hard numerical veto for the affected trained candidate, even if a later replacement batch permits training to continue. |
| Continuation veto | Non-finite trainer state, optimizer mutation on a rejected batch, corrupt/missing failure artifact, worker/infrastructure failure, retry exhaustion, or device/resource contract failure. |
| Repair trigger | A preserved row status identifies the failing UKF condition and supports a separate numerical or proposal-support repair. |
| Explanatory diagnostics | Raw value/score finite masks, status code, floor counts, row class, valid count, minimum innovation eigenvalue, worker/shard/PID, latent `z`, and transported `theta`. |
| Must not be concluded | Ignoring invalid rows is valid; the reverse-KL objective is repaired; NeuTra converged; the candidate is HMC-ready; or the SSL-LSTM target is scientifically valid. |

## Evidence contract

- Question: whether target-invalid batches can be preserved and recovered from without changing the update objective or losing terminal artifacts.
- Comparator: the current route, which converts target-invalid rows to NaN and aborts the process pool at its finite assertion.
- Pass criterion: a synthetic invalid-target test returns raw status without aborting the pool; the runner writes an exact failure event, leaves trainer/Adam state unchanged, uses a deterministic fresh batch, and either succeeds within the retry bound or writes `TARGET_VALIDITY_RECOVERY_EXHAUSTED` plus `result.json`.
- Vetoes: any partial-row update, masked/drop-row loss, reuse of an invalid batch, state change before successful target admission, or missing exact proposals/status makes the repair invalid.
- Explanatory only: the frequency and specific UKF status observed in a short canary do not estimate production failure probability.
- Nonclaim: successful error handling does not make a candidate with a target-invalid event promotable.
- Artifacts: versioned diagnostics under `docs/plans/artifacts/ssl-lstm-q20-neutra-target-validity-recovery-2026-08-06/`; historical continuation `r1` is never overwritten.

## Frozen recovery protocol

1. The batch-native target exposes raw value, score, and all row-level validity fields. Its legacy `batch_value_and_score` remains fail-closed by returning NaN for invalid rows.
2. The CPU pool adds an explicit status-preserving evaluation method. It serializes raw values, scores, and status tensors and does not abort merely because target status is invalid. Infrastructure exceptions still abort the pool.
3. The runner calls only the status-preserving method for training and evaluation. A batch is admitted only if every row has status zero, `valid_pre_regularized_score=true`, and finite value/score.
4. Before any optimizer call, an invalid attempted batch is written with continuation update, attempt index, deterministic seed fold, every `z` and `theta` row, row status/diagnostics, shard boundaries, worker indices, assigned CPUs, and PIDs.
5. The trainer state hash and optimizer step are checked unchanged after rejection. No row is dropped, masked, regularized, clipped into validity, or assigned an artificial target value.
6. Retry attempts use deterministic seed folds derived from the original training seed, update, and attempt. At most three fresh-batch retries are allowed after the initial attempt. The count `3` is a convenience-bounded diagnostic hypothesis, not evidence that three is statistically sufficient.
7. Any target-invalid event permanently records a promotion veto for this candidate. A later admitted batch may continue engineering diagnosis/training but cannot erase the event.
8. Retry exhaustion produces the controlled status `TARGET_VALIDITY_RECOVERY_EXHAUSTED`, preserves the latest finite checkpoint, and writes terminal result/manifest/progress artifacts.

## Default and numeric assumption audit

| Choice | Provenance/status | Justification | Failure mode | Early diagnostic |
|---|---|---|---|---|
| Whole-batch rejection | Derived from the reverse-KL estimator; reviewed requirement | Keeps each optimizer update an IID full-batch estimate from the declared proposal | Repeated invalidity may select for unusually benign batches | Preserve event as promotion veto and report retry count; do not claim objective validity |
| Three fresh retries | Convenience-bounded diagnostic hypothesis | Separates transient rare-tail events from persistent failure without unbounded compute | Too few for a rare stable process or too many for selection bias | Synthetic exhaustion test and explicit nonclaim |
| Exact raw status | Existing target status API, expanded | Identifies whether floors, row classification, valid-count, eigenvalue, or finite computation failed | Upstream diagnostics may still be insufficient to localize time step | Preserve exact proposal for targeted replay and extend upstream trace only if needed |
| Seed-A checkpoint 500 canary | Measured finite state; warm start | It is the last archived finite state before the observed failure | Failure may occur only after many later updates | Canary is mechanics evidence only; later bounded continuation is separate |
| GPU 1 and CPU/XLA 25x4 | Existing reviewed campaign topology | Holds hardware and batch route fixed | Resource contention can mimic failure | Existing device, affinity, XLA, and memory-growth gates remain active |

## Skeptical plan audit

- Wrong baseline: the engineering canary uses the last finite seed-A checkpoint, not the crashed process state or seed B result.
- Proxy promotion: recovery success is mechanics evidence only; any target-invalid event is a promotion veto.
- Missing stop conditions: retry exhaustion, state mutation, artifact failure, infrastructure error, non-finite trainer state, and campaign wall cap are explicit stops.
- Objective drift: dropping invalid rows, zeroing scores, flooring the target, or applying an update from a partial batch is forbidden.
- Hidden assumption: fresh-batch retry conditions on target validity and therefore may bias continued optimization. This is why recovery is diagnostic/engineering continuation and cannot promote the candidate.
- Stale context: serious NeuTra training remains GPU/XLA; target generation remains batched multicore CPU/XLA.
- Environment mismatch: the existing GPU visibility/memory-growth and worker CPU-affinity checks stay binding.
- Artifact adequacy: exact `z`, `theta`, raw target/status, attempt seed, and worker metadata can reproduce and classify a failure; if upstream status lacks the failing time step, the artifact triggers a smaller replay instrumentation repair.

Verdict: `PASS_FOR_FOCUSED_IMPLEMENTATION_AND_CANARY`. The plan repairs error handling and observability without claiming that conditional retries repair the training objective.

## Pre-mortem

The harness could appear repaired while silently updating on valid rows only; state-before/state-after assertions and synthetic mixed-validity tests prevent that. It could archive NaN JSON illegally; finite masks plus JSON-safe nullable raw values prevent artifact loss. It could retry the same batch; the exact folded seed and proposal arrays are compared. It could classify a worker crash as target invalidity; only a completed status-bearing response is recoverable, while worker exceptions remain infrastructure vetoes.

## Planned checks

1. Target unit test: invalid rows expose raw diagnostics while the legacy value/score call remains fail-closed.
2. Pool unit test: a synthetic status-bearing target returns mixed validity without worker abort and remains usable for a second request.
3. Runner unit tests: invalid batch writes exact JSON, performs no update, retries deterministically, and exhaustion writes a controlled terminal result.
4. CPU/XLA synthetic diagnostic: exercise the real process boundary without GPU.
5. GPU-1 canary from seed-A checkpoint 500, with memory growth and GPU 0 hidden, only after focused tests pass.

## Decision table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Harness canary passes | Exact status/artifact/no-update/retry behavior passes | Candidate remains promotion-vetoed if invalidity occurs | Frequency and UKF root cause | Replay exact failed row, then decide numerical/support repair | Objective validity or HMC readiness |
| Retries exhaust | Controlled result and finite checkpoint preserved | Continuation veto | Persistent transport/target incompatibility | Diagnose archived row before more training | Failure of NeuTra as a general method |

## Inference-status table

| Evidence class | Status |
|---|---|
| Hard veto screen | Target invalidity and state/artifact integrity are claim-bearing vetoes |
| Statistically supported ranking | None |
| Descriptive-only differences | Retry frequency, row location, eigenvalue/floor/status magnitudes |
| Default-readiness | Not evaluated |
| Next evidence needed | Exact-row UKF replay, root-cause repair, fresh candidate training, then downstream sequential HMC validation |

