# q=20 72-core process-parallel campaign result

Date: 2026-09-03  
Plan: `docs/plans/bayesfilter-ssl-lstm-q20-72core-process-parallel-plan-2026-09-03.md`  
Full attempt: `docs/plans/artifacts/ssl-lstm-q20-72core-process-parallel-2026-09-03/full/attempt-05/`  
Repair canary: `docs/plans/artifacts/ssl-lstm-q20-72core-process-parallel-2026-09-03/canary/attempt-06/`

## Outcome

The requested staged topology is mechanically valid: `8x4 + 2x8 + 6x4 = 72`
worker cores, with mutually exclusive barriers and disjoint CPU affinity.  The
post-repair canary passed in `585.254545083968` seconds.  The fresh full run
then completed all 48 screen tasks and 26 of 32 selection tasks before the
declared `14,400` second global cap.  Both selection workers were evaluating a
long candidate when the cap stopped them; scope finalization was not reached.

This is a resource/schedule result, not a numerical or scientific result.  It
does not establish that the 72-core schedule can complete the declared grid.
Repeating the same topology, task order, and cap would be non-discriminating;
a cap increase or a different task partition requires a new reviewed plan.

The old controller initially reported the cap as an untyped worker failure and
did not write a complete summary.  That was a harness/artifact defect.  The
localized repair added typed `CAP_STOP_INCOMPLETE` receipts and a partial
coverage summary.  Focused regression and the fresh canary passed after the
repair.  Attempt-05 remains preserved and quarantined; its completed records
are usable only for descriptive timing and diagnostic inspection.

## Reproduction record

The full command was:

```text
/home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/run_ssl_lstm_q20_72core_process_parallel_2026_09_03.py --full
```

The process used Python 3.13.13 in `/home/ubuntu/anaconda3/envs/tfgpu/bin/python`
and TensorFlow 2.20.0.  The preparation child used GPU0 with memory growth;
all staged workers used CPU-only TensorFlow/XLA with
`CUDA_VISIBLE_DEVICES=-1`.  The machine exposed 256 logical CPU IDs.  The
attempt was created at Git commit `2323065da5348fbf3aaabbd712afc2a028ca81a4`
with a dirty worktree.  Attempt-05 predates the timeout-closeout repair; the
repaired source hash recorded by canary attempt-06 is
`358dc5467164066000c5aa591053bbcdbc3455b6fd709437c56f3b26b1154a8e`.

The full attempt's durable counts were:

| Barrier | Expected | Durable | Result |
|---|---:|---:|---|
| Screen (`8x4`) | 48 | 48 | 16 passed, 32 candidate-local hard vetoes |
| Selection (`2x8`) | 32 | 26 | 26 finite task records; 6 were in-flight at cap |
| Scope finalization (`6x4`) | 6 | 0 | not started |

The completed selection calls took 561.6--1315.6 seconds each.  Their short
16-draw diagnostics are descriptive only; the observed R-hat/ESS values are
not convergence evidence and no candidate was promoted.  TensorFlow emitted
retracing warnings during fresh-chart preparation; they are explanatory
performance diagnostics, not a validity pass or failure.

## Decision table

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| 72-core topology | Affinity, worker environment, XLA, identity, and parity | Passed in canary | Full-run throughput under long grid | Keep topology as a diagnostic option | No CPU default or speedup claim |
| Full staged schedule | All declared tasks complete under cap | Resource/cap veto: incomplete selection | Remaining six selection tasks and all finalization | New reviewed schedule/cap plan required | No complete tuning comparison |
| Candidate diagnostics | Finite, typed task records | No global promotion; scope-local screen vetoes remain | Very short chains and missing scopes | Treat as descriptive evidence only | No posterior/HMC conclusion |
| Cap closeout repair | Typed partial receipt and non-overwrite behavior | Passed focused regression and canary | Timeout path not exercised by a long post-repair run | Preserve repair; use typed path next time | No claim that a full run passes |

## Inference status

| Inference class | Status |
|---|---|
| Hard veto screen | Mechanics canary passed. Full run stopped at the declared resource cap; incomplete barrier is a hard schedule veto. |
| Statistically supported ranking | None. No uncertainty analysis or complete candidate set exists. |
| Descriptive-only differences | Per-task wall time, RSS, acceptance, ESS, and R-hat for completed short calls. |
| Default readiness | Not assessed; repository GPU default and Phase 9B block are unchanged. |
| Next evidence needed | A separately reviewed cap increase or finer-grained selection partition, followed by complete held-out scope validation and sequential posterior checks. |

## Red-team closeout

The strongest alternative explanation is a controller/session artifact rather
than true schedule cost.  Attempt-05 reached the same 13/16 selection count on
both independent workers, with matching long-candidate runtimes and healthy
worker receipts; this makes a one-worker crash unlikely, while the old missing
timeout receipt remains a real harness defect.  Evidence that would overturn
the resource conclusion is a complete run under the same target and topology
within the same cap, or a measured equivalent partition that completes without
changing the scientific contract.  The weakest evidence is the extrapolation
from 26 completed selection calls to the uncompleted tail; it is a scheduling
diagnostic, not a theorem about all future implementations.

## Nonclaims

No posterior correctness, convergence, whitening, mode discovery, ergodicity,
sampler superiority, high-dimensional scaling law, HMC readiness, production
readiness, CPU-default promotion, or GPU-speedup claim follows from this
campaign.  The canary is a mechanics fixture, and the full attempt is a
cap-stopped diagnostic.
