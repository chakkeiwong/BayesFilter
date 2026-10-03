# 72-core process-parallel terminal reset memo

Date: 2026-09-03  
Plan: `docs/plans/bayesfilter-ssl-lstm-q20-72core-process-parallel-plan-2026-09-03.md`  
Result: `docs/plans/bayesfilter-ssl-lstm-q20-72core-process-parallel-result-2026-09-03.md`

## State transition

The `8x4 + 2x8 + 6x4` topology passed the repaired canary, including CPU
affinity, CUDA hiding, XLA declaration, target/bridge identity, durable
coverage, and serial/process parity.  The full attempt reached the declared
global cap after 48/48 screen tasks and 26/32 selection tasks.  It did not
reach scope finalization.  The active subplan is therefore closed as
`P3_FULL_CAP_BLOCKED`; Phase 9B remains blocked.

## Failure classification

The first terminal symptom was a harness defect: the controller represented a
planned wall-cap termination as `selection worker failure` and omitted a typed
summary.  The repair added `ParallelCampaignDeadline`, durable
`barrier_timeout.json` coverage, and `CAP_STOP_INCOMPLETE` full summaries.  The
repair was verified by nine focused tests and canary attempt-06.  The
underlying full-run limitation is a resource/schedule cap, not evidence that
the target, bridge, or HMC arithmetic is wrong.

## What is preserved

All attempt directories, worker logs, start receipts, completed task records,
preparation manifests, and repair history remain under
`docs/plans/artifacts/ssl-lstm-q20-72core-process-parallel-2026-09-03/`.
Attempt-05 records are quarantined from candidate ranking and posterior claims;
they may support descriptive runtime accounting.  No prior partial task is
reused as a new tuning or confirmation draw.

## Required next decision

Do not relaunch the same 72-core task order and 14,400-second cap: its measured
selection cost makes another identical attempt non-discriminating.  A future
continuation needs a new reviewed plan that chooses either (a) a justified
larger cap and budget, or (b) a finer-grained selection task partition while
proving that the two replication streams and seed semantics remain unchanged.
That choice changes the resource/evidence contract and is outside this repair.

## Red-team

The strongest alternative is hidden environmental contention.  The canary's
clean affinity/parity result, healthy memory observations, and matching
replication progress weaken that explanation but do not eliminate it.  A
complete same-contract run, or an independently measured schedule with the
same target and seeds, would overturn the cap interpretation.  The weakest
part of the evidence is the unobserved tail of six tasks; no scientific claim
is based on extrapolating it.

## Nonclaims

This memo does not certify whitening, mode exploration, posterior correctness,
convergence, sampler ranking, high-dimensional scaling, HMC readiness, CPU
default status, GPU speedup, or Phase 9B admission.
