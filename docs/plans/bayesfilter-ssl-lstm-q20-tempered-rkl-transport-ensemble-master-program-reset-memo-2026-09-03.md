# Master-program reset memo: 72-core schedule closeout

Date: 2026-09-03  
Master: `docs/plans/bayesfilter-ssl-lstm-q20-tempered-rkl-transport-ensemble-master-program-2026-09-02.md`  
Subplan: `docs/plans/bayesfilter-ssl-lstm-q20-72core-process-parallel-plan-2026-09-03.md`  
Terminal result: `docs/plans/bayesfilter-ssl-lstm-q20-72core-process-parallel-result-2026-09-03.md`

## State transition

M3P is now closed as `M3P_FULL_CAP_BLOCKED`.  The requested staged topology
passed its repaired canary, but the fresh full schedule reached the declared
14,400-second cap after 48/48 screen tasks and 26/32 selection tasks.  It did
not reach the six scope-finalization calls.  Phase 9B remains blocked, and no
posterior, whitening, mode-discovery, or sampler-ranking claim is opened.

## Repair history

The full attempt first exposed a controller artifact defect: a planned cap
termination was reported as an untyped worker failure and no summary was
written.  The controller now emits typed `CAP_STOP_INCOMPLETE` receipts with
worker/task coverage.  Nine focused tests and canary attempt-06 passed after
that repair.  The repair changed only timeout/artifact handling; it did not
change the target, bridge, chart, seeds, candidate grid, worker topology,
hardware class, or cap.

## Governing next step

The same topology, task order, and cap must not be relaunched because its
measured long-candidate cost makes an identical attempt non-discriminating.
A future M3P continuation requires a new reviewed subplan that explicitly
chooses a larger resource budget or a finer-grained selection partition while
proving unchanged replication and seed semantics.  That is a material
resource/evidence-contract change and is not authorized by this reset.

## Evidence boundaries

Attempt-05 and all prior attempts remain preserved under the process-parallel
artifact root.  Completed records can describe timing and short-chain
diagnostics, but cannot nominate a candidate or establish convergence.  The
canary is an engineering fixture only.  The repository GPU default and all
earlier M3/M3-C historical boundaries remain unchanged.

## Red-team

Hidden environmental contention is the strongest alternative explanation for
the schedule cap.  Matching progress and runtimes on both independent
selection workers, healthy memory observations, and the clean canary reduce
but do not eliminate that possibility.  A complete same-contract run or a
measured equivalent partition would overturn the resource conclusion.  The
unobserved six-task tail is the weakest evidence and is deliberately not
extrapolated into a scientific claim.
