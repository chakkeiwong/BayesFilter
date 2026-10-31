# Sequential lifecycle progress compatibility

The proposed outer lifecycle executes refinement and terminal fitting inside
one XLA call. Its fixed tensor history preserves completed event values and
order, but delivering that history after execution changes callback timing.
The public outer loop is still unchanged while this boundary is qualified.

The comparison against `cfbc32d2` checks full endpoint records, refinement
histories, movement fields, target-call order, and progress event contents.
Runs 01739--01746 pass eight actual CPU/GPU cases before the shared gather
repair. The subsequent regression matrix must establish the same records after
that repair. These tests do not establish live callback interruptibility.

Inspection of real consumers found two consequential uses of progress:

| Consumer | Current behavior | Integration requirement |
| --- | --- | --- |
| MacroFinance `scripts/run_ccma_full_partition_center_sweep.py:464` | `progress()` checks elapsed time against the 1,800-second cap and raises `TimeoutError`; `main()` executes the workload in its own process. | Enforce the existing wall cap from a parent process before enabling buffered outer progress. Preserve typed invalid closeout on termination. |
| MacroFinance `scripts/run_two_currency_double_zlb_dz5_center_first_continuation.py:693` | The parent polls a semantic sequence and calls `base.evaluate_progress_supervisor`; that function terminates when no progress arrives within its declared interval. | Treat an enclosing compiled call as an explicitly bounded phase. Do not interpret buffered historical events as evidence that the numerical worker is currently advancing. Preserve an independent termination bound. |

Paths in the table are relative to
`/home/ubuntu/workspace/MacroFinance-dz5-neutra`. The second supervisor is defined
in `scripts/run_two_currency_double_zlb_dz5_hierarchical_initializer.py:171`.
Its total/stage elapsed guards are reporting-only; its no-progress condition is
the actual termination condition. A synthetic heartbeat must not masquerade as
semantic progress or silently disable that condition.

The full-partition runner also binds exact historical/current source hashes in
`validate_coupled_runtime_lineage()` at line 154. Updating those hashes alone
would not qualify the new execution or callback semantics. Preserve historical
artifacts and validate a new integration against the exact intended target.
No external source or hash has been edited during this review.

For the public API, describe outer progress as buffered, identify its delivery
mode explicitly, and issue a real start event before the compiled call. Keep
the existing event contents/order for completed work. Callback exceptions can
propagate during delivery, but cannot interrupt an already running compiled
call. The consumer must use a parent-process deadline for that purpose. Do not
add a hidden eager fallback to recover live callbacks.

| Decision | Primary criterion | Veto | Main uncertainty | Next action | Nonclaim |
| --- | --- | --- | --- | --- | --- |
| Continue native lifecycle qualification; defer public switch | Same numerical recurrence and complete records | Changed numerical records, missing event, unbounded external execution | Actual external enclosing compile time and parent policy | Finish numerical/resource/cost checks, then qualify the public adapter and independently bounded external consumer | Existing callback-driven timeouts are not qualified for buffered progress |

This is the primary agent's source review, not an independent review. The
strongest alternative interpretation is that all callers use callbacks only
for logging; the inspected timeout and supervisor disprove that assumption.
The weakest remaining evidence is the external call chain: a source audit
does not replace an executed timeout/staleness test and actual DZ5 integration.
The campaign's numerical test workers already have independent parent-process
deadlines, so these findings do not prevent the authorized local qualification.
