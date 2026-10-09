# State-space continuation with progress-aware allocation

The owner added 48 CPU hours and 48 GPU hours and requested an audited repair
and continuation. This amendment replaces the September 30 stage and attempt
limits. Its numerical baseline remains frozen source
`a8c6c6107c2da78d25a144be28a3503431352681465e338e945fca899f0a0e43`
(commit `de80aaff5812ebfbed551977476c0868551a2c88`). Nine of 32 original main
fits already have final assessments. Preserve those outcomes, including the
exact-reference interval miss, and the five unfinished pilot workloads.

## Research question and evidence contract

Can the original state-space workloads complete when progressing, compatible
checkpoints receive further fair allocations and newly completed pilots unlock
dependent main fits? The comparator is the executed September 30 allocator,
which left 23 main slots incomplete/unstarted despite unspent resources.
Engineering success requires actual frozen-worker recovery beyond three
attempts, stage carryover, dynamic eligibility, unchanged earlier evidence,
and cumulative accounting within the authorized allowance. More completed fits
does not establish a statistical improvement of HMC.

The original target, data, seeds, broad L/epsilon search, candidate retention,
acceptance/health checks, posterior stopping criteria and member-selection rule
remain fixed. K5/K6 use the already identified repaired preparation designs.
R-hat is reporting-only in tuning; posterior assessment remains separate.
Numerical-reference sensitivity is descriptive, not an integration-error bound;
K7 targets the declared sigma-point approximation and K6 has no independent
joint posterior oracle. No default-policy or calibrated-coverage claim follows.

| Observation | Role | Response |
| --- | --- | --- |
| Resource stop with new durable evidence | Repair trigger | Requeue after peers while the global allowance remains |
| Completed favorable or unfavorable assessment | Terminal evidence | Preserve unchanged; never rerun for a favorable result |
| No durable progress, non-resource process failure | Affected-job continuation veto | Retain failure for diagnosis; allow independent jobs to continue |
| Changed source/design/evidence, corrupt checkpoint, missing GPU provenance | Continuation veto | Diagnose before affected execution; unexpected coordinator invariant failure stops the service |
| Posterior/reference failure | Promotion veto | Preserve outcome and continue other eligible work |
| Global deadline or budget exhaustion | Continuation veto | Stop children, settle once, report every original slot |
| Runtime, contention, candidate/observation counts | Explanatory and allocation diagnostics | Refresh workload estimates; never rank scientific candidates |

## Repair and execution

1. Extend the common pool to support deadline-bounded progress retries without
   a fixed attempt count, plus discovery of newly eligible jobs after each turn.
   Keep the existing finite-attempt option for callers that request it. Require
   advancing monotonic time for an unlimited resource retry, preventing a
   broken executor from busy-looping.
2. Build one inventory from the original 32 slots, the September 30 terminal
   result and its saved repair jobs. Resume all seven unfinished existing main
   fits and five pilots. Do not reset stage limits, restart completed fits, or
   relabel original identities. Native checkpoint paths stay fixed because
   receipts and archives bind those paths; new attempts use fresh external
   output directories, save prior mutable checkpoints and verify immutable
   evidence through the existing recovery helper.
3. Refresh pilot prices and original-slot dispositions after each allocation.
   Admit unstarted main fits only with a complete matching development workload,
   independent of favorable/unfavorable posterior outcome. Preserve exact
   source, prepared-data, reference-grid and workload identities. Update full
   observed costs without treating a pilot estimate as a lifetime fit cap.
4. The child receives only the current quantum beyond already charged fit
   work, and permits exactly the next continuation count. The enclosing
   deadline bounds every invocation; no fixed three-attempt cap survives in
   the executor. Use one GPU process at a time with memory growth and existing
   shared-GPU supervision. Contention can consume additional future quanta
   only when progress and the global allowance permit it.
5. Test the dispatcher, real frozen-child boundary, budget settlement and
   terminal reporting. Include analytic/linear-Gaussian and nonlinear sigma-point
   CPU integration fixtures, preserving numerical evidence across continuation.
   Audit source and all existing final assessments before launch. Launch the
   actual GPU continuation from a versioned root, inspect startup and durable
   progress, then leave it supervised by systemd.
6. After every turn write current dispositions, pilot prices, cumulative cost,
   remaining allowance and next eligibility. At close preserve all 32 slots,
   generate interval reports for final assessments, settle the enclosing charge
   once, and refresh master progress. Further localized infrastructure repair
   remains authorized within this unchanged contract and remaining allocation.

## Localized launch repair: exact GPU identity

The first launch exposed an incorrect environment assumption: frozen execution
binds `CUDA_VISIBLE_DEVICES` exactly, so the same GPU model on another UUID is
not a compatible checkpoint environment. All 12 attempted resumptions failed
at execution-binding reconstruction before numerical progress. Every saved
mutable checkpoint and preserved immutable record remains unchanged. The
67.995-second service plus 30-second closeout charge is retained.

Use the original UUID `GPU-4f1220f9-7ba2-21ad-3f9a-b24c2ca4ce91` and add a
framework-free device-binding preflight before any child starts. The recovery
helper may explicitly accept this diagnosed startup failure only when the
failure record is the exact device-policy mismatch, a prior resource-stopped
checkpoint had progress, the failed attempt made no progress, all before/after
checkpoint and immutable evidence hashes agree, no final assessment exists,
and the requested GPU now matches the original execution specification.
Retain every failed receipt and count its cost. Other failures remain
ineligible; no general retry-after-failure option is introduced.

Skeptical repair audit: the target, source, data, seeds, numerical policy and
candidate evidence remain fixed; only the launch environment is corrected.
Test wrong-device rejection before dispatch, the exact startup-repair path,
changed evidence/failure rejection, cumulative costs and completed-outcome
preservation. Resume into a fresh `r2` output under the remainder of the same
172,800-second campaign cap. This is an authorized infrastructure repair, not
new scientific evidence or a reset of the budget.

## Defaults, provenance and budgets

- The additional CPU and GPU grants are each **172,800 seconds**, derived from
  the owner's 48-hour allowance. Carry the previous settled GPU charges exactly
  once into a new ledger: 532,800 granted, 288,787.683 charged and 244,012.317
  remaining before this run. Preserve the unrelated 1,200-second allowance.
  Keep prior ledgers as historical evidence and record which ledger supersedes
  them. CPU follows the existing reference/test worker-hour convention, with
  the thread limits and measured wall time recorded; it is not a core-hour
  measure. GPU accounting includes the entire enclosing service's startup,
  waiting, compilation, failures and shutdown; nested fits are not charged twice.
- This launch has a **48-hour maximum including shutdown**, using at most the
  new GPU allowance; the previously unspent balance is not automatically spent.
  Every attempt under this amendment carries the same campaign ID; subtract
  its settled charges before any retry so the 48-hour limit is cumulative.
  Its new wall deadline starts at launch, explicitly authorized by this request.
  A 30-second shutdown reserve is inherited from the tested launcher. The child
  deadline leaves that reserve inside the service ceiling. Report early if all
  jobs finish or all remaining work is ineligible; never wait merely to spend.
  Settlement charges measured elapsed time plus the conservative 30-second
  closeout reserve once, covering terminal reporting and shutdown overhead.
- Resumed fits use **1,775-second** fairness quanta, inherited from successful
  checkpoint continuations. Fresh main fits get at most **3,600 seconds** for
  first initialization, the previous large-model pilot allocation. K6's saved
  preparation took 2,145 seconds, explaining why a shorter initial allocation
  is inadequate there. These are operational hypotheses, not convergence
  thresholds or guarantees. A fresh fit lacking a resumable checkpoint stops
  for diagnosis rather than silently repeating preparation forever.
- No finite retry-count threshold is a scientific policy. Productive jobs may
  continue until completion or the global ceiling. No-progress checks and
  monotonic elapsed time prevent unlimited empty retries.
- CPU regressions hide GPUs before framework imports and use two-thread limits.
  The Python environment remains `/home/ubuntu/anaconda3/envs/tfgpu/bin/python`.
  GPU launch requires trusted execution and `TF_FORCE_GPU_ALLOW_GROWTH=true`;
  each numerical child must verify GPU/XLA/memory-growth provenance.

Outputs: `docs/plans/artifacts/hmc-ssm-progress-continuation-2026-10-02/`, with
fresh `validation-r1` and `r1` directories. The active entry point remains
`scripts/run_hmc_ssm_pooled_campaign.py`, configured with the prior run and
new ledger. Record exact test/launch commands and coordinator hashes in the
validation and run manifests; the numerical worker snapshot remains unchanged.

## Skeptical audit before implementation

The first repair concept was incomplete: removing the queue cap alone leaves
the child's lifetime/attempt limit, and carrying only main fits leaves missing
pilots unable to unlock 16 slots. Both are included above. Reusing the previous
absolute deadline would prevent launch despite the new grant; the new deadline
is explicit. Copying checkpoints to arbitrary paths risks archive identity
mismatches, so use the tested native-path continuation mechanism with immutable
evidence checks and separate versioned attempt records.

Wrong baselines, favorable-outcome selection, proxy promotion, missing stopping
conditions, unfair initial scheduling, hidden work-price caps and environment
mismatch were checked. Full-pilot prices establish workload eligibility, not
posterior success or per-fit completion guarantees. Negative final assessments
remain terminal, all original slots remain counted, and the queue is fair by
allocated turn rather than scientific score. Preparation time and shared-device
cost may still make some cases unaffordable; that is measured, not assumed away.

Review decision: proceed to implementation and focused tests under this
contract. Before launch, confirm actual dispatcher behavior beyond three
attempts, dynamic pilot eligibility, unchanged evidence, no-progress/deadline
termination, and exactly-once settlement on both success and exception. This
audit does not promote the separately experimental acceptance-uncertainty rule.
