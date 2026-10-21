# Frozen full-model posterior continuation — 14 October 2026

The owner granted another 48 hours and requested continuation. Interpret this
as 172,800 additional aggregate worker-seconds, following this campaign's
existing accounting convention. Reserve 3,600 seconds for implementation
checks and terminal inspection and at most 169,200 seconds for one GPU
continuation including preparation, checkpoint replay and supervision. These
are operational allocations within the owner's ceiling, not convergence or
completion predictions. The previous 1,579.592954 unallocated seconds and held
local reserve are carried separately and are not spent twice.

## Question and evidence contract

Can the seven unfinished fixed-HMC members complete the original posterior
and finite-target reference assessment without repeating completed tuning or
discarding saved sampling work? The comparator is the uninterrupted execution
of the same frozen design and random streams. The engineering criterion is
equivalent saved draws, member identity, seed assignment and posterior
decisions under interruption/resume. The scientific criterion remains each
member's existing posterior and 18-quantity reference screen. All eight
verified members remain in every report, including completed failures.

Use the original `e68cdf1c4` numerical snapshot, full predator–prey SGQF
target/data, prepared starts/geometry, GPU 1, TensorFlow/TFP/XLA, and verified
memory growth. Reconstruct the original lugsail policy and member-index-derived
seeds. NUTS is excluded. The optional newer precision estimator does not enter
this continuation. A new orchestration script may import the immutable
snapshot's checked retained-member loader, posterior controller, policy factory,
model quantities and reference screen; it must not copy numerical algorithms.

| Diagnostic | Role and decision |
| --- | --- |
| Declared posterior and reference checks | Per-member pass/fail; no tuning membership authority |
| Changed source/data/seed/policy, corrupt or missing committed tensors, lost/duplicate member | Continuation veto and implementation/artifact repair trigger |
| Numerical failure confined to one member | Member veto; preserve the result and assess remaining members |
| Allocated aggregate time exhausted | Stop numerical work, retain all checkpoints and incomplete members |
| Shared-memory contention or slower chunks | Explanatory resource observation; no new arbitrary per-chunk timeout |
| Acceptance, elapsed time, ESS/MCSE margins | Descriptive; no comparative ranking or default promotion |

Completion does not establish global exploration, calibrated stopped-interval
coverage, superiority to MAMS, validity on other models/seeds, or scoped release.
N5's broader comparisons and N8 closure remain separate unfinished tasks.

## Implementation and early checks

1. Add a frozen-source posterior continuation driver with a fresh output root.
   Restore the original complete candidate order and exported identities.
   Carry complete posterior/reference decisions unchanged; when a saved terminal
   posterior result lacks a reference report, assess its saved draws. Copy an
   interrupted member's inactive checkpoint store and use the shared posterior
   controller to replay its committed prefix. Start unrun members with the
   original per-member seeds. Repeat interruption must remain recoverable.
2. Before any numerical work, verify source/input hashes, complete inventory,
   stored policy and checkpoint identity. The existing numerical loader and
   durable store remain responsible for their own evidence and tensor checks.
   Ordinary writer locks and checksums are sufficient; no approval tokens or
   extra launch authorities are introduced.
3. Test completed/interrupted/unstarted mixtures, exact native prefix replay,
   repeated interruption, missing/corrupt evidence, changed policy/seeds,
   terminal failed-member carry-forward, and timeout inventory preservation.
   CPU-only mechanics tests deliberately hide GPUs; a bounded trusted GPU
   preflight checks the actual immutable-source reload before long execution.
4. Record saved-price scenarios and launch the seven-member remainder under
   the single 169,200-second ceiling. Preserve the 1,000-transition chunk sizes
   and all scientific count limits. Emit heartbeats and committed-chunk
   progress separately. A long chunk is not killed for inactivity; only the
   total allocation is a time stop. Compilation/replay and failed attempts
   consume the allocation. A hard stop can lose only uncommitted numerical work.
5. Audit terminal source/input identities, complete candidate inventory,
   posterior/reference outcomes, carried prefixes and actual elapsed time.
   Settle the ledger and refresh the master from the resulting evidence.

Versioned artifacts live under
`docs/plans/artifacts/modern-hmc-implementation-2026-10-07/N5/generic-adaptation/priced-comparison-01/`:
`resume-preparation-01/` for implementation checks, source/price manifests and
`comparator-continuation-01/` for the detached numerical attempt. Every retry
uses a new suffix and preserves previous evidence. Exact launched commands,
orchestration checksum/Git revision, numerical source revision, environment,
physical device, memory policy, data identity, seeds and timing are saved.

## Default audit, pre-mortem and skeptical review

The posterior thresholds, count limits, four-chain starts, 1,000-step chunks,
epsilon/L settings and precision estimator are inherited from the frozen
comparison design. Keeping them is necessary for this continuation's identity;
their broader calibration remains unproved. The earliest identity check is
exact policy and seed equality with the saved member policy. The complete
reference checks are the downstream scientific assessment, not the replay test.
Two CPU TensorFlow thread settings are inherited resource-sharing choices;
performance claims cannot follow from them. The 3,600-second local reservation
is a conservative operational hypothesis; record actual charges. No new
scientific numerical threshold or algorithmic protection is introduced.

Potential misleading success: carrying a completed member under a new seed
index, silently retuning, mixing current precision code into the frozen source,
or treating heartbeat messages as new numerical progress. Test all four
mechanisms before launch. Potential misleading failure: rerunning a whole member
after a timeout, truncating the cohort, or replacing uncertainty failure with
resource failure. Preserve explicit dispositions and replay only the saved
prefix. The completed first member is not a statistically selected winner;
the other seven remain mandatory assessments.

Skeptical audit passes for this bounded continuation: the same target and
uninterrupted numerical program are the baselines; replay mechanics cannot
promote posterior validity; candidate failure is not a research-direction
veto; total time, invalid inputs and shared numerical failure have explicit
stop conditions. Immutable-source imports resolve the environment/version
hazard. The existing interface registry remains unchanged. The run can be
incomplete even if every command succeeds; its full inventory and saved draws
answer what actually happened. Implementation and focused tests precede launch.

Implementation review found two interruption hazards before execution: a saved
posterior result can precede its reference decision, and a deleted committed
chunk can otherwise look like unstarted work. The driver now reconstructs an
unfinished reference decision from saved draws and rejects gaps or missing
committed receipts before numerical calls. Regression checks cover both. The
driver calls the shared retained-member loader, `run_hmc_posterior` and
`DurableTensorCheckpoint`; exact uninterrupted/resumed Gaussian draws test that
call chain. The actual-model preflight must resolve these imports inside the
original source snapshot and load all eight members. This checks engineering
compatibility, not posterior validity.

The supervisor's 30-second heartbeat and 60-second worker closeout margin are
operational convenience choices. The single inclusive wall-time ceiling is
authoritative; no heartbeat or per-chunk speed threshold stops numerical work.
CPU mechanics checks explicitly use `CUDA_VISIBLE_DEVICES=-1` and non-XLA tiny
fixtures. They do not provide GPU scientific or performance evidence.

Execution checks: 42 focused continuation/member/consumer tests passed in
405.771 seconds, followed by six supplemental device/inventory/finalizer tests
in 3.072 seconds. The original 1,540 run files remain byte-identical to the
terminal audit. The new driver also checks the actual physical GPU UUID and
the source location of all imported numerical authorities. GPU 1 remains the
original device with approximately 26 GiB free at the trusted capacity check;
the existing shared-device policy applies. Busy-device readings are not a veto.
The prior 4,096 MiB free-memory admission screen is an inherited resource
hypothesis, not a guaranteed peak-memory bound; verified memory growth and
actual runtime errors remain authoritative.

The next preflight uses the frozen driver in `resume-preparation-01/source`,
the original numerical source, `--preflight-only` and a 1,200-second total cap.
This operational allowance is roughly 3.6 times the previous 334-second
full-model preparation price and is charged to the local reserve. It loads
every retained member without sampling. The production continuation uses the
same command without that switch, a fresh `comparator-continuation-01` output,
and the 169,200-second inclusive cap. Both set `CUDA_VISIBLE_DEVICES=1`,
`TF_FORCE_GPU_ALLOW_GROWTH=true` and two intra/inter-op/OMP threads before import,
using `/home/ubuntu/anaconda3/envs/tfgpu/bin/python` under trusted execution.

The actual-source GPU preflight passed: all eight members loaded, all saved
policies/seeds matched, all numerical imports resolved inside the immutable
snapshot, and no sampling was performed. Inclusive supervisor time was
216.728 seconds. Two additional fixed-trajectory tests passed in 29.852 seconds,
including repeated interruption with exact draw equality. Saved 64-transition
prices give 25.139-hour and 42.084-hour sampling-only scenarios at 12,000 and
20,000 total transitions per member respectively, subtracting the 7,000 saved
transitions for the interrupted member. These omit reload/compilation and
diagnostic costs and assume unchanged per-transition speed. They justify a
bounded continuation, not a promise of completion.

Pre-launch skeptical review passes: no criterion, cohort, target, seed,
algorithm, estimator or numerical source changed; the real saved-member loader
and repeated-interruption tests answer the relevant compatibility risks. The
idle-only GPU probe rejected shared occupancy, while the recorded trusted
inventory and existing shared-device policy establish resource admission with
verified memory growth. There is no claim of exclusive capacity. At terminal
exit, inspect cohort dispositions and new committed chunks, verify original
hashes again, and charge inclusive elapsed time before allocating further work.

## Launched continuation

The detached service `bayesfilter-modern-fixed-resume-20261014-01.service`
started on 14 October at 17:45 China time. Its parent manifest starts at
09:45:36 UTC and records a 169,200-second inclusive numerical allocation, giving
an approximate deadline of 16 October 16:45 China time. Its output root is
`comparator-continuation-01`; launch command, environment, GPU inventory, source
hashes and budget receipts are in `resume-preparation-01/launch-01.json`,
`launch-01-receipt.json`, and `source.json`. Preparation and replay count toward
this allocation. The measured local checks cost 657.246937 seconds; keep the
entire 3,600-second local reserve held for short inspection overhead and closeout.
The old settlement remains separate. GPU memory growth is verified before
initialization. Completion and scientific outcomes remain pending.

The startup audit at 17:59 China time verified all 1,540 original run files,
182 copied checkpoint files, completed-member carry-forward, unchanged policy
and seed assignments, all numerical import locations and GPU memory growth.
Seven chunks replayed; the first new chunk was executing. The audit cost
1.117468 seconds from the local reserve and is saved as
`comparator-continuation-01/startup-audit-01.json`.

At 18:03 China time the first new 1,000-transition chunk committed after
283.345 seconds of numerical work. Its bundle and 24 tensor checksums pass;
the second member now has eight committed chunks. The separate
`first-new-chunk-audit-01.json` records this observed progress. The cohort is
still running, and further member outcomes remain unassessed.

Continuation inspection at 22:49 China time found five delivered members, one
terminal precision-limited member, the seventh sampling and the eighth queued.
The second member has no hard numerical veto: its source-coordinate second
moment `second_5` misses both the declared lugsail precision screen and the
separate uncertainty-adjusted reference screen. Keep this terminal outcome;
do not retune, change sample caps, or replace the original estimator. The next
bounded check reads the committed checkpoint prefix and completed reports,
verifies original hashes, identities, seeds and reported decisions, and records
an interim snapshot without touching the live worker. This is a standard-library
artifact consistency audit, not numerical recomputation or new scientific
evidence. It charges measured execution time to the existing local reserve.
Skeptical audit: the snapshot can age during sampling, so it must report its
capture time and incomplete denominator; final cost settlement and scientific
interpretation still require the terminal cohort.

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Execute frozen continuation | Engineering replay and actual-source reload pass | No preflight identity or numerical veto | Changing workload and remaining posterior count requirements | Check replay, new chunks and terminal member dispositions | Complete cohort, superiority or release readiness |

| Inference status | Current evidence |
| --- | --- |
| Hard veto screen | Original source, member inventory and policy checks pass; runtime checks remain active |
| Statistically supported ranking | None |
| Descriptive-only differences | Saved timing scenarios; original completed-member diagnostics |
| Default-readiness | Not established; no numerical default changed |
| Next evidence needed | Complete retained cohort and reference assessments, then broader paired-model/seed work in the master |
