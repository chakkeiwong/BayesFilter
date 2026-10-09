# State-space pilot recovery and bootstrap diagnosis

The owner asks to repair the September 29 stop and diagnose its numerical
failures. The frozen shared-device run passed eight mechanics and four public
preflight cells, then exhausted six pilot allowances and failed K5/K6 bootstrap.
That was the repair baseline. Main execution ended at 23:23:42 Shanghai on
September 29: three K4 fits completed with declared posterior checks passed,
four K0 fits exhausted their allocations during search, one funded K4 fit
missed the original latest-start cutoff and 24 slots remained unfunded. The
18,893.277-second enclosing cost is settled once. Existing numerical evidence
and all 32 original slots stay in the record; validation is incomplete. No
worker remains or is queued. C1 is complete and will not run again. The audit is in
`artifacts/hmc-ssm-pilot-repair-2026-09-29/result.md`.

## Question and evidence contract

Can a resource-interrupted complete fit resume its exact numerical computation
using explicitly reassigned campaign time, and can the existing finite startup
initializer resolve the K5/K6 bootstrap failures? The baseline is frozen source
`a8c6c6107c2da78d25a144be28a3503431352681465e338e945fca899f0a0e43`
under `artifacts/hmc-shared-gpu-recovery-2026-09-29/prepared-r5/`.

Engineering success requires unchanged source/design/seed identities during
resume; preservation of completed tuning chunks and member results; no rerun
of a completed unfavorable assessment; cumulative attempt accounting; and
enforcement of the original outer deadlines and grant. Complete-fit pricing
requires the entire declared search and member workload with retained draws.
Partial work, short preflight and a nominated startup step cannot satisfy it.

K5/K6 diagnosis uses the already implemented optional
`bootstrap_initialization_rounds=20`, followed by the ordinary public
preparation. Twenty is the existing documented finite engineering hypothesis,
not a calibrated default. The failed baseline used the default zero. Preserve
the same data, target, starts and tuning seed for this comparison. A repaired
initialization has its own configuration identity and fresh directory; it
cannot resume a differently configured failed bootstrap. This diagnostic may
justify a repaired development pilot but cannot silently change the original
main design or establish posterior correctness.

| Diagnostic | Role |
| --- | --- |
| Exact identities, preserved completed evidence, correct cumulative charges and bounded attempts | Engineering pass criteria |
| Invalid initial/retained value or score, corrupt checkpoint, source mismatch, unexplained worker failure | Stop affected work and diagnose |
| Nonfinite rejected bootstrap proposal | Initializer repair trigger, never an accepted state or valid handoff |
| Clean resource stop with committed progress and no final assessment | Eligibility for additional campaign allocation |
| Posterior readiness/precision/reference failure | Promotion veto; never retry a completed assessment to improve its outcome |
| Grant, cumulative SSM cap, additional recovery pool, attempt cap and original wall deadlines | Continuation limits |
| Runtime, contention, candidate counts and bootstrap acceptance | Explanatory; startup acceptance only nominates preparation |

## Repair and execution

1. Add a framework-free campaign continuation helper. It validates the frozen
   numerical source and immutable original design, resumes via the original
   child CLI, and writes a new process-attempt receipt. Additional allocation
   is recorded separately from numerical identity. All earlier elapsed time
   remains charged; the per-fit cumulative cap is explicitly enlarged rather
   than reset. Preserve old receipts and use a fresh enclosing launch directory.
2. Test exhausted local budgets, later resource recovery, completed negative
   assessments, numerical-failure exclusion, source/config mismatch, pending
   launches, restart accounting, process cleanup and outer limits. Include real
   K0 and K7 CPU subprocess checkpoint recovery; these are mechanics tests with
   GPUs deliberately hidden, not GPU throughput or posterior evidence.
3. Run bounded K5/K6 GPU/XLA preparation diagnostics on the original frozen
   numerical source, with verified memory growth. Compare preserved baseline
   failure details with the optional initializer's proposal and retained-state
   checks. A nonfinite retained/initial state or runtime exception stops that
   lane. Do not relax assertions or candidate/posterior thresholds.
4. Continue eligible resource-stopped pilots using the frozen numerical source.
   Reuse complete preparation, tuning and member results. At most two further
   process attempts per original pilot are available. Each next attempt needs
   committed progress and another explicit allocation from the common pool.
   Record unchanged seeds and chunk checksums, plus any uncompleted disposition.
5. Reconstruct pricing from complete cumulative fit costs, including original
   and continuation startup, numerical work and shutdown. Preserve the original
   failed pricing index; publish a continuation inventory rather than rewriting
   historical outcomes. Recompute affordability before any main launch. A
   startup-repaired pilot is a separate configuration, never a price for the
   unmodified baseline. Refresh the master and result note after the phase.

## Budget and assumptions

Use at most six additional GPU hours for this repair, reassigned from the
unspent main allowance inside the existing 36-hour SSM ceiling. This is an
explicit scheduling allocation using the plan's existing six-hour repair
scale; it is not an estimate of required runtime. The pool includes all new
diagnostics, waiting, retries and overhead. Intersect it with settled grant
balance, cumulative SSM balance and the original compute deadline before launch.
No new work after September 29 23:03:52 Shanghai; compute stops September 30
03:03:52 and reporting is due at 05:03:52. These times are unchanged.

Each continuation initially receives up to 1,775 seconds, the measured exhausted
pilot interval (1,175 nominal plus 600 extension). This is a scheduling quantum,
not a complete-fit price or a new scientific default. Unused allocation remains
in the common pool. A second quantum requires durable progress; no more than
two new attempts may be launched for a given original pilot. K0 and K7 are
visited first as the declared linear/nonlinear sibling-mechanism cases, then
K1--K4 in their original order. This ordering is not a performance ranking.
K5/K6 preparation diagnostics each have the existing shared public-preflight
ceiling of 350 seconds. All nested costs are included in one enclosing charge.

Implementation and CPU regressions have a conservative 1,800 worker-second
allocation from the recorded 39,438.430 CPU-second balance. Use
`/home/ubuntu/anaconda3/envs/tfgpu/bin/python`; set `CUDA_VISIBLE_DEVICES=-1`
for CPU tests. New results go under
`docs/plans/artifacts/hmc-ssm-pilot-repair-2026-09-29/`. Launch manifests record
Git/source identities, commands, environment, datasets/seeds, device and memory
policy, elapsed time, plan and results. The existing additive grant ledger
remains authoritative; no already charged fit is charged again.

## Skeptical audit before implementation

The initial proposal to increase per-fit time inside the original design would
break its strict identity and lose valid checkpoint reuse. The repair therefore
keeps the numerical worker/source/design frozen and changes only the explicitly
recorded outer allocation. A new worker receipt must validate final assessment
hashes and GPU/XLA/growth provenance before completion is reported. Ordinary
checksums and launch receipts are sufficient in this trusted workspace.

The six timeouts do not show that contention is the only runtime cost; broad
search, compilation and multiple posterior assessments also matter. Reusing a
partial fit must not make its price appear cheap by counting only the last
attempt. The two bootstrap failures are not resource retries. The initializer
must preserve rejected-proposal diagnostics and still pass fresh preparation;
it cannot turn a nonfinite proposal into an accepted tuning candidate.

The plan passes with these corrections. The principal remaining risk is that
complete prices still exceed remaining wall time. That outcome limits the
funded main inventory but does not delete any of the original 32 reporting
slots or reject the state-space research direction. No stochastic ranking or
default promotion will be inferred from these few development runs.

## First diagnostic finding and bounded follow-up

The first K5 GPU diagnostic passed startup after shrinking epsilon from
0.4204482076 to 0.0525560260, then failed ordinary mass adaptation on one
nonfinite log-acceptance value. The whole diagnostic took 178.049 seconds.
This shows that repairing startup alone is insufficient. The next K5 diagnostic
also enables the already documented `preparation_max_restarts=3`. Its checked
implementation permits a restart only for a rejected nonfinite proposal after
independent retained-state/score/status and acceptance-consistency checks. It
discards the failed preparation statistics, contracts the consumed step ceiling,
and uses fresh recorded streams. An unclassified or retained-state failure is
still fatal. This is a development hypothesis, not a default change.

If K6's first diagnostic reaches its 350-second resource cap after successful
startup, its one follow-up keeps the same numerical configuration and gets the
existing 1,775-second scheduling quantum. If it instead completes, reuse it;
if it fails mass adaptation, the same explicitly recorded three-restart
hypothesis applies. No other unexplained failure permits this follow-up.
Each of these at-most-two follow-ups has a 1,775-second cap inside the same
six-hour pool. This bound accommodates full preparation and records compilation
cost; it is not evidence that the algorithm needs that long. Review passes:
the initializer, preparation retry and resource continuation have different
eligibility rules, and none can promote a bad state or waive a numerical check.

The affordability audit also finds that the original 2,100-second K0--K4
per-fit scheduling hypotheses are already below the inherited 1.5 margin times
their incomplete 1,775-second cost. That comparison does not make the target
scientifically invalid. After complete prices exist, the next phase must review
explicit reallocation of the remaining main pool using those costs; it must not
silently pass an incomplete price or claim the old lane ceilings were measured.
The global 22-hour main, 36-hour SSM and original wall ceilings remain fixed.

K6's longer same-config diagnostic ended after 724.814 seconds on two nonfinite
log-acceptance entries during mass adaptation (first at index 27 of the failed
segment). It thus needs the numerical warmup repair that its earlier timeout
could not diagnose. Add exactly one K6 diagnostic with the existing
three-restart option and the same 1,775-second cap, charged to the unchanged
six-hour pool. K5's one-restart successful preparation supplies mechanism
evidence, not cross-model proof; K6 must pass its own checks. A further cap or
unclassified failure is recorded as unresolved, without an automatic extra
diagnostic or changed criteria. This bounded follow-up passes audit because it
targets the newly observed failure, has unchanged validity checks and funding,
and does not substitute K5 evidence for K6 evidence.

The framework-free sequence launcher
`scripts/continue_hmc_ssm_after_diagnosis.py` waits for the K6 diagnostic service
to terminate and its coordinator to settle, then invokes the tested six-pilot
continuation automatically. Its dispatch tests cover an active predecessor,
successful settlement with a classified target failure, and refusal after an
unclassified coordinator failure. The observer initializes no GPU framework;
the diagnostic's enclosing receipt already accounts for its numerical time.
Its outer 23,405-second ceiling is derived as 1,775 diagnostic seconds plus the
entire 21,600-second repair pool plus 30 shutdown seconds. That ceiling grants
no additional compute: the continuation independently subtracts all settled
diagnostic costs and applies the original grant and wall limits.

## Closeout race found in the live diagnostic

The third diagnostic reached its service limit after 1,775.494 seconds. K6 had
discarded two preparation attempts (315 and 450 transitions) and had not
completed its third attempt. The service sent SIGTERM before the nested fit
supervisor could write its timeout receipt. The coordinator therefore recorded
`SystemExit(143)` and the sequence correctly refused its nonzero result. The
enclosing receipt was settled once; K6 remains unresolved, and neither that
receipt nor the failed sequence will be rewritten as success.

The repair is to give numerical work an earlier absolute monotonic deadline
and reserve 30 seconds inside the service allowance for child reaping,
validation and receipt settlement. Thirty seconds is the existing shutdown
allowance reused as a conservative engineering reserve, not a measured sampler
requirement. Passing the deadline from the launcher also charges startup delay
against available work. The outer service cap and shutdown allowance remain
inside the existing reservation. A real CPU subprocess regression must show a
resource timeout settling normally before the external guard; launch tests
must check the nested deadline. Unknown failures must still block automatic
dispatch. After these checks, launch the six independent continuations directly
in their unused versioned output directory; do not rerun K6 or reset its clock.

Skeptical audit: merely treating every exit 143 as a normal timeout would hide
unrelated termination and is rejected. The earlier work deadline fixes the
observed race without changing numerical source, outcomes, attempts, budget or
posterior thresholds. The bounded K6 result remains incomplete; it does not
veto recovery of separate valid numerical checkpoints. The patch and direct
recovery pass this audit subject to the focused closeout regression.

## Final nonlinear pilot allocation and main handoff

The six-pilot continuation settled successfully after 14,937.356 seconds.
K0, K2 and K4 completed their entire declared workloads, with cumulative prices
2,241.074, 4,841.006 and 2,533.406 seconds. K1/K3 still have 29/28 pending
search tasks; K7 has 12 pending tasks and one interrupted task, 13 verified
candidates, and no posterior assessment yet. No completed assessment is retried.
All three incomplete cases made durable progress; these are resource failures,
not stalled processes or demonstrated posterior failures.

At 17:33 Shanghai the six-hour repair pool had 3,332.905 seconds remaining.
Reassign at most that remainder, including closeout and shutdown, to exactly
one additional K7 continuation. This explicitly raises K7's additional-attempt
cap from two to three; K1/K3 retain two. The reason is the predeclared nonlinear
two-member coverage obligation and its smaller unfinished task inventory,
not its posterior outcome. This is the final repair-pool allocation: no fourth
K7 continuation or new K6 diagnostic. Use unchanged source, seeds and numerical
criteria. A completed K7 workload supplies a complete cumulative price even if
its posterior checks fail. An incomplete workload remains unpriced.

Audit: the earlier cap was an engineering allocation, not a validity condition.
The measured unused pool and completed receipts support one explicit revised
allocation without adding time. More candidate passes cannot substitute for
finishing the original search and both posterior assessments. Preserve original
and new costs, terminal failures and all 32 main slots. The new-work and compute
deadlines remain absolute. A regression must check that only K7 receives this
third attempt and that its cumulative cap includes both prior continuations.

In parallel with that numerical work, prepare the framework-free main allocator.
It must consume checked complete cumulative pilot prices, retain the original
32-fit inventory, and explicitly replace the obsolete per-lane 2,100-second
convenience ceilings with funding from the remaining global main pool. Keep
the inherited 1.5 planning margin as an uncalibrated allocation hypothesis,
not a runtime-tail guarantee. Consider complete four-fit lanes in original
case order, subject to the remaining 22-hour main allocation, 36-hour SSM cap,
settled grant, and original wall deadline. Within funded lanes, interleave
the original first dataset/seed slots across cases, then the remaining original
slots. The launcher checks the new-work cutoff before every cell. Any unstarted
slot remains explicitly unavailable in the original denominator. Preparation,
broad candidate search, member selection, warmup, retained counts, reference
checks and all seeds remain the original main design; only resource fields
and scheduling metadata change.

Skeptical audit of the main handoff: interrupted pilot tails alone are not
prices; changed numerical workloads cannot borrow an old price; complete
posterior failures remain eligible cost observations rather than selection
exclusions. A single pilot per case cannot justify calibrated runtime tails or
general posterior coverage. Test workload binding, cumulative costs, partial
rejection, lane allocation, denominator preservation, latest-start enforcement,
GPU provenance and terminal receipt verification before launch. The useful
output is execution and descriptive reference agreement for funded original
slots, with honest unavailable outcomes for the rest. It is not a 32/32 result.

The implemented launcher is `scripts/run_hmc_ssm_priced_main.py`. Its optional
`--after-service`/`--after-result` pair waits for normal K7 coordinator settlement,
then recomputes funding from the actual ledger and remaining clock. It starts
the unchanged frozen CLI separately for each funded original main slot.
Per-cell launcher/reporting overhead has a separately reserved 30-second cap,
using the existing shutdown scale as an engineering hypothesis. Unused actual
time stays unused or available inside the same funded stage; no later lane is
added based on posterior outcomes. At 8.5 remaining compute hours, whole-lane
case-order allocation funds the four original K0 slots and four K4 slots for
28,892 seconds including their cell overhead. This is a preview; launch uses
the live balance after K7 settles. K2's complete four-slot lane cannot fit
alongside K0 under that allowance. All unallocated or unstarted slots remain
in the 32-slot report.

An input audit found that K3's second original dataset uses reference-grid
resolution 161 whereas its pilot uses 321. The main allocator records this
workload mismatch and cannot use that pilot to price the whole lane. It does
not substitute a smaller grid or reinterpret the pilot as an exact timing
measurement for both workloads. K3 is already incomplete. K0/K2/K4 match their
pilot workloads except for the intentionally frozen data/seed repetitions.

The focused main/allocation suite passes 56 checks. It includes unknown-failure
dispatch refusal, shared-ledger accounting, latest-start enforcement, native
receipt/source/GPU/XLA/growth checks, preservation of unfavorable outcomes,
and the complete worker's command/settlement path. The first fixture omitted
`isolate_fits`; the worker roundtrip then exposed and fixed a real comparison
bug caused by JSON converting tuples to lists. The corrected comparison uses
the existing canonical JSON digest, with the roundtrip regression retained.
These are engineering checks; GPU posterior evidence still requires the actual
main runs. Audit passes with the workload exclusion and explicit overhead.
