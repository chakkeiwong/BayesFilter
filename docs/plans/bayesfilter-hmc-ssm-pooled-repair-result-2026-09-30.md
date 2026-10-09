# State-space pooled repair: execution and terminal audit

## October 2 terminal audit

The campaign ended normally at **October 1, 05:17:19 Shanghai**, after
80,398.171 charged seconds (22.333 hours). The service is inactive, with no main
PID, `Result=success` and exit code zero. This is coordinator completion:
**9 of the original 32 fits have final assessments; 23 remain incomplete or
unstarted.** The final JSON explicitly records `all_original_slots_assessed=false`.
The historical running checkpoints below are superseded by this audit.

| Original main slots | Complete | Remaining disposition |
| --- | ---: | --- |
| K0 | 1/4 | Three unfinished fits excluded from the main queue after the repair stage |
| K2 | 0/4 | All four reached the three-attempt limit while still progressing |
| K4 | 4/4 | Complete, including three preserved prior outcomes |
| K7 | 4/4 | Complete |
| K1, K3, K5, K6 | 0/16 | Unpriced: five required pilot workloads remain incomplete |

### Confirmed allocation limitations

The repair pool attempted nine jobs and completed one. Its eight unfinished
jobs all recorded additional numerical evidence, but the six-hour stage ended
before another quantum could be allocated. Unfinished pilots do not enter the
main queue; unfinished original K0 fits are explicitly classified as
`repair_stage_incomplete` rather than carried forward. As a result, no later
main-pool allocation can complete those jobs or unlock their dependent slots.

Each K2 main fit received three attempts and approximately 8,392 seconds of
additional runtime. The final attempts advanced evidence counts from
132 to 156, 122 to 152, 125 to 144, and 92 to 113. Their latest recorded progress
was approximately 7--60 seconds before termination. They stopped at an outer
deadline or cumulative fit cap, not a recorded no-progress or crash condition.
The common pool then classified them as `incomplete_attempt_limit`.

These outcomes follow the written stage and attempt limits. The limitation is
in the allocation design relative to completing the workload; it is not an
unexplained coordinator failure. The checked current copies of
`scripts/run_hmc_ssm_pooled_campaign.py` and
`bayesfilter/testing/inference_validation/campaign_pool.py` match their hashes
in the executed manifest. Shared-GPU contention was recorded, but its scheduling
allowance does not measure lost compute and cannot explain all runtime causally.

The settled grant ledger records 71,212.317 seconds remaining (19.781 hours).
After preserving the separate 1,200-second allowance, **70,012.317 seconds
(19.448 hours)** were left unspent by this campaign. Exhausted stage/attempt
eligibility emptied the queue despite that balance. Unspent compute does not
renew an expired calendar deadline or establish that all remaining work fits.

### Saved numerical evidence

The four K2 checkpoints contain 100, 100, 78 and 56 candidate records, with
1, 2, 6 and 2 verified candidate IDs respectively. All four remain
`partial_budget`; their internal work budgets were not exhausted. Across 565
observations, all evidence-validity fields are valid, with no recorded hard
veto or engineering-invalidity reason. There is one movement promotion veto.
There are 248 inconclusive-evidence observations and 18 conflict observations,
alongside 228 directional epsilon-repair decisions and 71 acceptance passes.
This establishes substantial unfinished search work and usable candidates at
the checkpoints. It does not establish posterior readiness for those candidates,
a causal tuner defect, or grounds to truncate the broad candidate search.
R-hat remains reporting-only in these tuning records.

The nine complete fits contain **14 assessed members**, plus 97 members left
unassessed under the predeclared member-selection design. All 14 assessed
members pass the recorded posterior checks, have no posterior hard veto, exclude
warmup correctly, contain no duplicate chains, and pass the declared descriptive
reference tolerances. None hit a warmup or retained-sampling cap. These are nine
fits, not 14 independent replications, and unassessed siblings gain no posterior
claim from another member's result.

The separate stopped-interval reports preserve one K0 median-interval miss:
three of four exact-reference comparisons cover their reference. Across K4/K7,
46 of 48 numerical-reference comparisons contain the empirical sensitivity
interval and two overlap it partially; none is disjoint. Numerical comparisons
have exact coverage marked unavailable and must not be counted as failed exact
coverage from their placeholder `covered=false` field. Sensitivity intervals
are not rigorous integration error bounds. One exact miss is insufficient to
establish systematic MCSE miscalibration, and these results do not establish
sequential coverage. K7 targets its declared sigma-point approximation; K6 still
has no independent joint posterior oracle.

The separate [acceptance-uncertainty calibration](bayesfilter-acceptance-uncertainty-validation-result-2026-10-01.md)
failed statistical promotion under strong dependence. That remains an open
admission-policy research problem. The experimental diagnostic did not govern
this frozen campaign; its failure is not evidence that it caused these timeouts.

### Decision and next justified repair

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Preserve completed fits | Nine final assessments; 14 members pass declared checks | No recorded posterior hard veto; one exact interval miss retained | Few independent fits and limited reference accuracy | Reuse unchanged results in the full 32-slot report | Universal correctness or calibrated interval coverage |
| Repair unfinished-work allocation | Durable progress is present; queue stopped with unused balance | No recorded crash/no-progress stop in the two pools | Cost to finish broad search and missing pilots | Carry unfinished pilots/fits across stages, allocate further quanta under cumulative limits, and reprice from observed work | A larger timeout alone fixes the entire campaign |
| Keep statistical-policy work separate | Experimental uncertainty calibration failed promotion | Strong-dependence false conflicts remain | Which uncertainty procedure is valid for the intended estimand | Follow the existing uncertainty repair plan with fresh validation | New acceptance thresholds are justified by these timeouts |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | No recorded posterior hard veto in 14 assessed members; one K2 observation has a movement promotion veto |
| Statistically supported ranking | None |
| Descriptive-only differences | Runtime, candidate counts, interval/reference agreement and differences among members |
| Default-readiness | All-model and interval-calibration claims remain open |
| Next evidence needed | Complete missing workloads under repaired allocation; preserve misses and assess calibration with adequate independent replications and appropriate references |

A revised allocator should use one dependency-aware queue, carrying incomplete
pilots and fits across phases. It should retain fairness quanta, cumulative
accounting, absolute deadlines and no-progress stops, while allowing recorded
progress to use the remaining shared allowance. Completion of a pilot should
refresh eligibility and workload prices during the run. Regression tests must
cover progress beyond three attempts, stage carryover, dynamic pilot completion,
unchanged checkpoint evidence, preservation of unfavorable final results, and
termination at the enclosing budget/deadline. Resume compatible frozen
checkpoints in fresh output directories; do not rerun completed fits to remove
unfavorable observations or change acceptance/posterior criteria.

This is a read-only saved-result diagnosis plus documentation refresh, with no
new sampling or GPU launch. Evidence comes from `r1/result.json`, `status.json`,
`execution.json`, `manifest.json`, each referenced independent assessment and
interval report, K2 tuning checkpoints, and the settled
`hmc-ssm-funded-2026-09-28/grant-ledger.json`. The original run manifest preserves
the numerical source, command, environment, seeds and device provenance.
Post-run red-team: the audit supports an allocation repair but does not yet
prove that remaining work is affordable, that contention dominates cost, or
that statistical tuning admission is calibrated. A failed numerical invariant
or corrupted checkpoint would veto the affected continuation; the observed
allocation stops do not reject the sampler or research direction.

## Historical checkpoints

**September 30 21:16 Shanghai update:** the service is still running, now on
`K2-K2-data-B-s0`, with recent durable checkpoint progress. **Seven of 32 fits
are complete**: K0 data-A seed slot 0, all four K4 fits and two K7 fits. The new
K7 data-B seed slot 0 assessment passed the declared posterior checks and
descriptive reference tolerances for both selected members; one needed 2,000
retained transitions per chain and the other 1,000. Three interrupted K2/K7
fits retain progress and are queued for automatic continuation. Approximately
98,863 seconds (27.462 hours) remain under the unchanged enclosing cap. Source
and coordinator checks still pass. The [latest audit](artifacts/hmc-ssm-pooled-repair-2026-09-30/monitor-20260930-r3/audit.json)
retains the K0 exact-reference interval miss and now reports 30 containing and
two partially overlapping numerical-reference intervals; no coverage calibration
or statistical ranking is established. No campaign failure or settlement has
been recorded.

As of **September 30 18:10 Shanghai**, the service remains active in its main
phase, currently running `K2-K2-data-A-s1`. **Six of 32 original main fits are
complete**: one recovered K0 fit, all four K4 fits, and `K7-K7-data-A-s1`.
The completed members passed their declared posterior checks and descriptive
reference tolerances. These are eight assessed members across six fits, not
eight independent replications. The [mid-run audit](artifacts/hmc-ssm-pooled-repair-2026-09-30/monitor-20260930-r2/audit.json)
verifies the numerical source, unchanged coordinator files, final receipt and
assessment bindings, and cumulative costs. The first K2/K7 main fits exhausted
their first quanta with progress and remain queued for automatic recovery after
peers; they have not been declared numerical failures.

The repair stage closed after 20,706.031 seconds, within its six-hour ceiling.
K0 data-A seed slot 0 completed after two additional turns. Its two predeclared
members each used 2,000 warmup and 1,000 retained transitions per chain. K0
data-A seed slot 1 completed tuning and began posterior sampling before its
stage allocation ended. Three K0 main fits and the five unfinished pilots remain
explicitly incomplete; this stage did not reset their caps for main execution.

K5 and K6 now have successful preparation records, taking 300.316 and 2,145.228
seconds respectively. Both reached actual tuning. K5 has two verified pairs;
K6 has none yet. This resolves the observed preparation interruption, while
complete tuned workloads, full pricing and posterior evidence remain missing.
K1 and K3 data-A pilots each reached 100 candidate records but are still tuning;
K3 data-B also remains incomplete. The 16 main slots in these four cases remain
unpriced. No new numerical settings or campaign budget were introduced during
this monitoring turn.

The current completed K0 fit has four exact-reference interval comparisons,
three covering the reference and one median interval missing it. That miss is
retained; it does not alone establish miscalibration or justify retrying a
completed fit. Across the completed K4/K7 fits, 22 of 24 numerical-reference
comparisons contain the empirical sensitivity interval and two overlap it
partially. These comparisons remain descriptive, with no exact or sequential
coverage claim and no ranking of candidates. K7's reference is for its declared
sigma-point approximate target; K6 still lacks a joint posterior oracle.

The service has approximately 109,967 seconds (30.546 hours) remaining under
its enclosing cap at this checkpoint. The grant balance remains reserved,
not settled; the original end ceiling below is unchanged. The audit launched
no GPU work and did not alter completed results. Its first console summary
needed conversion of a diagnostic integer to JSON; the saved audit and interval
reports were valid, and the corrected reporting command completed in a fresh
directory. This was a local reporting issue, not a campaign-worker failure.

The [plan](bayesfilter-hmc-ssm-pooled-repair-plan-2026-09-30.md) has been reviewed
and implemented. CPU regressions and saved-result checks pass; the official
tuning chapter and agent reference have been updated. The bounded GPU
continuation launched in a fresh `r1` directory at **September 30 06:57:20
Shanghai**. It is not a completed
campaign or an all-model validation result.

Service: `bayesfilter-hmc-ssm-pooled-20260930-r1.service`.
The first job is the original `K0-K0-data-A-s0` checkpoint continuation. Its new
native child manifest confirms the unchanged source/design, GPU execution,
XLA enabled, and memory growth configured before logical-device initialization.
The log confirms XLA compilation. The launch audit confirmed new numerical
progress beyond the original 94 observations and unchanged earlier evidence;
see `r1/launch-audit.json` for counts, hashes and timestamp.
Live status is `artifacts/hmc-ssm-pooled-repair-2026-09-30/r1/status.json`.

The enclosing maximum, including conservative shutdown allowance, is
150,410.488 seconds (41.781 hours), ending no later than approximately
**October 2 00:44:11 Shanghai**. This is a ceiling, not a completion forecast.
The old 1,200-second unrelated reservation remains protected. The current
reservation is charged once on settlement; it is not already spent. The
additional CPU validation charge is 900 conservative worker-seconds, leaving
36,738.430 CPU seconds.

Actual launch, with trusted execution:

```sh
/home/ubuntu/anaconda3/envs/tfgpu/bin/python scripts/run_hmc_ssm_pooled_campaign.py docs/plans/artifacts/hmc-ssm-pooled-repair-2026-09-30/r1 --gpu GPU-4f1220f9-7ba2-21ad-3f9a-b24c2ca4ce91 --service bayesfilter-hmc-ssm-pooled-20260930-r1
```

`r1-launch.json` records the complete systemd command, environment, process-group
shutdown policy and runtime cap. `r1/manifest.json` records source hashes and
copied coordinator sources. Numerical receipts remain under original frozen
fit paths; fresh continuation directories preserve their previous checkpoints.

Active output root:
`artifacts/hmc-ssm-pooled-repair-2026-09-30/`.
The frozen numerical source identity remains
`a8c6c6107c2da78d25a144be28a3503431352681465e338e945fca899f0a0e43`.
The current coordinator and diagnostic sources are recorded separately; the
old checkpoints are never moved into a different numerical implementation.

## Repairs and skeptical review

The common pool dispatches individual fits, calls the actual frozen-worker
continuation after a local timeout, and revisits progressing fits after peers.
It records each attempt and cumulative allocation. A complete unfavorable
assessment remains final. A missing checkpoint, no progress, invalid evidence
or exhausted attempt allowance is reported explicitly. The enclosing service
and grant ledger bound spending and shutdown; nested fits are not charged twice.

The coordinator first gives the nine repair/pricing jobs bounded turns, then
refreshes matching complete costs and allocates original main slots across
models. K7 is first in the main round-robin order. Unpriced lanes do not suppress
eligible lanes. The preview currently admits nine additional main slots from
K2/K4/K7 before any repairs succeed; all 32 original slots remain visible.
K5/K6 need completed pilots with the repaired preparation options. K3 data-B
gets its own price for its reference resolution.

Review found and fixed three draft defects: final negative assessments were
mistakenly tied to complete-workload pricing; pricing lacked full provenance
checks; and JSON lists versus Python tuples prevented equivalent designs from
matching. Tests now separate finality from price eligibility, check source and
assessment hashes, and normalize design content while rejecting genuinely
different reference workloads. Complete cumulative pilot costs include original
outer startup costs and all continuation attempts. No posterior outcome enters
cost selection. Queue closure is explicitly separate from assessment of all
32 original slots.

The reference repair compares saved final MCSE intervals with checked numerical
reference sensitivity. It rejects missing median probabilities, unsupported
quantities, nonfinite values, invalid precision, excessive sensitivity and
failed edge-mass checks. It keeps these comparisons outside exact-truth coverage.
New reports do not overwrite or alter original assessments. The same reporting
operation is run on final campaign assessments outside the frozen worker.

The strongest remaining alternative explanation is that K0's expansion or K6's
preparation is intrinsically too costly under the declared budgets. Scheduling
repairs cannot establish convergence or make a difficult workload affordable.
New nonfinite evidence, identity mismatches or missing required provenance would
invalidate the affected execution and trigger diagnosis. Repeated progress with
resource exhaustion instead measures an unresolved cost requirement. Neither
case licenses weaker numerical criteria.

## Saved-result diagnosis

The reproducible read-only diagnostic is
`artifacts/hmc-ssm-pooled-repair-2026-09-30/validation-r1/inspect_saved_results.py`;
its result is `validation-r1/saved-diagnosis.json` under the same root.

| K0 workload | Candidates | Candidate families | Directional-repair children | Verified pairs | Measurement chunk seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pilot | 16 | 12 | 4 | 5 | 676.85 |
| Data A, seed slot 0 | 84 | 25 | 59 | 2 | 2,578.48 |
| Data A, seed slot 1 | 70 | 26 | 44 | 5 | 2,709.27 |
| Data B, seed slot 0 | 98 | 33 | 65 | 4 | 2,772.22 |
| Data B, seed slot 1 | 98 | 34 | 64 | 4 | 2,464.23 |

These are descriptive costs at their recorded stopping points, not paired
speed comparisons. Main pilot-stage chunk times were about 198--203 seconds
versus 196 seconds in development; most additional work was in measurement.
Across the four main fits, 245 observations requested directional epsilon repair,
54 were inconclusive and 92 passed the acceptance screen. R-hat was explicitly
reporting-only throughout. No engineering-invalidity or hard-veto record was
found; eight movement promotion-veto observations occurred in the first fit.
This supports an allocation/pricing repair and does not establish a controller
bug or prove a causal slowdown from co-resident GPU work.

All three saved K4 main assessments were reprocessed without new sampling.
Ten of twelve nominal intervals contain the empirical reference-sensitivity
interval; two overlap it partially. There are no disjoint comparisons. The two
overlaps remain ambiguous, and the sensitivity is not a rigorous integration
bound. Exact-reference and sequential coverage remain unestablished. Original
input checksums were verified unchanged. Six, eight and four verified candidates
remain in the original fits respectively, with only the predeclared single
member assessed in each fit; siblings are not silently promoted.

## Verification and run record

Environment: `/home/ubuntu/anaconda3/envs/tfgpu/bin/python`; numerical worker
baseline commit `de80aaff5812ebfbed551977476c0868551a2c88`. The coordinator uses
the current dirty checkout with explicit source hashes and saved source copies.
Tests intentionally set `CUDA_VISIBLE_DEVICES=-1`, memory growth and two-thread
limits before any framework import. Data and seed identities come from the
original frozen designs; the new K3 development seed and K5/K6 option identities
are stated in the plan.

- Initial focused run: 60 passed, one failed on JSON tuple/list equivalence;
  repaired and rerun with 61 passed.
- Actual public linear-Gaussian and nonlinear state-space checkpoint recovery
  through the common-pool dispatcher: **2 passed**, 128.75 seconds.
- Broader selection: **99 passed**, 28.93 seconds, covering scheduling,
  settlement, preserved negative assessments, reference reporting and existing
  public-pipeline behavior. Dependency deprecation warnings remain.
- Later manifest/denominator changes: the affected dispatcher regressions pass.
- Read-only source, prepared-input, preflight and price audit: passed; preserves
  all 32 slots and the frozen source above.
- Official tuning chapter: compiled in 3.19 seconds; changed pages 26--27 were
  rendered and visually inspected. The standalone chapter has an expected
  unresolved cross-chapter diagnostics reference and existing overfull boxes
  outside the edits. This is an isolated build of the official source, not a
  separate user guide.

Exact pytest commands are the three named test files in `focused-tests-r2.log`,
`test_campaign_recovery.py -k real_frozen` in `real-ssm-tests.log`, and the six-file
selection in `regression-tests.log`. The accompanying validation manifest records
full commands and checksums. No statistical promotion follows from these tests.

## Decision and inference status

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Execute bounded pooled recovery | Dispatcher-to-child tests, source/input checks and accounting pass | No active implementation veto in checked paths | Full workload cost on shared GPU | Launch repair/pricing then automatic main refresh | All models will finish |
| Preserve K4 assessments and add diagnostic reports | Original records unchanged; quantities and sensitivity checked | Two intervals remain ambiguous; no disjoint interval | Grid sensitivity is not a bound; only three fits | Carry all comparisons into terminal reporting | Exact or sequential coverage |
| Test K5/K6 preparation repairs | New identities and full development requirements declared | K6 valid handoff still absent | Contiguous preparation and total cost | Bounded new pilots, diagnose failures | K5's prior repair transfers to K6 |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | No new K0 engineering invalidity/hard veto in saved records; movement promotion vetoes retained. New GPU work remains unchecked. |
| Statistically supported ranking | None; no method or candidate is ranked. |
| Descriptive-only differences | Candidate counts, stage costs and K4 interval/reference comparisons. |
| Default readiness | No numerical default changed or promoted. |
| Next evidence needed | Actual resumed GPU receipts and matching full pilots, main posterior checks, complete 32-slot dispositions and settled cost. |

Next action: continue the active main queue and its already allocated checkpoint
recoveries. Repair-stage repricing has completed automatically. At terminal
closure, audit all 32 slot dispositions, interval reports and the enclosing
settlement before allocating any unused balance to unresolved pilots. Do not repeat
C1, discard prior failures or infer completion from an empty eligible queue.
