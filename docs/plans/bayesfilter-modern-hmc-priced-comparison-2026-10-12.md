# Priced full-model continuation of the generic trajectory repair

Date: 12 October 2026. Status: pricing completed and audited; the bounded MAMS
feasibility allocation is recorded and fresh qualification is running on GPU 1.
This continues D4R4 of the modern-HMC master. It does not change a numerical
default, create a tuning authority or introduce NUTS.

Current execution: all 26 roots / 52 native price calls completed at 13:56 UTC,
charging 6851.941203 seconds of the 7200-second price reservation. The audited
allocation is `priced-comparison-01/continuation-02/allocation.json` under the
N5 generic-adaptation root below. It reserves 123019.624951 seconds for the
unchanged four-root MAMS design, including its four possible repairs. The
3600-second local reserve and 348.058797 unused price seconds remain held;
unallocated worker time is zero after the MAMS reservation. This is the scoped
feasibility branch below. The larger comparison and paired replications are
unfunded. See the [execution record](bayesfilter-modern-hmc-priced-execution-2026-10-12.md)
for the live service, evidence and next checks.

## Research question and evidence contract

Can the frozen proposals produced by actual discarded MAMS adaptation deliver
the existing model-coordinate posterior precision on full predator-prey SGQF,
and what is the cost relative to adequately searched classical HMC and SNAPER?
The original failure was high acceptance with short travel and an incomplete
method-specific search. Qualification/export mechanics are already checked;
another count-one, broad-acceptance-floor fixture cannot answer this question.

Use the committed numerical snapshot `generic-adaptation/full-model-source-e68cdf1c4`
under `docs/plans/artifacts/modern-hmc-implementation-2026-10-07/N5` (N5 below).
Its four independently checked finite-filter reference bridges are in
`generic-adaptation/preconfirmation-01/reference-consumer-05`. The source,
observations, saved local factor, target horizon and quantity chart stay fixed.
The factor is an unadmitted warm start, not a globally validated learned map.
New diagnostic orchestration has its own committed source hash. It imports
the numerical snapshot explicitly; it does not edit that snapshot or copy a
numerical implementation into a new consumer.

| Quantity or event | Role | Decision |
| --- | --- | --- |
| Public replicated qualification, floor .5, numerical health and independent verification | Tuning membership criterion | Preserve all verified roots and repair descendants |
| Source/data mutation, wrong target, missing trace, wrong device/compilation or invalid reference | Continuation veto for affected work | Preserve and repair before interpreting dependent results |
| Numerical failure confined to a candidate | Candidate veto and possible repair trigger | Preserve its outcome and continue healthy peers |
| R-hat, bulk/tail ESS, lugsail MCSE, independent reference agreement | Posterior delivery criteria | Assess all verified identities; never change tuning membership |
| Short native prices, pilot IACT, acceptance means and runtime | Explanatory/planning only | Price work; do not rank samplers or select by these observations |
| Count or wall allocation exhausted | Resource outcome | Preserve unfinished denominators; no scientific rejection |
| Paired replicated delivery/cost with uncertainty | Required for comparative robustness claims | No ranking from one replication or censored unequal coverage |

Stage A succeeds when every planned root has a current native price or explicit
failure, a separate cold/steady distinction, and a complete cost envelope for
measurement, fresh verification, repairs and all posterior members. It does
not establish sampler quality. Stage B's primary outcome is member-level
delivery under the unchanged posterior checks, reported with the full proposal,
created, verified, attempted-posterior and delivered denominators. A failed
member can motivate a later development cohort, never revision of this cohort.

The full twenty-cell/five-model inventory remains open. SGQF is first because
it contains the original failure and has completed adapted proposals and an
independent reference. An initial one-model result cannot establish generality.
The structural longer-window adaptation is still incomplete. Do not invent its
missing cohort. The bounded stationary MCSE/stopping reference study is now
complete, as recorded in its [result](bayesfilter-mcse-stopping-calibration-result-2026-10-12.md).
Its findings leave estimator/stopping repair and final guide/release review
open; they do not change this cohort's frozen posterior criteria.

## Frozen design and provenance of choices

The executable design builder will preserve the complete source/input hashes
and all settings in JSON. Stage A starts with the following full-model SGQF
cohorts, without eliminating a root on price or acceptance grounds:

* Fixed HMC: requested durations 1, 2, 4, 8 and epsilon .5, 1.0 (eight roots).
* Randomized HMC: the same **mean** durations and epsilon values (eight roots).
  Use the shared duration compiler; fixed count and uniform maximum count
  are different laws and must not be equated.
* SNAPER: all six hypotheses from `N5/method-proposals-02/nl-pp-sgqf-snaper`,
  with its actual discarded pilot and omissions preserved.
* MAMS: all four hypotheses from
  `N5/generic-adaptation/preconfirmation-01/window-256/NL-PP-SGQF/pilot`.
  These were generated by the shared adaptive proposer. Preserve their full
  pilot provenance, including source and slow-observable estimates.

This is a conditional comparison of frozen target-specific proposal cohorts.
It does not independently replicate the adaptation procedure. Pilot cost is
reported separately and included in end-to-end estimates; it is not charged
again as newly consumed campaign compute. Classical epsilon/duration coverage
is a broad geometric starting hypothesis in the prepared coordinates, not a
proven optimal classical search. If both epsilon edges fail or the useful
duration lies on a boundary, report search inadequacy and use a fresh cohort
before a superiority claim. Fixed/randomized cohorts include simple short
baselines and longer policies; the old grid-MAMS results are historical
development evidence, not a matched confirmatory comparator.

| Choice | Provenance and justification | Failure and early diagnostic | Status |
| --- | --- | --- | --- |
| Durations 1,2,4,8; epsilon .5,1 | Convenience geometric coverage in locally standardized coordinates; spans eightfold travel and twofold resolution | Edge optimum, instability or resonance; keep all native prices and later delivery outcomes | Search hypothesis |
| SNAPER/MAMS proposals | Actual target-specific discarded pilots above | Pilot estimates can be unreliable or stale; preserve source/configuration and freshly qualify | Frozen warm-start hypotheses |
| Epsilon domain [.001,4], count cap 128, one repair/family, factor 1.5 | Existing generic adaptation/consumer repair design; finite bounded exploration | Quantization or boundary prevents duration-preserving repair; record omission | Reviewed engineering bounds, not universal scientific defaults |
| 32/64/128 trials, 64 transitions, four starts | Existing independent-unit law and deterministic sensitivity table; 32 alone can be impossible/inconclusive | Real heterogeneous starts can still lack power; preflight and full rung accounting | Adequacy hypothesis, not power proof |
| Floor .5; family errors .05; preferred .65-.75 | Existing scientific-floor study policy | Low acceptance or too-wide intervals; report exact public decision | Inherited comparator policy |
| Shared start seed 71410 | Arbitrary fresh reproducibility identifier; same start bank across methods | Accidental stream collision; record stage seeds and binding identities | Convenience seed |
| Future paired start seeds 71510,71610 | Predeclared possible replications, only when affordable | Too few replications for ranking; exact sign/permutation uncertainty remains coarse | Unfunded until priced |
| 64-transition prices, two repeats/root | Exact qualification trial horizon; distinguishes cold call from reused graph | Load changes and posterior trajectory heterogeneity; record device load and rate ranges | Cost diagnostic only |
| Load multipliers 1,2,4 | Sensitivity scenarios, not measured percentiles or confidence bounds | Shared load may exceed fourfold; hard total budget still bounds spend | Explicit hypotheses |

Reserve every family's repair in the statistical candidate allowance: 16 fixed,
16 randomized, 12 SNAPER and 8 MAMS. Both measurement and independent
verification accumulate 32,64,128 trials; increments are 32,32,64 within each
stage. Do not sum cumulative looks as fresh work. Use the existing
`preserve_mean_duration` repair policy for randomized HMC, SNAPER and MAMS;
fixed HMC uses its supported `fixed_count` repair. In that arm an epsilon
repair changes physical travel, recorded in the child identity. There is no
claim that every method uses an identical adaptation mechanism. No legacy
pilot/refinement scheduler is enabled.
Price the full reserved repair count conservatively, without assuming rejected
roots or descendants vanish from cost or inventory.

For posterior work retain the previous SGQF design: four chains; warm-up chunks
1000, minimum 2000, recent window 1000, maximum 10000, R-hat <=1.05; retained
chunks 1000, minimum 1000, maximum 10000, R-hat <=1.01; bulk/tail ESS >=400 and
lugsail MCSE/SD <=.02 for all 18 original-coordinate/second-moment/upper-tail
quantities. Lugsail r=3,c=.5,min_batches=20,batch_size=None are inherited
estimator choices, not guaranteed conservative (the antithetic counterexample
is already recorded). Missing/invalid uncertainty cannot pass. The independent
finite-target screen uses (absolute mean error + 3 MCSE + reference sensitivity)
/reference SD <=.1; this is an operational agreement criterion, not a nominal
simultaneous or sequential confidence statement. Three and .1 are inherited
comparison thresholds, not freshly estimated guarantees. No estimator default
changes. Freeze distinct posterior streams per method/member; matching start
seeds does not imply common random numbers across different kernels.

## Budget, commands, stop and refresh mechanism

Previous settlement leaves **133819.624951 aggregate worker-seconds**. Reserve
3600 for local tests/design/terminal diagnostics and at most 7200 for Stage A
GPU prices including failed attempts. These are allocation choices, not claims
that work should finish at a fixed speed. The remaining 123019.624951 seconds
are unallocated until measured prices are reviewed. Release unused reservations;
charge actual wall time for every failed or interrupted attempt. Never count
parallel worker time only once.

Fresh output root: `N5/generic-adaptation/priced-comparison-01`. Write the
designs before launch; new retries use a new numbered directory. The pricing
driver takes `--source-root`, `--design`, `--output`, `--max-wall-seconds`; the
existing snapshot `scripts/run_modern_hmc_frozen_full_model.py` executes Stage B
with `--design`, `--output`, `--max-wall-seconds`. Use
`/home/ubuntu/anaconda3/envs/tfgpu/bin/python`, trusted GPU admission,
`TF_FORCE_GPU_ALLOW_GROWTH=true`, verified growth, XLA, and two host TF threads.
CPU mechanics explicitly set `CUDA_VISIBLE_DEVICES=-1`. Save the exact command,
environment, source commit/hash, device probe, elapsed time and seeds.

The GPU probe selects only permitted devices. If none is idle, the existing
trusted shared-device admission requires at least 4096 MiB free and records
contention; it never interrupts another workload. Contention is not sampler
failure. Native checkpoints provide progress; a per-stage hard wall stop is
an allocation bound. A healthy interrupted run can receive a new allocation
within the settled campaign balance, with prior evidence and cost preserved.

After Stage A, write a priced allocation and update the master. First try to
fund the complete four-method SGQF cohort and fresh replications with all-member
maximum count costs and load sensitivity. If that does not fit, authorize only
a clearly scoped bounded feasibility stage (initially adapted MAMS to answer
the original failure), preserve every unfunded cell, and report no comparative
ranking. Do not weaken posterior criteria, lower the .5 tuning floor, silently
omit slow candidates, or rename one replication a confirmation campaign. Keep
method-level comparisons descriptive until the same research contract is
complete across the compared cells and independent replications.

## Skeptical pre-execution audit

The initial interpretation of the next step as a guaranteed complete paired
twenty-cell comparison is unsupported by the remaining budget. The repaired
plan therefore prices all work before allocating it, retains unfunded cells,
and defines scoped feasibility as a different result from a comparison.
Classical baselines previously lacked broad travel/resolution coverage; they
now have explicit geometric coverage and repair allowances. That is still
finite search, not evidence of global optimal tuning. Historical pilots are
proposal provenance only; all qualification/posterior streams are fresh.

The audit checked wrong baselines, proxy promotion, missing repair and fresh
verification costs, count-vs-duration confusion, stale source/environment,
unequal starts and censoring. The isolated source and existing reference bridge
avoid interpreting unrelated dirty-worktree changes. Stage A artifacts answer
cost feasibility only. Stage B has unchanged substantive posterior criteria.
Budget/source invalidity stops execution; a failed candidate triggers its planned
repair or later diagnosis and does not stop the research direction. This plan
passes for implementation and Stage A. Stage B's allocation remains a required
data-dependent plan update, not a request for another human permission.

## Execution audit correction

The first builder mistakenly requested duration-preserving repair for fixed
HMC. The current controller explicitly rejects that combination; its guard is
correct. Deterministic acceptance attainability alone did not detect this
separate support error. The corrected builder emits fixed-count repair only
for fixed HMC, and a regression rejects reintroducing the mismatch. New
qualification uses `designs-02`, preserving `designs` as the original proposal.
Root trajectory prices in the active `prices-01` remain valid because that
stage never executes the repair/controller; root execution configurations,
starts and frozen pairs are identical. Any reuse checks those equalities and
recomputes cost from the corrected search. No original evidence is overwritten.
The initial repair-count expansion remains conservative for fixed-count
children; it is a scenario, not a measured bound. Qualification, including
descendants, is still fresh.

## Terminal pricing review and continuation

The terminal report will check the saved native checkpoints, numerical source,
compiled operator, common starts, complete root inventory and GPU/XLA/memory
policy before deriving costs. Reuse with `designs-02` must compare the complete
designs: the only permitted difference is the documented fixed-HMC repair
encoding (v3 duration repair to v2 fixed-count repair). A changed target, start,
root, execution policy or data/geometry cannot inherit these prices. This is a
read-only evidence check, not a new tuning or sampler implementation.

Report each first and repeated native call separately and retain numerical
vetoes in all cost denominators. Use the existing conservative maximum-call
envelope at load factors 1, 2 and 4. These remain sensitivity scenarios; two
calls do not estimate a runtime quantile. Per-trial non-native overhead is
reported separately, with the completed deployment as a measured planning
comparator. Include preparation, verification/export and posterior-assessment
cost uncertainty. If a price allocation ends before the full inventory, report
the missing prices explicitly and continue them in a new bounded attempt
within the campaign balance before claiming a complete four-method envelope.

Skeptical continuation audit: a root-only price remains valid after the narrow
fixed-count correction, but accepting arbitrary design changes would be wrong.
Checking pairs alone would miss changed starts or execution policy. The report
therefore checks those inputs and the saved call receipts as well. A pricing
veto can prevent interpreting a candidate, but cannot silently remove its
planned work. This review passes for terminal-report implementation and focused
accounting regressions, charged to the existing local reserve. A posterior
allocation still requires the completed price report and a recorded budget
decision; no statistical criteria are changed by the report.

The completed fixed-HMC prices already rule out funding the conservative full
four-method envelope from the current balance: its own maximum-count scenario
is 278.45 hours at load factor one. This number includes deliberate repeated
cold-call and repair overpricing; it is not a measured completion time. The
prior corrected deployment spent 1859.353110 seconds, of which 999.189142 were
native calls across 32 trials. Use the remaining 860.163968/32 = 26.880124
seconds/trial as a separate overhead sensitivity, explicitly amortizing setup,
verification and export. Adding the new preparation cost double-counts some
setup conservatively. This does not assert a universal per-trial cost.

To continue without leaving the GPU queue idle after terminal prices, the
continuation driver will write the audited report and an allocation before
launching the existing immutable full-model consumer. Its rule is fixed now:
keep the full four-method price denominator; reserve the existing local and
pricing allowances; subtract any pricing overrun from the 123019.624951
unallocated seconds; and consider **all four adapted MAMS roots and their four
reserved repair children**. The load-one minimum-count scenario is a
conservative forecast, not a lower bound on the cost of a useful attempt.
Allocate at most the actual remaining ceiling, or the load-four maximum-count
scenario when smaller. If even the minimum-count scenario exceeds the ceiling,
record that fact and label the launch bounded feasibility with potentially
incomplete qualification and posterior coverage. It does not promise a full
delivery study. An exhausted actual allocation prevents launch; a forecast
alone does not. Unless the load-four scenario fits, label
the launch bounded feasibility, with resource censoring possible. Qualification
and posterior criteria, streams, source, reference and design stay unchanged.
The three classical/SNAPER cells, later models and paired replications remain
unfunded in this allocation; their results cannot be inferred from MAMS.

The driver waits for the pricing supervisor's terminal settlement and uses
the existing trusted idle/shared-GPU selector. Capacity checks may retry for
at most one hour before recording a resource outcome; this is a convenience
queue bound, not a sampler timeout. Thirty-second polling is operational only.
Waiting consumes no GPU reservation; its elapsed time is recorded separately
from occupied child-worker seconds. Native child execution retains its existing
hard total allocation and durable checkpoints. Invalid prices, design/source
mismatch or unexpected device-probe errors stop dependent work. The terminal
child outcome, including failure or censoring, is preserved for the next repair.
No failed candidate is removed to make the allocation appear sufficient.

Skeptical audit of automatic continuation: the price report is a prerequisite,
not a statistical selector. The fallback method was specified before observing
these prices to address the original MAMS failure. The calculation never spends
released reserves twice, never treats a wall limit as expected completion, and
does not promise paired confirmation from a single target/seed. The ordinary
snapshot consumer still performs every membership, export, posterior and
reference check. Focused decision/accounting tests are required before enabling
this continuation. All launches remain within the already authorized campaign.

## Continuation-policy repair, 12 October

The first implementation used `minimum_path_seconds > available_ceiling` as
a continuation veto. That was wrong for the declared bounded feasibility
question: the scenario prices every reserved repair and repeats cold-call
costs conservatively. It is neither a measured lower bound nor evidence that
the next planned trial cannot run. The user has reaffirmed continuation.
Retain the scenario as a planning diagnostic, preserve the exact same four
roots and reserved descendants, and allow the existing consumer to attempt
them within the actual remaining wall allocation. Report every unattempted,
unfinished, verified and delivered identity separately. Partial coverage cannot
support comparative robustness, a full-cohort completion claim or release.

The scientific question, baseline, statistical rules, immutable source,
target/data, seed, device class and overall budget remain as above. This repair
changes the planning veto, not the sampler or its qualification rules. A zero
remaining allocation, invalid source/price/design or missing required numerical
evidence still prevents dependent execution. A negative forecast margin does
not consume or cancel the actual remaining budget. The consumer's existing
supervisor bounds wall time and preserves partial records and checkpoints.

Before implementation, review of `finalize_interruption` and the consumer's
`resource_stopped` path confirms that it preserves the declared primary cohort
alongside observed candidates and posterior outcomes. Add regressions for a
forecast that exceeds a positive budget, actual budget exhaustion, supervisor
overruns and incomplete child outcomes. Keep the held reserves separate and
retain all cost sensitivity values. Restart only the waiting orchestration
service in a fresh `continuation-02` directory; the live GPU price worker is
unaffected. Its former service must be inactive and have launched no child
before enabling the replacement, preventing duplicate spending.

Skeptical audit: this does not relabel an incomplete study as a complete one,
choose cheaper candidates or change a success threshold after seeing results.
The four roots were frozen before pricing. The engineering pass criterion is
bounded launch with complete planned denominators; posterior criteria remain
unchanged. Scientific invalidity remains a veto. Bounded partial evidence
answers feasibility only. The repaired continuation passes this audit.
