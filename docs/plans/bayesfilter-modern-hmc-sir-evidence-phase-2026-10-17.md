# Full-model SIR evidence phase

**Terminal execution, 20 October 2026:** The four-method cohort finished at
**01:42 China time**. The service is inactive. All 24 roots were verified;
23 posterior assessments passed and 21 passed the independent-reference screen.

| Method | Verified | Posterior checks passed | Delivered | Remaining failure/censoring |
| --- | ---: | ---: | ---: | --- |
| Fixed HMC | 8 | 7 | 6 | Member 1 reference failure; member 8 resource-censored |
| Randomized HMC | 8 | 8 | 7 | Member 1 reference failure |
| SNAPER | 6 | 6 | 6 | None |
| MAMS | 2 | 2 | 2 | None |

The [terminal result](bayesfilter-modern-hmc-sir-result-2026-10-20.md) records
the scientific limits, decision table, provenance and costs. All verified
members remain in the tuning set. No failed member was selectively retried,
and no numerical default changed. Unequal candidate sets and unreplicated fits
do not support a method ranking.

The [controller recovery](bayesfilter-modern-hmc-sir-controller-recovery-2026-10-18.md)
preserved fixed's terminal evidence. Saved strict audits passed for all methods.
The [closeout](bayesfilter-modern-hmc-sir-closeout-2026-10-20.md) independently
checked member decisions, accounting and 5,530 unique file hashes; it did not
recompute TensorFlow diagnostics. Numerical authority remains
`full-model-source-e68cdf1c4`; designs, seeds, GPU 1, XLA, TF32 and verified
pre-initialization memory growth are unchanged.

The numerical charge is `197315.48680924764` seconds. After the controller's
engineering/pricing charges, the outstanding submission cost and measured
closeout check, **`22375.605841124954` seconds** remain from this allocation.
The separate diagnostic pool and other held/historical balances are excluded.
All receipts are in `sir-cohort-20261017-01/closeout-20261020-01` and the
original controller output `continuation-04`.

This evidence phase is complete; broader N5 confirmation and N8 qualification
remain open. Refresh the next phase from measured costs and declared scope
before any new numerical launch. The
[active checkpoint](bayesfilter-modern-hmc-randomized-active-checkpoint-2026-10-15.md)
records exact next actions. Dated execution sections below are historical.

Date: 17 October 2026. Status: execution in progress; checkpoint refreshed
18 October 2026 after controller recovery and diagnostics documentation closeout. Preparation was authorized
by the instruction to continue the October 7 modern-HMC master. NUTS is
excluded.

## Question and evidence contract

Can the shared tuning and retained-member pipeline qualify and assess the
existing nonlinear SIR proposals, on the full horizon 20, three-parameter
finite-filter target? Compare fixed HMC, randomized HMC, SNAPER-proposed
randomized HMC and fully refreshed MAMS on identical data, affine geometry
and four starts. This extends the model inventory; it is not a method-ranking
or fresh-adaptation replication experiment.

The primary engineering criterion is a complete, auditable root/repair/member
inventory through the public tuner and cumulative posterior controller.
Acceptance lower-bound qualification and numerical health determine tuning
membership. R-hat, ESS, MCSE and reference agreement assess posterior delivery
only and cannot reject, rank, repair or delay tuning proposals. Preserve every
qualified member and every rejected, unfinished or censored row. The finite-
filter reference cannot establish an exact nonlinear state-space posterior.

| Diagnostic or event | Role and consequence |
| --- | --- |
| Acceptance lower bound and numerical/movement checks | Tuning qualification criterion and health veto; retain every qualified member. |
| Warm-up readiness, retained R-hat/ESS/MCSE, nine reference quantities | Posterior delivery criteria/vetoes only; preserve failed members. |
| Broken source/input identity, invalid reference scope, missing required telemetry or growth | Continuation veto until the harness or environment is repaired. |
| Resource interruption or infrastructure failure | Repair trigger under the same contract in a fresh directory; account all attempts. |
| Count-limited precision or reference failure | Candidate delivery failure, not research-direction rejection; no selective rerun or estimator change. |
| Short-path health, acceptance, timing and load scenarios | Cost/explanatory diagnostics only; never qualification, convergence or ranking evidence. |

The research intent is model breadth and complete pipeline accounting. Fixed
HMC is the classical baseline; randomized HMC separates randomization from the
SNAPER proposer; MAMS tests the alternative integrator with full refresh.
The cheap adversaries are the fixed one-step root, the same-step randomized
root and the longer fixed-duration roots, constructed by the classical grid.
Evaluate by method/root and physical/source-square/source-tail quantities;
do not pool failures away. There is no supported efficiency ranking from this
one-start-seed experiment. A failed cell may trigger later geometry or
adaptation work, but cannot erase its denominator.

## Frozen inputs and assumption audit

Artifact shorthand `N5` is
`docs/plans/artifacts/modern-hmc-implementation-2026-10-07/N5`.
Numerical source stays immutable at
`N5/generic-adaptation/full-model-source-e68cdf1c4`, commit
`e68cdf1c4c6b75c16f92b86a3363a1e24690d22c`. Do not edit that source or the
running SNAPER phase. New orchestration must import the declared source and
record its own hash. Target/data/geometry/pilot scope checks precede pricing.

| Choice | Provenance, failure risk and early diagnostic |
| --- | --- |
| Full NL-SIR, horizon 20, dimension 3 | Existing adapter and prepared-data fixture; assert full scope and input hashes. Synthetic finite-filter target, not a real MacroFinance/DSGE qualification. |
| Local affine geometry | `N5/full-model-geometry-02-sir/nl-sir`; unadmitted warm start, not a learned NeuTra map or proof of global coverage. Compare data and geometry hashes to both pilot designs and current binding. |
| Fixed/randomized requested travel 1,2,4,8 and epsilon .5,1 | Inherited exploratory classical hypotheses, not optimal settings; compile actual count laws, preserve quantization and qualify afresh. These hypotheses may miss useful epsilon/length regions. |
| Six SNAPER roots | Actual target-specific discarded pilot at `N5/method-proposals-02/nl-sir-snaper/discarded-pilot.json`; old qualification is not reused. Validate pilot target, geometry and transition profile. |
| Two MAMS roots | Actual discarded adaptation at `N5/generic-adaptation/full-model-pricing-02/NL-SIR/pilot/pilot-result.json`. It explored duration factors 1,2 times sqrt(3), not the missing 4,8 factors. This phase tests frozen-proposal feasibility only. |
| Common start seed 72410; RNG namespace 20261017 | Convenience independent replication identifiers, not tuned settings; record all streams. Tuning, posterior, discarded pilots and pricing use disjoint stream ranges/namespaces. |
| Qualification 32/64/128 independent 64-transition trials per stage; one repair/root; cap 128 | Existing reviewed lower-bound evidence contract; preflight attainable error allowances and full repair reserve. Separate fresh verification stream. |
| Four chains; warm-up 2000 minimum/10000 cap, retained 10000 cap, original lugsail r=3,c=.5/min_batches=20, ESS 400, MCSE/SD .02; reference 3 MCSE + .1 normalized tolerance | Frozen comparator hypotheses, not universal adequacy guarantees; reuse the explicit SGQF posterior rules on nine SIR quantities and report count limitations. Optional safeguarded MCSE is not substituted retrospectively. |

Reference compatibility report:
`N5/generic-adaptation/preconfirmation-01/reference-consumer-05/NL-SIR.json`.
The numerical worker must recheck its source, data, geometry and reference
identity before sampling; a saved `compatible` label alone is insufficient.

## Execution, budget and stops

New output root:
`N5/generic-adaptation/sir-cohort-20261017-01` (never overwrite old results).
Current unallocated balance is 233891.35098051262 aggregate worker-seconds;
the independent 180000-second live SNAPER reservation, old 3600-second reserve
and historical balance are excluded. Hold at most 3600 seconds for focused
implementation/tests/design audits and 7200 seconds for inclusive GPU pricing.
These are convenience planning caps, not measured runtime guarantees. Record
actual elapsed costs once, release unused allowance, and reserve a numerical
cohort only after complete cost diagnostics. Do not promise completion of all
twenty matrix cells within an unpriced balance.

1. Parameterize the existing design builder by target, pilot paths and random
   namespace. Keep historical defaults readable. Check pilot scope, disjoint
   streams, complete repair/error allowance and posterior settings with focused
   CPU-only tests (`CUDA_VISIBLE_DEVICES=-1`). No numerical sampler changes.
2. Build four SIR designs with the immutable numerical library. Use Python
   `/home/ubuntu/anaconda3/envs/tfgpu/bin/python`. Record command, environment,
   seeds, commit, all input hashes, wall time and plan/result paths.
3. Price every root with two independent 64-transition GPU/XLA calls, using
   a separate pricing namespace, memory growth established before import and
   verified at runtime. Preserve short-path vetoes as cost diagnostics. Share
   an otherwise available GPU only with measured capacity; contention consumes
   the same inclusive budget and is not scientific rejection. Do not disturb
   the running SNAPER cohort or other users' processes.
4. Audit all price receipts and make load 1/2/4 scenarios including every root,
   possible repair and posterior member. These are planning scenarios, not
   probabilistic bounds. A complete worst-case price above the balance means
   a bounded, explicitly censored feasibility phase or a smaller *new* phase
   contract is needed; do not silently drop expensive roots or failed prices.
5. Refresh this plan/master/checkpoint with measured costs and the exact
   execution reservation before launching qualification/posterior work. Run
   the existing immutable-source public-tuner driver with progress and terminal
   checks. Terminal integrity failure requires repair; a candidate failure does
   not stop unaffected peers. Settle inclusive supervisor cost exactly once.

Stop for exhausted total allocation, changed target/scientific assumptions,
unrepairable source/reference/telemetry invalidity or invalid GPU allocation.
A busy device triggers scheduling or a bounded retry, not method rejection.
No automatic restoration of NUTS, method ranking, default promotion, release
qualification, learned-map quality or complete broad adaptation claim.

## Skeptical audit before execution

The scope is deliberately the existing SIR frozen proposals, not a claim that
two MAMS durations are a comprehensive search. All four methods share starts,
data, geometry, qualification and posterior criteria; differing root counts
are explicit and do not support an equal-budget efficiency claim. Short paths
cannot promote members. Existing source/reference compatibility is rechecked
on the actual immutable library. Complete repair and all-member costs precede
allocation. The main risks are inadequate local whitening, narrow inherited
classical epsilon coverage, short MAMS pilot windows and contention; these
remain explanations to test later, not reasons to tune on posterior/reference
failures. The plan passes for this limited model-breadth feasibility question.

## Preparation result and pricing execution — 17 October, 03:18 China

The 79 focused scope, pricing, continuation and full-model-consumer checks
passed. The actual immutable-source builder produced all 24 roots (8 fixed,
8 randomized, 6 SNAPER, 2 MAMS), with 48 total candidate slots including repairs.
The CPU-only reference inspection revalidated the archived/current source and
input hashes and all nine SIR reference quantities. The eight runtime base
streams are distinct. These checks establish engineering scope only.

Measured preparation cost is 274.4537724782713 seconds, charged once in
`sir-cohort-20261017-01/preparation-accounting-01.json`. After the 7200-second
pricing hold, 226416.89720803435 seconds remain unallocated; the separate
SNAPER reservation and earlier reserves are unchanged. Release unused pricing
allowance only after the inclusive supervisor receipt is terminal.

The durable pricing command is preserved by `launch_pricing.py` in the phase
directory and its `pricing-launch.json`/`pricing-launch-receipt.json`. It uses
the copied `pricing-source-01/price_modern_hmc_frozen_design.py`, all four
`designs-01` files, the immutable numerical source, GPU 0 with memory growth
and XLA, seed namespace 20261018, and an inclusive 7200-second cap. Output is
`pricing-01`; unit is `bayesfilter-modern-sir-pricing-20261017-01.service`.
The price worker verifies memory growth before numerical work. GPU 0 has
ample free memory and is shared with the existing MacroFinance workload;
its contention affects measured costs, not tuning or posterior criteria.
No peer process is stopped. Before launch the skeptical audit found no new
scope, baseline, criterion or source mismatch. Full numerical allocation
remains conditional on complete audited prices.

The price service was launched with trusted GPU access at
`2026-10-16T19:20:09Z` (03:20 China time on 17 October). The launch receipt
records exit status 0 and the expected unit. The first status check found the
unit active, GPU memory growth verified, and the four-method price worker in
its first design-preparation step; no failure was reported. GPU 0 utilization
was 100% at the check, but that observation includes concurrent MacroFinance
work and is not attributed to this price worker. The independent SGQF SNAPER
service remains active with six qualified roots and posterior member 1
running; its source and memory-growth checks still pass. Neither service was
stopped or restarted. Continue with progress and terminal-receipt inspection;
do not allocate the SIR qualification/posterior cohort until all 48 calls and
the cost report have passed their audits.

## Price-cap reforecast and resumable receipts

At the first actual progress review, fixed HMC had completed eight calls
spanning `L=1,2,4`, with elapsed calls about 78–204 seconds apiece. The full
design also includes `L=8,16` fixed roots and `L=15,31` randomized roots.
Extrapolating the measured per-call growth makes 7200 seconds an unsupported
completion assumption. It remains a valid first bounded attempt; incomplete
pricing must not be interpreted as a complete cost report.

Preserve that attempt unchanged. If all 48 calls return, audit it directly.
If the supervisor reaches its cap with missing calls, settle its real
inclusive cost and use one fresh pricing-resume attempt, capped at 36000
seconds from the existing pool. This ten-hour ceiling is a planning allowance
based on observed increasing call costs and the remaining roughly 63-hour
pool, not a measured upper bound. Reuse only returned calls whose original
source/input/checkpoint receipts still pass. Price only missing calls, in
namespace 20261019. Retain interrupted calls in the old directory and in the
resume history; do not overwrite or treat their time as a complete price.

A combined terminal report needs all 24 roots, exactly two successful calls
per root, 48 distinct successful streams, and every attempt's inclusive cost,
driver and original checkpoint identity. If the second cap is reached, retain
the missing inventory for another explicitly budgeted recovery; no numerical
cohort may launch from incomplete prices. Non-timeout failures require an
identified localized repair before automatic recovery.

Skeptical re-audit: average-trajectory extrapolation can be optimistic for the
longest randomized roots, especially on a shared device. Reusing exact
completed receipts and extending only missing work is preferable to rerunning
the price grid. The recovery stays inside the authorized science and total
budget. Tests must reject unfinished prior supervisors, changed designs,
reused streams, missing completed checkpoints, double-counted costs and false
completion. The correction passes for the cost-only diagnostic.

## Durable continuation after pricing

The existing SGQF continuation script assumes four MAMS roots and a settled
SGQF funding chain. Reusing it unchanged for SIR would be wrong. Add a bounded
orchestrator around the existing immutable full-model consumer; do not change
numerical methods, candidate membership, seed derivation or posterior rules.
It will wait for a terminal inclusive pricing receipt, audit every root and
both calls with the existing tensor/source checker, and preserve all four
designs even when later work is unfunded or resource-censored.

Use the already audited 1859.353110-second SGQF deployment only as the named
non-native overhead sensitivity: subtract its 32 native call times and divide
by 32, giving approximately 26.880124 seconds/trial. Its transfer to SIR is a
planning hypothesis, not a measured SIR overhead or statistical bound. Fixed
additional overhead is zero because actual SIR preparation is included by the
price report and the amortized sensitivity already contains setup/export.
Keep load factors 1, 2 and 4 and all root/repair/member costs. Two prices per
root do not establish a runtime quantile.

Reserve at most 1800 additional seconds for focused orchestration tests and
receipt inspection (a convenience cap), then charge their measured elapsed
time once before numerical allocation. The current unallocated balance and
7200-second price hold are inputs, not new funds. Release unused pricing
allowance only from its terminal receipt. The existing SNAPER reservation,
old local reserve and historical balance remain excluded.

After audit, use the predeclared diagnostic order **fixed HMC, randomized
HMC, SNAPER-proposed randomized HMC, MAMS**: establish the classical anchor,
separate randomization, measure the target-specific proposal, then assess the
alternative integrator. This order is not a method ranking or an equal-budget
comparison. Since sequentially spending the entire balance on the first
method could leave no evidence for the remaining cohort, reserve method caps
before launch in proportion to each method's audited load-four
maximum-count scenario. Thus every method receives the same planned fraction
of its cost envelope, with a positive cap only if the remaining budget can
fund all four. Each cap is still limited by its own complete all-root/repair
scenario; unused ceiling is released after that method's terminal audit and
carried to the next method in the predeclared order. This is a transparent
budget-allocation hypothesis, not a statistical equivalence claim or completion
guarantee. If the budget cannot fund four positive caps at the minimum
all-member paths, do not silently drop methods; allocate the affordable
fraction to each by the same cost weights and mark the limitation before any
launch.

Before each launch write its exact command, environment, price/design/source
identities, root inventory and allocation. Run sequentially on GPU 0 with
verified growth, XLA and two host threads; do not add contention with concurrent
SIR methods. Each terminal child uses the existing source/input/checkpoint/
posterior integrity audit. Charge only the inclusive child supervisor and
exclusive orchestration work; never add overlapping child, native or worker
durations. Release unused child allowance only after terminal inspection.
Complete scientific outcomes, including failed posterior deliveries, permit
the next method. A missing receipt, invalid evidence, source drift or numerical
infrastructure failure stops automatic continuation for a localized repair.
Resource-censored outcomes retain their full inventory and cannot be silently
restarted. The orchestrator may continue unaffected later methods only after
the terminal audit succeeds and the remaining balance is positive.

Focused tests must cover non-four-root MAMS cohorts, complete four-method
coverage, missing/reordered/changed prices, negative or double-counted budget,
all-root/repair reserves, finite positive time caps, terminal candidate failure
versus infrastructure failure, no launch before price settlement, and fresh
output paths. An actual dry preparation should check the saved SIR designs
before the orchestration service is launched. Its own versioned source must
be recorded separately from the immutable numerical source.

Skeptical audit: automatic scheduling must not turn price-only diagnostics
into tuning admission, reuse SGQF root counts, spend the live SNAPER funds,
or select posterior successes. The proposed wrapper reads exact SIR designs,
checks complete price receipts, retains censored rows and uses the unchanged
numerical consumer. Method order and cost transfer are explicit hypotheses;
neither supports a performance ranking. The bounded continuation passes for
this feasibility question, subject to the stated focused tests and actual
price audit. No public API or scientific default changes.

### Continuation implementation audit — 17 October, 04:20 China

Source tracing found a misplaced indentation in the new recovery reporter:
the comparison with the original returned call was unreachable. Repair it
and test mutations of elapsed cost, vetoes and seed before using recovery
receipts. This is an engineering defect, not evidence against a sampler.

The 1800-second engineering hold is separate from the 233616.89720803435-
second post-preparation pool and must be deducted before numerical allocation.
It covers measured focused checks and exclusive orchestration/audit work;
polling sleeps and overlapping child wall time are excluded. Unused engineering
allowance remains held until terminal closeout. The total prices charge is
the sum of inclusive pricing attempts, deducted once; the original 7200 hold
is released through that settlement, not added as new funds.

Allocate every method a positive fraction of its load-four envelope even
when minimum all-member paths do not fit; record that limitation before
launch. Following audited settlements, release unused caps to the next method
without spending later methods' initial reservations. A supervisor overrun
reduces remaining reservations proportionally if necessary; never manufacture
a nonnegative balance. No completed or censored method is automatically
rerun. Terminal infrastructure invalidity stops this orchestration. This
clarifies the earlier minimum-path wording: affordability limits the claim,
not the preserved denominator or the start of a bounded diagnostic attempt.

Capacity admission uses the existing trusted shared-device GPU-0 headroom
check (4096 MiB free, inherited scheduling screen, not a memory guarantee).
Insufficient headroom permits at most 3600 seconds of waiting, a convenience
scheduling cap inherited from the existing continuation. Unexpected device
errors stop for diagnosis. Growth and XLA remain mandatory in every child.
The wrapper observes each child every 30 seconds (an inherited telemetry
interval), records progress, and calls the strict terminal auditor after exit.
It does not extend numerical caps, retune on posterior failure, or alter
immutable source. The pre-execution audit passes for this bounded feasibility
contract, subject to focused regressions and an actual saved-design dry check.

### Implementation check before the continuation tests

The snapshot records `git_commit`, not `commit`; the wrapper now uses the
actual schema. It also requires the visible GPU to match the capacity probe,
subtracts settled pricing before reporting affordability, and records signed
terminal balances and the unused engineering hold. A telemetry error waits for
the already bounded child to finish and retains an inspection requirement; it
cannot abandon, restart or retune the child. Terminal evidence invalidity
retains the unaccounted child charge and withholds balance release.

The skeptical audit rechecked baseline/source identity, price-only versus
posterior evidence, root/repair denominators, disjoint pricing streams, shared
GPU assumptions, cap overruns, missing terminal receipts, and source isolation.
Focused tests and a dry check of the actual saved designs are the next
engineering checks. They cannot establish sampler quality or completion. This
localized orchestration repair preserves the scientific contract and uses the
existing 1800-second engineering allowance.

### Deployment audit — 17 October, 04:35 China

The checkout suite passed 158 tests. Its initial two failures were incorrect
test fixtures (affordability and missing comparator provenance); their measured
cost remains charged. The versioned `continuation-source-01/scripts` copy then
passed 125 focused regressions and CLI/import checks. Namespace resolution
still selects the immutable full-model numerical consumer and numerical
package; no regular `scripts` package marker was added. The saved-input check
passed for the actual 8/8/6/2 roots, horizon 20, dimension 3, target/start seed,
proposal/source hashes, compiled operator, and preparation balance.

The final skeptical audit preserves the classical baseline, all roots and
repair/member denominators, pricing-only cost role, posterior-only convergence
criteria, disjoint recovery streams, original failed/censored outcomes and
independent SGQF reservation. The wrapper cannot allocate numerical methods
from partial prices. Every child remains bounded and audited before releasing
its unused ceiling. Tests cover original completion, missing-only timeout
recovery, four-method progression, source/tensor failure, signed overruns and
observer failure while a child remains alive. The deployment passes for the
declared feasibility question; it establishes no sampler-quality result.

Deploy the copied wrapper as
`bayesfilter-modern-sir-continuation-20261017-01.service` using the exact saved
`sir-cohort-20261017-01/launch_continuation.py`. The service first observes
the existing pricing attempt; it does not restart it. Its liveness ceiling is
derived from the remaining original-price observation allowance, the existing
233616.89720803435-second pool, up to five 3600-second capacity waits, and a
600-second closeout margin (convenience scheduling allowance). That conservative
outer ceiling adds no compute: cumulative inclusive pricing, numerical caps
and exclusive engineering remain charged to the existing pool. The 1800-second
engineering hold includes both test attempts, saved-input and snapshot checks,
and measured launch preparation; service-submission cost is retained for final
reconciliation. No unrelated service or immutable source is modified.

### Terminal pricing and device repair — 17 October, 13:50 China

The original pricing attempt stopped at its inclusive cap, with
7200.393969984725 seconds charged and 26 of 48 calls returned (16 fixed HMC,
10 randomized HMC). A CPU-only terminal reuse audit passed for all returned
source/input/checkpoint receipts. The single permitted recovery therefore
needs 22 calls: six randomized, twelve SNAPER and four MAMS. No completed call
will be repeated, and interrupted work remains in the original attempt.

The first continuation launch did not create a service or numerical child.
Its trusted prelaunch probe rejected GPU 0 because GPU 1 was available and
idle. The separate SGQF SNAPER run had meanwhile completed and passed its
terminal audit. A trusted GPU-1 probe passed and identified the same RTX 4080
SUPER hardware class as the original pricing device. Use GPU 1 for this
continuation; retain GPU 0 as the original-attempt provenance. This is a
scheduling repair within the same authorized scientific scope and budget.
Different contention remains a cost-explanation risk, not admission evidence.

The rejected launcher remains preserved. Deploy
`launch_continuation_02.py`, unit
`bayesfilter-modern-sir-continuation-20261017-02.service`, with fresh output
`continuation-02`. The tested `continuation-source-01` is unchanged. The new
launcher only updates device and output names and includes both the failed
prelaunch receipt and actual terminal reuse audit in the engineering charge.
Recovered SGQF funds are kept separate; this launcher still uses only the
original SIR pool. The skeptical recheck passes: no baseline, numerical code,
threshold, target, root denominator, random namespace or promotion criterion
changes; the 36000-second missing-call cap remains enforced. The new device
is checked again before each child and memory growth before framework import.

### Recovery binding diagnosis — 17 October, 14:00 China

Continuation 02 stopped during randomized-HMC binding construction, before
any new native price call. Its 90.53459248691797-second supervisor cost remains
charged separately from the original 7200.393969984725 seconds. The failed
run and source copy remain immutable. It is an infrastructure/provenance
failure, not candidate rejection. The observer correctly withheld settlement.

Source tracing shows the execution binding includes `cuda_visible_devices`
in its runtime policy. Moving from physical GPU 0 to GPU 1 therefore changes
the binding hash even on the same hardware class. Establish whether this is
the entire mismatch by reconstructing only the randomized binding on GPU 1,
comparing source, starts, geometry and data, and testing whether substituting
the *recorded* old runtime policy in a copied spec reproduces the old hash.
This diagnostic never mutates the real binding or launches transitions. Use
one 240-second inclusive diagnostic cap (planning allowance for the observed
90-second setup); charge actual wall time from the existing SIR pool.

If every non-runtime field matches and the exact old hash is reconstructed,
cost-only reuse may retain each call's originating binding while reissuing
the current-device binding. This would require explicit migration evidence,
per-origin checkpoint audits and regressions rejecting any other difference.
The numerical consumer and its strict device-bound identity stay unchanged.
If equivalence fails, do not weaken the guard: use fresh complete pricing on
the current device under a revised bounded allowance. The skeptical audit
passes for this small diagnostic because it distinguishes identity from
numerical drift without using either as sampler evidence.

### Confirmed diagnosis and complete repricing repair — 17 October, 14:25 China

The bounded diagnostic completed in 70.80671620601788 seconds without a
transition. Only the binding hash differed. Replacing the runtime policy in
a *copy* of the GPU-1 spec with the recorded GPU-0 policy exactly reproduced
the old hash; source, data, starts and geometry matched. The full diagnostic
is `binding-diagnostic-01/result.json`. This is a device-provenance mismatch,
not evidence of a different target or a failed sampler.

Use **fresh complete pricing on GPU 1**, rather than add mixed-origin binding
semantics to the reporter/checkpoint auditor. Preserve all 26 GPU-0 returns as
diagnostic evidence; none will enter the new price report. The strict numerical
binding stays unchanged. This revises the earlier preference for missing-only
recovery: a single-device price inventory answers the same cost question with
less harness complexity and stronger existing checks. All 48 calls must return
before qualification/posterior allocation. The fresh namespace is 20261020,
a convenience identifier disjoint from the previous three namespaces and the
unchanged runtime namespace. No posterior failure is being retried.

Reconcile the original price cost 7200.393969984725 seconds, failed recovery
90.53459248691797 seconds, diagnostic 70.80671620601788 seconds, exclusive
engineering 14.717859779712656 seconds, and the launch-submission receipt
0.04737961897626519 seconds before funding the next attempt. Submission time
is charged once as a conservative tiny additional overhead; child/native
durations are never added to their enclosing supervisor. The previous 36,000
second repair allowance now has **35838.65869130706 seconds** available for
fresh prices, after failed recovery and diagnostic. The original 7,200 seconds
is already spent, not a second allowance. Keep released SGQF funds separate.

The orchestrator needs one explicit scheduling control: a zero recovery cap
means no automatic price retry. Complete prices still proceed to the existing
all-four-method reservation and immutable numerical consumer. Incomplete
terminal prices settle their inclusive cost and stop before any numerical
method; they do not create a fresh retry allowance. Add focused regressions
for this branch and for complete prices with zero retry allowance, then copy
the orchestration to a new `continuation-source-02` without changing the old
snapshot. The remaining engineering hold is 1785.234760601311 seconds, after
prior engineering and submission; focused checks and new observer work consume
that hold. All preparation/pricing/engineering costs remain in the original
SIR pool. Record new receipts and launch into fresh directories and services.

Skeptical re-audit: the cheapest safe alternative, weakening the binding hash,
would mix device-bound checkpoints without existing per-origin audits. Fresh
pricing avoids that flaw; old receipts are not relabeled or rewritten. Its
ceiling is a budget stop, not an ETA, and may still produce an incomplete
cost report. Two calls/root remain descriptive cost evidence only. Actual
growth, source, inputs, seed separation and all-root/repair costs are checked
again; CPU regressions do not establish GPU/sampler validity. A trusted probe
must recheck GPU 1 before launch. No baseline, target, numerical method,
threshold, root denominator, posterior stopping rule or total allocation
changes. This repair passes the audit under the original feasibility contract.

### Fresh deployment audit — 17 October, 14:26 China

The checkout and new `continuation-source-02` each passed all 164 focused
regressions. The deployed snapshot resolved all six orchestration modules to
the new copy and the actual numerical consumer/package to the unchanged
`full-model-source-e68cdf1c4`; CLI checks passed without importing TensorFlow
in orchestration. The read-only saved-input check passed for all 24 roots,
48 candidate slots, horizon, geometry, starts, operator and source hashes.
No regular `scripts/__init__.py` was introduced.

The exact fresh commands are in `sir-cohort-20261017-01/launch_repricing_02.py`.
They use pricing namespace 20261020, GPU 1, memory growth and two host threads.
The pricing unit is `bayesfilter-modern-sir-pricing-20261017-02.service`; the
conditional numerical controller is
`bayesfilter-modern-sir-continuation-20261017-03.service`, with fresh outputs
`pricing-02` and `continuation-03`. The launcher waits for the initial pricing
receipt before submitting its consumer, preventing a startup file race. This
wait overlaps the inclusive price supervisor and adds no compute charge.

Skeptical audit passes for the unchanged feasibility contract. Reconciled
SIR funds before new prices/engineering are 226240.396689958 seconds; the
fresh pricing ceiling is 35838.65869130706 seconds and remaining engineering
hold is 1785.234760601311 seconds. Earlier inclusive attempts are charged once,
new check costs enter the remaining hold, and released SGQF funds stay
separate. Zero recovery cap prevents another automatic retry. All four
methods retain positive reservations after complete price audit; missing
prices cannot fund numerical work. The inherited 3600-second capacity wait,
30-second observation interval and 600-second outer closeout margin are
scheduling allowances, not scientific thresholds or new compute. Final
service-submission costs remain explicit pending charges. No sampler or
posterior conclusion follows from these engineering checks.
