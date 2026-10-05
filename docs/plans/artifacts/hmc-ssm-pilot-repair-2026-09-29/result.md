# State-space pilot repair execution

## Terminal result, checked September 30

The main run ended normally at **23:23:42 Shanghai on September 29**, after
18,893.277 seconds (5.248 hours). Both supervising services are inactive, and
a trusted host process check found no remaining campaign Python worker. The
enclosing receipt was charged exactly once; its reservation is cleared. The
settled grant balance is 151,610.488 GPU seconds (42.114 hours), but the original
latest-start deadline has passed. No further numerical work is queued.

| Original main disposition | Fits |
| --- | ---: |
| Complete, declared posterior checks passed | 3 |
| Timed out during tuning | 4 |
| Funded, not started before the original cutoff | 1 |
| Unfunded within the remaining wall allocation | 24 |
| **Original denominator** | **32** |

The three completed fits are K4 data-A/seed-0, data-A/seed-1 and data-B/seed-0:
the linear Gaussian model with unknown process and observation noise scales.
Each assessed the single member specified by its original design. Each passed
declared posterior checks and precision screens after 2,000 warmup and 1,000
retained transitions per chain, excluding warmup. All three were within the
descriptive independent-reference tolerances, without reported hard vetoes,
duplicate chains or observed false-favorable screens. Their inventories retain
six, eight and four verified candidates; other members remain unassessed by
design. Interval-coverage records are unavailable for all four declared K4
mean/quantile quantities. Passing the recorded checks therefore does not close
coverage calibration. There is no completed nonlinear main fit; K7 remains
development-pilot evidence only.

All four K0 fits exhausted 3,352-second worker allocations, including their
560.333-second contention extensions, during candidate search. Their ordered
data/seed slots preserve 84, 70, 98 and 98 candidates; two, five, four and four
verified candidates; and 19, 17, 18 and 28 pending work items. None reached
posterior sampling. The last K4 data-B/seed-1 fit was skipped when dispatch
reached it after the 23:03:52 latest-start cutoff. Coordinator exit zero means
the bounded queue and settlement completed; it does not mean the main matrix
passed.

The terminal audit verifies every executed cell's receipt hash, every completed
assessment hash, frozen source identity, GPU/XLA execution and memory growth.
All 32 dispositions are reconciled in
`../hmc-ssm-main-2026-09-29/r1/terminal-audit-2026-09-30.json` and
`../hmc-ssm-main-2026-09-29/r1/inventory-live.json`. Commands, environment, data,
seeds and source provenance remain in the main and per-cell manifests. This
closeout used existing artifacts only and introduced no numerical-policy change.

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Close bounded execution and accounting | Terminal receipts, shutdown and one enclosing charge verified | No coordinator failure or remaining worker | Main scientific workload incomplete | Preserve all 32 dispositions | Successful validation of the full matrix |
| Preserve three K4 results | Requested members pass declared posterior and descriptive reference checks | No reported hard veto in these members | Few fits; interval coverage unavailable | Retain as limited model-specific evidence | Calibration, superiority or default readiness |
| Record four K0 resource failures | Full funded searches did not finish | No posterior assessment available | Search expansion, compilation and contention contributions | Diagnose saved work inventory and pricing before another campaign | Invalid model, failed posterior checks or R-hat tuning rejection |

| Inference status | Evidence |
| --- | --- |
| Hard veto screen | Completed K4 assessed members have no reported hard veto; K0 posterior outcomes unavailable |
| Statistically supported ranking | None |
| Descriptive-only differences | Reference errors, candidate counts, runtime and finite-chain diagnostics |
| Default readiness | No numerical default changed or promoted |
| Next evidence needed | K0 runtime diagnosis; remaining model/main coverage; K1/K3 complete pilots, K5 full repaired pilot and K6 preparation repair |

The post-run audit rejects the inference that a successful pilot plus a 1.5
planning margin guarantees a complete main fit. It also rejects attributing all
runtime to foreign GPU work: the receipts establish co-residency and granted
time, not causal slowdown. The research targets remain viable, while runtime
prediction, search cost and missing main evidence remain unresolved. Existing
K3 reference-workload differences and K6 reference limitations remain open.

## Additional diagnosis of the last main run, September 30

The four K0 checkpoints contain 391 recorded observations: 166 decisions to
increase epsilon, 79 to decrease it, 54 inconclusive acceptance decisions and
92 passing acceptance decisions. These are repeated search-stage observations,
not independent posterior fits or final verified-candidate counts. Every
observation labels R-hat `reporting_only`. No recorded K0 observation has an
engineering-invalidity reason or hard veto. Eight observations in data-A/seed-0
have a movement promotion veto; those remain candidate-level screening evidence.
The source of the large search expansion is not yet fully explained, but the
record establishes substantial directional repair and evidence-extension work.
It does not establish a geometry failure, a fatal numerical exception or a
causal runtime contribution from GPU contention.

Recovery was only partially integrated into this main launcher. Its executed
source matches the recorded coordinator hash. In the frozen
`fit_process.py::contention_retry_budget`, an automatic retry can use only the
unspent original fit cap. All four K0 receipts exhausted that cap. The main
coordinator's `worker` loop then records the resource stop and advances to the
next slot. It does not invoke the existing campaign-allocation continuation
helper or reassign unused time from a cheaper completed fit to an interrupted
one. The contention extension was applied, and numerical checkpoints were
preserved, but no K0 main continuation was dispatched. This is the remaining
integration gap in recovery; another larger fixed multiplier would not repair
that dispatch behavior.

The main allocator also funds complete four-fit lanes in fixed case order and
checks the original latest-start cutoff before each dispatch. This explains
why completed K2/K7 pilots did not yield main coverage and why the last funded
K4 slot remained unstarted. Those policies were explicit, but together with
late complete pricing and variable search work they delivered only seven main
attempts. They should be reviewed for coverage and completion goals before a
new schedule is committed. The unused additive grant was not a live allowance
to ignore those limits.

The K4 interval-coverage gap is now localized. The frozen
`engines/pipeline.py::stopped_intervals` obtains reference functionals only from
`references/analytic.py::exact_functionals`. That function returns an empty
mapping for `ssm_campaign_two_noises`, so all four K4 mean/quantile coverage
records have `reference: null` and `available: false`, despite available MCSEs.
The independent grid reference is used by separate descriptive accuracy checks
but is not connected to this calculation. A repair must carry numerical
reference uncertainty explicitly; it must not relabel grid estimates as exact
truth. Current K4 results therefore support their declared checks and
descriptive agreement only, with coverage still unevaluated.

## Main progress checked September 29 at 22:00 Shanghai

The existing main service is still running. Six of the original 32 slots have
started: two K4 data-A fits completed, three K0 fits timed out during candidate
search, and K4 data-B/seed-0 is running. Two more funded slots await dispatch;
24 slots remain unfunded. The full denominator is preserved in
`../hmc-ssm-main-2026-09-29/r1/inventory-live.json`.

The two completed K4 fits each assessed the one member requested by the
original design. Both passed their declared posterior checks and precision
screens after 2,000 warmup and 1,000 retained transitions per chain, with warmup
excluded. Both were within the descriptive independent-reference tolerances.
Their full inventories retain six and eight verified candidates, respectively;
the remaining candidates are explicitly unassessed by design. Two fits do not
establish calibration, a statistical ranking, or a numerical default.

All three K0 stops have the recorded reason `fit_budget_exhausted`. Each received
560.333 seconds of contention allowance and exhausted its 3,352-second worker
allocation. K0 data-A/seed-0, data-A/seed-1 and data-B/seed-0 preserve 84, 70 and
98 candidate records; two, five and four candidates are verified. Their
checkpoints record 19, 17 and 18 pending work items, respectively, with no
posterior chunks. These are incomplete searches, not completed posterior
rejections. The inherited 1.5 pilot pricing margin did not cover these main
realizations. Foreign-process detection alone cannot establish how much of
their runtime was caused by contention rather than search or compilation.

This read-only audit checked the saved receipt and assessment hashes. Its
record is `../hmc-ssm-main-2026-09-29/r1/status-audit-20260929T140041Z.json`.
The skeptical review preserves the distinction between pilot and main evidence,
process completion and posterior checks, and observed runtime and a guaranteed
budget. No new numerical work, policy change or budget extension was made.
The settled grant remains 170,503.765 GPU seconds before the active 28,952-second
main reservation; settlement is pending and nested fits must not be charged
again. The existing queue continues under the original 23:03:52 latest-start
cutoff and September 30 03:03:52 compute deadline, Shanghai time.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Record two completed K4 fits | Requested members pass declared posterior and descriptive reference checks | No reported hard veto in those members | Only two fits on one dataset; most candidates unassessed | Preserve results and continue funded original slots | Calibration or superiority |
| Retain three K0 incomplete outcomes | Search incomplete at full allocated time | Resource limit, no posterior assessment | Remaining search cost and contribution of compilation/contention | Preserve checkpoints and pending work in terminal reconciliation | Invalid target or failed posterior check |
| Continue existing queue | Active service and receipts agree | Original deadlines and reservation still bind | Which remaining slots can start and finish | Audit terminal results and settle enclosing cost once | All 32 slots completed |

| Inference status | Evidence |
| --- | --- |
| Hard veto screen | None reported in completed K4 assessed members; K0 posterior results unavailable |
| Statistically supported ranking | None |
| Descriptive-only differences | Reference errors, candidate counts and runtime |
| Default readiness | No numerical default changed or promoted |
| Next evidence needed | Remaining main assessments, full slot accounting and timeout diagnosis |

Post-run red-team note: successful K4 members cannot compensate for missing K0
outcomes. A completed development pilot was insufficient to predict K0 main
cost under the declared margin. The main campaign remains incomplete; this
does not invalidate the state-space targets or justify discarding failed slots.

## Main stage launched at 18:08:49 Shanghai

K7 completed its final continuation, costing 1,830.454 enclosing GPU seconds;
all preserved numerical evidence passed validation. Its full cumulative pilot
price is 7,157.783 seconds. All 21 verified candidates remain retained. The two
predeclared assessed members passed their posterior checks: the first used
3,000 warmup and 3,000 retained transitions per chain, and the second used
2,000 warmup and 1,000 retained transitions. Both were within the separate
descriptive reference tolerance, with no observed false favorable screen.
This is one development fit of the declared sigma-point approximation, not
nonlinear posterior calibration or an exact-model accuracy claim.

The automatic handoff succeeded. The active numerical service is now
`bayesfilter-hmc-ssm-priced-main-20260929-r1.service`, with results under
`../hmc-ssm-main-2026-09-29/r1/`. The first original K0 data-A/seed-0 slot has
started with the frozen source, GPU/XLA placement and verified memory growth.
Of 32 original main slots, one is started and none has yet completed.

The measured whole-lane allocation funds the four original K0 and four K4
slots, interleaved by their original data/seed slot. At launch, 32,043.899
seconds were available under the original compute deadline. The selected
cells reserve 28,892 seconds including per-cell overhead, plus 60 seconds for
stage closeout and shutdown. The reservation is 28,952 seconds; the settled
grant balance before it is 170,503.765 seconds. K2's 29,168-second whole lane
cannot fit after K0; K7's 43,068-second whole lane exceeds the remaining wall
allocation. These cases remain completed development evidence and explicitly
unfunded main slots. The initial 24 unfunded slots and any later unstarted or
failed cells remain in the original 32-slot denominator. The latest-start
cutoff remains 23:03:52 Shanghai even if the stage still has allocated time.

The current service's maximum runtime is 8h 2m 2s from 18:08:48, but the
per-cell cutoff can end it earlier. This is a bound, not a completion forecast.
The main result is still pending. Neither pilot successes nor queued slots
are counted as completed main evidence.

## September 29 continuation update

The six-pilot continuation finished at 16:58:09 Shanghai and settled
14,937.356 GPU seconds. K0, K2 and K4 completed their original search and
declared member workload. All assessed members passed their declared posterior
checks and were within the separate descriptive reference tolerances. These
are individual development fits, not a replicated calibration result or a
statistical ranking.

| Case | Full workload | Cumulative seconds | Verified candidates | Declared assessed members |
| --- | --- | ---: | ---: | ---: |
| K0 | Complete | 2,241.074 | 5 | 2 |
| K2 | Complete | 4,841.006 | 13 | 1 |
| K4 | Complete | 2,533.406 | 8 | 1 |
| K1 | Search incomplete at resource cap | Unpriced | 5 | 0 |
| K3 | Search incomplete at resource cap | Unpriced | 4 | 0 |
| K7 | Two resource continuations incomplete; final continuation running | Unpriced until both assessments finish | 21 after full search completion | 2 required |
| K5 | Preparation repaired; no full repaired pilot | Unpriced | Not measured | 0 |
| K6 | Preparation unresolved | Unpriced | Not measured | 0 |

The remaining repair allocation was 3,332.905 seconds. The revised plan assigns
that remainder to one final K7 continuation, including all overhead, because
it supplies the predeclared nonlinear two-member coverage. It does not reset
the six-hour pool or change numerical work. K1/K3 receive no further automatic
attempt. K7's full search is now complete with all 21 verified candidates
retained, and its two member assessments are underway.

The automatic main handoff is queued in
`bayesfilter-hmc-ssm-main-sequence-20260929-r1.service`. It waits for K7's normal
coordinator settlement, then invokes `scripts/run_hmc_ssm_priced_main.py`.
That launcher checks original inputs, source, full cumulative prices and actual
remaining time before funding whole four-slot lanes in original case order.
The obsolete per-lane convenience caps are explicitly replaced by global
funding. An 8.5-hour preview funds the original K0 and K4 lanes for 28,892
seconds including per-cell closeout. The allocation is recomputed at launch;
every cell must also start before the unchanged 23:03:52 cutoff. All 32
original slots remain in the report, including unallocated or unstarted ones.
Main remains 0/32 while K7 runs. Exact dispatch is recorded in
`main-sequence-launch.json`; main artifacts will be under
`../hmc-ssm-main-2026-09-29/r1/`.

The new main coordinator and allocation regressions pass 56 tests, with GPUs
intentionally hidden. They check ledger limits, source/runtime receipts,
workload equality, completed unfavorable outcomes, original latest-start
enforcement and normal dispatch/settlement. A missing fixture isolation option
was corrected; a real tuple/list comparison bug in serialized allocation
revalidation was fixed and tested. The failed and passing reports are retained.
Executed coordinator sources are preserved in `coordinator-source-r5/`.

The input audit also found that K3's second original dataset uses reference
resolution 161 whereas its pilot uses 321. K3 cannot borrow a whole-lane price
without resolving that workload difference; it is already unpriced because its
search is incomplete. The other completed pilot workloads match their main
lanes apart from the originally declared data and sampler-seed repetitions.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Accept K0/K2/K4 cumulative prices | Full workloads and preserved receipts verified | Assessed members pass declared checks | One price per case cannot identify runtime tails | Use unchanged 1.5 planning margin and remaining global limits | General calibration or superiority |
| Finish nonlinear K7 pilot | Search complete; 21 verified candidates | Posterior checks still pending | Whether both assessments fit the remaining allocation | Final unchanged-workload continuation only | Posterior validity from search success |
| Queue measured main lanes | 56 engineering checks pass; original inputs verified | Unpriced/workload-mismatched lanes excluded; unknown failures stop dispatch | Actual remaining time and per-seed runtime | Recompute funding after K7, enforce original cutoff per cell | All 32 fits funded or completed |

| Inference status | Evidence |
| --- | --- |
| Hard veto screen | K5/K6 original failures retained; no unresolved veto in completed K0/K2/K4 assessed members |
| Statistically supported ranking | None |
| Descriptive-only differences | Runtime, candidate counts and reference errors from individual development fits |
| Default readiness | No numerical defaults changed or promoted |
| Next evidence | K7's complete assessments and original funded main replications |

Post-run audit: shared-machine contention is not the only cost explanation.
K1/K3 have reached 95/100 candidates while still having roughly thirty pending
tasks; K7 reached 100 candidates and 200 work units before closing its search.
Finishing a broad all-candidate search is substantial work even after some
candidates verify. This phase does not end search at the first success, shorten
sampling or use incomplete costs to make funding appear adequate.
The read-only `search-work-diagnosis.json` records 46 inconclusive observations
for K1, 43 for K3 and 74 for K7, including conflicting acceptance evidence.
Recorded directional repairs and evidence extensions account for the large
work inventory. None of those three observation sets has an engineering
invalidity or hard veto, and R-hat is reporting-only in every observation.
Thus these particular search timeouts are not R-hat rejection or demonstrated
nonfinite-state failures. The counts do not separate compilation cost from
GPU contention and are not a statistical performance comparison.

## Earlier repair and diagnosis record

The campaign-allocation repair is implemented in
`bayesfilter/testing/inference_validation/campaign_recovery.py`; its bounded
launcher is `scripts/run_hmc_ssm_pilot_repair.py`. The original numerical source
is preserved at the shared-device `prepared-r5/source` snapshot. Added time is
recorded outside the numerical design, so source, seeds, candidate identities
and native checkpoint paths remain unchanged. All original process costs are
included in cumulative fit prices and charged only once in the campaign ledger.

The focused scheduling and compatibility selection passed 102 tests. Two real
CPU subprocess tests passed for K0 and K7 in 123.23 seconds: both resumed after
an injected cooperative stop, preserved completed numerical checksums, retained
verified candidates and excluded warmup from posterior output. Separate unit
tests cover exhaustion of the original hard cap. These are CPU mechanics tests
with GPUs deliberately hidden, not GPU throughput or convergence evidence.
The existing startup/preparation-recovery selection passed 43 tests in 70.83
seconds. Its three numerical implementation modules are byte-identical to the
executed frozen source. The initial test fixtures omitted required isolation
and settled-C1 fields; their failed JUnit files are preserved, and the corrected
fixtures passed. Test sets overlap where repeated; counts are not additive.
The subsequent live closeout-race repair passed 14 focused launcher tests,
including a real CPU child timeout that settles its receipt before the external
service deadline. Tested coordinator sources are preserved in
`coordinator-source-r3/`; the unchanged frozen worker contains 596 checked
Python files.

The official tuning chapter and agent reference now explain explicit campaign
allocation after a local cap is spent. The standalone chapter builds to 40
pages and the recovery paragraph on page 26 was visually inspected. Its one
unresolved cross-chapter reference is expected in this standalone build; the
full book was not rebuilt with unrelated current chapter edits.

## GPU diagnosis

The six independent resource-stopped pilots are now running in
`bayesfilter-hmc-ssm-pilot-continuation-20260929-r1.service`, launched directly
at 12:49:11 Shanghai after diagnosing and repairing the closeout race below.
K0's continued child has verified GPU/XLA and memory growth and uses the same
frozen source, design and native checkpoint paths. K0 has now completed its
entire declared workload, with all 212 preserved records verified. Continuation
took 464.420 seconds including its invocation overhead; the complete cumulative
price is 2,241.074 seconds, including 1,776.654 seconds from the original work.
Both assessed members completed 2,000 warmup and 1,000 retained transitions per
chain, with warmup excluded, and passed their declared posterior checks. All
five verified candidates remain in the inventory; the other three were
unassessed under the original two-member design. This one development fit
establishes a complete price and working recovery, not general calibration or
superiority. `k0-completed-continuation-audit.json` preserves the cost calculation.
The queue has moved to K7. The continuation checks the
shared pool, original wall deadlines and completed evidence before every new
worker. Its exact launch is in `recovery-r1-launch.json`. The earlier automatic
sequence failed closed and remains preserved in `sequence-r1.log`; it is no
longer queued and must not be restarted beside the active recovery.

`diagnosis-r1` used the original GPU 2 UUID and frozen target/data/tuning seeds,
with verified memory growth and XLA. K5's existing bounded initializer tried
epsilons 0.4204482076, 0.2102241038, 0.1051120519 and 0.0525560260. Retained
states stayed valid. The second proposal was invalid and is preserved; fresh
bootstrap passed at the final step. Mass adaptation later failed on one
nonfinite log-acceptance value at transition index 23. Preparation therefore
remained invalid. Its child elapsed time was 178.049 seconds.

K6's initializer tried 0.2427458859, 0.1213729429 and 0.0606864715. The first
proposal was invalid with valid retained states; the smaller final step passed
fresh bootstrap. Subsequent preparation exhausted the 350-second diagnostic
allowance. This is resource-incomplete preparation, not a second demonstrated
numerical rejection. The enclosing first diagnosis receipt is settled in the
existing additive ledger.

`diagnosis-r2` is the bounded follow-up. K5 enables the already documented
`preparation_max_restarts=3`, which independently checks retained states, scores,
telemetry and accepted-state consistency before discarding a failed attempt
and contracting its step ceiling. K6 retains the first diagnostic's numerical
configuration and receives up to the 1,775-second scheduling quantum. These
settings have fresh diagnostic identities; they never overwrite the original
failed pilots or supply prices for the original unmodified main design.

K5's follow-up **completed full preparation in 295.646 seconds**. One rejected
nonfinite warmup proposal triggered a checked restart; all 116 transitions from
the discarded attempt stayed excluded. The failing consumed step was
1.1892071150 in that attempt's adapted coordinates, distinct from the initial
startup step. The successful attempt applied one operational metric update and
completed the public handoff. This supports the existing optional startup plus
warmup-recovery configuration for this development case. It does not establish
candidate verification, posterior convergence or a generally suitable default.
K6's longer same-configuration diagnostic failed after 724.814 seconds on two
nonfinite log-acceptance entries in mass adaptation (first at index 27 of the
failed segment). Its bootstrap repair therefore also needs a warmup repair.
`diagnosis-r3` tests K6 with the existing three-restart option, capped at 1,775
seconds within the same pool. K6 discarded two attempts of 315 and 450
transitions after independently classifying rejected nonfinite proposals with
valid retained states. Its third attempt had completed 135 transitions when
the enclosing service reached its limit. There is no successful K6 preparation
handoff. The first two enclosing diagnoses cost 530.080 and 1,024.165 seconds;
the third cost 1,775.494 seconds. Their failures and spent time are preserved.
K5's success does not substitute for K6-specific evidence.

The third diagnosis also exposed a coordinator defect: its numerical deadline
and service deadline coincided. Systemd sent SIGTERM before the fit supervisor
could save its normal timeout receipt. The coordinator recorded
`SystemExit(143)`, settled its enclosing charge, and the waiting sequence
correctly refused the nonzero result. The repaired launcher passes an absolute
monotonic work deadline with 30 seconds reserved inside the service allowance
for reaping, validation and settlement. This uses the existing shutdown scale
as a conservative engineering reserve, reduces numerical time, and grants no
extra time. Arbitrary exit 143 remains a failure. `closeout-diagnosis.json`
preserves this diagnosis and its focused regression evidence.

After diagnosis, the six eligible resource-stopped pilots resume in the
declared order K0, K7, K1--K4. Each gets at most two additional process attempts
of up to 1,775 seconds, contingent on durable progress and remaining common
allocation. Completed unfavorable assessments are reused. A complete price must
include original and new attempts and the entire declared retained-member
workload. Historical pricing indices are preserved; continued prices are
reported separately. The main matrix remains 0/32 until those prices establish
affordability under the original remaining wall time.

The entire repair has at most six additional GPU hours inside the existing
36-hour SSM cap; diagnostics and all retries consume that same pool. The
original new-work, compute and reporting cutoffs remain September 29 23:03:52,
September 30 03:03:52 and September 30 05:03:52 Shanghai. The conservative CPU
implementation/regression charge is 1,800 worker-seconds, leaving 37,638.430
seconds from the previously recorded balance. No package installation,
scientific threshold change, new grant or C1 rerun occurred.
At recovery launch the settled additive ledger balance was 187,271.575 GPU
seconds. The recovery reserves at most 18,270.261 seconds including shutdown;
that reservation is unavailable for other work. The three settled diagnoses
used 3,329.739 seconds of the common six-hour pool. Nested fit receipts will
not be charged again when the enclosing recovery settles.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Admit explicit campaign continuation | Tests pass; K0 full GPU continuation completed with preserved evidence | K0 assessed members pass declared checks; source/design guards remain | Costs and outcomes for the other five cases | Continue K7 and K1--K4 within the pool | General posterior calibration or guaranteed recovery |
| Separate K5 numerical repair | Follow-up startup and complete preparation pass | Failed initial attempt preserved and excluded | Candidate verification and posterior behavior remain untested | Use this explicit configuration for a separately identified full pilot | Posterior validity or new default |
| Keep K6 unresolved | Two checked discarded attempts; third incomplete at service cap | No valid preparation handoff | Whether a further preparation attempt would complete and remain valid | Inspect archived attempts before a separately reviewed development pilot; no automatic fourth diagnostic | Successful preparation, posterior validity or rejection of the target |
| Repair coordinator closeout | Real timeout regression and 13 other focused checks pass | Unknown failures still block dispatch | Actual long continuation cost | Run the six preserved pilots with an earlier numerical deadline | Completed recovery before its terminal receipts |

| Inference status | Evidence |
| --- | --- |
| Hard veto screen | Original numerical failures and discarded attempts remain preserved; K6 has no preparation handoff |
| Statistically supported ranking | None |
| Descriptive-only differences | Step probes, elapsed time and candidate/member counts |
| Default readiness | No numerical default promotion |
| Next evidence | Terminal continued-pilot receipts and complete cumulative costs; separate K5/K6 full development pilots remain necessary |

Post-run red-team note: a passing short bootstrap cannot establish stability
through mass adaptation. Both cases demonstrate that limitation. Longer time
alone does not establish a numerical repair. K6 provides evidence of rejected
nonfinite proposals in two attempts and resource-incomplete preparation in its
third; neither proves that another attempt would fail or pass. Partial retained
output cannot price the complete declared workload.
