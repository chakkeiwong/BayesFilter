# v7 release continuation after 24-hour GPU grant

Recorded UTC: 2026-10-04T10:02:29.954539+00:00

The user authorized an additional 24 GPU hours (86400 seconds) for the existing v7 release goal. The scientific target, source, model families, six-value L grid, starts, data/horizons, candidate retention, repair rules, 32 independent searches per family, delivery criterion, vetoes, and nonclaims are unchanged. The earlier affordability blocker is preserved as historical evidence; this grant resumes the same plan.

## Evidence contract for the next action

- **Question:** what are the complete source-23 costs of the QR LGSSM and residual-whitened funnel searches, and does the resulting three-family point forecast fit the remaining authorized GPU budget with explicit readiness and settlement margin?
- **Baseline:** the frozen source-23 `run_hmc_v7_release_prices.py` driver and its existing complete nonlinear price (`gpu-price-serial-restored-01`).
- **Primary pass condition:** each selected price completes the full unchanged search, exports and reloads every verified candidate, reconstructs its checkpoint, and accounts for every attempted transition; the three-family forecast is then computed from those same-source/device results.
- **Hard vetoes:** source-manifest drift, wrong GPU or memory policy, missing XLA/GPU evidence, non-finite or unaccounted work, incomplete search, missing export/reload, failed checkpoint reconstruction, unknown harness code, or a price that cannot be reconciled.
- **Explanatory diagnostics:** candidate counts, repairs, per-stage timing, member inventory, allocator usage, utilization and observed contention. They do not establish delivery probability or default readiness.
- **Nonclaims:** a complete development price is not independent confirmation, posterior convergence, stationary acceptance, speed superiority, or public/default promotion.
- **Artifact:** fresh price root under this release artifact directory, with source/configuration/runner hashes, manifests, result/progress JSON, logs, and later audit records.

## Skeptical pre-run audit

The current plan survives the required audit. The source-23 driver is the numerical authority and rejects modified source files. It preserves the original development seed per family and the full six-value primary grid, refinement, M100 cap, four starts, repairs, export/reload and checkpoint requirements. The QR and funnel prices are necessary because historical prices use different source closures and cannot be mixed into the current forecast. The command's 6,200-second enclosing budget covers two 1,800-second searches and two 1,200-second closeouts; the new allocation leaves 109846.965 GPU seconds inside the release ledger before pricing, plus 9278.015 untransferred authorized seconds.

The main failure mode is a resource or harness failure being mistaken for a numerical price. The runner records resource deferral, timeouts, unknown return codes and partial delivery separately; any incomplete price remains unsuccessful and is charged. A busy GPU may explain timing but cannot turn an incomplete search into a complete price. We will not use historical source-19/source-21 costs, partial members, or a reduced denominator. No confirmation launch is justified until all three current-source prices and a margin-aware affordability review pass.

## Bounded audit preparation and serial host-cost localization

While the two GPU prices run, reserve at most 120 CPU worker seconds for a diagnostic on a saved, completed source-23 nonlinear work item. Compare plain and cProfile-instrumented reconstruction of the same ordered raw trials through the unchanged serial implementation. The question is which calls account for host analysis cost, not whether CPU timings predict GPU speed. The comparator is the same source, inputs and CPU execution in both calls; require exact full-record equality before using the profile for localization. Preserve the input hash, profile, source, output equality and enclosing receipt. A mismatch invalidates this diagnostic and triggers inspection; it does not modify or stop the independent frozen-source prices.

This is an explicit CPU reference/debug exception: GPUs are hidden before import; no native sampling, tuner authority, concurrency, changed threshold, omission or code adoption is permitted. The 120-second ceiling is a convenience allowance under the remaining CPU allocation, not a measured price. Plain/instrumented equality is the engineering pass criterion; function times are explanatory only. No speed ranking, forecast replacement or release follows from this profile. The skeptical review passes because it uses saved evidence without altering live source or numerical inputs and cannot change the campaign denominator.

The terminal file auditor was also reviewed before use. A candidate-local invalid trial is not a campaign-wide failure when the completed procedure records it and independently verifies another member; the audit checks the complete valid/invalid and attempted-work accounting, not an invented zero-invalid-candidate requirement. Its output remains a saved-evidence audit, not new numerical validation.

The saved-work CPU diagnostic passed full-record equality for 32 original trials, with plain, cProfile and explicit-timer calls. Both receipts consumed 16.131 CPU seconds; unused reservation was released. cProfile omitted warmed frames, so its incomplete cumulative call table must not establish named-function attribution. Explicit timers counted five `_all_close` metadata assertions per trial: 0.083 seconds of a 0.340-second instrumented reconstruction. These inclusive CPU timings nominate a narrowly scoped validation-dispatch question only. They do not prove a sufficient saving, permit CPU relocation of numerical reductions, or replace any GPU price. No runtime code changed.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Keep the CPU diagnostic for localization | Plain and explicitly timed full records agree | Incomplete cProfile attribution disclosed; previous CPU-placement mismatch remains rejected | Share of these metadata calls in actual GPU execution | Finish the source-23 prices, then decide whether a bounded cost repair is needed | GPU saving, new numerical support or release |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | No record mismatch in this CPU diagnostic; no new native candidate evaluated |
| Statistically supported ranking | None |
| Descriptive-only differences | Cold, warm and instrumented CPU component times |
| Default-readiness | Unchanged; no numerical implementation change |
| Next evidence needed | Complete source-23 prices and margin-aware affordability audit |

Post-run red team: the primary limitation is CPU versus GPU placement. An apparent CPU timing advantage cannot establish a valid GPU optimization; the earlier whole-CPU relocation already changed summaries. A prospective local repair would require exact saved-record, boundary/mutation, actual native and recovery checks followed by new complete prices.

## Conditional serial validation-dispatch diagnostic

The complete source-23 QR price costs 2,377.165343 GPU seconds and retains/reloads all 22 members; nonlinear costs 2,227.336864 seconds. Their descriptive 32-per-family subtotal already exceeds the remaining allowance before funnel and margins. This does not prove a runtime lower bound, especially because a new GPU process joined during QR closeout. Finish the already-running funnel price to obtain the full matrix; do not substitute interrupted runs, historical sources or a smaller denominator.

The new narrowly scoped repair hypothesis is to replace only repeated eager `_all_close` **metadata assertions** with one stable TensorFlow graph that evaluates the same asymmetric predicate `all(abs(actual-expected) <= atol + rtol*abs(expected))`. Preserve tensor conversion, broadcasting, tolerances, raw trial reductions, score calculations, native HMC, device placement and every validation. This is distinct from the rejected whole-CPU placement, flattened trial reductions and concurrent host execution. No runtime edit or adoption is authorized by a diagnostic result alone.

First spend at most 120 CPU worker seconds on an isolated reference comparison of the unchanged helper with XLA and graph-only candidate predicates, then compare full saved-trial records under the original CPU reference placement. GPUs remain intentionally hidden. Stable input signatures must support the actual scalar/vector shapes; use the same source-23 helper and a deterministic boundary inventory. Test exact equality, one-ULP neighbors of the existing asymmetric tolerance, broadcasting, empty arrays, non-finite values and invalid shapes. The existing `rtol/atol` values are inherited engineering assertions, not newly calibrated numerical defaults. A changed Boolean, changed full record, skipped validation or unsupported shape rejects that variant; do not loosen tolerances. XLA is the preferred candidate. Graph-only execution is a diagnostic comparator for identifying fusion/compilation differences, not an adopted production exception.

The research question is whether dispatch can be reduced without changing any checked output. Baseline is the unchanged source-23 serial predicate on identical inputs; exact predicate and complete-record equality are the promotion screen for a later GPU diagnostic. Timing is explanatory only, and one component cannot establish a sufficient full-search saving. Use the original first nonlinear measurement work and save the input/source hashes, all comparisons, trace counts and enclosing receipt under a fresh `validation-graph-cpu-grant-*` directory. The 120-second cap is a convenience diagnostic allowance drawn from existing CPU funds.

Skeptical audit passes for this diagnostic, not implementation or confirmation. It introduces no mathematical reduction rewrite, parallel context, candidate selection, new seed or policy threshold. XLA may fuse arithmetic or change edge predicates; exact near-boundary controls are designed to expose that before any adoption. CPU agreement would still require original-placement trusted GPU agreement, native/public-route and recovery tests, a fresh source assembly, complete new-source prices and an affordable confirmation before release. A failed variant is rejected; it does not invalidate the tuner or stop the running unchanged price.

The CPU comparison completed in 5.792 enclosing seconds. Both variants matched
all 218 predicate cases, including invalid-shape error types, and all 32 complete
saved trial records; each graph traced once. Warm baseline assembly took 0.292
seconds, XLA 0.303, and graph-only 0.301. These single descriptive measurements
provide no CPU speed advantage. They permit a later original-placement GPU
comparison, but do not nominate a runtime change by themselves. The enclosing
receipt is charged and the unused CPU reservation released. The source-23
funnel price continues independently.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Keep predicate compilation as a diagnostic hypothesis | Exact CPU boundary and full-record agreement passed | No changed CPU output observed | GPU boundary behavior and complete-search cost | Finish pricing before a bounded GPU comparison | Runtime adoption, sufficient saving, or release |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | CPU comparison passed; GPU variant not checked |
| Statistically supported ranking | None |
| Descriptive-only differences | Single CPU assembly timings show no gain |
| Default-readiness | Unchanged |
| Next evidence needed | Complete three-family prices and original-placement GPU comparison if warranted |

Post-run red team: reducing dispatch for a small predicate may have negligible
effect on the complete procedure. A matching Boolean on this bounded CPU
inventory does not establish universal float64 equivalence or GPU behavior.

## GPU predicate comparison after terminal pricing

Reserve at most 180 enclosing GPU seconds for the predicate comparison, after
the current source-23 price worker exits. Use the same selected GPU, verified
memory growth and original source. The worker cap is 170 seconds; the extra ten
seconds are a convenience settlement allowance. Compare original, XLA and
graph-only predicates on the 218 unchanged cases, then compare complete saved
trial records with both the archive reader's placement and the live native
GPU placement. Each variant must match its own placement's baseline exactly;
cross-placement equality is not assumed. Three alternating-order repetitions
give descriptive timing only. Graph-only remains a diagnostic exception.

This extension retains the CPU diagnostic's question, comparator and vetoes.
It tests whether GPU dispatch explains why CPU timing was uninformative. No
sampler, target, seed, threshold, public API or runtime file changes. Passing
predicates without passing complete health/score records is insufficient. A
boundary mismatch rejects that variant. A small saving alone does not justify
new prices. Artifact: `validation-graph-gpu-grant-01`, including memory policy,
devices, input/source hashes, all timings/parity results and enclosing receipt.
The skeptical audit passes for this bounded diagnostic: it uses the actual
unchanged helper as comparator, checks both relevant placements, preserves
every failure, and cannot supply confirmation or release evidence.

## Complete prices and bounded resource-cost localization

All three source-23 prices and the independent file/accounting audit passed.
QR retained 22 members from 42 candidates, nonlinear 19 from 45, and the
residual-whitened funnel 17 from 32. The audit checked all 31,072 complete valid
trials, their source/device identity, raw tensor checksums, unique streams,
exports/reloads, checkpoints and attempted-work charges. The enclosing QR/funnel
queue consumed 3,277.527624 GPU seconds, charged once. The audit consumed
16.000530 CPU seconds. These are full development searches, not confirmation.

| Family | Enclosing allocated price (seconds) | Native calls | Trial count |
| --- | ---: | ---: | ---: |
| QR LGSSM | 2,377.167430 | 360.584627 | 11,968 |
| Nonlinear SSM | 2,227.336864 | 628.467880 | 10,816 |
| Residual-whitened funnel | 900.360195 | 27.382533 | 8,288 |

The complete point forecast is **176,155.663628 seconds (48.93 GPU hours)**.
After settled diagnostics there are 115,823.723440 authorized GPU seconds
including the untransferred balance: a 60,331.940188-second shortfall before
readiness, development and settlement margins. A cost repair must reduce the
three-family sum by more than 34.25% to fit before margins. This arithmetic is
descriptive, not a lower bound on another resource regime.

The GPU predicate comparison passed all boundaries and complete records on
both archive and native placements, using verified memory growth. It consumed
23.729676 GPU seconds. Across three alternating repeats the timing ranges
overlap and the archive XLA variant is descriptively slower. No runtime repair
is adopted and this component cannot fund confirmation.

There is a concrete unresolved resource confound: a foreign process joined
the selected GPU during QR closeout, whose reload took 922.675 seconds. The
later funnel reload took 282.678 seconds with fewer trials. Current saved-trial
analysis is also much cheaper than a previous unchanged-source diagnostic.
These observations do not isolate load from target or stage differences.

Before either another full price or a runtime edit, reserve **240 GPU seconds**
for a preselected replay diagnostic on the first L=25 measurement batch of each
completed QR and nonlinear price. Use the unchanged source-23 native batch
runner, exact original 32 seeds, starts, epsilon, 68-transition trajectories,
selected GPU and original four-CPU affinity (8--11). Compare every raw sample,
trace and assembled full record against the saved batch. Run one cold and two
warm repetitions, recording GPU process/utilization snapshots, CPU pressure,
native call time and assembly time. The worker cap is 230 seconds, leaving a
convenience ten-second settlement margin. Do not deliberately create contention
or affect other users' processes.

Question: does current resource-conditioned execution materially differ from
the expensive complete-price observation on identical actual native work?
Exact reconstruction is the engineering pass condition; any mismatch,
source/device/growth failure or time limit stops the diagnostic and prevents
cost interpretation. Timing is explanatory and nominates a next investigation
only. It is neither independent sampling evidence nor a replacement full price.
A small component saving does not justify repricing. A large change still
requires an explicit resource policy and complete new costs with all earlier
results retained, before any confirmation. Artifact:
`source23-current-cost-probe-grant-01`.

Skeptical audit passes for this bounded diagnostic. The batch-selection rule is
fixed before inspecting new timings, both state-space families are included,
the comparator is the actual saved native work, and unchanged CPU affinity
prevents an unmeasured placement change from posing as reduced device load.
The convenience budget comes from the existing grant. The strongest alternative
explanation is transient workload: a faster short repeat would not guarantee
that a long campaign remains uncontended. No denominator, threshold, default,
method or source change is permitted by this diagnostic result.

The original native-batch diagnostic passed every raw tensor and complete
record comparison across all six calls, costing 36.682091 GPU seconds. On the
unchanged nonlinear L=25 batch, the recorded original warm native call took
10.103519 seconds; current warm calls took 0.876085 and 0.881480 seconds.
QR's original 1.219598 seconds compares with current 1.425247 and 1.296149.
Both warm assembly pairs took 0.56--0.64 seconds for 32 trials. No foreign
compute process appeared on the selected GPU in the seven snapshots. This
strongly nominates resource-conditioned repricing of the complete procedure;
it does not isolate a causal workload effect or support a speed ranking.

## One complete resource-observed price repeat

Run exactly one new full three-family price queue, in the original QR,
nonlinear, funnel order, using the unchanged source-23 driver, original seeds,
all settings and original CPU affinity. Preserve both complete price vectors;
do not pick each family's fastest attempt. The question is whether the full
procedure, under explicitly observed resource conditions, is sufficiently
cheaper to justify a funded confirmation design with a corresponding bounded
resource-admission policy. The new price remains development evidence.

The observer waits up to 60 seconds before launch for at least 4,096 MiB free
and no other compute process on the selected GPU. This is a bounded admission
wait, not exclusive ownership; it never signals another user's process. Memory
provenance is inherited from the existing price preflight; the 60-second wait
and ten-second observation cadence are convenience operational hypotheses.
After launch, record device utilization/free memory, process ownership and CPU
pressure every ten seconds. Unknown ownership is reported conservatively. Such
snapshots cannot prove continuous absence of contention. Do not interrupt an
otherwise valid price merely because another user later starts a process.

Use `observe_complete_price.py` and fresh root
`gpu-price-resource-observed-grant-01`; the unchanged driver's complete files
go in its `prices/` child directory. Reserve **9,400 enclosing GPU seconds**:
the original three 3,000-second process caps, 9,300-second queue ceiling,
60-second readiness allowance and a 9,370-second observer cap with settlement
headroom. Charge the observer receipt once, including its child queue and wait;
never debit the child prices separately. Terminal provenance audit must link
the enclosing receipt to the child result and include the enclosing overhead
when forecasting. Reserve 60 existing CPU seconds for focused observer checks.

Primary pass criteria are the unchanged full-search, all-member replay,
checkpoint and work-accounting requirements. Source/device/memory-policy or
evidence failure vetoes the price; observer corruption or an exhausted enclosing
cap stops its own process group. A foreign workload is explanatory, not an
automatic numerical rejection. No faster component timing replaces these gates.

Before adopting a cheaper cost forecast, compare the **whole new vector** with
the preserved older vector and its resource observations. Use the new complete
enclosing cost, explicit readiness allowance and a predeclared cost margin;
show sensitivity to the earlier slower regime. An unaffordable price, a missing
family or lack of a credible bounded admission/recovery policy leaves confirmation
unlaunched. Never shrink 96 independent slots or treat a wait/timeout as success.
No independent confirmation or release follows from this repeated development
seed; no numerical/runtime implementation changes are made.

Skeptical audit passes for this one bounded repeat. Unlike a blind retry, exact
native work revealed a large unresolved cost change, while the same QR control
did not. Measuring a whole fixed-order vector avoids selecting favorable
per-family minima. All earlier costs remain charged, the new reservation fits
the existing grant, and the scientific question, targets and criteria are
unchanged. The principal risk remains future shared load; any confirmation
proposal must state that risk and charge all waiting/failed work.

Before seeing the repeated complete vector, use a 20% cost reserve as a
**convenience budget hypothesis**, with 0%, 10%, 20%, 30% and 50% sensitivity
rows. It is not a statistical confidence interval or evidence that future seed
variation fits. Add the full 96 times 60-second readiness allowance and a
600-second terminal GPU allowance; both are explicit operational ceilings,
not sampler settings. Any apparent fit must use the remaining ledger after
settling the price and diagnostics. Report the original slower vector under
the same allowances. If the 20% scenario does not fit, do not silently reduce
the reserve or select faster family-wise prices. If it fits, a reviewed bounded
resource-admission implementation and its regression tests are still needed
before freezing confirmation. All actual waits/timeouts count and hard total
budget enforcement remains necessary because these prices give no runtime
guarantee. Focused observer checks passed five cases before launch; the added
post-launch-contention check verifies that another user's arrival is recorded
without interrupting a numerically valid price.

## Bounded resource-admission repair while the price runs

Code inspection confirms a specific supervisor gap: its existing readiness
check accepts ample free memory even when another compute process occupies the
selected GPU. Add an optional, explicitly frozen
`require_no_foreign_compute` Boolean to the private confirmation design's
readiness policy. Preserve old designs' memory-only behavior when omitted.
When enabled, query compute-process identity before TensorFlow import and wait
within the existing charged wait cap until the selected GPU has no such process.
Other GPUs do not veto readiness. Missing/malformed telemetry fails closed;
late probe completion cannot bypass the deadline. Do not poll or interrupt
foreign workers after admission, promise exclusivity, change seeds or drop a
deferred slot. Existing process/total caps and full denominators remain binding.

This is engineering support for a prospective resource-conditioned budget,
independently useful even if the new prices remain unaffordable. It changes the
external confirmation supervisor and its tests only. The frozen sampler/target
source-23 remains unchanged; it does not contain this supervisor. Freeze the
supervisor's exact file/hash with any later design as the existing launcher
already does. Reserve 90 CPU seconds for focused regressions, from existing
funds. Test ample-memory contention/recovery, persistent contention, other-GPU
processes, malformed/duplicate/missing identity, absent required probe, and a
probe that finishes after the deadline, plus all existing supervisor/reporting
checks. No independent confirmation launches until full pricing and affordability
pass.

Skeptical audit passes: memory availability is not throughput evidence, so an
optional process-aware admission policy addresses a demonstrated limitation.
It cannot guarantee later load, and the plan makes no such claim. Backward
design behavior is preserved; failures consume their original slot and budget.
This is a harness repair, not a numerical policy/default change or permission
to alter another agent's workload.

The optional admission repair passed 154 distinct focused supervisor, pricing
and reporting tests (152 in the combined run, followed by two additional actual
worker/import-boundary cases in the 50-test supervisor rerun). The two receipts
cost 5.750422 CPU seconds; unused reservation is released. The worker cases
verify no TensorFlow import under persistent contention and import only after
the selected device clears. Legacy memory-only designs retain their behavior.
The exact checked supervisor and tests are preserved separately in
`confirmation-resource-supervisor-01`, with hashes and numerical source-23
identity. These tests overlap earlier counts; do not add both suites wholesale.
This closes the optional pre-import admission gap, not post-admission workload
uncertainty or confirmation affordability.

Terminal audit for the repeated vector will use the actual enclosing observer
receipt, verify that no nested price charge was added, and allocate all enclosing
overhead once in the forecast. It will also compare each original-seed candidate
inventory and raw sample/trace/seed/work record against the first complete
source-23 vector, ignoring timing fields only. A difference triggers inspection
before cost attribution; it does not silently become a claim about load. The
original auditor source has been preserved beside its prior result before this
extension. Use at most 120 existing CPU seconds for this saved-file audit after
the price is terminal and charged; no further numerical computation is involved.

A read-only storage preflight measured 17.681 GB of logical evidence in the
three completed family directories, implying about 565.794 GB for 32 searches
each at those observed sizes. The filesystem snapshot had about 1.05 TB
available. This fits the point projection but is not a guarantee under different
search lengths or other users' disk growth. Recheck available space before a
long confirmation launch and monitor it during the campaign. Preserve all raw
evidence; no deletion, compression migration or omission is authorized by this
capacity check. Result: `confirmation-storage-check-grant-01/result.json`.

## Terminal repeated price and host-affinity diagnosis

The resource-observed queue completed in 3,657.258328 enclosing GPU seconds,
charged once. Its complete allocated prices are 1,468.070101 seconds for QR,
1,319.514584 for nonlinear and 869.673643 for the residual funnel. Every one of
the 31,072 raw trials matched the original source-23 vector in samples, trace,
seed, work and count. Candidate inventories, states, source, memory policy,
XLA/device evidence, exports/reloads and accounting passed. The independent
audit consumed 40.874101 CPU seconds. There were 361 resource observations and
no observed foreign or unidentified compute process on the selected GPU.

The full 96-slot point forecast is 117,032.266481 GPU seconds. With the
predeclared 20% reserve, 5,760-second readiness allowance and 600-second terminal
allowance, it is 146,798.719777 seconds. The settled authorized remainder,
including the explicitly untransferred balance, is 112,129.783021 seconds.
The gap is **34,668.936756 seconds (9.63 GPU hours)**. Both whole-vector
forecasts and all five margin sensitivities are retained in
`resource-price-affordability-grant-01/result.json`. Confirmation remains
unlaunched; the faster repeat does not by itself fund it.

A trusted host inspection identified overlapping CPU affinity: our original
cores 8--11 share physical cores with two busy workers pinned to
`[8,9,136,137]` and `[10,11,138,139]`. A two-second snapshot found cores 32--35
and their sibling threads nearly idle on the same socket. This nominates a
placement diagnostic, not a causal attribution or an exclusive CPU reservation.
No other process is changed. Snapshot: `host-affinity-snapshot-grant-01`.

**Evidence contract.** Compare original affinity 8--11 with candidate affinity
32--35, using the unchanged source-23 first L=25 measurement batch, original
32 seeds and 68 transitions of each of the three completed model families.
Run fresh workers in the fixed order original, candidate, candidate, original;
each family has one cold and two warm calls. Apply affinity before TensorFlow
import so its threads inherit it. Keep the selected GPU, memory growth, TF
threads 2/1, starts, dtype, XLA, L, epsilon and all numerical code unchanged.
Require exact saved raw samples, traces and full trial records in every call.
Mismatch, missing provenance, changed source, GPU/memory failure or exhausted
diagnostic cap invalidates cost interpretation. Timing and sampled CPU/GPU load
are explanatory only; this cannot qualify a candidate, rank samplers, replace
a complete price or establish future capacity. Save every arm and its receipt
under `host-affinity-probe-grant-01`.

Reserve at most **400 GPU seconds** including all four workers and settlement,
with 90 seconds per worker. These are convenience diagnostic ceilings within
existing authorization, not measured costs or numerical policy. Failures are
charged; no automatic extra arm or favorable timing selection is allowed.
Only a material consistent effect across the two arm orders can nominate a
fresh complete price. A new price must still preserve the whole vector and
pass exact evidence comparison and the unchanged affordability rule.

**Skeptical audit:** the previous observation excluded only GPU-process
contention, while substantial work runs on host CPUs. The alternative affinity
uses the same CPU count and socket and does not alter another workload. Fresh
processes avoid falsely claiming TensorFlow worker threads follow a main-thread
affinity change. Reversed arm order reduces time-order confounding. Exact
saved-evidence checks detect placement-dependent numerical differences. A
short batch effect may not transfer to full reload or checkpoint cost, so it
only nominates, and cannot substitute for, a complete price. This bounded
diagnostic passes review; confirmation is still unfunded.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Retain both valid full-price vectors; investigate CPU placement | Repeated full procedure and exact original-seed audit passed | Affordability fails by 9.63 hours including margins | Host competition and independent-seed cost variation | Bounded paired-affinity native replay | Release, posterior correctness, or runtime guarantee |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | No numerical or artifact veto in the repeated price; funding veto remains |
| Statistically supported ranking | None |
| Descriptive-only differences | Complete runtime declined under changed observed resource conditions |
| Default-readiness | Unchanged; source-23 remains the numerical authority |
| Next evidence needed | Exact paired-affinity replay, complete pricing if nominated, then all 96 independent searches |

Post-run red team: the largest uncertainty is future workload and seed-dependent
search length. A matching same-seed repeat does not establish independent-seed
delivery reliability. CPU-affinity overlap is a plausible explanation, not yet
a demonstrated sufficient repair.

The first affinity diagnostic deferred before TensorFlow import because a
different campaign had started PID 2090115 on the selected GPU. Its 0.071332
seconds are charged and the unused reservation released. No arm ran and no
numerical result exists. The same bounded diagnostic can retry in a fresh
directory after that workload exits; it must not interpret this deferral as a
tuner failure or move/delete another user's work. The current command's own
declared duration is 1,470 seconds; this is an observation, not permission to
interrupt it or a guarantee of its exit time.

## Affinity diagnostic closed; confirmation funding remains unresolved

After the other worker exited, the fresh `host-affinity-probe-grant-02` attempt
completed all four predeclared arms in 177.608444 GPU seconds. All 36 native
calls and 1,152 full trial-record comparisons matched the saved source-23
evidence exactly. Memory growth, device, CPU affinity and source provenance
passed; no foreign GPU process appeared in the sampled inventories.

Both candidate-affinity repetitions were slower in warm record assembly for
every family. Their two-call means were 0.809--0.823 seconds for QR,
0.822--0.842 for nonlinear and 0.819--0.835 for the funnel, against original
affinity means of 0.594--0.612, 0.581--0.652 and 0.646--0.666 respectively.
Native sampling times overlapped. This does not support the proposed placement
repair. Keep original affinity; do not spend another complete price on it.
The comparison is in `host-affinity-probe-grant-02/comparison.json`.

The two diagnostic attempts together consumed 177.679775 GPU seconds, within
their 400-second total allowance. All reservations are released. The release
ledger has 102,674.087813 GPU seconds remaining, plus 9,278.015433 previously
authorized seconds outside the allocation: **111,952.103246 seconds (31.10
hours)** in total. The unchanged 20%-reserve design requires 146,798.719777
seconds (40.78 hours). Its current funding gap is **34,846.616531 seconds
(9.68 hours)**. These are measured-price scenarios, not guaranteed future costs.
The CPU release allocation has 440.205186 seconds left, with its earlier
48,936.183646-second unallocated authorization unchanged.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Reject the proposed affinity change | Exact replay passed, useful cost reduction did not | No numerical invalidity; cost nomination failed | Whether another independently justified repair can lower full cost | Preserve original placement and completed evidence | No claim that quieter cores are universally slower |
| Keep release pending at P4 | 96-search confirmation has not run | Funding requirement exceeds authorization by 9.68 hours | Seed variation and future machine load; 20% reserve is a convenience hypothesis | Fund the unchanged design or validate a concrete adequate cost repair before launch | No independent delivery claim, posterior claim, or release |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | Native replay and full records pass; funding veto remains |
| Statistically supported ranking | None |
| Descriptive-only differences | Four-arm native/assembly timings; two full development price vectors |
| Default-readiness | Unchanged; no sampler, policy or CPU-placement change adopted |
| Next evidence needed | Affordable frozen 96-slot independent confirmation, then terminal release review |

This result rejects one resource-repair hypothesis, not the tuner or research
direction. No current numerical or artifact failure was discovered. The next
planned release phase is independent confirmation, whose full budget does not
fit. No tested cost repair currently closes that gap, and another speculative
full price would reduce the remaining funds. All three families, starts, data,
horizons, primary L values, candidate cap, criteria and 96-slot denominator
remain unchanged. The goal is unresolved; no GPU run remains active.

Post-run red team: core placement is not a sufficient account of machine
workload. These few repeats cannot establish a universal timing ranking. The
remaining estimate relies on one full development seed per family under sampled
resource conditions, with unknown future seed/runtime variation. Neither exact
replay nor more favorable development costs replace independent confirmation.
