# M27 sibling assessment and GPU complete-fit checks

Status: bounded M27 execution complete. The independent terminal audit passes
for two CPU complete-fit identities and both GPU process pairs. All workers
exited normally; failures and duplicate work are charged. General statistical
coverage and default readiness remain open.

The optional validation member rule now assesses a predeclared finite set of
verified members, retains every verified tuning candidate and reports each
ordinal slot with the complete-fit denominator. It rejects a changed selection
on resume, preserves shortages and continues to the next selected member after
a numerical posterior stop. R-hat, ESS and MCSE remain posterior-only. Public
tuning authority and defaults are unchanged.

The implementation also records GPU placement/device details and TensorFlow
allocator current/peak bytes at complete-fit boundaries. A fresh process owns
a complete fit; its numerical outputs and ordinary exit are checked separately.

## CPU complete-fit evidence

Two new rotated-Gaussian ordinary fits used native broad tuning and the
`shortest_verified_l` rule, selecting the first candidate ID within the two
smallest distinct verified L values before any posterior draws. They generated
100 candidate records each and retained 23 and 21 verified candidates. L=3 and
L=4 were assessed in each fit; the other 40 remain `unassessed_by_design`.

| Root seed | L | Epsilon | Warmup/chain | Retained/chain | Largest declared MCSE | Posterior checks |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 2026092281 | 3 | 0.8372331050 | 10000 | 30000 | 0.04116 | Pass |
| 2026092281 | 4 | 0.7512949258 | 10000 | 30000 | 0.04245 | Pass |
| 2026092282 | 3 | 0.7611297039 | 10000 | 30000 | 0.04613 | Pass |
| 2026092282 | 4 | 0.7448202639 | 10000 | 30000 | 0.03921 | Pass |

Each selected member also has an independent 2000-warmup/60000-retained fixed
arm. All candidate inventories, independent verification receipts, saved tensor
checksums and checkpoint bundles passed the saved-evidence audit. Warmup is
excluded from estimates. The duplicate replay matched all 168 tensor files per
fit exactly and is not an additional independent replication.

These are four successful member assessments in **two independent fits**, not
four or eight independent fits. The interval for one stopped x mean missed its
analytic truth. Four fixed-arm intervals also missed their truths. Those
observations are retained; two fits cannot establish coverage, failure rates,
member superiority, estimator superiority or default readiness. Each slot's
planned denominator is one in its source design, and the result note reports
the two distinct fit identities without pooling sibling intervals.

## Confirmed repairs and deviations

The generated suite mistakenly set warmup maximum 10000 rather than the planned
30000. All four members passed at the first 10000 check, so no observed path
used the unimplemented extension. The original design is preserved, and these
results do not test readiness extension through 30000. The next phase requires
an automatic exact comparison of the prose count table with every resolved
design before launch.

The top-level report initially said `incomplete` even though both slot summaries
were complete. Its reused fixed-comparator summary sorted an unassessed member
before selected members. Filtering out deliberately unassessed candidates fixes
both comparator and interval selection; a regression puts an unassessed ID
first to expose the error. The corrected CPU summaries are saved separately
in `m27-r1/cpu-audit-r1/result.json`; original run results are unchanged. GPU
source-r2 includes the repair. No numerical tuning or posterior result was
altered to repair the reporting label.

`rotated-sibling-r1` failed before computation because its CLI output path was
the meter's already-created directory. A fresh nested `fits/` root repaired
that launch. The r2 worker was then wrongly declared interrupted from a sandbox
PID view; it was actually still running, and r3 duplicated it. Both exited
normally and both costs are charged. Their overlap stayed within two numerical
workers. Terminal receipts and trusted host process checks now establish exit.

## GPU compatibility and cost

Both Gaussian executions used the same frozen design, source-r2, seeds, target,
starts, GPU 1 and numerical settings. The persistent run took 648.78 seconds,
the isolated run 651.47 seconds. All 26 verified candidates, numerical receipt
contents and 84 tensor files match exactly. Both selected the same L=5 member,
which passed its posterior checks after 30000 warmup and 4000 retained draws.
Neither paired replay is an independent confirmation fit.

The beta-binomial persistent and isolated runs took
587.24 and 596.09 seconds. Their
20 verified candidates, numerical receipts and 94 saved tensor files match
exactly. Both selected the same L=3 member at epsilon 1.4751102943865118 and
passed after 2000 warmup and 5000 retained draws. Hardware identity, runtime
settings, memory policy and normal exits also match. Across the two GPU designs,
46 distinct verified candidates remain retained and two selected posteriors
were assessed; the paired executions are replays of those two fit identities.

The runtime records TensorFlow 2.20.0, TFP 0.25.0, XLA enabled, TF32 enabled,
RTX 4080 SUPER, trusted execution, verified memory growth before logical device
initialization, and actual GPU tensor placement. The persistent Gaussian peak
TensorFlow allocator count is 39106048 bytes; process RSS includes about 6 GiB
of host graph/runtime state. Memory growth is not a hard cap. Normal exit
releases process-local caches, but one fit per process cannot characterize an
arbitrarily long persistent process.

The Gaussian pipeline spent about 496--499 seconds in native tuning, including
preparation, and about 54 seconds in its sequential posterior controller.
These timings include compilation. They nominate tuning-stage profiling and do
not establish that any backend or estimator is faster. At the larger measured
Gaussian GPU cost, even 384 Gaussian fits alone would take about 69.49 GPU
hours, before retries or a second model. The remaining campaign budget cannot
support the original adequate all-model confirmation at that price.

## Tests, documentation and provenance

The terminal validation/documentation inventory contained 303 tests. Its first
bounded segment emitted 276 passing test marks and then hit a 300-second cap;
the exact remaining 27 IDs all passed in the second segment. The timeout is
preserved, and no single uninterrupted full-suite success is claimed. Earlier
integration and focused runs overlap those tests and are not added to inflate
the terminal count. Focused Gaussian and rotated-Gaussian public-path tests
cover a failed first posterior, continued second posterior, retention, fixed
comparators, saved restart and changed-selection rejection. Missing fits,
slot shortages, failed processes and unassessed-ID ordering have regressions.

The official chapter in `docs/chapters/ch21b_hmc_tuning_interfaces.tex`, the
agent interface reference and validation README describe the same optional
rule and denominator. `docs/main.pdf` was rebuilt with its bibliography, without
undefined citations/references; the changed PDF page 432 was visually inspected.
No separate competing guidebook was introduced.

The base commit is `0643b0adc`, already merged and pushed. CPU source-r1 is the
clean integrated package plus the M27 owned modules. GPU source-r2 differs only
in the confirmed summary filter. Frozen snapshots, exact commands, source
hashes, seeds, device settings, wall times, normal child-exit receipts and
artifact paths are in `m27-r1/`. Raw draws and complete source trees stay local;
compact manifests/audits identify their checksums. The original M27 plan and
execution note state the unchanged scientific contract and phase ceilings.

## Decision and interpretation

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Keep optional sibling assessment | Two full CPU fits and focused regressions pass; all 44 verified candidates retained | No health, archive or retention veto for the assessed members | Only two independent fits, cap extension untested | Use explicitly declared slots in the next supplied-map study | Universal short-L mixing or default selection |
| Keep GPU fit isolation optional | Both Gaussian and beta-binomial pairs have exact numerical/tensor parity and normal exits | No device/memory/exit veto | One fit identity per model; repeated process costs still descriptive | Use bounded children; profile confirmed cost bottlenecks | Arbitrary model/backend compatibility |
| Defer large confirmation | Adequate-denominator pricing exceeds available budget | Cost veto for that inventory | Future targeted cost repair may change price | Preserve criteria; profile before redesigning or funding confirmation | Coverage from pilot success |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | No assessed CPU or GPU health or artifact veto; failed launches and timeout remain visible |
| Statistically supported ranking | None |
| Descriptive-only differences | Member MCSE, interval misses, timings, graph counts and memory |
| Default readiness | Not established; optional diagnostic rule only |
| Next evidence needed | Adequate fresh coverage, supplied residual-map checks, global quantities and exact consumer reference |

The sanity comparators are exact model functionals, separate fixed-count arms
and matched persistent-process GPU execution. They are evaluated by model and
member. They do not establish superiority over a complete cheap-sampler baseline
set; machine-readable heuristic dominance remains `not_established`.

The strongest alternative explanation for apparent posterior delivery is that
these seeds/members were favorable or missed rare tail behavior. Adequate fresh
fit coverage or a failed independent stress test could overturn an optimistic
interpretation. The weakest evidence is the small development inventory; exact
replay demonstrates reproducibility and process isolation, not scientific
calibration. A member failure remains a repair trigger, not rejection of the
HMC tuning program.

M27 charged **3118.21 CPU and
2543.58 GPU worker-seconds**, below its 3600/4800 ceilings.
This includes failed tests/launches, the duplicate CPU replay, test timeout,
document build, audits and conservative unmetered-diagnostic allowances.
Nested child receipts are not charged twice. The terminal ledger leaves
**76517.20 CPU seconds (21.25 hours)** and
**82259.94 GPU seconds (22.85 hours)**
under the existing campaign authorization, with no active reservation.
The ledger is `m27-r1/reconciliation-terminal.json`; the full independent audit
is `m27-r1/terminal-audit-r1/result.json`.

The [next-phase program](bayesfilter-hmc-post-m27-next-phase-2026-09-23.md)
retains supplied-map geometry, global exploration, full-fit null/power,
consumer/reference inputs and measured maintenance as separate open items.
